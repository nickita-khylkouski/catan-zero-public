#!/usr/bin/env python3
"""Export public-masked holdout evidence for the topology learning gate.

This R&D-only tool intentionally creates neither the validation split nor the
topology-sensitive labels. Validation rows are every corpus row whose
``game_seed`` belongs to an immutable ``train-validation-game-seeds-v1``
manifest. Sensitive rows are exactly the ``members`` subset emitted by
``rnd_build_topology_sensitive_mask.py``.

The run manifest is a JSON object containing the exact ``run_provenance``
fields consumed by ``rnd_topology_learning_gate.py`` plus ``arm`` and
``training_seed``.  Every file hash is checked before model inference.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Callable, Mapping

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
for _path in (_ROOT, _SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from catan_zero.rl.entity_token_features import mask_player_tokens_public  # noqa: E402
from catan_zero.rl.entity_token_policy import EntityGraphPolicy  # noqa: E402
from catan_zero.rl.optim_state import optimizer_sidecar_path  # noqa: E402
from catan_zero.rl.pipeline_configs import TrainConfig  # noqa: E402
from tools.train_bc import (  # noqa: E402
    ENTITY_BATCH_KEYS,
    MemmapCorpus,
    _training_data_fingerprint,
    _validate_memmap_payload_inventory,
)


MASK_SCHEMA = "catan-zero-topology-sensitive-mask/v1"
RUN_SCHEMA = "catan-zero-topology-run/v1"
VALIDATION_SCHEMA = "train-validation-game-seeds-v1"
_DYNAMIC_TRAIN_CONFIG_FIELDS = frozenset(
    {
        "seed",
        "data",
        "data_fingerprint",
        "init_checkpoint",
        "init_checkpoint_sha256",
        "validation_game_seed_manifest",
        "a1_memmap_payload_inventory_sha256",
        "rnd_a1_artifact_dir",
        "topology_adapter_layers",
        "topology_adapter_width",
        "topology_adapter_bases",
        "topology_adapter_kind",
        "topology_adapter_heads",
        "topology_adapter_share_weights",
        "topology_adapter_edge_control",
    }
)
_TRAINING_RECIPE_META_FIELDS = frozenset(
    {"fresh_optimizer", "information_regime"}
)
_EXECUTING_LEARNER_SOURCE_FILES = (
    "tools/train_bc.py",
    "src/catan_zero/rl/pipeline_configs.py",
    "src/catan_zero/rl/entity_token_policy.py",
    "src/catan_zero/rl/sparse_topology_adapter.py",
)


class ExportError(ValueError):
    """An input is not sufficiently bound to produce admissible evidence."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _a1_legacy_canonical_sha(value: Any) -> str:
    """Match the immutable A1 contract's historical ASCII-escaped digest."""

    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _validate_corpus_payloads(corpus_dir: Path) -> str:
    meta = _load_object(corpus_dir / "corpus_meta.json", name="corpus metadata")
    try:
        verified = _validate_memmap_payload_inventory(corpus_dir, meta)
    except SystemExit as exc:
        raise ExportError(f"memmap payload inventory validation failed: {exc}") from exc
    declared = meta.get("payload_inventory_sha256")
    if verified != declared:
        raise ExportError("verified payload inventory digest differs from corpus metadata")
    return verified


def _load_object(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExportError(f"cannot read {name} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExportError(f"{name} must contain a JSON object")
    return value


def _validation_seeds(payload: Mapping[str, Any]) -> np.ndarray:
    if payload.get("schema_version") != VALIDATION_SCHEMA:
        raise ExportError("unsupported validation manifest schema")
    raw = payload.get("game_seeds")
    if not isinstance(raw, list) or not raw or any(type(seed) is not int for seed in raw):
        raise ExportError("validation manifest game_seeds must be non-empty integers")
    seeds = np.asarray(raw, dtype="<i8")
    if np.any(seeds[1:] <= seeds[:-1]):
        raise ExportError("validation manifest game_seeds must be sorted and unique")
    if payload.get("validation_game_seed_count") != int(seeds.size):
        raise ExportError("validation manifest game count does not match game_seeds")
    digest = "sha256:" + hashlib.sha256(seeds.tobytes()).hexdigest()
    if payload.get("validation_game_seed_set_sha256") != digest:
        raise ExportError("validation manifest game-seed digest is invalid")
    return seeds


def _prefixed_sha(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or len(value) != 71
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ExportError(f"{field} must be a sha256:-prefixed lowercase digest")
    return value


def _seed_set_sha(seeds: list[int] | np.ndarray) -> str:
    values = np.sort(np.unique(np.asarray(seeds, dtype=np.int64))).astype("<i8")
    return "sha256:" + hashlib.sha256(values.tobytes()).hexdigest()


def _validate_training_manifest(
    payload: Mapping[str, Any],
    *,
    validation_payload: Mapping[str, Any],
    validation_seeds: np.ndarray,
) -> set[int]:
    expected_fields = {
        "schema_version",
        "a1_contract_sha256",
        "selection_rule",
        "selected_game_count",
        "selected_game_seed_set_sha256",
        "category_game_counts",
        "training_game_count",
        "training_game_seed_set_sha256",
        "validation_game_count",
        "validation_game_seed_set_sha256",
        "records",
        "records_sha256",
    }
    if set(payload) != expected_fields:
        raise ExportError("training manifest fields differ from a1-selected-training-games-v1")
    if payload.get("schema_version") != "a1-selected-training-games-v1":
        raise ExportError("unsupported training manifest schema")
    contract_sha = _prefixed_sha(
        payload.get("a1_contract_sha256"), field="training manifest a1_contract_sha256"
    )
    if validation_payload.get("a1_contract_sha256") != contract_sha:
        raise ExportError("training and validation manifests bind different A1 contracts")
    if payload.get("selection_rule") != "lowest_seed_complete_per_job":
        raise ExportError("training manifest selection_rule is not canonical")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ExportError("training manifest records must be a non-empty list")
    if payload.get("records_sha256") != "sha256:" + _canonical_sha(records):
        raise ExportError("training manifest records_sha256 mismatch")

    record_fields = {
        "game_seed",
        "job_id",
        "worker_id",
        "category",
        "producer_checkpoint_sha256",
        "opponent_checkpoint_sha256",
        "split",
    }
    all_seeds: list[int] = []
    train_seeds: list[int] = []
    manifest_validation: list[int] = []
    category_counts: dict[str, int] = {}
    previous_seed: int | None = None
    prior_splits: dict[int, str] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != record_fields:
            raise ExportError(f"training manifest record {index} fields are not canonical")
        seed = record.get("game_seed")
        split = record.get("split")
        if type(seed) is not int:
            raise ExportError(f"training manifest record {index} has invalid game_seed")
        if seed in prior_splits and prior_splits[seed] != split:
            raise ExportError("training and holdout game seeds overlap")
        if previous_seed is not None and seed <= previous_seed:
            raise ExportError("training manifest records must be strictly game-seed sorted")
        previous_seed = seed
        if split not in {"train", "validation"}:
            raise ExportError(f"training manifest record {index} has invalid split")
        prior_splits[seed] = split
        for field in ("job_id", "worker_id", "category"):
            if not isinstance(record[field], str) or not record[field]:
                raise ExportError(f"training manifest record {index} has invalid {field}")
        _prefixed_sha(
            record["producer_checkpoint_sha256"],
            field=f"training manifest record {index} producer checkpoint",
        )
        opponents = record["opponent_checkpoint_sha256"]
        if not isinstance(opponents, list) or not opponents:
            raise ExportError(f"training manifest record {index} has no opponents")
        for opponent in opponents:
            _prefixed_sha(opponent, field=f"training manifest record {index} opponent")
        all_seeds.append(seed)
        (train_seeds if split == "train" else manifest_validation).append(seed)
        category = record["category"]
        category_counts[category] = category_counts.get(category, 0) + 1

    expected_digests = {
        "selected_game_count": len(all_seeds),
        "selected_game_seed_set_sha256": _seed_set_sha(all_seeds),
        "training_game_count": len(train_seeds),
        "training_game_seed_set_sha256": _seed_set_sha(train_seeds),
        "validation_game_count": len(manifest_validation),
        "validation_game_seed_set_sha256": _seed_set_sha(manifest_validation),
    }
    for field, expected in expected_digests.items():
        if payload.get(field) != expected:
            raise ExportError(f"training manifest {field} mismatch")
    declared_categories = payload.get("category_game_counts")
    if not isinstance(declared_categories, dict) or declared_categories != dict(
        sorted(category_counts.items())
    ):
        raise ExportError("training manifest category_game_counts mismatch")
    if manifest_validation != validation_seeds.astype(np.int64).tolist():
        raise ExportError("training manifest validation records differ from holdout manifest")
    train_set = set(train_seeds)
    validation_set = set(manifest_validation)
    if train_set & validation_set:
        raise ExportError("training and holdout game seeds overlap")
    return train_set


def _validate_mask_artifact(
    mask: Mapping[str, Any],
    *,
    validation_manifest_sha: str,
    corpus_meta_sha: str,
    payload_inventory_sha: str,
) -> set[tuple[str, str]]:
    if mask.get("schema_version") != MASK_SCHEMA:
        raise ExportError("unsupported topology mask schema")
    without_artifact_sha = dict(mask)
    declared_artifact_sha = without_artifact_sha.pop("artifact_sha256", None)
    if declared_artifact_sha != "sha256:" + _canonical_sha(without_artifact_sha):
        raise ExportError("topology mask artifact_sha256 is invalid")
    config = mask.get("config")
    if (
        not isinstance(config, dict)
        or config.get("schema_version")
        != "catan-zero-topology-sensitive-mask-config/v1"
        or mask.get("config_sha256") != "sha256:" + _canonical_sha(config)
    ):
        raise ExportError("topology mask config binding is invalid")
    members = mask.get("members")
    if not isinstance(members, list):
        raise ExportError("topology mask members must be a list")
    if mask.get("members_sha256") != "sha256:" + _canonical_sha(members):
        raise ExportError("topology mask members_sha256 is invalid")
    summary = mask.get("summary")
    if not isinstance(summary, dict) or summary.get("decision_count") != len(members):
        raise ExportError("topology mask summary decision_count is invalid")
    source = mask.get("source")
    if not isinstance(source, dict) or mask.get("source_sha256") != "sha256:" + _canonical_sha(source):
        raise ExportError("topology mask source binding is invalid")
    validation_source = source.get("validation_manifest")
    corpus_source = source.get("corpus")
    if not isinstance(validation_source, dict) or not isinstance(corpus_source, dict):
        raise ExportError("topology mask source is incomplete")
    if validation_source.get("file_sha256") != "sha256:" + validation_manifest_sha:
        raise ExportError("topology mask does not bind this validation manifest")
    if corpus_source.get("corpus_meta_file_sha256") != "sha256:" + corpus_meta_sha:
        raise ExportError("topology mask does not bind this corpus metadata")
    if corpus_source.get("payload_inventory_sha256") != payload_inventory_sha:
        raise ExportError("topology mask does not bind this corpus payload inventory")

    identities: set[tuple[str, str]] = set()
    for position, member in enumerate(members):
        if not isinstance(member, dict):
            raise ExportError(f"topology mask member {position} must be an object")
        game_seed = member.get("game_seed")
        decision_index = member.get("decision_index")
        if type(game_seed) is not int or type(decision_index) is not int or decision_index < 0:
            raise ExportError(f"topology mask member {position} has invalid indices")
        expected_game = f"seed:{game_seed}"
        expected_decision = f"{expected_game}:decision:{decision_index}"
        if member.get("game_id") != expected_game or member.get("decision_id") != expected_decision:
            raise ExportError(f"topology mask member {position} has noncanonical identity")
        identity = (expected_game, expected_decision)
        if identity in identities:
            raise ExportError("topology mask contains duplicate decision identities")
        identities.add(identity)
    return identities


def _required_sha(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExportError(f"{field} must be a lowercase SHA256 hex digest")
    return value


def _positive_int(value: Any, *, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ExportError(f"{field} must be a positive integer")
    return value


def _arm_contract(config: Mapping[str, Any], arm: str) -> tuple[dict[str, Any], dict[str, Any]]:
    common = config.get("common")
    arms = config.get("arms")
    gate = config.get("learning_gate")
    if not isinstance(common, dict) or not isinstance(arms, list) or not isinstance(gate, dict):
        raise ExportError("experiment config must contain common, arms, and learning_gate")
    matches = [item for item in arms if isinstance(item, dict) and item.get("arm_id") == arm]
    if len(matches) != 1:
        raise ExportError(f"experiment config must contain exactly one arm {arm!r}")
    return matches[0], gate


def _validate_experiment_self_hash(config: Mapping[str, Any]) -> None:
    if config.get("config_sha256_scope") != "canonical_json_without_config_sha256":
        raise ExportError("experiment config_sha256_scope is missing or unsupported")
    payload = dict(config)
    declared = payload.pop("config_sha256", None)
    if declared != _canonical_sha(payload):
        raise ExportError("experiment config_sha256 self-hash is invalid")


def _resolved_expected(common: Mapping[str, Any], arm: Mapping[str, Any]) -> dict[str, Any]:
    ignored = {"arm_id", "role", "expected_parameters", "note"}
    return {**common, **{key: value for key, value in arm.items() if key not in ignored}}


def _validate_loaded_policy(policy: Any, resolved: Mapping[str, Any]) -> None:
    if not bool(getattr(policy, "trained_with_masked_hidden_info", False)):
        raise ExportError("checkpoint is not attested as public-masked training")
    config = getattr(policy, "config", None)
    if config is None:
        raise ExportError("loaded checkpoint has no entity-graph config")
    translations = {
        "hidden_size": "hidden_size",
        "state_layers": "state_layers",
        "attention_heads": "attention_heads",
        "adapter_layers": "topology_adapter_layers",
        "adapter_width": "topology_adapter_width",
        "adapter_bases": "topology_adapter_bases",
        "adapter_heads": "topology_adapter_heads",
        "share_weights": "topology_adapter_share_weights",
        "edge_control": "topology_adapter_edge_control",
    }
    for field, attribute in translations.items():
        if field in resolved and getattr(config, attribute, None) != resolved[field]:
            raise ExportError(f"checkpoint config {attribute} does not match run manifest")
    expected_kind = resolved.get("adapter_kind")
    if expected_kind == "none":
        if str(getattr(config, "topology_adapter_layers", "") or "").strip():
            raise ExportError("incumbent checkpoint unexpectedly enables topology adapters")
    elif expected_kind is not None and getattr(config, "topology_adapter_kind", None) != expected_kind:
        raise ExportError("checkpoint topology_adapter_kind does not match run manifest")


def _artifact_reference(run: Mapping[str, Any], name: str) -> tuple[Path, str]:
    value = run.get(name)
    if not isinstance(value, dict) or set(value) != {"path", "file_sha256"}:
        raise ExportError(f"run manifest {name} reference is malformed")
    path = Path(value["path"]).expanduser().resolve()
    expected = _required_sha(value["file_sha256"], field=f"run manifest {name}.file_sha256")
    if not path.is_file() or _sha256_file(path) != expected:
        raise ExportError(f"run manifest {name} file hash mismatch")
    return path, expected


def _resolve_report_repo_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ExportError(f"train_bc report {label} path must be a non-empty string")
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    if ".." in path.parts:
        raise ExportError(f"train_bc report {label} relative path contains traversal")
    resolved = (_ROOT / path).resolve()
    try:
        resolved.relative_to(_ROOT.resolve())
    except ValueError as exc:
        raise ExportError(
            f"train_bc report {label} relative path escapes repository root"
        ) from exc
    return resolved


def _validate_executing_learner_sources(
    report: Mapping[str, Any], experiment_config: Mapping[str, Any]
) -> None:
    frozen = experiment_config.get("frozen_inputs")
    registered = (
        frozen.get("executing_learner_source_sha256")
        if isinstance(frozen, dict)
        else None
    )
    reported = report.get("rnd_executing_learner_source_sha256")
    expected_keys = set(_EXECUTING_LEARNER_SOURCE_FILES)
    if not isinstance(registered, dict) or set(registered) != expected_keys:
        raise ExportError("experiment executing learner source map has incorrect keys")
    if not isinstance(reported, dict) or set(reported) != expected_keys:
        raise ExportError("train_bc report executing learner source map has incorrect keys")
    for relative in _EXECUTING_LEARNER_SOURCE_FILES:
        expected = _required_sha(
            registered[relative],
            field=f"frozen_inputs.executing_learner_source_sha256[{relative!r}]",
        )
        actual = _sha256_file(_ROOT / relative)
        if reported[relative] != expected:
            raise ExportError(f"train_bc report source SHA differs for {relative}")
        if actual != expected:
            raise ExportError(f"live executing learner source SHA differs for {relative}")


def _validate_a1_artifact_relocation(
    report: Mapping[str, Any],
    experiment_config: Mapping[str, Any],
    resolved_fields: Mapping[str, Any],
    *,
    corpus_dir: Path,
    training_manifest: Path,
    validation_manifest: Path,
) -> None:
    frozen = experiment_config.get("frozen_inputs")
    registered = (
        frozen.get("a1_artifact_relocation") if isinstance(frozen, dict) else None
    )
    reported = report.get("rnd_a1_artifact_relocation")
    if registered != reported or not isinstance(registered, dict):
        raise ExportError("train_bc report A1 artifact relocation differs from registration")
    if registered.get("schema_version") != "catan-zero-rnd-a1-artifact-relocation/v1":
        raise ExportError("A1 artifact relocation schema is unsupported")
    files = registered.get("files")
    roles = {
        "selected_game_manifest",
        "post_wave_audit",
        "validation_manifest",
        "contract_lock",
    }
    if not isinstance(files, dict) or set(files) != roles:
        raise ExportError("A1 artifact relocation must contain exactly four roles")
    directory_raw = resolved_fields.get("rnd_a1_artifact_dir")
    if not isinstance(directory_raw, str) or not directory_raw:
        raise ExportError("resolved TrainConfig has no rnd_a1_artifact_dir")
    directory = Path(directory_raw).expanduser().resolve()
    if not directory.is_dir():
        raise ExportError("resolved rnd_a1_artifact_dir is not a directory")
    filenames: set[str] = set()
    for role in sorted(roles):
        value = files[role]
        if not isinstance(value, dict) or set(value) != {
            "logical_path",
            "filename",
            "sha256",
        }:
            raise ExportError(f"A1 relocation role {role} is malformed")
        logical = Path(str(value["logical_path"])).expanduser()
        filename = value["filename"]
        if (
            not logical.is_absolute()
            or not isinstance(filename, str)
            or filename != logical.name
            or Path(filename).name != filename
            or filename in filenames
        ):
            raise ExportError(f"A1 relocation role {role} has invalid path/filename")
        filenames.add(filename)
        expected = _prefixed_sha(value["sha256"], field=f"A1 relocation {role} SHA")
        physical = directory / filename
        if not physical.is_file() or "sha256:" + _sha256_file(physical) != expected:
            raise ExportError(f"A1 relocation role {role} physical file hash mismatch")
    actual_files = {path.name for path in directory.iterdir() if path.is_file()}
    if actual_files != filenames:
        raise ExportError("A1 artifact relocation directory file set is not exact")

    corpus_meta = _load_object(corpus_dir / "corpus_meta.json", name="corpus metadata")
    selected_meta = corpus_meta.get("selected_game_seed_manifest")
    audit_meta = corpus_meta.get("a1_post_wave_audit")
    if not isinstance(selected_meta, dict) or not isinstance(audit_meta, dict):
        raise ExportError("corpus metadata lacks A1 selected/audit bindings")
    validation_meta = audit_meta.get("validation_holdout")
    if not isinstance(validation_meta, dict):
        raise ExportError("corpus metadata lacks A1 validation-artifact binding")
    expected_role_hashes = {
        "selected_game_manifest": selected_meta.get("file_sha256"),
        "post_wave_audit": audit_meta.get("file_sha256"),
        "validation_manifest": validation_meta.get("file_sha256"),
    }
    expected_logical_paths = {
        "selected_game_manifest": selected_meta.get("path"),
        "post_wave_audit": audit_meta.get("path"),
        "validation_manifest": validation_meta.get("path"),
    }
    for role, expected_hash in expected_role_hashes.items():
        if files[role]["sha256"] != expected_hash:
            raise ExportError(f"A1 relocation role {role} differs from authenticated input")
    for role, expected_path in expected_logical_paths.items():
        if Path(files[role]["logical_path"]).resolve() != Path(str(expected_path)).resolve():
            raise ExportError(f"A1 relocation role {role} logical path differs from corpus binding")
    if files["selected_game_manifest"]["sha256"] != "sha256:" + _sha256_file(
        training_manifest
    ):
        raise ExportError("relocated selected-game manifest differs from training input")
    if files["validation_manifest"]["sha256"] != "sha256:" + _sha256_file(
        validation_manifest
    ):
        raise ExportError("relocated validation manifest differs from validation input")

    audit_path = directory / files["post_wave_audit"]["filename"]
    audit = _load_object(audit_path, name="relocated post-wave audit")
    contract_path = audit.get("contract_path")
    if Path(files["contract_lock"]["logical_path"]).resolve() != Path(
        str(contract_path)
    ).resolve():
        raise ExportError("relocated contract logical path differs from audit binding")
    contract = _load_object(
        directory / files["contract_lock"]["filename"], name="relocated contract"
    )
    contract_sha = audit.get("contract_sha256")
    semantic = dict(contract)
    semantic.pop("contract_sha256", None)
    if (
        contract.get("contract_sha256") != contract_sha
        or contract_sha != "sha256:" + _a1_legacy_canonical_sha(semantic)
    ):
        raise ExportError("relocated contract semantic hash differs from audit binding")


def _validate_training_report(
    run: Mapping[str, Any],
    *,
    checkpoint: Path,
    corpus_dir: Path,
    validation_manifest: Path,
    training_manifest: Path,
    payload_inventory_sha: str,
    experiment_config: Mapping[str, Any],
    experiment_config_path: Path,
    experiment_config_sha: str,
    expected_resolved: Mapping[str, Any],
    arm: Mapping[str, Any],
    training_seed: int,
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    report_path, report_sha = _artifact_reference(run, "training_report")
    experiment_ref_path, experiment_ref_sha = _artifact_reference(run, "experiment_config")
    if experiment_ref_path != experiment_config_path.resolve() or experiment_ref_sha != experiment_config_sha:
        raise ExportError("run manifest does not bind the evaluated experiment config")
    report = _load_object(report_path, name="train_bc report")
    _validate_executing_learner_sources(report, experiment_config)
    checkpoint_ref, checkpoint_sha = _artifact_reference(run, "checkpoint")
    if checkpoint_ref != checkpoint.expanduser().resolve():
        raise ExportError("run manifest checkpoint path differs from evaluated checkpoint")
    report_checkpoint = _resolve_report_repo_path(
        report.get("checkpoint"), label="checkpoint"
    )
    if report_checkpoint != checkpoint_ref:
        raise ExportError("train_bc report checkpoint path differs from evaluated checkpoint")
    sidecar_path, sidecar_sha = _artifact_reference(run, "optimizer_sidecar")
    if sidecar_path != optimizer_sidecar_path(checkpoint_ref).resolve():
        raise ExportError("optimizer sidecar is not the checkpoint's canonical sidecar")

    resolved_payload = report.get("resolved_train_config")
    if (
        not isinstance(resolved_payload, dict)
        or resolved_payload.get("pipeline") != "train"
        or not isinstance(resolved_payload.get("fields"), dict)
    ):
        raise ExportError("train_bc report has no canonical resolved TrainConfig payload")
    resolved_fields = resolved_payload["fields"]
    expected_field_names = {field.name for field in dataclasses.fields(TrainConfig)}
    if set(resolved_fields) != expected_field_names:
        raise ExportError("train_bc report resolved TrainConfig fields are incomplete")
    try:
        train_config = TrainConfig(**resolved_fields)
    except (TypeError, ValueError) as exc:
        raise ExportError(f"invalid resolved TrainConfig: {exc}") from exc
    if report.get("config_hash") != train_config.config_hash():
        raise ExportError("train_bc report config_hash differs from resolved TrainConfig")
    if report.get("full_config_hash") != train_config.full_config_hash():
        raise ExportError("train_bc report full_config_hash differs from resolved TrainConfig")
    if resolved_payload != train_config.canonical_payload():
        raise ExportError("train_bc report resolved TrainConfig payload is noncanonical")
    resolved_arch = resolved_fields.get("arch")
    report_graph_tokens = report.get("graph_tokens")
    if resolved_arch == "entity_graph":
        if report_graph_tokens is not None:
            raise ExportError(
                "entity_graph train_bc report graph_tokens telemetry must be null"
            )
    elif resolved_arch == "xdim_graph":
        if report_graph_tokens != resolved_fields["graph_tokens"]:
            raise ExportError(
                "xdim_graph train_bc report graph_tokens differs from resolved TrainConfig"
            )
    _validate_a1_artifact_relocation(
        report,
        experiment_config,
        resolved_fields,
        corpus_dir=corpus_dir,
        training_manifest=training_manifest,
        validation_manifest=validation_manifest,
    )
    for field, value in resolved_fields.items():
        if field == "graph_tokens":
            # This top-level field is conditional telemetry: entity_graph writes
            # null while the canonical TrainConfig retains its unused default.
            continue
        if field in report and report[field] != value:
            raise ExportError(f"train_bc report field {field} differs from resolved TrainConfig")

    report_architecture = {
        "hidden_size": report.get("hidden_size"),
        "state_layers": report.get("graph_layers"),
        "attention_heads": report.get("attention_heads"),
        "adapter_layers": report.get("topology_adapter_layers"),
        "adapter_width": report.get("topology_adapter_width"),
        "adapter_bases": report.get("topology_adapter_bases"),
        "adapter_kind": "none"
        if not str(report.get("topology_adapter_layers", "") or "").strip()
        else report.get("topology_adapter_kind"),
        "adapter_heads": report.get("topology_adapter_heads"),
        "share_weights": report.get("topology_adapter_share_weights"),
        "edge_control": report.get("topology_adapter_edge_control"),
    }
    for field, expected in expected_resolved.items():
        if field in {"warm_start_identity_required", "information_regime"}:
            continue
        if report_architecture.get(field) != expected:
            raise ExportError(f"train_bc report architecture field {field} differs from experiment")
    if report.get("parameter_count") != arm.get("expected_parameters"):
        raise ExportError("train_bc report parameter_count differs from experiment")
    if report.get("seed") != training_seed or report.get("mask_hidden_info") is not True:
        raise ExportError("train_bc report seed/public-mask provenance mismatch")
    if report.get("resume_optimizer") is not False or report.get("optimizer_restored") is not False:
        raise ExportError("learning-gate run must use a fresh optimizer")

    registered_recipe = experiment_config.get("training_recipe")
    if not isinstance(registered_recipe, dict) or not registered_recipe:
        raise ExportError("experiment config must register a non-empty training_recipe")
    required_recipe_fields = expected_field_names - _DYNAMIC_TRAIN_CONFIG_FIELDS
    expected_recipe_keys = required_recipe_fields | _TRAINING_RECIPE_META_FIELDS
    if set(registered_recipe) != expected_recipe_keys:
        raise ExportError(
            "experiment training_recipe fields are incomplete or unexpected: "
            f"missing={sorted(expected_recipe_keys - set(registered_recipe))} "
            f"extra={sorted(set(registered_recipe) - expected_recipe_keys)}"
        )
    for field, expected in registered_recipe.items():
        if field in {"fresh_optimizer", "information_regime"}:
            continue
        if field not in resolved_fields or resolved_fields[field] != expected:
            raise ExportError(f"resolved TrainConfig field {field} differs from registered recipe")
    if registered_recipe.get("fresh_optimizer") is not True:
        raise ExportError("experiment training_recipe must require fresh_optimizer")
    if registered_recipe.get("information_regime") != "public_only":
        raise ExportError("experiment training_recipe must require public_only information")
    frozen_inputs = experiment_config.get("frozen_inputs")
    warm_starts = (
        frozen_inputs.get("warm_start_checkpoint_sha256_by_arm")
        if isinstance(frozen_inputs, dict)
        else None
    )
    expected_init_sha = (
        warm_starts.get(arm.get("arm_id")) if isinstance(warm_starts, dict) else None
    )
    expected_init_bare = _required_sha(
        expected_init_sha, field="frozen_inputs warm-start checkpoint SHA"
    )
    if "sha256:" + expected_init_bare != report.get("init_checkpoint_sha256"):
        raise ExportError("train_bc report init checkpoint differs from registered arm")

    batch = _positive_int(report.get("batch_size"), field="report.batch_size")
    accum = _positive_int(report.get("grad_accum_steps"), field="report.grad_accum_steps")
    world = _positive_int(report.get("world_size"), field="report.world_size")
    global_batch = batch * accum * world
    if report.get("global_batch_size") != global_batch:
        raise ExportError("train_bc report global batch derivation mismatch")
    steps = _positive_int(report.get("steps_completed"), field="report.steps_completed")
    presentations = steps * global_batch
    if report.get("sample_presentations") != presentations or steps != gate.get("optimizer_steps") or global_batch != gate.get("global_batch_size") or presentations != gate.get("sample_presentations_per_arm_seed"):
        raise ExportError("train_bc report budget differs from experiment gate")

    actual_fingerprint = _training_data_fingerprint(str(corpus_dir), "memmap")
    if report.get("data_format") != "memmap" or report.get("data_fingerprint") != actual_fingerprint:
        raise ExportError("train_bc report data fingerprint differs from authenticated corpus")
    if report.get("a1_memmap_payload_inventory_sha256") != payload_inventory_sha:
        raise ExportError("train_bc report payload inventory digest mismatch")
    validation_sha = "sha256:" + _sha256_file(validation_manifest)
    if report.get("input_validation_game_seed_manifest_sha256") != validation_sha:
        raise ExportError("train_bc report validation-manifest digest mismatch")
    validation_payload = _load_object(validation_manifest, name="validation manifest")
    if report.get("validation_game_seed_set_sha256") != validation_payload.get("validation_game_seed_set_sha256"):
        raise ExportError("train_bc report heldout seed digest mismatch")
    if report.get("checkpoint_sha256") != "sha256:" + checkpoint_sha:
        raise ExportError("train_bc report checkpoint SHA differs from evaluated checkpoint")
    if report.get("optimizer_sidecar_sha256") != "sha256:" + sidecar_sha:
        raise ExportError("train_bc report optimizer-sidecar SHA mismatch")
    report_sidecar = _resolve_report_repo_path(
        report.get("optimizer_sidecar"), label="optimizer_sidecar"
    )
    if report_sidecar.resolve() != sidecar_path:
        raise ExportError("train_bc report optimizer-sidecar path mismatch")

    return {
        "checkpoint_sha256": checkpoint_sha,
        "resolved_config": dict(expected_resolved),
        "resolved_config_sha256": _canonical_sha(expected_resolved),
        "parameter_count": report["parameter_count"],
        "training_data_sha256": payload_inventory_sha.removeprefix("sha256:"),
        "optimizer_steps": steps,
        "global_batch_size": global_batch,
        "sample_presentations": presentations,
        "training_report_sha256": report_sha,
        "experiment_config_sha256": experiment_config_sha,
        "optimizer_sidecar_sha256": sidecar_sha,
        "train_config_hash": report["config_hash"],
    }


def _validate_artifacts(
    *,
    corpus: Any,
    corpus_dir: Path,
    checkpoint: Path,
    training_manifest: Path,
    holdout_manifest_path: Path,
    mask_path: Path,
    run_manifest_path: Path,
    experiment_config: Mapping[str, Any],
    experiment_config_path: Path,
    experiment_config_sha: str,
    arm: str,
    training_seed: int,
    payload_inventory_sha: str,
) -> tuple[list[dict[str, Any]], set[tuple[str, str]], dict[str, Any], dict[str, str]]:
    arm_config, gate = _arm_contract(experiment_config, arm)
    corpus_meta = getattr(corpus, "meta", None)
    if (
        not isinstance(corpus_meta, Mapping)
        or corpus_meta.get("payload_inventory_sha256") != payload_inventory_sha
    ):
        raise ExportError("loaded corpus metadata differs from verified payload inventory")
    expected_target_regime = experiment_config.get("common", {}).get(
        "information_regime"
    )
    if "target_information_regime" not in corpus:
        raise ExportError("corpus does not expose target_information_regime")
    actual_target_regimes = set(
        np.asarray(corpus["target_information_regime"]).astype(str).tolist()
    )
    if actual_target_regimes != {expected_target_regime}:
        raise ExportError("authenticated corpus target information regime mismatch")
    expected_seeds = gate.get("seeds")
    if not isinstance(expected_seeds, list) or training_seed not in expected_seeds:
        raise ExportError(f"training seed {training_seed} is not registered")

    hashes = {
        "checkpoint_sha256": _sha256_file(checkpoint),
        "training_manifest_sha256": _sha256_file(training_manifest),
        "holdout_manifest_sha256": _sha256_file(holdout_manifest_path),
        "topology_mask_registration_artifact_sha256": _sha256_file(mask_path),
        "corpus_meta_sha256": _sha256_file(corpus_dir / "corpus_meta.json"),
    }
    for field in (
        "training_manifest_sha256",
        "holdout_manifest_sha256",
        "topology_mask_registration_artifact_sha256",
    ):
        expected = _required_sha(gate.get(field), field=f"learning_gate.{field}")
        if hashes[field] != expected:
            raise ExportError(f"{field} does not match the experiment config")

    validation = _load_object(holdout_manifest_path, name="validation manifest")
    validation_seeds = _validation_seeds(validation)
    training = _load_object(training_manifest, name="training manifest")
    _validate_training_manifest(
        training,
        validation_payload=validation,
        validation_seeds=validation_seeds,
    )
    if "game_seed" not in corpus or "decision_index" not in corpus:
        raise ExportError("corpus must expose game_seed and decision_index")
    corpus_seeds = np.asarray(corpus["game_seed"], dtype=np.int64)
    selected_indices = np.flatnonzero(np.isin(corpus_seeds, validation_seeds))
    observed = np.unique(corpus_seeds[selected_indices])
    if not np.array_equal(observed, validation_seeds):
        raise ExportError("one or more validation games are missing from the corpus")
    decisions = np.asarray(corpus["decision_index"], dtype=np.int64)
    normalized_rows: list[dict[str, Any]] = []
    seen_ids: set[tuple[str, str]] = set()
    for index in selected_indices:
        seed = int(corpus_seeds[index])
        decision = int(decisions[index])
        if decision < 0:
            raise ExportError("validation row has no decision_index")
        game_id = f"seed:{seed}"
        decision_id = f"{game_id}:decision:{decision}"
        identity = (game_id, decision_id)
        if identity in seen_ids:
            raise ExportError(f"duplicate validation decision identity {decision_id}")
        seen_ids.add(identity)
        normalized_rows.append(
            {"row_index": int(index), "game_id": game_id, "decision_id": decision_id}
        )
    normalized_rows.sort(
        key=lambda row: (
            int(corpus_seeds[row["row_index"]]),
            int(decisions[row["row_index"]]),
        )
    )
    if not normalized_rows:
        raise ExportError("validation manifest selects no corpus decisions")

    mask = _load_object(mask_path, name="topology mask")
    sensitive_ids = _validate_mask_artifact(
        mask,
        validation_manifest_sha=hashes["holdout_manifest_sha256"],
        corpus_meta_sha=hashes["corpus_meta_sha256"],
        payload_inventory_sha=payload_inventory_sha,
    )
    if not sensitive_ids <= seen_ids:
        raise ExportError("topology mask contains decisions outside validation support")
    if mask["source"]["corpus"].get("row_count") != len(corpus):
        raise ExportError("topology mask corpus row_count does not match corpus")
    for member in mask["members"]:
        source_index = member.get("source_row_index")
        if type(source_index) is not int or not 0 <= source_index < len(corpus):
            raise ExportError("topology mask member has invalid source_row_index")
        seed = int(corpus_seeds[source_index])
        decision = int(decisions[source_index])
        expected = (f"seed:{seed}", f"seed:{seed}:decision:{decision}")
        if (member["game_id"], member["decision_id"]) != expected:
            raise ExportError("topology mask member identity differs from its corpus row")

    run = _load_object(run_manifest_path, name="run manifest")
    if run.get("schema_version") != RUN_SCHEMA or run.get("arm") != arm or run.get("training_seed") != training_seed:
        raise ExportError("run manifest schema/arm/training_seed mismatch")
    if run.get("training_manifest_sha256") != hashes["training_manifest_sha256"]:
        raise ExportError("run manifest does not bind this training manifest")
    expected_resolved = _resolved_expected(experiment_config["common"], arm_config)
    provenance = _validate_training_report(
        run,
        checkpoint=checkpoint,
        corpus_dir=corpus_dir,
        training_manifest=training_manifest,
        validation_manifest=holdout_manifest_path,
        payload_inventory_sha=payload_inventory_sha,
        experiment_config_path=experiment_config_path,
        experiment_config_sha=experiment_config_sha,
        experiment_config=experiment_config,
        expected_resolved=expected_resolved,
        arm=arm_config,
        training_seed=training_seed,
        gate=gate,
    )
    if provenance["training_data_sha256"] != _required_sha(
        gate.get("training_data_sha256"), field="learning_gate.training_data_sha256"
    ):
        raise ExportError("authenticated training data differs from experiment config")
    return normalized_rows, sensitive_ids, provenance, hashes


def export_holdout_evidence(
    *,
    checkpoint: Path,
    corpus_dir: Path,
    training_manifest: Path,
    validation_manifest: Path,
    topology_mask: Path,
    run_manifest: Path,
    experiment_config_path: Path,
    arm: str,
    training_seed: int,
    output: Path,
    batch_size: int = 256,
    device: str = "cpu",
    corpus_loader: Callable[[Path], Any] = MemmapCorpus,
    policy_loader: Callable[..., Any] = EntityGraphPolicy.load,
) -> int:
    """Validate all bindings, evaluate the frozen rows, and atomically write JSONL."""

    if output.exists():
        raise ExportError(f"refusing to overwrite {output}")
    _positive_int(batch_size, field="batch_size")
    experiment = _load_object(experiment_config_path, name="experiment config")
    _validate_experiment_self_hash(experiment)
    experiment_config_sha = _sha256_file(experiment_config_path)
    # Authenticate every flat file before opening any memmap view over it.
    payload_inventory_sha = _validate_corpus_payloads(corpus_dir)
    corpus = corpus_loader(corpus_dir)
    rows, sensitive, provenance, hashes = _validate_artifacts(
        corpus=corpus,
        corpus_dir=corpus_dir,
        checkpoint=checkpoint,
        training_manifest=training_manifest,
        holdout_manifest_path=validation_manifest,
        mask_path=topology_mask,
        run_manifest_path=run_manifest,
        experiment_config=experiment,
        experiment_config_path=experiment_config_path,
        experiment_config_sha=experiment_config_sha,
        arm=arm,
        training_seed=training_seed,
        payload_inventory_sha=payload_inventory_sha,
    )
    policy = policy_loader(checkpoint, device=device, strict_metadata=True)
    _validate_loaded_policy(policy, provenance["resolved_config"])
    actual_parameters = sum(parameter.numel() for parameter in policy.model.parameters())
    if actual_parameters != provenance["parameter_count"]:
        raise ExportError("loaded checkpoint parameter count does not match run provenance")

    import torch

    evidence: list[dict[str, Any]] = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            selected = rows[start : start + batch_size]
            indices = np.asarray([row["row_index"] for row in selected], dtype=np.int64)
            missing = [key for key in ENTITY_BATCH_KEYS if key not in corpus]
            if missing:
                raise ExportError(f"memmap corpus is missing entity columns: {missing}")
            entity = {key: corpus[key][indices] for key in ENTITY_BATCH_KEYS}
            entity["player_tokens"] = mask_player_tokens_public(entity["player_tokens"])
            legal_ids = np.asarray(corpus["legal_action_ids"][indices])
            contexts = np.asarray(corpus["legal_action_context"][indices])
            actions = np.asarray(corpus["action_taken"][indices], dtype=np.int64)
            valid = np.asarray(entity["legal_action_mask"], dtype=np.bool_)
            if valid.shape != legal_ids.shape or not np.array_equal(valid, legal_ids >= 0):
                raise ExportError("legal_action_mask must exactly match live legal_action_ids")
            if "target_policy" not in corpus or "target_policy_mask" not in corpus:
                raise ExportError("corpus must contain target_policy and target_policy_mask")
            target_policy = np.asarray(corpus["target_policy"][indices], dtype=np.float64)
            target_mask = np.asarray(corpus["target_policy_mask"][indices], dtype=np.bool_)
            if target_policy.shape != valid.shape or target_mask.shape != valid.shape:
                raise ExportError("target policy tensors must align with legal actions")
            if not np.array_equal(target_mask, valid):
                raise ExportError("target_policy_mask must provide full live-action coverage")
            if np.any(~np.isfinite(target_policy)) or np.any(target_policy < 0):
                raise ExportError("target_policy must be finite and non-negative")
            if np.any(target_policy[~valid] != 0):
                raise ExportError("target_policy padding must be zero")
            target_sums = np.sum(np.where(valid, target_policy, 0.0), axis=1)
            if not np.allclose(target_sums, 1.0, rtol=0.0, atol=1.0e-5):
                raise ExportError("target_policy must normalize to one over live actions")
            target_matches = legal_ids == actions[:, None]
            if np.any(target_matches.sum(axis=1) != 1):
                raise ExportError("every holdout action must occur exactly once in legal_action_ids")
            outputs = policy.forward_legal_np(entity, legal_ids, contexts, return_q=False)
            logits = outputs["logits"].float().masked_fill(
                ~torch.as_tensor(valid, device=outputs["logits"].device), float("-inf")
            )
            targets = torch.as_tensor(
                np.argmax(target_matches, axis=1), dtype=torch.long, device=logits.device
            )
            log_probs = torch.log_softmax(logits, dim=-1)
            target_tensor = torch.as_tensor(
                target_policy, dtype=log_probs.dtype, device=log_probs.device
            )
            valid_tensor = torch.as_tensor(valid, device=log_probs.device)
            safe_log_probs = torch.where(valid_tensor, log_probs, 0.0)
            soft_ce = -(target_tensor * safe_log_probs).sum(dim=-1)
            hard_ce = torch.nn.functional.cross_entropy(
                logits, targets, reduction="none"
            )
            for row, loss, hard_loss, legal_count in zip(
                selected,
                soft_ce.cpu().tolist(),
                hard_ce.cpu().tolist(),
                valid.sum(axis=1),
                strict=True,
            ):
                if not math.isfinite(loss) or loss < 0:
                    raise ExportError("model produced invalid decision CE")
                identity = (row["game_id"], row["decision_id"])
                evidence.append(
                    {
                        "arm": arm,
                        "training_seed": training_seed,
                        "game_id": identity[0],
                        "decision_id": identity[1],
                        "policy_ce": float(loss),
                        "hard_action_ce": float(hard_loss),
                        "forced": bool(legal_count <= 1),
                        "topology_sensitive": identity in sensitive,
                        "evaluation_split": "holdout",
                        "is_training_game": False,
                        "topology_mask_registration_artifact_sha256": hashes["topology_mask_registration_artifact_sha256"],
                        "training_manifest_sha256": hashes["training_manifest_sha256"],
                        "holdout_manifest_sha256": hashes["holdout_manifest_sha256"],
                        "experiment_config_sha256": provenance[
                            "experiment_config_sha256"
                        ],
                        "run_provenance": provenance,
                    }
                )

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for record in evidence:
                stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return len(evidence)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--topology-mask", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        count = export_holdout_evidence(
            checkpoint=args.checkpoint,
            corpus_dir=args.corpus,
            training_manifest=args.training_manifest,
            validation_manifest=args.validation_manifest,
            topology_mask=args.topology_mask,
            run_manifest=args.run_manifest,
            experiment_config_path=args.experiment_config,
            arm=args.arm,
            training_seed=args.training_seed,
            output=args.output,
            batch_size=args.batch_size,
            device=args.device,
        )
    except (ExportError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"topology holdout export failed: {exc}") from exc
    print(json.dumps({"output": str(args.output), "rows": count}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
