#!/usr/bin/env python3
"""Export authenticated public-masked holdout evidence for the E3 A1 screen.

The exporter consumes the immutable E3 registration and one admission manifest.
It reauthenticates the completed ``train_bc`` report, checkpoint, optimizer
sidecar, executing sources, A1 corpus, selected-game manifest, and validation
manifest before inference.  The output is an atomically published JSONL file
covering exactly the registered 596 games and 146,517 decisions.
"""

from __future__ import annotations

import argparse
import dataclasses
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

from catan_zero.rl.entity_token_features import (  # noqa: E402
    PLAYER_ACTOR_FLAG_SLOT,
    PUBLIC_MASK_PLAYER_SLOTS,
    mask_player_tokens_public,
)
from catan_zero.rl.entity_token_policy import EntityGraphPolicy  # noqa: E402
from catan_zero.rl.optim_state import optimizer_sidecar_path  # noqa: E402
from catan_zero.rl.pipeline_configs import TrainConfig  # noqa: E402
from tools.rnd_e3_a1_admission import (  # noqa: E402
    ADMISSION_SCHEMA,
    ARMS,
    SOURCE_FILES,
    _corpus_fingerprint,
    _validate_contract,
)
from tools.rnd_topology_holdout_export import (  # noqa: E402
    ENTITY_BATCH_KEYS,
    _canonical_sha,
    _load_object,
    _positive_int,
    _sha256_file,
    _validate_corpus_payloads,
    _validate_training_manifest,
    _validation_seeds,
)
from tools.train_bc import MemmapCorpus, _training_data_fingerprint  # noqa: E402


EVIDENCE_SCHEMA = "catan-zero-e3-holdout-evidence/v1"
EXPORT_CONTRACT_SCHEMA = "catan-zero-e3-evidence-export/v1"
EXPECTED_HOLDOUT_GAMES = 596
EXPECTED_HOLDOUT_ROWS = 146_517
EXPECTED_CORPUS_ROWS = 2_927_924
_REPORT_SOURCE_FILES = (
    "tools/train_bc.py",
    "src/catan_zero/rl/pipeline_configs.py",
    "src/catan_zero/rl/entity_token_policy.py",
    "src/catan_zero/rl/sparse_topology_adapter.py",
)


class ExportError(ValueError):
    """An input cannot produce admissible E3 evidence."""


def _validate_export_contract(
    contract: Mapping[str, Any],
    *,
    contract_path: Path,
    experiment: Mapping[str, Any],
    experiment_path: Path,
) -> dict[str, str]:
    if contract.get("schema_version") != EXPORT_CONTRACT_SCHEMA:
        raise ExportError("unsupported E3 evidence-export contract schema")
    semantic = dict(contract)
    declared = semantic.pop("config_sha256", None)
    if declared != _canonical_sha(semantic):
        raise ExportError("evidence-export contract self-hash is invalid")
    if contract.get("experiment_file_sha256") != _sha256_file(
        experiment_path
    ) or contract.get("experiment_semantic_sha256") != experiment.get("config_sha256"):
        raise ExportError("evidence-export contract does not bind this registration")
    required = {
        "evidence_schema": EVIDENCE_SCHEMA,
        "information_regime": experiment["common"]["information_regime"],
        "public_masking_required": True,
        "holdout_games": EXPECTED_HOLDOUT_GAMES,
        "holdout_rows": EXPECTED_HOLDOUT_ROWS,
    }
    for field, expected in required.items():
        if contract.get(field) != expected:
            raise ExportError(f"evidence-export contract {field} drifted")
    source_paths = {
        "exporter_source_sha256": Path(__file__).resolve(),
        "exporter_helper_source_sha256": (
            _ROOT / "tools/rnd_topology_holdout_export.py"
        ).resolve(),
    }
    result: dict[str, str] = {
        "evidence_export_contract_sha256": _sha256_file(contract_path),
        "evidence_export_contract_semantic_sha256": declared,
    }
    for field, path in source_paths.items():
        expected = _required_sha(contract.get(field), field=f"contract.{field}")
        if _sha256_file(path) != expected:
            raise ExportError(f"evidence-export contract source drift for {path.name}")
        result[field] = expected
    return result


def _required_sha(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExportError(f"{field} must be a lowercase SHA256 digest")
    return value


def _resolve_path(value: Any, *, field: str, repo_root: Path = _ROOT) -> Path:
    if not isinstance(value, str) or not value:
        raise ExportError(f"{field} must be a non-empty path")
    raw = Path(value).expanduser()
    if not raw.is_absolute() and ".." in raw.parts:
        raise ExportError(f"{field} contains relative traversal")
    resolved = raw.resolve() if raw.is_absolute() else (repo_root / raw).resolve()
    if not resolved.is_file():
        raise ExportError(f"{field} is not a file: {resolved}")
    return resolved


def _flag_value(argv: Any, flag: str) -> str:
    if not isinstance(argv, list) or any(not isinstance(item, str) for item in argv):
        raise ExportError("admission train_argv must be a string list")
    positions = [index for index, value in enumerate(argv) if value == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise ExportError(f"admission train_argv must contain exactly one {flag}")
    return argv[positions[0] + 1]


def _arm_config(experiment: Mapping[str, Any], arm_id: str) -> dict[str, Any]:
    arms = experiment.get("arms")
    matches = [
        item
        for item in arms or []
        if isinstance(item, dict) and item.get("arm_id") == arm_id
    ]
    if len(matches) != 1:
        raise ExportError(f"registration must contain exactly one arm {arm_id!r}")
    return matches[0]


def _validate_admission(
    admission: Mapping[str, Any],
    *,
    admission_path: Path,
    experiment: Mapping[str, Any],
    experiment_path: Path,
    checkpoint: Path,
    report: Path,
    corpus_dir: Path,
    validation_manifest: Path,
) -> tuple[str, int, dict[str, Any]]:
    if admission.get("schema_version") != ADMISSION_SCHEMA:
        raise ExportError("unsupported E3 admission schema")
    if admission.get("experiment_config_sha256") != _sha256_file(experiment_path):
        raise ExportError("admission does not bind the registered experiment bytes")
    if admission.get("experiment_semantic_sha256") != experiment.get("config_sha256"):
        raise ExportError("admission does not bind the registered experiment semantics")
    if admission.get("registered_hashes") != experiment.get("registration"):
        raise ExportError("admission registered hashes differ from the experiment")

    arm_id = admission.get("arm_id")
    seed = admission.get("training_seed")
    if arm_id not in ARMS or type(seed) is not int:
        raise ExportError("admission arm/training seed is invalid")
    matrix = experiment["run_matrix"]
    if seed not in matrix["seeds"]:
        raise ExportError("admission training seed is not registered")
    steps, parameters, capacity = ARMS[arm_id]
    arm = _arm_config(experiment, arm_id)
    if (
        admission.get("latent_deliberation_steps") != steps
        or admission.get("expected_parameters") != parameters
        or admission.get("capacity_class") != capacity
        or arm.get("latent_deliberation_steps") != steps
        or arm.get("expected_parameters") != parameters
        or arm.get("capacity_class") != capacity
    ):
        raise ExportError("admission architecture differs from the registered arm")

    expected_run = (
        _ROOT
        / matrix["run_directory_pattern"].format(arm_id=arm_id, training_seed=seed)
    ).resolve()
    expected_admission = expected_run / matrix["admission_manifest_filename"]
    if admission_path.resolve() != expected_admission:
        raise ExportError("admission manifest is outside its registered run directory")
    expected_checkpoint = expected_run / matrix["checkpoint_filename"]
    expected_report = expected_run / matrix["report_filename"]
    if (
        Path(str(admission.get("run_directory"))).resolve() != expected_run
        or Path(str(admission.get("checkpoint"))).resolve() != expected_checkpoint
        or Path(str(admission.get("report"))).resolve() != expected_report
        or checkpoint.resolve() != expected_checkpoint
        or report.resolve() != expected_report
    ):
        raise ExportError(
            "admission/report/checkpoint path differs from registered layout"
        )

    argv = admission.get("train_argv")
    bindings = {
        "--data": corpus_dir.resolve(),
        "--validation-game-seed-manifest": validation_manifest.resolve(),
        "--checkpoint": checkpoint.resolve(),
        "--report": report.resolve(),
    }
    for flag, expected in bindings.items():
        if Path(_flag_value(argv, flag)).expanduser().resolve() != expected:
            raise ExportError(f"admission train_argv {flag} binding differs")
    if (
        int(_flag_value(argv, "--seed")) != seed
        or int(_flag_value(argv, "--latent-deliberation-steps")) != steps
    ):
        raise ExportError("admission train_argv arm/seed differs")
    init_path = _resolve_path(
        _flag_value(argv, "--init-checkpoint"), field="admission initial checkpoint"
    )
    init_sha = _sha256_file(init_path)
    registered_init = experiment["registration"][
        "initial_checkpoint_sha256_by_arm_seed"
    ].get(f"{arm_id}@{seed}")
    if init_sha != registered_init:
        raise ExportError("admission initial checkpoint differs from registration")
    return arm_id, seed, arm


def _validate_sources(report: Mapping[str, Any], experiment: Mapping[str, Any]) -> None:
    registered = experiment["registration"].get("executing_learner_source_sha256")
    if not isinstance(registered, dict) or set(registered) != set(SOURCE_FILES):
        raise ExportError("registration executing-source set is incomplete")
    for relative in SOURCE_FILES:
        expected = _required_sha(registered[relative], field=f"source {relative}")
        if _sha256_file(_ROOT / relative) != expected:
            raise ExportError(f"live executing source differs for {relative}")
    reported = report.get("rnd_executing_learner_source_sha256")
    if not isinstance(reported, dict) or set(reported) != set(_REPORT_SOURCE_FILES):
        raise ExportError("training report executing-source set is incomplete")
    for relative in _REPORT_SOURCE_FILES:
        if reported[relative] != registered[relative]:
            raise ExportError(f"training report source differs for {relative}")


def _validate_relocation(
    report: Mapping[str, Any],
    experiment: Mapping[str, Any],
    *,
    training_manifest: Path,
    validation_manifest: Path,
) -> None:
    relocation = report.get("rnd_a1_artifact_relocation")
    if (
        not isinstance(relocation, dict)
        or relocation.get("schema_version")
        != "catan-zero-rnd-a1-artifact-relocation/v1"
        or not isinstance(relocation.get("files"), dict)
    ):
        raise ExportError("training report A1 artifact relocation is invalid")
    files = relocation["files"]
    registered = experiment["registration"]["a1_artifact_sha256"]
    if set(files) != set(registered):
        raise ExportError("training report A1 artifact role set differs")
    directory = Path(str(report.get("rnd_a1_artifact_dir"))).expanduser().resolve()
    if not directory.is_dir():
        raise ExportError("training report A1 artifact directory is missing")
    names: set[str] = set()
    for role, expected in registered.items():
        item = files[role]
        if not isinstance(item, dict) or set(item) != {
            "logical_path",
            "filename",
            "sha256",
        }:
            raise ExportError(f"training report A1 artifact {role} is malformed")
        filename = item["filename"]
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ExportError(f"training report A1 artifact {role} filename is invalid")
        names.add(filename)
        physical = directory / filename
        if (
            item["sha256"] != "sha256:" + expected
            or not physical.is_file()
            or _sha256_file(physical) != expected
        ):
            raise ExportError(f"training report A1 artifact {role} hash mismatch")
    if {item.name for item in directory.iterdir() if item.is_file()} != names:
        raise ExportError("training report A1 artifact directory is not exact")
    if files["selected_game_manifest"]["sha256"] != "sha256:" + _sha256_file(
        training_manifest
    ):
        raise ExportError("relocated selected-game manifest differs")
    if files["validation_manifest"]["sha256"] != "sha256:" + _sha256_file(
        validation_manifest
    ):
        raise ExportError("relocated validation manifest differs")


def _validate_report(
    report: Mapping[str, Any],
    *,
    report_path: Path,
    checkpoint: Path,
    corpus_dir: Path,
    training_manifest: Path,
    validation_manifest: Path,
    experiment: Mapping[str, Any],
    experiment_path: Path,
    admission_path: Path,
    arm_id: str,
    seed: int,
    arm: Mapping[str, Any],
    payload_inventory_sha: str,
) -> dict[str, Any]:
    _validate_sources(report, experiment)
    resolved_payload = report.get("resolved_train_config")
    if (
        not isinstance(resolved_payload, dict)
        or resolved_payload.get("pipeline") != "train"
        or not isinstance(resolved_payload.get("fields"), dict)
    ):
        raise ExportError("training report lacks canonical resolved TrainConfig")
    fields = resolved_payload["fields"]
    if set(fields) != {field.name for field in dataclasses.fields(TrainConfig)}:
        raise ExportError("resolved TrainConfig fields are incomplete")
    try:
        train_config = TrainConfig(**fields)
    except (TypeError, ValueError) as exc:
        raise ExportError(f"resolved TrainConfig is invalid: {exc}") from exc
    if (
        resolved_payload != train_config.canonical_payload()
        or report.get("config_hash") != train_config.config_hash()
        or report.get("full_config_hash") != train_config.full_config_hash()
    ):
        raise ExportError("training report TrainConfig hash/payload is invalid")

    common = experiment["common"]
    expected_fields = {
        "arch": "entity_graph",
        "data_format": "memmap",
        "mask_hidden_info": True,
        "seed": seed,
        "hidden_size": common["hidden_size"],
        "graph_layers": common["state_layers"],
        "attention_heads": common["attention_heads"],
        "entity_state_trunk": common["state_trunk"],
        "relational_block_pattern": common["relational_block_pattern"],
        "relational_ff_size": common["relational_ff_size"],
        "relational_bases": common["relational_bases"],
        "relational_action_cross_layers": common["relational_action_cross_layers"],
        "latent_deliberation_steps": arm["latent_deliberation_steps"],
        "latent_deliberation_slots": common["latent_deliberation_slots"],
        "resume_optimizer": False,
        "grow_from_checkpoint": "",
        "validation_game_seed_ranges": "",
        "validation_max_samples": 0,
        "rnd_allow_a1_learner_override": True,
    }
    recipe = experiment["training_recipe"]
    for key, value in recipe.items():
        if key in fields:
            expected_fields[key] = value
    for field, expected in expected_fields.items():
        if fields.get(field) != expected:
            raise ExportError(f"resolved TrainConfig {field} differs from registration")

    init_sha = experiment["registration"]["initial_checkpoint_sha256_by_arm_seed"][
        f"{arm_id}@{seed}"
    ]
    if (
        fields.get("init_checkpoint_sha256") != "sha256:" + init_sha
        or report.get("init_checkpoint_sha256") != "sha256:" + init_sha
    ):
        raise ExportError(
            "training report initial checkpoint differs from registration"
        )
    if (
        report.get("resume_optimizer") is not False
        or report.get("optimizer_restored") is not False
    ):
        raise ExportError("E3 run did not use a fresh optimizer")

    checkpoint_sha = _sha256_file(checkpoint)
    report_checkpoint = _resolve_path(
        report.get("checkpoint"), field="training report checkpoint"
    )
    if (
        report_checkpoint != checkpoint.resolve()
        or report.get("checkpoint_sha256") != "sha256:" + checkpoint_sha
    ):
        raise ExportError("training report checkpoint binding mismatch")
    sidecar = optimizer_sidecar_path(checkpoint).resolve()
    report_sidecar = _resolve_path(
        report.get("optimizer_sidecar"), field="training report optimizer sidecar"
    )
    if report_sidecar != sidecar or report.get(
        "optimizer_sidecar_sha256"
    ) != "sha256:" + _sha256_file(sidecar):
        raise ExportError("training report optimizer-sidecar binding mismatch")

    if report.get("parameter_count") != arm["expected_parameters"]:
        raise ExportError("training report parameter count differs from registration")
    batch = _positive_int(report.get("batch_size"), field="report.batch_size")
    accum = _positive_int(
        report.get("grad_accum_steps"), field="report.grad_accum_steps"
    )
    world = _positive_int(report.get("world_size"), field="report.world_size")
    global_batch = batch * accum * world
    steps = _positive_int(report.get("steps_completed"), field="report.steps_completed")
    presentations = steps * global_batch
    if (
        world != 1
        or global_batch != recipe["global_batch_size"]
        or report.get("global_batch_size") != global_batch
        or steps != recipe["max_steps"]
        or presentations != recipe["sample_presentations_per_arm_seed"]
        or report.get("sample_presentations") != presentations
    ):
        raise ExportError("training report budget differs from E3 registration")

    if report.get("data_fingerprint") != _training_data_fingerprint(
        str(corpus_dir), "memmap"
    ) or fields.get("data_fingerprint") != report.get("data_fingerprint"):
        raise ExportError("training report corpus fingerprint mismatch")
    if (
        report.get("a1_memmap_payload_inventory_sha256") != payload_inventory_sha
        or fields.get("a1_memmap_payload_inventory_sha256") != payload_inventory_sha
    ):
        raise ExportError("training report corpus payload inventory mismatch")
    validation_sha = "sha256:" + _sha256_file(validation_manifest)
    if report.get("input_validation_game_seed_manifest_sha256") != validation_sha:
        raise ExportError("training report validation-manifest binding mismatch")
    validation = _load_object(validation_manifest, name="validation manifest")
    if report.get("validation_game_seed_count") != EXPECTED_HOLDOUT_GAMES or report.get(
        "validation_game_seed_set_sha256"
    ) != validation.get("validation_game_seed_set_sha256"):
        raise ExportError("training report validation-game support mismatch")
    if report.get("validation_samples") != EXPECTED_HOLDOUT_ROWS:
        raise ExportError("training report validation-row support mismatch")
    if (
        report.get("samples") != EXPECTED_CORPUS_ROWS
        or report.get("train_samples") != EXPECTED_CORPUS_ROWS - EXPECTED_HOLDOUT_ROWS
    ):
        raise ExportError("training report authenticated corpus support mismatch")
    if report.get("graph_history_features") is not True:
        raise ExportError("training report must attest graph-history features")
    if Path(str(fields.get("data"))).expanduser().resolve() != corpus_dir.resolve():
        raise ExportError(
            "resolved TrainConfig data path differs from authenticated corpus"
        )
    if (
        Path(str(fields.get("validation_game_seed_manifest"))).expanduser().resolve()
        != validation_manifest.resolve()
    ):
        raise ExportError("resolved TrainConfig validation manifest path differs")
    input_validation = _resolve_path(
        report.get("input_validation_game_seed_manifest"),
        field="training report input validation manifest",
    )
    if input_validation != validation_manifest.resolve():
        raise ExportError("training report input validation path differs")
    output_validation = _resolve_path(
        report.get("validation_game_seed_manifest"),
        field="training report validation output sidecar",
    )
    if output_validation == validation_manifest.resolve() or not np.array_equal(
        _validation_seeds(
            _load_object(output_validation, name="validation output sidecar")
        ),
        _validation_seeds(validation),
    ):
        raise ExportError("training report validation output support differs")
    _validate_relocation(
        report,
        experiment,
        training_manifest=training_manifest,
        validation_manifest=validation_manifest,
    )
    return {
        "schema_version": "catan-zero-e3-run-provenance/v1",
        "arm_id": arm_id,
        "training_seed": seed,
        "latent_deliberation_steps": arm["latent_deliberation_steps"],
        "capacity_class": arm["capacity_class"],
        "parameter_count": arm["expected_parameters"],
        "checkpoint_sha256": checkpoint_sha,
        "optimizer_sidecar_sha256": _sha256_file(sidecar),
        "training_report_sha256": _sha256_file(report_path),
        "experiment_config_sha256": _sha256_file(experiment_path),
        "experiment_semantic_sha256": experiment["config_sha256"],
        "admission_manifest_sha256": _sha256_file(admission_path),
        "identity_report_sha256": experiment["registration"]["identity_report_sha256"],
        "initial_checkpoint_sha256": init_sha,
        "training_manifest_sha256": _sha256_file(training_manifest),
        "validation_manifest_sha256": _sha256_file(validation_manifest),
        "corpus_fingerprint": experiment["registration"]["corpus_fingerprint"],
        "payload_inventory_sha256": payload_inventory_sha,
        "optimizer_steps": steps,
        "global_batch_size": global_batch,
        "sample_presentations": presentations,
        "train_config_hash": report["config_hash"],
        "resolved_train_config": dict(resolved_payload),
        "resolved_train_config_sha256": _canonical_sha(resolved_payload),
        "graph_history_features": bool(report.get("graph_history_features")),
    }


def _validation_rows(
    corpus: Any,
    *,
    experiment: Mapping[str, Any],
    validation_manifest: Path,
    training_manifest: Path,
    payload_inventory_sha: str,
) -> list[dict[str, Any]]:
    meta = getattr(corpus, "meta", None)
    if (
        not isinstance(meta, Mapping)
        or meta.get("payload_inventory_sha256") != payload_inventory_sha
    ):
        raise ExportError("loaded corpus metadata differs from authenticated inventory")
    validation = _load_object(validation_manifest, name="validation manifest")
    seeds = _validation_seeds(validation)
    if len(seeds) != EXPECTED_HOLDOUT_GAMES:
        raise ExportError(
            f"E3 validation manifest must contain exactly {EXPECTED_HOLDOUT_GAMES} games"
        )
    training = _load_object(training_manifest, name="training manifest")
    _validate_training_manifest(
        training, validation_payload=validation, validation_seeds=seeds
    )
    if "game_seed" not in corpus or "decision_index" not in corpus:
        raise ExportError("corpus lacks game_seed/decision_index")
    corpus_seeds = np.asarray(corpus["game_seed"], dtype=np.int64)
    if len(corpus) != EXPECTED_CORPUS_ROWS or len(corpus_seeds) != EXPECTED_CORPUS_ROWS:
        raise ExportError(f"E3 corpus must contain exactly {EXPECTED_CORPUS_ROWS} rows")
    indices = np.flatnonzero(np.isin(corpus_seeds, seeds))
    if len(indices) != EXPECTED_HOLDOUT_ROWS:
        raise ExportError(
            f"E3 holdout must contain exactly {EXPECTED_HOLDOUT_ROWS} rows"
        )
    if not np.array_equal(np.unique(corpus_seeds[indices]), seeds):
        raise ExportError("E3 holdout game support differs from validation manifest")
    if "target_information_regime" not in corpus:
        raise ExportError("corpus lacks target_information_regime")
    regimes = set(np.asarray(corpus["target_information_regime"])[indices].astype(str))
    if regimes != {experiment["common"]["information_regime"]}:
        raise ExportError("E3 holdout is not entirely public-information target data")
    decisions = np.asarray(corpus["decision_index"], dtype=np.int64)
    rows: list[dict[str, Any]] = []
    identities: set[tuple[int, int]] = set()
    for index in indices:
        identity = (int(corpus_seeds[index]), int(decisions[index]))
        if identity[1] < 0 or identity in identities:
            raise ExportError("E3 holdout has invalid or duplicate decision identities")
        identities.add(identity)
        rows.append(
            {
                "row_index": int(index),
                "game_seed": identity[0],
                "decision_index": identity[1],
            }
        )
    rows.sort(key=lambda item: (item["game_seed"], item["decision_index"]))
    return rows


def _validate_loaded_policy(
    policy: Any, *, experiment: Mapping[str, Any], arm: Mapping[str, Any]
) -> None:
    if not bool(getattr(policy, "trained_with_masked_hidden_info", False)):
        raise ExportError("checkpoint is not attested as public-masked training")
    config = getattr(policy, "config", None)
    if config is None:
        raise ExportError("checkpoint has no entity-graph config")
    common = experiment["common"]
    expected = {
        "hidden_size": common["hidden_size"],
        "state_layers": common["state_layers"],
        "attention_heads": common["attention_heads"],
        "state_trunk": common["state_trunk"],
        "relational_block_pattern": common["relational_block_pattern"],
        "relational_ff_size": common["relational_ff_size"],
        "relational_bases": common["relational_bases"],
        "relational_action_cross_layers": common["relational_action_cross_layers"],
        "latent_deliberation_slots": common["latent_deliberation_slots"],
        "latent_deliberation_steps": arm["latent_deliberation_steps"],
    }
    for attribute, value in expected.items():
        if getattr(config, attribute, None) != value:
            raise ExportError(
                f"checkpoint config {attribute} differs from registered arm"
            )


def _publish_jsonl_atomic(output: Path, records: list[dict[str, Any]]) -> None:
    destination = output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ExportError(f"refusing to overwrite {destination}")
    descriptor, raw = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for record in records:
                stream.write(
                    json.dumps(
                        record, sort_keys=True, separators=(",", ":"), allow_nan=False
                    )
                )
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise ExportError(f"refusing to overwrite {destination}") from exc
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def export_holdout_evidence(
    *,
    experiment_config: Path,
    evidence_contract: Path,
    admission_manifest: Path,
    corpus_dir: Path,
    training_manifest: Path,
    validation_manifest: Path,
    output: Path,
    checkpoint: Path | None = None,
    training_report: Path | None = None,
    batch_size: int = 256,
    device: str = "cpu",
    corpus_loader: Callable[[Path], Any] = MemmapCorpus,
    policy_loader: Callable[..., Any] = EntityGraphPolicy.load,
) -> int:
    """Authenticate, evaluate, and atomically publish one E3 holdout JSONL."""

    if output.expanduser().resolve().exists():
        raise ExportError(f"refusing to overwrite {output}")
    _positive_int(batch_size, field="batch_size")
    experiment_config = experiment_config.expanduser().resolve()
    admission_manifest = admission_manifest.expanduser().resolve()
    experiment = _load_object(experiment_config, name="registered experiment")
    try:
        _validate_contract(experiment, registered=True)
    except ValueError as exc:
        raise ExportError(f"invalid E3 registration: {exc}") from exc
    evidence_contract = evidence_contract.expanduser().resolve()
    contract_provenance = _validate_export_contract(
        _load_object(evidence_contract, name="evidence-export contract"),
        contract_path=evidence_contract,
        experiment=experiment,
        experiment_path=experiment_config,
    )
    admission = _load_object(admission_manifest, name="admission manifest")
    checkpoint = (
        (checkpoint or Path(str(admission.get("checkpoint")))).expanduser().resolve()
    )
    training_report = (
        (training_report or Path(str(admission.get("report")))).expanduser().resolve()
    )
    if not checkpoint.is_file() or not training_report.is_file():
        raise ExportError("admitted checkpoint/report is missing")
    arm_id, seed, arm = _validate_admission(
        admission,
        admission_path=admission_manifest,
        experiment=experiment,
        experiment_path=experiment_config,
        checkpoint=checkpoint,
        report=training_report,
        corpus_dir=corpus_dir,
        validation_manifest=validation_manifest,
    )
    if (
        _corpus_fingerprint(corpus_dir)
        != experiment["registration"]["corpus_fingerprint"]
    ):
        raise ExportError("corpus fingerprint differs from E3 registration")
    payload_inventory_sha = _validate_corpus_payloads(corpus_dir)
    registration = experiment["registration"]
    if (
        _sha256_file(training_manifest) != registration["training_manifest_sha256"]
        or _sha256_file(validation_manifest)
        != registration["validation_manifest_sha256"]
    ):
        raise ExportError("training/validation manifest differs from E3 registration")
    report = _load_object(training_report, name="training report")
    provenance = _validate_report(
        report,
        report_path=training_report,
        checkpoint=checkpoint,
        corpus_dir=corpus_dir,
        training_manifest=training_manifest,
        validation_manifest=validation_manifest,
        experiment=experiment,
        experiment_path=experiment_config,
        admission_path=admission_manifest,
        arm_id=arm_id,
        seed=seed,
        arm=arm,
        payload_inventory_sha=payload_inventory_sha,
    )
    provenance.update(contract_provenance)
    corpus = corpus_loader(corpus_dir)
    rows = _validation_rows(
        corpus,
        experiment=experiment,
        validation_manifest=validation_manifest,
        training_manifest=training_manifest,
        payload_inventory_sha=payload_inventory_sha,
    )
    policy = policy_loader(checkpoint, device=device, strict_metadata=True)
    _validate_loaded_policy(policy, experiment=experiment, arm=arm)
    if (
        sum(parameter.numel() for parameter in policy.model.parameters())
        != arm["expected_parameters"]
    ):
        raise ExportError("loaded checkpoint parameter count differs from registration")

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
            players = np.asarray(entity["player_tokens"])
            if players.ndim != 3:
                raise ExportError(
                    "player_tokens must have batch/player/feature dimensions"
                )
            actor_mask = players[..., PLAYER_ACTOR_FLAG_SLOT] > 0.5
            if np.any(actor_mask.sum(axis=1) != 1):
                raise ExportError(
                    "every holdout row must identify exactly one acting player"
                )
            masked_players = mask_player_tokens_public(players)
            for slot in PUBLIC_MASK_PLAYER_SLOTS:
                if np.any(masked_players[..., slot][~actor_mask] != 0):
                    raise ExportError(
                        "public masking left opponent hidden information visible"
                    )
            entity["player_tokens"] = masked_players
            legal_ids = np.asarray(corpus["legal_action_ids"][indices])
            contexts = np.asarray(corpus["legal_action_context"][indices])
            actions = np.asarray(corpus["action_taken"][indices], dtype=np.int64)
            valid = np.asarray(entity["legal_action_mask"], dtype=np.bool_)
            if valid.shape != legal_ids.shape or not np.array_equal(
                valid, legal_ids >= 0
            ):
                raise ExportError(
                    "legal_action_mask differs from live legal_action_ids"
                )
            if "target_policy" not in corpus or "target_policy_mask" not in corpus:
                raise ExportError("corpus lacks soft target policy")
            target_policy = np.asarray(
                corpus["target_policy"][indices], dtype=np.float64
            )
            target_mask = np.asarray(
                corpus["target_policy_mask"][indices], dtype=np.bool_
            )
            if target_policy.shape != valid.shape or not np.array_equal(
                target_mask, valid
            ):
                raise ExportError(
                    "target policy support differs from live legal actions"
                )
            if (
                np.any(~np.isfinite(target_policy))
                or np.any(target_policy < 0)
                or np.any(target_policy[~valid] != 0)
            ):
                raise ExportError("target policy is invalid")
            if not np.allclose(
                np.where(valid, target_policy, 0).sum(axis=1), 1.0, rtol=0, atol=1e-5
            ):
                raise ExportError("target policy does not normalize over live actions")
            target_matches = legal_ids == actions[:, None]
            if np.any(target_matches.sum(axis=1) != 1):
                raise ExportError(
                    "holdout action does not occur exactly once in legal actions"
                )
            outputs = policy.forward_legal_np(
                entity, legal_ids, contexts, return_q=False
            )
            logits = (
                outputs["logits"]
                .float()
                .masked_fill(
                    ~torch.as_tensor(valid, device=outputs["logits"].device),
                    float("-inf"),
                )
            )
            hard_targets = torch.as_tensor(
                np.argmax(target_matches, axis=1),
                dtype=torch.long,
                device=logits.device,
            )
            log_probs = torch.log_softmax(logits, dim=-1)
            target_tensor = torch.as_tensor(
                target_policy, dtype=log_probs.dtype, device=log_probs.device
            )
            soft_ce = -(
                target_tensor
                * torch.where(
                    torch.as_tensor(valid, device=log_probs.device), log_probs, 0.0
                )
            ).sum(dim=-1)
            hard_ce = torch.nn.functional.cross_entropy(
                logits, hard_targets, reduction="none"
            )
            for row, soft, hard, legal_count in zip(
                selected,
                soft_ce.cpu().tolist(),
                hard_ce.cpu().tolist(),
                valid.sum(axis=1),
                strict=True,
            ):
                if (
                    not math.isfinite(soft)
                    or soft < 0
                    or not math.isfinite(hard)
                    or hard < 0
                ):
                    raise ExportError("model produced invalid holdout cross-entropy")
                evidence.append(
                    {
                        "schema_version": EVIDENCE_SCHEMA,
                        "arm_id": arm_id,
                        "training_seed": seed,
                        "game_id": f"seed:{row['game_seed']}",
                        "decision_id": f"seed:{row['game_seed']}:decision:{row['decision_index']}",
                        "soft_target_policy_ce": float(soft),
                        "hard_action_ce": float(hard),
                        "forced": bool(legal_count <= 1),
                        "public_masked": True,
                        "evaluation_split": "holdout",
                        "is_training_game": False,
                        "experiment_config_sha256": provenance[
                            "experiment_config_sha256"
                        ],
                        "corpus_fingerprint": provenance["corpus_fingerprint"],
                        "admission_manifest_sha256": provenance[
                            "admission_manifest_sha256"
                        ],
                        "evidence_export_contract_sha256": provenance[
                            "evidence_export_contract_sha256"
                        ],
                        "training_manifest_sha256": provenance[
                            "training_manifest_sha256"
                        ],
                        "validation_manifest_sha256": provenance[
                            "validation_manifest_sha256"
                        ],
                        "run_provenance": provenance,
                    }
                )
    if len(evidence) != EXPECTED_HOLDOUT_ROWS:
        raise ExportError("exported evidence row count drifted")
    _publish_jsonl_atomic(output, evidence)
    return len(evidence)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--evidence-contract", type=Path, required=True)
    parser.add_argument("--admission-manifest", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--training-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        rows = export_holdout_evidence(
            experiment_config=args.experiment_config,
            evidence_contract=args.evidence_contract,
            admission_manifest=args.admission_manifest,
            corpus_dir=args.corpus,
            training_manifest=args.training_manifest,
            validation_manifest=args.validation_manifest,
            checkpoint=args.checkpoint,
            training_report=args.training_report,
            output=args.output,
            batch_size=args.batch_size,
            device=args.device,
        )
    except (ExportError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"E3 holdout export failed: {exc}") from exc
    print(json.dumps({"output": str(args.output), "rows": rows}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
