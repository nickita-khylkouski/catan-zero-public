#!/usr/bin/env python3
"""Freeze and admit the E3 A1 latent-deliberation learning screen.

``register`` turns the checked-in, non-runnable template into an immutable
registration by hashing the exact A1 corpus/artifacts, learner sources,
identity-init evidence, and five initialization checkpoints. ``admit`` then
revalidates those bytes and exclusively publishes one arm/seed launch manifest
at the preregistered output directory. Neither command trains a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence


SCHEMA = "catan-zero-e3-a1-screen/v1"
ADMISSION_SCHEMA = "catan-zero-e3-a1-admission/v1"
IDENTITY_SCHEMA = "catan-zero-e3-identity-init/v1"
ARMS = {
    "rrt-k0": (0, 20_070_932, "smaller_k0"),
    "think-rrt-k1": (1, 22_146_068, "shared_think_22146068"),
    "think-rrt-k2": (2, 22_146_068, "shared_think_22146068"),
    "think-rrt-k4": (4, 22_146_068, "shared_think_22146068"),
    "think-rrt-k8": (8, 22_146_068, "shared_think_22146068"),
}
SEEDS = (11, 29, 47)
RUN_KEYS = tuple(f"{arm}@{seed}" for seed in SEEDS for arm in ARMS)
SOURCE_FILES = (
    "tools/train_bc.py",
    "tools/rnd_e3_a1_admission.py",
    "tools/rnd_latent_upgrade_checkpoint.py",
    "src/catan_zero/rl/pipeline_configs.py",
    "src/catan_zero/rl/entity_token_policy.py",
    "src/catan_zero/rl/sparse_topology_adapter.py",
)
ARTIFACT_ROLES = (
    "selected_game_manifest",
    "post_wave_audit",
    "validation_manifest",
    "contract_lock",
)


class AdmissionError(ValueError):
    """The requested registration or run is not cryptographically admissible."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _existing_file(path: Path, *, field: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise AdmissionError(f"{field} is not a readable file: {path}") from exc
    if not resolved.is_file():
        raise AdmissionError(f"{field} is not a file: {resolved}")
    return resolved


def _existing_dir(path: Path, *, field: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise AdmissionError(f"{field} is not a readable directory: {path}") from exc
    if not resolved.is_dir():
        raise AdmissionError(f"{field} is not a directory: {resolved}")
    return resolved


def _load_object(path: Path, *, field: str) -> dict[str, Any]:
    path = _existing_file(path, field=field)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AdmissionError(f"cannot read {field} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AdmissionError(f"{field} must contain a JSON object")
    return value


def _publish_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise AdmissionError(f"refusing to overwrite {destination}")
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    temporary: Path | None = None
    try:
        descriptor, raw = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        temporary = Path(raw)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise AdmissionError(f"refusing to overwrite {destination}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _publish_many_exclusive(items: Sequence[tuple[Path, Mapping[str, Any]]]) -> None:
    """Stage and publish a manifest set, rolling back every link on failure."""

    destinations = [path.expanduser().resolve() for path, _value in items]
    if len(set(destinations)) != len(destinations):
        raise AdmissionError("batch admission contains duplicate destinations")
    for destination in destinations:
        if destination.exists():
            raise AdmissionError(f"refusing to overwrite {destination}")
    staged: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for (raw_destination, value), destination in zip(items, destinations, strict=True):
            del raw_destination
            destination.parent.mkdir(parents=True, exist_ok=True)
            encoded = (
                json.dumps(
                    value,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
            descriptor, raw = tempfile.mkstemp(
                prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
            )
            temporary = Path(raw)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            staged.append((temporary, destination))
        # Recheck the whole set immediately before the first irreversible link.
        for _temporary, destination in staged:
            if destination.exists():
                raise AdmissionError(f"refusing to overwrite {destination}")
        for temporary, destination in staged:
            try:
                os.link(temporary, destination)
            except FileExistsError as exc:
                raise AdmissionError(f"refusing to overwrite {destination}") from exc
            published.append(destination)
        for parent in {path.parent for path in published}:
            descriptor = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    except BaseException:
        for destination in reversed(published):
            destination.unlink(missing_ok=True)
        raise
    finally:
        for temporary, _destination in staged:
            temporary.unlink(missing_ok=True)


def _corpus_fingerprint(corpus_dir: Path) -> str:
    meta_path = _existing_file(corpus_dir / "corpus_meta.json", field="corpus metadata")
    meta = _load_object(meta_path, field="corpus metadata")
    inventory = meta.get("payload_inventory_sha256")
    if (
        not isinstance(inventory, str)
        or not inventory.startswith("sha256:")
        or not _is_sha(inventory[7:])
    ):
        raise AdmissionError(
            "A1 corpus metadata must bind a sha256:-prefixed payload inventory"
        )
    records = meta.get("payload_inventory")
    if meta.get("payload_inventory_schema") != "memmap-payload-inventory-v1":
        raise AdmissionError("A1 corpus payload inventory schema is unsupported")
    if not isinstance(records, list) or not records:
        raise AdmissionError("A1 corpus payload inventory is missing or empty")
    if "sha256:" + _canonical_sha(records) != inventory:
        raise AdmissionError("A1 corpus payload inventory semantic digest mismatch")
    prior: str | None = None
    registered_names: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != {"filename", "size_bytes", "sha256"}:
            raise AdmissionError(f"A1 payload inventory record {index} is malformed")
        filename = record["filename"]
        digest = record["sha256"]
        size = record["size_bytes"]
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or (prior is not None and filename <= prior)
            or type(size) is not int
            or size < 0
            or not isinstance(digest, str)
            or not digest.startswith("sha256:")
            or not _is_sha(digest[7:])
        ):
            raise AdmissionError(f"A1 payload inventory record {index} is invalid")
        payload = _existing_file(corpus_dir / filename, field=f"A1 payload {filename}")
        if payload.stat().st_size != size or _sha256_file(payload) != digest[7:]:
            raise AdmissionError(f"A1 payload {filename} differs from its inventory")
        registered_names.add(filename)
        prior = filename
    actual_names = {
        path.name
        for path in corpus_dir.iterdir()
        if path.is_file() and (path.name.endswith(".dat") or path.name.endswith(".codes.dat"))
    }
    if actual_names != registered_names:
        raise AdmissionError("A1 on-disk payload filenames differ from inventory")
    return _canonical_sha(
        {
            "corpus_meta_file_sha256": "sha256:" + _sha256_file(meta_path),
            "payload_inventory_sha256": inventory,
        }
    )


def _validate_contract(config: Mapping[str, Any], *, registered: bool) -> None:
    if config.get("schema_version") != SCHEMA:
        raise AdmissionError("unsupported E3 experiment schema")
    if config.get("config_sha256_scope") != "canonical_json_without_config_sha256":
        raise AdmissionError("unsupported experiment self-hash scope")
    semantic = dict(config)
    declared = semantic.pop("config_sha256", None)
    if not _is_sha(declared) or declared != _canonical_sha(semantic):
        raise AdmissionError("experiment config self-hash is invalid")
    common = config.get("common")
    required_common = {
        "hidden_size": 384,
        "state_layers": 9,
        "attention_heads": 6,
        "state_trunk": "rrt",
        "relational_block_pattern": "RRTRRTRRT",
        "relational_ff_size": 1024,
        "relational_bases": 4,
        "relational_action_cross_layers": 1,
        "latent_deliberation_slots": 8,
        "identity_initialization_required": True,
    }
    if not isinstance(common, dict) or any(
        common.get(key) != value for key, value in required_common.items()
    ):
        raise AdmissionError("experiment architecture differs from RRT384/L9 RRTRRTRRT")
    recipe = config.get("training_recipe")
    required_recipe = {
        "max_steps": 250,
        "batch_size": 1024,
        "grad_accum_steps": 4,
        "global_batch_size": 4096,
        "sample_presentations_per_arm_seed": 1_024_000,
        "amp": "bf16",
        "resume_optimizer": False,
        "rnd_allow_a1_learner_override": True,
        "graph_history_features": True,
        "skip_teacher_quality_gate": True,
        "trust_curated_data_quality": True,
        "allow_concurrent_bc": True,
        "symmetry_augment": False,
        "soft_target_temperature": 0.7,
        "soft_target_weight": 0.9,
        "soft_target_source": "policy",
        "policy_loss_weight": 1.0,
        "value_loss_weight": 0.25,
        "final_vp_loss_weight": 0.0,
        "q_loss_weight": 0.0,
        "value_lr_mult": 0.3,
        "value_target_lambda": 1.0,
        "truncated_vp_margin_value_weight": 0.25,
        "winner_sample_weight": 1.0,
        "loser_sample_weight": 0.3,
        "forced_action_weight": 0.1,
        "forced_row_value_weight": 1.0,
        "device": "cuda",
        "progress_every_batches": 25,
    }
    if not isinstance(recipe, dict) or any(
        recipe.get(key) != value for key, value in required_recipe.items()
    ):
        raise AdmissionError("experiment differs from the frozen 250-step/B4096 recipe")
    arms = config.get("arms")
    if not isinstance(arms, list) or len(arms) != len(ARMS):
        raise AdmissionError("experiment must contain exactly the five E3 arms")
    by_id = {item.get("arm_id"): item for item in arms if isinstance(item, dict)}
    if set(by_id) != set(ARMS):
        raise AdmissionError("experiment E3 arm IDs are incomplete or duplicated")
    for arm, (steps, params, capacity) in ARMS.items():
        item = by_id[arm]
        if (
            item.get("latent_deliberation_steps") != steps
            or item.get("expected_parameters") != params
            or item.get("capacity_class") != capacity
        ):
            raise AdmissionError(f"experiment architecture drift for arm {arm}")
    comparisons = config.get("comparison_contract")
    if not isinstance(comparisons, dict) or (
        comparisons.get("primary_reference_arm") != "think-rrt-k1"
        or comparisons.get("primary_candidate_arms")
        != ["think-rrt-k2", "think-rrt-k4"]
        or comparisons.get("compute_control_arms") != ["rrt-k0"]
    ):
        raise AdmissionError("capacity-aware K1/K2/K4 comparison contract drifted")
    matrix = config.get("run_matrix")
    if (
        not isinstance(matrix, dict)
        or matrix.get("seeds") != [11, 29, 47]
        or matrix.get("required_run_count") != 15
        or matrix.get("run_directory_pattern")
        != "runs/rnd_e3_a1_screen_20260710/{arm_id}/seed_{training_seed}"
    ):
        raise AdmissionError("E3 run matrix or exact output layout drifted")
    if registered:
        if config.get("status") != "registered_ready":
            raise AdmissionError("experiment registration is not ready")
        registration = config.get("registration")
        if not isinstance(registration, dict):
            raise AdmissionError("registered experiment has no registration object")
        for field in (
            "corpus_fingerprint",
            "training_manifest_sha256",
            "validation_manifest_sha256",
            "identity_report_sha256",
        ):
            if not _is_sha(registration.get(field)):
                raise AdmissionError(f"registration.{field} is not frozen")
        sources = registration.get("executing_learner_source_sha256")
        artifacts = registration.get("a1_artifact_sha256")
        checkpoints = registration.get("initial_checkpoint_sha256_by_arm_seed")
        if not isinstance(sources, dict) or set(sources) != set(SOURCE_FILES):
            raise AdmissionError("registration learner-source set is incomplete")
        if not isinstance(artifacts, dict) or set(artifacts) != set(ARTIFACT_ROLES):
            raise AdmissionError("registration A1 artifact set is incomplete")
        if not isinstance(checkpoints, dict) or set(checkpoints) != set(RUN_KEYS):
            raise AdmissionError("registration initialization-checkpoint set is incomplete")
        if any(not _is_sha(value) for value in (*sources.values(), *artifacts.values(), *checkpoints.values())):
            raise AdmissionError("registration contains an invalid SHA-256 digest")


def _validate_identity_report(
    report: Mapping[str, Any], checkpoint_hashes: Mapping[str, str]
) -> None:
    if report.get("schema_version") != IDENTITY_SCHEMA:
        raise AdmissionError("identity report has an unsupported schema")
    if report.get("reference_arm") != "rrt-k0":
        raise AdmissionError("identity report reference must be rrt-k0")
    seed_entries = report.get("seeds")
    if not isinstance(seed_entries, list):
        raise AdmissionError("identity report seeds must be a list")
    by_seed = {
        item.get("training_seed"): item
        for item in seed_entries
        if isinstance(item, dict)
    }
    if set(by_seed) != set(SEEDS):
        raise AdmissionError("identity report must contain exactly seeds 11, 29, and 47")
    for seed in SEEDS:
        seed_item = by_seed[seed]
        if not _is_sha(seed_item.get("probe_batch_sha256")):
            raise AdmissionError(f"identity report seed {seed} has no bound probe batch")
        entries = seed_item.get("arms")
        if not isinstance(entries, list):
            raise AdmissionError(f"identity report seed {seed} arms must be a list")
        by_id = {item.get("arm_id"): item for item in entries if isinstance(item, dict)}
        if set(by_id) != set(ARMS):
            raise AdmissionError(f"identity report seed {seed} must contain all E3 arms")
        expanded_state_hashes: set[str] = set()
        for arm, (steps, params, _) in ARMS.items():
            item = by_id[arm]
            key = f"{arm}@{seed}"
            state_hash = item.get("model_state_sha256")
            shared_hash = item.get("shared_base_state_sha256")
            if (
                item.get("latent_deliberation_steps") != steps
                or item.get("parameter_count") != params
                or item.get("checkpoint_sha256") != checkpoint_hashes[key]
                or item.get("compared_to") != f"rrt-k0@{seed}"
                or item.get("exact_identity") is not True
                or item.get("max_abs_logit_diff") != 0.0
                or item.get("max_abs_value_diff") != 0.0
                or item.get("max_abs_final_vp_diff") != 0.0
                or not _is_sha(state_hash)
                or not _is_sha(shared_hash)
            ):
                raise AdmissionError(f"identity initialization failed or drifted for {key}")
            if arm != "rrt-k0":
                expanded_state_hashes.add(state_hash)
                if shared_hash != by_id["rrt-k0"].get("model_state_sha256"):
                    raise AdmissionError(f"shared K0 tensors are not byte-identical for {key}")
        if len(expanded_state_hashes) != 1:
            raise AdmissionError(
                f"K1/K2/K4/K8 expanded model weights differ for training seed {seed}"
            )


def _artifact_hashes(paths: Mapping[str, Path]) -> dict[str, str]:
    if set(paths) != set(ARTIFACT_ROLES):
        raise AdmissionError("exactly four named A1 artifacts are required")
    return {
        role: _sha256_file(_existing_file(paths[role], field=f"A1 artifact {role}"))
        for role in ARTIFACT_ROLES
    }


def _source_hashes(source_root: Path) -> dict[str, str]:
    root = _existing_dir(source_root, field="source root")
    return {
        relative: _sha256_file(
            _existing_file(root / relative, field=f"executing source {relative}")
        )
        for relative in SOURCE_FILES
    }


def _checkpoint_hashes(paths: Mapping[str, Path]) -> dict[str, str]:
    if set(paths) != set(RUN_KEYS):
        raise AdmissionError("one initialization checkpoint is required for every E3 arm/seed")
    return {
        key: _sha256_file(_existing_file(paths[key], field=f"{key} checkpoint"))
        for key in RUN_KEYS
    }


def _tensor_state_sha(state: Mapping[str, Any], *, include: set[str] | None = None) -> str:
    """Hash tensor names, dtypes, shapes, and exact CPU bytes deterministically."""

    digest = hashlib.sha256()
    for name in sorted(state):
        if include is not None and name not in include:
            continue
        tensor = state[name].detach().cpu().contiguous()
        descriptor = json.dumps(
            [name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")
        ).encode("utf-8")
        digest.update(len(descriptor).to_bytes(8, "big"))
        digest.update(descriptor)
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _array_bundle_sha(arrays: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        import numpy as np

        array = np.ascontiguousarray(arrays[name])
        descriptor = json.dumps(
            [name, str(array.dtype), list(array.shape)], separators=(",", ":")
        ).encode("utf-8")
        digest.update(len(descriptor).to_bytes(8, "big"))
        digest.update(descriptor)
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def create_initializations(*, repo_root: Path) -> dict[str, Any]:
    """Create paired K0/expanded initialization families for all three seeds.

    Every expanded arm copies all shared tensors from its seed's K0 model. K1
    supplies the one expanded state, which is then loaded strictly into K2/K4/K8;
    only the checkpoint config's fixed execution count differs.
    """

    import torch

    from catan_zero.rl.entity_token_policy import EntityGraphPolicy
    from tools.factory_common import parse_track
    from tools.rnd_latent_upgrade_checkpoint import upgrade_checkpoint

    root = _existing_dir(repo_root, field="repository root")
    init_root = root / "runs/rnd_e3_a1_screen_20260710/initialization"
    identity_path = init_root / "identity_report.json"
    expected_paths = {
        f"{arm}@{seed}": init_root / f"seed_{seed}" / f"{arm}.pt"
        for seed in SEEDS
        for arm in ARMS
    }
    occupied = [path for path in (*expected_paths.values(), identity_path) if path.exists()]
    if occupied:
        raise AdmissionError(f"refusing to overwrite initialization artifact {occupied[0]}")

    def make_policy(seed: int, steps: int) -> Any:
        return EntityGraphPolicy.create(
            env_config=parse_track(
                "2p_no_trade", vps_to_win=10, use_graph_history_features=True
            ),
            hidden_size=384,
            state_layers=9,
            attention_heads=6,
            dropout=0.05,
            seed=seed,
            device="cpu",
            state_trunk="rrt",
            relational_block_pattern="RRTRRTRRT",
            relational_ff_size=1024,
            relational_bases=4,
            relational_action_cross_layers=1,
            latent_deliberation_steps=steps,
            latent_deliberation_slots=8,
        )

    report_seeds: list[dict[str, Any]] = []
    for seed in SEEDS:
        k0 = make_policy(seed, 0)
        k0_path = expected_paths[f"rrt-k0@{seed}"]
        k0_path.parent.mkdir(parents=True, exist_ok=True)
        k0.save(k0_path, mask_hidden_info=True, soft_target_source="policy")
        base_state = k0.model.state_dict()
        base_state_sha = _tensor_state_sha(base_state)
        arm_rows: list[dict[str, Any]] = [
            {
                "arm_id": "rrt-k0",
                "latent_deliberation_steps": 0,
                "parameter_count": ARMS["rrt-k0"][1],
                "checkpoint_sha256": _sha256_file(k0_path),
                "model_state_sha256": base_state_sha,
                "shared_base_state_sha256": base_state_sha,
                "compared_to": f"rrt-k0@{seed}",
                "exact_identity": True,
                "max_abs_logit_diff": 0.0,
                "max_abs_value_diff": 0.0,
                "max_abs_final_vp_diff": 0.0,
            }
        ]
        expanded_hashes: set[str] = set()
        for arm in ("think-rrt-k1", "think-rrt-k2", "think-rrt-k4", "think-rrt-k8"):
            path = expected_paths[f"{arm}@{seed}"]
            upgrade = upgrade_checkpoint(
                k0_path, path, steps=ARMS[arm][0], slots=8
            )
            if (
                upgrade.get("source_checkpoint_sha256") != _sha256_file(k0_path)
                or upgrade.get("source_parameter_count") != ARMS["rrt-k0"][1]
                or upgrade.get("output_parameter_count") != ARMS[arm][1]
                or upgrade.get("function_preserving_verification", {}).get("exact") is not True
            ):
                raise AdmissionError(f"latent upgrader attestation failed for {arm}@{seed}")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            state = payload["model"]
            if _tensor_state_sha(state, include=set(base_state)) != base_state_sha:
                raise AdmissionError(f"shared weights failed byte identity for {arm}@{seed}")
            state_sha = _tensor_state_sha(state)
            expanded_hashes.add(state_sha)
            arm_rows.append(
                {
                    "arm_id": arm,
                    "latent_deliberation_steps": ARMS[arm][0],
                    "parameter_count": ARMS[arm][1],
                    "checkpoint_sha256": _sha256_file(path),
                    "model_state_sha256": state_sha,
                    "shared_base_state_sha256": base_state_sha,
                    "compared_to": f"rrt-k0@{seed}",
                    "exact_identity": True,
                    "max_abs_logit_diff": 0.0,
                    "max_abs_value_diff": 0.0,
                    "max_abs_final_vp_diff": 0.0,
                    "latent_upgrade_attestation": upgrade,
                }
            )
        if len(expanded_hashes) != 1:
            raise AdmissionError(f"expanded weights differ across K for seed {seed}")
        probe_sha = _canonical_sha(
            {
                "generator": "tools.rnd_latent_upgrade_checkpoint._public_synthetic_batch",
                "synthetic_batch_seed": 20260710,
            }
        )
        report_seeds.append(
            {"training_seed": seed, "probe_batch_sha256": probe_sha, "arms": arm_rows}
        )
        del k0

    report = {
        "schema_version": IDENTITY_SCHEMA,
        "reference_arm": "rrt-k0",
        "construction": "per-seed K0 base; copy every shared tensor into K1; strict-load identical expanded state into K2/K4/K8",
        "seeds": report_seeds,
    }
    checkpoint_hashes = _checkpoint_hashes(expected_paths)
    _validate_identity_report(report, checkpoint_hashes)
    _publish_exclusive(identity_path, report)
    return {
        "identity_report": str(identity_path),
        "identity_report_sha256": _sha256_file(identity_path),
        "initial_checkpoint_paths": {key: str(path) for key, path in expected_paths.items()},
        "initial_checkpoint_sha256_by_arm_seed": checkpoint_hashes,
    }


def register_experiment(
    *,
    template: Path,
    corpus_dir: Path,
    training_manifest: Path,
    validation_manifest: Path,
    artifact_paths: Mapping[str, Path],
    identity_report: Path,
    checkpoint_paths: Mapping[str, Path],
    source_root: Path,
    output: Path,
) -> dict[str, Any]:
    """Freeze all experiment inputs and exclusively publish a runnable registration."""

    config = _load_object(template, field="experiment template")
    _validate_contract(config, registered=False)
    if config.get("status") != "registration_pending":
        raise AdmissionError("only a registration_pending template may be finalized")
    corpus_dir = _existing_dir(corpus_dir, field="A1 corpus")
    training_manifest = _existing_file(training_manifest, field="training manifest")
    validation_manifest = _existing_file(validation_manifest, field="validation manifest")
    checkpoints = _checkpoint_hashes(checkpoint_paths)
    identity_report = _existing_file(identity_report, field="identity report")
    identity = _load_object(identity_report, field="identity report")
    _validate_identity_report(identity, checkpoints)
    registered = dict(config)
    registered["status"] = "registered_ready"
    registered["registration"] = {
        "corpus_fingerprint": _corpus_fingerprint(corpus_dir),
        "training_manifest_sha256": _sha256_file(training_manifest),
        "validation_manifest_sha256": _sha256_file(validation_manifest),
        "a1_artifact_sha256": _artifact_hashes(artifact_paths),
        "executing_learner_source_sha256": _source_hashes(source_root),
        "identity_report_sha256": _sha256_file(identity_report),
        "initial_checkpoint_sha256_by_arm_seed": checkpoints,
    }
    registered.pop("config_sha256", None)
    registered["config_sha256"] = _canonical_sha(registered)
    _validate_contract(registered, registered=True)
    _publish_exclusive(output, registered)
    return registered


def _require_registered_bytes(
    *,
    config: Mapping[str, Any],
    corpus_dir: Path,
    training_manifest: Path,
    validation_manifest: Path,
    artifact_paths: Mapping[str, Path],
    identity_report: Path,
    checkpoint_paths: Mapping[str, Path],
    source_root: Path,
) -> None:
    registration = config["registration"]
    if _corpus_fingerprint(corpus_dir) != registration["corpus_fingerprint"]:
        raise AdmissionError("A1 corpus fingerprint differs from registration")
    if _sha256_file(training_manifest) != registration["training_manifest_sha256"]:
        raise AdmissionError("training manifest differs from registration")
    if _sha256_file(validation_manifest) != registration["validation_manifest_sha256"]:
        raise AdmissionError("validation manifest differs from registration")
    if _artifact_hashes(artifact_paths) != registration["a1_artifact_sha256"]:
        raise AdmissionError("A1 artifacts differ from registration")
    if _source_hashes(source_root) != registration["executing_learner_source_sha256"]:
        raise AdmissionError("executing learner sources differ from registration")
    checkpoints = _checkpoint_hashes(checkpoint_paths)
    if checkpoints != registration["initial_checkpoint_sha256_by_arm_seed"]:
        raise AdmissionError("initialization checkpoints differ from registration")
    if _sha256_file(identity_report) != registration["identity_report_sha256"]:
        raise AdmissionError("identity report differs from registration")
    _validate_identity_report(
        _load_object(identity_report, field="identity report"), checkpoints
    )


def _train_argv(
    config: Mapping[str, Any],
    *,
    arm: str,
    seed: int,
    corpus_dir: Path,
    validation_manifest: Path,
    artifact_dir: Path,
    checkpoint_init: Path,
    run_dir: Path,
) -> list[str]:
    recipe = config["training_recipe"]
    steps = ARMS[arm][0]
    return [
        "python3", "tools/train_bc.py",
        "--data", str(corpus_dir),
        "--data-format", "memmap",
        "--arch", "entity_graph",
        "--track", str(recipe["track"]),
        "--vps-to-win", str(recipe["vps_to_win"]),
        "--mask-hidden-info",
        "--graph-history-features",
        "--rnd-allow-a1-learner-override",
        "--rnd-a1-artifact-dir", str(artifact_dir),
        "--skip-teacher-quality-gate",
        "--trust-curated-data-quality",
        "--allow-concurrent-bc",
        "--validation-game-seed-manifest", str(validation_manifest),
        "--validation-max-samples", "0",
        "--validation-seed", str(recipe["validation_seed"]),
        "--epochs", "1",
        "--max-steps", "250",
        "--batch-size", "1024",
        "--grad-accum-steps", "4",
        "--amp", "bf16",
        "--optimizer", "adam",
        "--weight-decay", "0.0",
        "--lr", "3e-05",
        "--lr-warmup-steps", "25",
        "--lr-schedule", "flat",
        "--hidden-size", "384",
        "--graph-layers", "9",
        "--attention-heads", "6",
        "--graph-dropout", "0.05",
        "--entity-state-trunk", "rrt",
        "--relational-block-pattern", "RRTRRTRRT",
        "--relational-ff-size", "1024",
        "--relational-bases", "4",
        "--relational-action-cross-layers", "1",
        "--latent-deliberation-steps", str(steps),
        "--latent-deliberation-slots", "8",
        "--no-symmetry-augment",
        "--symmetry-augment-events",
        "--soft-target-temperature", "0.7",
        "--soft-target-weight", "0.9",
        "--soft-target-source", "policy",
        "--soft-target-min-legal-coverage", "0.5",
        "--policy-loss-weight", "1.0",
        "--value-loss-weight", "0.25",
        "--final-vp-loss-weight", "0.0",
        "--q-loss-weight", "0.0",
        "--policy-kl-anchor-weight", "0.0",
        "--value-uncertainty-loss-weight", "0.0",
        "--aux-subgoal-loss-weight", "0.0",
        "--value-lr-mult", "0.3",
        "--action-module-lr-mult", "1.0",
        "--value-head-type", "mse",
        "--value-target-lambda", "1.0",
        "--truncated-vp-margin-value-weight", "0.25",
        "--winner-sample-weight", "1.0",
        "--loser-sample-weight", "0.3",
        "--forced-action-weight", "0.1",
        "--forced-row-value-weight", "1.0",
        "--init-checkpoint", str(checkpoint_init),
        "--no-resume-optimizer",
        "--seed", str(seed),
        "--device", "cuda",
        "--progress-every-batches", "25",
        "--checkpoint", str(run_dir / config["run_matrix"]["checkpoint_filename"]),
        "--report", str(run_dir / config["run_matrix"]["report_filename"]),
    ]


def admit_run(
    *,
    experiment: Path,
    arm: str,
    training_seed: int,
    corpus_dir: Path,
    training_manifest: Path,
    validation_manifest: Path,
    artifact_paths: Mapping[str, Path],
    identity_report: Path,
    checkpoint_paths: Mapping[str, Path],
    source_root: Path,
    output: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Reauthenticate and exclusively publish one exact arm/seed admission."""

    config = _load_object(experiment, field="registered experiment")
    _validate_contract(config, registered=True)
    if arm not in ARMS:
        raise AdmissionError(f"unregistered E3 arm: {arm}")
    if type(training_seed) is not int or training_seed not in config["run_matrix"]["seeds"]:
        raise AdmissionError(f"unregistered E3 training seed: {training_seed}")
    repo_root = _existing_dir(repo_root, field="repository root")
    corpus_dir = _existing_dir(corpus_dir, field="A1 corpus")
    training_manifest = _existing_file(training_manifest, field="training manifest")
    validation_manifest = _existing_file(validation_manifest, field="validation manifest")
    identity_report = _existing_file(identity_report, field="identity report")
    _require_registered_bytes(
        config=config,
        corpus_dir=corpus_dir,
        training_manifest=training_manifest,
        validation_manifest=validation_manifest,
        artifact_paths=artifact_paths,
        identity_report=identity_report,
        checkpoint_paths=checkpoint_paths,
        source_root=source_root,
    )
    expected_output, payload = _build_admission_payload(
        config=config,
        experiment=experiment,
        arm=arm,
        training_seed=training_seed,
        corpus_dir=corpus_dir,
        validation_manifest=validation_manifest,
        artifact_paths=artifact_paths,
        checkpoint_paths=checkpoint_paths,
        repo_root=repo_root,
    )
    if output.expanduser().resolve() != expected_output:
        raise AdmissionError(f"admission output must be exactly {expected_output}")
    _publish_exclusive(output, payload)
    return payload


def _build_admission_payload(
    *,
    config: Mapping[str, Any],
    experiment: Path,
    arm: str,
    training_seed: int,
    corpus_dir: Path,
    validation_manifest: Path,
    artifact_paths: Mapping[str, Path],
    checkpoint_paths: Mapping[str, Path],
    repo_root: Path,
) -> tuple[Path, dict[str, Any]]:
    relative_dir = config["run_matrix"]["run_directory_pattern"].format(
        arm_id=arm, training_seed=training_seed
    )
    run_dir = (repo_root / relative_dir).resolve()
    if not run_dir.is_relative_to(repo_root):
        raise AdmissionError("registered run directory escapes repository root")
    expected_output = run_dir / config["run_matrix"]["admission_manifest_filename"]
    artifact_dirs = {path.expanduser().resolve().parent for path in artifact_paths.values()}
    if len(artifact_dirs) != 1:
        raise AdmissionError("all relocated A1 artifacts must share one direct directory")
    payload = {
        "schema_version": ADMISSION_SCHEMA,
        "experiment_config_sha256": _sha256_file(_existing_file(experiment, field="registered experiment")),
        "experiment_semantic_sha256": config["config_sha256"],
        "arm_id": arm,
        "training_seed": training_seed,
        "latent_deliberation_steps": ARMS[arm][0],
        "expected_parameters": ARMS[arm][1],
        "capacity_class": ARMS[arm][2],
        "run_directory": str(run_dir),
        "checkpoint": str(run_dir / config["run_matrix"]["checkpoint_filename"]),
        "report": str(run_dir / config["run_matrix"]["report_filename"]),
        "registered_hashes": config["registration"],
        "train_argv": _train_argv(
            config,
            arm=arm,
            seed=training_seed,
            corpus_dir=corpus_dir,
            validation_manifest=validation_manifest,
            artifact_dir=next(iter(artifact_dirs)),
            checkpoint_init=_existing_file(
                checkpoint_paths[f"{arm}@{training_seed}"],
                field=f"{arm}@{training_seed} checkpoint",
            ),
            run_dir=run_dir,
        ),
    }
    return expected_output, payload


def admit_all(
    *,
    experiment: Path,
    corpus_dir: Path,
    training_manifest: Path,
    validation_manifest: Path,
    artifact_paths: Mapping[str, Path],
    identity_report: Path,
    checkpoint_paths: Mapping[str, Path],
    source_root: Path,
    repo_root: Path,
    run_keys: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Authenticate once and transactionally publish a registered run subset."""

    config = _load_object(experiment, field="registered experiment")
    _validate_contract(config, registered=True)
    repo_root = _existing_dir(repo_root, field="repository root")
    corpus_dir = _existing_dir(corpus_dir, field="A1 corpus")
    training_manifest = _existing_file(training_manifest, field="training manifest")
    validation_manifest = _existing_file(validation_manifest, field="validation manifest")
    identity_report = _existing_file(identity_report, field="identity report")
    selected = list(RUN_KEYS if run_keys is None else run_keys)
    if not selected:
        raise AdmissionError("admit-all requires at least one registered run")
    if len(set(selected)) != len(selected):
        raise AdmissionError("admit-all run subset contains duplicates")
    if any(key not in RUN_KEYS for key in selected):
        raise AdmissionError("admit-all run subset contains an unregistered arm/seed")

    # This is deliberately the only corpus/authentication pass for the batch.
    _require_registered_bytes(
        config=config,
        corpus_dir=corpus_dir,
        training_manifest=training_manifest,
        validation_manifest=validation_manifest,
        artifact_paths=artifact_paths,
        identity_report=identity_report,
        checkpoint_paths=checkpoint_paths,
        source_root=source_root,
    )
    publications: list[tuple[Path, Mapping[str, Any]]] = []
    payloads: list[dict[str, Any]] = []
    for key in selected:
        arm, raw_seed = key.rsplit("@", 1)
        output, payload = _build_admission_payload(
            config=config,
            experiment=experiment,
            arm=arm,
            training_seed=int(raw_seed),
            corpus_dir=corpus_dir,
            validation_manifest=validation_manifest,
            artifact_paths=artifact_paths,
            checkpoint_paths=checkpoint_paths,
            repo_root=repo_root,
        )
        publications.append((output, payload))
        payloads.append(payload)
    _publish_many_exclusive(publications)
    return payloads


def _named_paths(values: Sequence[str], *, expected: set[str], field: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for raw in values:
        name, separator, path = raw.partition("=")
        if not separator or name not in expected or name in result or not path:
            raise AdmissionError(f"invalid or duplicate {field}: {raw!r}")
        result[name] = Path(path)
    if set(result) != expected:
        raise AdmissionError(f"{field} names must be exactly {sorted(expected)}")
    return result


def _shared_parser(parser: argparse.ArgumentParser, *, output: bool = True) -> None:
    parser.add_argument("--corpus-dir", required=True, type=Path)
    parser.add_argument("--training-manifest", required=True, type=Path)
    parser.add_argument("--validation-manifest", required=True, type=Path)
    parser.add_argument(
        "--a1-artifact", action="append", default=[], metavar="ROLE=PATH", required=True
    )
    parser.add_argument("--identity-report", required=True, type=Path)
    parser.add_argument(
        "--init-checkpoint", action="append", default=[], metavar="ARM=PATH", required=True
    )
    parser.add_argument("--source-root", required=True, type=Path)
    if output:
        parser.add_argument("--output", required=True, type=Path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    initialize = subparsers.add_parser("initialize")
    initialize.add_argument("--repo-root", default=Path.cwd(), type=Path)
    register = subparsers.add_parser("register")
    register.add_argument("--template", required=True, type=Path)
    _shared_parser(register)
    admit = subparsers.add_parser("admit")
    admit.add_argument("--experiment", required=True, type=Path)
    admit.add_argument("--arm", required=True)
    admit.add_argument("--training-seed", required=True, type=int)
    admit.add_argument("--repo-root", default=Path.cwd(), type=Path)
    _shared_parser(admit)
    admit_all_parser = subparsers.add_parser("admit-all")
    admit_all_parser.add_argument("--experiment", required=True, type=Path)
    admit_all_parser.add_argument("--repo-root", default=Path.cwd(), type=Path)
    admit_all_parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="ARM@SEED",
        help="Registered run to publish; repeat for a host subset, omit for all 15.",
    )
    _shared_parser(admit_all_parser, output=False)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "initialize":
            payload = create_initializations(repo_root=args.repo_root)
            print(json.dumps(payload, sort_keys=True))
            return 0
        artifacts = _named_paths(
            args.a1_artifact, expected=set(ARTIFACT_ROLES), field="A1 artifact"
        )
        checkpoints = _named_paths(
            args.init_checkpoint, expected=set(RUN_KEYS), field="initial checkpoint"
        )
        common = dict(
            corpus_dir=args.corpus_dir,
            training_manifest=args.training_manifest,
            validation_manifest=args.validation_manifest,
            artifact_paths=artifacts,
            identity_report=args.identity_report,
            checkpoint_paths=checkpoints,
            source_root=args.source_root,
        )
        if args.command == "register":
            payload = register_experiment(
                template=args.template, output=args.output, **common
            )
        elif args.command == "admit":
            payload = admit_run(
                experiment=args.experiment,
                arm=args.arm,
                training_seed=args.training_seed,
                repo_root=args.repo_root,
                output=args.output,
                **common,
            )
        else:
            published = admit_all(
                experiment=args.experiment,
                repo_root=args.repo_root,
                run_keys=args.run or None,
                **common,
            )
            payload = {
                "published_count": len(published),
                "published": [
                    str(Path(item["run_directory"]) / "admission.json")
                    for item in published
                ],
            }
    except AdmissionError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
