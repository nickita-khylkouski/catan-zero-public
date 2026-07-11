#!/usr/bin/env python3
"""Freeze and admit the Transformer-think A1 learning screen.

This is deliberately independent of the frozen E3 registration.  It reuses
only E3's byte/atomic-publication primitives; all architecture, identity, run
matrix, and command contracts are versioned here.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.rnd_e3_a1_admission import (  # noqa: E402
    AdmissionError,
    _canonical_sha,
    _corpus_fingerprint,
    _existing_dir,
    _existing_file,
    _load_object,
    _publish_exclusive,
    _publish_many_exclusive,
    _sha256_file,
    _tensor_state_sha,
)


SCHEMA = "catan-zero-transformer-think-a1-screen/v1"
ADMISSION_SCHEMA = "catan-zero-transformer-think-a1-admission/v1"
IDENTITY_SCHEMA = "catan-zero-transformer-think-identity-init/v1"
FROZEN_INCUMBENT_CHECKPOINT_SHA256 = (
    "89aa133d629e747021bc725f2ad63e0563f3b76e71f0dd563f056c6de8f77ebb"
)
ARMS = {
    "transformer-k0": (0, 35_041_353, "smaller_k0"),
    "think-transformer-k1": (1, 40_793_673, "shared_think_40793673"),
    "think-transformer-k2": (2, 40_793_673, "shared_think_40793673"),
    "think-transformer-k4": (4, 40_793_673, "shared_think_40793673"),
}
SEEDS = (101, 103, 107)
RUN_KEYS = tuple(f"{arm}@{seed}" for seed in SEEDS for arm in ARMS)
SOURCE_FILES = (
    "tools/train_bc.py",
    "tools/rnd_transformer_think_a1_admission.py",
    "tools/rnd_e3_a1_admission.py",
    "tools/rnd_transformer_latent_upgrade_checkpoint.py",
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


def _is_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_contract(config: Mapping[str, Any], *, registered: bool) -> None:
    if config.get("schema_version") != SCHEMA:
        raise AdmissionError("unsupported Transformer-think experiment schema")
    if config.get("config_sha256_scope") != "canonical_json_without_config_sha256":
        raise AdmissionError("unsupported experiment self-hash scope")
    semantic = dict(config)
    declared = semantic.pop("config_sha256", None)
    if not _is_sha(declared) or declared != _canonical_sha(semantic):
        raise AdmissionError("experiment config self-hash is invalid")
    common = config.get("common")
    required_common = {
        "hidden_size": 640,
        "state_layers": 6,
        "attention_heads": 8,
        "state_trunk": "transformer",
        "latent_deliberation_slots": 8,
        "identity_initialization_required": True,
        "frozen_incumbent_checkpoint_sha256": FROZEN_INCUMBENT_CHECKPOINT_SHA256,
    }
    if not isinstance(common, dict) or any(
        common.get(key) != value for key, value in required_common.items()
    ):
        raise AdmissionError("experiment architecture differs from Transformer h640/L6")
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
        raise AdmissionError("experiment differs from the frozen A1 250-step/B4096 recipe")
    arms = config.get("arms")
    if not isinstance(arms, list) or len(arms) != len(ARMS):
        raise AdmissionError("experiment must contain exactly four Transformer-think arms")
    by_id = {item.get("arm_id"): item for item in arms if isinstance(item, dict)}
    if set(by_id) != set(ARMS):
        raise AdmissionError("Transformer-think arm IDs are incomplete or duplicated")
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
        comparisons.get("primary_reference_arm") != "think-transformer-k1"
        or comparisons.get("primary_candidate_arms")
        != ["think-transformer-k2", "think-transformer-k4"]
        or comparisons.get("compute_control_arms") != ["transformer-k0"]
        or comparisons.get("capacity_matched_arm_ids")
        != [
            "think-transformer-k1",
            "think-transformer-k2",
            "think-transformer-k4",
        ]
        or comparisons.get("primary_metric")
        != "game_macro_soft_target_policy_ce_nonforced"
        or comparisons.get("minimum_relative_improvement_vs_k1") != 0.02
        or comparisons.get("maximum_nonforced_decision_micro_ce_regression")
        != 0.005
    ):
        raise AdmissionError("capacity-aware Transformer K1/K2/K4 contract drifted")
    matrix = config.get("run_matrix")
    if (
        not isinstance(matrix, dict)
        or matrix.get("seeds") != list(SEEDS)
        or matrix.get("required_run_count") != len(RUN_KEYS)
        or matrix.get("run_directory_pattern")
        != "runs/rnd_transformer_think_a1_screen_20260711/{arm_id}/seed_{training_seed}"
    ):
        raise AdmissionError("Transformer-think run matrix or output layout drifted")
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
            "source_teacher_checkpoint_sha256",
        ):
            if not _is_sha(registration.get(field)):
                raise AdmissionError(f"registration.{field} is not frozen")
        if (
            registration["source_teacher_checkpoint_sha256"]
            != FROZEN_INCUMBENT_CHECKPOINT_SHA256
        ):
            raise AdmissionError("registration source teacher is not the frozen incumbent")
        sources = registration.get("executing_learner_source_sha256")
        artifacts = registration.get("a1_artifact_sha256")
        checkpoints = registration.get("initial_checkpoint_sha256_by_arm_seed")
        if not isinstance(sources, dict) or set(sources) != set(SOURCE_FILES):
            raise AdmissionError("registration learner-source set is incomplete")
        if not isinstance(artifacts, dict) or set(artifacts) != set(ARTIFACT_ROLES):
            raise AdmissionError("registration A1 artifact set is incomplete")
        if not isinstance(checkpoints, dict) or set(checkpoints) != set(RUN_KEYS):
            raise AdmissionError("registration checkpoint set is incomplete")
        if any(
            not _is_sha(value)
            for value in (*sources.values(), *artifacts.values(), *checkpoints.values())
        ):
            raise AdmissionError("registration contains an invalid SHA-256 digest")


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
        raise AdmissionError("one initialization checkpoint is required for every run")
    return {
        key: _sha256_file(_existing_file(paths[key], field=f"{key} checkpoint"))
        for key in RUN_KEYS
    }


def _validate_identity_report(
    report: Mapping[str, Any], checkpoint_hashes: Mapping[str, str]
) -> None:
    if report.get("schema_version") != IDENTITY_SCHEMA:
        raise AdmissionError("identity report has an unsupported schema")
    if report.get("reference_arm") != "transformer-k0":
        raise AdmissionError("identity report reference must be transformer-k0")
    if (
        report.get("source_teacher_checkpoint_sha256")
        != FROZEN_INCUMBENT_CHECKPOINT_SHA256
    ):
        raise AdmissionError("identity report source teacher differs from the incumbent")
    rows = report.get("seeds")
    if not isinstance(rows, list):
        raise AdmissionError("identity report seeds must be a list")
    by_seed = {row.get("training_seed"): row for row in rows if isinstance(row, dict)}
    if set(by_seed) != set(SEEDS):
        raise AdmissionError("identity report must contain exactly seeds 101, 103, and 107")
    teacher_state_hash: str | None = None
    for seed in SEEDS:
        seed_row = by_seed[seed]
        if not _is_sha(seed_row.get("probe_batch_sha256")):
            raise AdmissionError(f"identity report seed {seed} has no bound probe batch")
        arms = seed_row.get("arms")
        if not isinstance(arms, list):
            raise AdmissionError(f"identity report seed {seed} arms must be a list")
        by_arm = {row.get("arm_id"): row for row in arms if isinstance(row, dict)}
        if set(by_arm) != set(ARMS):
            raise AdmissionError(f"identity report seed {seed} must contain all arms")
        expanded_hashes: set[str] = set()
        base_hash = by_arm["transformer-k0"].get("model_state_sha256")
        if teacher_state_hash is None:
            teacher_state_hash = base_hash
        elif base_hash != teacher_state_hash:
            raise AdmissionError("K0 teacher model state differs across training seeds")
        for arm, (steps, params, _capacity) in ARMS.items():
            row = by_arm[arm]
            key = f"{arm}@{seed}"
            state_hash = row.get("model_state_sha256")
            if (
                row.get("latent_deliberation_steps") != steps
                or row.get("parameter_count") != params
                or row.get("checkpoint_sha256") != checkpoint_hashes[key]
                or row.get("compared_to") != f"transformer-k0@{seed}"
                or row.get("exact_identity") is not True
                or row.get("max_abs_logit_diff") != 0.0
                or row.get("max_abs_value_diff") != 0.0
                or row.get("max_abs_final_vp_diff") != 0.0
                or not _is_sha(state_hash)
                or row.get("shared_base_state_sha256") != base_hash
                or (
                    arm == "transformer-k0"
                    and checkpoint_hashes[key]
                    != FROZEN_INCUMBENT_CHECKPOINT_SHA256
                )
            ):
                raise AdmissionError(f"identity initialization failed or drifted for {key}")
            if arm != "transformer-k0":
                expanded_hashes.add(state_hash)
        if len(expanded_hashes) != 1:
            raise AdmissionError(f"K1/K2/K4 expanded weights differ for seed {seed}")


def _copy_checkpoint_no_overwrite(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise AdmissionError(f"refusing to overwrite initialization artifact {destination}")
    descriptor, raw = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(raw)
    try:
        shutil.copyfile(source, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise AdmissionError(
                f"refusing to overwrite initialization artifact {destination}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def create_initializations(
    *,
    repo_root: Path,
    source_checkpoint: Path,
    reuse_existing: bool = False,
) -> dict[str, Any]:
    """Create or attest teacher-based K0/K1/K2/K4 initializations."""
    import torch

    from catan_zero.rl.entity_token_policy import EntityGraphPolicy
    from tools.rnd_transformer_latent_upgrade_checkpoint import (
        PROVENANCE_KEY,
        _public_synthetic_batch,
        upgrade_checkpoint,
    )

    root = _existing_dir(repo_root, field="repository root")
    source = _existing_file(source_checkpoint, field="frozen incumbent checkpoint")
    if _sha256_file(source) != FROZEN_INCUMBENT_CHECKPOINT_SHA256:
        raise AdmissionError("source checkpoint is not the frozen incumbent Transformer")
    teacher = EntityGraphPolicy.load(source, device="cpu")
    teacher_config = teacher.config
    teacher_parameters = sum(parameter.numel() for parameter in teacher.model.parameters())
    if (
        teacher_config.state_trunk != "transformer"
        or teacher_config.hidden_size != 640
        or teacher_config.state_layers != 6
        or teacher_config.attention_heads != 8
        or teacher_config.latent_deliberation_steps != 0
        or teacher_parameters != ARMS["transformer-k0"][1]
        or getattr(teacher, "trained_with_masked_hidden_info", None) is not True
    ):
        raise AdmissionError("frozen incumbent checkpoint architecture/masking drifted")

    init_root = root / "runs/rnd_transformer_think_a1_screen_20260711/initialization"
    identity_path = init_root / "identity_report.json"
    paths = {
        f"{arm}@{seed}": init_root / f"seed_{seed}" / f"{arm}.pt"
        for seed in SEEDS
        for arm in ARMS
    }
    if identity_path.exists():
        raise AdmissionError(f"refusing to overwrite initialization artifact {identity_path}")
    existing = [path for path in paths.values() if path.exists()]
    if reuse_existing:
        if len(existing) != len(paths):
            raise AdmissionError("reuse-existing requires all 12 initialization checkpoints")
    elif existing:
        raise AdmissionError(f"refusing to overwrite initialization artifact {existing[0]}")
    else:
        for seed in SEEDS:
            k0_path = paths[f"transformer-k0@{seed}"]
            _copy_checkpoint_no_overwrite(source, k0_path)
            for arm in (
                "think-transformer-k1",
                "think-transformer-k2",
                "think-transformer-k4",
            ):
                attestation = upgrade_checkpoint(
                    source,
                    paths[f"{arm}@{seed}"],
                    steps=ARMS[arm][0],
                    slots=8,
                    initialization_seed=seed,
                )
                if (
                    attestation.get("source_checkpoint_sha256")
                    != FROZEN_INCUMBENT_CHECKPOINT_SHA256
                    or attestation.get("function_preserving_verification", {}).get(
                        "exact"
                    )
                    is not True
                ):
                    raise AdmissionError(
                        f"Transformer upgrader attestation failed for {arm}@{seed}"
                    )

    teacher.model.eval()
    batch = _public_synthetic_batch(teacher_config)
    with torch.no_grad():
        teacher_outputs = teacher.model(batch, return_q=True)
    base_state = teacher.model.state_dict()
    base_names = set(base_state)
    base_sha = _tensor_state_sha(base_state)
    report_seeds: list[dict[str, Any]] = []
    for seed in SEEDS:
        arm_rows: list[dict[str, Any]] = []
        expanded_hashes: set[str] = set()
        for arm, (steps, parameters, _capacity) in ARMS.items():
            key = f"{arm}@{seed}"
            path = _existing_file(paths[key], field=f"{key} initialization")
            if arm == "transformer-k0" and _sha256_file(path) != (
                FROZEN_INCUMBENT_CHECKPOINT_SHA256
            ):
                raise AdmissionError(f"{key} is not byte-identical to the frozen teacher")
            policy = EntityGraphPolicy.load(path, device="cpu")
            if (
                policy.config.state_trunk != "transformer"
                or policy.config.hidden_size != 640
                or policy.config.state_layers != 6
                or policy.config.attention_heads != 8
                or policy.config.latent_deliberation_steps != steps
                or policy.config.latent_deliberation_slots != 8
                or sum(parameter.numel() for parameter in policy.model.parameters())
                != parameters
            ):
                raise AdmissionError(f"initialization architecture drifted for {key}")
            state = policy.model.state_dict()
            if _tensor_state_sha(state, include=base_names) != base_sha:
                raise AdmissionError(f"shared teacher tensors differ for {key}")
            state_sha = _tensor_state_sha(state)
            if arm != "transformer-k0":
                expanded_hashes.add(state_sha)
                raw = torch.load(path, map_location="cpu", weights_only=False)
                provenance = raw.get(PROVENANCE_KEY)
                if (
                    not isinstance(provenance, Mapping)
                    or provenance.get("source_checkpoint_sha256")
                    != FROZEN_INCUMBENT_CHECKPOINT_SHA256
                    or provenance.get("initialization_seed") != seed
                    or provenance.get("steps") != steps
                    or provenance.get("slots") != 8
                    or provenance.get("function_preserving_verification", {}).get(
                        "exact"
                    )
                    is not True
                ):
                    raise AdmissionError(f"latent upgrade provenance drifted for {key}")
            policy.model.eval()
            with torch.no_grad():
                outputs = policy.model(batch, return_q=True)
            differences: dict[str, float] = {}
            for output_name in ("logits", "value", "final_vp", "q_values"):
                if not torch.equal(teacher_outputs[output_name], outputs[output_name]):
                    raise AdmissionError(
                        f"function-preserving identity failed for {key}:{output_name}"
                    )
                differences[output_name] = 0.0
            arm_rows.append(
                {
                    "arm_id": arm,
                    "latent_deliberation_steps": steps,
                    "parameter_count": parameters,
                    "checkpoint_sha256": _sha256_file(path),
                    "model_state_sha256": state_sha,
                    "shared_base_state_sha256": base_sha,
                    "compared_to": f"transformer-k0@{seed}",
                    "exact_identity": True,
                    "max_abs_logit_diff": differences["logits"],
                    "max_abs_value_diff": differences["value"],
                    "max_abs_final_vp_diff": differences["final_vp"],
                    "max_abs_q_diff": differences["q_values"],
                }
            )
        if len(expanded_hashes) != 1:
            raise AdmissionError(f"expanded weights differ across K for seed {seed}")
        report_seeds.append(
            {
                "training_seed": seed,
                "probe_batch_sha256": _canonical_sha(
                    {
                        "generator": "tools.rnd_transformer_latent_upgrade_checkpoint._public_synthetic_batch",
                        "synthetic_batch_seed": 20260710,
                    }
                ),
                "arms": arm_rows,
            }
        )

    report = {
        "schema_version": IDENTITY_SCHEMA,
        "reference_arm": "transformer-k0",
        "source_teacher_checkpoint_sha256": FROZEN_INCUMBENT_CHECKPOINT_SHA256,
        "construction": "one frozen incumbent teacher for all K0 runs; seed-specific identical expanded state for shared K1/K2/K4 block",
        "seeds": report_seeds,
    }
    hashes = _checkpoint_hashes(paths)
    _validate_identity_report(report, hashes)
    _publish_exclusive(identity_path, report)
    return {
        "identity_report": str(identity_path),
        "identity_report_sha256": _sha256_file(identity_path),
        "source_teacher_checkpoint_sha256": FROZEN_INCUMBENT_CHECKPOINT_SHA256,
        "initial_checkpoint_paths": {key: str(path) for key, path in paths.items()},
        "initial_checkpoint_sha256_by_arm_seed": hashes,
    }


def register_experiment(
    *, template: Path, corpus_dir: Path, training_manifest: Path,
    validation_manifest: Path, artifact_paths: Mapping[str, Path],
    identity_report: Path, checkpoint_paths: Mapping[str, Path],
    source_root: Path, output: Path,
) -> dict[str, Any]:
    config = _load_object(template, field="experiment template")
    _validate_contract(config, registered=False)
    if config.get("status") != "registration_pending":
        raise AdmissionError("only a registration_pending template may be finalized")
    corpus = _existing_dir(corpus_dir, field="A1 corpus")
    training = _existing_file(training_manifest, field="training manifest")
    validation = _existing_file(validation_manifest, field="validation manifest")
    hashes = _checkpoint_hashes(checkpoint_paths)
    identity = _existing_file(identity_report, field="identity report")
    _validate_identity_report(_load_object(identity, field="identity report"), hashes)
    registered = dict(config)
    registered["status"] = "registered_ready"
    registered["registration"] = {
        "corpus_fingerprint": _corpus_fingerprint(corpus),
        "training_manifest_sha256": _sha256_file(training),
        "validation_manifest_sha256": _sha256_file(validation),
        "a1_artifact_sha256": _artifact_hashes(artifact_paths),
        "executing_learner_source_sha256": _source_hashes(source_root),
        "identity_report_sha256": _sha256_file(identity),
        "source_teacher_checkpoint_sha256": FROZEN_INCUMBENT_CHECKPOINT_SHA256,
        "initial_checkpoint_sha256_by_arm_seed": hashes,
    }
    registered.pop("config_sha256", None)
    registered["config_sha256"] = _canonical_sha(registered)
    _validate_contract(registered, registered=True)
    _publish_exclusive(output, registered)
    return registered


def _authenticate(
    *, config: Mapping[str, Any], corpus_dir: Path, training_manifest: Path,
    validation_manifest: Path, artifact_paths: Mapping[str, Path],
    identity_report: Path, checkpoint_paths: Mapping[str, Path], source_root: Path,
) -> None:
    frozen = config["registration"]
    if _corpus_fingerprint(corpus_dir) != frozen["corpus_fingerprint"]:
        raise AdmissionError("A1 corpus fingerprint differs from registration")
    if _sha256_file(training_manifest) != frozen["training_manifest_sha256"]:
        raise AdmissionError("training manifest differs from registration")
    if _sha256_file(validation_manifest) != frozen["validation_manifest_sha256"]:
        raise AdmissionError("validation manifest differs from registration")
    if _artifact_hashes(artifact_paths) != frozen["a1_artifact_sha256"]:
        raise AdmissionError("A1 artifacts differ from registration")
    if _source_hashes(source_root) != frozen["executing_learner_source_sha256"]:
        raise AdmissionError("executing learner sources differ from registration")
    hashes = _checkpoint_hashes(checkpoint_paths)
    if hashes != frozen["initial_checkpoint_sha256_by_arm_seed"]:
        raise AdmissionError("initialization checkpoints differ from registration")
    if _sha256_file(identity_report) != frozen["identity_report_sha256"]:
        raise AdmissionError("identity report differs from registration")
    _validate_identity_report(_load_object(identity_report, field="identity report"), hashes)


def _train_argv(
    config: Mapping[str, Any], *, arm: str, seed: int, corpus_dir: Path,
    validation_manifest: Path, artifact_dir: Path, checkpoint_init: Path,
    run_dir: Path,
) -> list[str]:
    recipe = config["training_recipe"]
    return [
        "python3", "tools/train_bc.py", "--data", str(corpus_dir),
        "--data-format", "memmap", "--arch", "entity_graph", "--track", recipe["track"],
        "--vps-to-win", str(recipe["vps_to_win"]), "--mask-hidden-info",
        "--graph-history-features", "--rnd-allow-a1-learner-override",
        "--rnd-a1-artifact-dir", str(artifact_dir), "--skip-teacher-quality-gate",
        "--trust-curated-data-quality", "--allow-concurrent-bc",
        "--validation-game-seed-manifest", str(validation_manifest),
        "--validation-max-samples", "0", "--validation-seed", str(recipe["validation_seed"]),
        "--epochs", "1", "--max-steps", "250", "--batch-size", "1024",
        "--grad-accum-steps", "4", "--amp", "bf16", "--optimizer", "adam",
        "--weight-decay", "0.0", "--lr", "3e-05", "--lr-warmup-steps", "25",
        "--lr-schedule", "flat", "--hidden-size", "640", "--graph-layers", "6",
        "--attention-heads", "8", "--graph-dropout", "0.05",
        "--entity-state-trunk", "transformer",
        "--latent-deliberation-steps", str(ARMS[arm][0]),
        "--latent-deliberation-slots", "8", "--no-symmetry-augment",
        "--symmetry-augment-events", "--soft-target-temperature", "0.7",
        "--soft-target-weight", "0.9", "--soft-target-source", "policy",
        "--soft-target-min-legal-coverage", "0.5", "--policy-loss-weight", "1.0",
        "--value-loss-weight", "0.25", "--final-vp-loss-weight", "0.0",
        "--q-loss-weight", "0.0", "--policy-kl-anchor-weight", "0.0",
        "--value-uncertainty-loss-weight", "0.0", "--aux-subgoal-loss-weight", "0.0",
        "--value-lr-mult", "0.3", "--action-module-lr-mult", "1.0",
        "--value-head-type", "mse", "--value-target-lambda", "1.0",
        "--truncated-vp-margin-value-weight", "0.25", "--winner-sample-weight", "1.0",
        "--loser-sample-weight", "0.3", "--forced-action-weight", "0.1",
        "--forced-row-value-weight", "1.0", "--init-checkpoint", str(checkpoint_init),
        "--no-resume-optimizer", "--seed", str(seed), "--device", "cuda",
        "--progress-every-batches", "25", "--checkpoint",
        str(run_dir / config["run_matrix"]["checkpoint_filename"]), "--report",
        str(run_dir / config["run_matrix"]["report_filename"]),
    ]


def _build_payload(
    *, config: Mapping[str, Any], experiment: Path, arm: str, seed: int,
    corpus_dir: Path, validation_manifest: Path, artifact_paths: Mapping[str, Path],
    checkpoint_paths: Mapping[str, Path], repo_root: Path,
) -> tuple[Path, dict[str, Any]]:
    relative = config["run_matrix"]["run_directory_pattern"].format(
        arm_id=arm, training_seed=seed
    )
    run_dir = (repo_root / relative).resolve()
    if not run_dir.is_relative_to(repo_root):
        raise AdmissionError("registered run directory escapes repository root")
    artifact_dirs = {path.resolve().parent for path in artifact_paths.values()}
    if len(artifact_dirs) != 1:
        raise AdmissionError("all relocated A1 artifacts must share one direct directory")
    output = run_dir / config["run_matrix"]["admission_manifest_filename"]
    payload = {
        "schema_version": ADMISSION_SCHEMA,
        "experiment_config_sha256": _sha256_file(_existing_file(experiment, field="experiment")),
        "experiment_semantic_sha256": config["config_sha256"],
        "arm_id": arm, "training_seed": seed,
        "latent_deliberation_steps": ARMS[arm][0],
        "expected_parameters": ARMS[arm][1], "capacity_class": ARMS[arm][2],
        "run_directory": str(run_dir),
        "checkpoint": str(run_dir / config["run_matrix"]["checkpoint_filename"]),
        "report": str(run_dir / config["run_matrix"]["report_filename"]),
        "registered_hashes": config["registration"],
        "train_argv": _train_argv(
            config, arm=arm, seed=seed, corpus_dir=corpus_dir,
            validation_manifest=validation_manifest, artifact_dir=next(iter(artifact_dirs)),
            checkpoint_init=_existing_file(checkpoint_paths[f"{arm}@{seed}"], field="checkpoint"),
            run_dir=run_dir,
        ),
    }
    return output, payload


def _prepare(
    *, experiment: Path, corpus_dir: Path, training_manifest: Path,
    validation_manifest: Path, artifact_paths: Mapping[str, Path],
    identity_report: Path, checkpoint_paths: Mapping[str, Path], source_root: Path,
    repo_root: Path,
) -> tuple[dict[str, Any], Path, Path, Path, Path]:
    config = _load_object(experiment, field="registered experiment")
    _validate_contract(config, registered=True)
    repo = _existing_dir(repo_root, field="repository root")
    corpus = _existing_dir(corpus_dir, field="A1 corpus")
    training = _existing_file(training_manifest, field="training manifest")
    validation = _existing_file(validation_manifest, field="validation manifest")
    identity = _existing_file(identity_report, field="identity report")
    _authenticate(
        config=config, corpus_dir=corpus, training_manifest=training,
        validation_manifest=validation, artifact_paths=artifact_paths,
        identity_report=identity, checkpoint_paths=checkpoint_paths, source_root=source_root,
    )
    return config, repo, corpus, validation, identity


def admit_run(
    *, experiment: Path, arm: str, training_seed: int, corpus_dir: Path,
    training_manifest: Path, validation_manifest: Path, artifact_paths: Mapping[str, Path],
    identity_report: Path, checkpoint_paths: Mapping[str, Path], source_root: Path,
    output: Path, repo_root: Path,
) -> dict[str, Any]:
    if arm not in ARMS or type(training_seed) is not int or training_seed not in SEEDS:
        raise AdmissionError("unregistered Transformer-think arm or seed")
    config, repo, corpus, validation, _identity = _prepare(
        experiment=experiment, corpus_dir=corpus_dir, training_manifest=training_manifest,
        validation_manifest=validation_manifest, artifact_paths=artifact_paths,
        identity_report=identity_report, checkpoint_paths=checkpoint_paths,
        source_root=source_root, repo_root=repo_root,
    )
    expected, payload = _build_payload(
        config=config, experiment=experiment, arm=arm, seed=training_seed,
        corpus_dir=corpus, validation_manifest=validation, artifact_paths=artifact_paths,
        checkpoint_paths=checkpoint_paths, repo_root=repo,
    )
    if output.resolve() != expected:
        raise AdmissionError(f"admission output must be exactly {expected}")
    _publish_exclusive(expected, payload)
    return payload


def admit_all(
    *, experiment: Path, corpus_dir: Path, training_manifest: Path,
    validation_manifest: Path, artifact_paths: Mapping[str, Path], identity_report: Path,
    checkpoint_paths: Mapping[str, Path], source_root: Path, repo_root: Path,
    run_keys: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    selected = list(RUN_KEYS if run_keys is None else run_keys)
    if not selected or len(set(selected)) != len(selected) or any(key not in RUN_KEYS for key in selected):
        raise AdmissionError("admit-all run subset is empty, duplicated, or unregistered")
    config, repo, corpus, validation, _identity = _prepare(
        experiment=experiment, corpus_dir=corpus_dir, training_manifest=training_manifest,
        validation_manifest=validation_manifest, artifact_paths=artifact_paths,
        identity_report=identity_report, checkpoint_paths=checkpoint_paths,
        source_root=source_root, repo_root=repo_root,
    )
    publications: list[tuple[Path, Mapping[str, Any]]] = []
    payloads: list[dict[str, Any]] = []
    for key in selected:
        arm, raw_seed = key.rsplit("@", 1)
        output, payload = _build_payload(
            config=config, experiment=experiment, arm=arm, seed=int(raw_seed),
            corpus_dir=corpus, validation_manifest=validation, artifact_paths=artifact_paths,
            checkpoint_paths=checkpoint_paths, repo_root=repo,
        )
        publications.append((output, payload))
        payloads.append(payload)
    _publish_many_exclusive(publications)
    return payloads


def _named_paths(values: Sequence[str], expected: set[str], field: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for raw in values:
        name, separator, path = raw.partition("=")
        if not separator or name not in expected or name in result or not path:
            raise AdmissionError(f"invalid or duplicate {field}: {raw!r}")
        result[name] = Path(path)
    if set(result) != expected:
        raise AdmissionError(f"{field} names must be exactly {sorted(expected)}")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("initialize")
    initialize.add_argument("--repo-root", type=Path, default=Path.cwd())
    initialize.add_argument("--source-checkpoint", required=True, type=Path)
    initialize.add_argument("--reuse-existing", action="store_true")
    for name in ("register", "admit", "admit-all"):
        sub = commands.add_parser(name)
        sub.add_argument("--repo-root", type=Path, default=Path.cwd())
        sub.add_argument("--corpus-dir", required=True, type=Path)
        sub.add_argument("--training-manifest", required=True, type=Path)
        sub.add_argument("--validation-manifest", required=True, type=Path)
        sub.add_argument("--a1-artifact", action="append", default=[], required=True)
        sub.add_argument("--identity-report", required=True, type=Path)
        sub.add_argument("--init-checkpoint", action="append", default=[], required=True)
        sub.add_argument("--source-root", required=True, type=Path)
        if name == "register":
            sub.add_argument("--template", required=True, type=Path)
            sub.add_argument("--output", required=True, type=Path)
        else:
            sub.add_argument("--experiment", required=True, type=Path)
        if name == "admit":
            sub.add_argument("--arm", required=True)
            sub.add_argument("--training-seed", required=True, type=int)
            sub.add_argument("--output", required=True, type=Path)
        if name == "admit-all":
            sub.add_argument("--run", action="append", default=[])
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "initialize":
        payload = create_initializations(
            repo_root=args.repo_root,
            source_checkpoint=args.source_checkpoint,
            reuse_existing=args.reuse_existing,
        )
    else:
        artifacts = _named_paths(args.a1_artifact, set(ARTIFACT_ROLES), "A1 artifact")
        checkpoints = _named_paths(args.init_checkpoint, set(RUN_KEYS), "checkpoint")
        common = dict(
            corpus_dir=args.corpus_dir, training_manifest=args.training_manifest,
            validation_manifest=args.validation_manifest, artifact_paths=artifacts,
            identity_report=args.identity_report, checkpoint_paths=checkpoints,
            source_root=args.source_root,
        )
        if args.command == "register":
            payload = register_experiment(template=args.template, output=args.output, **common)
        elif args.command == "admit":
            payload = admit_run(experiment=args.experiment, arm=args.arm,
                training_seed=args.training_seed, output=args.output,
                repo_root=args.repo_root, **common)
        else:
            rows = admit_all(experiment=args.experiment, repo_root=args.repo_root,
                run_keys=args.run or None, **common)
            payload = {"published_count": len(rows)}
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
