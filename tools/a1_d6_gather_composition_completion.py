#!/usr/bin/env python3
"""Finalize and replay one completed D6 -> target-gather composition run.

The launcher proves authorization and immutable inputs.  This transaction proves
the realized run: successful systemd exit, exact 1,024-update/524,288-row
commissioning geometry, current rank-distinct D6 RNG state, fresh Adam, exact
trainable surface, policy-active dose, and a checkpoint delta confined to the
four gather tensors.  It remains diagnostic and non-promotable.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import a1_d6_gather_composition_arm as arm  # noqa: E402
from tools import a1_topology_gather_completion as base  # noqa: E402


SCHEMA = "a1-d6-gather-composition-completion-v1"
STATUS = "complete_nonpromotable"
COMPLETION_NAME = "diagnostic-completion.receipt.json"
EXPECTED_CHANGED_PARAMETERS = tuple(sorted(arm.gather.EXPECTED_NEW_PARAMETERS))


class CompletionError(RuntimeError):
    """The D6+gather run cannot be proven complete and isolated."""


def _digest(value: Any) -> str:
    return arm.gather.corrected._digest(value)  # noqa: SLF001


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CompletionError(f"cannot load {label}: {error}") from error
    if not isinstance(payload, dict):
        raise CompletionError(f"{label} must be a JSON object")
    return payload


def _file_ref(path: Path) -> dict[str, Any]:
    try:
        return base._file_ref(path)  # noqa: SLF001
    except base.CompletionError as error:
        raise CompletionError(str(error)) from error


def _compact_ref(path: Path) -> dict[str, str]:
    ref = _file_ref(path)
    return {"path": ref["path"], "sha256": ref["sha256"]}


def _verify_ref(value: Any, *, label: str) -> Path:
    try:
        return arm._verify_ref(value, label=label)  # noqa: SLF001
    except arm.CompositionArmError as error:
        raise CompletionError(str(error)) from error


def verify_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve(strict=True)
    manifest = _load_json(manifest_path, label="composition manifest")
    historical_executor = _verify_ref(
        manifest.get("diagnostic_executor"), label="diagnostic executor"
    )
    try:
        verified = arm.verify(
            manifest_path,
            expected_executor=historical_executor,
            require_fresh_outputs=False,
        )
    except arm.CompositionArmError as error:
        raise CompletionError(str(error)) from error
    if verified["output_root"] != manifest_path.parent.resolve():
        raise CompletionError("manifest/output-root layout is not canonical")
    finalizer = _verify_ref(
        verified["manifest"].get("completion_finalizer"),
        label="completion finalizer",
    )
    if finalizer != Path(__file__).resolve():
        raise CompletionError("manifest authorizes a different completion finalizer")
    return verified


def _systemd_command(verified: Mapping[str, Any], *, unit: str) -> list[str]:
    return arm.executor_base._systemd_command(verified, unit=unit)  # noqa: SLF001


def _verify_submission(verified: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    root = Path(verified["output_root"])
    manifest_ref = verified["manifest_ref"]
    receipt_path = root / "diagnostic-execution.receipt.json"
    receipt = _load_json(receipt_path, label="submission receipt")
    unhashed = dict(receipt)
    stated = unhashed.pop("receipt_sha256", None)
    if not (
        set(receipt)
        == {
            "schema_version",
            "diagnostic_only",
            "promotion_eligible",
            "created_at_unix_ns",
            "manifest",
            "claim",
            "unit",
            "command_sha256",
            "systemd_command_sha256",
            "systemd_stdout",
            "receipt_sha256",
        }
        and stated == _digest(unhashed)
        and receipt.get("schema_version") == arm.RECEIPT_SCHEMA
        and receipt.get("diagnostic_only") is True
        and receipt.get("promotion_eligible") is False
        and receipt.get("manifest") == manifest_ref
        and receipt.get("command_sha256") == verified["manifest"]["command_sha256"]
        and isinstance(receipt.get("unit"), str)
    ):
        raise CompletionError("D6+gather submission receipt drift")
    unit = str(receipt["unit"])
    claim_path = _verify_ref(receipt.get("claim"), label="execution claim")
    if claim_path != (root / "diagnostic-execution.claim.json").resolve(strict=True):
        raise CompletionError("execution claim escaped output root")
    claim = _load_json(claim_path, label="execution claim")
    claim_unhashed = dict(claim)
    claim_stated = claim_unhashed.pop("claim_sha256", None)
    if not (
        set(claim)
        == {"schema_version", "created_at_unix_ns", "manifest", "unit", "claim_sha256"}
        and claim_stated == _digest(claim_unhashed)
        and claim.get("schema_version") == arm.CLAIM_SCHEMA
        and claim.get("manifest") == manifest_ref
        and claim.get("unit") == unit
        and receipt.get("systemd_command_sha256")
        == _digest(_systemd_command(verified, unit=unit))
    ):
        raise CompletionError("D6+gather execution claim/systemd identity drift")
    status_path = root / "diagnostic-execution.status.jsonl"
    try:
        events = [
            json.loads(row) for row in status_path.read_text().splitlines() if row
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise CompletionError(
            f"cannot load execution status journal: {error}"
        ) from error
    if not (
        len(events) == 2
        and events[0].get("schema_version") == arm.STATUS_SCHEMA
        and events[0].get("event") == "authorized"
        and events[0].get("claim_sha256") == claim["claim_sha256"]
        and events[1].get("schema_version") == arm.STATUS_SCHEMA
        and events[1].get("event") == "submitted"
        and events[1].get("receipt_sha256") == receipt["receipt_sha256"]
        and events[1].get("unit") == unit
    ):
        raise CompletionError("D6+gather execution status journal drift")
    return unit, {
        "claim": _file_ref(claim_path),
        "submission": _file_ref(receipt_path),
        "status": _file_ref(status_path),
    }


def _verify_unit_state(value: Mapping[str, Any]) -> dict[str, str]:
    try:
        return base._verify_unit_state(value)  # noqa: SLF001
    except base.CompletionError as error:
        raise CompletionError(str(error).replace("gather", "D6+gather")) from error


def _read_live_unit_state(
    unit: str,
    *,
    state_reader: Callable[..., str] = subprocess.check_output,
) -> dict[str, str]:
    try:
        raw = state_reader(
            (
                "systemctl",
                "show",
                unit,
                "--property=LoadState,ActiveState,SubState,Result,ExecMainStatus,ExecMainCode",
            ),
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise CompletionError(
            f"cannot read D6+gather systemd state: {error}"
        ) from error
    return _verify_unit_state(
        dict(row.split("=", 1) for row in raw.splitlines() if "=" in row)
    )


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _verify_report(
    verified: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = Path(verified["output_root"])
    manifest = verified["manifest"]
    checkpoint = _file_ref(root / "candidate.pt")
    report_ref = _file_ref(root / "train.report.json")
    report = _load_json(Path(report_ref["path"]), label="training report")
    exact = {
        "init_checkpoint": manifest["initialization_treatment"]["path"],
        "init_checkpoint_sha256": manifest["initialization_treatment"]["sha256"],
        "checkpoint": checkpoint["path"],
        "data": manifest["descriptor"]["path"],
        "input_validation_game_sentinel_manifest": manifest["validation_sentinel"][
            "path"
        ],
        "world_size": arm.WORLD_SIZE,
        "batch_size": arm.LOCAL_BATCH_SIZE,
        "effective_global_batch_size": arm.GLOBAL_BATCH_SIZE,
        "max_steps": arm.OPTIMIZER_STEPS,
        "steps_completed": arm.OPTIMIZER_STEPS,
        "training_row_draws": arm.GLOBAL_ROW_DOSE,
        "base_training_row_draws": arm.GLOBAL_ROW_DOSE,
        "total_training_row_draws": arm.GLOBAL_ROW_DOSE,
        "optimizer": "adam",
        "resume_optimizer": False,
        "optimizer_restored": False,
        "fused_optimizer": False,
        "amp": "none",
        "float32_matmul_precision": "highest",
        "effective_float32_matmul_precision": "highest",
        "lr": 3e-5,
        "lr_warmup_steps": 100,
        "lr_schedule": "flat",
        "weight_decay": 0.0,
        "value_lr_mult": arm.VALUE_LR_MULT,
        "action_module_lr_mult": arm.ACTION_MODULE_LR_MULT,
        "policy_loss_weight": 1.0,
        "soft_target_source": "policy",
        "soft_target_weight": 0.9,
        "soft_target_temperature": 0.7,
        "value_loss_weight": 0.25,
        "value_target_lambda": 1.0,
        "q_loss_weight": 0.0,
        "policy_kl_anchor_weight": 0.0,
        "forced_action_weight": 0.0,
        "forced_row_value_weight": 1.0,
        "winner_sample_weight": 1.0,
        "loser_sample_weight": 1.0,
        "freeze_modules": arm.FREEZE_MODULES,
        "require_only_trainable_prefixes": arm.TRAINABLE_PREFIX,
        "action_target_gather": True,
        "symmetry_augment": True,
        "symmetry_augment_events": True,
        "ddp_find_unused_parameters": False,
        "ddp_shard_data": False,
        "training_rng_rank_offset": True,
        "diagnostic_only": True,
        "promotion_eligible": False,
    }
    drift = {
        key: {"expected": expected, "actual": report.get(key)}
        for key, expected in exact.items()
        if report.get(key) != expected
    }
    if drift:
        raise CompletionError(f"D6+gather report recipe/dose drift: {drift}")
    decisive = report.get("a1_decisive_training_semantics")
    if decisive != {
        "schema_version": "a1-decisive-training-semantics-v2",
        "decisive": False,
        "diagnostic_authority_present": True,
        "world_size": arm.WORLD_SIZE,
        "grad_accum_steps": 1,
        "gradient_accumulation_contract": "single_microbatch_exact",
        "symmetry_augmentation": True,
        "distributed_symmetry_contract": ("per_rank_seedsequence_checkpoint_resume_v1"),
        "advantage_policy_weighting": "none",
        "distributed_advantage_contract": "not_applicable",
    }:
        raise CompletionError("D6+gather decisive/distributed symmetry semantics drift")
    component_ids = list(arm.gather.production_temp.COMPONENT_IDS)
    composite = report.get("memmap_composite")
    if not (
        report.get("stored_policy_component_temperatures")
        == arm.gather.production_temp.COMPONENT_TEMPERATURES
        and isinstance(composite, Mapping)
        and composite.get("component_ids") == component_ids
        and composite.get("policy_distillation_component_ids") == component_ids
        and composite.get("value_training_component_ids") == component_ids
    ):
        raise CompletionError("D6+gather component/temperature scope drift")
    runtime = report.get("checkout_runtime_binding")
    current_trainer = Path(
        manifest["source_binding"]["files"]["tools/train_bc.py"]["path"]
    ).resolve(strict=True)
    if not (
        isinstance(runtime, Mapping)
        and Path(str(runtime.get("trainer", ""))).resolve(strict=True)
        == current_trainer
        and runtime.get("trainer_sha256")
        == arm.gather.corrected._file_ref(current_trainer)["sha256"]  # noqa: SLF001
    ):
        raise CompletionError("D6+gather report current trainer binding drift")
    surface = report.get("training_information_surface", {}).get(
        "required_trainable_surface"
    )
    if not (
        isinstance(surface, Mapping)
        and surface.get("prefixes") == [arm.TRAINABLE_PREFIX]
        and surface.get("parameter_tensors") == len(EXPECTED_CHANGED_PARAMETERS)
        and surface.get("parameters_by_prefix", {}).get(arm.TRAINABLE_PREFIX)
        == surface.get("parameters")
        and isinstance(surface.get("parameters"), int)
        and surface["parameters"] > 0
    ):
        raise CompletionError("D6+gather trainable surface drift")
    metrics = report.get("metrics")
    epoch = metrics[0] if isinstance(metrics, list) and len(metrics) == 1 else None
    matched = (
        epoch.get("validation_objective_matched")
        if isinstance(epoch, Mapping)
        else None
    )
    components = matched.get("components") if isinstance(matched, Mapping) else None
    optimizer_observability = (
        epoch.get("optimizer_observability") if isinstance(epoch, Mapping) else None
    )
    if not (
        isinstance(matched, Mapping)
        and matched.get("schema_version") == "composite-validation-measure-v2"
        and matched.get("objective_matched") is True
        and isinstance(components, Mapping)
        and set(components) == set(component_ids)
        and _finite_number(
            matched.get("metrics", {}).get("active_policy_teacher_gap_closure")
        )
        and epoch.get("samples") == arm.GLOBAL_ROW_DOSE
        and isinstance(epoch.get("policy_total_active_rows"), int)
        and 0 < epoch["policy_total_active_rows"] < arm.GLOBAL_ROW_DOSE
        and isinstance(optimizer_observability, Mapping)
        and optimizer_observability.get("observed_steps") == arm.OPTIMIZER_STEPS
        and optimizer_observability.get("zero_objective_steps_skipped") == 0
        and _finite_number(report.get("elapsed_sec"))
        and float(report["elapsed_sec"]) > 0.0
    ):
        raise CompletionError("D6+gather objective/optimizer telemetry drift")
    summary = {
        "elapsed_sec": float(report["elapsed_sec"]),
        "total_row_dose": arm.GLOBAL_ROW_DOSE,
        "policy_active_rows": int(epoch["policy_total_active_rows"]),
        "policy_active_fraction": float(epoch["policy_total_active_rows"])
        / arm.GLOBAL_ROW_DOSE,
        "effective_trainable_objective": manifest["effective_trainable_objective"],
        "objective_matched_teacher_gap_closure": float(
            matched["metrics"]["active_policy_teacher_gap_closure"]
        ),
        "component_teacher_gap_closure": {
            component_id: float(
                components[component_id]["metrics"]["active_policy_teacher_gap_closure"]
            )
            for component_id in component_ids
        },
        "optimizer_observability": dict(optimizer_observability),
        "trainable_surface": dict(surface),
    }
    return checkpoint, report_ref, summary


def _resolve_progress_output(
    progress_path: Path, value: Any, *, label: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise CompletionError(f"{label} reference is malformed")
    lexical = Path(str(value["path"]))
    path = lexical if lexical.is_absolute() else progress_path.parent / lexical
    ref = _file_ref(path)
    if ref["sha256"] != value["sha256"]:
        raise CompletionError(f"{label} bytes drift")
    return ref


def _verify_progress(
    root: Path, *, checkpoint: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    progress_path = root / "candidate.pt.training-progress.json"
    progress = _load_json(progress_path, label="training progress")
    unhashed = dict(progress)
    stated = unhashed.pop("progress_sha256", None)
    if stated != _digest(unhashed):
        raise CompletionError("D6+gather progress semantic digest drift")
    progress_checkpoint = _resolve_progress_output(
        progress_path, progress.get("checkpoint"), label="checkpoint"
    )
    optimizer = _resolve_progress_output(
        progress_path, progress.get("optimizer"), label="optimizer"
    )
    torch_states = progress.get("rank_torch_rng_states")
    numpy_states = progress.get("rank_numpy_rng_states")
    symmetry = progress.get("symmetry_rng_state")
    symmetry_states = (
        symmetry.get("rank_states") if isinstance(symmetry, Mapping) else None
    )
    recipe = progress.get("recipe_identity")
    torch_ranks = (
        sorted(row.get("rank") for row in torch_states if isinstance(row, Mapping))
        if isinstance(torch_states, list)
        else []
    )
    numpy_state_digests = (
        [_digest(state) for state in numpy_states]
        if isinstance(numpy_states, list)
        and all(isinstance(state, Mapping) for state in numpy_states)
        else []
    )
    symmetry_state_digests = (
        [_digest(state) for state in symmetry_states]
        if isinstance(symmetry_states, list)
        and all(isinstance(state, Mapping) for state in symmetry_states)
        else []
    )
    if not (
        progress.get("schema_version") == "train-bc-progress-v1"
        and progress.get("status") == "complete"
        and progress.get("optimizer_step") == arm.OPTIMIZER_STEPS
        and progress.get("completed_epochs") == 1
        and isinstance(recipe, Mapping)
        and recipe.get("schema_version") == "train-bc-resume-recipe-v1"
        and recipe.get("world_size") == arm.WORLD_SIZE
        and recipe.get("grad_accum_steps") == 1
        and recipe.get("ddp_shard_data") is False
        and isinstance(torch_states, list)
        and len(torch_states) == arm.WORLD_SIZE
        and torch_ranks == list(range(arm.WORLD_SIZE))
        and isinstance(numpy_states, list)
        and len(numpy_states) == arm.WORLD_SIZE
        and len(set(numpy_state_digests)) == arm.WORLD_SIZE
        and isinstance(symmetry, Mapping)
        and symmetry.get("schema_version") == "train-bc-rank-symmetry-rng-v1"
        and symmetry.get("world_size") == arm.WORLD_SIZE
        and isinstance(symmetry_states, list)
        and len(symmetry_states) == arm.WORLD_SIZE
        and all(isinstance(state, Mapping) for state in symmetry_states)
        and len(set(symmetry_state_digests)) == arm.WORLD_SIZE
        and progress_checkpoint["path"] == checkpoint["path"]
        and progress_checkpoint["sha256"] == checkpoint["sha256"]
        and optimizer["path"]
        == str((root / "candidate.pt.optimizer.pt").resolve(strict=True))
    ):
        raise CompletionError("D6+gather progress/RNG/optimizer dose drift")
    rng_summary = {
        "rank_torch_rng_states": len(torch_states),
        "rank_torch_rng_set": torch_ranks,
        "rank_numpy_rng_states": len(numpy_states),
        "rank_numpy_state_digests_unique": True,
        "symmetry_rng_schema": symmetry["schema_version"],
        "rank_symmetry_rng_states": len(symmetry_states),
        "rank_symmetry_state_digests_unique": True,
        "world_size": arm.WORLD_SIZE,
    }
    return _file_ref(progress_path), optimizer, rng_summary


def _verify_adapter_only_delta(initializer: Path, candidate: Path) -> dict[str, Any]:
    try:
        return base._verify_adapter_only_delta(initializer, candidate)  # noqa: SLF001
    except base.CompletionError as error:
        raise CompletionError(str(error)) from error


def _verify_optimizer_groups(path: Path) -> dict[str, Any]:
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ModuleNotFoundError) as error:
        raise CompletionError(f"cannot load D6+gather optimizer: {error}") from error
    optimizer = payload.get("optimizer") if isinstance(payload, Mapping) else None
    groups = optimizer.get("param_groups") if isinstance(optimizer, Mapping) else None
    state = optimizer.get("state") if isinstance(optimizer, Mapping) else None
    if not (
        payload.get("format") == "plain"
        and isinstance(groups, list)
        and len(groups) == 2
        and isinstance(state, Mapping)
    ):
        raise CompletionError("D6+gather optimizer envelope/group count drift")
    base_group, action_group = groups
    action_parameters = action_group.get("params")
    if not (
        isinstance(base_group, Mapping)
        and isinstance(action_group, Mapping)
        and base_group.get("lr") == 3e-5
        and base_group.get("base_lr") == 3e-5
        and base_group.get("params") == []
        and action_group.get("lr") == 1.2e-4
        and action_group.get("base_lr") == 1.2e-4
        and isinstance(action_parameters, list)
        and len(action_parameters) == len(EXPECTED_CHANGED_PARAMETERS)
        and set(state) == set(action_parameters)
    ):
        raise CompletionError(
            "D6+gather optimizer does not isolate four LR=1.2e-4 tensors"
        )
    return {
        "format": "plain",
        "base_group_parameter_tensors": 0,
        "base_group_lr": 3e-5,
        "action_group_parameter_tensors": len(action_parameters),
        "action_group_lr": 1.2e-4,
        "optimizer_state_tensors": len(state),
    }


def _required_artifacts(root: Path) -> dict[str, dict[str, Any]]:
    names = (
        "candidate.pt",
        "candidate.pt.optimizer.pt",
        "candidate.pt.training-progress.json",
        "diagnostic-execution.claim.json",
        "diagnostic-execution.receipt.json",
        "diagnostic-execution.status.jsonl",
        "stderr.log",
        "stdout.log",
        "train.report.json",
        "train.report.validation_seeds.json",
    )
    return {name: _file_ref(root / name) for name in names}


def build_completion(
    manifest_path: Path,
    *,
    expected_checkpoint_sha256: str,
    unit_state: Mapping[str, Any],
    created_at_unix_ns: int,
) -> dict[str, Any]:
    verified = verify_manifest(manifest_path)
    root = Path(verified["output_root"])
    unit, submission = _verify_submission(verified)
    checkpoint, report, report_summary = _verify_report(verified)
    if checkpoint["sha256"] != expected_checkpoint_sha256:
        raise CompletionError(
            "D6+gather checkpoint differs from explicitly expected completed bytes"
        )
    progress, optimizer, rng_summary = _verify_progress(root, checkpoint=checkpoint)
    optimizer_groups = _verify_optimizer_groups(Path(optimizer["path"]))
    model_delta = _verify_adapter_only_delta(
        Path(verified["manifest"]["initialization_treatment"]["path"]),
        Path(checkpoint["path"]),
    )
    state = _verify_unit_state(unit_state)
    completion = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "created_at_unix_ns": int(created_at_unix_ns),
        "manifest": verified["manifest_ref"],
        "completion_finalizer": _file_ref(Path(__file__)),
        "expected_checkpoint_sha256": expected_checkpoint_sha256,
        "checkpoint": checkpoint,
        "report": report,
        "progress": progress,
        "optimizer": optimizer,
        "optimizer_groups": optimizer_groups,
        "submission": submission,
        "unit": unit,
        "unit_state": state,
        "model_delta": model_delta,
        "verified_recipe": verified["manifest"]["matched_contract"],
        "optimizer_geometry": verified["manifest"]["optimizer_geometry_contract"],
        "rng_summary": rng_summary,
        "report_summary": report_summary,
        "artifacts": _required_artifacts(root),
    }
    completion["receipt_sha256"] = _digest(completion)
    return completion


def finalize(
    manifest_path: Path,
    *,
    expected_checkpoint_sha256: str,
    state_reader: Callable[..., str] = subprocess.check_output,
) -> dict[str, Any]:
    verified = verify_manifest(manifest_path)
    unit, _ = _verify_submission(verified)
    state = _read_live_unit_state(unit, state_reader=state_reader)
    payload = build_completion(
        manifest_path,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        unit_state=state,
        created_at_unix_ns=time.time_ns(),
    )
    path = Path(verified["output_root"]) / COMPLETION_NAME
    try:
        arm.executor_base._write_exclusive(path, payload)  # noqa: SLF001
    except FileExistsError as error:
        raise CompletionError(f"D6+gather completion already exists: {path}") from error
    return payload


def verify_completion(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    receipt = _load_json(path, label="D6+gather completion receipt")
    unhashed = dict(receipt)
    stated = unhashed.pop("receipt_sha256", None)
    if not (
        receipt.get("schema_version") == SCHEMA
        and receipt.get("status") == STATUS
        and receipt.get("diagnostic_only") is True
        and receipt.get("promotion_eligible") is False
        and stated == _digest(unhashed)
    ):
        raise CompletionError("D6+gather completion schema/status/digest drift")
    if receipt.get("completion_finalizer") != _file_ref(Path(__file__)):
        raise CompletionError("D6+gather completion finalizer bytes drift")
    replay = build_completion(
        Path(receipt["manifest"]["path"]),
        expected_checkpoint_sha256=str(receipt["expected_checkpoint_sha256"]),
        unit_state=receipt["unit_state"],
        created_at_unix_ns=int(receipt["created_at_unix_ns"]),
    )
    if replay != receipt:
        raise CompletionError("D6+gather completion replay differs from receipt")
    if path != Path(replay["checkpoint"]["path"]).parent / COMPLETION_NAME:
        raise CompletionError("D6+gather completion receipt escaped output root")
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    done = actions.add_parser("finalize")
    done.add_argument("--manifest", required=True, type=Path)
    done.add_argument("--expected-checkpoint-sha256", required=True)
    replay = actions.add_parser("verify")
    replay.add_argument("--receipt", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.action == "finalize":
            value = finalize(
                args.manifest,
                expected_checkpoint_sha256=args.expected_checkpoint_sha256,
            )
        else:
            value = verify_completion(args.receipt)
    except (CompletionError, OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
