#!/usr/bin/env python3
"""Finalize and replay one sealed topology-residual-only learner run.

The launcher proves immutable inputs and one-shot authorization.  This finalizer
proves the realized output: successful retained systemd state, exact selected
TEMP report/progress/RNG semantics, fresh Adam with every topology state at the
completed step, and a model delta containing all and only the eight reviewed
topology-adapter tensors.  It never grants promotion authority.
"""

from __future__ import annotations

import argparse
import hashlib
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

from tools import a1_d6_gather_composition_completion as completion_base  # noqa: E402
from tools import a1_topology_only_composition_arm as arm  # noqa: E402


SCHEMA = "a1-topology-only-composition-completion-v1"
STATUS = "complete_nonpromotable"
COMPLETION_NAME = "diagnostic-completion.receipt.json"
EXPECTED_CHANGED_PARAMETERS = arm.EXPECTED_TOPOLOGY_PARAMETERS


class CompletionError(RuntimeError):
    """The topology-only run cannot be proven complete and isolated."""


def _digest(value: Any) -> str:
    return arm._digest(value)  # noqa: SLF001


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CompletionError(f"cannot load {label}: {error}") from error
    if not isinstance(value, dict):
        raise CompletionError(f"{label} must be a JSON object")
    return value


def _file_ref(path: Path) -> dict[str, Any]:
    try:
        return completion_base._file_ref(path)  # noqa: SLF001
    except completion_base.CompletionError as error:
        raise CompletionError(str(error)) from error


def _verify_ref(value: Any, *, label: str) -> Path:
    try:
        return arm._verify_ref(value, label=label)  # noqa: SLF001
    except arm.TopologyCompositionError as error:
        raise CompletionError(str(error)) from error


def verify_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve(strict=True)
    manifest = _load_json(manifest_path, label="topology manifest")
    historical_executor = _verify_ref(
        manifest.get("diagnostic_executor"), label="diagnostic executor"
    )
    try:
        verified = arm.verify(
            manifest_path,
            expected_executor=historical_executor,
            require_fresh_outputs=False,
        )
    except arm.TopologyCompositionError as error:
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
        raise CompletionError("topology submission receipt drift")
    unit = str(receipt["unit"])
    claim_path = _verify_ref(receipt.get("claim"), label="execution claim")
    if claim_path != (root / "diagnostic-execution.claim.json").resolve(strict=True):
        raise CompletionError("execution claim escaped topology output root")
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
        raise CompletionError("topology execution claim/systemd identity drift")
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
        raise CompletionError("topology execution status journal drift")
    return unit, {
        "claim": _file_ref(claim_path),
        "submission": _file_ref(receipt_path),
        "status": _file_ref(status_path),
    }


def _verify_unit_state(value: Mapping[str, Any]) -> dict[str, str]:
    expected = {
        "LoadState": "loaded",
        "ActiveState": "active",
        "SubState": "exited",
        "Result": "success",
        "ExecMainStatus": "0",
        "ExecMainCode": "1",
    }
    observed = {str(key): str(item) for key, item in value.items()}
    if observed != expected:
        raise CompletionError(f"topology systemd unit is not complete: {observed}")
    return observed


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
        raise CompletionError(f"cannot read topology systemd state: {error}") from error
    return _verify_unit_state(
        dict(row.split("=", 1) for row in raw.splitlines() if "=" in row)
    )


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _manifest_dose(verified: Mapping[str, Any]) -> dict[str, Any]:
    matched = verified.get("manifest", {}).get("matched_contract")
    if not isinstance(matched, Mapping):
        raise CompletionError("topology manifest lacks matched dose contract")
    try:
        return arm._dose_geometry(matched.get("optimizer_steps"))  # noqa: SLF001
    except arm.TopologyCompositionError as error:
        raise CompletionError(str(error)) from error


def _verify_report(
    verified: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = Path(verified["output_root"])
    manifest = verified["manifest"]
    dose = _manifest_dose(verified)
    checkpoint = _file_ref(root / "candidate.pt")
    report_ref = _file_ref(root / "train.report.json")
    report = _load_json(Path(report_ref["path"]), label="training report")
    upgrade_flags = manifest.get("function_preserving_upgrade", {}).get("flags", {})
    parent_gather = bool(
        manifest["selected_parent"]["architecture"]["action_target_gather"]
    )
    effective_gather = bool(upgrade_flags.get("action_target_gather", parent_gather))
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
        "max_steps": dose["optimizer_steps"],
        "steps_completed": dose["optimizer_steps"],
        "training_row_draws": dose["global_row_dose"],
        "base_training_row_draws": dose["global_row_dose"],
        "total_training_row_draws": dose["global_row_dose"],
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
        "trunk_lr_mult": arm.TRUNK_LR_MULT,
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
        "state_trunk": "transformer",
        "action_target_gather": effective_gather,
        **arm.REPORT_ARCHITECTURE_DELTA,
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
        raise CompletionError(f"topology report recipe/dose drift: {drift}")

    decisive = report.get("a1_decisive_training_semantics")
    if decisive != {
        "schema_version": "a1-decisive-training-semantics-v2",
        "decisive": False,
        "diagnostic_authority_present": True,
        "world_size": arm.WORLD_SIZE,
        "grad_accum_steps": 1,
        "gradient_accumulation_contract": "single_microbatch_exact",
        "symmetry_augmentation": True,
        "distributed_symmetry_contract": "per_rank_seedsequence_checkpoint_resume_v1",
        "advantage_policy_weighting": "none",
        "distributed_advantage_contract": "not_applicable",
    }:
        raise CompletionError("topology decisive/distributed symmetry semantics drift")

    component_ids = list(arm.gather_arm.gather.production_temp.COMPONENT_IDS)
    composite = report.get("memmap_composite")
    if not (
        report.get("stored_policy_component_temperatures")
        == arm.gather_arm.gather.production_temp.COMPONENT_TEMPERATURES
        and isinstance(composite, Mapping)
        and composite.get("component_ids") == component_ids
        and composite.get("policy_distillation_component_ids") == component_ids
        and composite.get("value_training_component_ids") == component_ids
    ):
        raise CompletionError("topology component/temperature scope drift")
    runtime = report.get("checkout_runtime_binding")
    trainer = Path(
        manifest["source_binding"]["files"]["tools/train_bc.py"]["path"]
    ).resolve(strict=True)
    if not (
        isinstance(runtime, Mapping)
        and Path(str(runtime.get("trainer", ""))).resolve(strict=True) == trainer
        and runtime.get("trainer_sha256") == arm._file_ref(trainer)["sha256"]  # noqa: SLF001
    ):
        raise CompletionError("topology report current trainer binding drift")
    surface = report.get("training_information_surface", {}).get(
        "required_trainable_surface"
    )
    if surface != {
        "prefixes": list(arm.TRAINABLE_PREFIXES),
        "parameter_tensors": len(EXPECTED_CHANGED_PARAMETERS),
        "parameters": arm.EXPECTED_TOPOLOGY_PARAMETER_COUNT,
        "parameters_by_prefix": arm.EXPECTED_PARAMETER_COUNTS,
    }:
        raise CompletionError("topology trainable surface is not exact 8/823040")
    metrics = report.get("metrics")
    epoch = metrics[0] if isinstance(metrics, list) and len(metrics) == 1 else None
    matched = (
        epoch.get("validation_objective_matched")
        if isinstance(epoch, Mapping)
        else None
    )
    components = matched.get("components") if isinstance(matched, Mapping) else None
    observability = (
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
        and epoch.get("samples") == dose["global_row_dose"]
        and isinstance(epoch.get("policy_total_active_rows"), int)
        and 0 < epoch["policy_total_active_rows"] < dose["global_row_dose"]
        and isinstance(observability, Mapping)
        and observability.get("observed_steps") == dose["optimizer_steps"]
        and observability.get("zero_objective_steps_skipped") == 0
        and _finite_number(report.get("elapsed_sec"))
        and float(report["elapsed_sec"]) > 0.0
    ):
        raise CompletionError("topology objective/optimizer telemetry drift")
    summary = {
        "elapsed_sec": float(report["elapsed_sec"]),
        "total_row_dose": dose["global_row_dose"],
        "policy_active_rows": int(epoch["policy_total_active_rows"]),
        "policy_active_fraction": (
            float(epoch["policy_total_active_rows"]) / dose["global_row_dose"]
        ),
        "objective_matched_teacher_gap_closure": float(
            matched["metrics"]["active_policy_teacher_gap_closure"]
        ),
        "component_teacher_gap_closure": {
            component_id: float(
                components[component_id]["metrics"]["active_policy_teacher_gap_closure"]
            )
            for component_id in component_ids
        },
        "optimizer_observability": dict(observability),
        "trainable_surface": dict(surface),
    }
    return checkpoint, report_ref, summary


def _verify_progress(
    root: Path, *, checkpoint: Mapping[str, Any], optimizer_steps: int
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    try:
        return completion_base._verify_progress(  # noqa: SLF001
            root,
            checkpoint=checkpoint,
            optimizer_steps=optimizer_steps,
        )
    except completion_base.CompletionError as error:
        raise CompletionError(str(error).replace("D6+gather", "topology")) from error


def _optimizer_step(raw: Any) -> int:
    try:
        return int(raw.item()) if hasattr(raw, "item") else int(raw)
    except (TypeError, ValueError, RuntimeError) as error:
        raise CompletionError("topology optimizer state lacks a scalar step") from error


def _verify_optimizer_groups(path: Path, *, optimizer_steps: int) -> dict[str, Any]:
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ModuleNotFoundError) as error:
        raise CompletionError(f"cannot load topology optimizer: {error}") from error
    optimizer = payload.get("optimizer") if isinstance(payload, Mapping) else None
    groups = optimizer.get("param_groups") if isinstance(optimizer, Mapping) else None
    state = optimizer.get("state") if isinstance(optimizer, Mapping) else None
    if not (
        payload.get("format") == "plain"
        and isinstance(groups, list)
        and len(groups) == 2
        and isinstance(state, Mapping)
    ):
        raise CompletionError("topology optimizer envelope/group count drift")
    base_group, trunk_group = groups
    topology_parameters = (
        trunk_group.get("params") if isinstance(trunk_group, Mapping) else None
    )
    if not (
        isinstance(base_group, Mapping)
        and isinstance(trunk_group, Mapping)
        and base_group.get("lr") == 3e-5
        and base_group.get("base_lr") == 3e-5
        and base_group.get("params") == []
        and trunk_group.get("lr") == 1.2e-4
        and trunk_group.get("base_lr") == 1.2e-4
        and isinstance(topology_parameters, list)
        and len(topology_parameters) == len(EXPECTED_CHANGED_PARAMETERS)
        and set(state) == set(topology_parameters)
    ):
        raise CompletionError(
            "topology optimizer does not isolate eight LR=1.2e-4 tensors"
        )
    observed_steps = []
    for parameter_id in topology_parameters:
        parameter_state = state.get(parameter_id)
        if not isinstance(parameter_state, Mapping):
            raise CompletionError("topology optimizer parameter state is malformed")
        observed_steps.append(_optimizer_step(parameter_state.get("step")))
        for moment in ("exp_avg", "exp_avg_sq"):
            tensor = parameter_state.get(moment)
            if tensor is None or not bool(torch.isfinite(tensor).all()):
                raise CompletionError(
                    f"topology optimizer has missing/non-finite {moment}"
                )
    if observed_steps != [optimizer_steps] * len(topology_parameters):
        raise CompletionError(
            "topology optimizer state step does not match completed dose: "
            f"expected={optimizer_steps} observed={observed_steps}"
        )
    return {
        "format": "plain",
        "base_group_parameter_tensors": 0,
        "base_group_lr": 3e-5,
        "trunk_group_parameter_tensors": len(topology_parameters),
        "trunk_group_lr": 1.2e-4,
        "optimizer_state_tensors": len(state),
        "optimizer_state_step": optimizer_steps,
    }


def _tensor_digest(tensor: Any) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    metadata = json.dumps(
        {"dtype": str(contiguous.dtype), "shape": list(contiguous.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return (
        "sha256:"
        + hashlib.sha256(metadata + b"\0" + contiguous.numpy().tobytes()).hexdigest()
    )


def _verify_topology_only_delta(initializer: Path, candidate: Path) -> dict[str, Any]:
    try:
        import torch

        before = torch.load(initializer, map_location="cpu", weights_only=False)
        after = torch.load(candidate, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ModuleNotFoundError) as error:
        raise CompletionError(f"cannot load topology checkpoints: {error}") from error
    before_model = before.get("model") if isinstance(before, Mapping) else None
    after_model = after.get("model") if isinstance(after, Mapping) else None
    if not isinstance(before_model, Mapping) or not isinstance(after_model, Mapping):
        raise CompletionError("topology checkpoint model state is malformed")
    if set(before_model) != set(after_model):
        raise CompletionError("topology candidate parameter keys drifted")
    changed = tuple(
        sorted(
            name
            for name in before_model
            if not arm.architecture_upgrade._tensor_equal_exact(  # noqa: SLF001
                before_model[name], after_model[name]
            )
        )
    )
    if changed != EXPECTED_CHANGED_PARAMETERS:
        raise CompletionError(
            f"candidate changed tensors outside/excluding topology adapter: {changed}"
        )
    changed_parameters = sum(int(after_model[name].numel()) for name in changed)
    if changed_parameters != arm.EXPECTED_TOPOLOGY_PARAMETER_COUNT:
        raise CompletionError(
            "topology candidate changed parameter count drift: "
            f"{changed_parameters} != {arm.EXPECTED_TOPOLOGY_PARAMETER_COUNT}"
        )
    if any(not bool(torch.isfinite(after_model[name]).all()) for name in changed):
        raise CompletionError("topology candidate contains non-finite adapter weights")
    before_config = arm.gather_arm.gather._effective_config(before.get("config"))  # noqa: SLF001
    after_config = arm.gather_arm.gather._effective_config(after.get("config"))  # noqa: SLF001
    if before_config != after_config:
        raise CompletionError("topology candidate effective config drifted")
    evidence = {
        "inherited_parameter_tensors": len(before_model) - len(changed),
        "inherited_parameters_bit_identical": True,
        "changed_parameter_tensors": list(changed),
        "changed_parameter_count": changed_parameters,
        "changed_tensor_sha256": {
            name: _tensor_digest(after_model[name]) for name in changed
        },
        "initializer_tensor_sha256": {
            name: _tensor_digest(before_model[name]) for name in changed
        },
    }
    evidence["model_delta_sha256"] = _digest(evidence)
    return evidence


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
    dose = _manifest_dose(verified)
    root = Path(verified["output_root"])
    unit, submission = _verify_submission(verified)
    checkpoint, report, report_summary = _verify_report(verified)
    if checkpoint["sha256"] != expected_checkpoint_sha256:
        raise CompletionError(
            "topology checkpoint differs from explicitly expected completed bytes"
        )
    progress, optimizer, rng_summary = _verify_progress(
        root,
        checkpoint=checkpoint,
        optimizer_steps=dose["optimizer_steps"],
    )
    optimizer_groups = _verify_optimizer_groups(
        Path(optimizer["path"]), optimizer_steps=dose["optimizer_steps"]
    )
    model_delta = _verify_topology_only_delta(
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
        "parent_selection": verified["manifest"]["parent_selection"],
        "selected_parent": verified["manifest"]["selected_parent"],
        "function_preserving_upgrade_receipt": verified["manifest"][
            "function_preserving_upgrade_receipt"
        ],
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
        raise CompletionError(f"topology completion already exists: {path}") from error
    return payload


def verify_completion(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    receipt = _load_json(path, label="topology completion receipt")
    unhashed = dict(receipt)
    stated = unhashed.pop("receipt_sha256", None)
    if not (
        receipt.get("schema_version") == SCHEMA
        and receipt.get("status") == STATUS
        and receipt.get("diagnostic_only") is True
        and receipt.get("promotion_eligible") is False
        and stated == _digest(unhashed)
        and receipt.get("completion_finalizer") == _file_ref(Path(__file__))
    ):
        raise CompletionError(
            "topology completion schema/status/finalizer/digest drift"
        )
    replay = build_completion(
        Path(receipt["manifest"]["path"]),
        expected_checkpoint_sha256=str(receipt["expected_checkpoint_sha256"]),
        unit_state=receipt["unit_state"],
        created_at_unix_ns=int(receipt["created_at_unix_ns"]),
    )
    if replay != receipt:
        raise CompletionError("topology completion replay differs from receipt")
    if path != Path(replay["checkpoint"]["path"]).parent / COMPLETION_NAME:
        raise CompletionError("topology completion escaped output root")
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
