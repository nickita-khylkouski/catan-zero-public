#!/usr/bin/env python3
"""Finalize and replay a completed selected-dose topology-gather diagnostic.

The gather launcher writes its claim and submission receipt before systemd
starts training.  Those files authorize one run; they do not prove that the
run completed, consumed the selected row dose, or changed only the four gather
parameters.  This post-hoc transaction binds that evidence without granting
promotion authority or launching any compute.
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

from tools import a1_topology_gather_arm as prepare  # noqa: E402
from tools import a1_topology_gather_arm_execute as launch  # noqa: E402


SCHEMA = "a1-selected-dose-topology-gather-completion-v2"
STATUS = "complete_nonpromotable"
COMPLETION_NAME = "diagnostic-completion.receipt.json"
EXPECTED_CHANGED_PARAMETERS = tuple(sorted(prepare.EXPECTED_NEW_PARAMETERS))


class CompletionError(RuntimeError):
    """The selected gather run cannot be proven complete and isolated."""


def _digest(value: Any) -> str:
    return prepare.corrected._digest(value)  # noqa: SLF001


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CompletionError(f"cannot load {label}: {error}") from error
    if not isinstance(value, dict):
        raise CompletionError(f"{label} must be a JSON object")
    return value


def _file_ref(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    if not path.is_file():
        raise CompletionError(f"artifact is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "sha256": "sha256:" + digest.hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def _compact_ref(path: Path) -> dict[str, str]:
    ref = _file_ref(path)
    return {"path": ref["path"], "sha256": ref["sha256"]}


def _verify_ref(value: Any, *, label: str) -> Path:
    try:
        return launch._verify_ref(value, label=label)  # noqa: SLF001
    except launch.ExecutionError as error:
        raise CompletionError(str(error)) from error


def verify_manifest(manifest_path: Path) -> dict[str, Any]:
    """Replay the launch contract while explicitly allowing completed outputs."""

    manifest_path = manifest_path.expanduser().resolve(strict=True)
    try:
        manifest, _ = launch._read_manifest(manifest_path)  # noqa: SLF001
        historical_executor = _verify_ref(
            manifest.get("diagnostic_executor"), label="diagnostic_executor"
        )
        verified = launch.verify(
            manifest_path,
            expected_executor=historical_executor,
            require_fresh_outputs=False,
        )
    except launch.ExecutionError as error:
        raise CompletionError(str(error)) from error
    if verified["output_root"] != manifest_path.parent.resolve():
        raise CompletionError("manifest/output-root layout is not canonical")
    return verified


def _systemd_command(verified: Mapping[str, Any], *, unit: str) -> list[str]:
    return launch.base._systemd_command(verified, unit=unit)  # noqa: SLF001


def _verify_submission(verified: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    root = Path(verified["output_root"])
    manifest_ref = verified["manifest_ref"]
    receipt_path = root / "diagnostic-execution.receipt.json"
    receipt = _load_json(receipt_path, label="submission receipt")
    unhashed = dict(receipt)
    stated = unhashed.pop("receipt_sha256", None)
    expected_keys = {
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
    if not (
        set(receipt) == expected_keys
        and stated == _digest(unhashed)
        and receipt.get("schema_version") == launch.RECEIPT_SCHEMA
        and receipt.get("diagnostic_only") is True
        and receipt.get("promotion_eligible") is False
        and receipt.get("manifest") == manifest_ref
        and receipt.get("command_sha256")
        == verified["manifest"]["command_sha256"]
        and isinstance(receipt.get("unit"), str)
    ):
        raise CompletionError("gather submission receipt drift")
    unit = str(receipt["unit"])
    claim_path = _verify_ref(receipt.get("claim"), label="execution claim")
    if claim_path != (root / "diagnostic-execution.claim.json").resolve(strict=True):
        raise CompletionError("execution claim escaped the gather output root")
    claim = _load_json(claim_path, label="execution claim")
    claim_unhashed = dict(claim)
    claim_stated = claim_unhashed.pop("claim_sha256", None)
    if not (
        set(claim)
        == {"schema_version", "created_at_unix_ns", "manifest", "unit", "claim_sha256"}
        and claim_stated == _digest(claim_unhashed)
        and claim.get("schema_version") == launch.CLAIM_SCHEMA
        and claim.get("manifest") == manifest_ref
        and claim.get("unit") == unit
        and receipt.get("systemd_command_sha256")
        == _digest(_systemd_command(verified, unit=unit))
    ):
        raise CompletionError("gather execution claim/systemd identity drift")

    status_path = root / "diagnostic-execution.status.jsonl"
    try:
        events = [json.loads(row) for row in status_path.read_text().splitlines() if row]
    except (OSError, json.JSONDecodeError) as error:
        raise CompletionError(f"cannot load execution status journal: {error}") from error
    if not (
        len(events) == 2
        and events[0].get("schema_version") == launch.STATUS_SCHEMA
        and events[0].get("event") == "authorized"
        and events[0].get("claim_sha256") == claim["claim_sha256"]
        and events[1].get("schema_version") == launch.STATUS_SCHEMA
        and events[1].get("event") == "submitted"
        and events[1].get("receipt_sha256") == receipt["receipt_sha256"]
        and events[1].get("unit") == unit
    ):
        raise CompletionError("gather execution status journal drift")
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
        raise CompletionError(f"gather systemd unit is not complete: {observed}")
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
        raise CompletionError(f"cannot read gather systemd state: {error}") from error
    fields = dict(row.split("=", 1) for row in raw.splitlines() if "=" in row)
    return _verify_unit_state(fields)


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
    treatment = manifest["initialization_treatment"]
    descriptor = manifest["descriptor"]
    sentinel = manifest["validation_sentinel"]
    exact = {
        "init_checkpoint": treatment["path"],
        "init_checkpoint_sha256": treatment["sha256"],
        "checkpoint": checkpoint["path"],
        "data": descriptor["path"],
        "input_validation_game_sentinel_manifest": sentinel["path"],
        "world_size": prepare.WORLD_SIZE,
        "batch_size": prepare.LOCAL_BATCH_SIZE,
        "effective_global_batch_size": prepare.GLOBAL_BATCH_SIZE,
        "max_steps": prepare.OPTIMIZER_STEPS,
        "steps_completed": prepare.OPTIMIZER_STEPS,
        "training_row_draws": prepare.SELECTED_GLOBAL_ROW_DOSE,
        "base_training_row_draws": prepare.SELECTED_GLOBAL_ROW_DOSE,
        "total_training_row_draws": prepare.SELECTED_GLOBAL_ROW_DOSE,
        "optimizer": "adam",
        "resume_optimizer": False,
        "optimizer_restored": False,
        "fused_optimizer": False,
        "amp": "none",
        "lr": 3e-5,
        "lr_warmup_steps": 100,
        "lr_schedule": "flat",
        "weight_decay": 0.0,
        "value_lr_mult": 1.0,
        "action_module_lr_mult": prepare.ACTION_MODULE_LR_MULT,
        "policy_loss_weight": 1.0,
        "soft_target_source": "policy",
        "soft_target_weight": 0.9,
        "value_loss_weight": 0.25,
        "value_target_lambda": 1.0,
        "q_loss_weight": 0.0,
        "policy_kl_anchor_weight": 0.0,
        "forced_action_weight": 0.0,
        "forced_row_value_weight": 1.0,
        "winner_sample_weight": 1.0,
        "loser_sample_weight": 1.0,
        "freeze_modules": prepare.FREEZE_MODULES,
        "require_only_trainable_prefixes": prepare.TRAINABLE_PREFIX,
        "action_target_gather": True,
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
        raise CompletionError(f"gather report recipe/dose drift: {drift}")
    component_ids = list(prepare.production_temp.COMPONENT_IDS)
    composite = report.get("memmap_composite")
    if not (
        report.get("stored_policy_component_temperatures")
        == prepare.production_temp.COMPONENT_TEMPERATURES
        and isinstance(composite, dict)
        and composite.get("component_ids") == component_ids
        and composite.get("policy_distillation_component_ids") == component_ids
        and composite.get("value_training_component_ids") == component_ids
    ):
        raise CompletionError("gather report component/temperature scope drift")
    runtime = report.get("checkout_runtime_binding")
    selected_trainer = Path(
        manifest["selected_geometry_evidence"]["runtime"]["trainer"]
    ).resolve(strict=True)
    if not (
        isinstance(runtime, dict)
        and Path(str(runtime.get("trainer", ""))).resolve(strict=True) == selected_trainer
        and runtime.get("trainer_sha256")
        == prepare.corrected._file_ref(selected_trainer)["sha256"]  # noqa: SLF001
    ):
        raise CompletionError("gather report selected trainer binding drift")
    surface = report.get("training_information_surface", {}).get(
        "required_trainable_surface"
    )
    if not (
        isinstance(surface, dict)
        and surface.get("prefixes") == [prepare.TRAINABLE_PREFIX]
        and surface.get("parameter_tensors") == len(EXPECTED_CHANGED_PARAMETERS)
        and surface.get("parameters_by_prefix", {}).get(prepare.TRAINABLE_PREFIX)
        == surface.get("parameters")
        and isinstance(surface.get("parameters"), int)
        and surface["parameters"] > 0
    ):
        raise CompletionError("gather report trainable surface drift")
    metrics = report.get("metrics")
    matched = (
        metrics[0].get("validation_objective_matched")
        if isinstance(metrics, list) and len(metrics) == 1 and isinstance(metrics[0], dict)
        else None
    )
    observed_components = matched.get("components") if isinstance(matched, dict) else None
    if not (
        isinstance(matched, dict)
        and matched.get("schema_version") == "composite-validation-measure-v2"
        and matched.get("objective_matched") is True
        and isinstance(observed_components, dict)
        and set(observed_components) == set(component_ids)
        and _finite_number(matched.get("metrics", {}).get("active_policy_teacher_gap_closure"))
    ):
        raise CompletionError("gather report lacks objective-matched validation")
    epoch = metrics[0]
    optimizer_observability = epoch.get("optimizer_observability")
    if not (
        epoch.get("samples") == prepare.SELECTED_GLOBAL_ROW_DOSE
        and isinstance(epoch.get("policy_total_active_rows"), int)
        and epoch["policy_total_active_rows"] > 0
        and isinstance(optimizer_observability, dict)
        and optimizer_observability.get("observed_steps") == prepare.OPTIMIZER_STEPS
        and optimizer_observability.get("zero_objective_steps_skipped") == 0
        and _finite_number(report.get("elapsed_sec"))
        and float(report["elapsed_sec"]) > 0.0
    ):
        raise CompletionError("gather report optimizer/active-dose telemetry drift")
    summary = {
        "elapsed_sec": float(report["elapsed_sec"]),
        "policy_active_rows": int(epoch["policy_total_active_rows"]),
        "objective_matched_teacher_gap_closure": float(
            matched["metrics"]["active_policy_teacher_gap_closure"]
        ),
        "component_teacher_gap_closure": {
            component_id: float(
                observed_components[component_id]["metrics"][
                    "active_policy_teacher_gap_closure"
                ]
            )
            for component_id in component_ids
        },
        "optimizer_observability": optimizer_observability,
        "trainable_surface": surface,
    }
    return checkpoint, report_ref, summary


def _verify_progress(
    root: Path, *, checkpoint: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    progress_path = root / "candidate.pt.training-progress.json"
    optimizer_path = root / "candidate.pt.optimizer.pt"
    progress = _load_json(progress_path, label="training progress")
    unhashed = dict(progress)
    stated = unhashed.pop("progress_sha256", None)
    if stated != _digest(unhashed):
        raise CompletionError("gather progress semantic digest drift")

    def resolve_output(value: Any, *, label: str) -> dict[str, Any]:
        if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
            raise CompletionError(f"{label} reference is malformed")
        lexical = Path(str(value["path"]))
        path = lexical if lexical.is_absolute() else progress_path.parent / lexical
        ref = _compact_ref(path)
        if ref["sha256"] != value["sha256"]:
            raise CompletionError(f"{label} bytes drift")
        return _file_ref(path)

    progress_checkpoint = resolve_output(progress.get("checkpoint"), label="checkpoint")
    optimizer = resolve_output(progress.get("optimizer"), label="optimizer")
    if not (
        progress.get("optimizer_step") == prepare.OPTIMIZER_STEPS
        and progress.get("completed_epochs") == 1
        and isinstance(progress.get("rank_torch_rng_states"), list)
        and len(progress["rank_torch_rng_states"]) == prepare.WORLD_SIZE
        and progress_checkpoint["path"] == checkpoint["path"]
        and progress_checkpoint["sha256"] == checkpoint["sha256"]
        and optimizer["path"] == str(optimizer_path.resolve(strict=True))
    ):
        raise CompletionError("gather progress/RNG/optimizer dose drift")
    return _file_ref(progress_path), optimizer


def _tensor_digest(tensor: Any) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(json.dumps(list(contiguous.shape)).encode())
    digest.update(contiguous.numpy().tobytes())
    return "sha256:" + digest.hexdigest()


def _verify_adapter_only_delta(initializer: Path, candidate: Path) -> dict[str, Any]:
    try:
        import torch

        before = torch.load(initializer, map_location="cpu", weights_only=False)
        after = torch.load(candidate, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ModuleNotFoundError) as error:
        raise CompletionError(f"cannot load gather checkpoints: {error}") from error
    before_model = before.get("model") if isinstance(before, dict) else None
    after_model = after.get("model") if isinstance(after, dict) else None
    if not isinstance(before_model, dict) or not isinstance(after_model, dict):
        raise CompletionError("gather checkpoint model state is malformed")
    if set(before_model) != set(after_model):
        raise CompletionError("gather candidate parameter keys drifted")
    changed = tuple(
        sorted(
            name
            for name in before_model
            if not torch.equal(before_model[name], after_model[name])
        )
    )
    if changed != EXPECTED_CHANGED_PARAMETERS:
        raise CompletionError(
            f"candidate changed tensors outside/excluding gather adapter: {changed}"
        )
    if any(not bool(torch.isfinite(after_model[name]).all()) for name in changed):
        raise CompletionError("gather candidate contains non-finite adapter weights")
    evidence = {
        "inherited_parameter_tensors": len(before_model) - len(changed),
        "inherited_parameters_bit_identical": True,
        "changed_parameter_tensors": list(changed),
        "changed_tensor_sha256": {
            name: _tensor_digest(after_model[name]) for name in changed
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
    root = Path(verified["output_root"])
    unit, submission = _verify_submission(verified)
    checkpoint, report, report_summary = _verify_report(verified)
    if checkpoint["sha256"] != expected_checkpoint_sha256:
        raise CompletionError(
            "gather checkpoint differs from the explicitly expected completed bytes"
        )
    progress, optimizer = _verify_progress(root, checkpoint=checkpoint)
    model_delta = _verify_adapter_only_delta(
        Path(verified["manifest"]["initialization_treatment"]["path"]),
        Path(checkpoint["path"]),
    )
    state = _verify_unit_state(unit_state)
    finalizer = _file_ref(Path(__file__))
    completion = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "created_at_unix_ns": int(created_at_unix_ns),
        "manifest": verified["manifest_ref"],
        "completion_finalizer": finalizer,
        "expected_checkpoint_sha256": expected_checkpoint_sha256,
        "checkpoint": checkpoint,
        "report": report,
        "progress": progress,
        "optimizer": optimizer,
        "submission": submission,
        "unit": unit,
        "unit_state": state,
        "model_delta": model_delta,
        "verified_recipe": verified["manifest"]["adapter_commissioning_contract"],
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
        launch.base._write_exclusive(path, payload)  # noqa: SLF001
    except FileExistsError as error:
        raise CompletionError(f"gather completion already exists: {path}") from error
    return payload


def verify_completion(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    receipt = _load_json(path, label="gather completion receipt")
    unhashed = dict(receipt)
    stated = unhashed.pop("receipt_sha256", None)
    if not (
        receipt.get("schema_version") == SCHEMA
        and receipt.get("status") == STATUS
        and receipt.get("diagnostic_only") is True
        and receipt.get("promotion_eligible") is False
        and stated == _digest(unhashed)
    ):
        raise CompletionError("gather completion schema/status/digest drift")
    finalizer = _file_ref(Path(__file__))
    if receipt.get("completion_finalizer") != finalizer:
        raise CompletionError("gather completion finalizer bytes drift")
    replay = build_completion(
        Path(receipt["manifest"]["path"]),
        expected_checkpoint_sha256=str(receipt["expected_checkpoint_sha256"]),
        unit_state=receipt["unit_state"],
        created_at_unix_ns=int(receipt["created_at_unix_ns"]),
    )
    if replay != receipt:
        raise CompletionError("gather completion replay differs from receipt")
    expected_path = Path(replay["checkpoint"]["path"]).parent / COMPLETION_NAME
    if path != expected_path:
        raise CompletionError("gather completion receipt escaped output root")
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
