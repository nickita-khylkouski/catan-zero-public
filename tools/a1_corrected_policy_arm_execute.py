#!/usr/bin/env python3
"""Explicitly submit one immutable corrected learner manifest to systemd."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import a1_corrected_policy_arm as prepare  # noqa: E402


RECEIPT_SCHEMA = "a1-corrected-policy-arm-execution-receipt-v1"
STATUS_SCHEMA = "a1-corrected-policy-arm-execution-status-v1"


class ExecutionError(RuntimeError):
    """The immutable diagnostic cannot be submitted exactly once."""


def _read_manifest(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    payload, ref = prepare._load_json(path)  # noqa: SLF001
    stated = payload.get("manifest_sha256")
    unhashed = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if stated != prepare._digest(unhashed):  # noqa: SLF001
        raise ExecutionError("prepared manifest semantic digest drift")
    if (
        payload.get("schema_version") != prepare.SCHEMA
        or payload.get("diagnostic_only") is not True
        or payload.get("promotion_eligible") is not False
        or payload.get("diagnostic_execution_authorized") is not True
    ):
        raise ExecutionError("manifest does not authorize diagnostic execution")
    return payload, ref


def _verify_ref(value: Any, *, label: str) -> Path:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ExecutionError(f"{label} file reference is malformed")
    try:
        path = Path(str(value["path"])).expanduser().resolve(strict=True)
    except OSError as error:
        raise ExecutionError(f"cannot resolve {label}: {error}") from error
    if prepare._file_sha(path) != value["sha256"]:  # noqa: SLF001
        raise ExecutionError(f"{label} bytes drifted")
    return path


def _option(command: Sequence[str], flag: str) -> str:
    try:
        return prepare._option(command, flag)  # noqa: SLF001
    except prepare.ArmError as error:
        raise ExecutionError(str(error)) from error


def _git_head(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=repo, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ExecutionError(f"cannot identify execution checkout: {error}") from error


def _verify_event_history_training_contract(
    manifest: dict[str, Any], command: list[str], descriptor: Path
) -> None:
    contract = manifest.get("event_history_training_contract")
    if not isinstance(contract, dict) or not (
        contract.get("schema") == prepare.EVENT_HISTORY_COMMAND_CONTRACT_SCHEMA
        and contract.get("crop_authenticated_empty_event_history") is True
    ):
        raise ExecutionError("manifest has no authenticated event-history contract")
    try:
        descriptor_meta, _ = prepare._preflight_descriptor(descriptor)  # noqa: SLF001
        expected_contract = prepare._event_history_training_contract(  # noqa: SLF001
            descriptor_meta
        )
    except prepare.ArmError as error:
        raise ExecutionError(str(error)) from error
    if contract != expected_contract:
        raise ExecutionError("event-history contract differs from descriptor inventories")
    expected = [
        row["payload_inventory_sha256"]
        for row in contract["empty_payload_inventory_acknowledgements"]
    ]
    positions = [
        index
        for index, value in enumerate(command)
        if value == prepare.EVENT_HISTORY_ACK_FLAG
    ]
    observed = [
        command[index + 1]
        for index in positions
        if index + 1 < len(command) and not command[index + 1].startswith("--")
    ]
    if observed != expected or len(positions) != len(expected):
        raise ExecutionError("command lacks the exact event-history inventory ACK set")
    if command.count(prepare.EVENT_HISTORY_CROP_FLAG) != 1:
        raise ExecutionError("command lacks the authenticated empty-history crop flag")


def _verify_validation_independence_contract(
    manifest: dict[str, Any], descriptor_meta: dict[str, Any]
) -> None:
    contract = manifest.get("validation_independence_contract")
    if not isinstance(contract, dict):
        raise ExecutionError("manifest has no validation independence contract")
    stated = contract.get("contract_sha256")
    unhashed = {key: value for key, value in contract.items() if key != "contract_sha256"}
    if stated != prepare._digest(unhashed):  # noqa: SLF001
        raise ExecutionError("validation independence contract digest drift")
    source_path = _verify_ref(
        manifest.get("source_validation_sentinel"), label="source_validation_sentinel"
    )
    fresh_path = _verify_ref(
        manifest.get("validation_sentinel"), label="validation_sentinel"
    )
    try:
        source = json.loads(source_path.read_text(encoding="utf-8"))
        fresh = json.loads(fresh_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExecutionError(f"validation sentinel is unreadable: {error}") from error
    source_games = source.get("game_seeds") if isinstance(source, dict) else None
    fresh_games = fresh.get("game_seeds") if isinstance(fresh, dict) else None
    if not (
        isinstance(source_games, list)
        and source_games
        and isinstance(fresh_games, list)
        and fresh_games
        and not set(source_games).intersection(fresh_games)
    ):
        raise ExecutionError("validation sentinel does not use fresh disjoint games")
    component_ids = list(descriptor_meta.get("component_ids", ()))
    ratios = list(descriptor_meta.get("component_game_sampling_ratios", ()))
    rows = contract.get("component_rows")
    if not isinstance(rows, list) or [row.get("component_id") for row in rows] != component_ids:
        raise ExecutionError("validation independence component identity drift")
    for row, ratio in zip(rows, ratios, strict=True):
        if (
            not isinstance(row, dict)
            or not np.isclose(float(row.get("target_row_ratio", -1.0)), float(ratio))
            or int(row.get("selected_game_count", 0)) <= 0
            or int(row.get("selected_row_count", 0)) <= 0
            or abs(
                int(row.get("selected_row_count", 0))
                - int(row.get("target_row_count", -1))
            ) > int(row.get("max_whole_game_row_count", -1))
        ):
            raise ExecutionError("validation independence component allocation drift")
    if not (
        contract.get("source_selected_game_seed_set_sha256")
        == source.get("selected_game_seed_set_sha256")
        and contract.get("fresh_selected_game_seed_set_sha256")
        == fresh.get("selected_game_seed_set_sha256")
        and contract.get("selection_overlap_game_count") == 0
        and contract.get("selection_scope")
        == "fresh_whole_games_stratified_to_winning_operator"
        and contract.get("predecessor_component_id") == component_ids[-1]
        and np.isclose(float(contract.get("predecessor_target_row_ratio", -1.0)), 0.2)
        and contract.get("complete_component_holdouts_remain_training_excluded") is True
        and sum(int(row["selected_row_count"]) for row in rows)
        == int(fresh.get("selected_row_count", -1))
    ):
        raise ExecutionError("validation independence evidence differs from sentinels")


def verify(manifest_path: Path) -> dict[str, Any]:
    manifest, manifest_ref = _read_manifest(manifest_path)
    for field in (
        "source_receipt", "source_descriptor", "descriptor",
        "source_validation_sentinel", "validation_sentinel", "initialization",
        "evaluation_baseline",
    ):
        _verify_ref(manifest.get(field), label=field)
    lineage = manifest.get("failed_retry_lineage", {}).get("artifacts")
    if not isinstance(lineage, list) or len(lineage) != len(prepare.LINEAGE_ROLES):
        raise ExecutionError("failed/retry lineage is incomplete")
    for role, row in zip(prepare.LINEAGE_ROLES, lineage, strict=True):
        if not isinstance(row, dict) or row.get("role") != role:
            raise ExecutionError("failed/retry lineage order drift")
        _verify_ref(row.get("file"), label=f"lineage.{role}")
    source_binding = manifest.get("source_binding")
    if not isinstance(source_binding, dict):
        raise ExecutionError("manifest has no source checkout binding")
    repo = Path(str(source_binding.get("repository_root", ""))).resolve(strict=True)
    if _git_head(repo) != source_binding.get("git_commit"):
        raise ExecutionError("execution checkout commit differs from prepared manifest")
    files = source_binding.get("files")
    if not isinstance(files, dict) or not files:
        raise ExecutionError("source checkout binding has no files")
    for relative, ref in files.items():
        path = _verify_ref(ref, label=f"source.{relative}")
        if path != (repo / relative).resolve(strict=True):
            raise ExecutionError(f"source path escaped checkout: {relative}")
    command = manifest.get("command")
    if not isinstance(command, list) or not all(isinstance(value, str) for value in command):
        raise ExecutionError("manifest command is malformed")
    if manifest.get("command_sha256") != prepare._digest(command):  # noqa: SLF001
        raise ExecutionError("manifest command digest drift")
    trainer = [Path(value).resolve() for value in command if Path(value).name == "train_bc.py"]
    if trainer != [(repo / "tools/train_bc.py").resolve(strict=True)]:
        raise ExecutionError("manifest trainer is not the bound checkout")
    exact_inputs = {
        "--data": manifest["descriptor"]["path"],
        "--validation-game-sentinel-manifest": manifest["validation_sentinel"]["path"],
        "--init-checkpoint": manifest["initialization"]["path"],
    }
    if manifest["evaluation_baseline"] != manifest["initialization"]:
        raise ExecutionError("evaluation baseline differs from learner initializer")
    for flag, expected in exact_inputs.items():
        if _option(command, flag) != expected:
            raise ExecutionError(f"command differs from bound {flag}")
    _verify_event_history_training_contract(
        manifest, command, Path(manifest["descriptor"]["path"])
    )
    try:
        descriptor_meta, _ = prepare._preflight_descriptor(  # noqa: SLF001
            Path(manifest["descriptor"]["path"])
        )
        _verify_validation_independence_contract(manifest, descriptor_meta)
        policy_active_dose = prepare._derive_policy_active_dose(  # noqa: SLF001
            descriptor_meta
        )
        supervision_contract = prepare._next_supervision_contract(  # noqa: SLF001
            descriptor_meta, command, policy_active_dose
        )
    except prepare.ArmError as error:
        raise ExecutionError(str(error)) from error
    if manifest.get("supervision_contract") != supervision_contract:
        raise ExecutionError(
            "manifest supervision contract differs from executable descriptor/command"
        )
    if "--validation-game-seed-manifest" in command:
        raise ExecutionError("command contains a second validation control")
    output_root = Path(_option(command, "--checkpoint")).parent.resolve()
    checkpoint = output_root / "candidate.pt"
    report = output_root / "train.report.json"
    if Path(_option(command, "--checkpoint")) != checkpoint or Path(
        _option(command, "--report")
    ) != report:
        raise ExecutionError("command outputs are not canonical corrected-arm paths")
    forbidden = (
        checkpoint, Path(str(checkpoint) + ".optimizer.pt"),
        Path(str(checkpoint) + ".training-progress.json"), report,
        output_root / "diagnostic-execution.claim.json",
        output_root / "diagnostic-execution.receipt.json",
        output_root / "diagnostic-execution.status.jsonl",
        output_root / "stdout.log", output_root / "stderr.log",
    )
    existing = [str(path) for path in forbidden if path.exists()]
    if existing:
        raise ExecutionError(f"corrected-arm output/claim already exists: {existing}")
    return {
        "manifest": manifest, "manifest_ref": manifest_ref, "repo": repo,
        "command": command, "output_root": output_root,
    }


def verify_training_report(manifest_path: Path, report_path: Path) -> dict[str, Any]:
    """Authenticate the realized trainer report against the next recipe.

    This is intentionally separate from :func:`verify`: pre-submit verification
    requires fresh output paths, while completion verification necessarily runs
    after ``train.report.json`` exists.
    """

    manifest, manifest_ref = _read_manifest(manifest_path)
    command = manifest.get("command")
    if not isinstance(command, list) or manifest.get("command_sha256") != prepare._digest(  # noqa: SLF001
        command
    ):
        raise ExecutionError("manifest command is malformed or drifted")
    expected_path = Path(_option(command, "--report")).resolve()
    report_path = report_path.expanduser().resolve(strict=True)
    if report_path != expected_path:
        raise ExecutionError("training report path differs from sealed command")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExecutionError(f"training report is unreadable: {error}") from error
    if not isinstance(report, dict):
        raise ExecutionError("training report must contain an object")
    exact_execution = {
        "init_checkpoint_sha256": manifest["initialization"]["sha256"],
        "optimizer_restored": False,
        "resume_optimizer": False,
        "resumed_optimizer_step": None,
        "world_size": 8,
        "batch_size": 512,
        "grad_accum_steps": 1,
        "effective_global_batch_size": prepare.GLOBAL_BATCH_SIZE,
        "training_row_draws": prepare.GLOBAL_ROW_DOSE,
        "max_steps": prepare.OPTIMIZER_STEPS,
        "steps_completed": prepare.OPTIMIZER_STEPS,
        "total_training_steps": prepare.OPTIMIZER_STEPS,
        "data_fingerprint": manifest["descriptor_fingerprint"],
        "input_validation_game_seed_manifest": manifest["validation_sentinel"]["path"],
        "input_validation_game_seed_manifest_sha256": manifest["validation_sentinel"]["sha256"],
        "input_validation_game_sentinel_manifest": manifest["validation_sentinel"]["path"],
        "validation_game_seed_set_sha256": manifest[
            "validation_sentinel_selection_sha256"
        ],
    }
    observed_execution = {key: report.get(key) for key in exact_execution}
    if observed_execution != exact_execution:
        raise ExecutionError(
            f"training report one-dose execution identity drift: {observed_execution}"
        )
    command = manifest.get("command")
    if not isinstance(command, list):
        raise ExecutionError("manifest training command is malformed")
    command_bound_operator = {
        "epochs": int(prepare._option(command, "--epochs")),  # noqa: SLF001
        "lr": float(prepare._option(command, "--lr")),  # noqa: SLF001
        "lr_warmup_steps": int(prepare._option(command, "--lr-warmup-steps")),  # noqa: SLF001
        "lr_schedule": prepare._option(command, "--lr-schedule"),  # noqa: SLF001
        "value_loss_weight": float(prepare._option(command, "--value-loss-weight")),  # noqa: SLF001
        "value_lr_mult": float(prepare._option(command, "--value-lr-mult")),  # noqa: SLF001
        "value_target_lambda": float(prepare._option(command, "--value-target-lambda")),  # noqa: SLF001
        "forced_action_weight": float(prepare._option(command, "--forced-action-weight")),  # noqa: SLF001
        "forced_row_value_weight": float(prepare._option(command, "--forced-row-value-weight")),  # noqa: SLF001
        "policy_loss_weight": float(prepare._option(command, "--policy-loss-weight")),  # noqa: SLF001
        "soft_target_temperature": float(prepare._option(command, "--soft-target-temperature")),  # noqa: SLF001
        "soft_target_min_legal_coverage": float(
            prepare._option(command, "--soft-target-min-legal-coverage")  # noqa: SLF001
        ),
        "mask_hidden_info": "--mask-hidden-info" in command,
    }
    observed_operator = {key: report.get(key) for key in command_bound_operator}
    if observed_operator != command_bound_operator:
        raise ExecutionError(
            f"training report command-bound operator drift: {observed_operator}"
        )
    contract = manifest.get("supervision_contract")
    if not isinstance(contract, dict) or contract.get(
        "schema_version"
    ) != prepare.SUPERVISION_CONTRACT_SCHEMA:
        raise ExecutionError("manifest has no next-learner supervision contract")
    expected_components = contract.get("component_ids")
    if not isinstance(expected_components, list) or len(expected_components) < 2:
        raise ExecutionError("manifest winning component scope is malformed")
    policy_scope = report.get("policy_distillation_scope")
    value_scope = report.get("value_training_scope")
    composite = report.get("memmap_composite")
    if not (
        isinstance(policy_scope, dict)
        and policy_scope.get("component_ids") == expected_components
        and isinstance(value_scope, dict)
        and value_scope.get("component_ids") == expected_components
        and isinstance(composite, dict)
        and composite.get("policy_distillation_component_ids") == expected_components
        and composite.get("value_training_component_ids") == expected_components
        and composite.get("policy_kl_anchor_component_ids") == expected_components
        and composite.get("policy_distillation_scope_explicit") is True
        and composite.get("value_training_scope_explicit") is True
    ):
        raise ExecutionError("training report supervision-scope provenance drift")
    exact_scalars = {
        "soft_target_source": "policy",
        "soft_target_weight": 0.9,
        "policy_aux_active_batch_size": 0,
        "policy_kl_anchor_direction": "forward",
        "policy_kl_anchor_weight": prepare.REPLAY_ANCHOR_WEIGHT,
        "winner_sample_weight": 1.0,
        "loser_sample_weight": 1.0,
    }
    observed = {key: report.get(key) for key in exact_scalars}
    if observed != exact_scalars:
        raise ExecutionError(f"training report supervision scalar drift: {observed}")
    dose_contract = contract.get("policy_active_row_dose")
    recipe = manifest.get("recipe")
    reference = dose_contract.get("reference_base_active_rows") if isinstance(dose_contract, dict) else None
    expected_dose_shape = (
        isinstance(dose_contract, dict)
        and dose_contract.get("derivation")
        == "authenticated_game_uniform_activity_weighted_by_component_sampling_ratio"
        and dose_contract.get("component_sampling_ratios")
        == contract.get("component_game_sampling_ratios")
        and dose_contract.get("global_row_dose") == prepare.GLOBAL_ROW_DOSE
        and isinstance(dose_contract.get("available_training_rows"), int)
        and dose_contract["available_training_rows"] >= prepare.GLOBAL_ROW_DOSE
        and isinstance(reference, int) and not isinstance(reference, bool)
        and dose_contract.get("base_active_rows_tolerance")
        == prepare.POLICY_BASE_ACTIVE_ROW_TOLERANCE
        and dose_contract.get("min_base_active_rows")
        == reference - prepare.POLICY_BASE_ACTIVE_ROW_TOLERANCE
        and dose_contract.get("max_base_active_rows")
        == reference + prepare.POLICY_BASE_ACTIVE_ROW_TOLERANCE
        and dose_contract.get("expected_aux_active_rows")
        == prepare.EXPECTED_POLICY_AUX_ACTIVE_ROWS
        and dose_contract.get("accounting")
        == "realized_policy_active_rows_not_global_samples"
        and isinstance(dose_contract.get("component_statistics"), list)
        and isinstance(recipe, dict)
        and recipe.get("expected_policy_base_active_rows") == reference
    )
    if not expected_dose_shape:
        raise ExecutionError("manifest policy-active dose contract drift")
    base_active = report.get("policy_base_active_rows")
    aux_active = report.get("policy_aux_active_rows")
    total_active = report.get("policy_total_active_rows")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (base_active, aux_active, total_active)
    ):
        raise ExecutionError("training report policy-active dose is missing or malformed")
    if not (
        dose_contract["min_base_active_rows"]
        <= base_active
        <= dose_contract["max_base_active_rows"]
    ):
        raise ExecutionError(
            "training report base policy-active dose differs from the sealed band"
        )
    if aux_active != dose_contract["expected_aux_active_rows"]:
        raise ExecutionError(
            "training report auxiliary policy-active dose differs from the sealed recipe"
        )
    if total_active != base_active + aux_active:
        raise ExecutionError("training report total policy-active dose does not add up")
    metrics = report.get("metrics")
    if not isinstance(metrics, list) or len(metrics) != 1 or not isinstance(metrics[0], dict):
        raise ExecutionError("one-dose report must contain exactly one epoch of metrics")
    matched = metrics[0].get("validation_objective_matched")
    if not (
        isinstance(matched, dict)
        and matched.get("schema_version") == "composite-validation-measure-v2"
        and matched.get("objective_matched") is True
        and matched.get("component_sampling_ratios")
        == {
            component_id: float(ratio)
            for component_id, ratio in zip(
                expected_components,
                contract["component_game_sampling_ratios"],
                strict=True,
            )
        }
    ):
        raise ExecutionError("training report lacks objective-matched validation")
    components = matched.get("components")
    if not isinstance(components, dict) or set(components) != set(expected_components):
        raise ExecutionError("objective-matched validation component scope drift")
    required_metrics = (
        "loss",
        "policy_loss",
        "value_loss",
        "accuracy",
        "active_policy_teacher_gap_closure",
    )
    for component_id in expected_components:
        component = components[component_id]
        component_metrics = component.get("metrics") if isinstance(component, dict) else None
        if not (
            isinstance(component_metrics, dict)
            and int(component.get("games", 0)) > 0
            and int(component.get("rows", 0)) > 0
            and all(
                isinstance(component_metrics.get(key), (int, float))
                and not isinstance(component_metrics.get(key), bool)
                and np.isfinite(float(component_metrics[key]))
                for key in required_metrics
            )
        ):
            raise ExecutionError(
                f"objective-matched validation metrics are incomplete for {component_id}"
            )
    return {
        "manifest": manifest_ref,
        "report": {"path": str(report_path), "sha256": prepare._file_sha(report_path)},  # noqa: SLF001
        "supervision_contract_sha256": contract.get("contract_sha256"),
        "policy_active_row_dose": {
            "base": base_active,
            "aux": aux_active,
            "total": total_active,
        },
        "verified": True,
    }


def _probe_conflicting_compute(
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> list[str]:
    try:
        topology = runner(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True, text=True, capture_output=True,
        )
        result = runner(
            ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
            check=True, text=True, capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ExecutionError(f"cannot prove B200s are idle: {error}") from error
    names = [line.strip() for line in topology.stdout.splitlines() if line.strip()]
    if len(names) != 8 or any("B200" not in name for name in names):
        raise ExecutionError(f"executor requires exactly eight visible B200s: {names}")
    return [
        line.strip() for line in result.stdout.splitlines()
        if line.strip() and "nvidia-cuda-mps" not in line.lower()
    ]


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode() + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        raise


def _append_status(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, sort_keys=True).encode() + b"\n"
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def execute(
    manifest_path: Path, *, unit: str,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    conflict_probe: Callable[[], list[str]] = _probe_conflicting_compute,
) -> dict[str, Any]:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@-]{0,79}", unit) is None:
        raise ExecutionError("systemd unit name is invalid")
    verified = verify(manifest_path)
    return _submit_verified(
        verified,
        unit=unit,
        runner=runner,
        conflict_probe=conflict_probe,
        claim_schema="a1-corrected-policy-arm-execution-claim-v1",
        receipt_schema=RECEIPT_SCHEMA,
        status_schema=STATUS_SCHEMA,
    )


def _systemd_command(
    verified: Mapping[str, Any],
    *,
    unit: str,
    retain_exit_status: bool = True,
) -> list[str]:
    """Build the transient-unit command with durable child-exit evidence.

    ``systemd-run --collect`` removes a completed transient unit.  Querying the
    now-missing name with ``systemctl show`` returns synthetic defaults such as
    ``Result=success`` and ``ExecMainStatus=0`` even when torchrun failed.  A
    retained ``RemainAfterExit`` unit instead has an unambiguous terminal state:
    successful jobs are loaded/active/exited, while failures are loaded/failed
    with the real child exit status.  The legacy mode exists only so historical
    submission receipts can still be replayed from their bound checkout.
    """

    output_root = Path(verified["output_root"])
    command = [
        "sudo",
        "-n",
        "systemd-run",
        f"--unit={unit}",
        "--uid=ubuntu",
        "--gid=ubuntu",
        "--service-type=exec",
    ]
    if retain_exit_status:
        command.append("--property=RemainAfterExit=yes")
    else:
        command.append("--collect")
    command.extend(
        (
            "--property=LimitNOFILE=65536",
            f"--property=WorkingDirectory={verified['repo']}",
            f"--property=StandardOutput=append:{output_root / 'stdout.log'}",
            f"--property=StandardError=append:{output_root / 'stderr.log'}",
            "--setenv=HOME=/home/ubuntu",
            "--setenv=PYTHONNOUSERSITE=1",
            "--",
            *verified["command"],
        )
    )
    return command


def _submit_verified(
    verified: dict[str, Any],
    *,
    unit: str,
    runner: Callable[..., subprocess.CompletedProcess[str]],
    conflict_probe: Callable[[], list[str]],
    claim_schema: str,
    receipt_schema: str,
    status_schema: str,
) -> dict[str, Any]:
    """Submit one already schema-verified diagnostic with append-only evidence."""

    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@-]{0,79}", unit) is None:
        raise ExecutionError("systemd unit name is invalid")
    conflicts = conflict_probe()
    if conflicts:
        raise ExecutionError(f"B200 compute is not idle: {conflicts}")
    output_root = verified["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    now = time.time_ns()
    claim = {
        "schema_version": claim_schema,
        "created_at_unix_ns": now,
        "manifest": verified["manifest_ref"],
        "unit": unit,
    }
    claim["claim_sha256"] = prepare._digest(claim)  # noqa: SLF001
    claim_path = output_root / "diagnostic-execution.claim.json"
    try:
        _write_exclusive(claim_path, claim)
    except FileExistsError as error:
        raise ExecutionError(
            f"diagnostic execution was already claimed: {claim_path}"
        ) from error
    status_path = output_root / "diagnostic-execution.status.jsonl"
    _append_status(status_path, {
        "schema_version": status_schema, "event": "authorized",
        "created_at_unix_ns": now, "claim_sha256": claim["claim_sha256"],
    })
    systemd_command = _systemd_command(verified, unit=unit)
    try:
        result = runner(systemd_command, check=True, text=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as error:
        _append_status(status_path, {
            "schema_version": status_schema, "event": "submission_failed",
            "created_at_unix_ns": time.time_ns(), "error": str(error),
        })
        raise ExecutionError(f"systemd submission failed: {error}") from error
    receipt = {
        "schema_version": receipt_schema,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "created_at_unix_ns": time.time_ns(),
        "manifest": verified["manifest_ref"],
        "claim": {"path": str(claim_path), "sha256": prepare._file_sha(claim_path)},  # noqa: SLF001
        "unit": unit,
        "command_sha256": verified["manifest"]["command_sha256"],
        "systemd_command_sha256": prepare._digest(systemd_command),  # noqa: SLF001
        "systemd_stdout": result.stdout.strip(),
    }
    receipt["receipt_sha256"] = prepare._digest(receipt)  # noqa: SLF001
    receipt_path = output_root / "diagnostic-execution.receipt.json"
    _write_exclusive(receipt_path, receipt)
    _append_status(status_path, {
        "schema_version": status_schema, "event": "submitted",
        "created_at_unix_ns": receipt["created_at_unix_ns"],
        "receipt_sha256": receipt["receipt_sha256"], "unit": unit,
    })
    return receipt


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--unit", default="a1-corrected-anchor-k3")
    parser.add_argument("--go", action="store_true")
    parser.add_argument("--verify-report", type=Path)
    args = parser.parse_args(argv)
    if args.verify_report is not None:
        if args.go:
            raise SystemExit("--verify-report and --go are mutually exclusive")
        print(json.dumps(verify_training_report(args.manifest, args.verify_report), sort_keys=True))
        return
    if not args.go:
        verified = verify(args.manifest)
        print(json.dumps({"verified": True, "launched": False,
                          "manifest": verified["manifest_ref"]}, sort_keys=True))
        return
    receipt = execute(args.manifest, unit=args.unit)
    print(json.dumps({"submitted": True, "unit": receipt["unit"],
                      "receipt_sha256": receipt["receipt_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
