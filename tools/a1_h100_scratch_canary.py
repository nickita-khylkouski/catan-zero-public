#!/usr/bin/env python3
"""Bounded, non-promotable H100 scratch comparison for current A1 science.

This launcher is intentionally separate from :mod:`tools.a1_scratch_train`.
It can collect matched diagnostic evidence while the production B200 scratch
schedule remains unresolved, but neither its checkpoints nor its receipts can
be submitted to production admission or promotion.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import a1_current_science_contract as current_science  # noqa: E402
from tools import a1_feature_signal_admission as feature_signal  # noqa: E402
from tools import a1_scratch_train as scratch  # noqa: E402
from tools import train_bc  # noqa: E402


PLAN_SCHEMA = "a1-h100-scratch-canary-plan-v1"
EXECUTION_SCHEMA = "a1-h100-scratch-canary-execution-v1"
WORLD_SIZE = 8
LOCAL_BATCH_SIZE = 64
GLOBAL_BATCH_SIZE = WORLD_SIZE * LOCAL_BATCH_SIZE
MIN_STEPS = 128
MAX_STEPS = 256
ARM_IDS = ("C640", "T640")
CANARY_MAX_PARAMETER_COUNT = 43_000_000
CODE_SURFACE = (
    "tools/a1_h100_scratch_canary.py",
    "tools/a1_scratch_train.py",
    "tools/a1_current_science_contract.py",
    "tools/train_bc.py",
    "src/catan_zero/rl/entity_token_policy.py",
    "src/catan_zero/rl/relational_trunks.py",
)


class CanaryError(RuntimeError):
    """The bounded H100 diagnostic cannot be authenticated or executed."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _value_sha256(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _atomic_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    target = path.expanduser().absolute()
    if target.exists() or target.is_symlink():
        raise CanaryError(f"refusing non-fresh receipt path: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    value = dict(payload)
    value["receipt_sha256"] = _value_sha256(value)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="ascii") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _plan_identity_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "created_unix_ns",
            "status",
            "receipt_sha256",
            "plan_identity_sha256",
        }
    }


def _seal_plan_identity(payload: dict[str, Any]) -> None:
    payload["plan_identity_sha256"] = _value_sha256(
        _plan_identity_payload(payload)
    )


def _load_bound_plan(path: Path, current: Mapping[str, Any]) -> dict[str, Any]:
    target = path.expanduser().absolute()
    if not target.is_file() or target.is_symlink():
        raise CanaryError("--go requires a regular immutable plan receipt")
    try:
        plan = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CanaryError(f"cannot read canary plan receipt: {error}") from error
    if not isinstance(plan, dict):
        raise CanaryError("canary plan receipt root is not an object")
    unsigned = dict(plan)
    receipt_sha = unsigned.pop("receipt_sha256", None)
    plan_identity = unsigned.get("plan_identity_sha256")
    if (
        plan.get("schema_version") != PLAN_SCHEMA
        or plan.get("status") != "planned"
        or not isinstance(receipt_sha, str)
        or receipt_sha != _value_sha256(unsigned)
        or plan_identity != _value_sha256(_plan_identity_payload(plan))
        or plan_identity != current.get("plan_identity_sha256")
        or _plan_identity_payload(plan) != _plan_identity_payload(current)
    ):
        raise CanaryError("execution arguments differ from immutable canary plan")
    return {
        "plan": plan,
        "plan_receipt": {
            "path": str(target.resolve()),
            "file_sha256": _file_sha256(target),
            "receipt_sha256": receipt_sha,
            "plan_identity_sha256": plan_identity,
        },
    }


def _validate_step_bound(max_steps: int) -> int:
    steps = int(max_steps)
    if not MIN_STEPS <= steps <= MAX_STEPS:
        raise CanaryError(
            f"bounded H100 canary requires max_steps in {MIN_STEPS}..{MAX_STEPS}"
        )
    return steps


def _matched_identity(
    *, model: Mapping[str, Any], recipe: Mapping[str, Any], max_steps: int
) -> dict[str, Any]:
    matched_model = dict(model)
    matched_model.pop("topology_residual_adapter", None)
    return {
        "initialization": current_science.learner_initialization(),
        "model_construction_excluding_declared_delta": matched_model,
        "recipe": dict(recipe),
        "max_steps": int(max_steps),
        "exact_max_steps": True,
        "seed": int(recipe["seed"]),
        "world_size": WORLD_SIZE,
        "local_batch_size": LOCAL_BATCH_SIZE,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "ddp_shard_data": False,
        "training_rng_rank_offset": True,
        "max_parameter_count": CANARY_MAX_PARAMETER_COUNT,
    }


def build_arm_contracts(*, max_steps: int) -> dict[str, dict[str, Any]]:
    """Return the exact matched C640/T640 experiment contract."""

    steps = _validate_step_bound(max_steps)
    production_model = current_science.learner_model_construction()
    if (
        production_model.get("hidden_size") != 640
        or production_model.get("action_target_gather") is not True
    ):
        raise CanaryError("current control is no longer the sealed C640 gather model")
    recipe = current_science.learner_training_recipe()
    if float(recipe.get("value_trunk_grad_scale", -1.0)) != 0.25:
        raise CanaryError("current control is no longer the sealed V25 recipe")
    model_base = {**production_model, "topology_residual_adapter": False}
    matched = _matched_identity(model=model_base, recipe=recipe, max_steps=steps)
    matched_sha = _value_sha256(matched)
    result: dict[str, dict[str, Any]] = {}
    for arm_id, topology_residual in (("C640", False), ("T640", True)):
        model = {**model_base, "topology_residual_adapter": topology_residual}
        contract = {
            "arm_id": arm_id,
            "diagnostic_only": True,
            "promotion_eligible": False,
            "production_admission": "forbidden",
            "declared_delta": (
                "none_current_control"
                if arm_id == "C640"
                else "model_construction.topology_residual_adapter=false->true"
            ),
            "model_construction": model,
            "model_construction_sha256": _value_sha256(model),
            "recipe": copy.deepcopy(recipe),
            "recipe_sha256": _value_sha256(recipe),
            "max_steps": steps,
            "exact_max_steps": True,
            "max_parameter_count": CANARY_MAX_PARAMETER_COUNT,
            "matched_identity": matched,
            "matched_identity_sha256": matched_sha,
        }
        contract["config_sha256"] = _value_sha256(contract)
        result[arm_id] = contract
    observed = arm_drift(result["C640"], result["T640"])
    expected = {
        "model_construction.topology_residual_adapter": {
            "C640": False,
            "T640": True,
        }
    }
    if observed != expected:
        raise CanaryError(f"matched-arm contract drifted: {observed}")
    return result


def arm_drift(
    control: Mapping[str, Any], treatment: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Report scientific deltas, excluding labels, hashes, and declarations."""

    drift: dict[str, dict[str, Any]] = {}
    left_model = dict(control["model_construction"])
    right_model = dict(treatment["model_construction"])
    for key in sorted(set(left_model) | set(right_model)):
        if left_model.get(key) != right_model.get(key):
            drift[f"model_construction.{key}"] = {
                "C640": left_model.get(key),
                "T640": right_model.get(key),
            }
    if control["recipe"] != treatment["recipe"]:
        drift["recipe"] = {"C640": control["recipe"], "T640": treatment["recipe"]}
    if control["matched_identity_sha256"] != treatment["matched_identity_sha256"]:
        drift["matched_identity_sha256"] = {
            "C640": control["matched_identity_sha256"],
            "T640": treatment["matched_identity_sha256"],
        }
    return drift


def _replace_value(command: list[str], flag: str, value: object) -> None:
    try:
        index = command.index(flag)
    except ValueError as error:
        raise CanaryError(f"base scratch command lost required flag {flag}") from error
    if index + 1 >= len(command):
        raise CanaryError(f"base scratch flag has no value: {flag}")
    command[index + 1] = str(value)


def _remove_pair(command: list[str], flag: str) -> None:
    try:
        index = command.index(flag)
    except ValueError as error:
        raise CanaryError(f"base scratch command lost required flag {flag}") from error
    if index + 1 >= len(command):
        raise CanaryError(f"base scratch flag has no value: {flag}")
    del command[index : index + 2]


def _code_binding() -> dict[str, Any]:
    records = [
        {
            "kind": "learner_code",
            "relative_path": relative,
            "path": str((REPO_ROOT / relative).resolve(strict=True)),
            "sha256": _file_sha256((REPO_ROOT / relative).resolve(strict=True)),
        }
        for relative in CODE_SURFACE
    ]
    value: dict[str, Any] = {
        "schema_version": "a1-scratch-topology-code-binding-v1",
        "records": records,
    }
    value["code_tree_sha256"] = _value_sha256(value)
    return value


def diagnostic_authority(
    *,
    arm_id: str,
    max_steps: int,
    checkpoint_steps: Sequence[int],
    code_tree_sha256: str,
) -> dict[str, Any]:
    if arm_id not in ARM_IDS:
        raise CanaryError(f"unknown topology arm {arm_id!r}")
    steps = _validate_step_bound(max_steps)
    checkpoints = [int(step) for step in checkpoint_steps]
    if checkpoints != sorted(set(checkpoints)) or any(
        step <= 0 or step >= steps for step in checkpoints
    ):
        raise CanaryError("topology canary checkpoint steps are invalid")
    recipe = current_science.learner_training_recipe()
    model = current_science.learner_model_construction()
    topology = current_science.learner_execution_topology()
    expected_effective = train_bc._a1_scratch_topology_expected_effective_recipe(  # noqa: SLF001
        source_recipe=recipe,
        source_model=model,
        source_topology=topology,
        max_steps=steps,
    )
    return {
        "schema_version": "a1-scratch-topology-diagnostic-authority-v1",
        "campaign_id": "scratch-topology-c640-t640",
        "arm_id": arm_id,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "exact_max_steps": True,
        "max_steps": steps,
        "epochs": 1,
        "checkpoint_steps": checkpoints,
        "value_trunk_grad_scale": float(recipe["value_trunk_grad_scale"]),
        "topology_residual_adapter": arm_id == "T640",
        "max_parameter_count": CANARY_MAX_PARAMETER_COUNT,
        "effective_recipe_sha256": _value_sha256(expected_effective),
        "code_surface_sha256": _value_sha256(list(CODE_SURFACE)),
        "required_accelerator_family": "NVIDIA H100",
        "required_cuda_device_count": WORLD_SIZE,
        "source_recipe_sha256": _value_sha256(recipe),
        "source_execution_topology_sha256": _value_sha256(topology),
        "code_tree_sha256": str(code_tree_sha256),
    }


def _trainer_index(command: Sequence[str]) -> int:
    positions = [
        index for index, token in enumerate(command) if Path(token).name == "train_bc.py"
    ]
    if len(positions) != 1:
        raise CanaryError("scratch command must name exactly one train_bc.py")
    return positions[0]


def _effective_recipe_from_command(command: Sequence[str]) -> dict[str, object]:
    parser = train_bc.build_parser()
    try:
        args = parser.parse_args(list(command)[_trainer_index(command) + 1 :])
    except SystemExit as error:
        raise CanaryError("cannot parse rendered topology canary command") from error
    effective = train_bc._effective_a1_learner_training_recipe(  # noqa: SLF001
        args,
        {"world_size": 8, "rank": 0, "local_rank": 0, "enabled": True},
    )
    expected = train_bc._a1_scratch_topology_expected_effective_recipe(  # noqa: SLF001
        source_recipe=current_science.learner_training_recipe(),
        source_model=current_science.learner_model_construction(),
        source_topology=current_science.learner_execution_topology(),
        max_steps=int(args.max_steps),
    )
    train_bc._bind_late_a1_recipe_fields(  # noqa: SLF001
        effective, args, expected
    )
    if str(args.a1_learner_ablation_id or ""):
        effective["per_game_value_weight_mode"] = str(args.per_game_value_weight_mode)
        if str(args.value_player_outcome_balance_mode) != "none":
            effective["value_player_outcome_balance_mode"] = str(
                args.value_player_outcome_balance_mode
            )
    return effective


def build_commands(
    verified: Mapping[str, Any],
    *,
    python: Path,
    output_dir: Path,
    max_steps: int,
) -> dict[str, list[str]]:
    build_arm_contracts(max_steps=max_steps)
    output_root = output_dir.expanduser().absolute()
    code_binding = _code_binding()
    commands: dict[str, list[str]] = {}
    for arm_id in ARM_IDS:
        checkpoint = output_root / arm_id / "candidate.pt"
        report = output_root / arm_id / "training-report.json"
        arm_verified = copy.deepcopy(dict(verified))
        recipe = dict(arm_verified["recipe"])
        checkpoint_steps = tuple(
            step
            for step in scratch._checkpoint_steps(recipe)  # noqa: SLF001
            if step < int(max_steps)
        )
        recipe["epochs"] = 1
        recipe["max_steps"] = int(max_steps)
        recipe["checkpoint_steps"] = ",".join(map(str, checkpoint_steps))
        arm_verified["recipe"] = recipe
        command = scratch.build_train_command(
            arm_verified, python=python, checkpoint=checkpoint, report=report
        )
        _replace_value(
            command, "--max-35m-params", CANARY_MAX_PARAMETER_COUNT
        )
        if "--exact-max-steps" not in command:
            command.append("--exact-max-steps")
        command.append(
            "--topology-residual-adapter"
            if arm_id == "T640"
            else "--no-topology-residual-adapter"
        )
        authority = diagnostic_authority(
            arm_id=arm_id,
            max_steps=max_steps,
            checkpoint_steps=checkpoint_steps,
            code_tree_sha256=str(code_binding["code_tree_sha256"]),
        )
        command.extend(
            (
                "--a1-learner-ablation-id",
                f"scratch-topology-{arm_id.lower()}",
                "--a1-scratch-diagnostic-authority-json",
                _canonical_bytes(authority).decode("ascii"),
            )
        )
        effective = _effective_recipe_from_command(command)
        command.extend(
            (
                "--a1-effective-learner-recipe-json",
                _canonical_bytes(effective).decode("ascii"),
                "--a1-effective-learner-recipe-sha256",
                _value_sha256(effective),
                "--a1-ablation-code-binding-json",
                _canonical_bytes(code_binding).decode("ascii"),
                "--a1-ablation-code-tree-sha256",
                str(code_binding["code_tree_sha256"]),
                "--a1-reviewed-lock-file-sha256",
                _file_sha256(Path(str(verified["lock_path"])).resolve(strict=True)),
            )
        )
        commands[arm_id] = command
    if normalized_matched_command(commands["C640"]) != normalized_matched_command(
        commands["T640"]
    ):
        raise CanaryError("arm commands differ outside declared topology/output fields")
    return commands


def normalized_matched_command(command: Sequence[str]) -> list[str]:
    """Remove only the declared arm/output fields from command comparison."""

    values = list(command)
    for flag in (
        "--checkpoint",
        "--report",
        "--a1-learner-ablation-id",
        "--a1-scratch-diagnostic-authority-json",
        "--a1-effective-learner-recipe-json",
        "--a1-effective-learner-recipe-sha256",
    ):
        _remove_pair(values, flag)
    for flag in (
        "--topology-residual-adapter",
        "--no-topology-residual-adapter",
    ):
        if flag in values:
            values.remove(flag)
    return values


def _h100_inventory(python: Path) -> list[dict[str, Any]]:
    script = """
import json
import torch

records = []
for index in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(index)
    records.append({
        "index": index,
        "uuid": str(getattr(properties, "uuid", "") or ""),
        "name": str(properties.name),
        "total_memory_bytes": int(properties.total_memory),
        "compute_capability": [int(properties.major), int(properties.minor)],
    })
print(json.dumps({"cuda_available": torch.cuda.is_available(), "records": records}))
"""
    try:
        result = subprocess.run(
            [str(python), "-c", script],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise CanaryError(
            f"cannot authenticate CUDA-visible H100 inventory: {error}"
        ) from error
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise CanaryError("authenticated Python returned malformed CUDA inventory") from error
    records = payload.get("records") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("cuda_available") is not True
        or not isinstance(records, list)
        or len(records) != WORLD_SIZE
        or any(
            not isinstance(record, dict)
            or set(record)
            != {
                "index",
                "uuid",
                "name",
                "total_memory_bytes",
                "compute_capability",
            }
            for record in records
        )
        or [record["index"] for record in records] != list(range(WORLD_SIZE))
        or len({record["uuid"] for record in records}) != WORLD_SIZE
        or any(
            not isinstance(record["uuid"], str) or not record["uuid"]
            for record in records
        )
        or any("H100" not in str(record["name"]).upper() for record in records)
        or any(record["compute_capability"] != [9, 0] for record in records)
        or any(
            isinstance(record["total_memory_bytes"], bool)
            or not isinstance(record["total_memory_bytes"], int)
            or record["total_memory_bytes"] <= 0
            for record in records
        )
    ):
        raise CanaryError(
            "execution requires exactly eight unique CUDA-visible H100 GPUs"
        )
    return records


def _finite_json_tree(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return True
    if isinstance(value, list):
        return all(_finite_json_tree(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _finite_json_tree(item)
            for key, item in value.items()
        )
    return False


def summarize_report(
    report: Path,
    *,
    max_steps: int,
    global_batch_size: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CanaryError(f"cannot read canary training report {report}: {error}") from error
    if not isinstance(payload, dict):
        raise CanaryError("canary training report root is not an object")
    required_counts = (
        "parameter_count",
        "trainable_parameter_count",
        "forward_active_parameter_count",
    )
    if any(
        isinstance(payload.get(key), bool) or not isinstance(payload.get(key), int)
        for key in required_counts
    ):
        raise CanaryError("canary report lacks exact parameter counts")
    total = int(payload["parameter_count"])
    trainable = int(payload["trainable_parameter_count"])
    forward_active = int(payload["forward_active_parameter_count"])
    if not (
        0 < total <= CANARY_MAX_PARAMETER_COUNT
        and 0 < trainable <= total
        and 0 < forward_active <= total
    ):
        raise CanaryError("canary report parameter counts exceed diagnostic bounds")
    observed_steps = payload.get("steps_completed")
    if (
        isinstance(observed_steps, bool)
        or not isinstance(observed_steps, int)
        or observed_steps != int(max_steps)
        or payload.get("max_steps") != int(max_steps)
        or payload.get("exact_max_steps") is not True
    ):
        raise CanaryError("canary report did not prove the exact optimizer-step dose")
    telemetry_fields = (
        "module_optimizer_observability",
        "feature_learning_signal_admission",
        "objective_gradient_signal_admission",
        "objective_gradient_interference",
    )
    if any(
        not isinstance(payload.get(key), dict)
        or not payload[key]
        or not _finite_json_tree(payload[key])
        for key in telemetry_fields
    ):
        raise CanaryError("canary report lacks finite structured gradient telemetry")
    recipe = current_science.learner_training_recipe()
    required_modules = tuple(
        sorted(
            name.strip()
            for name in str(
                recipe["require_feature_learning_signal_modules"]
            ).split(",")
            if name.strip()
        )
    )
    minimum_observations = int(
        recipe["minimum_feature_learning_signal_observations"]
    )
    try:
        feature_contract = feature_signal.contract_from_cli(
            module_names=required_modules,
            cadence_batches=int(recipe["train_diagnostics_every_batches"]),
            minimum_observations=minimum_observations,
        )
        verified_feature_signal = feature_signal.verify_observability(
            payload["module_optimizer_observability"],
            contract=feature_contract,
            where="H100 scratch canary report",
        )
        verified_objective_signal = feature_signal.verify_objective_interference(
            payload["objective_gradient_interference"],
            cadence_batches=int(
                recipe["objective_gradient_interference_every_batches"]
            ),
            minimum_observations=minimum_observations,
            expected_world_size=WORLD_SIZE,
            expected_value_trunk_grad_scale=float(
                recipe["value_trunk_grad_scale"]
            ),
            where="H100 scratch canary report",
        )
    except feature_signal.FeatureSignalError as error:
        raise CanaryError(
            f"canary report gradient telemetry failed admission: {error}"
        ) from error
    if (
        payload["feature_learning_signal_admission"] != verified_feature_signal
        or payload["objective_gradient_signal_admission"]
        != verified_objective_signal
    ):
        raise CanaryError("canary report gradient admission echo drift")
    seconds = float(elapsed_seconds)
    if not math.isfinite(seconds) or not seconds > 0.0:
        raise CanaryError("canary elapsed time must be finite and positive")
    return {
        "report": {"path": str(report.resolve()), "file_sha256": _file_sha256(report)},
        "parameter_counts": {
            "total": total,
            "trainable": trainable,
            "forward_active": forward_active,
        },
        "throughput": {
            "elapsed_seconds": seconds,
            "optimizer_steps": observed_steps,
            "optimizer_steps_per_second": float(observed_steps) / seconds,
            "global_batch_size": int(global_batch_size),
            "rows_per_second": float(observed_steps * global_batch_size) / seconds,
            "scope": "end_to_end_including_validation_and_report_write",
        },
        "gradient_telemetry": {
            key: payload[key] for key in telemetry_fields
        },
    }


def require_production_admission(_receipt: Mapping[str, Any]) -> None:
    """Permanent tripwire: canary evidence can never become an admission token."""

    raise CanaryError("H100 scratch canary receipts are never production-admissible")


def _write_failure_receipt(
    receipt: Path,
    *,
    base: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    stage: str,
    error: Exception,
    inventory: Sequence[Mapping[str, Any]],
    results: Mapping[str, Any],
    failed_arm: str | None,
    returncode: int | None,
) -> None:
    payload = {
        **base,
        "schema_version": EXECUTION_SCHEMA,
        "status": "failed",
        "failure_stage": stage,
        "error_type": type(error).__name__,
        "error": str(error),
        "failed_arm": failed_arm,
        "returncode": returncode,
        "plan_binding": dict(plan_binding),
        "gpu_inventory": [dict(record) for record in inventory],
        "results": dict(results),
    }
    _atomic_new_json(receipt, payload)


def run(
    args: argparse.Namespace,
    *,
    runner: Any = subprocess.run,
) -> dict[str, Any]:
    steps = _validate_step_bound(args.max_steps)
    receipt = args.receipt.expanduser().absolute()
    if receipt.exists() or receipt.is_symlink():
        raise CanaryError(f"refusing non-fresh receipt path: {receipt}")
    verified = scratch.verify_inputs(
        lock_path=args.lock,
        data_path=args.data,
        composite_build_receipt=args.composite_build_receipt,
    )
    python_ref = scratch._executable_ref(  # noqa: SLF001
        args.python, where="H100 canary Python"
    )
    arms = build_arm_contracts(max_steps=steps)
    commands = build_commands(
        verified,
        python=Path(str(python_ref["path"])),
        output_dir=args.output_dir,
        max_steps=steps,
    )
    base: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA,
        "created_unix_ns": time.time_ns(),
        "status": "planned",
        "diagnostic_only": True,
        "promotion_eligible": False,
        "production_admission": "forbidden",
        "maximum_result": "bounded_training_signal_evidence_only",
        "max_steps": steps,
        "exact_max_steps": True,
        "arms": arms,
        "only_declared_arm_delta": (
            "model_construction.topology_residual_adapter=false->true"
        ),
        "commands": commands,
        "command_sha256": {
            arm_id: _value_sha256(command) for arm_id, command in commands.items()
        },
        "python": python_ref,
        "launcher": {
            "path": str(Path(__file__).resolve()),
            "file_sha256": _file_sha256(Path(__file__).resolve()),
        },
        "production_scratch_authority": {
            "topology": current_science.learner_execution_topology(),
            "go_authorized": bool(
                current_science.learner_execution_topology()["go_authorized"]
            ),
            "unchanged": True,
        },
    }
    _seal_plan_identity(base)
    if not bool(args.go):
        if getattr(args, "plan_receipt", None) is not None:
            raise CanaryError("dry-run planning may not consume an earlier plan")
        _atomic_new_json(receipt, base)
        return base

    plan_path = getattr(args, "plan_receipt", None)
    if plan_path is None:
        raise CanaryError("--go requires --plan-receipt from a reviewed dry run")
    if plan_path.expanduser().absolute() == receipt:
        raise CanaryError("execution receipt must differ from immutable plan receipt")
    plan_binding = _load_bound_plan(plan_path, base)

    inventory: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    failed_arm: str | None = None
    returncode: int | None = None
    stage = "cuda_inventory"
    started_all = time.monotonic()
    try:
        inventory = _h100_inventory(Path(str(python_ref["path"])))
        output_root = args.output_dir.expanduser().absolute()
        stage = "output_directory_preflight"
        for arm_id in ARM_IDS:
            arm_dir = output_root / arm_id
            arm_dir.mkdir(parents=True, exist_ok=False)
        for arm_id in ARM_IDS:
            failed_arm = arm_id
            stage = f"{arm_id}_training"
            started = time.monotonic()
            completed = runner(commands[arm_id], cwd=REPO_ROOT, check=False)
            elapsed = time.monotonic() - started
            returncode = int(completed.returncode)
            if returncode != 0:
                raise CanaryError(
                    f"{arm_id} learner exited with return code {returncode}"
                )
            stage = f"{arm_id}_artifact_validation"
            report = output_root / arm_id / "training-report.json"
            checkpoint = output_root / arm_id / "candidate.pt"
            if not checkpoint.is_file() or checkpoint.is_symlink():
                raise CanaryError(
                    f"{arm_id} did not emit a regular terminal checkpoint"
                )
            results[arm_id] = {
                "checkpoint": {
                    "path": str(checkpoint.resolve()),
                    "file_sha256": _file_sha256(checkpoint),
                },
                **summarize_report(
                    report,
                    max_steps=steps,
                    global_batch_size=GLOBAL_BATCH_SIZE,
                    elapsed_seconds=elapsed,
                ),
            }
            returncode = None
        failed_arm = None
    except Exception as error:
        _write_failure_receipt(
            receipt,
            base=base,
            plan_binding=plan_binding,
            stage=stage,
            error=error,
            inventory=inventory,
            results=results,
            failed_arm=failed_arm,
            returncode=returncode,
        )
        if isinstance(error, CanaryError):
            raise
        raise CanaryError(f"H100 scratch canary failed during {stage}: {error}") from error
    completed_payload = {
        **base,
        "schema_version": EXECUTION_SCHEMA,
        "status": "completed",
        "plan_binding": plan_binding,
        "gpu_inventory": inventory,
        "elapsed_seconds": time.monotonic() - started_all,
        "results": results,
    }
    _atomic_new_json(receipt, completed_payload)
    return completed_payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--composite-build-receipt", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--plan-receipt", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--max-steps", type=int, choices=range(MIN_STEPS, MAX_STEPS + 1), default=128)
    parser.add_argument("--go", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        payload = run(parse_args(argv))
    except (CanaryError, scratch.ScratchTrainError, OSError, ValueError) as error:
        print(f"a1_h100_scratch_canary: {error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
