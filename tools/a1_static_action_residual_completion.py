#!/usr/bin/env python3
"""Finalize and replay the exact-parent static-action residual diagnostic."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Iterator, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import a1_static_action_residual_arm as arm  # noqa: E402
from tools import a1_topology_only_composition_completion as base  # noqa: E402
from tools import a1_topology_target_gather_composition_completion as cost  # noqa: E402


SCHEMA = "a1-static-action-residual-completion-v1"
STATUS = base.STATUS
COMPLETION_NAME = base.COMPLETION_NAME
EXPECTED_CHANGED_PARAMETERS = arm.EXPECTED_TOPOLOGY_PARAMETERS
CompletionError = base.CompletionError


def _optimizer_step(raw: Any) -> int:
    try:
        return int(raw.item()) if hasattr(raw, "item") else int(raw)
    except (TypeError, ValueError, RuntimeError) as error:
        raise CompletionError(
            "static-action optimizer state lacks a scalar step"
        ) from error


def _verify_optimizer_groups(path: Path, *, optimizer_steps: int) -> dict[str, Any]:
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ModuleNotFoundError) as error:
        raise CompletionError(f"cannot load static-action optimizer: {error}") from error
    optimizer = payload.get("optimizer") if isinstance(payload, Mapping) else None
    groups = optimizer.get("param_groups") if isinstance(optimizer, Mapping) else None
    state = optimizer.get("state") if isinstance(optimizer, Mapping) else None
    if not (
        payload.get("format") == "plain"
        and isinstance(groups, list)
        and len(groups) == 2
        and isinstance(state, Mapping)
    ):
        raise CompletionError("static-action optimizer envelope/group count drift")
    base_group, action_group = groups
    action_parameters = (
        action_group.get("params") if isinstance(action_group, Mapping) else None
    )
    if not (
        isinstance(base_group, Mapping)
        and isinstance(action_group, Mapping)
        and base_group.get("lr") == 3e-5
        and base_group.get("base_lr") == 3e-5
        and base_group.get("params") == []
        and action_group.get("lr") == 1.2e-4
        and action_group.get("base_lr") == 1.2e-4
        and isinstance(action_parameters, list)
        and len(action_parameters) == len(EXPECTED_CHANGED_PARAMETERS) == 2
        and set(state) == set(action_parameters)
    ):
        raise CompletionError(
            "static-action optimizer does not isolate two LR=1.2e-4 tensors"
        )
    observed_steps: list[int] = []
    for parameter_id in action_parameters:
        parameter_state = state.get(parameter_id)
        if not isinstance(parameter_state, Mapping):
            raise CompletionError("static-action optimizer state is malformed")
        observed_steps.append(_optimizer_step(parameter_state.get("step")))
        for moment in ("exp_avg", "exp_avg_sq"):
            tensor = parameter_state.get(moment)
            if tensor is None or not bool(torch.isfinite(tensor).all()):
                raise CompletionError(
                    f"static-action optimizer has missing/non-finite {moment}"
                )
    if observed_steps != [optimizer_steps] * len(action_parameters):
        raise CompletionError(
            "static-action optimizer step does not match completed dose: "
            f"expected={optimizer_steps} observed={observed_steps}"
        )
    return {
        "format": "plain",
        "base_group_parameter_tensors": 0,
        "base_group_lr": 3e-5,
        "action_group_parameter_tensors": len(action_parameters),
        "action_group_lr": 1.2e-4,
        "optimizer_state_tensors": len(state),
        "optimizer_state_step": optimizer_steps,
    }


@contextmanager
def _configured() -> Iterator[None]:
    previous = {
        "arm": base.arm,
        "SCHEMA": base.SCHEMA,
        "EXPECTED_CHANGED_PARAMETERS": base.EXPECTED_CHANGED_PARAMETERS,
        "__file__": base.__file__,
        "_verify_optimizer_groups": base._verify_optimizer_groups,  # noqa: SLF001
    }
    try:
        base.arm = arm
        base.SCHEMA = SCHEMA
        base.EXPECTED_CHANGED_PARAMETERS = EXPECTED_CHANGED_PARAMETERS
        base.__file__ = __file__
        base._verify_optimizer_groups = _verify_optimizer_groups  # noqa: SLF001
        yield
    finally:
        for name, value in previous.items():
            setattr(base, name, value)


@contextmanager
def _cost_configured() -> Iterator[None]:
    previous_arm = cost.arm
    try:
        cost.arm = arm
        yield
    finally:
        cost.arm = previous_arm


def _call(name: str, *args: Any, **kwargs: Any) -> Any:
    with _configured():
        return getattr(base, name)(*args, **kwargs)


def _rich_file_ref(path: Path) -> dict[str, Any]:
    """Match the size-bound file-reference schema emitted by the base finalizer."""
    resolved = path.expanduser().resolve(strict=True)
    return {
        **arm._file_ref(resolved),  # noqa: SLF001
        "size_bytes": resolved.stat().st_size,
    }


def verify_manifest(manifest_path: Path) -> dict[str, Any]:
    return _call("verify_manifest", manifest_path)


def _inference_cost_telemetry(
    verified: Mapping[str, Any], *, candidate: Mapping[str, Any]
) -> dict[str, Any]:
    with _cost_configured():
        return cost._inference_cost_telemetry(verified, candidate=candidate)  # noqa: SLF001


def build_completion(
    manifest_path: Path,
    *,
    expected_checkpoint_sha256: str,
    unit_state: Mapping[str, Any],
    created_at_unix_ns: int,
) -> dict[str, Any]:
    payload = _call(
        "build_completion",
        manifest_path,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        unit_state=unit_state,
        created_at_unix_ns=created_at_unix_ns,
    )
    payload.pop("receipt_sha256", None)
    verified = verify_manifest(manifest_path)
    payload["inference_cost_telemetry"] = _inference_cost_telemetry(
        verified, candidate=payload["checkpoint"]
    )
    payload["receipt_sha256"] = arm._digest(payload)  # noqa: SLF001
    return payload


def finalize(
    manifest_path: Path,
    *,
    expected_checkpoint_sha256: str,
    state_reader: Callable[..., str] = subprocess.check_output,
) -> dict[str, Any]:
    verified = verify_manifest(manifest_path)
    unit, _ = _call("_verify_submission", verified)
    state = _call("_read_live_unit_state", unit, state_reader=state_reader)
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
        raise CompletionError(f"static-action completion already exists: {path}") from error
    return payload


def verify_completion(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CompletionError(f"cannot load static-action completion: {error}") from error
    if not isinstance(receipt, dict):
        raise CompletionError("static-action completion is not a JSON object")
    unhashed = dict(receipt)
    stated = unhashed.pop("receipt_sha256", None)
    if not (
        receipt.get("schema_version") == SCHEMA
        and receipt.get("status") == STATUS
        and receipt.get("diagnostic_only") is True
        and receipt.get("promotion_eligible") is False
        and stated == arm._digest(unhashed)  # noqa: SLF001
        and receipt.get("completion_finalizer") == _rich_file_ref(Path(__file__))
    ):
        raise CompletionError(
            "static-action completion schema/status/finalizer/digest drift"
        )
    replay = build_completion(
        Path(receipt["manifest"]["path"]),
        expected_checkpoint_sha256=str(receipt["expected_checkpoint_sha256"]),
        unit_state=receipt["unit_state"],
        created_at_unix_ns=int(receipt["created_at_unix_ns"]),
    )
    if replay != receipt:
        raise CompletionError("static-action completion replay differs from receipt")
    if path != Path(replay["checkpoint"]["path"]).parent / COMPLETION_NAME:
        raise CompletionError("static-action completion escaped output root")
    return receipt


def _verify_static_action_delta(initializer: Path, candidate: Path) -> dict[str, Any]:
    return _call("_verify_topology_only_delta", initializer, candidate)


def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch through this specialization so cost telemetry cannot be bypassed."""
    with _configured():
        args = base.build_parser().parse_args(argv)
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
    raise SystemExit(main())
