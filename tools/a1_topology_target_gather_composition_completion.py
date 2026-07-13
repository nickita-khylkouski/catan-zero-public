#!/usr/bin/env python3
"""Finalize the matched topology-residual + target-gather sibling.

The shared completion proof is reused under a scoped arm binding.  This module
adds the one treatment-specific check: fresh Adam must contain exactly four
action-local gather states and eight trunk topology states, both at LR 1.2e-4
and every state at the completed step.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Iterator, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import a1_topology_only_composition_completion as base  # noqa: E402
from tools import a1_topology_target_gather_composition_arm as arm  # noqa: E402


SCHEMA = "a1-topology-target-gather-composition-completion-v1"
STATUS = base.STATUS
COMPLETION_NAME = base.COMPLETION_NAME
EXPECTED_CHANGED_PARAMETERS = arm.EXPECTED_TOPOLOGY_PARAMETERS
CompletionError = base.CompletionError


def _optimizer_step(raw: Any) -> int:
    try:
        return int(raw.item()) if hasattr(raw, "item") else int(raw)
    except (TypeError, ValueError, RuntimeError) as error:
        raise CompletionError(
            "topology+gather optimizer state lacks a scalar step"
        ) from error


def _verify_optimizer_groups(path: Path, *, optimizer_steps: int) -> dict[str, Any]:
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ModuleNotFoundError) as error:
        raise CompletionError(
            f"cannot load topology+gather optimizer: {error}"
        ) from error
    optimizer = payload.get("optimizer") if isinstance(payload, Mapping) else None
    groups = optimizer.get("param_groups") if isinstance(optimizer, Mapping) else None
    state = optimizer.get("state") if isinstance(optimizer, Mapping) else None
    if not (
        payload.get("format") == "plain"
        and isinstance(groups, list)
        and len(groups) == 3
        and isinstance(state, Mapping)
    ):
        raise CompletionError("topology+gather optimizer envelope/group count drift")
    base_group, action_group, trunk_group = groups
    action_parameters = (
        action_group.get("params") if isinstance(action_group, Mapping) else None
    )
    topology_parameters = (
        trunk_group.get("params") if isinstance(trunk_group, Mapping) else None
    )
    if not (
        isinstance(base_group, Mapping)
        and isinstance(action_group, Mapping)
        and isinstance(trunk_group, Mapping)
        and base_group.get("lr") == 3e-5
        and base_group.get("base_lr") == 3e-5
        and base_group.get("params") == []
        and action_group.get("lr") == 1.2e-4
        and action_group.get("base_lr") == 1.2e-4
        and trunk_group.get("lr") == 1.2e-4
        and trunk_group.get("base_lr") == 1.2e-4
        and isinstance(action_parameters, list)
        and len(action_parameters) == 4
        and isinstance(topology_parameters, list)
        and len(topology_parameters) == 8
        and set(state) == set(action_parameters) | set(topology_parameters)
        and not (set(action_parameters) & set(topology_parameters))
    ):
        raise CompletionError(
            "optimizer does not isolate gather=4 and topology=8 tensors at LR=1.2e-4"
        )
    parameter_ids = [*action_parameters, *topology_parameters]
    observed_steps: list[int] = []
    for parameter_id in parameter_ids:
        parameter_state = state.get(parameter_id)
        if not isinstance(parameter_state, Mapping):
            raise CompletionError("topology+gather optimizer state is malformed")
        observed_steps.append(_optimizer_step(parameter_state.get("step")))
        for moment in ("exp_avg", "exp_avg_sq"):
            tensor = parameter_state.get(moment)
            if tensor is None or not bool(torch.isfinite(tensor).all()):
                raise CompletionError(
                    f"topology+gather optimizer has missing/non-finite {moment}"
                )
    if observed_steps != [optimizer_steps] * 12:
        raise CompletionError(
            "topology+gather optimizer state step does not match completed dose: "
            f"expected={optimizer_steps} observed={observed_steps}"
        )
    return {
        "format": "plain",
        "base_group_parameter_tensors": 0,
        "base_group_lr": 3e-5,
        "action_group_parameter_tensors": 4,
        "action_group_lr": 1.2e-4,
        "trunk_group_parameter_tensors": 8,
        "trunk_group_lr": 1.2e-4,
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


def _call(name: str, *args: Any, **kwargs: Any) -> Any:
    with _configured():
        return getattr(base, name)(*args, **kwargs)


def verify_manifest(manifest_path: Path) -> dict[str, Any]:
    return _call("verify_manifest", manifest_path)


def _profile_metric(profile: Mapping[str, Any], *keys: str) -> float:
    value: Any = profile
    for key in keys:
        value = value.get(key) if isinstance(value, Mapping) else None
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise CompletionError(f"invalid inference metric {'.'.join(keys)}={value!r}")
    return float(value)


def _load_profile(
    path: Path, *, checkpoint: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, str]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CompletionError(
            f"cannot load inference profile {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise CompletionError("inference profile must be a JSON object")
    try:
        observed_checkpoint = Path(str(value.get("checkpoint", ""))).resolve(
            strict=True
        )
        expected_checkpoint = Path(str(checkpoint["path"])).resolve(strict=True)
    except (OSError, KeyError) as error:
        raise CompletionError(
            f"inference profile checkpoint is unavailable: {error}"
        ) from error
    if observed_checkpoint != expected_checkpoint or arm._file_ref(  # noqa: SLF001
        observed_checkpoint
    ) != dict(checkpoint):
        raise CompletionError(
            "inference profile does not bind the exact checkpoint bytes"
        )
    parity = value.get("exact_vs_attributed_output_parity")
    if not (
        isinstance(parity, Mapping)
        and parity
        and all(
            isinstance(row, Mapping)
            and row.get("max_abs") == 0.0
            and row.get("mean_abs") == 0.0
            for row in parity.values()
        )
    ):
        raise CompletionError("inference profile lacks bit-exact attributed parity")
    return value, arm._file_ref(path)  # noqa: SLF001


def _inference_cost_telemetry(
    verified: Mapping[str, Any], *, candidate: Mapping[str, Any]
) -> dict[str, Any]:
    manifest = verified["manifest"]
    contract = manifest.get("inference_cost_contract")
    if (
        not isinstance(contract, Mapping)
        or contract.get("required_before_completion") is not True
    ):
        raise CompletionError(
            "combined sibling lacks mandatory inference-cost contract"
        )
    root = Path(verified["output_root"])
    reference_checkpoint = contract["reference_checkpoint"]
    reference, reference_ref = _load_profile(
        root / "reference-inference-profile.json", checkpoint=reference_checkpoint
    )
    treatment, treatment_ref = _load_profile(
        root / "candidate-inference-profile.json", checkpoint=candidate
    )
    expected_shape = contract["matched_shape"]
    exact_environment = {
        "device": reference.get("device"),
        "strict_fp32": {
            "matmul_precision": "highest",
            "cuda_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "autocast": False,
        },
        "shape": {
            "batch_size": expected_shape["batch_size"],
            "legal_width": expected_shape["legal_width"],
            "event_width": expected_shape["event_width"],
            "valid_players": expected_shape["valid_players"],
        },
        "warmup": expected_shape["warmup"],
        "iterations": expected_shape["iterations"],
        "return_q": expected_shape["return_q"],
    }
    for label, profile in (("reference", reference), ("candidate", treatment)):
        observed = {
            "device": profile.get("device"),
            "strict_fp32": profile.get("strict_fp32"),
            "shape": {
                key: profile.get("shape", {}).get(key)
                for key in ("batch_size", "legal_width", "event_width", "valid_players")
            },
            "warmup": profile.get("warmup"),
            "iterations": profile.get("iterations"),
            "return_q": profile.get("return_q"),
        }
        if observed != exact_environment:
            raise CompletionError(
                f"{label} inference profile environment drift: {observed}"
            )
    metric_paths = {
        "cuda_mean_ms": ("exact_window", "cuda_ms", "mean"),
        "cuda_median_ms": ("exact_window", "cuda_ms", "median"),
        "cuda_p95_ms": ("exact_window", "cuda_ms", "p95"),
        "wall_mean_ms": ("exact_window", "wall_ms", "mean"),
        "wall_median_ms": ("exact_window", "wall_ms", "median"),
        "wall_p95_ms": ("exact_window", "wall_ms", "p95"),
    }
    reference_metrics = {
        name: _profile_metric(reference, *path) for name, path in metric_paths.items()
    }
    candidate_metrics = {
        name: _profile_metric(treatment, *path) for name, path in metric_paths.items()
    }
    ratios = {
        name.removesuffix("_ms") + "_slowdown": candidate_metrics[name] / value
        for name, value in reference_metrics.items()
    }
    telemetry = {
        "schema_version": "a1-architecture-inference-cost-telemetry-v1",
        "contract": dict(contract),
        "reference_checkpoint": dict(reference_checkpoint),
        "candidate_checkpoint": dict(candidate),
        "reference_profile": reference_ref,
        "candidate_profile": treatment_ref,
        "matched_environment": exact_environment,
        "reference_metrics": reference_metrics,
        "candidate_metrics": candidate_metrics,
        "candidate_reference_ratios": ratios,
        "selection_cost_observed": True,
    }
    telemetry["telemetry_sha256"] = arm._digest(telemetry)  # noqa: SLF001
    return telemetry


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
        raise CompletionError(f"combined completion already exists: {path}") from error
    return payload


def verify_completion(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CompletionError(f"cannot load combined completion: {error}") from error
    if not isinstance(receipt, dict):
        raise CompletionError("combined completion is not a JSON object")
    unhashed = dict(receipt)
    stated = unhashed.pop("receipt_sha256", None)
    if not (
        receipt.get("schema_version") == SCHEMA
        and receipt.get("status") == STATUS
        and receipt.get("diagnostic_only") is True
        and receipt.get("promotion_eligible") is False
        and stated == arm._digest(unhashed)  # noqa: SLF001
        and receipt.get("completion_finalizer") == arm._file_ref(Path(__file__))  # noqa: SLF001
    ):
        raise CompletionError(
            "combined completion schema/status/finalizer/digest drift"
        )
    replay = build_completion(
        Path(receipt["manifest"]["path"]),
        expected_checkpoint_sha256=str(receipt["expected_checkpoint_sha256"]),
        unit_state=receipt["unit_state"],
        created_at_unix_ns=int(receipt["created_at_unix_ns"]),
    )
    if replay != receipt:
        raise CompletionError("combined completion replay differs from receipt")
    if path != Path(replay["checkpoint"]["path"]).parent / COMPLETION_NAME:
        raise CompletionError("combined completion escaped output root")
    return receipt


def _verify_topology_target_gather_delta(
    initializer: Path, candidate: Path
) -> dict[str, Any]:
    return _call("_verify_topology_only_delta", initializer, candidate)


def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch through this specialization, not the topology-only sibling.

    Calling ``base.main`` here is subtly wrong: the parser is reusable, but the
    function globals resolved by ``base.main`` are the topology-only
    ``finalize`` and ``verify_completion`` functions.  That bypasses this
    module's mandatory inference-cost telemetry while still producing a
    superficially valid topology-only receipt.
    """
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
