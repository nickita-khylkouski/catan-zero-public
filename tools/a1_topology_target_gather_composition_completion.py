#!/usr/bin/env python3
"""Finalize the matched topology-residual + target-gather sibling.

The shared completion proof is reused under a scoped arm binding.  This module
adds the one treatment-specific check: fresh Adam must contain exactly four
action-local gather states and eight trunk topology states, both at LR 1.2e-4
and every state at the completed step.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys
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


def build_completion(
    manifest_path: Path,
    *,
    expected_checkpoint_sha256: str,
    unit_state: Mapping[str, Any],
    created_at_unix_ns: int,
) -> dict[str, Any]:
    return _call(
        "build_completion",
        manifest_path,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        unit_state=unit_state,
        created_at_unix_ns=created_at_unix_ns,
    )


def finalize(
    manifest_path: Path,
    *,
    expected_checkpoint_sha256: str,
    state_reader: Callable[..., str] = subprocess.check_output,
) -> dict[str, Any]:
    return _call(
        "finalize",
        manifest_path,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        state_reader=state_reader,
    )


def verify_completion(path: Path) -> dict[str, Any]:
    return _call("verify_completion", path)


def _verify_topology_target_gather_delta(
    initializer: Path, candidate: Path
) -> dict[str, Any]:
    return _call("_verify_topology_only_delta", initializer, candidate)


def main(argv: Sequence[str] | None = None) -> int:
    with _configured():
        return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
