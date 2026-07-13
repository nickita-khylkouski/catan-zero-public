#!/usr/bin/env python3
"""Seal a matched topology-residual + target-gather commissioning sibling.

This is a deliberately thin specialization of the reviewed topology-only arm.
It changes only the allowlisted additive architecture module and its exact
trainable/optimizer surface.  Configuration is scoped to each call so importing
this module cannot mutate topology-only verification in another test or process.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import a1_function_preserving_upgrade as architecture_upgrade  # noqa: E402
from tools import a1_topology_only_composition_arm as base  # noqa: E402


SCHEMA = "a1-topology-target-gather-composition-arm-v1"
RECEIPT_SCHEMA = "a1-topology-target-gather-execution-receipt-v1"
STATUS_SCHEMA = "a1-topology-target-gather-execution-status-v1"
CLAIM_SCHEMA = "a1-topology-target-gather-execution-claim-v1"
EXECUTOR_RELATIVE_PATH = "tools/a1_topology_target_gather_composition_arm.py"
COMPLETION_RELATIVE_PATH = "tools/a1_topology_target_gather_composition_completion.py"
MANIFEST_NAME = "topology-target-gather-composition.manifest.json"

WORLD_SIZE = base.WORLD_SIZE
LOCAL_BATCH_SIZE = base.LOCAL_BATCH_SIZE
GLOBAL_BATCH_SIZE = base.GLOBAL_BATCH_SIZE
ALLOWED_OPTIMIZER_STEPS = base.ALLOWED_OPTIMIZER_STEPS
OPTIMIZER_STEPS = base.OPTIMIZER_STEPS
TRUNK_LR_MULT = 4.0
ACTION_MODULE_LR_MULT = 4.0
VALUE_LR_MULT = 1.0
FREEZE_MODULES = (
    "trunk_base,action_encoder,policy_head,value_heads,edge_policy,action_cross"
)
TRAINABLE_PREFIXES = ("topology_residual_adapter", "target_gather_proj")
TRAINABLE_PREFIX = ",".join(TRAINABLE_PREFIXES)
UPGRADE_MODULE = architecture_upgrade.MODULE_TOPOLOGY_TARGET_GATHER
EXPECTED_TOPOLOGY_PARAMETERS = tuple(
    sorted(
        architecture_upgrade.ALLOWLIST[UPGRADE_MODULE]["new_parameter_initialization"]
    )
)
EXPECTED_TOPOLOGY_PARAMETER_COUNT = 1_234_560
EXPECTED_PARAMETER_COUNTS = {
    "topology_residual_adapter": 823_040,
    "target_gather_proj": 411_520,
}
ONLY_DECLARED_MODEL_DELTA = (
    "train function-preserving topology_residual_adapter and target_gather_proj "
    "on a frozen exact selected parent"
)
ADAPTER_LR_CONTRACT = {
    "topology_lr": 3e-5 * TRUNK_LR_MULT,
    "target_gather_lr": 3e-5 * ACTION_MODULE_LR_MULT,
}
EFFECTIVE_TRAINABLE_OBJECTIVE = {
    "policy_loss_reaches_topology_adapter": True,
    "value_loss_reaches_topology_adapter": True,
    "policy_loss_reaches_target_gather": True,
    "all_inherited_policy_value_tensors_frozen": True,
}
TREATMENT_GEOMETRY_NAME = "treatment_topology_target_gather_commissioning"
TREATMENT_INTEGRATED_LR_CONTRACT = {
    "trunk_integrated_lr_step_equivalents": TRUNK_LR_MULT,
    "action_integrated_lr_step_equivalents": ACTION_MODULE_LR_MULT,
}
SOURCE_FILES = tuple(
    dict.fromkeys(
        (*base.SOURCE_FILES, EXECUTOR_RELATIVE_PATH, COMPLETION_RELATIVE_PATH)
    )
)

PARENT_SELECTION_SCHEMA = base.PARENT_SELECTION_SCHEMA
PARENT_DIRECT_SHORT_D6 = base.PARENT_DIRECT_SHORT_D6
PARENT_SELECTED_GATHER = base.PARENT_SELECTED_GATHER
PARENT_PROFILES = base.PARENT_PROFILES
TopologyCompositionError = base.TopologyCompositionError
executor_base = base.executor_base
gather_arm = base.gather_arm


_CONFIG = {
    "SCHEMA": SCHEMA,
    "RECEIPT_SCHEMA": RECEIPT_SCHEMA,
    "STATUS_SCHEMA": STATUS_SCHEMA,
    "CLAIM_SCHEMA": CLAIM_SCHEMA,
    "EXECUTOR_RELATIVE_PATH": EXECUTOR_RELATIVE_PATH,
    "COMPLETION_RELATIVE_PATH": COMPLETION_RELATIVE_PATH,
    "SOURCE_FILES": SOURCE_FILES,
    "TRUNK_LR_MULT": TRUNK_LR_MULT,
    "ACTION_MODULE_LR_MULT": ACTION_MODULE_LR_MULT,
    "VALUE_LR_MULT": VALUE_LR_MULT,
    "FREEZE_MODULES": FREEZE_MODULES,
    "TRAINABLE_PREFIX": TRAINABLE_PREFIX,
    "TRAINABLE_PREFIXES": TRAINABLE_PREFIXES,
    "EXPECTED_TOPOLOGY_PARAMETERS": EXPECTED_TOPOLOGY_PARAMETERS,
    "EXPECTED_TOPOLOGY_PARAMETER_COUNT": EXPECTED_TOPOLOGY_PARAMETER_COUNT,
    "EXPECTED_PARAMETER_COUNTS": EXPECTED_PARAMETER_COUNTS,
    "UPGRADE_MODULE": UPGRADE_MODULE,
    "MANIFEST_NAME": MANIFEST_NAME,
    "ONLY_DECLARED_MODEL_DELTA": ONLY_DECLARED_MODEL_DELTA,
    "ADAPTER_LR_CONTRACT": ADAPTER_LR_CONTRACT,
    "EFFECTIVE_TRAINABLE_OBJECTIVE": EFFECTIVE_TRAINABLE_OBJECTIVE,
    "TREATMENT_GEOMETRY_NAME": TREATMENT_GEOMETRY_NAME,
    "TREATMENT_INTEGRATED_LR_CONTRACT": TREATMENT_INTEGRATED_LR_CONTRACT,
}


@contextmanager
def _configured() -> Iterator[None]:
    previous = {name: getattr(base, name) for name in _CONFIG}
    previous_file = base.__file__
    try:
        for name, value in _CONFIG.items():
            setattr(base, name, value)
        # Parent-selection issuer and default executor identity are part of the
        # immutable artifact, so they must name this specialization.
        base.__file__ = __file__
        yield
    finally:
        base.__file__ = previous_file
        for name, value in previous.items():
            setattr(base, name, value)


def _call(name: str, *args: Any, **kwargs: Any) -> Any:
    with _configured():
        return getattr(base, name)(*args, **kwargs)


def _digest(value: Any) -> str:
    return base._digest(value)  # noqa: SLF001


def _file_ref(path: Path) -> dict[str, str]:
    return base._file_ref(path)  # noqa: SLF001


def _verify_ref(value: Any, *, label: str) -> Path:
    return base._verify_ref(value, label=label)  # noqa: SLF001


def _dose_geometry(optimizer_steps: int) -> dict[str, Any]:
    return _call("_dose_geometry", optimizer_steps)


def _derive_command(*args: Any, **kwargs: Any) -> Any:
    return _call("_derive_command", *args, **kwargs)


def _validate_upgrade_receipt(
    path: Path, *, parent_checkpoint: Mapping[str, Any]
) -> dict[str, Any]:
    return _call("_validate_upgrade_receipt", path, parent_checkpoint=parent_checkpoint)


def issue_parent_selection(**kwargs: Any) -> dict[str, Any]:
    return _call("issue_parent_selection", **kwargs)


def verify_parent_selection(path: Path) -> dict[str, Any]:
    return _call("verify_parent_selection", path)


def prepare(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    return _call("prepare", args)


def verify(
    manifest_path: Path,
    *,
    expected_executor: Path | None = None,
    require_fresh_outputs: bool = True,
) -> dict[str, Any]:
    return _call(
        "verify",
        manifest_path,
        expected_executor=expected_executor,
        require_fresh_outputs=require_fresh_outputs,
    )


def execute(manifest_path: Path, *, unit: str, runner: Any = None) -> dict[str, Any]:
    kwargs = {"unit": unit}
    if runner is not None:
        kwargs["runner"] = runner
    return _call("execute", manifest_path, **kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    with _configured():
        return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
