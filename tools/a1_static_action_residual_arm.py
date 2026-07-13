#!/usr/bin/env python3
"""Seal the exact-parent static-action dead-input commissioning arm.

This is a thin, scoped specialization of the reviewed selected-dose additive
architecture runner. It changes only the function-preserving static catalog
residual, freezes every inherited tensor, and trains the two new action-local
tensors for the authenticated 128-step/524,288-draw D6 geometry.
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


SCHEMA = "a1-static-action-residual-arm-v1"
RECEIPT_SCHEMA = "a1-static-action-residual-execution-receipt-v1"
STATUS_SCHEMA = "a1-static-action-residual-execution-status-v1"
CLAIM_SCHEMA = "a1-static-action-residual-execution-claim-v1"
EXECUTOR_RELATIVE_PATH = "tools/a1_static_action_residual_arm.py"
COMPLETION_RELATIVE_PATH = "tools/a1_static_action_residual_completion.py"
MANIFEST_NAME = "static-action-residual.manifest.json"

WORLD_SIZE = base.WORLD_SIZE
LOCAL_BATCH_SIZE = base.LOCAL_BATCH_SIZE
GLOBAL_BATCH_SIZE = base.GLOBAL_BATCH_SIZE
ALLOWED_OPTIMIZER_STEPS = base.ALLOWED_OPTIMIZER_STEPS
OPTIMIZER_STEPS = base.OPTIMIZER_STEPS
TRUNK_LR_MULT = 1.0
ACTION_MODULE_LR_MULT = 4.0
VALUE_LR_MULT = 1.0
FREEZE_MODULES = (
    "trunk,action_encoder,policy_head,value_heads,target_gather,"
    "edge_policy,action_cross"
)
TRAINABLE_PREFIXES = ("static_action_residual_proj",)
TRAINABLE_PREFIX = TRAINABLE_PREFIXES[0]
UPGRADE_MODULE = architecture_upgrade.MODULE_STATIC_ACTION_RESIDUAL
EXPECTED_TOPOLOGY_PARAMETERS = tuple(
    sorted(architecture_upgrade.ALLOWLIST[UPGRADE_MODULE]["new_parameter_initialization"])
)
EXPECTED_TOPOLOGY_PARAMETER_COUNT = 14_720
EXPECTED_PARAMETER_COUNTS = {TRAINABLE_PREFIX: EXPECTED_TOPOLOGY_PARAMETER_COUNT}
ONLY_DECLARED_MODEL_DELTA = (
    "train function-preserving static_action_residual_proj on frozen exact "
    "selected parent"
)
ADAPTER_LR_CONTRACT = {"static_action_residual_lr": 3e-5 * ACTION_MODULE_LR_MULT}
EFFECTIVE_TRAINABLE_OBJECTIVE = {
    "policy_loss_reaches_static_action_residual": True,
    "value_loss_reaches_static_action_residual": False,
    "all_inherited_policy_value_tensors_frozen": True,
}
REPORT_ARCHITECTURE_DELTA = {
    "topology_residual_adapter": False,
    "static_action_residual": True,
}
TREATMENT_GEOMETRY_NAME = "treatment_static_action_residual_commissioning"
TREATMENT_INTEGRATED_LR_CONTRACT = {
    "action_integrated_lr_step_equivalents": ACTION_MODULE_LR_MULT,
}
INFERENCE_COST_CONTRACT = {
    "schema_version": "a1-architecture-inference-cost-contract-v1",
    "required_before_completion": True,
    "benchmark_tool": "tools/bench_entity_graph_stages.py",
    "measurement_boundary": "forward_legal_np_plus_eval_server_d2h",
    "strict_fp32": True,
    "matched_shape": {
        "batch_size": 48,
        "legal_width": 54,
        "event_width": 0,
        "valid_players": 2,
        "return_q": True,
        "warmup": 20,
        "iterations": 100,
    },
    "required_metrics": [
        "exact_window.cuda_ms.mean",
        "exact_window.cuda_ms.median",
        "exact_window.cuda_ms.p95",
        "exact_window.wall_ms.mean",
        "exact_window.wall_ms.median",
        "exact_window.wall_ms.p95",
    ],
    "selection_semantics": (
        "strength is adjudicated with candidate/reference inference cost; "
        "architecture selection may not ignore latency"
    ),
}
SOURCE_FILES = tuple(
    dict.fromkeys(
        (
            *base.SOURCE_FILES,
            EXECUTOR_RELATIVE_PATH,
            COMPLETION_RELATIVE_PATH,
            "tools/bench_entity_graph_stages.py",
            "tools/a1_topology_target_gather_composition_arm.py",
            "tools/a1_topology_target_gather_composition_completion.py",
            "src/catan_zero/search/cuda_graph_inference.py",
        )
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
    "REPORT_ARCHITECTURE_DELTA": REPORT_ARCHITECTURE_DELTA,
    "TREATMENT_GEOMETRY_NAME": TREATMENT_GEOMETRY_NAME,
    "TREATMENT_INTEGRATED_LR_CONTRACT": TREATMENT_INTEGRATED_LR_CONTRACT,
    "INFERENCE_COST_CONTRACT": INFERENCE_COST_CONTRACT,
}


@contextmanager
def _configured() -> Iterator[None]:
    previous = {name: getattr(base, name) for name in _CONFIG}
    previous_file = base.__file__
    try:
        for name, value in _CONFIG.items():
            setattr(base, name, value)
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
