#!/usr/bin/env python3
"""Fail-closed admission for commissioned learner feature modules."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


CONTRACT_SCHEMA = "a1-feature-learning-signal-admission-v1"
OBSERVABILITY_SCHEMA = "module-optimizer-observability-v1"
POSITIVE_SIGNAL_FIELDS = (
    "mean_pre_clip_grad_norm",
    "max_pre_clip_grad_norm",
    "mean_parameter_delta_norm",
    "mean_parameter_update_rms",
)


class FeatureSignalError(RuntimeError):
    """Required feature-learning evidence is missing or non-positive."""


def _positive_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) and parsed > 0.0 else None


def validate_contract(contract: object) -> dict[str, Any]:
    if not isinstance(contract, Mapping):
        raise FeatureSignalError("feature learning-signal contract is missing")
    expected_keys = {
        "schema_version",
        "cadence_batches",
        "minimum_observations",
        "norm_scope",
        "required_modules",
    }
    if set(contract) != expected_keys:
        raise FeatureSignalError(
            "feature learning-signal contract shape drifted: "
            f"missing={sorted(expected_keys - set(contract))}, "
            f"extra={sorted(set(contract) - expected_keys)}"
        )
    modules = contract.get("required_modules")
    if (
        contract.get("schema_version") != CONTRACT_SCHEMA
        or isinstance(contract.get("cadence_batches"), bool)
        or not isinstance(contract.get("cadence_batches"), int)
        or int(contract["cadence_batches"]) <= 0
        or isinstance(contract.get("minimum_observations"), bool)
        or not isinstance(contract.get("minimum_observations"), int)
        or int(contract["minimum_observations"]) <= 0
        or contract.get("norm_scope") != "global_replicated"
        or not isinstance(modules, list)
        or not modules
        or any(not isinstance(name, str) or not name for name in modules)
        or modules != sorted(set(modules))
    ):
        raise FeatureSignalError("feature learning-signal contract is malformed")
    return {
        "schema_version": CONTRACT_SCHEMA,
        "cadence_batches": int(contract["cadence_batches"]),
        "minimum_observations": int(contract["minimum_observations"]),
        "norm_scope": "global_replicated",
        "required_modules": list(modules),
    }


def verify_observability(
    observability: object,
    *,
    contract: object,
    where: str,
) -> dict[str, Any]:
    required = validate_contract(contract)
    if not isinstance(observability, Mapping):
        raise FeatureSignalError(f"{where} has no module optimizer observability")
    if (
        observability.get("schema_version") != OBSERVABILITY_SCHEMA
        or observability.get("norm_scope") != required["norm_scope"]
        or observability.get("cadence_batches") != required["cadence_batches"]
        or isinstance(observability.get("observed_steps"), bool)
        or not isinstance(observability.get("observed_steps"), int)
        or int(observability["observed_steps"])
        < required["minimum_observations"]
    ):
        raise FeatureSignalError(
            f"{where} lacks the contract-bound feature observation cadence"
        )
    modules = observability.get("modules")
    if not isinstance(modules, Mapping):
        raise FeatureSignalError(f"{where} has no feature module observations")

    failures: dict[str, object] = {}
    admitted: dict[str, dict[str, float | int]] = {}
    for module_name in required["required_modules"]:
        row = modules.get(module_name)
        if not isinstance(row, Mapping):
            failures[module_name] = "missing"
            continue
        failed_fields = [
            field
            for field in POSITIVE_SIGNAL_FIELDS
            if _positive_float(row.get(field)) is None
        ]
        parameter_count = row.get("parameter_count")
        if (
            isinstance(parameter_count, bool)
            or not isinstance(parameter_count, int)
            or parameter_count <= 0
        ):
            failed_fields.append("parameter_count")
        if failed_fields:
            failures[module_name] = failed_fields
            continue
        admitted[module_name] = {
            field: float(row[field]) for field in POSITIVE_SIGNAL_FIELDS
        } | {"parameter_count": int(parameter_count)}

    if failures:
        raise FeatureSignalError(
            f"{where} did not demonstrate positive commissioned feature "
            f"gradients and updates: {failures}"
        )
    return {
        "schema_version": CONTRACT_SCHEMA,
        "authenticated": True,
        "observed_steps": int(observability["observed_steps"]),
        "cadence_batches": required["cadence_batches"],
        "norm_scope": required["norm_scope"],
        "required_modules": required["required_modules"],
        "positive_signal_fields": list(POSITIVE_SIGNAL_FIELDS),
        "modules": admitted,
    }


def verify_objective_interference(
    payload: object,
    *,
    cadence_batches: int,
    minimum_observations: int,
    expected_world_size: int,
    expected_value_trunk_grad_scale: float,
    where: str,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise FeatureSignalError(f"{where} has no objective-gradient observations")
    observations = payload.get("observations")
    if (
        payload.get("schema_version")
        != "objective-gradient-dose-observations-v1"
        or payload.get("cadence_batches") != int(cadence_batches)
        or isinstance(payload.get("observed_steps"), bool)
        or not isinstance(payload.get("observed_steps"), int)
        or not isinstance(observations, list)
        or payload["observed_steps"] != len(observations)
        or len(observations) < int(minimum_observations)
    ):
        raise FeatureSignalError(
            f"{where} lacks the contract-bound objective-gradient cadence"
        )

    distributed = int(expected_world_size) > 1
    expected_scope = (
        "global_ddp_microbatch" if distributed else "single_process_microbatch"
    )
    expected_aggregation = (
        "manual_all_reduce_then_world_average_of_ddp_scaled_gradients"
        if distributed
        else "single_process_exact_gradient"
    )
    positive_fields = (
        "policy_trunk_grad_norm",
        "value_trunk_grad_norm",
        "value_to_policy_grad_norm_ratio",
        "combined_trunk_grad_norm",
    )
    bounded_fields = {
        "trunk_gradient_cosine": (-1.0, 1.0),
        "opposing_coordinate_fraction": (0.0, 1.0),
    }
    failures: dict[int, object] = {}
    selected: list[dict[str, object]] = []
    steps: list[int] = []
    for index, observation in enumerate(observations):
        if not isinstance(observation, Mapping):
            failures[index] = "malformed"
            continue
        failed_fields: list[str] = []
        step = observation.get("optimizer_step")
        if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
            failed_fields.append("optimizer_step")
        else:
            steps.append(step)
        if observation.get("available") is not True:
            failed_fields.append("available")
        if observation.get("scope") != expected_scope:
            failed_fields.append("scope")
        if observation.get("aggregation") != expected_aggregation:
            failed_fields.append("aggregation")
        if observation.get("world_size") != int(expected_world_size):
            failed_fields.append("world_size")
        scale = observation.get("scalar_value_trunk_grad_scale")
        if (
            isinstance(scale, bool)
            or not isinstance(scale, (int, float))
            or not math.isfinite(float(scale))
            or float(scale) != float(expected_value_trunk_grad_scale)
        ):
            failed_fields.append("scalar_value_trunk_grad_scale")
        for field in positive_fields:
            if _positive_float(observation.get(field)) is None:
                failed_fields.append(field)
        for field, (lower, upper) in bounded_fields.items():
            value = observation.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not lower <= float(value) <= upper
            ):
                failed_fields.append(field)
        if failed_fields:
            failures[index] = failed_fields
        else:
            selected.append(
                {
                    "optimizer_step": int(step),
                    **{
                        field: float(observation[field])
                        for field in (*positive_fields, *bounded_fields)
                    },
                }
            )
    if failures or steps != sorted(set(steps)):
        raise FeatureSignalError(
            f"{where} objective-gradient evidence is invalid: "
            f"failures={failures}, steps={steps}"
        )
    return {
        "schema_version": "a1-objective-gradient-signal-admission-v1",
        "authenticated": True,
        "cadence_batches": int(cadence_batches),
        "observed_steps": len(selected),
        "world_size": int(expected_world_size),
        "scalar_value_trunk_grad_scale": float(
            expected_value_trunk_grad_scale
        ),
        "observations": selected,
    }


def contract_from_cli(
    *,
    module_names: Sequence[str],
    cadence_batches: int,
    minimum_observations: int,
) -> dict[str, Any]:
    return validate_contract(
        {
            "schema_version": CONTRACT_SCHEMA,
            "cadence_batches": int(cadence_batches),
            "minimum_observations": int(minimum_observations),
            "norm_scope": "global_replicated",
            "required_modules": sorted(set(module_names)),
        }
    )
