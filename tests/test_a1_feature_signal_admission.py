from __future__ import annotations

import copy
import math

import pytest

from tools import a1_feature_signal_admission as admission


REQUIRED_MODULES = [
    "event_encoder",
    "final_vp_head",
    "legal_action_value_residual_proj",
    "legal_action_value_static_proj",
    "meaningful_history_ordered_gate",
    "meaningful_history_residual_gate",
    "meaningful_history_sequence",
    "meaningful_history_target_proj",
    "public_card_count_residual",
    "public_rule_state_residual",
    "static_action_residual_proj",
    "value_blocks",
    "value_head",
    "value_state_norm",
]


def _contract() -> dict:
    return {
        "schema_version": admission.CONTRACT_SCHEMA,
        "cadence_batches": 16,
        "minimum_observations": 2,
        "norm_scope": "global_replicated",
        "required_modules": REQUIRED_MODULES,
    }


def _row() -> dict:
    return {
        "mean_pre_clip_grad_norm": 0.25,
        "max_pre_clip_grad_norm": 0.5,
        "mean_parameter_delta_norm": 0.01,
        "mean_parameter_update_rms": 0.001,
        "mean_relative_parameter_delta": 0.1,
        "parameter_count": 8,
    }


def _observability() -> dict:
    return {
        "schema_version": admission.OBSERVABILITY_SCHEMA,
        "observed_steps": 2,
        "cadence_batches": 16,
        "norm_scope": "global_replicated",
        "modules": {name: _row() for name in REQUIRED_MODULES},
    }


def _objective_observation(step: int) -> dict:
    return {
        "available": True,
        "optimizer_step": step,
        "scope": "global_ddp_microbatch",
        "aggregation": (
            "manual_all_reduce_then_world_average_of_ddp_scaled_gradients"
        ),
        "world_size": 8,
        "scalar_value_trunk_grad_scale": 1.0,
        "policy_trunk_grad_norm": 0.5,
        "value_trunk_grad_norm": 0.25,
        "value_to_policy_grad_norm_ratio": 0.5,
        "trunk_gradient_cosine": 0.1,
        "opposing_coordinate_fraction": 0.49,
        "combined_trunk_grad_norm": 0.6,
    }


def _objective_payload() -> dict:
    observations = [_objective_observation(step) for step in (1, 16, 32)]
    return {
        "schema_version": "objective-gradient-dose-observations-v1",
        "cadence_batches": 16,
        "observed_steps": len(observations),
        "observations": observations,
    }


def test_current_v5_feature_signal_admission_accepts_every_live_module() -> None:
    evidence = admission.verify_observability(
        _observability(),
        contract=_contract(),
        where="current v5 scratch report",
    )

    assert evidence["authenticated"] is True
    assert evidence["observed_steps"] == 2
    assert list(evidence["modules"]) == REQUIRED_MODULES
    assert evidence["positive_signal_fields"] == list(
        admission.POSITIVE_SIGNAL_FIELDS
    )


def test_current_v4_objective_gradient_admission_accepts_global_geometry() -> None:
    evidence = admission.verify_objective_interference(
        _objective_payload(),
        cadence_batches=16,
        minimum_observations=2,
        expected_world_size=8,
        expected_value_trunk_grad_scale=1.0,
        where="current v4 scratch report",
    )

    assert evidence["authenticated"] is True
    assert evidence["observed_steps"] == 3
    assert [
        row["optimizer_step"] for row in evidence["observations"]
    ] == [1, 16, 32]


def test_zero_shared_value_gradient_admission_requires_exact_stop_gradient() -> None:
    payload = _objective_payload()
    for observation in payload["observations"]:
        observation.update(
            {
                "scalar_value_trunk_grad_scale": 0.0,
                "value_trunk_grad_norm": 0.0,
                "value_to_policy_grad_norm_ratio": 0.0,
                "trunk_gradient_cosine": None,
                "opposing_coordinate_fraction": None,
                "combined_trunk_grad_norm": observation[
                    "policy_trunk_grad_norm"
                ],
            }
        )

    evidence = admission.verify_objective_interference(
        payload,
        cadence_batches=16,
        minimum_observations=2,
        expected_world_size=8,
        expected_value_trunk_grad_scale=0.0,
        where="split value-tower scratch report",
    )

    assert evidence["authenticated"] is True
    assert all(
        row["value_trunk_grad_norm"] == 0.0
        and row["value_to_policy_grad_norm_ratio"] == 0.0
        and row["trunk_gradient_cosine"] is None
        and row["opposing_coordinate_fraction"] is None
        for row in evidence["observations"]
    )


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("value_trunk_grad_norm", 0.01),
        ("value_to_policy_grad_norm_ratio", 0.01),
        ("trunk_gradient_cosine", 0.0),
        ("opposing_coordinate_fraction", 0.0),
        ("combined_trunk_grad_norm", 0.6),
    ],
)
def test_zero_shared_value_gradient_admission_rejects_boundary_leakage(
    field: str, bad_value: object
) -> None:
    payload = _objective_payload()
    for observation in payload["observations"]:
        observation.update(
            {
                "scalar_value_trunk_grad_scale": 0.0,
                "value_trunk_grad_norm": 0.0,
                "value_to_policy_grad_norm_ratio": 0.0,
                "trunk_gradient_cosine": None,
                "opposing_coordinate_fraction": None,
                "combined_trunk_grad_norm": observation[
                    "policy_trunk_grad_norm"
                ],
            }
        )
    payload["observations"][0][field] = bad_value

    with pytest.raises(admission.FeatureSignalError, match="objective-gradient"):
        admission.verify_objective_interference(
            payload,
            cadence_batches=16,
            minimum_observations=2,
            expected_world_size=8,
            expected_value_trunk_grad_scale=0.0,
            where="split value-tower scratch report",
        )


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("available", False),
        ("scope", "single_process_microbatch"),
        ("aggregation", "rank_local"),
        ("world_size", 4),
        ("scalar_value_trunk_grad_scale", 0.1),
        ("policy_trunk_grad_norm", 0.0),
        ("value_trunk_grad_norm", math.nan),
        ("value_to_policy_grad_norm_ratio", True),
        ("trunk_gradient_cosine", 1.1),
        ("opposing_coordinate_fraction", -0.1),
        ("combined_trunk_grad_norm", "1"),
    ],
)
def test_current_v4_objective_gradient_admission_rejects_bad_geometry(
    field: str,
    bad_value: object,
) -> None:
    payload = _objective_payload()
    payload["observations"][0][field] = bad_value

    with pytest.raises(admission.FeatureSignalError, match=field):
        admission.verify_objective_interference(
            payload,
            cadence_batches=16,
            minimum_observations=2,
            expected_world_size=8,
            expected_value_trunk_grad_scale=1.0,
            where="current v4 scratch report",
        )


def test_current_v4_objective_gradient_admission_rejects_duplicate_steps() -> None:
    payload = _objective_payload()
    payload["observations"][1]["optimizer_step"] = 1

    with pytest.raises(admission.FeatureSignalError, match="steps"):
        admission.verify_objective_interference(
            payload,
            cadence_batches=16,
            minimum_observations=2,
            expected_world_size=8,
            expected_value_trunk_grad_scale=1.0,
            where="current v4 scratch report",
        )


@pytest.mark.parametrize("module_name", REQUIRED_MODULES)
def test_current_v4_feature_signal_admission_rejects_missing_module(
    module_name: str,
) -> None:
    observability = _observability()
    del observability["modules"][module_name]

    with pytest.raises(admission.FeatureSignalError, match=module_name):
        admission.verify_observability(
            observability,
            contract=_contract(),
            where="current v4 scratch report",
        )


@pytest.mark.parametrize("field", admission.POSITIVE_SIGNAL_FIELDS)
@pytest.mark.parametrize("bad_value", [0.0, -1.0, math.nan, math.inf, True, "1"])
def test_current_v4_feature_signal_admission_rejects_nonpositive_or_bad_types(
    field: str,
    bad_value: object,
) -> None:
    observability = _observability()
    observability["modules"]["event_encoder"][field] = bad_value

    with pytest.raises(admission.FeatureSignalError, match=field):
        admission.verify_observability(
            observability,
            contract=_contract(),
            where="current v4 scratch report",
        )


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("observed_steps", 1),
        ("observed_steps", True),
        ("cadence_batches", 8),
        ("norm_scope", "rank_local_shard"),
        ("schema_version", "legacy"),
    ],
)
def test_current_v4_feature_signal_admission_rejects_observation_drift(
    field: str,
    bad_value: object,
) -> None:
    observability = copy.deepcopy(_observability())
    observability[field] = bad_value

    with pytest.raises(admission.FeatureSignalError, match="cadence"):
        admission.verify_observability(
            observability,
            contract=_contract(),
            where="current v4 scratch report",
        )


@pytest.mark.parametrize("bad_value", [0, -1, True, 1.5, "8"])
def test_current_v4_feature_signal_admission_rejects_bad_parameter_count(
    bad_value: object,
) -> None:
    observability = _observability()
    observability["modules"]["public_rule_state_residual"][
        "parameter_count"
    ] = bad_value

    with pytest.raises(admission.FeatureSignalError, match="parameter_count"):
        admission.verify_observability(
            observability,
            contract=_contract(),
            where="current v4 scratch report",
        )
