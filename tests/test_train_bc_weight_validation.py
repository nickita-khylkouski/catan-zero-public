from __future__ import annotations

import numpy as np
import pytest

from tools.train_bc import (
    _parse_weight_map,
    _validate_effective_training_weight_mass,
    _validate_training_weight_arguments,
    _validate_weight_map_keys,
    build_parser,
    build_sample_weights,
    build_value_sample_weights,
)


def _args():
    return build_parser().parse_args(
        ["--data", "data", "--checkpoint", "checkpoint.pt", "--report", "report.json"]
    )


def test_default_weight_configuration_is_valid() -> None:
    _validate_training_weight_arguments(_args())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("policy_loss_weight", -1.0, "--policy-loss-weight"),
        ("forced_action_weight", float("nan"), "--forced-action-weight"),
        ("forced_row_value_weight", float("inf"), "--forced-row-value-weight"),
        ("soft_target_weight", 1.01, "--soft-target-weight"),
        ("soft_target_temperature", 0.0, "--soft-target-temperature"),
        ("value_target_lambda", float("nan"), "--value-target-lambda"),
    ],
)
def test_invalid_scalar_weights_fail_closed(field: str, value: float, message: str) -> None:
    args = _args()
    setattr(args, field, value)
    with pytest.raises(SystemExit, match=message):
        _validate_training_weight_arguments(args)


@pytest.mark.parametrize(
    "raw",
    ["teacher=-1", "teacher=nan", "teacher=inf", "teacher=1,teacher=2"],
)
def test_invalid_or_duplicate_mapped_weights_fail_closed(raw: str) -> None:
    with pytest.raises(SystemExit):
        _parse_weight_map(raw, option="--teacher-weights")


def test_mapped_weight_typo_is_rejected_instead_of_becoming_noop() -> None:
    data = {
        "teacher_name": np.asarray(["n128", "n256"]),
        "phase": np.asarray(["PLAY_TURN", "MOVE_ROBBER"]),
    }
    with pytest.raises(SystemExit, match="n218"):
        _validate_weight_map_keys(
            data,
            teacher_weights={"n218": 2.0},
            policy_phase_weights={},
            value_phase_weights={},
        )


def test_observed_mapped_weight_keys_are_admitted() -> None:
    data = {
        "teacher_name": np.asarray(["n128", "n256"]),
        "phase": np.asarray(["PLAY_TURN", "MOVE_ROBBER"]),
    }
    report = _validate_weight_map_keys(
        data,
        teacher_weights={"n128": 2.0},
        policy_phase_weights={"PLAY_TURN": 3.0},
        value_phase_weights={"MOVE_ROBBER": 4.0},
    )
    assert report["validated"] is True


def test_policy_weight_builder_rejects_negative_scalar_before_reduction() -> None:
    data = {
        "action_taken": np.asarray([0, 0]),
        "legal_action_ids": np.asarray([[1, -1], [1, 2]]),
    }
    with pytest.raises(SystemExit, match="forced_action_weight"):
        build_sample_weights(
            data,
            teacher_weights={},
            phase_weights={},
            forced_action_weight=-1.0,
            winner_sample_weight=1.0,
            loser_sample_weight=1.0,
            vp_margin_weight=0.0,
            vps_to_win=10,
        )


def test_value_weight_builder_rejects_nonfinite_corpus_multiplier() -> None:
    data = {
        "action_taken": np.asarray([0, 0]),
        "value_weight_multiplier": np.asarray([1.0, np.nan]),
    }
    with pytest.raises(SystemExit, match="value_weight_multiplier"):
        build_value_sample_weights(data)


def test_enabled_objective_requires_positive_training_split_weight_mass() -> None:
    with pytest.raises(SystemExit, match="policy/Q objective"):
        _validate_effective_training_weight_mass(
            policy_weights=np.asarray([0.0, 0.0, 1.0]),
            value_weights=np.ones(3),
            train_indices=np.asarray([0, 1]),
            policy_objective_enabled=True,
            value_objective_enabled=True,
        )

    with pytest.raises(SystemExit, match="value/final-VP objective"):
        _validate_effective_training_weight_mass(
            policy_weights=np.ones(3),
            value_weights=np.asarray([0.0, 0.0, 1.0]),
            train_indices=np.asarray([0, 1]),
            policy_objective_enabled=True,
            value_objective_enabled=True,
        )


def test_disabled_objective_allows_zero_weight_mass() -> None:
    report = _validate_effective_training_weight_mass(
        policy_weights=np.zeros(2),
        value_weights=np.zeros(2),
        train_indices=np.arange(2),
        policy_objective_enabled=False,
        value_objective_enabled=False,
    )
    assert report == {
        "policy_sample_weight_mass": 0.0,
        "value_sample_weight_mass": 0.0,
    }
