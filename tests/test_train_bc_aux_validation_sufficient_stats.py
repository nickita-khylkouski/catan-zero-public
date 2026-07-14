from __future__ import annotations

import math

import numpy as np
import pytest

from tools import train_bc
from catan_zero.rl.aux_subgoal_targets import (
    AUX_SUBGOAL_TARGET_VERSION,
    AUX_SUBGOAL_TARGET_VERSION_KEY,
)

torch = pytest.importorskip("torch")


def test_aux_subgoal_returns_one_exact_statistic_pair_per_masked_head() -> None:
    outputs = {
        "aux_longest_road": torch.zeros(4, requires_grad=True),
        "aux_vp_in_n": torch.zeros(4, requires_grad=True),
    }
    data = {
        "aux_longest_road": np.asarray([0.0, 1.0, np.nan, np.nan]),
        "aux_vp_in_n": np.asarray([1.0, np.nan, 3.0, np.nan]),
        AUX_SUBGOAL_TARGET_VERSION_KEY: np.full(
            4, AUX_SUBGOAL_TARGET_VERSION, dtype=np.uint8
        ),
    }

    loss, active, parts = train_bc._aux_subgoal_loss(
        outputs,
        data,
        np.arange(4),
        torch.device("cpu"),
        return_sufficient_statistics=True,
    )

    assert active == 2
    assert float(loss.detach()) == pytest.approx(math.log(2.0) + 5.0)
    assert float(parts["aux_longest_road"]["weighted_sum"].detach()) == pytest.approx(
        2.0 * math.log(2.0)
    )
    assert float(parts["aux_longest_road"]["weight_sum"].detach()) == 2.0
    assert float(parts["aux_vp_in_n"]["weighted_sum"].detach()) == 10.0
    assert float(parts["aux_vp_in_n"]["weight_sum"].detach()) == 2.0


def test_composite_validation_reconstructs_aux_heads_before_summing() -> None:
    reports = [
        {
            "samples": 1,
            "loss": 1.0,
            "aux_subgoal_loss": 1.0,
            "aux_subgoal_loss_parts": {
                "head_a": {"weighted_sum": 1.0, "weight_sum": 1.0},
                "head_b": {"weighted_sum": 0.0, "weight_sum": 0.0},
            },
            "objective_coefficients": {"aux_subgoal_loss": 1.0},
        },
        {
            "samples": 3,
            "loss": 11.0,
            "aux_subgoal_loss": 11.0,
            "aux_subgoal_loss_parts": {
                "head_a": {"weighted_sum": 9.0, "weight_sum": 1.0},
                "head_b": {"weighted_sum": 6.0, "weight_sum": 3.0},
            },
            "objective_coefficients": {"aux_subgoal_loss": 1.0},
        },
    ]

    metrics, sufficient = train_bc._objective_measure_validation_aggregate(
        reports, np.asarray([0.5, 0.5])
    )

    # head_a: ((1/1 + 9/3)/2) / ((1/1 + 1/3)/2) = 3
    # head_b: ((0/1 + 6/3)/2) / ((0/1 + 3/3)/2) = 2
    assert metrics["raw_batch_mean_loss"] == 6.0
    assert metrics["aux_subgoal_loss"] == 5.0
    assert metrics["component_reconstructed_loss"] == 5.0
    assert metrics["loss"] == 5.0
    assert sufficient is not None
    assert sufficient["aux_subgoal_loss"]["heads"]["head_a"]["loss"] == 3.0
    assert sufficient["aux_subgoal_loss"]["heads"]["head_b"]["loss"] == 2.0


def test_composite_validation_reconstructs_nonzero_uncertainty_objective() -> None:
    reports = [
        {
            "samples": 1,
            "loss": 0.25,
            "value_uncertainty_loss": 1.0,
            "loss_denominators": {"value_uncertainty_loss": 1.0},
            "objective_coefficients": {"value_uncertainty_loss": 0.25},
        },
        {
            "samples": 3,
            "loss": 2.25,
            "value_uncertainty_loss": 9.0,
            "loss_denominators": {"value_uncertainty_loss": 1.0},
            "objective_coefficients": {"value_uncertainty_loss": 0.25},
        },
    ]

    metrics, sufficient = train_bc._objective_measure_validation_aggregate(
        reports, np.asarray([0.5, 0.5])
    )

    # The one eligible row in the three-row report contributes one third of
    # its game's density: ((1/1 + 9/3)/2) / ((1/1 + 1/3)/2) = 3.
    assert metrics["raw_batch_mean_loss"] == 1.25
    assert metrics["value_uncertainty_loss"] == 3.0
    assert metrics["component_reconstructed_loss"] == 0.75
    assert metrics["loss"] == 0.75
    assert sufficient is not None
    assert sufficient["value_uncertainty_loss"] == {
        "weighted_numerator_per_sample": 2.0,
        "weight_per_sample": pytest.approx(2.0 / 3.0),
    }


def test_scoped_policy_accuracy_excludes_zero_denominator_replay() -> None:
    reports = [
        {
            "samples": 100,
            "accuracy_active_count": 50,
            "accuracy": 0.613,
            "top3_accuracy": 0.91,
        },
        {
            "samples": 100,
            "accuracy_active_count": 0,
            "accuracy": 0.0,
            "top3_accuracy": 0.0,
        },
    ]

    metrics, sufficient = train_bc._objective_measure_validation_aggregate(
        reports, np.asarray([0.8, 0.2])
    )

    # Replay has 20% sampler mass but no rows in the scoped policy objective;
    # it must not turn .613 into the old diluted .4904.
    assert metrics["accuracy"] == pytest.approx(0.613)
    assert metrics["top3_accuracy"] == pytest.approx(0.91)
    assert sufficient is not None
    assert sufficient["accuracy"]["weight_per_sample"] == pytest.approx(0.4)


def test_nonzero_objective_term_without_exact_statistics_fails_closed() -> None:
    reports = [
        {
            "samples": 4,
            "loss": 0.2,
            "moe_balance_loss": 2.0,
            "objective_coefficients": {"moe_balance_loss": 0.1},
        }
    ]

    with pytest.raises(SystemExit, match="moe_balance_loss"):
        train_bc._objective_measure_validation_aggregate(
            reports, np.asarray([1.0])
        )


def test_inconsistent_objective_coefficients_fail_closed() -> None:
    reports = [
        {
            "samples": 2,
            "policy_loss": 1.0,
            "loss_denominators": {"policy_loss": 2.0},
            "objective_coefficients": {"policy_loss": 1.0},
        },
        {
            "samples": 2,
            "policy_loss": 1.0,
            "loss_denominators": {"policy_loss": 2.0},
            "objective_coefficients": {"policy_loss": 0.5},
        },
    ]
    with pytest.raises(SystemExit, match="inconsistent"):
        train_bc._objective_measure_validation_aggregate(
            reports, np.asarray([0.5, 0.5])
        )


def test_exact_zero_and_zero_coefficient_terms_need_no_additive_statistic() -> None:
    exact_zero_report = {
        "samples": 4,
        "loss": 0.0,
        "moe_balance_loss": 0.0,
        "objective_exact_zero_terms": ["moe_balance_loss"],
        "objective_coefficients": {"moe_balance_loss": 0.1},
    }
    metrics, sufficient = train_bc._objective_measure_validation_aggregate(
        [exact_zero_report], np.asarray([1.0])
    )
    assert metrics["loss"] == 0.0
    assert sufficient == {"moe_balance_loss": {"exact_zero": True}}

    zero_coefficient_report = {
        "samples": 4,
        "loss": 7.0,
        "moe_balance_loss": 7.0,
        "objective_coefficients": {"moe_balance_loss": 0.0},
    }
    metrics, sufficient = train_bc._objective_measure_validation_aggregate(
        [zero_coefficient_report], np.asarray([1.0])
    )
    assert metrics["loss"] == 0.0
    assert sufficient is None
