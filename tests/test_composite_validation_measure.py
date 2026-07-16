from __future__ import annotations

import numpy as np
import pytest

from tools.train_bc import (
    _combine_policy_aux_validation_metrics,
    _IndexedValidationWeights,
    _objective_measure_validation_aggregate,
    evaluate_composite_validation_measure,
    objective_matched_validation_component_metrics,
    objective_matched_validation_metrics,
)


def test_policy_aux_validation_reconstructs_the_training_objective() -> None:
    combined = _combine_policy_aux_validation_metrics(
        {
            "loss": 2.0,
            "raw_batch_mean_loss": 2.5,
            "policy_loss": 0.5,
            "value_loss": 1.5,
            "policy_kl_anchor_loss": 0.1,
            "loss_denominators": {"policy_kl_anchor_loss": 10.0},
        },
        {
            "samples": 40,
            "policy_loss": 1.2,
            "policy_kl_anchor_loss": 0.5,
            "loss_denominators": {
                "policy_loss": 3.0,
                "policy_kl_anchor_loss": 6.0,
            },
        },
        policy_loss_weight=1.0,
        policy_aux_loss_weight=0.25,
        policy_kl_anchor_weight=0.4,
    )

    assert combined["policy_base_loss"] == 0.5
    assert combined["policy_aux_loss"] == 1.2
    assert combined["policy_loss"] == pytest.approx(0.8)
    assert combined["policy_kl_anchor_loss"] == pytest.approx(0.225)
    assert combined["policy_kl_anchor_base_loss"] == 0.1
    assert combined["policy_kl_anchor_aux_loss"] == 0.5
    assert combined["policy_kl_anchor_controller_measure"] == pytest.approx(
        1.75 / 11.5
    )
    assert combined["loss"] == pytest.approx(2.35)
    assert combined["component_reconstructed_loss"] == pytest.approx(2.35)
    assert combined["base_raw_batch_mean_loss"] == 2.5
    assert combined["policy_aux_validation_effective_weight_sum"] == 3.0


def test_indexed_validation_weights_avoid_a_corpus_sized_allocation() -> None:
    weights = _IndexedValidationWeights(
        np.asarray([100, 5, 42], dtype=np.int64),
        np.asarray([0.1, 0.2, 0.7], dtype=np.float64),
    )

    assert weights[np.asarray([42, 5], dtype=np.int64)].tolist() == [0.7, 0.2]
    with pytest.raises(KeyError, match="outside"):
        weights[np.asarray([6], dtype=np.int64)]


class _Composite:
    component_ids = ("n128", "replay")
    component_game_sampling_ratios = (0.75, 0.25)
    corpora = (object(), object())

    def __init__(self) -> None:
        # n128: one one-row game and one three-row game. replay: one two-row
        # game. This deliberately makes raw-row and game-uniform measures differ.
        self._game_seed = np.asarray([11, 12, 12, 12, 21, 21], dtype=np.int64)
        self.component_offsets = np.asarray([0, 4, 6], dtype=np.int64)

    def __getitem__(self, key: str):
        if key != "game_seed":
            raise KeyError(key)
        return self._game_seed

    def component_indices_for_rows(self, rows) -> np.ndarray:
        return (
            np.searchsorted(self.component_offsets, np.asarray(rows), side="right") - 1
        )


def test_objective_matched_validation_is_component_then_game_then_row() -> None:
    data = _Composite()
    per_row_loss = np.asarray([1.0, 3.0, 3.0, 3.0, 10.0, 10.0])
    calls: list[tuple[int, ...]] = []

    def evaluate(indices: np.ndarray) -> dict:
        calls.append(tuple(map(int, indices)))
        value = float(per_row_loss[indices].mean())
        return {
            "loss": value,
            "policy_loss": value + 1.0,
            "accuracy": value / 10.0,
            "samples": int(len(indices)),
        }

    report = evaluate_composite_validation_measure(
        data, np.arange(6, dtype=np.int64), evaluate
    )

    # n128 game-uniform=(1+3)/2=2, replay=10; authenticated aggregate
    # .75*2 + .25*10 = 4. A raw-row concat would be 5 and is intentionally not
    # the reported objective-matched value.
    assert report["metrics"]["loss"] == 4.0
    assert report["metrics"]["policy_loss"] == 5.0
    assert report["components"]["n128"]["metrics"]["loss"] == 2.0
    assert report["components"]["n128"]["min_rows_per_game"] == 1
    assert report["components"]["n128"]["max_rows_per_game"] == 3
    assert report["components"]["replay"]["metrics"]["loss"] == 10.0
    assert report["component_sampling_ratios"] == {"n128": 0.75, "replay": 0.25}
    assert report["schema_version"] == "composite-validation-measure-v2"
    assert calls == [(0,), (1, 2, 3), (4, 5)]


def test_objective_matched_validation_rejects_missing_component_holdout() -> None:
    data = _Composite()

    try:
        evaluate_composite_validation_measure(
            data,
            np.arange(4, dtype=np.int64),
            lambda indices: {"loss": float(len(indices))},
        )
    except SystemExit as error:
        assert "replay" in str(error)
        assert "no rows" in str(error)
    else:  # pragma: no cover - fail-closed contract assertion
        raise AssertionError("missing authenticated validation component was accepted")


def test_downstream_metric_selector_prefers_matched_and_falls_back_historically() -> (
    None
):
    epoch = {
        "validation": {"loss": 99.0},
        "validation_objective_matched": {
            "objective_matched": True,
            "metrics": {"loss": 4.0},
        },
    }
    assert objective_matched_validation_metrics(epoch) == {"loss": 4.0}
    assert objective_matched_validation_metrics({"validation": {"loss": 2.0}}) == {
        "loss": 2.0
    }


def test_downstream_component_selector_requires_authenticated_wrapper() -> None:
    epoch = {
        "validation_objective_matched": {
            "objective_matched": True,
            "components": {"replay": {"metrics": {"loss": 3.0}}},
        }
    }
    assert objective_matched_validation_component_metrics(epoch) == {
        "replay": {"loss": 3.0}
    }
    with pytest.raises(ValueError, match="per-component"):
        objective_matched_validation_component_metrics({}, require_matched=True)


def test_downstream_metric_selector_does_not_trust_unmarked_wrapper() -> None:
    epoch = {
        "validation": {"loss": 2.0},
        "validation_objective_matched": {
            "objective_matched": False,
            "metrics": {"loss": 1.0},
        },
    }
    assert objective_matched_validation_metrics(epoch) == {"loss": 2.0}
    with pytest.raises(ValueError, match="raw concatenated-row fallback"):
        objective_matched_validation_metrics(epoch, require_matched=True)


def test_objective_measure_aggregates_weight_density_before_dividing() -> None:
    data = _Composite()

    def evaluate(indices: np.ndarray) -> dict:
        # Game 11 has one active row with loss 1. Game 12 has one active row out
        # of three with loss 9. Equal-game averaging of normalized losses gives
        # 5, but the actual component->game->row training objective is
        # ((1/1 + 9/3)/2) / ((1/1 + 1/3)/2) = 3.
        seed = int(data["game_seed"][indices[0]])
        policy_loss = {11: 1.0, 12: 9.0, 21: 5.0}[seed]
        denominator = {11: 1.0, 12: 1.0, 21: 2.0}[seed]
        return {
            "samples": int(len(indices)),
            "loss": policy_loss,
            "raw_batch_mean_loss": policy_loss,
            "component_reconstructed_loss": policy_loss,
            "policy_loss": policy_loss,
            "loss_denominators": {"policy_loss": denominator},
            "objective_coefficients": {"policy_loss": 1.0},
        }

    report = evaluate_composite_validation_measure(
        data, np.arange(6, dtype=np.int64), evaluate
    )

    assert report["components"]["n128"]["metrics"]["policy_loss"] == 3.0
    # Overall: n128 numerator density=2, denominator density=2/3; replay both
    # densities are 5 and 1. Apply .75/.25, then divide: 2.75 / .75.
    assert report["metrics"]["policy_loss"] == pytest.approx(11.0 / 3.0)
    assert report["metrics"]["loss"] == pytest.approx(11.0 / 3.0)
    assert report["metrics"]["raw_batch_mean_loss"] == 5.0


def test_exact_total_loss_reconstruction_includes_zero_weight_belief_term() -> None:
    """An always-reported zero-weight term must not disable reconstruction.

    Both reports contain 100 rows.  The first game has one active policy row at
    loss 10; the second has 100 active rows at loss 0.  Averaging their already
    normalised total losses gives 5.25, while the configured population
    objective is 10 / 101 + 0.25 = 0.3490099.
    """

    coefficient_keys = (
        "policy_loss",
        "value_loss",
        "final_vp_loss",
        "q_loss",
        "policy_kl_anchor_loss",
        "value_uncertainty_loss",
        "aux_subgoal_loss",
        "belief_resource_loss",
        "moe_balance_loss",
        "value_categorical_loss",
    )
    coefficients = {key: 0.0 for key in coefficient_keys}
    coefficients.update({"policy_loss": 1.0, "value_loss": 0.25})

    def report(*, policy_loss: float, policy_rows: float) -> dict:
        losses = {key: 0.0 for key in coefficient_keys}
        losses.update({"policy_loss": policy_loss, "value_loss": 1.0})
        denominators = {key: 0.0 for key in coefficient_keys}
        denominators.update({"policy_loss": policy_rows, "value_loss": 100.0})
        return {
            "samples": 100,
            "loss": policy_loss + 0.25,
            "raw_batch_mean_loss": policy_loss + 0.25,
            "component_reconstructed_loss": policy_loss + 0.25,
            "scalar_value_mse_diagnostic": 1.0,
            "primary_value_loss": 1.0,
            "loss_denominators": denominators,
            "objective_coefficients": coefficients,
            **losses,
        }

    metrics, sufficient = _objective_measure_validation_aggregate(
        [
            report(policy_loss=10.0, policy_rows=1.0),
            report(policy_loss=0.0, policy_rows=100.0),
        ],
        np.asarray([0.5, 0.5]),
    )

    expected_policy = 10.0 / 101.0
    expected_total = expected_policy + 0.25
    assert metrics["raw_batch_mean_loss"] == pytest.approx(5.25)
    assert metrics["policy_loss"] == pytest.approx(expected_policy)
    assert metrics["value_loss"] == pytest.approx(1.0)
    assert metrics["scalar_value_mse_diagnostic"] == pytest.approx(1.0)
    assert metrics["primary_value_loss"] == pytest.approx(1.0)
    assert metrics["component_reconstructed_loss"] == pytest.approx(expected_total)
    assert metrics["loss"] == pytest.approx(expected_total)
    assert sufficient is not None
    assert "belief_resource_loss" in sufficient


def test_objective_matched_validation_excludes_value_only_replay_policy_ce() -> None:
    data = _Composite()
    data.policy_distillation_scope_authenticated = True
    data.policy_distillation_component_indices = (0,)

    def evaluate(indices: np.ndarray) -> dict:
        replay = bool(np.all(indices >= 4))
        return {
            "samples": int(len(indices)),
            "loss": 0.0 if replay else 2.0,
            "policy_loss": 0.0 if replay else 2.0,
            "loss_denominators": {
                "policy_loss": 0.0 if replay else float(len(indices))
            },
            "objective_coefficients": {"policy_loss": 1.0},
        }

    report = evaluate_composite_validation_measure(
        data, np.arange(6, dtype=np.int64), evaluate
    )
    assert report["policy_distillation_component_ids"] == ["n128"]
    assert report["components"]["n128"]["policy_distillation_enabled"] is True
    assert report["components"]["replay"]["policy_distillation_enabled"] is False
    assert report["components"]["replay"]["metrics"]["policy_loss"] == 0.0
    assert report["metrics"]["policy_loss"] == 2.0


def test_teacher_gap_closure_uses_aggregate_kl_mass_not_mean_of_game_ratios() -> None:
    data = _Composite()

    def evaluate(indices: np.ndarray) -> dict:
        seed = int(data["game_seed"][indices[0]])
        # Game 11: one active row, prior gap 10, model gap 0 (closure 1).
        # Game 12: one active row among three, prior/model gap 1 (closure 0).
        # The correct n128 closure is 1 - (0/1 + 1/3)/(10/1 + 1/3),
        # not mean(1, 0)=.5.
        prior = {11: 10.0, 12: 1.0, 21: 2.0}[seed]
        model = {11: 0.0, 12: 1.0, 21: 1.0}[seed]
        return {
            "samples": int(len(indices)),
            "active_policy_teacher_gap_rows": 1,
            "active_policy_kl_target_model_mean": model,
            "active_policy_kl_target_prior_mean": prior,
            "active_policy_teacher_gap_closure": 1.0 - model / prior,
        }

    report = evaluate_composite_validation_measure(
        data, np.arange(6, dtype=np.int64), evaluate
    )

    assert report["components"]["n128"]["metrics"][
        "active_policy_teacher_gap_closure"
    ] == pytest.approx(1.0 - (1.0 / 3.0) / (10.0 + 1.0 / 3.0))
