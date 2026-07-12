from __future__ import annotations

import numpy as np
import pytest

from tools.train_bc import (
    evaluate_composite_validation_measure,
    objective_matched_validation_metrics,
)


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
        return np.searchsorted(
            self.component_offsets, np.asarray(rows), side="right"
        ) - 1


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


def test_downstream_metric_selector_prefers_matched_and_falls_back_historically() -> None:
    epoch = {
        "validation": {"loss": 99.0},
        "validation_objective_matched": {
            "objective_matched": True,
            "metrics": {"loss": 4.0},
        },
    }
    assert objective_matched_validation_metrics(epoch) == {"loss": 4.0}
    assert objective_matched_validation_metrics(
        {"validation": {"loss": 2.0}}
    ) == {"loss": 2.0}


def test_downstream_metric_selector_does_not_trust_unmarked_wrapper() -> None:
    epoch = {
        "validation": {"loss": 2.0},
        "validation_objective_matched": {
            "objective_matched": False,
            "metrics": {"loss": 1.0},
        },
    }
    assert objective_matched_validation_metrics(epoch) == {"loss": 2.0}


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
