from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "train_bc_value_strata", ROOT / "tools" / "train_bc.py"
)
assert SPEC is not None and SPEC.loader is not None
train_bc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train_bc)


def test_value_validation_strata_follow_exact_error_weight_and_eligibility() -> None:
    legal = np.full((5, 25), -1, dtype=np.int16)
    for row, width in enumerate((1, 1, 3, 12, 25)):
        legal[row, :width] = np.arange(width, dtype=np.int16)
    legal[1, 0] = 1
    data = {
        "legal_action_ids": legal,
        "action_taken": np.asarray([0, 1, 2, 2, 2], dtype=np.int16),
        "phase": np.asarray(
            ["ROLL", "PLAY_TURN", "PLAY_TURN", "ROBBER", "PLAY_TURN"]
        ),
    }
    parts = train_bc._value_validation_strata_parts(
        data,
        np.arange(5, dtype=np.int64),
        torch.tensor([1.0, 4.0, 9.0, 16.0, 25.0]),
        torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
        torch.tensor([True, True, True, True, False]),
        action_types_by_id=("ROLL", "END_TURN", "BUILD_ROAD"),
    )
    report = train_bc._finalize_value_validation_strata(parts)

    assert report["decision_class"]["forced"] == {
        "mse": 3.0,
        "weighted_sum": 9.0,
        "weight_sum": 3.0,
        "rows": 2,
    }
    assert report["decision_class"]["multi_action"] == {
        "mse": 13.0,
        "weighted_sum": 91.0,
        "weight_sum": 7.0,
        "rows": 2,
    }
    assert report["forced_action_type"]["ROLL"]["mse"] == 1.0
    assert report["forced_action_type"]["END_TURN"]["mse"] == 4.0
    assert report["forced_action_type"]["OTHER"]["weight_sum"] == 0.0
    assert report["legal_width"]["2-4"]["mse"] == 9.0
    assert report["legal_width"]["11-20"]["mse"] == 16.0
    assert report["legal_width"]["21+"]["rows"] == 0
    assert report["phase"]["PLAY_TURN"]["mse"] == pytest.approx(7.0)


def test_objective_measure_combines_value_strata_from_density_not_game_means() -> None:
    reports = [
        {
            "samples": 2,
            "value_mse_strata_sufficient_statistics": {
                "decision_class": {
                    "multi_action": {
                        "weighted_sum": 2.0,
                        "weight_sum": 1.0,
                        "rows": 1.0,
                    }
                }
            },
        },
        {
            "samples": 4,
            "value_mse_strata_sufficient_statistics": {
                "decision_class": {
                    "multi_action": {
                        "weighted_sum": 8.0,
                        "weight_sum": 4.0,
                        "rows": 4.0,
                    }
                }
            },
        },
    ]
    metrics, _ = train_bc._objective_measure_validation_aggregate(
        reports, np.asarray([0.5, 0.5], dtype=np.float64)
    )
    multi = metrics["value_mse_strata"]["decision_class"]["multi_action"]

    assert multi["weighted_numerator_per_sample"] == pytest.approx(1.5)
    assert multi["weight_per_sample"] == pytest.approx(0.75)
    assert multi["mse"] == pytest.approx(2.0)
    assert multi["row_probability"] == pytest.approx(0.75)
