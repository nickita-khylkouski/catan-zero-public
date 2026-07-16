from __future__ import annotations

import numpy as np
import pytest

from tools.train_bc import _validate_loss_weight_map_realization


def _data() -> dict[str, np.ndarray]:
    return {
        "action_taken": np.zeros(5, dtype=np.int16),
        "phase": np.asarray(
            ["ROLL", "PLAY_TURN", "PLAY_TURN", "END_TURN", "PLAY_TURN"]
        ),
        "teacher_name": np.asarray(["coherent_n128"] * 5),
    }


def test_phase_weight_realization_binds_the_rows_it_actually_changes() -> None:
    report = _validate_loss_weight_map_realization(
        _data(),
        {"PLAY_TURN": 4.0},
        column="phase",
        option="--phase-weights",
    )

    assert report["configured"] == {"PLAY_TURN": 4.0}
    assert report["realized_rows"] == {"PLAY_TURN": 3}


def test_phase_weight_typo_cannot_silently_turn_treatment_into_control() -> None:
    with pytest.raises(SystemExit, match="absent from corpus.*play_turn"):
        _validate_loss_weight_map_realization(
            _data(),
            {"play_turn": 4.0},
            column="phase",
            option="--phase-weights",
        )


@pytest.mark.parametrize("bad", [-0.25, float("nan"), float("inf")])
def test_loss_weight_map_cannot_reverse_or_poison_gradients(bad: float) -> None:
    with pytest.raises(SystemExit, match="finite values >= 0"):
        _validate_loss_weight_map_realization(
            _data(),
            {"PLAY_TURN": bad},
            column="phase",
            option="--phase-weights",
        )


def test_configured_map_requires_its_provenance_column() -> None:
    with pytest.raises(SystemExit, match="requires corpus column"):
        _validate_loss_weight_map_realization(
            {"action_taken": np.zeros(2, dtype=np.int16)},
            {"coherent_n128": 1.0},
            column="teacher_name",
            option="--teacher-weights",
        )
