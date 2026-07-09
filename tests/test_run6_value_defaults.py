"""RUN-6 consolidation regression tests for the EXP3 value recipe (task #62).

Guards two decisions baked into the canonical stack:
  1. --value-loss-weight now defaults to 0.10 (EXP3: strictly better than the old
     0.25 under multi-epoch reuse at a fixed policy recipe).
  2. --per-game-value-weight-mode is a wired {equal,sqrt} knob defaulting to
     "equal" (byte-identical to CAT-60), with "sqrt" applying the
     effective-sample-size correction.
"""
from __future__ import annotations

import numpy as np
import pytest

from tools.train_bc import build_parser, build_value_sample_weights


def test_value_loss_weight_default_is_010() -> None:
    parser = build_parser()
    assert parser.get_default("value_loss_weight") == pytest.approx(0.10)


def test_per_game_value_weight_mode_default_is_equal() -> None:
    parser = build_parser()
    assert parser.get_default("per_game_value_weight_mode") == "equal"


def test_per_game_value_weight_sqrt_scales_by_sqrt_of_game_length() -> None:
    """equal-mode equalizes game totals (4-row game == 1-row game); sqrt-mode makes
    a game of n value rows contribute ~sqrt(n) mass, so a 4-row game gets 2x a
    1-row game. Global mean-renormalization preserves the per-game ratio."""
    game_seed = np.asarray([1, 1, 1, 1, 2], dtype=np.int64)
    data = {"action_taken": np.zeros(5, dtype=np.int16), "game_seed": game_seed}

    w_equal = build_value_sample_weights(
        data, per_game_value_weight=True, per_game_value_weight_mode="equal"
    )
    w_sqrt = build_value_sample_weights(
        data, per_game_value_weight=True, per_game_value_weight_mode="sqrt"
    )

    g1_equal = float(w_equal[game_seed == 1].sum())
    g2_equal = float(w_equal[game_seed == 2].sum())
    assert g1_equal == pytest.approx(g2_equal, rel=1e-5)

    g1_sqrt = float(w_sqrt[game_seed == 1].sum())
    g2_sqrt = float(w_sqrt[game_seed == 2].sum())
    assert (g1_sqrt / g2_sqrt) == pytest.approx(2.0, rel=1e-5)


def test_per_game_value_weight_mode_default_matches_equal() -> None:
    """Omitting the mode reproduces equal-mode exactly (CAT-60 no-op guarantee)."""
    game_seed = np.asarray([1, 1, 1, 2], dtype=np.int64)
    data = {"action_taken": np.zeros(4, dtype=np.int16), "game_seed": game_seed}
    w_default = build_value_sample_weights(data, per_game_value_weight=True)
    w_equal = build_value_sample_weights(
        data, per_game_value_weight=True, per_game_value_weight_mode="equal"
    )
    assert w_default.tolist() == w_equal.tolist()


def test_unknown_per_game_value_weight_mode_raises() -> None:
    game_seed = np.asarray([1, 1, 2], dtype=np.int64)
    data = {"action_taken": np.zeros(3, dtype=np.int16), "game_seed": game_seed}
    with pytest.raises(ValueError, match="unknown per_game_value_weight_mode"):
        build_value_sample_weights(
            data, per_game_value_weight=True, per_game_value_weight_mode="cubed"
        )
