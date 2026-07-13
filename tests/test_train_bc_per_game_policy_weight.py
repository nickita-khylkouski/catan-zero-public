from __future__ import annotations

import numpy as np
import pytest

from catan_zero.rl.pipeline_configs import TrainConfig
from tools.train_bc import build_parser, build_sample_weights


def _base_kwargs():
    return {
        "teacher_weights": {},
        "phase_weights": {},
        "forced_action_weight": 1.0,
        "winner_sample_weight": 1.0,
        "loser_sample_weight": 1.0,
        "vp_margin_weight": 0.0,
        "vps_to_win": 10,
    }


def _data():
    # Games have 2, 4, and 3 rows. The third game has no active policy rows.
    return {
        "action_taken": np.arange(9, dtype=np.int16),
        "game_seed": np.asarray([1, 1, 2, 2, 2, 2, 3, 3, 3], dtype=np.int64),
        "legal_action_ids": np.tile(np.asarray([[1, 2]], dtype=np.int16), (9, 1)),
        "teacher_name": np.asarray(["a", "b", "a", "b", "a", "b", "z", "z", "z"]),
        "phase": np.asarray(["x", "y", "x", "y", "x", "y", "x", "x", "x"]),
        "policy_weight_multiplier": np.asarray(
            [1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 0.0, 0.0, 0.0], dtype=np.float32
        ),
    }


def test_default_off_is_bit_identical_to_explicit_historical_path():
    data = _data()
    kwargs = {
        **_base_kwargs(),
        "teacher_weights": {"a": 2.0},
        "phase_weights": {"y": 3.0},
    }
    implicit = build_sample_weights(data, **kwargs)
    explicit = build_sample_weights(
        data,
        **kwargs,
        per_game_policy_weight=False,
        per_game_policy_weight_mode="sqrt",
    )
    assert implicit.dtype == np.float32
    assert implicit.tobytes() == explicit.tobytes()


def test_equal_mode_equalizes_positive_policy_mass_across_game_lengths_and_categories():
    data = _data()
    weights = build_sample_weights(
        data,
        **{
            **_base_kwargs(),
            "teacher_weights": {"a": 2.0, "b": 0.5},
            "phase_weights": {"y": 3.0},
        },
        per_game_policy_weight=True,
        per_game_policy_weight_mode="equal",
    )
    totals = [float(weights[data["game_seed"] == seed].sum()) for seed in (1, 2, 3)]
    assert totals[0] == pytest.approx(totals[1])
    assert totals[2] == 0.0
    assert np.all(weights[6:] == 0.0)
    assert float(weights.mean()) == pytest.approx(1.0)


def test_sqrt_mode_retains_sqrt_original_positive_game_mass_ratio():
    data = _data()
    original = np.asarray(data["policy_weight_multiplier"], dtype=np.float64)
    original_totals = [original[data["game_seed"] == seed].sum() for seed in (1, 2)]
    weights = build_sample_weights(
        data,
        **_base_kwargs(),
        per_game_policy_weight=True,
        per_game_policy_weight_mode="sqrt",
    )
    totals = [float(weights[data["game_seed"] == seed].sum()) for seed in (1, 2)]
    assert totals[1] / totals[0] == pytest.approx(
        np.sqrt(original_totals[1] / original_totals[0])
    )
    assert np.all(weights[6:] == 0.0)


@pytest.mark.parametrize("mode", ["equal", "sqrt"])
def test_v2_sparse_policy_correction_uses_game_uniform_measure(mode: str):
    class _CompositeV2(dict):
        component_offsets = (0, 2, 6)
        component_game_sampling_ratios = (0.5, 0.5)

    # Both games have the same 50% active-policy density but different lengths.
    # V2 gives each game half the sampling mass, so a per-game correction must
    # not turn that into inverse-length or inverse-sqrt-length weighting.
    data = _CompositeV2(
        action_taken=np.arange(6, dtype=np.int16),
        game_seed=np.asarray([7, 7, 7, 7, 7, 7], dtype=np.int64),
        legal_action_ids=np.tile(np.asarray([[1, 2]], dtype=np.int16), (6, 1)),
        policy_weight_multiplier=np.asarray(
            [1.0, 0.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float32
        ),
    )
    weights = build_sample_weights(
        data,
        **_base_kwargs(),
        per_game_policy_weight=True,
        per_game_policy_weight_mode=mode,
    )
    short_mass = 0.5 * float(np.mean(weights[:2]))
    long_mass = 0.5 * float(np.mean(weights[2:]))
    assert short_mass == pytest.approx(long_mass)
    assert weights[1] == 0.0
    assert np.all(weights[4:] == 0.0)


def test_enabled_without_game_seed_fails_closed():
    data = _data()
    del data["game_seed"]
    with pytest.raises(SystemExit, match="requires a populated game_seed"):
        build_sample_weights(
            data,
            **_base_kwargs(),
            per_game_policy_weight=True,
            per_game_policy_weight_mode="equal",
        )


def test_unknown_mode_fails_when_enabled():
    with pytest.raises(ValueError, match="unknown per_game_policy_weight_mode"):
        build_sample_weights(
            _data(),
            **_base_kwargs(),
            per_game_policy_weight=True,
            per_game_policy_weight_mode="cubed",
        )


def test_cli_and_typed_config_defaults_are_off_and_equal():
    parser = build_parser()
    assert parser.get_default("per_game_policy_weight") is False
    assert parser.get_default("per_game_policy_weight_mode") == "equal"
    config = TrainConfig()
    assert config.per_game_policy_weight is False
    assert config.per_game_policy_weight_mode == "equal"
