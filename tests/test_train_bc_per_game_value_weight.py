from __future__ import annotations

import numpy as np
import pytest

from tools.train_bc import (
    _apply_authenticated_value_training_scope,
    _value_component_active_dose_for_batch,
    _value_training_scope_report,
    build_value_sample_weights,
    forced_action_type_value_mass_quality,
    per_game_weight_quality,
)
from catan_zero.rl.action_mask import ActionCatalog


def _legal_action_ids(counts: list[int], width: int = 4) -> np.ndarray:
    """Build a legal_action_ids array where row i has exactly counts[i] legal actions
    (remaining columns padded with -1, matching the corpus schema's padding convention)."""
    rows = []
    for count in counts:
        row = list(range(count)) + [-1] * (width - count)
        rows.append(row)
    return np.asarray(rows, dtype=np.int32)


def test_per_game_value_weight_equalizes_total_mass_across_game_lengths() -> None:
    """CAT-60 core property: a 5-row game and a 1-row game must contribute the SAME total
    value-loss mass once --per-game-value-weight is on, addressing '16k games = 16k
    independent outcomes, not 3.6M labels'."""
    game_seed = np.asarray([1, 1, 1, 1, 1, 2], dtype=np.int64)
    data = {
        "action_taken": np.zeros(6, dtype=np.int16),
        "game_seed": game_seed,
    }

    weights = build_value_sample_weights(data, per_game_value_weight=True)

    game1_total = float(np.sum(weights[game_seed == 1]))
    game2_total = float(np.sum(weights[game_seed == 2]))
    assert game1_total == pytest.approx(game2_total, rel=1e-5)


def test_per_game_value_weight_off_is_byte_identical_to_prior_behavior() -> None:
    """Flag-off regression: default arguments must reproduce the exact pre-CAT-60 formula
    (phase weights + value_weight_multiplier, mean-normalized), since per_game_value_weight
    defaults to False and forced_row_value_weight defaults to 1.0 (no-op)."""
    data = {
        "action_taken": np.asarray([1, 2, 3, 4], dtype=np.int16),
        "phase": np.asarray(["robber", "initial_build", "robber", "initial_build"]),
        "value_weight_multiplier": np.asarray([1.0, 2.0, 0.5, 1.0], dtype=np.float32),
        "game_seed": np.asarray([10, 10, 20, 20], dtype=np.int64),
        "legal_action_ids": _legal_action_ids([1, 3, 1, 2]),
    }

    default_weights = build_value_sample_weights(data, phase_weights={"robber": 4.0})
    explicit_off_weights = build_value_sample_weights(
        data,
        phase_weights={"robber": 4.0},
        forced_row_value_weight=1.0,
        per_game_value_weight=False,
    )

    assert default_weights.tolist() == explicit_off_weights.tolist()

    # Cross-check against the hand-computed pre-CAT-60 formula.
    expected = np.asarray([4.0, 2.0, 2.0, 1.0], dtype=np.float32)
    expected = expected / float(np.mean(expected))
    assert default_weights.tolist() == pytest.approx(expected.tolist())


def test_forced_row_value_weight_downweights_single_legal_action_rows() -> None:
    """Rows with exactly one legal action (forced moves) carry near-zero information for the
    value head; --forced-row-value-weight must suppress them specifically in the VALUE loss."""
    data = {
        "action_taken": np.asarray([1, 2, 3], dtype=np.int16),
        "legal_action_ids": _legal_action_ids([1, 3, 1]),
    }

    weights = build_value_sample_weights(data, forced_row_value_weight=0.1)

    # Forced rows (0 and 2) are downweighted relative to the free-choice row (1).
    assert weights[0] < weights[1]
    assert weights[2] < weights[1]
    assert weights[0] == pytest.approx(weights[2])


def test_forced_row_value_weight_default_is_noop() -> None:
    data = {
        "action_taken": np.asarray([1, 2], dtype=np.int16),
        "legal_action_ids": _legal_action_ids([1, 3]),
    }

    weights = build_value_sample_weights(data)

    assert weights[0] == pytest.approx(weights[1])


def _catalog_action_id(catalog: ActionCatalog, action_type: str) -> int:
    return next(
        action_id
        for action_id in range(catalog.size)
        if catalog.describe(action_id)["action_type"] == action_type
    )


def test_forced_row_action_type_weights_compose_after_global_multiplier() -> None:
    catalog = ActionCatalog(("RED", "BLUE"))
    roll = _catalog_action_id(catalog, "ROLL")
    end_turn = _catalog_action_id(catalog, "END_TURN")
    build_road = _catalog_action_id(catalog, "BUILD_ROAD")
    data = {
        "action_taken": np.asarray(
            [roll, end_turn, build_road, roll], dtype=np.int16
        ),
        "legal_action_ids": np.asarray(
            [
                [roll, -1],
                [end_turn, -1],
                [build_road, -1],
                [roll, build_road],
            ],
            dtype=np.int32,
        ),
    }

    weights = build_value_sample_weights(
        data,
        forced_row_value_weight=0.5,
        forced_row_value_action_type_weights={"ROLL": 0.2, "END_TURN": 0.5},
        action_catalog=catalog,
    )

    # Ratios survive the final global mean normalization. An unlisted forced
    # BUILD_ROAD gets only the existing 0.5 global forced multiplier, while a
    # non-forced ROLL row gets neither forced multiplier.
    assert weights[0] / weights[2] == pytest.approx(0.2)
    assert weights[1] / weights[2] == pytest.approx(0.5)
    assert weights[2] / weights[3] == pytest.approx(0.5)

    report = forced_action_type_value_mass_quality(
        data,
        weights,
        action_catalog=catalog,
        configured_weights={"ROLL": 0.2, "END_TURN": 0.5},
    )
    assert report["forced_rows"] == 3
    assert report["by_action_type"]["ROLL"]["rows"] == 1
    assert report["by_action_type"]["ROLL"]["configured_multiplier"] == 0.2
    assert report["by_action_type"]["BUILD_ROAD"][
        "configured_multiplier"
    ] == 1.0
    assert report["effective_forced_value_mass"] == pytest.approx(
        sum(float(weights[index]) for index in (0, 1, 2))
    )


def test_empty_forced_row_action_type_map_is_exact_historical_default() -> None:
    data = {
        "action_taken": np.asarray([1, 2], dtype=np.int16),
        "legal_action_ids": _legal_action_ids([1, 3]),
    }
    historical = build_value_sample_weights(data, forced_row_value_weight=0.25)
    explicit_empty = build_value_sample_weights(
        data,
        forced_row_value_weight=0.25,
        forced_row_value_action_type_weights={},
        action_catalog=None,
    )
    assert explicit_empty.tobytes() == historical.tobytes()


def test_forced_row_action_type_map_rejects_unknown_catalog_type() -> None:
    catalog = ActionCatalog(("RED", "BLUE"))
    roll = _catalog_action_id(catalog, "ROLL")
    data = {
        "action_taken": np.asarray([roll], dtype=np.int16),
        "legal_action_ids": np.asarray([[roll]], dtype=np.int32),
    }
    with pytest.raises(SystemExit, match="unknown ActionCatalog types"):
        build_value_sample_weights(
            data,
            forced_row_value_action_type_weights={"DRAW_CARD": 0.0},
            action_catalog=catalog,
        )


def test_authenticated_value_scope_can_make_replay_anchor_only() -> None:
    class _Composite:
        component_ids = ("n128", "n256", "gen3_replay")
        value_training_component_indices = (0, 1)
        value_training_scope_authenticated = True
        component_offsets = (0, 2, 4, 6)

        @staticmethod
        def component_indices_for_rows(rows):
            mapping = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
            return mapping[np.asarray(rows, dtype=np.int64)]

    weights = np.ones(6, dtype=np.float32)
    scoped = _apply_authenticated_value_training_scope(_Composite(), weights)
    assert scoped.tolist() == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
    assert weights.tolist() == [1.0] * 6
    report = _value_training_scope_report(_Composite(), scoped)
    assert report["component_ids"] == ["n128", "n256"]
    assert report["components"]["gen3_replay"]["positive_value_rows"] == 0

    dose = _value_component_active_dose_for_batch(
        _Composite(),
        np.arange(6, dtype=np.int64),
        np.asarray([True, True, True, False, False, False]),
    )
    assert dose == {"n128": 2.0, "n256": 1.0, "gen3_replay": 0.0}
    with pytest.raises(RuntimeError, match="excluded value component"):
        _value_component_active_dose_for_batch(
            _Composite(),
            np.arange(6, dtype=np.int64),
            np.asarray([True, True, True, False, False, True]),
        )


def test_per_game_value_weight_composes_with_forced_row_and_multiplier() -> None:
    """CAT-60 combination rule: per-game normalization is applied LAST, so it equalizes total
    game mass even when forced-row downweighting and the CAT-45 value_weight_multiplier are
    unevenly distributed within a game -- it does not undo the within-game suppression of
    low-information forced rows, it only equalizes across games."""
    game_seed = np.asarray([1, 1, 1, 2, 2], dtype=np.int64)
    data = {
        "action_taken": np.zeros(5, dtype=np.int16),
        "game_seed": game_seed,
        "value_weight_multiplier": np.asarray(
            [1.0, 0.5, 2.0, 1.0, 1.0], dtype=np.float32
        ),
        "legal_action_ids": _legal_action_ids([1, 3, 2, 3, 1]),
    }

    weights = build_value_sample_weights(
        data,
        forced_row_value_weight=0.1,
        per_game_value_weight=True,
    )

    game1_total = float(np.sum(weights[game_seed == 1]))
    game2_total = float(np.sum(weights[game_seed == 2]))
    assert game1_total == pytest.approx(game2_total, rel=1e-5)
    # Within game 1, the forced row (index 0) still ends up cheaper than the free rows.
    assert weights[0] < weights[1]
    assert weights[0] < weights[2]


def test_per_game_value_weight_missing_game_seed_fails_closed() -> None:
    data = {
        "action_taken": np.asarray([1, 2, 3], dtype=np.int16),
        "phase": np.asarray(["robber", "initial_build", "robber"]),
    }

    with pytest.raises(SystemExit, match="requires a populated game_seed"):
        build_value_sample_weights(
            data, phase_weights={"robber": 4.0}, per_game_value_weight=True
        )


def test_per_game_weight_quality_reports_equalization() -> None:
    game_seed = np.asarray([1, 1, 1, 1, 1, 2], dtype=np.int64)
    data = {"action_taken": np.zeros(6, dtype=np.int16), "game_seed": game_seed}

    unnormalized = build_value_sample_weights(data)
    normalized = build_value_sample_weights(data, per_game_value_weight=True)

    report_off = per_game_weight_quality(data, unnormalized)
    report_on = per_game_weight_quality(data, normalized)

    assert report_off["n_games"] == 2
    assert report_off["rows_per_game"]["min"] == 1
    assert report_off["rows_per_game"]["max"] == 5
    # Off: the 5-row game carries far more total mass than the 1-row game.
    assert report_off["total_weight_per_game"]["std"] > 0.0
    # On: per-game normalization drives the spread to ~0.
    assert report_on["total_weight_per_game"]["std"] == pytest.approx(0.0, abs=1e-6)


def test_composite_reused_seed_is_two_independent_games() -> None:
    """A seed is namespaced by component in a mixed learner corpus."""

    class _Composite(dict):
        component_offsets = (0, 2, 5)

    data = _Composite(
        action_taken=np.zeros(5, dtype=np.int16),
        # Both components legitimately use seed 7, but they contain different
        # games with two and three rows respectively.
        game_seed=np.asarray([7, 7, 7, 7, 7], dtype=np.int64),
    )

    weights = build_value_sample_weights(data, per_game_value_weight=True)

    assert float(weights[:2].sum()) == pytest.approx(float(weights[2:].sum()))
    quality = per_game_weight_quality(data, weights)
    assert quality["n_games"] == 2
    assert quality["rows_per_game"] == {"min": 2, "max": 3, "mean": 2.5}
    assert quality["total_weight_per_game"]["std"] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("mode", ["equal", "sqrt"])
def test_v2_game_uniform_sampler_is_not_double_corrected_for_length(mode: str) -> None:
    """V2 already selects games uniformly before selecting an in-game row."""

    class _CompositeV2(dict):
        component_offsets = (0, 1, 5)
        component_game_sampling_ratios = (0.5, 0.5)

    data = _CompositeV2(
        action_taken=np.zeros(5, dtype=np.int16),
        # One one-row game and one four-row game in distinct components.
        game_seed=np.asarray([7, 7, 7, 7, 7], dtype=np.int64),
    )
    weights = build_value_sample_weights(
        data,
        per_game_value_weight=True,
        per_game_value_weight_mode=mode,
    )
    # Effective mass is sampler probability times loss weight. With uniform
    # base weights, neither correction mode should invent a length bias.
    short_mass = 0.5 * float(weights[0])
    long_mass = 0.5 * float(np.mean(weights[1:]))
    assert short_mass == pytest.approx(long_mass)
    quality = per_game_weight_quality(data, weights)
    assert quality["sampling_measure"] == (
        "component_then_uniform_game_then_uniform_row"
    )
    assert quality["total_weight_per_game"]["std"] > 0.0
    assert quality["sampler_adjusted_weight_per_game"]["std"] == pytest.approx(
        0.0, abs=1e-6
    )


def test_invalid_composite_offsets_fail_closed() -> None:
    class _Composite(dict):
        component_offsets = (0, 4, 3)

    data = _Composite(
        action_taken=np.zeros(3, dtype=np.int16),
        game_seed=np.asarray([1, 1, 2], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="component_offsets"):
        build_value_sample_weights(data, per_game_value_weight=True)
