from __future__ import annotations

import numpy as np
import pytest

from tools.train_bc import (
    _MemmapCategoricalColumn,
    _apply_authenticated_value_training_scope,
    _forced_value_nominal_population_measure,
    _value_component_active_dose_for_batch,
    _value_independent_evidence_report,
    _value_training_scope_report,
    apply_value_player_outcome_balance,
    build_value_sample_weights,
    forced_action_type_value_mass_quality,
    per_game_weight_quality,
    value_player_outcome_balance_quality,
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


def test_value_evidence_reports_independent_games_not_repeated_rows() -> None:
    data = {
        "game_seed": np.asarray([10, 10, 20], dtype=np.int64),
        "winner": np.asarray(["RED", "RED", "BLUE"]),
        "truncated": np.asarray([False, False, False]),
    }
    report = _value_independent_evidence_report(
        data,
        np.arange(3, dtype=np.int64),
        np.asarray([0.5, 0.5, 1.0], dtype=np.float32),
    )

    assert report["schema_version"] == "value-independent-evidence-v2"
    assert report["status"] == "ok"
    assert report["clean_terminal_value_rows"] == 3
    assert report["independent_terminal_games"] == 2
    assert report["row_labels_per_independent_outcome"] == pytest.approx(1.5)
    assert report["game_weight_effective_sample_size"] == pytest.approx(2.0)
    assert report["game_weight_ess_fraction"] == pytest.approx(1.0)
    assert report["contradictory_terminal_outcome_games"] == 0


def test_value_evidence_surfaces_contradictory_outcomes_within_game() -> None:
    data = {
        "game_seed": np.asarray([7, 7], dtype=np.int64),
        "winner": np.asarray(["RED", "BLUE"]),
        "truncated": np.asarray([False, False]),
    }
    report = _value_independent_evidence_report(
        data,
        np.arange(2, dtype=np.int64),
        np.ones(2, dtype=np.float32),
    )

    assert report["status"] == "contradictory_game_outcomes"
    assert report["contradictory_terminal_outcome_games"] == 1


def test_value_evidence_namespaces_reused_seeds_by_component() -> None:
    class _Composite(dict):
        @staticmethod
        def component_indices_for_rows(rows):
            indices = np.asarray(rows, dtype=np.int64)
            return np.where(indices < 2, 0, 1)

    data = _Composite(
        game_seed=np.asarray([7, 7, 7, 7], dtype=np.int64),
        winner=np.asarray(["RED", "RED", "BLUE", "BLUE"]),
        truncated=np.asarray([False, False, False, False]),
    )
    report = _value_independent_evidence_report(
        data,
        np.arange(4, dtype=np.int64),
        np.asarray([0.5, 0.5, 1.0, 1.0], dtype=np.float32),
    )

    assert report["status"] == "ok"
    assert report["independent_terminal_games"] == 2
    assert report["contradictory_terminal_outcome_games"] == 0
    assert report["game_weight_effective_sample_size"] == pytest.approx(1.8)
    assert report["game_identity_namespace"] == "component_id+game_seed"


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


def test_forced_action_mass_report_uses_only_training_objective_measure() -> None:
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
                [build_road, roll],
                [roll, -1],
            ],
            dtype=np.int32,
        ),
    }
    # Rows 0 and 2 are the training split. Row 1 is a forced validation row
    # with enormous mass; row 3 is another excluded forced row. The supplied
    # vector is already the sampler-adjusted optimizer objective measure.
    objective_measure = np.asarray([0.25, 1000.0, 0.75, 1000.0], dtype=np.float64)

    report = forced_action_type_value_mass_quality(
        data,
        objective_measure,
        row_indices=np.asarray([0, 2], dtype=np.int64),
        objective_measure="coverage_importance_training_value_loss_v1",
        action_catalog=catalog,
        configured_weights={"ROLL": 1.0},
    )

    assert report["schema_version"] == "forced-action-type-value-mass-v2"
    assert report["scope"] == "training_rows"
    assert report["scope_rows"] == 2
    assert report["objective_measure"] == (
        "coverage_importance_training_value_loss_v1"
    )
    assert report["forced_rows"] == 1
    assert report["forced_row_fraction"] == pytest.approx(0.5)
    assert report["effective_forced_value_mass"] == pytest.approx(0.25)
    assert report["effective_total_value_mass"] == pytest.approx(1.0)
    assert report["effective_forced_value_mass_fraction"] == pytest.approx(0.25)
    assert report["by_action_type"]["ROLL"]["rows"] == 1
    assert "END_TURN" not in report["by_action_type"]


def test_forced_value_nominal_measure_composes_sampler_and_label_confidence() -> None:
    weights = np.asarray([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    train_indices = np.asarray([1, 3], dtype=np.int64)
    confidence = np.asarray([0.5, 0.25], dtype=np.float32)

    weighted, weighted_name = _forced_value_nominal_population_measure(
        weights,
        train_indices,
        base_sampler="weighted_replacement_v1",
        component_game_sampling=np.asarray([0.25, 0.75], dtype=np.float64),
        objective_confidence=confidence,
    )
    coverage_weights = weights.copy()
    coverage_weights[train_indices] *= np.asarray([0.5, 1.5], dtype=np.float32)
    coverage, coverage_name = _forced_value_nominal_population_measure(
        coverage_weights,
        train_indices,
        base_sampler="coverage_importance_v1",
        component_game_sampling=np.asarray([0.25, 0.75], dtype=np.float64),
        objective_confidence=confidence,
    )

    assert weighted.tolist() == pytest.approx([2.5, 7.5])
    assert weighted_name == (
        "authenticated_weighted_replacement_scalar_outcome_value_loss_v1"
    )
    assert coverage.tolist() == pytest.approx([5.0, 15.0])
    assert coverage.tolist() == pytest.approx((2.0 * weighted).tolist())
    assert coverage[0] / coverage.sum() == pytest.approx(
        weighted[0] / weighted.sum()
    )
    assert coverage_name == "coverage_importance_scalar_outcome_value_loss_v1"
    assert weights.tolist() == [10.0, 20.0, 30.0, 40.0]


@pytest.mark.parametrize(
    ("row_indices", "match"),
    [
        (np.asarray([0, 0], dtype=np.int64), "must be unique"),
        (np.asarray([-1], dtype=np.int64), "out of range"),
        (np.asarray([[0]], dtype=np.int64), "one-dimensional"),
    ],
)
def test_forced_action_mass_report_rejects_invalid_scope(
    row_indices: np.ndarray, match: str
) -> None:
    catalog = ActionCatalog(("RED", "BLUE"))
    roll = _catalog_action_id(catalog, "ROLL")
    data = {
        "action_taken": np.asarray([roll], dtype=np.int16),
        "legal_action_ids": np.asarray([[roll]], dtype=np.int32),
    }

    with pytest.raises(ValueError, match=match):
        forced_action_type_value_mass_quality(
            data,
            np.ones(1, dtype=np.float32),
            row_indices=row_indices,
            objective_measure="test_measure",
            action_catalog=catalog,
            configured_weights={"ROLL": 1.0},
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


def test_forced_row_action_type_map_rejects_action_that_is_not_sole_legal() -> None:
    catalog = ActionCatalog(("RED", "BLUE"))
    roll = _catalog_action_id(catalog, "ROLL")
    end_turn = _catalog_action_id(catalog, "END_TURN")
    data = {
        "action_taken": np.asarray([end_turn], dtype=np.int16),
        "legal_action_ids": np.asarray([[roll, -1]], dtype=np.int32),
    }
    with pytest.raises(SystemExit, match="does not match its sole legal action"):
        build_value_sample_weights(
            data,
            forced_row_value_action_type_weights={"ROLL": 1.0, "END_TURN": 1.0},
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


class _ValueBalanceComposite(dict):
    def __init__(self, corpora: tuple[dict, ...], ratios: tuple[float, ...]):
        self.corpora = corpora
        self.component_offsets = tuple(
            np.concatenate(
                (
                    np.asarray([0], dtype=np.int64),
                    np.cumsum(
                        [len(corpus["action_taken"]) for corpus in corpora],
                        dtype=np.int64,
                    ),
                )
            ).tolist()
        )
        self.component_game_sampling_ratios = ratios
        self.component_ids = tuple(f"component_{index}" for index in range(len(corpora)))
        self.value_training_scope_authenticated = True
        self.value_training_component_indices = tuple(range(len(corpora)))
        super().__init__(
            action_taken=np.concatenate(
                [corpus["action_taken"] for corpus in corpora]
            )
        )


def _outcome_component(
    game_seed: list[int], player: list[str], winner: list[str]
) -> dict:
    n = len(game_seed)
    return {
        "action_taken": np.zeros(n, dtype=np.int16),
        "game_seed": np.asarray(game_seed, dtype=np.int64),
        "player": np.asarray(player),
        "winner": np.asarray(winner),
        "truncated": np.zeros(n, dtype=np.bool_),
    }


def test_sampler_balanced_value_mode_balances_bilateral_and_one_sided_games() -> None:
    bilateral = _outcome_component(
        [1, 1, 1, 1, 2, 2, 2, 2],
        ["RED", "RED", "RED", "BLUE", "RED", "BLUE", "BLUE", "BLUE"],
        ["RED"] * 4 + ["BLUE"] * 4,
    )
    one_sided = _outcome_component(
        [3, 3, 4, 4, 5, 5],
        ["RED", "RED", "BLUE", "BLUE", "RED", "RED"],
        ["RED", "RED", "BLUE", "BLUE", "BLUE", "BLUE"],
    )
    data = _ValueBalanceComposite((bilateral, one_sided), (0.6, 0.4))
    before = np.ones(len(data["action_taken"]), dtype=np.float32)
    rows = np.arange(len(before), dtype=np.int64)

    after = apply_value_player_outcome_balance(
        data, before, rows, mode="sampler_balanced_v1"
    )
    report = value_player_outcome_balance_quality(
        data,
        before,
        after,
        rows,
        mode="sampler_balanced_v1",
    )

    assert report["before"]["sampler_adjusted_target_mean"] != pytest.approx(0.0)
    assert report["before"]["components"]["component_0"][
        "sampler_adjusted_target_mean"
    ] != pytest.approx(0.0)
    assert report["after"]["sampler_adjusted_target_mean"] == pytest.approx(
        0.0, abs=1e-7
    )
    assert report["after"]["components"]["component_0"][
        "sampler_adjusted_target_mean"
    ] == pytest.approx(0.0, abs=1e-7)
    assert report["after"]["components"]["component_1"][
        "sampler_adjusted_target_mean"
    ] == pytest.approx(0.0, abs=1e-7)
    assert report["after"]["bilateral_games"] == 2
    assert report["after"]["one_sided_positive_games"] == 2
    assert report["after"]["one_sided_negative_games"] == 1
    assert report["after"]["outcome_effective_sample_size"] == pytest.approx(2.0)


def test_sampler_balanced_value_mode_fails_when_component_has_one_outcome_sign() -> None:
    one_sign = _outcome_component(
        [1, 1, 2, 2],
        ["RED", "RED", "BLUE", "BLUE"],
        ["RED", "RED", "BLUE", "BLUE"],
    )
    data = _ValueBalanceComposite((one_sign,), (1.0,))
    with pytest.raises(SystemExit, match="requires both outcome signs"):
        apply_value_player_outcome_balance(
            data,
            np.ones(4, dtype=np.float32),
            np.arange(4, dtype=np.int64),
            mode="sampler_balanced_v1",
        )


def test_sampler_balanced_value_mode_requires_complete_outcome_provenance() -> None:
    malformed = _outcome_component(
        [1, 1, 2, 2],
        ["RED", "RED", "BLUE", "BLUE"],
        ["RED", "RED", "RED", "RED"],
    )
    del malformed["truncated"]
    data = _ValueBalanceComposite((malformed,), (1.0,))
    with pytest.raises(SystemExit, match="lacks required outcome provenance"):
        apply_value_player_outcome_balance(
            data,
            np.ones(4, dtype=np.float32),
            np.arange(4, dtype=np.int64),
            mode="sampler_balanced_v1",
        )


@pytest.mark.parametrize("field", ["player", "winner"])
@pytest.mark.parametrize("bad_code", [-1, 2])
def test_sampler_balanced_value_mode_rejects_invalid_categorical_identity_codes(
    field: str, bad_code: int
) -> None:
    categories = np.asarray(["RED", "BLUE"])
    player_codes = np.asarray([1, 1, 0, 0], dtype=np.int32)
    winner_codes = np.asarray([1, 1, 1, 1], dtype=np.int32)
    if field == "player":
        player_codes[:2] = bad_code
    else:
        winner_codes[:2] = bad_code
    component = {
        "action_taken": np.zeros(4, dtype=np.int16),
        "game_seed": np.asarray([1, 1, 2, 2], dtype=np.int64),
        "player": _MemmapCategoricalColumn(player_codes, categories),
        "winner": _MemmapCategoricalColumn(winner_codes, categories),
        "truncated": np.zeros(4, dtype=np.bool_),
    }
    data = _ValueBalanceComposite((component,), (1.0,))

    with pytest.raises(
        SystemExit,
        match=(
            "categorical identity code is out of range: "
            rf"field='{field}' categories=2 examples=\[{bad_code}\]"
        ),
    ):
        apply_value_player_outcome_balance(
            data,
            np.ones(4, dtype=np.float32),
            np.arange(4, dtype=np.int64),
            mode="sampler_balanced_v1",
        )


def test_sampler_balanced_value_mode_preserves_truncated_rows_and_reports_them() -> None:
    component = _outcome_component(
        [1, 1, 2, 2, 3, 3],
        ["RED", "RED", "RED", "RED", "RED", "RED"],
        ["RED", "RED", "BLUE", "BLUE", "", ""],
    )
    component["truncated"][4:] = True
    data = _ValueBalanceComposite((component,), (1.0,))
    before = np.asarray([1.0, 3.0, 2.0, 6.0, 5.0, 7.0], dtype=np.float32)
    rows = np.arange(6, dtype=np.int64)

    after = apply_value_player_outcome_balance(
        data, before, rows, mode="sampler_balanced_v1"
    )
    report = value_player_outcome_balance_quality(
        data,
        before,
        after,
        rows,
        mode="sampler_balanced_v1",
    )

    assert after[4:].tobytes() == before[4:].tobytes()
    assert after[:4].tobytes() != before[:4].tobytes()
    assert report["fit_partition_rows"] == 6
    assert report["value_enabled_partition_rows"] == 6
    assert report["completed_non_truncated_rows"] == 4
    assert report["truncated_rows"] == 2
    assert report["completed_non_truncated_games"] == 2
    assert report["truncated_games"] == 1
    assert report["normalization_policy"] == (
        "preserve_actual_completed_sampler_mass_when_truncated_else_legacy_mean_one"
    )
    assert report["before"]["rows"] == 4
    assert report["after"]["rows"] == 4
    assert report["before"]["completed_non_truncated_weight_sum"] == pytest.approx(12.0)
    assert report["after"]["completed_non_truncated_weight_sum"] == pytest.approx(12.0)
    assert report["before"]["truncated_weight_sum"] == pytest.approx(12.0)
    assert report["after"]["truncated_weight_sum"] == pytest.approx(12.0)
    assert report["after"]["total_effective_mass"] == pytest.approx(
        report["before"]["total_effective_mass"]
    )
    assert report["before"]["games"] == 2
    assert report["after"]["games"] == 2
    assert "" not in report["after"]["player_outcome_cells"]
    component_report = report["after"]["components"]["component_0"]
    assert component_report["partition_rows"] == 6
    assert component_report["completed_non_truncated_rows"] == 4
    assert component_report["truncated_rows"] == 2
    assert component_report["partition_games"] == 3
    assert component_report["completed_non_truncated_games"] == 2
    assert component_report["truncated_games"] == 1


def test_sampler_balanced_value_mode_preserves_game_uniform_mass_with_truncation() -> None:
    component = _outcome_component(
        [1, 2, 2, 2, 3, 3],
        ["RED"] * 6,
        ["RED", "BLUE", "BLUE", "BLUE", "", ""],
    )
    component["truncated"][4:] = True
    data = _ValueBalanceComposite((component,), (1.0,))
    before = np.asarray([2.0, 1.0, 3.0, 5.0, 7.0, 9.0], dtype=np.float32)
    rows = np.arange(6, dtype=np.int64)

    after = apply_value_player_outcome_balance(
        data, before, rows, mode="sampler_balanced_v1"
    )
    report = value_player_outcome_balance_quality(
        data,
        before,
        after,
        rows,
        mode="sampler_balanced_v1",
    )

    assert after[4:].tobytes() == before[4:].tobytes()
    assert float(after[:4].sum(dtype=np.float64)) != pytest.approx(
        float(before[:4].sum(dtype=np.float64))
    )
    assert report["before"]["total_effective_mass"] == pytest.approx(5.0 / 3.0)
    assert report["after"]["total_effective_mass"] == pytest.approx(5.0 / 3.0)


def test_sampler_balanced_value_mode_preserves_row_uniform_mass_with_truncation() -> None:
    data = _outcome_component(
        [1, 1, 2, 2, 3, 3],
        ["RED"] * 6,
        ["RED", "RED", "BLUE", "BLUE", "", ""],
    )
    data["truncated"][4:] = True
    before = np.asarray([1.0, 3.0, 2.0, 6.0, 5.0, 7.0], dtype=np.float32)
    rows = np.arange(6, dtype=np.int64)

    after = apply_value_player_outcome_balance(
        data, before, rows, mode="sampler_balanced_v1"
    )
    report = value_player_outcome_balance_quality(
        data,
        before,
        after,
        rows,
        mode="sampler_balanced_v1",
    )

    assert after[4:].tobytes() == before[4:].tobytes()
    assert report["before"]["total_effective_mass"] == pytest.approx(2.0)
    assert report["after"]["total_effective_mass"] == pytest.approx(2.0)


def test_sampler_balanced_value_mode_retains_completed_only_mean_one_normalization() -> None:
    component = _outcome_component(
        [1, 1, 2, 2],
        ["RED", "RED", "RED", "RED"],
        ["RED", "RED", "BLUE", "BLUE"],
    )
    data = _ValueBalanceComposite((component,), (1.0,))

    after = apply_value_player_outcome_balance(
        data,
        np.asarray([1.0, 3.0, 2.0, 6.0], dtype=np.float32),
        np.arange(4, dtype=np.int64),
        mode="sampler_balanced_v1",
    )

    expected = np.asarray([0.5, 1.5, 0.5, 1.5], dtype=np.float32)
    assert after.tobytes() == expected.tobytes()


def test_sampler_balanced_value_mode_rejects_mixed_truncation_within_game() -> None:
    component = _outcome_component(
        [1, 1, 2, 2, 3, 3],
        ["RED", "RED", "RED", "RED", "RED", "RED"],
        ["RED", "RED", "BLUE", "BLUE", "", ""],
    )
    component["truncated"][4] = True
    data = _ValueBalanceComposite((component,), (1.0,))

    with pytest.raises(SystemExit, match="mixed truncation state within a game"):
        apply_value_player_outcome_balance(
            data,
            np.ones(6, dtype=np.float32),
            np.arange(6, dtype=np.int64),
            mode="sampler_balanced_v1",
        )


def test_sampler_balanced_value_mode_fails_when_partition_omits_enabled_component() -> None:
    first = _outcome_component(
        [1, 1, 2, 2],
        ["RED", "RED", "BLUE", "BLUE"],
        ["RED", "RED", "RED", "RED"],
    )
    second = _outcome_component(
        [3, 3, 4, 4],
        ["RED", "RED", "BLUE", "BLUE"],
        ["RED", "RED", "RED", "RED"],
    )
    data = _ValueBalanceComposite((first, second), (0.5, 0.5))
    with pytest.raises(SystemExit, match="omits an authenticated"):
        apply_value_player_outcome_balance(
            data,
            np.ones(8, dtype=np.float32),
            np.arange(4, dtype=np.int64),
            mode="sampler_balanced_v1",
        )


def test_validation_winner_changes_cannot_change_training_value_weights() -> None:
    base = _outcome_component(
        [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        ["RED"] * 10,
        ["RED", "RED", "BLUE", "BLUE", "RED", "RED", "BLUE", "BLUE", "BLUE", "BLUE"],
    )
    changed = {key: np.array(value, copy=True) for key, value in base.items()}
    changed["winner"][6:8] = np.asarray(["RED", "RED"])
    train_rows = np.arange(4, dtype=np.int64)
    validation_rows = np.arange(4, 10, dtype=np.int64)
    initial = np.ones(10, dtype=np.float32)

    first = apply_value_player_outcome_balance(
        base, initial, train_rows, mode="sampler_balanced_v1"
    )
    first = apply_value_player_outcome_balance(
        base, first, validation_rows, mode="sampler_balanced_v1"
    )
    second = apply_value_player_outcome_balance(
        changed, initial, train_rows, mode="sampler_balanced_v1"
    )
    second = apply_value_player_outcome_balance(
        changed, second, validation_rows, mode="sampler_balanced_v1"
    )

    assert first[train_rows].tobytes() == second[train_rows].tobytes()
    assert first[validation_rows].tobytes() != second[validation_rows].tobytes()


def test_composite_equal_per_game_weighting_reports_inert_when_means_match() -> None:
    class _CompositeV2(dict):
        component_offsets = (0, 2, 4)
        component_game_sampling_ratios = (0.5, 0.5)

    data = _CompositeV2(
        action_taken=np.zeros(4, dtype=np.int16),
        game_seed=np.asarray([1, 1, 2, 2], dtype=np.int64),
    )
    diagnostics: dict[str, object] = {}
    build_value_sample_weights(
        data,
        per_game_value_weight=True,
        per_game_weight_diagnostics=diagnostics,
    )
    assert diagnostics["requested"] is True
    assert diagnostics["effective"] is False
    assert diagnostics["denominator"]["std"] == pytest.approx(0.0)
