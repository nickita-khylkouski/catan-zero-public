from __future__ import annotations

import math

import numpy as np
import pytest

from catan_zero.rl.target_reliability import (
    TARGET_RELIABILITY_COLUMNS,
    duplicate_search_reliability_fields,
    unaudited_target_reliability_fields,
)
from tools import a1_stage_c_teacher_alignment as alignment


def test_semantic_bundle_uses_only_explicit_sealed_operator_fallback() -> None:
    assert alignment._semantic_field_bundle(
        {"public_observation": True},
        {"preserve_search_evidence": True},
        ("public_observation", "preserve_search_evidence"),
    ) == {
        "public_observation": True,
        "preserve_search_evidence": True,
    }
    with pytest.raises(alignment.AlignmentError, match="both missing"):
        alignment._semantic_field_bundle(
            {"public_observation": True},
            {},
            ("public_observation", "preserve_search_evidence"),
        )


def test_complete_effective_search_config_binds_resolved_defaults() -> None:
    effective = alignment._complete_effective_search_config(  # noqa: SLF001
        {"n_full": 128, "c_scale": 0.1}
    )

    assert "seed" not in effective
    assert effective["n_full"] == 128
    assert effective["policy_target_min_visits"] == 0
    assert effective["max_root_candidates"] == 16
    assert effective["rng_stream_separation"] is False
    assert set(effective) == {
        field.name
        for field in __import__("dataclasses").fields(alignment.GumbelChanceMCTSConfig)
        if field.name != "seed"
    }


def test_stage_c_execution_identity_is_forced_full_and_seed_schema_bound() -> None:
    execution = alignment.STAGE_C_TARGET_EXECUTION

    assert execution == {
        "schema_version": "a1-stage-c-target-execution-v1",
        "mode": "forced_full_root_reanalysis",
        "force_full_override": True,
        "nominal_n_full": 128,
        "actual_simulations": "authoritative_per_row_deterministic_schedule_result",
        "simulation_accounting_schema": (
            "gumbel_root_candidate_count_plus_sequential_halving_v1"
        ),
        "budget_semantics": (
            "force_full selects n_full; legacy Sequential-Halving schedule accounting "
            "can realize a legal-width-dependent count different from nominal_n_full"
        ),
        "row_seed_schema": alignment.STAGE_C_ROW_SEED_SCHEMA,
    }


def test_operator_mismatch_quarantines_only_stored_policy() -> None:
    active = np.asarray([False, True, True, False], dtype=np.bool_)

    eligible, status = alignment._classify_policy_rows(
        active,
        source_identity_sha256="sha256:old",
        target_identity_sha256="sha256:new",
    )
    assert eligible.tolist() == [False, False, False, False]
    assert status.tolist() == [
        alignment.POLICY_STATUS["inactive_no_stored_policy"],
        alignment.POLICY_STATUS["quarantined_stale_operator"],
        alignment.POLICY_STATUS["quarantined_stale_operator"],
        alignment.POLICY_STATUS["inactive_no_stored_policy"],
    ]

    exact, exact_status = alignment._classify_policy_rows(
        active,
        source_identity_sha256="sha256:same",
        target_identity_sha256="sha256:same",
    )
    np.testing.assert_array_equal(exact, active)
    assert exact_status.tolist() == [
        alignment.POLICY_STATUS["inactive_no_stored_policy"],
        alignment.POLICY_STATUS["eligible_exact_operator"],
        alignment.POLICY_STATUS["eligible_exact_operator"],
        alignment.POLICY_STATUS["inactive_no_stored_policy"],
    ]


def _reliability_rows(*rows: dict[str, object]) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(
            [row[name] for row in rows], dtype=np.asarray(rows[0][name]).dtype
        )
        for name in TARGET_RELIABILITY_COLUMNS
    }


def test_reliability_inventory_never_calls_unaudited_one_confidence_audit() -> None:
    audited = duplicate_search_reliability_fields(
        primary_policy={1: 0.7, 2: 0.3},
        duplicate_policy={1: 0.65, 2: 0.35},
        primary_completed_q={1: 0.3, 2: 0.2},
        duplicate_completed_q={1: 0.25, 2: 0.22},
    )
    unaudited = unaudited_target_reliability_fields()
    classes, receipt = alignment._reliability_inventory(
        _reliability_rows(audited, unaudited), row_count=2
    )
    assert classes.tolist() == [
        alignment.RELIABILITY_CLASS["duplicate_search_audited"],
        alignment.RELIABILITY_CLASS["unaudited_neutral_sentinel"],
    ]
    assert receipt["audited_rows"] == 1
    assert receipt["unaudited_rows"] == 1
    assert receipt["confidence_weighting_authorized"] is False
    assert "never audited evidence" in receipt["unaudited_confidence_semantics"]

    absent_classes, absent = alignment._reliability_inventory({}, row_count=3)
    assert absent_classes.tolist() == [alignment.RELIABILITY_CLASS["not_collected"]] * 3
    assert absent["confidence_weighting_authorized"] is False

    materialized_not_collected = dict(unaudited)
    materialized_not_collected["target_reliability_version"] = np.uint8(0)
    zero_classes, zero_receipt = alignment._reliability_inventory(
        _reliability_rows(materialized_not_collected), row_count=1
    )
    assert zero_classes.tolist() == [alignment.RELIABILITY_CLASS["not_collected"]]
    assert zero_receipt["not_collected_rows"] == 1
    assert zero_receipt["audited_rows"] == 0
    assert zero_receipt["confidence_weighting_authorized"] is False
    assert zero_receipt["storage"] == "schema_columns_present_but_not_collected"


def test_reliability_inventory_refuses_partial_or_forged_evidence() -> None:
    unaudited = unaudited_target_reliability_fields()
    partial = _reliability_rows(unaudited)
    partial.pop("target_reliability_q_margin_duplicate")
    with pytest.raises(alignment.AlignmentError, match="partial"):
        alignment._reliability_inventory(partial, row_count=1)

    forged = dict(unaudited)
    forged["target_reliability_js_divergence"] = np.float32(0.0)
    with pytest.raises(alignment.AlignmentError, match="neutral typed sentinel"):
        alignment._reliability_inventory(_reliability_rows(forged), row_count=1)


def test_policy_surprise_masks_illegal_actions_and_is_finite() -> None:
    data = {
        "target_policy": np.asarray(
            [[0.75, 0.25, 99.0], [0.5, 0.5, 42.0]], dtype=np.float32
        ),
        "prior_policy": np.asarray(
            [[0.5, 0.5, 77.0], [0.5, 0.5, 13.0]], dtype=np.float32
        ),
        "legal_action_ids": np.asarray([[3, 4, -1], [5, 6, -1]], dtype=np.int16),
    }
    surprise = alignment._policy_surprise(data, np.asarray([0, 1]))
    expected = 0.75 * math.log(1.5) + 0.25 * math.log(0.5)
    assert surprise.tolist() == pytest.approx([expected, 0.0], abs=1e-6)


def _game_first_inputs(*, short_training_games: set[int] | None = None) -> dict:
    training_games = np.arange(100, 120, dtype=np.int64)
    validation_games = np.arange(200, 204, dtype=np.int64)
    all_games = np.concatenate((training_games, validation_games))
    phases = alignment.ROOT_BREADTH_REQUIRED_PHASES
    decisions = (5, 15, 35, 65, 105, 155, 205, 7, 25, 85)
    rows: list[int] = []
    games: list[int] = []
    decision_indices: list[int] = []
    phase_values: list[str] = []
    for game in all_games.tolist():
        roots = 7 if short_training_games and game in short_training_games else 10
        for ordinal in range(roots):
            rows.append(len(rows))
            games.append(game)
            decision_indices.append(decisions[ordinal])
            phase_values.append(phases[ordinal % len(phases)])
    count = len(rows)
    return {
        "rows": np.asarray(rows, dtype=np.int64),
        "game_seeds": np.asarray(games, dtype=np.int64),
        "decision_indices": np.asarray(decision_indices, dtype=np.int64),
        "phases": np.asarray(phase_values),
        "legal_widths": np.full(count, 4, dtype=np.int64),
        "surprise": np.linspace(0.0, 1.0, count, dtype=np.float32),
        "reliability_class": np.zeros(count, dtype=np.uint8),
        "policy_status": np.zeros(count, dtype=np.uint8),
        "population_game_seeds": all_games,
        "validation_game_seeds": validation_games,
        "selection_seed": 91,
        "max_rows_per_game": 10,
    }


def test_game_first_selector_is_deterministic_and_seals_split_breadth() -> None:
    kwargs = _game_first_inputs()

    first = alignment._select_game_first(**kwargs, limit=200)  # noqa: SLF001
    second = alignment._select_game_first(**kwargs, limit=200)  # noqa: SLF001

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert first[2:] == second[2:]
    selection = first[3]
    assert selection["breadth_root_count"] == 24 * 8
    assert selection["extra_root_count"] == 8
    assert selection["selected_game_counts"] == {"training": 20, "validation": 4}
    assert selection["root_breadth"]["passed"] is True
    selected_games = kwargs["game_seeds"][first[0]]
    roots_per_game = np.unique(selected_games, return_counts=True)[1]
    assert roots_per_game.min() >= 8
    assert roots_per_game.max() <= 10


def test_game_first_selector_fails_before_reanalysis_on_candidate_coverage() -> None:
    kwargs = _game_first_inputs(short_training_games={100, 101})

    with pytest.raises(
        alignment.AlignmentError,
        match="training has 18/20 games.*requires 19",
    ):
        alignment._select_game_first(**kwargs, limit=200)  # noqa: SLF001


def test_game_first_selector_accepts_exact_95_percent_game_coverage() -> None:
    kwargs = _game_first_inputs(short_training_games={100})

    selection = alignment._select_game_first(  # noqa: SLF001
        **kwargs, limit=200
    )[3]

    training = selection["root_breadth"]["scopes"]["training"]
    assert training["selected_game_count"] == 19
    assert training["unique_game_fraction"] == pytest.approx(0.95)
    assert training["roots_per_represented_game"]["minimum"] >= 8


def test_game_first_selector_fails_before_reanalysis_on_small_budget() -> None:
    kwargs = _game_first_inputs()

    with pytest.raises(
        alignment.AlignmentError,
        match="requested=183 required_at_least=184",
    ):
        alignment._select_game_first(**kwargs, limit=183)  # noqa: SLF001


def test_stage_c_alignment_defaults_fund_game_first_breadth() -> None:
    args = alignment.build_parser().parse_args(
        [
            "plan",
            "--coherent-corpus-admission",
            "admission.json",
            "--target-operator-contract",
            "operator.json",
            "--target-checkpoint",
            "checkpoint.pt",
            "--output-root",
            "output",
            "--write",
            "plan.json",
        ]
    )

    assert args.subset_rows == 65_536
    assert args.max_rows_per_game == 16
