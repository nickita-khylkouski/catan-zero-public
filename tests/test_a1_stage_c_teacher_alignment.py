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
        "effective_simulations": 128,
        "budget_source": "force_full_overrides_playout_cap_and_wide_root_gates",
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


def test_stratified_subset_is_deterministic_and_caps_each_game() -> None:
    rows = np.arange(12, dtype=np.int64)
    games = np.repeat(np.asarray([10, 11, 12, 13], dtype=np.int64), 3)
    decisions = np.tile(np.arange(3, dtype=np.int64), 4)
    phases = np.asarray(["opening"] * 6 + ["play_turn"] * 6)
    widths = np.asarray([2, 4, 8, 16, 32, 3] * 2, dtype=np.int64)
    surprise = np.linspace(0.0, 1.1, 12, dtype=np.float32)
    reliability = np.asarray([0, 1, 2] * 4, dtype=np.uint8)
    status = np.asarray([0, 1, 2] * 4, dtype=np.uint8)
    kwargs = dict(
        rows=rows,
        game_seeds=games,
        decision_indices=decisions,
        phases=phases,
        legal_widths=widths,
        surprise=surprise,
        reliability_class=reliability,
        policy_status=status,
        limit=8,
        selection_seed=91,
        max_rows_per_game=2,
    )
    first = alignment._select_stratified(**kwargs)
    second = alignment._select_stratified(**kwargs)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert first[2] == second[2]
    selected_games = games[first[0]]
    assert len(first[0]) == 8
    assert max(np.unique(selected_games, return_counts=True)[1]) <= 2
