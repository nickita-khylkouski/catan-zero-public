from __future__ import annotations

import numpy as np
import pytest
import torch

from tools.train_bc import _value_targets


def _base_data(**overrides):
    n = 4
    data = {
        "action_taken": np.zeros(n, dtype=np.int16),
        "winner": np.asarray(["RED", "", "", "BLUE"]),
        "player": np.asarray(["RED", "RED", "BLUE", "RED"]),
        "truncated": np.asarray([False, True, True, False]),
        "seat": np.asarray([1, 1, 0, 1], dtype=np.int8),
        # PLAYER_NAMES order: BLUE, RED, ORANGE, WHITE.
        "final_actual_vps": np.asarray(
            [
                [0, 10, 0, 0],  # row 0: clean win, unused by the soft path
                [4, 8, 0, 0],  # row 1: truncated, RED (seat=1) leads 8-4
                [7, 3, 0, 0],  # row 2: truncated, BLUE (seat=0) leads 7-3
                [0, 10, 0, 0],  # row 3: clean win, unused by the soft path
            ],
            dtype=np.int16,
        ),
        "has_final_actual_vps": np.asarray([True, False, False, True]),
        "final_public_vps": np.zeros((n, 4), dtype=np.int16),
        "has_final_public_vps": np.asarray([True, False, False, True]),
    }
    data.update(overrides)
    return data


def test_truncated_vp_margin_disabled_by_default_matches_prior_behavior():
    """Backward compatibility: omitting the new kwarg (or passing 0.0) must behave exactly
    like the pre-F3 function -- truncated rows contribute nothing to the value target,
    for BOTH the policy-safe (outcome/has_outcome) and value-specific views."""
    data = _base_data()
    batch = np.arange(4)

    (
        outcome,
        vp_target,
        has_outcome,
        has_vp,
        value_outcome,
        value_has_outcome,
        confidence,
    ) = _value_targets(data, batch, torch.device("cpu"), vps_to_win=10)

    assert has_outcome.tolist() == [True, False, False, True]
    assert value_has_outcome.tolist() == [True, False, False, True]
    assert confidence.tolist() == pytest.approx([1.0, 0.0, 0.0, 1.0])


def test_truncated_vp_margin_fills_in_a_soft_outcome_at_reduced_confidence():
    """FIX F3: a truncated row with valid final_actual_vps/seat data gets a soft value
    label derived from the VP margin at truncation, at the requested reduced weight --
    clean (non-truncated) rows are completely unaffected. The soft fill must ONLY show up
    in the value-specific view (value_outcome/value_has_outcome), never in the
    policy-safe outcome/has_outcome (which feeds POLICY advantage weighting elsewhere and
    must be untouched by this value-only fix)."""
    data = _base_data()
    batch = np.arange(4)

    (
        outcome,
        vp_target,
        has_outcome,
        has_vp,
        value_outcome,
        value_has_outcome,
        confidence,
    ) = _value_targets(
        data,
        batch,
        torch.device("cpu"),
        vps_to_win=10,
        truncated_vp_margin_value_weight=0.25,
    )

    # Policy-safe view is completely unaffected by F3.
    assert has_outcome.tolist() == [True, False, False, True]

    # Value-specific view is filled in for the truncated rows.
    assert value_has_outcome.tolist() == [True, True, True, True]
    # Row 0/3 (clean outcomes) keep full confidence; row 1/2 (soft-filled) get 0.25.
    assert confidence.tolist() == pytest.approx([1.0, 0.25, 0.25, 1.0])
    # Row 1: RED (seat=1) leads 8-4 -> margin = (8-4)/10 = 0.4 (positive, RED is "me").
    assert float(value_outcome[1]) == pytest.approx(0.4)
    # Row 2: BLUE (seat=0) leads 7-3 -> margin = (7-3)/10 = 0.4 from BLUE's perspective.
    assert float(value_outcome[2]) == pytest.approx(0.4)
    # Clean rows' original outcome values are untouched by the soft path.
    assert float(value_outcome[0]) == pytest.approx(1.0)
    assert float(value_outcome[3]) == pytest.approx(-1.0)


def test_truncated_vp_margin_clips_to_unit_range():
    """A lopsided VP margin must clip to [-1, 1], not overshoot."""
    data = _base_data(
        final_actual_vps=np.asarray(
            [
                [0, 10, 0, 0],
                [0, 9, 0, 0],  # RED (seat=1) leads 9-0 -> raw margin 0.9, within range
                [10, 0, 0, 0],  # BLUE (seat=0) leads 10-0 -> raw margin 1.0, at the edge
                [0, 10, 0, 0],
            ],
            dtype=np.int16,
        ),
    )
    batch = np.arange(4)

    _, _, _, _, value_outcome, _, _ = _value_targets(
        data, batch, torch.device("cpu"), vps_to_win=10, truncated_vp_margin_value_weight=0.25
    )

    assert float(value_outcome[1]) == pytest.approx(0.9)
    assert float(value_outcome[2]) == pytest.approx(1.0)


def test_truncated_vp_margin_does_not_override_a_real_clean_outcome():
    """A row that already has a clean outcome must never be touched by the soft path,
    even if truncated_vp_margin_value_weight is enabled."""
    data = _base_data()
    batch = np.arange(4)

    _, _, _, _, value_outcome, value_has_outcome, confidence = _value_targets(
        data, batch, torch.device("cpu"), vps_to_win=10, truncated_vp_margin_value_weight=0.25
    )

    assert bool(value_has_outcome[0]) is True
    assert float(confidence[0]) == pytest.approx(1.0)
    assert float(value_outcome[0]) == pytest.approx(1.0)


def test_truncated_vp_margin_skips_rows_with_no_seat_field():
    """Without a `seat` column at all, the soft path must be a no-op (matches the existing
    guard pattern used by the hard vp_target computation just below it)."""
    data = _base_data()
    del data["seat"]
    batch = np.arange(4)

    _, _, has_outcome, _, value_outcome, value_has_outcome, confidence = _value_targets(
        data, batch, torch.device("cpu"), vps_to_win=10, truncated_vp_margin_value_weight=0.25
    )

    assert has_outcome.tolist() == [True, False, False, True]
    assert value_has_outcome.tolist() == [True, False, False, True]
    assert confidence.tolist() == pytest.approx([1.0, 0.0, 0.0, 1.0])


def test_public_regime_truncated_margin_never_uses_opponent_hidden_actual_vp():
    data = _base_data(
        final_actual_vps=np.asarray(
            [[0, 10, 0, 0], [0, 9, 0, 0], [9, 0, 0, 0], [0, 10, 0, 0]],
            dtype=np.int16,
        ),
        final_public_vps=np.asarray(
            [[0, 10, 0, 0], [4, 5, 0, 0], [5, 4, 0, 0], [0, 10, 0, 0]],
            dtype=np.int16,
        ),
    )

    _, _, _, _, value_outcome, value_has_outcome, _ = _value_targets(
        data,
        np.arange(4),
        torch.device("cpu"),
        vps_to_win=10,
        truncated_vp_margin_value_weight=0.25,
        public_information_only=True,
    )

    assert value_has_outcome.tolist() == [True, True, True, True]
    # Public RED margin is (5-4)/10 and public BLUE margin is (5-4)/10.
    # Hidden actual VP would have produced 0.9 for each row instead.
    assert float(value_outcome[1]) == pytest.approx(0.1)
    assert float(value_outcome[2]) == pytest.approx(0.1)
