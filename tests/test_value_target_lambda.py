from __future__ import annotations

import numpy as np
import pytest
import torch

# SW-0 integration decision (2026-07-08): this test file (landed via cat-18
# "commit-debt") encodes the SUPERSEDED f94/f76 value-target-lambda convention:
#   target = (1 - lambda) * z + lambda * target_scores[played_action]
#   default lambda = 0.0 (pure realised-outcome z), keyed off the
#   target_scores / afterstate_target columns via `_played_action_bootstrap_value`.
# The reviewed-canonical SW-0 convention is CAT-39's `--value-target-lambda`:
#   target = lambda * z + (1 - lambda) * V_search
#   default lambda = 1.0 (pure z), keyed off the root_value / root_value_mask
#   columns, applied in `_train_xdim_batch` (scalar space for the MSE arm,
#   distribution space for the HL-Gauss head).
# The two are genuinely incompatible (opposite blend direction, opposite
# default, different value source, different call site), so the f94 convention
# and its helpers (`_played_action_bootstrap_value`, a `value_target_lambda`
# kwarg on `_value_targets`) were NOT adopted -- their implementation exists
# only in the unmerged f94/speed-czar tree, never in any SW-0 branch. Skipped
# rather than deleted so the decision is greppable and revisitable.
pytest.skip(
    "Superseded f94/f76 value-target-lambda convention; CAT-39's convention is "
    "canonical for SW-0 (see module header). Implementation intentionally not merged.",
    allow_module_level=True,
)

from tools.train_bc import _played_action_bootstrap_value, _value_targets  # noqa: E402


def _base_data(**overrides):
    n = 6
    # legal_action_ids width 3; action_taken picks a specific column per row so the
    # target_scores/afterstate_target fixtures below can target exactly the row's
    # PLAYED action (mirrors _build_decision_row's ragged, per-legal-action layout
    # in gumbel_self_play.py). Two-source precedence under test: target_scores (the
    # searched completed-Q) wins whenever usable; afterstate_target (a one-ply
    # chance-expectation, e.g. from a forced ROLL) is the FALLBACK used only where
    # target_scores is unusable for that row.
    data = {
        "action_taken": np.asarray([10, 20, 30, 10, 10, 20], dtype=np.int16),
        "legal_action_ids": np.asarray(
            [
                [10, 20, 30],
                [10, 20, 30],
                [10, 20, 30],
                [10, 20, 30],
                [10, 20, 30],
                [10, 20, 30],
            ],
            dtype=np.int16,
        ),
        "winner": np.asarray(["RED", "", "", "BLUE", "", ""]),
        "player": np.asarray(["RED", "RED", "BLUE", "RED", "RED", "RED"]),
        "truncated": np.asarray([False, True, True, False, False, True]),
        "seat": np.asarray([1, 1, 0, 1, 1, 1], dtype=np.int8),
        # PLAYER_NAMES order: BLUE, RED, ORANGE, WHITE.
        "final_actual_vps": np.asarray(
            [
                [0, 10, 0, 0],  # row 0: clean win, unused by the soft path
                [4, 8, 0, 0],  # row 1: truncated, RED (seat=1) leads 8-4 -> margin 0.4
                [7, 3, 0, 0],  # row 2: truncated, BLUE (seat=0) leads 7-3 -> margin 0.4
                [0, 10, 0, 0],  # row 3: clean win, unused by the soft path
                [0, 0, 0, 0],  # row 4: neither terminal nor truncated -- no z at all
                [3, 5, 0, 0],  # row 5: truncated, RED (seat=1) leads 5-3 -> margin 0.2
            ],
            dtype=np.int16,
        ),
        "has_final_actual_vps": np.asarray([True, False, False, True, False, False]),
        "final_public_vps": np.zeros((n, 4), dtype=np.int16),
        "has_final_public_vps": np.asarray([True, False, False, True, False, False]),
        # Row 0: played action (col 0) has BOTH a target_scores entry (0.6) and an
        #        afterstate_target entry (0.3) -- precedence must pick target_scores.
        # Row 1: played action (col 1) has target_scores=0.2 only (no afterstate) --
        #        target-scores-only coverage.
        # Row 2: played action (col 2) has NO target_scores (e.g. a forced decision
        #        never visited during search) but DOES have afterstate_target=0.5 --
        #        the fallback must engage here.
        # Row 3: played action (col 0) has BOTH target_scores=-0.9 and
        #        afterstate_target=0.1 -- precedence must again pick target_scores.
        # Row 4: played action (col 0) has target_scores=0.7 (no afterstate), but the
        #        row has no z/proxy target at all (has_outcome=False, not truncated)
        #        -- lambda-mixing must NOT invent new coverage here.
        # Row 5: played action (col 1) has NEITHER target_scores NOR afterstate_target
        #        -- combined mask is False, mix must leave the row at pure
        #        (truncated-proxy) z.
        "target_scores": np.asarray(
            [
                [0.6, np.nan, np.nan],
                [np.nan, 0.2, np.nan],
                [np.nan, np.nan, np.nan],
                [-0.9, np.nan, np.nan],
                [0.7, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
            ],
            dtype=np.float32,
        ),
        "target_scores_mask": np.asarray(
            [
                [True, False, False],
                [False, True, False],
                [False, False, False],
                [True, False, False],
                [True, False, False],
                [False, False, False],
            ],
        ),
        "afterstate_target": np.asarray(
            [
                [0.3, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, 0.5],
                [0.1, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
            ],
            dtype=np.float32,
        ),
        "afterstate_target_mask": np.asarray(
            [
                [True, False, False],
                [False, False, False],
                [False, False, True],
                [True, False, False],
                [False, False, False],
                [False, False, False],
            ],
        ),
    }
    data.update(overrides)
    return data


def test_played_action_bootstrap_value_reads_played_action_column():
    """The helper must read the SPECIFIC legal-action column matching each row's
    action_taken (the PLAYED action), not e.g. the first masked column or a
    row-wide reduction."""
    data = _base_data()
    batch = np.arange(6)

    values, mask, is_target_scores = _played_action_bootstrap_value(data, batch)

    assert mask.tolist() == [True, True, True, True, True, False]
    assert is_target_scores.tolist() == [True, True, False, True, True, False]
    assert values[mask].tolist() == pytest.approx([0.6, 0.2, 0.5, -0.9, 0.7])


def test_played_action_bootstrap_value_target_scores_wins_when_both_present():
    """Row 0/3 have BOTH a usable target_scores AND afterstate_target entry at the
    played action -- precedence must pick target_scores (the richer, searched
    completed-Q), never the one-ply afterstate fallback."""
    data = _base_data()
    batch = np.arange(6)

    values, mask, is_target_scores = _played_action_bootstrap_value(data, batch)

    assert bool(is_target_scores[0]) is True
    assert float(values[0]) == pytest.approx(0.6)
    assert bool(is_target_scores[3]) is True
    assert float(values[3]) == pytest.approx(-0.9)


def test_played_action_bootstrap_value_falls_back_to_afterstate():
    """Row 2's played action has NO target_scores entry (e.g. a forced decision
    never visited during search) but DOES have a real afterstate_target -- the
    fallback must engage and supply that value."""
    data = _base_data()
    batch = np.arange(6)

    values, mask, is_target_scores = _played_action_bootstrap_value(data, batch)

    assert bool(mask[2]) is True
    assert bool(is_target_scores[2]) is False
    assert float(values[2]) == pytest.approx(0.5)


def test_played_action_bootstrap_value_no_coverage_when_neither_source_has_it():
    """Row 5 has neither a usable target_scores nor afterstate_target entry at the
    played action -- combined mask must be False."""
    data = _base_data()
    batch = np.arange(6)

    _, mask, _ = _played_action_bootstrap_value(data, batch)

    assert bool(mask[5]) is False


def test_played_action_bootstrap_value_uses_target_scores_only_when_afterstate_absent():
    """A corpus with target_scores but no afterstate_target columns at all (e.g. a
    memmap corpus built before afterstate_target was added to LOADER_KEYS) must
    still work off target_scores alone."""
    data = _base_data()
    del data["afterstate_target"]
    del data["afterstate_target_mask"]
    batch = np.arange(6)

    values, mask, is_target_scores = _played_action_bootstrap_value(data, batch)

    assert mask.tolist() == [True, True, False, True, True, False]
    assert is_target_scores[mask].tolist() == [True, True, True, True]
    assert values[mask].tolist() == pytest.approx([0.6, 0.2, -0.9, 0.7])


def test_played_action_bootstrap_value_uses_afterstate_only_when_target_scores_absent():
    """A corpus with afterstate_target but no target_scores columns at all must
    fall back to afterstate_target alone (source (1) simply never wins)."""
    data = _base_data()
    del data["target_scores"]
    del data["target_scores_mask"]
    batch = np.arange(6)

    values, mask, is_target_scores = _played_action_bootstrap_value(data, batch)

    assert mask.tolist() == [True, False, True, True, False, False]
    assert not bool(np.any(is_target_scores))
    assert values[mask].tolist() == pytest.approx([0.3, 0.5, 0.1])


def test_played_action_bootstrap_value_is_a_no_op_without_either_column():
    """A corpus lacking BOTH target_scores and afterstate_target entirely must not
    crash -- just report zero coverage (an all-False mask)."""
    data = _base_data()
    del data["target_scores"]
    del data["target_scores_mask"]
    del data["afterstate_target"]
    del data["afterstate_target_mask"]
    batch = np.arange(6)

    values, mask, is_target_scores = _played_action_bootstrap_value(data, batch)

    assert mask.tolist() == [False] * 6
    assert values.tolist() == [0.0] * 6
    assert is_target_scores.tolist() == [False] * 6


def test_value_target_lambda_zero_is_bit_identical_to_current_behavior():
    """Default (0.0) must be an exact no-op, regardless of which bootstrap columns
    are present in the corpus."""
    data = _base_data()
    batch = np.arange(6)

    baseline = _value_targets(data, batch, torch.device("cpu"), vps_to_win=10)
    with_lambda_zero = _value_targets(
        data, batch, torch.device("cpu"), vps_to_win=10, value_target_lambda=0.0
    )

    for base_field, lambda_field in zip(baseline, with_lambda_zero):
        if base_field is None:
            assert lambda_field is None
            continue
        assert torch.equal(base_field, lambda_field)


def test_value_target_lambda_mixes_clean_outcome_rows_using_target_scores():
    """A row with a clean win/loss z AND both bootstrap sources mixes
    (1-lambda)*z + lambda*target_scores[played] (source (1) wins) -- both already
    share the acting-player perspective, so no sign flip is applied."""
    data = _base_data()
    batch = np.arange(6)

    _, _, _, _, value_outcome, value_has_outcome, confidence = _value_targets(
        data, batch, torch.device("cpu"), vps_to_win=10, value_target_lambda=0.5
    )

    # Row 0: z=+1.0 (RED won, player=RED), target_scores=0.6 (not the afterstate
    # 0.3) -> 0.5*1.0 + 0.5*0.6 = 0.8.
    assert float(value_outcome[0]) == pytest.approx(0.8)
    # Row 3: z=-1.0 (BLUE won, player=RED), target_scores=-0.9 (not the afterstate
    # 0.1) -> 0.5*-1.0 + 0.5*-0.9 = -0.95.
    assert float(value_outcome[3]) == pytest.approx(-0.95)
    assert bool(value_has_outcome[0]) is True
    assert bool(value_has_outcome[3]) is True
    assert float(confidence[0]) == pytest.approx(1.0)
    assert float(confidence[3]) == pytest.approx(1.0)


def test_value_target_lambda_mixes_truncated_row_using_afterstate_fallback():
    """Row 2 (truncated) has no target_scores at its played action but DOES have a
    usable afterstate_target -- the mix must use the fallback value, applied
    AFTER the F3 truncated-proxy fill."""
    data = _base_data()
    batch = np.arange(6)

    _, _, _, _, value_outcome, value_has_outcome, confidence = _value_targets(
        data,
        batch,
        torch.device("cpu"),
        vps_to_win=10,
        truncated_vp_margin_value_weight=0.25,
        value_target_lambda=0.5,
    )

    # Row 2: BLUE (seat=0) leads 7-3 -> proxy z = 0.4, afterstate fallback = 0.5 ->
    # 0.5*0.4 + 0.5*0.5 = 0.45.
    assert bool(value_has_outcome[2]) is True
    assert float(value_outcome[2]) == pytest.approx(0.45)
    # Confidence (the F3 soft-fill weight) is unaffected by the lambda mix.
    assert float(confidence[2]) == pytest.approx(0.25)


def test_value_target_lambda_masked_row_falls_back_to_pure_z():
    """Row 5's played action has NEITHER bootstrap source usable -- even though the
    row has a (truncated-proxy) z, the mix must leave it exactly at pure z."""
    data = _base_data()
    batch = np.arange(6)

    _, _, _, _, value_outcome, _, _ = _value_targets(
        data,
        batch,
        torch.device("cpu"),
        vps_to_win=10,
        truncated_vp_margin_value_weight=0.25,
        value_target_lambda=0.5,
    )

    # Row 5: RED (seat=1) leads 5-3 -> proxy margin = 0.2, untouched (no bootstrap value).
    assert float(value_outcome[5]) == pytest.approx(0.2)


def test_value_target_lambda_never_invents_new_value_loss_coverage():
    """Row 4 has a real played-action target_scores value but NO z/proxy target at
    all (has_outcome=False, not truncated) -- lambda-mixing must be a strict
    refinement of EXISTING targets, never a new source of value-loss coverage."""
    data = _base_data()
    batch = np.arange(6)

    _, _, _, _, value_outcome, value_has_outcome, confidence = _value_targets(
        data, batch, torch.device("cpu"), vps_to_win=10, value_target_lambda=0.5
    )

    assert bool(value_has_outcome[4]) is False
    assert float(value_outcome[4]) == pytest.approx(0.0)
    assert float(confidence[4]) == pytest.approx(0.0)


def test_value_target_lambda_one_is_pure_search_value():
    """lambda=1.0 is Willemsen's original soft-Z: the mixed target collapses to
    exactly the played action's bootstrap value (target_scores, since it wins
    precedence) for rows that have one."""
    data = _base_data()
    batch = np.arange(6)

    _, _, _, _, value_outcome, _, _ = _value_targets(
        data, batch, torch.device("cpu"), vps_to_win=10, value_target_lambda=1.0
    )

    assert float(value_outcome[0]) == pytest.approx(0.6)
    assert float(value_outcome[3]) == pytest.approx(-0.9)


def test_value_target_lambda_is_a_no_op_without_either_bootstrap_column():
    """A corpus lacking BOTH target_scores and afterstate_target entirely must
    train exactly as if --value-target-lambda were 0.0 -- never crash, never
    silently corrupt z."""
    data = _base_data()
    del data["target_scores"]
    del data["target_scores_mask"]
    del data["afterstate_target"]
    del data["afterstate_target_mask"]
    batch = np.arange(6)

    baseline = _value_targets(data, batch, torch.device("cpu"), vps_to_win=10)
    with_lambda = _value_targets(
        data, batch, torch.device("cpu"), vps_to_win=10, value_target_lambda=0.9
    )

    for base_field, lambda_field in zip(baseline, with_lambda):
        if base_field is None:
            assert lambda_field is None
            continue
        assert torch.equal(base_field, lambda_field)
