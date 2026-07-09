"""Tests for the two flag-gated placement/phase search arms (contingency f67):

  * n_full_wide -- placement budget asymmetry (spend more sims at wide roots)
  * raw_policy_above_width -- phase-gated search (skip search at wide roots)

Both default to None (disabled) and must be pure no-ops when off. The initial
settlement placement (a fresh 2p game's very first decision) is the widest root
in the game (>24 legal), which is exactly the root class these arms target.
"""

from __future__ import annotations

import json

import pytest

from catan_zero.search.gumbel_chance_mcts import (
    GumbelChanceMCTS,
    GumbelChanceMCTSConfig,
)
from catan_zero.search.rust_mcts import _require_rust_module


def _rust():
    try:
        return _require_rust_module()
    except RuntimeError as error:
        pytest.skip(str(error))


def _legal_ids(game) -> set[int]:
    return {int(a) for a in game.playable_action_indices(["RED", "BLUE"], None)}


def _wide_root_game(catanatron_rs, *, seed: int):
    """A fresh game's first decision is the wide initial-settlement placement."""
    game = catanatron_rs.Game.simple(["RED", "BLUE"], seed=seed)
    return game


def _narrow_root_game(catanatron_rs, *, seed: int, max_legal: int = 8):
    """Advance to a non-placement state with only a few legal actions."""
    game = catanatron_rs.Game.simple(["RED", "BLUE"], seed=seed)
    for _ in range(400):
        game.play_tick()
        if game.winning_color() is not None:
            break
        playable = json.loads(game.playable_actions_json())
        if 2 <= len(playable) <= max_legal:
            return game
    pytest.skip("did not reach a narrow (2..max_legal) multi-action state")


# ---------------------------------------------------------------------------
# Defaults: both arms off out of the box.
# ---------------------------------------------------------------------------


def test_arms_default_to_disabled():
    config = GumbelChanceMCTSConfig()
    assert config.n_full_wide is None
    assert config.raw_policy_above_width is None


# ---------------------------------------------------------------------------
# Arm A: placement budget asymmetry (n_full_wide).
# ---------------------------------------------------------------------------


def test_n_full_wide_spends_more_sims_at_a_wide_root():
    catanatron_rs = _rust()
    game = _wide_root_game(catanatron_rs, seed=11)
    num_legal = len(_legal_ids(game))
    assert num_legal > GumbelChanceMCTSConfig().wide_candidates_threshold

    base = GumbelChanceMCTS(
        GumbelChanceMCTSConfig(seed=3, n_full=32, p_full=1.0)
    ).search(game, force_full=True)
    wide = GumbelChanceMCTS(
        GumbelChanceMCTSConfig(seed=3, n_full=32, n_full_wide=128, p_full=1.0)
    ).search(game, force_full=True)

    assert wide.simulations_used > base.simulations_used
    assert wide.selected_action in _legal_ids(game)


def test_n_full_wide_is_a_noop_at_a_narrow_root():
    catanatron_rs = _rust()
    game = _narrow_root_game(catanatron_rs, seed=7)
    assert len(_legal_ids(game)) <= GumbelChanceMCTSConfig().wide_candidates_threshold

    base = GumbelChanceMCTS(
        GumbelChanceMCTSConfig(seed=3, n_full=32, p_full=1.0)
    ).search(game, force_full=True)
    with_arm = GumbelChanceMCTS(
        GumbelChanceMCTSConfig(seed=3, n_full=32, n_full_wide=512, p_full=1.0)
    ).search(game, force_full=True)

    # Narrow root is below the wide threshold, so the wide budget never applies.
    assert with_arm.simulations_used == base.simulations_used
    assert with_arm.selected_action == base.selected_action


def test_n_full_wide_none_matches_baseline_at_a_wide_root():
    catanatron_rs = _rust()
    game = _wide_root_game(catanatron_rs, seed=11)

    base = GumbelChanceMCTS(
        GumbelChanceMCTSConfig(seed=3, n_full=32, p_full=1.0)
    ).search(game, force_full=True)
    explicit_none = GumbelChanceMCTS(
        GumbelChanceMCTSConfig(seed=3, n_full=32, n_full_wide=None, p_full=1.0)
    ).search(game, force_full=True)

    assert explicit_none.simulations_used == base.simulations_used
    assert explicit_none.selected_action == base.selected_action
    assert explicit_none.improved_policy == base.improved_policy


# ---------------------------------------------------------------------------
# Arm B: phase-gated search (raw_policy_above_width).
# ---------------------------------------------------------------------------


def test_raw_policy_above_width_skips_search_at_a_wide_root():
    catanatron_rs = _rust()
    game = _wide_root_game(catanatron_rs, seed=11)
    legal = _legal_ids(game)
    assert len(legal) > 24

    result = GumbelChanceMCTS(
        GumbelChanceMCTSConfig(seed=3, n_full=32, p_full=1.0, raw_policy_above_width=24)
    ).search(game, force_full=True)

    # No search ran; the played move is the raw prior argmax (lower id breaks ties).
    assert result.used_full_search is False
    assert result.simulations_used == 0
    assert not result.visit_counts
    assert result.selected_action in legal
    assert result.improved_policy == result.priors
    expected = max(result.priors, key=lambda a: (result.priors[a], -int(a)))
    assert result.selected_action == expected


def test_raw_policy_above_width_still_searches_a_narrower_root():
    catanatron_rs = _rust()
    game = _wide_root_game(catanatron_rs, seed=11)
    num_legal = len(_legal_ids(game))

    # Threshold above this root's width => the gate does not trip, search runs.
    result = GumbelChanceMCTS(
        GumbelChanceMCTSConfig(
            seed=3, n_full=32, p_full=1.0, raw_policy_above_width=num_legal + 1
        )
    ).search(game, force_full=True)

    assert result.used_full_search is True
    assert result.simulations_used > 0
    assert result.visit_counts


def test_raw_policy_above_width_none_searches_normally():
    catanatron_rs = _rust()
    game = _wide_root_game(catanatron_rs, seed=11)

    base = GumbelChanceMCTS(
        GumbelChanceMCTSConfig(seed=3, n_full=32, p_full=1.0)
    ).search(game, force_full=True)
    explicit_none = GumbelChanceMCTS(
        GumbelChanceMCTSConfig(seed=3, n_full=32, p_full=1.0, raw_policy_above_width=None)
    ).search(game, force_full=True)

    assert explicit_none.used_full_search is True
    assert explicit_none.simulations_used == base.simulations_used
    assert explicit_none.selected_action == base.selected_action
