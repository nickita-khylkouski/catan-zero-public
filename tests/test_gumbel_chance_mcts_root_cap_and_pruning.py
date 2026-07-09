from __future__ import annotations

import json

import pytest

from catan_zero.search.gumbel_chance_mcts import (
    GumbelChanceMCTS,
    GumbelChanceMCTSConfig,
    _prune_policy_target,
    _root_candidate_count,
)
from catan_zero.search.rust_mcts import _require_rust_module


def _rust():
    try:
        return _require_rust_module()
    except RuntimeError as error:
        pytest.skip(str(error))


def _advance_to_multi_action_state(catanatron_rs, *, seed: int, min_legal: int = 2):
    game = catanatron_rs.Game.simple(["RED", "BLUE"], seed=seed)
    for _ in range(300):
        game.play_tick()
        if game.winning_color() is not None:
            break
        playable = json.loads(game.playable_actions_json())
        if len(playable) >= min_legal:
            return game
    raise AssertionError(f"did not reach a state with >= {min_legal} legal actions")


# ---------------------------------------------------------------------------
# _root_candidate_count: cap-binding math (pure function, no rust dependency).
# ---------------------------------------------------------------------------


def test_root_candidate_count_off_matches_narrow_default():
    config = GumbelChanceMCTSConfig()  # root_candidate_cap=None, wide_candidates_threshold=24
    assert config.root_candidate_cap is None
    # num_legal (20) <= wide_candidates_threshold (24) -> narrow branch, capped at
    # max_root_candidates (16).
    assert _root_candidate_count(20, config) == 16


def test_root_candidate_count_off_matches_wide_default():
    config = GumbelChanceMCTSConfig()
    # num_legal (60) > wide_candidates_threshold (24) -> wide branch, capped at
    # max_root_candidates_wide (54).
    assert _root_candidate_count(60, config) == 54


def test_root_candidate_count_off_is_noop_when_legal_below_cap():
    config = GumbelChanceMCTSConfig()
    # Wide branch, but only 30 legal actions -- below max_root_candidates_wide (54),
    # so the cap does not bind ("K only binds when legal > K").
    assert _root_candidate_count(30, config) == 30


def test_root_candidate_count_override_supersedes_narrow_and_wide_defaults():
    config = GumbelChanceMCTSConfig(root_candidate_cap=20)
    # Narrow root: the override (20) replaces max_root_candidates (16).
    assert _root_candidate_count(30, config) == 20
    # Wide root: the SAME override (20) replaces max_root_candidates_wide (54) too.
    assert _root_candidate_count(60, config) == 20


def test_root_candidate_count_override_only_binds_when_legal_exceeds_cap():
    config = GumbelChanceMCTSConfig(root_candidate_cap=20)
    assert _root_candidate_count(10, config) == 10


def test_root_candidate_count_never_below_one():
    config = GumbelChanceMCTSConfig(root_candidate_cap=0)
    assert _root_candidate_count(5, config) == 1
    assert _root_candidate_count(0, config) == 1


# ---------------------------------------------------------------------------
# _prune_policy_target: prune+renormalize correctness (pure function).
# ---------------------------------------------------------------------------


def test_prune_policy_target_default_is_exact_noop():
    policy = {1: 0.5, 2: 0.3, 3: 0.2}
    assert _prune_policy_target(policy, {1: 0, 2: 5, 3: 10}, min_visits=0) is policy


def test_prune_policy_target_zeros_low_evidence_and_renormalizes():
    policy = {1: 0.5, 2: 0.3, 3: 0.2}
    visits = {1: 0, 2: 2, 3: 5}

    pruned = _prune_policy_target(policy, visits, min_visits=1)

    # Key set (support of the DICT, not the probability mass) is unchanged --
    # SearchResult.improved_policy's key set must still cover every legal action.
    assert set(pruned) == set(policy)
    # Action 1 (0 visits, i.e. completed-Q rested entirely on v_mix) is zeroed.
    assert pruned[1] == pytest.approx(0.0)
    # Remaining mass renormalizes to a valid probability distribution.
    assert pruned[2] + pruned[3] == pytest.approx(1.0)
    assert pruned[2] == pytest.approx(0.3 / 0.5)
    assert pruned[3] == pytest.approx(0.2 / 0.5)
    assert sum(pruned.values()) == pytest.approx(1.0)


def test_prune_policy_target_support_is_strictly_smaller_or_equal():
    policy = {1: 0.5, 2: 0.3, 3: 0.2}
    visits = {1: 0, 2: 2, 3: 5}

    pruned = _prune_policy_target(policy, visits, min_visits=1)

    unpruned_support = {a for a, p in policy.items() if p > 0.0}
    pruned_support = {a for a, p in pruned.items() if p > 0.0}
    assert pruned_support <= unpruned_support
    assert len(pruned_support) < len(unpruned_support)


def test_prune_policy_target_higher_threshold_prunes_more():
    policy = {1: 0.5, 2: 0.3, 3: 0.2}
    visits = {1: 1, 2: 2, 3: 5}

    pruned_low = _prune_policy_target(policy, visits, min_visits=2)
    pruned_high = _prune_policy_target(policy, visits, min_visits=5)

    assert pruned_low[1] == pytest.approx(0.0)
    assert pruned_low[2] > 0.0
    assert pruned_high[1] == pytest.approx(0.0)
    assert pruned_high[2] == pytest.approx(0.0)
    assert pruned_high[3] == pytest.approx(1.0)


def test_prune_policy_target_falls_back_when_everything_would_be_pruned():
    """Defensive: if the threshold is so high every candidate is pruned (e.g. a tiny
    sim budget), fall back to the unpruned target rather than emit an all-zero
    distribution."""
    policy = {1: 0.6, 2: 0.4}
    visits = {1: 1, 2: 1}

    pruned = _prune_policy_target(policy, visits, min_visits=100)

    assert pruned == policy


def test_prune_policy_target_missing_visits_entry_treated_as_zero():
    policy = {1: 0.5, 2: 0.5}
    pruned = _prune_policy_target(policy, {1: 3}, min_visits=1)
    assert pruned[1] == pytest.approx(1.0)
    assert pruned[2] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Off-path identity / edge cases against a real search (requires catanatron_rs).
# ---------------------------------------------------------------------------


def test_root_candidate_cap_off_and_pruning_off_is_byte_identical_to_baseline():
    """Flag-off regression: leaving both CAT-62 knobs at their defaults must
    reproduce the exact pre-CAT-62 search output for a seeded search."""
    catanatron_rs = _rust()
    game = _advance_to_multi_action_state(catanatron_rs, seed=11)

    baseline_config = GumbelChanceMCTSConfig(seed=3, n_full=32, p_full=1.0)
    cat62_off_config = GumbelChanceMCTSConfig(
        seed=3, n_full=32, p_full=1.0, root_candidate_cap=None, policy_target_min_visits=0
    )

    baseline = GumbelChanceMCTS(baseline_config).search(game.copy(), force_full=True)
    off = GumbelChanceMCTS(cat62_off_config).search(game.copy(), force_full=True)

    assert off.selected_action == baseline.selected_action
    assert off.improved_policy == baseline.improved_policy
    assert off.visit_counts == baseline.visit_counts
    assert off.q_values == baseline.q_values


def test_root_candidate_cap_reduces_considered_set_without_crashing_on_edge_widths():
    """(a) the cap actually reduces the considered set to the expected size, and does
    not crash on roots with very few legal actions (the forced single-action root
    bypasses this entirely, but a small multi-action root should not either)."""
    catanatron_rs = _rust()
    narrow_game = _advance_to_multi_action_state(catanatron_rs, seed=11, min_legal=2)
    wide_game = _advance_to_multi_action_state(catanatron_rs, seed=1, min_legal=25)

    config = GumbelChanceMCTSConfig(seed=0, n_full=32, p_full=1.0, root_candidate_cap=4)
    mcts = GumbelChanceMCTS(config)

    narrow_result = mcts.search(narrow_game.copy(), force_full=True)
    wide_result = mcts.search(wide_game.copy(), force_full=True)

    # The improved_policy / visit_counts key sets must still cover ALL legal
    # actions regardless of the considered-set cap (the cap restricts which
    # actions get simulated, not which actions exist at the root).
    assert set(narrow_result.improved_policy) == set(narrow_result.priors)
    assert set(wide_result.improved_policy) == set(wide_result.priors)
    # At most `root_candidate_cap` actions receive any visits.
    assert sum(1 for v in narrow_result.visit_counts.values() if v > 0) <= 4
    assert sum(1 for v in wide_result.visit_counts.values() if v > 0) <= 4


def test_pruned_policy_target_is_a_valid_probability_distribution_from_real_search():
    """(c) the pruned pi' target sums to 1 over a strictly smaller-or-equal support
    than the unpruned version, using a real search result (not just the pure
    function in isolation)."""
    catanatron_rs = _rust()
    game = _advance_to_multi_action_state(catanatron_rs, seed=1, min_legal=25)

    unpruned = GumbelChanceMCTS(
        GumbelChanceMCTSConfig(seed=0, n_full=64, p_full=1.0, policy_target_min_visits=0)
    ).search(game.copy(), force_full=True)
    pruned = GumbelChanceMCTS(
        GumbelChanceMCTSConfig(seed=0, n_full=64, p_full=1.0, policy_target_min_visits=1)
    ).search(game.copy(), force_full=True)

    assert set(pruned.improved_policy) == set(unpruned.improved_policy)
    assert sum(pruned.improved_policy.values()) == pytest.approx(1.0)
    pruned_support = {a for a, p in pruned.improved_policy.items() if p > 0.0}
    unpruned_support = {a for a, p in unpruned.improved_policy.items() if p > 0.0}
    assert pruned_support <= unpruned_support
