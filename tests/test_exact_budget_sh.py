"""Tests for the exact-budget Sequential Halving schedule (task #61).

Two layers:
- Pure-schedule tests for `exact_budget_sh_phases` (no Rust wheel needed):
  exact budget totals across a grid, mctx-conformant phase shapes at the two
  production-critical configs (n_fast=16/m=16 and n_full=64/m=54), and the
  documented overrun of the legacy `sequential_halving_schedule` those
  configs suffer (the bug this fixes).
- Search-level tests (skipped without `catanatron_rs`): with
  `exact_budget_sh=True` a forced full search reports
  `simulations_used == n` exactly; with the flag OFF (default) behavior is
  bit-identical to before the change (same selected action, same visit
  counts, same simulations_used for the same seed).
"""
from __future__ import annotations

import math

import pytest

from catan_zero.search.gumbel_chance_mcts import (
    GumbelChanceMCTS,
    GumbelChanceMCTSConfig,
    exact_budget_sh_phases,
    sequential_halving_schedule,
)

try:  # wheel-dependent tests are skipped on hosts without the Rust engine
    import catanatron_rs  # type: ignore  # noqa: F401

    _HAS_WHEEL = True
except ImportError:
    _HAS_WHEEL = False


# ---------------------------------------------------------------------------
# Pure schedule tests (no wheel).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 8, 11, 16, 24, 32, 54])
@pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32, 64, 96, 128, 512])
def test_exact_budget_totals_are_exact(m: int, n: int) -> None:
    phases = exact_budget_sh_phases(m, n)
    total = sum(count * budget for count, budget in phases)
    assert total == n, f"m={m} n={n}: schedule spends {total} != {n}"


@pytest.mark.parametrize("m", [2, 8, 16, 54])
@pytest.mark.parametrize("n", [8, 16, 64, 512])
def test_phase_counts_never_exceed_survivors(m: int, n: int) -> None:
    # Each phase's candidate count must be <= the previous phase's (the
    # caller slices remaining[:count] from the survivors of the last phase).
    phases = exact_budget_sh_phases(m, n)
    previous = m
    for count, budget in phases:
        assert 1 <= count <= previous
        assert budget >= 1
        previous = count


def test_production_fast_search_config() -> None:
    # n_fast=16 at a typical m=16 root: one pass, 16 sims, no overrun.
    assert exact_budget_sh_phases(16, 16) == [(16, 1)]
    # Legacy schedule overruns the same config to 32 sims (2.0x).
    legacy = sequential_halving_schedule(16, 16)
    assert sum(count * budget for count, budget in legacy) == 32


def test_production_wide_full_search_config() -> None:
    # n_full=64 at a 54-wide placement root: full first pass over all 54,
    # then a truncated pass over the top-10 survivors. Exactly 64.
    assert exact_budget_sh_phases(54, 64) == [(54, 1), (10, 1)]
    # Legacy schedule overruns the same config to 119 sims (1.86x).
    legacy = sequential_halving_schedule(54, 64)
    assert sum(count * budget for count, budget in legacy) == 119


def test_production_fast_search_at_wide_root() -> None:
    # The root-width cap does NOT depend on full/fast: a FAST (n_fast=16)
    # search at a 54-wide placement root runs m=54 -- legacy spends 105 sims
    # (6.6x the nominal budget, costlier than a nominal full-64 search).
    # Exact-budget truncates to the top-16 by gumbel+logits, 16 sims total.
    assert exact_budget_sh_phases(54, 16) == [(16, 1)]
    legacy = sequential_halving_schedule(54, 16)
    assert sum(count * budget for count, budget in legacy) == 105


def test_legacy_underspend_is_also_fixed() -> None:
    # The legacy floor can also UNDERSPEND (int-floor budgets never refunded):
    # m=54 n=512 spends only 466. Exact-budget spends the full 512.
    legacy = sequential_halving_schedule(54, 512)
    assert sum(count * budget for count, budget in legacy) == 466
    exact = exact_budget_sh_phases(54, 512)
    assert sum(count * budget for count, budget in exact) == 512


def test_narrow_full_search_config_matches_legacy() -> None:
    # n_full=64 at m=16 was already exact under the legacy schedule; the
    # exact-budget phases reproduce the identical (16,1),(8,2),(4,4),(2,8).
    assert exact_budget_sh_phases(16, 64) == [(16, 1), (8, 2), (4, 4), (2, 8)]
    assert sequential_halving_schedule(16, 64) == [(16, 1), (8, 2), (4, 4), (2, 8)]


def test_single_candidate_gets_whole_budget() -> None:
    assert exact_budget_sh_phases(1, 7) == [(1, 7)]


def test_budget_smaller_than_width_truncates_first_pass() -> None:
    # n < m: only the top-n candidates (by gumbel+logits) get one sim each.
    assert exact_budget_sh_phases(54, 16) == [(16, 1)]


def test_large_budget_keeps_spending_at_two_survivors() -> None:
    # mctx halves to a floor of TWO considered candidates and keeps spending
    # passes there until the budget is gone -- the total must still be exact.
    phases = exact_budget_sh_phases(4, 512)
    assert sum(count * budget for count, budget in phases) == 512
    assert phases[-1][0] == 2


# ---------------------------------------------------------------------------
# Search-level tests (need the Rust wheel).
# ---------------------------------------------------------------------------


needs_wheel = pytest.mark.skipif(not _HAS_WHEEL, reason="catanatron_rs wheel not installed")


def _advance_to_multi_action_root(mcts: GumbelChanceMCTS, seed: int):
    """Play forced/searched moves until the game presents a root with more
    than one legal action (the opening placement decision arrives fast)."""
    game = mcts.new_game(seed=seed)
    for _ in range(40):
        legal, _actions, _spectra = mcts._fetch_legal_actions(game)
        if len(legal) > 1:
            return game
        result = mcts.search(game, force_full=False)
        game = game.apply_action(int(result.selected_action))
    raise RuntimeError("no multi-action root reached in 40 plies")


@needs_wheel
def test_exact_budget_flag_spends_exactly_n_simulations() -> None:
    for n_full in (16, 64):
        config = GumbelChanceMCTSConfig(seed=11, n_full=n_full, exact_budget_sh=True)
        mcts = GumbelChanceMCTS(config=config)
        game = _advance_to_multi_action_root(mcts, seed=11)
        result = mcts.search(game, force_full=True)
        assert result.simulations_used == n_full


@needs_wheel
def test_flag_off_is_bit_identical_to_legacy() -> None:
    results = []
    for _ in range(2):
        config = GumbelChanceMCTSConfig(seed=23, n_full=64, exact_budget_sh=False)
        mcts = GumbelChanceMCTS(config=config)
        game = _advance_to_multi_action_root(mcts, seed=23)
        results.append(mcts.search(game, force_full=True))
    assert results[0].selected_action == results[1].selected_action
    assert results[0].visit_counts == results[1].visit_counts
    assert results[0].simulations_used == results[1].simulations_used
    # And the legacy overrun is still what the default path spends at a wide
    # root: sims exceed n_full whenever the root is wide enough to overrun.
    m = len(results[0].priors)
    legacy_total = sum(
        count * budget
        for count, budget in sequential_halving_schedule(
            min(m, 54 if m > 24 else 16), 64
        )
    )
    assert results[0].simulations_used == legacy_total


@needs_wheel
def test_exact_budget_search_returns_valid_result() -> None:
    config = GumbelChanceMCTSConfig(seed=7, n_full=64, n_fast=16, exact_budget_sh=True)
    mcts = GumbelChanceMCTS(config=config)
    game = _advance_to_multi_action_root(mcts, seed=7)
    result = mcts.search(game, force_full=True)
    assert result.selected_action in result.priors
    assert math.isclose(sum(result.improved_policy.values()), 1.0, rel_tol=1e-6)
    assert result.simulations_used == 64
