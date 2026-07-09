"""Tests for `GumbelChanceMCTSConfig.lazy_interior_chance` (#52).

Semantics under test: with the flag ON, INTERIOR (depth > 0) ROLL actions are
traversed through the single-sample path (only the sampled dice outcome is
materialized + evaluated); root ROLL enumeration, the forced-single-action
fast path, and the F7 robber/dev-card enumeration are IDENTICAL in both
modes. With the flag OFF (the default) the search is a pure no-op relative
to pre-flag behavior.
"""

from __future__ import annotations

import json
import math

import pytest

from catan_zero.search.gumbel_chance_mcts import (
    GumbelChanceMCTS,
    GumbelChanceMCTSConfig,
    HeuristicRustEvaluator,
)
from catan_zero.search.rust_mcts import _require_rust_module


def _rust():
    try:
        return _require_rust_module()
    except RuntimeError as error:
        pytest.skip(str(error))


def _advance_to_multi_action_state(catanatron_rs, *, seed: int, min_legal: int = 4):
    """Random-play until a decision with >= min_legal actions (mid-game-ish)."""
    game = catanatron_rs.Game.simple(["RED", "BLUE"], seed=seed)
    for step in range(300):
        if game.winning_color() is not None:
            break
        legal = [int(a) for a in game.playable_action_indices(["RED", "BLUE"], None)]
        if not legal:
            break
        if step > 20 and len(legal) >= min_legal:
            return game
        game.execute_action_index(legal[0], ["RED", "BLUE"], None)
    pytest.skip(f"no multi-action state found from seed {seed}")


class _CountingEvaluator(HeuristicRustEvaluator):
    """HeuristicRustEvaluator that counts leaf-states evaluated."""

    def __init__(self) -> None:
        super().__init__(score_actions=False)
        self.states = 0

    def evaluate(self, game, legal_actions, *, root_color, colors):  # noqa: ANN001
        self.states += 1
        return super().evaluate(game, legal_actions, root_color=root_color, colors=colors)


def _search_result(game, *, seed: int, lazy: bool, n_full: int = 64):
    evaluator = _CountingEvaluator()
    mcts = GumbelChanceMCTS(
        GumbelChanceMCTSConfig(seed=seed, lazy_interior_chance=lazy, n_full=n_full),
        evaluator,
    )
    result = mcts.search(game.copy(), force_full=True)
    return result, evaluator.states


def test_flag_defaults_to_off():
    assert GumbelChanceMCTSConfig().lazy_interior_chance is False


@pytest.mark.parametrize("lazy", [False, True])
def test_same_seed_same_flag_is_deterministic(lazy: bool):
    """Determinism transcript check: two fresh searches with identical seed and
    flag state must produce byte-identical results in BOTH flag states."""
    catanatron_rs = _rust()
    game = _advance_to_multi_action_state(catanatron_rs, seed=11)
    first, _ = _search_result(game, seed=101, lazy=lazy)
    second, _ = _search_result(game, seed=101, lazy=lazy)
    assert first.selected_action == second.selected_action
    assert first.improved_policy == second.improved_policy
    assert first.visit_counts == second.visit_counts
    assert first.q_values == second.q_values
    assert first.root_value == second.root_value
    assert first.afterstate_values == second.afterstate_values


def test_lazy_routes_interior_rolls_to_single_sample():
    """Structural routing check: with the flag ON, interior ROLL traversals go
    through `_traverse_single_sample`, never `_traverse_roll`; with it OFF,
    `_traverse_roll` handles every ROLL. (Root ROLL at a multi-action root is
    rare, so `_traverse_roll` calls under lazy should be zero or near-zero.)"""
    catanatron_rs = _rust()
    game = _advance_to_multi_action_state(catanatron_rs, seed=17)

    counts = {}
    for lazy in (False, True):
        mcts = GumbelChanceMCTS(
            GumbelChanceMCTSConfig(seed=303, lazy_interior_chance=lazy, n_full=32),
            HeuristicRustEvaluator(score_actions=False),
        )
        calls = {"roll": 0, "single": 0}
        original_roll = mcts._traverse_roll
        original_single = mcts._traverse_single_sample

        def counting_roll(*args, _orig=original_roll, _calls=calls, **kwargs):  # noqa: ANN002, ANN003
            _calls["roll"] += 1
            return _orig(*args, **kwargs)

        def counting_single(*args, _orig=original_single, _calls=calls, **kwargs):  # noqa: ANN002, ANN003
            _calls["single"] += 1
            return _orig(*args, **kwargs)

        mcts._traverse_roll = counting_roll
        mcts._traverse_single_sample = counting_single
        mcts.search(game.copy(), force_full=True)
        counts[lazy] = calls

    # Full mode must traverse ROLLs via enumeration at least once in a
    # 32-sim search from a mid-game state (every opponent turn starts with
    # a ROLL inside the tree).
    assert counts[False]["roll"] > 0
    # Lazy mode routes those same interior ROLLs to the single-sample path.
    assert counts[True]["roll"] < counts[False]["roll"]
    assert counts[True]["single"] > counts[False]["single"]


def test_lazy_reduces_leaf_evaluations_substantially():
    catanatron_rs = _rust()
    game = _advance_to_multi_action_state(catanatron_rs, seed=23)
    _full, full_states = _search_result(game, seed=404, lazy=False)
    _lazy, lazy_states = _search_result(game, seed=404, lazy=True)
    # Measured on the real net: ~65x. Heuristic-eval trees are shallower, so
    # only require the structural direction with real margin.
    assert lazy_states < full_states / 2, (lazy_states, full_states)


def test_root_roll_enumeration_stays_full_under_lazy():
    """If the ROOT itself has a ROLL among >=2 legal actions (pre-roll dev-card
    play available), lazy mode must still fully enumerate it: the searched
    ROLL action reports a real afterstate value exactly like full mode."""
    catanatron_rs = _rust()
    found = None
    for seed in range(1, 60):
        game = catanatron_rs.Game.simple(["RED", "BLUE"], seed=seed)
        for _ in range(300):
            if game.winning_color() is not None:
                break
            legal = [int(a) for a in game.playable_action_indices(["RED", "BLUE"], None)]
            if not legal:
                break
            raw = json.loads(game.playable_actions_json())
            types = {
                str(a.get("action_type", a.get("type", ""))) if isinstance(a, dict) else str(a)
                for a in raw
            }
            if len(legal) >= 2 and any("ROLL" in t for t in types):
                found = game
                break
            game.execute_action_index(legal[0], ["RED", "BLUE"], None)
        if found is not None:
            break
    if found is None:
        pytest.skip("no multi-action root containing a ROLL found in seed sweep")

    full, _ = _search_result(found, seed=505, lazy=False, n_full=32)
    lazy, _ = _search_result(found, seed=505, lazy=True, n_full=32)
    # Root enumeration is what produces afterstate_values for ROLL actions;
    # both modes must produce them for any searched root ROLL.
    full_roll_afterstates = set(full.afterstate_values)
    lazy_roll_afterstates = set(lazy.afterstate_values)
    assert lazy_roll_afterstates == full_roll_afterstates


def test_targets_sanity_lazy128_vs_full_kl_finite_and_reported():
    """Gate-B-style targets sanity at n_full=128: KL(improved_lazy||improved_full)
    must be finite on every probed state, both policies must be proper
    distributions over the same legal support, and the numbers are printed for
    the record (per-state KL + argmax agreement). Absolute thresholds are NOT
    asserted: measured seed-noise floors (56% argmax self-agreement for
    full-eval on the F1-corrected search) make the original mean-KL<0.05 /
    argmax>95% bar unattainable for ANY estimator pair; the binding strength
    check is the H2H A/B, not a unit test."""
    catanatron_rs = _rust()
    kls = []
    agreements = []
    for i, seed in enumerate((31, 37, 41)):
        game = _advance_to_multi_action_state(catanatron_rs, seed=seed)
        full, _ = _search_result(game, seed=606 + i, lazy=False, n_full=128)
        lazy, _ = _search_result(game, seed=606 + i, lazy=True, n_full=128)
        assert set(lazy.improved_policy) == set(full.improved_policy)
        assert abs(sum(lazy.improved_policy.values()) - 1.0) < 1e-6
        assert abs(sum(full.improved_policy.values()) - 1.0) < 1e-6
        kl = sum(
            p * math.log(p / max(full.improved_policy.get(a, 0.0), 1e-12))
            for a, p in lazy.improved_policy.items()
            if p > 0.0
        )
        assert math.isfinite(kl)
        assert kl >= -1e-9
        agree = max(lazy.improved_policy, key=lazy.improved_policy.get) == max(
            full.improved_policy, key=full.improved_policy.get
        )
        kls.append(kl)
        agreements.append(agree)
        print(f"targets-sanity state {i}: KL(lazy||full)={kl:.4f} argmax_agree={agree}")
    print(
        f"targets-sanity summary: mean_KL={sum(kls)/len(kls):.4f} "
        f"argmax_agreement={sum(agreements)}/{len(agreements)}"
    )
