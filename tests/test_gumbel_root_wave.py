"""Focused tests for flag-gated batched Gumbel root waves."""

from __future__ import annotations

import json
import math
import random
from dataclasses import replace
from typing import Any

import pytest
import catan_zero.search.gumbel_chance_mcts as gcm

from catan_zero.search.gumbel_chance_mcts import (
    GumbelChanceMCTS,
    GumbelChanceMCTSConfig,
    _GAction,
    _GNode,
    sequential_halving_schedule,
)


class _Game:
    def __init__(self, branch: int = -1, depth: int = 0, outcome: int = 0) -> None:
        self.branch = branch
        self.depth = depth
        self.outcome = outcome

    def winning_color(self) -> None:
        return None

    def current_color(self) -> str:
        return "RED"

    def apply_chance_outcome(self, raw_action: str, outcome: int) -> "_Game":
        action = json.loads(raw_action)
        branch = self.branch
        if self.depth == 0:
            branch = int(action[2])
        return _Game(branch=branch, depth=self.depth + 1, outcome=outcome)

    def apply_chance_outcomes_batch(
        self, raw_action: str, outcomes: list[int]
    ) -> list["_Game"]:
        return [self.apply_chance_outcome(raw_action, outcome) for outcome in outcomes]


class _Evaluator:
    def __init__(self) -> None:
        self.single_calls = 0
        self.batch_sizes: list[int] = []

    @staticmethod
    def _result(
        game: _Game, legal: tuple[int, ...]
    ) -> tuple[dict[int, float], float, float]:
        priors = {action: (2.0 if action == legal[0] else 1.0) for action in legal}
        branch_value = 0.15 * float(game.branch)
        outcome_value = {2: -0.4, 8: 0.6}.get(game.outcome, 0.0)
        return priors, branch_value + outcome_value, 0.4

    def evaluate(
        self,
        game: _Game,
        legal: tuple[int, ...],
        *,
        root_color: str,
        colors: tuple[str, ...],
    ) -> tuple[dict[int, float], float, float]:
        del root_color, colors
        self.single_calls += 1
        return self._result(game, legal)

    def evaluate_many(
        self,
        requests: list[tuple[_Game, tuple[int, ...]]],
        *,
        root_color: str,
        colors: tuple[str, ...],
    ) -> list[tuple[dict[int, float], float, float]]:
        del root_color, colors
        self.batch_sizes.append(len(requests))
        return [self._result(game, legal) for game, legal in requests]


class _MalformedEvaluator(_Evaluator):
    def __init__(self, mode: str) -> None:
        super().__init__()
        self.mode = mode

    def evaluate_many(
        self,
        requests: list[tuple[_Game, tuple[int, ...]]],
        *,
        root_color: str,
        colors: tuple[str, ...],
    ) -> list[tuple[dict[int, float], float, float]]:
        if self.mode == "raise":
            raise ValueError("evaluator exploded")
        results = super().evaluate_many(
            requests, root_color=root_color, colors=colors
        )
        if self.mode == "short":
            return results[:-1]
        if self.mode == "long":
            return results + [results[-1]]
        raise AssertionError(f"unknown malformed evaluator mode: {self.mode}")


def _context(
    game: _Game, *, chance: bool
) -> tuple[
    tuple[int, ...],
    dict[int, Any],
    dict[int, tuple[tuple[int, float], ...]],
]:
    if chance and game.depth == 1:
        return (7,), {7: [None, "ROLL", 7]}, {7: ((2, 0.25), (8, 0.75))}
    legal = (10, 11)
    actions = {action: [None, "PLAY", action] for action in legal}
    spectra = {action: ((0, 1.0),) for action in legal}
    return legal, actions, spectra


def _build_search(
    config: GumbelChanceMCTSConfig,
    *,
    chance: bool,
    num_root_actions: int = 2,
    evaluator: _Evaluator | None = None,
) -> tuple[GumbelChanceMCTS, _GNode, _Evaluator]:
    evaluator = evaluator or _Evaluator()
    mcts = object.__new__(GumbelChanceMCTS)
    mcts.config = config
    mcts.evaluator = evaluator
    mcts.rng = random.Random(config.seed)
    mcts._fetch_legal_actions = lambda game: _context(game, chance=chance)  # type: ignore[method-assign]

    root = _GNode(game=_Game(), root_color="RED", expanded=True)
    prior = 1.0 / float(num_root_actions)
    root.actions = {
        action: _GAction(prior=prior) for action in range(num_root_actions)
    }
    root.action_logits = {
        action: math.log(prior) for action in range(num_root_actions)
    }
    root.action_json = {
        action: [None, "PLAY", action] for action in range(num_root_actions)
    }
    root.action_spectrum = {
        action: ((0, 1.0),) for action in range(num_root_actions)
    }
    return mcts, root, evaluator


def _search(
    config: GumbelChanceMCTSConfig,
    *,
    chance: bool,
    num_root_actions: int = 2,
    evaluator: _Evaluator | None = None,
) -> tuple[GumbelChanceMCTS, _GNode, _Evaluator, int, int]:
    mcts, root, evaluator = _build_search(
        config,
        chance=chance,
        num_root_actions=num_root_actions,
        evaluator=evaluator,
    )
    selected, used = mcts._run_root_search(root, config.n_full)
    return mcts, root, evaluator, selected, used


def test_root_wave_is_default_off() -> None:
    assert GumbelChanceMCTSConfig().root_wave_batching is False


def test_root_wave_preserves_budget_targets_and_selection_on_deterministic_tree() -> None:
    base = GumbelChanceMCTSConfig(
        seed=17,
        n_full=16,
        exact_budget_sh=True,
        max_root_candidates=2,
        max_depth=5,
    )
    legacy = _search(base, chance=False)
    wave = _search(replace(base, root_wave_batching=True), chance=False)

    _legacy_mcts, legacy_root, _legacy_eval, legacy_selected, legacy_used = legacy
    wave_mcts, wave_root, wave_eval, wave_selected, wave_used = wave
    assert legacy_used == wave_used == base.n_full
    assert legacy_selected == wave_selected
    assert {
        action: stats.visits for action, stats in legacy_root.actions.items()
    } == {action: stats.visits for action, stats in wave_root.actions.items()}
    legacy_q = wave_mcts._completed_q(legacy_root)
    wave_q = wave_mcts._completed_q(wave_root)
    assert legacy_q == wave_q
    assert wave_mcts._improved_policy(legacy_root, legacy_q) == (
        wave_mcts._improved_policy(wave_root, wave_q)
    )
    assert any(size == 2 for size in wave_eval.batch_sizes)


def test_root_wave_coalesces_ready_leaves_into_fewer_evaluator_calls() -> None:
    config = GumbelChanceMCTSConfig(
        seed=3,
        n_full=8,
        exact_budget_sh=True,
        max_root_candidates=2,
        max_depth=6,
    )
    *_prefix, legacy_eval, _selected, _used = _search(config, chance=False)
    *_prefix, wave_eval, _selected, _used = _search(
        replace(config, root_wave_batching=True), chance=False
    )
    legacy_calls = legacy_eval.single_calls + len(legacy_eval.batch_sizes)
    wave_calls = wave_eval.single_calls + len(wave_eval.batch_sizes)
    assert wave_calls < legacy_calls
    assert max(wave_eval.batch_sizes) == 2


def test_root_wave_exact_multiround_partial_prefix_spends_every_visit() -> None:
    config = GumbelChanceMCTSConfig(
        seed=5,
        n_full=13,
        exact_budget_sh=True,
        max_root_candidates=5,
        max_depth=5,
        root_wave_batching=True,
    )
    _mcts, root, _evaluator, _selected, used = _search(
        config, chance=False, num_root_actions=5
    )
    per_action = [stats.visits for stats in root.actions.values()]
    assert used == config.n_full == 13
    assert root.visits == used
    assert sum(per_action) == used
    # exact phases are (5,1),(2,2),(2,2): all candidates receive the first
    # pass and the final two receive both multi-visit rounds.
    assert sorted(per_action) == [1, 1, 1, 5, 5]


def test_root_wave_preserves_legacy_floor_overrun() -> None:
    config = GumbelChanceMCTSConfig(
        seed=5,
        n_full=4,
        exact_budget_sh=False,
        max_root_candidates=8,
        max_depth=5,
        root_wave_batching=True,
    )
    _mcts, root, _evaluator, _selected, used = _search(
        config, chance=False, num_root_actions=8
    )
    expected = sum(
        count * budget
        for count, budget in sequential_halving_schedule(8, config.n_full)
    )
    per_action = [stats.visits for stats in root.actions.values()]
    assert used == expected == 14
    assert used > config.n_full
    assert root.visits == used
    assert sum(per_action) == used
    assert sorted(per_action) == [1, 1, 1, 1, 2, 2, 3, 3]


def test_root_wave_keeps_exact_chance_probabilities_and_budget() -> None:
    config = GumbelChanceMCTSConfig(
        seed=29,
        n_full=8,
        exact_budget_sh=True,
        max_root_candidates=2,
        max_depth=4,
        root_wave_batching=True,
    )
    mcts, root, evaluator, selected, used = _search(config, chance=True)
    assert used == config.n_full
    assert selected in root.actions
    assert sum(stats.visits for stats in root.actions.values()) == config.n_full
    assert math.isclose(
        sum(mcts._improved_policy(root, mcts._completed_q(root)).values()),
        1.0,
        rel_tol=1.0e-12,
    )
    roll_stats = [
        next(iter(root_stats.children.values())).actions[7]
        for root_stats in root.actions.values()
        if root_stats.children
    ]
    assert roll_stats
    for stats in roll_stats:
        assert stats.probabilities == {2: 0.25, 8: 0.75}
        assert stats.afterstate_value is not None
    # Chance-child enumeration itself must remain batched.
    assert any(size == 2 for size in evaluator.batch_sizes)


def test_root_wave_matches_legacy_with_distinct_chance_outcome_values() -> None:
    base = GumbelChanceMCTSConfig(
        seed=31,
        n_full=8,
        exact_budget_sh=True,
        max_root_candidates=4,
        max_depth=2,
    )
    legacy = _search(base, chance=True, num_root_actions=4)
    wave = _search(
        replace(base, root_wave_batching=True),
        chance=True,
        num_root_actions=4,
    )
    legacy_mcts, legacy_root, _legacy_eval, legacy_selected, legacy_used = legacy
    wave_mcts, wave_root, _wave_eval, wave_selected, wave_used = wave
    assert legacy_used == wave_used == base.n_full
    assert legacy_selected == wave_selected
    assert {
        action: (stats.visits, stats.value_sum, stats.value_sq_sum)
        for action, stats in legacy_root.actions.items()
    } == {
        action: (stats.visits, stats.value_sum, stats.value_sq_sum)
        for action, stats in wave_root.actions.items()
    }
    assert legacy_mcts._completed_q(legacy_root) == wave_mcts._completed_q(
        wave_root
    )
    roll_stats = [
        next(iter(root_stats.children.values())).actions[7]
        for root_stats in wave_root.actions.values()
        if root_stats.children
    ]
    assert roll_stats
    assert all(
        stats.children[2].prior_value != stats.children[8].prior_value
        for stats in roll_stats
    )


def test_root_wave_matches_legacy_at_max_depth_with_weighted_backups() -> None:
    base = GumbelChanceMCTSConfig(
        seed=13,
        n_full=8,
        exact_budget_sh=True,
        max_root_candidates=4,
        max_depth=1,
        uncertainty_backup_weighting=True,
    )
    legacy = _search(base, chance=False, num_root_actions=4)
    wave = _search(
        replace(base, root_wave_batching=True),
        chance=False,
        num_root_actions=4,
    )
    legacy_root, wave_root = legacy[1], wave[1]
    assert legacy[3:] == wave[3:]
    assert legacy_root.visits == wave_root.visits == base.n_full
    assert {
        action: (
            stats.visits,
            stats.value_sum,
            stats.value_sq_sum,
            stats.weight_sum,
            stats.weighted_value_sum,
        )
        for action, stats in legacy_root.actions.items()
    } == {
        action: (
            stats.visits,
            stats.value_sum,
            stats.value_sq_sum,
            stats.weight_sum,
            stats.weighted_value_sum,
        )
        for action, stats in wave_root.actions.items()
    }
    assert all(
        child.visits == 0
        for stats in wave_root.actions.values()
        for child in stats.children.values()
    )


@pytest.mark.parametrize("mode,actual", [("short", 2), ("long", 4)])
def test_root_wave_rejects_malformed_batch_before_backup(
    mode: str, actual: int
) -> None:
    config = GumbelChanceMCTSConfig(
        seed=7,
        n_full=3,
        exact_budget_sh=True,
        max_root_candidates=3,
        root_wave_batching=True,
    )
    evaluator = _MalformedEvaluator(mode)
    mcts, root, _evaluator = _build_search(
        config,
        chance=False,
        num_root_actions=3,
        evaluator=evaluator,
    )
    with pytest.raises(
        RuntimeError, match=rf"returned {actual} results for 3 requests"
    ):
        mcts._run_root_search(root, config.n_full)
    assert root.visits == 0
    assert all(stats.visits == 0 for stats in root.actions.values())
    assert all(
        not child.expanded
        for stats in root.actions.values()
        for child in stats.children.values()
    )


def test_root_wave_propagates_evaluator_exception_without_backup() -> None:
    config = GumbelChanceMCTSConfig(
        seed=7,
        n_full=3,
        exact_budget_sh=True,
        max_root_candidates=3,
        root_wave_batching=True,
    )
    mcts, root, _evaluator = _build_search(
        config,
        chance=False,
        num_root_actions=3,
        evaluator=_MalformedEvaluator("raise"),
    )
    with pytest.raises(ValueError, match="evaluator exploded"):
        mcts._run_root_search(root, config.n_full)
    assert root.visits == 0
    assert all(stats.visits == 0 for stats in root.actions.values())


@pytest.mark.parametrize("mode,actual", [("short", 1), ("long", 3)])
@pytest.mark.parametrize("kind", ["roll", "robber"])
def test_chance_batches_validate_length_before_child_expansion(
    mode: str, actual: int, kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = GumbelChanceMCTSConfig(root_wave_batching=True, use_batch_api=False)
    mcts, _root, _evaluator = _build_search(
        config, chance=True, evaluator=_MalformedEvaluator(mode)
    )

    def forbidden_finish(*_args: Any, **_kwargs: Any) -> float:
        raise AssertionError("child expanded before batch length validation")

    mcts._finish_expand = forbidden_finish  # type: ignore[method-assign]
    stats = _GAction(prior=1.0)
    node = _GNode(game=_Game(branch=1), root_color="RED", expanded=True)
    node.actions = {7: stats}
    with pytest.raises(
        RuntimeError, match=rf"returned {actual} results for 2 requests"
    ):
        if kind == "roll":
            node.action_json = {7: [None, "ROLL", 7]}
            node.action_spectrum = {7: ((2, 0.25), (8, 0.75))}
            mcts._traverse_roll(node, 7, stats, 0)
        else:
            action_json = [None, "MOVE_ROBBER", [[0, 0], "BLUE"]]
            node.action_json = {7: action_json}
            candidates = [
                (2, 0.25, _Game(branch=1, depth=2, outcome=2)),
                (8, 0.75, _Game(branch=1, depth=2, outcome=8)),
            ]
            monkeypatch.setattr(
                gcm,
                "move_robber_victim_outcome_weights",
                lambda *_args, **_kwargs: candidates,
            )
            mcts._traverse_robber_or_dev(node, 7, stats, 0)
    assert stats.children == {}
    assert stats.visits == 0


def test_root_wave_reproducible_with_per_candidate_rng_streams() -> None:
    config = GumbelChanceMCTSConfig(
        seed=41,
        n_full=16,
        exact_budget_sh=True,
        max_root_candidates=2,
        max_depth=5,
        root_wave_batching=True,
    )
    first = _search(config, chance=True)
    second = _search(config, chance=True)
    assert first[3:] == second[3:]
    assert {
        action: (stats.visits, stats.value_sum)
        for action, stats in first[1].actions.items()
    } == {
        action: (stats.visits, stats.value_sum)
        for action, stats in second[1].actions.items()
    }
