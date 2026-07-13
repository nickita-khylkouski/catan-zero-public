from __future__ import annotations

import math
import random
from dataclasses import replace
from types import MethodType, SimpleNamespace

import pytest

from catan_zero.search.gumbel_chance_mcts import (
    GumbelChanceMCTS,
    GumbelChanceMCTSConfig,
    SearchResult,
    _GAction,
    _GNode,
)


class _AuthoritativeGame:
    def __init__(self, hidden_truth: str) -> None:
        self.hidden_truth = hidden_truth
        self.seeds: list[int] = []

    def current_color(self) -> str:
        return "RED"

    def determinize_for_player(self, observer: str, seed: int) -> "_SampledGame":
        assert observer == "RED"
        self.seeds.append(seed)
        # Deliberately ignore authoritative hidden_truth: this is the engine
        # contract the orchestration layer relies on.
        return _SampledGame(seed)


class _SampledGame:
    def __init__(self, seed: int) -> None:
        self.seed = seed

    def current_color(self) -> str:
        return "RED"

    def num_turns(self) -> int:
        return 7


def _mcts(*, particles: int = 4, n_full: int = 128) -> GumbelChanceMCTS:
    mcts = object.__new__(GumbelChanceMCTS)
    mcts.config = GumbelChanceMCTSConfig(
        seed=91,
        n_full=n_full,
        n_fast=16,
        p_full=1.0,
        information_set_search=True,
        determinization_particles=particles,
    )
    mcts.evaluator = SimpleNamespace(config=SimpleNamespace(public_observation=True))
    mcts.rng = random.Random(mcts.config.seed)

    def fetch(_self, _game):
        return (11, 12), {11: ["RED", "A", None], 12: ["RED", "B", None]}, {}

    def search_one(
        _self,
        game,
        *,
        force_full=None,
        n_simulations_override=None,
    ):
        assert isinstance(game, _SampledGame)
        assert force_full in {True, False}
        budget = int(n_simulations_override)
        # Particle-dependent but authoritative-truth-independent evidence.
        p11 = 0.25 + (game.seed % 100) / 1000.0
        return SearchResult(
            selected_action=11,
            improved_policy={11: p11, 12: 1.0 - p11},
            visit_counts={11: budget // 2, 12: budget - budget // 2},
            q_values={11: 0.1, 12: -0.1},
            priors={11: 0.4, 12: 0.6},
            root_value=0.2,
            used_full_search=True,
            simulations_used=budget,
            afterstate_values={11: 0.3, 12: -0.2},
        )

    mcts._fetch_legal_actions = MethodType(fetch, mcts)
    mcts._search_single_world = MethodType(search_one, mcts)
    return mcts


def test_information_set_result_is_invariant_to_authoritative_hidden_truth() -> None:
    first_game = _AuthoritativeGame("opponent has KNIGHT")
    second_game = _AuthoritativeGame("opponent has VICTORY_POINT")
    first = _mcts().search(first_game, force_full=True)
    second = _mcts().search(second_game, force_full=True)
    assert first == second
    assert first_game.seeds == second_game.seeds


def test_information_set_particles_share_one_exact_total_budget() -> None:
    result = _mcts(particles=4, n_full=128).search(
        _AuthoritativeGame("irrelevant"), force_full=True
    )
    assert result.simulations_used == 128
    assert sum(result.visit_counts.values()) == 128


def test_per_particle_override_enforces_exact_total_budget() -> None:
    """Particle sub-budgets cannot inherit legacy SH rounding overruns."""

    class _StopAfterBudgetAssertion(RuntimeError):
        pass

    class _Game:
        def current_color(self) -> str:
            return "RED"

        def copy(self):
            return self

    mcts = _mcts()

    def fetch(_self, _game):
        return (11, 12), {11: ["RED", "A", None], 12: ["RED", "B", None]}, {}

    def expand(_self, node, *, at_root=False):
        assert at_root is True
        node.actions = {11: _GAction(0.5), 12: _GAction(0.5)}
        node.action_logits = {11: 0.0, 12: 0.0}
        node.expanded = True

    def run_root(_self, _root, n_simulations, *, exact_budget_override=False):
        assert n_simulations == 32
        assert exact_budget_override is True
        raise _StopAfterBudgetAssertion

    mcts._fetch_legal_actions = MethodType(fetch, mcts)
    mcts._expand = MethodType(expand, mcts)
    mcts._run_root_search = MethodType(run_root, mcts)
    with pytest.raises(_StopAfterBudgetAssertion):
        GumbelChanceMCTS._search_single_world(
            mcts, _Game(), force_full=True, n_simulations_override=32
        )


def test_fast_search_uses_one_particle_instead_of_fragmenting_n16() -> None:
    game = _AuthoritativeGame("irrelevant")
    result = _mcts(particles=4, n_full=128).search(game, force_full=False)
    assert len(game.seeds) == 1
    assert result.simulations_used == 16


def test_information_set_search_fails_closed_without_native_determinizer() -> None:
    game = SimpleNamespace(current_color=lambda: "RED")
    with pytest.raises(RuntimeError, match="determinize_for_player"):
        _mcts().search(game, force_full=True)


def _belief_result(
    *,
    q_values: dict[int, float],
    visits: dict[int, int],
    improved: dict[int, float] | None = None,
    completed_q: dict[int, float] | None = None,
) -> SearchResult:
    return SearchResult(
        selected_action=11,
        improved_policy=improved or {11: 0.5, 12: 0.5},
        visit_counts=visits,
        q_values=q_values,
        priors={11: 0.5, 12: 0.5},
        root_value=0.0,
        used_full_search=True,
        simulations_used=sum(visits.values()),
        completed_q_values=completed_q or {11: 0.0, 12: 0.0},
        q_values_root_perspective=True,
    )


def _belief_target_mcts(*, sigma_reference_visits: int = 8) -> GumbelChanceMCTS:
    mcts = object.__new__(GumbelChanceMCTS)
    mcts.config = GumbelChanceMCTSConfig(
        information_set_search=True,
        information_set_target_aggregation="aggregate_q_then_improve",
        sigma_reference_visits=sigma_reference_visits,
        c_visit=0.0,
        c_scale=1.0,
    )
    mcts.rng = random.Random(7)
    return mcts


def test_belief_target_uniformly_weights_worlds_not_visits() -> None:
    mcts = _belief_target_mcts()
    results = [
        _belief_result(
            q_values={11: 1.0, 12: 0.0},
            visits={11: 100, 12: 1},
            completed_q={11: 1.0, 12: 0.0},
        ),
        _belief_result(
            q_values={11: -1.0, 12: 0.0},
            visits={11: 1, 12: 100},
            completed_q={11: -1.0, 12: 0.0},
        ),
    ]
    target = mcts._belief_level_improved_policy(
        results,
        legal_actions=(11, 12),
        aggregate_priors={11: 0.5, 12: 0.5},
    )
    assert target == pytest.approx({11: 0.5, 12: 0.5})


def test_gameplay_aggregation_changes_selection_without_changing_legacy_target() -> None:
    results = [
        _belief_result(
            q_values={11: -1.0, 12: 1.0},
            visits={11: 4, 12: 4},
            improved={11: 0.9, 12: 0.1},
            completed_q={11: -1.0, 12: 1.0},
        ),
        _belief_result(
            q_values={11: -0.5, 12: 0.5},
            visits={11: 4, 12: 4},
            improved={11: 0.8, 12: 0.2},
            completed_q={11: -0.5, 12: 0.5},
        ),
    ]

    def aggregate(mode: str) -> SearchResult:
        mcts = object.__new__(GumbelChanceMCTS)
        mcts.config = GumbelChanceMCTSConfig(
            information_set_search=True,
            information_set_target_aggregation="mean_improved_policy",
            gameplay_policy_aggregation=mode,
            sigma_reference_visits=8,
            c_visit=0.0,
            c_scale=1.0,
        )
        mcts.rng = random.Random(7)
        return mcts._aggregate_information_set_results(
            results, legal_actions=(11, 12), used_full_search=True
        )

    legacy = aggregate("mean_improved_policy")
    corrected = aggregate("aggregate_q_then_improve")
    assert legacy.selected_action == 11
    assert corrected.selected_action == 12
    # Gameplay is a separate opt-in: emitted learner targets stay exactly on
    # the historical mean-of-improved operator in both runs.
    assert corrected.improved_policy == legacy.improved_policy == pytest.approx(
        {11: 0.85, 12: 0.15}
    )


def test_corrected_gameplay_aggregation_fails_closed_without_fixed_sigma() -> None:
    with pytest.raises(ValueError, match="gameplay requires sigma_reference_visits"):
        GumbelChanceMCTS(
            GumbelChanceMCTSConfig(
                information_set_search=True,
                gameplay_policy_aggregation="aggregate_q_then_improve",
            )
        )


def test_belief_target_one_particle_matches_ordinary_completed_q() -> None:
    mcts = _belief_target_mcts(sigma_reference_visits=4)
    result = _belief_result(
        q_values={11: 0.75},
        visits={11: 4, 12: 0},
        completed_q={11: 0.75, 12: 0.64},
    )
    target = mcts._belief_level_improved_policy(
        [result],
        legal_actions=(11, 12),
        aggregate_priors={11: 0.5, 12: 0.5},
    )
    root = _GNode(
        game=SimpleNamespace(current_color=lambda: "RED"),
        root_color="RED",
        prior_value=0.2,
        actions={
            11: _GAction(prior=0.5, visits=4, value_sum=3.0),
            12: _GAction(prior=0.5),
        },
        action_logits={11: math.log(0.5), 12: math.log(0.5)},
    )
    expected = mcts._improved_policy(root, mcts._completed_q(root))
    assert target == pytest.approx(expected)


def test_fixed_sigma_makes_p4_and_duplicated_p8_target_equivalent() -> None:
    mcts = _belief_target_mcts(sigma_reference_visits=8)
    p4 = [
        _belief_result(
            q_values={11: 0.1}, visits={11: 8, 12: 0}, completed_q={11: 0.1, 12: 0.05}
        ),
        _belief_result(
            q_values={12: -0.3}, visits={11: 0, 12: 8}, completed_q={11: -0.1, 12: -0.3}
        ),
        _belief_result(
            q_values={11: -0.2, 12: 0.2},
            visits={11: 4, 12: 4},
            completed_q={11: -0.2, 12: 0.2},
        ),
        _belief_result(
            q_values={11: 0.4, 12: -0.4},
            visits={11: 4, 12: 4},
            completed_q={11: 0.4, 12: -0.4},
        ),
    ]

    def target(results: list[SearchResult]) -> dict[int, float]:
        return mcts._belief_level_improved_policy(
            results,
            legal_actions=(11, 12),
            aggregate_priors={11: 0.5, 12: 0.5},
        )

    assert target(p4) == pytest.approx(target(p4 + p4), abs=1.0e-12)


def test_belief_target_changes_training_target_not_selected_action() -> None:
    mcts = _belief_target_mcts()
    results = [
        _belief_result(
            q_values={11: -0.5, 12: 0.5},
            visits={11: 4, 12: 4},
            improved={11: 0.9, 12: 0.1},
            completed_q={11: -0.5, 12: 0.5},
        )
    ]
    aggregated = mcts._aggregate_information_set_results(
        results, legal_actions=(11, 12), used_full_search=True
    )
    assert aggregated.selected_action == 11
    assert aggregated.improved_policy[12] > aggregated.improved_policy[11]


def test_belief_target_fails_closed_without_q_perspective_attestation() -> None:
    mcts = _belief_target_mcts()
    result = _belief_result(
        q_values={11: 0.2},
        visits={11: 1, 12: 0},
        completed_q={11: 0.2, 12: 0.0},
    )
    result = replace(result, q_values_root_perspective=False)
    with pytest.raises(RuntimeError, match="root-actor Q perspective"):
        mcts._belief_level_improved_policy(
            [result],
            legal_actions=(11, 12),
            aggregate_priors={11: 0.5, 12: 0.5},
        )


def test_belief_target_requires_fixed_sigma_reference() -> None:
    with pytest.raises(ValueError, match="requires sigma_reference_visits"):
        GumbelChanceMCTS(
            GumbelChanceMCTSConfig(
                information_set_search=True,
                information_set_target_aggregation="aggregate_q_then_improve",
            ),
            SimpleNamespace(),
        )


def test_actor_turn_boundary_stops_on_opponent_or_new_turn() -> None:
    mcts = _mcts()
    mcts._information_set_root_turn = 7
    same = _GNode(
        game=SimpleNamespace(current_color=lambda: "RED", num_turns=lambda: 7),
        root_color="RED",
    )
    opponent = _GNode(
        game=SimpleNamespace(current_color=lambda: "BLUE", num_turns=lambda: 7),
        root_color="RED",
    )
    later = _GNode(
        game=SimpleNamespace(current_color=lambda: "RED", num_turns=lambda: 8),
        root_color="RED",
    )
    assert not mcts._is_information_set_turn_boundary(same, depth=1)
    assert mcts._is_information_set_turn_boundary(opponent, depth=1)
    assert mcts._is_information_set_turn_boundary(later, depth=1)
    assert not mcts._is_information_set_turn_boundary(opponent, depth=0)
