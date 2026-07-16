from __future__ import annotations

import json
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
    def __init__(self, hidden_truth: str, prompt: str = "BUILD_INITIAL_ROAD") -> None:
        self.hidden_truth = hidden_truth
        self.prompt = prompt
        self.seeds: list[int] = []

    def current_color(self) -> str:
        return "RED"

    def determinize_for_player(self, observer: str, seed: int) -> "_SampledGame":
        assert observer == "RED"
        self.seeds.append(seed)
        # Deliberately ignore authoritative hidden_truth: this is the engine
        # contract the orchestration layer relies on.
        return _SampledGame(seed, self.prompt)

    def json_snapshot(self) -> str:
        return json.dumps(
            {
                "colors": ["RED", "BLUE"],
                "current_prompt": self.prompt,
                "hidden_truth": self.hidden_truth,
            }
        )

    def apply_public_belief_development_draws(self, *args, **kwargs):
        raise AssertionError("root orchestration must not materialize chance directly")


class _SampledGame:
    def __init__(self, seed: int, prompt: str = "BUILD_INITIAL_ROAD") -> None:
        self.seed = seed
        self.prompt = prompt

    def current_color(self) -> str:
        return "RED"

    def num_turns(self) -> int:
        return 7

    def json_snapshot(self) -> str:
        raise AssertionError("hidden-world particle phase must not be inspected")


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
    mcts._gumbel_rng = mcts.rng
    mcts._chance_rng = mcts.rng
    mcts._belief_rng = mcts.rng
    mcts.attested_root_phases = []

    def fetch(_self, _game):
        return (11, 12), {11: ["RED", "A", None], 12: ["RED", "B", None]}, {}

    def search_one(
        _self,
        game,
        *,
        force_full=None,
        n_simulations_override=None,
        attested_root_phase=None,
    ):
        assert isinstance(game, _SampledGame)
        assert force_full in {True, False}
        mcts.attested_root_phases.append(attested_root_phase)
        budget = (
            int(n_simulations_override)
            if n_simulations_override is not None
            else int(_self.config.n_full if force_full else _self.config.n_fast)
        )
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


def test_information_set_forwards_attested_public_phase_to_belief_aggregation() -> None:
    observed: list[str | None] = []
    mcts = _mcts()
    mcts.config = replace(
        mcts.config,
        rescale_noise_floor_c=8.0,
        rescale_noise_floor_initial_road_only=True,
    )
    original = mcts._aggregate_information_set_results

    def aggregate(
        _self,
        results,
        *,
        legal_actions,
        used_full_search,
        root_phase=None,
    ):
        observed.append(root_phase)
        return original(
            results,
            legal_actions=legal_actions,
            used_full_search=used_full_search,
            root_phase=root_phase,
        )

    mcts._aggregate_information_set_results = MethodType(aggregate, mcts)
    mcts.search(
        _AuthoritativeGame("opponent has KNIGHT", "BUILD_INITIAL_ROAD"),
        force_full=True,
    )
    assert observed == ["BUILD_INITIAL_ROAD"]


def test_information_set_never_reads_phase_from_hidden_world_particle() -> None:
    class _BadPhaseGame(_AuthoritativeGame):
        def determinize_for_player(self, observer: str, seed: int) -> _SampledGame:
            assert observer == "RED"
            return _SampledGame(seed, "BUILD_INITIAL_SETTLEMENT")

    mcts = _mcts()
    mcts.config = replace(
        mcts.config,
        rescale_noise_floor_c=8.0,
        rescale_noise_floor_initial_road_only=True,
    )
    mcts.search(
        _BadPhaseGame("opponent has KNIGHT", "BUILD_INITIAL_ROAD"),
        force_full=True,
    )
    assert mcts.attested_root_phases == ["BUILD_INITIAL_ROAD"] * 4


def test_information_set_particles_share_one_exact_total_budget() -> None:
    result = _mcts(particles=4, n_full=128).search(
        _AuthoritativeGame("irrelevant"), force_full=True
    )
    assert result.simulations_used == 128
    assert sum(result.visit_counts.values()) == 128


def test_coherent_public_belief_uses_one_sanitized_full_budget_tree() -> None:
    first_game = _AuthoritativeGame("opponent has KNIGHT")
    second_game = _AuthoritativeGame("opponent has VICTORY_POINT")
    first_mcts = _mcts(particles=4, n_full=128)
    first_mcts.config = replace(
        first_mcts.config,
        information_set_search=False,
        coherent_public_belief_search=True,
    )
    second_mcts = _mcts(particles=4, n_full=128)
    second_mcts.config = replace(
        second_mcts.config,
        information_set_search=False,
        coherent_public_belief_search=True,
    )

    first = first_mcts.search(first_game, force_full=True)
    second = second_mcts.search(second_game, force_full=True)

    assert first == second
    assert first.simulations_used == 128
    assert len(first_game.seeds) == len(second_game.seeds) == 1
    assert first_game.seeds == second_game.seeds


def test_information_set_d6_reuses_one_public_root_without_operator_drift() -> None:
    class _CountingD6Evaluator:
        def __init__(self) -> None:
            self.config = SimpleNamespace(public_observation=True)
            self.calls = 0

        def evaluate_symmetry_averaged(
            self, game, legal_actions, *, root_color, colors
        ):
            assert isinstance(game, _SampledGame)
            assert tuple(legal_actions) == (11, 12)
            assert root_color == "RED"
            assert tuple(colors) == ("RED", "BLUE")
            self.calls += 1
            return {11: 0.4, 12: 0.6}, 0.2

    missing = object()

    def build(*, share: bool):
        mcts = _mcts()
        mcts.config = replace(
            mcts.config,
            symmetry_averaged_eval=True,
            symmetry_averaged_eval_threshold=2,
        )
        evaluator = _CountingD6Evaluator()
        mcts.evaluator = evaluator
        seen_evaluations: list[object] = []

        def search_one(
            _self,
            game,
            *,
            force_full=None,
            n_simulations_override=None,
            attested_root_phase=None,
            precomputed_root_evaluation=missing,
        ):
            del attested_root_phase
            if precomputed_root_evaluation is missing:
                root_evaluation = _self.evaluator.evaluate_symmetry_averaged(
                    game,
                    (11, 12),
                    root_color="RED",
                    colors=("RED", "BLUE"),
                )
            else:
                root_evaluation = precomputed_root_evaluation
            seen_evaluations.append(root_evaluation)
            priors, root_value = root_evaluation
            budget = int(n_simulations_override)
            p11 = 0.25 + (game.seed % 100) / 1000.0
            return SearchResult(
                selected_action=11,
                improved_policy={11: p11, 12: 1.0 - p11},
                visit_counts={11: budget // 2, 12: budget - budget // 2},
                q_values={11: 0.1, 12: -0.1},
                priors=dict(priors),
                root_value=float(root_value),
                used_full_search=bool(force_full),
                simulations_used=budget,
            )

        mcts._search_single_world = MethodType(search_one, mcts)
        if not share:

            def never_share(_self, _legal_width):
                return False

            mcts._can_share_information_set_root_evaluation = MethodType(
                never_share, mcts
            )
        return mcts, evaluator, seen_evaluations

    shared, shared_evaluator, shared_objects = build(share=True)
    repeated, repeated_evaluator, _repeated_objects = build(share=False)
    shared_result = shared.search(_AuthoritativeGame("truth A"), force_full=True)
    repeated_result = repeated.search(
        _AuthoritativeGame("truth B"), force_full=True
    )

    assert shared_evaluator.calls == 1
    assert repeated_evaluator.calls == 4
    assert len({id(result) for result in shared_objects}) == 1
    assert shared_result == repeated_result
    assert shared_result.evaluator_method_calls == 1
    assert shared_result.logical_leaf_evaluations == 1
    assert shared_result.orientation_evaluation_rows == 12
    assert repeated_result.evaluator_method_calls == 4
    assert repeated_result.logical_leaf_evaluations == 4
    assert repeated_result.orientation_evaluation_rows == 48


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


def test_belief_d1_uses_fractional_particle_mean_visits_at_sparse_root() -> None:
    mcts = _belief_target_mcts(sigma_reference_visits=8)
    mcts.config = replace(
        mcts.config,
        rescale_noise_floor_c=1.0,
        sigma_eval=1.0,
    )
    particles = [
        _belief_result(
            q_values={11: 1.0},
            visits={11: 1, 12: 0},
            completed_q={11: 1.0, 12: -1.0},
        ),
        _belief_result(
            q_values={12: -1.0},
            visits={11: 0, 12: 1},
            completed_q={11: 1.0, 12: -1.0},
        ),
        _belief_result(
            q_values={},
            visits={11: 0, 12: 0},
            completed_q={11: 1.0, 12: -1.0},
        ),
        _belief_result(
            q_values={},
            visits={11: 0, 12: 0},
            completed_q={11: 1.0, 12: -1.0},
        ),
    ]

    policy = mcts._belief_level_improved_policy(
        particles,
        legal_actions=(11, 12),
        aggregate_priors={11: 0.5, 12: 0.5},
    )

    # The exact mean is 2 visits / (4 particles * 2 actions) = 0.25.
    # Rounding each per-action particle mean first made both synthetic visit
    # counts zero, forcing D1 alpha=0 and returning an incorrect 50/50 policy.
    assert policy[11] > 0.5
    assert policy[12] < 0.5


def test_belief_level_d1_is_exact_off_phase_and_active_at_road_root() -> None:
    control = _belief_target_mcts(sigma_reference_visits=8)
    scoped = _belief_target_mcts(sigma_reference_visits=8)
    scoped.config = replace(
        scoped.config,
        rescale_noise_floor_c=8.0,
        sigma_eval=0.98,
        rescale_noise_floor_initial_road_only=True,
    )
    particles = [
        _belief_result(
            q_values={11: 0.400004, 12: 0.400000},
            visits={11: 4, 12: 4},
            completed_q={11: 0.400004, 12: 0.400000},
        ),
        _belief_result(
            q_values={11: 0.400000, 12: 0.399996},
            visits={11: 4, 12: 4},
            completed_q={11: 0.400000, 12: 0.399996},
        ),
    ]

    def target(mcts, phase):
        return mcts._belief_level_improved_policy(
            particles,
            legal_actions=(11, 12),
            aggregate_priors={11: 0.5, 12: 0.5},
            root_phase=phase,
        )

    assert target(scoped, "BUILD_INITIAL_SETTLEMENT") == target(
        control, "BUILD_INITIAL_SETTLEMENT"
    )
    assert target(scoped, "PLAY_TURN") == target(control, "PLAY_TURN")
    assert target(scoped, "BUILD_INITIAL_ROAD")[11] < target(
        control, "BUILD_INITIAL_ROAD"
    )[11]


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

    mcts.config = replace(
        mcts.config,
        information_set_search=False,
        coherent_public_belief_search=True,
    )
    assert mcts._is_information_set_turn_boundary(opponent, depth=1)
    assert mcts._is_information_set_turn_boundary(later, depth=1)


def test_boundary_particles_average_values_with_real_per_world_legal_sets() -> None:
    class Particle:
        def __init__(self, seed: int) -> None:
            self.seed = int(seed)

    class BoundaryGame:
        def winning_color(self):
            return None

        def current_color(self):
            return "BLUE"

        def num_turns(self):
            return 7

        def determinize_from_observer_information(
            self, observer: str, seed: int
        ) -> Particle:
            assert observer == "RED"
            return Particle(seed)

    class Evaluator:
        config = SimpleNamespace(public_observation=True, emit_uncertainty=False)

        def __init__(self) -> None:
            self.requests: list[list[tuple[Particle, tuple[int, ...]]]] = []

        def evaluate_many(self, requests, *, root_color, colors):
            assert root_color == "RED"
            assert colors == ("RED", "BLUE")
            self.requests.append(list(requests))
            return [
                ({action: 1.0 / len(legal) for action in legal}, seed.seed / 10.0)
                for seed, legal in requests
            ]

    evaluator = Evaluator()
    mcts = _mcts()
    mcts.config = replace(
        mcts.config,
        information_set_search=False,
        coherent_public_belief_search=True,
        boundary_value_particles=3,
    )
    mcts.evaluator = evaluator
    mcts._boundary_value_particle_seeds = (1, 2, 3)
    mcts._begin_expansion_accounting()

    def fetch(_self, game):
        legal = (11,) if game.seed == 1 else ((11, 12) if game.seed == 2 else (12, 13, 14))
        return legal, {}, {}

    mcts._fetch_legal_actions = MethodType(fetch, mcts)
    node = _GNode(game=BoundaryGame(), root_color="RED")
    mcts._information_set_root_turn = 7

    first = mcts._simulate(node, depth=1)
    second = mcts._simulate(node, depth=1)

    assert first == pytest.approx(0.2)
    assert second == first
    assert node.expanded is True
    assert node.actions == {}
    assert [[legal for _game, legal in batch] for batch in evaluator.requests] == [
        [(11,), (11, 12), (12, 13, 14)]
    ]
    assert mcts._expansion_accounting_snapshot() == (0, 1)


def test_boundary_particle_configuration_fails_closed_on_uncertainty() -> None:
    with pytest.raises(ValueError, match="uncertainty backup weighting"):
        GumbelChanceMCTS(
            GumbelChanceMCTSConfig(
                coherent_public_belief_search=True,
                boundary_value_particles=2,
                uncertainty_backup_weighting=True,
            ),
            SimpleNamespace(config=SimpleNamespace(public_observation=True)),
        )


def test_coherent_boundary_particle_seeds_are_hidden_truth_invariant() -> None:
    class BoundaryAuthoritative(_AuthoritativeGame):
        def determinize_from_observer_information(self, observer: str, seed: int):
            return _SampledGame(seed, self.prompt)

    def run(hidden_truth: str) -> tuple[int, ...]:
        mcts = _mcts(n_full=128)
        mcts.config = replace(
            mcts.config,
            information_set_search=False,
            coherent_public_belief_search=True,
            boundary_value_particles=4,
        )
        mcts.evaluator.config.emit_uncertainty = False
        mcts.search(BoundaryAuthoritative(hidden_truth), force_full=True)
        return tuple(mcts._boundary_value_particle_seeds)

    assert run("KNIGHT") == run("VICTORY_POINT")


def test_boundary_value_particles_k1_draws_no_additional_rng() -> None:
    mcts = _mcts(n_full=128)
    mcts.config = replace(
        mcts.config,
        information_set_search=False,
        coherent_public_belief_search=True,
        boundary_value_particles=1,
    )
    expected = random.Random(mcts.config.seed)
    expected.getrandbits(64)  # Historical sanitized-root determinization draw.

    mcts.search(_AuthoritativeGame("irrelevant"), force_full=True)

    assert mcts._boundary_value_particle_seeds == ()
    assert mcts._belief_rng.getstate() == expected.getstate()


def test_boundary_value_particles_do_not_perturb_shared_search_rng() -> None:
    class BoundaryAuthoritative(_AuthoritativeGame):
        def determinize_from_observer_information(self, observer: str, seed: int):
            return _SampledGame(seed, self.prompt)

    def run(particles: int):
        mcts = _mcts(n_full=128)
        mcts.config = replace(
            mcts.config,
            information_set_search=False,
            coherent_public_belief_search=True,
            rng_stream_separation=False,
            boundary_value_particles=particles,
        )
        mcts.evaluator.config.emit_uncertainty = False

        def stop_before_traversal(
            _self,
            _sampled,
            *,
            force_full=None,
            attested_root_phase=None,
        ):
            return force_full, attested_root_phase

        mcts._search_single_world = MethodType(stop_before_traversal, mcts)
        mcts.seed_search_rngs(918273)
        mcts.search(BoundaryAuthoritative("irrelevant"), force_full=True)
        return mcts.rng.getstate(), tuple(mcts._boundary_value_particle_seeds)

    legacy_state, legacy_seeds = run(1)
    particle_state, particle_seeds = run(4)

    assert legacy_seeds == ()
    assert len(particle_seeds) == 4
    assert particle_state == legacy_state


def test_boundary_value_particle_samples_are_matched_prefixes_at_each_root() -> None:
    class BoundaryAuthoritative(_AuthoritativeGame):
        def determinize_from_observer_information(self, observer: str, seed: int):
            return _SampledGame(seed, self.prompt)

    def run(particles: int) -> list[tuple[int, ...]]:
        mcts = _mcts(n_full=128)
        mcts.config = replace(
            mcts.config,
            information_set_search=False,
            coherent_public_belief_search=True,
            rng_stream_separation=False,
            boundary_value_particles=particles,
        )
        mcts.evaluator.config.emit_uncertainty = False

        def stop_before_traversal(
            _self,
            _sampled,
            *,
            force_full=None,
            attested_root_phase=None,
        ):
            return force_full, attested_root_phase

        mcts._search_single_world = MethodType(stop_before_traversal, mcts)
        mcts.seed_search_rngs(192837)
        roots: list[tuple[int, ...]] = []
        for _ in range(2):
            mcts.search(BoundaryAuthoritative("irrelevant"), force_full=True)
            roots.append(tuple(mcts._boundary_value_particle_seeds))
        return roots

    k2 = run(2)
    k4 = run(4)

    assert k2[0] == k4[0][:2]
    assert k2[1] == k4[1][:2]


def test_boundary_value_particles_k1_uses_exact_legacy_boundary_expansion() -> None:
    class BoundaryGame:
        def winning_color(self):
            return None

        def current_color(self):
            return "BLUE"

        def num_turns(self):
            return 7

    mcts = _mcts()
    mcts.config = replace(
        mcts.config,
        information_set_search=False,
        coherent_public_belief_search=True,
        boundary_value_particles=1,
    )
    mcts._information_set_root_turn = 7
    node = _GNode(game=BoundaryGame(), root_color="RED")
    calls: list[_GNode] = []

    def legacy_expand(_self, received):
        calls.append(received)
        received.expanded = True
        received.prior_value = 0.375
        return 0.375

    def forbidden_particles(_self, _received):
        raise AssertionError("K=1 must not enter boundary-particle expansion")

    mcts._expand = MethodType(legacy_expand, mcts)
    mcts._expand_boundary_value_particles = MethodType(forbidden_particles, mcts)

    assert mcts._simulate(node, depth=1) == pytest.approx(0.375)
    assert mcts._simulate(node, depth=1) == pytest.approx(0.375)
    assert calls == [node]
