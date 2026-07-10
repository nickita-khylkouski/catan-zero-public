from __future__ import annotations

from dataclasses import replace
import random
from types import SimpleNamespace

import pytest

from catan_zero.search.regularized_mcts import (
    RegularizedMCTSConfig,
    RegularizedPolicyMCTS,
)
from catan_zero.search.rust_mcts import RustMCTS, RustMCTSConfig, RustMCTSResult


class _Game:
    def __init__(self, public_key: int, hidden_truth: int, value: float = 0.0):
        self.public_key = public_key
        self.hidden_truth = hidden_truth
        self.value = value

    def current_color(self):
        return "RED"

    def playable_action_indices(self, _colors, _map_kind):
        return [1, 2]

    def num_turns(self):
        return 7

    def determinize_for_player(self, observer: str, seed: int):
        assert observer == "RED"
        # Deliberately independent of hidden_truth.
        value = ((self.public_key * 37 + seed) % 101) / 100.0
        return _Game(self.public_key, hidden_truth=-1, value=value)


class _FakePublicRustMCTS(RustMCTS):
    budgets: list[int] = []

    def __init__(self, config, evaluator):
        self.config = config
        self.evaluator = evaluator
        self.rng = random.Random(config.seed)

    def _spawn_information_set_particle(self, *, simulations: int, seed: int):
        self.budgets.append(int(simulations))
        particle = type(self)(
            replace(
                self.config,
                simulations=simulations,
                seed=seed,
                information_set_search=False,
            ),
            self.evaluator,
        )
        particle._root_actor_turn_only = True
        return particle

    def _search_single_world(self, game):
        budget = int(self.config.simulations)
        first = int(round(float(game.value) * budget))
        first = min(max(first, 0), budget)
        second = budget - first
        return RustMCTSResult(
            action=1 if first >= second else 2,
            policy={1: first / budget, 2: second / budget},
            visits={1: first, 2: second},
            q_values={1: game.value, 2: -game.value},
            priors={1: 0.5, 2: 0.5},
            root_value=game.value,
        )


def test_puct_public_particles_are_truth_invariant_and_split_total_budget():
    _FakePublicRustMCTS.budgets = []
    config = RustMCTSConfig(
        simulations=10,
        seed=19,
        information_set_search=True,
        determinization_particles=4,
        determinization_min_simulations=1,
    )
    evaluator = SimpleNamespace(config=SimpleNamespace(public_observation=True))
    first = _FakePublicRustMCTS(config, evaluator).search(_Game(3, hidden_truth=1))
    first_budgets = list(_FakePublicRustMCTS.budgets)
    _FakePublicRustMCTS.budgets = []
    second = _FakePublicRustMCTS(config, evaluator).search(_Game(3, hidden_truth=999))
    assert first == second
    assert first_budgets == [3, 3, 2, 2]
    assert sum(first.visits.values()) == 10


def test_puct_public_search_requires_public_evaluator():
    search = _FakePublicRustMCTS(
        RustMCTSConfig(information_set_search=True),
        SimpleNamespace(config=SimpleNamespace(public_observation=False)),
    )
    with pytest.raises(RuntimeError, match="public_observation"):
        search.search(_Game(1, hidden_truth=2))


def test_regularized_config_propagates_public_search_contract(monkeypatch):
    monkeypatch.setattr(
        "catan_zero.search.rust_mcts._require_rust_module", lambda: object()
    )
    config = RegularizedMCTSConfig(
        information_set_search=True,
        determinization_particles=3,
        determinization_min_simulations=8,
    )
    search = RegularizedPolicyMCTS(config, evaluator=object())
    assert search.config.information_set_search is True
    assert search.config.determinization_particles == 3
    assert search.config.determinization_min_simulations == 8
