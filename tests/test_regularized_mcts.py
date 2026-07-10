from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from catan_zero.search.regularized_mcts import (
    RegularizedPolicyMCTS,
    reverse_kl_improved_policy,
)
from catan_zero.search.rust_mcts import _ActionStats, _Node


def test_reverse_kl_policy_is_normalized_and_respects_q() -> None:
    policy = reverse_kl_improved_policy(
        {1: 0.8, 2: 0.2},
        {1: 0.0, 2: 1.0},
        root_to_act=True,
        temperature=0.25,
    )
    assert sum(policy.values()) == pytest.approx(1.0)
    assert policy[2] > policy[1]


def test_opponent_turn_reverses_root_perspective_q() -> None:
    root_policy = reverse_kl_improved_policy(
        {1: 0.5, 2: 0.5}, {1: 0.8, 2: -0.8}, root_to_act=True, temperature=1.0
    )
    opponent_policy = reverse_kl_improved_policy(
        {1: 0.5, 2: 0.5}, {1: 0.8, 2: -0.8}, root_to_act=False, temperature=1.0
    )
    assert root_policy[1] == pytest.approx(opponent_policy[2])
    assert root_policy[2] == pytest.approx(opponent_policy[1])


@pytest.mark.parametrize("temperature", [0.0, -1.0, math.inf, math.nan])
def test_invalid_temperature_fails_closed(temperature: float) -> None:
    with pytest.raises(ValueError, match="temperature"):
        reverse_kl_improved_policy(
            {1: 1.0}, {1: 0.0}, root_to_act=True, temperature=temperature
        )


def test_policy_deficit_selector_visits_underrepresented_action() -> None:
    selector = object.__new__(RegularizedPolicyMCTS)
    selector.regularized_config = SimpleNamespace(
        regularization_temperature=1.0,
        prior_floor=1.0e-12,
    )
    node = _Node(game=SimpleNamespace(current_color=lambda: "RED"), root_color="RED")
    node.actions = {
        1: _ActionStats(prior=0.5, visits=9, value_sum=0.0),
        2: _ActionStats(prior=0.5, visits=1, value_sum=0.0),
    }
    action, stats = selector._select_action(node)
    assert action == 2
    assert stats is node.actions[2]
