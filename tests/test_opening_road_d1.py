from __future__ import annotations

import json

import pytest

from catan_zero.search.gumbel_chance_mcts import (
    GumbelChanceMCTS,
    GumbelChanceMCTSConfig,
    _GAction,
    _GNode,
)


def _mcts(config: GumbelChanceMCTSConfig) -> GumbelChanceMCTS:
    # These are pure completed-Q/operator tests. Avoid constructing the engine
    # so they run even in the CPU unit-test environment without catanatron_rs.
    instance = object.__new__(GumbelChanceMCTS)
    instance.config = config
    return instance


def _near_tie_node(phase: str) -> _GNode:
    class _ColorGame:
        @staticmethod
        def current_color() -> str:
            return "RED"

    node = _GNode(game=_ColorGame(), root_color="RED", root_phase=phase)
    node.actions = {
        0: _GAction(prior=1.0 / 3.0, visits=4, value_sum=0.400004),
        1: _GAction(prior=1.0 / 3.0, visits=4, value_sum=0.400000),
        2: _GAction(prior=1.0 / 3.0, visits=4, value_sum=0.399996),
    }
    node.action_logits = {action: 0.0 for action in node.actions}
    return node


def test_scoped_d1_defaults_off_and_preserves_legacy_global_d1_semantics() -> None:
    assert GumbelChanceMCTSConfig().rescale_noise_floor_initial_road_only is False
    global_d1 = _mcts(
        GumbelChanceMCTSConfig(rescale_noise_floor_c=8.0, sigma_eval=0.98)
    )
    play_turn = _near_tie_node("PLAY_TURN")
    completed = global_d1._completed_q(play_turn)
    assert global_d1._rescaled_completed_q(play_turn, completed) != (
        global_d1._rescale_completed_q(completed)
    )


def test_scoped_d1_deconfidents_only_initial_road_near_ties() -> None:
    control = _mcts(GumbelChanceMCTSConfig())
    scoped = _mcts(
        GumbelChanceMCTSConfig(
            rescale_noise_floor_c=8.0,
            sigma_eval=0.98,
            rescale_noise_floor_initial_road_only=True,
        )
    )
    road = _near_tie_node("BUILD_INITIAL_ROAD")
    settlement = _near_tie_node("BUILD_INITIAL_SETTLEMENT")
    road_q = scoped._completed_q(road)
    settlement_q = scoped._completed_q(settlement)

    control_road = control._improved_policy(road, road_q)
    scoped_road = scoped._improved_policy(road, road_q)
    assert max(control_road.values()) > 0.40
    assert max(scoped_road.values()) == pytest.approx(1.0 / 3.0, abs=1.0e-4)

    # Exact dict equality, not approximate equality: off-phase takes the same
    # historical rescale/softmax path and never invokes attenuation.
    assert scoped._rescaled_completed_q(settlement, settlement_q) == (
        control._rescaled_completed_q(settlement, settlement_q)
    )
    assert scoped._improved_policy(settlement, settlement_q) == (
        control._improved_policy(settlement, settlement_q)
    )


def test_scoped_d1_phase_attestation_is_opt_in_and_fail_closed() -> None:
    class _Game:
        def json_snapshot(self) -> str:
            return json.dumps({"current_prompt": "BUILD_INITIAL_ROAD"})

    off = _mcts(GumbelChanceMCTSConfig())
    scoped = _mcts(
        GumbelChanceMCTSConfig(rescale_noise_floor_initial_road_only=True)
    )
    assert off._phase_gated_d1_root_phase(object()) is None
    assert scoped._phase_gated_d1_root_phase(_Game()) == "BUILD_INITIAL_ROAD"
    with pytest.raises(RuntimeError, match="json_snapshot"):
        scoped._phase_gated_d1_root_phase(object())
