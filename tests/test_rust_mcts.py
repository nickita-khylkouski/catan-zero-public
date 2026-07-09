from __future__ import annotations

import json

import pytest

from catan_zero.search.neural_rust_mcts import (
    _entity_payload_from_rust_snapshot,
    _structured_action,
)
from catan_zero.search.rust_mcts import RustMCTS, RustMCTSConfig, _require_rust_module


def _rust():
    try:
        return _require_rust_module()
    except RuntimeError as error:
        pytest.skip(str(error))


def test_rust_mcts_selects_legal_action_and_returns_visit_policy():
    catanatron_rs = _rust()
    game = catanatron_rs.Game.simple(["RED", "BLUE"], seed=11)
    search = RustMCTS(RustMCTSConfig(simulations=12, seed=3))

    result = search.search(game)
    legal = set(game.playable_action_indices(["RED", "BLUE"], None))

    assert result.action in legal
    assert set(result.policy).issubset(legal)
    assert sum(result.visits.values()) == 12
    assert abs(sum(result.policy.values()) - 1.0) < 1.0e-9
    assert all(action in result.q_values for action in result.policy)
    assert all(action in result.priors for action in result.policy)


def test_rust_mcts_is_deterministic_for_same_seed():
    catanatron_rs = _rust()
    first = catanatron_rs.Game.simple(["RED", "BLUE"], seed=19)
    second = catanatron_rs.Game.simple(["RED", "BLUE"], seed=19)
    config = RustMCTSConfig(simulations=16, seed=99)

    a = RustMCTS(config).search(first)
    b = RustMCTS(config).search(second)

    assert a.action == b.action
    assert a.visits == b.visits
    assert a.policy == b.policy


def test_rust_mcts_binding_exposes_chance_spectrum():
    catanatron_rs = _rust()
    game = catanatron_rs.Game.simple(["RED", "BLUE"], seed=7)
    for _ in range(100):
        playable = json.loads(game.playable_actions_json())
        if playable and playable[0][1] == "ROLL":
            break
        game.play_tick()
    else:
        pytest.fail("did not reach ROLL prompt")

    roll = [action for action in json.loads(game.playable_actions_json()) if action[1] == "ROLL"][0]
    spectrum = json.loads(game.spectrum_json(json.dumps(roll)))
    assert len(spectrum) == 11
    assert abs(sum(float(entry["probability"]) for entry in spectrum) - 1.0) < 1.0e-9
    outcome = game.apply_chance_outcome(json.dumps(roll), 0)
    assert outcome.state_index() == game.state_index() + 1


def test_rust_entity_adapter_filters_non_land_tiles():
    catanatron_rs = _rust()
    game = catanatron_rs.Game(
        colors=["BLUE", "RED"],
        seed=123,
        player_kinds=["random", "value_function"],
        vps_to_win=10,
    )
    snapshot = json.loads(game.json_snapshot())
    states = {
        color: json.loads(game.player_state_json(color))
        for color in ("BLUE", "RED")
    }

    payload = _entity_payload_from_rust_snapshot(
        snapshot,
        states_by_color=states,
        structured_legal_actions=[],
        legal_action_ids=(),
    )

    tiles = payload["board"]["tiles"]
    assert len(tiles) == 19
    assert {int(tile["tile_id"]) for tile in tiles} == set(range(19))
    assert all(
        0 <= int(node) < 54
        for tile in tiles
        for node in dict(tile.get("nodes", {})).values()
    )
    assert all(
        any(0 <= int(node) < 54 for node in port.get("nodes", ()))
        for port in payload["board"]["ports"]
    )

    rust_ports = {
        int(raw["tile"]["id"]): (
            str(raw["tile"].get("resource")).lower()
            if raw["tile"].get("resource") is not None
            else None
        )
        for raw in snapshot["tiles"]
        if raw.get("tile", {}).get("type") == "PORT"
    }
    adapted_ports = {
        int(port["port_id"]): port.get("resource")
        for port in payload["board"]["ports"]
    }
    assert adapted_ports == rust_ports


def test_rust_entity_adapter_parses_maritime_trade_tuple():
    structured = _structured_action(
        123,
        ["BLUE", "MARITIME_TRADE", ["SHEEP", "SHEEP", None, None, "BRICK"]],
    )

    assert structured["args"]["give"] == ["sheep", "sheep"]
    assert structured["args"]["want"] == ["brick"]
