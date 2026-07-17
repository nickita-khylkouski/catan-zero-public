from __future__ import annotations

from typing import Any

import numpy as np

from catan_zero.rl.entity_token_features import _node_pips_by_resource
from catan_zero.rl.entity_feature_adapter import (
    LEGACY_MISSING_CHECKPOINT_ADAPTER_VERSION,
    RUST_ENTITY_ADAPTER_V6,
    require_known_entity_feature_adapter,
)
from catan_zero.rl.multiagent_env import ColonistMultiAgentEnv


CONTEXT_ACTION_FEATURE_SIZE = 18

_PRIORITY = {
    "BUILD_CITY": 100,
    "BUILD_SETTLEMENT": 90,
    "BUILD_ROAD": 70,
    "BUY_DEVELOPMENT_CARD": 60,
    "PLAY_KNIGHT_CARD": 55,
    "PLAY_YEAR_OF_PLENTY": 54,
    "PLAY_MONOPOLY": 53,
    "PLAY_ROAD_BUILDING": 52,
    "offer_trade": 45,
    "MARITIME_TRADE": 42,
    "ROLL": 40,
    "MOVE_ROBBER": 35,
    "DISCARD_RESOURCE": 30,
    "accept_trade": 25,
    "reject_trade": 24,
    "confirm_trade": 23,
    "cancel_trade": 22,
    "END_TURN": 0,
}

_DISCARD_RANK = {"wood": 0, "brick": 1, "sheep": 2, "wheat": 3, "ore": 4}


def build_action_context_feature_table(
    env: ColonistMultiAgentEnv,
    info: dict[str, Any] | None = None,
    *,
    entity_feature_adapter_version: str = LEGACY_MISSING_CHECKPOINT_ADAPTER_VERSION,
) -> np.ndarray:
    """Build board/current-state action features for every discrete action id."""

    adapter_version = require_known_entity_feature_adapter(
        entity_feature_adapter_version
    )
    valid_actions = set(int(action) for action in (info or {}).get("valid_actions", ()))
    payload = env.observation_payload(env.current_player_name(), include_event_log=False)
    actor_public_vp = float(
        payload["players"].get(env.current_player_name(), {}).get("public_victory_points", 0)
    )
    prompt = str(payload.get("current_prompt", ""))
    table = np.zeros(
        (env.action_space.n, CONTEXT_ACTION_FEATURE_SIZE),
        dtype=np.float32,
    )
    for action_index in range(env.action_space.n):
        structured = env.structured_action(action_index)
        if structured is None:
            continue
        table[action_index] = _context_vector(
            env,
            structured,
            valid=action_index in valid_actions,
            actor_public_vp=actor_public_vp,
            payload=payload,
            prompt=prompt,
            actor=env.current_player_name(),
            entity_feature_adapter_version=adapter_version,
        )
    return table


def _context_vector(
    env: ColonistMultiAgentEnv,
    structured: dict[str, Any],
    *,
    valid: bool,
    actor_public_vp: float,
    payload: dict[str, Any],
    prompt: str,
    actor: str,
    entity_feature_adapter_version: str,
) -> np.ndarray:
    adapter_version = require_known_entity_feature_adapter(
        entity_feature_adapter_version
    )
    features = np.zeros(CONTEXT_ACTION_FEATURE_SIZE, dtype=np.float32)
    action_type = str(structured["action_type"])
    args = structured.get("args") or {}
    features[0] = 1.0 if valid else 0.0
    features[1] = float(_PRIORITY.get(action_type, 0)) / 100.0
    features[10] = min(max(actor_public_vp / 10.0, 0.0), 1.0)
    features[12] = 1.0 if "INITIAL" in prompt else 0.0

    if action_type in ("BUILD_SETTLEMENT", "BUILD_CITY") and "node" in args:
        node = int(args["node"])
        features[2] = _scaled_production(env, node)
        features[13] = _port_access_score(payload, node)
        features[14] = _port_resource_score(payload, node)
        features[15] = _occupied_neighbor_score(payload, node)
    elif action_type == "BUILD_ROAD" and "edge" in args:
        productions = [_scaled_production(env, int(node)) for node in args["edge"]]
        if productions:
            features[3] = max(productions)
            features[4] = float(sum(productions) / len(productions))
        features[16] = _road_expansion_score(
            env,
            payload,
            args["edge"],
            prompt=prompt,
            actor=actor,
            entity_feature_adapter_version=adapter_version,
        )
    elif action_type == "MOVE_ROBBER":
        victim = args.get("victim")
        if victim is not None:
            features[5] = _scaled_player_public_vp(payload, victim)
    elif action_type == "DISCARD_RESOURCE":
        resource = str(args.get("resource", "")).lower()
        features[6] = float(_DISCARD_RANK.get(resource, 0)) / 4.0

    give_total, receive_total = _trade_totals(action_type, args, payload)
    features[7] = give_total / 4.0
    features[8] = receive_total / 4.0
    features[9] = (receive_total - give_total) / 4.0
    features[11] = 1.0 if action_type == "END_TURN" else 0.0
    features[17] = _offers_remaining_score(payload)
    return features


def _scaled_production(env: ColonistMultiAgentEnv, node_id: int) -> float:
    production = env.game.state.board.map.node_production.get(node_id)
    if production is None:
        return 0.0
    total_pips = sum(_node_pips_by_resource(production))
    return min(max(float(total_pips) / 18.0, 0.0), 1.0)


def _port_access_score(payload: dict[str, Any], node_id: int) -> float:
    return 1.0 if _port_for_node(payload, node_id) is not None else 0.0


def _port_resource_score(payload: dict[str, Any], node_id: int) -> float:
    port = _port_for_node(payload, node_id)
    if not isinstance(port, dict):
        return 0.0
    resource = port.get("resource")
    if resource is None:
        return 0.5
    return 1.0


def _port_for_node(payload: dict[str, Any], node_id: int) -> dict[str, Any] | None:
    board = payload.get("board")
    if not isinstance(board, dict):
        return None
    for port in board.get("ports", ()):
        if not isinstance(port, dict):
            continue
        if node_id in {int(node) for node in port.get("nodes", ())}:
            return port
    return None


def _occupied_neighbor_score(payload: dict[str, Any], node_id: int) -> float:
    occupied_nodes = _occupied_nodes(payload)
    if not occupied_nodes:
        return 0.0
    neighbors = set(_neighbor_nodes(payload, node_id))
    if not neighbors:
        return 0.0
    return len(neighbors & occupied_nodes) / max(len(neighbors), 1)


def _road_expansion_score(
    env: ColonistMultiAgentEnv,
    payload: dict[str, Any],
    edge: Any,
    *,
    prompt: str,
    actor: str,
    entity_feature_adapter_version: str,
) -> float:
    adapter_version = require_known_entity_feature_adapter(
        entity_feature_adapter_version
    )
    if adapter_version == RUST_ENTITY_ADAPTER_V6 and prompt == "BUILD_INITIAL_ROAD":
        return _initial_road_two_hop_expansion_score(
            env,
            payload,
            edge,
            actor=actor,
        )

    # Frozen v2-v5 contract. Do not repair or refactor this branch: deployed
    # checkpoints were trained with the proposed edge's unoccupied endpoint,
    # even though that adjacent endpoint cannot host the next settlement.
    occupied_nodes = _occupied_nodes(payload)
    scores = []
    for node in edge or ():
        node_id = int(node)
        if node_id in occupied_nodes:
            scores.append(0.0)
        else:
            scores.append(_scaled_production(env, node_id))
    return max(scores, default=0.0)


def _initial_road_two_hop_expansion_score(
    env: ColonistMultiAgentEnv,
    payload: dict[str, Any],
    edge: Any,
    *,
    actor: str,
) -> float:
    """Best production site opened by one further road from an initial road.

    The initial road is incident to the actor's just-built settlement. Its
    other endpoint is distance one and therefore cannot itself host the next
    settlement. V6 scores the legal distance-two settlement sites that become
    connected after one additional, currently unoccupied land edge.
    """

    endpoints = tuple(int(node) for node in (edge or ()))
    if len(endpoints) != 2:
        return 0.0
    owners = _building_owners(payload)
    origins = [node for node in endpoints if owners.get(node) == str(actor)]
    if len(origins) != 1:
        return 0.0
    origin = origins[0]
    frontier = endpoints[1] if endpoints[0] == origin else endpoints[0]
    if frontier in owners:
        return 0.0

    occupied_nodes = set(owners)
    occupied_edges = _occupied_edges(payload)
    scores: list[float] = []
    for target in _neighbor_nodes(payload, frontier):
        target = int(target)
        if target == origin or target in occupied_nodes:
            continue
        next_edge = tuple(sorted((frontier, target)))
        if next_edge in occupied_edges:
            continue
        # Catan distance rule: no building may be adjacent to the target.
        if any(int(neighbor) in occupied_nodes for neighbor in _neighbor_nodes(payload, target)):
            continue
        scores.append(_scaled_production(env, target))
    return max(scores, default=0.0)


def _occupied_nodes(payload: dict[str, Any]) -> set[int]:
    board = payload.get("board")
    if not isinstance(board, dict):
        return set()
    return {
        int(building["node"])
        for building in board.get("buildings", ())
        if isinstance(building, dict) and "node" in building
    }


def _building_owners(payload: dict[str, Any]) -> dict[int, str]:
    board = payload.get("board")
    if not isinstance(board, dict):
        return {}
    return {
        int(building["node"]): str(building.get("player", ""))
        for building in board.get("buildings", ())
        if isinstance(building, dict) and "node" in building
    }


def _occupied_edges(payload: dict[str, Any]) -> set[tuple[int, int]]:
    board = payload.get("board")
    if not isinstance(board, dict):
        return set()
    occupied: set[tuple[int, int]] = set()
    for road in board.get("roads", ()):
        if not isinstance(road, dict):
            continue
        edge = road.get("edge")
        if isinstance(edge, (list, tuple)) and len(edge) == 2:
            occupied.add(tuple(sorted((int(edge[0]), int(edge[1])))))
    return occupied


def _neighbor_nodes(payload: dict[str, Any], node_id: int) -> tuple[int, ...]:
    neighbors = []
    board = payload.get("board")
    if not isinstance(board, dict):
        return ()
    seen_edges = set()
    for tile in board.get("tiles", ()):
        if not isinstance(tile, dict):
            continue
        edges = tile.get("edges")
        if not isinstance(edges, dict):
            continue
        for edge in edges.values():
            edge_tuple = tuple(sorted(int(node) for node in edge))
            if edge_tuple in seen_edges:
                continue
            seen_edges.add(edge_tuple)
            if node_id in edge_tuple:
                neighbors.extend(node for node in edge_tuple if node != node_id)
    return tuple(dict.fromkeys(neighbors))


def _offers_remaining_score(payload: dict[str, Any]) -> float:
    panel = payload.get("trade_panel")
    if not isinstance(panel, dict):
        return 0.0
    return min(max(float(panel.get("offers_remaining_this_turn", 0)) / 3.0, 0.0), 1.0)


def _trade_totals(
    action_type: str,
    args: dict[str, Any],
    payload: dict[str, Any],
) -> tuple[float, float]:
    if action_type == "offer_trade":
        return _resource_total(args.get("give")), _resource_total(args.get("want"))
    if action_type == "MARITIME_TRADE":
        return _resource_total(args.get("give")), _resource_total(
            args.get("want", args.get("receive"))
        )
    current_trade = _current_board_trade_tuple(payload)
    if current_trade is not None:
        proposer_gives = _resource_total(current_trade[:5])
        proposer_wants = _resource_total(current_trade[5:10])
        if action_type in ("accept_trade", "reject_trade"):
            return proposer_wants, proposer_gives
        if action_type in ("confirm_trade", "cancel_trade"):
            return proposer_gives, proposer_wants
    return 0.0, 0.0


def _current_board_trade_tuple(payload: dict[str, Any]) -> tuple[Any, ...] | None:
    panel = payload.get("trade_panel")
    if not isinstance(panel, dict):
        return None
    current = panel.get("current_board_trade")
    if not isinstance(current, dict):
        return None
    trade = current.get("trade")
    if isinstance(trade, (tuple, list)) and len(trade) >= 10:
        return tuple(trade)
    return None


def _scaled_player_public_vp(payload: dict[str, Any], player: Any) -> float:
    players = payload.get("players", {})
    for key in _player_lookup_keys(player):
        if key in players:
            return min(
                max(float(players[key].get("public_victory_points", 0)) / 10.0, 0.0),
                1.0,
            )
    return 0.0


def _player_lookup_keys(player: Any) -> tuple[str, ...]:
    keys = [str(player)]
    name = getattr(player, "name", None)
    if name is not None:
        keys.append(str(name))
    if "." in keys[0]:
        keys.append(keys[0].rsplit(".", 1)[-1])
    return tuple(dict.fromkeys(keys))


def _resource_total(value: Any) -> float:
    if isinstance(value, dict):
        return float(sum(float(count) for count in value.values()))
    if isinstance(value, (tuple, list)):
        return float(len(value))
    return 0.0
