from __future__ import annotations

from typing import Any

import numpy as np


GRAPH_HISTORY_FEATURE_SIZE = 192

RESOURCE_NAMES = ("wood", "brick", "sheep", "wheat", "ore")
RESOURCE_INDEX = {resource: index for index, resource in enumerate(RESOURCE_NAMES)}
PROMPTS = (
    "BUILD_INITIAL_SETTLEMENT",
    "BUILD_INITIAL_ROAD",
    "ROLL",
    "PLAY_TURN",
    "DISCARD",
    "MOVE_ROBBER",
    "RESPOND_TO_TRADE",
    "CONFIRM_TRADE",
)
ACTION_TYPES = (
    "BUILD_SETTLEMENT",
    "BUILD_ROAD",
    "BUILD_CITY",
    "BUY_DEVELOPMENT_CARD",
    "MARITIME_TRADE",
    "offer_trade",
    "accept_trade",
    "reject_trade",
    "MOVE_ROBBER",
    "DISCARD_RESOURCE",
    "PLAY_KNIGHT_CARD",
    "PLAY_YEAR_OF_PLENTY",
    "PLAY_MONOPOLY",
    "PLAY_ROAD_BUILDING",
    "ROLL",
    "END_TURN",
)
EVENT_TYPES = (
    "reset",
    "board_action",
    "trade_proposal",
    "trade_response",
    "trade_counteroffer",
    "chat",
    "timeout",
    "invalid_action",
)


def build_graph_history_feature_vector(
    env: Any,
    actor: str | None = None,
    *,
    history_limit: int = 64,
) -> np.ndarray:
    """Build a compact public board/history feature suffix for one actor.

    This is the first graph/history bridge for the existing PPO stack. It uses
    the Colonist-style public payload and keeps hidden opponent cards redacted.
    The output is intentionally fixed-width so it can be appended to the
    current Catanatron feature vector without changing the trainer interface.
    """

    actor_name = actor or env.current_player_name()
    payload = env.observation_payload(actor_name, include_event_log=True)
    builder = _FeatureBuilder(GRAPH_HISTORY_FEATURE_SIZE)
    _encode_prompt(builder, payload)
    _encode_board(builder, payload)
    _encode_players(builder, payload, actor_name)
    _encode_legal_actions(builder, payload)
    _encode_history(builder, payload, actor_name, history_limit=history_limit)
    return builder.finish()


class _FeatureBuilder:
    def __init__(self, size: int) -> None:
        self.values = np.zeros(size, dtype=np.float32)
        self.index = 0

    def add(self, value: float) -> None:
        if self.index >= self.values.shape[0]:
            return
        self.values[self.index] = float(np.nan_to_num(value, nan=0.0))
        self.index += 1

    def add_many(self, values: list[float] | tuple[float, ...]) -> None:
        for value in values:
            self.add(float(value))

    def finish(self) -> np.ndarray:
        return self.values


def _encode_prompt(builder: _FeatureBuilder, payload: dict[str, Any]) -> None:
    prompt = str(payload.get("current_prompt", ""))
    for known in PROMPTS:
        builder.add(1.0 if known in prompt else 0.0)
    builder.add(_scale_count(len(payload.get("legal_actions", ())), 128))
    builder.add(_scale_count(payload.get("replay_frame_count", 0), 512))


def _encode_board(builder: _FeatureBuilder, payload: dict[str, Any]) -> None:
    board = payload.get("board") if isinstance(payload.get("board"), dict) else {}
    tiles = tuple(board.get("tiles", ()))
    resource_pips = [0.0] * len(RESOURCE_NAMES)
    robber_pips = [0.0] * len(RESOURCE_NAMES)
    number_pips = [0.0] * 11
    node_pips = _node_pip_map(board)
    robber_coordinate = tuple(board.get("robber_coordinate", ()) or ())
    for tile in tiles:
        if not isinstance(tile, dict):
            continue
        resource = _resource_index(tile.get("resource"))
        pips = _dice_pips(tile.get("number"))
        if resource is not None:
            resource_pips[resource] += pips
            if bool(tile.get("has_robber")) or tuple(tile.get("coordinate", ())) == robber_coordinate:
                robber_pips[resource] += pips
        number = _safe_int(tile.get("number"))
        if number is not None and 2 <= number <= 12 and number != 7:
            number_pips[number - 2] += pips
    builder.add_many([value / 36.0 for value in resource_pips])
    builder.add_many([value / 5.0 for value in robber_pips])
    builder.add_many([value / 5.0 for value in number_pips])

    ports = tuple(board.get("ports", ()))
    port_counts = [0.0] * (len(RESOURCE_NAMES) + 1)
    for port in ports:
        if not isinstance(port, dict):
            continue
        resource = _resource_index(port.get("resource"))
        if resource is None:
            port_counts[-1] += 1.0
        else:
            port_counts[resource] += 1.0
    builder.add_many([value / 4.0 for value in port_counts])

    buildings = tuple(board.get("buildings", ()))
    roads = tuple(board.get("roads", ()))
    player_names = tuple(payload.get("players", {}).keys())
    for name in player_names[:4]:
        settlement_count = 0.0
        city_count = 0.0
        production = 0.0
        for building in buildings:
            if not isinstance(building, dict) or building.get("player") != name:
                continue
            node = _safe_int(building.get("node"))
            pips = node_pips.get(node, 0.0)
            if str(building.get("building_type")) == "CITY":
                city_count += 1.0
                production += 2.0 * pips
            else:
                settlement_count += 1.0
                production += pips
        road_count = sum(
            1
            for road in roads
            if isinstance(road, dict) and road.get("player") == name
        )
        builder.add(settlement_count / 5.0)
        builder.add(city_count / 4.0)
        builder.add(float(road_count) / 15.0)
        builder.add(production / 36.0)


def _encode_players(
    builder: _FeatureBuilder,
    payload: dict[str, Any],
    actor_name: str,
) -> None:
    players = payload.get("players") if isinstance(payload.get("players"), dict) else {}
    names = list(players.keys())[:4]
    if actor_name in names:
        names = [actor_name] + [name for name in names if name != actor_name]
    for name in names[:4]:
        player = players.get(name, {})
        builder.add(1.0 if name == actor_name else 0.0)
        builder.add(_scale_count(player.get("public_victory_points", 0), 10))
        builder.add(_scale_count(player.get("resource_card_count", 0), 20))
        builder.add(_scale_count(player.get("development_card_count", 0), 10))
        builder.add(1.0 - _scale_count(player.get("roads_left", 15), 15))
        builder.add(1.0 - _scale_count(player.get("settlements_left", 5), 5))
        builder.add(1.0 - _scale_count(player.get("cities_left", 4), 4))
        builder.add(float(bool(player.get("has_largest_army"))))
        builder.add(float(bool(player.get("has_longest_road"))))
        builder.add(_scale_count(player.get("longest_road_length", 0), 15))
        resources = player.get("resources")
        for resource in RESOURCE_NAMES:
            builder.add(_resource_count(resources, resource) / 10.0)


def _encode_legal_actions(builder: _FeatureBuilder, payload: dict[str, Any]) -> None:
    legal = payload.get("structured_legal_actions", ())
    counts = {action_type: 0 for action_type in ACTION_TYPES}
    for action in legal:
        if not isinstance(action, dict):
            continue
        action_type = str(action.get("action_type"))
        if action_type in counts:
            counts[action_type] += 1
    for action_type in ACTION_TYPES:
        builder.add(_scale_count(counts[action_type], 64))


def _encode_history(
    builder: _FeatureBuilder,
    payload: dict[str, Any],
    actor_name: str,
    *,
    history_limit: int,
) -> None:
    events = tuple(payload.get("event_log", ()))[-history_limit:]
    event_counts = {event_type: 0.0 for event_type in EVENT_TYPES}
    action_counts = {action_type: 0.0 for action_type in ACTION_TYPES}
    actor_event_count = 0.0
    trade_event_count = 0.0
    decay = 1.0
    decayed_actions = {action_type: 0.0 for action_type in ACTION_TYPES}
    for event in reversed(events):
        if not isinstance(event, dict):
            continue
        event_type = str(event.get("event_type", event.get("type", "")))
        if event_type in event_counts:
            event_counts[event_type] += 1.0
        if event.get("actor") == actor_name:
            actor_event_count += 1.0
        if "trade" in event_type:
            trade_event_count += 1.0
        action_type = _event_action_type(event)
        if action_type in action_counts:
            action_counts[action_type] += 1.0
            decayed_actions[action_type] += decay
        decay *= 0.92
    denom = max(float(len(events)), 1.0)
    for event_type in EVENT_TYPES:
        builder.add(event_counts[event_type] / denom)
    for action_type in ACTION_TYPES:
        builder.add(action_counts[action_type] / denom)
    for action_type in ACTION_TYPES:
        builder.add(decayed_actions[action_type])
    builder.add(actor_event_count / denom)
    builder.add(trade_event_count / denom)
    builder.add(_scale_count(len(events), history_limit))


def _node_pip_map(board: dict[str, Any]) -> dict[int, float]:
    result: dict[int, float] = {}
    for tile in board.get("tiles", ()):
        if not isinstance(tile, dict):
            continue
        pips = _dice_pips(tile.get("number"))
        nodes = tile.get("nodes") if isinstance(tile.get("nodes"), dict) else {}
        for node in nodes.values():
            node_id = _safe_int(node)
            if node_id is not None:
                result[node_id] = result.get(node_id, 0.0) + pips
    return result


def _event_action_type(event: dict[str, Any]) -> str | None:
    payload = event.get("payload")
    if not isinstance(payload, dict):
        return None
    action = payload.get("action")
    if isinstance(action, dict):
        return str(action.get("action_type"))
    return None


def _resource_index(value: Any) -> int | None:
    resource = _resource_name(value)
    return RESOURCE_INDEX.get(resource) if resource is not None else None


def _resource_name(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).lower()
    return normalized if normalized in RESOURCE_INDEX else None


def _resource_count(resources: Any, resource: str) -> float:
    if not isinstance(resources, dict):
        return 0.0
    for key, value in resources.items():
        if _resource_name(key) == resource:
            return float(_safe_int(value) or 0)
    return 0.0


def _dice_pips(number: Any) -> float:
    value = _safe_int(number)
    if value is None:
        return 0.0
    return float({2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}.get(value, 0))


def _scale_count(value: Any, denominator: int) -> float:
    numeric = float(_safe_int(value) or 0)
    return min(max(numeric / max(float(denominator), 1.0), 0.0), 1.0)


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
