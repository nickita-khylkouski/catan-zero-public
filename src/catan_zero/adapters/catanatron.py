from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from catan_zero.engine import CatanEngine
from catan_zero.schemas import (
    Action,
    ActionKind,
    DevCardKind,
    Event,
    Observation,
    Phase,
    PublicPlayerState,
    ResourceBundle,
    SeedBundle,
)


RULESET_ID = "CatanBench-4P-Full-v1"

_CATANATRON_RESOURCE_TO_ZERO = {
    "WOOD": "lumber",
    "BRICK": "brick",
    "SHEEP": "wool",
    "WHEAT": "grain",
    "ORE": "ore",
}
_ZERO_RESOURCE_TO_CATANATRON = {
    value: key for key, value in _CATANATRON_RESOURCE_TO_ZERO.items()
}
_CATANATRON_DEV_TO_ZERO = {
    "KNIGHT": DevCardKind.KNIGHT,
    "ROAD_BUILDING": DevCardKind.ROAD_BUILDING,
    "YEAR_OF_PLENTY": DevCardKind.YEAR_OF_PLENTY,
    "MONOPOLY": DevCardKind.MONOPOLY,
    "VICTORY_POINT": DevCardKind.VICTORY_POINT,
}


def _ensure_catanatron_importable() -> None:
    try:
        import catanatron  # noqa: F401

        return
    except ImportError:
        pass

    project_root = Path(__file__).resolve().parents[3]
    vendored = project_root / "vendor" / "catanatron" / "catanatron"
    if vendored.exists():
        sys.path.insert(0, str(vendored))

    try:
        import catanatron  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Catanatron is not installed and vendor/catanatron/catanatron was not importable."
        ) from exc


def _resource_bundle_from_freqdeck(freqdeck: list[int] | tuple[int, ...]) -> ResourceBundle:
    return ResourceBundle(
        lumber=int(freqdeck[0]),
        brick=int(freqdeck[1]),
        wool=int(freqdeck[2]),
        grain=int(freqdeck[3]),
        ore=int(freqdeck[4]),
    )


def _resource_bundle_from_listdeck(resources: Any) -> ResourceBundle:
    counts = {resource: 0 for resource in _CATANATRON_RESOURCE_TO_ZERO}
    for resource in resources or ():
        if resource is not None:
            counts[str(resource)] += 1
    return _resource_bundle_from_freqdeck(
        (counts["WOOD"], counts["BRICK"], counts["SHEEP"], counts["WHEAT"], counts["ORE"])
    )


def _resource_bundle_to_freqdeck(bundle: ResourceBundle) -> tuple[int, int, int, int, int]:
    data = bundle.to_dict()
    return (
        data["lumber"],
        data["brick"],
        data["wool"],
        data["grain"],
        data["ore"],
    )


def _single_resource_from_bundle(bundle: ResourceBundle) -> str | None:
    for resource, count in bundle.to_dict().items():
        if count:
            return _ZERO_RESOURCE_TO_CATANATRON[resource]
    return None


def _expand_dev_cards(state: Any, color: Any) -> tuple[DevCardKind, ...]:
    from catanatron.state_functions import player_key

    key = player_key(state, color)
    cards: list[DevCardKind] = []
    for catanatron_card, zero_card in _CATANATRON_DEV_TO_ZERO.items():
        cards.extend(
            [zero_card] * int(state.player_state[f"{key}_{catanatron_card}_IN_HAND"])
        )
    return tuple(cards)


def _encode_native_value(value: Any) -> Any:
    from catanatron.models.player import Color

    if isinstance(value, Color):
        return {"color": value.name}
    if isinstance(value, tuple):
        return [_encode_native_value(item) for item in value]
    if isinstance(value, list):
        return [_encode_native_value(item) for item in value]
    return value


def _decode_native_value(value: Any) -> Any:
    from catanatron.models.player import Color

    if isinstance(value, dict) and set(value) == {"color"}:
        return Color[value["color"]]
    if isinstance(value, list):
        return tuple(_decode_native_value(item) for item in value)
    return value


def _seat_for_color(state: Any, color: Any | None) -> int | None:
    if color is None:
        return None
    return int(state.color_to_index[color])


def _phase_for_state(game: Any) -> Phase:
    from catanatron.models.enums import ActionPrompt
    from catanatron.state_functions import player_has_rolled

    state = game.state
    if game.winning_color() is not None:
        return Phase.GAME_OVER
    if state.current_prompt == ActionPrompt.BUILD_INITIAL_SETTLEMENT:
        return Phase.SETUP_SETTLEMENT
    if state.current_prompt == ActionPrompt.BUILD_INITIAL_ROAD:
        return Phase.SETUP_ROAD
    if state.current_prompt == ActionPrompt.DISCARD:
        return Phase.DISCARD
    if state.current_prompt == ActionPrompt.MOVE_ROBBER:
        return Phase.ROBBER_MOVE
    if state.current_prompt in (ActionPrompt.DECIDE_TRADE, ActionPrompt.DECIDE_ACCEPTEES):
        return Phase.TRADE_RESPONSE
    if state.current_prompt == ActionPrompt.PLAY_TURN and not player_has_rolled(
        state, state.current_color()
    ):
        return Phase.ROLL
    return Phase.MAIN


def _native_metadata(native_action: Any) -> dict[str, Any]:
    return {
        "action_type": native_action.action_type.name,
        "color": native_action.color.name,
        "value": _encode_native_value(native_action.value),
    }


def _public_board(game: Any) -> dict[str, Any]:
    board = game.state.board
    tiles = []
    for coordinate, tile in board.map.land_tiles.items():
        tiles.append(
            {
                "id": tile.id,
                "coordinate": tuple(coordinate),
                "resource": (
                    _CATANATRON_RESOURCE_TO_ZERO[tile.resource]
                    if tile.resource is not None
                    else None
                ),
                "number": tile.number,
                "has_robber": coordinate == board.robber_coordinate,
                "nodes": {direction.value: node_id for direction, node_id in tile.nodes.items()},
                "edges": {
                    direction.value: tuple(sorted(edge))
                    for direction, edge in tile.edges.items()
                },
            }
        )

    ports = [
        {
            "id": port.id,
            "resource": (
                _CATANATRON_RESOURCE_TO_ZERO[port.resource]
                if port.resource is not None
                else None
            ),
            "direction": port.direction.value,
            "nodes": {direction.value: node_id for direction, node_id in port.nodes.items()},
        }
        for port in board.map.ports_by_id.values()
    ]

    buildings = {
        node_id: {"seat": _seat_for_color(game.state, color), "kind": building.lower()}
        for node_id, (color, building) in board.buildings.items()
    }
    roads = {
        tuple(sorted(edge)): _seat_for_color(game.state, color)
        for edge, color in board.roads.items()
        if edge[0] < edge[1]
    }

    return {
        "map_type": "BASE",
        "tiles": sorted(tiles, key=lambda tile: tile["id"]),
        "ports": sorted(ports, key=lambda port: port["id"]),
        "buildings": buildings,
        "roads": roads,
        "robber_coordinate": tuple(board.robber_coordinate),
        "bank_resource_counts": _resource_bundle_from_freqdeck(
            game.state.resource_freqdeck
        ).to_dict(),
        "development_cards_remaining": len(game.state.development_listdeck),
    }


def _public_players(game: Any) -> tuple[PublicPlayerState, ...]:
    from catanatron.state_functions import (
        player_key,
        player_num_dev_cards,
        player_num_resource_cards,
    )

    players = []
    for seat, color in enumerate(game.state.colors):
        key = player_key(game.state, color)
        ports = []
        for port_resource in game.state.board.get_player_port_resources(color):
            ports.append(
                "3:1"
                if port_resource is None
                else f"2:1:{_CATANATRON_RESOURCE_TO_ZERO[port_resource]}"
            )
        players.append(
            PublicPlayerState(
                seat=seat,
                public_victory_points=int(game.state.player_state[f"{key}_VICTORY_POINTS"]),
                resource_count=int(player_num_resource_cards(game.state, color)),
                development_card_count=int(player_num_dev_cards(game.state, color)),
                roads_remaining=int(game.state.player_state[f"{key}_ROADS_AVAILABLE"]),
                settlements_remaining=int(
                    game.state.player_state[f"{key}_SETTLEMENTS_AVAILABLE"]
                ),
                cities_remaining=int(game.state.player_state[f"{key}_CITIES_AVAILABLE"]),
                knights_played=int(game.state.player_state[f"{key}_PLAYED_KNIGHT"]),
                has_longest_road=bool(game.state.player_state[f"{key}_HAS_ROAD"]),
                has_largest_army=bool(game.state.player_state[f"{key}_HAS_ARMY"]),
                ports=tuple(sorted(ports)),
            )
        )
    return tuple(players)


def _action_kind_and_fields(native_action: Any, state: Any) -> dict[str, Any]:
    from catanatron.models.enums import ActionType

    action_type = native_action.action_type
    value = native_action.value
    fields: dict[str, Any] = {
        "actor": _seat_for_color(state, native_action.color),
        "metadata": {"catanatron": _native_metadata(native_action)},
    }

    if action_type == ActionType.ROLL:
        fields["kind"] = ActionKind.ROLL_DICE
    elif action_type == ActionType.END_TURN:
        fields["kind"] = ActionKind.END_TURN
    elif action_type == ActionType.BUILD_ROAD:
        fields["kind"] = ActionKind.BUILD_ROAD
        fields["target_edge"] = _stable_edge_id(value)
        fields["metadata"]["edge"] = tuple(value)
    elif action_type == ActionType.BUILD_SETTLEMENT:
        fields["kind"] = ActionKind.BUILD_SETTLEMENT
        fields["target_node"] = int(value)
    elif action_type == ActionType.BUILD_CITY:
        fields["kind"] = ActionKind.BUILD_CITY
        fields["target_node"] = int(value)
    elif action_type == ActionType.BUY_DEVELOPMENT_CARD:
        fields["kind"] = ActionKind.BUY_DEVELOPMENT_CARD
    elif action_type == ActionType.PLAY_KNIGHT_CARD:
        fields["kind"] = ActionKind.PLAY_DEVELOPMENT_CARD
        fields["dev_card"] = DevCardKind.KNIGHT
    elif action_type == ActionType.PLAY_YEAR_OF_PLENTY:
        fields["kind"] = ActionKind.PLAY_DEVELOPMENT_CARD
        fields["dev_card"] = DevCardKind.YEAR_OF_PLENTY
        fields["receive"] = _resource_bundle_from_listdeck(value)
    elif action_type == ActionType.PLAY_MONOPOLY:
        fields["kind"] = ActionKind.PLAY_DEVELOPMENT_CARD
        fields["dev_card"] = DevCardKind.MONOPOLY
        fields["receive"] = _resource_bundle_from_listdeck((value,))
    elif action_type == ActionType.PLAY_ROAD_BUILDING:
        fields["kind"] = ActionKind.PLAY_DEVELOPMENT_CARD
        fields["dev_card"] = DevCardKind.ROAD_BUILDING
    elif action_type == ActionType.MOVE_ROBBER:
        coordinate, victim = value
        fields["kind"] = ActionKind.MOVE_ROBBER
        fields["target_tile"] = _stable_tile_id(state, coordinate)
        fields["target_player"] = _seat_for_color(state, victim)
        fields["metadata"]["coordinate"] = tuple(coordinate)
    elif action_type == ActionType.DISCARD_RESOURCE:
        fields["kind"] = ActionKind.DISCARD_RESOURCES
        fields["give"] = _resource_bundle_from_listdeck((value,))
    elif action_type == ActionType.MARITIME_TRADE:
        fields["kind"] = ActionKind.MARITIME_TRADE
        fields["give"] = _resource_bundle_from_listdeck(value[:-1])
        fields["receive"] = _resource_bundle_from_listdeck(value[-1:])
    elif action_type == ActionType.OFFER_TRADE:
        fields["kind"] = ActionKind.OFFER_TRADE
        fields["give"] = _resource_bundle_from_freqdeck(value[:5])
        fields["receive"] = _resource_bundle_from_freqdeck(value[5:10])
    elif action_type == ActionType.ACCEPT_TRADE:
        fields["kind"] = ActionKind.ACCEPT_TRADE
    elif action_type == ActionType.REJECT_TRADE:
        fields["kind"] = ActionKind.REJECT_TRADE
    elif action_type == ActionType.CONFIRM_TRADE:
        fields["kind"] = ActionKind.ACCEPT_TRADE
        fields["target_player"] = _seat_for_color(state, value[10])
        fields["give"] = _resource_bundle_from_freqdeck(value[:5])
        fields["receive"] = _resource_bundle_from_freqdeck(value[5:10])
    elif action_type == ActionType.CANCEL_TRADE:
        fields["kind"] = ActionKind.REJECT_TRADE
    else:
        raise ValueError(f"Unsupported Catanatron action type: {action_type}")
    return fields


def _stable_edge_id(edge: tuple[int, int]) -> int:
    a, b = sorted(edge)
    return a * 100 + b


def _stable_tile_id(state: Any, coordinate: tuple[int, int, int]) -> int:
    return int(state.board.map.land_tiles[coordinate].id)


def _to_zero_action(native_action: Any, state: Any) -> Action:
    return Action(**_action_kind_and_fields(native_action, state))


def _to_native_action(action: Action, state: Any) -> Any:
    from catanatron.models.enums import Action as CatanatronAction
    from catanatron.models.enums import ActionType
    from catanatron.models.player import Color

    metadata = action.metadata.get("catanatron") if action.metadata else None
    if metadata:
        return CatanatronAction(
            Color[metadata["color"]],
            ActionType[metadata["action_type"]],
            _decode_native_value(metadata["value"]),
        )

    color = state.colors[action.actor]
    if action.kind == ActionKind.ROLL_DICE:
        return CatanatronAction(color, ActionType.ROLL, None)
    if action.kind == ActionKind.END_TURN:
        return CatanatronAction(color, ActionType.END_TURN, None)
    if action.kind == ActionKind.BUILD_SETTLEMENT:
        return CatanatronAction(color, ActionType.BUILD_SETTLEMENT, action.target_node)
    if action.kind == ActionKind.BUILD_CITY:
        return CatanatronAction(color, ActionType.BUILD_CITY, action.target_node)
    if action.kind == ActionKind.BUY_DEVELOPMENT_CARD:
        return CatanatronAction(color, ActionType.BUY_DEVELOPMENT_CARD, None)
    if action.kind == ActionKind.DISCARD_RESOURCES:
        resource = _single_resource_from_bundle(action.give)
        if resource is None:
            raise ValueError("discard action must include one resource in give")
        return CatanatronAction(color, ActionType.DISCARD_RESOURCE, resource)
    if action.kind == ActionKind.MARITIME_TRADE:
        give = []
        for resource, count in zip(
            ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"),
            _resource_bundle_to_freqdeck(action.give),
        ):
            give.extend([resource] * count)
        if len(give) > 4:
            raise ValueError("maritime trade gives at most four resources")
        receive = _single_resource_from_bundle(action.receive)
        if receive is None:
            raise ValueError("maritime trade must include one received resource")
        value = tuple(give + [None] * (4 - len(give)) + [receive])
        return CatanatronAction(color, ActionType.MARITIME_TRADE, value)
    if action.kind == ActionKind.PLAY_DEVELOPMENT_CARD:
        if action.dev_card == DevCardKind.KNIGHT:
            return CatanatronAction(color, ActionType.PLAY_KNIGHT_CARD, None)
        if action.dev_card == DevCardKind.ROAD_BUILDING:
            return CatanatronAction(color, ActionType.PLAY_ROAD_BUILDING, None)
        if action.dev_card == DevCardKind.MONOPOLY:
            resource = _single_resource_from_bundle(action.receive)
            return CatanatronAction(color, ActionType.PLAY_MONOPOLY, resource)
        if action.dev_card == DevCardKind.YEAR_OF_PLENTY:
            resources = []
            for resource, count in zip(
                ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"),
                _resource_bundle_to_freqdeck(action.receive),
            ):
                resources.extend([resource] * count)
            return CatanatronAction(color, ActionType.PLAY_YEAR_OF_PLENTY, tuple(resources))
    raise ValueError(
        "Action cannot be losslessly mapped to Catanatron without metadata: "
        f"{action.kind.value}"
    )


def _event_public(record: Any, state: Any) -> dict[str, Any]:
    from catanatron.models.enums import ActionType

    action = record.action
    action_type = action.action_type
    public = {
        "action_type": action_type.value,
        "value": _encode_native_value(action.value),
    }
    if action_type == ActionType.ROLL:
        public["dice"] = tuple(record.result)
    elif action_type == ActionType.BUY_DEVELOPMENT_CARD:
        public["value"] = None
        public["result"] = "hidden_development_card"
    elif action_type == ActionType.MOVE_ROBBER:
        coordinate, victim = action.value
        public["value"] = {
            "coordinate": tuple(coordinate),
            "victim_seat": _seat_for_color(state, victim),
        }
        public["result"] = "hidden_stolen_resource" if record.result is not None else None
    elif action_type == ActionType.DISCARD_RESOURCE:
        public["value"] = None
        public["discarded_count"] = 1
    return public


class CatanatronAdapter(CatanEngine):
    """Catanatron-backed implementation of the CatanZero simulator boundary."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self._game: Any | None = None

    def reset(self, seed_bundle: SeedBundle) -> Observation:
        _ensure_catanatron_importable()
        from catanatron.game import Game
        from catanatron.models.map import build_map
        from catanatron.models.player import Color, SimplePlayer

        map_type = self.config.get("map_type", "BASE")
        number_placement = self.config.get("number_placement", "official_spiral")
        players = [
            SimplePlayer(Color.RED),
            SimplePlayer(Color.BLUE),
            SimplePlayer(Color.WHITE),
            SimplePlayer(Color.ORANGE),
        ]
        catan_map = build_map(map_type, number_placement=number_placement)
        self._game = Game(
            players=players,
            seed=seed_bundle.board,
            discard_limit=int(self.config.get("discard_limit", 7)),
            friendly_robber=bool(self.config.get("friendly_robber", False)),
            vps_to_win=int(self.config.get("vps_to_win", 10)),
            catan_map=catan_map,
            number_placement=number_placement,
        )
        acting_seat = _seat_for_color(self._game.state, self._game.state.current_color())
        assert acting_seat is not None
        return self.observe(acting_seat)

    def legal_actions(self, player_id: int) -> tuple[Action, ...]:
        game = self._require_game()
        if _seat_for_color(game.state, game.state.current_color()) != player_id:
            return ()
        return tuple(_to_zero_action(action, game.state) for action in game.playable_actions)

    def observe(self, player_id: int) -> Observation:
        game = self._require_game()
        color = game.state.colors[player_id]
        from catanatron.state_functions import get_player_freqdeck

        acting_seat = _seat_for_color(game.state, game.state.current_color())
        assert acting_seat is not None
        observation = Observation(
            ruleset_id=RULESET_ID,
            acting_seat=acting_seat,
            phase=_phase_for_state(game),
            public_board=_public_board(game),
            public_players=_public_players(game),
            own_resources=_resource_bundle_from_freqdeck(
                get_player_freqdeck(game.state, color)
            ),
            own_development_cards=_expand_dev_cards(game.state, color),
            public_event_history=self.event_log(),
            legal_actions=self.legal_actions(player_id),
        )
        observation.assert_no_hidden_opponent_fields()
        return observation

    def step(self, action: Action) -> Any:
        game = self._require_game()
        native_action = _to_native_action(action, game.state)
        return game.execute(native_action)

    def clone(self) -> Any:
        return self._require_game().copy()

    def restore(self, snapshot: Any) -> None:
        self._game = snapshot.copy() if hasattr(snapshot, "copy") else snapshot

    def event_log(self) -> tuple[Event, ...]:
        game = self._require_game()
        return tuple(
            Event(
                ply=ply,
                event_type=record.action.action_type.value,
                actor=_seat_for_color(game.state, record.action.color),
                public=_event_public(record, game.state),
            )
            for ply, record in enumerate(game.state.action_records)
        )

    def _require_game(self) -> Any:
        if self._game is None:
            raise RuntimeError("CatanatronAdapter.reset() must be called first")
        return self._game
