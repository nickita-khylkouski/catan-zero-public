from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, Literal

try:  # pragma: no cover - fallback exists for dependency-light imports.
    import numpy as np
except ImportError:  # pragma: no cover - exercised in clean envs without numpy.
    np = None

from catan_zero.rl._catanatron import import_catanatron_module

MapType = Literal["BASE", "TOURNAMENT", "MINI"]


@dataclass(frozen=True, slots=True)
class ActionDescriptor:
    """Stable, serializable view of a flat Catanatron action-space entry."""

    index: int
    action_type: str
    value: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "action_type": self.action_type,
            "value": _serialize_value(self.value),
        }


def _color_names(colors: Iterable[Any]) -> tuple[str, ...]:
    return tuple(getattr(color, "name", str(color)) for color in colors)


def _serialize_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "name"):
        return value.name
    if isinstance(value, tuple):
        return tuple(_serialize_value(item) for item in value)
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    return repr(value)


def _load_color(name: str) -> Any:
    player_module = import_catanatron_module("catanatron.models.player")
    return getattr(player_module.Color, name)


@lru_cache(maxsize=None)
def _action_array(color_names: tuple[str, ...], map_type: MapType) -> tuple[Any, ...]:
    board_module = import_catanatron_module("catanatron.models.board")
    enums_module = import_catanatron_module("catanatron.models.enums")
    map_module = import_catanatron_module("catanatron.models.map")

    player_colors = tuple(_load_color(name) for name in color_names)
    catan_map = map_module.build_map(map_type)
    num_nodes = len(catan_map.land_nodes)

    actions = [
        (enums_module.ActionType.ROLL, None),
        *[
            (enums_module.ActionType.DISCARD_RESOURCE, resource)
            for resource in enums_module.RESOURCES
        ],
        *[
            (enums_module.ActionType.BUILD_ROAD, tuple(sorted(edge)))
            for edge in board_module.get_edges(catan_map.land_nodes)
        ],
        *[
            (enums_module.ActionType.BUILD_SETTLEMENT, node_id)
            for node_id in range(num_nodes)
        ],
        *[
            (enums_module.ActionType.BUILD_CITY, node_id)
            for node_id in range(num_nodes)
        ],
        (enums_module.ActionType.BUY_DEVELOPMENT_CARD, None),
        (enums_module.ActionType.PLAY_KNIGHT_CARD, None),
        *[
            (enums_module.ActionType.PLAY_YEAR_OF_PLENTY, (first_card, resources[j]))
            for resources in [enums_module.RESOURCES]
            for i, first_card in enumerate(resources)
            for j in range(i, len(resources))
        ],
        *[
            (enums_module.ActionType.PLAY_YEAR_OF_PLENTY, (first_card,))
            for first_card in enums_module.RESOURCES
        ],
        (enums_module.ActionType.PLAY_ROAD_BUILDING, None),
        *[
            (enums_module.ActionType.PLAY_MONOPOLY, resource)
            for resource in enums_module.RESOURCES
        ],
        *[
            (enums_module.ActionType.MOVE_ROBBER, (coordinates, victim_color))
            for coordinates in catan_map.land_tiles.keys()
            for victim_color in (None, *player_colors)
        ],
        *[
            (enums_module.ActionType.MARITIME_TRADE, tuple(4 * [give] + [receive]))
            for give in enums_module.RESOURCES
            for receive in enums_module.RESOURCES
            if give != receive
        ],
        *[
            (
                enums_module.ActionType.MARITIME_TRADE,
                tuple(3 * [give] + [None, receive]),
            )
            for give in enums_module.RESOURCES
            for receive in enums_module.RESOURCES
            if give != receive
        ],
        *[
            (
                enums_module.ActionType.MARITIME_TRADE,
                tuple(2 * [give] + [None, None, receive]),
            )
            for give in enums_module.RESOURCES
            for receive in enums_module.RESOURCES
            if give != receive
        ],
        (enums_module.ActionType.END_TURN, None),
    ]
    return tuple(sorted(actions, key=lambda action: str(action)))


class ActionCatalog:
    """Fixed flat action catalog plus mask helpers for Catanatron games.

    This intentionally mirrors Catanatron's maskable discrete space. Domestic
    trade offers are not represented in this first flat catalog; later trade
    training should add a separate structured head instead of widening this
    action dimension with every possible offer tuple.
    """

    version = "catanatron-flat-v1"

    def __init__(self, player_colors: Iterable[Any], map_type: MapType = "BASE") -> None:
        self.map_type = map_type
        self.color_names = _color_names(player_colors)
        self._actions = _action_array(self.color_names, self.map_type)
        self._index_by_action = {
            (action_type, value): index
            for index, (action_type, value) in enumerate(self._actions)
        }

    @property
    def size(self) -> int:
        return len(self._actions)

    def raw_entry(self, index: int) -> tuple[Any, Any]:
        return self._actions[index]

    def descriptor(self, index: int) -> ActionDescriptor:
        action_type, value = self.raw_entry(index)
        return ActionDescriptor(index=index, action_type=action_type.name, value=value)

    def describe(self, index: int) -> dict[str, Any]:
        return self.descriptor(index).to_dict()

    def encode(self, action: Any) -> int:
        return self._index_by_action[(action.action_type, action.value)]

    def try_encode(self, action: Any) -> int | None:
        return self._index_by_action.get((action.action_type, action.value))

    def decode(self, index: int, color: Any) -> Any:
        if index < 0 or index >= self.size:
            raise IndexError(f"action index {index} outside [0, {self.size})")
        enums_module = import_catanatron_module("catanatron.models.enums")
        action_type, value = self.raw_entry(index)
        return enums_module.Action(color, action_type, value)

    def valid_actions(self, playable_actions: Iterable[Any]) -> tuple[int, ...]:
        valid = [self.try_encode(action) for action in playable_actions]
        return tuple(sorted(index for index in valid if index is not None))

    def mask(self, playable_actions: Iterable[Any]) -> Any:
        if np is None:
            valid_actions = set(self.valid_actions(playable_actions))
            return tuple(index in valid_actions for index in range(self.size))

        mask = np.zeros(self.size, dtype=np.bool_)
        for index in self.valid_actions(playable_actions):
            mask[index] = True
        return mask

    def unmapped_actions(self, playable_actions: Iterable[Any]) -> tuple[str, ...]:
        return tuple(
            repr(action) for action in playable_actions if self.try_encode(action) is None
        )
