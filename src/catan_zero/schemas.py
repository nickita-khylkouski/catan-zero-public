from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Resource(StrEnum):
    BRICK = "brick"
    LUMBER = "lumber"
    ORE = "ore"
    GRAIN = "grain"
    WOOL = "wool"


class Phase(StrEnum):
    SETUP_SETTLEMENT = "setup_settlement"
    SETUP_ROAD = "setup_road"
    ROLL = "roll"
    DISCARD = "discard"
    ROBBER_MOVE = "robber_move"
    ROBBER_STEAL = "robber_steal"
    MAIN = "main"
    TRADE_RESPONSE = "trade_response"
    GAME_OVER = "game_over"


class DevCardKind(StrEnum):
    KNIGHT = "knight"
    ROAD_BUILDING = "road_building"
    YEAR_OF_PLENTY = "year_of_plenty"
    MONOPOLY = "monopoly"
    VICTORY_POINT = "victory_point"


class ActionKind(StrEnum):
    ROLL_DICE = "roll_dice"
    END_TURN = "end_turn"
    BUILD_ROAD = "build_road"
    BUILD_SETTLEMENT = "build_settlement"
    BUILD_CITY = "build_city"
    BUY_DEVELOPMENT_CARD = "buy_development_card"
    PLAY_DEVELOPMENT_CARD = "play_development_card"
    MOVE_ROBBER = "move_robber"
    STEAL_RESOURCE = "steal_resource"
    DISCARD_RESOURCES = "discard_resources"
    MARITIME_TRADE = "maritime_trade"
    OFFER_TRADE = "offer_trade"
    ACCEPT_TRADE = "accept_trade"
    REJECT_TRADE = "reject_trade"
    COUNTER_TRADE = "counter_trade"


@dataclass(frozen=True, slots=True)
class ResourceBundle:
    brick: int = 0
    lumber: int = 0
    ore: int = 0
    grain: int = 0
    wool: int = 0

    def __post_init__(self) -> None:
        for value in self.to_dict().values():
            if value < 0:
                raise ValueError("resource counts must be non-negative")

    def to_dict(self) -> dict[str, int]:
        return {
            Resource.BRICK.value: self.brick,
            Resource.LUMBER.value: self.lumber,
            Resource.ORE.value: self.ore,
            Resource.GRAIN.value: self.grain,
            Resource.WOOL.value: self.wool,
        }

    @classmethod
    def from_dict(cls, data: dict[str, int] | None) -> "ResourceBundle":
        data = data or {}
        return cls(
            brick=int(data.get(Resource.BRICK.value, 0)),
            lumber=int(data.get(Resource.LUMBER.value, 0)),
            ore=int(data.get(Resource.ORE.value, 0)),
            grain=int(data.get(Resource.GRAIN.value, 0)),
            wool=int(data.get(Resource.WOOL.value, 0)),
        )

    def total(self) -> int:
        return sum(self.to_dict().values())


@dataclass(frozen=True, slots=True)
class Action:
    kind: ActionKind
    actor: int
    target_player: int | None = None
    target_node: int | None = None
    target_edge: int | None = None
    target_tile: int | None = None
    give: ResourceBundle = field(default_factory=ResourceBundle)
    receive: ResourceBundle = field(default_factory=ResourceBundle)
    dev_card: DevCardKind | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "actor": self.actor,
            "target_player": self.target_player,
            "target_node": self.target_node,
            "target_edge": self.target_edge,
            "target_tile": self.target_tile,
            "give": self.give.to_dict(),
            "receive": self.receive.to_dict(),
            "dev_card": self.dev_card.value if self.dev_card else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Action":
        return cls(
            kind=ActionKind(data["kind"]),
            actor=int(data["actor"]),
            target_player=data.get("target_player"),
            target_node=data.get("target_node"),
            target_edge=data.get("target_edge"),
            target_tile=data.get("target_tile"),
            give=ResourceBundle.from_dict(data.get("give")),
            receive=ResourceBundle.from_dict(data.get("receive")),
            dev_card=DevCardKind(data["dev_card"]) if data.get("dev_card") else None,
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True, slots=True)
class PublicPlayerState:
    seat: int
    public_victory_points: int
    resource_count: int
    development_card_count: int
    roads_remaining: int
    settlements_remaining: int
    cities_remaining: int
    knights_played: int
    has_longest_road: bool = False
    has_largest_army: bool = False
    ports: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Event:
    ply: int
    event_type: str
    actor: int | None
    public: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Observation:
    ruleset_id: str
    acting_seat: int
    phase: Phase
    public_board: dict[str, Any]
    public_players: tuple[PublicPlayerState, ...]
    own_resources: ResourceBundle
    own_development_cards: tuple[DevCardKind, ...]
    public_event_history: tuple[Event, ...]
    legal_actions: tuple[Action, ...]

    def assert_no_hidden_opponent_fields(self) -> None:
        forbidden = {
            "opponent_resources",
            "opponent_development_cards",
            "development_deck_order",
            "future_dice",
            "future_steals",
        }
        payload = repr(self)
        leaks = [field for field in forbidden if field in payload]
        if leaks:
            raise AssertionError(f"observation contains hidden fields: {leaks}")


@dataclass(frozen=True, slots=True)
class SeedBundle:
    board: int
    dice: int
    development_deck: int
    robber_steal: int
    seat_order: int


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    game_id: str
    ply: int
    acting_seat: int
    observation: Observation
    selected_action: Action
    full_state_teacher: dict[str, Any] | None = None
    outcome: dict[str, Any] | None = None

