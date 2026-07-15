"""Bounded public action history for the two-player no-trade learner.

The entity model already has event tokens; this module only decides which
public events deserve those scarce tokens.  It deliberately removes UI/chance
plumbing (roll, end-turn, chat, timeouts, invalid actions and player-trade UI)
while retaining actions whose public consequences change a Catan plan.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


MEANINGFUL_PUBLIC_HISTORY_SCHEMA_VERSION = "meaningful_public_history_2p_no_trade_v1"
MEANINGFUL_PUBLIC_HISTORY_LIMIT = 32

# These actions expose strategic public information even when their private
# result is redacted.  In particular, repeated DISCARD_RESOURCE events encode
# the public discard count without revealing resource identities, and
# MOVE_ROBBER retains the public destination/victim while hiding the stolen
# card.  Road/settlement/knight events are retained because they can transfer
# the public Longest Road/Largest Army awards (the current holder itself is
# already represented in player tokens).
MEANINGFUL_PUBLIC_ACTION_TYPES = frozenset(
    {
        "BUILD_SETTLEMENT",
        "BUILD_ROAD",
        "BUILD_CITY",
        "BUY_DEVELOPMENT_CARD",
        "MARITIME_TRADE",
        "MOVE_ROBBER",
        "DISCARD_RESOURCE",
        "PLAY_KNIGHT_CARD",
        "PLAY_YEAR_OF_PLENTY",
        "PLAY_MONOPOLY",
        "PLAY_ROAD_BUILDING",
    }
)

# Player-to-player trade controls are intentionally absent: the adopted track
# is 2p_no_trade.  Keeping this list explicit prevents a future UI event from
# silently entering the model history merely because it contains an action.
AUTOMATIC_OR_UI_ACTION_TYPES = frozenset(
    {
        "ROLL",
        "END_TURN",
        "offer_trade",
        "accept_trade",
        "reject_trade",
        "cancel_trade",
        "confirm_trade",
        "OFFER_TRADE",
        "ACCEPT_TRADE",
        "REJECT_TRADE",
        "CANCEL_TRADE",
        "CONFIRM_TRADE",
    }
)


def event_action_type(event: dict[str, Any]) -> str:
    """Return the canonical action type carried by an event, if any."""

    payload = event.get("payload")
    if not isinstance(payload, dict):
        return ""
    action = payload.get("action")
    if isinstance(action, dict):
        return str(action.get("action_type", ""))
    return ""


def is_meaningful_public_event(event: Any) -> bool:
    """Whether ``event`` belongs in the next learner's public history.

    The producer may mark a transition automatic only when the old legal width
    was reconstructible from public state. Exact regular-turn/discard widths
    are deliberately ignored because they reveal hidden cards. Known
    strategic public action types are then admitted; everything else fails
    closed.
    """

    if not isinstance(event, dict) or str(event.get("event_type", "")) != "board_action":
        return False
    payload = event.get("payload")
    if not isinstance(payload, dict):
        return False
    action_type = event_action_type(event)
    if action_type in AUTOMATIC_OR_UI_ACTION_TYPES:
        return False
    public_sole = payload.get("public_was_sole_legal_action")
    if public_sole is not None:
        if not isinstance(public_sole, bool):
            return False
        if public_sole:
            return False
    public_legal_count = payload.get("public_legal_action_count_before")
    if public_legal_count is not None:
        try:
            if isinstance(public_legal_count, bool):
                return False
            if int(public_legal_count) <= 1:
                return False
        except (TypeError, ValueError):
            return False
    return action_type in MEANINGFUL_PUBLIC_ACTION_TYPES


def meaningful_public_events(
    events: Iterable[Any],
    *,
    limit: int = MEANINGFUL_PUBLIC_HISTORY_LIMIT,
) -> tuple[dict[str, Any], ...]:
    """Return the most recent bounded meaningful public events.

    Filtering happens *before* truncation, so automatic ROLL/END_TURN traffic
    cannot evict an older strategic build/dev/robber event from the window.
    """

    if isinstance(limit, bool) or int(limit) < 0:
        raise ValueError("meaningful public history limit must be >= 0")
    bounded = min(int(limit), MEANINGFUL_PUBLIC_HISTORY_LIMIT)
    if bounded == 0:
        return ()
    filtered = tuple(event for event in events if is_meaningful_public_event(event))
    return filtered[-bounded:]


def public_events_from_native_action_records(
    records: Iterable[Any],
    public_legal_action_counts: Iterable[Any] = (),
) -> tuple[dict[str, Any], ...]:
    """Translate native ``json_snapshot.action_records`` without secrets.

    Native records are ``{"action": [color, type, value], "result": ...}``.
    BUY_DEVELOPMENT_CARD, DISCARD_RESOURCE and robber-steal results contain
    authoritative hidden identities, so those fields are redacted before the
    event can cross into a model-input payload.
    """

    public_counts = tuple(public_legal_action_counts)
    events: list[dict[str, Any]] = []
    for record_index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        raw_action = record.get("action")
        if not isinstance(raw_action, (list, tuple)) or len(raw_action) < 2:
            continue
        actor = str(raw_action[0])
        action_type = str(raw_action[1])
        value: Any = raw_action[2] if len(raw_action) > 2 else None
        result: Any = record.get("result")
        if action_type == "BUY_DEVELOPMENT_CARD":
            value = "hidden_development_card"
            result = "hidden_development_card"
        elif action_type == "DISCARD_RESOURCE":
            value = "hidden_resource"
            result = "hidden_resource"
        elif action_type == "MOVE_ROBBER" and result is not None:
            result = "hidden_stolen_resource"
        public_width: int | None = None
        if record_index < len(public_counts):
            try:
                candidate = public_counts[record_index]
                if not isinstance(candidate, bool) and int(candidate) > 0:
                    public_width = int(candidate)
            except (TypeError, ValueError):
                public_width = None
        events.append(
            {
                "event_id": len(events) + 1,
                "event_type": "board_action",
                "turn_key": None,
                "actor": actor,
                "payload": {
                    "action_index": None,
                    "action": {
                        "index": None,
                        "action_type": action_type,
                        "value": value,
                    },
                    "result": result,
                    "next_player": None,
                    **(
                        {
                            "public_legal_action_count_before": public_width,
                            "public_was_sole_legal_action": public_width == 1,
                        }
                        if public_width is not None
                        else {}
                    ),
                },
            }
        )
    return tuple(events)
