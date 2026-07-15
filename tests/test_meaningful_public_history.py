from __future__ import annotations

import numpy as np

from catan_zero.rl.entity_token_features import (
    ACTION_TYPES,
    EVENT_FEATURE_SIZE,
    _event_mask,
    _event_tokens,
)
from catan_zero.rl.meaningful_history import (
    MEANINGFUL_PUBLIC_HISTORY_LIMIT,
    is_meaningful_public_event,
    meaningful_public_events,
    public_events_from_native_action_records,
)


def _event(
    action_type: str,
    *,
    actor: str = "RED",
    legal_count: int | None = None,
    value=None,
):
    payload = {
        "action": {"index": None, "action_type": action_type, "value": value},
    }
    if legal_count is not None:
        payload["public_legal_action_count_before"] = legal_count
        payload["public_was_sole_legal_action"] = legal_count == 1
    return {
        "event_type": "board_action",
        "actor": actor,
        "turn_key": None,
        "payload": payload,
    }


def test_taxonomy_keeps_strategy_and_excludes_automatic_ui_plumbing():
    for action_type in (
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
    ):
        assert is_meaningful_public_event(_event(action_type, legal_count=3))

    for action_type in (
        "ROLL",
        "END_TURN",
        "offer_trade",
        "accept_trade",
        "reject_trade",
    ):
        assert not is_meaningful_public_event(_event(action_type, legal_count=3))
    assert not is_meaningful_public_event(_event("BUILD_ROAD", legal_count=1))
    assert not is_meaningful_public_event({"event_type": "chat", "payload": {}})


def test_private_or_legacy_legal_width_never_controls_public_history():
    event = _event("DISCARD_RESOURCE")
    event["payload"]["legal_action_count_before"] = 1
    event["payload"]["was_sole_legal_action"] = True
    assert is_meaningful_public_event(event)

    malformed_public = _event("BUILD_ROAD")
    malformed_public["payload"]["public_legal_action_count_before"] = "secret"
    assert not is_meaningful_public_event(malformed_public)


def test_filter_runs_before_exact_32_event_cap():
    events = []
    for index in range(50):
        events.append(_event("BUILD_ROAD", actor="RED" if index % 2 else "BLUE"))
        events.append(_event("ROLL"))
        events.append(_event("END_TURN"))

    selected = meaningful_public_events(events, limit=64)
    assert len(selected) == MEANINGFUL_PUBLIC_HISTORY_LIMIT
    assert all(
        event["payload"]["action"]["action_type"] == "BUILD_ROAD"
        for event in selected
    )
    assert selected == tuple(events[54::3])


def test_existing_event_encoder_uses_32_meaningful_rows_without_schema_growth():
    payload = {
        "event_log": [
            _event("ROLL"),
            _event("BUILD_CITY", actor="BLUE"),
            _event("END_TURN"),
            _event("MOVE_ROBBER", value=[[0, 0, 0], "RED"]),
        ]
    }
    tokens = _event_tokens(
        payload,
        {},
        history_limit=32,
        meaningful_public_history=True,
    )
    mask = _event_mask(
        payload,
        history_limit=32,
        meaningful_public_history=True,
    )

    assert tokens.shape == (32, EVENT_FEATURE_SIZE)
    assert mask.sum() == 2
    assert np.flatnonzero(mask).tolist() == [30, 31]
    build_city = ACTION_TYPES.index("BUILD_CITY")
    move_robber = ACTION_TYPES.index("MOVE_ROBBER")
    assert tokens[30, 17 + build_city] == 1.0
    assert tokens[31, 17 + move_robber] == 1.0
    assert tokens[31, 36 + 1] == 1.0  # RED public robber victim.


def test_native_record_translation_redacts_every_hidden_card_identity():
    records = [
        {
            "action": ["RED", "BUY_DEVELOPMENT_CARD", "VICTORY_POINT"],
            "result": "VICTORY_POINT",
        },
        {
            "action": ["BLUE", "DISCARD_RESOURCE", "ORE"],
            "result": "ORE",
        },
        {
            "action": ["RED", "MOVE_ROBBER", [[0, 0, 0], "BLUE"]],
            "result": "WHEAT",
        },
    ]
    events = public_events_from_native_action_records(records, [2, 0, 1])
    rendered = repr(events)

    assert "VICTORY_POINT" not in rendered
    assert "ORE" not in rendered
    assert "WHEAT" not in rendered
    assert events[0]["payload"]["result"] == "hidden_development_card"
    assert events[0]["payload"]["public_legal_action_count_before"] == 2
    assert events[1]["payload"]["action"]["index"] is None
    assert "public_legal_action_count_before" not in events[1]["payload"]
    assert events[2]["payload"]["result"] == "hidden_stolen_resource"
    assert not is_meaningful_public_event(events[2])
