import random

import pytest

from catan_zero.rl import (
    ColonistMultiAgentConfig,
    ColonistMultiAgentEnv,
    load_replay_jsonl,
)
from catan_zero.rl.graph_history_features import (
    GRAPH_HISTORY_FEATURE_SIZE,
    build_graph_history_feature_vector,
)
RESOURCE_NAMES = ("wood", "brick", "sheep", "wheat", "ore")


def test_colonist_multiagent_env_reset_exposes_all_seats() -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    env = ColonistMultiAgentEnv(ColonistMultiAgentConfig(players=4))
    try:
        observations, info = env.reset(seed=100)

        assert set(observations) == {"BLUE", "RED", "ORANGE", "WHITE"}
        assert info["current_player"] in {"BLUE", "RED", "ORANGE", "WHITE"}
        assert len(info["action_mask"]) == env.action_space.n
        assert set(info["valid_actions"]) == {
            index for index, enabled in enumerate(info["action_mask"]) if enabled
        }
        assert len(info["legal_action_descriptions"]) == len(info["valid_actions"])
        assert {
            description["index"] for description in info["legal_action_descriptions"]
        } == set(info["valid_actions"])
        non_current = next(name for name in observations if name != info["current_player"])
        assert env.valid_actions(non_current) == ()
        assert all(flag is False for flag in env.action_mask(non_current))
    finally:
        env.close()


def test_colonist_multiagent_env_can_append_graph_history_features() -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    base_env = ColonistMultiAgentEnv(ColonistMultiAgentConfig(players=4))
    graph_env = ColonistMultiAgentEnv(
        ColonistMultiAgentConfig(players=4, use_graph_history_features=True)
    )
    try:
        base_observations, base_info = base_env.reset(seed=100)
        graph_observations, graph_info = graph_env.reset(seed=100)
        actor = graph_info["current_player"]
        suffix = build_graph_history_feature_vector(graph_env, actor)

        assert graph_info["valid_actions"] == base_info["valid_actions"]
        assert suffix.shape == (GRAPH_HISTORY_FEATURE_SIZE,)
        assert graph_observations[actor].shape[0] == (
            base_observations[actor].shape[0] + GRAPH_HISTORY_FEATURE_SIZE
        )
        assert graph_env.observation_space.shape == graph_observations[actor].shape
        assert suffix[0] == 1.0  # BUILD_INITIAL_SETTLEMENT prompt.
        assert suffix[10:15].sum() > 0.0  # board resource production signal.
    finally:
        base_env.close()
        graph_env.close()


def test_colonist_multiagent_env_surfaces_each_seat_without_autoplay() -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    env = ColonistMultiAgentEnv(ColonistMultiAgentConfig(players=4))
    try:
        _, info = env.reset(seed=101)
        seen_players = [info["current_player"]]

        for _ in range(12):
            _, _, terminated, truncated, info = env.step(info["valid_actions"][0])
            seen_players.append(info["current_player"])
            assert not (terminated or truncated)
            if "RED" in seen_players:
                break

        assert "RED" in seen_players
    finally:
        env.close()


def test_colonist_multiagent_env_chat_negotiation_and_timer_are_all_seat_public() -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    env = ColonistMultiAgentEnv(
        ColonistMultiAgentConfig(players=4, timer_profile="fast")
    )
    try:
        _, info = env.reset(seed=102)
        assert info["timer"]["phase"] == "initial_build"
        assert info["timer"]["budget_seconds"] == 120
        assert "accept_trade" in info["valid_chat_templates"]
        assert "robber_no_steal" in info["valid_chat_templates"]

        chat = env.post_chat_template(
            "trade_request",
            {"give": "brick", "want": "ore"},
            actor="RED",
        )
        offer = env.propose_trade(
            actor="RED",
            give={"brick": 1},
            want={"kind": "open", "count": 1},
        )
        response = env.respond_to_trade(offer["offer_id"], "countered", actor="BLUE")
        panel = env.trade_panel("BLUE")
        _, info_after = env.reset(seed=102)

        assert chat["actor"] == "RED"
        assert offer["actor"] == "RED"
        assert response["responses"]["BLUE"] == "countered"
        assert panel["offers"][0]["responder_statuses"]["BLUE"] == "countered"
        assert info_after["chat_log"] == ()
        assert info_after["negotiation_offers"] == ()
    finally:
        env.close()


def test_colonist_multiagent_env_observation_payload_is_actor_safe() -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    env = ColonistMultiAgentEnv(ColonistMultiAgentConfig(players=4))
    try:
        _, info = env.reset(seed=106)
        actor = info["current_player"]
        non_current = next(name for name in info["player_names"] if name != actor)

        payload = env.observation_payload(actor)
        assert payload["actor"] == actor
        assert payload["legal_actions"] == info["valid_actions"]
        assert payload["legal_action_descriptions"] == info["legal_action_descriptions"]
        assert isinstance(payload["is_road_building"], bool)
        assert isinstance(payload["free_roads_available"], int)
        assert isinstance(payload["current_discard_count"], int)
        assert "resources" in payload["players"][actor]
        assert "development_cards" in payload["players"][actor]
        assert "has_played_development_card_in_turn" in payload["players"][actor]
        assert "playable_development_cards" in payload["players"][actor]

        for opponent in set(info["player_names"]) - {actor}:
            opponent_payload = payload["players"][opponent]
            assert "resources" not in opponent_payload
            assert "development_cards" not in opponent_payload
            assert "has_played_development_card_in_turn" not in opponent_payload
            assert "playable_development_cards" not in opponent_payload
            assert "resource_card_count" in opponent_payload
            assert "development_card_count" in opponent_payload

        waiting_payload = env.observation_payload(non_current)
        assert waiting_payload["legal_actions"] == ()
        assert all(flag is False for flag in waiting_payload["action_mask"])
    finally:
        env.close()


def test_colonist_multiagent_env_structured_actions_match_legal_indices() -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    env = ColonistMultiAgentEnv(ColonistMultiAgentConfig(players=4))
    try:
        _, info = env.reset(seed=113)
        structured = info["structured_legal_actions"]

        assert len(structured) == len(info["valid_actions"])
        assert {action["index"] for action in structured} == set(info["valid_actions"])
        assert all("category" in action for action in structured)
        assert all("args" in action for action in structured)
        assert all("label" in action for action in structured)
        assert info["structured_legal_actions"] == env.structured_valid_actions(
            info["current_player"]
        )

        waiting_player = next(
            player for player in info["player_names"] if player != info["current_player"]
        )
        assert env.structured_valid_actions(waiting_player) == ()
    finally:
        env.close()


def test_colonist_multiagent_env_steps_structured_action() -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    env = ColonistMultiAgentEnv(ColonistMultiAgentConfig(players=4))
    try:
        _, info = env.reset(seed=114)
        action = info["structured_legal_actions"][0]

        _, rewards, terminated, truncated, next_info = env.step_structured_action(action)

        assert set(rewards) == {"BLUE", "RED", "ORANGE", "WHITE"}
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert next_info["replay_frame_count"] >= 2
    finally:
        env.close()


def test_colonist_multiagent_env_structures_trade_actions() -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    rng = random.Random(11)
    env = ColonistMultiAgentEnv(ColonistMultiAgentConfig(players=4))
    try:
        _, info = env.reset(seed=123)

        trade_action = None
        for _ in range(200):
            trade_action = next(
                (
                    action
                    for action in info["structured_legal_actions"]
                    if action["action_type"] == "offer_trade"
                ),
                None,
            )
            if trade_action is not None:
                break
            _, _, terminated, truncated, info = env.step(rng.choice(info["valid_actions"]))
            assert not (terminated or truncated)

        assert trade_action is not None
        assert trade_action["category"] == "trade"
        assert trade_action["args"]["trade_kind"] == "player_offer"
        assert trade_action["args"]["give"] or trade_action["args"]["want"]
        assert env.action_index_from_structured(trade_action) in info["valid_actions"]
    finally:
        env.close()


def test_colonist_multiagent_env_public_event_log_redacts_hidden_results() -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    env = ColonistMultiAgentEnv(ColonistMultiAgentConfig(players=4))
    try:
        _, _ = env.reset(seed=107)
        env._record_event(
            "board_action",
            actor="BLUE",
            payload={
                "action_index": 1,
                "action": {
                    "index": 1,
                    "action_type": "BUY_DEVELOPMENT_CARD",
                    "value": None,
                },
                "result": "VICTORY_POINT",
            },
        )
        env._record_event(
            "board_action",
            actor="BLUE",
            payload={
                "action_index": 2,
                "action": {
                    "index": 2,
                    "action_type": "DISCARD_RESOURCE",
                    "value": "ORE",
                },
                "result": "ORE",
            },
        )

        events = env.event_log()

        assert events[-2]["payload"]["result"] == "hidden_development_card"
        assert events[-1]["payload"]["action"]["value"] == "hidden_resource"
        assert events[-1]["payload"]["action"]["index"] is None
        assert events[-1]["payload"]["action_index"] is None
        assert events[-1]["payload"]["result"] == "hidden_resource"
    finally:
        env.close()


def test_discard_event_redaction_removes_resource_bearing_flat_action_id() -> None:
    """The discard catalog id is as private as the discarded value itself."""
    env = object.__new__(ColonistMultiAgentEnv)
    event = {
        "event_type": "board_action",
        "payload": {
            "action_index": 123,
            "action": {
                "index": 123,
                "action_type": "DISCARD_RESOURCE",
                "value": "ORE",
            },
            "result": "ORE",
        },
    }

    redacted = env._redact_event(event, actor="BLUE")

    assert redacted["payload"]["action_index"] is None
    assert redacted["payload"]["action"]["index"] is None
    assert redacted["payload"]["action"]["value"] == "hidden_resource"
    assert redacted["payload"]["result"] == "hidden_resource"
    # The caller-owned event remains reusable for another observer.
    assert event["payload"]["action_index"] == 123
    assert event["payload"]["action"]["index"] == 123


def test_colonist_multiagent_env_trade_panel_exposes_colonist_response_state() -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    env = ColonistMultiAgentEnv(ColonistMultiAgentConfig(players=4))
    try:
        _, _ = env.reset(seed=108)
        offer = env.propose_trade(
            actor="BLUE",
            give={"wood": 1},
            want={"kind": "open", "count": 1},
        )

        initial_panel = env.trade_panel("BLUE")
        initial_offer = initial_panel["offers"][0]
        assert initial_offer["eligible_responders"] == ("RED", "ORANGE", "WHITE")
        assert set(initial_offer["waiting_players"]) == {"RED", "ORANGE", "WHITE"}
        assert initial_offer["can_confirm"] is False

        env.respond_to_trade(offer["offer_id"], "accepted", actor="RED")
        env.respond_to_trade(offer["offer_id"], "rejected", actor="ORANGE")
        env.counter_trade(
            offer["offer_id"],
            actor="WHITE",
            give={"brick": 1},
            want={"wood": 1},
        )

        proposer_panel = env.trade_panel("BLUE")
        updated_offer = proposer_panel["offers"][0]

        assert updated_offer["responder_statuses"] == {
            "RED": "accepted",
            "ORANGE": "rejected",
            "WHITE": "countered",
        }
        assert updated_offer["accepted_players"] == ("RED",)
        assert updated_offer["rejected_players"] == ("ORANGE",)
        assert updated_offer["countered_players"] == ("WHITE",)
        assert updated_offer["can_confirm"] is True
        assert proposer_panel["open_offers"][1]["parent_offer_id"] == offer["offer_id"]

        red_panel = env.trade_panel("RED")
        assert red_panel["offers"][0]["can_accept"] is False
        assert red_panel["offers"][1]["can_accept"] is True
    finally:
        env.close()


def test_colonist_multiagent_env_resolves_open_offer_to_trade_action() -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    rng = random.Random(7)
    env = ColonistMultiAgentEnv(ColonistMultiAgentConfig(players=4))
    try:
        _, info = env.reset(seed=123)
        base_action_count = env._base_action_space_n
        trade_action = None

        for _ in range(200):
            trade_actions = [
                action for action in info["valid_actions"] if action >= base_action_count
            ]
            if trade_actions:
                trade_action = trade_actions[0]
                break
            _, _, terminated, truncated, info = env.step(rng.choice(info["valid_actions"]))
            assert not (terminated or truncated)

        assert trade_action is not None
        kind, trade_value = env._extended_actions[trade_action - base_action_count]
        assert kind == "offer_trade"

        give_exact = _freqdeck_to_spec(trade_value[:5])
        want_exact = _freqdeck_to_spec(trade_value[5:10])
        offer = env.propose_trade(
            give=give_exact,
            want={"kind": "open", "count": sum(trade_value[5:10])},
        )
        assert env.trade_action_for_offer(offer["offer_id"], want=want_exact) == trade_action

        _, _, _, _, next_info = env.step_negotiated_trade(
            offer["offer_id"],
            want=want_exact,
        )
        assert len(next_info["action_mask"]) == env.action_space.n
    finally:
        env.close()


def test_colonist_multiagent_env_records_simple_replay_events() -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    env = ColonistMultiAgentEnv(ColonistMultiAgentConfig(players=4))
    try:
        _, info = env.reset(seed=104)
        assert [event["event_type"] for event in info["event_log"]] == ["reset"]

        env.post_chat("ore for wheat?")
        env.propose_trade(give={"ore": 1}, want={"kind": "open", "count": 1})
        _, _, _, _, next_info = env.step(info["valid_actions"][0])

        event_types = [event["event_type"] for event in next_info["event_log"]]
        assert event_types[0] == "reset"
        assert "chat" in event_types
        assert "trade_proposal" in event_types
        assert event_types[-1] == "board_action"
    finally:
        env.close()


def test_colonist_multiagent_env_replay_trace_reconstructs_public_observations() -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    env = ColonistMultiAgentEnv(ColonistMultiAgentConfig(players=4))
    try:
        _, info = env.reset(seed=109)
        env.post_chat("ore for wheat?")
        env.propose_trade(give={"ore": 1}, want={"kind": "open", "count": 1})
        _, _, _, _, next_info = env.step(info["valid_actions"][0])

        trace = env.replay_trace()

        assert len(trace) == next_info["replay_frame_count"]
        assert [frame["frame_id"] for frame in trace] == list(range(1, len(trace) + 1))
        assert [frame["event"]["event_type"] for frame in trace][:3] == [
            "reset",
            "chat",
            "chat",
        ]
        assert "trade_proposal" in [frame["event"]["event_type"] for frame in trace]
        latest = trace[-1]
        assert set(latest["observations"]) == {"BLUE", "RED", "ORANGE", "WHITE"}
        assert set(latest["rewards"]) == {"BLUE", "RED", "ORANGE", "WHITE"}
        assert isinstance(latest["terminated"], bool)
        assert isinstance(latest["truncated"], bool)

        current_player = latest["observations"][next_info["current_player"]]
        assert current_player["legal_actions"] == next_info["valid_actions"]
        assert "event_log" not in current_player
        assert "trade_panel" in current_player

        non_current = next(
            player
            for player in latest["observations"]
            if player != next_info["current_player"]
        )
        assert latest["observations"][non_current]["legal_actions"] == ()
    finally:
        env.close()


def test_colonist_multiagent_env_replay_trace_redacts_hidden_event_results() -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    env = ColonistMultiAgentEnv(ColonistMultiAgentConfig(players=4))
    try:
        env.reset(seed=110)
        env._record_event(
            "board_action",
            actor="BLUE",
            payload={
                "action_index": 1,
                "action": {
                    "index": 1,
                    "action_type": "BUY_DEVELOPMENT_CARD",
                    "value": None,
                },
                "result": "VICTORY_POINT",
            },
        )

        assert env.replay_trace()[-1]["event"]["payload"]["result"] == (
            "hidden_development_card"
        )
    finally:
        env.close()


def test_colonist_multiagent_env_writes_replay_jsonl(tmp_path) -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    env = ColonistMultiAgentEnv(ColonistMultiAgentConfig(players=4))
    try:
        _, info = env.reset(seed=111)
        env.post_chat("ore for wheat?")
        env.propose_trade(give={"ore": 1}, want={"kind": "open", "count": 1})
        env.step(info["valid_actions"][0])

        replay_path = tmp_path / "game.jsonl"
        count = env.write_replay_jsonl(
            replay_path,
            metadata={"source": "unit-test"},
        )
        loaded = load_replay_jsonl(replay_path)

        assert count == len(env.replay_trace())
        assert len(loaded) == count
        assert loaded[0]["event"]["event_type"] == "reset"
        assert loaded[-1]["frame_id"] == count
        assert set(loaded[-1]["observations"]) == {"BLUE", "RED", "ORANGE", "WHITE"}
        assert isinstance(loaded[-1]["observations"]["BLUE"]["action_mask"], list)
    finally:
        env.close()


def test_colonist_multiagent_env_replay_jsonl_redacts_hidden_results(tmp_path) -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    env = ColonistMultiAgentEnv(ColonistMultiAgentConfig(players=4))
    try:
        env.reset(seed=112)
        env._record_event(
            "board_action",
            actor="BLUE",
            payload={
                "action_index": 1,
                "action": {
                    "index": 1,
                    "action_type": "BUY_DEVELOPMENT_CARD",
                    "value": None,
                },
                "result": "VICTORY_POINT",
            },
        )

        replay_path = tmp_path / "hidden.jsonl"
        env.write_replay_jsonl(replay_path)
        loaded = load_replay_jsonl(replay_path)

        assert loaded[-1]["event"]["payload"]["result"] == "hidden_development_card"
    finally:
        env.close()


def test_colonist_multiagent_env_targeted_trade_filters_non_target_accepts() -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    rng = random.Random(9)
    env = ColonistMultiAgentEnv(ColonistMultiAgentConfig(players=4))
    try:
        _, info = env.reset(seed=123)
        base_action_count = env._base_action_space_n
        trade_action = None

        for _ in range(200):
            trade_actions = [
                action for action in info["valid_actions"] if action >= base_action_count
            ]
            if trade_actions:
                trade_action = trade_actions[0]
                break
            _, _, terminated, truncated, info = env.step(rng.choice(info["valid_actions"]))
            assert not (terminated or truncated)

        assert trade_action is not None
        actor = info["current_player"]
        first_responder = next(
            color.name for color in env.game.state.colors if color.name != actor
        )
        target = next(
            player
            for player in info["player_names"]
            if player not in (actor, first_responder)
        )

        _, trade_value = env._extended_actions[trade_action - base_action_count]
        offer = env.propose_trade(
            target=target,
            give=_freqdeck_to_spec(trade_value[:5]),
            want={"kind": "open", "count": sum(trade_value[5:10])},
        )
        _, _, _, _, response_info = env.step_negotiated_trade(
            offer["offer_id"],
            want=_freqdeck_to_spec(trade_value[5:10]),
        )

        assert response_info["current_player"] == first_responder
        assert first_responder != target
        response_action_types = {
            env.describe_action(action)["action_type"]
            for action in response_info["valid_actions"]
            if action >= base_action_count
        }
        assert "reject_trade" in response_action_types
        assert "accept_trade" not in response_action_types
    finally:
        env.close()


def test_colonist_multiagent_env_random_playable_loop_smoke() -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    rng = random.Random(105)
    env = ColonistMultiAgentEnv(ColonistMultiAgentConfig(players=4))
    try:
        _, info = env.reset(seed=105)
        seen_players: set[str] = set()

        for _ in range(80):
            seen_players.add(info["current_player"])
            _, rewards, terminated, truncated, info = env.step(
                rng.choice(info["valid_actions"])
            )
            assert set(rewards) == {"BLUE", "RED", "ORANGE", "WHITE"}
            if terminated or truncated:
                break

        assert seen_players == {"BLUE", "RED", "ORANGE", "WHITE"}
    finally:
        env.close()


def _freqdeck_to_spec(freqdeck: tuple[int, int, int, int, int]) -> dict[str, int]:
    return {
        resource: count
        for resource, count in zip(RESOURCE_NAMES, freqdeck)
        if count > 0
    }
