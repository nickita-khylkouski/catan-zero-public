import pytest
import random

from catan_zero.rl import CatanZeroGymConfig, CatanZeroGymEnv


RESOURCE_NAMES = ("wood", "brick", "sheep", "wheat", "ore")


def test_catan_zero_gym_env_reset_and_mask() -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    env = CatanZeroGymEnv(CatanZeroGymConfig(players=4, representation="vector"))
    try:
        observation, info = env.reset(seed=123)
        mask = info["action_mask"]

        assert observation is not None
        assert len(mask) == env.action_space.n
        assert sum(mask) == len(info["valid_actions"])
        assert set(info["valid_actions"]) == {idx for idx, is_valid in enumerate(mask) if is_valid}
    finally:
        env.close()


def test_catan_zero_gym_env_exposes_structured_player_trades() -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    rng = random.Random(7)
    env = CatanZeroGymEnv(
        CatanZeroGymConfig(
            players=4,
            representation="vector",
            enable_player_trading=True,
        )
    )
    try:
        _, info = env.reset(seed=123)
        base_action_count = env._base_action_space_n
        trade_actions: list[int] = []

        for _ in range(100):
            trade_actions = [
                action for action in info["valid_actions"] if action >= base_action_count
            ]
            if trade_actions:
                break
            _, _, terminated, truncated, info = env.step(rng.choice(info["valid_actions"]))
            assert not (terminated or truncated)

        assert trade_actions
        observation, reward, terminated, truncated, next_info = env.step(trade_actions[0])

        assert observation is not None
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert len(next_info["action_mask"]) == env.action_space.n
        assert set(next_info["valid_actions"]) == {
            idx for idx, is_valid in enumerate(next_info["action_mask"]) if is_valid
        }
    finally:
        env.close()


def test_catan_zero_gym_env_valid_step() -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    env = CatanZeroGymEnv(CatanZeroGymConfig(players=4, representation="vector"))
    try:
        _, info = env.reset(seed=456)
        action = info["valid_actions"][0]
        observation, reward, terminated, truncated, next_info = env.step(action)

        assert observation is not None
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert len(next_info["action_mask"]) == env.action_space.n
    finally:
        env.close()


def test_catan_zero_gym_env_chat_is_public_side_channel() -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    env = CatanZeroGymEnv(
        CatanZeroGymConfig(
            players=4,
            representation="vector",
            max_chat_messages_per_turn=2,
        )
    )
    try:
        _, info = env.reset(seed=789)
        valid_actions_before = info["valid_actions"]
        mask_before = info["action_mask"]

        free_text = env.post_chat("  ore for wheat?  ")
        template_text = env.post_chat_template(
            "counteroffer",
            {"give": "ore", "want": "wheat"},
        )

        assert free_text["text"] == "ore for wheat?"
        assert free_text["intent"] == "free_text"
        assert template_text["intent"] == "counteroffer"
        assert len(env.chat.log()) == 2
        assert env.valid_actions() == valid_actions_before
        assert env.action_mask() == mask_before

        _, updated_info = env.reset(seed=789)
        assert updated_info["chat_log"] == ()
        assert updated_info["valid_actions"] == valid_actions_before
        assert updated_info["action_mask"] == mask_before
    finally:
        env.close()


def test_catan_zero_gym_env_chat_rate_limit() -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    env = CatanZeroGymEnv(
        CatanZeroGymConfig(
            players=4,
            representation="vector",
            max_chat_messages_per_turn=1,
        )
    )
    try:
        _, info = env.reset(seed=790)
        assert info["chat_messages_remaining"] == 1

        env.post_chat_template("trade_request", {"give": "brick", "want": "ore"})
        assert env.chat.remaining_messages("BLUE", env._current_turn_key()) == 0
        with pytest.raises(ValueError):
            env.post_chat("second message")
    finally:
        env.close()


def test_catan_zero_gym_env_open_and_wildcard_negotiation() -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    env = CatanZeroGymEnv(
        CatanZeroGymConfig(
            players=4,
            representation="vector",
        )
    )
    try:
        _, info = env.reset(seed=791)
        valid_actions_before = info["valid_actions"]

        offer = env.propose_trade(
            give={"wood": 1},
            want={"kind": "open", "count": 1},
        )
        response = env.respond_to_trade(offer["offer_id"], "accepted", actor="RED")
        counter = env.counter_trade(
            offer["offer_id"],
            actor="ORANGE",
            target="BLUE",
            give={"kind": "wildcard", "options": ["ore", "wheat"], "count": 1},
            want={"brick": 1},
        )

        assert offer["give"]["kind"] == "exact"
        assert offer["want"] == {
            "kind": "open",
            "resources": {},
            "options": (),
            "count": 1,
        }
        assert response["responses"]["RED"] == "accepted"
        assert counter["parent_offer_id"] == offer["offer_id"]
        assert counter["give"]["kind"] == "wildcard"
        assert env.valid_actions() == valid_actions_before

        _, _, _, _, next_info = env.step(valid_actions_before[0])
        assert len(next_info["negotiation_offers"]) == 2
        assert len(next_info["chat_log"]) == 2
    finally:
        env.close()


def test_catan_zero_gym_env_resolves_negotiation_to_board_trade() -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    rng = random.Random(7)
    env = CatanZeroGymEnv(
        CatanZeroGymConfig(
            players=4,
            representation="vector",
            enable_player_trading=True,
        )
    )
    try:
        _, info = env.reset(seed=123)
        base_action_count = env._base_action_space_n
        trade_action: int | None = None

        for _ in range(100):
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

        assert env.trade_action_for_offer(
            offer["offer_id"],
            want=want_exact,
        ) == trade_action

        observation, reward, terminated, truncated, next_info = env.step_negotiated_trade(
            offer["offer_id"],
            want=want_exact,
        )
        assert observation is not None
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert len(next_info["action_mask"]) == env.action_space.n
    finally:
        env.close()


def test_catan_zero_gym_env_exposes_colonist_timer_info() -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    env = CatanZeroGymEnv(
        CatanZeroGymConfig(
            players=4,
            representation="vector",
            timer_profile="fast",
        )
    )
    try:
        _, info = env.reset(seed=792)
        timer = info["timer"]

        assert timer["enabled"] is True
        assert timer["profile"] == "fast"
        assert timer["phase"] == "initial_build"
        assert timer["budget_seconds"] == 120
        assert timer["timeout_action"] in info["valid_actions"]
        assert timer["timeout_action_description"] is not None
    finally:
        env.close()


def test_catan_zero_gym_env_step_timeout_uses_valid_fallback() -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    env = CatanZeroGymEnv(CatanZeroGymConfig(players=4, representation="vector"))
    try:
        _, info = env.reset(seed=793)
        timeout_action = info["timer"]["timeout_action"]
        observation, reward, terminated, truncated, next_info = env.step_timeout()

        assert timeout_action in info["valid_actions"]
        assert observation is not None
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert len(next_info["action_mask"]) == env.action_space.n
    finally:
        env.close()


def _freqdeck_to_spec(freqdeck: tuple[int, int, int, int, int]) -> dict[str, int]:
    return {
        resource: count
        for resource, count in zip(RESOURCE_NAMES, freqdeck)
        if count > 0
    }
