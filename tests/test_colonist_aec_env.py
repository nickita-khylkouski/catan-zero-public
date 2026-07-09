import random

import pytest

from catan_zero.rl import ColonistAECEnv, ColonistMultiAgentConfig


def test_colonist_aec_env_reset_last_and_step() -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    env = ColonistAECEnv(ColonistMultiAgentConfig(players=4))
    try:
        env.reset(seed=200)

        assert set(env.possible_agents) == {"BLUE", "RED", "ORANGE", "WHITE"}
        assert set(env.agents) == set(env.possible_agents)
        observation, reward, terminated, truncated, info = env.last()

        assert observation is not None
        assert reward == 0.0
        assert terminated is False
        assert truncated is False
        assert info["is_current_player"] is True
        assert info["valid_actions"]
        assert len(info["action_mask"]) == env.action_space(env.agent_selection).n

        previous_agent = env.agent_selection
        env.step(info["valid_actions"][0])

        assert previous_agent in env.possible_agents
        assert env.agent_selection in env.possible_agents
        assert set(env.rewards) == set(env.possible_agents)
        assert set(env.terminations) == set(env.possible_agents)
        assert set(env.truncations) == set(env.possible_agents)
    finally:
        env.close()


def test_colonist_aec_env_non_current_agent_mask_is_empty() -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    env = ColonistAECEnv(ColonistMultiAgentConfig(players=4))
    try:
        env.reset(seed=201)
        current = env.agent_selection
        non_current = next(agent for agent in env.possible_agents if agent != current)

        assert env.infos[current]["valid_actions"]
        assert env.infos[non_current]["valid_actions"] == ()
        assert all(flag is False for flag in env.infos[non_current]["action_mask"])
    finally:
        env.close()


def test_colonist_aec_env_agent_iter_random_smoke() -> None:
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    rng = random.Random(202)
    env = ColonistAECEnv(ColonistMultiAgentConfig(players=4))
    try:
        env.reset(seed=202)
        seen: list[str] = []

        for agent in env.agent_iter(max_iter=40):
            observation, _, terminated, truncated, info = env.last()
            assert agent == env.agent_selection
            assert observation is not None
            if terminated or truncated:
                env.step(None)
                continue
            seen.append(agent)
            env.step(rng.choice(info["valid_actions"]))

        assert set(seen) == {"BLUE", "RED", "ORANGE", "WHITE"}
    finally:
        env.close()
