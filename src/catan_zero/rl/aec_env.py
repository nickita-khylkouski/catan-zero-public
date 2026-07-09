from __future__ import annotations

from typing import Any, Iterator

from catan_zero.rl.multiagent_env import ColonistMultiAgentConfig, ColonistMultiAgentEnv


class ColonistAECEnv:
    """PettingZoo-style AEC adapter for `ColonistMultiAgentEnv`.

    This intentionally avoids a hard PettingZoo dependency while matching the
    core AEC interaction shape used by turn-based self-play trainers.
    """

    metadata = {"name": "catan_zero_colonist_v0", "is_parallelizable": False}

    def __init__(self, config: ColonistMultiAgentConfig | None = None) -> None:
        self.env = ColonistMultiAgentEnv(config)
        self.possible_agents = list(self.env.player_names)
        self.agents = list(self.possible_agents)
        self.agent_selection = self.possible_agents[0]
        self.rewards = {agent: 0.0 for agent in self.possible_agents}
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.infos = {agent: {} for agent in self.possible_agents}
        self._observations: dict[str, Any] = {}
        self._last_info: dict[str, Any] = {}

    @property
    def action_spaces(self) -> dict[str, Any]:
        return {agent: self.env.action_space for agent in self.possible_agents}

    @property
    def observation_spaces(self) -> dict[str, Any]:
        return {agent: self.env.observation_space for agent in self.possible_agents}

    def action_space(self, agent: str) -> Any:
        self._assert_agent(agent)
        return self.env.action_space

    def observation_space(self, agent: str) -> Any:
        self._assert_agent(agent)
        return self.env.observation_space

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        self._observations, self._last_info = self.env.reset(seed=seed, options=options)
        self.possible_agents = list(self.env.player_names)
        self.agents = list(self.possible_agents)
        self.agent_selection = self._last_info["current_player"]
        self.rewards = {agent: 0.0 for agent in self.possible_agents}
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self._refresh_infos()

    def observe(self, agent: str) -> Any:
        self._assert_agent(agent)
        return self._observations[agent]

    def last(self, observe: bool = True) -> tuple[Any | None, float, bool, bool, dict[str, Any]]:
        agent = self.agent_selection
        observation = self.observe(agent) if observe and agent in self.agents else None
        return (
            observation,
            self.rewards.get(agent, 0.0),
            self.terminations.get(agent, False),
            self.truncations.get(agent, False),
            self.infos.get(agent, {}),
        )

    def step(self, action: int | None) -> None:
        if not self.agents:
            return
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return
        if action is None:
            raise ValueError("action cannot be None for a live agent")

        self._observations, rewards, terminated, truncated, self._last_info = self.env.step(action)
        self.rewards = {name: float(rewards.get(name, 0.0)) for name in self.possible_agents}
        self.terminations = {name: bool(terminated) for name in self.possible_agents}
        self.truncations = {name: bool(truncated) for name in self.possible_agents}
        if terminated or truncated:
            self.agents = []
        else:
            self.agent_selection = self._last_info["current_player"]
        self._refresh_infos()

    def agent_iter(self, max_iter: int | None = None) -> Iterator[str]:
        count = 0
        while self.agents and (max_iter is None or count < max_iter):
            yield self.agent_selection
            count += 1

    def close(self) -> None:
        self.env.close()

    def _refresh_infos(self) -> None:
        self.infos = {
            agent: {
                **self._last_info,
                "agent": agent,
                "is_current_player": agent == self._last_info.get("current_player"),
                "valid_actions": self.env.valid_actions(agent),
                "action_mask": self.env.action_mask(agent),
            }
            for agent in self.possible_agents
        }

    def _was_dead_step(self, action: int | None) -> None:
        if action is not None:
            raise ValueError("dead agents only accept None actions")
        self.agents = [
            agent
            for agent in self.agents
            if not (self.terminations[agent] or self.truncations[agent])
        ]
        if self.agents:
            self.agent_selection = self.agents[0]

    def _assert_agent(self, agent: str) -> None:
        if agent not in self.possible_agents:
            raise ValueError(f"unknown agent: {agent}")
