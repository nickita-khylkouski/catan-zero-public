from __future__ import annotations

from typing import Any

import numpy as np

from catan_zero.rl.multiagent_env import ColonistMultiAgentEnv
from catan_zero.rl.torch_ppo import TorchPPOPolicy


class OpeningThenPolicy:
    """Use a specialist opening policy, then fall back to a main policy."""

    def __init__(
        self,
        *,
        opening_policy: TorchPPOPolicy,
        main_policy: TorchPPOPolicy,
        opening_prompts: tuple[str, ...] = (
            "BUILD_INITIAL_SETTLEMENT",
            "BUILD_INITIAL_ROAD",
        ),
    ) -> None:
        self.opening_policy = opening_policy
        self.main_policy = main_policy
        self.opening_prompts = opening_prompts
        self.name = f"opening_then_{main_policy.name}"
        self.architecture = (
            "graph_history_candidate"
            if opening_policy.architecture == "graph_history_candidate"
            else main_policy.architecture
        )

    def select_action(
        self,
        env: ColonistMultiAgentEnv,
        observation: np.ndarray,
        info: dict[str, Any],
        rng: np.random.Generator,
        *,
        training: bool = False,
    ) -> int:
        prompt = str(info.get("current_prompt", ""))
        if prompt in self.opening_prompts:
            return self.opening_policy.select_action(
                env,
                _resize_observation(observation, self.opening_policy.observation_size),
                info,
                rng,
                training=training,
            )
        return self.main_policy.select_action(
            env,
            _resize_observation(observation, self.main_policy.observation_size),
            info,
            rng,
            training=training,
        )


def _resize_observation(observation: np.ndarray, size: int) -> np.ndarray:
    value = np.asarray(observation)
    if value.shape[0] == size:
        return value
    if value.shape[0] > size:
        return value[:size]
    padded = np.zeros(size, dtype=value.dtype)
    padded[: value.shape[0]] = value
    return padded
