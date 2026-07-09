from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from catan_zero.rl.self_play import Policy
from catan_zero.rl.torch_ppo import PPOTrajectory, TorchPPOPolicy, collect_ppo_episode
from catan_zero.rl.multiagent_env import ColonistMultiAgentConfig


@dataclass(frozen=True, slots=True)
class RolloutConfig:
    decisions_per_rank: int = 262_144
    max_decisions_per_game: int = 1000
    gamma: float = 0.997
    gae_lambda: float = 0.95
    seed: int = 1


def collect_rank_rollout(
    policy: TorchPPOPolicy,
    opponents: dict[str, Policy],
    *,
    env_config: ColonistMultiAgentConfig,
    rollout_config: RolloutConfig,
    rank: int = 0,
    training_seats: set[str] | None = None,
) -> list[PPOTrajectory]:
    rng = np.random.default_rng(rollout_config.seed + rank * 1_000_003)
    trajectories: list[PPOTrajectory] = []
    decisions = 0
    game_index = 0
    seats = training_seats or {"BLUE"}
    while decisions < rollout_config.decisions_per_rank:
        trajectory = collect_ppo_episode(
            policy,
            opponents,
            seed=int(rng.integers(2**31)),
            config=env_config,
            max_decisions=rollout_config.max_decisions_per_game,
            rng=rng,
            training_seats=seats,
            gamma=rollout_config.gamma,
            gae_lambda=rollout_config.gae_lambda,
        )
        trajectories.append(trajectory)
        decisions += len(trajectory.samples)
        game_index += 1
        if game_index > rollout_config.decisions_per_rank:
            break
    return trajectories
