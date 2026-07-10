"""Two-player Catan rollout and learner for the KLENT R&D arm.

The module is deliberately separate from PPO.  It samples the reverse-KL /
entropy improved policy, computes player-transition-aware lambda returns, and
fits policy, chosen-action Q, and the scalar value needed by search.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np

from catan_zero.rl.klent import KLENTConfig, catan_lambda_returns, loss as klent_loss
from catan_zero.rl.klent_actor import KLENTActorStep, sample_entity_policy_step
from catan_zero.rl.multiagent_env import ColonistMultiAgentConfig, ColonistMultiAgentEnv
from catan_zero.rl.self_play import StepSample
from catan_zero.rl.torch_ppo import _entity_graph_outputs

__all__ = [
    "KLENTTrajectory",
    "collect_trajectory",
    "update_entity_policy",
]


@dataclass(frozen=True, slots=True)
class KLENTTrajectory:
    steps: tuple[KLENTActorStep, ...]
    players: tuple[str, ...]
    rewards: tuple[float, ...]
    terminated: tuple[bool, ...]
    returns: tuple[float, ...]
    game_seed: int
    truncated: bool

    def validate(self) -> None:
        lengths = {
            len(self.steps),
            len(self.players),
            len(self.rewards),
            len(self.terminated),
            len(self.returns),
        }
        if lengths != {len(self.steps)} or not self.steps:
            raise ValueError("KLENT trajectory fields must be non-empty and aligned")
        if self.truncated and self.terminated[-1]:
            raise ValueError("a KLENT trajectory cannot be terminal and truncated")
        if not self.truncated and not self.terminated[-1]:
            raise ValueError("a complete KLENT trajectory must end at a terminal step")
        if not all(math.isfinite(float(value)) for value in (*self.rewards, *self.returns)):
            raise ValueError("KLENT trajectory rewards and returns must be finite")


def collect_trajectory(
    policy: Any,
    *,
    seed: int,
    env_config: ColonistMultiAgentConfig,
    config: KLENTConfig | None = None,
    max_decisions: int = 600,
) -> KLENTTrajectory:
    """Collect one on-policy two-player/no-trade trajectory.

    Both seats use the same policy, matching self-play policy optimization.
    ``max_player_trade_offers_per_turn=0`` is required because the first direct
    RL experiment is scoped to the repository's certified no-trade track.
    """

    if env_config.players != 2:
        raise ValueError("KLENT Catan rollout currently supports exactly two players")
    if env_config.max_player_trade_offers_per_turn != 0:
        raise ValueError("KLENT Catan rollout requires the no-player-trade track")
    if int(max_decisions) < 1:
        raise ValueError("max_decisions must be positive")

    cfg = config or KLENTConfig()
    cfg.validate()
    rng = np.random.default_rng(int(seed))
    env = ColonistMultiAgentEnv(env_config)
    steps: list[KLENTActorStep] = []
    players: list[str] = []
    rewards: list[float] = []
    terminal_flags: list[bool] = []
    try:
        _observations, info = env.reset(seed=int(seed))
        terminated = False
        truncated = False
        decisions = 0
        while not (terminated or truncated) and decisions < int(max_decisions):
            player = str(info["current_player"])
            step = sample_entity_policy_step(policy, env, info, rng, config=cfg)
            _observations, reward_map, terminated, truncated, info = env.step(step.action)
            steps.append(step)
            players.append(player)
            rewards.append(float(reward_map.get(player, 0.0)))
            terminal_flags.append(bool(terminated))
            decisions += 1

        time_limit = not terminated and decisions >= int(max_decisions)
        # Terminal takes precedence if an environment time limit fires on the
        # same transition; a terminal state has no bootstrap value.
        truncated = bool((truncated or time_limit) and not terminated)
        if not steps:
            raise RuntimeError("KLENT environment produced no decisions")
        player_to_id = {name: index for index, name in enumerate(env.player_names)}
        player_ids = np.asarray([player_to_id[player] for player in players], dtype=np.int64)
        expected_q = np.asarray([step.expected_q for step in steps], dtype=np.float32)
        reward_array = np.asarray(rewards, dtype=np.float32)
        terminal_array = np.asarray(terminal_flags, dtype=np.bool_)

        return_kwargs: dict[str, Any] = {}
        if truncated:
            bootstrap_player = str(info["current_player"])
            bootstrap_step = sample_entity_policy_step(
                policy, env, info, rng, config=cfg
            )
            return_kwargs = {
                "bootstrap_value": bootstrap_step.expected_q,
                "bootstrap_player_id": player_to_id[bootstrap_player],
            }
        returns = catan_lambda_returns(
            reward_array,
            expected_q,
            terminal_array,
            player_ids,
            config=cfg,
            **return_kwargs,
        ).detach().cpu().numpy()
        trajectory = KLENTTrajectory(
            steps=tuple(steps),
            players=tuple(players),
            rewards=tuple(float(value) for value in reward_array),
            terminated=tuple(bool(value) for value in terminal_array),
            returns=tuple(float(value) for value in returns),
            game_seed=int(seed),
            truncated=truncated,
        )
        trajectory.validate()
        return trajectory
    finally:
        env.close()


def _training_rows(
    trajectories: Iterable[KLENTTrajectory],
) -> tuple[list[StepSample], list[np.ndarray], np.ndarray]:
    samples: list[StepSample] = []
    targets: list[np.ndarray] = []
    returns: list[float] = []
    for trajectory in trajectories:
        trajectory.validate()
        for index, (step, player, q_return) in enumerate(
            zip(trajectory.steps, trajectory.players, trajectory.returns)
        ):
            samples.append(
                StepSample(
                    observation=np.zeros(0, dtype=np.float32),
                    valid_actions=tuple(int(action) for action in step.legal_action_ids),
                    action=int(step.action),
                    player=str(player),
                    action_context_features=step.action_context_table,
                    entity_features=step.entity_features,
                    decision_index=index,
                    teacher_name="klent-direct-selfplay",
                )
            )
            targets.append(np.asarray(step.policy_target, dtype=np.float32))
            returns.append(float(q_return))
    if not samples:
        raise ValueError("cannot update KLENT from zero trajectory rows")
    return samples, targets, np.asarray(returns, dtype=np.float32)


def update_entity_policy(
    policy: Any,
    trajectories: Iterable[KLENTTrajectory],
    optimizer: Any,
    *,
    config: KLENTConfig | None = None,
    epochs: int = 1,
    minibatch_size: int = 256,
    value_loss_weight: float = 0.25,
    gradient_clip_norm: float = 1.0,
    seed: int = 0,
) -> dict[str, float | int]:
    """Apply on-policy KLENT updates to an ``EntityGraphPolicy``."""

    import torch
    import torch.nn.functional as F

    cfg = config or KLENTConfig()
    cfg.validate()
    if int(epochs) < 1 or int(minibatch_size) < 1:
        raise ValueError("epochs and minibatch_size must be positive")
    if not math.isfinite(float(value_loss_weight)) or value_loss_weight < 0.0:
        raise ValueError("value_loss_weight must be finite and non-negative")
    samples, policy_targets, q_returns = _training_rows(trajectories)
    rng = np.random.default_rng(int(seed))
    totals = {"loss": 0.0, "policy_loss": 0.0, "q_loss": 0.0, "value_loss": 0.0}
    updates = 0
    rows_seen = 0
    policy.model.train()
    for _epoch in range(int(epochs)):
        indices = rng.permutation(len(samples))
        for start in range(0, len(indices), int(minibatch_size)):
            chosen_indices = indices[start : start + int(minibatch_size)]
            batch_samples = [samples[int(index)] for index in chosen_indices]
            max_legal = max(len(sample.valid_actions) for sample in batch_samples)
            target_array = np.zeros((len(batch_samples), max_legal), dtype=np.float32)
            action_columns = np.zeros(len(batch_samples), dtype=np.int64)
            legal_mask = np.zeros((len(batch_samples), max_legal), dtype=np.bool_)
            for row, index in enumerate(chosen_indices):
                sample = samples[int(index)]
                target = policy_targets[int(index)]
                legal_count = len(sample.valid_actions)
                if target.shape != (legal_count,):
                    raise ValueError("KLENT policy target does not align with legal actions")
                target_array[row, :legal_count] = target
                legal_mask[row, :legal_count] = True
                action_columns[row] = sample.valid_actions.index(sample.action)

            outputs = _entity_graph_outputs(policy, batch_samples, return_q=True)
            if "q_values" not in outputs:
                raise RuntimeError("KLENT learner requires q_values output")
            actions_t = torch.as_tensor(action_columns, dtype=torch.long, device=policy.device)
            targets_t = torch.as_tensor(target_array, dtype=torch.float32, device=policy.device)
            returns_t = torch.as_tensor(
                q_returns[chosen_indices], dtype=torch.float32, device=policy.device
            )
            mask_t = torch.as_tensor(legal_mask, dtype=torch.bool, device=policy.device)
            components = klent_loss(
                outputs["logits"],
                outputs["q_values"],
                actions_t,
                targets_t,
                returns_t,
                legal_mask=mask_t,
                config=cfg,
            )
            value_loss = F.mse_loss(outputs["value"].float(), returns_t)
            total_loss = components["loss"] + float(value_loss_weight) * value_loss
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            if float(gradient_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    policy.model.parameters(), float(gradient_clip_norm)
                )
            optimizer.step()

            batch_rows = len(batch_samples)
            totals["loss"] += float(total_loss.detach()) * batch_rows
            totals["policy_loss"] += float(components["policy_loss"].detach()) * batch_rows
            totals["q_loss"] += float(components["q_loss"].detach()) * batch_rows
            totals["value_loss"] += float(value_loss.detach()) * batch_rows
            rows_seen += batch_rows
            updates += 1
    policy.model.eval()
    return {
        "schema_version": "catan-zero-klent-update/v1",
        "rows": len(samples),
        "row_passes": rows_seen,
        "updates": updates,
        "epochs": int(epochs),
        "loss": totals["loss"] / rows_seen,
        "policy_loss": totals["policy_loss"] / rows_seen,
        "q_loss": totals["q_loss"] / rows_seen,
        "value_loss": totals["value_loss"] / rows_seen,
        "entropy_coefficient": float(cfg.entropy_coefficient),
        "reverse_kl_coefficient": float(cfg.reverse_kl_coefficient),
        "trace_horizon": float(cfg.trace_horizon),
        "value_loss_weight": float(value_loss_weight),
    }
