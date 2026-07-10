"""KLENT-style direct self-play targets for two-player Catan R&D.

This is an original PyTorch implementation of the update equations from Ota
et al., *Revisiting Regularized Policy Optimization for Stable and Efficient
Reinforcement Learning in Two-Player Games* (ICML 2026):
https://arxiv.org/abs/2602.10894

It supplies the algorithmic pieces needed by an actor/learner without coupling
them to PPO: reverse-KL plus entropy policy improvement, alternating-player
lambda returns, and policy/action-Q losses.  It is restricted to the current
two-player zero-sum research track.  Four-player Catan needs vector returns and
must not reuse ``gamma=-1``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True, slots=True)
class KLENTConfig:
    entropy_coefficient: float = 0.03
    reverse_kl_coefficient: float = 0.1
    trace_horizon: float = 8.0
    q_loss_weight: float = 1.0

    def validate(self) -> None:
        alpha = float(self.entropy_coefficient)
        beta = float(self.reverse_kl_coefficient)
        tau = float(self.trace_horizon)
        q_weight = float(self.q_loss_weight)
        if not math.isfinite(alpha) or alpha < 0.0:
            raise ValueError("entropy_coefficient must be finite and non-negative")
        if not math.isfinite(beta) or beta < 0.0:
            raise ValueError("reverse_kl_coefficient must be finite and non-negative")
        if alpha + beta <= 0.0:
            raise ValueError("entropy and reverse-KL coefficients cannot both be zero")
        if not math.isfinite(tau) or tau <= 0.0:
            raise ValueError("trace_horizon must be finite and positive")
        if not math.isfinite(q_weight) or q_weight < 0.0:
            raise ValueError("q_loss_weight must be finite and non-negative")

    @property
    def trace_lambda(self) -> float:
        self.validate()
        return math.exp(-1.0 / float(self.trace_horizon))


def improved_policy(
    policy_logits: Any,
    action_q_values: Any,
    legal_mask: Any,
    config: KLENTConfig | None = None,
):
    """Return the KLENT policy target over legal actions.

    ``policy_logits`` and ``action_q_values`` must have the same ``[..., A]``
    shape.  The returned tensor keeps gradients, allowing analytic tests, but
    actors and learners should detach it before using it as a fixed target.
    """

    import torch

    cfg = config or KLENTConfig()
    cfg.validate()
    logits = torch.as_tensor(policy_logits)
    q_values = torch.as_tensor(action_q_values, device=logits.device)
    mask = torch.as_tensor(legal_mask, device=logits.device, dtype=torch.bool)
    if logits.shape != q_values.shape or logits.shape != mask.shape:
        raise ValueError(
            "policy_logits, action_q_values, and legal_mask must have identical shapes"
        )
    if logits.ndim < 1 or logits.shape[-1] < 1:
        raise ValueError("KLENT policy inputs must have a non-empty action dimension")
    if not bool(mask.any(dim=-1).all()):
        raise ValueError("every KLENT policy row must contain a legal action")
    denominator = float(cfg.entropy_coefficient + cfg.reverse_kl_coefficient)
    target_logits = (
        float(cfg.reverse_kl_coefficient) * logits + q_values
    ) / denominator
    target_logits = target_logits.masked_fill(~mask, float("-inf"))
    return torch.softmax(target_logits.float(), dim=-1).to(logits.dtype)


def alternating_lambda_returns(
    rewards: Any,
    state_values: Any,
    terminated: Any,
    *,
    config: KLENTConfig | None = None,
    bootstrap_value: float | Any | None = None,
):
    """Compute two-player zero-sum lambda returns with a sign flip per ply.

    Inputs use ``[T]`` or ``[T,B,...]`` shape.  ``state_values[t]`` is the
    improved-policy expectation of action Q at state ``t``.  A nonterminal
    final row requires ``bootstrap_value``; silently treating truncation as a
    draw would bias the learner.
    """

    import torch

    cfg = config or KLENTConfig()
    cfg.validate()
    reward_t = torch.as_tensor(rewards)
    value_t = torch.as_tensor(state_values, device=reward_t.device, dtype=reward_t.dtype)
    terminal_t = torch.as_tensor(terminated, device=reward_t.device, dtype=torch.bool)
    if reward_t.shape != value_t.shape or reward_t.shape != terminal_t.shape:
        raise ValueError("rewards, state_values, and terminated must have identical shapes")
    if reward_t.ndim < 1 or reward_t.shape[0] < 1:
        raise ValueError("lambda-return inputs must contain at least one timestep")
    if bootstrap_value is None:
        if not bool(terminal_t[-1].all()):
            raise ValueError("bootstrap_value is required for a truncated trajectory")
        carry = torch.zeros_like(reward_t[-1])
    else:
        carry = torch.as_tensor(
            bootstrap_value, device=reward_t.device, dtype=reward_t.dtype
        ).expand_as(reward_t[-1])

    trace_lambda = float(cfg.trace_lambda)
    result = torch.empty_like(reward_t)
    for index in range(reward_t.shape[0] - 1, -1, -1):
        next_value = value_t[index + 1] if index + 1 < reward_t.shape[0] else carry
        continued = reward_t[index] - (
            (1.0 - trace_lambda) * next_value + trace_lambda * carry
        )
        carry = torch.where(terminal_t[index], reward_t[index], continued)
        result[index] = carry
    return result


def catan_lambda_returns(
    rewards: Any,
    state_values: Any,
    terminated: Any,
    player_ids: Any,
    *,
    config: KLENTConfig | None = None,
    bootstrap_value: float | Any | None = None,
    bootstrap_player_id: int | Any | None = None,
):
    """Compute lambda returns across Catan's non-alternating decision prompts.

    KLENT's board-game implementation uses ``gamma=-1`` because players swap
    after every action.  Catan can give one player consecutive decisions
    (roll, robber, build, road-building) and can prompt the opponent to discard
    out of turn.  This version applies ``+1`` when the next decision belongs to
    the same player and ``-1`` when perspective changes.
    """

    import torch

    cfg = config or KLENTConfig()
    cfg.validate()
    reward_t = torch.as_tensor(rewards)
    value_t = torch.as_tensor(state_values, device=reward_t.device, dtype=reward_t.dtype)
    terminal_t = torch.as_tensor(terminated, device=reward_t.device, dtype=torch.bool)
    player_t = torch.as_tensor(player_ids, device=reward_t.device)
    if reward_t.shape != value_t.shape or reward_t.shape != terminal_t.shape:
        raise ValueError("rewards, state_values, and terminated must have identical shapes")
    if player_t.shape != reward_t.shape:
        raise ValueError("player_ids must match the return tensor shape")
    if reward_t.ndim != 1 or reward_t.numel() < 1:
        raise ValueError("Catan lambda returns currently expect one [time] trajectory")
    if bootstrap_value is None:
        if not bool(terminal_t[-1]):
            raise ValueError("bootstrap_value is required for a truncated trajectory")
        carry = torch.zeros_like(reward_t[-1])
        next_player = player_t[-1]
    else:
        if bootstrap_player_id is None:
            raise ValueError("bootstrap_player_id is required with bootstrap_value")
        carry = torch.as_tensor(
            bootstrap_value, device=reward_t.device, dtype=reward_t.dtype
        )
        next_player = torch.as_tensor(bootstrap_player_id, device=reward_t.device)

    trace_lambda = float(cfg.trace_lambda)
    result = torch.empty_like(reward_t)
    for index in range(reward_t.shape[0] - 1, -1, -1):
        if index + 1 < reward_t.shape[0]:
            next_value = value_t[index + 1]
            next_player = player_t[index + 1]
        else:
            next_value = carry
        perspective_sign = torch.where(
            player_t[index] == next_player,
            torch.ones_like(reward_t[index]),
            -torch.ones_like(reward_t[index]),
        )
        continued = reward_t[index] + perspective_sign * (
            (1.0 - trace_lambda) * next_value + trace_lambda * carry
        )
        carry = torch.where(terminal_t[index], reward_t[index], continued)
        result[index] = carry
        next_player = player_t[index]
    return result


def loss(
    policy_logits: Any,
    action_q_values: Any,
    actions: Any,
    policy_targets: Any,
    q_targets: Any,
    *,
    legal_mask: Any | None = None,
    config: KLENTConfig | None = None,
) -> dict[str, Any]:
    """Return policy CE, chosen-action Q MSE, total loss, and telemetry."""

    import torch
    import torch.nn.functional as F

    cfg = config or KLENTConfig()
    cfg.validate()
    logits = torch.as_tensor(policy_logits)
    q_values = torch.as_tensor(action_q_values, device=logits.device)
    targets = torch.as_tensor(policy_targets, device=logits.device, dtype=logits.dtype)
    chosen = torch.as_tensor(actions, device=logits.device, dtype=torch.long)
    returns = torch.as_tensor(q_targets, device=logits.device, dtype=logits.dtype)
    if logits.shape != q_values.shape or logits.shape != targets.shape:
        raise ValueError("policy logits, action Q, and policy targets must share shape")
    if logits.ndim != 2:
        raise ValueError("KLENT loss expects [batch, action] policy tensors")
    if chosen.shape != (logits.shape[0],) or returns.shape != chosen.shape:
        raise ValueError("actions and q_targets must have shape [batch]")
    if legal_mask is not None:
        mask = torch.as_tensor(legal_mask, device=logits.device, dtype=torch.bool)
        if mask.shape != logits.shape:
            raise ValueError("legal_mask must match policy tensor shape")
        if not bool(mask.gather(1, chosen[:, None]).all()):
            raise ValueError("KLENT batch contains an illegal chosen action")
        if bool((targets.masked_select(~mask).abs() > 1.0e-8).any()):
            raise ValueError("KLENT policy target assigns mass to an illegal action")
        logits = logits.masked_fill(~mask, float("-inf"))
        targets = targets.masked_fill(~mask, 0.0)
    target_mass = targets.sum(dim=-1)
    if not torch.isfinite(target_mass).all() or not torch.allclose(
        target_mass, torch.ones_like(target_mass), atol=1.0e-5, rtol=1.0e-5
    ):
        raise ValueError("each KLENT policy target must be normalized")

    log_policy = F.log_softmax(logits.float(), dim=-1)
    policy_terms = torch.where(
        targets.float() > 0.0,
        targets.float() * log_policy,
        torch.zeros_like(log_policy),
    )
    policy_loss = -policy_terms.sum(dim=-1).mean()
    chosen_q = q_values.gather(1, chosen[:, None]).squeeze(1)
    q_loss = F.mse_loss(chosen_q.float(), returns.float())
    total = policy_loss + float(cfg.q_loss_weight) * q_loss
    entropy = -(targets.float() * targets.float().clamp_min(1.0e-12).log()).sum(-1).mean()
    return {
        "loss": total,
        "policy_loss": policy_loss,
        "q_loss": q_loss,
        "target_entropy": entropy,
    }
