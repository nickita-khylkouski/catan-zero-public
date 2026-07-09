from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PPOLossConfig:
    clip_ratio: float = 0.15
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0


def clipped_ppo_loss(
    *,
    log_probs,
    old_log_probs,
    advantages,
    values,
    returns,
    entropy,
    config: PPOLossConfig,
):
    import torch
    from torch import nn

    ratio = torch.exp(log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = torch.clamp(
        ratio,
        1.0 - config.clip_ratio,
        1.0 + config.clip_ratio,
    ) * advantages
    policy_loss = -torch.min(unclipped, clipped).mean()
    value_loss = nn.functional.mse_loss(values, returns)
    loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy.mean()
    with torch.no_grad():
        approx_kl = (old_log_probs - log_probs).mean()
        clip_fraction = (torch.abs(ratio - 1.0) > config.clip_ratio).float().mean()
    return loss, {
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "entropy": float(entropy.mean().item()),
        "approx_kl": float(approx_kl.item()),
        "clip_fraction": float(clip_fraction.item()),
    }
