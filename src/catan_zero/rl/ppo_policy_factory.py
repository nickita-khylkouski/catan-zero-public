"""Shared policy loader for distributed PPO — used by BOTH the Modal actors and the learner so
they construct the policy identically.

v1 always WARM-STARTS PPO from a behavior-cloned checkpoint (the 35M policy), per
the research (AlphaStar-style BC warm-start). The KL-to-BC anchor is just a second, FROZEN copy
of the same checkpoint passed to ``ppo_update(..., ema_policy=<frozen bc>, ema_policy_kl_coef=β)``
— no edit to ``ppo_update`` needed.
"""
from __future__ import annotations

from typing import Any


def load_ppo_policy(checkpoint: str, *, architecture: str = "xdim_graph", device: str | None = None) -> Any:
    """Load the trainable PPO policy from a checkpoint.

    ``xdim_graph`` (the 35M model) is the default and the path we use for the real runs. The flat
    architectures fall back to ``create_ppo_policy`` + a state-dict load for smoke tests.
    """
    if architecture in ("entity_graph", "entity"):
        from catan_zero.rl.entity_token_policy import EntityGraphPolicy
        return EntityGraphPolicy.load(checkpoint, device=device)
    if architecture in ("xdim_graph", "graph"):
        from catan_zero.rl.xdim_lite_policy import XDimGraphPolicy
        return XDimGraphPolicy.load(checkpoint, device=device)
    if architecture in ("xdim_lite", "lite"):
        from catan_zero.rl.xdim_lite_policy import XDimLitePolicy
        return XDimLitePolicy.load(checkpoint, device=device)
    # flat / candidate fallback (smoke tests only)
    from catan_zero.rl.torch_ppo import create_ppo_policy
    policy = create_ppo_policy(architecture=architecture, device=device)
    import torch
    policy.model.load_state_dict(torch.load(checkpoint, map_location=device or "cpu"))
    return policy


def load_frozen_bc_anchor(checkpoint: str, *, architecture: str = "xdim_graph", device: str | None = None) -> Any:
    """Load a SEPARATE, frozen copy of the BC checkpoint to use as the KL anchor (the "magnet").

    Passed to ``ppo_update`` as ``ema_policy`` with ``ema_policy_kl_coef=β``; never updated, so the
    learner keeps π_θ close to π_BC (Cicero piKL / AlphaStar distillation). Eval mode + no grad.
    """
    anchor = load_ppo_policy(checkpoint, architecture=architecture, device=device)
    freeze_in_place(anchor)
    return anchor


def freeze_in_place(policy: Any) -> Any:
    """Set a policy's underlying module to eval + requires_grad=False (for frozen anchors/opponents)."""
    model = getattr(policy, "model", None)
    if model is not None:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    return policy
