"""Shared policy loader for distributed PPO — used by BOTH the Modal actors and the learner so
they construct the policy identically.

v1 always WARM-STARTS PPO from a behavior-cloned checkpoint (the 35M policy), per
the research (AlphaStar-style BC warm-start). The KL-to-BC anchor is just a second, FROZEN copy
of the same checkpoint passed to ``ppo_update(..., ema_policy=<frozen bc>, ema_policy_kl_coef=β)``
— no edit to ``ppo_update`` needed.
"""
from __future__ import annotations

from typing import Any


CANONICAL_PPO_ARCHITECTURE = "entity_graph"


def require_canonical_ppo_architecture(architecture: str) -> str:
    """Fail closed unless the W7 entity-graph architecture is selected explicitly."""
    resolved = str(architecture).strip()
    if resolved != CANONICAL_PPO_ARCHITECTURE:
        raise ValueError(
            "canonical PPO requires architecture='entity_graph'; "
            f"legacy architecture {architecture!r} is not accepted"
        )
    return resolved


def load_ppo_policy(
    checkpoint: str,
    *,
    architecture: str = CANONICAL_PPO_ARCHITECTURE,
    device: str | None = None,
) -> Any:
    """Load the trainable PPO policy from a checkpoint.

    W7 has one production/R&D lane.  Legacy flat/xdim checkpoints must use their
    historical launchers; accepting them here makes a typo or omitted flag silently
    construct a different policy family.
    """
    require_canonical_ppo_architecture(architecture)
    from catan_zero.rl.entity_token_policy import EntityGraphPolicy

    return EntityGraphPolicy.load(checkpoint, device=device)


def load_frozen_bc_anchor(
    checkpoint: str,
    *,
    architecture: str = CANONICAL_PPO_ARCHITECTURE,
    device: str | None = None,
) -> Any:
    """Load a SEPARATE, frozen copy of the BC checkpoint to use as the KL anchor (the "magnet").

    Passed to ``ppo_update`` as ``ema_policy`` with ``ema_policy_kl_coef=β``; never updated, so the
    learner keeps π_θ close to π_BC (Cicero piKL / AlphaStar distillation). Eval mode + no grad.
    """
    anchor = load_ppo_policy(checkpoint, architecture=architecture, device=device)
    freeze_in_place(anchor)
    return anchor


def load_exact_parent_and_frozen_anchor(
    checkpoint: str,
    *,
    architecture: str = CANONICAL_PPO_ARCHITECTURE,
    device: str | None = None,
) -> tuple[Any, Any]:
    """Load an exact trainable parent plus an independent frozen KL anchor.

    The equality check makes cold-start checkpoint binding executable instead of
    relying on two loader calls being configured the same way.
    """
    import torch

    parent = load_ppo_policy(checkpoint, architecture=architecture, device=device)
    anchor = load_frozen_bc_anchor(checkpoint, architecture=architecture, device=device)
    if parent is anchor or parent.model is anchor.model:
        raise RuntimeError("PPO parent and KL anchor must be independent objects")
    parent_state = parent.model.state_dict()
    anchor_state = anchor.model.state_dict()
    if parent_state.keys() != anchor_state.keys() or any(
        not torch.equal(parent_state[name], anchor_state[name]) for name in parent_state
    ):
        raise RuntimeError("PPO parent and frozen anchor did not load identical checkpoint state")
    if any(parameter.requires_grad for parameter in anchor.model.parameters()):
        raise RuntimeError("PPO KL anchor is not fully frozen")
    return parent, anchor


def freeze_in_place(policy: Any) -> Any:
    """Set a policy's underlying module to eval + requires_grad=False (for frozen anchors/opponents)."""
    model = getattr(policy, "model", None)
    if model is not None:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    return policy
