"""Entity-policy actor adapter for KLENT-style direct self-play."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from catan_zero.rl.action_features import build_action_context_feature_table
from catan_zero.rl.klent import KLENTConfig, improved_policy

__all__ = ["KLENTActorStep", "sample_entity_policy_step"]


@dataclass(frozen=True, slots=True)
class KLENTActorStep:
    action: int
    action_column: int
    behavior_log_probability: float
    value: float
    expected_q: float
    chosen_q: float
    legal_action_ids: np.ndarray
    policy_target: np.ndarray
    action_q_values: np.ndarray
    legal_action_context: np.ndarray
    action_context_table: np.ndarray
    entity_features: dict[str, np.ndarray]


def sample_entity_policy_step(
    policy: Any,
    env: Any,
    info: dict[str, Any],
    rng: np.random.Generator,
    *,
    config: KLENTConfig | None = None,
) -> KLENTActorStep:
    """Sample one legal action from the current KLENT improved policy.

    The adapter records every quantity needed for a later learner update.  It
    fails if the policy does not emit an action-Q vector; silently substituting
    the scalar value would no longer implement KLENT.
    """

    import torch

    legal_actions = tuple(int(action) for action in info.get("valid_actions", ()))
    if not legal_actions:
        raise ValueError("KLENT actor received no legal actions")
    action_context_table = np.asarray(
        build_action_context_feature_table(env, info), dtype=np.float32
    )
    with torch.no_grad():
        outputs, entity, legal_context = policy._legal_outputs_from_env(
            env,
            info,
            legal_actions,
            return_q=True,
        )
        logits = outputs["logits"].squeeze(0)
        q_values = outputs.get("q_values")
        if q_values is None:
            raise RuntimeError("KLENT requires a trained per-action q_values head")
        q_values = q_values.squeeze(0)
        if logits.shape != q_values.shape or logits.numel() != len(legal_actions):
            raise RuntimeError("KLENT policy/Q output does not align with legal actions")
        target = improved_policy(
            logits.unsqueeze(0),
            q_values.unsqueeze(0),
            torch.ones_like(logits, dtype=torch.bool).unsqueeze(0),
            config,
        ).squeeze(0)
        target_np = target.detach().float().cpu().numpy().astype(np.float64)
        target_mass = float(target_np.sum(dtype=np.float64))
        if not np.isfinite(target_np).all() or not np.isfinite(target_mass) or target_mass <= 0.0:
            raise RuntimeError("KLENT actor produced an invalid policy target")
        # NumPy's categorical sampler checks normalization more tightly than a
        # float32 softmax guarantees. Normalize once in float64 at the actor
        # boundary; this does not change the model target within float32 error.
        target_np /= target_mass
        column = int(rng.choice(len(legal_actions), p=target_np))
        q_np = q_values.detach().float().cpu().numpy().astype(np.float32)
        value = float(outputs["value"].reshape(-1)[0].item())

    entity_copy = {
        key: np.asarray(value).copy()
        for key, value in entity.items()
        if key != "schema"
    }
    return KLENTActorStep(
        action=legal_actions[column],
        action_column=column,
        behavior_log_probability=float(np.log(max(target_np[column], 1.0e-30))),
        value=value,
        expected_q=float(np.dot(target_np, q_np.astype(np.float64))),
        chosen_q=float(q_np[column]),
        legal_action_ids=np.asarray(legal_actions, dtype=np.int64),
        policy_target=target_np.astype(np.float32),
        action_q_values=q_np,
        legal_action_context=np.asarray(legal_context, dtype=np.float32).copy(),
        action_context_table=action_context_table.copy(),
        entity_features=entity_copy,
    )
