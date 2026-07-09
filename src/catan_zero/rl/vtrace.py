"""IMPALA V-trace off-policy correction for the distributed PPO learner.

Implements V-trace exactly per:

    Espeholt et al., 2018, "IMPALA: Scalable Distributed Deep-RL with
    Importance Weighted Actor-Learner Architectures", arXiv:1802.01561.

The semantics mirror DeepMind's reference implementation
``seed_rl/common/vtrace.py`` (and the earlier ``scalable_agent`` repo),
specialised to a *single* sequence over time ``T`` (the learner reshapes
[T, B] batches by looping / vmapping over the batch dimension, or by treating
each environment column independently).

Definitions (all arrays length ``T`` over the time axis; ``bootstrap_value``
is the scalar value estimate ``V(s_T)`` for the state *after* the last
transition):

    log_rhos     = target_log_probs - behavior_log_probs
    rhos         = exp(log_rhos)
    clipped_rhos = min(clip_rho_threshold, rhos)          # \\bar{rho}
    cs           = min(1.0, rhos)                          # the trace cutting c_t
    V_{t+1}      = concat(values[1:], [bootstrap_value])
    deltas_t     = clipped_rhos_t * (r_t + gamma_t V_{t+1} - V_t)   # \\delta_t V

    # V-trace target, computed by the standard reverse recursion
    #   v_t = V_t + \\delta_t V + gamma_t c_t (v_{t+1} - V_{t+1})
    # equivalently, with vs_minus_v_t = v_t - V_t:
    #   vs_minus_v_t = deltas_t + gamma_t c_t vs_minus_v_{t+1}
    acc = 0
    for t in reversed(range(T)):
        acc = deltas_t + gamma_t c_t acc
        vs_minus_v_t = acc
    vs = values + vs_minus_v

    # Policy-gradient advantages use the *next* V-trace target as the baseline
    # bootstrap, and a (separately clipped) importance weight.
    clipped_pg_rhos = min(clip_pg_rho_threshold, rhos)
    vs_{t+1}        = concat(vs[1:], [bootstrap_value])
    pg_advantages_t = clipped_pg_rhos_t * (r_t + gamma_t vs_{t+1} - V_t)

Episode boundaries: the caller passes ``discounts[t] = gamma * (1 - done[t])``.
A ``discounts[t] == 0`` zeroes out ``gamma_t c_t acc`` in the recursion, which
stops credit from propagating across the boundary, so a single flat batch may
concatenate several games safely.

Two implementations are provided that return *identical* numbers:

* :func:`vtrace_from_log_probs_np` -- pure numpy, unit-testable without torch.
* :func:`vtrace_from_log_probs`    -- torch (used by the learner).  No gradient
  flows through V-trace; it operates on detached tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "VTraceReturns",
    "vtrace_from_log_probs_np",
    "vtrace_from_log_probs",
]


@dataclass
class VTraceReturns:
    """V-trace outputs for a single sequence of length ``T``.

    Attributes
    ----------
    vs:
        Value-function targets ``v_s`` (shape ``[T]``).  Use these as the
        regression targets for the critic.
    pg_advantages:
        Policy-gradient advantages (shape ``[T]``).  Multiply by the target
        policy log-prob gradient in the actor loss.

    ``vs`` and ``pg_advantages`` are numpy arrays for the numpy entry point and
    torch tensors for the torch entry point.
    """

    vs: Any
    pg_advantages: Any


# ---------------------------------------------------------------------------
# numpy implementation (reference / unit-testable)
# ---------------------------------------------------------------------------
def vtrace_from_log_probs_np(
    behavior_log_probs: np.ndarray,
    target_log_probs: np.ndarray,
    discounts: np.ndarray,
    rewards: np.ndarray,
    values: np.ndarray,
    bootstrap_value: float,
    clip_rho_threshold: float = 1.0,
    clip_pg_rho_threshold: float = 1.0,
) -> VTraceReturns:
    """Compute V-trace targets and PG advantages with numpy.

    Parameters
    ----------
    behavior_log_probs, target_log_probs:
        ``log mu(a_t|s_t)`` and ``log pi(a_t|s_t)``, shape ``[T]``.  ``mu`` is
        the (stale) actor policy that generated the trajectory; ``pi`` is the
        current learner policy.
    discounts:
        ``gamma_t = gamma * (1 - done_t)``, shape ``[T]``.  Zero resets the
        recursion at an episode boundary.
    rewards:
        ``r_t``, shape ``[T]``.
    values:
        ``V(s_t)`` from the learner critic, shape ``[T]``.
    bootstrap_value:
        scalar ``V(s_T)`` for the state after the last transition.
    clip_rho_threshold:
        ``\\bar{rho}`` -- caps the importance weight used for the value target.
        Controls the fixed point of the value update (paper eq. for V-trace).
    clip_pg_rho_threshold:
        caps the importance weight used for the policy-gradient advantage.

    Returns
    -------
    VTraceReturns with numpy ``vs`` and ``pg_advantages`` of shape ``[T]``.
    """
    behavior_log_probs = np.asarray(behavior_log_probs, dtype=np.float64)
    target_log_probs = np.asarray(target_log_probs, dtype=np.float64)
    discounts = np.asarray(discounts, dtype=np.float64)
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    bootstrap_value = float(bootstrap_value)

    if not (
        behavior_log_probs.shape
        == target_log_probs.shape
        == discounts.shape
        == rewards.shape
        == values.shape
    ):
        raise ValueError(
            "vtrace inputs must share shape [T]; got "
            f"behavior={behavior_log_probs.shape}, target={target_log_probs.shape}, "
            f"discounts={discounts.shape}, rewards={rewards.shape}, "
            f"values={values.shape}"
        )
    if behavior_log_probs.ndim != 1:
        raise ValueError(
            f"vtrace_from_log_probs_np expects 1-D sequences [T]; got ndim="
            f"{behavior_log_probs.ndim}"
        )

    log_rhos = target_log_probs - behavior_log_probs
    rhos = np.exp(log_rhos)

    if clip_rho_threshold is not None:
        clipped_rhos = np.minimum(clip_rho_threshold, rhos)
    else:
        clipped_rhos = rhos
    cs = np.minimum(1.0, rhos)

    # values_{t+1} = [V_1, V_2, ..., V_{T-1}, bootstrap]
    values_t_plus_1 = np.concatenate([values[1:], np.array([bootstrap_value])])
    deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)

    # Reverse recursion for vs_minus_v.
    T = values.shape[0]
    vs_minus_v = np.zeros_like(values)
    acc = 0.0
    for t in reversed(range(T)):
        acc = deltas[t] + discounts[t] * cs[t] * acc
        vs_minus_v[t] = acc

    vs = values + vs_minus_v

    # vs_{t+1} = [v_1, ..., v_{T-1}, bootstrap]
    vs_t_plus_1 = np.concatenate([vs[1:], np.array([bootstrap_value])])
    if clip_pg_rho_threshold is not None:
        clipped_pg_rhos = np.minimum(clip_pg_rho_threshold, rhos)
    else:
        clipped_pg_rhos = rhos
    pg_advantages = clipped_pg_rhos * (rewards + discounts * vs_t_plus_1 - values)

    return VTraceReturns(vs=vs, pg_advantages=pg_advantages)


# ---------------------------------------------------------------------------
# torch implementation (learner)
# ---------------------------------------------------------------------------
def vtrace_from_log_probs(
    behavior_log_probs: Any,
    target_log_probs: Any,
    discounts: Any,
    rewards: Any,
    values: Any,
    bootstrap_value: Any,
    clip_rho_threshold: float = 1.0,
    clip_pg_rho_threshold: float = 1.0,
) -> VTraceReturns:
    """Compute V-trace targets and PG advantages with torch.

    Numerically identical to :func:`vtrace_from_log_probs_np`.  V-trace is an
    off-policy *target* construction, not a differentiable loss: the learner
    backprops through ``target_log_probs`` and ``values`` *elsewhere* (the PPO
    / actor-critic loss), treating ``vs`` and ``pg_advantages`` as constants.
    Accordingly this function operates under ``torch.no_grad`` on detached
    tensors and returns detached tensors.

    Inputs may be torch tensors, numpy arrays, or python sequences; 1-D
    tensors of shape ``[T]`` are expected (``bootstrap_value`` is a scalar).
    """
    import torch

    with torch.no_grad():
        def _as_tensor(x: Any) -> "torch.Tensor":
            if isinstance(x, torch.Tensor):
                return x.detach().to(dtype=torch.float64).reshape(-1)
            return torch.as_tensor(np.asarray(x, dtype=np.float64))

        behavior = _as_tensor(behavior_log_probs)
        target = _as_tensor(target_log_probs)
        disc = _as_tensor(discounts)
        rew = _as_tensor(rewards)
        vals = _as_tensor(values)

        if isinstance(bootstrap_value, torch.Tensor):
            boot = bootstrap_value.detach().to(dtype=torch.float64).reshape(()).clone()
        else:
            boot = torch.as_tensor(float(bootstrap_value), dtype=torch.float64)

        if not (
            behavior.shape == target.shape == disc.shape == rew.shape == vals.shape
        ):
            raise ValueError(
                "vtrace inputs must share shape [T]; got "
                f"behavior={tuple(behavior.shape)}, target={tuple(target.shape)}, "
                f"discounts={tuple(disc.shape)}, rewards={tuple(rew.shape)}, "
                f"values={tuple(vals.shape)}"
            )

        log_rhos = target - behavior
        rhos = torch.exp(log_rhos)

        if clip_rho_threshold is not None:
            clipped_rhos = torch.clamp(rhos, max=clip_rho_threshold)
        else:
            clipped_rhos = rhos
        cs = torch.clamp(rhos, max=1.0)

        values_t_plus_1 = torch.cat([vals[1:], boot.reshape(1)])
        deltas = clipped_rhos * (rew + disc * values_t_plus_1 - vals)

        T = vals.shape[0]
        vs_minus_v = torch.zeros_like(vals)
        acc = torch.zeros((), dtype=torch.float64)
        for t in range(T - 1, -1, -1):
            acc = deltas[t] + disc[t] * cs[t] * acc
            vs_minus_v[t] = acc

        vs = vals + vs_minus_v

        vs_t_plus_1 = torch.cat([vs[1:], boot.reshape(1)])
        if clip_pg_rho_threshold is not None:
            clipped_pg_rhos = torch.clamp(rhos, max=clip_pg_rho_threshold)
        else:
            clipped_pg_rhos = rhos
        pg_advantages = clipped_pg_rhos * (rew + disc * vs_t_plus_1 - vals)

        return VTraceReturns(vs=vs, pg_advantages=pg_advantages)
