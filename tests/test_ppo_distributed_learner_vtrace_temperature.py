from __future__ import annotations

import argparse

import numpy as np

from catan_zero.rl.self_play import StepSample
from catan_zero.rl.torch_ppo import (
    TorchPPOPolicy,
    _masked_logits,
    _policy_observation_array,
)
from tools.ppo_distributed_learner import _recompute_target_logp_and_values_batched


def _make_policy() -> TorchPPOPolicy:
    return TorchPPOPolicy(4, 5, hidden_size=8, seed=7, architecture="flat")


def _make_trajectory(n: int = 6) -> argparse.Namespace:
    rng = np.random.default_rng(0)
    samples = [
        StepSample(
            observation=rng.normal(size=4).astype(np.float32),
            valid_actions=(0, 1, 2, 3, 4),
            action=int(rng.integers(0, 5)),
            player="BLUE",
        )
        for _ in range(n)
    ]
    return argparse.Namespace(samples=samples)


def test_recompute_target_logp_scales_by_behavior_temperature() -> None:
    policy = _make_policy()
    trajectory = _make_trajectory()

    raw_logp, raw_values = _recompute_target_logp_and_values_batched(
        policy, [trajectory], forward_chunk=4, behavior_temperature=1.0
    )
    scaled_logp, scaled_values = _recompute_target_logp_and_values_batched(
        policy, [trajectory], forward_chunk=4, behavior_temperature=0.25
    )

    # Values are unaffected by temperature scaling of the policy logits.
    np.testing.assert_allclose(raw_values, scaled_values, atol=1e-6)
    # Log-probs at behavior_temperature=0.25 must differ from T=1.0 (temperature actually
    # threaded through), mirroring ppo_update's behavior_logits scaling.
    assert not np.allclose(raw_logp, scaled_logp)


def test_recompute_target_logp_matches_manual_temperature_scaling() -> None:
    import torch

    policy = _make_policy()
    trajectory = _make_trajectory(n=3)
    samples = trajectory.samples

    target_logp, _ = _recompute_target_logp_and_values_batched(
        policy, [trajectory], forward_chunk=8192, behavior_temperature=0.25
    )

    observations = _policy_observation_array(policy, samples)
    obs_t = torch.as_tensor(observations, dtype=torch.float32, device=policy.device)
    logits, _ = policy.forward(obs_t, None)
    masked = _masked_logits(logits, [s.valid_actions for s in samples], policy.action_size)
    scaled = torch.clamp(masked / 0.25, min=-50.0, max=50.0)
    actions_t = torch.as_tensor(
        [s.action for s in samples], dtype=torch.long, device=policy.device
    )
    expected = (
        torch.distributions.Categorical(logits=scaled)
        .log_prob(actions_t)
        .detach()
        .cpu()
        .numpy()
    )

    np.testing.assert_allclose(target_logp, expected, atol=1e-5)


def test_recompute_target_logp_defaults_to_unscaled_temperature() -> None:
    """behavior_temperature must default to 1.0 (a no-op) for backward compatibility."""
    policy = _make_policy()
    trajectory = _make_trajectory(n=3)

    default_logp, _ = _recompute_target_logp_and_values_batched(
        policy, [trajectory], forward_chunk=8192
    )
    explicit_logp, _ = _recompute_target_logp_and_values_batched(
        policy, [trajectory], forward_chunk=8192, behavior_temperature=1.0
    )

    np.testing.assert_allclose(default_logp, explicit_logp, atol=1e-9)
