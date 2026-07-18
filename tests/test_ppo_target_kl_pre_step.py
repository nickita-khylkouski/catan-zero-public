from __future__ import annotations

import numpy as np
import torch

from catan_zero.rl.self_play import StepSample
from catan_zero.rl.torch_ppo import (
    PPOTrajectory,
    TorchPPOPolicy,
    _masked_logits,
    _policy_observation_array,
    _policy_parameters,
    ppo_update,
)


def test_target_kl_refuses_violating_minibatch_before_optimizer_step() -> None:
    policy = TorchPPOPolicy(4, 3, hidden_size=8, seed=7, architecture="flat")
    rng = np.random.default_rng(2)
    samples = [
        StepSample(
            observation=rng.normal(size=4).astype(np.float32),
            valid_actions=(0, 1, 2),
            action=index % 3,
            player="BLUE",
        )
        for index in range(12)
    ]
    observations = torch.as_tensor(
        _policy_observation_array(policy, samples), dtype=torch.float32
    )
    with torch.no_grad():
        logits, _values = policy.forward(observations, None)
        masked = _masked_logits(
            logits,
            [sample.valid_actions for sample in samples],
            policy.action_size,
        )
        actions = torch.as_tensor([sample.action for sample in samples])
        old_log_probs = (
            torch.distributions.Categorical(logits=masked)
            .log_prob(actions)
            .tolist()
        )
        old_action_probs = [
            torch.softmax(masked[index], dim=0).numpy()
            for index in range(len(samples))
        ]
    trajectory = PPOTrajectory(
        samples=samples,
        returns=np.linspace(-2, 2, len(samples)).tolist(),
        advantages=np.linspace(-1, 1, len(samples)).tolist(),
        old_log_probs=old_log_probs,
        old_values=[0.0] * len(samples),
        old_action_probs=old_action_probs,
        shaped_rewards=[0.0] * len(samples),
    )
    optimizer = torch.optim.Adam(_policy_parameters(policy), lr=0.05)
    real_step = optimizer.step
    applied_steps: list[None] = []

    def counted_step(*args, **kwargs):
        applied_steps.append(None)
        return real_step(*args, **kwargs)

    optimizer.step = counted_step
    np.random.seed(1)
    stats = ppo_update(
        policy,
        [trajectory],
        learning_rate=0.05,
        optimizer=optimizer,
        clip_ratio=0.1,
        value_coef=0.0,
        entropy_coef=0.0,
        epochs=4,
        minibatch_size=len(samples),
        target_kl=1e-6,
        advantage_normalization="none",
    )

    assert stats["early_stop"] == 1.0
    assert stats["approx_kl"] > 1e-6
    assert stats["minibatches"] == 1.0
    assert len(applied_steps) == 1
