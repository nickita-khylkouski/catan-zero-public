from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from catan_zero.rl.self_play import StepSample
from catan_zero.rl.torch_ppo import PPOTrajectory, TorchPPOPolicy, ppo_update


def _trajectory(*, entity: bool) -> PPOTrajectory:
    advantages = [2.0, 1.0, -1.0, -2.0]
    actions = [0, 0, 1, 1]
    samples = [
        StepSample(
            observation=np.asarray(
                [float(index == 0), float(index == 1), 1.0],
                dtype=np.float64,
            ),
            valid_actions=(0, 1),
            action=action,
            player="BLUE",
            entity_features=(
                {"test_placeholder": np.zeros(1, dtype=np.float32)} if entity else None
            ),
        )
        for index, action in enumerate(actions)
    ]
    return PPOTrajectory(
        samples=samples,
        returns=[0.0] * len(samples),
        advantages=advantages,
        old_log_probs=[-math.log(2.0)] * len(samples),
        old_values=[0.0] * len(samples),
        old_action_probs=[np.asarray([0.5, 0.5], dtype=np.float32) for _ in samples],
        shaped_rewards=[0.0] * len(samples),
        opponent_names={"RED": "random"},
    )


def _dense_actor_gradient(group_weight: float) -> tuple[float, dict[str, float]]:
    import torch

    policy = TorchPPOPolicy(3, 2, hidden_size=8, seed=91)
    # Make the behavior reference exact while preserving the action/advantage
    # correlation that produces a nonzero policy gradient.
    trajectory = _trajectory(entity=False)
    observations = torch.as_tensor(
        np.stack([sample.observation for sample in trajectory.samples]),
        dtype=torch.float32,
        device=policy.device,
    )
    with torch.no_grad():
        logits, _value = policy.forward(observations)
        dist = torch.distributions.Categorical(logits=logits)
        actions = torch.as_tensor(
            [sample.action for sample in trajectory.samples],
            dtype=torch.long,
            device=policy.device,
        )
        trajectory.old_log_probs = dist.log_prob(actions).cpu().tolist()
    optimizer = torch.optim.SGD(policy.model.parameters(), lr=0.0)
    # The actor is outside policy.model for the dense architecture.
    optimizer.add_param_group({"params": policy.actor.parameters()})
    optimizer.add_param_group({"params": policy.critic.parameters()})
    metrics = ppo_update(
        policy,
        [trajectory],
        learning_rate=0.0,
        clip_ratio=0.2,
        value_coef=0.0,
        entropy_coef=0.0,
        epochs=1,
        minibatch_size=64,
        optimizer=optimizer,
        advantage_normalization="global",
        advantage_group_weights={"random": group_weight},
    )
    gradient = torch.sqrt(
        sum(
            parameter.grad.detach().square().sum()
            for parameter in policy.actor.parameters()
            if parameter.grad is not None
        )
    )
    return float(gradient.item()), metrics


def test_dense_global_normalization_preserves_group_weight_multiplier() -> None:
    full_gradient, full_metrics = _dense_actor_gradient(1.0)
    scaled_gradient, scaled_metrics = _dense_actor_gradient(0.1)

    assert full_gradient > 0.0
    assert scaled_gradient / full_gradient == pytest.approx(0.1, rel=1e-5)
    assert full_metrics["advantage_group_weight_mean"] == pytest.approx(1.0)
    assert scaled_metrics["advantage_group_weight_mean"] == pytest.approx(0.1)


def _entity_actor_gradient(
    monkeypatch: pytest.MonkeyPatch,
    group_weight: float,
) -> tuple[float, dict[str, float]]:
    import torch

    from catan_zero.rl import torch_ppo

    class TinyEntityModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.action_logits = torch.nn.Parameter(torch.zeros(2))

    policy = SimpleNamespace(
        architecture="entity_graph",
        device=torch.device("cpu"),
        model=TinyEntityModel(),
    )

    def outputs(candidate, samples, **_kwargs):
        logits = candidate.model.action_logits.unsqueeze(0).expand(len(samples), -1)
        return {
            "logits": logits,
            "value": logits[:, 0] * 0.0,
        }

    monkeypatch.setattr(torch_ppo, "_entity_graph_outputs", outputs)
    optimizer = torch.optim.SGD(policy.model.parameters(), lr=0.0)
    metrics = ppo_update(
        policy,
        [_trajectory(entity=True)],
        learning_rate=0.0,
        clip_ratio=0.2,
        value_coef=0.0,
        entropy_coef=0.0,
        epochs=1,
        minibatch_size=64,
        optimizer=optimizer,
        advantage_normalization="global",
        advantage_group_weights={"random": group_weight},
    )
    gradient = policy.model.action_logits.grad
    assert gradient is not None
    return float(gradient.norm().item()), metrics


def test_entity_global_normalization_preserves_group_weight_multiplier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_gradient, full_metrics = _entity_actor_gradient(monkeypatch, 1.0)
    scaled_gradient, scaled_metrics = _entity_actor_gradient(monkeypatch, 0.1)

    assert full_gradient > 0.0
    assert scaled_gradient / full_gradient == pytest.approx(0.1, rel=1e-5)
    assert full_metrics["advantage_group_weight_mean"] == pytest.approx(1.0)
    assert scaled_metrics["advantage_group_weight_mean"] == pytest.approx(0.1)
