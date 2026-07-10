from __future__ import annotations

import numpy as np
import pytest
import torch

import catan_zero.rl.klent_train as trainer
from catan_zero.rl.klent_actor import KLENTActorStep
from catan_zero.rl.klent_train import KLENTTrajectory, update_entity_policy


def _step(action: int, policy: tuple[float, float]) -> KLENTActorStep:
    return KLENTActorStep(
        action=action,
        action_column=(0 if action == 11 else 1),
        behavior_log_probability=float(np.log(policy[0 if action == 11 else 1])),
        value=0.0,
        expected_q=0.0,
        chosen_q=0.0,
        legal_action_ids=np.asarray([11, 22], dtype=np.int64),
        policy_target=np.asarray(policy, dtype=np.float32),
        action_q_values=np.zeros(2, dtype=np.float32),
        legal_action_context=np.zeros((2, 18), dtype=np.float32),
        action_context_table=np.zeros((64, 18), dtype=np.float32),
        entity_features={"event_tokens": np.zeros((1, 2), dtype=np.float32)},
    )


def _trajectory() -> KLENTTrajectory:
    return KLENTTrajectory(
        steps=(_step(11, (0.8, 0.2)), _step(22, (0.3, 0.7))),
        players=("RED", "BLUE"),
        rewards=(0.0, 1.0),
        terminated=(False, True),
        returns=(-0.5, 1.0),
        game_seed=7,
        truncated=False,
    )


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(2))
        self.q = torch.nn.Parameter(torch.zeros(2))
        self.value = torch.nn.Parameter(torch.zeros(()))


class _Policy:
    def __init__(self) -> None:
        self.model = _Model()
        self.device = torch.device("cpu")


def test_trajectory_validation_rejects_incomplete_nontruncated() -> None:
    trajectory = _trajectory()
    broken = KLENTTrajectory(
        steps=trajectory.steps,
        players=trajectory.players,
        rewards=trajectory.rewards,
        terminated=(False, False),
        returns=trajectory.returns,
        game_seed=7,
        truncated=False,
    )
    with pytest.raises(ValueError, match="terminal"):
        broken.validate()


def test_update_trains_policy_q_and_value_without_ppo_ratio(monkeypatch) -> None:
    policy = _Policy()

    def outputs(_policy, samples, *, return_q):
        assert return_q is True
        batch = len(samples)
        return {
            "logits": policy.model.logits.unsqueeze(0).expand(batch, -1),
            "q_values": policy.model.q.unsqueeze(0).expand(batch, -1),
            "value": policy.model.value.expand(batch),
        }

    monkeypatch.setattr(trainer, "_entity_graph_outputs", outputs)
    optimizer = torch.optim.Adam(policy.model.parameters(), lr=0.05)
    before = [parameter.detach().clone() for parameter in policy.model.parameters()]
    report = update_entity_policy(
        policy,
        [_trajectory()],
        optimizer,
        epochs=2,
        minibatch_size=2,
        seed=3,
    )
    after = list(policy.model.parameters())
    assert report["schema_version"] == "catan-zero-klent-update/v1"
    assert report["rows"] == 2
    assert report["updates"] == 2
    assert all(np.isfinite(report[key]) for key in ("loss", "policy_loss", "q_loss", "value_loss"))
    assert all(not torch.equal(old, new) for old, new in zip(before, after))
