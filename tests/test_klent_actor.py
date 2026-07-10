from __future__ import annotations

import numpy as np
import pytest
import torch

import catan_zero.rl.klent_actor as actor_module
from catan_zero.rl.klent import KLENTConfig
from catan_zero.rl.klent_actor import sample_entity_policy_step


class _Policy:
    def __init__(self, *, q: bool = True) -> None:
        self.q = q

    def _legal_outputs_from_env(self, env, info, legal_actions, *, return_q):
        assert return_q is True
        outputs = {
            "logits": torch.tensor([[1.0, -1.0]]),
            "value": torch.tensor([0.25]),
        }
        if self.q:
            outputs["q_values"] = torch.tensor([[0.0, 2.0]])
        entity = {"schema": "ignored", "hex_tokens": np.ones((19, 2), dtype=np.float32)}
        context = np.zeros((2, 18), dtype=np.float32)
        return outputs, entity, context


def test_actor_samples_improved_policy_and_records_training_inputs(monkeypatch) -> None:
    monkeypatch.setattr(
        actor_module,
        "build_action_context_feature_table",
        lambda env, info: np.zeros((64, 18), dtype=np.float32),
    )
    step = sample_entity_policy_step(
        _Policy(),
        object(),
        {"valid_actions": (11, 22)},
        np.random.default_rng(7),
        config=KLENTConfig(entropy_coefficient=0.03, reverse_kl_coefficient=0.1),
    )
    assert step.action in (11, 22)
    assert step.policy_target.sum() == pytest.approx(1.0)
    assert step.policy_target[1] > step.policy_target[0]
    assert step.expected_q == pytest.approx(2.0 * step.policy_target[1])
    assert step.legal_action_ids.tolist() == [11, 22]
    assert step.legal_action_context.shape == (2, 18)
    assert step.action_context_table.shape == (64, 18)
    assert "schema" not in step.entity_features


def test_actor_is_seed_reproducible(monkeypatch) -> None:
    monkeypatch.setattr(
        actor_module,
        "build_action_context_feature_table",
        lambda env, info: np.zeros((64, 18), dtype=np.float32),
    )
    args = (_Policy(), object(), {"valid_actions": (11, 22)})
    first = sample_entity_policy_step(*args, np.random.default_rng(123))
    second = sample_entity_policy_step(*args, np.random.default_rng(123))
    assert first.action == second.action
    assert np.array_equal(first.policy_target, second.policy_target)


def test_actor_fails_without_action_q_head_or_legal_actions(monkeypatch) -> None:
    monkeypatch.setattr(
        actor_module,
        "build_action_context_feature_table",
        lambda env, info: np.zeros((64, 18), dtype=np.float32),
    )
    with pytest.raises(RuntimeError, match="q_values"):
        sample_entity_policy_step(
            _Policy(q=False), object(), {"valid_actions": (11, 22)}, np.random.default_rng(0)
        )
    with pytest.raises(ValueError, match="no legal"):
        sample_entity_policy_step(
            _Policy(), object(), {"valid_actions": ()}, np.random.default_rng(0)
        )
