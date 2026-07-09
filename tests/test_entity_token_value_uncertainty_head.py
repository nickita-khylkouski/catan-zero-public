"""Tests for the optional value-uncertainty auxiliary head (contingency f67, D2).

The head is gated by EntityGraphConfig.value_uncertainty_head (default False). It
must be a pure no-op by default -- bit-identical parameter set and forward outputs
to a model built before the field existed -- and, when enabled, emit a
non-negative scalar per state under the "value_uncertainty" output key.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from catan_zero.rl.action_features import CONTEXT_ACTION_FEATURE_SIZE
from catan_zero.rl.entity_token_features import (
    EDGE_FEATURE_SIZE,
    EVENT_FEATURE_SIZE,
    GLOBAL_FEATURE_SIZE,
    HEX_FEATURE_SIZE,
    LEGAL_ACTION_FEATURE_SIZE,
    PLAYER_FEATURE_SIZE,
    VERTEX_FEATURE_SIZE,
)
from catan_zero.rl.entity_token_policy import EntityGraphConfig, EntityGraphNet


def _config(*, value_uncertainty_head: bool) -> EntityGraphConfig:
    return EntityGraphConfig(
        action_size=64,
        static_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
        context_action_feature_size=CONTEXT_ACTION_FEATURE_SIZE,
        legal_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
        hidden_size=16,
        state_layers=1,
        attention_heads=2,
        dropout=0.0,
        value_uncertainty_head=value_uncertainty_head,
    )


def _synthetic_batch(batch_size: int = 3, num_actions: int = 5) -> dict:
    counts = {
        "hex": (19, HEX_FEATURE_SIZE),
        "vertex": (54, VERTEX_FEATURE_SIZE),
        "edge": (72, EDGE_FEATURE_SIZE),
        "player": (4, PLAYER_FEATURE_SIZE),
        "global": (1, GLOBAL_FEATURE_SIZE),
        "event": (64, EVENT_FEATURE_SIZE),
    }
    batch: dict = {}
    for name, (count, feat) in counts.items():
        batch[f"{name}_tokens"] = torch.randn(batch_size, count, feat)
        if name != "global":
            batch[f"{name}_mask"] = torch.ones(batch_size, count, dtype=torch.bool)
    batch["legal_action_tokens"] = torch.randn(batch_size, num_actions, LEGAL_ACTION_FEATURE_SIZE)
    batch["legal_action_context"] = torch.randn(batch_size, num_actions, CONTEXT_ACTION_FEATURE_SIZE)
    return batch


def test_default_config_omits_the_head():
    assert EntityGraphConfig(action_size=1, static_action_feature_size=1).value_uncertainty_head is False
    model = EntityGraphNet(_config(value_uncertainty_head=False))
    assert model.value_uncertainty_head is None
    model.eval()
    outputs = model(_synthetic_batch())
    assert "value_uncertainty" not in outputs
    assert "value" in outputs


def test_enabled_head_emits_nonnegative_per_state_scalar():
    model = EntityGraphNet(_config(value_uncertainty_head=True))
    assert model.value_uncertainty_head is not None
    model.eval()
    batch = _synthetic_batch(batch_size=3)
    outputs = model(batch)
    assert "value_uncertainty" in outputs
    unc = outputs["value_uncertainty"]
    assert unc.shape == (3,)
    # softplus output is strictly non-negative (predicted squared error).
    assert torch.all(unc >= 0.0)
    assert torch.isfinite(unc).all()


def test_enabling_the_head_adds_parameters():
    without = sum(p.numel() for p in EntityGraphNet(_config(value_uncertainty_head=False)).parameters())
    with_head = sum(p.numel() for p in EntityGraphNet(_config(value_uncertainty_head=True)).parameters())
    assert with_head > without


def test_head_gradient_flows():
    model = EntityGraphNet(_config(value_uncertainty_head=True))
    model.train()
    outputs = model(_synthetic_batch())
    loss = outputs["value_uncertainty"].mean()
    loss.backward()
    head_grads = [p.grad for p in model.value_uncertainty_head.parameters() if p.grad is not None]
    assert head_grads
    assert any(g.abs().sum().item() > 0.0 for g in head_grads)
