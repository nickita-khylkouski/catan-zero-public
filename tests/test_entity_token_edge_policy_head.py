"""Tests for the CAT-97 GATEAU-style edge/node-feature policy head.

EntityGraphConfig.edge_policy_head (default False) adds a direct per-action
logit read from each action's pooled target-entity token (AlphaGateau's
policy = f(edge/node feature) readout, arXiv 2410.23753). It must be:
  * a pure no-op by default (no params, no output change),
  * bit-identical at init when enabled (zero-init final Linear), so it
    warm-starts from a checkpoint trained without it,
  * a trainable, gradient-carrying logit contribution once its weights move.
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


def _config(*, edge_policy_head: bool) -> EntityGraphConfig:
    return EntityGraphConfig(
        action_size=64,
        static_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
        context_action_feature_size=CONTEXT_ACTION_FEATURE_SIZE,
        legal_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
        hidden_size=16,
        state_layers=1,
        attention_heads=2,
        dropout=0.0,
        edge_policy_head=edge_policy_head,
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
    # Each action targets a vertex (settlement-like); other target columns absent.
    target_ids = -torch.ones(batch_size, num_actions, 4, dtype=torch.long)
    target_ids[:, :, 1] = torch.arange(num_actions).remainder(54).view(1, -1)
    batch["legal_action_target_ids"] = target_ids
    return batch


def test_default_config_has_no_edge_policy_head():
    assert EntityGraphConfig(action_size=1, static_action_feature_size=1).edge_policy_head is False
    model = EntityGraphNet(_config(edge_policy_head=False))
    assert not hasattr(model, "edge_policy_mlp")


def test_enabling_head_adds_parameters():
    without = sum(p.numel() for p in EntityGraphNet(_config(edge_policy_head=False)).parameters())
    with_head = sum(p.numel() for p in EntityGraphNet(_config(edge_policy_head=True)).parameters())
    assert with_head > without


def test_bit_identical_at_init():
    """Enabled head is zero-init -> logits/value identical to the off model when
    both share the same trunk weights (warm-start guarantee)."""
    torch.manual_seed(0)
    off = EntityGraphNet(_config(edge_policy_head=False))
    torch.manual_seed(0)
    on = EntityGraphNet(_config(edge_policy_head=True))
    # Copy shared trunk/head weights; the edge head keeps its own zero-init.
    missing, unexpected = on.load_state_dict(off.state_dict(), strict=False)
    assert unexpected == []
    assert all(k.startswith("edge_policy_mlp.") for k in missing)
    off.eval()
    on.eval()
    batch = _synthetic_batch()
    out_off = off(batch)
    out_on = on(batch)
    for key in ("logits", "value", "final_vp"):
        assert torch.allclose(out_off[key], out_on[key], atol=0.0, rtol=0.0), key


def test_head_gradient_flows_and_moves_logits():
    model = EntityGraphNet(_config(edge_policy_head=True))
    # Perturb the zero-init final layer so the head contributes a nonzero logit.
    with torch.no_grad():
        model.edge_policy_mlp[4].weight.add_(torch.randn_like(model.edge_policy_mlp[4].weight))
        model.edge_policy_mlp[4].bias.add_(torch.randn_like(model.edge_policy_mlp[4].bias))
    model.train()
    outputs = model(_synthetic_batch())
    loss = outputs["logits"].sum()
    loss.backward()
    grads = [p.grad for p in model.edge_policy_mlp.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum().item() > 0.0 for g in grads)
