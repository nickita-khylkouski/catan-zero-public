"""Tests for the CAT-100 auxiliary Catan-subgoal heads.

EntityGraphConfig.aux_subgoal_heads (default False) adds prediction heads for
longest-road / largest-army / VP-in-N / next-settlement / robber-target off the
shared pooled state token (UNREAL, arXiv 1611.05397). They must:
  * be absent by default (no params, no outputs),
  * emit the five aux outputs with the right shapes when enabled,
  * NEVER change value/policy/final_vp outputs (warm-start safe by construction),
  * carry gradients.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from catan_zero.rl.action_features import CONTEXT_ACTION_FEATURE_SIZE  # noqa: E402
from catan_zero.rl.entity_token_features import (  # noqa: E402
    EDGE_FEATURE_SIZE,
    EVENT_FEATURE_SIZE,
    GLOBAL_FEATURE_SIZE,
    HEX_FEATURE_SIZE,
    LEGAL_ACTION_FEATURE_SIZE,
    PLAYER_FEATURE_SIZE,
    VERTEX_FEATURE_SIZE,
)
from catan_zero.rl.entity_token_policy import (  # noqa: E402
    AUX_NUM_HEXES,
    AUX_NUM_INTERSECTIONS,
    EntityGraphConfig,
    EntityGraphNet,
)
from tools import train_bc  # noqa: E402

_AUX_KEYS = (
    "aux_longest_road",
    "aux_largest_army",
    "aux_vp_in_n",
    "aux_next_settlement",
    "aux_robber_target",
)


def _config(*, aux_subgoal_heads: bool, dropout: float = 0.0) -> EntityGraphConfig:
    return EntityGraphConfig(
        action_size=64,
        static_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
        context_action_feature_size=CONTEXT_ACTION_FEATURE_SIZE,
        legal_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
        hidden_size=16,
        state_layers=1,
        attention_heads=2,
        dropout=dropout,
        aux_subgoal_heads=aux_subgoal_heads,
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


def test_default_config_has_no_aux_heads():
    assert EntityGraphConfig(action_size=1, static_action_feature_size=1).aux_subgoal_heads is False
    model = EntityGraphNet(_config(aux_subgoal_heads=False))
    assert not hasattr(model, "aux_next_settlement_head")
    outputs = model(_synthetic_batch())
    for key in _AUX_KEYS:
        assert key not in outputs


def test_enabled_heads_emit_expected_shapes():
    model = EntityGraphNet(_config(aux_subgoal_heads=True))
    model.eval()
    outputs = model(_synthetic_batch(batch_size=3))
    assert outputs["aux_longest_road"].shape == (3,)
    assert outputs["aux_largest_army"].shape == (3,)
    assert outputs["aux_vp_in_n"].shape == (3,)
    assert outputs["aux_next_settlement"].shape == (3, AUX_NUM_INTERSECTIONS)
    assert outputs["aux_robber_target"].shape == (3, AUX_NUM_HEXES)
    for key in _AUX_KEYS:
        assert torch.isfinite(outputs[key]).all()


def test_aux_heads_do_not_change_value_or_policy():
    """Value/policy/final_vp must be bit-identical to the aux-off model once the
    shared trunk is copied over -- the aux heads only add extra outputs."""
    torch.manual_seed(0)
    off = EntityGraphNet(_config(aux_subgoal_heads=False))
    torch.manual_seed(0)
    on = EntityGraphNet(_config(aux_subgoal_heads=True))
    missing, unexpected = on.load_state_dict(off.state_dict(), strict=False)
    assert unexpected == []
    assert all(k.startswith("aux_") for k in missing)
    off.eval()
    on.eval()
    batch = _synthetic_batch()
    out_off = off(batch)
    out_on = on(batch)
    for key in ("logits", "value", "final_vp"):
        assert torch.allclose(out_off[key], out_on[key], atol=0.0, rtol=0.0), key


def test_aux_head_gradients_flow():
    model = EntityGraphNet(_config(aux_subgoal_heads=True))
    report = train_bc._freeze_inactive_training_heads(
        model,
        final_vp_loss_weight=0.0,
        value_uncertainty_loss_weight=0.0,
        value_categorical_loss_weight=0.0,
        aux_subgoal_loss_weight=0.1,
        belief_resource_loss_weight=0.0,
    )
    model.train()
    outputs = model(_synthetic_batch())
    assert all(name in outputs for name in _AUX_KEYS)
    assert report["active_optional_submodules"] == [
        "aux_largest_army_head",
        "aux_longest_road_head",
        "aux_next_settlement_head",
        "aux_robber_target_head",
        "aux_vp_in_n_head",
    ]
    loss = (
        outputs["aux_longest_road"].mean()
        + outputs["aux_vp_in_n"].mean()
        + outputs["aux_next_settlement"].mean()
        + outputs["aux_robber_target"].mean()
    )
    loss.backward()
    for head in (
        model.aux_longest_road_head,
        model.aux_vp_in_n_head,
        model.aux_next_settlement_head,
        model.aux_robber_target_head,
    ):
        grads = [p.grad for p in head.parameters() if p.grad is not None]
        assert grads and any(g.abs().sum().item() > 0.0 for g in grads)


def test_inactive_aux_heads_do_not_perturb_main_training_rng():
    """A zero-weight architecture arm must follow the aux-off trajectory.

    The main outputs occur before the auxiliary readouts, so an unused aux
    dropout does not alter the *first* forward.  If the frozen head still runs,
    however, it advances the process-wide torch RNG and changes trunk dropout
    on the second batch.  This two-forward check catches that contamination.
    """

    torch.manual_seed(0)
    off = EntityGraphNet(_config(aux_subgoal_heads=False, dropout=0.25))
    torch.manual_seed(0)
    on = EntityGraphNet(_config(aux_subgoal_heads=True, dropout=0.25))
    missing, unexpected = on.load_state_dict(off.state_dict(), strict=False)
    assert unexpected == []
    assert missing and all(name.startswith("aux_") for name in missing)

    reports = []
    for model in (off, on):
        reports.append(
            train_bc._freeze_inactive_training_heads(
                model,
                # Keep final-VP active so this test isolates the aux-head delta.
                final_vp_loss_weight=0.1,
                value_uncertainty_loss_weight=0.0,
                value_categorical_loss_weight=0.0,
                aux_subgoal_loss_weight=0.0,
                belief_resource_loss_weight=0.0,
            )
        )
        model.train()

    batch = _synthetic_batch(batch_size=2)

    def trajectory(model):
        torch.manual_seed(12345)
        first = model(batch)
        second = model(batch)
        return first, second, torch.random.get_rng_state()

    off_first, off_second, off_rng = trajectory(off)
    on_first, on_second, on_rng = trajectory(on)

    for key in ("logits", "value", "final_vp"):
        assert torch.equal(off_first[key], on_first[key]), key
        assert torch.equal(off_second[key], on_second[key]), key
    assert torch.equal(off_rng, on_rng)
    assert not any(name in on_first for name in _AUX_KEYS)
    assert reports[1]["zero_weight_skips_forward"] is True
    assert all(
        not parameter.requires_grad
        for name, parameter in on.named_parameters()
        if name.startswith("aux_")
    )
