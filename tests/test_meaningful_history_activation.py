from __future__ import annotations

from dataclasses import replace

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
    EntityGraphConfig,
    EntityGraphNet,
)


def _config(*, history: bool) -> EntityGraphConfig:
    base = EntityGraphConfig(
        action_size=32,
        static_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
        context_action_feature_size=CONTEXT_ACTION_FEATURE_SIZE,
        legal_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
        hidden_size=16,
        state_layers=1,
        attention_heads=2,
        dropout=0.0,
    )
    return replace(
        base,
        meaningful_public_history=history,
        event_history_limit=32 if history else 64,
    )


def _batch(*, active_history: bool) -> dict:
    generator = torch.Generator().manual_seed(20260715)
    batch = {}
    for name, count, width in (
        ("hex", 19, HEX_FEATURE_SIZE),
        ("vertex", 54, VERTEX_FEATURE_SIZE),
        ("edge", 72, EDGE_FEATURE_SIZE),
        ("player", 4, PLAYER_FEATURE_SIZE),
        ("global", 1, GLOBAL_FEATURE_SIZE),
        ("event", 32, EVENT_FEATURE_SIZE),
    ):
        batch[f"{name}_tokens"] = torch.randn(2, count, width, generator=generator)
        if name != "global":
            batch[f"{name}_mask"] = torch.ones(2, count, dtype=torch.bool)
    batch["event_mask"].fill_(False)
    if active_history:
        batch["event_mask"][:, -4:] = True
    batch["legal_action_tokens"] = torch.randn(
        2, 3, LEGAL_ACTION_FEATURE_SIZE, generator=generator
    )
    batch["legal_action_context"] = torch.randn(
        2, 3, CONTEXT_ACTION_FEATURE_SIZE, generator=generator
    )
    return batch


def test_history_upgrade_is_exact_with_nonempty_events_at_zero_gate():
    torch.manual_seed(17)
    incumbent = EntityGraphNet(_config(history=False)).eval()
    upgraded = EntityGraphNet(_config(history=True)).eval()
    missing, unexpected = upgraded.load_state_dict(incumbent.state_dict(), strict=False)

    assert unexpected == []
    assert missing == ["meaningful_history_residual_gate"]
    assert torch.count_nonzero(upgraded.meaningful_history_residual_gate).item() == 0

    empty = _batch(active_history=False)
    active = _batch(active_history=True)
    with torch.no_grad():
        before = incumbent(empty)
        after = upgraded(active)
    for key in ("logits", "value", "final_vp"):
        assert torch.equal(before[key], after[key]), key


def test_zero_gate_learns_immediately_and_then_history_changes_outputs():
    torch.manual_seed(23)
    model = EntityGraphNet(_config(history=True)).train()
    batch = _batch(active_history=True)
    outputs = model(batch)
    loss = torch.nn.functional.cross_entropy(
        outputs["logits"], torch.tensor([0, 1])
    ) + outputs["value"].square().mean()
    loss.backward()

    gate = model.meaningful_history_residual_gate
    assert gate.grad is not None
    assert gate.grad.abs().sum().item() > 0.0
    with torch.no_grad():
        baseline = model.eval()(batch)["logits"].clone()
        gate.fill_(0.25)
        activated = model(batch)["logits"]
    assert not torch.equal(baseline, activated)


def test_history_upgrade_receipt_allowlist_accepts_only_the_zero_gate():
    from tools import a1_function_preserving_upgrade as upgrade

    spec = upgrade.ALLOWLIST[upgrade.MODULE_MEANINGFUL_PUBLIC_HISTORY]
    assert spec["new_parameter_initialization"] == {
        "meaningful_history_residual_gate": "zeros"
    }
    assert spec["config_delta"]["meaningful_public_history"] is True
    assert spec["config_delta"]["event_history_limit"] == 32


def test_next_wave_combined_upgrade_is_zero_initialized_and_binds_both_inputs():
    from tools import a1_function_preserving_upgrade as upgrade

    spec = upgrade.ALLOWLIST[
        upgrade.MODULE_PUBLIC_CARD_COUNT_MEANINGFUL_HISTORY_V2
    ]
    assert spec["new_parameter_initialization"] == {
        "meaningful_history_residual_gate": "zeros",
        "public_card_count_residual.weight": "zeros",
    }
    assert spec["config_delta"]["public_card_count_features"] is True
    assert spec["config_delta"]["public_card_count_residual_bias"] is False
    assert spec["config_delta"]["meaningful_public_history"] is True
    assert spec["config_delta"]["event_history_limit"] == 32


def test_legacy_combined_upgrade_receipt_remains_bias_bearing_v1():
    from tools import a1_function_preserving_upgrade as upgrade

    spec = upgrade.ALLOWLIST[
        upgrade.MODULE_PUBLIC_CARD_COUNT_MEANINGFUL_HISTORY
    ]
    assert spec["new_parameter_initialization"] == {
        "meaningful_history_residual_gate": "zeros",
        "public_card_count_residual.bias": "zeros",
        "public_card_count_residual.weight": "zeros",
    }
    assert "public_card_count_residual_bias" not in spec["config_delta"]
