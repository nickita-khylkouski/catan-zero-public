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
from catan_zero.rl.ordered_history import (  # noqa: E402
    MASKED_MEAN_V1,
    ORDERED_ATTENTION_V2,
    build_ordered_history_pool,
)


def _config(*, ordered: bool) -> EntityGraphConfig:
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
    if not ordered:
        return base
    return replace(
        base,
        meaningful_public_history=True,
        event_history_limit=32,
        meaningful_public_history_pooling=ORDERED_ATTENTION_V2,
    )


def _batch(*, history: bool) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260715)
    batch: dict[str, torch.Tensor] = {}
    for name, count, width in (
        ("hex", 19, HEX_FEATURE_SIZE),
        ("vertex", 54, VERTEX_FEATURE_SIZE),
        ("edge", 72, EDGE_FEATURE_SIZE),
        ("player", 4, PLAYER_FEATURE_SIZE),
        ("global", 1, GLOBAL_FEATURE_SIZE),
    ):
        batch[f"{name}_tokens"] = torch.randn(
            2, count, width, generator=generator
        )
        if name != "global":
            batch[f"{name}_mask"] = torch.ones(2, count, dtype=torch.bool)
    batch["legal_action_tokens"] = torch.randn(
        2, 3, LEGAL_ACTION_FEATURE_SIZE, generator=generator
    )
    batch["legal_action_context"] = torch.randn(
        2, 3, CONTEXT_ACTION_FEATURE_SIZE, generator=generator
    )
    width = 32 if history else 64
    batch["event_tokens"] = torch.randn(
        2, width, EVENT_FEATURE_SIZE, generator=generator
    )
    batch["event_mask"] = torch.zeros(2, width, dtype=torch.bool)
    if history:
        batch["event_mask"][:, -4:] = True
    return batch


def test_ordered_history_upgrade_is_exact_at_zero_gate() -> None:
    torch.manual_seed(17)
    incumbent = EntityGraphNet(_config(ordered=False)).eval()
    upgraded = EntityGraphNet(_config(ordered=True)).eval()
    missing, unexpected = upgraded.load_state_dict(
        incumbent.state_dict(), strict=False
    )

    assert unexpected == []
    assert set(missing) == {
        "meaningful_history_residual_gate",
        "meaningful_history_ordered_gate",
        "meaningful_history_sequence.position_embedding",
        "meaningful_history_sequence.query",
        "meaningful_history_sequence.norm.weight",
        "meaningful_history_sequence.norm.bias",
    }
    with torch.no_grad():
        before = incumbent(_batch(history=False))
        after = upgraded(_batch(history=True))
    for key in ("logits", "value", "final_vp"):
        assert torch.equal(before[key], after[key]), key


def test_ordered_branch_preserves_trained_v1_history_path() -> None:
    torch.manual_seed(19)
    v1_config = replace(
        _config(ordered=False),
        meaningful_public_history=True,
        event_history_limit=32,
        meaningful_public_history_pooling=MASKED_MEAN_V1,
    )
    v1 = EntityGraphNet(v1_config).eval()
    with torch.no_grad():
        v1.meaningful_history_residual_gate.fill_(0.3)
    upgraded = EntityGraphNet(
        replace(v1_config, meaningful_public_history_pooling=ORDERED_ATTENTION_V2)
    ).eval()
    missing, unexpected = upgraded.load_state_dict(v1.state_dict(), strict=False)
    assert unexpected == []
    assert set(missing) == {
        "meaningful_history_ordered_gate",
        "meaningful_history_sequence.position_embedding",
        "meaningful_history_sequence.query",
        "meaningful_history_sequence.norm.weight",
        "meaningful_history_sequence.norm.bias",
    }

    batch = _batch(history=True)
    with torch.no_grad():
        before = v1(batch)
        after = upgraded(batch)
    for key in ("logits", "value", "final_vp"):
        assert torch.equal(before[key], after[key]), key


def test_ordered_pool_is_mask_safe_and_can_distinguish_order() -> None:
    pool = build_ordered_history_pool(width=4, max_events=4).eval()
    tokens = torch.tensor(
        [[[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]]]
    )
    valid = torch.tensor([[True, True]])
    with torch.no_grad():
        pool.position_embedding[-2] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        pool.position_embedding[-1] = torch.tensor([0.0, 1.0, 0.0, 0.0])
        pool.query.copy_(torch.tensor([1.0, -0.5, 0.25, 0.0]))
        forward = pool(tokens, valid)
        reversed_order = pool(tokens.flip(1), valid)
        empty = pool(tokens, torch.zeros_like(valid))

    assert not torch.equal(forward, reversed_order)
    assert torch.equal(empty, torch.zeros_like(empty))
    assert torch.isfinite(forward).all()


def test_scratch_ordered_history_is_live_on_the_first_backward() -> None:
    from tools import train_bc

    torch.manual_seed(23)
    model = EntityGraphNet(_config(ordered=True)).train()
    report = train_bc._initialize_scratch_meaningful_history_path(
        model, scratch=True
    )

    assert report["masked_mean_gate_initialization"] == "ones"
    assert report["ordered_additive_gate_initialization"] == "ones"
    assert torch.equal(
        model.meaningful_history_ordered_gate,
        torch.ones_like(model.meaningful_history_ordered_gate),
    )

    output = model(_batch(history=True))
    loss = output["logits"].square().mean() + output["value"].square().mean()
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.meaningful_history_sequence.parameters()
    ]
    assert all(gradient is not None for gradient in gradients)
    assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0.0


def test_ordered_upgrade_has_typed_function_preserving_receipt() -> None:
    from tools import a1_function_preserving_upgrade as upgrade

    spec = upgrade.ALLOWLIST[upgrade.MODULE_ORDERED_MEANINGFUL_PUBLIC_HISTORY]
    assert spec["config_delta"]["meaningful_public_history_pooling"] == (
        ORDERED_ATTENTION_V2
    )
    assert spec["new_parameter_initialization"] == {
        "meaningful_history_residual_gate": "zeros",
        "meaningful_history_ordered_gate": "zeros",
        "meaningful_history_sequence.norm.bias": "zeros",
        "meaningful_history_sequence.norm.weight": "ones",
        "meaningful_history_sequence.position_embedding": "zeros",
        "meaningful_history_sequence.query": "seeded_torch_default",
    }

    from_v1 = upgrade.ALLOWLIST[
        upgrade.MODULE_ORDERED_MEANINGFUL_PUBLIC_HISTORY_FROM_V1
    ]
    assert from_v1["config_delta"] == {
        "meaningful_public_history_pooling": ORDERED_ATTENTION_V2
    }
    assert "meaningful_history_residual_gate" not in from_v1[
        "new_parameter_initialization"
    ]
