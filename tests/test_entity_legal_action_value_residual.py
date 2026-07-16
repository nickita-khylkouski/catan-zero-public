"""Contracts for the legal-affordance value repair."""

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
    STATIC_ACTION_RESIDUAL_FEATURE_SIZE,
    EntityGraphConfig,
    EntityGraphNet,
)


def _config(**overrides) -> EntityGraphConfig:
    values = {
        "action_size": 607,
        "static_action_feature_size": 45,
        "context_action_feature_size": CONTEXT_ACTION_FEATURE_SIZE,
        "legal_action_feature_size": LEGAL_ACTION_FEATURE_SIZE,
        "hidden_size": 32,
        "state_layers": 1,
        "attention_heads": 4,
        "dropout": 0.0,
    }
    values.update(overrides)
    return EntityGraphConfig(**values)


def _batch(*, batch_size: int = 2, action_width: int = 5) -> dict:
    generator = torch.Generator().manual_seed(20260715)
    batch = {}
    for name, count, width in (
        ("hex", 19, HEX_FEATURE_SIZE),
        ("vertex", 54, VERTEX_FEATURE_SIZE),
        ("edge", 72, EDGE_FEATURE_SIZE),
        ("player", 4, PLAYER_FEATURE_SIZE),
        ("global", 1, GLOBAL_FEATURE_SIZE),
        ("event", 0, EVENT_FEATURE_SIZE),
    ):
        batch[f"{name}_tokens"] = torch.randn(
            batch_size, count, width, generator=generator
        )
        if name != "global":
            batch[f"{name}_mask"] = torch.ones(
                batch_size, count, dtype=torch.bool
            )
    batch["legal_action_tokens"] = torch.randn(
        batch_size, action_width, LEGAL_ACTION_FEATURE_SIZE, generator=generator
    )
    batch["legal_action_context"] = torch.randn(
        batch_size, action_width, CONTEXT_ACTION_FEATURE_SIZE, generator=generator
    )
    batch["legal_action_static_features"] = torch.randn(
        batch_size,
        action_width,
        STATIC_ACTION_RESIDUAL_FEATURE_SIZE,
        generator=generator,
    )
    batch["legal_action_mask"] = torch.ones(
        batch_size, action_width, dtype=torch.bool
    )
    return batch


def test_default_off_has_no_new_parameters() -> None:
    legacy = EntityGraphNet(_config())
    explicit_off = EntityGraphNet(_config(legal_action_value_residual=False))

    assert set(legacy.state_dict()) == set(explicit_off.state_dict())
    assert not any(
        name.startswith("legal_action_value_residual_proj.")
        for name, _parameter in legacy.named_parameters()
    )


def test_combined_structured_action_upgrade_is_bit_exact_then_breaks_value_alias() -> None:
    torch.manual_seed(7)
    legacy = EntityGraphNet(_config()).eval()
    treatment = EntityGraphNet(
        _config(
            static_action_residual=True,
            legal_action_value_residual=True,
        )
    ).eval()
    missing, unexpected = treatment.load_state_dict(
        legacy.state_dict(), strict=False
    )

    assert not unexpected
    assert set(missing) == {
        "legal_action_value_residual_proj.weight",
        "legal_action_value_static_proj.weight",
        "static_action_residual_proj.bias",
        "static_action_residual_proj.weight",
    }
    batch = _batch()
    with torch.no_grad():
        baseline = legacy(batch)
        upgraded = treatment(batch)
    for key in baseline:
        assert torch.equal(baseline[key], upgraded[key]), key

    # The value-private catalog path can learn resource semantics without
    # changing the shared policy action encoder or static adapter.
    with torch.no_grad():
        treatment.legal_action_value_static_proj.weight[
            :STATIC_ACTION_RESIDUAL_FEATURE_SIZE, :
        ] = torch.eye(STATIC_ACTION_RESIDUAL_FEATURE_SIZE)

    changed = {key: value.clone() for key, value in batch.items()}
    changed["legal_action_static_features"][:, 0, 0] += 3.0
    with torch.no_grad():
        before = treatment(batch)["value"]
        after = treatment(changed)["value"]
    assert not torch.equal(before, after)


def test_value_private_static_adapter_gets_first_step_gradient() -> None:
    model = EntityGraphNet(
        _config(
            static_action_residual=True,
            legal_action_value_residual=True,
        )
    ).train()
    batch = _batch(batch_size=3)
    target = torch.tensor([-1.0, 0.0, 1.0])

    torch.nn.functional.mse_loss(model(batch)["value"], target).backward()

    gradient = model.legal_action_value_static_proj.weight.grad
    assert gradient is not None
    assert gradient.abs().sum().item() > 0.0
    # The shared policy adapter remains gated on the first step.
    assert model.static_action_residual_proj.weight.grad is not None
    assert model.static_action_residual_proj.weight.grad.abs().sum().item() == 0.0


def test_padded_actions_do_not_change_legal_affordance_value() -> None:
    torch.manual_seed(11)
    model = EntityGraphNet(_config(legal_action_value_residual=True)).eval()
    with torch.no_grad():
        model.legal_action_value_residual_proj.weight.copy_(
            torch.eye(model.config.hidden_size)
        )
    batch = _batch()
    batch["legal_action_mask"][:, -1] = False
    changed = {key: value.clone() for key, value in batch.items()}
    changed["legal_action_tokens"][:, -1] += 10_000.0
    changed["legal_action_context"][:, -1] -= 10_000.0

    with torch.no_grad():
        original = model(batch)["value"]
        padded_changed = model(changed)["value"]
    torch.testing.assert_close(original, padded_changed, rtol=0.0, atol=0.0)
