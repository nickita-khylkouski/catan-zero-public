"""Contracts for the legal-affordance value repair."""

from __future__ import annotations

import numpy as np
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
    EntityGraphPolicy,
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
    explicit_off = EntityGraphNet(
        _config(
            legal_action_value_residual=False,
            legal_action_value_set_statistics=False,
        )
    )

    assert set(legacy.state_dict()) == set(explicit_off.state_dict())
    assert not any(
        name.startswith("legal_action_value_residual_proj.")
        for name, _parameter in legacy.named_parameters()
    )
    assert not any(
        name.startswith("legal_action_value_count_proj.")
        for name, _parameter in legacy.named_parameters()
    )


def test_combined_structured_action_upgrade_is_bit_exact_then_breaks_value_alias() -> None:
    torch.manual_seed(7)
    legacy = EntityGraphNet(_config()).eval()
    treatment = EntityGraphNet(
        _config(
            static_action_residual=True,
            legal_action_value_residual=True,
            legal_action_value_set_statistics=True,
        )
    ).eval()
    missing, unexpected = treatment.load_state_dict(
        legacy.state_dict(), strict=False
    )

    assert not unexpected
    assert set(missing) == {
        "legal_action_value_residual_proj.weight",
        "legal_action_value_static_proj.weight",
        "legal_action_value_max_proj.weight",
        "legal_action_value_count_proj.weight",
        "legal_action_value_static_max_proj.weight",
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


def _repeat_first_batch_row(batch: dict) -> dict:
    return {
        key: value[:1].expand_as(value).clone()
        for key, value in batch.items()
    }


def _make_value_readout_depend_on_first_hidden_dimension(model) -> None:
    with torch.no_grad():
        first = model.value_head[0]
        last = model.value_head[-1]
        first.weight.zero_()
        first.bias.zero_()
        first.weight[0, 0] = 1.0
        last.weight.zero_()
        last.bias.zero_()
        last.weight[0, 0] = 1.0


def test_legal_count_breaks_equal_mean_value_alias() -> None:
    torch.manual_seed(19)
    model = EntityGraphNet(
        _config(
            legal_action_value_residual=True,
            legal_action_value_set_statistics=True,
        )
    ).eval()
    _make_value_readout_depend_on_first_hidden_dimension(model)

    batch = _repeat_first_batch_row(_batch(batch_size=2, action_width=5))
    batch["legal_action_tokens"][:] = batch["legal_action_tokens"][:, :1]
    batch["legal_action_context"][:] = batch["legal_action_context"][:, :1]
    batch["legal_action_mask"][0] = torch.tensor(
        [True, True, True, False, False]
    )
    batch["legal_action_mask"][1] = True

    with torch.no_grad():
        aliased = model(batch)["value"]
        model.legal_action_value_count_proj.weight[0, 0] = 100.0
        values = model(batch)["value"]
    torch.testing.assert_close(aliased[0], aliased[1], rtol=0.0, atol=0.0)
    assert values[0].item() != pytest.approx(values[1].item())


class _FirstActionScalar(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = int(hidden_size)

    def forward(self, value):
        result = value.new_zeros((*value.shape[:-1], self.hidden_size))
        result[..., 0] = value[..., 0]
        return result


def test_masked_max_preserves_rare_extreme_action_signal() -> None:
    torch.manual_seed(23)
    model = EntityGraphNet(
        _config(
            legal_action_value_residual=True,
            legal_action_value_set_statistics=True,
        )
    ).eval()
    model.action_encoder = _FirstActionScalar(model.config.hidden_size)
    _make_value_readout_depend_on_first_hidden_dimension(model)

    batch = _repeat_first_batch_row(_batch(batch_size=2, action_width=3))
    batch["legal_action_context"].zero_()
    batch["legal_action_tokens"].zero_()
    # Both sets have mean zero and the same count. Only the second set has a
    # rare extreme action, which a mean-only affordance cannot represent.
    batch["legal_action_tokens"][1, :, 0] = torch.tensor([-1.0, -1.0, 2.0])

    with torch.no_grad():
        aliased = model(batch)["value"]
        model.legal_action_value_max_proj.weight[0, 0] = 1.0
        values = model(batch)["value"]
    torch.testing.assert_close(aliased[0], aliased[1], rtol=0.0, atol=0.0)
    assert values[0].item() != pytest.approx(values[1].item())


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


def test_value_set_statistics_get_first_step_gradient() -> None:
    model = EntityGraphNet(
        _config(
            static_action_residual=True,
            legal_action_value_residual=True,
            legal_action_value_set_statistics=True,
        )
    ).train()
    batch = _batch(batch_size=3)
    batch["legal_action_mask"][0, -2:] = False
    batch["legal_action_mask"][1, -1:] = False
    target = torch.tensor([-1.0, 0.0, 1.0])

    torch.nn.functional.mse_loss(model(batch)["value"], target).backward()

    for module_name in (
        "legal_action_value_max_proj",
        "legal_action_value_count_proj",
        "legal_action_value_static_max_proj",
    ):
        gradient = getattr(model, module_name).weight.grad
        assert gradient is not None, module_name
        assert gradient.abs().sum().item() > 0.0, module_name


def test_padded_actions_do_not_change_legal_affordance_value() -> None:
    torch.manual_seed(11)
    model = EntityGraphNet(
        _config(
            static_action_residual=True,
            legal_action_value_residual=True,
            legal_action_value_set_statistics=True,
        )
    ).eval()
    with torch.no_grad():
        model.legal_action_value_residual_proj.weight.copy_(
            torch.eye(model.config.hidden_size)
        )
        model.legal_action_value_max_proj.weight.copy_(
            torch.eye(model.config.hidden_size)
        )
        model.legal_action_value_count_proj.weight.fill_(1.0)
        model.legal_action_value_static_max_proj.weight[
            :STATIC_ACTION_RESIDUAL_FEATURE_SIZE, :
        ] = torch.eye(STATIC_ACTION_RESIDUAL_FEATURE_SIZE)
    batch = _batch()
    batch["legal_action_mask"][:, -1] = False
    changed = {key: value.clone() for key, value in batch.items()}
    changed["legal_action_tokens"][:, -1] += 10_000.0
    changed["legal_action_context"][:, -1] -= 10_000.0
    changed["legal_action_static_features"][:, -1] += 10_000.0

    with torch.no_grad():
        original = model(batch)["value"]
        padded_changed = model(changed)["value"]
    torch.testing.assert_close(original, padded_changed, rtol=0.0, atol=0.0)


def test_policy_wrapper_preserves_legal_mask_for_value_affordance() -> None:
    torch.manual_seed(13)
    config = _config(legal_action_value_residual=True)
    policy = EntityGraphPolicy(
        config,
        np.zeros(
            (config.action_size, config.static_action_feature_size),
            dtype=np.float32,
        ),
        device="cpu",
    )
    policy.model.eval()
    with torch.no_grad():
        policy.model.legal_action_value_residual_proj.weight.copy_(
            torch.eye(config.hidden_size)
        )

    batch = _batch()
    batch["legal_action_mask"][:, -1] = False
    entity = {
        key: value.numpy()
        for key, value in batch.items()
        if key not in {"legal_action_context", "legal_action_static_features"}
    }
    legal_ids = np.tile(
        np.asarray([1, 2, 3, 4, -1], dtype=np.int64),
        (batch["legal_action_tokens"].shape[0], 1),
    )
    changed = {key: np.array(value, copy=True) for key, value in entity.items()}
    changed["legal_action_tokens"][:, -1] += 10_000.0
    changed_context = batch["legal_action_context"].numpy().copy()
    changed_context[:, -1] -= 10_000.0

    with torch.no_grad():
        original = policy.forward_legal_np(
            entity,
            legal_ids,
            batch["legal_action_context"].numpy(),
        )["value"]
        padded_changed = policy.forward_legal_np(
            changed,
            legal_ids,
            changed_context,
        )["value"]
    torch.testing.assert_close(original, padded_changed, rtol=0.0, atol=0.0)


def test_value_affordance_fails_closed_without_legal_mask() -> None:
    model = EntityGraphNet(_config(legal_action_value_residual=True)).eval()
    batch = _batch()
    del batch["legal_action_mask"]

    with pytest.raises(ValueError, match="requires the exact legal_action_mask"):
        model(batch)
