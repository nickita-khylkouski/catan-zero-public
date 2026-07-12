"""Parity tests for the split state-trunk/action-head inference API."""

from __future__ import annotations

import copy

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
    EntityGraphConfig,
    EntityGraphNet,
    event_batch_shape_telemetry,
)


def _config(**overrides) -> EntityGraphConfig:
    values = dict(
        action_size=64,
        static_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
        context_action_feature_size=CONTEXT_ACTION_FEATURE_SIZE,
        legal_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
        hidden_size=32,
        state_layers=2,
        attention_heads=4,
        dropout=0.0,
    )
    values.update(overrides)
    return EntityGraphConfig(**values)


def _batch(
    *,
    batch_size: int = 3,
    action_width: int = 5,
    event_width: int = 16,
    live_event_width: int = 7,
) -> dict:
    generator = torch.Generator().manual_seed(20260709)
    counts = {
        "hex": (19, HEX_FEATURE_SIZE),
        "vertex": (54, VERTEX_FEATURE_SIZE),
        "edge": (72, EDGE_FEATURE_SIZE),
        "player": (4, PLAYER_FEATURE_SIZE),
        "global": (1, GLOBAL_FEATURE_SIZE),
        "event": (event_width, EVENT_FEATURE_SIZE),
    }
    batch = {}
    for name, (count, feature_width) in counts.items():
        batch[f"{name}_tokens"] = torch.randn(
            batch_size,
            count,
            feature_width,
            generator=generator,
        )
        if name != "global":
            batch[f"{name}_mask"] = torch.ones(
                batch_size,
                count,
                dtype=torch.bool,
            )
    batch["event_mask"][:, live_event_width:] = False
    # Give different rows shorter live prefixes while keeping one row at the
    # batch-wide required width.
    if batch_size > 1 and live_event_width > 1:
        batch["event_mask"][1, live_event_width - 2 :] = False

    batch["legal_action_tokens"] = torch.randn(
        batch_size,
        action_width,
        LEGAL_ACTION_FEATURE_SIZE,
        generator=generator,
    )
    batch["legal_action_context"] = torch.randn(
        batch_size,
        action_width,
        CONTEXT_ACTION_FEATURE_SIZE,
        generator=generator,
    )
    targets = -torch.ones(batch_size, action_width, 4, dtype=torch.long)
    targets[:, :, 1] = torch.arange(action_width).remainder(54).view(1, -1)
    batch["legal_action_target_ids"] = targets
    return batch


def _assert_outputs_equal(left: dict, right: dict, *, exact: bool) -> None:
    assert left.keys() == right.keys()
    for key in left:
        if exact:
            assert torch.equal(left[key], right[key]), key
        else:
            torch.testing.assert_close(left[key], right[key], rtol=1e-6, atol=1e-6)


def test_forward_is_exact_composition_of_state_and_action_apis():
    model = EntityGraphNet(
        _config(
            action_target_gather=True,
            action_cross_attention_layers=1,
            value_attention_pool=True,
            value_uncertainty_head=True,
            value_categorical_bins=9,
            edge_policy_head=True,
            aux_subgoal_heads=True,
        )
    ).eval()
    batch = _batch()

    with torch.no_grad():
        combined = model(batch, return_q=True)
        encoded_state = model.encode_state(batch)
        split = model.score_actions(encoded_state, batch, return_q=True)

    _assert_outputs_equal(combined, split, exact=True)
    assert len(encoded_state) == 3
    assert encoded_state[0].shape[0] == batch["hex_tokens"].shape[0]


def test_zero_init_target_gather_gets_a_nonzero_first_step_gradient():
    """Exact warm-start must not make the topology branch permanently inert."""
    model = EntityGraphNet(_config(action_target_gather=True)).train()
    batch = _batch(batch_size=3, action_width=5)
    outputs = model(batch)
    target = torch.tensor([0, 1, 2], dtype=torch.long)
    torch.nn.functional.cross_entropy(outputs["logits"], target).backward()

    projection = model.target_gather_proj[1]
    # Its zero weight blocks first-step gradients *through* the new path into
    # the trunk (the exact-function warm-start contract), but the projection
    # itself must learn immediately so topology gradients reach the trunk on
    # subsequent steps.
    assert torch.count_nonzero(projection.weight).item() == 0
    assert projection.weight.grad is not None
    assert projection.weight.grad.abs().sum().item() > 0.0


def test_event_tail_crop_preserves_outputs_with_all_consumers_enabled():
    model = EntityGraphNet(
        _config(
            action_target_gather=True,
            action_cross_attention_layers=1,
            value_attention_pool=True,
            edge_policy_head=True,
        )
    ).eval()
    batch = _batch(event_width=16, live_event_width=7)

    with torch.no_grad():
        full = model(batch, return_q=True)
        cropped = model(batch, return_q=True, event_token_limit=7)
        full_tokens, _, _ = model.encode_state(batch)
        cropped_tokens, _, _ = model.encode_state(batch, event_token_limit=7)

    # Sequence-length changes can select a different attention kernel, so the
    # opt-in path promises numerical equivalence rather than bit identity.
    _assert_outputs_equal(full, cropped, exact=False)
    assert full_tokens.shape[1] - cropped_tokens.shape[1] == 9


def test_fully_masked_event_history_can_be_removed_entirely():
    model = EntityGraphNet(_config()).eval()
    batch = _batch(event_width=64, live_event_width=0)

    with torch.no_grad():
        full = model(batch)
        cropped = model(batch, event_token_limit=0)
        cropped_tokens, cropped_mask, _state = model.encode_state(
            batch, event_token_limit=0
        )

    _assert_outputs_equal(full, cropped, exact=False)
    # CLS + hex + vertex + edge + player + global = 151 fixed tokens.
    assert cropped_tokens.shape[1] == 151
    assert cropped_mask.shape[1] == 151


def test_event0_preserves_logits_value_loss_and_gradients_for_ddp_sized_batch():
    """The training optimization must preserve the complete learner signal."""

    full_model = EntityGraphNet(
        _config(
            action_target_gather=True,
            action_cross_attention_layers=1,
            value_attention_pool=True,
        )
    ).train()
    cropped_model = copy.deepcopy(full_model).train()
    full_batch = _batch(
        batch_size=8,
        action_width=7,
        event_width=64,
        live_event_width=0,
    )
    cropped_batch = {
        key: (
            value[:, :0]
            if key in {"event_tokens", "event_mask"}
            else value.clone()
        )
        for key, value in full_batch.items()
    }
    targets = torch.arange(8, dtype=torch.long).remainder(7)
    value_targets = torch.linspace(-1.0, 1.0, 8)

    full = full_model(full_batch)
    cropped = cropped_model(cropped_batch)
    full_loss = torch.nn.functional.cross_entropy(full["logits"], targets) + 0.25 * (
        full["value"] - value_targets
    ).square().mean()
    cropped_loss = torch.nn.functional.cross_entropy(
        cropped["logits"], targets
    ) + 0.25 * (cropped["value"] - value_targets).square().mean()
    full_loss.backward()
    cropped_loss.backward()

    torch.testing.assert_close(full["logits"], cropped["logits"], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(full["value"], cropped["value"], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(full_loss, cropped_loss, rtol=1e-6, atol=1e-6)
    cropped_parameters = dict(cropped_model.named_parameters())
    for name, parameter in full_model.named_parameters():
        left = parameter.grad
        right = cropped_parameters[name].grad
        if left is None or right is None:
            # The cropped event encoder is correctly disconnected. The full
            # all-masked path may expose either None or an exact zero gradient.
            present = left if left is not None else right
            if present is not None:
                assert torch.count_nonzero(present).item() == 0, name
            continue
        torch.testing.assert_close(left, right, rtol=2e-5, atol=2e-6, msg=name)


def test_event0_short_circuits_empty_event_encoder_without_changing_output():
    model = EntityGraphNet(_config()).eval()
    empty_batch = _batch(event_width=0, live_event_width=0)
    nonempty_batch = _batch(event_width=1, live_event_width=0)
    calls = 0

    def count_call(_module, _inputs, _output):
        nonlocal calls
        calls += 1

    handle = model.event_encoder.register_forward_hook(count_call)
    try:
        with torch.no_grad():
            empty_output = model(empty_batch, return_q=True)
            assert calls == 0
            empty_tokens, empty_mask, _state = model.encode_state(empty_batch)
            assert calls == 0
            nonempty_output = model(nonempty_batch, return_q=True)
            assert calls == 1
    finally:
        handle.remove()

    assert empty_tokens.shape[1] == 151
    assert empty_mask.shape[1] == 151
    assert empty_output["logits"].shape == nonempty_output["logits"].shape


def test_event_tail_crop_rejects_live_tokens_and_invalid_limits():
    model = EntityGraphNet(_config()).eval()
    batch = _batch(event_width=12, live_event_width=7)

    with pytest.raises(ValueError, match="remove at least one unmasked"):
        model.encode_state(batch, event_token_limit=6)
    with pytest.raises(ValueError, match="within the padded event width"):
        model.encode_state(batch, event_token_limit=13)
    with pytest.raises(ValueError, match="within the padded event width"):
        model.encode_state(batch, event_token_limit=-1)
    with pytest.raises(TypeError, match="must be an integer"):
        model.encode_state(batch, event_token_limit=6.9)
    with pytest.raises(TypeError, match="not bool"):
        model.encode_state(batch, event_token_limit=True)


def test_event_shape_telemetry_finds_smallest_safe_batch_prefix():
    mask = np.asarray(
        [
            [True, True, True, True, True, False, False, False],
            [True, True, True, False, False, False, False, False],
            [True, True, True, True, False, False, False, False],
        ],
        dtype=np.bool_,
    )

    telemetry = event_batch_shape_telemetry(mask)

    assert telemetry == {
        "batch_size": 3,
        "padded_event_width": 8,
        "required_event_width": 5,
        "min_row_event_width": 3,
        "max_row_event_width": 5,
        "active_event_tokens": 12,
        "event_token_utilization": 0.5,
    }


def test_event_shape_telemetry_handles_empty_batch_and_validates_rank():
    assert (
        event_batch_shape_telemetry(np.zeros((0, 64), dtype=np.bool_))[
            "required_event_width"
        ]
        == 0
    )
    with pytest.raises(ValueError, match="rank 2"):
        event_batch_shape_telemetry(np.zeros((64,), dtype=np.bool_))
