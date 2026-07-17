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
    EVENT_POSITION_OFFSET_KEY,
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


def test_late_value_tower_clone_is_function_preserving_at_activation():
    base = EntityGraphNet(_config(state_layers=3)).eval()
    split = EntityGraphNet(
        _config(state_layers=3, value_tower_split_layers=2)
    ).eval()
    missing, unexpected = split.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected
    assert missing
    assert all(
        key.startswith(("value_blocks.", "value_state_norm."))
        for key in missing
    )
    split.initialize_value_tower_from_policy()
    batch = _batch()

    with torch.no_grad():
        base_outputs = base(batch, return_q=True)
        split_outputs = split(batch, return_q=True)

    _assert_outputs_equal(base_outputs, split_outputs, exact=True)
    assert len(split.encode_state(batch)) == 5


def test_zero_init_action_cross_preserves_train_dropout_rng_and_value_path():
    """An identity action adapter must not perturb later value-dropout masks."""
    torch.manual_seed(17)
    base = EntityGraphNet(
        _config(state_layers=3, value_tower_split_layers=1, dropout=0.2)
    )
    upgraded = EntityGraphNet(
        _config(
            state_layers=3,
            value_tower_split_layers=1,
            dropout=0.2,
            action_cross_attention_layers=1,
        )
    )
    missing, unexpected = upgraded.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected
    assert all(key.startswith("action_cross_blocks.") for key in missing)
    upgraded.initialize_value_tower_from_policy()
    base.train()
    upgraded.train()
    batch = _batch()

    torch.manual_seed(20260717)
    base_output = base(batch, return_q=True)
    base_next_random = torch.rand(8)
    torch.manual_seed(20260717)
    upgraded_output = upgraded(batch, return_q=True)
    upgraded_next_random = torch.rand(8)

    _assert_outputs_equal(base_output, upgraded_output, exact=True)
    assert torch.equal(base_next_random, upgraded_next_random)


def test_split1_inference_cls_suffix_matches_loaded_full_block_parameters(
    monkeypatch,
):
    """The inference shortcut must preserve policy and scalar-value outputs."""

    optimized = EntityGraphNet(
        _config(state_layers=3, value_tower_split_layers=1, dropout=0.0)
    ).eval()
    reference = EntityGraphNet(copy.deepcopy(optimized.config))
    reference.load_state_dict(optimized.state_dict(), strict=True)
    reference.eval()
    # Force only the reference through the historical full-token suffix while
    # retaining identical eval-mode kernels everywhere else in the model.
    monkeypatch.setattr(reference, "_use_cls_only_value_suffix", lambda: False)
    batch = _batch(batch_size=3, action_width=5, event_width=16)

    with torch.no_grad():
        optimized_outputs = optimized(batch, return_q=True)
        reference_outputs = reference(batch, return_q=True)

    assert torch.equal(optimized_outputs["logits"], reference_outputs["logits"])
    for key in ("value", "q_values"):
        torch.testing.assert_close(
            optimized_outputs[key],
            reference_outputs[key],
            rtol=1e-6,
            atol=1e-6,
        )


def test_v7_compatibility_route_preserves_legacy_initial_road_input():
    """A corrected V6 road score must not overwrite the inherited input.

    V5's action encoder learned an endpoint-production feature in context slot
    16.  V6 repurposed that slot for a two-hop score.  The compatibility route
    feeds the reconstructed V5 feature to the inherited encoder and exposes the
    corrected score only through a new zero-initialized residual.
    """
    model = EntityGraphNet(
        _config(
            v6_compatibility_preserving_inputs=True,
            action_cross_attention_layers=1,
        )
    ).eval()
    batch = _batch(batch_size=1, action_width=2)
    batch.update(
        {
            "hex_vertex_ids": torch.zeros(1, 19, 6, dtype=torch.long),
            "hex_edge_ids": torch.zeros(1, 19, 6, dtype=torch.long),
            "edge_vertex_ids": -torch.ones(1, 72, 2, dtype=torch.long),
            "event_target_ids": -torch.ones(1, 16, 4, dtype=torch.long),
        }
    )
    batch["edge_vertex_ids"][0, 7] = torch.tensor((3, 11))
    batch["legal_action_target_ids"][:, :, 2] = -1
    batch["legal_action_target_ids"][0, 0, 2] = 7
    batch["legal_action_context"].zero_()
    batch["legal_action_context"][0, 0, 12] = 1.0  # initial-road row
    batch["legal_action_context"][0, 0, 16] = 0.91  # V6 two-hop score
    batch["vertex_tokens"].zero_()
    batch["vertex_tokens"][0, 3, 6] = 1.0
    batch["vertex_tokens"][0, 3, 9] = 4.0 / 18.0
    batch["vertex_tokens"][0, 11, 6] = 1.0
    batch["vertex_tokens"][0, 11, 9] = 13.0 / 18.0

    seen: list[torch.Tensor] = []
    hook = model.action_encoder.register_forward_pre_hook(
        lambda _module, args: seen.append(args[0].detach().clone())
    )
    try:
        with torch.no_grad():
            model(batch)
    finally:
        hook.remove()

    assert len(seen) == 1
    legacy_context_offset = LEGAL_ACTION_FEATURE_SIZE + 16
    # Recover integer pips before division so the inherited float32 context is
    # exact rather than inheriting vertex-token float16 quantization.
    assert seen[0][0, 0, legacy_context_offset].item() == np.float32(
        13.0 / 18.0
    ).item()
    # Non-initial-road actions retain their old context feature unchanged.
    assert seen[0][0, 1, legacy_context_offset].item() == pytest.approx(0.0)
    assert torch.count_nonzero(model.v6_initial_road_residual.weight) == 0

    model.zero_grad(set_to_none=True)
    model(batch)["logits"].sum().backward()
    residual_gradient = model.v6_initial_road_residual.weight.grad
    assert residual_gradient is not None
    assert torch.isfinite(residual_gradient).all()
    assert torch.count_nonzero(residual_gradient) > 0


def test_v7_compatibility_route_requires_live_action_cross_attention():
    with pytest.raises(ValueError, match="require at least one Transformer action"):
        EntityGraphNet(_config(v6_compatibility_preserving_inputs=True))


def test_action_decoder_attends_only_live_processed_public_history():
    from catan_zero.rl.ordered_history import ORDERED_ATTENTION_V2

    model = EntityGraphNet(
        _config(
            action_cross_attention_layers=1,
            v6_compatibility_preserving_inputs=True,
            meaningful_public_history=True,
            meaningful_public_history_pooling=ORDERED_ATTENTION_V2,
            meaningful_public_history_target_gather=True,
        )
    ).eval()
    batch = _batch(
        batch_size=1,
        action_width=3,
        event_width=16,
        live_event_width=4,
    )
    batch["event_mask"].zero_()
    batch["event_mask"][:, -4:] = True
    batch["event_target_ids"] = -torch.ones(1, 16, 4, dtype=torch.long)
    batch["event_target_ids"][0, -4:, 0] = torch.arange(4)
    batch["edge_vertex_ids"] = torch.zeros(1, 72, 2, dtype=torch.long)
    with torch.no_grad():
        # Isolate the decoder from the pre-existing pooled global-history path.
        model.meaningful_history_residual_gate.zero_()
        model.meaningful_history_ordered_gate.zero_()
        # Open the exact-identity decoder for this causal test.
        for name, parameter in model.action_cross_blocks.named_parameters():
            if name.endswith(("attn.out_proj.weight", "ff.3.weight")):
                parameter.copy_(torch.randn_like(parameter))
        model.meaningful_history_sequence.position_embedding.copy_(
            torch.randn_like(
                model.meaningful_history_sequence.position_embedding
            )
        )
        model.meaningful_history_target_proj[1].weight.copy_(
            torch.eye(model.config.hidden_size)
        )
        baseline = model(batch)["logits"]

        live_changed = copy.deepcopy(batch)
        live_changed["event_tokens"][0, -1, 0] += 3.0
        live_logits = model(live_changed)["logits"]

        padded_changed = copy.deepcopy(batch)
        padded_changed["event_tokens"][0, 0] += 1000.0
        padded_logits = model(padded_changed)["logits"]

        reordered = copy.deepcopy(batch)
        reordered["event_tokens"][:, [-2, -1]] = reordered["event_tokens"][
            :, [-1, -2]
        ]
        reordered["event_target_ids"][:, [-2, -1]] = reordered[
            "event_target_ids"
        ][:, [-1, -2]]
        reordered_logits = model(reordered)["logits"]

        retargeted = copy.deepcopy(batch)
        retargeted["event_target_ids"][0, -1, 0] = 8
        retargeted_logits = model(retargeted)["logits"]

    assert torch.max(torch.abs(live_logits - baseline)).item() > 0.0
    assert torch.equal(padded_logits, baseline)
    assert torch.max(torch.abs(reordered_logits - baseline)).item() > 0.0
    assert torch.max(torch.abs(retargeted_logits - baseline)).item() > 0.0


def test_bottleneck_v7_decoder_attends_to_processed_history_without_v6_adapter():
    from catan_zero.rl.ordered_history import ORDERED_ATTENTION_V2

    model = EntityGraphNet(
        _config(
            action_cross_attention_layers=1,
            action_cross_attention_bottleneck=80,
            meaningful_public_history=True,
            meaningful_public_history_pooling=ORDERED_ATTENTION_V2,
            meaningful_public_history_target_gather=True,
        )
    ).eval()
    batch = _batch(
        batch_size=1,
        action_width=3,
        event_width=16,
        live_event_width=4,
    )
    batch["event_target_ids"] = -torch.ones(1, 16, 4, dtype=torch.long)
    batch["event_target_ids"][0, :4, 0] = torch.arange(4)
    with torch.no_grad():
        model.meaningful_history_residual_gate.zero_()
        model.meaningful_history_ordered_gate.zero_()
        for name, parameter in model.action_cross_blocks.named_parameters():
            if name.endswith(("attn.out_proj.weight", "ff.3.weight")):
                parameter.copy_(torch.randn_like(parameter))
        model.meaningful_history_sequence.position_embedding.copy_(
            torch.randn_like(
                model.meaningful_history_sequence.position_embedding
            )
        )
        model.meaningful_history_target_proj[1].weight.copy_(
            torch.eye(model.config.hidden_size)
        )
        baseline = model(batch)["logits"]
        live_changed = copy.deepcopy(batch)
        live_changed["event_tokens"][0, 0, 0] += 3.0
        live_logits = model(live_changed)["logits"]
        padded_changed = copy.deepcopy(batch)
        padded_changed["event_tokens"][0, -1] += 1000.0
        padded_logits = model(padded_changed)["logits"]

    assert torch.max(torch.abs(live_logits - baseline)).item() > 0.0
    assert torch.equal(padded_logits, baseline)


def test_action_history_positions_survive_trailing_padding_crop():
    from catan_zero.rl.ordered_history import ORDERED_ATTENTION_V2

    model = EntityGraphNet(
        _config(
            action_cross_attention_layers=1,
            v6_compatibility_preserving_inputs=True,
            meaningful_public_history=True,
            meaningful_public_history_pooling=ORDERED_ATTENTION_V2,
            meaningful_public_history_target_gather=True,
        )
    ).eval()
    batch = _batch(
        batch_size=1,
        action_width=3,
        event_width=16,
        live_event_width=4,
    )
    batch["event_target_ids"] = -torch.ones(1, 16, 4, dtype=torch.long)
    batch["event_target_ids"][0, :4, 0] = torch.arange(4)
    batch["edge_vertex_ids"] = torch.zeros(1, 72, 2, dtype=torch.long)
    with torch.no_grad():
        model.meaningful_history_residual_gate.fill_(0.2)
        model.meaningful_history_ordered_gate.fill_(0.3)
        for name, parameter in model.action_cross_blocks.named_parameters():
            if name.endswith(("attn.out_proj.weight", "ff.3.weight")):
                parameter.copy_(torch.randn_like(parameter))
        model.meaningful_history_sequence.position_embedding.copy_(
            torch.randn_like(
                model.meaningful_history_sequence.position_embedding
            )
        )
        model.meaningful_history_target_proj[1].weight.copy_(
            torch.eye(model.config.hidden_size)
        )
        full = model(batch, return_q=True)
        cropped = model(batch, return_q=True, event_token_limit=4)
        physically_cropped = {
            key: (
                value[:, :4]
                if key in {"event_tokens", "event_mask", "event_target_ids"}
                else value
            )
            for key, value in batch.items()
        }
        physically_cropped[EVENT_POSITION_OFFSET_KEY] = torch.full(
            (1,),
            model.meaningful_public_history_normalization
            - int(batch["event_tokens"].shape[1]),
            dtype=torch.long,
        )
        transported = model(physically_cropped, return_q=True)

    for key in full:
        torch.testing.assert_close(full[key], cropped[key], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(
            full[key], transported[key], rtol=1e-6, atol=1e-6
        )


@pytest.mark.parametrize(
    ("split_layers", "value_attention_pool", "training", "expect_cls_only"),
    (
        (1, False, False, True),
        (1, False, True, False),
        (1, True, False, False),
        (2, False, False, False),
    ),
)
def test_cls_only_value_suffix_is_guarded_to_safe_split1_inference(
    monkeypatch,
    split_layers: int,
    value_attention_pool: bool,
    training: bool,
    expect_cls_only: bool,
):
    model = EntityGraphNet(
        _config(
            state_layers=3,
            value_tower_split_layers=split_layers,
            value_attention_pool=value_attention_pool,
            dropout=0.0,
        )
    )
    model.train(training)
    cls_calls = 0
    original = model.value_blocks[0].forward_cls

    def counted_cls(*args, **kwargs):
        nonlocal cls_calls
        cls_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(model.value_blocks[0], "forward_cls", counted_cls)
    with torch.no_grad():
        model(_batch(batch_size=2, action_width=4, event_width=8))

    assert (cls_calls == 1) is expect_cls_only


def test_late_value_tower_isolates_policy_and_value_suffix_gradients():
    model = EntityGraphNet(
        _config(state_layers=2, value_tower_split_layers=1)
    ).train()
    batch = _batch(batch_size=3, action_width=5)

    model.zero_grad(set_to_none=True)
    model(batch)["value"].square().mean().backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum().item() > 0.0
        for parameter in model.value_blocks.parameters()
    )
    assert all(parameter.grad is None for parameter in model.blocks[-1].parameters())

    model.zero_grad(set_to_none=True)
    model(batch)["logits"].square().mean().backward()
    assert all(parameter.grad is None for parameter in model.value_blocks.parameters())
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum().item() > 0.0
        for parameter in model.blocks[-1].parameters()
    )


def test_split_value_gradient_scale_zero_stops_shared_prefix_not_private_tower():
    model = EntityGraphNet(
        _config(state_layers=2, value_tower_split_layers=1)
    ).train()
    batch = _batch(batch_size=3, action_width=5)

    model.zero_grad(set_to_none=True)
    model(batch, value_trunk_grad_scale=0.0)["value"].square().mean().backward()

    assert all(parameter.grad is None for parameter in model.blocks[0].parameters())
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum().item() > 0.0
        for parameter in model.value_blocks.parameters()
    )


def test_split_value_gradient_scale_point_one_keeps_private_tower_full_strength():
    """The V7 routing scale applies only before the private value suffix."""
    control = EntityGraphNet(
        _config(state_layers=2, value_tower_split_layers=1)
    ).train()
    treatment = copy.deepcopy(control)
    batch = _batch(batch_size=3, action_width=5)

    results = []
    for model, scale in ((control, 1.0), (treatment, 0.1)):
        model.zero_grad(set_to_none=True)
        output = model(batch, value_trunk_grad_scale=scale)["value"]
        output.sum().backward()
        results.append(
            (
                output.detach().clone(),
                {
                    name: parameter.grad.detach().clone()
                    for name, parameter in model.named_parameters()
                    if parameter.grad is not None
                },
            )
        )

    (control_value, control_grads), (treatment_value, treatment_grads) = results
    assert torch.equal(treatment_value, control_value)
    private_value_prefixes = (
        "value_blocks.",
        "value_state_norm.",
        "value_head.",
    )
    for name, control_grad in control_grads.items():
        expected = (
            control_grad
            if name.startswith(private_value_prefixes)
            else 0.1 * control_grad
        )
        torch.testing.assert_close(
            treatment_grads[name],
            expected,
            rtol=2e-5,
            atol=1e-7,
            msg=name,
        )


def test_split_value_history_target_gather_does_not_leak_into_policy_suffix():
    """The history side input must respect the same late-tower boundary."""
    model = EntityGraphNet(
        _config(
            state_layers=3,
            value_tower_split_layers=1,
            meaningful_public_history=True,
            event_history_limit=16,
            meaningful_public_history_target_gather=True,
        )
    ).train()
    batch = _batch(
        batch_size=3,
        action_width=5,
        event_width=16,
        live_event_width=7,
    )
    event_targets = -torch.ones(3, 16, 4, dtype=torch.long)
    event_targets[:, :, 1] = torch.arange(16).remainder(54)
    batch["event_target_ids"] = event_targets
    with torch.no_grad():
        model.meaningful_history_residual_gate.fill_(0.1)
        model.meaningful_history_target_proj[1].weight.copy_(
            torch.eye(model.config.hidden_size)
        )

    model.zero_grad(set_to_none=True)
    model(batch, value_trunk_grad_scale=0.25)["value"].sum().backward()

    # Value still learns from the shared prefix and its private suffix, but the
    # final policy block is no longer a hidden route around the split.
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum().item() > 0.0
        for parameter in model.blocks[1].parameters()
    )
    assert all(parameter.grad is None for parameter in model.blocks[2].parameters())
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum().item() > 0.0
        for parameter in model.value_blocks.parameters()
    )


def test_policy_value_only_output_selection_is_bit_identical_and_skips_final_vp():
    model = EntityGraphNet(_config()).eval()
    batch = _batch()
    calls = 0

    def count_call(_module, _inputs, _output):
        nonlocal calls
        calls += 1

    handle = model.final_vp_head.register_forward_hook(count_call)
    try:
        with torch.no_grad():
            full = model(batch, return_final_vp=True)
            policy_value = model(batch, return_final_vp=False)
    finally:
        handle.remove()

    assert calls == 1
    assert "final_vp" in full
    assert "final_vp" not in policy_value
    assert torch.equal(full["logits"], policy_value["logits"])
    assert torch.equal(full["value"], policy_value["value"])


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
