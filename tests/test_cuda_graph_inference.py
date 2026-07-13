"""CPU-side contract tests for the opt-in CUDA Graph inference runner."""

from __future__ import annotations

from types import SimpleNamespace

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
    PUBLIC_AWARD_FEATURE_CONTRACT_AUTHORITATIVE,
    PUBLIC_AWARD_FEATURE_CONTRACT_LEGACY_ZERO,
    EntityGraphConfig,
    EntityGraphNet,
)
from catan_zero.search.cuda_graph_inference import (  # noqa: E402
    CudaGraphInferenceConfig,
    CudaGraphInferenceRunner,
)


def _policy(**overrides):
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
    config = EntityGraphConfig(**values)
    return SimpleNamespace(
        config=config,
        model=EntityGraphNet(config).eval(),
        device=torch.device("cpu"),
        public_award_feature_contract=PUBLIC_AWARD_FEATURE_CONTRACT_AUTHORITATIVE,
    )


def _batch(batch_size=3, legal_width=5, event_width=8, live_events=0):
    rng = np.random.default_rng(20260709)
    entity = {}
    for name, count, width in (
        ("hex", 19, HEX_FEATURE_SIZE),
        ("vertex", 54, VERTEX_FEATURE_SIZE),
        ("edge", 72, EDGE_FEATURE_SIZE),
        ("player", 4, PLAYER_FEATURE_SIZE),
        ("global", 1, GLOBAL_FEATURE_SIZE),
        ("event", event_width, EVENT_FEATURE_SIZE),
    ):
        entity[f"{name}_tokens"] = rng.normal(size=(batch_size, count, width)).astype(
            np.float16
        )
        if name != "global":
            entity[f"{name}_mask"] = np.ones((batch_size, count), dtype=np.bool_)
    entity["event_mask"][:, live_events:] = False
    entity["legal_action_tokens"] = rng.normal(
        size=(batch_size, legal_width, LEGAL_ACTION_FEATURE_SIZE)
    ).astype(np.float16)
    entity["legal_action_target_ids"] = np.full(
        (batch_size, legal_width, 4), -1, dtype=np.int16
    )
    entity["legal_action_target_ids"][:, :, 1] = (
        np.arange(legal_width, dtype=np.int16) % 54
    )
    legal_ids = np.tile(np.arange(legal_width, dtype=np.int64), (batch_size, 1))
    legal_ids[-1, -1] = -1
    entity["legal_action_target_ids"][legal_ids < 0] = -1
    entity["legal_action_mask"] = legal_ids >= 0
    entity["hex_vertex_ids"] = np.broadcast_to(
        np.arange(19 * 6, dtype=np.int16).reshape(1, 19, 6) % 54,
        (batch_size, 19, 6),
    ).copy()
    entity["hex_edge_ids"] = np.broadcast_to(
        np.arange(19 * 6, dtype=np.int16).reshape(1, 19, 6) % 72,
        (batch_size, 19, 6),
    ).copy()
    edges = np.arange(72, dtype=np.int16).reshape(1, 72, 1)
    edge_vertices = np.concatenate((edges % 54, (edges + 1) % 54), axis=2)
    entity["edge_vertex_ids"] = np.broadcast_to(
        edge_vertices, (batch_size, 72, 2)
    ).copy()
    entity["event_target_ids"] = np.full(
        (batch_size, event_width, 4), -1, dtype=np.int16
    )
    context = rng.normal(
        size=(batch_size, legal_width, CONTEXT_ACTION_FEATURE_SIZE)
    ).astype(np.float32)
    return entity, legal_ids, context


def test_bucket_selection_uses_close_ceiling_and_has_bounded_fallback():
    runner = CudaGraphInferenceRunner(
        _policy(),
        CudaGraphInferenceConfig(batch_buckets=(8, 16, 24)),
    )
    assert runner.selected_batch_bucket(1) == 8
    assert runner.selected_batch_bucket(9) == 16
    assert runner.selected_batch_bucket(17) == 24
    assert runner.selected_batch_bucket(25) is None


def test_enabled_cpu_path_falls_back_eager_and_trims_outputs():
    policy = _policy()
    entity, legal_ids, context = _batch()
    runner = CudaGraphInferenceRunner(
        policy,
        CudaGraphInferenceConfig(
            enabled=True,
            batch_buckets=(2, 4, 8),
            event_token_limit=0,
        ),
    )

    outputs = runner.forward_legal_np(
        entity,
        legal_ids,
        context,
        return_q=True,
    )

    assert runner.last_path == "eager_fallback"
    assert "requires a CUDA device" in runner.last_fallback_reason
    assert runner.graph_count == 0
    assert outputs["logits"].shape == (3, 5)
    assert outputs["q_values"].shape == (3, 5)
    assert outputs["value"].shape == (3,)
    assert outputs["logits"][-1, -1].item() == -1.0e9

    torch_batch = {
        key: torch.as_tensor(value)
        for key, value in entity.items()
        if key not in {"legal_action_tokens", "event_tokens", "event_mask"}
    }
    torch_batch["event_tokens"] = torch.as_tensor(entity["event_tokens"][:, :0])
    torch_batch["event_mask"] = torch.as_tensor(entity["event_mask"][:, :0])
    torch_batch["legal_action_tokens"] = torch.as_tensor(entity["legal_action_tokens"])
    torch_batch["legal_action_context"] = torch.as_tensor(context)
    with torch.no_grad():
        expected = policy.model(torch_batch, return_q=True)
        expected["logits"] = expected["logits"].masked_fill(
            torch.as_tensor(legal_ids) < 0, -1.0e9
        )
    for key in outputs:
        torch.testing.assert_close(outputs[key], expected[key], rtol=0.0, atol=0.0)


def test_legacy_award_bridge_also_applies_on_cuda_runner_bypass() -> None:
    policy = _policy()
    policy.public_award_feature_contract = PUBLIC_AWARD_FEATURE_CONTRACT_LEGACY_ZERO
    entity, legal_ids, context = _batch()
    entity["player_tokens"][..., 12] = 1.0
    original = entity["player_tokens"].copy()
    runner = CudaGraphInferenceRunner(policy, CudaGraphInferenceConfig(enabled=False))

    actual = runner.forward_legal_np(entity, legal_ids, context)

    manual = dict(entity)
    manual_players = entity["player_tokens"].copy()
    manual_players[..., 12] = 0.0
    manual["player_tokens"] = manual_players
    policy.public_award_feature_contract = (
        PUBLIC_AWARD_FEATURE_CONTRACT_AUTHORITATIVE
    )
    expected = CudaGraphInferenceRunner(
        policy, CudaGraphInferenceConfig(enabled=False)
    ).forward_legal_np(manual, legal_ids, context)
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(entity["player_tokens"], original)


def test_event_token_limit_zero_rejects_live_events_before_fallback():
    entity, legal_ids, context = _batch(live_events=1)
    runner = CudaGraphInferenceRunner(
        _policy(),
        CudaGraphInferenceConfig(enabled=True, event_token_limit=0),
    )
    with pytest.raises(ValueError, match="remove at least one unmasked"):
        runner.forward_legal_np(entity, legal_ids, context)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda entity, _context: entity.pop("legal_action_mask"),
            "missing entity batch field legal_action_mask",
        ),
        (
            lambda entity, _context: entity.__setitem__(
                "hex_mask", entity["hex_mask"][:, :-1]
            ),
            "hex_mask shape",
        ),
        (
            lambda entity, _context: entity.__setitem__(
                "hex_tokens", entity["hex_tokens"][:, :, :-1]
            ),
            "hex_tokens width",
        ),
        (
            None,
            "legal_action_context width mismatch",
        ),
    ),
)
def test_runner_preserves_canonical_policy_shape_validation(mutation, message):
    entity, legal_ids, context = _batch()
    if mutation is None:
        context = context[:, :, :-1]
    else:
        mutation(entity, context)
    runner = CudaGraphInferenceRunner(
        _policy(),
        CudaGraphInferenceConfig(enabled=True, event_token_limit=0),
    )

    with pytest.raises(ValueError, match=message):
        runner.forward_legal_np(entity, legal_ids, context)


def test_runner_preserves_policy_metadata_and_target_aware_action_head():
    policy = _policy(action_target_gather=True, edge_policy_head=True)
    policy.action_size = 64
    entity, legal_ids, context = _batch(batch_size=2, legal_width=4)
    runner = CudaGraphInferenceRunner(policy)

    outputs = runner.forward_legal_np(entity, legal_ids, context)

    assert runner.config is policy.config
    assert runner.action_size == policy.action_size
    assert runner.runner_config.enabled is False
    assert outputs["logits"].shape == (2, 4)


def test_topology_adapter_survives_cuda_graph_eager_fallback_and_event_crop():
    policy = _policy(topology_residual_adapter=True, action_target_gather=True)
    entity, legal_ids, context = _batch(
        batch_size=2, legal_width=4, event_width=8, live_events=2
    )
    entity["event_target_ids"][:, 0, 1] = 3
    runner = CudaGraphInferenceRunner(
        policy,
        CudaGraphInferenceConfig(enabled=True, event_token_limit=2),
    )

    outputs = runner.forward_legal_np(entity, legal_ids, context, return_q=True)

    assert runner.last_path == "eager_fallback"
    assert "requires a CUDA device" in runner.last_fallback_reason
    assert outputs["logits"].shape == (2, 4)
    assert outputs["q_values"].shape == (2, 4)


@pytest.mark.parametrize(
    ("key", "index", "bad_value", "width"),
    (
        ("hex_vertex_ids", (0, 0, 0), 54, 54),
        ("hex_edge_ids", (0, 0, 0), 72, 72),
        ("edge_vertex_ids", (0, 0, 0), -2, 54),
        ("event_target_ids", (0, 0, 2), 72, 72),
    ),
)
def test_topology_runner_rejects_out_of_range_incidence_ids(
    key, index, bad_value, width
):
    entity, legal_ids, context = _batch(batch_size=1, legal_width=2)
    entity[key][index] = bad_value
    runner = CudaGraphInferenceRunner(
        _policy(topology_residual_adapter=True),
        CudaGraphInferenceConfig(enabled=True),
    )

    with pytest.raises(
        ValueError,
        match=rf"{key}.*namespace_width={width}",
    ):
        runner.forward_legal_np(entity, legal_ids, context)


def test_configuration_rejects_unsafe_or_ambiguous_buckets():
    with pytest.raises(ValueError, match="strictly increasing"):
        CudaGraphInferenceConfig(batch_buckets=(8, 8, 16))
    with pytest.raises(ValueError, match="positive"):
        CudaGraphInferenceConfig(batch_buckets=(0, 8))
    with pytest.raises(TypeError, match="not bool"):
        CudaGraphInferenceConfig(event_token_limit=False)
