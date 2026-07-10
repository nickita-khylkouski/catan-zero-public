"""Focused regression tests for EvalServer forward-window assembly."""

from __future__ import annotations

import queue
import types

import numpy as np
import pytest

from catan_zero.search.eval_server import (
    RemoteEvalClient,
    _crop_masked_event_tail,
    _crop_payload_event_tails_before_merge,
    _legal_cell_counts,
    _merge_forward_payloads,
    _policy_needs_action_targets,
    _policy_needs_relational_topology,
)


def _payload(*, marker: int, rows: int, legal_width: int) -> dict:
    """Build one request with distinct, exactly representable values."""
    base = marker * 100
    legal_ids = (
        base + np.arange(rows * legal_width, dtype=np.int64)
    ).reshape(rows, legal_width)
    context = (
        base + 10 + np.arange(rows * legal_width * 3, dtype=np.float32)
    ).reshape(rows, legal_width, 3)
    legal_action_tokens = (
        (base + 30 + np.arange(rows * legal_width * 2, dtype=np.float32))
        .astype(np.float16)
        .reshape(rows, legal_width, 2)
    )
    legal_action_target_ids = (
        base + 50 + np.arange(rows * legal_width * 2, dtype=np.int16)
    ).reshape(rows, legal_width, 2)
    legal_action_mask = (
        np.arange(rows * legal_width).reshape(rows, legal_width) % 3 != 1
    )
    global_tokens = (
        (base + 70 + np.arange(rows * 3, dtype=np.float32))
        .astype(np.float16)
        .reshape(rows, 3)
    )
    return {
        "legal_ids": legal_ids,
        "context": context,
        "entity": {
            "legal_action_tokens": legal_action_tokens,
            "legal_action_target_ids": legal_action_target_ids,
            "legal_action_mask": legal_action_mask,
            "global_tokens": global_tokens,
        },
    }


def test_merge_forward_payloads_preserves_rows_dtypes_values_and_padding() -> None:
    payloads = [
        _payload(marker=1, rows=2, legal_width=2),
        _payload(marker=5, rows=1, legal_width=4),
        _payload(marker=9, rows=3, legal_width=1),
    ]

    entity, legal_ids, context, row_counts = _merge_forward_payloads(payloads)

    assert row_counts == [2, 1, 3]
    assert legal_ids.shape == (6, 4)
    assert context.shape == (6, 4, 3)
    assert entity["legal_action_tokens"].shape == (6, 4, 2)
    assert entity["legal_action_target_ids"].shape == (6, 4, 2)
    assert entity["legal_action_mask"].shape == (6, 4)
    assert entity["global_tokens"].shape == (6, 3)

    assert legal_ids.dtype == np.dtype(np.int64)
    assert context.dtype == np.dtype(np.float32)
    for key, value in payloads[0]["entity"].items():
        assert entity[key].dtype == value.dtype, key

    # Fixed-width rows concatenate in request order without padding.
    np.testing.assert_array_equal(
        entity["global_tokens"],
        np.concatenate([payload["entity"]["global_tokens"] for payload in payloads]),
    )

    offset = 0
    for payload in payloads:
        rows, legal_width = payload["legal_ids"].shape
        row_slice = slice(offset, offset + rows)

        # Every real cell keeps its request-local row order and exact value.
        np.testing.assert_array_equal(
            legal_ids[row_slice, :legal_width], payload["legal_ids"]
        )
        np.testing.assert_array_equal(
            context[row_slice, :legal_width], payload["context"]
        )
        for key in (
            "legal_action_tokens",
            "legal_action_target_ids",
            "legal_action_mask",
        ):
            np.testing.assert_array_equal(
                entity[key][row_slice, :legal_width], payload["entity"][key]
            )

        # Tail cells use the forward contract's field-specific fill values.
        if legal_width < legal_ids.shape[1]:
            assert np.all(legal_ids[row_slice, legal_width:] == -1)
            assert np.all(context[row_slice, legal_width:] == 0.0)
            assert np.all(entity["legal_action_tokens"][row_slice, legal_width:] == 0.0)
            assert np.all(
                entity["legal_action_target_ids"][row_slice, legal_width:] == -1
            )
            assert not np.any(entity["legal_action_mask"][row_slice, legal_width:])

        offset += rows

    assert offset == legal_ids.shape[0]


def test_merge_forward_payloads_promotes_mixed_fixed_field_dtypes() -> None:
    first = _payload(marker=0, rows=1, legal_width=1)
    second = _payload(marker=0, rows=1, legal_width=1)
    first["entity"]["global_tokens"] = np.array([[1, 2, 3]], dtype=np.int8)
    second["entity"]["global_tokens"] = np.array([[300, 301, 302]], dtype=np.int16)

    entity, _legal_ids, _context, _row_counts = _merge_forward_payloads([first, second])

    assert entity["global_tokens"].dtype == np.dtype(np.int16)
    np.testing.assert_array_equal(
        entity["global_tokens"], np.array([[1, 2, 3], [300, 301, 302]], dtype=np.int16)
    )


def test_legal_cell_counts_excludes_padding_inside_each_request() -> None:
    first = _payload(marker=0, rows=2, legal_width=4)
    second = _payload(marker=1, rows=1, legal_width=2)
    first["entity"]["legal_action_mask"][:] = np.array(
        [[True, True, False, False], [True, False, False, False]]
    )
    second["entity"]["legal_action_mask"][:] = np.array([[True, True]])

    real_cells, request_cells = _legal_cell_counts([first, second])

    assert real_cells == 5
    assert request_cells == 10


def test_premerge_event_crop_matches_postmerge_crop() -> None:
    payloads = [
        _payload(marker=0, rows=2, legal_width=3),
        _payload(marker=1, rows=1, legal_width=2),
    ]
    for index, payload in enumerate(payloads):
        rows = payload["legal_ids"].shape[0]
        payload["entity"]["event_tokens"] = np.arange(
            rows * 5 * 4, dtype=np.float16
        ).reshape(rows, 5, 4)
        payload["entity"]["event_mask"] = np.zeros((rows, 5), dtype=np.bool_)
        payload["entity"]["event_mask"][:, : index + 1] = True

    post_payloads = [
        {
            **payload,
            "entity": dict(payload["entity"]),
        }
        for payload in payloads
    ]
    post_entity, post_ids, post_context, post_rows = _merge_forward_payloads(
        post_payloads
    )
    post_required = _crop_masked_event_tail(post_entity, 2)

    pre_required = _crop_payload_event_tails_before_merge(payloads, 2)
    pre_entity, pre_ids, pre_context, pre_rows = _merge_forward_payloads(payloads)

    assert pre_required == post_required == 2
    np.testing.assert_array_equal(pre_ids, post_ids)
    np.testing.assert_array_equal(pre_context, post_context)
    assert pre_rows == post_rows
    assert pre_entity.keys() == post_entity.keys()
    for key in pre_entity:
        np.testing.assert_array_equal(pre_entity[key], post_entity[key], err_msg=key)


def test_premerge_event_crop_validates_entire_window_before_mutating() -> None:
    payloads = [
        _payload(marker=0, rows=1, legal_width=1),
        _payload(marker=1, rows=1, legal_width=1),
    ]
    for payload in payloads:
        payload["entity"]["event_tokens"] = np.zeros((1, 4, 2), dtype=np.float16)
        payload["entity"]["event_mask"] = np.zeros((1, 4), dtype=np.bool_)
    payloads[1]["entity"]["event_mask"][0, 3] = True
    original_shapes = [payload["entity"]["event_tokens"].shape for payload in payloads]

    with pytest.raises(ValueError, match="would remove an unmasked event token"):
        _crop_payload_event_tails_before_merge(payloads, 2)

    assert [payload["entity"]["event_tokens"].shape for payload in payloads] == original_shapes


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (None, True),
        (types.SimpleNamespace(action_target_gather=False, edge_policy_head=False), False),
        (types.SimpleNamespace(action_target_gather=True, edge_policy_head=False), True),
        (types.SimpleNamespace(action_target_gather=False, edge_policy_head=True), True),
        (
            types.SimpleNamespace(
                state_trunk="rrt",
                action_target_gather=False,
                edge_policy_head=False,
            ),
            True,
        ),
    ],
)
def test_policy_target_requirement_handshake_is_safe(config, expected: bool) -> None:
    policy = types.SimpleNamespace(config=config) if config is not None else object()
    assert _policy_needs_action_targets(policy) is expected


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (None, True),
        (types.SimpleNamespace(state_trunk="transformer"), False),
        (types.SimpleNamespace(state_trunk="rrt"), True),
        (types.SimpleNamespace(state_trunk="resrgcn"), True),
    ],
)
def test_policy_topology_requirement_handshake_is_safe(config, expected: bool) -> None:
    policy = types.SimpleNamespace(config=config) if config is not None else object()
    assert _policy_needs_relational_topology(policy) is expected


@pytest.mark.parametrize("needs_action_targets", [False, True])
def test_remote_client_transports_target_ids_only_when_policy_needs_them(
    needs_action_targets: bool,
) -> None:
    from catan_zero.search.neural_rust_mcts import EntityGraphRustEvaluatorConfig

    class _RequestQueue:
        def __init__(self) -> None:
            self.items: list[tuple[int, int, dict]] = []

        def put(self, item) -> None:
            self.items.append(item)

    class _ResponseQueue:
        def get(self, *, timeout: float):
            assert timeout > 0.0
            return (
                1,
                {
                    "logits": np.zeros((1, 2), dtype=np.float32),
                    "value": np.zeros((1,), dtype=np.float32),
                },
                None,
            )

    request_queue = _RequestQueue()
    target_ids = np.array([[[3, -1, 7, -1], [-1, 11, -1, -1]]], dtype=np.int16)
    client = RemoteEvalClient(
        request_queue,
        _ResponseQueue(),
        0,
        action_size=332,
        trained_with_masked_hidden_info=False,
        needs_action_targets=needs_action_targets,
        config=EntityGraphRustEvaluatorConfig(),
    )

    client._remote_forward(
        {
            "global_tokens": np.zeros((1, 1, 3), dtype=np.float16),
            "legal_action_target_ids": target_ids,
            "hex_vertex_ids": np.zeros((1, 19, 6), dtype=np.int16),
        },
        np.array([[1, 2]], dtype=np.int64),
        np.zeros((1, 2, 4), dtype=np.float32),
        False,
    )

    payload = request_queue.items[0][2]
    assert ("legal_action_target_ids" in payload["entity"]) is needs_action_targets
    assert "hex_vertex_ids" not in payload["entity"]
    if needs_action_targets:
        np.testing.assert_array_equal(payload["entity"]["legal_action_target_ids"], target_ids)


def test_remote_client_transports_topology_for_relational_trunks() -> None:
    from catan_zero.search.neural_rust_mcts import EntityGraphRustEvaluatorConfig

    class _RequestQueue:
        def __init__(self) -> None:
            self.items = []

        def put(self, item) -> None:
            self.items.append(item)

    class _ResponseQueue:
        def get(self, *, timeout: float):
            return (
                1,
                {
                    "logits": np.zeros((1, 2), dtype=np.float32),
                    "value": np.zeros((1,), dtype=np.float32),
                },
                None,
            )

    request_queue = _RequestQueue()
    topology = {
        "hex_vertex_ids": np.zeros((1, 19, 6), dtype=np.int16),
        "hex_edge_ids": np.zeros((1, 19, 6), dtype=np.int16),
        "edge_vertex_ids": np.zeros((1, 72, 2), dtype=np.int16),
        "event_target_ids": np.zeros((1, 64, 4), dtype=np.int16),
    }
    client = RemoteEvalClient(
        request_queue,
        _ResponseQueue(),
        0,
        action_size=332,
        trained_with_masked_hidden_info=False,
        needs_relational_topology=True,
        config=EntityGraphRustEvaluatorConfig(),
    )
    client._remote_forward(
        {"global_tokens": np.zeros((1, 1, 3), dtype=np.float16), **topology},
        np.array([[1, 2]], dtype=np.int64),
        np.zeros((1, 2, 4), dtype=np.float32),
        False,
    )
    payload = request_queue.items[0][2]["entity"]
    assert set(topology).issubset(payload)


def test_remote_client_event_limit_validates_and_crops_before_queue_put() -> None:
    from catan_zero.search.neural_rust_mcts import EntityGraphRustEvaluatorConfig

    class _RequestQueue:
        def __init__(self) -> None:
            self.items: list[tuple[int, int, dict]] = []

        def put(self, item) -> None:
            self.items.append(item)

    class _ResponseQueue:
        def get(self, *, timeout: float):
            assert timeout > 0.0
            return (
                1,
                {
                    "logits": np.zeros((2, 3), dtype=np.float32),
                    "value": np.zeros((2,), dtype=np.float32),
                },
                None,
            )

    request_queue = _RequestQueue()
    event_tokens = np.zeros((2, 64, 41), dtype=np.float16)
    event_mask = np.zeros((2, 64), dtype=np.bool_)
    entity = {
        "global_tokens": np.zeros((2, 1, 3), dtype=np.float16),
        "event_tokens": event_tokens,
        "event_mask": event_mask,
    }
    client = RemoteEvalClient(
        request_queue,
        _ResponseQueue(),
        0,
        action_size=332,
        trained_with_masked_hidden_info=False,
        event_token_limit=0,
        config=EntityGraphRustEvaluatorConfig(),
    )

    client._remote_forward(
        entity,
        np.ones((2, 3), dtype=np.int64),
        np.zeros((2, 3, 4), dtype=np.float32),
        False,
    )

    payload = request_queue.items[0][2]
    assert payload["entity"]["event_tokens"].shape == (2, 0, 41)
    assert payload["entity"]["event_mask"].shape == (2, 0)
    assert payload["_event_source_active_tokens"] == 0
    assert payload["_event_source_padded_tokens"] == 128
    # The evaluator-owned feature mapping and arrays are not mutated.
    assert entity["event_tokens"] is event_tokens
    assert entity["event_mask"] is event_mask
    assert entity["event_tokens"].shape == (2, 64, 41)


def test_no_fallback_client_latches_first_terminal_transport_failure() -> None:
    from catan_zero.search.neural_rust_mcts import EntityGraphRustEvaluatorConfig

    class _RequestQueue:
        def __init__(self) -> None:
            self.items: list[object] = []

        def put(self, item) -> None:
            self.items.append(item)

    class _UnresponsiveQueue:
        def __init__(self) -> None:
            self.calls = 0

        def get(self, *, timeout: float):
            self.calls += 1
            raise queue.Empty

    request_queue = _RequestQueue()
    response_queue = _UnresponsiveQueue()
    client = RemoteEvalClient(
        request_queue,
        response_queue,
        4,
        action_size=332,
        trained_with_masked_hidden_info=False,
        config=EntityGraphRustEvaluatorConfig(),
        client_timeout_ms=1.0,
    )
    entity = {"global_tokens": np.zeros((1, 1, 3), dtype=np.float16)}
    legal_ids = np.array([[1]], dtype=np.int64)
    context = np.zeros((1, 1, 4), dtype=np.float32)

    with pytest.raises(TimeoutError, match="client permanently failed"):
        client._remote_forward(entity, legal_ids, context, False)
    with pytest.raises(TimeoutError, match="client permanently failed"):
        client._remote_forward(entity, legal_ids, context, False)

    assert len(request_queue.items) == 1
    assert response_queue.calls == 1
