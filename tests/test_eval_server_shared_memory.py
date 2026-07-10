"""Cross-process coverage for EvalServer's shared request-slot transport."""

from __future__ import annotations

import multiprocessing as mp
import queue
from typing import Any

import numpy as np
import pytest

from catan_zero.search import eval_server


def _payload(marker: int = 0) -> dict[str, Any]:
    return {
        "entity": {
            "global_tokens": np.arange(12, dtype=np.float16).reshape(2, 2, 3) + marker,
            "legal_action_mask": np.array(
                [[True, False, True], [False, True, True]], dtype=np.bool_
            ),
        },
        "legal_ids": np.arange(6, dtype=np.int64).reshape(2, 3) + marker,
        "context": np.arange(24, dtype=np.float32).reshape(2, 3, 4) + marker,
        "return_q": False,
    }


def _spawn_send(endpoint: Any, payload: dict[str, Any]) -> None:
    endpoint.put((0, 17, payload))


def _spawn_receive(receiver: Any, result_queue: Any) -> None:
    client_id, req_id, payload = receiver.get(timeout=5.0)
    result_queue.put(
        {
            "client_id": client_id,
            "req_id": req_id,
            "legal_ids": payload["legal_ids"].copy(),
            "context": payload["context"].copy(),
            "global_tokens": payload["entity"]["global_tokens"].copy(),
            "mask": payload["entity"]["legal_action_mask"].copy(),
        }
    )


def _close_queue(value: Any) -> None:
    value.close()
    value.join_thread()


def test_shared_request_slot_round_trips_between_two_spawned_processes() -> None:
    """Producer and consumer attach through spawn, not inherited fork state."""
    ctx = mp.get_context("spawn")
    receiver, endpoints = eval_server._make_shared_request_transport(
        ctx, num_clients=1, slot_bytes=64 * 1024
    )
    result_queue = ctx.Queue()
    expected = _payload(marker=31)
    consumer = ctx.Process(target=_spawn_receive, args=(receiver, result_queue))
    producer = ctx.Process(target=_spawn_send, args=(endpoints[0], expected))

    try:
        consumer.start()
        producer.start()
        producer.join(timeout=10.0)
        consumer.join(timeout=10.0)

        assert producer.exitcode == 0
        assert consumer.exitcode == 0
        result = result_queue.get(timeout=1.0)
        assert result["client_id"] == 0
        assert result["req_id"] == 17
        np.testing.assert_array_equal(result["legal_ids"], expected["legal_ids"])
        np.testing.assert_array_equal(result["context"], expected["context"])
        np.testing.assert_array_equal(
            result["global_tokens"], expected["entity"]["global_tokens"]
        )
        np.testing.assert_array_equal(
            result["mask"], expected["entity"]["legal_action_mask"]
        )
    finally:
        for process in (producer, consumer):
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
        _close_queue(result_queue)
        _close_queue(receiver._notification_queue)


def test_shared_request_notification_contains_metadata_not_ndarrays() -> None:
    """The Queue message is small metadata; tensor bytes stay in RawArray."""

    class _RecordingQueue:
        def __init__(self) -> None:
            self.items: list[Any] = []

        def put(self, item: Any) -> None:
            self.items.append(item)

    ctx = mp.get_context("spawn")
    slot = ctx.RawArray("B", 64 * 1024)
    in_flight = ctx.Value("q", -1, lock=True)
    notifications = _RecordingQueue()
    endpoint = eval_server._SharedMemoryRequestEndpoint(
        notifications, slot, in_flight, 64 * 1024, client_id=0
    )
    expected = _payload(marker=7)

    endpoint.put((0, 19, expected))

    assert len(notifications.items) == 1
    marker, client_id, req_id, metadata = notifications.items[0]
    assert marker == eval_server._SHARED_REQUEST
    assert client_id == 0
    assert req_id == 19
    assert metadata["used_bytes"] < 64 * 1024

    def _contains_array(value: Any) -> bool:
        if isinstance(value, np.ndarray):
            return True
        if isinstance(value, dict):
            return any(_contains_array(v) for v in value.values())
        if isinstance(value, (tuple, list)):
            return any(_contains_array(v) for v in value)
        return False

    assert not _contains_array(notifications.items[0])
    decoded = eval_server._unpack_shared_request(slot, metadata)
    np.testing.assert_array_equal(decoded["legal_ids"], expected["legal_ids"])
    np.testing.assert_array_equal(decoded["context"], expected["context"])


def test_shared_request_overflow_falls_back_to_queue_payload() -> None:
    class _RecordingQueue:
        def __init__(self) -> None:
            self.items: list[Any] = []

        def put(self, item: Any) -> None:
            self.items.append(item)

    ctx = mp.get_context("spawn")
    slot = ctx.RawArray("B", 64)
    in_flight = ctx.Value("q", -1, lock=True)
    notifications = _RecordingQueue()
    endpoint = eval_server._SharedMemoryRequestEndpoint(
        notifications, slot, in_flight, 64, client_id=0
    )
    expected = _payload()

    endpoint.put((0, 23, expected))

    assert len(notifications.items) == 1
    client_id, req_id, payload = notifications.items[0]
    assert client_id == 0
    assert req_id == 23
    assert payload is expected


def test_event_tail_crop_is_fail_closed_and_supports_zero_width() -> None:
    entity = {
        "event_tokens": np.ones((2, 4, 3), dtype=np.float16),
        "event_mask": np.asarray(
            [[True, True, False, False], [True, False, False, False]],
            dtype=np.bool_,
        ),
    }

    assert eval_server._crop_masked_event_tail(entity, None) == 2
    assert entity["event_tokens"].shape == (2, 4, 3)
    assert eval_server._crop_masked_event_tail(entity, 2) == 2
    assert entity["event_tokens"].shape == (2, 2, 3)
    assert entity["event_mask"].shape == (2, 2)

    with pytest.raises(ValueError, match="remove an unmasked"):
        eval_server._crop_masked_event_tail(entity, 1)
    with pytest.raises(TypeError, match="must be an integer"):
        eval_server._crop_masked_event_tail(entity, 1.5)

    empty = {
        "event_tokens": np.ones((3, 64, 3), dtype=np.float16),
        "event_mask": np.zeros((3, 64), dtype=np.bool_),
    }
    assert eval_server._crop_masked_event_tail(empty, 0) == 0
    assert empty["event_tokens"].shape == (3, 0, 3)
    assert empty["event_mask"].shape == (3, 0)

    malformed = {
        "event_tokens": np.ones((1, 4, 3), dtype=np.float16),
        "event_mask": np.zeros((1, 2), dtype=np.bool_),
    }
    with pytest.raises(ValueError, match="shape mismatch"):
        eval_server._crop_masked_event_tail(malformed, 2)
    assert malformed["event_tokens"].shape == (1, 4, 3)
    assert malformed["event_mask"].shape == (1, 2)


def test_shared_slot_cannot_overwrite_an_in_flight_descriptor() -> None:
    """A second put must fall back until the matching response releases slot."""

    class _RecordingQueue:
        def __init__(self) -> None:
            self.items: list[Any] = []

        def put(self, item: Any) -> None:
            self.items.append(item)

    ctx = mp.get_context("spawn")
    slot = ctx.RawArray("B", 64 * 1024)
    in_flight = ctx.Value("q", -1, lock=True)
    notifications = _RecordingQueue()
    endpoint = eval_server._SharedMemoryRequestEndpoint(
        notifications, slot, in_flight, 64 * 1024, client_id=0
    )
    first = _payload(marker=1)
    second = _payload(marker=99)

    endpoint.put((0, 41, first))
    endpoint.put((0, 42, second))

    marker, _client_id, _req_id, metadata = notifications.items[0]
    assert marker == eval_server._SHARED_REQUEST
    # The queued descriptor still reads the first request; the second request
    # traveled as a complete Queue payload instead of overwriting the slot.
    decoded = eval_server._unpack_shared_request(slot, metadata)
    np.testing.assert_array_equal(decoded["legal_ids"], first["legal_ids"])
    assert notifications.items[1][:2] == (0, 42)
    assert notifications.items[1][2] is second

    endpoint.request_complete(0, 41)
    endpoint.put((0, 43, second))
    assert notifications.items[2][0] == eval_server._SHARED_REQUEST


def test_eval_server_validates_transport_and_client_endpoint() -> None:
    with pytest.raises(ValueError, match="unsupported EvalServer transport"):
        eval_server.EvalServer(
            "/unused.pt",
            num_clients=1,
            config=eval_server.EvalServerConfig(transport="invalid"),
        )

    # Endpoint range validation happens before any process is started.
    server = eval_server.EvalServer(
        "/unused.pt",
        num_clients=1,
        config=eval_server.EvalServerConfig(
            transport="shared_memory", shared_memory_slot_bytes=1024
        ),
    )
    assert server.request_queue_for_client(0) is server.request_queues[0]
    with pytest.raises(IndexError):
        server.request_queue_for_client(1)
    with pytest.raises(queue.Empty):
        server.request_queue.get_nowait()
    notification_queue = server.request_queue._notification_queue
    response_queues = list(server.response_queues)

    server.stop()

    assert notification_queue._closed
    assert all(response_queue._closed for response_queue in response_queues)


def test_server_honors_return_q_and_preserves_q_values(monkeypatch) -> None:
    import torch

    from catan_zero.rl.entity_token_policy import EntityGraphPolicy

    seen_return_q: list[bool] = []

    class _Policy:
        action_size = 8
        trained_with_masked_hidden_info = False

        def forward_legal_np(self, _entity, legal_ids, _context, *, return_q=False):
            seen_return_q.append(bool(return_q))
            rows, width = legal_ids.shape
            outputs = {
                "logits": torch.zeros((rows, width), dtype=torch.float32),
                "value": torch.zeros((rows,), dtype=torch.float32),
            }
            if return_q:
                outputs["q_values"] = torch.full(
                    (rows, width), 0.25, dtype=torch.float32
                )
            return outputs

    monkeypatch.setattr(EntityGraphPolicy, "load", lambda *_a, **_k: _Policy())

    request_queue: queue.Queue[Any] = queue.Queue()
    response_queue: queue.Queue[Any] = queue.Queue()
    request_queue.put(
        (
            0,
            51,
            {
                "entity": {
                    "global_tokens": np.zeros((1, 1, 1), dtype=np.float16),
                    "event_tokens": np.zeros((1, 4, 1), dtype=np.float16),
                    "event_mask": np.zeros((1, 4), dtype=np.bool_),
                },
                "legal_ids": np.asarray([[2, 3]], dtype=np.int64),
                "context": np.zeros((1, 2, 1), dtype=np.float32),
                "return_q": True,
            },
        )
    )
    request_queue.put(eval_server._STOP)

    class _Ready:
        def set(self) -> None:
            pass

    handshake: dict[str, Any] = {}
    eval_server._server_main(
        "/unused.pt",
        eval_server.EvalServerConfig(max_batch_size=2),
        request_queue,
        [response_queue],
        _Ready(),
        handshake,
        False,
    )

    req_id, result, error = response_queue.get_nowait()
    assert req_id == 51
    assert error is None
    assert seen_return_q == [True]
    np.testing.assert_array_equal(result["q_values"], np.full((1, 2), 0.25))


def test_server_trims_outputs_to_each_requests_legal_width(monkeypatch) -> None:
    import torch

    from catan_zero.rl.entity_token_policy import EntityGraphPolicy

    class _Policy:
        action_size = 8
        trained_with_masked_hidden_info = False

        def forward_legal_np(self, _entity, legal_ids, _context, *, return_q=False):
            rows, width = legal_ids.shape
            assert return_q is True
            values = torch.arange(rows * width, dtype=torch.float32).reshape(
                rows, width
            )
            return {
                "logits": values,
                "value": torch.zeros((rows,), dtype=torch.float32),
                "q_values": values + 100.0,
            }

    monkeypatch.setattr(EntityGraphPolicy, "load", lambda *_a, **_k: _Policy())

    request_queue: queue.Queue[Any] = queue.Queue()
    responses = [queue.Queue(), queue.Queue()]
    for client_id, width, wants_q in ((0, 2, True), (1, 4, False)):
        request_queue.put(
            (
                client_id,
                60 + client_id,
                {
                    "entity": {
                        "global_tokens": np.zeros((1, 1, 1), dtype=np.float16),
                        "event_tokens": np.zeros((1, 4, 1), dtype=np.float16),
                        "event_mask": np.zeros((1, 4), dtype=np.bool_),
                    },
                    "legal_ids": np.arange(width, dtype=np.int64)[None, :],
                    "context": np.zeros((1, width, 1), dtype=np.float32),
                    "return_q": wants_q,
                },
            )
        )
    request_queue.put(eval_server._STOP)

    class _Ready:
        def set(self) -> None:
            pass

    eval_server._server_main(
        "/unused.pt",
        eval_server.EvalServerConfig(max_batch_size=3),
        request_queue,
        responses,
        _Ready(),
        {},
        False,
    )

    _req0, result0, error0 = responses[0].get_nowait()
    _req1, result1, error1 = responses[1].get_nowait()
    assert error0 is error1 is None
    assert result0["logits"].shape == (1, 2)
    assert result0["q_values"].shape == (1, 2)
    assert result1["logits"].shape == (1, 4)
    assert "q_values" not in result1
