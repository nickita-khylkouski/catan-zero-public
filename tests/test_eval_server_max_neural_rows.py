from __future__ import annotations

from collections.abc import Callable
import queue
from typing import Any

import numpy as np
import pytest

from catan_zero.search import eval_server


class _Ready:
    def set(self) -> None:
        pass


def _payload(
    rows: int,
    width: int,
    *,
    row_base: int,
    return_q: bool = False,
) -> dict[str, Any]:
    row_values = np.arange(row_base, row_base + rows, dtype=np.float32)
    legal_ids = (
        np.arange(rows * width, dtype=np.int64).reshape(rows, width) + row_base * 100
    )
    return {
        "entity": {
            "global_tokens": row_values[:, None, None].astype(np.float16),
            "event_tokens": np.zeros((rows, 0, 1), dtype=np.float16),
            "event_mask": np.zeros((rows, 0), dtype=np.bool_),
            "legal_action_mask": np.ones((rows, width), dtype=np.bool_),
        },
        "legal_ids": legal_ids,
        "context": np.zeros((rows, width, 1), dtype=np.float32),
        "return_q": return_q,
    }


def _run_server(
    monkeypatch: pytest.MonkeyPatch,
    requests: list[tuple[int, int, dict[str, Any]]],
    *,
    max_neural_rows: int | None,
    forward_observer: Callable[[int, bool], None] | None = None,
    fail_forward: int | None = None,
    device: str = "cpu",
) -> tuple[list[queue.Queue[Any]], dict[str, Any]]:
    import torch

    from catan_zero.rl.entity_token_policy import EntityGraphPolicy

    forward_count = 0

    class _Policy:
        action_size = 512
        trained_with_masked_hidden_info = False

        def forward_legal_np(self, entity, legal_ids, _context, *, return_q=False):
            nonlocal forward_count
            forward_count += 1
            rows, _width = legal_ids.shape
            if forward_observer is not None:
                forward_observer(rows, bool(return_q))
            if forward_count == fail_forward:
                raise RuntimeError(f"injected forward failure {forward_count}")
            logits = torch.from_numpy(np.asarray(legal_ids, dtype=np.float32).copy())
            value = torch.from_numpy(
                np.asarray(entity["global_tokens"][:, 0, 0], dtype=np.float32).copy()
            )
            outputs = {
                "logits": logits,
                "value": value,
                "value_uncertainty": value + 0.5,
            }
            if return_q:
                outputs["q_values"] = logits + 1000.0
            return outputs

    monkeypatch.setattr(EntityGraphPolicy, "load", lambda *_a, **_k: _Policy())
    request_queue: queue.Queue[Any] = queue.Queue()
    response_queues = [queue.Queue() for _ in range(len(requests))]
    for request in requests:
        request_queue.put(request)
    request_queue.put(eval_server._STOP)
    handshake: dict[str, Any] = {}
    eval_server._server_main(
        "/unused.pt",
        eval_server.EvalServerConfig(
            max_batch_size=len(requests) + 1,
            max_neural_rows=max_neural_rows,
            event_token_limit=0,
            device=device,
        ),
        request_queue,
        response_queues,
        _Ready(),
        handshake,
        False,
    )
    return response_queues, handshake["stats"]


def test_oversized_243_row_request_is_capped_and_reassembled_once(monkeypatch) -> None:
    seen: list[tuple[int, bool]] = []
    payload = _payload(243, 5, row_base=7, return_q=True)
    responses, stats = _run_server(
        monkeypatch,
        [(0, 81, payload)],
        max_neural_rows=96,
        forward_observer=lambda rows, return_q: seen.append((rows, return_q)),
    )

    req_id, result, error = responses[0].get_nowait()
    assert (req_id, error) == (81, None)
    assert responses[0].empty(), "one logical request must receive exactly one response"
    assert seen == [(96, True), (96, True), (51, True)]
    np.testing.assert_array_equal(result["logits"], payload["legal_ids"])
    np.testing.assert_array_equal(
        result["value"], payload["entity"]["global_tokens"][:, 0, 0]
    )
    np.testing.assert_array_equal(result["q_values"], payload["legal_ids"] + 1000.0)
    np.testing.assert_array_equal(
        result["value_uncertainty"],
        payload["entity"]["global_tokens"][:, 0, 0] + 0.5,
    )
    assert stats["forward_calls"] == 3
    assert stats["max_forward_rows"] == 96
    assert stats["forward_row_histogram"] == {96: 2, 51: 1}
    assert stats["oversized_requests"] == 1
    assert stats["oversized_request_chunks"] == 3
    assert stats["windows"] == 1
    assert stats["requests"] == 1
    assert stats["rows"] == 243


def test_mixed_requests_pack_by_rows_and_preserve_per_request_width(
    monkeypatch,
) -> None:
    seen: list[tuple[int, bool]] = []
    payloads = [
        _payload(40, 2, row_base=10, return_q=False),
        _payload(90, 5, row_base=100, return_q=True),
        _payload(7, 3, row_base=300, return_q=False),
    ]
    responses, stats = _run_server(
        monkeypatch,
        [(index, 90 + index, payload) for index, payload in enumerate(payloads)],
        max_neural_rows=64,
        forward_observer=lambda rows, return_q: seen.append((rows, return_q)),
    )

    assert seen == [(64, True), (64, True), (9, True)]
    for index, (response_queue, payload) in enumerate(zip(responses, payloads)):
        req_id, result, error = response_queue.get_nowait()
        assert (req_id, error) == (90 + index, None)
        assert result["logits"].shape == payload["legal_ids"].shape
        np.testing.assert_array_equal(result["logits"], payload["legal_ids"])
        np.testing.assert_array_equal(
            result["value"], payload["entity"]["global_tokens"][:, 0, 0]
        )
        if payload["return_q"]:
            np.testing.assert_array_equal(
                result["q_values"], payload["legal_ids"] + 1000.0
            )
        else:
            assert "q_values" not in result
    assert stats["forward_calls"] == 3
    assert stats["max_forward_rows"] == 64
    assert stats["oversized_requests"] == 1
    # The 90-row request fills the first group's remaining 24 rows, then uses
    # one full fragment and one tail fragment: these are actual forward chunks.
    assert stats["oversized_request_chunks"] == 3


def test_default_none_keeps_one_uncapped_forward(monkeypatch) -> None:
    seen: list[tuple[int, bool]] = []
    payload = _payload(243, 4, row_base=1)
    responses, stats = _run_server(
        monkeypatch,
        [(0, 101, payload)],
        max_neural_rows=None,
        forward_observer=lambda rows, return_q: seen.append((rows, return_q)),
    )

    _req_id, result, error = responses[0].get_nowait()
    assert error is None
    assert seen == [(243, False)]
    np.testing.assert_array_equal(result["logits"], payload["legal_ids"])
    assert stats["max_neural_rows"] is None
    assert stats["forward_calls"] == 1
    assert stats["max_forward_rows"] == 243
    assert stats["oversized_requests"] == 0
    assert stats["oversized_request_chunks"] == 0
    assert stats["cuda_memory_stats_enabled"] is False
    assert stats["cuda_peak_memory_allocated_bytes"] is None
    assert stats["cuda_peak_memory_reserved_bytes"] is None


def test_chunk_failure_emits_one_error_and_no_partial_success(monkeypatch) -> None:
    payload = _payload(243, 4, row_base=1, return_q=True)
    responses, stats = _run_server(
        monkeypatch,
        [(0, 111, payload)],
        max_neural_rows=96,
        fail_forward=2,
    )

    req_id, result, error = responses[0].get_nowait()
    assert req_id == 111
    assert result is None
    assert "injected forward failure 2" in error
    assert responses[0].empty()
    assert stats["windows"] == 0
    assert stats["forward_calls"] == 1


def test_cuda_peak_memory_stats_are_recorded_for_cuda_device(monkeypatch) -> None:
    import torch

    reset_devices: list[str] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "reset_peak_memory_stats",
        lambda device: reset_devices.append(str(device)),
    )
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: 12345)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _device: 23456)

    _responses, stats = _run_server(
        monkeypatch,
        [(0, 121, _payload(3, 2, row_base=1))],
        max_neural_rows=2,
        device="cuda:3",
    )

    assert reset_devices == ["cuda:3"]
    assert stats["cuda_memory_stats_enabled"] is True
    assert stats["cuda_peak_memory_allocated_bytes"] == 12345
    assert stats["cuda_peak_memory_reserved_bytes"] == 23456


@pytest.mark.parametrize("value", [0, -1])
def test_max_neural_rows_must_be_positive(value: int) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        eval_server.EvalServerConfig(max_neural_rows=value)


def test_max_neural_rows_rejects_bool() -> None:
    with pytest.raises(TypeError, match="not bool"):
        eval_server.EvalServerConfig(max_neural_rows=True)
