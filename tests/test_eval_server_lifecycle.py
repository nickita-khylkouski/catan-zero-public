"""Deterministic lifecycle regressions for the cross-process EvalServer."""

from __future__ import annotations

import queue
import socket
import sys
import threading
import types
from pathlib import Path

import pytest

from catan_zero.rl.entity_feature_adapter import CURRENT_RUST_ENTITY_ADAPTER_VERSION
from catan_zero.search import eval_server

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import generate_gumbel_selfplay_data as generator  # type: ignore  # noqa: E402


def test_cuda_graph_forward_wrapper_is_opt_in_and_strict_fp32(monkeypatch) -> None:
    raw_policy = object()
    assert (
        eval_server._make_forward_policy(raw_policy, eval_server.EvalServerConfig())
        is raw_policy
    )

    with pytest.raises(ValueError, match="matmul_precision='highest'"):
        eval_server._make_forward_policy(
            raw_policy,
            eval_server.EvalServerConfig(cuda_graph=True, matmul_precision="high"),
        )

    captured: dict[str, object] = {}

    class _Runner:
        def __init__(self, policy, config) -> None:
            captured["policy"] = policy
            captured["config"] = config

    monkeypatch.setattr(
        "catan_zero.search.cuda_graph_inference.CudaGraphInferenceRunner", _Runner
    )
    wrapped = eval_server._make_forward_policy(
        raw_policy,
        eval_server.EvalServerConfig(
            cuda_graph=True,
            cuda_graph_batch_buckets=(4, 12),
            cuda_graph_warmup_iterations=5,
        ),
    )

    assert isinstance(wrapped, _Runner)
    assert captured["policy"] is raw_policy
    runner_config = captured["config"]
    assert runner_config.enabled is True
    assert runner_config.batch_buckets == (4, 12)
    assert runner_config.warmup_iterations == 5
    # EvalServer already cropped and validated events before model inference.
    assert runner_config.event_token_limit is None


def test_neural_row_cap_rejects_cuda_graph_bucket_rounding() -> None:
    with pytest.raises(ValueError, match="incompatible with cuda_graph"):
        eval_server.EvalServerConfig(max_neural_rows=100, cuda_graph=True)


def test_cuda_graph_stats_count_graphs_and_fallback_reasons() -> None:
    stats = {
        "cuda_graph_calls": 0,
        "cuda_graph_fallbacks": 0,
        "cuda_graph_graph_count": 0,
        "cuda_graph_last_fallback_reason": None,
        "cuda_graph_fallback_reason_histogram": {},
    }
    runner = types.SimpleNamespace(
        graph_count=2,
        last_path="cuda_graph",
        last_fallback_reason=None,
    )
    eval_server._record_cuda_graph_call(stats, runner)
    assert stats["cuda_graph_calls"] == 1
    assert stats["cuda_graph_fallbacks"] == 0
    assert stats["cuda_graph_graph_count"] == 2

    runner.last_path = "eager_fallback"
    runner.last_fallback_reason = "batch exceeds largest bucket"
    eval_server._record_cuda_graph_call(stats, runner)
    eval_server._record_cuda_graph_call(stats, runner)
    assert stats["cuda_graph_calls"] == 3
    assert stats["cuda_graph_fallbacks"] == 2
    assert stats["cuda_graph_last_fallback_reason"] == "batch exceeds largest bucket"
    assert stats["cuda_graph_fallback_reason_histogram"] == {
        "batch exceeds largest bucket": 2
    }


def test_wait_ready_fails_fast_when_server_process_has_exited() -> None:
    class _NeverReady:
        def __init__(self) -> None:
            self.wait_calls: list[float] = []

        def wait(self, *, timeout: float) -> bool:
            self.wait_calls.append(timeout)
            return False

    ready = _NeverReady()
    server = object.__new__(eval_server.EvalServer)
    server._ready = ready
    server._proc = types.SimpleNamespace(exitcode=23)

    with pytest.raises(RuntimeError, match=r"exited before ready \(exitcode=23\)"):
        server.wait_ready(timeout=30.0)

    assert len(ready.wait_calls) == 1
    assert 0.0 <= ready.wait_calls[0] <= 0.1


@pytest.mark.parametrize(
    "missing_field", ["meaningful_public_history", "event_history_limit"]
)
def test_wait_ready_requires_history_handshake_fields(missing_field: str) -> None:
    class _Ready:
        def wait(self, *, timeout: float) -> bool:
            assert timeout >= 0.0
            return True

    server = object.__new__(eval_server.EvalServer)
    server._ready = _Ready()
    server._proc = types.SimpleNamespace(exitcode=None)
    server._handshake = {
        "action_size": 332,
        "entity_feature_adapter": CURRENT_RUST_ENTITY_ADAPTER_VERSION,
        "trained_with_masked_hidden_info": False,
        "public_card_count_features": False,
        "meaningful_public_history": True,
        "event_history_limit": 32,
        "needs_action_targets": True,
        "needs_relational_topology": False,
        "matmul_precision": "highest",
        "transport": "mp_queue",
        "max_neural_rows": None,
        "event_token_limit": None,
        "cuda_graph": False,
        "cuda_graph_batch_buckets": (),
        "cuda_graph_warmup_iterations": 0,
        "value_categorical_bins": 0,
        "value_categorical_head_available": False,
    }
    del server._handshake[missing_field]

    with pytest.raises(KeyError, match=missing_field):
        server.wait_ready(timeout=0.1)


def test_history_handshake_mismatch_aborts_before_worker_spawn(monkeypatch) -> None:
    events: list[str] = []

    class _Context:
        def Queue(self):
            raise AssertionError("result queue must not be created on mismatch")

        def Process(self, **_kwargs):
            raise AssertionError("worker must not be spawned on mismatch")

    class _Server:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def start(self) -> None:
            events.append("start")

        def wait_ready(self, *, timeout: float) -> dict[str, object]:
            assert timeout > 0.0
            events.append("ready")
            return {
                "meaningful_public_history": False,
                "event_history_limit": 64,
            }

        def stop(self) -> dict[str, object]:
            events.append("stop")
            return {}

    monkeypatch.setattr(
        generator.multiprocessing, "get_context", lambda _kind: _Context()
    )
    monkeypatch.setattr(eval_server, "EvalServer", _Server)
    args = types.SimpleNamespace(
        checkpoint="/checkpoint.pt",
        eval_server_max_batch=64,
        eval_server_max_wait_ms=0.0,
        device="cuda:0",
        eval_server_matmul_precision="highest",
        eval_server_request_collector=False,
        public_observation=False,
    )

    with pytest.raises(ValueError, match="checkpoint public-history contract"):
        generator._run_eval_server_batch(
            [
                {
                    "worker_index": 3,
                    "meaningful_public_history": True,
                    "event_history_limit": 32,
                }
            ],
            args,
        )

    assert events == ["start", "ready", "stop"]


def test_no_fallback_batch_aborts_workers_when_ready_server_exits(
    monkeypatch,
) -> None:
    """A post-ready server crash must not become one timeout per remaining game."""
    events: list[tuple[str, object]] = []
    process_kwargs: list[dict[str, object]] = []

    class _ResultQueue:
        def get(self, *, timeout: float):
            events.append(("result.get", timeout))
            raise queue.Empty

    class _WorkerProcess:
        def __init__(self, **_kwargs) -> None:
            self.alive = False

        def start(self) -> None:
            self.alive = True
            events.append(("worker.start", None))

        def is_alive(self) -> bool:
            events.append(("worker.is_alive", self.alive))
            return self.alive

        def join(self, *, timeout: float) -> None:
            events.append(("worker.join", timeout))

        def terminate(self) -> None:
            events.append(("worker.terminate", None))
            self.alive = False

        def kill(self) -> None:
            events.append(("worker.kill", None))
            self.alive = False

    class _Context:
        def Queue(self):
            return _ResultQueue()

        def Process(self, **kwargs):
            process_kwargs.append(kwargs)
            return _WorkerProcess(**kwargs)

    class _Server:
        def __init__(self, *_args, num_clients: int, **_kwargs) -> None:
            self.request_queue = object()
            self.response_queues = [object() for _ in range(num_clients)]
            self.exitcode: int | None = None

        def start(self) -> None:
            events.append(("server.start", None))

        def wait_ready(self, *, timeout: float) -> dict[str, object]:
            events.append(("server.wait_ready", timeout))
            self.exitcode = 73
            return {
                "action_size": 332,
                "trained_with_masked_hidden_info": False,
                "entity_feature_adapter": CURRENT_RUST_ENTITY_ADAPTER_VERSION,
                "meaningful_public_history": True,
                "event_history_limit": 32,
                "matmul_precision": "highest",
            }

        def stop(self) -> dict[str, object]:
            events.append(("server.stop", None))
            return {}

    context = _Context()
    monkeypatch.setattr(generator.multiprocessing, "get_context", lambda _kind: context)
    monkeypatch.setattr(eval_server, "EvalServer", _Server)

    args = types.SimpleNamespace(
        checkpoint="/checkpoint.pt",
        eval_server_max_batch=64,
        eval_server_max_wait_ms=0.0,
        device="cuda:0",
        eval_server_matmul_precision="highest",
        eval_server_request_collector=False,
        public_observation=False,
        eval_server_timeout_ms=20_000.0,
        eval_server_local_fallback=False,
        eval_server_batch_timeout_sec=0.01,
    )
    worker_args = [
        {
            "worker_index": 0,
            "games": 1_500,
            "max_decisions": 600,
            "out_dir": "/tmp/worker-0",
            "meaningful_public_history": True,
            "event_history_limit": 32,
        }
    ]

    results, stats = generator._run_eval_server_batch(worker_args, args)

    assert stats == {}
    assert len(results) == 1
    spawned_worker_args = process_kwargs[0]["args"][0]
    assert spawned_worker_args["_eval_server_meaningful_public_history"] is True
    assert spawned_worker_args["_eval_server_event_history_limit"] == 32
    assert "exited after ready (exitcode=73)" in results[0]["errors"][0]["error"]
    result_get = next(value for event, value in events if event == "result.get")
    assert 0.0 < float(result_get) <= 0.25
    # Crash abort skips the normal five-second grace join and terminates first.
    assert events.index(("worker.terminate", None)) < next(
        index for index, event in enumerate(events) if event[0] == "worker.join"
    )
    assert ("worker.kill", None) not in events
    assert events[-1] == ("server.stop", None)


def test_stop_escalates_then_final_joins_and_shuts_down_manager() -> None:
    events: list[tuple[str, object]] = []

    class _Queue:
        def put(self, item) -> None:
            events.append(("queue.put", item))

    class _Process:
        pid = 1234

        def is_alive(self) -> bool:
            events.append(("proc.is_alive", True))
            return True

        def join(self, *, timeout: float) -> None:
            events.append(("proc.join", timeout))

        def terminate(self) -> None:
            events.append(("proc.terminate", None))

        def kill(self) -> None:
            events.append(("proc.kill", None))

    class _Manager:
        def shutdown(self) -> None:
            events.append(("manager.shutdown", None))

    server = object.__new__(eval_server.EvalServer)
    server._proc = _Process()
    server.request_queue = _Queue()
    server._handshake = {"stats": {"windows": 4, "requests": 11}}
    server._manager = _Manager()
    server._stopped = False
    server._last_stats = {}

    stats = server.stop()

    assert stats == {"windows": 4, "requests": 11}
    assert events == [
        ("proc.is_alive", True),
        ("queue.put", eval_server._STOP),
        ("proc.join", 10.0),
        ("proc.is_alive", True),
        ("proc.terminate", None),
        ("proc.join", 5.0),
        ("proc.is_alive", True),
        ("proc.kill", None),
        ("proc.join", 5.0),
        ("manager.shutdown", None),
    ]

    # Idempotence: a second stop returns the saved stats without touching dead
    # process/manager resources again.
    assert server.stop() == stats
    assert events[-1] == ("manager.shutdown", None)
    assert events.count(("manager.shutdown", None)) == 1


def test_collector_pause_waits_out_inflight_get_and_blocks_next_get() -> None:
    """The pause acknowledgement covers Queue.get's deserialize interval."""

    class _ControlledRequestQueue:
        def __init__(self) -> None:
            self._reader, self._writer = socket.socketpair()
            self._items: queue.Queue[object] = queue.Queue()
            self.calls = 0
            self.first_get_started = threading.Event()
            self.release_first_get = threading.Event()
            self.second_get_started = threading.Event()

        def enqueue(self, item: object) -> None:
            self._items.put(item)
            self._writer.send(b"x")

        def get(self) -> object:
            self._reader.recv(1)
            self.calls += 1
            if self.calls == 1:
                self.first_get_started.set()
                if not self.release_first_get.wait(timeout=1.0):
                    raise TimeoutError("test did not release the in-flight get")
            elif self.calls == 2:
                self.second_get_started.set()
            return self._items.get_nowait()

        def close(self) -> None:
            self._reader.close()
            self._writer.close()

    request_queue = _ControlledRequestQueue()
    collector = eval_server._GatedRequestCollector(request_queue)
    collector.start()
    pause_entered = threading.Event()
    release_pause = threading.Event()

    def _hold_pause() -> None:
        with collector.paused():
            pause_entered.set()
            if not release_pause.wait(timeout=1.0):
                raise TimeoutError("test did not release collector pause")

    pause_thread = threading.Thread(target=_hold_pause)
    try:
        request_queue.enqueue("request-1")
        assert request_queue.first_get_started.wait(timeout=1.0)
        pause_thread.start()

        # paused() cannot acknowledge until the already-started deserialize
        # finishes and releases the activity lock.
        assert not pause_entered.wait(timeout=0.05)
        request_queue.release_first_get.set()
        assert pause_entered.wait(timeout=1.0)
        assert collector.ready_requests.get(timeout=1.0) == "request-1"

        # A newly-readable queue pipe stays untouched for the whole pause.
        request_queue.enqueue("request-2")
        assert not request_queue.second_get_started.wait(timeout=0.05)
        release_pause.set()
        pause_thread.join(timeout=1.0)
        assert not pause_thread.is_alive()
        assert request_queue.second_get_started.wait(timeout=1.0)
        assert collector.ready_requests.get(timeout=1.0) == "request-2"

        request_queue.enqueue(eval_server._STOP)
        assert collector.ready_requests.get(timeout=1.0) == eval_server._STOP
    finally:
        request_queue.release_first_get.set()
        release_pause.set()
        pause_thread.join(timeout=1.0)
        collector.close(timeout=1.0)
        request_queue.close()


def test_collector_close_wakes_idle_wait_and_releases_resources(monkeypatch) -> None:
    """Requested shutdown wakes an idle collector without reporting failure."""

    class _IdleRequestQueue:
        def __init__(self) -> None:
            self._reader, self._writer = socket.socketpair()

        def get(self) -> object:
            raise AssertionError("an idle queue must not be consumed")

        def close(self) -> None:
            self._reader.close()
            self._writer.close()

    wait_started = threading.Event()
    real_wait = eval_server._wait_for_connections

    def _tracked_wait(readers):
        wait_started.set()
        return real_wait(readers)

    monkeypatch.setattr(eval_server, "_wait_for_connections", _tracked_wait)
    request_queue = _IdleRequestQueue()
    collector = eval_server._GatedRequestCollector(request_queue)
    collector.start()
    try:
        assert wait_started.wait(timeout=1.0)

        collector.close(timeout=1.0)

        assert collector._shutdown.is_set()
        assert collector._gate.is_set()
        assert not collector._thread.is_alive()
        assert collector._wake_reader.fileno() == -1
        assert collector._wake_writer.fileno() == -1
        with pytest.raises(queue.Empty):
            collector.ready_requests.get_nowait()
    finally:
        collector.close(timeout=1.0)
        request_queue.close()


def test_collector_failure_after_request_assembly_replies_with_error(
    monkeypatch,
) -> None:
    """A collected request is failed explicitly, never dropped on pipe failure."""
    from catan_zero.rl.entity_token_policy import EntityGraphPolicy

    class _Policy:
        action_size = 332
        trained_with_masked_hidden_info = False

        def forward_legal_np(self, *args, **kwargs):
            raise AssertionError("collector failure must bypass model forward")

    monkeypatch.setattr(EntityGraphPolicy, "load", lambda *args, **kwargs: _Policy())

    fake_torch = types.ModuleType("torch")
    fake_torch.set_num_threads = lambda _count: None
    fake_torch.set_num_interop_threads = lambda _count: None
    fake_torch.set_float32_matmul_precision = lambda _precision: None
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

    request = (0, 91, {"assembled": True})

    class _FailingRequestQueue:
        def __init__(self) -> None:
            self.calls = 0
            self._reader, self._writer = socket.socketpair()
            self._writer.send(b"xx")

        def get(self):
            self._reader.recv(1)
            self.calls += 1
            if self.calls == 1:
                return request
            raise OSError("request pipe failed")

        def close(self) -> None:
            self._reader.close()
            self._writer.close()

    class _ResponseQueue:
        def __init__(self) -> None:
            self.items: list[tuple[object, object, object]] = []

        def put(self, item) -> None:
            self.items.append(item)

    class _Ready:
        def __init__(self) -> None:
            self.was_set = False

        def set(self) -> None:
            self.was_set = True

    response = _ResponseQueue()
    ready = _Ready()
    handshake: dict[str, object] = {}

    request_queue = _FailingRequestQueue()
    try:
        eval_server._server_main(
            "/checkpoint.pt",
            eval_server.EvalServerConfig(
                request_collector=True,
                max_batch_size=2,
                max_wait_ms=100.0,
            ),
            request_queue,
            [response],
            ready,
            handshake,
            False,
        )
    finally:
        request_queue.close()

    assert ready.was_set
    assert len(response.items) == 1
    req_id, result, error = response.items[0]
    assert req_id == 91
    assert result is None
    assert "eval-server request collector failed" in str(error)
    assert handshake["stats"]["windows"] == 0
