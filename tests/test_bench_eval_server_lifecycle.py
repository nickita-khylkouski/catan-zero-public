"""Lifecycle regressions for ``tools/bench_eval_server.py``."""

from __future__ import annotations

import queue
import sys
from pathlib import Path
from typing import Any

import pytest

from catan_zero.search import eval_server
from catan_zero.search import neural_rust_mcts

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import bench_eval_server as benchmark  # type: ignore  # noqa: E402
import bench_leaf_eval_batching as leaf_benchmark  # type: ignore  # noqa: E402


class _WorkerProcess:
    def __init__(self, events: list[str], *, exit_on_join: bool = False) -> None:
        self._events = events
        self._exit_on_join = exit_on_join
        self._alive = False

    def start(self) -> None:
        self._events.append("worker.start")
        self._alive = True

    def is_alive(self) -> bool:
        self._events.append("worker.is_alive")
        return self._alive

    def join(self, *, timeout: float) -> None:
        self._events.append(f"worker.join:{timeout}")
        if self._exit_on_join:
            self._alive = False

    def terminate(self) -> None:
        self._events.append("worker.terminate")
        self._alive = False

    def kill(self) -> None:
        self._events.append("worker.kill")
        self._alive = False


class _Queue:
    def __init__(self, items: list[Any]) -> None:
        self._items = items

    def get(self, *, timeout: float) -> Any:
        del timeout
        if not self._items:
            raise queue.Empty
        return self._items.pop(0)


class _Event:
    def set(self) -> None:
        pass


class _Context:
    def __init__(
        self,
        events: list[str],
        *,
        ready_items: list[Any],
        result_items: list[Any] | None = None,
        exit_on_join: bool = False,
    ) -> None:
        self._events = events
        self._queues = [_Queue(ready_items), _Queue(result_items or [])]
        self._exit_on_join = exit_on_join

    def Queue(self) -> _Queue:
        return self._queues.pop(0)

    def Event(self) -> _Event:
        return _Event()

    def Process(self, **_kwargs: Any) -> _WorkerProcess:
        return _WorkerProcess(self._events, exit_on_join=self._exit_on_join)


class _Server:
    instances: list[_Server] = []

    def __init__(self, *_args: Any, num_clients: int, **_kwargs: Any) -> None:
        self.events: list[str] = []
        self.request_queue = object()
        self.response_queues = [object() for _ in range(num_clients)]
        self.__class__.instances.append(self)

    def start(self) -> None:
        self.events.append("server.start")

    def wait_ready(self, *, timeout: float) -> dict[str, Any]:
        self.events.append(f"server.wait_ready:{timeout}")
        return {
            "action_size": 332,
            "trained_with_masked_hidden_info": False,
            "needs_action_targets": False,
        }

    def stop(self) -> dict[str, Any]:
        self.events.append("server.stop")
        return {"stopped": True}


def _arm_args() -> dict[str, Any]:
    return {
        "workers": 1,
        "checkpoint": "/checkpoint.pt",
        "max_batch_size": 64,
        "max_wait_ms": 0.0,
        "device": "cpu",
        "matmul_precision": "highest",
        "request_collector": False,
        "public_observation": False,
    }


def test_cleanup_workers_joins_graceful_completion_without_signals() -> None:
    events: list[str] = []
    proc = _WorkerProcess(events, exit_on_join=True)
    proc.start()

    benchmark._cleanup_workers([proc], graceful=True)

    assert "worker.terminate" not in events
    assert "worker.kill" not in events
    join_timeouts = [
        float(event.partition(":")[2])
        for event in events
        if event.startswith("worker.join:")
    ]
    assert 0.0 <= join_timeouts[0] <= 30.0
    assert 0.0 <= join_timeouts[-1] <= 5.0


def test_cleanup_workers_escalates_from_terminate_to_kill() -> None:
    class _StubbornProcess(_WorkerProcess):
        def terminate(self) -> None:
            self._events.append("worker.terminate")

    events: list[str] = []
    proc = _StubbornProcess(events)
    proc.start()

    benchmark._cleanup_workers([proc], graceful=False)

    first_join = next(
        index for index, event in enumerate(events) if event.startswith("worker.join:")
    )
    assert events.index("worker.terminate") < first_join < events.index("worker.kill")
    assert events[-1].startswith("worker.join:")


def test_barrier_timeout_aborts_worker_and_stops_server(monkeypatch) -> None:
    events: list[str] = []
    context = _Context(events, ready_items=[])
    ticks = iter((0.0, 301.0))
    _Server.instances.clear()
    monkeypatch.setattr(benchmark.mp, "get_context", lambda _method: context)
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(eval_server, "EvalServer", _Server)

    with pytest.raises(TimeoutError, match="only 0/1 workers reached the barrier"):
        benchmark._run_arm("server", _arm_args())

    first_join = next(
        index for index, event in enumerate(events) if event.startswith("worker.join:")
    )
    assert events.index("worker.terminate") < first_join
    assert _Server.instances[0].events[-1] == "server.stop"


def test_result_timeout_raises_after_aborting_worker_and_stopping_server(
    monkeypatch,
) -> None:
    events: list[str] = []
    context = _Context(events, ready_items=[0])
    _Server.instances.clear()
    monkeypatch.setattr(benchmark.mp, "get_context", lambda _method: context)
    monkeypatch.setattr(eval_server, "EvalServer", _Server)

    with pytest.raises(
        TimeoutError, match="only 0/1 benchmark workers returned results"
    ):
        benchmark._run_arm("server", _arm_args())

    first_join = next(
        index for index, event in enumerate(events) if event.startswith("worker.join:")
    )
    assert events.index("worker.terminate") < first_join
    assert _Server.instances[0].events[-1] == "server.stop"


def test_successful_arm_keeps_report_and_reaps_workers(monkeypatch) -> None:
    events: list[str] = []
    worker_result = {
        "wid": 0,
        "rows": 42,
        "decisions": 7,
        "games_completed": 1,
        "play_sec": 0.5,
    }
    context = _Context(
        events,
        ready_items=[0],
        result_items=[worker_result],
        exit_on_join=True,
    )
    _Server.instances.clear()
    monkeypatch.setattr(benchmark.mp, "get_context", lambda _method: context)
    monkeypatch.setattr(eval_server, "EvalServer", _Server)

    result = benchmark._run_arm("server", _arm_args())

    assert result["total_rows"] == 42
    assert result["total_games"] == 1
    assert result["per_worker"] == [worker_result]
    assert result["errors"] == []
    assert "worker.terminate" not in events
    assert "worker.kill" not in events
    assert _Server.instances[0].events[-1] == "server.stop"


def test_parity_evaluation_exception_stops_server(monkeypatch) -> None:
    import torch

    class _Config:
        def __init__(self, **_kwargs: Any) -> None:
            pass

    class _LocalEvaluator:
        @classmethod
        def from_checkpoint(cls, *_args: Any, **_kwargs: Any) -> _LocalEvaluator:
            return cls()

        def evaluate(
            self, *_args: Any, **_kwargs: Any
        ) -> tuple[dict[int, float], float]:
            return {1: 1.0}, 0.0

    class _FailingClient:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def evaluate(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("parity forward failed")

    _Server.instances.clear()
    monkeypatch.setattr(
        leaf_benchmark,
        "_collect_leaf_states",
        lambda **_kwargs: [(object(), (1,), "RED")],
    )
    monkeypatch.setattr(neural_rust_mcts, "EntityGraphRustEvaluatorConfig", _Config)
    monkeypatch.setattr(neural_rust_mcts, "EntityGraphRustEvaluator", _LocalEvaluator)
    monkeypatch.setattr(eval_server, "EvalServer", _Server)
    monkeypatch.setattr(eval_server, "RemoteEvalClient", _FailingClient)
    monkeypatch.setattr(torch, "set_num_threads", lambda _threads: None)

    args = _arm_args() | {"num_evals": 1, "seed": 7}
    with pytest.raises(RuntimeError, match="parity forward failed"):
        benchmark._parity(args)

    assert _Server.instances[0].events[-1] == "server.stop"
