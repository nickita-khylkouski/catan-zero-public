"""OPT-1: --eval-cache-size lets self-play disable the per-leaf eval cache.

The eval cache keys every leaf by blake2b(json_snapshot). Self-play states are
unique (Catan transpositions over full state are measure-zero), so the cache
never hits and the key work is pure overhead. This flag threads a cache_size
into every EntityGraphRustEvaluatorConfig; 0 skips the key/store entirely.
Default 100000 preserves prior behavior (verified here so deploying the changed
gen script is a no-op until a launcher opts in with --eval-cache-size 0).
"""
from __future__ import annotations

import sys
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import generate_gumbel_selfplay_data as cli  # type: ignore  # noqa: E402


def _min_argv(extra):
    return ["--out-dir", "/tmp/x", "--games", "1", "--checkpoint", "/ckpt.pt", *extra]


def test_eval_cache_size_default_preserves_prior_behavior():
    args = cli.build_parser().parse_args(_min_argv([]))
    assert args.eval_cache_size == 100_000


def test_eval_cache_size_zero_parses():
    args = cli.build_parser().parse_args(_min_argv(["--eval-cache-size", "0"]))
    assert args.eval_cache_size == 0


def test_cache_size_zero_disables_store_and_key_work():
    """Correctness: cache_size <= 0 makes the evaluator memoize nothing, so its
    output is a pure function of state (identical to the cached path). Assert the
    config-level switch the flag drives (cache_enabled = cache_size > 0)."""
    from catan_zero.search.neural_rust_mcts import EntityGraphRustEvaluatorConfig

    assert int(EntityGraphRustEvaluatorConfig(cache_size=0).cache_size) == 0
    assert int(EntityGraphRustEvaluatorConfig().cache_size) == 100_000
    # The evaluate() paths gate ALL key/hash/store work on `cache_size > 0`
    # (neural_rust_mcts.py: cache_enabled = int(self.config.cache_size) > 0),
    # so 0 cannot change eval outputs -- it only removes the never-hit lookup.


def test_eval_server_worker_threads_zero_cache_size_to_remote_client(monkeypatch):
    """The eval-server path must not silently restore the 100k config default."""
    from catan_zero.search import eval_server

    captured: dict[str, object] = {}

    class _FakeRemoteEvalClient:
        def __init__(self, *args, **kwargs):
            captured["client"] = self
            captured["args"] = args
            captured["kwargs"] = kwargs

    def _fake_run_worker(worker_args, *, champion_evaluator=None):
        captured["worker_args"] = worker_args
        captured["champion_evaluator"] = champion_evaluator
        return {"worker_index": worker_args["worker_index"], "ok": True}

    class _ResultQueue:
        def __init__(self):
            self.items = []

        def put(self, item):
            self.items.append(item)

    monkeypatch.setattr(eval_server, "RemoteEvalClient", _FakeRemoteEvalClient)
    monkeypatch.setattr(cli, "_run_worker", _fake_run_worker)

    worker_args = {
        "worker_index": 7,
        "checkpoint": "/checkpoint.pt",
        "device": "cuda",
        "value_scale": 1.0,
        "prior_temperature": 1.0,
        "public_observation": True,
        "rust_featurize": True,
        "eval_cache_size": 0,
    }
    result_queue = _ResultQueue()

    cli._server_worker_entry(
        worker_args,
        object(),
        object(),
        3,
        332,
        True,
        False,
        20_000.0,
        False,
        result_queue,
    )

    kwargs = captured["kwargs"]
    assert kwargs["config"].cache_size == 0
    assert kwargs["needs_action_targets"] is False
    assert captured["champion_evaluator"] is captured["client"]
    assert result_queue.items == [{"worker_index": 7, "ok": True}]
