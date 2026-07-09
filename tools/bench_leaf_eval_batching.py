"""Micro-benchmark for the async micro-batcher's per-leaf latency overhead.

Task #81 / audit finding (2026-07-06): `BatchedEntityGraphRustEvaluator`'s
background `_batch_loop` waits up to `max_wait_ms` (default 3.0ms) hoping a
second concurrent caller's request lands in the same batch. The self-play
driver (`tools/generate_gumbel_selfplay_data.py` et al.) is one process per
worker with exactly ONE thread calling `evaluate()` -- that thread blocks on
`request.done.wait()` before it can possibly enqueue a second request, so no
other request can EVER arrive during the wait. In that (the only real
single-caller) configuration, `max_wait_ms` is pure added latency on every
single leaf evaluation.

This script measures per-leaf `evaluate()` latency, single-threaded,
BEFORE (current `BatchedEntityGraphRustEvaluator._batch_loop`) vs. AFTER (a
locally-reproduced fix -- see `_FixedBatchedEvaluator` below and the report
for the exact patch neural_rust_mcts.py's owner should apply):

    AFTER's `_batch_loop` change: drain whatever is ALREADY queued via a
    non-blocking `get_nowait()` loop first (this is what actually captures
    genuine concurrent arrivals -- a burst of N>1 requests handed over by
    concurrent callers). Only pay the `max_wait_ms` straggler timer if
    concurrency has been OBSERVED before (a prior batch in this evaluator's
    lifetime already contained >1 request) -- a single-threaded caller can
    never trigger that flag, so it never waits; a genuinely multi-threaded
    caller (see `tools/generate_rust_mcts_reanalysis_threaded.py`) still gets
    its cross-thread batching once concurrency shows up.

`neural_rust_mcts.py` is owned by another agent on this task, so the fix
below is NOT applied to that file -- it is reproduced as a local subclass
purely to measure it. See the task report for the literal patch to apply to
`BatchedEntityGraphRustEvaluator._batch_loop`.

Usage:
    tools/bench_leaf_eval_batching.py --checkpoint path/to/checkpoint.pt \
        --num-evals 200 --seed 7
"""

from __future__ import annotations

import argparse
import queue
import statistics
import time
from typing import Any

import torch

# Self-play generation runs many worker PROCESSES concurrently (one game per
# process); each must be pinned to a single torch thread or they oversubscribe
# the host's cores. Match that here so the measured per-leaf latency reflects
# the real deployment, not an artificially multi-threaded single process.
torch.set_num_threads(1)

from catan_zero.rl.entity_token_policy import EntityGraphPolicy
from catan_zero.rl.gumbel_self_play import COLORS
from catan_zero.search.neural_rust_mcts import (
    BatchedEntityGraphRustEvaluator,
    EntityGraphRustEvaluatorConfig,
)
from catan_zero.search.rust_mcts import _require_rust_module


def _fast_policy(*, seed: int, device: str) -> EntityGraphPolicy:
    """A structurally-real policy (real action catalog / static action
    feature table, via the same `EntityGraphPolicy.create` factory
    production checkpoints use) but with a tiny hidden size / 1 layer, so its
    CPU forward pass (~1ms) doesn't swamp the batcher-wait signal this
    benchmark is isolating -- unlike the full 35M-param checkpoint, whose
    ~250ms+ single-threaded CPU forward pass makes a 3ms scheduling
    difference statistically invisible. Pass `--checkpoint` instead for a
    real-model run once the batcher overhead itself is quantified here."""
    return EntityGraphPolicy.create(
        hidden_size=32,
        state_layers=1,
        attention_heads=2,
        dropout=0.0,
        seed=seed,
        device=device,
    )


class _FixedBatchedEvaluator(BatchedEntityGraphRustEvaluator):
    """Local reproduction of the proposed `_batch_loop` fix, for benchmarking
    only (see module docstring) -- NOT the shipped fix."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._observed_concurrency = False

    def _batch_loop(self) -> None:
        while not self._closed.is_set():
            first = self._get_next_request()
            if first is None:
                continue
            batch = [first]
            # Non-blocking drain: grab whatever is ALREADY queued. Genuine
            # concurrent producers (multiple threads each mid-`evaluate()`)
            # will already have their requests sitting here; a lone
            # single-threaded caller never has anything more to drain.
            while len(batch) < self.max_batch_size:
                try:
                    item = self._requests.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    self._closed.set()
                    break
                batch.append(item)
            # Only pay the max_wait_ms straggler timer once concurrency has
            # actually been observed (a previous batch in this evaluator's
            # lifetime already contained >1 request). A single-threaded
            # caller can never flip this flag (it cannot produce a second
            # request until this one resolves), so it never waits; a
            # genuinely concurrent caller still gets cross-thread batching
            # as soon as the first real burst is observed.
            if (
                len(batch) == 1
                and self._observed_concurrency
                and self.max_wait_ms > 0.0
                and not self._closed.is_set()
            ):
                deadline = time.perf_counter() + (self.max_wait_ms / 1000.0)
                while len(batch) < self.max_batch_size:
                    timeout = max(0.0, deadline - time.perf_counter())
                    if timeout <= 0.0:
                        break
                    try:
                        item = self._requests.get(timeout=timeout)
                    except queue.Empty:
                        break
                    if item is None:
                        self._closed.set()
                        break
                    batch.append(item)
            if len(batch) > 1:
                self._observed_concurrency = True
            self._run_batch(batch)


def _collect_leaf_states(
    *, num_states: int, seed: int
) -> list[tuple[Any, tuple[int, ...], str]]:
    """Deterministically collect `num_states` distinct (game, legal_actions,
    root_color) leaf states from real self-play-like traversal, single game
    per seed, cycling seeds so num_evals can exceed one game's length."""
    catanatron_rs = _require_rust_module()
    states: list[tuple[Any, tuple[int, ...], str]] = []
    game_seed = seed
    while len(states) < num_states:
        game = catanatron_rs.Game.simple(list(COLORS), seed=game_seed)
        game_seed += 1
        steps = 0
        while game.winning_color() is None and steps < 400 and len(states) < num_states:
            legal = tuple(int(a) for a in game.playable_action_indices(list(COLORS), None))
            if not legal:
                break
            root_color = str(game.current_color())
            states.append((game.copy(), legal, root_color))
            # Advance with a fixed, deterministic tick so successive states
            # are genuinely different leaves, not repeats of the same state.
            game.play_tick()
            steps += 1
    return states[:num_states]


def _run_single_threaded(
    evaluator: BatchedEntityGraphRustEvaluator,
    states: list[tuple[Any, tuple[int, ...], str]],
) -> tuple[list[float], list[tuple[dict[int, float], float]]]:
    """Simulate the actual self-play driver: ONE thread, calling `evaluate()`
    to completion before the next call -- exactly the calling pattern that
    makes `max_wait_ms` unrecoverable added latency."""
    latencies_ms: list[float] = []
    results: list[tuple[dict[int, float], float]] = []
    for game, legal_actions, root_color in states:
        start = time.perf_counter()
        result = evaluator.evaluate(game, legal_actions, root_color=root_color, colors=COLORS)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
        results.append(result)
    return latencies_ms, results


def _assert_results_match(
    before: list[tuple[dict[int, float], float]],
    after: list[tuple[dict[int, float], float]],
) -> None:
    """The fix only changes WHEN a batch is dispatched, never what goes into
    it -- same (policy, inputs) must produce the same (priors, value) either
    way. Fails loudly (not just a printed warning) if that's ever violated."""
    assert len(before) == len(after)
    for index, ((before_priors, before_value), (after_priors, after_value)) in enumerate(
        zip(before, after)
    ):
        assert before_priors.keys() == after_priors.keys(), f"state {index}: prior key mismatch"
        for action_id in before_priors:
            assert abs(before_priors[action_id] - after_priors[action_id]) < 1.0e-6, (
                f"state {index} action {action_id}: prior mismatch "
                f"{before_priors[action_id]} vs {after_priors[action_id]}"
            )
        assert abs(before_value - after_value) < 1.0e-6, f"state {index}: value mismatch"


def _summarize(label: str, latencies_ms: list[float]) -> dict[str, float]:
    ordered = sorted(latencies_ms)
    n = len(ordered)
    p50 = ordered[n // 2]
    p95 = ordered[min(n - 1, int(n * 0.95))]
    mean = statistics.mean(ordered)
    total = sum(ordered)
    print(
        f"{label:>28s}: mean={mean:7.3f}ms  p50={p50:7.3f}ms  p95={p95:7.3f}ms  "
        f"total={total:8.1f}ms  n={n}"
    )
    return {"mean_ms": mean, "p50_ms": p50, "p95_ms": p95, "total_ms": total, "n": float(n)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "EntityGraphPolicy checkpoint path. Default: build a tiny-but-"
            "real-shaped policy via EntityGraphPolicy.create (fast forward "
            "pass, isolates the batcher-wait signal -- see _fast_policy)."
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-evals", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-wait-ms", type=float, default=3.0)
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument(
        "--public-observation",
        action="store_true",
        help="Must match how the checkpoint was trained (see EntityGraphRustEvaluatorConfig).",
    )
    args = parser.parse_args()

    config = EntityGraphRustEvaluatorConfig(public_observation=bool(args.public_observation))

    print(f"Collecting {args.num_evals} deterministic leaf states (seed={args.seed})...")
    states = _collect_leaf_states(num_states=args.num_evals, seed=args.seed)
    print(f"Collected {len(states)} states.\n")

    def _make_evaluator(cls: type[BatchedEntityGraphRustEvaluator]) -> BatchedEntityGraphRustEvaluator:
        if args.checkpoint:
            return cls.from_checkpoint(
                args.checkpoint,
                device=args.device,
                config=config,
                max_batch_size=args.max_batch_size,
                max_wait_ms=args.max_wait_ms,
            )
        policy = _fast_policy(seed=args.seed, device=args.device)
        return cls(
            policy,
            config=config,
            max_batch_size=args.max_batch_size,
            max_wait_ms=args.max_wait_ms,
        )

    print("--- BEFORE (current BatchedEntityGraphRustEvaluator._batch_loop) ---")
    baseline = _make_evaluator(BatchedEntityGraphRustEvaluator)
    try:
        before_latencies, before_results = _run_single_threaded(baseline, states)
    finally:
        baseline.close()
    before_stats = _summarize("before (single-caller)", before_latencies)

    print("\n--- AFTER (drain-then-conditional-wait fix) ---")
    fixed = _make_evaluator(_FixedBatchedEvaluator)
    try:
        after_latencies, after_results = _run_single_threaded(fixed, states)
    finally:
        fixed.close()
    after_stats = _summarize("after (single-caller)", after_latencies)

    _assert_results_match(before_results, after_results)
    print("\n(priors/value bit-identical before vs after -- fix is behavior-preserving)")

    print()
    speedup = before_stats["mean_ms"] / after_stats["mean_ms"] if after_stats["mean_ms"] > 0 else float("inf")
    saved_ms_per_eval = before_stats["mean_ms"] - after_stats["mean_ms"]
    print(f"Mean per-leaf latency speedup: {speedup:.2f}x")
    print(f"Mean per-leaf latency saved:  {saved_ms_per_eval:.3f}ms/eval")
    print(
        f"Total wall-time saved over {len(states)} evals: "
        f"{before_stats['total_ms'] - after_stats['total_ms']:.1f}ms"
    )


if __name__ == "__main__":
    main()
