#!/usr/bin/env python3
"""CAT-67 eval-server prototype benchmark: cross-game leaf batching vs
per-worker in-process evaluators, on the real Gumbel self-play loop.

Two arms, identical checkpoint / node budget / seeds / game count, only the
evaluator differs:
  --mode local   : each worker builds its OWN EntityGraphRustEvaluator (its own
                   CUDA context) -- reproduces the production N-contexts-on-1-GPU
                   layout that CAT-87 showed is context-thrash-bound.
  --mode server  : one EvalServer process holds the single policy; each worker
                   uses a RemoteEvalClient (featurize+postprocess local, only
                   forward_legal_np centralized) -> one CUDA context, cross-game
                   batched forwards.

Throughput is measured with a START BARRIER: every worker builds its evaluator
and does one warmup eval, then blocks; the parent releases all of them at once
and times the concurrent play phase, so staggered per-worker setup (esp. the
local arm's per-worker model load) is excluded and the GPU contention is real.

  tools/bench_eval_server.py --checkpoint ckpt.pt --device cuda:0 \
      --mode server --workers 8 --games 2 --public-observation \
      --n-full 16 --n-fast 16 --p-full 0.25 --c-visit 50 --c-scale 0.03 \
      --base-seed 9800000001

  tools/bench_eval_server.py --parity --checkpoint ckpt.pt --device cuda:0 \
      --public-observation --num-evals 128
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue as queue_mod
import tempfile
import time
from pathlib import Path
from typing import Any


def _build_configs(a: dict[str, Any], worker_seed: int):
    from catan_zero.rl.gumbel_self_play import COLORS, GumbelSelfPlayConfig
    from catan_zero.search.gumbel_chance_mcts import GumbelChanceMCTSConfig

    config = GumbelSelfPlayConfig(
        colors=COLORS,
        max_decisions=int(a["max_decisions"]),
        temperature_move_fraction=float(a["temperature_move_fraction"]),
        temperature_high=1.0,
        temperature_low=0.0,
        correct_rust_chance_spectra=bool(a["correct_rust_chance_spectra"]),
    )
    search_config = GumbelChanceMCTSConfig(
        colors=COLORS,
        max_depth=int(a["max_depth"]),
        seed=int(worker_seed),
        c_visit=float(a["c_visit"]),
        c_scale=float(a["c_scale"]),
        prior_temperature=1.0,
        n_full=int(a["n_full"]),
        n_fast=int(a["n_fast"]),
        p_full=float(a["p_full"]),
        correct_rust_chance_spectra=bool(a["correct_rust_chance_spectra"]),
        lazy_interior_chance=bool(a["lazy_interior_chance"]),
        root_wave_batching=bool(a["root_wave_batching"]),
    )
    return config, search_config


def _worker(
    mode: str,
    wid: int,
    a: dict[str, Any],
    server_args: tuple | None,
    ready_q: "mp.Queue",
    start_event: "mp.Event",
    result_q: "mp.Queue",
) -> None:
    import torch

    torch.set_num_threads(1)
    from catan_zero.rl.gumbel_self_play import COLORS, run_worker_games
    from catan_zero.search.neural_rust_mcts import (
        EntityGraphRustEvaluator,
        EntityGraphRustEvaluatorConfig,
    )

    eval_config = EntityGraphRustEvaluatorConfig(
        public_observation=bool(a["public_observation"]),
        rust_featurize=True,
    )

    if mode == "server":
        from catan_zero.search.eval_server import RemoteEvalClient

        (
            request_queue,
            response_queue,
            action_size,
            trained_masked,
            entity_feature_adapter,
            needs_targets,
            needs_relational_topology,
            event_token_limit,
        ) = server_args
        evaluator: Any = RemoteEvalClient(
            request_queue,
            response_queue,
            wid,
            action_size=action_size,
            trained_with_masked_hidden_info=trained_masked,
            entity_feature_adapter=entity_feature_adapter,
            public_card_count_features=bool(
                a.get("server_public_card_count_features", False)
            ),
            needs_action_targets=needs_targets,
            needs_relational_topology=bool(needs_relational_topology),
            event_token_limit=event_token_limit,
            config=eval_config,
            client_timeout_ms=float(a["client_timeout_ms"]),
        )
    else:
        evaluator = EntityGraphRustEvaluator.from_checkpoint(
            a["checkpoint"], device=a["device"], config=eval_config
        )

    games = int(a["games"])
    base_seed = int(a["base_seed"])
    game_index_start = wid * games
    worker_seed = base_seed + 100003 * (wid + 1)
    config, search_config = _build_configs(a, worker_seed)

    # Warmup: play ONE decision's worth of eval to warm topology + CUDA kernels,
    # by running a 1-game/short probe is heavy; instead do a single evaluate on a
    # fresh game's root so the barrier-timed phase is steady-state.
    try:
        import catanatron_rs

        g = catanatron_rs.Game.simple(list(COLORS), seed=worker_seed)
        legal = tuple(int(x) for x in g.playable_action_indices(list(COLORS), None))
        if legal:
            evaluator.evaluate(
                g, legal, root_color=str(g.current_color()), colors=COLORS
            )
    except Exception as exc:  # pragma: no cover
        result_q.put({"wid": wid, "error": f"warmup: {exc!r}"})
        ready_q.put(wid)
        start_event.wait()
        return

    ready_q.put(wid)
    start_event.wait()

    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=f"cat67_w{wid}_") as td:
        try:
            summary = run_worker_games(
                out_dir=Path(td),
                games=games,
                game_index_start=game_index_start,
                base_seed=base_seed,
                worker_seed=worker_seed,
                config=config,
                search_config=search_config,
                evaluator=evaluator,
                shard_size=100000,
                fmt="npz",
            )
        except Exception as exc:  # pragma: no cover
            result_q.put({"wid": wid, "error": f"play: {exc!r}"})
            return
    dt = time.perf_counter() - t0
    result_q.put(
        {
            "wid": wid,
            "rows": int(summary.get("rows", 0)),
            "decisions": int(summary.get("decisions_total", 0)),
            "games_completed": int(summary.get("games_completed", 0)),
            "play_sec": dt,
        }
    )


def _cleanup_workers(procs: list[Any], *, graceful: bool) -> None:
    """Reap benchmark workers, escalating when they do not exit promptly."""
    if graceful:
        join_deadline = time.monotonic() + 30.0
        for proc in procs:
            proc.join(timeout=max(0.0, join_deadline - time.monotonic()))

    for proc in procs:
        if proc.is_alive():
            proc.terminate()
    join_deadline = time.monotonic() + 5.0
    for proc in procs:
        proc.join(timeout=max(0.0, join_deadline - time.monotonic()))

    for proc in procs:
        if proc.is_alive() and hasattr(proc, "kill"):
            proc.kill()
    # Always perform a final join, including for processes that exited after
    # terminate(), so no child is left as a zombie on any benchmark path.
    join_deadline = time.monotonic() + 5.0
    for proc in procs:
        proc.join(timeout=max(0.0, join_deadline - time.monotonic()))


def _run_arm(mode: str, a: dict[str, Any]) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    workers = int(a["workers"])
    ready_q = ctx.Queue()
    start_event = ctx.Event()
    result_q = ctx.Queue()

    server = None
    server_args_per_worker: list[tuple | None] = [None] * workers
    server_stats = {}
    try:
        if mode == "server":
            from catan_zero.search.eval_server import EvalServer, EvalServerConfig

            server = EvalServer(
                a["checkpoint"],
                num_clients=workers,
                config=EvalServerConfig(
                    max_batch_size=int(a["max_batch_size"]),
                    max_neural_rows=a.get("max_neural_rows"),
                    max_wait_ms=float(a["max_wait_ms"]),
                    device=a["device"],
                    transport=str(a.get("transport", "mp_queue")),
                    shared_memory_slot_bytes=int(
                        a.get("shared_memory_slot_bytes", 4 * 1024 * 1024)
                    ),
                    event_token_limit=a.get("event_token_limit"),
                    matmul_precision=str(a["matmul_precision"]),
                    request_collector=bool(a["request_collector"]),
                    cuda_graph=bool(a.get("cuda_graph", False)),
                    cuda_graph_batch_buckets=tuple(
                        a.get(
                            "cuda_graph_batch_buckets",
                            (8, 16, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192),
                        )
                    ),
                    cuda_graph_warmup_iterations=int(
                        a.get("cuda_graph_warmup_iterations", 3)
                    ),
                ),
                public_observation=bool(a["public_observation"]),
                mp_context=ctx,
            )
            server.start()
            meta = server.wait_ready(timeout=180.0)
            a["server_public_card_count_features"] = bool(
                meta.get("public_card_count_features", False)
            )
            for wid in range(workers):
                request_queue_for_client = getattr(
                    server, "request_queue_for_client", None
                )
                server_args_per_worker[wid] = (
                    request_queue_for_client(wid)
                    if request_queue_for_client is not None
                    else server.request_queue,
                    server.response_queues[wid],
                    int(meta["action_size"]),
                    bool(meta["trained_with_masked_hidden_info"]),
                    str(meta["entity_feature_adapter"]),
                    bool(meta.get("needs_action_targets", True)),
                    bool(meta.get("needs_relational_topology", False)),
                    meta.get("event_token_limit"),
                )

        procs = []
        workers_completed = False
        try:
            for wid in range(workers):
                proc = ctx.Process(
                    target=_worker,
                    args=(
                        mode,
                        wid,
                        a,
                        server_args_per_worker[wid],
                        ready_q,
                        start_event,
                        result_q,
                    ),
                    daemon=False,
                )
                proc.start()
                procs.append(proc)

            # Barrier: wait for all workers to finish setup + warmup.
            ready = 0
            ready_deadline = time.perf_counter() + 300.0
            while ready < workers and time.perf_counter() < ready_deadline:
                try:
                    ready_q.get(timeout=5.0)
                    ready += 1
                except queue_mod.Empty:
                    pass
            if ready < workers:
                raise TimeoutError(
                    f"only {ready}/{workers} workers reached the barrier"
                )

            wall0 = time.perf_counter()
            start_event.set()

            results = []
            for _ in range(workers):
                try:
                    results.append(result_q.get(timeout=1800.0))
                except queue_mod.Empty:
                    break
            workers_completed = len(results) == workers
        finally:
            _cleanup_workers(procs, graceful=workers_completed)
        if not workers_completed:
            raise TimeoutError(
                f"only {len(results)}/{workers} benchmark workers returned results"
            )
        wall = time.perf_counter() - wall0
    finally:
        if server is not None:
            server_stats = server.stop()

    errors = [r for r in results if "error" in r]
    ok = [r for r in results if "error" not in r]
    total_rows = sum(r["rows"] for r in ok)
    total_games = sum(r["games_completed"] for r in ok)
    return {
        "mode": mode,
        "workers": workers,
        "wall_sec": wall,
        "total_rows": total_rows,
        "total_games": total_games,
        "rows_per_hour": round(total_rows / wall * 3600) if wall > 0 else 0,
        "rows_per_sec": round(total_rows / wall, 3) if wall > 0 else 0,
        "per_worker": sorted(ok, key=lambda r: r["wid"]),
        "errors": errors,
        "server_stats": server_stats,
    }


def _parity_metrics(
    local_res: list[tuple[dict[int, float], float]],
    candidate_res: list[tuple[dict[int, float], float]],
    *,
    candidate_dtype: str,
    matmul_precision: str,
) -> dict[str, Any]:
    """Summarize policy/value drift against strict-FP32 evaluator outputs."""
    if len(local_res) != len(candidate_res):
        raise ValueError("parity result counts differ")

    prior_absdiffs: list[float] = []
    value_absdiffs: list[float] = []
    policy_l1: list[float] = []
    top_action_disagreements = 0
    for (pl, vl), (ps, vs) in zip(local_res, candidate_res):
        if pl.keys() != ps.keys():
            raise ValueError("parity legal-action keys differ")
        state_diffs = [abs(pl[k] - ps[k]) for k in pl]
        prior_absdiffs.extend(state_diffs)
        policy_l1.append(sum(state_diffs))
        value_absdiffs.append(abs(vl - vs))
        if pl and max(pl, key=pl.__getitem__) != max(ps, key=ps.__getitem__):
            top_action_disagreements += 1
    max_prior = max(prior_absdiffs, default=0.0)
    max_value = max(value_absdiffs, default=0.0)
    num_states = len(local_res)
    return {
        "num_evals": num_states,
        "reference_dtype": "fp32",
        "candidate_dtype": candidate_dtype,
        "matmul_precision": matmul_precision,
        "max_prior_absdiff": max_prior,
        "mean_prior_absdiff": (
            sum(prior_absdiffs) / len(prior_absdiffs) if prior_absdiffs else 0.0
        ),
        "max_policy_l1": max(policy_l1, default=0.0),
        "mean_policy_l1": sum(policy_l1) / len(policy_l1) if policy_l1 else 0.0,
        "max_value_absdiff": max_value,
        "mean_value_absdiff": (
            sum(value_absdiffs) / len(value_absdiffs) if value_absdiffs else 0.0
        ),
        "top_action_disagreements": top_action_disagreements,
        "top_action_disagreement_rate": (
            top_action_disagreements / num_states if num_states else 0.0
        ),
        "within_1e-5": bool(max_prior < 1e-5 and max_value < 1e-5),
    }


def _parity(a: dict[str, Any]) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)
    from bench_leaf_eval_batching import _collect_leaf_states
    from catan_zero.rl.gumbel_self_play import COLORS
    from catan_zero.search.eval_server import (
        EvalServer,
        EvalServerConfig,
        RemoteEvalClient,
    )
    from catan_zero.search.neural_rust_mcts import (
        EntityGraphRustEvaluator,
        EntityGraphRustEvaluatorConfig,
    )

    cfg = EntityGraphRustEvaluatorConfig(
        public_observation=bool(a["public_observation"]), rust_featurize=True
    )
    states = _collect_leaf_states(num_states=int(a["num_evals"]), seed=int(a["seed"]))

    local = EntityGraphRustEvaluator.from_checkpoint(
        a["checkpoint"], device=a["device"], config=cfg
    )
    local_res = [
        local.evaluate(game, legal, root_color=root, colors=COLORS)
        for (game, legal, root) in states
    ]

    ctx = mp.get_context("spawn")
    server = EvalServer(
        a["checkpoint"],
        num_clients=1,
        config=EvalServerConfig(
            max_batch_size=int(a["max_batch_size"]),
            max_neural_rows=a.get("max_neural_rows"),
            max_wait_ms=0.0,
            device=a["device"],
            transport=str(a.get("transport", "mp_queue")),
            shared_memory_slot_bytes=int(
                a.get("shared_memory_slot_bytes", 4 * 1024 * 1024)
            ),
            event_token_limit=a.get("event_token_limit"),
            matmul_precision=str(a["matmul_precision"]),
            request_collector=bool(a["request_collector"]),
            cuda_graph=bool(a.get("cuda_graph", False)),
            cuda_graph_batch_buckets=tuple(
                a.get(
                    "cuda_graph_batch_buckets",
                    (8, 16, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192),
                )
            ),
            cuda_graph_warmup_iterations=int(a.get("cuda_graph_warmup_iterations", 3)),
        ),
        public_observation=bool(a["public_observation"]),
        mp_context=ctx,
    )
    try:
        server.start()
        meta = server.wait_ready(timeout=180.0)
        request_queue_for_client = getattr(server, "request_queue_for_client", None)
        client = RemoteEvalClient(
            request_queue_for_client(0)
            if request_queue_for_client is not None
            else server.request_queue,
            server.response_queues[0],
            0,
            action_size=int(meta["action_size"]),
            trained_with_masked_hidden_info=bool(
                meta["trained_with_masked_hidden_info"]
            ),
            entity_feature_adapter=str(meta["entity_feature_adapter"]),
            public_card_count_features=bool(
                meta.get("public_card_count_features", False)
            ),
            needs_action_targets=bool(meta.get("needs_action_targets", True)),
            needs_relational_topology=bool(
                meta.get("needs_relational_topology", False)
            ),
            event_token_limit=meta.get("event_token_limit"),
            config=cfg,
        )
        srv_res = [
            client.evaluate(game, legal, root_color=root, colors=COLORS)
            for (game, legal, root) in states
        ]
    finally:
        server.stop()

    return _parity_metrics(
        local_res,
        srv_res,
        candidate_dtype="fp32",
        matmul_precision=str(a["matmul_precision"]),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mode", choices=["local", "server"], default="local")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--base-seed", type=int, default=9800000001)
    ap.add_argument("--public-observation", action="store_true")
    ap.add_argument("--n-full", type=int, default=16)
    ap.add_argument("--n-fast", type=int, default=16)
    ap.add_argument("--p-full", type=float, default=0.25)
    ap.add_argument("--c-visit", type=float, default=50.0)
    ap.add_argument("--c-scale", type=float, default=0.03)
    ap.add_argument("--max-decisions", type=int, default=600)
    ap.add_argument("--max-depth", type=int, default=80)
    ap.add_argument("--temperature-move-fraction", type=float, default=0.15)
    ap.add_argument("--lazy-interior-chance", action="store_true")
    ap.add_argument("--root-wave-batching", action="store_true")
    ap.add_argument("--correct-rust-chance-spectra", action="store_true")
    ap.add_argument("--max-batch-size", type=int, default=64)
    ap.add_argument(
        "--max-neural-rows",
        type=int,
        default=None,
        help="Optional hard cap on actual rows in one policy forward.",
    )
    ap.add_argument("--max-wait-ms", type=float, default=0.0)
    ap.add_argument(
        "--matmul-precision", choices=("highest", "high", "medium"), default="highest"
    )
    ap.add_argument(
        "--request-collector", action=argparse.BooleanOptionalAction, default=False
    )
    ap.add_argument(
        "--transport", choices=("mp_queue", "shared_memory"), default="mp_queue"
    )
    ap.add_argument("--shared-memory-slot-bytes", type=int, default=4 * 1024 * 1024)
    ap.add_argument("--event-token-limit", type=int, default=None)
    ap.add_argument(
        "--cuda-graph", action=argparse.BooleanOptionalAction, default=False
    )
    ap.add_argument(
        "--cuda-graph-batch-buckets",
        type=int,
        nargs="+",
        default=(8, 16, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192),
    )
    ap.add_argument("--cuda-graph-warmup-iterations", type=int, default=3)
    ap.add_argument("--client-timeout-ms", type=float, default=20000.0)
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--num-evals", type=int, default=128)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--summary-only",
        action="store_true",
        help="Omit the per-worker timing list from the final benchmark JSON.",
    )
    args = ap.parse_args()
    a = vars(args)

    if args.parity:
        print(json.dumps(_parity(a), indent=2))
        return
    result = _run_arm(args.mode, a)
    if args.summary_only:
        result.pop("per_worker", None)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
