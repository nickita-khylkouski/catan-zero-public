#!/usr/bin/env python3
"""Measure steady-state evaluator packing at 8/16/24 processes per GPU.

This is deliberately a leaf-evaluation microbenchmark, not a strength panel:
every packing point evaluates the exact same deterministic Rust game states
with the exact same checkpoint.  Worker construction and CUDA warm-up finish
before the timed barrier, so the reported rate isolates the process/GPU
packing choice instead of checkpoint I/O.  Results include an output digest;
the run fails if changing worker count changes any prior/value result.

Run this only on an otherwise-idle GPU.  Example::

    python tools/bench_eval_worker_packing.py \
      --checkpoint /abs/a1.pt --gpu 0 --workers 8,16,24 \
      --total-evals 480 --no-cpu-affinity --out packing.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import socket
import stat
import statistics
import subprocess
import sys
import time
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
for _path in (_SRC_ROOT, _REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


CPUSET_RE = re.compile(r"^[0-9,-]+$")
SCHEMA = "eval-worker-packing-benchmark-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _parse_workers(value: str) -> tuple[int, ...]:
    try:
        workers = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("workers must be comma-separated integers") from error
    if not workers or any(item <= 0 for item in workers) or len(set(workers)) != len(workers):
        raise argparse.ArgumentTypeError("workers must be unique positive integers")
    return workers


def _parse_cpuset(value: str) -> set[int]:
    if not CPUSET_RE.fullmatch(value):
        raise ValueError(f"invalid CPU set {value!r}")
    cpus: set[int] = set()
    for item in value.split(","):
        bounds = item.split("-", 1)
        lo = int(bounds[0])
        hi = int(bounds[-1])
        if lo < 0 or hi < lo:
            raise ValueError(f"invalid CPU range {item!r}")
        cpus.update(range(lo, hi + 1))
    if not cpus:
        raise ValueError("CPU set is empty")
    return cpus


def _gpu_cpuset(gpu: int) -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    label = f"GPU{gpu}"
    for line in result.stdout.splitlines():
        fields = line.split()
        if fields and fields[0] == label and len(fields) >= 4:
            # nvidia-smi's final three fields are CPU affinity, NUMA affinity,
            # and GPU NUMA ID.  This matches the production fleet launcher.
            candidate = fields[-3]
            return candidate if CPUSET_RE.fullmatch(candidate) else None
    return None


def _split_indices(total: int, workers: int) -> list[list[int]]:
    if total < workers:
        raise ValueError("total_evals must be >= every worker count")
    shards = [[] for _ in range(workers)]
    for index in range(total):
        shards[index % workers].append(index)
    return shards


def _result_digest(rows: Sequence[tuple[int, float, tuple[tuple[int, float], ...]]]) -> str:
    canonical = [
        [index, format(value, ".9g"), [[action, format(prior, ".9g")] for action, prior in priors]]
        for index, value, priors in sorted(rows)
    ]
    payload = json.dumps(canonical, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _worker(
    *,
    worker_index: int,
    indices: list[int],
    checkpoint: str,
    base_seed: int,
    public_observation: bool,
    rust_featurize: bool,
    barrier: Any,
    output: Any,
) -> None:
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        from catan_zero.rl.gumbel_self_play import COLORS
        from catan_zero.search.neural_rust_mcts import (
            BatchedEntityGraphRustEvaluator,
            EntityGraphRustEvaluatorConfig,
        )
        from catan_zero.search.rust_mcts import _require_rust_module

        evaluator = BatchedEntityGraphRustEvaluator.from_checkpoint(
            checkpoint,
            device="cuda",
            config=EntityGraphRustEvaluatorConfig(
                public_observation=public_observation,
                rust_featurize=rust_featurize,
                cache_size=0,
            ),
        )
        rust = _require_rust_module()
        states = []
        for index in indices:
            game = rust.Game.simple(list(COLORS), seed=base_seed + index)
            for _ in range(index % 7):
                if game.winning_color() is not None:
                    break
                game.play_tick()
            legal = tuple(int(action) for action in game.playable_action_indices(list(COLORS), None))
            states.append((index, game, legal, str(game.current_color())))
        # Initialize CUDA, topology, model buffers, and the async evaluator
        # thread before the timing barrier.
        warm = states[0]
        evaluator.evaluate(warm[1], warm[2], root_color=warm[3], colors=COLORS)
        barrier.wait()
        started = time.perf_counter()
        rows = []
        for index, game, legal, root_color in states:
            priors, value = evaluator.evaluate(
                game, legal, root_color=root_color, colors=COLORS
            )
            rows.append(
                (
                    index,
                    float(value),
                    tuple(sorted((int(action), float(prior)) for action, prior in priors.items())),
                )
            )
        elapsed = time.perf_counter() - started
        evaluator.close()
        output.put({"worker": worker_index, "elapsed_sec": elapsed, "rows": rows})
    except BaseException as error:  # noqa: BLE001 - child must report failures to parent.
        output.put({"worker": worker_index, "error": repr(error)})
        try:
            barrier.abort()
        except BaseException:  # noqa: BLE001
            pass


def _run_point(args: argparse.Namespace, workers: int) -> dict[str, Any]:
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(workers + 1)
    output = context.Queue()
    shards = _split_indices(int(args.total_evals), workers)
    processes = [
        context.Process(
            target=_worker,
            kwargs={
                "worker_index": index,
                "indices": shard,
                "checkpoint": str(args.checkpoint),
                "base_seed": int(args.base_seed),
                "public_observation": bool(args.public_observation),
                "rust_featurize": bool(args.rust_featurize),
                "barrier": barrier,
                "output": output,
            },
        )
        for index, shard in enumerate(shards)
    ]
    startup = time.perf_counter()
    for process in processes:
        process.start()
    barrier.wait(timeout=float(args.startup_timeout))
    startup_sec = time.perf_counter() - startup
    started = time.perf_counter()
    rows = [output.get(timeout=float(args.point_timeout)) for _ in processes]
    wall_sec = time.perf_counter() - started
    for process in processes:
        process.join(timeout=10.0)
        if process.is_alive():
            process.terminate()
            process.join()
    errors = [row for row in rows if "error" in row]
    if errors or any(process.exitcode != 0 for process in processes):
        raise RuntimeError(f"packing point workers={workers} failed: {errors}")
    results = [result for row in rows for result in row["rows"]]
    if len(results) != int(args.total_evals):
        raise RuntimeError(f"packing point workers={workers} lost evaluator rows")
    worker_elapsed = [float(row["elapsed_sec"]) for row in rows]
    return {
        "workers": workers,
        "startup_sec": startup_sec,
        "steady_wall_sec": wall_sec,
        "evaluations": len(results),
        "evaluations_per_sec": len(results) / wall_sec,
        "worker_elapsed_p50_sec": statistics.median(worker_elapsed),
        "worker_elapsed_max_sec": max(worker_elapsed),
        "result_digest": _result_digest(results),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--workers", type=_parse_workers, default=(8, 16, 24))
    parser.add_argument("--total-evals", type=int, default=480)
    parser.add_argument("--base-seed", type=int, default=6_199_900_000)
    parser.add_argument(
        "--cpu-affinity",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "bind the benchmark to nvidia-smi's GPU-local CPU set; default off "
            "because the B200 short-horizon control measured a regression"
        ),
    )
    parser.add_argument("--public-observation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rust-featurize", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--mps-pipe-directory",
        type=Path,
        default=Path("/tmp/mps_pipe_host"),
        help="active systemd-managed MPS pipe directory (required)",
    )
    parser.add_argument(
        "--mps-log-directory", type=Path, default=Path("/tmp/mps_log_host")
    )
    parser.add_argument("--startup-timeout", type=float, default=300.0)
    parser.add_argument("--point-timeout", type=float, default=300.0)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    if args.gpu < 0 or args.total_evals < max(args.workers):
        raise SystemExit("gpu must be non-negative and total-evals >= max(workers)")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.mps_pipe_directory = args.mps_pipe_directory.expanduser().resolve(strict=True)
    args.mps_log_directory = args.mps_log_directory.expanduser().resolve(strict=True)
    control = args.mps_pipe_directory / "control"
    if not control.exists() or not stat.S_ISSOCK(control.stat().st_mode):
        raise SystemExit(
            f"no live MPS control socket at {control}; packing results would not "
            "represent the production evaluator topology"
        )
    os.environ["CUDA_MPS_PIPE_DIRECTORY"] = str(args.mps_pipe_directory)
    os.environ["CUDA_MPS_LOG_DIRECTORY"] = str(args.mps_log_directory)
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    cpuset = _gpu_cpuset(args.gpu) if args.cpu_affinity else None
    if cpuset is not None:
        os.sched_setaffinity(0, _parse_cpuset(cpuset))
    points = [_run_point(args, workers) for workers in args.workers]
    digests = {point["result_digest"] for point in points}
    if len(digests) != 1:
        raise RuntimeError(f"worker packing changed evaluator outputs: {sorted(digests)}")
    payload = {
        "schema_version": SCHEMA,
        "host": socket.gethostname(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "gpu": args.gpu,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "mps_pipe_directory": str(args.mps_pipe_directory),
        "mps_log_directory": str(args.mps_log_directory),
        "cpu_affinity_requested": bool(args.cpu_affinity),
        "cpu_affinity": cpuset,
        "total_evals": int(args.total_evals),
        "base_seed": int(args.base_seed),
        "public_observation": bool(args.public_observation),
        "rust_featurize": bool(args.rust_featurize),
        "points": points,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.out)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
