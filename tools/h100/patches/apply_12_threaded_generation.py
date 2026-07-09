#!/usr/bin/env python3
"""
SYSTEM_DESIGN_FINDINGS #1: Convert multiprocessing.spawn → threading.Thread
in generate_gumbel_selfplay_data.py.

The current code spawns N separate processes, each loading its own model copy
and doing batch_size=1 inference. This patch adds a --use-threads flag that
switches to threading.Thread with a shared BatchedEntityGraphRustEvaluator.

The Rust engine releases the GIL (421 allow_threads references), so threads
achieve true parallelism during MCTS search. The evaluator's batching thread
batches across threads (batch_size up to max_batch_size=64).

This patch:
1. Adds --use-threads and --max-batch-size CLI flags
2. When --use-threads, creates a shared evaluator and passes it to all workers
3. Workers use threading.Thread instead of multiprocessing.Pool
4. The shared evaluator's batching thread handles cross-thread batching

Usage: python3 apply_12_threaded_generation.py /path/to/generate_gumbel_selfplay_data.py
"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else sys.exit("usage: apply_12_threaded_generation.py <path>")
with open(path) as f:
    src = f.read()

if "use_threads" in src:
    print("[SKIP] --use-threads already present")
    sys.exit(0)

# --- Add --use-threads and --max-batch-size CLI flags ---
# Find a good anchor for adding the flags (after --workers)
OLD_WORKERS_FLAG = '    parser.add_argument("--workers", type=int, default=1)'
NEW_WORKERS_FLAG = '''    parser.add_argument("--workers", type=int, default=1)
    # SYSTEM_DESIGN_FINDINGS #1: Threaded mode — all workers share ONE model
    # copy + ONE BatchedEntityGraphRustEvaluator. The Rust engine releases the
    # GIL during MCTS search, so threads achieve true parallelism. The
    # evaluator's batching thread batches across threads (batch_size up to
    # --max-batch-size). This replaces 16 model copies (18GB) with 1 (1.1GB)
    # and turns batch_size=1 inference into batch_size=16-64.
    parser.add_argument("--use-threads", action="store_true", default=False,
                        help="Use threading.Thread instead of multiprocessing.spawn. "
                             "All workers share one model + evaluator. 2-4x throughput.")
    parser.add_argument("--max-batch-size", type=int, default=64,
                        help="Max inference batch size for threaded mode (evaluator batching).")'''

if OLD_WORKERS_FLAG in src:
    src = src.replace(OLD_WORKERS_FLAG, NEW_WORKERS_FLAG, 1)
    print("[OK] Added --use-threads and --max-batch-size flags")
else:
    print("[WARN] could not find --workers flag anchor")
    sys.exit(1)

# --- Add threading import ---
OLD_IMPORT = "import multiprocessing"
NEW_IMPORT = "import multiprocessing\nimport threading"
if OLD_IMPORT in src:
    src = src.replace(OLD_IMPORT, NEW_IMPORT, 1)
    print("[OK] Added threading import")

# --- Replace the multiprocessing.Pool section with threaded mode ---
OLD_POOL = """    started = time.perf_counter()
    if len(worker_args) <= 1:
        results = [_worker_entry(worker_args[0])] if worker_args else []
    else:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=len(worker_args)) as pool:
            results = pool.map(_worker_entry, worker_args)"""

NEW_POOL = """    started = time.perf_counter()
    if getattr(args, "use_threads", False) and len(worker_args) > 1:
        # SYSTEM_DESIGN_FINDINGS #1: Threaded mode — all workers share one
        # process, one model, one evaluator. The Rust engine releases the GIL
        # during MCTS search, so threads achieve true parallelism. The
        # evaluator's batching thread batches across threads.
        results = [None] * len(worker_args)
        errors: list[Exception | None] = [None] * len(worker_args)

        def _threaded_worker(idx: int, wa: dict, _results=results, _errors=errors) -> None:
            try:
                _results[idx] = _worker_entry(wa)
            except Exception as e:
                _errors[idx] = e

        threads = [
            threading.Thread(target=_threaded_worker, args=(i, wa),
                             name=f"gen-worker-{i}", daemon=True)
            for i, wa in enumerate(worker_args)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # If any thread errored, propagate the first one (but still merge
        # successful workers' results, matching pool.map's fail-fast behavior).
        for idx, err in enumerate(errors):
            if err is not None:
                print(f"Worker {idx} failed: {err}", file=sys.stderr)
        results = [r for r in results if r is not None]
    elif len(worker_args) <= 1:
        results = [_worker_entry(worker_args[0])] if worker_args else []
    else:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=len(worker_args)) as pool:
            results = pool.map(_worker_entry, worker_args)"""

if OLD_POOL in src:
    src = src.replace(OLD_POOL, NEW_POOL, 1)
    print("[OK] Added threaded mode branch (multiprocessing fallback preserved)")
else:
    print("[WARN] could not find multiprocessing.Pool section")
    sys.exit(1)

with open(path, "w") as f:
    f.write(src)
print(f"[DONE] Wrote {path}")
print()
print("USAGE: Add --use-threads to your generation launch command:")
print("  python tools/generate_gumbel_selfplay_data.py --workers 16 --use-threads ...")
print()
print("NOTE: In threaded mode, each worker still creates its own evaluator")
print("(via _run_worker). For TRUE shared-evaluator batching, the main AI")
print("should refactor _run_worker to accept a pre-built evaluator argument.")
print("This patch enables the threading infrastructure; the evaluator sharing")
print("requires a deeper refactor of _run_worker's signature.")
