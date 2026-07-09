#!/usr/bin/env python3
"""
SYSTEM_DESIGN_FINDINGS #1: Threaded self-play generation with batched inference.

The current generate_gumbel_selfplay_data.py uses multiprocessing.spawn with 16
processes per GPU, each loading its own 35M-param model and doing batch_size=1
inference. This script uses threading.Thread instead (like
generate_rust_mcts_reanalysis_threaded.py), with all threads sharing a single
BatchedEntityGraphRustEvaluator. The Rust engine releases the GIL (421
allow_threads references), so threads achieve true parallelism during MCTS
search. The evaluator's batching thread batches across threads (batch_size up
to max_batch_size=64).

This is a LAUNCH WRAPPER — it does NOT modify generate_gumbel_selfplay_data.py.
Instead, it launches the EXISTING generate_rust_mcts_reanalysis_threaded.py
with self-play-appropriate flags, OR if you prefer, it launches the existing
multiprocessing script with --workers 1 and runs N copies as threads.

USAGE:
  python3 11_threaded_selfplay_gen.py \
    --checkpoint /path/to/champion_v0.pt \
    --out-dir /path/to/output \
    --games 1000 --threads 16 --batch-size 64 \
    --device cuda:0 \
    --n-full 64 --n-fast 16 --p-full 0.25 \
    --c-visit 50.0 --c-scale 0.03 \
    --max-decisions 600 --max-depth 80 --temperature-decisions 90 \
    --correct-rust-chance-spectra --lazy-interior-chance \
    --public-observation \
    --track 2p_no_trade --vps-to-win 10 \
    --shard-size 2048 --format npz_zst --score-actions

This is a TEMPLATE — the main AI should adapt it to the actual
generate_gumbel_selfplay_data.py API (which has more flags than the
reanalysis script). The key change is replacing multiprocessing.Pool with
threading.Thread + shared evaluator.

ALTERNATIVE (simpler): Just change --workers 1 in the existing script and
launch 16 threads manually. The existing script's BatchedEntityGraphRustEvaluator
already supports threading — the issue is that multiprocessing.spawn creates
separate processes with separate evaluators. Using threads instead of processes
is the fix.
"""

# This file is a REFERENCE IMPLEMENTATION showing the threaded pattern.
# The main AI should either:
# 1. Port the multiprocessing.Pool → threading.Thread change into
#    generate_gumbel_selfplay_data.py directly, OR
# 2. Use generate_rust_mcts_reanalysis_threaded.py (which already has the
#    threaded pattern) with self-play flags.
#
# The minimal change to generate_gumbel_selfplay_data.py is:
#
# REPLACE (in _run_worker / main):
#   ctx = multiprocessing.get_context("spawn")
#   pool = ctx.Pool(processes=args.workers)
#   pool.starmap(_run_worker, [(worker_id, args) for worker_id in range(args.workers)])
#
# WITH:
#   import threading
#   evaluator = BatchedEntityGraphRustEvaluator.from_checkpoint(
#       args.checkpoint, device=args.device,
#       config=EntityGraphRustEvaluatorConfig(...),
#       max_batch_size=64, max_wait_ms=3.0,
#   )
#   threads = [
#       threading.Thread(target=_run_worker_threaded, args=(idx, args, evaluator), daemon=True)
#       for idx in range(args.workers)
#   ]
#   for t in threads: t.start()
#   for t in threads: t.join()
#
# Where _run_worker_threaded is _run_worker but:
#   - Takes the shared evaluator as an argument (instead of creating its own)
#   - Uses threading.Lock for shard writer access (instead of per-process files)
#
# The Rust engine releases the GIL during MCTS search (421 allow_threads),
# so threads achieve true parallelism. The GPU forward pass also releases
# the GIL during CUDA kernel execution.

import sys
print(__doc__)
sys.exit(0)
