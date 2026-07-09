# System-Design Findings & Optimization Opportunities

**Date:** 2026-07-09
**Scope:** Read-only analysis of the Catan self-play generation + training pipeline.
**Method:** Code inspection (generation script, MCTS search, neural evaluator, training loop, harvest pipeline) + live fleet telemetry (GPU util, memory, power, process counts, per-worker progress).

---

## Executive Summary

The fleet is now MPS-converted (4.4x throughput gain realized), but the underlying generation architecture has a **fundamental inefficiency**: 16 separate processes per GPU each load their own 35M-param model copy and do batch_size=1 inference, when the codebase already contains a `BatchedEntityGraphRustEvaluator` designed for multi-threaded batched inference. The batching infrastructure is completely wasted by the multiprocessing.spawn pattern. This is the single biggest optimization opportunity — estimated **2-4x additional throughput** on top of MPS, bringing the combined speedup to **~8-18x** over the original non-MPS baseline.

Additionally, the inference path lacks bf16 mixed precision and `torch.compile`, both of which are standard H100 optimizations with negligible accuracy impact.

---

## Finding 1: Batch-Size-1 Inference Despite Batched Evaluator (CRITICAL)

### Evidence

- **GPU telemetry:** 88% SM utilization, **6% memory utilization**, 47% power draw (330W/700W TDP). This is the classic kernel-launch-bound signature — the GPU is "busy" context-switching between tiny kernels, not doing real computation.
- **Process model:** `generate_gumbel_selfplay_data.py` uses `multiprocessing.get_context("spawn")` with `ctx.Pool(processes=16)`. Each worker calls `BatchedEntityGraphRustEvaluator.from_checkpoint()` in `_run_worker()`, loading its own model copy.
- **Memory per worker:** ~1,144 MiB GPU memory per process (model + CUDA context). 16 workers = ~18 GB per GPU just for model copies.
- **Batching thread never batches:** `BatchedEntityGraphRustEvaluator._batch_loop()` has a `_observed_concurrency` flag that gates the `max_wait_ms` straggler timer. This flag only flips when a batch contains >1 request. Since each worker process has its own evaluator with its own single-threaded caller, **the flag never flips**. Every forward pass is batch_size=1.
- **The batching code is literally wasted:** The queue, the `_batch_loop`, the `max_wait_ms` timer, the `_merge_batched_eval_requests` padding logic — all of it runs with exactly 1 request per batch, forever.

### The Fix (design-level)

The codebase already has the correct architecture in `tools/generate_rust_mcts_reanalysis_threaded.py`:
- Uses `threading.Thread` (not `multiprocessing.spawn`)
- All threads share a single `BatchedEntityGraphRustEvaluator`
- The evaluator's batching thread actually batches across threads (batch_size up to `max_batch_size=64`)

The self-play generator should adopt this pattern: **1 process per GPU, 16 threads sharing 1 model**. The Rust engine releases the GIL (421 `allow_threads` references in the pyo3 .so, confirmed via `strings`), so threads achieve true parallelism during MCTS search. Only the Python-side featurization and the GPU forward pass take the GIL, and the forward pass is a single `torch.no_grad()` call that releases the GIL during CUDA kernel execution.

### Expected Impact

| Metric | Current (16 procs, batch=1) | Threaded (1 proc, batch=16) |
|--------|----------------------------|------------------------------|
| Model copies per GPU | 16 (18 GB) | 1 (1.1 GB) |
| GPU forward batch size | 1 | 16-64 |
| H2D transfers per second | 16× small | 1× large |
| Kernel launch overhead | Dominates | Amortized |
| SM utilization type | Launch-bound | Compute-bound |
| Estimated throughput gain | 1× (baseline) | **2-4x** |

### Why MPS Doesn't Fix This

MPS solves CUDA context switching (16 contexts → 1 MPS server), which is why we see 4.4x. But MPS doesn't solve batch_size=1 — each client still sends a tiny kernel to the MPS server, which serializes them. The MPS server can overlap kernels from different clients, but it can't merge them into a single batched matmul. Threading + shared evaluator does merge them.

---

## Finding 2: No Mixed Precision at Inference (HIGH)

### Evidence

- **Training** uses `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` (line 4829 of `train_bc.py`).
- **Inference** (`forward_legal_np` in `entity_token_policy.py`) has no autocast. All inference runs in fp32.
- The model is 35M params, 640 hidden, 6 transformer layers — small enough that kernel launch overhead dominates at batch_size=1, but bf16 would halve the memory bandwidth and double the FLOPS on H100's tensor cores.

### Fix

Wrap the forward pass in `forward_legal_np` with `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`:
```python
with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
    outputs = self.model(batch, return_q=return_q)
```

### Expected Impact

~1.5-2x inference throughput on H100. Combined with batching (Finding 1), this stacks multiplicatively.

---

## Finding 3: No torch.compile (MEDIUM)

### Evidence

- No `torch.compile` call anywhere in the inference or training path.
- The model has 6 transformer blocks with custom attention — exactly the kind of Python-heavy forward that `torch.compile` (with `mode="reduce-overhead"`) excels at fusing.

### Fix

```python
self.model = torch.compile(self.model, mode="reduce-overhead")
```

### Caveats

- First-call compilation overhead (~30s) — acceptable for long-running generation.
- Dynamic shapes (variable legal action counts) may cause recompilation. The padding in `_merge_batched_eval_requests` already normalizes to `max_legal` per batch, but this varies across batches. `torch.compile` with `dynamic=True` or a fixed `max_legal` pad would help.

### Expected Impact

~1.2-1.5x on top of bf16 + batching, primarily from kernel fusion reducing launch overhead.

---

## Finding 4: shard_size=2048 Too Large for n128 Teacher Generation (MEDIUM)

### Evidence

- c2 (teacher n128) has **44,012 total rows** across 64 workers (~688 rows/worker) after ~20 min, but **0 npz shards** because no worker has reached 2048 rows.
- `games_completed_local=0` in all progress.json files — this counter only advances on shard flush, so it's a symptom, not a bug.
- Max rows in any single c2 worker: 885. Needs 1,163 more for first shard.
- At n128, each game takes ~5-10 min and produces ~100-200 rows. First shard per worker: ~50-100 min.
- The original "c2 stall bug" from the previous session was this same issue — not a bug, just a configuration mismatch.

### Fix

Use a smaller `--shard-size` for teacher (n128) generation:
- n64 volume: `--shard-size 2048` (current, fine — games are fast)
- n128 teacher: `--shard-size 512` (4x faster corpus availability)
- n256 probe: `--shard-size 256` (8x faster, games are very slow)

### Expected Impact

Faster feedback loop for teacher corpus availability. First teacher shards arrive in ~15 min instead of ~60+ min. This gates gen-5 v1 training start.

---

## Finding 5: Uncompressed Shard Format (LOW-MEDIUM)

### Evidence

- Shards are written with `np.savez` (uncompressed). Each shard is **43.7 MB** on disk.
- The code has a `npz_zst` format option (`_try_zstd`) but it's not used (`--format npz` in all current runs).
- Harvest uses `rsync -az` (compression enabled for network), but on-disk storage is uncompressed.
- With the fleet producing thousands of shards, disk usage scales: 10,000 shards = 437 GB uncompressed vs ~100-150 GB with zstd.

### Fix

Switch to `--format npz_zst` for generation runs. The writer already supports it:
```python
if self.format == "npz_zst":
    path = _try_zstd(path)
```

### Trade-off

zstd compression adds ~0.5-1s per shard flush (CPU-bound, happens once per 2048 rows). Negligible vs the ~minutes of GPU time to produce 2048 rows. Decompression during training is also fast (zstd is ~1GB/s decompress).

---

## Finding 6: Training Loss Weighting — Value Loss Underweighted (MEDIUM, research-level)

### Evidence

- `--policy-loss-weight` default: 1.0
- `--value-loss-weight` default: 0.25
- `--final-vp-loss-weight` default: 0.05
- `--winner-sample-weight` default: 1.0
- `--loser-sample-weight` default: 0.3

The value loss is only 25% of the policy loss weight. In AlphaZero-style training, policy and value are typically co-equal (1:1 or 0.5:0.5). The 4:1 ratio toward policy means the value head learns 4x slower than the policy head.

Additionally, `loser-sample-weight=0.3` means losing positions get 30% weight. While this makes sense for policy distillation (you don't want to imitate losing moves), it's questionable for **value learning** — the value head needs to learn from ALL positions, including losing ones, to predict them correctly. Underweighting losers could bias the value head toward optimism.

The B200 logs show "value_rescue" training runs, which is consistent with a value head that's undertrained relative to policy.

### Hypothesis

The value head is undertrained because:
1. Loss weight is 0.25 vs policy's 1.0
2. Loser positions (where value prediction matters most for learning) are downweighted to 0.3
3. The `value_weight_multiplier=1` in the generation code is correct (all rows get value targets), but the training-side weighting undoes this

### Suggested Experiment

- Try `--value-loss-weight 0.5` or `--value-loss-weight 1.0` (equal to policy)
- Try `--loser-sample-weight 0.7` or `1.0` for value loss specifically (decouple policy/value sample weighting)
- A/B via a gate: train two models (current weights vs rebalanced) and run a head-to-head

---

## Finding 7: No LR Scheduler Decay (LOW, research-level)

### Evidence

- Training uses linear warmup → constant LR. No cosine annealing, no step decay.
- `--lr` default: 2e-4, held constant after warmup.
- For BC/distillation training (2 epochs), this may be fine. But for longer training runs (gen-5 v1 grow-from-champion), a decay schedule would help convergence in later epochs.

### Suggested Experiment

Cosine annealing from `--lr` to `--lr * 0.1` over the total step count. Standard for transformer training.

---

## Finding 8: Adam Optimizer State Not Persisted (LOW)

### Evidence

- Code comment (line 174): "Checkpoints do not persist optimizer state, so every resume restarts Adam's moment estimates from zero."
- `--lr-warmup-steps` exists as a mitigation (short ramp protects from fresh-Adam transient).

### Impact

Resume after preemption loses Adam momentum + variance estimates, causing a transient period of suboptimal updates. With the warmup mitigation, this is manageable but not ideal.

### Fix

Save `optimizer.state_dict()` alongside the model checkpoint. Standard PyTorch pattern.

---

## Finding 9: EvalServer Stub — Cross-Process Batching Never Implemented (INFO)

### Evidence

- `src/catan_zero/search/eval_server.py` is an **interface stub** (CAT-67, Phase D).
- It was designed to be a "THIRD batching layer, on top of the existing per-game batch API" — a shared eval server that multiple processes connect to via MP queues or shared memory.
- The docstring explicitly acknowledges the problem: "featurize-bound, in-process batch size ~1, MPS does not already close the gap."
- It was never implemented.

### Relevance

This confirms the team identified the batch_size=1 problem (Finding 1) but chose MPS as the mitigation instead. MPS helps (4.4x) but doesn't fully solve it. The thread-based approach (Finding 1's fix) is simpler than the EvalServer and doesn't need a new transport layer.

---

## Finding 10: H2D Transfer Without pin_memory/non_blocking (LOW)

### Evidence

- `forward_legal_np` does `torch.as_tensor(value, device=self.device)` for each entity batch key — synchronous CPU→GPU copy.
- No `pin_memory=True` or `non_blocking=True` anywhere in the inference path.
- At batch_size=1 with 16 processes, this is 16× the H2D transfer overhead.

### Fix

With the thread-based architecture (Finding 1), this becomes less critical (1 large transfer instead of 16 small ones). But for further optimization:
```python
# Pre-pin host arrays, then async transfer
batch = {
    key: torch.from_numpy(value).pin_memory().to(self.device, non_blocking=True)
    for key, value in entity_batch.items()
}
```

---

## Finding 11: MPS + EXCLUSIVE_PROCESS Incompatibility (OPERATIONAL)

### Evidence

- Setting H100 GPUs to EXCLUSIVE_PROCESS compute mode crashes the MPS server (all clients lose contexts, status 806 in server log).
- c5 (stable, 47+ min) uses Default mode. c1 crashed repeatedly in EXCLUSIVE_PROCESS.
- Fix already deployed: `mps_rollout.sh` patched with `nvidia-smi -c DEFAULT` preflight on c1/c4/c5.

### Note

This is already fixed but should be documented in any deployment guide. The NVIDIA MPS documentation recommends EXCLUSIVE_PROCESS, but the H100 + driver 580 combination has this issue.

---

## Finding 12: a100a Pilot Uses WRONG c-scale (0.1 vs 0.03) — Generating Garbage Data (CRITICAL)

### Evidence

- The a100a box (8× A100) is running a `cat91_n64_pilot` self-play generation using the **old `catan-zero` stack** (commit `34b16d9`, the f70-era codebase), NOT the production `catan-zero-runsix` stack.
- The pilot launch command: `python tools/generate_gumbel_selfplay_data.py --checkpoint runs/bc/gen3_20260706/checkpoint.pt --out-dir runs/selfplay/cat91_n64_pilot/gpu0 --n-full 64 --games 167 --base-seed 6100000000 --workers 4 --device cuda --max-decisions 600 --public-observation`
- **Missing flags:** `--c-scale 0.03`, `--c-visit 50.0`, `--n-fast 16`, `--p-full 0.25`, `--shard-size 2048`, `--format npz`, `--score-actions`, `--correct-rust-chance-spectra`, `--lazy-interior-chance`, `--track 2p_no_trade`, `--vps-to-win 10`
- The old stack defaults `--c-scale` to **0.1** (confirmed: `parser.add_argument("--c-scale", type=float, default=0.1)`).
- The production fleet uses `--c-scale 0.03` — a 3.3x difference in exploration scaling.
- The c-scale=0.1 value is the **known-broken pre-F1a/F1b calibration** that caused near-one-hot policy targets and 50-65% self-agreement collapse (per the F1 findings in the runsix repo).

### Impact

The pilot is generating self-play data with:
1. **Wrong search calibration** (c-scale=0.1 → over-exploitation, near-one-hot targets)
2. **No chance-spectrum correction** (missing `--correct-rust-chance-spectra` → A19/A20 bugs in robber/dev-card spectra)
3. **No lazy interior chance** (missing `--lazy-interior-chance` → 11x slower ROLL expansion at interior nodes)
4. **No score-actions** (missing `--score-actions` → no Q-score targets for training)
5. **No track/vps-to-win** (defaults may differ from production 2p_no_trade/10)
6. **Old Rust engine** (pre-batch-API → 11 sequential apply_chance_outcome calls instead of 1 batch call)

Any data from this pilot is **incompatible** with the production fleet corpus. Mixing it into gen-5 training would poison the policy targets with the exact near-one-hot collapse the F1 fixes were designed to prevent.

### Fix

Kill the pilot and relaunch with the production `catan-zero-runsix` stack and the full flag set matching the fleet:
```bash
python tools/generate_gumbel_selfplay_data.py \
  --checkpoint ... --out-dir ... --games ... --workers 16 --device cuda \
  --n-full 64 --n-fast 16 --p-full 0.25 --c-visit 50.0 --c-scale 0.03 \
  --max-decisions 600 --max-depth 80 --temperature-decisions 90 \
  --correct-rust-chance-spectra --lazy-interior-chance --public-observation \
  --track 2p_no_trade --vps-to-win 10 --shard-size 2048 --format npz --score-actions
```

---

## Finding 13: a100a GPU6 Idle — 1 of 8 A100s Wasted (HIGH)

### Evidence

- a100a has 8× A100 GPUs. Current utilization:
  - gpu0-5: 32-56% util, 4 worker procs each (pilot, 6 GPUs for 3 launch dirs — 2 GPUs per launch)
  - gpu6: **0% util, 0 MiB** — completely idle
  - gpu7: 96% util (gate A-vs-Gen3 h2h match running)
- The pilot uses `--workers 4` per launch (not 16 like the fleet), and spreads across 6 GPUs for only 3 launch dirs (gpu0-5), leaving gpu6 idle.

### Fix

Either:
1. Launch a 7th pilot dir on gpu6, or
2. Consolidate to fewer GPUs with more workers per GPU (matching the fleet's 16-worker pattern), or
3. Better: kill the pilot (Finding 12) and relaunch with the runsix stack using all 8 GPUs at 16 workers each.

---

## Finding 14: a100a Pilot Uses workers=4 (Not 16) — 4x Underutilization (HIGH)

### Evidence

- Fleet (H100s): `--workers 16` per GPU → 16 MCTS processes per GPU
- a100a pilot: `--workers 4` per GPU → only 4 MCTS processes per GPU
- A100 gpu0-5 show 32-56% util with 4 workers — far below saturation
- The A100 has fewer SMs than H100 (108 vs 132) but 4 workers is still drastically undersaturated

### Fix

Use `--workers 16` (or at least 8) on A100s. The batch-size-1 problem (Finding 1) means more workers = more GPU utilization up to the kernel-launch ceiling.

---

## Finding 15: Evaluator Cache Uses FIFO Eviction, Not LRU (MEDIUM)

### Evidence

- `neural_rust_mcts.py` line 426: `self._cache.pop(next(iter(self._cache)))`
- This is FIFO (first-in-first-out) eviction, not LRU.
- Python dicts preserve insertion order (3.7+), so `next(iter(self._cache))` returns the oldest-inserted key.
- The cache is never "touched" on read — a `self._cache.get(key)` doesn't move the key to the end.
- In MCTS, the same position can be visited multiple times across simulations. A position that was inserted early but is still being accessed will be evicted by a newer position that may never be accessed again.

### Impact

For a 100K-entry cache with ~7K sims/game and ~16 sims/decision, the cache hit rate matters. FIFO evicts the oldest entries regardless of whether they're still being accessed. In MCTS, early-game positions are revisited frequently (the root and its children are expanded every simulation), so evicting them is wasteful.

### Fix

Use `collections.OrderedDict` and `move_to_end` on cache hit:
```python
from collections import OrderedDict
self._cache = OrderedDict()
# On hit:
cached = self._cache.get(cache_key)
if cached is not None:
    self._cache.move_to_end(cache_key)
    return ...
# On insert (eviction):
if len(self._cache) >= int(self.config.cache_size):
    self._cache.popitem(last=False)  # LRU eviction
self._cache[cache_key] = ...
```

### Expected Impact

Modest — maybe 5-15% cache hit rate improvement, which translates to fewer evaluator calls. The cache is per-worker (not shared), so the absolute impact depends on how many transpositions occur within a single game's MCTS tree.

---

## Finding 16: Per-Worker Evaluator Cache — No Cross-Worker Sharing (MEDIUM)

### Evidence

- Each worker process has its own `BatchedEntityGraphRustEvaluator` with its own `self._cache` dict.
- 16 workers per GPU = 16 independent 100K-entry caches = 1.6M total cache entries per GPU, but no sharing.
- In self-play, both players use the same model. Positions from the same game are evaluated by the same worker (since one worker plays one full game), so within-game transpositions are cached. But cross-game transpositions (same board state reached via different move orders) are never shared.

### Impact

Catan has a moderate transposition rate — the same board state can be reached via different build orders. With 16 independent caches, the same position may be evaluated 16 times across workers.

### Fix

With the threaded architecture (Finding 1), all threads share one evaluator → one cache. This automatically solves the cross-worker cache sharing problem. The `BatchedEntityGraphRustEvaluator` already has a `self._cache_lock = threading.Lock()` (line 746), confirming it was designed for thread-safe shared cache access.

### Expected Impact

Modest — cross-game transpositions in Catan are less common than in chess/Go. But it's a free improvement that comes with the threaded architecture.

---

## Finding 17: 51% of Decisions Are Forced (Single Legal Action) — Wasted Search Budget (MEDIUM)

### Evidence

- Fleet telemetry (20 workers sampled): 69,451 total decisions, 35,779 forced (51.4%).
- Forced decisions (single legal action, e.g. ROLL) skip MCTS search entirely (`_expand_forced` returns immediately, `simulations_used=0`).
- But they still call `evaluator.evaluate()` once for the value target (except ROLL, which enumerates 11 outcomes via `evaluate_many`).
- The remaining 48.9% of decisions get ~33 simulations each (16 sims/decision averaged over all decisions, 33 over searched-only).

### Implication

The n_full=64, n_fast=16 budget is spent on only ~49% of decisions. The other 51% are forced ROLLs (and occasional forced discards). This is expected for Catan (ROLL is always forced when it's your turn), but it means:
1. The effective search budget per "real" decision is ~33 sims, not 64.
2. The `--n-full 64` flag is misleading — it's only applied to ~49% of decisions.
3. The forced ROLL path still calls `evaluate_many` for 11 outcomes (one batched forward pass), which is the most expensive forced-decision path.

### Optimization Opportunity

The forced ROLL path calls `_enumerate_roll_outcomes` which does `evaluate_many` on 11 children. Each child is a post-roll state that will likely never be revisited (the actual roll is sampled at game-play time). This is necessary for the value target, but the 11 children's subtrees are never searched (they're leaf-expanded only). Consider whether the 11-outcome enumeration is worth the cost vs. a single value estimate of the pre-roll state.

---

## Finding 18: JSON Serialization in the MCTS Hot Path — ~1.3ms Per Leaf Eval (MEDIUM)

### Evidence

- Per leaf evaluation, the evaluator calls:
  1. `game.json_snapshot()` — 0.946ms, 25.7KB output
  2. `game.decision_context_json()` — 0.392ms, 2.9KB output
  3. `json.loads(decision_context_json)` — parse the 2.9KB
  4. `hashlib.blake2b(snapshot_text, digest_size=16)` — 0.006ms (negligible)
- Total JSON overhead: ~1.3ms per leaf eval
- For comparison, the GPU forward pass at batch_size=1 is ~0.5-1ms on H100
- **JSON serialization is 50-70% of the leaf-eval latency**

- The `apply_chance_outcome` path adds another `json.dumps(action_json)` per child (10 `json.dumps` calls in the hot path).

### Root Cause

The Rust engine communicates game state via JSON strings (pyo3 boundary). Every leaf eval serializes the full game state to JSON, passes it to Python, which parses it back. The `rust_featurize` flag (default OFF) was designed to bypass this by doing featurization in Rust, but it's not enabled in production.

### Fix

Enable `--rust-featurize` (the `rust_featurize` config flag). This was designed (task #81) to do board topology featurization in Rust, avoiding the JSON round-trip. The code exists (`_entity_batch_via_rust`, `_context_batch_via_rust`) but is gated off.

### Expected Impact

~1.3ms → ~0.1ms per leaf eval (10x reduction in JSON overhead). Combined with batched inference (Finding 1), this would make the GPU forward pass the dominant cost instead of JSON serialization.

---

## Finding 19: Harvest rsync Runs Serially Across Boxes and Directories (LOW-MEDIUM)

### Evidence

- `wave1_harvest.sh` `pull_dirs()` loops over directories serially: `for d in "$@"; do rsync ...; done`
- `harvest-volume` loops over boxes serially: `for b in c1 c4 c5; do pull_dirs $b volume ${DIRS[$b]}; done`
- With 3 boxes × 12 dirs each = 36 sequential rsync calls
- Each rsync opens a new SSH connection (no `ControlMaster` reuse)

### Fix

1. Parallelize across boxes: `for b in c1 c4 c5; do pull_dirs $b volume ${DIRS[$b]} & done; wait`
2. Parallelize within box: `for d in "$@"; do rsync ... & done; wait`
3. Use SSH ControlMaster for connection reuse:
```bash
SSH="ssh -o ControlMaster=auto -o ControlPath=/tmp/ssh-%r@%h:%p -o ControlPersist=60"
```

### Expected Impact

Harvest time reduced from ~36× single-rsync to ~1× (limited by the slowest box/dir). With 43.7MB shards and gigabit network, each rsync is ~0.4s, so 36 serial = ~14s vs ~1s parallel. Minor for small harvests, but scales with corpus size.

---

## Finding 20: build_memmap_corpus Is Single-Threaded (LOW-MEDIUM)

### Evidence

- `build_memmap_corpus.py` processes shards in a single `for shard_index, file in enumerate(files):` loop.
- Each shard: `_load_npz` (disk I/O) + `_normalize_teacher_shard` (CPU) + column writes (disk I/O).
- With 2,114 shards (3.9M rows), this is CPU+I/O bound and takes minutes.
- No `ThreadPoolExecutor` or `multiprocessing.Pool` for parallel shard processing.

### Fix

Parallelize with a thread pool (I/O-bound shards can overlap):
```python
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=8) as executor:
    for result in executor.map(process_shard, files):
        write_columns(result)
```

### Caveat

Column writes must be ordered (shards concatenate in source order), so the write phase needs to be sequential or buffered. A producer-consumer pattern (parallel load+normalize, sequential write) would work.

### Expected Impact

~4-8x faster corpus build (limited by disk I/O bandwidth). Minor for one-time builds, but matters if rebuilding frequently during iteration.

---

## Finding 21: Training Loads Entire Corpus Into RAM (npz mode) — OOM Risk (MEDIUM)

### Evidence

- The old `train_bc.py` (f70, used by the old stack) uses `load_teacher_data()` which loads ALL shards into RAM:
  ```python
  arrays: dict[str, list[np.ndarray]] = {}
  for file in files:
      shard = _normalize_teacher_shard(_load_npz(file), file)
      for key in keys:
          arrays.setdefault(key, []).append(shard[key])
  return {key: _concat_padded(key, values) for key, values in arrays.items()}
  ```
- The docstring of `build_memmap_corpus.py` explicitly states: "That ceiling OOM'd a 32.6M-row corpus on a 708GB host."
- The new `c1_fsdp/repo` stack has `--data-format memmap` which streams via `MemmapCorpus` (only per-batch rows in RAM).
- But the a100a training (A_edge_pol, B_plain_pol) uses the **new stack with memmap** — good.
- The old stack (f70) is still used for the a100a pilot generation, but not for training.

### Status

Partially fixed — the new stack has memmap support. But the old stack (still used in some places) does not. Ensure all training uses `--data-format memmap`.

---

## Finding 22: Training Batch Size 1024 on A100 — Possibly Too Small for 35M Model (LOW, research-level)

### Evidence

- A_edge_pol and B_plain_pol training: `--batch-size 1024` on a single A100 (80GB).
- Model: 35.4M params, batch profile shows `max_allocated_mib: 38816` (~39GB) — plenty of headroom on 80GB A100.
- The default batch size in train_bc.py is 65536, but the actual training uses 1024.
- With 3.9M rows and batch_size=1024, that's ~3,800 batches/epoch.
- The c1_fsdp smoke tests tried batch sizes 512-768 with larger models (hidden=1024, layers=11).

### Hypothesis

Batch size 1024 may be too conservative for a 35M model on an 80GB A100. Larger batches (4096-8192) would:
1. Better utilize the A100's 312 TFLOPS (fp16)
2. Reduce kernel launch overhead (fewer, larger batches)
3. Provide more stable gradients

### Suggested Experiment

Try `--batch-size 4096` or `--batch-size 8192` on A100. Monitor loss curve stability and validation accuracy. The 39GB allocation at bs=1024 suggests bs=8192 would use ~50-60GB (not linear due to fixed model overhead), well within 80GB.

---

## Finding 23: No Gradient Accumulation in Single-GPU Training (LOW)

### Evidence

- The c1_fsdp stack supports grad accumulation (`--grad-accum-steps`), but the a100a training uses single-GPU without accumulation.
- Batch size 1024 × 1 step = 1024 samples per update. No accumulation.
- For FSDP training (c1), the smoke test uses `--batch-size 768` with 2 GPUs = 1536 effective batch.

### Fix

If larger batches are desired but memory-limited, use `--grad-accum-steps 4` with `--batch-size 1024` for an effective batch of 4096.

---

## Finding 24: Checkpoint Load Takes 3.2s Per Worker — 16x Amplified at Startup (LOW)

### Evidence

- `EntityGraphPolicy.load()` takes 3.177s (0.241s import + 2.936s load+to(cuda)).
- Each of 16 workers loads independently → 16 × 3.2s = ~51s of startup time (parallel, so wall-clock is still ~3.2s, but 16× the CPU/disk I/O).
- The warmup forward pass is ~0ms (lazy CUDA init).

### Impact

Minor — 3.2s startup is negligible vs hours of generation. But with the threaded architecture (Finding 1), only 1 load per GPU instead of 16, reducing startup I/O by 16x.

---

## Finding 25: 10 json.dumps Calls in MCTS Hot Path (LOW)

### Evidence

- `gumbel_chance_mcts.py` has 10 `json.dumps` calls in the search path.
- Each `apply_chance_outcome(json.dumps(action_json), outcome_index)` serializes the action JSON to pass it back to the Rust engine.
- The Rust engine then deserializes it internally.

### Root Cause

The pyo3 boundary requires JSON strings for action passing. The `apply_chance_outcomes_batch` API takes `json.dumps(action_json)` + a list of indices, but still requires the JSON string.

### Fix

This is a Rust-side API issue. A future Rust API could accept action IDs directly (the action is already identified by its integer ID in the game state) instead of round-tripping through JSON. This would require changes to the `catanatron_rs` Rust crate.

### Expected Impact

Small — `json.dumps` of a small action dict is ~0.01ms. But it's 10 calls per simulation, so ~0.1ms/sim. Over 7K sims/game, that's ~0.7s/game of JSON overhead.

---

## Finding 26: Training Uses npz Format (Old Stack) vs memmap (New Stack) — Stack Fragmentation (INFO)

### Evidence

- Old stack (`catan-zero`, commit 34b16d9): `train_bc.py` has NO `--data-format` flag. Always loads all npz into RAM.
- New stack (`c1_fsdp/repo`, branch c1-multigpu): `train_bc.py` has `--data-format memmap` with `MemmapCorpus` streaming.
- The a100a training uses the new stack (good).
- The a100a pilot generation uses the old stack (bad — Finding 12).

### Status

The codebase is mid-migration. The new stack is strictly better (streaming, FSDP, memmap). The old stack should be deprecated once all workflows are ported.

---

## Finding 27: FSDP Smoke Test Uses Only 2 GPUs — Not Production-Scale (INFO)

### Evidence

- `smoke_fsdp4.sh`: `torchrun --nproc_per_node=2` — only 2 of 4 H100s on c1.
- Model: hidden=832, layers=7, heads=8 (smaller than the 35M champion_v0 which is hidden=640, layers=6).
- Only 4 max steps — a smoke test, not a real training run.

### Status

This is a smoke test, not a production run. But it confirms FSDP works. Production training should use all available GPUs with the full model architecture.

---

## Finding 28: A_edge_pol Training — 35.4M Params, Same as Champion_v0 (INFO)

### Evidence

- A_edge_pol: `parameter_count: 35453514` (35.4M)
- Champion_v0: 35,041,353 (35.0M)
- The edge training uses the same architecture as the champion, not a larger model.
- The c1_fsdp smoke tests tried larger models (hidden=1024, layers=11 → ~150M params) but only as smoke tests.

### Observation

The gen-5 edge training is same-architecture BC from the gen-3 corpus. No model scaling is being attempted yet. The FSDP infrastructure exists for larger models but isn't being used.

---

## Finding 29: THREE Incompatible Code Forks — Stack Fragmentation (CRITICAL)

### Evidence

There are **three separate forks** of the Catan codebase, with incompatible model architectures:

| Fork | Location | Commit | Branch | edge_policy_head |
|------|----------|--------|--------|-----------------|
| Old stack | a100a `/home/ubuntu/catan-zero` | 34b16d9 | master | NO |
| CAT-97 edge | a100a `/home/ubuntu/cat97-a100a` | 8cdd509 | cat-97-graph-heads | **YES** |
| Production | fleet c1-c6 `/home/ubuntu/catan-zero-runsix` | various | runsix | NO |

**Model checkpoint incompatibility:**
- `champion_v0.pt` (production): 139 model keys, NO `edge_policy_mlp`
- `B_plain_pol.pt` (old stack): 139 model keys, NO `edge_policy_mlp`
- `A_edge_pol.pt` (CAT-97 fork): **145 model keys**, HAS `edge_policy_mlp.0.weight` through `edge_policy_mlp.4.bias`

**A_edge_pol.pt CANNOT be loaded by the production fleet stack.** The runsix `EntityGraphNet` does not have `edge_policy_mlp` layers, so loading would fail with unexpected keys. The gate match on a100a gpu7 works only because it sets `PYTHONPATH=/home/ubuntu/cat97-a100a/src` to use the CAT-97 fork.

### Impact

1. **A_edge_pol is stranded** — it can only be used on a100a with the CAT-97 fork. The fleet cannot generate self-play data from it.
2. **If A_edge_pol wins the gate**, it cannot be deployed to the fleet without either:
   a. Porting `edge_policy_head` to the runsix stack, or
   b. Stripping the `edge_policy_mlp` weights and re-evaluating (which changes the model)
3. **The pilot on a100a gpu0-5** (Finding 12) uses the OLD stack (`/home/ubuntu/catan-zero`), generating data with c-scale=0.1 and no chance corrections. This data is incompatible with both the CAT-97 fork and the production runsix stack.
4. **Three-way fragmentation** means improvements in one fork don't propagate to others without explicit porting.

### Fix

1. **Immediate:** Decide which fork is canonical. The production fleet uses runsix. All training and generation should use runsix.
2. **If A_edge_pol wins the gate:** Port `edge_policy_head` to runsix, retrain, or strip the edge head and re-evaluate.
3. **Kill the old-stack pilot** (Finding 12) — it's generating incompatible data.
4. **Consolidate:** Deprecate the old `catan-zero` and `cat97-a100a` forks. Merge any useful features (edge_policy_head) into runsix.

---

## Finding 30: Seed Ledger Is Local Stubs — No Cross-Host Collision Detection (HIGH)

### Evidence

- `SEED_LEDGER.md` on c1 explicitly states: "NOT the cross-host authoritative ledger... the fleet does not have live cross-host collision visibility until that file is synced here."
- Each fleet box has its own local stub ledger with its assigned base-seed block.
- The prelaunch guard checks for collisions within the local `.seed_claims/` directory and the local `SEED_LEDGER.md`, but **not across hosts**.
- Cross-host seed disjointness is enforced "by convention" (orchestrator assigns blocks), not by runtime checks.

### Impact

If two hosts accidentally use overlapping seed ranges (human error in orchestration), the same games would be generated twice. This would:
1. Waste GPU time
2. Create duplicate rows in the training corpus (the `_GameSeedRunTracker` in `build_memmap_corpus.py` would catch within-corpus duplicates, but only at build time, not at generation time)
3. Potentially bias training if duplicates aren't caught

### Fix

Sync the master `SEED_LEDGER.md` to all fleet boxes, or use a shared distributed lock (e.g., a file on shared storage, or a simple Redis instance) for seed range claims.

---

## Finding 31: All Optional Model Heads Disabled in champion_v0 (INFO)

### Evidence

champion_v0 config (the production model):
- `value_uncertainty_head: False` — no KataGo-style uncertainty prediction
- `value_categorical_bins: 0` — no HL-Gauss distributional value head
- `value_attention_pool: False` — no attention-based value pooling
- `action_target_gather: False` — no target-entity gather for actions
- `action_cross_attention_layers: 0` — no action-trunk cross-attention
- `belief_chance_spectra: False` — no belief-based chance resolution

All of these are implemented in the codebase but **none are enabled** in the production model. The model is the simplest possible configuration: 6 transformer layers, hidden=640, 8 heads, basic MLP value head, dot-product policy head.

### Implication

The codebase has significant architectural upgrades ready but unused:
1. **Distributional value (CAT-39):** HL-Gauss categorical value head — shown to beat MSE for stochastic dynamics. Catan is highly stochastic (dice rolls). This could improve value prediction significantly.
2. **Value uncertainty (CAT-61):** KataGo-style short-term-error prediction — enables uncertainty-weighted MCTS backup. Could reduce variance in search.
3. **Action cross-attention (f69):** Actions attend to board tokens — could improve policy quality on complex decisions (settlement placement).

### Suggested Experiments

1. Enable `value_categorical_bins=51` (HL-Gauss) and retrain from champion_v0 warm start
2. Enable `value_uncertainty_head=True` and use `uncertainty_backup_weighting` in search
3. Enable `action_cross_attention_layers=2` for the settlement-placement decisions

These are all designed to be warm-start safe (zero-initialized at init, bit-identical to base model), so they can be added without restarting training from scratch.

---

## Finding 32: A_edge_pol Has 6 Extra Model Keys (edge_policy_mlp) — 35.4M vs 35.0M Params (INFO)

### Evidence

- champion_v0: 35,041,353 params (139 model keys)
- A_edge_pol: 35,453,514 params (145 model keys) — 412K more params
- The extra params are `edge_policy_mlp`: Linear(h, h) + GELU + ... + Linear(h, 1) = ~410K params (640×640 + 640×1 ≈ 410K)
- The edge policy head emits a direct logit from edge-token features, added to the main policy logits.

### Status

This is the CAT-97 "GATEAU edge-feature policy head" — an experimental architectural upgrade. It's only in the cat97-a100a fork. If it proves effective in the gate match, it needs to be ported to the runsix stack.

---

## Priority Ranking

| # | Finding | Impact | Effort | Priority |
|---|---------|--------|--------|----------|
| 29 | THREE incompatible forks — A_edge_pol stranded | **Architecture fragmentation** | Consolidate | **P0** |
| 12 | a100a pilot wrong c-scale (0.1 vs 0.03) + old stack | **Garbage data** | Kill+relaunch | **P0** |
| 30 | Seed ledger local stubs — no cross-host detection | Duplicate games | Sync ledger | **P0** |
| 1 | Batch-1 inference → threaded batch-16 | **2-4x** | Medium (refactor worker model) | **P0** |
| 2 | No bf16 at inference | **1.5-2x** | Trivial (1 line) | **P0** |
| 13 | a100a GPU6 idle (1/8 wasted) | 12.5% capacity | Trivial (launch dir) | **P1** |
| 14 | a100a workers=4 (not 16) | 4x underutil | Trivial (flag) | **P1** |
| 3 | No torch.compile | 1.2-1.5x | Low (1 line + testing) | **P1** |
| 4 | shard_size too large for n128 | Faster feedback | Trivial (CLI flag) | **P1** |
| 6 | Value loss underweighted | Research-level | Experiment | **P1** |
| 18 | JSON serialization ~1.3ms/leaf | 10x leaf overhead | Enable --rust-featurize | **P1** |
| 31 | All optional heads disabled (distributional value, etc) | Research | Experiment | **P1** |
| 15 | FIFO cache eviction (not LRU) | 5-15% hit rate | Low (OrderedDict) | **P2** |
| 16 | No cross-worker cache sharing | Modest | Free with Finding 1 | **P2** |
| 17 | 51% forced decisions — budget allocation | Research | Experiment | **P2** |
| 5 | Uncompressed shards | Disk/storage | Trivial (CLI flag) | **P2** |
| 7 | No LR decay | Convergence | Low | **P2** |
| 8 | Adam state not persisted | Resume quality | Low | **P2** |
| 19 | Harvest rsync serial | ~14s→~1s | Low (parallelize) | **P2** |
| 20 | build_memmap single-threaded | 4-8x build | Medium | **P2** |
| 21 | npz loads all RAM (old stack) | OOM risk | Use memmap | **P2** |
| 10 | No pin_memory/non_blocking | H2D overhead | Low | **P3** |
| 22 | Batch size 1024 too small? | Research | Experiment | **P3** |
| 23 | No grad accumulation (single-GPU) | Effective batch | Low | **P3** |
| 24 | Checkpoint load 3.2s × 16 workers | Startup I/O | Free with Finding 1 | **P3** |
| 25 | 10 json.dumps in MCTS hot path | ~0.7s/game | Rust API change | **P3** |
| 32 | A_edge_pol 412K extra params (edge head) | Info | Port if wins | **P3** |
| 9 | EvalServer stub | Info only | — | **P3** |
| 11 | MPS + EXCLUSIVE_PROCESS | Operational | Already fixed | **Done** |
| 26 | Stack fragmentation (old vs new) | Info | Deprecate old | **P3** |
| 27 | FSDP smoke test 2 GPUs | Info | Scale up | **P3** |
| 28 | No model scaling attempted | Info | Future | **P3** |

---

## Combined Throughput Projection

If Findings 1+2+3 are implemented together:

| Stage | Throughput |
|-------|-----------|
| Pre-MPS baseline | 1× |
| + MPS (current) | 4.4× |
| + Threaded batched inference (Finding 1) | ~8-13× |
| + bf16 inference (Finding 2) | ~12-20× |
| + torch.compile (Finding 3) | ~15-25× |

The 15-25× projection assumes the model is currently 100% launch-bound (which the 6% memory util confirms). Once batched, the H100's 989 TFLOPS bf16 tensor cores become the bottleneck instead of kernel launch latency, and a 35M param model at batch=16-64 is still small enough to be heavily compute-underutilized — meaning even larger batch sizes (32+ workers) would continue to scale.

---

## Finding 33: Opponent Pool Evaluator Cache Never Evicts — Unbounded GPU Memory (HIGH)

### Evidence

- `gumbel_self_play.py` lines 1317, 1321: `pool_evaluator_cache: dict[str, RustEvaluator] = {}` and `mix_evaluator_cache: dict[str, RustEvaluator] = {}`.
- These caches grow without limit — every distinct opponent checkpoint path that gets sampled stays in GPU memory for the worker's entire lifetime.
- Each loaded `BatchedEntityGraphRustEvaluator` holds ~1.1 GB GPU memory (model weights + CUDA context).
- The `MixRuntime` docstring (line 310) explicitly warns: "a worker holds one resident model per distinct checkpoint it has sampled so far (the evaluator cache above never evicts), so a mix with several large 'older_champion'/'hard_experimental' checkpoints can pin `n_distinct_checkpoints_sampled x model_size` of device memory per worker."
- With 16 workers per GPU and an opponent pool of 10+ archived checkpoints, a single worker could load 10+ models = 11+ GB, on top of the champion's 1.1 GB. With 16 workers, this is 16 × 12 GB = 192 GB — far exceeding the 80 GB A100/H100.

### Impact

In practice, the opponent pool fraction (`--opponent-pool-fraction`) limits how many games use pool opponents, so not every worker samples every checkpoint. But over a long run (1000+ games per worker), a worker can sample many distinct checkpoints, slowly filling GPU memory until OOM.

### Fix

Add an LRU eviction policy to `pool_evaluator_cache` and `mix_evaluator_cache`:
```python
from collections import OrderedDict
pool_evaluator_cache: OrderedDict[str, RustEvaluator] = OrderedDict()
MAX_POOL_EVALUATORS = 3  # Keep at most 3 opponent models resident

# On hit:
opponent_evaluator = pool_evaluator_cache.get(choice.path)
if opponent_evaluator is not None:
    pool_evaluator_cache.move_to_end(choice.path)
else:
    opponent_evaluator = opponent_pool.evaluator_factory(choice.path)
    pool_evaluator_cache[choice.path] = opponent_evaluator
    if len(pool_evaluator_cache) > MAX_POOL_EVALUATORS:
        _evicted_path, _evicted_eval = pool_evaluator_cache.popitem(last=False)
        del _evicted_eval  # Free GPU memory
        torch.cuda.empty_cache()
```

---

## Finding 34: Weight Decay Default = 0.0 — No Regularization (LOW-MEDIUM)

### Evidence

- `train_bc.py` line 324: `parser.add_argument("--weight-decay", type=float, default=0.0, ...)`.
- The 35M param model is trained on 3.9M rows for 2 epochs — enough data that overfitting is unlikely, but some weight decay (1e-4 to 1e-5) is standard for transformer training and helps generalization.
- The code correctly uses AdamW (decoupled weight decay) when `--optimizer adamw` is passed, and refuses to apply weight decay with plain Adam (audit fix at line 6664).
- But the default optimizer is `adam` (not `adamw`), so even if `--weight-decay` is set, it would be refused unless `--optimizer adamw` is also passed.

### Fix

For gen-5 training, use:
```bash
--optimizer adamw --weight-decay 1e-4
```

This is a CLI flag change, not a code change. The infrastructure already supports it correctly.

---

## Finding 35: Symmetry-Averaged Eval Skips Adapter Resolution Cache (LOW)

### Evidence

- `neural_rust_mcts.py` line 430+: `evaluate_symmetry_averaged()` unconditionally calls `_resolve_entity_adapter()` (line 453), even when `rust_featurize=True` and the topology is already warm.
- The regular `evaluate()` method has a `need_adapter_resolve` guard that skips this when `rust_featurize=True` and `self._rust_topology` is not None.
- `_resolve_entity_adapter()` calls `json.loads(game.json_snapshot())` internally — a ~0.9ms JSON round-trip that's wasted when the adapter is already resolved.
- `evaluate_symmetry_averaged()` is only called at wide placement roots (~4 per game), so the absolute impact is small (~3.6ms/game).

### Fix

Add the same `need_adapter_resolve` guard to `evaluate_symmetry_averaged()`:
```python
need_adapter_resolve = (
    (not bool(self.config.rust_featurize)) or self._rust_topology is None
)
if need_adapter_resolve:
    resolved = _resolve_entity_adapter(...)
else:
    resolved = self._cached_adapter  # or re-derive from warm topology
```

---

## Finding 36: Training H2D Transfers Synchronous — No pin_memory/non_blocking (LOW)

### Evidence

- `train_bc.py` line 1960+: `torch.as_tensor(obs, dtype=..., device=policy.device)` — synchronous CPU→GPU copy.
- Same pattern for `context_t`, `actions`, `policy_weights`, `value_weights` — all synchronous H2D.
- No `pin_memory=True` or `non_blocking=True` anywhere in the training batch path.
- At batch_size=1024, this is 5+ synchronous H2D transfers per batch, ~3800 batches/epoch.

### Impact

Minor — the training loop is likely compute-bound (the forward+backward pass dominates). But for larger batch sizes (4096+), async transfers could overlap data movement with computation.

### Fix

Pre-pin host arrays and use async transfer:
```python
obs_t = torch.from_numpy(obs).pin_memory().to(policy.device, non_blocking=True)
```

Or use a DataLoader with `pin_memory=True` and `num_workers>0`.

---

## Finding 37: `dataclasses.replace(mcts.config, ...)` Every Decision — Allocation Overhead (LOW)

### Evidence

- `gumbel_self_play.py` line 786: `mcts.config = dataclasses.replace(mcts.config, temperature=temperature)` is called every decision (~100-200 times per game).
- `dataclasses.replace` creates a new config object each time, even when the temperature hasn't changed (early/late game temperature is constant within a phase).
- With 16 workers × 1000 games × 150 decisions = 2.4M config object allocations per worker.

### Impact

Negligible — `dataclasses.replace` is a cheap constructor call. But it's unnecessary when the temperature doesn't change.

### Fix

Only replace when temperature actually changes:
```python
if mcts.config.temperature != temperature:
    mcts.config = dataclasses.replace(mcts.config, temperature=temperature)
```

---

## Finding 38: Gradient Clipping Hardcoded to 1.0 — Not Configurable (LOW)

### Evidence

- `train_bc.py` line 2025: `torch.nn.utils.clip_grad_norm_(list(_params(policy)), 1.0)` — hardcoded to 1.0.
- The FSDP path (line 2579) also uses `_clip_grad_norm(policy, 1.0)`.
- No `--max-grad-norm` CLI flag exists.
- For a 35M param model with batch_size=1024, grad norm 1.0 is reasonable. But for larger models or different batch sizes, a configurable threshold would be useful.

### Fix

Add `--max-grad-norm` CLI flag (default 1.0) and use it in both clip sites.

---

## Finding 39: `_traverse_roll` Discards `_simulate` Return Value (LOW)

### Evidence

- `gumbel_chance_mcts.py` line 1349: `self._simulate(stats.children[outcome_index], depth=depth + 1)` — the return value is discarded.
- Line 1351-1353: `value = sum(stats.probabilities[index] * stats.children[index].value for index in stats.children)` — recomputes the full expectation over ALL children.
- The `_simulate` return value IS the updated `stats.children[outcome_index].value`, so the recomputation includes it. But the other 10 children's values were already computed during `_enumerate_roll_outcomes` and haven't changed.
- This is O(11) per ROLL backup — negligible per call, but ROLL is ~51% of decisions.

### Impact

Negligible — O(11) multiply-adds per ROLL backup. The code is correct (the expectation must be recomputed because the sampled child's value changed). The only optimization would be to cache the non-sampled children's contribution and only add the sampled child's new value, but this saves ~10 multiply-adds per ROLL — not worth the complexity.

### Status

Not a bug, not worth fixing. Documented for completeness.

---

## Finding 40: `evaluate_many` Missing `need_adapter_resolve` Guard — Wastes JSON Round-Trips on ROLL Children (MEDIUM)

### Evidence

- `neural_rust_mcts.py` line 608: `evaluate_many()` unconditionally calls `_resolve_entity_adapter()` for every request, even when `rust_featurize=True` and `self._rust_topology` is already warm.
- The regular `evaluate()` method (line 347) has a `need_adapter_resolve` guard that skips this: `need_adapter_resolve = (not bool(self.config.rust_featurize)) or self._rust_topology is None`.
- `evaluate_many` is called for EVERY ROLL chance node expansion (11 children per ROLL, ~51% of decisions are ROLL).
- Each `_resolve_entity_adapter` call does `json.loads(game.json_snapshot())` + `json.loads(game.player_state_json(color))` for each color — ~1.3ms per call.
- With 11 ROLL children × 1.3ms = ~14ms wasted per ROLL chance node when `rust_featurize=True`.

### Impact

When `--rust-featurize` is enabled (Finding #18's fix), `evaluate_many` would still pay the JSON overhead for all 11 ROLL children despite the topology being warm. This partially negates the `rust_featurize` optimization for the ROLL path specifically.

### Fix

Add the same `need_adapter_resolve` guard to `evaluate_many`:
```python
need_adapter_resolve = (
    (not bool(self.config.rust_featurize)) or self._rust_topology is None
)
if need_adapter_resolve:
    resolved = _resolve_entity_adapter(...)
else:
    resolved = None
# Then pass adapter=resolved[1] if resolved is not None else None
```

---

## Finding 41: FSDP Checkpoint Does NOT Save Optimizer State (MEDIUM)

### Evidence

- `_write_entity_checkpoint()` (line 7152) saves only: `policy_type`, `config`, `action_mask_version`, `mask_hidden_info`, `static_action_features_sha256`, `static_action_features`, `model`.
- No `optimizer` key in the saved dict.
- The FSDP path (`_save_policy`) calls `_write_entity_checkpoint` after gathering the full state_dict.
- The single-GPU/DDP path (`policy.save(tmp, ...)`) also doesn't save optimizer state (Finding #8).
- The code comment at line 1047 confirms: "Checkpoints do not persist optimizer state, so this resume restarts Adam's moment estimates from zero."

### Impact

Same as Finding #8 but for the FSDP path. Resuming FSDP training after preemption loses Adam momentum + variance estimates. The `--lr-warmup-steps` mitigation helps but doesn't fully solve it.

### Fix

Save optimizer state alongside model state. For FSDP, this requires gathering the optimizer state dict across ranks (similar to the model state_dict gather). Standard PyTorch FSDP pattern:
```python
with FSDP.state_dict_type(policy.model, StateDictType.FULL_STATE_DICT, gather_cfg):
    model_state = policy.model.state_dict()
    # Also gather optimizer state
    optim_state = FSDP.optim_state_dict(policy.model, optimizer)
```

---

## Finding 42: `player_state_json` Called Per-Color Per-Leaf in `_resolve_entity_adapter` (LOW-MEDIUM)

### Evidence

- `neural_rust_mcts.py` line 1083: `states_by_color = {str(color): json.loads(game.player_state_json(str(color))) for color in colors}`.
- For 2-player Catan, this is 2 `player_state_json` calls + 2 `json.loads` per leaf eval.
- Each `player_state_json` is a Rust→JSON round-trip (~0.2-0.5ms).
- This is called inside `_resolve_entity_adapter`, which is called once per leaf eval (when `need_adapter_resolve` is True).
- The `rust_featurize=True` path skips this entirely (Finding #40's fix would extend this to `evaluate_many`).

### Impact

~0.4-1ms per leaf eval in the non-rust_featurize path. With ~7K leaf evals/game, this is ~3-7 seconds per game of `player_state_json` overhead.

### Fix

1. Enable `--rust-featurize` (already available, Finding #18) to bypass this entirely.
2. Alternatively, batch the `player_state_json` calls or cache them per-game-state.

---

## Finding 43: Training RNG State Not Persisted — Resume Loses Shuffle Continuity (LOW)

### Evidence

- `train_bc.py` line 1082: `rng = np.random.default_rng(args.seed)` — the training RNG is seeded once at start.
- The RNG state advances through `rng.permutation(n)` (epoch order) and `rng.choice(...)` (weighted sampling).
- On resume (after preemption), the RNG is re-seeded from `args.seed`, producing the SAME epoch order as the first run.
- This means epoch 2 of a resumed run gets epoch 1's order (not epoch 2's), and epoch 3 gets epoch 1's again.
- For 2-epoch training, this means a resumed run after epoch 1 replays epoch 1's order for epoch 2 — the model sees the same batch order twice instead of a fresh shuffle.

### Impact

Minor for 2-epoch training (the model just sees a slightly biased epoch 2). More significant for longer training runs where the same order repeats every resume.

### Fix

Save the RNG state in the checkpoint and restore it on resume:
```python
# Save:
checkpoint["rng_state"] = rng.bit_generator.state
# Restore:
if "rng_state" in checkpoint:
    rng.bit_generator.state = checkpoint["rng_state"]
```

---

## Finding 44: DDP Data Sharding Uses Stride Split — Load Imbalance on Heterogeneous Shards (LOW)

### Evidence

- `train_bc.py` line 3624: `files = files[rank::world_size]` — DDP data sharding uses stride splitting.
- With 4 GPUs and 100 shards: rank 0 gets shards [0,4,8,...], rank 1 gets [1,5,9,...], etc.
- If shards have different sizes (ragged shard counts), some ranks get more data than others.
- The `_epoch_order` function pads to `total_size` to align batches across ranks, but the padding uses `np.resize(order, ...)` which wraps around — duplicated rows in the padded region.

### Impact

Minor — shards are typically uniform (each has `shard_size` rows). The padding only affects the last partial batch. The duplicated rows get sample weight 0 if they're padding, so the loss is unaffected. But the gradient sync still processes them.

### Fix

Use `DistributedSampler` from `torch.utils.data` which handles padding correctly, or drop the last incomplete batch.

---

## Finding 45: `_build_decision_row` Calls `json_snapshot` + `playable_action_indices` + `playable_actions_json` Redundantly (LOW)

### Evidence

- `gumbel_self_play.py` line 592-594: `_build_decision_row` calls `game.json_snapshot()`, `game.playable_action_indices()`, and `game.playable_actions_json()` — 3 Rust round-trips.
- These were already called during `mcts.search()` (inside `_fetch_legal_actions` and `_expand`).
- The search result doesn't carry the snapshot/action data forward, so `_build_decision_row` re-fetches them.

### Impact

~1.5ms per recorded decision (3 Rust calls × ~0.5ms each). With ~100-200 decisions per game, this is ~150-300ms per game — negligible vs the ~minutes of search time.

### Fix

Thread the snapshot/action_by_id from the search through to `_build_decision_row` (the search already has them from `_fetch_leaf_decision_inputs`). This would require adding them to `SearchResult` or passing them as separate arguments.

---

## Finding 46: Topology Cache Uses FIFO Eviction, Not LRU (LOW)

### Evidence

- `entity_token_features.py` line 252: `_TOPOLOGY_CACHE.pop(next(iter(_TOPOLOGY_CACHE)))` — FIFO eviction.
- Same pattern as Finding #15 (eval cache FIFO).
- The topology cache is bounded to 16 entries and keyed by board topology.
- In practice, only 1 board is used per game (and typically per worker lifetime), so the cache never evicts.

### Fix

Use `OrderedDict` with `move_to_end` on hit (same as Finding #15's fix). Negligible impact since the cache rarely fills.

---

## Finding 47: `prior_policy` Stored as fp16 — Precision Loss for KL Computation (LOW)

### Evidence

- `gumbel_self_play.py` line 636: `prior_policy = np.asarray([...], dtype=np.float16)`.
- `target_policy` is stored as fp32 (line 625).
- `prior_policy` is used for KL(improved_policy || prior) computation during training.
- fp16 has ~1e-3 relative precision — small priors (e.g. 0.001) can be flushed to zero.
- At 54-action placement roots, many priors are small (1e-4 to 1e-3 range).

### Impact

KL divergence computation may be slightly biased for small priors. The KL is used for diagnostics (not training loss), so this doesn't affect model quality — just telemetry accuracy.

### Fix

Store `prior_policy` as fp32:
```python
prior_policy = np.asarray([...], dtype=np.float32)
```

Trade-off: +4 bytes per legal action per row. For 54 actions × 3.9M rows = ~844 MB additional disk. Acceptable.

---

## Finding 48: No Early Stopping — Training Runs Full Epochs Even When Converged (LOW, research-level)

### Evidence

- No `--early-stopping` or `--patience` flag in `train_bc.py`.
- Training always runs `--epochs` epochs, even if the validation loss has plateaued.
- For 2-epoch BC training, this is fine. But for longer training runs (gen-5 grow-from-champion with 10+ epochs), early stopping could save GPU time.

### Fix

Add `--early-stopping-patience N` flag that stops training if validation loss doesn't improve for N epochs. Standard pattern:
```python
if validation_loss < best_val_loss:
    best_val_loss = validation_loss
    patience_counter = 0
else:
    patience_counter += 1
    if patience_counter >= args.early_stopping_patience:
        break
```

---

## Methodology Notes (Round 2)

- All code analysis done via SSH to live fleet boxes (c1, c2, B200, a100a).
- GPU telemetry from `nvidia-smi --query-gpu` and `nvidia-smi --query-compute-apps`.
- Process model from `ps -eo pid,cmd` + `/proc/<pid>/status` (67 threads per worker).
- Rust GIL release confirmed via `strings` on the pyo3 .so (421 `allow_threads` matches).
- Model architecture from `entity_token_policy.py` (35M params, 640 hidden, 6 layers, 8 heads).
- Training config from `train_bc.py` argparse defaults.
- No changes were made to any code or configuration. All findings are read-only analysis.
