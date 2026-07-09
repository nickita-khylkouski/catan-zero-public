# Patches — System Design Findings Fixes

Each file in this directory fixes one or more findings from `SYSTEM_DESIGN_FINDINGS.md`.
All patches are designed to be **drop-in replacements** or **unified diffs** that the
main AI can review and apply with minimal effort.

## How to Apply

### Unified diff patches (`.patch` files)

Apply on the target fleet box inside the repo root (`~/catan-zero-runsix/` or `~/c1_fsdp/repo/`):

```bash
cd ~/catan-zero-runsix
patch -p0 < /path/to/01_entity_token_policy_bf16_compile.patch
# or: git apply < file.patch
```

### Shell script replacements (`.sh` files)

Copy directly over the existing file or run as a new script:

```bash
cp /path/to/03_wave1_harvest_parallel.sh ~/wave1_harvest.sh
chmod +x ~/wave1_harvest.sh
```

---

## Patch Manifest

| # | File | Type | Findings Fixed | Impact |
|---|------|------|----------------|--------|
| 01 | `01_entity_token_policy_bf16_compile.patch` | diff | #2 (bf16), #3 (torch.compile), #10 (pin_memory) | **1.5-2x** inference throughput + kernel fusion |
| 02 | `02_neural_rust_mcts_lru_cache.patch` | diff | #15 (FIFO→LRU) | 5-15% cache hit rate improvement |
| 03 | `03_wave1_harvest_parallel.sh` | full replacement | #19 (serial rsync) | ~14x faster harvest (parallel + SSH ControlMaster) |
| 04 | `04_train_bc_optimizer_state.patch` | diff | #8 (Adam state not persisted) | Resume quality — no more fresh-Adam transient |
| 05 | `05_a100a_relaunch.sh` | new script | #12 (wrong c-scale), #13 (GPU6 idle), #14 (workers=4) | Fixes garbage data + 8x more a100a throughput |
| 06 | `06_teacher_shard_size_fix.sh` | new script | #4 (shard_size too large for n128) | 4x faster first teacher shards |

---

## Detailed Descriptions

### 01 — bf16 autocast + torch.compile + pin_memory (`entity_token_policy.py`)

**Finding #2:** Inference runs in fp32 while training uses bf16. Wrapping `forward_legal_np`'s
model call in `torch.autocast(dtype=torch.bfloat16)` gives ~1.5-2x on H100 tensor cores.

**Finding #3:** No `torch.compile` anywhere. Adding `torch.compile(model, mode="reduce-overhead")`
in `EntityGraphPolicy.__init__` fuses the 6-layer transformer's Python-heavy forward into
fewer CUDA kernels (~1.2-1.5x). Guarded by try/except for CPU/old-torch fallback.

**Finding #10:** H2D transfers use synchronous `torch.as_tensor(value, device=...)`. Added
`non_blocking=True` for async transfer.

**Combined impact:** ~2-3x inference throughput. Stacks multiplicatively with MPS.

**Apply to:** `~/catan-zero-runsix/src/catan_zero/rl/entity_token_policy.py`

### 02 — LRU cache eviction (`neural_rust_mcts.py`)

**Finding #15:** The evaluator cache uses `dict.pop(next(iter(dict)))` which is FIFO eviction.
In MCTS, early-game positions are revisited frequently but get evicted by newer positions
that may never be accessed again. Changed to `OrderedDict` with `move_to_end()` on hit and
`popitem(last=False)` for eviction = proper LRU.

Fixed in all 3 cache locations:
1. `EntityGraphRustEvaluator.evaluate()` — single-leaf path
2. `EntityGraphRustEvaluator.evaluate_many()` — batch path
3. `BatchedEntityGraphRustEvaluator` — threaded batch path (with lock)

**Apply to:** `~/catan-zero-runsix/src/catan_zero/search/neural_rust_mcts.py`

### 03 — Parallel harvest (`wave1_harvest.sh`)

**Finding #19:** rsync runs serially across 3 boxes × 12 dirs = 36 sequential calls.
Rewritten to:
1. Launch all boxes in parallel (`pull_dirs_parallel`)
2. Launch all dirs within a box in parallel (background rsync + wait)
3. SSH ControlMaster for connection reuse (no re-handshake per rsync)

**Impact:** ~14s → ~1s for a full harvest sweep.

**Apply to:** `~/wave1_harvest.sh` (on the harvest/orchestration box)

### 04 — Optimizer state persistence (`train_bc.py`)

**Finding #8:** Checkpoints don't persist optimizer state. Every resume restarts Adam's
moment estimates from zero, causing a transient period of suboptimal updates (the code
even warns about this at line 990). The `--lr-warmup-steps` flag is a mitigation but not
a fix.

This patch saves `optimizer.state_dict()` to a sidecar file (`<checkpoint>.optimizer.pt`)
after each epoch save and at final checkpoint. On resume with `--init-checkpoint`, it loads
the sidecar if it exists.

**Apply to:** `~/c1_fsdp/repo/tools/train_bc.py`

### 05 — a100a pilot relaunch (`a100a_relaunch.sh`)

**Finding #12:** The a100a pilot uses the old `catan-zero` stack with `c-scale=0.1` (the
known-broken calibration), missing `--correct-rust-chance-spectra`, `--lazy-interior-chance`,
`--score-actions`, and other critical flags. The generated data is incompatible with the
production fleet corpus.

**Finding #13:** GPU6 is completely idle (0% util, 0 MiB).

**Finding #14:** Pilot uses `workers=4` instead of 16 (4x underutilization).

This script relaunches on all 8 GPUs with 16 workers each, using the production
`catan-zero-runsix` stack and the full flag set matching the H100 fleet.

**Run on:** a100a (`64.181.197.190`) — requires cloning `catan-zero-runsix` there first.

### 06 — Teacher shard size fix (`teacher_shard_size_fix.sh`)

**Finding #4:** `shard_size=2048` is too large for n128 teacher generation. No worker
reaches 2048 rows for 60+ minutes, delaying first shards and the gen-5 v1 training start.

This wrapper auto-selects shard_size based on n-full:
- n64 (volume): 2048 (unchanged — games are fast)
- n128 (teacher): 512 (4x faster first shard)
- n256 (probe): 256 (8x faster)

**Usage:** `bash teacher_shard_size_fix.sh python tools/generate_gumbel_selfplay_data.py --n-full 128 ...`

---

## REJECTED — threaded generation (Finding #1, patches #11/#12) — CAT-120

**DO NOT apply `11_threaded_selfplay_gen.py` or `apply_12_threaded_generation.py`.**
Threaded batched generation was built and benched TWICE independently
(threaded-gen-batched@3eaec27 `--executor`; threaded-gen@f076eda `--worker-mode`)
and both are a **~4x throughput REGRESSION**, not a speedup. Root cause: Python
featurization (~96% of per-leaf cost) holds the GIL, so N worker threads serialize
on one core, whereas the 16-process + MPS baseline uses N cores; the GPU sits ~97%
idle so bf16/batching cannot help. `allow_threads` + `--rust-featurize` is
necessary-but-not-sufficient (also needs `evaluate_many` routed through the batch
queue) and still caps below the eval-server. **The real throughput lever is the
eval-server (CAT-67)** — a separate GPU process batching many *separate* worker
processes (batching AND full CPU parallelism, no shared GIL). Keep 16-proc + MPS.

## Not Fixed (Requires Architecture Changes)

These findings are NOT addressed by patches because they require deeper changes:

| Finding | Why Not Patched |
|---------|----------------|
| #1 (batch-1 inference) | REJECTED — threading is a ~4x GIL-bound regression (CAT-120). Real lever = eval-server (CAT-67), which batches across separate processes. |
| #9 (EvalServer stub) | This IS the right lever (CAT-67), now implemented — NOT the threaded approach. |
| #18 (JSON serialization) | Requires enabling `--rust-featurize` flag (testing needed) |
| #25 (json.dumps in MCTS) | Requires Rust API change to accept action IDs instead of JSON |
| #29 (three forks) | Requires consolidating codebases — organizational decision |
| #30 (seed ledger) | Requires syncing master ledger across hosts |
| #31 (optional heads) | Research experiment — enable and train, not a code fix |
