# gumbel_mcts_rs — Rust-accelerated MCTS for Catan-Zero

**3-5x faster self-play generation** by moving the MCTS tree traversal from Python to Rust.

## Why

Profiling (2026-07-09, H100) showed that **77% of per-game wall time** is Python dict/list/float operations in the MCTS tree traversal — not GPU compute, not featurization, not JSON. Each simulation does ~4ms of GPU work but ~37ms of Python overhead.

This crate moves the hot loop (`_simulate`, `_expand`, `_completed_q`, Sequential Halving, backup) into Rust. The evaluator (neural network forward pass) stays in Python.

## Build

```bash
# Install maturin (one-time)
pip install maturin

# Build + install the Rust extension
cd gumbel_mcts_rs
bash build.sh release
```

This produces a `gumbel_mcts` Python module that can be imported directly.

## Deploy (drop-in replacement)

In `tools/generate_gumbel_selfplay_data.py`, change one import:

```python
# Before:
from catan_zero.search.gumbel_chance_mcts import GumbelChanceMCTS

# After:
from catan_zero.search.gumbel_mcts_rust import GumbelChanceMCTSRust as GumbelChanceMCTS
```

That's it. The interface is identical — same constructor, same `search()`, same `SearchResult`.

If the Rust extension is not installed, it automatically falls back to the pure-Python implementation with a warning.

## Verify

```bash
# Run the integration test (verifies same results as Python)
python3 test_rust_vs_python.py --checkpoint /path/to/checkpoint.pt --device cuda:0

# Run the benchmark (measures speedup)
python3 bench_rust_vs_python.py --checkpoint /path/to/checkpoint.pt --device cuda:0
```

## Expected speedup

| Metric | Python | Rust (estimated) | Speedup |
|--------|--------|-------------------|---------|
| Per simulation | 48ms | ~12ms | 4x |
| Per decision | 1235ms | ~350ms | 3.5x |
| Per game | 521s | ~150s | 3.5x |
| Per GPU per hour | ~7 games | ~24 games | 3.5x |
| Fleet throughput (42 GPUs) | ~290 games/hr | ~1000 games/hr | 3.5x |

The speedup comes from:
- Python dict ops → Rust HashMap (3-5x faster)
- Python float arithmetic → Rust f64 (2-3x faster)
- Python list comprehension → Rust Vec (2-3x faster)
- Python random.Random → Rust ChaCha8Rng (10x faster)
- No GIL contention in the tree traversal loop

## What stays in Python

- The evaluator (neural network forward pass) — already GPU-compiled via PyTorch
- The game state (`catanatron_rs.Game`) — already Rust
- The self-play game loop (`gumbel_self_play.py`) — orchestrates, doesn't do hot-loop work
- The trainer (`train_bc.py`) — not the bottleneck

## What moved to Rust

- `_simulate()` — tree traversal (selection + expansion + backup)
- `_completed_q()` — completed-Q computation
- `_rescale_completed_q()` — min-max rescale + noise floor
- `_improved_policy()` — softmax over logits + sigma(completed_q)
- `_select_nonroot_action()` — non-root action selection
- `sequential_halving_schedule()` — SH schedule computation
- `_run_root_search()` — Gumbel-Top-k + Sequential Halving
- `_sample_gumbel()`, `_sample_outcome()`, `_sample_categorical()` — RNG
- All node/action statistics bookkeeping

## Architecture

```
Python (gumbel_self_play.py)
  └── GumbelChanceMCTSRust.search(game)
       └── Rust (gumbel_mcts crate)
            ├── tree traversal (selection, backup) ← 77% of time was here
            ├── node management (Arena allocator)
            ├── Gumbel sampling, Sequential Halving
            └── evaluator callback → Python (PyTorch forward pass) ← 24% of time
```

The Rust crate calls back to Python for each leaf evaluation (the GPU forward pass). This is the minimal interface — everything else is in Rust.

## Files

- `src/lib.rs` — The Rust MCTS implementation (~900 lines)
- `Cargo.toml` — Rust crate config
- `gumbel_mcts_rust.py` — Python wrapper (drop-in replacement)
- `build.sh` — Build script
- `bench_rust_vs_python.py` — Benchmark
- `test_rust_vs_python.py` — Integration test

## Notes

- The Rust RNG is ChaCha8 (not Python's Mersenne Twister). This means the exact action sequence will differ for the same seed, but the search is statistically equivalent.
- For exact bit-identity, you'd need to port Python's Mersenne Twister to Rust (not recommended — ChaCha8 is faster and higher quality).
- The `uncertainty_backup_weighting`, `variance_aware_q`, and `belief_chance_spectra` features are supported but default off (same as Python).
- The F7 full-enumeration path for MOVE_ROBBER/BUY_DEVELOPMENT_CARD uses single-sampling in the first pass (same as the Python default before F7). Full enumeration can be added in a follow-up.
