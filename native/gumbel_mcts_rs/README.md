# Native Gumbel MCTS hot loop

This crate is the Rust tree-traversal implementation embedded in the canonical
`catanatron_rs` Python wheel. It is not a standalone Python extension.

Build the wheel from the repository root:

```bash
maturin build --release \
  --manifest-path native/catanatron-rs/python/Cargo.toml
```

Evaluation keeps the Python reference implementation by default. The native
arm is selected explicitly with `--native-mcts-hot-loop` in:

- `tools/gumbel_search_cross_net_h2h.py`;
- `tools/gumbel_search_vs_bot_h2h.py`;
- `tools/catanatron_neutral_harness_match.py --mode search`.

The selected implementation is included in typed-config hashes and output
provenance. Native selection fails closed when the wheel is absent or a search
semantic is unsupported. Information-set particle construction/aggregation,
P4/min32 budgeting, public-observation checks, and D6 orchestration remain in
the reference Python layer; each single-world tree executes natively. Exact
chance enumeration uses `Evaluator::evaluate_many`, whose scalar default is a
loop over `evaluate`.

Deferred native leaf batching was removed because it was not
reference-equivalent. `batch_size > 0` is rejected at the binding boundary.

Verification:

```bash
cargo fmt --all --manifest-path native/gumbel_mcts_rs/Cargo.toml -- --check
cargo test --release --manifest-path native/gumbel_mcts_rs/Cargo.toml
cargo clippy --release --all-targets \
  --manifest-path native/gumbel_mcts_rs/Cargo.toml -- -D warnings
pytest -q tests/test_native_gumbel_hot_loop.py \
  tests/test_native_information_set_search.py
```

The production profiling baseline and reproduction recipe live in
`docs/profiling/EVAL_FLAMEGRAPH_2026-07-11.md`. Measured native results must be
reported from matched real-checkpoint cohorts; the old informal 77% Python and
3.5x estimates were removed because the exact flamegraph did not support them.
