# Canonical native search stack

This directory contains the exact Rust sources required to build the Python
engine wheel used by Catan Zero.  They are vendored here so the canonical
GitHub repository—not an untracked GPU-host checkout—is the source of truth.

The Catanatron-derived engine is distributed under GNU GPL-3.0. The verbatim
license text is already included at `vendor/catanatron/LICENSE`; retain that
file in every source or binary distribution. Upstream:
https://github.com/bcollazo/catanatron.

- `catanatron-rs/`: game engine and `catanatron_rs` Python bindings.

Build the CPython 3.11 wheel from the repository root:

```bash
maturin build --release \
  --manifest-path native/catanatron-rs/python/Cargo.toml \
  --interpreter python3.11 \
  --out dist
```

Version `0.1.4` adds `Game.determinize_for_player(observer, seed)`, an atomic
public-conservation world sampler for PIMC. It jointly resamples opponent
resources, face-down development cards, the remaining deck, and hidden victory
points while preserving public state, the observer hand, and root legal actions.
It does not yet condition samples on deductions from the complete public action
history, so artifacts must not label it a full Bayesian information-set posterior.

Version `0.1.5` adds the fail-closed `gumbel_search` Python binding used by the
native MCTS hot loop. The distinct version is intentional: wheels without that
required API must never share the new release artifact name or package identity.

Version `0.1.6` adds the native-search capability contract required by
belief-level completed-Q aggregation: `sigma_reference_visits` calibration,
root-perspective completed-Q evidence, and the fail-closed
`gumbel_search_capabilities()` advertisement. Version `0.1.5` wheels must not be
used for this operator even if they expose the older `gumbel_search` entry point.
