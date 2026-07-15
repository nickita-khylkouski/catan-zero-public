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

Version `0.1.7` adds `initial_road_d1_scope`. Python attests the authoritative
public root prompt once, the binding carries that immutable prompt into native
search, and Rust applies D1 only at the `BUILD_INITIAL_ROAD` decision root.
Interior nodes and every other phase retain the historical min-max operator.
Version `0.1.8` adds `public_award_feature_parity` and
`policy_temperature_semantics`. Both the Python snapshot adapter and the direct
Rust entity featurizer preserve authoritative, public longest-road ownership in
player-token slot 12, including for masked opponents. Native gameplay selection
also applies the configured temperature as `softmax(log(policy) / T)`; `T=1`
is an exact no-op and `T=0` retains deterministic argmax selection. Callers
requesting these semantics fail closed on older wheel digests that either
silently emitted zero for the public award or treated every positive gameplay
temperature as `T=1`.

Version `0.1.9` links `gumbel-mcts` `0.2.3` and adds
`coherent_public_belief_search` plus `forced_root_trajectory_only`. Coherent
belief search spends one unsplit budget in a sanitized two-player tree, stops
at the root actor's turn boundary, and materializes development-card draws
from public posterior support instead of the arbitrary concrete hidden deck.
Trajectory-only forced roots return the mathematically exact sole action with
no neural evaluation or invented Q/root-value evidence. Both semantics are
advertised through `gumbel_search_capabilities()` and production admission
must reject older wheels when either mode is requested.
