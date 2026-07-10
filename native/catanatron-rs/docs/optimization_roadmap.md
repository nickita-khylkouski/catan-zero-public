# Catanatron Rust Optimization and AI Training Roadmap

This file tracks the contribution ideas that matter for using the simulator as a
fast, rule-preserving self-play engine for AI training. Status is intentionally
conservative: an item is not "done" unless the current Rust implementation has
working code and verification coverage.

## Performance Engine

| Idea | Current Status | Notes | Required Proof |
| --- | --- | --- | --- |
| Improve package running time performance | In progress | Rust engine is already much faster than Python and currently benchmarks at about 43.1k 4-random-player games/s on GH200 with 64 threads. | GH200 CLI benchmarks, Criterion hot-path benches, correctness tests. |
| Refactor `State` toward primitive arrays | Partial | Resources/dev cards are compact arrays, but `State`, `Board`, buildings, roads, components, ports, and action records still use `HashMap`/`HashSet` in hot paths. | Golden gameplay tests plus before/after clone and full-game benchmarks. |
| Move `Resource` to ints | Partial | Rust `Resource` is a compact enum with `idx()`, so resource decks avoid enum hashing. Public JSON/Python APIs still expose resource names. | Feature/action JSON parity tests and full-game benchmarks. |
| Move `action_records` from `State` to `Game` | Partial | Native stats-only runs can disable action recording, but `State` still owns `action_records`. A full move would reduce clone cost for search/rollouts. | JSON/Python `state_index` compatibility tests, AlphaBeta/rollout clone benchmarks. |
| Remove `current_prompt` | Not started | `current_prompt` is still the central action-generation state machine. Removing it may reduce state size but has high rule-flow risk. | Exhaustive turn-flow tests for initial build, discard, robber, dev cards, trades, and replay. |
| Avoid per-tick full legal-action materialization | Investigated | A no-materialization random selector was proven equivalent in tests but was slower on GH200, so it is not enabled in CLI stats. | Keep equivalence test; only enable if benchmark improves. |
| Continue board/topology optimization | In progress | Longest-road recompute is scoped to the affected color on road placement; settlement still does full recompute because it can block opponents. | Scoped-vs-full recompute regression and longest-road edge-case tests. |

## Search Players and Heuristics

| Idea | Current Status | Notes | Required Proof |
| --- | --- | --- | --- |
| Improve AlphaBetaPlayer | Partial | Rust has AlphaBeta and same-turn AlphaBeta, but depth/pruning/search performance has not been deeply tuned. | Action-equivalence tests, search-node benchmarks, bot-vs-bot outcome sweeps. |
| Explore and improve pruning | Partial | Lightweight pruning exists for initial settlements, maritime trades, and robber moves. AlphaBeta pruning remains basic. | Pruned action set must retain best known actions under replay/golden scenarios. |
| Tune value weights with Bayesian methods or SPSA | Not started | Static weights are hardcoded. There is no tuning harness yet. | Reproducible tournament harness, seed suites, tracked objective metrics. |
| Research stronger policies | Not started | No research harness yet beyond current bots and rollout player. | Experiment scripts, reproducible configs, train/eval separation. |

## AI Training

| Idea | Current Status | Notes | Required Proof |
| --- | --- | --- | --- |
| Deep Q-Learning | Not started | Rust exposes feature vectors, action indices, board tensors, and batch simulation, but no DQN trainer is included. | Training script, checkpoint format, deterministic eval suite. |
| Simple AlphaGo-style approach | Not started | Requires policy/value model, MCTS/search integration, and self-play data pipeline. | MCTS correctness tests, self-play data schema, eval ladder. |
| Try Tensorforce with simple action space | Not started | Rust action-space helpers exist, but no Tensorforce integration is checked in. | Example environment script and action-mask parity tests. |
| Flat CSV training data | Partial | Native CLI writes CSV matrices, rewards, optional board tensors, per-row metadata, sparse legal ids, dense legal-action masks, and JSON sidecars for feature/action schemas. | Larger end-to-end training smoke tests and dataset versioning docs. |
| AlphaBeta-generated games for CSV | Partial | CLI supports AlphaBeta players, but slow throughput and training-data examples need more work. | CSV run with AB players, action/reward schema validation, throughput benchmark. |

## Features and Debugging

| Idea | Current Status | Notes | Required Proof |
| --- | --- | --- | --- |
| Continue implementing UI actions | Partial | Rust has JSON action parsing/execution and Flask route support mentioned in README. Trade-flow actions are now present in the static action space, but UI parity still needs a dedicated checklist. | Golden UI action JSON tests against Python routes. |
| Terminal UI for debugging | Not started | No TUI exists. Useful for inspecting state transitions, legal actions, robber/discard flow, and replay logs. | Manual debug workflow plus snapshot/replay tests. |
| Better debugging/replay support | Partial | Action records, JSON snapshots, bank/deck/trade flags, legal action masks, and stats-only no-recording runs exist. | Replay roundtrip tests and optional compact trace format. |

## Neural-Net Training Surface

| Gap | Current Status | Notes | Next Work |
| --- | --- | --- | --- |
| Batched environment API | Partial | Python binding now exposes dependency-free `BatchEnv` with batched observations, legal masks, current-player feature vectors, rewards, dones, winners, current colors, reset, action-id stepping, bytes-return variants, reusable bytearray fills, and generic writable-buffer fills for trainer-owned host memory. | Add a pure Rust batch env struct if needed by non-Python trainers. |
| Dense legal masks | Implemented | `ActionSpace` caches ids with typed keys; masks are derived only from `game.playable_actions`. CLI writes `legal_action_masks.csv`; sparse ids remain for compatibility. | Add optional mask dtype selection and decide whether domestic offer generation should become part of the policy action space. |
| Batched tensor extraction | Partial | Current-player board tensors can be concatenated into one batch; tensor shape is player-count aware; direct-layout `f64`/`f32` fill helpers, parallel f32 board batches, schema-width feature batches, direct bytearray fills, and generic writable-buffer fills exist. | Replace `BTreeMap` feature assembly with a compiled schema/SoA writer. |
| GPU/model integration | Boundary documented | Torch example keeps model inference/training outside the rules engine and feeds action ids back to Rust; generic writable-buffer APIs let Rust fill memoryviews over NumPy arrays or pinned Torch CPU tensors before non-blocking GPU transfer. | Add JAX/ONNX examples if needed and benchmark end-to-end trainer transfer cost. |

## Completed in Current Pass

- Added static action-space entries and normalized indices for domestic trade flow: offer, accept, reject, confirm, and cancel.
- Added domestic trade validation for offerer resources, confirmed trade payload matching, and accepted target checks.
- Fixed road expansion through enemy settlements and covered it with a regression test.
- Expanded JSON snapshots with replay/debug fields for seed, turn counters, prompt flags, bank counts, dev deck count, current trade, and acceptees.
- Expanded native CSV output with `metadata_rows.csv`, `legal_action_indices.csv`, `metadata.json`, `feature_ordering.json`, and `action_space.json`.
- Added tests for trade flow validation, trade action-space indexing, snapshot debug fields, CSV sidecars, and enemy settlement road blocking.
- Added `BatchEnv`, dense legal masks, deterministic robber action ordering, batched current-player board tensors, Torch-loop docs/example, and CSV mask verification.
- Added direct board-tensor fill helpers, schema-preserving feature batch vectors, and bytes-return `BatchEnv` methods for lower-overhead Python-to-GPU handoff.
- Added `BatchEnv.feature_ordering()`, `feature_vectors()`, and `feature_vectors_bytes()` so policy/value trainers can consume fixed-width scalar features without per-game Python calls.
- Added stable feature/action schema hashes and a direct `f32` board-tensor fill path for Python `BatchEnv` observations.
- Replaced per-action JSON string lookup in legal-mask indexing with typed action-space keys and in-place mask row filling for `BatchEnv`.
- Added reusable Python `bytearray` fill APIs for observations, legal masks, and feature vectors to reduce per-step allocation churn in trainer loops.
- Added row-parallel `f32` board-tensor batch fill and GH200-oriented prebuilt tensor benchmarks.
- Added row-parallel scalar feature batch filling plus direct `feature_vectors_bytes_into()` bytearray writes, avoiding an intermediate feature matrix allocation on that Python hot path.
- Added direct `observe_bytes_into()` observation and legal-mask bytearray filling, avoiding intermediate observation/mask matrices on the Python training hot path while preserving done-row zero masks and all-or-nothing buffer writes.
- Added generic writable-buffer `BatchEnv` APIs for observations, masks, reset, step, and feature vectors so trainers can reuse memoryviews over NumPy arrays or pinned Torch CPU tensors.

## Test Coverage Priorities

These guardrails are required before deeper structural rewrites:

- Longest-road edge cases: cycles, forks, ties, award transfer and loss. Opponent settlement road-expansion blocking now has direct coverage.
- Discard sequence: seating order, no-discard threshold, custom discard limit, restore roller before robber move.
- Development cards: one card per turn, bought-this-turn lockout, depleted-bank Year of Plenty fallbacks, Road Building partial/no-edge cases.
- Action ordering/index parity: generated action sets must remain sorted and stable for seeded random games and training masks.
- JSON/API parity: Python-compatible snapshots, `action_records`, `current_playable_actions`, `state_index`, board tensors, and action-space indices.
- Training data schemas: CSV/JSONL output consistency, rewards, action indices, board tensor shapes, and feature ordering.

## Current GH200 Baseline

Latest clean native Rust stats run:

```text
RUSTFLAGS="-C target-cpu=native"
RAYON_NUM_THREADS=64
target/release/catanatron --num 100000 --players R,R,R,R --seed 1 --quiet

games_per_second=43121.82
turns_per_second=14504973.88
```

The simulator remains CPU-bound. CUDA is available on the GH200 host, but the
current Catan rules engine is branch-heavy and stateful; the effective speed
path is parallel independent games across Grace CPU cores unless the simulator is
redesigned around batched fixed-shape kernels.
