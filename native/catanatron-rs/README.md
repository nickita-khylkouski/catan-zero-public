# catanatron-rs

Rust port of the Catanatron core game engine.

This crate currently covers the performance-critical engine layer:

- base and mini map generation
- board placement rules and longest-road tracking
- compact resource and development-card decks
- game state, action generation, and action application
- simple, random, weighted-random, victory-point, value-function, and alpha-beta
  players
- Rust greedy playout / MCTS-style player kind for fast rollout-based decisions
- stochastic outcome expansion for rolls, development-card buys, and robber steals
- lightweight action pruning for initial settlements, maritime trades, and robber moves
- dice, robber, discard, maritime trade, development-card, and turn flow logic
- numeric feature extraction for player/resource/tile/port/graph/game features
- sample vector creation and stable feature ordering helpers
- Gym-style board tensor generation through the Rust Python binding
- Gym action-space generation through the Rust Python binding
- batched Python `BatchEnv` for external neural-net policy loops
- fixed dense legal-action masks for NN training and CSV export
- batched current-player feature vectors with byte-return paths for GPU handoff
- `catanatron/RustCatanatron-v0` Gymnasium environment backed by `RustGame`
- Flask routes under `/api/rust/...` backed by `RustGame` snapshots
- optional production features with robber-aware tile blocking
- Python-compatible action JSON parsing/encoding helpers
- game snapshot JSON encoding for tiles, nodes, edges, actions, player state,
  colors, prompts, playable actions, longest roads, and winner state
- optional PyO3 Python extension module exposed as `catanatron_rs`
- optional Python package facade exposed as `catanatron.RustGame`
- Rust-backed Python batch simulator and `catanatron-rust-play` console command
- legacy `catanatron-play --engine rust` dispatch for supported fast batches,
  including mixed Rust player kinds
- native Rust `catanatron` simulator binary with stats and JSONL output
- release-mode speed harness
- property and stress coverage for action JSON roundtrips, action ordering,
  mixed bot rosters, maps, friendly robber, and resource/dev-card/piece
  conservation invariants

It intentionally lives beside the original Python package while the Rust API is
validated. The Python web, Gymnasium, and UI integrations can be moved onto this
crate once a binding layer is chosen.

## Verify

```bash
cargo fmt --check
cargo test
cargo test --features python
cargo clippy --all-targets --all-features -- -D warnings
cargo run --release --bin catanatron -- --num 1000 --players R,R,R,R --quiet
cargo run --release --bin catanatron -- --num 100 --players R,W --number-placement random --quiet
cargo run --release --bin catanatron -- --num 100 --players R,R --map TOURNAMENT --number-placement random --quiet
cargo run --release --bin catanatron -- --num 1 --tournament 'AlphaBeta(n=2),ValueFunction,GreedyPlayouts(n=5),WeightedRandom,VictoryPoint,Random,JSETTLERS,QSETTLERS' --seed 1 --vps-to-win 5 --quiet
cargo run --release --example speed -- 1000
cargo bench --bench engine -- --sample-size 10
```

Python extension smoke test:

```bash
python -m pip install maturin
maturin develop --features python-extension
python examples/python_smoke.py
catanatron-rust-play --num 100 --players R,R,R,R --quiet
catanatron-rust-play --num 100 --players R,W --config-number-placement random --quiet
catanatron-rust-play --num 100 --players R,R --config-map TOURNAMENT --config-number-placement random --quiet
catanatron-rust-play --num 100 --players R,R,R,R --output data/ --output-format jsonl --quiet
```

The Python facade exposes `RustGame.board_tensor(color, channels_first=False)`,
returning a NumPy array with the same board-plane ordering as the legacy Gym
helper for Rust-backed games.

When the extension is installed, the Python Gym action-space helper also uses
Rust to generate static action arrays before converting them into the existing
Python `ActionType`/value tuple contract.

The package also registers `catanatron/RustCatanatron-v0`, a Gymnasium
environment that keeps the game loop, action validation, feature samples, and
mixed board tensors on the Rust-backed path.

The Flask app registers separate Rust-backed endpoints at `/api/rust/games`,
`/api/rust/games/<game_id>/states/<state_index>`,
`/api/rust/games/<game_id>/actions`, and `/api/rust/stress-test`.

The Rust CLI writes JSONL as one compact final game snapshot per line. Passing a
directory writes `rust-games.jsonl` inside it; passing a path ending in `.jsonl`
writes that file directly.

The CLI also has a tournament runner for comparing Catan AI labels from this
engine and external projects:

```bash
cargo run --release --bin catanatron -- \
  --num 1 \
  --tournament 'AlphaBeta(n=2),ValueFunction,GreedyPlayouts(n=5),WeightedRandom,VictoryPoint,Random,JSETTLERS,QSETTLERS' \
  --seed 1 \
  --vps-to-win 5 \
  --quiet
```

For rosters larger than four players it evaluates every four-player pod with
seat rotations and prints ranked win rates. See
[docs/ai_tournament.md](docs/ai_tournament.md) for the online project aliases
and native adapter mapping, including the heavyweight `MCTS(n=100)` profile.

The native Rust binary is available without Python:

```bash
cargo run --release --bin catanatron -- --num 1000 --players R,R,R,R --quiet
cargo run --release --bin catanatron -- --num 100 --players R,W --number-placement random --quiet
cargo run --release --bin catanatron -- --num 100 --players R,R --map TOURNAMENT --number-placement random --quiet
cargo run --release --bin catanatron -- --num 100 --players R,R --jsonl data/rust-games.jsonl --quiet
```

The legacy `catanatron-play` command also accepts `--engine rust` for supported
bot batches, including mixed Rust player kinds. It keeps the old Python engine as the default and
raises clear errors for legacy output modes that are not yet available through
the Rust batch path.

For neural-net training, see [docs/training_api.md](docs/training_api.md) and
[examples/torch_policy_loop.py](examples/torch_policy_loop.py). The Rust engine
emits batched observations, current-player feature vectors, and dense legal
masks into CPU host buffers, including trainer-owned memoryviews over pinned
Torch CPU tensors; Torch/JAX/ONNX inference should stay outside the rules engine
and feed selected action ids back into `BatchEnv.step`.

Latest local speed sample on this machine:

```text
games=1000
wins=999
turns=334798
elapsed_ms=4296.003
games_per_second=232.77
turns_per_second=77932.43
```

GH200 remote speed sample with `RUSTFLAGS="-C target-cpu=native"` and
`RAYON_NUM_THREADS=64`:

```text
games=100000
wins=99918
turns=33637203
elapsed_ms=2319.012
games_per_second=43121.82
turns_per_second=14504973.88
```

Serial GH200 sample for the same 4-random-player workload:

```text
games=10000
wins=9992
turns=3368565
elapsed_ms=15523.291
games_per_second=644.19
turns_per_second=217000.70
```

Tournament-map GH200 sample:

```text
games=100000
wins=99986
turns=30384819
elapsed_ms=2205.750
games_per_second=45336.05
turns_per_second=13775275.90
```

Mixed bot remote samples:

```text
players=R,W,VP,F games=1000 elapsed_ms=232.598 games_per_second=4299.27
players=AB,R,W,VP games=500 elapsed_ms=2322.241 games_per_second=215.31
players=R,VP,W,S games=10000 elapsed_ms=2476.257 games_per_second=4038.35
players=AB,SAB,P,W games=200 vps_to_win=5 elapsed_ms=3059.146 games_per_second=65.38
```

Upstream Python Catanatron comparison on the same GH200 host:

```text
python serial, 500 games: 37.81 games/s, 12474.23 turns/s
python multiprocessing, 64 workers, 10000 games: 2229.17 games/s, 730338.92 turns/s
python weighted-random serial, 500 games: 45.68 games/s, 10092.77 turns/s
rust weighted-random serial, 500 games: 1030.77 games/s, 229786.73 turns/s
rust weighted-random 64-thread, 10000 games: 60307.31 games/s, 13338686.05 turns/s
```

Criterion board tensor samples on GH200 after direct layout fill support:

```text
board_tensor_flat_4p_channels_last time: [6.9720 us 7.8040 us 8.1816 us]
board_tensor_fill_4p_channels_last time: [8.1554 us 8.5971 us 8.8776 us]
board_tensor_batch_4p_channels_last_32 time: [266.12 us 272.68 us 277.74 us]
```

The game engine remains CPU-bound: complete Catan game simulation has
stateful, branch-heavy rules and dynamic action generation, so the current fast
path is many independent games across the GH200 Grace CPU cores. CUDA is present
on that host, but the Rust simulator does not offload game logic to the GPU.

The upstream Python suite was also verified separately under Python 3.12:

```text
227 passed, 7 warnings in 105.01s
```

## Roadmap

See [docs/optimization_roadmap.md](docs/optimization_roadmap.md) for the
performance, AI-training, search-player, UI/debugging, and test-coverage roadmap.
