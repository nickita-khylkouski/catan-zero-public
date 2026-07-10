# AI tournament adapters

The native Rust CLI can run a ranked tournament across more than four Catan AI
labels:

```bash
cargo run --release --bin catanatron -- \
  --num 1 \
  --tournament 'AlphaBeta(n=2),ValueFunction,GreedyPlayouts(n=5),WeightedRandom,VictoryPoint,Random,JSETTLERS,QSETTLERS' \
  --seed 1 \
  --vps-to-win 5 \
  --quiet
```

For rosters larger than four players, the runner evaluates every four-player
pod, rotates the seat colors for each pod, and reports wins, win rate, draws,
draw rate, and average turns per participating AI label.

`MCTS(n=100)` and `GreedyPlayouts(n=25)` are available, but they are much slower
than the light examples above.

## Online projects checked

These labels are compatibility adapter profiles, not direct source-code ports.
The external projects use different engines, languages, game-state models, and
licenses, so the tournament keeps all games inside this Rust simulator and maps
each public AI family to the closest native policy profile.

## Catanatron leaderboard aliases

The Catanatron docs leaderboard lists these bot families in descending strength.
The CLI accepts both compact codes and Catanatron-style names with `n` values.

| Label examples | Native adapter |
| --- | --- |
| `AlphaBeta(n=2)`, `AB2`, `CATANATRON` | Rust alpha-beta depth 2 |
| `ValueFunction`, `F` | Rust value-function heuristic |
| `GreedyPlayouts(n=25)`, `GreedyPlayouts25`, `GP` | Rust playout policy with 25 playouts/action |
| `MCTS(n=100)`, `MCTS100`, `MCTS` | Rust playout policy with 100 playouts/action |
| `WeightedRandom`, `W` | Rust weighted-random strategy |
| `VictoryPoint`, `VP` | Rust victory-point strategy |
| `Random`, `R` | Rust random strategy |

## External AI aliases

| Label | Online project | Native adapter |
| --- | --- | --- |
| `JSETTLERS` | https://github.com/jdmonin/JSettlers2 | Rust value-function heuristic |
| `QSETTLERS` | https://github.com/akrishna77/CS7641-QSettlers | Rust value-function heuristic |
| `RLCATAN` | https://github.com/henrycharlesworth/settlers_of_catan_RL | Rust light playout/MCTS-style policy |
| `CATANAI` | https://github.com/kvombatkere/Catan-AI | Rust medium playout/MCTS-style policy |
| `BOTAN` | https://github.com/sambattalio/settlers_of_botan | Rust victory-point strategy |
| `ZARNS` | Generic MCTS-style Catan bot profile | Rust light playout/MCTS-style policy |
| `PYCATAN` | Python strategy-bot family profile | Rust weighted-random strategy |
| `CATANATRON` | https://github.com/bcollazo/catanatron | Rust alpha-beta depth 2 |
| `MONTECATANO` | https://www.reddit.com/r/Catan/comments/hvx3ur/monte_catano_ai_simulation_engine_for_settlers_of/ | Rust playout policy with 25 playouts/action |
| `SMARTSETTLERS` | SmartSettlers / UCT-style Catan research family | Rust playout policy with 25 playouts/action |
| `HENRYHORSE` | https://github.com/HenryHorse/catan | Rust light playout/MCTS-style policy |
| `JUSTINASHER` | https://justinasher.me/catan_ai | Rust value-function heuristic |
| `RASMUSGREVE` | https://github.com/rasmusgreve/catan | Rust weighted-random strategy |
| `HRODGEIR` | https://github.com/hrodgeir/Catan | Rust victory-point strategy |
| `MATTYB5722` | https://github.com/mattyb5722/Catan-Smart-Edition | Rust weighted-random strategy |

Use the short native codes `R`, `W`, `VP`, `F`, `AB`, `SAB`, and `P` when you
want the underlying Rust policies directly.
