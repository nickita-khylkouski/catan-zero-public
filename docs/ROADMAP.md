# CatanZero Roadmap

## Goal

Build the strongest verifiably evaluated full four-player Catan agent.

The final target is Colonist-level strength, but any live-platform evaluation
must be permissioned. The primary research metric is CatanBench-4P-Full-v1.

## Roadmap

### Benchmark Contract

- Freeze full four-player rules.
- Define structured trade protocol.
- Add deterministic replay and seed bundles.
- Add hidden-information leakage tests.
- Add baseline opponent suite.

Exit gate: no known rule mismatch, deterministic replay, and no actor hidden
state leakage.

### Human Foundation

- Import human game logs into per-decision records.
- Train flat behavior-cloning baseline.
- Train graph/history model.
- Train value, belief, trade, and opponent heads.

Exit gate: beats heuristic baselines and calibrates beliefs/trade predictions.

### Regularized Self-Play

- Start from the human model.
- Train PPO/VRPO-style policies with centralized critics.
- Use terminal win reward.
- Keep human-policy KL anchor early.

Exit gate: beats human foundation on frozen mixed-opponent benchmark.

### Population League

- Main agents, historical champions, human clones, heuristics, and exploiters.
- Mixed four-player lineups.
- Fresh best-response attacks before promotion.

Exit gate: improves cross-play rating without exploitable regressions.

### Trading

- Structured offer generator.
- Acceptance/counteroffer predictor.
- Trade expected-value model.
- Trade exploiters and no-trade stress tests.

Exit gate: trade-enabled agent improves full-game win rate without spam or
self-play-only conventions.

### Belief-Aware Search

- Sample hidden worlds from belief model.
- Search top legal actions with exact simulator.
- Back up four-player value vectors.
- KL-constrain search policy from robust blueprint.

Exit gate: fixed-latency search improves win rate without increasing exploiter
success.

### Distillation

- Store search-improved targets.
- Train fast policy to imitate search.
- Keep championship search agent and fast distilled agent.

Exit gate: distilled policy retains most search gain.

