# Stage 0: Benchmark and Simulator Certification

Stage 0 exists to prevent later training from optimizing bugs.

## Build Order

1. Rules contract: `catan_rules_v1.json`.
2. Typed schemas: actions, observations, events, decisions.
3. Simulator adapter: Catanatron first, clean-room alternative if licensing or
   rule behavior blocks us.
4. Deterministic replay: seed bundle plus action log reproduces the same game.
5. Leakage audit: actor observations are invariant under changes to hidden
   opponent resources or future randomness.
6. Baseline evaluator: random, heuristic, AlphaBeta/MCTS where available.

## Actor/Critic Boundary

The deployed actor may see:

- Public board.
- Public player features.
- Acting player's private resources and development cards.
- Public event history.
- Legal actions.

The actor may not see:

- Opponent resource identities.
- Opponent development-card identities.
- Development deck order.
- Future dice.
- Future robber steal result.

The training critic may use full current simulator state, but it must be a
separate branch that cannot leak into actor inputs.

## Colonist Target

Ranking #1 on Colonist is an evaluation ambition, not a license to automate a
live account. We need either a permissioned integration, an official API/test
environment, or human-operated move entry using the model's recommendations.

