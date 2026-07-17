# CatanZero

CatanZero is an expert-iteration research stack for building a strong Catan
agent. The commissioned production track is currently two-player, no-trade
Catan; four-player trading remains a separate benchmark target.

## Production loop

```text
B12 champion
  -> coherent-public n128 self-play
  -> authenticated harvest, audit, and composite
  -> 12-step parent update on 8 B200 GPUs
  -> paired n128 candidate-vs-B12 evaluation
  -> statistical promotion or rejection
```

Use `tools/loop.py` for a complete turn. Use the `catan-zero` command for an
individual config-first stage. The supported stage launchers are:

- `tools/generate.py`
- `tools/train.py`
- `tools/evaluate.py`

Their complete science defaults live in `configs/production_recipes.json` and
the referenced generation, training, and evaluation configs. The large
internal engines (`generate_gumbel_selfplay_data.py`, `train_bc.py`, and the
search harnesses) are implementation details, not public experiment CLIs.

The current H100 authority is `configs/gpu_fleet_h100_8x6.json`: six identical
8×H100 nodes. New code first runs the one-node 8-GPU/88-game pilot described in
`docs/operations/A1_H100_8X_PILOT.md`; only a successful complete turn scales
to all 48 H100s. Training uses one 8×B200 node.

## Current science

- Champion: B12, SHA-256
  `1871f710623e0ee1ff8cb6d5fb659221f5e905f2718e502ca6a5b67a0bb6051c`.
- Search: single coherent public-belief tree, global n128, native Rust
  simulation/features/MCTS, strict-FP32 EvalServer batching, D6 root averaging,
  and a 5% duplicate-search reliability audit.
- Information: exact own private state plus complete public state/history and
  public-card deductions; inaccessible hidden truth is never exposed.
- Learner: current-v5 entity model with a split value suffix and zero-output
  topology upgrade, fresh AdamW, global batch 512, 12 steps, and a 0.25× shared
  trunk learning rate.
- Forced actions: zero policy mass, retained value supervision.
- Promotion: paired seeds and seat swaps under the same n128 operator; no
  automatic replacement from training loss.

Selection evidence is in
`docs/evidence/A1_COHERENT_V5_B12_SELECTION_20260717.json`.

## Repository map

- `src/catan_zero/search/` — coherent-public Gumbel chance MCTS, native
  evaluator, and EvalServer batching.
- `src/catan_zero/rl/` — entity model, self-play, features, histories, and
  learner support.
- `src/catan_zero/adapters/` — Python/Rust engine and action equivalence.
- `tools/fleet/` — receipt-bound generation, harvest, supervision, and stop
  transactions.
- `configs/generation/`, `configs/training/`, `configs/eval/` — complete
  production defaults.
- `tests/` — rule, information-set, model, search, learner, and operator
  regressions.
- `RL_AGENT_HANDOFF.md` — current operator and research handoff.
- `CODEBASE_GUIDE.md` — architecture guide.
- `docs/PRODUCTION_RL_LOOP.md` — transaction semantics.

Git history is the archive. Do not add historical archive directories or
duplicate launchers/runbooks to the working tree.

## Development rules

1. Preserve information-set invariance: changing hidden truth unavailable to
   the acting player must not change its inference or search.
2. Preserve exact recipe identity across generation, learning, and evaluation.
3. Never chain a rejected candidate or restore its optimizer state.
4. A policy target is valid only for its bound checkpoint/search/belief
   operator contract.
5. Performance changes require semantic parity; search-policy changes require
   paired playing-strength evidence.
6. One generator and one EvalServer belong to one physical GPU.
7. Never reinterpret a detached launch receipt as completed generation.
8. Keep commissioned defaults in configs, not optional CLI flags.

## Install

The project targets Python 3.11 and a matching `catanatron_rs` wheel. Follow
the pinned installer and environment checks in `tools/install_v1_freeze.sh`.
Do not silently fall back from the native Rust path to Python feature
construction or simulation.

## Benchmark scope

The long-term target is full four-player Catan with structured trades,
player-chosen discards, robber decisions, and hidden resources/development
cards. Current promotion claims must remain explicitly scoped to the mature
two-player/no-trade production track until the four-player simulator,
observation, action, and neutral-evaluation contracts are complete.
