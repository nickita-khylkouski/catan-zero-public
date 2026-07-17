# RL agent handoff

This document describes the current production path only. Git history is the
archive; do not revive retired launchers, recipes, or fleet layouts from old
commits unless an issued receipt must be replayed.

## Objective

Run a repeating 2-player/no-trade expert-iteration turn:

```text
B12 champion
  -> coherent-public n128 self-play on H100s
  -> authenticated harvest/audit/composite
  -> 12-step fresh-Adam parent update on 8 B200s
  -> paired n128 candidate-vs-B12 evaluation
  -> promote only on the sealed statistical gate
```

The top-level operator is `tools/loop.py`. Individual stages are available
through the `catan-zero` config-first CLI. Do not construct commands for the
internal 200-option trainer or generator engines manually.

## Current champion and architecture

- Selected checkpoint: B12.
- SHA-256: `1871f710623e0ee1ff8cb6d5fb659221f5e905f2718e502ca6a5b67a0bb6051c`.
- Selection evidence: `docs/evidence/A1_COHERENT_V5_B12_SELECTION_20260717.json`.
- Entity adapter: `rust_entity_adapter_v5_meaningful_history_v2`.
- Architecture: current-v5 action/public/history surface, relational edge
  policy, and one private value-tower layer.
- Next initializer: exact B12 plus the reviewed zero-output topology residual.
  The upgrade must use
  `entity_graph.current_v5_value_tower_split1+topology_residual_adapter.v1`;
  rebuilding the older f7 composite is forbidden.

## Current science defaults

Generation is globally coherent-public n128. It uses public-belief search,
native Rust simulation/features/MCTS, strict-FP32 EvalServer inference, D6
root averaging at the commissioned width threshold, `c_scale=0.1`, meaningful
public history v2, public card/rule features, and a 5% duplicate-search target
reliability audit. Hidden authoritative truth must never affect a player's
information-set-equivalent root.

The learner is `configs/training/a1_parent_update_35m_b200.schema1.json`:

- exact B12 parent and topology-only function-preserving initializer;
- fresh AdamW, never restored optimizer moments;
- 8 DDP ranks × 64 rows = global batch 512;
- 12 optimizer steps with the step-8 frontier retained;
- shared trunk learning-rate multiplier `0.25`;
- terminal outcomes for value learning;
- forced one-action rows have zero policy mass but retain value supervision;
- opening settlement and road policy mass must each be at least 2%;
- every newly enabled information path must show runtime learning signal.

Checkpoint selection is based on paired playing strength, not the terminal
training loss. Never chain a failed candidate into another learner.

## Fleet

The only checked-in generation authority is
`configs/gpu_fleet_h100_8x6.json`: six identical 8×H100 nodes, 48 GPUs total.
The first run is deliberately one node:

- GPUs 0-7;
- 8 current-producer, 2 recent-history, and 1 hard-negative game per GPU;
- 88 selected games total across 24 category jobs;
- the same n128 science and receipt path used by the full wave.

After the complete generate→train→evaluate pilot succeeds, changing the quota
policy to `balanced_prefix_v1` selects all 48 GPUs and preserves the aggregate
9,600/1,800/600 source quotas. Do not introduce a second fleet launcher or a
hard-coded worker count.

Before touching a host, inspect GPU processes and utilization. Persistent MPS
servers with about 70-80 MiB/GPU are idle infrastructure; active Python,
generator, evaluator, or trainer processes belong to their current owner and
must not be killed.

## Canonical files

- Full turn: `tools/loop.py`
- Generation launcher: `tools/generate.py`
- Training launcher: `tools/train.py`
- Evaluation launcher: `tools/evaluate.py`
- Generation recipe: `configs/generation/coherent_public_n128.schema20.json`
- Training recipe: `configs/training/a1_parent_update_35m_b200.schema1.json`
- Evaluation recipe: `configs/eval/coherent_public_n128.schema20.json`
- Recipe registry: `configs/production_recipes.json`
- Science authority: `tools/a1_current_science_contract.py`
- Fleet executor: `tools/fleet/a1_production_executor.py`
- Pilot instructions: `docs/operations/A1_H100_8X_PILOT.md`
- Loop contract: `docs/PRODUCTION_RL_LOOP.md`

## Operational order

1. Confirm the repository revision and B12 bytes.
2. Issue and replay the B12 topology-only upgrade receipt.
3. Seal the one-node pilot contract from the checked-in fleet manifest.
4. Run generation with `--go --wait`; do not accept a detached launch receipt
   as completion.
5. Harvest directly between GPU hosts, then audit and build the authenticated
   composite.
6. Train on all eight B200 GPUs using the canonical parent recipe.
7. Evaluate step 8 and step 12 against exact B12 using paired seeds and seat
   swaps under the same n128 operator.
8. Promote only if the adjudication passes; otherwise keep B12 and retain the
   diagnostic artifacts.
9. Scale the same sealed path to 48 H100 GPUs only after the pilot closes.

## Things not to reintroduce

- n64 production data or unconditional n256 policy supervision;
- PIMC policy targets for the coherent-public learner;
- hidden-truth features;
- candidate chaining or restored Adam state;
- surprise-only target weighting without reliability evidence;
- generic shell fleet launch/status/stop scripts;
- optional flags for commissioned science defaults;
- historical archive directories or duplicate runbooks.
