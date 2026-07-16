# Canonical production RL turn

`tools/loop.py` is the supported top-level coordinator for a new improvement
turn. It exposes three options:

```text
--config LOOP.json --state-dir /durable/turn-id [--go]
```

Without `--go`, it prints the exact remaining commands. With `--go`, it runs
the fixed transaction below and journals every completed boundary atomically.

```text
generate  H100 sealed executor receipt
   ↓
harvest   direct host-to-controller SSH harvest relocation receipt
   ↓
audit     post-wave audit and selected-game receipt
   ↓
composite authenticated memmap/composite build receipt
   ↓
train     B200 checkpoint, report, and training receipt
   ↓
evaluate  paired candidate/incumbent adjudication receipt
   ↓
promote   atomic registry/pointer promotion receipt
```

The loop configuration owns run identity, paths, placement, and the exact
argument vectors passed to existing sealed stage tools. Search and learner
science continue to live in their schema-versioned generation, training, and
evaluation configs. The coordinator does not reinterpret those settings.

## Required properties

- The repository commit is exact and the tracked checkout is clean.
- The Python interpreter is exact and every command invokes a stage-allowlisted
  tool inside that checkout.
- Commands are argument arrays, never shell strings.
- Each stage consumes at least one immutable receipt from an earlier stage.
- Stage outputs must be fresh files. Successful outputs, inputs, commands, and
  logs are content-addressed in `state.json`.
- Restarting the same turn replays hashes and resumes after the last committed
  stage. Drift fails closed instead of repeating a generation, learner, or
  promotion side effect.
- Promotion is always last and must invoke the existing typed promotion
  transaction with `promote --go`.

## Allowed stage entry points

| Stage | Allowed entry points |
|---|---|
| generate | `tools/generate.py`, `tools/fleet/a1_production_executor.py run --go` |
| harvest | `tools/fleet/a1_harvest_transaction.py` |
| audit | `tools/a1_pre_wave_contract.py audit` |
| composite | `tools/a1_build_post_wave_composite.py`, `tools/build_memmap_corpus.py` |
| train | `tools/train.py`, issued `a1_one_dose_train.py --go`, `a1_scratch_train.py` |
| evaluate | `tools/evaluate.py`, `tools/fleet/a1_h100_eval_fleet.py` |
| promote | `tools/a1_promotion_transaction.py promote --go` |

`tools/fleet/fleet_launch.sh` is intentionally absent. It is a historical
launcher that still expands legacy generation/training flags, including stale
search/history combinations. A new production turn cannot invoke it through
the coordinator. Issued historical receipts remain replayable through their
original tools; this exclusion applies to new work.

The current scratch learner is still scientifically uncommissioned until its
optimizer horizon is selected. The coordinator makes a commissioned turn
repeatable; it does not manufacture authorization for an unresolved recipe.
