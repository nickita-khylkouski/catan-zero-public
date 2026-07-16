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

- The repository commit is exact and the checkout is clean, including
  untracked files. Revision and tool bytes are revalidated before and after
  every stage.
- The Python interpreter is exact and every command invokes an exact
  repo-relative, stage-allowlisted tool (matching a basename is insufficient).
- Commands are argument arrays, never shell strings.
- Each stage consumes an immutable output from its immediate predecessor.
- Typed CLI bindings prove that the composite is the learner's actual
  `--data`, the learner checkpoint is the evaluator's actual `--candidate`,
  and the evaluation receipt is promotion's actual `--adjudication`.
- Stage outputs must be fresh file or directory artifacts. Successful outputs, inputs, commands, and
  logs are content-addressed in `state.json`.
- Restarting the same turn replays hashes and resumes after the last committed
  stage. Drift fails closed instead of repeating a generation, learner, or
  promotion side effect.
- Scratch training must use `--go` and emit a separate completed
  `--execution-receipt`; a successful plan-only invocation cannot advance the
  loop.
- Local stages run in isolated process groups, so a timeout kills descendants.
  Detached fleet tools are admitted only when they expose an exact
  receipt-bound cancellation transaction.
- Promotion is always last and must invoke the existing typed promotion
  transaction with `promote --go`.

## Allowed stage entry points

| Stage | Allowed entry points |
|---|---|
| generate | `tools/fleet/a1_production_executor.py run --go` |
| harvest | `tools/fleet/a1_harvest_transaction.py` |
| audit | `tools/a1_pre_wave_contract.py audit` |
| composite | `tools/a1_build_post_wave_composite.py` |
| train | `tools/train.py`, issued `a1_one_dose_train.py --go`, `a1_scratch_train.py` |
| evaluate | `tools/evaluate.py` |
| promote | `tools/a1_promotion_transaction.py promote --go` |

`tools/fleet/fleet_launch.sh` is intentionally absent. It is a historical
launcher that still expands legacy generation/training flags, including stale
search/history combinations. A new production turn cannot invoke it through
the coordinator. Issued historical receipts remain replayable through their
original tools; this exclusion applies to new work.

The current scratch learner is still scientifically uncommissioned until its
optimizer horizon is selected. The coordinator makes a commissioned turn
repeatable; it does not manufacture authorization for an unresolved recipe.
