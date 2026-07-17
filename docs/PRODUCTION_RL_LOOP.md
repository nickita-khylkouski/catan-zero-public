# Canonical production RL turn

`tools/loop.py` is the supported top-level coordinator for a new improvement
turn. It exposes three options:

```text
--config LOOP.json --state-dir /durable/turn-id [--go]
```

Without `--go`, it prints the exact remaining commands. With `--go`, it runs
the fixed transaction below and journals every completed boundary atomically.

```text
generate  H100 sealed executor terminal-success receipt
   ↓
harvest   direct host-to-controller SSH harvest relocation receipt
   ↓
audit     post-wave audit and selected-game manifest
   ↓
composite authenticated memmap descriptor and build receipt
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
- Fleet generation uses `--go --wait`. A detached `status=launched`
  acknowledgement is not stage completion; every sealed lane job must be
  `complete` before harvest starts.
- Each stage consumes an immutable output from its immediate predecessor.
- Typed CLI bindings prove that the composite is the learner's actual
  `--data`, its build receipt is the learner's actual
  `--composite-build-receipt`, the learner checkpoint is the evaluator's
  actual `--candidate`, and the typed evaluation adjudication is promotion's
  actual `--adjudication`.
- The audit edge is the pair `OUT` and `OUT.selected_games.json`; the composite
  edge is the pair `OUT/memmap_composite.json` and
  `OUT/build_receipt.json`. The output directory itself is not a substitute
  for either typed artifact.
- Stage outputs must be fresh file or directory artifacts. Successful outputs, inputs, commands, and
  logs are content-addressed in `state.json`.
- Restarting the same turn replays hashes and resumes after the last committed
  stage. Drift fails closed instead of repeating a generation, learner, or
  promotion side effect.
- Every stage input is re-hashed after the child exits and before the stage is
  committed, so a tool cannot mutate consumed evidence and advance the same
  turn toward promotion.
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
| generate | `tools/fleet/a1_production_executor.py run --go --wait` |
| harvest | `tools/fleet/a1_harvest_transaction.py` |
| audit | `tools/a1_pre_wave_contract.py audit` |
| composite | `tools/a1_build_post_wave_composite.py` |
| train | issued `a1_one_dose_train.py --go`, `a1_scratch_train.py --go` |
| evaluate | `tools/a1_candidate_promotion_pack.py` |
| promote | `tools/a1_promotion_transaction.py promote --go` |

The historical shell fleet launcher was deleted. New production turns must
enter through this coordinator and its sealed executor; Git history retains
the old implementation when an issued receipt needs forensic review.

The selected parent-update treatment is
`configs/training/a1_parent_update_35m_b200.schema1.json`: exact V2 incumbent,
an explicit V2->V6 information-contract migration initializer, fresh AdamW,
0.25x shared-trunk learning rate, 12 steps, and 8x64=512 global batch. The
migration is measured and reproducible but **not** function-preserving and is
non-promotable until it is commissioned by the matched strength gates. New
parent-update turns must pass the config and an immutable
`--information-contract-migration-receipt` to `a1_one_dose_train.py`; the loop
binds both as train inputs. Generic learner overrides remain diagnostic-only.

`tools/evaluate.py` emits a matched internal H2H source report; it is not a
promotion adjudication. After the existing matched evaluators have emitted
candidate/champion calibration, internal H2H, external-panel, and high-regret
reports, run `tools/a1_candidate_promotion_pack.py`. It derives and replays all
five required evidence kinds, bucket veto, prior-cohort exclusions, and the
final `a1-promotion-adjudication-v2`. In a canonical loop it is the evaluate
stage entry point and its exact `--out` becomes promotion's `--adjudication`;
its training receipt and report must be the immediate train-stage outputs.

```bash
python tools/a1_candidate_promotion_pack.py \
  --contract-lock "$LOCK" \
  --training-receipt "$TRAIN_RECEIPT" \
  --training-report "$TRAIN_REPORT" \
  --registry "$REGISTRY" --current-pointer "$CURRENT" \
  --candidate "$CANDIDATE" --candidate-version "$CANDIDATE_VERSION" \
  --champion "$CHAMPION" --champion-version "$CHAMPION_VERSION" \
  --candidate-calibration "$CANDIDATE_CALIBRATION" \
  --champion-calibration "$CHAMPION_CALIBRATION" \
  --internal-h2h "$INTERNAL_H2H" \
  --candidate-panel "$CANDIDATE_PANEL" \
  --champion-panel "$CHAMPION_PANEL" \
  --high-regret-report "$HIGH_REGRET" \
  --prior-cohort "dose-screen:internal_h2h=$DOSE_SCREEN" \
  --out "$EVAL_DIR/adjudication.json" \
  --cohort-exclusions-out "$EVAL_DIR/cohort-exclusions.json" \
  --receipt "$EVAL_DIR/pack.receipt.json"
```
