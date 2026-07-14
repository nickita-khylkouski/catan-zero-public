# V5 recovery wave: concrete post-wave learner handoff

This is the executable handoff for the first recovery-reference n128 wave. It
does not authorize a learner before the sealed wave has completed. All data
movement starts on the B200 coordinator and goes directly from the H100 hosts;
no artifact is routed through an operator workstation.

## Fixed host and paths

Run these commands on `ubuntu@149.118.65.110` from the canonical checkout that
created the wave lock. The host has eight B200s, 2.8 TiB RAM, and sufficient
local disk for the harvested wave and all four component memmaps.

```bash
set -euo pipefail

export REPO=/home/ubuntu/catan-zero-v1
export PY=$REPO/.venv/bin/python
export PROD=/home/ubuntu/catan-zero-production
export WAVE_ID=a1-v5-recovery-n128-p4-64000games-64gpu-20260714-r2
export WAVE=$PROD/private/$WAVE_ID
export LOCK=$WAVE/lock.json
export RENDER=$WAVE/rendered/commands.json
export FROZEN_REPO=/home/ubuntu/catan-zero-wave-5ba993a
export FROZEN_VERIFIER_SHA256=sha256:ab5d4ef8d4a3f82ecacb6c94ff613e24041ec9d1d4e2722ae6c65a19220f101c

export HARVEST=$PROD/runs/harvest/$WAVE_ID
export AUDIT=$WAVE/post-wave-audit.json
export COMPOSITE=$PROD/runs/composites/$WAVE_ID

export RECOVERY=$PROD/private/a1-v5-disaster-recovery
export RECOVERY_RECEIPT=$RECOVERY/private/receipts/a1-v5-disaster-recovery.receipt.json
export RECOVERY_REGISTRY=$RECOVERY/private/champion_registry.json
export RECOVERY_POINTER=$RECOVERY/private/CURRENT_CHAMPION

export V5=$PROD/runs/learner/a1-production-l1-one-dose-20260712-r3/candidate.pt
export F7=$PROD/runs/learner/a1-infoset-n128-20260710-r2/candidate.pt

export PRIOR_LOCK=/home/ubuntu/catan-zero/runs/rl_program_20260710/a1_infoset_n128_v133/contract.lock.json
export PRIOR_RUN=$PROD/runs/selfplay/a1-infoset-n128-p4-12000games-20260710-r1
export PRIOR_SELECTED=$PRIOR_RUN/a1_post_wave.audit.selected_games.json
export PRIOR_AUDIT=$PRIOR_RUN/a1_post_wave.audit.json
export PRIOR_CORPUS=/home/ubuntu/catan-zero/runs/memmap_a1_fresh_mixed_12000games
export HISTORICAL_ROOT=$PROD/private/historical-replay-v3-for-v5
export HISTORICAL_REF=$HISTORICAL_ROOT/historical_replay.component.json

export LEARNER=$PROD/runs/learner/$WAVE_ID-one-dose-r1
export CANARY=$LEARNER/ddp-canary.json
export CANDIDATE=$LEARNER/candidate.pt
export TRAIN_REPORT=$LEARNER/train.report.json
export TRAIN_RECEIPT=$LEARNER/training.receipt.json
```

The resolved v5 and f7 SHA-256 identities are fixed:

```text
v5  6817ab054506f962a758ebf48addce5cc7eb801bf451cf2d02b62fb91f5da39c
f7  f7e93dfb8cdb713d647b3e142c949d59083de9f719b6688b6faa6c918ce3eed4
```

The f7 path is a safety baseline, not the causal parent. The exact recovered
v5 bytes are both the wave producer and the learner initializer.

The scale profile selects exactly 1,000 complete games per GPU: 800 current
v5 self-play, 150 f7 recovery-reference, and 50 hard-negative. Each lane gets
a 1,024-seed block because the bounded reserves add 5/2/1 attempts, for a
maximum of 1,008 attempted games per lane. The sealed global selected totals
are therefore 51,200/9,600/3,200; the 16 unused seed slots per lane are spacing,
not training games.

## 1. Harvest the completed 64-H100 wave

The harvest transaction reads the 12 remote hosts directly from the immutable
render and publishes one all-or-nothing local tree. It may be started only
after the production executor reports all 192 jobs terminal-successful.

```bash
cd "$REPO"
"$PY" tools/fleet/a1_harvest_transaction.py \
  --lock "$LOCK" \
  --render "$RENDER" \
  --destination "$HARVEST" \
  --fetch-workers 12

"$PY" tools/a1_pre_wave_contract.py audit \
  --lock "$LOCK" \
  --harvest-relocation "$HARVEST/relocation_map.json" \
  --out "$AUDIT"
```

The audit deterministically emits `$WAVE/post-wave-audit.selected_games.json`.
That selection contains exactly 64,000 complete games: 51,200 current-producer,
9,600 recovery-reference, and 3,200 hard-negative games. Reserve attempts do
not enter the learner.

## 2. Seal the already-existing 20% replay component

This is a one-time preparation step. It replays the old lock/audit/source shard
hashes and binds the existing 2,927,924-row memmap as version-3 historical
replay for the version-5 learner. It does not copy or retrain those rows.

The old lock authenticates the historical runtime at its original absolute
paths. The sole intentional byte difference in the live runtime is
`tools/interpolate_checkpoints.py`; replay its exact v2 bytes only inside a
private mount namespace. This neither rewrites nor temporarily replaces the
live deployment seen by generation/evaluation jobs.

```bash
export PREP=/home/ubuntu/a1-learner-prep-20260714
export OLD_INTERPOLATE=$PROD/private/a1-v5-surviving-evidence-20260714/interpolate_checkpoints.v2.py
test "$(sha256sum "$OLD_INTERPOLATE" | cut -d' ' -f1)" = \
  8a8441aff43052e71e1d18799f6c039977ad1b96582d02a45b6b4e11d6da9e78

mkdir -p "$HISTORICAL_ROOT"
sudo unshare -m /bin/bash -s <<EOF
set -euo pipefail
mount --make-rprivate /
mount --bind "$OLD_INTERPOLATE" "$REPO/tools/interpolate_checkpoints.py"
cd "$PREP"
sudo -u ubuntu env PYTHONPATH="$PREP" "$PY" \
  tools/a1_seal_historical_replay_component.py \
  --lock "$PRIOR_LOCK" \
  --selected-game-manifest "$PRIOR_SELECTED" \
  --post-wave-audit "$PRIOR_AUDIT" \
  --corpus "$PRIOR_CORPUS" \
  --producer-version 3 \
  --current-version 5 \
  --out "$HISTORICAL_REF"
EOF
```

If `$HISTORICAL_REF` already exists, do not delete it. Replay its hashes through
the composite builder below; that builder rejects any semantic or byte drift.

## 3. Materialize the selected 64/12/4/20 composite

`a1_build_post_wave_composite.py` filters harvested NPZ files by the audited
whole-game selection, builds three source-pure fresh memmaps, attaches the
sealed historical component, and writes the promotion-eligible descriptor and
atomic build receipt.

The composite builder replays the old component authority as well as the new
wave authority. Run it in the same private namespace so the old code-tree
fingerprint remains replayable without changing the live checkout:

```bash
sudo unshare -m /bin/bash -s <<EOF
set -euo pipefail
mount --make-rprivate /
mount --bind "$OLD_INTERPOLATE" "$REPO/tools/interpolate_checkpoints.py"
cd "$PREP"
sudo -u ubuntu env PYTHONPATH="$PREP" "$PY" \
  tools/a1_build_post_wave_composite.py \
  --lock "$LOCK" \
  --selected-game-manifest "$WAVE/post-wave-audit.selected_games.json" \
  --post-wave-audit "$AUDIT" \
  --historical-replay-component "$HISTORICAL_REF" \
  --frozen-repo "$FROZEN_REPO" \
  --frozen-verifier-sha256 "$FROZEN_VERIFIER_SHA256" \
  --out "$COMPOSITE"
EOF
```

The learner input is `$COMPOSITE/memmap_composite.json`; the required build
authority is `$COMPOSITE/build_receipt.json`. The descriptor samples games at
64% current, 12% recovery-reference, 4% hard-negative, and 20% historical
replay. Only the three fresh n128 components supply policy, value, and eligible
auxiliary targets. Historical replay is an authenticated behavior anchor only:
it contributes forward `KL(prior || candidate)` at conditional weight `0.006`,
not stale policy cross-entropy or old-policy return/value labels. Forced fresh
rows carry zero policy weight and full value weight. The
legacy replay memmap predates preservation of `adapter_version`; the builder
recovers that identity from the original hash-bound raw NPZs, binds the version
for every component in the descriptor, and the loader lazily restores only the
missing legacy column. Mixed, missing, unknown, or checkpoint-incompatible
adapter semantics still fail closed.

## 4. Execute one independent 8-B200 dose

The canary is a cheap same-host topology receipt. The one-dose executor first
prints the exact command without touching optimizer state. The second command
is the actual 128-step run.

```bash
mkdir -p "$LEARNER"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  "$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
  tools/a1_ddp_epoch_canary.py --out "$CANARY"

# Read-only render of the exact training transaction.
"$PY" tools/a1_one_dose_train.py \
  --lock "$LOCK" \
  --data "$COMPOSITE/memmap_composite.json" \
  --composite-build-receipt "$COMPOSITE/build_receipt.json" \
  --checkpoint "$CANDIDATE" \
  --report "$TRAIN_REPORT" \
  --receipt "$TRAIN_RECEIPT" \
  --python "$PY" \
  --topology b200-8gpu-ddp \
  --ddp-canary-receipt "$CANARY" \
  --gpu 0

# Execute the byte-identical transaction.
"$PY" tools/a1_one_dose_train.py \
  --lock "$LOCK" \
  --data "$COMPOSITE/memmap_composite.json" \
  --composite-build-receipt "$COMPOSITE/build_receipt.json" \
  --checkpoint "$CANDIDATE" \
  --report "$TRAIN_REPORT" \
  --receipt "$TRAIN_RECEIPT" \
  --python "$PY" \
  --topology b200-8gpu-ddp \
  --ddp-canary-receipt "$CANARY" \
  --gpu 0 \
  --go
```

The effective production dose is fixed by the lock and topology binder:

- exact v5 initializer; fresh Adam state;
- eight ranks, local batch 512, global batch 4,096;
- 128 optimizer steps = 524,288 sampled rows;
- LR `3e-5`, 100-step warmup, flat schedule;
- policy/soft-target/value weights `1.0/0.9/0.25`;
- replay-only forward-KL behavior anchor `0.006`;
- value-head LR multiplier `0.3`;
- no train-time D6 augmentation;
- public-information masking and whole-game component-balanced validation.

The canonical trainer emits model-only step-64 and step-96 snapshots plus the
ordinary resumable step-128 checkpoint. All three come from the same fresh-Adam
trajectory and authenticated sample order. The intermediate snapshots have no
optimizer/progress sidecars by design: select among 64/96/128 with a matched
candidate-vs-v5 screen, then run the full disjoint gate only on the selected
snapshot. Never initialize another learner from a snapshot or candidate; that
would repeat the candidate-chaining failure this recipe was designed to remove.

### 4.1 Seal the matched dose screen and resolve the candidate

Run one internal-H2H quick screen for each of
`$LEARNER/candidate_step0064.pt`, `$LEARNER/candidate_step0096.pt`, and the
terminal `$LEARNER/candidate.pt`. Use the exact same v5 baseline, ordered seed
cohort, orientations, and search settings in all three fleet plans. Each input
below is the canonical `pooled/internal.json` emitted by the fleet collector,
not a lane report or a hand-written summary:

```bash
export GATE=$PROD/runs/eval/$WAVE_ID-one-dose-r1
# `collect` nests each pooled report under its plan's run_id.
export SCREEN64_PLAN=$GATE/quick-step64.plan.json
export SCREEN96_PLAN=$GATE/quick-step96.plan.json
export SCREEN128_PLAN=$GATE/quick-step128.plan.json
SCREEN64_RUN_ID=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_id"])' "$SCREEN64_PLAN")
SCREEN96_RUN_ID=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_id"])' "$SCREEN96_PLAN")
SCREEN128_RUN_ID=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_id"])' "$SCREEN128_PLAN")
export SCREEN64=$GATE/quick-step64-collected/$SCREEN64_RUN_ID/pooled/internal.json
export SCREEN96=$GATE/quick-step96-collected/$SCREEN96_RUN_ID/pooled/internal.json
export SCREEN128=$GATE/quick-step128-collected/$SCREEN128_RUN_ID/pooled/internal.json
export DOSE_SCREEN=$GATE/matched-dose-screen.json
export CHECKPOINT_SELECTION=$GATE/checkpoint-selection.json

"$PY" tools/a1_promotion_transaction.py build-dose-screen \
  --step64-report "$SCREEN64" \
  --step96-report "$SCREEN96" \
  --step128-report "$SCREEN128" \
  --output "$DOSE_SCREEN"

"$PY" tools/a1_promotion_transaction.py select-dose \
  --training-receipt "$TRAIN_RECEIPT" \
  --training-report "$TRAIN_REPORT" \
  --screen-evidence "$DOSE_SCREEN" \
  --output "$CHECKPOINT_SELECTION"

# From here onward CANDIDATE is the deterministically selected 64/96/128 model,
# not necessarily the terminal checkpoint used as the training output target.
export CANDIDATE=$("$PY" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["selected_checkpoint"]["path"])' \
  "$CHECKPOINT_SELECTION")
```

`build-dose-screen` replays the candidate checkpoint bytes in all three pooled
reports, requires byte-identical baseline/search configuration and identical
ordered `(game_seed, orientation)` keys, and derives every score from retained
game outcomes. The sealed rule chooses the earliest dose within 200 basis
points of the best observed win rate. `select-dose` independently replays the
screen and binds that derived checkpoint to the one-dose receipt; there is no
operator-selected step. Add `$DOSE_SCREEN` as a prior diagnostic cohort when
building the final cohort-exclusions manifest.

## 5. Gate against both required baselines

Evaluation must use fresh, disjoint cohorts. The ordinary adjudication is a
strict-H1 comparison against the exact recovered v5 parent. A second fixed
300-pair panel at base seed `6_199_100_000` compares the same candidate to f7.
The f7 panel is a veto: H0 rejects, while H1 or continue permits the manual
recovery decision.

Use the sealed 64-H100 manifest and launch both cohorts from the B200. The
ordinary internal/external intervals and the fixed f7 interval below are
pairwise disjoint and were unused in the authoritative VAL-only ledger when
this handoff was sealed.

```bash
export EVAL_CTL=$REPO/tools/fleet/a1_h100_eval_fleet.py
export EVAL_MANIFEST=$REPO/configs/operations/a1-r3-gather-aux64-reproduction-eval600-20260712-r1/fleet64.manifest.json
export GATE=$PROD/runs/eval/$WAVE_ID-one-dose-r1
export PARENT_PLAN=$GATE/exact-v5-parent.plan.json
export F7_PLAN=$GATE/fixed-f7-veto.plan.json
mkdir -p "$GATE"

"$PY" "$EVAL_CTL" --manifest "$EVAL_MANIFEST" plan \
  --candidate "$CANDIDATE" \
  --champion "$V5" \
  --candidate-parent "$V5" \
  --registry "$RECOVERY_REGISTRY" \
  --comparison-mode promotion_parent \
  --internal-pairs 600 --internal-base-seed 6199000000 \
  --external-pairs 500 --external-base-seed 6199010000 \
  --workers-per-gpu 16 \
  --iteration-id "$WAVE_ID-one-dose-exact-v5" \
  --seed-cohort-id "$WAVE_ID-one-dose-exact-v5" \
  --scope full \
  --candidate-c-scale 0.10 --champion-c-scale 0.10 \
  --candidate-value-squash tanh --champion-value-squash tanh \
  --out "$PARENT_PLAN"

for phase in internal external; do
  "$PY" "$EVAL_CTL" --manifest "$EVAL_MANIFEST" launch \
    --plan "$PARENT_PLAN" --phase "$phase" --dry-run
  "$PY" "$EVAL_CTL" --manifest "$EVAL_MANIFEST" launch \
    --plan "$PARENT_PLAN" --phase "$phase" --go
done

"$PY" "$EVAL_CTL" --manifest "$EVAL_MANIFEST" collect \
  --plan "$PARENT_PLAN" --phase internal --output-dir "$GATE/exact-v5-collected"
"$PY" "$EVAL_CTL" --manifest "$EVAL_MANIFEST" collect \
  --plan "$PARENT_PLAN" --phase external --output-dir "$GATE/exact-v5-collected"

"$PY" "$EVAL_CTL" --manifest "$EVAL_MANIFEST" plan \
  --candidate "$CANDIDATE" \
  --champion "$F7" \
  --candidate-parent "$V5" \
  --registry "$RECOVERY_REGISTRY" \
  --comparison-mode recovery_safety_reference \
  --historical-comparison-reason disaster_recovery_f7_non_regression_veto \
  --internal-pairs 300 --internal-base-seed 6199100000 \
  --external-pairs 32 --external-base-seed 6199130000 \
  --workers-per-gpu 16 \
  --iteration-id "$WAVE_ID-one-dose-fixed-f7" \
  --seed-cohort-id "$WAVE_ID-one-dose-fixed-f7" \
  --scope full \
  --candidate-c-scale 0.10 --champion-c-scale 0.10 \
  --candidate-value-squash tanh --champion-value-squash tanh \
  --out "$F7_PLAN"

"$PY" "$EVAL_CTL" --manifest "$EVAL_MANIFEST" launch \
  --plan "$F7_PLAN" --phase internal --dry-run
"$PY" "$EVAL_CTL" --manifest "$EVAL_MANIFEST" launch \
  --plan "$F7_PLAN" --phase internal --go
"$PY" "$EVAL_CTL" --manifest "$EVAL_MANIFEST" collect \
  --plan "$F7_PLAN" --phase internal --output-dir "$GATE/f7-collected"

PARENT_RUN_ID=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_id"])' "$PARENT_PLAN")
F7_RUN_ID=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_id"])' "$F7_PLAN")
export PARENT_REPORT=$GATE/exact-v5-collected/$PARENT_RUN_ID/pooled/internal.json
export F7_REPORT=$GATE/f7-collected/$F7_RUN_ID/pooled/internal.json
```

The parent report is the `internal_h2h` source for the ordinary promotion
evidence graph; its verdict must be strict H1. Complete the ordinary
calibration, external, high-regret, and bucket evidence from the same plan and
write `$GATE/standard-promotion-adjudication.json`. Build the candidate-bound
cohort-exclusions manifest at `$GATE/cohort-exclusions.json`; it must enumerate
every prior diagnostic/selection cohort so the gate can replay that both the
ordinary final cohorts and the fixed f7 veto cohort are fresh. Then build the
conjunctive recovery authority with the independently collected `$F7_REPORT`:

```bash
"$PY" tools/a1_v5_recovery_gate.py \
  --recovery-receipt "$RECOVERY_RECEIPT" \
  --contract-lock "$LOCK" \
  --standard-adjudication "$GATE/standard-promotion-adjudication.json" \
  --training-receipt "$TRAIN_RECEIPT" \
  --cohort-exclusions "$GATE/cohort-exclusions.json" \
  --registry "$RECOVERY_REGISTRY" \
  --current-pointer "$RECOVERY_POINTER" \
  --f7-nonregression-report "$F7_REPORT" \
  --out "$GATE/recovery-full-gate.authority.json"
```

This authority never auto-promotes. A candidate advances only if the standard
gate proves strict H1 over v5 and the independent f7 cohort does not reach H0.
If it fails, retain v5 and diagnose from the completed fixed cohorts; do not
continue training the failed candidate.
