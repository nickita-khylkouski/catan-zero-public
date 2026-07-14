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
export WAVE_ID=a1-v5-recovery-n128-p4-12000games-64gpu-20260714-r1
export WAVE=$PROD/contracts/$WAVE_ID
export LOCK=$WAVE/lock.json
export RENDER=$WAVE/render/commands.json

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
That selection contains exactly 12,000 complete games: 9,600 current-producer,
1,800 recovery-reference, and 600 hard-negative games. Reserve attempts do not
enter the learner.

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
  --out "$COMPOSITE"
EOF
```

The learner input is `$COMPOSITE/memmap_composite.json`; the required build
authority is `$COMPOSITE/build_receipt.json`. The descriptor samples games at
64% current, 12% recovery-reference, 4% hard-negative, and 20% historical
replay. Fresh search-policy targets train policy; all components retain value
supervision. Forced rows carry zero policy weight and full value weight.

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
- value-head LR multiplier `0.3`;
- no train-time D6 augmentation;
- public-information masking and whole-game component-balanced validation.

The current canonical trainer produces one decisive step-128 checkpoint. It
does not yet emit step-64/96 snapshots. Do not obtain those by chaining
candidates or running additional production doses; that would repeat the
failure mode this recipe was designed to eliminate.

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
write `$GATE/standard-promotion-adjudication.json`. Then build the conjunctive
recovery authority with the independently collected `$F7_REPORT`:

```bash
"$PY" tools/a1_v5_recovery_gate.py \
  --recovery-receipt "$RECOVERY_RECEIPT" \
  --contract-lock "$LOCK" \
  --standard-adjudication "$GATE/standard-promotion-adjudication.json" \
  --training-receipt "$TRAIN_RECEIPT" \
  --registry "$RECOVERY_REGISTRY" \
  --current-pointer "$RECOVERY_POINTER" \
  --f7-nonregression-report "$F7_REPORT" \
  --out "$GATE/recovery-full-gate.authority.json"
```

This authority never auto-promotes. A candidate advances only if the standard
gate proves strict H1 over v5 and the independent f7 cohort does not reach H0.
If it fails, retain v5 and diagnose from the completed fixed cohorts; do not
continue training the failed candidate.
