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
# Immutable staging checkout containing the builder/trainer code from this
# runbook revision. Do not reuse the stale 20260714 prep tree.
export LEARNER_CODE_ROOT=/home/ubuntu/catan-zero-learner-post-wave-final
export PROD=/home/ubuntu/catan-zero-production
export WAVE_ID=a1-v5-recovery-n128-p4-64000games-64gpu-20260714-r2
export WAVE=$PROD/private/$WAVE_ID
export LOCK=$WAVE/lock.json
export RENDER=$WAVE/rendered/commands.json
export FROZEN_REPO=/home/ubuntu/catan-zero-wave-5ba993a
export FROZEN_VERIFIER_SHA256=sha256:ab5d4ef8d4a3f82ecacb6c94ff613e24041ec9d1d4e2722ae6c65a19220f101c
export FLEET_MANIFEST=$FROZEN_REPO/configs/gpu_fleet_64.json
export HARVEST_SSH_TRANSPORT=$LEARNER_CODE_ROOT/tools/fleet/a1_harvest_ssh_transport.py
export HISTORICAL_FROZEN_REPO=/home/ubuntu/catan-zero-v1
export HISTORICAL_FROZEN_VERIFIER_SHA256=sha256:45594de3835242904a7c3257c5ff644531c4a3c70a447880b20b3b1a23d8c9cc

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
export OLD_INTERPOLATE=$PROD/private/a1-v5-surviving-evidence-20260714/interpolate_checkpoints.v2.py
test "$(sha256sum "$OLD_INTERPOLATE" | cut -d' ' -f1)" = \
  8a8441aff43052e71e1d18799f6c039977ad1b96582d02a45b6b4e11d6da9e78

export LEARNER=$PROD/runs/learner/$WAVE_ID-one-dose-r1
# Training hashes live bytes while evaluation stages a clean Git commit.  Keep
# those identities identical: the learner checkout must be detached and clean.
test -z "$(git -C "$LEARNER_CODE_ROOT" symbolic-ref -q --short HEAD || true)"
test -z "$(git -C "$LEARNER_CODE_ROOT" status --porcelain --untracked-files=all)"
export LEARNER_COMMIT=$(git -C "$LEARNER_CODE_ROOT" rev-parse HEAD)
export CANDIDATE=$LEARNER/candidate.pt
export TRAIN_REPORT=$LEARNER/train.report.json
export TRAIN_RECEIPT=$LEARNER/training.receipt.json
```

`FROZEN_REPO` is only the historical lock-verifier authority. It never supplies
the learner executable. For a `production_composite_v2` dose, the executor
always runs `$LEARNER_CODE_ROOT/tools/train_bc.py` and binds that canonical path
and its exact SHA-256 into the dry-run plan, training transaction, report, claim,
and terminal receipt. Any byte or path drift is rejected before the dose claim.

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
mkdir -p "$(dirname "$HARVEST")"
A1_SSH_FLEET_MANIFEST="$FLEET_MANIFEST" \
"$PY" "$FROZEN_REPO/tools/fleet/a1_harvest_transaction.py" \
  --lock "$LOCK" \
  --render "$RENDER" \
  --destination "$HARVEST" \
  --ssh-command "$HARVEST_SSH_TRANSPORT" \
  --fetch-workers 12

"$PY" "$LEARNER_CODE_ROOT/tools/a1_pre_wave_contract.py" audit \
  --lock "$LOCK" \
  --harvest-relocation "$HARVEST/relocation_map.json" \
  --frozen-repo "$FROZEN_REPO" \
  --frozen-verifier-sha256 "$FROZEN_VERIFIER_SHA256" \
  --out "$AUDIT"
```

Harvest deliberately executes from `$FROZEN_REPO`. The post-wave audit executes
from `$LEARNER_CODE_ROOT` so it understands the truthful
`native_mcts_hot_loop=true` runtime provenance, while the paired frozen flags
authenticate the unchanged r2 lock with the exact path-bound verifier that
sealed it. The emitted audit binds that frozen-verifier authority before any
corpus ingest.

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
export PREP=$LEARNER_CODE_ROOT
if [ ! -f "$HISTORICAL_REF" ]; then
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
fi
test -s "$HISTORICAL_REF"
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
if [ -d "$COMPOSITE" ] && [ ! -f "$COMPOSITE/build_receipt.json" ]; then
  mv "$COMPOSITE" "$COMPOSITE.failed-$(date -u +%Y%m%dT%H%M%SZ)"
fi
if [ ! -f "$COMPOSITE/build_receipt.json" ]; then
sudo unshare -m /bin/bash -s <<EOF
set -euo pipefail
mount --make-rprivate /
mount --bind "$OLD_INTERPOLATE" "$REPO/tools/interpolate_checkpoints.py"
cd "$LEARNER_CODE_ROOT"
sudo -u ubuntu env \
  PYTHONPATH="$LEARNER_CODE_ROOT/src:$LEARNER_CODE_ROOT" "$PY" \
  "$LEARNER_CODE_ROOT/tools/a1_build_post_wave_composite.py" \
  --lock "$LOCK" \
  --selected-game-manifest "$WAVE/post-wave-audit.selected_games.json" \
  --post-wave-audit "$AUDIT" \
  --historical-replay-component "$HISTORICAL_REF" \
  --frozen-repo "$FROZEN_REPO" \
  --frozen-verifier-sha256 "$FROZEN_VERIFIER_SHA256" \
  --historical-frozen-repo "$HISTORICAL_FROZEN_REPO" \
  --historical-frozen-verifier-sha256 "$HISTORICAL_FROZEN_VERIFIER_SHA256" \
  --out "$COMPOSITE"
EOF
fi
test -s "$COMPOSITE/build_receipt.json"
```

The learner input is `$COMPOSITE/memmap_composite.json`; the required build
authority is `$COMPOSITE/build_receipt.json`. The descriptor samples games at
64% current, 12% recovery-reference, 4% hard-negative, and 20% historical
replay. This is a **TEMP-derived V5 objective**, not an exact replication of
the historical TEMP winner: V5 changes the teacher/component composition and
the production descriptor enables per-game policy weighting, while historical
TEMP used n128+n256+replay and kept both per-game weighting flags off. The
closest historical-objective diagnostic therefore uses this same V5 corpus
with only `per_game_policy_weight=false`; per-game weighting ON remains a
separate treatment. All four authenticated components supply policy and value
targets and the KL anchor is zero. Fresh-only policy scope, fresh-only value
scope, and a nonzero replay KL anchor are separate treatments and must not be
bundled into that baseline. Only the three fresh n128 components supply
eligible auxiliary targets. Forced fresh rows carry zero policy weight and
full value weight. The
legacy replay memmap predates preservation of `adapter_version`; the builder
recovers that identity from the original hash-bound raw NPZs, binds the version
for every component in the descriptor, and the loader lazily restores only the
missing legacy column. Mixed, missing, unknown, or checkpoint-incompatible
adapter semantics still fail closed.

`source_authority.json` binds two distinct frozen verifier authorities. The
current r2 lock is replayed with `require_all_job_claims=true`; the already
sealed historical v2 lock is replayed by its exact old verifier with
`require_all_job_claims=false`. Neither verifier is allowed to stand in for the
other, and both authority records are covered by the source-authority digest.

## 4. Execute one independent 8-B200 dose

The canary is a cheap same-host topology receipt. The one-dose executor first
prints the exact command without touching optimizer state. The second command
is the actual 128-step run.

```bash
cd "$LEARNER_CODE_ROOT"
export PYTHONPATH="$LEARNER_CODE_ROOT/src:$LEARNER_CODE_ROOT"
mkdir -p "$LEARNER"
# Canary receipts expire after one hour. Mint one unique receipt immediately
# before the dry-run/GO pair; never reuse a commit-stable stale receipt.
export CANARY=$LEARNER/ddp-canary-${LEARNER_COMMIT:0:12}-$(date +%s%N).json
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
  --frozen-repo "$FROZEN_REPO" \
  --frozen-verifier-sha256 "$FROZEN_VERIFIER_SHA256" \
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
  --frozen-repo "$FROZEN_REPO" \
  --frozen-verifier-sha256 "$FROZEN_VERIFIER_SHA256" \
  --gpu 0 \
  --go
```

The effective production dose is fixed by the lock and topology binder:

- exact v5 initializer; fresh Adam state;
- eight ranks, local batch 512, global batch 4,096;
- 128 optimizer steps = 524,288 sampled rows;
- LR `3e-5`, 100-step warmup, flat schedule;
- policy/soft-target/value weights `1.0/0.9/0.25`;
- no replay KL anchor (`0.0`), preserving the selected TEMP control;
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
export EVAL_CTL=$LEARNER_CODE_ROOT/tools/fleet/a1_h100_eval_fleet.py
mkdir -p "$GATE"
export BASE_EVAL_MANIFEST=$LEARNER_CODE_ROOT/configs/operations/a1-r3-gather-aux64-reproduction-eval600-20260712-r1/fleet64.manifest.json
export EVAL_COMMIT=$LEARNER_COMMIT
export EVAL_REMOTE_REPO=/home/ubuntu/catan-zero-eval-${EVAL_COMMIT:0:12}
export EVAL_MANIFEST=$GATE/fleet64.eval-source-${EVAL_COMMIT:0:12}.json

# The evaluator proves remote_repo HEAD exactly; it does not mutate or update
# source on its own. Stage a separate detached, read-only checkout on all 12
# H100 hosts, preserving the sealed deployment and its Python environment.
"$PY" "$LEARNER_CODE_ROOT/tools/fleet/a1_stage_h100_eval_source.py" \
  --manifest "$BASE_EVAL_MANIFEST" \
  --git-url https://github.com/nickita-khylkouski/catan-zero-public.git \
  --commit "$EVAL_COMMIT" \
  --destination "$EVAL_REMOTE_REPO" \
  --out "$EVAL_MANIFEST"
"$PY" "$LEARNER_CODE_ROOT/tools/fleet/a1_stage_h100_eval_source.py" \
  --manifest "$BASE_EVAL_MANIFEST" \
  --git-url https://github.com/nickita-khylkouski/catan-zero-public.git \
  --commit "$EVAL_COMMIT" \
  --destination "$EVAL_REMOTE_REPO" \
  --out "$EVAL_MANIFEST" \
  --go

wait_eval_phase() {
  local plan="$1"
  local phase="$2"
  local status state
  while true; do
    status=$("$PY" "$EVAL_CTL" --manifest "$EVAL_MANIFEST" status \
      --plan "$plan" --phase "$phase")
    echo "$status"
    state=$("$PY" -c '
import json, sys
x = json.load(sys.stdin)
c = x["counts"]
if any(c[name] for name in ("failed", "stale", "missing", "unsafe")):
    print("failed")
elif c["done"] == len(x["jobs"]):
    print("done")
else:
    print("active")
' <<<"$status")
    case "$state" in
      done) return 0 ;;
      failed) echo "evaluation phase $phase failed" >&2; return 1 ;;
      active) sleep 10 ;;
      *) echo "unknown evaluation state: $state" >&2; return 1 ;;
    esac
  done
}

# All three plans intentionally share the exact same common-random-number
# cohort. Only the candidate checkpoint and run identity differ.
plan_and_run_screen() {
  local step="$1"
  local checkpoint="$2"
  local plan="$3"
  local collected="$4"
  if [ ! -f "$plan" ]; then
    "$PY" "$EVAL_CTL" --manifest "$EVAL_MANIFEST" plan \
    --candidate "$checkpoint" \
    --champion "$V5" \
    --candidate-parent "$V5" \
    --registry "$RECOVERY_REGISTRY" \
    --comparison-mode promotion_parent \
    --internal-pairs 128 --internal-base-seed 6198800000 \
    --external-pairs 64 --external-base-seed 6198801000 \
    --workers-per-gpu 16 \
    --iteration-id "$WAVE_ID-dose-screen-step$step" \
    --seed-cohort-id "$WAVE_ID-dose-screen" \
    --scope full \
    --candidate-c-scale 0.10 --champion-c-scale 0.10 \
    --candidate-value-squash tanh --champion-value-squash tanh \
      --out "$plan"
  fi
  "$PY" "$EVAL_CTL" --manifest "$EVAL_MANIFEST" resume \
    --plan "$plan" --phase internal --dry-run
  "$PY" "$EVAL_CTL" --manifest "$EVAL_MANIFEST" resume \
    --plan "$plan" --phase internal --go
  wait_eval_phase "$plan" internal
  "$PY" "$EVAL_CTL" --manifest "$EVAL_MANIFEST" collect \
    --plan "$plan" --phase internal --output-dir "$collected"
}

plan_and_run_screen 64 "$LEARNER/candidate_step0064.pt" \
  "$SCREEN64_PLAN" "$GATE/quick-step64-collected"
plan_and_run_screen 96 "$LEARNER/candidate_step0096.pt" \
  "$SCREEN96_PLAN" "$GATE/quick-step96-collected"
plan_and_run_screen 128 "$LEARNER/candidate.pt" \
  "$SCREEN128_PLAN" "$GATE/quick-step128-collected"

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
game outcomes. The sealed rule chooses the maximum matched-cohort candidate
score and uses the earliest dose only for an exact tie. The old 200-basis-point
indifference rule is forbidden because its tolerance was nearly the entire
15-Elo promotion target and could select a materially weaker checkpoint.
`select-dose` independently replays the screen and binds that derived checkpoint
to the one-dose receipt; there is no operator-selected step. Add `$DOSE_SCREEN`
as a prior diagnostic cohort when building the final cohort-exclusions manifest.

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
cd "$LEARNER_CODE_ROOT"
export PYTHONPATH="$LEARNER_CODE_ROOT/src:$LEARNER_CODE_ROOT"
export LEARNER_COMMIT=$(git -C "$LEARNER_CODE_ROOT" rev-parse HEAD)
export EVAL_CTL=$LEARNER_CODE_ROOT/tools/fleet/a1_h100_eval_fleet.py
export GATE=$PROD/runs/eval/$WAVE_ID-one-dose-r1
export CHECKPOINT_SELECTION=$GATE/checkpoint-selection.json
test -s "$CHECKPOINT_SELECTION"
export CANDIDATE=$("$PY" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["selected_checkpoint"]["path"])' \
  "$CHECKPOINT_SELECTION")
export EVAL_COMMIT=$LEARNER_COMMIT
export EVAL_MANIFEST=$GATE/fleet64.eval-source-${EVAL_COMMIT:0:12}.json
# Reuse the exact commit-bound manifest staged before the matched screen.
test -s "$EVAL_MANIFEST"
export PARENT_PLAN=$GATE/exact-v5-parent.plan.json
export F7_PLAN=$GATE/fixed-f7-veto.plan.json
mkdir -p "$GATE"

wait_eval_phase() {
  local plan="$1" phase="$2" status state
  while true; do
    status=$("$PY" "$EVAL_CTL" --manifest "$EVAL_MANIFEST" status \
      --plan "$plan" --phase "$phase")
    echo "$status"
    state=$("$PY" -c '
import json, sys
x = json.load(sys.stdin); c = x["counts"]
if any(c[name] for name in ("failed", "stale", "missing", "unsafe")):
    print("failed")
elif c["done"] == len(x["jobs"]):
    print("done")
else:
    print("active")
' <<<"$status")
    case "$state" in
      done) return 0 ;;
      failed) echo "evaluation phase $phase failed" >&2; return 1 ;;
      active) sleep 10 ;;
      *) echo "unknown evaluation state: $state" >&2; return 1 ;;
    esac
  done
}

if [ ! -f "$PARENT_PLAN" ]; then
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
fi

for phase in internal external; do
  "$PY" "$EVAL_CTL" --manifest "$EVAL_MANIFEST" resume \
    --plan "$PARENT_PLAN" --phase "$phase" --dry-run
  "$PY" "$EVAL_CTL" --manifest "$EVAL_MANIFEST" resume \
    --plan "$PARENT_PLAN" --phase "$phase" --go
  wait_eval_phase "$PARENT_PLAN" "$phase"
  "$PY" "$EVAL_CTL" --manifest "$EVAL_MANIFEST" collect \
    --plan "$PARENT_PLAN" --phase "$phase" \
    --output-dir "$GATE/exact-v5-collected"
done

if [ ! -f "$F7_PLAN" ]; then
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
fi

"$PY" "$EVAL_CTL" --manifest "$EVAL_MANIFEST" resume \
  --plan "$F7_PLAN" --phase internal --dry-run
"$PY" "$EVAL_CTL" --manifest "$EVAL_MANIFEST" resume \
  --plan "$F7_PLAN" --phase internal --go
wait_eval_phase "$F7_PLAN" internal
"$PY" "$EVAL_CTL" --manifest "$EVAL_MANIFEST" collect \
  --plan "$F7_PLAN" --phase internal --output-dir "$GATE/f7-collected"

PARENT_RUN_ID=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_id"])' "$PARENT_PLAN")
F7_RUN_ID=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_id"])' "$F7_PLAN")
export PARENT_REPORT=$GATE/exact-v5-collected/$PARENT_RUN_ID/pooled/internal.json
export F7_REPORT=$GATE/f7-collected/$F7_RUN_ID/pooled/internal.json
```

The parent report is the `internal_h2h` source for the ordinary promotion
evidence graph; its verdict must be strict H1. The fixed 600-pair cohort is a
first gate size, not a claim that 600 pairs guarantees power at the +15 Elo
boundary. Only a replayed `[0,+15]` superiority verdict of H1 authorizes
promotion. H0 rejects; `continue` at 600 pairs is unresolved and also blocks
promotion. Do not report `continue` as either proof of +15 Elo or proof that the
candidate regressed. Any extension must be declared as a fresh disjoint cohort
and pooled before adjudication.

Build the remaining immutable inputs from the selected checkpoint. Calibration
uses the trainer's exact whole-game validation seed set. The candidate and
incumbent calibration jobs may run concurrently on B200 GPUs 0 and 1.

```bash
export VAL_SEEDS=${TRAIN_REPORT%.json}.validation_seeds.json
export PROMO=$GATE/promotion-inputs
mkdir -p "$PROMO"
export CAND_CAL=$PROMO/candidate.calibration.json
export V5_CAL=$PROMO/v5.calibration.json
export REGRET=$PROMO/validation-regret.npz
export SUITE=$PROMO/high-regret.suite.json
export HIGH_REPORT=$PROMO/high-regret.report.json
export PARENT_EXTERNAL_CANDIDATE=$GATE/exact-v5-collected/$PARENT_RUN_ID/pooled/external-candidate.json
export PARENT_EXTERNAL_CHAMPION=$GATE/exact-v5-collected/$PARENT_RUN_ID/pooled/external-champion.json

"$PY" tools/phase_sliced_value_calibration.py \
  --shard-dir "$COMPOSITE/filtered_sources" \
  --shard-dir "$PRIOR_RUN" \
  --checkpoint "$CANDIDATE" --device cuda:0 \
  --value-readout scalar --deployed-value-scale 1 \
  --deployed-value-squash tanh \
  --validation-seed-manifest "$VAL_SEEDS" --require-held-out \
  --out "$CAND_CAL" &
CAL_CAND_PID=$!

"$PY" tools/phase_sliced_value_calibration.py \
  --shard-dir "$COMPOSITE/filtered_sources" \
  --shard-dir "$PRIOR_RUN" \
  --checkpoint "$V5" --device cuda:1 \
  --value-readout scalar --deployed-value-scale 1 \
  --deployed-value-squash tanh \
  --validation-seed-manifest "$VAL_SEEDS" --require-held-out \
  --out "$V5_CAL" &
CAL_V5_PID=$!
set +e
wait "$CAL_CAND_PID"; CAL_CAND_RC=$?
wait "$CAL_V5_PID"; CAL_V5_RC=$?
set -e
test "$CAL_CAND_RC" -eq 0
test "$CAL_V5_RC" -eq 0

"$PY" tools/extract_regret_states.py \
  --shard-root "$COMPOSITE/filtered_sources" \
  --shard-root "$PRIOR_RUN" \
  --validation-seed-manifest "$VAL_SEEDS" \
  --top-k 200000 --out "$REGRET"

"$PY" tools/a1_promotion_artifacts.py held-out-suite \
  --manifest "$REGRET" --holdout-fraction 1.0 --holdout-seed 17 \
  --pairs 240 --out "$SUITE"
```

Run the fixed 240-pair high-regret suite across all eight B200s. It uses the
same public information-set search surface as the fleet gate and the exact
native wheel. This panel and its bucket projection are secondary non-regression
vetoes, not additional superiority gates: H0 rejects, while `continue` or H1
permits the promotion decision to proceed. The per-bucket 45% floor and minimum
sample requirements remain independent vetoes. Likewise, the fixed 300-pair f7
panel permits `continue` or H1 and rejects H0. Strict H1 is required only from
the exact recovered-v5 parent gate above.

```bash
export NATIVE_WHEEL=/home/ubuntu/catan-rnd-audit/dist/catanatron_rs-0.1.8-cp311-cp311-manylinux_2_34_x86_64.whl
export NATIVE_WHEEL_SHA256=sha256:f311673efa4d1e697736415cdff38ebb1e7eed3f109b241d5a5097cfb6d7dc2e

"$PY" tools/gumbel_search_cross_net_h2h.py \
  --candidate "$CANDIDATE" --baseline "$V5" \
  --held-out-high-regret-suite "$SUITE" \
  --workers 8 \
  --devices cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7 \
  --threads-per-worker 1 --n-full 128 --c-visit 50 --c-scale .1 \
  --candidate-c-scale .1 --baseline-c-scale .1 --sigma-eval .98 \
  --rescale-noise-floor-c 0 --lazy-interior-chance \
  --correct-rust-chance-spectra --public-observation \
  --information-set-search --no-belief-chance-spectra \
  --determinization-particles 4 --determinization-min-simulations 32 \
  --symmetry-averaged-eval --symmetry-averaged-eval-threshold 20 \
  --evaluator-rust-featurize --native-mcts-hot-loop \
  --value-readout scalar --candidate-value-readout scalar \
  --baseline-value-readout scalar --value-squash tanh \
  --candidate-value-squash tanh --baseline-value-squash tanh \
  --max-depth 80 --max-decisions 600 --max-root-candidates 16 \
  --max-root-candidates-wide 54 --wide-candidates-threshold 24 \
  --gameplay-policy-aggregation mean_improved_policy --gate-config flywheel \
  --engine-repo-commit "$EVAL_COMMIT" \
  --native-wheel-path "$NATIVE_WHEEL" \
  --native-wheel-sha256 "$NATIVE_WHEEL_SHA256" \
  --out "$HIGH_REPORT"
```

Seal all five ordinary evidence envelopes, the high-regret/bucket sources, the
matched-screen exclusions, and the standard adjudication in one fresh pack.
The pack replays both the frozen verifier and the disaster-recovery authority;
it also binds the selected step-64/96/128 checkpoint to the training receipt.

```bash
export PACK=$GATE/promotion-pack-r1

"$PY" tools/a1_v5_recovery_promotion_pack.py \
  --contract-lock "$LOCK" \
  --frozen-repo "$FROZEN_REPO" \
  --frozen-verifier-sha256 "$FROZEN_VERIFIER_SHA256" \
  --recovery-receipt "$RECOVERY_RECEIPT" \
  --training-receipt "$TRAIN_RECEIPT" \
  --training-report "$TRAIN_REPORT" \
  --checkpoint-selection "$CHECKPOINT_SELECTION" \
  --registry "$RECOVERY_REGISTRY" \
  --current-pointer "$RECOVERY_POINTER" \
  --candidate "$CANDIDATE" --candidate-version 6 \
  --champion "$V5" --champion-version 5 \
  --candidate-calibration "$CAND_CAL" \
  --champion-calibration "$V5_CAL" \
  --internal-h2h "$PARENT_REPORT" \
  --candidate-panel "$PARENT_EXTERNAL_CANDIDATE" \
  --champion-panel "$PARENT_EXTERNAL_CHAMPION" \
  --high-regret-report "$HIGH_REPORT" \
  --dose-screen "$DOSE_SCREEN" \
  --out-dir "$PACK"

export STANDARD_ADJUDICATION=$PACK/standard-promotion-adjudication.json
export COHORT_EXCLUSIONS=$PACK/cohort-exclusions.json
```

Then build the conjunctive recovery authority with the independently collected
`$F7_REPORT`:

```bash
"$PY" tools/a1_v5_recovery_gate.py \
  --recovery-receipt "$RECOVERY_RECEIPT" \
  --contract-lock "$LOCK" \
  --standard-adjudication "$STANDARD_ADJUDICATION" \
  --training-receipt "$TRAIN_RECEIPT" \
  --cohort-exclusions "$COHORT_EXCLUSIONS" \
  --registry "$RECOVERY_REGISTRY" \
  --current-pointer "$RECOVERY_POINTER" \
  --f7-nonregression-report "$F7_REPORT" \
  --frozen-repo "$FROZEN_REPO" \
  --frozen-verifier-sha256 "$FROZEN_VERIFIER_SHA256" \
  --out "$GATE/recovery-full-gate.authority.json"
```

This authority never auto-promotes. A candidate advances only if the standard
gate proves strict H1 over v5 and the independent f7 cohort does not reach H0.
If it fails, retain v5 and diagnose from the completed fixed cohorts; do not
continue training the failed candidate. If it passes, replay the exact authority
through the ordinary recoverable promotion transaction. Dry-run first; `--go`
is the sole mutation boundary:

```bash
export FULL_GATE=$GATE/recovery-full-gate.authority.json
export PROMOTION_RECEIPT=$GATE/recovery-promotion.receipt.json

"$PY" tools/a1_promotion_transaction.py promote \
  --registry "$RECOVERY_REGISTRY" \
  --current-pointer "$RECOVERY_POINTER" \
  --contract-lock "$LOCK" \
  --adjudication "$STANDARD_ADJUDICATION" \
  --training-receipt "$TRAIN_RECEIPT" \
  --cohort-exclusions "$COHORT_EXCLUSIONS" \
  --recovery-gate-authority "$FULL_GATE" \
  --frozen-repo "$FROZEN_REPO" \
  --frozen-verifier-sha256 "$FROZEN_VERIFIER_SHA256" \
  --receipt "$PROMOTION_RECEIPT" \
  --reason "r2 64K n128 recovery child passed exact v5 H1 plus f7 veto"

"$PY" tools/a1_promotion_transaction.py promote \
  --registry "$RECOVERY_REGISTRY" \
  --current-pointer "$RECOVERY_POINTER" \
  --contract-lock "$LOCK" \
  --adjudication "$STANDARD_ADJUDICATION" \
  --training-receipt "$TRAIN_RECEIPT" \
  --cohort-exclusions "$COHORT_EXCLUSIONS" \
  --recovery-gate-authority "$FULL_GATE" \
  --frozen-repo "$FROZEN_REPO" \
  --frozen-verifier-sha256 "$FROZEN_VERIFIER_SHA256" \
  --receipt "$PROMOTION_RECEIPT" \
  --reason "r2 64K n128 recovery child passed exact v5 H1 plus f7 veto" \
  --go
```

The gate authority, promotion dry-run, prepared receipt, and committed receipt
all bind the exact frozen verifier path and SHA-256. Replaying either stage from
the ambient current checkout is forbidden.
