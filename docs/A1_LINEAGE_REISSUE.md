# A1 lineage reissue after loss of derived promotion files

This runbook is a fallback for one specific failure: the issued r1 dual-arm
locks still exist, but their post-promotion handoff and the receipt/registry/
pointer graph needed to replay it no longer exist. It does not repair or
reinterpret those locks. Their exact fingerprints remain permanently recorded
as superseded evidence, and neither old lock becomes valid again.

The fallback performs a new validation and promotion in a fresh private
namespace, then creates a new pending campaign revision. Every mutating command
has a read-only preflight first. Nothing below claims seeds, renders commands,
launches work, or publishes a release.

## Fixed inputs

Run from commit `c179fe7f3ea314f675af9207275c78ee012a245b` or a descendant whose
generation-runtime files are byte-identical to that commit. The revision binds
the native 0.1.5 runtime, `--native-mcts-hot-loop`, and Rust featurization for
both arms. All non-implementation search science and all three opponent
categories remain identical to r1.

```bash
REPO=/home/ubuntu/catan-zero-v1
PROD=/home/ubuntu/catan-zero-production
RECOVERY=$PROD/recovery/a1-lineage-20260711-r2
R1=$PROD/contracts/a1-dual-arm-20260710-r1
TRAIN_LOCK=/home/ubuntu/catan-zero/runs/rl_program_20260710/a1_infoset_n128_v133/contract.lock.json
CANDIDATE=$PROD/runs/learner/a1-infoset-n128-20260710-r2/candidate.pt
TRAINING_RECEIPT=$PROD/runs/learner/a1-infoset-n128-20260710-r2/training.receipt.json
TRAINING_REPORT=$PROD/runs/learner/a1-infoset-n128-20260710-r2/report.json
CHAMPION=/home/ubuntu/catan-zero/runs/bc/gen3_20260706/checkpoint.pt
REGISTRY=$RECOVERY/private/champion_registry.json
POINTER=$RECOVERY/private/CURRENT_CHAMPION
BOOTSTRAP_RECEIPT=$RECOVERY/private/registry-bootstrap.receipt.json
PROMOTION_RECEIPT=$RECOVERY/private/promotion.receipt.json
HANDOFF=$RECOVERY/private/post-promotion-handoff.json
ADJUDICATION=$RECOVERY/evidence/promotion.adjudication.json
CAMPAIGN=$RECOVERY/campaign/a1-dual-arm-20260711-r2.contract.json
PLACEMENT=$RECOVERY/campaign/a1-dual-arm-20260711-r2.placement.json
LOCKS=$RECOVERY/contracts/locks
```

Before using these commands, independently verify that the candidate SHA-256 is
`f7e93dfb8cdb713d647b3e142c949d59083de9f719b6688b6faa6c918ce3eed4`
and that the two r1 lock file hashes are respectively `88f56891...` (n256) and
`dfee05c8...` (n128). The revision builder enforces the complete hashes.

## 1. Recreate an isolated pre-promotion registry

The first command is read-only. It replays the original sealed A1 training lock
and refuses any incumbent, history, report, or checkpoint drift.

```bash
cd "$REPO"
python3 tools/a1_registry_bootstrap.py \
  --lock "$TRAIN_LOCK" --incumbent "$CHAMPION" \
  --training-receipt "$TRAINING_RECEIPT" --candidate "$CANDIDATE" \
  --registry "$REGISTRY" --current-pointer "$POINTER" \
  --receipt "$BOOTSTRAP_RECEIPT"
```

Only after reviewing that plan, publish the new isolated baseline:

```bash
python3 tools/a1_registry_bootstrap.py \
  --lock "$TRAIN_LOCK" --incumbent "$CHAMPION" \
  --training-receipt "$TRAINING_RECEIPT" --candidate "$CANDIDATE" \
  --registry "$REGISTRY" --current-pointer "$POINTER" \
  --receipt "$BOOTSTRAP_RECEIPT" --go
```

These are recovery-namespace paths, not the lost live registry paths.

## 2. Rebuild and replay typed evaluation evidence

Do not reuse an unverified old adjudication. Rebuild the five envelopes from
the still-immutable source reports. The variables below must name the source
JSON documents accepted by `a1_promotion_artifacts.py`; missing or mismatched
sources fail closed.

```bash
python3 tools/a1_promotion_artifacts.py evidence --kind mechanism_calibration \
  --contract-lock "$TRAIN_LOCK" --candidate "$CANDIDATE" --champion "$CHAMPION" \
  --source candidate_calibration="$CANDIDATE_CALIBRATION" \
  --source champion_calibration="$CHAMPION_CALIBRATION" \
  --out "$RECOVERY/evidence/mechanism-calibration.evidence.json"

python3 tools/a1_promotion_artifacts.py evidence --kind internal_h2h \
  --contract-lock "$TRAIN_LOCK" --candidate "$CANDIDATE" --champion "$CHAMPION" \
  --source internal_h2h="$INTERNAL_H2H" \
  --out "$RECOVERY/evidence/internal-h2h.evidence.json"

python3 tools/a1_promotion_artifacts.py evidence --kind external_panel \
  --contract-lock "$TRAIN_LOCK" --candidate "$CANDIDATE" --champion "$CHAMPION" \
  --source candidate_panel="$CANDIDATE_PANEL" \
  --source champion_panel="$CHAMPION_PANEL" \
  --out "$RECOVERY/evidence/external-panel.evidence.json"

python3 tools/a1_promotion_artifacts.py evidence --kind high_regret \
  --contract-lock "$TRAIN_LOCK" --candidate "$CANDIDATE" --champion "$CHAMPION" \
  --source high_regret="$HIGH_REGRET_SOURCE" \
  --out "$RECOVERY/evidence/high-regret.evidence.json"

python3 tools/a1_promotion_artifacts.py evidence --kind bucket_veto \
  --contract-lock "$TRAIN_LOCK" --candidate "$CANDIDATE" --champion "$CHAMPION" \
  --source bucket_veto="$BUCKET_VETO_SOURCE" \
  --out "$RECOVERY/evidence/bucket-veto.evidence.json"

python3 tools/a1_promotion_artifacts.py adjudicate \
  --contract-lock "$TRAIN_LOCK" --training-receipt "$TRAINING_RECEIPT" \
  --registry "$REGISTRY" --current-pointer "$POINTER" \
  --candidate "$CANDIDATE" --candidate-version 4 \
  --training-report "$TRAINING_REPORT" \
  --champion "$CHAMPION" --champion-version 3 \
  --evidence mechanism_calibration="$RECOVERY/evidence/mechanism-calibration.evidence.json" \
  --evidence internal_h2h="$RECOVERY/evidence/internal-h2h.evidence.json" \
  --evidence external_panel="$RECOVERY/evidence/external-panel.evidence.json" \
  --evidence high_regret="$RECOVERY/evidence/high-regret.evidence.json" \
  --evidence bucket_veto="$RECOVERY/evidence/bucket-veto.evidence.json" \
  --out "$ADJUDICATION"
```

## 3. Execute a new promotion transaction

Preflight is read-only:

```bash
python3 tools/a1_promotion_transaction.py promote \
  --registry "$REGISTRY" --current-pointer "$POINTER" \
  --contract-lock "$TRAIN_LOCK" --adjudication "$ADJUDICATION" \
  --training-receipt "$TRAINING_RECEIPT" --receipt "$PROMOTION_RECEIPT" \
  --reason "revalidated A1 lineage reissue after loss of derived r1 artifacts"
```

After review, repeat exactly with `--go`. Then create the immutable handoff from
the committed receipt:

```bash
python3 tools/a1_promotion_transaction.py promote \
  --registry "$REGISTRY" --current-pointer "$POINTER" \
  --contract-lock "$TRAIN_LOCK" --adjudication "$ADJUDICATION" \
  --training-receipt "$TRAINING_RECEIPT" --receipt "$PROMOTION_RECEIPT" \
  --reason "revalidated A1 lineage reissue after loss of derived r1 artifacts" --go

python3 tools/a1_post_promotion_handoff.py \
  --promotion-receipt "$PROMOTION_RECEIPT" --out "$HANDOFF"
```

## 4. Create, inspect, and materialize a new campaign revision

This creates only a pending blueprint. It fingerprints both exact r1 locks,
starts at seed `300000626944`, uses fresh output roots, and binds the c179fe7
native runtime.

```bash
python3 tools/a1_pre_wave_contract.py revise-generation-campaign \
  --source configs/operations/a1-dual-arm-56gpu-20260710/contract.json \
  --superseded-lock "$R1/locks/n256.lock.json" \
  --superseded-lock "$R1/locks/n128.lock.json" \
  --contract-id a1-dual-arm-n256-n128-56gpu-20260711-r2 \
  --output-root "$PROD/runs/selfplay/a1-dual-arm-20260711-r2" \
  --out "$CAMPAIGN"

python3 tools/a1_pre_wave_contract.py verify-generation-campaign \
  --contract "$CAMPAIGN"

python3 tools/a1_pre_wave_contract.py seal-generation-placement \
  --contract "$CAMPAIGN" \
  --assignments configs/operations/a1-dual-arm-56gpu-20260710/placement.assignments.json \
  --out "$PLACEMENT"

python3 tools/a1_pre_wave_contract.py materialize-generation-campaign \
  --contract "$CAMPAIGN" --promotion-handoff "$HANDOFF" \
  --placement "$PLACEMENT" --out-dir "$LOCKS"

python3 tools/a1_pre_wave_contract.py verify --lock "$LOCKS/n256.lock.json"
python3 tools/a1_pre_wave_contract.py verify --lock "$LOCKS/n128.lock.json"
```

Stop here for review. Rendering, seed claims, executor `--go`, and launch are
deliberately outside this recovery procedure.
