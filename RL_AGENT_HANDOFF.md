# CatanZero RL Fleet Operator Handoff

This document transfers the H100 generation, corpus, training, evaluation, and
promotion workflow to the RL operator. Read it before changing fleet state.

The current production track is **2-player, no-trade, 10 VP**. Results from this
track do not support a four-player full-trade strength claim.

## 1. Current handoff state

| Item | State |
|---|---|
| Local branch | codex/h100-simulation-throughput at 1610901-dirty |
| Publishable release | Blocked. v1.0-deploy predates the H100 work |
| Private masked champion | Missing from this checkout |
| Authoritative production seed ledger | Missing from this checkout |
| Authoritative ChampionRegistry and deployment pointers | Missing from this checkout |
| Full production fleet config | Missing from the default local path |
| 8-H100 canary | Validation lab outside the 24 production H100s |
| Local verification | 1,737 passed, 200 skipped, 4 warnings in 49.13s |
| H100 verification | 1,913 passed, 24 skipped in 18m18s |
| Final H100 delta | 184 targeted tests passed in 7.89s; all 8 GPUs returned to 0 MiB |
| Final CUDA parity | 128 states, prior diff 2.04e-8, value diff 6.54e-7 |
| End-to-end canary smoke | 16/16 games, 0 failed, 9,506 rows, automatic cleanup |
| Production launch | Blocked by the release, champion, ledger, fleet config, and real-champion repeat |
| Promotion/deployment | Blocked by private registry state and an owner-approved gate/veto record |

The canary smoke used a synthetic same-shape checkpoint. Its rows certify
pipeline behavior. They do not certify playing strength and must not train a
model.

## 2. Operator mission and authority

The RL operator owns this sequence:

1. Verify the immutable software and private artifacts.
2. Allocate non-overlapping seeds under one global allocator.
3. Launch and monitor generation one box at a time.
4. Reconcile manifests, run corpus QA, and build a fresh memmap corpus.
5. Launch the canonical 35M DDP control.
6. Run searched candidate-versus-incumbent, neutral, tripwire, and population
   evidence required by the approved promotion policy.
7. Record a promotion transaction and deploy the selected checkpoint only
   after every veto is resolved.

The operator must stop when a prerequisite or acceptance check fails. Do not
repair evidence after the run, reuse a consumed seed, bypass a guard, or promote
on an inconclusive gate.

Repository pushes, release tags, public champion changes, and production fleet
mutations require the authority assigned by the project owner. This handoff
describes the mechanics; it does not grant new external permissions.

## 3. System map

~~~text
Operator workstation
  -> fleet_launch.sh
  -> one detached generator per physical GPU
  -> CPU game workers
  -> one EvalServer and one CUDA context per GPU
  -> Rust game state, features, and Gumbel chance MCTS
  -> strict-FP32 entity-graph policy forward
  -> NPZ decision shards plus manifests
  -> manifest reconciliation and Gumbel QA
  -> duplicate-safe memmap corpus
  -> four-rank DDP train_bc on c4
  -> candidate checkpoint and report
  -> searched cross-net H2H plus neutral panels
  -> manual registry and CKPT deployment transaction
~~~

The generator and trainer have canonical launchers. Corpus attestation and
masked-search promotion still require an operator checklist. No current command
joins every stage into one safe transaction.

## 4. Source-of-truth order

Use each source for its declared scope:

1. Live code, manifests, checkpoint metadata, private registry, and canonical
   ledgers for what actually ran.
2. A dated owner-approved wave record for experiment thresholds and any policy
   decision that supersedes an older plan.
3. docs/plans/CATAN_ZERO_ROADMAP.md and CATAN_ZERO_MASTER_PLAN.md for the
   promotion ladder, tripwires, and research program.
4. RL_AGENT_HANDOFF.md for the end-to-end operator transaction.
5. FLEET.md and tools/fleet/FLEET_CONTROL.md for inventory, installation,
   status, launch, and stop behavior.
6. docs/plans/H100_EXECUTION_UPDATE_2026-07-09.md for measured H100 evidence
   and the current role-pure data-engine experiment.
7. CODEBASE_GUIDE.md for architecture and module ownership.

Stop and update the documentation when same-scope sources disagree. Do not
silently let a throughput recipe replace the roadmap's promotion policy.

## 5. Data quarantine

Never train on:

- Synthetic-checkpoint canary rows, including the 9,506-row smoke.
- The 21,120-row DDP training smoke corpus.
- Canary/evaluation outputs or seeds in [6190000000, 6200000000).
- TF32, torch.compile, MPS, or other rejected experiment outputs.
- Partial generation or harvests with unreconciled counts.
- Any run with worker errors, reused seeds, ambiguous masking, or wrong track.
- A memmap built with duplicate-seed or fill verification disabled.
- A failed memmap output directory left behind after an abort.

Keep canary and evaluation paths outside production DIRS entries. Never merge
CANARY_VAL_ONLY_LEDGER.md into the production seed ledger.

## 6. Production recipe

| Role | Four-GPU production node | Eight-GPU canary |
|---|---|---|
| Teacher | 128 workers/GPU, n128, p-full 1.0, shard 512, batch 96, collector on | 64 workers/GPU |
| Volume | 48 workers/GPU, n64, p-full 0.25, shard 2048, batch 64, collector off | 32 workers/GPU |
| Train | Four DDP ranks, batch 1024/rank | Eight ranks, batch 512/rank |

All generation uses:

- one generator and EvalServer per physical GPU;
- public observations and a masked checkpoint;
- Rust features and corrected chance spectra;
- lazy interior chance;
- c-visit 50.0 and c-scale 0.03;
- immediate queue drain;
- matmul_precision=highest;
- cache size 0;
- no local CUDA fallback;
- CPU affinity;
- MPS off.

The measured synthetic frontier reached 91.85k rows/hour/GPU at
w128/batch96/collector. The 2.20M rows/hour figure assumes all 24 H100s run the
teacher recipe. The standard role layout reserves c4 for training and mixes
teacher and volume nodes, so its aggregate rate differs.

## 7. Release and node acceptance

### 7.1 Publish one immutable release

The maintainer must publish the verified tree under a new release tag and attach:

~~~text
catanatron_rs-0.1.3-cp311-cp311-manylinux_2_34_x86_64.whl
~~~

Do not install v1.0-deploy.

On each node:

~~~bash
set -euo pipefail
export CATAN_REF=<new-h100-release-tag>
curl -fsSL \
  "https://raw.githubusercontent.com/nickita-khylkouski/catan-zero-public/$CATAN_REF/tools/install_v1_freeze.sh" \
  | CATAN_REF="$CATAN_REF" bash

cd /home/ubuntu/catan-zero-v1
test "$(git rev-parse HEAD)" = <recorded-release-commit>
~~~

Stage the real champion on every node that consumes it: c1-c6,
`h100-canary`, and `b200`. Stage the production ledger only on the five
generation nodes c1, c2, c3, c5, and c6:

~~~text
/home/ubuntu/bundle/champion_v0.pt
/home/ubuntu/catan-zero-v1/runs/SEED_LEDGER.md
~~~

Record the release commit, release tag, Rust-wheel SHA-256, checkpoint SHA-256,
and ledger SHA-256 in the wave record. Compare the checkpoint SHA-256 on every
consumer before its run and again before accepting its output; manifests record
a path, not the checkpoint's bytes.

### 7.2 Pin the remote interpreter

The launcher can prefer an older ~/venv over the tree environment. Pass the
tree interpreter:

~~~bash
export REMOTE_PY=/home/ubuntu/catan-zero-v1/.venv/bin/python
~~~

### 7.3 Verify CUDA and acceptance gates

On every GPU node:

~~~bash
cd /home/ubuntu/catan-zero-v1

.venv/bin/python - <<'PY'
import torch
assert torch.cuda.is_available()
assert torch.version.cuda == "12.8", torch.version.cuda
print(torch.__version__, torch.version.cuda)
PY

NOOP_ATOL=1e-4 PY=.venv/bin/python \
  bash scripts/gate.sh --only noop

PY=.venv/bin/python \
  bash scripts/gate.sh --only parity
~~~

The no-op gate needs the real masked champion. A synthetic checkpoint cannot
replace it.

### 7.4 Clear the real-champion capacity gate

Before the 24-H100 rollout, complete the private fleet configuration in Section
8, then repeat the production-shaped teacher recipe on four canary GPUs with
the real champion. Use only the validation seed band and its dedicated ledger;
these rows remain quarantined:

~~~bash
cd <verified-local-checkout>
test -x .venv/bin/python
test -f tools/fleet/fleet_launch.sh

export FLEET_CONF="$HOME/.catan_fleet.conf"
export CKPT=/home/ubuntu/bundle/champion_v0.pt
export LEDGER=/home/ubuntu/catan-zero-v1/runs/CANARY_VAL_ONLY_LEDGER.md
export REMOTE_PY=/home/ubuntu/catan-zero-v1/.venv/bin/python

PY="$REMOTE_PY" GEN_PY="$REMOTE_PY" \
  tools/fleet/fleet_launch.sh h100-canary teacher \
  --base-seed <fresh-seed-in-6190000000-to-6199998464> \
  --gpus 0-3 --workers 128 --games 384 \
  --wave real-champion-capacity
~~~

Inspect the dry run, repeat it with `--go`, and accept all four manifests as in
Section 12. This is at least three complete games per worker and exercises the
same total 512-worker CPU load as one four-GPU production node.

The predeclared capacity threshold for the current plan is a median of at least
80,000 rows/hour/GPU, no GPU below 72,000 rows/hour, zero failed games, zero
worker errors, and no local fallback. Compute each GPU rate as
`manifest.rows_per_sec * 3600`. Record simulations/second, EvalServer request
and row totals, effective batch size, CPU/GPU utilization, power, truncation,
and diversity beside the rate. If the rate gate misses, do not use the 2.20M
rows/hour projection or start all 24 GPUs; profile or re-plan from the measured
real-champion rate. Regardless of the result, never harvest this claim into a
training corpus.

## 8. Private fleet configuration

Create $FLEET_CONF, normally ~/.catan_fleet.conf, as an uncommitted Bash file:

~~~bash
declare -A HOST=(
  [c1]=...
  [c2]=...
  [c3]=...
  [c4]=...
  [c5]=...
  [c6]=...
  [h100-canary]=...
  [b200]=...
)

GPU_SSH_KEY="$HOME/.ssh/gpu_access_ed25519"

# Fill after each launch from the exact OUT path printed by fleet_launch.sh.
declare -A DIRS=(
  [c1]="/home/ubuntu/gen_out/<c1-claim>"
  [c2]="/home/ubuntu/gen_out/<c2-claim>"
  [c3]="/home/ubuntu/gen_out/<c3-claim>"
  [c5]="/home/ubuntu/gen_out/<c5-claim>"
  [c6]="/home/ubuntu/gen_out/<c6-claim>"
)
~~~

The operator machine needs Bash 4+, timeout, SSH, rsync, and the private key.
Remote commands use the ubuntu account.

## 9. Seed-ledger transaction

The launcher appends to each node's local ledger. It does not hold a shared
cross-host lock. One global operator must allocate every range.

The production-ledger participant set is c1, c2, c3, c5, and c6. c4 does not
generate or claim game seeds. Freeze all production allocation and launches
from the first pull until every redistributed hash matches. Before a wave:

1. Pull the ledger from every production generation node.
2. Merge the copies.
3. Inspect every reported overlap.
4. Atomically distribute one byte-identical canonical ledger to those five nodes.
5. Allocate all ranges from one next-safe value.

Fail-closed pull, merge, next-safe calculation, and distribution:

~~~bash
set -euo pipefail
cd <verified-local-checkout>
export FLEET_CONF="$HOME/.catan_fleet.conf"
export LOCAL_PY="$PWD/.venv/bin/python"
source tools/fleet/fleet_lib.sh

GEN_ALIASES=(c1 c2 c3 c5 c6)
PRIVATE_STATE=<private-state>
LEDGER_REMOTE=/home/ubuntu/catan-zero-v1/runs/SEED_LEDGER.md
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
COPY_DIR="$PRIVATE_STATE/ledger-copies/$STAMP"
CANON="$PRIVATE_STATE/SEED_LEDGER.md"
mkdir -p "$COPY_DIR"
[ ! -f "$CANON" ] || cp "$CANON" "$COPY_DIR/operator-previous.md"

for alias in "${GEN_ALIASES[@]}"; do
  scp -i "$(fleet_key)" \
    ubuntu@"$(fleet_host "$alias")":"$LEDGER_REMOTE" \
    "$COPY_DIR/$alias.md"
done

$LOCAL_PY tools/sync_seed_ledger.py "$COPY_DIR"/*.md -o "$CANON"
$LOCAL_PY tools/sync_seed_ledger.py "$CANON" --check

$LOCAL_PY - "$CANON" "$COPY_DIR/operator-previous.md" <<'PY'
import sys
from pathlib import Path
from tools import sync_seed_ledger

new_rows, _ = sync_seed_ledger.sync_ledgers([Path(sys.argv[1])])
new_overlaps = set(sync_seed_ledger._overlaps(new_rows))
previous = Path(sys.argv[2])
old_overlaps = set()
if previous.exists():
    old_rows, _ = sync_seed_ledger.sync_ledgers([previous])
    old_overlaps = set(sync_seed_ledger._overlaps(old_rows))
introduced = sorted(new_overlaps - old_overlaps)
if introduced:
    print("new overlapping production claims; refusing distribution:", file=sys.stderr)
    print("\n".join(introduced), file=sys.stderr)
    raise SystemExit(1)
PY

B=$($LOCAL_PY - "$CANON" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "tools"))
import prelaunch_guard

rows = prelaunch_guard.parse_seed_ledger(Path(sys.argv[1]))
if not rows:
    raise SystemExit("production ledger has no claims; require an owner-approved initial base")
lo, hi = prelaunch_guard.VAL_ONLY_SEED_RANGE
for start, end, label in rows:
    if end == prelaunch_guard._LEDGER_OPEN_END_SENTINEL:
        raise SystemExit(f"open-ended claim blocks allocation: {label}")
    if start < hi and end > lo:
        raise SystemExit(f"production ledger contains VAL-ONLY overlap [{start},{end}): {label}")
wave_seeds = 30_000
next_safe = max(end for _, end, _ in rows)
if next_safe < hi and next_safe + wave_seeds > lo:
    next_safe = hi
print(next_safe)
PY
)
echo "authoritative next-safe seed: $B"

CANON_SHA=$(shasum -a 256 "$CANON" | awk '{print $1}')
for alias in "${GEN_ALIASES[@]}"; do
  host=$(fleet_host "$alias")
  tmp="$LEDGER_REMOTE.incoming.$STAMP"
  scp -i "$(fleet_key)" "$CANON" ubuntu@"$host":"$tmp"
  ssh -i "$(fleet_key)" ubuntu@"$host" \
    "chmod 0644 $tmp && mv -f -- $tmp $LEDGER_REMOTE"
  remote_sha=$(ssh -i "$(fleet_key)" ubuntu@"$host" \
    "sha256sum $LEDGER_REMOTE" | awk '{print $1}')
  [ "$remote_sha" = "$CANON_SHA" ] || {
    echo "ledger hash mismatch on $alias" >&2
    exit 1
  }
done
~~~

sync_seed_ledger reports overlaps between different claim IDs but does not fail
for them. Treat any new overlap as fatal. Quarantine the affected data.

With 1,500 games/GPU and four GPUs, each generation box consumes 6,000 seeds.
For next-safe seed B, the standard layout is:

| Box | Role | Base | Half-open range |
|---|---|---:|---|
| c1 | volume | B | [B, B+6000) |
| c2 | teacher | B+6000 | [B+6000, B+12000) |
| c3 | teacher | B+12000 | [B+12000, B+18000) |
| c5 | volume | B+18000 | [B+18000, B+24000) |
| c6 | teacher | B+24000 | [B+24000, B+30000) |

c4 remains the training/control node in this layout.

After each --go attempt, pull, merge, inspect, and redistribute the updated
ledger before another operator can allocate seeds. A claim remains consumed
after a guard failure, crash, or partial wave. Never delete it or reuse its base.

## 10. Launch a production wave

Set local controls:

~~~bash
cd <verified-local-checkout>
test -x .venv/bin/python
test -f tools/fleet/fleet_launch.sh

export FLEET_CONF="$HOME/.catan_fleet.conf"
export REMOTE_PY=/home/ubuntu/catan-zero-v1/.venv/bin/python
export LOCAL_PY="$PWD/.venv/bin/python"
export CKPT=/home/ubuntu/bundle/champion_v0.pt
export LEDGER=/home/ubuntu/catan-zero-v1/runs/SEED_LEDGER.md

tools/fleet/fleet_status.sh all
tools/fleet/fleet_stop.sh <alias>
~~~

fleet_stop.sh is a dry run without --go.

Run each launch once without --go. Inspect the plan. Repeat the same command
with --go, one box at a time:

~~~bash
B=<authoritative-next-safe-seed>

PY="$REMOTE_PY" GEN_PY="$REMOTE_PY" \
  tools/fleet/fleet_launch.sh c1 volume \
  --base-seed "$B" --gpus 0-3 --games 1500 --wave <wave>

PY="$REMOTE_PY" GEN_PY="$REMOTE_PY" \
  tools/fleet/fleet_launch.sh c2 teacher \
  --base-seed "$((B + 6000))" --gpus 0-3 --games 1500 --wave <wave>

PY="$REMOTE_PY" GEN_PY="$REMOTE_PY" \
  tools/fleet/fleet_launch.sh c3 teacher \
  --base-seed "$((B + 12000))" --gpus 0-3 --games 1500 --wave <wave>

PY="$REMOTE_PY" GEN_PY="$REMOTE_PY" \
  tools/fleet/fleet_launch.sh c5 volume \
  --base-seed "$((B + 18000))" --gpus 0-3 --games 1500 --wave <wave>

PY="$REMOTE_PY" GEN_PY="$REMOTE_PY" \
  tools/fleet/fleet_launch.sh c6 teacher \
  --base-seed "$((B + 24000))" --gpus 0-3 --games 1500 --wave <wave>
~~~

Add --go only after each dry run passes.

The default GPU selection is 0-3. An eight-GPU canary requires --gpus 0-7.
Do not pass --mps.

The generator guards run inside the detached per-GPU children. A zero launcher
exit does not prove that every child survived. Record the printed claim ID and
OUT path, then inspect status, heartbeat, and every GPU log.

## 11. Monitor and stop

After each launch:

~~~bash
tools/fleet/fleet_status.sh <alias>
tools/fleet/fleet_stop.sh <alias>
~~~

The second command stays in dry-run mode and should identify one intended
detached Catan group.

Inspect a claim on its node:

~~~bash
source tools/fleet/fleet_lib.sh
ALIAS=<alias>
CLAIM=<claim-id>

ssh -i "$(fleet_key)" ubuntu@"$(fleet_host "$ALIAS")" \
  "cd /home/ubuntu/catan-zero-v1 && \
   source tools/fleet/launch_detached.sh && \
   heartbeat_status /home/ubuntu/fleet_runs/$CLAIM 60 && \
   tail -40 /home/ubuntu/gen_out/$CLAIM/gpu0/run.log"
~~~

Inspect all gpu0 through gpu3 logs, not only gpu0.

Normal stop:

~~~bash
tools/fleet/fleet_stop.sh <alias>
tools/fleet/fleet_stop.sh <alias> --go
tools/fleet/fleet_status.sh <alias>
~~~

Do not use pkill -f. If canonical stop fails, do not relaunch. Resolve the exact
PGID/PIDs that it reports, rerun canonical stop, and verify idle memory.

## 12. Accept generation before harvest

Each GPU output must contain:

~~~text
~/gen_out/<claim>/gpuN/
  run.log
  manifest.json
  worker_000/
    manifest.json
    progress.json
    gumbel_self_play_shard_00000.npz
~~~

For every gpuN manifest, require:

- games_completed equals games_requested;
- games_failed equals 0;
- errors is empty;
- rows is greater than 0;
- every listed shard exists;
- track is 2p_no_trade and vps_to_win is 10;
- cli_args.ledger_claim_label equals the recorded claim ID;
- base_seed and the distinct shard game_seed values exactly cover this GPU's
  allocated half-open range;
- a fresh SHA-256 of the checkpoint path still equals the champion SHA-256
  recorded before launch;
- role settings match Section 6;
- public observation, Rust features, corrected chance spectra, and lazy chance
  are enabled;
- precision is highest, cache is 0, and local fallback is off.

The generator can exit zero when some workers fail. Manifest reconciliation is
binding.

Run the independent Gumbel audit from the checkout on each generation node,
once for every gpuN root:

~~~bash
cd /home/ubuntu/catan-zero-v1
export REMOTE_PY=/home/ubuntu/catan-zero-v1/.venv/bin/python
GPU_BASE=<allocated-base-for-this-gpu>
GAMES=1500  # use 384 for the real-champion capacity canary
GPU_END=$((GPU_BASE + GAMES))
EXPECTED_CONFIG='{"track":"2p_no_trade","vps_to_win":10,"c_visit":50.0,"c_scale":0.03,"n_full":128,"n_fast":16,"p_full":1.0,"max_decisions":600,"max_depth":80,"shard_size":512,"format":"npz","public_observation":true,"rust_featurize":true,"correct_rust_chance_spectra":true,"lazy_interior_chance":true,"eval_server":true,"eval_server_max_batch":96,"eval_server_max_wait_ms":0.0,"eval_server_request_collector":true,"eval_server_local_fallback":false,"eval_server_matmul_precision":"highest","eval_cache_size":0}'

$REMOTE_PY tools/audit_gumbel_pilot_shards.py \
  --shards-dir /home/ubuntu/gen_out/<claim>/gpuN \
  --vps-to-win 10 \
  --p-full 1.0 \
  --colors RED,BLUE \
  --expected-config "$EXPECTED_CONFIG" \
  --expected-seed-range "$GPU_BASE,$GPU_END" \
  --out /home/ubuntu/gen_out/<claim>/gpuN/audit.json
~~~

For volume data, change n_full to 64, p_full to 0.25, shard_size to 2048,
eval_server_max_batch to 64, and eval_server_request_collector to false. Keep
the remaining expected fields binding.

Set truncation and diversity acceptance thresholds before generation. The
Gumbel audit defaults to a 0.40 truncation threshold. The synthetic smoke hit
14/16 truncated games, so it cannot define a production threshold.

The generic teacher quality report is an optional bounded diagnostic. Run it
on one worker leaf at a time on the source node, never on a full production
harvest:

~~~bash
$REMOTE_PY tools/report_teacher_data_quality.py \
  --data /home/ubuntu/gen_out/<claim>/gpuN/worker_NNN \
  --track 2p_no_trade \
  --vps-to-win 10 \
  --out /home/ubuntu/gen_out/<claim>/gpuN/worker_NNN/quality.json
~~~

Its strict and production 35M profiles target an older AB/JSettlers mixture.
Do not apply those profiles to Gumbel data without defining a compatible
Gumbel policy. It materializes NPZ rows in memory and is not a scalable
aggregate gate.

## 13. Harvest and reconcile

Populate DIRS from the accepted claim paths. Keep the harvest directory fresh:

~~~bash
export FLEET_CONF="$HOME/.catan_fleet.conf"
export HARV_DIR="$HOME/harvest/<wave>"

VOLUME_BOXES="c1 c5" \
TEACHER_BOXES="c2 c3 c6" \
tools/wave1_harvest.sh harvest-all
~~~

The hardened helper preserves NPZ shards plus JSON manifests, progress, audit
reports, and logs. Each box is pulled into a fresh hidden staging tree, every
configured DIRS root must yield at least one NPZ, and the prior published box
tree is replaced only after all of its sources pass. Any rsync, empty source,
or publish failure returns nonzero.

Before corpus build:

1. Compare harvested shard counts with the sum in accepted remote manifests.
2. Confirm every expected box, claim, GPU, and worker appears.
3. Confirm no canary path appears.
4. Compare every harvested worker leaf with its already accepted remote worker
   manifest; rsync preserves remote absolute paths inside JSON.
5. Preserve teacher and volume roots as separate declared sources.
6. Save the harvest inventory, source-list, and QA report hashes.

Do not run the Gumbel audit against the relocated harvest; it was intentionally
run before transfer where absolute manifest paths still resolve. For memmap
conversion, use explicit harvested `worker_NNN` leaf directories. The builder's
absolute-path fallback resolves each shard basename inside that leaf.

## 14. Build and accept the memmap corpus

The active H100 experiment compares role-pure n128 teacher and n64 volume data
engines at equal GPU-hours. Build and train those corpora separately first:

~~~bash
export TEACHER_CORPUS_DIR="$HOME/corpora/<wave>-teacher-<inventory-hash>"
export VOLUME_CORPUS_DIR="$HOME/corpora/<wave>-volume-<inventory-hash>"

tools/wave1_harvest.sh build-teacher
tools/wave1_harvest.sh build-volume
~~~

The helper enumerates every harvested worker manifest, writes one source per
line under `$HARV_DIR/source_lists/`, and calls the builder with
`--source-list`, avoiding shell argument limits. Hash and retain that list.

Do not build a pooled corpus unless the owner-approved wave record declares the
teacher/volume mixture and comparison. If pooling is approved, set a fresh
`POOLED_CORPUS_DIR` and run `build-pooled`; the builder concatenates rows and
does not balance roles or store a per-row role identity.

Volume or approved pooled corpora keep fast rows as value-only rows. Do not pass
`--full-rows-only` unless the experiment is a declared teacher-only ablation.

Never pass:

~~~text
--no-abort-on-duplicate-seeds
--no-verify-fill
~~~

If the builder aborts, quarantine or delete the partial output directory and
restart with a fresh path.

Require corpus_meta.json to report:

- schema memmap_corpus_v1;
- row_count greater than 0;
- the exact declared sources and shard count;
- game_seed_present true;
- verify_fill true;
- stats.has_duplicate_game_seeds is false;
- stats.duplicate_game_seed_count equals 0;
- stats.has_duplicate_legal_rows is false (no repeated legal action ID within a
  row; this is not a duplicate-corpus-row check);
- full_rows_only is false for volume or an approved pooled corpus.

Probe the stored player tokens:

~~~bash
$LOCAL_PY tools/probe_corpus_masking.py <fresh-corpus-dir>
~~~

The last line must be PROVENANCE=masked. Exit 0 also covers the omniscient
verdict, so parse the token. Treat ambiguous as a hard stop.

Run and retain diversity reports per original gpuN root before harvest, or on
another bounded tranche whose memory use was predeclared:

~~~bash
$REMOTE_PY tools/corpus_diversity_scan.py \
  --shards-dir /home/ubuntu/gen_out/<claim>/gpuN \
  --generation-label <wave> \
  --out /home/ubuntu/gen_out/<claim>/gpuN/diversity.json
~~~

The diversity scan has no binding threshold and can encode an error in JSON
while returning zero. Inspect the report. It materializes rows and builds Python
sets, so do not point it at the full 24-H100 harvest. Aggregate the bounded
report metrics in the wave record; scalable aggregate diversity QA remains a
known gap.

There is no single production Gumbel QA command behind
--trust-curated-data. The RL operator must sign the manifest/QA checklist in the
wave record before training.

## 15. Train the canonical 35M control

Stage the accepted corpus on c4 under an immutable path:

~~~text
/home/ubuntu/corpora/<wave>-<corpus-hash>
~~~

Use the project-approved private transport, then compare the complete corpus
inventory and cryptographic digest on the operator host and c4. Do not train
from a mutable synchronization target or a partially copied directory.

Dry run:

~~~bash
PY="$REMOTE_PY" tools/fleet/fleet_launch.sh c4 train \
  --gpus 0-3 \
  --data /home/ubuntu/corpora/<wave>-<corpus-hash> \
  --grow-from /home/ubuntu/bundle/champion_v0.pt \
  --trust-curated-data \
  --wave <wave>-bc
~~~

Repeat with --go after the plan passes.

The launcher pins an L6/h640 entity graph, eight heads, dropout 0.05, BF16,
fused Adam, LR 3e-5, 100 warmup steps, a flat schedule, one epoch, and exact
global batch 4096.

The trust flag asserts that external QA passed. It expands to trainer flags that
skip the expensive teacher-quality gate and trusted diagnostics. Training
success does not attest the corpus.

Accept the training output only when:

- parameter_count is 35,041,353;
- mask_hidden_info is true;
- world_size equals 4;
- rank batch times world size equals 4096;
- steps_completed is greater than 0;
- the game-grouped validation split is nonempty;
- train and validation metrics are finite;
- checkpoint metadata records the masked/public regime;
- model.pt, model.pt.optimizer.pt, report.json, and run.log exist;
- fleet_status shows c4 idle after completion.

The launcher prints the training claim and writes the result below
`/home/ubuntu/train_out/<claim-id>`. Preserve that claim ID in the wave record.

Keep the incumbent checkpoint untouched.

## 16. Evaluate the candidate

Run evaluation on the two-B200 `b200` hub, not on the operator workstation.
Stage the candidate and incumbent under immutable hash-qualified paths, verify
both hashes after transfer, log in through the alias in `$FLEET_CONF`, and set:

~~~bash
cd /home/ubuntu/catan-zero-v1
export EVAL_PY=/home/ubuntu/catan-zero-v1/.venv/bin/python
nvidia-smi --query-gpu=index,memory.used --format=csv
~~~

### 16.1 Claim validation seeds

H2H tools do not claim or guard seeds. Under one evaluation operator, use a
private `VAL_ONLY_EVAL_LEDGER.md` that never enters a training corpus. Every
pair consumes one seed and plays both color orientations. Claim the maximum
extension before starting: 300 seeds for a 150-to-300-pair flywheel gate, 100
separate seeds for an every-third n64 confirmation, and a separate panel range.
Candidate and incumbent bot panels intentionally share one claimed panel range
so their comparison uses identical seeds.

Before appending a claim, call `guard_ledger_overlap` against that explicit
ledger and fail on a collision:

~~~bash
set -euo pipefail
EVAL_LEDGER=<private-state>/VAL_ONLY_EVAL_LEDGER.md
EVAL_SEED=<fresh-val-only-base>
MAX_PAIRS=300
EVAL_CLAIM=<unique-promotion-id>
test -s "$EVAL_LEDGER"

$EVAL_PY - "$EVAL_SEED" "$MAX_PAIRS" "$EVAL_LEDGER" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "tools"))
from prelaunch_guard import VAL_ONLY_SEED_RANGE, guard_ledger_overlap

base, pairs, ledger = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
lo, hi = VAL_ONLY_SEED_RANGE
assert lo <= base and base + pairs <= hi
result = guard_ledger_overlap(base, pairs, ledger_path=ledger, purpose="eval")
print(f"[{result.status}] {result.reason}")
raise SystemExit(0 if result.passed else 1)
PY

printf '[%s – %s) | eval/b200 | flywheel claim=%s | %s\n' \
  "$EVAL_SEED" "$((EVAL_SEED + MAX_PAIRS))" "$EVAL_CLAIM" \
  "$(date -u +%Y-%m-%d)" >> "$EVAL_LEDGER"
tmp="$EVAL_LEDGER.incoming.$$"
$EVAL_PY tools/sync_seed_ledger.py "$EVAL_LEDGER" -o "$tmp"
mv -f -- "$tmp" "$EVAL_LEDGER"
$EVAL_PY tools/sync_seed_ledger.py "$EVAL_LEDGER" --check
~~~

`sync_seed_ledger --check` must pass. If the append makes it noncanonical,
canonicalize through a temporary output and atomically replace the ledger
before launching. The 300-pair extension intentionally replays the first 150
pairs; it is covered by the one maximum-range claim.

### 16.2 Run the roadmap flywheel gate

The roadmap's ordinary promotion valve is n16, 150 pairs/300 games, with
elo0=-10, elo1=+15, alpha=beta=0.05. n128 is a strength panel, not this gate.
Launch the gate through the teardown-safe detached runner:

~~~bash
set -euo pipefail
EVAL_ID="${EVAL_CLAIM}-gate300"
EVAL_OUT="/home/ubuntu/eval_out/$EVAL_ID"
RUNDIR="/home/ubuntu/fleet_runs/$EVAL_ID"
mkdir -p /home/ubuntu/eval_out
mkdir "$EVAL_OUT"
CANDIDATE=/immutable/eval/<candidate-hash>.pt
INCUMBENT=/immutable/eval/<incumbent-hash>.pt
source tools/fleet/launch_detached.sh
export PROGRESS_CMD="tail -1 $EVAL_OUT/run.log"

EVAL_CMD=(
  "$EVAL_PY" tools/gumbel_search_cross_net_h2h.py
  --candidate "$CANDIDATE"
  --baseline "$INCUMBENT"
  --pairs 150 \
  --workers 8 \
  --devices cuda:0,cuda:1 \
  --n-full 16 \
  --max-depth 80 \
  --max-decisions 600 \
  --prior-temperature 1.0 \
  --value-scale 1.0 \
  --value-squash tanh \
  --c-visit 50.0 \
  --c-scale 0.03 \
  --max-root-candidates 16 \
  --max-root-candidates-wide 54 \
  --correct-rust-chance-spectra \
  --lazy-interior-chance \
  --public-observation \
  --no-belief-chance-spectra \
  --no-symmetry-averaged-eval \
  --gate-config flywheel \
  --base-seed "$EVAL_SEED" \
  --dump-config "$EVAL_OUT/config.json" \
  --config-hash \
  --config-purpose flywheel-promotion-300 \
  --out "$EVAL_OUT/result.json"
)

PID=$(launch_detached "$RUNDIR" "$EVAL_OUT/run.log" 60 -- "${EVAL_CMD[@]}")
echo "eval pid=$PID"
~~~

On b200, monitor with `heartbeat_status "$RUNDIR" 60`. From the operator
workstation, run `tools/fleet/fleet_status.sh b200` and a dry-run
`tools/fleet/fleet_stop.sh b200`. Canonical stop recognizes the detached H2H
group. After `DONE`, retrieve the report/config/log, recheck checkpoint hashes,
and verify b200 is idle.

Read `pentanomial_sprt.decision`, not the naive SPRT:

- H1: eligible for the next panel, subject to vetoes;
- H0: reject;
- continue: hold and extend.

The tool does not extend itself. On `continue`, create a new detached run with
the same base seed and flags, changing only pairs to 300, output/config paths,
and config-purpose to `flywheel-promotion-600`. This replays all 600 games; it
does not append only the second half. Do not promote on `continue`.

Reject an output with errors, missing games, mismatched production flags, or
insufficient complete pairs.

### 16.3 Run strength panels and veto checks

Run candidate and incumbent separately against `catanatron_value` with the same
separately claimed 500-pair seed range and the same full flags. Use n128 here.
Launch each command through the detached pattern above, one after the other:

~~~bash
set -euo pipefail
PANEL_SEED=<fixed-panel-seed>
PANEL_PAIRS=500
PANEL_ID="${EVAL_CLAIM}-candidate-vs-value"
PANEL_OUT="/home/ubuntu/eval_out/$PANEL_ID"
PANEL_RUNDIR="/home/ubuntu/fleet_runs/$PANEL_ID"
mkdir -p /home/ubuntu/eval_out
mkdir "$PANEL_OUT"

PANEL_CMD=(
  "$EVAL_PY" tools/gumbel_search_vs_bot_h2h.py
  --candidate "$CANDIDATE" \
  --baseline-bot catanatron_value \
  --pairs "$PANEL_PAIRS" \
  --workers 8 \
  --devices cuda:0,cuda:1 \
  --n-full 128 \
  --max-depth 80 \
  --max-decisions 600 \
  --prior-temperature 1.0 \
  --value-scale 1.0 \
  --value-squash tanh \
  --c-visit 50.0 \
  --c-scale 0.03 \
  --max-root-candidates 16 \
  --max-root-candidates-wide 54 \
  --correct-rust-chance-spectra \
  --lazy-interior-chance \
  --public-observation \
  --no-belief-chance-spectra \
  --no-symmetry-averaged-eval \
  --base-seed "$PANEL_SEED" \
  --gate-config flywheel \
  --out "$PANEL_OUT/result.json"
)

PID=$(launch_detached "$PANEL_RUNDIR" "$PANEL_OUT/run.log" 60 -- "${PANEL_CMD[@]}")
echo "panel pid=$PID"
~~~

Repeat with only candidate and output changed to the incumbent. Reject either
report with errors, engine divergence, incomplete pairs, or a post-hoc panel
threshold. `catanatron_ab3` through `ab5` are optional additional panels. This
lockstep tool uses a fixed tournament board; do not compare its absolute win
rate with randomized-map cross-net H2H.

The roadmap also requires external tripwires, population/WHR evidence, and an
every-third 200-game n64 non-regression plus phase/opening/blowout bucket
vetoes. Current code does not provide a searched native-Catanatron production
panel or a bucket extractor with approved definitions. `population_arena.py`
generates matches with incompatible defaults and unledgered seeds. Therefore
promotion remains blocked unless those integrations land or the owner records
an explicit substitute, thresholds, and veto policy before evaluation.

Do not use promotion_gate_runner.py for the masked searched agent. It evaluates
raw policies through a deprecated unmasked path. The native Catanatron harness
is also raw-policy smoke tooling, not a searched 1,000-game strength panel.

## 17. Promotion and deployment transaction

Do not create a registry at a new path: `ChampionRegistry.load()` treats a
missing file as an empty registry and would reset history and the every-third
counter. Promotion requires one hash-verified, nonempty authoritative registry,
one writer, and checkpoints accessible on that registry host.

Before mutation, save `champion_registry show`, the registry hash, a byte-for-
byte backup, the incumbent role/version/hash, and the next promotion number:

~~~bash
set -euo pipefail
PROMOTION_ID=<safe-promotion-id>
REGISTRY=<private-authoritative-registry.json>
PROMOTION_RECORD=<private-promotion-record-dir>
mkdir -p "$(dirname "$PROMOTION_RECORD")"
mkdir "$PROMOTION_RECORD"
test -s "$REGISTRY"
BACKUP="$REGISTRY.before-$PROMOTION_ID"
[ ! -e "$BACKUP" ]
cp "$REGISTRY" "$BACKUP"
$EVAL_PY -m tools.champion_registry --registry "$REGISTRY" show \
  > "$PROMOTION_RECORD/registry-before.json"

NEXT=$($EVAL_PY - "$REGISTRY" <<'PY'
import sys
from tools.champion_registry import ChampionRegistry
registry = ChampionRegistry.load(sys.argv[1])
assert registry.get_role("generator_champion") is not None
print(registry.promotion_count("generator_champion") + 1)
PY
)
echo "next promotion number: $NEXT"
~~~

If `NEXT % 3 == 0`, run a separately ledgered 100-pair/200-game n64 searched
confirmation. Promotion remains blocked until every required bucket has enough
data and the approved bucket-veto procedure passes; the current library hook
cannot extract those buckets itself.

Only after the ordinary gate, panels, tripwires, population evidence, and any
nth confirmation all pass may one operator run this registry transaction. Add
the dethroned incumbent to the append-only pool, not the new candidate:

~~~bash
set -euo pipefail
: "${EVAL_PY:?set evaluation interpreter}"
: "${REGISTRY:?set authoritative registry path}"
: "${BACKUP:?set pre-transaction registry backup}"
: "${NEXT:?compute next promotion number first}"
: "${CANDIDATE:?set immutable candidate path}"
: "${INCUMBENT:?set immutable incumbent path}"
: "${INCUMBENT_VERSION:?set incumbent version}"
: "${CANDIDATE_VERSION:?set candidate version}"
: "${PROMOTION_ID:?set promotion id}"
: "${PROMOTION_REASON:?set promotion reason}"
: "${GATE_REPORT:?set accepted gate path}"
: "${PANEL_REPORT:?set accepted panel path}"
: "${WHR_REPORT:?set accepted WHR path}"
: "${POPULATION_REPORT:?set accepted population path}"
test -s "$REGISTRY"
test -s "$BACKUP"
test -f "$CANDIDATE"
test -f "$INCUMBENT"
for report in "$GATE_REPORT" "$PANEL_REPORT" "$WHR_REPORT" "$POPULATION_REPORT"; do
  test -s "$report"
done
INCUMBENT_MD5=$(md5sum "$INCUMBENT" | awk '{print $1}')
CANDIDATE_MD5=$(md5sum "$CANDIDATE" | awk '{print $1}')
restore_registry() { cp "$BACKUP" "$REGISTRY"; }
trap restore_registry ERR

$EVAL_PY -m tools.champion_registry \
  --registry "$REGISTRY" append-pool \
  --checkpoint "$INCUMBENT" \
  --expected-md5 "$INCUMBENT_MD5" \
  --version "$INCUMBENT_VERSION" \
  --status active \
  --provenance "{\"reason\":\"dethroned_generator\",\"promotion\":\"$PROMOTION_ID\"}" \
  --reason "dethroned generator champion"

$EVAL_PY -m tools.champion_registry \
  --registry "$REGISTRY" set-role \
  --role generator_champion \
  --checkpoint "$CANDIDATE" \
  --expected-md5 "$CANDIDATE_MD5" \
  --version "$CANDIDATE_VERSION" \
  --provenance "{\"gate\":\"$GATE_REPORT\",\"panel\":\"$PANEL_REPORT\",\"whr\":\"$WHR_REPORT\",\"population\":\"$POPULATION_REPORT\"}" \
  --reason "$PROMOTION_REASON"

$EVAL_PY -m tools.champion_registry \
  --registry "$REGISTRY" record-promotion \
  --role generator_champion

FINAL_COUNT=$($EVAL_PY - "$REGISTRY" <<'PY'
import sys
from tools.champion_registry import ChampionRegistry
print(ChampionRegistry.load(sys.argv[1]).promotion_count("generator_champion"))
PY
)
[ "$FINAL_COUNT" -eq "$NEXT" ]
trap - ERR
~~~

These three atomic writes are not one locked transaction. If any command or
post-check fails, restore the backup registry before another writer proceeds.
Verify the final count equals NEXT and hash the resulting registry.

The registry does not configure fleet_launch.sh. Stage the promoted checkpoint
under an immutable, hash-verified path on every generation node and set CKPT to
that exact remote path for future launches. Verify all remote hashes before
unfreezing allocation. Keep the incumbent path and registry backup for rollback.

`runs/CURRENT_CHAMPION` only feeds an obsolete launcher and is not the H100
deployment source. Do not run `auto_refill.sh`. Update and restart a legacy feed
daemon only if the wave record explicitly says that loop remains active.

Do not change public_champion after a flywheel gate. Public promotion requires
the certification budget and external-panel policy chosen by the project owner.

## 18. Recovery rules

- Guard or startup failure after claim: consume the range and inspect all logs.
- Partial generation: quarantine the output; restart under a fresh claim.
- Status reports an unintended group: use canonical stop before any relaunch.
- Heartbeat stalled: inspect process and GPU state, then canonical stop.
- Harvest count mismatch: stop corpus work and re-pull the missing claim.
- Gumbel audit failure: quarantine the source and diagnose before regeneration.
- Memmap abort: discard the partial output path.
- Training failure: preserve logs and candidate artifacts; do not register them.
- H2H continue: extend the predeclared sample, do not promote.
- H2H H0 or panel veto: keep the candidate as an experiment or opponent-pool
  entry only when policy permits.
- Registry/deployment mismatch: freeze launches, restore the pre-transaction
  registry backup and incumbent CKPT path, verify both hashes, then re-audit
  every consumer before unfreezing.

## 19. Wave evidence record

Create one immutable record per wave with:

~~~text
wave name and owner
release tag and commit
Rust wheel SHA-256
champion path, SHA-256, and no-op result
canonical seed-ledger SHA-256
real-champion capacity claim/range, manifests, per-GPU rates, threshold verdict, and quarantine path
per-box claim ID, role, GPUs, seed range, and output path
per-GPU manifest totals and config hash
predeclared truncation/diversity/Gumbel-QA thresholds and their pass/fail decisions
Gumbel audit and bounded diversity report paths/hashes
harvest inventory and reconciled shard count
role-pure source-list path/hash and selected data-engine policy
memmap corpus_meta path/hash and corpus file inventory
training claim, model hash, optimizer hash, and report
validation-ledger hash and every H2H/panel claim
cross-net H2H config hash, seed, budget, and verdict
neutral/external panel thresholds, seeds, results, and veto decisions
every-third confirmation/bucket evidence when required
WHR/population evidence and owner-approved promotion-policy record
registry before/after hashes, transition, and deployed CKPT path
rollback registry and checkpoint paths
~~~

Do not rely on terminal scrollback as the only record.

## 20. Wave-1 H100 low-level flags

The generation row writer now stores public-observation-masked entity and
action-context tensors when public generation is requested. This closes the
old mismatch where online MCTS was masked but NPZ `player_tokens` were
omniscient at rest. Keep `--public-observation`; do not compensate by relying
only on trainer-side masking.

Use these defaults for the real-champion capacity repeat:

~~~text
--eval-server-transport mp_queue
--eval-server-event-token-limit 0
--no-root-wave-batching
--no-eval-server-cuda-graph
--eval-server-matmul-precision highest
~~~

`event-token-limit 0` is the one new recommended canary flag: 2,048 live Rust
states and 30,720 retained rows had no active events, the server fails closed
if that changes, strict-FP32 leaf parity was within `8.03e-7`, and two H100
pairs improved rows/hour by 21.6--23.9%. Re-run the active-event audit and leaf
parity against the private champion before expanding past the canary.

The four-GPU-shaped, two-games-per-worker opening canary measured 243.9k
rows/hour at 128 workers/GPU versus 234.8k at 96, so the canonical teacher
launcher correctly keeps 128. These are synthetic-checkpoint short-harness
rates, not the real-champion capacity threshold.

Keep root-wave batching off for the initial fleet run even though the H100
canary improved throughput by roughly 33%. It changes RNG stream assignment and
still needs a real-champion target audit plus powered H2H non-inferiority.
Shared-memory request transport was 8--9% slower. CUDA Graphs were only
+0.27/+0.58% end to end in GPU-crossover pairs while increasing compute/memory
pressure. Both remain diagnostic-only.

After shard acceptance, `tools/build_memmap_corpus.py --omit-zero-events` can
write the audited v2 corpus without the two all-zero event files. The builder
checks every source value before publishing metadata, and the trainer
synthesizes exact zero columns lazily. Default v1 remains available for
rollback. Do not omit `event_target_ids`.

## 21. Known integration gaps

The RL operator owns manual checks for these gaps until code integrates them:

1. Seed claims have no shared cross-host lock.
2. Dynamic generation guards run in detached children; launcher dry-run does not
   execute the complete remote overlap guard.
3. A generation command can exit zero after partial worker failure.
4. No one Gumbel-specific QA profile backs --trust-curated-data.
5. Masked search promotion has no safe one-command runner.
6. Scalable full-fleet diversity QA and a post-harvest provenance binder are
   missing; current audits run per source and memmap builds from worker leaves.
7. Evaluation tools have no automatic validation-ledger claim transaction.
8. Searched native-Catanatron external panels and every-third bucket extraction
   are not implemented.
9. Population-arena generated matches are not production-config or seed safe,
   and no approved population veto threshold consumes their reports.
10. ChampionRegistry has no lock and does not update fleet CKPT paths.
11. Rust licensing remains unresolved in FLEET.md.

Stop when one of these gaps makes the evidence ambiguous.

## 22. First message for the RL operator

Copy this task into the operating agent:

~~~text
You own the CatanZero 2p_no_trade RL fleet workflow. Read
RL_AGENT_HANDOFF.md, docs/plans/CATAN_ZERO_ROADMAP.md,
docs/plans/CATAN_ZERO_MASTER_PLAN.md, FLEET.md,
tools/fleet/FLEET_CONTROL.md, and
docs/plans/H100_EXECUTION_UPDATE_2026-07-09.md before changing state.

Start read-only. Report the release tag/commit, Rust wheel hash, champion
hash/no-op result, canonical production and validation ledger hashes/overlaps,
authoritative ChampionRegistry hash/roles/promotion count, fleet aliases,
current status, free disk, CUDA/Torch versions, and every unmet prerequisite.

Do not launch until the immutable release, real masked champion, canonical
ledger, private fleet config, CUDA acceptance, and predeclared corpus/gate
criteria all pass. Use dry-run commands first and launch one box at a time.
Record every claim and output path. Never harvest canary data, reuse seeds,
trust a zero exit without manifests, bypass duplicate/fill checks, train before
QA sign-off, pool role-pure controls without approval, or promote on an
inconclusive/raw-policy gate or unresolved roadmap veto.
~~~
