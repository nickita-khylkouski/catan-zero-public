#!/usr/bin/env bash
# tools/fleet/fleet_launch.sh — ONE canonical, guarded fleet launcher (CAT-122).
#
# Supersedes the drifted per-box launch scripts (fire_*.sh, mps_rollout.sh,
# fleet_launch_safe.sh stub) that the 2026-07-09 freeze indicted. Every launch —
# generation (teacher/volume) and training (train) — goes through this one path,
# so the guards, the seed-claim discipline, the interpreter resolution, and the
# teardown-safe detach are identical on every box and can never drift again.
#
# The launch order is CLAIM -> GUARD -> LAUNCH (CAT-124):
#   1. CLAIM   append this launch's own row to the cross-host seed ledger with a
#              UNIQUE `claim=<id>` token, and export CATAN_LEDGER_CLAIM_ID=<id>.
#   2. GUARD   run tools/prelaunch_guard.py WITH guards on. ledger_overlap sees our
#              own just-written row but excludes it (via the claim id) — a peer's
#              overlapping claim still fails closed. This is why --skip-guards is
#              RETIRED here: the self-collision that forced it no longer happens.
#   3. LAUNCH  source launch_detached.sh -> setsid detach + atomic heartbeat, so
#              the job survives SSH teardown (the exit-137 root cause) and
#              fleet_status.sh can read liveness.
#
# Host IPs come from the FLEET_CONF resolver (fleet_lib.sh) — keyed by ALIAS, no
# IPs in the repo. Interpreter is resolved (~/venv, else tree .venv), NEVER a bare
# `torchrun`/`python3` (that loads system numpy<2 and crashes champion load —
# CAT-128), NEVER a hardcoded .venv (that stranded a GPU — CAT-123).
#
# Usage:
#   fleet_launch.sh <alias> <role> --base-seed N [opts] [--go]
#     role = teacher | volume | train
#     teacher : n_full 128, p_full 1.0   (scarce high-sim generation)
#     volume  : n_full  64, p_full 0.25  (bulk generation)
#     train   : multi-GPU DDP train_bc via torch.distributed.run (CAT-128)
#   Options:
#     --base-seed N     REQUIRED for gen roles: fresh, ledgered base seed.
#     --gpus SPEC       GPUs to use: "0-3" | "0,1,2,3" | "4" (default all visible / 0-3).
#     --games N         games per gen worker (default 1500).
#     --workers N       gen workers per GPU (default 16, MPS-collapsed).
#     --wave W          wave tag baked into the claim id (default 1).
#     --data DIR        train: corpus dir (REQUIRED for role=train).
#     --grow-from CKPT  train: depth-grow warm-start ckpt (default champion).
#     --go              actually launch (default: DRY-RUN, prints the full plan).
#   Env overrides: TREE, CKPT (champion), LEDGER, GEN_PY/PY, FLEET_CONF, PROGRESS_CMD.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$DIR/fleet_lib.sh" || { echo "fleet_launch: cannot load fleet_lib.sh"; exit 1; }

# ---- parse ----------------------------------------------------------------
ALIAS="${1:?usage: fleet_launch.sh <alias> <role> [opts]}"; shift
ROLE="${1:?role = teacher|volume|train}"; shift
BASE_SEED=""; GPUS="0-3"; GAMES=1500; WORKERS=16; WAVE=1; DATA=""; GROW_FROM=""; GO=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --base-seed) BASE_SEED="$2"; shift 2;;
    --gpus)      GPUS="$2"; shift 2;;
    --games)     GAMES="$2"; shift 2;;
    --workers)   WORKERS="$2"; shift 2;;
    --wave)      WAVE="$2"; shift 2;;
    --data)      DATA="$2"; shift 2;;
    --grow-from) GROW_FROM="$2"; shift 2;;
    --go)        GO=1; shift;;
    --skip-guards) echo "fleet_launch: --skip-guards is RETIRED (CAT-124). Claim your seed range first; guards now pass for a legitimate fresh claim." >&2; exit 2;;
    *) echo "fleet_launch: unknown option '$1'" >&2; exit 2;;
  esac
done

case "$ROLE" in
  teacher) NFULL=128; PFULL=1.0;;
  volume)  NFULL=64;  PFULL=0.25;;
  train)   ;;
  *) echo "fleet_launch: role must be teacher|volume|train" >&2; exit 2;;
esac

IP=$(fleet_host "$ALIAS") || exit 2
KEY=$(fleet_key)
EPOCH=$(date -u +%s)
CLAIM_ID="${ALIAS}-${ROLE}-w${WAVE}-${EPOCH}"   # globally unique: alias+role+wave+epoch
DATE_UTC=$(date -u +%Y-%m-%d)

# Expand --gpus "0-3" | "0,1,2,3" into a comma list + a count.
expand_gpus() {
  local spec="$1"
  if [[ "$spec" == *-* ]]; then seq -s, "${spec%-*}" "${spec#*-}"; else echo "$spec"; fi
}
GPU_CSV=$(expand_gpus "$GPUS"); NGPU=$(awk -F, '{print NF}' <<<"$GPU_CSV")

# ---- role-specific validation --------------------------------------------
if [ "$ROLE" = "train" ]; then
  [ -n "$DATA" ] || { echo "fleet_launch: role=train needs --data <corpus dir>" >&2; exit 2; }
else
  [ -n "$BASE_SEED" ] || { echo "fleet_launch: gen role needs --base-seed <fresh ledgered seed>" >&2; exit 2; }
fi

echo "===== fleet_launch $ALIAS/$ROLE gpus=$GPU_CSV ($NGPU) claim=$CLAIM_ID ($([ $GO = 1 ] && echo GO || echo DRY-RUN)) ====="

# The remote script is a QUOTED heredoc (no local expansion); every value is passed
# positionally to `bash -s`. It runs on the target box where the tree, venv, ledger,
# and GPUs live. It is idempotent-safe to DRY-RUN (writes nothing unless GO=1).
# shellcheck disable=SC2016
read -r -d '' REMOTE <<'REMOTE_EOF' || true
set -uo pipefail
GO="$1"; ROLE="$2"; ALIAS="$3"; GPU_CSV="$4"; NGPU="$5"; NFULL="$6"; PFULL="$7"
GAMES="$8"; WORKERS="$9"; BASE_SEED="${10}"; CLAIM_ID="${11}"; DATE_UTC="${12}"
DATA="${13}"; GROW_FROM_IN="${14}"

TREE="${TREE:-$HOME/catan-zero-runsix}"
CKPT="${CKPT:-$HOME/bundle/champion_v0.pt}"
LEDGER="${LEDGER:-$TREE/runs/SEED_LEDGER.md}"
GROW_FROM="${GROW_FROM_IN:-$CKPT}"
STAMP=$(date -u +%Y%m%d_%H%M%S)
OUT="$HOME/gen_out/${ALIAS}_${ROLE}_w_${STAMP}"     # unique => fresh out-dir by construction
RUNDIR="$HOME/fleet_runs/${CLAIM_ID}"

# --- interpreter resolution (CAT-123/128): venv first, tree .venv fallback, NEVER bare python3
PY="${PY:-}"; GEN_PY="${GEN_PY:-}"
resolve_py() { if [ -x "$HOME/venv/bin/python" ]; then echo "$HOME/venv/bin/python"; elif [ -x "$TREE/.venv/bin/python" ]; then echo "$TREE/.venv/bin/python"; else echo ""; fi; }
[ -z "$PY" ] && PY=$(resolve_py); [ -z "$GEN_PY" ] && GEN_PY="$PY"

FAIL=0
[ -n "$PY" ] && [ -x "$PY" ] && echo "ok: interpreter $PY" || { echo "FAIL: no python (~/venv or \$TREE/.venv) — refusing (bare python3 loads numpy<2, CAT-128)"; FAIL=1; }
[ -d "$TREE" ] && echo "ok: tree $TREE" || { echo "FAIL: tree $TREE missing"; FAIL=1; }
[ -f "$CKPT" ] && echo "ok: champion $CKPT" || { echo "FAIL: champion $CKPT missing"; FAIL=1; }
[ -f "$LEDGER" ] && echo "ok: ledger $LEDGER" || { echo "FAIL: ledger $LEDGER missing (sync it here first — cross-host seed safety, CAT-125)"; FAIL=1; }
if [ "$ROLE" = "train" ]; then
  [ -d "$DATA" ] && echo "ok: corpus $DATA" || { echo "FAIL: --data $DATA missing"; FAIL=1; }
  [ -f "$GROW_FROM" ] && echo "ok: grow-from $GROW_FROM" || { echo "FAIL: --grow-from $GROW_FROM missing"; FAIL=1; }
fi

# --- build the pinned command per role -------------------------------------
if [ "$ROLE" = "train" ]; then
  # CAT-128: torch.distributed.run via the RESOLVED venv python (never bare torchrun);
  # --tee=3 streams rank stdout/err; depth-grow warm-start (h640/L13 preserves warm-start,
  # width-grow cold-starts). Pins the leak-critical + curated-memmap flags.
  CMD="cd $TREE && CUDA_VISIBLE_DEVICES=$GPU_CSV $PY -m torch.distributed.run --standalone --nproc_per_node=$NGPU --tee=3 \
tools/train_bc.py --data $DATA --checkpoint $OUT/model.pt --report $OUT/report.json \
--grow-from-checkpoint $GROW_FROM --graph-layers 13 --hidden-size 640 \
--mask-hidden-info --soft-target-source policy --skip-teacher-quality-gate --trust-curated-data-quality \
--optimizer adam --weight-decay 0.0 --lr-schedule flat"
  CADENCE=120
  PROG="grep -oE 'step=[0-9]+/[0-9]+|epoch [0-9]+' $OUT/run.log 2>/dev/null | tail -1"
else
  # generation: ONE generator PER PHYSICAL GPU. The generator is single-device
  # (all workers use --device cuda = the one visible GPU), so we pin each process
  # with CUDA_VISIBLE_DEVICES=$g and give each a DISJOINT GAMES-wide seed sub-block
  # [BASE+i*GAMES, BASE+(i+1)*GAMES). Exposing all GPUs to one process (the old
  # bug) ran everything on gpu0 while claiming NGPU*WORKERS*GAMES seeds. guards ON;
  # ledger_overlap excludes our own claim via --ledger-claim-label=$CLAIM_ID.
  CMD="cd $TREE && i=0; for g in \$(echo '$GPU_CSV' | tr ',' ' '); do \
CUDA_VISIBLE_DEVICES=\$g CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe_host \
$GEN_PY tools/generate_gumbel_selfplay_data.py --out-dir $OUT/gpu\$g --checkpoint $CKPT --device cuda \
--games $GAMES --workers $WORKERS --base-seed \$(( $BASE_SEED + i * $GAMES )) --shard-size 2048 \
--n-full $NFULL --n-fast 16 --p-full $PFULL --c-visit 50.0 --c-scale 0.03 \
--max-decisions 600 --max-depth 80 --temperature-decisions 90 \
--correct-rust-chance-spectra --lazy-interior-chance --public-observation \
--track 2p_no_trade --vps-to-win 10 --format npz --score-actions \
--ledger-claim-label $CLAIM_ID > $OUT/gpu\$g.log 2>&1 & \
i=\$(( i + 1 )); done; echo \"launched \$i per-GPU generators on GPUs $GPU_CSV\"; wait"
  CADENCE=60
  PROG="ls $OUT/gpu*/*.npz 2>/dev/null | wc -l"
fi

# --- CLAIM row (append BEFORE guard so ledger_overlap sees + excludes our own) ---
if [ "$ROLE" = "train" ]; then
  CLAIM_ROW="# (train role claims no seed range) claim=$CLAIM_ID $ALIAS train $DATE_UTC"
else
  # one generator per GPU, each consuming a disjoint GAMES-wide sub-block → NGPU*GAMES total.
  END=$(( BASE_SEED + GAMES * NGPU ))
  CLAIM_ROW="[$BASE_SEED – $END) | fleet/$ALIAS | $ROLE n$NFULL p$PFULL gpus=$GPU_CSV claim=$CLAIM_ID | $DATE_UTC"
fi

echo "CLAIM ROW  : $CLAIM_ROW"
echo "CLAIM ID   : $CLAIM_ID  (exported as CATAN_LEDGER_CLAIM_ID)"
echo "OUT        : $OUT"
echo "WOULD RUN  : $CMD"

[ "$FAIL" -ne 0 ] && { echo "REFUSING: precondition(s) failed."; exit 3; }
if [ "$GO" != "1" ]; then echo "DRY-RUN: preconditions passed; not launched (pass --go)."; exit 0; fi

# ===== GO path =====
mkdir -p "$OUT" "$RUNDIR"
# 1. CLAIM: append to the box-local ledger (operator runs sync_seed_ledger.py to reconcile).
if [ "$ROLE" != "train" ]; then printf '%s\n' "$CLAIM_ROW" >> "$LEDGER"; echo "claimed: appended to $LEDGER"; fi
export CATAN_LEDGER_CLAIM_ID="$CLAIM_ID"
export CATAN_SEED_LEDGER="$LEDGER"
# 2+3. GUARD + LAUNCH via the detach lib (guards run inside the tool; --tee/exit-check for train).
source "$TREE/tools/fleet/launch_detached.sh" 2>/dev/null || source "$HOME/launch_detached.sh"
export PROGRESS_CMD="$PROG"
PID=$(launch_detached "$RUNDIR" "$OUT/run.log" "$CADENCE" -- bash -lc "$CMD")
echo "launched pid=$PID rundir=$RUNDIR log=$OUT/run.log"
sleep 3
echo "heartbeat: $(heartbeat_status "$RUNDIR" "$CADENCE")"
echo "early-exit check (2s): "; sleep 2
if ! kill -0 "$PID" 2>/dev/null; then echo "WARNING: pid $PID already exited — tail of log:"; tail -20 "$OUT/run.log"; fi
REMOTE_EOF

# Forward TREE/CKPT/LEDGER/PY/GEN_PY/GROW_FROM overrides + all positionals to the box.
timeout 90 ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=accept-new -i "$KEY" ubuntu@"$IP" \
  "TREE='${TREE:-}' CKPT='${CKPT:-}' LEDGER='${LEDGER:-}' PY='${PY:-}' GEN_PY='${GEN_PY:-}' bash -s -- \
   $GO $ROLE $ALIAS $GPU_CSV $NGPU ${NFULL:-0} ${PFULL:-0} $GAMES $WORKERS ${BASE_SEED:-0} $CLAIM_ID $DATE_UTC '${DATA:-}' '${GROW_FROM:-}'" \
  <<< "$REMOTE" 2>&1 | sed 's/^/  /'
