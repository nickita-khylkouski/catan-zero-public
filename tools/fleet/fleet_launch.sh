#!/usr/bin/env bash
# tools/fleet/fleet_launch.sh — ONE canonical, guarded fleet launcher (CAT-122).
#
# Supersedes the removed drifted per-box launch scripts (fire_*.sh,
# mps_rollout.sh, and the fleet_launch_safe.sh stub). Every launch —
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
#     teacher : n_full 128, p_full 1.0   (measured high-sim recipe: shard 512,
#               EvalServer batch 96 + collector, 128 workers/GPU on <=4 GPUs)
#     volume  : n_full  64, p_full 0.25  (bulk recipe: shard 2048,
#               EvalServer batch 64, 48 workers/GPU on <=4 GPUs)
#     train   : multi-GPU DDP train_bc via torch.distributed.run (CAT-128)
#   Options:
#     --base-seed N     REQUIRED for gen roles: fresh, ledgered base seed.
#     --gpus SPEC       GPUs to use: "0-3" | "0,1,2,3" | "4" (default 0-3).
#     --games N         total games per GPU, split across its workers (default 1500).
#     --workers N       CPU game workers per GPU EvalServer (teacher default:
#                       128 on <=4 GPUs, 64 on >4; volume default: 48/32).
#     --max-neural-rows N
#                       optional hard cap on rows per EvalServer forward
#                       (default unset/uncapped while root waves are off).
#     --n-full/--n-fast/--p-full/--c-scale VALUE
#                       generation-only typed search overrides; role defaults remain
#                       unchanged when omitted.
#     --symmetry-averaged-eval [--symmetry-averaged-eval-threshold N]
#     --n-full-wide N [--n-full-wide-threshold N] [--wide-roots-always-full]
#     --wide-candidates-threshold N / --value-readout scalar|categorical
#                       generation-only S1-S3 operator fields; all default to no-op.
#     --no-cpu-affinity disable automatic GPU-local CPU pinning (default on).
#     --mps             opt into CUDA MPS (default off; EvalServer already
#                       collapses each GPU to one CUDA process).
#     --wave W          wave tag baked into the claim id (default 1).
#     --data DIR        train: corpus dir (REQUIRED for role=train).
#     --trust-curated-data
#                       train: explicit acknowledgement that DIR already passed
#                       corpus QA (REQUIRED; the launcher never silently bypasses QA).
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
BASE_SEED=""; GPUS="0-3"; GAMES=1500; WORKERS=""; WAVE=1; DATA=""; GROW_FROM=""
EVAL_SERVER_MAX_NEURAL_ROWS=""
N_FULL_OVERRIDE=""; N_FAST_OVERRIDE=""; P_FULL_OVERRIDE=""; C_SCALE_OVERRIDE=""
SYMMETRY_AVERAGED_EVAL_THRESHOLD=""; N_FULL_WIDE=""; N_FULL_WIDE_THRESHOLD=""
WIDE_ROOTS_ALWAYS_FULL=0; WIDE_CANDIDATES_THRESHOLD=""; VALUE_READOUT=""
NFAST=16; CSCALE=0.03
TRUST_CURATED=0; USE_MPS=0; CPU_AFFINITY=1; GO=0
SYMMETRY_AVERAGED_EVAL=0; RESCALE_NOISE_FLOOR_C=""; SIGMA_EVAL=""
LATE_TEMPERATURE_DECISIONS=""; LATE_TEMPERATURE=""
OPPONENT_MIX_MANIFEST=""; EXPLOITER_FRACTION=""; RUST_FEATURIZE=0
EVAL_CACHE_SIZE=""; SHARD_SIZE=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --base-seed) BASE_SEED="$2"; shift 2;;
    --gpus)      GPUS="$2"; shift 2;;
    --games)     GAMES="$2"; shift 2;;
    --workers)   WORKERS="$2"; shift 2;;
    --max-neural-rows) EVAL_SERVER_MAX_NEURAL_ROWS="$2"; shift 2;;
    --n-full) N_FULL_OVERRIDE="$2"; shift 2;;
    --n-fast) N_FAST_OVERRIDE="$2"; shift 2;;
    --p-full) P_FULL_OVERRIDE="$2"; shift 2;;
    --c-scale) C_SCALE_OVERRIDE="$2"; shift 2;;
    --wave)      WAVE="$2"; shift 2;;
    --data)      DATA="$2"; shift 2;;
    --grow-from) GROW_FROM="$2"; shift 2;;
    --trust-curated-data) TRUST_CURATED=1; shift;;
    --mps)       USE_MPS=1; shift;;
    --no-cpu-affinity) CPU_AFFINITY=0; shift;;
    --symmetry-averaged-eval) SYMMETRY_AVERAGED_EVAL=1; shift;;
    --symmetry-averaged-eval-threshold) SYMMETRY_AVERAGED_EVAL_THRESHOLD="$2"; shift 2;;
    --n-full-wide) N_FULL_WIDE="$2"; shift 2;;
    --n-full-wide-threshold) N_FULL_WIDE_THRESHOLD="$2"; shift 2;;
    --wide-roots-always-full) WIDE_ROOTS_ALWAYS_FULL=1; shift;;
    --wide-candidates-threshold) WIDE_CANDIDATES_THRESHOLD="$2"; shift 2;;
    --value-readout) VALUE_READOUT="$2"; shift 2;;
    --rescale-noise-floor-c) RESCALE_NOISE_FLOOR_C="$2"; shift 2;;
    --sigma-eval) SIGMA_EVAL="$2"; shift 2;;
    --late-temperature-decisions) LATE_TEMPERATURE_DECISIONS="$2"; shift 2;;
    --late-temperature) LATE_TEMPERATURE="$2"; shift 2;;
    --opponent-mix-manifest) OPPONENT_MIX_MANIFEST="$2"; shift 2;;
    --exploiter-fraction) EXPLOITER_FRACTION="$2"; shift 2;;
    --rust-featurize) RUST_FEATURIZE=1; shift;;
    --eval-cache-size) EVAL_CACHE_SIZE="$2"; shift 2;;
    --shard-size) SHARD_SIZE="$2"; shift 2;;
    --go)        GO=1; shift;;
    --skip-guards) echo "fleet_launch: --skip-guards is RETIRED (CAT-124). Claim your seed range first; guards now pass for a legitimate fresh claim." >&2; exit 2;;
    *) echo "fleet_launch: unknown option '$1'" >&2; exit 2;;
  esac
done

case "$ROLE" in
  teacher) NFULL=128; PFULL=1.0;  EVAL_SERVER_MAX_BATCH=96; EVAL_SERVER_REQUEST_COLLECTOR=1;;
  volume)  NFULL=64;  PFULL=0.25; EVAL_SERVER_MAX_BATCH=64; EVAL_SERVER_REQUEST_COLLECTOR=0;;
  train)   EVAL_SERVER_MAX_BATCH=0; EVAL_SERVER_REQUEST_COLLECTOR=0;;
  *) echo "fleet_launch: role must be teacher|volume|train" >&2; exit 2;;
esac
[ -z "$N_FULL_OVERRIDE" ] || NFULL="$N_FULL_OVERRIDE"
[ -z "$N_FAST_OVERRIDE" ] || NFAST="$N_FAST_OVERRIDE"
[ -z "$P_FULL_OVERRIDE" ] || PFULL="$P_FULL_OVERRIDE"
[ -z "$C_SCALE_OVERRIDE" ] || CSCALE="$C_SCALE_OVERRIDE"

IP=$(fleet_host "$ALIAS") || exit 2
KEY=$(fleet_key)

[[ "$ALIAS" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
  || { echo "fleet_launch: unsafe alias '$ALIAS' (use letters, digits, dot, underscore, dash)" >&2; exit 2; }
[[ "$WAVE" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,31}$ ]] \
  || { echo "fleet_launch: --wave must be a 1-32 character safe slug" >&2; exit 2; }

# Seconds alone are not unique: two same-role launches in one second used to
# share a ledger exemption, run directory, and output directory.  A 64-bit
# local nonce makes the claim collision-resistant without relying on GNU date
# extensions (the operator workstation may be macOS).
EPOCH=$(date -u +%s)
NONCE=$(od -An -N8 -tx1 /dev/urandom 2>/dev/null | tr -d ' \n')
[ -n "$NONCE" ] || NONCE="${RANDOM}${RANDOM}${RANDOM}"
CLAIM_ID="${ALIAS}-${ROLE}-w${WAVE}-${EPOCH}-${NONCE}"
DATE_UTC=$(date -u +%Y-%m-%d)

# Expand --gpus "0-3" | "0,1,2,3" into a comma list + a count.
expand_gpus() {
  local spec="$1" start end gpu out="" seen="," ids
  if [[ "$spec" =~ ^([0-9]+)-([0-9]+)$ ]]; then
    start="${BASH_REMATCH[1]}"; end="${BASH_REMATCH[2]}"
    [ "$start" -le "$end" ] || { echo "fleet_launch: descending GPU range '$spec'" >&2; return 2; }
    for ((gpu=start; gpu<=end; gpu++)); do
      [ -z "$out" ] || out+=","
      out+="$gpu"
    done
    echo "$out"
  elif [[ "$spec" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    IFS=, read -r -a ids <<< "$spec"
    for gpu in "${ids[@]}"; do
      case "$seen" in
        *",$gpu,"*) echo "fleet_launch: duplicate GPU '$gpu' in --gpus '$spec'" >&2; return 2;;
      esac
      seen+="$gpu,"
    done
    echo "$spec"
  else
    echo "fleet_launch: invalid --gpus '$spec' (expected 0-3 or 0,1,2,3)" >&2
    return 2
  fi
}
GPU_CSV=$(expand_gpus "$GPUS") || exit 2
NGPU=$(awk -F, '{print NF}' <<<"$GPU_CSV")

# Measured generation defaults are role- and host-shape-specific. Teacher's
# smaller shards and collector-fed EvalServer sustain 128 workers/GPU on the
# <=4-GPU fleet shape and 64 on the shared-CPU 8-GPU canary; volume retains its
# measured 48/32 defaults. An explicit --workers always wins. Train keeps the
# prior 48/32 placeholder even though its DDP path does not consume WORKERS.
if [ -z "$WORKERS" ]; then
  if [ "$ROLE" = "teacher" ]; then
    if [ "$NGPU" -le 4 ]; then WORKERS=128; else WORKERS=64; fi
  else
    if [ "$NGPU" -le 4 ]; then WORKERS=48; else WORKERS=32; fi
  fi
fi

[[ "$GAMES" =~ ^[1-9][0-9]*$ ]] || { echo "fleet_launch: --games must be a positive integer" >&2; exit 2; }
[[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || { echo "fleet_launch: --workers must be a positive integer" >&2; exit 2; }
if [ -n "$EVAL_SERVER_MAX_NEURAL_ROWS" ]; then
  [[ "$EVAL_SERVER_MAX_NEURAL_ROWS" =~ ^[1-9][0-9]*$ ]] \
    || { echo "fleet_launch: --max-neural-rows must be a positive integer" >&2; exit 2; }
fi
for FIELD in "$N_FULL_OVERRIDE" "$N_FAST_OVERRIDE" "$N_FULL_WIDE" \
  "$N_FULL_WIDE_THRESHOLD" "$SYMMETRY_AVERAGED_EVAL_THRESHOLD" \
  "$WIDE_CANDIDATES_THRESHOLD"; do
  [ -z "$FIELD" ] || [[ "$FIELD" =~ ^[1-9][0-9]*$ ]] \
    || { echo "fleet_launch: search counts/thresholds must be positive integers" >&2; exit 2; }
done
case "$VALUE_READOUT" in
  ""|scalar|categorical) ;;
  *) echo "fleet_launch: --value-readout must be scalar or categorical" >&2; exit 2;;
esac

# ---- role-specific validation --------------------------------------------
if [ "$ROLE" = "train" ]; then
  if [ "$SYMMETRY_AVERAGED_EVAL" = 1 ] || [ -n "$RESCALE_NOISE_FLOOR_C" ] || [ -n "$SIGMA_EVAL" ] || \
     [ -n "$LATE_TEMPERATURE_DECISIONS" ] || [ -n "$LATE_TEMPERATURE" ] || \
     [ -n "$OPPONENT_MIX_MANIFEST" ] || [ -n "$EXPLOITER_FRACTION" ] || \
     [ "$RUST_FEATURIZE" = 1 ] || [ -n "$EVAL_CACHE_SIZE" ] || [ -n "$SHARD_SIZE" ] || \
     [ -n "$EVAL_SERVER_MAX_NEURAL_ROWS" ] || [ -n "$N_FULL_OVERRIDE" ] || \
     [ -n "$N_FAST_OVERRIDE" ] || [ -n "$P_FULL_OVERRIDE" ] || [ -n "$C_SCALE_OVERRIDE" ] || \
     [ -n "$SYMMETRY_AVERAGED_EVAL_THRESHOLD" ] || [ -n "$N_FULL_WIDE" ] || \
     [ -n "$N_FULL_WIDE_THRESHOLD" ] || [ "$WIDE_ROOTS_ALWAYS_FULL" = 1 ] || \
     [ -n "$WIDE_CANDIDATES_THRESHOLD" ] || [ -n "$VALUE_READOUT" ]; then
    echo "fleet_launch: generation science options are not valid for role=train" >&2
    exit 2
  fi
  [ -n "$DATA" ] || { echo "fleet_launch: role=train needs --data <corpus dir>" >&2; exit 2; }
  [ "$TRUST_CURATED" -eq 1 ] || {
    echo "fleet_launch: role=train requires --trust-curated-data after corpus QA; refusing to silently bypass teacher-quality gates" >&2
    exit 2
  }
  [ $(( 4096 % NGPU )) -eq 0 ] || {
    echo "fleet_launch: train GPU count $NGPU does not divide the required global batch 4096" >&2
    exit 2
  }
else
  [ -n "$BASE_SEED" ] || { echo "fleet_launch: gen role needs --base-seed <fresh ledgered seed>" >&2; exit 2; }
  [[ "$BASE_SEED" =~ ^(0|[1-9][0-9]*)$ ]] \
    || { echo "fleet_launch: --base-seed must be a non-negative decimal integer" >&2; exit 2; }
  if [ -n "$EXPLOITER_FRACTION" ] && [ -z "$OPPONENT_MIX_MANIFEST" ]; then
    echo "fleet_launch: --exploiter-fraction requires --opponent-mix-manifest" >&2
    exit 2
  fi
  if [ -n "$SYMMETRY_AVERAGED_EVAL_THRESHOLD" ] && [ "$SYMMETRY_AVERAGED_EVAL" != 1 ]; then
    echo "fleet_launch: --symmetry-averaged-eval-threshold requires --symmetry-averaged-eval" >&2
    exit 2
  fi
  if { [ -n "$N_FULL_WIDE_THRESHOLD" ] || [ "$WIDE_ROOTS_ALWAYS_FULL" = 1 ]; } && [ -z "$N_FULL_WIDE" ]; then
    echo "fleet_launch: wide threshold/always-full requires --n-full-wide" >&2
    exit 2
  fi
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
TRUST_CURATED="${15}"; USE_MPS="${16}"; CPU_AFFINITY="${17}"
SHARD_SIZE="${18}"; EVAL_SERVER_MAX_BATCH="${19}"
EVAL_SERVER_REQUEST_COLLECTOR="${20}"
SYMMETRY_AVERAGED_EVAL="${21}"; RESCALE_NOISE_FLOOR_C="${22}"; SIGMA_EVAL="${23}"
LATE_TEMPERATURE_DECISIONS="${24}"; LATE_TEMPERATURE="${25}"
OPPONENT_MIX_MANIFEST="${26}"; EXPLOITER_FRACTION="${27}"
RUST_FEATURIZE="${28}"; EVAL_CACHE_SIZE="${29}"
EVAL_SERVER_MAX_NEURAL_ROWS="${30}"
NFAST="${31}"; CSCALE="${32}"
SYMMETRY_AVERAGED_EVAL_THRESHOLD="${33}"; N_FULL_WIDE="${34}"
N_FULL_WIDE_THRESHOLD="${35}"; WIDE_ROOTS_ALWAYS_FULL="${36}"
WIDE_CANDIDATES_THRESHOLD="${37}"; VALUE_READOUT="${38}"

TREE="${TREE:-$HOME/catan-zero-v1}"
CKPT="${CKPT:-$HOME/bundle/champion_v0.pt}"
LEDGER="${LEDGER:-$TREE/runs/SEED_LEDGER.md}"
GROW_FROM="${GROW_FROM_IN:-$CKPT}"
# The random-suffixed claim id is also the filesystem identity.  This prevents
# same-second launches from sharing logs or state even before the ledger guard
# gets a chance to compare their seed ranges.
if [ "$ROLE" = "train" ]; then
  OUT="$HOME/train_out/${CLAIM_ID}"
else
  OUT="$HOME/gen_out/${CLAIM_ID}"
fi
RUNDIR="$HOME/fleet_runs/${CLAIM_ID}"

# --- interpreter resolution (CAT-123/128): venv first, tree .venv fallback, NEVER bare python3
PY="${PY:-}"; GEN_PY="${GEN_PY:-}"
resolve_py() { if [ -x "$HOME/venv/bin/python" ]; then echo "$HOME/venv/bin/python"; elif [ -x "$TREE/.venv/bin/python" ]; then echo "$TREE/.venv/bin/python"; else echo ""; fi; }
[ -z "$PY" ] && PY=$(resolve_py); [ -z "$GEN_PY" ] && GEN_PY="$PY"

FAIL=0
[ -n "$PY" ] && [ -x "$PY" ] && echo "ok: interpreter $PY" || { echo "FAIL: no python (~/venv or \$TREE/.venv) — refusing (bare python3 loads numpy<2, CAT-128)"; FAIL=1; }
[ -d "$TREE" ] && echo "ok: tree $TREE" || { echo "FAIL: tree $TREE missing"; FAIL=1; }
if [ "$ROLE" = "train" ]; then
  [ -d "$DATA" ] && echo "ok: corpus $DATA" || { echo "FAIL: --data $DATA missing"; FAIL=1; }
  [ "$TRUST_CURATED" = "1" ] \
    && echo "ok: operator explicitly acknowledged curated corpus QA" \
    || { echo "FAIL: train launch lacks --trust-curated-data acknowledgement"; FAIL=1; }
  [ -f "$GROW_FROM" ] && echo "ok: grow-from $GROW_FROM" || { echo "FAIL: --grow-from $GROW_FROM missing"; FAIL=1; }
else
  [ -f "$CKPT" ] && echo "ok: champion $CKPT" || { echo "FAIL: champion $CKPT missing"; FAIL=1; }
  [ -f "$LEDGER" ] && echo "ok: ledger $LEDGER" || { echo "FAIL: ledger $LEDGER missing (sync it here first — cross-host seed safety, CAT-125)"; FAIL=1; }
  if [ -n "$OPPONENT_MIX_MANIFEST" ]; then
    [ -f "$OPPONENT_MIX_MANIFEST" ] \
      && echo "ok: opponent mix $OPPONENT_MIX_MANIFEST" \
      || { echo "FAIL: --opponent-mix-manifest $OPPONENT_MIX_MANIFEST missing"; FAIL=1; }
  fi
  if [ "$USE_MPS" = "1" ]; then
    command -v nvidia-cuda-mps-control >/dev/null 2>&1 \
      && echo "ok: optional nvidia-cuda-mps-control available" \
      || { echo "FAIL: --mps requested but nvidia-cuda-mps-control is missing"; FAIL=1; }
  else
    echo "ok: MPS disabled (one EvalServer CUDA process per GPU)"
  fi
fi

# Fail before claiming seeds if the requested physical device list does not
# exist on this host. This also catches stale 4-GPU defaults on 2-GPU boxes and
# typos such as --gpus 0-8 on an 8-device node.
if command -v nvidia-smi >/dev/null 2>&1; then
  AVAILABLE_GPUS=","
  while IFS= read -r GPU_INDEX; do
    GPU_INDEX="${GPU_INDEX//[[:space:]]/}"
    [ -n "$GPU_INDEX" ] && AVAILABLE_GPUS+="$GPU_INDEX,"
  done < <(nvidia-smi --query-gpu=index --format=csv,noheader)
  IFS=, read -r -a REQUESTED_GPUS <<< "$GPU_CSV"
  for GPU_INDEX in "${REQUESTED_GPUS[@]}"; do
    case "$AVAILABLE_GPUS" in
      *",$GPU_INDEX,"*) ;;
      *) echo "FAIL: requested GPU $GPU_INDEX is not present (available: ${AVAILABLE_GPUS#,})"; FAIL=1;;
    esac
  done
else
  echo "FAIL: nvidia-smi missing; cannot validate --gpus $GPU_CSV"
  FAIL=1
fi

# --- build the pinned command per role -------------------------------------
if [ "$ROLE" = "train" ]; then
  # CAT-128: torch.distributed.run via the RESOLVED venv python (never bare torchrun);
  # --tee=3 streams rank stdout/err; depth-grow warm-start (h640/L6 preserves warm-start,
  # width-grow cold-starts). Pins the leak-critical + curated-memmap flags.
  # The canonical scientific control remains the audited ~35M entity model.
  # Larger 70-100M experiments need a separate, explicit launcher surface; they
  # must never silently replace the control.  NGPU divisibility was validated
  # locally, so per-rank batch * world size is exactly 4096.
  TRAIN_BATCH=$(( 4096 / NGPU ))
  CMD="$RUNDIR/run_training.sh"
  CADENCE=120
  PROG="grep -oE 'step=[0-9]+/[0-9]+|epoch [0-9]+' $OUT/run.log 2>/dev/null | tail -1"
else
  # Generation is one independent generator/eval-server per physical GPU. A
  # single process with CUDA_VISIBLE_DEVICES=0,1,... and --device cuda selects
  # logical cuda:0 for every worker, silently idling the other GPUs. The runner
  # below pins each generator to one physical GPU and assigns a disjoint block of
  # GAMES seeds. It is materialized only on the GO path.
  CMD="$RUNDIR/run_generation.sh"
  CADENCE=60
  PROG="find $OUT -type f -name '*.npz' 2>/dev/null | wc -l"
fi

# --- CLAIM row (append BEFORE guard so ledger_overlap sees + excludes our own) ---
if [ "$ROLE" = "train" ]; then
  CLAIM_ROW="# (train role claims no seed range) claim=$CLAIM_ID $ALIAS train $DATE_UTC"
else
  # --games is TOTAL per generator (workers partition it); one generator runs
  # per GPU, so the claimed interval is GAMES * NGPU, never * WORKERS.
  END=$(( BASE_SEED + GAMES * NGPU ))
  CLAIM_ROW="[$BASE_SEED – $END) | fleet/$ALIAS | $ROLE n$NFULL p$PFULL gpus=$GPU_CSV claim=$CLAIM_ID | $DATE_UTC"
fi

echo "CLAIM ROW  : $CLAIM_ROW"
echo "CLAIM ID   : $CLAIM_ID  (exported as CATAN_LEDGER_CLAIM_ID)"
echo "OUT        : $OUT"
echo "WOULD RUN  : $CMD"

[ ! -e "$RUNDIR" ] || { echo "FAIL: run directory already exists: $RUNDIR"; FAIL=1; }
[ ! -e "$OUT" ] || { echo "FAIL: output directory already exists: $OUT"; FAIL=1; }
[ "$FAIL" -ne 0 ] && { echo "REFUSING: precondition(s) failed."; exit 3; }
if [ "$GO" != "1" ]; then echo "DRY-RUN: preconditions passed; not launched (pass --go)."; exit 0; fi

# ===== GO path =====
mkdir -p "$HOME/fleet_runs" "$HOME/gen_out" "$HOME/train_out"
# Atomic no-clobber creation closes the precheck/use race.  The random claim id
# makes a collision extraordinarily unlikely; if one occurs we fail closed.
mkdir "$RUNDIR" || { echo "REFUSING: claim run directory appeared concurrently: $RUNDIR"; exit 4; }
if ! mkdir "$OUT"; then
  rmdir "$RUNDIR" 2>/dev/null || true
  echo "REFUSING: claim output directory appeared concurrently: $OUT"
  exit 4
fi
# 1. CLAIM: append to the box-local ledger (operator runs sync_seed_ledger.py to reconcile).
if [ "$ROLE" != "train" ]; then printf '%s\n' "$CLAIM_ROW" >> "$LEDGER"; echo "claimed: appended to $LEDGER"; fi
export CATAN_LEDGER_CLAIM_ID="$CLAIM_ID"
export CATAN_SEED_LEDGER="$LEDGER"
# 2+3. GUARD + LAUNCH via the detach lib (guards run inside the tool; --tee/exit-check for train).
source "$TREE/tools/fleet/launch_detached.sh" 2>/dev/null || source "$HOME/launch_detached.sh"
export PROGRESS_CMD="$PROG"
if [ "$ROLE" != "train" ]; then
  # Values are inherited through the environment so the runner stays readable
  # and no shell-quoted mega-command is duplicated for every GPU.
  export TREE CKPT GEN_PY OUT GPU_CSV GAMES WORKERS BASE_SEED NFULL NFAST PFULL CSCALE CLAIM_ID USE_MPS CPU_AFFINITY
  export SHARD_SIZE EVAL_SERVER_MAX_BATCH EVAL_SERVER_REQUEST_COLLECTOR
  export SYMMETRY_AVERAGED_EVAL RESCALE_NOISE_FLOOR_C SIGMA_EVAL
  export LATE_TEMPERATURE_DECISIONS LATE_TEMPERATURE OPPONENT_MIX_MANIFEST
  export EXPLOITER_FRACTION RUST_FEATURIZE EVAL_CACHE_SIZE
  export EVAL_SERVER_MAX_NEURAL_ROWS
  export SYMMETRY_AVERAGED_EVAL_THRESHOLD N_FULL_WIDE N_FULL_WIDE_THRESHOLD
  export WIDE_ROOTS_ALWAYS_FULL WIDE_CANDIDATES_THRESHOLD VALUE_READOUT
  cat > "$RUNDIR/run_generation.sh" <<'GEN_RUNNER_EOF'
#!/usr/bin/env bash
set -euo pipefail
cd "$TREE"
ulimit -n 65536

if [ "$USE_MPS" = "1" ]; then
  export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe_host
  export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_host
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  MPS_RUNNING=0
  ps -eo comm=,args= 2>/dev/null \
    | awk '$1 ~ /^nvidia-cuda-mps/ && ($0 ~ /mps-control -d/ || $0 ~ /mps-server/) {found=1} END {exit !found}' \
    && MPS_RUNNING=1
  if [ "$MPS_RUNNING" -ne 1 ]; then
    # A prior clean `quit` may leave a stale control socket behind.
    rm -rf "$CUDA_MPS_PIPE_DIRECTORY"
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY"
    nvidia-cuda-mps-control -d
    sleep 2
  fi
  echo get_server_list | nvidia-cuda-mps-control >/dev/null 2>&1 \
    || { echo "MPS daemon failed readiness check" >&2; exit 4; }
else
  unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
fi

IFS="," read -r -a GPU_IDS <<< "$GPU_CSV"
PIDS=()
GPU_ORDINAL=0
EVAL_SERVER_COLLECTOR_FLAG="--no-eval-server-request-collector"
if [ "$EVAL_SERVER_REQUEST_COLLECTOR" = "1" ]; then
  EVAL_SERVER_COLLECTOR_FLAG="--eval-server-request-collector"
fi
SCIENCE_ARGS=()
[ "$SYMMETRY_AVERAGED_EVAL" = "1" ] && SCIENCE_ARGS+=(--symmetry-averaged-eval)
[ -z "$RESCALE_NOISE_FLOOR_C" ] || SCIENCE_ARGS+=(--rescale-noise-floor-c "$RESCALE_NOISE_FLOOR_C")
[ -z "$SIGMA_EVAL" ] || SCIENCE_ARGS+=(--sigma-eval "$SIGMA_EVAL")
[ -z "$LATE_TEMPERATURE_DECISIONS" ] || SCIENCE_ARGS+=(--late-temperature-decisions "$LATE_TEMPERATURE_DECISIONS")
[ -z "$LATE_TEMPERATURE" ] || SCIENCE_ARGS+=(--late-temperature "$LATE_TEMPERATURE")
[ -z "$OPPONENT_MIX_MANIFEST" ] || SCIENCE_ARGS+=(--opponent-mix-manifest "$OPPONENT_MIX_MANIFEST")
[ -z "$EXPLOITER_FRACTION" ] || SCIENCE_ARGS+=(--exploiter-fraction "$EXPLOITER_FRACTION")
[ -z "$SHARD_SIZE" ] || SCIENCE_ARGS+=(--shard-size "$SHARD_SIZE")
[ -z "$SYMMETRY_AVERAGED_EVAL_THRESHOLD" ] || SCIENCE_ARGS+=(--symmetry-averaged-eval-threshold "$SYMMETRY_AVERAGED_EVAL_THRESHOLD")
[ -z "$N_FULL_WIDE" ] || SCIENCE_ARGS+=(--n-full-wide "$N_FULL_WIDE")
[ -z "$N_FULL_WIDE_THRESHOLD" ] || SCIENCE_ARGS+=(--n-full-wide-threshold "$N_FULL_WIDE_THRESHOLD")
[ "$WIDE_ROOTS_ALWAYS_FULL" != 1 ] || SCIENCE_ARGS+=(--wide-roots-always-full)
[ -z "$WIDE_CANDIDATES_THRESHOLD" ] || SCIENCE_ARGS+=(--wide-candidates-threshold "$WIDE_CANDIDATES_THRESHOLD")
[ -z "$VALUE_READOUT" ] || SCIENCE_ARGS+=(--value-readout "$VALUE_READOUT")
EVAL_SERVER_ROW_CAP_ARGS=()
if [ -n "$EVAL_SERVER_MAX_NEURAL_ROWS" ]; then
  EVAL_SERVER_ROW_CAP_ARGS=(--eval-server-max-neural-rows "$EVAL_SERVER_MAX_NEURAL_ROWS")
fi

# All generators, EvalServers, managers, and multiprocessing workers inherit
# this detached runner's PGID.  On direct runner termination, signal that exact
# group (never a name pattern), wait briefly, then KILL only remaining members.
group_members() {
  local inspector="$BASHPID"
  ps -eo pid=,pgid=,comm= 2>/dev/null \
    | awk -v group="$$" -v inspector="$inspector" \
        '$2 == group && $1 != group && $1 != inspector && $3 != "ps" && $3 != "awk" {print $1}'
}

cleanup_group() {
  RC="${1:-0}"
  trap - EXIT INT TERM
  trap '' INT TERM
  MEMBERS=$(group_members)
  if [ -n "$MEMBERS" ]; then
    kill -TERM -- "-$$" 2>/dev/null || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      sleep 0.5
      MEMBERS=$(group_members)
      [ -z "$MEMBERS" ] && break
    done
    MEMBERS=$(group_members)
    [ -z "$MEMBERS" ] || kill -KILL $MEMBERS 2>/dev/null || true
  fi
  [ "${#PIDS[@]}" -eq 0 ] || wait "${PIDS[@]}" 2>/dev/null || true
  exit "$RC"
}
trap 'cleanup_group $?' EXIT
trap 'cleanup_group 130' INT
trap 'cleanup_group 143' TERM

for GPU in "${GPU_IDS[@]}"; do
  GPU_BASE_SEED=$(( BASE_SEED + GPU_ORDINAL * GAMES ))
  GPU_OUT="$OUT/gpu${GPU}"
  mkdir -p "$GPU_OUT"
  CPU_PREFIX=()
  GPU_CPUSET=""
  if [ "$CPU_AFFINITY" = "1" ] && command -v taskset >/dev/null 2>&1; then
    # nvidia-smi topo's final columns are CPU Affinity, NUMA Affinity,
    # GPU NUMA ID.  Pinning the whole generator tree makes first-touch memory
    # and IPC local to the GPU's socket.  Invalid/unknown topology fails open to
    # the scheduler rather than inventing an affinity.
    # `nvidia-smi topo` can return nonzero transiently while Fabric Manager is
    # refreshing.  Affinity is an optimization, never a launch precondition.
    GPU_CPUSET=$(nvidia-smi topo -m 2>/dev/null \
      | awk -v gpu="GPU${GPU}" '$1 == gpu {print $(NF-2)}') || GPU_CPUSET=""
    if [[ "$GPU_CPUSET" =~ ^[0-9,-]+$ ]]; then
      CPU_PREFIX=(taskset -c "$GPU_CPUSET")
    fi
  fi
  echo "launching gpu=$GPU cpus=${GPU_CPUSET:-unbound} seed=[$GPU_BASE_SEED,$((GPU_BASE_SEED + GAMES))) out=$GPU_OUT"
  CUDA_VISIBLE_DEVICES="$GPU" \
    "${CPU_PREFIX[@]}" "$GEN_PY" tools/generate_gumbel_selfplay_data.py \
    --out-dir "$GPU_OUT" --checkpoint "$CKPT" --device cuda \
    --games "$GAMES" --workers "$WORKERS" --base-seed "$GPU_BASE_SEED" \
    --n-full "$NFULL" --n-fast "$NFAST" --p-full "$PFULL" --c-visit 50.0 --c-scale "$CSCALE" \
    --max-decisions 600 --max-depth 80 --temperature-decisions 90 \
    --correct-rust-chance-spectra --lazy-interior-chance --public-observation \
    --rust-featurize --eval-server --eval-server-max-batch "$EVAL_SERVER_MAX_BATCH" \
    --eval-server-max-wait-ms 0.0 --eval-server-matmul-precision highest \
    "${EVAL_SERVER_ROW_CAP_ARGS[@]}" \
    --eval-server-transport mp_queue --eval-server-event-token-limit 0 \
    --no-root-wave-batching --no-eval-server-cuda-graph \
    "$EVAL_SERVER_COLLECTOR_FLAG" --no-eval-server-local-fallback \
    --eval-cache-size "${EVAL_CACHE_SIZE:-0}" \
    --track 2p_no_trade --vps-to-win 10 --format npz --score-actions \
    --dump-config "$GPU_OUT/config.json" --config-purpose "fleet-$CLAIM_ID" \
    --ledger-claim-label "$CLAIM_ID" \
    "${SCIENCE_ARGS[@]}" \
    >"$GPU_OUT/run.log" 2>&1 &
  PIDS+=("$!")
  GPU_ORDINAL=$(( GPU_ORDINAL + 1 ))
done

RC=0
for CHILD_PID in "${PIDS[@]}"; do
  wait "$CHILD_PID" || RC=1
done
exit "$RC"
GEN_RUNNER_EOF
  chmod 0755 "$RUNDIR/run_generation.sh"
  PID=$(launch_detached "$RUNDIR" "$OUT/run.log" "$CADENCE" -- "$RUNDIR/run_generation.sh")
else
  export TREE PY DATA OUT GROW_FROM GPU_CSV NGPU TRAIN_BATCH
  cat > "$RUNDIR/run_training.sh" <<'TRAIN_RUNNER_EOF'
#!/usr/bin/env bash
set -euo pipefail
cd "$TREE"
ulimit -n 65536
export CUDA_VISIBLE_DEVICES="$GPU_CSV"
exec "$PY" -m torch.distributed.run --standalone --nproc_per_node="$NGPU" --tee=3 \
  tools/train_bc.py --arch entity_graph --data "$DATA" --data-format memmap \
  --data-loader-workers 4 --data-loader-prefetch 4 --batch-size "$TRAIN_BATCH" --epochs 1 \
  --checkpoint "$OUT/model.pt" --report "$OUT/report.json" \
  --grow-from-checkpoint "$GROW_FROM" --graph-layers 6 --hidden-size 640 \
  --attention-heads 8 --graph-dropout 0.05 --require-35m-model \
  --mask-hidden-info --soft-target-source policy \
  --skip-teacher-quality-gate --trust-curated-data-quality \
  --amp bf16 --fused-optimizer --lr 3e-5 --lr-warmup-steps 100 \
  --optimizer adam --weight-decay 0.0 --lr-schedule flat \
  --truncated-vp-margin-value-weight 0.25
TRAIN_RUNNER_EOF
  chmod 0755 "$RUNDIR/run_training.sh"
  PID=$(launch_detached "$RUNDIR" "$OUT/run.log" "$CADENCE" -- "$RUNDIR/run_training.sh")
fi
echo "launched pid=$PID rundir=$RUNDIR log=$OUT/run.log"
sleep 3
echo "heartbeat: $(heartbeat_status "$RUNDIR" "$CADENCE")"
echo "early-exit check (2s): "; sleep 2
if ! kill -0 "$PID" 2>/dev/null; then echo "WARNING: pid $PID already exited — tail of log:"; tail -20 "$OUT/run.log"; fi
REMOTE_EOF

# Forward overrides + positionals as shell-escaped words.  ssh joins command
# arguments into a remote-shell string, so passing a local argv array directly
# is not sufficient; Bash %q preserves spaces, quotes, and metacharacters.
REMOTE_WORDS=(
  env "TREE=${TREE:-}" "CKPT=${CKPT:-}" "LEDGER=${LEDGER:-}" "PY=${PY:-}" "GEN_PY=${GEN_PY:-}"
  bash -s -- "$GO" "$ROLE" "$ALIAS" "$GPU_CSV" "$NGPU" "${NFULL:-0}" "${PFULL:-0}"
  "$GAMES" "$WORKERS" "${BASE_SEED:-0}" "$CLAIM_ID" "$DATE_UTC" "${DATA:-}" "${GROW_FROM:-}"
  "$TRUST_CURATED" "$USE_MPS" "$CPU_AFFINITY" "$SHARD_SIZE" "$EVAL_SERVER_MAX_BATCH"
  "$EVAL_SERVER_REQUEST_COLLECTOR"
  "$SYMMETRY_AVERAGED_EVAL" "$RESCALE_NOISE_FLOOR_C" "$SIGMA_EVAL"
  "$LATE_TEMPERATURE_DECISIONS" "$LATE_TEMPERATURE" "$OPPONENT_MIX_MANIFEST"
  "$EXPLOITER_FRACTION" "$RUST_FEATURIZE" "$EVAL_CACHE_SIZE"
  "$EVAL_SERVER_MAX_NEURAL_ROWS"
  "$NFAST" "$CSCALE" "$SYMMETRY_AVERAGED_EVAL_THRESHOLD" "$N_FULL_WIDE"
  "$N_FULL_WIDE_THRESHOLD" "$WIDE_ROOTS_ALWAYS_FULL" "$WIDE_CANDIDATES_THRESHOLD"
  "$VALUE_READOUT"
)
printf -v REMOTE_COMMAND '%q ' "${REMOTE_WORDS[@]}"
timeout 90 ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=accept-new -i "$KEY" ubuntu@"$IP" \
  "$REMOTE_COMMAND" \
  <<< "$REMOTE" 2>&1 | sed 's/^/  /'
