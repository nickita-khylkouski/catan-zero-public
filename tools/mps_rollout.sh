#!/usr/bin/env bash
# MPS fleet rollout (team-lead adopted, report 8/9 grid evidence: MPS+16w =
# ~3x per-GPU generation throughput; no-MPS packing beyond 8 REGRESSES).
#
# DESIGN: ONE MPS control daemon per HOST (handles every GPU); workers opt in
# via CUDA_MPS_PIPE_DIRECTORY env at launch, so canary GPUs run under MPS
# while the rest keep their existing non-MPS processes untouched. MPS is a
# GPU-SCHEDULING change only — generated data is bit-identical (no bit-parity
# gate needed, per team-lead).
#
# MODES:
#   mps_rollout.sh canary <gpu> <base_seed>   one GPU -> 16w under MPS
#   mps_rollout.sh host "<gpu:seed> <gpu:seed> ..."   every listed GPU
#   mps_rollout.sh rollback "<gpu:seed> ..."  revert listed GPUs -> 8 workers,
#                                             NO MPS (daemon left up for any
#                                             GPUs still on it; quit manually
#                                             with `echo quit | nvidia-cuda-
#                                             mps-control` once none are)
#   mps_rollout.sh status                     daemon + per-GPU proc counts
#
# GEN-3 SYNC COMPOSITION (one restart per host, per team-lead's bundle
# decision): run GEN3_WHEEL_SYNC_RUNBOOK.md's stop + wheel-install steps
# FIRST (generation down, 0.1.3 wheel in), then THIS script's `host` mode as
# the relaunch step — it starts the daemon and brings generation back at
# 16w/MPS, with the runbook's flag flips supplied via GEN_EXTRA_ARGS (e.g.
# GEN_EXTRA_ARGS="--exact-budget-sh" if that gate passed). WHEEL rollback is
# the runbook's concern; this script's `rollback` reverts only the
# MPS/packing half. All completed shards on disk are durable across every
# stop/relaunch here (kills end processes, never touch files).
#
# HARD RULES: seeds MUST be ledgered by team-lead before invocation. Existing
# generation on a target GPU is stopped by EXPLICIT PID (parent whose cmdline
# carries that GPU's out-dir token, then nvidia-smi -i <gpu> compute PIDs) —
# never pattern kills, never other GPUs. Non-target GPUs are never touched.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

# CAT-126: generation opens many zstd/npz shards; 1024 is not enough.
ulimit -n 65536

CKPT="${CKPT:-runs/bc/gen2A_20260706/checkpoint.pt}"
WORKERS="${WORKERS:-16}"
GAMES="${GAMES:-1500}"
OUT_PREFIX="${OUT_PREFIX:-runs/selfplay/gen3_mps_$(date +%Y%m%d)}"

GEN_ARGS=(
  --checkpoint "$CKPT" --device cuda
  --n-full "${N_FULL:-64}" --n-fast "${N_FAST:-16}" --p-full "${P_FULL:-0.25}" --c-visit 50.0 --c-scale "${C_SCALE:-0.03}"
  --max-decisions 600 --max-depth 80 --temperature-decisions 90
  --correct-rust-chance-spectra --lazy-interior-chance --public-observation
  --track 2p_no_trade --vps-to-win 10 --shard-size 2048 --format npz
  --score-actions
)
# GEN_EXTRA_ARGS: space-separated extra CLI flags appended at launch (the
# gen-3 runbook's gate-dependent flag flips, e.g. "--exact-budget-sh").
read -r -a EXTRA_ARGS <<< "${GEN_EXTRA_ARGS:-}"

# Validate GEN_EXTRA_ARGS: only the runbook allowlist is permitted, and
# --skip-guards is categorically forbidden (it would bypass the prelaunch guard
# that the rest of this script relies on to prevent silent default overrides).
validate_extra_args() {
  local i=0
  while [ "$i" -lt "${#EXTRA_ARGS[@]}" ]; do
    local tok="${EXTRA_ARGS[$i]}"
    case "$tok" in
      --exact-budget-sh|--rust-featurize)
        i=$((i+1));;
      --exact-budget-sh-min-n)
        # skip the required numeric argument
        if [ "$((i+1))" -ge "${#EXTRA_ARGS[@]}" ]; then
          echo "mps_rollout: $tok requires a value" >&2; exit 1
        fi
        i=$((i+2));;
      --skip-guards|--no-public-observation|--no-lazy-interior-chance|--c-scale|--c-visit|--p-full|--n-full|--n-fast|--temperature-decisions|--public-observation|--lazy-interior-chance)
        echo "mps_rollout: GEN_EXTRA_ARGS token '$tok' is forbidden; it can override the production recipe or disable guards" >&2; exit 1;;
      --*)
        echo "mps_rollout: GEN_EXTRA_ARGS token '$tok' is not in the allowlist" >&2; exit 1;;
      *)
        # bare value (e.g. the number after --exact-budget-sh-min-n) is fine
        i=$((i+1));;
    esac
  done
}
validate_extra_args

export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe_host
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_host

mps_ensure() {
  if [ ! -e "$CUDA_MPS_PIPE_DIRECTORY/control" ]; then
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    nvidia-cuda-mps-control -d
    sleep 3
  fi
  echo get_server_list | nvidia-cuda-mps-control >/dev/null 2>&1 \
    && echo "mps daemon: up" || { echo "mps daemon FAILED"; exit 1; }
}

stop_gpu_generation() {
  local gpu="$1"
  # Digit boundary so gpu1 never matches gpu10-style out-dirs; log each
  # matched cmdline BEFORE killing (auditability of the explicit-PID rule).
  for pid in $(pgrep -f "out-dir .*gpu${gpu}([^0-9]|$)" || true); do
    echo "gpu$gpu: stopping parent $pid: $(ps -p "$pid" -o args= 2>/dev/null | head -c 160)"
    kill "$pid" 2>/dev/null || true
  done
  sleep 8
  for _pass in 1 2; do
    for pid in $(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader | tr -d ' '); do
      case "$(ps -p "$pid" -o comm= 2>/dev/null)" in *mps*) continue;; esac
      kill "$pid" 2>/dev/null || true
    done
    sleep 6
  done
  local left
  left=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader | grep -cv mps || true)
  echo "gpu$gpu cleared (non-mps procs left: $left)"
}

launch_gpu() {
  local gpu="$1" seed="$2"
  local out="${OUT_PREFIX}/gpu${gpu}"
  mkdir -p "$out"
  # CAT-132: append logs across restarts so prior stdout is not silently lost;
  # disown so a closing ssh session cannot SIGHUP the nohup'd job.
  CUDA_VISIBLE_DEVICES=$gpu nohup "${GEN_PY:-.venv/bin/python}" tools/generate_gumbel_selfplay_data.py \
    --out-dir "$out" --games "$GAMES" --workers "$WORKERS" \
    --base-seed "$seed" "${GEN_ARGS[@]}" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
    >> "$out/launch.log" 2>&1 &
  disown
  echo "gpu$gpu: launched pid $! seed=$seed workers=$WORKERS (MPS) -> $out"
}

launch_gpu_no_mps() {
  # Rollback launch: 8 workers, MPS env explicitly UNSET for this process
  # tree so workers never attach to the (possibly still-running) daemon.
  local gpu="$1" seed="$2"
  local out="${OUT_PREFIX}_rollback/gpu${gpu}"
  mkdir -p "$out"
  env -u CUDA_MPS_PIPE_DIRECTORY -u CUDA_MPS_LOG_DIRECTORY \
    CUDA_VISIBLE_DEVICES=$gpu nohup "${GEN_PY:-.venv/bin/python}" tools/generate_gumbel_selfplay_data.py \
    --out-dir "$out" --games "$GAMES" --workers 8 \
    --base-seed "$seed" "${GEN_ARGS[@]}" >> "$out/launch.log" 2>&1 &
  disown
  echo "gpu$gpu: ROLLED BACK to 8 workers no-MPS, pid $! seed=$seed -> $out"
}

case "${1:-}" in
  canary)
    gpu="${2:?gpu index}" ; seed="${3:?ledgered base seed}"
    mps_ensure
    stop_gpu_generation "$gpu"
    launch_gpu "$gpu" "$seed"
    echo "CANARY CHECKLIST: (1) after ~40 min compare rows/hr vs a non-MPS GPU's"
    echo "shard cadence (expect ~2.5-3x); (2) nvidia-smi -i $gpu memory.used stays"
    echo "<20GB at 16w; (3) pmon shows workers under one mps-server; (4) zero"
    echo "worker errors in $OUT_PREFIX/gpu$gpu/launch.log. Then roll host-wide."
    ;;
  host)
    mps_ensure
    for entry in ${2:?"space-separated gpu:seed list"}; do
      gpu="${entry%%:*}" ; seed="${entry##*:}"
      stop_gpu_generation "$gpu"
      launch_gpu "$gpu" "$seed"
    done
    ;;
  rollback)
    for entry in ${2:?"space-separated gpu:seed list"}; do
      gpu="${entry%%:*}" ; seed="${entry##*:}"
      stop_gpu_generation "$gpu"
      launch_gpu_no_mps "$gpu" "$seed"
    done
    echo "NOTE: daemon left running for any GPUs still under MPS; when none"
    echo "remain: CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY sh -c 'echo quit | nvidia-cuda-mps-control'"
    ;;
  status)
    echo get_server_list | nvidia-cuda-mps-control 2>/dev/null || echo "no daemon"
    for gpu in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
      n=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader | wc -l)
      echo "gpu$gpu compute procs: $n"
    done
    ;;
  *)
    echo "usage: $0 canary <gpu> <seed> | host \"<gpu:seed> ...\" | status" ; exit 1 ;;
esac
