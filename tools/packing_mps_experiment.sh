#!/usr/bin/env bash
# Speed-czar controlled experiment (team-lead approved 2026-07-06):
# worker-packing x CUDA-MPS on A100B GPU 0.
#
# Cells (cell_index -> ledgered seed base 69,000,000 + cell_index*100,000):
#   0 w1_off   1 worker,  MPS off, 10 min (uncontended per-eval baseline)
#   1 w8_off   8 workers, MPS off, 30 min (production baseline)
#   2 w12_off  3 w16_off  (MPS off)
#   4 w8_on    5 w12_on   6 w16_on  (MPS on)
#
# Games are REAL usable data (exact production recipe, gen2A ckpt, masked,
# lazy, temp-decisions 90); shard-size 512 instead of 2048 purely for finer
# cadence measurement granularity (same rows, same schema).
# After the grid: relaunch gpu0 production generation at seed 70,000,000
# (750 games, fresh dir, shard-size 2048) — done by this script's final step.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

GPU=0
CKPT=runs/bc/gen2A_20260706/checkpoint.pt
OUT_ROOT=runs/selfplay/packing_exp_20260706
CELL_MINUTES="${CELL_MINUTES:-30}"
SINGLE_MINUTES="${SINGLE_MINUTES:-10}"
GAMES_PER_CELL=4000
SEED_BASE=69000000
RELAUNCH_SEED=70000000
RELAUNCH_DIR=runs/selfplay/gen2a_spec2_20260706/a100b_gpu0_post_exp

GEN_ARGS=(
  --checkpoint "$CKPT" --device cuda
  --n-full 64 --n-fast 16 --p-full 0.25 --c-visit 50.0 --c-scale 0.03
  --max-decisions 600 --max-depth 80 --temperature-decisions 90
  --correct-rust-chance-spectra --lazy-interior-chance --public-observation
  --track 2p_no_trade --vps-to-win 10 --format npz
  --score-actions
)

gpu0_pids() {
  nvidia-smi -i $GPU --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '
}

sweep_gpu0() {
  # Kill any leftover compute procs on gpu0 (the spawn_main orphan trap), by
  # EXPLICIT PID from nvidia-smi -i 0 — never a pattern kill. The MPS server
  # process is exempted.
  for _pass in 1 2 3; do
    local pids
    pids=$(gpu0_pids)
    [ -z "$pids" ] && return 0
    for pid in $pids; do
      local comm
      comm=$(ps -p "$pid" -o comm= 2>/dev/null || true)
      case "$comm" in *mps*) continue;; esac
      kill "$pid" 2>/dev/null || true
    done
    sleep 8
    pids=$(gpu0_pids)
    for pid in $pids; do
      local comm
      comm=$(ps -p "$pid" -o comm= 2>/dev/null || true)
      case "$comm" in *mps*) continue;; esac
      kill -9 "$pid" 2>/dev/null || true
    done
    sleep 4
  done
}

mps_start() {
  export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe_$GPU
  export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_$GPU
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  CUDA_VISIBLE_DEVICES=$GPU nvidia-cuda-mps-control -d
  sleep 3
}

mps_stop() {
  if [ -n "${CUDA_MPS_PIPE_DIRECTORY:-}" ] && [ -e "$CUDA_MPS_PIPE_DIRECTORY" ]; then
    echo quit | nvidia-cuda-mps-control || true
    sleep 3
    unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
  fi
}
trap 'mps_stop' EXIT

run_cell() {
  local name="$1" workers="$2" cell_index="$3" minutes="$4"
  local seed=$((SEED_BASE + cell_index * 100000))
  local out="$OUT_ROOT/cell_$name"
  mkdir -p "$out"
  echo "=== cell $name: workers=$workers seed=$seed minutes=$minutes $(date -u +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES=$GPU timeout --signal=TERM --kill-after=30 "${minutes}m" \
    .venv/bin/python tools/generate_gumbel_selfplay_data.py \
    --out-dir "$out" --games "$GAMES_PER_CELL" --workers "$workers" \
    --base-seed "$seed" --shard-size 512 "${GEN_ARGS[@]}" > "$out/cell.log" 2>&1 || true
  sweep_gpu0
  .venv/bin/python - "$out" "$minutes" << 'PYEOF'
import sys, glob
import numpy as np
out, minutes = sys.argv[1], float(sys.argv[2])
rows = 0
games = set()
for f in glob.glob(f"{out}/worker_*/*.npz"):
    d = np.load(f, allow_pickle=True)
    rows += len(d["game_seed"]); games.update(d["game_seed"].tolist())
print(f"CELL {out}: rows={rows} distinct_games={len(games)} "
      f"rows_per_hr={rows/(minutes/60):.0f} games_per_hr~={rows/204.8/(minutes/60):.1f}",
      flush=True)
PYEOF
}

echo "pre-experiment gpu0 procs: $(gpu0_pids | tr '\n' ' ')"
sweep_gpu0
echo "post-sweep gpu0 procs: $(gpu0_pids | tr '\n' ' ') (must be empty)"

# --- MPS OFF -----------------------------------------------------------------
run_cell w1_off  1  0 "$SINGLE_MINUTES"
run_cell w8_off  8  1 "$CELL_MINUTES"
run_cell w12_off 12 2 "$CELL_MINUTES"
run_cell w16_off 16 3 "$CELL_MINUTES"

# --- MPS ON ------------------------------------------------------------------
mps_start
run_cell w8_on  8  4 "$CELL_MINUTES"
run_cell w12_on 12 5 "$CELL_MINUTES"
run_cell w16_on 16 6 "$CELL_MINUTES"
mps_stop

# --- relaunch production generation on gpu0 (approved: seed 70,000,000) ------
mkdir -p "$RELAUNCH_DIR"
CUDA_VISIBLE_DEVICES=$GPU nohup .venv/bin/python tools/generate_gumbel_selfplay_data.py \
  --out-dir "$RELAUNCH_DIR" --games 750 --workers 8 \
  --base-seed "$RELAUNCH_SEED" --shard-size 2048 "${GEN_ARGS[@]}" \
  > "$RELAUNCH_DIR/gpu0.log" 2>&1 &
echo "relaunched gpu0 production generation pid $! seed $RELAUNCH_SEED -> $RELAUNCH_DIR"
echo "EXPERIMENT COMPLETE $(date -u +%H:%M:%S)"
