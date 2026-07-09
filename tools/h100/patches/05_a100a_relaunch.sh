#!/bin/bash
# SYSTEM_DESIGN_FINDINGS #12, #13, #14: a100a pilot relaunch with CORRECT flags.
#
# The current a100a pilot uses the OLD catan-zero stack with c-scale=0.1 (the
# known-broken pre-F1a/F1b calibration), workers=4, missing critical flags, and
# leaves GPU6 idle. This script fixes ALL of those issues.
#
# BEFORE RUNNING: Kill the existing pilot first:
#   pkill -f "cat91_n64_pilot"
#
# This script uses the PRODUCTION catan-zero-runsix stack (must be cloned to a100a)
# with the full flag set matching the H100 fleet.
#
# Usage: bash a100a_relaunch.sh [GAMES_PER_GPU] [BASE_SEED]
#   GAMES_PER_GPU defaults to 1000
#   BASE_SEED defaults to 6100000000 (a100a's assigned block)

set -euo pipefail

GAMES=${1:-1000}
BASE_SEED=${2:-6100000000}
CHECKPOINT="${CHECKPOINT:-/home/ubuntu/catan-zero/runs/bc/gen3_20260706/checkpoint.pt}"
# REPO should point to a clone of catan-zero-runsix on a100a
REPO="${REPO:-/home/ubuntu/catan-zero-runsix}"
OUTDIR="${OUTDIR:-/home/ubuntu/gen_out/a100a_pilot}"

cd "$REPO"
source .venv/bin/activate 2>/dev/null || true

echo "=== a100a pilot relaunch (SYSTEM_DESIGN_FINDINGS #12/#13/#14) ==="
echo "Games per GPU: $GAMES"
echo "Base seed: $BASE_SEED"
echo "Checkpoint: $CHECKPOINT"
echo "Output: $OUTDIR"
echo "GPUs: 0-7 (all 8 A100s)"
echo "Workers per GPU: 16 (was 4)"
echo ""

# Launch on all 8 GPUs (was only 6, GPU6 was idle)
for gpu in 0 1 2 3 4 5 6 7; do
    OUT="$OUTDIR/gpu$gpu"
    mkdir -p "$OUT"
    SEED=$((BASE_SEED + gpu * 100000))
    echo "Launching GPU $gpu (seed=$SEED, games=$GAMES)..."
    CUDA_VISIBLE_DEVICES=$gpu nohup python tools/generate_gumbel_selfplay_data.py \
        --checkpoint "$CHECKPOINT" \
        --out-dir "$OUT" \
        --games "$GAMES" \
        --workers 16 \
        --base-seed "$SEED" \
        --device cuda \
        --n-full 64 --n-fast 16 --p-full 0.25 \
        --c-visit 50.0 --c-scale 0.03 \
        --max-decisions 600 --max-depth 80 --temperature-decisions 90 \
        --correct-rust-chance-spectra --lazy-interior-chance \
        --public-observation \
        --track 2p_no_trade --vps-to-win 10 \
        --shard-size 2048 --format npz --score-actions \
        > "$OUT/launch.log" 2>&1 &
    echo "  PID: $!"
done

echo ""
echo "All 8 GPUs launched. Monitor with:"
echo "  watch -n5 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader'"
echo "  tail -f $OUTDIR/gpu0/launch.log"
