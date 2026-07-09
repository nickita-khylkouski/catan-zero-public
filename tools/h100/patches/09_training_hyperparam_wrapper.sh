#!/bin/bash
# SYSTEM_DESIGN_FINDINGS #7, #22, #23: Training hyperparameter recommendations.
#
# Finding #7: No LR decay. The code has --lr-schedule cosine but it's not used.
# Finding #22: Batch size 1024 may be too small for 35M model on 80GB A100.
# Finding #23: No gradient accumulation on single-GPU training.
#
# This is a TRAINING LAUNCH WRAPPER that adds recommended flags:
#   --lr-schedule cosine (Finding #7)
#   --batch-size 4096 (Finding #22, if on A100/H100 with enough memory)
#   --grad-accum-steps 4 (Finding #23, if batch-size is memory-limited)
#
# Usage: bash 09_training_hyperparam_wrapper.sh python tools/train_bc.py [args...]
#
# Or use the recommended launch commands below directly.

set -euo pipefail

# Recommended training commands (copy-paste ready):

cat << 'EOF'
=== RECOMMENDED TRAINING LAUNCH COMMANDS ===

# A100 (80GB) — single GPU, 35M model:
# Finding #22: batch-size 4096 (was 1024, ~50GB estimated, fits in 80GB)
# Finding #7:  cosine LR decay (was flat)
python tools/train_bc.py \
  --arch entity_graph \
  --init-checkpoint runs/bc/gen3_20260706/checkpoint.pt \
  --data ~/corpora/gen5_pooled \
  --data-format memmap \
  --batch-size 4096 \
  --lr 2e-4 \
  --lr-schedule cosine \
  --epochs 2 \
  --checkpoint runs/bc/gen5_rebalanced.pt \
  --policy-loss-weight 1.0 \
  --value-loss-weight 0.5 \
  --final-vp-loss-weight 0.05 \
  --winner-sample-weight 1.0 \
  --loser-sample-weight 0.5 \
  --mask-hidden-info \
  --track 2p_no_trade --vps-to-win 10

# H100 (80GB) — FSDP 4 GPUs, 35M model:
# Finding #22+23: batch-size 2048 per GPU × 4 GPUs = 8192 effective
# Finding #7: cosine LR decay
torchrun --nproc_per_node=4 tools/train_bc.py \
  --arch entity_graph \
  --init-checkpoint runs/bc/gen3_20260706/checkpoint.pt \
  --data ~/corpora/gen5_pooled \
  --data-format memmap \
  --batch-size 2048 \
  --grad-accum-steps 1 \
  --lr 2e-4 \
  --lr-schedule cosine \
  --epochs 2 \
  --checkpoint runs/bc/gen5_fsdp.pt \
  --policy-loss-weight 1.0 \
  --value-loss-weight 0.5 \
  --final-vp-loss-weight 0.05 \
  --winner-sample-weight 1.0 \
  --loser-sample-weight 0.5 \
  --mask-hidden-info \
  --track 2p_no_trade --vps-to-win 10

# Memory-limited fallback (Finding #23):
# If batch-size 4096 OOMs, use 1024 + grad-accum-steps 4 = 4096 effective
python tools/train_bc.py \
  ... --batch-size 1024 --grad-accum-steps 4 ...

=== KEY CHANGES FROM CURRENT ===
1. --lr-schedule cosine (was flat/implicit)          [Finding #7]
2. --batch-size 4096 (was 1024)                      [Finding #22]
3. --value-loss-weight 0.5 (was 0.25)                [Finding #6]
4. --loser-sample-weight 0.5 (was 0.3)               [Finding #6]
5. --grad-accum-steps 4 (if memory-limited)          [Finding #23]
EOF

# If invoked with arguments, wrap the command
if [ $# -gt 0 ]; then
    ARGS="$*"
    EXTRA=""

    # Add --lr-schedule cosine if not present
    if ! echo "$ARGS" | grep -q -- "--lr-schedule"; then
        EXTRA="$EXTRA --lr-schedule cosine"
    fi

    # Add --value-loss-weight 0.5 if not present
    if ! echo "$ARGS" | grep -q -- "--value-loss-weight"; then
        EXTRA="$EXTRA --value-loss-weight 0.5"
    fi

    # Add --loser-sample-weight 0.5 if not present
    if ! echo "$ARGS" | grep -q -- "--loser-sample-weight"; then
        EXTRA="$EXTRA --loser-sample-weight 0.5"
    fi

    if [ -n "$EXTRA" ]; then
        echo "AUTO_TRAINING_FLAGS: adding$EXTRA" >&2
    fi
    exec "$@" $EXTRA
fi
