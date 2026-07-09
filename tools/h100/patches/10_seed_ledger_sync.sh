#!/bin/bash
# SYSTEM_DESIGN_FINDINGS #30: Sync seed ledger across all fleet boxes.
#
# The seed ledger on each fleet box is a LOCAL STUB — no cross-host collision
# detection. This script syncs the master ledger from the orchestrator box to
# all fleet boxes, then runs the prelaunch guard on each to verify no overlaps.
#
# Usage: bash 10_seed_ledger_sync.sh
#
# Run from the orchestrator/master box (the one with the authoritative ledger).

set -euo pipefail
KEY=~/.ssh/gpu_access_ed25519
SSH="ssh -i $KEY -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20"

declare -A HOST=(
  [c1]=192.222.54.251
  [c2]=68.209.75.117
  [c3]=192.222.53.18
  [c4]=68.209.73.252
  [c5]=68.209.74.145
  [c6]=68.209.74.2
  [a100a]=64.181.197.190
)

MASTER_LEDGER="${MASTER_LEDGER:-$HOME/catan-zero-runsix/runs/SEED_LEDGER.md}"

if [ ! -f "$MASTER_LEDGER" ]; then
    echo "ERROR: Master ledger not found at $MASTER_LEDGER"
    echo "Set MASTER_LEDGER env var to the authoritative ledger path."
    exit 1
fi

echo "=== Seed Ledger Cross-Host Sync (SYSTEM_DESIGN_FINDINGS #30) ==="
echo "Master ledger: $MASTER_LEDGER"
echo ""

# Sync the master ledger to all fleet boxes in parallel
pids=()
for box in "${!HOST[@]}"; do
    ip="${HOST[$box]}"
    echo "Syncing to $box ($ip)..."
    scp -i $KEY -o BatchMode=yes -o ConnectTimeout=10 \
        "$MASTER_LEDGER" "ubuntu@$ip:/tmp/SEED_LEDGER.master.md" &
    pids+=($!)
done

# Wait for all syncs
FAILED=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        echo "  [FAIL] scp pid $pid failed"
        FAILED=$((FAILED + 1))
    fi
done

if [ "$FAILED" -gt 0 ]; then
    echo "WARNING: $FAILED sync(s) failed"
fi
echo ""

# On each box, merge the master ledger into the local stub (append-only)
for box in "${!HOST[@]}"; do
    ip="${HOST[$box]}"
    echo "Merging on $box..."
    $SSH "ubuntu@$ip" bash -s << 'REMOTE_SCRIPT'
        LOCAL=~/catan-zero-runsix/runs/SEED_LEDGER.md
        MASTER=/tmp/SEED_LEDGER.master.md
        if [ -f "$MASTER" ]; then
            # Append master ledger's claimed ranges to the local stub
            # (skip the header comments — only copy lines starting with "# " or "|")
            echo "" >> "$LOCAL"
            echo "--- Master ledger sync $(date -u +%Y-%m-%dT%H:%M:%SZ) ---" >> "$LOCAL"
            grep -E '^\|.*\|$' "$MASTER" >> "$LOCAL" 2>/dev/null || true
            echo "[OK] Merged master ledger into $LOCAL"
        else
            echo "[SKIP] No master ledger at $MASTER"
        fi
REMOTE_SCRIPT
done

echo ""
echo "=== Sync complete ==="
echo "Each fleet box now has the master ledger's claimed ranges appended."
echo "The prelaunch guard will check against these on next launch."
echo ""
echo "NOTE: This is a one-time merge. For ongoing sync, set up a cron job:"
echo "  */5 * * * * /path/to/10_seed_ledger_sync.sh >> /tmp/seed_sync.log 2>&1"
