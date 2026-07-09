#!/bin/bash
# SYSTEM_DESIGN_FINDINGS #4: Teacher (n128) generation uses shard_size=2048
# which is too large — no worker reaches 2048 rows for ~60+ minutes, delaying
# the first shards and the gen-5 v1 training start.
#
# This wrapper detects n-full >= 128 and automatically reduces shard_size to 512.
# For n-full >= 256 (probe), reduces to 256.
#
# Usage: bash teacher_shard_size_fix.sh [original command...]
# Or source it to get the SHARD_SIZE variable:
#   source teacher_shard_size_fix.sh
#   SHARD_SIZE=$(compute_shard_size 128)

compute_shard_size() {
    local n_full=${1:-64}
    if [ "$n_full" -ge 256 ]; then
        echo 256   # n256 probe: games are very slow, ~8x faster first shard
    elif [ "$n_full" -ge 128 ]; then
        echo 512   # n128 teacher: 4x faster first shard vs 2048
    else
        echo 2048  # n64 volume: games are fast, 2048 is fine
    fi
}

# If invoked with arguments, wrap the command
if [ $# -gt 0 ]; then
    # Extract --n-full value from the command args
    N_FULL=64
    for ((i=1; i<=$#; i++)); do
        if [ "${!i}" = "--n-full" ] && [ $((i+1)) -le $# ]; then
            next=$((i+1))
            N_FULL=${!next}
        fi
    done
    SHARD_SIZE=$(compute_shard_size "$N_FULL")
    echo "AUTO_SHARD_SIZE: n_full=$N_FULL -> shard_size=$SHARD_SIZE" >&2
    # Replace --shard-size value or add it if missing
    if echo "$@" | grep -q -- "--shard-size"; then
        # Replace existing --shard-size value
        exec "${@/--shard-size */--shard-size $SHARD_SIZE}"
    else
        # Add --shard-size
        exec "$@" --shard-size "$SHARD_SIZE"
    fi
fi
