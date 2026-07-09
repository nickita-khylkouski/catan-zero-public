#!/bin/bash
# SYSTEM_DESIGN_FINDINGS #5, #18: Enable npz_zst compression + rust-featurize.
#
# Finding #5: Shards use np.savez (uncompressed, 43.7MB each). The code already
# supports --format npz_zst (zstd compression) but it's not used. Switching
# saves ~3x disk space with negligible CPU cost.
#
# Finding #18: JSON serialization is ~1.3ms/leaf (50-70% of leaf-eval latency).
# The --rust-featurize flag bypasses the JSON round-trip by doing featurization
# in Rust. The code exists but is not enabled in production.
#
# This is a LAUNCH WRAPPER — it adds --format npz_zst and --rust-featurize to
# any generate_gumbel_selfplay_data.py command if they're not already present.
#
# Usage: bash 08_compression_rustfeaturize_wrapper.sh python tools/generate_gumbel_selfplay_data.py [args...]
# Or source it to get the variables:
#   source 08_compression_rustfeaturize_wrapper.sh

set -euo pipefail

# If sourced, just export the helper
if [ "${BASH_SOURCE[0]}" != "${0}" ]; then
    echo "Sourced. Use: add_gen_flags <command...>"
    return 0
fi

# If invoked with arguments, wrap the command
if [ $# -gt 0 ]; then
    ARGS="$*"
    EXTRA=""

    # Add --format npz_zst if not present
    if ! echo "$ARGS" | grep -q -- "--format"; then
        EXTRA="$EXTRA --format npz_zst"
    fi

    # Add --rust-featurize if not present
    if ! echo "$ARGS" | grep -q -- "--rust-featurize"; then
        EXTRA="$EXTRA --rust-featurize"
    fi

    if [ -n "$EXTRA" ]; then
        echo "AUTO_FLAGS: adding$EXTRA" >&2
    fi
    exec "$@" $EXTRA
fi

echo "Usage: bash 08_compression_rustfeaturize_wrapper.sh <command> [args...]"
echo "  Automatically adds --format npz_zst and --rust-featurize if not present."
