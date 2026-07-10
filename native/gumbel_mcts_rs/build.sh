#!/bin/bash
# Build and install the gumbel_mcts Rust extension.
#
# Prerequisites:
#   pip install maturin
#
# Usage:
#   bash build.sh          # build + install (develop mode)
#   bash build.sh release  # build + install (release mode, optimized)
#   bash build.sh clean    # remove build artifacts

set -e

MODE="${1:-release}"

case "$MODE" in
    release)
        echo "Building gumbel_mcts (release)..."
        maturin develop --release --strip
        echo "Done. Import with: import gumbel_mcts"
        ;;
    dev|develop)
        echo "Building gumbel_mcts (debug)..."
        maturin develop
        echo "Done. Import with: import gumbel_mcts"
        ;;
    clean)
        echo "Cleaning build artifacts..."
        rm -rf target/ Cargo.lock
        echo "Done."
        ;;
    *)
        echo "Usage: $0 {release|dev|clean}"
        exit 1
        ;;
esac
