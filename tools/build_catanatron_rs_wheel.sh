#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
OUT_DIR="${OUT_DIR:-$ROOT/dist}"

command -v maturin >/dev/null 2>&1 || {
  echo "maturin is required (pip install 'maturin>=1.8,<2')" >&2
  exit 1
}
command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
}

mkdir -p "$OUT_DIR"
cargo test \
  --manifest-path "$ROOT/native/catanatron-rs/Cargo.toml" \
  public_belief_determinization_tests \
  --lib
maturin build \
  --release \
  --manifest-path "$ROOT/native/catanatron-rs/python/Cargo.toml" \
  --interpreter "$PYTHON_BIN" \
  --out "$OUT_DIR"
sha256sum "$OUT_DIR"/catanatron_rs-0.1.4-*.whl
