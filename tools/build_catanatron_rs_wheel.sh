#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
OUT_DIR="${OUT_DIR:-$SOURCE_ROOT/dist}"
CANONICAL_BUILD_ROOT="${CATAN_RS_CANONICAL_BUILD_ROOT:-/tmp/catan-zero-catanatron-rs-wheel-src}"

# Cargo includes the canonical manifest path in crate metadata and maturin's
# SBOM records source paths.  A remap flag only affects debug information, so
# two otherwise-identical checkouts at different paths still produced
# different wheels.  Always release-build the committed tree from one fixed,
# locked path.  Besides making the bytes reproducible, git-archive prevents an
# uncommitted local patch from leaking into a supposedly tagged wheel.
if [ "${CATAN_RS_BUILD_STAGED:-0}" != "1" ]; then
  command -v git >/dev/null 2>&1 || {
    echo "git is required to stage the canonical source tree" >&2
    exit 1
  }
  command -v flock >/dev/null 2>&1 || {
    echo "flock is required to serialize the canonical wheel build" >&2
    exit 1
  }
  if ! git -C "$SOURCE_ROOT" diff --quiet --exit-code \
    || ! git -C "$SOURCE_ROOT" diff --cached --quiet --exit-code; then
    echo "refusing to build a release wheel from a dirty tracked tree" >&2
    exit 1
  fi
  SOURCE_COMMIT="$(git -C "$SOURCE_ROOT" rev-parse --verify HEAD)"
  mkdir -p "$(dirname "$CANONICAL_BUILD_ROOT")" "$OUT_DIR"
  exec 9>"${CANONICAL_BUILD_ROOT}.lock"
  flock 9
  rm -rf "$CANONICAL_BUILD_ROOT"
  mkdir -p "$CANONICAL_BUILD_ROOT"
  git -C "$SOURCE_ROOT" archive "$SOURCE_COMMIT" \
    | tar -x -C "$CANONICAL_BUILD_ROOT"
  CATAN_RS_BUILD_STAGED=1 \
    CATAN_RS_SOURCE_COMMIT="$SOURCE_COMMIT" \
    OUT_DIR="$OUT_DIR" \
    "$CANONICAL_BUILD_ROOT/tools/build_catanatron_rs_wheel.sh"
  exit $?
fi

ROOT="$SOURCE_ROOT"

# Reproducible release bytes. Wheel ZIP metadata otherwise inherits wall-clock
# time, and Rust diagnostics can retain the absolute checkout path. Keep the
# epoch version-scoped and stable across the follow-up commit that records the
# resulting digest.
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-1783641600}"
export CARGO_INCREMENTAL=0
export CARGO_BUILD_JOBS=1
export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }--remap-path-prefix=$ROOT=/src/catan-zero-public -C link-arg=-Wl,--build-id=none"

command -v maturin >/dev/null 2>&1 || {
  echo "maturin is required (pip install 'maturin>=1.8,<2')" >&2
  exit 1
}
command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
}

echo "source_commit=${CATAN_RS_SOURCE_COMMIT:-unknown}"
echo "canonical_build_root=$ROOT"
rustc --version
cargo --version
maturin --version
"$PYTHON_BIN" --version

mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR"/catanatron_rs-0.1.4-*.whl
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
