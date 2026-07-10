#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
OUT_DIR="${OUT_DIR:-$SOURCE_ROOT/dist}"
SEALED_CANONICAL_BUILD_ROOT="/tmp/catan-zero-catanatron-rs-wheel-src"
CANONICAL_BUILD_ROOT="${CATAN_RS_CANONICAL_BUILD_ROOT:-$SEALED_CANONICAL_BUILD_ROOT}"
WHEEL_NAME="catanatron_rs-0.1.4-cp311-cp311-manylinux_2_34_x86_64.whl"
RECEIPT_NAME="catanatron_rs-0.1.4-build-receipt.json"
SEALED_SOURCE_DATE_EPOCH="1783641600"
SEALED_RUSTFLAGS="--remap-path-prefix=/tmp/catan-zero-catanatron-rs-wheel-src=/src/catan-zero-public -C link-arg=-Wl,--build-id=none"
SEALED_COMPILE_IDENTITY="catanatron-rs-0.1.4-infoset-wheel-v1"

die() {
  echo "build_catanatron_rs_wheel: $*" >&2
  exit 1
}

# Cargo includes the canonical manifest path in crate metadata and maturin's
# SBOM records source paths.  A remap flag only affects debug information, so
# two otherwise-identical checkouts at different paths still produced
# different wheels.  Always release-build the committed tree from one fixed,
# locked path.  Besides making the bytes reproducible, git-archive prevents an
# uncommitted local patch from leaking into a supposedly tagged wheel.
if [ "${CATAN_RS_BUILD_STAGED:-0}" != "1" ]; then
  [ "$CANONICAL_BUILD_ROOT" = "$SEALED_CANONICAL_BUILD_ROOT" ] \
    || die "CATAN_RS_CANONICAL_BUILD_ROOT must equal the sealed release path $SEALED_CANONICAL_BUILD_ROOT"
  case "$CANONICAL_BUILD_ROOT" in
    /*/catan-zero-catanatron-rs-wheel-src) ;;
    *) die "CATAN_RS_CANONICAL_BUILD_ROOT must be an absolute, dedicated catan-zero-catanatron-rs-wheel-src directory" ;;
  esac
  [ "$CANONICAL_BUILD_ROOT" != "$SOURCE_ROOT" ] \
    || die "canonical build root must not equal the source root"
  [ "$CANONICAL_BUILD_ROOT" != "$OUT_DIR" ] \
    || die "canonical build root must not equal the output directory"
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
  SOURCE_TREE="$(git -C "$SOURCE_ROOT" rev-parse --verify 'HEAD^{tree}')"
  mkdir -p "$(dirname "$CANONICAL_BUILD_ROOT")" "$OUT_DIR"
  exec 9>"${CANONICAL_BUILD_ROOT}.lock"
  flock 9
  rm -rf "$CANONICAL_BUILD_ROOT"
  mkdir -p "$CANONICAL_BUILD_ROOT"
  git -C "$SOURCE_ROOT" archive "$SOURCE_COMMIT" \
    | tar -x -C "$CANONICAL_BUILD_ROOT"
  # The inventory binds the final release artifact but is deliberately not a
  # native build input. This makes the follow-up checksum-only commit
  # non-recursive, which is verified by rebuilding the final tagged tree.
  rm -f "$CANONICAL_BUILD_ROOT/native/catanatron-rs/WHEEL_SHA256SUMS"
  # Strip ambient compiler/build variables. Only this explicit allowlist may
  # influence the release build; the inner stage seals the remaining flags.
  env -i \
    HOME="$HOME" \
    USER="${USER:-builder}" \
    PATH="$PATH" \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=UTC \
    PYTHONHASHSEED=0 \
    PYTHON_BIN="$PYTHON_BIN" \
    CATAN_RS_BUILD_STAGED=1 \
    CATAN_RS_SOURCE_COMMIT="$SEALED_COMPILE_IDENTITY" \
    CATAN_RS_SOURCE_TREE="$SEALED_COMPILE_IDENTITY" \
    OUT_DIR="$OUT_DIR" \
    "$CANONICAL_BUILD_ROOT/tools/build_catanatron_rs_wheel.sh"
  # Real source identity is release evidence, not a compiler input. Cargo/LLVM
  # nevertheless uses these environment keys in its crate disambiguator. They
  # therefore carry a fixed, version-scoped compile salt above; bind the actual
  # commit/tree into the already-built external receipt only after compilation.
  "$PYTHON_BIN" - "$OUT_DIR/$RECEIPT_NAME" "$SOURCE_COMMIT" "$SOURCE_TREE" <<'PY'
import json
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text())
if payload.get("source_commit") is not None or payload.get("source_tree") is not None:
    raise SystemExit("inner build receipt unexpectedly contains source identity")
payload["source_commit"] = sys.argv[2]
payload["source_tree"] = sys.argv[3]
temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
with temporary.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, path)
PY
  echo "build_receipt_final=$OUT_DIR/$RECEIPT_NAME"
  exit $?
fi

ROOT="$SOURCE_ROOT"

# Reproducible release bytes. Wheel ZIP metadata otherwise inherits wall-clock
# time, and Rust diagnostics can retain the absolute checkout path. Keep the
# epoch version-scoped and stable across the follow-up commit that records the
# resulting digest.
if [ -n "${SOURCE_DATE_EPOCH:-}" ] && [ "$SOURCE_DATE_EPOCH" != "$SEALED_SOURCE_DATE_EPOCH" ]; then
  die "SOURCE_DATE_EPOCH override does not match the sealed release value"
fi
if [ -n "${RUSTFLAGS:-}" ] && [ "$RUSTFLAGS" != "$SEALED_RUSTFLAGS" ]; then
  die "RUSTFLAGS override does not match the sealed release value"
fi
export SOURCE_DATE_EPOCH="$SEALED_SOURCE_DATE_EPOCH"
export CARGO_INCREMENTAL=0
export CARGO_BUILD_JOBS=1
export RUSTFLAGS="$SEALED_RUSTFLAGS"

command -v maturin >/dev/null 2>&1 || {
  echo "maturin is required (pip install 'maturin>=1.8,<2')" >&2
  exit 1
}
command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
}

RUSTC_VERSION="$(rustc --version)"
CARGO_VERSION="$(cargo --version)"
MATURIN_VERSION="$(maturin --version)"
PYTHON_VERSION="$("$PYTHON_BIN" --version 2>&1)"
[ "$RUSTC_VERSION" = "rustc 1.96.1 (31fca3adb 2026-06-26)" ] \
  || die "unexpected rustc: $RUSTC_VERSION"
[ "$CARGO_VERSION" = "cargo 1.96.1 (356927216 2026-06-26)" ] \
  || die "unexpected cargo: $CARGO_VERSION"
[ "$MATURIN_VERSION" = "maturin 1.14.1" ] \
  || die "unexpected maturin: $MATURIN_VERSION"
[ "$PYTHON_VERSION" = "Python 3.11.15" ] \
  || die "unexpected Python: $PYTHON_VERSION"

echo "source_commit=<bound-after-build>"
echo "source_tree=<bound-after-build>"
echo "canonical_build_root=$ROOT"
echo "$RUSTC_VERSION"
echo "$CARGO_VERSION"
echo "$MATURIN_VERSION"
echo "$PYTHON_VERSION"

mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR"/catanatron_rs-0.1.4-*.whl
rm -f "$OUT_DIR/$RECEIPT_NAME"
cargo test \
  --locked \
  --manifest-path "$ROOT/native/catanatron-rs/Cargo.toml" \
  public_belief_determinization_tests \
  --lib
maturin build \
  --locked \
  --release \
  --manifest-path "$ROOT/native/catanatron-rs/python/Cargo.toml" \
  --interpreter "$PYTHON_BIN" \
  --out "$OUT_DIR"

WHEEL_PATH="$OUT_DIR/$WHEEL_NAME"
[ -f "$WHEEL_PATH" ] || die "expected wheel was not produced: $WHEEL_PATH"
WHEEL_SHA256="$(sha256sum "$WHEEL_PATH" | awk '{print $1}')"
BUILDER_SHA256="$(sha256sum "$ROOT/tools/build_catanatron_rs_wheel.sh" | awk '{print $1}')"
CARGO_LOCK_SHA256="$(sha256sum "$ROOT/native/catanatron-rs/Cargo.lock" | awk '{print $1}')"
PYTHON_CARGO_LOCK_SHA256="$(sha256sum "$ROOT/native/catanatron-rs/python/Cargo.lock" | awk '{print $1}')"

"$PYTHON_BIN" - "$OUT_DIR/$RECEIPT_NAME" <<PY
import json
import pathlib
import sys

receipt = {
    "schema_version": "catanatron-rs-wheel-build-receipt-v1",
    "source_commit": None,
    "source_tree": None,
    "builder_sha256": "$BUILDER_SHA256",
    "cargo_lock_sha256": "$CARGO_LOCK_SHA256",
    "python_cargo_lock_sha256": "$PYTHON_CARGO_LOCK_SHA256",
    "rustc_version": "$RUSTC_VERSION",
    "cargo_version": "$CARGO_VERSION",
    "maturin_version": "$MATURIN_VERSION",
    "python_version": "$PYTHON_VERSION",
    "canonical_build_root": "$ROOT",
    "compile_identity": "$SEALED_COMPILE_IDENTITY",
    "source_date_epoch": int("$SOURCE_DATE_EPOCH"),
    "rustflags": "$RUSTFLAGS",
    "cargo_build_jobs": int("$CARGO_BUILD_JOBS"),
    "checksum_inventory_excluded": True,
    "wheel_filename": "$WHEEL_NAME",
    "wheel_sha256": "$WHEEL_SHA256",
}
path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
PY

sha256sum "$WHEEL_PATH"
echo "build_receipt=$OUT_DIR/$RECEIPT_NAME"
