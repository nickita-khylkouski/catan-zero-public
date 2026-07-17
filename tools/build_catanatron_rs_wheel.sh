#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
OUT_DIR="${OUT_DIR:-$SOURCE_ROOT/dist}"
SEALED_CANONICAL_BUILD_ROOT="/tmp/catan-zero-catanatron-rs-wheel-src"
CANONICAL_BUILD_ROOT="${CATAN_RS_CANONICAL_BUILD_ROOT:-$SEALED_CANONICAL_BUILD_ROOT}"
WHEEL_NAME="catanatron_rs-0.1.13-cp311-cp311-manylinux_2_34_x86_64.whl"
RECEIPT_NAME="catanatron_rs-0.1.13-build-receipt.json"
SEALED_SOURCE_DATE_EPOCH="1784160000"
SEALED_RUSTFLAGS="--remap-path-prefix=/tmp/catan-zero-catanatron-rs-wheel-src=/src/catan-zero-public -C link-arg=-Wl,--build-id=none"
SEALED_COMPILE_IDENTITY="catanatron-rs-0.1.13-dense-native-search-wheel-v1"

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
command -v strip >/dev/null 2>&1 || die "GNU strip is required"

RUSTC_VERSION="$(rustc --version)"
CARGO_VERSION="$(cargo --version)"
MATURIN_VERSION="$(maturin --version)"
PYTHON_VERSION="$("$PYTHON_BIN" --version 2>&1)"
STRIP_VERSION="$(strip --version | head -n 1)"
[ "$RUSTC_VERSION" = "rustc 1.96.1 (31fca3adb 2026-06-26)" ] \
  || die "unexpected rustc: $RUSTC_VERSION"
[ "$CARGO_VERSION" = "cargo 1.96.1 (356927216 2026-06-26)" ] \
  || die "unexpected cargo: $CARGO_VERSION"
[ "$MATURIN_VERSION" = "maturin 1.14.1" ] \
  || die "unexpected maturin: $MATURIN_VERSION"
[ "$PYTHON_VERSION" = "Python 3.11.15" ] \
  || die "unexpected Python: $PYTHON_VERSION"
[ "$STRIP_VERSION" = "GNU strip (GNU Binutils for Ubuntu) 2.38" ] \
  || die "unexpected strip: $STRIP_VERSION"

echo "source_commit=<bound-after-build>"
echo "source_tree=<bound-after-build>"
echo "canonical_build_root=$ROOT"
echo "$RUSTC_VERSION"
echo "$CARGO_VERSION"
echo "$MATURIN_VERSION"
echo "$PYTHON_VERSION"
echo "$STRIP_VERSION"

mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR"/catanatron_rs-0.1.13-*.whl
rm -f "$OUT_DIR/$RECEIPT_NAME"

# PyO3's Python-enabled Rust tests link libpython, unlike the final extension
# module.  uv-managed CPython installations can be relocated under a stable
# minor-version directory while sysconfig retains the original patch-version
# prefix.  Resolve the first real shared-library directory from the active
# interpreter instead of trusting that stale prefix, then scope it to the one
# test process that needs it.  This is test-runtime plumbing only: it is not
# exported to maturin or the release compilation.
PYTHON_TEST_LIBDIR="$($PYTHON_BIN - <<'PY'
import pathlib
import sys
import sysconfig

library = sysconfig.get_config_var("LDLIBRARY") or "libpython3.11.so"
candidates = []
configured = sysconfig.get_config_var("LIBDIR")
if configured:
    candidates.append(pathlib.Path(configured))
candidates.append(pathlib.Path(sys.base_prefix) / "lib")
for candidate in candidates:
    if candidate.is_absolute() and (candidate / library).is_file():
        print(candidate.resolve())
        break
else:
    raise SystemExit(f"cannot locate {library} for Python-enabled Rust tests")
PY
)"
[ -n "$PYTHON_TEST_LIBDIR" ] \
  || die "Python-enabled Rust test library directory resolved empty"
cargo test \
  --locked \
  --manifest-path "$ROOT/native/catanatron-rs/Cargo.toml" \
  public_belief_determinization_tests \
  --lib
cargo test \
  --locked \
  --manifest-path "$ROOT/native/catanatron-rs/Cargo.toml" \
  public_card_deductions \
  --lib
# Capability names are not evidence of semantics.  Run the exact source tests
# for every advertised corrected behavior before compiling and hashing the wheel.
LD_LIBRARY_PATH="$PYTHON_TEST_LIBDIR" \
RUSTFLAGS="$RUSTFLAGS -L native=$PYTHON_TEST_LIBDIR" \
cargo test \
  --locked \
  --manifest-path "$ROOT/native/catanatron-rs/Cargo.toml" \
  --features python \
  entity_player_tokens_preserve_public_awards_when_hidden_hands_are_masked \
  --lib
LD_LIBRARY_PATH="$PYTHON_TEST_LIBDIR" \
RUSTFLAGS="$RUSTFLAGS -L native=$PYTHON_TEST_LIBDIR" \
cargo test \
  --locked \
  --manifest-path "$ROOT/native/catanatron-rs/Cargo.toml" \
  --features python \
  entity_v6_preserves_exact_actor_resource_composition_and_total \
  --lib
LD_LIBRARY_PATH="$PYTHON_TEST_LIBDIR" \
RUSTFLAGS="$RUSTFLAGS -L native=$PYTHON_TEST_LIBDIR" \
cargo test \
  --locked \
  --manifest-path "$ROOT/native/catanatron-rs/Cargo.toml" \
  --features python \
  action_context_v6_initial_road_uses_legal_two_hop_settlement_sites \
  --lib
LD_LIBRARY_PATH="$PYTHON_TEST_LIBDIR" \
RUSTFLAGS="$RUSTFLAGS -L native=$PYTHON_TEST_LIBDIR" \
cargo test \
  --locked \
  --manifest-path "$ROOT/native/catanatron-rs/Cargo.toml" \
  --features python \
  meaningful_public_history_filters_plumbing_and_caps_at_32 \
  --lib
cargo test \
  --locked \
  --manifest-path "$ROOT/native/gumbel_mcts_rs/Cargo.toml" \
  --lib
cargo test \
  --locked \
  --manifest-path "$ROOT/native/gumbel_mcts_rs/Cargo.toml" \
  temperature \
  --lib
cargo test \
  --locked \
  --manifest-path "$ROOT/native/gumbel_mcts_rs/Cargo.toml" \
  coherent_public_belief_dev_chance_ignores_concrete_hidden_support \
  --lib
cargo test \
  --locked \
  --manifest-path "$ROOT/native/gumbel_mcts_rs/Cargo.toml" \
  forced_trajectory_only_selects_without_evaluator_or_fake_values \
  --lib
maturin build \
  --locked \
  --release \
  --manifest-path "$ROOT/native/catanatron-rs/python/Cargo.toml" \
  --interpreter "$PYTHON_BIN" \
  --out "$OUT_DIR"

WHEEL_PATH="$OUT_DIR/$WHEEL_NAME"
[ -f "$WHEEL_PATH" ] || die "expected wheel was not produced: $WHEEL_PATH"

# ThinLTO emits process/source-dependent `.llvm.<number>` names only in the
# non-runtime ELF symbol/string tables. GNU strip removes those tables and the
# resulting shared object is byte-identical across the observed variants. Then
# rebuild the wheel in stable path order with a fresh standards-compliant
# RECORD and fixed ZIP metadata. Runtime/loadable sections are unchanged.
NORMALIZE_TMP="$(mktemp -d "${TMPDIR:-/tmp}/catan-rs-wheel-normalize.XXXXXXXX")"
trap 'rm -rf -- "$NORMALIZE_TMP"' EXIT
"$PYTHON_BIN" - "$WHEEL_PATH" "$NORMALIZE_TMP" <<'PY'
import pathlib
import sys
import zipfile

wheel = pathlib.Path(sys.argv[1])
root = pathlib.Path(sys.argv[2])
with zipfile.ZipFile(wheel) as archive:
    for name in archive.namelist():
        path = pathlib.PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise SystemExit(f"unsafe wheel member: {name}")
    archive.extractall(root)
PY
mapfile -t SHARED_OBJECTS < <(find "$NORMALIZE_TMP" -type f -name '*.so' -print)
[ "${#SHARED_OBJECTS[@]}" -eq 1 ] \
  || die "expected exactly one shared object in wheel, found ${#SHARED_OBJECTS[@]}"
strip --strip-unneeded "${SHARED_OBJECTS[0]}"
"$PYTHON_BIN" - "$WHEEL_PATH" "$NORMALIZE_TMP" "$SOURCE_DATE_EPOCH" <<'PY'
import base64
import csv
import hashlib
import io
import os
import pathlib
import sys
import time
import zipfile

wheel = pathlib.Path(sys.argv[1])
root = pathlib.Path(sys.argv[2])
epoch = int(sys.argv[3])
files = sorted(path for path in root.rglob("*") if path.is_file())
records = [path for path in files if path.as_posix().endswith(".dist-info/RECORD")]
if len(records) != 1:
    raise SystemExit(f"expected one RECORD, found {len(records)}")
record = records[0]
rows: list[list[str]] = []
for path in files:
    relative = path.relative_to(root).as_posix()
    if path == record:
        continue
    data = path.read_bytes()
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    rows.append([relative, f"sha256={digest}", str(len(data))])
rows.append([record.relative_to(root).as_posix(), "", ""])
buffer = io.StringIO(newline="")
csv.writer(buffer, lineterminator="\n").writerows(rows)
record.write_text(buffer.getvalue(), encoding="utf-8")

normalized = wheel.with_name(f".{wheel.name}.normalized-{os.getpid()}")
timestamp = time.gmtime(epoch)[:6]
with zipfile.ZipFile(
    normalized, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
) as archive:
    for path in sorted(path for path in root.rglob("*") if path.is_file()):
        relative = path.relative_to(root).as_posix()
        info = zipfile.ZipInfo(relative, date_time=timestamp)
        info.create_system = 3
        info.compress_type = zipfile.ZIP_DEFLATED
        mode = 0o755 if path.suffix == ".so" else 0o644
        info.external_attr = mode << 16
        archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
os.replace(normalized, wheel)
PY

# Validate the normalized extension itself, not just the Rust source or wheel
# filename. This catches stale same-version artifacts before their digest can be
# recorded in the release inventory.
PYTHONPATH="$NORMALIZE_TMP" "$PYTHON_BIN" - <<'PY'
import json
from importlib.metadata import version

import catanatron_rs

assert version("catanatron-rs") == "0.1.13"
game = catanatron_rs.Game.simple(["RED", "BLUE"], seed=7)
observer = game.current_color()
public_cards = json.loads(game.public_card_deductions_json(observer))
assert public_cards["contract"] == "public_card_deductions_2p_v1", public_cards
assert public_cards["resource_composition_exact"] is True, public_cards
assert public_cards["development_composition_exact"] is False, public_cards
capability_fn = getattr(catanatron_rs, "gumbel_search_capabilities", None)
assert callable(capability_fn), "wheel lacks gumbel_search_capabilities"
capabilities = set(capability_fn())
required = {
    "sigma_reference_visits",
    "belief_target_evidence",
    "initial_road_d1_scope",
    "public_award_feature_parity",
    "entity_feature_adapter_version",
    "policy_temperature_semantics",
    "coherent_public_belief_search",
    "boundary_value_particles",
    "forced_root_trajectory_only",
}
assert required <= capabilities, (required, capabilities)
context_adapter_fn = getattr(
    catanatron_rs, "supported_action_context_adapter_versions", None
)
assert callable(context_adapter_fn), "wheel lacks versioned action-context support"
assert (
    "rust_entity_adapter_v6_exact_actor_resources_initial_road_two_hop"
    in set(context_adapter_fn())
), "wheel lacks complete adapter-v6 native context support"
PY

WHEEL_SHA256="$(sha256sum "$WHEEL_PATH" | awk '{print $1}')"
BUILDER_SHA256="$(sha256sum "$ROOT/tools/build_catanatron_rs_wheel.sh" | awk '{print $1}')"
CARGO_LOCK_SHA256="$(sha256sum "$ROOT/native/catanatron-rs/Cargo.lock" | awk '{print $1}')"
PYTHON_CARGO_LOCK_SHA256="$(sha256sum "$ROOT/native/catanatron-rs/python/Cargo.lock" | awk '{print $1}')"
GUMBEL_CARGO_LOCK_SHA256="$(sha256sum "$ROOT/native/gumbel_mcts_rs/Cargo.lock" | awk '{print $1}')"
GUMBEL_LIB_RS_SHA256="$(sha256sum "$ROOT/native/gumbel_mcts_rs/src/lib.rs" | awk '{print $1}')"
GUMBEL_PYTHON_BINDING_RS_SHA256="$(sha256sum "$ROOT/native/gumbel_mcts_rs/src/python_binding.rs" | awk '{print $1}')"

"$PYTHON_BIN" - "$OUT_DIR/$RECEIPT_NAME" <<PY
import json
import pathlib
import sys

receipt = {
    "schema_version": "catanatron-rs-wheel-build-receipt-v2",
    "source_commit": None,
    "source_tree": None,
    "builder_sha256": "$BUILDER_SHA256",
    "cargo_lock_sha256": "$CARGO_LOCK_SHA256",
    "python_cargo_lock_sha256": "$PYTHON_CARGO_LOCK_SHA256",
    "gumbel_cargo_lock_sha256": "$GUMBEL_CARGO_LOCK_SHA256",
    "gumbel_lib_rs_sha256": "$GUMBEL_LIB_RS_SHA256",
    "gumbel_python_binding_rs_sha256": "$GUMBEL_PYTHON_BINDING_RS_SHA256",
    "rustc_version": "$RUSTC_VERSION",
    "cargo_version": "$CARGO_VERSION",
    "maturin_version": "$MATURIN_VERSION",
    "python_version": "$PYTHON_VERSION",
    "strip_version": "$STRIP_VERSION",
    "elf_normalization": "strip--strip-unneeded+deterministic-wheel-v1",
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
