#!/usr/bin/env bash
# ============================================================================
# catan-zero frozen-release installer (CAT-117)
# Single deploy path for BOTH arches (H100 cu128 / A100 cu128). Replaces the
# non-git tarball snapshot layout with: git clone + checkout the frozen tag +
# a reproducible venv. An installed/dirty destination is intentionally refused;
# upgrades stage into a fresh destination instead of mutating a live runtime.
#
#   CATAN_REF=<published-h100-release> bash tools/install_v1_freeze.sh
#
# Overridable via env:
#   CATAN_REPO   git URL, OR a local git-bundle path (default GitHub PUBLIC repo
#                nickita-khylkouski/catan-zero-public; a bundle file works as-is
#                with `git clone` for an offline/air-gapped fallback)
#   CATAN_REF    REQUIRED immutable release tag to deploy. A commit SHA is also
#                accepted only when CATAN_RS_WHEEL names an already-staged wheel;
#                GitHub's automatic wheel fetch is release-tag based. There is no
#                stale fallback: v1.0-deploy predates the H100 hardening.
#   CATAN_DEST   fresh checkout dir (default ~/catan-zero-v1; an existing venv,
#                dirty checkout, or non-empty non-git directory is refused)
#   CATAN_RS_WHEEL  catanatron_rs 0.1.4 cp311 manylinux wheel (pip can't fetch it;
#                if unset/absent, auto-downloaded from the CATAN_REF release assets)
#   TORCH_INDEX  torch wheel index (default cu128)
#   PY           python interpreter (default python3.11; 3.11 REQUIRED). If
#                $PY isn't found on PATH (e.g. H100 canaries ship python3.10
#                only), it's bootstrapped via `uv` (installing uv itself first
#                if needed) — no sudo required; boxes that already have 3.11
#                (B200, legacy A100) are untouched.
# ============================================================================
set -euo pipefail

CATAN_REPO="${CATAN_REPO:-https://github.com/nickita-khylkouski/catan-zero-public}"
CATAN_REF="${CATAN_REF:-}"
CATAN_DEST="${CATAN_DEST:-$HOME/catan-zero-v1}"
CATAN_RS_WHEEL="${CATAN_RS_WHEEL:-$HOME/bundle/catanatron_rs-0.1.4-cp311-cp311-manylinux_2_34_x86_64.whl}"
RS_WHEEL_NAME="catanatron_rs-0.1.4-cp311-cp311-manylinux_2_34_x86_64.whl"
RS_WHEEL_SHA256_FILE_REL="native/catanatron-rs/WHEEL_SHA256SUMS"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
PY="${PY:-python3.11}"

INSTALL_TMP=""
cleanup_install_tmp() {
  if [ -n "$INSTALL_TMP" ] && [ -d "$INSTALL_TMP" ]; then
    rm -rf -- "$INSTALL_TMP"
  fi
}
trap cleanup_install_tmp EXIT
trap 'exit 130' INT TERM HUP

die() {
  echo "[install] ERROR: $*" >&2
  exit 3
}

if [ -z "$CATAN_REF" ]; then
  echo "[install] ERROR: CATAN_REF is required; v1.0-deploy predates the current H100 launcher/lifecycle fixes." >&2
  echo "          Publish this verified tree as an immutable release tag, then rerun with CATAN_REF=<that-tag>." >&2
  exit 2
fi

echo "[install] repo=$CATAN_REPO ref=$CATAN_REF dest=$CATAN_DEST py=$PY"

# 1. Clone (or update) and resolve the requested ref to one exact commit.  A
# deployment checkout is immutable input, not a working directory: refuse
# tracked/untracked drift and stale virtual environments instead of silently
# erasing or reusing them.  Operators upgrading an installed box must stage a
# fresh destination and switch it only after this installer succeeds.
if [ -d "$CATAN_DEST/.git" ]; then
  if [ -e "$CATAN_DEST/.venv" ] || [ -L "$CATAN_DEST/.venv" ]; then
    die "deployment checkout already contains .venv: $CATAN_DEST (use a fresh CATAN_DEST)"
  fi
  if [ -n "$(git -C "$CATAN_DEST" status --porcelain --untracked-files=all)" ] \
    || [ -n "$(git -C "$CATAN_DEST" clean -ndx)" ]; then
    die "deployment checkout is dirty: $CATAN_DEST (use a fresh CATAN_DEST)"
  fi
  git -C "$CATAN_DEST" fetch --tags --force origin
else
  if [ -e "$CATAN_DEST" ] && [ -n "$(find "$CATAN_DEST" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
    die "CATAN_DEST exists and is not an empty git checkout: $CATAN_DEST"
  fi
  git clone "$CATAN_REPO" "$CATAN_DEST"
fi
git -C "$CATAN_DEST" fetch --tags --force origin
if ! REF_COMMIT="$(git -C "$CATAN_DEST" rev-parse --verify "${CATAN_REF}^{commit}" 2>/dev/null)"; then
  die "CATAN_REF does not resolve to a commit: $CATAN_REF"
fi
git -C "$CATAN_DEST" checkout --detach --force "$REF_COMMIT"
cd "$CATAN_DEST"
HEAD_COMMIT="$(git rev-parse --verify HEAD)"
if [ "$HEAD_COMMIT" != "$REF_COMMIT" ]; then
  die "checked-out HEAD does not match resolved CATAN_REF: head=$HEAD_COMMIT ref=$REF_COMMIT"
fi
if [ -n "$(git status --porcelain --untracked-files=all)" ] \
  || [ -n "$(git clean -ndx)" ]; then
  die "deployment checkout drifted while resolving CATAN_REF: $CATAN_DEST"
fi

REF_KIND="commit"
TAG_COMMIT=""
if git show-ref --verify --quiet "refs/tags/$CATAN_REF"; then
  REF_KIND="tag"
  TAG_COMMIT="$(git rev-parse --verify "refs/tags/${CATAN_REF}^{commit}")"
  if [ "$TAG_COMMIT" != "$HEAD_COMMIT" ]; then
    die "tag $CATAN_REF resolves to $TAG_COMMIT but checked-out HEAD is $HEAD_COMMIT"
  fi
fi
echo "[install] checked out ref_kind=$REF_KIND $(git describe --tags --always) @ $HEAD_COMMIT"

# 2. Acquire and verify the exact Rust wheel before *any* sudo, systemd, venv,
# torch, or pip mutation.  Always install a private verified copy so a staged
# source wheel cannot change between hashing and pip consumption.
INSTALL_TMP="$(mktemp -d "${TMPDIR:-/tmp}/catan-zero-install.XXXXXXXX")"
chmod 0700 "$INSTALL_TMP"
VERIFIED_RS_WHEEL="$INSTALL_TMP/$RS_WHEEL_NAME"
RS_WHEEL_SHA256_FILE="$CATAN_DEST/$RS_WHEEL_SHA256_FILE_REL"
if [ ! -f "$RS_WHEEL_SHA256_FILE" ]; then
  die "missing canonical Rust-wheel checksum inventory: $RS_WHEEL_SHA256_FILE_REL"
fi

RS_WHEEL_EXPECTED_SHA256=""
RS_WHEEL_MATCHES=0
RS_WHEEL_RECORDS=0
RS_WHEEL_INVENTORY_LINE=0
while IFS= read -r inventory_line || [ -n "$inventory_line" ]; do
  RS_WHEEL_INVENTORY_LINE=$((RS_WHEEL_INVENTORY_LINE + 1))
  [ -z "$inventory_line" ] && continue
  if [[ ! "$inventory_line" =~ ^([0-9a-f]{64})[[:space:]]+([A-Za-z0-9][A-Za-z0-9._+-]*)$ ]]; then
    die "malformed Rust-wheel checksum inventory at $RS_WHEEL_SHA256_FILE_REL:$RS_WHEEL_INVENTORY_LINE"
  fi
  inventory_sha256="${BASH_REMATCH[1]}"
  inventory_name="${BASH_REMATCH[2]}"
  RS_WHEEL_RECORDS=$((RS_WHEEL_RECORDS + 1))
  if [ "$inventory_name" = "$RS_WHEEL_NAME" ]; then
    RS_WHEEL_MATCHES=$((RS_WHEEL_MATCHES + 1))
    RS_WHEEL_EXPECTED_SHA256="$inventory_sha256"
  fi
done < "$RS_WHEEL_SHA256_FILE"
if [ "$RS_WHEEL_RECORDS" -ne 1 ] || [ "$RS_WHEEL_MATCHES" -ne 1 ]; then
  die "$RS_WHEEL_SHA256_FILE_REL must contain exactly one non-empty record for $RS_WHEEL_NAME (records=$RS_WHEEL_RECORDS matches=$RS_WHEEL_MATCHES)"
fi

if [ -f "$CATAN_RS_WHEEL" ]; then
  if [ "$(basename -- "$CATAN_RS_WHEEL")" != "$RS_WHEEL_NAME" ]; then
    die "staged Rust wheel has the wrong filename: $CATAN_RS_WHEEL"
  fi
  cp -- "$CATAN_RS_WHEEL" "$VERIFIED_RS_WHEEL"
else
  if [ "$REF_KIND" != "tag" ]; then
    die "CATAN_REF '$CATAN_REF' is not an exact tag and CATAN_RS_WHEEL is absent"
  fi
  RS_WHEEL_URL="https://github.com/nickita-khylkouski/catan-zero-public/releases/download/${CATAN_REF}/${RS_WHEEL_NAME}"
  echo "[install] downloading release wheel: $RS_WHEEL_URL"
  curl --fail --location --silent --show-error \
    --proto '=https' --tlsv1.2 --retry 3 --retry-all-errors \
    "$RS_WHEEL_URL" -o "$VERIFIED_RS_WHEEL" \
    || die "failed to download release wheel for exact tag $CATAN_REF"
fi

printf '%s  %s\n' "$RS_WHEEL_EXPECTED_SHA256" "$RS_WHEEL_NAME" \
  > "$INSTALL_TMP/wheel.sha256"
(
  cd "$INSTALL_TMP"
  sha256sum -c --strict wheel.sha256
) || die "catanatron_rs wheel digest mismatch"
RS_WHEEL_ACTUAL_SHA256="$(sha256sum "$VERIFIED_RS_WHEEL" | awk '{print $1}')"
RS_WHEEL_INVENTORY_SHA256="$(sha256sum "$RS_WHEEL_SHA256_FILE" | awk '{print $1}')"
chmod 0444 "$VERIFIED_RS_WHEEL"
CATAN_RS_WHEEL="$VERIFIED_RS_WHEEL"
echo "[install] catanatron_rs wheel preflight verified: $RS_WHEEL_ACTUAL_SHA256"

# The production executor requires a boot-persistent foreground MPS daemon.
# Install the exact unit from this immutable checkout; ad-hoc `-d` daemons can
# disappear with their SSH session and strand every attached CUDA client.
MPS_UNIT_SOURCE="$CATAN_DEST/tools/fleet/systemd/nvidia-mps.service"
MPS_UNIT_DEST="/etc/systemd/system/nvidia-mps.service"
if [ ! -f "$MPS_UNIT_SOURCE" ]; then
  echo "[install] ERROR: canonical MPS unit is missing: $MPS_UNIT_SOURCE" >&2
  exit 3
fi
if ! sudo -n true 2>/dev/null; then
  echo "[install] ERROR: passwordless sudo is required to install nvidia-mps.service" >&2
  exit 3
fi
sudo install -m 0644 "$MPS_UNIT_SOURCE" "$MPS_UNIT_DEST"
sudo systemctl daemon-reload
sudo systemctl enable nvidia-mps.service
# `enable --now` does not reload an already-active service after unit bytes
# change.  Restart explicitly so preflight observes the unit from this tag,
# never a prior manually-staged definition.
sudo systemctl restart nvidia-mps.service
if [ "$(systemctl is-active nvidia-mps.service)" != "active" ] \
  || [ "$(systemctl is-enabled nvidia-mps.service)" != "enabled" ]; then
  echo "[install] ERROR: nvidia-mps.service is not active+enabled" >&2
  sudo systemctl status nvidia-mps.service --no-pager >&2 || true
  exit 3
fi
echo "[install] nvidia-mps.service active+enabled"

# 3. venv — Python 3.11 is REQUIRED (cp311 rust wheel; matches B200/H100 ~/venv).
#    Bootstrap 3.11 via `uv` when $PY is absent (H100 canaries ship python3.10
#    only); boxes that already have python3.11 keep the plain venv path.
if command -v "$PY" >/dev/null 2>&1; then
  "$PY" -m venv .venv
else
  echo "[install] $PY not found on PATH; bootstrapping Python 3.11 via uv"
  if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck disable=SC1091
    [ -f "$HOME/.local/bin/env" ] && . "$HOME/.local/bin/env"
    export PATH="$HOME/.local/bin:$PATH"
  fi
  command -v uv >/dev/null 2>&1 || { echo "[install] ERROR: uv install failed; cannot bootstrap Python 3.11"; exit 5; }
  uv python install 3.11
  uv venv --seed --python 3.11 .venv
fi
# shellcheck disable=SC1091
. .venv/bin/activate
python - <<'PY'
import sys
assert sys.version_info[:2] == (3, 11), f"Python 3.11 required, got {sys.version.split()[0]}"
PY
python -m pip install --quiet --upgrade pip

# 4. deps in the order that keeps the CUDA torch build:
#    torch (cu128) FIRST so the `rl` extra's torch>=2.0 is already satisfied and
#    pip never swaps in a CPU wheel; then the editable project + dev/rl extras;
#    then the local rust wheel (the one dep pip cannot resolve from PyPI); + modal.
python -m pip install "torch>=2.11" --index-url "$TORCH_INDEX" \
  || { echo "[install] cu128 index failed; falling back to default torch index"; python -m pip install "torch>=2.11"; }
# catanatron is vendored under vendor/catanatron and is NOT on PyPI in the
# exact version this repo uses; install it from the local copy before the
# main package so the editable dependency is available for catan-zero tests.
python -m pip install -e vendor/catanatron
python -m pip install -e '.[dev,rl]'
python -m pip install --force-reinstall --no-deps "$CATAN_RS_WHEEL"
python -m pip install modal

# 5. env-doctor — fail LOUD if the canonical stack is incomplete
python - <<'PY'
import importlib, sys
from importlib.metadata import version, PackageNotFoundError
mods = ("torch","scipy","whr","numpy","networkx","gymnasium","zstandard","catanatron_rs","modal","pytest")
missing = []
for m in mods:
    try: importlib.import_module(m)
    except Exception as e: missing.append(f"{m}: {e!r}")
if missing:
    print("env-doctor FAIL:\n  " + "\n  ".join(missing)); sys.exit(4)
try:
    rs = version("catanatron-rs")
except PackageNotFoundError:
    rs = version("catanatron_rs")
assert rs == "0.1.4", f"catanatron_rs must be 0.1.4, got {rs}"
import catanatron_rs
assert hasattr(catanatron_rs.Game, "determinize_for_player"), "wheel lacks information-set determinization"
import torch
assert torch.cuda.is_available(), "canonical fleet install requires a CUDA-enabled torch build"
assert torch.version.cuda == "12.8", f"canonical fleet install requires torch cu128, got CUDA {torch.version.cuda}"
print(f"env-doctor OK: py={sys.version.split()[0]} torch={torch.__version__} "
      f"cuda={torch.cuda.is_available()} catanatron_rs={rs}")
PY

# 6. smoke — rust featurizer parity + information-set API (0.1.4); fast, CPU-only
ulimit -n 65536 2>/dev/null || true
PYTHONPATH="$CATAN_DEST/src" python -m pytest \
  tests/test_rust_featurize_parity.py \
  tests/test_rust_action_context_parity.py \
  tests/test_rust_symmetry_averaging_parity.py \
  tests/test_native_information_set_search.py \
  -q -p no:cacheprovider

# 7. exact-recipe metrics exporter. It remains loopback-only; the observability
# hub reaches it through the committed SSH tunnel topology.
EXPORTER_UNIT_SOURCE="$CATAN_DEST/ops/observability/systemd/catan-fleet-exporter.service"
EXPORTER_UNIT_DEST="/etc/systemd/system/catan-fleet-exporter.service"
if [ ! -f "$EXPORTER_UNIT_SOURCE" ]; then
  echo "[install] ERROR: canonical fleet exporter unit is missing: $EXPORTER_UNIT_SOURCE" >&2
  exit 3
fi
sudo install -m 0644 "$EXPORTER_UNIT_SOURCE" "$EXPORTER_UNIT_DEST"
sudo systemctl daemon-reload
sudo systemctl enable catan-fleet-exporter.service
sudo systemctl restart catan-fleet-exporter.service
if [ "$(systemctl is-active catan-fleet-exporter.service)" != "active" ] \
  || [ "$(systemctl is-enabled catan-fleet-exporter.service)" != "enabled" ]; then
  echo "[install] ERROR: catan-fleet-exporter.service is not active+enabled" >&2
  sudo systemctl status catan-fleet-exporter.service --no-pager >&2 || true
  exit 3
fi
echo "[install] catan-fleet-exporter.service active+enabled (loopback :9500)"

# Editable installs may create ignored build metadata, but tracked/untracked
# source drift after installation is never acceptable for a frozen runtime.
if [ -n "$(git status --porcelain --untracked-files=all)" ]; then
  git status --short >&2
  die "deployment checkout drifted during installation"
fi

# 8. Durable, atomic install receipt.  It lives outside the immutable checkout
# so the evidence does not make a future integrity check report a dirty tree.
CATAN_INSTALL_RECEIPT="${CATAN_INSTALL_RECEIPT:-$HOME/.local/state/catan-zero/install-${HEAD_COMMIT}.json}"
export CATAN_INSTALL_RECEIPT CATAN_REPO CATAN_REF CATAN_DEST REF_KIND TAG_COMMIT HEAD_COMMIT
export RS_WHEEL_NAME RS_WHEEL_ACTUAL_SHA256 RS_WHEEL_EXPECTED_SHA256
export RS_WHEEL_SHA256_FILE_REL RS_WHEEL_INVENTORY_SHA256
export CATAN_MPS_ACTIVE="$(systemctl is-active nvidia-mps.service)"
export CATAN_MPS_ENABLED="$(systemctl is-enabled nvidia-mps.service)"
export CATAN_EXPORTER_ACTIVE="$(systemctl is-active catan-fleet-exporter.service)"
export CATAN_EXPORTER_ENABLED="$(systemctl is-enabled catan-fleet-exporter.service)"
python - <<'PY'
from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import sys

import catanatron_rs
import torch

try:
    rust_version = version("catanatron-rs")
except PackageNotFoundError:
    rust_version = version("catanatron_rs")

determinize_api = hasattr(catanatron_rs.Game, "determinize_for_player")
if rust_version != "0.1.4" or not determinize_api:
    raise SystemExit("refusing to write receipt for an invalid catanatron_rs install")

payload = {
    "schema_version": "catan-zero-install-receipt-v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "repository": os.environ["CATAN_REPO"],
    "requested_ref": os.environ["CATAN_REF"],
    "ref_kind": os.environ["REF_KIND"],
    "tag_commit": os.environ.get("TAG_COMMIT") or None,
    "source_commit": os.environ["HEAD_COMMIT"],
    "destination": str(Path(os.environ["CATAN_DEST"]).resolve()),
    "wheel": {
        "filename": os.environ["RS_WHEEL_NAME"],
        "sha256": os.environ["RS_WHEEL_ACTUAL_SHA256"],
        "expected_sha256": os.environ["RS_WHEEL_EXPECTED_SHA256"],
        "checksum_inventory": os.environ["RS_WHEEL_SHA256_FILE_REL"],
        "checksum_inventory_sha256": os.environ["RS_WHEEL_INVENTORY_SHA256"],
    },
    "runtime": {
        "python_version": sys.version.split()[0],
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "cuda_available": bool(torch.cuda.is_available()),
        "catanatron_rs_version": rust_version,
        "determinize_for_player": bool(determinize_api),
    },
    "services": {
        "nvidia_mps_active": os.environ["CATAN_MPS_ACTIVE"],
        "nvidia_mps_enabled": os.environ["CATAN_MPS_ENABLED"],
        "fleet_exporter_active": os.environ["CATAN_EXPORTER_ACTIVE"],
        "fleet_exporter_enabled": os.environ["CATAN_EXPORTER_ENABLED"],
    },
}

receipt = Path(os.environ["CATAN_INSTALL_RECEIPT"]).expanduser()
receipt.parent.mkdir(parents=True, exist_ok=True)
temporary = receipt.with_name(f".{receipt.name}.tmp-{os.getpid()}")
with temporary.open("w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.chmod(temporary, 0o600)
os.replace(temporary, receipt)
directory_fd = os.open(receipt.parent, os.O_RDONLY)
try:
    os.fsync(directory_fd)
finally:
    os.close(directory_fd)
print(f"[install] receipt={receipt} sha256_pending_shell")
PY
INSTALL_RECEIPT_SHA256="$(sha256sum "$CATAN_INSTALL_RECEIPT" | awk '{print $1}')"
echo "[install] receipt sha256=$INSTALL_RECEIPT_SHA256 path=$CATAN_INSTALL_RECEIPT"

echo "[install] $CATAN_REF READY at $CATAN_DEST (.venv activated-on-demand)"
echo "[install] runtime reminders: ulimit -n 65536; pass --optimizer/--weight-decay/"
echo "          --truncated-vp-margin-value-weight/--lr-schedule explicitly (prelaunch guards)."
