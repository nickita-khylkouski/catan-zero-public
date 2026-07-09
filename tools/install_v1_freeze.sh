#!/usr/bin/env bash
# ============================================================================
# catan-zero v1.0-deploy — one-command install (CAT-117)
# Single deploy path for BOTH arches (H100 cu128 / A100 cu128). Replaces the
# non-git tarball snapshot layout with: git clone + checkout the frozen tag +
# a reproducible venv. Idempotent; safe to re-run.
#
#   curl -fsSL https://raw.githubusercontent.com/nickita-khylkouski/catan-zero-public/v1.0-deploy/tools/install_v1_freeze.sh | bash
#   # or: bash tools/install_v1_freeze.sh
#
# Overridable via env:
#   CATAN_REPO   git URL, OR a local git-bundle path (default GitHub PUBLIC repo
#                nickita-khylkouski/catan-zero-public; a bundle file works as-is
#                with `git clone` for an offline/air-gapped fallback)
#   CATAN_REF    tag/branch to deploy (default v1.0-deploy)
#   CATAN_DEST   checkout dir (default ~/catan-zero-v1)
#   CATAN_RS_WHEEL  catanatron_rs 0.1.3 cp311 manylinux wheel (pip can't fetch it;
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
CATAN_REF="${CATAN_REF:-v1.0-deploy}"
CATAN_DEST="${CATAN_DEST:-$HOME/catan-zero-v1}"
CATAN_RS_WHEEL="${CATAN_RS_WHEEL:-$HOME/bundle/catanatron_rs-0.1.3-cp311-cp311-manylinux_2_34_x86_64.whl}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
PY="${PY:-python3.11}"

echo "[install] repo=$CATAN_REPO ref=$CATAN_REF dest=$CATAN_DEST py=$PY"

# 1. clone (or update) + checkout the frozen tag
if [ -d "$CATAN_DEST/.git" ]; then
  git -C "$CATAN_DEST" fetch --tags --force origin
else
  git clone "$CATAN_REPO" "$CATAN_DEST"
fi
git -C "$CATAN_DEST" fetch --tags --force origin
git -C "$CATAN_DEST" checkout --force "$CATAN_REF"
cd "$CATAN_DEST"
echo "[install] checked out $(git describe --tags --always) @ $(git rev-parse --short HEAD)"

# 2. venv — Python 3.11 is REQUIRED (cp311 rust wheel; matches B200/H100 ~/venv).
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

# 3. deps in the order that keeps the CUDA torch build:
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
if [ ! -f "$CATAN_RS_WHEEL" ]; then
  RS_WHEEL_NAME="catanatron_rs-0.1.3-cp311-cp311-manylinux_2_34_x86_64.whl"
  RS_WHEEL_URL="https://github.com/nickita-khylkouski/catan-zero-public/releases/download/${CATAN_REF}/${RS_WHEEL_NAME}"
  RS_WHEEL_TMP="$(mktemp -d)/${RS_WHEEL_NAME}"
  echo "[install] CATAN_RS_WHEEL not found locally; auto-fetching release asset: $RS_WHEEL_URL"
  if curl -fsSL "$RS_WHEEL_URL" -o "$RS_WHEEL_TMP"; then
    CATAN_RS_WHEEL="$RS_WHEEL_TMP"
  else
    echo "[install] ERROR: catanatron_rs wheel not found at $CATAN_RS_WHEEL and auto-download failed"
    echo "          set CATAN_RS_WHEEL to the 0.1.3 cp311 manylinux_2_34 wheel (maturin-built from catanatron-rs),"
    echo "          or check that the release asset exists at: $RS_WHEEL_URL"
    exit 3
  fi
fi
python -m pip install --force-reinstall --no-deps "$CATAN_RS_WHEEL"
python -m pip install modal

# 4. env-doctor — fail LOUD if the canonical stack is incomplete
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
assert rs == "0.1.3", f"catanatron_rs must be 0.1.3, got {rs}"
import torch
print(f"env-doctor OK: py={sys.version.split()[0]} torch={torch.__version__} "
      f"cuda={torch.cuda.is_available()} catanatron_rs={rs}")
PY

# 5. smoke — rust featurizer parity (0.1.3 bit-exact); fast, CPU-only
ulimit -n 65536 2>/dev/null || true
PYTHONPATH="$CATAN_DEST/src" python -m pytest \
  tests/test_rust_featurize_parity.py \
  tests/test_rust_action_context_parity.py \
  tests/test_rust_symmetry_averaging_parity.py \
  -q -p no:cacheprovider

echo "[install] $CATAN_REF READY at $CATAN_DEST (.venv activated-on-demand)"
echo "[install] runtime reminders: ulimit -n 65536; pass --optimizer/--weight-decay/"
echo "          --truncated-vp-margin-value-weight/--lr-schedule explicitly (prelaunch guards)."
