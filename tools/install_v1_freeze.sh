#!/usr/bin/env bash
# ============================================================================
# catan-zero frozen-release installer (CAT-117)
# Single deploy path for BOTH arches (H100 cu128 / A100 cu128). Replaces the
# non-git tarball snapshot layout with: git clone + checkout the frozen tag +
# a reproducible venv. Idempotent; safe to re-run.
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
#   CATAN_DEST   checkout dir (default ~/catan-zero-v1)
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
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
PY="${PY:-python3.11}"

if [ -z "$CATAN_REF" ]; then
  echo "[install] ERROR: CATAN_REF is required; v1.0-deploy predates the current H100 launcher/lifecycle fixes." >&2
  echo "          Publish this verified tree as an immutable release tag, then rerun with CATAN_REF=<that-tag>." >&2
  exit 2
fi

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

# The fallback URL below is keyed by the literal GitHub release tag. Fail before
# building a venv when a raw commit/branch was requested without a staged wheel;
# otherwise provisioning would spend minutes installing dependencies before a
# failure that cannot be repaired by retrying the same command.
if [ ! -f "$CATAN_RS_WHEEL" ] \
  && ! git show-ref --verify --quiet "refs/tags/$CATAN_REF"; then
  echo "[install] ERROR: CATAN_REF '$CATAN_REF' is not a local release tag and CATAN_RS_WHEEL is absent." >&2
  echo "          Use the published release tag (automatic wheel download), or set CATAN_RS_WHEEL for a commit ref." >&2
  exit 3
fi

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
  RS_WHEEL_URL="https://github.com/nickita-khylkouski/catan-zero-public/releases/download/${CATAN_REF}/${RS_WHEEL_NAME}"
  RS_WHEEL_TMP="$(mktemp -d)/${RS_WHEEL_NAME}"
  echo "[install] CATAN_RS_WHEEL not found locally; auto-fetching release asset: $RS_WHEEL_URL"
  if curl -fsSL "$RS_WHEEL_URL" -o "$RS_WHEEL_TMP"; then
    CATAN_RS_WHEEL="$RS_WHEEL_TMP"
  else
    echo "[install] ERROR: catanatron_rs wheel not found at $CATAN_RS_WHEEL and auto-download failed"
    echo "          set CATAN_RS_WHEEL to the 0.1.4 cp311 manylinux_2_34 wheel (built from native/catanatron-rs),"
    echo "          or check that the release asset exists at: $RS_WHEEL_URL"
    exit 3
  fi
fi
RS_WHEEL_SHA256_FILE="native/catanatron-rs/WHEEL_SHA256SUMS"
if [ ! -f "$RS_WHEEL_SHA256_FILE" ]; then
  echo "[install] ERROR: missing canonical Rust-wheel checksum inventory: $RS_WHEEL_SHA256_FILE" >&2
  exit 3
fi
RS_WHEEL_EXPECTED_SHA256="$(awk -v name="$RS_WHEEL_NAME" '$2 == name {print $1}' "$RS_WHEEL_SHA256_FILE")"
if [ -z "$RS_WHEEL_EXPECTED_SHA256" ]; then
  echo "[install] ERROR: $RS_WHEEL_NAME is not sealed in $RS_WHEEL_SHA256_FILE" >&2
  exit 3
fi
RS_WHEEL_ACTUAL_SHA256="$(sha256sum "$CATAN_RS_WHEEL" | awk '{print $1}')"
if [ "$RS_WHEEL_ACTUAL_SHA256" != "$RS_WHEEL_EXPECTED_SHA256" ]; then
  echo "[install] ERROR: catanatron_rs wheel digest mismatch" >&2
  echo "          expected=$RS_WHEEL_EXPECTED_SHA256" >&2
  echo "          actual=$RS_WHEEL_ACTUAL_SHA256 path=$CATAN_RS_WHEEL" >&2
  exit 3
fi
echo "[install] catanatron_rs wheel sha256 verified: $RS_WHEEL_ACTUAL_SHA256"
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
assert rs == "0.1.4", f"catanatron_rs must be 0.1.4, got {rs}"
import catanatron_rs
assert hasattr(catanatron_rs.Game, "determinize_for_player"), "wheel lacks information-set determinization"
import torch
assert torch.cuda.is_available(), "canonical fleet install requires a CUDA-enabled torch build"
assert torch.version.cuda == "12.8", f"canonical fleet install requires torch cu128, got CUDA {torch.version.cuda}"
print(f"env-doctor OK: py={sys.version.split()[0]} torch={torch.__version__} "
      f"cuda={torch.cuda.is_available()} catanatron_rs={rs}")
PY

# 5. smoke — rust featurizer parity + information-set API (0.1.4); fast, CPU-only
ulimit -n 65536 2>/dev/null || true
PYTHONPATH="$CATAN_DEST/src" python -m pytest \
  tests/test_rust_featurize_parity.py \
  tests/test_rust_action_context_parity.py \
  tests/test_rust_symmetry_averaging_parity.py \
  tests/test_native_information_set_search.py \
  -q -p no:cacheprovider

# 6. exact-recipe metrics exporter. It remains loopback-only; the observability
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

echo "[install] $CATAN_REF READY at $CATAN_DEST (.venv activated-on-demand)"
echo "[install] runtime reminders: ulimit -n 65536; pass --optimizer/--weight-decay/"
echo "          --truncated-vp-margin-value-weight/--lr-schedule explicitly (prelaunch guards)."
