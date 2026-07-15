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
#   CATAN_RS_WHEEL  exact cp311 manylinux wheel named by the runtime contract
#                (pip can't fetch it;
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
CATAN_RS_WHEEL="${CATAN_RS_WHEEL:-}"
RS_WHEEL_SHA256_FILE_REL="native/catanatron-rs/WHEEL_SHA256SUMS"
RUNTIME_CONTRACT_REL="configs/runtime/a1_production_runtime.json"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
PY="${PY:-python3.11}"
MPS_REQUIRED_LIMIT_NOFILE_SOFT=65536

INSTALL_TMP=""
EXPORTER_TRANSACTION_ARMED=0
cleanup_install_tmp() {
  if [ -n "$INSTALL_TMP" ] && [ -d "$INSTALL_TMP" ]; then
    rm -rf -- "$INSTALL_TMP"
  fi
}
finish_install_transaction() {
  local original_status=$?
  local cleanup_failed=0
  local active="unknown"
  local enabled="unknown"
  local main_pid="unknown"
  local active_rc=0
  local enabled_rc=0
  local pid_rc=0
  trap - EXIT
  set +e
  if [ "$EXPORTER_TRANSACTION_ARMED" -eq 1 ]; then
    if [ -n "${CATAN_INSTALL_RECEIPT:-}" ]; then
      rm -f -- "$CATAN_INSTALL_RECEIPT"
    fi
    sudo -n systemctl disable --now catan-fleet-exporter.service >/dev/null 2>&1 \
      || cleanup_failed=1
    active="$(systemctl show --property=ActiveState --value \
      catan-fleet-exporter.service 2>/dev/null)" || active_rc=$?
    enabled="$(systemctl show --property=UnitFileState --value \
      catan-fleet-exporter.service 2>/dev/null)" || enabled_rc=$?
    main_pid="$(systemctl show --property=MainPID --value \
      catan-fleet-exporter.service 2>/dev/null)" || pid_rc=$?
    if [ "$active_rc" -ne 0 ] || [ "$enabled_rc" -ne 0 ] || [ "$pid_rc" -ne 0 ] \
      || [ "$active" != "inactive" ] || [ "$enabled" != "disabled" ] \
      || [ "$main_pid" != "0" ]; then
      cleanup_failed=1
    fi
    if [ "$cleanup_failed" -eq 0 ]; then
      echo "[install] exporter rollback verified: active=$active enabled=$enabled main_pid=$main_pid" >&2
    else
      echo "[install] CRITICAL: exporter rollback FAILED: active=$active enabled=$enabled main_pid=$main_pid" >&2
    fi
  fi
  cleanup_install_tmp
  if [ "$cleanup_failed" -ne 0 ]; then
    exit 3
  fi
  exit "$original_status"
}
trap finish_install_transaction EXIT
trap 'exit 130' INT TERM HUP

die() {
  echo "[install] ERROR: $*" >&2
  exit 3
}

if [[ ! "$CATAN_DEST" =~ ^/[A-Za-z0-9._/-]+$ ]] \
  || [[ "/$CATAN_DEST/" == *"/../"* ]]; then
  die "CATAN_DEST must be an absolute systemd-safe path without '..': $CATAN_DEST"
fi

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

# One checked-in runtime identity controls both installation and the fleet
# executor's launch admission. Do not duplicate these versions in shell flags:
# a release checkout with a malformed or incomplete contract is not deployable.
if ! command -v python3 >/dev/null 2>&1; then
  die "python3 is required to parse $RUNTIME_CONTRACT_REL before bootstrapping the venv"
fi
if ! RUNTIME_CONTRACT_VALUES="$(
  python3 tools/production_runtime_contract.py \
    --contract "$RUNTIME_CONTRACT_REL" --format lines
)"; then
  die "invalid production runtime contract: $RUNTIME_CONTRACT_REL"
fi
mapfile -t RUNTIME_VALUES <<< "$RUNTIME_CONTRACT_VALUES"
if [ "${#RUNTIME_VALUES[@]}" -ne 13 ]; then
  die "production runtime contract emitted ${#RUNTIME_VALUES[@]} fields; expected 13"
fi
RUNTIME_PYTHON_VERSION="${RUNTIME_VALUES[0]}"
RUNTIME_TORCH_VERSION="${RUNTIME_VALUES[1]}"
RUNTIME_TORCH_CUDA_VERSION="${RUNTIME_VALUES[2]}"
RUNTIME_CATANATRON_RS_VERSION="${RUNTIME_VALUES[3]}"
RS_WHEEL_NAME="${RUNTIME_VALUES[4]}"
RUNTIME_CATANATRON_RS_WHEEL_SHA256="${RUNTIME_VALUES[5]}"
RUNTIME_NUMPY_VERSION="${RUNTIME_VALUES[6]}"
RUNTIME_NETWORKX_VERSION="${RUNTIME_VALUES[7]}"
RUNTIME_GYMNASIUM_VERSION="${RUNTIME_VALUES[8]}"
RUNTIME_ZSTANDARD_VERSION="${RUNTIME_VALUES[9]}"
RUNTIME_SCIPY_VERSION="${RUNTIME_VALUES[10]}"
RUNTIME_WHR_VERSION="${RUNTIME_VALUES[11]}"
RUNTIME_NVIDIA_DRIVER_VERSION="${RUNTIME_VALUES[12]}"
RUNTIME_CONTRACT_SHA256="$(sha256sum "$RUNTIME_CONTRACT_REL" | awk '{print $1}')"
if [ -z "$CATAN_RS_WHEEL" ]; then
  CATAN_RS_WHEEL="$HOME/bundle/$RS_WHEEL_NAME"
fi
echo "[install] runtime contract=$RUNTIME_CONTRACT_SHA256 python=$RUNTIME_PYTHON_VERSION torch=$RUNTIME_TORCH_VERSION cuda=$RUNTIME_TORCH_CUDA_VERSION"

# 2. Acquire and verify the exact Rust wheel before *any* sudo, systemd, venv,
# torch, or pip mutation.  Always install a private verified copy so a staged
# source wheel cannot change between hashing and pip consumption.
INSTALL_TMP="$(mktemp -d "${TMPDIR:-/tmp}/catan-zero-install.XXXXXXXX")"
chmod 0700 "$INSTALL_TMP"
RUNTIME_CONSTRAINTS="$INSTALL_TMP/runtime-constraints.txt"
printf '%s\n' \
  "torch==$RUNTIME_TORCH_VERSION" \
  "numpy==$RUNTIME_NUMPY_VERSION" \
  "networkx==$RUNTIME_NETWORKX_VERSION" \
  "gymnasium==$RUNTIME_GYMNASIUM_VERSION" \
  "zstandard==$RUNTIME_ZSTANDARD_VERSION" \
  "scipy==$RUNTIME_SCIPY_VERSION" \
  "whr==$RUNTIME_WHR_VERSION" > "$RUNTIME_CONSTRAINTS"
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
if [ "$RS_WHEEL_EXPECTED_SHA256" != "$RUNTIME_CATANATRON_RS_WHEEL_SHA256" ]; then
  die "$RS_WHEEL_SHA256_FILE_REL digest disagrees with the production runtime contract"
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

# A successful receipt is authoritative only for an uninterrupted installer
# transaction.  Invalidate any same-commit receipt immediately before the
# first privileged/runtime mutation so a later failure cannot leave stale
# success evidence behind.
CATAN_INSTALL_RECEIPT="${CATAN_INSTALL_RECEIPT:-$HOME/.local/state/catan-zero/install-${HEAD_COMMIT}.json}"
if [ -e "$CATAN_INSTALL_RECEIPT" ] && [ ! -f "$CATAN_INSTALL_RECEIPT" ]; then
  die "install receipt path exists and is not a regular file: $CATAN_INSTALL_RECEIPT"
fi
rm -f -- "$CATAN_INSTALL_RECEIPT"

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
if ! CATAN_MPS_LIMIT_NOFILE_SOFT="$(
  systemctl show nvidia-mps.service --property=LimitNOFILESoft --value
)"; then
  echo "[install] ERROR: cannot inspect nvidia-mps.service LimitNOFILESoft" >&2
  sudo systemctl status nvidia-mps.service --no-pager >&2 || true
  exit 3
fi
if [[ ! "$CATAN_MPS_LIMIT_NOFILE_SOFT" =~ ^[0-9]+$ ]] \
  || [ "$CATAN_MPS_LIMIT_NOFILE_SOFT" -lt "$MPS_REQUIRED_LIMIT_NOFILE_SOFT" ]; then
  echo "[install] ERROR: nvidia-mps.service effective LimitNOFILESoft is " \
    "$CATAN_MPS_LIMIT_NOFILE_SOFT; required >=$MPS_REQUIRED_LIMIT_NOFILE_SOFT" >&2
  sudo systemctl status nvidia-mps.service --no-pager >&2 || true
  exit 3
fi
echo "[install] nvidia-mps.service active+enabled LimitNOFILESoft=$CATAN_MPS_LIMIT_NOFILE_SOFT"

# 3. venv — the exact contracted Python patch is REQUIRED.  A command named
#    python3.11 may still be 3.11.x with a different patch; probe the executable
#    itself before it can create the production venv.  Missing or mismatched
#    interpreters take the exact `uv` bootstrap path.
SYSTEM_PYTHON_EXACT=0
if command -v "$PY" >/dev/null 2>&1 \
  && python3 tools/production_runtime_contract.py \
    --contract "$RUNTIME_CONTRACT_REL" --check-python "$PY"; then
  SYSTEM_PYTHON_EXACT=1
fi
if [ "$SYSTEM_PYTHON_EXACT" -eq 1 ]; then
  "$PY" -m venv .venv
else
  echo "[install] $PY is absent or not Python $RUNTIME_PYTHON_VERSION; bootstrapping exact runtime via uv"
  if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck disable=SC1091
    [ -f "$HOME/.local/bin/env" ] && . "$HOME/.local/bin/env"
    export PATH="$HOME/.local/bin:$PATH"
  fi
  command -v uv >/dev/null 2>&1 || { echo "[install] ERROR: uv install failed; cannot bootstrap Python $RUNTIME_PYTHON_VERSION"; exit 5; }
  uv python install "$RUNTIME_PYTHON_VERSION"
  uv venv --seed --python "$RUNTIME_PYTHON_VERSION" .venv
fi
# shellcheck disable=SC1091
. .venv/bin/activate
RUNTIME_PYTHON_VERSION="$RUNTIME_PYTHON_VERSION" python - <<'PY'
import os, platform
expected = os.environ["RUNTIME_PYTHON_VERSION"]
assert platform.python_version() == expected, (
    f"Python {expected} required, got {platform.python_version()}"
)
PY
python -m pip install --quiet --upgrade pip

# 4. deps in the order that keeps the CUDA torch build:
#    torch (cu128) FIRST so the `rl` extra's torch>=2.0 is already satisfied and
#    pip never swaps in a CPU wheel; then the editable project + dev/rl extras;
#    then the local rust wheel (the one dep pip cannot resolve from PyPI); + modal.
python -m pip install -c "$RUNTIME_CONSTRAINTS" \
  "torch==$RUNTIME_TORCH_VERSION" --index-url "$TORCH_INDEX"
python -m pip install -c "$RUNTIME_CONSTRAINTS" \
  "numpy==$RUNTIME_NUMPY_VERSION" \
  "networkx==$RUNTIME_NETWORKX_VERSION" \
  "gymnasium==$RUNTIME_GYMNASIUM_VERSION" \
  "zstandard==$RUNTIME_ZSTANDARD_VERSION" \
  "scipy==$RUNTIME_SCIPY_VERSION" \
  "whr==$RUNTIME_WHR_VERSION"
# catanatron is vendored under vendor/catanatron and is NOT on PyPI in the
# exact version this repo uses; install it from the local copy before the
# main package so the editable dependency is available for catan-zero tests.
python -m pip install -c "$RUNTIME_CONSTRAINTS" -e vendor/catanatron
python -m pip install -c "$RUNTIME_CONSTRAINTS" -e '.[dev,rl]'
python -m pip install --force-reinstall --no-deps "$CATAN_RS_WHEEL"
python -m pip install modal

# 5. env-doctor — fail LOUD if the canonical stack is incomplete
python - <<'PY'
import importlib, json, platform, subprocess, sys
from importlib.metadata import version, PackageNotFoundError
from tools.production_runtime_contract import load_runtime_contract

expected = load_runtime_contract()
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
assert platform.python_version() == expected["python_version"]
assert rs == expected["catanatron_rs_version"], (
    f"catanatron_rs must be {expected['catanatron_rs_version']}, got {rs}"
)
import catanatron_rs
assert hasattr(catanatron_rs.Game, "determinize_for_player"), "wheel lacks information-set determinization"
assert hasattr(catanatron_rs.Game, "public_card_deductions_json"), "wheel lacks public-card deduction boundary"
_card_game = catanatron_rs.Game.simple(["RED", "BLUE"], seed=7)
_card_actor = str(_card_game.current_color())
_card_payload = json.loads(_card_game.public_card_deductions_json(_card_actor))
assert _card_payload.get("contract") == "public_card_deductions_2p_v1"
assert _card_payload.get("observer") == _card_actor
assert _card_payload.get("resource_composition_exact") is True
assert _card_payload.get("development_composition_exact") is False
assert callable(getattr(catanatron_rs, "gumbel_search", None)), "wheel lacks native Gumbel MCTS"
capability_fn = getattr(catanatron_rs, "gumbel_search_capabilities", None)
assert callable(capability_fn), "wheel lacks native Gumbel capability contract"
capabilities = set(capability_fn())
required_capabilities = {
    "sigma_reference_visits",
    "belief_target_evidence",
    "initial_road_d1_scope",
    "public_award_feature_parity",
    "policy_temperature_semantics",
    "coherent_public_belief_search",
    "forced_root_trajectory_only",
}
assert required_capabilities <= capabilities, (
    f"wheel lacks required native Gumbel capabilities: "
    f"{sorted(required_capabilities - capabilities)}"
)
import torch
assert str(torch.__version__) == expected["torch_version"]
assert str(torch.version.cuda) == expected["torch_cuda_version"]
for distribution in ("numpy", "networkx", "gymnasium", "zstandard", "scipy", "whr"):
    actual = version(distribution)
    wanted = expected[f"{distribution}_version"]
    assert actual == wanted, f"{distribution} must be {wanted}, got {actual}"
driver_versions = {
    line.strip()
    for line in subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()
    if line.strip()
}
assert driver_versions == {expected["nvidia_driver_version"]}, driver_versions
assert torch.cuda.is_available(), "canonical fleet install requires a CUDA-enabled torch build"
print(f"env-doctor OK: py={platform.python_version()} torch={torch.__version__} "
      f"cuda={torch.cuda.is_available()} catanatron_rs={rs}")
PY

# 6. smoke — Rust featurizer, information-set, and native MCTS API; fast, CPU-only
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
EXPORTER_DROPIN_DIR="/etc/systemd/system/catan-fleet-exporter.service.d"
EXPORTER_UNIT_RENDERED="$INSTALL_TMP/catan-fleet-exporter.service"
if [ ! -f "$EXPORTER_UNIT_SOURCE" ]; then
  echo "[install] ERROR: canonical fleet exporter unit is missing: $EXPORTER_UNIT_SOURCE" >&2
  exit 3
fi
# The committed unit documents the fleet default.  Upgrades are allowed to use
# another fresh absolute checkout, so render only those two exact default path
# occurrences and reject paths that would require ambiguous systemd quoting.
export EXPORTER_UNIT_SOURCE EXPORTER_UNIT_RENDERED CATAN_DEST
python - <<'PY'
from pathlib import Path
import os
import re

source = Path(os.environ["EXPORTER_UNIT_SOURCE"])
destination = Path(os.environ["CATAN_DEST"]).resolve()
if not destination.is_absolute() or not re.fullmatch(r"/[A-Za-z0-9._/-]+", str(destination)):
    raise SystemExit(f"CATAN_DEST is not safe for exact systemd rendering: {destination}")
default = "/home/ubuntu/catan-zero-v1"
text = source.read_text(encoding="utf-8")
if text.count(default) != 2:
    raise SystemExit("canonical exporter unit default-path contract drifted")
rendered = text.replace(default, str(destination))
target = Path(os.environ["EXPORTER_UNIT_RENDERED"])
target.write_text(rendered, encoding="utf-8")
target.chmod(0o444)
PY

exporter_fail() {
  local message="$1"
  die "$message; transaction rollback will verify exporter inactive+disabled"
}
# A base unit does not override an existing systemd drop-in.  Old fleet
# deployments used an override.conf pointing at a versioned staging tree, so
# merely installing the canonical unit could leave an active, healthy-looking
# exporter running stale code and omitting the validation output root.  The
# frozen installer owns the complete exporter definition: remove its legacy
# /etc drop-in namespace and then fail closed if any drop-in remains elsewhere.
EXPORTER_TRANSACTION_ARMED=1
if sudo test -e "$EXPORTER_DROPIN_DIR" || sudo test -L "$EXPORTER_DROPIN_DIR"; then
  sudo rm -rf -- "$EXPORTER_DROPIN_DIR"
fi
sudo install -m 0644 "$EXPORTER_UNIT_RENDERED" "$EXPORTER_UNIT_DEST"
sudo systemctl daemon-reload
CATAN_EXPORTER_FRAGMENT_PATH="$(systemctl show \
  --property=FragmentPath --value catan-fleet-exporter.service)"
CATAN_EXPORTER_DROPIN_PATHS="$(systemctl show \
  --property=DropInPaths --value catan-fleet-exporter.service)"
if [ "$CATAN_EXPORTER_FRAGMENT_PATH" != "$EXPORTER_UNIT_DEST" ] \
  || [ -n "$CATAN_EXPORTER_DROPIN_PATHS" ]; then
  echo "[install] ERROR: exporter systemd provenance drift" >&2
  echo "[install] fragment=$CATAN_EXPORTER_FRAGMENT_PATH" >&2
  echo "[install] dropins=$CATAN_EXPORTER_DROPIN_PATHS" >&2
  exporter_fail "exporter systemd provenance drift"
fi

sudo systemctl enable catan-fleet-exporter.service
if ! sudo systemctl restart catan-fleet-exporter.service; then
  exporter_fail "cannot restart canonical exporter"
fi
if [ "$(systemctl is-active catan-fleet-exporter.service)" != "active" ] \
  || [ "$(systemctl is-enabled catan-fleet-exporter.service)" != "enabled" ]; then
  sudo systemctl status catan-fleet-exporter.service --no-pager >&2 || true
  exporter_fail "catan-fleet-exporter.service is not active+enabled"
fi

export CATAN_EXPORTER_FRAGMENT_PATH CATAN_EXPORTER_DROPIN_PATHS
if ! CATAN_EXPORTER_ATTESTATION_JSON="$(python - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time
from urllib.request import build_opener, ProxyHandler

destination = Path(os.environ["CATAN_DEST"]).resolve()
expected = [
    str(destination / ".venv/bin/python"),
    str(destination / "tools/fleet/fleet_metrics_exporter.py"),
    "--listen", "127.0.0.1",
    "--port", "9500",
    "--run-root", "/home/ubuntu/gen_out",
    "--run-root", "/home/ubuntu/catan-zero-production/runs/selfplay",
]
url = "http://127.0.0.1:9500/metrics"
opener = build_opener(ProxyHandler({}))
deadline = time.monotonic() + 15.0
last_error = "service did not expose a MainPID"
while time.monotonic() < deadline:
    try:
        raw_pid = subprocess.check_output(
            ["systemctl", "show", "--property=MainPID", "--value", "catan-fleet-exporter.service"],
            text=True,
        ).strip()
        if not raw_pid.isdigit() or int(raw_pid) <= 0:
            raise RuntimeError(f"invalid MainPID {raw_pid!r}")
        pid = int(raw_pid)
        actual = [
            item.decode("utf-8")
            for item in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            if item
        ]
        if actual != expected:
            raise RuntimeError(f"MainPID argv drift: {actual!r}")
        with opener.open(url, timeout=2) as response:
            body = response.read().decode("utf-8", errors="strict")
            if response.status != 200 or "catan_fleet_" not in body:
                raise RuntimeError("metrics response lacks canonical catan_fleet_* metrics")
        stable_pid = subprocess.check_output(
            ["systemctl", "show", "--property=MainPID", "--value", "catan-fleet-exporter.service"],
            text=True,
        ).strip()
        if stable_pid != raw_pid:
            raise RuntimeError(f"MainPID changed during attestation: {raw_pid}->{stable_pid}")
        print(json.dumps({
            "main_pid": pid,
            "argv": actual,
            "metrics_url": url,
            "metrics_prefix": "catan_fleet_",
        }, separators=(",", ":")))
        break
    except Exception as error:
        last_error = repr(error)
        time.sleep(0.25)
else:
    raise SystemExit(f"exporter failed stable exact-readiness attestation: {last_error}")
PY
)"; then
  exporter_fail "canonical exporter readiness attestation failed"
fi
export CATAN_EXPORTER_ATTESTATION_JSON
echo "[install] catan-fleet-exporter.service exact+active+enabled (loopback :9500)"

# Editable installs may create ignored build metadata, but tracked/untracked
# source drift after installation is never acceptable for a frozen runtime.
if [ -n "$(git status --porcelain --untracked-files=all)" ]; then
  git status --short >&2
  die "deployment checkout drifted during installation"
fi

# 8. Durable, atomic install receipt.  It lives outside the immutable checkout
# so the evidence does not make a future integrity check report a dirty tree.
export CATAN_INSTALL_RECEIPT CATAN_REPO CATAN_REF CATAN_DEST REF_KIND TAG_COMMIT HEAD_COMMIT
export RS_WHEEL_NAME RS_WHEEL_ACTUAL_SHA256 RS_WHEEL_EXPECTED_SHA256
export RS_WHEEL_SHA256_FILE_REL RS_WHEEL_INVENTORY_SHA256
export RUNTIME_CONTRACT_REL RUNTIME_CONTRACT_SHA256
export CATAN_MPS_ACTIVE="$(systemctl is-active nvidia-mps.service)"
export CATAN_MPS_ENABLED="$(systemctl is-enabled nvidia-mps.service)"
export CATAN_MPS_LIMIT_NOFILE_SOFT
export CATAN_EXPORTER_ACTIVE="$(systemctl is-active catan-fleet-exporter.service)"
export CATAN_EXPORTER_ENABLED="$(systemctl is-enabled catan-fleet-exporter.service)"
export CATAN_EXPORTER_FRAGMENT_PATH CATAN_EXPORTER_DROPIN_PATHS CATAN_EXPORTER_ATTESTATION_JSON
python - <<'PY'
from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import catanatron_rs
import torch
from tools.production_runtime_contract import load_runtime_contract

expected_runtime = load_runtime_contract()

try:
    rust_version = version("catanatron-rs")
except PackageNotFoundError:
    rust_version = version("catanatron_rs")

determinize_api = hasattr(catanatron_rs.Game, "determinize_for_player")
public_card_api = hasattr(catanatron_rs.Game, "public_card_deductions_json")
public_card_contract = None
if public_card_api:
    card_game = catanatron_rs.Game.simple(["RED", "BLUE"], seed=7)
    card_actor = str(card_game.current_color())
    public_card_contract = json.loads(
        card_game.public_card_deductions_json(card_actor)
    )
native_mcts_api = callable(getattr(catanatron_rs, "gumbel_search", None))
capability_fn = getattr(catanatron_rs, "gumbel_search_capabilities", None)
native_mcts_capabilities = set(capability_fn()) if callable(capability_fn) else set()
required_capabilities = {
    "sigma_reference_visits",
    "belief_target_evidence",
    "initial_road_d1_scope",
    "public_award_feature_parity",
    "policy_temperature_semantics",
    "coherent_public_belief_search",
    "forced_root_trajectory_only",
}
dependency_versions = {
    distribution: version(distribution)
    for distribution in (
        "numpy", "networkx", "gymnasium", "zstandard", "scipy", "whr"
    )
}
expected_dependencies = {
    distribution: expected_runtime[f"{distribution}_version"]
    for distribution in dependency_versions
}
driver_versions = {
    line.strip()
    for line in subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()
    if line.strip()
}
if (
    platform.python_version() != expected_runtime["python_version"]
    or str(torch.__version__) != expected_runtime["torch_version"]
    or str(torch.version.cuda) != expected_runtime["torch_cuda_version"]
    or dependency_versions != expected_dependencies
    or driver_versions != {expected_runtime["nvidia_driver_version"]}
    or rust_version != expected_runtime["catanatron_rs_version"]
    or not torch.cuda.is_available()
    or not determinize_api
    or not public_card_api
    or public_card_contract.get("contract") != "public_card_deductions_2p_v1"
    or public_card_contract.get("observer") != card_actor
    or public_card_contract.get("resource_composition_exact") is not True
    or public_card_contract.get("development_composition_exact") is not False
    or not native_mcts_api
    or not required_capabilities <= native_mcts_capabilities
):
    raise SystemExit("refusing to write receipt for an invalid production runtime")

payload = {
    "schema_version": "catan-zero-install-receipt-v2",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "repository": os.environ["CATAN_REPO"],
    "requested_ref": os.environ["CATAN_REF"],
    "ref_kind": os.environ["REF_KIND"],
    "tag_commit": os.environ.get("TAG_COMMIT") or None,
    "source_commit": os.environ["HEAD_COMMIT"],
    "destination": str(Path(os.environ["CATAN_DEST"]).resolve()),
    "runtime_contract": {
        "path": os.environ["RUNTIME_CONTRACT_REL"],
        "sha256": os.environ["RUNTIME_CONTRACT_SHA256"],
        "identity": expected_runtime,
    },
    "wheel": {
        "filename": os.environ["RS_WHEEL_NAME"],
        "sha256": os.environ["RS_WHEEL_ACTUAL_SHA256"],
        "expected_sha256": os.environ["RS_WHEEL_EXPECTED_SHA256"],
        "checksum_inventory": os.environ["RS_WHEEL_SHA256_FILE_REL"],
        "checksum_inventory_sha256": os.environ["RS_WHEEL_INVENTORY_SHA256"],
    },
    "runtime": {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "dependency_versions": dependency_versions,
        "nvidia_driver_version": next(iter(driver_versions)),
        "cuda_available": bool(torch.cuda.is_available()),
        "catanatron_rs_version": rust_version,
        "native_mcts_capabilities": sorted(native_mcts_capabilities),
        "determinize_for_player": bool(determinize_api),
        "gumbel_search": bool(native_mcts_api),
        "public_card_deductions": {
            "available": bool(public_card_api),
            "contract": public_card_contract.get("contract"),
            "resource_composition_exact": public_card_contract.get(
                "resource_composition_exact"
            ),
            "development_composition_exact": public_card_contract.get(
                "development_composition_exact"
            ),
        },
    },
    "services": {
        "nvidia_mps_active": os.environ["CATAN_MPS_ACTIVE"],
        "nvidia_mps_enabled": os.environ["CATAN_MPS_ENABLED"],
        "nvidia_mps_limit_nofile_soft": int(
            os.environ["CATAN_MPS_LIMIT_NOFILE_SOFT"]
        ),
        "fleet_exporter_active": os.environ["CATAN_EXPORTER_ACTIVE"],
        "fleet_exporter_enabled": os.environ["CATAN_EXPORTER_ENABLED"],
        "fleet_exporter_fragment_path": os.environ["CATAN_EXPORTER_FRAGMENT_PATH"],
        "fleet_exporter_dropin_paths": os.environ["CATAN_EXPORTER_DROPIN_PATHS"],
        "fleet_exporter_effective": json.loads(
            os.environ["CATAN_EXPORTER_ATTESTATION_JSON"]
        ),
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
EXPORTER_TRANSACTION_ARMED=0
echo "[install] receipt sha256=$INSTALL_RECEIPT_SHA256 path=$CATAN_INSTALL_RECEIPT"

echo "[install] $CATAN_REF READY at $CATAN_DEST (.venv activated-on-demand)"
echo "[install] runtime reminders: ulimit -n 65536; pass --optimizer/--weight-decay/"
echo "          --truncated-vp-margin-value-weight/--lr-schedule explicitly (prelaunch guards)."
