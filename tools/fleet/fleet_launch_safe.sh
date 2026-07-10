#!/usr/bin/env bash
echo "fleet_launch_safe.sh is retired: use tools/fleet/fleet_launch.sh (see RL_AGENT_HANDOFF.md)." >&2
exit 2

# fleet_launch_safe.sh — SAFE launch-path stub for fleet generation (CAT-123 → CAT-122 builds on this).
#
# Encodes the three preconditions that today's incidents proved non-negotiable. It VALIDATES and
# prints the exact command; it does NOT fire unless --go is passed AND all guards pass. CAT-122's
# full launcher can source these checks (preflight_or_die) rather than reimplement them.
#
#   1. FRESH out-dir      — refuse a populated dir (the tool also refuses, but fail early + explicit).
#   2. SEED CLAIM         — base seed must be a FRESH block, appended to the ledger before launch
#                           (wave restarts that REUSE a base produce duplicate game_seeds → the
#                           pooled-build dedup guard drops the whole partial wave). Never reuse.
#   3. GUARDS ON          — the safe canonical path runs WITH prelaunch guards (cli_flag_lint /
#                           seed_ledger / ledger_overlap / fd_limit). --skip-guards is the DELIBERATE
#                           exception for in-block self-collision wave restarts, and must be explicit.
#   Plus: $GEN_PY portability — the H100 boxes run gen under ~/venv/bin/python, NOT a tree .venv.
#         Always resolve the interpreter; never hardcode .venv/bin/python (that stranded a GPU today).
#
# Usage:
#   fleet_launch_safe.sh <alias> <gpu> <role> <base_seed> [--go]
#     role = teacher | volume        (teacher=n128 p1.0, volume=n64 p0.25)
# Env: GEN_PY (default: resolve ~/venv then tree .venv), CKPT, TREE, WORKERS(=16)
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Host/key come from the FLEET_CONF resolver (alias-keyed, no IPs in the repo); hosts.txt is retired.
# shellcheck source=/dev/null
source "$DIR/fleet_lib.sh" || { echo "fleet_launch_safe: cannot load fleet_lib.sh"; exit 1; }
KEY=$(fleet_key)

ALIAS="${1:?alias}"; GPU="${2:?gpu idx}"; ROLE="${3:?teacher|volume}"; BASE_SEED="${4:?fresh base seed}"; MODE="${5:---dry-run}"
IP=$(fleet_host "$ALIAS") || exit 2
case "$ROLE" in
  teacher) NFULL=128; PFULL=1.0;;
  volume)  NFULL=64;  PFULL=0.25;;
  *) echo "role must be teacher|volume"; exit 2;;
esac
WORKERS="${WORKERS:-16}"
GO=$([ "$MODE" = "--go" ] && echo 1 || echo 0)
OUTSUFFIX="$(date -u +%Y%m%d_%H%M%S)_${ROLE}/gpu${GPU}"   # unique => fresh by construction

# Quoted heredoc: no local expansion. All params passed positionally to bash -s.
# args: GO GPU NFULL PFULL WORKERS BASE_SEED OUTSUFFIX
# shellcheck disable=SC2016
read -r -d '' REMOTE <<'REMOTE_EOF' || true
set -uo pipefail
GO="$1"; GPU="$2"; NFULL="$3"; PFULL="$4"; WORKERS="$5"; BASE_SEED="$6"; OUTSUFFIX="$7"
TREE="${TREE:-$HOME/catan-zero-runsix}"
CKPT="${CKPT:-$HOME/bundle/champion_v0.pt}"
OUT="$HOME/gen_out/$OUTSUFFIX"
LEDGER="$HOME/catan-zero-runsix/runs/SEED_LEDGER.md"
# GEN_PY resolution: explicit, else ~/venv, else tree .venv (NEVER hardcode .venv — stranded a GPU today)
GEN_PY="${GEN_PY:-}"
if [ -z "$GEN_PY" ]; then
  if [ -x "$HOME/venv/bin/python" ]; then GEN_PY="$HOME/venv/bin/python"
  elif [ -x "$TREE/.venv/bin/python" ]; then GEN_PY="$TREE/.venv/bin/python"
  else GEN_PY=""; fi
fi
FAIL=0
# GUARD 1: fresh out-dir
if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then echo "GUARD FAIL: out-dir $OUT is populated (not fresh)"; FAIL=1; else echo "ok: out-dir fresh ($OUT)"; fi
# GUARD 2: base seed must fall inside a CLAIMED ledger RANGE (heuristic pre-check; the tool's own
# prelaunch_guard/parse_seed_ledger is authoritative for overlap — this fails fast before ssh cost).
# Parses "[start – end)" rows tolerating commas + en-dash(U+2013)/hyphen; tests start<=seed<end.
if [ -f "$LEDGER" ] && awk -v s="$BASE_SEED" '
    /^\[/ { line=$0; gsub(/,/,"",line); gsub(/[^0-9]/," ",line);   # commas out; en-dash/brackets/text -> space
            n=split(line,a," "); if (n>=2) { lo=a[1]+0; hi=a[2]+0; if (lo<=s+0 && s+0<hi) found=1 } }
    END { exit(found?0:1) }' "$LEDGER"; then
  echo "ok: base seed $BASE_SEED is inside a claimed ledger range"
else
  echo "GUARD FAIL: base seed $BASE_SEED is NOT inside any claimed ledger range — claim a FRESH block first"; FAIL=1
fi
# GUARD 3: interpreter + ckpt exist
[ -n "$GEN_PY" ] && [ -x "$GEN_PY" ] && echo "ok: GEN_PY=$GEN_PY" || { echo "GUARD FAIL: no python (~/venv or \$TREE/.venv)"; FAIL=1; }
[ -f "$CKPT" ] && echo "ok: ckpt $CKPT" || { echo "GUARD FAIL: ckpt $CKPT missing"; FAIL=1; }
CMD="cd $TREE && CUDA_VISIBLE_DEVICES=$GPU CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe_host nohup $GEN_PY tools/generate_gumbel_selfplay_data.py --out-dir $OUT --checkpoint $CKPT --device cuda --games 1500 --workers $WORKERS --base-seed $BASE_SEED --shard-size 2048 --n-full $NFULL --n-fast 16 --p-full $PFULL --c-visit 50.0 --c-scale 0.03 --max-decisions 600 --max-depth 80 --temperature-decisions 90 --correct-rust-chance-spectra --lazy-interior-chance --public-observation --track 2p_no_trade --vps-to-win 10 --format npz --score-actions > $OUT/launch.log 2>&1 &"
echo "WOULD RUN: $CMD"
echo "NOTE: guards ON (no --skip-guards). --skip-guards is ONLY for in-block wave-restart self-collision, and must be explicit."
[ "$FAIL" -ne 0 ] && { echo "REFUSING: guard(s) failed."; exit 3; }
if [ "$GO" = "1" ]; then mkdir -p "$OUT"; eval "$CMD"; echo "launched pid $! on gpu$GPU"; else echo "DRY-RUN: guards passed; not launched (pass --go)"; fi
REMOTE_EOF

echo "===== fleet_launch_safe $ALIAS/gpu$GPU role=$ROLE seed=$BASE_SEED ($([ "$GO" = 1 ] && echo GO || echo DRY-RUN)) ====="
timeout 40 ssh -o ConnectTimeout=10 -o BatchMode=yes -i "$KEY" ubuntu@"$IP" \
  "bash -s -- $GO $GPU $NFULL $PFULL $WORKERS $BASE_SEED $OUTSUFFIX" <<< "$REMOTE" 2>&1 | sed 's/^/  /'
