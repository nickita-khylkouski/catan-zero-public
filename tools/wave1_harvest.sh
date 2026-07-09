#!/bin/bash
# Fleet harvest -> memmap pipeline (gen-5 corpora). Read-only rsync pulls of npz
# shards from the fleet, then role-pure / pooled memmap corpus builds.
#
# CAT-126 #19 (in-code default): rsync runs PARALLEL across boxes AND across dirs
# within a box, with SSH ControlMaster connection reuse (was a serial loop of
# ~boxes*dirs rsync calls, each re-handshaking SSH). This is the adopted default.
#
# Fleet topology (HOST ip map + DIRS out-dir map) is NOT hardcoded here: this repo
# is public, and live out-dirs churn per wave. Provide them via an external,
# non-committed config sourced below (see FLEET_CONF). The authoritative source is
# CAT-131 FLEET.md; re-verify DIRS against live out-dirs before an authoritative build.
set -uo pipefail

# --- fleet config (external, non-committed) -----------------------------------
# Must define, as bash associative arrays:
#   declare -A HOST=( [c1]=<ip> [c2]=<ip> ... )
#   declare -A DIRS=( [c1]="relpath1 relpath2 ..." ... )   # rsync source dirs per box
# CAT-126 #19 dedup: use the ONE canonical resolver (CAT-122/131) instead of an
# inline FLEET_CONF source. fleet_lib.sh sources $FLEET_CONF (uncommitted,
# alias->ip), validates HOST, and exposes fleet_host <alias> / fleet_key. DIRS is
# harvest-only so it is asserted here (fleet_lib validates HOST only).
# shellcheck source=tools/fleet/fleet_lib.sh disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/fleet/fleet_lib.sh" || {
  echo "wave1_harvest: fleet config unavailable (see fleet_lib / \$FLEET_CONF)" >&2
  exit 2
}
: "${DIRS:?fleet config ($FLEET_CONF) must define a DIRS assoc array for harvest}"

KEY="$(fleet_key)"
SSH="ssh -i $KEY -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o ControlMaster=auto -o ControlPath=/tmp/ssh-harvest-%r@%h:%p -o ControlPersist=60"
HARV="${HARV_DIR:-$HOME/wave1_harvest}"
# Repo root = this script's parent dir's parent (tools/wave1_harvest.sh -> repo/).
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$REPO/.venv}"

# pull_dirs: PARALLEL rsync of ALL dirs for one box at once (bg + wait).
pull_dirs() { # box role dir...
  local box=$1 role=$2; shift 2
  local dest=$HARV/$role/$box; mkdir -p "$dest"
  local pids=()
  for d in "$@"; do
    local sub; sub=$(echo "$d" | tr '/' '_')   # path-unique local name
    mkdir -p "$dest/$sub"
    rsync -az --prune-empty-dirs --include='*/' --include='gumbel_self_play_shard_*.npz' --exclude='*' \
      -e "$SSH" "ubuntu@$(fleet_host "$box"):$d/" "$dest/$sub/" >/dev/null 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  echo "  $box ($role): $(find "$dest" -name '*.npz' | wc -l) npz shards -> $dest"
}

# pull_dirs_parallel: launch ALL boxes in parallel, wait for all.
pull_dirs_parallel() { # role box1 box2...
  local role=$1; shift
  local pids=()
  for b in "$@"; do pull_dirs "$b" "$role" ${DIRS[$b]} & pids+=($!); done
  for pid in "${pids[@]}"; do wait "$pid"; done
}

_nonempty() { for s in "$@"; do [ -n "$(find "$s" -name '*.npz' -print -quit 2>/dev/null)" ] && printf '%s ' "$s"; done; }

VOLUME_BOXES="${VOLUME_BOXES:-c1 c4 c5}"
TEACHER_BOXES="${TEACHER_BOXES:-c2 c3 c6}"

case "${1:-help}" in
  harvest-volume)  pull_dirs_parallel volume $VOLUME_BOXES ;;
  harvest-teacher) pull_dirs_parallel teacher $TEACHER_BOXES ;;
  harvest-all)     pull_dirs_parallel volume $VOLUME_BOXES; pull_dirs_parallel teacher $TEACHER_BOXES ;;
  build-volume)
    . "$VENV/bin/activate"
    python "$REPO/tools/build_memmap_corpus.py" --source $(_nonempty $HARV/volume/*) \
      --out ~/corpora/volume_gen5 --progress-every 50 ;;
  build-teacher)
    . "$VENV/bin/activate"
    python "$REPO/tools/build_memmap_corpus.py" --source $(_nonempty $HARV/teacher/*) \
      --out ~/corpora/teacher_gen5 --progress-every 50 ;;
  build-pooled)
    . "$VENV/bin/activate"
    python "$REPO/tools/build_memmap_corpus.py" --source $(_nonempty $HARV/teacher/* $HARV/volume/*) \
      --out ~/corpora/gen5_pooled --progress-every 50 ;;
  *) echo "usage: $0 {harvest-volume|harvest-teacher|harvest-all|build-volume|build-teacher|build-pooled}"; exit 1 ;;
esac

rm -f /tmp/ssh-harvest-* 2>/dev/null
