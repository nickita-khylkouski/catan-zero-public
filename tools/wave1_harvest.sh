#!/bin/bash
# Fleet harvest -> memmap pipeline (gen-5 corpora). Read-only rsync pulls of npz
# shards plus their run provenance/QA artifacts from the fleet, then role-pure /
# pooled memmap corpus builds.
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
#   declare -A DIRS=( [c1]="relpath1 relpath2 ..." ... )   # rsync run roots per box
# DIRS should name the per-run root (not only a shard leaf) so gpu*/manifest.json,
# progress files, run logs, and JSON QA reports can be reconciled with the shards.
# CAT-126 #19 dedup: use the ONE canonical resolver (CAT-122/131) instead of an
# inline FLEET_CONF source. fleet_lib.sh sources $FLEET_CONF (uncommitted,
# alias->ip), validates HOST, and exposes fleet_host <alias> / fleet_key. DIRS is
# harvest-only so it is asserted here (fleet_lib validates HOST only).
# shellcheck source=tools/fleet/fleet_lib.sh disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/fleet/fleet_lib.sh" || {
  echo "wave1_harvest: fleet config unavailable (see fleet_lib / \$FLEET_CONF)" >&2
  exit 2
}
case "$(declare -p DIRS 2>/dev/null)" in
  "declare -A "*) ;;
  *)
    echo "wave1_harvest: fleet config ($FLEET_CONF) must define a DIRS assoc array for harvest" >&2
    exit 2
    ;;
esac

KEY="$(fleet_key)"
SSH="ssh -i $KEY -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o ControlMaster=auto -o ControlPath=/tmp/ssh-harvest-%r@%h:%p -o ControlPersist=60"
HARV="${HARV_DIR:-$HOME/wave1_harvest}"
# Repo root = this script's parent dir's parent (tools/wave1_harvest.sh -> repo/).
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$REPO/.venv}"

# EXIT cleanup must not replace a failed harvest's status with rm's success.
cleanup_control_sockets() { rm -f /tmp/ssh-harvest-* 2>/dev/null || true; }
trap cleanup_control_sockets EXIT

# Preserve every artifact needed to reconcile a run, while excluding model and
# scratch files that do not belong in the harvested corpus staging tree. The
# explicit manifest/progress entries document the contract; the JSON globs also
# retain run-specific QA reports whose names can evolve independently.
RSYNC_FILTERS=(
  '--include=*/'
  '--include=gumbel_self_play_shard_*.npz'
  '--include=manifest.json'
  '--include=*progress*'
  '--include=*.log'
  '--include=*.json'
  '--include=*.jsonl'
  '--exclude=*'
)

# pull_dirs: PARALLEL rsync of ALL dirs for one box at once (bg + wait).
pull_dirs() { # box role dir...
  local box=$1 role=$2; shift 2
  local host dest=$HARV/$role/$box
  # Keep transaction state dot-prefixed so build-*'s $HARV/<role>/* expansion
  # can only see fully published box directories.
  local stage="$HARV/$role/.${box}.incoming.${BASHPID}"
  local previous="$HARV/$role/.${box}.previous.${BASHPID}"
  local d i sub source_shards shard_count
  host=$(fleet_host "$box") || return $?
  if [ "$#" -eq 0 ]; then
    echo "wave1_harvest: no source roots configured: box=$box role=$role" >&2
    return 2
  fi
  if ! mkdir -p "$(dirname "$dest")"; then
    echo "wave1_harvest: could not create role directory: box=$box role=$role" >&2
    return 5
  fi
  if ! rm -rf -- "$stage" "$previous" || ! mkdir -p "$stage"; then
    echo "wave1_harvest: could not create fresh staging tree: box=$box role=$role" >&2
    return 5
  fi
  local pids=() sources=() subs=()
  local rc=0 wait_rc=0
  i=0
  for d in "$@"; do
    # Prefix the readable path projection with its config index. Besides making
    # every source independently auditable, this prevents paths such as a/b and
    # a_b from colliding in the local staging tree.
    sub=$(printf '%03d_%s' "$i" "$(echo "$d" | tr '/' '_')")
    mkdir -p "$stage/$sub"
    rsync -az --prune-empty-dirs "${RSYNC_FILTERS[@]}" \
      -e "$SSH" "ubuntu@$host:$d/" "$stage/$sub/" >/dev/null 2>&1 &
    pids+=("$!")
    sources+=("$d")
    subs+=("$sub")
    i=$((i + 1))
  done
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
      source_shards=$(find "$stage/${subs[$i]}" -type f \
        -name 'gumbel_self_play_shard_*.npz' | wc -l)
      source_shards=${source_shards//[[:space:]]/}
      echo "  $box ($role) ${sources[$i]}: $source_shards npz shards"
      if [ "$source_shards" -eq 0 ]; then
        echo "wave1_harvest: no NPZ shards harvested: box=$box role=$role root=${sources[$i]}" >&2
        [ "$rc" -ne 0 ] || rc=4
      fi
    else
      wait_rc=$?
      [ "$rc" -ne 0 ] || rc=$wait_rc
      echo "wave1_harvest: rsync failed: box=$box role=$role dir=${sources[$i]} rc=$wait_rc" >&2
    fi
  done

  # Validate only this invocation's fresh staging tree. A previous accepted
  # harvest remains available on failure, but can never satisfy the current
  # pull's source checks or turn an empty transfer into success.
  if [ "$rc" -ne 0 ]; then
    rm -rf -- "$stage"
    return "$rc"
  fi

  shard_count=$(find "$stage" -type f -name 'gumbel_self_play_shard_*.npz' | wc -l)
  shard_count=${shard_count//[[:space:]]/}
  if [ -e "$dest" ]; then
    if ! mv -- "$dest" "$previous"; then
      echo "wave1_harvest: could not stage previous harvest: box=$box role=$role" >&2
      rm -rf -- "$stage"
      return 5
    fi
  fi
  if mv -- "$stage" "$dest"; then
    if ! rm -rf -- "$previous"; then
      echo "wave1_harvest: published current harvest but could not remove previous tree: box=$box role=$role" >&2
      return 5
    fi
  else
    wait_rc=$?
    echo "wave1_harvest: could not publish harvest: box=$box role=$role rc=$wait_rc" >&2
    if [ -e "$previous" ]; then
      mv -- "$previous" "$dest" || true
    fi
    rm -rf -- "$stage"
    return "$wait_rc"
  fi
  echo "  $box ($role): $shard_count npz shards -> $dest"
}

# pull_dirs_parallel: launch ALL boxes in parallel, wait for all.
pull_dirs_parallel() { # role box1 box2...
  local role=$1; shift
  local b configured i
  local pids=() boxes=() dirs=()
  local rc=0 wait_rc=0

  # Validate the complete role before starting any background transfer. This
  # keeps FLEET_CONF/DIRS failures explicit instead of turning a missing entry
  # into an rsync against an empty source.
  for b in "$@"; do
    fleet_host "$b" >/dev/null || return $?
    configured="${DIRS[$b]:-}"
    if [ -z "$configured" ]; then
      echo "wave1_harvest: DIRS[$b] is missing or empty in $FLEET_CONF" >&2
      return 2
    fi
  done

  for b in "$@"; do
    # DIRS is explicitly a whitespace-delimited list of remote run roots.
    read -r -a dirs <<< "${DIRS[$b]}"
    pull_dirs "$b" "$role" "${dirs[@]}" &
    pids+=("$!")
    boxes+=("$b")
  done
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
      :
    else
      wait_rc=$?
      [ "$rc" -ne 0 ] || rc=$wait_rc
      echo "wave1_harvest: box failed: box=${boxes[$i]} role=$role rc=$wait_rc" >&2
    fi
  done
  return "$rc"
}

build_from_worker_leaves() { # out-dir harvested-role-root...
  local out=$1; shift
  local root manifest source_list
  local roots=() sources=()
  for root in "$@"; do
    [ -d "$root" ] && roots+=("$root")
  done
  [ "${#roots[@]}" -gt 0 ] || {
    echo "wave1_harvest: no harvested role roots found for build: out=$out" >&2
    return 4
  }
  while IFS= read -r manifest; do
    sources+=("$(dirname "$manifest")")
  done < <(find "${roots[@]}" -type f -path '*/worker_*/manifest.json' -print | LC_ALL=C sort)
  [ "${#sources[@]}" -gt 0 ] || {
    echo "wave1_harvest: no worker leaf manifests found for build: out=$out" >&2
    return 4
  }
  echo "wave1_harvest: building from ${#sources[@]} explicit worker leaf source(s)"
  mkdir -p "$HARV/source_lists"
  source_list="$HARV/source_lists/$(basename "$out").txt"
  printf '%s\n' "${sources[@]}" > "$source_list"
  python "$REPO/tools/build_memmap_corpus.py" \
    --source-list "$source_list" --out "$out" --progress-every 50
}

VOLUME_BOXES="${VOLUME_BOXES:-c1 c5}"
TEACHER_BOXES="${TEACHER_BOXES:-c2 c3 c6}"

case "${1:-help}" in
  harvest-volume)  pull_dirs_parallel volume $VOLUME_BOXES ;;
  harvest-teacher) pull_dirs_parallel teacher $TEACHER_BOXES ;;
  harvest-all)
    rc=0
    pull_dirs_parallel volume $VOLUME_BOXES || rc=$?
    # Finish all requested roles even if one box failed, but never let a later
    # successful role erase the earlier failure status.
    if pull_dirs_parallel teacher $TEACHER_BOXES; then
      :
    else
      wait_rc=$?
      [ "$rc" -ne 0 ] || rc=$wait_rc
    fi
    exit "$rc"
    ;;
  build-volume)
    . "$VENV/bin/activate"
    build_from_worker_leaves "${VOLUME_CORPUS_DIR:-$HOME/corpora/volume_gen5}" "$HARV"/volume/* ;;
  build-teacher)
    . "$VENV/bin/activate"
    build_from_worker_leaves "${TEACHER_CORPUS_DIR:-$HOME/corpora/teacher_gen5}" "$HARV"/teacher/* ;;
  build-pooled)
    . "$VENV/bin/activate"
    build_from_worker_leaves "${POOLED_CORPUS_DIR:-$HOME/corpora/gen5_pooled}" \
      "$HARV"/teacher/* "$HARV"/volume/* ;;
  *) echo "usage: $0 {harvest-volume|harvest-teacher|harvest-all|build-volume|build-teacher|build-pooled}"; exit 1 ;;
esac
