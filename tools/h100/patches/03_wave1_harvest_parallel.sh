#!/bin/bash
# WAVE-1 HARVEST -> MEMMAP pipeline (gen-5 corpora). Read-only pulls from the H100
# fleet, then build role-pure memmap corpora. Lead policy (2026-07-09): POOL all
# valid dirs (live-MPS + legacy + c4 control); the build's seed-collision guard is
# the hard integrity gate (seeds globally disjoint per wave). full/fast is per-row
# via policy_weight_multiplier -> trainer zeroes fast+forced from policy loss.
#   TEACHER (policy, n128 p1.0) = c2 + c3 + c6
#   VOLUME  (value,  n64  p0.25) = c1 + c4(control) + c5
#
# SYSTEM_DESIGN_FINDINGS #19: Parallel rsync across boxes AND dirs.
# Previously: 3 boxes × N dirs = serial loop (~36 sequential rsync calls).
# Now: all boxes in parallel, all dirs within a box in parallel, SSH ControlMaster
# for connection reuse. ~14s -> ~1s for a full harvest sweep.
set -uo pipefail
KEY=~/.ssh/gpu_access_ed25519
# SSH ControlMaster for connection reuse (avoids re-handshaking per rsync)
SSH="ssh -i $KEY -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o ControlMaster=auto -o ControlPath=/tmp/ssh-harvest-%r@%h:%p -o ControlPersist=60"
HARV=~/wave1_harvest; REPO=~/c1_fsdp/repo
declare -A HOST=( [c1]=192.222.54.251 [c2]=68.209.75.117 [c3]=192.222.53.18 [c4]=68.209.73.252 [c5]=68.209.74.145 [c6]=68.209.74.2 )
# All valid dirs per box (stale-valid legacy + live-MPS). Empty/absent dirs are
# harmless no-ops. Re-confirm live dirs with sweep-orch before an authoritative build.
declare -A DIRS=(
  [c1]="gen_out/gpu0 gen_out/gpu1 gen_out/gpu2 gen_out/gpu3 gen_out/mps_wave2/gpu0 gen_out/mps_wave2/gpu1 gen_out/mps_wave2/gpu2 gen_out/mps_wave2/gpu3 gen_out/mps_wave2b/gpu0 gen_out/mps_wave2b/gpu1 gen_out/mps_wave2b/gpu3"
  [c4]="gen_out/gpu0 gen_out/gpu1 gen_out/gpu2 gen_out/gpu3"
  [c5]="gen_out/gpu1 gen_out/gpu2 gen_out/gpu3 gen_out/mps_gpu0 gen_out/mps_gpu1 gen_out/mps_gpu2 gen_out/mps_gpu3"
  [c2]="gen_run/mps_wave1/gpu0 gen_run/mps_wave1/gpu1 gen_run/mps_wave1/gpu2 gen_run/mps_wave1/gpu3"
  [c3]="gen_run/mps_wave1/gpu0 gen_run/mps_wave1/gpu1 gen_run/mps_wave1/gpu2 gen_run/mps_wave1/gpu3"
  [c6]="gen_out/mps_wave1_n128/gpu0 gen_out/mps_wave1_n128/gpu1 gen_out/mps_wave1_n128/gpu2 gen_out/mps_wave1_n128/gpu3"
)

# pull_dirs: parallel rsync of ALL dirs for one box simultaneously.
# Each rsync runs in background; we wait for all before reporting.
pull_dirs() { # box role dir...
  local box=$1 role=$2; shift 2
  local dest=$HARV/$role/$box; mkdir -p "$dest"
  local pids=()
  for d in "$@"; do
    local sub=$(echo "$d" | tr '/' '_')          # PATH-UNIQUE local name
    mkdir -p "$dest/$sub"
    rsync -az --prune-empty-dirs --include='*/' --include='gumbel_self_play_shard_*.npz' --exclude='*' \
      -e "$SSH" "ubuntu@${HOST[$box]}:$d/" "$dest/$sub/" >/dev/null 2>&1 &
    pids+=($!)
  done
  # Wait for all parallel rsyncs for this box
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
  echo "  $box ($role): $(find "$dest" -name '*.npz'|wc -l) npz shards -> $dest"
}

# pull_dirs_parallel: launch ALL boxes in parallel, wait for all.
pull_dirs_parallel() { # role box1 box2...
  local role=$1; shift
  local pids=()
  for b in "$@"; do
    pull_dirs "$b" "$role" ${DIRS[$b]} &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
}

_nonempty() { for s in "$@"; do [ -n "$(find "$s" -name '*.npz' -print -quit 2>/dev/null)" ] && printf '%s ' "$s"; done; }

case "${1:-help}" in
  harvest-volume)  pull_dirs_parallel volume c1 c4 c5 ;;
  harvest-teacher) pull_dirs_parallel teacher c2 c3 c6 ;;
  harvest-all)     pull_dirs_parallel volume c1 c4 c5; pull_dirs_parallel teacher c2 c3 c6 ;;
  build-volume)
    . $REPO/.venv/bin/activate
    python $REPO/tools/build_memmap_corpus.py --source $HARV/volume/c1 $HARV/volume/c4 $HARV/volume/c5 \
      --out ~/corpora/volume_gen5 --progress-every 50 ;;
  build-teacher)
    . $REPO/.venv/bin/activate
    python $REPO/tools/build_memmap_corpus.py --source $(_nonempty $HARV/teacher/c2 $HARV/teacher/c3 $HARV/teacher/c6) \
      --out ~/corpora/teacher_gen5 --progress-every 50 ;;
  build-pooled)
    . $REPO/.venv/bin/activate
    python $REPO/tools/build_memmap_corpus.py \
      --source $(_nonempty $HARV/teacher/c2 $HARV/teacher/c3 $HARV/teacher/c6 $HARV/volume/c1 $HARV/volume/c4 $HARV/volume/c5) \
      --out ~/corpora/gen5_pooled --progress-every 50 ;;
  *) echo "usage: $0 {harvest-volume|harvest-teacher|harvest-all|build-volume|build-teacher|build-pooled}"; exit 1 ;;
esac

# Clean up SSH control sockets
rm -f /tmp/ssh-harvest-* 2>/dev/null
