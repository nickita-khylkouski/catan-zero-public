#!/usr/bin/env bash
# fleet_stop.sh — canonical, robust GPU-work stop for the catan fleet (CAT-123).
#
# WHY THIS EXISTS: `pkill -f <pattern>` can match the operator's own ssh shell,
# while nvidia-smi reports the MPS server instead of its client PIDs.  The
# lifecycle authority is therefore each launch_detached .pid file: that PID is
# also a dedicated SID/PGID, and the entire validated process group can be
# stopped without guessing process names.  Explicit nvidia-smi PIDs remain a
# fallback for legacy/unmanaged jobs.  MPS and observability infrastructure are
# preserved, and GO fails if clients, owned groups, or GPU memory remain.
#
# Usage:
#   fleet_stop.sh <alias|all> [--go]     # default is DRY-RUN (prints, kills nothing)
#   fleet_stop.sh c6 --go                # actually stop GPU work on c6
#   fleet_stop.sh all --go               # stop GPU work fleet-wide
#
# HARD RULES (do not "simplify" away):
#   - NEVER pkill/pkill -f. Kill only validated recorded PGIDs or explicit PIDs.
#   - A recorded group must be an owned regular .pid file, PID=SID=PGID, and
#     contain a canonical Catan command signature before it is eligible.
#   - For unmanaged work, kill python SUPERVISORS first, then GPU leaf workers.
#   - PRESERVE the MPS daemon (nvidia-cuda-mps-control/-server) and observability
#     (dcgm-exporter, prometheus, grafana, node_exporter) — excluded by process_name.
#   - Verify no recorded group/client/worker remains.  GPU memory must be <=50
#     MiB without MPS, or <=128 MiB with only the preserved idle MPS server.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Host registry via the canonical FLEET_CONF resolver (no committed IPs; aliases only).
source "$DIR/fleet_lib.sh" || exit 1
KEY="$(fleet_key)"

TARGET="${1:-}"; MODE="${2:---dry-run}"
[ "$MODE" = "--go" ] && DRY=0 || DRY=1
if [ -z "$TARGET" ]; then echo "usage: fleet_stop.sh <alias|all> [--go]  (default dry-run)"; exit 2; fi

# --- remote routine (runs ON each box); prints a plan, and kills only when GO=1 ------------
# shellcheck disable=SC2016
read -r -d '' REMOTE <<'REMOTE_EOF' || true
set -uo pipefail
GO="$1"
comm_of(){ ps -o comm= -p "$1" 2>/dev/null | tr -d ' '; }
args_of(){ ps -o args= -p "$1" 2>/dev/null; }
# Processes we must NEVER kill as fallback PIDs (infra we preserve).
PRESERVE_RE='mps-server|mps-control|nvidia-cuda-mps|dcgm|nv-hostengine|prometheus|grafana|node_exporter|exporter'
# A .pid file is not sufficient authority by itself: PIDs are reusable.  Require
# the current session/group to contain a command emitted by a canonical Catan
# launcher before a negative-PGID signal is allowed.
CATAN_RE='run_generation\.sh|run_training\.sh|generate_gumbel_selfplay_data\.py|train_bc\.py|torch\.distributed\.run|gumbel_search_[^ ]*\.py|promotion_gate_runner\.py|selfplay_loop\.py|continuous_flywheel\.py'

group_snapshot() {
  # Zombies hold no GPU resources and cannot be signalled.  Excluding them also
  # prevents a slow parent reaper from turning a successful KILL into a false
  # residual-group failure.
  ps -eo pid=,pgid=,sid=,stat=,comm=,args= 2>/dev/null \
    | awk -v group="$1" '$2 == group && $4 !~ /^Z/ {print}'
}

OWNED_GROUPS=""
OWNED_FILES=""
for PIDFILE in "$HOME"/fleet_runs/*/.pid; do
  [ -f "$PIDFILE" ] && [ ! -L "$PIDFILE" ] && [ -O "$PIDFILE" ] || continue
  PGID=$(tr -d '[:space:]' < "$PIDFILE" 2>/dev/null || true)
  case "$PGID" in ''|*[!0-9]*) echo "WARN: ignoring invalid pid file $PIDFILE"; continue;; esac
  [ "$PGID" -gt 1 ] || { echo "WARN: ignoring unsafe PGID $PGID from $PIDFILE"; continue; }
  SNAP=$(group_snapshot "$PGID")
  [ -n "$SNAP" ] || continue
  # launch_detached uses setsid: every member selected by PGID must remain in
  # that same SID.  This rejects a stale file whose numeric PID was reused.
  if ! awk -v group="$PGID" '$2 != group || $3 != group {bad=1} END {exit bad}' <<< "$SNAP"; then
    echo "WARN: ignoring non-detached/reused group $PGID from $PIDFILE"
    continue
  fi
  if ! grep -Eq "$CATAN_RE" <<< "$SNAP"; then
    echo "WARN: ignoring stale/non-Catan group $PGID from $PIDFILE"
    continue
  fi
  OWNED_GROUPS="${OWNED_GROUPS}${PGID}"$'\n'
  OWNED_FILES="${OWNED_FILES}${PGID} ${PIDFILE}"$'\n'
done
OWNED_GROUPS=$(printf '%s' "$OWNED_GROUPS" | sed '/^$/d' | sort -un)

# MPS hides client PIDs from NVML/nvidia-smi.  Query its control interface for
# observability and post-stop verification; lifecycle signals still go to the
# owning detached group, not to the preserved MPS server.
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/mps_pipe_host}"
mps_running() {
  ps -eo comm=,args= 2>/dev/null \
    | awk '$1 ~ /^nvidia-cuda-mps/ && ($0 ~ /mps-control -d/ || $0 ~ /mps-server/) {found=1} END {exit !found}'
}
mps_clients() {
  mps_running || return 0
  [ -e "$CUDA_MPS_PIPE_DIRECTORY/control" ] || return 0
  command -v nvidia-cuda-mps-control >/dev/null 2>&1 || return 0
  SERVERS=$(printf 'get_server_list\n' | nvidia-cuda-mps-control 2>/dev/null \
    | awk '$1 ~ /^[0-9]+$/ {print $1}')
  for SERVER in $SERVERS; do
    printf 'get_client_list %s\n' "$SERVER" | nvidia-cuda-mps-control 2>/dev/null \
      | awk '$1 ~ /^[0-9]+$/ {print $1}'
  done | sort -un
}

# Legacy fallback: explicit compute PIDs actually holding GPU memory, excluding
# preserved infrastructure.  Under older MPS/NVML combinations this may be
# empty by design; the recorded group remains authoritative.
visible_workers() {
  nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',[ ]*' -v re="$PRESERVE_RE" '{pn=tolower($2); gsub(/ /,"",$1)} pn !~ re && $1!="" {print $1}' \
    | sort -un
}
WORKERS=$(visible_workers)
# SUPERVISORS = python/torchrun ancestors of each fallback worker. Climb stops at the first NON-python
#    ancestor (bash/sshd/systemd) so the operator's shell is NEVER a kill target.
SUPERVISORS=""
for pid in $WORKERS; do
  p="$pid"
  while :; do
    pp=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
    { [ -z "$pp" ] || [ "$pp" -le 1 ]; } && break
    case "$(comm_of "$pp")" in
      python|python3|python3.1[0-9]|torchrun) SUPERVISORS="$SUPERVISORS $pp"; p="$pp" ;;
      *) break ;;
    esac
  done
done
SUPERVISORS=$(printf '%s\n' $SUPERVISORS | sort -un)   # dedupe
NWORK=$(printf '%s' "$WORKERS" | grep -c .); NSUP=$(printf '%s' "$SUPERVISORS" | grep -c .)
NGROUP=$(printf '%s' "$OWNED_GROUPS" | grep -c .)
MPS_CLIENTS=$(mps_clients)
NMPS=$(printf '%s' "$MPS_CLIENTS" | grep -c .)

echo "== $(hostname) =="
echo "-- OWNED detached PGIDs: ${OWNED_GROUPS:-none} ($NGROUP) --"
for pgid in $OWNED_GROUPS; do
  GROUP_SIZE=$(group_snapshot "$pgid" | grep -c .)
  echo "     group $pgid from $(awk -v g="$pgid" '$1 == g {$1=""; sub(/^ /, ""); print; exit}' <<< "$OWNED_FILES")"
  group_snapshot "$pgid" | awk 'NR <= 16 {print "       " $0}'
  [ "$GROUP_SIZE" -le 16 ] || echo "       ... $((GROUP_SIZE - 16)) more process(es) omitted"
done
echo "-- MPS client PIDs (informational; server preserved): ${MPS_CLIENTS:-none} ($NMPS) --"
echo "-- GPU compute PIDs / WORKERS (preserved infra excluded): ${WORKERS:-none} ($NWORK) --"
echo "-- SUPERVISORS (python/torchrun parents, killed FIRST): ${SUPERVISORS:-none} ($NSUP) --"
for pid in $SUPERVISORS; do echo "     sup $pid [$(comm_of "$pid")] $(args_of "$pid" | cut -c1-110)"; done
[ "$NGROUP" -eq 0 ] && [ "$NWORK" -eq 0 ] && [ "$NMPS" -eq 0 ] \
  && echo "   (no managed or visible GPU work running — nothing to stop)"

if [ "$GO" != "1" ]; then
  echo "-- DRY-RUN: nothing killed. Re-run with --go to execute. --"
else
  # TERM validated canonical sessions first.  Negative PGID reaches generator,
  # EvalServer, manager, and multiprocessing grandchildren even under MPS.
  for pgid in $OWNED_GROUPS; do
    echo "TERM group $pgid"
    kill -TERM -- "-$pgid" 2>/dev/null || true
  done
  # TERM legacy supervisors first so they reap workers and do not respawn.
  for pid in $SUPERVISORS; do echo "TERM sup $pid"; kill -TERM "$pid" 2>/dev/null || true; done
  for pid in $WORKERS; do echo "TERM worker $pid"; kill -TERM "$pid" 2>/dev/null || true; done
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    LIVE=0
    for pgid in $OWNED_GROUPS; do [ -n "$(group_snapshot "$pgid")" ] && LIVE=1; done
    [ "$LIVE" -eq 0 ] && break
    sleep "${FLEET_STOP_POLL_SECONDS:-0.5}"
  done
  # KILL surviving exact groups, then explicit legacy PIDs.  Never a pattern.
  for pgid in $OWNED_GROUPS; do
    if [ -n "$(group_snapshot "$pgid")" ]; then
      echo "KILL group $pgid"
      kill -KILL -- "-$pgid" 2>/dev/null || true
    fi
  done
  for pid in $SUPERVISORS $WORKERS; do kill -0 "$pid" 2>/dev/null && { echo "KILL $pid"; kill -9 "$pid" 2>/dev/null || true; }; done
  sleep "${FLEET_STOP_RELEASE_SECONDS:-2}"
fi

# Verify.  A GO command is unsuccessful if a validated owned group, any MPS
# client, or >50 MiB remains.  This prevents a false-success stop from allowing
# a second launch to stack work or reuse claimed seeds.
VERIFY_RC=0
if [ "$GO" = "1" ]; then
  for pgid in $OWNED_GROUPS; do
    if [ -n "$(group_snapshot "$pgid")" ]; then
      echo "FAIL: owned process group $pgid is still live" >&2
      VERIFY_RC=1
    fi
  done
  MPS_CLIENTS=$(mps_clients)
  if [ -n "$MPS_CLIENTS" ]; then
    echo "FAIL: MPS client PID(s) still live: $MPS_CLIENTS" >&2
    VERIFY_RC=1
  fi
  RESIDUAL_WORKERS=$(visible_workers)
  if [ -n "$RESIDUAL_WORKERS" ]; then
    echo "FAIL: non-infrastructure GPU PID(s) still live: $RESIDUAL_WORKERS" >&2
    VERIFY_RC=1
  fi
fi

# Per-GPU memory report; retry as CUDA contexts release.
# On driver 580.105.08, the deliberately preserved idle host-wide MPS server
# measures 78 MiB/GPU with zero clients.  Treat that measured infrastructure
# baseline as idle while still failing on any client/worker above.  The limit is
# overrideable for a different driver, but remains tight enough to catch a CUDA
# model context (the 35M evaluator is ~1.0 GiB).
MEMORY_LIMIT_MIB=50
if mps_running && [ -z "${MPS_CLIENTS:-}" ]; then
  MEMORY_LIMIT_MIB="${FLEET_STOP_MPS_IDLE_MEMORY_LIMIT_MIB:-128}"
fi
echo "-- idle GPU memory limit: ${MEMORY_LIMIT_MIB}MiB --"
SEEN_GPU=0
for try in 1 2 3; do
  busy=0
  SEEN_GPU=0
  while IFS=',' read -r idx mem; do
    idx=$(echo "$idx"|tr -d ' '); mem=$(echo "$mem"|tr -d ' ')
    [ -z "$idx" ] && continue
    SEEN_GPU=1
    printf "   gpu%s mem=%sMiB\n" "$idx" "$mem"
    [ "${mem:-0}" -gt "$MEMORY_LIMIT_MIB" ] && busy=1
  done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null)
  { [ "$GO" != "1" ] || [ "$busy" -eq 0 ]; } && break
  sleep "${FLEET_STOP_MEMORY_POLL_SECONDS:-1}"
done
if [ "$GO" = "1" ]; then
  [ "$SEEN_GPU" -eq 1 ] || { echo "FAIL: nvidia-smi returned no GPUs; stop cannot be verified" >&2; VERIFY_RC=1; }
  [ "$busy" -eq 0 ] || { echo "FAIL: GPU memory remains above ${MEMORY_LIMIT_MIB} MiB" >&2; VERIFY_RC=1; }
fi
# MPS daemon status (informational — we intentionally leave it up)
if mps_running; then
  echo "   MPS daemon: PRESERVED (up)"; else echo "   MPS daemon: not present"; fi
exit "$VERIFY_RC"
REMOTE_EOF

run_box() { # $1=ip $2=alias
  local ip="$1" alias="$2"
  local out rc
  out=$(timeout 120 ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=accept-new -i "$KEY" ubuntu@"$ip" \
        "bash -s -- $((1-DRY))" <<< "$REMOTE" 2>&1)
  rc=$?
  if [ -z "$out" ] && [ "$rc" -ne 0 ]; then
    echo "[$alias $ip] UNREACHABLE/FAILED rc=$rc"
  else
    echo "[$alias $ip] rc=$rc"
    echo "$out" | sed 's/^/  /'
  fi
  return "$rc"
}

echo "===== fleet_stop ($([ $DRY -eq 1 ] && echo DRY-RUN || echo GO)) target=$TARGET $(date -u +%H:%M:%SZ) ====="
FOUND=0; FAILED=0
for alias in $(fleet_aliases); do
  if [ "$TARGET" = "all" ] || [ "$TARGET" = "$alias" ]; then
    ip="$(fleet_host "$alias")" || continue
    FOUND=1; run_box "$ip" "$alias" || FAILED=1
  fi
done
[ "$FOUND" -eq 0 ] && { echo "no host matched '$TARGET' (known: $(fleet_aliases | sort | tr '\n' ' '))"; exit 2; }
if [ "$FAILED" -ne 0 ]; then echo "===== FAILED: one or more boxes did not verify clean ====="; exit 1; fi
echo "===== done: verified ====="
