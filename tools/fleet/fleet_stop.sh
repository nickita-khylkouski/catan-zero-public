#!/usr/bin/env bash
# fleet_stop.sh — canonical, robust GPU-work stop for the catan fleet (CAT-123).
#
# WHY THIS EXISTS: today's fleet stop took ~8 passes because of `pkill -f <pattern>`,
# which (a) matches the operator's OWN ssh/bash shell and the pgrep command itself →
# drops the connection mid-kill, and (b) leaves torchrun/launcher parents alive so they
# respawn children. This script kills by nvidia-smi COMPUTE-PID (the ground truth of what
# is actually on the GPU), kills python SUPERVISORS/parents FIRST, never touches bash/sshd
# or the MPS daemon / observability stack, and verifies 0 MiB.
#
# Usage:
#   fleet_stop.sh <alias|all> [--go]     # default is DRY-RUN (prints, kills nothing)
#   fleet_stop.sh c6 --go                # actually stop GPU work on c6
#   fleet_stop.sh all --go               # stop GPU work fleet-wide
#
# HARD RULES (do not "simplify" away):
#   - NEVER pkill -f <pattern>. Only kill explicit PIDs derived from nvidia-smi.
#   - Kill python SUPERVISORS (torchrun / launcher python) FIRST, then leaf workers.
#   - Only ever kill comm=python|torchrun. Never bash/sshd/systemd (that's the operator's shell).
#   - PRESERVE the MPS daemon (nvidia-cuda-mps-control/-server) and observability
#     (dcgm-exporter, prometheus, grafana, node_exporter) — excluded by process_name.
#   - Verify 0 MiB per GPU after; report per-box.
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
# processes we must NEVER kill (infra we preserve), matched on process_name/comm
PRESERVE_RE='mps-server|mps-control|nvidia-cuda-mps|dcgm|nv-hostengine|prometheus|grafana|node_exporter|exporter'
# 1) WORKERS = PIDs actually holding GPU memory, minus preserved infra (newline list)
WORKERS=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader,nounits 2>/dev/null \
  | awk -F',[ ]*' -v re="$PRESERVE_RE" '{pn=tolower($2); gsub(/ /,"",$1)} pn !~ re && $1!="" {print $1}' | sort -un)
# 2) SUPERVISORS = python/torchrun ancestors of each worker. Climb stops at the first NON-python
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

echo "== $(hostname) =="
echo "-- GPU compute PIDs / WORKERS (preserved infra excluded): ${WORKERS:-none} ($NWORK) --"
echo "-- SUPERVISORS (python/torchrun parents, killed FIRST): ${SUPERVISORS:-none} ($NSUP) --"
for pid in $SUPERVISORS; do echo "     sup $pid [$(comm_of "$pid")] $(args_of "$pid" | cut -c1-110)"; done
[ "$NWORK" -eq 0 ] && echo "   (no GPU work running — nothing to stop)"

if [ "$GO" != "1" ]; then
  echo "-- DRY-RUN: nothing killed. Re-run with --go to execute. --"
else
  # 3a) SIGTERM supervisors FIRST so they reap workers and do not respawn
  for pid in $SUPERVISORS; do echo "TERM sup $pid"; kill -TERM "$pid" 2>/dev/null || true; done
  sleep 5
  # 3b) SIGKILL any survivors (supervisors then workers), explicit PID only — never a pattern
  for pid in $SUPERVISORS $WORKERS; do kill -0 "$pid" 2>/dev/null && { echo "KILL $pid"; kill -9 "$pid" 2>/dev/null || true; }; done
  sleep 3
fi

# 4) verify per-GPU memory (report; retry a couple times as memory releases)
for try in 1 2 3; do
  busy=0
  while IFS=',' read -r idx mem; do
    idx=$(echo "$idx"|tr -d ' '); mem=$(echo "$mem"|tr -d ' ')
    [ -z "$idx" ] && continue
    printf "   gpu%s mem=%sMiB\n" "$idx" "$mem"
    [ "${mem:-0}" -gt 50 ] && busy=1
  done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null)
  { [ "$GO" != "1" ] || [ "$busy" -eq 0 ]; } && break
  sleep 4
done
# MPS daemon status (informational — we intentionally leave it up)
if [ -e /tmp/mps_pipe_host/control ] || pgrep -x nvidia-cuda-mps-control >/dev/null 2>&1; then
  echo "   MPS daemon: PRESERVED (up)"; else echo "   MPS daemon: not present"; fi
REMOTE_EOF

run_box() { # $1=ip $2=alias
  local ip="$1" alias="$2"
  local out
  out=$(timeout 60 ssh -o ConnectTimeout=10 -o BatchMode=yes -i "$KEY" ubuntu@"$ip" \
        "bash -s -- $((1-DRY))" <<< "$REMOTE" 2>/dev/null)
  if [ -z "$out" ]; then echo "[$alias $ip] UNREACHABLE"; else echo "[$alias $ip]"; echo "$out" | sed 's/^/  /'; fi
}

echo "===== fleet_stop ($([ $DRY -eq 1 ] && echo DRY-RUN || echo GO)) target=$TARGET $(date -u +%H:%M:%SZ) ====="
FOUND=0
for alias in $(fleet_aliases); do
  if [ "$TARGET" = "all" ] || [ "$TARGET" = "$alias" ]; then
    ip="$(fleet_host "$alias")" || continue
    FOUND=1; run_box "$ip" "$alias"
  fi
done
[ "$FOUND" -eq 0 ] && { echo "no host matched '$TARGET' (known: $(fleet_aliases | sort | tr '\n' ' '))"; exit 2; }
echo "===== done ====="
