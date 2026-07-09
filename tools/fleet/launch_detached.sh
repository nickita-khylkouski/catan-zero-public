#!/usr/bin/env bash
# tools/fleet/launch_detached.sh — ONE detach + heartbeat implementation for the
# fleet (CAT-122 launcher + CAT-132 gate both source this). Survives SSH teardown
# and exposes mtime-based liveness for fleet_status.
#
# Usage (source, then call):
#   source tools/fleet/launch_detached.sh
#   pid=$(launch_detached "$RUNDIR" "$RUNDIR/run.log" 60 -- \
#           env FOO=bar /path/to/venv/bin/python train.py --flags ...)
#   # ... disconnect safely; job + heartbeat keep running.
#   heartbeat_status "$RUNDIR" 60     # -> ALIVE(age=..)/STALLED(age=..)/DONE/NO_HEARTBEAT
#
# WHY setsid: it starts the job in a NEW session with no controlling terminal, so
# it is NOT in the ssh session's process group — the SIGHUP/SIGTERM the login
# shell blasts to its pgroup on teardown never reaches it (the exit-137 root cause
# CAT-132 diagnosed). nohup is belt-and-suspenders vs SIGHUP; </dev/null detaches
# stdin so a dead tty can't wedge it. No double-fork/disown needed once setsid'd.
set -uo pipefail

# launch_detached <rundir> <logfile> <cadence_s> -- <cmd> [args...]  → echoes job PID
launch_detached() {
  local rundir="$1" logfile="$2" cadence="$3"; shift 3
  [ "${1:-}" = "--" ] && shift
  [ "$#" -ge 1 ] || { echo "launch_detached: no command given" >&2; return 2; }
  mkdir -p "$rundir"

  # Real job: detached, log-captured, stdin closed.
  setsid nohup "$@" >"$logfile" 2>&1 </dev/null &
  local job_pid=$!
  echo "$job_pid" > "$rundir/.pid"

  # Heartbeat writer: also detached, lives exactly as long as the job. Writes the
  # beat ATOMICALLY (tmp + mv) so fleet_status never reads a half-written file.
  setsid nohup bash -c '
    rundir="$1"; job_pid="$2"; cadence="$3"; progress_cmd="$4"
    while kill -0 "$job_pid" 2>/dev/null; do
      # Optional opt-in progress field (empty PROGRESS_CMD -> prior behavior). eval is
      # 2>/dev/null | tail -1 so a missing/expensive/failing progress cmd never stalls or
      # corrupts the beat; the beat still writes atomically even if prog is empty.
      prog=""; [ -n "$progress_cmd" ] && prog=$(eval "$progress_cmd" 2>/dev/null | tail -1)
      printf "%s pid=%s %s\n" "$(date -u +%FT%TZ)" "$job_pid" "$prog" > "$rundir/.heartbeat.tmp"
      mv -f "$rundir/.heartbeat.tmp" "$rundir/.heartbeat"
      sleep "$cadence"
    done
    printf "%s pid=%s EXITED\n" "$(date -u +%FT%TZ)" "$job_pid" > "$rundir/.heartbeat"
  ' _ "$rundir" "$job_pid" "$cadence" "${PROGRESS_CMD:-}" >/dev/null 2>&1 </dev/null &

  # Sanity: confirm the job is in its OWN session (detached), not the shell's.
  local sid; sid=$(ps -o sid= -p "$job_pid" 2>/dev/null | tr -d ' ')
  echo "launched pid=$job_pid sid=${sid:-?} (own session = teardown-safe) log=$logfile" >&2
  echo "$job_pid"
}

# heartbeat_status <rundir> [cadence_s]  → ALIVE(age=..)/STALLED(age=..)/DONE/NO_HEARTBEAT
# STALLED = heartbeat file stopped updating (hung job) — distinct from a slow one
# (still updating) and from DONE (job exited cleanly).
heartbeat_status() {
  local rundir="$1" cadence="${2:-60}" hb
  hb="$rundir/.heartbeat"
  [ -f "$hb" ] || { echo "NO_HEARTBEAT"; return; }
  if grep -q EXITED "$hb" 2>/dev/null; then echo "DONE"; return; fi
  local age=$(( $(date +%s) - $(stat -c %Y "$hb" 2>/dev/null || echo 0) ))
  if [ "$age" -lt $(( cadence * 2 )) ]; then echo "ALIVE(age=${age}s)"; else echo "STALLED(age=${age}s)"; fi
}

# Executable form: `launch_detached.sh <rundir> <log> <cadence> -- <cmd...>`
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  launch_detached "$@"
fi
