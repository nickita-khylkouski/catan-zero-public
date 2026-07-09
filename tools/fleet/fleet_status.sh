#!/usr/bin/env bash
# fleet_status.sh — one-shot per-box GPU util/mem + running-role for the whole fleet (CAT-123).
# Read-only. Parallel over the FLEET_CONF alias registry. Role is inferred from the live process cmdlines so you
# can see at a glance what each box is actually doing (teacher/volume gen, training, gate/eval,
# or idle) without hunting through ps on nine boxes.
#
# Usage: fleet_status.sh [alias|all]      (default all)
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Host registry via the canonical FLEET_CONF resolver (aliases only; no committed IPs).
source "$DIR/fleet_lib.sh" || exit 1
KEY="$(fleet_key)"
TARGET="${1:-all}"

# shellcheck disable=SC2016
read -r -d '' REMOTE <<'REMOTE_EOF' || true
set -uo pipefail
# GPU line: idx util mem
GPU=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null)
NG=$(echo "$GPU" | grep -c .); BUSY=$(echo "$GPU" | awk -F',[ ]*' '$2+0>50{c++}END{print c+0}')
UTILAVG=$(echo "$GPU" | awk -F',[ ]*' '{s+=$2;n++}END{if(n)printf "%d",s/n; else print 0}')
MEMMAX=$(echo "$GPU" | awk -F',[ ]*' '{if($3+0>m)m=$3}END{print m+0}')
# role inference from live cmdlines (first match wins), + MPS presence
ROLE="idle"; DETAIL=""
CMDS=$(ps -eo args= 2>/dev/null)
NF=$(echo "$CMDS" | grep -oE 'n-full [0-9]+' | head -1 | awk '{print $2}')
if echo "$CMDS" | grep -q "torchrun\|train_bc.py"; then ROLE="TRAINING"; DETAIL="train_bc$( echo "$CMDS"|grep -q grow-from && echo '/grow')"
elif echo "$CMDS" | grep -q "gumbel_search_cross_net_h2h"; then ROLE="GATE(cross-net)"
elif echo "$CMDS" | grep -q "gumbel_search_vs_bot_h2h\|gumbel_search_vs_raw"; then ROLE="EVAL(vs-bot)"
elif echo "$CMDS" | grep -q "generate_gumbel_selfplay_data"; then
    if [ "${NF:-0}" -ge 128 ]; then ROLE="GEN-TEACHER(n${NF})"; elif [ -n "${NF:-}" ]; then ROLE="GEN-VOLUME(n${NF})"; else ROLE="GEN"; fi
fi
MPS="no-mps"; { [ -e /tmp/mps_pipe_host/control ] || pgrep -x nvidia-cuda-mps-control >/dev/null 2>&1; } && MPS="MPS"
WORKERS=$(echo "$CMDS" | grep -c "generate_gumbel_selfplay_data\|train_bc.py")
printf "gpus=%s busy=%s util_avg=%s%% mem_max=%sMiB | role=%s %s | %s | launcher_procs=%s\n" \
  "$NG" "$BUSY" "$UTILAVG" "$MEMMAX" "$ROLE" "$DETAIL" "$MPS" "$WORKERS"
REMOTE_EOF

TMP=$(mktemp -d)
for alias in $(fleet_aliases); do
  [ "$TARGET" = "all" ] || [ "$TARGET" = "$alias" ] || continue
  ip="$(fleet_host "$alias")" || continue
  (
    out=$(timeout 25 ssh -o ConnectTimeout=8 -o BatchMode=yes -i "$KEY" ubuntu@"$ip" "bash -s" <<< "$REMOTE" 2>/dev/null)
    [ -z "$out" ] && out="UNREACHABLE"
    printf "%-6s %-16s %s\n" "$alias" "$ip" "$out" > "$TMP/$alias"
  ) &
done
wait
echo "===== FLEET STATUS $(date -u +%H:%M:%SZ) ====="
cat "$TMP"/* 2>/dev/null | sort
rm -rf "$TMP"
