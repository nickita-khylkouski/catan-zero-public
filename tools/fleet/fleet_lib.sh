#!/usr/bin/env bash
# fleet_lib.sh — canonical FLEET_CONF resolver (CAT-122/125/131). SOURCE this; never hardcode IPs.
#
# FLEET_CONF (default ~/.catan_fleet.conf) is a bash file (sourced, not JSON) defining:
#   declare -A HOST=( [c1]=<ip> [c2]=<ip> ... )   # alias -> ip   (REQUIRED)
#   GPU_SSH_KEY=~/.ssh/gpu_access_ed25519          # optional (default below)
#   declare -A DIRS=( [c1]="relpath ..." ... )     # harvest-only; other tools ignore it
#
# The filled conf is gitignored; only tools/fleet/fleet_conf.example (placeholder IPs) is committed,
# so NO live IPs ever land in the repo. Everything is keyed by ALIAS (c1..c6/a100a/b/b200), never ip.
#
# Usage in a script:   source "$(dirname "${BASH_SOURCE[0]}")/fleet_lib.sh" || exit 1
#                       ip=$(fleet_host c6); key=$(fleet_key)
FLEET_CONF="${FLEET_CONF:-$HOME/.catan_fleet.conf}"
if [ ! -f "$FLEET_CONF" ]; then
  echo "fleet_lib: FLEET_CONF missing: $FLEET_CONF — copy tools/fleet/fleet_conf.example to \$FLEET_CONF and fill real IPs" >&2
  return 1 2>/dev/null || exit 1
fi
# shellcheck disable=SC1090
source "$FLEET_CONF"
if ! declare -p HOST >/dev/null 2>&1; then
  echo "fleet_lib: FLEET_CONF invalid ($FLEET_CONF): defines no HOST assoc array — need 'declare -A HOST=( [c1]=ip ... )'" >&2
  return 1 2>/dev/null || exit 1
fi
: "${GPU_SSH_KEY:=$HOME/.ssh/gpu_access_ed25519}"

# fleet_host <alias> -> echo its ip; loud-fail (rc 2) on an unknown alias so callers never ssh "".
fleet_host() {
  local a="${1:?fleet_host <alias>}"
  local ip="${HOST[$a]:-}"
  if [ -z "$ip" ]; then echo "fleet_lib: unknown alias '$a' (known: ${!HOST[*]})" >&2; return 2; fi
  printf '%s\n' "$ip"
}
# fleet_key -> echo the ssh key path.
fleet_key() { printf '%s\n' "$GPU_SSH_KEY"; }
# fleet_aliases -> list all known aliases (one per line).
fleet_aliases() { printf '%s\n' "${!HOST[@]}"; }
