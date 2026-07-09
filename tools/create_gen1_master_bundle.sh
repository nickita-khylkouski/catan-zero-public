#!/bin/bash
# Task #80 (gen-1 turnkey prep, part 2): create a transportable git bundle of
# this host's `master` branch, to be scp'd to the A100 hosts and applied via
# tools/update_host_to_gen1_master.sh.
#
# Why a bundle, not a remote: B200/A100A/A100B are independent local git
# repos with no shared remote (every prior cross-host sync in this project
# used direct file copy + local commit, never git push/pull). A bundle is
# the standard git-native way to transport commit history without setting
# up persistent remote access between hosts.
#
# DO NOT RUN until team-lead gives the go (post base-decision H2H). This
# script only touches THIS host's repo (read-only git operations + writing
# the bundle file) -- it does not touch the A100 hosts, so it is safe to
# run any time master is in the desired state, but per team-lead's explicit
# instruction the actual host UPDATE (update_host_to_gen1_master.sh) must
# wait for the go.
set -euo pipefail
cd "$(dirname "$0")/.."

BUNDLE_PATH="${1:-/tmp/catan_zero_gen1_master.bundle}"
BRANCH="${2:-master}"

git rev-parse --verify "$BRANCH" >/dev/null
git bundle create "$BUNDLE_PATH" "$BRANCH"
echo "Bundle created at $BUNDLE_PATH for branch $BRANCH at $(git rev-parse "$BRANCH")"
echo "Next: scp $BUNDLE_PATH to each A100 host, then run tools/update_host_to_gen1_master.sh there."
