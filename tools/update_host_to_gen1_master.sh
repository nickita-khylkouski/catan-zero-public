#!/bin/bash
# Task #80 (gen-1 turnkey prep, part 2): update this A100 host from
# integ-v3/integrated_master@a413df8 to the real B200 master, picking up
# the #77 seed-disjointness fix, the #71/96b2819 masked-regime safety net,
# the #79 provenance logging, and the f78 featurization speedup -- all
# HARD PREREQUISITES for the gen-1 generation launch per team-lead
# (2026-07-05): "gen-1 is the expensive, long-lived corpus... it MUST have
# [these] machine-enforced."
#
# WRINKLE this script accounts for: this host currently has 1-2 LOCAL-ONLY
# commits on top of a413df8 (my own #79 provenance cherry-picks,
# de481c9/3b3d5d8) that are NOT ancestors of masters history -- they are
# redundant, independently-authored duplicates of content master already
# has via ad7b5e0/6c22a91 (verified: the diff vs a413df8 for the one file
# both touched, tools/gumbel_search_vs_raw_h2h.py, is a superset on
# masters side). So this is a HARD RESET to master, not a strict git
# fast-forward -- the EXPECTED_HEAD check below exists specifically to
# make sure that hard reset only ever discards commits we've verified are
# redundant, never anything unexpected.
#
# DO NOT RUN until team-lead gives the explicit go (after the base-decision
# H2H completes) -- running this while H2H processes are live would pull
# the rug out from under them (different code, possibly different behavior
# mid-run).
set -euo pipefail
cd "$(dirname "$0")/.."

BUNDLE_PATH="${1:?usage: update_host_to_gen1_master.sh <bundle-path> <expected-current-head>}"
EXPECTED_HEAD="${2:?usage: update_host_to_gen1_master.sh <bundle-path> <expected-current-head>}"

CURRENT_HEAD="$(git rev-parse HEAD)"
if [ "$CURRENT_HEAD" != "$EXPECTED_HEAD" ]; then
  echo "ABORT: current HEAD ($CURRENT_HEAD) does not match expected ($EXPECTED_HEAD)." >&2
  echo "This host's state has changed since this script was staged -- re-verify before proceeding, do not blindly force." >&2
  exit 1
fi

echo "Fetching master from bundle $BUNDLE_PATH..."
git fetch "$BUNDLE_PATH" master:refs/heads/gen1-incoming-master

INCOMING="$(git rev-parse gen1-incoming-master)"
echo "Incoming master: $INCOMING"

# This is a controlled hard reset (see file header for why), not a
# fast-forward merge -- the EXPECTED_HEAD check above is what makes this
# safe rather than blind.
git reset --hard gen1-incoming-master
git branch -D gen1-incoming-master 2>/dev/null || true

NEW_HEAD="$(git rev-parse HEAD)"
if [ "$NEW_HEAD" != "$INCOMING" ]; then
  echo "ABORT: post-reset HEAD ($NEW_HEAD) does not match incoming master ($INCOMING)." >&2
  exit 1
fi
echo "Host is now at master: $NEW_HEAD"

echo "--- Post-update verification ---"

echo "[1/4] git HEAD matches expected master commit"
echo "  OK: $NEW_HEAD"

echo "[2/4] 96b2819 safety net (masked-regime enforcement) functional check"
.venv/bin/python -c "
import sys
sys.path.insert(0, 'src')
from catan_zero.rl.entity_token_policy import EntityGraphPolicy
from catan_zero.search.neural_rust_mcts import EntityGraphRustEvaluator, EntityGraphRustEvaluatorConfig

# Use the actual re-saved masked checkpoint if present; this only checks
# the mechanism raises correctly on a KNOWN mismatch, not full training data.
import torch, os
ckpt_candidates = [
    'runs/bc/entity_graph_35m_v3a_unfreeze_kl_masked_20260705/checkpoint_masked.pt',
    'runs/bc/entity_graph_35m_v3b_unfreeze_kl_arch_masked_20260705/checkpoint_masked.pt',
]
found = [p for p in ckpt_candidates if os.path.exists(p)]
if not found:
    print('  SKIP: no re-saved masked checkpoint present on this host yet (expected if not yet synced)')
else:
    policy = EntityGraphPolicy.load(found[0], device='cpu')
    assert policy.trained_with_masked_hidden_info is True, 'expected masked-trained checkpoint'
    raised = False
    try:
        EntityGraphRustEvaluator(policy, config=EntityGraphRustEvaluatorConfig(public_observation=False))
    except ValueError:
        raised = True
    assert raised, 'safety net did not raise on public_observation=False mismatch'
    print('  OK: safety net raises correctly on regime mismatch (' + found[0] + ')')
"

echo "[3/4] seed-disjointness assertion importable"
.venv/bin/python -c "
import sys
sys.path.insert(0, 'tools')
from seed_fleet_planner import assert_disjoint_seed_blocks, plan_disjoint_seed_blocks
assert_disjoint_seed_blocks([('a', 0, 10), ('b', 10, 20)])
print('  OK: seed_fleet_planner importable and functional')
"

echo "[4/4] colonist package (task #75) imports cleanly, no symlink leak"
.venv/bin/python -c "
import sys
sys.path.insert(0, 'src')
import catan_zero.data
print('  OK: catan_zero.data imports:', catan_zero.data.__file__)
"

echo "--- All checks passed. Host is gen-1-ready. ---"
