#!/usr/bin/env bash
# Phase-2 window-feed deploy (task #94) — run on the B200 ONLY after team-lead approval
# and ONLY after the flywheel has exited cleanly on a STOP at a round boundary.
#
# Usage:  bash ~/catan-zero-f94feed/tools/deploy_phase2.sh
# It refuses to run while the orchestrator process is alive.
set -euo pipefail

LIVE=/home/ubuntu/catan-zero
DEV=/home/ubuntu/catan-zero-f94feed
LOOP=$LIVE/runs/flywheel_20260707b

# 0. safety: the loop must be down (STOP-exited), and this must not run twice concurrently
if pgrep -f "tools/continuous_flywheel.py --loop-dir $LOOP" >/dev/null; then
  echo "REFUSING: flywheel orchestrator still running. touch $LOOP/STOP and wait for clean exit."
  exit 1
fi

# 1. backup (never overwrite an existing backup — one deploy, one backup)
B=$LOOP/backup_prephase2
if [ -e "$B" ]; then echo "REFUSING: $B already exists (previous deploy?)"; exit 1; fi
mkdir -p "$B"
cp -v $LIVE/tools/continuous_flywheel.py $LIVE/tools/train_bc.py \
      $LIVE/src/catan_zero/rl/flywheel/replay_window.py \
      $LOOP/flywheel_state.json $LOOP/window_state.json "$B/"

# 2. deploy the four files + tests
cp -v $DEV/tools/train_bc.py                $LIVE/tools/train_bc.py
cp -v $DEV/tools/continuous_flywheel.py    $LIVE/tools/continuous_flywheel.py
cp -v $DEV/tools/flywheel_feed_daemon.py   $LIVE/tools/flywheel_feed_daemon.py
cp -v $DEV/src/catan_zero/rl/flywheel/replay_window.py \
      $LIVE/src/catan_zero/rl/flywheel/replay_window.py
cp -v $DEV/tests/test_concat_memmap_corpus.py \
      $DEV/tests/test_flywheel_phase2_integration.py $LIVE/tests/

# 3. post-deploy sanity in the LIVE tree (no GPU, no state writes)
( cd $LIVE && python3 src/catan_zero/rl/flywheel/replay_window.py )
( cd $LIVE && python3 -c "import ast; [ast.parse(open(f).read()) for f in ('tools/continuous_flywheel.py','tools/train_bc.py','tools/flywheel_feed_daemon.py')]; print('deploy syntax OK')" )

echo
echo "DEPLOYED. Next steps (manual, in order):"
echo "  1. rm $LOOP/STOP"
echo "  2. relaunch flywheel (same production flags + --max-ckpt-lag 2), update $LOOP/.claim"
echo "  3. verify feed config at $LOOP/feed_config.json, then:"
echo "     cd $LIVE && PYTHONUNBUFFERED=1 nohup .venv/bin/python tools/flywheel_feed_daemon.py \\"
echo "        --loop-dir $LOOP --config $LOOP/feed_config.json > $LOOP/feed/daemon.out 2>&1 &"
