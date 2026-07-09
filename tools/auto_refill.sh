#!/usr/bin/env bash
# Auto-refill watchdog: relaunches gen-3 generation on any idle GPU (cron */10).
# Seeds: host-owned counter in runs/.seed_counter (200k stride, GAMES=1500).
cd "$(dirname "$0")/.." || exit 1
LO=6100000000; HI=6200000000
PIPE=$(ls -d /tmp/mps_pipe* 2>/dev/null | head -1)
# Champion pointer (promotion runbook): update runs/CURRENT_CHAMPION on rotation —
# one file per host instead of editing this script (ml-czar, 2026-07-07).
CKPT=$(cat runs/CURRENT_CHAMPION 2>/dev/null || echo runs/bc/gen3_20260706/checkpoint.pt)
for g in 0 1 2 3 4 5 6 7; do
  busy=0
  for pid in $(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    comm=$(ps -p $pid -o comm= 2>/dev/null); case "$comm" in *mps*) ;; *) busy=1;; esac
  done
  [ $busy = 1 ] && continue
  seed=$(cat runs/.seed_counter)
  [ $seed -ge $HI ] && { echo "$(date -u) seed block exhausted" >> runs/auto_refill.log; exit 0; }
  echo $(( seed + 200000 )) > runs/.seed_counter
  DIR=runs/selfplay/gen3_auto/$(date -u +%m%d_%H%M)_gpu$g
  mkdir -p "$DIR"
  CUDA_MPS_PIPE_DIRECTORY=$PIPE CUDA_VISIBLE_DEVICES=$g nohup .venv/bin/python tools/generate_gumbel_selfplay_data.py     --out-dir "$DIR" --games 1500 --workers 16 --base-seed $seed --shard-size 2048     --checkpoint "$CKPT" --device cuda     --n-full 64 --n-fast 16 --p-full 0.25 --c-visit 50.0 --c-scale 0.03     --max-decisions 600 --max-depth 80 --temperature-decisions 90     --correct-rust-chance-spectra --lazy-interior-chance --public-observation     --track 2p_no_trade --vps-to-win 10 --format npz --score-actions     > "$DIR/launch.log" 2>&1 &
  echo "owner=auto-refill pid=$! gpu=$g seed=$seed ts=$(date -u +%FT%TZ)" > "$DIR/.claim"
  echo "$(date -u) refilled gpu$g seed=$seed ckpt=$CKPT" >> runs/auto_refill.log
done
