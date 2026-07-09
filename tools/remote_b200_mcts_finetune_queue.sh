#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/ubuntu/catan-zero}"
PY="${PY:-.venv/bin/python}"
BASE_CKPT="${BASE_CKPT:-runs/bc/entity_graph_35m_2p10_hq_1000parts_cumulative_shardddp_bs4k_20260629_200420/checkpoint_epoch0002.pt}"
IMPORT_ROOT="${IMPORT_ROOT:-runs/data/a100_imports}"
MIN_ROWS="${MIN_ROWS:-100000}"
MAX_ROWS="${MAX_ROWS:-0}"
WAIT_SECONDS="${WAIT_SECONDS:-600}"
RUN_TAG="${RUN_TAG:-mcts_import_ft_$(date +%Y%m%d_%H%M%S)}"

cd "$ROOT"
mkdir -p logs runs/bc runs/data/curated

count_rows() {
  "$PY" - <<'PY'
from pathlib import Path
import numpy as np
import os
import subprocess
import tempfile

root = Path(os.environ.get("IMPORT_ROOT", "runs/data/a100_imports"))
rows = 0
files = [
    path for path in root.glob("**/entity_teacher_shard_*")
    if path.name.endswith(".npz") or path.name.endswith(".npz.zst")
]
for p in files:
    path = str(p)
    if path.endswith(".zst"):
        raw = subprocess.check_output(["zstd", "-dc", path])
        fd, tmp = tempfile.mkstemp(suffix=".npz")
        os.write(fd, raw)
        os.close(fd)
        arr = np.load(tmp, allow_pickle=False)
        os.unlink(tmp)
    else:
        arr = np.load(path, allow_pickle=False)
    rows += len(arr["action_taken"])
print(rows)
PY
}

wait_for_rows() {
  while true; do
    local rows
    rows="$(IMPORT_ROOT="$IMPORT_ROOT" count_rows)"
    echo "$(date -Is) b200_queue rows=$rows min_rows=$MIN_ROWS import_root=$IMPORT_ROOT"
    if [ "$rows" -ge "$MIN_ROWS" ]; then
      return 0
    fi
    sleep "$WAIT_SECONDS"
  done
}

make_flat_dataset() {
  local name="$1"
  local out="runs/data/curated/$name"
  rm -rf "$out"
  mkdir -p "$out"
  "$PY" - <<'PY' "$IMPORT_ROOT" "$out" "$MAX_ROWS"
from pathlib import Path
import os
import sys
import numpy as np
import subprocess
import tempfile

src = Path(sys.argv[1])
out = Path(sys.argv[2])
max_rows = int(sys.argv[3])
rows = 0
count = 0
for shard in sorted(src.glob("**/entity_teacher_shard_*")):
    if not (shard.name.endswith(".npz") or shard.name.endswith(".npz.zst")):
        continue
    if max_rows > 0 and rows >= max_rows:
        break
    target = out / f"entity_teacher_shard_{count:06d}{''.join(shard.suffixes)}"
    if not target.exists():
        target.symlink_to(shard.resolve())
    path = str(shard)
    if path.endswith(".zst"):
        raw = subprocess.check_output(["zstd", "-dc", path])
        fd, tmp = tempfile.mkstemp(suffix=".npz")
        os.write(fd, raw)
        os.close(fd)
        arr = np.load(tmp, allow_pickle=False)
        os.unlink(tmp)
    else:
        arr = np.load(path, allow_pickle=False)
    rows += len(arr["action_taken"])
    count += 1
print({"dataset": str(out), "shards": count, "rows": rows}, file=sys.stderr)
PY
  echo "$out"
}

run_train() {
  local name="$1"
  local data="$2"
  local init="$3"
  local epochs="$4"
  local lr="$5"
  local soft_weight="$6"
  local q_weight="$7"
  local out="runs/bc/$name"
  mkdir -p "$out"
  if [ -f "$out/report.json" ]; then
    echo "$(date -Is) b200_queue train_exists name=$name"
    return 0
  fi
  echo "$(date -Is) b200_queue train_start name=$name data=$data init=$init"
  PYTHONPATH="$ROOT:${PYTHONPATH:-}" "$PY" -m torch.distributed.run --standalone --nproc_per_node=2 tools/train_bc.py \
    --arch entity_graph \
    --data "$data" \
    --track 2p_no_trade \
    --vps-to-win 10 \
    --epochs "$epochs" \
    --batch-size 4096 \
    --validation-fraction 0.05 \
    --validation-max-samples 100000 \
    --hidden-size 640 \
    --graph-layers 6 \
    --attention-heads 8 \
    --graph-dropout 0.05 \
    --lr "$lr" \
    --optimizer adamw \
    --weight-decay 0.01 \
    --fused-optimizer \
    --amp bf16 \
    --soft-target-source policy \
    --soft-target-weight "$soft_weight" \
    --soft-target-temperature 0.7 \
    --soft-target-min-legal-coverage 0.01 \
    --value-loss-weight 0.15 \
    --final-vp-loss-weight 0.02 \
    --q-loss-weight "$q_weight" \
    --allow-teacher-score-q-loss \
    --forced-action-weight 0.05 \
    --winner-sample-weight 1.0 \
    --loser-sample-weight 0.5 \
    --init-checkpoint "$init" \
    --checkpoint "$out/checkpoint.pt" \
    --report "$out/report.json" \
    --save-each-epoch \
    --progress-every-batches 20 \
    --skip-teacher-quality-gate \
    --ddp-shard-data \
    > "$out/train.log" 2>&1
  echo "$(date -Is) b200_queue train_done name=$name status=$?"
}

main() {
  if pgrep -af '[t]rain_bc.py|[t]orchrun' >/dev/null; then
    echo "$(date -Is) b200_queue existing_training_detected refusing_to_stack"
    exit 0
  fi

  wait_for_rows
  local dataset
  dataset="$(make_flat_dataset "${RUN_TAG}_dataset")"

  local exp1="entity_graph_35m_${RUN_TAG}_mcts_policy"
  run_train "$exp1" "$dataset" "$BASE_CKPT" 2 5e-5 1.0 0.0

  local exp2="entity_graph_35m_${RUN_TAG}_mcts_qaux"
  run_train "$exp2" "$dataset" "runs/bc/$exp1/checkpoint.pt" 1 3e-5 0.8 0.03

  echo "$(date -Is) b200_queue complete dataset=$dataset exp1=$exp1 exp2=$exp2"
}

main "$@"
