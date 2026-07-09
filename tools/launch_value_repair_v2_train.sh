#!/bin/bash
cd /home/ubuntu/catan-zero || exit 1
exec .venv/bin/python -m tools.train_bc \
  --arch entity_graph --device cuda:1 \
  --data runs/raw_selfplay_gen1_subset \
  --init-checkpoint runs/bc/entity_graph_35m_oldbase_hardtarget_ab45_robber_opening_20260630_220320/checkpoint.pt \
  --checkpoint runs/bc/entity_graph_35m_value_repair_v2_raw_selfplay_20260704/checkpoint.pt \
  --report runs/bc/entity_graph_35m_value_repair_v2_raw_selfplay_20260704/report.json \
  --hidden-size 640 --graph-layers 6 --attention-heads 8 --graph-dropout 0.05 \
  --track 2p_no_trade --vps-to-win 10 \
  --train-value-only --policy-loss-weight 0 \
  --lr 1e-4 --lr-warmup-steps 176 \
  --batch-size 4096 --amp bf16 --epochs 1 \
  --truncated-vp-margin-value-weight 0.25 \
  --trust-curated-data-quality \
  --validation-game-seed-ranges "5006335:5006667,5106335:5106667,6406335:6406667,6506335:6506667,6605701:6606000,6706335:6706667,7006335:7006667,7106335:7106667,7206335:7206667,7306335:7306667,7406335:7406667,7506335:7506667,7706335:7706667" \
  > runs/bc/entity_graph_35m_value_repair_v2_raw_selfplay_20260704/train_v4.log 2>&1
