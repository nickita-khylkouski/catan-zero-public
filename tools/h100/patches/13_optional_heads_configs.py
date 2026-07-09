#!/usr/bin/env python3
"""
SYSTEM_DESIGN_FINDINGS #31: Enable optional model heads for champion_v0 warm start.

champion_v0 has ALL optional heads disabled:
  - value_uncertainty_head: False (KataGo-style uncertainty)
  - value_categorical_bins: 0 (HL-Gauss distributional value)
  - value_attention_pool: False
  - action_target_gather: False
  - action_cross_attention_layers: 0
  - belief_chance_spectra: False

These are all implemented, warm-start safe (zero-initialized), and potentially
high-impact for Catan's stochastic dynamics. This script creates training
launch commands for 3 experimental configs.

Usage: python3 13_optional_heads_configs.py
"""
import sys
print(__doc__)
print("""
=== EXPERIMENTAL TRAINING CONFIGS (Finding #31) ===

All configs use --init-checkpoint champion_v0.pt as warm start.
The optional heads are zero-initialized at init, so the warm-started model
is bit-identical to champion_v0 on the first forward pass. Training then
gradually learns to use the new heads.

--- Config A: Distributional Value (HL-Gauss) ---
Hypothesis: Catan is highly stochastic (dice rolls). A distributional value
head (CAT-39) can represent the full distribution of outcomes, not just the
mean. Shown to beat MSE for stochastic dynamics in MuZero/AlphaZero variants.

python tools/train_bc.py \\
  --arch entity_graph \\
  --init-checkpoint ~/bundle/champion_v0.pt \\
  --grow-from-checkpoint \\
  --value-categorical-bins 51 \\
  --data ~/corpora/gen5_pooled \\
  --data-format memmap \\
  --batch-size 4096 \\
  --lr 2e-4 --lr-schedule cosine \\
  --epochs 2 \\
  --checkpoint runs/bc/gen5_distval.pt \\
  --policy-loss-weight 1.0 --value-loss-weight 0.5 \\
  --winner-sample-weight 1.0 --loser-sample-weight 0.5 \\
  --mask-hidden-info \\
  --track 2p_no_trade --vps-to-win 10

--- Config B: Value Uncertainty (KataGo-style) ---
Hypothesis: Predicting short-term value uncertainty enables uncertainty-weighted
MCTS backup, reducing variance in search. KataGo uses this for game-theoretic
draw detection and search focus.

python tools/train_bc.py \\
  --arch entity_graph \\
  --init-checkpoint ~/bundle/champion_v0.pt \\
  --grow-from-checkpoint \\
  --value-uncertainty-head \\
  --data ~/corpora/gen5_pooled \\
  --data-format memmap \\
  --batch-size 4096 \\
  --lr 2e-4 --lr-schedule cosine \\
  --epochs 2 \\
  --checkpoint runs/bc/gen5_uncval.pt \\
  --policy-loss-weight 1.0 --value-loss-weight 0.5 \\
  --winner-sample-weight 1.0 --loser-sample-weight 0.5 \\
  --mask-hidden-info \\
  --track 2p_no_trade --vps-to-win 10

--- Config C: Action Cross-Attention ---
Hypothesis: Actions attending to board tokens (f69) can improve policy quality
on complex decisions (settlement placement, road building).

python tools/train_bc.py \\
  --arch entity_graph \\
  --init-checkpoint ~/bundle/champion_v0.pt \\
  --grow-from-checkpoint \\
  --action-cross-attention-layers 2 \\
  --data ~/corpora/gen5_pooled \\
  --data-format memmap \\
  --batch-size 4096 \\
  --lr 2e-4 --lr-schedule cosine \\
  --epochs 2 \\
  --checkpoint runs/bc/gen5_xattn.pt \\
  --policy-loss-weight 1.0 --value-loss-weight 0.5 \\
  --winner-sample-weight 1.0 --loser-sample-weight 0.5 \\
  --mask-hidden-info \\
  --track 2p_no_trade --vps-to-win 10

=== GATE MATCH ===
After training, run a head-to-head gate match against champion_v0:
  python tools/gumbel_search_cross.py \\
    --candidate runs/bc/gen5_distval.pt \\
    --baseline ~/bundle/champion_v0.pt \\
    --games 200 --n-full 64

If any config wins the gate (>55% win rate), deploy it to the fleet.
""")
sys.exit(0)
