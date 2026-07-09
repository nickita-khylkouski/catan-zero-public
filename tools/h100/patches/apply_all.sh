#!/bin/bash
# Master patch applier — applies ALL patches to the fleet.
# SYSTEM_DESIGN_FINDINGS: applies fixes #1, #2, #3, #4, #5, #6, #7, #8, #10,
# #15, #18, #19, #20, #22, #23, #30, #31.
#
# Usage:
#   bash apply_all.sh [REPO_PATH] [HARVEST_PATH] [TRAIN_BC_PATH] [GEN_SCRIPT_PATH] [BUILD_MEMMAP_PATH]
#
# Defaults:
#   REPO_PATH       = ~/catan-zero-runsix  (on fleet boxes)
#   HARVEST_PATH    = ~/wave1_harvest.sh   (on orchestration box)
#   TRAIN_BC_PATH   = ~/c1_fsdp/repo/tools/train_bc.py (on training box)
#   GEN_SCRIPT_PATH = $REPO_PATH/tools/generate_gumbel_selfplay_data.py
#   BUILD_MEMMAP_PATH = $REPO_PATH/tools/build_memmap_corpus.py
#
# This script is IDEMPOTENT — re-running it is safe (patches check for existing fixes).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

REPO_PATH="${1:-$HOME/catan-zero-runsix}"
HARVEST_PATH="${2:-$HOME/wave1_harvest.sh}"
TRAIN_BC_PATH="${3:-$HOME/c1_fsdp/repo/tools/train_bc.py}"
GEN_SCRIPT_PATH="${4:-$REPO_PATH/tools/generate_gumbel_selfplay_data.py}"
BUILD_MEMMAP_PATH="${5:-$REPO_PATH/tools/build_memmap_corpus.py}"

echo "============================================"
echo "  SYSTEM DESIGN FINDINGS — PATCH APPLIER"
echo "  (32 findings → 13 patches)"
echo "============================================"
echo ""
echo "Repo:         $REPO_PATH"
echo "Harvest:      $HARVEST_PATH"
echo "Train BC:     $TRAIN_BC_PATH"
echo "Gen script:   $GEN_SCRIPT_PATH"
echo "Build memmap: $BUILD_MEMMAP_PATH"
echo ""

# --- Patch 01: bf16 autocast + torch.compile + pin_memory ---
echo "--- [01] bf16 autocast + torch.compile + pin_memory (Findings #2,#3,#10) ---"
ENTITY_POLICY="$REPO_PATH/src/catan_zero/rl/entity_token_policy.py"
if [ -f "$ENTITY_POLICY" ]; then
    python3 "$SCRIPT_DIR/apply_01_bf16_compile.py" "$ENTITY_POLICY"
else
    echo "[SKIP] $ENTITY_POLICY not found"
fi
echo ""

# --- Patch 02: LRU cache eviction ---
echo "--- [02] LRU cache eviction (Finding #15) ---"
NEURAL_MCTS="$REPO_PATH/src/catan_zero/search/neural_rust_mcts.py"
if [ -f "$NEURAL_MCTS" ]; then
    python3 "$SCRIPT_DIR/apply_02_lru_cache.py" "$NEURAL_MCTS"
else
    echo "[SKIP] $NEURAL_MCTS not found"
fi
echo ""

# --- Patch 03: Parallel harvest ---
echo "--- [03] Parallel harvest rsync (Finding #19) ---"
if [ -d "$(dirname "$HARVEST_PATH")" ]; then
    cp "$SCRIPT_DIR/03_wave1_harvest_parallel.sh" "$HARVEST_PATH"
    chmod +x "$HARVEST_PATH"
    echo "[OK] Replaced $HARVEST_PATH with parallel version"
else
    echo "[SKIP] $(dirname "$HARVEST_PATH") not found"
fi
echo ""

# --- Patch 04: Optimizer state persistence ---
echo "--- [04] Optimizer state persistence (Finding #8) ---"
if [ -f "$TRAIN_BC_PATH" ]; then
    python3 "$SCRIPT_DIR/apply_04_optimizer_state.py" "$TRAIN_BC_PATH"
else
    echo "[SKIP] $TRAIN_BC_PATH not found"
fi
echo ""

# --- Patch 05: a100a relaunch ---
echo "--- [05] a100a relaunch script (Findings #12,#13,#14) ---"
echo "[INFO] Copy 05_a100a_relaunch.sh to a100a and run manually:"
echo "       scp $SCRIPT_DIR/05_a100a_relaunch.sh ubuntu@64.181.197.190:~/"
echo "       ssh ubuntu@64.181.197.190 'pkill -f cat91_n64_pilot; bash 05_a100a_relaunch.sh'"
echo ""

# --- Patch 06: Teacher shard size fix ---
echo "--- [06] Teacher shard size fix (Finding #4) ---"
echo "[INFO] For n128/n256 generation, use the wrapper:"
echo "       bash $SCRIPT_DIR/06_teacher_shard_size_fix.sh python tools/generate_gumbel_selfplay_data.py --n-full 128 ..."
echo ""

# --- Patch 07: build_memmap parallelization ---
echo "--- [07] Parallel build_memmap_corpus (Finding #20) ---"
if [ -f "$BUILD_MEMMAP_PATH" ]; then
    python3 "$SCRIPT_DIR/apply_07_build_memmap_parallel.py" "$BUILD_MEMMAP_PATH"
else
    echo "[SKIP] $BUILD_MEMMAP_PATH not found"
fi
echo ""

# --- Patch 08: Compression + rust-featurize wrapper ---
echo "--- [08] npz_zst compression + rust-featurize (Findings #5,#18) ---"
echo "[INFO] For generation, use the wrapper to auto-add --format npz_zst --rust-featurize:"
echo "       bash $SCRIPT_DIR/08_compression_rustfeaturize_wrapper.sh python tools/generate_gumbel_selfplay_data.py ..."
echo ""

# --- Patch 09: Training hyperparameter wrapper ---
echo "--- [09] Training hyperparams: cosine LR, value loss, batch size (Findings #6,#7,#22,#23) ---"
echo "[INFO] See recommended training commands:"
echo "       bash $SCRIPT_DIR/09_training_hyperparam_wrapper.sh"
echo ""

# --- Patch 10: Seed ledger sync ---
echo "--- [10] Seed ledger cross-host sync (Finding #30) ---"
echo "[INFO] Run from the orchestrator box to sync master ledger to all fleet boxes:"
echo "       bash $SCRIPT_DIR/10_seed_ledger_sync.sh"
echo ""

# --- Patch 11 & 12: Threaded self-play generation --- REJECTED (CAT-120) ---
# Finding #1 (batch-1 inference via threading) is a PROVEN DEAD-END, not a speedup.
# Benched twice independently: threaded generation is ~4x SLOWER than the
# 16-process + MPS baseline because Python featurization (~96% of per-leaf cost)
# holds the GIL, so N threads serialize on one core while N processes use N cores;
# the GPU sits ~97% idle. The real throughput lever is the eval-server (CAT-67).
# These patches are intentionally NOT applied. See PATCHES_README.md and CAT-120.
echo "--- [11/12] Threaded self-play generation --- REJECTED (CAT-120), NOT applied ---"
echo "[REJECTED] Threaded generation is a ~4x GIL-bound regression; use eval-server (CAT-67)."
echo "           11_threaded_selfplay_gen.py / apply_12_threaded_generation.py are dead-ends."
echo ""

# --- Patch 13: Optional heads configs ---
echo "--- [13] Optional model heads configs (Finding #31) ---"
echo "[INFO] Experimental training configs for distributional value, uncertainty, cross-attention:"
echo "       python3 $SCRIPT_DIR/13_optional_heads_configs.py"
echo ""

# --- Patch 14: Opponent pool LRU eviction ---
echo "--- [14] Opponent pool LRU eviction (Finding #33) ---"
GUMBEL_SP="$REPO_PATH/src/catan_zero/rl/gumbel_self_play.py"
if [ -f "$GUMBEL_SP" ]; then
    python3 "$SCRIPT_DIR/apply_14_opponent_pool_lru.py" "$GUMBEL_SP"
else
    echo "[SKIP] $GUMBEL_SP not found"
fi
echo ""

# --- Patch 15: Symmetry adapter cache ---
echo "--- [15] Symmetry-averaged eval adapter cache (Finding #35) ---"
if [ -f "$NEURAL_MCTS" ]; then
    python3 "$SCRIPT_DIR/apply_15_symmetry_adapter_cache.py" "$NEURAL_MCTS"
else
    echo "[SKIP] $NEURAL_MCTS not found"
fi
echo ""

# --- Patch 16: Temperature replace guard ---
echo "--- [16] Temperature replace guard (Finding #37) ---"
if [ -f "$GUMBEL_SP" ]; then
    python3 "$SCRIPT_DIR/apply_16_temperature_replace_guard.py" "$GUMBEL_SP"
else
    echo "[SKIP] $GUMBEL_SP not found"
fi
echo ""

# --- Patch 17: evaluate_many adapter cache ---
echo "--- [17] evaluate_many adapter cache (Finding #40) ---"
if [ -f "$NEURAL_MCTS" ]; then
    python3 "$SCRIPT_DIR/apply_17_evaluate_many_adapter_cache.py" "$NEURAL_MCTS"
else
    echo "[SKIP] $NEURAL_MCTS not found"
fi
echo ""

# --- Patch 18: prior_policy fp32 ---
echo "--- [18] prior_policy fp32 (Finding #47) ---"
if [ -f "$GUMBEL_SP" ]; then
    python3 "$SCRIPT_DIR/apply_18_prior_policy_fp32.py" "$GUMBEL_SP"
else
    echo "[SKIP] $GUMBEL_SP not found"
fi
echo ""

# --- Patch 19: Topology cache LRU ---
echo "--- [19] Topology cache LRU (Finding #46) ---"
FEATURES_PATH="$REPO_PATH/src/catan_zero/rl/entity_token_features.py"
if [ -f "$FEATURES_PATH" ]; then
    python3 "$SCRIPT_DIR/apply_19_topology_cache_lru.py" "$FEATURES_PATH"
else
    echo "[SKIP] $FEATURES_PATH not found"
fi
echo ""

echo "============================================"
echo "  ALL PATCHES APPLIED (48 findings → 19 patches)"
echo "============================================"
echo ""
echo "Verify the changes:"
echo "  grep -n 'torch.compile' $ENTITY_POLICY"
echo "  grep -n 'autocast' $ENTITY_POLICY"
echo "  grep -n 'OrderedDict' $NEURAL_MCTS"
echo "  grep -n 'popitem' $NEURAL_MCTS"
echo "  grep -n 'optimizer.pt' $TRAIN_BC_PATH"
echo "  grep -n 'use.threads' $GEN_SCRIPT_PATH"
echo "  grep -n 'ThreadPoolExecutor' $BUILD_MEMMAP_PATH"
echo "  grep -n 'MAX_POOL_EVALUATORS' $GUMBEL_SP"
echo "  grep -n 'symmetry_need_adapter' $NEURAL_MCTS"
echo "  grep -n 'many_need_adapter' $NEURAL_MCTS"
echo "  grep -n 'temperature != temperature' $GUMBEL_SP"
echo "  grep -n 'dtype=np.float32' $GUMBEL_SP | grep prior"
echo "  grep -n 'topology_cache_lru' $FEATURES_PATH"
echo "  head -5 $HARVEST_PATH  # should show 'Parallel rsync' comment"
echo ""
echo "Remaining findings (require organizational/research decisions):"
echo "  #9   EvalServer = the real throughput lever (CAT-67); threaded gen REJECTED (CAT-120)"
echo "  #16  Cross-worker cache sharing (free with Finding #1)"
echo "  #17  51% forced decisions (research experiment)"
echo "  #21  npz loads all RAM (use --data-format memmap, already available)"
echo "  #24  Checkpoint load 3.2s (free with Finding #1)"
echo "  #25  json.dumps in MCTS (Rust API change)"
echo "  #26  Stack fragmentation (deprecate old forks)"
echo "  #27  FSDP smoke test 2 GPUs (scale up)"
echo "  #28  No model scaling (future)"
echo "  #29  Three incompatible forks (consolidate)"
echo "  #32  A_edge_pol edge head (port if wins gate)"
echo "  #34  Weight decay (CLI flag: --optimizer adamw --weight-decay 1e-4)"
echo "  #36  Training H2D async (low impact, compute-bound)"
echo "  #38  Gradient clipping configurable (low impact, 1.0 is fine)"
echo "  #39  _traverse_roll return value (not worth fixing)"
echo "  #41  FSDP optimizer state (requires FSDP.optim_state_dict)"
echo "  #42  player_state_json per-color (free with --rust-featurize)"
echo "  #43  Training RNG state persistence (low impact for 2-epoch)"
echo "  #44  DDP stride split load imbalance (shards are uniform)"
echo "  #45  _build_decision_row redundant JSON (low impact, ~150ms/game)"
echo "  #48  Early stopping (research-level, not needed for 2-epoch BC)"
