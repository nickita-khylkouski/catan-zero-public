# 35M Teacher Root-Fix Plan

Date: 2026-06-28

Objective: get the 35M teacher-trained model above 50% win rate versus AB3 before PPO. Do not brute-force more epochs on suspect data. Fix the root data and representation issues, regenerate verified data, then retrain and gate.

## Current Diagnosis

The current best 35M checkpoint is not close to the target:

- vs random: 155/160 = 96.9%
- vs heuristic: 87/160 = 54.4%
- vs jsettlers_lite: 56/160 = 35.0%
- vs catanatron_value: 12/160 = 7.5%
- vs AB3: 13/160 = 8.1%
- vs AB4: 17/160 = 10.6%
- vs value_rollout_search: 8/160 = 5.0%

This is not a simple undertraining problem. Validation top-1 around 45% and top-3 around 75% means the model is learning something, but game strength is not transferring.

## Root Issues Found

### 1. AB Teacher RNG Contaminated Game State

Fresh semantic replay audits failed before the fix:

- AB-only fresh data diverged at decision 15.
- Graph-history mixed data diverged at decision 9.

Cause: `CatanatronAlphaBetaPolicy._root_search()` called Catanatron alpha-beta search, which consumes Python global `random`, and did not restore the random state. Since the environment also depends on Python global `random`, teacher search changed future dice/resource randomness during data generation.

Fix applied:

- Save `random.getstate()` before alpha-beta.
- Restore it in `finally`.
- Added regression test: `test_alphabeta_root_search_restores_global_random_state`.

Post-fix semantic replay:

- AB-only smoke: 168/168 rows replayed, 0 mismatches.
- Graph-history mixed smoke: 411/411 rows replayed, 0 mismatches.

Consequence: pre-fix AB-heavy shards are suspect. They can be archived for diagnostics, but they should not be used for the next production BC run.

### 2. The 35M "Graph" Model Was Not Getting Graph-History Inputs

The model is named `xdim_graph`, but its encoder currently chunks a flat observation vector into arbitrary tokens:

```text

obs -> resize -> [batch, token_count, chunk_size] -> transformer blocks

```

That is not a true Catan board graph. The repo already has a public graph/history feature suffix (`graph_history_features.py`), but current teacher generation used `parse_track()` without `use_graph_history_features=True`.

Fix applied:

- `parse_track(..., use_graph_history_features=True)` support.
- `tools/generate_teacher_data.py --graph-history-features`.
- Modal payload support for `graph_history_features`.

Consequence: all next production teacher data for the 35M model should use `--graph-history-features`.

### 3. High-Leverage Phases Are Weak

Current validation has the weakest accuracy where Catan is most strategic:

- initial_build top-1: about 33%
- robber top-1: about 21%

More generic mixed data will mostly add main-turn and forced decisions. We need phase-targeted data and metrics.

Required before promotion:

- opening/initial_build eval
- robber eval
- per-phase top-1/top-3
- per-teacher top-1/top-3

### 4. Mixed Teacher BC Can Average Away Strong Play

AB4, AB5, value_rollout_search, value, and jsettlers_lite disagree. One mixed model trained with uniform or weak weights can become a compromise policy that beats none of them.

Next training must compare:

- mixed strong corpus
- AB5-only specialist
- AB4/AB5/VRS-only specialist
- teacher-conditioned diagnostic if mixed remains weak

### 5. Data Scale Must Be Verified, Not Blind

We likely need tens of millions of useful rows, but not raw rows. Forced moves and roll rows are not policy learning signal.

Use production curation:

- drop roll rows from policy loss
- set forced rows to value-only or near-zero policy weight
- preserve clean terminal rows for value/final-VP targets
- fail closed on missing soft targets, labels, or outcomes

## New Data Plan

### Wave A: Verified 2p10 No-Trade Strong Corpus

Run on Modal 600 CPUs:

```bash
.venv/bin/modal run --detach tools/modal_teacher_factory.py::launch_600_ab45 \
  --containers 75 \
  --games-per-container 128 \
  --cpu-workers 8 \
  --teacher-sampling-weights 'catanatron_ab5=4.0,catanatron_ab4=3.5,value_rollout_search=3.0,catanatron_ab3=0.5,catanatron_value=0.25,jsettlers_lite=0.5' \
  --commit-every-chunks 4 \
  --graph-history-features
```

Expected per wave:

- 9,600 games
- roughly 2M-3M raw samples
- graph-history observations
- AB root soft scores and anchored soft policies

Repeat until at least:

- 30M-50M raw rows
- 15M+ policy-active rows after curation
- 1M+ initial_build rows if possible
- 1M+ robber rows if possible

### Wave B: Hard-Phase Corpus

Add generation modes that record focused windows:

- opening-only: first 8-12 decisions
- robber-heavy: games/states where robber decisions occur
- discard-heavy: discard states

If exact phase filtering at generation is not available, generate full games and curate with high phase weights.

### Wave C: Robustness Corpus

Keep lower weight:

- AB2/AB3
- catanatron_value
- jsettlers_lite
- random/noisy only for robustness, not as main teacher

This should be less than 15%-20% of policy-active training mass.

## Quality Gates For Every Corpus

Run:

```bash
python tools/audit_teacher_semantics.py \
  --data DATA_DIR \
  --track 2p_no_trade \
  --vps-to-win 10 \
  --graph-history-features \
  --max-seeds 64 \
  --max-rows 250000 \
  --out DATA_DIR/semantic_audit.json

python tools/report_teacher_data_quality.py \
  --data DATA_DIR \
  --production-35m-teacher \
  --out DATA_DIR/quality.json
```

Required:

- 0 invalid teacher actions
- 0 semantic replay mismatches on audited seeds
- clean terminal outcome fraction >= 0.99
- final actual VP fraction >= 0.99
- AB root score fraction high for AB teachers
- soft target coverage high
- forced policy rows controlled by curation

## Training Plan

### Step 1: Curate Corrected Data

```bash
python tools/curate_teacher_data.py \
  --data RAW_MODAL_DIR \
  --production-35m-teacher \
  --out runs/teacher/curated_rngfix_graph_v1
```

### Step 2: Train New 35M Graph-History BC

Train from scratch or from the best compatible graph-history checkpoint only. Do not fine-tune the old flat-observation checkpoint into graph-history data unless config compatibility is explicit.

Baseline command shape:

```bash
torchrun --standalone --nproc_per_node=2 tools/train_bc.py \
  --arch xdim_graph \
  --data runs/teacher/curated_rngfix_graph_v1 \
  --epochs 3 \
  --batch-size 16384 \
  --hidden-size 768 \
  --graph-tokens 32 \
  --graph-layers 6 \
  --soft-target-source prefer_scores \
  --soft-target-weight 0.7 \
  --soft-target-temperature 0.7 \
  --forced-action-weight 0.05 \
  --winner-sample-weight 1.0 \
  --loser-sample-weight 0.25 \
  --value-loss-weight 0.25 \
  --final-vp-loss-weight 0.05 \
  --teacher-weights 'catanatron_ab5=1.5,catanatron_ab4=1.3,value_rollout_search=1.3,catanatron_ab3=0.7,catanatron_value=0.4,jsettlers_lite=0.5' \
  --phase-weights 'initial_build=3.0,robber=3.0,discard=1.5,main_turn=1.0' \
  --require-35m-model
```

### Step 3: Scoreboard Gate

Run at least 1,000-2,000 games per opponent for dev, then 10,000 for promotion.

Minimum dev gate:

```bash
python tools/evaluate_scoreboard.py \
  --candidate CHECKPOINT \
  --games 2000 \
  --tracks 2p_no_trade \
  --opponents random,heuristic,jsettlers_lite,catanatron_value,catanatron_ab3,catanatron_ab4,catanatron_ab5,value_rollout_search \
  --workers 120 \
  --device cpu \
  --out runs/scoreboards/CHECKPOINT_dev_2k.json
```

Promotion target for this phase:

- AB3 win rate > 50%
- no regression vs random/heuristic
- improving vs catanatron_value and AB4
- 0 illegal actions
- no stuck-game spike

## Next Architecture Step If Still Weak

If corrected graph-history data improves imitation but still cannot approach AB3:

1. Keep the dot-product legal-action head.
2. Replace arbitrary flat chunk tokens with entity tokens:
   - 19 hex tokens
   - 54 vertex tokens
   - 72 edge tokens
   - player tokens
   - global token
3. Keep parameter target around 35M.
4. Train on the same corrected corpus for a fair architecture comparison.

Do not start PPO until the BC checkpoint is at least competitive with AB3.

