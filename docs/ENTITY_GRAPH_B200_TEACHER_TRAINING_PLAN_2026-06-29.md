# Entity-Graph 35M B200 Teacher Training Plan

> **Historical plan, not a current production recipe.** The retained
> `loser_sample_weight=0.3` command records the old experiment exactly. Current
> MCTS policy distillation uses `1.0` because eventual game loss does not
> invalidate a searched policy target.

Objective: replace the flat/chunked 35M BC model with a true Catan entity-token model, keep all current teacher data, convert it into typed tensors, and train on the 2x B200 box until the checkpoint is competitive with the strongest teachers before PPO.

## Current Decision

Use the new `entity_graph` architecture as the production BC model:

- 33.6M parameters at `hidden_size=640`, `graph_layers=6`, `attention_heads=8`.
- Entity tokens for hexes, vertices, edges, players, global state, legal actions, and event history.
- Sparse legal-action scoring only over valid candidates.
- Policy, value, and final-VP heads trained during BC.
- No Q-head loss during PPO warm start (`--q-loss-weight 0`).

The old 2M model and old flat/chunked 35M model remain useful as baselines only. Do not throw their checkpoints away, but do not make them the main training target.

## Why This Change

The old xdim graph model chunked a flat observation vector into arbitrary tokens. That is better than a plain MLP, but the tokens mix unrelated Catan concepts. The entity model gives attention clean objects:

- 19 hex tokens
- 54 vertex/intersection tokens
- 72 edge/road tokens
- 4 player tokens
- 1 global token
- up to 64 event-history tokens
- per-legal-action tokens plus the existing action context

This preserves the useful parts of the current teacher pipeline while giving the model board structure and history.

## Data Strategy

Do not stop current data generation. Keep accumulating raw corrected teacher shards from Modal/A100/GH200.

Use raw teacher shards as the source of truth:

- `obs`
- `legal_action_ids`
- `legal_action_context`
- `action_taken`
- `target_policy`
- `target_scores`
- `winner`
- `final_public_vps`
- `final_actual_vps`
- `game_seed`
- `decision_index`
- `teacher_name`
- `phase`

Then convert them by replaying `game_seed + decision_index` into entity-token shards with:

```bash
PYTHONPATH=.:src:tools python tools/convert_teacher_to_entity_tokens.py \
  --data <curated_teacher_root> \
  --out <entity_teacher_root>/part_000 \
  --track 2p_no_trade \
  --vps-to-win 10 \
  --graph-history-features \
  --partition-count 75 \
  --partition-index 0 \
  --format npz_zst \
  --shard-size 200000
```

For Modal conversion, launch one converter per partition with `--partition-index 0..74`, then train on the parent directory or merge manifests after every partition reports `mismatches: []`.

The converter must report:

- `mismatches: []`
- `converted_rows == loaded_rows` unless using an explicit `--max-rows`
- `schema: entity_tokens_v1`

If mismatches occur, stop and inspect. Do not train B200 on mismatched entity data.

## Production Curation

Before conversion, curate raw teacher data with exact dedupe and strict quality gates:

```bash
PYTHONPATH=.:src:tools python tools/curate_teacher_data.py \
  --data <modal_raw_root_1> \
  --data <modal_raw_root_2> \
  --data <a100_raw_root> \
  --data <gh200_raw_root> \
  --out runs/teacher/curated/teacher_hq_2p10_ab45_search_v1 \
  --production-35m-teacher \
  --dedupe-keys exact \
  --format npz_zst
```

Keep all raw data. Dedupe only the curated training copy.

## B200 Training Command

Run on the 2x B200 host with DDP:

```bash
mkdir -p runs/bc/entity_graph_35m_2p10_hq
torchrun --standalone --nproc_per_node=2 tools/train_bc.py \
  --arch entity_graph \
  --data runs/teacher/entity/modal_entity_2p_hq_state_replayfix2_p1000_35m_20260629 \
  --track 2p_no_trade \
  --vps-to-win 10 \
  --epochs 5 \
  --batch-size 16384 \
  --hidden-size 640 \
  --graph-layers 6 \
  --attention-heads 8 \
  --graph-dropout 0.05 \
  --amp bf16 \
  --optimizer adamw \
  --weight-decay 0.01 \
  --fused-optimizer \
  --soft-target-source prefer_scores \
  --soft-target-temperature 0.7 \
  --soft-target-weight 0.7 \
  --soft-target-min-legal-coverage 0.5 \
  --forced-action-weight 0.05 \
  --winner-sample-weight 1.0 \
  --loser-sample-weight 0.3 \
  --value-loss-weight 0.25 \
  --final-vp-loss-weight 0.05 \
  --q-loss-weight 0 \
  --validation-fraction 0.05 \
  --validation-max-samples 200000 \
  --require-35m-model \
  --skip-teacher-quality-gate \
  --checkpoint runs/bc/entity_graph_35m_2p10_hq/current.pt \
  --save-each-epoch \
  --report runs/bc/entity_graph_35m_2p10_hq/report.json \
  2>&1 | tee runs/bc/entity_graph_35m_2p10_hq/train.log
```

The old production quality gate expects top-level raw/curated teacher metadata and is not the source of truth for these entity partition shards. Use the frozen entity-corpus QA in `docs/TEACHER_DATA_FREEZE_2026-06-29.md` for this launch; keep `--require-35m-model` enabled.

Initial batch size is conservative. If B200 utilization is low and HBM is under pressure target, increase:

- `16384 -> 32768`
- then `32768 -> 49152`
- then `49152 -> 65536`

Do not raise batch size if validation accuracy collapses or if CPU loader stalls dominate.

## Monitoring

Watch B200 training:

```bash
tail -f runs/bc/entity_graph_35m_2p10_hq/train.log
```

Check GPU use:

```bash
nvidia-smi dmon -s pucm
```

Important run-log fields:

- `parameter_count` should be about `33,605,192`.
- `first_batch_profile` must include `hex_tokens_shape`, `vertex_tokens_shape`, `edge_tokens_shape`, `legal_action_tokens_shape`, and `event_tokens_shape`.
- `invalid_teacher_actions` must be `0`.
- `effective_soft_distillation_fraction` should stay high for AB/search/value rows.
- `forced_action_fraction` can be high, but `forced-action-weight` must be low.
- validation split must be by game seed.

## Scoreboard Gate

Do not judge by BC accuracy alone. Evaluate every epoch checkpoint against:

- random
- catanatron_value
- value_rollout_search
- AB3
- AB4
- AB5
- mixed teacher lineup

Minimum dev eval:

```bash
PYTHONPATH=.:src:tools python tools/evaluate_scoreboard.py \
  --candidate runs/bc/entity_graph_35m_2p10_hq/current_epoch0005.pt \
  --games 2000 \
  --tracks 2p_no_trade \
  --opponents random,catanatron_value,value_rollout_search,catanatron_ab3,catanatron_ab4,catanatron_ab5 \
  --out runs/scoreboards/entity_graph_35m_epoch5_dev.json
```

Promotion eval:

```bash
PYTHONPATH=.:src:tools python tools/evaluate_scoreboard.py \
  --candidate <best_epoch.pt> \
  --games 20000 \
  --tracks 2p_no_trade \
  --opponents catanatron_value,value_rollout_search,catanatron_ab3,catanatron_ab4,catanatron_ab5 \
  --out runs/scoreboards/entity_graph_35m_best_promotion.json
```

Target before PPO:

- beat random hard
- beat the old champion
- exceed 50% versus AB3
- approach AB4 if possible
- value head is not cold for PPO

BC does not need to beat AB5 alone. AB5/search should become the teacher/search ceiling for the MCTS/reanalysis stage.

## Data Movement

Do not route large datasets through the Mac.

Preferred:

1. Raw data lands on Modal volume / A100 / GH200.
2. Curate and convert as close to the data as possible.
3. Direct `rsync` from data host to B200 over SSH.
4. Train from local B200 NVMe.

Example:

```bash
rsync -a --info=progress2 \
  -e "ssh -i ~/.ssh/gpu_access_ed25519" \
  ubuntu@<source-host>:/path/to/runs/teacher/entity/teacher_hq_2p10_ab45_search_v1/ \
  /home/ubuntu/catan-zero/runs/teacher/entity/teacher_hq_2p10_ab45_search_v1/
```

## Efficiency Follow-Ups

These are required before running on the full tens-of-millions corpus. The local converter/trainer smoke path is correct, but it is intentionally not the final large-corpus input pipeline.

1. Add streaming or shard-partitioned conversion so `tools/convert_teacher_to_entity_tokens.py` does not hold all rows as Python dicts in RAM.
2. Add seed-hash partitioning for conversion, e.g. `--partition-index` / `--partition-count`, so Modal can convert entity shards across 75 containers.
3. Add a streaming/pinned-memory training loader so full datasets do not need eager RAM load on every DDP rank.
4. Precompute soft distillation targets into shards instead of rebuilding them every batch.
5. Add a direct Modal converter so the 600 CPU fleet converts raw shards into entity shards without copying raw data elsewhere.
6. Add scoreboard automation after every epoch.
7. Add DAgger/relabel pass once the entity BC checkpoint is competent.

## Stop Conditions

Stop and debug if any of these happen:

- converter mismatches are non-empty
- invalid teacher actions > 0
- entity tensor shapes missing in first batch profile
- validation accuracy rises but win rate stays below random/value baselines
- value loss stays huge or NaN
- B200 GPU0 works while GPU1 is idle during DDP
- training uses the wrong parameter count

## Verified Locally

Local smoke passed on 2026-06-29:

- `tools/convert_teacher_to_entity_tokens.py` converted 127 rows with `mismatches: []`.
- `tools/train_bc.py --arch entity_graph` trained one tiny epoch and saved a checkpoint.
- Production config parameter count: `33,605,192`.
- Focused tests: `39 passed`.
