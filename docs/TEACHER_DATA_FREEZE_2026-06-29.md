# Teacher Data Freeze - 2026-06-29

This file freezes the current pre-training teacher data state. Do not delete or mutate the raw Modal runs listed here. Training should use the converted entity-token shards on the B200 after loader QA.

## Compute State At Freeze

- Modal `dev-dennis`: no active Catan task apps; production conversion/status apps stopped.
- A100 box `ubuntu@a100-legacy`: no active teacher/training process; 8x A100 idle.
- GH200 box `ubuntu@gh200`: no active teacher/training process; GH200 idle.
- B200 box `ubuntu@B200`: no active teacher/training process; 2x B200 idle.

## Raw Modal Runs - Preserve As Source Of Truth

Modal volume: `catan-zero-teacher-data`

| Raw run | Track | Observed games | Observed samples | Invalid teacher actions | Truncated fraction | Notes |
|---|---:|---:|---:|---:|---:|---|
| `modal_600cpu_ab45_tmux_50k_hq_20260628_202857` | 2p no-trade | 25,356 | 5,936,145 | 0 | 0.0 | AB4/AB5/value-rollout/search-heavy mixed seats |
| `modal_600cpu_elite2p_ab5_search_50k_20260628_204311` | 2p no-trade | 19,712 | 4,538,852 | 0 | 0.0 | AB5/value-rollout/search-heavy elite mix |
| `modal_600cpu_recovery2p_8gb_ab45_50k_20260628_205800` | 2p no-trade | 18,028 | 4,218,168 | 0 | 0.0 | Recovery 2p AB4/AB5/search mix |
| `modal_600cpu_4p_bank_trade_hq_50k_20260628_204311` | 4p bank/trade | 9,808 | 8,222,746 | 0 | 0.021404 | 4p trade-enabled mix; truncated games dropped in curation |

Raw total observed: 72,904 games and 22,915,911 samples.

## Curated Runs

Curated rows are source shards after dropping bad rows only. Forced actions are kept for value/diagnostics and should be down-weighted in training, not deleted.

| Curated run | Raw samples | Kept samples | Dropped duplicates | Dropped invalid | Dropped truncated | Value-only samples |
|---|---:|---:|---:|---:|---:|---:|
| `modal_curated_2p_ab45_hq_state_35m_20260629` | 6,055,454 | 6,036,481 | 18,973 | 0 | 0 | 3,594,641 |
| `modal_curated_2p_elite_ab5_search_hq_state_35m_20260629` | 4,670,660 | 4,659,592 | 11,068 | 0 | 0 | 2,773,367 |
| `modal_curated_2p_recovery_ab45_hq_state_35m_20260629` | 4,284,642 | 4,268,550 | 16,092 | 0 | 0 | 2,544,276 |
| `modal_curated_4p_bank_trade_hq_state_35m_20260629` | 8,316,821 | 8,116,250 | 21,371 | 0 | 179,200 | 4,010,880 |

The earlier exact-dedupe 4p entity conversion `modal_entity_4p_bank_trade_hq_p1000_35m_20260629` is not training-safe because it exposed duplicate action conflicts. Use the state-curated 4p run instead.

## Converted Entity-Token Shards On B200

These are the training-ready converted shards. They were copied directly from Modal to the B200, not routed through the Mac.

| Dataset | B200 path | Rows | Seeds | Parts | Replay mismatch parts | Duplicate decision rows | Size | Files |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2p entity | `/home/ubuntu/catan-zero/runs/teacher/entity/modal_entity_2p_hq_state_replayfix2_p1000_35m_20260629` | 14,933,075 | 64,127 | 1000 | 0 | 19,183 | 3.7G | 3000 |
| 4p entity | `/home/ubuntu/catan-zero/runs/teacher/entity/modal_entity_4p_bank_trade_hq_state_p1000_35m_20260629` | 8,080,346 | 9,766 | 1000 | 0 | 26,092 | 2.5G | 3000 |

Converted total: 23,013,421 entity-token rows across 73,893 converted seeds.

Each B200 entity dataset has a local `SHA256SUMS.txt` checksum file covering its 1000 `entity_teacher_shard_*.npz.zst` files and 1000 `manifest.json` files.

## Verified Shard Shapes

Sample decompression/readback on B200 succeeded for both datasets.

2p sample shard:

```text
obs: [13747, 806]
legal_action_ids: [13747, 54]
legal_action_context: [13747, 54, 18]
target_policy: [13747, 54]
target_scores: [13747, 54]
teacher_name/winner/phase/player/action fields present
```

4p sample shard:

```text
obs: [8487, 1194]
legal_action_ids: [8487, 265]
legal_action_context: [8487, 265, 18]
target_policy: [8487, 265]
target_scores: [8487, 265]
teacher_name/winner/phase/player/action fields present
```

## Full Converted-Corpus QA

Full CPU QA was run over all 2000 compressed entity shards on the B200.

| Dataset | Rows scanned | Files scanned | Action not legal | Zero legal rows | Bad policy sums | Nonfinite obs/context/policy | Active nonfinite scores | True legal-candidate range | Forced fraction | Soft policy fraction | Soft score fraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2p entity | 14,933,075 | 1000 | 0 | 0 | 0 | 0 | 0 | 1-54 | 0.571680 | 0.935293 | 1.000000 |
| 4p entity | 8,080,346 | 1000 | 0 | 0 | 0 | 0 | 0 | 1-270 | 0.484201 | 0.851543 | 0.729706 |

Inactive `target_scores` padding slots contain nonfinite sentinel values; active score-mask slots are finite. Training loaders must mask before consuming `target_scores`.

## Next Gate Before Training

Before launching full BC on B200:

1. Run loader QA over a random shard sample from both entity datasets.
2. Confirm forced-action weighting is low in the training config.
3. Confirm value-head pretraining uses `winner` / final outcome fields.
4. Run a one-batch 35M entity-model smoke test on B200.
5. Only then launch the full 2xB200 BC job.
