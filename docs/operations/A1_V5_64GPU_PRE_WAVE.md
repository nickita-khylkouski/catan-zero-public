# A1 v5 64-GPU pre-wave contract

This is the current seal/render boundary for the 12,000-game A1 wave. It does
not authorize a launch.

## Producer identity

- Registry role/version: `generator_champion` v5.
- Checkpoint:
  `/home/ubuntu/catan-zero-production/runs/learner/a1-production-l1-one-dose-20260712-r3/candidate.pt`
- SHA-256:
  `6817ab054506f962a758ebf48addce5cc7eb801bf451cf2d02b62fb91f5da39c`
- Deployed generation identity: global n128/n_fast16, `p_full=0.25`,
  determinization P4, `c_scale=0.10`. The registry's `p_full=1` is an
  evaluation-only override and is not a generation recipe.
- Adaptive wide n256 is disabled. The final S3 r3 panel completed 400 games
  (202-198), remained at strict-SPRT `continue`, and cost +6.9% simulations /
  +8.8% search time. Its pooled evidence SHA-256 is
  `1ccc3484ced10df0b2f122e805d585895a6b8f5c0f9bf234359e1e1428622573`.
  Seal must replay the signed S3 adjudication selecting `n_full_wide=null`,
  `n_full_wide_threshold=null`, and `wide_roots_always_full=false`.

The v3 draft does not trust these prose values. Seal replays a committed
post-promotion handoff and refuses a different checkpoint or search identity.

## Exact 64-GPU quotas

`configs/gpu_fleet_64.json` is the topology authority. Manifest host order and
ascending physical GPU index define the worker order used by
`balanced_prefix_v1`.

| Lanes | current | recent history | hard negative |
|---|---:|---:|---:|
| first 8 | 150 | 29 | 10 |
| next 16 | 150 | 28 | 10 |
| final 40 | 150 | 28 | 9 |
| global | 9,600 | 1,800 | 600 |

Each of the 64 category jobs receives a fixed reserve of 5/2/1 attempted games,
so attempt totals are 9,920/1,928/664. Selection remains lowest-seed complete
per job and happens before row expansion.

Per host this yields:

- c1-c2: 600/116/40 selected games each;
- c3-c6: 600/112/40 selected games each;
- c7-c8: 600/112/36 selected games each;
- h100-8a through h100-8d: 1,200/224/72 selected games each.

## Opponent refresh requirements

- `recent_history` must be the exact incumbent displaced by the v5 promotion.
  Seal derives and verifies it from the committed promotion receipt; a
  hand-written "recent" path is refused.
- `hard_negative` must bind an `a1-hard-negative-selection-v1` record. That
  record authenticates the checkpoint bytes, version, selection reason, and
  immutable evaluation evidence used to choose it.
- Producer, history, and hard-negative checkpoint bytes must all be distinct.
- Rendered opponent manifests include the authenticated registry version,
  MD5 required by the runtime loader, and SHA-256 required by the v3 contract.

Historical v2/40-GPU drafts and locks remain replayable but cannot be used to
seal a new post-promotion wave.

## Learner handoff (no training authorized here)

The current v3 learner recipe is strict FP32, fresh Adam, one epoch, LR
`3e-5`, global batch 4096, zero forced-action policy mass, full forced-row
value mass, equal per-game policy weighting, scalar-MSE value weight 0.25, and
rank-offset PyTorch RNG. The exact production replay mix is 64% current, 12%
recent history, 4% hard negative, and 20% historical replay. Validation is a
deterministic 5% whole-game split within every component and acceptance uses
the matching component-weighted objective, not a raw concatenated-row mean.

`tools/a1_one_dose_train.py` retains the historical one-GPU B200 path and also
supports one eight-GPU B200 host at local batch 512 per rank. Both realize the
same 4096 global batch; ranks share one weighted global sampler stream and do
not shard the corpus. Eight-GPU execution fails closed without a same-host,
code-bound, hour-fresh NCCL canary from `tools/a1_ddp_epoch_canary.py`. The
canary proves eight distinct B200s, a CUDA all-reduce, rank-offset dropout RNG,
and exact reconstruction of the padded global sampler draw. It is diagnostic
only and cannot itself promote a checkpoint.
