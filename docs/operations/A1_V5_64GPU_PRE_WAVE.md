# A1 v5 64-GPU pre-wave contract

This is the current seal/render boundary for the 12,000-game A1 wave. It does
not authorize a launch.

## Producer identity

- Registry role/version: `generator_champion` v5.
- Checkpoint:
  `/home/ubuntu/catan-zero-production/runs/learner/a1-production-l1-one-dose-20260712-r3/candidate.pt`
- SHA-256:
  `6817ab054506f962a758ebf48addce5cc7eb801bf451cf2d02b62fb91f5da39c`
- Deployed search identity: global n128, `p_full=0.4`, `c_scale=0.10`.

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
