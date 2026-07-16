# A1 f7 H100 gameplay audit — 2026-07-16

## Scope and identity

- Track: two-player, no trading.
- The exclusive current GPU host is `ubuntu@192.222.54.137`. The earlier line
  naming `192.222.55.12` as the continuing host is superseded. All verification
  reported by the current repair pass ran on `192.222.54.137`; other GPU hosts
  were not used.
- Hardware: eight NVIDIA H100 80GB HBM3 GPUs; CUDA preflight passed.
- Public/tournament checkpoint: f7, SHA-256
  `f7e93dfb8cdb713d647b3e142c949d59083de9f719b6688b6faa6c918ce3eed4`.
- The recovered v5 checkpoint with SHA-256
  `6817ab054506f962a758ebf48addce5cc7eb801bf451cf2d02b62fb91f5da39c`
  is generator-only and is not treated as the deployed player.
- B200 evidence was read and transferred only; no B200 job was run.

## What f7 appears to understand

- It usually recognizes high-production opening intersections.
- Search can repair some raw END_TURN mistakes: in the observed game it found
  a four-brick-for-sheep trade with Q 0.2823 versus END_TURN at 0.2005, and
  later selected road `[10,29]` with Q 0.0179 versus END_TURN at -0.1192.
- Opening n128 choices were stable under n256 on 38 of 40 fresh boards.

## What f7 misses

### Opening resource composition

Across eight inspected opening boards, raw top-one choices repeatedly clustered
on high-pip but resource-duplicated nodes. Examples included wood/wood,
wheat/wheat, and brick/brick intersections. Search often moved to slightly
lower-pip but more balanced wheat/ore/sheep or wood/sheep/wheat mixes.

F7 has no action-target gather, action cross-attention, edge-policy head, or
relational trunk. Its action context exposes target total production, port,
and occupancy, but not a clean target-to-adjacent-resource binding. The
observed behavior is consistent with that information surface.

### Development-card timing and multi-action road plans

In raw game seed 610101, RED held Knight and Road Building from turn 40 through
a 10-7 loss. At decision 142 / turn 46:

- raw END_TURN prior: 78.4%;
- raw Road Building prior: 5.9%;
- n128 selected END_TURN in 8/8 independent search seeds;
- Road Building minus END_TURN completed-Q was consistently positive,
  +0.0009306 to +0.0010946;
- paired raw-policy continuations: END_TURN won 0/8 for RED, forced Road
  Building won 3/8 and immediately raised road length from four to six.

This is a small diagnostic counterfactual, not a promotion gate. It is enough
to prove that search does not automatically repair this tactical conversion
failure and that dev-card timing belongs in the frozen decision suite.

## Fresh opening search sweep

Forty fresh roots, seeds 650001 onward, used public observation plus the
adopted coherent public-belief search operator and lazy interior chance:

| Arm | Flip from raw | Agreement with n128 | Mean Q spread / floor |
|---|---:|---:|---:|
| n128, c-scale .10 | 60.0% | 40/40 | 1.978 |
| n256, c-scale .10 | 60.0% | 38/40 | 2.512 |
| n128, noise-floor c=1 | 57.5% | 38/40 | 1.975 |
| n128, variance-aware Q | 60.0% | 40/40 | 1.978 |
| n128, c-scale .03 | 52.5% | 35/40 | 1.966 |

Doubling visits and the cheap Q transforms did not materially change the
selected moves. None is authorized as a fix from this evidence.

## GPU utilization finding

Four 12-root, 512-simulation afterstate-oracle jobs were stopped at ten
minutes. Each used only 11--15% GPU. This matches the existing profile:
ordinary leaves remain serial, the evaluator queue was 64.09% idle, and
Python/JSON decision-input construction accounted for 43.59% inclusive wall
time. Scaling deep behavioral panels requires:

1. neural-row/evaluator-call/GPU-time accounting;
2. deterministic sequential-halving leaf-wave batching;
3. a fused native decision payload;
4. exact tensor, action, visit, Q, logit, and value parity gates.

## Current scratch contract correction

The current contract is width 640 with `action_target_gather=true` and
`value_trunk_grad_scale=0.25` after `ad7f303`. The largest remaining
representation gap is
topology: vertex construction discards topology, edge tokens omit endpoints,
and the plain Transformer consumes no adjacency. Local gather does not by
itself represent road connectivity, cutoffs, or Longest Road plans.

## Remote diagnostic artifacts

- `/home/ubuntu/f7-gameplay-observer/raw-selfplay-seed610101.json`
- `/home/ubuntu/f7-gameplay-observer/fixed-dev-roots-seed610101-n128.json`
- `/home/ubuntu/f7-gameplay-observer/repeated-dev-root-seed610101-d142-n128.json`
- `/home/ubuntu/f7-gameplay-observer/dev-root-counterfactual-raw-continuations-seed610101-d142.json`
- `/home/ubuntu/f7-gameplay-observer/search-vs-raw-seed610101-n32.json`
- `/home/ubuntu/codex-diag-20260716/{base_n128,base_n256,noise_floor,variance_q,low_scale}.json`

The repaired public-behavior/opening tests passed 118/118 on the same H100
host. One paired n32 search-versus-raw smoke test split 1-1 and is not treated
as statistical strength evidence.

## Bounded evaluator-query holdout

A bounded evaluator-query diagnostic using explicit non-overlapping seed ranges
was run against the authenticated f7 checkpoint. The immutable result is
`/home/ubuntu/value-holdout-data/f7_query_holdout_37e64f0.json`, SHA-256
`b8d785d1a0e4a43d5d634df95f18609553ed88a9931070d1b252c540e3194251`.

| Slice | RMSE | Pearson | Spearman | ECE |
|---|---:|---:|---:|---:|
| Global | .819 | .565 | .591 | .065 |
| Opening | .957 | .297 | — | — |
| END_TURN | .836 | .555 | — | — |
| Pre-roll | .752 | .674 | — | .156 |

Actor-handoff mean absolute antisymmetry error was .246 across 62 pairs from
16 games. The weak opening correlation, elevated pre-roll calibration error,
and actor-handoff asymmetry are concrete value-learning targets; they are not
evidence for promoting a checkpoint or changing the commissioned objective by
themselves.

The run used explicit seed ranges but no authenticated science contract.
Therefore cohort disjointness and evaluator-transform identity are diagnostic
claims rather than promotion-authenticated facts. The artifact is permanently
`diagnostic_only=true` and `promotion_eligible=false`.

## Repair closeout

The clean H100 full suite at `1f4228d` passed 5,888 tests with 26 skips and no
failures in 20m13s. A later suite pinned to `259a0a7` completed 5,987 passes and
26 skips in 20m22s with one stale scratch-recipe digest assertion. Current
`main` binds the updated recipe/file digests; its focused recipe and canonical
launcher panel passes 81/81 on `db118d1`.

Later focused H100 panels pass 68/68 for strict teacher-gap award evidence and
59/59 for function-preserving value-tower source topology. These results do not
authorize a training run. No long training job was run, and the topology canary
correctly remained blocked because no authenticated current-v3 composite and
reviewed lock exists.
