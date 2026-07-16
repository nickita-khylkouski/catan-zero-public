# AI Workstream Guide

Use this guide when spawning other agents. Each workstream should produce a
small, reviewable artifact and must not silently change the champion.

## Current priority override — 2026-07-15

The immediate work is no longer generic distributed infrastructure or another
large data wave. Read
`docs/audits/A1_RL_SOFTWARE_DIAGNOSIS_20260715.md` and claim one of its WP1-WP6
repair packages. In priority order:

1. public rule state and development-card playability;
2. development-card action semantics and value affordances;
3. ordered public history;
4. value/trunk protection;
5. reliable coherent-search targets;
6. one canonical entity-graph PPO lane.

Do not launch a large learner or self-play wave until WP1-WP5 are integrated
into one function-preserving corrected parent.

## Global Rules

- Do not run local RL training.
- Remote training is allowed only through the fleet/controller scripts.
- Do not promote a checkpoint unless the configured gate says promotion.
- Do not scrape private data or bypass authorization.
- Do not change multiple subsystems in one PR unless the change is explicitly a
  cross-cutting architecture change.
- Every workstream must write a short result note under `docs/evidence/` or a
  machine-readable report under `runs/self_play/`.

## Current Champion And Evidence

Current champion:

```text
runs/self_play/champions/current_best_s9752_iter0002.pt
```

Current known issue:

```text
candidates often improve one opponent leg and regress on another
dominant failures: jsettlers_lite, value_rollout, heuristic
```

Do not claim a model is better without strict-gate evidence.

## Workstream A: Distributed Training Architecture

Objective:

Build the central learner/actor/inference architecture skeleton.

Deliverables:

- `src/catan_zero/distributed/` package skeleton
- typed trajectory schema
- local in-memory queue implementation for smoke tests
- learner loop stub that consumes batches but can run with fake data
- actor loop stub that emits trajectory records from existing env

Acceptance:

- Unit test creates fake actor data and one learner batch.
- No local long-running training.
- No champion modification.

## Workstream B: Batched Inference Server

Objective:

Create a service abstraction for batched policy inference.

Deliverables:

- inference request/response schema
- local process implementation
- batcher with max batch size and max wait time
- support for legal action candidate tensors
- benchmark script with fake observations

Acceptance:

- Can batch at least 100 requests into fewer model forwards.
- Reports decisions/sec and batch-size distribution.
- Does not change policy outputs versus direct inference on a small sample.

## Workstream C: Graph/History Model V1

Objective:

Replace flat observation dependence with a stronger graph/history representation.

Deliverables:

- `GraphHistoryPolicyV1` or improvement to `graph_history_candidate`
- board token encoder
- event-history encoder
- legal candidate scorer
- policy/value/Q outputs

Acceptance:

- Existing trainer can instantiate the architecture.
- Existing evaluator can load the checkpoint.
- Smoke forward pass handles all current legal action contexts.
- No hidden-state leakage.

## Workstream D: Belief Heads

Objective:

Train supervised hidden-state prediction from simulator truth.

Deliverables:

- belief target extractor
- resource-count target tensors
- dev-card target tensors where available
- belief loss in the learner
- calibration report

Acceptance:

- Belief prediction beats a simple public-count/prior baseline.
- Actor input remains public/legal.
- Teacher-only simulator truth never enters actor features.

## Workstream E: Reanalysis And Failure Buffer

Objective:

Turn rejected gates into training data.

Deliverables:

- failure buffer schema
- extractor from strict gate reports/replays
- reanalysis worker that labels failure states with teacher/search targets
- sampler that oversamples failure tags

Acceptance:

- Can ingest a rejected candidate report.
- Produces JSONL or parquet training records.
- Includes failure tag and opponent id.

## Workstream F: Promotion And Evaluation

Objective:

Make promotion harder and more reliable.

Deliverables:

- triage, strict, and champion gate profiles
- seat-balanced seed schedule
- confidence interval reporting
- anti-timeout accounting
- payoff-matrix summary

Acceptance:

- Can run dry-run plans.
- Can explain why each candidate was rejected or promoted.
- Promotion requires no opponent regression.

## Workstream G: Exploiter Agents

Objective:

Train agents that find champion weaknesses.

Deliverables:

- opening exploiter recipe
- JSettlers-style exploiter recipe
- value-rollout exploiter recipe
- robber/blocking exploiter recipe
- report showing exploit success against champion

Acceptance:

- Exploiters are separate from champion candidates.
- Any found weaknesses are added to failure buffer.

## Workstream H: Cluster Ops

Objective:

Keep expensive hardware busy without corrupting evidence.

Deliverables:

- cluster launch config
- machine role map
- actor/learner/evaluator process supervisor
- utilization dashboard/report
- cleanup script for stale jobs

Acceptance:

- Can show CPU actor utilization, GPU utilization, queue depth, learner samples/sec,
  inference batch sizes, and evaluator throughput.
- Does not launch duplicate jobs for same checkpoint/worker pair.

## Recommended Parallel Assignment

If spawning 8 agents:

```text
Agent 1: distributed schemas + replay queues
Agent 2: batched inference server
Agent 3: graph/history model
Agent 4: belief target extraction
Agent 5: failure buffer + reanalysis
Agent 6: promotion/eval profiles
Agent 7: exploiter recipes
Agent 8: cluster ops/utilization
```

If spawning 3 agents:

```text
Agent 1: distributed actor/learner/inference skeleton
Agent 2: graph/history + belief model path
Agent 3: evaluation/failure/reanalysis loop
```

## Integration Order

1. Schemas and buffers
2. Batched inference
3. Actor loop
4. Learner loop
5. Evaluator loop
6. Graph/history model
7. Belief heads
8. Failure reanalysis
9. Exploiters
10. Large cluster run

## Definition Of Useful Progress

A workstream is useful if it makes one of these more true:

- stronger candidate passes strict gate
- more reliable rejection of bad candidates
- more samples/sec into the learner
- better GPU utilization
- better failure-state reuse
- better hidden-state belief accuracy
- better search/distillation targets
- better evaluation confidence

Everything else is secondary.
