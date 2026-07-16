# A1 representation and value recovery plan

**Date:** 2026-07-15  
**Authority:** implementation plan for
`docs/audits/A1_RL_SOFTWARE_DIAGNOSIS_20260715.md`  
**Execution state:** code repair only; no large learner, generation wave, or
champion mutation is authorized by this plan.

## Outcome

Build one function-preserving corrected v5 parent whose neural input
distinguishes every rules-critical public/actor-known state used by the Rust
engine, whose value path is not destroyed by policy distillation, and whose
policy targets come from an exact-parent coherent teacher with explicit
reliability.

The first accepted experiment must answer:

> Can one independently initialized, parent-KL-bounded dose improve the exact
> corrected parent under the same search operator without degrading phase-wise
> value calibration?

## Dependency graph

```text
W0 evidence snapshot
  |
  +--> W1 public rule state -------+
  +--> W2 structured actions ------+--> corrected no-op parent
  +--> W3 ordered public history --+
                                      |
                                      +--> W4 value/trunk repair
                                      +--> W5 teacher reliability
                                      +--> W6 dose/trust region
                                               |
                                               v
                                      parent-matched learner arms
                                               |
                                               v
                                      W7 short PPO comparison
                                               |
                                               v
                                      W8 same-operator evaluation
```

## Workstream contracts

| ID | Deliverable | Primary files | Depends on | Completion evidence |
|---|---|---|---|---|
| W0 | Sanitized checkpoint/config/corpus/report identity receipts | `docs/evidence/` | none | hashes and identities committed |
| W1 | `public_rule_state_v1` residual | Rust snapshot, neural adapter, entity features | W0 | rules-distinct fixtures produce distinct features; old parent remains a no-op |
| W2 | Monopoly/YOP and board-target action semantics; value affordance summary | action translator, token builder, model | W0 | structured resource/target identity reaches policy and value |
| W3 | ordered `meaningful_public_history_v2` | native action records, history translator/encoder | W0 | turn/order/target/payload survive native-to-model path |
| W4 | protected value learning | model and `train_bc` | W1-W3 | phase calibration, gradient accounting, split/low-trunk arms |
| W5 | reliable exact-parent teacher | MCTS target transform, sampler, contracts | W0 | reliability-qualified policy targets and operator identity |
| W6 | functional dose/trust region | one-dose learner | W4-W5 | parent KL, trunk drift, objective dose ledger |
| W7 | canonical entity-graph PPO lane | PPO factory, actor, learner | W1-W6 | exact parent/anchor, gamma 1, legal-mask and rollout-version contract |
| W8 | checkpoint and full-agent evaluation | H2H/evaluation tools | W4-W7 | same-operator candidate/parent plus deployed-agent reports |

## Parallel ownership

Agents may work concurrently on W1, W2, W3, and W5. W4 owns late
policy/value architecture and must not also rewrite the state schema. W6 owns
learner dose and may not change search targets. W7 begins only after the
corrected parent schema is stable.

Every agent must record:

- finding/workstream ID;
- exact files owned;
- old and new schema/config versions;
- train-time and serve-time consumers;
- migration behavior for historical rows/checkpoints;
- evidence location;
- status transition in
  `docs/evidence/A1_OBSERVATION_VALUE_DIAGNOSIS_20260715.json`.

## W0: evidence snapshot

Commit sanitized receipts for:

- current v5 checkpoint SHA and architecture config;
- function-preserving upgraded parent SHA/config;
- current coherent corpus metadata and event counts;
- Stage-C value/teacher fingerprints;
- Q-spread/noise-floor report;
- exact parent/producer/evaluation identities.

Remote absolute paths alone are not durable evidence.

## W1: public rule state

Add a versioned residual containing:

- actor already played a development card this turn;
- per-type actor new/unplayable and old/playable dev counts;
- Road Building active;
- free roads remaining;
- current discard remainder;
- exact unsaturated public played-development counts.

Preserve actor-private/opponent-public masking. The model may consume everything
a perfect public card counter can infer, but not authoritative opponent hidden
truth.

## W2: action semantics and value affordances

Bind semantic fields directly:

- Monopoly selected resource;
- Year-of-Plenty resource pair;
- build/road/robber board target;
- resource bundle and trade direction;
- action type and actor.

Provide a masked aggregate of current legal affordances to the value tower.
The value function must not be blind to the set of futures available from the
state.

## W3: ordered public history

Define history v2 rather than reinterpreting v1:

- public turn index;
- action ordinal;
- actor;
- public board target;
- public resource payload;
- dev purchase/play timing;
- redacted steal/discard semantics.

Use an ordered causal encoder or recurrent state. Do not mean-pool away order.

## W4: value repair

The control restores `ROLL` and `END_TURN` value weights to `1.0`. Policy weight
remains zero on single-action prompts.

Compare independent arms:

- shared trunk LR `1.0`;
- shared trunk LR `0.25`;
- shared trunk LR `0.10`;
- split final policy/value tower.

Keep terminal outcomes as the main value target. Record phase-sliced value
calibration, value/policy gradient norms, gradient cosine by layer, parent
policy KL, and functional drift at least every eight steps.

Match the scalar function used during training and search or explicitly
calibrate the transform.

## W5: teacher reliability

Bind every policy label to:

- producer and parent checkpoint SHA;
- search implementation SHA;
- coherent-belief/operator version;
- budget and candidate cap;
- chance/symmetry contract;
- Q transform and D1/noise-floor contract;
- target-temperature semantics.

On an audit fraction, repeat search with independent search/chance seeds.
Estimate target disagreement and use:

```text
policy weight = bounded surprise * reliability
```

not surprise alone. Unreliable labels become lower-weight, value-only, or
reanalysis candidates. Do not grant n256 policy authority solely because it
used more simulations.

## W6: dose and trust region

Every arm starts from the exact corrected parent and fresh Adam. No candidate
chaining.

Track separately:

- policy-active exposure;
- value exposure;
- objective gradient norm into the shared trunk;
- parent policy KL on a fixed anchor corpus;
- layerwise parameter drift;
- top-one policy flips;
- value-output drift.

Select comparable checkpoints by functional distance, not the same nominal
optimizer step.

## W7: PPO diagnostic/finisher

Retire or hard-fail launchers that silently construct a legacy architecture.
The canonical lane must load `entity_graph` explicitly.

Initial contract:

- exact corrected parent and separate frozen anchor;
- terminal reward, gamma `1.0`;
- GAE lambda `0.95-0.98`;
- clip `0.1`;
- two to four epochs;
- parent target KL `0.005-0.01`;
- seat-balanced frozen opponent league per rollout batch;
- legal mask included in behavior and learner log probabilities;
- stale rollout/version rejection or explicit V-trace bounds;
- protected trunk/split towers inherited from W4.

PPO is compared after corrected distillation; it is not the new default merely
because it runs.

## W8: evaluation

Decompose:

1. checkpoint improvement: candidate and parent under the same operator;
2. full-agent improvement: each sealed deployed configuration;
3. raw-policy change;
4. search uplift over each raw policy;
5. phase/value calibration;
6. population and external-opponent robustness.

Promotion policy is unchanged.

## Stop conditions

Stop implementation and resolve the contract if:

- Python and Rust disagree on new fields;
- hidden opponent truth reaches actor features;
- old checkpoint loading is not function preserving;
- a row cannot prove its operator/parent identity;
- a treatment changes multiple workstreams without an independent control;
- a launcher can silently select the wrong architecture.

## Definition of done before the next large wave

- W1-W6 merged into one canonical collaboration branch.
- A corrected parent artifact loads through Python and Rust with the declared
  migration.
- Fresh rows contain complete rule-state/history/action semantics.
- A parent-matched learner produces at least one candidate with stable value
  calibration.
- Same-operator evaluation demonstrates whether it improved.
- The production science contract is updated only after that evidence.

