# A1 gameplay and training next plan — 2026-07-16

## Decision

Do not authorize the large scratch run yet.

The next work must answer three root questions:

1. Can the policy bind each legal settlement, road, city, or robber action to
   the exact live board entity it affects?
2. Can value supervision shape a fresh shared representation, rather than
   being isolated from it at step zero?
3. Are the coherent-search policy targets stable enough to imitate, especially
   when completed-Q differences are microscopic?

Optimizer tuning comes after these are measured. A cleaner optimizer cannot
recover information the model never receives or make an unstable teacher
target correct.

## Evidence behind the decision

### 1. The planned policy lacks a direct action-to-board join

The current scratch construction enables structured static-action features but
does not enable `action_target_gather`, action cross-attention, or
`edge_policy_head`. `legal_action_target_ids` therefore do not affect policy
logits.

Static action features identify the catalog action and provide useful generic
semantics. They cannot directly join a candidate road or settlement to the
current ownership, production, blocking, and connectivity state stored in its
live entity token.

This is most damaging for:

- first and second settlement placement;
- initial road direction;
- expansion roads and blocking;
- city location;
- robber placement;
- the two linked Road Building choices.

### 2. Width 632 is an H100 performance cliff

An exact synthetic training-step comparison used the current six-layer,
one-private-value-block construction, BF16, batch 256, and the real audit-shard
tensor shapes on an H100 80GB:

| Width | Heads | Head dimension | Parameters | Step ms | Rows/s | Peak GiB |
|---|---:|---:|---:|---:|---:|---:|
| 632 | 8 | 79 | 39,840,089 | 129.66 | 1,974 | 18.00 |
| 640 | 8 | 80 | 40,851,273 | 88.14 | 2,904 | 11.23 |

Shrinking by 1.25% made the model about 47% slower per step and used about 60%
more peak memory. The 79-dimensional heads fall off the efficient attention
path.

The parameter ceiling must not force an inefficient hidden dimension. Prefer
width 640 and either:

- raise the checkpoint-class ceiling enough to cover the useful architecture;
  or
- remove/compress a dormant readout such as the Q head, which is frozen and
  unused when `q_loss_weight=0`.

Do not retain width 632 for production training.

### 3. Fresh-scratch value learning is isolated too early

The planned model uses:

- `value_tower_split_layers=1`;
- `value_trunk_grad_scale=0.0`;
- random from-scratch initialization.

This stops value/final-VP gradients at the input to the private value block.
Value supervision cannot shape the first five Transformer blocks, history
representation, action encoder, or shared public feature paths. That is a
sensible mature-policy protection mechanism, but it is uncommissioned for a
random representation.

The first scratch experiment must compare shared value-gradient scales. Zero
must not be the assumed baseline.

### 4. Historical policy targets can be extremely sharp on tiny Q spreads

On the 331 policy-active rows in the H100 audit shard:

- mean target top-one probability: `0.704`;
- rows with target top-one above `0.9`: `143`;
- prior-to-target argmax flip rate: `54.1%`;
- median completed-Q range: `0.0404`;
- rows with Q range below `0.01` and target top-one above `0.9`: `50`;
- rows with Q range below `0.005` and target top-one above `0.9`: `36`.

The phase breakdown is more concerning:

| Phase | Rows | Mean target top-one | Median Q range | Prior flip rate | Q range < .01 and top-one > .9 |
|---|---:|---:|---:|---:|---:|
| Initial road | 20 | .902 | .00024 | .700 | 9 |
| Initial settlement | 20 | .573 | .77934 | .500 | 0 |
| Discard | 68 | .927 | .01547 | .500 | 23 |
| Move robber | 101 | .458 | .08226 | .683 | 1 |
| Play turn | 122 | .771 | .03834 | .426 | 17 |

The new duplicate-search reliability audit is the right evidence mechanism,
but its five-percent slice is diagnostic and current production correctly does
not yet use it as a learner weight. Before training, reliability must be
calibrated by phase against a stronger reference.

### 5. The historical networks have recognizable but uneven Catan knowledge

The H100 behavioral probe compared three historical checkpoints on 2,610
shared states:

| Metric | F7 history base | Prod policy-1024 | Trust-v25 step-24 |
|---|---:|---:|---:|
| JSettlers agreement | 79.0% | 78.4% | 81.0% |
| First-settlement best-production choice | 34.4% | 100.0% | 35.9% |
| First-settlement teacher agreement | 21.9% | 81.3% | 18.8% |
| Value/public-VP-lead correlation | -.004 | .307 | .257 |
| Robber agreement | 61.8% | 64.6% | 78.8% |
| Build selected when a build is legal | 57.8% | 52.2% | 47.0% |
| Buy development card when legal | 33.0% | 42.2% | 74.3% |

The trust treatment improved robber play and small-sample match strength but
regressed opening placement and heavily shifted toward development cards.
These are exactly the tradeoffs a phase/scenario position book must expose.

## Work already completed

- Raw and restart trajectories generated by the hidden-information heuristic
  now fail closed to the authoritative information regime instead of claiming
  public coherence.
- The entity-checkpoint opening evaluator now loads entity checkpoints, calls
  their action-probability API correctly, and defaults to two-player openings.
- The opening panel now supports the exact adopted operator:
  public observation, determinization off, coherent public-belief search on.
- Both repaired evaluators ran end to end on the H100.

## Experiment sequence

### P0-A: preserve efficient attention and add action-to-target binding

Run matched architecture arms at width 640:

1. `B0`: current policy path, no target binding.
2. `G1`: `action_target_gather`.
3. `E1`: `edge_policy_head`.
4. `GE1`: gather plus edge head, only if each individual arm is positive.

If the 40M ceiling is immovable, first test omitting/compressing the dormant Q
head rather than shrinking the Transformer width.

Use the same initialization seed and identical rows. Start with:

- one-step gradient admission;
- 128–256 optimizer-step overfit/holdout canaries;
- no long training.

Required measurements:

- exact parameter count and H100 step throughput;
- gradient norm in each new target-binding module;
- policy CE/KL by phase and action type;
- settlement/road/robber target-sensitivity;
- symmetry consistency;
- unchanged forced-action legality.

Hard success criteria:

- permuting `legal_action_target_ids` changes logits in the commissioned arm;
- the baseline remains invariant, proving the probe is causal;
- settlement/road holdout loss improves without degrading robber/discard;
- throughput remains on the width-640 efficient path.

### P0-B: commission value-gradient routing for scratch

Using the best P0-A architecture, run identical-seed arms:

1. `V0`: `value_trunk_grad_scale=0.0`;
2. `V25`: `value_trunk_grad_scale=0.25`;
3. `V100`: `value_trunk_grad_scale=1.0`.

Keep the private final value block in all arms so only shared-gradient routing
changes.

Required measurements:

- value RMSE, bias, Pearson/Spearman correlation, and ECE;
- the same metrics at opening, turn-boundary, robber, discard, and normal-play
  states;
- policy CE/KL and policy/value gradient cosine;
- gradient norm by shared state trunk, action representation, history path,
  and private value tower;
- H100 throughput and clipping contribution.

Selection rule:

- reject `V0` if it learns value more slowly or plateaus lower;
- select the smallest nonzero scale that preserves policy learning while
  materially improving game-disjoint value calibration;
- only consider an annealed scale after a fixed-scale arm proves the need.

### P0-C: calibrate teacher reliability before weighting it

Run duplicate coherent n128 search on a large diagnostic panel, stratified by:

- phase;
- legal width;
- action type;
- initial road;
- discard;
- dev-card decisions;
- robber decisions.

For unstable or tiny-margin roots, run a stronger reference:

- more simulations and multiple independent seeds;
- paired counterfactual rollouts where feasible.

Measure:

- duplicate-policy JS divergence;
- policy and completed-Q top-one agreement;
- completed-Q top margin;
- agreement with the stronger reference;
- target entropy and prior-to-target flip.

Then compare learner-target treatments:

1. unchanged target;
2. duplicate-reliability confidence weighting;
3. a calibrated phase-aware treatment for tiny-margin roots.

Do not invent a global Q-margin threshold. Q scale is phase dependent.

### P1: build the human Catan position book

Extend `tools/fixed_root_search_stability.py`; do not create another isolated
evaluator.

The first book should cover:

1. first settlement: pips, diversity, and useful port;
2. second settlement: complementarity and initial resource grant;
3. initial road: live expansion versus dead coast or blocking;
4. expansion settlement and missing-resource access;
5. port settlement;
6. normal road, blocking, and Longest Road race;
7. city choice;
8. buy development card versus build or end turn;
9. maritime trade completing a build;
10. robber tile and victim;
11. discard preserving a near-term build;
12. Knight and Largest Army timing;
13. Year of Plenty pair;
14. Monopoly resource and timing;
15. both Road Building actions as one plan;
16. win-now action versus end turn.

Every record must include:

- human-readable action descriptions;
- raw prior;
- repeated-search target;
- completed-Q values and margin;
- selected action;
- public board/resources/rule-state summary;
- paired counterfactual outcome estimate;
- exact checkpoint, operator, seed, and state hash.

### P1: generate an authenticated current-v5 corpus

The historical audit data is useful for diagnosing old networks, but it cannot
certify the v5/history-v2/rule-state path. No existing B200 corpus was found
that authenticates the complete current input contract.

Before a current scratch canary:

- gather fresh v5 learner rows;
- verify meaningful history v2 and event target IDs;
- verify actor public rule state;
- preserve search evidence and duplicate-search reliability fields;
- keep the B200 read-only for this audit workflow;
- run bounded experiments on the H100 only.

### P2: bounded scratch canary

Only after P0-A/B/C and fresh v5 data:

- use width 640;
- use the selected target-binding path;
- use the selected nonzero value-gradient route;
- run the commissioned LR/warmup schedule;
- stop at a small fixed optimizer horizon;
- evaluate on the frozen position book and game-disjoint value-query holdout.

Do not promote based on aggregate loss alone.

## Promotion gates for the first real retrain

The large retrain may start only when all are true:

- H100-efficient width and parameter budget are explicitly resolved;
- action-to-target binding passes causal sensitivity and holdout tests;
- value routing is selected by a matched scratch experiment;
- teacher reliability is calibrated by phase;
- current-v5 corpus identity and feature-signal admission pass;
- opening, robber, discard, dev-card, build/conserve, and value-query panels
  have frozen baselines;
- scratch optimizer schedule is marked authorized by evidence, not default.

## Immediate next implementation order

1. Add a width-640 target-binding experiment configuration and parameter-budget
   alternative.
2. Add the action-target permutation/sensitivity probe.
3. Add value-routing matched canaries.
4. Extend the fixed-root position book and run the 16 scenario slices.
5. Run the duplicate-search reliability calibration.
6. Generate authenticated v5 data.
7. Run the bounded scratch canary.
