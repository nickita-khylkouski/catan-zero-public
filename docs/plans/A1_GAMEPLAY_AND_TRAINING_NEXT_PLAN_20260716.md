# A1 gameplay and training next plan — 2026-07-16

## Decision

Do not authorize the large scratch run yet.

The next work must answer three root questions:

1. Can the policy reason from each legal target through the live board graph,
   rather than only gathering the target's local token?
2. Can value supervision shape a fresh shared representation, rather than
   being isolated from it at step zero?
3. Are the coherent-search policy targets stable enough to imitate, especially
   when completed-Q differences are microscopic?

Optimizer tuning comes after these are measured. A cleaner optimizer cannot
recover information the model never receives or make an unstable teacher
target correct.

## Evidence behind the decision

### 1. F7 lacks target binding; the scratch contract adds local gather but not topology

The authenticated f7 tournament checkpoint has no `action_target_gather`,
action cross-attention, edge-policy head, or relational trunk. The current
scratch contract is newer and **does** enable `action_target_gather`; it must
not be described or tested as a no-gather architecture.

The larger unresolved gap is topology. Vertex token construction discards the
topology object, edge tokens omit their endpoints, and the commissioned plain
Transformer consumes no adjacency. Target gather can retrieve the chosen
road's or settlement's local token, but it cannot follow that target through
neighboring roads, cutoffs, blocking opportunities, or Longest Road plans.

This is most damaging for:

- first and second settlement placement;
- initial road direction;
- expansion roads and blocking;
- city location;
- robber placement;
- the two linked Road Building choices.

### 2. The scratch contract now preserves the efficient width-640 attention path

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

Commit `ad7f303` moved the commissioned contract to width 640, so the inefficient
632-width result is now historical evidence rather than an open contract bug.
Keep width 640 fixed while testing topology so architecture and kernel-path
effects are not confounded.

### 3. Fresh-scratch value routing is selected in prose but not commissioned by evidence

The current scratch contract uses:

- `value_tower_split_layers=1`;
- `value_trunk_grad_scale=0.25`;
- random from-scratch initialization.

This routes a reduced value/final-VP gradient into the shared trunk. It is a
reasonable candidate, but no matched fresh-scratch result yet establishes that
0.25 is better than 0 or 1. The first scratch experiment must therefore compare
shared value-gradient scales with V25 as the exact commissioned control.

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

### 6. Fresh H100 gameplay shows opening myopia and a stable dev-card conversion failure

On eight f7 opening roots, the raw policy repeatedly preferred high-pip but
resource-duplicated settlements. Search often moved to a slightly lower-pip,
more balanced mix. This is consistent with f7 seeing total target production
without a clean action-local resource-composition binding.

On 40 fresh opening boards under the adopted coherent public-belief operator:

| Arm | Raw-to-search flip rate | Agreement with n128 |
|---|---:|---:|
| n128, `c_scale=.10` | 60.0% | 40/40 |
| n256, `c_scale=.10` | 60.0% | 38/40 |
| n128, noise-floor rescale | 57.5% | 38/40 |
| n128, variance-aware Q | 60.0% | 40/40 |
| n128, `c_scale=.03` | 52.5% | 35/40 |

Doubling visits did not materially change the opening choices. The cheap
search transforms also moved few roots, so none is yet a demonstrated fix.

A replayable f7 game exposed a sharper tactical failure. At turn 46, raw f7
assigned `END_TURN` 78.4% and Road Building 5.9%. Eight independent n128
searches selected `END_TURN` 8/8 even though Road Building's completed Q was
consistently higher by 0.00093--0.00109. In eight paired raw-policy
continuations, ending won 0/8 while forcing Road Building won 3/8 and raised
road length from four to six immediately. This is diagnostic, not a strength
gate, but it proves the decision museum must include dev-card timing and
multi-action road plans; search does not automatically repair them.

### 7. The current deep-oracle path cannot use the H100 efficiently

Four bounded opening-oracle jobs used only 11--15% GPU and hit a ten-minute
stop before completing 12 roots. Historical profiling explains the result:
ordinary native MCTS leaves remain serial, the evaluator queue was 64.09% idle,
and Python/JSON decision-input construction consumed 43.59% inclusive wall
time. Before scaling oracle panels, add neural-row accounting, deterministic
leaf-wave batching, and a fused native decision payload with exact parity
gates.

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

### P0-A: preserve efficient attention and add graph-aware target reasoning

Run matched fresh-init architecture arms:

1. `C640`: exact current width-640, gather, V25 contract.
2. `T640`: `C640` plus topology residual/relational consumption.
3. `E640`: edge-policy head only after `T640` identifies remaining policy
   aliasing.

Keep a no-gather permutation probe as a causal test fixture, not as the claimed
current baseline.

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
- the no-gather test control remains invariant, proving the probe is causal;
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

The first checked-in examples must include the observed f7 Road Building root,
its eight repeated-search seeds, and its paired continuation branches.

### P1: make fixed-root diagnostics H100-efficient and compute-accountable

Before expanding deep-oracle panels:

1. count neural rows, evaluator calls, unique leaf expansions, wall time, and
   GPU time in every search result;
2. batch one deterministic sequential-halving leaf wave at a time;
3. replace repeated JSON/Python decision preambles with a fused native payload;
4. require byte-exact entity/action tensors and exact selected-action, visit,
   completed-Q, logit, and value parity on frozen roots.

Stop an arm immediately on semantic drift. Throughput is secondary to parity.

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

1. Check in the replayable decision museum, beginning with the observed f7
   opening and Road Building failures plus paired counterfactuals.
2. Add a separate non-promotable bounded H100 scratch runner; do not weaken the
   B200 production authority or `go_authorized=false` gate.
3. Run matched `C640`/`T640` 128--256-step canaries on identical rows and
   initialization, including the action-target permutation probe.
4. Materialize the evaluator-query/turn-boundary value holdout and run
   V0/V25/V100 with matched optimizer exposure.
5. Add neural-row accounting, deterministic ordinary-leaf batching, and the
   fused native decision payload before scaling deep-oracle evidence.
6. Run phase-calibrated duplicate-search reliability, generate authenticated
   v5 data, and only then run the bounded scratch canary.
7. Consider PPO only after the supervised architecture, value route, and
   search teacher pass the frozen behavioral gates.
