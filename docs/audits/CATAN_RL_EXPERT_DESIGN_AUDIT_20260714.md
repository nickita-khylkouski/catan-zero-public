# Catan RL expert design audit: forced prompts, search, belief, and learner semantics

**Date:** 2026-07-14

**Scope:** Current A1 `2p_no_trade`, 10-VP self-play and learner pipeline

**Status:** Evidence-backed design audit. Measurements below come from the authenticated V5 composite and its fresh current-producer component. This document separates implemented safeguards from deliberate approximations and unresolved hypotheses.

## Executive summary

The commonly repeated statement that "50% of Catan moves are forced" is wrong as a description of the game. The measured statistic is:

> Approximately half of serialized **engine prompts** have exactly one legal catalog action.

Nearly all such prompts are mandatory `ROLL` or sole-remaining `END_TURN` lifecycle actions. Forced prompts receive zero policy loss, so the policy is not being trained to imitate trivial actions. They do, however, receive full terminal-outcome value loss. That makes the forced-prompt statistic relevant to shared-trunk and value learning even though it is not a policy-label problem.

The audit identified three higher-leverage design risks:

1. **Belief quality:** four public-conservation determinizations are not a history-conditioned posterior, while the current model's event-history payload is empty.
2. **Search meaning:** n128 is a total budget split into four 32-simulation worlds, and information-set traversal stops when the root actor's turn ends.
3. **Learner information use:** only about 12.4% of rows carry policy targets, while expensive root-Q, afterstate, uncertainty, VP, categorical-value, and auxiliary information is mostly not consumed by the current learner.

The current pipeline is a meaningful and internally coherent **two-player, no-domestic-trade** research track. It must not be presented as a solved model of standard four-player negotiated Catan.

## 1. What `is_forced` actually means

The producer defines a forced row mechanically:

```python
legal_rust = tuple(sorted(result.improved_policy.keys()))
is_forced = len(legal_rust) <= 1
```

This is an engine-prompt property. It is not a human judgment about whether a move is strategically forced, obvious, dominated, or effectively forced.

The production engine represents the game as a sequence of atomic prompts. A normal human turn commonly becomes:

```text
pre-roll PLAY_TURN
  -> ROLL, plus any legal pre-roll development-card plays

dice outcome
  -> possibly repeated one-resource-at-a-time DISCARD_RESOURCE prompts
  -> MOVE_ROBBER destination + victim

post-roll PLAY_TURN
  -> build / buy / play development card / maritime trade / END_TURN
  -> eventually only END_TURN may remain
```

Consequences:

- `ROLL` is forced when no pre-roll development card is playable.
- A roll-selected row is nonforced when the player chose `ROLL` while a development-card play was also legal.
- `END_TURN` is forced only when it is the sole remaining catalog action.
- Discarding is represented one resource card at a time. A discard row is forced only when one resource type is currently available to discard.
- Initial settlements, initial roads, and robber placements were never classified as forced in the measured fresh component.

The metric should be named `single_legal_action_prompt` in reports and dashboards. Retaining `is_forced` as a storage field for compatibility is reasonable, but user-facing language should not call these all "forced moves."

## 2. Exact measured breakdown

### 2.1 Authenticated V5 composite

- Total rows: `16,809,562`
- Raw forced rows: `8,380,837` (`49.8576%`)
- Expected forced fraction under the actual source -> game -> row sampler: `50.2680%`

The sampler expectation differs from the raw row fraction because the learner samples sources and games before sampling rows. Long games are therefore not selected merely because they contain more rows.

### 2.2 Fresh current-producer component

- Total rows: `12,301,676`
- Forced rows: `6,194,547` (`50.3553%`)
- Stored `is_forced` agrees with ragged `legal_count == 1` on every row.

| Single-action prompt | Rows | Fraction of all rows | Fraction of forced rows |
|---|---:|---:|---:|
| `ROLL` | 3,629,125 | 29.5011% | 58.5858% |
| `END_TURN` | 2,497,327 | 20.3007% | 40.3149% |
| `DISCARD_RESOURCE` | 68,095 | 0.5535% | 1.0993% |
| **Total** | **6,194,547** | **50.3553%** | **100%** |

Thus 98.90% of the forced bucket is mandatory roll/end-turn lifecycle handling.

Additional prompt semantics:

- `ROLL` represents 32.205% of all rows and is forced 91.603% of the time.
- `END_TURN` represents 31.837% of all rows and is forced 63.765% of the time.
- `DISCARD` rows are forced 7.971% of the time.
- `BUILD_INITIAL_SETTLEMENT`, `BUILD_INITIAL_ROAD`, and `MOVE_ROBBER` phases are 0% forced in this component.

### 2.3 Legal-action width

| Legal actions | Fraction of fresh rows |
|---:|---:|
| 1 | 50.355% |
| 2 | 9.966% |
| 3 | 5.637% |
| 4 | 1.223% |
| 5 | 8.239% |
| 6-10 | 7.968% |
| 11-20 | 14.504% |
| 21-40 | 0.443% |
| 41+ | 1.665% |

Among nonforced prompts, mean legal width is 9.47 and median legal width is 5.

### 2.4 Per-game interpretation

The average fresh game contains:

- 240.27 stored prompts
- 120.99 forced lifecycle prompts
- 89.47 nonforced n16 trajectory prompts
- 29.81 nonforced n128 policy-target prompts

This explains how a superficially implausible 50% number arises: most human turns contribute an explicit roll prompt and often a sole end-turn prompt in addition to strategically meaningful decisions.

## 3. How the learner uses each row class

The current producer/learner contract is:

| Row class | Search behavior | Policy weight | Value weight |
|---|---|---:|---:|
| Forced single-action prompt | Special forced path; no Sequential Halving simulations | 0 | 1 |
| Nonforced fast prompt | n16; one determinization | 0 | 1 |
| Nonforced full prompt | n128 total; four determinizations | 1 | 1 |

Fresh measured proportions:

- Forced: `50.355%`
- Nonforced n16: `37.238%`
- Nonforced n128 policy-active: `12.407%`
- Full-search rate among nonforced prompts: `24.985%`, matching configured `p_full=0.25`
- Forced rows with nonzero policy weight: exactly zero

### 3.1 Implemented safeguard: policy loss is not diluted

Policy cross-entropy is normalized by positive policy weight. Forced and n16 rows enter neither its numerator nor denominator. A global batch containing roughly 12.4% policy-active rows therefore has fewer policy examples, but the policy loss magnitude is not silently divided by the full batch size.

The correct concern is not "the model learns to roll." It is:

- fewer independent policy targets per learner draw;
- shared-forward compute spent on value-only rows;
- shared-trunk gradients dominated by terminal-value supervision on repetitive lifecycle states.

### 3.2 Open issue: forced rows receive full terminal-value weight

Current settings include:

```text
forced_action_weight = 0.0
forced_row_value_weight = 1.0
value_target_lambda = 1.0
```

Every forced row receives the realized terminal outcome `z` as its value label. Approximately half of value/trunk updates therefore come from roll/end-turn states, with many correlated labels from the same trajectory.

This is not automatically wrong. The evaluator is queried at pre-roll, post-action, and other lifecycle states, so deleting all such coverage could create phase-specific value blind spots. The unresolved question is the correct phase-specific weighting.

Recommended causal comparison:

1. Current control: all forced value weight 1.0.
2. Downweight all forced rows to 0.25.
3. Phase-specific arm: retain meaningful discard/pre-roll coverage, heavily downweight sole `END_TURN`, and test a smaller `ROLL` weight.

Judge the arms by rollout calibration and paired playing strength, not only validation MSE.

### 3.3 Expensive search evidence is generated but unused

The forced-roll path enumerates all 11 dice sums and evaluates the resulting states. Forced non-roll prompts call the evaluator once. However, `value_target_lambda=1.0` means stored root search values are not used by the current value objective.

Current primary training also leaves these signals disabled or zero-weighted:

- root search-value blend;
- action-Q loss;
- afterstate loss;
- categorical/HL-Gauss value;
- final-VP prediction;
- value uncertainty;
- auxiliary subgoal heads.

Forced handling accounts for approximately 42.5 million neural state evaluations in the measured fresh component whose root-search labels the current learner ignores.

Two distinct changes should be tested rather than conflated:

- Retain forced rows but synthesize the trivial action result without expensive evaluator/chance work when no configured loss consumes that evidence.
- Separately test whether consuming calibrated forced-roll/afterstate values improves the learner.

### 3.4 Minor telemetry defect

The inner forced-search path reports `used_full_search=True`, while the information-set aggregation wrapper records forced rows with `used_full_search=False`. Explicit policy masking keeps training correct, but comments and telemetry describing forced rows as always full-search are stale.

## 4. Current production scope is deliberately narrow

The A1 track is:

```text
2 players
10 VP
no domestic player-to-player trade
maritime trade enabled
```

This is not standard four-player negotiated Catan.

The scope changes strategic and statistical properties:

- A state with no build, buy, development-card, or maritime action has only `END_TURN`; in standard Catan it might still contain a meaningful domestic-trade opportunity.
- Resource scarcity and port value differ without a player market.
- Robber incentives are strictly adversarial rather than political or coalition-sensitive.
- There is no kingmaking, table balancing, signaling, trade reputation, or multi-opponent threat ordering.
- Binary zero-sum value backup is valid for this track but does not generalize to four players.

The surrounding entity schema is substantially four-player-shaped, but production search math is not. A future four-player track requires a separately designed multi-seat value/backup convention and structured negotiation policy; it cannot be obtained by merely adding colors or widening the flat action catalog.

## 5. Hidden information and belief quality

### 5.1 Implemented safeguard: authoritative hidden truth is no longer searched

Neural observations are public-masked. Under `information_set_search`, the authoritative game supplies the root actor and legal root actions but is not evaluated or expanded. Search operates on rules-valid determinizations sampled by the Rust engine.

This closes the previous failure mode where a clone of authoritative state leaked actual opponent resources or development cards into traversal.

### 5.2 Deliberate approximation: public-conservation PIMC

The current belief mechanism samples worlds consistent with:

- bank/deck conservation;
- public hand sizes;
- the root player's private information;
- rules-valid hidden allocations.

It is explicitly **not** a history-conditioned Bayesian posterior.

It therefore fails to exploit deductions a strong player can make from:

- exact production history;
- build and maritime-trade spending;
- robber steals and likely stolen resource;
- discard sequences;
- development-card purchase and play timing;
- missed build opportunities;
- port access and feasible conversion chains;
- Monopoly exposure.

This is likely the largest Catan-specific strength ceiling.

### 5.3 The current event-history channel is empty

The current checkpoint contains an event-history consumer in its architecture contract, but native inference and the training corpora supply empty event tensors/masks. The network therefore has no usable temporal public-action sequence from which to learn opponent-hand deductions.

The current snapshot still exposes public board and count information, but a snapshot is not a sufficient statistic for the posterior over hidden hands and development cards.

Required direction:

1. Emit a compact, canonical public event stream from Rust generation and inference.
2. Encode resource production, spending, steals, discards, trades, development-card timing, and award changes.
3. Build a belief head or exact deduction state over opponent resource/development-card distributions.
4. Condition determinizations on that posterior.
5. Measure posterior calibration separately from playing strength.

### 5.4 Four particles are thin

Full n128 search uses four determinizations. Fast n16 uses one determinization because the minimum per-particle budget is 32.

Consequences:

- A full row receives only 32 simulations per hidden world.
- Fast trajectory play is based on one sampled hidden world.
- Rare but strategically decisive holdings may be absent from all four worlds.
- The selected policy can suffer PIMC strategy fusion and nonlocality.

Particle count, simulations per particle, and posterior quality are separate variables and should not be collapsed into a single "n128" label.

### 5.5 Policy aggregation is noncommutative

The production default averages per-world improved policies:

```text
E_world[ improve(prior_world, completed_Q_world) ]
```

An implemented experimental alternative aggregates completed Q first and improves once:

```text
improve(E_world[completed_Q], E_world[prior])
```

Min-max normalization, softmax, and completed-Q transforms are nonlinear, so the two operators are not equivalent. This should be judged by target stability, teacher-versus-raw strength, and downstream candidate strength.

## 6. What n128 means in practice

Current search configuration:

```text
n_full = 128
n_fast = 16
p_full = 0.25
determinization_particles = 4
determinization_min_simulations = 32
c_visit = 50
c_scale = 0.10
D6 root averaging from legal width >= 20
```

For a nonforced full row, n128 is one total nominal budget divided into four 32-simulation searches. It is not 128 simulations in each hidden world.

### 6.1 Sparse policy supervision

Only the 25% full-search draw among nonforced prompts becomes a policy target. This yields 12.4% policy-active rows overall.

Fast n16 decisions still shape the state distribution and terminal outcomes, even though their policies are not distilled. The flywheel therefore learns from trajectories partially generated by a much weaker, single-world operator.

Questions to test:

- Should `p_full` depend on strategic phase rather than a random draw?
- Should high-entropy, high-regret, or high-value-of-information states always receive full search?
- Should fast-row actions be selected by the raw policy, a cheap belief ensemble, or n16 search?
- Would fewer but consistently strong trajectories beat more mixed-quality trajectories?

### 6.2 Opening-placement budget is especially shallow

An initial settlement can expose 54 legal vertices. A full row has 32 simulations per particle, fewer simulations than root actions. Gumbel candidate selection and completed-Q allow the operator to produce a distribution, but many actions remain dominated by network prior/value completion rather than direct rollout evidence.

This makes the phrase "n128 opening search" stronger-sounding than the actual per-world coverage.

High-value alternatives:

- always-full opening settlement and road;
- adaptive n256 only for wide opening roots;
- uncertainty/entropy-triggered compute;
- phase-specific candidate caps backed by Catan expertise;
- targeted opening-book reanalysis.

Global n256 was previously inconclusive and more expensive. That result does not answer whether adaptive opening-only compute is valuable.

## 7. Search horizon stops at the actor-turn boundary

Information-set search terminates explicit traversal when control leaves the root player's current turn. The value network supplies the continuation value from the first opponent boundary.

MCTS can reason over within-turn sequences such as:

- maritime trade -> build;
- development-card play -> robber move;
- multiple builds before ending the turn.

It does not explicitly search the opponent's response or a later return to the root player. Therefore it cannot directly tree-search:

- an opponent blocking a road or settlement next turn;
- robber retaliation;
- multi-turn races;
- development-card timing across turns;
- a plan whose value depends on the opponent's belief-conditioned reply.

Increasing n128 to n256 improves the current-turn search but does not remove this horizon ceiling.

Possible research directions include a belief-state value model with much stronger opponent-response calibration, opponent-boundary shallow rollouts, or a recurrent/search latent model. Any extension must avoid letting the root actor condition future choices on one sampled hidden world.

## 8. Chance-node semantics

Implemented behavior includes:

- exact 11-sum enumeration for root/forced rolls;
- enumerated robber-steal and development-card-draw outcomes in the sampled world;
- corrected Rust chance spectra;
- lazy interior chance in production for throughput.

Lazy interior chance is an unbiased but higher-variance approximation and differs from full expectimax. Because information-set search stops at the actor boundary, not all theoretical interior chance paths are equally common, but the distinction remains science-bearing.

Belief chance is also different from sampled-world chance. Exact enumeration within one determinization does not make the distribution a calibrated public belief over the opponent's hand.

## 9. Exploration temperature uses the wrong clock

Production uses temperature 1.0 through engine `decision_index=90`, then temperature 0.0. Late-game temperature is disabled.

`decision_index` counts every serialized prompt, including forced rolls, sole end turns, and discard subprompts. In the measured fresh corpus, the first 90 prompt indices contain only about 39.15 nonforced choices per game on average.

Therefore "temperature for the first 90 decisions" means neither:

- 90 human turns;
- 90 strategic decisions;
- a fixed VP/game phase;
- a fixed number of policy-active targets.

Its effective strategic duration also changes with seven/discard frequency and prompt density.

Better clocks to test:

- nonforced-choice count;
- human turn count;
- public VP/game phase;
- policy entropy or margin;
- explicit opening/midgame/endgame state;
- remaining strategic horizon.

## 10. Symmetry and phase allocation

D6 root symmetry averaging activates only at legal width >=20.

This captures 54-wide initial settlement roots and other very wide positions, but many robber roots are near width 18 and fall immediately below the threshold. Robber placement is spatially structured, strategically important, and susceptible to representation noise; there is no Catan-theoretic reason that width 20 should be its boundary.

Training-time symmetry augmentation is also disabled in the current checkpoint. Root-time averaging and training-time augmentation solve related but different problems.

Recommended comparisons:

- D6 for all opening and robber prompts regardless of width;
- threshold control versus phase-gated D6;
- training-time symmetry augmentation;
- calibration and action agreement across transformed states.

## 11. Current neural architecture and blind spots

The current producer checkpoint is an approximately 35M-parameter entity/graph transformer with:

```text
hidden size: 640
state layers: 6
attention heads: 8
dropout: 0.05
public board/player/global/action tokens
scalar tanh value readout
```

The deployed checkpoint has these architectural options disabled:

- action target-ID gather;
- action-to-board cross-attention;
- attention-pooled value head;
- categorical value head;
- value uncertainty head;
- auxiliary subgoal heads;
- usable event-history input.

### 11.1 Spatial action grounding

Road, settlement, city, and robber actions refer to specific edges, vertices, or hexes. Without direct target-token gather or cross-attention, action scoring relies on a global state representation plus hand-engineered action context rather than explicitly reading the referenced board entity.

This may limit fine distinctions such as:

- road connectivity and future blocking;
- settlement production/port combinations;
- robber placement versus victim choice;
- local race geometry;
- longest-road branch structure.

The action context mitigates the problem but does not prove the global representation preserves every target-local relation.

### 11.2 Sparse scalar value supervision

The primary value head predicts terminal win/loss. Current training does not directly supervise:

- final VP or VP margin;
- production strength;
- resource flexibility;
- longest-road/largest-army progress;
- calibrated uncertainty;
- next settlement/robber subgoals;
- belief quality.

Dense auxiliary targets can improve the shared representation even when they are not used directly at inference. Their value must be tested on fresh authenticated data and paired playing strength, not assumed from training loss.

## 12. Replay and opponent distribution

The current learner composite uses source -> game -> row sampling with effective game ratios:

```text
64% current producer
12% recent history
 4% selected hard negative
20% historical replay
```

Equivalently, the fresh 80% is split 80/15/5 among current/recent/hard-negative sources.

In opponent-pool games, only the current producer's seat is retained as a distillation target. This correctly avoids imitating the old or exploitative opponent, but it also changes seat/state coverage relative to pure two-seat self-play.

Replay trades stability against stale semantics:

- old improved policies were produced by older checkpoints/search operators;
- old terminal outcomes are conditional on old continuation policies;
- replay can prevent forgetting but can also pull the candidate toward an exhausted fixed point.

Authenticated component scopes exist to isolate fresh policy/value objectives. The production choice should be explicit about which replay signals remain active: policy imitation, value outcome, stored-prior KL, or only representation rehearsal.

A single recent checkpoint and one selected hard negative are also a narrow approximation to a population. Exploitability should be measured against a broader league and targeted counter-strategies.

## 13. Evaluation scope

Internal evaluation has important safeguards:

- paired seeds;
- seat swapping;
- searched candidate-versus-incumbent play;
- pentanomial/SPRT-style statistical decisions;
- public-observation and information-set requirements.

Its conclusions remain scoped to 2p no-trade. Internal self-play head-to-head can also miss shared blind spots: two related checkpoints may agree on a bad strategy and appear evenly matched.

A credible strength claim should combine:

- incumbent head-to-head;
- recent-champion and population panels;
- hard-negative/exploiter panels;
- external Catanatron bots under a synchronized legal engine;
- opening, robber, development-card, and endgame slices;
- value and belief calibration;
- regression checks for known tactical positions.

External evaluation must use the same public-information and search semantics as production. A raw network, authoritative-state search, and public-conservation PIMC agent are different estimands and must not share one headline score.

## 14. Prioritized issue register

### P0: likely direct strength or compute impact

1. Build and train on canonical public event history.
2. Replace uniform conservation sampling with a calibrated history-conditioned belief or deduction model.
3. Split forced lifecycle categories in learner weighting; test phase-specific value weights.
4. Stop expensive forced-root evaluation when configured learner objectives cannot consume it.
5. Replace prompt-index temperature scheduling with a strategic clock.
6. Prove that the deployed search operator is stronger than raw policy for each promoted checkpoint.

### P1: high-value causal experiments

1. Always-full/adaptive compute for opening settlement, opening road, robber, and other high-information phases.
2. Mean-improved-policy versus aggregate-Q-then-improve belief targets.
3. Consume calibrated root-Q/afterstate evidence instead of storing and discarding it.
4. Action target gather and/or action-to-board cross-attention.
5. Value-attention pooling, categorical value, uncertainty, VP, and auxiliary heads.
6. Phase-gated D6 including robber positions below width 20.
7. Wider opponent population and explicit exploitability panels.

### P2: separate product/research scope

1. Four-player value semantics and multi-seat MCTS backup.
2. Structured domestic-trade/negotiation policy.
3. Four-player belief, coalition, and opponent-modeling semantics.
4. Multiplayer rating and promotion statistics.

These are not small extensions to the present 2p zero-sum system.

## 15. Questions for a top Catan player

The following expert judgments would directly improve experiment design:

1. Which public-history deductions most often change your move: exact resource holdings, development-card likelihood, Monopoly exposure, build feasibility, or something else?
2. Which phases deserve guaranteed expensive search: both initial placements, initial roads, robber placement, Monopoly, Road Building, settlement races, or late endgames?
3. In 2p no-trade, when `END_TURN` is the only engine action, are there strategically relevant choices the abstraction has already removed?
4. Which robber decisions depend most on inferred hand composition versus visible board/VP threat?
5. Is early exploration better indexed by turns, nonforced decisions, public VP, or opening completion?
6. Which heuristics distinguish a merely legal opening from an elite opening that a 32-simulation/world teacher may miss?
7. Which repeated tactical patterns should become a permanent regression suite?
8. What external opponents or fixed positions best expose inbred self-play strategies?

## 16. Code map

- Forced-row construction and row weights: `src/catan_zero/rl/gumbel_self_play.py`
- Forced single-action and chance handling: `src/catan_zero/search/gumbel_chance_mcts.py`
- Information-set PIMC, particle budgets, turn boundary, and target aggregation: `src/catan_zero/search/gumbel_chance_mcts.py`
- Public Rust evaluator and entity features: `src/catan_zero/search/neural_rust_mcts.py`
- Policy/value sample weighting and target construction: `tools/train_bc.py`
- Flat action catalog and domestic-trade scope: `src/catan_zero/rl/action_mask.py`
- Entity/action architecture: `src/catan_zero/rl/entity_token_policy.py`
- Production configuration authority: `RL_AGENT_HANDOFF.md`
- Four-player/trade extension analysis: `docs/audits/CAT74_4P_TRADE_EXTENSION_AUDIT.md`

## Bottom line

The 50% number is not evidence that the simulated game is obviously broken. It is evidence that engine-prompt granularity was mislabeled as Catan-move granularity.

The forced-policy handling is correct. The unresolved forced-row issue is full terminal-value/trunk weighting plus expensive search evidence that the learner currently discards.

The larger strength ceiling is more fundamental: a top player reasons from public history about hidden hands and opponent intent, while the current agent uses an empty history channel, four conservation-only particles, shallow per-world search, and a value network at the opponent-turn boundary. Those are the assumptions most worth attacking.
