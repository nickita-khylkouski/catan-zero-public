# A1 learner recovery: smallest high-information path to a winning model

Status: diagnostic plan. It does not authorize promotion or a production wave.

## What the current evidence rules in and rules out

The n128/n256 corpus is not failing basic integrity. The important failure is
competitive overfitting: optimizer updates improve some same-lineage matchups
while losing external strength. The learner is also not using all trustworthy
search supervision.

The authoritative n128/n256 locks rule out a producer/operator mismatch. Both
arms bind the promoted f7 checkpoint and handoff, and every current-producer
job uses `c_scale=0.10`; only gen3 recent-history and gen4 hard-negative jobs
intentionally use `.03`. A fresh 576-game native-runtime pilot also matched a
576-game historical current-producer sample: target/prior KL differed by
0.43%, forced fraction by 0.14 percentage points, full-search fraction by 0.13
points, and every phase fraction by less than 0.18 points. Both samples had
zero failures and truncations. The generator and teacher-target distribution
are therefore not the proximal regression source.

The strongest surviving evidence is learner-side. Failed candidates improved
held-out imitation metrics while regressing in the original external panel,
and 96--98% of their update energy landed in the shared trunk. Trunk drift
rose from 8.96% to 30.18% with learning rate while value-head drift remained
2.19--5.13%. The same 35M architecture produced both gen3 and the stronger f7
agent, so capacity is not the first hypothesis; behavior forgetting and
objective imbalance are.

The original internal labels did **not** establish monotonic learner
improvement. The combined candidate was initialized from an already-trained
n256 candidate, and the corrective n128 candidate was initialized from its
corrective n256 candidate. Each chain therefore accumulated 44,692,523 sampled
rows and 10,365 optimizer steps after f7 rather than taking one independent
champion-started dose. Their 52.14% and 55.45% internal H1 results were then
measured against the older gen3 checkpoint under a shared `c_scale=0.03`, not
against their actual initializer or f7's deployed `c_scale=0.10` agent
identity. Those H1 labels were a baseline/operator confound, not proof that the
chained candidates improved over their parent.

A zero-training correction made the consequence concrete: the independent
n256 LR=1.2e-4 checkpoint, previously reported as inconclusive internally,
beat its actual f7 initializer 360--240 over 600 games (60.0%, pentanomial
LLR +9.10) when both agents used their matched `c_scale=0.10` operator. Its
external matched-operator panel remains the binding generalization test. This
does not rehabilitate candidate chaining or oversized doses; it establishes
that checkpoint and search operator are one agent identity and that evaluation
against the wrong baseline/operator can reverse the scientific conclusion.

Forced rows are **not** the missing policy signal. Generation writes
`policy_weight_multiplier=0` for every single-legal-action row and every
fast-PCR row. `train_bc` multiplies by that field before its weighted-mean
policy loss, so those rows contribute neither policy numerator nor policy
denominator. Forced rows remain value rows by design. They represent real
states and realized outcomes; dropping them is an ablation, not a safe default.

The most defensible unused search signal is `root_value` on non-forced
full-search rows. Current runs use `value_target_lambda=1`, hence pure terminal
outcomes. `target_scores` must not be enabled blindly as a Q loss: current
Gumbel shards explicitly contain raw visit Q for visited actions, **not** the
completed-Q values used to form the improved policy. Moreover the optional
learner loss row-standardizes those values, which is incompatible with a head
later interpreted as return-scale Q. Other teacher sources can carry preference
scores in the same generic column. Bind source provenance and one head semantic
contract before a Q arm.

The existing one-epoch LR curve also used a much larger update dose than the
incumbent's training dose. LR, batch size and epochs must therefore be compared
at equal **samples seen**, not equal optimizer steps or equal epochs.

## Defaults for this recovery program

These defaults apply to every arm unless the arm names an explicit delta:

- initialization: the f7 producer bytes, independently reloaded
  for every arm; never chain arms through each other's optimizer state;
- provenance enforcement: every direct arm must report an initialization SHA
  equal to the declared producer/incumbent SHA. Sequential checkpoint curricula
  are refused unless an `a1-curriculum-declaration-v1` authenticates the parent
  receipt/checkpoint and carries prior plus cumulative sampled-row and optimizer-
  step dose under `a1-lineage-dose-v1`. These recovery doses intentionally use
  fresh Adam state, so cumulative steps mean lineage exposure rather than a
  restored scheduler counter. The schema also reserves typed policy-active,
  value-active and anchor-eligible exposure fields; they remain explicitly null
  until the trainer emits exact training-split counts (corpus-wide weight
  telemetry is not an acceptable substitute);
- data: globally shuffled n128+n256 rows plus authenticated incumbent-era
  mixed replay (approximately 80% f7, 15% gen3 history, 5% gen4 hard-negative)
  sampled component → uniform game → uniform row, with game-disjoint validation;
- loss weighting: use the independently validated L1 correction: winner and
  loser policy weights both 1.0. The historical f7 learner used loser weight
  0.3, but that outcome-conditioned the search-policy objective and allocated
  only 18.14% of policy mass to losing trajectories; L1 removed that bias and
  won both the direct incumbent gate and the matched external panel. Keep the
  value-head LR multiplier at 0.3, no per-game policy/value loss correction,
  forced policy unchanged (already zero from the corpus multiplier), and
  forced value weight 1.0 initially. The composite sampler already samples
  component → uniform game → uniform row; another inverse-length loss
  factor would double-correct game length and over-weight short games;
- optimizer: Adam, FP32, 100-step warmup, flat LR for the matched baseline.
  BF16 remains a separately measured systems treatment; changing precision in
  a learner arm would violate the one-axis contract;
- LR: flat `3e-5` for P1. The f7/gen3 topology has no action-local gather or
  cross-attention parameters, so an action-module `2x` multiplier is a fake
  no-op and is rejected. P2 localizes whether shared-trunk updates are the
  source of forgetting before adding any discriminative LR;
- primary value objective: scalar MSE until the stability recipe is selected;
- search-value blend: lambda 1.0 until the stability recipe is selected;
- update dose: first adjudicate the already-written 524,288- and
  4,194,304-sample checkpoints on identical paired seeds. Select the smallest
  dose within two percentage points of the best behavior result; do not infer
  dose from offline loss or automatically escalate to 8,388,608 samples;
- validation: a deterministic, game-disjoint 262,144-row sentinel for each
  short arm; the external playing panel is binding, so repeatedly scoring the
  full multi-million-row holdout is wasted latency;
- batch size: use the largest measured batch before throughput plateaus or
  quality changes. HBM occupancy is telemetry, not the objective. Recompute
  `max_steps = ceil(sample_dose / (local_batch * world_size * grad_accum))`.

## Pareto-ranked execution sequence

### P0 — free recovery: checkpoint interpolation (completed)

The alpha-0.10 candidate scored 46.43% internally and 48.21% externally versus
the incumbent's 50% external result. It was not a winner, so larger alphas
were pruned. Interpolation remains a valid cheap recovery tool, but it did not
solve this regression.

### P1 — anti-forgetting anchor sweep (three short B200 arms)

Prerequisite: repair the anchor scope and direction before spending this sweep.
The current implementation averages reverse `KL(new || prior)` over all rows
with priors. Forced single-action rows have identically zero KL but still enter
that denominator, diluting the configured coefficient by roughly their corpus
fraction. Use multi-action rows only (retain fast-PCR multi-action rows as
rehearsal) and forward `KL(prior || new)`/old-policy cross-entropy as the
behavior-preservation objective. Preserve the reverse direction only as an
explicit legacy ablation. This implementation repair is complete and
regression-tested.

Hold every field fixed and sweep only `policy_kl_anchor_weight`:

| Arm | Conditional KL weight | Global-mass equivalent | Purpose |
|---|---:|---:|---|
| K0 | 0.000 | 0.00 | corrected mixed-data control |
| K3 | 0.006 | 0.03 | light behavioral anchor |
| K10 | 0.020 | 0.10 | strong behavioral anchor |

All three arms use a fixed 20% authenticated incumbent-era mixed replay
component; K0 tests
replay without a behavior anchor, while K3/K10 isolate anchor strength. Run
the 4.19M-sample sentinel from the same initialization. Advance at most the
Pareto winner to 8.39M samples, again from the original initialization. Current
rows' stored priors came from the current wave and are anchor-ineligible. The
anchor is computed only against verified priors on the replay component
(mostly f7, with gen3/gen4 population coverage); it is not a second teacher
target.

The trainer normalizes KL over authenticated replay rows that both carry a
prior and have more than one legal action. Its coefficient therefore multiplies
`E[KL | eligible replay row]`, not an all-corpus row mean. With fixed replay
mass 0.20, the configured weights are `0.20 * {0.03, 0.10}`. Training and eval
reports record this normalization and aggregate the KL numerator over the exact
eligible-row denominator; batches without eligible rows no longer dilute the
reported anchor metric.

### P2 — localize the destructive update

Reuse the selected P1 checkpoint as the full-update control at zero additional
training cost. From an independent f7 initialization, run one matched
`--freeze-modules trunk` arm with the selected P1 replay/KL recipe. If this
head-only arm restores external strength, destructive shared-trunk updates are
causal. Only then add a parameter-group trunk LR multiplier and compare trunk
multipliers `{0, 0.1, 0.25}` at fixed total sample dose; do not infer a
discriminative LR from a full-model LR change.

If every K arm forgets despite 20% replay, add one R0 control with no replay at
the selected KL weight. Do not sweep replay ratios unless that matched control
shows replay itself is harmful. Replay remains component → uniform game →
uniform row so long games cannot dominate by row count.

### P3 — use trustworthy search value

With the selected anti-forgetting recipe, compare:

| Arm | Value target | Value head |
|---|---|---|
| V100 | `1.00*z` | scalar MSE |
| V75 | `0.75*z + 0.25*V_search` on masked root rows | scalar MSE |
| VH75 | same lambda 0.75 blend | 33-bin HL-Gauss |

V100 is the matched control. V75 asks whether search values improve the noisy
terminal target without making it mostly self-referential. VH75 is launched
only after the MSE comparison establishes that the target blend itself is not
harmful. HL-Gauss uses sigma/bin-width 0.75 and no scalar auxiliary, matching
the primary-objective budget.

Do not enable `q_loss_weight` until an audit proves that every admitted score
has completed-Q return scale, consistent perspective, finite mask and no
Gumbel/preference transform. If proven, test Q loss 0.02 as a later additive
auxiliary, never as part of P1.

### P4 — forced-value and dose curve

Only after P1-P3, compare `forced_row_value_weight=1.0` against 0.25. The policy
side is already zero. The question is whether forced-state value labels help
state coverage or merely add correlated terminal labels. The f7-matched
component → game → row sampler remains unchanged in both arms; no second
per-game loss correction is added.

For the winning objective, run independent producer-started sample doses at
4.19M, 8.39M and 16.78M samples. This is the actual early-stopping curve.
Epoch boundaries are not comparable when corpus or batch size changes.

### P5 — architecture only after the learner is stable

Run `tools/a1_post_p1_diagnosis_plan.py`.  It reuses the selected P1 full-update
checkpoint as the no-cost control, then adds only two matched 4.19M-sample
arms: a trunk-frozen head update and that same trunk-frozen update with the
zero-init `gather,cross:1` action path.  This is smaller and more attributable
than the older mixed relational probe, whose recipe predates the corrected
forced-value, per-game, replay/anchor and producer-identity work.

The three-way outcome separates the proximal hypotheses: head-only recovering
external strength implicates trunk optimization/forgetting; gather/cross
improving over head-only implicates missing action-local capacity.  Auxiliary
heads and root-value blending are held off because they alter the objective,
not just the architecture. Root-value blending remains deferred here because
it changes the objective and the stored roots are correlated, stale f7 search
estimates—not because their operator identity is invalid. Scale beyond 35M
only if the corrected 35M
learner still underfits active teacher targets and its external strength is
non-regressing.

## Arm adjudication

Every arm is diagnostic and must use the same common-random-number cohorts.
Rank lexicographically, then Pareto-check secondary metrics:

1. external bot/population result (binding anti-overfit tripwire);
2. internal candidate-vs-champion result;
3. active-only teacher-gap closure and target-to-model KL;
4. value calibration/RMSE by phase and root width;
5. parameter drift, model-to-champion KL and optimizer clipping telemetry;
6. throughput and peak HBM as constraints, not strength metrics.

Terminate an arm's dose escalation when it is externally dominated, parameter
drift exceeds the best non-regressing arm without a strength gain, validation
and teacher-gap metrics jointly worsen, or non-finite/clipping telemetry trips.
Do not select by validation loss alone: the current LR curve already showed it
does not predict external playing strength.

The primary learner comparison fixes both candidate and exact f7 to the same
deployed `c_scale=0.10` operator on common random numbers. Do not tune a
candidate-specific `c_scale` against old gen3: that changes checkpoint ancestry
and search behavior together and recreates the confound that invalidated the
historical 52--55% labels. A predeclared same-checkpoint `.03`/`.10` operator
crossover may be run as a separate search experiment, but it cannot select or
relabel a learner arm.

## Expected compute order

P0 is complete. P1 trains sequentially on 8xB200 while completed checkpoints
evaluate on the approved 40-H100 fleet. P1's three 4.19M-sample arms are the
highest-information training spend. P2 adds one head-only localization arm;
only a positive result unlocks a trunk-LR sweep. P3 adds one value-blend arm,
then categorical value only if the blend wins. P4 adds one forced-value arm.
P5 is conditional. This is successive halving, not an exhaustive Cartesian
grid.

Primary methodological references already reflected in the repository review:
Farebrother et al., *Stop Regressing* (2024) for categorical value robustness;
Schrittwieser et al., *MuZero Reanalyse* (2021) and ReZero (2024) for refreshed
search targets; Wu, *Accelerating Self-Play Learning in Go* (2019) for
playout-cap randomization, weighted sampling and auxiliary supervision.
