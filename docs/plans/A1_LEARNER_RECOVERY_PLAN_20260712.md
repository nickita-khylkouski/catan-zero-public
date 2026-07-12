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
held-out loss or internal play while regressing externally, and 96--98% of
their update energy landed in the shared trunk. Trunk drift rose from 8.96%
to 30.18% with learning rate while value-head drift remained 2.19--5.13%.
The same 35M architecture produced both gen3 and the stronger f7 agent, so
capacity is not the first hypothesis; behavior forgetting and objective
imbalance are.

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
- data: globally shuffled n128+n256 rows plus authenticated incumbent-era
  mixed replay (approximately 80% f7, 15% gen3 history, 5% gen4 hard-negative)
  sampled component → uniform game → uniform row, with game-disjoint validation;
- loss weighting: loser weight 1.0, per-game policy `sqrt`, per-game value
  `sqrt`, forced policy unchanged (already zero from corpus multiplier), forced
  value weight 1.0 initially;
- optimizer: Adam, bf16, 100-step warmup, flat LR for a short dose;
- LR: flat `3e-5` for P1. The f7/gen3 topology has no action-local gather or
  cross-attention parameters, so an action-module `2x` multiplier is a fake
  no-op and is rejected. P2 localizes whether shared-trunk updates are the
  source of forgetting before adding any discriminative LR;
- primary value objective: scalar MSE until the stability recipe is selected;
- search-value blend: lambda 1.0 until the stability recipe is selected;
- update dose: first sentinel at 4,194,304 global samples, then 8,388,608 only
  for Pareto-surviving recipes;
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

| Arm | KL weight | Purpose |
|---|---:|---|
| K0 | 0.00 | corrected mixed-data control |
| K3 | 0.03 | light behavioral anchor |
| K10 | 0.10 | strong behavioral anchor |

All three arms use a fixed 20% authenticated incumbent-era mixed replay
component; K0 tests
replay without a behavior anchor, while K3/K10 isolate anchor strength. Run
the 4.19M-sample sentinel from the same initialization. Advance at most the
Pareto winner to 8.39M samples, again from the original initialization. Current
rows' stored priors came from the current wave and are anchor-ineligible. The
anchor is computed only against verified priors on the replay component
(mostly f7, with gen3/gen4 population coverage); it is not a second teacher
target.

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
state coverage or merely add correlated terminal labels. Per-game sqrt remains
on in both arms.

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

Before external evaluation, calibrate each candidate checkpoint internally at
candidate `c_scale ∈ {0.03, 0.10}` against gen3 at `.03` on common random
numbers. The selected checkpoint+operator pair is the candidate identity used
for the external panel; never silently evaluate every new checkpoint under a
shared `.03` assumption.

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
