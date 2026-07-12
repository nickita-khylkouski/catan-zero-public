# A1 learner recovery: smallest high-information path to a winning model

Status: diagnostic plan. It does not authorize promotion or a production wave.

## What the current evidence rules in and rules out

The n128/n256 corpus is not failing basic integrity. The important failure is
competitive overfitting: optimizer updates improve some same-lineage matchups
while losing external strength. The learner is also not using all trustworthy
search supervision.

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

- initialization: the current champion/producer bytes, independently reloaded
  for every arm; never chain arms through each other's optimizer state;
- data: globally shuffled n128+n256 rows, with game-disjoint validation;
- loss weighting: loser weight 1.0, per-game policy `sqrt`, per-game value
  `sqrt`, forced policy unchanged (already zero from corpus multiplier), forced
  value weight 1.0 initially;
- optimizer: Adam, bf16, 100-step warmup, flat LR for a short dose;
- discriminative LR: trunk `3e-5`, action modules `2x` (effective `6e-5`),
  value modules `1x` (effective `3e-5`);
- primary value objective: scalar MSE until the stability recipe is selected;
- search-value blend: lambda 1.0 until the stability recipe is selected;
- update dose: first sentinel at 4,194,304 global samples, then 8,388,608 only
  for Pareto-surviving recipes;
- batch size: use the largest measured batch before throughput plateaus or
  quality changes. HBM occupancy is telemetry, not the objective. Recompute
  `max_steps = ceil(sample_dose / (local_batch * world_size * grad_accum))`.

## Pareto-ranked execution sequence

### P0 — free recovery: checkpoint interpolation

Evaluate champion-to-candidate interpolation at alpha 0.10, 0.25, 0.50 and
1.00 using common-random-number internal and external panels. This can recover
a useful partial update without another training run. Reject any alpha that
loses external strength even if its internal panel improves.

### P1 — anti-forgetting anchor sweep (three short B200 arms)

Prerequisite: repair the anchor scope and direction before spending this sweep.
The current implementation averages reverse `KL(new || prior)` over all rows
with priors. Forced single-action rows have identically zero KL but still enter
that denominator, diluting the configured coefficient by roughly their corpus
fraction. Use multi-action rows only (retain fast-PCR multi-action rows as
rehearsal) and forward `KL(prior || new)`/old-policy cross-entropy as the
behavior-preservation objective. Preserve the reverse direction only as an
explicit legacy ablation.

Hold every field fixed and sweep only `policy_kl_anchor_weight`:

| Arm | KL weight | Purpose |
|---|---:|---|
| K0 | 0.00 | corrected mixed-data control |
| K3 | 0.03 | light behavioral anchor |
| K10 | 0.10 | strong behavioral anchor |

Run the 4.19M-sample sentinel from the same initialization. Advance at most the
Pareto winner to 8.39M samples, again from the original initialization. The
anchor is computed against stored pre-search champion priors on eligible rows;
it is not a second teacher target.

### P2 — replay and discriminative-LR separation

First compare the P1 winner against one full-update arm with base LR `6e-5`,
action multiplier 1.0 and the same winning KL. This isolates whether protecting
the trunk matters.

Then add old-gen3 replay by game, not by raw row. Compare replay ratios 0%, 10%
and 20% while keeping the total sample dose fixed. Replay games must receive the
same per-game weighting as fresh games. Do not let a long old game acquire more
mass merely because it contains more rows. If a replay corpus lacks current
soft-target/root-value semantics, use it for a champion KL/rehearsal objective,
not as if it were fresh n256 supervision.

Advance replay only when it improves the external sentinel or materially
reduces parameter/behavior drift without erasing active teacher-gap closure.

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

Run the existing matched transformer-vs-relational-action probe using the
winning data/loss/dose recipe. Do not attribute an architecture loss to the
architecture if it was trained with the old unstable value/forgetting recipe.
Scale beyond 35M only if the corrected 35M learner still underfits active
teacher targets and its external strength is non-regressing.

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

## Expected compute order

P0 runs on the evaluator fleet while P1 trains sequentially on 8xB200. P1's
three 4.19M-sample arms are the highest-information training spend. P2 adds at
most two new arms after the P1 loser arms are killed. P3 adds two arms. P4 and
P5 are conditional. This is successive halving, not an exhaustive Cartesian
grid.

Primary methodological references already reflected in the repository review:
Farebrother et al., *Stop Regressing* (2024) for categorical value robustness;
Schrittwieser et al., *MuZero Reanalyse* (2021) and ReZero (2024) for refreshed
search targets; Wu, *Accelerating Self-Play Learning in Go* (2019) for
playout-cap randomization, weighted sampling and auxiliary supervision.
