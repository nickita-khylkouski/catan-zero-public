# A1 historical training-signal root audit — 2026-07-16

## Scope and decision

This is a read-only forensic audit of historical learner reports on
`ubuntu@149.118.65.110` plus the canonical repository. No GPU jobs were started
and no remote artifact was changed.

The dominant failure was not a lack of rows or a model that could not fit the
teacher. The learner repeatedly made the training objective look better while
moving away from broad playing strength. The largest causes, in priority order,
were:

1. the legacy policy corpus teaches a different PIMC search operator than the
   current coherent-public agent;
2. active-policy experiments changed dose semantics without preserving a
   comparable objective;
3. concentrated samplers repeatedly trained on a few thousand roots and made
   training loss look substantially better while validation became worse;
4. the coherent corpus silently cut ordinary strategic-turn policy mass roughly
   in half relative to the successful learner;
5. learning-rate selection against one related parent produced a different
   ranking against the recovered v5 model;
6. offline teacher closure kept improving after actual playing strength had
   started regressing;
7. value-pressure controls affected the small value tower but not value
   gradients entering the 29.5M-parameter shared trunk;
8. several failed "n128/n256" conclusions were actually candidate-chaining,
   oversized-dose, and wrong-parent evaluation conclusions.

The root-level repair is to define one learner update by objective coefficient,
unique-root exposure, functional parent drift, and multi-parent playing
strength—not by optimizer steps or sampled-row events alone.

## Ranked finding 1 — the legacy policy corpus teaches the wrong search operator

The legacy 196k composite's policy targets are all
`public_conservation_pimc_v1`, while the current agent uses coherent
public-belief single-tree search:

```text
configs/operations/a1-target-identity-coherent-n128-rd-v1/README.md:3-16
configs/operations/a1-target-identity-coherent-n128-rd-v1/contract.json:131-148
```

The contract explicitly forbids mixing those legacy PIMC rows into the
coherent learner's policy objective. They remain eligible only as separate
value evidence or as state evidence after valid reanalysis.

This is not a metadata-only distinction. PIMC distributes work across
separately determinized hidden worlds; the coherent operator maintains one
public-belief tree. Their action distributions can disagree even with the same
checkpoint and root budget. Distilling PIMC targets and evaluating the result
inside coherent search asks the network to imitate one improvement operator
while serving another.

The archived-opponent 20% cannot be reconstructed by seed alone because only
the producer seat's action trace was retained. Opponent decisions made by a
different network are absent. Quietly "reanalysing" it as if it were complete
self-play would fabricate a trajectory.

### Root fix

- Policy-target eligibility must bind search implementation, belief regime,
  producer checkpoint, budget, D6, chance, and completed-Q semantics.
- Keep old PIMC rows value-only.
- Reanalyse reusable complete states with the exact coherent n128 operator.
- Generate new complete two-seat traces where historical trajectories are
  incomplete.
- Measure policy-gradient cosine between legacy PIMC and coherent targets on
  shared roots; negative or near-zero rows are actively contradictory.

## Ranked finding 2 — historical and current active-policy arms are different optimizers

The historical campaign at commit `b412bff` varied auxiliary batch size:

| Arm | AUX batch | AUX draws | AUX effective-weight share |
|---|---:|---:|---:|
| P10 | 46 | 47,104 | 14.84% |
| P25 | 116 | 118,784 | 30.51% |
| P50 | 232 | 237,568 | 46.75% |
| P100 | 463 | 474,112 | 63.66% |

Evidence:

```text
/home/ubuntu/experimental_nonpromotable/
  coherent-n128-active-policy-20260715-r1/
  campaign/run-b412bff/arms/{P10,P25,P50,P100}/train.report.json
```

Those reports have no `policy_aux_loss_weight`. At `b412bff`,
`tools/train_bc.py:14578-14585` added base and AUX weighted sums and divided by
their combined denominator:

```python
policy_loss_sum = policy_loss_sum + aux_sum
policy_loss_denominator = policy_loss_denominator + aux_denominator
policy_loss = weighted_mean(policy_loss_sum, policy_loss_denominator)
```

Therefore the arm names did not mean a 0.10/0.25/0.50/1.00 objective
coefficient. They changed the mixture measure and its sampling variance. Their
real AUX shares were 14.84%, 30.51%, 46.75%, and 63.66%.

The current trainer instead normalizes base and AUX independently and applies
an explicit coefficient:

```text
tools/train_bc.py:17117-17123
```

```python
policy_loss = policy_base_loss + policy_aux_loss_weight * policy_aux_loss
```

This current contract is scientifically cleaner, but it means a current "P10"
is not a replay of historical P10. Historical conclusions cannot be transferred
to the new optimizer by arm name.

The historical loss response was also small relative to the 10.06x AUX draw
range:

| Arm | Train policy loss |
|---|---:|
| P10 | 1.42826 |
| P25 | 1.42719 |
| P50 | 1.42590 |
| P100 | 1.42347 |

### Root fix

Bind these independently in every recipe and report:

- base-policy mean;
- AUX-policy mean;
- explicit AUX coefficient;
- each objective's effective-weight sum;
- unique source roots and reuse factor;
- parent-policy KL and layerwise drift.

Do not use batch size as a science coefficient.

## Ranked finding 3 — concentrated sampling reversed the train/validation ranking

The Stage-C aligned learner used 16,384 AUX draw events in only 32 optimizer
steps:

```text
/home/ubuntu/experimental_nonpromotable/
  stage-c-aligned-learner-ced4b14b-9fd85c3-r1/
  arms/{strategic-balanced,production-weighted}/learner/train.report.json
```

| Sampler | Unique AUX roots | Reuse factor | Train policy loss | Validation policy loss |
|---|---:|---:|---:|---:|
| strategic-balanced | 6,832 | 2.398x | 1.31371 | 1.31330 |
| production-weighted | 5,917 | 2.769x | 1.20780 | 1.31770 |

The production-weighted sampler improved training policy loss by `0.10592`,
yet validation policy loss was `0.00441` worse. This is a direct generalization
failure, not an inference from game noise. The apparently better optimizer
trajectory came from a narrower, more frequently repeated root distribution.

Current accounting explicitly acknowledges the broader problem:

```text
tools/train_bc.py:32017-32049
```

`training_row_draws` are draw events, sampling is with replacement, and
`unique_training_rows_drawn` is `None`. A reported 500k active-row dose is not
500k independent strategic examples.

### Root fix

- Define an AUX dose by unique root IDs, unique game IDs, and maximum reuse.
- Sample without replacement within an epoch-sized window where possible.
- Set a hard reuse ceiling for short-dose studies.
- Report effective sample size by phase, source, and teacher.
- Keep a source/phase-balanced whole-game validation set outside the active
  sampler's measure.
- Reanalyse new roots instead of replaying the same high-surprise subset harder.

## Ranked finding 4 — the coherent corpus halved strategic-turn policy mass

Under equal per-game weighting, the admitted 959,142-row coherent corpus assigns
only 34.16% of policy objective mass to ordinary `PLAY_TURN` decisions. The
historically successful selected-dose corpus assigned 66.08%:

```text
configs/operations/a1-next-wave-coherent-public-v3/README.md:81-93
```

This is a two-fold change in what the learner practices. Opening, discard, and
robber prompts are legitimate multi-action decisions, but letting them consume
65.84% of policy mass changes the trained skill distribution. A learner can
improve global CE while undertraining the repeated build/buy/road decisions
that dominate normal play.

The current `PLAY_TURN=4.0` phase repair restores 66.49%, close to the successful
66.08% reference. This is a substantive training-signal correction, not merely
sampling hygiene.

### Root fix

- Bind policy objective mass by phase, not just raw row counts.
- Preserve equal-per-game weighting inside each phase allocation.
- Report phase dose separately for base and AUX streams.
- Hold phase allocation fixed during LR, trunk, and target-reliability studies.
- Keep value sampling independently game-uniform so policy phase repair does
  not silently redefine the value distribution.

## Ranked finding 5 — the LR sweep selected different winners against f7 and v5

All four LR arms consumed the same nominal dose:

```text
base draws = 524,288
AUX draws = 474,112
policy-active draws = 539,687
steps = 128
```

Training reports:

```text
/home/ubuntu/experimental_nonpromotable/
  b200-lr-dose-f7-20260715-r5/arms/{A,B,C,D}/train.report.json
```

Evaluation summary:

```text
/home/ubuntu/experimental_nonpromotable/
  b200-lr-dose-f7-20260715-r5/
  eval-matrix-r5-native-0aa3cae/r5-results-summary.json
```

| Arm | LR / warmup | Train policy loss | Block delta norm | vs f7 | vs v5 |
|---|---|---:|---:|---:|---:|
| A | 3e-5 / 100 | 1.31230 | 0.02895 | 57.03% | 49.61% |
| B | 3e-5 / 16 | 1.29568 | 0.04238 | **58.98%** | **46.48%** |
| C | 6e-5 / 16 | 1.28656 | 0.07890 | 55.86% | **51.56%** |
| D | 1.2e-4 / 16 | **1.27930** | 0.14650 | 52.73% | 49.22% |

The lowest training loss was D, but it was not the strongest player. B was best
against f7 and worst against v5. C was weaker than B against f7 but was the only
arm above 50% against v5.

Changing only warmup from A to B improved f7 by 1.95 points while degrading v5
by 3.13 points. This is lineage-specific adaptation, not a universal LR win.

### Root fix

Never select optimizer settings against one related checkpoint. The minimum
short panel must include:

- exact initializer/current parent;
- recovered v5;
- one stylistically distinct historical checkpoint;
- a fixed-root teacher-regret suite.

Treat the result as a vector or worst-case score, not one win rate.

## Ranked finding 6 — teacher closure improved while playing strength regressed

The coherent dose frontier reports:

```text
docs/evidence/A1_COHERENT_DOSE_FRONTIER_20260716.json
/home/ubuntu/experimental_nonpromotable/
  coherent-n128-active-policy-20260715-r1/
  eval-126afe1/stage-a-no-selection-frontier/results.summary.json
```

| Step | Parent KL | Trunk relative L2 | Validation closure | vs f7 | vs v5 |
|---:|---:|---:|---:|---:|---:|
| 32 | 0.07918 | 0.01197 | 0.00391 | 56.25% | 51.17% |
| 64 | 0.10016 | 0.01816 | 0.02000 | 56.25% | 49.22% |
| 128 | 0.11761 | 0.02846 | 0.04454 | 51.95% | 45.31% |

From step 32 to 128, offline closure improved more than 11x while v5 playing
strength fell 5.86 points. This rules out using closure, CE, or more teacher
imitation as a monotonic promotion signal.

The likely mechanism is over-distillation of a finite/noisy teacher combined
with shared-trunk drift. Search targets are policy-improvement evidence, not
ground truth, and their reliability is heterogeneous.

### Root fix

- Stop or reduce the policy branch at a parent-KL/trunk-drift budget.
- Select checkpoints at 8/12/16/32/64/128 steps.
- Weight high-surprise targets by independent-search reliability.
- Move stale states to reanalysis rather than repeatedly optimizing closure.
- Promote only on playing-strength evidence under the deployed operator.

## Ranked finding 7 — `value_lr_mult` did not protect the shared representation

The canonical audit already establishes:

```text
docs/audits/A1_OPTIMIZER_OWNERSHIP_TRAINING_SIGNAL_AUDIT_20260716.md:71-139
```

- Transformer blocks: 29,541,120 parameters.
- Value-specific telemetry group: 410,881 parameters.
- `value_lr_mult=0.3` affects the latter, not value gradients entering the
  shared trunk.
- Historical coherent arms used `value_trunk_grad_scale=1.0`.
- Mean policy/value shared-gradient cosine ranged from `-0.0575` to `+0.0197`.
- 41.2% to 67.6% of observations had negative cosine.
- Value gradient norm was 0.514x to 0.891x policy gradient norm.

The named "lower value LR" treatment therefore did not test the main
representation-interference hypothesis. The causal knob is
`value_trunk_grad_scale`, a lower shared-trunk LR, or a split late tower.

### Root fix

Report and tune these independently:

- value-head LR;
- value-to-trunk gradient scale;
- shared-trunk LR;
- policy/value gradient cosine by layer.

The first production architecture change should be a small late policy/value
split or reduced trunk value routing, not a larger whole model.

## Ranked finding 8 — failed n128/n256 learners were not independent data tests

The prior forensic reconstruction is decisive:

```text
docs/audits/A1_LEARNER_END_TO_END_FORENSICS_20260713.md:27-74
```

- `combined-196k` initialized from an already trained n256 candidate.
- `corrective n128` initialized from corrective n256.
- Each chain accumulated 42.46M examples and 10,365 steps.
- Candidate drift from f7 reached 9.763%, 15.313%, and 34.129%.
- Internal adjudication used old gen3 at `c_scale=0.03`, not the actual
  initializer/f7 at deployed `c_scale=0.10`.

The independent n256 `lr=1.2e-4` arm beat f7 360-240/600 under the matched
operator. Stronger-search data was therefore not disproven. Candidate chaining,
oversized dose, and wrong-parent evaluation were.

### Root fix

Every scientific arm must:

- reload exact parent bytes independently;
- start fresh Adam/scheduler/scaler state;
- bind one target/search contract;
- consume one explicit objective dose;
- compare against its actual initializer under the same operator.

## Source-mixture conclusion

The successful selected dose used approximately:

| Source | Policy-active share |
|---|---:|
| current | 63.25% |
| replay | 20.79% |
| recent | 12.00% |
| hard-negative | 3.97% |

The later LR arms preserved nearly the same source proportions
(`63.29/20.72/12.06/3.93%`) while changing dose and optimizer strength.
Therefore source mixture is not the first-order explanation for those failures.
The larger defects were dose semantics, repeated roots, target/operator
identity, and shared-trunk drift.

Source mixture still needs a separate population/diversity study, but changing
it before repairing dose accounting would confound the result again.

## Required next learner contract

Before another expensive learner, bind:

1. exact parent SHA and fresh optimizer;
2. exact target/search/operator hash;
3. base and AUX objectives normalized independently;
4. explicit AUX coefficient;
5. unique roots, unique games, reuse factor, and ESS by source/phase;
6. policy/value gradient norms and cosine by shared layer;
7. parent-policy KL, top-one flip rate, and layerwise drift;
8. evaluation against parent, v5, and a distinct historical opponent;
9. checkpoint selection across the dose frontier;
10. no promotion from train loss, validation closure, or one related matchup.

This is the smallest root-level repair that makes the next training result
interpretable and gives the optimizer a realistic chance to preserve the
playing-strength signal already observed in short independent doses.
