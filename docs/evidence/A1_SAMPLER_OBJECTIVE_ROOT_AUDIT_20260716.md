# A1 sampler/objective root audit — 2026-07-16

## Scope

This is a read-only audit of the current learner plus historical reports on
`ubuntu@149.118.65.110`. No GPU job was started and no remote artifact was
changed.

The main conclusion is that several historical “active-policy dose” results
were not measuring the treatment their names implied. The old learner coupled
the active-stream batch size, selected-root sparsity, and row-weight scale into
one hidden objective coefficient. The current learner has repaired the
base/AUX normalization and DDP denominator, but production still needs a
unique-root/reuse contract before another active-policy campaign is
interpretable.

## Finding 1 — Stage-C was a 94% AUX-policy objective, not a sampler comparison

The two Stage-C reports used only 32 optimizer steps and 16,384 AUX draw events,
but the base corpus had policy loss on only the 8,178 reanalysed rows out of
959,142 total rows. After the distillation scope was applied, selected policy
rows had a mean policy weight of `117.2832`.

Historical sufficient statistics:

| arm | base effective weight | AUX effective weight | AUX objective share | AUX/base |
|---|---:|---:|---:|---:|
| strategic-balanced | 128,887.16 | 1,922,913.56 | 93.718% | 14.919x |
| production-weighted | 128,887.16 | 1,893,456.66 | 93.627% | 14.691x |

Sources:

```text
/home/ubuntu/experimental_nonpromotable/
  stage-c-aligned-learner-ced4b14b-9fd85c3-r1/
  arms/{strategic-balanced,production-weighted}/learner/train.report.json
```

At the historical code revision, base and AUX weighted sums were added and
divided by their combined denominator. Therefore these arms were overwhelmingly
training on the repeatedly sampled selected-root stream. They were not a
small policy correction laid over the base learner.

This explains the otherwise paradoxical result:

| sampler | unique AUX roots | reuse | train policy loss | validation policy loss |
|---|---:|---:|---:|---:|
| strategic-balanced | 6,832 | 2.398x | 1.31371 | 1.31330 |
| production-weighted | 5,917 | 2.769x | 1.20780 | 1.31770 |

The narrower production sampler improved training loss by `0.10592` while
making validation loss `0.00441` worse. The optimizer learned the repeated
subset more aggressively; it did not learn a more general policy.

### Smallest production fix

The current independent objective

```text
base_policy_mean + explicit_aux_coefficient * aux_policy_mean
```

is the correct normalization and must remain. Historical Stage-C results must
not be used to select the current AUX coefficient. Any new active campaign must
bind:

- the explicit AUX coefficient;
- unique roots and unique games;
- draw/unique reuse factor;
- maximum per-root reuse;
- base and AUX effective-weight sums separately.

## Finding 2 — historical P10/P25/P50/P100 labels were hidden mixture weights

The older coherent active-policy campaign varied AUX batch size but had no
explicit `policy_aux_loss_weight`:

| arm | AUX draws | realized AUX share |
|---|---:|---:|
| P10 | 47,104 | 14.839% |
| P25 | 118,784 | 30.512% |
| P50 | 237,568 | 46.747% |
| P100 | 474,112 | 63.659% |

Sources:

```text
/home/ubuntu/experimental_nonpromotable/
  coherent-n128-active-policy-20260715-r1/
  campaign/run-b412bff/arms/{P10,P25,P50,P100}/train.report.json
```

Thus “P10” was not a 0.10 objective coefficient, and “P100” was not a 1.00
additive coefficient. Batch size changed the actual objective mixture as well
as coverage and estimator variance.

Current code fixes this at `tools/train_bc.py:16979-17142` by retaining
independent sufficient statistics and applying `policy_aux_loss_weight`
explicitly. This rules out the historical hidden-mixture bug in new runs, but
also means old arm rankings are not transferable to the repaired optimizer.

## Finding 3 — draw events still are not an information-dose contract

Current AUX sampling remains with replacement:

```text
tools/train_bc.py:31547-31569
```

and authenticated weighted base sampling also draws with replacement:

```text
tools/train_bc.py:32045-32102
```

The current report is honest about the limitation:

```text
tools/train_bc.py:32155-32187
```

`training_row_draws` are draw events and `unique_training_rows_drawn` is
`None`. Only the AUX path keeps a cumulative unique-row set.

This is a root scientific problem for a short-dose learner. The same number of
draws can mean:

- broad one-pass coverage;
- repeated exposure to a few thousand high-weight roots;
- or a weighted composite that omits many physical rows during an “epoch.”

For a uniform with-replacement sampler, `n` draws from `n` rows cover only
about `63.2%` of rows in expectation. A nonuniform sampler covers less.
Therefore a weighted composite “epoch” is not a corpus pass, and optimizer-step
or draw-event equality does not establish dose equality.

### Smallest production fix

Before enabling the AUX stream in a decisive run:

1. count unique base rows, unique AUX rows, unique games, and maximum reuse;
2. report effective sample size from the realized draw histogram, not only the
   static weight vector;
3. cap AUX reuse in short-dose studies;
4. prefer a shuffled without-replacement active-root cycle until it is
   exhausted, then reshuffle, rather than immediate replacement sampling.

For authenticated component/game mixtures, implement per-component draw quotas
plus shuffled game/root cycles. This preserves the mixture without allowing a
few roots to dominate merely because the requested dose is short.

## Finding 4 — phase allocation and phase loss weights can multiply each other

The authenticated AUX phase allocator sets exact **sampling** shares at:

```text
tools/train_bc.py:31447-31509
```

The sampled AUX rows then receive `policy_sample_weights` at:

```text
tools/train_bc.py:17019-17023
```

and those weights already include `phase_weights`:

```text
tools/train_bc.py:27398-27401
```

If a future composite configures both an AUX phase allocation and non-unit
policy phase weights, the final gradient mass is proportional to:

```text
configured sampling share * configured phase loss weight
```

not the declared sampling share. For example, a 2/3 `PLAY_TURN` AUX allocation
combined with `PLAY_TURN=4` produces 8/9 of pre-normalized AUX objective mass,
not 2/3.

No current v3 scratch recipe enables the AUX stream, so this is not the cause
of the current scratch run. It is a latent production footgun for the planned
active-policy learner.

### Smallest production fix

Fail closed when authenticated AUX phase allocation and non-unit policy phase
weights are both present, unless the contract explicitly declares
`sampling_share_x_loss_weight_v1`. Prefer one authority for final phase
objective mass.

## Finding 5 — current DDP normalization and value/AUX isolation are correct

Two suspected failures are ruled out in the current code.

### DDP denominators

`tools/train_bc.py:24927-25002` all-reduces each loss denominator and scales the
local numerator by world size before DDP averages gradients. This avoids the
biased mean-of-rank-means estimator when masks or row weights differ by rank.

The generic gradient-accumulation path still averages normalized microbatch
means, but decisive A1 runs already reject `grad_accum_steps != 1` at
`tools/train_bc.py:8121-8193`. The current 8xB200 recipe binds accumulation 1.

### Active stream objective isolation

The shared base order is not modified by policy surprise
(`tools/train_bc.py:12212-12218`). AUX forward computes policy CE and, when
enabled, the parent-policy KL anchor. It does not contribute value, final-VP,
Q, belief, or subgoal losses (`tools/train_bc.py:17007-17142`).

Therefore the current active stream does not silently change the value-state
distribution. Historical value drift must be sought in the base sampler,
value weighting, shared-trunk gradients, or target semantics—not in an AUX
value loss that does not exist.

## Production decision

Do not launch another active-policy sweep by varying AUX batch size. The next
decisive learner should first use the repaired direct-corpus scratch path with
no AUX stream. If active correction is reintroduced, commission it with:

```text
explicit coefficient
+ without-replacement/capped-reuse root cycle
+ unique-root/game dose ledger
+ fixed phase authority
+ parent KL and trunk drift
+ held-out whole-game validation
```

The highest-value immediate code change after the scratch launch blockers is
the realized unique-root/game accounting and capped-reuse AUX sampler. It
directly prevents the failure mode that made historical training loss improve
while broad playing strength regressed.
