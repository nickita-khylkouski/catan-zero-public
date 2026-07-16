# A1 sampler/objective root audit — 2026-07-16

## Scope

This is a read-only audit of the current learner plus historical reports on
`ubuntu@149.118.65.110`. No GPU job was started and no remote artifact was
changed.

The main conclusion is that several historical “active-policy dose” results
were not measuring the treatment their names implied. The old learner coupled
the active-stream batch size, selected-root sparsity, and row-weight scale into
one hidden objective coefficient. The current learner has repaired the
base/AUX normalization and DDP denominator and now implements the missing
weighted-cycle reuse contract and single phase authority. Historical results
remain invalid for selecting the repaired recipe.

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

## Finding 3 — weighted cycles now bind historical coverage/reuse ambiguity

The historical AUX sampler drew with replacement, and the authenticated base
weighted-replacement path remains available for recipes that explicitly bind
that legacy measure. Draw counts from those runs were not information-dose
contracts:

The historical behavior remains available as
`POLICY_AUX_SAMPLING_LEGACY_REPLACEMENT_V1`.

Authenticated legacy base sampling also draws with replacement:

The corresponding base behavior is reported as
`weighted_replacement_v1`.

Those historical reports were honest about the limitation:

The report identifies draw-event semantics independently from unique coverage.

`training_row_draws` were draw events and `unique_training_rows_drawn` could be
`None`. Only the AUX path retained a cumulative unique-row set.

This was a root scientific problem for a short-dose learner. The same number of
draws can mean:

- broad one-pass coverage;
- repeated exposure to a few thousand high-weight roots;
- or a weighted composite that omits many physical rows during an “epoch.”

For a uniform with-replacement sampler, `n` draws from `n` rows cover only
about `63.2%` of rows in expectation. A nonuniform sampler covers less.
Therefore a weighted composite “epoch” is not a corpus pass, and optimizer-step
or draw-event equality does not establish dose equality.

### Implemented repair

The AUX path now supports `weighted_without_replacement_cycles_v1`:

- every positive-mass source row appears at most once before a cycle is
  exhausted;
- Gumbel-top-k/Plackett-Luce ordering preserves weighted priority inside each
  cycle without permitting an early duplicate;
- DDP ranks consume a deterministic rank-strided global stream;
- the cumulative global draw offset is checkpointed and required on resume;
- reports bind eligible rows, cycle boundaries, effective sample size,
  cumulative maximum reuse, realized unique source rows/games, and reuse
  percentiles.

This resolves the root omission. A decisive AUX recipe must bind the weighted
cycle mode; the legacy replacement mode remains explicit for reproduction and
must not be interpreted as equivalent.

### Current scratch coverage correction

The authenticated current scratch split contains 15,968,808 rows. Its target
component measure is 64/12/4/20, while raw row shares are
73.18/7.10/2.30/17.42. The historical weighted replacement sampler therefore
cannot be replaced by a naive unweighted permutation without silently changing
the training objective.

Under the historical replacement sampler, expected aggregate unique-row
coverage is only 60.85% after one nominal epoch and 92.46% after three. Roughly
1,204,068 training rows remain unseen despite 47.9 million draw events.

The current scratch recipe now binds `coverage_importance_v1`:

- traverse one seeded global permutation, giving every training row exposure
  once per epoch before deterministic DDP padding;
- retain the authenticated component→game→row probability `p_i`;
- multiply train-only policy and value weights by `N * p_i`;
- keep validation on its natural holdout measure;
- report the self-normalized minibatch importance estimator explicitly.

This removes the million-row blind spot without reverting source exposure to
raw row proportions. The production schedule remains blocked pending current-v5
data and optimizer commissioning.

## Finding 4 — AUX allocation is now the sole phase authority

Historically, the authenticated AUX phase allocator set exact **sampling**
shares, while the sampled AUX rows then received `policy_sample_weights` that
already included ordinary `phase_weights`.

If a future composite configures both an AUX phase allocation and non-unit
policy phase weights, the final gradient mass is proportional to:

```text
configured sampling share * configured phase loss weight
```

not the declared sampling share. For example, a 2/3 `PLAY_TURN` AUX allocation
combined with `PLAY_TURN=4` produces 8/9 of pre-normalized AUX objective mass,
not 2/3.

No v3 scratch recipe enabled the AUX stream, so this was not the cause of that
scratch run. The footgun is now repaired: when authenticated AUX phase
allocation is present, `_policy_aux_loss_weights_without_phase_multiplication`
removes the ordinary policy phase multiplier from the AUX loss. The base policy
objective retains its phase repair, while AUX gradient mass is governed only
by the authenticated sampling allocation. Reports bind this as
`allocation_only_remove_duplicate_loss_multiplier_v1`, and non-positive phase
weights fail closed because they cannot be inverted safely.

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
decisive learner should first use the repaired direct-corpus path with no AUX
stream. If active correction is reintroduced, commission the implemented
controls together:

```text
explicit coefficient
+ weighted without-replacement root cycle
+ unique-root/game dose ledger
+ fixed phase authority
+ parent KL and trunk drift
+ held-out whole-game validation
```

These controls are implemented. The remaining scientific task is to commission
their combined setting against the direct-corpus baseline; it is no longer a
missing sampler implementation.
