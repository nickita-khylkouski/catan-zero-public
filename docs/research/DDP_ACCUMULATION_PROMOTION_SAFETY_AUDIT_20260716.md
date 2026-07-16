# DDP accumulation promotion-safety audit (2026-07-16)

## Finding

The reviewed two-rank fallback declares `batch_size=512`, `world_size=2`, and
`grad_accum_steps=4` so its nominal global batch is 4096. That matches the
eight-rank B200 row count, but it does **not** currently match its conditional
policy/value objective.

`train_bc` normalizes every policy, value, KL, and auxiliary objective inside
each microbatch, then divides the resulting loss by `grad_accum_steps`. For
conditional objectives this computes a mean of microbatch means. The exact
4096-row objective is instead one union-weighted numerator divided by the union
of eligible weight mass. Those estimators differ whenever eligible mass is not
identical in all microbatches.

For four-way accumulation, if one microbatch contains a sparse label and the
other three contain none, the implemented diagnostic operator produces
`(g + 0 + 0 + 0) / 4 = 0.25g`; the union-weighted conditional objective produces
`g`. That is 75% signal attenuation. More generally, unequal nonzero masses can
change gradient direction, not only magnitude.

The trainer already described this honestly as
`diagnostic_approximate_microbatch_means`, but two gaps remained:

1. a topology-authorized dual report inherited the corpus's promotable flags;
2. the promotion verifier accepted `{world_size: 2, grad_accum_steps: 4}` as a
   global-batch-equivalent candidate without requiring the typed accumulation
   semantics to be exact.

## Historical B200 scope

Ten archived learner reports in the 2026-07-15 evidence bundle were inspected.
All ten used `world_size=8`, `batch_size=512`, `grad_accum_steps=1`, and 128
optimizer steps. Their declared accumulation operator is
`single_microbatch_exact`, so this finding does not explain or invalidate those
recorded B200 trajectories. It is a next-retrain/fallback safety issue.

## Resolution

- Training-report eligibility is now derived from the typed semantics. Any
  missing, malformed, or approximate declared accumulation contract is forced
  to `diagnostic_only=true` and `promotion_eligible=false`.
- Dual output verification binds the declared world size, accumulation count,
  and exact/approximate contract to the reviewed effective recipe.
- Promotion requires `single_microbatch_exact`, `grad_accum_steps=1`, and a
  matching top-level report value. The check is contract-based, not hardcoded
  to the current two-rank spelling, so future approximate topologies also fail
  closed.
- Bounded two-rank diagnostics remain runnable. Their checkpoints cannot enter
  promotion until exact union-weighted gradient accumulation is implemented.
