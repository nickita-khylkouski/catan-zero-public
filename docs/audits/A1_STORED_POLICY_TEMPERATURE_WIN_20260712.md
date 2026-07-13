# A1 stored-policy calibration result (2026-07-12)

## Outcome

Per-source stored-policy temperature calibration produced the first decisive
fresh-data learner win over the f7 incumbent in this campaign.

- Initializer: f7
  `sha256:f7e93dfb8cdb713d647b3e142c949d59083de9f719b6688b6faa6c918ce3eed4`
- Candidate:
  `sha256:fefba044df58b9508de751d76d09bedeb630a2e832f6db46b70d95b5d4c77394`
- Training dose: `1024 * 8 * 512 = 4,194,304` sampled rows
- Optimizer: fresh Adam, LR `3e-5`, 100-step warmup, no candidate chaining
- DDP RNG: rank-offset enabled after identical model construction
- Component game-sampling ratios: n128 `4/7`, n256 `1.6/7`, replay `1.4/7`
- Stored-policy temperatures: n128 `1.0`, n256 `1.11`, replay `0.52`

The exact64 paired evaluation used 600 seat-swapped seed pairs (1,200 games),
n128 search on both roles, identical evaluator/search settings, public
observation, and the same tanh scalar-value readout.

| Metric | Result |
|---|---:|
| Candidate wins | 670 |
| f7 wins | 530 |
| Candidate score | 55.833% |
| Complete pairs | 600 |
| Truncations / errors | 0 / 0 |
| Ordinary SPRT | H1, LLR 9.453 |
| Pentanomial SPRT | H1, LLR 11.113 |
| Superiority pentanomial SPRT | H1, LLR 5.788 |

All three LLRs crossed their H1 boundary (`+2.944`). The pooled result is on
the B200 controller under:

`experimental_nonpromotable/n256-temperature-student-f7-dose4194304-20260713-r1/TEMP_ALL/eval600-v-f7/collected/a1-eval-f06986f9fb5b85c9/pooled/internal.json`

## Interpretation

The n128 and n256 stored policies had materially different entropy. Mixing
their raw probabilities under one cross-entropy objective made search budget
change both teacher strength and target sharpness. The learner therefore saw a
source-dependent label scale, not merely a stronger teacher. Exact per-source
temperature mapping removes that confound without modifying the stored corpus.

This result also corrects two earlier experimental validity problems:

1. every arm reloads f7 independently instead of chaining candidates; and
2. every arm receives one fixed 4.19M-sample dose instead of accumulating
   oversized lineage updates.

Offline loss alone was not used to select the winner because changing target
temperature changes the objective being measured. The paired playing-strength
panel is the decision evidence.

## Rejected alternatives

- CAT-100 AUX2 versus matched AUX0: `585-615` over 1,200 games (48.75%),
  pentanomial H0. The existing generic auxiliary bundle is rejected.
- Same-f7 scalar `clip` versus `tanh`: `565-635` (47.083%), pentanomial H0.
  Search-time tanh remains the incumbent readout.
- Earlier candidate-chained/oversized n128+n256 runs are not independent
  evidence about the fresh-data recipe.

## Promotion status and next tests

This checkpoint is diagnostic-only and must not be relabelled as promotable.
A promotion candidate requires an immutable production replication of the
same winning recipe, followed by the normal internal and external gates.

Two clean follow-ups run independently from f7:

1. replay as value rehearsal plus a small forward-KL behavior anchor, with no
   obsolete replay policy distillation; and
2. a zero-output topology residual plus action-target gather, using the exact
   winning calibrated learner recipe.

The next data wave should separately test belief-level completed-Q aggregation
plus D1 normalization. Existing n128/n256 corpora used mean-of-improved
per-determinization policies; fixed-root probes show the belief-level operator
is much more stable, but it is not authorized as a production gameplay or data
default until its corrected evaluator passes.
