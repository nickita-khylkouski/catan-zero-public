# Adaptive n256: preserve per-particle dose

Status: implementation and fixed-root mechanism validation complete; bounded
same-network strength canary running. This document does not authorize a full
generation wave or promotion.

## Failure mechanism

Under public-information PIMC, the configured root budget is divided across
hidden-world determinizations. The old global settings changed both total
search and per-particle search:

| operator | resolved particle budgets |
|---|---|
| n128, requested P4, min32 | `4 x 32` |
| global n256, requested P4, min32 | `4 x 64` |
| adaptive wide n256, requested P8, min32 | `8 x 32` |

The improved policy uses
`softmax(log_prior + (c_visit + max_visits) * c_scale * minmax(completed_q))`.
The old P4 n256 operator therefore bought more visits in the same four sampled
worlds and simultaneously sharpened the distilled target. The effect depends
on branching factor, so one global `c_scale` cannot repair it.

The sealed fixed-root P4 report showed that global n256 improved top-1
agreement by 7.8 percentage points and reduced width-41+ JS instability by
52.8%, but increased global JS instability by 57% and width-11--20 instability
by 117%. Matching the medium-root sigma with `c_scale=.083076923` reduced the
11--20 regression to 9.7%, establishing visit-dependent sharpening as causal.

Lowering the D6 threshold from 20 to 11 did not remove that mechanism. The
matched report at
`/home/ubuntu/experimental_nonpromotable/r3-fixed-root-search-20260712-r1/d6-t11-report.json`
(SHA-256 `d6f148583ec146765cd5e12de5038ee7c8b5c3cbfe08cd161812cd267a058c29`)
made global JS essentially neutral (`0.44%` reduction) but n256 still increased
width-11--20 JS by `66.7%`. D6 reduces evaluator-orientation noise; it does not
decouple budget from sigma.

## Corrected operator

Use global n128 and spend n256 only at roots with at least 40 legal actions:

- `n_full=128`;
- `n_full_wide=256`;
- `n_full_wide_threshold=40`;
- `wide_roots_always_full=true`;
- `determinization_particles=8`;
- `determinization_min_simulations=32`.

This resolves ordinary n128 roots to P4x32 and adaptive wide roots to P8x32.
The existing sealed adaptive fixed-root probe reduced width-40+ instability by
53% while keeping per-particle dose constant. Its panel contained only opening
settlements, so it is mechanism evidence, not a general playing-strength gate.

`information_set_particle_budgets` is now the single runtime/launcher budget
calculation. The direct generator and continuous flywheel reject adaptive PIMC
recipes whose base and wide per-particle doses differ. The new guard
`configs/guards/a1_generation_adaptive_n256_wide40.json` binds every adaptive
field and refuses the old P4x64 configuration.

## Training recipe after the operator canary

Additional search data is useful only if the learner comparison is clean:

1. Independently initialize every arm from the exact f7 producer bytes.
2. Use one fixed 4,194,304-sample dose; never chain candidates.
3. Globally shuffle current n128 plus adaptive-wide n256 and authenticated 20%
   incumbent-era replay.
4. Keep loser policy weight 1.0 and forced policy weight 0. Forced rows remain
   value rows.
5. Compare policy-KL anchors K0/K3/K10 at the same sample dose and select using
   the external panel, not imitation validation alone.
6. Evaluate against the actual initializer with identical operator fields.

The next binding evidence is a same-checkpoint n128-vs-adaptive-n256 paired
canary, followed by a producer-started one-dose student comparison if search
strength is non-regressing. Only the student gate can establish that additional
lookahead improved the learned network.
