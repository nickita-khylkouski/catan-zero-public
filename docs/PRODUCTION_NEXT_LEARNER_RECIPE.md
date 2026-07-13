# Production-next learner recipe

This is the high-level flywheel default for the next iteration.  Historical
`train_bc` CLI defaults remain unchanged so old receipts are reproducible.

## What the current corpora actually contain

Measured directly from the authenticated memmaps on 2026-07-12:

| corpus | rows | forced rows | active policy rows | root-value rows | forced policy mass |
|---|---:|---:|---:|---:|---:|
| n128 | 31,919,276 | 51.537% | 12.115% | 12.115% | 0 |
| n256 | 12,773,247 | 51.582% | 12.113% | 12.113% | 0 |

`policy_weight_multiplier` is zero for every forced row and one only for a
non-forced full-search row.  `build_sample_weights` multiplies by that column
before its final mean normalization; therefore forced rows contribute exactly
zero to both the policy-loss numerator and denominator.  `forced_action_weight`
was already redundant for these corpora, but the new launcher pins it to zero so
the intended invariant is explicit.

Forced rows are still useful state/value observations. They currently receive
value loss, but the production default does **not** guess a new forced-value
coefficient. It uses `1.0`; `{1.0, 0.25}` is a matched follow-up experiment.
The authenticated composite-v2 sampler already draws component → game → row,
so every game has equal expected value-target sampling mass before loss
weights. A second per-game value normalization would double-correct and favor
short games, so that transform stays off. Policy targets occur only on a sparse
subset of rows; their separate sqrt per-game mass correction remains enabled.

## Extracting search information without self-distillation

The expensive search is not discarded: `target_policy`, the search-improved
distribution, is the primary policy target on every active full-search row.
The scalar `root_value` is also stored on exactly those rows.  It is not yet
mixed into the value target because it was produced by the generating network
and is a stale bootstrap.  `value_target_lambda=1` keeps the unbiased realized
outcome target until a refreshed/lagged-root experiment wins.  Likewise,
`q_loss_weight=0` remains required until the Q head and target have the same
semantics.  Current Gumbel shards store raw return-scale visit Q for visited
actions only, but the learner's optional Q objective row-standardizes those
targets; that would train relative z-scores into a head other consumers treat as
return-scale Q.  Other teacher families may store preference scores under the
same generic column, so source provenance remains mandatory.

The forced-ROLL afterstate column was stored in NPZ shards but omitted by both
training loader paths, along with `simulations_used`; existing n128/n256 memmaps
therefore do not contain it.  Schema v2 now preserves both columns in future
NPZ and memmap loads, but the canonical loss still does not consume them.  A
superseded implementation blended the played-action Q/afterstate into the value
target.  It was intentionally not merged because it silently changed the blend
direction and introduced a self-estimated target.  Reintroduce that signal only
as an authenticated, matched refreshed-target arm.

## Default iteration recipe

- authenticated game-uniform current + replay composite; refuse current-only training;
- forced policy weight `0`;
- sqrt per-game policy weighting for sparse active rows; no additional per-game
  value loss weighting (the sampler is already game-uniform);
- loser rows retained at weight `1`;
- forced value weight `1` pending the `{1,.25}` result;
- outcome-only scalar value target (`lambda=1`);
- Q loss off;
- replay required by default; a positive policy-KL anchor is an explicit
  alternative rather than an unmeasured constant.

This addresses the proximal failure—large fresh-Adam drift on one correlated
distribution—without claiming that stale search values are ground truth.
