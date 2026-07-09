# Value-Uncertainty Head + Uncertainty-Weighted Search (contingency f67, D2)

**Status:** the auxiliary HEAD and its training loss are IMPLEMENTED and tested
in this branch (default-off, no-op). The SEARCH-side consumption below is a
DESIGN, not yet wired -- deliberately, because it changes the search's backup
accounting and must clear its own H2H A/B before it ships.

Provenance note: KataGo details verified against `docs/KataGoMethods.md` and the
KataGo C++ source (sections "Dynamic Variance-Scaled cPUCT", "Short-term Value
and Score Targets", "Uncertainty-Weighted MCTS Playouts"). The ~75 Elo KataGo
reports is for the COMBINATION of dynamic cPUCT + uncertainty-weighted playouts,
not uncertainty weighting alone (~25 Elo of it is recoverable by better static
cPUCT tuning). The noise-floor shrinkage formula in section 3 is OUR OWN
construction (classical reliability / Kalman-gain shrinkage), analogous to but
distinct from KataGo's empirical-Bayes cPUCT blend -- do not cite it as
"KataGo's".

---

## 1. The auxiliary head (IMPLEMENTED this branch)

`EntityGraphConfig.value_uncertainty_head: bool = False` gates an optional head
shaped like `value_head` (`Linear(h,h) -> GELU -> Dropout -> Linear(h,1)`),
emitting `outputs["value_uncertainty"]` through a `softplus` (forces a
non-negative predicted-squared-error). Default False => the head is not built,
`"value_uncertainty"` is absent from outputs, parameter count and forward are
bit-identical to before, and old checkpoints load unchanged (the reverse
direction -- a head-bearing model init'd from a pre-head seed -- is covered by
adding `value_uncertainty_head.` to `EntityGraphNet.load`'s allowed-missing
prefixes). Enable on a fresh model via `EntityGraphPolicy.create(...,
value_uncertainty_head=True)` or `EntityGraphConfig(value_uncertainty_head=True)`.

### Training target (KataGo short-term-error style)
`train_bc.py --value-uncertainty-loss-weight W` (default 0.0 = no-op) adds

    loss += W * Huber( uncertainty_pred , stop_grad[(z - v)^2] )   over value rows

where `z` = the value outcome target (`value_outcome_targets`), `v` = the value
head output, and the target is the value head's OWN squared error with a
**stop-gradient on `v`** so the uncertainty loss trains only the uncertainty head
and never distorts the value head (this stop-grad placement is exactly KataGo's:
"do not attempt to adjust the value head to make the squared difference closer to
the error head's output"). Huber (smooth_l1) rather than MSE because the target
is already a squared quantity, so plain MSE would be 4th-order in value error and
outlier-dominated -- KataGo uses Huber for the same reason. Masked/weighted like
the value loss (`value_has_outcome`, `value_weights * outcome_confidence`).

The head can therefore be trained during ANY retrain (including the D1
unfreeze-with-KL run) by adding `--value-uncertainty-loss-weight 0.25` and
building the model with the head enabled. It costs one extra small head and one
extra masked-Huber term; it does not touch the value/policy/vp losses.

### Remaining wiring (one-line, left for the operator's call)
The retrain inits from a seed checkpoint whose saved config has the head off, so
enabling it on that run needs the loaded `EntityGraphConfig` to be overridden to
`value_uncertainty_head=True` at build time. The architecture preflight
(`_checkpoint_config_mismatches`) does NOT check this field, and the loader
tolerates the missing head weights, so the override is safe -- it just isn't
exposed as a `train_bc` CLI flag yet (kept out of the D1 LAND-READY diff to avoid
coupling two experiments). Adding `--value-uncertainty-head` that
`dataclasses.replace(config, value_uncertainty_head=True)` on the loaded config is
the whole change.

---

## 2. Uncertainty-weighted search (DESIGN -- KataGo playout weighting)

KataGo ships (verified in `cpp/search/searchupdatehelpers.cpp`):

    weight = uncertaintyCoeff / ( uncertainty^uncertaintyExponent + uncertaintyCoeff/uncertaintyMaxWeight )

with shipped constants `uncertaintyCoeff=0.25`, `uncertaintyExponent=1.0`,
`uncertaintyMaxWeight=8.0`, and `uncertainty` = a utility-scaled short-term value
error. As `uncertainty -> 0` the weight saturates at `uncertaintyMaxWeight` (the
divide-by-zero floor is `uncertaintyCoeff/uncertaintyMaxWeight`); terminal nodes
get max weight (fully certain). This `weight` replaces the "+1 visit" in BOTH the
Q running-average AND the visit-count terms -- KataGo's tree is weight-based, not
count-based, precisely to support this.

### Port for our Gumbel-chance MCTS
Our tree is currently count-based (`_GAction.visits`, `value_sum`, `q =
value_sum/visits`; backups in `_simulate`/`_backup`). A faithful port would:

1. Add `weight_sum: float` alongside `visits` on `_GAction`/`_GNode`; back up
   `weight * leaf_value` into `value_sum` and `weight` into `weight_sum`; redefine
   `q = value_sum / weight_sum`.
2. At each leaf evaluation, compute `sigma_eval = sqrt(value_uncertainty(leaf))`
   (the head predicts (z-v)^2, so sqrt is the error/stdev scale) and
   `weight = c / (sigma_eval + c/w_max)` with `c=0.25`, `w_max=8.0` as a starting
   point (KataGo's own numbers; re-tune for Catan's value scale).
3. Leave the Gumbel-Top-k + Sequential Halving BUDGET (number of leaf visits)
   unchanged -- only the WEIGHT each leaf contributes changes. Sequential
   Halving's elimination still uses the (now weighted) Q.

**Scope warning:** this touches the hot backup path and the completed-Q /
`_sigma_scale` accounting (which reads `stats.visits` and `max_visits`). It is a
real change to the improvement operator and MUST be flag-gated (default off) and
cleared by its own strength H2H A/B before generation. It is NOT part of the
LAND-READY D1/D3 set.

Diagnostic-only alternative (cheaper, no backup rewrite): expose the leaf
`sigma_eval` as telemetry in `sigma_trace_placement_root.py` first, to confirm
the head actually predicts HIGHER uncertainty at the wide placement roots that
lose -- if it doesn't, the whole mechanism is moot and we save the backup rewrite.

---

## 3. Noise-floor rescale attenuation (DESIGN -- the cheapest, most targeted fix)

This is the most direct antidote to the verified Gate-A mechanism and needs NO
backup rewrite. Recap of the mechanism (from
`catan_postrepair_revalidation_protocol_20260704.md`): `_rescale_completed_q`
min-max-rescales the completed-Q spread to fill [0,1] regardless of whether that
spread is genuine signal or 1-2-sample noise; then `_improved_policy` adds
`scale * rescaled_q` to the prior logits. At a 54-wide placement root with
n_full=64 (~1.2 sims/candidate) the spread is frequently pure sampling noise, but
the rescale manufactures full [0,1] confidence from it and overrides a near-flat
prior on 57-72% of decisions.

Attenuate the rescaled-Q contribution by how much of the observed spread is
plausibly real vs. evaluation noise. In `_improved_policy`, replace

    scores[a] = logits[a] + scale * rescaled_q[a]

with

    scores[a] = logits[a] + scale * alpha * rescaled_q[a]

    raw_spread = max(completed_q) - min(completed_q)              # PRE-rescale
    noise      = c_noise * sigma_eval / sqrt(max(mean_visits, 1)) # per-candidate SEM
    alpha      = raw_spread / (raw_spread + noise)                # in (0,1]

where `sigma_eval` is the root's predicted value stdev (`sqrt(value_uncertainty)`
at the root; or a fixed constant tuned from the calibration probe's residual std
if the head isn't trained yet -- the head is NOT required for this arm),
`mean_visits` = mean candidate visit count, `c_noise` a single tunable (~1.0
start). Behavior: when the real Q spread dwarfs the per-candidate standard error
(`raw_spread >> noise`) `alpha -> 1` (trust the search); when the spread is on the
order of the sampling noise `alpha -> 0.5` and below (shrink toward the prior).
This is the James-Stein/Kalman-gain reliability coefficient
`signal/(signal+noise)`; it is provably a no-op when `sigma_eval=0` or when the
spread is large, and it specifically kills the "1.2-sample noise stretched to
full confidence" failure without weakening genuine deep-search signal at narrow
mid-game roots (where `raw_spread` is real and `mean_visits` is high, so `noise`
is small and `alpha ~ 1`).

Flag-gate as `noise_floor_c: float | None = None` on `GumbelChanceMCTSConfig`
(default None = disabled, `alpha=1` everywhere, pure no-op) plus an h2h
`--noise-floor-c`. This is the recommended FIRST uncertainty-related arm to try:
one config field, one multiply, no backup rewrite, no trained head required, and
it targets the measured mechanism head-on. It is complementary to D3's
`n_full_wide` (more samples at wide roots shrinks `noise` directly) -- the two can
be A/B'd together.

Note (2026-07-04): the team's task #67 ("noise-floor rescale attenuation") and
#68 ("variance-aware completed-Q") track exactly these two ideas -- this doc is
the design writeup they can build the flag-gated implementations from.
