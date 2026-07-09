# Reanalyze Value Targets for Gen-2 (contingency f67, D4 -- DESIGN ONLY)

**Status:** design doc, no code. Proposes mixing the Monte-Carlo game outcome `z`
with a search-derived root value `V_search` into the value TARGET for gen-2
training. Motivated by the Gate-A finding that the value head is per-position
BIASED at wide roots; a lower-variance, search-informed target can supply a
richer learning signal than the raw ±1 outcome alone.

Provenance (verified against primary sources): MuZero (arXiv:1911.08265) uses the
pure terminal outcome `z=u_T` as the board-game value target and, in *Reanalyze*
(Appendix H), a value-loss weight of **0.25** (vs 1.0 for policy/reward), n=5-step
returns, a target network for the bootstrap, and fresh MCTS re-runs supplying
policy targets for 80% of updates. Gumbel-MuZero / mctx's `v_mix`
(`qtransform_completed_by_mix_value`, Danihelka et al. 2022 Eq. 33) is a
SEARCH-TIME Q-completion estimator, `v_mix = (v_hat + sum_b N_b * weighted_Q) /
(1 + sum_b N_b)` -- this repo already ports it verbatim in
`gumbel_chance_mcts.py::_completed_q`. ReZero (arXiv:2404.16364) makes reanalysis
cheap via backward-view value reuse + entire-buffer periodic reanalysis.
EfficientZero (arXiv:2111.00210) shrinks the bootstrap horizon for staler data.
No paper anneals a single mixing lambda over training time; a training-time
schedule (below) is our own proposal, closest in spirit to EfficientZero's
staleness-indexed horizon.

---

## 1. What already exists in the shards (and what doesn't)

Persisted per decision row (`gumbel_self_play.py`): the improved policy
(`target_policy`), the seed prior (`prior_policy`), per-action Q
(`target_scores` = `result.q_values`), per-action afterstate values
(`afterstate_target`), and the terminal outcome fields (`winner`,
`final_public_vps`, `final_actual_vps`) from which `train_bc._value_targets`
builds the scalar value target `z`. Also `used_full_search`, `simulations_used`,
`is_forced`.

**Not persisted today:** the scalar search ROOT value `V_search =
result.root_value`. It is computed by every full search but dropped on the floor
at row-build time. This is the one missing ingredient for the cheap mixing path.

The repo also already has reanalysis tooling: `tools/generate_reanalysis.py`,
`tools/reanalysis_orchestrator.py`, `tools/generate_rust_mcts_reanalysis.py` --
the full-reanalyze path (section 4) should build on these, not from scratch.

## 2. Target construction

Define the mixed value target per row:

    v_target = lambda * z + (1 - lambda) * V_search_root

- `z`: the existing scalar outcome target from `_value_targets` (see section 5
  for the truncated-vp-margin interaction) -- root-perspective, in [-1, 1].
- `V_search_root`: the search's root value at that state, root-perspective, in
  [-1, 1]. Two ways to get it, cheap vs. faithful:

  **(A) Cheap / persist-and-mix (recommended first step).** Add `root_value` to
  `gumbel_self_play.EXTRA_KEYS` (forward-compatible -- the loader ignores unknown
  columns) and write `float(result.root_value)` for full-search rows (NaN/masked
  for fast-search and forced rows, mirroring `afterstate_target_mask`). Then at
  train time mix against the STORED root value. This costs one fp32 column and a
  one-line generator change; no extra compute. Caveat: the stored `V_search` is
  from the *generation-time* network, so it is a fixed (stale) bootstrap, not a
  fresh reanalysis -- fine as a variance-reduction target, not a true
  policy-iteration bootstrap.

  **(B) Faithful / reanalyze.** Re-run search on stored states with the CURRENT
  (or a target-network) checkpoint to produce a fresh `V_search` (and, for free, a
  fresh `target_policy`). This is MuZero Reanalyze proper and is what the existing
  reanalysis tools do; ReZero's tricks (section 4) make it affordable.

Mask discipline: only mix on rows that HAVE a real `V_search` (full-search rows).
Fast-search / forced / raw-policy rows keep `v_target = z` (lambda effectively 1).
This composes cleanly with the D3 phase-gated arm, which emits `used_full_search =
False` for skipped-search rows.

## 3. Lambda schedule

Bias/variance intuition: early in gen-2 the value net is still the (repaired but
imperfect) gen-1 net, so `V_search` is only as good as that net -- trust the
outcome `z` more (lambda high). As the value net improves across gen-2 epochs/
generations, `V_search` becomes a lower-variance, better-calibrated signal than a
single ±1 outcome -- shift weight toward it (lambda down).

Proposed schedule (a proposal, not from any paper):
- Start `lambda = 0.8` (mostly outcome; matches the spirit of MuZero board-game
  targets which are pure outcome, while still injecting some search signal).
- Anneal linearly to `lambda = 0.5` over the generation (or hold 0.8 for gen-2
  and only lower it in gen-3+ once the value head is demonstrably calibrated).
- Never go below ~0.4: the outcome `z` is the only fully unbiased anchor; letting
  `V_search` dominate risks a value head that agrees with its own (possibly still
  biased) search -- the exact self-confirmation trap the SNR arms exposed.

Safer alternative to a global lambda: EfficientZero-style PER-ROW weighting by
search budget -- rows with more `simulations_used` (already persisted) get a lower
lambda (trust their `V_search` more) than thin fast-search rows. This ties the
mix to actual estimate quality rather than wall-clock training time.

Implement as `train_bc --value-target-lambda L` (default 1.0 = pure outcome =
today's behavior, a pure no-op) reading the `root_value` column; a scheduled
variant can take `--value-target-lambda-final` + linear interpolation over epochs.

## 4. Making reanalysis cheap (ReZero) -- if going with path (B)

- **Entire-buffer periodic reanalysis:** reanalyze the whole gen-1 corpus once
  with the repaired checkpoint, in large batches (ReZero found batch ~2000
  optimal), then train several epochs off those frozen fresh targets -- rather
  than re-searching every minibatch. Acts like a DQN target network and amortizes
  MCTS cost. Fits the existing `reanalysis_orchestrator.py` batch model.
- **Backward-view value reuse:** when reanalyzing a stored trajectory in reverse,
  substitute the already-computed successor root value for the on-trajectory
  child and terminate that simulation early -- skips re-searching subtrees.
  ReZero reports 2-4x wall-time speedups with comparable strength.
- **Reanalyze fraction:** ReZero's default reanalyze_ratio=1 (all targets from
  reanalysis) is aggressive; a periodic frequency (e.g. once per training epoch,
  ReZero ablates {0, 1/5, 1, 2}) is the tunable. Start conservative: reanalyze
  once before gen-2 training, not continuously.

## 5. Interaction with truncated-vp-margin (F3) labels

`--truncated-vp-margin-value-weight` (F3) already replaces the hard ±1 outcome
with a soft VP-margin-derived label for games that hit the decision cap without a
clean winner (a truncated game's "who was ahead" signal). That soft label IS the
`z` term in `v_target = lambda*z + (1-lambda)*V_search`; the reanalyze mix
composes on TOP of F3, it does not replace it. Two consistency requirements:

1. Perspective/sign: `V_search_root` is root-to-move perspective (see
   `_completed_q`'s `sign` handling); `z` from `_value_targets` must be in the
   SAME perspective before mixing. Verify against `_value_targets` sign
   conventions when implementing -- a sign flip here silently poisons the target
   (the exact "fabricated label" bug class the forced-ROLL path was written to
   avoid).
2. Confidence weighting: `_value_targets` already emits `outcome_confidence`
   (down-weights truncated/soft-label rows in the value loss). Keep applying it to
   the MIXED target -- do not let the search value `V_search` smuggle full
   confidence into a row whose outcome was only a truncated-margin estimate.

## 6. Recommended sequencing

1. Land the cheap path first: persist `root_value` (path A), add
   `--value-target-lambda` (default 1.0 no-op), unit-test the mix + masking +
   sign, A/B a fixed lambda=0.8 value-repair-v3 vs pure-outcome on the calibration
   probe and a small H2H.
2. Only if the fixed-lambda mix helps, invest in path (B) reanalysis + the
   schedule. Persisting `root_value` now (even before deciding on B) is a free,
   forward-compatible change that keeps the option open.
