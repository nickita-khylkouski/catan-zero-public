# f69 design: root-Q-in-one-forward search integration

Status: design only (no implementation in this branch). Scope: how the
`entity_graph` `q_head` (root candidate Q-values obtained in the single root
forward) plugs into `GumbelChanceMCTS._run_root_search`.

## The asset

`EntityGraphNet.forward(..., return_q=True)` returns, in **one forward pass**,
`value` plus a per-candidate `q_values` vector `Q(s, a_i)` for *every* legal
action `a_i` at the state. In the f69 upgrade these Q-values additionally see
the board through the target-gather and cross-attention paths, and are computed
as `q_head([state, action, state*action])` with `state` **shared across all
candidates**. The root evaluation (`_expand` -> `evaluator.evaluate`) already
runs this forward for priors+value; requesting `return_q` there yields root-Q at
essentially zero marginal cost (one extra MLP head on an already-batched
forward).

## Why relative root-Q is trustworthy where absolute value is not

The sigma-trace diagnosis (`tools/sigma_trace_placement_root.py`) showed the
failure at wide roots is **noise, not signal**: with ~1.2 simulations per
candidate, `_completed_q` sees a raw-Q spread that is pure sampling noise, and
`_rescale_completed_q` stretches it to fill `[0,1]`, manufacturing false
confidence that swamps a near-flat prior.

Root-Q attacks this because all candidates share the identical `state`
embedding. Any additive state-level value bias `b(s)` cancels in a **relative**
comparison `Q(s,a_i) - Q(s,a_j)` — the exact bias that dominates the *absolute*
value estimate does not enter candidate *ranking*. (Cancellation is exact only
for the additive part; the `state*action` interaction term in `q_head` means it
is approximate, which is worth stating plainly — but the dominant shared-state
component still cancels.) Measure the realised root-Q candidate spread with
`tools/f69_ranking_probe.py`; on the current pre-finetune checkpoint the q
spread is tiny (mean range ~0.037 over 54 candidates), so this integration is
predicated on the finetune actually widening it — the probe is the gate.

## Integration with Gumbel-Top-k + Sequential Halving

Today (`_run_root_search`, lines 453-492): choose `m` candidates by
`argtop_m(gumbel + logit)` (prior only), then Sequential Halving refines them
using `completed_q` from real `_simulate` calls. At a 54-wide root with
`n_full ~ 64`, `m` is capped by `max_root_candidates_wide` and the budget still
spreads to ~1 sim/candidate.

Proposed change — **root-Q as the pruner into Sequential Halving**:

1. **Candidate selection uses root-Q.** Replace the initial top-k key with
   `gumbel + logit + sigma_root * root_q`, where `root_q` is the (rescaled)
   per-candidate root-Q from the root forward and `sigma_root` is a small,
   separately-tuned scale (not the visit-driven `_sigma_scale`, since there are
   zero visits yet). This is the Gumbel-Top-k selection of mctx but with a prior
   *and* a value estimate, at no extra eval cost. It lets a poor-prior /
   good-Q candidate survive and a good-prior / bad-Q candidate get pruned before
   any budget is spent.
2. **Sequential Halving still uses true child lookahead.** The `m` survivors are
   refined by real `_simulate` calls exactly as today (`_completed_q` ->
   `_rescale_completed_q` -> halving). Root-Q is a *pruner and prior*, never the
   final arbiter: the decision among survivors comes from actual child
   evaluation, so a systematically wrong root-Q on the winner is corrected by
   lookahead. This is the fallback that keeps the change safe.
3. **Optionally shrink `m` on wide roots.** Because root-Q gives a first
   value-aware ranking for free, `max_root_candidates_wide` can be cut (e.g.
   54 -> 8..16) with the freed budget spent as *depth* on the survivors.

## Expected eval-count savings at a 54-wide root

Let `k` = simulations per surviving candidate needed for real SH signal, and let
`C` be the mean child evaluations per simulation (with lazy interior-chance
evaluation, `C` is small; without it, up to the enumerated-chance fan-out). The
first SH round dominates the wide-root cost.

- **Today:** to give all 54 candidates `k` sims each costs `~ 54 * k * C` evals.
  At the shipped budget that is `k ~ 1`, i.e. noise.
- **Root-Q pruned to `m = 8`:** `~ 8 * k * C` evals for the same `k`.
  - At **equal total budget**, per-candidate depth rises `54/8 ~ 6.75x`
    (`k: 1 -> ~7`), turning the noise-dominated first round into real signal —
    the direct fix for the sigma-trace failure.
  - At **equal per-candidate depth `k`**, first-round evals drop
    `~ (54 - 8)/54 ~ 85%`.
- The root forward itself does **not** grow: `return_q` is one head on the
  existing batched root forward. The pruning is free; the win is either fewer
  evals or deeper survivors.

`m = 8` is illustrative; sweep `m in {6,8,12,16}` and `sigma_root` and gate on
(a) agreement of the pruned survivor set with the unpruned SH winner on a
held-out opening panel, and (b) end-to-end strength (the G2 >=55%/1000 gate).

## Risks / guards to implement alongside

- **Untrained/miscalibrated root-Q prunes the true best move.** Guard: keep a
  prior-only floor in the top-k (always include the top-`p` by `logit` in the
  survivor set regardless of root-Q), and gate rollout on the probe showing the
  finetuned q spread is real, not noise.
- **Interaction-term bias.** The `state*action` term breaks exact cancellation;
  keep root-Q as a *selection prior*, never the final decision (item 2).
- **Consistency with self-play targets.** If root-Q reshapes which candidates
  are visited, the improved-policy target (`_improved_policy`) still derives from
  real visits/`completed_q`, so the training target stays lookahead-grounded;
  document that root-Q affects *which* candidates get lookahead, not the target
  construction.
- **Symmetry synergy.** Averaging root-Q over the 12 board orientations
  (see `f69_hex_symmetry_augmentation.md`) denoises the pruning key before it is
  used, and is the recommended companion at wide roots.
