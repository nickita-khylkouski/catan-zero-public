# Self-play SNR: independent evidence-bundle review (updated for July-16 main)

**Author:** Michael (external review, with Claude-assisted analysis)
**Date:** 2026-07-16
**Evidence:** the `catan_rl_evidence_bundle_20260715` artifact set (sealed panel reports, July-15 sweep receipts, 5 production NPZ shards; SHA-manifested).
**Updated against:** main as of `900738c` (2026-07-16), including `A1_TRAINING_SIGNAL_ROOT_AUDIT_20260716.md`, `A1_HISTORY_VALUE_ARCHITECTURE_AUDIT_20260716.md`, adapter-v4 public rule state, and the prior-temperature ownership fix.

## Purpose

An independent measurement of *why self-play stopped compounding*, done on the July-15 evidence bundle before the July-16 repairs landed. Several conclusions were independently reached and shipped on main within the same day (noted below), which is corroboration, not redundancy: the measurements here are from production shards and sealed receipts, not from code reading. Two findings are not yet covered by the in-repo audits and are proposed as standing telemetry.

## 1. Strength state (sealed receipts, July 15)

From the bundle's pooled reports (paired seeds, seat-swapped):

| Matchup | Result |
|---|---|
| four coherent learner arms vs **f7** | 52.7–59.0% (some H1) |
| the same arms vs **v5** (actual incumbent) | 46.5–51.6% (all `continue`) |
| step-24 recovery arm vs f7 / vs v5 | 52.3% / 49.0% |

Every candidate beats the old baseline and ties the incumbent: the loop was recovering to v5, not compounding past it. This is the quantitative form of "self-play stopped improving."

## 2. Shard-level measurements (5 production shards, 17 games, 1,026 full-search decisions)

### 2.1 Value-outcome correlation by game stage

Correlation of stored root value with actual game outcome:

| Decision index | n | corr(root value, z) |
|---|---:|---:|
| < 30 | 189 | **0.32** |
| 30–80 | 177 | **0.24** |
| 80–200 | 432 | 0.67 |
| > 200 | 228 | 0.81 |

This independently corroborates the July-16 root audit's opening-value finding (corr ≈ 0.15–0.19 at opening/wide roots) and extends it: the value function is weakly informative through the entire early development phase (~first 80 decisions), not just the opening placements. In a 2p race largely decided by early engine-building, the leaf evaluator carries little information exactly where search needs it most.

### 2.2 Root Q-spread sits at the value-noise floor

Across the 1,026 full-search roots (n=128, D6-averaged, median legal width 12, ~10 visits/action): the median completed-Q spread between best and worst legal action is **0.044** on [−1,1] — at the project's own measured post-D6 orientation-noise level (σ ≈ 0.049). The preferences search is being asked to certify are the same magnitude as the evaluator's noise. This measurement is independent of the stored prior and unaffected by the prior-temperature bug.

### 2.3 The policy-improvement targets are mostly below the noise floor

Search changed the prior's argmax on 50.7% of policy-active rows. Among those flips, the completed-Q margin between search's choice and the prior's choice was:

- median **0.015**;
- **76% under 0.05** (≈ the noise floor);
- 88% under 0.10.

Interpretation: the majority of what the current improvement operator "teaches" the policy is evaluator noise being re-distilled, which moves the policy sideways — consistent with §1's shuffle around v5, and with the root audit's conclusion that more simulations cannot rescue a low-information leaf evaluator.

**Caveat:** the July-16 `Apply prior temperature exactly once` fix implies the stored `prior_policy` in these pre-fix shards may be temperature-distorted, which would inflate the flip *rate* (50.7%). The margin *distribution* is much less sensitive (it conditions on a flip and measures Q, not the prior), and §2.2 is fully prior-independent. Re-measure on post-fix shards before quoting the 50.7%.

### 2.4 Proposed standing telemetry

Two per-wave health metrics fall out of this analysis and are cheap to compute in the generator or shard QA:

1. **Target-flip margin profile:** among full-search rows where the improved policy's argmax differs from the prior's, the distribution of completed-Q margins, reported against the current measured value-noise floor. A wave whose flips are predominantly sub-floor is generating rehearsal data, not improvement data — knowable *before* spending a learner dose and a gate on it.
2. **Stage-sliced value-outcome correlation** (as §2.1) on a game-level holdout — this generalizes the opening-panel calibration to the full game and complements the planned turn-boundary/evaluator-query holdout from the root audit's blocker A.

## 3. Reconciliation with July-16 main (what already landed)

Independently proposed from this analysis and found already shipped on main within the day — listed so the remaining asks in §4 are clearly *not* covered:

- **Adapter-v4 public rule state** (global slots 8–16): `has_played_development_card_in_turn`, `is_road_building`, `free_roads_available`, `current_discard_count` — the state-aliasing gaps this review's input audit flagged (the featurizer previously dropped the played-dev flag the env emitted).
- **`final_vp_loss_weight: 0.05`** in the v3 contract — dense early-game value gradient, previously 0.0.
- **Production-pips semantics fix** (live-Python probabilities vs Rust integer pips silently zeroing production features).
- **`tools/strategic_root_exam.py`** — human-readable fixed-root exam rendering with counterfactual support.

## 4. Remaining recommendations (not yet on main)

1. **`aux_subgoal_loss_weight` is still 0.0** while every production shard already carries the targets (`aux_vp_in_n`, `aux_longest_road`, `aux_largest_army`, ...). Given §2.1, VP-in-N is the most direct dense supervision for early-game value. One 524k-dose arm vs control.
2. **Informative-outcome data (H8).** Self vs self at parity yields maximally uninformative outcomes precisely when the value function is the bottleneck. The opponent-mix/exploiter machinery exists; mixing external-bot seats and prior champions into generation creates positions where errors are punished, i.e., outcomes with gradient. Worth one source-mix arm before the next big wave.
3. **Search-budget reallocation.** In 2p no-trade, resource hands are exactly deducible; genuinely hidden state is dev-card identities only. Any belief machinery (particles, boundary sampling) should be budgeted against dev-card uncertainty specifically — pairs naturally with the root audit's planned fixed-root belief-variance panel.
4. **Expert labels for the exam tool.** `strategic_root_exam.py` renders positions; the missing layer is a curated, hand-labeled position set (graded best/acceptable/blunder per root) so per-phase skill is scored against ground truth rather than only cross-seed stability. Offer stands: labeled opening/robber/dev-timing/endgame sets in the tool's format, keyed by `(seed, decision_index)`.

## Reproduction

Analysis script: `analyze_shards.py` inside the evidence bundle (stdlib + numpy). Rerun over a full wave's shards — the shard-level percentages here rest on 17 games and should be treated as strong-signal/preliminary; the §1 receipts are 256–1,536-game sealed evaluations and stand on their own.
