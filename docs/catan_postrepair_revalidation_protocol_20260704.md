# Post-Repair Re-Validation Protocol (2026-07-04)

Execution checklist for the moment the retrained value-head checkpoint
(task #65, value-head repair v2) lands. This file is the authoritative
source for tonight's execution -- read from here, not from message
history.

## Background

Gate A (corrected search, full-eval n_full=64, vs the raw hard-target
policy) FAILED on two independent replicates: A100A pooled 128 games,
19.2% search win rate, SPRT accepts H0; A100B replicate, 22.1%, same
verdict. The lazy-eval arm failed even harder (2/23, 8.7%).

Mechanism analysis (see `tools/sigma_trace_placement_root.py`, landed
commit c8972f4): search losses concentrate in blowouts (44/59 losses had
search finishing with <=4 VP while raw reached 9-11), consistent with a
bad initial-placement decision compounding for the whole game. A
controlled sigma trace on 40 real 54-wide placement roots (the widest,
highest-stakes root in the game) shows the search's completed-Q term
overrides the network's own prior ranking on 57.5-72.5% of these
decisions, depending on config -- with only ~1.2 simulations per
candidate at that width. Root cause: `_rescale_completed_q`'s min-max
rescale stretches whatever raw-Q spread it observes to fill [0,1]
regardless of whether that spread is genuine signal or sampling noise; at
1-4 samples/candidate against a value net with corr(q,z)~0.61 to true
outcome, the observed spread is frequently pure noise, but the rescale
manufactures the same apparent confidence from it as from a genuine
strong signal, swamping a near-flat prior.

Two repair tracks were in flight: (1) value-head repair v2 -- retrain the
value head on raw-selfplay (no-search) games with true outcomes, via
`--train-value-only` (policy weights frozen -- CONFIRMED from
value-repair v1: post-retrain policy logits came back bit-identical, so
the "raw hard-target policy" side of every H2H arm below is the same
regardless of which checkpoint file it loads); (2) SNR arms (more
samples/candidate via narrower width and/or larger n_full, 32
samples/candidate) run by the optimizer track.

**SNR ARMS RESULT (2026-07-04): all 4 came back NEGATIVE** -- pooled
5-21 (19%), statistically indistinguishable from Gate A's 19.2% baseline,
even at 32 samples/candidate where sampling noise should be largely
averaged out. Interpretation: the value net's problem is per-position
BIAS (it systematically misranks placements), not just estimate
variance -- more search of a biased function converges confidently to
the WRONG answer, it doesn't average out. This is why value-head repair
v2 is the remaining lever, not a wider/deeper search. Arm D (the SNR
track's locked config) is DROPPED from Step 3 below -- there is no
config from that track worth testing.

**RETRAIN STATUS (2026-07-04): SUCCEEDED.** Attempts 1-2 died on a loader
memory issue (padded-concat inflation, ~45GB/M rows -- task #66's
streaming loader is the permanent fix, not yet landed). Attempt 3
completed cleanly: 10.1M rows, 1 epoch, train value_loss=0.523, val
value_loss=0.529 (close train/val gap, converged not diverged).
Checkpoint: `runs/bc/entity_graph_35m_value_repair_v2_raw_selfplay_20260704/checkpoint.pt`.

## Step 1 -- Calibration probe

**Goal:** confirm the retrained value head is meaningfully better
calibrated before spending any search-side compute on it.

**Corpus holdout:** hold out by GAME_SEED ranges, not row sampling. A
game's rows must not leak across the train/holdout split (they are
highly correlated -- same board, same policy, same eventual outcome).

For attempt 3's specific checkpoint (10.1M rows, 4 included blocks), the
holdout is the game_seed ranges WITHIN those 4 included blocks that
attempt 3 itself did not train on:
`5006335:5006667`, `5106335:5106667`, `7006335:7006667`,
`7106335:7106667`. Use held-out games from those ranges specifically --
NOT the general historical "86k corpus" framing from earlier in this doc
(that was the full raw-selfplay corpus before attempts 1-2's loader
issue forced attempt 3 down to a smaller, measured-safe subset). If a
later retrain attempt uses a different row/block selection, get the
holdout ranges for THAT attempt before running this step -- don't reuse
these four ranges blindly.

**Run:** `tools/value_repair_calibration_probe.py --checkpoint <new> --device cuda:0 --out runs/value_repair_calibration/post_repair_v2_attempt3.json`
(landed commit c374f3b) evaluates the new checkpoint's value head over the
held-out game states via a direct forward pass on the already-stored
entity-token features (no search, no live game replay), computes corr(q,
z) against the actual game outcome, same methodology as the original
pilot probe.

**Time estimate:** ~10 min, 1 GPU (no search, just forward passes over a
few thousand held-out decisions).

**RESULT (2026-07-04, attempt 3 checkpoint, 6410 held-out rows):**
corr(q,z) 0.514 (pre-repair) -> 0.683 (post-repair). E[q|win]/E[q|loss]:
0.633/-0.003 -> 0.487/-0.472 -- win/loss separation widened from 0.636 to
0.959 despite E[q|win] itself dropping (q distribution re-centered near
zero, q_mean 0.309 -> 0.039). Real, meaningful calibration improvement.
Per the "guidance only" bar, this was not gated further -- proceeded
straight to Step 2.

## Step 2 -- Sigma trace, same 40 roots, new checkpoint

**Goal:** verify the repair shows up mechanistically at the root, not
just as an aggregate number.

**Run:**
```
tools/sigma_trace_placement_root.py \
  --checkpoint <new_checkpoint> \
  --n-states 40 --base-seed 500001 --n-full 64 \
  --configs 50:0.1,50:0.03,1:0.1 \
  --out runs/sigma_trace/placement_root_sweep_postrepair.json
```
Same `--base-seed 500001` as the original (pre-repair) run -- this
reproduces the IDENTICAL 40 placement states, so it's a direct
before/after comparison, not a resample.

**Time estimate:** ~20-30 min, 1 GPU.

**RESULT (2026-07-04, attempt 3 checkpoint, same 40 states):**

| Config | Pre-repair | Post-repair | Delta |
|---|---|---|---|
| cv50/cs0.1 (shipped) | 72.5% (29/40) | 70.0% (28/40) | -2.5pt (barely moved) |
| cv50/cs0.03 | 62.5% (25/40) | 45.0% (18/40) | -17.5pt (real improvement) |
| cv1/cs0.1 | 57.5% (23/40) | 47.5% (19/40) | -10pt (real improvement) |

**VERDICT: neither the pass bar nor the escalation clause fired.** The
shipped config alone did NOT clear the 30-40% target (barely moved) --
repair alone does not suffice. But NOT every config is >60% either
(cs0.03 and cv1 both dropped below 50%) -- so this is not the "repair
didn't help at all" escalation case. Reading: the repair genuinely
helped, but only becomes visible at this mechanism level when PAIRED
with a config change (cs=0.03 or cv=1), not with the shipped config
alone. Reported to team-lead for the go/no-go on Step 3 -- NOT
pre-selecting arms based on this signal regardless of their decision,
since flip rate is a mechanism diagnostic, not a config-selection proxy
(see caveat below).

**CAVEAT (2026-07-04, post-armV): flip rate is a MECHANISM diagnostic,
not a config-selection proxy.** cv=1/cs=0.1 had the LOWEST flip rate in
the pre-repair sweep (57.5%, best of the three configs tested) but WENT
2-14 (12% win rate) in the real H2H -- the worst of any arm run so far.
Do not rank or select configs by flip rate; use it only to confirm
whether the repair changed the underlying mechanism (real signal vs.
noise at the root). Config selection is decided EXCLUSIVELY by Step 3's
H2H results and Step 4's promotion rule.

## Step 3 -- H2H gate arms

**Configs to test -- THREE arms** (arm D, the SNR track's locked config,
is DROPPED: all 4 SNR arms failed at 32 samples/candidate, see
Background -- there is no SNR config worth testing):

| Arm | Config | Question |
|---|---|---|
| A | `--checkpoint <new> --n-full 64` (cv50/cs0.1, shipped) | Does repair alone fix it? |
| B | `--checkpoint <new> --n-full 64 --c-scale 0.03` | Repair + softer sigma |
| C | `--checkpoint <new> --n-full 64 --c-visit 1` | Repair + within-mctx visit-scaled sigma |

**Arm C pre-repair result (old checkpoint, same config, real H2H, cap=600):**
2-14 (12% win rate), 0 WW / 6 LL decisive pairs, 0/16 truncated -- the
WORST result of any arm run so far, despite cv=1 having the BEST
(lowest) flip rate in the Step 2 sigma-trace sweep. Kept in this bracket
because its failure mode is noise-specific (visit-scaled sigma amplifies
noise at the final SH comparisons) -- a genuinely improved value head may
fix exactly this. But treat its pre-repair result as a low prior, not a
reason to expect it to win post-repair.

All three vs raw (same policy, old-or-new checkpoint file, per the
frozen-policy confirmation above), using `tools/gumbel_search_vs_raw_h2h.py`
with F5 pair-level SPRT as the primary read (naive per-game reported
side-by-side, for reference only). Settings: `--max-decisions 600` (NOT
300 -- the adopted post-Gate-A policy; the 44-49% truncation rate at cap
300 bled nearly all decisive-pair power), 16 pairs per arm, a FRESH
`--base-seed` per arm (do not reuse Gate A's / armS's / armB's / the SNR
arms' seed ranges -- avoid any seed-overlap ambiguity in the final
numbers).

**TIMING CALIBRATION -- DONE (2026-07-04, measured pre-emptively; timing
is checkpoint-independent so this number stands for the real run too):**
2 pairs / 4 concurrent workers, cap=600, n_full=64, on 1 A100 GPU -> 4,335s
wall-clock, bounded by the longest single game (347 decisions, ~72 min).
Games ranged 181-347 decisions with ZERO truncation at cap=600 (8/8
pairs fully informative -- the cap=600 policy is fully vindicated: at
cap=300 roughly half the sample was truncation-excluded from the
pair-level analysis; at cap=600, none was).

**Arm sizing from this measurement:** 16 pairs = 32 games. Run 32
concurrent workers on ONE A100 GPU (1 game/worker) -> wall-clock per arm
~ the longest game's duration, ~1.2-1.5h. With only THREE arms now (D
dropped), all three run in parallel across 3 GPUs (32 workers each) ->
total wall-clock under 2h -- an even lower GPU footprint than the
original 4-arm plan. This replaces the earlier ~88-GPU-hour
extrapolation, which significantly overestimated cost (it assumed
CPU-bound serialization per GPU rather than accounting for how many
concurrent workers a single GPU's evaluator can actually absorb).

## Step 4 -- Promotion rule

1. Compute pair-level (F5 concordant) win rate per arm: wins / (wins +
   losses) among decisive pairs only (WW + LL; excludes splits and
   truncation-incomplete pairs).
2. Filter to arms with win rate > 55%.
3. **If zero arms clear 55%:** no config lock. Do not generate. This is a
   deeper problem than search-config tuning -- escalate for a fresh
   diagnosis rather than trying more configs ad hoc.
4. **If one or more arms clear 55%:** pick the arm with the MOST decisive
   pairs (largest informative N -- most statistical confidence in that
   estimate, not just the highest raw rate on a possibly-thin sample).
5. **Tiebreak** (same decisive-pair count, comparable win rate), in
   order: (a) lower n_full, (b) narrower `--max-root-candidates-wide`,
   (c) higher c_visit (cheaper to reason about / no rescale-noise risk).
6. The locked config (checkpoint, n_full, c_visit, c_scale,
   max_root_candidates_wide) feeds directly into the generation launch
   command. No further gate.

## Step 3 -- RESULTS (FINAL, all 12/12 fleet files landed, 2026-07-05)

Fleet: 3 arms x 4 files (2 hosts x 2 GPUs/host), 16 pairs/file -> 64
pairs/arm pooled. Pooling done by `tools/h2h_postrepair_aggregate.py`
(pools by `game_seed`, not the per-file-local `pair_id` -- see that
tool's docstring for why pair_id alone is unsafe across independently
-launched invocations). One operational note: armC's A100A shard (gpu4
+gpu5, seeds 270000/271000) was killed as collateral during an unrelated
tmux cleanup around 01:56-01:58 UTC, leaving an orphaned stuck worker;
it was killed and cleanly relaunched with the identical spec (same
seeds, same config) once discovered, so the final numbers below reflect
the intended 64-pair sample for all three arms, not a reduced one.

| Arm | Config | Pair win rate | Decisive (W-L) | Split excl. | Trunc excl. | SPRT (elo0=0,elo1=30) |
|---|---|---|---|---|---|---|
| A | cv50/cs0.1 (shipped) | **68.4%** | 38 (26-12) | 26 | 0 | LLR 1.067, continue |
| B | cs0.03 | **71.4%** | 28 (20-8) | 36 | 0 | LLR 0.932, continue |
| C | cv1/cs0.1 | **67.5%** | 40 (27-13) | 23 | 1 | LLR 1.060, continue |

All three arms are a sharp reversal from Gate A's pre-repair failure
(19-22%) and from armC's own pre-repair result (12%, 2-14). None has
formally resolved the SPRT to H1 at n=64 pairs/arm (LLR needs 2.944;
that requires low hundreds to ~1000 paired games per elo1=30's own
sizing, per sprt_gate.py's docstring) -- but the promotion rule (below)
does not require SPRT resolution, only the >55% pair-level win-rate
filter.

## Step 4 -- APPLIED (mechanical, per the rule above)

1. Pair-level win rate per arm: A=68.4%, B=71.4%, C=67.5%.
2. Filter >55%: **all three arms clear the bar.**
3. (n/a -- not zero arms.)
4. Pick most decisive pairs among arms that cleared: A=38, B=28, C=40.
   **C has the most decisive pairs (40) -- wins outright, no tie.**
5. (n/a -- no tie, tiebreak not needed.)
6. Mechanically-selected config: **arm C -- checkpoint =
   runs/bc/entity_graph_35m_value_repair_v2_raw_selfplay_20260704/checkpoint.pt,
   n_full=64, c_visit=1.0, c_scale=0.1, max_root_candidates_wide=54.**
   Reported to team-lead for confirmation per the standing rule (the
   locked config is pending their sign-off, not unilaterally declared
   here).

## FINAL DECISION (team-lead, 2026-07-05, overriding the mechanical step-4 pick above)

**Binding conclusion: GATE PASSED.** All three arms clear 55% decisively
(67.5-71.4% pair win rate vs. Gate A's pre-repair 19-22%) -- search now
beats raw policy. The repair worked; this result is not in doubt.

**Config lock: NOT granted to armC.** Team-lead overrode the mechanical
step-4 pick (see above) with a stated reason, not a preference --
**rule-flaw identified**: "most decisive pairs" was meant as a proxy for
"most confident winner," but here decisive-pair count instead reflects
armC's LOWER split rate (cv1 produces fewer draw-like/split outcomes --
an artifact of sharper, more decisive play), not higher strength. armC's
actual win rate (67.5%) is the LOWEST of the three point estimates. At
n=64 with only ~30-40 decisive pairs each, the three point estimates
(67.5% / 68.4% / 71.4%) are statistically indistinguishable (overlapping
CIs of roughly +-15%) -- this gate has enough power to answer "does
search work" but not enough to discriminate between these three configs.
Correct output: **ALL PASS, CONFIG UNDECIDED.**

**Lesson for future gates:** decisive-pair-count must always be paired
with a win-rate-adequate-N check before using it as a selection
criterion -- a config that produces fewer split/drawn-like outcomes will
accumulate decisive pairs faster than a stronger config with a higher
split rate, and count alone can select on decisiveness-of-outcome rather
than strength.

**Resolution:** config lock deferred to a properly-powered confirmation
leg (n=128+) run on the v3 checkpoint (masked + unfrozen + KL) -- v3 is
the actual gen-1 substrate anyway, and the value net changes again
before then, so locking a config against the value-repair-v2 checkpoint
would not carry forward cleanly. Carried forward as the two confirmation
configs: **armA** (baseline cv50/cs0.1 -- simplest, no deviation, avoids
cv1's known noise-amplification fragility under distribution shift) and
**armB** (highest point estimate, 71.4%). **armC (cv1/cs0.1) drops as
primary but stays eligible** if a future signal favors it again.

This does not block the merge session or the v3a/v3b retrains, which
use the training recipe (not the search config) and start now.
