# Expert review index

External reviews of the Catan-Zero project, ingested in order. Each review reads
the prior state of `CATAN_ZERO_RESEARCH_CHRONICLE.md` (in `docs/plans/`) and is
tagged `R1`...`R8` in `docs/plans/CATAN_ZERO_MASTER_PLAN.md`, which records the
verdict (adopted / adopted-modified / downgraded / rejected) for every
recommendation. This file is just the provenance map; the master plan is the
source of truth for what was actually acted on.

| File | Original filename | What it added |
|---|---|---|
| `review-1.md` | `first55` | R1 — first external review: diagnosed the promotion gate as mis-calibrated for a maturing flywheel (fixed elo0=0/elo1=30 SPRT rejects true +20 Elo candidates by design); recommended promoting the +20 Elo candidate for data generation without over-claiming certified strength; flagged belief/card-tracking as the single biggest missing system component. |
| `review-2.md` | `second55` | R2 — categorical value head / "stop regressing" framing, plus a concrete external benchmark spec; independently converged with R1 on the gate diagnosis and on belief tracking as a public-information gap ("public belief, not omniscience"). |
| `review-3.md` | `third55` | R3 — producer-window A/B design, AlphStar-style league precedent, adversarial-exploiter evidence, and a frozen-battery evaluation protocol; supplied the compute argument against extending the gate ("would not spend 900-1200 games trying to make a true +20 Elo candidate pass a +30 Elo SPRT"). |
| `review-4.md` | `fourth55` | R4 — MSE-collapse mechanism for the value head, frozen-corpus 3-epoch probe, and a challenge to the Q-backup design; weakest citation hygiene of the eight reviews but strong structural argument for exact-deduction features (transformer latent space is an inefficient place to implicitly track cards). |
| `review-5.md` | `fifthh` | R5 — "Catan-Zero 2.0" design synthesis: PFSP league commitment, categorical lambda-blend, an explicit truncation bin in the value support, and the counterweight position that plain masked AlphaZero may already be sufficient without belief modeling (first reviewer split). |
| `review-6.md` | `sixthh` | R6 — statistical audit of the pooled turn-4 claim (demoted to "suggestive"), the science-ladder-vs-champion-ladder framing, a canary generation lane, mandatory game-level validation splits, and MAPLE/BetaZero/AlphaGateau as escalation precedent; highest-reliability review (accurate quotes, correct mctx defaults). |
| `review-7.md` | `seventhh` | R7 — executive summary of the live primary-source sweep (~70 sources checked against project claims); points to `review-7-full.md` for the full report. |
| `review-7-full.md` | `catan_zero_expert_review_20260708.md` | R7 full report — [LIT]/[JUDGMENT]/[EXPERIMENT]-tagged sweep; identified the lambda-blend self-distillation mechanism as a plausible drift co-cause, an anchor-staleness instrument error, WHR rating-ladder fit, "raise the expert" (unused search budget), a disambiguation factor for the compression-trend claim, and a report-hygiene/citation audit (several unverifiable or non-existent citations flagged). Highest-evidence review of the eight; its "over-invested in certainty, under-invested in compounding" framing was adopted as the working critique of the whole project posture. |
| `review-8.md` | `EXPERT_REVIEW_REPORT_2026-07-08.md` | R8 — the second, independent 2026-07-08 review; converged with R7 on promoting the turn-4 candidate now and re-specifying the gate as regression protection rather than an improvement bar. Introduced the unified plateau theory (the c_scale=0.03 fix and the Elo plateau are the same SNR-decay object), an HL-Gauss value-head fix, lambda self-distillation to Reanalyze, and RGSC restarts; also identifies a paper (Kao et al., TAAI'22) that undercuts the project's Gumbel+chance-node novelty claim, and reframes the ~45.7% result vs. a value-only bot as parity rather than a -30 regression. |
| `catan_zero_critique_report_20260706.md` | (same name) | Internal 4-agent audit from 2026-07-06 predating the R1-R8 external review sequence: found the project narrative overclaiming a "flywheel" that didn't exist in code yet, plus several live bugs (no-op weight decay, truncated value signal off, unmasked featurization in one code path). |
| `EXPERT_REVIEW_PROMPT.md` | (same name) | The prompt sent to reviewers to solicit R7/R8, given here for reference/reproducibility. |

See `docs/plans/CATAN_ZERO_MASTER_PLAN.md` section 0 for the live status table
and section 9 for how reviewer disagreements were ruled on.
