The expert review is done. Full report (with all citations, per-question answers, and the ranked experiment list) is at catan_zero_expert_review_20260708.md
  in this worktree, built from four parallel live literature sweeps (Gumbel/low-budget search, value-head robustness, gating/league practice, Catan +
  imperfect-info landscape — ~70 sources checked against your claims). Here's the executive summary:

  The ten things that matter most

  1. Your gate is the bug — promote now, re-spec to regression protection. A fixed elo0=0/elo1=30 SPRT on a maturing system is an anti-pattern with direct
  precedent against it: fishtest shrinks its bounds as Stockfish matures (production patches gate at ~{0,2} Elo); KataGo's author calls the gatekeeper optional
  "training wheels" and never controlled-tested it; AlphaZero/MuZero ran fully ungated (verified primary sources). Your +20-Elo turn-4 candidate (52.8%/1000,
  p≈0.04) should already be champion — withholding promotion withholds the only thing expert iteration compounds on. Recommended: elo0=−10/elo1=+15 for the
  flywheel, external panel as the binding tripwire (two consecutive declines → revert), and a WHR (Whole-History Rating) fit over every game you've ever played
  — pooled power across all gates, zero GPU cost, and no AZ-lineage project has done it (verified gap).
  2. The structural value-head fix you asked for exists: HL-Gauss categorical value (Farebrother et al., ICML 2024 Oral, arXiv:2403.03950). Their controlled
  result: MSE value regression "degrades with prolonged training; cross-entropy losses do not" — a near-exact description of all four of your value failures.
  You already built a categorical head. The killer experiment isn't the parity gate — it's re-running your 3-epoch arm under cross-entropy: if val value no
  longer blows up, you've unlocked sample reuse >1, which at your generation-bound economics is equivalent to multiplying the fleet.
  3. Your λ=0.5 blend is a self-distillation amplifier: the v-component comes from the generating net's archived root value. MuZero-Reanalyse, EfficientZero-v2
  (Search-Based Value Estimation), and ReZero all say the same thing — recompute the bootstrap component with the current/lagged net. This plausibly co-caused
  your +69% lineage drift, and value-only reanalyze on the stored window is your zero-new-games plateau-breaker.
  4. The plateau diagnosis is partly instrument error. The anchor holdout is one pinned gen-3-era wave; flat telemetry can't distinguish "window distilled"
  from "anchor stale/off-distribution." Refresh anchors per generation before concluding anything from flatness.
  5. Biggest untapped strength lever: raise the expert. You made sims 13–19× cheaper, the fleet ~3–4× faster — and left n_full at 64 with symmetry averaging
  off. Raise n_full to 128+, raise p_full, enable root symmetry averaging at wide roots (KataGo ships exactly this; MiniZero shows a +276-Elo jump when budget
  rises after the net is competent).
  6. Wire what you built. Opponent pool, Go-Exploit restarts, categorical head, D1, symmetry averaging — all built, none deployed, and the first two attack
  your named top problems. Spinning-top theory (Czarnecki et al.) says cyclic strategy density peaks exactly at your skill band — population play isn't
  optional there. Also: train against the target — 10–20% of generation games vs catanatron_value with search targets; poker exploiter precedent exists
  (AlphaExploitem 2026), no board-game AZ project has done it.
  7. Statistical overclaims to walk back: "λ=0.5 best result in history" is winner's-curse over 7 arms (59.0 vs 57.0 is not significant; the direct H1-vs-gen2A
  H2H was never run — it's one 34-minute gate); the "compression trend" has overlapping turn-to-turn CIs; and your pentanomial rationale is stated backwards —
  fishtest's own data shows correct pairing adds ~15% power (the naive binomial is conservative, not anticonservative).
  8. Citation bugs before anyone external reads the report: the Charlesworth Catan arXiv paper doesn't exist (his arXiv paper is about Big 2; the Catan work is
  a blog/repo); "CatAnalysis" is unfindable under that name; you're missing Deep Catan (Cazenave, AAAI 2022 — prior AZ-style 4p Catan attempt); and preempt
  HexMachina (OpenReview 2026, LLM code-evolution claiming 54.1% vs AlphaBeta, gray lit) by qualifying "first" to "first peer-reviewed learning-based."
  9. Your novelty claims mostly hold. The Gumbel+chance gap is confirmed by DeepMind's maintainer declining it in mctx issue #66; the rescale-noise mechanism
  is genuinely undiagnosed (mctx issue #108 is the same symptom class, open and unanswered — cite it as corroboration). One reframe: mctx's default c_scale=0.1
  is the Atari constant; the paper's board-game value is 1.0 — so the paper already established 10× domain-dependence, and your contribution is characterizing
  the wide/near-tied/stochastic/sub-1-sim failure regime, not contradicting the constant.
  10. What you don't know you don't know: there is no human calibration of catanatron anywhere — "#1 bot" is currently a bot-relative claim with no defined
  finish line (colonist.io, with its non-uniform dice, is the only human-rated Catan world); nobody has computed Catan's disambiguation factor (Long et al.,
  AAAI 2010) — one afternoon with your replay tooling turns "masking suffices" from assertion into a measured claim and tells you whether MAPLE-style
  multi-world search (+136–291 Elo on strongly-hidden games) matters here; and your policy head may be quietly data-starved at 7.7% policy-target rows now that
  full searches are cheap.

  Recommended sequence: promote + re-spec gate (today, free) → WHR fit (free) → reanalyze-lite on the existing window → HL-Gauss head + 3-epoch retest →
  n_full=128 generation + root symmetry → wire pool + exploiter games → 91M retry only after those land.

  One caveat on the sweep itself: the "lc0 removed gating" narrative could not be confirmed from a primary lczero.org source (community-analysis grade only) —
  treat that one datapoint as lore; everything else in the report is tagged [LIT]/[JUDGMENT]/[EXPERIMENT] with URLs in §9.
