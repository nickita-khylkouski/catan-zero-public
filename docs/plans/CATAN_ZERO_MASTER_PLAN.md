# Catan-Zero Master Plan (living document)

**Goal: build the #1 Catan bot in the world** — decisively beat catanatron's hand-tuned ValueFunction bot (currently ~30 Elo ahead of us externally), then every other bot, then superhuman play.

This plan is distilled from expert reviews of `CATAN_ZERO_RESEARCH_CHRONICLE.md` **plus our own analysis of each claim**. It is not a digest: every recommendation is evaluated against our own evidence; some reviewer ideas are modified or rejected (§9, §10). Updated after every review.

---

## 0. Document status

| Field | Value |
|---|---|
| Last updated | 2026-07-09 (A0 scalar-retention ruling + S1 pre-wave execution) |
| Reviews ingested | **R1** (`first55`), **R2** (`second55` — categorical value/Stop-Regressing + benchmark spec), **R3** (`third55` — producer-window A/B, AlphaStar league, adversarial-exploiter evidence, frozen battery), **R4** (`fourth55` — MSE-collapse mechanism, frozen-corpus 3-epoch probe, Q-backup challenge; weakest citation hygiene), **R5** (`fifthh` — "Catan-Zero 2.0" design synthesis; PFSP league commitment, categorical λ-blend, truncation bin, masked-AZ-is-enough counterweight on belief), **R6** (`sixthh` — statistical audit of our pooled claim, science-vs-champion ladder, canary lane, game-level splits, MAPLE/BetaZero/AlphaGateau), **R7** (`seventhh` + full report `catan_zero_expert_review_20260708.md` — live primary-source sweep, ~70 sources, [LIT]/[JUDGMENT]/[EXPERIMENT] tagged; λ-blend self-distillation mechanism + reanalyze-lite, anchor-staleness instrument error, WHR, raise-the-expert, disambiguation factor, report-hygiene audit) |
| | **R8** (`EXPERT_REVIEW_REPORT_2026-07-08.md` — second live-verified sweep, [L]/[J]/[E] tagged, stats re-computed from our own numbers; SNR-decay plateau theory, may-already-be-at-parity finding, diagnostics bundle, exploitability probe, Kao claim-kill, exact-SH confound) |
| Reviews pending | R9+ (one at a time; plan re-thought after each) |
| Reviewer conflicts ruled on | 8 (§9; C6 substantially revised, C8 refined by R8's diagnose-first discipline) + 1 tracked split (two instruments assigned) |

**Tags:** `[R1]`…`[R5]` = source review. `[US]` = our own pre-existing evidence. `[ME]` = my own analysis. Agreement alone does not set priority; every item carries a verdict — **✅ adopted / ⚠️ adopted-modified / ⏬ downgraded / ❌ rejected** — with reasoning.

**Consensus tracker (for calibration, not authority):** unblock the flywheel by promoting +20 candidates: **5/5** (gate at −10/+15: R1/R2/R3/R5 explicit — R5 adds the Stockfish normalized-Elo framing; R4 gateless dissent ruled §9-C4). Categorical/distributional value was a **5/5 hypothesis**, but A0's exact local test rejects the evaluated HL formulation for this wave and binds A1 to scalar. Opponent pool/league now: **5/5** (R5 the most committed — full PFSP league). Value fragility = target-quality/loss-structure problem: **5/5**. D6 symmetry now and opening/wide-root special treatment were **5/5** and **6/6** reviewer hypotheses respectively; binding S1–S3 evidence, not vote count, selects D6 and n64/n128/conditional n256. Population/external eval revamp: **4/5**. Go-Exploit/high-regret restarts: **3/5** (R4/R5 silent — stays in; nobody argues against and it's pre-built [US]). `c_scale=0.03` had **4/5 + R4 trajectory dissent** and is now the S1 control, not a pre-approved production result. Variance-aware completed-Q / uncertainty-scaled shrinkage: **4/5** (R5: our D2 arm "is basically what Gumbel MuZero hints at in its footnotes" — retry it once V4's uncertainty head exists). Belief/imperfect-info: **SPLIT — 4 for cheap-belief-features-first (R1/R2/R4/R6), 1 silent (R3), 1 lean-against (R5)** — ruled §9-T1: workstream split into uncontested deduction-features (proceed) vs contested belief machinery (behind error-atlas evidence; escalation ladder = MAPLE/BetaZero [R6]). Engineering before bigger nets: **6/6**.

**⚠ R4 reliability note [ME]:** R4 misreads our report at least once ("reducing c_scale from 50.0 to 0.03" — conflates c_visit=50 with c_scale 1.0→0.03), cites at least two works I cannot verify exist ("Treant-Gumbel", "Lambda-Reachability"), ships no reference links, and sets fantasy success criteria (external panel ">50% within 48h"). Its *mechanisms* are often sound; its *facts and citations* are treated as unverified unless corroborated.
**R5 reliability note [ME]:** medium — real, checkable sources for its load-bearing claims (fishtest/normalized-Elo, KataGo SelfplayTraining.md, AlphaStar, risk-diversity 2305.11476, Deep Catan/Cazenave), but several citations are mismatched to their sentences (an MCTS survey cited for graph-transformer claims) and many links just point back at our own uploaded report. Trust the design synthesis, spot-check the attributions.
**R6 reliability note [ME]:** highest of the first six — explicitly separates "my judgment / needs experiment / established in literature," quotes our report sections accurately, gets the actual mctx defaults right (value_scale=0.1, maxvisit_init=50 — correcting R4's misread), and is the only one of R1–R6 to audit OUR statistics (the pooled turn-4 p-value). Weight its judgment calls accordingly.
**R7 reliability note [ME]:** highest evidence grade with R8 — every load-bearing claim tagged [LIT]/[JUDGMENT]/[EXPERIMENT] with primary sources verified live (mctx issues #66/#108, fishtest #348, KataGo docs, LZ #1524), it audits our chronicle's citations (finding real bugs: the Charlesworth conflation, "CatAnalysis" unfindable, Deep Catan missing), and it self-reports its own single unverified datapoint (lc0-gating lore). R7-verified [LIT] items are treated as *established*, not opinion. Its overall diagnosis — "you have over-invested in certainty and under-invested in compounding" — is adopted as the plan's one-line frame.
**R8 reliability note [ME]:** same evidence grade as R7 (live-verified, tagged, and it *re-computed our statistics from our own numbers* — the z-scores check out). The two sweeps are complementary, not redundant: where they overlap (gate precedents, HL-Gauss, reanalyze, pool percentages) they agree; R8 adds the SNR-decay mechanism, the parity finding, the diagnostics bundle, the exploitability probe, and three claim-corrections R7 missed (Kao kills the flat Gumbel+chance claim; exact-SH confound; anchor-doesn't-predict-gates). Where they differ on numbers (elo1=+10 vs +15) the difference is immaterial at a true +20 — noted, not ruled.

**⚠ R4 reliability note [ME]:** R4 misreads our report at least once ("reducing c_scale from 50.0 to 0.03" — conflates c_visit=50 with c_scale 1.0→0.03), cites at least two works I cannot verify exist ("Treant-Gumbel", "Lambda-Reachability"), ships no reference links, and sets fantasy success criteria (external panel ">50% within 48h"). Its *mechanisms* are often sound and well-argued; its *facts and citations* are treated as unverified unless corroborated by R1–R3 or checked ourselves.

---

## 1. The consensus verdict — and why I believe it's correct

All three reviews, independently, converge:

> "You are still acting like you are in discrete-generation AlphaZero mode after switching to a continuous small-compute flywheel." [R1]
> "Your flywheel is being throttled by a gate designed for an earlier regime." [R2]
> "Holding a +15–20 Elo candidate forever because it is not +30 Elo is equivalent to freezing the policy while asking training to extract more from a fully distilled window." [R3]

**My own check [ME]:** this follows from our own numbers, not reviewer groupthink. At a true +20 Elo, the elo0=0/elo1=30 SPRT holds forever *by design*; nine rounds of flat anchor telemetry (policy 1.397–1.407, value 0.240–0.244) independently prove the gen-3 window is fully distilled. Cost asymmetry seals it: a false promote costs one flywheel turn (~hours, reversible, tripwired); a false hold stalls compounding permanently. R3 adds the compute argument: "I would not spend 900–1200 games trying to make a true +20 Elo candidate pass a +30 Elo SPRT" — which also formally kills the extend-the-gates option. R4 states the theory version: "Expert iteration derives its mathematical power from generating data that is marginally better than the previous generation. Holding a definitively better network back... halts the core compounding mechanism." **✅ diagnosis confirmed 4/4 + own math.**

The risk that replaces "can AZ work?" is named identically by all three: **self-play inbreeding.** R3's closing framing is the sharpest statement of the project's actual danger:

> "The most dangerous failure mode is not that Catan-Zero is broken. It is that it becomes very good at beating its own previous self while the catanatron_value gap stops closing." [R3]

Top-level organization (all three):
- **STRENGTH TRACK** — beat catanatron_value ASAP with the deployed bot; "the deployed bot is allowed to be stronger than the generator" [R2]; purity optional, strength not [R2].
- **FLYWHEEL TRACK** — expert iteration under regression-protection gating + population evaluation + structural value fixes + diversified data.

---

## 2. IMMEDIATE DECISIONS

> **Current execution override (2026-07-09):** this section records the
> strategic promotion/data-distribution decisions that shaped the roadmap; it
> is not launch authority for the bounded pre-wave lane. The active executable
> order is A0 (complete) → S1 → S2 → conditional S3 → sealed/rendered A1
> handoff → **STOP**. Generator promotion, canary generation, and fleet window
> A/B begin only after the separately owned production data lane accepts that
> handoff and supplies a synchronized seed ledger.

### 2.1 Historical/future promotion ruling — turn-4 (+20 Elo) generator candidate [R1][R2][R3]

Unanimous. R3 adds the precise precondition, which we adopt since the instrument is already running:

> "Do this now **unless the 1000-game external panel shows a clear, statistically meaningful external regression**." [R3]

R7 closes the argument with the precedent floor (all [LIT], primary-source verified): AlphaZero and MuZero ran **fully ungated**; KataGo's author calls the gatekeeper optional "training wheels" and never controlled-tested it; fishtest *shrinks its bounds as the engine matures* — production Stockfish patches gate at ~{0, 2} Elo today; LZ community analysis: a 55% gate yielded ~7.6 Elo/net vs ~12.0 at 50% — the stricter gate produced *slower* progress. "You built a certification instrument and are using it as a promotion valve." And on the external-flatness counterargument: **"don't let an underpowered instrument veto a powered one."** [R7]

**⚠ Honesty caveat on the plateau premise [R7, adopted]:** "window fully distilled" is inferred from flat anchor telemetry, but the anchor is one pinned gen-3-era wave — flatness there cannot distinguish distillation-complete from anchor-gone-stale/off-distribution. This *softens* one leg of the promote rationale without changing the decision (cost asymmetry + the 52–53% gates stand on their own). Action: **refresh the anchor from the current window's held-out seeds every generation, keeping old anchors as a longitudinal series** — which also resolves our longest-standing open question (anchor staleness, unaddressed by R1–R6). R8 goes further and demotes the anchor permanently: gen-4 showed "the historical promotion signature" and still gated flat — **anchor telemetry is a drift tripwire, never a promotion signal.** [R8]

**THE UNIFIED PLATEAU THEORY [R8 — the single best mechanism across all eight reviews, adopted as the working hypothesis]:** our c_scale=0.03 fix and our plateau are *the same object*. The min-max rescale forces the completed-Q term to fill a fixed ~1.6–2.4-nat logit budget at every root regardless of how informative search was. Early on, true Q-spread >> value noise → the budget was signal → +49 Elo/turn. As the policy sharpens, true Q-spread shrinks toward the (constant) value-noise floor → a growing fraction of the same fixed budget is noise → **the improvement operator's SNR decays by construction**, predicting +49→+49→+33→+20 at constant data volume. "'Window fully distilled' is the training-side shadow of this search-side fact." Remedy order: **denoise roots (symmetry averaging — measured 3.3×, never deployed) → re-tune c_scale upward (the 0.03 optimum was measured at CURRENT noise) → then raise sims.** Alongside two compatible co-mechanisms: (B) ExIt fixed-point (apprentice–expert gap shrinks at fixed budget; ELF OpenGo: ~200 Elo per rollout-doubling even at the end — the fixed point moves outward with budget) and (C) distribution narrowing/inbreeding (the external-flatness signature). **Diagnostics bundle (~1 GPU-day, runs before spending weeks on any single remedy [R8]):** (i) search-SNR probe — same search twice with different seeds at ~200 roots per generation checkpoint; SNR theory predicts KL(π′‖prior) constant while cross-seed agreement decays; (ii) rollout-doubling Elo (n=64 vs n=128, 400 paired games) — big gap ⇒ budget binds, tiny ⇒ capacity/data; (iii) diversity telemetry (unique-state fraction, opening entropy, line concentration per generation); (iv) D6 orientation-noise vs top-5 Q-spread trend per generation — spread→noise confirms (A) directly. The three mechanisms have *different* remedies; diagnose before spending.

**[ME] Mandatory pre-steps (validated independently by R6's statistical audit):**
1. Our "52.8%/1000" pools two *different checkpoints* (gen-4 discrete 400g + flywheel round-17 600g) — and, as R6 adds, **two different sim budgets (n=8 and n=16) with optional extensions, quoted as a post-hoc p-value after looking**: "suggestive, not a clean 'verified promotion'... no casual pooled p-values after looking" [R6]. Language corrected project-wide: turn-4 is *suggestive*. Pick ONE candidate — lean gen-4 (cleaner provenance, ledgered seeds, artifact-verified anchor telemetry) — and re-gate it under the new −10/+15 spec (~34 min). A true +20 passes elo1=+15 quickly.
2. Read the 1000-game panel result (`runs/gates/v16_external/vs_value_500pairs.json`) first, per R3's precondition; per R6, the panel otherwise runs *asynchronously* and blocks only on significant decline — a flat 200-game read never freezes the flywheel.
3. **Future canary option [R6]:** after production-wave authorization, a
   candidate may produce 20–30% of new generation data while an external panel
   runs. This is not executed by the bounded pre-wave lane. It remains the
   lighter-weight complement to the §2.4 window A/B — "if the candidate is a
   self-play overfit, the next candidate/anchor/external metrics will show it
   quickly; if it is genuinely better, you stop wasting compounding time."
   [R6]

### 2.2 ✅ Registry roles: dual champion + league [R1][R2][R3]

| Role | Promotion rule |
|---|---|
| `generator_champion` (R3: "self-play producer") | regression gate below + anchor tripwire clean + "no external catastrophic regression" [R3] + clean provenance |
| `public_champion` | external population-suite confirmation at real power (1000+ games) [R1][R2][R3] |
| `tournament_bot` | public champion wrapped in strength-track config (§4.3); explicitly allowed to be stronger than the generator [R2][ME] |
| `opponent_pool[]` | prior champions + regressed nets (kept, not deleted [R1]) + external bots + **hard negatives**: "older checkpoints or exploiters that beat the latest disproportionately" [R3] |

**Generator gate spec (now 5/6 on the exact numbers):** SPRT `elo0 = −10, elo1 = +15` ("prove not worse than −10 Elo and likely better than +15" [R5], Stockfish normalized-Elo framing [R5]; R6 adds concrete error rates and cap: "α≈0.05–0.10, β≈0.05, cap 600 games"), paired color-swapped seeds, pentanomial accounting. Alternative trigger: "two positive gates + clean anchor" [R2][R3]. Bayesian form: `P(Elo>0) ≥ 0.75–0.85` and `P(Elo<−10) ≤ 0.05` [R1][R6]. Gate cheaply at low sims (n=8–16, where distillation gains show) with occasional confirmation at production n=64 [R5][R6][US]. Precedent depth from R6: KataGo's gate was a light 100/200-game check and its docs make gatekeeping *optional*; Leela Zero community experience found ~50–52% thresholds more efficient than 55% — "your current system is in exactly the 'small improvements get rejected forever' regime" [R6]. Old `0/+30` gate survives ONLY for public-champion claims.

**Science ladder vs champion ladder [R6, adopted as the registry's operating principle]:** "Keep clean controls, but the deployed champion should be the strongest statistically credible arm... The champion ladder should be ruthless." Clean-science lineages stay as labeled experiments; they never again cost us a stronger champion (see the λ0.5/gen2A retro-fix, §4.1).

**R7's exact flywheel spec (adopted as the written config):** SPRT elo0=−10, elo1=+15, α=β=0.05, 300 games n=16, extensions as now; **the binding tripwire is the external panel, not the internal gate** (DeepNash's validation pattern: fixed external panel, not self-play Elo); the +30-style gate survives only for discrete "gen-N" *turn announcements* — "two instruments, two jobs." [R7]

**NEW — WHR over the whole ladder [R7, adopt immediately]:** fit Whole-History Rating (Coulom 2008) over *every game ever played* — all gates, extensions, ablation arms, panels. Pooled power with honest error bars; resolves the compression-trend question; detects persistent +20 drift across individually-"continue" gates. Verified gap: no AZ-lineage project has done it. ~1–2 days engineering, **zero GPU**. [R7]

**End-state trajectory (from R4, staged not immediate — ruling §9-C4):** R4 argues for fully gateless EMA-weight deployment (generators continuously pull θ_EMA, β≈0.995; external panel demoted to passive tripwire). Rejected *for now* — our round-5/round-11 postmortems show exactly what ungated compounding does to our current value head (+69% drift, 37% gates). But adopted as the destination: only after scalar A1 demonstrates stable fresh-data training, the independently tested per-game/value-weight discipline lands, and **two consecutive clean promoted turns** complete under the −10/+15 gate may EMA-pull be piloted with the gate running async as a tripwire instead of a blocker. Independently of deployment, **EMA/SWA weight averaging of recent checkpoints is worth testing now as a cheap candidate-smoothing step** [R4; SWA precedent in KataGo [R2]].

### 2.3 ✅ Rollback rules [R1][R2][R3]

- External: "roll back automatically after two consecutive external/population regressions" [R3] (= R1's rule; R3 makes it *automatic*). R6 supplies the pre-set number we lacked: block/revert on `P(external Elo delta < −25) > 0.9` or two consecutive promoted champions declining [R6][ME: adopt as the written bound].
- Internal: if producer-fed gen-5 doesn't beat gen-4/gen-3 internally, roll back the generator [R2].

### 2.4 ✅ NEW — the promotion is run as a controlled experiment: producer-window vs gen-3-window A/B [R3][ME]

R3's key experimental-design contribution:

> "Generate one window from gen-3 and one from the new producer if you can afford a short A/B. Train champion-init candidates on both windows. Compare on internal population, external panel, and high-regret suite." [R3]

**[ME] Once the production data lane is authorized, this controlled A/B is the
default rather than an optional interpretation.** Its exact fleet topology and
seed ranges come only from the sealed handoff plus the live synchronized ledger;
the current R&D lane neither claims those ranges nor launches the windows. The
comparison turns the central hypothesis ("gate blockage was the ceiling") into
a measured outcome, and R3's falsifier comes free: "if producer-fed data improves
internal H2H but repeatedly worsens external population scores, you have
confirmed inbreeding and should stop linear promotion until the opponent pool is
live." [R3]

### 2.5 ✅ Post-promotion data mix [R1][R2][R3]

R1: 70–80/10–20/5–15. R2: 75/15/10. R3: "80% latest self-play, 10% previous champion, 5% older champion, 5% hard negative/exploiter" and separately 75–85/10–15/5–10. All the same shape. **Adopt: 75–80% producer self-play / 10–15% recent+older champions / 5–10% hard negatives + targeted restarts.** Start simple; "use PFSP-style sampling later. The first goal is simply to stop the distribution from narrowing." [R3] Guardrails: per-opponent telemetry (win rate, KL, policy entropy, value calibration separately) [R3]; "do not let opponent-pool rows dominate the main distribution at first" [R3]; don't abandon from one weak turn [R3].

### 2.6 ✅ Define the benchmark NOW — spec + frozen battery [R2][R3]

R2 demands the tournament spec ("board distribution, time/search budget, rules, engine version, allowed books/heuristics, opponent set, statistical protocol — otherwise you optimize to whatever harness is most convenient"). R3 extends it into a **frozen public-style evaluation battery**:

> "Fixed tournament maps, random map suite, opening-placement suite, robber/development-card suite, old champion population, catanatron variants, exploiters, raw-policy and search-budget ablations, calibration reports by decision type." [R3]

And the strategic point: "there is no universally accepted AlphaZero-class Catan leaderboard... **you probably have to create the benchmark that will make your eventual '#1' claim credible.**" [R3]

Spec skeleton (fill in): 2p no-trade 10VP base rules; catanatron @ pinned version; seeded board distribution; per-move budget (wall-clock, which credits our Rust/MPS work [ME]); opponents = catanatron_value (primary), AB3/AB4, prior champions, any runnable published agent; ≥1000 paired color-swapped games/matchup, pentanomial, ledgered disjoint seeds; position suites held out from all training [R3].

**R7's finish-line correction (adopted):** "#1 Catan bot" is currently a **bot-relative claim with no defined finish line** — *no human calibration of any Catan bot exists anywhere* [R7-LIT: verified absence]; the only human-rated Catan world is colonist.io (non-uniform balanced-dice variant, 4p-centric, own full-information GNN+MCTS bot). Define "won" as staged: **(a) SPRT-certified vs every catanatron bot at 2p no-trade → (b) colonist.io-rules variant + human trials.** Also [R7]: the 2p-no-trade moat is also a ceiling — audit NOW how expensive the 4p+trading extension is (token layout, trade actions) before more infrastructure calcifies around 2p; the 4p+negotiation literature (Cuayáhuitl DQN, ~82% vs humans on the trade subtask; Keizer EACL 2017 negotiation-only DRL beat humans 81.8% [R8]) becomes the goalpost the moment we claim #1 publicly. The defensible 2026 claim: "strongest 2p no-trade agent, first leak-free gated AZ-class Catan system." [R8]

**⚡ R8's parity finding (may reframe the whole goal):** gen-3's 45.7%/200 vs catanatron_value is z=−1.22, **p≈0.22 vs 50% — parity is not excluded. "You may already have beaten your north star and can't see it."** If the running 1000-game panel lands ~48–52%, the headline goal flips from "close the −30 gap" to "prove superiority with power" — which changes what we build next (power > levers). **Harness-neutrality requirement [R8, adopted]:** our entire external ladder runs inside our own Rust cross-play harness, and catanatron's own (tiny-sample) ladder disagrees with our bot ordering (their AB2 > VF) — before any public #1 claim, run the definitive matches **in catanatron's own engine or a neutral referee**, so nobody can attribute the result to engine bias. That neutral 1000-game number "is the number the outside world will judge."

---

## 3. RANKED EXPERIMENT QUEUE (my merged ranking; departures from individual reviews ruled in §9)

Post-R7 re-rank (ruling §9-C8): **reanalyze-lite enters at #2** — the only lever that extracts more from data already paid for, zero new games, and it repairs a live defect (the archived-λ self-distillation path). **Raise-the-expert enters at #6** — we made search 13–19× cheaper and never spent it on generation quality; R7: "per-turn gain in expert iteration is bounded by the search's edge over its own prior."

| # | Item | Why here | Decision rule | Tags |
|---:|---|---|---|---|
| 1 | Re-spec gate → confirm ONE candidate → check external panel → promote as producer → **producer-vs-gen3 window A/B** (§2.4). Free adds ride along: anchor refresh per generation [R7], WHR ladder fit [R7] | Only mechanism that advances the distribution; now a controlled experiment | Producer window wins internally + no external tripwire → keep; internal-up/external-down twice → inbreeding confirmed, stop linear promotion | [R1][R2][R3][R7][ME] |
| 2 | **Reanalyze-lite (NEW):** batch-forward current champion over the stored window's states, overwrite the λ-blend's v-component (archived generating-net root values = self-distillation amplifier), retrain one dose | Zero new games; strongest precedent multipliers in the MuZero family (Reanalyse ablation 92→240; EfficientZero-v2 SBVE; ReZero); lr≈0 probe infra is ~80% of the build [US] | "The 'fully distilled' window contains another candidate when its value targets are refreshed" [R7] — if the anchor moves, graduate to root-search reanalyze (n=16 over stored states, on explicitly allocated between-wave compute) | [R7] |
| 2b | **Diagnostics bundle (NEW, ~1 GPU-day):** search-SNR probe (same search, two seeds, per checkpoint), rollout-doubling Elo (n=64 vs 128), diversity telemetry, noise-vs-Q-spread trend (§1) | The three plateau mechanisms (SNR-decay / fixed-point / inbreeding) have *different* remedies — "diagnosing before spending the next two weeks matters" [R8] | Directs weeks 2–4; no-go on nothing | [R8] |
| 3 | Population arena + frozen benchmark battery + sims-ladder (n=8→128) diagnostics + WHR + Nash-averaged population rating [R7] + **1000-game panels re-run in catanatron's own harness** (neutrality, §2.6) [R8] | n=200 can't resolve 4–5%; fixed bots hide blind spots [R3]; Elo vs own lineage is the redundant-population eval Nash averaging exists to correct [R7]; payoff matrix makes cycling *visible* instead of inferred [R8] | Public-champion changes gated on it; watch external panel **trend**, not level (ROA-Star decay warning [R7]) | [R1][R2][R3][R7][R8] |
| 4 | Tournament-config candidate (no training): S1 selects the D6 threshold + `c_scale`/D1 setting, S2 selects global n64 versus n128, and only a qualifying S2 result may open S3's adaptive n256-at-`>=40`-action test. Neither D6 nor n128/n256 is a default before the binding S1–S3 artifacts adjudicate it. | Potential external Elo at bounded cost; symmetry noise (0.175 nats) > placement spread (~0.06), but the measured denoising result does not itself approve a tournament default. | Render and test only the exact S1–S3-selected config; otherwise retain the current control. | [R1][R2][R3][R5][R7][local S1–S3] |
| 5 | Categorical value head hypothesis (HL-Gauss preferred over two-hot). **Executed 2026-07-09:** the exact gen2B scalar control reproduced `.665247→.809018→.841849`, while 33-bin HL regressed `1.198052→1.532889→1.710083`; typed verdict retains scalar for A1. Historical 87.85M HL stress is closed for this wave. | The literature motivated the falsifiable test; local evidence now overrides the prior for this formulation/regime. | **REJECTED for this wave**; reopen only with a separately predeclared formulation, never an in-place bin/sigma retune. | [R2][R3][R4][R7][R1] |
| 6 | **Raise the expert — R8-sequenced:** (i) S1 tests root 12-symmetry averaging at `>=20`-action roots and the **five-arm** re-grid against `.03/off`: `.03/on`, `.1/off`, `.1/on`, `.3/off`, `.3/on`, using a checkpoint-specific D1 `sigma_eval` artifact → (ii) S2 tests n64 versus n128 globally → (iii) qualifying evidence may open S3's n256 test only at `>=40`-action opening/wide roots, always full; `p_full` 0.25→0.4 remains a separate future arm | SNR theory (§1): denoise first, retune second, spend sims third; separate D6 and adaptive-budget thresholds keep the intended compute allocation honest | A post-D6 arm beats `.03/off` H2H; global sims: H1 with measured cost <1.6× (up to 1.8× only for a clear margin); adaptive n256: H1 or predeclared stability gain at ≤20% whole-game overhead | [R7][R8; R2/R3/R6 precedent][local S1–S3] |
| 7 | Per-game value-loss weighting (+ forced-row downweight), staged after the scalar A1 one-dose result and isolated from other value-side changes | "16k games = 16k independent outcomes, not 3.6M labels" [R2]; R3 concurs | Drift decreases, gate neutral-positive | [R1][R2][R3] |
| 8 | Opponent-pool generation (mix §2.5, AlphaStar template: 35/50/15 PFSP [R7]; KataGo's own adversarial hardening used **18% frozen-past opponents** [R8]) + **RGSC-prioritized restarts** (upgrade f71 from uniform to ranking-based regret prioritization [R8: Tsai et al. ICLR 2026, github.com/rlglab/rgsc]) + **exploiter lane: 10–20% vs catanatron_value/AB3 with OUR search targets** [R7] + policy-surprise weighting in the loader (half of KataGo's sample-frequency weight [R8]) | RGSC "continued improving a nearly-converged 9×9 Go model 69.3%→78.2% where both vanilla AZ and Go-Exploit flatlined" — the one lever with direct evidence of un-sticking a *converged* AZ [R8]; Territory Paint Wars: self-play winrate holds ~50% while generalization collapses 73.5%→21.6%, undetectable internally — fix was 20% vs a fixed opponent [R8]; spinning-top [R7] | External population Elo improves without internal collapse; exploiter read = value-panel moves >5 pts while internal ≥ parity [R7]; restarts capped per §9-C3 | [R1][R2][R3][R7][R8][US] |
| 9 | Error atlas + held-out high-regret suite (never trained on [R3]) + **disambiguation-factor measurement** (Long et al. 2010 — one afternoon with replay tooling [R7]) | "You have a ladder, not a diagnosis" [R1]; df turns "masking suffices" from assertion into a measured, publishable claim AND adjudicates the §9-T1 split in advance [R7] | Per-phase loss attribution + df number in hand; plan re-ranked from findings | [R1][R3][R7] |
| 10 | **Deduction tracker** (exact recovered-public-info features) → net features + planner spectra + aux heads; probabilistic belief machinery beyond deduction SPLIT OFF — quantified trigger now [R7]: revisit only if within ~10 Elo of catanatron_value and stalled | **[ME] mostly EXACT deduction in 2p — public information**; R7 corroborates the premise ("spends are public, robber steals reveal, hands are bounded") | +Elo vs same net without it, esp. robber/dev buckets | [R1][R2][R4][ME; R5 split; R7 trigger] |
| 11 | Aux heads on at small weight (final-VP margin, road/army state, production potential, belief summaries) + **TD-horizon value heads on REALIZED outcomes** (upgraded from V6 — ruling §9-C6-rev) | Built but zero-weighted [US]; KataGo ablation: removing aux value-adjacent targets costs **190 Elo / 1.65× training speed** — largest single component in the paper; "aux targets are how KataGo un-starves" the one-sample-per-game value head [R7] | One-dose single-variable; calibration/H2H; rides the next retrain | [R1][R2][R3][R7] |
| 12 | Root candidate cap (top 16–24) ≈ policy-target pruning [R3] + **backup-side uncertainty weighting** (KataGo: weight = min(cap, a·err^b), ~20–60 Elo at low playouts [R7]) + turn D1 back on (mild winner, dormant [R7]) | Key R7 lesson: D2 (selection-side) was neutral; KataGo's win is on the **backup weights, not the qtransform** — aim V4's error head there | H2H at production sims, not target-KL | [R1][R2][R3][R7][ME] |
| 13 | **Full root-search reanalyze** (n=16–256 fresh searches over selected high-KL/high-width/high-uncertainty stored states) — the graduation of #2 | Selective, not blanket [R3]; MuZero Reanalyse + ReZero precedent; embarrassingly parallel on explicitly allocated between-wave compute [R7] | Transfer to gates per row of compute; anchor + high-regret holdout armed | [R1][R2][R3][R7] |
| 14 | 80–100M net | Only after a promoted, stable 35M scalar candidate, completed S1–S3 search calibration, and `>=10M` fresh audited rows. The historical 87.85M stress de-risks plumbing only; it is not scale approval. | Equal-exposure positive gates with no value blowup | [R1][R2][R3][R7][local A0/C0] |
| 15 | Architecture v2 rerun — action-target cross-attention as the real experiment [R3]; **full D6 equivariance SKIPPED**. Graph-distance/adjacency bias remains a cheap hypothesis but is deferred until a runnable module, tests, and its own causal matrix row exist; it is not silently folded into another retrain. | One confounded A/B ≠ theorem (unanimous); “at 3–4M rows/turn and a policy that's still prior-starved, the data engine binds, not capacity” [R7] | two module seeds; equal rows and wall clock; wide-root/opening buckets + H2H | [R1][R2][R3][R7] |
| 16 | Temperature-schedule test: small nonzero temperature (or Gumbel sampling) to ~decision 150, vs current argmax-after-90 | Late-game data diversity currently rests entirely on board/dice variation [R7]; KataGo keeps stochasticity late | Cheap A/B; diversity telemetry + gate | [R7] |
| 17 | Engineering (when next touching search): MCGS cross-move subtree reuse (+69 Elo chess / +310 crazyhouse) + Speculative-MCTS inter-decision parallelism (up to 5.81× self-play latency) | "Worth ~another 2× fleet-equivalent, but it's throughput, not strength-per-sample — sequence behind the science items" [R7] | Profiler-confirmed before build | [R7][R1] |
| 18 | **Exploitability probe (NEW):** train a small adversary net via self-play vs the FROZEN champion (Wang et al. methodology — found >97% exploits vs superhuman KataGo at <14% compute) | The only instrument that answers "is masked-AZ leaving a strategic hole"; doubles as the Perolat mixing test (does Catan need genuine mixed strategies?) and the second §9-T1 instrument | Exploit >70% found ⇒ raise pool %, consider R-NaD-style regularization; none found ⇒ **close the belief-state question for this phase** [R8]. 3–5 GPU-days | [R8] |
| 19 | LR-schedule arms in the flywheel (0.5×/2× of the constant 3e-5) | KataGo's run history shows large discontinuous gains at LR drops; constant-LR may be leaving a step-gain on the table [R8] | Anchor + gate vs control round; ~free | [R8] |

---

## 4. WORKSTREAMS

### 4.1 Gating & registry (Flywheel track)

Covered in §2. Standing audit item, now upgraded from audit to ACTION [R6]: **λ=0.5 provenance inconsistency** [R1] — trace which λ the champion lineage actually used from run artifacts, not docs [US]. R6 sharpens the fix: direct-match λ-arm vs gen2A and champion the winner. **R7 corrects the claim's strength while confirming the action:** 59.0% vs 57.0% is two independent 400-game estimates with ~±5% CIs — *not significant*, and the winner of a 7-arm matrix is biased upward by construction (**winner's curse**). So the λ arm is not "the best verified arm"; it's an unresolved 34-minute question. The λ *direction* is still probably right (Willemsen NCA 2022; Abrams z/q-averaging) [R7-LIT]. Run the direct gate — but re-derive the λ conclusion under the categorical head, since the two interact [R7]. **Deeper R7 finding (drives queue #2):** the λ-blend's v-component is the *generating net's archived root value* — a self-distillation amplifier in the continuous loop, plausibly co-causal in the +69% drift; the bootstrap component must come from the *current/lagged* net (MuZero-Reanalyse; EfficientZero-v2 SBVE; ReZero) [R7-LIT].

**Statistics hygiene (R7 audit, all adopted):** (a) compression trend (+49→+49→+33→+20) has almost entirely overlapping turn-to-turn CIs — "a flat +35/turn is consistent with your data"; don't build strategy on it until the WHR fit resolves it. (b) Our pentanomial rationale is stated *backwards*: fishtest #348 measured within-pair correlation ≈ −0.15 — correct pairing *adds* ~15% power; the naive binomial is conservative, not anticonservative. Keep pentanomial, fix the stated rationale, and measure our empirical pair correlation once (~zero cost). (c) Report citation bugs to fix before anyone external reads the chronicle: Charlesworth Catan-arXiv doesn't exist (Big 2 conflation; the Catan work is blog+repo), "CatAnalysis" unfindable, **Deep Catan (Cazenave, AAAI 2022) missing** — a prior AZ-style 4p Catan attempt, so qualify first-mover claims; also qualify vs HexMachina (OpenReview 2026 gray lit, 54.1% vs AlphaBeta) → "first peer-reviewed learning-based." (d) c_scale framing: mctx's shipped default 0.1 is the paper's *Atari* constant; 1.0 is its board-game value — our 0.03 is 3× below the default, 33× below the board setting; **the mechanism, not the constant, is the finding** (cite mctx #108 as corroborating symptom, #66 for the chance-node gap). [R7]

**R8 additions to the hygiene list (all adopted):** (e) **Kao, Guei, Wu & Wu, "Gumbel MuZero for 2048" (TAAI 2022) KILLS the flat "no one combined Gumbel with stochastic chance nodes" claim** — it's even linked from mctx #66; the defensible claim narrows to "perfect-simulator *enumerated* chance nodes + hidden-information masking + paired statistical gating in a 2-player board game." (f) The pooled turn-4 p is also **one-sided** (two-sided ≈ 0.08) on top of R6's pooling objections. (g) The λ non-result quantified: z≈0.57 vs control — "adopting it was reasonable; the label is not." (h) **Exact-SH negative result has an unexamined confound**: Karnin-style SH *discards* statistics between rounds while mctx *stockpiles* — if our port was restart-style, the 45.9% loss conflates budget accounting with statistics-discarding. Check the implementation before publishing (touches kill-list entry + task #61 [US]). (i) "AB5 is weakest" quirk unverifiable in catanatron sources — re-derive from our own data or drop. (j) Willemsen cite is ALA-2020-workshop/NCA-2022, not NCA 2021. (k) **Repro hygiene**: H2H gate tooling untracked in git on one host [US confirms]; ~2 days of repo hygiene is the price of the novelty claims' credibility. [R8]

### 4.2 Population arena + frozen benchmark battery (both tracks' measuring stick)

Pool: current champion + candidate, old-champion population ("v3a, gen-1, gen-2A, gen-3, current candidate" [R3]), catanatron_value + **style-randomized variants** ("weight noise, opening randomization, search-depth variants" [R3]), AB3/AB4, raw policy, low/high-sim variants, scripted styles, exploiters. R5 adds concrete non-catanatron baselines worth trying to run: Deep Catan (Cazenave) and Settlers-RL-type agents (e.g. kvombatkere/Catan-AI) — "at least one non-catanatron RL agent as extra baseline"; adopt if runnable in our harness, don't sink days into porting [ME].

**Why exploiters are non-negotiable now [R3, new evidence]:** "adversarial policies have beaten very strong Go AIs by exploiting narrow blind spots even when the victim is superhuman by normal benchmarks" (arXiv 2211.00241). A fixed-bot panel structurally cannot see these holes.

**Two scores, kept separate:** "world ranking" (absolute population Elo) vs "self-play ladder" (internal H2H) [R3] — decisions require both [R1][R2]. Concrete matrix shape [R6]: **all-pairs cross-play among the last 8–12 nets** plus external bots — latest-vs-latest gating alone can hide non-transitivity/rock-paper-scissors cycles (the Leela Zero failure mode) [R6]; add varied board/opening suites to the panel [R6].

**Diagnostics baked in:** sims-ladder gates n=8/16/32/64/128 (policy-vs-value-vs-search ceiling separation [R2]); n=256-vs-64 same-net [R2]; calibration on fresh-policy states [R2] and **by decision type/action bucket** ("if opening/robber/dev-card buckets fail, value target quality is the ceiling" [R3]); diversity telemetry [R2]; **producer-fed vs gen-3-fed window comparison** as the gate-blockage-vs-inbreeding discriminator [R3].

Position suites (frozen, held out from training forever [R3]): opening placements, robber choices, dev-card timing, longest-road races, high-resource swing turns [R3].

**[ME] Cost check:** nightly 2000-game league ≈ 1 GPU-day across fleet — a standing job. The running 1000-game panel is installment #1. [US] All runs: ledgered disjoint seeds; masked nets eval masked on master code.

### 4.3 Tournament config (Strength track)

Two configs, explicitly [R2][R3]: generator = fast/stable/cheap/exploratory; tournament = maximize win rate under the benchmark budget. Candidate components are adaptive n128/n256 at openings/wide/high-entropy roots and D6 12-fold averaging at high-noise roots (batch the symmetry transforms in Rust so Python overhead doesn't multiply [R3]), plus a belief tracker once built and a possibly re-tuned Q-transform at high sims. The exact tournament stack remains unresolved: S1 selects D6 + `c_scale`/D1, S2 selects n64 versus n128, and S3 may test adaptive n256 only if S2 qualifies it. No candidate component becomes a default before those binding artifacts exist.

**Opening treatment ⚠️ (modified from R2; deepened by R6; tightened by the current bounded protocol):** distilled opening-head (offline n≥1024 + symmetry search on sampled boards → auxiliary head) remains a shelf item; literal book only if the benchmark spec (§2.6) makes boards repeatable [ME]. The immediate operational test for wide roots (≥40 legal actions) is **n=256 versus n=128**, with an independent D6 threshold and every selected wide root forced full so its policy target is full-weight. n512 is deferred until n256 itself clears strength/stability/≤20%-overhead gates. A reanalyzed opening-root corpus from old games remains a later way to refresh the noisiest targets without new self-play [R6]. Its falsifier: "if opening-special improves external but not internal, still keep it for the competition bot." [R6]

**D6 train-time augmentation** should also be on "in at least one serious candidate after promotion" [R3]. [US] Check status first: symmetry-augment machinery exists (f74/f74b + task #91 verified it) — confirm whether production training actually enables it before treating this as new work.

### 4.4 Value overhaul — STAGED, not bundled (ruling §9-C1)

Shared diagnosis, now 4/4. R3's statement: "the value head is not mysteriously fragile; it is being asked to learn low-noise long-horizon values from high-correlation, sparse, reused ±1 labels." R4 adds the loss-function mechanism: "the network is forced to collapse a complex, multimodal probability distribution into a single mean scalar... larger models memorize the stochastic noise of the scalar targets far faster than smaller models" — which is the cleanest available explanation of the 91M epoch-2 blowup, and it's *testable* (below).

- **V1 — categorical value hypothesis tested and rejected for this wave**
  (queue #5 closed).  A0 ran the exact gen2B scalar control against the matched
  33-bin HL-Gauss formulation: scalar reproduced
  `.665247→.809018→.841849`, while HL regressed
  `1.198052→1.532889→1.710083`.  The binding verdict retains scalar MSE/readout
  for the one-dose fresh A1 candidate and closes the historical 87.85M C0
  stress.  The literature motivation remains useful, but categorical value may
  return only as a separately predeclared future formulation—not an in-place
  bin/sigma retune.
- **V2 — per-game value weighting** + forced-row downweight [R1][R2][R3].
- **V3 — aux heads on, small weight** (queue #9): final VP margin, longest-road/army state, production potential, settlement/city potential, resource-count/hidden-VP belief summaries [R2][R3]; R4 concurs — "not just as zero-weighted telemetry, but as dense predictive regularizers."
- **V4 — uncertainty/value-error head** (stop-gradient) → search-time shrinkage (§4.7) + restart targeting.
- **V5 — value-specific optimization knobs, if drift persists:** decoupled/lower value LR and stop-gradient between trunk and value head (now 4-source: [R2][R3][R4][R5 "separate optimizers"]), Huber/log-cosh value loss, stronger value weight decay [R3]; stronger L2/dropout in value-feeding layers [R5]; EMA/lagged target net for the λ-blend [R1][R5 "lagged target networks for value only, like in DQN"]; value-gradient clipping [R2]; adjacent-state value-smoothness penalty (same board ± small hand change → bounded value delta) [R5 — speculative, needs paired-state sampling machinery; shelf item, last in line [ME]].
- **V0 — reanalyze-lite (queue #2, precedes everything here):** overwrite the archived-λ v-component with current-champion forwards on the stored window; the λ-blend as shipped is a self-distillation amplifier (§4.1, [R7]). This is a *defect fix*, not an enhancement — do it before tuning anything else value-side. **R8 supplies the missing mechanism paper:** Kumar et al. (2010.14498) — self-referential regression targets cause *effective-rank collapse* that compounds with reuse and **vanishes with pure MC outcome targets**; "almost a controlled experiment for your situation." Also from R8: at plateau, v_root ≈ v_net, so the target degenerates toward 0.5·z + 0.5·v_net — value learning slows exactly when fresh signal is needed; and our configuration (value weight 1.0 + λ-blend + reuse 3.0) is "hotter than the closest precedent ran" — **MuZero-Reanalyse set value weight 0.25 specifically as the sample-reuse countermeasure** → drop flywheel value-loss weight to 0.25–0.5. Options ladder: anneal λ toward z as the flywheel matures / lagged-EMA net for the blend half / Reanalyze so the bootstrap term is always fresh. The scale target: **the banked 32.6M-row / 417GB corpus is states-good, targets-stale — value-only reanalyze is nearly free at our leaf costs and "the principled escape from 'the window is fully distilled' that doesn't require promotion at all."** [R8]
- **V6 (REVISED after R7, ruling §9-C6-rev) — split by target source:** (a) **Aux TD-horizon heads on REALIZED trajectory outcomes** (KataGo's ~6/16/50-horizon analog): UPGRADED to ride the next retrain alongside V3 — my drift objection doesn't apply when targets are realized outcomes rather than our own value estimates, and KataGo's ablation (−190 Elo without aux targets) makes this the biggest documented aux effect in the literature [R7]. (b) **Bootstrapped TD(λ)/n-step machinery** (targets derived from our own net): stays gated behind V0/V1/V5 + EMA target — the drift channel is now *proven* (§4.1), not just suspected.
- **V7 — two-phase training retest, conditional** [R6]: "your policy head tolerates reuse; your value head does not" → one conservative joint pass, then a policy-only (or policy+low-LR-torso) pass with the value head frozen, on soft search targets. Our earlier two-phase arm scored only 53.25% [US], but that was pre-diagnosis; R6's condition for the retest — after value classification or on a fresh window — is adopted verbatim.
- ❌ **Symlog transform** (R4, from DreamerV3): rejected — symlog exists to compress unbounded/extreme-variance returns; ours are already bounded in [−1,1]. Cargo-cult risk, zero expected value here. [ME]

Calibration telemetry throughout: by game phase, root width, color, hidden-info entropy, opening bucket [R2], and decision type [R3].

### 4.5 Population data: opponent pool + targeted restarts

League blueprint = AlphaStar (Nature 2019): "a league of diverse strategies and counter-strategies, not a single linear self-play chain" [R3] — start with the simple mix (§2.5), PFSP later [R3]. R5 commits harder to the league than anyone ("train in a 'mini-ecosystem' of strategies") and supplies the phase-2 mechanics, adopted with my trigger [ME]:

- **PFSP activation trigger [ME]:** switch from the fixed mix to PFSP sampling when per-opponent telemetry shows the mix has gone uninformative — e.g. several pool members beaten >90% (wasted games) while others stagnate. Until then the simple mix wins on ops simplicity and attributability.
- **League refresh** [R5]: retire very weak nets, add experiment nets (different architectures, different *risk preferences* — safe-builder vs dev-card-gambler vs road-rush profiles, per the population-diversity literature, arXiv 2305.11476), maintain current champs + old champs + fixed bots.
- **Style-diversity telemetry** [R5], merging with R2's diversity metrics: classify games into strategy styles and track pool coverage explicitly.

Restarts (Go-Exploit 2302.12359; regret-guided 2602.20809): archive = wide opening placements, large search-vs-raw disagreements, value-ensemble disagreement, robber/dev swing states, positions where gen-3 and catanatron_value choose different plans [R3] + external losses, high value-error, unstable-symmetry openings, long-road pivots, high hidden-info-entropy states [R1][R2]. Generate 0.5–1.0M rows from starts with color swaps; capped weight (§9-C3); **held-out high-regret suite never trained on** [R3]. Falsifier: "if high-regret data improves the suite but hurts full-game external panel, the restart distribution is too adversarial or over-weighted" [R3].

Mechanism guard (all three): external bots are **state-distribution generators**, never imitation teachers — train from OUR search targets at those states [R1]. R6 adds the missing operational detail: **for bot games, train only on OUR side's decision rows initially** (the bot's action rows have no search targets and would be imitation) — and store opponent metadata but train a single unconditional policy [R6]. If restart data helps the high-regret suite but harms full-game play, **demote it to value-only rows** (policy_weight=0, our existing PCR machinery [US]) before cutting the fraction [R6].

[US] Head start: f71-regret-restarts branch has extraction + bit-exact replay + generation; was blocked on public_observation, which has landed. Revive, don't rebuild — **and upgrade sampling from uniform to RGSC's ranking-based regret prioritization** [R8]; "the NeurIPS-2025 recipe (branch from high-uncertainty roots, ensemble outcomes) is nearly wired for you" [R7].

**Diversity strangulation (R8 §1.4 — a third data-side failure mode beyond inbreeding):** deterministic search + seed-derived chance + argmax after decision 90 + 100% champion-vs-champion "together shrink the support of the self-play distribution every generation." AlphaZero's own supplement measured the cost (5.8%→14% vs Stockfish from near-tie sampling). Cheap shipped countermeasures [R8]: root policy temperature 1.1–1.25 early-game, **policy-surprise weighting** in the loader (oversample states where search disagreed with the prior — half of KataGo's sample-frequency weight), forced playouts + target pruning, small nonzero late-game temperature (queue #16). Key subtlety: our temperature applies only to *played-action* sampling — the search itself is deterministic, so identical seeds produce identical trees; the §1 diversity telemetry decides whether this actually bites before we spend on it.

### 4.6 Belief tracking — SPLIT WORKSTREAM after R5 (ruling §9-T1, revised)

The reviewer field is genuinely divided: **for** = R1 ("biggest missing system component"), R2 ("public belief, not omniscience"), R4 (structural explanation of the external gap — the net "relying entirely on the transformer's latent space to implicitly track cards [is] an extraordinarily inefficient use of network capacity"); **silent** = R3; **lean-against** = R5 ("plain masked AlphaZero can be very strong in imperfect-information games... no ReBeL/Student-of-Games style continual re-solving yet", citing the AlphaZe∗∗ line of work we already relied on [US]).

**My ruling — split the workstream, because the two halves have different consensus status [ME]:**

**(a) Exact-deduction features — UNCONTESTED, keeps its Phase-C schedule.** My analysis from v2: in 2-player Catan there are no third parties — when the opponent robs us we see which card left our own hand, so their resource hand is plausibly an exact deterministic function of public history + own observations. Genuinely hidden: unplayed dev-card identities (trivial hypergeometric posterior) and hidden-VP probability. The key point vs R5: **this is not belief modeling — it is recovering PUBLIC information that per-state masking threw away.** Even R5's "masked AZ is enough" frame endorses richer *public* observations; deduction features are exactly that. No reviewer's position argues against them.

**(b) Probabilistic belief machinery beyond deduction (posterior features where deduction is impossible, IS-MCTS/determinization, belief-conditioned search) — CONTESTED, moves behind evidence.** Gate on the error atlas: build it only if losses to catanatron_value concentrate in dev-card/hidden-VP decisions *after* deduction features land. R5's counterweight + R3's silence mean this no longer rides in on R1/R2/R4's votes alone.

R6 (belief tally now 4-for / 1-silent / 1-lean-against) lands exactly on our split — "build cheap Catan-specific public belief features **before** a ReBeL rewrite" — and supplies the escalation ladder for half (b) if the atlas ever demands it: **MAPLE-style multi-state root aggregation** (2026: aggregate policy/value over multiple sampled world states in one tree — mitigates PIMC's strategy-fusion weakness that we flagged in v4) applied "only for hidden-info-sensitive roots" [R6], with **BetaZero-style progressive widening + policy-prior sampling** for belief-space branching control [R6]. Also noted from R6: AlphaZe∗∗ itself reports strategy-fusion/hopping issues — the same flaw we cited against PIMC cuts against pure masked-AZ too, at some strength ceiling.

**R7 closes the loop with the measuring instrument and the trigger.** Instrument: compute Catan's **disambiguation factor** (Long/Sturtevant/Buro/Furtak, AAAI 2010 — predicts PIMC/masking success from leaf correlation, bias, and how fast hidden info resolves). R7's prior: Catan's df is high ("spends are public, robber steals reveal, hands are bounded" — independently corroborating my exact-deduction analysis), but *nobody has ever computed it* — one afternoon with our replay tooling makes "masking suffices" a measured claim and predicts in advance whether MAPLE's +136–291 Elo (DarkHex/Phantom Go — strongly-hidden games) transfers here at all. Trigger for half (b), quantified: "revisit only if you close to within ~10 Elo of catanatron_value and stall." [R7] The T1 split now resolves by measurement, not by vote-counting.

**R8 adds the second, stronger instrument: the exploitability probe** (queue #18) — train a small adversary net via self-play against the *frozen* champion (Wang et al. 2211.00241 methodology: >97% exploits found vs superhuman KataGo at <14% of its compute). "The only instrument that answers 'is masked-AZ leaving a strategic hole'": exploit >70% found ⇒ the masking/equilibrium story has a hole (raise pool %, consider R-NaD-style regularization); none found ⇒ **"close the belief-state question for this phase."** It also answers the Perolat mixing question (does optimal Catan need genuine mixed strategies — robber/discard bluffs — where plain self-play provably cycles?). R8's prior matches R7's and mine: "Catan 2p no-trade hidden info is mostly fog, not bluffing... but this is exactly the kind of prior worth 3 GPU-days to test." [R8]

**Verifications first (Phase A):**
1. Does the victim's trajectory record the stolen card's identity in our engine and catanatron? Yes → cheap exact tracker (running counts). No → posterior version.
2. **[ME, prompted by an unverified R4 claim]** R4 asserts the true unmasked hidden-state labels "are banked in the corpus" — check whether our shards actually retain hidden-state columns. If yes, belief aux heads train on existing data; if not, labels need regeneration or replay-time reconstruction. Don't build on R4's say-so.
3. **[ME] Flag:** R4's claim that catanatron_value "mathematically reconstructs the deck and models opponent hands" is unverified (R1 called it "belief-based" too, also unverified). Read catanatron's source before crediting the external gap to belief. The error atlas (§4.8) adjudicates empirically either way.

Consumption: (a) net input features (near-zero-init projections, warm-start-safe [ME]); (b) planner chance spectra for steals/dev draws [R1]; (c) aux heads predicting opponent resource-hand distribution + unplayed dev cards — true-state *labels* legal at training since inference consumes only public history [R1][R4]; R4's starting loss weight ≈0.25 is a reasonable initial setting, tune from there. Gate: same net ± belief, vs gen-3 AND catanatron_value, inspecting robber/dev buckets [R2].

⏬ **IS-MCTS / PIMC root determinization** (R4): downgraded, not adopted now [ME]. (a) If exact deduction holds, there's ~nothing to determinize over except dev-card identities — a small, spectra-representable uncertainty; (b) PIMC has known failure modes R4 doesn't mention (strategy fusion, non-locality); (c) belief features + posterior chance spectra capture most of the value at a fraction of the search-cost multiplier. Revisit only if the error atlas shows persistent dev-card/hidden-VP losses *after* belief features land.

### 4.7 Search (keep c_scale=0.03 as the control pending binding S1; treat it as a noise-era compromise, not an end-state)

R1/R2/R3 independently endorse the finding and mechanism ("min-max rescaling turns noise into false confidence" [R3]; mctx guarantee precondition). Framing for the paper: "a domain/regime difference" [R2].

**R4's partial dissent — absorbed as the trajectory [ME]:** "a c_scale of 0.03 means the search is functioning as little more than a policy regularizer rather than a deep tactical planner." Overstated — post-repair search beats raw policy 67–71% [US], so the backup contributes real strength — but the direction is correct: at cs0.03 the Q-signal is heavily attenuated, and that's headroom, not a law. Stable fresh-data scalar A1 evidence (and V4 if it is later tested) may *lower the value-noise floor*, at which point the optimal Q-contribution can move up. This gives R2's "recurring calibration sweep" its purpose: use binding S1 now, then re-sweep `c_scale` only after scalar A1 establishes a changed, stable noise floor; measure search-vs-raw margin as the direct readout of reclaimed backup strength.

**R8 completes this into the unified plateau theory (§1)** and adds three concrete search items: (a) **re-test D2 with a weight CAP** — KataGo's uncertainty-weighted playouts "required a weight cap to work"; our D2-neutral result may be the uncapped version of a real win; (b) the **James-Stein shrinkage coefficient has a closed form** — λ* = v²/(s²+v²) from measured within-arm vs across-arm variance, no hand-tuning; (c) **forced playouts + policy-target pruning** is KataGo's shipped mechanism "most directly aimed at near-tied wide roots" (merges into our §4.7-1 workstream). The governing sequence stands: **"denoise → re-tune, not more operator engineering at current noise."** [R8]

1. **Root candidate cap ≈ policy-target pruning** — one merged, flag-gated feature [ME]: cap considered set at top 16–24 (+ symmetry diversity [R1]), restrict π′ support to genuinely-evaluated ("low-evidence" pruning [R3]) actions.
2. **Per-root uncertainty / completed-Q shrinkage toward prior/root value** [R3] — after V4's error head exists ("D2 was neutral without a trained value-error model" [R1][US]). R5 corroborates and extends: our D2 James-Stein arm "is basically what Gumbel MuZero hints at in its footnotes about variance normalization" — retry D2 with the *learned* uncertainty head scaling the shrinkage (high variance → shrink toward v_mix, low variance → trust Q) [R5].
3. **Adaptive/progressive sims** (§3 #12) with R3's diversity caution.
4. **Recurring c_scale re-sweep** after each structural change, especially after selected scalar A1/value-stability evidence changes the measured noise floor [R2; local A0].
5. **Search-value calibration by action bucket** as standing telemetry [R3].
6. ⏸ **Progressive widening at interior chance nodes** [R5]: start with a subset of chance outcomes, widen by visit count/variance — a middle ground between our lazy(1-sample) and full enumeration. Shelf item [ME]: lazy interior chance is validated unbiased and 13–19x cheaper [US]; only revisit if interior-node variance shows up as a measured problem.
7. ❌ Exact-budget SH n16 stays dead "unless the value/search target machinery changes; your negative result is meaningful" [R3]. ❌ Paper defaults stay rejected (5/5 — R5 explicitly keeps our tuned cv50/cs0.03 + lazy chance as the base).

### 4.8 Error atlas / loss attribution [R1][R3]

"Where, exactly, does catanatron_value take games from us?" [R1] — per-phase/decision-type loss attribution over arena games. Doubles as restart-archive builder (§4.5) and belief validator (§4.6): losses concentrated in robber/dev buckets → belief ranks up; in openings → tournament-config opening work ranks up [ME]. R3's position suites (§4.2) are the frozen, repeatable version of the same idea — build them as one system.

### 4.9 Architecture (deferred, not dead — 4/4)

Order [R3, tightened by the current protocol]: (1) D6 symmetry first (root averaging; train-time augmentation remains a separate check; full equivariance skipped); (2) a promoted stable fresh 35M candidate under the selected objective (currently scalar); (3) retain the cheap exact-deduction input-feature lane behind its error-atlas evidence, separate from architecture/scale; (4) after ≥10M fresh audited rows, run **two separate conditional arms**—action-target gather/cross-attention and 80–100M scale—never bundled, with scheduling decided by measured strength/compute; (5) graph-distance bias only after a runnable isolated implementation exists. Catan's parameterized action locality is still the reason to test cross-attention ("settlement action should see node token; road action should see edge token" [R6]/AlphaGateau precedent), while the v3b result remains only “that upgrade did not win under that data/test regime.” Rerun discipline: deterministic new-module seed, independent LR multiplier, `value_heads` freeze/lower LR for policy warmup, ≥2 seeds, equal data and wall clock [R1][R2].

Ceilings on record: adjacency unused, CLS-only action scoring, CLS-only value, D6 violation [R1][R2][R3].

R8 scaling refinements: size the fresh-data budget by **Neumann & Gros** (2210.00849 — bigger nets need *proportionally more* fresh data; the 2412.11979 follow-up warns of **inverse scaling when end-game states are over-represented — check our phase distribution before the 90M run**); when scaling, the **VISA trick** (2301.11857: symmetry variants chosen for *maximum value disagreement* as extra value-training points, 50% value-generalization-error reduction) plugs directly into our D6 measurements. Honest v3b read sharpened [R8]: one seed, one data scale, v3b only 36% bigger — "architecture doesn't matter **yet at this data scale**."

### 4.10 Engineering — "changes the science budget" [R3]

> "Rust featurization, parallel search, batching, and subtree reuse buy more experiments per day; a bigger net is cheap only after the search/data plumbing stops wasting leaves." [R3] — and explicitly *before* betting on 91M+ [R3].

Order [R1]: (1) typed serialized configs (CLI trap = "science-corruption vector" [R1]; 7+ incidents [US]); (2) config-hash registry across train/generate/gate/eval; (3) eval server/batched leaves + parallel search [R3]; (4) subtree reuse; (5) compiled tree ops if profiling says so. Rust-batched symmetry transforms join the list [R3].

---

## 5. WHAT WE'RE DOING WRONG (merged audit, deduped, now 3-source)

1. **Gate blocks the improvement operator** — "the highest-confidence critique" [R3]; 3/3 → fixed by §2.
2. **External eval too narrow for the claim we want** — "your goal is not 'beat gen-3'" [R3]; n=200 is telemetry; fixed bots hide exploitable holes (adversarial-Go, 2211.00241) [R3] → §4.2.
3. **Underusing diversity; pool designed-but-unwired is "a strategic blocker"** [R3]; "plateaued self-play needs diversity" [R3] → §4.5.
4. **Value-target QUALITY problem misdiagnosed as a value-head law** — 3/3 → §4.4.
5. **Kill-list over-scoping** — add "closed under assumptions" column [R2]; R3 concurs implicitly ("unless the value/search target machinery changes").
6. **λ=0.5 provenance inconsistency** [R1] → §4.1.
7. **Opening strength thrown away** (D6 + no adaptive search at noisiest roots) — 3/3 → §4.3.
8. **Under-optimizing the deployed bot** [R2][R3] → §4.3.
9. **Under-modeling recoverable information post-leak-fix** [R1][R2]; possibly exactly recoverable [ME] → §4.6.
10. **No mechanism-level loss diagnosis** [R1][R3] → §4.8.
11. **No benchmark definition** — and no one else will build the leaderboard for us [R2][R3]; no finish line at all per R7 (no human calibration of catanatron exists anywhere) → §2.6.
12. **Build-and-shelve is a systematic pattern** [R7]: opponent pool (built, unwired), Go-Exploit restarts (built, unwired), categorical head (built and now tested; exact HL formulation rejected for this wave), symmetry averaging (built, now in S1), D1 (mild winner, under S1 recalibration). "The bottleneck is not ideas or code; it's wiring decisions." Standing check adopted: every roadmap item gets a wired-by target or an explicit shelf-reason.
13. **Over-invested in certainty, under-invested in compounding** [R7] — the one-line diagnosis spanning items 1/3/12: the measurement layer is stronger than the learning loop it measures.
14. **Statistical overclaims in our own report** [R6][R7]: pooled turn-4 p-value (§2.1), λ winner's curse, unmeasured compression trend, backwards pentanomial rationale, citation bugs — all with written fixes (§4.1).

---

## 6. ANSWERS TO OUR EIGHT QUESTIONS (four-review merge)

| Q | Merged answer |
|---|---|
| Q1 promotion | 4/4 unblock. Mechanism: regression-protection gate (−10/+15 or two-positive-gates+clean-anchor) for producer now [R1][R2][R3]; R4's gateless-EMA as the staged end-state after value overhaul (§9-C4); strict bar for public champion; automatic rollback after two external/population regressions [R3]; don't burn 900–1200 games forcing +20 through a +30 gate [R3]. |
| Q2 plateau | Current order: denoise/calibrate search → test n128/adaptive n256 → fresh scalar A1 → promote → pool/restarts → separately ablated aux/reanalyze → bigger net only when earned. The tested categorical formulation is not on this wave's path. |
| Q3 compression | Discriminators (union): sims-ladder + n=256-vs-64 [R2]; producer-fed vs gen-3-fed windows [R3]; external population eval [R3]; calibration by decision type [R3]; diversity telemetry [R2]. A0 tested and rejected the evaluated categorical escape; fresh scalar A1 now tests whether data freshness and the exact one-dose recipe move the stable 35M candidate. |
| Q4 external gap | 4/4 live inbreeding warning; R4 adds the imperfect-info component (we play "blind" vs an opponent with memory/belief logic — plausible but its claim about catanatron's belief model is UNVERIFIED, §4.6). Fix = population eval + style-randomized bots + exploiters + position suites + mixed generation + belief work; the error atlas attributes the gap empirically. |
| Q5 value fragility | Structural, but A0 falsified the tested categorical cure.  This wave uses fresh one-dose scalar training with lower value weight/LR and exact provenance; per-game weighting, aux regularizers, uncertainty, and any new value formulation remain isolated follow-ons. |
| Q6 wide roots | Keep cs0.03 as the S1 control, not a production conclusion. S1 selects D6/`c_scale`/D1; S2 selects n64 versus n128; only a qualifying result opens S3 adaptive n256 at `>=40` actions. Later candidates include per-root uncertainty, completed-Q shrinkage, low-evidence/policy-target pruning, and action-bucket calibration. Exact-budget n16 stays dead. |
| Q7 architecture | 4/4 not now, real later. Order: D6/search calibration → promoted stable fresh 35M objective (currently scalar) → conditional scale and isolated action-target cross-attention. Graph-distance bias is deferred until it exists as a tested, separate arm. |
| Q8 unknown unknowns | Four answers, three adopted: error atlas [R1]; benchmark spec + productized strength [R2]; frozen battery [R3]. R4's answer (terminal-reward density / TD(λ) credit assignment) is the one we *partially* decline: legitimate question, but its proposed fix is the known drift channel — staged as conditional V6, with aux-heads-as-regularizers (its safe half) already adopted in V3. |

---

## 7. LITERATURE TO MINE (deduped across R1+R2+R3)

| Source | Steal | Tags |
|---|---|---|
| Stop Regressing (2403.03950) | HL-Gauss/two-hot categorical value; robustness to noisy/non-stationary targets | [R2] |
| KataGo (1902.10565 + KataGoMethods.md) | <30-GPU analogue; target pruning; aux targets; uncertainty playouts; PCR; SWA; window training | [R1][R2][R3] |
| **AlphaStar (Nature s41586-019-1724-z)** | League: diverse strategies + counter-strategies + PFSP sampling; the anti-inbreeding blueprint | [R3] |
| **Adversarial policies beat superhuman Go AIs (2211.00241)** | Fixed benchmarks hide exploitable holes even in superhuman agents → exploiter testing is required, not optional | [R3] |
| Go-Exploit (2302.12359) | Restarts from archived states → independent value targets; "almost tailor-made for your value problem" [R3] | [R1][R2][R3] |
| Regret-guided search control (2602.20809) | Prioritize eval-vs-outcome divergence states | [R1] |
| **MuZero Reanalyse (2104.06294)** | Target refresh on existing data — the principled basis for *selective* reanalyze | [R3] |
| ReZero (2404.16364) | Cheap backward-view/entire-buffer reanalyze | [R1][R3] |
| mctx (repo + qtransforms.py) | Guarantee precondition = frame for cs0.03; qtransform pipeline confirms mechanism | [R1][R2][R3] |
| MiniZero (2310.11305) | Progressive simulation schedules | [R1][R2][R3] |
| LightZero (2310.08348 + repo) | Reference implementations hub | [R1][R2][R3] |
| ReBeL (2007.13544) / Student of Games (2112.03178) | Information-state principle only | [R1][R2] |
| **Finite Group Equivariant NNs for Games (2009.05027)** | D6 equivariance precedent when we go beyond averaging | [R3] |
| **Learned look-ahead in chess transformers (2406.00877)** | Transformers do learn planning structure — argument for fixing action locality, not abandoning the transformer | [R3] |
| Chess graph representation (2410.23753) | Graph/edge features in AZ-likes | [R2] |
| Gendre/Kaneko Catan (2008.07079); catanatron repo; LLM-Catanatron (2506.04651, dismissed) | Prior-art landscape; pin catanatron version | [R1][R2][R3] |
| **Willemsen et al. — value targets in AlphaZero (Soft-Z / A0GB)** | Independent support for our λ-blend result; taxonomy of value-target constructions for the V-stages | [R4][R5 — both name it; verified real] |
| **AlphaZe∗∗ (Frontiers in AI, 2023)** | Masked AlphaZero is competitive in imperfect-info games (Stratego/DarkHex) — the load-bearing cite for R5's "masked AZ is enough" position and our current regime | [R5][US] |
| **Learning Diverse Risk Preferences in Population-based Self-play (2305.11476)** | Risk-profile diversity for league experiment nets | [R5] |
| **Deep Catan (Cazenave et al.)**; kvombatkere/Catan-AI | Additional external baselines for the arena, if runnable | [R5] |
| **fishtest normalized-Elo doc; vdbergh/pentanomial** | The formal framework for the −10/+15 re-spec; independent endorsement of our pentanomial machinery | [R5] |
| DreamerV3 (Hafner et al.) | Two-hot precedent for V1; its symlog component rejected for bounded returns (§10) | [R4] |
| DeepNash / R-NaD (Stratego) | Evidence AZ-style search struggles in high-uncertainty imperfect-info; R4 itself concedes conversion would abandon our MCTS investment — principle only | [R4] |
| ⚠️ "Treant-Gumbel", "Lambda-Reachability" | Cited by R4; **could not be verified to exist** — do not build on these until located | [R4, flagged] |
| **MAPLE (2026)** | Multi-state root aggregation for imperfect info — the PIMC-fixing middle ground; escalation path for contested belief half (§4.6-b) | [R6] |
| **BetaZero** | Belief-space planning: progressive widening + policy-prior sampling to control belief branching | [R6] |
| **AlphaGateau** | Graph chess net: node features → value, edge features → policy; faster learning than CNN AZ — the design precedent for action-entity binding (§4.9) | [R6] |
| **Leela Zero gating discussions** | Community evidence: ~50–52% gate thresholds beat 55%; no-gating riskier; rock-paper-scissors cycles under latest-vs-latest eval | [R6] |
| **AZ scaling-laws work** | Strength scales with params when not compute-bottlenecked; larger nets more sample-efficient — consistent with our params^0.88 note [US] | [R6] |
| **MuZero-Reanalyse (2104.06294) / EfficientZero-v2 SBVE (2403.00564)** | Recompute bootstrap components with current/lagged net — Reanalyse is the largest single contributor in its ablation (92→240); the basis for queue #2 | [R3][R7] |
| **Coulom WHR (2008)** | Whole-history rating over every game played; verified gap — no AZ project uses it | [R7] |
| **Czarnecki spinning-top (2004.09468) + Balduzzi Nash averaging (1806.02643)** | Cyclic strategy density peaks at middling skill; Elo-vs-own-lineage is the eval Nash averaging corrects — the theory behind pool + population rating | [R7] |
| **Long et al. AAAI 2010 (disambiguation factor)** | The instrument that adjudicates the belief split (§9-T1); never computed for Catan | [R7] |
| **AlphaExploitem (2605.09150) / L2E (2102.09381)** | Exploiter training vs target opponents without Nash regression; board-game-AZ gap verified open | [R7] |
| **KataGo uncertainty-weighted playouts (docs/PR #449)** | Backup-side weighting (weight=min(cap, a·err^b), ~20–60 Elo at low playouts) — the D2 lesson: leverage is in backups, not the qtransform | [R7] |
| **MCGS (2012.11045) / Speculative MCTS (NeurIPS 2024)** | Subtree reuse (+69/+310 Elo) and inter-decision parallelism (≤5.81×) — throughput items behind the science | [R7] |
| **Uncertainty-Guided AZ (NeurIPS 2025) / Epistemic MCTS (ICLR 2025)** | Branch from high-uncertainty roots + ensemble value labels (47–58% sample-efficiency on Go) — nearly wired given our f71 tooling | [R7] |
| **Deep Catan (Cazenave, AAAI 2022)** | UPGRADE from R5's "baseline" framing: prior AZ-style expert-iteration attempt on 4p Catan (internal ablations only, never vs catanatron) — cite for novelty hygiene | [R5][R7] |
| **mctx issues #66 / #108** | #66: maintainer declines Gumbel+chance (gap confirmed); #108: our rescale-noise symptom class, open and unanswered (corroboration for the note) | [R7] |
| ⚠️ lc0 "removed gating" narrative | Community-analysis grade only per R7; R8 states lc0 gated at "promote unless worse than −50 (later −150) Elo" citing project history — directionally consistent; still verify before quoting | [R7, self-flagged][R8] |
| **Kumar et al. (2010.14498)** | Effective-rank collapse from self-referential regression targets; compounds with reuse; vanishes with pure MC targets — THE mechanism under our value-fragility "law" | [R8] |
| **RGSC (Tsai et al., ICLR 2026, 2602.20809 + rlglab/rgsc)** | Regret-prioritized restarts un-stick a *converged* AZ (69.3→78.2% where Go-Exploit flatlined) — upgrade target for our f71 tooling | [R1][R8] |
| **Territory Paint Wars (2604.04983)** | "Competitive overfitting": self-play winrate flat at 50% while generalization collapses 73.5→21.6%, invisible internally; fixed-opponent 20% mix is the fix | [R8] |
| **Wang et al. adversarial KataGo (2211.00241) + Tseng hardening (2406.12843)** | Exploitability-probe methodology (>97% exploits at <14% compute); 18% frozen-opponent hardening mixture; blind spots persist 9 iterations | [R3][R8] |
| **Kao et al. Gumbel-MuZero-2048 (TAAI 2022)** | THE prior art that narrows our Gumbel+chance novelty claim — linked from mctx #66 | [R7][R8] |
| **Neumann & Gros scaling (2210.00849, 2412.11979)** | Fresh-data budget scales with params; inverse scaling when end-game states over-represented | [R8] |
| **VISA (2301.11857)** | Max-value-disagreement symmetry variants as value-training points (−50% value generalization error) | [R8] |
| **Anthony ExIt (1705.08439) + ELF OpenGo (1902.04522)** | Fixed-point framing of compression; ~200 Elo per rollout-doubling even at convergence — mechanism (B) | [R8] |
| **"Survive or Collapse" (2605.22217)** | Keep *a* gate: data-admission gating is the binding stability constraint — supports ruling C4 against fully-gateless | [R8] |
| **Keizer et al. (EACL 2017)** | Negotiation-only DRL beat humans 81.8% at the trade subtask — the 4p+trade goalpost | [R8] |
| **Batch SH (RLJ 2024, 2406.00424); Cazenave "SH Using Scores"** | Advance-first≡serial SH at large batch; stockpiling-vs-discarding — the exact-SH confound check | [R8] |

Positioning: no reviewer found a public leak-free AZ-class 2p no-trade Catan system above catanatron_value; phrase as "we did not find" [R1][R2] and build the leaderboard ourselves [R3].

---

## 8. EXECUTION SEQUENCE

**Phase A — now (unblock + measure + free Elo; several items literally free):**
1. Benchmark spec + frozen battery skeleton + **finish-line definition** (§2.6). [R2][R3][R7]
2. Re-spec gate (−10/+15, α=β=0.05, 300g n=16 producer; +30 for turn announcements only); rollback triggers pre-set. [unanimous]
3. Read the running 1000-game external panel → confirmation gate on ONE candidate (lean gen-4) → promote as producer. [ME][R3][R6][R7]
4. **Producer-vs-gen3 window A/B** (parallel windows, disjoint ledgered seeds); canary lane as the lightweight fallback. [R3][R6][ME]
5. **WHR fit over every game ever played** (free, 1–2 days, resolves the compression question). [R7]
5b. **Diagnostics bundle** (~1 GPU-day): search-SNR probe + rollout-doubling + diversity telemetry + noise-vs-spread trend — directs Phases B–C among the three plateau mechanisms. [R8]
6. **Anchor-refresh protocol**: new anchor from current window's held-out seeds each generation; old anchors kept as a longitudinal series; anchor = drift tripwire only, never a promotion signal. [R7][R8]
7. Population arena + sims-ladder standing job + Nash-averaged rating; style panel from recipe-matrix rejects + v3a (free checkpoints). [R1][R2][R3][R7]
8. Complete the bounded S1–S3 selection chain: S1 D6/`c_scale`/D1 → S2 n64-vs-n128 → conditional S3 adaptive n256 at `>=40`-action, always-full roots. Render an external-test config only from the binding artifacts; no n128/n256 or D6 default is pre-approved. [reviewer hypothesis + local binding protocol]
9. λ=0.5 **direct match** (if checkpoint survives) + provenance audit + pentanomial pair-correlation measurement + chronicle citation fixes. [R1][R6][R7]
10. Verify steal-observability (§4.6 fork) + symmetry-augment status [US] + **disambiguation-factor measurement** (one afternoon). [ME][R7]

**Phase B — extract-what-we-paid-for + distribution change + value repair (items run concurrently on different resources [§9-C8]):**
11. **Reanalyze-lite** (V0, queue #2): overwrite archived-λ v-components with current-champion forwards on the stored window, retrain one dose. If the anchor moves → schedule full root-search reanalyze. [R7]
12. **A1 scalar candidate:** A0 completed and rejected the tested HL-Gauss formulation; run one exact fresh scalar-MSE dose on the 35M model.  C0 is closed; any future categorical formulation needs a new predeclared mechanism test. [local A0 overrides R2/R3/R4/R7 prior]
13. Data mix 75–80/10–15/5–10; wire pool (AlphaStar 35/50/15 PFSP template [R7]) + revive f71 restarts + **exploiter lane (10–20% vs catanatron_value/AB3, our search targets)** [R7]; error atlas + held-out high-regret suite.
14. **Raise the expert, R8-sequenced**: consume S1's binding D6/`c_scale`/D1 result → consume S2's binding n64-vs-n128 result → run adaptive n256 only if S2 qualifies S3 and only at `>=40`-action, always-full roots → leave `p_full=.4` as a separate, predeclared future arm. [R7][R8][local S1–S3]
15. V2 per-game weighting + **flywheel value-loss weight 0.25–0.5** (MuZero-Reanalyse precedent [R8]). 15b. EMA/SWA checkpoint-averaging side-test [R4]. 15c. LR 0.5×/2× flywheel arms [R8]. 15d. **1000-game neutral-harness panels in catanatron's own engine** (the outside world's number [R8]).

**Phase C — structural upgrades:**
16. Deduction tracker (uncontested half of §4.6) → features + spectra + aux heads (start weight ≈0.25 [R4]); gate ± deduction. Probabilistic-belief half waits on the df number + the ~10-Elo-stall trigger (§9-T1). [R1][R2][R4][R7][ME]
16b. **Exploitability probe** (3–5 GPU-days): adversary vs frozen champion — closes or opens the belief-state question decisively; also the strongest external-validity instrument available. [R8]
17. V3 aux weights on **+ TD-horizon heads on realized outcomes** (§9-C6-rev); V4 uncertainty head → **backup-side weighting first** (KataGo lesson), completed-Q shrinkage second. [R7]
18. Root cap/target pruning; D1 back on; full selective reanalyze (graduated from #11); `c_scale` re-sweep after selected scalar A1/value-stability evidence (with direction determined by measurement, not by the rejected value-head hypothesis; §4.7); temperature-schedule A/B. [R7; local A0]
18b. If scalar A1 and the independently tested value-discipline follow-on are stable and two consecutive promoted turns are clean: pilot gateless EMA-pull deployment, with the gate demoted to an async tripwire (§9-C4). [R4, staged; local A0]

**Phase D — after ≥10M fresh rows + stable value + plumbing done:**
19. 80–100M equal-exposure fresh-data A/B (de-risked, not approved, by the historical 87.85M stress). 20. Architecture v2 (action-target cross-attention as the isolated experiment; graph-distance bias only after a runnable separate arm exists; full D6 equivariance skipped), multi-seed. [unanimous+R7]
21. Conditional: bootstrapped TD(λ)/n-step machinery, anchor-tripwired (§9-C6-rev half b). [gated]

**Ongoing:** typed configs + config-hash registry; eval server/parallel search; subtree reuse; Rust-batched symmetry; kill-list "closed under assumptions" migration. [3/3]

---

## 9. REVIEWER CONFLICTS & TRACKED OMISSIONS — MY RULINGS

**C1. Value changes: one package (R1) vs isolated staged (R2, R3 implicitly).**
Ruling unchanged: **staged.** Recipe-B's 6-variable confound cost us a generation of un-attributable confusion [US]. R3's designs (scalar control arm, one-dose, single additions) side with staging. Escalate to pairs only with a stated interaction hypothesis.

**C2. Priority: belief (R1 #2) vs the historical categorical-value-first recommendation (R2, R3-by-omission).**
Updated ruling after A0: **fresh scalar A1 first.** The exact 33-bin HL-Gauss
formulation was wired, tested, and rejected for this wave. Belief still needs
instrumentation + regenerated data, so its verification starts Phase A but its
gate lands Phase C; any future categorical formulation requires a separately
predeclared mechanism test and cannot silently replace the scalar A1 contract.

**C3 (updated after R6). Restart-data weight: 5–10% (R1/R2) vs 10–25% (R3) vs 20–40% (R6).**
Reviewer range now spans 5–40%. Ruling unchanged in shape: **enter at 10%**, climb the ladder (→25% →40%) only while the held-out high-regret suite improves with no external-panel damage. R6's value-only-rows fallback (§4.5) is the intermediate step before cutting the fraction. Entering low stays right: two of three proposers built falsifiers around over-weighting.

**T1 (revised after R5). Belief tracking is a genuine reviewer SPLIT, not an omission.**
Tally: for = R1/R2/R4; silent = R3; lean-against = R5 ("masked AlphaZero is enough for base play," AlphaZe∗∗ evidence, "no ReBeL/SoG-style re-solving yet"). Ruling: split the workstream (§4.6) — **exact-deduction features proceed on schedule** because they are recovered *public* information, which every reviewer's frame endorses (including R5's); **probabilistic belief machinery beyond deduction goes behind error-atlas evidence** — it no longer rides in on the 3 votes alone. This is the plan's first true reviewer split; future reviews should be read specifically for evidence on it.

**C4. Promotion mechanism: soft gate −10/+15 (R1/R2/R3) vs gateless EMA deployment (R4). [NEW]**
Ruling: **soft gate now, gateless EMA as the staged end-state.** R4's own-evidence problem: we already ran the ungated-compounding experiment by accident — rounds 6–11 without gates produced the +69% value drift and 37% gates [US]. That was with lineage-init and tiny windows (EMA smoothing would help), but the failure class is proven in *our* system while gateless is proven in KataGo's — a system with the stable value head we don't yet have. Sequence: −10/+15 gate → stable scalar A1 + an independently tested value-discipline follow-on → two consecutive clean promoted turns → pilot EMA-pull with the gate as async tripwire. R4's EMA/SWA *weight averaging* itself is adopted early as a cheap candidate-smoothing test. Its "48h to >50% external" success criterion is discarded as uncalibrated.

**C5. Categorical value bin count: 128–256 (R4) vs ~51–64 (R2/[ME]). [NEW]**
Ruling: **~64 bins.** With only ~16k independent game outcomes per window, 256 bins over a bounded [−1,1] support fragments the label mass for no representational gain; Stop-Regressing-scale results don't require it. Revisit upward only if VP-margin support needs resolution.

**C6 (SUBSTANTIALLY REVISED after R7). Dense credit assignment — my blanket downgrade was half wrong.**
Original ruling: all short-horizon/TD targets downgraded as disguised self-distillation. R7's evidence forces a split: KataGo's TD-horizon heads train on **realized** trajectory outcomes — not our own value estimates — so the drift objection doesn't apply to them, and the −190-Elo-without ablation is the largest single documented aux effect [R7-LIT]. **Revised: (a) realized-outcome TD-horizon heads UPGRADED (queue #11, rides next retrain); (b) bootstrapped-target machinery stays gated** — and is now *more* suspect, not less: R7 showed our shipped λ-blend already runs a bootstrapped channel (archived generating-net root values) and it plausibly co-caused the +69% drift. The general lesson recorded: evaluate credit-assignment proposals by **target source** (realized vs self-estimated), not by horizon.

**C8 (refined by R8). Queue priority: reanalyze-lite + raise-the-expert (R7) vs pool-first (R1/R2/R3) vs search-side-first (R8). **
R7 historically ranked reanalyze-lite #1; R8 historically ranked promote → pool → symmetry+cs-regrid → RGSC → reanalyze+HL-Gauss because its SNR theory says the search side is the root cause. A0 now overrides the HL-Gauss recommendation for this wave. Ruling: **these run on different resources, so the serialization question is mostly false** — training-side (reanalyze-lite, the exact scalar A1 dose, and only later separately predeclared follow-ons) and generation-side (pool, symmetry, re-grid) may proceed concurrently in Phase B [ME]. The genuinely new discipline adopted from R8: **run the ~1-GPU-day diagnostics bundle FIRST** — the three plateau mechanisms have different remedies, and one day of measurement beats two weeks of the wrong lever. R8's sequencing *within* the search side is adopted verbatim (denoise → re-tune cs → raise sims), superseding a bare n_full raise; the binding implementation is S1 → S2 → conditional S3.

**Gate-number note [ME]:** R8 says elo1=+10 where six prior reviews (and R7) say +15. At a true +20 both pass quickly; fishtest sizes elo1 to achievable effects (2–5 Elo), which argues low. Keep **−10/+15 as written** (broad consensus, marginally stronger evidence per promotion); treat +10 as an acceptable variant if promotions start timing out. R8's add-ons adopted: every 3rd promotion gets a 200-game n=64 non-regression confirmation (closes the low-sim/production gap gen-4 exposed), bucketed win rates (phase/opening/blowout-vs-close) with per-bucket regression veto, and Nash-averaged trend consulted at promotion. R8 also independently supports C4: "keep *a* gate — data admission gating is the binding stability constraint" (Survive-or-Collapse).

---

## 10. REJECTED / DOWNGRADED (kept on record so later reviews can challenge)

- ⏬→⚠️ Short-horizon/TD targets SPLIT after R7 (§9-C6-rev): realized-outcome TD-horizon heads UPGRADED (queue #11); only bootstrapped-target machinery stays gated. [ME, revised]
- ❌ Full D6-equivariant architecture (R4/R5 leanings): R7 settles it — augmentation + root averaging captures most of the value; equivariant nets skipped. Graph-distance/adjacency bias is **not** a cheap ride-along exception: it remains deferred until a runnable module, tests, and its own predeclared causal arm exist. [R7; local causal-isolation rule]
- ❌ **Symlog transform** (R4): built for unbounded/extreme-variance returns; ours are bounded [−1,1]. Pure cargo-cult here. [ME]
- ❌ **Gateless EMA deployment NOW** (R4): our rounds-6–11 postmortem is a direct counterexample in this system; staged only after stable scalar A1, an independently tested value-discipline follow-on, and two clean promoted turns (§9-C4). [ME][US]
- ⏬ **IS-MCTS / PIMC root determinization** (R4): mostly mooted by exact deduction in 2p; PIMC strategy-fusion/non-locality flaws unmentioned by R4; belief features + posterior spectra first. Revisit only on error-atlas evidence (§4.6). [ME]
- ⚠️ Literal opening book (R2): only if benchmark boards repeat; default = distilled opening head. [ME]
- ⚠️ "WDL" value bins (R3): no draws in 2p Catan. The historical categorical proposal used win/loss bins plus a distinct truncation class and VP-margin auxiliary; A0 rejected the evaluated HL formulation for this wave, so A1's primary search readout remains scalar. [ME; local A0]
- ⚠️ Full AlphaStar league complexity/PFSP now (R3 itself warns): start with the simple 4-slot mix. [R3][ME]
- ⚠️ R4's 128–256 value bins: ~64 (§9-C5). ⚠️ R4's success criteria (48h → >50% external; augmentation → "near zero" orientation std): directionally useful, numerically uncalibrated — our own falsifiers govern. [ME]
- ❌ Reverting to Gumbel paper defaults (4/4 against — even R4's dissent doesn't propose reverting).
- ❌ Exact-budget SH n16 resurrection (R3's condition; R4's "structurally misaligned" read concurs).
- ❌ Extending gates to 900–1200 games to force +20 through +30 spec (R3 explicitly; R4 implicitly).
- ❌ Full ReBeL/SoG/R-NaD machinery (3/3 of the reviews that discussed it, incl. R4's own concession).
- ⏬ ReSCALE as guidance (R1; R4 cites it favorably but generically — no new argument). ⏬ `external_best` as a fourth registry label (R2). [ME]

---

## 11. OPEN QUESTIONS FOR NEXT REVIEWS

Answered since v3: ~~HL-Gauss projection details + λ-blend compatibility~~ → project the blended targets, expectation readout [R4] (historical formulation; A0 rejected it for this wave); ~~belief aux head starting weight~~ → ≈0.25 [R4]. Historical timing answers are superseded: ~~when to raise n beyond 64~~ → S1–S3 now independently adjudicate n64/n128 and conditional adaptive n256 before the wave; ~~91M re-test protocol~~ → C0 is closed, and scale waits for a promoted stable 35M scalar candidate plus completed search calibration and `>=10M` fresh audited rows. [local A0/C0/S1–S3]

Answered since v4: ~~PFSP weighting scheme~~ → PFSP with league refresh + risk-diversity nets, activation trigger defined [R5+ME §4.5]; ~~bin design~~ → partially: add explicit truncation category, blend in distribution space [R5]; ~~window reuse target~~ → R5 confirms KataGo's 4–8× range (we run target-reuse 3 [US]; modest headroom, not urgent).

Answered since v5: ~~loss weighting of bot-game rows~~ → largely: train only OUR side's rows in bot games; value-only-rows demotion as the fallback [R6]; ~~pre-set external rollback bound~~ → P(external ΔElo < −25) > 0.9 or two consecutive declines [R6]; ~~escalation path for contested belief half~~ → MAPLE multi-state aggregation at hidden-info roots, BetaZero-style widening [R6]; ~~value-arm hyperparameters~~ → weight 0.25–0.5, value LR ≈0.3× torso, game-level splits [R6].

Answered since v6 [R7]: ~~anchor staleness~~ → refresh per generation, keep the series (and the plateau inference gets an asterisk until then); ~~exploiter method~~ → direct games vs the target bots with our search targets (AlphaExploitem precedent), plus recipe-matrix rejects as free style opponents; ~~compression-trend interpretation~~ → don't interpret until the WHR fit exists; ~~how to adjudicate the belief split~~ → measure the disambiguation factor.

Answered since v7 [R8]: ~~HL-Gauss σ~~ → σ ≈ bin width, 31–51 bins; ~~two-hot-vs-HL-Gauss~~ → two-hot underperforms MSE, convert the shelved head; ~~D2/James-Stein tuning~~ → closed form λ* = v²/(s²+v²) + retry with a weight cap; ~~which plateau mechanism~~ → the diagnostics bundle discriminates (A)/(B)/(C); ~~is masked-AZ enough (partially)~~ → exploitability probe decides.

Still open:
- Gumbel-compatible target-pruning rule (R4's FPU hint; R7's backup-side lesson; R8 adds forced-playouts+pruning as the shipped mechanism — spec the Gumbel port after V4).
- Does train-time D6 augmentation interact with the categorical value head (two target-side changes — sequencing)?
- EMA decay β for the eventual gateless pilot (R4 says 0.995; untested in our round cadence).
- Verify or refute: does catanatron_value actually do belief/card tracking? (df measurement + exploitability probe largely substitute; reading the source is still one hour.)
- Opponent-pool fraction interaction with the canary lane (both inject non-champion data simultaneously — sequence or cap jointly?) [ME].
- Reanalyze-lite target hygiene: current vs lagged/EMA reanalyzer net [ME; R8's Kumar mechanism argues for lagged — self-reference is the disease].
- What "won" means numerically for finish-line stage (b) (colonist.io variant + human trials — protocol undefined). [R7][R8 both raise, neither specs.]
- If the 1000-game panel lands at parity (~48–52%): what replaces "close the −30 gap" as the organizing goal? (Candidates: prove-superiority-with-power in the neutral harness; then 4p+trade scope decision.) [R8, contingent.]

---

## 12. CHANGE LOG

- **2026-07-07 (v8, after R8 = second live-verified sweep):** ADOPTED AS WORKING HYPOTHESIS: the **SNR-decay unified plateau theory** — the min-max rescale's fixed logit budget means the improvement operator's signal fraction decays as the policy sharpens; our c_scale fix and our plateau are the same object; predicts the compression curve; comes with a falsifiable ~1-GPU-day **diagnostics bundle** (SNR probe / rollout-doubling / diversity / noise-vs-spread) that now runs before any big remedy spend. GOAL-LEVEL: **parity finding** — 45.7%/200 vs catanatron_value is p≈0.22 vs 50%; we may already be at parity; 1000-game panel + **neutral-harness re-runs in catanatron's own engine** become the decisive numbers. VALUE: Kumar rank-collapse mechanism under the fragility law; two-hot-underperforms warning (convert shelved head to HL-Gauss, σ≈bin width); flywheel value weight → 0.25–0.5 (Reanalyse precedent); banked-corpus (32.6M rows) value-only reanalyze scoped. DATA: RGSC prioritization upgrade for f71 (un-sticks converged AZ where Go-Exploit flatlines); Territory-Paint-Wars competitive-overfitting evidence; policy-surprise weighting; diversity-strangulation diagnosis (deterministic search + argmax-after-90). SEARCH: denoise→re-grid→raise-sims sequencing supersedes bare n_full raise; D2 retry with weight cap; James-Stein closed form. INSTRUMENTS: **exploitability probe** (adversary vs frozen champion — decides the belief question + external validity); anchor demoted to tripwire-only (gen-4 signature failed to predict its gate). HYGIENE: Kao TAAI-2022 narrows the Gumbel+chance novelty claim; exact-SH confound (stockpiling vs discarding — check before publishing); one-sided-p correction; AB5-quirk unverifiable; Willemsen cite fix; repo-reproducibility (~2 days). RULINGS: C8 refined (diagnose-first; concurrent resource lanes); gate-number note (−10/+15 kept, +10 acceptable; every-3rd n=64 confirmation + bucket vetoes added); C4 reinforced by Survive-or-Collapse. ~12 new literature entries.
- **2026-07-07 (v7, after R7 = the live-sweep expert review, summary + full report both read):** Highest-evidence review; its frame adopted: "over-invested in certainty, under-invested in compounding." BIGGEST CHANGES: (1) **Reanalyze-lite to queue #2** — R7's mechanism finding that our λ-blend trains toward the *generating net's archived root values* (self-distillation amplifier, plausible co-cause of the +69% drift); fix costs zero new games. (2) **Plateau premise gets an honesty asterisk** — flat anchor can't distinguish distilled-window from stale-anchor; anchor-refresh protocol adopted (answers our oldest open question). (3) **Raise-the-expert to queue #6** (n_full 128, p_full 0.4, root symmetry on; kill-criteria attached) — we never spent the 13–19× search savings; also fixes 7.7% policy-row starvation. (4) **C6 substantially revised (partial self-reversal):** realized-outcome TD-horizon heads upgraded (KataGo −190 Elo ablation); bootstrapped machinery stays gated, now with proof not suspicion. (5) WHR ladder fit + Nash-averaged population rating + finish-line definition (no human calibration of catanatron exists) + 4p-extension audit + df measurement as the T1 instrument (+~10-Elo-stall trigger) + exploiter lane (verified open gap) + statistics hygiene (λ winner's curse, backwards pentanomial rationale, unmeasured compression trend) + chronicle citation fixes (Charlesworth/CatAnalysis/Deep Catan/HexMachina) + c_scale framing correction (mechanism is the finding, not the constant) + backup-side-not-qtransform uncertainty lesson + temperature-schedule test + build-and-shelve standing check. New ruling C8 (queue re-rank, concurrent-not-serialized pool). ~14 new literature entries.
- **2026-07-07 (v6, after R6):** R6 = highest-reliability review (calibration structure, accurate quotes, correct mctx defaults — fixes R4's misread). TURNED ON US, adopted: turn-4 pooled claim demoted to "suggestive" (heterogeneous sim budgets + optional extensions + post-hoc pooling — the confirmation-gate pre-step is now doubly mandated); λ0.5-vs-gen2A upgraded from audit to direct H2H match if the checkpoint survives; "science ladder vs champion ladder — the champion ladder should be ruthless" adopted as registry principle. ADOPTED: canary generation lane (20–30% candidate data during async panels); quantified external rollback bound; V1 restructured as a 4-arm parallel tournament (C1 discipline preserved — parallel singles); value LR 0.3× torso + loss weight 0.25–0.5; **game-level validation splits everywhere** (retro-explains round-11 val leak); own-side-rows-only for bot games + value-only-rows fallback; V7 two-phase retest (conditional); opening roots n=256/512 + reanalyzed opening corpus; all-pairs cross-play last 8–12 nets; AlphaGateau node/edge design precedent; MAPLE/BetaZero as the contested-belief escalation ladder. UPDATED RULINGS: C3 range widened (5–40% across reviewers; enter-at-10% ladder unchanged), T1 tally 4-for/1-silent/1-against with R6 explicitly endorsing our cheap-first split. Gate re-spec precedent deepened (KataGo optional gatekeeping, LZ 50–52% experience); consensus tracker now 6-review.
- **2026-07-07 (v5, after R5):** R5 mostly *converges* (kept masked-AZ core, cs0.03, league, categorical value, D6, one-dose discipline — the plan's spine is now 5-review stable). ADOPTED: categorical λ-blend in distribution space (V1 detail); explicit truncation bin in the value support (sharp catch — F3 truncated games exist [US]); D6 test-time averaging promoted to default; PFSP league mechanics + refresh + risk-diversity nets with my activation trigger; D2-jamesstein-retry-with-learned-uncertainty (§4.7-2); low-sim-gate + n=64 confirmation protocol; new arena baselines (Deep Catan, Catan-AI); normalized-Elo framing for the gate (5th vote). SHELVED: progressive widening at interior chance nodes (lazy is validated [US]); adjacent-state value-smoothness penalty (V5 shelf). MAJOR RULING REVISION — §9-T1: belief is now a genuine reviewer SPLIT (3 for / 1 silent / R5 lean-against) → workstream split into (a) exact-deduction PUBLIC-info features (uncontested — they're recovered public information, endorsed by every frame including R5's) proceeding on schedule, and (b) probabilistic belief machinery (contested) moved behind error-atlas evidence. R5 reliability: medium (real sources, some mismatched attributions).
- **2026-07-07 (v4, after R4):** R4 audited hard (reliability note §0: c_visit/c_scale misread, two unverifiable citations, uncalibrated success criteria) — mechanisms mined, facts quarantined. ADOPTED: frozen-corpus 3-epoch mechanism probe as V1's primary test + chained 91M re-probe; MSE-memorization explanation of the 91M blowup as the testable hypothesis; "cs0.03 = noise-era compromise" trajectory (§4.7 — expect optimal Q-contribution to rise post-V1); EMA/SWA checkpoint averaging as cheap smoothing test; belief aux weight ≈0.25; graph-distance attention biases (Phase D detail); v3b-failure-masked-by-MSE attribution → arch rerun explicitly post-V1; label-banking verification item. NEW RULINGS: C4 (gateless EMA rejected now — rounds-6–11 drift is our own counterexample — staged as post-V1/V2 pilot), C5 (64 bins not 128–256), C6 (TD(λ) graduates from rejected to conditional V6 — honest update under two independent votes). REJECTED: symlog (bounded returns), IS-MCTS/PIMC-now (exact-deduction moots it; strategy-fusion unaddressed). Consensus tracker moved to 4-vote counts; categorical value now 4/4.
- **2026-07-07 (v3, after R3):** Consensus tracker added (most core moves now 3/3). NEW: producer-vs-gen3 window A/B adopted as the *default* promotion protocol (§2.4) — promotion becomes a controlled experiment; frozen benchmark battery + position suites merged into §2.6/§4.2; exploiters upgraded from "eventually" to required (adversarial-Go 2211.00241); hard-negatives slot in pool; held-out high-regret suite; selective-reanalyze refinement; value-optimization knobs (Huber/log-cosh/weight-decay); completed-Q shrinkage toward prior/root; R3's Q3 discriminator set; 5 new literature entries. Rulings: C3 (restart weight 10%→25% ladder), T1 (R3's belief silence tracked, belief stays). WDL corrected to win/loss×VP-margin [ME]. Promotion precondition tied to the running 1000-game panel [R3].
- **2026-07-07 (v2, after R2):** Ruled synthesis; dual registry + confirmation-gate pre-step [ME]; benchmark spec; categorical value V1; staged-vs-bundled ruling; exact-deduction belief analysis [ME]; root-cap≈target-pruning merge [ME]; kill-list assumptions column; §9/§10 added.
- **2026-07-07 (v1):** Created from R1 (`first55`).
