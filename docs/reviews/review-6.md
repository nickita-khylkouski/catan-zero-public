## Executive summary — the 10 things that matter most

1. **Your +30 Elo promotion gate is now wrong for the continuous loop.** It was appropriate when each generation gained ~50 Elo; it is now blocking real +15–25 Elo improvements from compounding. KataGo used a light gate — ≥100/200 games — and its docs explicitly say the loop can run without gatekeeping, mainly using it for debugging and regression protection [KataGo paper; KataGo docs]. I would switch to regression-protection gating immediately.

2. **The turn-4 signal is suggestive, not a clean “verified promotion.”** Pooling 400 n=8 games and 600 n=16 games into “528/1000, p≈0.04” is too casual under different search budgets and optional extensions. It is enough to justify canary/promotion under a softer gate; it is not enough to claim a fourth certified improvement.

3. **Your value-head problem is under-fixed.** The repeated failures look like noisy, correlated value-target overfitting with MSE, not a mysterious Catan-specific law. Replace scalar MSE with categorical / two-hot / HL-Gauss value classification, reduce value LR/loss, split and sample by game, and use target networks/reanalysis. 2024 evidence directly supports classification losses for noisy/non-stationary value learning [Stop Regressing].

4. **You are partly optimizing for clean science rather than maximum strength.** If the λ=0.5 gen-2 arm really got 59.0% vs gen-1 and gen2A got 57.0%, the maximum-strength move was to direct-match λ vs gen2A and champion the winner. Keep clean controls, but the deployed champion should be the strongest statistically credible arm.

5. **The fastest plateau escape is new distribution, not more epochs.** Promote non-regressing candidates, wire the opponent pool, add Go-Exploit-style high-regret restarts, and selectively reanalyze high-impact states. Go-Exploit is especially relevant because it produces shorter trajectories and more independent value targets [Go-Exploit]; ReZero supports periodic/targeted reanalysis for stale MCTS targets [ReZero].

6. **Special-case wide opening roots.** Your biggest pathology is 54-wide near-tied roots with noisy Q. Spend n=256/512, symmetry averaging, and/or reanalysis on opening/wide roots. There are only a few such decisions per game, so this is likely high Elo per compute.

7. **Your c_scale=0.03 result is credible.** mctx really does complete missing Q-values, min-max rescale, and multiply by default value_scale/maxvisit terms [mctx]. The Gumbel guarantee assumes Qs are correctly evaluated; your 1–2-visit 54-way roots violate that. Keep the small scale and develop shrinkage/noise-floor completed-Q.

8. **Masked observation is a good baseline but probably not the endpoint.** Catan has rich inferable hidden information. Modern imperfect-information work ranges from masked/sampled AlphaZero being surprisingly strong [AlphaZe**] to public-belief systems [Student of Games; ReBeL] and 2026 MAPLE-style multi-state aggregation [MAPLE]. Build cheap Catan-specific public belief features before a ReBeL rewrite.

9. **Do not overread the failed architecture A/B.** One v3b loss says “not free under that setup,” not “architecture closed.” Your own D6-noise measurement plus graph/equivariant game-network results argue that symmetry and action-entity interaction matter [AlphaGateau; FGNN]. But fix value/search/data first.

10. **Your evaluation is strong but too narrow externally.** Add all-pairs cross-play among recent champions, ≥1000-game fixed-bot panels for promoted nets, varied board/opening suites, and population-style evaluation. AlphaStar used league/PFSP specifically to avoid self-play cycles and forgotten exploiters [AlphaStar].

## What I would do first

**Promotion rule:** change to regression protection. Candidate must pass provenance and anchor checks, then promote if P(Elo > 0) is reasonably high and P(Elo < -10) is low — or use SPRT elo0=-10 / elo1=+15 with a 600-game cap. External panels should run asynchronously and block/revert only on significant decline, not on a noisy 200-game flat read.

**Top experiments:**
1. Categorical value head tournament: current MSE vs two-hot vs HL-Gauss, with lower value LR/loss.
2. Opponent pool: 70–80% latest self-play, 10–15% previous champions, 5–10% exploiters/bot-specialists; evaluate with all-pairs matrix.
3. Go-Exploit restarts from high-regret/wide/robber/dev/external-loss states.
4. Opening/wide-root special search and 12-fold D6 averaging where affordable.
5. Selective reanalysis of high-KL/high-width states with n=128/256 targets.
6. Only then progressive simulation and 70–100M scaling.

## Direct answers to your live decisions

**Q1 — Promotion criterion:** promote under regression protection unless the 1000-game external panel significantly refutes transfer. The current +30 gate will hold forever and prevent compounding.

**Q2 — Plateau:** the current window is fully distilled only under current targets. Move the data/target distribution: promotion, opponent pool, restarts, reanalysis, opening-special search.

**Q3 — Compression:** partly expected at fixed search/net/data, but diagnose with n=64-vs-n=128 target disagreement, value calibration by action width, search-vs-raw trend, opening diversity, and all-pairs cross-play.

**Q4 — External gap:** normal internal inflation plus possible inbreeding. A 200-game external read is underpowered; build population evaluation.

**Q5 — Value fragility:** structural fixes are categorical value, lower value LR/loss, game-level sampling/splits, target networks, selective reanalysis, ensembles/uncertainty, and weighted auxiliary heads.

**Q6 — Wide stochastic roots:** your c_scale ablation beats the literature default in this setting. Use noise-floor/shrinkage completed-Q, opening-special budgets, policy-target pruning, and symmetry averaging.

**Q7 — Architecture:** do not rewrite immediately, but D6 and action-entity relation should be retested after value fixes. Scale after ≥10M fresh rows and categorical value.

**Q8 — Not asking:** the biggest missing question is public belief. After fixing the leak, you may have overcorrected to ignorance. A cheap Catan belief tracker may close more external Elo than another generic AZ turn.

**Source notes**

**My judgment**

- Promote turn-4-like candidates under regression-protection unless the 1000-game external panel significantly refutes transfer.
- Opening-special search and cheap public-belief features are likely higher immediate Elo than a general architecture rewrite.
- The value-head failures are mostly noisy/correlated-target overfitting plus MSE/plasticity, not a unique Catan phenomenon.

**Needs experiment**

- Categorical value head in your exact Gumbel/chance-node setting.
- Opponent-pool fraction and whether bot-opponent data improves external without overfitting.
- D6 equivariance/canonicalization after value-loss fix.
- Whether high-budget opening reanalysis improves external catanatron_value results.

**Established in literature**

- KataGo used a light gate and made gatekeeping optional in training docs; it also used auxiliary targets and playout cap randomization for compute efficiency.
- mctx completed-Q transform rescales completed Q-values and applies value_scale/maxvisit_init defaults, supporting your concern about noise amplification at low visits.
- Categorical value learning is supported by 2024 evidence as more robust/scalable than MSE in deep RL.
- Go-Exploit supports restart/search-control data for more independent value targets and better sample efficiency.
- ReZero/MuZero Reanalyze support target refresh/reanalysis for stale MCTS targets.
- AlphaStar/PFSP supports opponent pools to fight non-transitivity and forgotten exploiters.
- Student of Games/ReBeL/MAPLE/BetaZero show the spectrum from public-belief search to practical belief-state aggregation/progressive widening for imperfect information.

**Ranked actions**

| Run | Rank | Change | Hypothesis | Mind changer |
| --- | --- | --- | --- | --- |
| Use champion-init, anchor no-drift check, paired pentanomial n=16 gate. Promote if posterior/LLR indicates non-regression: e.g. P(Elo > 0) ≥ 0.75–0.85 and P(Elo < -10) ≤ 0.05, or SPRT elo0=-10/elo1=+15 with cap 600. Keep previous champion and prior two champions in self-play pool for 15–25% of games. External 1000-game panel runs asynchronously; revert/block only on significant external decline, e.g. P(external Elo delta < -25) > 0.9 or two consecutive promoted candidates declining. | 1 | Change the continuous-loop promotion rule now. | At the current slope, refusing +15–25 Elo candidates prevents the policy-iteration data distribution from moving; a non-regression gate plus external tripwires compounds faster than a +30 Elo gate. | If promoted +20 candidates produce subsequent candidates that lose internally and significantly decline externally, re-tighten the gate and increase opponent-pool proportion. |
| Train 4 arms on the same current window: current MSE; 21-bin two-hot over [-1,1]; 51-bin HL-Gauss over [-1,1]; categorical over outcome plus VP-margin auxiliary. Keep λ=0.5 blend but project blended scalar to distribution. Try value loss weight 0.25–0.5 and value-head LR 0.3× torso LR. Same anchor + n=16 gate. | 2 | Replace scalar value MSE with categorical value classification, and decouple value from policy learning. | Your value failures are noisy, non-stationary regression failures; classification losses mitigate noisy targets and non-stationarity in scalable deep RL [Stop Regressing]. | If classification improves anchor value calibration but loses H2H, inspect search Q calibration by root-width; otherwise make it default before scaling to 91M. |
| Generation mix: 70–80% latest self-play, 10–15% previous champions sampled by PFSP/hard or fvar, 5–10% exploiters/specialists (candidate lines, high-search variants). Store opponent metadata but train a single unconditional policy; for bot games, train only your side’s decision rows initially. Evaluate with all-pairs matrix among last 8–12 nets plus external bots. | 3 | Wire opponent-pool and population evaluation. | The internal/external gap and plateau are partly self-play inbreeding / non-transitivity. PFSP-style opponent sampling prevents forgetting and exposes exploitable weaknesses [AlphaStar]. | If mixed-opponent data lowers internal gates but improves external panel, keep it; if both decline, reduce bot/exploiter share and keep only champion-history pool. |
| Archive states with high search-vs-prior KL, high root value variance, wide legal action count, opening/robber/dev-card decisions, and external-loss positions. Generate 20–40% of games from these states to terminal. Use valid public belief/state reconstruction. Compare equal-row and equal-GPU candidates. | 4 | Use Go-Exploit/high-regret restarts. | Catan’s long horizon and correlated row labels make value learning data-inefficient. Starting from archived high-regret/search-disagreement states yields shorter trajectories and more independent value targets [Go-Exploit]. | If restart data improves value anchor but harms full-game play, lower its fraction or use it as value-only rows. |
| For first placement/road decisions and other ≥40-action roots: n=256/512, 12-fold D6 symmetry averaging if cheap after Rust, and policy-target rows always full-weight. Gate external vs catanatron_value and internal vs champion with only opening-special search changed. Also generate a reanalyzed opening-root corpus from old games. | 5 | Special-case wide opening roots. | A few wide roots dominate losses and target noise; spending more search there has unusually high Elo/compute payoff. | If opening-special improves external but not internal, still keep it for the competition bot; if it worsens due to overfitting/noise, combine with value shrinkage not raw min-max completed-Q. |
| Reanalyze top 5–10% high-KL/high-width/high-uncertainty states with n=128/256 and current champion, not entire 4.6M rows. Replace policy targets and blended root values; train one-dose candidate. | 6 | Selective reanalysis of stale/high-impact states. | The window is fully distilled only with old targets; better targets from current champion/high search can move it [ReZero, MuZero Reanalyze]. | If selective reanalysis gives <1 Elo per million reanalyzed rows, prioritize fresh generation instead. |
| Maintain exact/particle belief over opponent resource counts and dev deck from public action history, costs, discards, steals, and draws. Feed belief summary tokens to net and use belief spectra for steals/dev draws. A/B against masked-only on robber/dev buckets, external panel, and H2H. | 7 | Build cheap public-belief features before heavy imperfect-information search. | Masked observations throw away inferable hidden information; Catan resource/deck beliefs are much cheaper than ReBeL/MAPLE and likely help robber/dev-card decisions. | If belief features improve calibration but not play, try MAPLE-style multi-state root aggregation only for hidden-info-sensitive roots. |
| After value classification, try n_full=96/128 with p_full reduced to keep row/hour constant, and n_fast=24/32. Gate low-sim and production-sim separately; inspect target entropy and root value error. | 8 | Progressive simulation only after value/opening fixes. | Fixed n=64 eventually caps policy improvement; MiniZero reports progressive simulation improves board-game performance, but larger search with bad value can amplify noise [MiniZero]. | If higher search improves targets but not H2H, you are architecture/value-limited; if H2H improves, schedule visits upward. |
| After value classification and ≥10M rows, run 2 seeds each: current 35M; D6-canonical/equivariant or train-aug+test-avg; action-target cross-attention only; 70–100M scaled model. Equal data, equal wall-clock, same gate suite. | 9 | Retest architecture, but later and with better protocol. | D6 equivariance/action-entity interaction should matter, but the earlier v3b test was underpowered and confounded by value training. | If graph/action models repeatedly lose with fixed value loss and enough data, invest in data/search not architecture. |

**Detailed report:** ## Expert review of Catan-Zero

### What you are doing right

You have done several things unusually well: paired color-swapped evaluation, pentanomial SPRT rather than naïve binomial tests, fail-closed provenance after the hidden-info leak, engine-equivalence work, a serious kill-list, and profiler-driven engineering. Those are not cosmetic; most failed self-play projects die from exactly the bugs you catalogued. Your diagnosis that low-simulation gates are more sensitive to network improvement than production-depth gates is also basically right: deep search can wash out differences between priors, while low search exposes prior/value quality [Catan-Zero §8.3].

The Gumbel/c_scale finding is also credible. The mctx implementation’s completed-Q transform replaces unvisited actions by a mixed value, optionally rescales completed Qs to [0,1], then multiplies by `(maxvisit_init + max_visit) * value_scale`, with defaults `maxvisit_init=50`, `value_scale=0.1`, and `epsilon=1e-8` [mctx]. That is exactly the kind of transform that can turn a tiny empirical Q spread at a 54-way root into an overconfident ranking if the Qs are mostly 1–2-sample noise. The Gumbel policy-improvement story assumes correctly evaluated Q-values; your opening roots violate that assumption [mctx]. So: do not go back to c_scale=1.0 because the paper used it. Your ablation beats the default in your game; believe the ablation.

### What you are doing wrong / overclaiming

**1. The promotion gate is now optimizing the wrong objective.** Your +30 Elo H1 gate was appropriate when turns were +49 Elo. It is not appropriate when the loop is producing +15–25 Elo candidates. KataGo’s published gate was light — at least 100/200 games versus current net — and its docs say gatekeeping is optional and mainly useful for debugging / ensuring progress [KataGo paper; KataGo docs]. Leela Zero community simulations argued that too-high gating wastes good small improvements and that ~50–52% can be more efficient than 55%, while no-gating/very-low gating is riskier [Leela Zero discussion]. Your current system is in exactly the “small improvements get rejected forever” regime.

**2. The turn-4 evidence is suggestive, not a clean promotion claim.** “528/1000 = 52.8%, p≈0.04” mixes different simulation budgets, two candidate training procedures, and an optional-extension process. It is enough evidence to justify a regression-protection promotion/canary because the cost of not moving the data distribution is high. It is not enough to write “turn 4 verified improvement.” Keep the discipline that got you here: paired pentanomial model, predeclared gate, and no casual pooled p-values after looking.

**3. You are treating value fragility as a law, but your evidence also screams ordinary target-noise overfitting.** Your value labels have effective independence closer to “games” than “rows”: 3–4M positions from ~16k games are massively correlated, and all positions in a game share a terminal ±1 outcome component. Extra epochs can improve train loss while memorizing game outcomes. That is not exotic; it is expected. The 2024 “Stop Regressing” result is directly relevant: categorical cross-entropy value learning improves robustness to noisy/non-stationary RL targets and scales better than MSE [Stop Regressing]. MuZero also uses categorical/two-hot-style value support for Atari-scale values [MuZero via searched source]. Your value head should not still be plain scalar MSE if value is the load-bearing organ.

**4. You left strength on the table for clean science.** If the gen-2 λ=0.5 arm got 59.0% with LLR 6.12 and gen2A got 57.0%, then promoting the clean control was defensible for a paper narrative but wrong for “build the #1 Catan bot.” The right maximum-strength move was: immediately direct-match λ-arm vs gen2A, promote the winner, and keep the clean-control lineage as a labeled experiment. Going forward, separate “science ladder” from “champion ladder.” The champion ladder should be ruthless.

**5. Your external panel is too weak to support some of the conclusions you draw from it.** A 200-game 45.7% vs 41.0% comparison with overlapping CIs is not evidence of external regression; it is evidence that you do not know. Conversely, the internal +150 Elo vs external +70 Elo gap is a real warning sign, because sequential self-play Elo often inflates and non-transitivity can hide under latest-vs-latest gates. AlphaStar used league training specifically because pure self-play can chase cycles, and PFSP focuses training on opponents the agent struggles to beat [AlphaStar]. Leela Zero discussions similarly identify rock-paper-scissors / opening-bias failure modes and recommend broader opponent/opening panels [Leela Zero discussion].

**6. Masked AlphaZero is a good baseline, but not the endpoint.** AlphaZe** shows AlphaZero-like methods can be surprisingly strong in imperfect-information games using sampling/PIMC-style adaptations, but it also notes strategy-fusion/hopping issues [AlphaZe**]. Student of Games and ReBeL show the principled way to handle two-player zero-sum imperfect information is public-belief-state search/learning, though at high complexity [Student of Games; ReBeL]. MAPLE is a 2026 middle ground: aggregate policy/value evaluations over multiple sampled world states in one tree to mitigate PIMC weaknesses while controlling cost [MAPLE]. For Catan, you do not need to jump straight to ReBeL. You likely can build a cheap exact/particle public belief tracker for opponent resources and dev deck, and feed belief summaries to the net and chance spectra.

**7. The architecture A/B did not close architecture.** A single controlled loss by a 47.8M upgrade says “not free under that recipe/data,” not “architecture does not matter.” Recent graph-game work is relevant: AlphaGateau represents chess as a graph with node features for value and edge features for policy, and reports faster learning than CNN AlphaZero variants [AlphaGateau]. Finite-group equivariant game nets reduce overfitting and improve parameter efficiency by enforcing symmetries [FGNN]. Your own D6 audit shows symmetry noise larger than the true opening spread [Catan-Zero §4.3]. That is not a small architectural nicety; it is directly on your failure mode.

**8. You are underusing Catan-specific strength hacks.** If the goal is #1 bot, purity is optional. Use opening-special search, symmetry averaging, belief features, bot-opponent data, and possibly a competition-time opening/early-game module if it wins. KataGo’s huge efficiency came partly from domain-specific auxiliary targets and score/ownership shaping, not pure AlphaZero minimalism [KataGo paper].

### How I would run the project from here

#### Promotion / flywheel

I would immediately replace the gate with a **two-layer rule**:

1. **Internal self-play promotion = regression protection.** Candidate must pass provenance checks, anchor no-drift, and a paired gate showing it is unlikely to be materially worse. Use something like elo0=-10, elo1=+15, α≈0.05–0.10, β≈0.05, cap 600 games. Or use a Bayesian posterior: promote if P(Elo > 0) ≥ 0.8 and P(Elo < -10) ≤ 0.05. This is more aligned with KataGo’s light gating and the Leela Zero experience than your current +30 Elo H1 [KataGo paper; Leela Zero discussion].
2. **External validity = asynchronous block/revert, not precondition for every small step.** Run the 1000-game catanatron_value/AB panel for each promoted champion, but only block/revert on a significant decline: e.g. P(external Elo delta < -25) > 0.9, or two consecutive promoted champions decline externally. A 200-game flat read should not freeze the flywheel.

I would also start a **canary generation lane**: let the candidate produce 20–30% of new data while the external panel finishes. If the candidate is a self-play overfit, the next candidate/anchor/external metrics will show it quickly; if it is genuinely better, you stop wasting compounding time.

#### Plateau escape

The current plateau means “the current gen-3 distribution with current targets has been distilled.” It does not mean “the game is exhausted.” The next distributional moves I would make are:

- Promote non-regressing candidates so self-play changes.
- Add champion-history opponent pool, initially 15–25% of generation. Use PFSP-like weighting: more games against opponents that current champ does not crush, and periodic all-pairs evaluation [AlphaStar].
- Add Go-Exploit restarts. The Go-Exploit paper’s diagnosis is almost your diagnosis: standard AlphaZero starts from the initial state, samples exploratory actions mostly early, and produces correlated long-game value labels; restart archives produce shorter trajectories and more independent value targets [Go-Exploit]. Your high-regret archive should include opening roots, high search/prior KL states, high value-variance roots, robber/dev-card decisions, and positions from external losses.
- Selectively reanalyze stale/high-impact states. ReZero’s periodic entire-buffer/backward-view reanalysis is aimed at improving target freshness while reducing search cost [ReZero]. You do not need full-buffer reanalysis first; do targeted reanalysis where old targets are likely wrong or high leverage.

#### Value head

I would make value classification the next training-science tournament. Arms:

- Current scalar MSE.
- Two-hot categorical over [-1,1], 21 or 41 bins.
- HL-Gauss categorical over [-1,1], 41 or 51 bins.
- Distribution over final VP margin plus win/loss expectation.

Keep policy recipe fixed. Try value loss 0.25–0.5 and value-head LR 0.3× torso LR. Use game-level validation splits, not row-level random splits, and report effective sample sizes by game. “Stop Regressing” is sufficiently on-point that I would not scale the net again until this is tested [Stop Regressing].

I would also allow **more policy learning without more value damage**. Your policy head tolerates reuse; your value head does not. So try two-phase training: one conservative joint pass, then policy-only or policy+torso low-LR pass with value head frozen, using soft search targets. You already had a two-phase policy arm that was only 53.25%, but it was before the full current diagnosis; retest only after value classification or on a fresh window.

#### Search

Keep c_scale=0.03. The more principled search work is not “use the paper default”; it is:

- **Noise-floor rescale / shrinkage**: make D1/D2 production-quality. Completed-Q should not min-max stretch a root when empirical spread is below estimated value noise.
- **Opening/wide-root special budget**: at ≥40 legal actions, use n=256/512 and/or symmetry averaging. Only a handful of decisions per game have this shape, so the strength/compute ratio is excellent.
- **Policy-target pruning**: KataGo prunes low-mass/noisy target moves, reducing noise in policy targets [KataGo methods/docs via KataGo paper]. This is especially relevant to your 54-action opening roots.
- **Chance/belief progressive widening**: for hidden-info/world-state sampling, borrow the BetaZero idea: in belief-space planning under limited search, use progressive widening and policy-prior sampling to control branching [BetaZero].

#### Architecture

Do not rewrite architecture this week. But do not conclude dense CLS-only action scoring is fine. My ordering:

1. D6 handling first: symmetry canonicalization or equivariant wrapper; at minimum use 12-fold TTA on opening/wide roots if Rust makes it cheap. Your own measured orientation noise demands this.
2. Action-entity interaction second: settlement action should see node token; road action should see edge token. AlphaGateau’s node/edge policy/value split is directly conceptually relevant [AlphaGateau].
3. Scale third: your 91M failure was a value-target/training failure on a frozen corpus, not evidence against scaling. Scaling-law work finds larger AlphaZero models are more sample-efficient and strength scales with parameter count when not compute-bottlenecked [Scaling laws]. But only scale after value classification and more fresh data.

#### Engineering

Your “NN is 4% of leaf” conclusion is correct today, but it is not a law of nature. Once Rust featurization lands, the bottleneck shifts. The next big engineering prize is not a smaller evaluator; it is **parallel/batched search with an eval-server model** and eventually moving more tree logic out of Python. KataGo’s self-play architecture is built around efficient asynchronous components [KataGo docs]. Subtree reuse is also obvious, but less important than batching/parallelism if Python still dominates.

### Answers to §16 questions

**Q1 — Promotion criterion.** Use regression-protection gating. Promote +20 Elo candidates if anchor/provenance checks pass and external panel is not significantly negative. Your current +30 bar will hold forever and prevents the improved policy from generating data. This is aligned with KataGo’s light 100/200 gate and optional gatekeeper, and with Leela Zero experience that high thresholds waste progress [KataGo paper; KataGo docs; Leela Zero discussion].

**Q2 — Escaping plateau.** Order: promote → opponent pool → Go-Exploit restarts → value classification → opening/wide-root special search → selective reanalysis → progressive sims → scale/architecture. More same-window training is dead by your own anchor telemetry.

**Q3 — Compression trend.** Some compression is expected at fixed net/search/data: policy iteration is a contraction toward the fixed point induced by current search budget. But distinguish expected diminishing returns from pathology by measuring: n=64 vs n=128 target disagreement; root value calibration by phase/action width; search-vs-raw absolute strength over generations; opening entropy/diversity; all-pairs cross-play among generations; external Elo slope. If high-budget search still finds large improvements but distillation cannot capture them, it is value/architecture. If high-budget search agrees with n=64, it is search-budget/data distribution.

**Q4 — External-transfer gap.** Internal inflation is normal; the size and latest-flatness are a warning, not proof. Replace fixed latest-vs-latest evaluation with all-pairs rating, opponent panels, and style diversity. AlphaStar’s league design exists because pure self-play cycles [AlphaStar].

**Q5 — Value-head fragility.** Structural fixes: categorical/HL-Gauss value, lower value LR/loss, target networks, game-level splits/sampling, selective reanalysis, ensembles/uncertainty for completed-Q shrinkage, and meaningful auxiliaries. KataGo’s auxiliary ownership/score targets were central to sample efficiency [KataGo paper]; your Catan-native auxiliaries should not remain weight-zero forever.

**Q6 — Wide stochastic roots.** Your c_scale result is believable. Adopt variance-aware completed-Q, noise-floor rescale, opening-special sims, symmetry averaging, policy-target pruning, and possibly Bayesian root action selection. The Gumbel guarantee is not contradicted; it assumes Q estimates are good enough, and your 54-wide 1–2-visit roots violate that premise [mctx].

**Q7 — Architecture.** The A/B result says “not now under that setup,” not “architecture closed.” I would invest first in D6 and action-target relation, then scale. Graph/equivariant game-network literature supports this direction [AlphaGateau; FGNN], but data/search/value fixes have higher immediate payoff.

**Q8 — Not asking.** The biggest missing question is: **what public belief state is your agent actually playing from?** After fixing the leak, you may have overcorrected to a deliberately ignorant observation. Catan has rich inferable hidden information; catanatron’s classical bots use belief logic. A cheap Catan-specific belief tracker may close more external Elo than another generic AlphaZero generation. The second missing question is: **are you building a pure research artifact or the strongest bot?** Those are now diverging. If #1 bot is the objective, promote strongest arms, use domain knowledge, use opening-special computation, and train against the bot you need to beat.

**Executive summary**

- Your +30 Elo promotion gate is now the wrong tool. It was sensible for early discrete jumps; it is actively blocking compounding at the current +15–25 Elo regime. I would move to regression-protection gating immediately, with external-panel revert/block rules rather than requiring the external panel to pre-confirm every small internal improvement.
- The reported +20 Elo turn-4 signal is suggestive, not a settled fact. Pooling 400 n=8 games with 600 n=16 games and quoting p≈0.04 is too casual under heterogeneous settings and optional extensions. Treat it as enough to canary/promote under regression-protection, not enough to claim a clean fourth scientific promotion.
- You are under-fixing the value head. The repeated “value cannot revisit the same distribution” observation is probably not a mysterious Catan law; it is standard noisy-target overfitting under high row-correlation, MSE regression, and too much value weight relative to effective independent games. Replace scalar MSE with categorical / HL-Gauss / two-hot value, reduce value update aggressiveness, split and sample at game-level, and use target networks / reanalysis.
- Your strongest immediate plateau escape is not “more epochs”; it is new target/data distribution: promote a non-regressing candidate, wire the opponent pool, add Go-Exploit-style restarts from high-regret states, and reanalyze selected high-value states with higher search. The literature supports search-control/restart data for better value targets and sample efficiency [Go-Exploit], and periodic reanalysis for stale targets [ReZero].
- You should spend compute on the opening/wide-root problem specifically. Search failures and symmetry noise concentrate at 54-wide opening placements. Use special opening search budgets, symmetry averaging/canonicalization, and/or an opening-root policy-target refresh. Since there are only a few opening moves per game, this is likely the cheapest external Elo.
- Your Gumbel c_scale=0.03 result is credible. mctx really does min-max rescale completed Q-values with default value_scale=0.1 and maxvisit_init=50 [mctx]; the Gumbel guarantee assumes correctly evaluated Qs, which you do not have at 1–2 samples/action. Do not revert to the paper default. Instead make your shrinkage/noise-floor variants first-class and test them on wide-root buckets.
- You are optimizing partly for clean science, not maximum strength. Promoting the clean gen2A control instead of the 59% λ=0.5 arm was the wrong choice if the sole goal is #1 Catan bot. Going forward, keep science branches, but champion the strongest statistically credible arm.
- The hidden-information repair is good, but masked observation is probably not the end state. Catan has tractable public belief information from action history, resource impossibility constraints, robber steals, and dev-card depletion. The modern imperfect-information literature says masked AlphaZero can be strong [AlphaZe**], but belief aggregation / public-belief search is the principled upgrade [Student of Games, ReBeL, MAPLE, BetaZero]. Build cheap Catan-specific belief features before a full ReBeL rewrite.
- Your architecture is probably leaving strength on the table: no D6 equivariance, adjacency unused, actions scored only via CLS. One failed one-seed A/B does not close architecture. But do not make this the first plateau move; do D6/opening/value fixes first, then retest graph/action-attention with the new value loss and ≥10M fresh rows.
- Your evaluation methodology is unusually strong for a hobby/research bot, but the external-validity instrument is underpowered and too narrow. Add all-pairs cross-play among recent champions, fixed-bot panel at ≥1000 games for promoted nets, varied opening/board suites, and population-style evaluation to detect non-transitivity / inbreeding [AlphaStar, Leela Zero discussions].