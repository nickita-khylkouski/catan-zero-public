# Expert review: Catan-Zero

I’m treating the attached report as the source of truth for your system and numbers. 

## Executive summary

1. **Soft-promote the +20 Elo candidate for self-play.** Your +30 Elo gate is now mis-specified: it was built for +50 Elo discrete jumps, but your loop is producing smaller real gains and then refusing to compound them. Keep a stricter “public champion” bar, but let non-regressing +15–20 Elo candidates feed the data engine. 

2. **Your biggest risk is no longer “can AZ work?” It is self-play inbreeding.** The internal ladder improved far more than the external panel, and the latest internal gain does not clearly transfer externally. That is a warning, not a verdict. Fix it with an opponent pool, exploiter tests, and broader external style coverage. 

3. **Stop treating “the value head cannot revisit the same distribution” as a universal law.** Your evidence strongly supports a narrower claim: your current scalar MSE value head, trained on correlated ±1 outcomes, is fragile under repeated passes and lineage accumulation. That points to distributional value, auxiliary targets, uncertainty, and start-state diversification—not only “one dose forever.” 

4. **The best plateau escape is not more epochs, more gates, or another c_scale grid.** It is new data from better and more diverse policies: soft promotion, opponent-pool games, and Go-Exploit/high-regret restart data. Go-Exploit’s central result is especially relevant: shorter restarted trajectories from archived states produce more independent value targets and better sample efficiency. ([arXiv][1])

5. **Your c_scale=0.03 result is credible. Keep it.** The mctx/Gumbel guarantee assumes action values are correctly evaluated; your wide, near-tied, low-visit stochastic roots violate that assumption through value noise amplification. The right next work is per-root uncertainty/shrinkage and better value targets, not reverting to paper defaults.  ([GitHub][2])

6. **Wire the opponent pool immediately.** AlphaStar’s success depended on a league of diverse strategies and counter-strategies, not a single linear self-play chain. Your own report says the pool is designed but unwired; that is now a strategic blocker.  ([Nature][3])

7. **Architecture is not the main blocker today, but D6 symmetry is too large to leave unused.** One failed v3b A/B does not close the architecture question. I would first productionize symmetry averaging/augmentation where affordable, then retry action-local and graph-biased variants after the data loop is healthier. 

8. **Finish the engineering that changes the science budget.** Your bottleneck is Python/process overhead, not neural inference. Rust featurization, parallel search, batching, and subtree reuse buy more experiments per day; a bigger net is cheap only after the search/data plumbing stops wasting leaves. 

---

## What you are doing wrong

### 1. The promotion gate is now blocking the improvement operator

This is the highest-confidence critique. Your own pooled result says the candidate is probably stronger than gen-3, around 52.8% over 1000 games, but not strong enough for a gate designed around +30 Elo. 

In expert iteration, improvement compounds only when the improved policy generates the next distribution. Holding a +15–20 Elo candidate forever because it is not +30 Elo is equivalent to freezing the policy while asking training to extract more from a fully distilled window. Your anchor telemetry already says that window is flat. 

**My rule:** split “self-play producer” from “official champion.”

* Promote to **self-play producer** if: anchor holdout is non-regressed, value drift tripwire is clean, paired gate is positive, and no external catastrophic regression is seen.
* Promote to **official champion** only after stronger confirmation: broader internal population eval plus external panel.
* Roll back automatically after two consecutive external/population regressions.

That is closer to KataGo/lc0-style small-compute pragmatism than an AlphaZero-style “always replace” or a rigid discrete SPRT. KataGo is the closest compute-class analogy: it achieved major efficiency gains on fewer than 30 GPUs by combining many system improvements rather than brute force. ([arXiv][4])

### 2. Your external evaluation is too narrow for the claim you want

Your goal is not “beat gen-3.” It is “be the strongest Catan bot.” The current external panel is useful but underpowered and too stylistically narrow. Gen-3 reached 45.7% vs catanatron_value, while the turn-4 candidate showed 41.0% in a 200-game panel with overlapping intervals. 

That does not prove the candidate is worse, but it does prove the panel is not yet resolving the decision you need it to resolve. Worse, fixed bots can become a blind spot: a single self-play lineage can improve internally while preserving exploitable holes. This is not theoretical—adversarial policies have beaten very strong Go AIs by exploiting narrow blind spots even when the victim is superhuman by normal benchmarks. ([arXiv][5])

**What I would add:**

* Old-champion population: v3a, gen-1, gen-2A, gen-3, current candidate.
* Style-randomized catanatron_value variants if easy: weight noise, opening randomization, search-depth variants.
* Exploiter agents trained specifically to beat the current champion.
* Position-suite eval: opening placements, robber choices, dev-card timing, longest-road races, high-resource swing turns.
* Separate “world ranking” score from “self-play ladder” score.

### 3. You are underusing diversity

The report says opponent-pool play is designed but not wired.  That is a mistake at the current stage. Early self-play can be a clean ladder; plateaued self-play needs diversity. AlphaStar’s league explicitly maintained diverse strategies and counter-strategies, and that design choice maps well to your external-transfer concern. ([Nature][3])

I would start with a simple mix, not a complicated league:

* 75–85% latest self-play producer vs itself.
* 10–15% vs recent champions.
* 5–10% vs “hard negatives”: older checkpoints or exploiters that beat the latest disproportionately.

Use PFSP-style sampling later. The first goal is simply to stop the distribution from narrowing.

### 4. You have a value-target quality problem, not merely a value-head overfitting problem

Your strongest training result is λ=0.5 blending of terminal outcome and search root value.  Your worst training failures are value failures under reuse or lineage accumulation.  

That pattern says: terminal ±1 outcomes are too sparse, too correlated, and too high-variance for the way you are reusing data. Avoidance helped, but I would now test structural fixes:

* **Distributional value**: WDL or two-hot scalar value bins instead of one tanh scalar with MSE.
* **Auxiliary value-adjacent heads**: final VP margin, expected resource production, hidden VP belief, longest-road/army status, settlement/city potential.
* **Value-specific optimization**: smaller value LR, Huber or log-cosh loss, stronger value weight decay, or stop-gradient mixtures.
* **Ensembled/uncertainty value for search**: use uncertainty to shrink completed-Q at wide roots.
* **Reanalyze only selectively**, not “more epochs on same targets.”

Reanalyse-style methods are relevant because they refresh targets on existing data with improved search/model estimates, and ReZero specifically targets cheaper, more efficient reanalysis. ([arXiv][6])

### 5. Your architecture A/B result is not strong enough to close architecture

The current 35M transformer has real ceilings: dense attention with unused adjacency, action scoring through CLS rather than action-local board tokens, and value from CLS only.  The failed v3b test is useful, but one controlled comparison at one data scale is not enough to conclude “architecture does not matter.”

That said, I would not make architecture the top priority today. Your data loop and value targets are higher leverage. The exception is **D6 symmetry**, because your measured symmetry violation is enormous relative to opening value spreads.  Finite-group equivariant networks have a track record in board games, and even if you do not build a full equivariant transformer, train-time augmentation plus targeted test-time averaging is low-regret. ([arXiv][7])

---

## What I would do next, ranked

### 1. Promote the current +20 Elo candidate as a self-play producer

**Judgment:** do this now unless the 1000-game external panel shows a clear, statistically meaningful external regression.

**Hypothesis:** compounding is blocked by the old gate; a +15–20 Elo producer will generate a better next distribution than gen-3, even if it is not a clean +30 Elo champion.

**Run:**

* Set regression gate around `elo0=-10, elo1=+15`, or use “two positive gates + clean anchor” as the self-play promotion rule.
* Keep gen-3 as rollback champion.
* Generate one window from gen-3 and one from the new producer if you can afford a short A/B.
* Train champion-init candidates on both windows.
* Compare on internal population, external panel, and high-regret suite.

**Changes my mind:** if producer-fed data improves internal H2H but repeatedly worsens external population scores, you have confirmed inbreeding and should stop linear promotion until the opponent pool is live.

### 2. Wire opponent-pool generation

**Established:** league/population training is a standard answer to strategy cycling and brittle self-play. AlphaStar is the clearest modern example. ([Nature][3])

**Run:**

* Start simple: 80% latest self-play, 10% previous champion, 5% older champion, 5% hard negative/exploiter.
* Keep color-swapped seeds.
* Track per-opponent win rate, KL, policy entropy, and value calibration separately.
* Do not let opponent-pool rows dominate the main distribution at first.

**Changes my mind:** if opponent-pool rows reduce internal gain and do not improve external transfer after one or two full windows, lower the mix—but I would not abandon it from one weak turn.

### 3. Add Go-Exploit/high-regret restart data

**Established:** Go-Exploit samples start states from an archive to produce shorter trajectories and more independent value targets, improving sample efficiency and strength against reference opponents. ([arXiv][1])

This is almost tailor-made for your value problem. You have long games, correlated rows, wide near-tied roots, and sparse terminal signal. Restart data attacks all four.

**Run:**

* Build an archive of high-regret states:

  * wide opening placements,
  * large search-policy vs raw-policy disagreements,
  * large value ensemble disagreement,
  * robber/development-card swing states,
  * positions where gen-3 and catanatron_value choose different plans.
* Generate 0.5–1.0M rows from these starts with color swaps.
* Train with capped weight, perhaps 10–25% of a window.
* Create a held-out high-regret suite that is never trained on.

**Changes my mind:** if high-regret data improves the suite but hurts full-game external panel, the restart distribution is too adversarial or over-weighted.

### 4. Test distributional value and Catan-native auxiliary heads

**Judgment:** this is the most likely structural fix for the value fragility.

**Run:**

* Keep the current scalar value head as control.
* Add WDL/two-hot value bins or VP-margin bins.
* Turn on auxiliary heads at small weight: final VP margin, longest road/army state, production potential, resource-count belief summaries.
* Use one-dose champion-init training only.
* Gate with search-vs-raw, internal population, and external panel.

**Changes my mind:** if distributional value improves calibration but search strength worsens, inspect completed-Q scaling; the value representation may need a search-time mapping, not abandonment.

### 5. Selective reanalyze, not blanket reuse

Reanalyse and ReZero-style target refresh are attractive, but your system has already shown value drift under repeated training on the same distribution.  So use reanalyze surgically.

**Run:**

* Reanalyze only high-uncertainty or stale rows.
* Use stronger search, maybe n=64/96, only for selected roots.
* Replace or blend root value targets; do not add extra epochs over all rows.
* Keep an anchor tripwire and a high-regret holdout.

LightZero is useful as a reference implementation ecosystem because it includes MuZero/Gumbel/ReZero-family algorithms in one framework. ([GitHub][8])

### 6. Progressive search-budget increase

MiniZero reports progressive simulation schedules as a useful knob across AlphaZero/MuZero/Gumbel variants. ([arXiv][9]) Your fixed n=64 may now be part of the compression trend.

**Run:**

* Try n_full 96 or 128 only after the next producer promotion.
* Keep n_fast where it is unless exact-budget behavior is revisited with a different target scheme.
* Compare not just H2H, but root calibration, policy entropy, and external transfer.
* Avoid a global budget increase if it reduces data diversity too much.

### 7. Productionize targeted D6 symmetry

Given your measured D6 violation, I would not wait for a full equivariant architecture.

**Run:**

* Use 12-fold averaging only at opening placements and other high-branching roots.
* Batch symmetry transforms in Rust to avoid multiplying Python overhead.
* Train-time augmentation should be on in at least one serious candidate after promotion.
* Measure external transfer, not just internal H2H.

### 8. Finish Rust/parallel search/subtree reuse

This is not glamorous, but it changes every other decision. Your own profile says neural inference is only 4% of leaf cost, and Python/process overhead dominates.  Your listed weaknesses—unbatched leaf evals and no subtree reuse—are exactly where a mature AZ engine should be stronger. 

Do this before betting hard on 91M+ models. Bigger models are attractive only after the data/search engine is no longer the constraint.

---

## Answers to your eight questions

### Q1 — Promotion criterion

Use regression-protection gating for the continuous loop. Your current gate is mathematically coherent but operationally wrong for +20 Elo increments. Promote for data generation under a softer rule, keep a stricter public champion rule, and use external/population regression tripwires.

I would not spend 900–1200 games trying to make a true +20 Elo candidate pass a +30 Elo SPRT. That burns compute while preventing the only mechanism that can move the distribution. 

### Q2 — Escaping the plateau

Order:

1. Soft promotion.
2. Opponent pool.
3. Go-Exploit/high-regret restarts.
4. Distributional/auxiliary value.
5. Selective reanalyze.
6. Progressive search-budget increase.
7. Bigger net.

Do not start with bigger net. Your 91M probe failed exactly where your system is already fragile: value under reused data. 

### Q3 — Compression trend

Some compression is expected at fixed net size, fixed search budget, and fixed data per turn. But your external-transfer gap and flat window also suggest distribution narrowing.

To distinguish:

* Run search-depth sweeps: if deeper search restores gain, net/search budget is the ceiling.
* Run external population eval: if gains vanish outside gen-3, inbreeding is the ceiling.
* Track value calibration by decision type: if opening/robber/dev-card buckets fail, value target quality is the ceiling.
* Compare producer-fed vs gen-3-fed next windows: if producer data moves again, gate blockage was the ceiling.

### Q4 — External-transfer gap

Internal +150 Elo and external +70 Elo is not fatal, but it is a serious warning. The last +20 internal with no visible external move raises the risk. 

A fixed bot panel is necessary but insufficient. Add population-based eval, style-randomized bots, exploiters, and a fixed tactical/strategic position suite.

### Q5 — Value-head fragility

Your mitigations are correct but defensive. Structural fixes worth trying:

* Distributional WDL/two-hot value.
* VP-margin/value multi-tasking.
* Catan-native auxiliary heads.
* Value uncertainty or ensembles for completed-Q shrinkage.
* Selective reanalyze.
* High-regret restart data for less correlated value targets.

The key reframing: the value head is not mysteriously fragile; it is being asked to learn low-noise long-horizon values from high-correlation, sparse, reused ±1 labels.

### Q6 — Search at wide stochastic roots

Keep c_scale=0.03. Your ablation beats literature defaults in your regime, and the contradiction is explainable: Gumbel’s policy-improvement story depends on useful Q estimates, while your wide roots get 1–2 visits per action and min-max rescaling turns noise into false confidence.  ([GitHub][2])

Next experiments should be:

* Per-root Q uncertainty.
* Shrinkage of completed-Q toward prior/root value.
* Policy-target pruning for low-evidence actions.
* Symmetry averaging at wide roots.
* Search-value calibration by action-bucket.

Do not re-open exact-budget sequential halving unless the value/search target machinery changes; your negative result is meaningful. 

### Q7 — Architecture

Architecture probably matters, but not before data diversity and value targets. The failed v3b A/B means “that upgrade did not win under that data/test regime,” not “graph/action-local architecture is irrelevant.” 

I would invest in:

1. D6 symmetry first.
2. Action-target gather/cross-attention second.
3. Bigger net third, only with more fresh diverse data.
4. Full graph-biased/equivariant architecture later.

Transformer game nets can learn nontrivial planning structure, as LCZero analyses suggest, but Catan’s parameterized action locality is a strong reason not to rely on CLS-only scoring forever. ([arXiv][10])

### Q8 — What you are not asking

You need a **Catan bot benchmark suite**, not just gates.

For a “#1 Catan bot” claim, build a frozen public-style evaluation battery:

* fixed tournament maps,
* random map suite,
* opening-placement suite,
* robber/development-card suite,
* old champion population,
* catanatron variants,
* exploiters,
* raw-policy and search-budget ablations,
* calibration reports by decision type.

Catan AI prior work is sparse and not directly comparable; Gendre & Kaneko showed RL progress in Catan, and catanatron provides a strong simulator/bot ecosystem, but there is no universally accepted AlphaZero-class Catan leaderboard. ([arXiv][11]) So you probably have to create the benchmark that will make your eventual “#1” claim credible.

---

## Bottom line

I would **not** rewrite the whole system. The core is sound: expert iteration, paired gates, hidden-info fix, lazy chance, value-target λ, provenance discipline, and throughput work are all real progress. But I would change the control policy immediately:

**Promote smaller non-regressing gains for data generation, diversify the self-play distribution, attack value-target quality structurally, and make external robustness a first-class metric.**

Right now, the most dangerous failure mode is not that Catan-Zero is broken. It is that it becomes very good at beating its own previous self while the catanatron_value gap stops closing.

[1]: https://arxiv.org/abs/2302.12359?utm_source=chatgpt.com "Targeted Search Control in AlphaZero for Effective Policy Improvement"
[2]: https://github.com/google-deepmind/mctx "https://github.com/google-deepmind/mctx"
[3]: https://www.nature.com/articles/s41586-019-1724-z "Grandmaster level in StarCraft II using multi-agent reinforcement learning | Nature"
[4]: https://arxiv.org/abs/1902.10565 "https://arxiv.org/abs/1902.10565"
[5]: https://arxiv.org/abs/2211.00241?utm_source=chatgpt.com "Adversarial Policies Beat Superhuman Go AIs"
[6]: https://arxiv.org/abs/2104.06294?utm_source=chatgpt.com "Online and Offline Reinforcement Learning by Planning with a Learned Model"
[7]: https://arxiv.org/abs/2009.05027?utm_source=chatgpt.com "Finite Group Equivariant Neural Networks for Games"
[8]: https://github.com/opendilab/LightZero "GitHub - opendilab/LightZero: [NeurIPS 2023 Spotlight] LightZero: A Unified Benchmark for Monte Carlo Tree Search in General Sequential Decision Scenarios (awesome MCTS) · GitHub"
[9]: https://arxiv.org/abs/2310.11305 "https://arxiv.org/abs/2310.11305"
[10]: https://arxiv.org/abs/2406.00877?utm_source=chatgpt.com "Evidence of Learned Look-Ahead in a Chess-Playing Neural Network"
[11]: https://arxiv.org/abs/2008.07079?utm_source=chatgpt.com "Playing Catan with Cross-dimensional Neural Network"
