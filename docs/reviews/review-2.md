# Executive summary: the 10 things that matter most

1. **Promote the +20 Elo candidate for self-play generation, but do not call it the public champion yet.** Your current +30 Elo SPRT was designed for discrete ~50 Elo jumps; it is now blocking the policy-improvement loop even though two independent measurements show a real but smaller gain, about 52.8% over 1000 games. The right fix is a **dual registry**: “generator champion” uses a regression-protection gate; “public champion” requires external/population confirmation. Your own plateau evidence says the current gen-3 window is fully distilled, so holding forever is more likely to starve the loop than protect it. 

2. **Your biggest current mistake is treating one fixed external bot panel as enough anti-inbreeding protection.** You know internal Elo is up ~150 while external strength is up ~70 and the latest internal gain does not visibly transfer; that is not proof of failure, but it is a serious warning. The fix is not “hold all promotions”; it is a **population evaluation league**: prior champions, multiple search budgets, catanatron variants, handcrafted style exploiters, and dedicated anti-Catan-Zero opponents. 

3. **The value-head story is real, but your conclusion is too absolute.** The evidence supports “your current scalar-MSE value recipe overfits correlated self-play rows under reuse”; it does not prove “the value head cannot revisit the same distribution” as a law. Treat the value target as a noisy, correlated, game-level target, not as millions of independent row labels. Your own failures—Recipe-B, continuous lineage drift, and the 91M probe—are enough to justify structural value changes.  

4. **Switch the value head from scalar MSE to categorical / two-hot / histogram loss immediately.** This is the most obvious 2024-era literature miss. “Stop Regressing” argues that categorical value losses improve robustness to noisy and non-stationary RL targets, exactly your failure mode, and describes two-hot / histogram-style target distributions. ([arXiv][1])

5. **You are probably underusing Catan-specific belief inference.** Masked AlphaZero was the right emergency fix for the leak, but “zero hidden slots” is not the same as “optimal public belief.” In two-player no-trade Catan, resource/development-card belief is much more tractable than poker: most resource changes are public, and the uncertainty comes from robber steals and dev-card draws. Build an exact or particle belief tracker and feed belief summaries to the net/search. Your report’s leak fix proves you now respect public observation; the next step is public belief, not omniscience. 

6. **Your Gumbel `c_scale=0.03` result is credible; do not revert to paper defaults.** The mctx implementation really does complete missing Q-values, optionally min-max rescale to `[0,1]`, then multiply by `value_scale` and visit scale; your wide-root, 1–2 visits/action setting is exactly where min-max noise amplification can dominate. The contradiction with Gumbel MuZero defaults is not embarrassing; it is a domain/regime difference. ([GitHub][2]) 

7. **But “search config closed permanently” is too strong.** It is closed for the current net, current value noise, current root widths, and current generation budget. Once you add categorical values, belief features, policy-target pruning, symmetry, or more sims, the right Q-transform scale may move. Keep `c_scale=0.03` as production, but keep a small recurring calibration sweep.

8. **The fastest route to #1 is not pure AlphaZero purity; it is a hybrid product.** Use self-play as the strength engine, but add Catan-specific modules where they are cheap: opening-placement book/head, public-belief tracker, external-bot exploit data, and final-match search tuning. The report says opening placement is a 54-wide near-tied root where value noise is destructive; that is exactly where a specialized offline opening solver/book is worth it.  

9. **Architecture probably matters, but not before data diversity and value robustness.** The current architecture throws away graph adjacency, action-target locality, and richer value pooling; that is a real ceiling. But one failed v3b A/B is weak evidence, not a theorem. Revisit graph/action-centric architecture after you stabilize value training and get a 10M+ fresh-row corpus. 

10. **The thing you are not asking sharply enough: what exactly is “#1 Catan bot”?** Define the tournament spec now: board distribution, time/search budget, no-trade rules, hidden-info rules, engine version, allowed opening books, allowed classical heuristics, opponent set, and statistical protocol. Without this, you can beat catanatron_value under one setup and still not know whether you built the strongest bot or the strongest bot for your harness.

---

# What I would do next

## 1. Promotion: split “generator champion” from “public champion”

Your current situation is mechanically clear: gen-4 / flywheel candidates are likely better than gen-3 internally, but not by the +30 Elo the old gate demands. The gate was designed when you were getting ~+49, +49, +33 Elo jumps; now the true improvement appears closer to +20 Elo. At that true effect size, your own analysis says the SPRT will tend to hold forever.  

My rule:

**Generator promotion gate:** promote a candidate to feed self-play if all are true: anchor drift is clean; paired gate is positive under a regression-protection SPRT such as `elo0=-10, elo1=+15`, or two consecutive gates have positive LLR; and no external tripwire fires. This is not a claim that the candidate is final-best. It is a claim that it is safe and likely useful as the next data generator.

**Public champion gate:** only update the public / headline champion when it wins the external population suite, especially catanatron_value, at high enough power. Your n=200 panels are useful telemetry, not final truth; your own confidence intervals overlap. 

This matches the spirit of small-compute self-play projects better than your current hard +30 Elo bar. KataGo’s published training used a moving window, playout-cap randomization, progressive net growth, SWA, and a gating/evaluation process on modest GPU counts; the point was not “require large Elo jumps,” but “avoid letting bad nets poison the data stream.” ([arXiv][3])

**Concrete implementation:** keep three registry labels: `generator_champion`, `public_champion`, and `external_best`. Promote the +20 Elo candidate to `generator_champion`; keep gen-3 as `public_champion` until the 1000-game external panel resolves. If the promoted generator produces gen-5 data and gen-5 does not beat gen-4/gen-3 internally, roll back the generator. If two consecutive generator promotions show external decline beyond a pre-set bound, roll back and increase population/external data.

---

## 2. Plateau escape: promote, diversify, then target value

The window is flat across nine rounds, and the report says the current gen-3 self-play window is fully distilled. More training on that same distribution is not the answer. 

My ordering:

1. **Promote the safe +20 Elo generator and generate 3–5M fresh rows from it.** This tests the core hypothesis: small improvements compound only when they become the data-generating policy.

2. **Wire the opponent pool immediately.** Use 70–80% current-generator self-play, 15–25% historical/pool opponents, and 5–10% targeted restarts. Your report says the pool is designed but not wired; it should be wired before architecture work. 

3. **Add Go-Exploit-style restarts.** Go-Exploit starts self-play trajectories from archived states of interest, improving exploration, value generalization, and value-target independence; that directly addresses your “many correlated rows from few games” value problem. ([arXiv][4])

4. **Fix value structurally.** Categorical value, per-game weighting, lower value LR, value calibration metrics, and belief-conditioned value features should come before a bigger net.

5. **Then test progressive sims / reanalyze / bigger net.** MiniZero’s comparison across AlphaZero/MuZero/Gumbel variants found more simulations generally help board games and introduced progressive simulation as a compute-efficient schedule; that is worth testing after value is stable. ([arXiv][5])

---

# What you are doing wrong

## A. You are over-trusting internal head-to-heads

The internal ladder is valuable, but it is not the objective. Your evidence shows a large internal gain and a smaller external gain, with the newest internal gain not visibly moving the external panel. That is exactly the self-play inbreeding pattern you say you fear. 

The mistake is not running internal gates; paired pentanomial gates are a good design. The mistake is letting a single fixed bot panel be the only external-validity instrument. You need a population evaluator: prior nets, raw policies, production-search nets, low/high-sim variants, catanatron_value, AB3/AB4, scripted Catan styles, and exploiters trained or tuned to beat the current champion.

**Experiment that changes my mind:** if a promoted +20 internal generator produces a gen-5 that improves both internal H2H and a 1000+ game population-Elo suite, then the current plateau was mostly gate starvation. If it improves internal but loses population Elo twice, it is inbreeding.

---

## B. You are treating rows as more independent than they are

You have millions of rows, but value labels come from thousands of terminal outcomes. A 16k-game corpus has 16k independent win/loss outcomes, not 3.6M independent value labels. The same terminal result is smeared across ~200 correlated decisions. That makes value-head overfitting unsurprising, especially under extra epochs or continuous lineage reuse. Your own Recipe-B autopsy shows train improves while validation value degrades and policy remains fine. 

I would change value training now:

* Normalize value loss by game, not row, or cap each game’s total value weight.
* Train a categorical value distribution over win/loss plus VP-margin buckets, not a scalar MSE.
* Use a lower LR multiplier for the value head and value-facing trunk gradients.
* Track calibration by game phase, root width, player color, hidden-info entropy, and opening bucket.
* Keep the λ=0.5 root-value blend, but audit whether root value targets are themselves stale/noisy under the current champion.

The categorical-value recommendation is not speculative handwaving. Farebrother et al. argue that cross-entropy value training improves robustness to noisy and non-stationary RL targets and describes two-hot / histogram-style projections from scalar targets to categorical distributions. ([arXiv][1])

---

## C. You fixed hidden-info leakage, but have not exploited public belief

The leak fix was essential: omniscient opponent hand/dev cards would invalidate external comparisons. Your three-layer fix—masking, invariance tests, and belief spectra—is the right emergency repair. 

But now you may be under-modeling the game. In two-player no-trade Catan, public belief is relatively tractable. Opponent resource gains from dice are public; spending is public; uncertainty is introduced by robber steals and dev-card draws. You can maintain a belief distribution over opponent hand and dev deck, then feed summaries to the net: expected counts, entropy, top-k possible hands, dev-card distribution, steal-value distribution, and “can opponent build X next turn?” probabilities.

ReBeL and Student of Games are the heavy versions of this idea: public-belief search/value for imperfect-information games. They are likely too big a rewrite for your next week, but they establish that “masked perfect-information AlphaZero” is not the theoretically clean endpoint for imperfect information. ReBeL frames imperfect-information search around public belief and convergence in two-player zero-sum games; Student of Games combines guided search, self-play, and game-theoretic reasoning across perfect and imperfect information games. ([arXiv][6]) ([arXiv][7])

**Low-cost version:** build an exact/particle Catan belief tracker and add belief features without changing the search algorithm. Gate it first against gen-3 and specifically against catanatron_value.

---

## D. You are under-optimizing the deployed bot

Your report optimizes the data engine, not the final player. That is correct for self-play throughput, but the stated goal is to beat every bot. The deployed bot can have a different configuration from the generator: more sims, symmetry averaging, opening book, belief tracker, external-specific time controls, and perhaps a slower but stronger search path.

The report says NN forward is only 4% of leaf cost, and Python/tree/featurization dominates. It also says symmetry averaging gives ~3.3x denoising on openings.  

So define two configs:

**Generator config:** fast, stable, cheap, enough exploration.

**Tournament config:** maximize win rate under the allowed budget. It may use n=128/256, 12-fold symmetry at openings, a precomputed opening-placement book, exact belief, and a different Q-transform.

This is not cheating unless your tournament spec forbids it. If your goal is #1, purity is optional; strength is not.

---

## E. You are too willing to declare negative results “closed forever”

Your kill list is excellent science, but some conclusions are over-scoped. `c_scale=1.0`, fixed-bounds Q-transform, exact-budget SH, and multi-epoch reuse are dead for the current regime. They are not necessarily dead after categorical value, belief features, policy-target pruning, more data, or a new architecture. 

Keep the kill list, but add a column: **“closed under assumptions.”** Otherwise old negative results will block future gains after the underlying failure mode is removed.

---

# Literature and systems worth stealing from

## KataGo

KataGo is the closest compute-class analogue. It used fewer than 30 GPUs and got large efficiency gains from playout-cap randomization, policy-target pruning, global pooling, auxiliary targets, moving-window training, progressive net growth, and gating/evaluation discipline. Your PCR design and small-compute gating instincts are aligned with KataGo, but you have not yet stolen enough of its target shaping and auxiliary-target discipline. ([arXiv][3])

Most relevant steal: **policy-target pruning / forced-playout separation.** Your wide roots suffer because exploratory visits and value noise contaminate the target. KataGo explicitly separated “force exploration” from “train the policy to like everything explored.” You need a Gumbel-compatible version.

Second steal: **auxiliary value structure.** KataGo’s auxiliary ownership/score targets reduce reliance on one sparse binary outcome. In Catan terms, train resource-production control, port access, longest-road potential, army/knight pressure, steal equity, VP-margin distribution, and opponent-build-threat heads. Your auxiliary heads exist but are zero-weighted; that is leaving free regularization on the floor. 

## Go-Exploit

Go-Exploit is directly relevant to your plateau and value fragility. It argues AlphaZero’s normal start-from-initial self-play undersamples deeper states, and that starting from archived states improves exploration, value generalization, and independence of value targets. ([arXiv][4])

Steal this now. Your archive should include: high value-error states, high regret/search-disagreement roots, openings with unstable symmetry averages, positions lost to catanatron_value, long-road race pivots, robber/discard states, and high hidden-info-entropy states.

## MiniZero / LightZero / mctx

MiniZero is a useful reminder that algorithm choice and sim budget are domain-dependent; it supports AlphaZero, MuZero, Gumbel AlphaZero, and Gumbel MuZero and found progressive simulation helpful in board games. LightZero frames the practical bottlenecks as modular search/system issues: complex action spaces, stochasticity, simulation cost, exploration, and throughput. ([arXiv][5]) ([arXiv][8])

mctx itself supports your diagnosis: Gumbel MuZero’s completed-Q transform completes unvisited actions, min-max rescales, and scales Q-values; the README recommends Gumbel because of a policy-improvement guarantee **if action values are correctly evaluated**. Your whole problem is that action values at wide stochastic roots are not correctly evaluated at 1–2 visits/action. ([GitHub][9]) ([GitHub][2])

## ReBeL / Student of Games

These are not immediate rewrites, but they are the north-star literature for imperfect information. They say: public belief matters; plain MCTS over hidden state is not generally sound; and search/value learning can be made game-theoretic in imperfect-information settings. ([arXiv][6]) ([arXiv][7])

For Catan, I would not jump to full ReBeL. I would build the Catan-specific public-belief tracker first.

## Catan-specific prior work

Gendre & Kaneko’s Catan RL work is relevant mainly as prior art: it studies two-player no-trade Catan and emphasizes Catan’s imperfect information, stochasticity, and board/action structure. It is not evidence that a stronger public bot already exists, and I did not find a public leak-free AlphaZero-style Catan system with explicit chance nodes and paired statistical gating. Still, phrase that claim conservatively: “we did not find,” not “there is none.” ([arXiv][10])

catanatron is a real target because it is a public Catan simulator with strong AI-player support; your external target choice is reasonable. ([GitHub][11])

The 2025 LLM/Catanatron work I found is not a serious comparator for your stated goal: it evaluates LLM strategic-planning agents in Catanatron with small game counts and reports improvements against static baselines, not high-powered bot strength against your target. ([arXiv][12])

---

# Ranked experiment roadmap

| Rank | Experiment                                   | Hypothesis                                                                      | How to run                                                                                                                                 | Decision rule                                                                            |
| ---: | -------------------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------- |
|    1 | **Dual-registry promotion**                  | +20 Elo generator improvements compound if allowed to generate data.            | Promote current candidate only as `generator_champion`; generate 3–5M rows; train one-dose candidate; gate vs gen-3 and current generator. | Keep if next gen is positive internally and external tripwire does not fire.             |
|    2 | **1000–2000 game population external suite** | Current external flatness may be noise; fixed n=200 panels are underpowered.    | Evaluate gen-3, current candidate, next candidate vs catanatron_value, AB variants, prior champions, raw policies, scripted styles.        | Public champion changes only on population Elo / target-bot confirmation.                |
|    3 | **Categorical value head**                   | Scalar MSE is amplifying noisy/non-stationary value targets.                    | 64 or 101 bins over `[-1,1]`, two-hot or HL-Gauss; same policy head; one-dose train; no other changes.                                     | Promote recipe if H2H positive and value calibration improves without policy regression. |
|    4 | **Per-game value weighting**                 | Millions of row labels overweight long games and duplicate one terminal result. | Reweight value loss so each game contributes fixed total value mass; compare to current row weighting.                                     | Adopt if value drift/overfit decreases and gate is neutral-positive.                     |
|    5 | **Opponent-pool generation**                 | External gap is partly self-play inbreeding.                                    | 75% current self-play, 15% prior champions, 10% exploit/style bots or best-response-ish agents.                                            | Adopt if external population Elo improves without internal collapse.                     |
|    6 | **Go-Exploit restarts**                      | Value needs more independent targets and deeper-state coverage.                 | Sample starts from high-regret/high-error/lost-to-value-bot archive; short rollouts with public observations.                              | Adopt if value calibration and external transfer improve.                                |
|    7 | **Belief features**                          | Masking is safe but under-informative.                                          | Exact/particle tracker for opponent resources/dev deck; add belief summaries to player/context tokens.                                     | Gate vs gen-3 and catanatron_value; inspect robber/dev-card decisions.                   |
|    8 | **Opening-placement book/head**              | Opening wide-root noise is a disproportionate strength leak.                    | Offline high-budget search/rollouts for sampled boards; distill to opening auxiliary head or table.                                        | Adopt in tournament config if external/opening bucket improves.                          |
|    9 | **Gumbel Q-transform calibration**           | `c_scale=0.03` is good but static; uncertainty-aware transforms may be better.  | Re-test D1/D2 plus calibrated shrinkage after categorical value; root-only variants.                                                       | Adopt only on H2H, not target KL.                                                        |
|   10 | **Progressive sim schedule / reanalyze**     | Fixed n=64 may be a search ceiling.                                             | After value fix, compare n=64→96/128 generation or reanalyze current corpus with stronger search.                                          | Adopt if extra compute produces external transfer per row.                               |
|   11 | **Graph/action-centric architecture**        | Current CLS/cosine model leaves structure unused.                               | Multi-seed A/B after ≥10M fresh rows; action tokens attend to target entities; graph relative biases; D6 handling.                         | Adopt only on gates and calibration, not loss alone.                                     |
|   12 | **91M+ scaling retry**                       | Bigger nets are cheap at inference but need fresh data and stable value.        | Retry only after categorical value and ≥10–20M fresh rows.                                                                                 | Continue if value does not blow up and gates are positive.                               |

---

# Answers to your §16 questions

## Q1 — Promotion criterion

Use regression-protection gating for **generator promotion**, not the old +30 Elo gate. I would use something like `elo0=-10, elo1=+15`, plus anchor-value tripwires, plus external-population tripwires. Keep the stricter criterion only for **public champion** updates. Your current evidence is exactly the case where a hard +30 gate is harmful: true +20 improvements cannot compound if they never feed the data loop. 

## Q2 — Escaping the plateau

Promotion is the first plateau escape, because the current window is fully distilled. After that: opponent pool, Go-Exploit restarts, categorical value, belief features, target pruning, and then progressive sims/reanalyze. Bigger net is later, not because scale is bad, but because your 91M probe already showed the current value setup cannot safely absorb extra capacity on reused data. 

## Q3 — Compression trend

The +49 → +49 → +33 → +20 trend is plausibly normal diminishing returns at fixed search budget, fixed model size, and increasingly stronger opponents. But it could also be a symptom of value-target ceiling, search noise at wide roots, or self-play narrowing. 

Measurements to distinguish them:

* Gate each champion at n=8/16/32/64/128. If improvement vanishes at high sims, net prior improved but search erases it. If improvement exists only at low sims, training is helping policy but not value.
* Compare high-budget search with the same net against production search. If n=256 crushes n=64, search budget is the ceiling.
* Track value calibration on fresh-policy states, not only anchor states.
* Measure diversity: opening entropy, resource-strategy clusters, road/army/dev-card style frequencies, and opponent exploitability.
* Run external population Elo every generation, not only catanatron_value.

## Q4 — External-transfer gap

Some internal inflation is normal. But +150 internal vs +70 external, and the latest +20 internal showing ~0 external, is enough to treat inbreeding as a live risk. 

Better instruments: population Elo, prior-champion round robins, exploiters, external bot style variants, high-search mirrors, opening-specific evaluation, and lost-position restart suites. A fixed catanatron panel is necessary but not sufficient.

## Q5 — Value-head fragility

Structural fixes I would try, in order:

1. Categorical/two-hot value head.
2. Per-game value weighting.
3. Lower value LR / value-gradient clipping.
4. VP-margin distribution, not just win/loss.
5. Auxiliary heads with nonzero weights.
6. Belief-conditioned value features.
7. Small value ensemble for search-time uncertainty.
8. Reanalyze only after value is stable.

Your current avoidance strategy—one-dose training, champion-init, anchor tripwire—is good operational medicine, but it does not cure the disease. 

## Q6 — Search at wide stochastic roots

Keep `c_scale=0.03` for now. Your ablation is more relevant than the paper default because your root has 54 near-tied actions and only 1–2 visits/action. The mctx default transform’s min-max rescale makes your diagnosis plausible. ([GitHub][2])

Next search experiments should be:

* Root-only variance-aware completed-Q shrinkage.
* Signal/noise calibrated Q scaling.
* Gumbel-compatible policy-target pruning.
* Opening-specific symmetry averaging or opening book.
* Re-test after categorical value; the value-noise distribution will change.

## Q7 — Architecture

The current architecture is not what I would design for final Catan strength: it discards graph adjacency, scores actions only through CLS, and reads value only from CLS. 

But I would not spend the next week there. The failed v3b A/B is weak evidence against architectural improvements because it is one comparison at one data scale. Recent graph-representation work in chess suggests graph/edge features can improve sample efficiency in AlphaZero-like board-game settings, but that does not mean your first zero-init cross-attention attempt was the right implementation. ([arXiv][13])

Priority: value/belief/data diversity first; graph/action architecture later.

## Q8 — What you are not asking

You are not asking enough about **productizing final strength**. If the goal is the #1 Catan bot, you should not restrict yourself to the self-play generator’s clean AlphaZero loop. Build the strongest legal player: public belief, opening book, final-search tuning, value-bot adversarial data, symmetry, and population-tested robustness.

You are also not asking enough about the **official benchmark definition**. Decide now what counts as victory, or the system will optimize to whatever harness is most convenient.

---

# Blunt closing take

This is a strong project. The bug discipline, paired gates, hidden-info audit, and negative-result accounting are much better than most hobby or academic AlphaZero attempts. The report is self-contained and explicitly asks for critique rather than validation, which is the right posture. 

The main thing I would change is strategic: stop thinking of the system as one line of champions. It should be a **league plus registry**:

* one checkpoint generates data,
* one checkpoint is the public champion,
* one checkpoint is the experimental candidate,
* many opponents test for inbreeding,
* and the deployed bot is allowed to be stronger than the generator.

Right now, your flywheel is being throttled by a gate designed for an earlier regime. Promote cautiously, evaluate brutally, and spend the next cycle on value robustness, belief, population evaluation, and targeted data—not on another round of training the same window.

[1]: https://arxiv.org/abs/2403.03950 "Stop Regressing: Training Value Functions via Classification for Scalable Deep RL"
[2]: https://raw.githubusercontent.com/google-deepmind/mctx/main/mctx/_src/qtransforms.py "raw.githubusercontent.com"
[3]: https://arxiv.org/abs/1902.10565 "Accelerating Self-Play Learning in Go"
[4]: https://arxiv.org/abs/2302.12359 "[2302.12359] Targeted Search Control in AlphaZero for Effective Policy Improvement"
[5]: https://arxiv.org/abs/2310.11305 "[2310.11305] MiniZero: Comparative Analysis of AlphaZero and MuZero on Go, Othello, and Atari Games"
[6]: https://arxiv.org/abs/2007.13544 "[2007.13544] Combining Deep Reinforcement Learning and Search for Imperfect-Information Games"
[7]: https://arxiv.org/abs/2112.03178 "[2112.03178] Student of Games: A unified learning algorithm for both perfect and imperfect information games"
[8]: https://arxiv.org/abs/2310.08348 "[2310.08348] LightZero: A Unified Benchmark for Monte Carlo Tree Search in General Sequential Decision Scenarios"
[9]: https://github.com/google-deepmind/mctx "GitHub - google-deepmind/mctx: Monte Carlo tree search in JAX · GitHub"
[10]: https://arxiv.org/abs/2008.07079 "Playing Catan with Cross-dimensional Neural Network"
[11]: https://github.com/bcollazo/catanatron "GitHub - bcollazo/catanatron: Settlers of Catan Bot Simulator and Strong AI Player · GitHub"
[12]: https://arxiv.org/abs/2506.04651 "Agents of Change: Self-Evolving LLM Agents for Strategic Planning"
[13]: https://arxiv.org/abs/2410.23753 "Enhancing Chess Reinforcement Learning with Graph Representation"
