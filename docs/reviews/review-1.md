I treat your uploaded 2026-07-08 report as the project record; I did not inspect code or private artifacts beyond that report, so anything below is an external-review judgment based on your reported evidence plus current public literature. 

# Executive summary: the 10 things that matter most

**1. Promote the +20 Elo candidate for data generation, but stop calling that “certified champion strength.”** Your current gate is doing exactly what it was designed to do: reject candidates below +30 Elo. That was right for discrete +50 Elo jumps and wrong for a continuous flywheel producing +20 Elo improvements. The plateau evidence says the gen-3 data window is fully distilled; holding the candidate means withholding the only mechanism that creates a new distribution. Use a **training-champion / release-champion split**: promote for self-play if the candidate is probably non-regressing, but only update public strength claims after external validation. Your own §12 basically proves this. 

**2. Your external evaluation is underpowered and too narrow to protect you from inbreeding.** A 200-game panel with ±7% uncertainty cannot adjudicate a 4–5 point external change, so “external flatness” is a warning, not a verdict. Make an always-on arena: champion, candidate, older champions, catanatron_value, AB3/AB4, raw-policy variants, search-config variants, and eventually exploiters. Decisions should use internal paired gates **and** absolute arena Elo.

**3. The biggest missing system component is legal belief tracking.** You fixed the hidden-info leak by masking, which was necessary, but you may have overcorrected into an impoverished public state. In no-trade two-player Catan, opponent resource/development-card beliefs are tractable enough to track from public history. Feed posterior summaries to the net and use posterior chance spectra for steals/dev draws. This is not cheating; using the true hidden hand was cheating. Uniform belief is probably leaving free Elo on the table, especially against catanatron’s belief-based bots. 

**4. Stop training only against yourself.** Your goal is not “clean AlphaZero.” Your goal is the strongest Catan bot. Use catanatron_value, AB bots, old champions, and high-regret archived states as **state-distribution generators**, not as imitation teachers. Go-Exploit and later regret-guided search-control work are directly aimed at the value-head/generalization failure you are seeing. ([arXiv][1])

**5. The value-head problem is real, but your “law” is overgeneralized.** Your evidence proves that your current scalar MSE value head plus correlated outcome rows plus repeated frozen-window training is fragile. It does **not** prove that value heads categorically cannot revisit a distribution. The fix should be structural: short-horizon value targets, uncertainty/error heads, distributional/two-hot value, per-game value weighting, and belief-aware auxiliary heads. KataGo’s later methods are a giant hint: auxiliary value/error targets and uncertainty-weighted playouts were not decorative; they became search-strength tools. ([GitHub][2])

**6. Your Gumbel/c_scale result is credible; do not revert to paper defaults.** The apparent contradiction with Gumbel MuZero mostly disappears because the Gumbel policy-improvement guarantee assumes correctly evaluated action values, while your 54-wide stochastic roots at 1–2 visits/action do not satisfy that condition. Your small `c_scale` is acting as a necessary noise regularizer.  ([GitHub][3])

**7. You are probably throwing away opening strength.** Opening placement is exactly where your report says value noise dominates. Use adaptive high-search, symmetry averaging, candidate pruning/top-k root action selection, and possibly an opening-specific evaluator/book. For the goal “#1 Catan bot,” match-time strength and training-time efficiency should be separate configs.

**8. Your architecture conclusion is too strong.** The fact that one 47.8M warm-start-safe architecture lost one controlled comparison does not prove graph/action attention is bad. It may prove that newly zero-initialized modules do not learn under a one-epoch, low-LR, value-fragile recipe. Your current architecture still has obvious ceilings: no consumed adjacency, no action-to-target cross-attention, value from CLS only, and severe D6 symmetry violation. 

**9. “Production + λ=0.5” has a provenance/name inconsistency.** §8 says the locked production recipe includes `value-target-lambda 0.5`; §9.2 says “production + λ0.5” beat “production verbatim,” and the weaker arm was promoted as the clean-science control. Clean science is valuable, but if the goal is #1 bot, leaving the best verified arm unused is a goal mismatch unless there was an unreported robustness reason.  

**10. The thing you most need to know is not “what is the next AlphaZero trick?” It is “where, exactly, does catanatron_value take games from us?”** Build loss attribution: opening placement EV, robber EV, dev-card timing, race/endgame conversion, longest-road swings, discard decisions, hidden-VP uncertainty, and high-regret positions. Right now you have a ladder, not a diagnosis.

# What I would do first

I would run two tracks immediately.

The **strength track** is about beating catanatron_value as fast as possible. It should use adaptive match-time search, legal belief tracking, D6 averaging at high-noise roots, and targeted state generation from losses to catanatron_value. It should not be constrained by training-time throughput aesthetics.

The **flywheel track** is about continuing expert iteration safely. It should promote +20 Elo candidates into self-play under a regression-protection gate, while retaining a separate release champion and external arena. The current plateau is not mysterious: you have distilled the current self-play distribution and are refusing to let the improved policy generate the next one. 

# What you are doing wrong

## 1. The promotion rule is now mis-specified

Your old SPRT was a “large discrete improvement” detector. Your current process needs a “non-regression plus useful-improvement” detector. At true 53% win rate, your own analysis says the current cap will land below the LLR needed for H1, so the gate will hold forever despite real improvement. 

My rule would be:

Promote to **training champion** if all are true: internal paired gate posterior says `P(Elo > 0) high`, `P(Elo < -10) very low`; anchor value drift is below tripwire; no obvious external sentinel collapse; no provenance/config mismatch. Keep **release champion** unchanged until external arena confirms absolute strength.

A concrete version: promote for data generation at `P(Elo > 0) ≥ 0.80` and `P(Elo < -10) ≤ 0.05`, or with your SPRT framing use `elo0=-10, elo1=+15` as a regression-protection test. Keep the 0/+30 gate only for “public champion certified +30” claims.

This mirrors the practical lesson from KataGo-class compute: small-compute systems need guardrails, but not so much conservatism that the data distribution never advances. KataGo’s original paper is directly relevant because it reached high strength with fewer than 30 GPUs, i.e. compute in your regime rather than AlphaZero’s thousands of TPUs. ([arXiv][4])

## 2. Your external panel is a warning light, not a decision instrument

The turn-4 candidate’s 41.0% vs catanatron_value compared with gen-3’s 45.7% is not enough to conclude external regression, because you already note the CIs overlap at n=200. But it is enough to say the internal +20 Elo is not transferring cleanly. 

Fix this by changing the evaluation object. Do not just ask “candidate vs champion?” Ask “where does this network sit in a population?” Run a round-robin or anchored Elo model over:

current champion, current candidate, last 5 champions, catanatron_value, AB3, AB4, raw policy, search-disabled policy, maybe catanatron variants, and eventually exploiters.

The binding anti-inbreeding metric should be an absolute Elo estimate against this pool, not a single 200-game fixed bot panel.

## 3. You are underusing the strongest external opponent

You correctly rejected behavior cloning as the main path because it cannot exceed the teacher, especially with noisy teachers. But that does **not** imply “never train on catanatron_value states.” Use the value bot as a **state distribution adversary**.

Have the champion play catanatron_value. Archive positions where your root value, eventual outcome, and external bot action imply high regret. Reanalyze those positions with your search. Generate short continuation games from them. This is exactly the family of ideas in Go-Exploit: start self-play from states of interest to improve value generalization and get more independent value targets. ([arXiv][1])

The 2026 regret-guided extension is even closer to your pathology: it prioritizes states where evaluation diverges from outcome and reports sizable gains over AlphaZero and Go-Exploit in board games. Treat it as a design sketch, not gospel, because it is newer and not Catan-specific. ([arXiv][5])

## 4. You fixed hidden information, but not belief

The leak was a serious invalidator: opponent hand, dev-card identities, and true VP were visible; planner chance spectra also used true hidden state. Your three-layer fix was necessary. 

But “masked AlphaZero is sufficient so far” is not the same as “belief does not matter.” In no-trade Catan, much of the hidden state is inferable from public history. You should build a public-history belief tracker:

* posterior over opponent resource multiset;
* posterior over dev deck / hidden VP probability;
* expected robber-steal distribution;
* entropy/confidence features;
* maybe particles if exact belief becomes expensive.

Feed those features to the net and use them in planner chance spectra. Also train auxiliary heads to predict hidden quantities from public observations; labels may use true simulator hidden state during training because inference only sees public history. ReBeL and Student of Games are overkill for your current two-player no-trade setting, but they support the broader point: imperfect-information search-learning systems usually need an information-state/belief story once masked observation stops being enough. ([arXiv][6])

## 5. You are treating value fragility as a law, but it may be a data-weighting bug

Your report shows the value head is the load-bearing failure point: search lost to raw policy before value repair; multi-epoch reuse hurt value; continuous lineage drift hurt value; the 91M probe’s value blew up on epoch 2.   

But the deeper statistical issue is that your “millions of rows” are not millions of independent value labels. They are tens of thousands of games, with one terminal outcome smeared over hundreds of correlated decisions, plus many forced or low-choice decisions. The value head may be overfitting because the effective value-label sample size is much closer to game count than row count.

Try per-game normalization for value loss, lower value weight on forced rows, phase-balanced value sampling, and short-horizon root-value targets. KataGo later added short-term value/score targets and uncertainty/error predictions; those are exactly the kind of lower-variance scaffolding your value head lacks. ([GitHub][2])

## 6. Your architecture A/B does not close the architecture question

The current model has three known ceilings: adjacency not consumed, actions not cross-attending to board tokens, and value read only from CLS. You built a warm-start-safe upgrade and it lost a comparison, so you kept the simpler net. That was reasonable locally, but not a permanent conclusion. 

A zero-initialized add-on tested with one epoch at low LR can lose because the new modules did not learn, because the value head destabilized, or because the data distribution did not expose the benefit. Architecture changes should be tested with a new-module LR multiplier, value-head freeze or lower LR, policy-only warmup, and at least two seeds. Your own 91M result also says bigger models need fresh data, not more epochs on a frozen corpus. 

## 7. You are not exploiting symmetry hard enough

Your D6 symmetry violation is enormous relative to opening-placement value spread: policy orientation-noise std 0.175 nats versus ~0.06 nats separating 54 candidates, with value std 0.049 and ranges up to 0.29. That is not a small regularization issue; it is a direct opening-strength leak. 

Use test-time symmetry averaging at least for opening placements and other wide, near-tied roots. If full 12-fold at every leaf is too costly, do it adaptively: root only, first two settlement/road phases, or roots with high legal count and low prior spread.

## 8. Your “closed forever” language is too strong

It is fair to kill `c_scale=1.0`, exact-budget SH at n_fast=16, multi-epoch frozen reuse, CPU generation, small eval-net distillation, and BC+PPO for the current system. Your kill list is valuable. 

But “c_visit/c_scale closed permanently” should mean “closed for this architecture, value calibration, action width, and sim budget.” If you later add uncertainty-aware values, reduce candidate action count, use a bigger model, or change root target construction, the optimal scale may move. Do not refund dead arms blindly; just avoid turning local ablations into laws of nature.

# The live decision: what to do about turn 4

I would promote the current +20 Elo candidate into the data-generation role, with rollback guardrails.

The strongest argument is your own plateau evidence: nine consecutive flywheel rounds have flat anchor telemetry, and the current gen-3 self-play window is fully distilled. More training on the same distribution will not create a larger candidate. 

The strongest counterargument is external flatness. My answer is not to ignore it; it is to separate roles:

**Training champion:** advances under regression-protection gating, because expert iteration needs fresh data from the improved policy.

**Release champion:** advances only after external arena confirmation.

**Rollback rule:** if two consecutive training promotions reduce external arena Elo by a meaningful margin, revert data generation to the last externally stable champion and mix the regressed net into the opponent pool rather than deleting it.

**Data mix after promotion:** do not generate 100% from the new champion. Use something like 70–80% current training champion, 10–20% old champions/opponent pool, and 5–15% high-regret restarts or external-bot states. This directly attacks inbreeding while preserving compounding.

# How I would escape the plateau

The ordering matters.

| Rank | Experiment/change               | Hypothesis                                                     | Run                                                                                                 | Result that changes my mind                                  |
| ---: | ------------------------------- | -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
|    1 | Regression-protection promotion | +20 Elo candidates compound if allowed to feed self-play       | Promote training champion, keep release champion; run two flywheel turns                            | If external arena drops significantly twice, stop and revert |
|    2 | Legal belief tracker            | Uniform masked belief leaves robber/dev/endgame Elo on table   | Add posterior features + posterior chance spectra; gate vs same net without belief                  | +20 Elo internal or clear external gain vs catanatron_value  |
|    3 | Opponent-pool data              | Self-play is narrowing; fixed bots expose missing styles       | 10–20% games vs old champions and catanatron bots, train from your search targets                   | Better external arena without internal collapse              |
|    4 | High-regret restarts            | Value head needs more independent, diagnostic targets          | Start games from archive states: external losses, high KL, high value error                         | Improved value calibration and external transfer             |
|    5 | Value auxiliary package         | Scalar terminal MSE is too noisy                               | Short-horizon root values, distributional/two-hot value, uncertainty/error head, per-game weighting | Lower anchor value error and H2H gain without drift          |
|    6 | Adaptive match-time search      | Bot strength is search-limited in openings                     | n=128–256 plus D6 averaging for openings/wide roots; n=64 elsewhere                                 | Immediate external gain vs catanatron_value                  |
|    7 | Reanalyze subset                | Current targets are stale/noisy                                | Reanalyze 1–2M high-impact rows at n=128/256; train one pass                                        | Policy/value target improvement transfers to gates           |
|    8 | Bigger net on fresh data        | Current 35M model is capacity-limited only after data advances | Retry 80–100M after ≥10M fresh rows, with value-safe training                                       | Clear H2H gain without value blowup                          |
|    9 | Architecture v2 properly tested | Action/graph bias helps once trained correctly                 | New-module LR multiplier, policy warmup, 2 seeds                                                    | Beats current net on wide-root/opening buckets and H2H       |

ReZero-style reanalysis is worth testing because it is designed to refresh stale MCTS targets more efficiently, but I would not put it ahead of promotion, belief, opponent-pool data, or high-regret restarts. ([arXiv][7])

# Search: what I would change

Your `c_scale=0.03` result is one of the most credible findings in the report. The mechanism is coherent: at a 54-wide root with 64 sims, min-max rescaling stretches 1–2 sample Q noise into false confidence. 

The fix I would try is **not** “go back to defaults.” It is:

1. **Cap considered root actions at wide roots.** Gumbel top-k should not force one visit to all 54 opening placements. Consider top 16–24 by prior plus Gumbel/noise exploration plus symmetry diversity. If you give 64 sims to 20 plausible actions instead of 54, completed-Q noise becomes less dominant.

2. **Use policy-target pruning.** KataGo’s docs explicitly connect policy target pruning to preserving rare good moves rather than training the net to model background noise. Your full-search policy-target fraction is already small; make those targets cleaner. ([GitHub][2])

3. **Add uncertainty-aware completed-Q only after you have an uncertainty head.** Your D2 James-Stein arm was neutral, but it used estimated variance without a trained value-error model. KataGo’s uncertainty-weighted playouts work because the net predicts when its utility estimates are likely wrong. ([GitHub][2])

4. **Use adaptive search budget.** MiniZero’s progressive simulation result supports increasing simulations during training when the net can use them, but I would apply the idea selectively: more sims for openings/high-entropy roots; fewer for forced/low-branching roots. ([arXiv][8])

5. **Do not overgeneralize ReSCALE.** The 2026 ReSCALE paper is interesting because it finds Gumbel + sequential halving can fix budget scaling in LLM reasoning tasks, but that setting is not your 54-wide stochastic-root value-noise regime. It supports “Gumbel/SH is a useful family,” not “use default Gumbel parameters.” ([arXiv][9])

# Training and value targets

The production recipe is admirably disciplined: one epoch, champion init, soft policy targets, λ=0.5 value blend, final-VP aux, masking on. It is also too conservative around structural value fixes. 

I would test this value package as one coherent branch:

* distributional/two-hot value over win/loss plus VP margin buckets;
* short-horizon root-value heads, e.g. 6, 16, 50 decision horizons adjusted to Catan game length;
* value-error/uncertainty head trained with stop-gradient;
* per-game value-loss normalization;
* lower value weight for forced actions;
* belief-state auxiliary heads;
* EMA/lagged target net for root-value blend, if drift persists.

The reason to test as a package is that the value head is failing as a system. A single isolated tweak may wash because the scalar value target, row correlation, and search dependence are entangled.

KataGo’s later methods are unusually relevant here: short-term value/score targets, uncertainty-weighted playouts, and optimistic policy became part of a strong small-compute self-play system. Your auxiliary heads are currently built but zero-weighted; that is probably the wrong default for Catan.  ([GitHub][2])

# Architecture

I would not rewrite the whole net this week. I would do three targeted things.

First, use **D6 symmetry** at inference for the roots where it matters. You have already measured that symmetry noise is larger than the real opening-placement spread. That is low-risk Elo.

Second, add **belief features** before graph attention. Belief is more likely to move external strength than another transformer block because catanatron_value is a belief-based opponent and Catan hidden state affects steals/dev/endgame.

Third, rerun the action/graph architecture properly later. The current architecture throws away board topology and scores actions through CLS similarity; that is hard to believe is optimal long-term. But the failed v3b A/B should be treated as “not proven under that recipe,” not “architecture does not matter.” Graph-based AlphaZero variants and Catan-specific neural designs exist in the literature, but none are a drop-in answer for your system. Gendre/Kaneko’s Catan work used a cross-dimensional network to handle Catan’s mixed board/card/action structure and reported beating jsettler, but it is not a benchmark-equivalent AlphaZero system against your current target. ([arXiv][10])

Scaling is still attractive because your NN forward is only a small part of leaf cost today, but reprofile after Rust featurization lands. Your report says the current bottleneck is Python/process overhead, not FLOPs; after Rust, the economics may shift.  

# Engineering priorities

Your MPS rollout and Rust featurization are exactly the right kind of engineering: they increase data flow without changing scientific semantics. 

After that, I would prioritize:

1. typed, serialized configs instead of CLI composition;
2. a single config-hash registry for train/generate/gate/eval;
3. an eval server or parallel search driver so leaf evaluations can batch usefully;
4. subtree reuse;
5. Rust-side or compiled tree operations only after profiling confirms Python tree ops remain dominant.

The CLI-default trap is not an ops annoyance; it is a science-corruption vector. You have already had missing flags nearly poison generation/training. Move orchestration away from shell-assembled argparse strings. 

# What others did that is worth stealing

**KataGo:** The main lesson is not just playout-cap randomization. It is “small-compute AlphaZero succeeds by adding auxiliary targets, better data weighting, uncertainty, and domain-specific search/training improvements.” KataGo’s paper reports a 50× compute reduction over comparable methods and surpassing ELF’s final model with fewer than 30 GPUs; that is the closest public compute analogue to you. ([arXiv][4]) Its later docs describe policy surprise weighting, short-term value targets, dynamic variance-scaled cPUCT, uncertainty-weighted MCTS playouts, and optimistic policy. These map much more directly to your plateau than another vanilla self-play round. ([GitHub][2])

**mctx / Gumbel MuZero:** mctx recommends Gumbel MuZero and notes the policy-improvement guarantee if action values are correctly evaluated. Your failure mode is precisely that values are not correctly evaluated at wide low-budget stochastic roots, so your ablation is not heresy; it is a boundary condition of the guarantee. ([GitHub][3])

**MiniZero / LightZero:** MiniZero is useful mostly as evidence that algorithm choice and simulation schedule are game-dependent, and that progressive simulation can help board games. LightZero is useful as a reference implementation hub for MuZero, Sampled MuZero, Stochastic MuZero, Gumbel MuZero, ReZero, and related methods. ([arXiv][8]) ([GitHub][11])

**Go-Exploit and regret-guided search control:** These are very high relevance. They are about starting from states of interest to improve value generalization and sample efficiency. Your value head is starving for independent, high-information targets; this is the most direct public literature match. ([arXiv][1])

**ReBeL / Student of Games / DeepNash:** These say imperfect information often needs more than masked perfect-information search. I would not import full game-theoretic machinery yet, because two-player no-trade Catan has a much more tractable belief structure than poker or Stratego. But I would absolutely import the principle: operate on legal information states, not raw masked observations alone. ([arXiv][6])

**Catan public landscape:** The public Catan AI literature I found is much thinner than Go/chess/poker. Gendre/Kaneko’s 2020 paper is the standard RL reference and reports outperforming jsettler; catanatron is an actively maintained simulator/AI project whose stated goal is strong Catan bots. I did not find a public, leak-free, Gumbel/AlphaZero-style two-player no-trade Catan system benchmarked above catanatron_value. ([arXiv][10])

# Answers to your eight questions

**Q1 — Promotion criterion.** Use regression-protection gating for the training champion, not +30 Elo certification. Keep paired color-swapped seeds and pentanomial accounting. Add external arena tripwires. Promote the data generator on likely non-regression; promote the release champion only on absolute validation.

**Q2 — Escaping the plateau.** First promote. Then add opponent-pool data, high-regret restarts, legal belief tracking, and value auxiliary targets. Reanalyze and bigger nets come after that. More training on the current window is contradicted by your own anchor telemetry.

**Q3 — Compression trend.** Some compression is expected at fixed net/search/data. But your pattern could also indicate search-target ceiling, value-target ceiling, or inbreeding. Distinguish them by: high-sim teacher reanalysis; n=64/128/256 cross-gates; raw-policy vs search deltas; phase-specific value calibration; opening-placement regret; and external arena transfer.

**Q4 — External-transfer gap.** Some internal inflation is normal; +150 internal versus +70 external is not shocking. The last +20 showing no external movement is an inbreeding warning. Solve with population evaluation and mixed data generation, not by freezing the flywheel.

**Q5 — Value-head fragility.** Structural fixes: distributional value, short-horizon root values, uncertainty/error heads, per-game weighting, forced-row value downweighting, EMA target net, belief auxiliaries, and maybe ensemble/MC-dropout uncertainty for search. Avoid “just more epochs.”

**Q6 — Wide stochastic roots.** Keep small `c_scale`. Add root action candidate caps, policy-target pruning, D6 root averaging, adaptive sims, and uncertainty-aware completed-Q after training an uncertainty head. Do not resurrect exact-budget n16.

**Q7 — Architecture.** The failed v3b A/B is not enough to kill graph/action architecture. But I would not spend the next week there. Do belief + data + value first. Then rerun action cross-attention/graph bias with new-module LR, policy warmup, and multiple seeds. D6 equivariance or at least systematic D6 averaging is more urgent.

**Q8 — Anything not asked.** You need a **Catan-specific error atlas**. Which decisions lose to catanatron_value? Opening node selection? Robber placement? Dev-card buying? Hidden-VP belief? Longest road races? Discards? Endgame risk? Without that atlas, you are optimizing a single scalar win rate and guessing at mechanisms.

# Final blunt recommendation

For the next serious run, I would do this:

Promote the +20 Elo candidate as the training champion under regression guardrails. Immediately wire opponent-pool and high-regret restart generation. Build legal belief tracking. Turn on targeted D6 averaging/adaptive high-search for openings and wide roots. Run a 1000+ game external arena, not just a 200-game panel. In parallel, prepare a value-head package with short-horizon, uncertainty, distributional, and per-game-weighted targets.

The current system is not “whole-system wrong.” The core loop is working: three promotions, a repaired search operator, masked regime, correct paired gates, and real throughput. The main wrong thing is that you are still acting like you are in discrete-generation AlphaZero mode after switching to a continuous small-compute flywheel. The project now needs population data, legal beliefs, lower-variance value learning, and a gate that allows small real improvements to compound.

[1]: https://arxiv.org/abs/2302.12359?utm_source=chatgpt.com "Targeted Search Control in AlphaZero for Effective Policy Improvement"
[2]: https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md "KataGo/docs/KataGoMethods.md at master · lightvector/KataGo · GitHub"
[3]: https://github.com/google-deepmind/mctx "GitHub - google-deepmind/mctx: Monte Carlo tree search in JAX · GitHub"
[4]: https://arxiv.org/abs/1902.10565 "[1902.10565] Accelerating Self-Play Learning in Go"
[5]: https://arxiv.org/abs/2602.20809?utm_source=chatgpt.com "Regret-Guided Search Control for Efficient Learning in AlphaZero"
[6]: https://arxiv.org/abs/2007.13544?utm_source=chatgpt.com "Combining Deep Reinforcement Learning and Search for Imperfect-Information Games"
[7]: https://arxiv.org/abs/2404.16364?utm_source=chatgpt.com "ReZero: Boosting MCTS-based Algorithms by Backward-view and Entire-buffer Reanalyze"
[8]: https://arxiv.org/abs/2310.11305?utm_source=chatgpt.com "MiniZero: Comparative Analysis of AlphaZero and MuZero on Go, Othello, and Atari Games"
[9]: https://arxiv.org/abs/2603.21162?utm_source=chatgpt.com "Revisiting Tree Search for LLMs: Gumbel and Sequential Halving for Budget-Scalable Reasoning"
[10]: https://arxiv.org/abs/2008.07079?utm_source=chatgpt.com "Playing Catan with Cross-dimensional Neural Network"
[11]: https://github.com/opendilab/LightZero "GitHub - opendilab/LightZero: [NeurIPS 2023 Spotlight] LightZero: A Unified Benchmark for Monte Carlo Tree Search in General Sequential Decision Scenarios (awesome MCTS) · GitHub"
