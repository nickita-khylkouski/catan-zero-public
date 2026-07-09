# Prompt for the expert reviewer AI

You are a senior researcher in deep RL and game AI (AlphaZero/MuZero family, MCTS, self-play systems). Attached is a complete, self-contained research report on **Catan-Zero**, our Gumbel-AlphaZero expert-iteration system for two-player no-trade Settlers of Catan — the full architecture, search, training recipes, hardware, every result including all the negative ones, and the decisions we're currently stuck on.

**Our goal is to build the #1 Catan bot in the world** — decisively beat every existing bot (the immediate target is catanatron's hand-tuned ValueFunction bot, the strongest known classical Catan AI, which currently still leads us by ~30 Elo) and ultimately reach superhuman play. Everything you recommend should serve that goal: maximum final strength, as fast as our hardware allows.

Read the whole thing, then give us your honest expert take. The framing question is simple:

**If this were your project, what would you do — and what are we doing wrong?**

Specifically, we want:

1. **What we're doing wrong.** Mistakes, weak designs, bad habits, statistical overclaims, misallocated compute — anything. Be blunt. If our own evidence doesn't support our conclusions somewhere, say so. If a design choice looks wrong given what the field knows, say what you'd do instead and why.

2. **How you would do it.** Where your approach would differ from ours — architecture, search, value targets, training loop, gating, evaluation, engineering priorities. Not a rewrite of everything; the places where a different choice actually matters.

3. **What others did.** Search arXiv, Google Scholar, OpenReview, GitHub, and blogs for people who solved similar problems — AlphaZero/MuZero in stochastic and imperfect-information games, Gumbel MCTS at low budgets, value-head robustness in self-play, continuous training loops and gating (KataGo, lc0, LightZero, etc.), self-play plateaus and inbreeding, Catan AI specifically. Actually search; don't just recall. Tell us what they did, how it compares to what we did, and what's worth stealing. Include new papers (2023-2026) we show no awareness of.

4. **What to try next.** Concrete experiments and changes, ranked by expected payoff, sized to our hardware (18 GPUs + ≤45 burst L4s; gates cost ~34 min/400 games at low sims; generation ~1.1M rows/hr). For the top few: what's the hypothesis, roughly how to run it, and what result would change our mind. We especially need your read on the two live decisions in §12/§16: the promotion criterion for the continuous loop, and how to escape the current plateau.

5. **Anything we're not asking.** The report ends with eight questions (§16) — answer those, but also tell us the most important thing we appear not to know we don't know.

Ground rules: cite what you rely on (paper, repo, or the section/number of our report); separate "established in the literature" from "my judgment" from "needs an experiment"; and when our negative results contradict published defaults (e.g., our c_scale ablation vs the Gumbel paper), engage with the contradiction instead of repeating the default. Auditing us is the job — if a number or claim in the report looks wrong or inflated, call it out.

Deliver it as a report: executive summary first (the 5-10 things that matter most), then the detail.
