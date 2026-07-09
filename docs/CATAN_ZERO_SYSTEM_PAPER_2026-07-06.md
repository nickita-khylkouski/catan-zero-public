# Catan-Zero: An Expert-Iteration System for Two-Player Settlers of Catan

**A system-and-engineering research narrative**
**Status as of 2026-07-06**

> Audience: a strong ML researcher joining the project cold. This document describes the entire system stage by stage, the reasoning behind each design decision, the failures and dead ends that shaped it (they carry most of the lessons), and what is happening right now. Where a design choice derives from published work, that work is named. Numbers are pulled from measured runs; where two sources disagreed, the more recent end-to-end-verified measurement wins.

---

## Abstract

Catan-Zero is an AlphaZero-style expert-iteration system targeting the strongest possible bot for **two-player, no-trade Settlers of Catan** (standard 10-victory-point rules) on top of a custom Rust game engine, `catanatron_rs`. The system began as behavior cloning (BC) on hand-coded teacher bots plus a PPO fine-tune; that lineage plateaued for reasons we now understand precisely (no mechanism to exceed the teachers, a mis-wired PPO learner, noisy teachers, and an evaluation layer that laundered noise). In July 2026 the project pivoted to **Gumbel-AlphaZero self-play with explicit chance nodes**, keeping BC only as the initial policy prior.

The current pipeline is: (1) a masked, leak-free BC checkpoint (`v3a`) serves as the base policy/value network; (2) that network drives Gumbel MCTS self-play over the true stochastic game, producing improved-policy and outcome targets; (3) the targets are distilled back into the network; (4) a paired, color-swapped pentanomial SPRT gate decides whether the new network is genuinely stronger; (5) iterate. The pivotal result — reached 2026-07-06 — is that **generation 1 beats its own base `v3a` 57.0% (228–172 of 400 games, 95% CI [52.1, 61.9]) at low search budget**, the first empirical proof that the flywheel turns: distilling searched play produced a stronger network.

The paper documents the neural architecture (a ~35M-parameter entity/graph transformer), the Gumbel search over enumerated dice chance nodes, a **hidden-information leak** that silently invalidated every earlier external strength claim (and the fail-closed masking regime that fixed it), the self-play data-generation knobs (playout-cap randomization, temperature, seed disjointness, truncation labeling), training, the pentanomial statistics of gating, and the results to date. It closes with a four-agent **audit wave** run on 2026-07-06 that found several latent-but-serious bugs (an unmasked gate path, a missing `--max-steps`, provenance gaps) and set the research direction toward a KataGo-style continuous windowed-replay flywheel. Throughout, a recurring theme: **infrastructure discipline is research velocity** — the hardest-won lessons here are about seed collisions, CLI default drift, orphaned worker processes, and provenance, not about network architecture.

---

## 1. Problem and Environment: why two-player no-trade Catan is hard

Catan is a deceptively hard testbed for game AI, and the two-player no-trade variant isolates the parts that matter most for a self-play learner while removing the parts (multi-party negotiation, coalition dynamics) that are orthogonal to search-and-learn.

**Stochasticity / explicit chance nodes.** Every turn begins with a 2d6 dice roll that distributes resources; the robber (on a 7) steals a random card; development cards are drawn from a shuffled deck. Unlike Go or Chess, the game tree is not deterministic — it contains **chance nodes** with non-trivial branching (11 distinct dice sums, robber steal outcomes, dev-card draws). A search that ignores this and treats a single sampled roll as "the" transition will systematically misvalue positions. Our search therefore models chance nodes explicitly (Section 4).

**Hidden information.** Each player's hand composition and unplayed development cards (including hidden victory-point cards) are private. A network or search that can *see* the opponent's hand is solving a strictly easier game than the one it will be evaluated in. This sounds obvious, but it was the single most consequential bug in the project's history (Section 5).

**Long horizons.** Games run ~150 decisions typically and up to a 600-decision cap in adversarial self-play. Value signal is sparse (win/loss at the end) and credit must propagate across hundreds of stochastic transitions.

**Variable, structured action spaces.** The legal action set changes every ply and ranges from a single forced action (must-discard, must-roll) to ~54 candidates at an opening settlement placement. Wide roots are where search is both most valuable and most fragile — a fixed simulation budget spread over 54 candidates leaves barely more than one simulation each, which turned out to be the mechanism behind an early search-vs-policy failure (Section 4, "the c_scale question").

**The engine.** All of this runs on `catanatron_rs`, a Rust reimplementation of the Python `catanatron` engine, shipped as a custom wheel (`0.1.2`). It provides a **corrected longest-road rule** (the source of a whole class of engine-equivalence bugs — see below) and a batched chance-apply API, delivering roughly **5.86× the simulation throughput** of the reference engine. Trusting this engine took real work: a 1000-game fixed-pair equivalence sweep against the vendored Python engine drove divergences from ~60% down to ~2%, and closing the remaining gap required fixing **eight** subtle rules bugs (labeled A17, A24, A26, A27, A28, and others), each verified by TDD against specific replay seeds. Longest-road, in particular, is a graph problem (longest acyclic path with incumbent-aware tie-breaking on settlement severs) that is easy to get wrong by ±1 road length in loop-closing and enemy-cut-node cases. **Deploying the corrected wheel fleet-wide was explicitly treated as a rules change, not a silent perf swap**, because it changes the data distribution.

---

## 2. System Overview

The pipeline, end to end:

```
  teacher bots (AlphaBeta / value bots)
        │  behavior cloning
        ▼
  BC network  ──►  [HIDDEN-INFO LEAK found + fixed]  ──►  masked re-baseline (v3a)
        │
        │  v3a is the base policy+value prior
        ▼
  Gumbel MCTS self-play  ──►  improved-policy + outcome targets  (shards on disk)
        │       (over TRUE chance nodes, masked/public observation)
        ▼
  distillation training (train_bc.py, init-from-previous, 1 epoch)
        │
        ▼
  paired color-swapped pentanomial SPRT gate (v_new vs v_base)
        │   pass → promote → new base
        ▼
  iterate  (gen-1 → gen-2 → …)   [moving to continuous windowed replay]
```

The distinction from vanilla AlphaZero worth flagging up front: the network is **initialized from BC, not random**, because Catan's action space is large and structured and a cold-start random policy generates almost no informative self-play. BC gives search a usable prior; search then improves on it; distillation captures the improvement; the gate certifies it. That loop is **expert iteration** (Anthony et al. 2017 "Thinking Fast and Slow"; the AlphaGo Zero / AlphaZero family, Silver et al. 2017/2018). Our search backbone is the **Gumbel MuZero** policy-improvement operator (Danihelka et al. 2022), chosen for its behavior at small simulation budgets (Section 4).

---

## 3. Neural Architecture and Observation Encoding

The network is a **~35M-parameter "entity_graph" transformer**. Its distinctive feature is that the board and game state are encoded as **entity tokens and graph tokens** rather than as a flat feature vector or a fixed CNN grid.

**Why entity/graph tokens.** Catan's board is a hexagonal graph: 19 hex tiles, 54 settlement nodes, 72 edges, plus per-player state. A CNN grid is an awkward fit for a hex adjacency structure, and a flat vector throws away the relational structure that determines everything (which node is adjacent to which tiles, which edges connect which nodes, whose roads touch whose). Representing tiles, nodes, edges, and players as **tokens** and letting attention learn the relations is a natural fit — it lets the same attention machinery reason over "this settlement is on a 6-8 wheat/ore corner" and "this player is one card from a dev-card VP." The concrete config:

- entity tokens + graph tokens; **6 graph attention layers**; hidden size **640**; **8 attention heads**; dropout **0.05**.
- **policy head** (over the legal-action catalog) and **value head** (scalar, ±1 outcome).
- Built-but-unweighted **Catan-native auxiliary heads** (KataGo-style ownership/score-style targets), staged for later — never allowed to be a primary value target.
- Observation width **806** in the masked / public-observation regime.

Two architectural ceilings were found by audit and are relevant to anyone reading the code: (1) the adjacency-id tables the featurizer builds are, in the current architecture, **never actually consumed by the model**; (2) action tokens do not cross-attend to board tokens (actions are scored by cosine-to-CLS), and the value head reads only the CLS token. A warm-start-safe architecture upgrade (action cross-attention, target-id gather, value attention pooling, all zero-initialized so it reproduces the current net bit-for-bit at init) was built on a branch (`f69`) and is a candidate for the next generation; it is **not** in the gen-1 base.

A separate, cheaper lever than the arch change was discovered during the "value noise dominates placements" investigation: the value net **badly violates the board's D6 (hexagonal) symmetry**. Across 12 rotations/reflections of the *same* opening position, the network's value output has std ~0.049 and prior-orientation noise std ~0.175 nats — *larger* than the ~0.06-nat spread that separates the 54 placement candidates it is supposed to rank. So a large share of what looked like irreducible value noise is really **violation of a known invariance**. Test-time 12-fold symmetry averaging gives a ~3.3× denoise (near √12, i.e. errors decorrelate) with **no retraining**, and a `--symmetry-augment` training flag is a candidate recipe. This is a clean example of the general principle: measure whether your "noise" is actually a symmetry your model failed to learn.

---

## 4. Search: Gumbel MCTS over enumerated chance nodes

### Why Gumbel, not classic PUCT

Classic AlphaZero uses PUCT selection with Dirichlet noise at the root for exploration, and it is tuned for **hundreds to thousands** of simulations per move. Our per-decision simulation budget is small (tens of simulations) because each leaf evaluation is expensive (Section 6 — a full search can be thousands of leaf evals once chance nodes fan out). At small budgets, classic PUCT + Dirichlet is known to be noisy and to under-use the prior.

The **Gumbel MuZero** operator (Danihelka et al. 2022) is designed exactly for this regime. At the root it draws **Gumbel noise once per action and samples the top-k candidates** (this *replaces* Dirichlet noise — there is no separate exploration-noise knob), then allocates the simulation budget across those candidates via **sequential halving**, and finally forms an **improved policy** from the visit-completed Q-values (`completed-Q`: visited actions use their estimated Q, unvisited actions are completed from a mixed value). The guarantee is a *policy improvement* even with as few as one simulation per candidate. That guarantee is why Gumbel is the right backbone for a small-budget, expensive-leaf game. Our implementation (`gumbel_chance_mcts.py`) mirrors the semantics of `mctx.gumbel_muzero_policy`, including the sequential-halving schedule and the completed-Q formation.

### Enumerated chance nodes

Because chance is real (Section 1), interior transitions through a dice roll are modeled as **chance nodes**. The naive-but-correct version enumerates dice outcomes recursively at every tree depth: one `n_full=64` full search then costs **~5,400 leaf evaluations**, not 64 — the ~11-way dice fan-out compounds with depth. A 40-decision game is ~35,000 evals; a 600-decision game approaches **~500,000 evals**. This combinatorial chance enumeration, not the hardware, is the root cause of generation being the day-long pole of the pipeline (Section 6).

### Lazy interior chance

The mitigation is **lazy interior chance** (`lazy_interior_chance=True`): the root chance node is still enumerated, but *interior* ROLL nodes are **single-sampled** (one dice outcome, via the existing single-sample traversal path). This drops a full search from ~5,400 to tens of evals — a **13–19×** speedup depending on cadence. The statistical justification is that single-sample interior backups are **unbiased** (±1.81 SE, checkpoint-independent). The caution: lazy search agrees with full enumeration on only ~43% of interior decisions vs a ~57% same-semantics variance floor, so there *is* a real systematic difference on ~15% of states, and by the project's standing rule only a **strength H2H** — not a fidelity-to-noise-floor metric — can decide whether that difference is harmful.

That rule was learned the hard way. An earlier lazy-128 configuration "looked within the target noise floor" on KL and argmax-stability metrics yet **lost 2–21 to the raw policy** in strength H2H. The real early-warning signs were not the target-fidelity numbers; they were **cross-agreement collapse (37–53%)** and **root-value noise (6–9× the floor)**. The "lazy loses 21-2" result was later shown to be a **stale-checkpoint artifact** — it was measured on the pre-value-repair broken-value-head net, on which *enumerated* search also lost to raw. On the repaired `v3a` masked checkpoint, lazy recovers to near-parity. Gen-1 generation ran on lazy interior chance with 0% truncation (a quality win over the ~22–27% truncation of the enumerated base). The lazy-vs-full non-inferiority SPRT is still formally "continue" (58.3% over 24 pairs) and sits on the science backlog.

### completed-Q, the rescale, and the c_scale question

The improved policy is `softmax(logits + σ(completed_q))`, where `σ(q) = (c_visit + max_visits) · c_scale · rescale_to_unit_interval(q)`. The **`rescale_to_unit_interval`** step is a stock-mctx behavior (`_rescale_qvalues`, min–max to [0,1] with only a 1e-8 epsilon floor) and it is the origin of an important failure mode we call **Gate-A**.

**Gate-A (2026-07-04):** full-64 search *lost to its own raw policy* (19.2% / 22.1% on two replicates, SPRT H0). A controlled trace over 40 real placement roots found the mechanism: at a 54-wide placement root with 64 simulations, each candidate gets ~1.2 sims, so completed-Q is dominated by 1–2-simulation raw-Q noise (spread ~0.04). The min–max rescale then **stretches that noise to fill [0,1]**, manufacturing false confidence that swamps the (near-tied) prior. 74.6% of the losses were placement blowouts. This is a genuine property of stock mctx — no prior report of it exists in the mctx issue tracker or the literature — and is considered publishable as a short note.

Ablation across the mctx knobs found that **`c_scale=0.03` removes the harm** (pooled 51%, i.e. search stops hurting) — the default dataclass value is `c_scale=0.1`, and gen-1 runs at `0.03` via CLI. This is the "c_scale question": **`c_scale=0.03` is 33× below the Gumbel paper's validated `(c_visit=50, c_scale=1.0)` pair**. We have only ever ablated `c_scale` *alone*; the 2026-07-06 research verdict flags that our low value may be *compensating* for something else (the rescale-noise problem, or the value-head calibration gap) rather than being intrinsically correct, and a **joint `c_visit × c_scale` re-ablation is queued**. Two flag-gated search-side arms were also built as fallbacks: **D1** (`rescale_noise_floor_c`, blends the rescaled Q toward 0.5 by a signal-to-noise ratio) and **D2** (`variance_aware_q`, James-Stein shrinkage of each completed-Q toward the mixed value by its standard error, adapted from the UCT-V-P idea in arXiv:2512.21648). Both default OFF (exact no-op) and neither is in gen-1.

The deeper fix that made Gate-A pass was not a search knob at all — it was **value-head repair** (Section 5/9): after retraining the value head on self-play outcomes, post-repair search beats raw **67–71%** (was 19%). The lesson: a search pathology can be a *value-calibration* pathology in disguise.

---

## 5. The hidden-information leak and the masking regime

This is the most important correctness story in the project.

**The leak (confirmed empirically, 2026-07-05).** The featurized observation handed to the network included **the opponent's full hand composition** (player-token slots 16–20), **unplayed development-card identities including hidden VP cards** (slots 22–26), and **actual victory points** (slots 4–5). Separately, the planner's chance spectra used the victim's *true* hand for robber-steal outcomes and the *true* remaining deck for dev-card draws. (The live environment's transitions were correct; the leak was entirely on the observation/planner side.)

**Why it invalidated results.** `catanatron`'s reference value and AlphaBeta bots are deliberately **belief-based** — they re-add face-down cards to the deck and model steals as uniform. So *every* external comparison our omniscient network made was **omniscient-vs-belief**, which is not a fair strength claim; it is a strictly easier game for our side. The previously reported "−147 Elo vs baseline" was itself measured *with* omniscience, so the true gap is worse, not better. (Internal search-vs-raw H2H stays valid because the leak is symmetric there.)

**The fix (`#71`, public-observation masking).** Three pieces, all opt-in behind flags:

1. A canonical token-level transform `mask_player_tokens_public` zeros the non-actor hidden slots ({4,5,15,16–20,21,22–26}), keeping only public counts. This is wired into training via `train_bc --mask-hidden-info` (masks the banked corpus at load time — **no regeneration needed**) and into inference via `EntityGraphRustEvaluatorConfig.public_observation`.
2. A perspective-relative masking in the Rust evaluator (`_mask_players_to_public`) so search sees only public state.
3. A planner-only `belief_chance_spectra` flag (uniform steal; belief deck = base deck minus actor's own minus all played cards); the live environment is untouched.

Model-invariance was **proven**: with `public_observation=ON`, permuting the opponent's hidden hand changes the value by <1e-5 and logits by <1e-4; with it OFF, it leaks. 9 new tests plus 90 affected tests pass.

**The fail-closed guard.** Because whether a checkpoint was masked-trained is a *correctness* property that must match the *evaluation* regime, the system records `mask_hidden_info` in the checkpoint and **asserts at eval time that the recorded flag matches the requested `--public-observation` flag** — if they disagree, it fails closed rather than silently producing a wrong-regime number. This matters because `train_bc` historically did **not** serialize the flag anywhere durable (it was a module global absent from `report.json` and `train.log`), so directory names like `_masked_` are *intent, not proof*. That gap already bit us once: a checkpoint named `value_repair_v4` turned out to be **omniscient** because its launch script lacked `--mask-hidden-info`. To verify pre-guard checkpoints, we built a **controlled both-regime calibration test** (`masked_vs_unmasked_calibration_check.py`): a net calibrates better in its own training regime, so you compute `corr(q,z)` on the same held-out states twice (as-stored vs masked) — but you **must** include a known-omniscient control so you can distinguish "this net is masked-trained" from the confound "masking universally helps calibration." On a 456,658-row holdout: `v3a` masked 0.733 vs unmasked 0.683 (+0.050 → masked), `v3b` +0.062 (→ masked), and the omniscient control `v4` −0.031 (→ omniscient, opposite sign). The control discriminating in *both* directions is what makes the test valid; `v3a`/`v3b` are genuinely masked-trained and gen-1's base is leak-free.

The general lesson, recorded as a standing rule: **any correctness fix shipped as an opt-in flag is a default-override trap.** You must audit every launch command, not just the code — because argparse defaults silently override dataclass defaults, and a fix that is `default=OFF` will not protect a pipeline that forgot to turn it on.

---

## 6. Self-play Data Generation

Generation (`generate_gumbel_selfplay_data.py` → `gumbel_self_play.py`) is where the network's search-improved decisions become training targets. Every knob has a reason.

**Playout-cap randomization (PCR).** Following KataGo (Wu 2019), each decision is randomly either a **full search** (`n_full=64`, probability `p_full=0.25`) or a **fast search** (`n_fast=16`). The crucial asymmetry: **only full-search rows carry a policy target** (`policy_weight_multiplier = 1.0` for full, `0.0` for fast/forced), while **all rows carry a value target**. This is KataGo's PCR lever — cheap fast searches still give you value signal (the game outcome is the same regardless of search depth at a given state), but you only distill the *policy* from decisions you actually searched hard. The effective policy-sample fraction is ~7.7% of rows. Forced-action states (≤1 legal action, e.g. must-roll/must-discard) also get `multiplier=0` regardless of search — there is no policy to learn there. (Audit note: this value-only-on-fast-rows design was **confirmed correct** in the 2026-07-06 review, matching KataGo's documented behavior.)

**Temperature.** Action *sampling* uses `T=1.0` for the first `round(max_decisions × temperature_move_fraction)` decisions of a game, then `T=0.0` (argmax) thereafter. With `max_decisions=600` and `temperature_move_fraction≈0.075–0.15`, that is roughly the first ~45–90 decisions. Temperature promotes opening diversity so the corpus isn't 10,000 copies of the same game. **Critically, temperature is applied only to which action is played, never to the recorded improved-policy target** — the target is always the un-tempered search-improved distribution. Getting this wrong would train the network to imitate its own exploration noise. (A related trap: `--max-decisions` also gates the temperature cutoff via `fraction × cap`, so changing the cap without rescaling the fraction silently doubles the exploration window — the 600 cap needs fraction 0.075 to preserve the ~45-decision cutoff.)

**Seeds and reproducibility.** Each game uses `game_seed = base_seed + index`; the board is `Game.simple(colors, seed=game_seed)` and the chance RNG is `Random(game_seed ^ 0xA17E)`, deliberately decorrelated from the board seed so board layout and dice history are independent but both fully reproducible from `game_seed` alone. This determinism is what makes the paired-gate and the regret-restart replay (Section 12) bit-exact.

**Seed disjointness — the collision that nearly poisoned a generation.** Because the chance RNG is a deterministic function of `game_seed` and search is deterministic given a checkpoint, **two workers that draw the same `game_seed` produce bit-identical games**. A 64-seed block (314000–317015) was once assigned to *both* A100 hosts, producing 128 pairs that were really 64 duplicated pairs — the aggregator double-counted them into a **false H1 (significant) verdict**. The permanent fix was a **seed fleet planner** (`#77`) that hands every worker a globally disjoint seed range, plus a duplicate-`game_seed` detector in the corpus builder as defense-in-depth, plus an aggregator that dedupes exact-duplicate games (by a `(game_seed, orientation, search_color, winner, …)` signature) *before* pairing. This class of bug also silently corrupted a *staged* gen-1 generation (7/8 GPU-index seed overlap), caught pre-launch. Seed hygiene is not a nicety here; it is the difference between a real result and a fabricated one.

**Truncation labeling (F3).** Games that hit the 600-decision cap without a winner are not discarded — their rows are kept with a **VP-margin proxy label** (whoever is ahead on victory points is treated as the likely winner). Discarding truncated games biases the corpus toward short, decisive games; keeping them with a proxy label preserves the long-game distribution. The 600 cap was itself chosen after a 300 cap produced 44–47% truncation and *zero* decisive pairs in gating — you cannot run a paired strength test on games that don't finish.

**Throughput reality.** The per-leaf cost model (measured, B200) is instructive: GPU ≈ 3.4 ms/leaf of which the **NN forward is only ~0.18 ms (~4%)** — featurization (pure-Python `entity_token_features.py`, ~42%) and Rust FFI/JSON marshalling (~26%) dominate. So the single biggest throughput lever is not a faster GPU or a smaller net; it is **moving featurization into Rust** (a ~3.4× win, ranked the top open optimization). On CPU (int8, for Modal), the forward *does* dominate (34–36 ms/leaf, flat vs batch size at one thread), which is why CPU batching gives ~0 benefit and why a Modal-CPU fleet was NO-GO for gen-1 (at the current eval count, 8–12k games = 45,600–68,400 core-hours → 19–28h even at 2400 cores, worse than the GPU fleet). Modal only wins *after* the eval count is cut (lazy chance + a distilled eval net).

---

## 7. Training (Distillation)

Training (`train_bc.py`, ~5,700 lines) turns the self-play shards into the next network. It is deliberately cheap — gen-1 trained in **540 seconds** on **2,736,128 rows** — because in this regime **sample reuse, not training FLOPs, is the free lever**.

- **Init from the previous generation** (gen-1 init from `v3a`'s `checkpoint_masked.pt`), not from scratch. Expert iteration compounds on the prior network.
- **1 epoch** for gen-1, **batch 4096**, LR warmup 100 steps (`--lr` default 2e-4; gen-1 value-repair used 3e-5). *Audit caveat:* 1-epoch-on-one-generation is exactly **KataGo's documented big-net overfit trap** — the policy head only sees gradient from ~7.7% of rows, so ≥2 epochs and effective reuse ≫1 are queued training quick-wins.
- **Soft policy targets** (weight 0.9) renormalized over the *legal* support — the network is distilling the search's improved distribution, not a one-hot.
- **Value loss on the ±1 outcome**, per-actor sign (the winner-sign is propagated so each row's value target is from that actor's perspective). `--value-loss-weight` default 0.25, but the gen-1/AZ-board-game recipe resolved this to **1.0** (AZ uses a 1:1 policy:value ratio on Monte-Carlo outcomes; MuZero's 0.25 was for Atari TD targets, a different situation).
- **Gradient clip 1.0**, global-permutation epoch shuffle.
- **Masking is applied at batch level** so it is regime-safe for both the in-RAM `npz` path and the streaming `memmap` path (byte-identical).
- **Memmap streaming loader** (`--data-format memmap`, `#66`) removes the host-RAM corpus ceiling. In-RAM npz is ~43.8 KB/row (so 32.6M rows would need ~1.43 TB — impossible on a 708 GB host); the memmap corpus is ~13.7 KB/row on disk and cuts training peak RSS from 26.75 GB to 10.5 GB (2.6×). It is ~24% slower synchronously, partly recovered by a thread prefetcher. (A batch of 16384 OOMs the 35M net on a 178 GB B200; batch 4096 is the safe ceiling.)

**A dead end worth recording: value-repair v1.** The first attempt to fix the miscalibrated value head trained *value-only* with a frozen trunk (bit-identical policy logits, proven). It came back *regressed* on rollout-value calibration, and two follow-up diagnostics (10× LR; 10× LR + neutral phase weights) both showed **flat loss**. Conclusion: the value head is already at its objective's optimum *on the BC corpus*. Repair cannot come from more corpus training — **it must come from self-play outcomes**. That is exactly what value-repair v2 did (retrain on true self-play outcomes), and it is what reversed Gate-A. The frozen-trunk tension is real: if calibration stalls at ~0.6–0.65, the prime suspect is the frozen imitation representation itself, and an unfreeze-with-policy-KL-distillation variant is pre-planned.

---

## 8. Evaluation and Statistics

**Why paired, color-swapped games.** Catan is asymmetric — the first player has an advantage, and board layout matters. Comparing two networks on *independent* games conflates network strength with seat/board luck. Instead we play each seed **twice with colors swapped** so both networks see both sides of the same board and dice history. This is the standard chess-engine paired-game design.

**Why pentanomial, not binomial.** A naive binomial SPRT on individual game wins ignores the correlation *within* a paired game — the two color-swapped games on the same seed are not independent draws. The correct object is the **pentanomial** outcome of the *pair* (both-loss / loss-draw / split / win-draw / both-win) and its variance, following the Van den Bergh generalized SPRT (GSPRT) math used by fishtest/fastchess. Using binomial here overstates significance. The gate (`sprt_gate.py`) ports the elo0/elo1 → LLR math; standard promotion gates use `--elo0 0 --elo1 30` (resolves a real ~55%-target effect within budget; the tighter `elo1 5` is reserved for small effects and needs far more games).

**Low-vs-high simulation sensitivity — where distillation gains show.** This is a subtle and important evaluation finding. **Distillation gains appear at LOW search budget and wash out at HIGH search budget.** The intuition: at low budget the network's *prior* dominates the decision, so a better prior (a better-distilled network) wins; at high budget, deep search corrects *both* sides toward the same strong moves, erasing the prior difference. This matches the literature (prior-dominated vs search-dominated regimes). The practical consequence: **the low-search H2H is the true "did the network improve?" signal**, while a high-search H2H measures "did the network improve *beyond what search already recovers at production budget?*" — a different, harder question that needs its own large-sample run. It also means a search-vs-own-raw *ratio* is not a cross-net strength claim: a higher ratio can mean a *weaker raw policy*, not a stronger network.

**Operational eval discipline (hard-won).** Gates must be **peekable and multi-GPU from the start**: the original gate tool used `Pool.map` and wrote results only at the end, producing a 2.5-hour blind run with no ETA and no partial signal. The fix was per-worker progress files (`progress/worker_*.json`) summed for a live win rate, plus `--devices cuda:0,cuda:1`. And **~6–8 workers per GPU is optimal** — 20 workers on one GPU is catastrophic oversubscription (the GPU saturates and per-game latency balloons). Full-search H2H games are ~3–5 minutes *each* at `n_full=64`, so a 600-game full-search gate is inherently ~2 hours on 1–2 GPUs; use `n_full=8` for a fast sensitive read (~34 min / 400 games) and reserve `n_full=64` for a final confirm.

---

## 9. Results to Date

**The pivotal result (G1 gate, 2026-07-06).** Generation 1 — trained on the ~10k-game (2,736,128-row) corpus of `v3a` search-improved self-play — **beats its own base `v3a` 57.0% (228–172 of 400 games), 95% CI [52.1, 61.9]**, significant, ~+49 Elo, at `n_full=8` (low search). Zero truncations, zero errors; ~34 minutes on 2 B200 GPUs. **This is the first empirical proof that the expert-iteration flywheel turns**: distilling `v3a`'s searched play produced a genuinely stronger network. A small 32-game `n_full=64` read showed ~47% (noise, ±9%) — inconclusive, and consistent with the low-vs-high wash-out above; a proper large-sample high-search confirm (`n_full=64` on both GPUs with progress files) is running (~57% through 188 games).

**What raw-vs-searched vs AlphaBeta means.** Gen-1's *raw policy* loses to `catanatron` AlphaBeta-depth-3 (~31%), but **gen-1 + search vs AB3 reads ~8/14 early** — search does the heavy lifting. This is expected and healthy: the network is a *prior for search*, not a standalone player, and the whole point of the system is that search + a good prior beats a strong classical searcher. Note the distinction from the gate result above: this AB3 read is a preliminary *production-strength* data point (external bot, n=14, search budget not fixed or logged), not the *network-improvement* claim (v3a vs gen-1, fixed low search, n=400, significant) — the two should not be conflated, and a proper paired, gated H2H vs AlphaBeta at a fixed search budget is still needed before any production-strength claim vs external bots can be made.

**Value repair reversed Gate-A.** As above: pre-repair, full search *lost* to its own raw policy (19%); post-value-repair-v2, search beats raw **67–71%**. The value fix bought placement *calibration* and *decisiveness* — not necessarily a higher raw margin — which is exactly what a miscalibrated value head at wide roots would predict.

**Base decision, resolved cleanly.** The `v3a`-vs-`v3b` base choice went through a genuinely instructive false alarm. An early masked-eval put `v3b` (a 47.8M action-attention arch) ahead on opening-placement calibration (+0.135). A *fresh, correctly-masked* re-run (each checkpoint evaluated under its own trained regime, n=456,658 true holdout) **did not reproduce it** — global corr `v3a` 0.733 > `v3b` 0.720, and the 54-wide bucket (the exact Gate-A mechanism) favored `v3a`. The clean master pentanomial re-run then settled it: **`v3a_cs0.03` wins** (H1, LLR 4.48, per-game 0.760) while `v3b_cs0.03` does not reach significance (CONTINUE, LLR 1.69). `v3a` is also 35M (cheaper/faster). The meta-lesson embedded here: **`c_scale=0.03 ≫ 0.1` is a bigger lever than the architecture change** (+7 points on `v3b`), and the earlier `+0.135` was a stale/mis-masked-eval artifact — trust the end-to-end-verified fresh measurement over the earlier one.

**Right now (2026-07-06).** **Gen-2 generation is live on 16× A100** (plus Modal L4 augmentation), with gen-1 as the base, the *identical* recipe to gen-1 (verified at the manifest level — `--lazy-interior-chance` and `--temperature-decisions 90` match gen-1's outcome-validated manifest, not drift), seeds 30M+ and globally disjoint, all GPUs at ~95–100% utilization.

---

## 10. The Flywheel: discrete vs continuous

Gen-1 ran as a **discrete** generation: generate → merge corpus → 1-epoch train → SPRT gate → promote → repeat. The forward-looking research question, studied by three agents on 2026-07-05 and revisited in the 2026-07-06 audit, is whether to stay discrete or go continuous.

**Verdict: a KataGo-style hybrid** — a continuous training loop over a **growing windowed replay buffer**, with a **cheap discrete gate** on only the checkpoint that feeds self-play. Not pure discrete (it wastes the A100 fleet idling on a synchronization barrier). Not pure ungated-AlphaZero (a bad checkpoint poisons a small buffer — and we have *already hit* the corpus-poisoning class via the seed collision). The counterintuitive core finding: **small-compute teams (KataGo, Leela Zero) kept a gate; giant-compute teams (AlphaZero on 5000 TPUs, MuZero) dropped it** — the opposite of naive intuition, because a huge dedicated fleet dilutes a bad network's data in a 44M-game corpus, while a small team cannot afford poisoning. Our 16 A100 + burst L4 sits squarely in KataGo's compute class, and KataGo is the only published system that solved *our* problem (max Elo per GPU-hour on ~28-GPU-class hardware; 50× efficiency over a naive AZ reproduction).

**The KataGo window math (what we built).** The replay window grows as `N_window = c · (1 + β·((N_total/c)^α − 1)/α)` with **α=0.75, β=0.4** — and our built window formula **already matches these constants**. The window stays large in absolute terms but *shrinks to ~9% of history* at steady state, keeping the training distribution near the *current* policy's visitation (consistent with the general finding that AZ value error tracks current visitation rather than cumulative volume [citation to be verified — original arXiv ID incorrect]). Target **sample reuse ≈ 4–8× train-steps per generated sample** (KataGo realized ~4×); Atari-style 20–32× reuse needs primacy-bias resets and is out of scope. Checkpoint refresh to self-play workers every ~250k–1M new samples, not per-gradient-step. And an **opponent pool**: play 15–25% of self-play against *older* checkpoints, because pure latest-vs-latest is provably non-convergent (can cycle) and Catan is asymmetric (first-player advantage + hidden dev cards → higher forgetting risk than symmetric Go; Tablut, an asymmetric 2p game, reports catastrophic attacker/defender forgetting). The continuous infra (`src/catan_zero/rl/flywheel/`: `replay_window.py`, `opponent_pool.py`, `checkpoint_registry.py`) exists; note that the *old* continuous plumbing (`vtrace.py`, `ppo_distributed.py`) was built for the retired PPO learner and **V-trace does not transfer** — Gumbel-AZ trains visit-count distillation + value regression with no importance ratio, so staleness is handled by windowing + a lagged target network, not V-trace.

**The sequencing decision:** finish the *discrete* gen-1 first (validate the loop cleanly — does distilling search beat `v3a`? Yes, 57%), *then* flip to the continuous hybrid. Skipping straight to continuous risks locking in a silent bug with no clean checkpoint to blame.

**The 2026-07-06 audit wave** (four parallel agents: one literature-research + three line-by-line code auditors). Fixes landed in this wave:

- **[CRITICAL, fixed] The flywheel gate path had no masking anywhere.** The path `promotion_gate_runner → evaluate_scoreboard → entity_token_policy` would have gated *masked* networks with *omniscient* features — the Section-5 leak, reintroduced at the gate. Fixed by swapping the gate to the masked, guarded `gumbel_search_cross_net_h2h.py`.
- **[CRITICAL, fixed] `train_bc` had no `--max-steps`** though `continuous_flywheel` passes it → argparse would exit 2 on the first real continuous round. Added (optimizer-step-counted, clean break, recorded in the report).
- **[MEDIUM, fixed] `report.json` provenance gaps.** `graph_layers`/`attention_heads`/`graph_dropout` were hardcoded null unless `arch == xdim_graph`, so **gen-1's real config (6 layers, 640 hidden, 8 heads, dropout 0.05) existed only in the checkpoint binary**. Fixed, plus `lr`/`hidden_size`/`mask_hidden_info`/`seed`/`symmetry_augment`/`data_format` now recorded.
- **[MEDIUM, fixed] H2H tools lost an entire worker's completed games on a single exception.** Changed to per-**pair** isolation (a bad pair is logged, the worker continues).
- **[MAJOR, fixed] Flywheel housekeeping:** `corpus/round_NNN` dirs never cleaned (disk exhaustion at ~100 full-window corpora); a *fictional* `opponent_pool_realized` metric (the pool wasn't actually wired into generation — now reported as not-wired); a promotion crash-atomicity journal gap (two-phase journal); zero-row-round detection; a `flock` on the loop dir so two loops can't stomp each other.
- **[LOW, fixed] `build_memmap_corpus.py`** now detects duplicate `game_seed`s (defense-in-depth after the seed collision).

**Verified-solid (the audit's clean bill):** masking equivalence train-vs-eval (9/9 tests), policy-target-is-pre-temperature correctness, winner-sign propagation, pentanomial pairing, seed disjointness, and the fast-search-rows-are-value-only PCR design. Also flagged: both H2H gate tools and `build_synthetic_manifest.py` are **untracked on the B200 master tree** — they die on a reimage until committed.

---

## 11. Ops Lessons: infrastructure discipline is research velocity

The through-line of this project is that most of the near-misses were *infrastructure*, not modeling, and each one could have silently produced a wrong scientific conclusion. Collected:

- **`pkill -f <pattern>` self-kill.** Running `pkill -f foo` inside an SSH heredoc whose *body contains* `foo` kills the shell itself → silent "no output" launch failures. Kill by PID/pgid or by `nvidia-smi` GPU-bound PIDs, never by a pattern that matches your own command line.
- **Orphaned spawn workers.** Killing a Python `multiprocessing` *parent* does not kill spawned *workers* — they orphan and keep occupying the GPU (once caused a 28-process traffic jam). Clear them via `nvidia-smi --query-compute-apps=pid`, not by killing the parent.
- **GPU oversubscription.** ~6–8 workers/GPU is optimal; 20/GPU saturates and balloons latency.
- **CLI-default drift (the default-override trap).** argparse defaults silently override dataclass defaults and are invisible in a diff of either file alone. The `c_scale` 1.0→0.1 dataclass fix would *not* have fixed the CLI-launched generation (which had `--c-scale default=1.0`), caught pre-launch. And Python loads code at import, so an already-running process keeps the *old* code — always compare a running process's start time against the code's commit/mtime, not against current HEAD. Any leak/correctness fix shipped as an opt-in flag belongs to this class: audit every launch command.
- **Seed collisions.** Covered in Section 6 — deterministic seeds mean a duplicated seed range produces bit-identical games and a double-counted, false-significant gate. The fix is a seed fleet planner plus dedup-before-pair in the aggregator.
- **Provenance gaps.** If a correctness-relevant flag (masking) isn't serialized into the artifact, you *cannot* later prove which regime a checkpoint was trained in — directory names are intent, not proof. This forced a whole controlled-calibration verification procedure that a single logged field would have obviated. (Now fixed.)
- **Verify version/perf claims on the *production* environment.** A "batch API is dormant / longest-road rule bug in the data" alarm turned out to be checking the *system* `python3` (wheel 0.1.0), not the production `.venv` (wheel 0.1.2, batch API active, correct rules). There was no longest-road bug in the data and no dormant perf win — both were an env-check error. Check `.venv`, not `PATH`.
- **Config serialization.** A positional-pickle config format created a whole class of silent "field SHIFT" bugs when fields were added; the fix (`#74`) was a name-keyed dict + schema version (legacy pickles still load). Relatedly, `catanatron_rs.json_snapshot()` node/edge ordering is **non-canonical across same-seed constructions** — never hash a raw snapshot for a cache/transposition key.
- **Worktree hygiene.** Never `git add -A` after worktree symlink hacks; trailing-slash `.gitignore` patterns miss symlinks, which then commit and clobber real directories on merge. Shared master trees on the GPU hosts have live jobs — never `checkout`, always use a `git worktree`.

None of these are glamorous. All of them are the difference between a result you can publish and a result that quietly wasn't real.

---

## 12. Open Problems and Roadmap

**Search science.**
- **Joint `c_visit × c_scale` re-ablation** (top priority): our `c_scale=0.03` is 33× below the Gumbel paper's validated pair and may be masking a value-calibration problem rather than being intrinsically right.
- **Lazy-interior-chance non-inferiority SPRT** is unresolved (58.3% vs full enumeration, "continue") — needs a decisive strength verdict, since it is the 13–19× generation-throughput lever.

**Throughput.**
- **Featurize in Rust** — the single biggest per-leaf win (~3.4× on GPU; featurization + FFI are 96% of per-leaf cost). Currently pure-Python.
- Board-topology caching (`_topology()` is recomputed per leaf despite being board-invariant, ~20% non-eval tax — safe to ship).
- A distilled small eval net (~5× on CPU/Modal, where the forward dominates), and eventually a full Rust MCTS tree port (a multi-week, gen-2+ effort).

**Data / exploration.**
- **Go-Exploit / high-regret restarts (`#64`):** restart self-play from archived mid-game / high-regret states (extract by an additive regret score, reconstruct by bit-exact replay via `game_seed ^ 0xA17E`, generate with an explicit start-mode mix). The published analog (arXiv:2302.12359, Go-Exploit) shows real measured gains; ranked high. Extraction currently uses omniscient stored tokens for state *selection* (fine — selection, not training), but restart *generation* must run masked.
- **Opponent pool** is built but **not yet wired into generation** — a prerequisite for the continuous flywheel to avoid strategy cycling.

**Training.**
- ≥2 epochs / effective reuse ≫1 (escape the 1-epoch overfit trap); AdamW + weight-decay A/B; LR decay after warmup; EMA / snapshot-EMA weights as a gate candidate; per-sample-normalized LR before scaling batch size up.
- Auxiliary-head loss weights set by *measured gradient-norm ratio* (10–40% of the main loss), never as a primary target.
- A MuZero-style **lagged target network** for the value bootstrap under the continuous window.

**The flywheel flip.** Finish validating gen-2 discretely, then flip to the KataGo-style continuous windowed-replay hybrid (window math already matches, gate now masked, `--max-steps` now exists, opponent pool to be wired). The genuinely publishable pieces along the way: the mctx completed-Q rescale-noise finding; a clean discrete-vs-continuous ablation on Catan (no un-confounded one exists in the literature); and, if it holds, the first credible AlphaZero-class result on two-player Catan (we found no directly comparable leak-free Gumbel/AlphaZero-style two-player no-trade Catan system with explicit chance nodes and paired statistical gating; notable prior Catan RL includes the cross-dimensional network of Gendre & Kaneko (arXiv:2008.07079), Catanatron's search bots, and small-compute AlphaZero-style attempts such as CatAnalysis, at 50–150 undiagnosed sims).

**External review round (2026-07-06).** Three independent reviews of this document concurred on priorities for the next iteration: Rust featurization (Section 6's largest per-leaf lever); D6 symmetry augmentation as a training default (Section 3); a joint `c_visit × c_scale` ablation run with fixed bounds and variance-aware arms (Section 4); sample reuse of 2–4 epochs plus EMA/snapshot-EMA weights (Section 7); wiring the opponent pool before flipping to continuous mode (Section 10); belief-aware auxiliary heads as the next research direction beyond the current unweighted KataGo-style heads (Section 3); and bootstrapped/hybrid value targets (Willemsen et al. 2021) as a challenger to the current pure Monte-Carlo outcome value target (Section 7).

---

### Selected references

- Silver et al., *Mastering the game of Go without human knowledge* (AlphaGo Zero), Nature 2017; *A general reinforcement learning algorithm…* (AlphaZero), Science 2018.
- Anthony, Tian, Barber, *Thinking Fast and Slow with Deep Learning and Tree Search* (expert iteration), NeurIPS 2017.
- Danihelka, Guez, Schrittwieser, Silver, *Policy improvement by planning with Gumbel* (Gumbel MuZero), ICLR 2022.
- Wu, *Accelerating Self-Play Learning in Go* (KataGo), 2019 (PCR, windowed replay, auxiliary heads, small-compute efficiency).
- Trudeau & Bowling / Go-Exploit, *Targeted Search Control in AlphaZero for Effective Policy Improvement*, arXiv:2302.12359.
- Van den Bergh, generalized (pentanomial) SPRT; fishtest/fastchess LLR implementation.
- Meta-analysis of self-play value-error vs state visitation [citation to be verified — original arXiv ID incorrect; arXiv:2311.01609 is "Responsible Emergent Multi-Agent Behavior Via Theory of Mind" and does not match this claim].
- UCT-V-P uncertainty-aware completed-Q, arXiv:2512.21648 (adapted for the D2 variance-aware arm).
- ReZero backward-view reanalysis, arXiv:2404.16364.
- Gendre & Kaneko, cross-dimensional neural network for Settlers of Catan, arXiv:2008.07079.
- Willemsen et al., bootstrapped/hybrid value targets for self-play RL, 2021 [full citation to be verified] (flagged in the 2026-07-06 external review as a challenger to pure Monte-Carlo outcome value targets).
