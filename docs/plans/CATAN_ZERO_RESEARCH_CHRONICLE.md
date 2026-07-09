# Catan-Zero: Complete Research Report

**A Gumbel-AlphaZero Expert-Iteration System for Two-Player No-Trade Settlers of Catan**

*Status as of 2026-07-08 (UTC). This document is self-contained: it is written for an external expert reviewer who has no access to our code, machines, or prior documents. Everything needed to critique the system — architecture, search math, training recipes, all experimental results (positive and negative), infrastructure, and open questions — is in this file. All numbers come from gate artifacts, training reports, and profiler runs.*

---

## 0. Abstract

**The goal of this project is to build the #1 Catan bot — the strongest two-player no-trade Settlers of Catan agent in existence** (standard 10-victory-point rules), decisively beating every published bot (the immediate milestone: beat catanatron's hand-tuned ValueFunction bot, the strongest known classical Catan AI, which no learning-based agent has ever beaten) and ultimately reaching superhuman play. The method is AlphaZero-style expert iteration: a neural network plays against itself under Gumbel MCTS with explicit chance nodes, the search-improved play is distilled back into the network, a paired statistical gate certifies improvement, and the cycle repeats.

Starting from a behavior-cloning base that lost to classical heuristic bots by ~140 Elo, we have completed **three verified expert-iteration promotions in three days** (v3a → gen-1 → gen-2A → gen-3), each certified by paired color-swapped pentanomial SPRT gates:

- **Turn 1:** gen-1 beats v3a **57.0%** (400 games, significant, ~+49 Elo) — also confirmed at production search depth (58%/300 games, H1).
- **Turn 2:** gen-2A beats gen-1 **57.0%** (400 games, LLR 3.57, H1). A recipe-variant arm reached **59.0% (LLR 6.12)**.
- **Turn 3:** gen-3 beats gen-2A **54.71%** (700 games, H1).
- **Turn 4 (in progress):** two independent gate measurements put the current candidate at ~**52-54%** vs gen-3 (pooled 52.8% over 1000 games, one-sided p≈0.04; two-sided ≈0.08) — real but below our +30-Elo promotion bar, and **plateaued** (details §12). *[Hygiene note: the pooled p was computed one-sided; two-sided it is only **suggestive**, not conventionally significant — CAT-70.]*

Externally, against catanatron's hand-tuned bots (the strongest available classical Catan AI): win rate vs the AlphaBeta-depth-3 bot went 49.5% → 58.6% → 66.8% across turns; vs the tuned ValueFunction bot (our "north star", the one bot still ahead of us) 35.5% → 37.0% → **45.7%**. The remaining external gap is roughly −30 Elo.

Getting here required: fixing ~30 verified bugs across two game engines, the search, the trainer, and the measurement layer; diagnosing a noise-amplification pathology in the stock mctx Gumbel implementation that made search *lose to its own raw policy* (we believe this is unreported in the literature); discovering and sealing a hidden-information leak that invalidated all early external comparisons; a throughput program (CUDA MPS scheduling + Rust featurization) worth ~3x; and converting the discrete generation loop into a continuous KataGo-style training flywheel, which surfaced a **value-head self-distillation drift** pathology unique to the continuous setting (+69% value error over six ungated training rounds).

The recurring scientific theme: **the value head is the fragile, load-bearing organ**. Four independent failures (search losing to raw policy; a multi-epoch recipe losing to its own initialization; continuous-lineage drift; a 91M scaling probe blowing up on epoch 2) all reduce to the same law — the value head cannot tolerate revisiting the same distribution, while the policy head shrugs everything off.

We want an expert critique of: the architecture, the search configuration, the training recipe, the promotion-criterion design, the plateau we are currently in, and anything the recent literature (2023-2026) says we should be doing differently. Specific questions are enumerated in §16.

---

## 1. Hardware and Compute Inventory

Everything below runs on this fleet (relevant to any cost/benefit judgment):

| Resource | Spec | Role |
|---|---|---|
| 1× B200 host | 2× NVIDIA B200 GPUs, Xeon 8592+ (~50 usable cores), ~680 GB RAM | Training, gating/H2H evaluation, continuous-flywheel orchestration |
| 2× A100 hosts | 8× A100-80GB each (16 A100s total), 240 CPU cores each | Self-play generation fleet (~1.1M training rows/hour aggregate) |
| Modal cloud | Burst L4 GPUs, **hard cap < 45 GPUs** | Auxiliary generation waves (~33.6 games/hr/L4) |

Totals: 18 owned GPUs + ≤45 burst L4s. This is **KataGo-class compute** (tens of GPUs), not AlphaZero-class (thousands of TPUs) — a distinction that turned out to matter for the gating-strategy literature (§14.1). Monitoring is Prometheus + Grafana fed by an SSH-polling collector (GPU util, generation rates, flywheel round progress, gate progress, Modal part completion; 16-17 alert rules).

Key measured throughput facts that govern all economics:

| Quantity | Value |
|---|---|
| One full (n=64) search, chance nodes enumerated | ~5,400 leaf evaluations |
| One full search, lazy interior chance | ~47-63 leaf evaluations (13-19x) |
| Per-leaf cost on GPU | ~3.4 ms, of which **NN forward = 0.18 ms (4%)** |
| Per-leaf cost breakdown | featurization (pure Python) ~42%, Rust-FFI/JSON ~26%, Python tree ops ~28%, NN 4% |
| Per-leaf on CPU (int8 ONNX) | 34-38 ms (forward-dominated; batching gives ~0 on CPU) |
| Self-play game | ~205 decisions, 54.8% forced actions, ~3,200 sims, ~7,800 leaf evals |
| Generation rate per A100 (16 workers + CUDA MPS) | ~90k rows/hr (≈440 games/hr-equivalent) |
| Training | ~540 s per epoch per 2.7M rows (B200, batch 4096, bf16) |
| Gate games | ~3-5 min each at n=64; ~34 min per 400 games at n=8 |

The punchline of the cost model: **the bottleneck is Python and process scheduling, not FLOPs.** The NN forward is 4% of a leaf. This inverts most published intuitions (GPU-tree-search papers assume the forward dominates) and killed several "obvious" optimizations (distilled small eval net, aggressive GPU batching as a first move).

---

## 2. The Problem: Why Two-Player No-Trade Catan Is Hard

The two-player no-trade variant isolates what matters for a search-and-learn agent (stochasticity, hidden information, long horizons, structured actions) while removing multi-party negotiation, which is orthogonal.

1. **Explicit chance nodes.** Every turn starts with a 2d6 roll (11 outcomes with known probabilities) distributing resources; a 7 moves the robber and steals a random card; development cards come off a shuffled deck. The game tree contains real chance nodes with nontrivial branching. Enumerated recursively, they compound: one 64-simulation search costs ~5,400 leaf evals (§5.2).
2. **Hidden information.** Opponent hand composition and unplayed development cards (including hidden victory-point cards) are private. A network that can see them is solving a strictly easier game — and ours silently did, for days (§6).
3. **Long horizons, sparse signal.** Games run ~150-200 decisions (600-decision cap in self-play); the only ground-truth signal is win/loss at the end.
4. **Variable, wide, near-tied action spaces.** Legal actions per decision range from 1 (forced: must-roll, must-discard) to **54** (opening settlement placement). The opening placements are near-tied in value (the true value spread across the 54 candidates is ~0.06 nats of prior mass) — so wide roots are simultaneously where search matters most and where value noise is most destructive. ~55% of all decisions are forced.
5. **Strong classical baselines.** The open-source catanatron project ships a hand-tuned ValueFunctionPlayer and AlphaBeta search bots. Its maintainer tried and abandoned AlphaZero-style approaches; no published Catan RL agent beats the hand-tuned bots. The ValueFunction bot is our external north star and still leads us by ~30 Elo.
6. **Two engines.** A Python reference engine (catanatron, vendored) and our Rust port (catanatron_rs) used for speed (~5.9x simulation throughput, batched chance-spectrum API). Cross-engine trust had to be earned: a 1000-game fixed-pair equivalence sweep drove divergences from ~60% to ~2%, and closing the rest required fixing **eight subtle rules bugs in longest-road computation** (graph problems: longest acyclic path with incumbent-aware tie-breaking under settlement severs, enemy-cut-node traversal, loop closure) — each verified by TDD against specific replay seeds, on both engines. Deploying the corrected engine fleet-wide was treated as a **rules change** (it changes the data distribution), not a perf swap.

---

## 3. Pre-History: The BC + PPO Era and Why It Failed

Before 2026-07-02 the project was behavior cloning on AlphaBeta-teacher data plus PPO fine-tuning. Over 30 experiment branches, everything plateaued. A systematic audit found 11 verified bugs that explain the plateau, and one structural truth:

- The AB teachers' chance enumeration **rolled fresh dice instead of enumerating outcomes** — teacher labels were far noisier than believed.
- Robber-placement teacher scoring covered 1/18 of outcomes.
- The value head was **unreachable by the phase-weighting scheme** (value training was effectively off).
- PPO was configured to a no-op (V-trace temperature bug, forced-row inclusion, eval-mode error).
- Smoke evaluations ran with victory threshold 4 instead of 10, laundering noise into "wins."
- The single prior MCTS attempt fabricated Q=0.0 labels at 16 sims and was abandoned for the wrong reason.

**Structural truth: BC on teachers has no mechanism to exceed the teachers**, and the teachers were weaker and noisier than assumed. The pivot: keep BC only as the *initial prior* (Catan's action space is too structured for cold-start random self-play to produce informative games), and let search + distillation provide the improvement operator. That is expert iteration (Anthony et al. 2017; the AlphaGo-Zero/AlphaZero family).

**Baselines at pivot:** our best BC net ("hard-target") measured **−136 / −145 Elo vs catanatron_value / AB3** — and even that was flattering, because it was measured with the hidden-info leak active (§6). The goal required a ~+200-Elo swing.

---

## 4. Neural Architecture

### 4.1 The entity-graph transformer (35M parameters)

Catan's board is a hexagonal graph: 19 hex tiles, 54 settlement nodes, 72 edges, plus per-player state and game context. We encode the state as **entity tokens** — one token per tile, node, edge, player, and a game-context/CLS token — and run a transformer over them:

- 6 attention layers, hidden size 640, 8 heads, dropout 0.05, ~35M parameters.
- **Policy head:** actions are scored by scaled cosine similarity between per-action context embeddings and the CLS representation, plus an additive bias, over the legal-action catalog (a fixed catalog of all possible parameterized actions; the legal subset is masked in).
- **Value head:** scalar in [−1, 1], read from the CLS token only. Trained with MSE; evaluated through a tanh squash at search time.
- Per-player token feature width 31; observation width 806 in the masked regime.
- Auxiliary heads (KataGo-style ownership/score analogs, Catan-native) are built but carried at zero weight — never allowed to be a primary value target.

### 4.2 Known architectural ceilings (verified by audit; deliberate non-fixes so far)

1. **The adjacency tables are computed by the featurizer and never consumed by the model.** Attention is dense and unmasked; there is no graph bias, no graph convolution. The hex-graph structure is available and thrown away — the model must rediscover adjacency statistically.
2. **Actions do not cross-attend to board tokens.** Action scoring is cosine-to-CLS; an action like "build settlement at node 23" never directly attends to node 23's token.
3. **The value head reads only CLS** (no attention pooling).

A warm-start-safe upgrade addressing all three (action-target gather, action cross-attention layers, value attention pooling, all zero-initialized so it reproduces the current net bit-for-bit at init) was **built and tested** — and then *lost the controlled comparison*: the 47.8M upgraded architecture ("v3b") failed to beat the plain 35M ("v3a") in the base-decision H2H (v3a: pentanomial H1, LLR 4.48, pair-score 0.762; v3b: continue, LLR 1.69), and v3a also won on value-calibration metrics (global corr(q,z) 0.733 vs 0.720, and won the critical 54-wide placement bucket). We kept the 35M. The upgrade remains available as a flag for a future generation.

### 4.3 The symmetry finding

The value net **badly violates the board's D6 (dihedral hexagonal) symmetry**: across 12 rotations/reflections of the *same* opening position, the prior's orientation-noise std is 0.175 nats — **larger than the ~0.06-nat spread separating the 54 placement candidates it must rank** — and value std is 0.049 (range up to 0.29 on [−1,1]). A large fraction of what looked like irreducible value noise at wide roots is violation of a known invariance. Test-time 12-fold symmetry averaging gives ~3.3x denoise (near √12 — errors decorrelate) with no retraining, implemented as a search-evaluator flag. Train-time symmetry augmentation was tested as a recipe arm (§9.2): good (58.5% gate) but not better than the value-target fix alone (59.0%), so not adopted. Architectural D6 equivariance has not been attempted (no precedent for hex-graph transformers; KataGo itself only does augmentation + test-time averaging).

### 4.4 The 91M scaling probe

An 87.85M-parameter version (wider/deeper same family) was trained on the then-current ~4M-row corpus. Epoch 1: policy val loss 1.6252 / value 0.2665 (worse than the 35M — undertrained for its size). Epoch 2: policy improved to 1.5966 but **value blew up to 0.3929** — the value head overfit on data reuse (§10's recurring law). No gate was run. Parked conclusion: bigger nets need proportionally more *fresh* data, not more epochs; revisit at ≥10M-row corpus scale. (Scaling motivation remains: strength ∝ params^0.88 in published board-game scaling work, and our NN-is-4%-of-leaf economics make a 2-3x net nearly free at generation time — an advantage no published AZ project had.)

---

## 5. Search: Gumbel MCTS over Enumerated Chance Nodes

### 5.1 Why Gumbel

Classic AlphaZero PUCT + Dirichlet noise is tuned for hundreds-to-thousands of simulations per move. Our budget is tens of simulations (leaves are expensive). The Gumbel MuZero operator (Danihelka et al., ICLR 2022) is designed for this regime: draw Gumbel noise once per root action, take the top-k, allocate budget by **sequential halving**, and form the improved policy from **completed Q-values** — with a policy-improvement guarantee even at one simulation per candidate. Gumbel top-k replaces Dirichlet entirely. Our implementation mirrors mctx (`gumbel_muzero_policy`) semantics: the improved policy is

```
π'(a) ∝ softmax( logits(a) + σ(completedQ(a)) )
σ(q)  = (c_visit + max_visits) · c_scale · rescale_to_unit_interval(q)
```

where visited actions use their backed-up Q and unvisited actions are completed with a prior-weighted mixed value v_mix.

### 5.2 Chance nodes

Interior transitions through dice rolls are explicit chance nodes. The correct-but-naive version enumerates ~11 outcomes recursively at every depth: one n=64 search ≈ **5,400 leaf evals**; a 600-decision game ≈ 500k evals. Chance-node backups are probability-weighted expectations over materialized outcomes (Rao-Blackwellized, lower variance than sampling) — this was audited line-by-line and verified correct.

**Lazy interior chance** (production since gen-1): the *root* chance layer stays enumerated; *interior* ROLL nodes are single-sampled. 5,400 → ~47-63 evals (13-19x). Single-sample interior backup is statistically unbiased (verified ±1.81 SE, checkpoint-independent). History worth knowing: lazy was originally **rejected** ("lost 2-21 to raw policy") — that verdict later proved to be a **stale-checkpoint artifact**, measured on a broken-value-head net on which *enumerated* search also lost. On the repaired net, lazy is near-parity in strength and strictly better on game completion (0% truncation vs 22-27% for enumerated at the 600-decision cap). Lesson: when a component fails, check whether the surrounding system was broken at measurement time.

### 5.3 Gate-A: search loses to its own raw policy (the pivotal failure)

The pre-generation sanity gate asked: does full n=64 search beat the raw policy it searches with? **It lost: 19.2% and 22.1% on two host replicates (SPRT H0).** A controlled trace over 40 real placement roots found the mechanism:

At a 54-wide placement root with 64 sims, each candidate gets ~1.2 simulations. Completed-Q is then dominated by 1-2-sample value noise (spread ~0.04). The stock mctx **min-max rescale (`rescale_to_unit_interval`) stretches that noise to fill [0,1]**, manufacturing false confidence that swamps the near-tied prior. 74.6% of losses were opening-placement blowouts. This is a property of stock mctx (c_visit=50/c_scale=0.1 are its defaults; the rescale has only an epsilon floor, no noise guard). We searched the mctx issue tracker and literature and found no prior report; the Gumbel paper itself has a footnote flagging c_scale sensitivity and floats variance normalization as future work (partial prior art). We consider the wide-root/low-budget false-confidence mechanism + the fix publishable as a short note.

**The two-sided fix:**
- **Search side:** `c_scale = 0.03` (noise amplification scales with c_scale) removes the harm. Two principled flag-gated arms were also built: **D1** noise-floor rescale attenuation (blend rescaled Q toward 0.5 by signal-spread/noise ratio, in the spirit of KataGo's uncertainty-weighted playouts) and **D2** variance-aware completed-Q (James-Stein shrinkage of each visited Q toward v_mix by its standard error, adapted from UCT-V-P-style variance-aware search).
- **Value side (the real fix):** value-head repair — retraining the value head on **true self-play outcomes** (86k raw-policy self-play games) rather than the BC corpus. Post-repair, **search beats raw 67-71%** (from 19%). A prior "value-repair v1" that retrained value-only on the *BC corpus* had failed with flat loss at 10x LR — the value head was already at that objective's optimum; repair had to come from self-play outcomes. This observation is what makes the whole flywheel necessary.

The standing methodological rule extracted from Gate-A: **only strength H2H binds a search-semantics decision.** Target-fidelity metrics (KL to full-search targets, argmax stability vs the seed-noise floor) were *not predictive* — lazy-128 looked "within noise" on targets while losing 2-21. The predictive early warnings were cross-agreement collapse (37-53% vs a 57% same-semantics floor) and root-value noise inflation (6-9x).

### 5.4 The search-config ablation (production config survives everything)

A 15-arm joint ablation (~170 games/arm vs production cv=50/cs=0.03, paired):

| Arm | Result | Verdict |
|---|---|---|
| Gumbel-paper default (c_visit=50, **c_scale=1.0**) | **32.7%** | **H0 — decisively refuted** |
| Fixed-bounds qtransform ([−1,1], no min-max rescale) | 45.5% | H0 — refuted |
| D2 variance-aware | 49.3% | neutral |
| c_visit=25, cs=0.03 | 55.0% | inconclusive (mild lead) |
| D1 noise-floor | 54.6% | inconclusive (mild lead) |
| ~10 others (cs≥0.3 grid, squash variants...) | ≤50% | dead |

So cs=0.03 is **3x below mctx's shipped default (0.1, the paper's Atari constant)** and **33x below the paper's own board-game value (1.0)** — two distinct reference points, not one "validated pair." This is not a compensating hack; a small sigma-scale is genuinely right for 54-wide stochastic roots at 1-2 visits per action. The theory-clean alternative (fixed bounds) is worse. This closed the c_visit/c_scale question permanently. *[Hygiene note, CAT-70: the finding here is the **mechanism** (min-max rescale manufactures false confidence from value noise at low-visit wide roots), not the specific constant — mctx issue #108 reports a corroborating symptom (rescale sensitivity at low visit counts) and #66 documents the chance-node gap this project fills; see also §14.]*

### 5.5 Production search configuration (pinned)

```
n_full=64, n_fast=16, p_full=0.25      (playout-cap randomization, §7.1)
c_visit=50, c_scale=0.03
value_squash=tanh
max_decisions=600, max_depth=80
lazy_interior_chance=on
public_observation=on                    (masking, §6)
correct_rust_chance_spectra=on
temperature: T=1.0 for first 90 decisions (sampling only), then argmax
```

Gates run n=8 or n=16 (fast sensitive reads; see §8.3 for why low-sim gates are the right sensitivity regime).

### 5.6 Known search-implementation weaknesses (verified, unfixed)

An honest list for the reviewer — these are real and we know it:

1. **Leaf evaluations in the main selection loop are un-batched** — one at a time, synchronously. The batched evaluator fires only on chance fan-outs (≤11 children). The self-play driver is single-threaded, so the async micro-batcher's ~3ms wait is pure overhead today. (KataGo's dedicated eval-server-thread batching is the known fix; prerequisite is parallelizing search.)
2. **No cross-move subtree reuse** — every decision builds a cold tree.
3. **Board topology was recomputed per leaf** (~500k redundant rebuilds/game) — fixed via caching; and the board state was reconstructed twice per leaf (entity features + action-context features independently) — fixed in the Rust port.
4. The sequential-halving schedule **overspends at wide roots** (a nominal n=16 fast search actually spends ~32 sims; 105 at a 54-wide root) due to a ≥1-sim-per-candidate floor. An exact-budget port was built and **gated: it lost** (45.9%, LLR −3.9 at n_fast=16 — the "overspend" was accidentally protective; exact-16 = one visit per candidate = zero halving rounds = noise argmax) and washed at n=64 (49.25%). Mechanism kept dormant. We consider this closed but it is a genuinely interesting negative result: **mctx-conformant budget accounting is *worse* than the accidental overspend at tiny budgets.** *[Confound check (CAT-70, `docs/audits/CAT70_EXACT_SH_AUDIT.md`): verified by reading `_run_root_search`/`_simulate` on both `master` and `origin/f61-exact-budget-sh` (`src/catan_zero/search/gumbel_chance_mcts.py`) — the port **stockpiles** per-candidate Q/visit statistics across SH rounds on the same `_GNode`/`_GAction` objects (mctx-style), it does not discard/restart them (Karnin-style). No confound: this result is clean evidence against exact-budget accounting alone, not a mix of budget-accounting and statistics-discarding effects.]*

---

## 6. The Hidden-Information Leak

The most important correctness story in the project.

**The leak (found by empirical audit, day 3).** The featurized observation contained the opponent's full hand composition, unplayed dev-card identities (including hidden VP cards), and true victory points. Separately, the planner's chance spectra used the victim's *true* hand for robber steals and the *true* deck for dev draws. The live environment was correct; the observation/planner side was omniscient.

**Why it invalidated results.** catanatron's bots are deliberately *belief-based* (they re-add face-down cards to the deck and model steals as uniform). Every external comparison was omniscient-vs-belief — not a strength claim. The −145 Elo baseline was measured *with* omniscience, so the true starting gap was worse. (Internal search-vs-raw comparisons stayed valid — the leak was symmetric.)

**The fix (three layers, all verified):**
1. A canonical token-level transform zeroing non-actor hidden slots, applied at training (`--mask-hidden-info`, masks the banked corpus at load — no regeneration) and at inference (`public_observation` evaluator flag with perspective-relative masking).
2. Model-invariance proof: with masking on, permuting the opponent's hidden hand changes value by <1e-5 and logits by <1e-4.
3. A planner-only belief-spectra option (uniform steal; belief deck = base − own − played).

**The regime-provenance problem.** The trainer originally did not serialize the masking flag anywhere durable — directory names were intent, not proof, and one checkpoint named "value_repair" turned out omniscient. Post-fix, checkpoints carry the flag and evaluators **fail closed** on mismatch. For pre-fix checkpoints we built a controlled both-regime calibration test: a net calibrates better in its own training regime, so compute corr(q,z) on the same held-out states as-stored vs masked — *with a known-omniscient control to break the "masking universally helps" confound*. Result on a 456k-row holdout: v3a +0.050 masked, v3b +0.062 masked, omniscient control −0.031 (opposite sign ✓). The bases are genuinely leak-free.

**Meta-lesson (a recurring bug class):** any correctness fix shipped as an *opt-in flag* is a landmine — argparse defaults silently override dataclass defaults, and the staged generation commands written before the fix would have generated the entire next corpus leaked. Audit launch commands, not just code. Masked-AZ (observation-only, no belief-state machinery) turned out empirically sufficient — the masked nets beat the belief-based bots — consistent with recent findings that plain masked AlphaZero is surprisingly strong on imperfect-info games (Stratego/DarkHex-class results). Belief-aware machinery (ReBeL/Student-of-Games class) has not been needed yet.

---

## 7. Self-Play Data Generation

### 7.1 Design

- **Playout-cap randomization (KataGo).** Each decision is a full search (n=64) with probability 0.25 or a fast search (n=16). **Only full-search rows carry a policy target** (policy weight multiplier 1.0 vs 0.0); **all rows carry value targets**. Fast searches are cheap value-signal generators. Effective policy-sample fraction ≈ 7.7% of rows. Forced actions (~55% of decisions) get weight 0.1 and no policy target.
- **Temperature** (T=1.0 first ~90 decisions, then argmax) applies **only to which action is played, never to the recorded improved-policy target** — the target is always the un-tempered search distribution. (Trap discovered: the temperature cutoff is computed as fraction × max_decisions cap, so raising the cap silently widens the exploration window — must rescale.)
- **Determinism & seeds.** `game_seed = base_seed + index`; board from seed; chance RNG from `seed XOR 0xA17E` (board layout and dice history independent, both reproducible). Deterministic search + deterministic chance means **two workers on the same seed produce bit-identical games** — see §13.2 for the false-H1 this caused. Seeds are governed by a cross-host ledger; generators refuse unledgered ranges.
- **Truncation labeling.** Games hitting the 600-decision cap keep their rows with a VP-margin proxy value label (weight 0.25) rather than being discarded (discarding biases toward short decisive games). The 600 cap itself was chosen after a 300 cap produced 44-47% truncation and zero decisive gate pairs.
- Shards carry: improved-policy soft targets, outcome targets, root value (`target_scores`), prior policy (for KL telemetry), per-row weights, and full config provenance in the manifest.

### 7.2 Corpus sizes per generation

| Generation | Games | Rows | Notes |
|---|---|---|---|
| gen-1 | ~10-12k | 2,736,128 | v3a base, 17 GPUs, 0% truncation |
| gen-2 | ~16k | 3,648,516 | gen-1 base, A100 fleet + Modal L4 wave |
| gen-3 | ~16k | 3,930,000 | gen-2A base |
| gen-4 / flywheel window | continuous | 4,575,429 (concat) | gen-3 base, fleet-fed (§12) |

A controlled half-corpus experiment (§9.2, "ARM-C") established **≥3M rows as the practical floor** — half the corpus lost both ~3 points of win rate and statistical significance.

---

## 8. Training (Distillation)

### 8.1 The production recipe (locked, 3-for-3 on promotions)

```
1 epoch, batch 4096, bf16, Adam, lr 3e-5, warmup 100 steps
init from current champion (never from scratch)
soft policy targets (weight 0.9) renormalized over legal support
value loss weight 1.0 (AZ 1:1 convention; MC outcome targets)
value-target-lambda 0.5      ← z blended 50/50 with the search root value
final-VP auxiliary loss weight 0.1
truncated-game VP-margin value weight 0.25
hidden-info masking ON
```

Training is deliberately cheap (~9 minutes per generation) — **sample generation, not training FLOPs, is the scarce resource**, so the recipe's job is to extract signal in one pass without damaging the value head.

`value-target-lambda 0.5` — blending the terminal outcome z with the search's root value estimate (a soft-Z/hybrid target in the spirit of Willemsen et al.'s value-target study) — produced the strongest single gate in project history (59.0%, LLR 6.12, §9.2). *[Winner's-curse caveat, CAT-70: 59.0% vs the 57.0% production-verbatim control are two independent 400-game estimates with ~±5% CIs — the difference is not statistically significant (z≈0.57), and this arm was the best of a 7-arm matrix (§9.2), whose winner is biased upward by construction. So this is not "the single most validated training-science result we have"; the λ direction is independently plausible (Willemsen et al., ALA-2020-workshop/NCA-2022; Abrams z/q-averaging) and adopting it was reasonable, but the 59.0% label overstates the evidence — treat it as an unresolved question pending a direct λ-arm-vs-control gate, not a settled result.]*

### 8.2 The memmap streaming loader

npz shards in RAM cost ~43.8 KB/row (a 32.6M-row corpus would need 1.4 TB); the flat memmap corpus is 13.7 KB/row on disk and trains at 10.5 GB RSS (2.6x lower), ~24% slower synchronously, partly recovered by a thread prefetcher. Concatenated multi-corpus training (for the flywheel window) opens all part files — which produced a file-descriptor exhaustion crash at 43 parts (fixed with ulimit + planned consolidation).

### 8.3 Evaluation & statistics methodology

- **Paired color-swapped games on identical seeds** — both nets see both sides of the same board and dice history (Catan has significant first-player advantage and board luck).
- **Pentanomial SPRT** (fishtest/GSPRT math): the unit of analysis is the *pair outcome* (WW/WL-split/LL), which correctly models within-pair correlation. Promotion gates use elo0=0, elo1=30, α=β=0.05. *[Rationale correction, CAT-70: this was previously stated backwards. Fishtest's own measurement (issue #348) found within-pair correlation ≈ −0.15 — correct pairing **adds** ~15% statistical power relative to naive per-game binomial, which is conservative (understates significance), not anticonservative. We measured our own empirical within-pair correlation with `tools/measure_pair_correlation.py` (synthetic-data unit-tested, `tests/test_measure_pair_correlation.py`) against a pooled, opportunistic sample of 383 pairs of local gate records: **+0.015** — near zero and opposite in sign from fishtest's −0.15, but this sample mixes several distinct H2H arms rather than one clean experiment and should be treated as a rough first read, not a final number. A clean per-arm re-measurement on canonical gate output (host command: `python tools/measure_pair_correlation.py 'runs/h2h_v3conf/<one_arm>_*.json'`) is still open. Regardless of our own arm-mixed sample's sign, pentanomial pairing is kept — it is the theoretically correct unit of analysis for color-swapped paired games either way.]*
- **Low-vs-high-simulation sensitivity:** distillation gains show at LOW search budget (prior-dominated) and shrink at high budget (deep search corrects both sides toward the same moves). So n=8 gates are the sensitive "did the net improve" instrument; n=64 confirms production-budget relevance (gen-1 passed both: 57% @ n8, 58% @ n64). A search-vs-own-raw ratio is *not* a cross-net strength claim (a higher ratio can mean a weaker raw policy).
- **External panel:** fixed opponents (catanatron_value = north star, AB3/AB4), cross-engine lockstep, tournament map, 200 games per matchup standard (95% CI ≈ ±7%; we are currently running a 1000-game high-power panel to resolve a ±5% question — §12.4). Adopted specifically as the **anti-blind-spot rule**: internal gates can stay "healthy" while true strength collapses (self-play inbreeding), so every promoted champion must also play the external panel and the *external trend* is the true success metric.
- **Anchor holdout (continuous era):** a pinned 196,535-row corpus from a specific fleet wave, excluded from all gradients forever, with per-candidate telemetry measured by an lr≈0 probe (one optimizer step at lr 1e-12 = pure evaluation through the training code path). Champion baseline: policy CE 1.4132 / value MSE 0.2492 / top-1 accuracy 0.5827. This exists because the flywheel's rolling-window re-splitting leaked ~95% of validation rows from prior training sets, making val metrics measure memorization (§11.3).

---

## 9. Results: The Generation Ladder

### 9.1 The champion ladder (all paired pentanomial gates)

| Turn | Matchup | Games | Result | Verdict |
|---|---|---|---|---|
| base | v3a vs v3b (arch A/B at cs0.03, masked, clean seeds) | 128 | pair-score 0.762 | v3a H1 (LLR 4.48); v3b continue (1.69) |
| 1 | gen-1 vs v3a | 400 @ n8 | **57.0%** (228-172), CI [52.1, 61.9] | significant (~+49 Elo) |
| 1' | gen-1 vs v3a | 300 @ n64 | **58%** | H1 (gain survives production search) |
| 2 | gen-2A vs gen-1 | 400 @ n8 | **57.0%**, LLR 3.57 | H1 |
| 3 | gen-3 vs gen-2A | 700 @ n8 | **54.71%** | H1 (needed extension past 400) |
| 4a | gen-4 (discrete-style) vs gen-3 | 400 @ n8 | 52.25%, pentanomial LLR +0.06 | continue → hold |
| 4b | flywheel round-17 candidate vs gen-3 | 600 @ n16 | 53.17%, LLR +1.26 | continue at cap → hold |

Turns 1-2 compounded at an identical +~49 Elo/turn; turn 3 compressed to ~+33; turn 4 is at ~+20 and plateaued (§12). Note the compression trend — one of our open questions (§16, Q3).

### 9.2 The gen-2 recipe matrix (training science by tournament)

Seven candidates trained on the same gen-2 corpus, each gated vs gen-1 (400 paired games, n=8):

| Arm | Recipe | Result | Verdict |
|---|---|---|---|
| **H1** | production + value-target-λ 0.5 | **59.0%, LLR 6.12** | H1 — best of this 7-arm matrix, *not significant vs the 57.0% control (§8.1 winner's-curse caveat)* |
| H2 | + λ0.5 + train-time symmetry augment | 58.5%, LLR 4.88 | H1 |
| arm3 | 1 epoch, AdamW | 57.75% | H1 |
| **gen2A** | production verbatim | **57.0%, LLR 3.57** | **H1 (promoted — clean-science control)** |
| arm2 | two-phase policy-only reuse | 53.25% | continue |
| arm1 | 3 epochs, full weight | 51.25% | continue |
| recipe-B | 6 changes at once (3 epochs, AdamW+wd, cosine, lr 1e-4, ...) | **47.0%** | **H0 — lost to its own initialization** |
| ARM-C | production on a random **half** corpus | 53.75% | continue (corpus-size floor evidence) |

**Recipe-B's autopsy is the most instructive negative result:** per-epoch validation showed the **value head overfitting** — val value loss 0.665 → 0.809 → 0.842 across epochs while train loss improved and val policy loss stayed flat (~1.48). ±1 outcome targets from ~16k games are low-information relative to 3.6M correlated rows; extra epochs memorize outcomes. Even epoch 1 at lr 1e-4 was already worse than production's final — LR magnitude implicated independently of reuse. The policy head (rich soft targets) tolerated everything.

### 9.3 External ladder (200 games/matchup, tournament map, masked regime)

| Net | vs catanatron_value | vs AB3 | vs AB4 |
|---|---|---|---|
| v3a (base) | 35.5% (H0) | 49.5% | 52.5% |
| gen-1 | 37.0% (H0) | 58.6% | 64.0% (H1) |
| gen-3 | **45.7%** [39.0-52.7] | **66.8%** | — |
| flywheel candidate (turn 4) | 41.0% [34.4-47.9] | 66.8% | — |

Notes: gen-1's *raw* policy scores only ~31% vs AB3 — search contributes a large absolute chunk; the net is a prior for search, not a standalone player. Pre-leak-fix external numbers existed and are excluded as invalid. The turn-4 candidate's 41.0% vs 45.7% is within overlapping CIs at n=200 — a 1000-game panel is running to resolve it. *[Dropped claim, CAT-70: an earlier draft asserted "the AB-depth ladder is inverted in catanatron (AB5 is weakest; known quirk)." This project's own §9.3 external ladder has never included an AB5 column, and no gate/H2H/log artifact anywhere in the repo contains an AB5 outcome — AB5 appears only as an opponent config used for teacher-data generation, never as a measured result. The claim could not be verified against catanatron's own sources or re-derived from our data and has been removed.]*

### 9.4 The kill list (decisively closed negatives — do not refund)

| Hypothesis | Evidence | Status |
|---|---|---|
| Gumbel-paper c_scale=1.0 | 32.7% vs cs0.03 | H0, dead |
| Fixed-bounds qtransform | 45.5% | H0, dead |
| Exact-budget SH at n_fast=16 | 45.9%, LLR −3.9 | H0, dead (confound-checked clean — CAT-70, `docs/audits/CAT70_EXACT_SH_AUDIT.md`: stockpiling-style, no statistics-discarding) |
| 3-epoch frozen-corpus reuse | 51.25%; value val 0.665→0.842 | dead (value overfit) |
| 6-change recipe bundle | 47.0% vs own init | H0, dead |
| Half-corpus cadence | 53.75% continue vs 57.0% | ≥3M-row floor stands |
| Candidate-lineage continuous training (no per-round gate) | gates 37.3%/38.7%, value drift +69% | dead without guardrails (§11) |
| 91M on frozen corpus, epoch 2 | value 0.267→0.393 | parked (needs fresh data) |
| Pure-CPU Modal generation | 19-28h/generation at 2400 cores | dead (economics) |
| Distilled small eval-net for GPU fleet | NN is 4% of leaf cost | dead (wrong bottleneck) |
| BC+PPO (pre-pivot) | 30+ branches, plateau | dead (no exceed-teacher mechanism) |
| c_visit/c_scale further tuning | 16 challengers, 0 winners | closed |

---

## 10. Performance Engineering

### 10.1 CUDA MPS: the biggest operational win (~3x, zero code)

The fleet anomaly: generation ran at ~50ms/eval-equivalent vs 3.4ms benched (10-15x gap). GPU profiling signature: ~90% SM *time-occupancy* with ~0-1% memory bandwidth = tiny-kernel context-thrash from 8 independent worker processes per GPU. Fix: CUDA MPS (Multi-Process Service). Controlled packing grid (workers ∈ {1,8,12,16} × MPS on/off):

| Config | Rows/hr/GPU |
|---|---|
| 8 workers, no MPS (old production) | 26.6k |
| 16 workers, no MPS | 18.4k (**worse** — thrash) |
| 8 workers + MPS | 50.2k |
| 16 workers + MPS | **90.1k** (~linear scaling; 2.75-3.4x production) |

Deployed fleet-wide after a canary. Ops discovery worth recording: **one host-wide MPS daemon caps at ~80 clients** (= five 16-worker GPUs); additional launches hang in CUDA init, and a second daemon cannot coexist with a host-wide one. Correct topology = per-GPU daemons with disjoint device visibility. Fleet aggregate: ~1.1M training rows/hour.

### 10.2 Rust featurization (~20-38x on the featurize slice, staged)

Entity-token and action-context featurizers ported to the Rust engine crate, bit-exact (260-state parity tests, both masking regimes), with raw-byte-buffer marshalling replacing per-element Python object boxing. Featurize slice: ~1.0-1.4 ms → 36-52 µs per leaf. Plus a crate-level fix (an internal function rebuilt the action-space table *per action*: 4,893 → 98 µs at wide roots). All flag-gated and bundled for a single fleet restart. Combined with MPS, the projected next-restart fleet gain is another ~1.3-1.5x.

### 10.3 Modal L4 burst fleet

One-worker-per-container L4 fleet (≤45 GPUs cap), 33.6 games/hr/GPU. Instructive failure: Modal's auto-retry after preemption reinvoked the same run id, whose "resume" path *deleted the partial output and restarted from game 0* — 39 L4s ran 16 hours producing zero completed parts before this was traced. Fixes: incremental shard-level resume + part sizes small enough (~100-150 games) to finish between preemptions.

---

## 11. The Continuous Flywheel

### 11.1 Design rationale

After three discrete promotions, we flipped to a KataGo-style continuous loop. Literature synthesis that guided the design: small-compute teams (KataGo, Leela Zero) *kept* a promotion gate; giant-compute teams (AlphaZero, MuZero) dropped it — the opposite of naive intuition, because a 5000-TPU fleet dilutes a bad checkpoint's data while a small team gets poisoned. We're in KataGo's class, so: **continuous training over a growing window + a cheap gate on the checkpoint that feeds self-play.** Window/replay parameters follow KataGo's published guidance (windowed replay, sample-reuse targets ~4-8x, checkpoint refresh to workers every few hundred k samples, opponent-pool play 15-25% against older checkpoints to prevent strategy cycling — the pool is designed but not yet wired into generation).

The loop (one round ≈ 13-15 min):
1. Generate 24 own-games with the champion (B200 GPU 0).
2. Ingest fleet data batches (16 A100s generating champion self-play; an md5 contract asserts the fleet's checkpoint equals the registry champion — this guard caught one contaminated stream generating against a stale champion).
3. Train: champion-init, production recipe, steps = fresh rows / reuse-target (3.0), anchor-holdout validation.
4. Gate every 6 rounds: 300 paired games at n=16 vs champion, pentanomial SPRT, auto-extension +150 games (disjoint seeds) while the verdict is continue-with-LLR > +0.5, ceiling 600.
5. Promote (rotate the whole fleet to the new champion) or hold.

### 11.2 What broke first: the launch bug chain

The first loop instance died repeatedly of the flag-list trap (§13.1): a missing `--lazy-interior-chance` (0 games in 27 minutes — the 65x lever off), missing architecture flags, a training command missing 7 recipe flags (which mistrained a candidate at lr 2e-4 — quarantined before it could be promoted), missing `--c-scale`/temperature flags in generation. The loop was retired wholesale (its seed range burned) and relaunched with every command pinned. The general lesson: **an orchestrator that composes CLI commands inherits every default-override trap of every tool it calls, multiplied.**

### 11.3 The deep finding: value-head self-distillation drift

As originally designed, each round trained from the *previous candidate* (lineage), accumulating gradient between gates. The first two gates came back catastrophic: **37.3% (LLR −4.58)** and **38.7% (LLR −3.82)** — training was making the candidate *worse*.

Root cause, quantified with the lr≈0 probe: the **value head migrated to the window distribution**. On champion-distribution data, champion value error 0.2185 vs candidate 0.3698 (**+69%**), while the policy remained a near-clone (KL 0.018). Gates at n=16 are completed-Q-dominated, so value miscalibration directly loses games. Two accomplices: (a) per-round random re-splitting of a persistent window leaked ~95% of validation rows from earlier training sets — validation metrics *improved* while board play collapsed, i.e. the val set was measuring memorization; (b) fresh-optimizer restarts without warmup added a real but recoverable loss bump.

This is the third independent manifestation of the value-fragility law (after recipe-B and the 91M probe), now in continuous form: **six compounded training doses on the same distribution without a gate = the pathology; one dose = the proven recipe.**

**Fix package (current regime):** champion-init every round (one dose per gate candidate); the pinned anchor holdout (§8.3) with a drift tripwire (abort if value error > 2x champion baseline); gate power raised (150 games can't accept a ~55% candidate at elo1=30 → 300 + extensions); sample-reuse target cut 6 → 3.

**Validation of the fix:** the first champion-init round's candidate beat the champion baseline on *all* anchor axes (policy 1.3972 / value 0.2418 / top-1 0.5909 vs 1.4132/0.2492/0.5827) with zero drift — and, notably, its numbers converged to within noise of an independently-trained discrete-style generation on the same data (policy 1.3961/1.3972). A flywheel round with fleet feed ≡ a discrete mini-generation, mechanically confirmed.

---

## 12. Where We Are Right Now: The Plateau (turn 4)

### 12.1 The two gate measurements

- **Gen-4** (discrete-style: one training run over the full 4.58M-row fed window): anchor telemetry showed the historical promotion signature (policy-led improvement, zero value drift). Gate at n=8: **52.25%** over 400 games, pentanomial LLR +0.06 — statistically flat.
- **Flywheel round-17 candidate** (champion-init round on the same data family): gate at n=16: 54.0% (LLR +1.17) → extension 1: 54.0% (+0.50) → extension 2: 50.7% (−0.50) → **capped at 600 games: 53.17%, cumulative LLR +1.26** — continue → hold.

Pooled across both independent measurements: **528/1000 = 52.8% vs gen-3 (one-sided p ≈ 0.04 vs coin-flip; two-sided ≈0.08 — suggestive, not conventionally significant, §0 hygiene note)** — a real ~+20-Elo improvement that cannot pass a +30-Elo gate.

### 12.2 The plateau evidence

Anchor telemetry across nine consecutive flywheel rounds is **flat** (policy CE 1.397-1.407, value 0.240-0.244, top-1 0.586-0.592 — all pinned at the same level regardless of fresh feed volume, including a 750k-row round). **The current window — all gen-3 self-play — has been fully distilled.** More training rounds on the same distribution provably do not move the candidate.

### 12.3 The external check

The turn-4 candidate's external panel: 41.0% vs catanatron_value (champion: 45.7%; CIs overlap at n=200) and 66.8% vs AB3 (identical to champion). So the internal +20 Elo has not visibly moved the external needle — though 200-game panels cannot resolve a 4-5-point difference; a 1000-game panel is in flight.

### 12.4 The decision on the table (see §16, Q1)

The gate (elo0=0/elo1=30) was designed for discrete generations with ~50-Elo turns. The flywheel now produces real +20-Elo candidates that will **hold forever** under it — at a true 53%, each 150-game extension adds ~+0.5 LLR and the cap lands near +1.2-2.2, never the +2.94 needed. Meanwhile expert iteration's entire compounding mechanism is *generating data with the improved policy* — withholding promotion withholds the compounding. Options:

1. **Re-spec the flywheel gate as regression protection** (e.g., elo0=−10/elo1=+15, or promote on two consecutive gates with LLR>+1), with the external panel as the binding anti-inbreeding tripwire (two consecutive external declines → revert). Matches KataGo's soft gating and original-AlphaZero's ungated replacement.
2. **Keep the bar, extend gates to 900-1200 games** — a half-day per verdict and probably still "continue" at a true +20.
3. **Hold and improve training** — contradicted by the flat anchor telemetry; the same data does not contain a bigger candidate.

Our lean is (1); the counterargument is the external flatness (possible self-play inbreeding: the candidate may be learning to exploit gen-3's specific style rather than getting absolutely better).

---

## 13. The Bug-Class Catalog

Compressed but complete — most near-misses were infrastructure, and each could have silently produced a wrong scientific conclusion.

### 13.1 The CLI-default-override trap (7+ live instances)
Tools pass every config field explicitly from argparse, so dataclass-default fixes don't propagate, and omitted flags silently revert. Instances: the c_scale fix nearly missed generation; the leak fix nearly missed the gen-1 corpus; the flywheel mistrained a candidate on 7 missing recipe flags; an omitted `--max-decisions` confounded a gate. Countermeasures: pinned complete flag lists, fail-closed regime guards, artifact-embedded provenance.

### 13.2 Seed discipline
Deterministic search + seed-derived chance RNG ⇒ duplicated seed ranges produce bit-identical games. One duplicated 64-seed block double-counted into a **false H1**. Countermeasures: cross-host seed ledger + claim files, generator-side range guards, exact-duplicate-game dedup before pairing in every aggregator, disjoint per-extension gate seeds, a reserved validation-only seed range enforced at both ingest and training.

### 13.3 Regime provenance
Correctness-relevant flags (masking) must be serialized into artifacts; names are intent, not proof. Fail-closed checks beat trust. Gate artifacts now self-certify their observation regime (added after a full H2H table had to be re-run because nothing proved which regime it ran in).

### 13.4 Ops
Orphaned multiprocessing workers survive parent kills (clear via GPU-process listings, kill by explicit PID); `pkill -f` patterns matching your own command kill your own shell; venvs cannot be rsync-cloned (shebang paths); process start time must be compared to code mtime (a gate once launched 42 minutes before its fix landed on disk); mtime-sorted file lists break adjacency-based duplicate detectors (path-sort); one wave-root per ingest batch (mixed batches created absurd seed envelopes that false-quarantined 2.9M rows).

---

## 14. Prior Work and Competitive Landscape (as known to us)

- **Direct competition:** no published Catan agent beats catanatron's hand-tuned AlphaBeta/value bots. *[Citation corrections, CAT-70 — do not repeat the prior versions of these claims:]* "CatAnalysis," previously cited here as the closest AlphaZero-for-Catan attempt, is **unfindable/unverifiable** — we could not locate a primary source for it and it should not be cited further pending one. The "Charlesworth" citation was a **conflation**: there is no Catan arXiv paper under that name; the Charlesworth citation we had actually refers to unrelated Big-2 (a different card game) work, and the real Catan-adjacent PPO material under that general description is a blog post + repo, not a peer-reviewed paper, with no benchmarked superhuman result. **Deep Catan (Driss & Cazenave; EvoApplications/Springer LNCS 2022, also presented at the AAAI-22 RLG workshop — the paper's first author, Driss, was dropped in earlier drafts' "Cazenave, AAAI 2022" shorthand; verified against the primary source, `lamsade.dauphine.fr/~cazenave/papers/DeepCatanEvo.pdf`)** is a prior AlphaZero-style 4-player Catan attempt that was missing from this report — its existence means any first-mover claim below must be qualified, not absolute. Gendre & Kaneko (arXiv:2008.07079) remains the standard academic Catan-RL reference. HexMachina (OpenReview 2026, gray literature, 54.1% vs AlphaBeta) is a more recent learning-based Catan agent but is not peer-reviewed. Given Deep Catan and HexMachina, our defensible claim is: **we found no leak-free, *peer-reviewed*, learning-based two-player Catan system with explicit chance nodes and paired statistical gating** — "first peer-reviewed learning-based" system of this kind, not an unqualified "first" — and we still have zero directly-comparable reference implementations to check our integration choices against.
- **Kao, Guei, Wu & Wu, "Gumbel MuZero for 2048" (TAAI 2022)** — linked from mctx issue #66 — combines Gumbel action selection with stochastic chance nodes for 2048, which **kills the flat claim that no reference implementation combines the two.** The defensible, narrower claim: **perfect-simulator *enumerated* chance nodes + hidden-information masking + paired statistical gating, in a 2-player board game** — real surface Kao et al.'s single-player, fully-observed 2048 setting does not cover, but not an unqualified "Gumbel+chance is new."
- Methods we consciously borrowed: Gumbel MuZero (Danihelka et al. 2022) — search operator; KataGo (Wu 2019) — playout-cap randomization, windowed replay, gating philosophy, auxiliary-head discipline, uncertainty-weighted playout precedent; fishtest/GSPRT — pentanomial paired SPRT; Willemsen et al. (**ALA 2020 workshop / NCA 2022**, not NCA 2021) — hybrid value targets (our λ=0.5 blend is our own extension; their soft-Z is full replacement; see §8.1 for the winner's-curse caveat on our own λ result); Go-Exploit (arXiv:2302.12359) — high-regret restart tooling (built: bit-exact state reconstruction + restart generation; not yet in production data); UCT-V-P-style variance-aware search (D2 arm); AlphaZe∗∗ (masked-AZ viability on imperfect info); MiniZero (search-budget curricula — raising sims before the net is ready hurts); Neumann & Gros (arXiv:2210.00849, params^0.88 scaling); PFSP/AlphaStar league sampling (planned for the opponent pool); ReZero/reanalyze (arXiv:2404.16364, on the roadmap for sample reuse).
- Claims we checked and could NOT verify in sources: "KataGo grew visit counts across the run as a schedule" (playout-cap randomization is per-move, not a cross-generation ramp).

---

## 15. Code Inventory (descriptive, for architecture critique)

~35 tools + a core library, Python with a Rust engine crate. The load-bearing components:

| Component | What it does | Size/notes |
|---|---|---|
| `gumbel_chance_mcts.py` | Gumbel MCTS + chance nodes; all flag-gated arms (D1/D2/lazy/exact-SH/symmetry) | core, line-audited |
| `gumbel_self_play.py` + `generate_gumbel_selfplay_data.py` | self-play driver (single-threaded per worker), PCR cadence, shard writer, seed guards | |
| `neural_rust_mcts.py` | evaluator: masking, chance-spectra batching, eval cache, Rust-featurize wiring | |
| `entity_token_features[_rust].py` | featurization (Python reference + bit-exact Rust port) | |
| `train_bc.py` | trainer: memmap/concat corpora, λ-targets, masking, anchor validation | ~5,700 lines |
| `build_memmap_corpus.py` | npz→memmap, duplicate-seed detection, path-sorted manifests | |
| `gumbel_search_cross_net_h2h.py` | THE promotion gate (paired pentanomial SPRT, per-pair isolation, progress files, provenance) | |
| `gumbel_search_vs_bot_h2h.py` | external panel vs catanatron bots (cross-engine lockstep) | |
| `continuous_flywheel.py` | the loop: generate → ingest → train → gate → promote; journaled rounds | |
| `flywheel_feed_daemon.py` | fleet→window ingestion (md5 champion contract, dedup, quarantine, val-range enforcement) | |
| `sprt_gate.py` | pentanomial GSPRT math (Van den Bergh/fishtest port, verified) | |
| catanatron_rs (Rust crate) | engine: corrected rules, batched chance spectra, featurizers, 0.1.2/0.1.3 wheels | |
| monitoring collector + Grafana | SSH-polled fleet metrics, 16-17 alert rules | |

Everything experimental is flag-gated with default-off = bit-exact no-op, enforced by tests. Full test suite ~750-900 tests depending on branch.

---

## 16. Questions for the Expert Reviewer

These are the decisions and uncertainties where outside perspective has the highest value. Ranked.

**Q1 — Promotion criterion for the continuous loop.** Given §12: real +20-Elo candidates, a gate spec'd for +30, a plateaued window, and an external panel that hasn't confirmed transfer — what is the right promotion rule for a KataGo-compute-class continuous loop on an *asymmetric, stochastic* game? Is regression-protection gating (elo0=−10/elo1=+15) sound, or does the external flatness warrant holding? What did comparable projects actually do at this decision point, and what failure modes followed?

**Q2 — Escaping the plateau.** The window has been fully distilled (flat anchor telemetry across 9 rounds). Beyond promotion (fresh data from a better policy), what else moves a plateaued expert-iteration loop? Candidates we know of: opponent-pool data (built, unwired), Go-Exploit/high-regret restarts (built, unwired), search-budget increase (n=64→96, we gate it on calibration per MiniZero), reanalyze/ReZero-style target refresh, bigger net. What does recent work say about ordering these?

**Q3 — The compression trend.** Turn gains went +49 → +49 → +33 → +20 Elo at roughly constant data per turn. Is this the expected diminishing-returns curve of expert iteration at fixed search budget and net size, or a symptom (e.g., value-target quality ceiling, search operator ceiling at n=64, self-play distribution narrowing)? What measurements would distinguish these?

**Q4 — The external-transfer gap.** Internal ladder +150 Elo cumulative, external +70 Elo, and the last internal +20 shows ~0 external. How much of this is normal (internal ladders always inflate) vs an inbreeding warning? Are there better external-validity instruments than a fixed bot panel (e.g., population-based eval, held-out style opponents)?

**Q5 — Value-head fragility.** Four independent failures reduce to "the value head cannot revisit the same distribution." Our mitigations are all *avoidance* (one-dose training, champion-init, anchor tripwires). Is there a *structural* fix — e.g., categorical/two-hot value (built, untested in production), lagged target networks, value-head-specific LR/regularization, auxiliary heads finally weighted on, ensembling? What does 2023-2026 work say about value-target quality in stochastic games specifically?

**Q6 — Search at wide stochastic roots.** Our c_scale=0.03 (33x below the Gumbel paper) survived a 15-arm ablation, and the exact-budget-SH "fix" lost. Is there principled recent work on Gumbel/completed-Q at *wide, near-tied, stochastic* roots we should adopt — variance-aware selection, progressive widening for chance, forced-playouts + policy-target-pruning (KataGo-style), value-uncertainty propagation?

**Q7 — Architecture.** Dense transformer over entity tokens, adjacency thrown away, actions scored by cosine-to-CLS, value from CLS, 35M params, and the built cross-attention upgrade *lost* its A/B at 47.8M. Is that A/B result more likely "architecture doesn't matter yet at this data scale" or "the upgrade was tested wrong" (single data point, one seed)? Would you invest in: graph-biased attention, D6-equivariance, scale (params^0.88 says yes; our leaf-cost economics make it nearly free at inference), or nothing until the data engine improves?

**Q8 — Anything we're not asking.** Given the full record above — what's the most important thing we appear not to know?

---

## Appendix A: Production Configurations (exact)

**Search (generation & gates):** n_full=64 / n_fast=16 / p_full=0.25 / c_visit=50 / c_scale=0.03 / value_squash=tanh / max_decisions=600 / max_depth=80 / lazy_interior_chance / public_observation / correct_rust_chance_spectra / temperature 1.0 for 90 decisions then argmax. Gates: n=8 (sensitive) or n=16 (flywheel), 8 workers/GPU, paired color-swap, pentanomial SPRT elo0=0/elo1=30, α=β=0.05.

**Training:** entity_graph 640×6×8 heads, dropout 0.05; 1 epoch, batch 4096, bf16, Adam lr 3e-5, warmup 100; soft policy targets weight 0.9 renormalized over legal support; value weight 1.0; value-target-λ 0.5; final-VP aux 0.1; truncated-VP-margin 0.25; masking on; init from champion.

**Flywheel:** 24 own-games/round + fleet ingestion; reuse target 3.0; gate every 6 rounds, 300 games n=16 + extensions (+150 while continue & LLR>+0.5, cap 600); champion-init; anchor holdout validation with 2x-value tripwire; md5 champion contract on ingestion; seed stride 10,000,019/round, gate seeds offset +1e11, validation-only seed range reserved.

## Appendix B: Elo Arithmetic Used

Win rate ↔ Elo: 55% ≈ +35, 57% ≈ +49, 54.7% ≈ +33, 53% ≈ +21 (logistic). SPRT bounds at α=β=0.05: log(19) ≈ ±2.944. Pentanomial pair-score variance estimated from the WW/split/LL empirical distribution (typical split rate at n=8-16 gates: 0.4-0.6).

## Appendix C: Timeline

| Date (2026) | Event |
|---|---|
| 07-02 | Pivot from BC+PPO; 11-bug audit; ~20 fixes |
| 07-03 | Engine-equivalence closed (8 rules bugs); search built; value-repair v1 fails informatively |
| 07-04 | **Gate-A failure** (search loses to raw 19%); mctx rescale diagnosis; c_scale fix; D1/D2 built |
| 07-05 | Value-repair v2 reverses Gate-A (67-71%); **hidden-info leak found + fixed**; symmetry finding; v3a beats v3b; lazy chance exonerated; gen-1 generation |
| 07-06 | **Gen-1 promoted** (57%); external ladder; gen-2 recipe matrix (7 arms); **gen-2A promoted** (57%); λ-target discovery (59%); joint search ablation (kill list); Strategy v2 adopted |
| 07-07 | **Gen-3 promoted** (54.71%/700); external 45.7% vs value bot; MPS fleet rollout (~3x); Rust featurize staged; continuous flywheel launched; lineage-drift pathology found + fixed; fleet feed live |
| 07-08 | Gen-4 gate flat (52.25%); flywheel round-17 hold-at-cap (53.17%/600); plateau established; promotion-criterion decision pending; 1000-game external panel in flight |




NOW BAKC TO HUMAN OR TWV HERE SHWT AI NEED YOY TODO TODO RN YOUA R ETHINKING HARD:
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



ALSO WHEN GIVING ADVCI EYOU CNA TELL US WORU HWOEL SYSTEM IS WORNG YUOUA RE TO DO WHATVER GEST THE GAOL IS TO CETRATE TEH #1 CATAN BOT DO ANYTHING TO AHCIEV TAHT
ALSO WHEN GIVING ADVCI EYOU CNA TELL US WORU HWOEL SYSTEM IS WORNG YUOUA RE TO DO WHATVER GEST THE GAOL IS TO CETRATE TEH #1 CATAN BOT DO ANYTHING TO AHCIEV TAHT
ALSO WHEN GIVING ADVCI EYOU CNA TELL US WORU HWOEL SYSTEM IS WORNG YUOUA RE TO DO WHATVER GEST THE GAOL IS TO CETRATE TEH #1 CATAN BOT DO ANYTHING TO AHCIEV TAHT