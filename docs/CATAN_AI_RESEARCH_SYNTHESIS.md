# Why Nobody Has Cracked Catan — and the Recipe to Win

Synthesis of a deep literature pass (2020–2026 arXiv) across the five hard parts, mapped to
**our** system (35M graph net, BC → PPO self-play + league, model-free, cheap CPU sim fleet +
big GPUs). Last updated 2026-06-28.

---

## The thesis (the surprising part)

**Catan's unsolved status is ~60% an ENGINEERING gap, ~25% an ALGORITHM gap, ~15% a COMPUTE
gap — not a fundamental wall.** The evidence is blunt: *every* strong published result either
**deletes the hard parts** (2-player, no trading, hidden info removed) or **beats only a depth-2
heuristic AlphaBeta by a few points.** The strongest learned agent as of 2026 (HexMachina) clears
depth-2 AlphaBeta by ~3 points in 2-player, no-trade-emphasis games. Nobody has a strong
**4-player, hidden-info, *with trading*** bot. The simulator (Catanatron) runs thousands of
games/sec on CPU, so scale is *not* the binding constraint the way it was for StarCraft.

> **Translation:** the frontier is open, and the gap is *investment + the right stack*, not a
> known theoretical barrier. A GNN + autoregressive trade head + piKL-anchored trading + belief
> head + PFSP league has **never been combined for Catan** — that exact design would be novel and
> is a credible path to the first strong 4p-with-trading agent.

The two best historical precedents tell the whole story:
- **Gendre & Kaneko 2020** (arXiv:2008.07079): "first time RL beats existing benchmarks" — *but
  only 2 players and NO TRADING.* They dropped the elephant.
- **Charlesworth (settlers-rl)**: full 4-player PPO+search from scratch, ~1 GPU-month → "definite
  learning progress, not at good-human level," and explicitly: "agents waste so much time
  proposing stupid trades… remove trading because it was probably a bit ambitious." The trade
  head collapsed.

**The trade head is the elephant everyone walks around.** That's our opening.

---

## The five hard parts → the modern technique that solves each

### 1. Four-player non-zero-sum / non-transitivity / kingmaker
- **Key result (2025):** *Reevaluating Policy Gradient Methods for Imperfect-Information Games*
  (arXiv:2502.08938) — a 7000-run bake-off found **plain regularized PPO/PPG/MMD ≥ the entire
  CFR/PSRO/R-NaD family.** For a model-free team, **PPO is the frontier, not a compromise.**
- **Why our BC-anchor is theoretically blessed:** Magnetic Mirror Descent (arXiv:2206.05825) is
  *structurally identical* to KL/trust-region PPO toward a "magnet" reference. **Our BC policy is
  that magnet.** So a KL-regularized PPO toward BC = a principled last-iterate solver, for free.
- **The kingmaker/alliance dilemma** (arXiv:2003.00799) is the Catan "gang up on the leader"
  dynamic. There is **no clean equilibrium fix** — independent self-play converges to bad joint
  outcomes. The only practical defense is **robustness-via-population**: a **PFSP league +
  exploiters** (AlphaStar). Nash is intractable *and the wrong target* for 4p general-sum.
- **If we see rock-paper-scissors strategy cycling:** A-PSRO (ICML 2025) extends PSRO to
  general-sum; use α-Rank/(C)CE only *offline* to rank league members on the non-transitive ladder.
- **➡️ Our move:** KL-regularized PPO (anchored to BC) + PFSP league + exploiters. We already
  have the league scaffolding; the missing piece is the explicit BC-KL anchor and real PFSP.

### 2. Hidden cards / belief tracking
- **Do NOT build ReBeL / PBS-CFR** (arXiv:2007.13544): its tractability relies on 2p-zero-sum
  *convexity* that 4p general-sum breaks, and PBS subgames blow up with Catan's large hidden state
  (each player's resource multiset + dev-card hand ≫ poker cards).
- **The pragmatic SOTA = a learned belief auxiliary head** (DouZero+ arXiv:2204.02558; ODMC,
  ASOC 2024): predict each opponent's **hidden resource counts + dev-card holdings**, supervised
  from the privileged simulator state, and **feed the prediction back into the policy+value net.**
  ODMC reports **~4× faster training at equal strength.** This is also what **denoises the value
  baseline** — the real fix for dice/opponent-luck variance (see #3).
- **If we ever add eval-time search:** prefer **MMDS / update-equivalence** (arXiv:2304.13138),
  which is non-PBS, scales to large hidden-info games, and *shares our PPO update dynamics* — not
  ReBeL. Or cheap **determinized rollouts** (PIMC-lite) sampled from the belief head's posterior.
- **➡️ Our move:** add a belief aux head (count/multinomial prediction off privileged state),
  concatenate its output into policy+value inputs. Model-free, no CFR. This is the highest-value
  *new* model component.

### 3. Dice variance / sparse, delayed reward
- **The math of why it's brutal:** RUDDER (arXiv:1806.07857) — *return-estimator variance grows
  exponentially with reward delay.* Catan = win/lose at 10 VP, ~75+ moves away, gated by dice.
- **Cheapest wins (do first):** tune **GAE λ down** (0.9–0.95) to trade dice variance for bias;
  add a strong **belief-conditioned value baseline** + **reward centering** (control variates).
- **Potential-based VP reward shaping** (policy-invariant, safe) + optional **RUDDER-style return
  redistribution** to credit the load-bearing moves (the 6/8-hex settlement, the longest-road grab).
- **Reanalyze** (EfficientZero, arXiv:2111.00210): replay old CPU-fleet games through the *current*
  value net to refresh targets — neutralizes PPO's on-policy data-reuse weakness **without MCTS.**
- **Do NOT rewrite to Stochastic MuZero as a first move.** Its sample-efficiency edge is wasted
  when games are nearly free (our case); it adds MCTS cost over the large trade action space; and
  procgen evidence shows tuned PPO stays competitive with a cheap simulator. *Steal its ideas* —
  the **afterstate framing** (action → afterstate → dice chance node) for any chance modeling, and
  **Reanalyze** — not the whole machine. KLENT (arXiv:2602.10894, 2026) is the model-free
  "stable+efficient without tree search" reference.
- **➡️ Our move:** λ-tune + potential-based VP shaping + Reanalyze on the CPU fleet + belief-
  conditioned baseline. All cheap, all reuse our existing PPO + sim throughput.

### 4. Trading / negotiation (THE elephant — our differentiator)
- **The documented failure mode:** unregularized self-play trade policies **collapse to no-trade
  or spam** (settlers-rl; Deal-or-No-Deal arXiv:1706.05125 showed unregularized RL negotiators
  turn antisocial/exploitative).
- **The SOTA fix = piKL / human-regularized planning** (CICERO line, arXiv:2210.05492): a trade
  proposal policy with a **KL penalty to a BC anchor** — `U = u(π) − λ·KL(π ‖ τ_BC)` — annealing
  λ high→low so it starts sensible and gradually optimizes. **This is the single most important
  anti-collapse mechanism**; every prior Catan project that omitted it saw trading collapse.
- **Separate the two jobs:** (a) a **proposal generator** (autoregressive: give-set → receive-set
  → target), BC-trained on heuristic/human trade logs to form the anchor τ; (b) a **separate
  acceptance model** `P(accept | offer, opponent)`, supervised on observed accept/reject, with
  **EV(offer) = P(accept) × Δ(win-prob) − cost.** Optimize the generator toward high-EV,
  accept-likely trades.
- **➡️ Our move:** a trade sub-policy = autoregressive proposal head + acceptance head + EV
  scoring + **piKL KL-anchor**. This is the part nobody has done well — and where we can be first.

### 5. Huge structured / combinatorial action space
- **Autoregressive *factored* action head, NOT a flat 607-way categorical** (AlphaStar; Conditional
  Action Trees arXiv:2104.07294 → exponential becomes linear). Top-level action-type
  (build/buy/knight/robber/steal/maritime/propose-trade/accept/reject/end), then condition the
  next factors on it.
- **Pointer-over-graph-nodes** for build/robber/trade *targets* — Symbolic Relational Deep RL
  (arXiv:2009.12462) is literally **GNN embeddings → autoregressive action + params**, i.e. *our
  architecture.* Pointer selection over vertices/edges/tiles **generalizes across board layouts**
  (no fixed categorical slots).
- **Proper logit-level masking at EVERY factor**, integrated into the PPO loss (MaskablePPO).
  Naive *post-hoc* masking leaves logits unchanged and **inflates PPO KL → destabilizes training**
  (arXiv:2601.09293). Each factor becomes a small masked categorical — the combinatorial blow-up
  disappears.
- For the combinatorial give/receive bundles, **Sampled/Gumbel MuZero** sampling (arXiv:2104.06303)
  is the fallback if a factor can't be enumerated.
- **➡️ Our move:** replace the flat 607-way head with an autoregressive, pointer-over-graph-nodes,
  per-factor-masked head. Plays directly to the 35M graph net's strengths.

---

## Why OUR system is uniquely positioned to be first

The SOTA recipe (AlphaStar-style BC→league-PPO + Cicero-style anchored trading + a belief head)
**requires exactly the assets we already have:**

| SOTA recipe needs | We have |
|---|---|
| A competent **BC warm-start** (the KL magnet) | ✅ 35M graph net BC'd to 82% teacher-acc |
| A **fast, cheap sim fleet** (volume beats sample-efficiency tricks) | ✅ Modal + GCP Spot + Catanatron (thousands/sec) |
| **Model-free PPO + league** machinery | ✅ `train_ppo.py` + league orchestrator already built |
| A **graph substrate** for pointer-over-nodes + belief heads | ✅ `XDimGraphPolicy` (graph net) |
| **Big GPUs** for batched inference + Reanalyze | ✅ B200 / GH200 / A100 |

We are *not* deleting the hard parts (we kept the trade action space + added the 4p path). The
2025 reevaluation says our model-free PPO is the **frontier**, not a compromise. The literature
gap — "no one has combined GNN + autoregressive trade head + piKL anchor for Catan" — is precisely
the stack we're positioned to build.

---

## The winning stack (one picture)

```
35M GRAPH NET  ──►  ┌─ autoregressive, pointer-over-nodes, per-factor-masked ACTION HEAD ─┐
   │                │   type → (build/robber target = node ptr) | (trade: give→recv→target)│
   │  + BELIEF AUX HEAD (predict opp resources/dev-cards → fed back to policy+value)        │
   │  + VALUE head (belief-conditioned → low-variance baseline)                             │
   │  + TRADE sub-policy: proposal generator + acceptance model + EV, piKL KL-anchor to BC  │
   └────────────────────────────────────────────────────────────────────────────────────┘
                                   │
        TRAINED BY  ►  KL-regularized PPO (magnet = BC)  +  potential-based VP reward shaping
                       +  GAE λ-tuned  +  Reanalyze (CPU fleet)  +  PFSP LEAGUE + exploiters
                                   │
        EVAL  ►  held-out opponents (AlphaBeta/jSettlers/frozen) + moves-to-win
                 + optional determinized rollouts / MMDS at decision time only
```

## Priority order (signal-per-dollar)

1. **GAE λ-tune + potential-based VP reward shaping + belief-conditioned value baseline** — cheap,
   directly attacks dice variance. *First.*
2. **Belief auxiliary head** — the highest-value new model component; denoises value + speeds
   training ~4× (ODMC). 
3. **Explicit KL-to-BC anchor in PPO** (MMD) + **real PFSP league + exploiters** — kills the
   non-transitive cycling (our 98.5%-reject symptom).
4. **Reanalyze on the CPU fleet** — off-policy target refresh, more signal without MCTS.
5. **Autoregressive pointer action head** — replaces the flat 607-way head; unlocks #6.
6. **Trade sub-policy with piKL anchor + acceptance/EV model** — highest *impact*, highest effort;
   the actual unsolved problem and our differentiator. Do it on the new action head.
7. **Eval-time determinized rollouts / MMDS** — last-mile strength, inference only.

## Explicit "do NOT" list (saves months)
- ❌ ReBeL / PBS-CFR (unsound general-sum, intractable hidden state).
- ❌ Full Stochastic MuZero rewrite as a first move (sample-efficiency wasted on a cheap sim).
- ❌ Flat 607-way categorical for trades (can't express give/receive dependency; collapses).
- ❌ Unregularized trade self-play (collapses to no-trade/spam — the universal prior failure).
- ❌ Chasing Nash / α-Rank as a training target for 4p (intractable + wrong; use PFSP).
- ❌ Naive post-hoc action masking (inflates PPO KL, destabilizes — mask at the logits).

## Honest caveats
- The 2025 "PPO ≥ CFR" result used homogeneous games (no chance, binary rewards). Catan has heavy
  stochasticity + dense VP shaping — **re-validate empirically**; keep R-NaD/PSRO as ablation
  fallbacks, don't delete on faith.
- **There is no human-grounded ELO eval for any Catan bot.** "Superhuman" is currently unmeasurable.
  Building a credible **human eval ladder** (e.g. via the Colonist replay data + online play) may
  be the single highest-value missing artifact — and the thing that would make a result *count*.
- No published work combines our full stack for Catan → this is a research frontier (upside:
  novelty/publishability; downside: no off-the-shelf baseline to copy for 4p-hidden-trading).

## Key sources (full lists in the three research threads)
PPO≥CFR arXiv:2502.08938 · MMD arXiv:2206.05825 · alliance/kingmaker arXiv:2003.00799 ·
A-PSRO (ICML 2025) · DouZero+ arXiv:2204.02558 · ODMC (ASOC 2024) · MMDS arXiv:2304.13138 ·
RUDDER arXiv:1806.07857 · EfficientZero arXiv:2111.00210 · Stochastic MuZero (ICLR 2022) ·
KLENT arXiv:2602.10894 · piKL/Diplomacy arXiv:2210.05492 · CICERO (Science 2022) ·
Deal-or-No-Deal arXiv:1706.05125 · Conditional Action Trees arXiv:2104.07294 ·
Symbolic Relational Deep RL arXiv:2009.12462 · MaskablePPO/masking arXiv:2601.09293 ·
Sampled MuZero arXiv:2104.06303 · Catan: Gendre&Kaneko arXiv:2008.07079, settlers-rl,
Deep Catan (Driss&Cazenave 2022), Catanatron, HexMachina arXiv:2506.04651.
