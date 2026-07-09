# PPO Training Architecture Plan (catan-zero)

**Status:** PLAN — not yet implemented for the 35M graph model. PPO machinery exists
(`tools/train_ppo.py`, `src/catan_zero/rl/torch_ppo.py`) but runs only the legacy flat
`TorchPPOPolicy`. This doc specifies the *system* architecture for running PPO self-play
on the **35M graph model (`XDimGraphPolicy`)** across our heterogeneous fleet (B200 / GH200 /
A100 GPUs + Modal/GCP CPU).

Owner: catan-zero. Last updated: 2026-06-28.

---

## 0. The one insight that drives everything

**A 35M-param model is trivially small for our GPUs.** A single B200 (~180 GB) can hold
hundreds of copies and do the gradient step in microseconds. Therefore:

> The learner's gradient compute is **never** the bottleneck. The bottlenecks are
> (1) **rollout throughput** (CPU game simulation), (2) **inference batching efficiency**
> (keeping the GPU fed with enough observations), and (3) **training-target quality**
> (reanalysis/search). 

This flips the usual "big model → big GPU" intuition. Our B200/GH200/A100 headroom should go
to **batched inference, MCTS reanalysis, and running many league agents in parallel** — NOT to
a single tiny gradient loop. Designing around this is the whole point.

---

## 1. Hardware inventory & assigned roles

| Machine | Spec | Best role |
|---|---|---|
| **B200 node** (2× Blackwell, ~180 GB ea, world_size=2) | flagship | **Primary online RL cell** + MCTS reanalysis. GPU0 = inference server, GPU1 = learner (or 2 league agents). |
| **GH200** (H100 96 GB + 72 Grace ARM cores, 480 GB unified) | unified mem + lots of local CPU | **Self-contained cell**: 72 Grace cores = local actors, Hopper = inference+learner. Ideal SEED-RL node. |
| **A100** (40/80 GB) | mid GPU | **League-agent cell** (an exploiter or 2p-curriculum agent) + scoring. |
| **Modal** (≤600 vCPU, 75×8) | burst CPU, separate cloud | **Offline** teacher/reanalysis data factory (latency-insensitive). Already in use. |
| **GCP Spot** (1,000 vCPU, pending quota) | cheap bulk CPU, GCP us-central1 | **Online actors for a GCP cell** (co-locate a GCP GPU) **or** offline eval/data. |

**Critical placement rule:** the online RL hot loop (actor → inference → action) is
**latency-sensitive** (one round-trip per decision). Cross-cloud (Modal CPU → Oracle B200)
adds tens of ms per decision and **kills throughput**. So:

- **Online RL = node-local or same-cloud cells.** Actors co-located with their inference GPU.
- **Offline work = the big cloud CPU fleets** (Modal + GCP Spot), which batch and upload shards
  where latency doesn't matter.

---

## 2. Core architecture: two tiers

```
┌─────────────────────────────────────────────────────────────────────────┐
│ TIER 1 — ONLINE RL (latency-sensitive, GPU-node-local "cells")            │
│                                                                           │
│   ┌─ B200 cell ──────────┐  ┌─ GH200 cell ─────────┐  ┌─ A100 cell ────┐  │
│   │ local CPU actors ────┼─▶│ local CPU actors ────┼─▶│ local actors   │  │
│   │  (Catanatron games)  │  │  (72 Grace cores)    │  │                │  │
│   │        │ obs/act      │  │        │              │  │       │        │  │
│   │  GPU0 inference svr   │  │  Hopper infer+learn   │  │ A100 inf+learn │  │
│   │  GPU1 learner (V-trace│  │                       │  │                │  │
│   │  /PPO update)         │  │  = main agent OR      │  │ = exploiter /  │  │
│   │  = MAIN agent         │  │    league member      │  │   2p curriculum│  │
│   └──────────┬───────────┘  └──────────┬───────────┘  └───────┬────────┘  │
│              │  checkpoints + payoff results                  │            │
└──────────────┼───────────────────────────────────────────────┼───────────┘
               ▼                                                 ▼
   ┌───────────────────────────── SHARED STATE (GCS) ───────────────────────┐
   │  league registry (agents, snapshots, PFSP payoff matrix)               │
   │  checkpoint store   │  reanalysis targets (JSONL)  │  teacher shards    │
   └───────────────────────────────────────────────┬────────────────────────┘
               ▲                                    ▲
┌──────────────┼────────────────────────────────────┼──────────────────────┐
│ TIER 2 — OFFLINE (throughput, latency-insensitive cloud CPU)              │
│   Modal 600 vCPU  ── teacher data + MCTS reanalysis game generation       │
│   GCP Spot 1k vCPU ─ league scoreboard/eval matches (the promotion gate)  │
└───────────────────────────────────────────────────────────────────────────┘
```

**Tier 1 (online):** each GPU node is a self-contained **cell** running the SEED-RL pattern
(below). Each cell trains one *league role* (main agent, exploiter, curriculum agent). Cells
never talk to each other in the hot loop — they sync only through the shared checkpoint/league
store.

**Tier 2 (offline):** the big cloud CPU fleets do embarrassingly-parallel, latency-insensitive
work — teacher data, **reanalysis** (run MCTS/search on visited states to produce better
training targets), and **eval matches** for the promotion gate. This is exactly what Modal /
A100 / GH200 already do today; we formalize it.

---

## 3. The cell internals — SEED-RL (centralized batched inference)

For a 35M model, **do NOT run inference on the CPU actors** (the classic IMPALA layout). A 35M
forward pass on CPU is slow and you'd ship weights to every worker. Instead use the **SEED-RL**
layout: **inference is centralized on the GPU**; actors only step the environment.

```
   N actor procs (local CPU)          GPU (same node)
 ┌───────────────────────┐    obs    ┌──────────────────────────┐
 │ each runs M parallel  │ ────────▶ │  Inference server         │
 │ Catanatron games      │           │  - dynamic batching       │
 │ (env step only,       │ ◀──────── │  - 35M policy fwd → π,V,Q │
 │  no model weights)    │   action  │  (holds latest weights)   │
 └──────────┬────────────┘           │                          │
            │ finished trajectories  │  Learner                  │
            └──────────────────────▶ │  - V-trace/PPO update     │
                                     │  - KL-to-BC anchor        │
                                     │  - pushes weights→inf svr │
                                     └──────────────────────────┘
```

- **Actors** (local CPU cores): run `ColonistMultiAgentEnv` self-play. Each actor vector-steps
  M games; at each decision it sends the masked observation (`obs`, `legal_action_ids`,
  `legal_action_context`) to the local inference server and gets back `(action, log_prob, V, Q,
  action_probs)`. Communication = localhost gRPC / shared memory (sub-ms).
- **Inference server** (GPU): dynamically batches observations across all actors → one big
  forward pass of the 35M net → scatters results back. This is what saturates the GPU. Holds
  the latest published weights.
- **Learner** (GPU): consumes finished trajectories, computes GAE/V-trace, runs the
  PPO/V-trace update + KL-to-BC anchor, publishes new weights to the inference server every K
  steps. (Reuse `torch_ppo.ppo_update` / `_gae_returns` / `clipped_ppo_loss`.)
- **Off-policy correction:** because actors act with weights a few updates stale,
  use **V-trace** (clipped importance sampling, IMPALA) on top of the PPO update, OR bound actor
  staleness to ≤2–3 updates and keep vanilla PPO. Start with bounded-staleness PPO (simpler,
  reuses existing code); add V-trace if the inference→learner queue forces more staleness.

**Why this saturates a B200:** one GPU inference server can serve *thousands* of CPU actors.
Catanatron does ~thousands of games/min/core; ~600 actor cores ≈ tens of thousands of
decisions/sec, which a B200 eats in modest batches. **So a single GPU's inference can feed a
huge actor fleet — which is exactly why the *other* GPUs go to the league + reanalysis, not to
duplicate learners.**

---

## 4. What the spare GPU headroom is for (the league + reanalysis)

Since 1 GPU saturates a big actor fleet, the rest of the fleet does the things that actually
make the agent strong:

1. **Population / league (the cure for our 98.5%-reject cycling).** Run multiple cells, each a
   different league role:
   - **Main agent** (B200 cell) — trained vs the whole league by **PFSP** (sample opponents ∝
     their win-rate *against* the main agent).
   - **Main exploiter** (A100 cell) — hunts the current main agent's weaknesses; resets to the
     **BC checkpoint** periodically; snapshots into the league when it beats main.
   - **League exploiter** (GH200 cell) — hunts the whole league's weaknesses.
   - Each exploiter snapshot is added to the frozen pool. Track a **payoff matrix** to confirm
     we've stopped cycling (AlphaStar's Nash-over-league diagnostic).
2. **MCTS reanalysis** (B200 spare compute + Modal CPU). Periodically run a shallow PUCT search
   (policy head = prior, value head = leaf eval, **chance nodes for dice**, **determinization
   for hidden cards**) on a sample of visited states to produce *better* training targets than
   the raw policy — the AlphaZero "search makes the data better than the net" loop, applied as a
   side-channel. Feeds `generate_reanalysis.py` / `reanalysis.py` (already exist).
3. **Eval / promotion gate** (GCP Spot CPU): `evaluate_scoreboard.py` matches vs held-out
   opponents {AlphaBeta, JSettlers, value, BC, frozen snapshots}.

---

## 5. Algorithm spec (what the learner runs)

Reuse the existing PPO stack; add the pieces the research says are non-negotiable for Catan:

- **Base:** clipped PPO (`clipped_ppo_loss`) + multi-player GAE (`_gae_returns`), γ high
  (Catan games are long, ~75+ moves to a sparse win — start γ≈0.997, tune).
- **Action masking (mandatory):** logits masked to legal actions (already done in BC). Without
  it PPO drowns in the ~607-action, mostly-invalid space — *especially* the trade head.
- **KL-to-BC anchor (the single most important addition):** add `β·KL(π_θ ‖ π_BC)` to the loss,
  anchored to the **frozen BC checkpoint**, annealed down on a schedule. Prevents catastrophic
  forgetting of teacher competence while climbing above it (AlphaStar distillation / Cicero
  piKL). The existing EMA/old-policy KL is *not* the same thing — anchor specifically to BC.
- **Reward shaping:** potential-based VP shaping (Φ = f(victory points, settlements,
  longest-road/largest-army progress)) on top of terminal win/loss. Potential-based so the
  optimal policy is unchanged (no reward hacking). Sparse-only is documented to make Catan RL
  "impractical."
- **Curriculum:** start **6-VP-to-win, 1 opponent** (dense signal, low variance) → ramp to
  **10-VP, 3 opponents** (true target). Randomize seat order (strong turn-order bias).
- **Q-aux head + DAgger + anchor-imitation:** already implemented; keep as stabilizers.
- **V-trace:** add when async staleness demands it.

Off-policy/throughput note: PPO is on-policy; the SEED-RL async pipeline makes it slightly
off-policy. Bounded staleness + V-trace handle this. Keep minibatch PPO epochs low (1–2) to
limit drift.

---

## 6. Data flow (one training iteration)

1. Inference server publishes current weights `θ`.
2. Local actors play self-play games (opponents drawn by PFSP from the league), querying the
   inference server per decision; emit trajectories `(obs, mask, action, logp, V, Q, reward)`.
3. Learner pulls a batch of trajectories, computes GAE/V-trace returns + advantages.
4. Learner runs PPO update + KL-to-BC anchor (+ Q-aux, + reanalysis targets when available);
   publishes new `θ`.
5. Every N iters: checkpoint → shared store; trigger offline `evaluate_scoreboard.py` on GCP
   Spot; if it passes the gate, promote into the league registry + payoff matrix.
6. Offline (continuous): Modal/A100/GH200 generate teacher + reanalysis shards; reanalysis
   targets stream back to learners.

---

## 7. Mapping to existing code — build vs reuse

| Component | Status | Action |
|---|---|---|
| PPO loss / GAE / update | EXISTS (`torch_ppo.py`, `ppo_loss.py`) | reuse |
| `XDimGraphPolicy` (35M) | EXISTS, BC-trained; NOT in PPO | **wire into `train_ppo.py:780`** (interface matches) |
| value/log-prob/Q for PPO | EXISTS (`sample_action_value_q`) | reuse; warm value head from BC outcomes |
| League opponent sampling | EXISTS but "shallow" | **upgrade to PFSP + exploiters + payoff matrix** (`league_orchestrator.py`) |
| KL-to-BC anchor | partial (EMA/old-policy KL) | **add explicit frozen-BC KL term + β schedule** |
| Reward shaping / curriculum | MISSING | **add potential-based VP shaping + 6→10 VP, 1→3 opp curriculum** |
| Centralized inference server | MISSING (rollout is serial Python) | **BUILD: batched GPU inference server + actor RPC** (the core new infra) |
| Vectorized / async actors | MISSING | **BUILD: M-parallel env actors per core** |
| MCTS reanalysis | scaffolding (`reanalysis.py`, `generate_reanalysis.py`) | **add PUCT + chance nodes + determinization** |
| Offline data factory | EXISTS (`modal_teacher_factory.py`) | reuse for reanalysis too |
| Eval gate | EXISTS (`evaluate_scoreboard.py`) | run on GCP Spot; add payoff-matrix output |

**The genuinely new infra to build = the SEED-RL cell** (batched inference server + async
vectorized actors + weight publishing). Everything else is reuse or a focused addition.

---

## 8. Phased rollout

- **Phase 0 — 35M into PPO.** Instantiate `XDimGraphPolicy` in `train_ppo.py`; warm-start
  policy+value from the BC checkpoint; run the smoke config
  (`configs/selfplay/ppo_2p_xdim_lite_smoke_a100.yaml`) on one A100 cell with the *current*
  serial rollout. Goal: confirm the 35M model trains under PPO at all, beats `random`/`heuristic`.
- **Phase 1 — SEED-RL cell.** Build the batched inference server + async actors on one node
  (GH200 is ideal: 72 local Grace cores + Hopper). Target ≥10× the serial throughput. Add the
  **KL-to-BC anchor + VP reward shaping + 6VP/1opp curriculum**. Goal: beat AlphaBeta in 1v1.
- **Phase 2 — League.** Bring up exploiter cells (A100, GH200) + the B200 main-agent cell;
  PFSP + payoff matrix; exploiters reset-to-BC. Goal: kill the cycling (reject rate ↓, no
  regression on jsettlers/value); ramp curriculum to 10VP/3opp.
- **Phase 3 — Reanalysis + eval-MCTS.** B200 spare compute + Modal run MCTS reanalysis to
  improve targets; add MCTS (determinization + chance nodes) at *eval/decision* time only.
- **Phase 4 — True board graph.** Replace the tokenized-flat-vector encoder with a real
  hex-tile node/edge graph + public event-history encoder (the handoff's #1 architectural gap);
  re-BC, re-enter the loop.

---

## 9. Sizing / throughput notes

- **Inference:** one big GPU saturates thousands of actors → **1 inference server per cell is
  plenty**; don't shard the 35M model. Dynamic batch target: keep GPU batch ≥1–4k obs.
- **Actors:** ~M=8–32 parallel games per core; cell throughput ≈ (cores × decisions/sec/core).
  GH200's 72 cores alone is a strong cell; add same-cloud Spot CPU to scale a GCP cell.
- **Learner:** trivial for 35M — gradient step is tiny; the learner mostly waits on data. This
  is why spare GPUs go to league/reanalysis, not more learners.
- **Cost lever:** actors are the cost; run them on **Spot CPU** (~$5–7/vCPU/mo) and checkpoint
  to GCS so preemption is free. Keep the **GPU on-demand** (Spot GPUs preempt badly and the
  learner is stateful).

---

## 10. Failure modes & gotchas (from the precedent research)

- **Non-transitivity / cycling** (this is our current 98.5%-reject symptom) → PFSP league +
  exploiters-reset-to-BC; verify with a payoff matrix, not just self-play win rate.
- **BC ceiling + forgetting** → KL-to-BC anchor with a tuned β schedule (too high = stuck at
  teacher level; too low = collapse). This β is the make-or-break knob.
- **Sparse reward** → potential-based VP shaping + VP-to-win curriculum.
- **Trade head** is the hardest to learn → keep it masked; consider curriculum-gating trades
  (build first, enable trades later); anchor trade proposals to BC (Cicero piKL lesson).
- **Eval bias** → always evaluate vs **held-out** opponents (AlphaBeta ~75 moves is the bar);
  report win rate AND moves-to-win; randomize seat order.
- **Cross-cloud latency** → never put online actors in a different cloud from their inference
  GPU. Use the big CPU fleets for offline only, or co-locate a GCP GPU with GCP Spot CPU.
- **Don't over-rely on "surgery"/continual hacks** → if obs/action space changes materially,
  prefer a clean re-run (OpenAI Five's "Rerun" beat the surgically-evolved model).

---

## 11. Open questions to resolve before building

1. Does the BC `XDimGraphPolicy` checkpoint already have a usable value head, or do we warm-start
   it from teacher-game outcomes first?
2. Is the trade head already action-masked end-to-end (prior Catan RL notoriously left it open)?
3. Primary eval target: **1v1 vs AlphaBeta** (clean non-transitivity story, matches the
   literature bar) as the curriculum entry, ramping to **4-player** (true target)?
4. Which cloud hosts the scalable online cell — a GCP GPU + GCP Spot CPU cell (cleanest,
   same-region), with the Oracle/Lambda B200/GH200/A100 as node-local cells?
5. V-trace now, or bounded-staleness PPO first?
```
