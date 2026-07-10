# CatanZero Technical Handoff

Date: 2026-06-26

Repo path:

```text
<verified-local-checkout>
```

## End Goal

Build a Catan AI strong enough to rank #1 on Colonist.io under permissioned, legitimate use.

The practical technical target is:

1. A four-player Catan agent that can play legal games end to end.
2. A model that beats current internal baselines with statistical confidence.
3. A system that handles realistic Colonist-style play: hidden information, robber choices, discards, development cards, trades, ports, and multiplayer dynamics.
4. A training loop that keeps only candidates that beat the current champion under a fixed gate.
5. Eventually, a deployable agent with low-latency inference and enough robustness to beat strong humans.

We are not there yet.

## Current High-Level State

The system has a playable RL/training stack, but the learned model is still weak relative to the final goal.

What exists:

- Catanatron-backed Catan environment wrappers.
- Multiagent four-player self-play environment.
- Structured legal action handling.
- Candidate-action Torch PPO policy.
- Value/heuristic/search teacher policies.
- Self-play training loop with PPO, Q-value auxiliary head, KL regularization, anchor imitation, and checkpointing.
- Evaluation against random, heuristic, and Catanatron value baselines.
- Box launcher/orchestration for remote CPU training.
- Reanalysis data path for search-generated training targets.

What does not exist yet:

- A true graph/history transformer policy.
- Explicit hidden-resource/development-card belief model.
- True public-belief or information-set search.
- A production-quality population league.
- Strong, verified trading intelligence.
- A Colonist.io integration/evaluation loop.
- Evidence that the current learned model is better than strong public or human bots.

## Current Champion

Current champion checkpoint:

```text
runs/self_play/champions/current_best_s4806_iter0002.pt
```

Known gate evidence:

```text
heuristic baseline aggregate: 18 / 64
value baseline aggregate:     11 / 64
```

Interpretation:

- This is not a strong Catan bot.
- It is only the best internal checkpoint from the current prototype ladder.
- The model is playable and can make legal decisions, but it is not close to a credible Colonist #1 agent.

## Current Model Architecture

The main learned model is `TorchPPOPolicy` in:

```text
src/catan_zero/rl/torch_ppo.py
```

Supported architectures:

```text
flat
candidate
```

The current champion is:

```text
architecture: candidate
hidden_size: 1024
observation_size: 1002
action_size: 607
context_action_feature_size: 12
```

The candidate architecture:

- Encodes the public observation with an MLP.
- Encodes static and contextual action features.
- Scores legal action candidates.
- Has a scalar value critic.
- Has a Q-value head for chosen-action and legal-action Q estimates.

This is directionally better than a flat action softmax, but it is still not the model we actually want for top-tier Catan.

Main weakness:

```text
The model is still mostly a flat MLP over engineered observations, not a board graph plus public event-history model.
```

## Environment / Game API

Important files:

```text
src/catan_zero/rl/multiagent_env.py
src/catan_zero/rl/gym_env.py
src/catan_zero/rl/action_features.py
src/catan_zero/rl/self_play.py
```

The environment can run full four-player games through Catanatron-derived mechanics.

Implemented/partially implemented:

- Four-player game loop.
- Legal action masks.
- Structured legal actions.
- Player trade offer/response plumbing.
- Action context features for candidate scoring.
- Evaluation across seats.

Known concern:

```text
Catanatron is useful as a rules/simulation backend, but the final system should not be just "Catanatron plus PPO."
```

We still need a more explicit CatanZero state/action contract and a better match to Colonist behavior.

## Training Algorithm So Far

Current training pieces:

```text
tools/train_ppo.py
tools/evaluate_self_play.py
tools/league_orchestrator.py
tools/reanalysis_orchestrator.py
tools/generate_reanalysis.py
```

Implemented algorithmic ideas:

- Behavior cloning from heuristic/value/search teacher traces.
- PPO self-play.
- One-learner-seat vs fixed opponents.
- All-seat shared-policy self-play.
- Strong mixed opponents.
- Adaptive league style sampling.
- Q-value auxiliary critic.
- Small Q-advantage mixing toward VRPO-like updates.
- KL penalty toward old rollout policy.
- Anchor imitation from teacher games.
- DAgger-style teacher labels on learner-visited states.
- Value-shaped reward experiments.
- Search/reanalysis distillation.

This is a reasonable prototype, but not yet a high-end game-AI system.

## Research Pattern We Are Trying To Follow

The intended final pattern is closer to AlphaGo/AlphaStar/ReBeL/Student-of-Games than plain PPO:

```text
search/value teacher data
    -> structured policy pretraining
    -> regularized self-play
    -> population/league training
    -> belief-aware search
    -> reanalysis/distillation
    -> strict gated promotion
```

Current system only implements fragments of that:

- Search teacher: partial.
- Reanalysis buffer: now implemented.
- PPO/VRPO-like training: partial.
- Population league: shallow.
- Belief-aware search: missing.
- Graph/history policy: missing.
- Full trade intelligence: missing.

## Remote Training Runs

Old long PPO branches:

```text
runs/self_play/league_v2/manifest.json
```

Branches:

```text
s6001_league_vrpo
s6002_ema_anchor
s6003_allseat_long
```

Result:

```text
All three were killed on boxes with no checkpoints.
```

Interpretation:

- Long raw PPO branches were too expensive/fragile in the current setup.
- The dense data and training memory behavior are a problem.
- Blindly launching more of these is not a good path.

Reanalysis branches:

```text
runs/self_play/reanalysis_v1/manifest.json
```

Branches:

```text
s7001_reanalysis on bx_332dnnhy
s7002_reanalysis on bx_xw7vm8b7
```

Generated search data:

```text
s7001_reanalysis.jsonl: 4246 samples, 326 MB
s7002_reanalysis.jsonl: 5704 samples, 436 MB
```

Problem:

```text
Those JSONL files were generated before sparse context storage was implemented, so they are huge.
```

Training from those dense files was started with:

```text
--reanalysis-max-samples 2048
--reanalysis-epochs 2
--reanalysis-value-coef 0.35
--reanalysis-score-coef 0.05
```

Observed reanalysis update:

```text
s7001:
  samples: 2048
  policy_loss: 1.3288722038269043
  value_loss: 0.168742835521698
  score_loss: 0.8361881375312805

s7002:
  samples: 2048
  policy_loss: 1.3832182884216309
  value_loss: 0.17504757642745972
  score_loss: 1.045001745223999
```

Then PPO started.

At user request, active remote training was stopped. No new promoted checkpoint resulted from these branches.

## Work Completed In The Latest Iterations

### 1. Reanalysis data path

Added:

```text
src/catan_zero/rl/reanalysis.py
tools/generate_reanalysis.py
```

This allows expensive search/value-rollout teacher targets to be saved as JSONL and reused for training.

Why it matters:

```text
Search can be run once on CPU boxes, then the policy can train repeatedly from the generated targets.
```

### 2. Training from reanalysis

Modified:

```text
tools/train_ppo.py
```

Added flags:

```text
--reanalysis-input
--reanalysis-max-samples
--reanalysis-epochs
--reanalysis-value-coef
--reanalysis-score-coef
--reanalysis-checkpoint
```

This allows:

```text
search data -> supervised distillation -> optional checkpoint -> PPO
```

### 3. Sparse reanalysis records

Modified:

```text
src/catan_zero/rl/reanalysis.py
```

Old behavior:

```text
Every sample stored the full action_context_features table for all 607 actions.
```

New behavior:

```text
Only stores context rows for valid legal actions.
Loader reconstructs dense context before training.
```

This is important because dense JSONL became hundreds of MB for only a few thousand samples.

### 4. Reanalysis orchestrator

Added:

```text
tools/reanalysis_orchestrator.py
```

It can:

```text
poll remote reanalysis branches
pull ready checkpoints/reports/jsonl
gate ready checkpoints
```

Important command:

```bash
cd <verified-local-checkout>
.venv/bin/python tools/reanalysis_orchestrator.py gate-ready --manifest runs/self_play/reanalysis_v1/manifest.json --promote-if-better
```

### 5. Tests added/updated

Important tests:

```text
tests/test_reanalysis_orchestrator.py
tests/test_self_play.py::test_search_reanalysis_targets_round_trip_and_train
tests/test_self_play.py::test_reanalysis_records_store_only_valid_action_context
tests/test_self_play.py::test_train_ppo_writes_reanalysis_checkpoint
```

Recent passing checks:

```text
12 passed
7 passed
6 passed
7 passed
```

These verify the mechanics, not model strength.

## How Good Is The Algorithm Right Now?

Blunt answer:

```text
Not good enough.
```

It is good enough to:

- Play legal games.
- Train policies end to end.
- Generate search-teacher data.
- Run PPO and imitation updates.
- Compare candidates against baselines.
- Reject candidates that regress.

It is not good enough to:

- Claim strong Catan performance.
- Beat high-level humans.
- Rank #1 on Colonist.
- Reliably beat strong value/search baselines.
- Handle trading at expert level.

The best current checkpoint is only an internal prototype champion.

## Why Progress Feels Slow

Main reasons:

1. We are training a weak architecture.

   Current model is still an MLP/candidate scorer. It should become a graph/history model.

2. Dense reanalysis records caused huge data files.

   The first reanalysis files were 326 MB and 436 MB for only a few thousand samples. This made training slow and memory-heavy.

3. Long PPO runs were killed before checkpointing.

   Three `s600*` branches died with no checkpoint. That means the current run recipe is fragile.

4. Evaluation is still small.

   The gate is useful, but 32-game legs are only a development gate. Final claims need many more games.

5. Catan-specific intelligence is incomplete.

   Trading, belief inference, robber politics, and opening placement need stronger dedicated modeling.

## Most Important Technical Problems

### Problem 1: Architecture is not strong enough

Current:

```text
observation vector -> MLP -> candidate action scorer
```

Needed:

```text
board graph encoder
public event-history encoder
legal action candidate scorer
four-player value vector
Q(s,a) head
belief heads for hidden resources/dev cards
trade accept/counter heads
```

### Problem 2: Search is not belief-aware

Current search/value teachers mostly use Catanatron value estimates and rollouts.

Needed:

```text
sample hidden states from legal public history
run action search across belief particles
score with vector value
distill root policies
```

### Problem 3: League is too shallow

Current "league" is mostly opponent sampling and snapshots.

Needed:

```text
main learner
historical champions
fresh exploiters
opening exploiters
trade exploiters
robber/blocking exploiters
endgame exploiters
population payoff matrix
best-response gate
```

### Problem 4: Trading is not first-class enough

Current trade support exists but is not an expert subsystem.

Needed:

```text
offer generator
acceptance model
counteroffer model
trade EV model
opponent benefit/risk model
anti-spam protocol
human-compatible strategy
```

### Problem 5: Current remote training infra is inefficient

Issues:

- Repeated venv setup is slow.
- `box ssh` wrappers sometimes stay attached after `setsid`.
- Remote jobs need better lifecycle management.
- Dense JSONL caused memory pressure.

## Recommended Next Technical Direction

Do not keep launching random PPO branches.

Next sequence should be:

1. Regenerate reanalysis with sparse context format.

   Use `tools/generate_reanalysis.py` after the sparse JSONL patch.

2. Train a reanalysis-only candidate and gate it.

   Use `--reanalysis-checkpoint` and `--iterations 0`.

3. If reanalysis-only improves, use it as PPO init.

   Then run short PPO with checkpoints every 2 or 5 iterations.

4. Build graph/history model.

   This is the real architectural step. Without it, the model will likely plateau.

5. Add belief targets.

   Even simple supervised hidden-resource reconstruction from simulator state would be a major step.

6. Add opening-specialized evaluation.

   A lot of Catan strength comes from initial settlement/road choices.

## Concrete Commands For Next Run

Poll current reanalysis manifest:

```bash
cd <verified-local-checkout>
.venv/bin/python tools/reanalysis_orchestrator.py poll --manifest runs/self_play/reanalysis_v1/manifest.json
```

Pull/gate ready reanalysis checkpoints:

```bash
cd <verified-local-checkout>
.venv/bin/python tools/reanalysis_orchestrator.py gate-ready --manifest runs/self_play/reanalysis_v1/manifest.json --promote-if-better
```

Generate sparse reanalysis locally for smoke:

```bash
cd <verified-local-checkout>
.venv/bin/python tools/generate_reanalysis.py \
  --output runs/self_play/sparse_reanalysis_smoke.jsonl \
  --seed 9001 \
  --games 1 \
  --vps-to-win 3 \
  --max-decisions 32 \
  --candidate-limit 2 \
  --presearch-candidate-limit 2 \
  --rollout-decisions 1
```

Train reanalysis-only candidate:

```bash
cd <verified-local-checkout>
.venv/bin/python tools/train_ppo.py \
  --seed 9002 \
  --vps-to-win 3 \
  --max-decisions 64 \
  --init-checkpoint runs/self_play/champions/current_best_s4806_iter0002.pt \
  --teacher value \
  --warmup-games 0 \
  --iterations 0 \
  --reanalysis-input runs/self_play/sparse_reanalysis_smoke.jsonl \
  --reanalysis-max-samples 512 \
  --reanalysis-epochs 2 \
  --reanalysis-value-coef 0.35 \
  --reanalysis-score-coef 0.05 \
  --reanalysis-checkpoint runs/self_play/sparse_reanalysis_only.pt \
  --checkpoint runs/self_play/sparse_reanalysis_only.final.pt \
  --report runs/self_play/sparse_reanalysis_only.json \
  --eval-games 0 \
  --eval-value-games 0
```

## Questions For Senior Engineer

1. Should we keep Catanatron as the simulation backend, or should we fork/replace the engine now?

   My recommendation: keep it as backend for now, but build a CatanZero state/action abstraction above it.

2. Should we prioritize graph/history model before more PPO?

   My recommendation: yes. The current MLP/candidate model is likely too weak.

3. What is the right first graph/history architecture?

   Candidate answer:

   ```text
   typed board tokens + player tokens + global tokens
   cross-attention to event-history transformer
   candidate-action scorer
   scalar/vector value heads
   belief heads
   ```

4. Should we invest in belief modeling before search?

   My recommendation: yes. Catan is hidden-information; naive search can cheat or learn bad strategy.

5. How much compute should we spend before architecture upgrade?

   My recommendation: very little. Use current system only to validate data/eval loops, not as final algorithm.

6. Should reanalysis use JSONL or a binary format?

   My recommendation: move to compressed shards or Arrow/Parquet/NPZ for large runs. JSONL is okay for debugging but too heavy at scale.

7. What should the promotion gate be?

   Current gate:

   ```text
   no heuristic regression
   no value regression
   strict aggregate improvement
   ```

   This is fine for development, but final claims need thousands of games.

8. What is the minimum credible Colonist-readiness milestone?

   Suggested answer:

   ```text
   beats current champion
   beats heuristic
   beats value baseline
   beats search baseline at fixed latency
   survives fresh exploiters
   handles trades legally and profitably
   ```

## Current Artifacts

Champion:

```text
runs/self_play/champions/current_best_s4806_iter0002.pt
```

Old killed league manifest:

```text
runs/self_play/league_v2/manifest.json
```

Reanalysis manifest:

```text
runs/self_play/reanalysis_v1/manifest.json
```

New reanalysis code:

```text
src/catan_zero/rl/reanalysis.py
tools/generate_reanalysis.py
tools/reanalysis_orchestrator.py
```

Modified trainer:

```text
tools/train_ppo.py
```

Tests:

```text
tests/test_reanalysis_orchestrator.py
tests/test_self_play.py
```

## Current Resource State

At the time this handoff was written:

- Active remote training jobs were stopped at user request.
- No local `box ssh` / `box scp` wrappers should remain.
- No new `s700*` or `s7101*` checkpoint was promoted.

## Bottom Line

The project has a real playable Catan RL prototype and a working candidate promotion loop.

But the current algorithm is not yet good enough.

The biggest next step is not more random PPO. It is:

```text
sparse search reanalysis
    -> reanalysis-only candidate gate
    -> graph/history model
    -> belief heads
    -> population/exploiter league
    -> belief-aware search
```

If the goal is Colonist #1, the current implementation should be treated as infrastructure and baseline scaffolding, not the final model.
