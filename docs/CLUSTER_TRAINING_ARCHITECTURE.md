# CatanZero Cluster Training Architecture

This is the target design for using a real GPU cluster well. The current remote
fleet runs many independent PPO jobs, which is useful for search over recipes but
is not the final training system. The final system should centralize learning,
batch inference on GPUs, and use CPUs for simulation/search.

## Goal

Train a four-player Catan agent that improves against the strict gate without
regressing against:

- `jsettlers_lite`
- `value_rollout`
- `heuristic`
- `catanatron_value`
- historical champions and fresh exploiters

The cluster should optimize for promoted checkpoints, not for raw games played.

## Machine Roles

### CPU Actor Workers

Use CPUs for environment-heavy work:

- Catan environment stepping
- legal action enumeration
- structured trade generation
- self-play trajectory collection
- scripted opponent games
- search rollouts when the rollout policy is mostly environment-bound

Actors should not train. They should:

1. Pull the latest policy version or assigned branch policy.
2. Run games against an assigned opponent mixture.
3. Emit compact trajectory records.
4. Include full simulator truth only in teacher-only fields.
5. Never expose hidden truth to actor observations.

Target ratio to start:

```text
1 H100 learner : 128-512 CPU actor processes
```

Increase actors until the learner/inference servers are saturated.

### GPU Inference Servers

Use GPUs for batched forward passes:

- policy logits over legal candidates
- value and Q heads
- belief heads
- trade heads

Actors should send batches to an inference service instead of each actor owning a
GPU process. This avoids hundreds of tiny GPU calls.

Start with:

```text
1 H100 inference server per 256-1024 active actors
batch target: 512-8192 decisions
max wait: 5-25 ms
precision: bf16
```

If the policy is still small, one H100 can handle both inference and learning.
Once the graph/history transformer grows, split learner and inference GPUs.

### GPU Learner

The learner owns backprop. It samples from shared buffers and publishes policy
versions.

Initial target:

```text
global batch: 32k-256k decision samples
minibatch: 2k-16k
precision: bf16 autocast
optimizer: AdamW or fused AdamW
compile: torch.compile after shapes stabilize
```

The learner should train one model with multiple heads:

```text
policy over legal candidates
scalar/four-seat value
chosen-action Q
hidden-resource belief
dev-card belief
trade acceptance/counter heads
```

### Search/Reanalysis Workers

Use CPUs plus batched GPU inference:

- sample difficult states from failure buffer
- run value-rollout/root search
- create improved policy targets
- create Q/value targets
- add records to `reanalysis_buffer`

Search should focus on:

- opening placements
- robber decisions
- discards
- development-card timing
- trades
- endgame build-order choices
- states where the policy lost to `jsettlers_lite` or `value_rollout`

### Evaluator Workers

Evaluators are separate from training. They run:

- triage gates
- strict promotion gates
- escalated gates
- opening evals
- fresh exploiter evals

Evaluators write reports and update the population payoff ledger. They do not
promote unless every configured gate says promotion is allowed.

## Data Flow

```text
actors/search/evaluators
        |
        v
trajectory + failure + reanalysis buffers
        |
        v
GPU learner
        |
        v
candidate checkpoints
        |
        v
strict evaluator fleet
        |
        +--> reject -> failure buffer
        |
        +--> promote -> champion registry
```

## Buffers

### Rollout Buffer

Normal on-policy or near-on-policy samples:

```text
policy_version
game_seed
seat
public observation
private legal observation
legal action context
chosen action
logprob
value
q_chosen
reward/return
done
opponent policy ids
```

Use for PPO/VRPO updates.

### Failure Buffer

States from rejected candidates and lost games:

```text
candidate_policy
champion_policy
opponent_id
gate_profile
state_before_bad_action
chosen_action
teacher/search action
event history
result
failure tag
```

Oversample this buffer. Current dominant failure tags are:

```text
opponent_regression:jsettlers_lite
opponent_regression:value_rollout
opponent_regression:heuristic
```

### Reanalysis Buffer

Search/teacher-improved labels:

```text
state
legal candidates
teacher distribution
search scores
search value
belief particles if available
source failure tag
```

Use for supervised distillation before and during RL.

### Belief Buffer

Supervised hidden-state labels from simulator truth:

```text
public history
acting player private observation
opponent resource counts by type
opponent dev-card counts/types
can-build probabilities
```

This does not require human data; self-play simulator truth is enough.

## Training Losses

The learner should optimize:

```text
L =
  PPO_or_VRPO_policy_loss
+ value_loss
+ chosen_Q_loss
+ expected_SARSA_Q_loss
+ entropy_bonus
+ KL_to_champion_or_human_anchor
+ EMA_policy_KL
+ reanalysis_policy_distillation
+ teacher_score_margin_loss
+ belief_resource_loss
+ belief_dev_card_loss
+ trade_acceptance_loss
```

Near-term priority:

1. PPO/VRPO
2. Q/Expected-SARSA
3. reanalysis distillation
4. belief heads
5. trade heads

## GPU Split

For a small cluster:

```text
1 H100 learner
1 H100 inference
many CPU actors
CPU evaluator/search workers
```

For a medium cluster:

```text
2-4 H100 learners for separate branches
2-4 H100 inference servers
512-2048 CPU actor processes
64-256 CPU search workers
dedicated evaluator workers
```

For a large cluster:

```text
main learner: graph/history champion branch
branch learners: recipe/exploiter/ablation branches
inference pool: batched policy service
search pool: failure reanalysis and opening/endgame search
eval pool: strict gates and promotion tournaments
```

Do not give every actor a GPU. That wastes the cluster.

## What H100s Should Do

Good H100 work:

- graph/history transformer training
- large-batch PPO/VRPO updates
- batched policy/value inference
- belief and trade auxiliary heads
- search-target distillation
- checkpoint ensemble distillation

Bad H100 work:

- one Python env per GPU
- tiny unbatched action decisions
- Catanatron stepping
- legal action enumeration
- file parsing
- remote orchestration

## Custom Kernels

Do not start with custom CUDA kernels. First use:

- bf16 autocast
- `torch.compile`
- batched inference
- fused optimizer if available
- static-ish tensor layouts for legal action candidates
- profiling

Only write kernels after profiling proves the GPU-side model is the bottleneck.
For Catan, Python simulation and legal action generation are more likely to
bottleneck first.

## Promotion Ladder

Use three levels:

```text
smoke gate:
  2-4 games/opponent
  fast rejection only

strict gate:
  16-64 games/opponent
  no opponent regression allowed

champion gate:
  256-1000+ games/opponent
  seat-balanced, seed-balanced, confidence intervals
```

Never promote from training loss. Promote from gate evidence only.

## First Cluster Implementation Target

Build this minimal distributed loop:

```text
ActorPool -> rollout_queue -> Learner -> checkpoint_queue -> Evaluator
                ^              |
                |              v
           inference_server <- policy_registry
```

Keep the current remote independent-PPO system as branch exploration, but use the
central learner as the main path once it exists.

