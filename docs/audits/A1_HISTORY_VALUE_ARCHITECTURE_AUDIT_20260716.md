# A1 history/value architecture audit

**Date:** 2026-07-16
**Scope:** 2p no-trade learner input, value path, and coherent-public search
boundary
**Priority:** root-level learning-signal failures only

## Executive finding

The learner was missing important state in two different ways:

1. the v5 producer can emit typed public-history targets, but the production
   dense Transformer discarded `event_target_ids`; and
2. coherent-public n128 searches one sanitized hidden world for the whole root
   turn, then uses a single sampled opponent development-card allocation in the
   turn-boundary value.

The first defect is repaired in code by an opt-in, zero-output event-to-entity
join. The second is an operator/value experiment, not a safe silent default
change: it needs a fixed-root belief-variance panel and a boundary-particle
arm before the adopted search contract changes.

## 1. Event targets were produced but not consumed

The v5 history schema stores each retained event's typed target in
`event_target_ids`:

```text
column 0: hex
column 1: vertex
column 2: edge
column 3: player
```

The current dense Transformer path excluded that tensor in
`EntityGraphPolicy.forward_legal_np`. It was transferred only for relational
trunks or the topology residual. The history side path then pooled only
`event_encoder(event_tokens)`. Consequently, a historical road, settlement,
city, robber move, or victim event had no direct join to the post-trunk entity
that it changed. A normalized scalar target id in event-token slot 14 was the
only indirect spatial signal.

### Implemented repair

`EntityGraphConfig.meaningful_public_history_target_gather` is an append-only,
default-off flag. When enabled:

- `event_target_ids` is validated and transferred for the dense Transformer;
- each event gathers its typed post-trunk hex/vertex/edge/player token;
- valid targets are mean-pooled;
- a bias-free, zero-initialized projection adds the target representation to
  that event before masked/ordered history pooling.

The zero projection makes activation exactly function preserving for an
existing checkpoint. A scratch model can commission the path without changing
the stored feature schema.

Primary implementation:

- `src/catan_zero/rl/entity_token_policy.py`
- `tests/test_meaningful_history_target_gather.py`

## 2. History v2 is still using the v1 pooling contract

The current scratch science contract binds:

```text
history schema: meaningful_public_history_2p_no_trade_v2
history cap: 64
pooling: masked_mean_v1
```

`ordered_attention_v2` is already implemented, checkpoint-compatible, and
tested. It retains the established masked-mean branch and adds a separately
zero-gated ordered branch. Therefore switching the declared pooling contract
is safe at activation.

Masked mean is not completely blind to time because event tokens carry age and
turn-key fields, but it cannot learn content-to-content temporal relations as
directly as the ordered branch. The mismatch is especially important because
65.0% of audited rows saturated the old 32-event cap.

Recommendation for the scratch architecture: bind
`ordered_attention_v2`, then commission the ordered gate from scratch rather
than claiming that schema v2 alone solved order.

## 3. The coherent tree reuses one hidden-world sample at every boundary

`GumbelChanceMCTS._search_coherent_public_belief` calls
`determinize_for_player` exactly once and searches that sanitized game with the
full n128 budget.

Native traversal forcibly terminates when control leaves the root actor's turn.
At that boundary it expands the node and returns the neural prior value.
Evaluation is actor-centric, so the next actor sees its own exact private hand
from the sampled determinization; the evaluator then sign-flips that result to
the root player's perspective.

This is not authoritative-hidden-state leakage. The allocation is sampled from
public constraints. It is nevertheless a major variance source:

- the root player does not know the opponent's face-down dev identities;
- all 128 simulations reuse one sampled allocation;
- increasing n64 to n128 does not reduce this belief-boundary noise;
- a stronger value head can still learn the sampled-world function while the
  search target remains unstable across belief seeds.

### Required commissioning experiment

On a fixed, phase-stratified root set:

1. sample 16–32 `determinize_for_player` worlds per root;
2. evaluate the exact turn-boundary value in each world;
3. report mean, standard deviation, sign-flip rate, and top-action changes;
4. stratify by opponent face-down dev count, public-history length, phase, and
   legal width.

Then compare:

- current one-world coherent n128;
- n128 with 2 or 4 boundary-value particles;
- the same total evaluator-call budget allocated to more tree simulations.

Adopt boundary particles only if they improve fixed-root regret/target
stability and paired playing strength at acceptable cost.

Relevant code:

- `src/catan_zero/search/gumbel_chance_mcts.py`
- `native/gumbel_mcts_rs/src/lib.rs`
- `src/catan_zero/search/neural_rust_mcts.py`

## 4. The value network is the long-horizon planner

Coherent-public traversal stops at the end of the current turn. Therefore n128
search optimizes within-turn tactics and delegates every opponent response and
later turn to the value network. This makes value calibration load-bearing,
not auxiliary.

The current scratch recipe improves older learner defects:

- terminal outcomes only (`value_target_lambda=1`);
- per-game value weighting;
- outcome-balanced sampling;
- value phase weights separated from policy phase weights;
- `ROLL` and `END_TURN` restored to value weight 1;
- scalar training uses the deployed tanh readout.

The remaining architectural risk is shared-representation interference.
Existing gradient evidence shows policy/value trunk cosine near zero with many
negative batches. The current recipe scales value-to-trunk gradients to 0.5,
but still routes them through the entire policy trunk and action encoder.

Next causal arms, all independently initialized from identical bytes/data:

1. value trunk scale 0.5 control;
2. value trunk scale 0.25;
3. split final one Transformer block;
4. split final two Transformer blocks.

Compare phase-sliced calibration, search uplift over raw policy, parent policy
KL, and external paired strength. Parameter-count matching matters because a
split tower adds capacity.

## 5. Action locality remains an architecture ceiling, not the proven proximal
cause

The dense Transformer has no structural board relation mask. Vertex and edge
tokens are treated as sets, while legal actions contain absolute catalog ids.
With `action_target_gather=False`, policy actions cannot directly read the
post-trunk board token they modify. The existing gather-only short-dose
experiment tied its control, so this did not explain the prior chained learner
collapse. It remains a strong from-scratch architecture question.

The correct commissioning order is:

1. stabilize scratch dose/value learning;
2. enable event-target gather;
3. compare action-target gather;
4. only then test a relational trunk or cross-attention bundle.

Do not infer that a tied inherited-checkpoint short dose proves action locality
is useless to a new network.

## Ranked execution order

1. **Now:** use v5 rule state/history rows and stop claiming target identity is
   consumed unless event-target gather is enabled.
2. **Now:** bind `ordered_attention_v2` for the scratch architecture.
3. **Before another sim-budget increase:** measure boundary belief-value
   variance; n128/n256 cannot cure a one-world boundary estimator.
4. **First learner architecture sweep:** value trunk 0.25 versus one/two-layer
   split towers.
5. **After the learner is stable:** action-target gather and relational
   action-local architecture.
