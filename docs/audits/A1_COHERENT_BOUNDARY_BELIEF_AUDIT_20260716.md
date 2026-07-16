# A1 coherent-public boundary-belief audit

Date: 2026-07-16

Scope: static, implementation-level audit of the two-player
`coherent_public_belief_search` teacher. This report does not claim an empirical
strength win for a replacement operator. It identifies the exact remaining
information-set error, explains why n128/n256 cannot remove it, and specifies a
matched-compute experiment and implementation boundary.

## Executive finding

The current operator is coherent inside the root player's current turn, but its
continuation value is still a one-particle estimate at the opponent-turn
boundary.

The Python orchestration:

1. calls `Game.determinize_for_player(root_color, seed)` exactly once;
2. spends the complete n128/n256 budget in that sampled world;
3. stops traversal when control leaves the root actor;
4. evaluates the boundary from the new actor's perspective.

The Rust hot loop implements the same boundary.

For two-player no-trade Catan, resource conservation makes the sole opponent's
resource composition exact from the public bank plus the root actor's own hand.
The material remaining uncertainty is primarily the opponent's face-down
development-card identities, hidden victory points, and the posterior over
those identities. The current determinizer samples that allocation once from a
conservation prior and every simulation in the tree reuses it.

This makes the value network more load-bearing than the nominal simulation
count suggests. Search is only explicit until the end of the current turn. At
the first opponent state, the backed-up value is conditional on one sampled
opponent hand.

## Static proof in the code

### One root determinization

`src/catan_zero/search/gumbel_chance_mcts.py`:

- `GumbelChanceMCTS._search_coherent_public_belief`
- one call to `game.determinize_for_player(...)`
- one call to `_search_single_world(sampled, ...)`

There is no loop over determinizations in this mode. The existing focused test
`tests/test_information_set_mcts.py::
test_coherent_public_belief_uses_one_sanitized_full_budget_tree` explicitly
asserts that each coherent search records exactly one determinization seed.

The older `information_set_search` path does loop over particles, but fragments
the total budget into separate trees. Coherent mode deliberately rejects that
combination.

### One sampled world in the native arena

`src/catan_zero/search/native_gumbel_mcts.py` inherits the Python coherent
orchestration. The sampled game is then passed to
`catanatron_rs.gumbel_search`.

`native/gumbel_mcts_rs/src/lib.rs`:

- `GumbelMctsEngine::search` constructs one arena from the supplied game;
- every simulation calls `simulate` on that arena;
- `is_root_turn_boundary` returns true when the current actor changes or the
  turn counter changes;
- the boundary node is expanded once and `prior_value` is reused thereafter.

Increasing the simulation budget therefore revisits/refines one arena. It does
not create new opponent-hand particles.

### Why the boundary value sees the sampled opponent hand

`src/catan_zero/search/neural_rust_mcts.py` evaluates every leaf from the
side-to-act perspective:

- `acting_color = game.current_color()`;
- public-observation masking preserves the acting player's own exact private
  hand;
- the scalar value is negated when `acting_color != root_color`.

At an opponent-turn boundary, the opponent is the acting player. Their exact
development-card identities in the evaluator input are therefore the identities
from the one root determinization. This is correct for any one possible world,
but a public-belief root must integrate over all worlds compatible with the root
actor's information.

### What the determinizer samples

`native/catanatron-rs/src/lib.rs::Game::determinize_for_player`:

- preserves the observer's hand;
- reconstructs opponent resources from public conservation;
- reconstructs the unknown development-card pool from the base deck, public
  plays, and the observer's own cards;
- samples opponent face-down cards and the remaining deck;
- repairs hidden victory points and `owned_at_start`;
- requires the observer to be the current player.

The sampling is independent of authoritative hidden identities, which closes
the original hidden-truth leak. It is still a conservation prior, not a
history- and policy-conditioned posterior.

`public_belief_development_draws` independently sanitizes BUY-DEVELOPMENT-CARD
children and is not the main bug described here. The uncovered case is the
continuation value at ordinary opponent-turn boundaries and the residual
one-particle conditional value inside conditioned draw children.

## Why this is more than variance

Let `H` be the unknown opponent development-card allocation and let
`Q_a(H)` be the current-turn search value of root action `a` when the boundary
continuation is evaluated in world `H`.

The current row target is approximately:

```text
pi_H = Improve(prior, Q(H))
```

where `Improve` includes Sequential Halving, completed-Q, min-max
normalization, visit-dependent sigma scaling, and softmax.

A public-belief target should instead be based on an integrated continuation:

```text
Qbar_a = E[Q_a(H) | root information, public branch]
pi_public = Improve(prior, Qbar)
```

In general:

```text
E[Improve(prior, Q(H))] != Improve(prior, E[Q(H)])
```

because min-max normalization, candidate elimination, and softmax are nonlinear.
The current operator is therefore not merely an unbiased target with extra
noise. It can rank and sharpen actions according to one imaginary opponent
development-card allocation.

This becomes more dangerous as search gets stronger:

- more visits reduce within-world tree noise;
- they do not reduce cross-world boundary error;
- with `sigma_reference_visits=None`, the completed-Q logit scale grows with
  realized maximum visits:

```text
(c_visit + realized_max_visits) * c_scale
```

- the teacher can become more confident in the action preferred by the sampled
  world.

This is a concrete mechanism by which n256 can look more stable internally yet
distill a worse general policy. It is not claimed to be the sole explanation of
the historical n256 specialist/generalist split, but it is an unresolved
operator error that extra simulations cannot cure.

## Catan-specific impact

The error is not uniform over positions.

Highest-risk roots:

- opponent has one or more face-down development cards;
- hidden victory points can materially change race urgency;
- Knight/Monopoly/Road Building likelihood changes the next-turn response;
- the root action changes exposure to robber retaliation or road blocking;
- late game, where one sampled hidden VP can strongly alter the value margin;
- roots whose top actions have small true value margins.

Lower-risk roots:

- no opponent face-down development cards;
- opening positions before development-card purchases;
- actions with large, stable value margins;
- forced transitions carrying no policy target.

The resource-hand portion is less problematic in two-player no-trade than the
generic phrase "one hidden hand" suggests. Given public bank composition and
the root actor's exact hand, the sole opponent's resource composition is fixed
by conservation. This report should not be used to justify an unnecessary
resource-particle explosion for the current track.

## Matched-compute replacement

The preferred near-term operator is one coherent current-turn tree with a
particle-averaged value only at the opponent boundary.

For a boundary state `b`:

```text
V_boundary(b) =
    mean_k [
        -V_opponent(b, sampled_hidden_world_k)
    ]
```

The averaged value is backed up once through the existing tree. Sequential
Halving and completed-Q then operate on integrated boundary values rather than
on separate per-world improved policies.

This preserves the main benefit of coherent search:

- one shared within-turn tree;
- no fragmented n32 subtrees;
- one policy-improvement operator after belief integration;
- particles spent only where private opponent information becomes visible to
  the continuation evaluator.

### Required native primitive

Do not reuse `determinize_for_player` unchanged at the boundary. It rejects a
non-current observer, while the required observer is the original root actor.

Add a separate, explicitly named primitive such as:

```text
determinize_from_observer_information(observer, seed)
```

It must:

1. permit `observer != current_color`;
2. preserve the observer's current private hand after the searched branch;
3. reconstruct every other hidden hand/deck from public conservation;
4. redact hidden action-record payloads;
5. regenerate the current actor's legal actions for the sampled world;
6. not compare those legal actions with the source world's hidden-dependent
   legal actions;
7. remain independent of authoritative hidden identities.

At a boundary after the root actor ends their turn, this samples possible
opponent development-card hands while preserving exactly what the root actor
knows.

### Required evaluator boundary

For `K > 1`, the searcher should call a value-only batch boundary:

```text
evaluate_boundary_values(
    sampled_games,
    root_color,
) -> [root-perspective values]
```

Do not fabricate or average policy priors across particles with different
hidden-dependent legal sets. No opponent action is traversed at this boundary;
only the continuation value is consumed.

The `K=1` path must retain the current byte-for-byte implementation so the new
code is safely commissioning-only until selected.

### Budget matching

Nominal simulations are not a valid matched-compute unit once one boundary visit
can request `K` value evaluations. Match on:

1. actual neural rows evaluated;
2. actual native simulations;
3. wall time;
4. optional GPU forward time.

The first implementation must report both `simulations_used` and
`neural_evaluations_used`.

Do not assume that `n64/K2` or `n32/K4` is automatically matched to n128/K1.
Chance enumeration, node reuse, and root width change the evaluator-call ratio.
Calibrate the nominal simulation count on a fixed root set to keep neural rows
within +/-5% of control.

## Decisive fixed-root experiment

Use 200-500 authenticated replay roots stratified by:

- phase;
- legal width;
- opponent public development-card count;
- score margin;
- policy entropy;
- top-two prior gap;
- whether BUY-DEVELOPMENT-CARD is legal;
- whether the line can end the turn quickly.

Arms:

1. current coherent n128, K1;
2. coherent boundary K2 at matched neural rows;
3. coherent boundary K4 at matched neural rows;
4. legacy root PIMC P4 with total matched compute;
5. adjudicator: boundary K16, multiple independent tree seeds, larger budget.

For each root, separately estimate:

- fixed belief seed, varied tree/chance seed;
- fixed tree/chance seed, varied belief seed.

This decomposes within-tree search noise from boundary-belief noise.

Primary metrics:

- policy JSD to adjudicator;
- top-one agreement;
- regret under adjudicator completed-Q;
- Q-rank stability;
- target entropy;
- top-one flip rate across belief seeds;
- wall time and neural rows.

Required slices:

- opponent dev-card count 0 versus >0;
- early versus late game;
- narrow versus wide root;
- small versus large adjudicator Q margin.

Graduation criterion:

- K2 or K4 improves adjudicator agreement/regret at matched neural rows;
- benefit concentrates in the predicted development-card/late-game slices;
- no material regression when opponent dev-card count is zero;
- candidate then wins a matched seat-swapped playing panel.

## Posterior limitation after K-particle repair

Uniform conservation particles solve one-sample reuse, not belief quality.

The current determinizer does not condition development-card identities on
public action history or an opponent policy model. A later belief module should
weight particles by:

- public development-card purchase/play timing;
- publicly observed actions;
- rule constraints;
- a frozen opponent-policy likelihood model;
- meaningful public history.

The immediate experiment should first establish whether particle averaging
helps under the existing conservation belief. Do not bundle a new learned
posterior into the same causal arm.

## Alternative medium-term architecture

A public-observer continuation head could replace particles:

```text
V_observer(public boundary, root observer private information)
```

It would be trained on post-END-TURN states from the previous actor's
perspective with terminal outcomes, rather than evaluating the next actor with a
sampled private hand.

This is potentially much cheaper at inference, but it is a new value target and
perspective contract. The current value head is trained/evaluated from the
side-to-act perspective, so simply calling it with the previous actor would be
off-distribution and is not a valid shortcut.

The existing per-action Q head is also not a drop-in solution: production keeps
its Q loss disabled because stored teacher-score semantics are not yet a
verified terminal-return action-value contract.

## Execution order

1. Add the fixed-root seed-decomposition diagnostic.
2. Measure cross-belief target variance for n64/n128/n256.
3. Add the non-current-observer native determinization primitive.
4. Add value-only K2/K4 boundary batching, default K1.
5. Report neural-evaluation counts and calibrate matched nominal budgets.
6. Run fixed-root adjudication.
7. Run a small matched playing panel.
8. Only after a win, bind the selected particle count into teacher identity,
   science contract, shard provenance, and promotion gates.

## Production recommendation

Keep current coherent n128 as the baseline while this experiment is
commissioned. Do not launch global n256 as a presumed stronger teacher.

The highest-value search question is no longer "128 or 256 simulations?" It is:

> At the same number of neural evaluations, is compute better spent refining
> one imagined opponent development-card world or averaging the continuation
> value across several worlds?

The implementation currently spends all extra compute on the first option.
