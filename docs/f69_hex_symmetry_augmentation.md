# f69 design: 12-fold hex symmetry augmentation

Status: design only (no implementation in this branch). Scope: the
`entity_graph` model and its featurizer (`entity_token_features.py`),
2p no-trade Catan.

## Motivation

The measured failure is that the value/prior cannot rank near-tied opening
placements: a ~0.06-nat prior spread over 54 candidates, with value noise
dominating (see `tools/sigma_trace_placement_root.py` and
`tools/f69_ranking_probe.py`). The Catan board has a dihedral symmetry group
**D6** (6 rotations x 2 reflections = 12 elements) about the centre hex. Every
element `g` maps a game state to a strategically **identical** state with the
board relabelled. Two independent wins follow:

1. **Training augmentation** — each real state yields up to 12 equivalent
   `(state, policy-target, value-target)` samples, a 12x data multiplier that
   is exactly label-consistent (not a heuristic perturbation), pushing the net
   toward the invariance the true value function has.
2. **Test-time averaging** — at a wide root, evaluate all 12 orientations of
   the same position and average. The value target is *invariant* under `g`, so
   if the head's error is even partly independent across orientations, the
   average shrinks value noise by up to `sqrt(12) ~= 3.46x`. That noise is the
   term the sigma-trace diagnosis found "manufacturing false confidence" among
   near-tied candidates, so this directly raises the ranking ceiling.

## What is and isn't orientation-dependent (verified against the featurizer)

The trunk is a **set transformer with type embeddings only** — there is *no*
per-entity positional encoding (`entity_token_policy.py`: `type_embedding` has
7 rows, one per entity type; `cls_token`; nothing indexes an individual hex/
vertex/edge slot). So the trunk is already permutation-equivariant over tokens
*within a type*. Symmetry therefore reduces to: permute the token rows by the
automorphism `g` induces on each entity type, relabel the id tables, and fix up
the one genuinely position-dependent feature.

Per-token feature audit (`entity_token_features.py`):

| Token | Feature content | Orientation-dependent? |
|---|---|---|
| hex (`_hex_tokens`) | present flag; **cube coordinate `/4` (dims 1:4)**; resource one-hot; number; dice pips; robber flag | **YES — dims 1:4 only.** All others intrinsic. |
| vertex (`_vertex_tokens`) | present; ownership; building type; production pips per resource; adjacent-robber; port resource one-hot; is-actor | No (all intrinsic to the node; ports move with the node) |
| edge (`_edge_tokens`) | present; ownership; adjacent-hex count; is-actor | No |
| player, global | prompt, VPs, resources, bank, trade panel, player-count | No (non-spatial) |
| event (`_event_tokens`) | event/action type; actor; turn key; **scaled `action_id` (dim 35)** | **YES — dim 35** (action id encodes a board target) |

Two position-dependent features exist: the **hex cube coordinate** and the
**event-log `action_id`**. Everything else is intrinsic and is handled purely by
row permutation.

### The hex-coordinate fix is trivial

Because the coordinate is a fixed function of the *slot* (tile_id `a` always
sits at canonical coordinate `c_a`), and `g` sends the tile at `c_a` to the
slot at `c_b = M_g c_a` (i.e. `pi_hex(a)=b`), the coordinate that slot `b` must
report after augmentation is exactly `c_b/4` — the canonical value slot `b`
already held. **So: permute the intrinsic hex dims (0, 4:13) by `pi_hex`, and
leave the coordinate dims (1:4) at their canonical per-slot values (unchanged).**
No matrix multiply on the feature is needed at all; the coordinate column is
slot-fixed.

### Event action_id caveat

`event_tokens[:,35]` is a scaled scalar of a past action's policy id, which
encodes a board target and so changes under `g`. `event_target_ids` is
currently all `-1` (unused). Options, in order of fidelity: (i) relabel each
event's `action_id` through the action-permutation `pi_act(g)` when augmenting;
(ii) accept the small inconsistency (it is one scaled scalar among 41 event
dims and the event stream is history, not the decision surface); (iii) zero dim
35 in augmented copies. Recommend (i) for training augmentation and note that
test-time value averaging at *opening* roots is unaffected (event log is
near-empty at placement).

## The permutation tables

`g in D6` acts on cube coordinates by the standard hex rotation/reflection
linear maps `M_g` (rotations: cyclic (x,y,z) permutations with sign; reflection:
swap two axes). From the board's fixed geometry (catanatron map: hex
coordinates, node coordinates, edge = unordered node pair) precompute, **once**:

- `pi_hex[g]`: length-19 permutation, `a -> slot at M_g c_a`.
- `pi_vertex[g]`: length-54 permutation from node geometric positions under `M_g`.
- `pi_edge[g]`: length-72 permutation; an edge `(u,v)` maps to the edge
  `(pi_vertex[u], pi_vertex[v])` (look up its id via `edge_to_id`).
- `pi_act[g]`: permutation on the policy action space induced by relabelling
  each action's target (settlement@node, road@edge, robber@hex, ...); non-spatial
  actions (ROLL, END_TURN, dev cards, bank trades) are fixed points.

These are pure functions of the map, so they are computed once and cached (12
triples). Validate them by asserting they form a group closed under
composition and that each is an automorphism of the board incidence
(`hex_vertex_ids`, `edge_vertex_ids`) — i.e. applying `pi` to the id tables
yields a table equal to relabelling, which is the correctness test.

## Consistent transform of a featurized state

Given a state's entity dict and `g`:

- `hex_tokens`  = rows permuted by `pi_hex`; then restore canonical coord dims 1:4.
- `vertex_tokens` = rows permuted by `pi_vertex`.
- `edge_tokens` = rows permuted by `pi_edge`.
- `hex_vertex_ids` = rows by `pi_hex`, **values** mapped through `pi_vertex` (keep `-1`).
- `hex_edge_ids`  = rows by `pi_hex`, values through `pi_edge`.
- `edge_vertex_ids` = rows by `pi_edge`, values through `pi_vertex`.
- `legal_action_target_ids` (`[A,4]`): col0 through `pi_hex`, col1 `pi_vertex`,
  col2 `pi_edge`, col3 **unchanged** (players are not spatial); `-1` preserved.
  Action rows may keep their order (a set); the paired `legal_action_tokens`
  and `legal_action_context` rows move with them (all intrinsic).
- masks: permuted with their tokens.
- player/global tokens: unchanged.
- event tokens: rows unchanged; relabel dim 35 via `pi_act` (see caveat).

The `-1` sentinels must survive every value-relabel (guard `id >= 0`), matching
the model's gather guard in `_gather_target_tokens`.

## Training integration

- Augment at collate time: for each sampled `(state, target)`, draw `g` (uniform
  over 12, or expand each sample x12 for exhaustive coverage) and apply the
  transform above to the entity tensors. The **policy target** (visit/prior
  distribution over legal actions) is relabelled by permuting the per-action
  target ids identically — since candidate rows carry their own relabelled
  target ids, the distribution over rows is unchanged and no reindexing of the
  target vector is required if targets are stored per-row; if targets are
  stored per-policy-action-id, map them through `pi_act`. The **value target**
  is copied unchanged (invariant).
- Because the trunk is already permutation-equivariant, augmentation teaches the
  net only to be invariant to the *coordinate* feature and any residual
  order-sensitivity — a small, well-posed target. Consider a curriculum: start
  with x12 exhaustive on the widest (opening) roots where the ranking failure
  lives, sample-1-of-12 elsewhere for throughput.

## Test-time value averaging at wide roots

At a root (especially the 54-wide opening), for value and prior:

1. Featurize the canonical state once; derive the 11 other orientations by
   applying the cached permutations to the tensors (cheap; no re-featurization).
2. Batch all 12 through the net in one forward.
3. **Value**: average the 12 scalars (invariant target -> variance reduction).
4. **Prior / root-Q**: map each orientation's per-candidate output back to
   canonical candidate identity via `pi_act[g]^{-1}`, then average per candidate.
   This is the ranking-relevant averaging: it denoises the near-tied candidate
   scores the sigma trace found to be dominated by noise.

Cost: 12x the root forward only (roots are rare vs. total evals; the 35M net at
~34 ms/eval int8 CPU or far less batched on GPU makes 12x-at-root negligible
against a 54-wide search). Measure the post-averaging candidate-score spread
with `tools/f69_ranking_probe.py` (extend it to average over orientations) and
compare against the single-orientation baseline it already reports.

## Correctness tests to ship with the implementation

- Group laws: `pi_*` closed under composition; identity element is identity.
- Incidence automorphism: relabelled id tables equal recomputed tables.
- **Model invariance**: for a trained checkpoint, `value(g . s) == value(s)` and
  `prior(g . s)` equals `prior(s)` permuted by `pi_act[g]`, to within float
  tolerance, for all 12 `g` on real states. (With the coordinate fix and the
  set-transformer trunk this should hold to numerical precision even *before*
  augmentation training, except for the event-`action_id` caveat — a useful
  pre-flight assertion.)
