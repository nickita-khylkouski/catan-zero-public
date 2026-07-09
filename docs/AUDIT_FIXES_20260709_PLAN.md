# Audit-Fixes 2026-07-09 — Deferred Optimizations (PLAN)

Branch: `audit-fixes-20260709` (off `cat-runsix-harden`). Author: audit-fixer.

The audit's BUG-1..4 and the safe optimizations (OPT-1, OPT-3, OPT-4, OPT-7 CSE,
OPT-8) are APPLIED on this branch with tests. The items below are deferred as
bigger surgery / low-value-vs-risk and want a separate review before landing.

## OPT-2 — redundant 3x JSON fetch per decision [HIGH IMPACT, bigger surgery]
Files: `rl/gumbel_self_play.py` (build_decision_row, apply_selected_action),
`search/neural_rust_mcts.py` (_fetch_leaf_decision_inputs).
Each decision serializes the SAME unchanged game state up to 3x
(`json_snapshot` + `playable_action_indices` + `playable_actions_json`) across
search entry, row build, and action apply. Plan: fetch once per decision at the
top of `play_one_game` and thread a `(snapshot_text, action_by_id)` bundle into
`mcts.search`, `_build_decision_row`, and `_apply_selected_action` (the helper at
neural_rust_mcts.py already accepts precomputed inputs — see the "skip a second
json_snapshot" docstring). Risk: changes the search entry signature + game-loop
contract. MUST gate with a shard-parity run (identical `target_policy`/
`target_scores`/seeds vs current) before deploy. Highest expected self-play
speedup of the whole audit.

## OPT-5 — `_scores_to_policy` per-row loop vectorization [SKIP, low value]
File: `tools/train_bc.py`. DELIBERATELY NOT APPLIED. Two reasons: (1) it only
fires on the `prefer_scores` soft-target path, which BUG-1 just made non-default
— so it is now rarely called; (2) vectorizing a per-row masked softmax changes
the `np.sum`/`np.max` reduction structure (kept-subset vs full-row-with-zeros),
which can differ at the ULP level from the current output. The task requires
identical outputs; a per-row masked softmax cannot be bit-identically vectorized
across ragged supports. Not worth the target-drift risk for a near-dead path.

## OPT-6 — `move_robber_victim_outcome_weights` N+1 snapshot parses [MEDIUM]
File: `search/gumbel_chance_mcts.py:~1718`. Parses `json.loads(json_snapshot())`
for the victim hand, then again per outcome candidate (1 + N full-snapshot JSON
parses per robber-with-victim action). Plan: read just the victim's hand via a
narrower Rust accessor (e.g. a `player_state_json(color)` if exposed) or parse
the base snapshot once and diff only the hand field across candidates. Touches
the search core → needs the same shard-parity gate as OPT-2.

## OPT-7 (node-level cache) — cache `json.dumps(action_json)` on `_GNode` [deferred]
The per-index CSE inside chance expansion IS applied (raw_action_json hoisted).
The remaining idea — caching the serialized action_json on each `_GNode` so
repeated descents through the same chance node across simulations reuse it
(sites ~1378/1526/1533/1562) — touches `_GNode` layout and several hot methods.
Deferred; low impact, and each affected method currently dumps at most once per
call.

## OPT-9 — `_GNode` shares parent `action_json` instead of copying [MEMORY]
File: `search/gumbel_chance_mcts.py:~599`. Each `_GNode` holds a full game clone
+ its own `action_json` dict. Sharing `action_json` from the parent's
`_fetch_legal_actions` result (rather than a per-node copy) is a minor memory win
but requires proving the shared dict is never mutated per node. Inherent MCTS
game-clone memory is unavoidable. Low priority.
