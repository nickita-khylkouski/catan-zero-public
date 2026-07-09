# CAT-70 Audit: Is the exact-budget SH port restart-style or stockpiling-style?

Generated: 2026-07-08

## Question

The chronicle's kill-list entry (`docs/plans/CATAN_ZERO_RESEARCH_CHRONICLE.md` ~L188,
~L448/kill-list table) records the exact-budget Sequential Halving port
(task #61, branch `origin/f61-exact-budget-sh`) as decisively beaten
(45.9%, LLR −3.9 at n_fast=16). Review-8 (`docs/reviews/review-8.md:58`,
`docs/plans/CATAN_ZERO_MASTER_PLAN.md:161,367,485`) flags an unexamined
confound: mctx's real Sequential Halving *stockpiles* Q/visit stats across
rounds, while a naive Karnin-style port *discards* stats between rounds.
If the #61 port is restart-style, the 45.9% loss is confounded and cannot
be cited as clean evidence against exact-budget accounting alone.

## Verdict: STOCKPILING-STYLE — no confound. The kill-list result stands.

## Evidence

Both master's current schedule (`_run_root_search`,
`src/catan_zero/search/gumbel_chance_mcts.py:615-654`) and the f61 branch's
replacement (`origin/f61-exact-budget-sh:src/catan_zero/search/gumbel_chance_mcts.py:461-509`)
share the identical structural property that determines stockpiling vs.
restart: **a single `_GNode` root object is constructed once, per-action
stats live in `_GAction.visits`/`_GAction.value_sum` on that one object,
and the SH round/phase loop never re-creates or zeroes those stats between
rounds — it only re-sorts/truncates the `remaining` candidate list.**

Root construction happens exactly once, before the SH loop is entered
(master `gumbel_chance_mcts.py:...` / identical in f61):
```python
root = _GNode(game=game.copy(), root_color=root_color)
self._expand(root, at_root=True)
...
sh_winner_action, used = self._run_root_search(root, n_simulations)
```

The f61 branch's `_run_root_search` (`origin/f61-exact-budget-sh:gumbel_chance_mcts.py:461-509`):
```python
def _run_root_search(self, root: _GNode, n_simulations: int) -> tuple[int, int]:
    ...
    remaining = list(top_k)
    schedule = sequential_halving_schedule(m, n_simulations)

    used = 0
    for phase_index, (count, sweeps) in enumerate(schedule):
        considered = remaining[: min(count, len(remaining))]
        for _sweep in range(sweeps):
            for action_id in considered:
                if used >= n_simulations:
                    break
                self._simulate(root, depth=0, forced_action=action_id)
                used += 1
            if used >= n_simulations:
                break
        completed_q = self._completed_q(root)
        rescaled_q = self._rescale_completed_q(completed_q)
        scale = self._sigma_scale(root)
        remaining = sorted(
            remaining,
            key=lambda action_id: (
                gumbel[action_id] + logits.get(action_id, 0.0)
                + scale * rescaled_q.get(action_id, 0.0)
            ),
            reverse=True,
        )
        if used >= n_simulations:
            break
        remaining = remaining[: schedule[phase_index + 1][0]]

    final_action = remaining[0] if remaining else top_k[0]
    return int(final_action), used
```

Every `self._simulate(root, depth=0, forced_action=action_id)` call, in
every phase, mutates the SAME `root.actions[action_id]` `_GAction` object:
`_simulate` (`gumbel_chance_mcts.py:855-910`, identical on both branches)
ends with
```python
node.visits += 1
node.value_sum += value
```
and — for the forced-root-action case — the per-action stats it updates
are `node.actions[action_id]` (`_simulate:876-878`, `stats = node.actions[action_id]`),
which is the exact same `_GAction` instance across every SH phase. There is
no code path anywhere in the diff (`git diff master origin/f61-exact-budget-sh
-- src/catan_zero/search/gumbel_chance_mcts.py`) that re-initializes,
zeroes, or replaces `_GAction.visits`/`value_sum` between phases, and no
second `_GNode(...)` or `_expand(...)` call inside the loop — the full diff
of that file (683 lines) contains exactly one `_GNode(game=game.copy(), ...)`
construction (root creation, before the loop; grepped for `reset`,
`visits = 0`, `value_sum = 0` inside the diff — zero matches). `count` in
the schedule only controls how many candidates are `considered` /kept in
`remaining`; it never touches accumulated stats for a surviving candidate.
A candidate that survives from phase 1 to phase 2 walks into phase 2 with
its phase-1 visits and value_sum intact, and phase 2's sims are ADDED on
top. This is stockpiling by construction, not restart.

What actually changed between master and f61 (confirmed via full diff) is
**only the schedule/budget arithmetic and truncation order**:
`sequential_halving_schedule` switched from a fixed-round-count schedule
with an "at least 1 sim" floor that could overspend
(`master:138-158`, comment: "total usage can exceed n_simulations — this
is expected/standard behavior") to a phase-accumulating schedule with
mid-sweep truncation so realized total sims == `n_simulations` exactly
(`f61:136-...`, docstring: "Exact-budget port of mctx's
`get_sequence_of_considered_visits` ... the realized total is EXACTLY
`n_simulations`"), plus a switch from action-major to sweep-major visit
order inside a phase. Neither change touches the persistence of
`_GAction` stats across phases — both variants stockpile identically.

The f61 branch's own docstring (`sequential_halving_schedule`,
`origin/f61-exact-budget-sh:gumbel_chance_mcts.py:136-157`) explicitly
frames the port as targeting mctx's `get_sequence_of_considered_visits`
truncation semantics ("sim-for-sim... matching mctx's
`sequence[:num_simulations]` truncation... sweep-major order"), i.e. the
intent and the implementation are both about budget/order, not about
discarding statistics — consistent with the stockpiling verdict.

## Baseline: what "stockpiling" means in this codebase's own terms

`docs/reviews/review-8.md:58` and `docs/plans/CATAN_ZERO_MASTER_PLAN.md:161`
state the mctx baseline explicitly: "Karnin-style SH formally *discards
statistics between rounds* while mctx *stockpiles* visits, and the bandit
literature reports stockpiling dominates empirically." The code evidence
above shows this repo's #61 port matches the mctx/stockpiling side of that
dichotomy, not the Karnin/restart side.

## Conclusion for the chronicle

No confound. The 45.9%/LLR −3.9 kill-list result for exact-budget SH is
clean: the port under test carried per-candidate Q/visit statistics
forward across all SH rounds exactly as mctx does, so the loss can only be
attributed to the budget-accounting change (fixed-round overspend vs.
exact-budget truncation + sweep-major order), not to any statistics-discarding
effect. The existing mechanistic explanation already on record ("exact-16 =
one visit per candidate = zero halving rounds = noise argmax," i.e. the
accidental overspend was protective at tiny budgets) remains the best
explanation and needs no revision.

### Recommended annotation for the chronicle (kill-list entries, ~L188 and ~L448 table row)

> Confound check (2026-07-08, `docs/audits/CAT70_EXACT_SH_AUDIT.md`): verified
> by reading `_run_root_search`/`_simulate` on both `master` and
> `origin/f61-exact-budget-sh` — the port stockpiles per-candidate Q/visit
> stats across SH rounds on the same `_GNode`/`_GAction` objects (mctx-style),
> it does not discard/restart them (Karnin-style). No confound: the 45.9%
> loss is clean evidence against exact-budget accounting alone.

## Confidence

High. Verified by reading the actual loop bodies and the full file diff
(not inferred from names): single root/single `_GAction` per action,
mutated in place every phase, no reset anywhere in either branch's code.
