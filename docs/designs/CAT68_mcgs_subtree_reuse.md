# CAT-68 — MCGS cross-move subtree reuse (Phase D, behind eval server)

**Status:** DESIGN-COMPLETE, BUILD-DEFERRED. This document is the deliverable for
CAT-68. No subtree-reuse implementation is built by this ticket. Building is gated
on a measured trigger (see [BUILD TRIGGER](#build-trigger)) from CAT-71's per-leaf
profiler AND on the resolution of CAT-67 (build or explicit deprioritize).

**Linear:** [CAT-68](https://linear.app/catann/issue/CAT-68)
· blocked by [CAT-67](https://linear.app/catann/issue/CAT-67) (eval server)
· trigger source [CAT-71](https://linear.app/catann/issue/CAT-71) (standing perf profiler)

**Interface stub:** `src/catan_zero/search/subtree_reuse.py` (typed signatures,
`NotImplementedError`, points back here).

---

## 1. Problem statement

Today every decision throws its search tree away. `GumbelChanceMCTS.search()`
(`src/catan_zero/search/gumbel_chance_mcts.py:426`) builds a fresh root each call:

```python
root = _GNode(game=game.copy(), root_color=root_color)   # gumbel_chance_mcts.py:465
self._expand(root, at_root=True)
```

and the tree is garbage-collected when `search()` returns its `SearchResult`. But
after we play the chosen move and the opponent(s) respond and a chance outcome
resolves, the position we search *next* is a node we very likely already expanded
and accumulated statistics for under the previous root. MCGS-style reuse retains
that subtree — its visit counts and value estimates — as the new root instead of
re-expanding from scratch, so the search budget is spent deepening known-relevant
lines rather than re-discovering them.

Cited precedent effect sizes (deterministic games): **+69 Elo in chess, +310 Elo
in crazyhouse** (master plan queue #17 [R7]). Those are the numbers that make this
worth designing. They are also from **deterministic** games, which is exactly why
the design below spends most of its length on why Catan is harder.

This is a **throughput / sample-efficiency** win, not a strength-per-sample-of-
*training* win, which is why it is Phase D and sequenced behind the eval server
(see §6).

---

## 2. Current tree representation (what we would persist)

Two dataclasses in `gumbel_chance_mcts.py`:

**`_GNode` (`:384`)** — a decision node:
- `game`: the Rust `Game` snapshot (`game.copy()`).
- `root_color`, `prior_value`, `visits`, `value_sum`.
- `actions: dict[int, _GAction]` — one edge per legal action id.
- `action_json`, `action_logits`, `action_spectrum` — cached expansion metadata.
- `expanded: bool`.

**`_GAction` (`:351`)** — an edge (an action from a node):
- `prior`, `visits`, `value_sum`, `value_sq_sum` (the last for variance-aware Q).
- **`children: dict[int, _GNode]`** — keyed by **chance-outcome index**, not by a
  single deterministic successor. This is the crux for Catan (see §3).
- `probabilities: dict[int, float]` — the outcome distribution over those children.
- `afterstate_value: float | None`.

So the tree is already a proper *chance tree*: a decision node → action edges →
(per chance outcome) child decision nodes. `_traverse_roll()`
(`gumbel_chance_mcts.py:982`) samples one `outcome_index` and recurses into
`stats.children[outcome_index]`; the ≤11 dice children are materialized together
by `_enumerate_roll_outcomes()` (`:912`) via `Game.apply_chance_outcomes_batch`.

**What reuse would retain:** after the real game advances from the searched root
to the next decision point, find the `_GNode` in the old tree that corresponds to
the *actually realized* successor state, and adopt it (with its subtree,
`visits`, `value_sum`, and descendants' stats) as the new root.

### Tree vs. graph (MCGS proper)

True MCGS uses a **transposition graph** (DAG) so distinct move orders reaching the
same state share one node. Our current structure is a **tree** — transpositions are
separate nodes. This design proposes reuse in **two stages**, and deliberately does
NOT start with the graph:

- **Stage 1 (this ticket's target): tree re-rooting.** Keep the tree structure,
  just retain the correct child subtree as the next root. This captures the bulk
  of the +Elo (which comes from not re-searching the line you committed to) with
  far less risk.
- **Stage 2 (explicitly out of scope, future ticket): transposition graph.**
  Requires a canonical state key (hash of the Rust snapshot) and node dedup, and
  raises correctness questions around stat double-counting on DAG backups. Deferred
  until Stage 1 is measured. Catan's high branching + hidden info make cheap
  transpositions rarer than in chess, so the marginal Stage-2 win is likely small —
  measure Stage 1 first.

---

## 3. The chance-node complication (the hard part)

In chess, "play move m" maps the root to exactly one child, and MCGS re-roots
there. Catan breaks this in three compounding ways.

### 3.1 The successor is behind a chance node, keyed by outcome

To reach the next decision you must pass through the acting player's ROLL (and
possibly robber-steal / dev-card-draw) chance node. In our tree the successors are
`stats.children[outcome_index]`. Re-rooting therefore requires mapping the
**realized real-game outcome** (the dice that were actually rolled, the card
actually drawn) to the `outcome_index` that keys the child. If the realized
outcome was never materialized (e.g. a low-probability outcome pruned by
`positive_outcomes` in `_enumerate_roll_outcomes`, `:936`), **there is no subtree
to reuse and we must fall back to a fresh expansion** for that move. The design
must treat "realized outcome not in `children`" as the common, expected case, not
an error.

### 3.2 Lazy interior chance means most interior subtrees don't exist to reuse

With `GumbelChanceMCTSConfig.lazy_interior_chance` (`gumbel_chance_mcts.py:243`),
interior (depth > 0) ROLL actions are single-sampled via
`_traverse_single_sample` rather than fully enumerated — only the sampled outcome's
child is ever built, and even that is transient per-simulation. Under lazy interior
chance, the interior subtree that reuse would want to adopt **mostly does not exist
as persisted, statistically-meaningful nodes**. Only the root ROLL (depth 0) keeps
full enumeration. **Consequence:** meaningful reuse is realistically limited to
re-rooting at the *root's own* fully-enumerated chance children, not deep interior
lines. This sharply bounds the achievable win vs. deterministic games and MUST be
stated up front — the +69/+310 chess/crazyhouse numbers will not transfer at full
size.

### 3.3 Hidden-information reweighting makes carried-over stats potentially stale

The search resolves some chance nodes against **beliefs**, not ground truth:
`correct_rust_chance_spectra` (`:248`) recomputes MOVE_ROBBER-with-victim weights
from the victim's real hand and filters BUY_DEVELOPMENT_CARD phantom outcomes; the
`belief_*` spectra fields (`belief_chance_spectra`, `:328`, and the belief entries
around `:314`) reweight outcomes by a belief deck/hand. The statistics accumulated
in a subtree were produced **under the old root's belief state and information
set**. After the move resolves, the information set changes (a card is revealed, a
resource is stolen). A subtree's `value_sum`/`visits` computed under the old belief
may no longer describe the same distribution — this is the ticket's warning that
"the next-state distribution seen in future search may not match what was explored
under the old root." **Carrying those stats over uncritically is the corruption
risk the VERIFICATION clause spot-checks for.**

### 3.4 Design response

- **Re-root only across transitions where the information set is unchanged and the
  realized outcome maps to a materialized child.** Concretely: safe to reuse the
  subtree under the *root's* own resolved ROLL when the roll introduced no hidden-
  info revelation (plain resource production). Do **not** reuse across
  MOVE_ROBBER-steal or dev-card-draw transitions (belief-dependent) in Stage 1 —
  fall back to fresh expansion there.
- **Validate the adopted node's `game` snapshot equals the real successor state**
  before adopting (compare the Rust snapshot / state key), never trust the
  `outcome_index` mapping alone.
- **Stat hygiene:** when a subtree is adopted, it is used as-is (visits/value_sum
  intact) only if the above guards pass; otherwise it is discarded. No partial
  rescaling of stats in Stage 1 — either the subtree is valid to reuse wholesale or
  it is dropped. (Discounting/decaying carried stats is a Stage-2 refinement.)

---

## 4. Proposed design (Stage 1: guarded tree re-rooting)

Introduce a small, opt-in reuse controller that sits beside `GumbelChanceMCTS`,
not inside `search()`'s hot path:

1. `search()` optionally returns (or retains) the root `_GNode` so the caller can
   hold the tree between decisions. Gated behind a new
   `GumbelChanceMCTSConfig.subtree_reuse` flag, default `False` → exact current
   behavior (no-op when off, matching the flag-gated pattern used throughout this
   config).
2. A `SubtreeReuseController` holds the previous root. On the next decision it is
   given the previous root, the action actually played, and the realized chance
   outcome(s) that led to the new state, plus the new state's `game` snapshot.
3. It walks `prev_root.actions[played_action].children`, finds the child whose
   `game` snapshot matches the new state (§3.4 guard), and returns it as the new
   root — or returns `None` (fall back to fresh `search()`) if any guard fails.
4. `search()` accepts an optional `reuse_root: _GNode | None`; when provided and
   non-None it skips the fresh `root = _GNode(...)` construction and the initial
   `_expand`, using the retained node (already expanded, already carrying stats)
   instead. Everything downstream (Gumbel root selection, Sequential Halving,
   completed-Q) is unchanged — it just starts from a warm tree.

The reuse controller is **the only new component**; the node/edge dataclasses are
reused unchanged (the ticket's "extend the existing representation, don't fork it"
requirement). The one representational addition is that `_GNode` must be safe to
detach from its parent and re-root, which it already is (it holds its own `game`
snapshot and its own `actions` dict; no upward pointers exist).

---

## 5. Prototype & verification plan (only after trigger)

Per the ticket's VERIFICATION clause — small-scale, correctness-first:

1. Fixed small set of multi-move sequences (a handful of games), equal **total**
   node/visit budget with and without reuse.
2. **Correctness (a):** spot-check specific adopted nodes — assert the adopted
   node's `game` snapshot equals the real successor and that no stats were carried
   across a belief-changing (robber/dev-card) transition. Assert no stale/corrupt
   value in reused nodes.
3. **Policy quality (b):** top-choice agreement between reused-tree search and
   fresh search at equal budget should be comparable (not worse).
4. **Throughput (c):** measure real wall-clock / node-count savings. **A finding
   of "small or negligible win because lazy interior chance + hidden info limit
   reusable subtrees" is a valid, expected output** (§3.2) — quantifying it is the
   point, not a failure.

---

## 6. Why this sequences BEHIND the eval server (CAT-67)

The roadmap Tier 3 table sequences subtree reuse "Phase D; behind eval server per
profiling." Concretely:

- **Different levers, measured in the right order.** Subtree reuse changes *how
  many leaves* a game issues (fewer, because warm-started); the eval server
  changes *how efficiently a stream of leaves is served*. If you build reuse first,
  you re-tune the serving path afterward and the reuse measurement's baseline
  moves under you. Fixing serving first (CAT-67) gives reuse a stable rows/hr
  baseline to demonstrate leaf-count savings against.
- **Shared profiler dependency.** Both consume CAT-71's per-leaf split. Reuse only
  pays off if **tree-ops / re-expansion** are a non-trivial fraction of leaf cost;
  if the profile says leaf cost is ~all featurize+FFI+forward and tree-ops are
  negligible, reuse saves little wall-clock even when it saves nodes.
- **Escape hatch.** If CAT-67 is deprioritized after profiling (its trigger does
  not fire), CAT-68 is unblocked to proceed on its own trigger below — it is
  sequenced behind, not hard-forbidden until, the eval server.

---

## BUILD TRIGGER

Build CAT-68 (move from design-complete to active implementation) **only when
all** of the following hold:

1. **CAT-67 is resolved** — either built/landed, or explicitly deprioritized after
   its own profiling (its trigger did not fire). CAT-68 must not start while CAT-67
   is still an open, unprofiled question.
2. **Tree-ops are a material fraction of leaf cost.** CAT-71's per-leaf split
   (featurize / FFI / NN-forward / **tree-op**) shows the tree-op /
   node-management term is **≥ ~15%** of per-leaf cost, i.e. re-expanding trees
   from scratch is actually costing measurable wall-clock. If tree-ops are
   negligible (leaf cost is dominated by featurize+FFI+forward), the node savings
   from reuse do not translate into rows/hr — **close CAT-68 as "no measurable
   wall-clock win: leaf cost is not tree-op-bound."**
3. **Reusable subtree fraction is non-trivial.** A cheap one-off probe (instrument
   `search()` to log, per decision, whether the realized successor state maps to an
   already-materialized, information-set-safe child per §3.4) shows that a
   meaningful fraction of decisions (say **≥ ~20%**) would actually find a valid
   reusable subtree given the current `lazy_interior_chance` setting. If lazy
   interior chance + hidden-info guards mean almost nothing is ever safely
   reusable (§3.1–3.3), **close CAT-68 as "structurally limited: Catan's chance +
   hidden-info structure leaves too little safely-reusable subtree"** — and record
   that finding, which is itself valid ticket output.

Record the CAT-71 profiler snapshot and the reusable-fraction probe output that
drove the decision in the ticket, whichever way it goes.
