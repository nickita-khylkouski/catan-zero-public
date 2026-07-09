# CAT-67 — Eval server / cross-game leaf batching (Phase D, profile-gated)

**Status:** DESIGN-COMPLETE, BUILD-DEFERRED. This document is the deliverable for
CAT-67. No production eval server is built by this ticket. Building is gated on a
specific, measured trigger condition (see [BUILD TRIGGER](#build-trigger)) that
consumes CAT-71's per-leaf profiler output.

**Linear:** [CAT-67](https://linear.app/catann/issue/CAT-67)
· blocked by [CAT-65](https://linear.app/catann/issue/CAT-65) (Rust featurizer)
· blocks [CAT-68](https://linear.app/catann/issue/CAT-68) (MCGS subtree reuse)
· trigger source [CAT-71](https://linear.app/catann/issue/CAT-71) (standing perf profiler)

**Interface stub:** `src/catan_zero/search/eval_server.py` (typed signatures,
`NotImplementedError`, points back here).

---

## 1. Problem statement

At generation scale we run many concurrent games, each doing Gumbel-AlphaZero
search, each search issuing a stream of NN leaf evaluations (policy priors +
value). The question this ticket answers is whether we can raise NN throughput by
batching leaf evaluations **across independent games** into fewer, larger forward
passes — as opposed to the batching we already do **within** one game's search
tree.

Crucially, this is a **throughput** optimization ("more experiments per day"),
not a **strength-per-sample** one. Per the master plan queue #17 [R7] it is
"worth ~another 2× fleet-equivalent, but it's throughput, not
strength-per-sample — sequence behind the science items." That framing is why
the whole thing is Phase D and profile-gated rather than built now.

---

## 2. What already exists (do not rebuild)

Two distinct batching mechanisms already ship. Cross-game batching is a **third,
new layer on top of these**, not a replacement for either.

### 2.1 Within-tree / chance-fan-out batching (tasks #32/#37)

`GumbelChanceMCTS._enumerate_roll_outcomes()`
(`src/catan_zero/search/gumbel_chance_mcts.py:912`) materializes all ≤11 dice
children in one `Game.apply_chance_outcomes_batch()` call and evaluates them in
one `evaluator.evaluate_many()` forward pass, gated by
`GumbelChanceMCTSConfig.use_batch_api` + `batch_api_available()`
(`gumbel_chance_mcts.py:90`). This batches the leaves of a **single chance node
inside a single game's tree**.

### 2.2 Within-process, cross-thread micro-batcher (the "batch API")

`BatchedEntityGraphRustEvaluator`
(`src/catan_zero/search/neural_rust_mcts.py:479`) is a thread-safe wrapper: each
calling thread prepares its entity/action tensors, enqueues a
`_BatchedEvalRequest` (`neural_rust_mcts.py:731`) onto `self._requests`, and
blocks on `request.done.wait()`. A single background worker thread
(`_batch_loop`, `neural_rust_mcts.py:630`) drains the queue, merges requests via
`_merge_batched_eval_requests()`, runs **one** `policy.forward_legal_np()`
(`neural_rust_mcts.py:696`), and fans results back out.

**The load-bearing limitation for this ticket:** self-play generation runs **one
process per worker, one thread per process** (see the header comment in
`tools/bench_leaf_eval_batching.py`, which calls `torch.set_num_threads(1)` to
match the real deployment). A lone thread blocks on `request.done.wait()` before
it can enqueue a second request, so `_batch_loop` never sees a batch of size > 1.
The code already knows this: it only pays the `max_wait_ms` straggler timer once
`self._observed_concurrency` flips true (`neural_rust_mcts.py:657`), which a
single-threaded caller can never trigger. **So in the real generation
configuration the existing micro-batcher runs at effective batch size 1.** That
is precisely the gap cross-game batching would close — the concurrency exists, it
is just spread across *processes*, where an in-process queue cannot see it.

### 2.3 The evaluator contract

Both real evaluators satisfy the `RustEvaluator` Protocol
(`src/catan_zero/search/rust_mcts.py:62`):

```python
def evaluate(self, game, legal_actions, *, root_color, colors)
    -> tuple[dict[int, float], float]   # {action_id: prior}, value
```

plus an optional `evaluate_many(requests, *, root_color, colors)` used by the
chance fan-out. **Any eval server client must present exactly this surface** so it
drops into `GumbelChanceMCTS(evaluator=...)` with zero search-code changes. This
is the key design constraint: the cross-game layer is a *new evaluator
implementation*, not a change to search or to the existing batcher.

---

## 3. Why the premise may already be dead (read before building)

The Rust featurizer work (CAT-65 / task #81) is expected to leave the per-leaf
cost split at roughly (CAT-71 baseline to validate against):

- **GPU per-leaf ≈ 3.4 ms, of which NN forward is only ~4%; featurize + FFI ≈ 96%.**
- CPU int8 ≈ 38 ms/eval, forward-pass-dominated.

Cross-game batching **only helps the NN-forward fraction** — it packs more leaves
into one `forward_legal_np()`. If, after the Rust featurizer lands, featurize+FFI
is still ~96% of GPU leaf cost, then even driving NN-forward to zero would cut
leaf cost by ≤4%, and this entire architecture is not worth building on the GPU
path. The CPU path (forward-dominated) is where cross-game batching could pay off,
**if** generation is CPU-bound at that point.

This is the single most important thing this design says: **do not build until
CAT-71's post-Rust profile shows NN-forward is actually a meaningful, GPU-bound
fraction of leaf cost.** The ticket's own scope text says the same
("confirm this ticket's premise still holds after that work lands before
investing heavily here"). This document exists so that when someone picks this up
they re-run the profile *first* instead of trusting the pre-Rust 3.4ms number.

---

## 4. Proposed architecture (if the trigger fires)

A separate **eval-server process** holds the single GPU-resident (or int8-CPU)
policy and services inference requests from N game-worker processes over IPC.

```
  worker proc 1 ──┐        (IPC: shared-mem ring / UNIX socket)
  worker proc 2 ──┤            ┌───────────────────────────┐
  worker proc 3 ──┼──requests──▶│  eval-server process      │
       ...        │            │  - single policy on GPU    │
  worker proc N ──┘            │  - collector queue         │
       ▲                        │  - window batcher (size B, │
       │                        │    timeout T_ms)           │
       └────────results─────────│  - one forward per window  │
                                └───────────────────────────┘
```

### 4.1 Client side (`RemoteEvalClient`)

Lives inside each game-worker process. Implements the `RustEvaluator` Protocol so
it is a drop-in `evaluator=` for `GumbelChanceMCTS`. On `evaluate()` /
`evaluate_many()` it:

1. Does the **featurization locally** (`rust_game_to_entity_batch`,
   `rust_action_context_batch`, `rust_policy_action_ids` — all already in
   `neural_rust_mcts.py`). Featurization stays distributed across worker cores;
   only the packed tensors cross the IPC boundary. This is deliberate — moving
   featurization into the server would recreate the 96%-cost bottleneck as a
   serial chokepoint.
2. Serializes the packed request (entity dict of ndarrays, legal-action ids,
   context, metadata) and hands it to the server.
3. Blocks on a per-request completion primitive (analogous to
   `_BatchedEvalRequest.done`), then applies the **same** post-processing the
   in-process batcher does today: softmax at `prior_temperature`, value squash,
   two-player perspective negation, clip (`neural_rust_mcts.py:704-716`). Keeping
   post-processing on the client keeps the server a pure tensor-in/tensor-out
   service and guarantees bit-identical outputs vs. the local path.

### 4.2 Server side (`EvalServer`)

Single process. A collector coroutine/thread receives requests from all workers
into one queue; a batcher forms windows by the classic **max-batch-size B OR
max-wait T_ms** rule (the same two knobs `BatchedEntityGraphRustEvaluator`
already exposes as `max_batch_size` / `max_wait_ms`, now operating across
processes instead of threads); each window is one `forward_legal_np()`; results
are scattered back by request id. The shared `_merge_batched_eval_requests()` /
`_pad_1d_np` / `_pad_2d_np` padding helpers (`neural_rust_mcts.py:748-804`) are
reused verbatim for window assembly.

### 4.3 Transport

Prototype with the simplest thing that works and measure before optimizing:
- **v0 (prototype):** `multiprocessing` `Queue`/`Pipe`, pickle the packed ndarrays.
  Adequate to answer "does cross-game batching improve throughput at all."
- **v1 (only if v0 shows a win but IPC serialization dominates):** shared-memory
  ring buffer (`multiprocessing.shared_memory`) for the tensor payloads, socket
  only for control/wakeup. Do not build v1 speculatively.

---

## 5. Risks & open questions

1. **Latency vs. throughput.** Cross-process batching adds a round trip + windowing
   delay to every leaf. Search is latency-sensitive: a worker cannot advance its
   tree until the leaf resolves. The win requires that enough workers are in
   flight that the server's window fills fast relative to T_ms. Below some worker
   count the added latency is pure loss. The prototype must sweep worker count.
2. **This interacts with, and may be redundant to, MPS.** NVIDIA MPS already lets
   multiple worker processes share one GPU context. If MPS gives most of the
   sharing benefit, a bespoke eval server buys little. **The eval server and MPS
   are alternative answers to the same question** and must be compared head to
   head, not stacked blindly.
3. **The 50 ms GPU-context-thrash anomaly.** The speed-czar program observed
   50 ms/eval on the fleet with `pmon` showing ~90% SM / ~0% memory utilization —
   the signature of GPU **context thrashing** (many processes fighting over one
   context), not genuine compute. A single-context eval server is *one plausible
   fix* for exactly this pathology. That makes CAT-67 partly a candidate remedy
   for a known anomaly — but it must be validated against MPS (risk 2) as the
   competing remedy, and CAT-71's dashboard must confirm the thrash signature is
   actually present before attributing the fix to this work.
4. **Determinism.** Batching must change only latency/throughput, never outputs.
   Because featurization and all post-processing stay client-side and only the
   raw `forward_legal_np()` is centralized, outputs should be bit-identical to the
   local path modulo GPU nondeterminism inherent to batched matmul. The
   verification harness must assert priors/value match the non-server baseline
   within tolerance (the ticket's VERIFICATION requirement).
5. **Failure isolation.** A crashed eval server strands every worker. Need a
   client-side timeout + fall-back to a local in-process evaluator so one server
   fault does not kill a whole generation wave.

---

## 6. Prototype & verification plan (only after trigger)

Per the ticket's VERIFICATION clause — prototype-and-measure only, no
production deployment:

1. Fixed small set of concurrent games (sweep e.g. 4/8/16/32 workers), single GPU.
2. Measure rows/hr and per-leaf latency **with** the eval server vs. **without**
   (each worker using its own in-process evaluator), holding the checkpoint,
   node budget, and seeds fixed.
3. Assert output parity: for a captured set of leaf inputs, server priors/value ==
   local priors/value within tolerance.
4. Compare against an **MPS-only** configuration (risk 2) as a third arm.
5. Report throughput delta as a function of worker count; identify the break-even
   worker count below which the server loses.

---

## 7. Sequencing

CAT-68 (MCGS subtree reuse) is explicitly sequenced **behind** this ticket
("Phase D; behind eval server per profiling"). Rationale: subtree reuse changes
*how many* leaves search issues; the eval server changes *how efficiently* a given
stream of leaves is served. Fixing the serving path first gives subtree reuse a
clean, measured baseline to show its leaf-count savings against. If the eval
server is deprioritized after profiling (trigger below does not fire), CAT-68 is
unblocked to proceed independently.

---

## BUILD TRIGGER

Build CAT-67 (move from design-complete to active implementation) **only when
all** of the following are measured true from CAT-71's standing per-leaf
profiler output, on the post-Rust-featurizer (CAT-65) code:

1. **NN-forward is a material fraction of leaf cost.** CAT-71's per-leaf split
   (featurize / FFI / NN-forward / tree-op) shows **NN-forward ≥ 25%** of total
   per-leaf time on the path generation actually runs. (Pre-Rust baseline was
   ~4% on GPU — at ~4% this ticket is dead on the GPU path regardless of anything
   else below.)
2. **Generation is GPU-forward-bound, not featurize-bound.** After the Rust
   featurizer lands, featurize+FFI is **no longer the dominant** per-leaf term on
   the active path (i.e. featurize+FFI < NN-forward + tree-op), so that improving
   NN batching can actually move aggregate rows/hr.
3. **Effective NN batch size is ~1 in production.** CAT-71 (or a one-off probe)
   confirms `BatchedEntityGraphRustEvaluator._observed_concurrency` stays false in
   the real one-process-per-worker deployment — i.e. the in-process batcher is
   genuinely running at batch size 1, so cross-*process* batching is the only way
   to recover batch efficiency.
4. **MPS does not already close the gap.** An MPS-only configuration measured under
   CAT-71 does **not** already deliver the throughput the eval server is projected
   to add (i.e. MPS alone leaves ≥ ~1.5× headroom on the forward path), AND/OR the
   50 ms context-thrash signature (90% SM / 0% mem via `pmon`) is present, which
   a single-context server is expected to remove.

If (1) or (2) is false, **close CAT-67 as "not worth building — profile shows
featurize-bound, not forward-bound"** and unblock CAT-68 directly. If (4) shows
MPS already suffices, close as "superseded by MPS." Record the profiler snapshot
(the CAT-71 JSON) that drove the decision in the ticket either way.
