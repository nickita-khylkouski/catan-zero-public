# H100 Execution Update — 2026-07-09

Status: active plan of record for the current H100 canary. This narrows and
updates the older H100 blueprint; it does not erase the research history.

## Decision

Keep the old five-cluster architecture portfolio on hold. We validated the
corrected canonical launcher with a four-GPU lifecycle smoke. Next, supply the
real champion artifact, repeat the strict-precision frontier with that
checkpoint, then spend most generation capacity on fresh independent outcomes.
Scale and architecture bake-offs remain gated on stable value learning and
enough fresh data.

Reasons:

1. The old blueprint says 20 H100s across five clusters; the later fleet source
   of truth says 24, while the newly supplied canary is an 8-GPU node. Resource
   placement must be based on live inventory, not that static diagram.
2. V9 falsified the assumed external deficit and HL-Gauss-as-reuse-cure. The
   strongest current diagnosis is correlated Monte Carlo label scarcity.
3. The canonical launcher contained order-of-magnitude utilization bugs, so a
   larger fleet launch would have multiplied bad placement rather than useful
   learning.

## Canary inventory and acceptance

- 8× H100 80GB SXM, NVSwitch/NV18, healthy P2P and ECC.
- 2× Xeon 8480+ (104 physical cores), 1.7 TiB RAM, 22 TiB local disk.
- Fresh canonical install: Python 3.11.15, Torch 2.11.0+cu128,
  `catanatron_rs` 0.1.3.
- Native feature/context/symmetry acceptance: 19/19 passed.
- The final local verification run completed with 1,737 passed and 200 skipped.
- We ran the H100 full suite: 1,913 passed and 24 skipped, including the CLI
  goldens and Rust parity checks.
- The final handoff delta passed 184 targeted lifecycle, harvest, audit,
  source-list, launcher, and guard tests on the H100 and returned all eight
  GPUs to 0 MiB.
- The canonical four-GPU training smoke consumed a 21,120-row, 352-shard
  deduplicated memmap corpus. The L6/h640 35M BF16/fused DDP run completed all
  five steps in 5.76 seconds, wrote model/report/optimizer artifacts, preserved
  masking and soft-target provenance, and returned every GPU to 0 MiB.
- Production champion weights are not part of the public release. A synthetic
  same-shape 35M checkpoint may be used for throughput only, never for the
  champion no-op or strength claims. The private champion artifact remains the
  only missing input for the no-op gate.

## P0 launcher corrections

The pre-change launcher:

- exposed several GPUs to one `--device cuda` generator, making every worker
  select logical `cuda:0`;
- described `--games` as games per worker and claimed
  `games × workers × GPUs`, although workers partition a total game count;
- defaulted to the retired `~/catan-zero-runsix` tree instead of the canonical
  install destination;
- omitted Rust featurization and EvalServer from generation;
- built a training command that passed `--grow-from-checkpoint` while omitting
  `--arch entity_graph`, a combination the trainer rejects;
- omitted memmap and BF16 from the advertised H100 training path;
- used `seq -s,` for GPU ranges, which emitted a trailing comma on one operator
  platform and turned `0-7` into nine devices.

The corrected contract is:

- one generator and one EvalServer per physical GPU;
- disjoint `games`-sized seed block and output subtree per GPU;
- claim interval `games × GPUs`;
- explicit `--rust-featurize --eval-server --eval-cache-size 0`;
- immediate EvalServer queue drain (`max_wait_ms=0`), strict
  `matmul_precision=highest`, and fail-fast server clients (no silent per-worker
  CUDA fallback);
- post-ready server-process supervision at intervals of at most 250 ms, plus a terminal
  no-fallback client latch, so a dead or unresponsive server cannot turn into
  one full request timeout per remaining game;
- collector pause acknowledgement around merge/scatter (OS wakeup plus an
  in-flight deserialize lock), and a policy-capability handshake that omits
  `legal_action_target_ids` IPC/padding for the base 35M model while retaining
  it for target-aware heads;
- MPS off by default because EvalServer already collapses each GPU to one CUDA
  process (`--mps` remains an explicit diagnostic option);
- GPU-local CPU affinity from `nvidia-smi topo -m` and `nofile=65536`;
- for n128 teacher generation on the production shape (up to four selected
  GPUs), 128 workers/GPU, EvalServer request collection, and a 96-request
  maximum batch; the all-8 canary defaults to 64 workers/GPU to hold host
  concurrency at 512, and volume-generation worker counts remain a separate
  workload-specific tuning decision;
- the canonical L6/h640 entity-graph 35M control, enforced by
  `--require-35m-model`, with memmap, BF16, and an exact global batch of 4096;
- pure-Bash, validated GPU range expansion.

An 8-GPU dry-run now resolves devices 0–7, `nproc_per_node=8`, per-rank batch
512, and an 80-seed claim for 10 games/GPU.

## Measured native-evaluator improvement

`evaluate()` already skipped the Python adapter once Rust topology was warm;
`evaluate_many()` and the batched evaluator did not. They rebuilt snapshot,
player-state, and adapter payloads on every native leaf despite the native
entity/context builders no longer reading that payload.

H100 A/B/A, same synthetic 35M-shape model, public masking, Rust features,
cache disabled, 8 leaves per batch, 5 warmups + 60 measured batches:

| Variant | Median batch | Median per leaf | p95 batch |
|---|---:|---:|---:|
| Baseline | 19.810 ms | 2.476 ms | 20.112 ms |
| Warm-topology skip | 15.715 ms | 1.964 ms | 15.914 ms |
| Baseline repeat | 19.803 ms | 2.475 ms | 21.144 ms |

Result: 1.26× lower median leaf latency for the batched native path. New
regressions failed 2/2 before the change; the current shared-payload suite passes
15/15 (26/26 with adjacent Rust entity/context/symmetry coverage).

## Live H100 throughput frontier

All figures below use the synthetic same-shape 35M masked checkpoint and are
throughput evidence only. They are not playing-strength evidence. Runs used
real Rust games, public-observation masking, native featurization, corrected
chance spectra, lazy interior chance, and the shared EvalServer.

### Scheduler and context choices

- EvalServer vs one local evaluator per worker, identical eight-game n16 arm:
  **155,619 vs 87,742 rows/hour (1.77×)**.
- MPS on vs off after EvalServer consolidation: **153,473 vs 155,619
  rows/hour**. MPS added no benefit and reserved 78 MiB on every idle GPU.
- EvalServer wait sweep at eight workers/GPU:
  `0/0.1/0.25/0.5/1/2/3/5 ms` produced
  `163/152/149/150/152/144/136/126k rows/hour`. Immediate drain won; the old
  3 ms setting was about 17% slower.
- GPU-local CPU affinity won in both crossover directions, improving the
  production-volume arm by roughly **3–6%**.

### Worker and topology sweeps

At n16 on the 4-GPU CPU budget, workers `4/8/12/16` produced roughly
`89/163/220/283k rows/hour/GPU`. A four-GPU all-32 run reached **383k/GPU**;
48 reached **393k/GPU** (+2.4% for 50% more processes), placing the n16 knee
near 32.

The earlier production recipes moved the knee:

| Recipe | 16 workers | 32 workers | 48 workers | 64 workers |
|---|---:|---:|---:|---:|
| volume: n64@25%, n16@75% | 180k | 244k | 265k | 272k rows/h/GPU |
| teacher: n128@100% | 45k | 63k | 67k | not run |

The 8-GPU canary is not the 4-GPU fleet shape. Eight simultaneous volume
generators at 32 workers/GPU delivered **~207k/GPU, 1.65M/node**, versus
~244k/GPU in a 4-GPU-shaped run, because both shapes share 104 physical CPU
cores. Those volume results must not be used to choose the n128 teacher
concurrency; the teacher frontier was measured separately below.

### Final n128 teacher frontier

All measurements in this subsection are rows/hour/GPU using the synthetic
same-shape masked checkpoint. They certify throughput mechanics, not champion
strength or final real-checkpoint throughput.

At 48 workers, adding an aggregation delay never beat immediate draining:

| EvalServer wait | 0 ms | 0.05 ms | 0.1 ms | 0.25 ms |
|---|---:|---:|---:|---:|
| n128 rows/hour | 72.26k | 70.54k | 70.04k | 71.07k |

The pre-collector concurrency sweep continued improving beyond the old
48-worker baseline:

| Workers/GPU | 48 | 64 | 80 | 96 |
|---|---:|---:|---:|---:|
| n128 rows/hour | 68.07k | 74.41k | 74.65k | 75.98k |

With the request collector fixed and enabled, four w96 repetitions averaged
**81.93k rows/hour**. A paired concurrency comparison then measured **83.42k at
w96 versus 89.57k at w128**, a **7.4%** gain. At w128, the paired EvalServer
maximum-batch comparison measured **90.50k at batch 64 versus 91.85k at batch
96**, another **1.5%**.

The canonical n128 production-shape recipe (up to four selected GPUs) is:

- 128 workers/GPU;
- EvalServer request collector enabled;
- maximum batch 96 and wait 0 ms;
- `mp_queue` request transport and audited `event_token_limit=0`;
- `matmul_precision=highest` (strict FP32), cache size 0;
- root-wave batching and CUDA Graphs disabled for the safe fleet recipe;
- GPU-local CPU affinity and no MPS.

The 8-GPU canary defaults to 64 workers/GPU so its total 512-worker host load
matches four production GPUs at 128 workers/GPU.

At **91.85k rows/hour/GPU**, this is about **37% faster** than the earlier
~67k w48 teacher baseline and projects to approximately **2.20M rows/hour over
24 H100s**. That projection remains provisional until the identical final
recipe is repeated with the real masked champion checkpoint.

### Wave-1 low-level canary (2026-07-10)

These are isolated-canary measurements from the same synthetic 35M-shape
checkpoint. They are not yet a replacement for the 91.85k production-frontier
run, and gains from different harnesses must not be multiplied as if they were
independent.

The NPZ row writer now requests public-observation masking directly for both
entity tokens and action context. This fixes the old storage contract where
evaluator inputs were masked but persisted player tensors were omniscient. A
5,120-row audit over four new H100 arms found zero non-actor values in every
hidden player slot.

This does **not** make the search information-set correct. MCTS still traverses
the authoritative Rust `Game`, so future legal moves and chance materialization
can depend on opponents' true hidden cards. Stored inputs are public, but their
search targets can remain hidden-state-conditioned. Do not call this corpus a
fully public-belief teacher until hidden-hand permutation invariance or a real
determinization/belief-state search closes that separate issue.

The retained event stream is currently dead data. Across 2,048 sampled live
Rust states and 30,720 retained teacher rows, `event_mask` had zero live tokens.
Cropping the fully masked 64-token event tail before H2D/model inference gave:

| Paired arm | Full 64-token tail | Event limit 0 | Change |
|---|---:|---:|---:|
| GPU 0 | 54.16k rows/h | 65.85k rows/h | +21.6% |
| GPU 4 | 52.93k rows/h | 65.56k rows/h | +23.9% |

The server fails closed if any omitted event token is live. A strict-FP32
256-state leaf comparison had maximum prior difference `1.95e-8` and maximum
value difference `8.03e-7`. The model default remains the full event width;
`--eval-server-event-token-limit 0` is an explicit audited deployment flag.

Root-wave batching evaluates one ready leaf from each independent Gumbel root
candidate together. With event limit 0, the paired short arm measured **88.49k
versus 66.37k rows/hour**, a **33.3%** gain. Requests fell 54.3% and server
merge time fell 50.8%. A separate retained 128-game, 10-decision n128 pair
wrote 1,280 rows in **60.74 versus 80.29 seconds** (24.3% more rows/second,
32.2% less wall time).

Root waves preserve visit budgets and exact/root chance semantics, but they use
independent per-candidate RNG streams. They are statistically rather than
bitwise equivalent to legacy action-major traversal. On the 128 identical
decision-0 states, prior tensors were exact, target-policy L1 difference was
0.00122 on average, and target argmax agreed on 127/128 states. Opening actions
agreed only 6/128 because high-temperature action sampling consumed a different
RNG stream; this is trajectory diversity, not evidence of a target failure.
The arm stays default-off until a real-champion target audit and powered H2H
non-inferiority gate pass.

The final worker choice was also repeated in a real four-GPU-shaped run, using
GPUs 0/1/4/5 across both CPU sockets instead of multiplying a single-card
number. On the hard 10-decision opening harness, one game per w128 worker gave
**196.6k rows/hour/node (49.1k/GPU)** with the safe event-0 path and **256.0k
rows/hour/node (64.0k/GPU)** with root waves, a 30.2% node-level gain. The
root-wave arms reached 20--33 GiB sampled peak memory because a single batched
root request can contain thousands of neural rows, still within an 80 GiB H100.

To remove one-game tail bias, a matched two-games-per-worker safe-path sweep
then measured:

| Four-GPU worker setting | Node rows/hour | Per GPU |
|---|---:|---:|
| 96 workers/GPU | 234.8k | 58.7k |
| 128 workers/GPU | 243.9k | 61.0k |

Thus w128 remains the four-GPU teacher default (+3.9% over w96). These absolute
rates are from the unusually expensive opening-only synthetic-checkpoint
harness and must not replace the longer 91.85k/GPU frontier or be projected as
real-champion production capacity. They do establish that four-card scaling is
sublinear and that the worker choice survives node-level CPU contention.

The shared-memory request transport was correct but slower. It measured 49.61k
versus 54.16k rows/hour alone (-8.4%) and 59.89k versus 65.85k with event
cropping (-9.1%) because the server still had to merge/pad final tensors.
Production therefore remains on `mp_queue`; the transport is retained only as
an opt-in basis for a future zero-copy final-batch design.

Event columns also account for 5,312 zero bytes per corpus row. The opt-in
`build_memmap_corpus.py --omit-zero-events` v2 format verifies every source
event token and mask is zero before omitting the two files, then the trainer
lazily synthesizes exact zero columns. Default v1 storage is unchanged. This
saves about 155.6 MiB on the audited 30,720-row corpus and preserves
`event_target_ids`.

CUDA Graph trunk capture was also rejected for production. Independent review
found that the first microbenchmark wrongly compared full-width eager inference
against an event-cropped graph path. The corrected event-0 sweep measured
roughly +6--12% for exactly filled small buckets, +0--3% for larger exact
buckets, and **-16.5%** when 44 rows were padded to a 48-row graph. In two
GPU-crossover end-to-end pairs, graphs measured 56.10k versus 55.95k rows/hour
(+0.27%) and 56.84k versus 56.51k (+0.58%). Graph forward time was actually
slower in the crossover repeat (52.38 versus 51.95 seconds); unrelated merge
variance supplied the small wall-time lead. The graph arms also drove higher
SM utilization/power and captured 10--11 graph shapes. Keep this correct,
strict-FP32 implementation opt-in for future fixed-shape models, but leave it
off for the variable-row production server.

### Rejected or gated fast paths

- `torch.compile(dynamic=True)` repeatedly recompiled on real variable
  entity/legal-action shapes and never reached acceptable steady state. Reject.
- TF32 (`matmul_precision=high`) increased paired w48 production-volume
  throughput by 56%:

  | Matmul precision | Rows/hour/GPU |
  |---|---:|
  | `highest` | ~274k |
  | `high` | ~427.7k |

  We rejected output equivalence after the same-seed eight-game semantic A/B.
  One of eight games matched. The first differing decision indices were
  `[3, 3, 1, 0, 3, 2, None, 2]`, where `None` marks the exact game. Actions
  agreed on 94.2% of the matching trajectory prefix. The maximum target-policy
  difference reached 0.831. We measured more trajectory drift than the earlier
  64-leaf probe showed. TF32 is rejected; the canonical launcher pins strict
  `--eval-server-matmul-precision highest`.

### Real lifecycle smoke

A canonical 4-GPU launch created four 48-client EvalServers, disjoint per-GPU
seed ranges/output trees, strict `matmul_precision=highest`, no MPS, and
verified CPU masks. The recorded detached PGID contained the runner, four
generators, four servers, managers/resource trackers, and all worker
grandchildren. Canonical stop removed the entire group and verified zero
remaining GPU clients and 0 MiB on all eight GPUs. This also caught and fixed a
`pipefail`/SIGPIPE startup bug in the first affinity implementation.

After the final collector/supervisor hardening, a fresh canonical one-GPU n128
smoke ran 16 complete full-budget games through the synthetic 35M checkpoint:
16 completed, 0 failed, 14 reached the expected 600-decision cap, and 9,506
training rows were written. The server handled 672,176 requests / 954,653 leaf
rows in 143,072 windows (maximum 15 requests / 45 rows), then the detached
group exited and GPU 0 returned to 0 MiB without an operator stop. This
low-concurrency worst-case-latency smoke is lifecycle/correctness evidence, not
a replacement for the w128 throughput frontier above.

## Execution sequence

### Gate 0 — software and artifact completeness

1. **Complete:** we ran the H100 full suite with 1,913 passes and 24 skips,
   including CLI goldens and Rust parity. The local full run separately passed
   1,737 tests with 200 skips and 4 warnings after the final cleanup patch.
2. **Blocked on artifact:** supply the real masked champion checkpoint and pass
   the bit-identical no-op gate. Do not replace this with synthetic weights.
3. **Complete:** we used the canonical four-GPU generation lifecycle smoke to
   verify one generator and EvalServer per GPU, disjoint seeds and outputs, CPU
   affinity, detached process ownership, and a clean stop.
4. **Complete:** the canonical four-GPU training path completed all five steps
   on a 21,120-row memmap corpus, wrote a 35,041,353-parameter masked checkpoint
   plus report/optimizer state, and cleaned the detached process group to idle.

### Gate 1 — H100 throughput frontier

The synthetic-checkpoint frontier is complete. Repeat the final
w128/batch96/request-collector/wait0/event0 recipe with the real champion and
identical seeds. Use `mp_queue`, strict `matmul_precision=highest`, no root
waves, and no CUDA Graphs; TF32 `high` was removed after the semantic A/B. Run
enough games to amortize startup and tail effects.

Measure complete games/hour, rows/hour, simulations/second, p50/p95 leaf
latency, effective neural batch, CPU/GPU utilization, power, fallback count,
truncation, and output-distribution drift. Use more than two games per worker and
do not extrapolate an unpaired 32-worker baseline.

### Gate 2 — spend compute on information

With the fastest certified configuration, compare equal GPU-hour data engines:

1. fresh n64 independent games;
2. n128 teacher games;
3. n64 plus 10% regret-state restarts;
4. n64 plus a small past-opponent mixture (after EvalServer compatibility is
   implemented or measured separately).

Train one-dose canonical L6/h640 35M controls at value weight 0.10 and
game-level validation. Cross per-game value weighting off/equal/sqrt. Promote
by powered paired H2H, neutral external panels, value calibration, and
population payoffs.

### Gate 3 — simulation and architecture upgrades

1. Powered n64-vs-n128 rollout-doubling gate.
2. Differentially certify Rust tree traversal before deployment; trivial and
   JSON-evaluator speedups are not neural semantic parity.
3. Test D6 throughput and H2H before re-gridding c-scale.
4. Compare 35M and 70–100M only after at least roughly 10M fresh diverse rows
   and stable value reuse. Do not begin with 150M or bundled architecture
   changes.

## Claim discipline

The current learned/search agent is a two-player, no-trade result. The declared
four-player full-trade benchmark is a separate milestone until its runner,
population evaluation, and powered external battery exist. “Best Catan bot”
must always name the ruleset and evaluation protocol.
