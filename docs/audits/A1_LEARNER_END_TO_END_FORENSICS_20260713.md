# A1 learner end-to-end forensics — 2026-07-13

## Decision

The n128/n256 data did not fail because stronger search is intrinsically bad,
and the 35M entity model is not yet shown to be too small. The dominant failed
experiments combined four avoidable learner/evaluation errors:

1. **Candidate chaining:** later candidates initialized from already-updated
   candidates instead of independently reloading f7.
2. **Oversized dose:** chained lineages consumed about 44.7M sampled rows and
   10,365 optimizer steps instead of one 4.19M-row dose.
3. **Wrong adjudication parent/operator:** apparent 52–55% internal wins were
   measured against old gen3 at `c_scale=0.03`, not against the initializer/f7
   incumbent at deployed `c_scale=0.10`.
4. **Target-distribution mismatch:** n128, n256, and replay stored policies have
   materially different entropy. Treating them at one distillation temperature
   distorted the teacher. Per-component temperatures
   `n128=1.00, n256=1.11, replay=0.52` produced the first decisive matched win
   (670–530/1200, 55.83%); a production replica retained 54.08%.

The working recipe is therefore an **independent f7-started, fresh-Adam,
one-dose TEMP learner**, not a chained curriculum. Architecture changes remain
diagnostic arms until they beat this corrected baseline under the same operator.

## Causal reconstruction of the failed lineage

```text
f7 champion
  └─ n256 candidate (large dose)
       └─ combined-196k candidate (another 31.9M sampled rows)
            └─ corrective n128 candidate (another 31.9M sampled rows)

reported comparison: candidate vs old gen3 @ c_scale .03
required comparison: candidate vs its actual initializer/f7 @ c_scale .10
```

This is not an independent n128-vs-n256 experiment. It compounds optimizer
updates, replay exposure, drift, and parent changes. The learner reports show
roughly 96–98% of update energy in the shared trunk and trunk drift from 8.96%
to 30.18%, while value-head drift was only about 2–5%. That is consistent with
over-updating a shared representation, not with a value head that simply needs
more epochs.

The independent n256 `lr=1.2e-4` arm did contain real signal: it beat f7
360–240/600 under the matched `c_scale=0.10` operator. That result rules out the
blanket claim that n256 data is harmful. The larger chained doses are what are
unsupported.

## What each stored row actually teaches

The learner intentionally separates policy and value support:

- Full-search, non-forced rows carry expensive search-policy supervision.
- Fast/forced rows have policy multiplier zero under the production recipe.
- Those same rows remain valid outcome/value rows when
  `forced_row_value_weight=1`.
- `forced_action_weight=0` is therefore not a missing-policy regression; it is
  the deliberate rule that prevents trivial forced actions from dominating the
  policy CE denominator.

The old loser downweight of 0.3 was harmful for this corpus: only 18.14% of
policy mass came from losing trajectories, starving the model of correction
signal. `loser_sample_weight=1` won the controlled comparison.

### Value target caveats

- `value_target_lambda=1` is currently correct. The stored `root_value` is the
  producer's stale self-bootstrap, not a stronger independently reanalysed
  target.
- Stored `target_scores` are standardized within each row. The current q head is
  interpreted on a return-like scale, so enabling q loss would train
  incompatible semantics. Keep q loss off until the head/target contract is
  redesigned.
- Training optimizes raw scalar MSE, but search consumes `tanh(raw_value)`. A
  matched tanh-vs-clip calibration probe is still required; offline raw MSE does
  not adjudicate the deployed value operator.

## Learner implementation audit

| Surface | Finding | Status |
|---|---|---|
| Objective diagnostics | Per-batch diagnostics silently executed two extra full shared-trunk `autograd.grad` passes. Historical throughput measurements included this work. | Fixed: explicit default-off interference cadence (`2b3afd8`). |
| Timed batch probe | Even after disabling interference, cadence-1 diagnostics cloned every trainable parameter before every optimizer step. | Fixed: timed geometry arms run both diagnostic cadences at zero and use cheap epoch aggregates (`f333921`, `2ba5ae1`). |
| Probe geometry | A purported matched microbatch test used gradient accumulation. Weighted task means were normalized independently per microbatch, so unequal policy/value support made the aggregate only approximate. | Fixed probe: compare 8x512 with 4x1024 at accumulation 1; both are exact global batch 4096 (`f333921`). General exact accumulation remains unresolved. |
| Zero-signal batches | AdamW could decay parameters or advance old momentum when the entire configured objective and global gradient were exactly zero. | Fixed: skip only exact zero-objective + zero-gradient groups (`6e952b1`, `1c6efe4`). |
| Non-finite gradients | `clip_grad_norm_` defaults to `error_if_nonfinite=False`; a finite loss followed by NaN/Inf gradients could corrupt Adam moments/checkpoints. | Fixed in both dense and entity trainers: abort before `optimizer.step` (`6e952b1`). |
| Optional heads | Zero-weight heads stayed trainable and were still subject to AdamW decay; some were forwarded unnecessarily. Requested head losses could silently target absent heads. | Fixed: fail preflight if requested head absent, freeze zero-weight heads, skip unused forwards (`e81ffb2`). |
| Halt head | `deliberation_halt_head` had no BC objective but remained trainable. | Fixed/frozen (`e81ffb2`). |
| Empty event history | All three current TEMP components authenticate all-zero event payloads, yet the model paid the full event MLP/memory cost. | Fixed authenticated crop (`aafe236`). This is objective-equivalent, but changes dropout RNG sequence versus historical runs. |
| DDP weighted mean | At accumulation 1, loss numerator gradients are scaled by the globally reduced denominator and DDP's gradient average correctly yields the global weighted mean. | Confirmed correct. |
| LR/max-step clock | A skipped optimizer group does not advance `global_step`; LR scheduling repeats the same step and max-step dose is not consumed. | Fixed/tested (`1c6efe4`). |
| Advantage weighting | The optional multiplier was normalized per rank; changing DDP geometry changed the objective, and empty-rank early return precluded a safe collective. | Fixed: all ranks participate in a globally weighted normalizer. |
| Decisive distributed modes | Gradient accumulation, distributed symmetry augmentation, and distributed outcome-value advantage did not yet have a sealed equivalence contract for a promotion-bearing A1 run. | Production execution now refuses these modes unless an explicit diagnostic/nondecisive authority is bound. DDP at accumulation 1 remains the sealed path (`28f42cf`). |
| Geometry GPU binding | The geometry launcher referenced `WORLD_SIZE` before defining it, so a true `--go` run failed before binding any GPU. | Fixed and covered by launch tests (`b59983b`). |
| Composite validation cap | A row-count validation cap can split a game and invalidate the signed game-disjoint validation sentinel. The first geometry command mistakenly requested 8,192 rows despite supplying the sentinel. | Planner and trainer now require `--validation-max-samples 0` for authenticated composites; the sentinel is the sole validation bound (`30b669f`). |
| Validation aggregation | Objective-matched validation now aggregates sufficient statistics; legacy raw `validation.loss` is a row-concatenated diagnostic and not promotion evidence. | Confirmed. |
| Head weight decay | Requested zero-weight optional heads previously changed despite no objective. | Fixed (`e81ffb2`). |

## Layer/architecture audit

### Shared trunk

The six-layer, width-640 transformer is where nearly all learner update energy
landed. More epochs or a higher LR therefore increases representation drift long
before it proves that the value head needs capacity. Fresh-optimizer, fixed-dose
arms are mandatory before interpreting any architecture result.

### Policy/action binding

The f7 policy scores a global state representation against an action embedding,
but it lacks a direct gather of the target vertex/edge state into the action
query. This is a plausible spatial-binding ceiling: two actions with similar
static encodings can require different board-local evidence, and the head asks
the shared CLS token to preserve all of it.

The correct architecture arm is a **zero-initialized, function-preserving target
gather**, independently initialized from f7 and trained for the same TEMP dose.
It must not be chained after another candidate. A gather win would show a binding
bottleneck; a loss would reject the mechanism without contaminating the baseline.

### Value readout

The value head itself moved much less than the trunk. Before scaling the network,
run these matched arms:

1. raw-MSE baseline evaluated through the deployed tanh operator;
2. calibrated scalar output with the same search/operator;
3. categorical/HL-Gauss value head at the same sample dose;
4. optional value-attention pooling, zero-init where possible.

Do not compare offline head loss alone. The decisive metric is candidate-vs-f7
under the same search budget, information regime, seats, and `c_scale`.

### Auxiliary heads

Auxiliary subgoal/belief/uncertainty heads exist, but old frozen-corpus results
do not establish a win. They also consume dropout RNG and can change later trunk
masks even when the shared first forward is identical. Each arm needs an
independent f7 start and equal dose. Zero-weight heads are now frozen so baseline
runs no longer pay or drift them.

### Event history

The architecture accepts event history, but the current corpora contain only
authenticated zeros. That is a representation/data ceiling, not evidence that
the event encoder is bad. The learner now crops this dead path for current data.
A future nonzero event schema must re-enable it and receive a separate feature
parity/no-op audit.

### Public-award feature contract

Every historical TEMP component has player-token slot 12 (longest-road public
ownership) identically zero. That makes the absence of this feature part of the
legacy corpus/checkpoint contract, even though the authoritative game state can
populate it. Enabling the corrected value shifts f7 outputs and therefore is not
a no-op feature toggle.

The P0 reproduction below remains explicitly **legacy-corpus / legacy-feature**.
A future authoritative-v1 run must bind producer and memmap provenance to that
contract and deterministically zero-initialize the new input column in the f7
checkpoint before constructing DDP/FSDP or the optimizer. Mixed legacy and
authoritative payloads must fail closed unless a separately reviewed migration
operator exists.

## Distributed-training semantics

### Safe now

- DDP, accumulation 1.
- Global weighted task means.
- Rank-offset torch RNG for independent dropout streams.
- 8x512 and 4x1024 geometry at the same global batch 4096.

### Not safe for decisive comparison yet

- **Gradient accumulation >1:** task numerators are divided by each
  microbatch's denominator before accumulation. If support/weights differ, the
  result is a mean of means, not the union mean. Do not use it for a decisive
  learner arm until raw per-objective numerators/denominators are accumulated
  exactly.
- **Symmetry augmentation:** the rank-offset torch RNG does not automatically
  make the separate symmetry RNG rank-independent, and only rank-0 symmetry RNG
  state is saved. Symmetry is off in the winning recipe. Fix/bind its distributed
  stream before enabling it in production.
- **Outcome-value advantage:** global normalization is now mathematically
  invariant to empty/nonempty rank partitions, but the corrected objective has
  not yet been resealed against the A1 baseline. It is allowed for explicitly
  nondecisive diagnostics and refused for production comparison until resealed.

## Corrected experimental program (Pareto order)

### P0 — preserve the known win

Reproduce TEMP from f7 with:

- fresh Adam, no optimizer restore;
- 4,194,304 sampled rows / 1,024 optimizer steps;
- global batch 4,096;
- flat `lr=3e-5`, 100-step warmup;
- policy/value weights 1.0/0.25;
- soft policy weight 0.9;
- loser/winner weights 1/1;
- forced policy/value weights 0/1;
- component temperatures n128 1.00, n256 1.11, replay 0.52;
- q, KL-anchor, categorical, aux, belief, uncertainty losses off;
- matched evaluation against f7 at the deployed search operator.

### P1 — choose systems geometry without changing learning

Run sequential, isolated arms:

- 8 ranks × local batch 512 × accumulation 1;
- 4 ranks × local batch 1024 × accumulation 1.

Both have global batch 4096, the same optimizer steps, sample dose, warmup rows,
LR trajectory, objective, and initializer. Heavy diagnostics are disabled in the
timed arms. This chooses throughput/CPU-I/O geometry; it is not a model-quality
sweep.

The 128-step B200 comparison selected **8x512**:

| geometry | rows | elapsed | rows/s | active teacher-gap closure | worst component closure | preclip mean/max | clipped steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8x512 | 524,288 | 198.347 s | 2,643.28 | 0.102290 | 0.076152 | 0.6203 / 1.0077 | 1/128 |
| 4x1024 | 524,288 | 268.385 s | 1,953.49 | 0.102206 | 0.075848 | 0.6221 / 0.9966 | 0/128 |

The four-rank arm delivered 73.90% of the eight-rank throughput while its
equal-dose closure differed by only -0.000161 per million samples. Higher HBM
occupancy therefore did not buy learning or wall-clock efficiency. The full P0
reproduction uses 8x512, global batch 4,096, accumulation 1.

The live rehearsal also found three fail-fast tooling defects before the full
dose: an undefined GPU-binding constant (`b59983b`), an invalid row-capped
authenticated validation plan (`30b669f`), and a post-run plan schema that
omitted the per-run LR consumed by the summarizer (`84c12e9`). Both geometry
trainers completed successfully; the last defect affected postprocessing only.

### P2 — highest-information learner arms

Every arm independently reloads f7 and consumes one identical dose:

1. TEMP baseline reproduction;
2. zero-init target gather;
3. scalar value calibration/operator alignment;
4. categorical value head;
5. one auxiliary-head bundle only after its requested targets are proven present.

Promote nothing from offline loss. First use a short matched internal panel, then
the full seat-swapped neutral gate for survivors.

### P3 — only after the proximal mechanisms are exhausted

- grow depth/width from the same f7 checkpoint with a function-preserving warm
  start where possible;
- introduce nonzero event history;
- exact gradient accumulation;
- globally normalized advantage weighting;
- opponent/reanalysis changes in the next data wave.

## Ruled out or not yet supported

- “n128/n256 data is bad”: ruled out by independent matched wins.
- “just train longer”: contradicted by chained-dose drift and corrected one-dose
  success.
- “just raise LR”: LR helped one independent arm, but chained high-dose runs
  confound LR with lineage/dose; no universal LR conclusion follows.
- “35M is too small”: not established. The same capacity produced f7 and the TEMP
  win; binding/operator issues are more proximal.
- “GPU memory must be filled”: false objective. TEMP used about 32.4 GiB/GPU at
  8x512 FP32; throughput and learning per wall/sample, not HBM occupancy, choose
  geometry.

## Evidence/implementation commits

- `aafe236` — authenticated empty-event fast path.
- `2b3afd8` — separate objective-gradient interference from normal diagnostics.
- `e81ffb2` — align trainable heads with requested objectives.
- `6e952b1` — fail closed before non-finite or signal-free optimizer updates.
- `1c6efe4` — accumulation-boundary and global-step accounting tests.
- `f333921` — exact 8x512-vs-4x1024 geometry probe, no accumulation confound.
- `2ba5ae1` — clean aggregate telemetry and dedicated-host GPU ownership support.
- `3bcad3c` — globally normalize optional outcome-value advantage across DDP.
- `b59983b` — define and bind selected geometry-probe GPU ranks before launch.
- `28f42cf` — fail closed on distributed semantics that lack a decisive A1 seal.
- `30b669f` — reject row-capped validation for authenticated composites.
- `84c12e9` — bind matched LR into every geometry run and its summary.

The immediate criterion is simple: preserve the independent TEMP win, select the
fastest mathematically matched DDP geometry, and spend subsequent B200 time only
on arms that isolate one causal mechanism.
