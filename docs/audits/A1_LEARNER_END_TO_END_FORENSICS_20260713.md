# A1 learner end-to-end forensics — 2026-07-13

## Decision

The n128/n256 data did not fail because stronger search is intrinsically bad,
and the 35M entity model is not yet shown to be too small. The dominant failed
experiments combined four avoidable learner/evaluation errors:

1. **Candidate chaining:** later candidates initialized from already-updated
   candidates instead of independently reloading f7.
2. **Oversized dose:** the actual checkpoint/launcher lineage shows each failed
   chain consumed 42.46M scalar-target examples and 10,365 optimizer steps
   instead of one 4.19M-row dose.
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
gen3
  └─ f7
      └─ n256-early (2,962 steps / 12.13M scalar-target examples)
          └─ combined-196k (+7,403 steps / +30.32M examples)

f7
  └─ corrective n256 (2,962 steps / 12.13M examples at lr=1.2e-4)
      └─ corrective n128 (+7,403 steps / +30.32M examples at lr=1.2e-4)

reported comparison: candidate vs old gen3 @ c_scale .03
required comparison: candidate vs its actual initializer/f7 @ c_scale .10
```

This is not an independent n128-vs-n256 experiment. It compounds optimizer
updates, replay exposure, drift, and parent changes. The learner reports show
more than 98% of update energy outside the dedicated value head. Exact tensor
comparison against f7 gives:

| Checkpoint | Global relative drift | Global cosine | Value-head drift |
|---|---:|---:|---:|
| P0 midpoint | 0.691% | 0.999976 | 0.408% |
| TEMP full | 2.598% | 0.999663 | 1.544% |
| replay anchor | 2.652% | 0.999648 | 1.628% |
| n256-early | 5.167% | 0.998668 | 3.304% |
| combined-196k | 9.763% | 0.995273 | 6.760% |
| corrective n256 | 15.313% | 0.988578 | 7.189% |
| corrective n128 | 34.129% | 0.948453 | 13.935% |

The corrective-n256 to corrective-n128 step alone moved 26.09%. Transformer
block 0 absorbed 36.8% of that step's energy; its attention input projection
alone absorbed 29.1%, moved 75.2% relative to corrective n256, and grew in norm
from 36.88 to 59.47. Every tensor remained finite. This is severe shared-trunk
deformation from the chained high-LR dose, not numerical corruption or a value
head that merely needs more epochs.

All audited artifacts are exactly compatible 35,041,353-parameter, six-layer,
width-640 scalar-value models. Their effective policy logit scales range only
from 4.226 to 5.526 against a clamp of 50, and no fixed-sample output reached
`abs(tanh(value)) >= 0.95`. Architecture mismatch, policy-logit saturation, and
value-head explosion are therefore ruled out for this failure.

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
- Training optimizes raw scalar MSE, but search consumes `tanh(raw_value)`. The
  calibration tool now fits a positive tanh scale on one explicit held-out game
  subset and scores it on a disjoint held-out subset without mutating the
  operator. Offline raw MSE still cannot adjudicate playing strength and can
  reverse a ranking even when both agents later share the same search operator:
  for target `+1`, raw
  predictions `0.9` and `1.2` have MSE `0.01` and `0.04`, but after tanh their
  squared errors are about `0.0807` and `0.0276`. Dose adjudication therefore
  uses deployed tanh as primary and repeats the same short/full/f7 seeds with
  clip only as an operator-sensitivity diagnostic.

An exploratory identical-row n256 panel (104 games split 52/52 for scale fit
and evaluation; training-adjacent, not promotion evidence) selected scales
0.968 for f7, 1.067 for both P0 midpoint and full TEMP, and 1.215 for the damaged
corrective n128 artifact. The fitted f7 scale regressed held-game RMSE; midpoint
slightly beat full TEMP on held-game value RMSE/correlation despite using one
eighth the samples; corrective n128 calibrated best after scaling despite its
severe trunk damage. This is direct evidence that offline value calibration is
a diagnostic, not a champion selector. Search H2H remains mandatory.

## Learner implementation audit

| Surface | Finding | Status |
|---|---|---|
| Objective diagnostics | Per-batch diagnostics silently executed two extra full shared-trunk `autograd.grad` passes. Historical throughput measurements included this work. | Fixed: explicit default-off interference cadence (`2b3afd8`). |
| Timed batch probe | Even after disabling interference, cadence-1 diagnostics cloned every trainable parameter before every optimizer step. | Fixed: timed geometry arms run both diagnostic cadences at zero and use cheap epoch aggregates (`f333921`, `2ba5ae1`). |
| Probe geometry | A purported matched microbatch test used gradient accumulation. Weighted task means were normalized independently per microbatch, so unequal policy/value support made the aggregate only approximate. | Fixed probe: compare 8x512 with 4x1024 at accumulation 1; both are exact global batch 4096 (`f333921`). General exact accumulation remains unresolved. |
| Zero-signal batches | AdamW could decay parameters or advance old momentum when the entire configured objective and global gradient were exactly zero. | Fixed: skip only exact zero-objective + zero-gradient groups (`6e952b1`, `1c6efe4`). |
| Non-finite gradients | `clip_grad_norm_` defaults to `error_if_nonfinite=False`; a finite loss followed by NaN/Inf gradients could corrupt Adam moments/checkpoints. | Fixed in both dense and entity trainers: abort before `optimizer.step` (`6e952b1`). |
| Optional heads | Zero-weight heads stayed trainable and were subject to AdamW decay. The first fix froze them, but enabled frozen aux heads still executed Dropout and advanced the RNG, so an aux-ON/zero-loss control diverged from aux-OFF on the second main forward. | Fixed: fail-closed preflight/freeze plus a nonpersistent inactive-output gate skips those forwards; active/inference outputs are unchanged and two-forward RNG parity is tested (`e81ffb2`, `03bf5e2`). Baseline f7/TEMP configs had these heads absent, so this invalidated optional-head controls but did not cause the baseline failure. |
| Value-only policy surface | `--train-value-only` froze the legacy trunk/action-encoder/policy-head groups but left upgraded target-gather, edge-policy, and action cross-attention adapters optimizer-visible. A nominal value-only arm could therefore alter policy behavior; under AdamW, graph-reachable exact-zero policy gradients can still trigger decoupled decay. | Fixed: the shortcut now freezes the complete named policy surface, and an upgraded-architecture optimizer smoke proves every adapter remains unchanged while value readouts move. Historical f7/P0/TEMP did not contain these adapters, so their weights are unaffected; upgraded value-only arms from the old code are not causally isolated. |
| Target-aware batch padding | The PPO/entity batcher padded `legal_action_target_ids` with zero. Zero is a valid local entity id, not the no-target sentinel, so any variable-legal-width batch presented to target-gather, edge-policy, or relational target-aware models failed pre-forward with `padded legal action carries a target id`. | Fixed: padding now uses `-1`, matching feature extraction, inference batching, and the model contract. A mixed-width upgraded-architecture value-only smoke reaches the real forward and optimizer step. Legacy f7/P0/TEMP did not consume action target ids and are unaffected; upgraded learner paths were nonfunctional on ordinary mixed-width batches. |
| Halt head | `deliberation_halt_head` had no BC objective but remained trainable. | Fixed/frozen (`e81ffb2`). |
| Empty event history | All three current TEMP components authenticate all-zero event payloads, yet the model paid the full event MLP/memory cost. | Fixed authenticated crop (`aafe236`). This is objective-equivalent, but changes dropout RNG sequence versus historical runs. |
| DDP weighted mean | At accumulation 1, loss numerator gradients are scaled by the globally reduced denominator and DDP's gradient average correctly yields the global weighted mean. | Confirmed correct. |
| LR/max-step clock | A skipped optimizer group does not advance `global_step`; LR scheduling repeats the same step and max-step dose is not consumed. | Fixed/tested (`1c6efe4`). |
| Resume recipe identity | The typed training config omitted trajectory-changing precision, topology, sampling, and objective fields: AMP/fused optimizer, gradient accumulation/DDP-shard/FSDP mode, graph-history schema, teacher/phase/value-phase weights, Q teacher mask, and root-value blend scope. A checkpoint could therefore restore Adam moments, RNG, and the LR/max-step clock while changing the actual learner and still pass the recipe digest. | Fixed: schema v11 binds every listed field in both the standalone config hash and resume identity; focused tests mutate each gradient/precision field and prove identity divergence. P0/TEMP used fresh Adam with optimizer restore disabled, so their weights are unaffected; any older resumed ablation lacking an external exact argv receipt is not a sealed causal result. |
| Grow-checkpoint continuation | A run started with `--grow-from-checkpoint` saved optimizer/progress sidecars whose identity retained the grow source. Continuation must use the mutually exclusive `--init-checkpoint` path, so the expected identity could never match and valid grown runs were not resumable. | Fixed: both checkpoint selectors are normalized out of continuation identity while model bytes remain hash-bound; all optimizer/objective/schedule fields remain exact. |
| Sharded-DDP sampler resume | Progress saved only rank 0's NumPy epoch sampler state. With `--ddp-shard-data`, ranks permute different local corpora and can advance their generators differently; resume reset every rank to rank 0's stream. | Fixed: new progress commits gather and restore per-rank NumPy RNG states. Legacy multi-rank sharded checkpoints without them fail closed; shared-global-corpus legacy DDP retains its exact single-stream fallback. P0/TEMP did not use sharded-data resume. |
| DDP zero-objective step consensus | The zero-signal guard combined a globally synchronized gradient norm with a rank-local scalar loss. At an exact zero-gradient sparse/stationary step, an empty rank could skip while a peer with a nonzero objective advanced Adam/AdamW, immediately diverging optimizer state and decoupled weight decay across replicas. | Fixed: only the rare zero-gradient branch collectively resolves whether any rank has objective mass; all ranks then step or skip together. Nonzero-gradient hot steps add no collective. P0/TEMP have active base objectives and are unaffected. |
| Advantage weighting | The optional multiplier was normalized per rank; changing DDP geometry changed the objective, and empty-rank early return precluded a safe collective. | Fixed: all ranks participate in a globally weighted normalizer. |
| Decisive distributed modes | Gradient accumulation and distributed outcome-value advantage still lack a sealed equivalence contract. Distributed symmetry had a separate concrete defect: every rank constructed `default_rng(seed+20260705)`, so the same 512-orientation pattern repeated across all eight local batches, and progress saved only rank 0's augmentation stream. | Symmetry is fixed: its deterministic `SeedSequence` binds `(seed, world_size, rank)`, all rank streams are gathered into a versioned progress envelope, exact rank-local resume is tested, and legacy/malformed multi-rank resumes fail closed (`83ad050`). Composite diagnostics now report the same semantics. Gradient accumulation and distributed advantage remain diagnostic-only; accumulation-1 DDP remains the sealed base path. |
| Geometry GPU binding | The geometry launcher referenced `WORLD_SIZE` before defining it, so a true `--go` run failed before binding any GPU. | Fixed and covered by launch tests (`b59983b`). |
| Composite validation cap | A row-count validation cap can split a game and invalidate the signed game-disjoint validation sentinel. The first geometry command mistakenly requested 8,192 rows despite supplying the sentinel. | Planner and trainer now require `--validation-max-samples 0` for authenticated composites; the sentinel is the sole validation bound (`30b669f`). |
| Validation aggregation | Objective-matched validation now aggregates sufficient statistics; legacy raw `validation.loss` is a row-concatenated diagnostic and not promotion evidence. | Confirmed. |
| Posthoc composite validation | The standalone teacher-gap probe accepted only one memmap directory, even though the selected A1 learner consumes an authenticated `memmap_composite_v2` descriptor. Pointing it at the real descriptor aborted as “not a memmap corpus”; pointing it at one component would have measured the wrong population. It also omitted authenticated policy/value component scopes, policy-KL direction, belief loss, value-root blend, and the report-bound matmul mode. | Fixed: authenticate and load the exact composite, replay its policy/value scopes and objective arguments, evaluate the raw concatenation only as a compatibility diagnostic, and report teacher-gap closure from the same component→game→row objective measure used by training. Single-corpus behavior remains unchanged. |
| Head weight decay | Requested zero-weight optional heads previously changed despite no objective. | Fixed (`e81ffb2`). |
| Composite per-game weighting | Numeric `game_seed` values were treated as globally unique. The same seed in two corpus components was merged into one game for equal/sqrt weighting and quality telemetry. | Fixed: game identity is now `(component, game_seed)` and component offsets are validated (`cf54d5a`). |
| Adjacent duplicate-game exposure | Pre-wave auditing and corpus conversion treated one maximal run of equal `game_seed` values as one game. They caught a seed that reappeared after another seed, but two byte-for-byte or independently regenerated copies placed directly back-to-back never changed seed and were silently merged. That could double one trajectory's sampling mass while acceptance, ordinary conversion, and selected-manifest conversion reported no duplicate. | Fixed: current pre-wave acceptance and both conversion trackers reject a non-increasing `decision_index` within the same seed, including across shard boundaries; legacy conversion without that field retains the seed-run check. Regressions cover adjacent reset, cross-shard reset, valid monotonic continuation, and the selected-source path. Existing P0/TEMP artifact impact is unproven: their prior attestations did not test this exact adjacency class, so do not retrospectively claim either contamination or absence from the old audit alone. |
| Entity-adapter provenance | Rust generation and entity conversion wrote `adapter_version`, but NPZ normalization omitted it and memmap conversion therefore dropped it. Same-shaped rows from different feature semantics could be mixed without learner admission seeing the mismatch; the data-quality report also could not expose the loss. Checkpoints also authenticated only shapes/config, so a same-shaped runtime adapter change could silently alter input meanings. | Fixed end-to-end: normalization, in-memory loading, memmap conversion, lazy quality counts, and schema admission preserve the row field; checkpoints now append an independent `entity_feature_adapter` semantic contract; single/DDP/FSDP save paths agree; EMA/interpolation normalize the one explicit legacy mapping; and Rust evaluator construction requires exact checkpoint/runtime agreement. Mixed known/unknown rows, multiple known versions, malformed/unknown checkpoint metadata, and runtime mismatch fail closed. All-missing legacy corpora remain admissible. Pre-metadata checkpoints (including deployed f7/gen3, verified directly) map explicitly to pinned historical v2 rather than whatever a future current default becomes. |
| DDP active-fraction telemetry | The active numerator was globally reduced but divided by a rank-local denominator, producing impossible fractions such as `7.95` in an eight-rank report. This did not alter gradients, but it corrupted experimental interpretation. | Fixed: numerator and denominator now share global scope and bounded fractions fail closed (`cf54d5a`). |
| Diagnostic run receipts | Completed non-promotable runs could retain a launch plan without a final identity binding for checkpoint, report, runtime, optimizer, source code, and finalizer. | Fixed: deterministic finalization/replay receipt binds all run artifacts and the finalizer itself (`efcc94b`, `d9bf335`). |
| Shared-trunk gradient probe | The probe enabled ordinary diagnostics but had drifted from the separately gated objective-interference cadence, so it could start without emitting its defining measurement. | Fixed: both diagnostic cadences are explicit and tested (`58fb7e6`). |
| Post-P1 causal-arm planner | Every arm silently inherited the historical 4.19M-row dose despite the matched saturation result, and the planner specified BF16 even though the sealed TEMP baseline and both dose artifacts are FP32. That made a supposed one-axis arm a dose-assumption plus precision change. | Fixed: the existing 0.52M/full checkpoints must select the Pareto dose before any new arm; FP32 and all three component temperatures are now explicit and bound (`a1_post_p1_diagnosis_plan.py`, schema v2). |
| Learner/operator adjudication | The post-P1 planner still proposed tuning each candidate at `c_scale in {0.03, 0.10}` before the external panel. That repeats the historical failure mode: checkpoint ancestry and search operator change together, and old gen3 can make an updated checkpoint look improved relative to its actual f7 parent. | Fixed in planner schema v4: every learner arm is compared with exact f7 at the same deployed `c_scale=0.10`; candidate-specific operator tuning is forbidden for selection and moved to a separate same-checkpoint crossover diagnostic. The older recovery prose is aligned with the FP32 and dose-selection contracts. |
| Objective-matched validation loss | The sufficient-statistic registries omitted `belief_resource_loss`, while every evaluator coefficient map includes it even at zero weight. Exact total-loss reconstruction was therefore never reached: a sparse-policy example reported `5.25` instead of the configured objective `0.3490`. Individual reconstructed policy/value losses and teacher-gap closure remained valid, but aggregate validation loss could mis-rank arms. | Fixed: reconstruct every objective term, then derive total/scalar/primary aliases from the same measure; regression covers the 5.25-to-0.3490 failure. Training gradients/checkpoints were unaffected. |
| Auxiliary validation measures | Value-uncertainty and optional auxiliary/MoE validation used batch means or fabricated row denominators rather than exact per-head numerators and valid-label counts. Composite/DDP aggregation could therefore change the reported objective and silently rank sparse-head arms incorrectly. | Fixed: every active head now emits exact sufficient statistics; nonzero objectives fail closed when the measure or coefficient is missing/inconsistent; zero-weight non-MoE heads remain exact zero (`f7b2064`). Training gradients/checkpoints were unaffected. |
| Composite-v2 per-game weighting | V2 already samples `component -> game -> row`. The loss normalizer still divided by summed in-game weight, the formula for uniform-row sampling, so enabling equal/sqrt per-game weighting applied a second inverse-length correction. A 1-row and 4-row game went from 50/50 sampling mass to 80/20 (`equal`) or 67/33 (`sqrt`). | Fixed: v2 normalizes mean in-game weight; v1/ordinary uniform-row behavior stays unchanged. Telemetry now reports sampler-adjusted game mass so raw row totals cannot be mistaken for the optimized measure. P0/TEMP had both flags off and was unaffected. |
| Prefetched KL-anchor scope | The sealed memmap loader uses two threaded prefetch workers. Materialization converted the authenticated composite into a plain dictionary and preserved source temperatures, but dropped `policy_kl_anchor_component_indices` and its authenticated row lookup. A nonzero replay-only KL arm therefore anchored every prior-bearing component during both training and validation instead of only the descriptor-selected replay component. | Fixed: resolve the authenticated row mask before materialization, carry it as a private batch column, and use one fail-closed scope resolver in synchronous/prefetched training and validation (`f78fe81`). P0/TEMP use `policy_kl_anchor_weight=0` and are unaffected. Any nonzero-anchor result produced through the old threaded composite path is not a valid replay-only causal result. The historical checkpoint labeled “replay anchor” has no surviving report/receipt, so its exact loader/weight cannot be attested; its tensor-drift measurement remains factual, but it must not be cited as anchor efficacy without reconstruction or rerun. |
| Sparse DDP objective collectives | KL anchoring returned early when one rank's local batch had no prior/eligible row, while a peer entered the global-denominator all-reduce. Q-score loss did the same when a rank had fewer than two scored legal actions. These rank-local branches could deadlock a DDP step or make mocked/distributed comparisons follow different objective paths. | Fixed: gradient-enabled KL term presence is decided collectively, locally empty ranks contribute graph-connected zero numerators, and Q-score ranks always enter the shared weighted-mean reduction (`2761a1a`). A two-rank Gloo regression places the only eligible KL/Q row on rank 1 and proves both ranks finish with the expected local denominators and gradients. P0/TEMP have both coefficients zero and are unaffected; pre-fix K3/K10-style anchor runs or any nonzero-Q DDP run require rerun. All other sparse masked objectives were traced and already enter their collectives on empty local masks. |
| Replay objective scope | A replay `z` is conditional on the old policy's continuation, while its stored search policy is an older teacher. An initial planner edit removed both at once and mislabeled that two-axis treatment as the TEMP control. The known winning TEMP artifact actually trained policy and value on all three components. | Fixed in the causal plan: `TEMP_CONTROL` exactly preserves all-component policy/value scope; `CURRENT_POLICY_SCOPE` and `CURRENT_VALUE_SCOPE` each change one axis. A both-current interaction is forbidden unless both independent arms survive. |
| Scalar search-value selection | Validation ranked raw scalar MSE although MCTS consumes `tanh(raw * value_scale)`; these rankings can reverse. | Fixed diagnostic: fit scale only on one explicit held-out game subset, score on a disjoint subset with bounded memory, and never mutate the operator or authorize promotion without matched search H2H (`5838cec`). |
| Evaluator cache concurrency | `evaluate_many` performed LRU get/touch as separate unlocked operations while async stores could evict the key, producing a deterministic `KeyError` under capacity pressure. | Fixed: atomic lock-scoped LRU get/store across sync, batch, and async evaluators; model forward stays outside the lock; root-perspective and deterministic eviction races are covered (`e1ae5bf`). This affected evaluation reliability, not trained weights. |
| Historical candidate provenance | P0/TEMP/anchor/combined/corrective checkpoint files have no adjacent report, receipt, or optimizer sidecar on the audited host, and historical payloads omit a standardized initializer hash/recipe/code digest. | Future artifacts must bind initializer, fresh/resumed optimizer state, exact dose and integrated LR area, source/runtime digests, report, receipt, and finalizer. True historical lineage was reconstructed from launchers/checkpoint hashes rather than inferred from filenames. |

## Layer/architecture audit

### Shared trunk

The six-layer, width-640 transformer is where nearly all learner update energy
landed. More epochs or a higher LR therefore increases representation drift long
before it proves that the value head needs capacity. Fresh-optimizer, fixed-dose
arms are mandatory before interpreting any architecture result.

### Policy/action binding

The f7 policy scores a global state representation against an action embedding,
but it lacks a direct gather of the target vertex/edge state into the action
query. This is an exact information alias, not merely a capacity suspicion:

- vertex tokens contain no vertex ID or coordinate;
- edge tokens contain no edge ID or endpoint IDs;
- the incumbent Transformer adds only a shared type embedding and consumes none
  of `hex_vertex_ids`, `hex_edge_ids`, or `edge_vertex_ids`;
- legal actions carry their identity only as one normalized fp16 scalar plus
  semantic one-hots and 18 handcrafted context values. All 607 scalar IDs are
  distinct, so this is a poor one-dimensional inductive bias rather than a
  literal ID collision, but it does not bind an action to its board token.

Consequently, arbitrary within-type permutations of the 54 vertex rows and 72
edge rows leave f7 logits and value unchanged (apart from approximately `1e-7`
attention-reduction roundoff) when legal target IDs remain fixed. Connected and
disconnected road layouts or spatially distinct occupancy states can therefore
map to the same representation. The executable regression is
`tests/test_entity_graph_representation_aliasing.py`.

There is a second exact join failure in the player surface. Player tokens carry
actor/current flags but no absolute or actor-relative seat identity. The global
token knows the actor/current color and board tokens use fixed-color ownership,
but the trunk cannot bind each non-actor player's remaining stats to that
player's board pieces. Permuting the three non-actor/non-current player rows is
also function-invariant. This matters for relative turn order, opponent strength,
and trade/robber decisions; it is not repaired by simply widening the same
set-Transformer.

The correct architecture arm is a **zero-initialized, function-preserving target
gather**, independently initialized from f7 and trained for the same TEMP dose.
It must not be chained after another candidate. A gather win would show a binding
bottleneck; a loss would reject the mechanism without contaminating the baseline.
The regression proves that a learned nonzero gather breaks the action-local
vertex/edge alias while leaving the CLS/value alias intact. A separate zero-output
topology residual is therefore the function-preserving state-side treatment; do
not bundle it into the gather causal arm. Follow-up identity treatments, also
without changing f7 at initialization, are zero-initialized actor-relative seat
embeddings and a zero-initialized categorical action-ID residual. These remain
lower priority than target gather/topology because the current action scalar is
injective and seat identity has not yet been isolated by matched behavior.

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
do not establish a win. Frozen zero-objective aux heads previously still consumed
dropout RNG and changed later trunk masks even when the shared first forward was
identical. They are now skipped by a nonpersistent trainer gate; active heads and
normal inference retain their full output API. Each arm still needs an independent
f7 start and equal dose.

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

The same conclusion applies to the legacy Rust adapter's trade surface. Its v2
contract also pins the case-sensitive misses in trade action/prompt one-hots,
zero `offers_remaining`/`current_offer`, legacy maritime list-cardinality totals,
BASE-layout topology lookup, and empty event history. Python and native Rust
currently reproduce those omissions intentionally for parity. Correcting any of
them underneath v2 would make the version string a lie and feed old weights new
slot meanings. The append-only registry therefore requires a separately named
v3 (or later) adapter, versioned shards, and a retrained/migrated checkpoint;
there is no silent-fix path.

This creates a **double-blind longest-road surface**, not merely one missing
boolean. `entity_token_features._edge_tokens` records road owner and whether the
owner is the actor, but records no edge endpoints. The legacy Transformer does
not consume `edge_vertex_ids`, and the permutation regression above proves that
rearranging all edge rows leaves its function unchanged. It can count an
owner's road tokens, but it cannot distinguish one connected chain from the same
roads split across components. With slot 12 also constant-zero, it has neither
the authoritative public award bit nor enough topology to reconstruct the award
or reason exactly about a road action stealing/defending its two VP. Public VP
totals may provide a lossy correlate; they do not restore road connectivity or
the action-local counterfactual.

This is a representation/data ceiling for the next authoritative-feature arm,
not an explanation for the recent same-representation regressions: f7, TEMP,
combined-196k, and the corrective candidates all shared the same blind surface.
Those failures still localize first to lineage, dose, target calibration, and
evaluation binding. A future repair must be a separately named f7-start arm with
the input-column initialization and topology treatment explicitly bound; it
must not be silently mixed into the current TEMP control.

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

### Full P0 reproduction and dose saturation

The full independent TEMP reproduction completed on one eight-B200 NVLink host.
It reloaded the authenticated f7 checkpoint, created fresh Adam state, used
8x512/global-batch 4096, and consumed exactly 1,024 optimizer steps / 4,194,304
row draws. The sealed run produced:

- checkpoint SHA-256
  `ce29663fe519b88537d54afec3dfa4e0033f79a649f8b04d364baead48c462f4`;
- report SHA-256
  `4dbfa0b28156d482eae9f01e3a80bf450e0fb6d71f1e2dc4495293658d8779de`;
- receipt file SHA-256
  `2333caed6178450a27bdd9cffafd98f9ea1dbca5c16a973e3692458a23eb225b`;
- semantic receipt digest
  `sha256:2adc5973e2dae15d2208bfd031aeeb82d0699c44bb6c110a13f9600e56f25d38`;
- 586.87 seconds trainer time, 44/1,024 clipped steps, no zero-objective
  steps, and no non-finite failure;
- objective-matched validation teacher-gap closure 0.135757, with component
  closures replay=0.212590, n128=0.123271, and n256=0.108810;
- global relative parameter drift 2.5954% and cosine 0.999663; the six trunk
  blocks drifted from 2.49% to 3.32%, while the value head drifted 1.54%.

The matched 128-step checkpoint is not merely a systems rehearsal; it reveals a
strong dose-saturation mechanism. At 524,288 row draws it already reached
teacher-gap closure 0.102290 with global relative drift 0.6913%. The full run
used 8x more samples but gained only 0.03347 absolute closure (1.33x total)
while parameter drift grew 3.75x. Closure per million samples collapsed from
0.1951 to 0.03237, roughly 6x.

The optimizer exposure contrast is stronger than the row-count contrast. With
the trainer's 100-step linear warmup, the 128-step run integrated only 78.5
full-LR-equivalent steps, whereas the 1,024-step run integrated 974.5. The full
run therefore received about **12.41x** the LR-area exposure yet achieved only
1.33x the teacher-gap closure. This is direct evidence of early policy-target
saturation followed by continued representation movement; it is not evidence
that the short checkpoint is competitively stronger, which still requires the
matched behavior panel.

Offline distillation therefore saturates far earlier than representation
movement. The matched behavior screens now decide whether that extra movement
buys strength. Against
exact f7 at the matched deployed operator, the 524,288-row checkpoint scored
`75-53` over 128 games (`58.59375%`), with 64 complete seat-swapped pairs,
pentanomial counts `WW=17, split=41, LL=6`, and zero truncations/errors. The
superiority pentanomial LLR is `1.177`, so this short screen is encouraging but
correctly remains `continue`, not promotion evidence. Its pooled artifact is
`/home/ubuntu/experimental_nonpromotable/p0-temp-midpoint-v-f7-screen-20260713-r1/collected/a1-eval-d801f3ef6377fcad/pooled/internal.json`
on the evaluation controller.

The 4,194,304-row checkpoint then scored only `65-63` over the **same 128
`(game_seed, orientation)` keys** (`50.78125%`), with 64 complete pairs and zero
truncations/errors. Pairing the two candidate screens game-for-game gives 41
games both candidates won, 29 both lost, 34 won only by the midpoint, and 24
won only by the full-dose candidate: an observed midpoint advantage of 10
games, or 7.8125 percentage points. The discordant count is not large enough
to claim a statistically certain negative dose-response; it is, however,
decisive for the preregistered Pareto rule. The full dose is not more than two
percentage points stronger--it is not stronger at all in this screen--so there
is no behavioral return for its 8x samples, 12.41x LR-area, and 3.75x parameter
drift.

**Selected learner dose: 524,288 global row draws / 128 optimizer steps at
8x512.** This is an experimental dose selection, not a promotion claim. Every
next one-axis arm must independently reload exact f7, use the identical sampled
rows/order and selected dose, and evaluate against exact f7 under the same
deployed search operator.

### P2 — highest-information learner arms

The matched screen selected the 524,288-row / 128-step dose above. Offline loss
cannot authorize continuation to the full dose. Every arm independently reloads
f7 and consumes that one selected identical dose:

1. TEMP baseline reproduction;
2. zero-init target gather;
3. scalar value calibration/operator alignment;
4. categorical value head;
5. one auxiliary-head bundle only after its requested targets are proven present.

Fresh modules require a commissioning schedule, not the mature TEMP schedule by
rote. With 100 warmup steps, the selected 128-step run provides only 78.5
full-LR-equivalent updates; a fresh value head at `value_lr_mult=0.3` receives
23.55 head-LR equivalents. The target-gather screen therefore preserves the
same 524,288 rows but uses 8x64/global-512 for 1,024 optimizer steps, freezes
every mature surface, and trains only the zero-init gather at action LR x4.
Pure-soft remains an exact 128-step one-axis arm (`0.9 -> 1.0`). Stale launchers
that coupled fresh heads to the rejected 4.19M-row dose now fail closed; see
`A1_SHORT_DOSE_MODULE_COMMISSIONING_20260713.md`.

The pure-soft arm has now been behavior-screened on the same 128 keys as the
selected TEMP midpoint. It scored 72-56 (56.25%; `WW=14`, split=44, `LL=6`,
zero errors/truncations), below the midpoint's 75-53 (58.59%). Its offline
closure improved only from 0.102290 to 0.104274 and replay closure regressed
from 0.193881 to 0.183544. Therefore removing the 10% played-action hard CE is
rejected as a successor; preserve soft/hard weights 0.9/0.1 in the control.

The two value-localization arms also fail to improve the selected TEMP recipe.
Removing replay outcomes from the value denominator scored 69-59 (`53.91%`,
`WW=19`, split=31, `LL=14`); disabling scalar value loss entirely scored 63-65
(`49.22%`, `WW=13`, split=37, `LL=14`). Both used the same f7 start, row order,
dose, and behavior keys as the 75-53 TEMP midpoint. Therefore preserve
`value_loss_weight=0.25` and all-component value scope. The result also refutes
the tempting explanation that continuing value gradients alone caused the
midpoint/full behavior reversal: dose remains the supported intervention.

The function-preserving target-gather commissioning arm scored 71-57 against
f7 (`55.47%`) with every inherited tensor bit-identical and only its four new
projection tensors trained. This is positive architecture evidence, though it
does not beat the TEMP midpoint. The next composition test therefore adds the
same zero-output gather adapter to the independently positive D6 checkpoint,
freezes every inherited tensor, and consumes exactly the same 524,288-row dose.
Because this arm must commission four zero-initialized projection tensors, it
preserves the independently proven gather schedule of 8x64/global-512 for
1,024 optimizer updates (rather than the mature-model 8x512/128 schedule), with
action-module LR x4. The batch partition and number of optimizer updates change;
the row dose, sampled data contract, initializer, and all inherited tensors do
not. It compares first against that exact D6 parent.

The independently initialized short-dose D6 arm is also positive against the
binding v5 incumbent: `69-59` over 128 paired games (`53.91%`; `WW=18`,
split=33, `LL=13`) with zero errors/truncations. It used exact f7, fresh Adam,
8x512 for 128 updates/524,288 rows, and rank-distinct resumable symmetry RNG;
candidate SHA-256 is
`9dd1d261a39d7b04713505a301097faf18e84e8a3508b4abb92a8b964f7ab921`.
Its objective-matched teacher-gap closure was only `0.086684` (below TEMP's
`0.102290`) while global drift stayed comparable at `0.7021%`. This is direct
evidence that offline closure is not a strength selector and supports the
mechanism of symmetry regularization/denoising at a controlled short dose. The
screen remains `continue`, not promotion evidence; D6+gather must first beat
this exact D6 parent rather than merely inherit its gain.

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
- `cf54d5a` — namespace composite game identity and repair DDP coverage telemetry.
- `efcc94b`, `d9bf335` — seal completed diagnostic runs and bind finalizer identity.
- `58fb7e6` — repair the dedicated shared-trunk gradient-interference probe.
- `9c98473`, `42ada94` — bind P0 dose saturation, drift, and integrated LR-area
  exposure into the learner forensics.
- `d2ddbc4`, `f7b2064` — repair objective-matched scalar/auxiliary validation
  and sampler-measure semantics.
- `e1ae5bf` — make evaluator LRU operations atomic across sync/batch/async paths.
- `5838cec` — fit scalar tanh value scale on disjoint held-out games without
  mutating the sealed search operator.
- `03bf5e2` — skip frozen zero-objective head forwards and preserve two-forward
  RNG/main-output parity for optional-head controls.
- `22c1ad6` — remove dead entity-batch transfers and bind an explicit TF32
  diagnostic mode without changing the production default.
- `cea5e3c`, `c89dfee` — parallelize and authenticate the full architecture
  target audit, including legacy replay full-search equivalence.
- `ab35ba7` — bind completed topology-gather artifacts, dose, optimizer,
  systemd result, and exact adapter-only model delta in a replayable receipt.
- `e09eb37` — preserve identical validation games across descriptor-scope arms
  and retain trustworthy systemd child-exit evidence.
- `dfebf5e` — bind value-axis treatment descriptors to their exact learner
  objective so the trainer can distinguish source and treatment contracts.
- `83ad050` — make distributed D6 augmentation rank-distinct and exactly
  resumable, and add the exact selected-dose D6 launcher.

The immediate criterion is simple: preserve the independent TEMP win, select the
fastest mathematically matched DDP geometry, and spend subsequent B200 time only
on arms that isolate one causal mechanism.
