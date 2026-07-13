# A1 paid-search target semantics audit

Status: code-and-artifact audit. This document does not authorize a production
default change or a long training run.

## Decision

The learner is **not discarding the primary paid-search signal**. On every
non-forced full-search row, `target_policy` is the Gumbel-MCTS improved policy
and `policy_weight_multiplier=1`; it is the main policy target. Forced and fast
rows have `policy_weight_multiplier=0`, so they contribute neither numerator
nor denominator to policy loss. Forced rows remain legitimate realized-outcome
value examples.

Do not enable `root_value`, `target_scores`, or `afterstate_target` in the
production learner merely because their columns now survive both loader paths.
The safe current defaults remain:

- improved-policy distillation on active full-search rows;
- realized outcome value target (`value_target_lambda=1`);
- Q loss off (`q_loss_weight=0`);
- `simulations_used` retained as provenance/telemetry, not a label.

The first matched learner probe is `V100` versus `V75` from
`A1_LEARNER_RECOVERY_PLAN_20260712.md`, but only after the anti-forgetting
recipe is selected. It changes one field: blend 25% stored `root_value` into
the scalar value target on its authenticated mask. Do not combine this probe
with Q loss, afterstate blending, HL-Gauss, a dose change, or an LR change.

## End-to-end field meanings

| Field | Exact producer meaning | Coverage/mask | Scale and perspective | Default learner decision |
|---|---|---|---|---|
| `target_policy` | `SearchResult.improved_policy`: completed-Q-informed Gumbel policy, after optional target pruning | Policy-active only when non-forced and full search | Probability distribution over legal actions | **Use now**; this is the primary paid-search target |
| `root_value` | `root.value`: mean backed-up value at the search root, including the root evaluator prior and tree backups | Explicitly stored only for non-forced full-search rows | Root/acting-player perspective, bounded outcome scale `[-1,1]` | Keep off by default; probe `lambda=.75` after stability recipe |
| `target_scores` | `SearchResult.q_values`: raw visit mean `stats.q` for **visited actions only**; despite completed-Q driving policy improvement, this column is not completed-Q and does not cover unvisited actions | finite mask per visited legal action; source stamped `gumbel_mcts_visit_q` | Root-player perspective, outcome scale `[-1,1]` | Keep Q loss off; current loss row-standardizes these values, so the trained Q head is not return-scale |
| `afterstate_target` | For enumerated chance actions, probability-weighted evaluator value immediately after applying the action/chance outcomes, before deeper search; chiefly ROLL and enumerated robber/dev-card actions | finite per-action mask; forced ROLL can be covered even though forced policy weight is zero | Root-player perspective, evaluator-derived `[-1,1]` | Preserve, audit, but do not blend by default; it is a stale one-ply self-estimate |
| `simulations_used` | Search algorithm tree-simulation counter returned by the operator | all rows; zero for forced/raw paths, configured fast/full budgets otherwise | Nonnegative integer; **not** neural forward work (chance enumeration and symmetry alter forward rows) | Telemetry/stratification only; never treat as a supervised target or exact compute weight |

The Python and native Rust search agree on these definitions. Both emit root
`q_values` from `stats.q()` only for actions with visits and emit
`root_value=root.value()`. The completed-Q dictionary used by Gumbel policy
improvement separately fills unvisited actions with `v_mix` and may rescale it;
that transformed dictionary is not what is persisted in `target_scores`.

## Artifact evidence

A direct inspection of an authenticated reconstructed n128 shard
(`selected_000279.npz`, 512 rows) found:

- `target_score_source = gumbel_mcts_visit_q` on every row;
- `target_information_regime = public_conservation_pimc_v1` on every row;
- 59 non-forced full-search/root-value rows;
- `simulations_used` values of 128, 16, or 0 according to full, fast, or forced
  execution;
- 2,071 finite visited-action Q slots and 398 finite afterstate slots;
- masks exactly matching finite payload slots.

The current n128/n256 memmaps were built before the loader correction and do
not contain `afterstate_target` or `simulations_used`; original NPZ shards do.
Commit `145dd03` makes future NPZ and memmap loads retain them. This is a schema
recovery, not evidence that either target improves play.

## Optional completed-Q/visit evidence: measured storage decision

Historical shards retain the improved policy, visited-action Q, prior, root
value, and total simulations, but not the two root statistics needed to
recompute a different Gumbel target after generation: the all-legal-action
completed-Q vector and per-action visit counts. The smallest safe addition is
therefore an **opt-in raw-NPZ payload**, not another always-materialized memmap
column:

- active rows are inferred from the already-mandatory
  `policy_weight_multiplier > 0` column (no duplicated row IDs);
- `search_evidence_offsets`: uint32 offsets for active rows;
- `search_visit_counts_flat`: uint16, with a fail-closed 65,535 upper bound;
- `search_completed_q_flat`: float32, never float16 (observed opening-road
  completed-Q margins reach roughly `1e-7`);
- a uint8 schema version.

On the authenticated n64 control corpus at
`memmap_gen5_n64_control_20260709_s1`, the measured counts are:

- `N = 3,409,920` total rows across `S = 1,665` shards;
- `A = 418,553` policy-active rows (12.2746%);
- `L = 4,163,736` active legal-action entries (9.9479 per active row).

The exact raw-array cost over a sharded corpus is
`4*A + 6*L + 5*S`: four bytes per active-row offset, six bytes per legal entry,
and one version byte plus one terminal offset per shard. For these measured
counts that is **26,664,953 bytes = 7.8198 bytes per corpus row**. A direct
`numpy.savez` incremental-size measurement under both NumPy 1.21.5 (B200) and
2.2.6 (local) adds exactly 1,114 bytes of NPY/ZIP headers per shard for the four
new arrays, yielding **28,519,763 bytes = 8.3638 bytes per row on disk** before
optional outer zstd compression. This is 0.0604% of the existing 47,215,544,261
byte memmap corpus. A naive width-54 padded `(float32 Q, uint16 visits)` pair
would cost 324 bytes per row, or 1,104,814,080 bytes on the same corpus.

The CLI flag `--preserve-search-evidence` enables this payload. Default-off
generation still drops the private transient arrays before buffering rows, so
the historical shard schema and learner input are unchanged. The learner
ignores the four optional arrays; posthoc tooling reads them from raw NPZ. They
are deliberately not copied into the training memmap: current prefetch code
would materialize every registered column per batch, turning audit-only
evidence into a permanent learner I/O cost.

Sufficiency is explicit and fail-closed. A single-world target can be
recalibrated from completed-Q, visits, the existing prior/phase columns, and
the manifest search config. This is target recalibration at the precision of
the existing raw-NPZ `prior_policy` (float16), not bitwise recovery of the
original evaluator's float prior. Preserving a duplicate float32 prior would
add another `4*L = 16,654,944` raw bytes (4.8843 bytes per corpus row) and was
rejected from this smallest missing-statistics schema; it can be added in a
future evidence version if bit-exact prior-logit replay becomes a requirement.
Public-belief `aggregate_q_then_improve` is likewise reconstructible from its
aggregate completed-Q and the manifest particle count. Historical PIMC
`mean_improved_policy` is **not** reconstructible from
only a mean completed-Q because `mean(softmax(x)) != softmax(mean(x))`; the
writer refuses to attest that combination rather than emitting misleading
evidence. This payload supports target-operator recalibration (for example
`c_scale`, fixed sigma visits, D1 attenuation, or pruning), not replaying MCTS,
changing pre-completion uncertainty shrinkage, or recovering per-particle
statistics that were never stored.

## Why the other columns are not defaults yet

### Stored root value

It is the generating network's archived search value. Mixing it into the value
target is a self-distillation channel: at a plateau it approaches the model's
own value output and weakens the fresh terminal signal. The earlier lambda arm
was only a 400-game, seven-arm winner with overlapping uncertainty, while the
continuous lineage later showed severe value drift. That makes a small direct
matched comparison appropriate, not a default flip.

### Visited-action Q

The payload itself is return-scale visit Q, but the current `_q_score_loss_parts`
centers and divides target scores by their within-row standard deviation before
MSE. A model trained by that objective learns relative z-scores, while search
and PPO-style consumers describe `q_values` as return-scale action values.
Turning it on would silently overload one head with incompatible semantics.
It also supervises only visited actions, whose count and selection depend on
the Gumbel schedule.

Before a Q learner probe, choose and bind one contract:

1. a dedicated ranking auxiliary head trained on centered Q, never consumed as
   return-scale Q; or
2. the existing Q head trained on raw `[-1,1]` visit Q, with held-out calibration
   evidence and explicit provenance.

Do not test both interpretations through the same output head.

### Afterstate value

This is useful provenance and may eventually support a chance-specific
auxiliary, especially for forced ROLL states. It is nevertheless a one-ply
generating-network estimate rather than a realized return or a searched root
value. Applying it to the roughly half of rows that are forced would create a
large self-estimated loss channel. It must therefore follow, not precede, the
root-value probe.

## Smallest validation ladder

1. **Offline calibration, no training:** on game-disjoint held-out NPZ games,
   report coverage, range, correlation/MAE versus realized outcome, and bias by
   phase for `root_value`, played-action visit Q, and played-action afterstate.
   Stratify full-search rows by `simulations_used` and n128/n256. This detects
   sign, scale, and mask mistakes; it does not establish counterfactual action-Q
   accuracy.
2. **Matched root-value probe:** same initialization, data order, sample dose,
   LR, batch, and anti-forgetting recipe; compare V100 (`lambda=1`) with V75
   (`lambda=.75`). Gate on the same internal and external panels. This is the
   only paid-value learner arm authorized before further evidence.
3. **Conditional Q probe:** only if an explicit head contract is selected and
   offline calibration passes. Compare Q weight 0 to one small weight at equal
   sample dose. Require both strength non-regression and held-out Q calibration;
   a lower normalized-Q training loss alone is not success.
4. **Conditional afterstate probe:** only after V75 and the Q decision. Use a
   separate chance/afterstate auxiliary or a tightly masked forced-ROLL arm;
   never silently substitute it for `root_value` in the existing lambda flag.

This ordering extracts information already paid for without mixing three
self-estimated targets into the learner at once or confusing a richer schema
with validated supervision.
