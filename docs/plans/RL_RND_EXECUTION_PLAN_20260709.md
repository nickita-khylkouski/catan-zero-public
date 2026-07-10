# Catan-Zero RL R&D execution plan (2026-07-09)

This is the execution layer for `CATAN_ZERO_ROADMAP.md`, not a replacement for
it.  The roadmap and master plan remain the source of truth.  This document
turns their highest-leverage claims into falsifiable B200 experiments and an
exact next-training recipe.

## Mandate and boundary

**Outcome:** make the search teacher, value objective, training loop, model
selection, and promotion gate capable of producing a neural checkpoint that is
measurably stronger than gen3.

**In scope now:** code, tests, corpus audits, short B200 probes, matched training
experiments, and exact launch manifests.

**Out of scope now:** a new H100 production wave, bulk self-play operations, and
a ground-up Catan-specific architecture rewrite.  Another lane owns data
production; this lane defines what data is useful and how it is learned from.

## Program status dashboard

Status is split deliberately.  A feature being runnable is not evidence that
its hypothesis wins, and a clean training run is not a promoted model.

| layer | status | meaning |
|---|---:|---|
| Core RL software | **26/26 capabilities (100%)** | learner/search/eval/provenance paths are runnable, including fixed-root stability/cost, typed S1-S3 adjudication, semantic pre-wave sealing, the legacy scalar producer bridge, and audit-bound selected-game corpus ingest |
| CPU regression | **1,982 passed / 155 skipped** | fresh full local suite after typed A0/S1-S3 adjudication, semantic pre-wave validation, exact live-ledger claims, audited A1 ingest, payload content addressing, and transitive runtime-code binding; two inherited warnings |
| Binding R&D evidence | **A0 + foundations complete; S1 running** | corpus/diversity, scalar calibration, D6 denoising, exact A0 replication, and the scalar-retention learner decision are bound; corrected S1 and then S2/S3 remain before handoff |
| Production manifest | **not frozen** | A0 is resolved; intentionally blocked on S1-S3 plus synchronized seed-base/ledger resolution |

The 26 software capabilities are: matched MSE/HL objectives; categorical model
construction/grow; trained-readout checkpoint provenance; sync/batched/server
readout selection; generation readout selection; search-identical reanalysis;
game-level validation manifests; scalar/categorical calibration; D6 controls;
post-D6 c-scale/D1 grid; D1 cross-tool semantics; global/adaptive budgets;
independent D6/wide thresholds and always-full wide roots; attributable
role-specific search cost; neutral external search; mixed-readout promotion;
opponent-manifest fail-closed handling; realized auxiliary targets; independent
value/action LR plus value freeze; deterministic upgrades; the exact A0
seal/run/verify workflow; the fixed-root repeated-search stability/cost
probe; typed fail-closed S1-S3 decision envelopes; the semantic/non-executing
A1 pre-wave contract and post-wave audit; an immutable legacy-scalar
readout attestation for the actual gen3 producer without rewriting its bytes;
and selected-game/audit-bound memmap ingest that excludes reserve attempts
before sizing, statistics, and training.
Deliberately deferred experiments
(production 88M, graph-distance bias, or a full rewrite) are not counted as
missing software.

### End-to-end execution map

| phase | owner | input | action | binding output |
|---|---|---|---|---|
| P0 | RL R&D | historical artifacts + code | full-byte seal, exact game split, recipe/order hashes | immutable A0 lock or fail closed |
| A0 | RL R&D / 2 B200s | gen1 + gen2 corpus | completed exact three-epoch MSE-vs-HL mechanism replication | tested HL rejected; retain scalar for A1 |
| S1 | RL R&D / B200s | gen3 + D6 | five-arm post-D6 c-scale/D1 grid | one completed-Q operator, or `.03/off` fallback |
| S2 | RL R&D / B200s | locked S1 operator | n64-vs-n128 screen/confirmation with attributable cost | global n128 or retain n64 |
| S3 | RL R&D / B200s | locked base budget | n128-vs-adaptive-n256 at `>=40` roots, repeated-root stability | adaptive n256 or retain global winner; never blanket n256 |
| D0 | data lane, contract from RL R&D | winning teacher manifest | 12,000 complete games at 80/15/5 current/history/hard-negative mix | immutable A1 corpus + source/search provenance |
| A1 | RL R&D / B200 | locked fresh corpus | one-dose 35M scalar training selected by A0 | calibrated candidate checkpoint |
| G1 | RL R&D | A1 candidates | internal H2H, neutral 1,000-game panel, high-regret/opening vetoes | promote one 35M checkpoint or hold |
| F1 | shared flywheel | promoted checkpoint | next data turn; isolated reanalysis/aux/surprise arms | one-change-at-a-time improvements |
| D1 | RL R&D, conditional | promoted stable 35M candidate + >=10M fresh rows | separate fresh scale and action-local two-seed arms; C0 is closed this wave | keep 35M unless a larger/local model earns its compute |

## Governing filter: the Bitter Lesson

Prefer methods that improve with more general computation and data:

1. stronger and less noisy search targets;
2. stable distributional value learning and safe target refresh;
3. diverse, replayable data with honest provenance;
4. compute-conditional model scaling;
5. neutral evaluation that permits incremental promotion.

Free labels (VP, road, army, phase) and free board topology are acceptable
small inductive aids, but they are ablated add-ons rather than the center of the
program.  A bespoke trunk rewrite fails this filter until controlled evidence
shows the current trunk is the bottleneck.

Primary statement: [Sutton, “The Bitter Lesson”](http://www.incompleteideas.net/IncIdeas/BitterLesson.html).

## Decisions already made

### Search budget

- `n_full=128` is the only global higher-budget candidate.
- `n_full=256` is **not** a whole-game default.  It is an adaptive candidate for
  wide/opening roots only, with `n_full=128` everywhere else.
- The enforced order is: D6 root averaging -> post-D6 `c_scale`/D1 calibration
  -> n64-vs-n128 -> uniform-n128-vs-adaptive-n256.
- Keep playout-cap randomization.  First compare `n_fast=16, n_full=128,
  p_full=.25`; change `p_full` to `.4` only in a separate experiment after the
  search-budget winner is known.

Why: Gumbel search was designed to remain useful at small budgets, so “more
simulations” is not automatically better per unit compute.  KataGo's
playout-cap randomization similarly favors mixing cheap and full searches over
a fixed high cap.  The local roadmap's ordering is therefore stronger than a
blanket n256 bet.  Sources: [Gumbel MuZero](https://openreview.net/forum?id=bERaNdoegnO),
[KataGo](https://arxiv.org/abs/1902.10565), and
[ELF OpenGo](https://arxiv.org/abs/1902.04522).  ELF's rollout-doubling result is
Go inference evidence, not a promised Catan training gain.

### Training objective

- Completed mechanism experiment: the exact gen2B scalar control reproduced
  the known failure, while the matched 33-bin HL-Gauss arm was less stable.
  A1 therefore retains scalar MSE with primary weight `0.25` and value-head LR
  `0.3x` torso; the tested categorical formulation is closed for this wave.
- Keep final-VP weight identical between matched arms (explicitly zero in A1);
  realized VP/subgoal auxiliaries are later single-dose arms.
- Use outcome targets only (`value_target_lambda=1`) in the first comparison.
  Refreshed root targets are tested separately so stale self-distillation is
  not bundled with the loss-shape test.
- Validation splits are by `game_seed`, never by row.  Report the trained
  `primary_value_loss`; scalar MSE remains an explicit diagnostic in the
  categorical arm.

Why: classification-based value training can be robust to noisy and
nonstationary RL targets, but local A0 evidence rejected this tested Catan
formulation.  Any categorical return must be a separately predeclared future
mechanism rather than an in-place retune.  Source:
[Stop Regressing](https://arxiv.org/abs/2403.03950).

### Model and architecture

- Keep the existing ~35M entity-graph model as the controlled baseline and
  warm-start it from gen3.
- Do not adopt ~91M yet.  C0 is closed this wave; a future scale arm requires a
  promoted stable 35M objective and at least 10M fresh audited rows.
- Do not adopt action-target cross-attention yet.  The existing module becomes
  one isolated, two-seed arm only after the 35M candidate is promoted and at
  least 10M fresh, phase-audited rows exist.
- Graph-distance/adjacency bias remains a plausible cheap arm, but it is
  **explicitly deferred**: there is no runnable bias module or causal matrix row
  in the current code.  It must not be smuggled into the value, scale, or
  cross-attention experiments.

Scaling evidence says larger AlphaZero models can be more sample-efficient,
but optimal size depends on compute and data.  Later work also finds inverse
scaling when frequent late-game states dominate.  Therefore 91M is a probe,
not a default.  Sources: [AlphaZero scaling laws](https://arxiv.org/abs/2210.00849),
[AlphaZero scaling and Zipf effects](https://arxiv.org/abs/2412.11979), and
[AlphaGateau](https://arxiv.org/abs/2410.23753).

## Current evidence

### Corpus audit

The corrected scanner hashes real entity tensors; the legacy `obs` column is an
all-zero compatibility placeholder and must never be used for diversity.

| corpus | rows | games | unique entity states | normalized target-policy entropy | unique 8-ply lines |
|---|---:|---:|---:|---:|---:|
| gen1 | 2,736,128 | 8,204 | 99.9770% | 0.7419 | 8,204 / 8,204 |
| gen2 | 3,648,516 | 12,000 | 99.9937% | 0.7158 | 11,998 / 12,000 |
| gen3 | 3,930,920 | 14,047 | 99.9986% | 0.6950 | 14,040 / 14,047 |
| fresh gen5 n64 control | 3,409,920 | 13,111 | 99.9993% | 0.6737 | 13,099 / 13,111 |

Verdict: no state or exact-trajectory collapse.  There is a monotonic 9.2%
decline in target-policy entropy from gen1 to fresh gen5, so policy
concentration is real but modest.  Improve the teacher and population mix;
do not diagnose the problem as a collapsed corpus.

### Search diagnostics

- Pre-D6 n64 SNR over 200 roots/checkpoint is non-monotonic across
  v3a/gen1/gen2/gen3.  The simple “agreement steadily decays while search-prior
  KL stays flat” signature is not observed.
- A fresh 200-root masked gen3 D6 probe shows 1-orientation/12-orientation RMS
  error ratios of `3.47x` (value), `3.45x` (prior), and `3.41x` (Q), essentially
  the `sqrt(12)` independent-noise ideal.  D6 removes a real, highly
  averageable nuisance component.
- The interrupted old n128-vs-n64 run has no valid verdict.  Partial asynchronous
  wins are non-binding and must not select n128.
- The first post-D6 c-scale run used the legacy `legal_width > 24` symmetry
  fallback.  It does not implement the independently declared inclusive D6
  threshold `>=20`, so its partial `.1`/`.3` results are diagnostic-only and
  **non-binding**.  A first threshold-20 retry was also stopped and marked
  non-binding because its historical 300-decision cap would truncate paired
  games.  The binding five-arm run is isolated at
  `runs/rl_rnd_20260709/search_calibration_d6_t20_d600`, uses explicit D6
  threshold `20`, max decisions `600`, checkpoint-calibrated sigma `.98`, and
  five disjoint VAL-only seed blocks beginning at `6,195,000,001`.  The runner
  now fails before GPU allocation if any of those protocol fields drift, and
  resume refuses partial/truncated/error arms.  All 85 pairs per arm must
  complete before S1 adjudication.

### A0 value-reuse verdict

The exact three-epoch gen2B mechanism replication is complete and
interpretable.  The lock binds 3,648,516 corpus rows, the gen1 checkpoint
(`c8d496ab...`), 598 validation games/182,594 validation rows, fresh optimizers,
and the identical epoch order (`742ed6c5...`) for both arms.

- Scalar MSE reproduced the historical validation trace exactly:
  `0.665247 -> 0.809018 -> 0.841849`.
- HL-Gauss primary CE regressed:
  `1.198052 -> 1.532889 -> 1.710083`, violating the predeclared one-percent
  stability gate.
- Typed verdict `a0-binding-verdict-v1` is stage-complete and binding.  It
  retains `learner_objective=mse` and `learner_value_readout=scalar` for A1.
  The scalar checkpoint in this mechanism replication is not a production
  candidate, and the current gen3 teacher remains independently scalar.

This is a negative result for the tested 33-bin/sigma-ratio `.75` HL-Gauss
formulation, not a reason to reopen two-hot or silently change several value
knobs.  A1 therefore uses the scalar branch selected by A0; categorical value
can return only as a new, separately predeclared formulation after the fresh
turn, not as part of this pre-wave critical path.

## B200 decision matrix

The lanes are concurrent conceptually but compete for the same two GPUs.  The
value mechanism probe gets the first free two-GPU slot because the local plan
identifies stable reuse as the highest-leverage result.  Within the search lane
the order is strict: D6 -> post-D6 Q calibration -> n128 -> adaptive n256.
Screens are small; only a winner receives confirmation.

| lane/order | question | comparison | bounded budget | binding go criterion | stop/hold criterion |
|---|---|---|---|---|---|
| P0 | Are inputs and validation immutable? | checkpoint/report/corpus hashes; identical seed manifest; fresh optimizers | CPU only | every hash and seed-set digest matches both arms | any missing/mismatched artifact blocks GPU use |
| V0 | Does HL-Gauss fix the known reuse failure? | **A0:** exact gen2B/gen1-init scalar recipe versus the same recipe with only HL-Gauss changed | 3 epochs/arm, one B200 each | scalar reproduces the historical failure and HL remains stable/calibrated | scalar non-reproduction makes the replication invalid; HL instability rejects it |
| S0 | Does D6 remove real root noise? | 12 orientations on the same 200 masked roots | completed | `~sqrt(12)` denoising, with measured root overhead | implementation or overhead regression reopens it |
| S1 | What Q contribution works after denoising? | D6 on both sides; baseline `cs=.03,D1=off` plus **five** arms: `.03/on`, `.1/off`, `.1/on`, `.3/off`, `.3/on` | 85 pairs/arm | H1 winner; D1 arms must name the checkpoint-specific `sigma_eval` artifact | no winner -> `.03`, D1 off; placeholder `.79` cannot bind |
| S2 | Does global n128 improve the expert? | same gen3 net, locked D6/c-scale; n128 vs n64 | 50-pair screen, 200-pair confirmation | pentanomial H1 at +15 Elo and separately measured search-cost ratio `<1.6x` (allow `<1.8x` only with a clear H1 margin) | flat/inferior or cost above bound |
| S3 | Is selective n256 worth it? | n128 both sides; candidate n256 only at `>=40` legal actions, D6 independently at `>=20`; wide roots always full | 50-pair screen; 200 if positive; repeated-root panel | H1 **or** >=15% lower cross-seed wide-root JS with non-worse top-1 agreement, and <=20% whole-game overhead | no blanket n256; no adoption from an undefined “stability” score |
| V1 | Does the selected scalar learner improve a real next candidate? | **A1:** gen3 warm start, post-search fresh mixed window, one scalar dose | 3-5M rows, one pass | calibration + internal H1/noninferiority contract + neutral/high-regret vetoes | hold if any binding veto fails |
| C0 | Does a future replacement value objective survive old 87.85M reuse stress? | disabled by the A0 HL rejection | no compute in this wave | reopen only for a new predeclared formulation that first beats A0 | this cannot promote a big model |

The H2H artifact alone does **not** provide the S2 cost ratio or S3 stability
metric.  A fixed-root timing/stability artifact is required; combined H2H wall
time cannot be misreported as per-role throughput.  Search rows S1-S3 are
budgeted at roughly 9-13 B200-GPU-hours total if screens stop losers.

## Artifact lock: required before either training arm

“Same corpus/checkpoint” means byte identity, not matching filenames.  Each run
directory must contain an `inputs.sha256` created before GPU allocation and
verified again after the run.  Hash the initialization checkpoint, the
historical report that defines the recipe, `corpus_meta.json`, every memmap
`.dat` file, and any pre-existing explicit validation-seed manifest.  When the
split is derived by fraction/seed, lock those inputs plus the corpus first,
then append/hash the trainer-produced seed manifest before calibration or H2H.
A metadata-only corpus hash is not sufficient because it cannot detect mutated
array bytes.

```bash
set -euo pipefail
cd /home/ubuntu/catan-zero
RUNDIR=runs/rl_program_20260709/value_rnd
mkdir -p "$RUNDIR"
sha256sum \
  runs/bc/gen1_20260705/checkpoint.pt \
  runs/bc/gen2B_20260706/report.json \
  runs/memmap_gen2_20260706/corpus_meta.json \
  runs/memmap_gen2_20260706/*.dat \
  > "$RUNDIR/a0.inputs.sha256"
sha256sum -c "$RUNDIR/a0.inputs.sha256"
```

The literal digests from this file, rather than shortened MD5 labels, are copied
into the A0 manifest and final report.  The exact historical epoch-1 87.85M
checkpoint and report receive a separate hash lock before C0.  If the expected
epoch-1 file is absent, a generic final `checkpoint.pt` must not be relabelled
as epoch 1.

Both training arms use `--no-resume-optimizer`.  This is binding: an Adam
sidecar cannot be restored on the scalar arm while the config-upgraded
categorical arm starts fresh.  Reports must say `resume_optimizer=false` and
`optimizer_restored=false` for both arms.

The trainer writes `<report>.validation_seeds.json`.  Before calibration or
H2H, the two arms' sorted `game_seeds` arrays, seed counts, and
`validation_game_seed_set_sha256` must be identical.  Calibration consumes
exactly that seed set (or independently derives it from the raw shards and
proves the arrays equal); it may not silently use the older
`DEFAULT_HOLDOUT_BLOCKS`.

## A0 — exact gen2B reuse-failure replication

This answers one mechanism question only: does changing scalar regression to
HL-Gauss remove the already-observed multi-epoch value failure?  It is **not**
the next production recipe.

Locked historical inputs:

- corpus: `runs/memmap_gen2_20260706`, exactly 3,648,516 rows;
- initialization: `runs/bc/gen1_20260705/checkpoint.pt`;
- recipe authority: `runs/bc/gen2B_20260706/report.json` and its epoch reports;
- historical scalar trace: fixed-validation value MSE
  `0.6652 -> 0.8090 -> 0.8418`;
- validation: the exact game-level seed set reconstructed and frozen from the
  historical report/corpus.  If the old trace cannot be mapped to a valid
  game-level split, report that limitation and run a new matched stress test;
  do not call it a replication.

The historical report is normalized into an immutable `a0.recipe.lock.json`
containing every optimizer, LR, warmup/schedule, precision, batch, policy,
sample-weight, masking, and validation field.  Both arms consume that lock and
use the same training order.  The only permitted differences are:

| arm | model addition | primary value objective | search/calibration readout | scalar auxiliary |
|---|---|---|---|---:|
| A0-MSE | none | historical scalar MSE | scalar | n/a |
| A0-HL | deterministic `catbins:33`, seed 1 | 33-bin HL-Gauss CE, sigma ratio `.75` | categorical expectation | `0` |

Both primary objectives use the same configured weight and the same `.25`
effective truncation-row weight.  MSE uses the historical VP-margin soft target
on truncations; HL uses a separate truncation class.  Report clean and
truncated loss/counts separately.  Use `value_target_lambda=1` in both arms
unless the historical scalar trace itself used a different value; in that case
the exact historical value is retained in **both** arms and recorded as a known
target-source confound.

A0 is valid only if the scalar control reproduces the direction and material
size of the known failure.  A scalar non-reproduction invalidates the stress
test; it is not evidence for or against HL-Gauss.  A0-HL passes when epoch 3
primary CE is no worse than epoch 1, neither later epoch regresses by more than
1%, categorical calibration remains stable globally and in opening/`41+`
buckets, and policy drift stays within 2% of the scalar control.

## Search-teacher lock before fresh training

A0 establishes whether training compute can be reused.  The search lane then
finishes S1-S3 in its own strict order.  The resulting immutable teacher
manifest records D6 threshold, `c_scale`, D1 and its measured `sigma_eval`,
`n_fast`, `n_full`, `p_full`, adaptive budget and threshold, value readout,
masking, and checkpoint hash.

If a D1 arm wins, its `rescale_noise_floor_c` and `sigma_eval` must first be
threaded through typed internal H2H and neutral external evaluation.  A
generator-only D1 setting cannot become the teacher while promotion silently
evaluates a different completed-Q operator.  The neutral harness must likewise
support and attest the selected D6 threshold and adaptive wide-budget/
always-full settings; categorical readout parity alone is insufficient.

The intended decision is global `n_full=128`, not global n256.  The only n256
candidate is `n_full_wide=256` at `>=40` legal actions with
`wide_roots_always_full=true`; D6 has its independent `>=20` threshold.  This
prevents the H2H’s forced-full behavior from hiding a production generator that
would otherwise send 75% of wide roots through n_fast.

## A1 — exact post-A0/post-search fresh 35M candidate contract

A1 is the next-model tournament the data-production lane must feed.  This lane
does not launch H100 generation; it publishes and validates the data contract.

Data contract:

- exactly 12,000 complete, globally unique games selected before row expansion:
  9,600 (80%) current producer under the winning search manifest, 1,800 (15%)
  recent/older champions, and 600 (5%) hard-negative/RGSC restart games. This
  should land in the local plan's 3-5M-row window without cutting games to hit a
  cosmetic row count; immutable source/shard/corpus hashes are mandatory;
- source labels and source checkpoint hashes remain in the corpus metadata;
- public-observation features only; no VAL-ONLY seeds; zero invalid teacher
  actions; truncation, forced-row, phase, decision-index, legal-width,
  target-entropy, and full-search-policy-mass reports are mandatory;
- the existing fresh n64 control window is retained as the control artifact,
  not silently mixed into the stronger-teacher arm unless its declared source
  percentage permits it;
- a 5% game-level validation set is frozen once for the scalar A1 run;
  `validation_max_samples=0` keeps every row from those games.  No
  row-level cap may change which games constitute the holdout.

### Fail-closed pre-wave handoff boundary

`configs/experiments/a1_pre_wave_contract.template.json` is the only supported
starting point for the 24-GPU handoff.  It is intentionally not launchable:
every result selected by A0/S1-S3 is written as `__UNRESOLVED__`, and
`tools/a1_pre_wave_contract.py seal` refuses while any such value remains.
Sealing also refuses missing/mutated A0/S1/S2/S3 evidence, an unmasked or
uninspectable checkpoint, a config-only categorical readout, generator-guard
drift, a changed seed-ledger snapshot prefix, seed overlap, learner-code drift,
or any VAL-ONLY seed. The shared ledger is not incorrectly frozen forever:
the lock preserves its exact pre-claim bytes, `render` emits one exact claim
row per job, live verification permits only append-only disjoint growth, and
the post-wave audit requires every one of the 72 exact own claims.

The sealed plan expands deterministically to three category-specific jobs per
GPU, not a probabilistic opponent mix: each worker attempts 408
current-producer, 77 recent/history, and 26 hard-negative games.  Postflight
selects the lowest-seed complete 400/75/25 per job, so rare healthy
truncations consume only the bounded reserve and the selected pre-row-expansion
quotas remain exactly 9,600/1,800/600.  Search and
evaluator configs are stored in full and separately hashed; checkpoint,
generator-code, explicit learner-code, the complete transitive local runtime
tree, guard, evidence, ledger-snapshot, and complete seed-plan hashes are part
of the contract.  D6 and adaptive-n256 have independent thresholds,
`wide_roots_always_full`, D1/`sigma_eval`, `p_full`, late temperature, public
observation, and value readout are all explicit.  A0's learner objective is a
separate binding from the producer's search readout: the binding verdict
retains scalar MSE for A1, and the current gen3 teacher also remains scalar.
No categorical readout is authorized in this wave.  Typed A0/S1/S2/S3
decision envelopes are
replayed in-process by their canonical adjudicators, must inherit the exact
prior-stage decision bytes, and must match the final learner/search/evaluator/
checkpoint fields; arbitrary JSON files cannot unlock `seal`.
The legacy gen3 scalar checkpoint predates embedded `value-training-v1`, so
its exact bytes are authorized only through
`legacy-scalar-readout-attestation-v1`, which binds the checkpoint to its
immutable training report and reconstructs positive scalar loss telemetry.
That bridge is scalar-only and can never authorize a categorical readout.
The accepted producers are `tools/a0_binding_verdict.py` for A0 and
`tools/search_teacher_adjudicator.py` for S1-S3; their own bytes and all source
artifacts are re-hashed into the contract.

`render` writes immutable per-job argv/environment records and frozen
category manifests, plus one job attestation that the data lane must copy to
`<output_dir>/a1_contract.json` before that job starts.  It has no
execute/SSH/subprocess path: this R&D lane stops there and the data lane owns
the deliberate wave.  After the wave, `audit`
must pass before corpus ingest.  It hashes every attempted shard, selects only
the predeclared lowest-seed complete quota, and checks all 12,000 unique
category-labelled games, zero selected truncations,
zero invalid teacher actions, no VAL-ONLY overlap, the public-observation and
readout attestations, source checkpoint identities, shard SHA-256 inventory,
and the mandatory truncation/forced/phase/decision-index/legal-width/entropy/
active-full-search reports.  The existing n64 control is not an input to this
contract.  The passing audit JSON is the required corpus-provenance sidecar:
its per-shard category, producer/opponent checkpoint SHA-256, search/evaluator
digests, and shard inventory digest must be copied into the memmap corpus
metadata rather than discarded after validation.  The same audit deterministically
materializes an immutable `a1-selected-training-games-v1` manifest (including
the exact train/validation partition and source identity per game) plus the
trainer-compatible 5% game-seed holdout (seed 17,
`validation_max_samples=0`) and its byte-level seed-set digest; the A1 learner
must consume and reproduce that sidecar exactly.

This is enforced in software, not by an operator checklist:
`tools/build_memmap_corpus.py` requires the selected-game sidecar and passing
audit together, auto-detects A1 job attestations so the generic path cannot
strip the selection boundary, verifies the actual canonical shard paths and SHA-256 inventory,
and retains all selected train plus validation games while filtering only
reserve/unselected attempts.  `tools/train_bc.py
--validation-game-seed-manifest ...` then verifies the audit contract and exact
validation seed/count/row digests from `corpus_meta.json`, re-hashes every
`.dat`/`.codes.dat`/offset payload, and re-hashes the bound 208-file local
runtime tree before the first optimizer step; range overrides and row caps are
forbidden on this path.

Training contract selected by the binding A0 verdict:

- gen3 warm start, scalar-MSE head/readout; no categorical upgrade in this wave;
- existing ~35M entity-graph trunk: hidden 640, 6 layers, 8 heads, dropout `.05`;
- one single-B200 dose: seed 1, one epoch, no step cap, micro/global batch 4096,
  accumulation 1, world size 1, BF16, graph-history features on, public masking
  on, and DDP data sharding/symmetry augmentation off;
- fresh Adam (no optimizer resume/fusion/weight decay), LR `3e-5`, 100 warmup
  steps, flat schedule, `value_lr_mult=.3`, and action-module multiplier `1`;
- policy weight `1`; stored-policy soft targets at weight `.9`, temperature `.7`,
  and minimum legal coverage `.5`; primary value weight `.25` with
  `value_target_lambda=1`;
- categorical override, HL scalar auxiliary, final-VP, Q, policy-KL,
  uncertainty, subgoal, surprise, advantage, per-game, and VP-margin weights
  are all off; truncated VP-margin value supervision remains `.25`;
- forced-action/value weights are `.1`/`1`; winner/loser sample weights are
  `1`/`.3`; teacher/phase/value-phase overlays and freeze lists are empty;
- `track=2p_no_trade`, `vps_to_win=10`, and the graph-history flag are bound
  explicitly rather than inherited from mutable CLI defaults;
- final-VP loss is explicitly zero; turning it on is a later single-dose arm;
- the saved checkpoint must positively attest scalar training/readout through
  `value-training-v1`; config-only categorical metadata remains fail-closed.

The complete effective one-dose recipe above is stored as
`science.learner_training_recipe` in the draft/lock and separately hashed.
An A1 memmap is auto-detected by `train_bc`; the exact validation manifest is
mandatory, and the trainer replays the selected-game, audit, validation, and
contract-lock bytes before comparing the effective optimizer/loss/exposure/
masking recipe field-for-field.  Corpus rows, train/validation seed sets, the
recipe digest, complete memmap-payload inventory digest, learner-code digest,
and producer checkpoint SHA-256 are written back into the training report and
`value-training-v1` checkpoint provenance.

These effective values live in the sealed lock as
`science.learner_training_recipe` with its own canonical SHA-256.  The trainer
reconstructs the same effective dictionary, including derived world/global
batch fields, and rejects missing, extra, type-drifted, or value-drifted knobs
before the first optimizer step.

Calibration runs on every saved checkpoint with the exact validation seed set
and reports global, phase, forced/unforced, and legal-width buckets `1`, `2-4`,
`5-10`, `11-20`, `21-40`, `41+`.  Binding A1 criteria are:

1. scalar Brier/RMSE and no critical phase/`41+` bucket regress beyond the
   predeclared 2%/5% tripwires versus gen3;
2. unforced policy loss and prior KL regress by no more than 2%;
3. scalar search readout reaches the promotion contract below on confirmation;
4. corpus, seed-set, optimizer, objective, and readout provenance all match the
   locked manifest.

## Follow-on training arms (one dose each)

Run only after the one-dose scalar A1 verdict, one change at a time:

1. **safe value refresh:** identical corpus, bounded search-consistent root
   values, immutable readout/squash/scale provenance; compare with outcome-only.
   This is “reanalysis-lite,” not full MuZero Reanalyse.  Full Reanalyse would
   recompute MCTS policy and value targets.  Source:
   [MuZero Reanalyse](https://arxiv.org/abs/2104.06294).
2. **realized auxiliary targets:** one arm enabling the already-wired road,
   army, VP-in-N, settlement, and robber targets; keep only if two seeds are
   noninferior and at least one strength/calibration metric improves.
3. **policy-surprise weighting:** upweight roots where search meaningfully
   changes the prior; do not combine with the auxiliary arm.  KataGo reference:
   [official methods](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md).
4. **population/restart data:** training mix target is 75-80% current producer,
   10-15% recent/older champions, and 5-10% hard negatives/restarts.  Track each
   source separately.  Archived-state restart evidence:
   [Go-Exploit](https://arxiv.org/abs/2302.12359).

## Frozen 87.85M stress versus production scale

These are different decisions and must not share a “91M passed” label.

**C0 frozen reuse stress is closed for this wave.**  A0 rejected the tested
categorical objective, so spending two more 87.85M exposures would not answer
the pre-wave question.  It may return only after a different value formulation
first passes its own matched 35M mechanism probe.  The historical protocol uses the
hash-verified historical **epoch-1** 87.85M artifact, its exact historical
corpus, and two post-branch epochs with fresh Adam.  Scalar and HL arms differ
only in the primary value objective.  C0 asks whether categorical value avoids
the old epoch-2 pathology at this capacity.  It does not compare 35M and 87.85M
strength, does not require 10M fresh rows, and cannot promote a large model.

The artifact identity is no longer ambiguous:

- epoch 1 is `runs/bc/bignet91M_20260707/checkpoint.pt`, SHA-256
  `bac77a2ae41ad3d8d6327d0b9f3f591e25882d3eee9515f110574204eb76602e`;
  its one-epoch report SHA-256 is
  `f975642f6ffb3b772cbfaf62c9a367a4b11862516587280cb269f79cac9d3049`;
- the historical epoch-2 continuation is the separate
  `runs/bc/bignet91M_20260707_ep2/checkpoint.pt`, initialized from the file
  above, SHA-256
  `8bbeb872919358c65c9f9f18463ad1926ba240f42ff093ce617c0c56b32abc44`;
  its report SHA-256 is
  `26d83f8e5bf6a736c44378c6f6e5cd53a42af518d1772b26699c6327befdb637`.

Both reports attest 87,845,705 parameters, hidden 896, eight layers/eight
heads, the 3,930,920-row gen3 corpus, batch 1024, FP32, Adam, and LR `3e-5`.
The validation trace is `.266470 -> .392948` while policy improves
`1.625174 -> 1.596607`.

**Production 80-100M scale** remains blocked until all are true:

- the scalar A1 candidate passes the full promotion contract;
- at least 10M fresh, source-diverse rows exist;
- phase, decision-index, forced-action, and legal-width audits exclude
  late/forced-state domination;
- a 35M and large-model comparison is defined at equal fresh rows, optimizer
  steps, validation games, and at least two initialization seeds;
- initialization is explicit.  A grow run must report copied parameter
  fraction; a width change that copies little is treated as fresh initialization,
  not “warm-started” marketing;
- same-search evaluator/search timing provides attributable per-model
  throughput.  Mixed-model H2H wall time is not a throughput measurement.

Only that fresh scale experiment may decide 35M versus 80-100M.  Keep 35M if
the larger model lacks a confirmed n64 H1 strength gain, degrades any
opening/`41+` bucket by more than 5%, or loses more than 15% same-search
throughput.  A 15-30% throughput loss requires at least a +30 Elo confirmed
gain; a loss above 30% is rejected in this phase.

The action-local architecture arm is also behind a promoted A1 scalar
candidate and 10M fresh rows.  Deterministic upgrade seeding,
`--action-module-lr-mult`, and the `--freeze-modules value_heads` policy-warmup
contract now exist.  Run unchanged 35M versus `gather,cross:2` at two
independent module seeds, equal rows and
wall clock, with scale and auxiliary heads unchanged.  It is never bundled
with 87.85M.

Graph-distance/adjacency bias has no current implementation and is outside this
execution wave.  It returns only as its own causal arm after a runnable module,
tests, and a go/no-go row exist.

## Promotion contract

1. Mechanism checks first: objective stability, calibration, phase buckets,
   policy drift, and provenance.
2. Cheap 50-pair screen.
3. Paired, seat-swapped neutral H2H with candidate/baseline readouts explicitly
   attested.
4. Promote at +15 Elo H1; reject below -10 Elo; continue sampling when
   inconclusive.
5. Run the fixed neutral external panel against `catanatron_value` under the
   same referee/search manifest, 500 paired seeds/1,000 games.  It is a binding
   regression tripwire: a candidate that is internally positive but accepts
   the -10 Elo external H0 is held.
6. Run the immutable high-regret/opening suite.  Any critical phase or `41+`
   bucket worse by more than 5%, or a paired high-regret H0, vetoes promotion.
7. Every third promotion receives an n64 confirmation.  Inconclusive external
   or high-regret results are extended with disjoint seeds, never called wins.

This follows Expert Iteration's actual contract: search is the expert and the
network is the apprentice, so both teacher improvement and faithful
distillation must be measured.  Source: [Expert Iteration](https://arxiv.org/abs/1705.08439).

## Software closure checklist

- [x] D6 wide-root averaging is wired through generate/eval configs.
- [x] Post-D6 c-scale/D1 grid is runnable.
- [x] Categorical readout is wired through sync, batched, and eval-server paths.
- [x] Matched MSE/HL objective weights and primary-value telemetry are explicit.
- [x] Categorical head is included in the value-head LR group.
- [x] Config-only categorical upgrades fail closed; trained readout provenance
      is saved in checkpoints.
- [x] Candidate categorical versus baseline scalar promotion is supported and
      attested.
- [x] Reanalysis values use search-identical squash/scale/readout semantics and
      immutable provenance.
- [x] Opponent manifests resolve paths, hashes, and aliases before generation.
- [x] Neutral-harness search-vs-search evaluation is resumable and fail-closed.
- [x] Deterministic upgrade seeding and a separate action-module LR group exist.
- [x] Neutral-harness search explicitly selects and attests scalar/categorical
      readout; config-only categorical heads fail closed.
- [x] `--freeze-modules value_heads` provides the action-local policy-warmup
      freeze without freezing the policy/trunk.
- [x] Exact A0 seal/run/verify software locks artifacts, validation games,
      three epoch orders, arm recipes, GPUs, and postflight provenance.
- [x] Lock the exact gen2B recipe/report/checkpoint/corpus hashes and prove the
      A0 scalar trace reproduces before interpreting HL-Gauss.
- [x] Candidate/baseline-specific wide-root budgets **and thresholds**, separate
      D6 threshold, and wide-root always-full semantics must pass tests before
      S3.
- [x] Add attributable per-role search-cost artifacts for S2/S3; combined H2H
      elapsed time is not used as a substitute.
- [x] Add the fixed-root repeated-search JS/top-1 stability artifact before S3.
- [x] Add typed, hash-bound S1/S2/S3 adjudication; partial screens, old D6
      thresholds, mixed wall-time cost, and global n256 fail closed.
- [x] Thread D1 `rescale_noise_floor_c`/`sigma_eval` through internal H2H,
      typed EvalConfig, and neutral external search, plus D6/adaptive-wide
      semantics through neutral search, before a selected search manifest can
      bind A1 promotion.
- [x] Run the targeted/full CPU suite after the completed software paths.
- [x] Add an unresolved-until-selected, immutable A1 pre-wave contract that
      guarantees category-specific 9,600/1,800/600 seed/job allocations,
      renders commands without executing them, and fail-closes post-wave
      provenance/quality before ingest.
- [x] Bind the actual legacy gen3 scalar teacher to its immutable report with a
      scalar-only attestation instead of mutating checkpoint bytes or inventing
      categorical provenance.
- [ ] Run the bounded B200 matrix; freeze the winning config only after artifacts
      exist.
