# RL architecture, value, and scale protocol — bounded B200 R&D

Status: executable protocol with fail-closed prerequisites. It authorizes short
B200 probes only. It does **not** authorize an H100 production wave or a full
80–100M training campaign.

This document tightens `CATAN_ZERO_ROADMAP.md` and
`CATAN_ZERO_MASTER_PLAN.md`; it does not replace them. Paper-derived ideas are
evidence for local hypotheses. The local failure traces and promotion contract
remain authoritative.

## Decision in one page

The Bitter Lesson says to prefer mechanisms that improve as general search,
data, and compute increase. For this project the Pareto path is:

1. **A0: close the value-objective question for this wave.** The exact gen2B
   scalar failure reproduced, and the matched HL-Gauss arm failed its primary
   stability gate; retain scalar MSE/readout.
2. **Improve the expert in the local-plan order.** Keep D6 root denoising;
   calibrate c-scale/D1; test global n128; test n256 only at wide/opening roots.
3. **A1: train the next 35M candidate.** Use one locked fresh mixed window from
   the winning search teacher and exactly one scalar-MSE dose.
4. **Keep C0 closed.** The tested categorical formulation failed A0, so the
   historical 87.85M reuse stress spends no compute in this wave.
5. **Spend capacity only when earned.** A production 80–100M comparison and
   action-local cross-attention both wait for a promoted stable 35M candidate
   plus at least 10M fresh, phase-audited rows. They remain separate experiments;
   the current baseline objective is scalar.

Therefore:

- global search candidate: `n_full=128`, not n256;
- adaptive candidate: n256 only at `>=40` legal actions, with those roots
  always full; D6 uses an independent `>=20` threshold;
- model now: keep and retrain the 35M entity-graph family;
- value now: exact A0 matched scalar MSE versus 33-bin HL-Gauss is complete;
  HL failed its primary stability gate, so A1 is bound to scalar MSE/readout;
- architecture now: no trunk rewrite, no adjacency-bias bundle, no PPO/MuZero/
  CFR conversion.

### 2026-07-09 A0 execution ruling

The scalar arm exactly reproduced the locked historical trace
`0.665247 -> 0.809018 -> 0.841849`; the matched HL-Gauss arm regressed
`1.198052 -> 1.532889 -> 1.710083`.  The typed binding verdict therefore says
`retain_scalar_for_a1`.  This closes A0 as an interpretable negative result:
the pre-wave program does not spend more B200 time tuning this exact HL
formulation, does not promote either mechanism checkpoint, and does not block
the independent S1-S3 search-teacher sequence.

## Known architecture facts

| artifact | fact that must be re-attested in the run manifest |
|---|---|
| gen3 checkpoint | 35,041,353 parameters; hidden 640; 6 state layers; 8 heads; dropout .05 |
| gen3 report | 3,930,920 rows; batch 4096; BF16; 912 steps; about 889 s/epoch on one B200 |
| historical large checkpoint | 87,845,705 parameters; hidden 896; 8 state layers; 8 heads; dropout .05 |
| historical large epoch 1 | policy val 1.6252; scalar value val MSE .2665 |
| historical large epoch 2 | policy val 1.5966; scalar value val MSE .3929 |
| historical large resource trace | batch 1024; FP32; 117–121 GiB peak reserved; about 8,085 s/epoch |

The large failure does not prove that capacity is harmful. It proves that more
capacity/reuse amplified the scalar-value pathology under that run.

## P0 — immutable inputs and matched-state preflight

No GPU probe starts until all checks below pass.

### Full-byte input lock

The run directory stores full SHA-256 digests for:

- initialization checkpoint;
- historical report that defines the recipe;
- corpus `corpus_meta.json`;
- every corpus `.dat` file;
- any pre-existing explicit validation-seed manifest. For a fraction/seed
  split, lock the split inputs and corpus first, then append/hash the
  trainer-produced seed manifest before calibration/H2H;
- config-upgraded checkpoint after it is created.

Hashing `corpus_meta.json` alone is not a corpus identity because array bytes can
change without changing metadata. Example A0 lock:

```bash
set -euo pipefail
cd /home/ubuntu/catan-zero
RUNDIR=runs/rl_program_20260709/a0_gen2b_hlgauss
mkdir -p "$RUNDIR"
sha256sum \
  runs/bc/gen1_20260705/checkpoint.pt \
  runs/bc/gen2B_20260706/report.json \
  runs/memmap_gen2_20260706/corpus_meta.json \
  runs/memmap_gen2_20260706/*.dat \
  > "$RUNDIR/inputs.sha256"
sha256sum -c "$RUNDIR/inputs.sha256"
```

The full literal digests in `inputs.sha256` are copied into the experiment
manifest. They are not replaced by a filename or shortened MD5. This protocol
does not guess hashes for remote-only artifacts: absence of the digest file is
a launch blocker.

### Recipe and validation lock

`runs/bc/gen2B_20260706/report.json` is normalized into
`a0.recipe.lock.json`, including every optimizer, LR, schedule/warmup,
precision, batch, policy target, sample weight, masking, split, and step field.
The scalar arm must reproduce the historical recipe; otherwise A0 is a new
stress test, not the local plan's mechanism replication.

Both arms must report:

- `resume_optimizer=false` and `optimizer_restored=false`;
- the same corpus fingerprint and exact full-byte hash manifest;
- the same sorted validation-game list, count, and
  `validation_game_seed_set_sha256`;
- the same training-game order, batch count, optimizer-step count, and
  non-value loss weights.

The trainer's `<report>.validation_seeds.json` artifacts are compared directly.
Calibration must consume the identical seed list from the corresponding raw
shards. The older fixed `DEFAULT_HOLDOUT_BLOCKS` cannot bind this experiment.

### Software preconditions

- MSE mode trains scalar only; HL mode trains categorical only unless
  `--hlgauss-scalar-aux-loss-weight` is explicitly nonzero.
- `value_categorical_head` is in the value LR group.
- trained checkpoints carry positive `value-training-v1` provenance; a
  config-only `catbins:33` checkpoint fails closed for categorical readout.
- internal H2H supports categorical candidate versus scalar baseline.
- phase calibration selects scalar/categorical explicitly and emits legal-width
  buckets `1`, `2-4`, `5-10`, `11-20`, `21-40`, `41+`.
- external neutral-harness search explicitly selects scalar/categorical
  readout, includes it in fingerprints/artifacts, and fails closed through the
  centralized trained-readout provenance validator.
- the selected search semantics are representable in generator, typed internal
  H2H, and neutral external configs: D6 + threshold, adaptive wide budget +
  threshold + always-full, and any winning D1
  `rescale_noise_floor_c`/`sigma_eval`. Cross-tool D1, D6, and adaptive-budget
  semantics are now implemented and regression-tested; the external veto must
  consume the same typed configuration selected by S1-S3.
- `ulimit -n 65536`; no unrelated B200 GPU process is present.

## A0 — exact gen2B reuse-failure replication

> **Executed historical protocol — do not rerun in this wave.** The commands,
> invariants, and thresholds below are retained as the audit record for the
> completed A0 result. The binding verdict is `retain_scalar_for_a1`; changing
> the HL bins, sigma, or any other mechanism would require a new predeclared
> experiment rather than an A0 retry.

### Question

On the exact distribution and recipe where scalar validation worsened
`.6652 -> .8090 -> .8418`, does HL-Gauss remain stable for three exposures?

Locked inputs:

- data: `runs/memmap_gen2_20260706`, 3,648,516 rows;
- init: `runs/bc/gen1_20260705/checkpoint.pt`;
- recipe/report authority: `runs/bc/gen2B_20260706/report.json`;
- validation: exact game-level seed set reconstructed from the historical
  report and corpus;
- three epochs, with the old scalar settings retained in both arms.

If the historical trace used a validation scheme that cannot be reconstructed
as a game-level holdout, disclose it and run a new matched stress test under a
frozen game split. Do not label the new test an exact replication.

### Only changed variable

| setting | A0-MSE | A0-HL |
|---|---:|---:|
| init function | gen1 | bit-identical gen1 + deterministic cat head |
| primary objective | scalar MSE | 33-bin HL-Gauss CE |
| primary objective weight | historical locked value | same value |
| scalar auxiliary | n/a | 0 |
| HL sigma/bin width | n/a | .75 |
| value target | historical locked source | same source projected distributionally |
| search/calibration readout | scalar | categorical expectation |
| optimizer state | fresh | fresh |

Create the behavior-preserving categorical init:

```bash
set -euo pipefail
cd /home/ubuntu/catan-zero
PY=.venv/bin/python
GEN1=runs/bc/gen1_20260705/checkpoint.pt
RUNDIR=runs/rl_program_20260709/a0_gen2b_hlgauss

"$PY" tools/f69_upgrade_checkpoint_config.py \
  --in-checkpoint "$GEN1" \
  --out-checkpoint "$RUNDIR/gen1_catbins33_init.pt" \
  --flags catbins:33 --seed 1 --device cpu
sha256sum "$RUNDIR/gen1_catbins33_init.pt" >> "$RUNDIR/inputs.sha256"
```

Required upgrade evidence: `forward_max_diff == 0.0`, source checkpoint SHA-256
present, initialization seed `1`, and trained readouts still scalar-only.

Both arms were launched from the exact normalized recipe, with
`--no-resume-optimizer --save-each-epoch` to both and changing only
`--value-head-type`/init checkpoint. Do **not** substitute the later gen3
production recipe when reproducing or auditing A0.

### A0 metrics and hard verdict

For every epoch record trained `primary_value_loss`, clean/truncated loss and
weight mass, Brier, RMSE, ECE/reliability, correlation, policy loss/KL/top-k,
effective weights, examples/s, and peak VRAM. Slice calibration by phase,
forced/unforced, and the six legal-width buckets above.

A0 is valid only if A0-MSE reproduces the historical regression direction and
material magnitude. If it does not, stop and resolve recipe/corpus/init/split
drift; do not interpret A0-HL.

A0-HL passes when all are true:

1. epoch 3 primary CE is no worse than epoch 1 and neither later epoch regresses
   by more than 1% from its predecessor;
2. global Brier/RMSE do not regress by more than 2% versus the matched scalar
   checkpoint, at least one improves, and no critical phase/`41+` bucket
   regresses by more than 5%;
3. unforced policy loss and prior KL regress by no more than 2%;
4. categorical provenance is positive and calibration/search actually consumes
   `value_categorical`.

A0 proves a mechanism. It does not itself select a production checkpoint.

## Search teacher — strict local-plan sequence

D6 evidence is already positive: the 12-orientation RMS reduction is
approximately `sqrt(12)`. The remaining bounded probes are:

1. **Post-D6 Q calibration.** Baseline `.03/D1-off` versus five arms:
   `.03/on`, `.1/off`, `.1/on`, `.3/off`, `.3/on`. Every D1 arm names the
   checkpoint-specific `sigma_eval` artifact; the placeholder `.79` cannot
   bind. Select only an H1 winner, then require the winning semantics to be
   expressible in generator, internal H2H, and neutral external configs before
   using it for A1.
2. **Global n128.** Same checkpoint, D6, c-scale, and readout; n128 versus n64.
   Screen 50 pairs and confirm 200. Adopt only at pentanomial +15 Elo H1 and an
   attributable search cost below 1.6x (up to 1.8x only for a clear margin).
3. **Adaptive n256.** Base n128 both sides; candidate n256 only at `>=40` legal
   actions, D6 independently at `>=20`, selected wide roots always full.
   Screen 50 pairs and confirm 200 only if positive. Adopt at H1 or a predeclared
   >=15% cross-seed JS-stability gain with non-worse top-1 agreement, provided
   whole-game overhead is <=20%.
4. **p_full .40** is a separate single-dose experiment after the budget winner.

The H2H output's combined elapsed time is not a per-role cost measurement. A
fixed-root timing artifact is required. Likewise “wide-root stability” means
the repeated-root JS/agreement metrics above; an informal log impression cannot
select n256.

## A1 — one-dose fresh 35M production tournament

A1 is the actual next-model experiment. The data lane produces the window; this
R&D lane validates the contract and trains on B200.

### Fresh data contract

- exactly 12,000 complete, globally unique games before row expansion and no
  VAL-ONLY overlap: 9,600 (80%) current producer under the winning immutable
  search manifest, 1,800 (15%) recent/older champions, and 600 (5%)
  hard-negative or RGSC restart games. This preserves whole games while landing
  in the local 3–5M-row range;
- source category, checkpoint SHA-256, search config hash, and row counts in
  metadata;
- public-observation features, zero invalid teacher actions, audited truncation,
  forced fraction, phase/decision-index mix, legal width, target entropy, and
  active full-search policy mass;
- fixed 5% game holdout for the scalar A1 run, with
  `validation_max_samples=0` so every selected game's rows remain in the
  validation artifact;
- the existing n64 fresh window remains a named control source and is not
  silently relabelled as stronger-teacher data.

The machine-readable boundary for this section is
`configs/experiments/a1_pre_wave_contract.template.json`, sealed/rendered by
`tools/a1_pre_wave_contract.py` only after A0/S1-S3 resolve every marked
science field.  Its 40 × (245 current + 47 history + 16 hard-negative)
bounded attempts deterministically select the lowest-seed complete 240/45/15
per physical GPU, making 9,600/1,800/600 exact rather than relying on a random
realized mix or assuming truncations never occur.  A seal binds full
search/evaluator configs and their
hashes, readout and checkpoint bytes, A0/S1-S3 evidence,
generator/learner/guard bytes, the complete transitive local runtime tree, the immutable pre-claim seed-ledger snapshot,
and disjoint non-VAL seed plan. Rendering emits the exact ledger row for every
job; verification permits append-only disjoint ledger growth, and post-wave
audit requires all exact own claims. Rendering is
non-executing; its mandatory post-wave audit must attest the exact complete
game quotas and row/provenance diagnostics above before A1 training can read
the corpus.  The A0 learner decision and the teacher search readout are bound
separately. A0 resolved the learner side to scalar MSE/readout for A1 and leaves
the scalar gen3 producer unchanged; S1-S3 may select different search semantics
without changing that learner objective. No categorical branch remains open in
this wave.

After the post-wave audit passes, memmap conversion must consume both audit
sidecars and the exact audited shard inventory.  It is not valid to rebuild a
corpus from other shards that merely reuse the selected seeds:

```bash
A1_RAW=runs/selfplay/a1-fresh-mixed-12000games
A1_AUDIT="$A1_RAW/a1_post_wave.audit.json"
A1_SELECTED="${A1_AUDIT%.json}.selected_games.json"
mapfile -t A1_SOURCES < <(find "$A1_RAW" -mindepth 1 -maxdepth 1 -type d -print | sort)

"$PY" tools/build_memmap_corpus.py \
  --source "${A1_SOURCES[@]}" \
  --out runs/memmap_a1_fresh_mixed_12000games \
  --selected-game-seed-manifest "$A1_SELECTED" \
  --a1-post-wave-audit "$A1_AUDIT"
```

The resulting memmap retains all 12,000 selected complete games, including the
audited validation games.  Reserve, truncated, incomplete, unselected, and
same-seed substitute rows are rejected or excluded before corpus sizing and
statistics. Direct A1 job attestations also block the generic converter path,
and the converter content-addresses every flat payload file. Before any
optimizer is constructed, `train_bc` verifies that complete inventory plus the
contract-bound 208-file local runtime tree and persists both digests into the
report/checkpoint provenance.

### A1 scalar recipe selected by A0

Shared:

- gen3 warm start; 35M entity graph, hidden 640, 6 layers, 8 heads, dropout .05;
- one epoch/one dose, batch 4096, BF16, immutable example order and steps;
- fresh Adam (`--no-resume-optimizer`), LR `3e-5`, warmup 100, flat
  schedule, value LR `.3x`;
- policy weight `1`, soft policy target weight `.9`, target temperature `.7`;
- primary value weight `.25`, `value_target_lambda=1`, truncation weight `.25`;
- final-VP/Q/uncertainty/subgoal/KL-anchor losses all zero for this isolation;
- public mask on, train-time D6 augmentation off in this first comparison;
- save/check/measure every epoch artifact and full provenance.

These are not advisory flags: the exact effective recipe (including single
process/global batch, fresh optimizer, all confounder losses off, sample
weights, masking, graph-history features, and BF16) is hash-bound in the A1
contract.  `train_bc` auto-detects an A1 memmap and refuses any recipe,
payload-byte, or learner-code drift before the first optimizer step.

The binding A0 verdict removed the categorical arm from this wave.  A1 uses
scalar MSE/readout and must save positive `value-training-v1` provenance.  A
new categorical formulation may return only as a separately predeclared
mechanism experiment; it cannot be smuggled into A1 by changing bins/sigma.

Runnable B200 launch once the data contract and input hashes pass:

```bash
set -euo pipefail
cd /home/ubuntu/catan-zero
ulimit -n 65536
export PYTHONPATH=src
PY=.venv/bin/python
DATA=runs/memmap_a1_fresh_mixed_12000games
GEN3=runs/bc/gen3_20260706/checkpoint.pt
RUNDIR=runs/rl_program_20260709/a1_fresh35
A1_AUDIT=runs/selfplay/a1-fresh-mixed-12000games/a1_post_wave.audit.json
A1_VALIDATION="${A1_AUDIT%.json}.validation_seeds.json"
mkdir -p "$RUNDIR"

COMMON=(
  --data "$DATA" --data-format memmap
  --data-loader-workers 2 --data-loader-prefetch 2
  --arch entity_graph --track 2p_no_trade --vps-to-win 10
  --graph-history-features --seed 1
  --hidden-size 640 --graph-layers 6 --attention-heads 8 --graph-dropout 0.05
  --epochs 1 --save-each-epoch --batch-size 4096
  --amp bf16 --optimizer adam --weight-decay 0 --no-fused-optimizer
  --no-resume-optimizer
  --lr 3e-5 --lr-warmup-steps 100 --lr-schedule flat --value-lr-mult 0.3
  --policy-loss-weight 1 --soft-target-source policy --soft-target-weight 0.9
  --soft-target-temperature 0.7 --soft-target-min-legal-coverage 0.5
  --value-loss-weight 0.25 --value-categorical-loss-weight 0
  --hlgauss-scalar-aux-loss-weight 0 --value-hlgauss-sigma-ratio 0.75
  --value-target-lambda 1 --final-vp-loss-weight 0 --q-loss-weight 0
  --policy-kl-anchor-weight 0 --value-uncertainty-loss-weight 0
  --aux-subgoal-loss-weight 0 --truncated-vp-margin-value-weight 0.25
  --forced-action-weight 0.1 --forced-row-value-weight 1
  --winner-sample-weight 1 --loser-sample-weight 0.3
  --validation-fraction 0.05 --validation-seed 17
  --validation-max-samples 0
  --validation-game-seed-manifest "$A1_VALIDATION"
  --mask-hidden-info --no-symmetry-augment
  --skip-teacher-quality-gate --trust-curated-data-quality --require-35m-model
  --progress-every-batches 50 --train-diagnostics-every-batches 0
  --allow-concurrent-bc --device cuda:0
)

CUDA_VISIBLE_DEVICES=0 "$PY" tools/train_bc.py "${COMMON[@]}" \
  --init-checkpoint "$GEN3" --value-head-type mse \
  --checkpoint "$RUNDIR/mse/checkpoint.pt" \
  --report "$RUNDIR/mse/report.json" \
  2>&1 | tee "$RUNDIR/mse.log"
```

The trainer writes the run's validation seed manifest. Derive the calibration
selection from the raw shards with the same fraction/seed and compare the
sorted seed arrays byte-for-byte before accepting any metric:

```bash
PY=.venv/bin/python
RUNDIR=runs/rl_program_20260709/a1_fresh35
RAW_FRESH_WINDOW=runs/selfplay/a1-fresh-mixed-12000games
A1_MSE_EPOCH="$RUNDIR/mse/checkpoint_epoch0001.pt"

"$PY" tools/phase_sliced_value_calibration.py \
  --shard-dir "$RAW_FRESH_WINDOW" \
  --checkpoint "$A1_MSE_EPOCH" --value-readout scalar \
  --validation-fraction 0.05 --validation-seed 17 --require-held-out \
  --write-validation-seed-manifest "$RUNDIR/calibration.validation_seeds.json" \
  --device cuda:0 --out "$RUNDIR/mse.calibration.json"
```

The calibration seed list must equal the trainer manifest; a different held-out
set invalidates the run.

### A1 strength and promotion

1. 50-pair internal screen with the scalar A1 candidate versus frozen gen3,
   both using the selected S3 search operator;
2. 200-pair paired/seat-swapped confirmation, extended with disjoint seeds when
   inconclusive; promote internally at +15 Elo H1, reject at -10 Elo H0;
3. fixed neutral catanatron referee panel versus `catanatron_value`, using the
   scalar readout, 500 paired seeds/1,000 games. External -10 Elo
   H0 is a veto;
4. immutable high-regret/opening suite. A paired H0 or >5% regression in a
   critical phase/`41+` bucket is a veto;
5. n64 confirmation every third promotion.

The neutral artifact must report `value_readout=scalar` and positive
trained-readout provenance for A1; a default-only artifact cannot be
substituted.

## C0 — closed for this wave

A0 rejected the tested HL-Gauss formulation, so the historical 87.85M
categorical reuse stress is not on the pre-wave or A1 path.  The artifact
identity below is retained for a future, newly predeclared value formulation;
it is not authorization to spend this compute now.

The actual epoch-1 artifact is confirmed as
`runs/bc/bignet91M_20260707/checkpoint.pt` (SHA-256
`bac77a2ae41ad3d8d6327d0b9f3f591e25882d3eee9515f110574204eb76602e`),
with one-epoch report SHA-256
`f975642f6ffb3b772cbfaf62c9a367a4b11862516587280cb269f79cac9d3049`.
The historical continuation is a separate directory,
`runs/bc/bignet91M_20260707_ep2/checkpoint.pt`, and its report explicitly names
the epoch-1 file as `init_checkpoint`; their SHA-256 values are respectively
`8bbeb872919358c65c9f9f18463ad1926ba240f42ff093ce617c0c56b32abc44`
and `26d83f8e5bf6a736c44378c6f6e5cd53a42af518d1772b26699c6327befdb637`.
A generic checkpoint from any other directory remains forbidden.

The archived protocol would branch that same epoch-1 checkpoint into:

- C0-MSE: scalar primary;
- C0-HL: deterministic catbins33 upgrade, categorical primary, scalar aux 0.

Both use fresh Adam, identical historical data/split/order, FP32, batch 1024,
and two post-branch epochs. C0 passes only if HL remains stable through both
exposures and avoids the scalar failure without policy/calibration regression.
It makes **no** 35M-versus-87.85M strength or throughput claim.

Passing a future C0 would mean only “the new value formulation survives this
historical capacity stress.” Production 80–100M remains behind 10M fresh
audited rows, two seeds, equal exposure against the selected 35M objective,
explicit initialization/copy-fraction provenance, and attributable same-search
timing.

## Later action-local arm — isolated, not a rewrite

The current network scores actions mainly from global state. The later arm lets
legal actions query their target board entities while keeping width, depth,
value objective, search, and auxiliary heads unchanged.

Already present:

- deterministic `f69` upgrade seed/provenance;
- independent `--action-module-lr-mult`;
- `--freeze-modules value_heads`, covering scalar/categorical/final-VP/
  uncertainty and optional value-attention-pool modules while leaving policy
  and trunk trainable;
- `gather,cross:2` module path.

Run the short policy warmup with `--freeze-modules value_heads`, then two
independent module seeds on at least 10M fresh rows, equal rows and wall clock
versus the unchanged promoted 35M baseline objective (currently scalar).
Promote only if both seeds improve opening/`41+`
metrics and the pooled paired gate accepts H1 without neutral/high-regret
regression.

Graph-distance/adjacency bias is not implemented and has no current experiment
row. It is explicitly deferred rather than bundled into this arm.

## Stop list

- no full D6-equivariant transformer;
- no global n256 default;
- no simultaneous 90M + cross-attention + auxiliaries bundle;
- no PPO conversion, learned MuZero dynamics, CFR/ReBeL rewrite, or bespoke
  belief trunk in this phase;
- no production decision from training loss alone;
- no H100 production launch from this R&D protocol.

The governing critical path is now: **A0 scalar-retention verdict ->
D6/c-scale -> global n128 -> adaptive n256 -> sealed fresh-data handoff -> A1
fresh 35M scalar candidate -> conditional scale/action-local work**.
