# A1 short-dose module commissioning audit — 2026-07-13

## Decision

Do not interpret the selected TEMP schedule (8 ranks x 512 rows, 128 optimizer
steps, 100-step warmup, LR 3e-5) as a fair negative test of a newly introduced
head or adapter.

The mature f7 network receives 78.5 full-LR-equivalent updates.  A fresh value
head at `value_lr_mult=0.3` receives only 23.55 base-LR-equivalent updates.  The
zero-output target-gather adapter receives 78.5 equivalents at multiplier 1,
and its LayerNorm/target-token branch receives no gradient on the first step
because the following linear matrix is initialized to zero.  Those schedules
measure short-dose stability of a mature model, not learnability of a new
module.

## Smallest decisive protocols

### Target gather

Commission only `target_gather_proj` on the same 524,288 row draws:

- 8 ranks x 64 rows = global batch 512;
- 1,024 optimizer steps, 100-step warmup;
- LR 3e-5, `action_module_lr_mult=4`;
- freeze `trunk,action_encoder,policy_head,value_heads`;
- require the only trainable prefix to be `target_gather_proj`;
- initialize by a deterministic, forward-bit-identical gather-only upgrade of
  exact f7;
- require valid target-token incidence on every policy/value-supervised TEMP
  component, including replay.

This preserves the selected 524,288-row measure but supplies 974.5 integrated
LR-step equivalents, or 3,898 action-module equivalents.  It answers the narrow
question: *do fixed f7 target-token features contain useful action-local
signal?* It is not a joint candidate and cannot be promoted directly.
A single seed is a screen, not a mechanism rejection: a positive screen moves
to an independently initialized integrated A/B; a negative screen must be
repeated with separately sealed seed/data-order evidence before closing gather.

### Pure search targets

Keep the old 4.19M-row pure-soft launcher fail-closed.  Its dose is a causal
confound.  Use `tools/a1_selected_dose_pure_soft_arm.py` instead.  It derives
from the executed 128-step TEMP control and changes only:

- `soft_target_weight: 0.9 -> 1.0`;
- implied played-action hard CE mass: `0.1 -> 0.0`.

This retains the high-information hypothesis that `action_taken` contains
post-temperature behavior noise, while preserving f7, component temperatures,
sampler, value objective, optimizer, LR trajectory, and 524,288-row dose.

### Categorical/value heads

Do not launch `a1_mixed_value_objective_probe.py` or the architecture wrapper.
They combine a non-selected LR, current-only component scope, forced-row value
weight 0.1, per-game/sqrt weighting, and uncapped whole-corpus dose.  A future
value-head experiment must first commission the fresh head independently (or
freeze/lower the mature surface) and only then run a matched integrated arm.
The old negative HL-Gauss result does not close the mechanism.

The historical belief-resource executor is likewise fail-closed, and the
auxiliary plan is explicitly marked obsolete-dose.  Both hard-code 4.19M rows
for fresh heads.  They remain useful decoders of the intended causal axes, but
must be rebuilt on the same TEMP+geometry bridge (with head commissioning)
before consuming GPUs.

## Proven provenance bridge

The historical full-dose TEMP manifest remains the immutable recipe/data/f7
identity, but its old selection receipts and execution checkout were cleaned.
The executed B200 geometry evidence provides the surviving selected-dose
runtime authority.  `a1_topology_gather_arm.py` and the selected-dose pure-soft
arm therefore require all three:

1. sealed production TEMP manifest (v1 or v2);
2. authenticated `a1-b200-microbatch-quality-plan-v1` plan;
3. completed `ddp8-b512/train.report.json`.

On `149.118.65.110`, the bridge was replayed successfully against:

- `/home/ubuntu/experimental_nonpromotable/learner-forensics-2ba5ae1/source-temp-r3.manifest.json`
- `/home/ubuntu/experimental_nonpromotable/learner-forensics-2ba5ae1/geometry-probe-128step-r4/plan.json`
- `/home/ubuntu/experimental_nonpromotable/learner-forensics-2ba5ae1/geometry-probe-128step-r4/ddp8-b512/train.report.json`

It authenticated the same f7/data/objective at 524,288 rows / 128 steps and the
surviving trainer checkout
`/home/ubuntu/catan-learner-geometry-5019432/tools/train_bc.py`.  The normalized
evidence digest was
`sha256:1ef5dd822495198c6e59d0bc7b8772e5e4474ed9a916e633b350ff0ecb4f13d1`.
No cleaned failed-candidate lineage artifact is required.

## Preparation commands

First make the deterministic gather initializer and the three-component target
coverage audit (paths are illustrative fresh diagnostic output paths):

```bash
python tools/f69_upgrade_checkpoint_config.py \
  --in-checkpoint /home/ubuntu/catan-zero-production/runs/learner/a1-infoset-n128-20260710-r2/candidate.pt \
  --out-checkpoint "$ROOT/f7-gather-init.pt" \
  --flags gather \
  --seed 1

python tools/audit_memmap_architecture_targets.py \
  /home/ubuntu/experimental_nonpromotable/a1-combined-80-20-20260711/n128/n128.memmap \
  /home/ubuntu/experimental_nonpromotable/a1-combined-80-20-20260711/n256-early/n256.memmap \
  /home/ubuntu/catan-zero/runs/memmap_a1_fresh_mixed_12000games \
  --out "$ROOT/architecture-targets.audit.json"
```

Prepare gather without launching:

```bash
python tools/a1_topology_gather_arm.py \
  --source-manifest /home/ubuntu/experimental_nonpromotable/learner-forensics-2ba5ae1/source-temp-r3.manifest.json \
  --selected-dose-plan /home/ubuntu/experimental_nonpromotable/learner-forensics-2ba5ae1/geometry-probe-128step-r4/plan.json \
  --selected-dose-report /home/ubuntu/experimental_nonpromotable/learner-forensics-2ba5ae1/geometry-probe-128step-r4/ddp8-b512/train.report.json \
  --gather-checkpoint "$ROOT/f7-gather-init.pt" \
  --architecture-audit "$ROOT/architecture-targets.audit.json" \
  --output-root "$ROOT/gather" \
  --repo "$CANONICAL_REPO"
```

Prepare pure-soft without launching:

```bash
python tools/a1_selected_dose_pure_soft_arm.py prepare \
  --source-manifest /home/ubuntu/experimental_nonpromotable/learner-forensics-2ba5ae1/source-temp-r3.manifest.json \
  --selected-dose-plan /home/ubuntu/experimental_nonpromotable/learner-forensics-2ba5ae1/geometry-probe-128step-r4/plan.json \
  --selected-dose-report /home/ubuntu/experimental_nonpromotable/learner-forensics-2ba5ae1/geometry-probe-128step-r4/ddp8-b512/train.report.json \
  --output-root "$ROOT/pure-soft" \
  --repo "$CANONICAL_REPO"
```

Both preparation commands write immutable manifests and report
`launched=false`.  A GPU job requires the separately explicit executor with
`--go`.
