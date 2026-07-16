# Canonical training entrypoint

New learner runs use `tools/train.py`, not the internal `tools/train_bc.py`
engine.

The public entrypoint exposes eight options:

```text
--config
--data
--checkpoint
--report
--init-checkpoint
--device
--host-lock-file
--allow-concurrent-bc
```

Architecture, optimization, masking, sampling, value objectives, diagnostics,
and model-admission settings live in the checked-in typed recipe:

```text
configs/training/a1_current_35m_b200.schema1.json
```

The launcher decodes that recipe into `TrainConfig` and hands an in-memory
namespace to the engine. It does not reconstruct or parse the legacy
experimental CLI.

The B200 recipe also seals its input-pipeline topology: four background
materialization threads keep four memmap batches in flight. This overlaps host
reconstruction with the current GPU step without changing row order, loss
weights, batch size, or optimizer semantics. The iterator materializes only
that bounded window rather than allocating an array for every batch in the
epoch.

An eight-GPU B200 launch is therefore:

```bash
torchrun --standalone --nproc-per-node=8 tools/train.py \
  --config configs/training/a1_current_35m_b200.schema1.json \
  --data /path/to/memmap_composite.json \
  --checkpoint /path/to/candidate.pt \
  --report /path/to/report.json
```

The checked-in `a1_current_35m_b200.schema1.json` recipe is the native-v5
**fresh-scratch** architecture recipe.  It is not the checkpoint-initialized
coherent dose-frontier experiment.  In particular, the measured 32-step
frontier must not be copied into this recipe: that evidence started from the
exact parent checkpoint, while this recipe constructs a new 41.7M-parameter
model with history-v2, rule/card inputs, structured action residuals, and a
private value block.  Its optimizer horizon and shared-value routing remain a
separate commissioning decision.

The public recipe does bind the loss semantics that are already safe to carry
forward: complete whole-game composite validation (`validation_max_samples=0`),
no phantom MoE objective when no experts exist, pure search-policy fallback,
outcome-only scalar value targets, and full value supervision at forced
`ROLL`/`END_TURN` boundaries.  Production composite execution still requires
the existing data-bound scratch authority; the compact launcher does not turn
an unresolved scratch schedule into an authorized run.

`tools/train_bc.py` remains temporarily importable as an internal compatibility
engine because sealed historical receipts and the authenticated scratch/one-dose
executors bind its script path, functions, and bytes. Direct execution now
refuses immediately; it is not a supported interface for new hand-authored
runs. Once sealed replay and those executors are routed through an explicit
legacy adapter, the parser implementation can be deleted from the engine
entirely.

GitHub Actions workflows were removed. Cluster execution and local explicit
commands are now the only supported run surfaces.
