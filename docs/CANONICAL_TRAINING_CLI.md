# Canonical training configuration adapter

`tools/train.py` is the compact config adapter for a commissioned learner
recipe. Fresh-scratch execution requires the stronger authority in
`tools/a1_scratch_train.py`, which authenticates the lock, admitted composite,
build receipt, topology, and planning receipt. The independently commissioned
parent-update recipe may use the compact adapter with an exact parent.

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
and model-admission settings live in one of two exact checked-in recipes:

```text
configs/training/a1_current_35m_b200.schema1.json         # fresh scratch
configs/training/a1_parent_update_35m_b200.schema1.json   # exact parent update
```

The launcher authenticates the complete normalized JSON payload and binds its
hash to its initialization role. Changing a field requires commissioning a new
recipe; changing only `initialization_mode` cannot turn one recipe into the
other.

The launcher decodes that recipe into `TrainConfig` and hands an in-memory
namespace to the engine. It does not reconstruct or parse the legacy
experimental CLI.

The B200 recipe also seals its input-pipeline topology: four background
materialization threads keep four memmap batches in flight. This overlaps host
reconstruction with the current GPU step without changing row order, loss
weights, batch size, or optimizer semantics. The iterator materializes only
that bounded window rather than allocating an array for every batch in the
epoch.

Plan the eight-GPU B200 scratch launch first:

```bash
python tools/a1_scratch_train.py \
  --lock /path/to/reviewed-lock.json \
  --data /path/to/memmap_composite.json \
  --composite-build-receipt /path/to/composite-build-receipt.json \
  --checkpoint /path/to/candidate.pt \
  --report /path/to/report.json \
  --receipt /path/to/authenticated-plan.json
```

Inspect and commission that exact plan, then rerun the same command with
`--go`. Do not add `--go` to an unreviewed plan and do not invoke
`tools/train.py` directly for this recipe.

The independently commissioned split1 FULL parent update is launched with the
separate parent recipe and an exact parent checkpoint:

```bash
torchrun --standalone --nproc-per-node=8 tools/train.py \
  --config configs/training/a1_parent_update_35m_b200.schema1.json \
  --data /path/to/authenticated_single_corpus.json \
  --init-checkpoint /path/to/exact-f7-parent.pt \
  --checkpoint /path/to/candidate.pt \
  --report /path/to/report.json
```

That recipe reproduces the commissioned 8×64 global batch, fresh AdamW,
48-step exact dose, flat `6e-5` learning rate with 16 warmup steps, full value
and shared-trunk routing, split1 value topology, and authenticated coherent
teacher semantics. It retains model-only review snapshots at steps 8, 12, 16,
24, and 32; the normal terminal `candidate.pt` is step 48. Candidate chaining,
optimizer resume, and grow-from initialization are rejected. The recipe does
not pin one turn's producer hash: the authenticated single corpus supplies its
exact policy-target identity, and the trainer admits it only when all active
policy rows resolve to one uniform operator identity.

The checked-in `a1_current_35m_b200.schema1.json` recipe remains the native-v5
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
`ROLL`/`END_TURN` boundaries. Production composite execution requires the
existing data-bound scratch authority. The compact adapter explicitly refuses
the checked-in scratch recipe so an unresolved schedule cannot be mistaken for
an authorized run.

`tools/train_bc.py` remains temporarily importable as an internal compatibility
engine because sealed historical receipts and the authenticated scratch/one-dose
executors bind its script path, functions, and bytes. It remains executable only
for those issued authorities; it is not a supported interface for new
hand-authored runs. Once sealed replay and those executors are routed through an
explicit legacy adapter, the parser implementation can be deleted from the
engine entirely.

GitHub Actions workflows were removed. Cluster execution and local explicit
commands are now the only supported run surfaces.
