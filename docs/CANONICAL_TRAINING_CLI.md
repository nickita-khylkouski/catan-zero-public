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

An eight-GPU B200 launch is therefore:

```bash
torchrun --standalone --nproc-per-node=8 tools/train.py \
  --config configs/training/a1_current_35m_b200.schema1.json \
  --data /path/to/memmap_composite.json \
  --checkpoint /path/to/candidate.pt \
  --report /path/to/report.json
```

`tools/train_bc.py` remains executable as an internal compatibility engine
because sealed historical receipts and the authenticated scratch/one-dose
executors bind its script path, functions, and bytes. It is not a supported
interface for new hand-authored runs. Once sealed replay and those executors are
routed through an explicit legacy adapter, the parser implementation can be
deleted from the engine entirely.

GitHub Actions workflows were removed. Cluster execution and local explicit
commands are now the only supported run surfaces.
