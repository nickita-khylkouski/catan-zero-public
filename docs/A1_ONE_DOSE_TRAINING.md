# A1 one-dose training transaction

Use `tools/a1_one_dose_train.py` for A1. Do not use the generic fleet
`role=train` path: it does not own the sealed A1 corpus/learner transaction.

The executor verifies the sealed contract (including the complete live claim
set and runtime tree), audited memmap payload inventory, exact selected games,
and immutable game-seed validation sidecar before it renders `train_bc`. The
current operator decision is global `n_full=128`; n64 and blanket n196/n256 are
refused. `p_full` remains whatever the sealed search contract selected—the
learner does not infer or change it.

The bound learner is one direct process on one selected B200: world size 1,
global batch 4096, one epoch, scalar MSE, fresh unfused Adam, LR `3e-5`, value
head LR multiplier `.3`, and no optimizer-state resume. Every other active or
disabled learner field comes from the exact recipe inside the seal and is
rechecked independently by `train_bc` before optimizer construction.

First run the default verified dry run:

```bash
python tools/a1_one_dose_train.py \
  --lock /absolute/path/a1.lock.json \
  --data /absolute/path/a1-memmap \
  --validation-manifest /absolute/path/a1.audit.validation_seeds.json \
  --checkpoint /absolute/fresh/output/candidate.pt \
  --report /absolute/fresh/output/report.json \
  --receipt /absolute/fresh/output/training.receipt.json \
  --gpu 0
```

Inspect the printed command and hashes. Add `--go` only on the chosen B200.
`--go` verifies the physical GPU name, pins that one device with
`CUDA_VISIBLE_DEVICES`, raises the child file-descriptor limit, and executes
without `torchrun`. Output paths must be fresh.

Every claimed attempt produces one no-clobber atomic receipt. A successful
receipt binds the contract, corpus payload inventory, validation manifest,
producer, exact command, candidate checkpoint, fresh optimizer sidecar, report,
GPU, and optimizer-step count. A nonzero or malformed run produces a failure
receipt and cannot be mistaken for a candidate. A stale `.claim` file means an
attempt was interrupted before receipting; investigate it rather than deleting
or rerunning blindly.
