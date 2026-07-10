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

The dose claim is keyed by the sealed contract SHA-256 at
`<seed-ledger-dir>/.a1-one-dose-training-claims/<contract-hex>.json`—not by
`--receipt`. Choosing different output or receipt paths therefore cannot create
a second dose. The claim is permanent:
it begins as `claimed` and is durably replaced with `complete` or `failed`
terminal evidence before the optional human-facing receipt is published. If
receipt publication fails after training, the terminal `complete` claim still
binds the candidate/report hashes and prevents a repeat.

Every claimed attempt also produces one no-clobber atomic receipt when its
receipt directory remains writable. A successful receipt binds the contract,
corpus payload inventory and row counts, validation manifest, producer, exact
command, candidate checkpoint, fresh optimizer sidecar, report, GPU, and exact
optimizer-step count. The executor rejects reports that do not semantically
prove the sealed corpus/init/checkpoint and exactly one complete epoch. A
nonzero or malformed run produces durable `failed` claim evidence and cannot be
mistaken for a candidate. Never delete or edit a contract claim to retry;
investigate it and issue a new sealed contract if another dose is authorized.
