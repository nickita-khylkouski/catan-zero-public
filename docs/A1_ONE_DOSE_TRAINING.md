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
mistaken for a candidate. Never delete or edit a contract claim to retry.

There is one narrow, typed exception to issuing a new science contract. If the
v3 attempt failed before optimizer construction solely because the historical
argv omitted `--graph-layers` and therefore selected `4` against the sealed
six-layer producer, the executor may authorize one derived v4 repair. The
repair must add exactly `--graph-layers 6`, keep every other learner semantic
unchanged, use fresh outputs, and preserve the failed v3 claim and receipt. Its
stable retry identity keys a different `O_EXCL` claim, so changing r2 filenames
cannot mint a second retry.

If the v4 retry completed before an iteration state existed, adopt it without
running the learner again:

```bash
python tools/a1_iteration_orchestrator.py adopt-retry \
  --state /absolute/iteration.state.json \
  --lock /absolute/a1.lock.json \
  --data /absolute/a1-memmap \
  --validation-manifest /absolute/a1.audit.validation_seeds.json \
  --parent-claim /absolute/.a1-one-dose-training-claims/<v3-contract>.json \
  --retry-contract /absolute/r2/learner-retry.contract.json \
  --retry-receipt /absolute/r2/training.receipt.json \
  --python /absolute/learner-venv/bin/python \
  --gpu 0
```

`adopt-retry` is a read-only evidence replay followed by one atomic state-file
publication. It verifies the v3 claim/receipt, zero-step architecture mismatch,
sole command correction, retry identity and contract, v4 receipt and derived
claim, exact child environment, output semantics, and every output hash. It
never invokes `train_bc`; repeating the command validates and returns the same
sealed `dose_complete` state.
