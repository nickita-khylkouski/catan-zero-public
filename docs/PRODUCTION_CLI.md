# Production CLI

`catan-zero` is the operator interface for new two-player/no-trade runs. It has
five commands and no science flags:

```bash
catan-zero status
catan-zero plan /absolute/path/job.json
catan-zero prepare /absolute/path/job.json
catan-zero doctor /absolute/path/job.json
catan-zero run /absolute/path/job.json
```

From a checkout, `python tools/catan.py ...` is the equivalent command. The
job format is `catan-zero-production-job-v1`; its machine-readable schema is
[`configs/production/job.schema.json`](../configs/production/job.schema.json).

The job contains only run identity, immutable input locations, and physical
placement. Search, model, optimizer, and gate science resolve from exact
checked-in configs. Unknown keys and relative artifact paths are refused.
Approved recipes come from the same `configs/production_recipes.json` catalog
used by the compact launchers; this CLI does not maintain a second list of
recipe hashes.

Use `tools/loop.py` for a complete generate-to-promote improvement turn. Use
`catan-zero` to inspect, attest, and run an individual generation, learner, or
evaluation transaction.

## Safe launch sequence

1. Run `status` and confirm the desired pipeline is commissioned.
2. Run `plan` and archive the emitted plan if an external scheduler needs it.
3. For scratch training only, run `prepare`. This invokes the authenticated
   scratch planner without `--go` and creates the immutable plan receipt named
   by the job. Review that receipt; generation and evaluation do not use this
   step.
4. Run `doctor` under the interpreter that will launch the job.
5. Run `run`; it repeats the doctor immediately before starting the compact
   launcher and writes an atomic sibling `<run_id>.run.json` receipt.

The doctor requires the exact Python, PyTorch/CUDA, dependency, NVIDIA driver,
sealed native-wheel archive, and native capability identities in
`configs/runtime/a1_production_runtime.json`. It also requires a clean Git
worktree and re-hashes the job and every input immediately before launch. The
contract also checks every visible accelerator model: generation and evaluation
require H100s, while the currently commissioned learner recipes require B200s.
An eight-H100 host therefore cannot authorize a B200 learner merely because its
device count and software runtime match.

Generation may specify `gpu` to bind the child process through
`CUDA_VISIBLE_DEVICES`; one generator job owns one physical GPU. Training is
the exact eight-rank DDP topology. Evaluation accepts a typed `devices` list.

## Current authorization state

- Generation: commissioned coherent-public n128 recipe and exact prelaunch
  guard.
- Evaluation: commissioned coherent-public n128 candidate/champion recipe.
- Scratch training: routed through `tools/a1_scratch_train.py` and represented
  but blocked until the scratch optimizer schedule is commissioned in the
  current science contract. Its typed job binds the reviewed lock, composite
  descriptor, composite build receipt, and authenticated plan receipt.
- Parent-update training: the commissioned `a1-parent-update-35m-b200` recipe
  is ready and requires an exact `init_checkpoint`. It launches the cataloged
  compact trainer with eight DDP ranks and a fresh optimizer.
- PPO: represented but blocked. The retained exact-initializer canary was
  harmful and no canonical PPO recipe exists. The local H100 path now has a
  strict hashed v2 manifest, manifest-stamped shards, and per-update exact
  recovery, validated by a one-update smoke and restart. This does not
  commission PPO: the checked manifest is a template and the Modal wrappers
  are not yet v2-manifest-bound. See
  [`PPO_V2_MANIFEST_H100_SMOKE_20260716.md`](evidence/PPO_V2_MANIFEST_H100_SMOKE_20260716.md).

`plan` and `prepare` remain available for blocked scratch training so operators
can inspect and authenticate the exact future command and artifacts without
launching a learner. `doctor` and `run` refuse it. PPO gets
an immediate typed refusal instead of falling through to the historical
86-option research launcher.

The large historical executors remain available for authenticated replay and
R&D. They are not supported interfaces for new production runs.
