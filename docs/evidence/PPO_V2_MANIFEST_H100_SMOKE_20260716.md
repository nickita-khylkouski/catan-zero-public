# PPO v2 manifest H100 smoke, 2026-07-16

This is execution evidence for the production plumbing, not evidence that PPO
improves play and not a promotion result.

## Scope

- Checkout: `e76cb5f` (`main` at execution time)
- Host: `192.222.54.137`
- Learner: NVIDIA H100 GPU 0
- Actor: NVIDIA H100 GPU 1
- Initializer: `/home/ubuntu/f7e93dfb.pt`
- Initializer SHA-256:
  `f7e93dfb8cdb713d647b3e142c949d59083de9f719b6688b6faa6c918ce3eed4`
- Bound manifest SHA-256:
  `14a1780f4a92152e34bf500d812a011f622cb6d0d6f98de0597e151b480f3a8d`
- Track: two-player, no trading, 3 VP diagnostic cutoff
- Opponent: fixed `random`
- Dose: two games, one shard, one learner update
- Learning rate: `1e-6`
- Evaluation and promotion: not run

The diagnostic used a separate `/tmp` run root and a one-step bound manifest.
It did not modify a model registry or production pointer.

## Result

The actor completed two games in 5.33 seconds and wrote one manifest-stamped
shard containing 66 samples. The learner accepted exactly that shard, applied
one update, and exited at the manifest's `max_steps=1` boundary.

The recovery transaction produced:

- `step_1.pt`: 140,324,675 bytes
- `step_1.opt.pt`: 267,331,115 bytes
- one consumed-shard frontier marker
- published policy version 2
- immutable `run_manifest_v2.json`

The update reported finite losses, 66 V-trace steps, zero bad trajectories,
and `vtrace_skipped=0`. No scoreboard was run because this was an execution
smoke, not an efficacy test.

## Restart proof

The learner was restarted on GPU 0 with the same manifest and initializer. It:

1. found `step_1.pt`;
2. authenticated and restored the optimizer sidecar;
3. finalized the already-consumed frontier idempotently;
4. restored RNG state;
5. republished the recovered step as policy version 3; and
6. exited without applying another update.

This verifies the bounded local path from one bound manifest through actor
shard identity, learner filtering, per-update recovery, and exact resume.

## Remaining boundary

PPO remains uncommissioned. The retained exact-initializer efficacy canary was
negative, the checked v2 manifest is still a template, and the Modal wrappers
do not yet consume/stamp the v2 manifest. New production PPO jobs must remain
blocked until those wrappers are wired and a reviewed recipe has positive
bounded evidence.
