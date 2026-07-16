# Catan Zero differential review — 2026-07-16

## Scope

- Repository: `nickita-khylkouski/catan-zero-public`
- Branch: `main`
- Reviewed range: `7ae188c^..63341b1`
- Primary focus: 2-player, no-trading learner signal, authenticated data flow,
  native runtime identity, and mutation/promotion safety
- Data host: B200 `149.118.65.110`, read-only inspection and transfer only
- Test host: eight-H100 node `192.222.55.12`

This was a differential and adversarial review, not a claim that every commit
in the range was authored by one reviewer. Concurrent changes were rebased and
reviewed before each push.

## Outcome

The reviewed tree closes seven high-impact failure classes. The most important
new control is a pre-step policy-signal admission gate: the canonical and
sealed scratch launch paths now refuse a corpus whose final policy weights are
too sparse or concentrated to produce a credible learning signal.

No unresolved code-level release blocker was found in the final tree. A full
scratch retrain was intentionally not started; the remaining risks are
experiment-level and are listed below.

## Findings and dispositions

### F1 — Sparse policy supervision could pass preflight

- Severity: High
- Status: Fixed in `63341b1`
- Affected path: `tools/train_bc.py`, canonical training recipe, current
  science contract, and `tools/a1_scratch_train.py`

The old coverage sampler rejected only exactly zero policy mass. The failed
Stage-C corpus had 8,178 policy-active rows among 959,142 rows (0.853%), yet it
was admissible. Its policy Kish ESS was only 6,835.96 rows. At global batch 512
this represents 4.366 active and 3.649 effective policy rows per optimizer
update. Broad value supervision therefore dominated the shared trunk while a
small policy population was repeatedly reused.

The fix computes Kish ESS from the final training policy weights after the
authenticated split, policy scope, phase weighting, equal-per-game weighting,
forced-row exclusion, and coverage importance. It scales the population ESS
fraction to the synchronous global batch and enforces a recipe-bound minimum
of 32 effective policy rows.

The scan is chunked and runs through the rank-0 authoritative-call mechanism on
single-node DDP, broadcasting either the report or refusal to all ranks. This
avoids eight redundant full-corpus temporary allocations and prevents peers
from hanging when rank 0 refuses the corpus.

The exact current-corpus preflight on the H100 host produced:

| Measure | Result |
| --- | ---: |
| Training rows after whole-game holdout | 13,187,313 |
| Policy-active training rows | 1,639,264 |
| Active rows per global-512 update | 63.645 |
| Kish ESS rows | 1,056,056.953 |
| Effective rows per global-512 update | 41.002 |
| Commissioned floor | 32.000 |

Thus the current corpus passes with a measured 28% margin, while the failed
corpus is 8.77 times below the floor.

### F2 — Optimized native search reused an existing wheel identity

- Severity: High
- Status: Fixed in `34aeab5` and `4156fed`

Native dense-action-map search changes landed while the build still identified
itself as `catanatron-rs 0.1.11`. That created an artifact identity collision:
the source behavior changed without changing the immutable runtime name.

The native source, builder contract, manifests, and production runtime were
bumped to 0.1.12. The builder now runs the complete Gumbel Rust library suite
in addition to the core native feature tests.

Release `v1.8-optimized-native-search` is pinned to runtime commit `4156fed`.
Its wheel SHA-256 is:

`e8e61626e5e99c9c61dcf79e3cc639d2070eb73dc79ea99a50efdece4cf34765`

The GitHub-downloaded asset, local artifact, release inventory, and build
receipt agree. A clean exact-tag H100 smoke verified version 0.1.12, required
feature/search exports, native capabilities, and the supported generation and
evaluation CLIs.

### F3 — Canonical generation payload and semantic hash diverged

- Severity: High (production availability)
- Status: Fixed in `71897a9`

A concurrent edit changed the canonical generation JSON without updating the
launcher’s authenticated semantic SHA. Every canonical generation launch
therefore failed closed, including otherwise valid commands. The hash was
recomputed from the exact commissioned payload and the launcher/config tests
were rerun.

### F4 — Authenticated coherent corpus shards could escape their root

- Severity: High (data integrity)
- Status: Fixed in `9abb66a`

The executor and posthoc materializer accepted absolute paths, `..` traversal,
or symlink-resolved shard paths outside the authenticated corpus directory.
Path confinement now applies to both consumers before a shard is admitted.
Tests cover absolute, traversal, and symlink escape variants.

### F5 — Legacy league promotion could reuse stale evidence

- Severity: High
- Status: Fixed in `a6cfd84`

The legacy promotion cache omitted checkpoint bytes, VP target, and decision
cap from its reuse identity. It could compare against a replaced champion,
reuse 6-VP evidence for a 10-VP gate, or overwrite the champion using hardcoded
historical counts.

Live overwrite now requires explicit acknowledgement, 10 VP, at least 200
games per leg, at least 1,200 decisions, fresh evaluation of both sides across
all eight legs, and atomic replacement. Diagnostic-only weak configurations
remain available without mutation.

### F6 — Retired operational paths remained capable of live mutation

- Severity: High
- Status: Fixed in `2520298`, `b99c315`, and `4bc1da8`

Retired JSON fleet mutation, stale Modal Gumbel launch, and legacy GCP
`cluster_big_push` paths could still make live changes. Each now fails closed
unless its exact legacy acknowledgement is supplied, with guards placed before
directory creation, remote refresh, subprocess launch, or fleet mutation.

### F7 — Historical learner evidence exposed silent non-learning

- Severity: High
- Status: Current code repaired; must be revalidated during retrain

The B200 Stage-C evidence showed:

- `event_encoder`: 438,400 parameters, zero gradient, zero update RMS, and zero
  relative delta because a hidden overlay froze a zero-initialized history
  gate;
- objective-interference cadence of 64 for a 32-step run, so policy/value
  gradient conflict was never measured;
- one attempted control that never reached an optimizer step because
  `RLIMIT_NOFILE=1024` was below the required 65,536;
- positive teacher-gap closure only at parent KL about 0.113–0.115, roughly
  3.8 times the 0.03 trust budget.

Current code removes the hidden history freeze, requires feature-module
learning observations, uses a 16-batch diagnostics/interference cadence, and
binds resource-limit preflight in production execution. These controls prove
that modules receive gradients and updates; they do not by themselves prove
playing-strength improvement.

## Concurrent-change review

The final rebases also included and retained:

- DDP storage-topology fail-closed validation;
- coherent-arm sampler compatibility;
- direct policy-dose checkpoint accounting;
- initializer topology/byte binding for coherent learner arms;
- function-preserving value-tower upgrade authentication;
- one authenticated holdout authority for teacher-gap probes;
- exact prior-temperature ownership; and
- checkpoint feature-contract binding in ranking, target, and symmetry probes.

No conflicting implementation was overridden or force-pushed.

## Verification

- Policy admission, canonical launcher, scratch command/science contract,
  resume identity, and compatibility suite after final rebase:
  `255 passed, 1 skipped`.
- Broader runtime/wheel/installer/executor suite:
  `104 passed`.
- H100 pre-merge policy overlay:
  `53 passed`.
- Native 0.1.12 H100 build:
  core native feature tests and all 22 Gumbel Rust library tests passed.
- H100 native benchmark:
  250 searches, 38,500 simulations, 0.815878 seconds,
  306.418 searches/second.
- Ruff checks, Python compilation, and `git diff --check` passed for the
  changed training paths.

## Remaining experiment-level risks

1. The new gate is a population admission statistic. Shuffled update-level
   variance still needs to be observed in the retrain through realized policy
   rows, policy denominators, module gradients, and policy/value interference.
2. The retained PPO one-update canary scored 86/200 (43%, 95% CI
   36.14–49.86) against its exact initializer with a -0.755 VP margin despite
   small reported KL. No multi-step on-policy campaign should proceed without
   an updated-policy-versus-exact-initializer paired-seed gate.
3. The 32-row floor is commissioned for the current global-512 recipe and
   current authenticated corpus. A materially different batch topology or
   corpus must carry an explicit new floor and will be measured again.
4. No long training run was performed during this review. The next scratch run
   should stop at the early checkpoint frontier and require module-signal,
   teacher-gap, KL, and matched H2H evidence before continuing.
