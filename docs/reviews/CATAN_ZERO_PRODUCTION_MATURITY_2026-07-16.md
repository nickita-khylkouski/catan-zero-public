# Catan Zero production maturity assessment

Date: 2026-07-16  
Scope: two-player, no-trade generation, training, evaluation, and promotion  
Method: repository-wide static inventory, production-contract review, focused
tests, read-only retained-evidence inspection, and a bounded eight-H100 runtime
canary. This is an engineering maturity assessment, not a claim that the current
policy is strong or that a new training run is scientifically authorized.

## Executive result

Overall maturity is **moderate (2.6/4)**. The strongest part of the system is
fail-closed artifact and run identity: canonical JSON hashes, immutable inputs,
clean-commit checks, atomic receipts, restart checks, native capability gates,
and typed promotion transactions are already unusually thorough. The weakest
part is the path an operator must use to invoke them. The repository still has
270 Python tools, 3,643 literal `add_argument` calls in 238 tool files, no
automated repository workflow, and several very large orchestration modules.

The central production risk is therefore not a lack of safety primitives. It is
that those primitives are distributed across too many entry points and are not
automatically exercised whenever an authority changes. A concrete instance was
found during this audit: a generation-config change reached `main` without its
catalog digest being updated, so the exact production interface refused its own
checked-in recipe. The refusal was safe, but the integration should have been
caught before merge. The catalog was synchronized and the production doctor now
also binds accelerator model, preventing an eight-H100 runtime from authorizing
either B200-qualified learner recipe.

## Scorecard

Scores use 0 = absent, 1 = ad hoc, 2 = developing, 3 = strong, and 4 =
exemplary. “Decentralization” is interpreted as operational concentration and
recovery because this is a training system rather than a public consensus
protocol.

| Category | Score | Evidence-based assessment |
|---|---:|---|
| Arithmetic and numerical safety | 3/4 | Explicit precision recipes, parity/no-op gates, finite-output GPU canaries, and exact target-quality admission exist. Some training numerics remain distributed across large launchers, and Rust release overflow behavior is not explicitly configured. |
| Auditing and observability | 3/4 | Content-addressed plans, atomic stage/run receipts, logs, exact runtime identity, and promotion evidence make runs reconstructable. Evidence is filesystem-centric and there is no single queryable run index or automated retention check. |
| Access controls and authorization | 3/4 | New work fails closed on recipe, Git, artifact, wheel, capability, runtime, readiness, and hardware placement. SSH host access and operator privileges remain external to the repository, with no centrally enforced role boundary. |
| Complexity management | 1/4 | The compact `catan-zero` interface and seven-stage loop are meaningful improvements, but 270 tool modules, 3,643 flag definitions, a 39,776-line trainer, and a 5,591-line fleet controller remain a high change and review burden. |
| Operational concentration and recovery | 2/4 | Durable receipts, resumable stages, direct host transfers, and disaster-recovery evidence reduce single-process fragility. Canonical data, champion state, and orchestration still depend on a small number of SSH hosts and filesystem authorities. |
| Documentation quality | 3/4 | 124 documents cover recipes, incidents, architecture, evidence, and operational gates. Some documents are historical or conflicting; the production CLI is now explicit about the supported surface, but archival status is not consistently machine-readable. |
| Transaction ordering and concurrency | 3/4 | Atomic replace patterns, locks, stage dependencies, immutable receipts, and retry identity protect the improvement loop. Concurrent changes can still create cross-file authority drift, as the generation catalog incident demonstrated. |
| Low-level and native safety | 2/4 | The Rust extension is capability-gated and covered by Python/Rust parity and wheel-identity checks. Ten Rust files contain 38 `unsafe` references; release builds use `panic = "abort"`, and the buffer/FFI boundary needs a dedicated documented unsafe review. |
| Testing and verification | 3/4 | There are 430 test files, an explicit full-suite gate, parity suites, CLI goldens, no-op checkpoint checks, and exact H100 validation. There are no checked-in CI workflows, two gate quarantines, and native/GPU coverage is environment-sensitive. |

## Detailed findings

### 1. Arithmetic and numerical safety — 3/4

The codebase has strong empirical numerical checks: native/Python feature
parity, no-op checkpoint equivalence, precision-specific training recipes,
finite-output checks, target-quality admission, optimizer-resume validation,
and bounded H100 probes. The current training status also correctly separates a
commissioned parent update from unresolved scratch optimizer science.

The remaining weakness is consolidation. Numerically material behavior is
spread across `train_bc.py`, `train_ppo.py`, recipe files, guards, recovery
paths, and compact launchers. A future recipe change can therefore alter
clipping, accumulation, precision, schedule, or optimizer restoration without
one concise semantic diff. Rust's release profile does not explicitly state
overflow checks, which makes review of integer assumptions more difficult.

### 2. Auditing and observability — 3/4

The production plan records exact job, config, guard, input, source commit,
runtime, driver, native wheel, capability, command, environment, and receipt
identities. The full-turn coordinator journals each boundary atomically and
validates earlier hashes on resume. These controls make a completed transaction
substantially more reproducible than a conventional training shell script.

However, evidence remains a collection of JSON files and directories rather
than a unified ledger. Operators can still lose visibility through incomplete
copying, stale paths, or retention drift. A read-only B200 inventory found a
generator champion whose registry records only checkpoint/search identity and
explicitly says promotion proof was not recreated. That checkpoint is useful
for diagnostics but is not a fully proven production promotion.

### 3. Access controls and authorization — 3/4

Production authorization is fail-closed and recipe-specific. The interface
rejects unknown job fields, relative or mutable inputs, dirty commits, changed
plans, runtime drift, unsupported native capabilities, blocked science, invalid
resume identity, and wrong accelerator models. PPO remains represented but
blocked instead of silently falling through to an 86-option research launcher.

The repository does not control SSH identities, host sudo rights, cloud IAM, or
who can edit the champion registry. These are deployment responsibilities, but
they should be recorded as an explicit threat model and least-privilege runbook.
The production runner should never require a broadly privileged service account.

### 4. Complexity management — 1/4

The supported surface is now small: `catan-zero` has five commands and each
state-changing command accepts one typed job file; `tools/loop.py` coordinates a
complete turn from one config and state directory. This provides a viable
strangler path around historical flag-heavy tools.

The underlying complexity is still severe. Current inventory:

- 910 Python files, including 270 under `tools/`;
- 3,643 literal `add_argument` calls across 238 tools;
- 39,776 lines in `tools/train_bc.py`;
- 5,591 lines in `tools/gcp_fleet_controller.py`;
- 2,192 lines in `tools/train_ppo.py`;
- 2,274 lines in the production generation executor;
- 162 broad `except Exception` handlers in `src/` and `tools/`.

These numbers do not prove defects, but they do increase the number of possible
launch surfaces, make global invariants hard to review, and encourage stale
code to remain apparently authoritative. Historical replay must remain
possible, but archival entry points should be visibly separated from supported
production code.

### 5. Operational concentration and recovery — 2/4

Atomic receipts, restart verification, content addressing, direct transfers,
and recovery tools provide good protection against interrupted processes and
partial stages. The loop refuses to repeat a completed side effect after a
restart, and promotion is last.

Operational authority is nevertheless concentrated in a few filesystem roots
and SSH hosts. The retained B200 corpus and champion registry are valuable, but
their availability and interpretation depend on host access and path-specific
knowledge. A loss or silent divergence of registry, corpus receipts, or
checkpoint store would require manual reconciliation. Periodic restore drills
and an independently stored manifest are needed.

### 6. Documentation quality — 3/4

The repository contains 124 Markdown documents, including detailed research
postmortems, topology evidence, production recipes, promotion contracts, and
recovery procedures. The new production CLI document explicitly distinguishes
supported operator APIs from historical research executors.

The volume is also a navigation cost. Some older plans describe fleets or
recipes that are no longer current, and “latest” is inferred from prose and
dates. Each operational document should carry machine-readable status such as
`current`, `superseded`, `historical-replay`, or `research-only`, plus the
authority that supersedes it.

### 7. Transaction ordering and concurrency — 3/4

The seven-stage loop has clear dependencies: generate, harvest, audit,
composite, train, evaluate, promote. Commands are argument arrays, outputs must
be fresh, earlier receipts are rehashed, and state is written atomically. The
repository also uses file locks and more than one hundred atomic replacements
in stateful paths.

Cross-file change ordering is not yet automated. During the audit, a concurrent
generation-config change altered its canonical hash without updating the
production catalog. Main was safe but unusable for that recipe until the digest
was repaired. A required authority-consistency gate must run whenever a
cataloged config, guard, launcher, or runtime contract changes.

### 8. Low-level and native safety — 2/4

The native engine is not trusted merely by version: production requires the
exact installed wheel archive and a named capability set. Rust/Python parity,
action-context parity, symmetry checks, and native search tests reduce semantic
drift risk.

The native tree has ten Rust source files and 38 `unsafe` references, including
buffer/FFI work. It also contains 425 `unwrap`/`expect` call sites and seven
`panic!` sites across production and tests. Counts alone are not findings, but
the release profile's abort-on-panic behavior makes reachable panics an
availability boundary. Every production `unsafe` block needs a written safety
invariant, and production-reachable unwrap/panic paths should be classified.

### 9. Testing and verification — 3/4

The repository has 430 test files. `scripts/gate.sh` runs the full suite plus
native parity, CLI goldens, and a bit-identical champion no-op check. This audit
ran 129 focused production and training-contract tests locally. On an isolated
checkout of pushed commit `cac1466`, the supplied host reported eight NVIDIA
H100 80GB HBM3 devices; 42 exact-runtime production tests passed and an
eight-device BF16 matrix smoke returned finite results on every GPU. The final
placement gate was then tested from clean pushed commit `e01b246`: generation
passed with zero doctor errors, while the B200 parent-update recipe failed closed
with exit code 2 on the same eight H100s. The exact runtime and plan hashes are
preserved in
[`PRODUCTION_PLACEMENT_H100_E01B246_20260716.json`](../evidence/PRODUCTION_PLACEMENT_H100_E01B246_20260716.json).
No long training or generation job was started.

The gap is automatic enforcement. There are zero checked-in GitHub workflow
files, and the gate intentionally quarantines two tests. Local macOS cannot load
the Linux native extension, so a subset of environment-dependent tests fails at
collection there rather than exercising behavior. A clean Linux native/runtime
gate must be required before an authority-changing commit is accepted.

## Prioritized remediation roadmap

### P0 — required before the next from-scratch training campaign

1. **Make authority consistency a required integration gate.** Run catalog,
   canonical-config, guard, launcher, runtime, and production-CLI tests whenever
   any corresponding authority changes. Put this stage in `scripts/gate.sh` and
   enforce it in the repository's actual merge mechanism, even if GitHub Actions
   is not the chosen runner.
2. **Keep hardware qualification fail-closed.** Generation and evaluation are
   H100 contracts; current learner recipes are B200 contracts. Do not relabel a
   B200 recipe as H100-ready. Commission a distinct H100 learner recipe only
   after bounded memory, throughput, DDP/NCCL, and learning-parity evidence.
3. **Resolve the scratch optimizer authority.** Select and seal the update
   horizon, accumulation, precision, clipping, schedule, and resume semantics.
   Until then, scratch training must remain blocked even when data and hardware
   are available.
4. **Run a complete dry production turn with real retained artifacts.** Use the
   H100 for bounded generation/evaluation checks, keep the B200 data source
   read-only, stop before learner execution, and prove every receipt transition
   through the promotion dry-run boundary.

### P1 — productionization work

1. Extract trainer internals from `train_bc.py` into typed library modules;
   retain a compatibility parser only for authenticated historical replay.
2. Make `catan-zero` the stable individual-stage API and `tools/loop.py` the
   stable full-turn API; mark other launchers as internal, replay, or research.
3. Add a non-privileged runtime bootstrap that creates an exact isolated H100
   environment without installing services or modifying host-global state.
4. Create a queryable run/evidence index with checksum verification, retention
   status, parent checkpoint, dataset lineage, and promotion proof completeness.
5. Classify broad exception handlers and require structured failure receipts at
   every process and host boundary.

### P2 — hardening and maintenance

1. Complete a native unsafe/FFI review, document invariants, and add property or
   mutation tests around buffer shapes, action legality, and serialization.
2. Tag operational documents and tools with lifecycle status and owning
   authority; move unreferenced historical launchers behind a replay namespace.
3. Exercise a disaster-recovery drill from independently stored manifests and
   verify that registry, checkpoint, corpus, and receipts reconstruct exactly.
4. Add trend reporting for test duration, skipped/quarantined tests, native
   parity, data admission, target entropy, gradient health, and optimizer state.

## Current launch decision

- **Generation on the supplied H100:** software/runtime path is commissioned;
  a real wave still requires an authenticated checkpoint and exact job inputs.
- **Evaluation on the supplied H100:** software/runtime path is commissioned;
  strength claims still require the sealed paired gate.
- **Parent-update training:** scientifically commissioned only for its exact
  eight-B200 recipe; the supplied H100 must refuse it.
- **Scratch training:** blocked because the optimizer schedule is unresolved.
- **PPO:** blocked because the retained exact-initializer canary was harmful and
  no canonical PPO recipe exists.

That is the correct production posture: make safe paths easy to invoke, make
unsupported paths impossible to invoke accidentally, and do not confuse idle
GPU capacity with scientific authorization.
