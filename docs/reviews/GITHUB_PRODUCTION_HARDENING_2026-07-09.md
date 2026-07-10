# GitHub production hardening review — 2026-07-09

## Executive summary

This review compared the current GitHub `main` branch with the retained local
H100 optimization work and audited the generation, fleet-control, typed-config,
EvalServer, one-dose training, and promotion transaction boundaries. The review
found several fail-open production contracts. The accompanying change set fixes
the issues that can be closed safely without another GPU experiment and adds
regressions for the reproduced failure modes.

The retained performance result remains the measured n128 frontier of about
91.85k stored decision rows/hour/H100 on the synthetic checkpoint. The optional
two-pipeline topology is included because its retained canary exceeded the 20%
implementation threshold, but it remains opt-in pending a longer real-champion
certification. No new H100 run was performed during this GitHub hardening pass.

Two architectural limitations remain explicit release constraints:

1. seed allocation is not atomic across independent hosts; one operator must
   reconcile the per-host ledgers before concurrent fleet launches; and
2. public neural inputs and stored rows do not make MCTS an information-set
   search, because the authoritative Rust game still contains true hidden state.

## What changed

- Fleet launch now fails closed on busy requested GPUs, hidden MPS clients,
  ledger-write failure, missing detach support, failed child startup, and early
  process death. Published launch PIDs are verified as live session/group
  leaders, failed launches reap their exact owned process group, and the
  generation supervisor stops every sibling pipeline on the first child error.
- Fleet stop no longer treats every visible CUDA process as Catan work. Legacy
  fallback targets require a canonical Catan command token in the process or
  ancestor chain; unrelated CUDA jobs are reported and preserved.
- Unsupported fleet combinations are rejected before SSH or seed claim:
  opponent-mix generation with the mandatory shared EvalServer, non-certified
  `c_scale`, and generation-only topology flags on training.
- The opt-in `--pipelines-per-gpu 2` path partitions the existing per-GPU games,
  workers, and contiguous seed block across two independent pipelines. It does
  not double the claim. Per-pipeline logs, PIDs, manifests, config dumps, and
  status counts are recorded; the default one-pipeline layout is unchanged.
- Generation applies typed config before derived values and guards. Executable
  config files must match the expected pipeline and current schema and must
  satisfy argparse type/choice contracts. Explicit CLI values win even when
  equal to parser defaults.
- `p_full` is finite and bounded to `[0, 1]`; other science floats reject
  non-finite values. Partial or missing games now return nonzero while retaining
  the top-level manifest and partial artifacts. EvalServer/bootstrap exceptions
  also produce a diagnostic top-level manifest.
- Checkpoint provenance binds the bytes actually loaded: generation copies the
  source once into a run-owned, fsynced, read-only checkpoint, hashes that same
  stream, and gives only the staged path to the server/workers. The source path,
  staged path, and SHA-256 remain auditable.
- The generation config hash now binds checkpoint SHA-256 and the number of
  independent pipelines sharing a GPU. Per-pipeline identity remains manifest
  provenance so sibling data keeps a common science hash.
- The hard physical neural-row cap is rejected with CUDA Graph buckets, whose
  padded execution could exceed the cap. Rejected BF16/FP16 inference autocast
  plumbing was removed after its measured throughput regression.
- Typed-config schema consumers use the shared version. The search-teacher
  adjudicator also replays sealed schema-4 eval evidence while accepting the
  current schema, avoiding accidental invalidation of historical artifacts.
- The A1 one-dose and promotion transactions were hardened so contract identity,
  candidate bytes, evidence semantics, locks, receipts, and recovery behavior
  are fail-closed rather than caller-path or self-assertion based.
  Candidate/incumbent metric comparisons also require identical calibration or
  paired-game cohorts and evaluation configuration.
- Fleet observability now follows the canonical launcher tree/output paths,
  exports both dual-pipeline slots, treats failed/fatal manifests as unhealthy,
  alerts on an exited-incomplete generator, prunes stale generated tunnel units,
  and binds Grafana to loopback by default.

## Findings and disposition

### Resolved in this change set

| Severity | Finding | Disposition |
|---|---|---|
| Critical | A1 promotion checked evidence file hashes but did not prove those artifacts evaluated the promoted candidate | Candidate/incumbent/contract bindings and typed semantic evidence validation are required |
| High | Typed config could apply a wrong-pipeline, stale, or invalid JSON value and then synthesize it as guard-approved CLI input | Pipeline/schema/type/choice validation runs before application and guard synthesis |
| High | Fleet stop legacy fallback could kill unrelated CUDA jobs | Exact Catan ancestry admission replaces broad GPU-PID targeting |
| High | Fleet launch could report success for a dead child or let siblings continue after one pipeline failed | PID/SID/PGID checks and concurrent child supervision fail nonzero and reap all owned descendants |
| High | One-dose uniqueness was keyed by a caller-selected receipt path | The sealed contract identity is the durable dose key |
| High | Producer checkpoint SHA-256 had a path reopen TOCTOU window | Workers load only the run-owned bytes hashed during staging |
| Medium | Partial generation and EvalServer startup failures could leave no top manifest or return success | Incomplete runs retain diagnostics and exit nonzero |
| Medium | Generator config was applied after derived defaults and guards | Config resolution precedes derivation, validation, and guards |
| Medium | CUDA Graph bucket padding could violate a declared hard neural-row cap | The incompatible combination is rejected |
| Medium | Two shared-GPU pipelines were absent from GitHub and not represented in config identity | Opt-in exact partitioning and topology hashing added |
| Medium | Global schema bump broke replay of sealed schema-4 search evidence | Current and sealed historical eval schemas are explicitly supported |
| Medium | Exporter paths/alerts missed canonical and dual-pipeline generation | Runtime paths, per-pipeline labels, fail-closed health, and death alerts aligned |

### Remaining constraints

| Severity | Constraint | Required follow-up |
|---|---|---|
| High | Per-host seed ledgers have no shared cross-host lock | Use one operator and reconcile all ledgers before concurrent launches, or add a central transactional claim service |
| High | MCTS targets can depend on authoritative opponent hidden state even though model inputs and stored rows are public-masked | Add hidden-hand permutation invariance tests and belief-state/determinization search before calling the teacher information-set correct |
| Medium | The checked-in n128 canary cites artifacts that are not committed or independently rehashable | Keep `train_admissible=false`; publish a complete signed/hashed evidence bundle before using it as a training admission record |
| Medium | Dual pipelines have short synthetic-checkpoint evidence, not a long real-champion fleet certification | Run the predeclared real-checkpoint target/parity and sustained-throughput gate before making it the default |
| Medium | The checked-in A1 draft still has unresolved S1/S2/S3 bindings and seed-ledger fields | Materialize operator bindings, resolve the draft, then seal/render/postflight; the safe mechanism alone is not launch readiness |
| Medium | Promotion has no production emitters yet for the typed high-regret and bucket-veto reports | Implement those report producers; hardened promotion correctly refuses generic or incomplete substitutes |
| Low | Public-masking integration tests skip when the native Rust extension is unavailable | Add a CI lane with the production native wheel/extension |

## Test coverage

Focused tests cover typed-config ordering/validation, checkpoint staging,
partial-run manifests, dual-pipeline seed partitioning and default compatibility,
busy-GPU/MPS preflight, detached lifecycle, unrelated CUDA preservation,
EvalServer cap incompatibility, historical search-evidence replay, one-dose
identity/recovery, and promotion evidence/transaction semantics.

Final verification against `origin/main` at `d55c23d`:

- full local suite: **2,099 passed, 245 skipped**;
- native arm64 Linux fleet lifecycle/topology suite: **48 passed**;
- Ruff over every changed Python file: passed;
- Python compileall over `src`, `tools`, `ops/observability`, and `tests`: passed;
- Bash syntax for all changed fleet scripts: passed;
- `git diff --check`: passed.

The skipped tests are dependency/platform-gated lanes already represented in
the suite; notably, native Linux process-group coverage was run separately.

## Blast radius

- Generation manifests and typed config payloads move to schema version 5.
- Existing schema-4 search adjudication evidence remains replayable; executable
  schema-4 config files are intentionally rejected so stale defaults cannot
  silently launch new work.
- The default fleet topology and output directories are unchanged. Dual mode is
  explicit and generation-only.
- `EvalServerConfig(experimental_autocast_dtype=...)` and the corresponding
  benchmark flag are removed. They were benchmark-only, not fleet-wired.
- Promotion is intentionally stricter: untyped or semantically incomplete
  evidence that previously passed by hash alone now fails closed.

## Historical context

The baseline already contained the large Rust/native-feature, shared EvalServer,
event-tail cropping, root-wave, ragged-corpus, H100 documentation, and public
row-masking work. This review did not re-land dirty local experiments wholesale.
It selected the measured dual-pipeline topology and production-contract fixes,
removed the measured-loss autocast experiment, and preserved unrelated user
work in the original worktree.

## Recommendations

1. Treat central seed allocation and information-set-correct search as the two
   highest-priority blockers before unattended 24-GPU operation.
2. Keep one pipeline/GPU as the default until dual mode passes the real champion
   sustained run and target-quality gate.
3. Run promotion only through the hardened transaction and only with the typed
   evidence envelopes; never substitute a generic `passed: true` JSON file.
4. Add Linux + native-Rust CI because macOS-only unit coverage cannot exercise
   detached process groups or the real feature/masking boundary.

## Methodology

The review used a clean worktree based on fetched GitHub `origin/main`, compared
that tree with retained local optimization work, traced the launch/generation/
training/promotion data flow, reproduced fail-open behaviors with local fakes,
and used independent reviewers for fleet safety, generator/config contracts,
EvalServer/provenance, and the new A1 transaction commits. No production GPU,
remote launcher, seed ledger, champion registry, or external service was
mutated during the audit.

## Appendix: reviewed boundaries

- `tools/fleet/fleet_launch.sh`, `launch_detached.sh`, `fleet_status.sh`,
  `fleet_stop.sh`
- `tools/generate_gumbel_selfplay_data.py`
- `src/catan_zero/rl/config_cli.py`, `pipeline_configs.py`
- `src/catan_zero/search/eval_server.py`
- `tools/a1_one_dose_train.py`, `a1_promotion_transaction.py`
- `tools/search_operator_binding.py`, A1 40-H100 contract arithmetic, and
  `ops/observability`
- related focused tests and RL/H100 operator documentation
