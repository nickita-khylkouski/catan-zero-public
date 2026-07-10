# Pre-wave RL boundary differential review — 2026-07-09

## Executive summary

| Severity | Found | Fixed |
|---|---:|---:|
| Critical | 2 | 2 |
| High | 8 | 8 |
| Medium | 3 | 3 |

**Overall risk before fixes:** critical.  **Recommendation after fixes:**
approve the non-executing pre-wave boundary.  The review found no remaining
known bypass in the audited scope and the repository-wide test gate is green.

Files deeply reviewed: `tools/a1_pre_wave_contract.py`,
`tools/search_teacher_adjudicator.py`,
`tools/legacy_scalar_readout_attestation.py`, their tests and the A1 template.
The selected-game ingest bridge in `tools/build_memmap_corpus.py` and trainer
validation-manifest bridge were integration-tested as downstream consumers.

## Findings and fixes

### Critical — arbitrary typed decision JSON could authorize a wave

The contract previously checked hashes and selected fields but did not replay
the producer that made an A0/S1/S2/S3 decision.  An operator-authored JSON file
could therefore claim `passed=true` and point at arbitrary hashed input JSON.

The contract now reruns `a0_binding_verdict.build_binding_verdict` and
`search_teacher_adjudicator.adjudicate` in-process and requires exact semantic
equality (`tools/a1_pre_wave_contract.py:865-1052`).  It also requires S2/S3 to
bind the exact preceding decision bytes and requires all search stages to name
the exact production teacher checkpoint (`:1402-1416`).  Regression tests
cover a rehashed fabricated envelope and a swapped typed predecessor.

### Critical — a healthy wave could not satisfy the data contract

The old design requested exactly 12,000 attempts while rejecting every
truncation.  Prior production telemetry already showed rare healthy
max-decision truncations, so wave acceptance was probabilistic and likely to
fail after consuming the fleet allocation.

The v2 contract requests a bounded 408/77/26 attempts per worker and selects
the lowest-seed complete 400/75/25 per job (`tools/a1_pre_wave_contract.py:72,
:624-659`).  It still accepts exactly 9,600/1,800/600 unique non-VAL games.
Every raw shard is hashed, but reserve/truncated rows are excluded before all
metrics and holdout construction.  The test corpus includes 72 actual reserve
truncations and still proves exactly 12,000 accepted games.

### High — accepted-game selection was not enforceable at ingest

Counts in an audit report did not stop a later memmap build from reading every
row in the same directories, including reserve or truncated attempts.

The audit now emits immutable `a1-selected-training-games-v1` and
`train-validation-game-seeds-v1` sidecars.  The former contains all 12,000
seeds, source identity, train/validation split and canonical digests
(`tools/a1_pre_wave_contract.py:2470`).  The memmap builder requires the
passing v2 audit and sidecars, verifies the exact input shard inventory and
bytes, filters to the selected set, proves set equality, and persists the
contract/audit/selection provenance.

### High — audited ingest still allowed a second row-dropping filter

Even after exact game selection, `build_memmap_corpus(...,
full_rows_only=True)` could physically discard every fast-search row.  The seed
set still matched, so training exposure could differ from the audit without a
missing game.

Audited A1 ingest now rejects `full_rows_only` before reading or writing any
output (`tools/build_memmap_corpus.py:753`) and binds exact selected/training row
counts through audit, corpus metadata and trainer replay.  A focused regression
proves the second filter is unavailable on the immutable path.

### High — H2H and fixed-root configuration drift was under-validated

An H2H result could use a different decision cap, evaluator readout, search
setting or role budget while still donating its outcome to the production
operator.  Fixed-root aggregate cost/stability values were accepted without
reconstructing them from raw runs.

The adjudicator now validates the complete typed EvalConfig, raw color-swapped
pair records and typed seed plan (`tools/search_teacher_adjudicator.py:547-680`).
It checks every selected search/evaluator field in fixed-root roles, distinct
seed manifests, raw run seeds and recomputes stability slices from `per_root`
(`:700-887`).  Tests cover rehashed decision-cap drift, raw-outcome tampering
and rehashed aggregate tampering.

### High — S1 could differ outside the predeclared c-scale/D1 dose

The S1 consumer checked a subset of the baseline and candidate search configs.
It now requires the exact stock-search baseline, exact candidate override and
all 170 raw paired games, with consecutive disjoint seeds and reconstructed
pentanomial counts.

### High — shard basename fallback allowed path aliasing

An unavailable absolute path silently fell back to a same-basename file beside
the manifest.  The resolver now has exactly one interpretation: absolute means
that exact path; relative means relative to the owning manifest
(`tools/a1_pre_wave_contract.py:1899`).  Canonically duplicated shard paths are
rejected.

### High — one selected seed could contribute multiple raw game runs

The audit coalesced termination/truncation state by seed.  A seed could appear,
end, and then reappear in a later shard; the sidecar still contained one game
record while ingest retained both raw runs, silently overweighting duplicated
game rows.

The audit now carries run state across the ordered shard sequence, permits one
game to span an adjacent shard boundary, and rejects any seed that starts a
second non-contiguous run (`tools/a1_pre_wave_contract.py:2119-2151,
:2403-2448`).  The same invariant is independently replayed against the actual
memmap seed column by the trainer.  Regression coverage proves both the legal
adjacent split and illegal later reappearance.

### High — the wave contract did not bind the learner dose

The search teacher, selected corpus and learner value objective were sealed,
but optimizer, batching, loss mixture, AMP, masking and sample weighting could
still drift at training time.  That made a nominal A1 result scientifically
non-reproducible despite a valid data lock.

The v2 science envelope now contains a strict, type-checked effective
`learner_training_recipe` plus canonical digest
(`tools/a1_pre_wave_contract.py:82-139, :875-904, :1590-1608`).  It pins the
single-B200 topology and global batch, fresh Adam/LR schedule, all active and
disabled loss weights, public masking, the gen3 graph-history regime, and
track/VP semantics.  `tools/train_bc.py:1675-1709, :1777-1865` reconstructs
those values from the resolved CLI and DDP context, checks the warm-start
producer hash and rejects missing, extra or drifted fields before the first
optimizer step.

### High — mandatory reports could be synthesized from absent columns

`is_forced`, `used_full_search`, `phase`, `decision_index`, and target-policy
arrays used permissive defaults.  A shard missing the real telemetry could
therefore yield zeros, `<missing>`, or `None` while the nominally mandatory
reports still existed.

Every shard contributing selected rows must now carry row-aligned forced/full,
phase, decision-index, target-policy and target-policy-mask arrays
(`tools/a1_pre_wave_contract.py:2049-2116`).  Empty phases, out-of-range
decisions, empty masks, non-finite/negative probabilities and non-positive
policy mass all fail closed.  Parameterized regressions remove every required
column in turn and cover empty target evidence.

### Medium — a rehashed lock did not reconstruct its source draft

Lock verification now rebuilds the complete lock from the immutable draft and
requires canonical equality (`tools/a1_pre_wave_contract.py:1613`).

### Medium — legacy scalar evidence accepted contradictory/incomplete telemetry

The bridge now requires the real entity-graph checkpoint/report shape,
completed sequential epochs with train and validation value telemetry, and
rejects a non-positive resolved scalar objective
(`tools/legacy_scalar_readout_attestation.py:184-267`).  The exact checkpoint,
report and attestation bytes remain rehashed on every lock verification.

### Medium — symlinked contract paths disagreed across audit and training

The auditor reported `Path.absolute()` while the downstream trainer required a
canonical resolved path.  A lock reached through a symlink could therefore
pass audit and then fail the immutable learner replay.  Audit now resolves the
lock strictly at entry and records that canonical identity
(`tools/a1_pre_wave_contract.py:2250-2256, :2721-2724`); a symlink regression
proves the handoff uses the real path.

## Follow-up adversarial readiness audit

The final independent replay found and closed an additional wave-blocking
ledger lifecycle bug plus four ingest/runtime gaps:

- the shared seed ledger is now an immutable pre-claim prefix with append-only
  live semantics; `render` emits the exact range/claim/contract/job row, and
  post-wave verification requires exactly one such row for all 72 jobs while
  rejecting peers, spoofs and duplicates;
- A1 attestations are detected at the supplied source or any ancestor, so a
  nested `worker_000/` directory cannot enter the generic converter path;
- the converter hashes the exact schema-implied `.dat`, `.codes.dat`, and
  offset inventory, while the trainer verifies every filename, size and SHA-256
  before optimizer construction and persists the aggregate digest;
- the seal binds the explicit 17-file learner implementation and a 208-file
  transitive runtime tree covering all Python modules under `src/catan_zero`
  and `tools` plus both guard configs; training re-hashes the tree and persists
  its digest in job attestations, reports and checkpoints;
- relative checked-in provenance paths are canonicalized with
  `resolve(strict=False)`, eliminating self-rejection from preserved `..`
  components.

The final adversarial re-audit reports **no remaining Critical or Important
findings** in the scoped pre-wave boundary.

## Test evidence

- Latest focused adversarial boundary replay: **88 passed**; the root combined
  contract/ingest/trainer slice is **72 passed**.
- Ruff on the audited/new files: **passed**.
- `py_compile` and `git diff --check`: **passed**.
- Final full repository suite: **1,982 passed, 155 skipped, 0 failed** (two
  inherited warnings).  The intermediate CLI golden drift for the new
  `train_bc --validation-game-seed-manifest` was corrected and re-gated.

## Methodology and limits

Strategy: deep, high-risk differential review.  The reviewed files are new and
uncommitted relative to `origin/main`, so there is no earlier implementation
or meaningful blame history for their lines.  Baseline invariants came from
the local master/R&D plans and the actual producer/consumer code paths.  The
review traced seal → render → output audit → selected-game sidecar → memmap →
trainer holdout, modeled forged/stale/drifted artifacts, and added executable
regressions for each concrete bypass.

This boundary cannot cryptographically prove that a trusted local operator
actually performed GPU computation; it can prove that all accepted artifacts
are internally reproducible, byte-bound, config-identical and derived from the
raw records under the declared adjudication rules.  The 24-GPU production wave
was not launched by this work.
