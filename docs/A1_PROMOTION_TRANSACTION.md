# A1 promotion transaction

`tools/a1_promotion_transaction.py` is the only atomic/recoverable mutation
boundary for the current A1 generator-champion promotion. It does not run a
gate, alter `public_champion`, or deploy checkpoint bytes to the fleet.

## Preconditions

- The sealed `a1-pre-wave-contract-lock-v2` verifies with all 120 job claims and
  selects global `n_full=128` with no adaptive alternate budget. Global n64,
  n196, and adaptive/global n256 are refused for this wave.
- A typed `a1-promotion-adjudication-v2` has `passed=true`,
  `decision="promote"`, a reproducible `adjudication_sha256`, and binds the
  exact contract, candidate, incumbent, training report, and five evidence
  artifacts by SHA-256. Each referenced artifact is itself a sealed
  `a1-promotion-evidence-v2` envelope binding the contract, candidate, incumbent,
  pass verdict, and role-labelled source artifacts. A digest alone is not a
  verdict: the transaction parses and replays the source report semantics.
- Its six checks (`provenance`, `mechanism_calibration`, `internal_h2h`,
  `external_panel`, `high_regret`, `bucket_veto`) all pass. Every-third n64
  confirmation is derived from the authoritative registry promotion count and
  cannot be waived by the adjudication.
- The candidate training report reproduces the sealed A1 learner recipe and
  contract digests, records masking, names the exact candidate checkpoint,
  binds the sealed producer hash, and has positive optimizer steps/epochs.
- A successful direct `a1-one-dose-training-receipt-v3` or the single
  authorized graph-layer-repair `a1-one-dose-training-receipt-v4` is required
  separately from the adjudication. For v4, the orchestrator additionally
  replays the failed v3 parent, immutable retry contract and identity, and
  derived terminal claim through `adopt-retry`. The transaction verifies the
  receipt semantic digest, exact command/allowlisted child environment,
  executor-owned report binding, candidate/report/optimizer hashes, and
  terminal durable-claim agreement. A plausible standalone training report
  cannot authorize promotion.
- A candidate-specific `a1-promotion-cohort-exclusions-v1` binds every prior
  diagnostic/arm-selection source by SHA-256 and lists its half-open game-seed
  intervals. The final internal H2H and external neutral-panel evidence must
  retain exact per-game seeds, and the transaction proves their union is
  disjoint from every listed prior cohort. This prevents selecting an arm and
  then "confirming" it on the same random cohort (winner's curse). Exploratory
  diagnostic panels do not require this manifest; only the promotion boundary
  does.
- The registry is nonempty, its `generator_champion` path/version/MD5 matches
  the adjudicated incumbent, and `CURRENT_CHAMPION` contains that same single
  path.

The exclusion manifest is candidate- and contract-specific. Each source is an
immutable diagnostic/adjudication result; intervals are half-open:

```json
{
  "schema_version": "a1-promotion-cohort-exclusions-v1",
  "contract_sha256": "sha256:...",
  "candidate_sha256": "sha256:...",
  "cohorts": [{
    "label": "p1-arm-selection",
    "kind": "internal_h2h",
    "source": {"path": "/immutable/p1-selection.json", "sha256": "sha256:..."},
    "seed_intervals": [{"base_seed": 9000000, "end_seed": 9000200}]
  }],
  "manifest_sha256": "sha256:..."
}
```

## One-time A1 registry bootstrap

If no pre-A1 registry was ever persisted, create the A1 lineage only through
the audited bootstrap. Dry-run first, then repeat with `--go`:

```bash
python tools/a1_registry_bootstrap.py \
  --lock /immutable/a1.lock.json \
  --incumbent /immutable/champion.pt \
  --registry /private/champion_registry.json \
  --current-pointer /private/CURRENT_CHAMPION \
  --receipt /private/receipts/a1-registry-bootstrap.json

python tools/a1_registry_bootstrap.py \
  --lock /immutable/a1.lock.json \
  --incumbent /immutable/champion.pt \
  --registry /private/champion_registry.json \
  --current-pointer /private/CURRENT_CHAMPION \
  --receipt /private/receipts/a1-registry-bootstrap.json --go
```

The tool binds the exact contract producer and its typed historical scalar
attestation, initializes promotion count zero, and imports only the contract's
history/hard-negative checkpoints. It durably publishes a read-only prepared
journal first, deterministic registry/current-pointer bytes second, and the
committed receipt last. Repeating the identical `--go` command after a hard
interruption resumes missing exact publications; unknown or drifted partial
bytes fail closed.

The adjudication has this exact top-level shape (extra keys fail closed):

```json
{
  "schema_version": "a1-promotion-adjudication-v2",
  "passed": true,
  "decision": "promote",
  "contract_sha256": "sha256:...",
  "candidate": {
    "path": "/immutable/candidate.pt",
    "sha256": "sha256:...",
    "version": 5,
    "training_report": {"path": "/immutable/report.json", "sha256": "sha256:..."}
  },
  "champion": {"path": "/immutable/champion.pt", "sha256": "sha256:...", "version": 4},
  "checks": {
    "provenance": true,
    "mechanism_calibration": true,
    "internal_h2h": true,
    "external_panel": true,
    "high_regret": true,
    "bucket_veto": true
  },
  "nth_confirmation_required": false,
  "nth_confirmation": null,
  "evidence": [
    {"kind": "mechanism_calibration", "path": "/immutable/calibration.json", "sha256": "sha256:..."},
    {"kind": "internal_h2h", "path": "/immutable/h2h.json", "sha256": "sha256:..."},
    {"kind": "external_panel", "path": "/immutable/external.json", "sha256": "sha256:..."},
    {"kind": "high_regret", "path": "/immutable/regret.json", "sha256": "sha256:..."},
    {"kind": "bucket_veto", "path": "/immutable/buckets.json", "sha256": "sha256:..."}
  ],
  "adjudication_sha256": "sha256 of canonical JSON excluding this field"
}
```

Each evidence path above contains an envelope with this exact outer contract:

```json
{
  "schema_version": "a1-promotion-evidence-v2",
  "kind": "internal_h2h",
  "passed": true,
  "verdict": "H1",
  "contract_sha256": "sha256:...",
  "candidate": {"path": "/immutable/candidate.pt", "sha256": "sha256:..."},
  "champion": {"path": "/immutable/champion.pt", "sha256": "sha256:..."},
  "sources": [
    {"role": "internal_h2h", "path": "/immutable/h2h.raw.json", "sha256": "sha256:..."}
  ],
  "result": {
    "regression_protection_verdict": "H1",
    "superiority_verdict": "H1",
    "superiority_elo0": 0.0,
    "superiority_elo1": 15.0
  },
  "evidence_sha256": "sha256 of canonical JSON excluding this field"
}
```

The mechanism envelope references candidate and incumbent
`phase-sliced-value-calibration-v2` reports and replays held-out/readout/RMSE
semantics. Both reports must bind the identical shard directory, validation
manifest SHA-256, selection mode/fraction/seed/ranges, observed seed count, and
observed row count; comparing metrics from different cohorts is refused.
Internal H2H retains and replays all paired games and both pentanomial GSPRTs,
requiring global n128, at least 200 complete pairs, regression-protection
`[-10,+15]` H1, and positive-Elo superiority `[0,+15]` H1. A regression-band
H1 with superiority `continue` or H0 is a hard refusal. Evidence-v1 remains
identifiable only in immutable, already-recorded registry/receipt history; it
cannot authorize a new transaction.
External evidence references candidate and incumbent neutral-harness reports,
replays every raw outcome, and applies the fixed `0.02` maximum win-rate
regression. Individual absolute SPRT verdicts are diagnostic only (both honest
panels may be H0); changing their Elo thresholds cannot change eligibility. The
two panels must have identical opponent, map/search config, requested pair
counts, and exact `(pair_id, game_seed, orientation)` cohort.
Calibration likewise uses an exact fixed `0.02` maximum global-RMSE
regression; envelopes cannot select either tolerance. High-regret evidence must use
`a1-high-regret-comparison-v1` and prove a passing held-out paired result.
Bucket evidence must use `a1-bucket-veto-v1`; every included bucket must be a
real pass with at least eight games. `insufficient_data` is a promotion refusal,
not a silent non-veto.

## Build the evidence graph

Do not hand-author any of the JSON above. The canonical producer is
`tools/a1_promotion_artifacts.py`. It creates every output with `O_EXCL`, makes
it read-only, hashes each source, and replays the transaction validator before
publishing an evidence envelope or adjudication.

Derive the high-regret comparison from the held-out evaluator report and the
bucket veto from raw, bucket-labelled game records. Aggregate counts are not
accepted because they cannot prove cohort identity or be independently replayed:

```bash
# One-time frozen suite: use the full trainer-authenticated validation manifest,
# then select 240 regret-ranked states from 240 distinct source games with fixed
# opening/robber/chance/build/41+ stratification. No second per-state thinning.
python tools/a1_promotion_artifacts.py held-out-suite \
  --manifest /immutable/raw_validation_regret.npz \
  --holdout-fraction 1.0 --holdout-seed 17 --pairs 240 \
  --out /immutable/a1-high-regret.suite.json

# Real candidate-vs-champion continuations from every frozen archived state.
python tools/gumbel_search_cross_net_h2h.py \
  --candidate /immutable/candidate.pt --baseline /immutable/champion.pt \
  --held-out-high-regret-suite /immutable/a1-high-regret.suite.json \
  --workers 8 --devices cuda:0,cuda:1,cuda:2,cuda:3 \
  --n-full 128 --max-decisions 600 --gate-config flywheel \
  --public-observation --information-set-search \
  --determinization-particles 4 --determinization-min-simulations 32 \
  --c-scale 0.03 --c-visit 50 --sigma-eval 0.98 \
  --lazy-interior-chance --symmetry-averaged-eval \
  --symmetry-averaged-eval-threshold 20 \
  --out /immutable/high-regret.report.json

python tools/a1_promotion_artifacts.py high-regret \
  --report /immutable/high-regret.report.json \
  --candidate /immutable/candidate.pt \
  --champion /immutable/champion.pt \
  --out /immutable/high-regret.source.json

python tools/a1_promotion_artifacts.py bucket-report \
  --report /immutable/high-regret.report.json \
  --candidate /immutable/candidate.pt \
  --champion /immutable/champion.pt \
  --out /immutable/bucket-games.report.json

python tools/a1_promotion_artifacts.py bucket-veto \
  --report /immutable/bucket-games.report.json \
  --candidate /immutable/candidate.pt \
  --champion /immutable/champion.pt \
  --out /immutable/bucket-veto.source.json
```

The high-regret input is `a1-held-out-high-regret-report-v1` and must bind the
exact checkpoint bytes, `suite=held_out_high_regret`, `held_out=true`, a
no errors, immutable held-out-suite provenance, and raw paired games whose
pentanomial statistics replay to `H1`. The bucket input is
`a1-bucket-game-report-v1`; the builder computes status, sample size, win rate,
and veto directly from unique bucket-labelled games rather than accepting
caller-authored result fields.

Wrap each verified source and then build the final adjudication. `--source` and
`--evidence` are repeatable `ROLE=PATH` / `KIND=PATH` arguments:

```bash
python tools/a1_promotion_artifacts.py evidence \
  --kind high_regret \
  --contract-lock /immutable/a1.lock.json \
  --registry /private/champion_registry.json \
  --candidate /immutable/candidate.pt \
  --champion /immutable/champion.pt \
  --source high_regret=/immutable/high-regret.source.json \
  --out /immutable/high-regret.evidence.json

python tools/a1_promotion_artifacts.py adjudicate \
  --contract-lock /immutable/a1.lock.json \
  --training-receipt /immutable/training.receipt.json \
  --registry /private/champion_registry.json \
  --current-pointer /private/CURRENT_CHAMPION \
  --candidate /immutable/candidate.pt --candidate-version 5 \
  --training-report /immutable/report.json \
  --champion /immutable/champion.pt --champion-version 4 \
  --evidence mechanism_calibration=/immutable/calibration.evidence.json \
  --evidence internal_h2h=/immutable/internal.evidence.json \
  --evidence external_panel=/immutable/external.evidence.json \
  --evidence high_regret=/immutable/high-regret.evidence.json \
  --evidence bucket_veto=/immutable/bucket-veto.evidence.json \
  --out /immutable/a1.promotion.json
```

For every third generator promotion, the registry-derived policy additionally
requires `--nth-confirmation /immutable/n64-h2h.raw.json`; non-third promotions
require this field to remain null. The transaction hashes the artifact, binds
the exact candidate and incumbent checkpoints, replays the global-n64 operator,
verifies the paired seed/orientation cohort, and requires the fixed
`-10/+15` pentanomial SPRT to replay to H1. The builder is not a second
authorization boundary.

## Dry-run, commit, and recovery

Dry-run is the default and writes nothing:

```bash
python tools/a1_promotion_transaction.py promote \
  --registry /private/champion_registry.json \
  --current-pointer /private/CURRENT_CHAMPION \
  --contract-lock /immutable/a1.lock.json \
  --adjudication /immutable/a1.promotion.json \
  --training-receipt /immutable/a1.one-dose.receipt.json \
  --cohort-exclusions /immutable/a1.cohort-exclusions.json \
  --receipt /private/receipts/a1-p5.json \
  --reason "A1 typed promotion"
```

Repeat with `--go` only after reviewing the printed plan. One exclusive lock
covers the full transaction. The incumbent is appended to the opponent pool,
only `generator_champion` is changed, the promotion counter increments, and
the plain-text current pointer changes to the candidate path. Receipt
provenance explicitly records `fleet_ckpt_updated=false`; remote fleet paths
remain a separate hash-verified deployment action.

New transactions use `a1-promotion-transaction-receipt-v3` and bind the
verified v3 direct-dose or v4 derived-retry receipt plus the exact exclusion
manifest, its source artifacts, the excluded intervals, the final promotion
intervals, and a zero-overlap result. Recovery remains compatible with
already-prepared v1 and v2 promotion receipts.

The lock is always derived from the canonical registry path as
`<registry>.a1.lock`. `--lock-file` remains accepted for command compatibility
only when it names that exact canonical path; alternate locks are refused.
Registry, current-pointer, receipt, backup, and lock paths may not traverse
symlinks.

POSIX has no atomic two-path replace. Before either mutation, `--go` durably
writes the receipt with status `prepared` plus exact registry/pointer backups.
Each destination is atomically replaced under the same lock. An ordinary error
restores both before bytes and records `rolled_back`; a hard interruption is
recovered from the receipt:

```bash
python tools/a1_promotion_transaction.py recover --receipt /private/receipts/a1-p5.json
python tools/a1_promotion_transaction.py recover --receipt /private/receipts/a1-p5.json --go
```

Recovery accepts only known before/after bytes. Unknown mutations, backup hash
drift, receipt semantic-digest drift, noncanonical paths, an already-held lock,
or a nonrecoverable receipt status are refused. If the second recovery write or
either post-write verification fails, the transaction restores the exact
pre-recovery registry and pointer bytes before returning an error.
