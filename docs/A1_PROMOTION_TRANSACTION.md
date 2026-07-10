# A1 promotion transaction

`tools/a1_promotion_transaction.py` is the only atomic/recoverable mutation
boundary for the current A1 generator-champion promotion. It does not run a
gate, alter `public_champion`, or deploy checkpoint bytes to the fleet.

## Preconditions

- The sealed `a1-pre-wave-contract-lock-v2` verifies with all 72 job claims and
  selects global `n_full=128` with no adaptive alternate budget. Global n64,
  n196, and adaptive/global n256 are refused for this wave.
- A typed `a1-promotion-adjudication-v1` has `passed=true`,
  `decision="promote"`, a reproducible `adjudication_sha256`, and binds the
  exact contract, candidate, incumbent, training report, and five evidence
  artifacts by SHA-256. Each referenced artifact is itself a sealed
  `a1-promotion-evidence-v1` envelope binding the contract, candidate, incumbent,
  pass verdict, and role-labelled source artifacts. A digest alone is not a
  verdict: the transaction parses and replays the source report semantics.
- Its six checks (`provenance`, `mechanism_calibration`, `internal_h2h`,
  `external_panel`, `high_regret`, `bucket_veto`) all pass. Every-third n64
  confirmation is derived from the authoritative registry promotion count and
  cannot be waived by the adjudication.
- The candidate training report reproduces the sealed A1 learner recipe and
  contract digests, records masking, names the exact candidate checkpoint,
  binds the sealed producer hash, and has positive optimizer steps/epochs.
- The registry is nonempty, its `generator_champion` path/version/MD5 matches
  the adjudicated incumbent, and `CURRENT_CHAMPION` contains that same single
  path.

The adjudication has this exact top-level shape (extra keys fail closed):

```json
{
  "schema_version": "a1-promotion-adjudication-v1",
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
  "nth_confirmation_passed": false,
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
  "schema_version": "a1-promotion-evidence-v1",
  "kind": "internal_h2h",
  "passed": true,
  "verdict": "H1",
  "contract_sha256": "sha256:...",
  "candidate": {"path": "/immutable/candidate.pt", "sha256": "sha256:..."},
  "champion": {"path": "/immutable/champion.pt", "sha256": "sha256:..."},
  "sources": [
    {"role": "internal_h2h", "path": "/immutable/h2h.raw.json", "sha256": "sha256:..."}
  ],
  "result": {},
  "evidence_sha256": "sha256 of canonical JSON excluding this field"
}
```

The mechanism envelope references candidate and incumbent
`phase-sliced-value-calibration-v2` reports and replays held-out/readout/RMSE
semantics. Both reports must bind the identical shard directory, validation
manifest SHA-256, selection mode/fraction/seed/ranges, observed seed count, and
observed row count; comparing metrics from different cohorts is refused.
Internal H2H retains and replays all paired games and the flywheel
pentanomial GSPRT, requiring global n128, at least 200 complete pairs, and H1.
External evidence references candidate and incumbent neutral-harness reports,
rejects H0/errors/divergence, and applies the fixed `0.02` maximum win-rate
regression. The two panels must have identical opponent, map/search/gate config,
requested pair counts, and exact `(pair_id, game_seed, orientation)` cohort.
Calibration likewise uses an exact fixed `0.02` maximum global-RMSE
regression; envelopes cannot select either tolerance. High-regret evidence must use
`a1-high-regret-comparison-v1` and prove a passing held-out paired result.
Bucket evidence must use `a1-bucket-veto-v1`; every included bucket must be a
real pass with at least eight games. `insufficient_data` is a promotion refusal,
not a silent non-veto.

## Dry-run, commit, and recovery

Dry-run is the default and writes nothing:

```bash
python tools/a1_promotion_transaction.py promote \
  --registry /private/champion_registry.json \
  --current-pointer /private/CURRENT_CHAMPION \
  --contract-lock /immutable/a1.lock.json \
  --adjudication /immutable/a1.promotion.json \
  --receipt /private/receipts/a1-p5.json \
  --reason "A1 typed promotion"
```

Repeat with `--go` only after reviewing the printed plan. One exclusive lock
covers the full transaction. The incumbent is appended to the opponent pool,
only `generator_champion` is changed, the promotion counter increments, and
the plain-text current pointer changes to the candidate path. Receipt
provenance explicitly records `fleet_ckpt_updated=false`; remote fleet paths
remain a separate hash-verified deployment action.

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
