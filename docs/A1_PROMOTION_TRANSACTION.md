# A1 promotion transaction

`tools/a1_promotion_transaction.py` is the only atomic/recoverable mutation
boundary for the current A1 generator-champion promotion. It does not run a
gate, alter `public_champion`, or deploy checkpoint bytes to the fleet.

## Preconditions

- The sealed `a1-pre-wave-contract-lock-v2` verifies with all 120 job claims and
  selects global `n_full=128` with no adaptive alternate budget. Global n64,
  n196, and adaptive/global n256 are refused for this wave.
- A typed `a1-promotion-adjudication-v1` has `passed=true`,
  `decision="promote"`, a reproducible `adjudication_sha256`, and binds the
  exact contract, candidate, incumbent, training report, and five evidence
  artifacts by SHA-256.
- Its six checks (`provenance`, `mechanism_calibration`, `internal_h2h`,
  `external_panel`, `high_regret`, `bucket_veto`) all pass. Every-third n64
  confirmation is derived from the authoritative registry promotion count and
  cannot be waived by the adjudication.
- The candidate training report reproduces the sealed A1 learner recipe and
  contract digests, records masking, and has positive optimizer steps/epochs.
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
drift, an already-held lock, or a nonrecoverable receipt status are refused.
