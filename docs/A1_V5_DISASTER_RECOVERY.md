# A1 v5 disaster-recovery boundary

This path exists for one evidence-loss incident. It does not reconstruct the
missing v5 promotion, and it must not be reused as a generic way to bypass the
normal promotion transaction.

## What survived

- The exact surviving v5 producer handoff and its deployed search identity.
- The exact recovered v5 generator checkpoint named by that handoff.
- The exact f7 checkpoint that remains the public/tournament safety reference.

The original promotion receipt, registry, and current-pointer bytes did not
survive. Therefore the recovery authority says, permanently:

- `promotion_proof_recreated = false`
- recovered v5 is `generator_champion` only
- f7 remains the public/tournament/opponent-pool safety reference
- f7's relationship to v5 is
  `safety_reference_unproven_predecessor`, with
  `causal_parent_proven = false`
- no verified promotion count is inferred

The recovery tool has an exact fingerprint allowlist. A different handoff,
checkpoint, f7 file, runtime, or source tree is rejected.

## Recovery transaction

Dry-run first; it performs no writes:

```bash
python tools/a1_v5_disaster_recovery.py \
  --surviving-handoff /absolute/path/to/surviving-v5-handoff.json \
  --safety-reference /absolute/path/to/f7.pt \
  --namespace /absolute/path/to/a1-v5-disaster-recovery
```

Commit uses the same arguments plus `--go`. The commit holds a sibling
non-blocking `flock` across journal, registry, pointer, and receipt publication.
It uses fresh files, `O_NOFOLLOW`, mode `0444`/`0600` as appropriate, fsync, and
exact crash resume. A second writer or a non-identical resume is refused.

The ordinary post-promotion handoff builder deliberately rejects this receipt.
The only canonical reader is
`tools.a1_v5_disaster_recovery.verify_committed_receipt`.

## Search-operator continuity

The recovered deployed producer used `c_scale = 0.1`, while the surviving S1
artifact predates that value. Create the one recovery-only bridge with:

```bash
python tools/search_operator_binding.py \
  --legacy-s1-decision /absolute/path/to/legacy-s1.decision.json \
  --recovery-receipt /absolute/path/to/a1-v5-disaster-recovery.receipt.json \
  --s1-out /fresh/path/s1.recovery.binding.json \
  --s2-out /fresh/path/s2.binding.json \
  --s3-out /fresh/path/s3.binding.json
```

The bridge replays both sources, changes only the authenticated continuity
field, and seals `promotion_proof_recreated = false`.

## Wave lineage

The generation scheduler keeps the stable lane ID `recent_history`; storage and
quota code depend on that ID. In a recovery wave its sealed meaning is instead:

```text
semantic = recovery_reference
relation = safety_reference_unproven_predecessor
causal_parent_proven = false
promotion_proof_recreated = false
checkpoint = exact authenticated f7
```

That semantic record is copied exactly into new job and opponent-mix
attestations, selected-game records, the post-wave audit, composite source
bindings/authority, and the learner descriptor. Removing it or changing it to
`immediate_displaced_incumbent` is a hard failure. Ordinary clean-lineage waves
retain their existing `recent_history` / `immediate_displaced_incumbent`
meaning. Previously issued locks that predate this field remain verifiable.

## Candidate gate after a recovery wave

No recovered-wave candidate is auto-promoted. The gate is conjunctive:

1. the normal full training/calibration/external/high-regret/bucket gate and a
   strict H1 result against the recovered v5 generator parent;
2. a separate, fresh, fixed 300-pair f7 panel at base seed `6_199_100_000`;
   f7 H0 vetoes promotion, while H1 or `continue` passes the veto.

The two cohorts must be exact, fresh, and disjoint from each other and from all
prior diagnostic/selection cohorts. Pass the same candidate-bound
cohort-exclusions manifest used by the ordinary gate to
`tools/a1_v5_recovery_gate.py`; the recovery gate replays it against both the
ordinary final intervals and the fixed f7 veto interval. Its output is
promotion-eligible only as a manual recovery authority; `auto_promotion`
remains false.

## No-wave rule

Do not launch a production wave until all of these exist and replay cleanly:

- committed disaster-recovery receipt;
- recovery S1/S2/S3 bindings;
- v3 pre-wave lock with recovery category semantics;
- sealed learner/evaluator software that validates the same semantics;
- a planned strict-parent cohort and disjoint f7 veto cohort.
