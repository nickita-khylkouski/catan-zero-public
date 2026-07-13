# Public award feature contract

## Why the runtime fix is not sufficient

`player_tokens[..., 12]` is the public `has_longest_road` bit.  Python snapshot
conversion and direct Rust featurization accidentally emitted zero through
`catanatron_rs` 0.1.7.  Version 0.1.8 fixes both producers, but enabling that
bit for an old checkpoint would not be function-preserving.

A read-only audit on the B200 controller found **zero nonzero slot-12 values**
in every inspected corpus:

| corpus | rows | nonzero slot 12 | nonzero largest-army slot 11 |
|---|---:|---:|---:|
| `memmap_gen2_20260706` | 3,648,516 | 0 | 653,411 |
| `memmap_a1_fresh_mixed_12000games` | 2,927,924 | 0 | 449,915 |
| A1 n128 full-140k | 31,919,276 | 0 | 8,443,850 |
| A1 n256 full-56k | 12,773,247 | 0 | 3,429,529 |
| R3 stored-v2 | 138,664 | 0 | 24,993 |

All five also had zero event-mask entries, while road length and largest army
were populated.  The problem is isolated to longest-road ownership, not a
general award/history read failure.

The corresponding `player_encoder.0.weight[:, 12]` bytes are exactly identical
(SHA-256 `92e5b71539513a3ee110c1bc7933cfc8fc9bfad1b72c4da578781787705d29ee`)
across gen1, gen2A, gen3, gen4, f7, the n256 temperature student, AUX0, and the
combined-196k candidate.  That proves the column never received a data
gradient; its nonzero values are initialization, not learned semantics.

The risk is not merely theoretical.  On a real award-active state (random-game
seed 3001, tick 173), feeding the corrected bit directly to f7 changed one
two-action prior by as much as `0.0527168` and changed the raw scalar value by
`0.4045253`.  The legacy bridge restores the old zero-input function.

## Checkpoint-owned bridge

Entity checkpoints now carry `public_award_feature_contract`:

- `legacy_zero_v0`: zero only player slot 12 immediately before the forward
  pass.  Missing metadata resolves to this value.  This preserves every old
  checkpoint's function under a 0.1.8 runtime.
- `authoritative_v1`: pass the public longest-road bit through.  This may be
  stamped only by a training transaction whose corpus proves corrected feature
  production and that actually optimized the column.

Unknown values fail closed.  The bridge does not modify largest army, road
length, hidden-information masking, action inputs, or the caller's feature
batch.

## Rollout gate

The software path for this transition is now explicit and fail-closed:

- New generation manifests carry `public-award-feature-provenance-v1`.  The
  native producer refuses to attest unless the installed wheel advertises
  `public_award_feature_parity`.
- `build_memmap_corpus.py` binds every source manifest hash into
  `public-award-corpus-provenance-v1`; absent producer metadata is legacy and
  old+corrected input is labelled `mixed_v0`, never guessed from observed
  feature values.
- `train_bc.py --public-award-feature-contract authoritative_v1` accepts only
  an entirely corrected, authenticated memmap corpus.  On a legacy initializer
  it deterministically zero-initializes `player_encoder.0.weight[:, 12]` before
  optimizer construction, then stamps the output checkpoint and report.
- Mixed corpora require `--allow-mixed-public-award-feature-contracts` and may
  train only under `legacy_zero_v0`; they cannot authorize an authoritative
  checkpoint.  Omitting both flags preserves the historical legacy function.

The remaining operational rollout is:

1. Build fresh shards with 0.1.8 and retain the emitted `authoritative_v1`
   shard/corpus provenance.
2. Assert the corrected corpus contains plausible nonzero slot-12 rows and
   that slot 12 remains public under masked and unmasked feature paths.
3. Train from f7 with the learner consuming corrected rows.  Stamp the output
   checkpoint `authoritative_v1`; never stamp the initializer.
4. Prove the trained column differs from the frozen legacy bytes and run
   masked/unmasked Python/native feature parity plus candidate evaluation.
5. Only then may an authoritative checkpoint consume the 0.1.8 bit in
   production.  Existing v5/f7-lineage checkpoints remain on the legacy bridge.

The event-history rollout is a separate schema change and is specified in
`NATIVE_EVENT_HISTORY_ROLLOUT.md`; it must not be bundled into this compatibility
transition.
