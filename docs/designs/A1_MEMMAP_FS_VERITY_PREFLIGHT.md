# A1 memmap preflight caching

## Decision

Do not cache A1 payload verification from file timestamps, inode metadata, or an
ordinary receipt.  The current preflight deliberately re-hashes every payload
byte once per single-node job; all local DDP ranks reuse rank 0's result.  Any
replacement must retain detection of payload mutation and silent storage
corruption.

The supported future fast path is **fs-verity**, provisioned on a dedicated A1
corpus volume.  fs-verity makes each payload file permanently read-only, checks
Merkle proofs on every read (including `mmap`), and exposes the enforced file
digest in constant time.  This preserves the integrity purpose of the full
SHA-256 pass while removing the repeated whole-corpus read after one sealing
pass.

## Current B200 finding (2026-07-12)

- Kernel `6.8.0-1046-nvidia` has `CONFIG_FS_VERITY=y`.
- The A1 corpora live on ext4 `/dev/vda1`.
- That filesystem does **not** have the ext4 `verity` superblock feature.
- An isolated loopback ext4 filesystem formatted with `-O verity` was tested:
  payload mutation was rejected, reads remained kernel-verified, and
  `fsverity digest` completed in constant time.

Do not run `tune2fs -O verity /dev/vda1` as an experiment.  It is an
irreversible filesystem-wide `RO_COMPAT` change: older kernels can only mount
the filesystem read-only and older `e2fsck` versions cannot check it.  The live
root volume therefore keeps the existing full-hash preflight.

## Provisioning design

For the next corpus volume:

1. Format a dedicated ext4 volume with the `verity` feature and mount it only
   on kernels and recovery images known to support fs-verity.
2. Build the corpus normally.  During a single sealing transaction, verify the
   existing raw SHA-256 inventory in full, then enable fs-verity on every
   authenticated payload file.
3. Write a root-owned receipt in a root-owned directory.  Bind the canonical
   corpus path, corpus metadata SHA-256, inventory SHA-256, filename, size,
   device/inode identity, declared raw SHA-256, and measured fs-verity digest
   for every payload.  Seal the receipt with fs-verity too.
4. The trainer fast path must fail closed unless every expected payload is a
   verity file and its constant-time measured digest, size, device/inode, and
   receipt binding all match.  Any absent, extra, replaced, unsealed, or
   mismatched file falls back to full verification or aborts.
5. Keep the current full-hash path for unsupported filesystems, legacy corpora,
   multi-node paths whose storage identity is not proven shared, and receipt
   schema/version drift.

An immutable-bit-only receipt is not sufficient: ext4 does not checksum file
data, so it cannot prove that unchanged metadata still refers to unchanged
payload bytes after silent storage corruption.

## Safe fast path on the current ext4 volume: one matched-sweep transaction

Do not add a cross-process or cross-job preflight cache on the current volume.
There is, however, a safe way to avoid one 600+ GiB scan **per arm** without
claiming that mutable ext4 metadata authenticates bytes: make the matched sweep
one trust transaction and keep one DDP executor alive for its entire lifetime.

The transaction has two byte-verification boundaries:

1. Before allocating a model, rank 0 performs the existing full inventory
   verification over every ordered composite component. It broadcasts the
   canonical authenticated metadata and digest to the other local ranks.
2. The executor loads each corpus exactly once and runs all reviewed arms in
   the same process group. No arm starts another `train_bc` process and no arm
   accepts an ordinary on-disk cache receipt in place of byte verification.
3. After the last arm, rank 0 repeats the full inventory verification. Arm
   reports and checkpoints remain tentative until this postflight succeeds and
   reproduces the exact preflight digest. A mismatch quarantines every output
   from the sweep; it does not merely invalidate the final arm.

This changes `N` full scans into two scans for an `N`-arm sweep. The postflight
is required on mutable ext4: it detects accidental writes and persistent silent
storage corruption anywhere in the transaction. It provides the same accepted-
artifact guarantee as independently preflighting each arm for the repository's
non-adversarial experiment threat model. It does not protect against a malicious
writer that changes bytes and restores the exact original bytes during the
transaction; neither does the current preflight protect an individual arm from
such a writer after its startup scan. Use fs-verity when that stronger threat
model is required.

### Independence contract between arms

Reusing the authenticated corpus and process group must not turn the sweep into
continued training. Before every arm, every rank must:

- reconstruct a new model object and reload the same authenticated parent
  checkpoint state (not the preceding arm's state);
- construct a new optimizer, scheduler, gradient scaler and DDP wrapper;
- reset Python, NumPy, Torch CPU and Torch CUDA RNGs to the arm's sealed seed;
- reset samplers, batch order, metrics, early-stop state and accumulation
  counters;
- assert that no optimizer/scaler/scheduler state object from a prior arm is
  reachable, and synchronize all ranks before the first step;
- write to a unique tentative output directory and never use another arm's
  checkpoint as initialization.

After an arm, destroy its DDP/model/optimizer objects, synchronize ranks, run a
CUDA leak assertion against a reviewed tolerance, and only then construct the
next arm. The parent checkpoint should be loaded once into immutable CPU state
and cloned into each fresh model; tensor storage must not alias a trained arm.

### Sealed sweep authority

The executor must consume one reviewed manifest that binds:

- clean repository commit and runtime-code inventory;
- ordered corpus descriptors, validation manifests, metadata hashes and payload
  inventory hashes;
- parent checkpoint bytes and architecture;
- world size, local/global batch, exact sample dose and warmup in samples;
- the complete ordered arm list and the fields permitted to differ;
- identical train/validation seed sets and batch-order seed where the comparison
  requires common random numbers;
- unique output paths and a transaction identifier.

Cheap manifest, checkpoint, topology, output-path and host checks run before the
large preflight. Once byte authentication begins, the arm list is immutable.
Failure of any rank, any arm, the process group, or postflight makes the whole
transaction non-publishable. Resume starts a new transaction and repeats the
preflight; it never trusts a partial transaction's receipt.

### Implementation boundary

This is deliberately a persistent **trainer** API, not a wrapper that launches
the existing `train_bc.py` CLI repeatedly. Repeated child CLIs necessarily
repeat their fail-closed preflight; adding a `--skip-payload-hash` flag or an
environment variable would create an unauthenticated production bypass.

Refactor the current single-run trainer into these internal operations before
building the sweep executor:

1. `authenticate_and_load_corpus()` -- current preflight plus one read-only
   corpus construction;
2. `build_fresh_training_state(parent_state, arm)` -- model, optimizer,
   scheduler, scaler, DDP and RNG reset;
3. `train_one_arm(authenticated_corpus, state, arm)` -- no authentication
   switches and no process creation;
4. `destroy_training_state(state)` -- barrier and leak checks;
5. `postflight_and_publish(transaction)` -- repeat byte authentication, compare
   the digest, then atomically publish all individual receipts.

Unit tests must prove one preflight and one postflight for multiple arms,
distinct optimizer/model object identities, bit-identical starting parameters,
no state carryover, refusal on an altered payload, quarantine on postflight
failure, refusal of partial resume, and no public CLI/hash-skip escape hatch. An
end-to-end two-arm tiny-corpus DDP test must compare each arm with the same arm
run alone.

Until that persistent API and its tests exist, retain the current per-job full
scan. The active P1 service must not be changed or restarted to adopt an
unfinished optimization.
