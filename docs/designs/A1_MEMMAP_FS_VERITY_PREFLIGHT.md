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
