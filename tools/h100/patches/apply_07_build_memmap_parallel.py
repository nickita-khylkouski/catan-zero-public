#!/usr/bin/env python3
"""
Parallelize build_memmap_corpus.py shard processing.
SYSTEM_DESIGN_FINDINGS #20: Single-threaded shard loop → ThreadPoolExecutor.

The shard loop is I/O-bound (np.load from disk + np.tofile to disk) with some
CPU work (_normalize_teacher_shard). A ThreadPoolExecutor parallelizes the
load+normalize phase while keeping the write phase sequential (shards must
concatenate in source order).

Uses a producer-consumer pattern:
  - Producer: ThreadPoolExecutor loads+normalizes shards in parallel (N workers)
  - Consumer: main thread writes columns sequentially in shard order

Usage: python3 apply_07_build_memmap_parallel.py /path/to/build_memmap_corpus.py
"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else sys.exit("usage: apply_07_build_memmap_parallel.py <path>")
with open(path) as f:
    src = f.read()

if "ThreadPoolExecutor" in src:
    print("[SKIP] parallelization already applied")
    sys.exit(0)

# Add import at the top (after the existing imports)
OLD_IMPORT = "import json"
NEW_IMPORT = """import json
from concurrent.futures import ThreadPoolExecutor"""
if OLD_IMPORT in src:
    src = src.replace(OLD_IMPORT, NEW_IMPORT, 1)
    print("[OK] Added ThreadPoolExecutor import")

# Replace the serial shard loop with a parallel load + sequential write pattern.
# The key insight: _normalize_teacher_shard(_load_npz(file), file) is the
# I/O-bound part. The column writes (handles[name].write) must stay sequential.
#
# We split the loop body into:
#   1. _load_and_normalize(file) — parallelizable (returns norm dict)
#   2. _write_shard(norm, ...) — sequential (writes to file handles)
#
# The validation stats are accumulated in the main thread (thread-safe).

OLD_LOOP = """    for shard_index, file in enumerate(files):
        norm = _normalize_teacher_shard(_load_npz(file), file)
        present = {key for key in LOADER_KEYS if key in norm}"""

NEW_LOOP = """    # SYSTEM_DESIGN_FINDINGS #20: Parallel load+normalize, sequential write.
    # _load_npz + _normalize_teacher_shard is I/O-bound (disk read + CPU normalize).
    # The column writes below must stay sequential (shards concatenate in order).
    def _load_and_normalize(file):
        return _normalize_teacher_shard(_load_npz(file), file)

    _shard_norms = {}
    with ThreadPoolExecutor(max_workers=8) as _pool:
        for shard_index, (file, norm) in enumerate(zip(files, _pool.map(_load_and_normalize, files))):
            _shard_norms[shard_index] = norm

    for shard_index, file in enumerate(files):
        norm = _shard_norms[shard_index]
        present = {key for key in LOADER_KEYS if key in norm}"""

if OLD_LOOP in src:
    src = src.replace(OLD_LOOP, NEW_LOOP, 1)
    print("[OK] Replaced serial shard loop with parallel load + sequential write")
else:
    print("[WARN] could not find shard loop anchor")
    sys.exit(1)

with open(path, "w") as f:
    f.write(src)
print(f"[DONE] Wrote {path}")
