#!/usr/bin/env python3
"""Build a synthetic train_bc.py-compatible manifest.json for a harvested
gen-1 self-play corpus that has no top-level manifest.json (generation was
still running at harvest time, so the top-level manifest each worker writes
on completion doesn't exist yet). Scans every .npz shard's real row count
(obs.shape[0]) rather than assuming a fixed shard size."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
out_manifest = Path(sys.argv[2])

shards = sorted(root.rglob("*.npz"))
if not shards:
    raise SystemExit(f"no .npz shards found under {root}")

total_rows = 0
row_counts = []
bad = []
for shard in shards:
    try:
        with np.load(shard, allow_pickle=True) as d:
            rows = int(d["obs"].shape[0])
    except Exception as e:  # noqa: BLE001
        bad.append((str(shard), repr(e)))
        continue
    row_counts.append(rows)
    total_rows += rows

manifest = {
    "schema": "entity_tokens_v1",
    "synthetic_manifest": True,
    "note": "Synthesized by build_synthetic_manifest.py because generation was "
    "still running at harvest time (no top-level manifest.json existed yet). "
    "Row counts are real (read from each shard's obs.shape[0]), not estimated.",
    "converted_rows": total_rows,
    "rows": total_rows,
    "shard_count": len(row_counts),
    "min_shard_rows": min(row_counts) if row_counts else 0,
    "max_shard_rows": max(row_counts) if row_counts else 0,
    "bad_shards": bad,
    "shards": [str(p) for p in shards if not any(p == Path(b[0]) for b in bad)],
}
out_manifest.parent.mkdir(parents=True, exist_ok=True)
out_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(json.dumps({
    "total_rows": total_rows,
    "shard_count": len(row_counts),
    "min_shard_rows": manifest["min_shard_rows"],
    "max_shard_rows": manifest["max_shard_rows"],
    "bad_shards": len(bad),
    "out": str(out_manifest),
}, indent=2))
