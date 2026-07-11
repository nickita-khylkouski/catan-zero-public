#!/usr/bin/env python3
"""Derive immutable deterministic n128 comparison subsets from its full manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools import build_memmap_corpus as corpus  # noqa: E402

SCHEMA = corpus.DUAL_ARM_SELECTED_GAMES_SCHEMA
RULE = "stable-hash-per-worker-category-v1"
TARGETS = {
    "matched-56k": {"current_producer": 1600, "recent_history": 300, "hard_negative": 100},
    "compute-112k": {"current_producer": 3200, "recent_history": 600, "hard_negative": 200},
}


def _rank(record: dict[str, Any], subset_id: str) -> bytes:
    return hashlib.sha256(
        f"{subset_id}\0{record['worker_id']}\0{record['category']}\0{record['game_seed']}".encode()
    ).digest()


def build_subsets(source: Path, out_dir: Path) -> dict[str, Path]:
    source = source.expanduser().resolve(strict=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    validated = corpus._load_a1_selected_game_manifest(source)  # noqa: SLF001
    if validated.get("arm_id") != "n128" or validated.get("subset_id") != "full-140k":
        raise SystemExit("subset source must be the full n128 140k manifest")
    records = list(payload["records"])
    workers = sorted({str(record["worker_id"]) for record in records})
    if len(workers) != 28:
        raise SystemExit("full n128 manifest must contain exactly 28 workers")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["worker_id"]), str(record["category"]))].append(record)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for subset_id, targets in TARGETS.items():
        selected: list[dict[str, Any]] = []
        for worker in workers:
            for category, want in targets.items():
                choices = sorted(grouped[(worker, category)], key=lambda row: (_rank(row, subset_id), row["game_seed"]))
                if len(choices) < want:
                    raise SystemExit(f"{worker}/{category} has {len(choices)} games, need {want}")
                selected.extend(choices[:want])
        selected.sort(key=lambda row: (row["game_seed"], row["job_id"]))
        training = [row["game_seed"] for row in selected if row["split"] == "train"]
        validation = [row["game_seed"] for row in selected if row["split"] == "validation"]
        if not training or not validation:
            raise SystemExit(f"{subset_id} has an empty train or validation split")
        category_counts = {name: count * len(workers) for name, count in targets.items()}
        value = {
            "schema_version": SCHEMA,
            "arm_id": "n128",
            "subset_id": subset_id,
            "a1_contract_sha256": validated["a1_contract_sha256"],
            "selection_rule": RULE,
            "selected_game_count": len(selected),
            "selected_game_seed_set_sha256": corpus._game_seed_set_sha256([row["game_seed"] for row in selected]),  # noqa: SLF001
            "category_game_counts": category_counts,
            "training_game_count": len(training),
            "training_game_seed_set_sha256": corpus._game_seed_set_sha256(training),  # noqa: SLF001
            "validation_game_count": len(validation),
            "validation_game_seed_set_sha256": corpus._game_seed_set_sha256(validation),  # noqa: SLF001
            "records_sha256": corpus._value_sha256(selected),  # noqa: SLF001
            "records": selected,
            "parent_manifest_sha256": corpus._file_sha256(source),  # noqa: SLF001
        }
        path = out_dir / f"n128-{subset_id}.json"
        data = json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o444)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        outputs[subset_id] = path
    return outputs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n128-full-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    outputs = build_subsets(args.n128_full_manifest, args.out_dir)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
