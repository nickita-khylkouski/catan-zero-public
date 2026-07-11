#!/usr/bin/env python3
"""Derive immutable deterministic n128 comparison subsets from its full manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import uuid
from collections import Counter, defaultdict
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


def _existing_bytes_match(path: Path, expected: bytes) -> bool:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError:
        return False
    except OSError as error:
        raise SystemExit(f"cannot inspect immutable subset artifact {path}: {error}") from error
    try:
        before = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or not stat.S_ISREG(named.st_mode)
            or (before.st_dev, before.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise SystemExit(f"existing immutable subset artifact is unsafe: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or b"".join(chunks) != expected
        ):
            raise SystemExit(f"existing immutable subset artifact drift: {path}")
        return True
    finally:
        os.close(descriptor)


def _write_immutable(path: Path, value: dict[str, Any]) -> None:
    data = json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"
    if _existing_bytes_match(path, data):
        return
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o444,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            if not _existing_bytes_match(path, data):
                raise SystemExit(f"existing immutable subset artifact drift: {path}")
        parent_fd = os.open(
            path.parent,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        temporary.unlink(missing_ok=True)
        parent_fd = os.open(
            path.parent,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)


def _rows_by_seed(
    data_shards: list[dict[str, Any]], selected: set[int]
) -> Counter[int]:
    import numpy as np

    selected_array = np.asarray(sorted(selected), dtype=np.int64)
    counts: Counter[int] = Counter()
    for shard in data_shards:
        with np.load(Path(str(shard["path"])), allow_pickle=False) as payload:
            seeds = np.asarray(payload["game_seed"], dtype=np.int64)
            counts.update(map(int, seeds[np.isin(seeds, selected_array)].tolist()))
    missing = selected - set(counts)
    if missing:
        raise SystemExit(f"full dual-arm audit is missing {len(missing)} selected games")
    return counts


def build_subsets(
    source: Path, parent_audit: Path, out_dir: Path
) -> dict[str, Path]:
    source = source.expanduser().resolve(strict=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    validated = corpus._load_a1_selected_game_manifest(source)  # noqa: SLF001
    if validated.get("arm_id") != "n128" or validated.get("subset_id") != "full-140k":
        raise SystemExit("subset source must be the full n128 140k manifest")
    parent_audit = parent_audit.expanduser().resolve(strict=True)
    parent_validated = corpus._load_a1_post_wave_audit(  # noqa: SLF001
        parent_audit, validated
    )
    parent_payload = json.loads(parent_audit.read_text(encoding="utf-8"))
    if parent_payload.get("schema_version") != corpus.DUAL_ARM_AUDIT_SCHEMA:
        raise SystemExit("subset parent must be a full dual-arm post-wave audit")
    records = list(payload["records"])
    rows_by_seed = _rows_by_seed(
        list(parent_validated["data_shards"]),
        {int(record["game_seed"]) for record in records},
    )
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
        value: dict[str, Any] = {
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
        _write_immutable(path, value)

        selected_set = {int(row["game_seed"]) for row in selected}
        validation_set = {int(seed) for seed in validation}
        selected_rows = sum(rows_by_seed[seed] for seed in selected_set)
        validation_rows = sum(rows_by_seed[seed] for seed in validation_set)
        if validation_rows <= 0 or validation_rows >= selected_rows:
            raise SystemExit(
                "derived subset has invalid selected/validation row exposure"
            )
        validation_path = out_dir / f"n128-{subset_id}.validation_seeds.json"
        validation_value = {
            "schema_version": "train-validation-game-seeds-v1",
            "a1_contract_sha256": validated["a1_contract_sha256"],
            "validation_fraction": 0.05,
            "validation_seed": 17,
            "validation_max_samples": 0,
            "validation_game_seed_ranges": [],
            "validation_game_seed_count": len(validation),
            "validation_row_count": validation_rows,
            "validation_game_seed_set_sha256": corpus._game_seed_set_sha256(validation),  # noqa: SLF001
            "game_seeds": sorted(validation),
        }
        _write_immutable(validation_path, validation_value)

        audit_path = out_dir / f"n128-{subset_id}.audit.json"
        selected_binding = {
            "manifest": str(path.resolve()),
            "manifest_sha256": corpus._value_sha256(value),  # noqa: SLF001
            "manifest_file_sha256": corpus._file_sha256(path),  # noqa: SLF001
            "selected_game_count": len(selected),
            "selected_game_seed_set_sha256": value["selected_game_seed_set_sha256"],
            "records_sha256": value["records_sha256"],
        }
        validation_binding = {
            "manifest": str(validation_path.resolve()),
            "manifest_sha256": corpus._value_sha256(validation_value),  # noqa: SLF001
            "manifest_file_sha256": corpus._file_sha256(validation_path),  # noqa: SLF001
            "validation_game_seed_count": len(validation),
            "validation_game_seed_set_sha256": validation_value[
                "validation_game_seed_set_sha256"
            ],
        }
        audit_value = {
            "schema_version": corpus.DUAL_ARM_DERIVED_AUDIT_SCHEMA,
            "arm_id": "n128",
            "subset_id": subset_id,
            "contract_path": parent_payload["contract_path"],
            "contract_sha256": parent_payload["contract_sha256"],
            "passed": True,
            "errors": [],
            "category_game_counts": category_counts,
            "total_unique_games": len(selected),
            "selection_rule": RULE,
            "rows": selected_rows,
            "shards": parent_payload["shards"],
            "shard_inventory_sha256": parent_payload["shard_inventory_sha256"],
            "source_provenance": parent_payload["source_provenance"],
            "harvest_relocation": parent_payload["harvest_relocation"],
            "selected_training_games": selected_binding,
            "validation_holdout": validation_binding,
            "parent_audit": {
                "path": str(parent_audit),
                "file_sha256": corpus._file_sha256(parent_audit),  # noqa: SLF001
                "audit_sha256": parent_payload["audit_sha256"],
                "selected_manifest_file_sha256": corpus._file_sha256(source),  # noqa: SLF001
                "shard_inventory_sha256": parent_payload[
                    "shard_inventory_sha256"
                ],
            },
        }
        audit_value["audit_sha256"] = corpus._value_sha256(audit_value)  # noqa: SLF001
        _write_immutable(audit_path, audit_value)
        outputs[subset_id] = path
    return outputs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n128-full-manifest", type=Path, required=True)
    parser.add_argument("--n128-full-audit", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    outputs = build_subsets(
        args.n128_full_manifest, args.n128_full_audit, args.out_dir
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
