#!/usr/bin/env python3
"""Derive an immutable whole-game validation sentinel from a composite holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Sequence

import numpy as np

import train_bc


SCHEMA = "train-validation-game-sentinel-v1"


def _selection_key(seed: int, selection_seed: int) -> bytes:
    return hashlib.sha256(
        struct.pack("<qq", int(selection_seed), int(seed))
    ).digest()


def select_whole_games_near_target(
    row_counts: dict[int, int], *, target_rows: int, selection_seed: int
) -> tuple[list[int], int]:
    """Choose deterministic whole games without exceeding target, then round closest."""
    if target_rows <= 0 or not row_counts:
        raise ValueError("target_rows and row_counts must be positive")
    if any(count <= 0 for count in row_counts.values()):
        raise ValueError("every game row count must be positive")
    ordered = sorted(
        row_counts,
        key=lambda game_seed: (_selection_key(game_seed, selection_seed), game_seed),
    )
    selected: list[int] = []
    selected_set: set[int] = set()
    total = 0
    for game_seed in ordered:
        count = int(row_counts[game_seed])
        if total + count <= target_rows:
            selected.append(game_seed)
            selected_set.add(game_seed)
            total += count
    # A final whole game may be closer than the under-target result. Ties stay
    # under target, making memory/runtime estimates conservative.
    remaining = [seed for seed in ordered if seed not in selected_set]
    if remaining:
        best = min(
            remaining,
            key=lambda seed: (
                abs(target_rows - (total + int(row_counts[seed]))),
                _selection_key(seed, selection_seed),
                seed,
            ),
        )
        candidate_total = total + int(row_counts[best])
        if abs(target_rows - candidate_total) < abs(target_rows - total):
            selected.append(best)
            total = candidate_total
    if not selected:
        best = min(
            ordered,
            key=lambda seed: (abs(target_rows - int(row_counts[seed])), seed),
        )
        selected = [best]
        total = int(row_counts[best])
    return sorted(selected), total


def _write_immutable(path: Path, payload: dict) -> None:
    path = path.expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    try:
        with path.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise SystemExit(f"refusing to overwrite immutable sentinel {path}") from error
    path.chmod(0o444)


def derive(
    descriptor_path: Path,
    *,
    target_rows: int,
    selection_seed: int,
    validation_fraction: float,
    validation_seed: int,
) -> dict:
    meta = train_bc._preflight_memmap_composite_descriptor(descriptor_path)
    full = train_bc._load_composite_validation_contract(
        meta,
        validation_fraction=validation_fraction,
        validation_seed=validation_seed,
        validation_max_samples=0,
        validation_game_seed_ranges=[],
    )
    corpus = train_bc.load_teacher_data_memmap(descriptor_path, composite_meta=meta)
    full_seed_set = set(map(int, np.asarray(full["game_seeds"], dtype=np.int64)))
    row_counts: dict[int, int] = {}
    for component in corpus.corpora:
        seeds = np.asarray(component["game_seed"], dtype=np.int64)
        heldout = seeds[np.isin(seeds, np.fromiter(full_seed_set, dtype=np.int64))]
        unique, counts = np.unique(heldout, return_counts=True)
        for game_seed, count in zip(unique, counts, strict=True):
            key = int(game_seed)
            if key in row_counts:
                raise SystemExit(f"validation game seed overlaps components: {key}")
            row_counts[key] = int(count)
    if set(row_counts) != full_seed_set:
        raise SystemExit("composite corpus rows differ from authenticated validation games")
    selected, selected_rows = select_whole_games_near_target(
        row_counts, target_rows=target_rows, selection_seed=selection_seed
    )
    contracts = full["component_contracts"]
    payload = {
        "schema_version": SCHEMA,
        "source_composite_descriptor_file_sha256": meta["descriptor_file_sha256"],
        "source_composite_descriptor_fingerprint": meta["descriptor_fingerprint"],
        "source_validation_bindings": [
            {
                "component_index": index,
                "validation_manifest_file_sha256": contract["file_sha256"],
                "validation_manifest_sha256": contract["manifest_sha256"],
                "validation_game_seed_set_sha256": contract[
                    "validation_game_seed_set_sha256"
                ],
            }
            for index, contract in enumerate(contracts)
        ],
        "selection_seed": int(selection_seed),
        "target_row_count": int(target_rows),
        "selected_row_count": int(selected_rows),
        "selected_game_seed_count": len(selected),
        "selected_game_seed_set_sha256": train_bc._game_seed_set_sha256(
            np.asarray(selected, dtype=np.int64)
        ),
        "excluded_game_seed_count": len(full_seed_set),
        "excluded_game_seed_set_sha256": full["validation_game_seed_set_sha256"],
        "game_seeds": selected,
    }
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composite", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-rows", type=int, default=262_144)
    parser.add_argument("--selection-seed", type=int, default=20260711)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--validation-seed", type=int, default=17)
    args = parser.parse_args(argv)
    payload = derive(
        args.composite,
        target_rows=args.target_rows,
        selection_seed=args.selection_seed,
        validation_fraction=args.validation_fraction,
        validation_seed=args.validation_seed,
    )
    _write_immutable(args.out, payload)
    print(json.dumps({
        "progress": "validation_game_sentinel_written",
        "path": str(args.out.expanduser().absolute()),
        "selected_rows": payload["selected_row_count"],
        "selected_games": payload["selected_game_seed_count"],
        "target_rows": payload["target_row_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
