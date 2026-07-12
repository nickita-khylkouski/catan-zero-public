#!/usr/bin/env python3
"""Derive an immutable whole-game validation sentinel from a composite holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Mapping, Sequence

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
    excluded_selection_game_seeds: set[int] | None = None,
    component_target_ratios: Mapping[str, float] | None = None,
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
    component_ids = list(meta.get("component_ids", ()))
    if len(component_ids) != len(corpus.corpora):
        raise SystemExit("composite component identities differ from loaded corpora")
    excluded = set() if excluded_selection_game_seeds is None else {
        int(seed) for seed in excluded_selection_game_seeds
    }
    if not excluded.issubset(full_seed_set):
        raise SystemExit("excluded sentinel games are outside the authenticated holdout")
    row_counts: dict[int, int] = {}
    component_row_counts: dict[str, dict[int, int]] = {}
    for component_id, component in zip(component_ids, corpus.corpora, strict=True):
        seeds = np.asarray(component["game_seed"], dtype=np.int64)
        heldout = seeds[np.isin(seeds, np.fromiter(full_seed_set, dtype=np.int64))]
        unique, counts = np.unique(heldout, return_counts=True)
        per_component: dict[int, int] = {}
        for game_seed, count in zip(unique, counts, strict=True):
            key = int(game_seed)
            if key in row_counts:
                raise SystemExit(f"validation game seed overlaps components: {key}")
            row_counts[key] = int(count)
            if key not in excluded:
                per_component[key] = int(count)
        component_row_counts[str(component_id)] = per_component
    if set(row_counts) != full_seed_set:
        raise SystemExit("composite corpus rows differ from authenticated validation games")
    eligible = {seed: rows for seed, rows in row_counts.items() if seed not in excluded}
    if not eligible:
        raise SystemExit("excluded sentinel games consume the complete authenticated holdout")
    if component_target_ratios is None:
        selected, selected_rows = select_whole_games_near_target(
            eligible, target_rows=target_rows, selection_seed=selection_seed
        )
    else:
        ratios = {str(key): float(value) for key, value in component_target_ratios.items()}
        if (
            set(ratios) != set(component_ids)
            or any(not np.isfinite(value) or value <= 0.0 for value in ratios.values())
            or not np.isclose(sum(ratios.values()), 1.0, rtol=0.0, atol=1e-12)
        ):
            raise SystemExit(
                "component target ratios must positively cover every composite component"
            )
        selected = []
        selected_rows = 0
        assigned = 0
        for index, component_id in enumerate(component_ids):
            component_id = str(component_id)
            component_rows = component_row_counts[component_id]
            if not component_rows:
                raise SystemExit(
                    f"component {component_id} has no fresh validation games after exclusion"
                )
            if index + 1 == len(component_ids):
                component_target = target_rows - assigned
            else:
                component_target = int(round(target_rows * ratios[component_id]))
                assigned += component_target
            component_selected, component_selected_rows = select_whole_games_near_target(
                component_rows,
                target_rows=component_target,
                selection_seed=selection_seed,
            )
            selected.extend(component_selected)
            selected_rows += component_selected_rows
        selected.sort()
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
    parser.add_argument(
        "--exclude-selected-games-from",
        type=Path,
        help=(
            "Prior train-validation-game-sentinel-v1 whose evaluated games must "
            "not be selected again. The complete authenticated holdout remains excluded "
            "from training."
        ),
    )
    parser.add_argument(
        "--component-target-ratios-json",
        help="Optional JSON object mapping every component_id to its validation row ratio.",
    )
    args = parser.parse_args(argv)
    excluded: set[int] | None = None
    if args.exclude_selected_games_from is not None:
        try:
            prior = json.loads(args.exclude_selected_games_from.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise SystemExit(f"cannot read prior validation sentinel: {error}") from error
        seeds = prior.get("game_seeds") if isinstance(prior, dict) else None
        if (
            not isinstance(prior, dict)
            or prior.get("schema_version") != SCHEMA
            or not isinstance(seeds, list)
            or not seeds
            or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        ):
            raise SystemExit("prior validation sentinel is malformed")
        excluded = set(seeds)
    ratios: dict[str, float] | None = None
    if args.component_target_ratios_json is not None:
        try:
            raw_ratios = json.loads(args.component_target_ratios_json)
        except json.JSONDecodeError as error:
            raise SystemExit(f"component target ratios are invalid JSON: {error}") from error
        if not isinstance(raw_ratios, dict):
            raise SystemExit("component target ratios must be a JSON object")
        ratios = raw_ratios
    payload = derive(
        args.composite,
        target_rows=args.target_rows,
        selection_seed=args.selection_seed,
        validation_fraction=args.validation_fraction,
        validation_seed=args.validation_seed,
        excluded_selection_game_seeds=excluded,
        component_target_ratios=ratios,
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
