#!/usr/bin/env python3
"""CLI: merge Gumbel self-play generation shards + a bounded teacher-replay mix
into one train_bc.py-compatible manifest.json.

Generated (self-play) data is always included in full -- it is the whole
point of the run. Teacher (BC corpus) data is included as a "replay mix" to
anchor training against catastrophic forgetting, sized as a FRACTION OF THE
FINAL COMBINED TOTAL (not a fraction of the teacher corpus itself, which is
typically far larger than one generation's worth of self-play games): e.g.
--replay-fraction 0.15 with 24k gen games worth of rows means the final
manifest is ~85% gen-1 rows / ~15% teacher rows, regardless of how much
teacher data exists.

Per-shard row counts aren't recorded anywhere (only the aggregate
`converted_rows` and shard count per source manifest -- see
`tools/build_combined_entity_manifest.py`), so teacher shards are selected
via each source's *average* rows-per-shard, shuffled deterministically by
`--seed` and taken until the row budget is (at least) met. This is an
approximation: actual shard sizes vary somewhat, so the true replay fraction
lands close to but not exactly at the requested value -- `actual_replay_fraction`
in the output manifest reports the real number.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge Gumbel self-play generation shards with a bounded teacher-replay "
        "mix into one train_bc.py-compatible manifest.json."
    )
    parser.add_argument(
        "--gen-input",
        action="append",
        required=True,
        help="Self-play generation manifest.json or its directory (repeatable; e.g. one per "
        "host). Included in full.",
    )
    parser.add_argument(
        "--teacher-input",
        action="append",
        default=[],
        help="Teacher/BC corpus manifest.json or its directory (repeatable). Subsampled "
        "(whole shards) to hit --replay-fraction of the final combined total.",
    )
    parser.add_argument(
        "--replay-fraction",
        type=float,
        default=0.15,
        help="Target fraction of the FINAL combined row count contributed by --teacher-input "
        "(0.0-0.9). Default 0.15 (i.e. ~15%% teacher replay, ~85%% gen-1).",
    )
    parser.add_argument("--out", required=True, help="Output directory for the combined manifest.json")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic teacher-shard selection seed.")
    args = parser.parse_args()

    result = build_manifest(
        gen_inputs=[Path(p) for p in args.gen_input],
        teacher_inputs=[Path(p) for p in args.teacher_input],
        replay_fraction=float(args.replay_fraction),
        out_dir=Path(args.out),
        seed=int(args.seed),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def build_manifest(
    *,
    gen_inputs: list[Path],
    teacher_inputs: list[Path],
    replay_fraction: float,
    out_dir: Path,
    seed: int,
) -> dict[str, Any]:
    if not (0.0 <= replay_fraction < 1.0):
        raise SystemExit(f"--replay-fraction must be in [0.0, 1.0), got {replay_fraction!r}")

    gen_shards: list[str] = []
    gen_rows = 0
    gen_summaries: list[dict[str, Any]] = []
    for gen_input in gen_inputs:
        manifest_path = _resolve_manifest_path(gen_input)
        shards, rows = _load_manifest_shards_and_rows(manifest_path)
        gen_shards.extend(shards)
        gen_rows += rows
        gen_summaries.append({"manifest": str(manifest_path), "rows": rows, "shards": len(shards)})

    teacher_sources: list[tuple[list[str], int]] = []
    teacher_summaries: list[dict[str, Any]] = []
    for teacher_input in teacher_inputs:
        manifest_path = _resolve_manifest_path(teacher_input)
        shards, rows = _load_manifest_shards_and_rows(manifest_path)
        teacher_sources.append((shards, rows))
        teacher_summaries.append({"manifest": str(manifest_path), "rows": rows, "shards": len(shards)})

    if replay_fraction > 0.0 and teacher_sources:
        # total = gen_rows + teacher_rows, want teacher_rows / total == replay_fraction
        # => teacher_rows = replay_fraction * gen_rows / (1 - replay_fraction)
        target_teacher_rows = int(round(replay_fraction * gen_rows / (1.0 - replay_fraction)))
        teacher_shards, teacher_rows = _select_teacher_shards_for_budget(
            teacher_sources, target_rows=target_teacher_rows, seed=seed
        )
    else:
        teacher_shards, teacher_rows = [], 0

    all_shards = gen_shards + teacher_shards
    converted_rows = gen_rows + teacher_rows
    actual_replay_fraction = (teacher_rows / converted_rows) if converted_rows else 0.0

    combined = {
        "schema": "entity_tokens_v1",
        "combined_manifest_schema": "gumbel_gen_replay_mix_v1",
        "converted_rows": converted_rows,
        "gen_rows": gen_rows,
        "teacher_rows": teacher_rows,
        "requested_replay_fraction": replay_fraction,
        "actual_replay_fraction": actual_replay_fraction,
        "gen_inputs": gen_summaries,
        "teacher_inputs": teacher_summaries,
        "seed": seed,
        "shards": all_shards,
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(combined, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {
        "out": str(manifest_path),
        "gen_rows": gen_rows,
        "teacher_rows": teacher_rows,
        "converted_rows": converted_rows,
        "actual_replay_fraction": actual_replay_fraction,
        "shards": len(all_shards),
    }


def _resolve_manifest_path(path: Path) -> Path:
    if path.is_file():
        return path
    manifest = path / "manifest.json"
    if manifest.exists():
        return manifest
    raise SystemExit(f"missing manifest.json for {path}")


def _load_manifest_shards_and_rows(manifest_path: Path) -> tuple[list[str], int]:
    """Resolve a manifest's shard paths (relative-to-manifest fallback, same
    convention as `tools/build_combined_entity_manifest.py`) and its aggregate
    row count."""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    files: list[Path] = []
    missing: list[str] = []
    for value in payload.get("shards", ()):
        raw = Path(value)
        candidates = [raw] if raw.is_absolute() else [raw, manifest_path.parent / raw]
        if raw.is_absolute():
            candidates.append(manifest_path.parent / raw.name)
        chosen = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
        files.append(chosen)
        if not chosen.exists():
            missing.append(str(chosen))
    if missing:
        preview = ", ".join(missing[:5])
        raise SystemExit(f"{manifest_path} points to missing shards: {preview}")
    rows = 0
    for key in ("converted_rows", "rows", "samples"):
        value = payload.get(key)
        if value is not None:
            rows = int(value)
            break
    return [str(path) for path in files], rows


def _select_teacher_shards_for_budget(
    sources: list[tuple[list[str], int]], *, target_rows: int, seed: int
) -> tuple[list[str], int]:
    """Select whole shards across `sources` (each a (shard_paths, aggregate_rows)
    pair from one teacher manifest) until at least `target_rows` is reached,
    using each source's *average* rows-per-shard as a per-shard estimate
    (real per-shard row counts aren't recorded anywhere -- see module
    docstring). Shard order within each source is shuffled deterministically
    by `seed` (round-robin across sources) so repeated runs with the same
    seed are reproducible and successive generations draw a fresh mix rather
    than always the same shard prefix.
    """
    if target_rows <= 0:
        return [], 0

    per_source: list[tuple[list[str], float]] = []
    for shards, rows in sources:
        if not shards:
            continue
        rows_per_shard = rows / len(shards) if len(shards) else 0.0
        shuffled = list(shards)
        random.Random(seed).shuffle(shuffled)
        per_source.append((shuffled, rows_per_shard))

    selected: list[str] = []
    selected_rows = 0.0
    indices = [0] * len(per_source)
    exhausted = [False] * len(per_source)
    while selected_rows < target_rows and not all(exhausted):
        for source_index, (shuffled, rows_per_shard) in enumerate(per_source):
            if exhausted[source_index]:
                continue
            position = indices[source_index]
            if position >= len(shuffled):
                exhausted[source_index] = True
                continue
            selected.append(shuffled[position])
            selected_rows += rows_per_shard
            indices[source_index] += 1
            if selected_rows >= target_rows:
                break

    return selected, int(round(selected_rows))


if __name__ == "__main__":
    main()
