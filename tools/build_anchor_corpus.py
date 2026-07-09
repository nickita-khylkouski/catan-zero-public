#!/usr/bin/env python3
"""Build a per-generation anchor corpus from reserved, held-out ``.valonly`` seeds.

CAT-30 (Roadmap Step A6 / CAT-26's anchor-refresh-series operation). Generalizes
the one-off ``anchor_r7`` build into a repeatable script: given one or more
source shard roots (a window/corpus this generation trained on, or a dedicated
held-out self-play batch) and the reserved ``.valonly`` game_seed range(s) from
the seed ledger, extracts ONLY the rows whose ``game_seed`` falls in those
ranges into a new, permanently-held-out anchor corpus.

DECISION RULE (Roadmap Sec 1 standing rule; R8/gen-4 lesson, adopted CAT-26):
anchor telemetry measured against a corpus built by this script is a DRIFT
TRIPWIRE ONLY -- it must never be read by any promotion/gate decision code
path. gen-4 showed "the historical promotion signature" (the pattern that
previously correlated with a good gate) and still gated flat: a flat anchor
does not reliably predict a flat gate, so it cannot be trusted the other
direction either. See ``src/catan_zero/rl/flywheel/config.py``'s
``anchor_corpora``/``anchor_*`` fields and ``tools/continuous_flywheel.py``'s
anchor-probe wiring for how this is kept out of the promotion decision.

ANCHOR TYPES (CAT-26 R9 edit adds the third):
  - "current_window": built from the CURRENT window's held-out .valonly seeds
    (e.g. anchor_gen4). Freshest, most representative of current self-play
    distribution, but shortest-lived signal.
  - "longitudinal": an OLDER anchor kept forever once built (e.g. anchor_r7)
    so drift can be measured against a fixed point further in the past, not
    just the immediately preceding generation. This script never overwrites
    an existing anchor of this type; each run produces a new, separately
    named corpus and manifest entry.
  - "external_hard" (STUB, R9 #14): built from states where we lost to an
    external opponent (catanatron_value), high-regret openings, or specific
    failure-phase (robber/dev/discard) states -- NEVER our own self-play
    distribution, so it can catch "beats itself, loses to a different style"
    drift the other two anchor types structurally cannot see. Provenance
    tagging to populate this does not exist yet anywhere in the data
    pipeline (see HARD_ANCHOR_TAG_SCHEMA below) -- this mode is wired and
    tested against synthetic tag columns, but will raise a clear error
    against real production shards until a generator starts writing those
    columns.

REUSE: row extraction reuses ``train_bc._load_npz``/``_normalize_teacher_shard``
(the exact normalization ``build_memmap_corpus.py`` and the live training loader
both use, so a filtered shard is byte-identical in schema to an unfiltered one).
The actual memmap corpus is then built with ``build_memmap_corpus.build_memmap_corpus``
UNMODIFIED -- this script only does the row-level seed/tag filtering into a
staging directory of ordinary filtered .npz shards, then hands off.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from train_bc import _load_npz, _normalize_teacher_shard, _teacher_shard_files  # noqa: E402
from build_memmap_corpus import build_memmap_corpus  # noqa: E402

ANCHOR_MANIFEST_SCHEMA = "anchor_manifest_v1"
ANCHOR_TYPES = ("current_window", "longitudinal", "external_hard")

# --hard-anchor tag schema (STUB, CAT-26 R9). These column names are RESERVED
# for when provenance tagging lands in the self-play/curation pipeline; no
# generator populates them yet. Documenting the schema now so the eventual
# tagging work has an agreed contract, and so --hard-anchor fails loudly
# (not silently-empty) against today's untagged shards.
HARD_ANCHOR_TAG_SCHEMA: dict[str, str] = {
    "outcome_vs_external": (
        "string column, one value per row (repeated per game, like game_seed). "
        "e.g. 'loss_vs_catanatron_value' for every row of a game we lost to that "
        "external opponent; '' otherwise."
    ),
    "is_high_regret_opening": (
        "bool column, one value per row. True for decisions in openings flagged "
        "high-regret by tools/opening_panel.py's regret scoring."
    ),
    "failure_mode": (
        "string column, one value per row. One of {'robber', 'dev_card', "
        "'discard', ''} -- the specific weak-phase failure class called out in "
        "the Roadmap, '' for rows with no flagged failure."
    ),
}


def _parse_seed_ranges(raw: str) -> list[tuple[int, int]]:
    """Parse "start1:end1,start2:end2" (inclusive bounds -- BOTH start and end
    are included in the range). Mirrors train_bc._parse_game_seed_ranges's
    format for consistency across tools.

    CAUTION (off-by-one vs. the seed ledger's own notation): the seed ledger /
    Roadmap Sec 1 documents ranges HALF-OPEN, e.g. the VAL-ONLY band is written
    "[6.19B, 6.2B)" -- meaning 6_200_000_000 itself is NOT part of that range.
    This function's bounds are inclusive-inclusive, so passing the ledger's
    literal upper-bound number verbatim (``...:6200000000``) would incorrectly
    include seed 6_200_000_000. Translate before use: subtract 1 from the
    ledger's half-open upper bound (``...:6199999999``), or otherwise confirm
    the ledger's true inclusive/exclusive convention for the specific range
    you are consuming before passing it to --seed-ranges/--exclude-seed-ranges.
    """
    ranges: list[tuple[int, int]] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise SystemExit(f"invalid seed range entry {chunk!r}: expected start:end")
        start_str, end_str = chunk.split(":", 1)
        start, end = int(start_str), int(end_str)
        if end < start:
            raise SystemExit(f"invalid seed range entry {chunk!r}: end < start")
        ranges.append((start, end))
    return ranges


def _seed_mask(seeds: np.ndarray, ranges: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(seeds.shape[0], dtype=bool)
    for start, end in ranges:
        mask |= (seeds >= start) & (seeds <= end)
    return mask


def ranges_overlap(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> bool:
    """True if any inclusive range in ``a`` overlaps any inclusive range in ``b``."""
    for a_lo, a_hi in a:
        for b_lo, b_hi in b:
            if a_lo <= b_hi and b_lo <= a_hi:
                return True
    return False


def filter_shards_by_seed_ranges(
    shard_paths: list[Path],
    seed_ranges: list[tuple[int, int]],
    staging_dir: Path,
) -> dict:
    """Read+normalize each shard (train_bc's own normalization), keep only rows
    whose ``game_seed`` falls in ``seed_ranges``, write filtered shards into
    ``staging_dir`` as ordinary .npz files + a manifest.json build_memmap_corpus
    can consume directly. Returns extraction stats."""
    staging_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    total_rows_in = 0
    total_rows_out = 0
    seen_seeds: set[int] = set()
    for idx, path in enumerate(shard_paths):
        norm = _normalize_teacher_shard(_load_npz(path), path)
        if "game_seed" not in norm:
            raise SystemExit(
                f"{path}: no game_seed column -- cannot safely extract a seed-range "
                "anchor from a shard without game_seed (CAT-52 AUDIT.md documents "
                "this exact hazard for the training split path; the same hazard "
                "applies here: without game_seed there is no way to verify the "
                "extracted rows are actually the reserved .valonly games)."
            )
        seeds = np.asarray(norm["game_seed"], dtype=np.int64)
        total_rows_in += len(seeds)
        mask = _seed_mask(seeds, seed_ranges)
        if not mask.any():
            continue
        filtered = {key: value[mask] for key, value in norm.items()}
        out_path = staging_dir / f"anchor_shard_{idx:06d}.npz"
        np.savez(out_path, **filtered)
        written.append(str(out_path))
        total_rows_out += int(mask.sum())
        seen_seeds.update(int(s) for s in np.unique(seeds[mask]).tolist())
    if not written:
        raise SystemExit(
            f"no rows matched seed_ranges={seed_ranges} across {len(shard_paths)} "
            "source shard(s) -- refusing to build an empty anchor corpus."
        )
    (staging_dir / "manifest.json").write_text(
        json.dumps({"shards": written, "rows": total_rows_out})
    )
    return {
        "rows_in": total_rows_in,
        "rows_out": total_rows_out,
        "shards_written": len(written),
        "seeds": sorted(seen_seeds),
    }


def filter_shards_by_tags(
    shard_paths: list[Path],
    tag_column: str,
    tag_values: list[str],
    staging_dir: Path,
) -> dict:
    """--hard-anchor row filter: keep rows whose ``tag_column`` value is in
    ``tag_values``. Raises a clear "not yet populated" error if no source
    shard has ``tag_column`` at all (the expected state today, per
    HARD_ANCHOR_TAG_SCHEMA's docstring) rather than silently producing an
    empty anchor.

    Reads ``tag_column`` from the RAW (pre-normalization) npz, not the
    normalized dict: ``train_bc._normalize_teacher_shard`` builds its output
    from a fixed, known column whitelist and silently drops anything outside
    it -- a provenance tag column would never survive normalization even
    after a generator starts writing one, so this must look at the raw
    shard directly to see it at all. The row mask derived from the raw
    column is then applied to the *normalized* dict for the actual filtered
    output (so the written columns still match the training schema)."""
    staging_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    total_rows_in = 0
    total_rows_out = 0
    any_shard_has_column = False
    for idx, path in enumerate(shard_paths):
        raw = _load_npz(path)
        norm = _normalize_teacher_shard(raw, path)
        total_rows_in += len(norm.get("action_taken", []))
        if tag_column not in raw:
            continue
        any_shard_has_column = True
        values = np.asarray(raw[tag_column]).astype(str)
        mask = np.isin(values, np.asarray(tag_values, dtype=str))
        if not mask.any():
            continue
        filtered = {key: value[mask] for key, value in norm.items()}
        out_path = staging_dir / f"hard_anchor_shard_{idx:06d}.npz"
        np.savez(out_path, **filtered)
        written.append(str(out_path))
        total_rows_out += int(mask.sum())
    if not any_shard_has_column:
        raise SystemExit(
            f"--hard-anchor-tag-column {tag_column!r} is not present in ANY of the "
            f"{len(shard_paths)} source shard(s). This is expected: provenance "
            "tagging for the external-hard anchor (CAT-26 R9 #14) is a documented "
            "STUB -- see HARD_ANCHOR_TAG_SCHEMA in this file's module docstring -- "
            "no generator populates it yet. This is not a bug in this script; "
            "wire the tagging producer first."
        )
    if not written:
        raise SystemExit(
            f"tag_column={tag_column!r} present but no row matched "
            f"tag_values={tag_values} across {len(shard_paths)} source shard(s)."
        )
    (staging_dir / "manifest.json").write_text(
        json.dumps({"shards": written, "rows": total_rows_out})
    )
    return {"rows_in": total_rows_in, "rows_out": total_rows_out, "shards_written": len(written)}


def _extracted_game_seeds(corpus_dir: Path) -> np.ndarray:
    """Read back the game_seed column of a just-built memmap corpus, for the
    self-consistency / no-overlap verification below (never trust the filter
    step alone -- verify what was actually written to disk)."""
    meta = json.loads((corpus_dir / "corpus_meta.json").read_text())
    if not meta.get("game_seed_present", False):
        raise SystemExit(
            f"{corpus_dir}: built anchor corpus has no game_seed column -- cannot "
            "verify seed-range containment. This should be unreachable (the "
            "filter step above already requires game_seed); if you see this, the "
            "source schema changed underneath this script."
        )
    row_count = int(meta["row_count"])
    return np.fromfile(corpus_dir / "game_seed.dat", dtype=np.int64, count=row_count)


def verify_anchor_corpus(
    corpus_dir: Path,
    seed_ranges: list[tuple[int, int]] | None,
    exclude_seed_ranges: list[tuple[int, int]] | None,
) -> dict:
    """Self-consistency check on the just-built corpus (not the staging
    filter step): every extracted game_seed must fall inside seed_ranges (if
    given), and NONE may fall inside exclude_seed_ranges (the caller's
    training-window ranges) -- the "never overlaps a training window"
    standing rule from the seed ledger, checked mechanically rather than
    trusted by convention."""
    seeds = _extracted_game_seeds(corpus_dir)
    result = {"row_count": int(len(seeds)), "unique_game_seeds": int(np.unique(seeds).size)}
    if seed_ranges:
        contained = _seed_mask(seeds, seed_ranges)
        if not bool(np.all(contained)):
            bad = seeds[~contained]
            raise SystemExit(
                f"{corpus_dir}: {len(bad)} extracted row(s) have game_seed outside "
                f"the declared seed_ranges={seed_ranges} (e.g. {sorted(set(bad.tolist()))[:5]}). "
                "Refusing to publish an anchor whose contents don't match its own manifest."
            )
        result["seed_ranges_verified"] = True
    if exclude_seed_ranges:
        overlap = _seed_mask(seeds, exclude_seed_ranges)
        if bool(np.any(overlap)):
            bad = seeds[overlap]
            raise SystemExit(
                f"{corpus_dir}: {len(bad)} extracted row(s) have game_seed inside "
                f"the excluded (training-window) ranges={exclude_seed_ranges} "
                f"(e.g. {sorted(set(bad.tolist()))[:5]}). This anchor would overlap "
                "a training window -- refusing to publish (seed ledger standing rule)."
            )
        result["no_train_overlap_verified"] = True
    return result


def _atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    tmp.replace(path)


def load_anchor_manifest(manifest_path: Path) -> dict:
    if manifest_path.exists():
        data = json.loads(manifest_path.read_text())
        sv = data.get("schema")
        if sv != ANCHOR_MANIFEST_SCHEMA:
            raise SystemExit(
                f"{manifest_path}: schema {sv!r} != {ANCHOR_MANIFEST_SCHEMA!r}; migrate explicitly"
            )
        return data
    return {"schema": ANCHOR_MANIFEST_SCHEMA, "anchors": []}


def append_anchor_manifest_entry(manifest_path: Path, entry: dict, *, force: bool) -> dict:
    """Append-only longitudinal series (CAT-30 step 4): a given anchor name is
    recorded exactly once unless ``force`` explicitly replaces its entry (the
    corpus files themselves are never silently overwritten either -- see
    ``main()``'s ``--force`` handling)."""
    manifest = load_anchor_manifest(manifest_path)
    existing = [a for a in manifest["anchors"] if a["name"] == entry["name"]]
    if existing and not force:
        raise SystemExit(
            f"anchor {entry['name']!r} already exists in {manifest_path} "
            f"(built {existing[0].get('created_at')}). Anchors are a permanent "
            "longitudinal series and are not overwritten by default -- pass "
            "--force if you intentionally want to replace this entry."
        )
    manifest["anchors"] = [a for a in manifest["anchors"] if a["name"] != entry["name"]]
    manifest["anchors"].append(entry)
    _atomic_write_json(manifest_path, manifest)
    return manifest


def build_anchor_corpus(
    *,
    source_roots: list[Path],
    anchor_name: str,
    anchor_type: str,
    out_root: Path,
    seed_ranges: list[tuple[int, int]] | None,
    exclude_seed_ranges: list[tuple[int, int]] | None,
    hard_anchor_tag_column: str | None,
    hard_anchor_tag_values: list[str] | None,
    force: bool,
    abort_on_duplicate_seeds: bool = True,
) -> dict:
    if anchor_type not in ANCHOR_TYPES:
        raise SystemExit(f"anchor_type must be one of {ANCHOR_TYPES}, got {anchor_type!r}")
    if hard_anchor_tag_column and seed_ranges:
        raise SystemExit("pass either seed_ranges or a --hard-anchor tag filter, not both")
    if not hard_anchor_tag_column and not seed_ranges:
        raise SystemExit("must pass --seed-ranges, or --hard-anchor-tag-column + --hard-anchor-tag-values")

    corpus_dir = out_root / anchor_name
    if corpus_dir.exists() and not force:
        raise SystemExit(
            f"{corpus_dir} already exists. Anchors are a permanent longitudinal "
            "series (CAT-30) and are not overwritten by default -- pass --force "
            "if you intentionally want to rebuild this exact anchor."
        )

    shard_paths: list[Path] = []
    for root in source_roots:
        shard_paths.extend(_teacher_shard_files(root))
    if not shard_paths:
        raise SystemExit(f"no teacher shards found under {source_roots}")

    staging_dir = out_root / f".staging_{anchor_name}"
    started = time.perf_counter()
    if hard_anchor_tag_column:
        filter_stats = filter_shards_by_tags(
            shard_paths, hard_anchor_tag_column, hard_anchor_tag_values or [], staging_dir
        )
    else:
        filter_stats = filter_shards_by_seed_ranges(shard_paths, seed_ranges or [], staging_dir)

    meta = build_memmap_corpus(
        staging_dir, corpus_dir, abort_on_duplicate_seeds=abort_on_duplicate_seeds
    )
    verify_stats = verify_anchor_corpus(corpus_dir, seed_ranges, exclude_seed_ranges)

    import shutil
    shutil.rmtree(staging_dir, ignore_errors=True)

    entry = {
        "name": anchor_name,
        "anchor_type": anchor_type,
        "seed_ranges": seed_ranges,
        "hard_anchor_tag_column": hard_anchor_tag_column,
        "hard_anchor_tag_values": hard_anchor_tag_values,
        "corpus_path": str(corpus_dir),
        "sources": [str(root) for root in source_roots],
        "row_count": meta["row_count"],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "build_seconds": round(time.perf_counter() - started, 2),
        "filter_stats": {k: v for k, v in filter_stats.items() if k != "seeds"},
        "verify_stats": verify_stats,
        # TRIPWIRE-ONLY marker: consumers of anchor telemetry (continuous_flywheel.py)
        # must read this field and never gate a promotion decision on this anchor's
        # results -- see FlywheelConfig's anchor_* fields and this module's docstring.
        "promotion_signal": False,
    }
    manifest_path = out_root / "anchor_manifest.json"
    append_anchor_manifest_entry(manifest_path, entry, force=force)
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source", required=True, type=Path, nargs="+",
        help="One or more teacher shard roots (each a dir with manifest.json) to extract from.",
    )
    parser.add_argument(
        "--anchor-name", required=True,
        help="Name for this anchor, e.g. anchor_gen4. Must be new (or pass --force).",
    )
    parser.add_argument(
        "--anchor-type", choices=ANCHOR_TYPES, default="current_window",
        help="current_window (freshest, e.g. anchor_gen4) | longitudinal (a permanently-"
        "kept older anchor, e.g. anchor_r7) | external_hard (STUB, R9 #14).",
    )
    parser.add_argument(
        "--out-root", required=True, type=Path,
        help="Root directory holding all anchor corpora + the longitudinal anchor_manifest.json.",
    )
    parser.add_argument(
        "--seed-ranges", default="",
        help="Comma-separated start:end (BOTH bounds inclusive) .valonly game_seed ranges "
        "-- consult the seed ledger before picking any range (standing rule, not "
        "optional). NOTE: the ledger documents ranges half-open (e.g. VAL-ONLY = "
        "'[6.19B, 6.2B)'); subtract 1 from the ledger's upper bound before passing it "
        "here, or you will include one seed the ledger did not reserve. Mutually "
        "exclusive with --hard-anchor-tag-column.",
    )
    parser.add_argument(
        "--exclude-seed-ranges", default="",
        help="Comma-separated start:end (BOTH bounds inclusive) TRAINING-WINDOW game_seed "
        "ranges -- same half-open-ledger-to-inclusive-flag translation caveat as "
        "--seed-ranges applies here too. If given, the built anchor is verified to "
        "have zero rows in these ranges (never-overlaps-a-training-window guarantee, "
        "checked mechanically).",
    )
    parser.add_argument(
        "--hard-anchor-tag-column", default=None,
        help="STUB (CAT-26 R9): filter by a provenance tag column instead of seed ranges. "
        "See HARD_ANCHOR_TAG_SCHEMA in this file for the documented (not-yet-populated) schema.",
    )
    parser.add_argument(
        "--hard-anchor-tag-values", default="",
        help="Comma-separated tag values to keep (used with --hard-anchor-tag-column).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Allow rebuilding/overwriting an existing anchor of this exact name. Anchors "
        "are a permanent longitudinal series by default -- this is an explicit escape hatch.",
    )
    parser.add_argument(
        "--abort-on-duplicate-seeds", action=argparse.BooleanOptionalAction, default=True,
        help="Passed through to build_memmap_corpus (task #77 seed-collision guard).",
    )
    args = parser.parse_args()

    seed_ranges = _parse_seed_ranges(args.seed_ranges) if args.seed_ranges else None
    exclude_seed_ranges = _parse_seed_ranges(args.exclude_seed_ranges) if args.exclude_seed_ranges else None
    if seed_ranges and exclude_seed_ranges and ranges_overlap(seed_ranges, exclude_seed_ranges):
        raise SystemExit(
            f"--seed-ranges {seed_ranges} overlaps --exclude-seed-ranges {exclude_seed_ranges} "
            "-- these are supposed to be disjoint (anchor ranges vs. training-window ranges) "
            "by construction; refusing before even touching data."
        )
    tag_values = [v.strip() for v in args.hard_anchor_tag_values.split(",") if v.strip()]

    entry = build_anchor_corpus(
        source_roots=args.source,
        anchor_name=args.anchor_name,
        anchor_type=args.anchor_type,
        out_root=args.out_root,
        seed_ranges=seed_ranges,
        exclude_seed_ranges=exclude_seed_ranges,
        hard_anchor_tag_column=args.hard_anchor_tag_column,
        hard_anchor_tag_values=tag_values or None,
        force=args.force,
        abort_on_duplicate_seeds=args.abort_on_duplicate_seeds,
    )
    print(json.dumps({"progress": "anchor_build_done", **entry}, sort_keys=True, default=str), flush=True)


if __name__ == "__main__":
    main()
