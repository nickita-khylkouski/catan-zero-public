#!/usr/bin/env python3
"""Convert npz teacher shards into a flat memmap corpus for streaming training.

``tools/train_bc.py``'s ``load_teacher_data`` materialises the whole corpus in
host RAM: it accumulates per-column lists, ``np.concatenate``s them (a transient
2x spike), and pads every ragged per-decision column (``legal_action_ids`` and
friends) to the global maximum legal width (54). Mean legal width in the
raw-selfplay corpus is ~4.8, so >90% of the ragged storage is padding, and the
whole set has to be resident at once. That ceiling OOM'd a 32.6M-row corpus on a
708GB host.

This tool performs the one-time conversion into a directory of flat files that
``MemmapCorpus`` (see ``train_bc.py``) streams per batch:

* Fixed-width columns (``obs``, board/entity tokens, scalars, VP arrays) are
  written as flat ``<col>.dat`` files, one row after another -- reloaded as an
  ``(N, *inner_shape)`` ``np.memmap``.
* Ragged per-decision columns are stored TRIMMED to each row's true legal count
  (no padding on disk) in flat ``<col>.dat`` value files, sharing a single
  ``row_offsets.dat`` (``int64``, ``N+1``). The batch collate re-pads them to the
  global legal width so batches are byte-identical to the in-RAM loader.
* Unicode columns are factorised into an ``int32`` ``<col>.codes.dat`` plus a
  category list in ``corpus_meta.json``.

The normalisation (dtypes, defaults, string coercion, schema checks) is reused
verbatim from ``train_bc._normalize_teacher_shard`` so the reconstructed batches
match ``load_teacher_data`` exactly. Trimming is lossless only because every
ragged column is exactly its fill value beyond the per-row legal count; the
converter asserts this per shard and aborts if a shard ever violates it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from train_bc import (  # noqa: E402  (sibling module bootstrap above)
    _load_npz,
    _normalize_teacher_shard,
    _teacher_shard_files,
)

MEMMAP_CORPUS_SCHEMA = "memmap_corpus_v1"

# The exact column set (and order) load_teacher_data keeps in its local ``keys``
# tuple. Anything not present in a normalised shard is simply skipped, matching
# load_teacher_data's ``if key in shard`` guard.
LOADER_KEYS: tuple[str, ...] = (
    "obs",
    "legal_action_ids",
    "legal_action_context",
    "action_taken",
    "target_policy",
    "prior_policy",
    "target_scores",
    "target_policy_mask",
    "target_scores_mask",
    "target_score_source",
    "game_seed",
    "teacher_name",
    "player",
    "seat",
    "phase",
    "decision_index",
    "action_mask_version",
    "winner",
    "terminated",
    "truncated",
    "final_public_vps",
    "has_final_public_vps",
    "final_actual_vps",
    "has_final_actual_vps",
    "policy_weight_multiplier",
    "value_weight_multiplier",
    "hex_tokens",
    "hex_vertex_ids",
    "hex_edge_ids",
    "vertex_tokens",
    "edge_tokens",
    "edge_vertex_ids",
    "player_tokens",
    "global_tokens",
    "legal_action_tokens",
    "legal_action_target_ids",
    "event_tokens",
    "event_target_ids",
    "hex_mask",
    "vertex_mask",
    "edge_mask",
    "player_mask",
    "legal_action_mask",
    "event_mask",
)

# Columns padded on the legal-action axis by load_teacher_data._concat_padded,
# and the fill value used there. These are stored ragged (trimmed) on disk.
RAGGED_FILLS: dict[str, float] = {
    "legal_action_ids": -1.0,
    "target_policy": 0.0,
    "prior_policy": 0.0,
    "target_scores": float("nan"),
    "target_policy_mask": 0.0,  # False
    "target_scores_mask": 0.0,  # False
    "legal_action_mask": 0.0,  # False
    "legal_action_context": 0.0,
    "legal_action_tokens": 0.0,
    "legal_action_target_ids": -1.0,
}


def _fill_matches(values: np.ndarray, fill: float) -> np.ndarray:
    """Elementwise "is this the pad fill" test, treating NaN fill specially."""
    if np.isnan(fill):
        return np.isnan(values)
    return values == fill


def _classify(name: str, array: np.ndarray) -> dict:
    """Return the on-disk schema record for a normalised column."""
    if array.dtype.kind == "U":
        return {"kind": "string"}
    if name in RAGGED_FILLS:
        if array.ndim == 2:
            return {
                "kind": "ragged2d",
                "dtype": array.dtype.str,
                "fill": RAGGED_FILLS[name],
            }
        if array.ndim == 3:
            return {
                "kind": "ragged3d",
                "dtype": array.dtype.str,
                "feat": int(array.shape[2]),
                "fill": RAGGED_FILLS[name],
            }
        raise SystemExit(f"ragged column {name} has unexpected ndim {array.ndim}")
    return {
        "kind": "fixed",
        "dtype": array.dtype.str,
        "inner_shape": [int(d) for d in array.shape[1:]],
    }


class _GameSeedRunTracker:
    """Tracks maximal contiguous runs of equal ``game_seed`` values across the
    whole corpus (all shards, in order), flagging a value as duplicated the
    moment a genuinely NEW, non-contiguous run starts with a value that has
    already started a run before.

    ``game_seed`` is one value per GAME, repeated across every decision row of
    that game -- not a per-row identity. Seeds are globally disjoint by
    design, so a seed value starting a SECOND, non-contiguous run anywhere in
    the corpus indicates a collision (the class task #77 nearly missed).

    A naive per-shard unique-value set has a false-positive trap:
    GumbelShardWriter flushes shards purely by ROW COUNT
    (``if len(self.rows) >= self.shard_size: self.flush()``), not by game
    boundary, so a game in progress at a shard's end routinely continues with
    the same game_seed at the very start of the next shard -- one game split
    across two files, not a duplicate. This tracker merges a shard's leading
    run into the previous shard's still-open trailing run when they share a
    value, treating the whole corpus as one contiguous stream of runs.

    Earlier revision of this logic (fixed here) deferred registering an
    open/merged run until it was explicitly "closed" by a later, differing
    value. That meant a run continuation-merged across a shard boundary
    (e.g. shard N-1 ends with seed S, shard N opens with seed S) was never
    added to the seen-set while it stayed the open/pending run -- so if S
    reappeared LATER in that same shard N as a second, non-contiguous run
    (a same-shard duplicate), the reappearance became the new pending run
    directly and bypassed the seen-set check entirely, escaping detection.
    This tracker instead registers a value into the seen-set the instant its
    run *starts* (whether merged-open or freshly closed), so any later run
    of an already-registered value is caught regardless of whether the
    earlier run was ever formally closed.
    """

    def __init__(self) -> None:
        self._seen: set[int] = set()
        self._duplicates: set[int] = set()
        self._current: int | None = None

    def _start_run(self, value: int) -> None:
        if value in self._seen:
            self._duplicates.add(value)
        else:
            self._seen.add(value)
        self._current = value

    def observe_shard(self, seed_column: np.ndarray) -> None:
        seed_col = np.asarray(seed_column).reshape(-1)
        if not seed_col.size:
            return
        run_starts = np.concatenate(([0], np.flatnonzero(np.diff(seed_col) != 0) + 1))
        run_values = seed_col[run_starts]
        for value in run_values:
            value = int(value)
            if value == self._current:
                continue  # merges into the still-open run (shard boundary or not)
            self._start_run(value)

    @property
    def duplicate_count(self) -> int:
        return len(self._duplicates)

    @property
    def has_duplicates(self) -> bool:
        return bool(self._duplicates)


def build_memmap_corpus(
    source: Path | str | Sequence[Path | str],
    out_dir: Path,
    *,
    max_shards: int | None = None,
    verify_fill: bool = True,
    progress_every: int = 500,
    abort_on_duplicate_seeds: bool = True,
    full_rows_only: bool = False,
) -> dict:
    """Stream one or more sources' npz shards into a flat memmap corpus.

    ``source`` may be a single teacher-shard root or a sequence of them (e.g.
    tranche-1 combined + tranche-2). Shards are concatenated in source order into
    one corpus; every shard across all sources must share the same column schema
    (enforced per shard), and the global legal width, string categories and row
    offsets span the whole set. Returns the written ``corpus_meta.json`` payload.
    """
    sources = [source] if isinstance(source, (str, Path)) else list(source)
    files: list[Path] = []
    for src in sources:
        src_files = _teacher_shard_files(Path(src))
        if not src_files:
            raise SystemExit(f"no teacher shards found in {src}")
        files.extend(src_files)
    if max_shards is not None:
        files = files[:max_shards]
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    first = _normalize_teacher_shard(_load_npz(files[0]), files[0])
    columns = [key for key in LOADER_KEYS if key in first]
    schemas = {name: _classify(name, first[name]) for name in columns}
    column_set = set(columns)

    handles = {name: open(out_dir / f"{name}.dat", "wb") for name in columns if schemas[name]["kind"] != "string"}
    code_handles = {name: open(out_dir / f"{name}.codes.dat", "wb") for name in columns if schemas[name]["kind"] == "string"}
    # Global string factorisation: category -> code, stable in first-seen order.
    categories: dict[str, dict[str, int]] = {name: {} for name in code_handles}
    category_lists: dict[str, list[str]] = {name: [] for name in code_handles}

    row_lengths: list[np.ndarray] = []
    row_count = 0
    flat_count = 0
    legal_width = 0
    stats = {
        "max_legal_action_id": -1,
        "action_taken_min": None,
        "action_taken_max": None,
        "has_duplicate_legal_rows": False,
        "duplicate_game_seed_count": 0,
        "has_duplicate_game_seeds": False,
    }
    # See _GameSeedRunTracker's docstring for the duplicate-detection contract.
    _seed_tracker = _GameSeedRunTracker()

    dropped_fast_rows = 0
    for shard_index, file in enumerate(files):
        raw = _load_npz(file)
        norm = _normalize_teacher_shard(raw, file)
        if full_rows_only:
            # Keep only FULL-search rows (drop fast rows). used_full_search is the
            # ground-truth per-row full/fast marker written by the generator; it is
            # NOT part of LOADER_KEYS (so it never lands in the memmap), so read it
            # from the raw shard here. Forced rows report used_full_search=True (they
            # pay a full enumeration and carry value signal) and are KEPT -- only
            # fast-search rows are dropped. policy_weight_multiplier already zeroes
            # fast+forced rows out of POLICY loss at train time; this filter is the
            # physical-drop variant for building a fast-free corpus (e.g. the
            # pure-teacher ablation arm).
            if "used_full_search" not in raw:
                raise SystemExit(
                    f"{file}: --full-rows-only requires a 'used_full_search' column, "
                    "but this shard has none (pre-marker generation?). Rebuild the "
                    "shards with the current generator or drop --full-rows-only."
                )
            keep = np.asarray(raw["used_full_search"]).astype(bool)
            if keep.shape[0] != int(np.asarray(norm["action_taken"]).shape[0]):
                raise SystemExit(
                    f"{file}: used_full_search length {keep.shape[0]} != row count "
                    f"{int(np.asarray(norm['action_taken']).shape[0])}"
                )
            dropped_fast_rows += int((~keep).sum())
            if not keep.all():
                norm = {name: np.asarray(value)[keep] for name, value in norm.items()}
        present = {key for key in LOADER_KEYS if key in norm}
        if present != column_set:
            raise SystemExit(
                f"{file} column set differs from first shard; refusing to mix schemas. "
                f"missing={sorted(column_set - present)} extra={sorted(present - column_set)}"
            )
        # Re-validate every column's dtype and inner shape against the schema
        # recorded from shard 0. The raw bytes are appended with tofile(), so a
        # shard whose dtype or feature width drifted would not crash here -- it
        # would silently misalign EVERY subsequent row when the flat file is
        # reinterpreted by np.memmap at load time. Fail loudly instead.
        for name in columns:
            schema = schemas[name]
            array = norm[name]
            kind = schema["kind"]
            if kind == "string":
                continue
            if array.dtype.str != schema["dtype"]:
                raise SystemExit(
                    f"{file}: column {name!r} dtype {array.dtype.str} != first shard's "
                    f"{schema['dtype']}; mixed dtypes would corrupt the flat memmap."
                )
            if kind == "fixed":
                inner = [int(d) for d in array.shape[1:]]
                if inner != list(schema["inner_shape"]):
                    raise SystemExit(
                        f"{file}: column {name!r} inner shape {inner} != first shard's "
                        f"{list(schema['inner_shape'])}; mixed widths would corrupt the flat memmap."
                    )
            elif kind == "ragged3d":
                if int(array.shape[2]) != int(schema["feat"]):
                    raise SystemExit(
                        f"{file}: column {name!r} feature width {int(array.shape[2])} != "
                        f"first shard's {int(schema['feat'])}; mixed widths would corrupt the flat memmap."
                    )
            elif kind == "ragged2d" and array.ndim != 2:
                raise SystemExit(f"{file}: column {name!r} ndim {array.ndim} != 2")
        legal_ids = norm["legal_action_ids"]
        width = int(legal_ids.shape[1])
        legal_width = max(legal_width, width)
        counts = np.sum(legal_ids >= 0, axis=1).astype(np.int64)
        n = int(legal_ids.shape[0])
        prefix_mask = np.arange(width)[None, :] < counts[:, None]

        # Trimming is lossless only if the valid legal entries are a contiguous
        # prefix (guaranteed by legal_action_mask == legal_action_ids>=0) and
        # everything past the count is exactly the pad fill. Verify per shard so
        # a schema drift aborts the conversion instead of silently dropping data.
        if verify_fill:
            if "legal_action_mask" in norm and not np.array_equal(
                norm["legal_action_mask"], legal_ids >= 0
            ):
                raise SystemExit(f"{file}: legal_action_mask != (legal_action_ids>=0)")
            tail_mask = ~prefix_mask
            for name in columns:
                if name not in RAGGED_FILLS:
                    continue
                tail = norm[name][tail_mask]
                if tail.size and not np.all(_fill_matches(tail, RAGGED_FILLS[name])):
                    raise SystemExit(
                        f"{file}: column {name!r} has non-fill values beyond the legal "
                        "count; per-row trimming would lose data. Regenerate the shard "
                        "or extend build_memmap_corpus to store it at full width."
                    )

        # Cheap corpus-wide validation stats (mirror validate_teacher_data_schema)
        valid_legal = legal_ids[legal_ids >= 0]
        if valid_legal.size:
            stats["max_legal_action_id"] = max(stats["max_legal_action_id"], int(valid_legal.max()))
        actions = norm["action_taken"]
        if actions.size:
            amin, amax = int(actions.min()), int(actions.max())
            stats["action_taken_min"] = amin if stats["action_taken_min"] is None else min(stats["action_taken_min"], amin)
            stats["action_taken_max"] = amax if stats["action_taken_max"] is None else max(stats["action_taken_max"], amax)
        if not stats["has_duplicate_legal_rows"]:
            # A duplicate legal id within a row shows up as adjacent equal values
            # (both >= 0) once each row is sorted; -1 pads sort to the front.
            row_sorted = np.sort(legal_ids, axis=1)
            adjacent_equal = (row_sorted[:, 1:] == row_sorted[:, :-1]) & (row_sorted[:, 1:] >= 0)
            if bool(np.any(adjacent_equal)):
                stats["has_duplicate_legal_rows"] = True
        if "game_seed" in norm:
            _seed_tracker.observe_shard(norm["game_seed"])

        for name in columns:
            schema = schemas[name]
            array = norm[name]
            kind = schema["kind"]
            if kind == "string":
                catmap = categories[name]
                catlist = category_lists[name]
                uniq, inverse = np.unique(array, return_inverse=True)
                mapped = np.empty(uniq.shape[0], dtype=np.int32)
                for u_index, value in enumerate(uniq):
                    text = str(value)
                    code = catmap.get(text)
                    if code is None:
                        code = len(catlist)
                        catmap[text] = code
                        catlist.append(text)
                    mapped[u_index] = code
                codes = mapped[inverse].astype(np.int32, copy=False)
                code_handles[name].write(np.ascontiguousarray(codes).tobytes())
            elif kind == "fixed":
                np.ascontiguousarray(array).tofile(handles[name])
            else:  # ragged2d / ragged3d
                flat = array[prefix_mask]  # row-major prefix concat -> (sum counts, [feat])
                np.ascontiguousarray(flat).tofile(handles[name])

        row_lengths.append(counts)
        row_count += n
        flat_count += int(counts.sum())
        if progress_every and (shard_index + 1) % progress_every == 0:
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "progress": "memmap_convert",
                        "shards_done": shard_index + 1,
                        "shards_total": len(files),
                        "rows": row_count,
                        "elapsed_s": round(elapsed, 1),
                        "shards_per_s": round((shard_index + 1) / max(elapsed, 1e-9), 2),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    for handle in handles.values():
        handle.close()
    for handle in code_handles.values():
        handle.close()

    stats["duplicate_game_seed_count"] = _seed_tracker.duplicate_count
    stats["has_duplicate_game_seeds"] = _seed_tracker.has_duplicates
    if stats["has_duplicate_game_seeds"]:
        message = (
            f"{stats['duplicate_game_seed_count']} game_seed value(s) recur "
            "as a SEPARATE, non-contiguous game elsewhere in this corpus -- games are "
            "supposed to have globally disjoint seeds, so this indicates duplicated "
            "games (the seed-collision class from task #77) silently doubling their "
            "weight in training."
        )
        if abort_on_duplicate_seeds:
            raise SystemExit(
                f"ABORTING: {message} Investigate the source shards (or re-run with "
                "--no-abort-on-duplicate-seeds to only warn and proceed at your own risk)."
            )
        print(
            f"WARNING: {message} NOT aborting the conversion (--no-abort-on-duplicate-seeds "
            "was set); the operator should investigate corpus_meta.json's "
            "stats.duplicate_game_seed_count before training on this corpus.",
            file=sys.stderr,
        )

    lengths = np.concatenate(row_lengths) if row_lengths else np.zeros(0, dtype=np.int64)
    offsets = np.empty(row_count + 1, dtype=np.int64)
    offsets[0] = 0
    if lengths.size:
        np.cumsum(lengths, out=offsets[1:])
    offsets.tofile(out_dir / "row_offsets.dat")

    for name in code_handles:
        schemas[name] = {"kind": "string", "categories": category_lists[name]}

    meta = {
        "schema": MEMMAP_CORPUS_SCHEMA,
        "row_count": int(row_count),
        "flat_count": int(flat_count),
        "legal_width": int(legal_width),
        "source": str(sources[0]),
        "sources": [str(src) for src in sources],
        "shard_count": len(files),
        "columns": schemas,
        "game_seed_present": "game_seed" in column_set,
        # --full-rows-only provenance: whether fast rows were physically dropped,
        # and how many. False + 0 for a normal (pooled) build.
        "full_rows_only": bool(full_rows_only),
        "dropped_fast_rows": int(dropped_fast_rows),
        # Whether the lossless-trim guarantee (ragged tails are exactly pad
        # fill) was actually VERIFIED for this corpus. A corpus built with
        # --no-verify-fill is otherwise indistinguishable from a verified one.
        "verify_fill": bool(verify_fill),
        "stats": stats,
        "conversion_seconds": round(time.perf_counter() - started, 2),
    }
    (out_dir / "corpus_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"progress": "memmap_convert_done", **{k: meta[k] for k in ("row_count", "flat_count", "legal_width", "shard_count", "conversion_seconds")}}, sort_keys=True), flush=True)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        nargs="+",
        help=(
            "One or more teacher shard roots (each a dir with manifest.json); "
            "shards are concatenated in the given order, e.g. "
            "--source runs/raw_selfplay_gen1_combined runs/raw_selfplay_gen2_combined"
        ),
    )
    parser.add_argument("--out", required=True, type=Path, help="output corpus directory")
    parser.add_argument("--max-shards", type=int, default=None, help="convert only the first N shards (slice/estimate)")
    parser.add_argument(
        "--full-rows-only",
        action="store_true",
        help=(
            "Physically DROP fast-search rows (keep rows with used_full_search=True, "
            "including forced-full rows). Builds a fast-free corpus for the "
            "pure-teacher ablation arm. Normal (pooled) builds omit this: fast rows "
            "are kept and already carry policy_weight_multiplier=0, so they train "
            "value only and are excluded from policy loss at train time. Requires the "
            "shards to carry a 'used_full_search' column."
        ),
    )
    parser.add_argument("--no-verify-fill", action="store_true", help="skip the per-shard lossless-trim assertion (faster)")
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument(
        "--abort-on-duplicate-seeds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Hard-exit if a game_seed value starts a second, non-contiguous run "
        "anywhere in the corpus (the seed-collision class from task #77). Default "
        "on; pass --no-abort-on-duplicate-seeds to only warn (via stats."
        "duplicate_game_seed_count in corpus_meta.json) and proceed at your own risk.",
    )
    args = parser.parse_args()
    build_memmap_corpus(
        args.source,
        args.out,
        max_shards=args.max_shards,
        verify_fill=not args.no_verify_fill,
        progress_every=args.progress_every,
        abort_on_duplicate_seeds=args.abort_on_duplicate_seeds,
        full_rows_only=args.full_rows_only,
    )


if __name__ == "__main__":
    main()
