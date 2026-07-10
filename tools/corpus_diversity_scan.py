#!/usr/bin/env python3
"""CLI: corpus diversity scan (CAT-25 measurement 3, mechanism C:
distribution narrowing / self-play inbreeding).

Fully local/CPU, no GPU dependency at all: this tool only reads a directory
of `.npz` teacher shards matching the schema in `tools/build_memmap_corpus.py`
(``LOADER_KEYS``), reusing `find_shard_files` and the generic ragged-column
loading pattern (`load_rows`) from `tools/audit_gumbel_pilot_shards.py`
rather than writing a new loader from scratch.

It has no separate `--dry-run` flag: because it's a pure local file scan, its
unit test builds a small synthetic `.npz` shard on disk (via `np.savez` in a
pytest `tmp_path`) with a handful of fabricated rows covering exactly the
columns this tool reads, and exercises the real scan functions against it --
that IS this tool's dry-run-equivalent.

Computes, for ONE generation (one `--shards-dir` per invocation -- a
multi-generation sweep is just repeated invocations, not a built-in
abstraction here):

  1. unique-state fraction, two variants:
     (a) "cheap" dedup: fraction of rows whose (game_seed, decision_index)
         pair is unique across the whole corpus -- catches literal
         duplicate-row bugs (e.g. the base-seed-collision class this project
         has hit before).
     (b) "content" dedup, only when an `obs` column is present: blake2b hash
         (digest_size=8) of each row's `obs` array bytes, fraction of rows
         with a unique hash. `null` (not fabricated) when `obs` is absent.
  2. opening entropy over decisions in `--decision-range` (default "1,30",
     BOTH ENDS INCLUSIVE: `low <= decision_index <= high`): for rows in that
     window with a `target_policy` (preferred) or `prior_policy` column,
     reconstruct each row's policy dict by zipping `legal_action_ids[row]`
     with the policy column's same row (dropping the `legal_action_ids`
     fill value -1, per `RAGGED_FILLS` in build_memmap_corpus.py), then run
     `diag_common.normalized_entropy` on it. Reports mean/median.
  3. opening-line concentration: group rows by `game_seed`, and for each game
     build a "line" = tuple of `action_taken` values at decision_index
     0..`--line-length`-1 (sorted by decision_index within the game; a game
     with fewer decisions than `--line-length` simply yields a shorter
     tuple). Fed into `diag_common.line_concentration`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from audit_gumbel_pilot_shards import find_shard_files  # noqa: E402
from diag_common import line_concentration, normalized_entropy  # noqa: E402
from factory_common import write_json  # noqa: E402


_SCAN_FIXED_COLUMNS = ("game_seed", "decision_index", "action_taken", "obs")
_SCAN_RAGGED_COLUMNS = ("legal_action_ids", "target_policy", "prior_policy")
_ENTITY_STATE_COLUMNS = (
    "hex_tokens",
    "vertex_tokens",
    "edge_tokens",
    "player_tokens",
    "global_tokens",
    "event_tokens",
    "hex_mask",
    "vertex_mask",
    "edge_mask",
    "player_mask",
    "event_mask",
)


def _pad_ragged_rows(values: list[np.ndarray], width: int, *, fill: float) -> np.ndarray:
    """Pad per-shard ``(rows, legal)`` arrays to one diagnostic width."""
    if not values:
        return np.empty((0, int(width)), dtype=np.float32)
    dtype = values[0].dtype
    padded: list[np.ndarray] = []
    for value in values:
        if int(value.shape[1]) == int(width):
            padded.append(value)
            continue
        out = np.full((int(value.shape[0]), int(width)), fill, dtype=dtype)
        out[:, : int(value.shape[1])] = value
        padded.append(out)
    return np.concatenate(padded, axis=0)


def _load_scan_rows(shard_files: list[Path]) -> dict[str, np.ndarray]:
    """Load only the columns this diagnostic consumes.

    The old implementation called ``audit_gumbel_pilot_shards.load_rows``,
    which materialized every entity-token tensor (tens of GB) even though the
    diversity metrics need only three scalars, policies/legal ids, and ``obs``.
    This selective loader keeps the same NPZ semantics while bounding RAM to
    the diagnostic payload itself.
    """
    columns: dict[str, list[np.ndarray]] = {
        name: [] for name in (*_SCAN_FIXED_COLUMNS, *_SCAN_RAGGED_COLUMNS)
    }
    max_width = 0
    for path in shard_files:
        with np.load(path, allow_pickle=False) as handle:
            for name in _SCAN_FIXED_COLUMNS:
                if name in handle:
                    columns[name].append(np.asarray(handle[name]))
            for name in _SCAN_RAGGED_COLUMNS:
                if name in handle:
                    value = np.asarray(handle[name])
                    if value.ndim != 2:
                        raise ValueError(f"{path}: diagnostic column {name!r} must be 2-D")
                    columns[name].append(value)
                    max_width = max(max_width, int(value.shape[1]))

    rows: dict[str, np.ndarray] = {}
    for name in _SCAN_FIXED_COLUMNS:
        if columns[name]:
            rows[name] = np.concatenate(columns[name], axis=0)
    for name in _SCAN_RAGGED_COLUMNS:
        if columns[name]:
            fill = -1.0 if name == "legal_action_ids" else 0.0
            rows[name] = _pad_ragged_rows(columns[name], max_width, fill=fill)
    return rows


def _memmap_fixed(
    corpus_dir: Path, meta: dict[str, Any], name: str
) -> np.memmap | None:
    schema = (meta.get("columns") or {}).get(name)
    if not schema:
        return None
    if schema.get("kind") != "fixed":
        raise ValueError(f"{corpus_dir}: expected fixed memmap column {name!r}, got {schema}")
    row_count = int(meta["row_count"])
    inner = tuple(int(value) for value in schema.get("inner_shape", ()))
    return np.memmap(
        corpus_dir / f"{name}.dat",
        dtype=np.dtype(schema["dtype"]),
        mode="r",
        shape=(row_count, *inner),
    )


def _memmap_ragged_flat(
    corpus_dir: Path, meta: dict[str, Any], name: str
) -> np.memmap | None:
    schema = (meta.get("columns") or {}).get(name)
    if not schema:
        return None
    if schema.get("kind") != "ragged2d":
        raise ValueError(f"{corpus_dir}: expected ragged2d memmap column {name!r}, got {schema}")
    return np.memmap(
        corpus_dir / f"{name}.dat",
        dtype=np.dtype(schema["dtype"]),
        mode="r",
        shape=(int(meta["flat_count"]),),
    )


def _unique_pair_stats(game_seed: np.ndarray, decision_index: np.ndarray) -> dict[str, Any]:
    n = int(len(game_seed))
    if n == 0:
        return {"rows_total": 0, "unique_pairs": 0, "unique_fraction": None}
    pairs = np.empty(n, dtype=[("game_seed", "<i8"), ("decision_index", "<i4")])
    pairs["game_seed"] = game_seed
    pairs["decision_index"] = decision_index
    unique_pairs = int(np.unique(pairs).size)
    return {
        "rows_total": n,
        "unique_pairs": unique_pairs,
        "unique_fraction": unique_pairs / n,
    }


def _content_hash_stats(
    obs: np.ndarray, *, chunk_rows: int = 8192, representation: str = "obs"
) -> dict[str, Any]:
    n = int(len(obs))
    if n == 0:
        return {"rows_total": 0, "unique_hashes": 0, "unique_fraction": None}
    hashes: set[bytes] = set()
    for start in range(0, n, max(1, int(chunk_rows))):
        chunk = np.asarray(obs[start : start + int(chunk_rows)])
        for row in chunk:
            hashes.add(
                hashlib.blake2b(
                    np.ascontiguousarray(row).tobytes(), digest_size=8
                ).digest()
            )
    unique_hashes = len(hashes)
    return {
        "rows_total": n,
        "unique_hashes": unique_hashes,
        "unique_fraction": unique_hashes / n,
        "hash": "blake2b-64",
        "representation": representation,
    }


def _content_hash_stats_columns(
    columns: dict[str, np.ndarray], *, chunk_rows: int = 2048
) -> dict[str, Any]:
    """Hash the concatenated bytes of fixed entity-token state columns.

    Entity-graph shards keep ``obs`` only as an all-zero compatibility
    placeholder. Falling back to the actual fixed model-input tensors prevents
    that placeholder from masquerading as catastrophic state collapse.
    """
    if not columns:
        raise ValueError("entity-token content hash requires at least one column")
    lengths = {int(len(value)) for value in columns.values()}
    if len(lengths) != 1:
        raise ValueError(f"entity-token columns disagree on row count: {lengths}")
    n = lengths.pop()
    hashes: set[bytes] = set()
    ordered = [(name, columns[name]) for name in _ENTITY_STATE_COLUMNS if name in columns]
    for start in range(0, n, max(1, int(chunk_rows))):
        stop = min(n, start + int(chunk_rows))
        byte_chunks = [
            np.ascontiguousarray(value[start:stop])
            .view(np.uint8)
            .reshape(stop - start, -1)
            for _name, value in ordered
        ]
        combined = np.concatenate(byte_chunks, axis=1)
        for row in combined:
            hashes.add(hashlib.blake2b(row, digest_size=8).digest())
    unique_hashes = len(hashes)
    return {
        "rows_total": n,
        "unique_hashes": unique_hashes,
        "unique_fraction": (unique_hashes / n) if n else None,
        "hash": "blake2b-64",
        "representation": "entity_tokens",
        "columns": [name for name, _value in ordered],
        "note": "obs was an all-zero compatibility placeholder; hashed actual fixed entity-token state",
    }


def _content_hash_stats_npz_entity(shard_files: list[Path]) -> dict[str, Any] | None:
    hashes: set[bytes] = set()
    rows_total = 0
    columns_used: list[str] | None = None
    for path in shard_files:
        with np.load(path, allow_pickle=False) as handle:
            available = [name for name in _ENTITY_STATE_COLUMNS if name in handle]
            if not available:
                return None
            if columns_used is None:
                columns_used = available
            elif available != columns_used:
                raise ValueError(
                    f"{path}: entity-token diagnostic columns drifted: "
                    f"{available} != {columns_used}"
                )
            arrays = [np.asarray(handle[name]) for name in available]
            n = int(len(arrays[0]))
            if any(int(len(array)) != n for array in arrays):
                raise ValueError(f"{path}: entity-token columns disagree on row count")
            byte_arrays = [
                np.ascontiguousarray(array).view(np.uint8).reshape(n, -1)
                for array in arrays
            ]
            combined = np.concatenate(byte_arrays, axis=1)
            for row in combined:
                hashes.add(hashlib.blake2b(row, digest_size=8).digest())
            rows_total += n
    unique_hashes = len(hashes)
    return {
        "rows_total": rows_total,
        "unique_hashes": unique_hashes,
        "unique_fraction": (unique_hashes / rows_total) if rows_total else None,
        "hash": "blake2b-64",
        "representation": "entity_tokens",
        "columns": columns_used or [],
        "note": "obs was an all-zero compatibility placeholder; hashed actual fixed entity-token state",
    }


def _opening_entropy_memmap(
    decision_index: np.ndarray,
    offsets: np.ndarray,
    policy: np.ndarray | None,
    *,
    policy_column_name: str | None,
    decision_low: int,
    decision_high: int,
) -> dict[str, Any]:
    if policy is None or policy_column_name is None:
        return {
            "policy_column_used": None,
            "rows_in_window": 0,
            "mean_normalized_entropy": None,
            "median_normalized_entropy": None,
            "reason": "missing target_policy+prior_policy memmap column",
        }
    indices = np.flatnonzero(
        (decision_index >= int(decision_low)) & (decision_index <= int(decision_high))
    )
    entropies: list[float] = []
    for row_index in indices:
        start = int(offsets[row_index])
        stop = int(offsets[row_index + 1])
        if stop - start <= 1:
            continue
        probs = np.maximum(np.asarray(policy[start:stop], dtype=np.float64), 0.0)
        total = float(probs.sum())
        if total <= 0.0:
            continue
        probs /= total
        entropy = -float(np.sum(probs[probs > 0] * np.log(probs[probs > 0])))
        entropies.append(entropy / float(np.log(stop - start)))
    if not entropies:
        return {
            "policy_column_used": policy_column_name,
            "rows_in_window": int(indices.size),
            "mean_normalized_entropy": None,
            "median_normalized_entropy": None,
        }
    values = np.asarray(entropies, dtype=np.float64)
    return {
        "policy_column_used": policy_column_name,
        "rows_in_window": int(indices.size),
        "rows_with_entropy": int(values.size),
        "mean_normalized_entropy": float(values.mean()),
        "median_normalized_entropy": float(np.median(values)),
    }


def _opening_lines_from_columns(
    game_seed: np.ndarray,
    decision_index: np.ndarray,
    action_taken: np.ndarray,
    *,
    line_length: int,
) -> dict[str, Any]:
    by_game: dict[int, list[tuple[int, int]]] = {}
    mask = (decision_index >= 0) & (decision_index < int(line_length))
    for index in np.flatnonzero(mask):
        by_game.setdefault(int(game_seed[index]), []).append(
            (int(decision_index[index]), int(action_taken[index]))
        )
    lines = [
        tuple(action for _decision, action in sorted(by_game[seed]))
        for seed in sorted(by_game)
    ]
    return line_concentration(lines)


def _run_memmap_scan(
    corpus_dir: Path,
    *,
    generation_label: str,
    line_length: int,
    decision_low: int,
    decision_high: int,
) -> dict[str, Any]:
    meta_path = corpus_dir / "corpus_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("schema") != "memmap_corpus_v1":
        return {"error": f"unsupported memmap schema {meta.get('schema')!r}", "shards_dir": str(corpus_dir)}
    row_count = int(meta["row_count"])
    offsets = np.memmap(
        corpus_dir / "row_offsets.dat", dtype=np.int64, mode="r", shape=(row_count + 1,)
    )
    if int(offsets[-1]) != int(meta["flat_count"]):
        raise ValueError(
            f"{corpus_dir}: final row offset {int(offsets[-1])} != flat_count {meta['flat_count']}"
        )
    game_seed = _memmap_fixed(corpus_dir, meta, "game_seed")
    decision_index = _memmap_fixed(corpus_dir, meta, "decision_index")
    action_taken = _memmap_fixed(corpus_dir, meta, "action_taken")
    if game_seed is None or decision_index is None or action_taken is None:
        return {
            "error": "memmap corpus lacks game_seed / decision_index / action_taken",
            "shards_dir": str(corpus_dir),
        }
    obs = _memmap_fixed(corpus_dir, meta, "obs")
    if obs is not None and bool(np.any(obs)):
        content_stats = _content_hash_stats(obs, representation="obs")
    else:
        entity_columns = {
            name: value
            for name in _ENTITY_STATE_COLUMNS
            if (value := _memmap_fixed(corpus_dir, meta, name)) is not None
        }
        content_stats = (
            _content_hash_stats_columns(entity_columns) if entity_columns else None
        )
    policy_name = "target_policy" if "target_policy" in meta.get("columns", {}) else (
        "prior_policy" if "prior_policy" in meta.get("columns", {}) else None
    )
    policy = _memmap_ragged_flat(corpus_dir, meta, policy_name) if policy_name else None

    return {
        "generation_label": generation_label,
        "shards_dir": str(corpus_dir),
        "source_format": "memmap_corpus_v1",
        "shard_files_count": int(meta.get("shard_count", 0)),
        "rows_total": row_count,
        "games_total": int(np.unique(game_seed).size),
        "line_length": int(line_length),
        "decision_range": [int(decision_low), int(decision_high)],
        "decision_range_inclusive": True,
        "unique_state_fraction_cheap": _unique_pair_stats(game_seed, decision_index),
        "unique_state_fraction_content": content_stats,
        "opening_entropy": _opening_entropy_memmap(
            decision_index,
            offsets,
            policy,
            policy_column_name=policy_name,
            decision_low=decision_low,
            decision_high=decision_high,
        ),
        "opening_line_concentration": _opening_lines_from_columns(
            game_seed,
            decision_index,
            action_taken,
            line_length=line_length,
        ),
        "corpus_meta": {
            "source": meta.get("source"),
            "sources": meta.get("sources"),
            "flat_count": int(meta.get("flat_count", 0)),
            "legal_width": int(meta.get("legal_width", 0)),
            "stats": meta.get("stats"),
        },
    }


def compute_unique_state_fraction_cheap(rows: dict[str, np.ndarray]) -> dict[str, Any]:
    """Fraction of rows whose (game_seed, decision_index) pair is unique
    within the corpus -- catches literal duplicate-row bugs."""
    return _unique_pair_stats(rows["game_seed"], rows["decision_index"])


def compute_unique_state_fraction_content(rows: dict[str, np.ndarray]) -> dict[str, Any] | None:
    """blake2b-hash each row's `obs` array bytes; fraction of rows with a
    unique hash. Returns None (not a fabricated number) when `obs` isn't
    present in the corpus."""
    if "obs" not in rows:
        return None
    return _content_hash_stats(rows["obs"])


def compute_opening_entropy(
    rows: dict[str, np.ndarray], *, decision_low: int, decision_high: int
) -> dict[str, Any]:
    """Normalized-entropy stats over rows with `decision_low <= decision_index
    <= decision_high` (both ends inclusive), using `target_policy` if present,
    else `prior_policy`. Zips `legal_action_ids[row]` with the policy column's
    same row, dropping legal_action_ids' padding fill (-1)."""
    decision_index = rows.get("decision_index")
    legal_action_ids = rows.get("legal_action_ids")
    policy_column_name = "target_policy" if "target_policy" in rows else (
        "prior_policy" if "prior_policy" in rows else None
    )
    if decision_index is None or legal_action_ids is None or policy_column_name is None:
        return {
            "policy_column_used": policy_column_name,
            "rows_in_window": 0,
            "mean_normalized_entropy": None,
            "median_normalized_entropy": None,
            "reason": "missing decision_index / legal_action_ids / target_policy+prior_policy column(s)",
        }
    policy_column = rows[policy_column_name]
    mask = (decision_index >= int(decision_low)) & (decision_index <= int(decision_high))
    idx = np.nonzero(mask)[0]

    entropies: list[float] = []
    for i in idx:
        row_legal = legal_action_ids[i]
        row_policy = policy_column[i]
        policy_dict: dict[int, float] = {}
        for action_id, prob in zip(row_legal, row_policy):
            if int(action_id) == -1:
                continue
            policy_dict[int(action_id)] = float(prob)
        entropy = normalized_entropy(policy_dict)
        if entropy is not None:
            entropies.append(entropy)

    if not entropies:
        return {
            "policy_column_used": policy_column_name,
            "rows_in_window": int(len(idx)),
            "mean_normalized_entropy": None,
            "median_normalized_entropy": None,
        }
    arr = np.asarray(entropies, dtype=np.float64)
    return {
        "policy_column_used": policy_column_name,
        "rows_in_window": int(len(idx)),
        "rows_with_entropy": int(len(entropies)),
        "mean_normalized_entropy": float(arr.mean()),
        "median_normalized_entropy": float(np.median(arr)),
    }


def compute_opening_line_concentration(rows: dict[str, np.ndarray], *, line_length: int) -> dict[str, Any]:
    """Group rows by game_seed; each game's "line" is the tuple of
    action_taken values at decision_index 0..line_length-1, sorted by
    decision_index within the game."""
    game_seed = rows.get("game_seed")
    decision_index = rows.get("decision_index")
    action_taken = rows.get("action_taken")
    if game_seed is None or decision_index is None or action_taken is None:
        return line_concentration([])
    return _opening_lines_from_columns(
        game_seed, decision_index, action_taken, line_length=line_length
    )


def run_scan(
    shards_dir: Path,
    *,
    generation_label: str,
    line_length: int,
    decision_low: int,
    decision_high: int,
) -> dict[str, Any]:
    if (shards_dir / "corpus_meta.json").exists():
        return _run_memmap_scan(
            shards_dir,
            generation_label=generation_label,
            line_length=line_length,
            decision_low=decision_low,
            decision_high=decision_high,
        )
    shard_files = find_shard_files(shards_dir)
    if not shard_files:
        return {"error": f"no shard files found under {shards_dir}", "shards_dir": str(shards_dir)}
    rows = _load_scan_rows(shard_files)
    if not rows:
        return {"error": "shards found but contained no rows", "shards_dir": str(shards_dir)}

    obs = rows.get("obs")
    if obs is not None and bool(np.any(obs)):
        content_stats = compute_unique_state_fraction_content(rows)
    else:
        content_stats = _content_hash_stats_npz_entity(shard_files)

    return {
        "generation_label": generation_label,
        "shards_dir": str(shards_dir),
        "source_format": "npz",
        "shard_files_count": len(shard_files),
        "rows_total": int(len(rows.get("game_seed", []))),
        "games_total": int(len(np.unique(rows["game_seed"]))) if "game_seed" in rows else 0,
        "line_length": int(line_length),
        "decision_range": [int(decision_low), int(decision_high)],
        "decision_range_inclusive": True,
        "unique_state_fraction_cheap": compute_unique_state_fraction_cheap(rows),
        "unique_state_fraction_content": content_stats,
        "opening_entropy": compute_opening_entropy(
            rows, decision_low=decision_low, decision_high=decision_high
        ),
        "opening_line_concentration": compute_opening_line_concentration(rows, line_length=line_length),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards-dir", required=True)
    parser.add_argument(
        "--generation-label",
        required=True,
        help="Free string stamped into the output, for joining across generations downstream.",
    )
    parser.add_argument("--line-length", type=int, default=8)
    parser.add_argument(
        "--decision-range",
        default="1,30",
        help="comma-separated low,high decision_index bounds, BOTH INCLUSIVE (default 1,30)",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    low_str, high_str = args.decision_range.split(",")
    report = run_scan(
        Path(args.shards_dir),
        generation_label=args.generation_label,
        line_length=int(args.line_length),
        decision_low=int(low_str),
        decision_high=int(high_str),
    )
    write_json(args.out, report)
    import json

    print(json.dumps(report, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
