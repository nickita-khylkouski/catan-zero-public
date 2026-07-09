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
import sys
from pathlib import Path
from typing import Any

import numpy as np

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from audit_gumbel_pilot_shards import find_shard_files, load_rows  # noqa: E402
from diag_common import line_concentration, normalized_entropy  # noqa: E402
from factory_common import write_json  # noqa: E402


def compute_unique_state_fraction_cheap(rows: dict[str, np.ndarray]) -> dict[str, Any]:
    """Fraction of rows whose (game_seed, decision_index) pair is unique
    within the corpus -- catches literal duplicate-row bugs."""
    game_seed = rows["game_seed"]
    decision_index = rows["decision_index"]
    n = len(game_seed)
    if n == 0:
        return {"rows_total": 0, "unique_pairs": 0, "unique_fraction": None}
    pairs = list(zip((int(v) for v in game_seed), (int(v) for v in decision_index)))
    unique_pairs = len(set(pairs))
    return {
        "rows_total": n,
        "unique_pairs": unique_pairs,
        "unique_fraction": unique_pairs / n,
    }


def compute_unique_state_fraction_content(rows: dict[str, np.ndarray]) -> dict[str, Any] | None:
    """blake2b-hash each row's `obs` array bytes; fraction of rows with a
    unique hash. Returns None (not a fabricated number) when `obs` isn't
    present in the corpus."""
    if "obs" not in rows:
        return None
    obs = rows["obs"]
    n = len(obs)
    if n == 0:
        return {"rows_total": 0, "unique_hashes": 0, "unique_fraction": None}
    hashes = [
        hashlib.blake2b(np.ascontiguousarray(obs[i]).tobytes(), digest_size=8).hexdigest()
        for i in range(n)
    ]
    unique_hashes = len(set(hashes))
    return {
        "rows_total": n,
        "unique_hashes": unique_hashes,
        "unique_fraction": unique_hashes / n,
    }


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

    by_game: dict[int, list[tuple[int, int]]] = {}
    for i in range(len(game_seed)):
        d = int(decision_index[i])
        if d < 0 or d >= int(line_length):
            continue
        seed = int(game_seed[i])
        by_game.setdefault(seed, []).append((d, int(action_taken[i])))

    lines: list[tuple[int, ...]] = []
    for seed in sorted(by_game):
        entries = sorted(by_game[seed], key=lambda pair: pair[0])
        lines.append(tuple(action for _decision, action in entries))

    return line_concentration(lines)


def run_scan(
    shards_dir: Path,
    *,
    generation_label: str,
    line_length: int,
    decision_low: int,
    decision_high: int,
) -> dict[str, Any]:
    shard_files = find_shard_files(shards_dir)
    if not shard_files:
        return {"error": f"no shard files found under {shards_dir}", "shards_dir": str(shards_dir)}
    rows = load_rows(shard_files)
    if not rows:
        return {"error": "shards found but contained no rows", "shards_dir": str(shards_dir)}

    return {
        "generation_label": generation_label,
        "shards_dir": str(shards_dir),
        "shard_files_count": len(shard_files),
        "rows_total": int(len(rows.get("game_seed", []))),
        "games_total": int(len(np.unique(rows["game_seed"]))) if "game_seed" in rows else 0,
        "line_length": int(line_length),
        "decision_range": [int(decision_low), int(decision_high)],
        "decision_range_inclusive": True,
        "unique_state_fraction_cheap": compute_unique_state_fraction_cheap(rows),
        "unique_state_fraction_content": compute_unique_state_fraction_content(rows),
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
