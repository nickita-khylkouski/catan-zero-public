from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from corpus_diversity_scan import (  # type: ignore  # noqa: E402
    compute_opening_entropy,
    compute_opening_line_concentration,
    compute_unique_state_fraction_cheap,
    compute_unique_state_fraction_content,
    run_scan,
)

# ---------------------------------------------------------------------------
# Shared synthetic corpus: 18 rows across 3 games (seeds 100, 200, 300).
#   - game 100 (rows 0-5): plants a cheap (game_seed, decision_index) duplicate
#     at rows 0/1 (both (100, 0)).
#   - game 200 (rows 6-9): plants a SECOND cheap duplicate at rows 7/9 (both
#     (200, 1)), and row 8's obs is a bytewise duplicate of row 5's obs
#     (different game_seed/decision_index -- only content-dedup catches it).
#   - game 300 (rows 10-17): clean, 8 sequential decisions 0..7, no
#     duplicates -- used for the exact-line assertion.
# ---------------------------------------------------------------------------

_GAME_SEED = [100, 100, 100, 100, 100, 100, 200, 200, 200, 200, 300, 300, 300, 300, 300, 300, 300, 300]
_DECISION_INDEX = [0, 0, 1, 2, 3, 4, 0, 1, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7]
_ACTION_TAKEN = [10, 10, 11, 12, 10, 12, 20, 21, 20, 21, 30, 31, 32, 33, 34, 35, 36, 37]
_LEGAL_ACTION_IDS = [
    [10, 11, 12],
    [10, 11, 12],
    [10, 11, -1],
    [10, 11, 12],
    [10, -1, -1],
    [10, 11, 12],
    [20, 21, 22],
    [20, 21, 22],
    [20, 21, -1],
    [20, 21, 22],
    [30, 31, 32],
    [31, 32, 33],
    [32, 33, 34],
    [33, 34, 35],
    [34, 35, 36],
    [35, 36, 37],
    [36, 37, 38],
    [37, 38, 39],
]
_TARGET_POLICY = [
    [0.5, 0.3, 0.2],
    [0.5, 0.3, 0.2],
    [0.6, 0.4, 0.0],
    [0.34, 0.33, 0.33],
    [1.0, 0.0, 0.0],
    [0.1, 0.1, 0.8],
    [0.2, 0.3, 0.5],
    [0.4, 0.3, 0.3],
    [0.5, 0.5, 0.0],
    [0.4, 0.3, 0.3],
    [1 / 3, 1 / 3, 1 / 3],
    [1 / 3, 1 / 3, 1 / 3],
    [1 / 3, 1 / 3, 1 / 3],
    [1 / 3, 1 / 3, 1 / 3],
    [1 / 3, 1 / 3, 1 / 3],
    [1 / 3, 1 / 3, 1 / 3],
    [1 / 3, 1 / 3, 1 / 3],
    [1 / 3, 1 / 3, 1 / 3],
]
_OBS = [
    [0.0, 0.0],
    [0.1, 0.1],
    [0.2, 0.2],
    [0.3, 0.3],
    [0.4, 0.4],
    [9.0, 9.0],
    [0.6, 0.6],
    [0.7, 0.7],
    [9.0, 9.0],
    [0.75, 0.75],
    [10.0, 10.0],
    [11.0, 11.0],
    [12.0, 12.0],
    [13.0, 13.0],
    [14.0, 14.0],
    [15.0, 15.0],
    [16.0, 16.0],
    [17.0, 17.0],
]


def _rows() -> dict[str, np.ndarray]:
    return {
        "game_seed": np.asarray(_GAME_SEED, dtype=np.int64),
        "decision_index": np.asarray(_DECISION_INDEX, dtype=np.int32),
        "action_taken": np.asarray(_ACTION_TAKEN, dtype=np.int16),
        "legal_action_ids": np.asarray(_LEGAL_ACTION_IDS, dtype=np.int16),
        "target_policy": np.asarray(_TARGET_POLICY, dtype=np.float32),
        "obs": np.asarray(_OBS, dtype=np.float32),
    }


def test_unique_state_fraction_cheap_catches_planted_duplicates():
    result = compute_unique_state_fraction_cheap(_rows())
    assert result["rows_total"] == 18
    # Two duplicate (game_seed, decision_index) pairs planted: (100,0) and (200,1).
    assert result["unique_pairs"] == 16
    assert result["unique_fraction"] == pytest.approx(16 / 18)


def test_unique_state_fraction_content_catches_bytewise_duplicate_obs():
    result = compute_unique_state_fraction_content(_rows())
    assert result is not None
    assert result["rows_total"] == 18
    # Rows 5 and 8 share identical obs bytes despite different game_seed/decision_index.
    assert result["unique_hashes"] == 17
    assert result["unique_fraction"] == pytest.approx(17 / 18)


def test_unique_state_fraction_content_is_none_without_obs_column():
    rows = _rows()
    del rows["obs"]
    assert compute_unique_state_fraction_content(rows) is None


def test_opening_entropy_window_filters_and_excludes_degenerate_rows():
    result = compute_opening_entropy(_rows(), decision_low=1, decision_high=3)
    assert result["policy_column_used"] == "target_policy"
    # decision_index in {1,2,3}: rows 2,3,4,7,8,9,11,12,13 (9 rows).
    assert result["rows_in_window"] == 9
    # Row 4 has only 1 legal action (degenerate) -> normalized_entropy is None,
    # excluded from the entropy list even though it's counted in the window.
    assert result["rows_with_entropy"] == 8
    assert 0.0 <= result["mean_normalized_entropy"] <= 1.0
    assert 0.0 <= result["median_normalized_entropy"] <= 1.0


def test_opening_entropy_outside_window_is_excluded():
    # decision_index 4..7 excludes all rows in [1,3].
    result = compute_opening_entropy(_rows(), decision_low=4, decision_high=7)
    # decision_index in {4,5,6,7}: rows 5(d4),14(d4),15(d5),16(d6),17(d7) = 5 rows.
    assert result["rows_in_window"] == 5


def test_opening_line_concentration_groups_by_game_seed():
    result = compute_opening_line_concentration(_rows(), line_length=8)
    assert result["n_games"] == 3
    assert 0.0 <= result["top1_fraction"] <= 1.0
    assert result["n_unique_lines"] <= 3


def test_opening_line_concentration_exact_line_on_clean_game():
    """Game 300's rows are duplicate-free, so its line is exactly the
    sorted-by-decision_index action_taken sequence."""
    rows = _rows()
    mask = rows["game_seed"] == 300
    clean_rows = {key: value[mask] for key, value in rows.items()}
    result = compute_opening_line_concentration(clean_rows, line_length=8)
    assert result["n_games"] == 1
    assert result["top1_fraction"] == pytest.approx(1.0)
    assert result["n_unique_lines"] == 1


def test_opening_line_concentration_respects_line_length():
    rows = _rows()
    mask = rows["game_seed"] == 300
    clean_rows = {key: value[mask] for key, value in rows.items()}
    result = compute_opening_line_concentration(clean_rows, line_length=3)
    # Only decisions 0,1,2 should contribute -> a 3-long line, still 1 unique game.
    assert result["n_games"] == 1


def test_run_scan_end_to_end_on_synthetic_npz(tmp_path):
    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()
    rows = _rows()
    np.savez(shards_dir / "shard_000.npz", **rows)

    report = run_scan(
        shards_dir,
        generation_label="gen-test",
        line_length=8,
        decision_low=1,
        decision_high=3,
    )
    assert report["generation_label"] == "gen-test"
    assert report["rows_total"] == 18
    assert report["games_total"] == 3
    assert report["unique_state_fraction_cheap"]["unique_fraction"] == pytest.approx(16 / 18)
    assert report["unique_state_fraction_content"]["unique_fraction"] == pytest.approx(17 / 18)
    assert report["opening_entropy"]["rows_in_window"] == 9
    assert report["opening_line_concentration"]["n_games"] == 3


def test_npz_scan_hashes_entity_tokens_when_obs_is_zero_placeholder(tmp_path):
    shards_dir = tmp_path / "entity_shards"
    shards_dir.mkdir()
    rows = _rows()
    rows["obs"] = np.zeros_like(rows["obs"])
    rows["global_tokens"] = np.arange(18, dtype=np.float32).reshape(18, 1, 1)
    np.savez(shards_dir / "shard_000.npz", **rows)

    report = run_scan(
        shards_dir,
        generation_label="entity-placeholder",
        line_length=8,
        decision_low=1,
        decision_high=3,
    )
    content = report["unique_state_fraction_content"]
    assert content["representation"] == "entity_tokens"
    assert content["columns"] == ["global_tokens"]
    assert content["unique_hashes"] == 18
    assert content["unique_fraction"] == pytest.approx(1.0)


def test_run_scan_errors_cleanly_on_empty_dir(tmp_path):
    shards_dir = tmp_path / "empty"
    shards_dir.mkdir()
    report = run_scan(shards_dir, generation_label="gen-empty", line_length=8, decision_low=1, decision_high=30)
    assert "error" in report


def _write_memmap_corpus(corpus_dir: Path) -> None:
    corpus_dir.mkdir()
    rows = _rows()
    counts = np.sum(rows["legal_action_ids"] >= 0, axis=1).astype(np.int64)
    offsets = np.empty(len(counts) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    offsets.tofile(corpus_dir / "row_offsets.dat")
    for name in ("game_seed", "decision_index", "action_taken", "obs"):
        rows[name].tofile(corpus_dir / f"{name}.dat")
    prefix = rows["legal_action_ids"] >= 0
    rows["legal_action_ids"][prefix].tofile(corpus_dir / "legal_action_ids.dat")
    rows["target_policy"][prefix].tofile(corpus_dir / "target_policy.dat")
    meta = {
        "schema": "memmap_corpus_v1",
        "row_count": len(counts),
        "flat_count": int(offsets[-1]),
        "legal_width": int(rows["legal_action_ids"].shape[1]),
        "shard_count": 1,
        "source": "synthetic",
        "sources": ["synthetic"],
        "stats": {"has_duplicate_game_seeds": False},
        "columns": {
            "game_seed": {"kind": "fixed", "dtype": rows["game_seed"].dtype.str, "inner_shape": []},
            "decision_index": {"kind": "fixed", "dtype": rows["decision_index"].dtype.str, "inner_shape": []},
            "action_taken": {"kind": "fixed", "dtype": rows["action_taken"].dtype.str, "inner_shape": []},
            "obs": {"kind": "fixed", "dtype": rows["obs"].dtype.str, "inner_shape": [2]},
            "legal_action_ids": {"kind": "ragged2d", "dtype": rows["legal_action_ids"].dtype.str, "fill": -1.0},
            "target_policy": {"kind": "ragged2d", "dtype": rows["target_policy"].dtype.str, "fill": 0.0},
        },
    }
    (corpus_dir / "corpus_meta.json").write_text(json.dumps(meta), encoding="utf-8")


def test_run_scan_memmap_matches_npz_metrics(tmp_path):
    corpus_dir = tmp_path / "memmap"
    _write_memmap_corpus(corpus_dir)

    report = run_scan(
        corpus_dir,
        generation_label="gen-memmap",
        line_length=8,
        decision_low=1,
        decision_high=3,
    )
    assert report["source_format"] == "memmap_corpus_v1"
    assert report["rows_total"] == 18
    assert report["games_total"] == 3
    assert report["unique_state_fraction_cheap"]["unique_fraction"] == pytest.approx(16 / 18)
    assert report["unique_state_fraction_content"]["unique_fraction"] == pytest.approx(17 / 18)
    assert report["opening_entropy"]["rows_in_window"] == 9
    assert report["opening_entropy"]["rows_with_entropy"] == 8
    assert report["opening_line_concentration"]["n_games"] == 3


def test_memmap_scan_hashes_entity_tokens_when_obs_is_zero_placeholder(tmp_path):
    corpus_dir = tmp_path / "entity_memmap"
    _write_memmap_corpus(corpus_dir)
    rows = _rows()
    np.zeros_like(rows["obs"]).tofile(corpus_dir / "obs.dat")
    global_tokens = np.arange(18, dtype=np.float32).reshape(18, 1, 1)
    global_tokens.tofile(corpus_dir / "global_tokens.dat")
    meta_path = corpus_dir / "corpus_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["columns"]["global_tokens"] = {
        "kind": "fixed",
        "dtype": global_tokens.dtype.str,
        "inner_shape": [1, 1],
    }
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    report = run_scan(
        corpus_dir,
        generation_label="entity-memmap-placeholder",
        line_length=8,
        decision_low=1,
        decision_high=3,
    )
    content = report["unique_state_fraction_content"]
    assert content["representation"] == "entity_tokens"
    assert content["columns"] == ["global_tokens"]
    assert content["unique_hashes"] == 18


def test_memmap_scan_fails_on_inconsistent_final_offset(tmp_path):
    corpus_dir = tmp_path / "bad_memmap"
    _write_memmap_corpus(corpus_dir)
    offsets = np.fromfile(corpus_dir / "row_offsets.dat", dtype=np.int64)
    offsets[-1] -= 1
    offsets.tofile(corpus_dir / "row_offsets.dat")
    with pytest.raises(ValueError, match="final row offset"):
        run_scan(
            corpus_dir,
            generation_label="bad",
            line_length=8,
            decision_low=1,
            decision_high=3,
        )
