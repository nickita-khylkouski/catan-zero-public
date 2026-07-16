from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from train_bc import _parse_game_seed_ranges, split_train_validation_indices  # type: ignore  # noqa: E402


# ---------------------------------------------------------------------------
# _parse_game_seed_ranges
# ---------------------------------------------------------------------------


def test_parse_game_seed_ranges_empty_string():
    assert _parse_game_seed_ranges("") == []


def test_parse_game_seed_ranges_single():
    assert _parse_game_seed_ranges("100:200") == [(100, 200)]


def test_parse_game_seed_ranges_multiple():
    assert _parse_game_seed_ranges("100:200,5000001:5006667") == [(100, 200), (5000001, 5006667)]


def test_parse_game_seed_ranges_rejects_end_before_start():
    with pytest.raises(SystemExit):
        _parse_game_seed_ranges("200:100")


# ---------------------------------------------------------------------------
# split_train_validation_indices with explicit ranges (holdout.json convention)
# ---------------------------------------------------------------------------


def _make_data(game_seeds: np.ndarray) -> dict:
    n = len(game_seeds)
    return {
        "action_taken": np.zeros(n, dtype=np.int64),
        "game_seed": game_seeds,
    }


def test_explicit_ranges_hold_out_exact_game_seeds_not_a_random_fraction():
    # 10 games x 3 rows each; explicit range [5,7] (3 games) must be the ENTIRE
    # validation set, not a random-permutation-selected subset.
    game_seeds = np.repeat(np.arange(10, dtype=np.int64), 3)
    data = _make_data(game_seeds)

    split = split_train_validation_indices(
        data,
        validation_fraction=0.05,  # deliberately different from the true held-out
        # fraction (3/10=30%) -- explicit ranges must override this, not blend with it.
        validation_seed=17,
        validation_max_samples=0,
        validation_game_seed_ranges=[(5, 7)],
    )
    validation_seeds = set(game_seeds[split["validation"]].tolist())
    train_seeds = set(game_seeds[split["train"]].tolist())
    assert validation_seeds == {5, 6, 7}
    assert train_seeds == {0, 1, 2, 3, 4, 8, 9}
    assert len(split["validation"]) == 9  # 3 games x 3 rows
    assert len(split["train"]) == 21  # 7 games x 3 rows


def test_explicit_ranges_no_row_leakage_across_train_and_validation():
    game_seeds = np.repeat(np.arange(20, dtype=np.int64), 5)
    data = _make_data(game_seeds)
    split = split_train_validation_indices(
        data,
        validation_fraction=0.05,
        validation_seed=17,
        validation_max_samples=0,
        validation_game_seed_ranges=[(0, 0), (10, 12), (19, 19)],
    )
    validation_seeds = set(game_seeds[split["validation"]].tolist())
    train_seeds = set(game_seeds[split["train"]].tolist())
    assert validation_seeds == {0, 10, 11, 12, 19}
    assert validation_seeds.isdisjoint(train_seeds)
    assert len(split["train"]) + len(split["validation"]) == len(game_seeds)


def test_explicit_ranges_supports_multiple_disjoint_process_blocks():
    """Mirrors the real task #65 holdout: one contiguous range per generation
    process, at disjoint base-seed blocks."""
    seeds_a = np.arange(5000001, 5000001 + 20)  # process A's block
    seeds_b = np.arange(6400001, 6400001 + 20)  # process B's block
    game_seeds = np.repeat(np.concatenate([seeds_a, seeds_b]), 2)
    data = _make_data(game_seeds)

    # Top 5% (1 game) of each 20-game block held out.
    ranges = [(5000001 + 19, 5000001 + 19), (6400001 + 19, 6400001 + 19)]
    split = split_train_validation_indices(
        data, validation_fraction=0.05, validation_seed=17,
        validation_max_samples=0, validation_game_seed_ranges=ranges,
    )
    validation_seeds = set(game_seeds[split["validation"]].tolist())
    assert validation_seeds == {5000001 + 19, 6400001 + 19}


def test_explicit_ranges_refuse_row_level_validation_cap():
    game_seeds = np.repeat(np.arange(100, dtype=np.int64), 10)
    data = _make_data(game_seeds)
    with pytest.raises(
        SystemExit,
        match="row cap would split held-out games",
    ):
        split_train_validation_indices(
            data,
            validation_fraction=0.05,
            validation_seed=17,
            validation_max_samples=50,
            validation_game_seed_ranges=[(0, 99)],
        )


def test_random_game_split_caps_by_complete_games() -> None:
    game_seeds = np.repeat(np.arange(100, dtype=np.int64), 10)
    data = _make_data(game_seeds)
    split = split_train_validation_indices(
        data,
        validation_fraction=0.5,
        validation_seed=17,
        validation_max_samples=55,
    )

    validation_seeds, counts = np.unique(
        game_seeds[split["validation"]], return_counts=True
    )
    assert len(validation_seeds) == 6
    assert counts.tolist() == [10] * 6
    assert len(split["validation"]) == 60
    assert not set(validation_seeds).intersection(
        game_seeds[split["train"]].tolist()
    )


def test_no_explicit_ranges_falls_back_to_existing_random_behavior():
    """Regression: passing None/empty must not change any existing behavior."""
    game_seeds = np.repeat(np.arange(50, dtype=np.int64), 4)
    data = _make_data(game_seeds)

    baseline = split_train_validation_indices(
        data, validation_fraction=0.1, validation_seed=17, validation_max_samples=0,
    )
    with_none = split_train_validation_indices(
        data, validation_fraction=0.1, validation_seed=17, validation_max_samples=0,
        validation_game_seed_ranges=None,
    )
    with_empty = split_train_validation_indices(
        data, validation_fraction=0.1, validation_seed=17, validation_max_samples=0,
        validation_game_seed_ranges=[],
    )
    np.testing.assert_array_equal(baseline["train"], with_none["train"])
    np.testing.assert_array_equal(baseline["validation"], with_none["validation"])
    np.testing.assert_array_equal(baseline["train"], with_empty["train"])
    np.testing.assert_array_equal(baseline["validation"], with_empty["validation"])


# ---------------------------------------------------------------------------
# CAT-52 audit regressions: split_train_validation_indices must NEVER silently
# degrade to a row-level split (the round-11 val-leak mechanism -- see
# AUDIT.md). Before this fix, a missing "game_seed" column silently defaulted
# to np.arange(n), making every row its own "game" and turning the supposedly
# game-level fallback into a plain row-level permutation with no warning.
# ---------------------------------------------------------------------------


def _make_data_no_game_seed(n: int) -> dict:
    return {"action_taken": np.zeros(n, dtype=np.int64)}


def test_missing_game_seed_column_raises_by_default():
    """Regression: the old silent np.arange(n) default reproduced the exact
    round-11 mechanism (each row treated as its own game). Must now refuse."""
    data = _make_data_no_game_seed(2000)
    with pytest.raises(SystemExit):
        split_train_validation_indices(
            data, validation_fraction=0.1, validation_seed=17, validation_max_samples=0,
        )


def test_missing_game_seed_column_raises_even_with_explicit_ranges():
    """The explicit --validation-game-seed-ranges path had the identical
    np.arange(n) default; a caller who believes they are using the "safe"
    explicit path must not be silently downgraded to row-level either."""
    data = _make_data_no_game_seed(2000)
    with pytest.raises(SystemExit):
        split_train_validation_indices(
            data, validation_fraction=0.05, validation_seed=17, validation_max_samples=0,
            validation_game_seed_ranges=[(0, 10)],
        )


def test_missing_game_seed_allowed_via_explicit_opt_in():
    """allow_missing_game_seed=True (--allow-missing-game-seed-validation-split)
    is the only sanctioned way to get row-level behavior back, and it must
    still actually run (not silently no-op)."""
    data = _make_data_no_game_seed(200)
    split = split_train_validation_indices(
        data, validation_fraction=0.1, validation_seed=17, validation_max_samples=0,
        allow_missing_game_seed=True,
    )
    assert len(split["train"]) + len(split["validation"]) == 200
    assert len(split["validation"]) > 0


def test_degenerate_single_game_seed_small_corpus_warns_on_stderr(capsys):
    """The other guarded-but-quiet corner: a corpus with a real game_seed
    column but only one unique value (n < 1000) still falls back to a
    row-level permutation (by design, for tiny synthetic/smoke corpora) --
    but it must be OBSERVABLE, not silently indistinguishable from a safe
    game-level split. Exercise the path (not just assume it's absent) and
    assert the loud warning actually fires."""
    game_seeds = np.full(200, 7, dtype=np.int64)
    data = _make_data(game_seeds)
    split = split_train_validation_indices(
        data, validation_fraction=0.1, validation_seed=17, validation_max_samples=0,
    )
    assert len(split["train"]) + len(split["validation"]) == 200
    err = capsys.readouterr().err
    assert "WARNING" in err and "ROW-LEVEL" in err


def test_degenerate_single_game_seed_large_corpus_still_raises():
    """Regression guard: the n>=1000 large-corpus case must keep raising
    (unchanged behavior) rather than being downgraded to only a warning."""
    game_seeds = np.full(1000, 7, dtype=np.int64)
    data = _make_data(game_seeds)
    with pytest.raises(SystemExit):
        split_train_validation_indices(
            data, validation_fraction=0.1, validation_seed=17, validation_max_samples=0,
        )


def test_synthetic_overlapping_windows_game_seed_ranges_yield_zero_row_overlap():
    """VERIFICATION smoke test (CAT-52 ticket): two 'windows' of the flywheel
    corpus with a deliberately overlapping band of game seeds. Using
    --validation-game-seed-ranges to hold out the overlap band must produce
    train/validation sets with ZERO shared game seeds."""
    window_a_seeds = np.arange(100, 140)   # window A: seeds [100, 140)
    window_b_seeds = np.arange(120, 160)   # window B: seeds [120, 160) -- overlaps A in [120,140)
    all_seeds = np.concatenate([window_a_seeds, window_b_seeds])
    game_seeds = np.repeat(all_seeds, 5)
    data = _make_data(game_seeds)

    # Hold out exactly the overlap band -- the ambiguous region between windows.
    split = split_train_validation_indices(
        data, validation_fraction=0.05, validation_seed=17, validation_max_samples=0,
        validation_game_seed_ranges=[(120, 139)],
    )
    train_seeds = set(game_seeds[split["train"]].tolist())
    validation_seeds = set(game_seeds[split["validation"]].tolist())
    assert train_seeds.isdisjoint(validation_seeds)
    assert validation_seeds == set(range(120, 140))
