"""Task #76 side-finding: _duplicate_legal_action_rows used a per-row Python
loop with a np.unique() call inside (O(n) Python-level overhead, dramatically
worse under memory pressure on very large corpora -- discovered while
diagnosing a multi-hour pre-training stall on a 14.4M-row corpus during task
#65's value-repair-v2 relaunch). Vectorizes it to pure numpy, no per-row
Python-level work.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from train_bc import _duplicate_legal_action_rows  # type: ignore  # noqa: E402


def _reference_loop_impl(legal_action_ids: np.ndarray) -> np.ndarray:
    """The original (slow, correct) per-row Python-loop implementation,
    preserved here as the ground truth for equivalence testing."""
    legal = np.asarray(legal_action_ids)
    if legal.ndim != 2:
        return np.asarray([], dtype=np.int64)
    duplicates: list[int] = []
    for row_index, row in enumerate(legal):
        valid = row[row >= 0]
        if valid.size and np.unique(valid).size != valid.size:
            duplicates.append(row_index)
    return np.asarray(duplicates, dtype=np.int64)


def test_no_duplicates_returns_empty():
    legal = np.array([[0, 1, 2, -1, -1], [5, 6, -1, -1, -1]], dtype=np.int16)
    result = _duplicate_legal_action_rows(legal)
    assert result.size == 0


def test_detects_a_single_duplicate_row():
    legal = np.array([[0, 1, 1, -1, -1], [5, 6, 7, -1, -1]], dtype=np.int16)
    result = _duplicate_legal_action_rows(legal)
    assert list(result) == [0]


def test_detects_multiple_duplicate_rows():
    legal = np.array(
        [
            [0, 1, 2, -1],
            [3, 3, -1, -1],
            [4, 5, 6, -1],
            [7, 8, 8, 8],
        ],
        dtype=np.int16,
    )
    result = _duplicate_legal_action_rows(legal)
    assert list(result) == [1, 3]


def test_padding_only_rows_are_not_false_positives():
    """Multiple -1 fill slots in the same row must NOT be flagged as a
    duplicate -- -1 is the "no action here" sentinel, not a real action id."""
    legal = np.array([[-1, -1, -1, -1]], dtype=np.int16)
    result = _duplicate_legal_action_rows(legal)
    assert result.size == 0


def test_single_legal_action_row_is_never_a_duplicate():
    legal = np.array([[42, -1, -1, -1]], dtype=np.int16)
    result = _duplicate_legal_action_rows(legal)
    assert result.size == 0


def test_width_one_column_never_flags():
    legal = np.array([[0], [1], [2]], dtype=np.int16)
    result = _duplicate_legal_action_rows(legal)
    assert result.size == 0


def test_non_rank_2_input_returns_empty():
    legal = np.array([0, 1, 2], dtype=np.int16)
    result = _duplicate_legal_action_rows(legal)
    assert result.size == 0


def test_matches_reference_implementation_on_random_data():
    rng = np.random.default_rng(7)
    n, w = 5000, 54
    legal = rng.integers(-1, 60, size=(n, w)).astype(np.int16)
    # Salt in a handful of rows with an explicit forced duplicate to guarantee
    # the positive case is exercised (random data alone might not collide often
    # enough at this id range/width to be a meaningful equivalence check).
    legal[10, :3] = [5, 5, -1]
    legal[2500, :4] = [1, 2, 1, -1]
    expected = _reference_loop_impl(legal)
    actual = _duplicate_legal_action_rows(legal)
    np.testing.assert_array_equal(np.sort(actual), np.sort(expected))


def test_vectorized_version_is_dramatically_faster_at_scale():
    import time

    rng = np.random.default_rng(0)
    n, w = 200_000, 54
    legal = rng.integers(-1, 400, size=(n, w)).astype(np.int16)
    started = time.perf_counter()
    _duplicate_legal_action_rows(legal)
    elapsed = time.perf_counter() - started
    # The original per-row Python-loop implementation took ~1s for this exact
    # input on an idle host (and far longer under real memory pressure on a
    # 300+GB-resident training process -- see the task #76 finding). A fully
    # vectorized numpy implementation should clear 200k rows in well under
    # 100ms even on a loaded host.
    assert elapsed < 0.5, f"expected a vectorized implementation, took {elapsed:.3f}s"
