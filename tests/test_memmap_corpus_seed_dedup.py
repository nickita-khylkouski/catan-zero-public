"""Unit tests for build_memmap_corpus.py's game_seed duplicate-run detector
(FIX 1, task #85): a shard-boundary false-negative in the earlier
pending/closed-set logic, and the new --abort-on-duplicate-seeds behavior.

These use tiny synthetic seed columns / npz shards -- no real teacher data
required.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
_TOOLS_DIR = _REPO / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from build_memmap_corpus import _GameSeedRunTracker, build_memmap_corpus  # type: ignore  # noqa: E402
from train_bc import MemmapCorpus, load_teacher_data  # type: ignore  # noqa: E402


# ---------------------------------------------------------------------------
# FIX 1a: confirm the shard-boundary false-negative in the tracker.
# ---------------------------------------------------------------------------


def test_tracker_detects_duplicate_that_reappears_after_boundary_continuation():
    """Shard 1 ends with seed 3 (run stays open). Shard 2 OPENS with a
    continuation of seed 3 (merged, not a new run), then seed 4, then seed 3
    AGAIN as a genuinely separate, non-contiguous run within the SAME shard.
    That second occurrence of 3 is a real duplicate (collision) and must be
    flagged -- this is exactly the case the earlier pending/closed-set logic
    missed, because the continuation-merged run of 3 was never registered in
    the closed-set while open, so its later reappearance was mistaken for a
    first-time occurrence.
    """
    tracker = _GameSeedRunTracker()
    tracker.observe_shard(np.array([1, 1, 2, 2, 3, 3], dtype=np.int64))
    assert not tracker.has_duplicates

    tracker.observe_shard(np.array([3, 3, 4, 4, 3, 3], dtype=np.int64))
    assert tracker.has_duplicates, "seed 3's second, non-contiguous run must be flagged"
    assert tracker.duplicate_count == 1


def test_tracker_no_false_positive_for_simple_boundary_continuation():
    """A game_seed that merely spans a shard boundary once (no reappearance)
    must NOT be flagged."""
    tracker = _GameSeedRunTracker()
    tracker.observe_shard(np.array([1, 1, 2, 2], dtype=np.int64))
    tracker.observe_shard(np.array([2, 2, 3, 3], dtype=np.int64))
    assert not tracker.has_duplicates


def test_tracker_detects_duplicate_within_single_shard_no_boundary():
    """Duplicate run entirely inside one shard call (no cross-shard merge
    involved) is also caught."""
    tracker = _GameSeedRunTracker()
    tracker.observe_shard(np.array([5, 5, 6, 6, 5, 5], dtype=np.int64))
    assert tracker.has_duplicates
    assert tracker.duplicate_count == 1


def test_tracker_detects_duplicate_separated_by_a_later_shard():
    """The duplicate run doesn't have to be in the immediately-next shard --
    it can reappear several shards later, as long as the boundary-continuation
    merge doesn't hide it."""
    tracker = _GameSeedRunTracker()
    tracker.observe_shard(np.array([1, 1, 2, 2], dtype=np.int64))  # 2 stays open
    tracker.observe_shard(np.array([2, 2, 3, 3], dtype=np.int64))  # continues 2, then 3
    tracker.observe_shard(np.array([4, 4, 2, 2], dtype=np.int64))  # 2 reappears: dup
    assert tracker.has_duplicates
    assert tracker.duplicate_count == 1


def test_tracker_empty_shard_is_noop():
    tracker = _GameSeedRunTracker()
    tracker.observe_shard(np.array([], dtype=np.int64))
    tracker.observe_shard(np.array([7, 7], dtype=np.int64))
    assert not tracker.has_duplicates


# ---------------------------------------------------------------------------
# FIX 1b: --abort-on-duplicate-seeds (default on) hard-exits; the escape
# hatch only warns. Uses tiny synthetic npz shards (minimal required columns).
# ---------------------------------------------------------------------------


def _write_synthetic_shard(
    path: Path,
    *,
    game_seed: np.ndarray,
    include_aux: bool = True,
) -> None:
    n = int(game_seed.shape[0])
    legal_width = 2
    arrays = {
        "obs": np.zeros((n, 4), dtype=np.float16),
        "legal_action_ids": np.zeros((n, legal_width), dtype=np.int16),
        "legal_action_context": np.zeros((n, legal_width, 1), dtype=np.float16),
        "action_taken": np.zeros(n, dtype=np.int16),
        "game_seed": game_seed.astype(np.int64),
    }
    if include_aux:
        arrays.update(
            {
                "aux_longest_road": np.arange(n, dtype=np.float32) % 2,
                "aux_largest_army": (np.arange(n, dtype=np.float32) + 1) % 2,
                "aux_vp_in_n": np.arange(n, dtype=np.float32) / 10.0,
                "aux_next_settlement": np.arange(n, dtype=np.int16) % 54,
                "aux_robber_target": np.arange(n, dtype=np.int16) % 19,
            }
        )
    np.savez(path, **arrays)


def _make_duplicate_seed_shards(tmp_path: Path) -> Path:
    """Two shards whose game_seed columns reproduce the shard-boundary
    reappearance case above."""
    teacher_dir = tmp_path / "teacher"
    teacher_dir.mkdir()
    _write_synthetic_shard(teacher_dir / "shard0.npz", game_seed=np.array([1, 1, 2, 2, 3, 3]))
    _write_synthetic_shard(teacher_dir / "shard1.npz", game_seed=np.array([3, 3, 4, 4, 3, 3]))
    return teacher_dir


def _make_clean_shards(tmp_path: Path) -> Path:
    teacher_dir = tmp_path / "teacher"
    teacher_dir.mkdir()
    _write_synthetic_shard(teacher_dir / "shard0.npz", game_seed=np.array([1, 1, 2, 2]))
    _write_synthetic_shard(teacher_dir / "shard1.npz", game_seed=np.array([2, 2, 3, 3]))
    return teacher_dir


def test_build_memmap_corpus_aborts_by_default_on_duplicate_seeds(tmp_path):
    teacher_dir = _make_duplicate_seed_shards(tmp_path)
    with pytest.raises(SystemExit, match="ABORTING"):
        build_memmap_corpus(
            teacher_dir,
            tmp_path / "corpus",
            progress_every=0,
            abort_on_duplicate_seeds=True,
        )


def test_build_memmap_corpus_warns_only_with_escape_hatch(tmp_path, capsys):
    teacher_dir = _make_duplicate_seed_shards(tmp_path)
    meta = build_memmap_corpus(
        teacher_dir,
        tmp_path / "corpus",
        progress_every=0,
        abort_on_duplicate_seeds=False,
    )
    assert meta["stats"]["has_duplicate_game_seeds"] is True
    assert meta["stats"]["duplicate_game_seed_count"] == 1
    captured = capsys.readouterr()
    assert "WARNING" in captured.err


def test_build_memmap_corpus_default_kwarg_is_abort_on():
    import inspect

    sig = inspect.signature(build_memmap_corpus)
    assert sig.parameters["abort_on_duplicate_seeds"].default is True


def test_build_memmap_corpus_no_duplicates_does_not_abort(tmp_path):
    teacher_dir = _make_clean_shards(tmp_path)
    meta = build_memmap_corpus(
        teacher_dir,
        tmp_path / "corpus",
        progress_every=0,
        abort_on_duplicate_seeds=True,
    )
    assert meta["stats"]["has_duplicate_game_seeds"] is False
    assert meta["stats"]["duplicate_game_seed_count"] == 0

    corpus = MemmapCorpus(tmp_path / "corpus")
    for key in (
        "aux_longest_road",
        "aux_largest_army",
        "aux_vp_in_n",
        "aux_next_settlement",
        "aux_robber_target",
    ):
        assert key in corpus
        assert len(corpus[key]) == len(corpus)


def test_legacy_and_aux_sources_mix_with_aligned_ignore_fills(tmp_path):
    legacy = tmp_path / "legacy"
    labeled = tmp_path / "labeled"
    legacy.mkdir()
    labeled.mkdir()
    _write_synthetic_shard(
        legacy / "shard0.npz",
        game_seed=np.array([10, 10]),
        include_aux=False,
    )
    _write_synthetic_shard(
        labeled / "shard0.npz",
        game_seed=np.array([11, 11]),
        include_aux=True,
    )

    # In-RAM NPZ loading backfills already-seen legacy rows when the first
    # labeled shard appears.
    mixed = tmp_path / "mixed"
    mixed.mkdir()
    _write_synthetic_shard(
        mixed / "shard0.npz",
        game_seed=np.array([10, 10]),
        include_aux=False,
    )
    _write_synthetic_shard(
        mixed / "shard1.npz",
        game_seed=np.array([11, 11]),
        include_aux=True,
    )
    loaded = load_teacher_data(mixed)
    assert len(loaded["aux_vp_in_n"]) == 4
    assert np.all(np.isnan(loaded["aux_vp_in_n"][:2]))
    assert np.all(loaded["aux_next_settlement"][:2] == -1)
    np.testing.assert_array_equal(loaded["aux_next_settlement"][2:], np.array([0, 1]))

    # The production memmap path detects CAT-100 at the per-source level and
    # applies the same fills before enforcing its uniform column schema.
    build_memmap_corpus(
        [legacy, labeled],
        tmp_path / "mixed_corpus",
        progress_every=0,
    )
    corpus = MemmapCorpus(tmp_path / "mixed_corpus")
    assert len(corpus["aux_vp_in_n"]) == 4
    assert np.all(np.isnan(corpus["aux_vp_in_n"][:2]))
    assert np.all(corpus["aux_next_settlement"][:2] == -1)
