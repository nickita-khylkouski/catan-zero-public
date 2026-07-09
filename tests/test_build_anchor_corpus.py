"""Unit tests for tools/build_anchor_corpus.py (CAT-30).

Uses tiny synthetic npz shards (minimal required columns, same convention as
tests/test_memmap_corpus_seed_dedup.py) -- no real teacher data, no GPU/torch.
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

from build_anchor_corpus import (  # type: ignore  # noqa: E402
    ANCHOR_MANIFEST_SCHEMA,
    append_anchor_manifest_entry,
    build_anchor_corpus,
    filter_shards_by_seed_ranges,
    filter_shards_by_tags,
    load_anchor_manifest,
    ranges_overlap,
    verify_anchor_corpus,
    _parse_seed_ranges,
    _seed_mask,
)


def _write_synthetic_shard(path: Path, *, game_seed: np.ndarray, tag: np.ndarray | None = None) -> None:
    n = int(game_seed.shape[0])
    legal_width = 2
    kwargs = dict(
        obs=np.zeros((n, 4), dtype=np.float16),
        legal_action_ids=np.zeros((n, legal_width), dtype=np.int16),
        legal_action_context=np.zeros((n, legal_width, 1), dtype=np.float16),
        action_taken=np.zeros(n, dtype=np.int16),
        game_seed=game_seed.astype(np.int64),
    )
    if tag is not None:
        kwargs["outcome_vs_external"] = tag
    np.savez(path, **kwargs)


# ---------------------------------------------------------------------------
# parsing / mask helpers
# ---------------------------------------------------------------------------


def test_parse_seed_ranges_basic():
    assert _parse_seed_ranges("100:200,5000:6000") == [(100, 200), (5000, 6000)]


def test_parse_seed_ranges_rejects_end_before_start():
    with pytest.raises(SystemExit):
        _parse_seed_ranges("200:100")


def test_seed_mask_multi_range():
    seeds = np.array([5, 50, 150, 250, 9999])
    mask = _seed_mask(seeds, [(0, 10), (100, 200)])
    assert mask.tolist() == [True, False, True, False, False]


def test_ranges_overlap_true_and_false():
    assert ranges_overlap([(0, 10)], [(10, 20)])  # touching endpoints count as overlap
    assert not ranges_overlap([(0, 9)], [(10, 20)])


def test_seed_mask_is_inclusive_on_both_bounds_pins_down_ledger_translation():
    """--seed-ranges/--exclude-seed-ranges bounds are BOTH inclusive (mirrors
    train_bc._parse_game_seed_ranges). The seed ledger documents ranges
    half-open (e.g. VAL-ONLY = "[6.19B, 6.2B)"), so a caller must subtract 1
    from the ledger's upper bound before passing it here -- 6_200_000_000
    itself is deliberately INCLUDED by this function when passed verbatim as
    an end bound, which is why the ledger's own upper-bound number is NOT the
    correct value to pass without translation. This test pins down that this
    is the actual, current, intentional behavior (see _parse_seed_ranges's
    docstring caution) so a future change to exclusive-upper-bound semantics
    is a deliberate, reviewed decision rather than an accidental drift."""
    seeds = np.array([6_199_999_999, 6_200_000_000, 6_200_000_001])
    mask = _seed_mask(seeds, [(6_190_000_000, 6_200_000_000)])
    assert mask.tolist() == [True, True, False]


# ---------------------------------------------------------------------------
# seed-range extraction correctness
# ---------------------------------------------------------------------------


def test_filter_shards_by_seed_ranges_extracts_only_matching_rows(tmp_path):
    teacher_dir = tmp_path / "teacher"
    teacher_dir.mkdir()
    # shard0: seeds 1,1,2,2 (2 games x 2 rows); shard1: seeds 3,3, 6190000000,6190000000
    _write_synthetic_shard(teacher_dir / "shard0.npz", game_seed=np.array([1, 1, 2, 2]))
    _write_synthetic_shard(
        teacher_dir / "shard1.npz",
        game_seed=np.array([3, 3, 6_190_000_000, 6_190_000_000]),
    )
    staging = tmp_path / "staging"
    stats = filter_shards_by_seed_ranges(
        [teacher_dir / "shard0.npz", teacher_dir / "shard1.npz"],
        [(6_190_000_000, 6_200_000_000)],
        staging,
    )
    assert stats["rows_out"] == 2
    assert stats["seeds"] == [6_190_000_000]
    manifest = json.loads((staging / "manifest.json").read_text())
    assert manifest["rows"] == 2
    assert len(manifest["shards"]) == 1  # shard0 contributed zero rows, not written


def test_filter_shards_by_seed_ranges_refuses_empty_result(tmp_path):
    teacher_dir = tmp_path / "teacher"
    teacher_dir.mkdir()
    _write_synthetic_shard(teacher_dir / "shard0.npz", game_seed=np.array([1, 1, 2, 2]))
    with pytest.raises(SystemExit, match="no rows matched"):
        filter_shards_by_seed_ranges(
            [teacher_dir / "shard0.npz"], [(6_190_000_000, 6_200_000_000)], tmp_path / "staging",
        )


# ---------------------------------------------------------------------------
# end-to-end anchor build + no-overlap-with-train guarantee
# ---------------------------------------------------------------------------


def _make_window_shards(tmp_path: Path) -> Path:
    """A 'window' with both TRAINING seeds (low range) and reserved .valonly
    seeds (the [6_190_000_000, 6_200_000_000) band from the seed ledger)."""
    teacher_dir = tmp_path / "window"
    teacher_dir.mkdir()
    train_seeds = np.repeat(np.arange(100, 110), 3)  # 10 "training" games
    valonly_seeds = np.repeat(np.array([6_190_000_001, 6_190_000_002, 6_190_000_003]), 3)
    _write_synthetic_shard(teacher_dir / "shard0.npz", game_seed=train_seeds)
    _write_synthetic_shard(teacher_dir / "shard1.npz", game_seed=valonly_seeds)
    return teacher_dir


def test_build_anchor_corpus_end_to_end_and_manifest_entry(tmp_path):
    source = _make_window_shards(tmp_path)
    out_root = tmp_path / "anchors"
    entry = build_anchor_corpus(
        source_roots=[source],
        anchor_name="anchor_gen4",
        anchor_type="current_window",
        out_root=out_root,
        seed_ranges=[(6_190_000_000, 6_200_000_000)],
        exclude_seed_ranges=[(0, 999)],  # the training-window range above
        hard_anchor_tag_column=None,
        hard_anchor_tag_values=None,
        force=False,
    )
    assert entry["row_count"] == 9  # 3 valonly games x 3 rows
    assert entry["verify_stats"]["seed_ranges_verified"] is True
    assert entry["verify_stats"]["no_train_overlap_verified"] is True
    assert entry["promotion_signal"] is False

    corpus_dir = out_root / "anchor_gen4"
    assert corpus_dir.is_dir()
    meta = json.loads((corpus_dir / "corpus_meta.json").read_text())
    assert meta["row_count"] == 9

    manifest = load_anchor_manifest(out_root / "anchor_manifest.json")
    assert manifest["schema"] == ANCHOR_MANIFEST_SCHEMA
    assert [a["name"] for a in manifest["anchors"]] == ["anchor_gen4"]

    # staging dir must not survive (no leftover mutable state between builds)
    assert not (out_root / ".staging_anchor_gen4").exists()


def test_build_anchor_corpus_rejects_train_window_overlap(tmp_path):
    """The 'never overlaps a training window' standing rule, checked
    mechanically: if the caller's declared exclude range actually contains
    extracted rows, refuse to publish."""
    source = _make_window_shards(tmp_path)
    with pytest.raises(SystemExit, match="training-window"):
        build_anchor_corpus(
            source_roots=[source],
            anchor_name="anchor_bad",
            anchor_type="current_window",
            out_root=tmp_path / "anchors",
            seed_ranges=[(6_190_000_000, 6_200_000_000)],
            exclude_seed_ranges=[(6_190_000_000, 6_200_000_000)],  # deliberately wrong: overlaps itself
            hard_anchor_tag_column=None,
            hard_anchor_tag_values=None,
            force=False,
        )


def test_build_anchor_corpus_refuses_to_overwrite_existing_anchor_without_force(tmp_path):
    source = _make_window_shards(tmp_path)
    out_root = tmp_path / "anchors"
    kwargs = dict(
        source_roots=[source], anchor_name="anchor_r7", anchor_type="longitudinal",
        out_root=out_root, seed_ranges=[(6_190_000_000, 6_200_000_000)],
        exclude_seed_ranges=None, hard_anchor_tag_column=None, hard_anchor_tag_values=None,
    )
    build_anchor_corpus(**kwargs, force=False)
    with pytest.raises(SystemExit, match="already exists"):
        build_anchor_corpus(**kwargs, force=False)
    # --force explicitly allows the rebuild
    build_anchor_corpus(**kwargs, force=True)


def test_verify_anchor_corpus_catches_seed_outside_declared_range(tmp_path):
    """Direct unit test of the self-consistency verifier (not trusting the
    upstream filter step alone): fabricate a corpus whose game_seed.dat
    contains a value outside its own declared seed_ranges."""
    from build_memmap_corpus import build_memmap_corpus

    teacher_dir = tmp_path / "teacher"
    teacher_dir.mkdir()
    # A row with seed 999999999 -- outside [6_190_000_000, 6_200_000_000).
    _write_synthetic_shard(teacher_dir / "shard0.npz", game_seed=np.array([999_999_999, 999_999_999]))
    corpus_dir = tmp_path / "corpus"
    build_memmap_corpus(teacher_dir, corpus_dir, progress_every=0)
    with pytest.raises(SystemExit, match="outside the declared seed_ranges"):
        verify_anchor_corpus(corpus_dir, [(6_190_000_000, 6_200_000_000)], None)


# ---------------------------------------------------------------------------
# manifest append: longitudinal series semantics
# ---------------------------------------------------------------------------


def test_append_anchor_manifest_entry_is_longitudinal_append_only(tmp_path):
    manifest_path = tmp_path / "anchor_manifest.json"
    append_anchor_manifest_entry(manifest_path, {"name": "anchor_r7", "row_count": 1}, force=False)
    append_anchor_manifest_entry(manifest_path, {"name": "anchor_gen4", "row_count": 2}, force=False)
    manifest = load_anchor_manifest(manifest_path)
    assert [a["name"] for a in manifest["anchors"]] == ["anchor_r7", "anchor_gen4"]


def test_append_anchor_manifest_entry_refuses_duplicate_name_without_force(tmp_path):
    manifest_path = tmp_path / "anchor_manifest.json"
    append_anchor_manifest_entry(manifest_path, {"name": "anchor_r7", "row_count": 1}, force=False)
    with pytest.raises(SystemExit, match="already exists"):
        append_anchor_manifest_entry(manifest_path, {"name": "anchor_r7", "row_count": 99}, force=False)
    # force replaces the ONE entry, not append a duplicate
    append_anchor_manifest_entry(manifest_path, {"name": "anchor_r7", "row_count": 99}, force=True)
    manifest = load_anchor_manifest(manifest_path)
    assert len(manifest["anchors"]) == 1
    assert manifest["anchors"][0]["row_count"] == 99


# ---------------------------------------------------------------------------
# --hard-anchor stub (CAT-26 R9): documented, tested against synthetic tags,
# fails loudly against untagged (today's real) shards.
# ---------------------------------------------------------------------------


def test_hard_anchor_tag_filter_raises_not_yet_populated_when_column_absent(tmp_path):
    teacher_dir = tmp_path / "teacher"
    teacher_dir.mkdir()
    _write_synthetic_shard(teacher_dir / "shard0.npz", game_seed=np.array([1, 1, 2, 2]))
    with pytest.raises(SystemExit, match="not present in ANY"):
        filter_shards_by_tags(
            [teacher_dir / "shard0.npz"], "outcome_vs_external", ["loss_vs_catanatron_value"],
            tmp_path / "staging",
        )


def test_hard_anchor_tag_filter_works_against_synthetic_tagged_shard(tmp_path):
    teacher_dir = tmp_path / "teacher"
    teacher_dir.mkdir()
    tags = np.array(["loss_vs_catanatron_value", "loss_vs_catanatron_value", "", ""])
    _write_synthetic_shard(
        teacher_dir / "shard0.npz", game_seed=np.array([1, 1, 2, 2]), tag=tags,
    )
    stats = filter_shards_by_tags(
        [teacher_dir / "shard0.npz"], "outcome_vs_external", ["loss_vs_catanatron_value"],
        tmp_path / "staging",
    )
    assert stats["rows_out"] == 2


def test_build_anchor_corpus_hard_anchor_mode_end_to_end(tmp_path):
    teacher_dir = tmp_path / "teacher"
    teacher_dir.mkdir()
    tags = np.array(["loss_vs_catanatron_value", "loss_vs_catanatron_value", "", ""])
    _write_synthetic_shard(
        teacher_dir / "shard0.npz", game_seed=np.array([1, 1, 2, 2]), tag=tags,
    )
    entry = build_anchor_corpus(
        source_roots=[teacher_dir],
        anchor_name="anchor_hard_v1",
        anchor_type="external_hard",
        out_root=tmp_path / "anchors",
        seed_ranges=None,
        exclude_seed_ranges=None,
        hard_anchor_tag_column="outcome_vs_external",
        hard_anchor_tag_values=["loss_vs_catanatron_value"],
        force=False,
    )
    assert entry["row_count"] == 2
    assert entry["anchor_type"] == "external_hard"
