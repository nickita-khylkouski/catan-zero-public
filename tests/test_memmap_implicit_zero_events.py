"""Lossless file-free storage for disabled event-history features."""
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

from build_memmap_corpus import build_memmap_corpus  # type: ignore  # noqa: E402
from train_bc import MemmapCorpus  # type: ignore  # noqa: E402


def _write_shard(path: Path, *, seed: int, live_event: bool = False) -> None:
    n = 3
    event_tokens = np.zeros((n, 64, 41), dtype=np.float16)
    event_mask = np.zeros((n, 64), dtype=np.bool_)
    if live_event:
        event_tokens[-1, -1, 0] = 1
        event_mask[-1, -1] = True
    np.savez(
        path,
        obs=np.zeros((n, 4), dtype=np.float16),
        legal_action_ids=np.zeros((n, 2), dtype=np.int16),
        legal_action_context=np.zeros((n, 2, 1), dtype=np.float16),
        action_taken=np.zeros(n, dtype=np.int16),
        game_seed=np.full(n, seed, dtype=np.int64),
        event_tokens=event_tokens,
        event_target_ids=np.full((n, 64, 4), -1, dtype=np.int16),
        event_mask=event_mask,
    )


def _teacher_dir(tmp_path: Path, *, live_second: bool = False) -> Path:
    teacher = tmp_path / "teacher"
    teacher.mkdir()
    _write_shard(teacher / "shard0.npz", seed=10)
    _write_shard(teacher / "shard1.npz", seed=11, live_event=live_second)
    (teacher / "manifest.json").write_text(
        json.dumps({"shards": ["shard0.npz", "shard1.npz"]}), encoding="utf-8"
    )
    return teacher


def test_implicit_zero_events_round_trip_without_data_files(tmp_path):
    teacher = _teacher_dir(tmp_path)
    out = tmp_path / "corpus"
    meta = build_memmap_corpus(
        teacher, out, progress_every=0, omit_zero_events=True
    )

    assert meta["schema"] == "memmap_corpus_v2"
    assert meta["implicit_zero_columns"] == ["event_mask", "event_tokens"]
    assert meta["implicit_zero_bytes_saved_per_row"] == 64 + 64 * 41 * 2
    assert not (out / "event_tokens.dat").exists()
    assert not (out / "event_mask.dat").exists()
    # Unrelated event metadata remains losslessly stored.
    assert (out / "event_target_ids.dat").exists()

    corpus = MemmapCorpus(out)
    indices = np.array([5, 0, 3], dtype=np.int64)
    tokens = corpus["event_tokens"][indices]
    mask = corpus["event_mask"][indices]
    assert tokens.shape == (3, 64, 41)
    assert tokens.dtype == np.float16
    assert mask.shape == (3, 64)
    assert mask.dtype == np.bool_
    assert not np.any(tokens)
    assert not np.any(mask)
    assert corpus["event_tokens"][0].shape == (64, 41)
    assert corpus["event_mask"][0].shape == (64,)


def test_implicit_zero_column_matches_numpy_row_index_semantics(tmp_path):
    teacher = _teacher_dir(tmp_path)
    out = tmp_path / "corpus"
    build_memmap_corpus(teacher, out, progress_every=0, omit_zero_events=True)
    column = MemmapCorpus(out)["event_tokens"]

    assert column[-1].shape == (64, 41)
    assert column[1:6:2].shape == (3, 64, 41)
    assert column[::-2].shape == (3, 64, 41)
    assert column[np.asarray([[0, 1], [5, 2]], dtype=np.int64)].shape == (
        2,
        2,
        64,
        41,
    )
    assert column[np.asarray([True, False, True, False, False, True])].shape == (
        3,
        64,
        41,
    )
    assert column[True].shape == (1, 6, 64, 41)
    assert column[False].shape == (0, 6, 64, 41)
    assert column[[]].shape == (0, 64, 41)
    assert column[[[], []]].shape == (2, 0, 64, 41)

    with pytest.raises(IndexError, match="out of bounds"):
        column[6]
    with pytest.raises(IndexError, match="out of bounds"):
        column[-7]
    with pytest.raises(IndexError, match="out of bounds"):
        column[np.asarray([0, 6])]
    with pytest.raises(IndexError, match="boolean index did not match"):
        column[np.asarray([True, False])]
    with pytest.raises(IndexError, match="integer or boolean"):
        column[np.asarray([1.0])]

    materialized = column[np.asarray([0, 1])]
    materialized[0, 0, 0] = 9
    assert not np.any(column[np.asarray([0, 1])])


def test_default_v1_still_stores_event_files(tmp_path):
    teacher = _teacher_dir(tmp_path)
    out = tmp_path / "corpus"
    meta = build_memmap_corpus(teacher, out, progress_every=0)

    assert meta["schema"] == "memmap_corpus_v1"
    assert meta["implicit_zero_columns"] == []
    assert meta["implicit_zero_bytes_saved_per_row"] == 0
    assert (out / "event_tokens.dat").stat().st_size == 6 * 64 * 41 * 2
    assert (out / "event_mask.dat").stat().st_size == 6 * 64
    corpus = MemmapCorpus(out)
    assert not np.any(corpus["event_tokens"][np.array([1, 4])])


def test_implicit_zero_events_fails_closed_on_any_live_source_row(tmp_path):
    teacher = _teacher_dir(tmp_path, live_second=True)
    out = tmp_path / "corpus"

    with pytest.raises(SystemExit, match="found live/non-zero event data"):
        build_memmap_corpus(
            teacher, out, progress_every=0, omit_zero_events=True
        )
    assert not (out / "corpus_meta.json").exists()


def test_implicit_zero_events_requires_both_source_columns(tmp_path):
    teacher = tmp_path / "teacher"
    teacher.mkdir()
    np.savez(
        teacher / "shard.npz",
        obs=np.zeros((1, 4), dtype=np.float16),
        legal_action_ids=np.zeros((1, 1), dtype=np.int16),
        legal_action_context=np.zeros((1, 1, 1), dtype=np.float16),
        action_taken=np.zeros(1, dtype=np.int16),
        event_tokens=np.zeros((1, 64, 41), dtype=np.float16),
    )

    with pytest.raises(SystemExit, match=r"missing=\['event_mask'\]"):
        build_memmap_corpus(
            teacher, tmp_path / "corpus", progress_every=0, omit_zero_events=True
        )


@pytest.mark.parametrize("tamper", ["missing_pair", "nonzero_fill"])
def test_v2_metadata_fails_closed_on_invalid_implicit_zero_contract(
    tmp_path, tamper
):
    teacher = _teacher_dir(tmp_path)
    out = tmp_path / "corpus"
    build_memmap_corpus(teacher, out, progress_every=0, omit_zero_events=True)
    meta_path = out / "corpus_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    if tamper == "missing_pair":
        del meta["columns"]["event_mask"]
        meta["implicit_zero_columns"] = ["event_tokens"]
    else:
        meta["columns"]["event_mask"]["fill"] = 1
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(SystemExit, match="requires exactly|fill=0"):
        MemmapCorpus(out)
