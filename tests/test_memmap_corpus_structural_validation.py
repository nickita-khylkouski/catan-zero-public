from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from catan_zero.rl import memmap_corpus as memmap_module
from catan_zero.rl.memmap_corpus import MemmapCorpus


def _write_corpus(root: Path) -> Path:
    root.mkdir()
    meta = {
        "schema": "memmap_corpus_v1",
        "row_count": 3,
        "flat_count": 4,
        "legal_width": 2,
        "columns": {
            "legal_action_ids": {
                "kind": "ragged2d",
                "dtype": "<i2",
                "fill": -1,
            },
            "phase": {
                "kind": "string",
                "categories": ["opening", "play"],
            },
            "weight": {
                "kind": "fixed",
                "dtype": "<f4",
                "inner_shape": [],
            },
            "search_offsets": {
                "kind": "row_offsets",
                "dtype": "<i8",
            },
            "search_values": {
                "kind": "independent_ragged1d",
                "dtype": "<f4",
                "fill": 0.0,
                "offsets": "search_offsets",
            },
        },
        "stats": {},
    }
    (root / "corpus_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    np.asarray([0, 2, 3, 4], dtype=np.int64).tofile(root / "row_offsets.dat")
    np.asarray([2, 5, 7, 9], dtype=np.int16).tofile(
        root / "legal_action_ids.dat"
    )
    np.asarray([0, 1, 0], dtype=np.int32).tofile(root / "phase.codes.dat")
    np.asarray([1.0, 2.0, 3.0], dtype=np.float32).tofile(root / "weight.dat")
    np.asarray([0, 1, 1, 3], dtype=np.int64).tofile(
        root / "search_offsets.dat"
    )
    np.asarray([10.0, 20.0, 30.0], dtype=np.float32).tofile(
        root / "search_values.dat"
    )
    return root


def test_valid_offsets_stay_lazy_memmaps(tmp_path: Path) -> None:
    corpus = MemmapCorpus(_write_corpus(tmp_path / "corpus"))

    assert isinstance(corpus._offsets, np.memmap)
    assert isinstance(corpus["search_offsets"]._offsets, np.memmap)
    np.testing.assert_array_equal(
        corpus["legal_action_ids"][:],
        [[2, 5], [7, -1], [9, -1]],
    )


@pytest.mark.parametrize(
    ("offsets", "message"),
    [
        ([1, 2, 3, 4], "must start at 0"),
        ([-1, 1, 3, 4], "must start at 0"),
        ([0, 2, 1, 4], "offsets decrease"),
        ([0, 1, 2, 3], "final offset"),
        ([0, 2, 3, 5], "final offset"),
        ([0, 3, 3, 4], "exceeds legal_width"),
    ],
)
def test_shared_offset_corruption_fails_during_construction(
    tmp_path: Path,
    offsets: list[int],
    message: str,
) -> None:
    root = _write_corpus(tmp_path / "corpus")
    np.asarray(offsets, dtype=np.int64).tofile(root / "row_offsets.dat")

    with pytest.raises(SystemExit, match=message):
        MemmapCorpus(root)


def test_offset_decrease_across_validation_block_boundary_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _write_corpus(tmp_path / "corpus")
    np.asarray([0, 1, 2, 1], dtype=np.int64).tofile(root / "row_offsets.dat")
    monkeypatch.setattr(memmap_module, "_VALIDATION_BLOCK_ROWS", 2)

    with pytest.raises(SystemExit, match="offsets decrease at row 2"):
        MemmapCorpus(root)


def test_offset_decrease_cannot_hide_behind_int64_subtraction_overflow(
    tmp_path: Path,
) -> None:
    root = _write_corpus(tmp_path / "corpus")
    np.asarray(
        [0, np.iinfo(np.int64).max, np.iinfo(np.int64).min, 4],
        dtype=np.int64,
    ).tofile(root / "row_offsets.dat")

    with pytest.raises(SystemExit, match="offsets decrease at row 1"):
        MemmapCorpus(root)


@pytest.mark.parametrize(
    ("offsets", "message"),
    [
        ([1, 1, 1, 3], "must start at 0"),
        ([0, 2, 1, 3], "offsets decrease"),
        ([0, 3, 3, 3], "exceeds legal_width"),
        ([0, 1, 1, 2], "search_values.dat size"),
        ([0, 2, 2, 4], "search_values.dat size"),
    ],
)
def test_independent_offset_corruption_fails_during_construction(
    tmp_path: Path,
    offsets: list[int],
    message: str,
) -> None:
    root = _write_corpus(tmp_path / "corpus")
    np.asarray(offsets, dtype=np.int64).tofile(root / "search_offsets.dat")

    with pytest.raises(SystemExit, match=message):
        MemmapCorpus(root)


def test_independent_offsets_require_int64_schema(tmp_path: Path) -> None:
    root = _write_corpus(tmp_path / "corpus")
    meta_path = root / "corpus_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["columns"]["search_offsets"]["dtype"] = "<f8"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    np.asarray([0.0, 1.5, 1.5, 3.0], dtype=np.float64).tofile(
        root / "search_offsets.dat"
    )

    with pytest.raises(SystemExit, match="must use int64"):
        MemmapCorpus(root)


@pytest.mark.parametrize("bad_code", [-1, 2])
def test_categorical_code_bounds_fail_during_construction(
    tmp_path: Path,
    bad_code: int,
) -> None:
    root = _write_corpus(tmp_path / "corpus")
    np.asarray([0, bad_code, 1], dtype=np.int32).tofile(
        root / "phase.codes.dat"
    )

    with pytest.raises(SystemExit, match="categorical code"):
        MemmapCorpus(root)


def test_categorical_dictionary_must_be_a_sequence(tmp_path: Path) -> None:
    root = _write_corpus(tmp_path / "corpus")
    meta_path = root / "corpus_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["columns"]["phase"]["categories"] = "opening"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(SystemExit, match="one-dimensional sequence"):
        MemmapCorpus(root)


def test_empty_corpus_uses_empty_arrays_instead_of_invalid_zero_byte_memmaps(
    tmp_path: Path,
) -> None:
    root = _write_corpus(tmp_path / "corpus")
    meta_path = root / "corpus_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["row_count"] = 0
    meta["flat_count"] = 0
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    np.asarray([0], dtype=np.int64).tofile(root / "row_offsets.dat")
    np.asarray([0], dtype=np.int64).tofile(root / "search_offsets.dat")
    for filename in (
        "legal_action_ids.dat",
        "phase.codes.dat",
        "weight.dat",
        "search_values.dat",
    ):
        (root / filename).write_bytes(b"")

    corpus = MemmapCorpus(root)

    assert len(corpus) == 0
    assert corpus["phase"].shape == (0,)
    assert corpus["weight"].shape == (0,)
    assert corpus["legal_action_ids"].shape == (0, 2)


@pytest.mark.parametrize(
    "filename",
    [
        "row_offsets.dat",
        "legal_action_ids.dat",
        "phase.codes.dat",
        "weight.dat",
        "search_offsets.dat",
        "search_values.dat",
    ],
)
@pytest.mark.parametrize("mutation", ["short", "trailing"])
def test_payload_size_mismatch_fails_before_mapping(
    tmp_path: Path,
    filename: str,
    mutation: str,
) -> None:
    root = _write_corpus(tmp_path / "corpus")
    path = root / filename
    payload = path.read_bytes()
    path.write_bytes(payload[:-1] if mutation == "short" else payload + b"\0")

    with pytest.raises(SystemExit, match="size .* != expected"):
        MemmapCorpus(root)
