from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "tools" / "mixed_memmap_corpus.py"
    spec = importlib.util.spec_from_file_location("mixed_memmap_corpus", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _TrackedColumn:
    def __init__(self, array: np.ndarray):
        self.array = np.asarray(array)
        self.shape = self.array.shape
        self.ndim = self.array.ndim
        self.dtype = self.array.dtype
        self.requests: list[np.ndarray] = []

    def __getitem__(self, index):
        self.requests.append(np.asarray(index).copy())
        return self.array[index]


class _Corpus:
    def __init__(self, start: int, rows: int, *, legal_width: int = 3):
        ids = np.arange(start, start + rows, dtype=np.int64)
        self._eager = {
            "row": ids,
            "text": np.asarray([f"row-{value}" for value in ids]),
        }
        self._lazy = {
            "matrix": _TrackedColumn(
                np.stack((ids, ids + 100), axis=1).astype(np.float32)
            )
        }
        self._columns = {
            "row": {"kind": "fixed", "dtype": "int64", "inner_shape": []},
            "text": {
                "kind": "string",
                "dtype": "int32",
                "categories": [f"part-{start}"],
            },
            "matrix": {
                "kind": "fixed",
                "dtype": "float32",
                "inner_shape": [2],
            },
        }
        self.row_count = rows
        self.legal_width = legal_width
        self.meta = {"shard_count": 1}
        self.stats = {"start": start}

    def keys(self):
        return [*self._eager, *self._lazy]

    def __getitem__(self, key):
        return self._eager[key] if key in self._eager else self._lazy[key]


@pytest.fixture
def composite():
    module = _module()
    left = _Corpus(0, 4)
    right = _Corpus(4, 5)
    return module.ConcatMemmapCorpus([left, right]), left, right


@pytest.mark.parametrize(
    "index",
    [
        np.asarray([7, 0, 4, 0, 8, 3]),
        slice(2, 7),
        np.asarray([True, False, False, True, False, True, False, False, True]),
        np.asarray([], dtype=np.int64),
        [],
    ],
)
def test_arbitrary_global_indexing_matches_physical_concatenation(composite, index):
    mixed, left, right = composite
    for key in mixed.keys():
        expected = np.concatenate(
            (
                np.asarray(left[key].array if key == "matrix" else left[key]),
                np.asarray(right[key].array if key == "matrix" else right[key]),
            )
        )
        assert np.array_equal(mixed[key][index], expected[index])


def test_scalar_negative_and_cross_boundary_order(composite):
    mixed, _, _ = composite
    assert mixed["row"][0] == 0
    assert mixed["row"][-1] == 8
    assert np.array_equal(mixed["row"][[3, 4, -1, 4]], [3, 4, 8, 4])


def test_lazy_gather_reads_only_requested_component_rows(composite):
    mixed, left, right = composite
    got = mixed["matrix"][[5, 1, 8, 6]]
    assert np.array_equal(got[:, 0], [5, 1, 8, 6])
    assert np.array_equal(left._lazy["matrix"].requests[-1], [1])
    assert np.array_equal(right._lazy["matrix"].requests[-1], [1, 4, 2])
    assert "matrix" in mixed._lazy


def test_interface_and_component_order(composite):
    mixed, _, _ = composite
    assert len(mixed) == 9
    assert mixed.row_count == 9
    assert mixed.legal_width == 3
    assert set(mixed.keys()) == {"row", "text", "matrix"}
    assert "row" in mixed
    assert mixed.get("missing", "sentinel") == "sentinel"
    assert np.array_equal(np.asarray(mixed["row"]), np.arange(9))
    with pytest.raises(KeyError):
        mixed["missing"]


def test_schema_mismatch_fails_closed():
    module = _module()
    with pytest.raises(SystemExit, match="not schema-compatible"):
        module.ConcatMemmapCorpus([_Corpus(0, 2), _Corpus(2, 2, legal_width=4)])
    wrong = _Corpus(2, 2)
    wrong._columns["matrix"] = {"kind": "fixed", "dtype": "float32", "inner_shape": [3]}
    with pytest.raises(SystemExit, match="column 'matrix' differs"):
        module.ConcatMemmapCorpus([_Corpus(0, 2), wrong])


def test_known_optional_columns_are_synthesized_with_safe_semantics():
    module = _module()
    current = _Corpus(0, 2)
    current._eager.update(
        {
            "legal_action_ids": np.asarray([[1, -1], [1, 2]], dtype=np.int16),
            "policy_weight_multiplier": np.asarray([0.0, 1.0], dtype=np.float32),
            "is_forced": np.asarray([True, False]),
            "used_full_search": np.asarray([False, True]),
            "root_value": np.asarray([0.2, 0.4], dtype=np.float32),
            "root_value_mask": np.asarray([True, True]),
            "aux_subgoal_target_version": np.asarray([1, 1], dtype=np.uint8),
            "aux_vp_in_n": np.asarray([0.5, 1.5], dtype=np.float32),
        }
    )
    current._columns.update(
        {
            "legal_action_ids": {"kind": "fixed", "dtype": "int16", "inner_shape": [2]},
            "policy_weight_multiplier": {"kind": "fixed", "dtype": "float32", "inner_shape": []},
            "is_forced": {"kind": "fixed", "dtype": "bool", "inner_shape": []},
            "used_full_search": {"kind": "fixed", "dtype": "bool", "inner_shape": []},
            "root_value": {"kind": "fixed", "dtype": "float32", "inner_shape": []},
            "root_value_mask": {"kind": "fixed", "dtype": "bool", "inner_shape": []},
            "aux_subgoal_target_version": {
                "kind": "fixed",
                "dtype": "uint8",
                "inner_shape": [],
            },
            "aux_vp_in_n": {
                "kind": "fixed",
                "dtype": "float32",
                "inner_shape": [],
            },
        }
    )
    old = _Corpus(2, 2)
    old._eager.update(
        {
            "legal_action_ids": np.asarray([[3, -1], [3, 4]], dtype=np.int16),
            "policy_weight_multiplier": np.asarray([0.0, 1.0], dtype=np.float32),
        }
    )
    old._columns.update(
        {
            "legal_action_ids": {"kind": "fixed", "dtype": "int16", "inner_shape": [2]},
            "policy_weight_multiplier": {"kind": "fixed", "dtype": "float32", "inner_shape": []},
        }
    )

    mixed = module.ConcatMemmapCorpus([current, old])

    assert np.array_equal(mixed["is_forced"][:], [True, False, True, False])
    assert np.array_equal(mixed["used_full_search"][:], [False, True, False, True])
    assert np.allclose(mixed["root_value"][:], [0.2, 0.4, 0.0, 0.0])
    assert np.array_equal(mixed["root_value_mask"][:], [True, True, False, False])
    assert np.array_equal(mixed["aux_subgoal_target_version"][:], [1, 1, 0, 0])
    assert np.array_equal(
        mixed["aux_vp_in_n"][:], [0.5, 1.5, np.nan, np.nan], equal_nan=True
    )
    assert mixed.synthesized_columns_by_component == {
        1: (
            "aux_subgoal_target_version",
            "aux_vp_in_n",
            "is_forced",
            "root_value",
            "root_value_mask",
            "used_full_search",
        )
    }


def test_unknown_missing_column_still_fails_closed():
    module = _module()
    left = _Corpus(0, 2)
    right = _Corpus(2, 2)
    left._eager["unknown_future_target"] = np.zeros(2, dtype=np.float32)
    left._columns["unknown_future_target"] = {
        "kind": "fixed",
        "dtype": "float32",
        "inner_shape": [],
    }
    with pytest.raises(SystemExit, match="unsupported missing columns"):
        module.ConcatMemmapCorpus([left, right])


def test_invalid_indices_match_numpy_fail_closed_behavior(composite):
    mixed, _, _ = composite
    with pytest.raises(IndexError):
        mixed["row"][9]
    with pytest.raises(IndexError):
        mixed["row"][np.zeros(8, dtype=np.bool_)]
    with pytest.raises(IndexError):
        mixed["row"][np.asarray([1.5])]
