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


def test_fixed_and_implicit_constant_columns_share_decoded_schema() -> None:
    module = _module()
    fixed = {"kind": "fixed", "dtype": "<f2", "inner_shape": [64, 41]}
    implicit = {
        "kind": "implicit_constant",
        "dtype": "<f2",
        "inner_shape": [64, 41],
        "fill": 0,
    }
    assert module._semantic_column_schema(fixed) == (  # noqa: SLF001
        module._semantic_column_schema(implicit)  # noqa: SLF001
    )


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
    current = _Corpus(0, 2, legal_width=2)
    current._eager.update(
        {
            "legal_action_ids": np.asarray([[1, -1], [1, 2]], dtype=np.int16),
            "policy_weight_multiplier": np.asarray([0.0, 1.0], dtype=np.float32),
            "is_forced": np.asarray([True, False]),
            "used_full_search": np.asarray([False, True]),
            "root_value": np.asarray([0.2, 0.4], dtype=np.float32),
            "root_value_mask": np.asarray([True, True]),
            "root_prior_value": np.asarray([0.1, 0.2], dtype=np.float32),
            "root_prior_value_mask": np.asarray([True, True]),
            "afterstate_target": np.asarray(
                [[0.1, np.nan], [0.2, 0.3]], dtype=np.float32
            ),
            "afterstate_target_mask": np.asarray(
                [[True, False], [True, True]], dtype=np.bool_
            ),
            "simulations_used": np.asarray([128, 128], dtype=np.int32),
            "aux_subgoal_target_version": np.asarray([1, 1], dtype=np.uint8),
            "aux_vp_in_n": np.asarray([0.5, 1.5], dtype=np.float32),
        }
    )
    current._columns.update(
        {
            "legal_action_ids": {"kind": "fixed", "dtype": "int16", "inner_shape": [2]},
            "policy_weight_multiplier": {
                "kind": "fixed",
                "dtype": "float32",
                "inner_shape": [],
            },
            "is_forced": {"kind": "fixed", "dtype": "bool", "inner_shape": []},
            "used_full_search": {"kind": "fixed", "dtype": "bool", "inner_shape": []},
            "root_value": {"kind": "fixed", "dtype": "float32", "inner_shape": []},
            "root_value_mask": {"kind": "fixed", "dtype": "bool", "inner_shape": []},
            "root_prior_value": {
                "kind": "fixed",
                "dtype": "float32",
                "inner_shape": [],
            },
            "root_prior_value_mask": {
                "kind": "fixed",
                "dtype": "bool",
                "inner_shape": [],
            },
            "afterstate_target": {
                "kind": "ragged2d",
                "dtype": "float32",
                "fill": float("nan"),
            },
            "afterstate_target_mask": {
                "kind": "ragged2d",
                "dtype": "bool",
                "fill": False,
            },
            "simulations_used": {
                "kind": "fixed",
                "dtype": "int32",
                "inner_shape": [],
            },
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
    old = _Corpus(2, 2, legal_width=2)
    old._eager.update(
        {
            "legal_action_ids": np.asarray([[3, -1], [3, 4]], dtype=np.int16),
            "policy_weight_multiplier": np.asarray([0.0, 1.0], dtype=np.float32),
        }
    )
    old._columns.update(
        {
            "legal_action_ids": {"kind": "fixed", "dtype": "int16", "inner_shape": [2]},
            "policy_weight_multiplier": {
                "kind": "fixed",
                "dtype": "float32",
                "inner_shape": [],
            },
        }
    )

    mixed = module.ConcatMemmapCorpus([current, old])

    assert np.array_equal(mixed["is_forced"][:], [True, False, True, False])
    assert np.array_equal(mixed["used_full_search"][:], [False, True, False, True])
    assert np.allclose(mixed["root_value"][:], [0.2, 0.4, 0.0, 0.0])
    assert np.array_equal(mixed["root_value_mask"][:], [True, True, False, False])
    np.testing.assert_allclose(
        mixed["root_prior_value"][:],
        [0.1, 0.2, np.nan, np.nan],
        equal_nan=True,
    )
    assert np.array_equal(mixed["root_prior_value_mask"][:], [True, True, False, False])
    assert np.allclose(
        mixed["afterstate_target"][:],
        [[0.1, np.nan], [0.2, 0.3], [np.nan, np.nan], [np.nan, np.nan]],
        equal_nan=True,
    )
    assert np.array_equal(
        mixed["afterstate_target_mask"][:],
        [[True, False], [True, True], [False, False], [False, False]],
    )
    assert np.array_equal(mixed["simulations_used"][:], [128, 128, 0, 0])
    assert mixed["afterstate_target"][3].shape == (2,)
    assert mixed["afterstate_target"][[3, 0]].shape == (2, 2)
    assert mixed["afterstate_target"].dtype == np.dtype(np.float32)
    assert mixed["afterstate_target_mask"].dtype == np.dtype(np.bool_)
    assert mixed["simulations_used"].dtype == np.dtype(np.int32)
    assert np.array_equal(mixed["aux_subgoal_target_version"][:], [1, 1, 0, 0])
    assert np.array_equal(
        mixed["aux_vp_in_n"][:], [0.5, 1.5, np.nan, np.nan], equal_nan=True
    )
    assert mixed.synthesized_columns_by_component == {
        1: (
            "afterstate_target",
            "afterstate_target_mask",
            "aux_subgoal_target_version",
            "aux_vp_in_n",
            "is_forced",
            "root_prior_value",
            "root_prior_value_mask",
            "root_value",
            "root_value_mask",
            "simulations_used",
            "used_full_search",
        )
    }


def test_old_component_gets_unknown_opponent_provenance_not_fabricated_identity():
    module = _module()
    current = _Corpus(0, 2)
    current._eager.update(
        {
            "is_pool_game": np.asarray([False, True]),
            "opponent_version": np.asarray([-1, 7], dtype=np.int32),
            "opponent_tag": np.asarray(["", "recent_history"]),
            "opponent_checkpoint_md5": np.asarray(["", "deadbeef"]),
            "opponent_type": np.asarray(["", ""]),
            "opponent_provenance_present": np.asarray([False, True]),
            "training_source_category": np.asarray(
                ["current_producer", "recent_history"]
            ),
            "training_source_category_verified": np.asarray([True, True]),
        }
    )
    for name, value in tuple(current._eager.items()):
        if name not in module.SYNTHESIZABLE_COLUMNS:
            continue
        array = np.asarray(value)
        current._columns[name] = (
            {
                "kind": "string",
                "dtype": "int32",
                "categories": sorted(set(array.tolist())),
            }
            if array.dtype.kind in {"U", "S", "O"}
            else {"kind": "fixed", "dtype": array.dtype.str, "inner_shape": []}
        )
    old = _Corpus(2, 2)
    mixed = module.ConcatMemmapCorpus([current, old])

    np.testing.assert_array_equal(
        mixed["training_source_category"][:],
        ["current_producer", "recent_history", "", ""],
    )
    np.testing.assert_array_equal(
        mixed["training_source_category_verified"][:],
        [True, True, False, False],
    )
    np.testing.assert_array_equal(
        mixed["opponent_provenance_present"][:],
        [False, True, False, False],
    )
    np.testing.assert_array_equal(mixed["opponent_version"][:], [-1, 7, -1, -1])
    assert set(mixed.synthesized_columns_by_component[1]) >= {
        "is_pool_game",
        "opponent_version",
        "opponent_tag",
        "opponent_checkpoint_md5",
        "opponent_type",
        "opponent_provenance_present",
        "training_source_category",
        "training_source_category_verified",
    }


def test_legacy_component_gets_explicit_absent_restart_provenance():
    module = _module()
    restart = _Corpus(0, 2)
    restart._eager.update(
        {
            "restart_provenance_present": np.asarray([True, True]),
            "start_mode": np.asarray(["archived_public_state"] * 2),
            "start_bucket": np.asarray(["opening"] * 2),
            "archived_game_seed": np.asarray([123, 123], dtype=np.int64),
            "archived_decision_index": np.asarray([4, 4], dtype=np.int64),
            "restart_select_seed": np.asarray([700_001, 700_001], dtype=np.int64),
        }
    )
    for name, value in tuple(restart._eager.items()):
        if name not in module.SYNTHESIZABLE_COLUMNS:
            continue
        array = np.asarray(value)
        restart._columns[name] = (
            {"kind": "string", "dtype": "int32"}
            if array.dtype.kind in {"U", "S", "O"}
            else {"kind": "fixed", "dtype": array.dtype.str, "inner_shape": []}
        )
    legacy = _Corpus(2, 2)

    mixed = module.ConcatMemmapCorpus([restart, legacy])

    assert mixed["restart_provenance_present"][:].tolist() == [
        True,
        True,
        False,
        False,
    ]
    assert mixed["start_mode"][:].tolist() == [
        "archived_public_state",
        "archived_public_state",
        "legacy_unknown",
        "legacy_unknown",
    ]
    assert mixed["archived_game_seed"][:].tolist() == [123, 123, -1, -1]
    assert set(mixed.synthesized_columns_by_component[1]) >= {
        "restart_provenance_present",
        "start_mode",
        "start_bucket",
        "archived_game_seed",
        "archived_decision_index",
        "restart_select_seed",
    }


@pytest.mark.parametrize("missing", ["afterstate_target", "afterstate_target_mask"])
def test_afterstate_target_and_mask_must_be_an_atomic_pair(missing: str) -> None:
    module = _module()
    complete = _Corpus(0, 2)
    complete._eager.update(
        {
            "afterstate_target": np.zeros((2, 3), dtype=np.float32),
            "afterstate_target_mask": np.ones((2, 3), dtype=np.bool_),
        }
    )
    complete._columns.update(
        {
            "afterstate_target": {
                "kind": "ragged2d",
                "dtype": "float32",
                "fill": float("nan"),
            },
            "afterstate_target_mask": {
                "kind": "ragged2d",
                "dtype": "bool",
                "fill": False,
            },
        }
    )
    broken = _Corpus(2, 2)
    present = (
        "afterstate_target_mask"
        if missing == "afterstate_target"
        else "afterstate_target"
    )
    broken._eager[present] = complete._eager[present].copy()
    broken._columns[present] = dict(complete._columns[present])

    with pytest.raises(SystemExit, match="incomplete afterstate target/mask pair"):
        module.ConcatMemmapCorpus([complete, broken])


def _add_search_evidence(module, corpus: _Corpus, *, include_prior: bool = True) -> None:
    width = int(corpus.legal_width)
    rows = int(corpus.row_count)
    corpus._eager.update(
        {
            "search_evidence_version": np.full(
                rows, 2 if include_prior else 1, dtype=np.uint8
            ),
            "search_evidence_mask": np.full(rows, True, dtype=np.bool_),
            "search_evidence_offsets": np.stack(
                (
                    np.arange(rows, dtype=np.int64),
                    np.arange(1, rows + 1, dtype=np.int64),
                ),
                axis=1,
            ),
            "search_visit_counts_flat": np.ones((rows, width), dtype=np.uint16),
            "search_completed_q_flat": np.zeros((rows, width), dtype=np.float32),
        }
    )
    corpus._columns.update(
        {
            "search_evidence_version": {
                "kind": "fixed",
                "dtype": np.dtype(np.uint8).str,
                "inner_shape": [],
            },
            "search_evidence_mask": {
                "kind": "fixed",
                "dtype": np.dtype(np.bool_).str,
                "inner_shape": [],
            },
            "search_evidence_offsets": {
                "kind": "row_offsets",
                "dtype": np.dtype(np.int64).str,
            },
            "search_visit_counts_flat": {
                "kind": "independent_ragged1d",
                "dtype": np.dtype(np.uint16).str,
                "fill": 0,
                "offsets": "search_evidence_offsets",
            },
            "search_completed_q_flat": {
                "kind": "independent_ragged1d",
                "dtype": np.dtype(np.float32).str,
                "fill": float("nan"),
                "offsets": "search_evidence_offsets",
            },
        }
    )
    if include_prior:
        corpus._eager[module.SEARCH_EVIDENCE_PRIOR_COLUMN] = np.full(
            (rows, width), 1.0 / width, dtype=np.float32
        )
        corpus._columns[module.SEARCH_EVIDENCE_PRIOR_COLUMN] = {
            "kind": "independent_ragged1d",
            "dtype": np.dtype(np.float32).str,
            "fill": float("nan"),
            "offsets": "search_evidence_offsets",
        }
    corpus.meta["search_evidence"] = {
        "schema": (
            module.SEARCH_EVIDENCE_SCHEMA_V2
            if include_prior
            else module.SEARCH_EVIDENCE_SCHEMA_V1
        )
    }


@pytest.mark.parametrize(
    "missing",
    [
        "search_evidence_version",
        "search_evidence_mask",
        "search_evidence_offsets",
        "search_visit_counts_flat",
        "search_completed_q_flat",
    ],
)
def test_search_evidence_base_bundle_must_be_atomic(missing: str) -> None:
    module = _module()
    complete = _Corpus(0, 2)
    _add_search_evidence(module, complete)
    broken = _Corpus(2, 2)
    _add_search_evidence(module, broken)
    broken._eager.pop(missing)
    broken._columns.pop(missing)

    with pytest.raises(SystemExit, match="incomplete search evidence bundle"):
        module.ConcatMemmapCorpus([complete, broken])


def test_search_prior_requires_base_evidence_bundle() -> None:
    module = _module()
    complete = _Corpus(0, 2)
    _add_search_evidence(module, complete)
    prior_only = _Corpus(2, 2)
    prior_only._eager[module.SEARCH_EVIDENCE_PRIOR_COLUMN] = np.ones(
        (2, 3), dtype=np.float32
    )
    prior_only._columns[module.SEARCH_EVIDENCE_PRIOR_COLUMN] = dict(
        complete._columns[module.SEARCH_EVIDENCE_PRIOR_COLUMN]
    )

    with pytest.raises(SystemExit, match="search prior evidence without the base"):
        module.ConcatMemmapCorpus([complete, prior_only])


def test_v1_search_evidence_synthesizes_only_missing_prior() -> None:
    module = _module()
    v2 = _Corpus(0, 2)
    _add_search_evidence(module, v2)
    v1 = _Corpus(2, 2)
    _add_search_evidence(module, v1, include_prior=False)

    mixed = module.ConcatMemmapCorpus([v2, v1])

    np.testing.assert_allclose(
        mixed[module.SEARCH_EVIDENCE_PRIOR_COLUMN][:2],
        np.full((2, 3), 1.0 / 3, dtype=np.float32),
    )
    assert bool(
        np.all(np.isnan(mixed[module.SEARCH_EVIDENCE_PRIOR_COLUMN][2:]))
    )
    assert mixed.synthesized_columns_by_component == {
        1: (module.SEARCH_EVIDENCE_PRIOR_COLUMN,)
    }


def test_v2_search_evidence_without_exact_prior_fails_closed() -> None:
    module = _module()
    complete = _Corpus(0, 2)
    _add_search_evidence(module, complete)
    malformed_v2 = _Corpus(2, 2)
    _add_search_evidence(module, malformed_v2)
    malformed_v2._eager.pop(module.SEARCH_EVIDENCE_PRIOR_COLUMN)
    malformed_v2._columns.pop(module.SEARCH_EVIDENCE_PRIOR_COLUMN)

    with pytest.raises(SystemExit, match="v2 search evidence is missing its exact prior"):
        module.ConcatMemmapCorpus([complete, malformed_v2])


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


def test_authenticated_adapter_version_backfills_only_legacy_component():
    module = _module()
    version = "rust_entity_adapter_v2_land_topology_ports_maritime"
    current = _Corpus(0, 2)
    current._eager["adapter_version"] = np.full(2, version, dtype="U64")
    current._columns["adapter_version"] = {
        "kind": "string",
        "dtype": "int32",
        "categories": [version],
    }
    legacy = _Corpus(2, 3)

    with pytest.raises(SystemExit, match="unsupported missing columns"):
        module.ConcatMemmapCorpus([current, legacy])

    mixed = module.ConcatMemmapCorpus(
        [current, legacy], component_adapter_versions=[version, version]
    )
    assert mixed["adapter_version"].present_values() == {version}
    assert np.asarray(mixed["adapter_version"]).tolist() == [version] * 5

    with pytest.raises(SystemExit, match="descriptor differs from stored rows"):
        module.ConcatMemmapCorpus(
            [current, legacy], component_adapter_versions=["wrong", "wrong"]
        )


def test_adapter_version_backfill_refuses_current_semantics_without_stored_rows():
    module = _module()
    current = "rust_entity_adapter_v3_structured_action_resources"

    with pytest.raises(
        SystemExit, match="may only use the explicit legacy missing-metadata mapping"
    ):
        module.ConcatMemmapCorpus(
            [_Corpus(0, 2), _Corpus(2, 3)],
            component_adapter_versions=[current, current],
        )


def test_invalid_indices_match_numpy_fail_closed_behavior(composite):
    mixed, _, _ = composite
    with pytest.raises(IndexError):
        mixed["row"][9]
    with pytest.raises(IndexError):
        mixed["row"][np.zeros(8, dtype=np.bool_)]
    with pytest.raises(IndexError):
        mixed["row"][np.asarray([1.5])]
