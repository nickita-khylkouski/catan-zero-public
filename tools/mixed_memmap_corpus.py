"""No-copy global row view over multiple compatible memmap corpora.

The composite owns no payload arrays. Each column maps global row indices to
the corresponding component column, gathers only those rows, and scatters them
back into the caller's original order. This lets the trainer's existing global
epoch permutation interleave corpora without rebuilding concatenated ``.dat``
files.
"""

from __future__ import annotations

import operator
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from catan_zero.rl.aux_subgoal_targets import (
    AUX_SUBGOAL_TARGET_VERSION_KEY,
    AUX_TARGET_KEYS,
)
from catan_zero.rl.entity_feature_adapter import (
    LEGACY_MISSING_CHECKPOINT_ADAPTER_VERSION,
)


# These columns were added after the original gen3 corpus was converted. Their
# absence has an exact, loss-safe interpretation that can be reconstructed from
# older required columns. Any other schema gap remains a hard error.
SYNTHESIZABLE_COLUMNS = frozenset(
    {
        "is_forced",
        "used_full_search",
        # Strict-future waves preserve the public decision taxonomy. Historical
        # replay predates it, so its only honest decoded value is unknown; never
        # infer a mandatory/normal label from action width after the fact.
        "decision_class",
        "root_value",
        "root_value_mask",
        # The sealed gen3 replay corpus predates preserved search-accounting
        # and one-ply afterstate columns.  Their absence has an exact neutral
        # meaning: no authenticated afterstate target and no recorded search
        # simulations.  `used_full_search` remains derived independently from
        # the historical policy-weight column, so this does not disable its
        # stored policy targets.
        "afterstate_target",
        "afterstate_target_mask",
        "simulations_used",
        "is_pool_game",
        "opponent_version",
        "opponent_tag",
        "opponent_checkpoint_md5",
        "opponent_type",
        "opponent_provenance_present",
        "training_source_category",
        "training_source_category_verified",
        *AUX_TARGET_KEYS,
        # Historical replay predates the strict-future auxiliary-target
        # contract.  Version 0 means unversioned/ineligible and is synthesized
        # lazily so those rows stay available for policy/value objectives.
        AUX_SUBGOAL_TARGET_VERSION_KEY,
    }
)


def _semantic_column_schema(schema: dict[str, Any]) -> tuple[Any, ...]:
    """Compatibility fields that affect the decoded NumPy column surface."""
    kind = schema.get("kind")
    common = (kind, np.dtype(schema.get("dtype", "float32")).str)
    if kind in {"fixed", "implicit_constant"}:
        # ``implicit_constant`` is a storage optimization for an ordinary
        # decoded fixed array.  Fresh meaningful event columns are physical
        # while legacy replay is authenticated implicit-zero; their decoded
        # tensor ABI is compatible when dtype/shape agree.
        return ("fixed", common[1], tuple(schema.get("inner_shape", ())))
    if kind == "ragged3d":
        return (*common, int(schema.get("feat", 0)), schema.get("fill"))
    if kind == "ragged2d":
        return (*common, schema.get("fill"))
    if kind == "string":
        # Category-code tables are component-local. MemmapCorpus has already
        # decoded them to strings, so category ordering is not structural.
        return (kind,)
    return common


def _normalize_global_index(index: Any, row_count: int) -> tuple[np.ndarray, bool]:
    """Return normalized non-negative indices and whether input was scalar."""
    if isinstance(index, slice):
        return np.arange(*index.indices(row_count), dtype=np.int64), False
    array = np.asarray(index)
    if array.ndim == 0:
        try:
            value = operator.index(index)
        except TypeError:
            try:
                value = operator.index(array.item())
            except (TypeError, ValueError) as error:
                raise IndexError(
                    "composite corpus row index must be an integer"
                ) from error
        if not -row_count <= value < row_count:
            raise IndexError(
                f"index {value} is out of bounds for axis 0 with size {row_count}"
            )
        if value < 0:
            value += row_count
        return np.asarray(value, dtype=np.int64), True
    if array.dtype.kind == "b":
        if array.ndim != 1 or array.shape[0] != row_count:
            raise IndexError(
                "boolean index did not match composite row axis; "
                f"axis has size {row_count} but mask shape is {array.shape}"
            )
        return np.flatnonzero(array).astype(np.int64, copy=False), False
    literal_empty = isinstance(index, list) and array.size == 0
    if array.dtype.kind not in {"i", "u"} and not literal_empty:
        raise IndexError("arrays used as indices must be of integer or boolean type")
    normalized = array.astype(np.int64, copy=True)
    if normalized.size:
        invalid = (normalized < -row_count) | (normalized >= row_count)
        if bool(np.any(invalid)):
            bad = int(normalized[invalid].flat[0])
            raise IndexError(
                f"index {bad} is out of bounds for axis 0 with size {row_count}"
            )
        normalized[normalized < 0] += row_count
    return normalized, False


class _ConcatColumn:
    """One decoded column spanning component corpora without payload copying."""

    def __init__(self, columns: Sequence[Any], row_counts: Sequence[int]):
        self._columns = tuple(columns)
        self._offsets = np.concatenate(
            (np.asarray([0], dtype=np.int64), np.cumsum(row_counts, dtype=np.int64))
        )
        self._n = int(self._offsets[-1])
        shapes = [tuple(column.shape) for column in self._columns]
        inner_shapes = {shape[1:] for shape in shapes}
        if len(inner_shapes) != 1:
            raise SystemExit(
                f"composite memmap column inner shapes differ: {sorted(inner_shapes)}"
            )
        self._inner_shape = shapes[0][1:]
        self.shape = (self._n, *self._inner_shape)
        self.ndim = len(self.shape)
        self.dtype = np.result_type(
            *(np.dtype(column.dtype) for column in self._columns)
        )
        self.supports_grouped_weights = all(
            callable(getattr(column, "grouped_weights", None))
            for column in self._columns
        )
        self.supports_value_counts = all(
            callable(getattr(column, "value_counts", None))
            for column in self._columns
        )

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: Any):
        indices, scalar = _normalize_global_index(index, self._n)
        if scalar:
            global_index = int(indices)
            part = int(np.searchsorted(self._offsets, global_index, side="right") - 1)
            return np.asarray(self._columns[part][global_index - self._offsets[part]])

        flat = indices.reshape(-1)
        output = np.empty((flat.size, *self._inner_shape), dtype=self.dtype)
        for part, column in enumerate(self._columns):
            selected = (flat >= self._offsets[part]) & (flat < self._offsets[part + 1])
            if not bool(np.any(selected)):
                continue
            local = flat[selected] - self._offsets[part]
            output[selected] = np.asarray(column[local])
        return output.reshape((*indices.shape, *self._inner_shape))

    def __array__(self, dtype=None, copy=None):
        array = self[np.arange(self._n, dtype=np.int64)]
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        if copy:
            array = array.copy()
        return array

    def grouped_weights(
        self, weights: np.ndarray, *, limit: int
    ) -> dict[str, dict[str, float | int]]:
        """Reduce concatenated dictionary columns without decoding all rows.

        String codebooks are component-local, so merge the small decoded
        category summaries rather than materialising a global Unicode array.
        Numeric/other columns reject this path and retain normal NumPy
        semantics through ``__array__``.
        """

        if not self.supports_grouped_weights:
            raise TypeError("concatenated column is not dictionary-encoded")
        values = np.asarray(weights)
        if values.ndim != 1 or values.shape[0] != self._n:
            raise ValueError("concatenated grouped-weight vector shape drift")
        merged: dict[str, dict[str, float | int]] = {}
        for part, column in enumerate(self._columns):
            start, stop = int(self._offsets[part]), int(self._offsets[part + 1])
            category_count = len(getattr(column, "categories", ()))
            report = column.grouped_weights(
                values[start:stop], limit=max(int(limit), category_count)
            )
            for category, row in report.items():
                target = merged.setdefault(
                    str(category), {"raw_samples": 0, "weight_sum": 0.0}
                )
                target["raw_samples"] = int(target["raw_samples"]) + int(
                    row["raw_samples"]
                )
                target["weight_sum"] = float(target["weight_sum"]) + float(
                    row["weight_sum"]
                )
        ordered = sorted(
            merged.items(), key=lambda item: (-int(item[1]["raw_samples"]), item[0])
        )[: int(limit)]
        return {
            category: {
                "raw_samples": int(row["raw_samples"]),
                "weight_sum": float(row["weight_sum"]),
                "mean_weight": float(row["weight_sum"])
                / max(int(row["raw_samples"]), 1),
            }
            for category, row in ordered
        }

    def value_counts(self, index: Any = None) -> dict[str, int]:
        """Count concatenated dictionary labels without global decoding."""

        if not self.supports_value_counts:
            raise TypeError("concatenated column is not dictionary-encoded")
        if index is None:
            indices = None
        else:
            indices = np.asarray(index, dtype=np.int64).reshape(-1)
            if indices.size and bool(
                np.any((indices < 0) | (indices >= self._n))
            ):
                raise IndexError("value-count index outside concatenated row range")
        merged: dict[str, int] = {}
        for part, column in enumerate(self._columns):
            start, stop = int(self._offsets[part]), int(self._offsets[part + 1])
            local = None
            if indices is not None:
                selected = (indices >= start) & (indices < stop)
                if not bool(np.any(selected)):
                    continue
                local = indices[selected] - start
            for category, count in column.value_counts(local).items():
                merged[str(category)] = merged.get(str(category), 0) + int(count)
        return merged

    def present_values(self) -> set[str]:
        if self.supports_value_counts:
            return set(self.value_counts())
        return set(map(str, np.unique(np.asarray(self)).tolist()))


class _ConstantColumn:
    """Lazy fixed-shape column used for semantically absent optional targets."""

    def __init__(
        self,
        row_count: int,
        value: Any,
        dtype: Any,
        *,
        inner_shape: Sequence[int] = (),
    ):
        self._n = int(row_count)
        self._value = value
        self._inner_shape = tuple(int(size) for size in inner_shape)
        self.shape = (self._n, *self._inner_shape)
        self.ndim = len(self.shape)
        self.dtype = np.dtype(dtype)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: Any):
        indices, scalar = _normalize_global_index(index, self._n)
        if scalar:
            return np.full(self._inner_shape, self._value, dtype=self.dtype)
        return np.full(
            (*indices.shape, *self._inner_shape), self._value, dtype=self.dtype
        )

    def present_values(self) -> set[str]:
        return {str(self._value)} if self._n else set()

    def value_counts(self, index: Any = None) -> dict[str, int]:
        if index is None:
            count = self._n
        else:
            indices, scalar = _normalize_global_index(index, self._n)
            count = 1 if scalar else int(indices.size)
        return {str(self._value): count} if count else {}


class _DerivedBooleanColumn:
    """Lazy boolean derived from an older corpus' required source column."""

    def __init__(self, source: Any, *, mode: str):
        self._source = source
        self._mode = mode
        self.shape = (int(source.shape[0]),)
        self.ndim = 1
        self.dtype = np.dtype(np.bool_)

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, index: Any):
        source = np.asarray(self._source[index])
        if self._mode == "is_forced":
            return np.sum(source >= 0, axis=-1) == 1
        if self._mode == "used_full_search":
            return source > 0.0
        raise AssertionError(self._mode)


def _synthesized_column(corpus: Any, key: str):
    if key == "is_forced":
        return _DerivedBooleanColumn(corpus["legal_action_ids"], mode=key)
    if key == "used_full_search":
        return _DerivedBooleanColumn(corpus["policy_weight_multiplier"], mode=key)
    if key == "decision_class":
        return _ConstantColumn(corpus.row_count, "legacy_unknown", "<U14")
    if key == "root_value":
        return _ConstantColumn(corpus.row_count, 0.0, np.float32)
    if key == "root_value_mask":
        return _ConstantColumn(corpus.row_count, False, np.bool_)
    if key == "afterstate_target":
        return _ConstantColumn(
            corpus.row_count,
            np.nan,
            np.float32,
            inner_shape=(int(corpus.legal_width),),
        )
    if key == "afterstate_target_mask":
        return _ConstantColumn(
            corpus.row_count,
            False,
            np.bool_,
            inner_shape=(int(corpus.legal_width),),
        )
    if key == "simulations_used":
        return _ConstantColumn(corpus.row_count, 0, np.int32)
    if key in {
        "is_pool_game",
        "opponent_provenance_present",
        "training_source_category_verified",
    }:
        return _ConstantColumn(corpus.row_count, False, np.bool_)
    if key == "opponent_version":
        return _ConstantColumn(corpus.row_count, -1, np.int32)
    if key in {
        "opponent_tag",
        "opponent_checkpoint_md5",
        "opponent_type",
        "training_source_category",
    }:
        return _ConstantColumn(corpus.row_count, "", "<U1")
    if key == AUX_SUBGOAL_TARGET_VERSION_KEY:
        return _ConstantColumn(corpus.row_count, 0, np.uint8)
    if key in {"aux_next_settlement", "aux_robber_target"}:
        return _ConstantColumn(corpus.row_count, -1, np.int16)
    if key in set(AUX_TARGET_KEYS):
        return _ConstantColumn(corpus.row_count, np.nan, np.float32)
    raise KeyError(key)


# Historical phase-2 equivalence test name; every composite column uses the
# same lazy global-index mapping, while `_lazy` identifies large payloads.
_ConcatLazyColumn = _ConcatColumn


class ConcatMemmapCorpus:
    """Dict-like global row view over two or more compatible MemmapCorpus parts."""

    def __init__(
        self,
        corpora: Sequence[Any],
        *,
        dirs: Sequence[str | Path] | None = None,
        component_adapter_versions: Sequence[str] | None = None,
    ) -> None:
        if len(corpora) < 2:
            raise SystemExit("composite memmap corpus requires at least two components")
        self.corpora = tuple(corpora)
        self.component_dirs = (
            tuple(Path(path) for path in dirs) if dirs is not None else tuple()
        )
        if self.component_dirs and len(self.component_dirs) != len(self.corpora):
            raise SystemExit("component directory count differs from corpus count")

        adapter_versions = (
            None
            if component_adapter_versions is None
            else tuple(str(value or "") for value in component_adapter_versions)
        )
        if adapter_versions is not None and (
            len(adapter_versions) != len(self.corpora)
            or any(not value for value in adapter_versions)
        ):
            raise SystemExit(
                "component adapter-version count/value differs from corpus components"
            )

        first = self.corpora[0]
        component_key_sets = [set(corpus.keys()) for corpus in self.corpora]
        afterstate_pair = {"afterstate_target", "afterstate_target_mask"}
        for index, keys in enumerate(component_key_sets):
            present_afterstate = keys & afterstate_pair
            if present_afterstate and present_afterstate != afterstate_pair:
                raise SystemExit(
                    "memmap component has an incomplete afterstate target/mask pair: "
                    f"component={index} present={sorted(present_afterstate)}"
                )
        union_keys = set.union(*component_key_sets)
        if adapter_versions is not None:
            union_keys.add("adapter_version")
        common_keys = set.intersection(*component_key_sets)
        missing_keys = union_keys - common_keys
        adapter_backfill = {"adapter_version"} if adapter_versions is not None else set()
        unsupported_missing = missing_keys - SYNTHESIZABLE_COLUMNS - adapter_backfill
        if unsupported_missing:
            raise SystemExit(
                "memmap components are not schema-compatible: unsupported missing "
                f"columns {sorted(unsupported_missing)}"
            )
        first_keys = tuple(first.keys()) + tuple(sorted(union_keys - set(first.keys())))
        for index, corpus in enumerate(self.corpora[1:], start=1):
            if int(corpus.legal_width) != int(first.legal_width):
                raise SystemExit(
                    "memmap components are not schema-compatible: "
                    f"legal_width component0={first.legal_width} "
                    f"component{index}={corpus.legal_width}"
                )
            for key in first_keys:
                present_schemas = [
                    part._columns[key]
                    for part, keys in zip(self.corpora, component_key_sets, strict=True)
                    if key in keys
                ]
                if not present_schemas and key == "adapter_version" and adapter_versions:
                    continue
                if any(
                    _semantic_column_schema(schema)
                    != _semantic_column_schema(present_schemas[0])
                    for schema in present_schemas[1:]
                ):
                    raise SystemExit(
                        "memmap components are not schema-compatible: "
                        f"column {key!r} differs"
                    )

        self.row_count = sum(int(corpus.row_count) for corpus in self.corpora)
        self.component_offsets = np.concatenate(
            (
                np.asarray([0], dtype=np.int64),
                np.cumsum(
                    [int(corpus.row_count) for corpus in self.corpora],
                    dtype=np.int64,
                ),
            )
        )
        # Populated only by an authenticated v2 descriptor.  Keeping these
        # attributes inert by default preserves the exact v1 runtime surface.
        self.component_ids: tuple[str, ...] = tuple()
        self.component_game_sampling_ratios: tuple[float, ...] = tuple()
        self.policy_kl_anchor_component_indices: tuple[int, ...] = tuple()
        self.policy_kl_anchor_scope_authenticated = False
        self.policy_distillation_component_indices: tuple[int, ...] = tuple()
        self.policy_distillation_scope_authenticated = False
        self.policy_aux_phase_sampling_weights: dict[str, float] | None = None
        self.policy_aux_phase_scope_authenticated = False
        self.aux_subgoal_component_indices: tuple[int, ...] = tuple()
        self.aux_subgoal_scope_authenticated = False
        self.legal_width = int(first.legal_width)
        self._columns = {
            key: (
                next(
                    corpus._columns[key]
                    for corpus, keys in zip(
                        self.corpora, component_key_sets, strict=True
                    )
                    if key in keys
                )
                if any(key in keys for keys in component_key_sets)
                else {
                    "kind": "string",
                    "dtype": f"<U{max(map(len, adapter_versions or ('',)))}",
                }
            )
            for key in first_keys
        }
        self.synthesized_columns_by_component: dict[int, tuple[str, ...]] = {
            index: tuple(sorted(union_keys - keys))
            for index, keys in enumerate(component_key_sets)
            if union_keys - keys
        }
        self._eager: dict[str, _ConcatColumn] = {}
        self._lazy: dict[str, _ConcatColumn] = {}
        row_counts = [int(corpus.row_count) for corpus in self.corpora]
        for key in first_keys:
            columns = []
            for index, (corpus, keys) in enumerate(
                zip(self.corpora, component_key_sets, strict=True)
            ):
                if key in keys:
                    column = corpus[key]
                    if key == "adapter_version" and adapter_versions is not None:
                        present = (
                            column.present_values()
                            if callable(getattr(column, "present_values", None))
                            else set(map(str, np.unique(np.asarray(column)).tolist()))
                        )
                        if present != {adapter_versions[index]}:
                            raise SystemExit(
                                "component adapter-version descriptor differs from "
                                f"stored rows: component={index} "
                                f"descriptor={adapter_versions[index]!r} "
                                f"stored={sorted(present)}"
                            )
                elif key == "adapter_version" and adapter_versions is not None:
                    value = adapter_versions[index]
                    if value != LEGACY_MISSING_CHECKPOINT_ADAPTER_VERSION:
                        raise SystemExit(
                            "component without stored adapter_version may only use the "
                            "explicit legacy missing-metadata mapping: "
                            f"component={index} descriptor={value!r} "
                            f"legacy={LEGACY_MISSING_CHECKPOINT_ADAPTER_VERSION!r}"
                        )
                    column = _ConstantColumn(
                        corpus.row_count, value, f"<U{max(1, len(value))}"
                    )
                else:
                    column = _synthesized_column(corpus, key)
                columns.append(column)
            column = _ConcatColumn(columns, row_counts)
            destination = (
                self._lazy
                if all(
                    key not in keys or key in getattr(corpus, "_lazy", {})
                    for corpus, keys in zip(
                        self.corpora, component_key_sets, strict=True
                    )
                )
                else self._eager
            )
            destination[key] = column
        self.meta = {
            "schema": "memmap_composite_runtime_v1",
            "row_count": self.row_count,
            "legal_width": self.legal_width,
            "component_count": len(self.corpora),
            "synthesized_columns_by_component": {
                str(index): list(keys)
                for index, keys in self.synthesized_columns_by_component.items()
            },
            "shard_count": sum(
                int(getattr(corpus, "meta", {}).get("shard_count", 0))
                for corpus in self.corpora
            ),
        }
        self.stats = {
            "components": [getattr(corpus, "stats", {}) for corpus in self.corpora]
        }
        self.component_adapter_versions = adapter_versions

    def __contains__(self, key: str) -> bool:
        return key in self._eager or key in self._lazy

    def __getitem__(self, key: str):
        if key in self._eager:
            return self._eager[key]
        if key in self._lazy:
            return self._lazy[key]
        raise KeyError(key)

    def get(self, key: str, default=None):
        return self[key] if key in self else default

    def keys(self) -> list[str]:
        return [*self._eager, *self._lazy]

    def __len__(self) -> int:
        return self.row_count

    def component_indices_for_rows(self, rows: Any) -> np.ndarray:
        """Map global row indices to their authenticated component index."""
        indices, _scalar = _normalize_global_index(rows, self.row_count)
        return np.searchsorted(
            self.component_offsets, indices, side="right"
        ).astype(np.int64, copy=False) - 1
