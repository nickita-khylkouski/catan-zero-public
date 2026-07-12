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


def _semantic_column_schema(schema: dict[str, Any]) -> tuple[Any, ...]:
    """Compatibility fields that affect the decoded NumPy column surface."""
    kind = schema.get("kind")
    common = (kind, np.dtype(schema.get("dtype", "float32")).str)
    if kind in {"fixed", "implicit_constant"}:
        return (*common, tuple(schema.get("inner_shape", ())), schema.get("fill"))
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
    ) -> None:
        if len(corpora) < 2:
            raise SystemExit("composite memmap corpus requires at least two components")
        self.corpora = tuple(corpora)
        self.component_dirs = (
            tuple(Path(path) for path in dirs) if dirs is not None else tuple()
        )
        if self.component_dirs and len(self.component_dirs) != len(self.corpora):
            raise SystemExit("component directory count differs from corpus count")

        first = self.corpora[0]
        first_keys = tuple(first.keys())
        first_key_set = set(first_keys)
        first_schemas = first._columns
        for index, corpus in enumerate(self.corpora[1:], start=1):
            if int(corpus.legal_width) != int(first.legal_width):
                raise SystemExit(
                    "memmap components are not schema-compatible: "
                    f"legal_width component0={first.legal_width} "
                    f"component{index}={corpus.legal_width}"
                )
            if set(corpus.keys()) != first_key_set:
                raise SystemExit(
                    "memmap components are not schema-compatible: column keys differ"
                )
            for key in first_keys:
                if _semantic_column_schema(
                    corpus._columns[key]
                ) != _semantic_column_schema(first_schemas[key]):
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
        self.legal_width = int(first.legal_width)
        self._columns = dict(first_schemas)
        self._eager: dict[str, _ConcatColumn] = {}
        self._lazy: dict[str, _ConcatColumn] = {}
        row_counts = [int(corpus.row_count) for corpus in self.corpora]
        for key in first_keys:
            column = _ConcatColumn([corpus[key] for corpus in self.corpora], row_counts)
            destination = (
                self._lazy
                if all(key in getattr(corpus, "_lazy", {}) for corpus in self.corpora)
                else self._eager
            )
            destination[key] = column
        self.meta = {
            "schema": "memmap_composite_runtime_v1",
            "row_count": self.row_count,
            "legal_width": self.legal_width,
            "component_count": len(self.corpora),
            "shard_count": sum(
                int(getattr(corpus, "meta", {}).get("shard_count", 0))
                for corpus in self.corpora
            ),
        }
        self.stats = {
            "components": [getattr(corpus, "stats", {}) for corpus in self.corpora]
        }

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
