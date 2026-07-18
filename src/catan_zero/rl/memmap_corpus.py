"""Lazy, row-addressable storage for production teacher corpora.

This module owns the on-disk memmap ABI and its NumPy-compatible column
views.  Keeping storage mechanics outside the learner makes the training
entry point responsible for learning semantics rather than file formats.
"""

from __future__ import annotations

import json
import operator
from pathlib import Path
from typing import BinaryIO, Mapping

import numpy as np


MEMMAP_LAZY_COLUMNS = frozenset(
    {
        "obs",
        "legal_action_ids",
        "legal_action_context",
        "legal_action_tokens",
        "legal_action_target_ids",
        "legal_action_mask",
        "hex_tokens",
        "hex_vertex_ids",
        "hex_edge_ids",
        "hex_mask",
        "vertex_tokens",
        "vertex_mask",
        "edge_tokens",
        "edge_vertex_ids",
        "edge_mask",
        "player_tokens",
        "player_mask",
        "global_tokens",
        "event_tokens",
        "event_target_ids",
        "event_mask",
        "prior_policy",
        "target_policy",
        "target_policy_mask",
        "target_scores",
        "target_scores_mask",
        "completed_q_values",
        "completed_q_mask",
        "afterstate_target",
        "afterstate_target_mask",
        "search_evidence_version",
        "search_evidence_mask",
        "search_evidence_offsets",
        "search_visit_counts_flat",
        "search_completed_q_flat",
        "search_prior_policy_flat",
    }
)

_VALIDATION_BLOCK_ROWS = 1024 * 1024


def _require_exact_file_size(path: Path, expected_bytes: int, *, label: str) -> None:
    """Fail before mapping a missing, truncated, or trailing-byte payload."""

    try:
        actual_bytes = path.stat().st_size
    except OSError as error:
        raise SystemExit(f"{label}: cannot stat {path}: {error}") from error
    if actual_bytes != expected_bytes:
        raise SystemExit(
            f"{label}: {path.name} size {actual_bytes} bytes != expected "
            f"{expected_bytes} bytes"
        )


def _validate_offsets(
    offsets: np.ndarray,
    *,
    expected_final: int | None,
    legal_width: int,
    label: str,
) -> None:
    """Block-scan a ragged row index without materializing it in RAM."""

    if int(offsets[0]) != 0:
        raise SystemExit(f"{label}: offsets must start at 0, got {int(offsets[0])}")
    for start in range(0, len(offsets) - 1, _VALIDATION_BLOCK_ROWS):
        stop = min(len(offsets), start + _VALIDATION_BLOCK_ROWS + 1)
        block = np.asarray(offsets[start:stop], dtype=np.int64)
        decreasing = block[1:] < block[:-1]
        if bool(np.any(decreasing)):
            local = int(np.flatnonzero(decreasing)[0])
            raise SystemExit(f"{label}: offsets decrease at row {start + local}")
        widths = block[1:] - block[:-1]
        if bool(np.any(widths > legal_width)):
            local = int(np.flatnonzero(widths > legal_width)[0])
            raise SystemExit(
                f"{label}: row {start + local} width {int(widths[local])} "
                f"exceeds legal_width {legal_width}"
            )
    if expected_final is not None and int(offsets[-1]) != expected_final:
        raise SystemExit(
            f"{label}: final offset {int(offsets[-1])} != flat count {expected_final}"
        )


def _validate_categorical_codes(
    codes: np.ndarray,
    *,
    category_count: int,
    label: str,
) -> None:
    """Reject negative and upper-bound dictionary codes during corpus construction."""

    for start in range(0, len(codes), _VALIDATION_BLOCK_ROWS):
        stop = min(len(codes), start + _VALIDATION_BLOCK_ROWS)
        block = np.asarray(codes[start:stop], dtype=np.int64)
        invalid = (block < 0) | (block >= category_count)
        if bool(np.any(invalid)):
            local = int(np.flatnonzero(invalid)[0])
            raise SystemExit(
                f"{label}: categorical code {int(block[local])} at row "
                f"{start + local} is outside [0, {category_count})"
            )


def normalize_index(idx, n: int) -> np.ndarray:
    """Coerce an indexing key into an int64 row-index array."""
    if isinstance(idx, slice):
        return np.arange(*idx.indices(n), dtype=np.int64)
    arr = np.asarray(idx)
    if arr.dtype == np.bool_:
        return np.flatnonzero(arr).astype(np.int64, copy=False)
    return arr.astype(np.int64, copy=False)


class MemmapFixedColumn:
    """Fixed-width column backed by a flat memmap."""

    def __init__(self, mm: np.memmap, n: int):
        self._mm = mm
        self.shape = tuple(mm.shape)
        self.ndim = mm.ndim
        self.dtype = mm.dtype
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx):
        return np.asarray(self._mm[idx])

    def __array__(self, dtype=None):
        arr = np.asarray(self._mm)
        return arr.astype(dtype) if dtype is not None else arr


class MemmapCategoricalColumn:
    """Dictionary-encoded strings decoded only for requested rows."""

    def __init__(self, codes: np.memmap, categories: np.ndarray):
        self._codes = codes
        self.categories = categories
        self.shape = tuple(codes.shape)
        self.ndim = codes.ndim
        self.dtype = categories.dtype

    def __len__(self) -> int:
        return int(self._codes.shape[0])

    def __getitem__(self, idx):
        return self.categories[np.asarray(self._codes[idx])]

    def __array__(self, dtype=None):
        values = self.categories[np.asarray(self._codes)]
        return values.astype(dtype) if dtype is not None else values

    def grouped_weights(
        self, weights: np.ndarray, *, limit: int
    ) -> dict[str, dict[str, float | int]]:
        codes = np.asarray(self._codes, dtype=np.int64)
        counts = np.bincount(codes, minlength=len(self.categories))
        totals = np.bincount(
            codes,
            weights=np.asarray(weights, dtype=np.float64),
            minlength=len(self.categories),
        )
        order = np.argsort(-counts)
        result: dict[str, dict[str, float | int]] = {}
        for index in order[:limit]:
            raw = int(counts[index])
            if raw == 0:
                continue
            total = float(totals[index])
            result[str(self.categories[index])] = {
                "raw_samples": raw,
                "weight_sum": total,
                "mean_weight": total / raw,
            }
        return result

    def _code_counts(self) -> np.ndarray:
        counts = np.zeros(len(self.categories), dtype=np.int64)
        block_rows = 8 * 1024 * 1024
        for start in range(0, len(self), block_rows):
            stop = min(start + block_rows, len(self))
            block = np.asarray(self._codes[start:stop], dtype=np.int64)
            partial = np.bincount(block, minlength=len(self.categories))
            if len(partial) != len(self.categories):
                raise SystemExit("categorical memmap contains an out-of-range code")
            counts += partial
        return counts

    def present_values(self) -> set[str]:
        counts = self._code_counts()
        return {
            str(self.categories[index]) for index in np.flatnonzero(counts)
        }

    def value_counts(self, index=None) -> dict[str, int]:
        if index is None:
            counts = self._code_counts()
        else:
            codes = np.asarray(self._codes[index], dtype=np.int64)
            counts = np.bincount(codes, minlength=len(self.categories))
            if len(counts) != len(self.categories):
                raise SystemExit("categorical memmap contains an out-of-range code")
        return {
            str(self.categories[position]): int(counts[position])
            for position in np.flatnonzero(counts)
        }


class ImplicitConstantColumn:
    """File-free fixed-width column materialized only for requested rows."""

    def __init__(self, n: int, inner_shape: tuple[int, ...], dtype, fill):
        self._n = int(n)
        self._inner_shape = tuple(int(d) for d in inner_shape)
        self.dtype = np.dtype(dtype)
        self._fill = fill
        self.shape = (self._n, *self._inner_shape)
        self.ndim = len(self.shape)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx):
        output_prefix = self._indexed_prefix_shape(idx)
        if not output_prefix:
            return np.full(self._inner_shape, self._fill, dtype=self.dtype)
        return np.full(
            (*output_prefix, *self._inner_shape), self._fill, dtype=self.dtype
        )

    def _indexed_prefix_shape(self, idx) -> tuple[int, ...]:
        if isinstance(idx, slice):
            return (len(range(*idx.indices(self._n))),)
        array = np.asarray(idx)
        if array.ndim == 0:
            if array.dtype.kind == "b":
                return (int(bool(array)), self._n)
            try:
                value = operator.index(idx)
            except TypeError:
                try:
                    value = operator.index(array.item())
                except (TypeError, ValueError) as error:
                    raise IndexError(
                        "implicit column indices must be integers, slices, or "
                        "integer/boolean arrays"
                    ) from error
            if not -self._n <= value < self._n:
                raise IndexError(
                    f"index {value} is out of bounds for axis 0 with size {self._n}"
                )
            return ()
        if array.dtype.kind == "b":
            if array.ndim != 1 or array.shape[0] != self._n:
                raise IndexError(
                    "boolean index did not match implicit column row axis; "
                    f"axis has size {self._n} but mask shape is {array.shape}"
                )
            return (int(np.count_nonzero(array)),)
        literal_empty_list = isinstance(idx, list) and array.size == 0
        if array.dtype.kind not in {"i", "u"} and not literal_empty_list:
            raise IndexError("arrays used as indices must be of integer or boolean type")
        if array.size and bool(np.any((array < -self._n) | (array >= self._n))):
            bad = int(array[(array < -self._n) | (array >= self._n)].flat[0])
            raise IndexError(
                f"index {bad} is out of bounds for axis 0 with size {self._n}"
            )
        return tuple(int(d) for d in array.shape)

    def __array__(self, dtype=None):
        arr = np.full(self.shape, self._fill, dtype=self.dtype)
        return arr.astype(dtype) if dtype is not None else arr


class MemmapRaggedColumn:
    """Legal-action-ragged column stored trimmed on disk."""

    def __init__(
        self,
        flat: np.memmap,
        offsets: np.ndarray,
        legal_width: int,
        fill,
        dtype,
        feat,
    ):
        self._flat = flat
        self._offsets = offsets
        self._width = int(legal_width)
        self._fill = fill
        self.dtype = np.dtype(dtype)
        self._feat = feat
        self._n = int(offsets.shape[0] - 1)
        self.ndim = 3 if feat is not None else 2
        self.shape = (
            (self._n, self._width, feat)
            if feat is not None
            else (self._n, self._width)
        )

    def __len__(self) -> int:
        return self._n

    def row_counts(self, indices: np.ndarray | None = None) -> np.ndarray:
        if indices is None:
            return (self._offsets[1:] - self._offsets[:-1]).astype(
                np.int64, copy=False
            )
        rows = normalize_index(indices, self._n)
        assert rows is not None
        return (self._offsets[rows + 1] - self._offsets[rows]).astype(
            np.int64, copy=False
        )

    def _reconstruct(self, indices: np.ndarray | None) -> np.ndarray:
        width = self._width
        if indices is None:
            counts = (self._offsets[1:] - self._offsets[:-1]).astype(np.int64)
            prefix = np.arange(width)[None, :] < counts[:, None]
            out = self._new_full(self._n)
            out[prefix] = np.asarray(self._flat)
            return out
        starts = self._offsets[indices]
        counts = (self._offsets[indices + 1] - starts).astype(np.int64)
        out = self._new_full(int(indices.shape[0]))
        total = int(counts.sum())
        if total:
            prefix = np.arange(width)[None, :] < counts[:, None]
            within = np.arange(total, dtype=np.int64) - np.repeat(
                np.cumsum(counts) - counts, counts
            )
            src = np.repeat(starts, counts) + within
            out[prefix] = np.asarray(self._flat[src])
        return out

    def _new_full(self, rows: int) -> np.ndarray:
        if self._feat is not None:
            return np.full(
                (rows, self._width, self._feat), self._fill, dtype=self.dtype
            )
        return np.full((rows, self._width), self._fill, dtype=self.dtype)

    def __getitem__(self, idx):
        return self._reconstruct(normalize_index(idx, self._n))

    def __array__(self, dtype=None):
        arr = self._reconstruct(None)
        return arr.astype(dtype) if dtype is not None else arr


class MemmapRowOffsetsColumn:
    """Row-addressable view of an independent ragged column's offsets."""

    def __init__(self, offsets: np.ndarray):
        self._offsets = offsets
        self._n = int(offsets.shape[0] - 1)
        self.shape = (self._n, 2)
        self.ndim = 2
        self.dtype = offsets.dtype

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx):
        indices = normalize_index(idx, self._n)
        return np.stack(
            (self._offsets[indices], self._offsets[indices + 1]), axis=-1
        )

    def __array__(self, dtype=None):
        array = np.stack((self._offsets[:-1], self._offsets[1:]), axis=-1)
        return array.astype(dtype) if dtype is not None else array


class MemmapCorpus:
    """Dict-of-arrays view over a production teacher memmap corpus."""

    def __init__(
        self,
        corpus_dir: Path,
        *,
        authenticated_files: Mapping[str, BinaryIO] | None = None,
        authenticated_meta: Mapping[str, object] | None = None,
    ):
        corpus_dir = Path(corpus_dir)
        self._authenticated_files = (
            {} if authenticated_files is None else dict(authenticated_files)
        )

        def source(path: Path):
            handle = self._authenticated_files.get(path.name)
            if handle is None:
                return path
            handle.seek(0)
            return handle

        meta_path = corpus_dir / "corpus_meta.json"
        meta_handle = self._authenticated_files.get("corpus_meta.json")
        if (
            authenticated_meta is None
            and meta_handle is None
            and not meta_path.exists()
        ):
            raise SystemExit(
                f"{corpus_dir} is not a memmap corpus (no corpus_meta.json). "
                "Build it with tools/build_memmap_corpus.py or use --data-format npz."
            )
        if authenticated_meta is not None:
            self.meta = dict(authenticated_meta)
        elif meta_handle is None:
            raw_meta = meta_path.read_text(encoding="utf-8")
            self.meta = json.loads(raw_meta)
        else:
            meta_handle.seek(0)
            raw_meta = meta_handle.read().decode("utf-8")
            self.meta = json.loads(raw_meta)
        if self.meta.get("schema") not in {"memmap_corpus_v1", "memmap_corpus_v2"}:
            raise SystemExit(
                f"{meta_path}: unsupported corpus schema {self.meta.get('schema')!r}"
            )
        self.row_count = int(self.meta["row_count"])
        self.legal_width = int(self.meta["legal_width"])
        if self.row_count < 0 or self.legal_width < 0:
            raise SystemExit(
                f"{meta_path}: row_count and legal_width must be non-negative"
            )
        flat_count = int(self.meta["flat_count"])
        if flat_count < 0:
            raise SystemExit(f"{meta_path}: flat_count must be non-negative")
        self.stats = self.meta.get("stats", {})
        self._columns = self.meta["columns"]
        implicit_columns = {
            name
            for name, schema in self._columns.items()
            if schema.get("kind") == "implicit_constant"
        }
        declared_implicit_raw = self.meta.get("implicit_zero_columns", ())
        try:
            declared_implicit_columns = set(declared_implicit_raw)
        except TypeError as error:
            raise SystemExit(
                f"{meta_path}: implicit_zero_columns must be a sequence of names"
            ) from error
        if implicit_columns != declared_implicit_columns:
            raise SystemExit(
                f"{meta_path}: implicit column metadata mismatch: "
                f"columns={sorted(implicit_columns)} "
                f"declared={sorted(declared_implicit_columns)}"
            )
        unsupported_implicit = implicit_columns - {"event_tokens", "event_mask"}
        if unsupported_implicit:
            raise SystemExit(
                f"{meta_path}: unsupported implicit columns {sorted(unsupported_implicit)}"
            )
        if self.meta.get("schema") == "memmap_corpus_v2":
            required_implicit = {"event_tokens", "event_mask"}
            if (
                implicit_columns != required_implicit
                or len(declared_implicit_raw) != len(required_implicit)
            ):
                raise SystemExit(
                    f"{meta_path}: memmap_corpus_v2 requires exactly implicit-zero "
                    f"columns {sorted(required_implicit)}; got {sorted(implicit_columns)}"
                )
            nonzero_fill = [
                name
                for name in sorted(required_implicit)
                if self._columns[name].get("fill") != 0
            ]
            if nonzero_fill:
                raise SystemExit(
                    f"{meta_path}: implicit-zero columns must declare fill=0; "
                    f"nonzero/missing fill for {nonzero_fill}"
                )
        shared_offsets_path = corpus_dir / "row_offsets.dat"
        _require_exact_file_size(
            shared_offsets_path,
            (self.row_count + 1) * np.dtype(np.int64).itemsize,
            label=str(meta_path),
        )
        self._offsets = np.memmap(
            source(shared_offsets_path),
            dtype=np.int64,
            mode="r",
            shape=(self.row_count + 1,),
        )
        _validate_offsets(
            self._offsets,
            expected_final=flat_count,
            legal_width=self.legal_width,
            label=f"{meta_path}: row_offsets",
        )
        self._eager: dict[str, np.ndarray] = {}
        self._lazy: dict[str, object] = {}
        independent_offsets: dict[str, np.memmap] = {}
        for name, schema in self._columns.items():
            if schema["kind"] != "row_offsets":
                continue
            dtype = np.dtype(schema["dtype"])
            if dtype != np.dtype(np.int64):
                raise SystemExit(
                    f"{meta_path}: row-offset column {name!r} must use int64, "
                    f"got {dtype}"
                )
            path = corpus_dir / f"{name}.dat"
            _require_exact_file_size(
                path,
                (self.row_count + 1) * dtype.itemsize,
                label=str(meta_path),
            )
            offsets = np.memmap(
                source(path),
                dtype=dtype,
                mode="r",
                shape=(self.row_count + 1,),
            )
            _validate_offsets(
                offsets,
                expected_final=None,
                legal_width=self.legal_width,
                label=f"{meta_path}: {name}",
            )
            independent_offsets[name] = offsets
            self._lazy[name] = MemmapRowOffsetsColumn(offsets)
        for name, schema in self._columns.items():
            kind = schema["kind"]
            if kind == "row_offsets":
                continue
            if kind == "string":
                path = corpus_dir / f"{name}.codes.dat"
                _require_exact_file_size(
                    path,
                    self.row_count * np.dtype(np.int32).itemsize,
                    label=str(meta_path),
                )
                codes = (
                    np.memmap(
                        source(path),
                        dtype=np.int32,
                        mode="r",
                        shape=(self.row_count,),
                    )
                    if self.row_count
                    else np.empty(0, dtype=np.int32)
                )
                categories = np.asarray(schema["categories"], dtype=str)
                if categories.ndim != 1:
                    raise SystemExit(
                        f"{meta_path}: string categories for {name!r} must be "
                        "a one-dimensional sequence"
                    )
                if categories.size == 0:
                    categories = np.asarray([""], dtype=str)
                _validate_categorical_codes(
                    codes,
                    category_count=len(categories),
                    label=f"{meta_path}: {name}",
                )
                self._lazy[name] = MemmapCategoricalColumn(codes, categories)
                continue
            if kind == "fixed":
                inner = tuple(int(d) for d in schema["inner_shape"])
                dtype = np.dtype(schema["dtype"])
                path = corpus_dir / f"{name}.dat"
                element_count = self.row_count * int(np.prod(inner, dtype=np.int64))
                _require_exact_file_size(
                    path,
                    element_count * dtype.itemsize,
                    label=str(meta_path),
                )
                shape = (self.row_count, *inner)
                mm = (
                    np.memmap(
                        source(path),
                        dtype=dtype,
                        mode="r",
                        shape=shape,
                    )
                    if element_count
                    else np.empty(shape, dtype=dtype)
                )
                if name in MEMMAP_LAZY_COLUMNS:
                    self._lazy[name] = MemmapFixedColumn(mm, self.row_count)
                else:
                    self._eager[name] = np.asarray(mm)
                continue
            if kind == "implicit_constant":
                if self.meta.get("schema") != "memmap_corpus_v2":
                    raise SystemExit(
                        f"{meta_path}: implicit_constant column {name!r} requires "
                        "memmap_corpus_v2"
                    )
                inner = tuple(int(d) for d in schema["inner_shape"])
                self._lazy[name] = ImplicitConstantColumn(
                    self.row_count, inner, schema["dtype"], schema["fill"]
                )
                continue
            if kind == "independent_ragged1d":
                offsets_name = schema.get("offsets")
                offsets = independent_offsets.get(offsets_name)
                if offsets is None:
                    raise SystemExit(
                        f"{meta_path}: independent ragged column {name!r} has "
                        f"unknown offsets {offsets_name!r}"
                    )
                independent_flat_count = int(offsets[-1])
                dtype = np.dtype(schema["dtype"])
                path = corpus_dir / f"{name}.dat"
                _require_exact_file_size(
                    path,
                    independent_flat_count * dtype.itemsize,
                    label=str(meta_path),
                )
                flat = (
                    np.memmap(
                        source(path),
                        dtype=dtype,
                        mode="r",
                        shape=(independent_flat_count,),
                    )
                    if independent_flat_count
                    else np.empty(0, dtype=np.dtype(schema["dtype"]))
                )
                self._lazy[name] = MemmapRaggedColumn(
                    flat,
                    offsets,
                    self.legal_width,
                    schema["fill"],
                    schema["dtype"],
                    None,
                )
                continue
            if kind not in {"ragged2d", "ragged3d"}:
                raise SystemExit(
                    f"{meta_path}: unsupported storage kind {kind!r} for {name!r}"
                )
            feat = int(schema["feat"]) if kind == "ragged3d" else None
            flat_shape = (
                (flat_count, feat)
                if feat is not None
                else (flat_count,)
            )
            dtype = np.dtype(schema["dtype"])
            path = corpus_dir / f"{name}.dat"
            element_count = flat_count * (feat if feat is not None else 1)
            _require_exact_file_size(
                path,
                element_count * dtype.itemsize,
                label=str(meta_path),
            )
            flat = (
                np.memmap(
                    source(path),
                    dtype=dtype,
                    mode="r",
                    shape=flat_shape,
                )
                if flat_count
                else np.empty(flat_shape, dtype=dtype)
            )
            column = MemmapRaggedColumn(
                flat,
                self._offsets,
                self.legal_width,
                schema["fill"],
                schema["dtype"],
                feat,
            )
            if name in MEMMAP_LAZY_COLUMNS:
                self._lazy[name] = column
            else:
                self._eager[name] = column._reconstruct(None)

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

    def keys(self):
        return list(self._eager.keys()) + list(self._lazy.keys())

    def __len__(self) -> int:
        return self.row_count


__all__ = [
    "ImplicitConstantColumn",
    "MEMMAP_LAZY_COLUMNS",
    "MemmapCategoricalColumn",
    "MemmapCorpus",
    "MemmapFixedColumn",
    "MemmapRaggedColumn",
    "MemmapRowOffsetsColumn",
    "normalize_index",
]
