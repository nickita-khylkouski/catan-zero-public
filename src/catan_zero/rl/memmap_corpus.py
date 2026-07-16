"""Lazy, row-addressable storage for production teacher corpora.

This module owns the on-disk memmap ABI and its NumPy-compatible column
views.  Keeping storage mechanics outside the learner makes the training
entry point responsible for learning semantics rather than file formats.
"""

from __future__ import annotations

import json
import operator
from pathlib import Path

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

    def row_counts(self) -> np.ndarray:
        return (self._offsets[1:] - self._offsets[:-1]).astype(
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

    def __init__(self, corpus_dir: Path):
        corpus_dir = Path(corpus_dir)
        meta_path = corpus_dir / "corpus_meta.json"
        if not meta_path.exists():
            raise SystemExit(
                f"{corpus_dir} is not a memmap corpus (no corpus_meta.json). "
                "Build it with tools/build_memmap_corpus.py or use --data-format npz."
            )
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if self.meta.get("schema") not in {"memmap_corpus_v1", "memmap_corpus_v2"}:
            raise SystemExit(
                f"{meta_path}: unsupported corpus schema {self.meta.get('schema')!r}"
            )
        self.row_count = int(self.meta["row_count"])
        self.legal_width = int(self.meta["legal_width"])
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
        self._offsets = np.fromfile(corpus_dir / "row_offsets.dat", dtype=np.int64)
        if self._offsets.shape[0] != self.row_count + 1:
            raise SystemExit(
                f"{corpus_dir}: row_offsets length {self._offsets.shape[0]} != "
                f"row_count+1 {self.row_count + 1}"
            )
        self._eager: dict[str, np.ndarray] = {}
        self._lazy: dict[str, object] = {}
        independent_offsets: dict[str, np.memmap] = {}
        for name, schema in self._columns.items():
            if schema["kind"] != "row_offsets":
                continue
            offsets = np.memmap(
                corpus_dir / f"{name}.dat",
                dtype=np.dtype(schema["dtype"]),
                mode="r",
                shape=(self.row_count + 1,),
            )
            if int(offsets[0]) != 0 or bool(np.any(offsets[1:] < offsets[:-1])):
                raise SystemExit(f"{meta_path}: invalid row offsets in {name!r}")
            independent_offsets[name] = offsets
            self._lazy[name] = MemmapRowOffsetsColumn(offsets)
        for name, schema in self._columns.items():
            kind = schema["kind"]
            if kind == "row_offsets":
                continue
            if kind == "string":
                codes = np.memmap(
                    corpus_dir / f"{name}.codes.dat",
                    dtype=np.int32,
                    mode="r",
                    shape=(self.row_count,),
                )
                categories = np.asarray(schema["categories"], dtype=str)
                if categories.size == 0:
                    categories = np.asarray([""], dtype=str)
                self._lazy[name] = MemmapCategoricalColumn(codes, categories)
                continue
            if kind == "fixed":
                inner = tuple(int(d) for d in schema["inner_shape"])
                mm = np.memmap(
                    corpus_dir / f"{name}.dat",
                    dtype=np.dtype(schema["dtype"]),
                    mode="r",
                    shape=(self.row_count, *inner),
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
                flat_count = int(offsets[-1])
                flat = (
                    np.memmap(
                        corpus_dir / f"{name}.dat",
                        dtype=np.dtype(schema["dtype"]),
                        mode="r",
                        shape=(flat_count,),
                    )
                    if flat_count
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
                (int(self.meta["flat_count"]), feat)
                if feat is not None
                else (int(self.meta["flat_count"]),)
            )
            flat = np.memmap(
                corpus_dir / f"{name}.dat",
                dtype=np.dtype(schema["dtype"]),
                mode="r",
                shape=flat_shape,
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
