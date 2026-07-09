"""ConcatMemmapCorpus equivalence tests (phase-2 window feed, task #94 PR1).

Gold-standard property: training over ``ConcatMemmapCorpus([A, B])`` must be
indistinguishable from training over one corpus REBUILT over the same shards in
the same order — that equivalence is exactly what lets the flywheel stop
rebuilding the whole window every round (T4). So the test builds, with the real
``tools/build_memmap_corpus.py``:

  A = corpus over [shard1],  B = corpus over [shard2],  M = corpus over [shard1, shard2]

and asserts element-for-element equality between ``ConcatMemmapCorpus([A, B])``
and ``M`` on EVERY column under contiguous / shuffled / cross-boundary / scalar /
negative / empty indexing, plus interface parity and the schema-mismatch guard.

Needs two real self-play npz shards; defaults to the flywheel round_000 worker
shards and skips if absent. Override with CATAN_TEST_SHARD_1/_2.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent

_DEFAULT_SHARD_1 = Path(
    "/home/ubuntu/catan-zero/runs/flywheel_20260707b/gen/round_000/worker_000/gumbel_self_play_shard_00000.npz")
_DEFAULT_SHARD_2 = Path(
    "/home/ubuntu/catan-zero/runs/flywheel_20260707b/gen/round_000/worker_001/gumbel_self_play_shard_00000.npz")
SHARD_1 = Path(os.environ.get("CATAN_TEST_SHARD_1", _DEFAULT_SHARD_1))
SHARD_2 = Path(os.environ.get("CATAN_TEST_SHARD_2", _DEFAULT_SHARD_2))

pytestmark = pytest.mark.skipif(
    not (SHARD_1.exists() and SHARD_2.exists()),
    reason="real self-play shards not found (set CATAN_TEST_SHARD_1/_2)",
)


def _load_train_bc():
    spec = importlib.util.spec_from_file_location("train_bc_under_test", REPO / "tools" / "train_bc.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_corpus(shards: list[Path], out_dir: Path) -> Path:
    """Build a memmap corpus over ``shards`` exactly the way the flywheel does:
    a synthetic manifest dir passed as --source."""
    src_root = out_dir.parent / f"{out_dir.name}_src"
    src_root.mkdir(parents=True, exist_ok=True)
    (src_root / "manifest.json").write_text(
        json.dumps({"shards": [str(s) for s in shards], "rows": None}))
    proc = subprocess.run(
        [sys.executable, str(REPO / "tools" / "build_memmap_corpus.py"),
         "--source", str(src_root), "--out", str(out_dir)],
        capture_output=True, text=True, cwd=str(REPO))
    assert proc.returncode == 0, f"build_memmap_corpus failed:\n{proc.stdout}\n{proc.stderr}"
    return out_dir


@pytest.fixture(scope="module")
def corpora(tmp_path_factory):
    tb = _load_train_bc()
    root = tmp_path_factory.mktemp("concat_corpora")
    a = tb.MemmapCorpus(_build_corpus([SHARD_1], root / "a"))
    b = tb.MemmapCorpus(_build_corpus([SHARD_2], root / "b"))
    merged = tb.MemmapCorpus(_build_corpus([SHARD_1, SHARD_2], root / "m"))
    concat = tb.ConcatMemmapCorpus([a, b], dirs=[root / "a", root / "b"])
    return tb, a, b, merged, concat


def test_row_count_and_interface(corpora):
    tb, a, b, merged, concat = corpora
    assert concat.row_count == a.row_count + b.row_count == merged.row_count
    assert len(concat) == len(merged)
    assert concat.legal_width == merged.legal_width
    assert sorted(concat.keys()) == sorted(merged.keys())
    for key in merged.keys():
        assert key in concat
    assert concat.get("definitely_not_a_column", "sentinel") == "sentinel"
    with pytest.raises(KeyError):
        concat["definitely_not_a_column"]


def _index_patterns(rng: np.ndarray, n: int, boundary: int) -> dict[str, np.ndarray]:
    r = np.random.default_rng(7)
    return {
        "contiguous_head": np.arange(min(64, n), dtype=np.int64),
        "cross_boundary": np.arange(max(0, boundary - 16), min(n, boundary + 16), dtype=np.int64),
        "shuffled_global": r.permutation(n)[: min(256, n)].astype(np.int64),
        "repeated": np.asarray([0, boundary % n, 0, n - 1, boundary % n], dtype=np.int64),
        "empty": np.empty(0, dtype=np.int64),
    }


def test_every_column_matches_gold_rebuild(corpora):
    tb, a, b, merged, concat = corpora
    n = merged.row_count
    patterns = _index_patterns(None, n, boundary=a.row_count)
    for key in merged.keys():
        gold_col = merged[key]
        test_col = concat[key]
        for name, idx in patterns.items():
            gold = np.asarray(gold_col[idx])
            got = np.asarray(test_col[idx])
            assert gold.shape == got.shape, (key, name, gold.shape, got.shape)
            assert gold.dtype == got.dtype or gold.dtype.kind == got.dtype.kind == "U", (key, name)
            if gold.dtype.kind == "f":
                assert np.array_equal(gold, got, equal_nan=True), (key, name)
            else:
                assert np.array_equal(gold, got), (key, name)


def test_scalar_negative_and_slice_parity(corpora):
    tb, a, b, merged, concat = corpora
    n = merged.row_count
    lazy_keys = [k for k in merged.keys() if k in getattr(concat, "_lazy", {})]
    assert lazy_keys, "expected at least one lazy column in a real corpus"
    key = lazy_keys[0]
    # scalar: one row, no leading batch dim; negative: numpy semantics
    assert np.array_equal(np.asarray(merged[key][3]), np.asarray(concat[key][3]))
    assert np.array_equal(np.asarray(merged[key][n - 1]), np.asarray(concat[key][-1]))
    sl = slice(a.row_count - 5, a.row_count + 5)  # slice across the part boundary
    assert np.array_equal(np.asarray(merged[key][sl]), np.asarray(concat[key][sl]))


def test_lazy_columns_stay_lazy(corpora):
    tb, a, b, merged, concat = corpora
    # RAM-parity contract: the big per-decision columns must NOT be materialised.
    for key in tb.MEMMAP_LAZY_COLUMNS:
        if key in merged:
            assert key in concat._lazy, f"{key} was eagerly materialised in the concat"
            assert isinstance(concat[key], tb._ConcatLazyColumn)


def test_schema_mismatch_refused(corpora):
    tb, a, b, merged, concat = corpora

    class _Fake:
        pass

    fake = _Fake()
    fake.legal_width = a.legal_width + 1
    fake._columns = a._columns
    fake.row_count = 1
    fake.meta = {"shard_count": 1}
    fake.stats = {}
    with pytest.raises(SystemExit, match="not schema-compatible"):
        tb.ConcatMemmapCorpus([a, fake])  # type: ignore[list-item]


def test_prefetch_iteration_matches_direct(corpora):
    """The threaded prefetch path must yield batches element-identical to direct
    global indexing (this is the path the flywheel trainer actually runs)."""
    tb, a, b, merged, concat = corpora
    n = concat.row_count
    r = np.random.default_rng(11)
    order = r.permutation(n).astype(np.int64)
    train_indices = np.arange(n, dtype=np.int64)
    pw = r.random(n)
    vw = r.random(n)
    keys = list(concat.keys())
    direct = []
    for start in range(0, n, 128):
        batch = train_indices[order[start:start + 128]]
        if len(batch):
            direct.append({k: np.asarray(concat[k][batch]) for k in keys})
    prefetched = []
    for data, batch, pws, vws in tb._iterate_training_batches(
            concat, order, train_indices, 128, pw, vw, num_workers=2, prefetch=2):
        prefetched.append({k: np.asarray(data[k][batch]) for k in keys})
    assert len(direct) == len(prefetched)
    for d, p in zip(direct, prefetched):
        for k in keys:
            assert np.array_equal(d[k], p[k]) or (
                d[k].dtype.kind == "f" and np.array_equal(d[k], p[k], equal_nan=True)), k
