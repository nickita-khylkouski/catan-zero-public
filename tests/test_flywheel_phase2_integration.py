"""Phase-2 window-feed integration tests (task #94, PR3).

Covers the orchestrator-side pieces that PR1's ConcatMemmapCorpus tests can't:

  1. ``ingest_feed_batches`` — pure: .ready gating, manifest parsing, bad-manifest
     skip, idempotency, ckpt_version propagation into the window registry.
  2. ``Runner.build_round_corpus`` — real build over real shards, authoritative
     row count read-back.
  3. ``Runner.train_window`` over a MIXED window (two corpus dirs + one legacy
     npz shard) with the real gen-3 champion checkpoint: asserts the train
     subprocess ran the --data concat list, trained the bounded step count, and
     published a candidate. Needs CUDA + the real checkpoint; skipped elsewhere.

Heavy bits are guarded by skipif; the pure test always runs.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

# SW-0 integration decision (2026-07-08): this test file (landed via cat-18
# "commit-debt") targets the phase-2 window-feed feature (task #94):
# `ingest_feed_batches` and `Runner.build_round_corpus/train_window` in
# tools/continuous_flywheel.py. That implementation was never committed to any
# SW-0 branch -- it lives only in the unmerged f94/speed-czar tree and, per
# project state, is deliberately held awaiting lead approval. cat-18 committed
# only the test, not the driver changes. Skipped (not deleted) so the gap stays
# greppable and the test is ready the moment task #94 is merged.
pytest.skip(
    "Phase-2 window-feed (task #94) impl is unmerged/awaiting-approval; only the "
    "test landed via cat-18 commit-debt. See module header.",
    allow_module_level=True,
)

_SHARDS = [
    Path(os.environ.get(f"CATAN_TEST_SHARD_{i}", (
        f"/home/ubuntu/catan-zero/runs/flywheel_20260707b/gen/round_000/"
        f"worker_00{i - 1}/gumbel_self_play_shard_00000.npz")))
    for i in (1, 2, 3)
]
_GEN3_CKPT = Path(os.environ.get(
    "CATAN_TEST_CHAMPION", "/home/ubuntu/catan-zero/runs/bc/gen3_20260706/checkpoint.pt"))


def _load_flywheel():
    spec = importlib.util.spec_from_file_location(
        "continuous_flywheel_under_test", REPO / "tools" / "continuous_flywheel.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ 1. feed ingest (pure)
def test_ingest_feed_batches(tmp_path):
    fw = _load_flywheel()
    from catan_zero.rl.flywheel import WindowedReplay

    loop_dir = tmp_path / "loop"
    feed = loop_dir / "feed" / "corpus"
    window = WindowedReplay(loop_dir / "window_state.json", c=1000)

    # no feed dir yet -> no-op
    assert fw.ingest_feed_batches(loop_dir, window) == []

    ready = feed / "batch_000"
    ready.mkdir(parents=True)
    (ready / "feed_manifest.json").write_text(json.dumps({"row_count": 1234, "ckpt_version": 3}))
    (ready / ".ready").write_text("")

    not_ready = feed / "batch_001"  # daemon still building: no .ready
    not_ready.mkdir()
    (not_ready / "feed_manifest.json").write_text(json.dumps({"row_count": 99, "ckpt_version": 3}))

    broken = feed / "batch_002"  # .ready but unreadable manifest -> skipped, not fatal
    broken.mkdir()
    (broken / "feed_manifest.json").write_text("{not json")
    (broken / ".ready").write_text("")

    ingested = fw.ingest_feed_batches(loop_dir, window)
    assert [b["batch"] for b in ingested] == ["batch_000"]
    assert ingested[0] == {"batch": "batch_000", "rows": 1234, "ckpt_version": 3,
                           "wave_roots": None}  # journal separates backfill vs live waves
    meta = window._by_path[str(ready)]
    assert meta.rows == 1234 and meta.ckpt_version == 3

    # idempotent: second scan registers nothing new
    assert fw.ingest_feed_batches(loop_dir, window) == []
    # a batch becoming ready later is picked up
    (not_ready / ".ready").write_text("")
    assert [b["batch"] for b in fw.ingest_feed_batches(loop_dir, window)] == ["batch_001"]


# ------------------------------------------------------------------ 2+3. real-build path
_heavy = pytest.mark.skipif(
    not (all(s.exists() for s in _SHARDS) and _GEN3_CKPT.exists()),
    reason="real shards / gen-3 checkpoint not found (B200-hosted test)",
)


def _runner(fw, loop_dir: Path, device: str):
    from catan_zero.rl.flywheel import FlywheelConfig, ensure_dirs, seed_champion

    cfg = FlywheelConfig(regime="continuous", window_c_rows=3_000_000,
                         opponent_pool_fraction=0.2, gate_games=150,
                         train_batch_size=4096).validate()
    ensure_dirs(loop_dir)
    seed_champion(loop_dir, str(_GEN3_CKPT), version=0)
    return fw.Runner(cfg, loop_dir, dry_run=False, workers=2, device=device,
                     base_seed=999), cfg


@_heavy
def test_build_round_corpus_and_mixed_train_window(tmp_path):
    fw = _load_flywheel()
    loop_dir = tmp_path / "loop"
    loop_dir.mkdir()
    runner, cfg = _runner(fw, loop_dir, device=os.environ.get("CATAN_TEST_DEVICE", "cuda:0"))

    # 2. per-round corpus build over real shards (authoritative row count)
    rc_a = runner.build_round_corpus([str(_SHARDS[0])], round_idx=0)
    rc_b = runner.build_round_corpus([str(_SHARDS[1])], round_idx=1)
    assert rc_a["ok"] and rc_b["ok"], (rc_a, rc_b)
    assert rc_a["rows"] > 0 and Path(rc_a["corpus_dir"]).joinpath("corpus_meta.json").exists()
    assert "window_corpus" in rc_a["corpus_dir"]  # NOT under corpus/ (cleanup_old_corpora)
    assert not (Path(rc_a["corpus_dir"]).parent / "round_000_src").exists()  # staging cleaned

    # 3. mixed window: two corpora + one legacy npz -> --data concat + legacy rebuild
    window_entries = [rc_a["corpus_dir"], rc_b["corpus_dir"], str(_SHARDS[2])]
    tr = runner.train_window(window_entries, str(_GEN3_CKPT), round_idx=2,
                             new_rows_this_round=1000)
    assert tr.get("ok"), tr
    # bounded steps: 1000 rows * target_reuse / 4096, floored at 1
    assert tr["steps"] == max(1, int(1000 * cfg.target_reuse / 4096))
    assert Path(tr["candidate"]).exists()
    assert tr.get("telemetry"), "per-round KL/val telemetry missing from train result"
    train_log = (loop_dir / "train.log").read_text()
    assert "bc_memmap_concat" in train_log, "train_bc did not run the concat loader"
    # the concat must cover all three parts: corpus a, corpus b, legacy_window rebuild
    concat_line = next(l for l in train_log.splitlines() if '"bc_memmap_concat"' in l)
    assert json.loads(concat_line)["parts"] == 3
    legacy_dir = loop_dir / "corpus" / "round_002" / "legacy_window"
    assert legacy_dir.joinpath("corpus_meta.json").exists()
