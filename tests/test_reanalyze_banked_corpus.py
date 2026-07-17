"""Unit tests for tools/reanalyze_banked_corpus.py (CAT-63).

Covers, on a tiny SYNTHETIC multi-chunk memmap corpus crafted directly on disk
(no real shards, no model, no GPU -- reanalyze_lite's forward is faked):

* chunk planning -- contiguous, gap-free, no-overlap partition snapped to
  row_offsets; flat spans sum to flat_count.
* value-only CORRECTNESS -- the chunked+merged column is BYTE-IDENTICAL to a
  single-shot reanalyze_lite full run over the same corpus with the same forward.
* RESUMABILITY / preemption -- kill mid-way (run with --max-chunks, plus a
  deliberately truncated piece with no done marker), resume completes EXACTLY the
  remaining chunks, already-done chunks are NOT reprocessed (no double-processing),
  and a corrupt piece is detected and redone.
* PROVENANCE -- job + per-chunk + merge manifests carry checkpoint md5, piece
  hashes, row counts, GPU-hours.
* MERGE -- versioned overlay loads through MemmapCorpus, unchanged columns are
  byte-identical to source, row counts before==after (no loss/dup).
* per-state root_value materialisation through the fleet path.
* claim files -- O_EXCL claim, fresh-claim refusal, stale-claim steal.
* mix plan arithmetic.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
_TOOLS_DIR = _REPO / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import reanalyze_banked_corpus as rbc  # type: ignore  # noqa: E402
import reanalyze_lite as rl  # type: ignore  # noqa: E402
import train_bc  # type: ignore  # noqa: E402
from train_bc import MemmapCorpus  # type: ignore  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic multi-chunk corpus (trimmed-flat layout, arbitrary row count)
# --------------------------------------------------------------------------- #
_LEGAL_WIDTH = 4
_PHASES = ("robber", "roll", "build")


def _row_scores(i: int) -> list[float]:
    """Deterministic per-row legal scores; count cycles 1..4, with a sprinkling of
    in-prefix NaNs (every 5th row's slot 0) that must be PRESERVED by the rewrite."""
    count = (i % _LEGAL_WIDTH) + 1
    scores = [round(0.1 * i + 0.01 * j, 4) for j in range(count)]
    if i % 5 == 0 and count >= 1:
        scores[0] = float("nan")  # in-prefix NaN -> preserved
    return scores


def _make_corpus(corpus_dir: Path, n_rows: int) -> None:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    w = _LEGAL_WIDTH
    rows = [_row_scores(i) for i in range(n_rows)]
    counts = np.array([len(r) for r in rows], dtype=np.int64)

    legal_ids = np.full((n_rows, w), -1.0, dtype=np.float32)
    scores = np.full((n_rows, w), np.nan, dtype=np.float32)
    for i, r in enumerate(rows):
        c = len(r)
        legal_ids[i, :c] = np.arange(c, dtype=np.float32)
        scores[i, :c] = np.asarray(r, dtype=np.float32)
    scores_mask = np.isfinite(scores)

    prefix = np.arange(w)[None, :] < counts[:, None]
    offsets = np.empty(n_rows + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    offsets.tofile(corpus_dir / "row_offsets.dat")

    np.ascontiguousarray(legal_ids[prefix].astype(np.float32)).tofile(
        corpus_dir / "legal_action_ids.dat"
    )
    np.ascontiguousarray(scores[prefix].astype(np.float32)).tofile(
        corpus_dir / "target_scores.dat"
    )
    np.ascontiguousarray(scores_mask[prefix].astype(np.bool_)).tofile(
        corpus_dir / "target_scores_mask.dat"
    )

    seat = np.arange(n_rows, dtype=np.int64)
    np.ascontiguousarray(seat).tofile(corpus_dir / "seat.dat")

    categories: list[str] = []
    codes = np.empty(n_rows, dtype=np.int32)
    for i in range(n_rows):
        phase = _PHASES[i % len(_PHASES)]
        if phase not in categories:
            categories.append(phase)
        codes[i] = categories.index(phase)
    np.ascontiguousarray(codes).tofile(corpus_dir / "phase.codes.dat")

    meta = {
        "schema": "memmap_corpus_v1",
        "row_count": n_rows,
        "flat_count": int(counts.sum()),
        "legal_width": w,
        "columns": {
            "legal_action_ids": {"kind": "ragged2d", "dtype": "<f4", "fill": -1.0},
            "target_scores": {"kind": "ragged2d", "dtype": "<f4", "fill": float("nan")},
            "target_scores_mask": {"kind": "ragged2d", "dtype": "|b1", "fill": 0.0},
            "seat": {"kind": "fixed", "dtype": "<i8", "inner_shape": []},
            "phase": {"kind": "string", "categories": categories},
        },
        "stats": {},
    }
    (corpus_dir / "corpus_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
    )


def _authenticate_corpus(corpus_dir: Path) -> dict:
    meta_path = corpus_dir / "corpus_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    inventory = [
        {
            "filename": filename,
            "size_bytes": (corpus_dir / filename).stat().st_size,
            "sha256": "sha256:" + rl.sha256_file(corpus_dir / filename),
        }
        for filename in sorted(train_bc._expected_memmap_payload_filenames(meta))
    ]
    meta["payload_inventory_schema"] = train_bc.MEMMAP_PAYLOAD_INVENTORY_SCHEMA
    meta["payload_inventory"] = inventory
    meta["payload_inventory_sha256"] = train_bc._canonical_json_sha256(inventory)
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return meta


def _fresh_q_for_indices(indices: np.ndarray, legal_width: int, corpus) -> np.ndarray:
    """Deterministic fresh per-action q aligned to GLOBAL row indices: entry
    (row, slot) = 1000 + 10*global_row + slot for legal slots, NaN elsewhere. Makes
    every rewritten entry uniquely traceable to its source row + slot."""
    legal = np.asarray(corpus["legal_action_ids"][indices])
    counts = np.sum(legal >= 0, axis=1).astype(np.int64)
    q = np.full((len(indices), legal_width), np.nan, dtype=np.float32)
    for k, gi in enumerate(np.asarray(indices)):
        c = int(counts[k])
        q[k, :c] = 1000.0 + 10.0 * float(gi) + np.arange(c, dtype=np.float32)
    return q


@pytest.fixture()
def corpus_dir(tmp_path: Path) -> Path:
    d = tmp_path / "corpus"
    _make_corpus(d, n_rows=20)
    return d


@pytest.fixture()
def fake_forward(monkeypatch):
    """Patch rl.load_policy + rl.batch_forward so both the fleet job and the
    single-shot reference use the SAME deterministic forward (no torch/model)."""
    monkeypatch.setattr(rl, "load_policy", lambda *a, **k: object())

    calls: list[tuple[int, int]] = []

    def _fake_batch_forward(
        policy,
        corpus,
        indices,
        *,
        batch_size,
        want_q,
        legal_width,
        progress_every=0,
        value_materialization=None,
    ):
        indices = np.asarray(indices, dtype=np.int64)
        if indices.size:
            calls.append((int(indices.min()), int(indices.max()) + 1))
        result = {"value": (indices.astype(np.float64) * 0.01).astype(np.float32)}
        if want_q:
            result["q_values"] = _fresh_q_for_indices(indices, legal_width, corpus)
        return result

    monkeypatch.setattr(rl, "batch_forward", _fake_batch_forward)
    return calls


def _fake_ckpt(tmp_path: Path) -> Path:
    torch = pytest.importorskip("torch")
    ckpt = tmp_path / "champion.pt"
    torch.save(
        {"policy_type": "entity_graph", "mask_hidden_info": False, "model": {}}, ckpt
    )
    return ckpt


def _write_q_head_provenance(path: Path, checkpoint: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": rl.Q_HEAD_PROVENANCE_SCHEMA,
                "checkpoint_md5": rl.md5_file(checkpoint),
                "q_head": {
                    "trained": True,
                    "target_semantics": rl.Q_HEAD_TARGET_SEMANTICS,
                    "value_range": [-1, 1],
                },
                "validation": {
                    "passed": True,
                    "evidence": "pytest://banked-q-head-calibration",
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _plan(corpus_dir, job_dir, ckpt, *, v_component="target_scores", chunk_rows=3):
    _, meta = rl.resolve_reanalyzer_checkpoint(
        mode="checkpoint",
        checkpoint=ckpt,
        ema_checkpoints=None,
        ema_decay=0.75,
        work_dir=job_dir,
    )
    q_head_provenance = (
        _write_q_head_provenance(
            job_dir.parent / f"{job_dir.name}_q_provenance.json", ckpt
        )
        if rl.V_COMPONENTS[v_component]["forward_output"] == "q_values"
        else None
    )
    return rbc.do_plan(
        corpus_dir=corpus_dir,
        job_dir=job_dir,
        reanalyzer_meta=meta,
        v_component=v_component,
        chunk_rows=chunk_rows,
        mask_hidden_info=False,
        force=False,
        q_head_provenance=q_head_provenance,
    )


def _run(job_dir, ckpt, *, max_chunks=None, use_claim=False):
    return rbc.do_run(
        job_dir=job_dir,
        reanalyzer_path=ckpt,
        device="cpu",
        batch_size=2,
        max_chunks=max_chunks,
        chunk_ids=None,
        use_claim=use_claim,
        claim_stale_sec=rbc.DEFAULT_CLAIM_STALE_SEC,
        progress_every=0,
    )


# --------------------------------------------------------------------------- #
# Chunk planning
# --------------------------------------------------------------------------- #
def test_plan_chunks_partition_is_contiguous_and_complete(corpus_dir):
    meta = rbc.load_meta(corpus_dir)
    offsets = rbc.load_row_offsets(corpus_dir, meta["row_count"])
    chunks = rbc.plan_chunks(offsets, chunk_rows=3)
    # 20 rows / 3 -> 7 chunks (six of 3, one of 2).
    assert len(chunks) == 7
    assert chunks[0]["row_start"] == 0
    assert chunks[-1]["row_end"] == 20
    assert chunks[-1]["n_rows"] == 2
    # contiguous, no gaps/overlaps
    for a, b in zip(chunks, chunks[1:]):
        assert a["row_end"] == b["row_start"]
    assert sum(c["n_rows"] for c in chunks) == 20
    assert sum(c["n_flat"] for c in chunks) == meta["flat_count"]
    # does not raise
    rbc.validate_partition(chunks, meta["row_count"], meta["flat_count"])


def test_validate_partition_detects_gap(corpus_dir):
    meta = rbc.load_meta(corpus_dir)
    offsets = rbc.load_row_offsets(corpus_dir, meta["row_count"])
    chunks = rbc.plan_chunks(offsets, chunk_rows=3)
    chunks[2]["row_start"] += 1  # introduce a gap
    with pytest.raises(SystemExit, match="gap/overlap"):
        rbc.validate_partition(chunks, meta["row_count"], meta["flat_count"])


def test_plan_cli_requires_explicit_q_component():
    with pytest.raises(SystemExit):
        rbc.build_arg_parser().parse_args(
            [
                "plan",
                "--corpus",
                "corpus",
                "--job-dir",
                "job",
                "--checkpoint",
                "checkpoint.pt",
            ]
        )


def test_plan_refuses_q_values_without_provenance(corpus_dir, tmp_path):
    ckpt = _fake_ckpt(tmp_path)
    _, meta = rl.resolve_reanalyzer_checkpoint(
        mode="checkpoint",
        checkpoint=ckpt,
        ema_checkpoints=None,
        ema_decay=0.75,
        work_dir=tmp_path,
    )
    with pytest.raises(SystemExit, match="REFUSING --v-component target_scores"):
        rbc.do_plan(
            corpus_dir=corpus_dir,
            job_dir=tmp_path / "unsafe_job",
            reanalyzer_meta=meta,
            v_component="target_scores",
            chunk_rows=3,
            mask_hidden_info=False,
            force=False,
        )


def test_run_refuses_legacy_q_plan_without_provenance(
    corpus_dir,
    tmp_path,
    fake_forward,
):
    ckpt = _fake_ckpt(tmp_path)
    job_dir = tmp_path / "job"
    _plan(corpus_dir, job_dir, ckpt, chunk_rows=3)
    manifest_path = job_dir / rbc._MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("q_head_provenance")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(SystemExit, match="REFUSING --v-component target_scores"):
        _run(job_dir, ckpt)
    with pytest.raises(SystemExit, match="REFUSING --v-component target_scores"):
        rbc.do_merge(
            job_dir=job_dir,
            out_dir=tmp_path / "unsafe_overlay",
            link_mode="copy",
            mix_fraction=None,
        )


def test_run_and_merge_refuse_legacy_root_plan_without_materialization_provenance(
    corpus_dir,
    tmp_path,
    fake_forward,
):
    job_dir = tmp_path / "root_job"
    job_dir.mkdir()
    manifest_path = job_dir / rbc._MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(
            {
                "v_component": "root_value",
                "reanalyzer": {"md5": "0" * 32},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="single stored-feature forward"):
        _run(job_dir, _fake_ckpt(tmp_path))
    with pytest.raises(SystemExit, match="single stored-feature forward"):
        rbc.do_merge(
            job_dir=job_dir,
            out_dir=tmp_path / "unsafe_root_overlay",
            link_mode="copy",
            mix_fraction=None,
        )


# --------------------------------------------------------------------------- #
# Value-only correctness: chunked+merged == single-shot reference
# --------------------------------------------------------------------------- #
def test_chunked_merge_matches_single_shot_reference(
    corpus_dir, tmp_path, fake_forward
):
    pytest.importorskip("torch")
    source_meta = _authenticate_corpus(corpus_dir)
    ckpt = _fake_ckpt(tmp_path)
    v_component = "target_scores"

    # Reference: single-shot reanalyze_lite full run.
    _, meta = rl.resolve_reanalyzer_checkpoint(
        mode="checkpoint",
        checkpoint=ckpt,
        ema_checkpoints=None,
        ema_decay=0.75,
        work_dir=tmp_path,
    )
    ref_out = tmp_path / "reference"
    rl.run_reanalyze(
        corpus_dir=corpus_dir,
        out_dir=ref_out,
        reanalyzer_path=ckpt,
        reanalyzer_meta=meta,
        v_component=v_component,
        device="cpu",
        batch_size=2,
        mask_hidden_info=False,
        sample=None,
        seed=0,
        progress_every=0,
        q_head_provenance=(
            _write_q_head_provenance(tmp_path / "reference_q_provenance.json", ckpt)
            if rl.V_COMPONENTS[v_component]["forward_output"] == "q_values"
            else None
        ),
    )

    # Fleet: plan -> run (all) -> merge.
    job_dir = tmp_path / "job"
    _plan(corpus_dir, job_dir, ckpt, v_component=v_component, chunk_rows=3)
    _run(job_dir, ckpt)
    overlay = tmp_path / "overlay"
    rbc.do_merge(job_dir=job_dir, out_dir=overlay, link_mode="copy", mix_fraction=None)

    # The reanalyzed column is byte-identical between the two paths.
    ref_bytes = (ref_out / f"{v_component}.dat").read_bytes()
    overlay_bytes = (overlay / f"{v_component}.dat").read_bytes()
    assert overlay_bytes == ref_bytes

    # And it loads back through MemmapCorpus identically.
    ref_col = np.asarray(MemmapCorpus(ref_out)[v_component])
    ovl_col = np.asarray(MemmapCorpus(overlay)[v_component])
    np.testing.assert_array_equal(
        np.nan_to_num(ref_col, nan=-999.0), np.nan_to_num(ovl_col, nan=-999.0)
    )
    overlay_meta = json.loads((overlay / "corpus_meta.json").read_text())
    assert overlay_meta["payload_inventory_sha256"] != source_meta[
        "payload_inventory_sha256"
    ]
    assert (
        train_bc._validate_memmap_payload_inventory(overlay, overlay_meta)
        == overlay_meta["payload_inventory_sha256"]
    )


def test_merge_no_loss_no_dup_and_overlay_loads(corpus_dir, tmp_path, fake_forward):
    pytest.importorskip("torch")
    ckpt = _fake_ckpt(tmp_path)
    job_dir = tmp_path / "job"
    _plan(corpus_dir, job_dir, ckpt, chunk_rows=3)
    _run(job_dir, ckpt)
    overlay = tmp_path / "overlay"
    manifest = rbc.do_merge(
        job_dir=job_dir, out_dir=overlay, link_mode="copy", mix_fraction=0.2
    )

    assert manifest["no_loss_no_dup_verified"] is True
    assert manifest["row_count_before"] == manifest["row_count_after"] == 20
    assert manifest["assembled_entries"] == manifest["expected_entries"]

    # Unchanged columns byte-identical to source.
    src, ovl = MemmapCorpus(corpus_dir), MemmapCorpus(overlay)
    assert len(ovl) == len(src)
    for key in ("legal_action_ids", "seat", "phase", "target_scores_mask"):
        a, b = np.asarray(src[key]), np.asarray(ovl[key])
        if a.dtype.kind == "U":
            np.testing.assert_array_equal(a.astype(str), b.astype(str), err_msg=key)
        else:
            np.testing.assert_array_equal(a, b, err_msg=key)

    # Mix plan recorded.
    assert manifest["mix_plan"]["mix_fraction"] == 0.2
    assert manifest["mix_plan"]["reanalyzed_rows"] == 20


def test_merge_refuses_before_all_chunks_done(corpus_dir, tmp_path, fake_forward):
    pytest.importorskip("torch")
    ckpt = _fake_ckpt(tmp_path)
    job_dir = tmp_path / "job"
    _plan(corpus_dir, job_dir, ckpt, chunk_rows=3)
    _run(job_dir, ckpt, max_chunks=2)  # only 2 of 7 chunks
    with pytest.raises(SystemExit, match="not done"):
        rbc.do_merge(
            job_dir=job_dir,
            out_dir=tmp_path / "overlay",
            link_mode="copy",
            mix_fraction=None,
        )


# --------------------------------------------------------------------------- #
# Resumability / preemption safety
# --------------------------------------------------------------------------- #
def test_resume_completes_remaining_without_reprocessing(
    corpus_dir, tmp_path, fake_forward
):
    pytest.importorskip("torch")
    ckpt = _fake_ckpt(tmp_path)
    job_dir = tmp_path / "job"
    manifest = _plan(corpus_dir, job_dir, ckpt, chunk_rows=3)
    md5 = manifest["reanalyzer"]["md5"]
    n_chunks = manifest["n_chunks"]

    # First slot: process 2 chunks, then "preempt".
    fake_forward.clear()
    _run(job_dir, ckpt, max_chunks=2)
    done_after_first = [
        c["chunk_id"] for c in manifest["chunks"] if rbc.chunk_is_done(job_dir, c, md5)
    ]
    assert len(done_after_first) == 2
    first_ranges = list(fake_forward)
    assert len(first_ranges) == 2  # exactly 2 chunks forwarded

    # Capture the done markers of the finished chunks to prove they aren't rewritten.
    markers_before = {
        cid: json.loads(rbc._done_path(job_dir, cid).read_text())["completed_utc"]
        for cid in done_after_first
    }

    # Simulate a crash MID-CHUNK on a not-yet-done chunk: a truncated piece, NO done
    # marker (the "never assume a chunk finishes" case).
    victim = next(
        c for c in manifest["chunks"] if c["chunk_id"] not in done_after_first
    )
    rbc._piece_path(job_dir, victim["chunk_id"]).write_bytes(
        b"\x00\x00\x00"
    )  # partial garbage
    assert not rbc.chunk_is_done(job_dir, victim, md5)

    # Resume: process the remainder.
    fake_forward.clear()
    _run(job_dir, ckpt)

    # All chunks now done.
    done_all = [
        c["chunk_id"] for c in manifest["chunks"] if rbc.chunk_is_done(job_dir, c, md5)
    ]
    assert len(done_all) == n_chunks

    # The resume forwarded EXACTLY the previously-unfinished chunks (5), not the 2
    # already-done -> no double-processing. The truncated victim was redone.
    resumed_row_starts = {lo for lo, _hi in fake_forward}
    already_done_starts = {
        c["row_start"] for c in manifest["chunks"] if c["chunk_id"] in done_after_first
    }
    assert len(fake_forward) == n_chunks - 2
    assert resumed_row_starts.isdisjoint(already_done_starts)

    # Finished chunks' markers are UNTOUCHED (same completed_utc).
    for cid, ts in markers_before.items():
        assert (
            json.loads(rbc._done_path(job_dir, cid).read_text())["completed_utc"] == ts
        )

    # Merged result still matches the reference single-shot run byte-for-byte.
    _, meta2 = rl.resolve_reanalyzer_checkpoint(
        mode="checkpoint",
        checkpoint=ckpt,
        ema_checkpoints=None,
        ema_decay=0.75,
        work_dir=tmp_path,
    )
    ref_out = tmp_path / "reference"
    rl.run_reanalyze(
        corpus_dir=corpus_dir,
        out_dir=ref_out,
        reanalyzer_path=ckpt,
        reanalyzer_meta=meta2,
        v_component="target_scores",
        device="cpu",
        batch_size=2,
        mask_hidden_info=False,
        sample=None,
        seed=0,
        progress_every=0,
        q_head_provenance=_write_q_head_provenance(
            tmp_path / "reference_q_provenance.json", ckpt
        ),
    )
    overlay = tmp_path / "overlay"
    rbc.do_merge(job_dir=job_dir, out_dir=overlay, link_mode="copy", mix_fraction=None)
    assert (overlay / "target_scores.dat").read_bytes() == (
        ref_out / "target_scores.dat"
    ).read_bytes()


def test_corrupt_piece_after_done_is_detected(corpus_dir, tmp_path, fake_forward):
    pytest.importorskip("torch")
    ckpt = _fake_ckpt(tmp_path)
    job_dir = tmp_path / "job"
    manifest = _plan(corpus_dir, job_dir, ckpt, chunk_rows=3)
    md5 = manifest["reanalyzer"]["md5"]
    _run(job_dir, ckpt, max_chunks=1)
    chunk0 = manifest["chunks"][0]
    assert rbc.chunk_is_done(job_dir, chunk0, md5)

    # Corrupt the piece AFTER the done marker was written (silent bit-rot). The hash
    # in the marker no longer matches -> chunk reported NOT done -> would be redone.
    piece = rbc._piece_path(job_dir, 0)
    data = bytearray(piece.read_bytes())
    data[0] ^= 0xFF
    piece.write_bytes(bytes(data))
    assert not rbc.chunk_is_done(job_dir, chunk0, md5)


def test_done_marker_from_different_checkpoint_is_not_trusted(
    corpus_dir, tmp_path, fake_forward
):
    pytest.importorskip("torch")
    ckpt = _fake_ckpt(tmp_path)
    job_dir = tmp_path / "job"
    manifest = _plan(corpus_dir, job_dir, ckpt, chunk_rows=3)
    _run(job_dir, ckpt, max_chunks=1)
    chunk0 = manifest["chunks"][0]
    # A piece built by the manifest checkpoint is not "done" for a different net.
    assert rbc.chunk_is_done(job_dir, chunk0, manifest["reanalyzer"]["md5"])
    assert not rbc.chunk_is_done(job_dir, chunk0, "deadbeef" * 4)


def test_done_marker_with_stale_row_range_is_not_trusted(
    corpus_dir, tmp_path, fake_forward
):
    """A piece/marker left over from a discarded --force re-plan (different
    --chunk-rows) must never be silently reused just because its chunk_id number
    recurred: the row range it actually covers no longer matches this chunk_id's
    new span, so it must be redone."""
    pytest.importorskip("torch")
    ckpt = _fake_ckpt(tmp_path)
    job_dir = tmp_path / "job"
    manifest = _plan(corpus_dir, job_dir, ckpt, chunk_rows=3)
    md5 = manifest["reanalyzer"]["md5"]
    _run(job_dir, ckpt, max_chunks=1)
    chunk0 = manifest["chunks"][0]
    assert rbc.chunk_is_done(job_dir, chunk0, md5)

    # Simulate chunk_id 0 now meaning a DIFFERENT (wider) row range, as it would
    # after a --force re-plan with a larger --chunk-rows.
    reshaped = dict(chunk0)
    reshaped["row_end"] = chunk0["row_end"] + 5
    reshaped["n_rows"] = chunk0["n_rows"] + 5
    reshaped["n_flat"] = chunk0["n_flat"] + 5
    assert not rbc.chunk_is_done(job_dir, reshaped, md5)


def test_force_replan_with_different_shape_purges_stale_chunks(
    corpus_dir, tmp_path, fake_forward
):
    """--force replacing a plan whose shape actually changed (different
    --chunk-rows) discards the old chunks/ state, as the docstring promises,
    instead of leaving orphaned pieces from the discarded configuration on disk."""
    pytest.importorskip("torch")
    ckpt = _fake_ckpt(tmp_path)
    job_dir = tmp_path / "job"
    _plan(corpus_dir, job_dir, ckpt, chunk_rows=3)
    _run(job_dir, ckpt, max_chunks=2)
    assert any(rbc._chunks_dir(job_dir).glob("chunk_*.done.json"))

    _, meta = rl.resolve_reanalyzer_checkpoint(
        mode="checkpoint",
        checkpoint=ckpt,
        ema_checkpoints=None,
        ema_decay=0.75,
        work_dir=job_dir,
    )
    new_manifest = rbc.do_plan(
        corpus_dir=corpus_dir,
        job_dir=job_dir,
        reanalyzer_meta=meta,
        v_component="target_scores",
        chunk_rows=7,
        mask_hidden_info=False,
        force=True,
        q_head_provenance=_write_q_head_provenance(
            tmp_path / "replan_q_provenance.json", ckpt
        ),
    )
    assert new_manifest["chunk_rows"] == 7
    assert not any(rbc._chunks_dir(job_dir).glob("chunk_*.done.json"))
    assert not any(rbc._chunks_dir(job_dir).glob("chunk_*.dat"))


# --------------------------------------------------------------------------- #
# Checkpoint-swap detection (manifest pins the md5; `run` must enforce it)
# --------------------------------------------------------------------------- #
def test_run_detects_checkpoint_swapped_after_plan(corpus_dir, tmp_path, fake_forward):
    pytest.importorskip("torch")
    torch = pytest.importorskip("torch")
    ckpt = _fake_ckpt(tmp_path)
    job_dir = tmp_path / "job"
    _plan(corpus_dir, job_dir, ckpt, chunk_rows=3)

    # Swap the checkpoint file IN PLACE at the same path (different weights/content)
    # after planning -- the manifest's pinned md5 no longer matches reality.
    torch.save(
        {
            "policy_type": "entity_graph",
            "mask_hidden_info": False,
            "model": {},
            "swapped": True,
        },
        ckpt,
    )

    with pytest.raises(SystemExit, match="swapped after planning"):
        _run(job_dir, ckpt)


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
def test_provenance_manifests(corpus_dir, tmp_path, fake_forward):
    pytest.importorskip("torch")
    ckpt = _fake_ckpt(tmp_path)
    job_dir = tmp_path / "job"
    manifest = _plan(corpus_dir, job_dir, ckpt, chunk_rows=3)
    assert manifest["reanalyzer"]["md5"]
    assert manifest["tool"] == "reanalyze_banked_corpus"
    assert manifest["n_chunks"] == 7
    assert manifest["q_head_provenance"]["schema"] == rl.Q_HEAD_PROVENANCE_SCHEMA
    assert manifest["q_head_provenance"]["source_sha256"]

    _run(job_dir, ckpt, max_chunks=1)
    marker = json.loads(rbc._done_path(job_dir, 0).read_text())
    assert marker["piece_sha256"]
    assert marker["reanalyzer_md5"] == manifest["reanalyzer"]["md5"]
    assert marker["piece_bytes"] > 0
    assert "elapsed_s" in marker and "rows_per_s" in marker

    _run(job_dir, ckpt)
    overlay = tmp_path / "overlay"
    merge = rbc.do_merge(
        job_dir=job_dir, out_dir=overlay, link_mode="copy", mix_fraction=None
    )
    on_disk = json.loads((overlay / "reanalyze_merge_manifest.json").read_text())
    assert on_disk == merge
    assert merge["reanalyzer"]["md5"] == manifest["reanalyzer"]["md5"]
    assert merge["q_head_provenance"] == manifest["q_head_provenance"]
    assert merge["kind"] == "versioned_overlay"
    assert "total_gpu_hours" in merge


# --------------------------------------------------------------------------- #
# Status
# --------------------------------------------------------------------------- #
def test_status_counts_and_cost(corpus_dir, tmp_path, fake_forward, capsys):
    pytest.importorskip("torch")
    ckpt = _fake_ckpt(tmp_path)
    job_dir = tmp_path / "job"
    _plan(corpus_dir, job_dir, ckpt, chunk_rows=3)
    _run(job_dir, ckpt, max_chunks=3)
    status = rbc.do_status(job_dir=job_dir)
    assert status["n_chunks"] == 7
    assert status["done"] == 3
    assert status["pending"] == 4
    assert status["complete"] is False
    assert status["gpu_hours_consumed"] >= 0.0
    _run(job_dir, ckpt)
    status2 = rbc.do_status(job_dir=job_dir)
    assert status2["done"] == 7
    assert status2["complete"] is True


# --------------------------------------------------------------------------- #
# Overlay uses hardlinks by default (near-zero extra disk vs a 417GB copy)
# --------------------------------------------------------------------------- #
def test_overlay_hardlinks_unchanged_columns(corpus_dir, tmp_path, fake_forward):
    pytest.importorskip("torch")
    ckpt = _fake_ckpt(tmp_path)
    job_dir = tmp_path / "job"
    _plan(corpus_dir, job_dir, ckpt, chunk_rows=3)
    _run(job_dir, ckpt)
    overlay = tmp_path / "overlay"
    rbc.do_merge(
        job_dir=job_dir, out_dir=overlay, link_mode="hardlink", mix_fraction=None
    )

    # Unchanged column shares an inode with the source (hardlink).
    assert (overlay / "seat.dat").stat().st_ino == (
        corpus_dir / "seat.dat"
    ).stat().st_ino
    # Rewritten column is a DISTINCT file (source untouched).
    assert (overlay / "target_scores.dat").stat().st_ino != (
        corpus_dir / "target_scores.dat"
    ).stat().st_ino


# --------------------------------------------------------------------------- #
# Claim files (fleet parallelism)
# --------------------------------------------------------------------------- #
def test_claim_excl_and_stale_steal(tmp_path):
    job_dir = tmp_path / "job"
    rbc._chunks_dir(job_dir).mkdir(parents=True)
    # First claim succeeds.
    assert rbc._try_claim(job_dir, 0, stale_sec=3600.0) is True
    # A second (fresh) claim on the same chunk is refused.
    assert rbc._try_claim(job_dir, 0, stale_sec=3600.0) is False
    # With a zero staleness window, the claim is considered stale and stolen.
    time.sleep(0.01)
    assert rbc._try_claim(job_dir, 0, stale_sec=0.0) is True
    rbc._release_claim(job_dir, 0)
    assert not rbc._claim_path(job_dir, 0).exists()


def test_run_with_claims_skips_already_claimed(corpus_dir, tmp_path, fake_forward):
    pytest.importorskip("torch")
    ckpt = _fake_ckpt(tmp_path)
    job_dir = tmp_path / "job"
    _plan(corpus_dir, job_dir, ckpt, chunk_rows=3)
    # Pre-claim chunk 0 as if another worker holds it (fresh).
    rbc._try_claim(job_dir, 0, stale_sec=3600.0)
    summary = _run(job_dir, ckpt, use_claim=True)
    assert 0 in summary["skipped_claimed"]
    assert 0 not in summary["processed"]


# --------------------------------------------------------------------------- #
# Mix plan
# --------------------------------------------------------------------------- #
def test_compute_mix_plan_basic():
    plan = rbc.compute_mix_plan(
        reanalyzed_rows=1_000_000, window_size=100_000, mix_fraction=0.2
    )
    assert plan["banked_rows_to_draw"] == 20_000
    assert plan["fresh_rows"] == 80_000
    assert plan["capped"] is False


def test_compute_mix_plan_capped_by_available():
    plan = rbc.compute_mix_plan(
        reanalyzed_rows=5_000, window_size=100_000, mix_fraction=0.2
    )
    assert plan["banked_rows_to_draw"] == 5_000  # capped at what exists
    assert plan["fresh_rows"] == 95_000
    assert plan["capped"] is True


def test_compute_mix_plan_rejects_bad_fraction():
    with pytest.raises(SystemExit, match="mix-fraction"):
        rbc.compute_mix_plan(reanalyzed_rows=10, window_size=10, mix_fraction=1.5)
