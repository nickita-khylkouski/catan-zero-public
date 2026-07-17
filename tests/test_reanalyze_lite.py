"""Unit tests for tools/reanalyze_lite.py (CAT-34).

Covers, on a tiny SYNTHETIC memmap corpus crafted directly on disk (no real
shards, no model, no GPU):

* rewrite correctness -- only finite (masked) legal entries of the v-component are
  overwritten with the fresh values; NaN pads and in-prefix NaNs are preserved.
* original untouched -- the source corpus bytes are byte-identical before/after.
* provenance manifest -- present, with source hash, checkpoint md5, reanalyzer
  config, before/after stats, and the integrity result.
* backward-compat load -- the rewritten corpus loads through MemmapCorpus and the
  v-component column reads back exactly the rewritten values.
* per-state root_value materialisation -- a brand-new scalar column is added and
  loads back through MemmapCorpus.
* integrity guard -- an unexpected change to another column is detected.
* EMA reanalyzer resolution -- --reanalyzer-net ema averages checkpoints.
* batch-forward assembly -- batching + q/value alignment (fake forward).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
_TOOLS_DIR = _REPO / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import reanalyze_lite as rl  # type: ignore  # noqa: E402
import train_bc  # type: ignore  # noqa: E402
from train_bc import MemmapCorpus  # type: ignore  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic corpus builder (matches build_memmap_corpus's trimmed-flat layout)
# --------------------------------------------------------------------------- #
# Row 0: 3 legal, scores [0.1, 0.2, nan]  (one in-prefix NaN -> stays NaN)
# Row 1: 1 legal, scores [-0.5]
# Row 2: 4 legal, scores [0.3, 0.4, 0.5, 0.6]
# Row 3: 2 legal, scores [nan, 0.9]        (leading in-prefix NaN -> stays NaN)
# Row 4: 2 legal, scores [0.0, -0.1]
_LEGAL_WIDTH = 4
_ROWS = [
    ([0.1, 0.2, np.nan], "robber"),
    ([-0.5], "roll"),
    ([0.3, 0.4, 0.5, 0.6], "build"),
    ([np.nan, 0.9], "robber"),
    ([0.0, -0.1], "build"),
]


def _make_synthetic_corpus(corpus_dir: Path) -> None:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    n = len(_ROWS)
    w = _LEGAL_WIDTH
    counts = np.array([len(scores) for scores, _ in _ROWS], dtype=np.int64)

    legal_ids = np.full((n, w), -1.0, dtype=np.float32)
    scores = np.full((n, w), np.nan, dtype=np.float32)
    for i, (row_scores, _phase) in enumerate(_ROWS):
        c = len(row_scores)
        legal_ids[i, :c] = np.arange(c, dtype=np.float32)
        scores[i, :c] = np.asarray(row_scores, dtype=np.float32)
    scores_mask = np.isfinite(scores)

    prefix = np.arange(w)[None, :] < counts[:, None]
    offsets = np.empty(n + 1, dtype=np.int64)
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

    # A decoy scalar fixed column (must stay byte-identical through the rewrite).
    seat = np.arange(n, dtype=np.int64)
    np.ascontiguousarray(seat).tofile(corpus_dir / "seat.dat")

    # phase string column -> codes + categories.
    phases = [phase for _scores, phase in _ROWS]
    categories: list[str] = []
    codes = np.empty(n, dtype=np.int32)
    for i, phase in enumerate(phases):
        if phase not in categories:
            categories.append(phase)
        codes[i] = categories.index(phase)
    np.ascontiguousarray(codes).tofile(corpus_dir / "phase.codes.dat")

    meta = {
        "schema": "memmap_corpus_v1",
        "row_count": n,
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


def _fresh_q_all() -> np.ndarray:
    """Deterministic fresh per-action q for all rows: 100 + action_id per legal
    slot, so overwritten entries are trivially distinguishable from originals."""
    n = len(_ROWS)
    w = _LEGAL_WIDTH
    q = np.full((n, w), np.nan, dtype=np.float32)
    for i, (row_scores, _phase) in enumerate(_ROWS):
        c = len(row_scores)
        q[i, :c] = 100.0 + np.arange(c, dtype=np.float32)
    return q


@pytest.fixture()
def corpus_dir(tmp_path: Path) -> Path:
    d = tmp_path / "corpus"
    _make_synthetic_corpus(d)
    return d


def _write_q_head_provenance(
    directory: Path,
    checkpoint: Path,
    *,
    checkpoint_md5: str | None = None,
) -> Path:
    path = directory / "q_head_provenance.json"
    path.write_text(
        json.dumps(
            {
                "schema": rl.Q_HEAD_PROVENANCE_SCHEMA,
                "checkpoint_md5": checkpoint_md5 or rl.md5_file(checkpoint),
                "q_head": {
                    "trained": True,
                    "target_semantics": rl.Q_HEAD_TARGET_SEMANTICS,
                    "value_range": [-1, 1],
                },
                "validation": {
                    "passed": True,
                    "evidence": "pytest://trained-q-head-calibration",
                },
            }
        ),
        encoding="utf-8",
    )
    return path


# --------------------------------------------------------------------------- #
# Rewrite correctness + backward-compat load
# --------------------------------------------------------------------------- #
def test_per_action_rewrite_overwrites_only_finite_legal_entries(corpus_dir, tmp_path):
    corpus = MemmapCorpus(corpus_dir)
    out_dir = tmp_path / "out"
    import shutil

    shutil.copytree(corpus_dir, out_dir)
    fresh = _fresh_q_all()
    rewrite = rl.rewrite_per_action_column(
        corpus, out_dir, "target_scores", fresh, legal_width=_LEGAL_WIDTH
    )

    # Row 0 had one in-prefix NaN (slot 2) -> only slots 0,1 rewritten.
    # Row 3 had a leading in-prefix NaN (slot 0) -> only slot 1 rewritten.
    expected_rewritten = 2 + 1 + 4 + 1 + 2  # rows 0..4 finite legal counts
    assert rewrite["entries_rewritten"] == expected_rewritten

    reloaded = MemmapCorpus(out_dir)
    got = np.asarray(reloaded["target_scores"])
    old = np.asarray(corpus["target_scores"])

    # Slots that were finite are now the fresh q; NaNs (pad or in-prefix) unchanged.
    finite = np.isfinite(old)
    np.testing.assert_array_equal(got[finite], fresh[finite])
    assert np.all(np.isnan(got[~finite]))
    # Specifically: in-prefix NaNs preserved.
    assert np.isnan(got[0, 2])
    assert np.isnan(got[3, 0])
    assert got[3, 1] == pytest.approx(101.0)


def test_source_corpus_untouched(corpus_dir, tmp_path, monkeypatch):
    before = rl.hash_corpus_dats(corpus_dir)
    before_meta = rl.sha256_file(corpus_dir / "corpus_meta.json")

    _run_full_with_fake_forward(
        corpus_dir, tmp_path, monkeypatch, v_component="target_scores"
    )

    after = rl.hash_corpus_dats(corpus_dir)
    after_meta = rl.sha256_file(corpus_dir / "corpus_meta.json")
    assert before == after, "source .dat files changed"
    assert before_meta == after_meta, "source corpus_meta.json changed"


def test_full_run_manifest_and_integrity(corpus_dir, tmp_path, monkeypatch):
    source_meta = _authenticate_corpus(corpus_dir)
    manifest = _run_full_with_fake_forward(
        corpus_dir, tmp_path, monkeypatch, v_component="target_scores"
    )
    out_dir = Path(manifest["output_corpus"])

    # Manifest is present and self-consistent.
    on_disk = json.loads((out_dir / "reanalyze_manifest.json").read_text())
    assert on_disk == manifest
    assert manifest["tool"] == "reanalyze_lite"
    assert manifest["v_component"] == "target_scores"
    assert manifest["forward_output"] == "q_values"
    assert manifest["q_head_provenance"]["schema"] == rl.Q_HEAD_PROVENANCE_SCHEMA
    assert manifest["q_head_provenance"]["source_sha256"]
    assert manifest["reanalyzer"]["md5"]  # checkpoint md5 recorded
    assert manifest["source_corpus"]["corpus_meta_sha256"]
    assert manifest["source_corpus"]["dat_sha256"]  # per-file source hashes
    assert manifest["integrity"]["unchanged_columns_verified"] is True
    assert manifest["integrity"]["unexpectedly_changed_files"] == []
    assert (
        manifest["integrity"]["row_count_before"]
        == manifest["integrity"]["row_count_after"]
    )
    assert manifest["integrity"]["expected_changed_files"] == ["target_scores.dat"]
    output_meta = json.loads((out_dir / "corpus_meta.json").read_text())
    assert output_meta["payload_inventory_sha256"] != source_meta[
        "payload_inventory_sha256"
    ]
    assert (
        train_bc._validate_memmap_payload_inventory(out_dir, output_meta)
        == output_meta["payload_inventory_sha256"]
    )

    # Stats carry a real mean shift + per-phase deltas.
    stats = manifest["stats"]
    assert stats["entries"] == 10
    assert "mean_shift" in stats and "correlation" in stats
    assert set(stats["per_phase"]) == {"robber", "roll", "build"}


def test_backward_compat_load_through_memmap_corpus(corpus_dir, tmp_path, monkeypatch):
    manifest = _run_full_with_fake_forward(
        corpus_dir, tmp_path, monkeypatch, v_component="target_scores"
    )
    out_dir = Path(manifest["output_corpus"])
    reloaded = MemmapCorpus(out_dir)
    assert len(reloaded) == len(MemmapCorpus(corpus_dir))
    # Every other column reads back identically to the source.
    src = MemmapCorpus(corpus_dir)
    for key in ("legal_action_ids", "seat", "phase", "target_scores_mask"):
        a = np.asarray(src[key])
        b = np.asarray(reloaded[key])
        if a.dtype.kind == "U":
            np.testing.assert_array_equal(a.astype(str), b.astype(str), err_msg=key)
        else:
            np.testing.assert_array_equal(a, b, err_msg=key)


@pytest.mark.parametrize("component", ["root_value", "root_prior_value"])
def test_value_component_reanalysis_is_refused(component):
    with pytest.raises(SystemExit, match="single stored-feature forward"):
        rl.validate_v_component(component)


def test_afterstate_target_reanalysis_is_refused():
    with pytest.raises(SystemExit, match="immediate one-ply evaluator"):
        rl.validate_v_component("afterstate_target")


def test_integrity_guard_detects_unexpected_change(corpus_dir, tmp_path, monkeypatch):
    # Corrupt an unrelated column in the COPY after copytree, before hashing, by
    # patching rewrite_per_action_column to also scribble on seat.dat.
    real_rewrite = rl.rewrite_per_action_column

    def _sabotage(corpus, dst_dir, name, fresh_q, *, legal_width):
        result = real_rewrite(corpus, dst_dir, name, fresh_q, legal_width=legal_width)
        # Tamper with a column the rewrite is not supposed to touch.
        (Path(dst_dir) / "seat.dat").write_bytes(b"\x00" * 8 * len(corpus))
        return result

    monkeypatch.setattr(rl, "rewrite_per_action_column", _sabotage)
    with pytest.raises(SystemExit, match="INTEGRITY FAILURE"):
        _run_full_with_fake_forward(
            corpus_dir, tmp_path, monkeypatch, v_component="target_scores"
        )


def test_out_dir_refuses_to_overwrite(corpus_dir, tmp_path, monkeypatch):
    out = tmp_path / "existing"
    out.mkdir()
    with pytest.raises(SystemExit, match="already exists"):
        _run_full_with_fake_forward(
            corpus_dir, tmp_path, monkeypatch, v_component="target_scores", out_dir=out
        )


# --------------------------------------------------------------------------- #
# Q-head safety boundary
# --------------------------------------------------------------------------- #
def test_cli_requires_an_explicit_q_component():
    with pytest.raises(SystemExit):
        rl.build_arg_parser().parse_args(
            ["--corpus", "corpus", "--checkpoint", "ckpt.pt"]
        )


def test_q_values_component_requires_explicit_provenance(tmp_path):
    meta = {"md5": "a" * 32}
    with pytest.raises(SystemExit, match="REFUSING --v-component target_scores"):
        rl.validate_q_head_provenance(
            None,
            reanalyzer_meta=meta,
            v_component="target_scores",
        )


def test_q_head_provenance_is_bound_to_exact_checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"real checkpoint")
    provenance = _write_q_head_provenance(
        tmp_path,
        checkpoint,
        checkpoint_md5="0" * 32,
    )
    with pytest.raises(SystemExit, match="does not match reanalyzer"):
        rl.validate_q_head_provenance(
            provenance,
            reanalyzer_meta={"md5": rl.md5_file(checkpoint)},
            v_component="target_scores",
        )


def test_programmatic_run_metadata_is_bound_to_actual_reanalyzer_bytes(tmp_path):
    checkpoint_a = tmp_path / "a.pt"
    checkpoint_b = tmp_path / "b.pt"
    checkpoint_a.write_bytes(b"checkpoint a")
    checkpoint_b.write_bytes(b"checkpoint b")

    assert rl.verify_reanalyzer_identity(
        checkpoint_a, {"md5": rl.md5_file(checkpoint_a)}
    ) == rl.md5_file(checkpoint_a)
    with pytest.raises(SystemExit, match="reanalyzer checkpoint md5 mismatch"):
        rl.verify_reanalyzer_identity(checkpoint_b, {"md5": rl.md5_file(checkpoint_a)})


def test_root_value_rejects_irrelevant_q_provenance(tmp_path):
    with pytest.raises(SystemExit, match="single stored-feature forward"):
        rl.validate_q_head_provenance(
            {"schema": rl.Q_HEAD_PROVENANCE_SCHEMA},
            reanalyzer_meta={"md5": "a" * 32},
            v_component="root_value",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("trained", False, "q_head.trained must be true"),
        ("target_semantics", "normalized_teacher_preference", "target_semantics"),
        ("passed", False, "validation.passed must be true"),
    ],
)
def test_q_head_provenance_rejects_untrained_wrong_semantics_or_unvalidated(
    tmp_path,
    field,
    value,
    message,
):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    provenance = _write_q_head_provenance(tmp_path, checkpoint)
    payload = json.loads(provenance.read_text(encoding="utf-8"))
    if field == "passed":
        payload["validation"][field] = value
    else:
        payload["q_head"][field] = value
    provenance.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SystemExit, match=message):
        rl.validate_q_head_provenance(
            provenance,
            reanalyzer_meta={"md5": rl.md5_file(checkpoint)},
            v_component="target_scores",
        )


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #
def test_compute_stats_mean_shift_and_correlation():
    before = np.array([0.0, 1.0, 2.0, 3.0])
    after = np.array([0.5, 1.5, 2.5, 3.5])
    rewrite = {"before": before, "after": after, "row_index_per_entry": np.arange(4)}
    stats = rl.compute_stats(rewrite, phases=None)
    assert stats["mean_shift"] == pytest.approx(0.5)
    assert stats["correlation"] == pytest.approx(1.0)
    assert stats["entries"] == 4


# --------------------------------------------------------------------------- #
# EMA reanalyzer resolution
# --------------------------------------------------------------------------- #
def test_resolve_reanalyzer_ema(tmp_path):
    torch = pytest.importorskip("torch")

    def _ckpt(bias: float, step: int) -> dict:
        return {
            "policy_type": "entity_graph",
            "config": {"hidden_size": 8, "graph_layers": 2},
            "action_mask_version": "v1",
            "mask_hidden_info": True,
            "static_action_features": torch.zeros(3, 3),
            "model": {
                "trunk.weight": torch.full((4, 4), bias),
                "num_batches_tracked": torch.tensor(step, dtype=torch.int64),
            },
        }

    paths = []
    for i, bias in enumerate([0.0, 1.0, 2.0]):
        p = tmp_path / f"ckpt_{i}.pt"
        torch.save(_ckpt(bias, i), p)
        paths.append(p)

    work = tmp_path / "work"
    resolved, meta = rl.resolve_reanalyzer_checkpoint(
        mode="ema", checkpoint=None, ema_checkpoints=paths, ema_decay=0.5, work_dir=work
    )
    assert resolved == work / "reanalyzer_ema.pt"
    assert resolved.exists()
    assert meta["mode"] == "ema"
    assert meta["ema_decay"] == 0.5
    assert len(meta["ema_weights"]) == 3
    assert meta["ema_weights"][-1] > meta["ema_weights"][0]  # newest heaviest
    assert meta["policy_type"] == "entity_graph"
    assert meta["mask_hidden_info"] is True
    assert meta["md5"]


def test_resolve_reanalyzer_checkpoint_missing(tmp_path):
    with pytest.raises(SystemExit, match="checkpoint not found"):
        rl.resolve_reanalyzer_checkpoint(
            mode="checkpoint",
            checkpoint=tmp_path / "nope.pt",
            ema_checkpoints=None,
            ema_decay=0.75,
            work_dir=tmp_path,
        )


# --------------------------------------------------------------------------- #
# batch_forward assembly (fake forward, no model)
# --------------------------------------------------------------------------- #
def test_batch_forward_assembles_value_and_q(corpus_dir, monkeypatch):
    torch = pytest.importorskip("torch")
    import train_bc

    corpus = MemmapCorpus(corpus_dir)

    class _FakeModel:
        def eval(self):
            return self

    fake_policy = type(
        "P", (), {"model": _FakeModel(), "policy_type": "entity_graph"}
    )()

    def _fake_forward(policy, data, batch, legal_action_ids, *, return_q, **kwargs):
        w = legal_action_ids.shape[1]
        out = {"value": torch.tensor((np.asarray(batch) * 0.01), dtype=torch.float32)}
        if return_q:
            # q[row, col] = 100*global_row + col, so alignment is checkable.
            q = np.asarray(batch)[:, None] * 100.0 + np.arange(w)[None, :]
            out["q_values"] = torch.tensor(q, dtype=torch.float32)
        return out

    monkeypatch.setattr(train_bc, "_forward_legal_np_for_batch", _fake_forward)

    idx = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    fwd = rl.batch_forward(
        fake_policy, corpus, idx, batch_size=2, want_q=True, legal_width=_LEGAL_WIDTH
    )
    np.testing.assert_allclose(fwd["value"], idx * 0.01, atol=1e-6)
    for r in range(len(idx)):
        for c in range(_LEGAL_WIDTH):
            assert fwd["q_values"][r, c] == pytest.approx(idx[r] * 100.0 + c)


@pytest.mark.parametrize("defect", ["rank", "rows", "width", "nonfinite"])
def test_batch_forward_refuses_malformed_q_output(corpus_dir, monkeypatch, defect):
    torch = pytest.importorskip("torch")

    corpus = MemmapCorpus(corpus_dir)

    class _FakeModel:
        def eval(self):
            return self

    fake_policy = type("P", (), {"model": _FakeModel(), "policy_type": "entity_graph"})()

    def _fake_forward(policy, data, batch, legal_action_ids, *, return_q, **kwargs):
        rows, width = legal_action_ids.shape
        q = torch.zeros((rows, width), dtype=torch.float32)
        if defect == "rank":
            q = q.reshape(-1)
        elif defect == "rows":
            q = torch.zeros((rows + 1, width), dtype=torch.float32)
        elif defect == "width":
            q = torch.zeros((rows, width - 1), dtype=torch.float32)
        elif defect == "nonfinite":
            q[0, 0] = float("nan")
        return {"value": torch.zeros(rows), "q_values": q}

    monkeypatch.setattr(train_bc, "_forward_legal_np_for_batch", _fake_forward)
    with pytest.raises(SystemExit, match="q_values output"):
        rl.batch_forward(
            fake_policy,
            corpus,
            np.array([0, 1], dtype=np.int64),
            batch_size=2,
            want_q=True,
            legal_width=_LEGAL_WIDTH,
        )


def test_rewrite_refuses_masked_in_nonfinite_q_before_write(corpus_dir, tmp_path):
    corpus = MemmapCorpus(corpus_dir)
    out_dir = tmp_path / "out"
    import shutil

    shutil.copytree(corpus_dir, out_dir)
    before = (out_dir / "target_scores.dat").read_bytes()
    fresh = _fresh_q_all()
    fresh[0, 0] = np.nan
    with pytest.raises(SystemExit, match="masked target slot"):
        rl.rewrite_per_action_column(
            corpus, out_dir, "target_scores", fresh, legal_width=_LEGAL_WIDTH
        )
    assert (out_dir / "target_scores.dat").read_bytes() == before


# --------------------------------------------------------------------------- #
# Helper: full run with model + forward faked out
# --------------------------------------------------------------------------- #
def _run_full_with_fake_forward(
    corpus_dir, tmp_path, monkeypatch, *, v_component, out_dir=None
):
    torch = pytest.importorskip("torch")

    ckpt = tmp_path / "champion.pt"
    torch.save(
        {"policy_type": "entity_graph", "mask_hidden_info": False, "model": {}}, ckpt
    )

    monkeypatch.setattr(rl, "load_policy", lambda *a, **k: object())

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
        n = len(indices)
        result = {"value": (np.asarray(indices) * 0.01).astype(np.float32)}
        if want_q:
            q = np.full((n, legal_width), np.nan, dtype=np.float32)
            fresh = _fresh_q_all()
            q[:] = fresh[np.asarray(indices)]
            result["q_values"] = q
        return result

    monkeypatch.setattr(rl, "batch_forward", _fake_batch_forward)

    reanalyzer_path, reanalyzer_meta = rl.resolve_reanalyzer_checkpoint(
        mode="checkpoint",
        checkpoint=ckpt,
        ema_checkpoints=None,
        ema_decay=0.75,
        work_dir=tmp_path,
    )
    q_head_provenance = (
        _write_q_head_provenance(tmp_path, ckpt)
        if rl.V_COMPONENTS[v_component]["forward_output"] == "q_values"
        else None
    )
    return rl.run_reanalyze(
        corpus_dir=corpus_dir,
        out_dir=out_dir,
        reanalyzer_path=reanalyzer_path,
        reanalyzer_meta=reanalyzer_meta,
        v_component=v_component,
        device="cpu",
        batch_size=2,
        mask_hidden_info=False,
        sample=None,
        seed=0,
        progress_every=0,
        q_head_provenance=q_head_provenance,
    )
