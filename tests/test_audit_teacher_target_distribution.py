from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.audit_teacher_target_distribution import (
    analyze_memmap,
    analyze_npz,
    compare_reports,
)


def _write_shard(path: Path) -> None:
    path.parent.mkdir(parents=True)
    np.savez_compressed(
        path,
        game_seed=np.asarray([1, 1, 2]),
        decision_index=np.asarray([0, 1, 0], dtype=np.int32),
        phase=np.asarray(["PLAY_TURN", "PLAY_TURN", "MOVE_ROBBER"]),
        action_taken=np.asarray([10, 10, 21], dtype=np.int16),
        legal_action_ids=np.asarray([[10, 11], [10, -1], [20, 21]], dtype=np.int16),
        is_forced=np.asarray([False, True, False]),
        used_full_search=np.asarray([True, False, True]),
        simulations_used=np.asarray([128, 0, 128], dtype=np.int32),
        policy_weight_multiplier=np.asarray([1.0, 0.0, 1.0], dtype=np.float32),
        value_weight_multiplier=np.ones(3, dtype=np.float32),
        target_policy=np.asarray(
            [[0.75, 0.25], [1.0, 0.0], [0.5, 0.5]], dtype=np.float32
        ),
        prior_policy=np.asarray(
            [[0.5, 0.5], [1.0, 0.0], [0.75, 0.25]], dtype=np.float16
        ),
        target_policy_mask=np.asarray([[True, True], [True, False], [True, True]]),
        terminated=np.asarray([True, True, True]),
        truncated=np.asarray([False, False, False]),
    )
    progress = path.parent / "progress.json"
    progress.write_text(json.dumps({"games_failed": 0}))


def test_npz_audit_excludes_forced_rows_from_policy_metrics(tmp_path: Path) -> None:
    _write_shard(tmp_path / "worker_000" / "shard.npz")
    report, provenance = analyze_npz(tmp_path, compact_out=tmp_path / "compact.npz")
    assert report["rows"] == 3
    assert report["games"] == 2
    assert report["forced"]["rows"] == 1
    assert report["policy_active"]["rows"] == 2
    assert report["full_search"]["rows"] == 2
    assert report["failures"] == 0
    assert report["policy_targets"]["kl_target_prior"]["count"] == 2
    blend = report["policy_targets"]["played_action_blend"]
    assert blend["soft_target_weight"] == 0.9
    assert blend["hard_action_weight"] == pytest.approx(0.1)
    assert blend["played_target_probability"]["mean"] == pytest.approx((0.75 + 0.5) / 2)
    assert blend["played_is_target_mode_fraction"] == 1.0
    assert (
        blend["effective_target_entropy"]["mean"]
        < report["policy_targets"]["target_entropy"]["mean"]
    )
    assert provenance["shard_count"] == 1
    assert provenance["compact_corpus"]["sha256"].startswith("sha256:")


def test_comparison_materiality_uses_target_signal_not_mix() -> None:
    base = {
        "forced": {"fraction": 0.5},
        "full_search": {"fraction": 0.12},
        "policy_active": {"fraction": 0.12},
        "rows_per_game": {"mean": 200.0},
        "phase_distribution": {"PLAY_TURN": {"fraction": 1.0}},
        "policy_targets": {
            "target_entropy": {"mean": 0.7},
            "prior_entropy": {"mean": 0.9},
            "kl_target_prior": {"mean": 0.5},
            "by_phase": {},
        },
    }
    changed = json.loads(json.dumps(base))
    changed["policy_targets"]["target_entropy"]["mean"] = 0.64
    comparison = compare_reports(changed, base)
    assert comparison["material_target_change"] is True
    assert comparison["target_entropy_mean"]["absolute_delta"] < -0.05


def test_memmap_segmented_policy_metrics_and_category_filter(tmp_path: Path) -> None:
    columns = {
        name: {"kind": "fixed", "dtype": dtype, "inner_shape": []}
        for name, dtype in {
            "game_seed": "<i8",
            "is_forced": "|b1",
            "used_full_search": "|b1",
            "policy_weight_multiplier": "<f4",
            "terminated": "|b1",
            "truncated": "|b1",
        }.items()
    }
    columns["phase"] = {"kind": "string", "categories": ["PLAY_TURN", "MOVE_ROBBER"]}
    columns["action_taken"] = {"kind": "fixed", "dtype": "<i2", "inner_shape": []}
    columns["legal_action_ids"] = {"kind": "ragged2d", "dtype": "<i2", "fill": -1}
    (tmp_path / "corpus_meta.json").write_text(
        json.dumps({"schema": "memmap_corpus_v1", "row_count": 3, "columns": columns})
    )
    arrays = {
        "game_seed": np.asarray([10, 10, 20], dtype=np.int64),
        "is_forced": np.asarray([False, True, False]),
        "used_full_search": np.asarray([True, False, True]),
        "policy_weight_multiplier": np.asarray([1.0, 0.0, 1.0], dtype=np.float32),
        "terminated": np.asarray([True, True, True]),
        "truncated": np.asarray([False, False, False]),
        "action_taken": np.asarray([10, 12, 20], dtype=np.int16),
    }
    for name, array in arrays.items():
        array.tofile(tmp_path / f"{name}.dat")
    np.asarray([0, 0, 1], dtype=np.int32).tofile(tmp_path / "phase.codes.dat")
    np.asarray([0, 2, 3, 6], dtype=np.int64).tofile(tmp_path / "row_offsets.dat")
    np.asarray([10, 11, 12, 20, 21, 22], dtype=np.int16).tofile(
        tmp_path / "legal_action_ids.dat"
    )
    np.asarray([0.75, 0.25, 1.0, 0.5, 0.25, 0.25], dtype=np.float32).tofile(
        tmp_path / "target_policy.dat"
    )
    np.asarray([0.5, 0.5, 1.0, 0.6, 0.2, 0.2], dtype=np.float32).tofile(
        tmp_path / "prior_policy.dat"
    )
    seeds = tmp_path / "seeds.json"
    seeds.write_text(
        json.dumps({"records": [{"game_seed": 10, "category": "current"}]})
    )
    report, provenance = analyze_memmap(
        tmp_path, seed_manifest=seeds, category="current", chunk_rows=2
    )
    assert report["rows"] == 2
    assert report["games"] == 1
    assert report["policy_active"]["rows"] == 1
    assert report["policy_targets"]["kl_target_prior"]["count"] == 1
    assert report["policy_targets"]["played_action_blend"]["played_target_probability"][
        "mean"
    ] == pytest.approx(0.75)
    assert provenance["category"] == "current"

    # Replay memmaps created before the explicit is_forced column remain
    # auditable; forcedness is exactly reconstructed from ragged row width.
    meta = json.loads((tmp_path / "corpus_meta.json").read_text())
    del meta["columns"]["is_forced"]
    del meta["columns"]["used_full_search"]
    (tmp_path / "corpus_meta.json").write_text(json.dumps(meta))
    (tmp_path / "is_forced.dat").unlink()
    (tmp_path / "used_full_search.dat").unlink()
    legacy_report, _ = analyze_memmap(tmp_path, chunk_rows=2)
    assert legacy_report["forced"]["rows"] == 1
    assert legacy_report["full_search"]["rows"] == 2
