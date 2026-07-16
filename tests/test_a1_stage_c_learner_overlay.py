from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tools import a1_stage_c_learner_overlay as overlay
from tools import train_bc


def _write(path: Path, values: np.ndarray) -> None:
    values.tofile(path)


def test_policy_projection_disables_old_targets_and_maps_action_ids(tmp_path: Path) -> None:
    base = tmp_path / "base"
    derived = tmp_path / "derived"
    base.mkdir()
    derived.mkdir()
    offsets = np.asarray([0, 2, 5, 7], dtype=np.int64)
    legal = np.asarray([1, 2, 10, 20, 30, 4, 5], dtype=np.int64)
    _write(base / "row_offsets.dat", offsets)
    _write(base / "game_seed.dat", np.asarray([100, 200, 300], dtype=np.int64))
    _write(base / "decision_index.dat", np.asarray([1, 2, 3], dtype=np.int64))
    _write(base / "legal_action_ids.dat", legal)
    _write(base / "value_target.dat", np.asarray([0.1, -0.2, 0.3], dtype=np.float32))
    _write(base / "teacher_name.codes.dat", np.zeros(3, dtype=np.int32))

    meta = {
        "row_count": 3,
        "flat_count": 7,
        "columns": {
            "game_seed": {"kind": "fixed", "dtype": "int64", "inner_shape": []},
            "decision_index": {
                "kind": "fixed",
                "dtype": "int64",
                "inner_shape": [],
            },
            "legal_action_ids": {"kind": "ragged2d", "dtype": "int64"},
            "value_target": {
                "kind": "fixed",
                "dtype": "float32",
                "inner_shape": [],
            },
            "policy_weight_multiplier": {
                "kind": "fixed",
                "dtype": "float32",
                "inner_shape": [],
            },
            "prior_policy": {"kind": "ragged2d", "dtype": "float32"},
            "target_policy": {"kind": "ragged2d", "dtype": "float32"},
            "target_policy_mask": {"kind": "ragged2d", "dtype": "bool"},
            "target_scores": {"kind": "ragged2d", "dtype": "float32"},
            "target_scores_mask": {"kind": "ragged2d", "dtype": "bool"},
            "teacher_name": {"kind": "string", "categories": ["historical"]},
        },
    }
    overlay._hardlink_payloads(base, derived, meta["columns"])  # noqa: SLF001
    patch = {
        "row_index": np.asarray([1], dtype=np.int64),
        "game_seed": np.asarray([200], dtype=np.int64),
        "decision_index": np.asarray([2], dtype=np.int64),
        "legal_action_offsets": np.asarray([0, 3], dtype=np.int64),
        # Deliberately differs from base order [10, 20, 30].
        "legal_action_ids_flat": np.asarray([30, 10, 20], dtype=np.int64),
        "target_policy_flat": np.asarray([0.6, 0.1, 0.3], dtype=np.float32),
        "target_policy_mask_flat": np.asarray([True, True, True]),
        "prior_policy_flat": np.asarray([0.5, 0.2, 0.3], dtype=np.float32),
        "target_scores_flat": np.asarray([3.0, 1.0, 2.0], dtype=np.float32),
        "target_scores_mask_flat": np.asarray([True, True, True]),
    }

    evidence = overlay._project_policy_patch(  # noqa: SLF001
        base_root=base,
        output_root=derived,
        meta=meta,
        patch=patch,
    )

    assert evidence["selected_rows"] == 1
    assert evidence["base_value_rows_retained"] == 3
    assert (base / "value_target.dat").stat().st_ino == (
        derived / "value_target.dat"
    ).stat().st_ino
    weights = np.fromfile(derived / "policy_weight_multiplier.dat", dtype=np.float32)
    targets = np.fromfile(derived / "target_policy.dat", dtype=np.float32)
    target_mask = np.fromfile(derived / "target_policy_mask.dat", dtype=np.bool_)
    priors = np.fromfile(derived / "prior_policy.dat", dtype=np.float32)
    scores = np.fromfile(derived / "target_scores.dat", dtype=np.float32)
    teacher_codes = np.fromfile(derived / "teacher_name.codes.dat", dtype=np.int32)
    assert weights.tolist() == [0.0, 1.0, 0.0]
    assert not target_mask[:2].any() and not target_mask[5:].any()
    assert targets[2:5] == pytest.approx([0.1, 0.3, 0.6])
    assert priors[2:5] == pytest.approx([0.2, 0.3, 0.5])
    assert scores[2:5] == pytest.approx([1.0, 2.0, 3.0])
    assert np.all(targets[:2] == 0.0) and np.all(targets[5:] == 0.0)
    assert np.isnan(scores[:2]).all() and np.isnan(scores[5:]).all()
    assert teacher_codes.tolist() == [0, 1, 0]
    assert meta["columns"]["teacher_name"]["categories"] == [
        "historical",
        overlay.POLICY_TEACHER,
    ]


def test_unique_source_row_count_is_exact_and_fail_closed() -> None:
    local = {7, 2, 7, 4}
    ddp = {"enabled": False, "world_size": 1, "rank": 0, "local_rank": 0}
    assert train_bc._reduce_unique_row_count(local, total_rows=8, ddp=ddp) == 3
    with pytest.raises(ValueError, match="outside the corpus"):
        train_bc._reduce_unique_row_count({8}, total_rows=8, ddp=ddp)
