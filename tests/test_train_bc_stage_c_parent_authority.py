from __future__ import annotations

from pathlib import Path

import pytest

from tools import a1_stage_c_learner_overlay as overlay
from tools import train_bc


def _sha(character: str) -> str:
    return "sha256:" + character * 64


def test_stage_c_overlay_binds_reanalyzer_without_rewriting_trajectory_producer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    teacher = {"path": "/checkpoints/v15.pt", "sha256": _sha("1")}
    producer = _sha("2")
    identity = _sha("3")
    meta_overlay = {
        "schema_version": overlay.ADMISSION_OVERLAY_SCHEMA,
        "target_policy_target_identity_sha256": identity,
        "target_reanalyzer_checkpoint": teacher,
        "selected_policy_rows": 91_208,
    }
    admission = {
        "corpus": {
            "data_path": str(tmp_path),
            "producer_checkpoint_sha256": producer,
        },
        "stage_c_policy_overlay": dict(meta_overlay),
    }
    monkeypatch.setattr(
        overlay,
        "verify_overlay_admission",
        lambda path: {
            "path": str(path),
            "admission": admission,
            "receipt": {"target_reanalyzer_checkpoint": teacher},
        },
    )

    result = train_bc._validate_stage_c_overlay_learner_parent(  # noqa: SLF001
        tmp_path, {"stage_c_policy_overlay": meta_overlay}
    )

    assert result == {
        "learner_parent_checkpoint_sha256": teacher["sha256"],
        "learner_initializer_sha256": teacher["sha256"],
        "policy_target_producer_checkpoint_sha256": teacher["sha256"],
        "trajectory_producer_checkpoint_sha256": producer,
    }


def test_stage_c_overlay_refuses_teacher_identity_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    meta_teacher = {"path": "/checkpoints/v15.pt", "sha256": _sha("1")}
    admitted_teacher = {"path": "/checkpoints/other.pt", "sha256": _sha("4")}
    identity = _sha("3")
    meta_overlay = {
        "schema_version": overlay.ADMISSION_OVERLAY_SCHEMA,
        "target_policy_target_identity_sha256": identity,
        "target_reanalyzer_checkpoint": meta_teacher,
        "selected_policy_rows": 91_208,
    }
    monkeypatch.setattr(
        overlay,
        "verify_overlay_admission",
        lambda _path: {
            "admission": {
                "corpus": {
                    "data_path": str(tmp_path),
                    "producer_checkpoint_sha256": _sha("2"),
                },
                "stage_c_policy_overlay": {
                    **meta_overlay,
                    "target_reanalyzer_checkpoint": admitted_teacher,
                },
            },
            "receipt": {"target_reanalyzer_checkpoint": admitted_teacher},
        },
    )

    with pytest.raises(SystemExit, match="one authenticated learner authority"):
        train_bc._validate_stage_c_overlay_learner_parent(  # noqa: SLF001
            tmp_path, {"stage_c_policy_overlay": meta_overlay}
        )


def test_non_stage_c_corpus_retains_historical_parent_rule(tmp_path: Path) -> None:
    assert (
        train_bc._validate_stage_c_overlay_learner_parent(  # noqa: SLF001
            tmp_path, {"row_count": 1}
        )
        is None
    )
