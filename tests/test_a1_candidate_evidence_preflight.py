from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tools import a1_candidate_evidence_preflight as preflight


def _seed_digest(seeds: list[int]) -> str:
    values = np.sort(np.asarray(seeds, dtype=np.int64))
    return "sha256:" + hashlib.sha256(values.astype("<i8").tobytes()).hexdigest()


def _fixture(tmp_path: Path, *, count: int = 240) -> argparse.Namespace:
    candidate = tmp_path / "candidate.pt"
    candidate.write_bytes(b"candidate")
    candidate_sha = preflight._sha256(candidate)
    seeds = list(range(1000, 1000 + count))
    seed_sha = _seed_digest(seeds)
    manifest = tmp_path / "validation.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "train-validation-game-seeds-v1",
                "game_seeds": seeds,
                "validation_game_seed_count": count,
                "validation_game_seed_set_sha256": seed_sha,
            }
        ),
        encoding="utf-8",
    )
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "promotion_eligible": True,
                "promotion_block_reason": None,
                "checkpoint": str(candidate),
                "checkpoint_sha256": candidate_sha,
                "a1_contract_sha256": "sha256:" + "1" * 64,
                "a1_central_published_executor_authority": {"schema": "test"},
                "validation_game_seed_manifest": str(manifest),
                "validation_game_seed_count": count,
                "validation_game_seed_set_sha256": seed_sha,
            }
        ),
        encoding="utf-8",
    )
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}", encoding="utf-8")
    lock = tmp_path / "lock.json"
    lock.write_text("{}", encoding="utf-8")
    return argparse.Namespace(
        candidate=candidate,
        training_report=report,
        training_receipt=receipt,
        contract_lock=lock,
        minimum_validation_games=240,
    )


def test_ready_candidate_passes_cheap_scheduling_preflight(tmp_path: Path) -> None:
    result = preflight.inspect_candidate(_fixture(tmp_path))

    assert result["promotion_evidence_ready"] is True
    assert result["failures"] == []


def test_diagnostic_candidate_refuses_expensive_evidence(tmp_path: Path) -> None:
    args = _fixture(tmp_path, count=64)
    report = json.loads(args.training_report.read_text(encoding="utf-8"))
    report.update(
        {
            "promotion_eligible": False,
            "promotion_block_reason": "requires_sealed_a1_one_dose_execution_receipt",
            "checkpoint_sha256": None,
            "a1_contract_sha256": None,
            "a1_central_published_executor_authority": None,
        }
    )
    args.training_report.write_text(json.dumps(report), encoding="utf-8")
    args.training_receipt = None
    args.contract_lock = None

    result = preflight.inspect_candidate(args)
    codes = {failure["code"] for failure in result["failures"]}

    assert result["promotion_evidence_ready"] is False
    assert result["diagnostic_evaluation_allowed"] is True
    assert {
        "training_report_not_promotion_eligible",
        "training_report_has_promotion_block",
        "training_report_missing_checkpoint_sha256",
        "training_report_missing_contract_authority",
        "training_report_missing_executor_authority",
        "validation_cohort_too_small_for_high_regret",
        "missing_training_receipt",
        "missing_contract_lock",
    } <= codes
