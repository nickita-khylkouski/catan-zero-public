from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.fleet import a1_h100_eval_fleet as fleet
from tools.fleet import a1_lr_dose_eval_matrix as matrix


def test_completed_arms_share_exact_upgraded_initializer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    f7 = tmp_path / "f7.pt"
    initializer = tmp_path / "f7.upgraded.pt"
    f7.write_bytes(b"raw-f7")
    initializer.write_bytes(b"upgraded-f7")
    upgrade_receipt = tmp_path / "upgrade.receipt.json"
    upgrade_receipt.write_text("{}", encoding="utf-8")
    upgrade = {
        "source": {"path": str(f7), "sha256": fleet._sha256(f7)},
        "upgraded_initializer": {
            "path": str(initializer),
            "sha256": fleet._sha256(initializer),
        },
    }
    monkeypatch.setattr(
        matrix.architecture_upgrade, "verify_receipt", lambda _path: upgrade
    )

    candidates: dict[str, Path] = {}
    completed: dict[str, dict] = {}
    for arm in matrix.ARMS:
        candidate = tmp_path / f"{arm}.pt"
        candidate.write_bytes(f"candidate-{arm}".encode())
        candidates[arm] = candidate
        receipt = tmp_path / f"{arm}.receipt.json"
        receipt.write_text(
            json.dumps(
                {
                    "function_preserving_upgrade": upgrade,
                    "command": [
                        "train_bc.py",
                        "--init-checkpoint",
                        str(initializer),
                    ],
                }
            ),
            encoding="utf-8",
        )
        completed[arm] = {
            "receipt": str(receipt),
            "receipt_file_sha256": matrix._file_sha256(receipt),
            "status": "complete",
            "returncode": 0,
            "artifacts": {
                "checkpoint": {
                    "path": str(candidate),
                    "sha256": fleet._sha256(candidate),
                },
                "report": {"path": str(tmp_path / f"{arm}.report.json"), "sha256": "x"},
            },
        }
    monkeypatch.setattr(
        matrix.lr_campaign,
        "_verify_completed_arm_receipt",
        lambda _campaign, *, arm: completed[arm],
    )
    campaign = {
        "inputs": {
            "architecture_upgrade_receipt": str(upgrade_receipt),
            "architecture_upgrade_receipt_sha256": matrix._file_sha256(
                upgrade_receipt
            ),
        }
    }

    observed_upgrade, receipts = matrix._authenticate_completed_arms(
        campaign, f7=f7, candidates=candidates
    )
    assert observed_upgrade == upgrade
    assert {
        value["actual_initializer"]["sha256"] for value in receipts.values()
    } == {fleet._sha256(initializer)}

    bad = Path(completed["B"]["receipt"])
    bad.write_text(
        json.dumps(
            {
                "function_preserving_upgrade": upgrade,
                "command": ["train_bc.py", "--init-checkpoint", str(f7)],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(matrix.MatrixError, match="actual init"):
        matrix._authenticate_completed_arms(campaign, f7=f7, candidates=candidates)
