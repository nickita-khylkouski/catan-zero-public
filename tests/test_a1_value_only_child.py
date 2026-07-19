from __future__ import annotations

import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import a1_value_only_child as child  # type: ignore  # noqa: E402


def _ref(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": child._sha256(path)}  # noqa: SLF001


def _receipt(tmp_path: Path) -> tuple[Path, Path]:
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"parent")
    checkpoint = tmp_path / "child.pt"
    checkpoint.write_bytes(b"policy child")
    report = tmp_path / "child.report.json"
    report.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint.resolve()),
                "init_checkpoint_sha256": child._sha256(parent),  # noqa: SLF001
                "train_value_only": False,
                "policy_training_signal": {
                    "trained_policy_objective": True,
                    "status": "trained",
                },
            }
        ),
        encoding="utf-8",
    )
    payload = {
        "schema_version": child.SCHEMA,
        "mode": "value_only_child",
        "promotion_eligible": False,
        "parent_producer": _ref(parent),
        "child_checkpoint": _ref(checkpoint),
        "child_training_report": _ref(report),
    }
    payload["receipt_sha256"] = child._canonical_sha256(payload)  # noqa: SLF001
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    return receipt, checkpoint


def test_value_only_child_receipt_binds_trained_child(tmp_path: Path) -> None:
    receipt, checkpoint = _receipt(tmp_path)
    value = child.verify_receipt(receipt)
    assert value["child_checkpoint"]["path"] == str(checkpoint.resolve())
    assert value["promotion_eligible"] is False


def test_value_only_child_receipt_rejects_checkpoint_drift(tmp_path: Path) -> None:
    receipt, checkpoint = _receipt(tmp_path)
    checkpoint.write_bytes(b"tampered")
    try:
        child.verify_receipt(receipt)
    except child.ValueOnlyChildError as error:
        assert "child checkpoint" in str(error)
    else:
        raise AssertionError("tampered child checkpoint was accepted")


def test_value_only_child_receipt_rejects_nonpolicy_child(tmp_path: Path) -> None:
    receipt, _ = _receipt(tmp_path)
    payload = json.loads(receipt.read_text())
    report_path = Path(payload["child_training_report"]["path"])
    report = json.loads(report_path.read_text())
    report["policy_training_signal"]["trained_policy_objective"] = False
    report_path.write_text(json.dumps(report), encoding="utf-8")
    payload["child_training_report"] = _ref(report_path)
    unsigned = dict(payload)
    unsigned.pop("receipt_sha256")
    payload["receipt_sha256"] = child._canonical_sha256(unsigned)  # noqa: SLF001
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    try:
        child.verify_receipt(receipt)
    except child.ValueOnlyChildError as error:
        assert "policy-trained child" in str(error)
    else:
        raise AssertionError("non-policy child was accepted")
