from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import train_bc  # type: ignore  # noqa: E402
import a1_lineage_dose as lineage  # type: ignore  # noqa: E402


def _parent(tmp_path: Path) -> tuple[argparse.Namespace, dict[str, object]]:
    producer_sha = "sha256:" + "1" * 64
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"sealed-parent")
    checkpoint_sha = train_bc._sha256_existing_file(checkpoint)  # noqa: SLF001
    receipt = {
        "schema_version": "a1-dual-arm-training-receipt-v1",
        "status": "complete",
        "arm_id": "n256",
        "subset_id": "full-56k",
        "inputs": {"producer": {"path": "/producer.pt", "sha256": producer_sha}},
        "outputs": {
            "checkpoint": {"path": str(checkpoint.resolve()), "sha256": checkpoint_sha}
        },
        "lineage_dose": lineage.direct_lineage_dose(
            declared_producer_sha256=producer_sha,
            init_checkpoint_sha256=producer_sha,
            current_sampled_rows=56_000,
            current_optimizer_steps=1_000,
        ),
    }
    receipt["receipt_sha256"] = train_bc._canonical_json_sha256(receipt)  # noqa: SLF001
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    args = argparse.Namespace(
        a1_curriculum_parent_receipt=str(receipt_path),
        init_checkpoint=str(checkpoint),
        init_checkpoint_sha256=checkpoint_sha,
    )
    return args, {
        "producer_checkpoint_sha256": producer_sha,
        "arm_id": "n128",
        "subset_id": "full-140k",
    }


def test_curriculum_parent_authenticates_completed_n256_dose(tmp_path: Path) -> None:
    args, bound = _parent(tmp_path)
    value = train_bc._validate_a1_curriculum_parent(args, bound)  # noqa: SLF001
    assert value is not None
    assert value["parent_arm_id"] == "n256"
    assert value["parent_checkpoint"]["sha256"] == args.init_checkpoint_sha256


def test_curriculum_parent_rejects_checkpoint_byte_drift(tmp_path: Path) -> None:
    args, bound = _parent(tmp_path)
    Path(args.init_checkpoint).write_bytes(b"changed")
    try:
        train_bc._validate_a1_curriculum_parent(args, bound)  # noqa: SLF001
    except SystemExit as error:
        assert "does not bind producer/init checkpoint" in str(error)
    else:
        raise AssertionError("checkpoint drift was accepted")
