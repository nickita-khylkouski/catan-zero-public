from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import a1_diagnostic_training_receipt as receipt


def _write(path: Path, value: bytes | str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value.encode() if isinstance(value, str) else value)
    return path


def _json(path: Path, value: dict) -> Path:
    return _write(path, json.dumps(value, sort_keys=True) + "\n")


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    source = _write(tmp_path / "repo" / "tools" / "train_bc.py", "trainer")
    probe = _write(tmp_path / "repo" / "tools" / "probe.py", "probe")
    descriptor = _write(tmp_path / "data.json", "descriptor")
    parent = _write(tmp_path / "f7.pt", b"parent")
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "candidate.pt"
    for name in receipt.EXPECTED_ARTIFACTS:
        _write(run_dir / name, b"artifact:" + name.encode())
    command = ["python", str(source), "--data", str(descriptor)]
    run = {
        "run_id": "ddp8-b512",
        "run_dir": str(run_dir),
        "command": command,
        "command_sha256": receipt._digest(command),  # noqa: SLF001
        "gpu_ids": list(range(8)),
        "world_size": 8,
        "local_batch_size": 512,
        "global_batch_size": 4096,
        "grad_accum_steps": 1,
        "max_steps": 1024,
        "planned_samples": 4_194_304,
    }
    plan = {
        "schema_version": "fixture-plan-v1",
        "diagnostic_only": True,
        "promotion_eligible": False,
        "inputs": {
            "data": str(descriptor),
            "data_sha256": receipt._file_ref(descriptor)["sha256"],  # noqa: SLF001
            "init_checkpoint": str(parent),
            "init_checkpoint_sha256": receipt._file_ref(parent)["sha256"],  # noqa: SLF001
        },
        "runs": [run],
        "runtime": {
            "trainer": str(source),
            "trainer_sha256": receipt._file_ref(source)["sha256"],  # noqa: SLF001
            "quality_probe": str(probe),
            "quality_probe_sha256": receipt._file_ref(probe)["sha256"],  # noqa: SLF001
        },
    }
    plan["plan_sha256"] = receipt._digest(plan)  # noqa: SLF001
    plan_path = _json(tmp_path / "plan.json", plan)
    _json(
        run_dir / "runtime.json",
        {
            "returncode": 0,
            "run_id": run["run_id"],
            "plan_sha256": plan["plan_sha256"],
            "command_sha256": run["command_sha256"],
            "gpu_ids": run["gpu_ids"],
            "started_unix_ns": 10,
            "finished_unix_ns": 20,
        },
    )
    _json(
        run_dir / "train.report.json",
        {
            "diagnostic_only": True,
            "promotion_eligible": False,
            "checkpoint": str(checkpoint.resolve()),
            "init_checkpoint_sha256": plan["inputs"]["init_checkpoint_sha256"],
            "world_size": 8,
            "batch_size": 512,
            "effective_global_batch_size": 4096,
            "grad_accum_steps": 1,
            "max_steps": 1024,
            "steps_completed": 1024,
            "training_row_draws": 4_194_304,
            "optimizer_restored": False,
            "resume_optimizer": False,
            "lr": 3e-5,
            "value_lr_mult": 0.3,
            "checkout_runtime_binding": {
                "trainer": str(source.resolve()),
                "trainer_sha256": receipt._file_ref(source)["sha256"],  # noqa: SLF001
            },
        },
    )
    return plan_path, run_dir


def test_finalize_and_replay_exact_receipt(tmp_path: Path) -> None:
    plan, run_dir = _fixture(tmp_path)
    target = run_dir / "training.receipt.json"
    payload = receipt.build_receipt(plan, run_id="ddp8-b512")
    receipt._write_exclusive(target, payload)  # noqa: SLF001

    assert receipt.verify_receipt(target) == payload
    assert payload["status"] == "complete_nonpromotable"
    assert payload["outputs"]["candidate.pt"]["sha256"].startswith("sha256:")
    with pytest.raises(FileExistsError):
        receipt._write_exclusive(target, payload)  # noqa: SLF001


def test_report_or_artifact_drift_is_rejected(tmp_path: Path) -> None:
    plan, run_dir = _fixture(tmp_path)
    report = json.loads((run_dir / "train.report.json").read_text())
    report["steps_completed"] = 1023
    _json(run_dir / "train.report.json", report)
    with pytest.raises(receipt.ReceiptError, match="report invariant drift"):
        receipt.build_receipt(plan, run_id="ddp8-b512")


def test_receipt_replay_rejects_mutated_checkpoint(tmp_path: Path) -> None:
    plan, run_dir = _fixture(tmp_path)
    target = run_dir / "training.receipt.json"
    payload = receipt.build_receipt(plan, run_id="ddp8-b512")
    receipt._write_exclusive(target, payload)  # noqa: SLF001
    (run_dir / "candidate.pt").write_bytes(b"different")
    with pytest.raises(receipt.ReceiptError, match="no longer replays"):
        receipt.verify_receipt(target)
