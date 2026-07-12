from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import a1_b200_microbatch_quality as quality


def _base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    descriptor = tmp_path / "composite.json"
    descriptor.write_text(
        json.dumps(
            {
                "schema_version": "memmap_composite_v2",
                "diagnostic_only": True,
                "promotion_eligible": False,
            }
        ),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "champion.pt"
    checkpoint.write_bytes(b"checkpoint")
    trainer = Path(quality.__file__).resolve().parent / "train_bc.py"
    monkeypatch.setattr(
        quality,
        "_runtime",
        lambda: {
            "repository_root": str(trainer.parents[1]),
            "repository_commit": "a" * 40,
            "trainer": str(trainer),
            "trainer_sha256": "sha256:" + "1" * 64,
            "quality_probe": str(Path(quality.__file__).resolve()),
            "quality_probe_sha256": "sha256:" + "2" * 64,
        },
    )
    command = [
        "/venv/python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=8",
        "/old/tools/train_bc.py",
        "--data",
        str(descriptor),
        "--data-format",
        "memmap",
        "--init-checkpoint",
        str(checkpoint),
        "--batch-size",
        "512",
        "--grad-accum-steps",
        "1",
        "--max-steps",
        "1024",
        "--epochs",
        "1",
        "--lr",
        "3e-5",
        "--lr-warmup-steps",
        "100",
        "--lr-schedule",
        "flat",
        "--seed",
        "1",
        "--checkpoint",
        "/old/candidate.pt",
        "--report",
        "/old/report.json",
    ]
    path = tmp_path / "command.json"
    path.write_text(json.dumps(command), encoding="utf-8")
    return path


def test_plan_matches_global_batch_warmup_and_total_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = quality.build_plan(
        base_command_json=_base(tmp_path, monkeypatch),
        output_dir=tmp_path / "probe",
    )
    assert plan["diagnostic_only"] is True
    assert plan["promotion_eligible"] is False
    assert plan["only_intended_drift"] == ["batch_size", "grad_accum_steps"]
    assert [run["local_batch_size"] for run in plan["runs"]] == [512, 1024]
    assert [run["grad_accum_steps"] for run in plan["runs"]] == [2, 1]
    assert {run["global_batch_size"] for run in plan["runs"]} == {8192}
    assert {run["warmup_samples"] for run in plan["runs"]} == {819_200}
    assert {run["planned_samples"] for run in plan["runs"]} == {4_194_304}
    assert plan["matched_invariants"]["warmup_samples"] == 819_200
    for run in plan["runs"]:
        command = run["command"]
        assert command[command.index("--max-steps") + 1] == "512"
        assert command[command.index("--lr-warmup-steps") + 1] == "100"
        assert command[command.index("--lr") + 1] == "3e-5"
        assert command[command.index("--seed") + 1] == "1"


def test_plan_refuses_incomplete_warmup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(quality.QualityProbeError, match="exceed"):
        quality.build_plan(
            base_command_json=_base(tmp_path, monkeypatch),
            output_dir=tmp_path / "probe",
            optimizer_steps=100,
        )


def test_plan_refuses_non_diagnostic_composite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = _base(tmp_path, monkeypatch)
    command = json.loads(base.read_text(encoding="utf-8"))
    descriptor = Path(command[command.index("--data") + 1])
    payload = json.loads(descriptor.read_text(encoding="utf-8"))
    payload["promotion_eligible"] = True
    descriptor.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(quality.QualityProbeError, match="diagnostic-only"):
        quality.build_plan(
            base_command_json=base,
            output_dir=tmp_path / "probe",
        )


def test_plan_digest_drift_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    quality.build_plan(
        base_command_json=_base(tmp_path, monkeypatch),
        output_dir=tmp_path / "probe",
    )
    path = tmp_path / "probe" / "plan.json"
    plan = json.loads(path.read_text(encoding="utf-8"))
    plan["matched_invariants"]["planned_samples"] += 1
    path.chmod(0o644)
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(quality.QualityProbeError, match="digest drift"):
        quality._read_plan(path)  # noqa: SLF001
