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
    assert plan["only_intended_drift"] == ["world_size", "batch_size", "gpu_ids"]
    assert [run["local_batch_size"] for run in plan["runs"]] == [512, 1024]
    assert [run["world_size"] for run in plan["runs"]] == [8, 4]
    assert [run["grad_accum_steps"] for run in plan["runs"]] == [1, 1]
    assert [run["gpu_ids"] for run in plan["runs"]] == [list(range(8)), list(range(4))]
    assert {run["global_batch_size"] for run in plan["runs"]} == {4096}
    assert {run["warmup_samples"] for run in plan["runs"]} == {409_600}
    assert {run["planned_samples"] for run in plan["runs"]} == {2_097_152}
    assert plan["matched_invariants"]["warmup_samples"] == 409_600
    assert plan["measurement_contract"] == {
        "train_diagnostics_every_batches": 0,
        "objective_gradient_interference_every_batches": 0,
        "timed_arms_run_sequentially": True,
        "reason": (
            "parameter snapshots, extra autograd probes, and concurrent host I/O "
            "would contaminate systems throughput"
        ),
    }
    for run in plan["runs"]:
        command = run["command"]
        assert command[command.index("--max-steps") + 1] == "512"
        assert command[command.index("--lr-warmup-steps") + 1] == "100"
        assert command[command.index("--lr") + 1] == "3e-5"
        assert command[command.index("--seed") + 1] == "1"
        assert command[command.index("--train-diagnostics-every-batches") + 1] == "0"
        assert (
            command[
                command.index("--objective-gradient-interference-every-batches")
                + 1
            ]
            == "0"
        )
    assert "--nproc-per-node=8" in plan["runs"][0]["command"]
    assert "--nproc-per-node=4" in plan["runs"][1]["command"]


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


@pytest.mark.parametrize(
    ("world_size", "gpu_ids"),
    [(8, list(range(8))), (4, list(range(4)))],
)
def test_selected_b200_names_accepts_both_geometry_arms(
    world_size: int, gpu_ids: list[int]
) -> None:
    names = ["NVIDIA B200"] * 8
    selected = quality._selected_b200_names(  # noqa: SLF001
        names, {"world_size": world_size, "gpu_ids": gpu_ids}
    )
    assert selected == ["NVIDIA B200"] * world_size


@pytest.mark.parametrize(
    ("run", "message"),
    [
        ({"world_size": 4, "gpu_ids": [0, 1, 2]}, "does not select 4"),
        ({"world_size": 4, "gpu_ids": [0, 1, 1, 2]}, "unique host GPUs"),
        ({"world_size": 4, "gpu_ids": [0, 1, 2, 8]}, "host_gpu_count=8"),
    ],
)
def test_selected_b200_names_rejects_invalid_binding(
    run: dict[str, object], message: str
) -> None:
    with pytest.raises(quality.QualityProbeError, match=message):
        quality._selected_b200_names(["NVIDIA B200"] * 8, run)  # noqa: SLF001


def test_selected_b200_names_rejects_non_b200_selected_gpu() -> None:
    names = ["NVIDIA B200"] * 8
    names[3] = "NVIDIA H100"
    with pytest.raises(quality.QualityProbeError, match="selected GPUs"):
        quality._selected_b200_names(  # noqa: SLF001
            names, {"world_size": 4, "gpu_ids": [0, 1, 2, 3]}
        )
