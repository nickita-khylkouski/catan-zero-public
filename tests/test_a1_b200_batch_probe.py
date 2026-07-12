from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import a1_b200_batch_probe as probe


def _receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    receipt = tmp_path / "training.receipt.json"
    receipt.write_text("{}", encoding="utf-8")
    command = [
        "/venv/python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=8",
        "/repo/tools/train_bc.py",
        "--data",
        "/data/n256.memmap",
        "--batch-size",
        "512",
        "--max-steps",
        "0",
        "--epochs",
        "1",
        "--lr",
        "0.00012",
        "--validation-game-seed-manifest",
        "/data/validation.json",
        "--a1-dual-learner-lock",
        "/data/lock.json",
        "--a1-dual-reviewed-lock-file-sha256",
        "sha256:" + "1" * 64,
        "--checkpoint",
        "/old/candidate.pt",
        "--report",
        "/old/report.json",
    ]
    monkeypatch.setattr(
        probe.dual,
        "verify_receipt",
        lambda _path: {
            "arm_id": "n256",
            "subset_id": "full-56k",
            "inputs": {
                "learner_ablation": {
                    "ablation_id": "all-196k-corrective-lr120u-loser1",
                    "diagnostic_only": True,
                    "effective_recipe": {
                        "lr": 0.00012,
                        "loser_sample_weight": 1.0,
                    },
                }
            },
            "command": command,
        },
    )
    return receipt


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        ("fixed", [0.00012, 0.00012, 0.00012]),
        ("sqrt", [0.00012, 0.00012 * (1.5**0.5), 0.00012 * (2**0.5)]),
        ("linear", [0.00012, 0.00018, 0.00024]),
    ],
)
def test_plan_separates_fixed_step_and_equal_sample_questions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    policy: str,
    expected: list[float],
) -> None:
    plan = probe.build_plan(
        midpoint_receipt=_receipt(tmp_path, monkeypatch),
        output_dir=tmp_path / "probe",
        lr_policy=policy,
    )
    assert plan["diagnostic_only"] is True
    assert plan["promotion_eligible"] is False
    throughput = [run for run in plan["runs"] if run["cohort"] == "throughput_fixed_steps"]
    learning = [run for run in plan["runs"] if run["cohort"] == "learning_equal_samples"]
    assert [run["max_steps"] for run in throughput] == [24, 24, 24]
    assert [run["max_steps"] for run in learning] == [48, 32, 24]
    assert len({run["planned_samples"] for run in learning}) == 1
    assert [run["lr"] for run in throughput] == pytest.approx(expected)
    for run in plan["runs"]:
        command = run["command"]
        assert not any(item.startswith("--a1-") for item in command)
        assert "--validation-game-seed-manifest" not in command
        assert command[command.index("--train-diagnostics-every-batches") + 1] == "1"
        assert command[command.index("--batch-size") + 1] == str(run["local_batch_size"])
        assert command[command.index("--max-steps") + 1] == str(run["max_steps"])
    assert plan["ranking_policy"]["diagnostic_only"] == ["hbm_memory_mib"]
    assert "never" in plan["ranking_policy"]["note"]


def test_plan_requires_exact_eight_b200_midpoint_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = _receipt(tmp_path, monkeypatch)
    original = probe.dual.verify_receipt

    def wrong(path: Path) -> dict:
        value = original(path)
        value["command"] = [
            "--nproc_per_node=2" if item == "--nproc_per_node=8" else item
            for item in value["command"]
        ]
        return value

    monkeypatch.setattr(probe.dual, "verify_receipt", wrong)
    with pytest.raises(probe.ProbeError, match="8-B200"):
        probe.build_plan(
            midpoint_receipt=receipt,
            output_dir=tmp_path / "probe",
            lr_policy="fixed",
        )


def test_summary_reports_efficiency_and_never_ranks_hbm(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "runtime.json").write_text(
        json.dumps({"started_unix_ns": 0, "finished_unix_ns": 2_000_000_000}),
        encoding="utf-8",
    )
    (run_dir / "train.report.json").write_text(
        json.dumps(
            {
                "steps_completed": 2,
                "metrics": [
                    {
                        "optimizer_observability": {
                            "clipped_fraction": 0.25,
                            "mean_pre_clip_total_grad_norm": 1.2,
                        },
                        "validation": {"active_policy_teacher_gap_closure": 0.4},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "gpu.csv").write_text(
        "timestamp,index,utilization.gpu [%],power.draw [W],memory.used [MiB]\n"
        "t,0,90,500,10000\n"
        "t,1,70,400,12000\n",
        encoding="utf-8",
    )
    (run_dir / "train.log").write_text(
        json.dumps(
            {
                "progress": "bc_optimizer_observability",
                "pre_clip_total_grad_norm": 1.2,
                "clipped": True,
                "module_pre_clip_grad_norms": {"trunk": 1.0},
                "module_parameter_delta_norms": {"trunk": 0.01},
                "module_norm_scope": "global_replicated",
            }
        )
        + "\n"
        + json.dumps(
            {
                "progress": "bc_optimizer_observability",
                "pre_clip_total_grad_norm": 0.8,
                "clipped": False,
                "module_pre_clip_grad_norms": {"trunk": 0.6},
                "module_parameter_delta_norms": {"trunk": 0.02},
                "module_norm_scope": "global_replicated",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = probe.summarize(
        {"run_id": "x", "run_dir": str(run_dir), "global_batch_size": 4096, "local_batch_size": 512, "lr": 0.00012}
    )
    assert result["samples_per_second"] == 4096
    assert result["active_teacher_gap_closure_per_wall_second"] == 0.2
    assert result["gpu"]["sm_util_mean_pct"] == 80
    assert result["gpu"]["hbm_memory_mean_mib"] == 11000
    assert result["gpu"]["hbm_is_ranking_objective"] is False
    assert result["optimizer_observability"]["preclip_grad_norm_mean"] == 1.0
    assert result["optimizer_observability"]["clipped_fraction"] == 0.5
    assert result["optimizer_observability"]["module_parameter_update_norm_mean"][
        "trunk"
    ] == pytest.approx(0.015)
