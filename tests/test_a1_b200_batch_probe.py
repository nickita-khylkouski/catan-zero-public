from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools import a1_b200_batch_probe as probe
from tools import train_bc


def _receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    receipt = tmp_path / "training.receipt.json"
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
        "--loser-sample-weight",
        "1.0",
        "--validation-fraction",
        "0.05",
        "--validation-seed",
        "17",
        "--validation-max-samples",
        "0",
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
    trainer_index = command.index("/repo/tools/train_bc.py")
    args = train_bc.build_parser().parse_args(command[trainer_index + 1 :])
    recipe = train_bc._effective_a1_learner_training_recipe(  # noqa: SLF001
        args, {"world_size": 8, "rank": 0, "enabled": True}
    )
    recipe["per_game_value_weight_mode"] = "equal"
    payload = {
        "status": "complete",
        "arm_id": "n256",
        "subset_id": "full-56k",
        "inputs": {
            "learner_ablation": {
                "ablation_id": "all-196k-corrective-lr120u-loser1",
                "diagnostic_only": True,
                "effective_recipe": recipe,
            }
        },
        "command": command,
    }
    payload["receipt_sha256"] = probe._digest(payload)  # noqa: SLF001
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        probe.dual,
        "verify_receipt",
        lambda _path: payload,
    )
    trainer = (Path(probe.__file__).resolve().parents[1] / "tools" / "train_bc.py")
    monkeypatch.setattr(
        probe,
        "_current_runtime",
        lambda: {
            "repository_root": str(trainer.parents[1]),
            "repository_commit": "a" * 40,
            "trainer": str(trainer),
            "trainer_sha256": probe._file_sha(trainer),  # noqa: SLF001
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
    assert plan["runtime"]["repository_commit"] == "a" * 40
    throughput = [run for run in plan["runs"] if run["cohort"] == "throughput_fixed_steps"]
    learning = [run for run in plan["runs"] if run["cohort"] == "learning_equal_samples"]
    assert [run["max_steps"] for run in throughput] == [24, 24, 24]
    assert [run["max_steps"] for run in learning] == [48, 32, 24]
    assert len({run["planned_samples"] for run in learning}) == 1
    assert [run["lr"] for run in throughput] == pytest.approx(expected)
    for run in plan["runs"]:
        command = run["command"]
        assert plan["runtime"]["trainer"] in command
        assert "/repo/tools/train_bc.py" not in command
        assert {
            item for item in command if item.startswith("--a1-")
        } == {"--a1-batch-probe-plan", "--a1-batch-probe-run-id"}
        assert command[command.index("--a1-batch-probe-run-id") + 1] == run["run_id"]
        assert command[command.index("--a1-batch-probe-plan") + 1] == str(
            tmp_path / "probe" / "plan.json"
        )
        assert command.count("--validation-game-seed-manifest") == 1
        expected_validation = {
            "--validation-fraction": "0.05",
            "--validation-seed": "17",
            "--validation-max-samples": "0",
            "--validation-game-seed-manifest": "/data/validation.json",
        }
        for flag, value in expected_validation.items():
            assert command.count(flag) == 1
            assert command[command.index(flag) + 1] == value
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


def test_runtime_drift_is_rejected_before_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = probe.build_plan(
        midpoint_receipt=_receipt(tmp_path, monkeypatch),
        output_dir=tmp_path / "probe",
        lr_policy="fixed",
    )
    run = plan["runs"][0]
    changed = dict(plan["runtime"])
    changed["repository_commit"] = "b" * 40
    monkeypatch.setattr(probe, "_current_runtime", lambda: changed)
    with pytest.raises(probe.ProbeError, match="runtime drift"):
        probe._verify_runtime(plan, run)  # noqa: SLF001


def test_mps_handoff_restores_service_after_failure() -> None:
    calls: list[tuple[str, ...]] = []
    active = True

    def runner(command: list[str], **_kwargs: object):
        nonlocal active
        calls.append(tuple(command))
        if command[:2] == ["systemctl", "is-active"]:
            return probe.subprocess.CompletedProcess(
                command, 0 if active else 3, "active\n" if active else "inactive\n", ""
            )
        if command[-2:] == ["stop", "nvidia-mps.service"]:
            active = False
        elif command[-2:] == ["start", "nvidia-mps.service"]:
            active = True
        return probe.subprocess.CompletedProcess(command, 0, "", "")

    with pytest.raises(RuntimeError, match="boom"):
        with probe._without_mps(runner=runner):  # noqa: SLF001
            assert active is False
            raise RuntimeError("boom")
    assert active is True
    assert ("sudo", "-n", "systemctl", "stop", "nvidia-mps.service") in calls
    assert ("sudo", "-n", "systemctl", "start", "nvidia-mps.service") in calls


def test_train_bc_sibling_contract_import_works_without_repo_on_pythonpath() -> None:
    repo = Path(probe.__file__).resolve().parents[1]
    tools_dir = repo / "tools"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from a1_pre_wave_contract import EXPECTED_LEARNER_TRAINING_RECIPE; "
            "assert EXPECTED_LEARNER_TRAINING_RECIPE",
        ],
        cwd=tools_dir,
        env={"PATH": str(Path(sys.executable).parent)},
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode == 0, result.stderr


def test_train_bc_authenticates_exact_probe_command_before_memmap_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = probe.build_plan(
        midpoint_receipt=_receipt(tmp_path, monkeypatch),
        output_dir=tmp_path / "probe",
        lr_policy="fixed",
    )
    run = plan["runs"][1]
    trainer_index = next(
        index
        for index, value in enumerate(run["command"])
        if Path(value).name == "train_bc.py"
    )
    argv = run["command"][trainer_index:]
    monkeypatch.setattr(sys, "argv", argv)
    args = train_bc.build_parser().parse_args(argv[1:])
    effective = train_bc._effective_a1_learner_training_recipe(  # noqa: SLF001
        args, {"world_size": 8, "rank": 0, "enabled": True}
    )

    authorization = train_bc._validate_a1_batch_probe_authorization(  # noqa: SLF001
        args, effective
    )

    assert authorization is not None
    assert authorization["run_id"] == run["run_id"]
    assert set(authorization["recipe_drift"]) == {
        "batch_size",
        "global_batch_size",
        "max_steps",
    }


def test_train_bc_probe_authorization_rejects_unplanned_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = probe.build_plan(
        midpoint_receipt=_receipt(tmp_path, monkeypatch),
        output_dir=tmp_path / "probe",
        lr_policy="fixed",
    )
    run = plan["runs"][0]
    trainer_index = next(
        index
        for index, value in enumerate(run["command"])
        if Path(value).name == "train_bc.py"
    )
    argv = run["command"][trainer_index:]
    args = train_bc.build_parser().parse_args(argv[1:])
    effective = train_bc._effective_a1_learner_training_recipe(  # noqa: SLF001
        args, {"world_size": 8, "rank": 0, "enabled": True}
    )
    tampered = list(argv)
    tampered[tampered.index("--batch-size") + 1] = "513"
    monkeypatch.setattr(sys, "argv", tampered)

    with pytest.raises(SystemExit, match="does not bind the executing argv"):
        train_bc._validate_a1_batch_probe_authorization(args, effective)  # noqa: SLF001


def test_train_bc_probe_authorization_rejects_lr_scaling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = probe.build_plan(
        midpoint_receipt=_receipt(tmp_path, monkeypatch),
        output_dir=tmp_path / "probe",
        lr_policy="sqrt",
    )
    run = plan["runs"][1]
    trainer_index = next(
        index
        for index, value in enumerate(run["command"])
        if Path(value).name == "train_bc.py"
    )
    argv = run["command"][trainer_index:]
    monkeypatch.setattr(sys, "argv", argv)
    args = train_bc.build_parser().parse_args(argv[1:])
    effective = train_bc._effective_a1_learner_training_recipe(  # noqa: SLF001
        args, {"world_size": 8, "rank": 0, "enabled": True}
    )

    with pytest.raises(SystemExit, match="exceeds its allowed drift"):
        train_bc._validate_a1_batch_probe_authorization(args, effective)  # noqa: SLF001


def test_gpu_samples_parses_nvidia_smi_spaced_csv(tmp_path: Path) -> None:
    telemetry = tmp_path / "gpu.csv"
    telemetry.write_text(
        "timestamp, index, utilization.gpu [%], power.draw [W], memory.used [MiB]\n"
        "2026/07/12 00:00:00.000, 0, 80, 600.0, 22000\n"
        "2026/07/12 00:00:00.000, 1, 100, 700.0, 24000\n",
        encoding="utf-8",
    )

    result = probe._gpu_samples(telemetry)  # noqa: SLF001

    assert result["sm_util_mean_pct"] == 90.0
    assert result["power_mean_w"] == 650.0
    assert result["hbm_memory_mean_mib"] == 23000.0


def test_plan_supports_conditional_large_batch_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = probe.build_plan(
        midpoint_receipt=_receipt(tmp_path, monkeypatch),
        output_dir=tmp_path / "probe",
        lr_policy="fixed",
        batches=(1536, 2048),
    )

    throughput = [run for run in plan["runs"] if run["cohort"] == "throughput_fixed_steps"]
    learning = [run for run in plan["runs"] if run["cohort"] == "learning_equal_samples"]
    assert [run["local_batch_size"] for run in throughput] == [1536, 2048]
    assert [run["max_steps"] for run in learning] == [16, 12]
    assert len({run["planned_samples"] for run in learning}) == 1


def test_gpu_occupancy_allows_only_mps_server() -> None:
    def runner(command: list[str], **_kwargs: object):
        return probe.subprocess.CompletedProcess(
            command, 0, "10, nvidia-cuda-mps-server\n", ""
        )

    probe._require_no_non_mps_compute(runner=runner)  # noqa: SLF001

    def occupied(command: list[str], **_kwargs: object):
        return probe.subprocess.CompletedProcess(command, 0, "11, python\n", "")

    with pytest.raises(probe.ProbeError, match="active non-MPS compute"):
        probe._require_no_non_mps_compute(runner=occupied)  # noqa: SLF001


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
