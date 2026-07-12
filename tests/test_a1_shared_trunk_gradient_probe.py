from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from tools import a1_shared_trunk_gradient_probe as probe


def _event(step: int) -> dict:
    return {
        "progress": "bc_optimizer_observability",
        "optimizer_step": step,
        "pre_clip_total_grad_norm": 2.0,
        "clipped": True,
        "module_parameter_delta_norms": {"blocks": 0.02, "value_head": 0.01},
        "objective_gradient_interference": {
            "available": True,
            "policy_trunk_grad_norm": 2.0,
            "value_trunk_grad_norm": 1.0,
            "value_to_policy_grad_norm_ratio": 0.5,
            "trunk_gradient_cosine": -0.25,
            "opposing_coordinate_fraction": 0.6,
            "modules": {
                "blocks.0": {
                    "policy_grad_norm": 1.5,
                    "value_grad_norm": 0.75,
                    "cosine": -0.5,
                }
            },
        },
    }


def test_single_gpu_command_removes_only_torchrun_prefix() -> None:
    source = [
        "/venv/python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=8",
        "/repo/tools/train_bc.py",
        "--data",
        "/data/composite.json",
        "--seed",
        "1",
    ]
    assert probe._single_gpu_command(source) == [  # noqa: SLF001
        "/venv/python",
        "/repo/tools/train_bc.py",
        "--data",
        "/data/composite.json",
        "--seed",
        "1",
    ]


def test_aggregate_retains_per_block_conflict_and_actual_update_norms() -> None:
    result = probe._aggregate([_event(1), _event(2)])  # noqa: SLF001
    assert result["trunk_gradient_cosine"]["mean"] == pytest.approx(-0.25)
    assert result["objective_gradient_modules"]["blocks.0"]["cosine"][
        "mean"
    ] == pytest.approx(-0.5)
    assert result["module_parameter_delta_norms"]["blocks"]["mean"] == pytest.approx(
        0.02
    )
    assert result["clipped_fraction"] == 1.0


def test_run_terminates_before_validation_and_writes_no_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "f7.pt"
    descriptor = tmp_path / "composite.json"
    trainer = tmp_path / "train_bc.py"
    parent.write_bytes(b"f7")
    descriptor.write_text("{}", encoding="utf-8")
    trainer.write_text("# bound trainer\n", encoding="utf-8")
    runtime = {
        "repository_root": str(tmp_path),
        "repository_commit": "a" * 40,
        "trainer": str(trainer),
        "trainer_sha256": probe._file_sha(trainer),  # noqa: SLF001
    }
    emitter = tmp_path / "emit.py"
    emitter.write_text(
        "import json,time\n"
        f"event={_event(1)!r}\n"
        "for i in range(100):\n"
        " event['optimizer_step']=i+1\n"
        " print(json.dumps(event),flush=True)\n"
        " time.sleep(.002)\n"
        "raise SystemExit('the runner failed to terminate me')\n",
        encoding="utf-8",
    )
    output = tmp_path / "probe"
    output.mkdir()
    plan = {
        "schema_version": probe.SCHEMA,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "steps": 32,
        "gpu": 0,
        "parent_checkpoint": str(parent),
        "parent_checkpoint_sha256": probe._file_sha(parent),  # noqa: SLF001
        "authenticated_composite": str(descriptor),
        "authenticated_composite_sha256": probe._file_sha(descriptor),  # noqa: SLF001
        "runtime": runtime,
        "command": [sys.executable, str(emitter)],
    }
    plan["plan_sha256"] = probe._digest(plan)  # noqa: SLF001
    plan_path = output / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    monkeypatch.setattr(probe, "_runtime_binding", lambda _path: runtime)
    monkeypatch.setattr(probe, "_require_gpu_idle", lambda _gpu: None)

    result = probe.run_plan(plan_path)

    assert result["steps_observed"] == 32
    assert result["termination"] == "bounded_before_validation_and_checkpoint"
    assert result["promotion_artifacts_emitted"] is False
    assert not any((output / name).exists() for name in probe.FORBIDDEN_OUTPUT_NAMES)
    assert (output / "gradient-probe.result.json").is_file()
