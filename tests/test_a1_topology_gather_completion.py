from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest
import torch

from tools import a1_topology_gather_arm as arm
from tools import a1_topology_gather_arm_execute as executor
from tools import a1_topology_gather_completion as completion
from test_a1_topology_gather_arm_execute import _manifest


def _write_json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _complete_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, dict]:
    manifest_path, manifest = _manifest(tmp_path, monkeypatch)
    executor.execute(
        manifest_path,
        unit="a1-topology-gather-completion-test",
        runner=lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 0, stdout="Running as unit.", stderr=""
        ),
        conflict_probe=lambda: [],
    )
    root = manifest_path.parent
    initializer = Path(manifest["initialization_treatment"]["path"])
    candidate = torch.load(initializer, map_location="cpu", weights_only=False)
    for name in completion.EXPECTED_CHANGED_PARAMETERS:
        candidate["model"][name] = candidate["model"][name] + 1.0
    checkpoint_path = root / "candidate.pt"
    torch.save(candidate, checkpoint_path)
    optimizer_path = root / "candidate.pt.optimizer.pt"
    torch.save({"state": {}, "step": arm.OPTIMIZER_STEPS}, optimizer_path)

    component_ids = list(arm.production_temp.COMPONENT_IDS)
    component_metrics = {
        component_id: {
            "metrics": {"active_policy_teacher_gap_closure": 0.01 + index * 0.01}
        }
        for index, component_id in enumerate(component_ids)
    }
    trainer = Path(manifest["selected_geometry_evidence"]["runtime"]["trainer"])
    report = {
        "init_checkpoint": manifest["initialization_treatment"]["path"],
        "init_checkpoint_sha256": manifest["initialization_treatment"]["sha256"],
        "checkpoint": str(checkpoint_path.resolve()),
        "data": manifest["descriptor"]["path"],
        "input_validation_game_sentinel_manifest": manifest["validation_sentinel"][
            "path"
        ],
        "world_size": arm.WORLD_SIZE,
        "batch_size": arm.LOCAL_BATCH_SIZE,
        "effective_global_batch_size": arm.GLOBAL_BATCH_SIZE,
        "max_steps": arm.OPTIMIZER_STEPS,
        "steps_completed": arm.OPTIMIZER_STEPS,
        "training_row_draws": arm.SELECTED_GLOBAL_ROW_DOSE,
        "base_training_row_draws": arm.SELECTED_GLOBAL_ROW_DOSE,
        "total_training_row_draws": arm.SELECTED_GLOBAL_ROW_DOSE,
        "optimizer": "adam",
        "resume_optimizer": False,
        "optimizer_restored": False,
        "fused_optimizer": False,
        "amp": "none",
        "lr": 3e-5,
        "lr_warmup_steps": 100,
        "lr_schedule": "flat",
        "weight_decay": 0.0,
        "value_lr_mult": 1.0,
        "action_module_lr_mult": arm.ACTION_MODULE_LR_MULT,
        "policy_loss_weight": 1.0,
        "soft_target_source": "policy",
        "soft_target_weight": 0.9,
        "value_loss_weight": 0.25,
        "value_target_lambda": 1.0,
        "q_loss_weight": 0.0,
        "policy_kl_anchor_weight": 0.0,
        "forced_action_weight": 0.0,
        "forced_row_value_weight": 1.0,
        "winner_sample_weight": 1.0,
        "loser_sample_weight": 1.0,
        "freeze_modules": arm.FREEZE_MODULES,
        "require_only_trainable_prefixes": arm.TRAINABLE_PREFIX,
        "action_target_gather": True,
        "ddp_find_unused_parameters": False,
        "ddp_shard_data": False,
        "training_rng_rank_offset": True,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "stored_policy_component_temperatures": (
            arm.production_temp.COMPONENT_TEMPERATURES
        ),
        "memmap_composite": {
            "component_ids": component_ids,
            "policy_distillation_component_ids": component_ids,
            "value_training_component_ids": component_ids,
        },
        "checkout_runtime_binding": {
            "trainer": str(trainer.resolve()),
            "trainer_sha256": arm.corrected._file_ref(trainer)["sha256"],
        },
        "training_information_surface": {
            "required_trainable_surface": {
                "prefixes": [arm.TRAINABLE_PREFIX],
                "parameter_tensors": len(completion.EXPECTED_CHANGED_PARAMETERS),
                "parameters": 1234,
                "parameters_by_prefix": {arm.TRAINABLE_PREFIX: 1234},
            }
        },
        "metrics": [
            {
                "samples": arm.SELECTED_GLOBAL_ROW_DOSE,
                "policy_total_active_rows": 64_309,
                "optimizer_observability": {
                    "observed_steps": arm.OPTIMIZER_STEPS,
                    "zero_objective_steps_skipped": 0,
                },
                "validation_objective_matched": {
                    "schema_version": "composite-validation-measure-v2",
                    "objective_matched": True,
                    "metrics": {"active_policy_teacher_gap_closure": 0.0543},
                    "components": component_metrics,
                },
            }
        ],
        "elapsed_sec": 73.5,
    }
    _write_json(root / "train.report.json", report)
    _write_json(root / "train.report.validation_seeds.json", {"seeds": [1]})
    checkpoint_ref = completion._compact_ref(checkpoint_path)  # noqa: SLF001
    optimizer_ref = completion._compact_ref(optimizer_path)  # noqa: SLF001
    progress = {
        "checkpoint": {"path": checkpoint_path.name, "sha256": checkpoint_ref["sha256"]},
        "optimizer": {"path": optimizer_path.name, "sha256": optimizer_ref["sha256"]},
        "optimizer_step": arm.OPTIMIZER_STEPS,
        "completed_epochs": 1,
        "rank_torch_rng_states": [f"rank-{rank}" for rank in range(arm.WORLD_SIZE)],
    }
    progress["progress_sha256"] = arm.corrected._digest(progress)
    _write_json(root / "candidate.pt.training-progress.json", progress)
    (root / "stdout.log").write_text("completed\n", encoding="utf-8")
    (root / "stderr.log").write_text("", encoding="utf-8")
    return manifest_path, checkpoint_path, manifest


def _state_reader(*_args, **_kwargs) -> str:
    return "ActiveState=inactive\nResult=success\nExecMainStatus=0\n"


def test_completion_replays_exact_dose_and_adapter_only_delta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, checkpoint, _manifest_value = _complete_run(tmp_path, monkeypatch)
    payload = completion.finalize(
        manifest_path,
        expected_checkpoint_sha256=completion._file_ref(checkpoint)["sha256"],  # noqa: SLF001
        state_reader=_state_reader,
    )
    receipt = manifest_path.parent / completion.COMPLETION_NAME
    assert completion.verify_completion(receipt) == payload
    assert payload["report_summary"]["policy_active_rows"] == 64_309
    assert payload["model_delta"]["changed_parameter_tensors"] == list(
        completion.EXPECTED_CHANGED_PARAMETERS
    )
    assert payload["model_delta"]["inherited_parameters_bit_identical"] is True


def test_completion_refuses_wrong_expected_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, _checkpoint, _manifest_value = _complete_run(tmp_path, monkeypatch)
    with pytest.raises(completion.CompletionError, match="explicitly expected"):
        completion.build_completion(
            manifest_path,
            expected_checkpoint_sha256="sha256:" + "0" * 64,
            unit_state={
                "ActiveState": "inactive",
                "Result": "success",
                "ExecMainStatus": "0",
            },
            created_at_unix_ns=1,
        )


def test_completion_refuses_mature_parameter_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, checkpoint, _manifest_value = _complete_run(tmp_path, monkeypatch)
    candidate = torch.load(checkpoint, map_location="cpu", weights_only=False)
    candidate["model"]["policy.weight"] += 1.0
    torch.save(candidate, checkpoint)
    progress_path = manifest_path.parent / "candidate.pt.training-progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["checkpoint"]["sha256"] = completion._file_ref(checkpoint)[  # noqa: SLF001
        "sha256"
    ]
    progress.pop("progress_sha256")
    progress["progress_sha256"] = arm.corrected._digest(progress)
    _write_json(progress_path, progress)
    with pytest.raises(completion.CompletionError, match="outside/excluding gather"):
        completion.build_completion(
            manifest_path,
            expected_checkpoint_sha256=completion._file_ref(checkpoint)["sha256"],  # noqa: SLF001
            unit_state={
                "ActiveState": "inactive",
                "Result": "success",
                "ExecMainStatus": "0",
            },
            created_at_unix_ns=1,
        )


def test_completion_refuses_optimizer_step_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, checkpoint, _manifest_value = _complete_run(tmp_path, monkeypatch)
    progress_path = manifest_path.parent / "candidate.pt.training-progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["optimizer_step"] -= 1
    progress.pop("progress_sha256")
    progress["progress_sha256"] = arm.corrected._digest(progress)
    _write_json(progress_path, progress)
    with pytest.raises(completion.CompletionError, match="dose drift"):
        completion.build_completion(
            manifest_path,
            expected_checkpoint_sha256=completion._file_ref(checkpoint)["sha256"],  # noqa: SLF001
            unit_state={
                "ActiveState": "inactive",
                "Result": "success",
                "ExecMainStatus": "0",
            },
            created_at_unix_ns=1,
        )
