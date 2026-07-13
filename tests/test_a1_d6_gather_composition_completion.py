from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest
import torch

from tools import a1_d6_gather_composition_arm as arm
from tools import a1_d6_gather_composition_completion as completion
from test_a1_d6_gather_composition_arm import _composition_args


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _state_reader(*_args, **_kwargs) -> str:
    return (
        "LoadState=loaded\nActiveState=active\nSubState=exited\n"
        "Result=success\nExecMainStatus=0\nExecMainCode=1\n"
    )


def _submit(manifest_path: Path, *, executor: Path) -> None:
    verified = arm.verify(manifest_path, expected_executor=executor)
    arm.executor_base._submit_verified(  # noqa: SLF001
        verified,
        unit="a1-d6-gather-completion-test",
        runner=lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 0, stdout="Running as unit.", stderr=""
        ),
        conflict_probe=lambda: [],
        claim_schema=arm.CLAIM_SCHEMA,
        receipt_schema=arm.RECEIPT_SCHEMA,
        status_schema=arm.STATUS_SCHEMA,
    )


def _complete_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, object]:
    args = _composition_args(tmp_path, monkeypatch)
    manifest, manifest_path = arm.prepare(args)
    fake_finalizer = Path(
        manifest["source_binding"]["files"][arm.COMPLETION_RELATIVE_PATH]["path"]
    )
    monkeypatch.setattr(completion, "__file__", str(fake_finalizer))
    _submit(manifest_path, executor=args.bound_executor)
    root = args.output_root

    initializer = Path(manifest["initialization_treatment"]["path"])
    candidate = torch.load(initializer, map_location="cpu", weights_only=False)
    for name in completion.EXPECTED_CHANGED_PARAMETERS:
        candidate["model"][name] = candidate["model"][name] + 1.0
    checkpoint = root / "candidate.pt"
    torch.save(candidate, checkpoint)

    optimizer = root / "candidate.pt.optimizer.pt"
    action_ids = list(range(len(completion.EXPECTED_CHANGED_PARAMETERS)))
    torch.save(
        {
            "format": "plain",
            "optimizer": {
                "state": {index: {"step": torch.tensor(1024)} for index in action_ids},
                "param_groups": [
                    {"lr": 3e-5, "base_lr": 3e-5, "params": []},
                    {"lr": 1.2e-4, "base_lr": 1.2e-4, "params": action_ids},
                ],
            },
        },
        optimizer,
    )

    component_ids = list(arm.gather.production_temp.COMPONENT_IDS)
    components = {
        component_id: {
            "metrics": {"active_policy_teacher_gap_closure": 0.01 + 0.01 * index}
        }
        for index, component_id in enumerate(component_ids)
    }
    trainer = Path(manifest["source_binding"]["files"]["tools/train_bc.py"]["path"])
    report = {
        "init_checkpoint": manifest["initialization_treatment"]["path"],
        "init_checkpoint_sha256": manifest["initialization_treatment"]["sha256"],
        "checkpoint": str(checkpoint.resolve()),
        "data": manifest["descriptor"]["path"],
        "input_validation_game_sentinel_manifest": manifest["validation_sentinel"][
            "path"
        ],
        "world_size": 8,
        "batch_size": 64,
        "effective_global_batch_size": 512,
        "max_steps": 1024,
        "steps_completed": 1024,
        "training_row_draws": 524_288,
        "base_training_row_draws": 524_288,
        "total_training_row_draws": 524_288,
        "optimizer": "adam",
        "resume_optimizer": False,
        "optimizer_restored": False,
        "fused_optimizer": False,
        "amp": "none",
        "float32_matmul_precision": "highest",
        "effective_float32_matmul_precision": "highest",
        "lr": 3e-5,
        "lr_warmup_steps": 100,
        "lr_schedule": "flat",
        "weight_decay": 0.0,
        "value_lr_mult": 1.0,
        "action_module_lr_mult": 4.0,
        "policy_loss_weight": 1.0,
        "soft_target_source": "policy",
        "soft_target_weight": 0.9,
        "soft_target_temperature": 0.7,
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
        "symmetry_augment": True,
        "symmetry_augment_events": True,
        "ddp_find_unused_parameters": False,
        "ddp_shard_data": False,
        "training_rng_rank_offset": True,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "a1_decisive_training_semantics": {
            "schema_version": "a1-decisive-training-semantics-v2",
            "decisive": False,
            "diagnostic_authority_present": True,
            "world_size": 8,
            "grad_accum_steps": 1,
            "gradient_accumulation_contract": "single_microbatch_exact",
            "symmetry_augmentation": True,
            "distributed_symmetry_contract": (
                "per_rank_seedsequence_checkpoint_resume_v1"
            ),
            "advantage_policy_weighting": "none",
            "distributed_advantage_contract": "not_applicable",
        },
        "stored_policy_component_temperatures": (
            arm.gather.production_temp.COMPONENT_TEMPERATURES
        ),
        "memmap_composite": {
            "component_ids": component_ids,
            "policy_distillation_component_ids": component_ids,
            "value_training_component_ids": component_ids,
        },
        "checkout_runtime_binding": {
            "trainer": str(trainer.resolve()),
            "trainer_sha256": arm.gather.corrected._file_ref(trainer)[  # noqa: SLF001
                "sha256"
            ],
        },
        "training_information_surface": {
            "required_trainable_surface": {
                "prefixes": [arm.TRAINABLE_PREFIX],
                "parameter_tensors": 4,
                "parameters": 1234,
                "parameters_by_prefix": {arm.TRAINABLE_PREFIX: 1234},
            }
        },
        "metrics": [
            {
                "samples": 524_288,
                "policy_total_active_rows": 64_309,
                "optimizer_observability": {
                    "observed_steps": 1024,
                    "zero_objective_steps_skipped": 0,
                },
                "validation_objective_matched": {
                    "schema_version": "composite-validation-measure-v2",
                    "objective_matched": True,
                    "metrics": {"active_policy_teacher_gap_closure": 0.0543},
                    "components": components,
                },
            }
        ],
        "elapsed_sec": 73.5,
    }
    _write_json(root / "train.report.json", report)
    _write_json(root / "train.report.validation_seeds.json", {"seeds": [1]})

    progress = {
        "schema_version": "train-bc-progress-v1",
        "status": "complete",
        "checkpoint": completion._compact_ref(checkpoint),  # noqa: SLF001
        "optimizer": completion._compact_ref(optimizer),  # noqa: SLF001
        "optimizer_step": 1024,
        "completed_epochs": 1,
        "recipe_identity": {
            "schema_version": "train-bc-resume-recipe-v1",
            "world_size": 8,
            "grad_accum_steps": 1,
            "ddp_shard_data": False,
        },
        "rank_torch_rng_states": [{"rank": rank} for rank in range(8)],
        "rank_numpy_rng_states": [{"state": rank} for rank in range(8)],
        "symmetry_rng_state": {
            "schema_version": "train-bc-rank-symmetry-rng-v1",
            "world_size": 8,
            "rank_states": [{"state": rank} for rank in range(8)],
        },
    }
    progress["checkpoint"]["path"] = checkpoint.name
    progress["optimizer"]["path"] = optimizer.name
    progress["progress_sha256"] = arm.gather.corrected._digest(progress)
    _write_json(root / "candidate.pt.training-progress.json", progress)
    (root / "stdout.log").write_text("completed\n", encoding="utf-8")
    (root / "stderr.log").write_text("", encoding="utf-8")
    return manifest_path, checkpoint, args


def test_completion_proves_exact_geometry_rng_optimizer_and_model_delta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, checkpoint, _args = _complete_run(tmp_path, monkeypatch)
    checkpoint_sha = completion._file_ref(checkpoint)["sha256"]  # noqa: SLF001
    receipt = completion.finalize(
        manifest,
        expected_checkpoint_sha256=checkpoint_sha,
        state_reader=_state_reader,
    )
    assert receipt["status"] == "complete_nonpromotable"
    assert (
        receipt["optimizer_geometry"]["treatment_adapter_commissioning"][
            "optimizer_steps"
        ]
        == 1024
    )
    assert receipt["optimizer_groups"] == {
        "format": "plain",
        "base_group_parameter_tensors": 0,
        "base_group_lr": 3e-5,
        "action_group_parameter_tensors": 4,
        "action_group_lr": 1.2e-4,
        "optimizer_state_tensors": 4,
    }
    assert receipt["rng_summary"] == {
        "rank_torch_rng_states": 8,
        "rank_torch_rng_set": list(range(8)),
        "rank_numpy_rng_states": 8,
        "rank_numpy_state_digests_unique": True,
        "symmetry_rng_schema": "train-bc-rank-symmetry-rng-v1",
        "rank_symmetry_rng_states": 8,
        "rank_symmetry_state_digests_unique": True,
        "world_size": 8,
    }
    assert receipt["model_delta"]["inherited_parameters_bit_identical"] is True
    assert receipt["model_delta"]["changed_parameter_tensors"] == list(
        completion.EXPECTED_CHANGED_PARAMETERS
    )
    assert receipt["report_summary"]["policy_active_rows"] == 64_309
    assert (
        completion.verify_completion(manifest.parent / completion.COMPLETION_NAME)
        == receipt
    )


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda root: _mutate_report(root, "steps_completed", 1023),
            "report recipe/dose drift",
        ),
        (
            lambda root: _mutate_progress_symmetry(root),
            "progress/RNG/optimizer dose drift",
        ),
        (
            lambda root: _mutate_optimizer_lr(root),
            "does not isolate four LR=1.2e-4 tensors",
        ),
        (
            lambda root: _mutate_inherited_tensor(root),
            "outside/excluding gather adapter",
        ),
    ],
)
def test_completion_refuses_partial_or_confounded_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate,
    match: str,
) -> None:
    manifest, checkpoint, _args = _complete_run(tmp_path, monkeypatch)
    mutate(manifest.parent)
    with pytest.raises(completion.CompletionError, match=match):
        completion.finalize(
            manifest,
            expected_checkpoint_sha256=completion._file_ref(checkpoint)[  # noqa: SLF001
                "sha256"
            ],
            state_reader=_state_reader,
        )


def _mutate_report(root: Path, key: str, value: object) -> None:
    path = root / "train.report.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[key] = value
    _write_json(path, payload)


def _mutate_progress_symmetry(root: Path) -> None:
    path = root / "candidate.pt.training-progress.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["symmetry_rng_state"]["rank_states"].pop()
    payload["progress_sha256"] = arm.gather.corrected._digest(
        {key: value for key, value in payload.items() if key != "progress_sha256"}
    )
    _write_json(path, payload)


def _mutate_optimizer_lr(root: Path) -> None:
    path = root / "candidate.pt.optimizer.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["optimizer"]["param_groups"][1]["lr"] = 6e-5
    torch.save(payload, path)
    progress_path = root / "candidate.pt.training-progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["optimizer"]["sha256"] = completion._compact_ref(path)[  # noqa: SLF001
        "sha256"
    ]
    progress["progress_sha256"] = arm.gather.corrected._digest(
        {key: value for key, value in progress.items() if key != "progress_sha256"}
    )
    _write_json(progress_path, progress)


def _mutate_inherited_tensor(root: Path) -> None:
    path = root / "candidate.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["model"]["encoder.weight"] = payload["model"]["encoder.weight"] + 1.0
    torch.save(payload, path)
    progress_path = root / "candidate.pt.training-progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["checkpoint"]["sha256"] = completion._compact_ref(path)[  # noqa: SLF001
        "sha256"
    ]
    progress["progress_sha256"] = arm.gather.corrected._digest(
        {key: value for key, value in progress.items() if key != "progress_sha256"}
    )
    _write_json(progress_path, progress)
