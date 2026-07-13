from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tools import a1_d6_gather_composition_arm as arm
from test_a1_topology_gather_arm import _args as topology_args


def test_real_d6_composition_artifact_identities_are_pinned() -> None:
    assert arm.D6_PARENT_SHA256 == (
        "sha256:761135ead3e9ec2d3b2816e2bc0b4fcd1fda1b2f897115e46295ed9198a1d28b"
    )
    assert arm.D6_REPORT_SHA256 == (
        "sha256:dc360a97c1d6659684483deeb47295b9d48f4042799d64ae3cded3ad4818383b"
    )
    assert arm.D6_PROGRESS_SHA256 == (
        "sha256:f56ce788dbc31d51cd250a55843fde36a20fcb9021c07f63355a5cf7ee881f62"
    )
    assert arm.D6_GATHER_INIT_SHA256 == (
        "sha256:015be3463b424d5694fd459c819d677fb1f7a2b1aaf590101bdc403e2411858d"
    )
    assert arm.D6_SHORT_PARENT_SHA256 == (
        "sha256:9dd1d261a39d7b04713505a301097faf18e84e8a3508b4abb92a8b964f7ab921"
    )
    assert arm.D6_SHORT_REPORT_SHA256 == (
        "sha256:42b8f620b2d22edffd4e0d223052f0e5873c48de4b3cf8f037c53af0b08cdae5"
    )
    assert arm.D6_SHORT_PROGRESS_SHA256 == (
        "sha256:9e2019557268281144bc7b06cece2831fe3e3abe5fdf9aea3ab6d0ee32b72492"
    )
    assert arm.D6_SHORT_GATHER_INIT_SHA256 == (
        "sha256:14f0a8634d61afccea8eade03f4bb40304ed5e68729d1fda85bb28d2ab1708ef"
    )
    assert arm.D6_PARENT_SHA256 != arm.D6_F7_PARENT_SHA256


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _d6_artifacts(
    tmp_path: Path,
    *,
    source_checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path]:
    checkpoint = tmp_path / "d6-parent" / "candidate.pt"
    checkpoint.parent.mkdir()
    d6 = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    # Exercise a genuine mature-parent composition, rather than aliasing f7.
    d6["model"]["encoder.weight"] = d6["model"]["encoder.weight"] + 0.25
    torch.save(d6, checkpoint)
    checkpoint_sha = arm.gather.corrected._file_sha(checkpoint)
    f7_sha = arm.gather.corrected._file_sha(source_checkpoint)
    monkeypatch.setattr(arm, "D6_PARENT_SHA256", checkpoint_sha)
    monkeypatch.setattr(arm, "D6_F7_PARENT_SHA256", f7_sha)
    report = _write_json(
        checkpoint.parent / "train.report.json",
        {
            "init_checkpoint_sha256": f7_sha,
            "world_size": 8,
            "batch_size": 512,
            "effective_global_batch_size": 4096,
            "max_steps": 1024,
            "steps_completed": 1024,
            "training_row_draws": 4_194_304,
            "optimizer": "adam",
            "resume_optimizer": False,
            "optimizer_restored": False,
            "lr": 3e-5,
            "lr_schedule": "flat",
            "lr_warmup_steps": 100,
            "action_module_lr_mult": 1.0,
            "soft_target_weight": 0.9,
            "soft_target_temperature": 0.7,
            "value_loss_weight": 0.25,
            "forced_action_weight": 0.0,
            "forced_row_value_weight": 1.0,
            "mask_hidden_info": True,
            "graph_history_features": True,
            "symmetry_augment": True,
            "diagnostic_only": True,
            "promotion_eligible": False,
        },
    )
    monkeypatch.setattr(arm, "D6_REPORT_SHA256", arm.gather.corrected._file_sha(report))
    progress_payload = {
        "schema_version": "train-bc-progress-v1",
        "status": "complete",
        "optimizer_step": 1024,
        "checkpoint": {"path": "candidate.pt", "sha256": checkpoint_sha},
        "optimizer": {
            "path": "candidate.pt.optimizer.pt",
            "sha256": "sha256:" + "a" * 64,
        },
        "completed_epochs": 1,
        "recipe_identity": {
            "schema_version": "train-bc-resume-recipe-v1",
            "world_size": 8,
            "grad_accum_steps": 1,
            "ddp_shard_data": False,
            "fsdp": False,
        },
        "rank_torch_rng_states": [{"rank": rank} for rank in range(8)],
        "symmetry_rng_state": {"bit_generator": "PCG64", "state": {"state": 1}},
    }
    progress_payload["progress_sha256"] = arm.gather.corrected._digest(progress_payload)
    progress = _write_json(
        checkpoint.parent / "candidate.pt.training-progress.json", progress_payload
    )
    monkeypatch.setattr(
        arm, "D6_PROGRESS_SHA256", arm.gather.corrected._file_sha(progress)
    )
    return checkpoint, report, progress


def _composition_args(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source = topology_args(tmp_path, monkeypatch)
    source_manifest = json.loads(source.source_manifest.read_text(encoding="utf-8"))
    f7 = Path(source_manifest["f7_parent"]["path"])
    d6_checkpoint, d6_report, d6_progress = _d6_artifacts(
        tmp_path, source_checkpoint=f7, monkeypatch=monkeypatch
    )
    f7_gather_checkpoint = source.gather_checkpoint
    gather_raw = torch.load(
        f7_gather_checkpoint, map_location="cpu", weights_only=False
    )
    d6_raw = torch.load(d6_checkpoint, map_location="cpu", weights_only=False)
    for name, tensor in d6_raw["model"].items():
        gather_raw["model"][name] = tensor
    gather_raw["upgrade_provenance"]["source_checkpoint_sha256"] = (
        arm.gather.corrected._file_sha(d6_checkpoint).removeprefix("sha256:")
    )
    gather_checkpoint = tmp_path / "d6-gather-init.pt"
    torch.save(gather_raw, gather_checkpoint)
    monkeypatch.setattr(
        arm,
        "D6_GATHER_INIT_SHA256",
        arm.gather.corrected._file_sha(gather_checkpoint),
    )

    repo = tmp_path / "composition-checkout"
    files = {}
    for relative in arm.SOURCE_FILES:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# bound composition source: {relative}\n", encoding="utf-8")
        files[relative] = arm.gather.corrected._file_ref(path)
    binding = {
        "repository_root": str(repo.resolve()),
        "git_commit": "composition-test-head",
        "files": files,
        "files_sha256": arm.gather.corrected._digest(files),
    }
    monkeypatch.setattr(arm, "_source_binding", lambda _repo: binding)
    monkeypatch.setattr(
        arm.executor_base,
        "_git_head",
        lambda _repo: "composition-test-head",
    )
    return type(
        "Args",
        (),
        {
            "source_manifest": source.source_manifest,
            "selected_dose_plan": source.selected_dose_plan,
            "selected_dose_report": source.selected_dose_report,
            "d6_checkpoint": d6_checkpoint,
            "d6_report": d6_report,
            "d6_progress": d6_progress,
            "gather_checkpoint": gather_checkpoint,
            "f7_gather_checkpoint": f7_gather_checkpoint,
            "architecture_audit": source.architecture_audit,
            "output_root": tmp_path / "composition-output",
            "repo": repo,
            "bound_executor": Path(files[arm.EXECUTOR_RELATIVE_PATH]["path"]),
        },
    )()


def test_prepares_and_replays_exact_selected_dose_d6_gather_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _composition_args(tmp_path, monkeypatch)
    manifest, path = arm.prepare(args)

    assert path.is_file()
    assert manifest["launch_authorized"] is False
    assert manifest["diagnostic_execution_authorized"] is True
    assert manifest["only_declared_model_delta"] == (
        "train function-preserving target_gather_proj on frozen exact D6 parent"
    )
    assert manifest["d6_parent"]["checkpoint"]["sha256"] == arm.D6_PARENT_SHA256
    assert (
        manifest["function_preserving_upgrade"]["shared_parameters_bit_identical"]
        is True
    )
    assert manifest["function_preserving_upgrade"]["new_parameters"] == list(
        arm.gather.EXPECTED_NEW_PARAMETERS
    )
    assert manifest["initialization_treatment"]["sha256"] == (arm.D6_GATHER_INIT_SHA256)
    assert manifest["matched_contract"] == {
        "reference_checkpoint": manifest["d6_parent"]["checkpoint"],
        "evaluation_reference": "exact_D6_parent",
        "candidate_chaining": False,
        "world_size": 8,
        "local_batch_size": 64,
        "global_batch_size": 512,
        "optimizer_steps": 1024,
        "global_row_dose": 524_288,
        "fresh_adam": True,
        "action_module_lr_mult": 4.0,
        "value_lr_mult": 1.0,
        "freeze_modules": ["trunk", "action_encoder", "policy_head", "value_heads"],
        "required_trainable_prefixes": ["target_gather_proj"],
        "new_trainable_parameter_names": list(arm.gather.EXPECTED_NEW_PARAMETERS),
        "mature_parameters_trainable": False,
        "symmetry_augment": True,
        "symmetry_augment_events": True,
        "selected_TEMP_data_descriptor_and_seed_unchanged": True,
        "sampler_batch_partition_unchanged": False,
        "selected_TEMP_policy_value_loss_coefficients_and_forward_loss_unchanged": True,
        "treatment_distributed_symmetry_contract": (
            "per_rank_seedsequence_checkpoint_resume_v1"
        ),
    }
    assert manifest["effective_trainable_objective"] == {
        "policy_only": True,
        "policy_loss_reaches_target_gather_proj": True,
        "value_loss_forward_computed": True,
        "value_loss_reaches_target_gather_proj": False,
        "reason": (
            "target_gather_proj affects policy logits only; trunk and value heads "
            "are frozen, and the completion receipt must bind policy-active dose"
        ),
    }
    assert manifest["d6_parent"]["symmetry_rng_provenance"].startswith(
        "historical_single_stream_receipt"
    )
    assert manifest["optimizer_geometry_contract"] == {
        "source_selected_TEMP": {
            "world_size": 8,
            "local_batch_size": 512,
            "global_batch_size": 4096,
            "optimizer_steps": 128,
            "global_row_dose": 524_288,
        },
        "treatment_adapter_commissioning": {
            "world_size": 8,
            "local_batch_size": 64,
            "global_batch_size": 512,
            "optimizer_steps": 1024,
            "global_row_dose": 524_288,
            "lr_warmup_steps": 100,
            "integrated_lr_step_equivalents": 974.5,
            "action_integrated_lr_step_equivalents": 3898.0,
        },
        "optimizer_update_count_multiplier": 8.0,
        "row_dose_unchanged": True,
        "reason": (
            "the zero-output gather residual needs the proven 1024-update "
            "commissioning geometry; this is not the 128-update D6-short arm"
        ),
    }
    command = manifest["command"]
    assert arm.gather.corrected._option(command, "--batch-size") == "64"
    assert arm.gather.corrected._option(command, "--max-steps") == "1024"
    assert arm.gather.corrected._option(command, "--grad-accum-steps") == "1"
    assert arm.gather.corrected._option(command, "--action-module-lr-mult") == "4.0"
    assert arm.gather.corrected._option(command, "--value-lr-mult") == "1.0"
    assert arm.gather.corrected._option(command, "--freeze-modules") == (
        "trunk,action_encoder,policy_head,value_heads"
    )
    assert (
        arm.gather.corrected._option(command, "--require-only-trainable-prefixes")
        == "target_gather_proj"
    )
    assert command.count("--symmetry-augment") == 1
    assert command.count("--symmetry-augment-events") == 1
    assert "--no-resume-optimizer" in command
    assert (
        manifest["evaluation_contract"]["primary_opponent"]
        == manifest["d6_parent"]["checkpoint"]
    )

    verified = arm.verify(path, expected_executor=args.bound_executor)
    assert verified["command"] == command
    assert verified["output_root"] == args.output_root.resolve()


def test_selected_short_d6_is_an_independent_exact_parent_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _composition_args(tmp_path, monkeypatch)
    checkpoint_sha = arm.gather.corrected._file_sha(args.d6_checkpoint)

    report = json.loads(args.d6_report.read_text(encoding="utf-8"))
    report.update(
        {
            "max_steps": 128,
            "steps_completed": 128,
            "training_row_draws": 524_288,
            "symmetry_augment_events": True,
        }
    )
    args.d6_report.write_text(json.dumps(report), encoding="utf-8")

    progress = json.loads(args.d6_progress.read_text(encoding="utf-8"))
    progress["optimizer_step"] = 128
    progress["symmetry_rng_state"] = {
        "schema_version": "train-bc-rank-symmetry-rng-v1",
        "world_size": 8,
        "rank_states": [{"state": rank} for rank in range(8)],
    }
    progress["progress_sha256"] = arm.gather.corrected._digest(
        {key: value for key, value in progress.items() if key != "progress_sha256"}
    )
    args.d6_progress.write_text(json.dumps(progress), encoding="utf-8")

    monkeypatch.setattr(arm, "D6_SHORT_PARENT_SHA256", checkpoint_sha)
    monkeypatch.setattr(
        arm, "D6_SHORT_REPORT_SHA256", arm.gather.corrected._file_sha(args.d6_report)
    )
    monkeypatch.setattr(
        arm,
        "D6_SHORT_PROGRESS_SHA256",
        arm.gather.corrected._file_sha(args.d6_progress),
    )
    monkeypatch.setattr(
        arm,
        "D6_SHORT_GATHER_INIT_SHA256",
        arm.gather.corrected._file_sha(args.gather_checkpoint),
    )

    manifest, path = arm.prepare(args)
    assert manifest["d6_parent"]["parent_profile"] == "selected_short_d6"
    assert manifest["d6_parent"]["checkpoint"]["sha256"] == checkpoint_sha
    assert manifest["d6_parent"]["training_contract"]["max_steps"] == 128
    assert manifest["d6_parent"]["training_contract"]["training_row_draws"] == 524_288
    assert manifest["d6_parent"]["symmetry_rng_provenance"] == (
        "per_rank_seedsequence_checkpoint_resume_v1"
    )
    assert (
        arm.verify(path, expected_executor=args.bound_executor)["manifest"] == manifest
    )


def test_refuses_d6_parent_that_did_not_complete_symmetry_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _composition_args(tmp_path, monkeypatch)
    report = json.loads(args.d6_report.read_text(encoding="utf-8"))
    report["symmetry_augment"] = False
    args.d6_report.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(
        arm, "D6_REPORT_SHA256", arm.gather.corrected._file_sha(args.d6_report)
    )
    with pytest.raises(arm.CompositionArmError, match="training provenance drift"):
        arm.prepare(args)


def test_refuses_treatment_not_upgraded_from_exact_d6_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _composition_args(tmp_path, monkeypatch)
    raw = arm.gather._torch_load(args.gather_checkpoint)
    raw["upgrade_provenance"]["source_checkpoint_sha256"] = "0" * 64
    torch.save(raw, args.gather_checkpoint)
    with pytest.raises(arm.CompositionArmError, match="function-preserving provenance"):
        arm.prepare(args)


def test_refuses_function_preserving_f7_gather_when_parent_is_d6(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _composition_args(tmp_path, monkeypatch)
    args.gather_checkpoint = args.f7_gather_checkpoint
    monkeypatch.setattr(
        arm,
        "D6_GATHER_INIT_SHA256",
        arm.gather.corrected._file_sha(args.f7_gather_checkpoint),
    )
    with pytest.raises(arm.CompositionArmError, match="function-preserving provenance"):
        arm.prepare(args)


def test_verifier_rejects_command_drift_and_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _composition_args(tmp_path, monkeypatch)
    _, path = arm.prepare(args)
    payload = json.loads(path.read_text(encoding="utf-8"))
    command = payload["command"]
    command[command.index("--action-module-lr-mult") + 1] = "2.0"
    payload["command_sha256"] = arm.gather.corrected._digest(command)
    payload["manifest_sha256"] = arm.gather.corrected._digest(
        {key: value for key, value in payload.items() if key != "manifest_sha256"}
    )
    path.chmod(0o644)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(arm.CompositionArmError, match=r"exact D6\+gather derivation"):
        arm.verify(path, expected_executor=args.bound_executor)

    # Re-prepare a clean manifest in a separate fixture, then prove the launch
    # verifier is one-shot rather than merely documenting that intent.
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    args = _composition_args(fresh, monkeypatch)
    _, path = arm.prepare(args)
    (args.output_root / "candidate.pt").write_bytes(b"pre-existing output")
    with pytest.raises(arm.CompositionArmError, match="output already exists"):
        arm.verify(path, expected_executor=args.bound_executor)


def test_selected_temp_geometry_must_remain_524288_draws(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _composition_args(tmp_path, monkeypatch)
    plan = json.loads(args.selected_dose_plan.read_text(encoding="utf-8"))
    command = plan["runs"][0]["command"]
    command[command.index("--max-steps") + 1] = "129"
    plan["runs"][0]["command_sha256"] = arm.gather.corrected._digest(command)
    plan["plan_sha256"] = arm.gather.corrected._digest(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )
    args.selected_dose_plan.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(arm.gather.ArmError, match="exact short-dose TEMP derivation"):
        arm.prepare(args)
