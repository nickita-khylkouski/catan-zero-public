from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from tools import a1_corrected_policy_arm as arm


def _write_json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _source_receipt(
    path: Path,
    command: list[str],
    *,
    parent_checkpoint_sha256: str = "sha256:parent",
    descriptor_sha256: str = "sha256:descriptor",
    validation_manifest_sha256: str = "sha256:validation",
) -> Path:
    payload = {
        "schema_version": "existing-sealed-training-receipt-v4",
        "diagnostic_only": True,
        "promotion_eligible": False,
        "command": command,
        "command_sha256": arm._digest(command),
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "descriptor_sha256": descriptor_sha256,
        "validation_manifest_sha256": validation_manifest_sha256,
    }
    payload["receipt_sha256"] = arm._digest(payload)
    return _write_json(path, payload)


def _base_command(tmp_path: Path) -> list[str]:
    return [
        "/venv/bin/python", "-m", "torch.distributed.run", "--standalone",
        "--nproc_per_node=8", "/old/tools/train_bc.py",
        "--data", "/old/descriptor.json", "--data-format", "memmap",
        "--validation-game-seed-manifest", "/old/validation.json",
        "--init-checkpoint", "/old/f7.pt", "--checkpoint", "/old/candidate.pt",
        "--report", "/old/report.json", "--batch-size", "512",
        "--grad-accum-steps", "1", "--max-steps", "1024", "--epochs", "1",
        "--loser-sample-weight", "0.3", "--winner-sample-weight", "1.0",
        "--forced-action-weight", "0.0", "--forced-row-value-weight", "1.0",
        "--policy-loss-weight", "1.0", "--soft-target-source", "policy",
        "--soft-target-weight", "0.9", "--soft-target-temperature", "0.7",
        "--soft-target-min-legal-coverage", "0.5",
        "--policy-kl-anchor-direction", "forward", "--policy-kl-anchor-weight", "0.0",
        "--value-loss-weight", "0.25", "--value-lr-mult", "0.3",
        "--value-target-lambda", "1.0", "--lr", "3e-5",
        "--lr-warmup-steps", "100", "--lr-schedule", "flat",
        "--no-resume-optimizer", "--fsdp", "--mask-hidden-info",
    ]


def _args(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> argparse.Namespace:
    descriptor = (tmp_path / "descriptor.json")
    descriptor.write_text("{}", encoding="utf-8")
    validation = (tmp_path / "validation.json")
    validation.write_text("{}", encoding="utf-8")
    f7 = (tmp_path / "f7.pt")
    f7.write_bytes(b"f7")
    source = _source_receipt(
        tmp_path / "source.json",
        _base_command(tmp_path),
        parent_checkpoint_sha256=arm._file_sha(f7),
        descriptor_sha256=arm._file_sha(descriptor),
        validation_manifest_sha256=arm._file_sha(validation),
    )
    lineage = []
    for role in arm.LINEAGE_ROLES:
        payload = {"schema_version": f"{role}-v1"}
        payload[arm.LINEAGE_DIGEST_FIELDS[role]] = arm._digest(payload)
        artifact = _write_json(tmp_path / f"{role}.json", payload)
        lineage.append(f"{role}={artifact}")
    monkeypatch.setattr(
        arm,
        "_validate_descriptor",
        lambda path: (
            {"descriptor_fingerprint": "sha256:descriptor"}, arm._file_ref(path)
        ),
    )
    monkeypatch.setattr(
        arm,
        "_source_binding",
        lambda repo: {"repository_root": str(repo), "git_commit": "abc", "files": {}},
    )
    monkeypatch.setattr(
        arm,
        "_rebind_a1_metadata",
        lambda command, repo: {"effective_recipe": {}, "code_binding": {}},
    )
    return argparse.Namespace(
        source_receipt=source,
        descriptor=descriptor,
        validation_manifest=validation,
        f7_checkpoint=f7,
        expected_f7_sha256=arm._file_sha(f7),
        failed_lineage_artifact=lineage,
        output_root=tmp_path / "out",
        repo=tmp_path,
    )


def test_prepares_exact_one_dose_pure_current_policy_arm_without_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, path = arm.prepare(_args(tmp_path, monkeypatch))
    assert path.is_file()
    assert manifest["launch_authorized"] is False
    assert manifest["launch_interface_present"] is False
    assert manifest["recipe"]["base_value_row_dose"] == 4_194_304
    assert manifest["recipe"]["policy_aux_active_row_dose"] == 1_048_576
    assert manifest["recipe"]["policy_distillation_component_ids"] == [
        "n128_current", "n256_current"
    ]
    assert manifest["semantic_risk"]["replay_value_is_off_policy"] is True
    assert manifest["semantic_risk"]["descriptor_supports_value_component_scope"] is False
    command = manifest["command"]
    assert arm._option(command, "--soft-target-weight") == "1.0"
    assert arm._option(command, "--policy-aux-active-batch-size") == "128"
    assert arm._option(command, "--policy-kl-anchor-weight") == "0.0"
    assert arm._option(command, "--loser-sample-weight") == "1.0"
    assert arm._option(command, "--max-steps") == "1024"
    assert command[command.index("torch.distributed.run") + 1] == "--standalone"
    assert [row["role"] for row in manifest["failed_retry_lineage"]["artifacts"]] == list(
        arm.LINEAGE_ROLES
    )


def test_refuses_incomplete_failed_retry_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path, monkeypatch)
    args.failed_lineage_artifact.pop()
    with pytest.raises(arm.ArmError, match="lineage is incomplete"):
        arm.prepare(args)


def test_refuses_source_receipt_command_digest_drift(tmp_path: Path) -> None:
    path = _source_receipt(tmp_path / "source.json", _base_command(tmp_path))
    payload = json.loads(path.read_text())
    payload["command"].extend(("--lr", "999"))
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(arm.ArmError, match="semantic digest"):
        arm._load_source_receipt(path)


def test_descriptor_requires_current_only_policy_and_zero_replay_kl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "descriptor.json"
    path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        arm.train_bc,
        "_preflight_memmap_composite_descriptor",
        lambda _path: {
            "schema_version": "memmap_composite_v2",
            "policy_distillation_scope_explicit": True,
            "policy_distillation_component_ids": ["n128_current", "n256_current", "gen3_replay"],
            "component_ids": ["n128_current", "n256_current", "gen3_replay"],
            "policy_kl_anchor_component_ids": ["gen3_replay"],
            "learner_recipe_overrides": {
                "policy_kl_anchor_weight": 0.0,
                "policy_kl_anchor_direction": "forward",
                "loser_sample_weight": 1.0,
            },
        },
    )
    with pytest.raises(arm.ArmError, match="n128/n256 current policy teachers"):
        arm._validate_descriptor(path)


def test_command_requires_fsdp_and_hidden_information_masking(tmp_path: Path) -> None:
    command = _base_command(tmp_path)
    command.remove("--fsdp")
    with pytest.raises(arm.ArmError, match="required safety flag"):
        arm._derive_command(
            command,
            repo=tmp_path,
            descriptor=tmp_path / "d",
            validation=tmp_path / "v",
            f7=tmp_path / "f",
            output_root=tmp_path / "out",
        )


def test_rebinds_full_tracked_runtime_closure_and_effective_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = tmp_path / "tools" / "train_bc.py"
    trainer.parent.mkdir()
    trainer.write_text("# trainer\n", encoding="utf-8")
    effective = {
        "batch_size": 512, "grad_accum_steps": 1, "global_batch_size": 4096,
        "world_size": 8, "max_steps": 1024, "epochs": 1,
        "loser_sample_weight": 0.3, "winner_sample_weight": 1.0,
        "forced_action_weight": 0.0, "forced_row_value_weight": 1.0,
        "policy_loss_weight": 1.0, "soft_target_source": "policy",
        "soft_target_weight": 0.9, "soft_target_temperature": 0.7,
        "soft_target_min_legal_coverage": 0.5, "policy_kl_anchor_weight": 0.0,
        "value_loss_weight": 0.25, "value_lr_mult": 0.3,
        "value_target_lambda": 1.0, "lr": 3e-5,
        "lr_warmup_steps": 100, "lr_schedule": "flat",
    }
    binding = {
        "schema_version": "a1-learner-ablation-code-binding-v1",
        "repository_root": "/old",
        "records": [{"kind": "learner_code", "relative_path": "tools/train_bc.py",
                     "path": "/old/train_bc.py", "sha256": "sha256:old"}],
        "code_tree_sha256": "sha256:old",
    }
    command = [
        "python", "tools/train_bc.py",
        "--a1-learner-ablation-id", "old",
        "--a1-effective-learner-recipe-json", arm._canonical(effective).decode(),
        "--a1-effective-learner-recipe-sha256", arm._digest(effective),
        "--a1-ablation-code-binding-json", arm._canonical(binding).decode(),
        "--a1-ablation-code-tree-sha256", "sha256:old",
        "--a1-reviewed-lock-file-sha256", "sha256:lock",
    ]
    monkeypatch.setattr(arm.subprocess, "run", lambda *args, **kwargs: None)
    result = arm._rebind_a1_metadata(command, tmp_path)
    assert result["effective_recipe"]["policy_aux_active_batch_size"] == 128
    assert result["effective_recipe"]["soft_target_weight"] == 1.0
    assert arm._option(command, "--a1-learner-ablation-id") == "l1-pure-current-aux128"
    rebound = json.loads(arm._option(command, "--a1-ablation-code-binding-json"))
    assert rebound["records"][0]["path"] == str(trainer)
    assert rebound["records"][0]["sha256"] == arm._file_sha(trainer)


def test_rebind_refuses_gradient_probe_in_training_runtime(tmp_path: Path) -> None:
    probe = tmp_path / "tools" / "a1_shared_trunk_gradient_probe.py"
    probe.parent.mkdir()
    probe.write_text("# diagnostic only\n", encoding="utf-8")
    effective = {
        "batch_size": 512, "grad_accum_steps": 1, "global_batch_size": 4096,
        "world_size": 8, "max_steps": 1024, "epochs": 1,
        "loser_sample_weight": 1.0, "winner_sample_weight": 1.0,
        "forced_action_weight": 0.0, "forced_row_value_weight": 1.0,
        "policy_loss_weight": 1.0, "soft_target_source": "policy",
        "soft_target_weight": 1.0, "soft_target_temperature": 0.7,
        "soft_target_min_legal_coverage": 0.5, "policy_kl_anchor_weight": 0.0,
        "value_loss_weight": 0.25, "value_lr_mult": 0.3,
        "value_target_lambda": 1.0, "lr": 3e-5,
        "lr_warmup_steps": 100, "lr_schedule": "flat",
    }
    binding = {"records": [{"kind": "learner_code",
        "relative_path": "tools/a1_shared_trunk_gradient_probe.py"}]}
    command = ["python", "tools/train_bc.py"]
    for flag, value in (
        ("--a1-learner-ablation-id", "old"),
        ("--a1-effective-learner-recipe-json", arm._canonical(effective).decode()),
        ("--a1-effective-learner-recipe-sha256", arm._digest(effective)),
        ("--a1-ablation-code-binding-json", arm._canonical(binding).decode()),
        ("--a1-ablation-code-tree-sha256", "sha256:old"),
        ("--a1-reviewed-lock-file-sha256", "sha256:lock"),
    ):
        command.extend((flag, value))
    with pytest.raises(arm.ArmError, match="untracked gradient probe"):
        arm._rebind_a1_metadata(command, tmp_path)
