from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tools import a1_static_action_residual_arm as arm
from tools import a1_static_action_residual_completion as completion
from tools import a1_topology_only_composition_arm as base


def _source_command(tmp_path: Path) -> list[str]:
    return [
        "python",
        str(tmp_path / "historical" / "train_bc.py"),
        "--max-steps",
        "128",
        "--batch-size",
        "512",
        "--grad-accum-steps",
        "1",
        "--optimizer",
        "adam",
        "--no-resume-optimizer",
        "--no-fused-optimizer",
        "--lr",
        "3e-05",
        "--lr-warmup-steps",
        "100",
        "--soft-target-weight",
        "0.9",
        "--value-loss-weight",
        "0.25",
        "--action-module-lr-mult",
        "1.0",
        "--value-lr-mult",
        "0.3",
        "--init-checkpoint",
        str(tmp_path / "old.pt"),
        "--checkpoint",
        str(tmp_path / "old-candidate.pt"),
        "--report",
        str(tmp_path / "old-report.json"),
    ]


def test_arm_is_scoped_and_changes_only_static_action_surface(tmp_path: Path) -> None:
    original = (
        base.SCHEMA,
        base.ACTION_MODULE_LR_MULT,
        base.TRAINABLE_PREFIX,
        base.REPORT_ARCHITECTURE_DELTA,
    )
    trainer = tmp_path / "repo" / "tools" / "train_bc.py"
    trainer.parent.mkdir(parents=True)
    trainer.write_text("# trainer\n", encoding="utf-8")
    initializer = tmp_path / "static-init.pt"
    initializer.write_bytes(b"checkpoint")

    command, _ = arm._derive_command(
        _source_command(tmp_path),
        trainer=trainer,
        initializer=initializer,
        output_root=tmp_path / "run",
        optimizer_steps=128,
    )
    option = arm.gather_arm.gather.corrected._option  # noqa: SLF001
    assert option(command, "--batch-size") == "512"
    assert option(command, "--max-steps") == "128"
    assert option(command, "--trunk-lr-mult") == "1.0"
    assert option(command, "--action-module-lr-mult") == "4.0"
    assert option(command, "--value-lr-mult") == "1.0"
    assert option(command, "--freeze-modules") == arm.FREEZE_MODULES
    assert option(command, "--require-only-trainable-prefixes") == (
        "static_action_residual_proj"
    )
    assert command.count("--symmetry-augment") == 1
    assert command.count("--symmetry-augment-events") == 1
    assert arm.EXPECTED_TOPOLOGY_PARAMETERS == (
        "static_action_residual_proj.bias",
        "static_action_residual_proj.weight",
    )
    assert arm.EXPECTED_TOPOLOGY_PARAMETER_COUNT == 14_720
    assert arm.INFERENCE_COST_CONTRACT["schema_version"] == (
        "a1-architecture-inference-cost-contract-v2"
    )
    assert set(arm.INFERENCE_COST_CONTRACT["required_profiles"]) == {
        "operational_b1",
        "d6_b12",
    }
    assert all(
        profile["return_q"] is False
        for profile in arm.INFERENCE_COST_CONTRACT["required_profiles"].values()
    )
    assert (
        base.SCHEMA,
        base.ACTION_MODULE_LR_MULT,
        base.TRAINABLE_PREFIX,
        base.REPORT_ARCHITECTURE_DELTA,
    ) == original


def test_upgrade_receipt_must_be_static_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "parent.pt"
    initializer = tmp_path / "static.pt"
    receipt = tmp_path / "upgrade.json"
    for path in (source, initializer, receipt):
        path.write_bytes(path.name.encode())
    parent_ref = arm._file_ref(source)
    evidence = {
        "module": arm.UPGRADE_MODULE,
        "source": parent_ref,
        "upgraded_initializer": arm._file_ref(initializer),
        "new_parameters": list(arm.EXPECTED_TOPOLOGY_PARAMETERS),
        "receipt": arm._file_ref(receipt),
    }
    monkeypatch.setattr(
        arm.architecture_upgrade, "verify_receipt", lambda _path: evidence
    )
    assert arm._validate_upgrade_receipt(receipt, parent_checkpoint=parent_ref) == evidence
    evidence["module"] = arm.architecture_upgrade.MODULE_TARGET_GATHER
    with pytest.raises(arm.TopologyCompositionError, match="upgrade is not"):
        arm._validate_upgrade_receipt(receipt, parent_checkpoint=parent_ref)


def _checkpoint(path: Path, *, changed: bool, inherited: float = 0.0) -> Path:
    value = float(changed)
    torch.save(
        {
            "config": {
                "fields": {
                    "action_size": 607,
                    "static_action_feature_size": 45,
                    "static_action_residual": True,
                }
            },
            "model": {
                "blocks.0.weight": torch.full((2,), inherited),
                "static_action_residual_proj.weight": torch.full((640, 22), value),
                "static_action_residual_proj.bias": torch.full((640,), value),
            },
        },
        path,
    )
    return path


def test_completion_requires_all_and_only_two_new_tensors(tmp_path: Path) -> None:
    initializer = _checkpoint(tmp_path / "init.pt", changed=False)
    candidate = _checkpoint(tmp_path / "candidate.pt", changed=True)
    result = completion._verify_static_action_delta(initializer, candidate)
    assert result["changed_parameter_tensors"] == list(
        arm.EXPECTED_TOPOLOGY_PARAMETERS
    )
    assert result["changed_parameter_count"] == 14_720
    bad = _checkpoint(tmp_path / "bad.pt", changed=True, inherited=1.0)
    with pytest.raises(completion.CompletionError, match="outside/excluding"):
        completion._verify_static_action_delta(initializer, bad)


def test_completion_requires_exact_action_local_optimizer_group(tmp_path: Path) -> None:
    state = {
        index: {
            "step": torch.tensor(128),
            "exp_avg": torch.zeros(1),
            "exp_avg_sq": torch.zeros(1),
        }
        for index in range(2)
    }
    payload = {
        "format": "plain",
        "optimizer": {
            "param_groups": [
                {"params": [], "lr": 3e-5, "base_lr": 3e-5},
                {"params": [0, 1], "lr": 1.2e-4, "base_lr": 1.2e-4},
            ],
            "state": state,
        },
    }
    path = tmp_path / "candidate.pt.optimizer.pt"
    torch.save(payload, path)
    result = completion._verify_optimizer_groups(path, optimizer_steps=128)
    assert result["action_group_parameter_tensors"] == 2
    assert result["optimizer_state_step"] == 128
    state[1]["step"] = torch.tensor(127)
    torch.save(payload, path)
    with pytest.raises(completion.CompletionError, match="completed dose"):
        completion._verify_optimizer_groups(path, optimizer_steps=128)


def test_completion_replays_rich_finalizer_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    finalizer = tmp_path / "static_completion.py"
    finalizer.write_text("# exact finalizer bytes\n", encoding="utf-8")
    candidate = tmp_path / "candidate.pt"
    candidate.write_bytes(b"candidate")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(completion, "__file__", str(finalizer))

    receipt = {
        "schema_version": completion.SCHEMA,
        "status": completion.STATUS,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "completion_finalizer": {
            **arm._file_ref(finalizer),
            "size_bytes": finalizer.stat().st_size,
        },
        "manifest": {"path": str(manifest)},
        "checkpoint": {"path": str(candidate)},
        "expected_checkpoint_sha256": "sha256:candidate",
        "unit_state": {"Result": "success"},
        "created_at_unix_ns": 1,
    }
    receipt["receipt_sha256"] = arm._digest(receipt)
    receipt_path = tmp_path / completion.COMPLETION_NAME
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    monkeypatch.setattr(
        completion,
        "build_completion",
        lambda *_args, **_kwargs: dict(receipt),
    )

    assert completion.verify_completion(receipt_path) == receipt

    incomplete = dict(receipt)
    incomplete["completion_finalizer"] = arm._file_ref(finalizer)
    incomplete.pop("receipt_sha256")
    incomplete["receipt_sha256"] = arm._digest(incomplete)
    receipt_path.write_text(json.dumps(incomplete), encoding="utf-8")
    with pytest.raises(completion.CompletionError, match="finalizer/digest drift"):
        completion.verify_completion(receipt_path)
