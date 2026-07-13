from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tools import a1_topology_only_composition_arm as topology_only
from tools import a1_topology_target_gather_composition_arm as arm
from tools import a1_topology_target_gather_composition_completion as completion


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


def test_specialization_is_scoped_and_command_is_exact(tmp_path: Path) -> None:
    original = (
        topology_only.SCHEMA,
        topology_only.ACTION_MODULE_LR_MULT,
        topology_only.TRAINABLE_PREFIX,
    )
    trainer = tmp_path / "repo" / "tools" / "train_bc.py"
    trainer.parent.mkdir(parents=True)
    trainer.write_text("# trainer\n", encoding="utf-8")
    initializer = tmp_path / "combined-init.pt"
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
    assert option(command, "--trunk-lr-mult") == "4.0"
    assert option(command, "--action-module-lr-mult") == "4.0"
    assert option(command, "--value-lr-mult") == "1.0"
    assert option(command, "--amp") == "none"
    assert option(command, "--float32-matmul-precision") == "highest"
    assert option(command, "--freeze-modules") == arm.FREEZE_MODULES
    assert option(command, "--require-only-trainable-prefixes") == (
        "topology_residual_adapter,target_gather_proj"
    )
    assert len(arm.EXPECTED_TOPOLOGY_PARAMETERS) == 12
    assert arm.EXPECTED_TOPOLOGY_PARAMETER_COUNT == 1_234_560
    assert (
        topology_only.SCHEMA,
        topology_only.ACTION_MODULE_LR_MULT,
        topology_only.TRAINABLE_PREFIX,
    ) == original


def test_upgrade_receipt_must_be_exact_combined_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "parent.pt"
    initializer = tmp_path / "combined.pt"
    receipt = tmp_path / "upgrade.json"
    for path in (source, initializer, receipt):
        path.write_bytes(path.name.encode())
    parent_ref = arm._file_ref(source)
    combined = {
        "module": arm.UPGRADE_MODULE,
        "source": parent_ref,
        "upgraded_initializer": arm._file_ref(initializer),
        "new_parameters": list(arm.EXPECTED_TOPOLOGY_PARAMETERS),
        "receipt": arm._file_ref(receipt),
    }
    monkeypatch.setattr(
        arm.architecture_upgrade, "verify_receipt", lambda _path: combined
    )
    assert (
        arm._validate_upgrade_receipt(receipt, parent_checkpoint=parent_ref) == combined
    )
    combined["module"] = arm.architecture_upgrade.MODULE_TOPOLOGY_RESIDUAL
    with pytest.raises(arm.TopologyCompositionError, match="upgrade is not"):
        arm._validate_upgrade_receipt(receipt, parent_checkpoint=parent_ref)


def _adapter_state(*, changed: bool) -> dict[str, torch.Tensor]:
    tensors = {
        "topology_residual_adapter.source_norm.weight": torch.zeros(640),
        "topology_residual_adapter.source_norm.bias": torch.zeros(640),
        "topology_residual_adapter.source_projection.weight": torch.zeros(640, 640),
        "topology_residual_adapter.source_projection.bias": torch.zeros(640),
        "topology_residual_adapter.message_norm.weight": torch.zeros(640),
        "topology_residual_adapter.message_norm.bias": torch.zeros(640),
        "topology_residual_adapter.output_projection.weight": torch.zeros(640, 640),
        "topology_residual_adapter.output_projection.bias": torch.zeros(640),
        "target_gather_proj.0.weight": torch.zeros(640),
        "target_gather_proj.0.bias": torch.zeros(640),
        "target_gather_proj.1.weight": torch.zeros(640, 640),
        "target_gather_proj.1.bias": torch.zeros(640),
    }
    return {name: value + int(changed) for name, value in tensors.items()}


def _checkpoint(path: Path, *, changed: bool, inherited: float = 0.0) -> Path:
    model = {"blocks.0.weight": torch.full((2,), inherited)}
    model.update(_adapter_state(changed=changed))
    torch.save(
        {
            "config": {
                "fields": {
                    "action_size": 607,
                    "static_action_feature_size": 50,
                    "action_target_gather": True,
                    "topology_residual_adapter": True,
                }
            },
            "model": model,
        },
        path,
    )
    return path


def test_completion_requires_all_and_only_twelve_additions(tmp_path: Path) -> None:
    initializer = _checkpoint(tmp_path / "init.pt", changed=False)
    candidate = _checkpoint(tmp_path / "candidate.pt", changed=True)
    result = completion._verify_topology_target_gather_delta(initializer, candidate)
    assert result["changed_parameter_tensors"] == list(arm.EXPECTED_TOPOLOGY_PARAMETERS)
    assert result["changed_parameter_count"] == 1_234_560
    bad = _checkpoint(tmp_path / "bad.pt", changed=True, inherited=1.0)
    with pytest.raises(completion.CompletionError, match="outside/excluding"):
        completion._verify_topology_target_gather_delta(initializer, bad)


def test_completion_requires_exact_three_optimizer_groups(tmp_path: Path) -> None:
    state = {
        index: {
            "step": torch.tensor(128),
            "exp_avg": torch.zeros(1),
            "exp_avg_sq": torch.zeros(1),
        }
        for index in range(12)
    }
    payload = {
        "format": "plain",
        "optimizer": {
            "param_groups": [
                {"params": [], "lr": 3e-5, "base_lr": 3e-5},
                {"params": list(range(4)), "lr": 1.2e-4, "base_lr": 1.2e-4},
                {"params": list(range(4, 12)), "lr": 1.2e-4, "base_lr": 1.2e-4},
            ],
            "state": state,
        },
    }
    path = tmp_path / "candidate.pt.optimizer.pt"
    torch.save(payload, path)
    result = completion._verify_optimizer_groups(path, optimizer_steps=128)
    assert result["action_group_parameter_tensors"] == 4
    assert result["trunk_group_parameter_tensors"] == 8
    state[11]["step"] = torch.tensor(127)
    torch.save(payload, path)
    with pytest.raises(completion.CompletionError, match="completed dose"):
        completion._verify_optimizer_groups(path, optimizer_steps=128)
