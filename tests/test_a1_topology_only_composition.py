from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tools import a1_topology_only_composition_arm as arm
from tools import a1_topology_only_composition_completion as completion


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


def test_command_is_exact_short_topology_only_geometry(tmp_path: Path) -> None:
    trainer = tmp_path / "repo" / "tools" / "train_bc.py"
    trainer.parent.mkdir(parents=True)
    trainer.write_text("# trainer\n", encoding="utf-8")
    initializer = tmp_path / "topology-init.pt"
    initializer.write_bytes(b"checkpoint")
    output = tmp_path / "run"

    command, _changes = arm._derive_command(  # noqa: SLF001
        _source_command(tmp_path),
        trainer=trainer,
        initializer=initializer,
        output_root=output,
        optimizer_steps=128,
    )

    option = arm.gather_arm.gather.corrected._option  # noqa: SLF001
    assert option(command, "--batch-size") == "512"
    assert option(command, "--max-steps") == "128"
    assert option(command, "--trunk-lr-mult") == "4.0"
    assert option(command, "--action-module-lr-mult") == "1.0"
    assert option(command, "--value-lr-mult") == "1.0"
    assert option(command, "--amp") == "none"
    assert option(command, "--float32-matmul-precision") == "highest"
    assert option(command, "--freeze-modules") == arm.FREEZE_MODULES
    assert option(command, "--require-only-trainable-prefixes") == arm.TRAINABLE_PREFIX
    assert command.count("--no-resume-optimizer") == 1
    assert command.count("--symmetry-augment") == 1
    assert command.count("--symmetry-augment-events") == 1
    assert arm._dose_geometry(128)["global_row_dose"] == 524_288  # noqa: SLF001
    with pytest.raises(arm.TopologyCompositionError, match="selected short geometry"):
        arm._dose_geometry(256)  # noqa: SLF001


def test_parent_selection_is_explicit_and_replayable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "short-d6.pt"
    report = tmp_path / "short-d6.report.json"
    progress = tmp_path / "short-d6.progress.json"
    evidence = tmp_path / "gather-adjudication.json"
    for path in (checkpoint, report, progress, evidence):
        path.write_bytes(path.name.encode())
    parent = {
        "parent_profile": arm.PARENT_DIRECT_SHORT_D6,
        "checkpoint": arm._file_ref(checkpoint),  # noqa: SLF001
        "report": arm._file_ref(report),  # noqa: SLF001
        "progress": arm._file_ref(progress),  # noqa: SLF001
        "completion_receipt": None,
        "source_d6_evidence": {"parent_profile": "selected_short_d6"},
        "architecture": {
            "state_trunk": "transformer",
            "action_target_gather": False,
            "topology_residual_adapter": False,
        },
    }
    parent["parent_sha256"] = arm._digest(parent)  # noqa: SLF001
    monkeypatch.setattr(arm, "_direct_short_d6_parent", lambda *_args: parent)

    selection_path = tmp_path / "parent-selection.json"
    arm.issue_parent_selection(
        parent_profile=arm.PARENT_DIRECT_SHORT_D6,
        output=selection_path,
        selection_evidence=[evidence],
        selection_basis="all sealed gather doses were rejected by exact-parent H2H",
        d6_checkpoint=checkpoint,
        d6_report=report,
        d6_progress=progress,
    )
    verified = arm.verify_parent_selection(selection_path)
    assert verified["status"] == "selected"
    assert verified["parent"] == parent

    forged = json.loads(selection_path.read_text(encoding="utf-8"))
    forged["status"] = "candidate"
    forged.pop("selection_sha256")
    forged["selection_sha256"] = arm._digest(forged)  # noqa: SLF001
    forged_path = tmp_path / "unselected.json"
    forged_path.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(arm.TopologyCompositionError, match="schema/status"):
        arm.verify_parent_selection(forged_path)


def _topology_state(*, changed: bool) -> dict[str, torch.Tensor]:
    result = {
        "topology_residual_adapter.source_norm.weight": torch.zeros(640),
        "topology_residual_adapter.source_norm.bias": torch.zeros(640),
        "topology_residual_adapter.source_projection.weight": torch.zeros(640, 640),
        "topology_residual_adapter.source_projection.bias": torch.zeros(640),
        "topology_residual_adapter.message_norm.weight": torch.zeros(640),
        "topology_residual_adapter.message_norm.bias": torch.zeros(640),
        "topology_residual_adapter.output_projection.weight": torch.zeros(640, 640),
        "topology_residual_adapter.output_projection.bias": torch.zeros(640),
    }
    if changed:
        result = {name: value + 1 for name, value in result.items()}
    return result


def _checkpoint(path: Path, *, treatment: bool, mutate_inherited: bool = False) -> Path:
    model = {"blocks.0.weight": torch.arange(4, dtype=torch.float32)}
    if mutate_inherited:
        model["blocks.0.weight"][0] = 99
    model.update(_topology_state(changed=treatment))
    torch.save(
        {
            "config": {
                "fields": {
                    "action_size": 607,
                    "static_action_feature_size": 50,
                    "action_target_gather": False,
                    "topology_residual_adapter": True,
                }
            },
            "model": model,
        },
        path,
    )
    return path


def test_completion_requires_all_and_only_eight_topology_tensors(
    tmp_path: Path,
) -> None:
    initializer = _checkpoint(tmp_path / "init.pt", treatment=False)
    candidate = _checkpoint(tmp_path / "candidate.pt", treatment=True)
    evidence = completion._verify_topology_only_delta(initializer, candidate)  # noqa: SLF001
    assert evidence["changed_parameter_tensors"] == list(
        arm.EXPECTED_TOPOLOGY_PARAMETERS
    )
    assert evidence["changed_parameter_count"] == 823_040
    assert evidence["inherited_parameters_bit_identical"] is True

    bad = _checkpoint(tmp_path / "bad.pt", treatment=True, mutate_inherited=True)
    with pytest.raises(completion.CompletionError, match="outside/excluding"):
        completion._verify_topology_only_delta(initializer, bad)  # noqa: SLF001


def test_completion_requires_every_adam_state_at_completed_step(
    tmp_path: Path,
) -> None:
    state = {
        index: {
            "step": torch.tensor(128),
            "exp_avg": torch.zeros(1),
            "exp_avg_sq": torch.zeros(1),
        }
        for index in range(8)
    }
    payload = {
        "format": "plain",
        "optimizer": {
            "param_groups": [
                {"params": [], "lr": 3e-5, "base_lr": 3e-5},
                {"params": list(range(8)), "lr": 1.2e-4, "base_lr": 1.2e-4},
            ],
            "state": state,
        },
    }
    path = tmp_path / "candidate.pt.optimizer.pt"
    torch.save(payload, path)
    summary = completion._verify_optimizer_groups(path, optimizer_steps=128)  # noqa: SLF001
    assert summary["trunk_group_parameter_tensors"] == 8
    assert summary["optimizer_state_step"] == 128

    state[0]["step"] = torch.tensor(127)
    torch.save(payload, path)
    with pytest.raises(completion.CompletionError, match="completed dose"):
        completion._verify_optimizer_groups(path, optimizer_steps=128)  # noqa: SLF001


def test_completion_rejects_missing_or_failed_systemd_state() -> None:
    success = {
        "LoadState": "loaded",
        "ActiveState": "active",
        "SubState": "exited",
        "Result": "success",
        "ExecMainStatus": "0",
        "ExecMainCode": "1",
    }
    assert completion._verify_unit_state(success) == success  # noqa: SLF001
    with pytest.raises(completion.CompletionError, match="not complete"):
        completion._verify_unit_state({**success, "LoadState": "not-found"})  # noqa: SLF001
