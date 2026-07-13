from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from tools import a1_belief_resource_arm as arm
from tools import a1_belief_resource_arm_execute as execute
from tools import a1_function_preserving_upgrade as upgrade


def _source_command(tmp_path: Path) -> list[str]:
    return [
        "/venv/python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=8",
        str(tmp_path / "source" / "tools" / "train_bc.py"),
        "--data",
        str(tmp_path / "descriptor.json"),
        "--init-checkpoint",
        str(tmp_path / "f7.pt"),
        "--checkpoint",
        str(tmp_path / "temp" / "candidate.pt"),
        "--report",
        str(tmp_path / "temp" / "train.report.json"),
        "--policy-kl-anchor-weight",
        "0.0",
        "--training-rng-rank-offset",
        "--mask-hidden-info",
    ]


def _upgrade_receipt(tmp_path: Path) -> dict:
    f7 = {"path": str(tmp_path / "f7.pt"), "sha256": arm.temperature.F7_SHA256}
    upgraded = {
        "path": str(tmp_path / "belief-f7.pt"),
        "sha256": "sha256:" + "b" * 64,
    }
    return {
        "schema_version": upgrade.SCHEMA,
        "module": upgrade.MODULE_BELIEF_RESOURCE_HEAD,
        "source": f7,
        "upgraded_initializer": upgraded,
        "flags": {"belief_resource_head": True},
        "initialization_seed": 73,
        "forward_max_diff": 0.0,
        "forward_identical_at_init": True,
        "shared_parameters_bit_identical": True,
        "shared_parameter_count": 1,
        "new_parameters": ["belief_resource_head.0.weight"],
        "new_parameter_initialization": {
            "belief_resource_head.0.weight": "ones"
        },
        "effective_source_config_sha256": "sha256:" + "1" * 64,
        "effective_upgraded_config_sha256": "sha256:" + "2" * 64,
        "seeded_parameter_sha256": {
            "belief_resource_head.1.weight": "sha256:" + "3" * 64
        },
        "receipt": {
            "path": str(tmp_path / "upgrade.receipt.json"),
            "sha256": "sha256:" + "4" * 64,
        },
    }


def _belief_config(**overrides) -> dict:
    config = {
        "belief_resource_head": True,
        "action_target_gather": False,
        "topology_residual_adapter": False,
        "edge_policy_head": False,
        "aux_subgoal_heads": False,
        "action_cross_attention_layers": 0,
    }
    config.update(overrides)
    return config


def test_command_is_exact_temp_projection_plus_belief_axis(tmp_path: Path) -> None:
    trainer = tmp_path / "repo" / "tools" / "train_bc.py"
    trainer.parent.mkdir(parents=True)
    trainer.write_text("# trainer\n", encoding="utf-8")
    initializer = tmp_path / "belief-f7.pt"
    initializer.write_bytes(b"belief")
    root = tmp_path / "belief"
    source = _source_command(tmp_path)

    command = arm._derive_command(
        source,
        trainer=trainer,
        initializer=initializer,
        output_root=root,
    )

    assert arm.temperature.base._option(command, "--data") == str(  # noqa: SLF001
        tmp_path / "descriptor.json"
    )
    assert arm.temperature.base._option(  # noqa: SLF001
        command, "--policy-kl-anchor-weight"
    ) == "0.0"
    assert arm.temperature.base._option(  # noqa: SLF001
        command, "--init-checkpoint"
    ) == str(initializer)
    assert command.count("--belief-resource-head") == 1
    assert arm.temperature.base._option(  # noqa: SLF001
        command, "--belief-resource-loss-weight"
    ) == "0.01"
    assert "--action-target-gather" not in command
    assert "--topology-residual-adapter" not in command


def test_command_refuses_preexisting_treatment_flag(tmp_path: Path) -> None:
    trainer = tmp_path / "repo" / "tools" / "train_bc.py"
    trainer.parent.mkdir(parents=True)
    trainer.write_text("# trainer\n", encoding="utf-8")
    initializer = tmp_path / "belief-f7.pt"
    initializer.write_bytes(b"belief")
    source = [*_source_command(tmp_path), "--belief-resource-head"]

    with pytest.raises(arm.BeliefArmError, match="already contains treatment"):
        arm._derive_command(
            source,
            trainer=trainer,
            initializer=initializer,
            output_root=tmp_path / "belief",
        )


def test_upgrade_contract_requires_exact_f7_and_belief_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = _upgrade_receipt(tmp_path)
    monkeypatch.setattr(arm, "_effective_config", lambda _path: _belief_config())

    assert arm._validate_upgrade_contract(
        receipt, f7_ref=receipt["source"]
    )["belief_resource_head"] is True

    receipt["source"] = {
        "path": str(tmp_path / "candidate.pt"),
        "sha256": "sha256:" + "c" * 64,
    }
    with pytest.raises(arm.BeliefArmError, match="exact f7"):
        arm._validate_upgrade_contract(
            receipt,
            f7_ref={
                "path": str(tmp_path / "f7.pt"),
                "sha256": arm.temperature.F7_SHA256,
            },
        )


def test_checkpoint_config_numpy_scalars_are_canonical_json_values() -> None:
    normalized = arm._json_config(
        {
            "layers": np.int64(6),
            "dropout": np.float32(0.05),
            "flags": (np.bool_(True),),
        }
    )

    assert normalized == {
        "layers": 6,
        "dropout": pytest.approx(0.05),
        "flags": [True],
    }
    assert type(normalized["layers"]) is int
    assert type(normalized["dropout"]) is float
    assert type(normalized["flags"][0]) is bool
    json.dumps(normalized, sort_keys=True)


def test_upgrade_contract_rejects_topology_or_gather(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = _upgrade_receipt(tmp_path)
    monkeypatch.setattr(
        arm,
        "_effective_config",
        lambda _path: _belief_config(action_target_gather=True),
    )

    with pytest.raises(arm.BeliefArmError, match="unrelated architecture"):
        arm._validate_upgrade_contract(receipt, f7_ref=receipt["source"])


def test_output_root_must_be_independent_and_fresh(tmp_path: Path) -> None:
    source = tmp_path / "temp"
    with pytest.raises(arm.BeliefArmError, match="independent"):
        arm._fresh_output_root(source / "belief", source)

    root = tmp_path / "belief"
    root.mkdir()
    (root / "candidate.pt").write_bytes(b"old")
    with pytest.raises(arm.BeliefArmError, match="not fresh"):
        arm._fresh_output_root(root, source)


def test_prepare_binds_temp_contract_and_only_belief_delta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_repo = tmp_path / "source"
    trainer = source_repo / "tools" / "train_bc.py"
    trainer.parent.mkdir(parents=True)
    trainer.write_text("# source trainer\n", encoding="utf-8")
    treatment_repo = tmp_path / "repo"
    treatment_trainer = treatment_repo / "tools" / "train_bc.py"
    executor = treatment_repo / arm.EXECUTOR_RELATIVE_PATH
    treatment_trainer.parent.mkdir(parents=True)
    executor.parent.mkdir(parents=True, exist_ok=True)
    treatment_trainer.write_text("# treatment trainer\n", encoding="utf-8")
    executor.write_text("# executor\n", encoding="utf-8")
    receipt = _upgrade_receipt(tmp_path)
    Path(receipt["upgraded_initializer"]["path"]).write_bytes(b"belief")
    source_command = _source_command(tmp_path)
    f7 = receipt["source"]
    source_manifest = {
        "manifest_sha256": "sha256:" + "a" * 64,
        "f7_parent": f7,
        "source_descriptor": {
            "path": str(tmp_path / "descriptor.json"),
            "sha256": "sha256:" + "d" * 64,
        },
        "validation_sentinel": {
            "path": str(tmp_path / "sentinel.json"),
            "sha256": "sha256:" + "e" * 64,
        },
        "component_bindings": [{"component_id": "n128_current"}],
        "stored_policy_component_temperatures": {
            "n128_current": 1.0,
            "n256_current": 1.11,
            "gen3_replay": 0.52,
        },
        "event_history_training_contract": {"public_observation_masked": True},
        "selected_dose": {
            "optimizer_steps": 1024,
            "world_size": 8,
            "per_rank_batch_size": 512,
            "global_samples": 4_194_304,
            "optimizer": "fresh_adam",
            "lr": 3e-5,
            "training_rng_rank_offset": True,
        },
        "runtime_python": {"lexical_path": "/venv/python"},
        "execution_preconditions": {"visible_gpu_count": 8},
    }
    source = {
        "manifest": source_manifest,
        "manifest_ref": {
            "path": str(tmp_path / "temp.manifest.json"),
            "sha256": "sha256:" + "f" * 64,
        },
        "command": source_command,
        "output_root": tmp_path / "temp",
    }
    files = {
        relative: {
            "path": str(
                treatment_trainer
                if relative == "tools/train_bc.py"
                else executor
                if relative == arm.EXECUTOR_RELATIVE_PATH
                else treatment_repo / relative
            ),
            "sha256": "sha256:" + "7" * 64,
        }
        for relative in arm.SOURCE_FILES
    }
    binding = {
        "repository_root": str(treatment_repo),
        "public_main_commit": "commit",
        "files": files,
        "files_sha256": arm.temperature.base._digest(files),  # noqa: SLF001
    }
    monkeypatch.setattr(arm.temperature, "verify", lambda _path: source)
    monkeypatch.setattr(arm.temperature, "_validate_recipe", lambda *_a, **_k: None)
    monkeypatch.setattr(arm.upgrade, "verify_receipt", lambda _path: receipt)
    monkeypatch.setattr(arm, "_effective_config", lambda _path: _belief_config())
    monkeypatch.setattr(arm, "_source_binding", lambda _repo: binding)

    manifest = arm.prepare(
        source_temperature_manifest=tmp_path / "temp.manifest.json",
        upgrade_receipt=tmp_path / "upgrade.receipt.json",
        repo=treatment_repo,
        output_root=tmp_path / "belief",
        manifest_path=tmp_path / "belief.manifest.json",
    )

    assert manifest["selected_dose"] == source_manifest["selected_dose"]
    assert manifest["stored_policy_component_temperatures"] == source_manifest[
        "stored_policy_component_temperatures"
    ]
    assert manifest["only_declared_optimization_delta"] == {
        "belief_resource_head": True,
        "belief_resource_loss_weight": 0.01,
    }
    assert manifest["matched_contract"]["candidate_chaining"] is False
    assert manifest["matched_contract"]["topology_or_gather"] is False
    assert manifest["diagnostic_only"] is True
    assert manifest["promotion_eligible"] is False
    assert manifest["diagnostic_execution_authorized"] is False
    assert manifest["obsolete_reason"] == arm.OBSOLETE_REASON


def test_legacy_full_dose_executor_is_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "belief"
    manifest_ref = {
        "path": str(tmp_path / "manifest.json"),
        "sha256": "sha256:" + "8" * 64,
    }
    verified = {
        "manifest": {"command_sha256": "sha256:" + "9" * 64},
        "manifest_ref": manifest_ref,
        "repo": tmp_path,
        "command": ["/venv/python", "trainer.py"],
        "output_root": root,
    }
    submitted: list[list[str]] = []

    def runner(command, **_kwargs):
        submitted.append(list(command))
        return subprocess.CompletedProcess(command, 0, stdout="queued\n", stderr="")

    monkeypatch.setattr(execute.arm, "verify", lambda _path: verified)
    with pytest.raises(execute.ExecutionError, match="4,194,304 rows / 1024 steps"):
        execute.execute(
            tmp_path / "manifest.json",
            unit="belief-test",
            runner=runner,
            idle_probe=lambda: [],
        )
    assert submitted == []
    assert not root.exists()
