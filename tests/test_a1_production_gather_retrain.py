from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import a1_production_gather_retrain as gather


def _ref(path: Path) -> dict[str, str]:
    return gather.base._ref(path)  # noqa: SLF001


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict, Path]:
    repo = tmp_path / "repo"
    for relative in gather.BOUND_SOURCE_FILES:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\n", encoding="utf-8")
    descriptor = tmp_path / "descriptor.json"
    descriptor.write_text("{}", encoding="utf-8")
    sentinel = tmp_path / "sentinel.json"
    sentinel.write_text("{}", encoding="utf-8")
    f7 = tmp_path / "f7.pt"
    f7.write_bytes(b"f7 corpus producer")
    r3 = tmp_path / "r3.pt"
    r3.write_bytes(b"r3 learner incumbent")
    upgraded = tmp_path / "r3-gather.pt"
    upgraded.write_bytes(b"r3 plus inert gather")
    upgrade_receipt = tmp_path / "upgrade.json"
    upgrade_receipt.write_text("{}", encoding="utf-8")
    completion_path = tmp_path / "r3.completion.json"
    completion_path.write_text("{}", encoding="utf-8")
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    python.chmod(0o755)

    command = [
        str(python), "-m", "torch.distributed.run", "--nproc-per-node", "8",
        str(repo / "tools/train_bc.py"), "--arch", "entity_graph",
        "--hidden-size", "640", "--graph-layers", "6", "--attention-heads", "8",
        "--epochs", "1", "--max-steps", "1024", "--batch-size", "512",
        "--grad-accum-steps", "1", "--optimizer", "adam", "--lr", "3e-05",
        "--lr-warmup-steps", "100", "--soft-target-weight", "0.9",
        "--value-loss-weight", "0.25", "--loser-sample-weight", "1.0",
        "--policy-aux-active-batch-size", "0", "--value-lr-mult", "0.3",
        "--action-module-lr-mult", "1.0", "--data", str(descriptor),
        "--validation-game-sentinel-manifest", str(sentinel),
        "--init-checkpoint", str(f7), "--checkpoint", str(tmp_path / "old.pt"),
        "--report", str(tmp_path / "old.json"), "--no-resume-optimizer",
        "--no-fused-optimizer", "--mask-hidden-info", "--graph-history-features",
        "--trust-curated-data-quality",
    ]
    source_manifest = {
        "command": command,
        "source_descriptor": _ref(descriptor),
        "validation_sentinel": _ref(sentinel),
        "f7_parent": _ref(f7),
    }
    completion = {
        "checkpoint": _ref(r3),
        "manifest": {"path": str(tmp_path / "source.manifest.json"), "sha256": "sha256:source"},
    }
    (tmp_path / "source.manifest.json").write_text(
        json.dumps(source_manifest), encoding="utf-8"
    )
    completion_ref = {"path": str(completion_path), "sha256": "sha256:completion"}
    upgrade_value = {
        "module": "entity_graph.action_target_gather.v1",
        "source": _ref(r3),
        "upgraded_initializer": _ref(upgraded),
        "flags": {"action_target_gather": True},
        "receipt_sha256": "sha256:" + "a" * 64,
        "receipt": _ref(upgrade_receipt),
    }
    components = [
        {"component_id": name, "corpus_meta": {"path": name, "sha256": name},
         "validation_manifest": {"path": name, "sha256": name},
         "payload_inventory_sha256": "sha256:" + str(index) * 64}
        for index, name in enumerate(("n128_current", "n256_current", "gen3_replay"), 1)
    ]
    inventories = [row["payload_inventory_sha256"] for row in components]
    monkeypatch.setattr(gather.base, "_assert_bound_checkout", lambda *_args: "abc")
    monkeypatch.setattr(
        gather, "_source_completion", lambda _path: (completion, completion_ref, source_manifest)
    )
    monkeypatch.setattr(gather.upgrade, "verify_receipt", lambda _path: upgrade_value)
    monkeypatch.setattr(
        gather.base, "_descriptor_inventory", lambda _path: (inventories, components)
    )
    monkeypatch.setattr(
        gather.base,
        "_verify_python_binding",
        lambda value: str(value["lexical_path"]),
    )
    output = tmp_path / "run"
    manifest_path = tmp_path / "gather.manifest.json"
    manifest = gather.prepare(
        source_completion=completion_path,
        architecture_upgrade_receipt=upgrade_receipt,
        repo=repo,
        output_root=output,
        manifest_path=manifest_path,
        python=python,
    )
    return manifest, manifest_path


def _rewrite(path: Path, mutate) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    value.pop("manifest_sha256", None)
    value["manifest_sha256"] = gather.base._digest(value)  # noqa: SLF001
    path.chmod(0o600)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_prepares_and_replays_exact_four_rank_adapter_operator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, path = _fixture(tmp_path, monkeypatch)
    assert manifest["operator"]["global_base_draws"] == 4_194_304
    assert manifest["operator"]["optimizer_steps"] == 2048
    assert manifest["operator"]["current_fraction"] == 0.8
    assert manifest["operator"]["exact_predecessor_replay_fraction"] == 0.2
    assert manifest["corpus_producer"] != manifest["learner_source_incumbent"]
    assert gather.base._option(manifest["command"], "--nproc-per-node") == "4"  # noqa: SLF001
    assert gather.base._option(  # noqa: SLF001
        manifest["command"], "--require-only-trainable-prefixes"
    ) == "target_gather_proj"
    assert gather.verify(path)["manifest"]["operator"] == manifest["operator"]


def test_rejects_semantically_rehashed_geometry_command_and_identity_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, path = _fixture(tmp_path, monkeypatch)
    _rewrite(path, lambda value: value["operator"].__setitem__("optimizer_steps", 2047))
    with pytest.raises(gather.GatherRetrainError, match="geometry"):
        gather.verify(path)

    path.unlink()
    _, path = _fixture(tmp_path / "command", monkeypatch)
    def command_drift(value):
        index = value["command"].index("--max-steps")
        value["command"][index + 1] = "2047"
        value["command_sha256"] = gather.base._digest(value["command"])  # noqa: SLF001
    _rewrite(path, command_drift)
    with pytest.raises(gather.GatherRetrainError, match="geometry"):
        gather.verify(path)

    path.unlink()
    _, path = _fixture(tmp_path / "source", monkeypatch)
    _rewrite(
        path,
        lambda value: value.__setitem__("learner_source_incumbent", value["corpus_producer"]),
    )
    with pytest.raises(gather.GatherRetrainError, match="source champion binding"):
        gather.verify(path)

    path.unlink()
    _, path = _fixture(tmp_path / "upgrade", monkeypatch)
    _rewrite(
        path,
        lambda value: value["function_preserving_upgrade"].__setitem__(
            "flags", {"action_target_gather": False}
        ),
    )
    with pytest.raises(gather.GatherRetrainError, match="upgrade/source"):
        gather.verify(path)


def _completion_outputs(manifest: dict, path: Path, *, unit: str) -> Path:
    root = Path(manifest["output_root"])
    root.mkdir(parents=True)
    checkpoint = root / "candidate.pt"
    checkpoint.write_bytes(b"trained gather candidate")
    optimizer = root / "candidate.pt.optimizer.pt"
    optimizer.write_bytes(b"fresh adam state at step 2048")
    report = {
        "checkpoint": str(checkpoint),
        "init_checkpoint": manifest["function_preserving_upgrade"][
            "upgraded_initializer"
        ]["path"],
        "init_checkpoint_sha256": manifest["function_preserving_upgrade"][
            "upgraded_initializer"
        ]["sha256"],
        "world_size": 4,
        "batch_size": 512,
        "effective_global_batch_size": 2048,
        "max_steps": 2048,
        "steps_completed": 2048,
        "training_row_draws": 4_194_304,
        "soft_target_weight": 0.9,
        "value_loss_weight": 0.25,
        "loser_sample_weight": 1.0,
        "action_module_lr_mult": 4.0,
        "freeze_modules": gather.FREEZE_MODULES,
        "require_only_trainable_prefixes": gather.TRAINABLE_PREFIX,
        "action_target_gather": True,
        "ddp_find_unused_parameters": True,
        "ddp_shard_data": False,
        "value_target_lambda": 1.0,
        "forced_action_weight": 0.0,
        "forced_row_value_weight": 1.0,
        "winner_sample_weight": 1.0,
        "lr_schedule": "flat",
        "lr_warmup_steps": 100,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "seed": 1,
        "training_information_surface": {
            "required_trainable_surface": {
                "prefixes": [gather.TRAINABLE_PREFIX],
                "parameter_tensors": 4,
                "parameters": 1234,
            }
        },
    }
    (root / "train.report.json").write_text(json.dumps(report), encoding="utf-8")
    progress = {
        "checkpoint": _ref(checkpoint),
        "optimizer": _ref(optimizer),
        "optimizer_step": 2048,
        "completed_epochs": 1,
        "rank_torch_rng_states": [f"rank-{index}" for index in range(4)],
    }
    progress["progress_sha256"] = gather.base._digest(progress)  # noqa: SLF001
    (root / "candidate.pt.training-progress.json").write_text(
        json.dumps(progress), encoding="utf-8"
    )
    claim = {
        "schema_version": gather.CLAIM_SCHEMA,
        "created_at_unix_ns": 1,
        "manifest": _ref(path),
        "unit": unit,
    }
    claim["claim_sha256"] = gather.base._digest(claim)  # noqa: SLF001
    claim_path = root / "execution.claim.json"
    claim_path.write_text(json.dumps(claim), encoding="utf-8")
    verified = gather.verify(path)
    execution_binding = gather._execution_binding(verified)  # noqa: SLF001
    submission = {
        "schema_version": gather.SUBMISSION_SCHEMA,
        "diagnostic_only": False,
        "production_eligible": True,
        "created_at_unix_ns": 2,
        "manifest": _ref(path),
        "claim": _ref(claim_path),
        "unit": unit,
        "command_sha256": manifest["command_sha256"],
        "systemd_command_sha256": gather.base._digest(  # noqa: SLF001
            gather._systemd_command(verified, unit)  # noqa: SLF001
        ),
        "execution_binding": execution_binding,
        "execution_binding_sha256": gather.base._digest(execution_binding),  # noqa: SLF001
        "systemd_stdout": "Running as unit.",
    }
    submission["receipt_sha256"] = gather.base._digest(submission)  # noqa: SLF001
    (root / "submission.receipt.json").write_text(
        json.dumps(submission), encoding="utf-8"
    )
    return root / "candidate.pt.training-progress.json"


def test_finalize_binds_exact_progress_rng_and_fresh_optimizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, path = _fixture(tmp_path, monkeypatch)
    unit = "a1-gather-test"
    progress_path = _completion_outputs(manifest, path, unit=unit)
    monkeypatch.setattr(
        gather,
        "_verify_adapter_only_model_delta",
        lambda *_args: {"inherited_parameters_bit_identical": True},
    )
    state = lambda *_args, **_kwargs: (  # noqa: E731
        "ActiveState=inactive\nResult=success\nExecMainStatus=0\n"
    )
    completion = gather.finalize(path, unit=unit, state_reader=state)
    assert completion["operator_sha256"] == manifest["operator_sha256"]
    assert completion["progress"] == _ref(progress_path)
    assert completion["optimizer"]["path"].endswith("candidate.pt.optimizer.pt")


def test_finalize_rejects_rehashed_progress_geometry_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, path = _fixture(tmp_path, monkeypatch)
    unit = "a1-gather-test"
    progress_path = _completion_outputs(manifest, path, unit=unit)
    monkeypatch.setattr(
        gather,
        "_verify_adapter_only_model_delta",
        lambda *_args: {"inherited_parameters_bit_identical": True},
    )
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["optimizer_step"] = 2047
    progress.pop("progress_sha256")
    progress["progress_sha256"] = gather.base._digest(progress)  # noqa: SLF001
    progress_path.write_text(json.dumps(progress), encoding="utf-8")
    with pytest.raises(gather.GatherRetrainError, match="progress/RNG/optimizer"):
        gather.finalize(
            path,
            unit=unit,
            state_reader=lambda *_args, **_kwargs: (
                "ActiveState=inactive\nResult=success\nExecMainStatus=0\n"
            ),
        )


def test_model_delta_requires_all_and_only_four_gather_tensors(tmp_path: Path) -> None:
    import torch

    names = sorted(
        gather.upgrade.ALLOWLIST[gather.upgrade.MODULE_TARGET_GATHER][
            "new_parameter_initialization"
        ]
    )
    model = {"trunk.weight": torch.ones(2, 2)}
    for name in names:
        model[name] = torch.zeros(2, 2)
    initializer = tmp_path / "init.pt"
    candidate = tmp_path / "candidate.pt"
    torch.save({"model": model}, initializer)
    changed = {key: value.clone() for key, value in model.items()}
    for index, name in enumerate(names, start=1):
        changed[name].fill_(float(index))
    torch.save({"model": changed}, candidate)
    evidence = gather._verify_adapter_only_model_delta(initializer, candidate)  # noqa: SLF001
    assert evidence["changed_parameter_tensors"] == names
    assert evidence["inherited_parameters_bit_identical"] is True

    changed["trunk.weight"][0, 0] = 2
    torch.save({"model": changed}, candidate)
    with pytest.raises(gather.GatherRetrainError, match="exact gather adapter"):
        gather._verify_adapter_only_model_delta(initializer, candidate)  # noqa: SLF001
