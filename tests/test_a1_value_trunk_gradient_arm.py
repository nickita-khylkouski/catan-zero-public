from __future__ import annotations

from pathlib import Path

import pytest

from tools import a1_value_trunk_gradient_arm as arm


def _source_command(tmp_path: Path) -> list[str]:
    trainer = tmp_path / "source" / "tools" / "train_bc.py"
    trainer.parent.mkdir(parents=True, exist_ok=True)
    trainer.write_text("# source\n", encoding="utf-8")
    return [
        "/venv/python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=8",
        str(trainer),
        "--arch",
        "entity_graph",
        "--data",
        str(tmp_path / "descriptor.json"),
        "--data-format",
        "memmap",
        "--init-checkpoint",
        str(tmp_path / "f7.pt"),
        "--checkpoint",
        str(tmp_path / "temp" / "candidate.pt"),
        "--report",
        str(tmp_path / "temp" / "train.report.json"),
        "--batch-size",
        "512",
        "--grad-accum-steps",
        "1",
        "--max-steps",
        "1024",
        "--epochs",
        "1",
        "--lr",
        "3e-05",
        "--value-lr-mult",
        "0.3",
        "--value-loss-weight",
        "0.25",
        "--final-vp-loss-weight",
        "0.0",
        "--truncated-vp-margin-value-weight",
        "0.0",
        "--soft-target-weight",
        "0.9",
        "--training-rng-rank-offset",
        "--mask-hidden-info",
        "--no-resume-optimizer",
    ]


def _binding(tmp_path: Path) -> dict:
    root = tmp_path / "repo"
    files = {}
    for relative in arm.SOURCE_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\n", encoding="utf-8")
        files[relative] = {
            "path": str(path.resolve()),
            "sha256": "sha256:" + "7" * 64,
        }
    return {
        "repository_root": str(root.resolve()),
        "public_main_commit": "commit",
        "files": files,
    }


def test_command_is_exact_one_axis_independent_f7_8x512_derivation(
    tmp_path: Path,
) -> None:
    source = _source_command(tmp_path)
    binding = _binding(tmp_path)
    root = tmp_path / "value-stop"
    trainer = Path(binding["files"]["tools/train_bc.py"]["path"])

    command, source_recipe, effective = arm._derive_command(
        source,
        trainer=trainer,
        output_root=root,
        repo_binding=binding,
        reviewed_source_sha256="sha256:" + "8" * 64,
    )

    assert (
        command[: command.index(str(trainer))]
        == source[: source.index(str(tmp_path / "source" / "tools" / "train_bc.py"))]
    )
    assert arm.temperature.base._option(command, "--init-checkpoint") == str(  # noqa: SLF001
        tmp_path / "f7.pt"
    )
    assert arm.temperature.base._option(command, "--batch-size") == "512"  # noqa: SLF001
    assert arm.temperature.base._option(command, "--grad-accum-steps") == "1"  # noqa: SLF001
    assert arm.temperature.base._option(command, "--value-trunk-grad-scale") == "0.0"  # noqa: SLF001
    assert "value_trunk_grad_scale" not in source_recipe
    assert effective["value_trunk_grad_scale"] == pytest.approx(0.0)
    assert effective["world_size"] == 8
    assert effective["global_batch_size"] == 4096
    assert (
        arm.temperature.base._option(  # noqa: SLF001
            command, "--a1-effective-learner-recipe-sha256"
        )
        == arm.temperature.base._digest(effective)  # noqa: SLF001
    )


def test_derivation_refuses_stacked_ablation_authority(tmp_path: Path) -> None:
    source = _source_command(tmp_path)
    source.extend(("--a1-learner-ablation-id", "old-arm"))
    binding = _binding(tmp_path)
    with pytest.raises(arm.ValueTrunkArmError, match="already carries"):
        arm._derive_command(
            source,
            trainer=Path(binding["files"]["tools/train_bc.py"]["path"]),
            output_root=tmp_path / "value-stop",
            repo_binding=binding,
            reviewed_source_sha256="sha256:" + "8" * 64,
        )


def test_prepare_seals_geometry_delta_and_falsifier_without_launch_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_command = _source_command(tmp_path)
    source_manifest = {
        "manifest_sha256": "sha256:" + "a" * 64,
        "f7_parent": {
            "path": str(tmp_path / "f7.pt"),
            "sha256": arm.temperature.F7_SHA256,
        },
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
    binding = _binding(tmp_path)
    monkeypatch.setattr(arm.temperature, "verify", lambda _path: source)
    monkeypatch.setattr(arm.temperature, "_validate_recipe", lambda *_a, **_k: None)
    monkeypatch.setattr(arm, "_repo_binding", lambda _repo: binding)

    plan = arm.prepare(
        source_temperature_manifest=tmp_path / "temp.manifest.json",
        repo=tmp_path / "repo",
        output_root=tmp_path / "value-stop",
        manifest_path=tmp_path / "value-stop.plan.json",
    )

    assert plan["launch_authorized"] is False
    assert plan["diagnostic_only"] is True
    assert plan["promotion_eligible"] is False
    assert plan["only_declared_optimization_delta"] == {
        "value_trunk_grad_scale": {"source": 1.0, "treatment": 0.0}
    }
    assert plan["matched_contract"]["world_size"] == 8
    assert plan["matched_contract"]["per_rank_batch_size"] == 512
    assert plan["matched_contract"]["global_batch_size"] == 4096
    assert plan["matched_contract"]["global_samples"] == 4_194_304
    assert plan["predicted_falsifier"]["strength_test"]["games"] == 1200
    assert (
        plan["predicted_falsifier"]["strength_test"]["falsified_if"]
        == "superiority_pentanomial_sprt_decision_H0"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("world_size", 4),
        ("per_rank_batch_size", 1024),
        ("global_samples", 8_388_608),
        ("optimizer", "resumed_adam"),
    ],
)
def test_prepare_refuses_nonselected_source_geometry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    source_command = _source_command(tmp_path)
    dose = {
        "optimizer_steps": 1024,
        "world_size": 8,
        "per_rank_batch_size": 512,
        "global_samples": 4_194_304,
        "optimizer": "fresh_adam",
    }
    dose[field] = value
    source = {
        "manifest": {
            "manifest_sha256": "sha256:" + "a" * 64,
            "source_descriptor": {"path": "/descriptor"},
            "validation_sentinel": {"path": "/sentinel"},
            "f7_parent": {"path": "/f7"},
            "selected_dose": dose,
        },
        "manifest_ref": {"path": "/manifest", "sha256": "sha256:" + "f" * 64},
        "command": source_command,
        "output_root": tmp_path / "temp",
    }
    monkeypatch.setattr(arm.temperature, "verify", lambda _path: source)
    monkeypatch.setattr(arm.temperature, "_validate_recipe", lambda *_a, **_k: None)
    with pytest.raises(arm.ValueTrunkArmError, match="8x512"):
        arm.prepare(
            source_temperature_manifest=tmp_path / "manifest",
            repo=tmp_path,
            output_root=tmp_path / "out",
            manifest_path=tmp_path / "plan.json",
        )
