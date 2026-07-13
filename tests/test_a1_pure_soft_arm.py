from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from tools import a1_pure_soft_arm as arm


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
        "--data",
        str(tmp_path / "descriptor.json"),
        "--init-checkpoint",
        str(tmp_path / "f7.pt"),
        "--checkpoint",
        str(tmp_path / "temp" / "candidate.pt"),
        "--report",
        str(tmp_path / "temp" / "train.report.json"),
        "--lr",
        "3e-05",
        "--max-steps",
        "1024",
        "--soft-target-weight",
        "0.9",
        "--soft-target-temperature",
        "0.7",
        "--policy-kl-anchor-weight",
        "0.0",
        "--training-rng-rank-offset",
        "--mask-hidden-info",
    ]


def _trainer(tmp_path: Path) -> Path:
    path = tmp_path / "treatment" / "tools" / "train_bc.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# treatment\n", encoding="utf-8")
    return path


def test_command_changes_only_soft_blend_trainer_and_outputs(tmp_path: Path) -> None:
    source = _source_command(tmp_path)
    trainer = _trainer(tmp_path)
    root = tmp_path / "pure-soft"

    treatment = arm._derive_command(source, trainer=trainer, output_root=root)

    expected = list(source)
    expected[expected.index(str(tmp_path / "source" / "tools" / "train_bc.py"))] = str(
        trainer.resolve()
    )
    expected[expected.index("--soft-target-weight") + 1] = "1.0"
    expected[expected.index("--checkpoint") + 1] = str(root / "candidate.pt")
    expected[expected.index("--report") + 1] = str(root / "train.report.json")
    assert treatment == expected
    assert arm.temperature.base._option(treatment, "--init-checkpoint") == str(  # noqa: SLF001
        tmp_path / "f7.pt"
    )
    assert arm.temperature.base._option(treatment, "--soft-target-weight") == "1.0"  # noqa: SLF001


@pytest.mark.parametrize(
    "mutation",
    [
        ("value", "--lr", "0.0001"),
        ("value", "--max-steps", "2048"),
        ("value", "--soft-target-temperature", "1.0"),
        ("value", "--policy-kl-anchor-weight", "0.006"),
        ("remove", "--training-rng-rank-offset", None),
        ("remove", "--mask-hidden-info", None),
        ("append", "--symmetry-augment", None),
        ("append-valued", "--loser-sample-weight", "0.3"),
    ],
)
def test_exact_derivation_rejects_every_other_causal_drift(
    tmp_path: Path, mutation: tuple[str, str, str | None]
) -> None:
    source = _source_command(tmp_path)
    trainer = _trainer(tmp_path)
    root = tmp_path / "pure-soft"
    treatment = arm._derive_command(source, trainer=trainer, output_root=root)
    kind, flag, value = mutation
    if kind == "value":
        treatment[treatment.index(flag) + 1] = str(value)
    elif kind == "remove":
        treatment.remove(flag)
    elif kind == "append":
        treatment.append(flag)
    else:
        treatment.extend((flag, str(value)))

    with pytest.raises(arm.PureSoftArmError, match="exact one-axis"):
        arm._verify_exact_derivation(
            source, treatment, trainer=trainer, output_root=root
        )


def test_source_must_be_exact_winning_point_nine_recipe(tmp_path: Path) -> None:
    source = _source_command(tmp_path)
    source[source.index("--soft-target-weight") + 1] = "1.0"
    with pytest.raises(arm.PureSoftArmError, match="winning 0.9-soft"):
        arm._derive_command(
            source,
            trainer=_trainer(tmp_path),
            output_root=tmp_path / "pure-soft",
        )


def test_prepare_inherits_temp_contract_and_declares_only_blend_axis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_command = _source_command(tmp_path)
    trainer = _trainer(tmp_path)
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
                trainer if relative == "tools/train_bc.py" else tmp_path / relative
            ),
            "sha256": "sha256:" + "7" * 64,
        }
        for relative in arm.SOURCE_FILES
    }
    binding = {
        "repository_root": str(tmp_path / "treatment"),
        "public_main_commit": "commit",
        "files": files,
    }
    monkeypatch.setattr(arm.temperature, "verify", lambda _path: source)
    monkeypatch.setattr(arm.temperature, "_validate_recipe", lambda *_a, **_k: None)
    monkeypatch.setattr(arm, "_repo_binding", lambda _repo: binding)

    manifest = arm.prepare(
        source_temperature_manifest=tmp_path / "temp.manifest.json",
        repo=tmp_path / "treatment",
        output_root=tmp_path / "pure-soft",
        manifest_path=tmp_path / "pure-soft.manifest.json",
    )

    assert manifest["f7_parent"] == source_manifest["f7_parent"]
    assert manifest["source_descriptor"] == source_manifest["source_descriptor"]
    assert manifest["selected_dose"] == source_manifest["selected_dose"]
    assert (
        manifest["stored_policy_component_temperatures"]
        == source_manifest["stored_policy_component_temperatures"]
    )
    assert manifest["only_declared_optimization_delta"] == {
        "soft_target_weight": {"source": 0.9, "treatment": 1.0},
        "played_action_hard_ce_weight": {"source": 0.1, "treatment": 0.0},
    }
    assert manifest["diagnostic_only"] is True
    assert manifest["promotion_eligible"] is False
    assert manifest["matched_contract"]["fresh_adam"] is True
    assert manifest["matched_contract"]["candidate_chaining"] is False
    assert (
        arm.temperature.base._option(  # noqa: SLF001
            manifest["command"], "--soft-target-weight"
        )
        == "1.0"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("soft_target_weight", 0.9),
        ("lr", 1.2e-4),
        ("base_training_row_draws", 8_388_608),
        ("init_checkpoint_sha256", "sha256:" + "0" * 64),
        ("mask_hidden_info", False),
        ("training_rng_rank_offset", False),
        ("policy_kl_anchor_weight", 0.006),
        ("value_loss_weight", 1.0),
        ("forced_action_weight", 1.0),
        ("symmetry_augment", True),
    ],
)
def test_completed_report_rejects_any_non_pure_soft_recipe(
    field: str, value: object
) -> None:
    report = dict(arm.temperature.SEALED_REPORT_RECIPE)
    report["soft_target_weight"] = 1.0
    report[field] = value
    assert arm._pure_soft_report_drift(report) == {
        field: {
            "expected": (
                1.0
                if field == "soft_target_weight"
                else arm.temperature.SEALED_REPORT_RECIPE[field]
            ),
            "actual": value,
        }
    }


def test_legacy_full_dose_execute_is_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "pure-soft"
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

    monkeypatch.setattr(arm, "verify", lambda _path: verified)
    with pytest.raises(arm.PureSoftArmError, match="4,194,304 rows / 1024 steps"):
        arm.execute(
            tmp_path / "manifest.json",
            unit="pure-soft-test",
            runner=runner,
            idle_probe=lambda: [],
        )
    assert submitted == []
    assert not root.exists()


def test_legacy_full_dose_refusal_precedes_idle_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        arm,
        "verify",
        lambda _path: {
            "manifest": {"command_sha256": "sha256:" + "9" * 64},
            "manifest_ref": {"path": "/manifest", "sha256": "sha256:" + "8" * 64},
            "repo": tmp_path,
            "command": ["python", "trainer.py"],
            "output_root": tmp_path / "pure-soft",
        },
    )
    called = False

    def idle_probe():
        nonlocal called
        called = True
        return ["gpu0:python"]

    with pytest.raises(arm.PureSoftArmError, match="4,194,304 rows / 1024 steps"):
        arm.execute(
            tmp_path / "manifest.json",
            unit="pure-soft-test",
            idle_probe=idle_probe,
        )
    assert called is False
