from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from tools import a1_selected_dose_symmetry_arm as arm
from test_a1_topology_gather_arm import _args as topology_args


def _args(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source = topology_args(tmp_path, monkeypatch)
    executor = tmp_path / arm.EXECUTOR_RELATIVE_PATH
    executor.parent.mkdir(exist_ok=True)
    executor.write_text("# symmetry executor\n", encoding="utf-8")
    ref = arm.bridge.corrected._file_ref(executor)  # noqa: SLF001
    trainer = tmp_path / "current-trainer" / "tools" / "train_bc.py"
    trainer.parent.mkdir(parents=True)
    trainer.write_text("# current trainer\n", encoding="utf-8")
    files = {
        arm.EXECUTOR_RELATIVE_PATH: ref,
        "tools/train_bc.py": arm.bridge.corrected._file_ref(trainer),  # noqa: SLF001
    }
    monkeypatch.setattr(
        arm,
        "_source_binding",
        lambda repo: {
            "repository_root": str(repo),
            "git_commit": "symmetry-test-head",
            "files": files,
            "files_sha256": arm.bridge.corrected._digest(files),  # noqa: SLF001
        },
    )
    return type(
        "Args",
        (),
        {
            "source_manifest": source.source_manifest,
            "selected_dose_plan": source.selected_dose_plan,
            "selected_dose_report": source.selected_dose_report,
            "output_root": tmp_path / "symmetry",
            "repo": tmp_path,
        },
    )()


def test_prepare_changes_only_symmetry_at_selected_dose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, path = arm.prepare(_args(tmp_path, monkeypatch))
    assert path.is_file()
    assert manifest["only_declared_causal_delta"] == {
        "symmetry_augment": {"source": False, "treatment": True},
        "symmetry_augment_events": {
            "source": False,
            "treatment": True,
            "conditional_on_symmetry": True,
        },
    }
    assert manifest["matched_contract"]["global_row_dose"] == 524_288
    assert manifest["matched_contract"]["optimizer_steps"] == 128
    command = manifest["command"]
    assert command.count("--symmetry-augment") == 1
    assert command.count("--symmetry-augment-events") == 1
    assert "--no-symmetry-augment" not in command
    assert "--no-symmetry-augment-events" not in command
    assert arm.bridge.corrected._option(command, "--batch-size") == "512"  # noqa: SLF001
    assert arm.bridge.corrected._option(command, "--max-steps") == "128"  # noqa: SLF001


def test_boolean_derivation_removes_explicit_negative(tmp_path: Path) -> None:
    source = [
        "python",
        "train_bc.py",
        "--max-steps",
        "128",
        "--batch-size",
        "512",
        "--grad-accum-steps",
        "1",
        "--lr",
        "3e-05",
        "--lr-warmup-steps",
        "100",
        "--soft-target-weight",
        "0.9",
        "--value-loss-weight",
        "0.25",
        "--no-symmetry-augment",
        "--no-symmetry-augment-events",
        "--checkpoint",
        "old.pt",
        "--report",
        "old.json",
    ]
    trainer = tmp_path / "current" / "train_bc.py"
    trainer.parent.mkdir()
    trainer.write_text("# current trainer\n", encoding="utf-8")
    command, changes = arm._derive_command(
        source, trainer=trainer, output_root=tmp_path / "out"
    )
    assert changes["--symmetry-augment"]["source"] == "--no-symmetry-augment"
    assert changes["--symmetry-augment-events"]["source"] == "--no-symmetry-augment-events"
    assert command[-2:] == ["--symmetry-augment", "--symmetry-augment-events"]


def test_derivation_rejects_hidden_dose_change(tmp_path: Path) -> None:
    source = [
        "python",
        "train_bc.py",
        "--max-steps",
        "1024",
        "--batch-size",
        "512",
        "--grad-accum-steps",
        "1",
        "--lr",
        "3e-05",
        "--lr-warmup-steps",
        "100",
        "--soft-target-weight",
        "0.9",
        "--value-loss-weight",
        "0.25",
        "--checkpoint",
        "old.pt",
        "--report",
        "old.json",
    ]
    with pytest.raises(arm.SymmetryArmError, match="524,288-row"):
        trainer = tmp_path / "train_bc.py"
        trainer.write_text("# current trainer\n", encoding="utf-8")
        arm._derive_command(source, trainer=trainer, output_root=tmp_path / "bad")


def test_verify_replays_source_and_trainer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, path = arm.prepare(_args(tmp_path, monkeypatch))
    repo = Path(arm.__file__).resolve().parents[1]
    files = {
        relative: arm.bridge.corrected._file_ref(repo / relative)  # noqa: SLF001
        for relative in arm.SOURCE_FILES
    }
    manifest["source_binding"] = {
        "repository_root": str(repo),
        "git_commit": "symmetry-test-head",
        "files": files,
        "files_sha256": arm.bridge.corrected._digest(files),  # noqa: SLF001
    }
    manifest["diagnostic_executor"] = files[arm.EXECUTOR_RELATIVE_PATH]
    current_trainer = str((repo / "tools/train_bc.py").resolve())
    trainer_index = next(
        index
        for index, value in enumerate(manifest["command"])
        if Path(value).name == "train_bc.py"
    )
    manifest["command"][trainer_index] = current_trainer
    manifest["allowlisted_command_changes"]["trainer"]["treatment"] = current_trainer
    manifest["runtime_contract_delta"]["current_trainer"] = files["tools/train_bc.py"]
    manifest["command_sha256"] = arm.bridge.corrected._digest(  # noqa: SLF001
        manifest["command"]
    )
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = arm.bridge.corrected._digest(manifest)  # noqa: SLF001
    path.chmod(0o644)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(arm.executor_base, "_git_head", lambda _repo: "symmetry-test-head")
    verified = arm.verify(path)
    assert verified["command"] == manifest["command"]
    assert verified["repo"] == repo


def test_execute_submits_only_verified_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = {
        "manifest": {"command_sha256": "sha256:" + "1" * 64},
        "manifest_ref": {"path": "/manifest", "sha256": "sha256:" + "2" * 64},
        "repo": tmp_path,
        "command": ["python", "train_bc.py"],
        "output_root": tmp_path / "out",
    }
    monkeypatch.setattr(arm, "verify", lambda _path: verified)
    calls: list[list[str]] = []

    def runner(command, **_kwargs):
        calls.append(list(command))
        return subprocess.CompletedProcess(command, 0, stdout="Running as unit.", stderr="")

    receipt = arm.execute(
        tmp_path / "manifest.json",
        unit="a1-symmetry-test",
        runner=runner,
        conflict_probe=lambda: [],
    )
    assert receipt["schema_version"] == arm.RECEIPT_SCHEMA
    assert calls[0][calls[0].index("--") + 1 :] == verified["command"]
