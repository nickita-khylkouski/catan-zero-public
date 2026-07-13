from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from tools import a1_selected_dose_pure_soft_arm as arm
from test_a1_topology_gather_arm import _args as topology_args


def _args(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source = topology_args(tmp_path, monkeypatch)
    executor = tmp_path / arm.EXECUTOR_RELATIVE_PATH
    executor.parent.mkdir(exist_ok=True)
    executor.write_text("# pure-soft executor\n", encoding="utf-8")
    monkeypatch.setattr(arm, "_source_binding", lambda repo: {
        "repository_root": str(repo),
        "git_commit": "pure-soft-test-head",
        "files": {
            arm.EXECUTOR_RELATIVE_PATH: arm.bridge.corrected._file_ref(executor),
        },
        "files_sha256": arm.bridge.corrected._digest({
            arm.EXECUTOR_RELATIVE_PATH: arm.bridge.corrected._file_ref(executor),
        }),
    })
    return type("Args", (), {
        "source_manifest": source.source_manifest,
        "selected_dose_plan": source.selected_dose_plan,
        "selected_dose_report": source.selected_dose_report,
        "output_root": tmp_path / "pure-soft",
        "repo": tmp_path,
    })()


def test_prepare_is_exact_selected_dose_pure_soft_axis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, path = arm.prepare(_args(tmp_path, monkeypatch))
    assert path.is_file()
    assert manifest["launch_authorized"] is False
    assert manifest["diagnostic_execution_authorized"] is True
    assert manifest["only_declared_causal_delta"] == {
        "soft_target_weight": {"source": 0.9, "treatment": 1.0},
        "played_action_hard_ce_weight": {"source": 0.1, "treatment": 0.0},
    }
    assert manifest["matched_contract"]["global_row_dose"] == 524_288
    assert manifest["matched_contract"]["optimizer_steps"] == 128
    assert manifest["matched_contract"]["candidate_chaining"] is False
    command = manifest["command"]
    assert arm.bridge.corrected._option(command, "--soft-target-weight") == "1.0"  # noqa: SLF001
    assert arm.bridge.corrected._option(command, "--max-steps") == "128"  # noqa: SLF001
    assert arm.bridge.corrected._option(command, "--batch-size") == "512"  # noqa: SLF001
    assert arm.bridge.corrected._option(command, "--lr") == "3e-05"  # noqa: SLF001


def test_verify_replays_bridge_and_complete_source_binding(
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
        "git_commit": "pure-soft-test-head",
        "files": files,
        "files_sha256": arm.bridge.corrected._digest(files),  # noqa: SLF001
    }
    manifest["diagnostic_executor"] = files[arm.EXECUTOR_RELATIVE_PATH]
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = arm.bridge.corrected._digest(manifest)  # noqa: SLF001
    path.chmod(0o644)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        arm.executor_base, "_git_head", lambda _repo: "pure-soft-test-head"
    )
    verified = arm.verify(path)
    assert verified["command"] == manifest["command"]
    assert verified["repo"] == (tmp_path / "geometry-checkout").resolve()


def test_derivation_refuses_nonselected_source_or_hidden_axis(tmp_path: Path) -> None:
    source = [
        "python", "train_bc.py", "--soft-target-weight", "0.9",
        "--max-steps", "128", "--batch-size", "512", "--lr", "3e-05",
        "--lr-warmup-steps", "100", "--checkpoint", "old.pt",
        "--report", "old.json",
    ]
    command, _ = arm._derive_command(source, output_root=tmp_path / "out")
    assert arm.bridge.corrected._option(command, "--soft-target-weight") == "1.0"  # noqa: SLF001
    bad = list(source)
    bad[bad.index("--max-steps") + 1] = "1024"
    with pytest.raises(arm.PureSoftError, match="selected 524,288-row"):
        arm._derive_command(bad, output_root=tmp_path / "bad")


def test_prepare_refuses_geometry_objective_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path, monkeypatch)
    plan = json.loads(args.selected_dose_plan.read_text())
    command = plan["runs"][0]["command"]
    command[command.index("--value-loss-weight") + 1] = "1.0"
    plan["runs"][0]["command_sha256"] = arm.bridge.corrected._digest(command)  # noqa: SLF001
    plan["plan_sha256"] = arm.bridge.corrected._digest(  # noqa: SLF001
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )
    args.selected_dose_plan.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(arm.bridge.ArmError, match="exact short-dose TEMP derivation"):
        arm.prepare(args)


def test_execute_submits_only_verified_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "out"
    verified = {
        "manifest": {"command_sha256": "sha256:" + "1" * 64},
        "manifest_ref": {"path": "/manifest", "sha256": "sha256:" + "2" * 64},
        "repo": tmp_path,
        "command": ["python", "train_bc.py"],
        "output_root": root,
    }
    monkeypatch.setattr(arm, "verify", lambda _path: verified)
    calls: list[list[str]] = []

    def runner(command, **_kwargs):
        calls.append(list(command))
        return subprocess.CompletedProcess(command, 0, stdout="Running as unit.", stderr="")

    receipt = arm.execute(
        tmp_path / "manifest.json",
        unit="a1-pure-soft-test",
        runner=runner,
        conflict_probe=lambda: [],
    )
    assert receipt["schema_version"] == arm.RECEIPT_SCHEMA
    assert len(calls) == 1
    assert calls[0][calls[0].index("--") + 1 :] == verified["command"]


def test_cli_help_names_selected_evidence(tmp_path: Path) -> None:
    result = subprocess.run(
        ["python3", str(Path(arm.__file__).resolve()), "prepare", "--help"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--selected-dose-plan" in result.stdout
    assert "--selected-dose-report" in result.stdout
