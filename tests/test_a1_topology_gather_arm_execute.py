from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools import a1_topology_gather_arm as arm
from tools import a1_topology_gather_arm_execute as executor
from test_a1_topology_gather_arm import _args


def _write_manifest(path: Path, payload: dict) -> None:
    payload.pop("manifest_sha256", None)
    payload["manifest_sha256"] = arm.corrected._digest(payload)
    path.chmod(0o644)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict]:
    manifest, path = arm.prepare(_args(tmp_path, monkeypatch))
    repo = Path(arm.__file__).resolve().parents[1]
    executor_path = Path(executor.__file__).resolve()
    files = {
        relative: arm.corrected._file_ref(repo / relative)
        for relative in arm.SOURCE_FILES
    }
    manifest["source_binding"] = {
        "repository_root": str(repo),
        "git_commit": "topology-test-head",
        "files": files,
        "files_sha256": arm.corrected._digest(files),
    }
    manifest["diagnostic_executor"] = arm.corrected._file_ref(executor_path)
    _write_manifest(path, manifest)
    monkeypatch.setattr(
        executor.base,
        "_git_head",
        lambda _candidate: "topology-test-head",
    )
    return path, manifest


def test_direct_cli_help_works_outside_repository(tmp_path: Path) -> None:
    completed = subprocess.run(
        (sys.executable, str(Path(executor.__file__).resolve()), "--help"),
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--manifest" in completed.stdout


def test_verify_binds_executor_source_and_exact_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, manifest = _manifest(tmp_path, monkeypatch)
    verified = executor.verify(path)
    assert verified["command"] == manifest["command"]
    assert manifest["diagnostic_executor"]["path"] == str(Path(executor.__file__).resolve())
    assert arm.corrected._option(verified["command"], "--init-checkpoint") == (
        manifest["initialization_treatment"]["path"]
    )
    assert verified["repo"] == (tmp_path / "geometry-checkout").resolve()


def test_verify_preserves_bound_geometry_trainer_instead_of_current_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, payload = _manifest(tmp_path, monkeypatch)
    current_trainer = Path(arm.__file__).resolve().parents[1] / "tools/train_bc.py"
    trainer_index = next(
        index
        for index, value in enumerate(payload["command"])
        if Path(value).name == "train_bc.py"
    )
    payload["command"][trainer_index] = str(current_trainer)
    payload["command_sha256"] = arm.corrected._digest(payload["command"])
    _write_manifest(path, payload)
    with pytest.raises(executor.ExecutionError, match="bound selected-geometry trainer"):
        executor.verify(path)


def test_verify_refuses_bound_geometry_trainer_byte_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _payload = _manifest(tmp_path, monkeypatch)
    trainer = tmp_path / "geometry-checkout/tools/train_bc.py"
    trainer.write_text("# drifted after geometry plan\n", encoding="utf-8")
    with pytest.raises(executor.ExecutionError, match="selected geometry trainer checkout/bytes drifted"):
        executor.verify(path)


def test_verify_refuses_semantic_digest_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, payload = _manifest(tmp_path, monkeypatch)
    payload["only_declared_optimization_delta"] = "something-else"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(executor.ExecutionError, match="semantic digest drift"):
        executor.verify(path)


def test_verify_refuses_topology_manifest_without_crop_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, payload = _manifest(tmp_path, monkeypatch)
    payload["command"].remove(arm.corrected.EVENT_HISTORY_CROP_FLAG)
    payload["command_sha256"] = arm.corrected._digest(payload["command"])
    _write_manifest(path, payload)
    with pytest.raises(executor.ExecutionError, match="crop flag"):
        executor.verify(path)


def test_verify_refuses_topology_manifest_without_event_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, payload = _manifest(tmp_path, monkeypatch)
    payload.pop("event_history_training_contract")
    _write_manifest(path, payload)
    with pytest.raises(executor.ExecutionError, match="event-history contract drift"):
        executor.verify(path)


def test_verify_refuses_bound_source_bytes_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, payload = _manifest(tmp_path, monkeypatch)
    treatment = Path(payload["initialization_treatment"]["path"])
    with treatment.open("ab") as handle:
        handle.write(b"drift")
    with pytest.raises(executor.ExecutionError, match="initialization_treatment bytes drifted"):
        executor.verify(path)


def test_verify_refuses_rehashed_input_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, payload = _manifest(tmp_path, monkeypatch)
    replacement = tmp_path / "replacement.pt"
    replacement.write_bytes(b"replacement")
    index = payload["command"].index("--init-checkpoint") + 1
    payload["command"][index] = str(replacement)
    payload["command_sha256"] = arm.corrected._digest(payload["command"])
    _write_manifest(path, payload)
    with pytest.raises(executor.ExecutionError, match="bound --init-checkpoint"):
        executor.verify(path)


def test_verify_refuses_preexisting_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, payload = _manifest(tmp_path, monkeypatch)
    checkpoint = Path(arm.corrected._option(payload["command"], "--checkpoint"))
    checkpoint.write_bytes(b"existing")
    with pytest.raises(executor.ExecutionError, match="output/claim already exists"):
        executor.verify(path)
    assert executor.verify(path, require_fresh_outputs=False)["manifest"] == payload


def test_idle_probe_requires_exactly_eight_b200s_and_no_compute() -> None:
    calls = 0

    def idle_runner(command, **_kwargs):
        nonlocal calls
        calls += 1
        stdout = "\n".join(["NVIDIA B200"] * 8) if calls == 1 else ""
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    assert executor.base._probe_conflicting_compute(idle_runner) == []

    def wrong_topology(command, **_kwargs):
        return subprocess.CompletedProcess(
            command, 0, stdout="\n".join(["NVIDIA H100"] * 8), stderr=""
        )

    with pytest.raises(executor.ExecutionError, match="exactly eight visible B200s"):
        executor.base._probe_conflicting_compute(wrong_topology)


def test_execute_is_one_shot_and_submits_exact_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, payload = _manifest(tmp_path, monkeypatch)
    calls = []

    def runner(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="Running as unit.", stderr="")

    receipt = executor.execute(
        path,
        unit="a1-topology-gather-test",
        runner=runner,
        conflict_probe=lambda: [],
    )
    assert receipt["schema_version"] == executor.RECEIPT_SCHEMA
    assert len(calls) == 1
    separator = calls[0].index("--")
    assert calls[0][separator + 1 :] == payload["command"]
    output = Path(arm.corrected._option(payload["command"], "--checkpoint")).parent
    assert (output / "diagnostic-execution.claim.json").is_file()
    assert (output / "diagnostic-execution.receipt.json").is_file()
    with pytest.raises(executor.ExecutionError, match="already exists"):
        executor.execute(
            path,
            unit="a1-topology-gather-test",
            runner=runner,
            conflict_probe=lambda: [],
        )
    assert len(calls) == 1
