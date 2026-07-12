from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from tools import a1_corrected_policy_arm as arm
from tools import a1_corrected_policy_arm_execute as executor
from test_a1_corrected_policy_arm import _args


def _manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict]:
    args = _args(tmp_path, monkeypatch)
    trainer = tmp_path / "tools" / "train_bc.py"
    trainer.parent.mkdir(exist_ok=True)
    trainer.write_text("# exact trainer\n", encoding="utf-8")
    manifest, path = arm.prepare(args)
    manifest["source_binding"] = {
        "repository_root": str(tmp_path),
        "git_commit": "abc123",
        "files": {"tools/train_bc.py": arm._file_ref(trainer)},
    }
    trainer_index = next(
        index for index, value in enumerate(manifest["command"])
        if Path(value).name == "train_bc.py"
    )
    manifest["command"][trainer_index] = str(trainer)
    manifest["command_sha256"] = arm._digest(manifest["command"])
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = arm._digest(manifest)
    path.chmod(0o644)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    monkeypatch.setattr(executor, "_git_head", lambda repo: "abc123")
    return path, manifest


def test_verify_replays_manifest_inputs_and_real_sentinel_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _manifest_payload = _manifest(tmp_path, monkeypatch)
    verified = executor.verify(path)
    assert "--validation-game-sentinel-manifest" in verified["command"]
    assert "--validation-game-seed-manifest" not in verified["command"]


def test_verify_refuses_manifest_semantic_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, payload = _manifest(tmp_path, monkeypatch)
    payload["recipe"]["replay_forward_kl_weight"] = 999
    path.write_text(json.dumps(payload))
    with pytest.raises(executor.ExecutionError, match="semantic digest drift"):
        executor.verify(path)


def test_explicit_execute_submits_exact_command_and_writes_append_only_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, payload = _manifest(tmp_path, monkeypatch)
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="Running as unit.", stderr="")

    receipt = executor.execute(
        path, unit="a1-corrected-anchor-k3-test", runner=runner,
        conflict_probe=lambda: [],
    )
    assert receipt["diagnostic_only"] is True
    assert len(calls) == 1
    submitted = calls[0]
    assert submitted[:3] == ["sudo", "-n", "systemd-run"]
    assert "--property=LimitNOFILE=65536" in submitted
    separator = submitted.index("--")
    assert submitted[separator + 1 :] == payload["command"]
    out = Path(arm._option(payload["command"], "--checkpoint")).parent
    assert (out / "diagnostic-execution.claim.json").is_file()
    assert (out / "diagnostic-execution.receipt.json").is_file()
    events = [json.loads(line)["event"] for line in (
        out / "diagnostic-execution.status.jsonl"
    ).read_text().splitlines()]
    assert events == ["authorized", "submitted"]
    with pytest.raises(executor.ExecutionError, match="already exists"):
        executor.execute(
            path, unit="a1-corrected-anchor-k3-test", runner=runner,
            conflict_probe=lambda: [],
        )


def test_execute_refuses_conflicting_b200_compute_without_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, payload = _manifest(tmp_path, monkeypatch)
    with pytest.raises(executor.ExecutionError, match="not idle"):
        executor.execute(
            path, unit="a1-corrected-anchor-k3-test",
            conflict_probe=lambda: ["1234, python"],
        )
    out = Path(arm._option(payload["command"], "--checkpoint")).parent
    assert not (out / "diagnostic-execution.claim.json").exists()
