from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from tools import a1_experimental_training_sequence as sequence


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _config(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    root.mkdir()
    (root / sequence.READY_MARKER).write_text("ready\n")
    python = tmp_path / "python"
    executor = tmp_path / "executor.py"
    python.write_text("python")
    python.chmod(0o755)
    executor.write_text("executor")
    rows = []
    for arm in sequence.ARMS:
        data = root / f"{arm}.memmap"
        data.mkdir()
        validation = root / f"{arm}.validation.json"
        producer = root / "producer.pt"
        lock = root / f"{arm}.learner.lock.json"
        validation.write_text("{}")
        producer.write_text("model")
        lock.write_text("{}")
        rows.append(
            {
                "arm_id": arm,
                "data": str(data),
                "validation_manifest": str(validation),
                "producer_checkpoint": str(producer),
                "learner_lock": str(lock),
                "reviewed_lock_file_sha256": _sha(lock),
                "checkpoint": str(root / "runs" / arm / "candidate.pt"),
                "report": str(root / "runs" / arm / "report.json"),
                "receipt": str(root / "runs" / arm / "receipt.json"),
            }
        )
    value = {
        "schema_version": sequence.SCHEMA,
        "root": str(root),
        "python": str(python),
        "executor": str(executor),
        "mps_unit": sequence.MPS_UNIT,
        "arms": rows,
    }
    path = tmp_path / "sequence.json"
    path.write_text(json.dumps(value))
    return path


def test_plan_requires_ready_and_reviewed_lock_bytes(tmp_path: Path) -> None:
    config = _config(tmp_path)
    plan = sequence.build_plan(config)
    assert plan["execution_order"] == ["n128", "n256"]
    Path(json.loads(config.read_text())["arms"][0]["learner_lock"]).write_text("drift")
    with pytest.raises(sequence.SequenceError, match="reviewed SHA-256"):
        sequence.build_plan(config)


def test_plan_preserves_lexical_virtualenv_python_symlink(tmp_path: Path) -> None:
    """Do not resolve venv/bin/python to a dependency-free base interpreter."""
    config = _config(tmp_path)
    value = json.loads(config.read_text())
    base = tmp_path / "base-python-without-torch"
    base.write_text("#!/bin/sh\nexit 1\n")
    base.chmod(0o755)
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    lexical = venv_bin / "python"
    lexical.symlink_to(base)
    value["python"] = str(lexical)
    config.write_text(json.dumps(value))

    plan = sequence.build_plan(config)

    assert Path(plan["commands"][0]["argv"][0]) == lexical.absolute()
    assert Path(plan["commands"][0]["argv"][0]) != base.resolve()


def test_plan_refuses_missing_ready_marker(tmp_path: Path) -> None:
    config = _config(tmp_path)
    root = Path(json.loads(config.read_text())["root"])
    (root / sequence.READY_MARKER).unlink()
    with pytest.raises(sequence.SequenceError, match="readiness marker"):
        sequence.build_plan(config)


class _Runner:
    def __init__(self, root: Path, fail_arm: str | None = None):
        self.root = root
        self.fail_arm = fail_arm
        self.active = True
        self.calls: list[list[str]] = []

    def __call__(self, argv, **_kwargs):
        argv = list(map(str, argv))
        self.calls.append(argv)
        if argv[:2] == ["systemctl", "is-active"]:
            return subprocess.CompletedProcess(argv, 0, "active\n" if self.active else "inactive\n", "")
        if "systemctl" in argv and "stop" in argv:
            self.active = False
            return subprocess.CompletedProcess(argv, 0)
        if "systemctl" in argv and "start" in argv:
            self.active = True
            return subprocess.CompletedProcess(argv, 0)
        arm = "n128" if "n128" in " ".join(argv) else "n256"
        if argv[-1] == "--go":
            if arm == self.fail_arm:
                return subprocess.CompletedProcess(argv, 1)
            config = json.loads((self.root.parent / "sequence.json").read_text())
            row = next(row for row in config["arms"] if row["arm_id"] == arm)
            receipt = Path(row["receipt"])
            receipt.parent.mkdir(parents=True, exist_ok=True)
            receipt.write_text("{}")
        return subprocess.CompletedProcess(argv, 0)


def test_execute_is_sequential_and_restores_mps(tmp_path: Path) -> None:
    config = _config(tmp_path)
    plan = sequence.build_plan(config)
    runner = _Runner(Path(plan["root"]))
    sequence.execute(plan, runner=runner)
    go = [call for call in runner.calls if call[-1:] == ["--go"]]
    assert "n128" in " ".join(go[0]) and "n256" in " ".join(go[1])
    assert runner.active is True


def test_execute_restores_mps_after_training_failure(tmp_path: Path) -> None:
    config = _config(tmp_path)
    plan = sequence.build_plan(config)
    runner = _Runner(Path(plan["root"]), fail_arm="n128")
    with pytest.raises(sequence.SequenceError, match="n128 sealed training failed"):
        sequence.execute(plan, runner=runner)
    assert runner.active is True
    assert not any("n256" in " ".join(call) for call in runner.calls if call[-1:] == ["--go"])
