from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from tools.reanalysis_orchestrator import (
    build_remote_poll_command,
    infer_phase,
    parse_reanalysis_poll_stdout,
    pull_reanalysis_manifest,
)


def _manifest() -> dict:
    return {
        "branches": [
            {
                "box": "bx_a",
                "checkpoint": "runs/self_play/s7001_reanalysis.pt",
                "log": "runs/self_play/logs/s7001_reanalysis.log",
                "name": "s7001_reanalysis",
                "remote_dir": "/tmp/catan-zero-s6001_league_vrpo",
                "report": "runs/self_play/s7001_reanalysis.json",
            }
        ]
    }


def test_reanalysis_poll_command_tracks_processes_and_artifacts() -> None:
    branch = _manifest()["branches"][0]
    command = build_remote_poll_command(branch)

    assert "pgrep -af s7001_reanalysis" in command
    assert "runs/self_play/s7001_reanalysis.pt" in command
    assert "runs/self_play/s7001_reanalysis.json" in command
    assert "runs/self_play/s7001_reanalysis.jsonl" in command
    assert "tail -n 5 runs/self_play/logs/s7001_reanalysis.log" in command


def test_parse_reanalysis_poll_stdout_extracts_phase_artifacts_and_log() -> None:
    parsed = parse_reanalysis_poll_stdout(
        "123 sh -c pgrep -af s7001_reanalysis\n"
        "456 .venv/bin/python -u tools/generate_reanalysis.py --output runs/self_play/s7001_reanalysis.jsonl\n"
        "__ARTIFACTS__\n"
        "runs/self_play/s7001_reanalysis.jsonl 1000 7\n"
        "runs/self_play/s7001_reanalysis.iter0005.pt 281 0\n"
        "runs/self_play/s7001_reanalysis.json 200 12\n"
        "__LOG__\n"
        '{"reanalysis": {"game": 7}}\n'
    )

    assert parsed["running"] is True
    assert parsed["phase"] == "generate"
    assert parsed["jsonl"] == [
        {"file": "runs/self_play/s7001_reanalysis.jsonl", "bytes": 1000, "lines": 7}
    ]
    assert parsed["checkpoints"] == [
        {"file": "runs/self_play/s7001_reanalysis.iter0005.pt", "bytes": 281, "lines": 0}
    ]
    assert parsed["reports"] == [
        {"file": "runs/self_play/s7001_reanalysis.json", "bytes": 200, "lines": 12}
    ]
    assert parsed["log_tail"] == ['{"reanalysis": {"game": 7}}']


def test_parse_reanalysis_poll_ignores_shell_parent_train_text() -> None:
    parsed = parse_reanalysis_poll_stdout(
        "1 sh -c env PYTHONPATH=x .venv/bin/python -u tools/generate_reanalysis.py && "
        ".venv/bin/python -u tools/train_ppo.py --seed 1\n"
        "2 .venv/bin/python -u tools/generate_reanalysis.py --output x.jsonl\n"
        "__ARTIFACTS__\n"
        "__LOG__\n"
    )

    assert parsed["phase"] == "generate"
    assert parsed["processes"] == [
        "2 .venv/bin/python -u tools/generate_reanalysis.py --output x.jsonl"
    ]


def test_infer_phase_prefers_training_over_generation() -> None:
    assert infer_phase(["1 .venv/bin/python -u tools/generate_reanalysis.py"]) == "generate"
    assert infer_phase(["1 .venv/bin/python -u tools/train_ppo.py"]) == "train"
    assert (
        infer_phase(
            [
                "1 .venv/bin/python -u tools/generate_reanalysis.py",
                "2 .venv/bin/python -u tools/train_ppo.py",
            ]
        )
        == "train"
    )
    assert infer_phase([]) == "idle"


def test_pull_reanalysis_manifest_pulls_only_existing_artifacts(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_run(command, *, dry_run, check=True):
        calls.append(command)
        destination = Path(command[-1])
        destination.write_text("artifact", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    def fake_poll(_manifest):
        return [
            {
                "name": "s7001_reanalysis",
                "checkpoints": [
                    {"file": "runs/self_play/s7001_reanalysis.iter0005.pt", "bytes": 10, "lines": 0}
                ],
                "reports": [
                    {"file": "runs/self_play/s7001_reanalysis.json", "bytes": 10, "lines": 1}
                ],
                "jsonl": [
                    {"file": "runs/self_play/s7001_reanalysis.jsonl", "bytes": 10, "lines": 1}
                ],
            }
        ]

    monkeypatch.setattr("tools.reanalysis_orchestrator.run", fake_run)
    monkeypatch.setattr("tools.reanalysis_orchestrator.poll_reanalysis_manifest", fake_poll)

    pulled = pull_reanalysis_manifest(_manifest(), tmp_path, include_jsonl=False)

    assert pulled == {
        "checkpoints": [str(tmp_path / "s7001_reanalysis.iter0005.pt")],
        "reports": [str(tmp_path / "s7001_reanalysis.json")],
        "jsonl": [],
    }
    assert len(calls) == 2
    assert calls[0][0:2] == ["box", "scp"]
