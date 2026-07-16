from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from catan_zero.rl.production_loop import STAGES, execute, load_config, plan
from tools import loop


def test_public_loop_cli_has_three_options() -> None:
    parser = loop.build_parser()
    options = [
        action
        for action in parser._actions  # noqa: SLF001
        if action.option_strings and action.dest != "help"
    ]
    assert len(options) == 3


def test_config_binds_clean_checkout_and_connected_canonical_tools(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    tools_dir = repository / "tools"
    fleet_dir = tools_dir / "fleet"
    fleet_dir.mkdir(parents=True)
    tool_paths = {
        "generate": tools_dir / "generate.py",
        "harvest": fleet_dir / "a1_harvest_transaction.py",
        "audit": tools_dir / "a1_pre_wave_contract.py",
        "composite": tools_dir / "a1_build_post_wave_composite.py",
        "train": tools_dir / "train.py",
        "evaluate": tools_dir / "evaluate.py",
        "promote": tools_dir / "a1_promotion_transaction.py",
    }
    for tool in tool_paths.values():
        tool.write_text("# canonical fixture\n", encoding="utf-8")
    subprocess.run(("git", "init", "-q", str(repository)), check=True)
    subprocess.run(("git", "-C", str(repository), "add", "."), check=True)
    subprocess.run(
        (
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Loop Test",
            "-c",
            "user.email=loop@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ),
        check=True,
    )
    commit = subprocess.check_output(
        ("git", "-C", str(repository), "rev-parse", "HEAD"), text=True
    ).strip()
    state_dir = tmp_path / "state"
    previous = str(tmp_path / "initial.json")
    stages = {}
    for name in STAGES:
        output = str(tmp_path / f"{name}.json")
        command = [sys.executable, str(tool_paths[name])]
        if name in {"audit", "promote"}:
            command.append(name)
        if name == "promote":
            command.append("--go")
        stages[name] = {
            "command": command,
            "inputs": [previous],
            "outputs": [output],
            "timeout_seconds": 10,
        }
        previous = output
    config_path = tmp_path / "loop.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "catan-zero-production-loop-v1",
                "loop_id": "fixture",
                "repository": str(repository),
                "repository_commit": commit,
                "python": sys.executable,
                "require_clean_repository": True,
                "stages": stages,
            }
        ),
        encoding="utf-8",
    )

    loaded = load_config(config_path, state_dir=state_dir)
    assert loaded["repository_commit"] == commit
    assert tuple(loaded["stages"]) == STAGES


def test_loop_executes_once_and_resumes_from_hashed_receipts(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    writer = repository / "writer.py"
    writer.write_text(
        """from pathlib import Path
import sys
source, destination, stage = sys.argv[1:]
Path(destination).write_text(Path(source).read_text() + stage + "\\n")
""",
        encoding="utf-8",
    )
    initial = tmp_path / "initial.receipt.json"
    initial.write_text("start\n", encoding="utf-8")
    previous = initial
    stages = {}
    for name in STAGES:
        output = tmp_path / f"{name}.receipt.json"
        stages[name] = {
            "command": [sys.executable, str(writer), str(previous), str(output), name],
            "inputs": [str(previous)],
            "outputs": [str(output)],
            "timeout_seconds": 10,
        }
        previous = output
    config = {
        "loop_id": "unit-turn",
        "config_sha256": "sha256:unit",
        "repository_commit": "0" * 40,
        "repository": str(repository),
        "stages": stages,
    }
    state_dir = tmp_path / "state"
    state = execute(config, state_dir=state_dir)
    assert state["completed_stages"] == list(STAGES)
    assert previous.read_text(encoding="utf-8").splitlines() == ["start", *STAGES]

    resumed = execute(config, state_dir=state_dir)
    assert resumed == state
    dry_run = plan(config, state_dir=state_dir)
    assert dry_run["pending_stages"] == []
    persisted = json.loads((state_dir / "state.json").read_text(encoding="utf-8"))
    assert persisted == state
