from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

from catan_zero.rl import production_loop
from catan_zero.rl.production_loop import (
    ProductionLoopError,
    STAGES,
    execute,
    load_config,
    plan,
)
from tools import loop


WRITER = r'''from pathlib import Path
import sys

args = sys.argv[1:]
source = None
for flag in ("--source", "--executor-receipt", "--harvest-relocation", "--post-wave-audit", "--data", "--candidate", "--adjudication"):
    if flag in args:
        source = Path(args[args.index(flag) + 1])
        break
outputs = []
for flag in ("--emit", "--execution-receipt", "--checkpoint", "--report", "--out", "--receipt"):
    if flag in args:
        outputs.append(Path(args[args.index(flag) + 1]))
assert outputs
prefix = "" if source is None else source.read_text()
for output in outputs:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(prefix + Path(sys.argv[0]).name + "\n")
'''


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, object]]:
    repository = tmp_path / "repo"
    tools_dir = repository / "tools"
    fleet_dir = tools_dir / "fleet"
    fleet_dir.mkdir(parents=True)
    tool_paths = {
        "generate": fleet_dir / "a1_production_executor.py",
        "harvest": fleet_dir / "a1_harvest_transaction.py",
        "audit": tools_dir / "a1_pre_wave_contract.py",
        "composite": tools_dir / "a1_build_post_wave_composite.py",
        "train": tools_dir / "train.py",
        "evaluate": tools_dir / "evaluate.py",
        "promote": tools_dir / "a1_promotion_transaction.py",
    }
    for tool in tool_paths.values():
        tool.write_text(WRITER, encoding="utf-8")
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
    initial = tmp_path / "initial.json"
    initial.write_text("start\n", encoding="utf-8")
    previous = initial
    stages: dict[str, dict[str, object]] = {}
    for name in STAGES:
        output = (
            tmp_path / "harvest" / "relocation_map.json"
            if name == "harvest"
            else tmp_path / f"{name}.json"
        )
        command = [sys.executable, str(tool_paths[name])]
        if name == "generate":
            command += [
                "run",
                "--go",
                "--lock",
                str(previous),
                "--receipt",
                str(output),
            ]
        elif name == "harvest":
            command += [
                "--executor-receipt",
                str(previous),
                "--destination",
                str(output.parent),
                "--emit",
                str(output),
            ]
        elif name == "audit":
            command += [
                "audit",
                "--harvest-relocation",
                str(previous),
                "--out",
                str(output),
            ]
        elif name == "composite":
            command += ["--post-wave-audit", str(previous), "--out", str(output)]
        elif name == "train":
            training_receipt = tmp_path / "training-receipt.json"
            command += [
                "--data",
                str(previous),
                "--checkpoint",
                str(output),
                "--report",
                str(training_receipt),
            ]
        elif name == "evaluate":
            command += ["--candidate", str(previous), "--out", str(output)]
        else:
            command += [
                "promote",
                "--go",
                "--adjudication",
                str(previous),
                "--training-receipt",
                str(training_receipt),
                "--receipt",
                str(output),
            ]
        outputs = [str(output)]
        if name == "train":
            outputs.append(str(training_receipt))
        stages[name] = {
            "command": command,
            "inputs": [str(previous)],
            "outputs": outputs,
            "timeout_seconds": 10,
        }
        if name == "promote":
            stages[name]["inputs"].append(str(training_receipt))  # type: ignore[union-attr]
        previous = output
    payload: dict[str, object] = {
        "schema_version": "catan-zero-production-loop-v1",
        "loop_id": "fixture",
        "repository": str(repository),
        "repository_commit": commit,
        "python": sys.executable,
        "require_clean_repository": True,
        "stages": stages,
    }
    config_path = tmp_path / "loop.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return repository, config_path, previous, payload


def test_public_loop_cli_has_three_options() -> None:
    parser = loop.build_parser()
    options = [
        action
        for action in parser._actions  # noqa: SLF001
        if action.option_strings and action.dest != "help"
    ]
    assert len(options) == 3


def test_loop_binds_and_executes_exact_artifact_chain(tmp_path: Path) -> None:
    _repository, config_path, final_output, _payload = _fixture(tmp_path)
    state_dir = tmp_path / "state"
    loaded = load_config(config_path, state_dir=state_dir)

    assert loaded["repository_commit"]
    assert loaded["stages"]["train"]["artifact_bindings"][0] == {
        "kind": "training_data",
        "direction": "input",
        "flag": "--data",
        "path": str((tmp_path / "composite.json").resolve()),
    }
    state = execute(loaded, state_dir=state_dir)
    assert state["completed_stages"] == list(STAGES)
    assert final_output.read_text(encoding="utf-8").splitlines() == [
        "a1_production_executor.py",
        "a1_harvest_transaction.py",
        "a1_pre_wave_contract.py",
        "a1_build_post_wave_composite.py",
        "train.py",
        "evaluate.py",
        "a1_promotion_transaction.py",
    ]
    assert execute(loaded, state_dir=state_dir) == state
    assert plan(loaded, state_dir=state_dir)["pending_stages"] == []


def test_train_data_must_be_exact_immediate_composite_output(tmp_path: Path) -> None:
    _repository, config_path, _final, payload = _fixture(tmp_path)
    stages = payload["stages"]
    assert isinstance(stages, dict)
    train = stages["train"]
    assert isinstance(train, dict)
    command = train["command"]
    assert isinstance(command, list)
    command[command.index("--data") + 1] = str(tmp_path / "initial.json")
    train["inputs"] = [str(tmp_path / "initial.json")]
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProductionLoopError, match="exact immediate predecessor"):
        load_config(config_path, state_dir=tmp_path / "state")


def test_exact_repo_relative_tool_and_untracked_cleanliness(tmp_path: Path) -> None:
    repository, config_path, _final, payload = _fixture(tmp_path)
    collision = repository / "other" / "train.py"
    collision.parent.mkdir()
    collision.write_text(WRITER, encoding="utf-8")
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
            "collision",
        ),
        check=True,
    )
    payload["repository_commit"] = subprocess.check_output(
        ("git", "-C", str(repository), "rev-parse", "HEAD"), text=True
    ).strip()
    stages = payload["stages"]
    assert isinstance(stages, dict) and isinstance(stages["train"], dict)
    stages["train"]["command"][1] = str(collision)  # type: ignore[index]
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ProductionLoopError, match="cannot invoke 'other/train.py'"):
        load_config(config_path, state_dir=tmp_path / "state")

    stages["train"]["command"][1] = str(repository / "tools" / "train.py")  # type: ignore[index]
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    (repository / "untracked.txt").write_text("drift", encoding="utf-8")
    with pytest.raises(ProductionLoopError, match="including untracked files"):
        load_config(config_path, state_dir=tmp_path / "state")


def test_scratch_requires_go_and_fresh_execution_receipt(tmp_path: Path) -> None:
    repository, config_path, _final, payload = _fixture(tmp_path)
    scratch = repository / "tools" / "a1_scratch_train.py"
    scratch.write_text(WRITER, encoding="utf-8")
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
            "scratch",
        ),
        check=True,
    )
    payload["repository_commit"] = subprocess.check_output(
        ("git", "-C", str(repository), "rev-parse", "HEAD"), text=True
    ).strip()
    stages = payload["stages"]
    assert isinstance(stages, dict) and isinstance(stages["train"], dict)
    train = stages["train"]
    train["command"] = [
        sys.executable,
        str(scratch),
        "--data",
        str(tmp_path / "composite.json"),
        "--checkpoint",
        str(tmp_path / "train.json"),
    ]
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ProductionLoopError, match="execute with --go"):
        load_config(config_path, state_dir=tmp_path / "state")


def test_scratch_plan_only_or_invalid_execution_receipt_cannot_advance(
    tmp_path: Path,
) -> None:
    repository, config_path, _final, payload = _fixture(tmp_path)
    scratch = repository / "tools" / "a1_scratch_train.py"
    scratch.write_text(WRITER, encoding="utf-8")
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
            "scratch executor",
        ),
        check=True,
    )
    payload["repository_commit"] = subprocess.check_output(
        ("git", "-C", str(repository), "rev-parse", "HEAD"), text=True
    ).strip()
    plan_receipt = tmp_path / "scratch-plan.json"
    plan_receipt.write_text("{}", encoding="utf-8")
    execution_receipt = tmp_path / "scratch-execution.json"
    stages = payload["stages"]
    assert isinstance(stages, dict) and isinstance(stages["train"], dict)
    train = stages["train"]
    train["command"] = [
        sys.executable,
        str(scratch),
        "--data",
        str(tmp_path / "composite.json"),
        "--checkpoint",
        str(tmp_path / "train.json"),
        "--receipt",
        str(plan_receipt),
        "--execution-receipt",
        str(execution_receipt),
        "--go",
    ]
    train["inputs"] = [str(tmp_path / "composite.json"), str(plan_receipt)]
    train["outputs"] = [str(tmp_path / "train.json"), str(execution_receipt)]
    promote = stages["promote"]
    assert isinstance(promote, dict) and isinstance(promote["command"], list)
    promotion_command = promote["command"]
    promotion_command[promotion_command.index("--training-receipt") + 1] = str(
        execution_receipt
    )
    promote["inputs"] = [str(tmp_path / "evaluate.json"), str(execution_receipt)]
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_config(config_path, state_dir=tmp_path / "state")

    with pytest.raises(ProductionLoopError, match="readable execution receipt"):
        execute(loaded, state_dir=tmp_path / "state")
    state = json.loads((tmp_path / "state" / "state.json").read_text())
    assert state["completed_stages"] == ["generate", "harvest", "audit", "composite"]


def test_timeout_kills_local_stage_process_group(tmp_path: Path) -> None:
    marker = tmp_path / "descendant-survived"
    command = [
        sys.executable,
        "-c",
        (
            "import subprocess,sys,time;"
            "subprocess.Popen([sys.executable,'-c',"
            f"\"import time,pathlib;time.sleep(2);pathlib.Path({str(marker)!r}).touch()\"]);"
            "time.sleep(30)"
        ),
    ]
    with (tmp_path / "timeout.log").open("wb") as log:
        with pytest.raises(subprocess.TimeoutExpired):
            production_loop._run_local_stage(  # noqa: SLF001
                command, cwd=str(tmp_path), log=log, timeout_seconds=1
            )
    time.sleep(2.2)
    assert not marker.exists()
