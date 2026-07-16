"""Config-first orchestration for one production RL improvement turn.

The stage tools own Catan/search/training science.  This module only gives them
one durable transaction boundary: fixed ordering, exact input/output hashes,
crash-safe resume, and a single repository revision.  Commands are executed as
argument vectors (never through a shell) and may invoke only the canonical
entry point assigned to their stage.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Iterator, Mapping, Sequence


SCHEMA_VERSION = "catan-zero-production-loop-v1"
STATE_SCHEMA_VERSION = "catan-zero-production-loop-state-v1"
STAGES = (
    "generate",
    "harvest",
    "audit",
    "composite",
    "train",
    "evaluate",
    "promote",
)
STAGE_TOOLS = {
    "generate": frozenset(("generate.py", "a1_production_executor.py")),
    "harvest": frozenset(("a1_harvest_transaction.py",)),
    "audit": frozenset(("a1_pre_wave_contract.py",)),
    "composite": frozenset(
        ("a1_build_post_wave_composite.py", "build_memmap_corpus.py")
    ),
    "train": frozenset(("train.py", "a1_one_dose_train.py", "a1_scratch_train.py")),
    "evaluate": frozenset(("evaluate.py", "a1_h100_eval_fleet.py")),
    "promote": frozenset(("a1_promotion_transaction.py",)),
}
PLACEHOLDERS = frozenset(("repo", "state_dir", "python"))


class ProductionLoopError(RuntimeError):
    """The loop is malformed, stale, or cannot advance safely."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _value_sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _file_ref(path: Path, *, where: str) -> dict[str, Any]:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise ProductionLoopError(f"cannot resolve {where}: {error}") from error
    if not resolved.is_file():
        raise ProductionLoopError(f"{where} must be a regular file: {resolved}")
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "sha256": _file_sha256(resolved),
        "size_bytes": stat.st_size,
    }


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ("git", "-C", str(repo), *args), text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "output", "")
        raise ProductionLoopError(f"git {' '.join(args)} failed: {detail}") from error


def _expand(value: str, values: Mapping[str, str]) -> str:
    result = value
    for name in PLACEHOLDERS:
        result = result.replace("{" + name + "}", values[name])
    if "{" in result or "}" in result:
        raise ProductionLoopError(f"unknown or malformed command placeholder: {value!r}")
    return result


def _command_tool(command: Sequence[str]) -> str:
    if len(command) < 2:
        raise ProductionLoopError("stage command must invoke Python and one tool")
    # The first token is the bound Python interpreter.  Keeping the internal
    # executor behind it makes virtualenv identity explicit and replayable.
    return Path(command[1]).name


def load_config(path: Path, *, state_dir: Path) -> dict[str, Any]:
    """Load, normalize, and validate a complete loop configuration."""

    try:
        source = path.expanduser().resolve(strict=True)
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProductionLoopError(f"cannot load loop config: {error}") from error
    if not isinstance(payload, dict):
        raise ProductionLoopError("loop config must be a JSON object")
    required = {
        "schema_version",
        "loop_id",
        "repository",
        "repository_commit",
        "python",
        "require_clean_repository",
        "stages",
    }
    if set(payload) != required:
        raise ProductionLoopError(
            f"loop config fields must be exactly {sorted(required)}"
        )
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ProductionLoopError(f"schema_version must be {SCHEMA_VERSION!r}")
    loop_id = payload["loop_id"]
    if not isinstance(loop_id, str) or not loop_id.strip() or "/" in loop_id:
        raise ProductionLoopError("loop_id must be a nonempty path-safe string")
    repo = Path(str(payload["repository"])).expanduser().resolve(strict=True)
    if not (repo / ".git").exists() and _git(repo, "rev-parse", "--git-dir") == "":
        raise ProductionLoopError("repository is not a Git checkout")
    commit = _git(repo, "rev-parse", "HEAD")
    if payload["repository_commit"] != commit:
        raise ProductionLoopError(
            "repository_commit does not match checkout HEAD: "
            f"expected={payload['repository_commit']} actual={commit}"
        )
    require_clean = payload["require_clean_repository"]
    if require_clean is not True:
        raise ProductionLoopError("production loops require a clean repository")
    if _git(repo, "status", "--porcelain", "--untracked-files=no"):
        raise ProductionLoopError("production loop repository has tracked modifications")
    python = Path(str(payload["python"])).expanduser().resolve(strict=True)
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ProductionLoopError("python must be an executable file")

    stages = payload["stages"]
    if not isinstance(stages, dict) or set(stages) != set(STAGES):
        raise ProductionLoopError(
            f"stages must contain exactly: {', '.join(STAGES)}"
        )
    values = {
        "repo": str(repo),
        "state_dir": str(state_dir.expanduser().resolve(strict=False)),
        "python": str(python),
    }
    normalized_stages: dict[str, Any] = {}
    for name in STAGES:
        stage = stages[name]
        if not isinstance(stage, dict) or set(stage) != {
            "command",
            "inputs",
            "outputs",
            "timeout_seconds",
        }:
            raise ProductionLoopError(
                f"stage {name!r} requires command, inputs, outputs, timeout_seconds"
            )
        command = stage["command"]
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(item, str) and item for item in command)
        ):
            raise ProductionLoopError(f"stage {name!r} command must be string argv")
        command = [_expand(item, values) for item in command]
        if Path(command[0]).resolve(strict=False) != python:
            raise ProductionLoopError(
                f"stage {name!r} must use the config-bound Python interpreter"
            )
        tool = Path(command[1]).expanduser().resolve(strict=True)
        try:
            tool.relative_to(repo)
        except ValueError as error:
            raise ProductionLoopError(
                f"stage {name!r} tool escapes the bound repository"
            ) from error
        command[1] = str(tool)
        if _command_tool(command) not in STAGE_TOOLS[name]:
            raise ProductionLoopError(
                f"stage {name!r} cannot invoke {_command_tool(command)!r}; "
                f"choose from {sorted(STAGE_TOOLS[name])}"
            )
        if name == "audit" and "audit" not in command[2:]:
            raise ProductionLoopError("audit stage must select the audit subcommand")
        if name == "promote" and "promote" not in command[2:]:
            raise ProductionLoopError("promote stage must select the promote subcommand")
        tool_name = _command_tool(command)
        if tool_name == "a1_production_executor.py" and not {
            "run",
            "--go",
        }.issubset(command[2:]):
            raise ProductionLoopError(
                "fleet generation must select the executor run transaction with --go"
            )
        if tool_name == "a1_one_dose_train.py" and "--go" not in command[2:]:
            raise ProductionLoopError(
                "one-dose training stage must execute rather than emit another dry-run"
            )
        if tool_name == "a1_promotion_transaction.py" and "--go" not in command[2:]:
            raise ProductionLoopError(
                "promotion stage must commit the verified transaction with --go"
            )
        inputs = stage["inputs"]
        outputs = stage["outputs"]
        if not isinstance(inputs, list) or not all(isinstance(v, str) for v in inputs):
            raise ProductionLoopError(f"stage {name!r} inputs must be string paths")
        if (
            not isinstance(outputs, list)
            or not outputs
            or not all(isinstance(v, str) for v in outputs)
        ):
            raise ProductionLoopError(f"stage {name!r} must bind output receipt files")
        inputs = [_expand(item, values) for item in inputs]
        outputs = [_expand(item, values) for item in outputs]
        timeout = stage["timeout_seconds"]
        if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout < 1:
            raise ProductionLoopError(
                f"stage {name!r} timeout_seconds must be a positive integer"
            )
        normalized_stages[name] = {
            "command": command,
            "inputs": inputs,
            "outputs": outputs,
            "timeout_seconds": timeout,
        }
    normalized = dict(payload)
    normalized["repository"] = str(repo)
    normalized["python"] = str(python)
    normalized["stages"] = normalized_stages
    normalized["config_path"] = str(source)
    normalized["config_sha256"] = _value_sha256(payload)
    # Prove this is one connected turn, not seven unrelated commands sharing a
    # JSON file.  Each downstream transaction must consume an immutable output
    # from an earlier transaction; the fixed runtime order remains STAGES.
    prior_outputs: set[str] = set()
    all_outputs: set[str] = set()
    for index, name in enumerate(STAGES):
        stage = normalized_stages[name]
        duplicate_outputs = all_outputs.intersection(stage["outputs"])
        if duplicate_outputs:
            raise ProductionLoopError(
                f"stage {name!r} reuses output identities: {sorted(duplicate_outputs)}"
            )
        if index and not prior_outputs.intersection(stage["inputs"]):
            raise ProductionLoopError(
                f"stage {name!r} is disconnected: it must consume at least one "
                "receipt produced by an earlier stage"
            )
        prior_outputs.update(stage["outputs"])
        all_outputs.update(stage["outputs"])
    return normalized


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextlib.contextmanager
def _lock(state_dir: Path) -> Iterator[None]:
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / "loop.lock").open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ProductionLoopError("another loop process owns this state directory") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _new_state(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "loop_id": config["loop_id"],
        "config_sha256": config["config_sha256"],
        "repository_commit": config["repository_commit"],
        "completed_stages": [],
        "stages": {},
    }


def _load_state(path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return _new_state(config)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProductionLoopError(f"cannot load loop state: {error}") from error
    if not isinstance(value, dict) or value.get("schema_version") != STATE_SCHEMA_VERSION:
        raise ProductionLoopError("loop state schema is invalid")
    for key in ("loop_id", "config_sha256", "repository_commit"):
        if value.get(key) != config.get(key):
            raise ProductionLoopError(f"loop state {key} differs from configuration")
    completed = value.get("completed_stages")
    if not isinstance(completed, list) or completed != list(STAGES[: len(completed)]):
        raise ProductionLoopError("loop state stages are not a canonical prefix")
    if set(value.get("stages", {})) != set(completed):
        raise ProductionLoopError("loop state stage receipts are incomplete")
    for name in completed:
        receipt = value["stages"][name]
        stage = config["stages"][name]
        if receipt.get("command_sha256") != _value_sha256(stage["command"]):
            raise ProductionLoopError(f"completed stage {name!r} command drifted")
        current_inputs = [_file_ref(Path(item), where=f"{name} input") for item in stage["inputs"]]
        current_outputs = [
            _file_ref(Path(item), where=f"{name} output") for item in stage["outputs"]
        ]
        if current_inputs != receipt.get("inputs") or current_outputs != receipt.get("outputs"):
            raise ProductionLoopError(f"completed stage {name!r} artifact bytes drifted")
    return value


def plan(config: Mapping[str, Any], *, state_dir: Path) -> dict[str, Any]:
    """Return the exact stage plan without creating or modifying run state."""

    state = _load_state(state_dir / "state.json", config)
    completed = set(state["completed_stages"])
    return {
        "schema_version": SCHEMA_VERSION,
        "loop_id": config["loop_id"],
        "repository_commit": config["repository_commit"],
        "completed_stages": list(state["completed_stages"]),
        "pending_stages": [name for name in STAGES if name not in completed],
        "commands": {
            name: config["stages"][name]["command"]
            for name in STAGES
            if name not in completed
        },
    }


def execute(config: Mapping[str, Any], *, state_dir: Path) -> dict[str, Any]:
    """Execute all pending stages, stopping at the first failed transaction."""

    state_dir = state_dir.expanduser().resolve(strict=False)
    state_path = state_dir / "state.json"
    with _lock(state_dir):
        state = _load_state(state_path, config)
        for name in STAGES[len(state["completed_stages"]) :]:
            stage = config["stages"][name]
            inputs = [
                _file_ref(Path(item), where=f"{name} input") for item in stage["inputs"]
            ]
            for output in stage["outputs"]:
                if Path(output).expanduser().exists():
                    raise ProductionLoopError(
                        f"stage {name!r} refuses pre-existing unreceipted output: {output}"
                    )
            log_path = (
                state_dir
                / "logs"
                / f"{name}.attempt-{time.time_ns()}.log"
            )
            log_path.parent.mkdir(parents=True, exist_ok=True)
            started_ns = time.time_ns()
            with log_path.open("xb") as log:
                try:
                    completed = subprocess.run(
                        stage["command"],
                        cwd=config["repository"],
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=stage["timeout_seconds"],
                        check=False,
                    )
                except (OSError, subprocess.TimeoutExpired) as error:
                    raise ProductionLoopError(
                        f"stage {name!r} could not complete; see {log_path}: {error}"
                    ) from error
            if completed.returncode != 0:
                raise ProductionLoopError(
                    f"stage {name!r} exited {completed.returncode}; see {log_path}"
                )
            outputs = [
                _file_ref(Path(item), where=f"{name} output") for item in stage["outputs"]
            ]
            receipt = {
                "stage": name,
                "command": stage["command"],
                "command_sha256": _value_sha256(stage["command"]),
                "inputs": inputs,
                "outputs": outputs,
                "log": _file_ref(log_path, where=f"{name} log"),
                "started_unix_ns": started_ns,
                "completed_unix_ns": time.time_ns(),
            }
            state["completed_stages"].append(name)
            state["stages"][name] = receipt
            _atomic_json(state_path, state)
        return state
