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
import signal
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
    "generate": frozenset(("tools/fleet/a1_production_executor.py",)),
    "harvest": frozenset(("tools/fleet/a1_harvest_transaction.py",)),
    "audit": frozenset(("tools/a1_pre_wave_contract.py",)),
    "composite": frozenset(("tools/a1_build_post_wave_composite.py",)),
    "train": frozenset(
        ("tools/a1_one_dose_train.py", "tools/a1_scratch_train.py")
    ),
    "evaluate": frozenset(("tools/evaluate.py",)),
    "promote": frozenset(("tools/a1_promotion_transaction.py",)),
}
PLACEHOLDERS = frozenset(("repo", "state_dir", "python"))

# These are the semantic edges that make a turn one RL transaction.  A path
# merely appearing in ``inputs``/``outputs`` is not evidence that the stage
# tool actually consumed or produced it.  Each binding below must occur once
# as the value of its typed CLI flag.  The three predecessor-bound inputs are
# the load-bearing learning loop: composite -> learner data, learner -> eval
# candidate, and eval -> promotion adjudication.
STAGE_ARTIFACT_BINDINGS: Mapping[str, tuple[tuple[str, str, str, bool], ...]] = {
    "generate": (("generation_receipt", "output", "--receipt", False),),
    "audit": (
        ("harvest_relocation", "input", "--harvest-relocation", True),
        ("audit_receipt", "output", "--out", False),
    ),
    "train": (
        ("training_data", "input", "--data", True),
        ("candidate_checkpoint", "output", "--checkpoint", False),
    ),
    "evaluate": (
        ("candidate_checkpoint", "input", "--candidate", True),
        ("evaluation_adjudication", "output", "--out", False),
    ),
    "promote": (
        ("evaluation_adjudication", "input", "--adjudication", True),
        ("training_execution_receipt", "input", "--training-receipt", False),
        ("promotion_receipt", "output", "--receipt", False),
    ),
}

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
        "kind": "file",
        "sha256": _file_sha256(resolved),
        "size_bytes": stat.st_size,
    }


def _artifact_ref(path: Path, *, where: str) -> dict[str, Any]:
    """Content-address one immutable file or directory artifact."""

    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise ProductionLoopError(f"cannot resolve {where}: {error}") from error
    if resolved.is_file():
        return _file_ref(resolved, where=where)
    if not resolved.is_dir():
        raise ProductionLoopError(f"{where} must be a regular file or directory")
    records: list[dict[str, Any]] = []
    total_size = 0
    for child in sorted(resolved.rglob("*")):
        if child.is_symlink():
            raise ProductionLoopError(f"{where} contains a symlink: {child}")
        if child.is_dir():
            continue
        if not child.is_file():
            raise ProductionLoopError(f"{where} contains a non-regular file: {child}")
        size = child.stat().st_size
        total_size += size
        records.append(
            {
                "path": child.relative_to(resolved).as_posix(),
                "sha256": _file_sha256(child),
                "size_bytes": size,
            }
        )
    return {
        "path": str(resolved),
        "kind": "directory",
        "sha256": _value_sha256(records),
        "size_bytes": total_size,
        "file_count": len(records),
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


def _command_tool(command: Sequence[str], *, repo: Path) -> str:
    if len(command) < 2:
        raise ProductionLoopError("stage command must invoke Python and one tool")
    try:
        tool = Path(command[1]).expanduser().resolve(strict=True)
        return tool.relative_to(repo).as_posix()
    except (OSError, ValueError) as error:
        raise ProductionLoopError(
            "stage tool must be an exact checked-in repository path"
        ) from error


def _normalize_artifact_path(value: str) -> str:
    return str(Path(value).expanduser().resolve(strict=False))


def _flag_path(command: Sequence[str], flag: str, *, stage: str) -> str:
    positions = [index for index, value in enumerate(command) if value == flag]
    if len(positions) != 1:
        raise ProductionLoopError(
            f"stage {stage!r} must bind {flag} exactly once, found {len(positions)}"
        )
    index = positions[0]
    if index + 1 >= len(command) or command[index + 1].startswith("--"):
        raise ProductionLoopError(f"stage {stage!r} has no path value for {flag}")
    return _normalize_artifact_path(command[index + 1])


def _bind_stage_artifacts(
    *,
    name: str,
    command: Sequence[str],
    inputs: Sequence[str],
    outputs: Sequence[str],
    predecessor_outputs: set[str],
) -> list[dict[str, str]]:
    bindings: list[dict[str, str]] = []
    if name == "harvest":
        executor_receipt = _flag_path(command, "--executor-receipt", stage=name)
        if executor_receipt not in inputs or executor_receipt not in predecessor_outputs:
            raise ProductionLoopError(
                "harvest must consume the immediate generation --receipt through "
                "--executor-receipt"
            )
        destination = Path(_flag_path(command, "--destination", stage=name))
        relocation = _normalize_artifact_path(str(destination / "relocation_map.json"))
        if relocation not in outputs:
            raise ProductionLoopError(
                "harvest output must declare DESTINATION/relocation_map.json"
            )
        bindings.extend(
            (
                {
                    "kind": "generation_receipt",
                    "direction": "input",
                    "flag": "--executor-receipt",
                    "path": executor_receipt,
                },
                {
                    "kind": "harvest_relocation",
                    "direction": "output",
                    "flag": "--destination/relocation_map.json",
                    "path": relocation,
                },
            )
        )
    if name == "audit":
        audit_out = Path(_flag_path(command, "--out", stage=name))
        selected_games = _normalize_artifact_path(
            str(audit_out.with_suffix(".selected_games.json"))
        )
        if selected_games not in outputs:
            raise ProductionLoopError(
                "audit outputs must declare OUT.selected_games.json"
            )
        bindings.append(
            {
                "kind": "selected_game_manifest",
                "direction": "output",
                "flag": "--out.selected_games.json",
                "path": selected_games,
            }
        )
    if name == "composite":
        audit_receipt = _flag_path(command, "--post-wave-audit", stage=name)
        selected_games = _flag_path(
            command, "--selected-game-manifest", stage=name
        )
        if audit_receipt not in inputs or audit_receipt not in predecessor_outputs:
            raise ProductionLoopError(
                "composite must consume the immediate audit --out through "
                "--post-wave-audit"
            )
        if selected_games not in inputs or selected_games not in predecessor_outputs:
            raise ProductionLoopError(
                "composite must consume the immediate audit selected-game manifest "
                "through --selected-game-manifest"
            )
        output_root = Path(_flag_path(command, "--out", stage=name))
        descriptor = _normalize_artifact_path(
            str(output_root / "memmap_composite.json")
        )
        build_receipt = _normalize_artifact_path(
            str(output_root / "build_receipt.json")
        )
        missing = {descriptor, build_receipt}.difference(outputs)
        if missing:
            raise ProductionLoopError(
                "composite outputs must declare OUT/memmap_composite.json and "
                f"OUT/build_receipt.json; missing={sorted(missing)}"
            )
        bindings.extend(
            (
                {
                    "kind": "audit_receipt",
                    "direction": "input",
                    "flag": "--post-wave-audit",
                    "path": audit_receipt,
                },
                {
                    "kind": "selected_game_manifest",
                    "direction": "input",
                    "flag": "--selected-game-manifest",
                    "path": selected_games,
                },
                {
                    "kind": "training_data",
                    "direction": "output",
                    "flag": "--out/memmap_composite.json",
                    "path": descriptor,
                },
                {
                    "kind": "composite_build_receipt",
                    "direction": "output",
                    "flag": "--out/build_receipt.json",
                    "path": build_receipt,
                },
            )
        )
    for kind, direction, flag, requires_predecessor in STAGE_ARTIFACT_BINDINGS.get(
        name, ()
    ):
        path = _flag_path(command, flag, stage=name)
        declared = set(inputs if direction == "input" else outputs)
        if path not in declared:
            raise ProductionLoopError(
                f"stage {name!r} {kind} bound by {flag} is not declared as an "
                f"{direction}: {path}"
            )
        if requires_predecessor and path not in predecessor_outputs:
            raise ProductionLoopError(
                f"stage {name!r} {kind} must be the exact immediate predecessor "
                f"output passed through {flag}: {path}"
            )
        bindings.append(
            {"kind": kind, "direction": direction, "flag": flag, "path": path}
        )
    argv_paths = {
        _normalize_artifact_path(value)
        for value in command[2:]
        if value and not value.startswith("--")
    }
    semantically_bound = {binding["path"] for binding in bindings}
    unbound = [
        path
        for path in (*inputs, *outputs)
        if path not in argv_paths and path not in semantically_bound
    ]
    if unbound:
        raise ProductionLoopError(
            f"stage {name!r} declares artifacts absent from its argv: {unbound}"
        )
    return bindings


def _repository_guard(config: Mapping[str, Any], *, stage: str) -> None:
    repo = Path(str(config["repository"]))
    expected = str(config["repository_commit"])
    actual = _git(repo, "rev-parse", "HEAD")
    if actual != expected:
        raise ProductionLoopError(
            f"stage {stage!r} repository revision drift: expected={expected} "
            f"actual={actual}"
        )
    if _git(repo, "status", "--porcelain", "--untracked-files=all"):
        raise ProductionLoopError(
            f"stage {stage!r} repository is not clean, including untracked files"
        )
    stage_config = config["stages"][stage]
    current_tool = _file_ref(
        Path(stage_config["command"][1]), where=f"{stage} stage tool"
    )
    if current_tool != stage_config.get("tool"):
        raise ProductionLoopError(f"stage {stage!r} tool bytes drifted")


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    """Contain a timed-out local stage and all descendants in its session."""

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    # The session leader may exit on SIGTERM while a descendant ignores it.
    # Kill the original group directly: a separate signal-0 probe races group
    # teardown and can return EPERM on macOS after the leader is reaped.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        return
    if process.poll() is None:
        process.wait()


def _run_local_stage(
    command: Sequence[str],
    *,
    cwd: str,
    log: Any,
    timeout_seconds: int,
) -> subprocess.CompletedProcess[bytes]:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        returncode = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        _terminate_process_group(process)
        raise
    except BaseException:
        _terminate_process_group(process)
        raise
    return subprocess.CompletedProcess(command, returncode)


def _remote_cancellation_command(
    command: Sequence[str], *, repo: Path
) -> list[str] | None:
    """Return only an entrypoint's explicit receipt-bound cancellation contract."""

    if _command_tool(command, repo=repo) != "tools/fleet/a1_production_executor.py":
        return None
    result = list(command)
    try:
        operation = result.index("run", 2)
    except ValueError as error:
        raise ProductionLoopError(
            "production executor command lost its run operation"
        ) from error
    result[operation] = "stop"
    while "--resume" in result:
        result.remove("--resume")
    if "--go" not in result:
        result.append("--go")
    return result


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
    if _git(repo, "status", "--porcelain", "--untracked-files=all"):
        raise ProductionLoopError(
            "production loop repository must be clean, including untracked files"
        )
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
    previous_outputs: set[str] = set()
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
        command[1] = str(tool)
        tool_name = _command_tool(command, repo=repo)
        if tool_name not in STAGE_TOOLS[name]:
            raise ProductionLoopError(
                f"stage {name!r} cannot invoke {tool_name!r}; "
                f"choose from {sorted(STAGE_TOOLS[name])}"
            )
        if name == "audit" and "audit" not in command[2:]:
            raise ProductionLoopError("audit stage must select the audit subcommand")
        if name == "promote" and "promote" not in command[2:]:
            raise ProductionLoopError("promote stage must select the promote subcommand")
        if tool_name == "tools/fleet/a1_production_executor.py" and not {
            "run",
            "--go",
        }.issubset(command[2:]):
            raise ProductionLoopError(
                "fleet generation must select the executor run transaction with --go"
            )
        if tool_name == "tools/a1_one_dose_train.py" and "--go" not in command[2:]:
            raise ProductionLoopError(
                "one-dose training stage must execute rather than emit another dry-run"
            )
        if tool_name == "tools/a1_scratch_train.py":
            if "--go" not in command[2:]:
                raise ProductionLoopError(
                    "scratch training stage must execute with --go"
                )
            if "--execution-receipt" not in command[2:]:
                raise ProductionLoopError(
                    "scratch training stage must bind a fresh --execution-receipt"
                )
        if tool_name == "tools/a1_promotion_transaction.py" and "--go" not in command[2:]:
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
        inputs = [_normalize_artifact_path(_expand(item, values)) for item in inputs]
        outputs = [_normalize_artifact_path(_expand(item, values)) for item in outputs]
        if len(inputs) != len(set(inputs)) or len(outputs) != len(set(outputs)):
            raise ProductionLoopError(f"stage {name!r} repeats artifact paths")
        artifact_bindings = _bind_stage_artifacts(
            name=name,
            command=command,
            inputs=inputs,
            outputs=outputs,
            predecessor_outputs=previous_outputs,
        )
        if name == "train" and tool_name == "tools/a1_one_dose_train.py":
            composite_receipt = _flag_path(
                command, "--composite-build-receipt", stage=name
            )
            if (
                composite_receipt not in inputs
                or composite_receipt not in previous_outputs
            ):
                raise ProductionLoopError(
                    "one-dose training must consume the immediate composite "
                    "OUT/build_receipt.json through --composite-build-receipt"
                )
            training_receipt = _flag_path(command, "--receipt", stage=name)
            training_report = _flag_path(command, "--report", stage=name)
            if training_receipt not in outputs:
                raise ProductionLoopError(
                    "one-dose --receipt must be a declared train output"
                )
            if training_report not in outputs:
                raise ProductionLoopError(
                    "one-dose --report must be a declared train output"
                )
            artifact_bindings.extend(
                (
                    {
                        "kind": "composite_build_receipt",
                        "direction": "input",
                        "flag": "--composite-build-receipt",
                        "path": composite_receipt,
                    },
                    {
                        "kind": "training_report",
                        "direction": "output",
                        "flag": "--report",
                        "path": training_report,
                    },
                    {
                        "kind": "training_execution_receipt",
                        "direction": "output",
                        "flag": "--receipt",
                        "path": training_receipt,
                    },
                )
            )
            if "--architecture-upgrade-receipt" in command:
                upgrade_receipt = _flag_path(
                    command, "--architecture-upgrade-receipt", stage=name
                )
                if upgrade_receipt not in inputs:
                    raise ProductionLoopError(
                        "--architecture-upgrade-receipt must be a declared train input"
                    )
                artifact_bindings.append(
                    {
                        "kind": "architecture_upgrade_receipt",
                        "direction": "input",
                        "flag": "--architecture-upgrade-receipt",
                        "path": upgrade_receipt,
                    }
                )
        if name == "train" and tool_name == "tools/a1_scratch_train.py":
            execution_receipt = _flag_path(
                command, "--execution-receipt", stage=name
            )
            if execution_receipt not in outputs:
                raise ProductionLoopError(
                    "scratch --execution-receipt must be a declared train output"
                )
            plan_receipt = _flag_path(command, "--receipt", stage=name)
            if plan_receipt not in inputs:
                raise ProductionLoopError(
                    "scratch --receipt plan authority must be a declared train input"
                )
            artifact_bindings.extend(
                (
                    {
                        "kind": "scratch_plan_receipt",
                        "direction": "input",
                        "flag": "--receipt",
                        "path": plan_receipt,
                    },
                    {
                        "kind": "training_execution_receipt",
                        "direction": "output",
                        "flag": "--execution-receipt",
                        "path": execution_receipt,
                    },
                )
            )
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
            "tool": _file_ref(tool, where=f"{name} stage tool"),
            "artifact_bindings": artifact_bindings,
        }
        previous_outputs = set(outputs)
    normalized = dict(payload)
    normalized["repository"] = str(repo)
    normalized["python"] = str(python)
    normalized["stages"] = normalized_stages
    normalized["config_path"] = str(source)
    normalized["config_sha256"] = _value_sha256(payload)
    # Prove this is one connected turn, not seven unrelated commands sharing a
    # JSON file. Every stage consumes its immediate predecessor, while typed
    # bindings above prove the critical data/checkpoint/adjudication edges.
    previous_outputs = set()
    all_outputs: set[str] = set()
    for index, name in enumerate(STAGES):
        stage = normalized_stages[name]
        duplicate_outputs = all_outputs.intersection(stage["outputs"])
        if duplicate_outputs:
            raise ProductionLoopError(
                f"stage {name!r} reuses output identities: {sorted(duplicate_outputs)}"
            )
        if index and not previous_outputs.intersection(stage["inputs"]):
            raise ProductionLoopError(
                f"stage {name!r} is disconnected: it must consume an output "
                "from its immediate predecessor"
            )
        previous_outputs = set(stage["outputs"])
        all_outputs.update(stage["outputs"])
    training_receipt = next(
        binding["path"]
        for binding in normalized_stages["promote"]["artifact_bindings"]
        if binding["kind"] == "training_execution_receipt"
    )
    issued_training_receipts = {
        binding["path"]
        for binding in normalized_stages["train"]["artifact_bindings"]
        if binding["kind"] == "training_execution_receipt"
        and binding["direction"] == "output"
    }
    if training_receipt not in issued_training_receipts:
        raise ProductionLoopError(
            "promotion --training-receipt must be the exact typed train-stage "
            "execution receipt"
        )
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
        current_inputs = [
            _artifact_ref(Path(item), where=f"{name} input")
            for item in stage["inputs"]
        ]
        current_outputs = [
            _artifact_ref(Path(item), where=f"{name} output")
            for item in stage["outputs"]
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
            _repository_guard(config, stage=name)
            inputs = [
                _artifact_ref(Path(item), where=f"{name} input")
                for item in stage["inputs"]
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
                    completed = _run_local_stage(
                        stage["command"],
                        cwd=config["repository"],
                        log=log,
                        timeout_seconds=stage["timeout_seconds"],
                    )
                except subprocess.TimeoutExpired as error:
                    cancellation = _remote_cancellation_command(
                        stage["command"], repo=Path(config["repository"])
                    )
                    if cancellation is not None:
                        try:
                            stopped = _run_local_stage(
                                cancellation,
                                cwd=config["repository"],
                                log=log,
                                timeout_seconds=min(stage["timeout_seconds"], 120),
                            )
                        except (OSError, subprocess.TimeoutExpired) as cancel_error:
                            raise ProductionLoopError(
                                f"stage {name!r} timed out and its exact remote "
                                f"cancellation failed; see {log_path}: {cancel_error}"
                            ) from cancel_error
                        if stopped.returncode != 0:
                            raise ProductionLoopError(
                                f"stage {name!r} timed out and its exact remote "
                                f"cancellation exited {stopped.returncode}; see {log_path}"
                            )
                    raise ProductionLoopError(
                        f"stage {name!r} could not complete; see {log_path}: {error}"
                    ) from error
                except OSError as error:
                    raise ProductionLoopError(
                        f"stage {name!r} could not complete; see {log_path}: {error}"
                    ) from error
            if completed.returncode != 0:
                raise ProductionLoopError(
                    f"stage {name!r} exited {completed.returncode}; see {log_path}"
                )
            _repository_guard(config, stage=name)
            if (
                name == "train"
                and _command_tool(stage["command"], repo=Path(config["repository"]))
                == "tools/a1_scratch_train.py"
            ):
                execution_path = Path(
                    _flag_path(stage["command"], "--execution-receipt", stage=name)
                )
                try:
                    execution = json.loads(execution_path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError) as error:
                    raise ProductionLoopError(
                        "scratch stage did not emit a readable execution receipt"
                    ) from error
                if not isinstance(execution, dict) or not (
                    execution.get("schema_version")
                    == "a1-coherent-scratch-training-execution-v2"
                    and execution.get("status") == "completed"
                    and execution.get("go") is True
                ):
                    raise ProductionLoopError(
                        "scratch stage execution receipt is not a completed --go run"
                    )
            outputs = [
                _artifact_ref(Path(item), where=f"{name} output")
                for item in stage["outputs"]
            ]
            receipt = {
                "stage": name,
                "command": stage["command"],
                "command_sha256": _value_sha256(stage["command"]),
                "inputs": inputs,
                "outputs": outputs,
                "artifact_bindings": stage.get("artifact_bindings", []),
                "log": _file_ref(log_path, where=f"{name} log"),
                "started_unix_ns": started_ns,
                "completed_unix_ns": time.time_ns(),
            }
            state["completed_stages"].append(name)
            state["stages"][name] = receipt
            _atomic_json(state_path, state)
        return state
