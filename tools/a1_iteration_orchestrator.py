#!/usr/bin/env python3
"""Durable, fail-closed orchestration for one sealed A1 learner iteration.

This module deliberately contains no training, evaluation, or promotion science.
It sequences the authoritative A1 tools and records enough immutable evidence to
resume after an orchestrator crash without repeating a one-dose side effect.

State progression is monotonic::

    corpus_verified -> dose_dry_run -> dose_complete
        -> evaluation_verified -> promoted

``evaluation_verified`` means that ``a1_promotion_transaction`` has replayed the
typed calibration and evaluation evidence and produced a valid dry-run plan.
The state file is digest-sealed and updated atomically under an exclusive lock.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Iterator, Sequence
import uuid

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools import a1_one_dose_train as one_dose  # noqa: E402
from tools import a1_promotion_transaction as promotion  # noqa: E402


STATE_SCHEMA = "a1-iteration-state-v1"
STAGES = (
    "corpus_verified",
    "dose_dry_run",
    "dose_complete",
    "evaluation_verified",
    "promoted",
)
STATE_KEYS = {
    "schema_version",
    "iteration_id",
    "stage",
    "created_unix_ns",
    "updated_unix_ns",
    "training",
    "training_plan",
    "training_outputs",
    "evaluation",
    "promotion",
    "history",
    "state_sha256",
}


class IterationError(RuntimeError):
    """A fail-closed A1 iteration refusal."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _value_sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _file_ref(path: Path, *, where: str) -> dict[str, str]:
    try:
        canonical = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise IterationError(f"cannot resolve {where}: {error}") from error
    if not canonical.is_file() or canonical.stat().st_size <= 0:
        raise IterationError(f"{where} must be an existing non-empty file: {canonical}")
    return {"path": str(canonical), "sha256": _file_sha256(canonical)}


def _executable_ref(path: Path, *, where: str) -> dict[str, str]:
    """Bind a lexical executable path and its target without dropping a venv."""

    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    try:
        target = lexical.resolve(strict=True)
    except OSError as error:
        raise IterationError(f"cannot resolve {where}: {error}") from error
    if (
        not lexical.is_file()
        or not target.is_file()
        or not os.access(lexical, os.X_OK)
        or not os.access(target, os.X_OK)
    ):
        raise IterationError(f"{where} must be an executable file: {lexical}")
    return {
        "path": str(lexical),
        "target_path": str(target),
        "sha256": _file_sha256(target),
    }


def _new_path(path: Path, *, where: str) -> str:
    canonical = path.expanduser().resolve(strict=False)
    if canonical.exists():
        raise IterationError(f"refusing non-fresh {where}: {canonical}")
    return str(canonical)


def _verify_ref(value: Any, *, where: str) -> Path:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise IterationError(f"{where} is not an exact file reference")
    path = Path(str(value["path"]))
    try:
        canonical = path.resolve(strict=True)
    except OSError as error:
        raise IterationError(f"cannot resolve {where}: {error}") from error
    if str(canonical) != value["path"]:
        raise IterationError(f"{where} path is not canonical")
    actual = _file_sha256(canonical)
    if actual != value["sha256"]:
        raise IterationError(
            f"{where} hash drift: expected {value['sha256']} actual {actual}"
        )
    return canonical


def _verify_executable_ref(value: Any, *, where: str) -> Path:
    if not isinstance(value, dict) or set(value) != {
        "path",
        "target_path",
        "sha256",
    }:
        raise IterationError(f"{where} is not an exact executable reference")
    lexical = Path(str(value["path"]))
    if lexical != Path(os.path.abspath(os.fspath(lexical.expanduser()))):
        raise IterationError(f"{where} lexical path is not absolute")
    try:
        target = lexical.resolve(strict=True)
    except OSError as error:
        raise IterationError(f"cannot resolve {where}: {error}") from error
    if (
        str(target) != value["target_path"]
        or not lexical.is_file()
        or not os.access(lexical, os.X_OK)
        or _file_sha256(target) != value["sha256"]
    ):
        raise IterationError(f"{where} executable target drift")
    return lexical


def _seal(state: dict[str, Any]) -> dict[str, Any]:
    sealed = dict(state)
    sealed.pop("state_sha256", None)
    sealed["state_sha256"] = _value_sha256(sealed)
    return sealed


def _verify_state(state: Any) -> dict[str, Any]:
    if not isinstance(state, dict) or set(state) != STATE_KEYS:
        raise IterationError(
            "iteration state has unexpected or missing top-level fields"
        )
    if state["schema_version"] != STATE_SCHEMA:
        raise IterationError(f"iteration state schema must be {STATE_SCHEMA!r}")
    unhashed = dict(state)
    stated = unhashed.pop("state_sha256")
    if stated != _value_sha256(unhashed):
        raise IterationError("iteration state semantic digest mismatch")
    if state["stage"] not in STAGES:
        raise IterationError(f"unknown iteration stage: {state['stage']!r}")
    if not isinstance(state["history"], list) or not state["history"]:
        raise IterationError("iteration state history is missing")
    training = state["training"]
    if not isinstance(training, dict):
        raise IterationError("iteration training binding is not an object")
    for key in ("lock", "corpus_meta", "validation_manifest"):
        _verify_ref(training.get(key), where=f"training.{key}")
    _verify_executable_ref(training.get("python"), where="training.python")
    if not Path(str(training.get("data"))).resolve(strict=True).is_dir():
        raise IterationError("bound A1 corpus directory is missing")
    return state


def _load_state(path: Path) -> dict[str, Any]:
    try:
        canonical = path.expanduser().resolve(strict=True)
        value = json.loads(canonical.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise IterationError(f"cannot load iteration state: {error}") from error
    return _verify_state(value)


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_state(path: Path, state: dict[str, Any], *, create: bool = False) -> None:
    path = path.expanduser().resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    sealed = _seal(state)
    payload = json.dumps(sealed, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    if create:
        try:
            with path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError as error:
            raise IterationError(f"iteration state already exists: {path}") from error
        _fsync_directory(path.parent)
        return
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with tmp.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        _fsync_directory(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


@contextlib.contextmanager
def _state_lock(state_path: Path) -> Iterator[None]:
    state_path = state_path.expanduser().resolve(strict=False)
    lock_path = state_path.with_name(state_path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _transition(
    state: dict[str, Any],
    *,
    expected: str,
    target: str,
    action: str,
    updates: dict[str, Any],
) -> dict[str, Any]:
    if state["stage"] != expected:
        raise IterationError(
            f"{action} requires stage {expected!r}; current stage is {state['stage']!r}"
        )
    result = dict(state)
    result.update(updates)
    now = time.time_ns()
    result["stage"] = target
    result["updated_unix_ns"] = now
    result["history"] = [
        *state["history"],
        {"action": action, "from": expected, "to": target, "unix_ns": now},
    ]
    return result


def initialize(
    *,
    state_path: Path,
    lock_path: Path,
    data_path: Path,
    validation_path: Path,
    checkpoint: Path,
    report: Path,
    training_receipt: Path,
    python: Path,
    gpu: int,
    verify_fn: Callable[..., dict[str, Any]] = one_dose.verify_training_inputs,
) -> dict[str, Any]:
    """Verify and bind one immutable corpus and one fresh learner dose."""

    if gpu < 0:
        raise IterationError("gpu must be non-negative")
    python_ref = _executable_ref(python, where="learner python")
    try:
        verified = verify_fn(
            lock_path=lock_path, data_path=data_path, validation_path=validation_path
        )
        checkpoint_path = checkpoint.expanduser().resolve(strict=False)
        report_path = report.expanduser().resolve(strict=False)
        receipt_path = training_receipt.expanduser().resolve(strict=False)
        claim = one_dose._claim_path(verified)  # noqa: SLF001
        one_dose._require_fresh_outputs(  # noqa: SLF001
            checkpoint_path, report_path, receipt_path, claim=claim
        )
        one_dose._require_unconsumed_contract(verified)  # noqa: SLF001
    except (one_dose.ExecutorError, OSError, KeyError, TypeError) as error:
        raise IterationError(
            f"A1 corpus/dose initialization refused: {error}"
        ) from error

    meta_path = Path(verified["data_path"]) / "corpus_meta.json"
    now = time.time_ns()
    state = {
        "schema_version": STATE_SCHEMA,
        "iteration_id": uuid.uuid4().hex,
        "stage": "corpus_verified",
        "created_unix_ns": now,
        "updated_unix_ns": now,
        "training": {
            "contract_sha256": verified["contract_sha256"],
            "lock": _file_ref(Path(verified["lock_path"]), where="contract lock"),
            "data": str(Path(verified["data_path"]).resolve(strict=True)),
            "corpus_meta": _file_ref(meta_path, where="corpus metadata"),
            "payload_inventory_sha256": verified["payload_inventory_sha256"],
            "validation_manifest": _file_ref(
                Path(verified["validation_path"]), where="validation manifest"
            ),
            "selected_game_seed_set_sha256": verified["selected_game_seed_set_sha256"],
            "training_game_seed_set_sha256": verified["training_game_seed_set_sha256"],
            "validation_game_seed_set_sha256": verified[
                "validation_game_seed_set_sha256"
            ],
            "corpus_row_count": verified["corpus_row_count"],
            "training_row_count": verified["training_row_count"],
            "validation_row_count": verified["validation_row_count"],
            "checkpoint": _new_path(checkpoint_path, where="training checkpoint"),
            "report": _new_path(report_path, where="training report"),
            "receipt": _new_path(receipt_path, where="training receipt"),
            "python": python_ref,
            "gpu": gpu,
        },
        "training_plan": None,
        "training_outputs": None,
        "evaluation": None,
        "promotion": None,
        "history": [
            {
                "action": "initialize",
                "from": None,
                "to": "corpus_verified",
                "unix_ns": now,
            }
        ],
        "state_sha256": "",
    }
    _write_state(state_path, state, create=True)
    return _load_state(state_path)


def adopt_completed_retry(
    *,
    state_path: Path,
    lock_path: Path,
    data_path: Path,
    validation_path: Path,
    parent_claim: Path,
    retry_contract: Path,
    retry_receipt: Path,
    python: Path,
    gpu: int,
    verify_fn: Callable[..., dict[str, Any]] = one_dose.verify_training_inputs,
) -> dict[str, Any]:
    """Create or resume state from the single authorized completed v4 retry.

    This transition never invokes the learner.  It exists for the narrow case
    where a v3 dose failed before optimizer construction, the executor issued
    its immutable graph-layer-only v4 repair, and the orchestrator did not yet
    own a state file.  Every byte in both attempts is replayed before adoption.
    """

    if gpu < 0:
        raise IterationError("gpu must be non-negative")
    python_ref = _executable_ref(python, where="learner python")
    try:
        verified = verify_fn(
            lock_path=lock_path, data_path=data_path, validation_path=validation_path
        )
    except (one_dose.ExecutorError, OSError, KeyError, TypeError) as error:
        raise IterationError(
            f"A1 retry corpus verification refused: {error}"
        ) from error
    chain = _verify_completed_retry_chain(
        verified=verified,
        parent_claim_path=parent_claim,
        retry_contract_path=retry_contract,
        retry_receipt_path=retry_receipt,
        gpu=gpu,
    )
    plan = chain["retry_plan"]
    refs = chain["refs"]
    training_binding = {
        "contract_sha256": verified["contract_sha256"],
        "lock": _file_ref(Path(verified["lock_path"]), where="contract lock"),
        "data": str(Path(verified["data_path"]).resolve(strict=True)),
        "corpus_meta": _file_ref(
            Path(verified["data_path"]) / "corpus_meta.json",
            where="corpus metadata",
        ),
        "payload_inventory_sha256": verified["payload_inventory_sha256"],
        "validation_manifest": _file_ref(
            Path(verified["validation_path"]), where="validation manifest"
        ),
        "selected_game_seed_set_sha256": verified["selected_game_seed_set_sha256"],
        "training_game_seed_set_sha256": verified["training_game_seed_set_sha256"],
        "validation_game_seed_set_sha256": verified["validation_game_seed_set_sha256"],
        "corpus_row_count": verified["corpus_row_count"],
        "training_row_count": verified["training_row_count"],
        "validation_row_count": verified["validation_row_count"],
        "checkpoint": plan["checkpoint"],
        "report": plan["report"],
        "receipt": plan["receipt"],
        "python": python_ref,
        "gpu": gpu,
        "attempt_kind": "derived-retry-v4",
        "claim_identity_sha256": chain["claim_identity_sha256"],
        "parent_training_plan": chain["parent_plan"],
    }

    with _state_lock(state_path):
        canonical_state = state_path.expanduser().resolve(strict=False)
        if canonical_state.exists():
            state = _load_state(canonical_state)
            if state["training"]["contract_sha256"] != verified["contract_sha256"]:
                raise IterationError("retry adoption state binds a different contract")
            if state["stage"] == "dose_dry_run":
                if state["training_plan"] != chain["parent_plan"]:
                    raise IterationError(
                        "retry parent differs from the recorded v3 dry-run plan"
                    )
                next_state = _transition(
                    state,
                    expected="dose_dry_run",
                    target="dose_complete",
                    action="adopt_completed_retry_v4",
                    updates={
                        "training": training_binding,
                        "training_plan": plan,
                        "training_outputs": refs,
                    },
                )
                _write_state(canonical_state, next_state)
                return _load_state(canonical_state)
            if STAGES.index(state["stage"]) >= STAGES.index("dose_complete"):
                if (
                    state["training"] != training_binding
                    or state["training_plan"] != plan
                    or state["training_outputs"] != refs
                ):
                    raise IterationError(
                        "stored retry adoption differs from live evidence"
                    )
                return state
            raise IterationError(
                "retry adoption requires no state or a recorded v3 dose dry-run"
            )

        now = time.time_ns()
        adopted = {
            "schema_version": STATE_SCHEMA,
            "iteration_id": uuid.uuid4().hex,
            "stage": "dose_complete",
            "created_unix_ns": now,
            "updated_unix_ns": now,
            "training": training_binding,
            "training_plan": plan,
            "training_outputs": refs,
            "evaluation": None,
            "promotion": None,
            "history": [
                {
                    "action": "adopt_completed_retry_v4",
                    "from": None,
                    "to": "dose_complete",
                    "unix_ns": now,
                }
            ],
            "state_sha256": "",
        }
        _write_state(canonical_state, adopted, create=True)
        return _load_state(canonical_state)


def _dose_argv(state: dict[str, Any], *, go: bool) -> list[str]:
    training = state["training"]
    argv = [
        sys.executable,
        str(_REPO_ROOT / "tools" / "a1_one_dose_train.py"),
        "--lock",
        training["lock"]["path"],
        "--data",
        training["data"],
        "--validation-manifest",
        training["validation_manifest"]["path"],
        "--checkpoint",
        training["checkpoint"],
        "--report",
        training["report"],
        "--receipt",
        training["receipt"],
        "--python",
        training["python"]["path"],
        "--gpu",
        str(training["gpu"]),
    ]
    if go:
        argv.append("--go")
    return argv


def _run_tool(
    argv: list[str],
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> subprocess.CompletedProcess[str]:
    try:
        result = runner(
            argv,
            cwd=str(_REPO_ROOT),
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as error:
        raise IterationError(f"cannot start authoritative A1 tool: {error}") from error
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise IterationError(
            f"authoritative A1 tool refused (exit {result.returncode}): {detail}"
        )
    return result


def _run_json_tool(
    argv: list[str],
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    result = _run_tool(argv, runner=runner)
    try:
        value = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError) as error:
        raise IterationError(
            "authoritative A1 tool returned non-JSON output"
        ) from error
    if not isinstance(value, dict):
        raise IterationError("authoritative A1 tool returned a non-object")
    return value


def _load_digest_object(
    path: Path,
    *,
    where: str,
    digest_field: str,
    schema: str,
) -> tuple[dict[str, Any], Path]:
    """Load one canonical, non-symlink JSON object with an internal digest."""

    lexical = path.expanduser()
    if lexical.is_symlink():
        raise IterationError(f"{where} must not be a symlink")
    try:
        canonical = lexical.resolve(strict=True)
        payload = json.loads(canonical.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise IterationError(f"cannot load {where}: {error}") from error
    if not canonical.is_file() or not isinstance(payload, dict):
        raise IterationError(f"{where} must be a JSON object file")
    unhashed = dict(payload)
    stated = unhashed.pop(digest_field, None)
    if stated != one_dose._value_sha256(unhashed):  # noqa: SLF001
        raise IterationError(f"{where} semantic digest mismatch")
    if payload.get("schema_version") != schema:
        raise IterationError(f"{where} schema must be {schema!r}")
    return payload, canonical


def _expected_terminal_claim_from_receipt(
    receipt: dict[str, Any],
    *,
    claim_schema: str,
    receipt_path: Path,
) -> dict[str, Any]:
    """Reconstruct the exact terminal claim that must back ``receipt``."""

    claim = {
        key: value
        for key, value in receipt.items()
        if key not in {"receipt_sha256", "claim", "claim_state_sha256"}
    }
    claim["schema_version"] = claim_schema
    claim["receipt_target"] = str(receipt_path)
    return one_dose._with_digest(claim, "state_sha256")  # noqa: SLF001


def _verify_retry_command_repair(
    parent_command: list[str], retry_command: list[str]
) -> tuple[int, int]:
    """Replay the sole authorized r1->r2 semantic change."""

    if one_dose._literal_option_values(  # noqa: SLF001
        parent_command, "--graph-layers"
    ):
        raise IterationError("retry parent command must literally omit --graph-layers")
    if one_dose._literal_option_values(  # noqa: SLF001
        retry_command, "--graph-layers"
    ) != ["6"]:
        raise IterationError("retry command must contain exactly one --graph-layers 6")
    try:
        parent_args = one_dose._train_command_namespace(parent_command)  # noqa: SLF001
        retry_args = one_dose._train_command_namespace(retry_command)  # noqa: SLF001
        parent_mismatches = one_dose._checkpoint_architecture_mismatches(  # noqa: SLF001
            parent_args
        )
        retry_mismatches = one_dose._checkpoint_architecture_mismatches(  # noqa: SLF001
            retry_args
        )
    except one_dose.ExecutorError as error:
        raise IterationError(
            f"cannot replay retry architecture repair: {error}"
        ) from error
    if parent_mismatches != ["graph_layers checkpoint=6 cli=4"]:
        raise IterationError("retry parent is not the authorized graph-layer failure")
    if retry_mismatches:
        raise IterationError(
            "retry command still fails checkpoint architecture preflight"
        )
    allowed_drift = {"graph_layers", "checkpoint", "report"}
    parent_values = vars(parent_args)
    retry_values = vars(retry_args)
    drift = sorted(
        key
        for key in set(parent_values) | set(retry_values)
        if key not in allowed_drift and parent_values.get(key) != retry_values.get(key)
    )
    if drift:
        raise IterationError(
            f"retry changes non-architecture learner semantics: {drift}"
        )
    if Path(retry_args.init_checkpoint).resolve(strict=False) != Path(
        parent_args.init_checkpoint
    ).resolve(strict=False):
        raise IterationError("retry changes the sealed producer checkpoint")
    return int(parent_args.graph_layers), int(retry_args.graph_layers)


def _parent_plan_from_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    command = receipt["command"]
    checkpoint = one_dose._train_command_namespace(command).checkpoint  # noqa: SLF001
    report = one_dose._train_command_namespace(command).report  # noqa: SLF001
    return {
        "schema_version": one_dose.PLAN_SCHEMA,
        "mode": "dry-run",
        "contract_sha256": receipt["contract_sha256"],
        "claim_identity_sha256": receipt["contract_sha256"],
        "retry_contract": None,
        "global_n_full": 128,
        "world_size": receipt["world_size"],
        "gpu": receipt["gpu"],
        "command": command,
        "command_sha256": receipt["command_sha256"],
        "execution_binding": receipt["execution_binding"],
        "checkpoint": str(Path(checkpoint).expanduser().resolve(strict=False)),
        "report": str(Path(report).expanduser().resolve(strict=False)),
        "receipt": str(Path(receipt["receipt_target"]).resolve(strict=True)),
    }


def _verify_completed_retry_chain(
    *,
    verified: dict[str, Any],
    parent_claim_path: Path,
    retry_contract_path: Path,
    retry_receipt_path: Path,
    gpu: int,
) -> dict[str, Any]:
    """Verify the complete immutable v3-failure -> v4-success evidence chain."""

    expected_parent_claim = one_dose._claim_path(verified).resolve(strict=False)  # noqa: SLF001
    if parent_claim_path.expanduser().is_symlink():
        raise IterationError("retry parent claim must not be a symlink")
    try:
        parent_claim_path = parent_claim_path.expanduser().resolve(strict=True)
    except OSError as error:
        raise IterationError(f"cannot resolve retry parent claim: {error}") from error
    if parent_claim_path != expected_parent_claim:
        raise IterationError("retry parent claim is not the contract-keyed v3 claim")
    try:
        parent_claim = one_dose._load_claim_state(  # noqa: SLF001
            parent_claim_path, contract_sha256=verified["contract_sha256"]
        )
    except one_dose.ExecutorError as error:
        raise IterationError(f"cannot adopt retry parent claim: {error}") from error
    parent_receipt_path = Path(str(parent_claim.get("receipt_target", "")))
    parent_receipt, parent_receipt_path = _load_digest_object(
        parent_receipt_path,
        where="v3 failed training receipt",
        digest_field="receipt_sha256",
        schema=one_dose.RECEIPT_SCHEMA,
    )
    if (
        parent_claim.get("status") != "failed"
        or parent_claim.get("outputs") is not None
        or parent_receipt.get("status") != "failed"
        or parent_receipt.get("outputs") is not None
        or not isinstance(parent_claim.get("returncode"), int)
        or parent_claim["returncode"] == 0
        or parent_receipt.get("claim") != str(parent_claim_path)
        or parent_receipt.get("claim_state_sha256") != parent_claim.get("state_sha256")
        or parent_claim
        != _expected_terminal_claim_from_receipt(
            parent_receipt,
            claim_schema=one_dose.CLAIM_SCHEMA,
            receipt_path=parent_receipt_path,
        )
    ):
        raise IterationError(
            "v3 parent claim/receipt do not prove one zero-output failure"
        )

    retry_contract, retry_contract_path = _load_digest_object(
        retry_contract_path,
        where="learner retry contract",
        digest_field="retry_contract_sha256",
        schema=one_dose.RETRY_CONTRACT_SCHEMA,
    )
    parent_command = parent_receipt.get("command")
    if not isinstance(parent_command, list) or not all(
        isinstance(item, str) for item in parent_command
    ):
        raise IterationError("v3 parent receipt has no canonical command")
    if (
        parent_receipt.get("command_sha256") != one_dose._value_sha256(parent_command)  # noqa: SLF001
        or parent_receipt.get("execution_binding")
        != parent_claim.get("execution_binding")
    ):
        raise IterationError("v3 parent command/execution binding drift")
    try:
        one_dose._validate_execution_binding(  # noqa: SLF001
            parent_receipt["execution_binding"]
        )
    except one_dose.ExecutorError as error:
        raise IterationError(f"v3 parent execution binding invalid: {error}") from error

    parent_evidence = {
        "claim": str(parent_claim_path),
        "claim_file_sha256": _file_sha256(parent_claim_path),
        "claim_state_sha256": parent_claim["state_sha256"],
        "receipt": str(parent_receipt_path),
        "receipt_file_sha256": _file_sha256(parent_receipt_path),
        "receipt_sha256": parent_receipt["receipt_sha256"],
        "command_sha256": parent_receipt["command_sha256"],
        "returncode": parent_receipt["returncode"],
        "failure": parent_receipt["failure"],
    }
    identity_evidence = {
        "schema_version": one_dose.RETRY_IDENTITY_SCHEMA,
        "repair_kind": one_dose.RETRY_REPAIR_KIND,
        "parent_contract_sha256": verified["contract_sha256"],
        "parent": parent_evidence,
    }
    retry_identity = one_dose._value_sha256(identity_evidence)  # noqa: SLF001
    retry_receipt, retry_receipt_path = _load_digest_object(
        retry_receipt_path,
        where="v4 completed training receipt",
        digest_field="receipt_sha256",
        schema=one_dose.RETRY_RECEIPT_SCHEMA,
    )
    retry_command = retry_receipt.get("command")
    if not isinstance(retry_command, list) or not all(
        isinstance(item, str) for item in retry_command
    ):
        raise IterationError("v4 retry receipt has no canonical command")
    before_layers, after_layers = _verify_retry_command_repair(
        parent_command, retry_command
    )
    retry_args = one_dose._train_command_namespace(retry_command)  # noqa: SLF001
    checkpoint = Path(retry_args.checkpoint).expanduser().resolve(strict=False)
    report = Path(retry_args.report).expanduser().resolve(strict=False)
    optimizer = Path(str(checkpoint) + ".optimizer.pt")
    expected_preserved = {
        "parent_contract_sha256": verified["contract_sha256"],
        "parent_lock": str(verified["lock_path"]),
        "parent_lock_file_sha256": verified["lock_file_sha256"],
        "corpus": str(verified["data_path"]),
        "corpus_meta_file_sha256": verified["corpus_meta_file_sha256"],
        "payload_inventory_sha256": verified["payload_inventory_sha256"],
        "data_fingerprint": verified["data_fingerprint"],
        "producer_checkpoint_sha256": verified["producer"]["sha256"],
        "producer_checkpoint": str(verified["producer"]["path"]),
        "learner_training_recipe_sha256": one_dose._value_sha256(  # noqa: SLF001
            verified["recipe"]
        ),
        "learner_value_objective_sha256": one_dose._value_sha256(  # noqa: SLF001
            verified["objective"]
        ),
        "selected_game_seed_set_sha256": verified["selected_game_seed_set_sha256"],
        "training_game_seed_set_sha256": verified["training_game_seed_set_sha256"],
        "validation_manifest_file_sha256": verified["validation_file_sha256"],
        "validation_manifest": str(verified["validation_path"]),
        "validation_game_seed_set_sha256": verified["validation_game_seed_set_sha256"],
    }
    expected_retry_contract = {
        "schema_version": one_dose.RETRY_CONTRACT_SCHEMA,
        "retry_identity": identity_evidence,
        "retry_identity_sha256": retry_identity,
        "parent": {
            **parent_evidence,
            "pre_optimizer_proof": {
                "kind": "replayed_init_checkpoint_architecture_preflight",
                "mismatches": ["graph_layers checkpoint=6 cli=4"],
                "optimizer_steps": 0,
                "outputs": None,
            },
        },
        "preserved_bindings": expected_preserved,
        "retry": {
            "command_sha256": one_dose._value_sha256(retry_command),  # noqa: SLF001
            "architecture_correction": {
                "graph_layers_before": before_layers,
                "graph_layers_after": after_layers,
            },
            "checkpoint": str(checkpoint),
            "optimizer_sidecar": str(optimizer),
            "report": str(report),
            "receipt": str(retry_receipt_path),
        },
    }
    expected_retry_contract["retry_contract_sha256"] = one_dose._value_sha256(  # noqa: SLF001
        expected_retry_contract
    )
    if retry_contract != expected_retry_contract:
        raise IterationError("learner retry contract drifts from the proven v3 repair")
    retry_reference = {
        "path": str(retry_contract_path),
        "file_sha256": _file_sha256(retry_contract_path),
        "retry_contract_sha256": retry_contract["retry_contract_sha256"],
    }
    execution_binding = retry_receipt.get("execution_binding")
    try:
        one_dose._validate_execution_binding(execution_binding)  # noqa: SLF001
    except one_dose.ExecutorError as error:
        raise IterationError(f"v4 retry execution binding invalid: {error}") from error
    expected_receipt_bindings = {
        "status": "complete",
        "contract_sha256": verified["contract_sha256"],
        "claim_identity_sha256": retry_identity,
        "retry_contract": retry_reference,
        "lock": str(verified["lock_path"]),
        "lock_file_sha256": verified["lock_file_sha256"],
        "corpus": str(verified["data_path"]),
        "corpus_meta_file_sha256": verified["corpus_meta_file_sha256"],
        "payload_inventory_sha256": verified["payload_inventory_sha256"],
        "validation_manifest": str(verified["validation_path"]),
        "validation_manifest_file_sha256": verified["validation_file_sha256"],
        "producer_checkpoint_sha256": verified["producer"]["sha256"],
        "learner_training_recipe_sha256": one_dose._value_sha256(  # noqa: SLF001
            verified["recipe"]
        ),
        "command_sha256": one_dose._value_sha256(retry_command),  # noqa: SLF001
        "world_size": 1,
        "gpu": gpu,
        "returncode": 0,
        "failure": None,
    }
    drift = {
        key: {"expected": value, "actual": retry_receipt.get(key)}
        for key, value in expected_receipt_bindings.items()
        if retry_receipt.get(key) != value
    }
    if (
        drift
        or execution_binding.get("command_sha256")
        != expected_receipt_bindings["command_sha256"]
    ):
        raise IterationError(f"v4 retry receipt binding drift: {drift}")
    try:
        outputs = one_dose._verify_training_outputs(  # noqa: SLF001
            checkpoint=checkpoint,
            report=report,
            verified=verified,
            execution_binding=execution_binding,
        )
    except (one_dose.ExecutorError, OSError) as error:
        raise IterationError(
            f"v4 retry output verification refused: {error}"
        ) from error
    if retry_receipt.get("outputs") != outputs:
        raise IterationError("v4 retry receipt output hashes/semantics drift")
    expected_derived_claim = one_dose._claim_path(  # noqa: SLF001
        {**verified, "claim_identity_sha256": retry_identity}
    ).resolve(strict=False)
    try:
        derived_claim_path = Path(str(retry_receipt.get("claim", ""))).resolve(
            strict=True
        )
        derived_claim = one_dose._load_claim_state(  # noqa: SLF001
            derived_claim_path,
            contract_sha256=verified["contract_sha256"],
            claim_identity_sha256=retry_identity,
        )
    except (OSError, one_dose.ExecutorError) as error:
        raise IterationError(f"cannot adopt v4 derived claim: {error}") from error
    if (
        derived_claim_path != expected_derived_claim
        or derived_claim.get("status") != "complete"
        or retry_receipt.get("claim_state_sha256") != derived_claim.get("state_sha256")
        or derived_claim
        != _expected_terminal_claim_from_receipt(
            retry_receipt,
            claim_schema=one_dose.RETRY_CLAIM_SCHEMA,
            receipt_path=retry_receipt_path,
        )
    ):
        raise IterationError("v4 retry receipt and derived terminal claim disagree")
    retry_plan = {
        "schema_version": one_dose.PLAN_SCHEMA,
        "mode": "dry-run",
        "contract_sha256": verified["contract_sha256"],
        "claim_identity_sha256": retry_identity,
        "retry_contract": retry_contract,
        "global_n_full": 128,
        "world_size": 1,
        "gpu": gpu,
        "command": retry_command,
        "command_sha256": expected_receipt_bindings["command_sha256"],
        "execution_binding": execution_binding,
        "checkpoint": str(checkpoint),
        "report": str(report),
        "receipt": str(retry_receipt_path),
    }
    return {
        "parent_plan": _parent_plan_from_receipt(
            {**parent_receipt, "receipt_target": str(parent_receipt_path)}
        ),
        "retry_plan": retry_plan,
        "outputs": outputs,
        "refs": {
            "checkpoint": _file_ref(checkpoint, where="retry candidate checkpoint"),
            "optimizer_sidecar": _file_ref(
                optimizer, where="retry candidate optimizer sidecar"
            ),
            "report": _file_ref(report, where="retry candidate training report"),
            "receipt": _file_ref(
                retry_receipt_path, where="v4 completed training receipt"
            ),
            "derived_claim": _file_ref(
                derived_claim_path, where="v4 derived terminal claim"
            ),
            "retry_contract": _file_ref(
                retry_contract_path, where="learner retry contract"
            ),
            "parent_claim": _file_ref(
                parent_claim_path, where="v3 failed parent claim"
            ),
            "parent_receipt": _file_ref(
                parent_receipt_path, where="v3 failed parent receipt"
            ),
        },
        "claim_identity_sha256": retry_identity,
    }


def dose_dry_run(
    *,
    state_path: Path,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    with _state_lock(state_path):
        state = _load_state(state_path)
        if STAGES.index(state["stage"]) >= STAGES.index("dose_dry_run"):
            return state
        if state["stage"] != "corpus_verified":
            raise IterationError("dose dry-run cannot skip the verified-corpus stage")
        plan = _run_json_tool(_dose_argv(state, go=False), runner=runner)
        if (
            plan.get("schema_version") != one_dose.PLAN_SCHEMA
            or plan.get("mode") != "dry-run"
            or plan.get("contract_sha256") != state["training"]["contract_sha256"]
            or plan.get("global_n_full") != 128
            or plan.get("world_size") != 1
            or plan.get("checkpoint") != state["training"]["checkpoint"]
            or plan.get("report") != state["training"]["report"]
            or plan.get("receipt") != state["training"]["receipt"]
        ):
            raise IterationError(
                "one-dose dry-run plan drifted from the iteration binding"
            )
        next_state = _transition(
            state,
            expected="corpus_verified",
            target="dose_dry_run",
            action="dose_dry_run",
            updates={"training_plan": plan},
        )
        _write_state(state_path, next_state)
        return _load_state(state_path)


def _load_complete_training_receipt(state: dict[str, Any]) -> dict[str, Any]:
    training = state["training"]
    receipt_path = Path(training["receipt"])
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise IterationError(f"cannot adopt A1 training receipt: {error}") from error
    if not isinstance(payload, dict):
        raise IterationError("A1 training receipt is not an object")
    unhashed = dict(payload)
    stated = unhashed.pop("receipt_sha256", None)
    if stated != one_dose._value_sha256(unhashed):  # noqa: SLF001
        raise IterationError("A1 training receipt digest mismatch")
    if (
        payload.get("schema_version") != one_dose.RECEIPT_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("contract_sha256") != training["contract_sha256"]
        or payload.get("command_sha256") != state["training_plan"].get("command_sha256")
    ):
        raise IterationError(
            "A1 training receipt does not bind the planned complete dose"
        )
    execution_binding = payload.get("execution_binding")
    if not isinstance(execution_binding, dict) or execution_binding != state[
        "training_plan"
    ].get("execution_binding"):
        raise IterationError(
            "A1 training receipt environment/command differs from the dry-run"
        )
    try:
        verified = one_dose.verify_training_inputs(
            lock_path=Path(training["lock"]["path"]),
            data_path=Path(training["data"]),
            validation_path=Path(training["validation_manifest"]["path"]),
        )
        actual_outputs = one_dose._verify_training_outputs(  # noqa: SLF001
            checkpoint=Path(training["checkpoint"]),
            report=Path(training["report"]),
            verified=verified,
            execution_binding=execution_binding,
        )
        claim_path = Path(str(payload.get("claim", ""))).resolve(strict=True)
        expected_claim = one_dose._claim_path(verified).resolve(strict=True)  # noqa: SLF001
        if claim_path != expected_claim:
            raise IterationError("A1 training receipt binds the wrong durable claim")
        claim = one_dose._load_claim_state(  # noqa: SLF001
            claim_path, contract_sha256=training["contract_sha256"]
        )
    except (one_dose.ExecutorError, OSError) as error:
        raise IterationError(
            f"A1 training receipt adoption refused: {error}"
        ) from error
    if claim.get("status") != "complete":
        raise IterationError("A1 durable training claim is not complete")
    if payload.get("claim_state_sha256") != claim.get("state_sha256"):
        raise IterationError("A1 receipt and durable claim disagree")
    if payload.get("outputs") != actual_outputs:
        raise IterationError("A1 training receipt output bindings drifted")
    return payload


def dose_go(
    *,
    state_path: Path,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    with _state_lock(state_path):
        state = _load_state(state_path)
        if STAGES.index(state["stage"]) >= STAGES.index("dose_complete"):
            return state
        if state["stage"] != "dose_dry_run":
            raise IterationError("dose execution requires a recorded one-dose dry-run")
        receipt_path = Path(state["training"]["receipt"])
        if not receipt_path.exists():
            # The one-dose process prints the JSON plan before launching
            # ``train_bc``; trainer progress then shares stdout.  Do not attempt
            # to parse that mixed stream.  The terminal v3 receipt below is the
            # authoritative command/environment/output proof.
            _run_tool(_dose_argv(state, go=True), runner=runner)
        receipt = _load_complete_training_receipt(state)
        outputs = receipt["outputs"]
        artifact_refs = {
            "checkpoint": _file_ref(
                Path(outputs["checkpoint"]), where="candidate checkpoint"
            ),
            "optimizer_sidecar": _file_ref(
                Path(outputs["optimizer_sidecar"]), where="candidate optimizer sidecar"
            ),
            "report": _file_ref(
                Path(outputs["report"]), where="candidate training report"
            ),
            "receipt": _file_ref(receipt_path, where="one-dose training receipt"),
        }
        next_state = _transition(
            state,
            expected="dose_dry_run",
            target="dose_complete",
            action="dose_go_or_resume",
            updates={"training_outputs": artifact_refs},
        )
        _write_state(state_path, next_state)
        return _load_state(state_path)


def verify_evaluation(
    *,
    state_path: Path,
    registry_path: Path,
    current_pointer: Path,
    adjudication_path: Path,
    promotion_receipt: Path,
    reason: str,
    promotion_lock: Path | None = None,
    promotion_fn: Callable[..., dict[str, Any]] = promotion.execute_promotion,
) -> dict[str, Any]:
    """Replay all typed evaluation evidence through the promotion verifier."""

    with _state_lock(state_path):
        state = _load_state(state_path)
        if STAGES.index(state["stage"]) >= STAGES.index("evaluation_verified"):
            return state
        if state["stage"] != "dose_complete":
            raise IterationError(
                "evaluation verification requires a complete learner dose"
            )
        try:
            plan = promotion_fn(
                registry_path=registry_path,
                current_pointer=current_pointer,
                contract_lock=Path(state["training"]["lock"]["path"]),
                adjudication_path=adjudication_path,
                training_receipt=Path(state["training_outputs"]["receipt"]["path"]),
                receipt_path=promotion_receipt,
                reason=reason,
                lock_path=promotion_lock,
                go=False,
            )
        except (promotion.PromotionError, OSError) as error:
            raise IterationError(
                f"A1 evaluation/promotion preflight refused: {error}"
            ) from error
        outputs = state["training_outputs"]
        candidate = plan.get("candidate", {})
        if (
            plan.get("schema_version") != promotion.RECEIPT_SCHEMA
            or plan.get("status") != "dry_run"
            or plan.get("contract", {}).get("contract_sha256")
            != state["training"]["contract_sha256"]
            or Path(str(candidate.get("path", ""))).resolve(strict=True)
            != Path(outputs["checkpoint"]["path"])
            or candidate.get("sha256") != outputs["checkpoint"]["sha256"]
            or candidate.get("training_report", {}).get("sha256")
            != outputs["report"]["sha256"]
            or plan.get("training_receipt", {}).get("sha256")
            != outputs["receipt"]["sha256"]
            or Path(str(plan.get("training_receipt", {}).get("path", ""))).resolve(
                strict=True
            )
            != Path(outputs["receipt"]["path"])
        ):
            raise IterationError(
                "typed evaluation selected a candidate other than this dose"
            )
        evaluation = {
            "registry": _file_ref(
                registry_path, where="champion registry before promotion"
            ),
            "current_pointer": _file_ref(
                current_pointer, where="CURRENT_CHAMPION before promotion"
            ),
            "adjudication": _file_ref(
                adjudication_path, where="promotion adjudication"
            ),
            "promotion_receipt": _new_path(
                promotion_receipt, where="promotion receipt"
            ),
            "promotion_lock": (
                str(promotion_lock.expanduser().resolve(strict=False))
                if promotion_lock is not None
                else None
            ),
            "reason": reason,
            "dry_run_plan": plan,
        }
        next_state = _transition(
            state,
            expected="dose_complete",
            target="evaluation_verified",
            action="verify_evaluation_and_promotion_dry_run",
            updates={"evaluation": evaluation},
        )
        _write_state(state_path, next_state)
        return _load_state(state_path)


def _adopt_committed_promotion(state: dict[str, Any]) -> dict[str, Any]:
    evaluation = state["evaluation"]
    receipt_path = Path(evaluation["promotion_receipt"])
    try:
        receipt, _, registry, pointer, _, _ = promotion._load_recovery_receipt(  # noqa: SLF001
            receipt_path
        )
    except (promotion.PromotionError, OSError) as error:
        raise IterationError(f"cannot adopt promotion transaction: {error}") from error
    if receipt.get("status") != "committed":
        raise IterationError(
            "promotion receipt is not committed; use a1_promotion_transaction.py "
            "recover before resuming the iteration"
        )
    if (
        promotion._sha256(registry) != receipt["registry"]["after_sha256"]  # noqa: SLF001
        or promotion._sha256(pointer)  # noqa: SLF001
        != receipt["current_pointer"]["after_sha256"]
    ):
        raise IterationError("committed promotion state does not match its receipt")
    expected = evaluation["dry_run_plan"]
    # The go transaction must be the exact transaction that was preflighted.
    # ``transaction_id``, timestamps, and status are intentionally fresh at go;
    # every science binding and every before/after mutation hash is immutable.
    for key in (
        "registry",
        "current_pointer",
        "contract",
        "adjudication",
        "training_receipt",
        "candidate",
        "champion",
        "evidence",
        "promotion_count",
        "nth_confirmation_required",
        "reason",
        "fleet_ckpt_updated",
    ):
        if receipt.get(key) != expected.get(key):
            raise IterationError(
                f"promotion receipt drifted from dry-run field {key!r}"
            )
    return receipt


def promote(
    *,
    state_path: Path,
    promotion_fn: Callable[..., dict[str, Any]] = promotion.execute_promotion,
) -> dict[str, Any]:
    with _state_lock(state_path):
        state = _load_state(state_path)
        if state["stage"] == "promoted":
            return state
        if state["stage"] != "evaluation_verified":
            raise IterationError(
                "promotion requires verified calibration/evaluation evidence"
            )
        evaluation = state["evaluation"]
        receipt_path = Path(evaluation["promotion_receipt"])
        if not receipt_path.exists():
            try:
                promotion_fn(
                    registry_path=Path(evaluation["registry"]["path"]),
                    current_pointer=Path(evaluation["current_pointer"]["path"]),
                    contract_lock=Path(state["training"]["lock"]["path"]),
                    adjudication_path=Path(evaluation["adjudication"]["path"]),
                    training_receipt=Path(state["training_outputs"]["receipt"]["path"]),
                    receipt_path=receipt_path,
                    reason=evaluation["reason"],
                    lock_path=(
                        Path(evaluation["promotion_lock"])
                        if evaluation["promotion_lock"] is not None
                        else None
                    ),
                    go=True,
                )
            except (promotion.PromotionError, OSError) as error:
                raise IterationError(f"A1 promotion refused: {error}") from error
        receipt = _adopt_committed_promotion(state)
        committed = {
            "receipt": _file_ref(receipt_path, where="committed promotion receipt"),
            "transaction_id": receipt["transaction_id"],
            "registry_after_sha256": receipt["registry"]["after_sha256"],
            "current_pointer_after_sha256": receipt["current_pointer"]["after_sha256"],
            "promotion_count": receipt["promotion_count"],
        }
        next_state = _transition(
            state,
            expected="evaluation_verified",
            target="promoted",
            action="promote_or_resume",
            updates={"promotion": committed},
        )
        _write_state(state_path, next_state)
        return _load_state(state_path)


def status(*, state_path: Path) -> dict[str, Any]:
    with _state_lock(state_path):
        state = _load_state(state_path)
        if STAGES.index(state["stage"]) >= STAGES.index("dose_complete"):
            for name, ref in state["training_outputs"].items():
                _verify_ref(ref, where=f"training_outputs.{name}")
        if state["stage"] == "evaluation_verified":
            _verify_ref(state["evaluation"]["registry"], where="evaluation.registry")
            _verify_ref(
                state["evaluation"]["current_pointer"],
                where="evaluation.current_pointer",
            )
            _verify_ref(
                state["evaluation"]["adjudication"], where="evaluation.adjudication"
            )
        if state["stage"] == "promoted":
            _verify_ref(state["promotion"]["receipt"], where="promotion.receipt")
            receipt = _adopt_committed_promotion(state)
            if receipt["transaction_id"] != state["promotion"]["transaction_id"]:
                raise IterationError("promotion transaction identity drift")
        return state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="verify and bind the sealed A1 corpus")
    init.add_argument("--state", required=True, type=Path)
    init.add_argument("--lock", required=True, type=Path)
    init.add_argument("--data", required=True, type=Path)
    init.add_argument("--validation-manifest", required=True, type=Path)
    init.add_argument("--checkpoint", required=True, type=Path)
    init.add_argument("--report", required=True, type=Path)
    init.add_argument("--training-receipt", required=True, type=Path)
    init.add_argument("--python", type=Path, default=Path(sys.executable))
    init.add_argument("--gpu", type=int, default=0)
    retry = sub.add_parser(
        "adopt-retry",
        help="adopt the authorized completed v4 retry without rerunning training",
    )
    retry.add_argument("--state", required=True, type=Path)
    retry.add_argument("--lock", required=True, type=Path)
    retry.add_argument("--data", required=True, type=Path)
    retry.add_argument("--validation-manifest", required=True, type=Path)
    retry.add_argument("--parent-claim", required=True, type=Path)
    retry.add_argument("--retry-contract", required=True, type=Path)
    retry.add_argument("--retry-receipt", required=True, type=Path)
    retry.add_argument("--python", type=Path, default=Path(sys.executable))
    retry.add_argument("--gpu", type=int, default=0)
    for name in ("dose-dry", "dose-go", "promote", "status"):
        command = sub.add_parser(name)
        command.add_argument("--state", required=True, type=Path)
    evidence = sub.add_parser(
        "verify-evaluation", help="verify typed calibration/evaluation evidence"
    )
    evidence.add_argument("--state", required=True, type=Path)
    evidence.add_argument("--registry", required=True, type=Path)
    evidence.add_argument("--current-pointer", required=True, type=Path)
    evidence.add_argument("--adjudication", required=True, type=Path)
    evidence.add_argument("--promotion-receipt", required=True, type=Path)
    evidence.add_argument("--reason", required=True)
    evidence.add_argument("--promotion-lock", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "init":
            result = initialize(
                state_path=args.state,
                lock_path=args.lock,
                data_path=args.data,
                validation_path=args.validation_manifest,
                checkpoint=args.checkpoint,
                report=args.report,
                training_receipt=args.training_receipt,
                python=args.python,
                gpu=args.gpu,
            )
        elif args.command == "adopt-retry":
            result = adopt_completed_retry(
                state_path=args.state,
                lock_path=args.lock,
                data_path=args.data,
                validation_path=args.validation_manifest,
                parent_claim=args.parent_claim,
                retry_contract=args.retry_contract,
                retry_receipt=args.retry_receipt,
                python=args.python,
                gpu=args.gpu,
            )
        elif args.command == "dose-dry":
            result = dose_dry_run(state_path=args.state)
        elif args.command == "dose-go":
            result = dose_go(state_path=args.state)
        elif args.command == "verify-evaluation":
            result = verify_evaluation(
                state_path=args.state,
                registry_path=args.registry,
                current_pointer=args.current_pointer,
                adjudication_path=args.adjudication,
                promotion_receipt=args.promotion_receipt,
                reason=args.reason,
                promotion_lock=args.promotion_lock,
            )
        elif args.command == "promote":
            result = promote(state_path=args.state)
        else:
            result = status(state_path=args.state)
    except (IterationError, OSError) as error:
        print(f"REFUSING: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
