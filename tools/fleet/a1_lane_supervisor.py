#!/usr/bin/env python3
"""Remote one-GPU A1 supervisor: three exact category jobs, sequentially."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

SCHEMA = "a1-production-lane-v1"
RECEIPT_SCHEMA = "a1-production-job-receipt-v1"
CATEGORY_ORDER = ("current_producer", "recent_history", "hard_negative")
CLIENT_ENVIRONMENT = {
    "CUDA_MPS_PIPE_DIRECTORY": "/tmp/mps_pipe_host",
    "CUDA_MPS_LOG_DIRECTORY": "/tmp/mps_log_host",
}
FORBIDDEN_ADAPTIVE_ARGV = (
    "--n-full-wide",
    "--n-full-wide-threshold",
    "--raw-policy-above-width",
)


class SupervisorError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SupervisorError(f"cannot load {path}: {error}") from error
    if not isinstance(value, dict):
        raise SupervisorError(f"{path} must contain an object")
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _create_receipt(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise SupervisorError(f"receipt already exists: {path}") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


@contextmanager
def _lock(path: Path, *, blocking: bool) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(descriptor, flags)
        except BlockingIOError as error:
            raise SupervisorError(f"lane supervisor is already active: {path}") from error
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def load_lane(path: Path) -> dict[str, Any]:
    lane = _load(path)
    if lane.get("schema_version") != SCHEMA:
        raise SupervisorError(f"lane schema must be {SCHEMA}")
    digest = lane.get("lane_sha256")
    unhashed = dict(lane)
    unhashed.pop("lane_sha256", None)
    if digest != _digest(unhashed):
        raise SupervisorError("lane semantic digest mismatch")
    commands = lane.get("commands")
    if not isinstance(commands, list) or len(commands) != 3:
        raise SupervisorError("lane must contain exactly three jobs")
    if tuple(command.get("category") for command in commands) != CATEGORY_ORDER:
        raise SupervisorError("lane category order drift")
    if lane.get("client_environment") != CLIENT_ENVIRONMENT:
        raise SupervisorError("lane MPS client environment drift")
    for key in ("repo_dir", "python", "receipt_dir", "quarantine_dir", "log_dir", "lane_lock"):
        if not isinstance(lane.get(key), str) or not lane[key]:
            raise SupervisorError(f"lane {key} must be an explicit path")
    operator_manifests = lane.get("operator_manifests")
    if not isinstance(operator_manifests, dict) or set(operator_manifests) != {"lock", "render"}:
        raise SupervisorError("lane must bind exact lock/render operator manifests")
    for name, record in operator_manifests.items():
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise SupervisorError(f"invalid {name} operator-manifest binding")
        path_value = Path(record["path"])
        if not path_value.is_file() or _sha256(path_value) != record["sha256"]:
            raise SupervisorError(f"{name} operator manifest drift")
    worker = lane.get("worker_id")
    gpu = str(lane.get("gpu"))
    previous: str | None = None
    for index, command in enumerate(commands):
        if command.get("worker_id") != worker or str(command.get("gpu")) != gpu:
            raise SupervisorError("lane mixes workers or physical GPUs")
        argv = command.get("argv")
        if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
            raise SupervisorError("job argv must be a string list")
        if command.get("argv_sha256") != _digest(argv):
            raise SupervisorError("job argv digest mismatch")
        if "--skip-guards" in argv or "--no-seed-claim" in argv:
            raise SupervisorError("guard/seed-claim bypass is forbidden")
        if "--resume" not in argv:
            raise SupervisorError("exact A1 jobs must carry explicit --resume")
        try:
            n_full = int(argv[argv.index("--n-full") + 1])
        except (ValueError, IndexError) as error:
            raise SupervisorError("exact A1 job lacks --n-full") from error
        if n_full != 128 or any(flag in argv for flag in FORBIDDEN_ADAPTIVE_ARGV):
            raise SupervisorError("exact A1 jobs require n128 with no adaptive/wide override")
        expected_dependencies = [] if index == 0 else [previous]
        if command.get("must_run_after") != expected_dependencies:
            raise SupervisorError("job dependency order drift")
        environment = command.get("environment", {})
        if environment.get("CUDA_VISIBLE_DEVICES") != gpu:
            raise SupervisorError("job CUDA_VISIBLE_DEVICES differs from lane GPU")
        for key, value in CLIENT_ENVIRONMENT.items():
            if environment.get(key) != value:
                raise SupervisorError(f"job {key} differs from the systemd MPS service")
        previous = str(command["job_id"])
    return lane


def _receipt_path(lane: dict[str, Any], job_id: str) -> Path:
    return Path(lane["receipt_dir"]) / f"{job_id}.json"


def _pid_alive(pid: Any) -> bool:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _pid_is_exact(pid: Any, lane: dict[str, Any], command: dict[str, Any]) -> bool:
    if not _pid_alive(pid):
        return False
    try:
        raw = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
    except OSError:
        return False
    actual = [part.decode(errors="surrogateescape") for part in raw.split(b"\0") if part]
    return actual == [lane["python"], *command["argv"]]


@contextmanager
def _job_process_lock(path: Path) -> Iterator[int]:
    """A lock inherited by the generator, closing the spawn/PID receipt race."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            # A generator orphaned by supervisor death still holds this exact
            # open-file-description lock. Wait for it before inspecting or
            # moving any output.
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield descriptor
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _mark_completed(
    receipt_path: Path, receipt: dict[str, Any], command: dict[str, Any]
) -> dict[str, Any]:
    try:
        completed = _validate_completed(command)
    except SupervisorError as error:
        receipt.update(
            {
                "status": "failed",
                "validation_error": str(error),
                "ended_at": time.time(),
            }
        )
        _atomic_json(receipt_path, receipt)
        raise
    receipt.update(
        {"status": "complete", "return_code": 0, "ended_at": time.time(), **completed}
    )
    _atomic_json(receipt_path, receipt)
    return receipt


def _quarantine_attempt(
    lane: dict[str, Any],
    command: dict[str, Any],
    receipt_path: Path,
    receipt: dict[str, Any],
    *,
    reason: str,
) -> dict[str, Any]:
    """Atomically preserve an exact-but-incomplete attempt before fresh replay.

    Worker-level incremental progress is intentionally not trusted here: a
    fixed-size shard can contain a prefix of the next game. Keeping that shard
    and replaying the game would duplicate training rows. Completed manifests
    are handled before this function and are never moved or regenerated.
    """
    argv = command["argv"]
    out_dir = Path(argv[argv.index("--out-dir") + 1])
    quarantine_root = Path(lane["quarantine_dir"])
    quarantine_root.mkdir(parents=True, exist_ok=True)
    pending = receipt.get("quarantine_pending")
    if pending is None and (not out_dir.exists() or not any(out_dir.iterdir())):
        return receipt
    if pending is None:
        target = quarantine_root / (
            f"{command['job_id']}.attempt-{int(receipt.get('attempts', 0)):03d}."
            f"{time.time_ns()}"
        )
        pending = {
            "schema_version": "a1-production-quarantine-receipt-v1",
            "status": "prepared",
            "job_id": command["job_id"],
            "lane_sha256": lane["lane_sha256"],
            "argv_sha256": command["argv_sha256"],
            "source": str(out_dir),
            "destination": str(target),
            "reason": reason,
            "created_at": time.time(),
        }
        _create_receipt(target.with_name(target.name + ".receipt.json"), pending)
        receipt["quarantine_pending"] = pending
        _atomic_json(receipt_path, receipt)
    if (
        pending.get("job_id") != command["job_id"]
        or pending.get("lane_sha256") != lane["lane_sha256"]
        or pending.get("argv_sha256") != command["argv_sha256"]
        or pending.get("source") != str(out_dir)
    ):
        raise SupervisorError("pending quarantine receipt drift")
    target = Path(pending["destination"])
    if target.parent != quarantine_root:
        raise SupervisorError("pending quarantine escaped its root")
    if out_dir.exists() and not target.exists():
        try:
            os.rename(out_dir, target)
        except OSError as error:
            raise SupervisorError(
                f"cannot atomically quarantine attempt {command['job_id']}: {error}"
            ) from error
    elif not target.exists():
        raise SupervisorError("pending quarantine has neither source nor destination")
    elif out_dir.exists():
        raise SupervisorError("pending quarantine has both source and destination")
    quarantine_receipt = dict(pending)
    quarantine_receipt.update({"status": "complete", "completed_at": time.time()})
    _atomic_json(target.with_name(target.name + ".receipt.json"), quarantine_receipt)
    quarantines = list(receipt.get("quarantines", []))
    quarantines.append(quarantine_receipt)
    receipt.update({"status": "prepared", "quarantines": quarantines})
    receipt.pop("quarantine_pending", None)
    _atomic_json(receipt_path, receipt)
    return receipt


def _validate_completed(command: dict[str, Any]) -> dict[str, Any]:
    argv = command["argv"]
    out_dir = Path(argv[argv.index("--out-dir") + 1])
    attempts = int(argv[argv.index("--games") + 1])
    manifest_path = out_dir / "manifest.json"
    attestation_path = Path(command["output_attestation"]["destination"])
    if not manifest_path.is_file() or not attestation_path.is_file():
        raise SupervisorError(f"completed job {command['job_id']} lacks manifest/attestation")
    manifest = _load(manifest_path)
    if (
        int(manifest.get("games_requested", -1)) != attempts
        or int(manifest.get("games_completed", -1)) != attempts
        or int(manifest.get("games_failed", -1)) != 0
        or manifest.get("errors") not in ([], None)
        or int(manifest.get("base_seed", -1))
        != int(argv[argv.index("--base-seed") + 1])
    ):
        raise SupervisorError(f"completed job {command['job_id']} manifest is not exact/clean")
    if _sha256(attestation_path) != command["output_attestation"]["source_file_sha256"]:
        raise SupervisorError(f"job {command['job_id']} attestation bytes drifted")
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "attestation_sha256": _sha256(attestation_path),
    }


def _ensure_attestation(command: dict[str, Any]) -> None:
    source = Path(command["output_attestation"]["source"])
    destination = Path(command["output_attestation"]["destination"])
    expected = command["output_attestation"]["source_file_sha256"]
    if _sha256(source) != expected:
        raise SupervisorError(f"source attestation drift for {command['job_id']}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _sha256(destination) != expected:
            raise SupervisorError(f"destination attestation drift for {command['job_id']}")
        return
    try:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError:
        if _sha256(destination) != expected:
            raise SupervisorError(f"destination attestation race for {command['job_id']}")
        return
    with os.fdopen(descriptor, "wb") as handle, source.open("rb") as source_handle:
        shutil.copyfileobj(source_handle, handle)
        handle.flush()
        os.fsync(handle.fileno())


def _run_job(lane: dict[str, Any], command: dict[str, Any]) -> dict[str, Any]:
    receipt_path = _receipt_path(lane, command["job_id"])
    receipt = _load(receipt_path) if receipt_path.exists() else None
    if receipt is not None:
        if receipt.get("schema_version") != RECEIPT_SCHEMA:
            raise SupervisorError(f"job receipt schema drift: {receipt_path}")
        if receipt.get("lane_sha256") != lane["lane_sha256"]:
            raise SupervisorError(f"job receipt lane drift: {receipt_path}")
        if receipt.get("argv_sha256") != command["argv_sha256"]:
            raise SupervisorError(f"job receipt argv drift: {receipt_path}")
        if receipt.get("status") == "complete":
            _validate_completed(command)
            return receipt
    else:
        out_dir = Path(command["argv"][command["argv"].index("--out-dir") + 1])
        if out_dir.exists() and any(out_dir.iterdir()):
            raise SupervisorError(
                f"output exists without O_EXCL receipt for {command['job_id']}"
            )
        receipt = {
            "schema_version": RECEIPT_SCHEMA,
            "job_id": command["job_id"],
            "lane_sha256": lane["lane_sha256"],
            "argv_sha256": command["argv_sha256"],
            "status": "prepared",
            "attempts": 0,
            "created_at": time.time(),
        }
        _create_receipt(receipt_path, receipt)

    process_lock = Path(lane["receipt_dir"]) / f"{command['job_id']}.process.lock"
    with _job_process_lock(process_lock) as process_lock_fd:
        # The lock may have waited for an orphaned exact child. Reload what
        # the prior supervisor durably recorded before making any decision.
        receipt = _load(receipt_path)
        out_dir = Path(command["argv"][command["argv"].index("--out-dir") + 1])
        if (out_dir / "manifest.json").exists():
            try:
                return _mark_completed(receipt_path, receipt, command)
            except SupervisorError:
                receipt = _load(receipt_path)
                if receipt.get("status") == "complete":
                    raise
                receipt = _quarantine_attempt(
                    lane,
                    command,
                    receipt_path,
                    receipt,
                    reason="invalid_terminal_manifest",
                )
        else:
            receipt = _quarantine_attempt(
                lane,
                command,
                receipt_path,
                receipt,
                reason="incomplete_attempt",
            )
        _ensure_attestation(command)
        environment = os.environ.copy()
        environment.update({str(k): str(v) for k, v in command["environment"].items()})
        log_path = Path(lane["log_dir"]) / f"{command['job_id']}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        receipt.update(
            {
                "status": "running",
                "attempts": int(receipt.get("attempts", 0)) + 1,
                "started_at": time.time(),
                "log": str(log_path),
            }
        )
        with log_path.open("ab", buffering=0) as log:
            process = subprocess.Popen(
                [lane["python"], *command["argv"]],
                cwd=lane["repo_dir"],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=(process_lock_fd,),
            )
            receipt["pid"] = process.pid
            _atomic_json(receipt_path, receipt)
            return_code = process.wait()
        if return_code != 0:
            receipt.update({"status": "failed", "return_code": return_code, "ended_at": time.time()})
            _atomic_json(receipt_path, receipt)
            raise SupervisorError(f"job {command['job_id']} exited {return_code}")
        return _mark_completed(receipt_path, receipt, command)


def run_lane(path: Path) -> dict[str, Any]:
    lane = load_lane(path)
    with _lock(Path(lane["lane_lock"]), blocking=False):
        receipts = [_run_job(lane, command) for command in lane["commands"]]
    return {"worker_id": lane["worker_id"], "status": "complete", "receipts": receipts}


def status_lane(path: Path) -> dict[str, Any]:
    lane = load_lane(path)
    jobs = []
    for command in lane["commands"]:
        receipt_path = _receipt_path(lane, command["job_id"])
        receipt = _load(receipt_path) if receipt_path.exists() else None
        jobs.append(
            {
                "job_id": command["job_id"],
                "category": command["category"],
                "status": "pending" if receipt is None else receipt.get("status", "invalid"),
                "pid_alive": False if receipt is None else _pid_is_exact(receipt.get("pid"), lane, command),
            }
        )
    return {"worker_id": lane["worker_id"], "host_alias": lane["host_alias"], "gpu": lane["gpu"], "jobs": jobs}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "status"):
        item = sub.add_parser(name)
        item.add_argument("--lane", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = run_lane(args.lane) if args.command == "run" else status_lane(args.lane)
    except SupervisorError as error:
        print(f"REFUSING: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
