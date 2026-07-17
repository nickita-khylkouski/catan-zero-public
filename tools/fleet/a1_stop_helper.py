#!/usr/bin/env python3
"""Stop one sealed A1 lane without pattern matching or touching other GPU work.

The production executor runs this helper on the owning host.  A process is a
signal target only when its complete argv and detached session/process-group
identity match the immutable lane payload.  The supervisor is frozen before
the final child discovery so it cannot start another category while the stop
is in progress.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.fleet import a1_lane_supervisor as supervisor  # noqa: E402


class StopError(RuntimeError):
    """The lane cannot be stopped without risking an unrelated process."""


def _cmdline(pid: int) -> list[str] | None:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return None
    return [part.decode(errors="surrogateescape") for part in raw.split(b"\0") if part]


def _ppid(pid: int) -> int | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        return int(fields[3])
    except (OSError, UnicodeError, ValueError, IndexError):
        return None


def _iter_pids() -> list[int]:
    result: list[int] = []
    for item in Path("/proc").iterdir():
        if item.name.isdigit():
            result.append(int(item.name))
    return result


def _matches(pid: int, expected: Sequence[str]) -> bool:
    return _cmdline(pid) == list(expected)


def _detached_identity(pid: int, expected: Sequence[str], *, role: str) -> None:
    actual = _cmdline(pid)
    if actual is None:
        return
    if actual != list(expected):
        raise StopError(
            f"{role} PID {pid} argv drift; refusing to signal a reused/unrelated PID"
        )
    try:
        sid, pgid = os.getsid(pid), os.getpgid(pid)
    except ProcessLookupError:
        return
    if sid != pid or pgid != pid:
        raise StopError(
            f"{role} PID {pid} is not its exact detached SID/PGID "
            f"(sid={sid}, pgid={pgid})"
        )


def _expected_supervisor(lane: dict[str, Any], lane_path: Path) -> list[str]:
    return [
        lane["python"],
        str(Path(lane["repo_dir"]) / "tools/fleet/a1_lane_supervisor.py"),
        "run",
        "--lane",
        str(lane_path),
    ]


def _find_exact(expected: Sequence[str]) -> list[int]:
    return [pid for pid in _iter_pids() if _matches(pid, expected)]


def _receipt_pids(lane: dict[str, Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    for command in lane["commands"]:
        path = Path(lane["receipt_dir"]) / f"{command['job_id']}.json"
        if not path.exists():
            continue
        receipt = supervisor._load(path)
        if receipt.get("schema_version") != supervisor.RECEIPT_SCHEMA:
            raise StopError(f"job receipt schema drift: {path}")
        if receipt.get("lane_sha256") != lane["lane_sha256"]:
            raise StopError(f"job receipt lane drift: {path}")
        if receipt.get("argv_sha256") != command["argv_sha256"]:
            raise StopError(f"job receipt argv drift: {path}")
        # A completed receipt keeps its historical PID for provenance.  Do not
        # let later PID reuse by an unrelated process turn a clean stop into a
        # false target/refusal; exact live command discovery still catches any
        # impossible duplicate generator with the completed argv.
        pid = None if receipt.get("status") == "complete" else receipt.get("pid")
        if isinstance(pid, int) and not isinstance(pid, bool) and pid > 1:
            result[str(command["job_id"])] = pid
    return result


def inspect_lane(lane_path: Path, supervisor_pid: int | None) -> dict[str, Any]:
    lane_path = lane_path.resolve()
    lane = supervisor.load_lane(lane_path)
    expected_supervisor = _expected_supervisor(lane, lane_path)
    supervisor_matches = _find_exact(expected_supervisor)
    if supervisor_pid is not None and supervisor_pid > 1:
        if _cmdline(supervisor_pid) is not None:
            _detached_identity(
                supervisor_pid, expected_supervisor, role="lane supervisor"
            )
            if supervisor_pid not in supervisor_matches:
                raise StopError("recorded supervisor PID discovery drift")
    if len(supervisor_matches) > 1:
        raise StopError("multiple exact supervisors exist for one lane")
    discovered_supervisor = supervisor_matches[0] if supervisor_matches else None

    receipts = _receipt_pids(lane)
    generators: dict[str, int] = {}
    for command in lane["commands"]:
        expected = [lane["python"], *command["argv"]]
        matches = _find_exact(expected)
        if len(matches) > 1:
            raise StopError(f"multiple exact generators exist for {command['job_id']}")
        recorded = receipts.get(str(command["job_id"]))
        if recorded is not None and _cmdline(recorded) is not None:
            _detached_identity(recorded, expected, role=str(command["job_id"]))
            if recorded not in matches:
                raise StopError(f"recorded generator PID discovery drift for {command['job_id']}")
        if matches:
            pid = matches[0]
            _detached_identity(pid, expected, role=str(command["job_id"]))
            generators[str(command["job_id"])] = pid
    if len(generators) > 1:
        raise StopError("one lane has multiple live category generators")
    return {
        "worker_id": lane["worker_id"],
        "supervisor_pid": discovered_supervisor,
        "generator_pids": generators,
        "status": "active" if discovered_supervisor or generators else "idle",
    }


def _alive_exact(pid: int, expected: Sequence[str]) -> bool:
    return _matches(pid, expected)


def _signal_exact(pid: int, expected: Sequence[str], sig: signal.Signals, *, role: str) -> bool:
    if _cmdline(pid) is None:
        return False
    _detached_identity(pid, expected, role=role)
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        return False
    return True


def _wait_gone(targets: list[tuple[int, list[str]]], timeout: float) -> list[tuple[int, list[str]]]:
    deadline = time.monotonic() + timeout
    remaining = list(targets)
    while remaining and time.monotonic() < deadline:
        remaining = [(pid, argv) for pid, argv in remaining if _alive_exact(pid, argv)]
        if remaining:
            time.sleep(0.05)
    return [(pid, argv) for pid, argv in remaining if _alive_exact(pid, argv)]


def stop_lane(
    lane_path: Path,
    supervisor_pid: int | None,
    *,
    term_timeout: float = 10.0,
    kill_timeout: float = 5.0,
) -> dict[str, Any]:
    lane_path = lane_path.resolve()
    lane = supervisor.load_lane(lane_path)
    initial = inspect_lane(lane_path, supervisor_pid)
    expected_supervisor = _expected_supervisor(lane, lane_path)
    sup = initial["supervisor_pid"]

    # Freeze the exact supervisor first.  This closes the receipt/spawn race:
    # after SIGSTOP it cannot launch the next category while we rescan every
    # immutable command argv and receipt.
    if sup is not None:
        _signal_exact(sup, expected_supervisor, signal.SIGSTOP, role="lane supervisor")
    try:
        frozen = inspect_lane(lane_path, sup)
        targets: list[tuple[int, list[str]]] = []
        for command in lane["commands"]:
            pid = frozen["generator_pids"].get(str(command["job_id"]))
            if pid is not None:
                targets.append((pid, [lane["python"], *command["argv"]]))
        for pid, argv in targets:
            _signal_exact(pid, argv, signal.SIGTERM, role="lane generator")
        # Stop the supervisor too, even if a child just completed naturally;
        # otherwise it could advance to the next category after SIGCONT.
        if sup is not None:
            _signal_exact(sup, expected_supervisor, signal.SIGTERM, role="lane supervisor")
            try:
                os.killpg(sup, signal.SIGCONT)
            except ProcessLookupError:
                pass
        all_targets = targets + ([] if sup is None else [(sup, expected_supervisor)])
        remaining = _wait_gone(all_targets, term_timeout)
        killed: list[int] = []
        for pid, argv in remaining:
            if _signal_exact(pid, argv, signal.SIGKILL, role="A1 stop survivor"):
                killed.append(pid)
        remaining = _wait_gone(remaining, kill_timeout)
        if remaining:
            raise StopError(f"exact A1 process(es) survived SIGKILL: {[pid for pid, _ in remaining]}")
    except BaseException:
        if sup is not None and _alive_exact(sup, expected_supervisor):
            try:
                os.killpg(sup, signal.SIGCONT)
            except ProcessLookupError:
                pass
        raise
    final = inspect_lane(lane_path, supervisor_pid)
    if final["status"] != "idle":
        raise StopError("lane still has an exact live process after stop")
    return {
        "worker_id": lane["worker_id"],
        "status": "stopped",
        "term_targets": [pid for pid, _ in all_targets],
        "kill_targets": killed,
        "gpu_runtime_preserved": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect", "stop"))
    parser.add_argument("--lane", required=True, type=Path)
    parser.add_argument("--supervisor-pid", type=int, default=0)
    parser.add_argument("--term-timeout", type=float, default=10.0)
    parser.add_argument("--kill-timeout", type=float, default=5.0)
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            result = inspect_lane(args.lane, args.supervisor_pid or None)
        else:
            result = stop_lane(
                args.lane,
                args.supervisor_pid or None,
                term_timeout=args.term_timeout,
                kill_timeout=args.kill_timeout,
            )
    except (StopError, supervisor.SupervisorError) as error:
        print(f"REFUSING: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
