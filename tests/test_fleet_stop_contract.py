"""Behavioral contract for MPS-safe fleet teardown.

The canonical remote routine is exercised with real Linux sessions/process
groups and a fake nvidia-smi/MPS control surface.  This catches the production
failure mode that source-string assertions missed: under MPS, NVML exposes the
server rather than generator clients, so teardown must work from the recorded
launch_detached PGID even when the compute-PID query is empty.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
STOP = ROOT / "tools" / "fleet" / "fleet_stop.sh"


def _remote_routine() -> str:
    source = STOP.read_text(encoding="utf-8")
    match = re.search(
        r"read -r -d '' REMOTE <<'REMOTE_EOF' \|\| true\n(.*?)\nREMOTE_EOF",
        source,
        flags=re.DOTALL,
    )
    assert match is not None
    return match.group(1)


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def _fake_host(tmp_path: Path, *, memory_mib: int = 0, mps_client: bool = False) -> dict[str, str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "nvidia-smi",
        """#!/usr/bin/env bash
case "$*" in
  *--query-compute-apps=*) exit 0 ;;
  *--query-gpu=index,memory.used*) printf '0, %s\\n' "${FAKE_GPU_MEMORY:-0}" ;;
  *) exit 1 ;;
esac
""",
    )
    mps_pipe = tmp_path / "mps"
    mps_pipe.mkdir()
    if mps_client:
        (mps_pipe / "control").touch()
        _write_executable(
            fake_bin / "nvidia-cuda-mps-control",
            """#!/usr/bin/env bash
read -r command rest
case "$command" in
  get_server_list) echo 41001 ;;
  get_client_list) echo 41002 ;;
esac
""",
        )
    env = os.environ.copy()
    env.update(
        {
            "HOME": str(tmp_path),
            "PATH": f"{fake_bin}:{env['PATH']}",
            "FAKE_GPU_MEMORY": str(memory_mib),
            "CUDA_MPS_PIPE_DIRECTORY": str(mps_pipe),
            "FLEET_STOP_POLL_SECONDS": "0.02",
            "FLEET_STOP_RELEASE_SECONDS": "0.02",
            "FLEET_STOP_MEMORY_POLL_SECONDS": "0.02",
        }
    )
    return env


def _run_remote(tmp_path: Path, *, go: bool, memory_mib: int = 0, mps_client: bool = False):
    env = _fake_host(tmp_path, memory_mib=memory_mib, mps_client=mps_client)
    return subprocess.run(
        ["bash", "-s", "--", "1" if go else "0"],
        input=_remote_routine(),
        text=True,
        capture_output=True,
        env=env,
        timeout=20,
    )


def _linux_process_group_or_skip(tmp_path: Path, *, canonical: bool) -> subprocess.Popen[str]:
    if not sys.platform.startswith("linux") or shutil.which("setsid") is None:
        pytest.skip("behavioral PGID teardown contract requires Linux setsid/ps sid support")
    name = "run_generation.sh" if canonical else "unrelated_worker.sh"
    runner = tmp_path / name
    _write_executable(
        runner,
        """#!/usr/bin/env bash
trap '' TERM INT
( trap '' TERM INT; while :; do sleep 1; done ) &
while :; do sleep 1; done
""",
    )
    proc = subprocess.Popen(
        ["setsid", str(runner)],
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            if os.getpgid(proc.pid) == proc.pid and os.getsid(proc.pid) == proc.pid:
                return proc
        except ProcessLookupError:
            break
        time.sleep(0.02)
    proc.kill()
    raise AssertionError("setsid test process did not become its own SID/PGID")


def _record_pid(tmp_path: Path, pid: int) -> None:
    run_dir = tmp_path / "fleet_runs" / "claim-test"
    run_dir.mkdir(parents=True)
    (run_dir / ".pid").write_text(f"{pid}\n", encoding="utf-8")


def _force_cleanup(proc: subprocess.Popen[str]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def test_stop_script_shell_syntax_is_valid() -> None:
    subprocess.run(["bash", "-n", str(STOP)], check=True)


def test_recorded_group_stops_mps_hidden_parent_and_grandchild(tmp_path: Path) -> None:
    proc = _linux_process_group_or_skip(tmp_path, canonical=True)
    _record_pid(tmp_path, proc.pid)
    try:
        result = _run_remote(tmp_path, go=True)
        assert result.returncode == 0, result.stdout + result.stderr
        assert f"TERM group {proc.pid}" in result.stdout
        assert f"KILL group {proc.pid}" in result.stdout
        proc.wait(timeout=5)
        with pytest.raises(ProcessLookupError):
            os.killpg(proc.pid, 0)
    finally:
        _force_cleanup(proc)


def test_stale_pid_file_cannot_kill_unrelated_reused_group(tmp_path: Path) -> None:
    proc = _linux_process_group_or_skip(tmp_path, canonical=False)
    _record_pid(tmp_path, proc.pid)
    try:
        result = _run_remote(tmp_path, go=True)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "ignoring stale/non-Catan group" in result.stdout
        assert proc.poll() is None
    finally:
        _force_cleanup(proc)


def test_go_fails_when_gpu_memory_remains(tmp_path: Path) -> None:
    result = _run_remote(tmp_path, go=True, memory_mib=128)
    assert result.returncode != 0
    assert "GPU memory remains above 50 MiB" in result.stderr


def test_stale_mps_control_socket_is_not_treated_as_running_daemon(tmp_path: Path) -> None:
    result = _run_remote(tmp_path, go=True, mps_client=True)
    assert result.returncode == 0
    assert "MPS daemon: not present" in result.stdout


def test_dry_run_is_non_mutating_even_when_work_is_visible(tmp_path: Path) -> None:
    proc = _linux_process_group_or_skip(tmp_path, canonical=True)
    _record_pid(tmp_path, proc.pid)
    try:
        result = _run_remote(tmp_path, go=False)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "DRY-RUN: nothing killed" in result.stdout
        assert proc.poll() is None
    finally:
        _force_cleanup(proc)
