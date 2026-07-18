from __future__ import annotations

import json
import signal
import sys
from pathlib import Path

import pytest

from tools.fleet import a1_lane_supervisor as supervisor
from tools.fleet import a1_production_executor as executor
from tools.fleet import a1_stop_helper as stop_helper


def _fixture(tmp_path: Path, *, ignore_generator_term: bool = False) -> tuple[Path, dict]:
    repo = tmp_path / "repo"
    script = repo / "tools/fleet/a1_lane_supervisor.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        """import signal,sys,time
if '--dummy-ignore-term' in sys.argv:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
while True:
    time.sleep(0.05)
""",
        encoding="utf-8",
    )
    lock = tmp_path / "contract.lock.json"
    render = tmp_path / "commands.json"
    lock.write_text("lock\n", encoding="utf-8")
    render.write_text("render\n", encoding="utf-8")
    commands = []
    previous = None
    for category in supervisor.CATEGORY_ORDER:
        job_id = f"c1_gpu0__{category}"
        argv = [
            str(script),
            "--out-dir",
            str(tmp_path / "output" / job_id),
            "--n-full",
            "128",
            "--resume",
            "--no-eval-server",
        ]
        if ignore_generator_term and category == "current_producer":
            argv.append("--dummy-ignore-term")
        out_dir = Path(argv[argv.index("--out-dir") + 1])
        environment = {
            **supervisor.SEALED_RUNTIME_ENVIRONMENT,
            "PYTHONPATH": f"{repo}/src:{repo}",
            "CUDA_VISIBLE_DEVICES": "0",
            **supervisor.CLIENT_ENVIRONMENT,
            "CATAN_SEED_LEDGER": str(tmp_path / "ledger.md"),
            "CATAN_A1_CONTRACT_SHA256": "sha256:test-contract",
            supervisor.CONFIG_REGISTRY_ENVIRONMENT_VARIABLE: str(
                out_dir / supervisor.CONFIG_REGISTRY_FILENAME
            ),
        }
        source_environment = {
            **environment,
            "PYTHONPATH": (
                f"{supervisor.RUNTIME_REPO_TOKEN}/src:"
                f"{supervisor.RUNTIME_REPO_TOKEN}"
            ),
        }
        provenance = {
            "pipeline": "generate",
            "config_hash": f"generate-{category}",
            "full_config_hash": f"generate-full-{category}",
            "config": {},
        }
        provenance["provenance_sha256"] = supervisor._digest(provenance)
        command = {
            "job_id": job_id,
            "worker_id": "c1_gpu0",
            "host_alias": "c1",
            "gpu": 0,
            "category": category,
            "argv": argv,
            "argv_sha256": supervisor._digest(argv),
            "render_argv_sha256": supervisor._digest(
                [
                    f"{supervisor.RUNTIME_REPO_TOKEN}/tools/fleet/"
                    "a1_lane_supervisor.py",
                    *argv[1:],
                ]
            ),
            "runtime_repo_argv_indices": [0],
            "environment": environment,
            "environment_sha256": supervisor._digest(environment),
            "render_environment_sha256": supervisor._digest(source_environment),
            "config_provenance": provenance,
            "must_run_after": [] if previous is None else [previous],
        }
        commands.append(command)
        previous = job_id
    lane = {
        "schema_version": supervisor.SCHEMA,
        "worker_id": "c1_gpu0",
        "host_alias": "c1",
        "gpu": 0,
        "repo_dir": str(repo),
        "python": sys.executable,
        "receipt_dir": str(tmp_path / "receipts"),
        "quarantine_dir": str(tmp_path / "quarantine"),
        "log_dir": str(tmp_path / "logs"),
        "lane_lock": str(tmp_path / "lane.lock"),
        "client_environment": dict(supervisor.CLIENT_ENVIRONMENT),
        "operator_manifests": {
            "lock": {"path": str(lock), "sha256": executor._sha256(lock)},
            "render": {"path": str(render), "sha256": executor._sha256(render)},
        },
        "commands": commands,
    }
    lane["lane_sha256"] = supervisor._digest(lane)
    path = tmp_path / "lane.json"
    path.write_text(json.dumps(lane), encoding="utf-8")
    return path, lane


def _receipt(lane: dict, command: dict, pid: int) -> None:
    path = Path(lane["receipt_dir"]) / f"{command['job_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": supervisor.RECEIPT_SCHEMA,
                "job_id": command["job_id"],
                "lane_sha256": lane["lane_sha256"],
                "argv_sha256": command["argv_sha256"],
                "status": "running",
                "pid": pid,
            }
        ),
        encoding="utf-8",
    )


def _fake_processes(monkeypatch: pytest.MonkeyPatch, processes: dict[int, dict]) -> list:
    signals: list[tuple[int, signal.Signals]] = []

    monkeypatch.setattr(stop_helper, "_iter_pids", lambda: sorted(processes))
    monkeypatch.setattr(
        stop_helper,
        "_cmdline",
        lambda pid: None if pid not in processes else list(processes[pid]["argv"]),
    )
    monkeypatch.setattr(
        stop_helper.os,
        "getsid",
        lambda pid: processes[pid]["sid"],
    )
    monkeypatch.setattr(
        stop_helper.os,
        "getpgid",
        lambda pid: processes[pid]["pgid"],
    )

    def killpg(pid: int, sig: signal.Signals) -> None:
        if pid not in processes:
            raise ProcessLookupError(pid)
        signals.append((pid, sig))
        if sig == signal.SIGKILL or (
            sig == signal.SIGTERM and not processes[pid].get("ignore_term", False)
        ):
            processes.pop(pid)

    monkeypatch.setattr(stop_helper.os, "killpg", killpg)
    return signals


def test_stop_targets_only_exact_detached_groups_and_preserves_unrelated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lane_path, lane = _fixture(tmp_path, ignore_generator_term=True)
    supervisor_argv = [
            sys.executable,
            str(Path(lane["repo_dir"]) / "tools/fleet/a1_lane_supervisor.py"),
            "run",
            "--lane",
            str(lane_path.resolve()),
        ]
    command = lane["commands"][0]
    processes = {
        101: {"argv": supervisor_argv, "sid": 101, "pgid": 101},
        202: {
            "argv": [sys.executable, *command["argv"]],
            "sid": 202,
            "pgid": 202,
            "ignore_term": True,
        },
        303: {
            "argv": [sys.executable, "-c", "unrelated"],
            "sid": 303,
            "pgid": 303,
        },
    }
    signals = _fake_processes(monkeypatch, processes)
    _receipt(lane, command, 202)
    report = stop_helper.stop_lane(
        lane_path, 101, term_timeout=0.0, kill_timeout=0.0
    )
    assert report["status"] == "stopped"
    assert report["gpu_runtime_preserved"] is True
    assert 202 in report["kill_targets"]
    assert 303 in processes
    assert not any(pid == 303 for pid, _signal in signals)


def test_reused_or_unrelated_receipt_pid_refuses_before_any_signal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lane_path, lane = _fixture(tmp_path)
    processes = {
        303: {
            "argv": [sys.executable, "-c", "unrelated"],
            "sid": 303,
            "pgid": 303,
        }
    }
    signals = _fake_processes(monkeypatch, processes)
    _receipt(lane, lane["commands"][0], 303)
    with pytest.raises(stop_helper.StopError, match="argv drift"):
        stop_helper.inspect_lane(lane_path, None)
    assert 303 in processes and signals == []


def test_exact_process_without_own_session_and_group_is_never_signalled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lane_path, lane = _fixture(tmp_path)
    command = lane["commands"][0]
    processes = {
        404: {
            "argv": [sys.executable, *command["argv"]],
            "sid": 1,
            "pgid": 1,
        }
    }
    signals = _fake_processes(monkeypatch, processes)
    _receipt(lane, command, 404)
    with pytest.raises(stop_helper.StopError, match="detached SID/PGID"):
        stop_helper.stop_lane(lane_path, None, term_timeout=0.0)
    assert 404 in processes and signals == []


def _plan() -> dict:
    lane = [{"host_alias": "c1", "gpu": 0}]
    return {
        "schema_version": executor.RECEIPT_SCHEMA,
        "contract_sha256": "sha256:" + "a" * 64,
        "plan_sha256": "sha256:" + "b" * 64,
        "repo_artifacts_sha256": "sha256:" + "c" * 64,
        "_private": {
            "hosts": {"remote_root": "/remote", "python": "/venv/bin/python"},
            "lanes": {"c1_gpu0": lane},
        },
    }


def test_executor_stop_is_dry_by_default_and_writes_resumable_stopped_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan()
    receipt_path = tmp_path / "executor.json"
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": executor.RECEIPT_SCHEMA,
                "plan_sha256": plan["plan_sha256"],
                "status": "launched",
                "lane_pids": {"c1_gpu0": 123},
                "launch_pending_worker_id": "c1_gpu0",
            }
        ),
        encoding="utf-8",
    )
    calls: list[str] = []

    def call(_plan, worker, _lane, *, action, supervisor_pid):
        calls.append(action)
        assert worker == "c1_gpu0" and supervisor_pid == 123
        return {
            "worker_id": worker,
            "status": "active" if action == "inspect" else "stopped",
            "gpu_runtime_preserved": True,
        }

    monkeypatch.setattr(executor, "_stop_helper_call", call)
    dry = executor.stop_execution(plan, receipt_path=receipt_path, go=False)
    assert dry["status"] == "stop_dry_run"
    assert json.loads(receipt_path.read_text())["status"] == "launched"
    assert calls == ["inspect"]

    stopped = executor.stop_execution(plan, receipt_path=receipt_path, go=True)
    assert calls == ["inspect", "inspect", "stop"]
    assert stopped["status"] == "stopped"
    assert stopped["gpu_runtime_preserved"] is True
    persisted = json.loads(receipt_path.read_text())
    assert persisted["status"] == "stopped"
    assert "launch_pending_worker_id" not in persisted
    # The existing exact receipt is deliberately accepted only with --resume;
    # the lane supervisor will quarantine incomplete output before replay.
    assert executor._resume_receipt(receipt_path, plan, resume=True)["status"] == "stopped"
    with pytest.raises(executor.ExecutorError, match="pass --resume"):
        executor._resume_receipt(receipt_path, plan, resume=False)
