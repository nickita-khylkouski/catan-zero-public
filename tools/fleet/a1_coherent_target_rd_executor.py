#!/usr/bin/env python3
"""Launch the sealed, self-play-only coherent-target R&D corpus on one host.

This executor is deliberately small and narrow.  It is not a replacement for
the production-wave control plane: it accepts only the authenticated
``a1-coherent-target-rd-contract-v1`` intervention, atomically claims its exact
seed lanes, pins one generator to each declared GPU, and records the launched
PIDs/argv.  The contract verifier forbids opponent mixing and adaptive n256 so
the resulting corpus answers one question: do coherent public-belief n128
targets train differently from the legacy PIMC targets?
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import resource
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
for import_root in (REPO_ROOT, REPO_ROOT / "tools"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from tools import a1_target_eligibility_inventory as identity  # noqa: E402
from tools.prelaunch_guard import parse_seed_ledger  # noqa: E402


class ExecutorError(RuntimeError):
    """The sealed R&D transaction cannot be launched exactly."""


REQUIRED_NOFILE_LIMIT = 65_536
LAUNCH_RECEIPT_SCHEMA = "a1-coherent-target-rd-launch-receipt-v1"
STATUS_SCHEMA = "a1-coherent-target-rd-status-v1"
COMPLETION_RECEIPT_SCHEMA = "a1-coherent-target-rd-completion-receipt-v1"
DEFAULT_STALE_SECONDS = 900.0


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExecutorError(f"cannot load {path}: {error}") from error
    if not isinstance(value, dict):
        raise ExecutorError(f"{path} must contain a JSON object")
    return value


def _durable_replace(path: Path, data: bytes, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_receipt(path: Path, value: Mapping[str, Any]) -> None:
    payload = dict(value)
    payload["receipt_sha256"] = _digest(payload)
    data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    if path.exists():
        if path.read_bytes() != data:
            raise ExecutorError(f"immutable receipt drift: {path}")
        return
    _durable_replace(path, data, mode=0o444)
    os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def _read_stable_json(path: Path) -> tuple[dict[str, Any], str, os.stat_result]:
    """Read one regular JSON file without accepting a symlink or torn read."""

    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or path.is_symlink():
            raise ExecutorError(
                f"JSON artifact is not a canonical regular file: {path}"
            )
        data = path.read_bytes()
        after = path.stat()
    except OSError as error:
        raise ExecutorError(f"cannot read JSON artifact {path}: {error}") from error
    identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(
        getattr(before, field) != getattr(after, field) for field in identity_fields
    ):
        raise ExecutorError(f"JSON artifact changed while being read: {path}")
    try:
        value = json.loads(data)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ExecutorError(f"malformed JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise ExecutorError(f"JSON artifact is not an object: {path}")
    return value, "sha256:" + hashlib.sha256(data).hexdigest(), after


def _authenticated_receipt(
    path: Path, *, expected_schema: str
) -> tuple[dict[str, Any], str]:
    value, file_sha256, _stat = _read_stable_json(path)
    unhashed = dict(value)
    declared = unhashed.pop("receipt_sha256", None)
    if declared != _digest(unhashed):
        raise ExecutorError(f"receipt semantic digest mismatch: {path}")
    if value.get("schema_version") != expected_schema:
        raise ExecutorError(
            f"receipt schema drift at {path}: {value.get('schema_version')!r}"
        )
    return value, file_sha256


def _overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _claim_rows(contract: Mapping[str, Any]) -> tuple[list[str], dict[str, Any]]:
    execution = contract["execution"]
    ledger = Path(str(execution["seed_ledger"])).expanduser().resolve(strict=True)
    contract_sha = str(contract["contract_sha256"])
    rows = [
        (
            int(lane["base_seed"]),
            int(lane["base_seed"]) + int(lane["games"]),
            str(lane["claim_label"]),
            str(lane["lane_id"]),
        )
        for lane in execution["lanes"]
    ]
    rendered = [
        f"[{start} – {end}) | target-identity-rd/{lane_id} "
        f"claim={claim} contract={contract_sha}"
        for start, end, claim, lane_id in rows
    ]
    sidecar = ledger.with_name(ledger.name + ".a1-target-rd.lock")
    descriptor = os.open(sidecar, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        before = ledger.read_bytes()
        if before and not before.endswith(b"\n"):
            raise ExecutorError(f"seed ledger {ledger} does not end with a newline")
        live = parse_seed_ledger(ledger)
        own_counts: list[int] = []
        for start, end, claim, lane_id in rows:
            requested = (start, end)
            token = f"claim={claim}"
            own = [
                row
                for row in live
                if (int(row[0]), int(row[1])) == requested
                and token in str(row[2]).split()
            ]
            if len(own) > 1:
                raise ExecutorError(f"ledger repeats own claim for {lane_id}")
            own_counts.append(len(own))
            collisions = [
                row
                for row in live
                if _overlap(requested, (int(row[0]), int(row[1]))) and row not in own
            ]
            if collisions:
                raise ExecutorError(
                    f"seed lane {lane_id} {requested} overlaps {collisions[:3]}"
                )
        present = sum(own_counts)
        if present not in (0, len(rows)):
            raise ExecutorError(
                f"refusing partial own seed-claim set: {present}/{len(rows)} present"
            )
        status = "already_claimed" if present == len(rows) else "claimed"
        after = before
        if present == 0:
            after = before + b"".join(line.encode("utf-8") + b"\n" for line in rendered)
            _durable_replace(ledger, after, mode=stat.S_IMODE(ledger.stat().st_mode))
        receipt = {
            "status": status,
            "ledger": str(ledger),
            "ledger_before_sha256": "sha256:" + hashlib.sha256(before).hexdigest(),
            "ledger_after_sha256": "sha256:" + hashlib.sha256(after).hexdigest(),
            "claim_count": len(rows),
            "claims_sha256": _digest(rendered),
        }
        return rendered, receipt
    finally:
        os.close(descriptor)


def _run_text(command: Sequence[str], *, cwd: Path | None = None) -> str:
    try:
        return subprocess.run(
            list(command),
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ExecutorError(f"command failed: {list(command)!r}: {error}") from error


def _non_mps_compute_processes(raw: str) -> list[str]:
    """Return active compute rows except the exact persistent MPS server.

    `nvidia-smi` reports the host-wide MPS server even while it has no clients.
    It is launch infrastructure, not a competing workload.  Match only the
    process basename so a malicious or unrelated name containing the token is
    never accidentally exempted.
    """

    busy: list[str] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",", 2)]
        if len(fields) != 3:
            busy.append(line.strip())
            continue
        process_name = Path(fields[1]).name
        if process_name == "nvidia-cuda-mps-server":
            continue
        busy.append(line.strip())
    return busy


def _python_executable(path: Path) -> Path:
    """Authenticate a venv interpreter without resolving away its prefix.

    Virtualenv Python entry points are commonly symlinks to the base
    interpreter.  Executing the resolved target silently drops the venv's
    site-packages, so retain the lexical absolute path after proving that its
    target exists and the entry point is executable.
    """

    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    try:
        target = lexical.resolve(strict=True)
    except OSError as error:
        raise ExecutorError(
            f"cannot resolve Python executable {lexical}: {error}"
        ) from error
    if not target.is_file() or not os.access(lexical, os.X_OK):
        raise ExecutorError(f"python is not executable: {lexical}")
    return lexical


def _ensure_worker_fd_limit() -> tuple[int, int]:
    """Raise the inherited soft fd limit required by multi-worker generation."""

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard < REQUIRED_NOFILE_LIMIT:
        raise ExecutorError(
            "hard RLIMIT_NOFILE is below the generator contract: "
            f"hard={hard} required={REQUIRED_NOFILE_LIMIT}"
        )
    if soft < REQUIRED_NOFILE_LIMIT:
        resource.setrlimit(resource.RLIMIT_NOFILE, (REQUIRED_NOFILE_LIMIT, hard))
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < REQUIRED_NOFILE_LIMIT:
        raise ExecutorError(
            "could not raise soft RLIMIT_NOFILE for generation: "
            f"soft={soft} required={REQUIRED_NOFILE_LIMIT}"
        )
    return int(soft), int(hard)


def _preflight(
    contract_path: Path,
    *,
    repo: Path,
    python: Path,
    host_address: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    verified = identity.inspect_rd_contract(contract_path)
    contract = _load(contract_path)
    execution = contract["execution"]
    if host_address != execution["host"]:
        raise ExecutorError(
            f"--host-address {host_address!r} does not match sealed host {execution['host']!r}"
        )
    repo = repo.expanduser().resolve(strict=True)
    python = _python_executable(python)
    contract_repo = contract_path.resolve(strict=True).parents[3]
    if repo != contract_repo:
        raise ExecutorError(
            f"--repo {repo} differs from the repository authenticated by the "
            f"contract path ({contract_repo})"
        )
    if not os.access(python, os.X_OK):
        raise ExecutorError(f"python is not executable: {python}")
    generator = repo / "tools/generate_gumbel_selfplay_data.py"
    if not generator.is_file():
        raise ExecutorError(f"generator is missing: {generator}")
    checkpoint = Path(str(contract["producer_checkpoint"]["path"]))
    if _file_sha256(checkpoint) != contract["producer_checkpoint"]["sha256"]:
        raise ExecutorError(f"producer checkpoint hash drift: {checkpoint}")
    output_root = Path(str(execution["output_root"]))
    if output_root.exists():
        raise ExecutorError(f"fresh output root already exists: {output_root}")

    gpu_indices = {
        int(line.strip())
        for line in _run_text(
            [
                "nvidia-smi",
                "--query-gpu=index",
                "--format=csv,noheader,nounits",
            ]
        ).splitlines()
        if line.strip()
    }
    required_gpus = {int(lane["gpu"]) for lane in execution["lanes"]}
    if not required_gpus <= gpu_indices:
        raise ExecutorError(
            f"sealed GPUs are unavailable: required={sorted(required_gpus)}, "
            f"visible={sorted(gpu_indices)}"
        )
    compute_processes = _run_text(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    busy_processes = _non_mps_compute_processes(compute_processes)
    if busy_processes:
        raise ExecutorError(
            "refusing to stack coherent generation on active GPU work: "
            + "; ".join(busy_processes)
        )
    git_commit = _run_text(["git", "rev-parse", "HEAD"], cwd=repo)
    tracked_diff = _run_text(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo
    )
    return contract, {
        "verified_contract": verified,
        "repo": str(repo),
        "python": str(python),
        "generator": str(generator),
        "generator_sha256": _file_sha256(generator),
        "git_commit": git_commit,
        "tracked_diff_present": bool(tracked_diff),
        "checkpoint_sha256": _file_sha256(checkpoint),
        "required_gpus": sorted(required_gpus),
    }


def _argv(
    contract: Mapping[str, Any],
    lane: Mapping[str, Any],
    *,
    repo: Path,
    python: Path,
) -> list[str]:
    root = Path(str(contract["execution"]["output_root"]))
    output = root / str(lane["lane_id"])
    return [
        str(python),
        str(repo / "tools/generate_gumbel_selfplay_data.py"),
        "--config",
        str(repo / contract["artifacts"]["typed_generation_config"]["path"]),
        "--prelaunch-guard-config",
        str(repo / contract["artifacts"]["generation_guard"]["path"]),
        "--checkpoint",
        str(contract["producer_checkpoint"]["path"]),
        "--out-dir",
        str(output),
        "--base-seed",
        str(lane["base_seed"]),
        "--games",
        str(lane["games"]),
        "--workers",
        str(contract["execution"]["workers_per_gpu"]),
        "--ledger-claim-label",
        str(lane["claim_label"]),
        "--device",
        "cuda",
        "--preserve-search-evidence",
        "--dump-config",
        str(output / "config.registry.jsonl"),
        "--config-purpose",
        str(contract["contract_id"]),
    ]


def _proc_start_ticks(pid: int) -> int:
    try:
        text = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8")
    except OSError as error:
        raise ExecutorError(
            f"cannot read process identity for PID {pid}: {error}"
        ) from error
    closing = text.rfind(")")
    if closing < 0:
        raise ExecutorError(f"malformed /proc/{pid}/stat")
    fields_after_comm = text[closing + 2 :].split()
    # starttime is field 22; fields_after_comm begins at field 3.
    if len(fields_after_comm) <= 19:
        raise ExecutorError(f"truncated /proc/{pid}/stat")
    return int(fields_after_comm[19])


def _process_identity(
    pid: int,
    *,
    argv: Sequence[str],
    environment: Mapping[str, str],
) -> dict[str, Any]:
    boot_id = (
        Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    )
    value = {
        "boot_id": boot_id,
        "start_ticks": _proc_start_ticks(pid),
        "argv_sha256": _digest(list(argv)),
        "environment": {
            key: environment[key]
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "CATAN_LEDGER_CLAIM_ID",
                "CUDA_MPS_PIPE_DIRECTORY",
                "CUDA_MPS_LOG_DIRECTORY",
            )
        },
    }
    value["identity_sha256"] = _digest(value)
    return value


def _live_process_status(command: Mapping[str, Any]) -> dict[str, Any]:
    pid = int(command["pid"])
    root = Path("/proc") / str(pid)
    if not root.exists():
        return {"state": "exited", "pid": pid, "authenticated": False}
    try:
        cmdline = [
            token.decode("utf-8")
            for token in (root / "cmdline").read_bytes().split(b"\0")
            if token
        ]
        environment = {}
        for raw in (root / "environ").read_bytes().split(b"\0"):
            if b"=" not in raw:
                continue
            key, value = raw.split(b"=", 1)
            environment[key.decode("utf-8")] = value.decode("utf-8")
        start_ticks = _proc_start_ticks(pid)
        boot_id = (
            Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
        )
    except (OSError, UnicodeError, ExecutorError) as error:
        return {
            "state": "unverifiable",
            "pid": pid,
            "authenticated": False,
            "reason": str(error),
        }

    issues: list[str] = []
    if cmdline != list(command["argv"]):
        issues.append("argv_mismatch_or_pid_reuse")
    expected_environment = {
        "CUDA_VISIBLE_DEVICES": str(command["gpu"]),
        "CATAN_LEDGER_CLAIM_ID": str(command["claim_label"]),
        "CUDA_MPS_PIPE_DIRECTORY": str(
            command["mps_environment"]["CUDA_MPS_PIPE_DIRECTORY"]
        ),
        "CUDA_MPS_LOG_DIRECTORY": str(
            command["mps_environment"]["CUDA_MPS_LOG_DIRECTORY"]
        ),
    }
    for key, expected in expected_environment.items():
        if environment.get(key) != expected:
            issues.append(f"environment_mismatch:{key}")
    sealed = command.get("process_identity")
    if isinstance(sealed, Mapping):
        unhashed = dict(sealed)
        declared = unhashed.pop("identity_sha256", None)
        if declared != _digest(unhashed):
            issues.append("sealed_process_identity_digest_mismatch")
        if (
            sealed.get("boot_id") != boot_id
            or int(sealed.get("start_ticks", -1)) != start_ticks
        ):
            issues.append("pid_reuse_identity_mismatch")
        if sealed.get("argv_sha256") != _digest(cmdline):
            issues.append("sealed_process_argv_mismatch")
    return {
        "state": "alive_authenticated" if not issues else "alive_mismatch",
        "pid": pid,
        "authenticated": not issues,
        "authentication_strength": (
            "boot_start_argv_environment"
            if isinstance(sealed, Mapping)
            else "argv_environment"
        ),
        "start_ticks": start_ticks,
        "issues": issues,
    }


def execute(
    contract_path: Path,
    *,
    repo: Path,
    python: Path,
    host_address: str,
    go: bool,
) -> dict[str, Any]:
    contract_path = contract_path.expanduser().resolve(strict=True)
    contract, preflight = _preflight(
        contract_path, repo=repo, python=python, host_address=host_address
    )
    repo = repo.expanduser().resolve(strict=True)
    # Keep the authenticated lexical venv entry point. Resolving this symlink
    # to the base interpreter drops the venv's site-packages at launch time.
    python = _python_executable(python)
    commands = [
        {
            "lane_id": lane["lane_id"],
            "gpu": int(lane["gpu"]),
            "argv": _argv(contract, lane, repo=repo, python=python),
        }
        for lane in contract["execution"]["lanes"]
    ]
    plan = {
        "schema_version": "a1-coherent-target-rd-launch-receipt-v1",
        "status": "dry_run" if not go else "launching",
        "contract": {
            "path": str(contract_path),
            "file_sha256": _file_sha256(contract_path),
            "contract_sha256": contract["contract_sha256"],
        },
        "preflight": preflight,
        "commands": commands,
    }
    if not go:
        plan["plan_sha256"] = _digest(plan)
        return plan

    execution = contract["execution"]
    service = str(execution["mps_service"])
    active = (
        subprocess.run(
            ["systemctl", "is-active", "--quiet", service], check=False
        ).returncode
        == 0
    )
    if not active:
        _run_text(["sudo", "-n", "systemctl", "start", service])
    if (
        subprocess.run(
            ["systemctl", "is-active", "--quiet", service], check=False
        ).returncode
        != 0
    ):
        raise ExecutorError(f"MPS service is not active: {service}")

    nofile_soft, nofile_hard = _ensure_worker_fd_limit()

    _rendered_claims, claim_receipt = _claim_rows(contract)
    output_root = Path(str(execution["output_root"]))
    output_root.mkdir(parents=True, exist_ok=False)
    processes: list[tuple[subprocess.Popen[bytes], Any]] = []
    launched: list[dict[str, Any]] = []
    base_env = os.environ.copy()
    base_env.update(
        {str(key): str(value) for key, value in execution["mps_environment"].items()}
    )
    base_env["CATAN_SEED_LEDGER"] = str(execution["seed_ledger"])
    base_env["PYTHONUNBUFFERED"] = "1"
    import_roots = [str(repo / "src"), str(repo / "tools")]
    inherited_pythonpath = base_env.get("PYTHONPATH")
    if inherited_pythonpath:
        import_roots.append(inherited_pythonpath)
    base_env["PYTHONPATH"] = os.pathsep.join(import_roots)
    try:
        for command in commands:
            lane_id = str(command["lane_id"])
            log_path = output_root / f"{lane_id}.log"
            log_handle = log_path.open("xb")
            environment = dict(base_env)
            environment["CUDA_VISIBLE_DEVICES"] = str(command["gpu"])
            environment["CATAN_LEDGER_CLAIM_ID"] = str(
                next(
                    lane["claim_label"]
                    for lane in execution["lanes"]
                    if lane["lane_id"] == lane_id
                )
            )
            process = subprocess.Popen(
                command["argv"],
                cwd=repo,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append((process, log_handle))
            process_identity = _process_identity(
                process.pid,
                argv=command["argv"],
                environment=environment,
            )
            launched.append(
                {
                    **command,
                    "pid": process.pid,
                    "log": str(log_path),
                    "out_dir": str(output_root / lane_id),
                    "claim_label": environment["CATAN_LEDGER_CLAIM_ID"],
                    "mps_environment": {
                        key: environment[key]
                        for key in (
                            "CUDA_MPS_PIPE_DIRECTORY",
                            "CUDA_MPS_LOG_DIRECTORY",
                        )
                    },
                    "process_identity": process_identity,
                }
            )
        time.sleep(2.0)
        early = [
            item
            for item, (process, _handle) in zip(launched, processes)
            if process.poll() is not None
        ]
        if early:
            raise ExecutorError(
                f"generator exited during launch preamble: {[item['lane_id'] for item in early]}"
            )
    except BaseException:
        for process, _handle in processes:
            if process.poll() is None:
                process.terminate()
        raise
    finally:
        for _process, handle in processes:
            handle.close()

    plan.update(
        {
            "status": "launched",
            "launched_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "mps_service": service,
            "rlimit_nofile": {"soft": nofile_soft, "hard": nofile_hard},
            "claim_receipt": claim_receipt,
            "commands": launched,
        }
    )
    receipt_path = output_root / "launch.receipt.json"
    _write_receipt(receipt_path, plan)
    plan["receipt"] = str(receipt_path)
    return plan


def _authenticate_launch(
    contract_path: Path,
    *,
    host_address: str,
) -> tuple[dict[str, Any], dict[str, Any], str, list[dict[str, Any]]]:
    identity.inspect_rd_contract(contract_path)
    contract = _load(contract_path)
    if str(contract["execution"]["host"]) != host_address:
        raise ExecutorError("status host does not match the sealed execution host")
    output_root = Path(str(contract["execution"]["output_root"]))
    launch_path = output_root / "launch.receipt.json"
    launch, launch_file_sha256 = _authenticated_receipt(
        launch_path, expected_schema=LAUNCH_RECEIPT_SCHEMA
    )
    if launch.get("status") != "launched":
        raise ExecutorError(f"launch receipt is not launched: {launch.get('status')!r}")
    contract_record = launch.get("contract")
    if not isinstance(contract_record, Mapping):
        raise ExecutorError("launch receipt has no contract record")
    if contract_record.get("contract_sha256") != contract[
        "contract_sha256"
    ] or contract_record.get("file_sha256") != _file_sha256(contract_path):
        raise ExecutorError("launch receipt binds a different contract")
    preflight = launch.get("preflight")
    if not isinstance(preflight, Mapping):
        raise ExecutorError("launch receipt has no preflight identity")
    if preflight.get("checkpoint_sha256") != contract["producer_checkpoint"]["sha256"]:
        raise ExecutorError("launch receipt checkpoint identity drift")
    launch_repo = Path(str(preflight.get("repo", "")))
    launch_python = Path(str(preflight.get("python", "")))
    raw_commands = launch.get("commands")
    if not isinstance(raw_commands, list) or len(raw_commands) != len(
        contract["execution"]["lanes"]
    ):
        raise ExecutorError("launch receipt lane count drift")
    by_lane = {
        str(item.get("lane_id")): item
        for item in raw_commands
        if isinstance(item, Mapping)
    }
    if len(by_lane) != len(raw_commands):
        raise ExecutorError("launch receipt contains duplicate/malformed lane commands")

    commands: list[dict[str, Any]] = []
    for lane in contract["execution"]["lanes"]:
        lane_id = str(lane["lane_id"])
        raw = by_lane.get(lane_id)
        if raw is None:
            raise ExecutorError(f"launch receipt omits lane {lane_id}")
        expected_argv = _argv(contract, lane, repo=launch_repo, python=launch_python)
        expected_output = output_root / lane_id
        if (
            list(raw.get("argv", ())) != expected_argv
            or int(raw.get("gpu", -1)) != int(lane["gpu"])
            or str(raw.get("out_dir")) != str(expected_output)
            or str(raw.get("log")) != str(output_root / f"{lane_id}.log")
            or not isinstance(raw.get("pid"), int)
            or int(raw["pid"]) <= 0
        ):
            raise ExecutorError(f"launch receipt command drift for {lane_id}")
        augmented = dict(raw)
        augmented["claim_label"] = str(lane["claim_label"])
        augmented["mps_environment"] = dict(contract["execution"]["mps_environment"])
        commands.append(augmented)
    return contract, launch, launch_file_sha256, commands


def _parse_time(value: str) -> dt.datetime:
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError as error:
        raise ExecutorError(f"invalid receipt timestamp {value!r}") from error
    if parsed.tzinfo is None:
        raise ExecutorError(f"receipt timestamp is timezone-naive: {value!r}")
    return parsed.astimezone(dt.timezone.utc)


def _expected_worker_games(total_games: int, workers: int, index: int) -> int:
    return total_games // workers + (1 if index < total_games % workers else 0)


def _progress_snapshot(
    out_dir: Path,
    *,
    lane: Mapping[str, Any],
    workers: int,
    observed_at: dt.datetime,
    launched_at: dt.datetime,
    stale_seconds: float,
) -> dict[str, Any]:
    worker_records: list[dict[str, Any]] = []
    totals = {
        "games_completed": 0,
        "games_failed": 0,
        "games_truncated": 0,
        "rows": 0,
        "simulations_used_total": 0,
    }
    cursor = 0
    for index in range(workers):
        worker_id = f"worker_{index:03d}"
        expected_games = _expected_worker_games(int(lane["games"]), workers, index)
        path = out_dir / worker_id / "progress.json"
        if not path.exists():
            age = max(0.0, (observed_at - launched_at).total_seconds())
            worker_records.append(
                {
                    "worker_id": worker_id,
                    "state": "missing_stale" if age > stale_seconds else "starting",
                    "expected_games": expected_games,
                    "game_index_start": cursor,
                    "progress": str(path),
                    "activity_age_seconds": age,
                }
            )
            cursor += expected_games
            continue
        progress, progress_sha256, metadata = _read_stable_json(path)
        issues: list[str] = []
        if int(progress.get("games_requested", -1)) != expected_games:
            issues.append("games_requested_drift")
        if int(progress.get("game_index_start", -1)) != cursor:
            issues.append("game_index_start_drift")
        if int(progress.get("base_seed", -1)) != int(lane["base_seed"]):
            issues.append("base_seed_drift")
        completed = int(
            progress.get("games_succeeded", progress.get("games_completed_local", 0))
        )
        failed = int(progress.get("games_failed", 0))
        truncated = int(progress.get("games_truncated", 0))
        rows = int(progress.get("rows_confirmed", progress.get("rows", 0)))
        simulations = int(progress.get("simulations_used_total", 0))
        if completed < 0 or completed > expected_games:
            issues.append("games_completed_out_of_range")
        if failed < 0 or truncated < 0 or rows < 0 or simulations < 0:
            issues.append("negative_progress_counter")
        if progress.get("errors") not in ([], None):
            issues.append("worker_errors")
        totals["games_completed"] += completed
        totals["games_failed"] += failed
        totals["games_truncated"] += truncated
        totals["rows"] += rows
        totals["simulations_used_total"] += simulations
        mtime = dt.datetime.fromtimestamp(metadata.st_mtime, tz=dt.timezone.utc)
        age = max(0.0, (observed_at - mtime).total_seconds())
        complete = completed == expected_games and failed == 0 and truncated == 0
        if issues or failed or truncated:
            state = "failed"
        elif complete:
            state = "complete_pending_lane_manifest"
        elif age > stale_seconds:
            state = "stale"
        else:
            state = "running"
        worker_records.append(
            {
                "worker_id": worker_id,
                "state": state,
                "expected_games": expected_games,
                "game_index_start": cursor,
                "games_completed": completed,
                "games_failed": failed,
                "games_truncated": truncated,
                "rows": rows,
                "simulations_used_total": simulations,
                "progress": str(path),
                "progress_sha256": progress_sha256,
                "confirmed_shards": len(progress.get("confirmed_shards", ())),
                "activity_age_seconds": age,
                "issues": issues,
            }
        )
        cursor += expected_games
    return {"workers": worker_records, "totals": totals}


def _validate_lane_manifest(
    path: Path,
    *,
    lane: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> tuple[dict[str, Any], str, list[str]]:
    manifest, manifest_sha256, _metadata = _read_stable_json(path)
    issues: list[str] = []
    expected = {
        "base_seed": int(lane["base_seed"]),
        "games_requested": int(lane["games"]),
        "games_completed": int(lane["games"]),
        "games_failed": 0,
        "games_truncated": 0,
        "target_information_regime": contract["target_information_regime"],
        "search_evidence_schema": contract["acceptance"][
            "require_search_evidence_schema"
        ],
        "producer_checkpoint_sha256": contract["producer_checkpoint"]["sha256"],
        "workers": int(contract["execution"]["workers_per_gpu"]),
    }
    for key, expected_value in expected.items():
        if manifest.get(key) != expected_value:
            issues.append(f"{key}_drift")
    if manifest.get("errors") not in ([], None):
        issues.append("lane_errors")
    if int(manifest.get("rows", 0)) <= 0:
        issues.append("empty_rows")
    if int(manifest.get("simulations_used_total", 0)) <= 0:
        issues.append("empty_simulations")
    cli = manifest.get("cli_args")
    required_cli = {
        "public_observation": True,
        "coherent_public_belief_search": True,
        "information_set_search": False,
        "determinization_particles": 1,
        "n_full": 128,
        "n_fast": 16,
        "p_full": 0.25,
        "n_full_wide": None,
        "n_full_wide_threshold": None,
        "wide_roots_always_full": False,
        "opponent_mix_manifest": None,
        "opponent_pool_manifest": None,
        "preserve_search_evidence": True,
        "record_automatic_transitions": False,
        "meaningful_public_history": True,
    }
    if not isinstance(cli, Mapping):
        issues.append("missing_cli_args")
    else:
        issues.extend(
            f"cli_{key}_drift"
            for key, expected_value in required_cli.items()
            if cli.get(key) != expected_value
        )
    return manifest, manifest_sha256, issues


def status(
    contract_path: Path,
    *,
    host_address: str,
    stale_seconds: float = DEFAULT_STALE_SECONDS,
    now: dt.datetime | None = None,
) -> dict[str, Any]:
    if stale_seconds <= 0.0:
        raise ExecutorError("--stale-seconds must be positive")
    contract_path = contract_path.expanduser().resolve(strict=True)
    contract, launch, launch_file_sha256, commands = _authenticate_launch(
        contract_path, host_address=host_address
    )
    observed_at = now or dt.datetime.now(dt.timezone.utc)
    if observed_at.tzinfo is None:
        raise ExecutorError("status observation time must be timezone-aware")
    observed_at = observed_at.astimezone(dt.timezone.utc)
    launched_at = _parse_time(str(launch["launched_at"]))
    execution = contract["execution"]
    output_root = Path(str(execution["output_root"]))
    lanes_by_id = {str(lane["lane_id"]): lane for lane in execution["lanes"]}
    lane_results: list[dict[str, Any]] = []
    aggregate = {
        "games_requested": int(execution["total_games"]),
        "games_completed": 0,
        "games_failed": 0,
        "games_truncated": 0,
        "rows": 0,
        "simulations_used_total": 0,
    }
    for command in commands:
        lane_id = str(command["lane_id"])
        lane = lanes_by_id[lane_id]
        out_dir = Path(str(command["out_dir"]))
        process = _live_process_status(command)
        progress = _progress_snapshot(
            out_dir,
            lane=lane,
            workers=int(execution["workers_per_gpu"]),
            observed_at=observed_at,
            launched_at=launched_at,
            stale_seconds=stale_seconds,
        )
        manifest_path = out_dir / "manifest.json"
        manifest_record: dict[str, Any] | None = None
        issues: list[str] = []
        if manifest_path.exists():
            manifest, manifest_sha256, manifest_issues = _validate_lane_manifest(
                manifest_path, lane=lane, contract=contract
            )
            issues.extend(manifest_issues)
            manifest_record = {
                "path": str(manifest_path),
                "sha256": manifest_sha256,
                "games_completed": int(manifest.get("games_completed", 0)),
                "games_failed": int(manifest.get("games_failed", 0)),
                "games_truncated": int(manifest.get("games_truncated", 0)),
                "rows": int(manifest.get("rows", 0)),
                "simulations_used_total": int(
                    manifest.get("simulations_used_total", 0)
                ),
            }
            counters = manifest_record
        else:
            counters = progress["totals"]
        for key in (
            "games_completed",
            "games_failed",
            "games_truncated",
            "rows",
            "simulations_used_total",
        ):
            aggregate[key] += int(counters[key])

        worker_states = {item["state"] for item in progress["workers"]}
        if manifest_record is None and process["state"] in {
            "alive_mismatch",
            "unverifiable",
        }:
            issues.append("launch_pid_authentication_failed")
        if issues or "failed" in worker_states:
            lane_state = "failed"
        elif manifest_record is not None:
            lane_state = "complete"
        elif process["state"] == "exited":
            issues.append("generator_exited_without_final_manifest")
            lane_state = "failed"
        elif "stale" in worker_states or "missing_stale" in worker_states:
            lane_state = "stale"
        else:
            lane_state = "running"
        lane_results.append(
            {
                "lane_id": lane_id,
                "gpu": int(lane["gpu"]),
                "base_seed": int(lane["base_seed"]),
                "games_requested": int(lane["games"]),
                "state": lane_state,
                "process": process,
                "progress": progress,
                "manifest": manifest_record,
                "issues": issues,
            }
        )

    failed = [item["lane_id"] for item in lane_results if item["state"] == "failed"]
    stale = [item["lane_id"] for item in lane_results if item["state"] == "stale"]
    complete = [item["lane_id"] for item in lane_results if item["state"] == "complete"]
    if failed:
        fleet_state = "failed"
    elif stale:
        fleet_state = "stale"
    elif len(complete) == len(lane_results):
        fleet_state = "complete_uncollected"
    else:
        fleet_state = "running"
    completion_path = output_root / "completion.receipt.json"
    completion: dict[str, Any] | None = None
    if completion_path.exists():
        receipt, receipt_file_sha256 = _authenticated_receipt(
            completion_path, expected_schema=COMPLETION_RECEIPT_SCHEMA
        )
        if (
            receipt.get("contract", {}).get("contract_sha256")
            != contract["contract_sha256"]
            or receipt.get("launch_receipt", {}).get("file_sha256")
            != launch_file_sha256
        ):
            raise ExecutorError("completion receipt binds a different launch")
        completion = {
            "path": str(completion_path),
            "file_sha256": receipt_file_sha256,
            "receipt_sha256": receipt["receipt_sha256"],
        }
        if fleet_state == "complete_uncollected":
            fleet_state = "complete"
    return {
        "schema_version": STATUS_SCHEMA,
        "state": fleet_state,
        "observed_at": observed_at.isoformat(),
        "stale_seconds": float(stale_seconds),
        "contract": {
            "path": str(contract_path),
            "file_sha256": _file_sha256(contract_path),
            "contract_sha256": contract["contract_sha256"],
        },
        "launch_receipt": {
            "path": str(output_root / "launch.receipt.json"),
            "file_sha256": launch_file_sha256,
            "receipt_sha256": launch["receipt_sha256"],
        },
        "totals": aggregate,
        "complete_lanes": complete,
        "failed_lanes": failed,
        "stale_lanes": stale,
        "lanes": lane_results,
        "completion_receipt": completion,
    }


def wait_for_completion(
    contract_path: Path,
    *,
    host_address: str,
    poll_seconds: float = 30.0,
    stale_seconds: float = DEFAULT_STALE_SECONDS,
    timeout_seconds: float = 0.0,
) -> dict[str, Any]:
    """Wait for the sealed launch and collect it exactly once when complete.

    This path never calls :func:`execute` and therefore cannot claim seeds or
    relaunch work. A failed/stale lane is surfaced immediately instead of
    silently waiting forever or fabricating a partial corpus.
    """

    if poll_seconds <= 0.0:
        raise ExecutorError("--poll-seconds must be positive")
    if timeout_seconds < 0.0:
        raise ExecutorError("--timeout-seconds cannot be negative")
    started = time.monotonic()
    while True:
        snapshot = status(
            contract_path,
            host_address=host_address,
            stale_seconds=stale_seconds,
        )
        totals = snapshot["totals"]
        print(
            json.dumps(
                {
                    "event": "coherent_target_wait",
                    "observed_at": snapshot["observed_at"],
                    "state": snapshot["state"],
                    "games_completed": totals["games_completed"],
                    "games_requested": totals["games_requested"],
                    "rows": totals["rows"],
                    "failed_lanes": snapshot["failed_lanes"],
                    "stale_lanes": snapshot["stale_lanes"],
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        if snapshot["state"] in {"complete_uncollected", "complete"}:
            return collect(
                contract_path,
                host_address=host_address,
                stale_seconds=stale_seconds,
            )
        if snapshot["state"] in {"failed", "stale"}:
            raise ExecutorError(
                "coherent corpus stopped making admissible progress: "
                f"state={snapshot['state']} failed={snapshot['failed_lanes']} "
                f"stale={snapshot['stale_lanes']}"
            )
        elapsed = time.monotonic() - started
        if timeout_seconds and elapsed >= timeout_seconds:
            raise ExecutorError(
                "timed out waiting for coherent corpus: "
                f"elapsed={elapsed:.1f}s games={totals['games_completed']}/"
                f"{totals['games_requested']}"
            )
        delay = poll_seconds
        if timeout_seconds:
            delay = min(delay, max(0.0, timeout_seconds - elapsed))
        time.sleep(delay)


def _resolved_config_record(
    path: Path,
    *,
    lane: Mapping[str, Any],
    contract: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    config, file_sha256, _metadata = _read_stable_json(path)
    if config.get("pipeline") != "generate" or config.get("schema_version") != 13:
        raise ExecutorError(f"invalid resolved generation config: {path}")
    fields = config.get("fields")
    if not isinstance(fields, Mapping):
        raise ExecutorError(f"resolved generation config has no fields: {path}")
    expected = {
        "base_seed": int(lane["base_seed"]),
        "games": int(lane["games"]),
        "workers": int(contract["execution"]["workers_per_gpu"]),
        "producer_checkpoint_sha256": contract["producer_checkpoint"]["sha256"],
        "public_observation": True,
        "coherent_public_belief_search": True,
        "information_set_search": False,
        "determinization_particles": 1,
        "n_full": 128,
        "n_fast": 16,
        "p_full": 0.25,
        "n_full_wide": None,
        "n_full_wide_threshold": None,
        "wide_roots_always_full": False,
        "opponent_mix_manifest": None,
        "opponent_pool_manifest": None,
        "record_automatic_transitions": False,
        "meaningful_public_history": True,
    }
    drift = {
        key: {"expected": value, "actual": fields.get(key)}
        for key, value in expected.items()
        if fields.get(key) != value
    }
    if drift:
        raise ExecutorError(f"resolved generation config drift at {path}: {drift}")
    full_config_hash = _digest(config)
    config_hash = "sha256:" + full_config_hash.removeprefix("sha256:")[:16]
    if (
        manifest.get("config_hash") != config_hash
        or manifest.get("full_config_hash") != full_config_hash
    ):
        raise ExecutorError(f"manifest/config hash mismatch for {lane['lane_id']}")
    return {
        "path": str(path),
        "sha256": file_sha256,
        "config_hash": config_hash,
        "full_config_hash": full_config_hash,
    }


def _normalized_sha256(value: object) -> str:
    text = str(value)
    return text if text.startswith("sha256:") else "sha256:" + text


def _verify_shard_arrays(
    path: Path,
    *,
    contract: Mapping[str, Any],
    trace: dict[str, Any],
) -> dict[str, int]:
    try:
        import numpy as np

        with np.load(path, allow_pickle=False) as shard:
            required = {
                "game_seed",
                "decision_index",
                "seat",
                "terminated",
                "truncated",
                "policy_weight_multiplier",
                "target_information_regime",
                "legal_action_mask",
                "simulations_used",
                "search_evidence_version",
                "search_evidence_offsets",
                "search_visit_counts_flat",
                "search_completed_q_flat",
            }
            missing = required - set(shard.files)
            if missing:
                raise ExecutorError(
                    f"shard lacks required closure columns {sorted(missing)}: {path}"
                )
            seeds = np.asarray(shard["game_seed"]).reshape(-1)
            decisions = np.asarray(shard["decision_index"]).reshape(-1)
            seats = np.asarray(shard["seat"]).reshape(-1)
            terminated = np.asarray(shard["terminated"], dtype=np.bool_).reshape(-1)
            truncated = np.asarray(shard["truncated"], dtype=np.bool_).reshape(-1)
            weights = np.asarray(
                shard["policy_weight_multiplier"], dtype=np.float32
            ).reshape(-1)
            regimes = np.asarray(shard["target_information_regime"]).reshape(-1)
            rows = int(seeds.size)
            if not all(
                array.size == rows
                for array in (decisions, seats, terminated, truncated, weights, regimes)
            ):
                raise ExecutorError(f"shard scalar column length mismatch: {path}")
            if bool(np.any((seats != 0) & (seats != 1))):
                raise ExecutorError(f"invalid seat identity in {path}")
            if bool(np.any(regimes != contract["target_information_regime"])):
                raise ExecutorError(f"mixed/non-coherent target regimes in {path}")
            if bool(np.any(~np.isfinite(weights))) or bool(np.any(weights < 0.0)):
                raise ExecutorError(f"invalid policy weights in {path}")
            active = weights > 0.0
            offsets = np.asarray(shard["search_evidence_offsets"], dtype=np.uint32)
            visits = np.asarray(shard["search_visit_counts_flat"], dtype=np.uint16)
            completed_q = np.asarray(shard["search_completed_q_flat"], dtype=np.float32)
            version = int(np.asarray(shard["search_evidence_version"]).item())
            if version != 1 or offsets.shape != (int(active.sum()) + 1,):
                raise ExecutorError(
                    f"malformed search evidence offsets/version in {path}"
                )
            if int(offsets[0]) != 0 or bool(np.any(offsets[1:] < offsets[:-1])):
                raise ExecutorError(f"non-monotone search evidence offsets in {path}")
            if visits.shape != completed_q.shape or int(offsets[-1]) != visits.size:
                raise ExecutorError(f"search evidence flat payload mismatch in {path}")
            if not bool(np.all(np.isfinite(completed_q))):
                raise ExecutorError(f"non-finite completed-Q evidence in {path}")
            widths = np.asarray(shard["legal_action_mask"], dtype=np.bool_).sum(axis=1)[
                active
            ]
            if not bool(np.array_equal(widths.astype(np.uint32), np.diff(offsets))):
                raise ExecutorError(f"search evidence/legal width mismatch in {path}")
            simulations = np.asarray(shard["simulations_used"]).reshape(-1)[active]
            cumulative = np.concatenate(
                (np.asarray([0], dtype=np.uint64), np.cumsum(visits, dtype=np.uint64))
            )
            evidence_simulations = cumulative[offsets[1:]] - cumulative[offsets[:-1]]
            if not bool(
                np.array_equal(evidence_simulations, simulations.astype(np.uint64))
            ):
                raise ExecutorError(f"search evidence visit sum mismatch in {path}")

            for index in range(rows):
                seed = int(seeds[index])
                decision = int(decisions[index])
                terminal = bool(terminated[index] or truncated[index])
                if bool(truncated[index]):
                    raise ExecutorError(
                        f"truncated trajectory in accepted shard {path}"
                    )
                current_seed = trace.get("current_seed")
                if current_seed is None or seed != current_seed:
                    if current_seed is not None and not bool(trace["current_complete"]):
                        raise ExecutorError(
                            f"trajectory {current_seed} lacks terminal row before {seed}"
                        )
                    if current_seed is not None and trace["current_seats"] != {0, 1}:
                        raise ExecutorError(
                            f"trajectory {current_seed} lacks a complete two-seat trace"
                        )
                    if seed in trace["seen"]:
                        raise ExecutorError(
                            f"duplicate/non-contiguous game seed {seed}"
                        )
                    if decision != 0:
                        raise ExecutorError(
                            f"trajectory {seed} starts at decision {decision}"
                        )
                    trace["seen"].add(seed)
                    trace["current_seed"] = seed
                    trace["last_decision"] = decision
                    trace["current_seats"] = {int(seats[index])}
                else:
                    if decision != int(trace["last_decision"]) + 1:
                        raise ExecutorError(
                            f"trajectory {seed} jumps {trace['last_decision']}->{decision}"
                        )
                    trace["last_decision"] = decision
                    trace["current_seats"].add(int(seats[index]))
                trace["current_complete"] = terminal
                if terminal and trace["current_seats"] != {0, 1}:
                    raise ExecutorError(
                        f"trajectory {seed} terminates without actions from both seats"
                    )
            return {
                "rows": rows,
                "policy_active_rows": int(np.count_nonzero(active)),
            }
    except ExecutorError:
        raise
    except Exception as error:  # noqa: BLE001 - corrupt NPZ must fail closure.
        raise ExecutorError(
            f"cannot authenticate shard payload {path}: {error}"
        ) from error


def _verify_worker_payload(
    out_dir: Path,
    *,
    worker_index: int,
    lane: Mapping[str, Any],
    contract: Mapping[str, Any],
    trace: dict[str, Any],
) -> dict[str, Any]:
    worker_id = f"worker_{worker_index:03d}"
    worker_dir = out_dir / worker_id
    manifest_path = worker_dir / "manifest.json"
    progress_path = worker_dir / "progress.json"
    manifest, manifest_sha256, _manifest_stat = _read_stable_json(manifest_path)
    progress, progress_sha256, _progress_stat = _read_stable_json(progress_path)
    workers = int(contract["execution"]["workers_per_gpu"])
    expected_games = _expected_worker_games(int(lane["games"]), workers, worker_index)
    expected_start = sum(
        _expected_worker_games(int(lane["games"]), workers, prior)
        for prior in range(worker_index)
    )
    expected = {
        "games_requested": expected_games,
        "games_completed": expected_games,
        "games_failed": 0,
        "games_truncated": 0,
        "game_index_start": expected_start,
        "base_seed": int(lane["base_seed"]),
        "target_information_regime": contract["target_information_regime"],
        "search_evidence_schema": contract["acceptance"][
            "require_search_evidence_schema"
        ],
    }
    drift = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if drift or manifest.get("errors") not in ([], None):
        raise ExecutorError(
            f"worker manifest drift for {lane['lane_id']}/{worker_id}: {drift}"
        )
    if (
        int(progress.get("games_succeeded", -1)) != expected_games
        or int(progress.get("games_failed", -1)) != 0
        or int(progress.get("games_truncated", -1)) != 0
        or progress.get("errors") not in ([], None)
    ):
        raise ExecutorError(f"worker progress is not cleanly complete: {progress_path}")
    confirmed = progress.get("confirmed_shards")
    manifest_shards = manifest.get("shards")
    if not isinstance(confirmed, list) or not isinstance(manifest_shards, list):
        raise ExecutorError(f"worker shard inventory is malformed: {worker_id}")
    if len(confirmed) != len(manifest_shards) or not confirmed:
        raise ExecutorError(f"worker shard inventory count drift: {worker_id}")
    shard_records: list[dict[str, Any]] = []
    payload_rows = 0
    policy_active_rows = 0
    for expected_index, record in enumerate(confirmed):
        if (
            not isinstance(record, Mapping)
            or int(record.get("index", -1)) != expected_index
        ):
            raise ExecutorError(
                f"non-contiguous confirmed shard inventory: {worker_id}"
            )
        shard_path = worker_dir / str(record.get("filename", ""))
        if str(shard_path) != str(manifest_shards[expected_index]):
            raise ExecutorError(
                f"worker manifest/progress shard path drift: {shard_path}"
            )
        try:
            shard_stat = shard_path.stat()
        except OSError as error:
            raise ExecutorError(
                f"missing confirmed shard {shard_path}: {error}"
            ) from error
        if not stat.S_ISREG(shard_stat.st_mode) or shard_path.is_symlink():
            raise ExecutorError(f"confirmed shard is not a regular file: {shard_path}")
        actual_sha256 = _file_sha256(shard_path)
        if (
            actual_sha256 != _normalized_sha256(record.get("sha256"))
            or int(record.get("size_bytes", -1)) != shard_stat.st_size
        ):
            raise ExecutorError(f"confirmed shard bytes drift: {shard_path}")
        arrays = _verify_shard_arrays(shard_path, contract=contract, trace=trace)
        if int(record.get("rows", -1)) != arrays["rows"]:
            raise ExecutorError(f"confirmed shard row count drift: {shard_path}")
        payload_rows += arrays["rows"]
        policy_active_rows += arrays["policy_active_rows"]
        shard_records.append(
            {
                "index": expected_index,
                "path": str(shard_path),
                "sha256": actual_sha256,
                "size_bytes": shard_stat.st_size,
                "rows": arrays["rows"],
            }
        )
    if payload_rows != int(manifest.get("rows", -1)):
        raise ExecutorError(f"worker payload rows differ from manifest: {worker_id}")
    progress_rows = int(progress.get("rows_confirmed", progress.get("rows", -1)))
    if (
        progress_rows != payload_rows
        or int(progress.get("simulations_used_total", -1))
        != int(manifest.get("simulations_used_total", -2))
        or int(progress.get("shard_count_confirmed", -1)) != len(shard_records)
    ):
        raise ExecutorError(
            f"worker progress/manifest/payload totals drift: {worker_id}"
        )
    return {
        "worker_id": worker_id,
        "manifest": {"path": str(manifest_path), "sha256": manifest_sha256},
        "progress": {"path": str(progress_path), "sha256": progress_sha256},
        "games_completed": expected_games,
        "rows": payload_rows,
        "policy_active_rows": policy_active_rows,
        "simulations_used_total": int(manifest.get("simulations_used_total", 0)),
        "shard_count": len(shard_records),
        "shards_sha256": _digest(shard_records),
        "shard_bytes": sum(int(item["size_bytes"]) for item in shard_records),
        "shards": shard_records,
    }


def _verify_existing_completion(
    path: Path,
    *,
    contract: Mapping[str, Any],
    launch_file_sha256: str,
) -> dict[str, Any]:
    receipt, file_sha256 = _authenticated_receipt(
        path, expected_schema=COMPLETION_RECEIPT_SCHEMA
    )
    if (
        receipt.get("status") != "complete"
        or receipt.get("output_root") != str(contract["execution"]["output_root"])
        or receipt.get("contract", {}).get("contract_sha256")
        != contract["contract_sha256"]
        or receipt.get("launch_receipt", {}).get("file_sha256") != launch_file_sha256
        or receipt.get("producer_checkpoint") != contract["producer_checkpoint"]
        or receipt.get("target_information_regime")
        != contract["target_information_regime"]
        or receipt.get("search_evidence_schema")
        != contract["acceptance"]["require_search_evidence_schema"]
    ):
        raise ExecutorError("completion receipt binds a different transaction")
    totals = receipt.get("totals", {})
    seeds = receipt.get("seed_inventory", {})
    expected_games = int(contract["execution"]["total_games"])
    expected_ranges = [
        {
            "lane_id": str(lane["lane_id"]),
            "start": int(lane["base_seed"]),
            "end_exclusive": int(lane["base_seed"]) + int(lane["games"]),
        }
        for lane in contract["execution"]["lanes"]
    ]
    expected_seed_values = [
        seed
        for lane in contract["execution"]["lanes"]
        for seed in range(
            int(lane["base_seed"]),
            int(lane["base_seed"]) + int(lane["games"]),
        )
    ]
    if (
        int(totals.get("games_requested", -1)) != expected_games
        or int(totals.get("games_completed", -1)) != expected_games
        or int(totals.get("games_failed", -1)) != 0
        or int(totals.get("games_truncated", -1)) != 0
        or int(totals.get("complete_trace_games", -1)) != expected_games
        or int(totals.get("incomplete_trace_games", -1)) != 0
        or int(seeds.get("count", -1)) != expected_games
        or int(seeds.get("unique_count", -1)) != expected_games
        or seeds.get("lane_ranges") != expected_ranges
        or seeds.get("seeds_sha256") != _digest(expected_seed_values)
    ):
        raise ExecutorError("completion receipt violates accepted games/seeds totals")
    payload_inventory: list[dict[str, Any]] = []
    for lane in receipt.get("lanes", ()):
        for record_name in ("manifest", "config_registry"):
            record = lane.get(record_name, {})
            artifact = Path(str(record.get("path", "")))
            if not artifact.is_file() or _file_sha256(artifact) != record.get("sha256"):
                raise ExecutorError(f"completion-bound {record_name} drift: {artifact}")
        for worker in lane.get("workers", ()):
            for record_name in ("manifest", "progress"):
                record = worker.get(record_name, {})
                artifact = Path(str(record.get("path", "")))
                if not artifact.is_file() or _file_sha256(artifact) != record.get(
                    "sha256"
                ):
                    raise ExecutorError(
                        f"completion-bound worker {record_name} drift: {artifact}"
                    )
            shards = worker.get("shards")
            if not isinstance(shards, list) or _digest(shards) != worker.get(
                "shards_sha256"
            ):
                raise ExecutorError("completion-bound NPZ inventory digest drift")
            for shard in shards:
                artifact = Path(str(shard.get("path", "")))
                try:
                    metadata = artifact.lstat()
                except OSError as error:
                    raise ExecutorError(
                        f"completion-bound NPZ is missing: {artifact}: {error}"
                    ) from error
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or artifact.is_symlink()
                    or metadata.st_size != int(shard.get("size_bytes", -1))
                    or _file_sha256(artifact) != shard.get("sha256")
                ):
                    raise ExecutorError(f"completion-bound NPZ drift: {artifact}")
        payload_inventory.append(
            {
                "lane_id": lane.get("lane_id"),
                "manifest_sha256": lane.get("manifest", {}).get("sha256"),
                "config_sha256": lane.get("config_registry", {}).get("sha256"),
                "full_config_hash": lane.get("generator_semantics", {}).get(
                    "full_config_hash"
                ),
                "shards_sha256": lane.get("payload", {}).get("shards_sha256"),
            }
        )
    inventory_sha256 = _digest(payload_inventory)
    if (
        len(payload_inventory) != len(contract["execution"]["lanes"])
        or receipt.get("payload_inventory_sha256") != inventory_sha256
        or receipt.get("npz_inventory_sha256") != inventory_sha256
    ):
        raise ExecutorError("completion-bound fleet NPZ inventory digest drift")
    return {
        **receipt,
        "path": str(path),
        "file_sha256": file_sha256,
    }


def collect(
    contract_path: Path,
    *,
    host_address: str,
    stale_seconds: float = DEFAULT_STALE_SECONDS,
) -> dict[str, Any]:
    contract_path = contract_path.expanduser().resolve(strict=True)
    contract, launch, launch_file_sha256, _commands = _authenticate_launch(
        contract_path, host_address=host_address
    )
    output_root = Path(str(contract["execution"]["output_root"]))
    completion_path = output_root / "completion.receipt.json"
    if completion_path.exists():
        return _verify_existing_completion(
            completion_path,
            contract=contract,
            launch_file_sha256=launch_file_sha256,
        )
    snapshot = status(
        contract_path,
        host_address=host_address,
        stale_seconds=stale_seconds,
    )
    if snapshot["state"] != "complete_uncollected":
        raise ExecutorError(
            "cannot collect an incomplete coherent corpus: "
            f"state={snapshot['state']} complete={len(snapshot['complete_lanes'])}/"
            f"{len(snapshot['lanes'])} failed={snapshot['failed_lanes']} "
            f"stale={snapshot['stale_lanes']}"
        )

    lane_results: list[dict[str, Any]] = []
    global_seen: set[int] = set()
    totals = {
        "lane_count": 0,
        "games_requested": 0,
        "games_completed": 0,
        "games_failed": 0,
        "games_truncated": 0,
        "rows": 0,
        "policy_active_rows": 0,
        "simulations_used_total": 0,
        "shard_count": 0,
        "shard_bytes": 0,
        "complete_trace_games": 0,
        "incomplete_trace_games": 0,
    }
    for lane in contract["execution"]["lanes"]:
        lane_id = str(lane["lane_id"])
        out_dir = output_root / lane_id
        manifest_path = out_dir / "manifest.json"
        manifest, manifest_sha256, issues = _validate_lane_manifest(
            manifest_path, lane=lane, contract=contract
        )
        if issues:
            raise ExecutorError(
                f"lane manifest acceptance failed for {lane_id}: {issues}"
            )
        config_record = _resolved_config_record(
            out_dir / "config.registry.jsonl",
            lane=lane,
            contract=contract,
            manifest=manifest,
        )
        trace: dict[str, Any] = {
            "seen": global_seen,
            "current_seed": None,
            "last_decision": None,
            "current_complete": False,
            "current_seats": set(),
        }
        workers = [
            _verify_worker_payload(
                out_dir,
                worker_index=index,
                lane=lane,
                contract=contract,
                trace=trace,
            )
            for index in range(int(contract["execution"]["workers_per_gpu"]))
        ]
        if trace["current_seed"] is not None and not bool(trace["current_complete"]):
            raise ExecutorError(f"lane {lane_id} ends with an incomplete trajectory")
        if trace["current_seed"] is not None and trace["current_seats"] != {0, 1}:
            raise ExecutorError(
                f"lane {lane_id} ends without a complete two-seat trace"
            )
        lane_game_count = sum(int(item["games_completed"]) for item in workers)
        lane_rows = sum(int(item["rows"]) for item in workers)
        lane_policy_active = sum(int(item["policy_active_rows"]) for item in workers)
        lane_simulations = sum(int(item["simulations_used_total"]) for item in workers)
        if (
            lane_game_count != int(lane["games"])
            or lane_rows != int(manifest["rows"])
            or lane_simulations != int(manifest["simulations_used_total"])
            or lane_policy_active <= 0
        ):
            raise ExecutorError(f"lane transitive totals drift for {lane_id}")
        top_workers = set(map(str, manifest.get("worker_summaries", ())))
        expected_workers = {item["manifest"]["path"] for item in workers}
        if top_workers != expected_workers:
            raise ExecutorError(
                f"top-level worker manifest inventory drift for {lane_id}"
            )
        lane_record = {
            "lane_id": lane_id,
            "gpu": int(lane["gpu"]),
            "base_seed": int(lane["base_seed"]),
            "games_requested": int(lane["games"]),
            "games_completed": lane_game_count,
            "games_failed": 0,
            "games_truncated": 0,
            "rows": lane_rows,
            "policy_active_rows": lane_policy_active,
            "simulations_used_total": lane_simulations,
            "manifest": {"path": str(manifest_path), "sha256": manifest_sha256},
            "config_registry": config_record,
            "generator_semantics": {
                "config_hash": str(manifest["config_hash"]),
                "full_config_hash": str(manifest["full_config_hash"]),
                "generator_sha256": str(launch["preflight"]["generator_sha256"]),
                "source_commit": str(launch["preflight"]["git_commit"]),
            },
            "workers": workers,
            "payload": {
                "shard_count": sum(int(item["shard_count"]) for item in workers),
                "shard_bytes": sum(int(item["shard_bytes"]) for item in workers),
                "shards_sha256": _digest(
                    [
                        {
                            "worker_id": item["worker_id"],
                            "shard_count": item["shard_count"],
                            "shards_sha256": item["shards_sha256"],
                        }
                        for item in workers
                    ]
                ),
            },
            "trace": {
                "game_count": lane_game_count,
                "complete_action_trace_game_count": lane_game_count,
                "incomplete_action_trace_game_count": 0,
                "method": "game_seed_plus_contiguous_two_seat_decision_trace",
            },
        }
        lane_results.append(lane_record)
        totals["lane_count"] += 1
        totals["games_requested"] += int(lane["games"])
        totals["games_completed"] += lane_game_count
        totals["rows"] += lane_rows
        totals["policy_active_rows"] += lane_policy_active
        totals["simulations_used_total"] += lane_simulations
        totals["shard_count"] += int(lane_record["payload"]["shard_count"])
        totals["shard_bytes"] += int(lane_record["payload"]["shard_bytes"])
        totals["complete_trace_games"] += lane_game_count

    expected_games = int(contract["execution"]["total_games"])
    expected_seeds = {
        seed
        for lane in contract["execution"]["lanes"]
        for seed in range(
            int(lane["base_seed"]),
            int(lane["base_seed"]) + int(lane["games"]),
        )
    }
    if (
        totals["lane_count"] != len(contract["execution"]["lanes"])
        or totals["games_requested"] != expected_games
        or totals["games_completed"] != expected_games
        or totals["complete_trace_games"] != expected_games
        or len(global_seen) != expected_games
        or global_seen != expected_seeds
        or totals["games_failed"] != 0
        or totals["games_truncated"] != 0
        or totals["incomplete_trace_games"] != 0
        or totals["policy_active_rows"] <= 0
    ):
        raise ExecutorError(f"fleet completion totals violate the contract: {totals}")
    payload_inventory = [
        {
            "lane_id": item["lane_id"],
            "manifest_sha256": item["manifest"]["sha256"],
            "config_sha256": item["config_registry"]["sha256"],
            "full_config_hash": item["generator_semantics"]["full_config_hash"],
            "shards_sha256": item["payload"]["shards_sha256"],
        }
        for item in lane_results
    ]
    receipt = {
        "schema_version": COMPLETION_RECEIPT_SCHEMA,
        "status": "complete",
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "output_root": str(output_root),
        "contract": {
            "path": str(contract_path),
            "file_sha256": _file_sha256(contract_path),
            "contract_sha256": contract["contract_sha256"],
        },
        "launch_receipt": {
            "path": str(output_root / "launch.receipt.json"),
            "file_sha256": launch_file_sha256,
            "receipt_sha256": launch["receipt_sha256"],
        },
        "producer_checkpoint": dict(contract["producer_checkpoint"]),
        "producer_source": {
            "repo": str(launch["preflight"]["repo"]),
            "source_commit": str(launch["preflight"]["git_commit"]),
            "tracked_diff_present": bool(
                launch["preflight"].get("tracked_diff_present", False)
            ),
            "generator_path": str(launch["preflight"]["generator"]),
            "generator_sha256": str(launch["preflight"]["generator_sha256"]),
        },
        "coherent_operator": {
            "semantic_sha256": _digest(contract["operator"]),
            "source_commit": str(launch["preflight"]["git_commit"]),
            "config": dict(contract["operator"]),
        },
        "target_information_regime": contract["target_information_regime"],
        "search_evidence_schema": contract["acceptance"][
            "require_search_evidence_schema"
        ],
        "seed_inventory": {
            "count": len(expected_seeds),
            "unique_count": len(global_seen),
            "minimum": min(expected_seeds),
            "maximum_exclusive": max(expected_seeds) + 1,
            "contiguous": expected_seeds
            == set(range(min(expected_seeds), max(expected_seeds) + 1)),
            "seeds_sha256": _digest(sorted(global_seen)),
            "lane_ranges": [
                {
                    "lane_id": str(lane["lane_id"]),
                    "start": int(lane["base_seed"]),
                    "end_exclusive": int(lane["base_seed"]) + int(lane["games"]),
                }
                for lane in contract["execution"]["lanes"]
            ],
        },
        "totals": totals,
        "payload_inventory_sha256": _digest(payload_inventory),
        "npz_inventory_sha256": _digest(payload_inventory),
        "lanes": lane_results,
    }
    _write_receipt(completion_path, receipt)
    return _verify_existing_completion(
        completion_path,
        contract=contract,
        launch_file_sha256=launch_file_sha256,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=REPO_ROOT
        / "configs/operations/a1-target-identity-coherent-n128-rd-v1/contract.json",
    )
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--host-address", required=True)
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--go",
        action="store_true",
        help="claim the sealed seeds and detach all declared lanes; omitted is read-only",
    )
    action.add_argument(
        "--status",
        action="store_true",
        help="authenticate the existing launch and report lane/worker progress",
    )
    action.add_argument(
        "--wait",
        action="store_true",
        help="wait without relaunching and seal the corpus when every lane completes",
    )
    action.add_argument(
        "--collect",
        action="store_true",
        help="verify a completed corpus and write its immutable completion receipt",
    )
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--stale-seconds", type=float, default=DEFAULT_STALE_SECONDS)
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=0.0,
        help="wait timeout; zero means no timeout",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.status:
            result = status(
                args.contract,
                host_address=args.host_address,
                stale_seconds=args.stale_seconds,
            )
        elif args.wait:
            result = wait_for_completion(
                args.contract,
                host_address=args.host_address,
                poll_seconds=args.poll_seconds,
                stale_seconds=args.stale_seconds,
                timeout_seconds=args.timeout_seconds,
            )
        elif args.collect:
            result = collect(
                args.contract,
                host_address=args.host_address,
                stale_seconds=args.stale_seconds,
            )
        else:
            if args.python is None:
                raise ExecutorError("--python is required for launch/dry-run")
            result = execute(
                args.contract,
                repo=args.repo,
                python=args.python,
                host_address=args.host_address,
                go=bool(args.go),
            )
    except (ExecutorError, identity.InventoryError, OSError, ValueError) as error:
        print(f"a1_coherent_target_rd_executor: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
