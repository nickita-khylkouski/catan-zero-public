#!/usr/bin/env python3
"""Manual, fail-closed executor for a sealed 40-lane/120-job A1 render."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shlex
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools import a1_pre_wave_contract as contract  # noqa: E402

HOST_SCHEMA = "a1-production-hosts-v1"
RECEIPT_SCHEMA = "a1-production-executor-receipt-v1"
BRIDGE_SCHEMA = "a1-frozen-plan-hardened-executor-bridge-v1"
LANE_SCHEMA = "a1-production-lane-v1"
SAFE_ALIAS = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
CATEGORY_ORDER = ("current_producer", "recent_history", "hard_negative")
CLIENT_ENVIRONMENT = {
    "CUDA_MPS_PIPE_DIRECTORY": "/tmp/mps_pipe_host",
    "CUDA_MPS_LOG_DIRECTORY": "/tmp/mps_log_host",
}
SUPERVISOR_ENVIRONMENT = {"PYTHONDONTWRITEBYTECODE": "1"}
REQUIRED_NOFILE_SOFT = 65_536
STOP_SSH_TIMEOUT_SECONDS = 45.0
MPS_UNIT_PATH = _REPO_ROOT / "tools/fleet/systemd/nvidia-mps.service"
FORBIDDEN_ADAPTIVE_ARGV = (
    "--n-full-wide",
    "--n-full-wide-threshold",
    "--raw-policy-above-width",
)


class ExecutorError(RuntimeError):
    pass


def _materialize_job_environment(
    command: Mapping[str, Any], *, repo_dir: str
) -> dict[str, Any]:
    """Resolve the one sealed repo token without inheriting host environment."""

    environment = command.get("environment")
    if not isinstance(environment, dict):
        raise ExecutorError("job environment is not a mapping")
    expected_pythonpath = (
        f"{contract.RUNTIME_REPO_TOKEN}/src:{contract.RUNTIME_REPO_TOKEN}"
    )
    if environment.get("PYTHONPATH") != expected_pythonpath:
        raise ExecutorError("rendered PYTHONPATH token drift")
    if command.get("environment_sha256") != _digest(environment):
        raise ExecutorError("rendered environment digest drift before materialization")
    runtime = {str(key): str(value) for key, value in environment.items()}
    runtime["PYTHONPATH"] = f"{repo_dir}/src:{repo_dir}"
    materialized = dict(command)
    materialized["render_environment_sha256"] = command["environment_sha256"]
    materialized["environment"] = runtime
    materialized["environment_sha256"] = _digest(runtime)
    return materialized


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
        raise ExecutorError(f"cannot load {path}: {error}") from error
    if not isinstance(value, dict):
        raise ExecutorError(f"{path} must contain an object")
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


def _create_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ExecutorError(f"O_EXCL receipt already exists: {path}") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_hosts(path: Path, aliases: set[str]) -> dict[str, Any]:
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise ExecutorError(f"private host config must be mode 0600: {path} is {mode:o}")
    value = _load(path)
    expected = {"schema_version", "ssh_user", "ssh_key", "remote_root", "python", "hosts"}
    if set(value) != expected or value["schema_version"] != HOST_SCHEMA:
        raise ExecutorError(f"host config must use exact {HOST_SCHEMA} schema")
    if not isinstance(value["hosts"], dict) or set(value["hosts"]) != aliases:
        raise ExecutorError("private host aliases must exactly match the sealed render")
    for name, host in value["hosts"].items():
        if not SAFE_ALIAS.fullmatch(name) or not isinstance(host, str) or not SAFE_ALIAS.fullmatch(host):
            raise ExecutorError(f"unsafe alias/host in private config: {name!r}")
    for key in ("ssh_user", "remote_root", "python"):
        if not isinstance(value[key], str) or not value[key].strip():
            raise ExecutorError(f"host config {key} must be non-empty")
    if not PurePosixPath(value["remote_root"]).is_absolute():
        raise ExecutorError("host config remote_root must be an absolute remote path")
    if not PurePosixPath(value["python"]).is_absolute():
        raise ExecutorError("host config python must be an absolute external venv path")
    ssh_key = Path(value["ssh_key"]).expanduser().resolve()
    if not ssh_key.is_file():
        raise ExecutorError(f"SSH key is missing: {ssh_key}")
    value["ssh_key"] = str(ssh_key)
    return value


def verify_render(
    lock_path: Path,
    render_path: Path,
    *,
    verify_lock_fn: Callable[..., dict[str, Any]] = contract.verify_lock,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    try:
        lock = verify_lock_fn(lock_path, require_all_job_claims=True)
    except Exception as error:
        raise ExecutorError(f"sealed lock/all-claim verification failed: {error}") from error
    rendered = _load(render_path)
    if rendered.get("schema_version") != contract.RENDER_SCHEMA:
        raise ExecutorError(f"render schema must be {contract.RENDER_SCHEMA}")
    unhashed = dict(rendered)
    declared_render_sha = unhashed.pop("render_sha256", None)
    if declared_render_sha != contract._digest_value(unhashed):
        raise ExecutorError("render semantic digest mismatch")
    if rendered.get("contract_sha256") != lock["contract_sha256"]:
        raise ExecutorError("render binds a different contract")
    commands = rendered.get("commands")
    if not isinstance(commands, list) or len(commands) != 120:
        raise ExecutorError("production render must contain exactly 120 commands")
    jobs = {job["job_id"]: job for job in lock["fleet"]["jobs"]}
    if len(jobs) != 120:
        raise ExecutorError("sealed production lock must contain exactly 120 jobs")
    search = lock.get("science", {}).get("search_operator", {})
    if int(search.get("n_full", -1)) != 128:
        raise ExecutorError("A1 production science is locked to n_full=128")
    if (
        search.get("n_full_wide") is not None
        or search.get("n_full_wide_threshold") is not None
        or bool(search.get("wide_roots_always_full"))
        or search.get("raw_policy_above_width") is not None
    ):
        raise ExecutorError("A1 production forbids adaptive/wide search overrides")
    mix_paths = {
        Path(record["path"]).stem: Path(record["path"])
        for record in rendered["required_artifacts"]["rendered_opponent_mix"]
    }
    by_lane: dict[str, list[dict[str, Any]]] = {}
    seen: set[str] = set()
    for command_row in commands:
        command = dict(command_row)
        job_id = command.get("job_id")
        if job_id not in jobs or job_id in seen:
            raise ExecutorError(f"unknown/duplicate rendered job {job_id!r}")
        seen.add(job_id)
        job = jobs[job_id]
        if command.get("argv_sha256") != contract._digest_value(command.get("argv")):
            raise ExecutorError(f"argv digest mismatch for {job_id}")
        expected_argv = contract._generator_argv(lock, job, mix_paths=mix_paths)
        if command.get("argv") != expected_argv:
            raise ExecutorError(f"rendered argv differs from sealed command for {job_id}")
        if "--skip-guards" in expected_argv or "--no-seed-claim" in expected_argv:
            raise ExecutorError(f"guard bypass in {job_id}")
        if "--resume" not in expected_argv:
            raise ExecutorError(f"{job_id} lacks explicit exact-run resume semantics")
        try:
            rendered_n_full = int(expected_argv[expected_argv.index("--n-full") + 1])
        except (ValueError, IndexError) as error:
            raise ExecutorError(f"{job_id} lacks an exact --n-full value") from error
        if rendered_n_full != 128 or any(flag in expected_argv for flag in FORBIDDEN_ADAPTIVE_ARGV):
            raise ExecutorError(f"{job_id} is not the sealed n128/no-adaptive recipe")
        expected_environment = contract._job_environment(lock, job)
        if command.get("environment") != expected_environment:
            raise ExecutorError(f"exact client environment drift for {job_id}")
        if command.get("environment_sha256") != contract._digest_value(
            expected_environment
        ):
            raise ExecutorError(f"client environment digest mismatch for {job_id}")
        expected_config_provenance = contract._expected_generate_config_provenance(
            lock,
            job,
            opponent_mix_manifest=(
                None
                if job["category"] == "current_producer"
                else str(mix_paths[job["category"]])
            ),
        )
        if command.get("config_provenance") != expected_config_provenance:
            raise ExecutorError(f"typed config provenance drift for {job_id}")
        claim = command.get("ledger_claim", {})
        expected_row = contract._ledger_claim_row(lock, job)
        if claim.get("row") != expected_row or claim.get("row_sha256") != contract._digest_value(expected_row):
            raise ExecutorError(f"claim row drift for {job_id}")
        source = Path(command["output_attestation"]["source"])
        if not source.is_file() or _sha256(source) != command["output_attestation"]["source_file_sha256"]:
            raise ExecutorError(f"job attestation drift for {job_id}")
        by_lane.setdefault(command["worker_id"], []).append(command)
    if seen != set(jobs) or len(by_lane) != 40:
        raise ExecutorError("render must cover exactly 40 physical lanes and 120 jobs")
    for worker_id, lane in by_lane.items():
        lane.sort(key=lambda item: CATEGORY_ORDER.index(item["category"]))
        if tuple(item["category"] for item in lane) != CATEGORY_ORDER:
            raise ExecutorError(f"lane {worker_id} category/dependency order drift")
        if len({(item["host_alias"], item["gpu"]) for item in lane}) != 1:
            raise ExecutorError(f"lane {worker_id} mixes host/GPU identities")
        for index, item in enumerate(lane):
            expected_dependency = [] if index == 0 else [lane[index - 1]["job_id"]]
            if item.get("must_run_after") != expected_dependency:
                raise ExecutorError(f"lane {worker_id} dependency drift")
    return lock, rendered, by_lane


def _repo_artifacts(
    rendered: dict[str, Any], *, repo_root: Path = _REPO_ROOT
) -> list[dict[str, Any]]:
    root = repo_root.resolve(strict=True)
    required = rendered["required_artifacts"]
    records = [
        required["guard_config"],
        *required["generator_code"],
        *required["runtime_code_tree"],
    ]
    files: dict[str, Path] = {}
    for record in records:
        path = Path(record["path"]).resolve()
        if _sha256(path) != record["sha256"]:
            raise ExecutorError(f"required repo artifact drift: {path}")
        try:
            relative = path.relative_to(root)
        except ValueError as error:
            raise ExecutorError(f"runtime artifact is outside canonical repo: {path}") from error
        files[str(relative)] = path
    supervisor = (root / "tools/fleet/a1_lane_supervisor.py").resolve()
    files[str(supervisor.relative_to(root))] = supervisor
    stop_helper = (root / "tools/fleet/a1_stop_helper.py").resolve()
    files[str(stop_helper.relative_to(root))] = stop_helper
    return [
        {
            "path": key,
            "sha256": _sha256(files[key]),
            "mode": 0o555 if os.access(files[key], os.X_OK) else 0o444,
        }
        for key in sorted(files)
    ]


def _repo_files(
    artifacts: Sequence[Mapping[str, Any]], *, repo_root: Path = _REPO_ROOT
) -> list[Path]:
    root = repo_root.resolve(strict=True)
    files: list[Path] = []
    for record in artifacts:
        relative = PurePosixPath(str(record["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ExecutorError(f"unsafe repo artifact path: {relative}")
        source = (root / Path(*relative.parts)).resolve(strict=True)
        try:
            source.relative_to(root)
        except ValueError as error:
            raise ExecutorError(f"repo artifact escapes source root: {relative}") from error
        if not source.is_file() or _sha256(source) != record["sha256"]:
            raise ExecutorError(f"repo artifact source drift: {source}")
        files.append(source)
    return files


def _execution_repo_root(plan: Mapping[str, Any]) -> Path:
    """Validate an optional frozen-plan/hardened-executor bridge."""
    private = plan["_private"]
    bridge = private.get("executor_bridge")
    if bridge is None:
        return _REPO_ROOT
    expected_keys = {
        "schema_version",
        "frozen_repo_root",
        "frozen_executor",
        "hardened_executor",
        "bridge_tool",
        "plan_sha256",
        "repo_artifacts_sha256",
        "bridge_sha256",
    }
    if not isinstance(bridge, dict) or set(bridge) != expected_keys:
        raise ExecutorError("invalid frozen-plan executor bridge schema")
    unhashed = dict(bridge)
    declared = unhashed.pop("bridge_sha256")
    if bridge["schema_version"] != BRIDGE_SCHEMA or declared != _digest(unhashed):
        raise ExecutorError("frozen-plan executor bridge digest mismatch")
    if (
        bridge["plan_sha256"] != plan.get("plan_sha256")
        or bridge["repo_artifacts_sha256"] != plan.get("repo_artifacts_sha256")
    ):
        raise ExecutorError("frozen-plan executor bridge plan binding drift")
    root = Path(str(bridge["frozen_repo_root"])).resolve(strict=True)
    frozen_path = (root / "tools/fleet/a1_production_executor.py").resolve(strict=True)
    hardened_path = Path(__file__).resolve(strict=True)
    bridge_path = (_REPO_ROOT / "tools/fleet/a1_executor_bridge.py").resolve(strict=True)
    expected_references = (
        (bridge["frozen_executor"], frozen_path, "frozen"),
        (bridge["hardened_executor"], hardened_path, "hardened"),
        (bridge["bridge_tool"], bridge_path, "bridge tool"),
    )
    for reference, path, label in expected_references:
        if (
            not isinstance(reference, dict)
            or set(reference) != {"path", "sha256"}
            or Path(str(reference["path"])).resolve(strict=True) != path
            or reference["sha256"] != _sha256(path)
        ):
            raise ExecutorError(f"{label} executor bridge code binding drift")
    return root


def _build_repo_tar(files: Sequence[Path], destination: Path) -> str:
    with tarfile.open(destination, "w") as archive:
        for source in files:
            relative = source.relative_to(_REPO_ROOT)
            info = tarfile.TarInfo(str(relative))
            data = source.read_bytes()
            info.size = len(data)
            info.mode = 0o755 if os.access(source, os.X_OK) else 0o444
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            archive.addfile(info, io.BytesIO(data))
    return _sha256(destination)


def build_plan(
    *,
    lock_path: Path,
    render_path: Path,
    hosts_path: Path,
    receipt_path: Path,
    verify_lock_fn: Callable[..., dict[str, Any]] = contract.verify_lock,
    repo_root: Path = _REPO_ROOT,
) -> dict[str, Any]:
    lock, rendered, lanes = verify_render(
        lock_path, render_path, verify_lock_fn=verify_lock_fn
    )
    repo_artifacts = _repo_artifacts(rendered, repo_root=repo_root)
    live_ledger_path = Path(rendered["required_artifacts"]["seed_ledger"]["path"])
    live_seed_ledger_sha256 = _sha256(live_ledger_path)
    aliases = {lane[0]["host_alias"] for lane in lanes.values()}
    hosts = load_hosts(hosts_path, aliases)
    plan = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "dry_run",
        "contract_sha256": lock["contract_sha256"],
        "render_sha256": rendered["render_sha256"],
        "lock": str(lock_path.resolve()),
        "render": str(render_path.resolve()),
        "operator_manifests": {
            "lock": {"sha256": _sha256(lock_path), "remote_name": "contract.lock.json"},
            "render": {"sha256": _sha256(render_path), "remote_name": "commands.json"},
        },
        "hosts_config_sha256": _sha256(hosts_path),
        "remote_root": hosts["remote_root"],
        "lane_count": len(lanes),
        "job_count": sum(len(lane) for lane in lanes.values()),
        "claim_count": len(lock["fleet"]["jobs"]),
        "category_order": list(CATEGORY_ORDER),
        "client_environment": dict(CLIENT_ENVIRONMENT),
        "repo_artifacts_sha256": _digest(repo_artifacts),
        "live_seed_ledger_sha256": live_seed_ledger_sha256,
        "receipt": str(receipt_path.resolve()),
        "lanes": [
            {
                "worker_id": worker_id,
                "host_alias": lane[0]["host_alias"],
                "gpu": lane[0]["gpu"],
                "jobs": [item["job_id"] for item in lane],
            }
            for worker_id, lane in sorted(lanes.items())
        ],
    }
    plan["plan_sha256"] = _digest(plan)
    plan["_private"] = {
        "hosts": hosts,
        "lanes": lanes,
        "rendered": rendered,
        "repo_artifacts": repo_artifacts,
    }
    return plan


def _public(plan: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in plan.items() if key != "_private"}


def _verify_plan_digest(plan: Mapping[str, Any]) -> None:
    public = {key: value for key, value in plan.items() if key != "_private"}
    declared = public.pop("plan_sha256", None)
    if not isinstance(declared, str) or declared != _digest(public):
        raise ExecutorError("execution plan semantic digest mismatch")


def _ssh(
    hosts: dict[str, Any],
    alias: str,
    remote_command: str,
    *,
    timeout_seconds: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=10", "-i", hosts["ssh_key"],
            f"{hosts['ssh_user']}@{hosts['hosts'][alias]}", remote_command,
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout_seconds,
    )


def _scp(hosts: dict[str, Any], alias: str, source: Path, destination: str) -> None:
    result = subprocess.run(
        [
            "scp", "-q", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=10", "-i", hosts["ssh_key"], str(source),
            f"{hosts['ssh_user']}@{hosts['hosts'][alias]}:{destination}",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise ExecutorError(f"scp to {alias} failed: {result.stderr.strip()}")


def _remote_install(
    hosts: dict[str, Any], alias: str, source: Path, destination: str, expected: str
) -> None:
    incoming = f"{hosts['remote_root']}/incoming/{uuid.uuid4().hex}"
    mkdir = _ssh(hosts, alias, f"mkdir -p {shlex.quote(str(Path(incoming).parent))}")
    if mkdir.returncode != 0:
        raise ExecutorError(f"remote mkdir failed on {alias}: {mkdir.stderr.strip()}")
    _scp(hosts, alias, source, incoming)
    script = (
        "import hashlib,os,pathlib,shutil,sys;"
        "src=pathlib.Path(sys.argv[1]);dst=pathlib.Path(sys.argv[2]);exp=sys.argv[3];"
        "h=lambda p:'sha256:'+hashlib.sha256(p.read_bytes()).hexdigest();"
        "dst.parent.mkdir(parents=True,exist_ok=True);"
        "assert h(src)==exp;"
        "(None if not dst.exists() else (_ for _ in ()).throw(SystemExit(0 if h(dst)==exp else 9)));"
        "fd=os.open(dst,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444);"
        "f=os.fdopen(fd,'wb');f.write(src.read_bytes());f.flush();os.fsync(f.fileno());f.close();"
        "assert h(dst)==exp"
    )
    command = " ".join(
        shlex.quote(value)
        for value in (hosts["python"], "-c", script, incoming, destination, expected)
    )
    result = _ssh(hosts, alias, command)
    _ssh(hosts, alias, f"rm -f {shlex.quote(incoming)}")
    if result.returncode != 0:
        raise ExecutorError(f"immutable install failed on {alias}: {result.stderr.strip()}")


def _append_only_bytes(existing: bytes, desired: bytes) -> bytes:
    if desired == existing or desired.startswith(existing):
        return desired
    raise ExecutorError("remote seed ledger is not an exact prefix of the bound live ledger")


def _remote_sync_append_only_ledger(
    hosts: dict[str, Any], alias: str, source: Path, destination: str, expected: str
) -> None:
    incoming = f"{hosts['remote_root']}/incoming/{uuid.uuid4().hex}"
    mkdir = _ssh(hosts, alias, f"mkdir -p {shlex.quote(str(Path(incoming).parent))}")
    if mkdir.returncode != 0:
        raise ExecutorError(f"remote ledger mkdir failed on {alias}: {mkdir.stderr.strip()}")
    _scp(hosts, alias, source, incoming)
    script = r'''import hashlib,os,pathlib,sys,uuid
src=pathlib.Path(sys.argv[1]);dst=pathlib.Path(sys.argv[2]);expected=sys.argv[3]
data=src.read_bytes();sha=lambda value:'sha256:'+hashlib.sha256(value).hexdigest()
if sha(data)!=expected: raise SystemExit('incoming live ledger digest drift')
dst.parent.mkdir(parents=True,exist_ok=True)
if dst.exists():
    old=dst.read_bytes()
    if old==data: raise SystemExit(0)
    if not data.startswith(old): raise SystemExit('remote ledger is not an exact append-only prefix')
tmp=dst.parent/('.'+dst.name+'.'+uuid.uuid4().hex+'.tmp')
fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
with os.fdopen(fd,'wb') as handle: handle.write(data);handle.flush();os.fsync(handle.fileno())
os.replace(tmp,dst)
if sha(dst.read_bytes())!=expected: raise SystemExit('installed live ledger digest drift')'''
    command = " ".join(
        shlex.quote(value)
        for value in (hosts["python"], "-c", script, incoming, destination, expected)
    )
    result = _ssh(hosts, alias, command)
    _ssh(hosts, alias, f"rm -f {shlex.quote(incoming)}")
    if result.returncode != 0:
        raise ExecutorError(f"append-only ledger sync failed on {alias}: {result.stderr.strip()}")


def _preflight_host(
    hosts: dict[str, Any], alias: str, expected_gpus: Sequence[int]
) -> dict[str, Any]:
    """Read-only launch preflight: topology, resources, idle compute plane, MPS."""
    expected_mps_unit_sha256 = _sha256(MPS_UNIT_PATH)
    script = r'''import hashlib,importlib.metadata,json,os,pathlib,resource,subprocess,sys
expected=json.loads(sys.argv[1])
required_nofile=int(sys.argv[2])
expected_mps_unit_sha256=sys.argv[3]
nofile_soft_before,nofile_hard=resource.getrlimit(resource.RLIMIT_NOFILE)
unlimited=resource.RLIM_INFINITY
if nofile_hard!=unlimited and nofile_hard<required_nofile: raise SystemExit(f'hard RLIMIT_NOFILE {nofile_hard} is below required {required_nofile}')
if nofile_soft_before!=unlimited and nofile_soft_before<required_nofile:
    try: resource.setrlimit(resource.RLIMIT_NOFILE,(required_nofile,nofile_hard))
    except (OSError,ValueError) as error: raise SystemExit(f'cannot raise soft RLIMIT_NOFILE {nofile_soft_before} to {required_nofile}: {error!r}')
nofile_soft,nofile_hard_after=resource.getrlimit(resource.RLIMIT_NOFILE)
if nofile_hard_after!=nofile_hard: raise SystemExit(f'hard RLIMIT_NOFILE changed during preflight: {nofile_hard}->{nofile_hard_after}')
if nofile_soft!=unlimited and nofile_soft<required_nofile: raise SystemExit(f'soft RLIMIT_NOFILE {nofile_soft} is below required {required_nofile} after raise')
try:
    import torch,catanatron_rs
except Exception as error: raise SystemExit('configured interpreter dependency failure: '+repr(error))
if not torch.cuda.is_available() or torch.cuda.device_count()!=len(expected): raise SystemExit(f'torch CUDA topology drift: available={torch.cuda.is_available()} count={torch.cuda.device_count()} expected={len(expected)}')
run=lambda *args:subprocess.run(args,text=True,capture_output=True,check=False)
gpu=run('nvidia-smi','--query-gpu=index','--format=csv,noheader,nounits')
if gpu.returncode: raise SystemExit('nvidia-smi gpu query failed: '+gpu.stderr)
indices=sorted(int(line.strip()) for line in gpu.stdout.splitlines() if line.strip())
if indices!=expected: raise SystemExit(f'GPU topology drift: expected {expected}, got {indices}')
apps=run('nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader,nounits')
if apps.returncode not in (0,): raise SystemExit('nvidia-smi compute query failed: '+apps.stderr)
foreign=[]
for line in apps.stdout.splitlines():
    if not line.strip() or 'No running processes found' in line: continue
    fields=[part.strip() for part in line.split(',',1)]
    if len(fields)!=2 or 'nvidia-cuda-mps-server' not in fields[1]: foreign.append(line.strip())
if foreign: raise SystemExit('non-MPS compute applications present: '+repr(foreign))
show=run('systemctl','show','nvidia-mps.service','--property=ActiveState,UnitFileState,MainPID,Environment,FragmentPath,LimitNOFILESoft')
if show.returncode: raise SystemExit('cannot inspect nvidia-mps.service: '+show.stderr)
properties={}
for line in show.stdout.splitlines():
    if '=' in line:
        key,value=line.split('=',1);properties[key]=value
required_properties={'ActiveState','UnitFileState','MainPID','Environment','FragmentPath','LimitNOFILESoft'}
if not required_properties.issubset(properties): raise SystemExit('incomplete nvidia-mps.service properties: '+repr(properties))
active=properties['ActiveState'];enabled=properties['UnitFileState'];main_pid_raw=properties['MainPID'];environment=properties['Environment']
if active!='active' or enabled!='enabled': raise SystemExit(f'MPS service not active+enabled: {active}/{enabled}')
try: mps_limit_nofile_soft=int(properties['LimitNOFILESoft'])
except ValueError: raise SystemExit('invalid MPS LimitNOFILESoft: '+properties['LimitNOFILESoft'])
if mps_limit_nofile_soft<required_nofile: raise SystemExit(f'MPS LimitNOFILESoft {mps_limit_nofile_soft} is below required {required_nofile}')
try: main_pid=int(main_pid_raw)
except ValueError: raise SystemExit('invalid MPS MainPID: '+main_pid_raw)
if main_pid<=0 or not pathlib.Path(f'/proc/{main_pid}').exists(): raise SystemExit('MPS MainPID is not live')
required={'CUDA_MPS_PIPE_DIRECTORY':'/tmp/mps_pipe_host','CUDA_MPS_LOG_DIRECTORY':'/tmp/mps_log_host'}
for key,value in required.items():
    if f'{key}={value}' not in environment: raise SystemExit(f'MPS service {key} drift')
    path=pathlib.Path(value)
    if not path.is_dir() or not os.access(path,os.R_OK|os.W_OK|os.X_OK): raise SystemExit(f'MPS directory inaccessible: {path}')
fragment=pathlib.Path(properties['FragmentPath'])
if not fragment.is_file(): raise SystemExit('MPS service FragmentPath is not a file: '+str(fragment))
mps_unit_sha256='sha256:'+hashlib.sha256(fragment.read_bytes()).hexdigest()
if mps_unit_sha256!=expected_mps_unit_sha256: raise SystemExit(f'MPS service unit digest drift: expected {expected_mps_unit_sha256}, got {mps_unit_sha256}')
try: rust_version=importlib.metadata.version('catanatron-rs')
except importlib.metadata.PackageNotFoundError: rust_version='unknown'
print(json.dumps({'gpu_indices':indices,'compute_apps':'mps_only_or_empty','mps_active':active,'mps_enabled':enabled,'mps_main_pid':main_pid,'mps_unit_sha256':mps_unit_sha256,'mps_limit_nofile_soft':mps_limit_nofile_soft,'client_environment':required,'python':sys.executable,'torch_version':str(torch.__version__),'torch_cuda_version':str(torch.version.cuda),'catanatron_rs_version':rust_version,'required_nofile_soft':required_nofile,'nofile_soft_before':nofile_soft_before,'nofile_soft':nofile_soft,'nofile_hard':nofile_hard},sort_keys=True))'''
    command = " ".join(
        shlex.quote(value)
        for value in (
            hosts["python"], "-c", script, json.dumps(sorted(expected_gpus)),
            str(REQUIRED_NOFILE_SOFT), expected_mps_unit_sha256,
        )
    )
    result = _ssh(hosts, alias, command)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise ExecutorError(f"host preflight failed on {alias}: {detail}")
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ExecutorError(f"host preflight returned invalid JSON on {alias}") from error
    if report.get("client_environment") != CLIENT_ENVIRONMENT:
        raise ExecutorError(f"host MPS client environment drift on {alias}")
    if report.get("mps_unit_sha256") != expected_mps_unit_sha256:
        raise ExecutorError(f"host MPS service unit digest drift on {alias}")
    mps_limit_nofile_soft = report.get("mps_limit_nofile_soft")
    if type(mps_limit_nofile_soft) is not int:
        raise ExecutorError(f"invalid MPS LimitNOFILESoft report on {alias}")
    if mps_limit_nofile_soft < REQUIRED_NOFILE_SOFT:
        raise ExecutorError(
            f"host MPS LimitNOFILESoft {mps_limit_nofile_soft} is below required "
            f"{REQUIRED_NOFILE_SOFT} on {alias}"
        )
    limit_fields = (
        "required_nofile_soft", "nofile_soft_before", "nofile_soft", "nofile_hard",
    )
    if any(type(report.get(field)) is not int for field in limit_fields):
        raise ExecutorError(f"invalid RLIMIT_NOFILE report on {alias}")
    if report["required_nofile_soft"] != REQUIRED_NOFILE_SOFT:
        raise ExecutorError(f"required soft RLIMIT_NOFILE drift on {alias}")
    soft_before = report["nofile_soft_before"]
    soft = report["nofile_soft"]
    hard = report["nofile_hard"]
    if soft_before < -1 or soft < -1 or hard < -1:
        raise ExecutorError(f"invalid RLIMIT_NOFILE report on {alias}")
    if hard != -1 and hard < REQUIRED_NOFILE_SOFT:
        raise ExecutorError(
            f"host hard RLIMIT_NOFILE {hard} is below required "
            f"{REQUIRED_NOFILE_SOFT} on {alias}"
        )
    if soft != -1 and soft < REQUIRED_NOFILE_SOFT:
        raise ExecutorError(
            f"host soft RLIMIT_NOFILE {soft} is below required "
            f"{REQUIRED_NOFILE_SOFT} after preflight raise on {alias}"
        )
    if hard != -1 and (
        soft == -1 or soft > hard or soft_before == -1 or soft_before > hard
    ):
        raise ExecutorError(f"invalid RLIMIT_NOFILE soft/hard relationship on {alias}")
    return report


def _supervisor_launch_command(
    *,
    python: str,
    supervisor: str,
    remote_lane: str,
    log: str,
    repo_dir: str,
    extra_environment: Mapping[str, str] | None = None,
) -> str:
    """Build a narrow launcher which raises nofile before creating the session."""
    extra = dict(extra_environment or {})
    protected = {
        *CLIENT_ENVIRONMENT,
        *SUPERVISOR_ENVIRONMENT,
        *contract.SEALED_RUNTIME_ENVIRONMENT,
    }
    if protected & set(extra):
        raise ExecutorError("supervisor launch environment cannot override invariants")
    environment = {
        **contract.SEALED_RUNTIME_ENVIRONMENT,
        **CLIENT_ENVIRONMENT,
        **SUPERVISOR_ENVIRONMENT,
        "PYTHONPATH": f"{repo_dir}/src:{repo_dir}",
        **extra,
    }
    if any(
        not isinstance(key, str) or not isinstance(value, str) or not key
        or "=" in key or "\x00" in key or "\x00" in value
        for key, value in environment.items()
    ):
        raise ExecutorError("invalid supervisor launch environment")
    argv = [python, supervisor, "run", "--lane", remote_lane]
    script = r'''import json,pathlib,resource,subprocess,sys
required=int(sys.argv[1]);log=pathlib.Path(sys.argv[2]);environment=json.loads(sys.argv[3]);argv=json.loads(sys.argv[4])
soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE);unlimited=resource.RLIM_INFINITY
if hard!=unlimited and hard<required: raise SystemExit(f'hard RLIMIT_NOFILE {hard} is below required {required}')
if soft!=unlimited and soft<required:
    try: resource.setrlimit(resource.RLIMIT_NOFILE,(required,hard))
    except (OSError,ValueError) as error: raise SystemExit(f'cannot raise soft RLIMIT_NOFILE {soft} to {required}: {error!r}')
raised,_=resource.getrlimit(resource.RLIMIT_NOFILE)
if raised!=unlimited and raised<required: raise SystemExit(f'soft RLIMIT_NOFILE {raised} is below required {required} after raise')
log.parent.mkdir(parents=True,exist_ok=True)
with log.open('ab',buffering=0) as output:
    process=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,env=environment,start_new_session=True,close_fds=True)
print(process.pid,flush=True)'''
    invocation = [
        "/usr/bin/env",
        "-i",
        *(f"{key}={value}" for key, value in sorted(environment.items())),
        python,
        "-c",
        script,
        str(REQUIRED_NOFILE_SOFT),
        log,
        json.dumps(environment, sort_keys=True, separators=(",", ":")),
        json.dumps(argv, separators=(",", ":")),
    ]
    return " ".join(shlex.quote(value) for value in invocation)


_STAGE_REPO_SCRIPT = r'''import hashlib,json,os,pathlib,shutil,stat,sys,tarfile,time,uuid
src,root,manifest_path,receipt_path=map(pathlib.Path,sys.argv[1:5])
manifest=json.loads(manifest_path.read_text())
sha=lambda p:'sha256:'+hashlib.sha256(p.read_bytes()).hexdigest()
if sha(src)!=manifest['repo_tar_sha256']: raise SystemExit('repo tar digest drift')
expected={r['path']:r for r in manifest['artifacts']}
if len(expected)!=len(manifest['artifacts']) or not expected or any(str(pathlib.PurePosixPath(name)) in ('','.') or pathlib.PurePosixPath(name).is_absolute() or '..' in pathlib.PurePosixPath(name).parts for name in expected): raise SystemExit('unsafe repo artifact path')
if any(int(record.get('mode',-1)) not in (0o444,0o555) for record in expected.values()): raise SystemExit('unsafe repo artifact mode')
expected_dirs=set()
for name in expected:
    parent=pathlib.PurePosixPath(name).parent
    while str(parent)!='.': expected_dirs.add(str(parent));parent=parent.parent
def verify_tree(base):
    if not base.is_dir() or base.is_symlink() or stat.S_IMODE(base.stat().st_mode)!=0o555: return False
    if any(p.is_symlink() for p in base.rglob('*')): return False
    actual_files={str(p.relative_to(base)) for p in base.rglob('*') if p.is_file()}
    actual_dirs={str(p.relative_to(base)) for p in base.rglob('*') if p.is_dir()}
    if actual_files!=set(expected) or actual_dirs!=expected_dirs: return False
    return all(sha(base/name)==record['sha256'] and stat.S_IMODE((base/name).stat().st_mode)==int(record['mode']) for name,record in expected.items()) and all(stat.S_IMODE((base/name).stat().st_mode)==0o555 for name in expected_dirs)
def remove_stage(stage):
    if not stage.exists(): return
    for directory in [stage,*[p for p in stage.rglob('*') if p.is_dir()]]:
        os.chmod(directory,0o700)
    shutil.rmtree(stage)
def seal_legacy_bytecode_tree(base):
    if not base.is_dir() or base.is_symlink(): return False
    if any(p.is_symlink() for p in base.rglob('*')): return False
    for name,record in expected.items():
        artifact=base/name
        if not artifact.is_file() or sha(artifact)!=record['sha256']: return False
    cache_dirs=[]
    for path in base.rglob('*'):
        relative=path.relative_to(base);parts=relative.parts
        if '__pycache__' in parts:
            if path.is_dir() and path.name=='__pycache__': cache_dirs.append(path)
            continue
        if path.is_file() and str(relative) not in expected: return False
        if path.is_dir() and str(relative) not in expected_dirs: return False
    for cache in sorted(set(cache_dirs),key=lambda p:len(p.parts),reverse=True):
        if cache.exists(): os.chmod(cache.parent,0o700);remove_stage(cache)
    for name,record in expected.items(): os.chmod(base/name,int(record['mode']))
    for directory in sorted([base/name for name in expected_dirs],key=lambda p:len(p.parts),reverse=True): os.chmod(directory,0o555)
    os.chmod(base,0o555)
    return verify_tree(base)
receipt=None
if receipt_path.exists():
    receipt=json.loads(receipt_path.read_text())
    if receipt.get('schema_version')!='a1-production-repo-stage-v1' or receipt.get('repo_tar_sha256')!=manifest['repo_tar_sha256'] or receipt.get('manifest_sha256')!=manifest['manifest_sha256']:
        raise SystemExit('repo stage receipt binds different bytes')
    if receipt.get('status')=='complete':
        if not verify_tree(root) and not seal_legacy_bytecode_tree(root): raise SystemExit('completed repo tree drift')
        raise SystemExit(0)
else:
    stage=root.parent/('.repo-stage-'+uuid.uuid4().hex)
    receipt={'schema_version':'a1-production-repo-stage-v1','status':'prepared','repo_tar_sha256':manifest['repo_tar_sha256'],'manifest_sha256':manifest['manifest_sha256'],'stage':str(stage),'created_at':time.time()}
    receipt_path.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(receipt_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as f: json.dump(receipt,f,sort_keys=True);f.flush();os.fsync(f.fileno())
if root.exists():
    if verify_tree(root) or seal_legacy_bytecode_tree(root): receipt['status']='complete'
    else: raise SystemExit('repo exists without valid completed stage')
else:
    stage=pathlib.Path(receipt['stage'])
    if stage.parent!=root.parent or not stage.name.startswith('.repo-stage-'): raise SystemExit('unsafe stage path')
    remove_stage(stage);stage.mkdir(parents=True)
    with tarfile.open(src) as archive:
        members=archive.getmembers()
        if {m.name for m in members}!=set(expected) or any(not m.isfile() for m in members): raise SystemExit('repo tar member drift')
        for member in members:
            destination=stage/member.name;destination.parent.mkdir(parents=True,exist_ok=True)
            data=archive.extractfile(member).read()
            fd=os.open(destination,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            with os.fdopen(fd,'wb') as f: f.write(data);f.flush();os.fsync(f.fileno())
            os.chmod(destination,int(expected[member.name]['mode']))
    if not all(sha(stage/name)==record['sha256'] for name,record in expected.items()): raise SystemExit('staged repo bytes drift')
    for directory in sorted([p for p in stage.rglob('*') if p.is_dir()],key=lambda p:len(p.parts),reverse=True): os.chmod(directory,0o555)
    os.chmod(stage,0o555)
    if not verify_tree(stage): raise SystemExit('staged repo seal verification failed')
    os.chmod(stage,0o755);os.rename(stage,root);os.chmod(root,0o555);receipt['status']='complete'
receipt['completed_at']=time.time()
tmp=receipt_path.parent/('.'+receipt_path.name+'.'+uuid.uuid4().hex+'.tmp')
with tmp.open('x') as f: json.dump(receipt,f,sort_keys=True);f.flush();os.fsync(f.fileno())
os.replace(tmp,receipt_path)
if not verify_tree(root): raise SystemExit('installed repo verification failed')'''


def _stage_repo(
    hosts: dict[str, Any],
    alias: str,
    repo_tar: Path,
    repo_sha: str,
    artifacts: Sequence[Mapping[str, Any]],
    temporary_path: Path,
    repo_dir: str,
) -> None:
    """Atomically install the exact repo tree and bind it with an O_EXCL receipt."""
    manifest = {
        "schema_version": "a1-production-repo-v1",
        "repo_tar_sha256": repo_sha,
        "artifacts": list(artifacts),
    }
    manifest["manifest_sha256"] = _digest(manifest)
    local_manifest = temporary_path / f"repo-manifest-{alias}.json"
    local_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    token = repo_sha.removeprefix("sha256:")
    remote_tar = f"{hosts['remote_root']}/operator/repo-{token}.tar"
    remote_manifest = f"{hosts['remote_root']}/operator/repo-{token}.json"
    receipt_path = f"{hosts['remote_root']}/receipts/repo-stage-{token}.json"
    _remote_install(hosts, alias, repo_tar, remote_tar, repo_sha)
    _remote_install(hosts, alias, local_manifest, remote_manifest, _sha256(local_manifest))
    script = _STAGE_REPO_SCRIPT
    result = _ssh(
        hosts,
        alias,
        " ".join(
            shlex.quote(value)
            for value in (
                hosts["python"],
                "-c",
                script,
                remote_tar,
                repo_dir,
                remote_manifest,
                receipt_path,
            )
        ),
    )
    if result.returncode != 0:
        raise ExecutorError(f"immutable repo stage failed on {alias}: {result.stderr.strip()}")


def _lane_payload(
    worker_id: str,
    lane: list[dict[str, Any]],
    *,
    hosts: dict[str, Any],
    operator_manifests: Mapping[str, Mapping[str, str]],
    repo_dir: str,
) -> dict[str, Any]:
    remote = hosts["remote_root"]
    materialized_lane = [
        _materialize_job_environment(command, repo_dir=repo_dir) for command in lane
    ]
    payload = {
        "schema_version": LANE_SCHEMA,
        "worker_id": worker_id,
        "host_alias": lane[0]["host_alias"],
        "gpu": lane[0]["gpu"],
        "repo_dir": repo_dir,
        "python": hosts["python"],
        "receipt_dir": f"{remote}/receipts",
        "quarantine_dir": f"{remote}/quarantine",
        "log_dir": f"{remote}/logs",
        "lane_lock": f"{remote}/locks/{worker_id}.lock",
        "client_environment": dict(CLIENT_ENVIRONMENT),
        "operator_manifests": dict(operator_manifests),
        "commands": materialized_lane,
    }
    payload["lane_sha256"] = _digest(payload)
    return payload


def _resume_receipt(
    receipt_path: Path, public: Mapping[str, Any], *, resume: bool
) -> dict[str, Any] | None:
    """Load only a receipt for this exact plan; stopped waves remain resumable."""
    if not receipt_path.exists():
        return None
    if not resume:
        raise ExecutorError("executor receipt exists; pass --resume for exact incomplete jobs")
    receipt = _load(receipt_path)
    if receipt.get("schema_version") != RECEIPT_SCHEMA:
        raise ExecutorError("executor receipt schema drift")
    if receipt.get("plan_sha256") != public["plan_sha256"]:
        raise ExecutorError("resume receipt binds a different execution plan")
    pending_worker = receipt.get("launch_pending_worker_id")
    if pending_worker is not None:
        raise ExecutorError(
            f"receipt has unresolved pending supervisor launch for {pending_worker}; "
            "exact-stop is required before resume"
        )
    return receipt


def execute(plan: dict[str, Any], *, receipt_path: Path, resume: bool) -> dict[str, Any]:
    _verify_plan_digest(plan)
    repo_source_root = _execution_repo_root(plan)
    private = plan["_private"]
    hosts = private["hosts"]
    lanes = private["lanes"]
    rendered = private["rendered"]
    public = _public(plan)
    expected_by_alias: dict[str, list[int]] = {}
    for lane in private["lanes"].values():
        expected_by_alias.setdefault(lane[0]["host_alias"], []).append(int(lane[0]["gpu"]))
    preflight = {
        alias: _preflight_host(hosts, alias, sorted(set(gpus)))
        for alias, gpus in sorted(expected_by_alias.items())
    }
    receipt = _resume_receipt(receipt_path, public, resume=resume)
    if receipt is None:
        receipt = dict(public)
        receipt.update(
            {
                "status": "prepared",
                "created_at": time.time(),
                "host_preflight": preflight,
                "lane_pids": {},
            }
        )
        _create_json(receipt_path, receipt)
    bridge = private.get("executor_bridge")
    if bridge is not None:
        existing_bridge = receipt.get("executor_bridge")
        if existing_bridge is not None and existing_bridge != bridge:
            raise ExecutorError("executor receipt bridge binding drift")
        receipt["executor_bridge"] = bridge
    receipt["host_preflight"] = preflight
    _atomic_json(receipt_path, receipt)

    with tempfile.TemporaryDirectory(prefix="a1-executor-") as temporary:
        temporary_path = Path(temporary)
        repo_tar = temporary_path / "repo.tar"
        artifacts = private["repo_artifacts"]
        if _digest(artifacts) != public["repo_artifacts_sha256"]:
            raise ExecutorError("repo artifact plan drift")
        repo_sha = _build_repo_tar(
            _repo_files(artifacts, repo_root=repo_source_root), repo_tar
        )
        repo_token = public["repo_artifacts_sha256"].removeprefix("sha256:")
        repo_dir = f"{hosts['remote_root']}/repo-{repo_token}"
        aliases = sorted({lane[0]["host_alias"] for lane in lanes.values()})
        required = rendered["required_artifacts"]
        stage_files = [
            *[(Path(item["path"]), item["path"], item["sha256"]) for item in required["checkpoints"]],
            *[(Path(item["path"]), item["path"], item["sha256"]) for item in required["rendered_opponent_mix"]],
        ]
        if _sha256(Path(required["seed_ledger"]["path"])) != public["live_seed_ledger_sha256"]:
            raise ExecutorError("live seed ledger changed after dry-run plan binding")
        attestation_sources = {
            item["output_attestation"]["source"]: item["output_attestation"]["source_file_sha256"]
            for lane in lanes.values() for item in lane
        }
        stage_files.extend((Path(path), path, digest) for path, digest in attestation_sources.items())
        operator_manifests = {
            name: {
                "path": f"{hosts['remote_root']}/operator/{record['remote_name']}",
                "sha256": record["sha256"],
            }
            for name, record in public["operator_manifests"].items()
        }
        operator_sources = {
            "lock": Path(public["lock"]),
            "render": Path(public["render"]),
        }
        for alias in aliases:
            _stage_repo(
                hosts, alias, repo_tar, repo_sha, artifacts, temporary_path, repo_dir
            )
            for name, source in operator_sources.items():
                _remote_install(
                    hosts,
                    alias,
                    source,
                    operator_manifests[name]["path"],
                    operator_manifests[name]["sha256"],
                )
            _remote_sync_append_only_ledger(
                hosts,
                alias,
                Path(required["seed_ledger"]["path"]),
                required["seed_ledger"]["path"],
                public["live_seed_ledger_sha256"],
            )
            for source, destination, digest in stage_files:
                _remote_install(hosts, alias, source, destination, digest)

        lane_pids: dict[str, int] = dict(receipt.get("lane_pids", {}))
        for worker_id, lane in sorted(lanes.items()):
            alias = lane[0]["host_alias"]
            payload = _lane_payload(
                worker_id,
                lane,
                hosts=hosts,
                operator_manifests=operator_manifests,
                repo_dir=repo_dir,
            )
            local_lane = temporary_path / f"{worker_id}.json"
            local_lane.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            remote_lane = f"{hosts['remote_root']}/lanes/{worker_id}.json"
            _remote_install(hosts, alias, local_lane, remote_lane, _sha256(local_lane))
            supervisor = f"{repo_dir}/tools/fleet/a1_lane_supervisor.py"
            log = f"{hosts['remote_root']}/logs/{worker_id}.supervisor.log"
            launch = _supervisor_launch_command(
                python=hosts["python"],
                supervisor=supervisor,
                remote_lane=remote_lane,
                log=log,
                repo_dir=repo_dir,
            )
            # Persist intent after the immutable lane exists but before the
            # detached spawn. If SSH returns a PID and the subsequent receipt
            # write fails, the caller can still exact-scan this lane by argv.
            receipt.update(
                {
                    "status": "launching",
                    "lane_pids": lane_pids,
                    "launch_pending_worker_id": worker_id,
                }
            )
            _atomic_json(receipt_path, receipt)
            result = _ssh(hosts, alias, launch)
            if result.returncode != 0 or not result.stdout.strip().splitlines()[-1].isdigit():
                raise ExecutorError(f"detached supervisor launch failed for {worker_id}")
            lane_pids[worker_id] = int(result.stdout.strip().splitlines()[-1])
            receipt.update({"status": "launching", "lane_pids": lane_pids})
            receipt.pop("launch_pending_worker_id", None)
            _atomic_json(receipt_path, receipt)
        acknowledgements: dict[str, Any] = {}
        for worker_id, lane in sorted(lanes.items()):
            alias = lane[0]["host_alias"]
            remote_lane = f"{hosts['remote_root']}/lanes/{worker_id}.json"
            supervisor = f"{repo_dir}/tools/fleet/a1_lane_supervisor.py"
            status_command = " ".join(
                shlex.quote(value)
                for value in (hosts["python"], supervisor, "status", "--lane", remote_lane)
            )
            command = f"kill -0 {int(lane_pids[worker_id])} && {status_command}"
            response = _ssh(hosts, alias, command)
            try:
                acknowledgement = json.loads(response.stdout) if response.returncode == 0 else None
            except json.JSONDecodeError:
                acknowledgement = None
            if not isinstance(acknowledgement, dict) or any(
                job.get("status") in {"failed", "invalid"}
                for job in acknowledgement.get("jobs", [])
            ):
                receipt.update(
                    {
                        "status": "launch_failed",
                        "launch_error": f"supervisor acknowledgement failed for {worker_id}",
                    }
                )
                _atomic_json(receipt_path, receipt)
                raise ExecutorError(f"supervisor acknowledgement failed for {worker_id}")
            acknowledgements[worker_id] = acknowledgement
    receipt.update(
        {
            "status": "launched",
            "launched_at": time.time(),
            "lane_pids": lane_pids,
            "lane_acknowledgements": acknowledgements,
        }
    )
    receipt.pop("launch_pending_worker_id", None)
    _atomic_json(receipt_path, receipt)
    return receipt


def _stop_helper_call(
    plan: dict[str, Any],
    worker_id: str,
    lane: list[dict[str, Any]],
    *,
    action: str,
    supervisor_pid: int,
) -> dict[str, Any]:
    private = plan["_private"]
    hosts = private["hosts"]
    alias = lane[0]["host_alias"]
    repo_token = plan["repo_artifacts_sha256"].removeprefix("sha256:")
    repo_dir = f"{hosts['remote_root']}/repo-{repo_token}"
    helper = f"{repo_dir}/tools/fleet/a1_stop_helper.py"
    remote_lane = f"{hosts['remote_root']}/lanes/{worker_id}.json"
    argv = (
        hosts["python"],
        helper,
        action,
        "--lane",
        remote_lane,
        "--supervisor-pid",
        str(supervisor_pid),
    )
    command = (
        f"if [ ! -f {shlex.quote(remote_lane)} ]; then "
        + (
            "echo 'recorded supervisor exists but immutable lane is missing' >&2; exit 9; "
            if supervisor_pid > 0
            else "printf '%s\\n' "
            + shlex.quote(
                json.dumps(
                    {
                        "worker_id": worker_id,
                        "status": "not_staged",
                        "supervisor_pid": None,
                        "generator_pids": {},
                    },
                    sort_keys=True,
                )
            )
            + "; "
        )
        + f"else env PYTHONPATH={shlex.quote(repo_dir + '/src:' + repo_dir)} "
        + " ".join(shlex.quote(value) for value in argv)
        + "; fi"
    )
    try:
        response = _ssh(
            hosts, alias, command, timeout_seconds=STOP_SSH_TIMEOUT_SECONDS
        )
    except subprocess.TimeoutExpired as error:
        raise ExecutorError(
            f"A1 {action} timed out for {worker_id} after "
            f"{STOP_SSH_TIMEOUT_SECONDS:g}s"
        ) from error
    if response.returncode != 0:
        detail = (response.stderr or response.stdout).strip()
        raise ExecutorError(f"A1 {action} refused for {worker_id}: {detail}")
    try:
        result = json.loads(response.stdout)
    except json.JSONDecodeError as error:
        raise ExecutorError(f"A1 {action} returned invalid JSON for {worker_id}") from error
    if not isinstance(result, dict) or result.get("worker_id") != worker_id:
        raise ExecutorError(f"A1 {action} identity drift for {worker_id}")
    return result


def stop_execution(
    plan: dict[str, Any], *, receipt_path: Path, go: bool
) -> dict[str, Any]:
    """Inspect, then stop only exact receipt-bound lane/generator process groups."""
    if not receipt_path.exists():
        raise ExecutorError("cannot stop A1: executor receipt is missing")
    receipt = _load(receipt_path)
    if receipt.get("schema_version") != RECEIPT_SCHEMA:
        raise ExecutorError("executor receipt schema drift")
    if receipt.get("plan_sha256") != plan["plan_sha256"]:
        raise ExecutorError("stop receipt binds a different execution plan")
    lane_pids = receipt.get("lane_pids")
    if not isinstance(lane_pids, dict):
        raise ExecutorError("executor receipt has no lane PID map")

    # All identities are checked fleet-wide before the first signal.  Each
    # remote stop revalidates immediately before signalling to close PID reuse.
    inspection: dict[str, Any] = {}
    for worker_id, lane in sorted(plan["_private"]["lanes"].items()):
        pid = lane_pids.get(worker_id, 0)
        if isinstance(pid, bool) or not isinstance(pid, int) or pid < 0:
            raise ExecutorError(f"invalid supervisor PID for {worker_id}")
        inspection[worker_id] = _stop_helper_call(
            plan, worker_id, lane, action="inspect", supervisor_pid=pid
        )
    if not go:
        return {
            "contract_sha256": plan["contract_sha256"],
            "status": "stop_dry_run",
            "lanes": inspection,
            "mps_preserved": True,
        }

    receipt.update({"status": "stopping", "stop_started_at": time.time()})
    _atomic_json(receipt_path, receipt)
    stopped: dict[str, Any] = {}
    try:
        for worker_id, lane in sorted(plan["_private"]["lanes"].items()):
            stopped[worker_id] = _stop_helper_call(
                plan,
                worker_id,
                lane,
                action="stop",
                supervisor_pid=int(lane_pids.get(worker_id, 0)),
            )
    except ExecutorError as error:
        receipt.update(
            {
                "status": "stop_failed",
                "stop_error": str(error),
                "stopped_lanes": stopped,
            }
        )
        _atomic_json(receipt_path, receipt)
        raise
    receipt.update(
        {
            "status": "stopped",
            "stopped_at": time.time(),
            "stopped_lanes": stopped,
            "mps_preserved": True,
        }
    )
    receipt.pop("launch_pending_worker_id", None)
    _atomic_json(receipt_path, receipt)
    return receipt


def status(plan: dict[str, Any], *, receipt_path: Path) -> dict[str, Any]:
    private = plan["_private"]
    hosts = private["hosts"]
    repo_token = plan["repo_artifacts_sha256"].removeprefix("sha256:")
    repo_dir = f"{hosts['remote_root']}/repo-{repo_token}"
    receipt_status = "not_launched"
    if receipt_path.exists():
        receipt = _load(receipt_path)
        if receipt.get("plan_sha256") != plan["plan_sha256"]:
            raise ExecutorError("status receipt binds a different execution plan")
        receipt_status = str(receipt.get("status", "invalid"))
    results = []
    for worker_id, lane in sorted(private["lanes"].items()):
        alias = lane[0]["host_alias"]
        supervisor = f"{repo_dir}/tools/fleet/a1_lane_supervisor.py"
        remote_lane = f"{hosts['remote_root']}/lanes/{worker_id}.json"
        command = " ".join(
            shlex.quote(value)
            for value in (hosts["python"], supervisor, "status", "--lane", remote_lane)
        )
        response = _ssh(hosts, alias, command)
        if response.returncode != 0:
            results.append({"worker_id": worker_id, "host_alias": alias, "status": "unreachable_or_invalid", "error": response.stderr.strip()})
            continue
        try:
            results.append(json.loads(response.stdout))
        except json.JSONDecodeError:
            results.append({"worker_id": worker_id, "host_alias": alias, "status": "invalid_status_output"})
    counts: dict[str, int] = {}
    for lane in results:
        for job in lane.get("jobs", []):
            key = str(job.get("status", "unknown"))
            counts[key] = counts.get(key, 0) + 1
    return {
        "contract_sha256": plan["contract_sha256"],
        "executor_status": receipt_status,
        "lanes": results,
        "job_status_counts": counts,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "status", "stop"):
        item = sub.add_parser(name)
        item.add_argument("--lock", required=True, type=Path)
        item.add_argument("--render", required=True, type=Path)
        item.add_argument("--hosts", required=True, type=Path)
        item.add_argument("--receipt", required=True, type=Path)
    run = sub.choices["run"]
    run.add_argument("--resume", action="store_true")
    run.add_argument("--go", action="store_true", help="stage and launch; default dry-run")
    sub.choices["stop"].add_argument(
        "--go", action="store_true", help="stop exact A1 process groups; default dry-run"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plan = build_plan(lock_path=args.lock, render_path=args.render, hosts_path=args.hosts, receipt_path=args.receipt)
        if args.command == "status":
            result = status(plan, receipt_path=args.receipt)
        elif args.command == "stop":
            result = stop_execution(plan, receipt_path=args.receipt, go=bool(args.go))
        elif not args.go:
            result = _public(plan)
        else:
            result = execute(plan, receipt_path=args.receipt, resume=bool(args.resume))
    except ExecutorError as error:
        print(f"REFUSING: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
