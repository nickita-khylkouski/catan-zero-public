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
LANE_SCHEMA = "a1-production-lane-v1"
SAFE_ALIAS = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
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


class ExecutorError(RuntimeError):
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
        expected_environment = {
            "CUDA_VISIBLE_DEVICES": str(job["gpu"]),
            **CLIENT_ENVIRONMENT,
            "CATAN_SEED_LEDGER": lock["fleet"]["seed_ledger"]["path"],
            "CATAN_A1_CONTRACT_SHA256": lock["contract_sha256"],
        }
        if command.get("environment") != expected_environment:
            raise ExecutorError(f"exact client environment drift for {job_id}")
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


def _repo_artifacts(rendered: dict[str, Any]) -> list[dict[str, Any]]:
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
            relative = path.relative_to(_REPO_ROOT)
        except ValueError as error:
            raise ExecutorError(f"runtime artifact is outside canonical repo: {path}") from error
        files[str(relative)] = path
    supervisor = (_REPO_ROOT / "tools/fleet/a1_lane_supervisor.py").resolve()
    files[str(supervisor.relative_to(_REPO_ROOT))] = supervisor
    return [
        {
            "path": key,
            "sha256": _sha256(files[key]),
            "mode": 0o555 if os.access(files[key], os.X_OK) else 0o444,
        }
        for key in sorted(files)
    ]


def _repo_files(artifacts: Sequence[Mapping[str, Any]]) -> list[Path]:
    return [(_REPO_ROOT / str(record["path"])).resolve() for record in artifacts]


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
) -> dict[str, Any]:
    lock, rendered, lanes = verify_render(
        lock_path, render_path, verify_lock_fn=verify_lock_fn
    )
    repo_artifacts = _repo_artifacts(rendered)
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


def _ssh(hosts: dict[str, Any], alias: str, remote_command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=10", "-i", hosts["ssh_key"],
            f"{hosts['ssh_user']}@{hosts['hosts'][alias]}", remote_command,
        ],
        text=True,
        capture_output=True,
        check=False,
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
    """Read-only launch preflight: topology, idle compute plane, durable MPS."""
    script = r'''import importlib.metadata,json,os,pathlib,subprocess,sys
expected=json.loads(sys.argv[1])
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
show=run('systemctl','show','nvidia-mps.service','--property=ActiveState,UnitFileState,MainPID,Environment')
if show.returncode: raise SystemExit('cannot inspect nvidia-mps.service: '+show.stderr)
properties={}
for line in show.stdout.splitlines():
    if '=' in line:
        key,value=line.split('=',1);properties[key]=value
required_properties={'ActiveState','UnitFileState','MainPID','Environment'}
if not required_properties.issubset(properties): raise SystemExit('incomplete nvidia-mps.service properties: '+repr(properties))
active=properties['ActiveState'];enabled=properties['UnitFileState'];main_pid_raw=properties['MainPID'];environment=properties['Environment']
if active!='active' or enabled!='enabled': raise SystemExit(f'MPS service not active+enabled: {active}/{enabled}')
try: main_pid=int(main_pid_raw)
except ValueError: raise SystemExit('invalid MPS MainPID: '+main_pid_raw)
if main_pid<=0 or not pathlib.Path(f'/proc/{main_pid}').exists(): raise SystemExit('MPS MainPID is not live')
required={'CUDA_MPS_PIPE_DIRECTORY':'/tmp/mps_pipe_host','CUDA_MPS_LOG_DIRECTORY':'/tmp/mps_log_host'}
for key,value in required.items():
    if f'{key}={value}' not in environment: raise SystemExit(f'MPS service {key} drift')
    path=pathlib.Path(value)
    if not path.is_dir() or not os.access(path,os.R_OK|os.W_OK|os.X_OK): raise SystemExit(f'MPS directory inaccessible: {path}')
try: rust_version=importlib.metadata.version('catanatron-rs')
except importlib.metadata.PackageNotFoundError: rust_version='unknown'
print(json.dumps({'gpu_indices':indices,'compute_apps':'mps_only_or_empty','mps_active':active,'mps_enabled':enabled,'mps_main_pid':main_pid,'client_environment':required,'python':sys.executable,'torch_version':str(torch.__version__),'torch_cuda_version':str(torch.version.cuda),'catanatron_rs_version':rust_version},sort_keys=True))'''
    command = " ".join(
        shlex.quote(value)
        for value in (hosts["python"], "-c", script, json.dumps(sorted(expected_gpus)))
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
    return report


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
    script = r'''import hashlib,json,os,pathlib,shutil,sys,tarfile,time,uuid
src,root,manifest_path,receipt_path=map(pathlib.Path,sys.argv[1:5])
manifest=json.loads(manifest_path.read_text())
sha=lambda p:'sha256:'+hashlib.sha256(p.read_bytes()).hexdigest()
if sha(src)!=manifest['repo_tar_sha256']: raise SystemExit('repo tar digest drift')
expected={r['path']:r for r in manifest['artifacts']}
def verify_tree():
    if not root.is_dir(): return False
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    if actual!=set(expected): return False
    return all(sha(root/name)==record['sha256'] for name,record in expected.items())
receipt=None
if receipt_path.exists():
    receipt=json.loads(receipt_path.read_text())
    if receipt.get('schema_version')!='a1-production-repo-stage-v1' or receipt.get('repo_tar_sha256')!=manifest['repo_tar_sha256'] or receipt.get('manifest_sha256')!=manifest['manifest_sha256']:
        raise SystemExit('repo stage receipt binds different bytes')
    if receipt.get('status')=='complete':
        if not verify_tree(): raise SystemExit('completed repo tree drift')
        raise SystemExit(0)
else:
    stage=root.parent/('.repo-stage-'+uuid.uuid4().hex)
    receipt={'schema_version':'a1-production-repo-stage-v1','status':'prepared','repo_tar_sha256':manifest['repo_tar_sha256'],'manifest_sha256':manifest['manifest_sha256'],'stage':str(stage),'created_at':time.time()}
    receipt_path.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(receipt_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as f: json.dump(receipt,f,sort_keys=True);f.flush();os.fsync(f.fileno())
if root.exists():
    if verify_tree():
        receipt['status']='complete'
    else: raise SystemExit('repo exists without valid completed stage')
else:
    stage=pathlib.Path(receipt['stage'])
    if stage.parent!=root.parent or not stage.name.startswith('.repo-stage-'): raise SystemExit('unsafe stage path')
    shutil.rmtree(stage,ignore_errors=True);stage.mkdir(parents=True)
    with tarfile.open(src) as archive:
        members=archive.getmembers()
        if {m.name for m in members}!=set(expected) or any(not m.isfile() for m in members): raise SystemExit('repo tar member drift')
        for member in members:
            destination=stage/member.name;destination.parent.mkdir(parents=True,exist_ok=True)
            data=archive.extractfile(member).read()
            fd=os.open(destination,os.O_WRONLY|os.O_CREAT|os.O_EXCL,int(expected[member.name]['mode']))
            with os.fdopen(fd,'wb') as f: f.write(data);f.flush();os.fsync(f.fileno())
    if not all(sha(stage/name)==record['sha256'] for name,record in expected.items()): raise SystemExit('staged repo bytes drift')
    os.rename(stage,root);receipt['status']='complete'
receipt['completed_at']=time.time()
tmp=receipt_path.parent/('.'+receipt_path.name+'.'+uuid.uuid4().hex+'.tmp')
with tmp.open('x') as f: json.dump(receipt,f,sort_keys=True);f.flush();os.fsync(f.fileno())
os.replace(tmp,receipt_path)
if not verify_tree(): raise SystemExit('installed repo verification failed')'''
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
        "commands": lane,
    }
    payload["lane_sha256"] = _digest(payload)
    return payload


def execute(plan: dict[str, Any], *, receipt_path: Path, resume: bool) -> dict[str, Any]:
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
    if receipt_path.exists():
        if not resume:
            raise ExecutorError("executor receipt exists; pass --resume for exact incomplete jobs")
        receipt = _load(receipt_path)
        if receipt.get("plan_sha256") != public["plan_sha256"]:
            raise ExecutorError("resume receipt binds a different execution plan")
    else:
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
    receipt["host_preflight"] = preflight
    _atomic_json(receipt_path, receipt)

    with tempfile.TemporaryDirectory(prefix="a1-executor-") as temporary:
        temporary_path = Path(temporary)
        repo_tar = temporary_path / "repo.tar"
        artifacts = private["repo_artifacts"]
        if _digest(artifacts) != public["repo_artifacts_sha256"]:
            raise ExecutorError("repo artifact plan drift")
        repo_sha = _build_repo_tar(_repo_files(artifacts), repo_tar)
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
            launch = (
                f"mkdir -p {shlex.quote(str(Path(log).parent))} && "
                f"setsid env "
                f"CUDA_MPS_PIPE_DIRECTORY={shlex.quote(CLIENT_ENVIRONMENT['CUDA_MPS_PIPE_DIRECTORY'])} "
                f"CUDA_MPS_LOG_DIRECTORY={shlex.quote(CLIENT_ENVIRONMENT['CUDA_MPS_LOG_DIRECTORY'])} "
                f"PYTHONPATH={shlex.quote(repo_dir + '/src:' + repo_dir)} "
                f"{shlex.quote(hosts['python'])} {shlex.quote(supervisor)} run --lane {shlex.quote(remote_lane)} "
                f">{shlex.quote(log)} 2>&1 </dev/null & echo $!"
            )
            result = _ssh(hosts, alias, launch)
            if result.returncode != 0 or not result.stdout.strip().splitlines()[-1].isdigit():
                raise ExecutorError(f"detached supervisor launch failed for {worker_id}")
            lane_pids[worker_id] = int(result.stdout.strip().splitlines()[-1])
            receipt.update({"status": "launching", "lane_pids": lane_pids})
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
    for name in ("run", "status"):
        item = sub.add_parser(name)
        item.add_argument("--lock", required=True, type=Path)
        item.add_argument("--render", required=True, type=Path)
        item.add_argument("--hosts", required=True, type=Path)
        item.add_argument("--receipt", required=True, type=Path)
    run = sub.choices["run"]
    run.add_argument("--resume", action="store_true")
    run.add_argument("--go", action="store_true", help="stage and launch; default dry-run")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plan = build_plan(lock_path=args.lock, render_path=args.render, hosts_path=args.hosts, receipt_path=args.receipt)
        if args.command == "status":
            result = status(plan, receipt_path=args.receipt)
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
