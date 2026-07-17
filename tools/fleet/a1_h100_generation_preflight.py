#!/usr/bin/env python3
"""Bootstrap and attest the canonical six-node 8xH100 generation fleet.

This tool deliberately stops before generation.  It gives one coordinator a
single idempotent path for installing an exact public GitHub revision and then
proving the runtime, checkpoint, file-descriptor limits, single-CUDA-owner
recipe, GPU topology, output ownership, and disjoint seed authority on every
node.

Run this *on the configured coordinator*.  All fleet traffic then travels
directly coordinator -> H100; checkpoint and wheel bytes never transit an
operator laptop.

``remote_repo`` is an immutable deployment directory.  Re-running bootstrap
with the same commit re-installs and re-attests the sealed native wheel;
requesting a different commit fails closed.  An upgrade must use a fresh
manifest path (for example fleet-v2), leaving the old runtime untouched until
the new path passes this preflight.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import tempfile
from typing import Any, Sequence


SCHEMA = "a1-h100-generation-fleet-v1"
AUTHORITY = "catan-h100-8x6-v1"
RECEIPT_SCHEMA = "a1-h100-8x6-preflight-v1"
EXPECTED_HOSTS = {
    "h100-8a": ("192.222.53.175", 8),
    "h100-8b": ("192.222.54.137", 8),
    "h100-8c": ("209.20.158.117", 8),
    "h100-8d": ("192.222.55.12", 8),
    "h100-8e": ("209.20.156.201", 8),
    "h100-8f": ("192.222.52.228", 8),
}
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
SAFE_ADDRESS = re.compile(r"[A-Za-z0-9][A-Za-z0-9.:-]*\Z")
GITHUB_RE = re.compile(
    r"https://github\.com/[A-Za-z0-9-]+/[A-Za-z0-9_.-]+\.git\Z"
)
REQUIRED_NOFILE = 65_536
REQUIRED_NATIVE_CAPABILITIES = (
    "belief_target_evidence",
    "coherent_public_belief_search",
    "forced_root_trajectory_only",
    "initial_road_d1_scope",
    "policy_temperature_semantics",
    "public_award_feature_parity",
    "sigma_reference_visits",
)
REQUIRED_LEARNER_ENTITY_ADAPTER = (
    "rust_entity_adapter_v6_exact_actor_resources_initial_road_two_hop"
)


class PreflightError(RuntimeError):
    """Fleet bootstrap or admission failed closed."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _absolute(value: Any, *, field: str) -> str:
    path = Path(str(value))
    if not path.is_absolute() or any(part in {".", ".."} for part in path.parts):
        raise PreflightError(f"{field} must be a canonical absolute path")
    if any(character in str(path) for character in "\n\r\0"):
        raise PreflightError(f"{field} contains an unsafe character")
    return str(path)


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PreflightError(f"cannot read fleet manifest: {error}") from error
    if not isinstance(value, dict):
        raise PreflightError("fleet manifest must be one JSON object")
    if value.get("schema_version") != SCHEMA or value.get("fleet_authority") != AUTHORITY:
        raise PreflightError("fleet manifest schema/authority drift")
    if value.get("coordinator_alias") != "h100-8a":
        raise PreflightError("coordinator_alias must be h100-8a")
    if not GITHUB_RE.fullmatch(str(value.get("git_url", ""))):
        raise PreflightError("git_url must be a canonical public GitHub .git URL")
    if not SAFE_NAME.fullmatch(str(value.get("ssh_user", ""))):
        raise PreflightError("unsafe ssh_user")
    checking = str(value.get("strict_host_key_checking", ""))
    if checking not in {"yes", "accept-new"}:
        raise PreflightError("strict_host_key_checking must be yes or accept-new")
    for field in (
        "ssh_key",
        "remote_repo",
        "remote_python",
        "remote_root",
    ):
        value[field] = _absolute(value.get(field), field=field)
    minima = value.get("resource_minima")
    required_minima = {
        "physical_cores",
        "physical_cores_per_gpu",
        "numa_nodes",
        "usable_logical_cpus_per_gpu",
        "ram_bytes",
        "disk_available_bytes",
        "disk_available_inodes",
    }
    if not isinstance(minima, dict) or set(minima) != required_minima:
        raise PreflightError("resource_minima shape drift")
    for field in required_minima:
        if isinstance(minima[field], bool) or not isinstance(minima[field], int) or minima[field] <= 0:
            raise PreflightError(f"resource_minima.{field} must be a positive integer")
    for section in ("checkpoint", "native_wheel"):
        record = value.get(section)
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise PreflightError(f"{section} must contain exactly path and sha256")
        record["path"] = _absolute(record["path"], field=f"{section}.path")
        if not SHA_RE.fullmatch(str(record["sha256"])):
            raise PreflightError(f"{section}.sha256 is malformed")
    seed = value.get("seed_authority")
    if not isinstance(seed, dict) or set(seed) != {
        "path",
        "range_start",
        "host_stride",
        "gpu_stride",
    }:
        raise PreflightError("seed_authority shape drift")
    seed["path"] = _absolute(seed["path"], field="seed_authority.path")
    for field in ("range_start", "host_stride", "gpu_stride"):
        if isinstance(seed[field], bool) or int(seed[field]) <= 0:
            raise PreflightError(f"seed_authority.{field} must be positive")
        seed[field] = int(seed[field])
    if seed["host_stride"] != 8 * seed["gpu_stride"]:
        raise PreflightError("host_stride must equal eight gpu_stride blocks")
    raw_hosts = value.get("hosts")
    if not isinstance(raw_hosts, list):
        raise PreflightError("hosts must be a list")
    hosts: list[dict[str, Any]] = []
    for raw in raw_hosts:
        if not isinstance(raw, dict):
            raise PreflightError("host entries must be objects")
        alias, address = str(raw.get("alias", "")), str(raw.get("address", ""))
        if not SAFE_NAME.fullmatch(alias) or not SAFE_ADDRESS.fullmatch(address):
            raise PreflightError("unsafe host identity")
        hosts.append(
            {
                "alias": alias,
                "address": address,
                "gpu_count": int(raw.get("gpu_count", 0)),
                "accelerator": str(raw.get("accelerator", "")),
            }
        )
    actual = {
        host["alias"]: (host["address"], host["gpu_count"]) for host in hosts
    }
    if actual != EXPECTED_HOSTS or len(hosts) != len(EXPECTED_HOSTS):
        raise PreflightError(f"six-node fleet mapping drift: {actual}")
    if any(host["accelerator"] != "NVIDIA H100 80GB HBM3" for host in hosts):
        raise PreflightError("fleet accelerator identity drift")
    value["hosts"] = hosts
    value["manifest_sha256"] = _digest(
        {key: item for key, item in value.items() if key != "manifest_sha256"}
    )
    return value


def _seed_authority(
    manifest: dict[str, Any], *, repo_commit: str, host: dict[str, Any]
) -> dict[str, Any]:
    index = [item["alias"] for item in manifest["hosts"]].index(host["alias"])
    seed = manifest["seed_authority"]
    start = seed["range_start"] + index * seed["host_stride"]
    lanes = [
        {
            "gpu": gpu,
            "start": start + gpu * seed["gpu_stride"],
            "end": start + (gpu + 1) * seed["gpu_stride"],
        }
        for gpu in range(8)
    ]
    return {
        "schema_version": "a1-h100-seed-authority-v1",
        "fleet_authority": AUTHORITY,
        "manifest_sha256": manifest["manifest_sha256"],
        "repo_commit": repo_commit,
        "host_alias": host["alias"],
        "address": host["address"],
        "output_root": f"{manifest['remote_root'].rstrip('/')}/{host['alias']}",
        "host_range": {"start": start, "end": start + seed["host_stride"]},
        "gpu_ranges": lanes,
    }


def _ssh_base(manifest: dict[str, Any], host: dict[str, Any]) -> list[str]:
    return [
        "ssh",
        "-i",
        manifest["ssh_key"],
        "-o",
        "BatchMode=yes",
        "-o",
        f"StrictHostKeyChecking={manifest['strict_host_key_checking']}",
        "-o",
        "ConnectTimeout=12",
        f"{manifest['ssh_user']}@{host['address']}",
    ]


def _run(
    argv: Sequence[str], *, input_text: str | None = None, timeout: int = 180
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        input=input_text,
        text=True,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )


REMOTE_INSPECT = r'''from __future__ import annotations
import base64, hashlib, json, os, pathlib, resource, subprocess, sys

payload = json.loads(base64.urlsafe_b64decode(sys.argv[1] + "=" * (-len(sys.argv[1]) % 4)))

def run(argv):
    p = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p.returncode, p.stdout.strip(), p.stderr.strip()

def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

checks, details = {}, {}
def check(name, passed, detail):
    checks[name] = bool(passed); details[name] = str(detail)

rc, out, err = run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
names = out.splitlines() if rc == 0 else []
check("gpu_count", len(names) == payload["gpu_count"], len(names))
check("gpu_model", bool(names) and all(name.strip() == payload["accelerator"] for name in names), names)

affinity = set(os.sched_getaffinity(0))
physical = set()
for cpu in affinity:
    topology = pathlib.Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
    try:
        package = int((topology / "physical_package_id").read_text())
        core = int((topology / "core_id").read_text())
    except (OSError, ValueError):
        continue
    physical.add((package, core))
numa_nodes = [path for path in pathlib.Path("/sys/devices/system/node").glob("node[0-9]*") if path.is_dir()]
page_size = os.sysconf("SC_PAGE_SIZE")
ram_bytes = page_size * os.sysconf("SC_PHYS_PAGES")
disk = os.statvfs(str(pathlib.Path(payload["remote_repo"]).parent))
resources = {
    "physical_cores": len(physical),
    "logical_cpus": len(affinity),
    "numa_nodes": len(numa_nodes),
    "ram_bytes": ram_bytes,
    "disk_available_bytes": disk.f_bavail * disk.f_frsize,
    "disk_available_inodes": disk.f_favail,
}
details["resources"] = resources
minima = payload["resource_minima"]
check("physical_cores", resources["physical_cores"] >= minima["physical_cores"], resources)
check("numa_nodes", resources["numa_nodes"] >= minima["numa_nodes"], resources)
check("logical_cpu_lane_capacity", resources["logical_cpus"] >= payload["gpu_count"] * minima["usable_logical_cpus_per_gpu"], resources)
check("ram_capacity", resources["ram_bytes"] >= minima["ram_bytes"], resources)
check("disk_headroom", resources["disk_available_bytes"] >= minima["disk_available_bytes"] and resources["disk_available_inodes"] >= minima["disk_available_inodes"], resources)

def parse_cpu_list(raw):
    result=set()
    for item in raw.strip().split(','):
        if not item: continue
        first,separator,last=item.partition('-');start=int(first);end=int(last) if separator else start
        result.update(range(start,end+1))
    return result

rc, bus_out, bus_err = run(["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader,nounits"])
gpu_nodes={}
if rc == 0:
    for line in bus_out.splitlines():
        raw_index, separator, raw_bus = line.partition(',')
        if not separator: continue
        index=int(raw_index.strip());tail=raw_bus.strip().lower().split(':',1)[-1]
        matches=list(pathlib.Path('/sys/bus/pci/devices').glob(f'*:{tail}'))
        if len(matches)==1:
            try: gpu_nodes[index]=int((matches[0]/'numa_node').read_text())
            except (OSError,ValueError): pass
lane_topology=[]
for index in range(payload["gpu_count"]):
    node=gpu_nodes.get(index,-1);local_gpus=sorted(gpu for gpu,value in gpu_nodes.items() if value==node)
    groups={}
    if node>=0:
        try: node_cpus=parse_cpu_list(pathlib.Path(f'/sys/devices/system/node/node{node}/cpulist').read_text()) & affinity
        except OSError: node_cpus=set()
        for cpu in sorted(node_cpus):
            try: siblings=parse_cpu_list(pathlib.Path(f'/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list').read_text()) & node_cpus
            except OSError: siblings=set()
            if siblings: groups[tuple(sorted(siblings))]=siblings
    ordered=[groups[key] for key in sorted(groups)]
    if index in local_gpus:
        lane=local_gpus.index(index);start=len(ordered)*lane//len(local_gpus);end=len(ordered)*(lane+1)//len(local_gpus)
        selected=ordered[start:end]
    else: selected=[]
    lane_topology.append({"gpu":index,"numa_node":node,"physical_cores":len(selected),"logical_cpus":len(set().union(*selected) if selected else set())})
details["gpu_cpu_topology"] = lane_topology
check("gpu_numa_topology", set(gpu_nodes)==set(range(payload["gpu_count"])) and all(item["numa_node"]>=0 and item["physical_cores"]>=minima["physical_cores_per_gpu"] and item["logical_cpus"]>=minima["usable_logical_cpus_per_gpu"] for item in lane_topology), lane_topology or bus_err)

repo = pathlib.Path(payload["remote_repo"])
if (repo / ".git").is_dir():
    rc, head, _ = run(["git", "-C", str(repo), "rev-parse", "HEAD"])
    check("code_sha", rc == 0 and head == payload["repo_commit"], head)
    rc, status, _ = run(["git", "-C", str(repo), "status", "--porcelain=v1", "--untracked-files=all"])
    check("code_clean", rc == 0 and not status, status or "clean")
    rc, origin, _ = run(["git", "-C", str(repo), "remote", "get-url", "origin"])
    check("github_origin", rc == 0 and origin == payload["git_url"], origin)
else:
    for name in ("code_sha", "code_clean", "github_origin"):
        check(name, False, "remote repository missing")

python = pathlib.Path(payload["remote_python"])
runtime_program = """import hashlib, importlib, json, pathlib, sys
from importlib.metadata import distribution
value={"python":sys.version.split()[0]}
try:
 import torch
 value.update(torch=torch.__version__,cuda=str(torch.version.cuda),cuda_available=torch.cuda.is_available())
except Exception as error: value["torch_error"]=repr(error)
try:
 import catanatron_rs
 metadata=distribution("catanatron-rs")
 native=importlib.import_module("catanatron_rs.catanatron_rs")
 extension_path=pathlib.Path(native.__file__).resolve(strict=True)
 digest=hashlib.sha256()
 with extension_path.open("rb") as handle:
  for block in iter(lambda:handle.read(1<<20),b""): digest.update(block)
 direct_url_raw=metadata.read_text("direct_url.json")
 direct_url=json.loads(direct_url_raw) if direct_url_raw is not None else None
 archive=direct_url.get("archive_info") if isinstance(direct_url,dict) else None
 stated=set()
 if isinstance(archive,dict):
  direct_hash=archive.get("hash")
  if isinstance(direct_hash,str): stated.add(direct_hash)
  hashes=archive.get("hashes")
  if isinstance(hashes,dict) and isinstance(hashes.get("sha256"),str): stated.add("sha256="+hashes["sha256"])
 caps=getattr(catanatron_rs,"gumbel_search_capabilities",lambda:())()
 adapter_fn=getattr(catanatron_rs,"supported_action_context_adapter_versions",None)
 adapters=sorted(map(str,adapter_fn())) if callable(adapter_fn) else []
 value.update(catanatron_rs=metadata.version,capabilities=sorted(caps),action_context_adapters=adapters,installed_wheel_hashes=sorted(stated),extension_path=str(extension_path),extension_sha256=digest.hexdigest())
except Exception as error: value["rust_error"]=repr(error)
try:
 import catan_zero
 value["catan_zero_file"]=str(pathlib.Path(catan_zero.__file__).resolve())
except Exception as error: value["catan_zero_error"]=repr(error)
print(json.dumps(value,sort_keys=True))"""
if python.is_file() and os.access(python, os.X_OK):
    rc, out, err = run([str(python), "-c", runtime_program])
    try: runtime = json.loads(out) if rc == 0 else {"error": err}
    except Exception: runtime = {"error": out or err}
else: runtime = {"error": "remote python missing"}
details["runtime"] = runtime
check("python_runtime", runtime.get("python") == payload["python_version"], runtime.get("python"))
check("torch_runtime", runtime.get("torch") == payload["torch_version"] and runtime.get("cuda") == payload["torch_cuda_version"] and runtime.get("cuda_available") is True, runtime)
check("rust_runtime", runtime.get("catanatron_rs") == payload["rust_version"] and set(payload["required_capabilities"]) <= set(runtime.get("capabilities", [])), runtime)
check("rust_wheel_install_binding", runtime.get("installed_wheel_hashes") == ["sha256=" + payload["wheel_sha256"]], runtime)
check("rust_extension_identity", runtime.get("extension_sha256") == payload["rust_extension_sha256"], runtime)
check("rust_v6_action_context", payload["required_entity_adapter"] in runtime.get("action_context_adapters", []), runtime)
repo_prefix = str(repo / "src") + os.sep
check("python_code_origin", str(runtime.get("catan_zero_file", "")).startswith(repo_prefix), runtime.get("catan_zero_file"))

checkpoint = pathlib.Path(payload["checkpoint_path"])
check("checkpoint_sha", checkpoint.is_file() and sha(checkpoint) == payload["checkpoint_sha256"], sha(checkpoint) if checkpoint.is_file() else "missing")
wheel = pathlib.Path(payload["wheel_path"])
check("wheel_sha", wheel.is_file() and sha(wheel) == payload["wheel_sha256"], sha(wheel) if wheel.is_file() else "missing")

soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
details["nofile"] = {"soft":soft,"hard":hard}
check("nofile_hard", hard >= payload["required_nofile"], details["nofile"])
limit_program = f"""import resource, sys
sys.path[:0] = [{str(repo)!r}, {str(repo / "src")!r}]
from tools.generate import _ensure_runtime_limits
_ensure_runtime_limits()
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
assert soft >= {payload["required_nofile"]!r} and hard >= {payload["required_nofile"]!r}
"""
rc, out, err = run([str(python), "-c", limit_program]) if python.is_file() else (1, "", "python missing")
check("nofile_wrapper", rc == 0, err or out or "wrapper raised soft limit")

recipe_path = repo / "configs/generation/coherent_public_n128.schema20.json"
try: recipe = json.loads(recipe_path.read_text(encoding="utf-8"))["fields"]
except Exception as error: recipe = {"error": repr(error)}
single_owner = (
    recipe.get("device") == "cuda"
    and recipe.get("workers") == 24
    and recipe.get("eval_server") is True
    and recipe.get("eval_server_transport") == "mp_queue"
    and recipe.get("eval_server_local_fallback") is False
    and recipe.get("fleet_pipelines_per_gpu") == 1
)
check("single_cuda_owner_recipe", single_owner, recipe)

rc, compute_out, compute_err = run(["nvidia-smi", "--query-compute-apps=process_name", "--format=csv,noheader"])
mps_compute = [line.strip() for line in compute_out.splitlines() if "nvidia-cuda-mps-server" in line]
active_rc, active_out, _ = run(["systemctl", "is-active", "nvidia-mps.service"])
enabled_rc, enabled_out, _ = run(["systemctl", "is-enabled", "nvidia-mps.service"])
mps_paths = [str(path) for path in (pathlib.Path("/tmp/mps_pipe_host"), pathlib.Path("/tmp/mps_log_host")) if path.exists()]
mps_retired = (
    rc == 0
    and not mps_compute
    and (active_rc != 0 or active_out not in {"active", "activating"})
    and (enabled_rc != 0 or enabled_out not in {"enabled", "enabled-runtime"})
    and not mps_paths
)
check("mps_retired", mps_retired, {"active": active_out or active_rc, "enabled": enabled_out or enabled_rc, "compute": mps_compute, "paths": mps_paths, "error": compute_err})

authority_path = pathlib.Path(payload["seed_authority_path"])
try: authority = json.loads(authority_path.read_text(encoding="utf-8"))
except Exception as error: authority = {"error": repr(error)}
check("seed_authority", authority == payload["seed_authority"], authority)
output = pathlib.Path(payload["seed_authority"]["output_root"])
check("output_root", output.is_dir() and not output.is_symlink() and os.access(output, os.W_OK | os.X_OK), output)

print(json.dumps({"checks":checks,"details":details,"ready":all(checks.values())},sort_keys=True))
'''


def _runtime_contract_for_commit(
    manifest: dict[str, Any], repo_commit: str
) -> dict[str, Any]:
    """Read runtime authority from the exact public GitHub commit, not local HEAD."""

    with tempfile.TemporaryDirectory(prefix="catan-runtime-authority-") as raw:
        checkout = Path(raw)
        commands = (
            ["git", "init", "--quiet", str(checkout)],
            ["git", "-C", str(checkout), "remote", "add", "origin", manifest["git_url"]],
            [
                "git", "-C", str(checkout), "fetch", "--quiet", "--depth=1",
                "origin", repo_commit,
            ],
            ["git", "-C", str(checkout), "checkout", "--quiet", "--detach", repo_commit],
        )
        for command in commands:
            completed = _run(command, timeout=180)
            if completed.returncode != 0:
                raise PreflightError(
                    "cannot resolve exact GitHub runtime authority: "
                    + (completed.stderr or completed.stdout)
                )
        path = checkout / "configs/runtime/a1_production_runtime.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise PreflightError(f"runtime contract at GitHub commit is invalid: {error}") from error
    if not isinstance(value, dict):
        raise PreflightError("runtime contract at GitHub commit is not an object")
    return value


def _inspect_host(
    manifest: dict[str, Any],
    *,
    repo_commit: str,
    host: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    authority = _seed_authority(manifest, repo_commit=repo_commit, host=host)
    payload = {
        "git_url": manifest["git_url"],
        "repo_commit": repo_commit,
        "remote_repo": manifest["remote_repo"],
        "remote_python": manifest["remote_python"],
        "gpu_count": host["gpu_count"],
        "accelerator": host["accelerator"],
        "python_version": runtime["python_version"],
        "torch_version": runtime["torch_version"],
        "torch_cuda_version": runtime["torch_cuda_version"],
        "rust_version": runtime["catanatron_rs_version"],
        "rust_extension_sha256": runtime["catanatron_rs_extension_sha256"],
        "required_capabilities": list(REQUIRED_NATIVE_CAPABILITIES),
        "required_entity_adapter": REQUIRED_LEARNER_ENTITY_ADAPTER,
        "checkpoint_path": manifest["checkpoint"]["path"],
        "checkpoint_sha256": manifest["checkpoint"]["sha256"],
        "wheel_path": manifest["native_wheel"]["path"],
        "wheel_sha256": manifest["native_wheel"]["sha256"],
        "required_nofile": REQUIRED_NOFILE,
        "resource_minima": manifest["resource_minima"],
        "seed_authority_path": manifest["seed_authority"]["path"],
        "seed_authority": authority,
    }
    token = base64.urlsafe_b64encode(_canonical(payload)).decode().rstrip("=")
    remote_command = shlex.join(["python3", "-c", REMOTE_INSPECT, token])
    if host["alias"] == manifest["coordinator_alias"]:
        command = ["python3", "-c", REMOTE_INSPECT, token]
    else:
        command = [*_ssh_base(manifest, host), remote_command]
    completed = _run(
        command,
        timeout=90,
    )
    if completed.returncode != 0:
        return {
            "alias": host["alias"],
            "address": host["address"],
            "reachable": False,
            "ready": False,
            "error": completed.stderr.strip() or completed.stdout.strip(),
        }
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError:
        report = {"ready": False, "error": completed.stdout}
    return {
        "alias": host["alias"],
        "address": host["address"],
        "reachable": True,
        **report,
    }


def _install_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = _file_sha256(source)
    if destination.exists():
        if destination.is_file() and _file_sha256(destination) == expected:
            return
        raise PreflightError(f"destination exists with different bytes: {destination}")
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    shutil.copyfile(source, temporary)
    os.chmod(temporary, 0o444)
    os.replace(temporary, destination)


def _stage_to_host(
    manifest: dict[str, Any], host: dict[str, Any], source: Path, remote: str
) -> None:
    expected = _file_sha256(source)
    if host["alias"] == manifest["coordinator_alias"]:
        if expected != _file_sha256(source):
            raise AssertionError("source changed while staging")
        _install_file(source, Path(remote))
        return
    probe = (
        f"test -f {shlex.quote(remote)} && "
        f"test \"$(sha256sum {shlex.quote(remote)} | cut -d' ' -f1)\" = {expected}"
    )
    if _run([*_ssh_base(manifest, host), probe], timeout=30).returncode == 0:
        return
    absent = _run(
        [*_ssh_base(manifest, host), f"test ! -e {shlex.quote(remote)}"],
        timeout=30,
    )
    if absent.returncode != 0:
        raise PreflightError(
            f"{host['alias']} has stale/different bytes at {remote}; refusing overwrite"
        )
    parent = str(Path(remote).parent)
    temporary = remote + f".tmp.{os.getpid()}"
    target = f"{manifest['ssh_user']}@{host['address']}:{temporary}"
    _run([*_ssh_base(manifest, host), "mkdir", "-p", parent], timeout=30)
    scp = [
        "scp", "-i", manifest["ssh_key"], "-o", "BatchMode=yes", "-o",
        f"StrictHostKeyChecking={manifest['strict_host_key_checking']}",
        str(source), target,
    ]
    copied = _run(scp, timeout=300)
    if copied.returncode != 0:
        raise PreflightError(f"stage to {host['alias']} failed: {copied.stderr}")
    finalize = (
        f"test \"$(sha256sum {shlex.quote(temporary)} | cut -d' ' -f1)\" = {expected} "
        f"&& chmod 0444 {shlex.quote(temporary)} && mv {shlex.quote(temporary)} {shlex.quote(remote)}"
    )
    done = _run([*_ssh_base(manifest, host), finalize], timeout=60)
    if done.returncode != 0:
        raise PreflightError(f"stage finalize on {host['alias']} failed: {done.stderr}")


def _assert_idle_host(manifest: dict[str, Any], host: dict[str, Any]) -> None:
    """Refuse all bootstrap mutation while a training/eval/generation job exists."""

    program = r'''from __future__ import annotations
import os, pathlib, subprocess

markers = (
    "/tools/generate.py", "generate_gumbel_selfplay_data.py",
    "/tools/train.py", "/tools/train_bc.py", "torch.distributed.run",
    "/tools/evaluate.py", "gumbel_search_vs_bot_h2h.py",
    "catanatron_neutral_harness_match.py",
)
active = []
ignored = set()
cursor = os.getpid()
while cursor > 1 and cursor not in ignored:
    ignored.add(cursor)
    try: cursor = int(pathlib.Path(f"/proc/{cursor}/stat").read_text().split()[3])
    except (OSError, ValueError, IndexError): break
for path in pathlib.Path("/proc").glob("[0-9]*/cmdline"):
    if int(path.parent.name) in ignored: continue
    try: command = path.read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except (OSError, PermissionError): continue
    if any(marker in command for marker in markers): active.append(command)
gpu = subprocess.run(
    ["nvidia-smi", "--query-compute-apps=process_name", "--format=csv,noheader"],
    text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
)
gpu_apps = [
    line.strip() for line in gpu.stdout.splitlines()
    if line.strip() and "nvidia-cuda-mps-server" not in line
]
if active or gpu_apps:
    print({"workloads": active[:8], "gpu_apps": gpu_apps[:16]})
    raise SystemExit(73)
'''
    if host["alias"] == manifest["coordinator_alias"]:
        command = ["python3", "-c", program]
    else:
        command = [*_ssh_base(manifest, host), shlex.join(["python3", "-c", program])]
    completed = _run(command, timeout=45)
    if completed.returncode != 0:
        detail = completed.stdout.strip() or completed.stderr.strip()
        raise PreflightError(
            f"{host['alias']} is not idle; bootstrap refused before mutation: {detail}"
        )


def _retire_mps_host(manifest: dict[str, Any], host: dict[str, Any]) -> None:
    """Disable the retired multi-context runtime after workload refusal."""

    command = r'''set -euo pipefail
sudo -n systemctl disable --now nvidia-mps.service >/dev/null 2>&1 || true
sudo -n pkill -TERM -f '^/usr/bin/nvidia-cuda-mps-(control|server)' >/dev/null 2>&1 || true
for _ in $(seq 1 40); do
  pgrep -f '^/usr/bin/nvidia-cuda-mps-(control|server)' >/dev/null || break
  sleep 0.1
done
if pgrep -f '^/usr/bin/nvidia-cuda-mps-(control|server)' >/dev/null; then
  echo 'retired MPS process remains active' >&2
  exit 74
fi
sudo -n rm -rf /tmp/mps_pipe_host /tmp/mps_log_host
if systemctl is-active --quiet nvidia-mps.service; then
  echo 'retired nvidia-mps.service remains active' >&2
  exit 75
fi
if systemctl is-enabled --quiet nvidia-mps.service; then
  echo 'retired nvidia-mps.service remains enabled' >&2
  exit 76
fi
'''
    if host["alias"] == manifest["coordinator_alias"]:
        argv = ["bash", "-lc", command]
    else:
        argv = [*_ssh_base(manifest, host), shlex.join(["bash", "-lc", command])]
    result = _run(argv, timeout=60)
    if result.returncode != 0:
        raise PreflightError(
            f"cannot retire MPS on {host['alias']}: {result.stderr or result.stdout}"
        )


def _bootstrap_host(
    manifest: dict[str, Any],
    *,
    repo_commit: str,
    host: dict[str, Any],
    runtime: dict[str, Any],
    checkpoint_source: Path,
    wheel_source: Path,
) -> None:
    _assert_idle_host(manifest, host)
    _retire_mps_host(manifest, host)
    _stage_to_host(manifest, host, checkpoint_source, manifest["checkpoint"]["path"])
    _stage_to_host(manifest, host, wheel_source, manifest["native_wheel"]["path"])
    authority = _seed_authority(manifest, repo_commit=repo_commit, host=host)
    authority_token = base64.urlsafe_b64encode(_canonical(authority)).decode().rstrip("=")
    command = f'''set -euo pipefail
repo={shlex.quote(manifest["remote_repo"])}
python={shlex.quote(manifest["remote_python"])}
wheel={shlex.quote(manifest["native_wheel"]["path"])}
reused=0
if [ -x {shlex.quote(manifest["remote_python"])} ]; then
  if [ "$(git -C "$repo" rev-parse HEAD 2>/dev/null || true)" != {repo_commit} ]; then
    echo "immutable deployment path contains a different commit; select a fresh remote_repo path" >&2
    exit 44
  fi
  reused=1
else
  test ! -e "$repo" || test -z "$(find "$repo" -mindepth 1 -maxdepth 1 -print -quit)"
  temp=$(mktemp -d /tmp/catan-bootstrap.XXXXXXXX)
  trap 'rm -rf "$temp"' EXIT
  git clone --quiet --filter=blob:none --no-checkout {shlex.quote(manifest["git_url"])} "$temp/src"
  git -C "$temp/src" fetch --quiet --depth=1 origin {repo_commit}
  git -C "$temp/src" checkout --quiet --detach {repo_commit}
  CATAN_REPO={shlex.quote(manifest["git_url"])} CATAN_REF={repo_commit} \
    CATAN_DEST="$repo" CATAN_RS_WHEEL={shlex.quote(manifest["native_wheel"]["path"])} \
    bash "$temp/src/tools/install_v1_freeze.sh"
fi
test "$(git -C "$repo" rev-parse HEAD)" = {repo_commit}
test "$(sha256sum "$wheel" | cut -d' ' -f1)" = {shlex.quote(manifest["native_wheel"]["sha256"])}
if [ "$reused" = 1 ]; then
  "$python" -m pip install --force-reinstall --no-deps "$wheel"
fi
"$python" - \
  {shlex.quote(runtime["catanatron_rs_version"])} \
  {shlex.quote(manifest["native_wheel"]["sha256"])} \
  {shlex.quote(runtime["catanatron_rs_extension_sha256"])} \
  {shlex.quote(REQUIRED_LEARNER_ENTITY_ADAPTER)} <<'PY'
import hashlib, importlib, json, pathlib, sys
from importlib.metadata import distribution

expected_version, wheel_sha256, extension_sha256, required_adapter = sys.argv[1:]
metadata = distribution("catanatron-rs")
if metadata.version != expected_version:
    raise SystemExit(f"catanatron-rs version drift: expected={{expected_version}} got={{metadata.version}}")
raw = metadata.read_text("direct_url.json")
try:
    direct_url = json.loads(raw) if raw is not None else None
except json.JSONDecodeError as error:
    raise SystemExit(f"installed catanatron-rs direct_url.json is invalid: {{error}}") from error
archive = direct_url.get("archive_info") if isinstance(direct_url, dict) else None
stated = set()
if isinstance(archive, dict):
    if isinstance(archive.get("hash"), str):
        stated.add(archive["hash"])
    hashes = archive.get("hashes")
    if isinstance(hashes, dict) and isinstance(hashes.get("sha256"), str):
        stated.add("sha256=" + hashes["sha256"])
expected_wheel = "sha256=" + wheel_sha256
if stated != {{expected_wheel}}:
    raise SystemExit(f"installed wheel provenance drift: expected={{expected_wheel}} got={{sorted(stated)}}")
native = importlib.import_module("catanatron_rs.catanatron_rs")
extension = pathlib.Path(native.__file__).resolve(strict=True)
digest = hashlib.sha256()
with extension.open("rb") as handle:
    for block in iter(lambda: handle.read(1 << 20), b""):
        digest.update(block)
observed_extension = digest.hexdigest()
if observed_extension != extension_sha256:
    raise SystemExit(f"loaded extension drift: expected={{extension_sha256}} got={{observed_extension}} path={{extension}}")
package = importlib.import_module("catanatron_rs")
adapter_fn = getattr(package, "supported_action_context_adapter_versions", None)
adapters = set(map(str, adapter_fn())) if callable(adapter_fn) else set()
if required_adapter not in adapters:
    raise SystemExit(f"native runtime lacks required V6 action-context adapter: {{required_adapter}}; supported={{sorted(adapters)}}")
PY
mkdir -p {shlex.quote(authority["output_root"])} {shlex.quote(str(Path(manifest["seed_authority"]["path"]).parent))}
python3 - {shlex.quote(manifest["seed_authority"]["path"])} {authority_token} <<'PY'
import base64, json, os, pathlib, sys
path=pathlib.Path(sys.argv[1]); token=sys.argv[2] + "=" * (-len(sys.argv[2]) % 4)
expected=json.loads(base64.urlsafe_b64decode(token).decode())
if path.exists():
    actual=json.loads(path.read_text())
    if actual != expected: raise SystemExit("seed authority exists with different bytes")
else:
    tmp=path.with_name(path.name+f".tmp.{{os.getpid()}}")
    with tmp.open("x",encoding="utf-8") as f:
        json.dump(expected,f,indent=2,sort_keys=True); f.write("\\n"); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)
PY
'''
    if host["alias"] == manifest["coordinator_alias"]:
        argv = ["bash", "-lc", command]
    else:
        argv = [*_ssh_base(manifest, host), shlex.join(["bash", "-lc", command])]
    result = _run(argv, timeout=1800)
    if result.returncode != 0:
        raise PreflightError(
            f"bootstrap failed on {host['alias']}: {result.stderr or result.stdout}"
        )


def _validate_sources(
    manifest: dict[str, Any], checkpoint_source: Path, wheel_source: Path
) -> None:
    for source, record, label in (
        (checkpoint_source, manifest["checkpoint"], "checkpoint"),
        (wheel_source, manifest["native_wheel"], "native wheel"),
    ):
        if not source.is_file() or source.is_symlink():
            raise PreflightError(f"{label} source is not a regular file: {source}")
        observed = _file_sha256(source)
        if observed != record["sha256"]:
            raise PreflightError(
                f"{label} source SHA mismatch: expected {record['sha256']} got {observed}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo-commit", required=True)
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument(
        "--host",
        action="append",
        default=[],
        help="Bootstrap and inspect only this alias (repeatable); omit for all six.",
    )
    parser.add_argument("--checkpoint-source", type=Path)
    parser.add_argument("--wheel-source", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if not COMMIT_RE.fullmatch(args.repo_commit):
        raise SystemExit("--repo-commit must be a full lowercase Git commit")
    manifest = load_manifest(args.manifest)
    coordinator = next(
        host for host in manifest["hosts"]
        if host["alias"] == manifest["coordinator_alias"]
    )
    expected_hostname = coordinator["address"].replace(".", "-")
    if socket.gethostname() != expected_hostname:
        raise SystemExit(
            "run this command on the configured coordinator: "
            f"expected hostname {expected_hostname!r}, got {socket.gethostname()!r}"
        )
    ssh_key = Path(manifest["ssh_key"])
    if not ssh_key.is_file() or ssh_key.is_symlink():
        raise SystemExit(f"coordinator fleet SSH key is missing/unsafe: {ssh_key}")
    runtime = _runtime_contract_for_commit(manifest, args.repo_commit)
    if runtime["catanatron_rs_wheel_sha256"] != manifest["native_wheel"]["sha256"]:
        raise SystemExit("manifest native wheel differs from runtime contract")
    if not SHA_RE.fullmatch(str(runtime.get("catanatron_rs_extension_sha256", ""))):
        raise SystemExit("runtime contract has no canonical native extension SHA-256")
    selected_aliases = set(args.host) or {host["alias"] for host in manifest["hosts"]}
    known_aliases = {host["alias"] for host in manifest["hosts"]}
    unknown_aliases = sorted(selected_aliases - known_aliases)
    if unknown_aliases:
        raise SystemExit(f"unknown --host aliases: {unknown_aliases}")
    selected_hosts = [
        host for host in manifest["hosts"] if host["alias"] in selected_aliases
    ]
    if args.bootstrap:
        if args.checkpoint_source is None or args.wheel_source is None:
            raise SystemExit("--bootstrap requires --checkpoint-source and --wheel-source")
        checkpoint_source = args.checkpoint_source.expanduser().resolve(strict=True)
        wheel_source = args.wheel_source.expanduser().resolve(strict=True)
        _validate_sources(manifest, checkpoint_source, wheel_source)
        bootstrap_errors: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=min(6, len(selected_hosts))) as pool:
            futures = {
                pool.submit(
                    _bootstrap_host,
                    manifest,
                    repo_commit=args.repo_commit,
                    host=host,
                    runtime=runtime,
                    checkpoint_source=checkpoint_source,
                    wheel_source=wheel_source,
                ): host["alias"]
                for host in selected_hosts
            }
            for future in as_completed(futures):
                alias = futures[future]
                try:
                    future.result()
                except Exception as error:
                    bootstrap_errors[alias] = f"{type(error).__name__}: {error}"
        if bootstrap_errors:
            ordered = {alias: bootstrap_errors[alias] for alias in sorted(bootstrap_errors)}
            raise SystemExit(
                "fleet bootstrap failed before inspection: "
                + json.dumps(ordered, sort_keys=True)
            )
    reports: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {
            pool.submit(
                _inspect_host,
                manifest,
                repo_commit=args.repo_commit,
                host=host,
                runtime=runtime,
            ): host
            for host in selected_hosts
        }
        for future in as_completed(futures):
            reports.append(future.result())
    reports.sort(key=lambda item: item["alias"])
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "controller_hostname": socket.gethostname(),
        "direct_coordinator_transport": True,
        "manifest_sha256": manifest["manifest_sha256"],
        "repo_commit": args.repo_commit,
        "checkpoint_sha256": manifest["checkpoint"]["sha256"],
        "native_wheel_sha256": manifest["native_wheel"]["sha256"],
        "bootstrap_requested": bool(args.bootstrap),
        "bootstrap_hosts": sorted(set(args.host)) if args.host else (
            [host["alias"] for host in manifest["hosts"]] if args.bootstrap else []
        ),
        "full_generation_started": False,
        "ready_hosts": sum(bool(report.get("ready")) for report in reports),
        "host_count": len(reports),
        "fleet_host_count": len(manifest["hosts"]),
        "hosts": reports,
    }
    receipt["selection_ready"] = receipt["ready_hosts"] == receipt["host_count"]
    receipt["fleet_ready"] = (
        receipt["host_count"] == receipt["fleet_host_count"]
        and receipt["selection_ready"]
    )
    receipt["receipt_sha256"] = _digest(receipt)
    rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.receipt is not None:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    raise SystemExit(0 if receipt["selection_ready"] else 2)


if __name__ == "__main__":
    main()
