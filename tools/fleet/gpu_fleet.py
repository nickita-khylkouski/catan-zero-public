#!/usr/bin/env python3
"""Daemonless, fail-closed scheduler for the canonical 56-H100 fleet."""

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
import subprocess
import sys
from typing import Any, Callable, Iterable, Sequence


MANIFEST_SCHEMA = "catan-gpu-fleet-v1"
JOBSET_SCHEMA = "catan-gpu-jobset-v1"
PLAN_SCHEMA = "catan-gpu-plan-v1"
RECEIPT_SCHEMA = "catan-gpu-job-receipt-v1"
EXPECTED_HOSTS = {
    "c1": ("192.222.54.251", 4),
    "c2": ("68.209.75.117", 4),
    "c3": ("192.222.53.18", 4),
    "c4": ("68.209.73.252", 4),
    "c5": ("68.209.74.145", 4),
    "c6": ("68.209.74.2", 4),
    "h100-8a": ("192.222.53.119", 8),
    "h100-8b": ("192.222.55.216", 8),
    "h100-8c": ("192.222.54.141", 8),
    "h100-8d": ("209.20.158.82", 8),
}
EXPECTED_SHAPES = {alias: shape[1] for alias, shape in EXPECTED_HOSTS.items()}
EXPECTED_ACCELERATOR = "NVIDIA H100 80GB HBM3"
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SAFE_ADDRESS = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.:-]*$")
SAFE_ENV = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class FleetError(RuntimeError):
    """A fleet action could not be proved safe."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FleetError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise FleetError(f"{path} must contain one JSON object")
    return value


def load_manifest(path: Path) -> dict[str, Any]:
    value = _read(path)
    if value.get("schema_version") != MANIFEST_SCHEMA:
        raise FleetError("unsupported fleet manifest schema")
    for field in ("ssh_user", "remote_repo", "remote_root", "hosts"):
        if field not in value:
            raise FleetError(f"fleet manifest is missing {field}")
    if not SAFE_NAME.fullmatch(str(value["ssh_user"])):
        raise FleetError("unsafe ssh_user")
    checking = str(value.get("strict_host_key_checking", "accept-new"))
    if checking not in {"yes", "accept-new"}:
        raise FleetError("strict_host_key_checking must be yes or accept-new")
    value["strict_host_key_checking"] = checking
    for field in ("remote_repo", "remote_root"):
        item = Path(str(value[field]))
        if not item.is_absolute() or any(char in str(item) for char in "\n\r\0"):
            raise FleetError(f"{field} must be a safe absolute path")
    key = value.get("ssh_key")
    value["ssh_key"] = str(Path(str(key)).expanduser()) if key else None
    if not isinstance(value["hosts"], list):
        raise FleetError("hosts must be a list")
    hosts: list[dict[str, Any]] = []
    seen: set[str] = set()
    seen_addresses: set[str] = set()
    for raw in value["hosts"]:
        if not isinstance(raw, dict):
            raise FleetError("each host must be an object")
        alias, address = str(raw.get("alias", "")), str(raw.get("address", ""))
        if not SAFE_NAME.fullmatch(alias) or alias in seen:
            raise FleetError(f"invalid or duplicate host alias {alias!r}")
        if not SAFE_ADDRESS.fullmatch(address):
            raise FleetError(f"invalid address for {alias}")
        if address in seen_addresses:
            raise FleetError(f"duplicate host address {address}")
        try:
            gpu_count = int(raw["gpu_count"])
        except (KeyError, TypeError, ValueError) as error:
            raise FleetError(f"invalid gpu_count for {alias}") from error
        accelerator = str(raw.get("accelerator", ""))
        repo_commit = str(raw.get("repo_commit", ""))
        if accelerator != EXPECTED_ACCELERATOR:
            raise FleetError(
                f"accelerator for {alias} must be exactly {EXPECTED_ACCELERATOR!r}"
            )
        if not re.fullmatch(r"[0-9a-f]{40}", repo_commit):
            raise FleetError(f"repo_commit for {alias} must be a full Git commit")
        hosts.append(
            {
                "alias": alias,
                "address": address,
                "gpu_count": gpu_count,
                "accelerator": accelerator,
                "repo_commit": repo_commit,
            }
        )
        seen.add(alias)
        seen_addresses.add(address)
    actual = {host["alias"]: (host["address"], host["gpu_count"]) for host in hosts}
    if actual != EXPECTED_HOSTS:
        raise FleetError(
            "canonical fleet alias/address/GPU mapping drift: "
            f"expected {EXPECTED_HOSTS}, got {actual}"
        )
    # Manifest order is allocation policy; do not sort it.
    value["hosts"] = hosts
    value["manifest_hash"] = _digest(
        {key: item for key, item in value.items() if key != "manifest_hash"}
    )
    return value


def _normalize_jobset(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("schema_version") != JOBSET_SCHEMA:
        raise FleetError("unsupported jobset schema")
    run_id = str(value.get("run_id", ""))
    if not SAFE_NAME.fullmatch(run_id):
        raise FleetError("invalid run_id")
    jobs = value.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise FleetError("jobs must be a nonempty list")
    normalized, seen = [], set()
    for raw in jobs:
        if not isinstance(raw, dict):
            raise FleetError("each job must be an object")
        job_id = str(raw.get("job_id", ""))
        if not SAFE_NAME.fullmatch(job_id) or job_id in seen:
            raise FleetError(f"invalid or duplicate job_id {job_id!r}")
        try:
            gpus = int(raw.get("gpus", 1))
        except (TypeError, ValueError) as error:
            raise FleetError(f"invalid gpus for {job_id}") from error
        if not 1 <= gpus <= 8:
            raise FleetError(f"gpus out of range for {job_id}")
        argv = raw.get("argv")
        if (
            not isinstance(argv, list)
            or not argv
            or any(not isinstance(part, str) or "\0" in part for part in argv)
        ):
            raise FleetError(f"argv for {job_id} must be a nonempty string list")
        env = raw.get("env", {})
        if not isinstance(env, dict) or any(
            not SAFE_ENV.fullmatch(str(key))
            or not isinstance(item, str)
            or "\0" in item
            for key, item in env.items()
        ):
            raise FleetError(f"invalid env for {job_id}")
        preferred = raw.get("host")
        if preferred is not None and not SAFE_NAME.fullmatch(str(preferred)):
            raise FleetError(f"invalid host preference for {job_id}")
        normalized.append(
            {
                "job_id": job_id,
                "gpus": gpus,
                "argv": argv,
                "env": env,
                "host": preferred,
            }
        )
        seen.add(job_id)
    return {"schema_version": JOBSET_SCHEMA, "run_id": run_id, "jobs": normalized}


def load_jobset(path: Path) -> dict[str, Any]:
    return _normalize_jobset(_read(path))


def build_plan(
    manifest: dict[str, Any], jobset: dict[str, Any], *, repo_commit: str
) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{40}", repo_commit):
        raise FleetError("repo_commit must be a full Git commit")
    by_alias = {host["alias"]: host for host in manifest["hosts"]}
    free = {alias: list(range(host["gpu_count"])) for alias, host in by_alias.items()}
    assignments = []
    for job in jobset["jobs"]:
        if job["host"] is not None:
            choices = [str(job["host"])]
        else:
            choices = [
                host["alias"]
                for host in manifest["hosts"]
                if host["repo_commit"] == repo_commit
            ]
        selected = None
        for alias in choices:
            if alias not in by_alias:
                raise FleetError(f"job {job['job_id']} requests unknown host {alias}")
            if by_alias[alias]["repo_commit"] != repo_commit:
                raise FleetError(
                    f"job {job['job_id']} requests {alias} at "
                    f"{by_alias[alias]['repo_commit']}, not {repo_commit}"
                )
            if len(free[alias]) >= job["gpus"]:
                selected = alias
                break
        if selected is None:
            raise FleetError(f"insufficient same-host capacity for {job['job_id']}")
        gpu_ids = free[selected][: job["gpus"]]
        del free[selected][: job["gpus"]]
        host = by_alias[selected]
        assignments.append(
            {
                **job,
                "alias": selected,
                "address": host["address"],
                "host_gpu_count": host["gpu_count"],
                "gpu_ids": gpu_ids,
                "job_dir": (
                    f"{manifest['remote_root'].rstrip('/')}/{jobset['run_id']}/"
                    f"{job['job_id']}"
                ),
            }
        )
    plan: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA,
        "run_id": jobset["run_id"],
        "manifest_hash": manifest["manifest_hash"],
        "repo_commit": repo_commit,
        "assignments": assignments,
    }
    plan["plan_hash"] = _digest(plan)
    return plan


def load_plan(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    plan = _read(path)
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise FleetError("unsupported plan schema")
    replay = _digest({key: item for key, item in plan.items() if key != "plan_hash"})
    if plan.get("plan_hash") != replay:
        raise FleetError("plan hash does not replay")
    if plan.get("manifest_hash") != manifest["manifest_hash"]:
        raise FleetError("plan belongs to a different fleet manifest")
    embedded = _normalize_jobset(
        {
            "schema_version": JOBSET_SCHEMA,
            "run_id": plan.get("run_id"),
            "jobs": [
                {key: row[key] for key in ("job_id", "gpus", "argv", "env", "host")}
                for row in plan.get("assignments", [])
            ],
        }
    )
    expected = build_plan(manifest, embedded, repo_commit=str(plan.get("repo_commit")))
    if expected != plan:
        raise FleetError("plan allocation or command drift")
    return plan


def write_new(path: Path, value: dict[str, Any]) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _ssh(manifest: dict[str, Any], host: dict[str, Any]) -> list[str]:
    command = ["ssh"]
    if manifest["ssh_key"]:
        command += ["-i", manifest["ssh_key"], "-o", "IdentitiesOnly=yes"]
    command += [
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        f"StrictHostKeyChecking={manifest['strict_host_key_checking']}",
        "-o",
        "ControlMaster=auto",
        "-o",
        "ControlPersist=60",
        "-o",
        "ControlPath=~/.ssh/catan-cm-%C",
        f"{manifest['ssh_user']}@{host['address']}",
    ]
    return command


def _run(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
    )


def _parallel(rows: Iterable[Any], function: Callable[[Any], Any]) -> list[Any]:
    values = list(rows)
    results = []
    with ThreadPoolExecutor(max_workers=min(16, len(values) or 1)) as pool:
        futures = {pool.submit(function, value): value for value in values}
        for future in as_completed(futures):
            results.append(future.result())
    return results


def inventory(
    manifest: dict[str, Any],
    *,
    runner: Callable[..., subprocess.CompletedProcess] = _run,
) -> dict[str, Any]:
    probe = """set -euo pipefail
printf 'hostname='; hostname
printf 'gpu_count='; nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l
printf 'gpu_names='; nvidia-smi --query-gpu=name --format=csv,noheader | sort -u | paste -sd, -
printf 'busy_gpus='; nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 128 {n++} END {print n+0}'
printf 'repo_commit='; git -C "$HOME_REPO" rev-parse HEAD 2>/dev/null || echo missing
"""

    def inspect(host: dict[str, Any]) -> dict[str, Any]:
        command = f"HOME_REPO={shlex.quote(manifest['remote_repo'])}\n{probe}"
        result = runner([*_ssh(manifest, host), command])
        fields = dict(
            line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
        )
        count = int(fields.get("gpu_count", "-1"))
        busy = int(fields.get("busy_gpus", "-1"))
        topology_ok = (
            count == host["gpu_count"]
            and fields.get("gpu_names") == EXPECTED_ACCELERATOR
        )
        commit_ok = fields.get("repo_commit") == host["repo_commit"]
        return {
            "alias": host["alias"],
            "address": host["address"],
            **fields,
            "expected_gpus": host["gpu_count"],
            "expected_repo_commit": host["repo_commit"],
            "valid": topology_ok and commit_ok,
            "ready": topology_ok and commit_ok and busy == 0,
        }

    hosts = sorted(_parallel(manifest["hosts"], inspect), key=lambda row: row["alias"])
    return {
        "manifest_hash": manifest["manifest_hash"],
        "valid": all(host["valid"] for host in hosts),
        "gpu_capacity": sum(host["expected_gpus"] for host in hosts),
        "hosts": hosts,
    }


def _host_preflight(
    manifest: dict[str, Any], plan: dict[str, Any], host: dict[str, Any]
) -> str:
    return f"""set -euo pipefail
test "$(git -C {shlex.quote(manifest["remote_repo"])} rev-parse HEAD)" = {shlex.quote(plan["repo_commit"])}
test "$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l)" -eq {host["gpu_count"]}
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -Fxc {shlex.quote(EXPECTED_ACCELERATOR)})" -eq {host["gpu_count"]}
"""


def _gpu_idle_check(row: dict[str, Any]) -> str:
    gpu_ids = " ".join(str(gpu) for gpu in row["gpu_ids"])
    return f"""for gpu in {gpu_ids}; do
  test "$gpu" -ge 0 -a "$gpu" -lt {row["host_gpu_count"]}
  uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader,nounits)
  used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
  test "$used" -le 128
  ! nvidia-smi --query-compute-apps=gpu_uuid,process_name --format=csv,noheader 2>/dev/null | awk -F, -v u="$uuid" '$1 == u && $2 !~ /nvidia-cuda-mps-server/ {{found=1}} END {{exit !found}}'
done
"""


def _runtime(
    manifest: dict[str, Any], plan: dict[str, Any], row: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    """Return the immutable receipt and exact detached bash command body."""
    job_dir = row["job_dir"]
    environment = {
        **row["env"],
        "CUDA_VISIBLE_DEVICES": ",".join(map(str, row["gpu_ids"])),
        "PYTHONUNBUFFERED": "1",
    }
    argv = [
        "env",
        *[f"{key}={item}" for key, item in sorted(environment.items())],
        *row["argv"],
    ]
    inner = (
        f"cd {shlex.quote(manifest['remote_repo'])}; set +e; "
        + " ".join(shlex.quote(part) for part in argv)
        + f"; rc=$?; printf '%s\\n' \"$rc\" > {shlex.quote(job_dir + '/.rc.tmp')}; "
        + f"mv -f {shlex.quote(job_dir + '/.rc.tmp')} {shlex.quote(job_dir + '/.rc')}; "
        + f'if [ "$rc" -eq 0 ]; then touch {shlex.quote(job_dir + "/.done")}; '
        + f'else touch {shlex.quote(job_dir + "/.failed")}; fi; exit "$rc"'
    )
    cmdline = b"bash\0-lc\0" + inner.encode() + b"\0"
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "run_id": plan["run_id"],
        "plan_hash": plan["plan_hash"],
        "assignment_hash": _digest(row),
        "job_id": row["job_id"],
        "alias": row["alias"],
        "gpu_ids": row["gpu_ids"],
        "repo_commit": plan["repo_commit"],
        "argv": row["argv"],
        "env": row["env"],
        "cmdline_sha256": "sha256:" + hashlib.sha256(cmdline).hexdigest(),
        "heartbeat_cadence_seconds": 5,
    }
    return receipt, inner


def _validated_live_function(job_dir: str, expected_cmdline_sha: str) -> str:
    """Shell function matching launch_detached's PID/SID/PGID contract."""
    return f"""job_pid_valid() {{
  pid=$(cat {shlex.quote(job_dir + "/.pid")} 2>/dev/null || true)
  case "$pid" in ''|*[!0-9]*) return 1;; esac
  [ "$pid" -gt 1 ] || return 1
  read -r sid pgid state < <(ps -o sid=,pgid=,stat= -p "$pid" 2>/dev/null) || return 1
  [ "$sid" = "$pid" ] && [ "$pgid" = "$pid" ] || return 1
  case "$state" in Z*) return 1;; esac
  [ -r "/proc/$pid/cmdline" ] || return 1
  actual="sha256:$(sha256sum "/proc/$pid/cmdline" | cut -d' ' -f1)"
  [ "$actual" = {shlex.quote(expected_cmdline_sha)} ] || return 1
  hb={shlex.quote(job_dir + "/.heartbeat")}
  [ -f "$hb" ] || return 1
  grep -Fq "pid=$pid" "$hb" || return 1
  ! grep -Fq EXITED "$hb" || return 1
  age=$(( $(date +%s) - $(stat -c %Y "$hb") ))
  [ "$age" -le 15 ] || return 1
}}
"""


def _launch_command(
    manifest: dict[str, Any], plan: dict[str, Any], row: dict[str, Any]
) -> str:
    receipt, inner = _runtime(manifest, plan, row)
    encoded = base64.b64encode(_canonical(receipt)).decode()
    job_dir = row["job_dir"]
    receipt_path = shlex.quote(job_dir + "/receipt.json")
    launch = " ".join(
        [
            shlex.quote(
                manifest["remote_repo"].rstrip("/") + "/tools/fleet/launch_detached.sh"
            ),
            shlex.quote(job_dir),
            shlex.quote(job_dir + "/run.log"),
            "5 --",
            "bash -lc",
            shlex.quote(inner),
        ]
    )
    body = "\n".join(
        [
            "set -euo pipefail",
            'lock_root="/run/user/$(id -u)/catan-zero-gpu-fleet"',
            'install -d -m 0700 "$lock_root"',
            # The outer read is useful for fast failure, but it is not the
            # transactional authority: deployment/topology can change while
            # this process waits for another allocator. Re-read both only
            # after this job owns the per-host allocation lock.
            _host_preflight(
                manifest,
                plan,
                {
                    "gpu_count": row["host_gpu_count"],
                    "accelerator": EXPECTED_ACCELERATOR,
                },
            ),
            f"mkdir -p {shlex.quote(job_dir)}",
            _validated_live_function(job_dir, receipt["cmdline_sha256"]),
            f"if [ -e {receipt_path} ] || [ -L {receipt_path} ]; then",
            f"  test -f {receipt_path} && test ! -L {receipt_path} || exit 70",
            f'  test "$(stat -c %u {receipt_path})" = "$(id -u)" || exit 70',
            f'  test "$(stat -c %a {receipt_path})" = 444 || exit 70',
            f'  test "$(base64 -w0 {receipt_path})" = {shlex.quote(encoded)} || '
            "{ echo receipt-drift >&2; exit 71; }",
            f"  if [ -f {shlex.quote(job_dir + '/.done')} ]; then",
            f"    test ! -e {shlex.quote(job_dir + '/.failed')} || exit 72",
            f'    test "$(cat {shlex.quote(job_dir + "/.rc")})" = 0 || exit 72',
            f"    echo {shlex.quote(row['job_id'] + ':done')}",
            f"  elif [ -f {shlex.quote(job_dir + '/.failed')} ]; then",
            f"    test -f {shlex.quote(job_dir + '/.rc')} || exit 72",
            f'    test "$(cat {shlex.quote(job_dir + "/.rc")})" != 0 || exit 72',
            f"    echo {shlex.quote(row['job_id'] + ':failed')}",
            f"  elif job_pid_valid; then echo {shlex.quote(row['job_id'] + ':active')}",
            "  else echo stale-job-identity >&2; exit 73; fi",
            "else",
            f"  test ! -L {receipt_path} || {{ echo receipt-symlink >&2; exit 70; }}",
            f"  for stale in {shlex.quote(job_dir + '/.pid')} "
            f"{shlex.quote(job_dir + '/.done')} {shlex.quote(job_dir + '/.failed')} "
            f"{shlex.quote(job_dir + '/.rc')} {shlex.quote(job_dir + '/.heartbeat')}; "
            'do [ ! -e "$stale" ] || { echo preexisting-marker >&2; exit 74; }; done',
            "  lease_fds=()",
            *[
                f'  exec {{lease_fd}}>"$lock_root/gpu-{gpu}.lock"; '
                'flock --exclusive --nonblock "$lease_fd" '
                "|| { echo gpu-lease-busy >&2; exit 75; }; "
                'lease_fds+=("$lease_fd")'
                for gpu in row["gpu_ids"]
            ],
            # The selected-GPU locks are already held here. They remain open
            # across the idle check, immutable receipt creation, detached
            # launch, and inside both detached job and heartbeat descendants.
            _gpu_idle_check(row),
            f"  (set -o noclobber; printf %s {shlex.quote(encoded)} | base64 -d > "
            f"{receipt_path}) || {{ echo receipt-create-race >&2; exit 76; }}",
            f"  chmod 0444 {receipt_path}",
            f"  if ! {launch}; then touch {shlex.quote(job_dir + '/.failed')}; exit 77; fi",
            "fi",
        ]
    )
    # flock's parent holds the allocation lock; --close prevents inheritance.
    # Selected GPU lock FDs are intentionally inherited by the detached job
    # and heartbeat, spanning the transaction without an unlock/relock gap.
    return "\n".join(
        [
            'lock_root="/run/user/$(id -u)/catan-zero-gpu-fleet"',
            'install -d -m 0700 "$lock_root"',
            'flock --exclusive --close "$lock_root/allocation.lock" '
            f"bash -lc {shlex.quote(body)}",
        ]
    )


def submit(
    manifest: dict[str, Any],
    plan: dict[str, Any],
    *,
    go: bool,
    runner: Callable[..., subprocess.CompletedProcess] = _run,
) -> dict[str, Any]:
    hosts = {host["alias"]: host for host in manifest["hosts"]}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in plan["assignments"]:
        grouped.setdefault(row["alias"], []).append(row)
    commands = []
    for alias, rows in grouped.items():
        host = hosts[alias]
        command = (
            _host_preflight(manifest, plan, host)
            + "\n"
            + "\n".join(_launch_command(manifest, plan, row) for row in rows)
        )
        commands.append((host, command, rows))
    if not go:
        return {
            "dry_run": True,
            "plan_hash": plan["plan_hash"],
            "hosts": [
                {
                    "alias": host["alias"],
                    "jobs": [row["job_id"] for row in rows],
                    "gpu_ids": sorted({gpu for row in rows for gpu in row["gpu_ids"]}),
                }
                for host, _, rows in commands
            ],
        }

    def launch(item: tuple[Any, ...]) -> dict[str, Any]:
        host, command, rows = item
        result = runner([*_ssh(manifest, host), command])
        return {
            "alias": host["alias"],
            "jobs": [row["job_id"] for row in rows],
            "stdout": result.stdout.strip(),
        }

    return {
        "dry_run": False,
        "plan_hash": plan["plan_hash"],
        "launched": sorted(_parallel(commands, launch), key=lambda row: row["alias"]),
    }


def status(
    manifest: dict[str, Any],
    plan: dict[str, Any],
    *,
    runner: Callable[..., subprocess.CompletedProcess] = _run,
) -> dict[str, Any]:
    hosts = {host["alias"]: host for host in manifest["hosts"]}

    def inspect(row: dict[str, Any]) -> dict[str, Any]:
        receipt, _ = _runtime(manifest, plan, row)
        encoded = base64.b64encode(_canonical(receipt)).decode()
        directory = shlex.quote(row["job_dir"])
        command = "\n".join(
            [
                "set -euo pipefail",
                f"d={directory}",
                _validated_live_function(row["job_dir"], receipt["cmdline_sha256"]),
                'if [ ! -e "$d/receipt.json" ] && [ ! -L "$d/receipt.json" ]; then s=NOT_SUBMITTED;',
                'elif [ ! -f "$d/receipt.json" ] || [ -L "$d/receipt.json" ]; then s=DRIFT;',
                'elif [ "$(stat -c %u "$d/receipt.json")" != "$(id -u)" ] || '
                '[ "$(stat -c %a "$d/receipt.json")" != 444 ]; then s=DRIFT;',
                f'elif [ "$(base64 -w0 "$d/receipt.json")" != {shlex.quote(encoded)} ]; then s=DRIFT;',
                'elif [ -f "$d/.done" ] && [ -f "$d/.failed" ]; then s=DRIFT;',
                'elif [ -f "$d/.done" ]; then '
                'if [ "$(cat "$d/.rc" 2>/dev/null)" = 0 ]; then s=DONE; else s=DRIFT; fi;',
                'elif [ -f "$d/.failed" ]; then '
                'if [ "$(cat "$d/.rc" 2>/dev/null)" != 0 ]; then s=FAILED; else s=DRIFT; fi;',
                "elif job_pid_valid; then s=RUNNING;",
                'elif [ -e "$d/.pid" ]; then s=STALE_IDENTITY;',
                "else s=LOST; fi",
                "printf '%s' \"$s\"",
            ]
        )
        result = runner([*_ssh(manifest, hosts[row["alias"]]), command])
        return {
            "job_id": row["job_id"],
            "alias": row["alias"],
            "gpu_ids": row["gpu_ids"],
            "status": result.stdout.strip(),
        }

    jobs = sorted(
        _parallel(plan["assignments"], inspect), key=lambda row: row["job_id"]
    )
    return {"plan_hash": plan["plan_hash"], "jobs": jobs}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("inventory")
    plan = commands.add_parser("plan")
    plan.add_argument("--jobset", type=Path, required=True)
    plan.add_argument("--repo-commit", required=True)
    plan.add_argument("--out", type=Path, required=True)
    for name in ("submit", "status"):
        child = commands.add_parser(name)
        child.add_argument("--plan", type=Path, required=True)
        if name == "submit":
            child.add_argument("--go", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        if args.command == "inventory":
            result = inventory(manifest)
        elif args.command == "plan":
            result = build_plan(
                manifest, load_jobset(args.jobset), repo_commit=args.repo_commit
            )
            write_new(args.out, result)
            result = {"plan": str(args.out), "plan_hash": result["plan_hash"]}
        else:
            plan = load_plan(args.plan, manifest)
            if args.command == "submit":
                result = submit(manifest, plan, go=args.go)
            else:
                result = status(manifest, plan)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (FleetError, OSError, subprocess.SubprocessError) as error:
        print(f"gpu_fleet: ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
