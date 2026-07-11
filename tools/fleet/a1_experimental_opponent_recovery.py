#!/usr/bin/env python3
"""One-off, non-promotable recovery for failed r1 opponent-only generation.

This file is intentionally not imported or called by the production executor.
It consumes the immutable r1 locks/renders and failure receipts read-only,
selects only recent-history and hard-negative jobs, and relocates their outputs
into a fresh experimental namespace while retaining the original unused seed
ranges, claim labels, quotas, opponents, and search science.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Sequence

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools import a1_pre_wave_contract as contract  # noqa: E402


CONFIG_SCHEMA = "a1-r1-opponent-recovery-config-v1"
PLAN_SCHEMA = "a1-r1-opponent-recovery-plan-v1"
RECEIPT_SCHEMA = "a1-r1-opponent-recovery-lane-receipt-v1"
LABEL = "experimental_nonpromotable"
ALLOWED = ("recent_history", "hard_negative")


class RecoveryError(RuntimeError):
    """The one-off recovery cannot preserve its sealed source semantics."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _load(path: Path, *, where: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RecoveryError(f"cannot read {where}: {error}") from error
    if not isinstance(value, dict):
        raise RecoveryError(f"{where} must contain one JSON object")
    return value


def _regular(path: Path, *, where: str) -> Path:
    try:
        if path.expanduser().is_symlink():
            raise RecoveryError(f"{where} may not be a symlink")
        canonical = path.expanduser().resolve(strict=True)
        info = canonical.stat()
    except OSError as error:
        raise RecoveryError(f"cannot resolve {where}: {error}") from error
    if not stat.S_ISREG(info.st_mode):
        raise RecoveryError(f"{where} must be a regular file")
    return canonical


def _file_ref(path: Path, *, where: str) -> dict[str, str]:
    path = _regular(path, where=where)
    return {"path": str(path), "sha256": _sha256(path)}


def _atomic_exact(path: Path, value: dict[str, Any], *, mode: int = 0o444) -> None:
    path = Path(os.path.abspath(os.fspath(path.expanduser())))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.resolve(strict=True) != path.parent:
        raise RecoveryError("output parent must be canonical")
    payload = json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, mode)
    except FileExistsError:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise RecoveryError(f"existing immutable file differs: {path}")
        return
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, mode)


def _atomic_mutable(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _verify_semantic(
    value: dict[str, Any], field: str, expected: str, *, where: str
) -> None:
    unhashed = dict(value)
    declared = unhashed.pop(field, None)
    if declared != expected or declared != contract._digest_value(unhashed):  # noqa: SLF001
        raise RecoveryError(f"{where} semantic digest drift")


def _replace_option(argv: list[str], name: str, value: str) -> None:
    if argv.count(name) != 1:
        raise RecoveryError(f"source argv must contain exactly one {name}")
    argv[argv.index(name) + 1] = value


def _science_projection(argv: list[str]) -> list[str]:
    """Remove only paths and implementation switches from command identity."""
    result = list(argv)
    for name in ("--out-dir",):
        index = result.index(name)
        del result[index : index + 2]
    for flag in ("--no-rust-featurize", "--rust-featurize", "--native-mcts-hot-loop"):
        result = [item for item in result if item != flag]
    return result


def _failure_ref(path: Path) -> dict[str, Any]:
    path = _regular(path, where="failed receipt")
    value = _load(path, where="failed receipt")
    status = value.get("status")
    if status == "complete":
        raise RecoveryError(f"failed receipt is complete, not failed: {path}")
    if status not in {"failed", "stopped", "prepared", "refused", "blocked"}:
        raise RecoveryError(f"failed receipt has unsupported status {status!r}: {path}")
    return {**_file_ref(path, where="failed receipt"), "status": status}


def _verify_ledger_snapshot_record(record: dict[str, Any]) -> None:
    required = {
        "kind",
        "path",
        "sha256",
        "snapshot_text",
        "snapshot_size_bytes",
        "claims",
        "claims_sha256",
    }
    if set(record) != required or record.get("kind") != "seed_ledger_snapshot":
        raise RecoveryError("render seed ledger snapshot fields drift")
    snapshot = record.get("snapshot_text")
    if not isinstance(snapshot, str):
        raise RecoveryError("render seed ledger snapshot text drift")
    encoded = snapshot.encode("utf-8")
    if (
        int(record.get("snapshot_size_bytes", -1)) != len(encoded)
        or "sha256:" + hashlib.sha256(encoded).hexdigest() != record.get("sha256")
        or record.get("claims_sha256") != contract._digest_value(record.get("claims"))  # noqa: SLF001
    ):
        raise RecoveryError("render seed ledger snapshot digest drift")


def _verify_live_ledger_bytes(
    data: bytes, *, expected_rows: Sequence[str], snapshot_prefixes: Sequence[str]
) -> str:
    for snapshot in snapshot_prefixes:
        if not data.startswith(snapshot.encode("utf-8")):
            raise RecoveryError(
                "live seed ledger is not an append-only snapshot extension"
            )
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise RecoveryError("live seed ledger is not UTF-8") from error
    for row in expected_rows:
        count = lines.count(row)
        if count != 1:
            raise RecoveryError(
                f"live seed ledger claim must occur exactly once (found {count}): {row}"
            )
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _require_ledger_consensus(preflight: Sequence[dict[str, Any]]) -> str:
    values = [str(row.get("live_ledger_sha256", "")) for row in preflight]
    hashes = set(values)
    if (
        not values
        or any(not value.startswith("sha256:") or len(value) != 71 for value in values)
        or len(hashes) != 1
    ):
        raise RecoveryError(
            f"fleet live seed ledger SHA consensus failed: {sorted(hashes)}"
        )
    return next(iter(hashes))


def build_plan(
    *,
    config_path: Path,
    failed_receipts: Sequence[Path],
    out: Path,
) -> dict[str, Any]:
    config_path = _regular(config_path, where="recovery config")
    config = _load(config_path, where="recovery config")
    if config.get("schema_version") != CONFIG_SCHEMA or config.get("label") != LABEL:
        raise RecoveryError("recovery config schema/label drift")
    if tuple(config.get("allowed_categories", [])) != ALLOWED:
        raise RecoveryError("recovery categories must be exactly opponent-only")
    failures = [_failure_ref(path) for path in failed_receipts]
    historical_parent = config.get("historical_parent_receipt")
    if not failures and historical_parent != "historical_parent_receipt_unavailable":
        raise RecoveryError(
            "missing failed receipts require the explicit historical annotation"
        )
    placement_ref = config["placement"]
    placement_path = _regular(_ROOT / placement_ref["path"], where="placement")
    if _sha256(placement_path) != placement_ref["sha256"]:
        raise RecoveryError("placement hash drift")
    placement = _load(placement_path, where="placement")
    assignments = {
        item["logical_lane"]: (item["host_alias"], int(item["gpu"]))
        for item in placement["assignments"]
    }
    runtime_files = []
    for record in config["runtime_files"]:
        path = _regular(_ROOT / record["path"], where="runtime file")
        if _sha256(path) != record["sha256"]:
            raise RecoveryError(f"runtime file hash drift: {record['path']}")
        runtime_files.append({**record, "path": str(Path(record["path"]))})
    recovery_root = str(Path(config["recovery_root"]))
    runtime_source_repo = str(Path(config["runtime_repo"]))
    runtime_python = Path(config["runtime_python"])
    if not runtime_python.is_absolute():
        raise RecoveryError("runtime_python must be an absolute remote path")
    staged_runtime_repo = (
        f"{recovery_root}/runtime-{str(config['runtime_commit'])[:12]}"
    )
    lanes: list[dict[str, Any]] = []
    source_refs: dict[str, Any] = {}
    required_files: dict[str, dict[str, str]] = {}
    wheel_config = config["native_wheel"]
    wheel = _file_ref(Path(wheel_config["path"]), where="native wheel")
    if (
        wheel["sha256"] != wheel_config["sha256"]
        or Path(wheel["path"]).name != wheel_config["filename"]
    ):
        raise RecoveryError("native wheel identity drift")
    required_files[wheel["path"]] = wheel
    for arm in ("n128", "n256"):
        arm_config = config["arms"][arm]
        lock_path = _regular(Path(arm_config["lock"]), where=f"{arm} lock")
        render_path = _regular(Path(arm_config["render"]), where=f"{arm} render")
        if _sha256(lock_path) != arm_config["lock_file_sha256"]:
            raise RecoveryError(f"{arm} lock file hash drift")
        if _sha256(render_path) != arm_config["render_file_sha256"]:
            raise RecoveryError(f"{arm} render file hash drift")
        lock = _load(lock_path, where=f"{arm} lock")
        render = _load(render_path, where=f"{arm} render")
        _verify_semantic(
            lock, "contract_sha256", arm_config["lock_sha256"], where=f"{arm} lock"
        )
        _verify_semantic(
            render, "render_sha256", arm_config["render_sha256"], where=f"{arm} render"
        )
        for checkpoint in render["required_artifacts"]["checkpoints"]:
            actual = _file_ref(Path(checkpoint["path"]), where="required checkpoint")
            if actual["sha256"] != checkpoint["sha256"]:
                raise RecoveryError("required checkpoint hash drift")
            required_files[actual["path"]] = actual
        # The render binds an immutable historical prefix. The live ledger is
        # append-only and must never be staged/replaced with these older bytes.
        ledger_record = dict(render["required_artifacts"]["seed_ledger"])
        _verify_ledger_snapshot_record(ledger_record)
        source_refs[arm] = {
            "lock": _file_ref(lock_path, where=f"{arm} lock"),
            "lock_sha256": lock["contract_sha256"],
            "render": _file_ref(render_path, where=f"{arm} render"),
            "render_sha256": render["render_sha256"],
            "seed_ledger_snapshot": ledger_record,
        }
        lock_jobs = {job["job_id"]: job for job in lock["fleet"]["jobs"]}
        commands = [
            command for command in render["commands"] if command["category"] in ALLOWED
        ]
        if len(commands) != 56 or any(
            command["category"] == "current_producer" for command in commands
        ):
            raise RecoveryError(f"{arm} must yield exactly 56 opponent-only commands")
        by_worker: dict[str, list[dict[str, Any]]] = {}
        for source in commands:
            sealed = lock_jobs[source["job_id"]]
            if (source["host_alias"], int(source["gpu"])) != assignments[
                source["worker_id"]
            ]:
                raise RecoveryError("render/placement lane drift")
            argv = list(source["argv"])
            source_science = _science_projection(argv)
            output = f"{recovery_root}/outputs/{arm}/{source['job_id']}"
            _replace_option(argv, "--out-dir", output)
            if "--no-rust-featurize" not in argv:
                raise RecoveryError(
                    "source command lacks sealed no-rust implementation flag"
                )
            argv[argv.index("--no-rust-featurize")] = "--rust-featurize"
            argv.append("--native-mcts-hot-loop")
            if _science_projection(argv) != source_science:
                raise RecoveryError("recovery changed source command science")
            opponent_index = argv.index("--opponent-mix-manifest") + 1
            opponent = _file_ref(Path(argv[opponent_index]), where="opponent manifest")
            expected_opponents = {
                row["path"]: row["sha256"]
                for row in render["required_artifacts"]["rendered_opponent_mix"]
            }
            if expected_opponents.get(opponent["path"]) != opponent["sha256"]:
                raise RecoveryError("opponent manifest is not render-bound")
            required_files[opponent["path"]] = opponent
            if int(argv[argv.index("--base-seed") + 1]) != int(sealed["base_seed"]):
                raise RecoveryError(
                    "recovery seed differs from original unused subrange"
                )
            if int(argv[argv.index("--games") + 1]) != int(sealed["attempts"]):
                raise RecoveryError("recovery attempts differ from sealed maximum")
            ledger_claim = dict(source["ledger_claim"])
            if (
                set(ledger_claim) != {"path", "row", "row_sha256"}
                or ledger_claim["path"] != ledger_record["path"]
                or ledger_claim["row_sha256"] != _digest(ledger_claim["row"])
            ):
                raise RecoveryError("rendered live ledger claim identity drift")
            command = {
                "job_id": source["job_id"],
                "worker_id": source["worker_id"],
                "arm_id": arm,
                "category": source["category"],
                "host_alias": source["host_alias"],
                "gpu": int(source["gpu"]),
                "selected_quota": int(sealed["games"]),
                "max_attempts": int(sealed["attempts"]),
                "base_seed": int(sealed["base_seed"]),
                "seed_end": int(sealed["seed_end"]),
                "claim_label": sealed["claim_label"],
                "ledger_claim": ledger_claim,
                "opponent_manifest": opponent,
                "source_argv_sha256": source["argv_sha256"],
                "source_science_sha256": _digest(source_science),
                "output_dir": output,
                "argv": argv,
                "argv_sha256": _digest(argv),
            }
            by_worker.setdefault(source["worker_id"], []).append(command)
        if len(by_worker) != 28:
            raise RecoveryError(f"{arm} must contain exactly 28 physical lanes")
        for worker_id, lane_commands in sorted(by_worker.items()):
            lane_commands.sort(key=lambda item: ALLOWED.index(item["category"]))
            if tuple(item["category"] for item in lane_commands) != ALLOWED:
                raise RecoveryError("lane does not contain both opponent categories")
            lanes.append(
                {
                    "lane_id": f"{arm}-{worker_id}",
                    "arm_id": arm,
                    "worker_id": worker_id,
                    "host_alias": lane_commands[0]["host_alias"],
                    "gpu": lane_commands[0]["gpu"],
                    "commands": lane_commands,
                    "receipt": f"{recovery_root}/receipts/{arm}/{worker_id}.json",
                    "log_dir": f"{recovery_root}/logs/{arm}/{worker_id}",
                    "quarantine_dir": f"{recovery_root}/quarantine/{arm}/{worker_id}",
                }
            )
    if len(lanes) != 56:
        raise RecoveryError("recovery must contain exactly 28 lanes per arm")
    plan: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA,
        "label": LABEL,
        "promotable": False,
        "mode": "dry-run",
        "runtime_source_repo": runtime_source_repo,
        "runtime_repo": staged_runtime_repo,
        "runtime_python": str(runtime_python),
        "runtime_commit": config["runtime_commit"],
        "native_wheel": {**wheel, "filename": wheel_config["filename"]},
        "runtime_files": runtime_files,
        "recovery_root": recovery_root,
        "config": _file_ref(config_path, where="recovery config"),
        "placement": _file_ref(placement_path, where="placement"),
        "source_artifacts": source_refs,
        "required_host_files": [
            required_files[path] for path in sorted(required_files)
        ],
        "failed_receipts": failures,
        "historical_parent_receipt": historical_parent,
        "lanes": lanes,
    }
    plan["plan_sha256"] = _digest(plan)
    _atomic_exact(out, plan)
    return plan


def _verify_plan(path: Path, *, verify_sources: bool = True) -> dict[str, Any]:
    plan = _load(_regular(path, where="recovery plan"), where="recovery plan")
    stated = plan.get("plan_sha256")
    unhashed = dict(plan)
    unhashed.pop("plan_sha256", None)
    if (
        plan.get("schema_version") != PLAN_SCHEMA
        or plan.get("label") != LABEL
        or plan.get("promotable") is not False
        or stated != _digest(unhashed)
    ):
        raise RecoveryError("recovery plan identity/digest drift")
    if len(plan.get("lanes", [])) != 56:
        raise RecoveryError("recovery plan lane count drift")
    if verify_sources:
        for record in [plan["config"], plan["placement"], *plan["failed_receipts"]]:
            path_value = _regular(Path(record["path"]), where="plan source")
            if _sha256(path_value) != record["sha256"]:
                raise RecoveryError("recovery plan source drift")
    return plan


def _lane(plan: dict[str, Any], lane_id: str) -> dict[str, Any]:
    rows = [lane for lane in plan["lanes"] if lane["lane_id"] == lane_id]
    if len(rows) != 1:
        raise RecoveryError(f"unknown/duplicate lane {lane_id}")
    return rows[0]


def run_lane(plan_path: Path, lane_id: str, *, resume: bool) -> dict[str, Any]:
    plan = _verify_plan(plan_path, verify_sources=False)
    lane = _lane(plan, lane_id)
    repo = Path(plan["runtime_repo"])
    marker = repo / ".experimental_recovery_commit"
    if (
        not marker.is_file()
        or marker.read_text(encoding="utf-8").strip() != plan["runtime_commit"]
    ):
        raise RecoveryError("runtime commit drift")
    for record in plan["runtime_files"]:
        if _sha256(repo / record["path"]) != record["sha256"]:
            raise RecoveryError("runtime code drift")
    receipt_path = Path(lane["receipt"])
    if receipt_path.exists():
        receipt = _load(receipt_path, where="lane receipt")
        if (
            receipt.get("plan_sha256") != plan["plan_sha256"]
            or receipt.get("lane_id") != lane_id
        ):
            raise RecoveryError("existing lane receipt drift")
        if receipt.get("status") == "complete":
            return receipt
        if not resume:
            raise RecoveryError("incomplete lane receipt exists; pass --resume")
    else:
        receipt = {
            "schema_version": RECEIPT_SCHEMA,
            "label": LABEL,
            "promotable": False,
            "plan_sha256": plan["plan_sha256"],
            "lane_id": lane_id,
            "status": "prepared",
            "jobs": {},
        }
        _atomic_exact(receipt_path, receipt, mode=0o600)
        os.chmod(receipt_path, 0o600)
    for command in lane["commands"]:
        prior = receipt["jobs"].get(command["job_id"])
        if prior and prior.get("status") == "complete":
            continue
        output = Path(command["output_dir"])
        if output.exists() and any(output.iterdir()):
            if not resume:
                raise RecoveryError("nonempty recovery output requires --resume")
            quarantine = (
                Path(lane["quarantine_dir"]) / f"{command['job_id']}-{time.time_ns()}"
            )
            quarantine.parent.mkdir(parents=True, exist_ok=True)
            os.replace(output, quarantine)
        output.mkdir(parents=True, exist_ok=True)
        _atomic_exact(
            output / "experimental_nonpromotable.json",
            {
                "label": LABEL,
                "promotable": False,
                "plan_sha256": plan["plan_sha256"],
                "job_id": command["job_id"],
            },
        )
        environment = {
            "HOME": "/home/ubuntu",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": f"{repo}/src:{repo}",
            "TMPDIR": "/tmp",
            "TZ": "UTC",
            "CUDA_VISIBLE_DEVICES": str(lane["gpu"]),
            "CUDA_MPS_PIPE_DIRECTORY": "/tmp/mps_pipe_host",
            "CUDA_MPS_LOG_DIRECTORY": "/tmp/mps_log_host",
            "CATAN_SEED_LEDGER": command["ledger_claim"]["path"],
            "CATAN_ZERO_CONFIG_REGISTRY": str(output / "config_registry.jsonl"),
        }
        log = Path(lane["log_dir"]) / f"{command['job_id']}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("ab", buffering=0) as handle:
            process = subprocess.Popen(
                [plan["runtime_python"], *command["argv"]],
                cwd=repo,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                # The supervisor itself owns a fresh session/process group.
                # Keep the generator in that same group so controller rollback
                # cannot miss the Popen->receipt-write window.
                start_new_session=False,
            )
            receipt["status"] = "running"
            receipt["jobs"][command["job_id"]] = {
                "status": "running",
                "pid": process.pid,
                "process_group": os.getpgrp(),
                "log": str(log),
            }
            _atomic_mutable(receipt_path, receipt)
            return_code = process.wait()
        if return_code != 0:
            receipt["status"] = "failed"
            receipt["jobs"][command["job_id"]].update(
                {"status": "failed", "return_code": return_code}
            )
            _atomic_mutable(receipt_path, receipt)
            raise RecoveryError(f"{command['job_id']} exited {return_code}")
        if not (output / "manifest.json").is_file():
            raise RecoveryError(f"{command['job_id']} completed without manifest")
        receipt["jobs"][command["job_id"]].update(
            {"status": "complete", "return_code": 0}
        )
        _atomic_mutable(receipt_path, receipt)
    receipt["status"] = "complete"
    receipt["completed_at"] = time.time()
    _atomic_mutable(receipt_path, receipt)
    return receipt


def status(plan_path: Path) -> dict[str, Any]:
    plan = _verify_plan(plan_path, verify_sources=False)
    counts: dict[str, int] = {}
    lanes = []
    for lane in plan["lanes"]:
        path = Path(lane["receipt"])
        value = (
            _load(path, where="lane receipt")
            if path.exists()
            else {"status": "pending"}
        )
        state = str(value.get("status", "invalid"))
        counts[state] = counts.get(state, 0) + 1
        lanes.append({"lane_id": lane["lane_id"], "status": state})
    return {"label": LABEL, "counts": counts, "lanes": lanes}


def stop(plan_path: Path, *, go: bool) -> dict[str, Any]:
    plan = _verify_plan(plan_path, verify_sources=False)
    targets = []
    for lane in plan["lanes"]:
        path = Path(lane["receipt"])
        if not path.exists():
            continue
        receipt = _load(path, where="lane receipt")
        for job_id, row in receipt.get("jobs", {}).items():
            pid = row.get("pid")
            if row.get("status") == "running" and isinstance(pid, int) and pid > 0:
                targets.append(
                    {"lane_id": lane["lane_id"], "job_id": job_id, "pid": pid}
                )
                if go:
                    process_group = row.get("process_group", pid)
                    if not isinstance(process_group, int) or process_group <= 0:
                        raise RecoveryError("invalid receipt process group")
                    os.killpg(process_group, signal.SIGTERM)
    return {"mode": "committed" if go else "dry-run", "targets": targets}


def _fleet(path: Path) -> dict[str, Any]:
    value = _load(_regular(path, where="fleet manifest"), where="fleet manifest")
    hosts = value.get("hosts")
    if value.get("schema_version") != "catan-gpu-fleet-v1" or not isinstance(
        hosts, list
    ):
        raise RecoveryError("unsupported fleet manifest")
    result = {str(host["alias"]): str(host["address"]) for host in hosts}
    if len(result) != 10:
        raise RecoveryError("recovery requires the canonical ten-host fleet")
    return {**value, "by_alias": result}


def _ssh_base(fleet: dict[str, Any], alias: str, ssh_key: Path | None) -> list[str]:
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        f"StrictHostKeyChecking={fleet.get('strict_host_key_checking', 'accept-new')}",
    ]
    key = ssh_key or (
        Path(str(fleet["ssh_key"])).expanduser() if fleet.get("ssh_key") else None
    )
    if key is not None:
        command += ["-i", str(key)]
    command.append(f"{fleet['ssh_user']}@{fleet['by_alias'][alias]}")
    return command


def _scp_base(fleet: dict[str, Any], ssh_key: Path | None) -> list[str]:
    command = [
        "scp",
        "-q",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        f"StrictHostKeyChecking={fleet.get('strict_host_key_checking', 'accept-new')}",
    ]
    key = ssh_key or (
        Path(str(fleet["ssh_key"])).expanduser() if fleet.get("ssh_key") else None
    )
    if key is not None:
        command += ["-i", str(key)]
    return command


def _remote_python(base: list[str], program: str) -> Any:
    # Stream the program over stdin.  The sealed runtime/tree and append-only
    # ledger checks can be larger than Linux's argv limit; passing them through
    # ``python -c`` made a valid 56-lane launch fail before preflight.
    try:
        result = subprocess.run(
            [*base, "ulimit -n 65536; exec python3 -"],
            check=True,
            text=True,
            capture_output=True,
            input=program,
        )
    except subprocess.CalledProcessError as error:
        stderr = (error.stderr or "")[-8_000:]
        stdout = (error.stdout or "")[-2_000:]
        raise RecoveryError(
            f"remote Python failed on {base[-1]} with exit {error.returncode}; "
            f"stdout={stdout!r}; stderr={stderr!r}"
        ) from error
    return json.loads(result.stdout)


def _git_tree_manifest(repo: Path, commit: str) -> dict[str, Any]:
    output = subprocess.check_output(
        ["git", "-C", str(repo), "ls-tree", "-r", "-z", "--full-tree", commit]
    )
    files = []
    for entry in output.split(b"\0"):
        if not entry:
            continue
        metadata, raw_path = entry.split(b"\t", 1)
        mode, kind, object_id = metadata.decode().split()
        if kind != "blob":
            raise RecoveryError(
                f"runtime tree contains unsupported {kind}: {raw_path!r}"
            )
        files.append(
            {
                "path": raw_path.decode("utf-8"),
                "mode": mode,
                "git_blob_sha1": object_id,
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": "a1-experimental-recovery-runtime-tree-v1",
        "commit": commit,
        "files": files,
    }
    manifest["manifest_sha256"] = _digest(manifest)
    return manifest


def _stage_file(
    *,
    source: Path,
    destination: str,
    expected_sha256: str,
    fleet: dict[str, Any],
    alias: str,
    ssh_key: Path | None,
    operator_dir: str,
) -> None:
    source = _regular(source, where="staged source")
    if _sha256(source) != expected_sha256:
        raise RecoveryError(f"staged source hash drift: {source}")
    base = _ssh_base(fleet, alias, ssh_key)
    probe = _remote_python(
        base,
        "import hashlib,json,pathlib\n"
        f"p=pathlib.Path({destination!r})\n"
        "def sha(q):\n"
        " h=hashlib.sha256()\n"
        " with q.open('rb') as f:\n"
        "  for b in iter(lambda:f.read(1048576),b''): h.update(b)\n"
        " return 'sha256:'+h.hexdigest()\n"
        "print(json.dumps({'state':'missing'} if not p.exists() else "
        "{'state':'match' if p.is_file() and not p.is_symlink() and sha(p)=="
        f"{expected_sha256!r} else 'drift'}}))",
    )
    if probe["state"] == "match":
        return
    if probe["state"] != "missing":
        raise RecoveryError(f"remote artifact drift on {alias}: {destination}")
    incoming = f"{operator_dir}/incoming-{expected_sha256.removeprefix('sha256:')}"
    subprocess.run(
        [
            *_scp_base(fleet, ssh_key),
            str(source),
            f"{fleet['ssh_user']}@{fleet['by_alias'][alias]}:{incoming}",
        ],
        check=True,
    )
    installed = _remote_python(
        base,
        "import hashlib,json,os,pathlib\n"
        f"s=pathlib.Path({incoming!r}); d=pathlib.Path({destination!r}); expected={expected_sha256!r}\n"
        "def sha(q):\n"
        " h=hashlib.sha256()\n"
        " with q.open('rb') as f:\n"
        "  for b in iter(lambda:f.read(1048576),b''): h.update(b)\n"
        " return 'sha256:'+h.hexdigest()\n"
        "assert s.is_file() and not s.is_symlink() and sha(s)==expected\n"
        "d.parent.mkdir(parents=True,exist_ok=True)\n"
        "if d.exists():\n"
        " assert d.is_file() and not d.is_symlink() and sha(d)==expected\n"
        "else:\n"
        " fd=os.open(d,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o444)\n"
        " with os.fdopen(fd,'wb') as out, s.open('rb') as inp:\n"
        "  for b in iter(lambda:inp.read(1048576),b''): out.write(b)\n"
        "  out.flush(); os.fsync(out.fileno())\n"
        " assert sha(d)==expected\n"
        "s.unlink(missing_ok=True)\n"
        "print(json.dumps({'state':'installed'}))",
    )
    if installed != {"state": "installed"}:
        raise RecoveryError(f"remote artifact install failed on {alias}: {destination}")


def _stage_host(
    *,
    plan: dict[str, Any],
    plan_path: Path,
    fleet: dict[str, Any],
    alias: str,
    ssh_key: Path | None,
    runtime_archive: Path,
    tree_manifest: dict[str, Any],
) -> dict[str, Any]:
    base = _ssh_base(fleet, alias, ssh_key)
    operator_dir = f"{plan['recovery_root']}/operator"
    subprocess.run([*base, f"set -e; mkdir -p {shlex.quote(operator_dir)}"], check=True)
    archive_sha = _sha256(runtime_archive)
    archive_remote = f"{operator_dir}/runtime-{archive_sha.removeprefix('sha256:')}.tar"
    runtime_repo = plan["runtime_repo"]
    manifest_sha = tree_manifest["manifest_sha256"]
    runtime_state = _remote_python(
        base,
        "import json,pathlib\n"
        f"r=pathlib.Path({runtime_repo!r}); m=r/'.experimental_recovery_commit'; t=r/'.experimental_recovery_tree.json'\n"
        f"ok=m.is_file() and m.read_text().strip()=={plan['runtime_commit']!r} and t.is_file() and json.load(t.open()).get('manifest_sha256')=={manifest_sha!r}\n"
        "print(json.dumps({'state':'match' if ok else ('missing' if not r.exists() else 'drift')}))",
    )
    if runtime_state["state"] == "drift":
        raise RecoveryError(f"staged runtime drift on {alias}")
    if runtime_state["state"] == "missing":
        subprocess.run(
            [
                *_scp_base(fleet, ssh_key),
                str(runtime_archive),
                f"{fleet['ssh_user']}@{fleet['by_alias'][alias]}:{archive_remote}",
            ],
            check=True,
        )
        _remote_python(
            base,
            "import hashlib,json,os,pathlib,shutil,tarfile,tempfile\n"
            f"a=pathlib.Path({archive_remote!r}); expected={archive_sha!r}; d=pathlib.Path({runtime_repo!r}); tree={tree_manifest!r}\n"
            "def sha(q):\n"
            " h=hashlib.sha256()\n"
            " with q.open('rb') as f:\n"
            "  for b in iter(lambda:f.read(1048576),b''): h.update(b)\n"
            " return 'sha256:'+h.hexdigest()\n"
            "assert a.is_file() and sha(a)==expected and not d.exists()\n"
            "d.parent.mkdir(parents=True,exist_ok=True); t=pathlib.Path(tempfile.mkdtemp(prefix='.runtime-',dir=d.parent))\n"
            "try:\n"
            " with tarfile.open(a) as tf:\n"
            "  root=t.resolve()\n"
            "  for member in tf.getmembers():\n"
            "   target=(t/member.name).resolve()\n"
            "   assert target==root or root in target.parents\n"
            "  tf.extractall(t)\n"
            f" (t/'.experimental_recovery_commit').write_text({(plan['runtime_commit'] + chr(10))!r})\n"
            " (t/'.experimental_recovery_tree.json').write_text(json.dumps(tree,sort_keys=True,separators=(',',':'))+'\\n')\n"
            " os.rename(t,d)\n"
            "finally:\n"
            " if t.exists(): shutil.rmtree(t)\n"
            " a.unlink(missing_ok=True)\n"
            "print(json.dumps({'state':'installed'}))",
        )
    tool_source = Path(__file__).resolve()
    remote_tool = _remote_tool_path(plan)
    _stage_file(
        source=tool_source,
        destination=remote_tool,
        expected_sha256=_sha256(tool_source),
        fleet=fleet,
        alias=alias,
        ssh_key=ssh_key,
        operator_dir=operator_dir,
    )
    remote_plan = f"{operator_dir}/plan.json"
    _stage_file(
        source=plan_path,
        destination=remote_plan,
        expected_sha256=_sha256(plan_path),
        fleet=fleet,
        alias=alias,
        ssh_key=ssh_key,
        operator_dir=operator_dir,
    )
    for record in plan["required_host_files"]:
        _stage_file(
            source=Path(record["path"]),
            destination=record["path"],
            expected_sha256=record["sha256"],
            fleet=fleet,
            alias=alias,
            ssh_key=ssh_key,
            operator_dir=operator_dir,
        )
    return {"host_alias": alias, "runtime": runtime_repo, "operator": operator_dir}


def _remote_tool_path(plan: dict[str, Any]) -> str:
    digest = _sha256(Path(__file__).resolve()).removeprefix("sha256:")
    return (
        f"{plan['recovery_root']}/operator/"
        f"a1_experimental_opponent_recovery-{digest}.py"
    )


def _preflight_host(
    plan: dict[str, Any],
    fleet: dict[str, Any],
    alias: str,
    ssh_key: Path | None,
    tree_manifest: dict[str, Any],
) -> dict[str, Any]:
    records = [
        {"path": f"{plan['runtime_repo']}/{row['path']}", "sha256": row["sha256"]}
        for row in plan["runtime_files"]
    ] + list(plan["required_host_files"])
    host_lanes = [lane for lane in plan["lanes"] if lane["host_alias"] == alias]
    ledger_claims = [
        command["ledger_claim"] for lane in host_lanes for command in lane["commands"]
    ]
    ledger_paths = {str(claim["path"]) for claim in ledger_claims}
    if len(ledger_paths) != 1:
        raise RecoveryError(f"host {alias} has ambiguous live ledger paths")
    expected_rows = [str(claim["row"]) for claim in ledger_claims]
    if len(expected_rows) != len(set(expected_rows)):
        raise RecoveryError(f"host {alias} has duplicate planned ledger claims")
    snapshots = [
        str(plan["source_artifacts"][arm]["seed_ledger_snapshot"]["snapshot_text"])
        for arm in ("n128", "n256")
    ]
    wheel_hash = plan["native_wheel"]["sha256"].removeprefix("sha256:")
    native_probe = (
        "import json\n"
        "from importlib.metadata import distribution, version\n"
        "import catanatron_rs as r\n"
        "assert version('catanatron_rs') == '0.1.5'\n"
        "d = distribution('catanatron_rs')\n"
        "p = d.locate_file('catanatron_rs-0.1.5.dist-info/direct_url.json')\n"
        "u = json.load(open(p))\n"
        f"assert u['archive_info']['hash'] == {'sha256=' + wheel_hash!r}\n"
        "assert callable(getattr(r, 'gumbel_search', None))\n"
        "assert hasattr(r.Game, 'determinize_for_player')\n"
        "assert hasattr(r, 'build_entity_features_flat')\n"
    )
    ledger_path = next(iter(ledger_paths))
    base = _ssh_base(fleet, alias, ssh_key)
    result = _remote_python(
        base,
        "import hashlib,json,os,pathlib,resource,shutil,subprocess\n"
        f"records={records!r}; repo=pathlib.Path({plan['runtime_repo']!r}); commit={plan['runtime_commit']!r}; wheel={plan['native_wheel']!r}; tree={tree_manifest!r}; ledger_path=pathlib.Path({ledger_path!r}); expected_rows={expected_rows!r}; snapshots={snapshots!r}\n"
        "def sha(q):\n"
        " h=hashlib.sha256()\n"
        " with q.open('rb') as f:\n"
        "  for b in iter(lambda:f.read(1048576),b''): h.update(b)\n"
        " return 'sha256:'+h.hexdigest()\n"
        "assert (repo/'.experimental_recovery_commit').read_text().strip()==commit\n"
        "stored=json.load((repo/'.experimental_recovery_tree.json').open()); assert stored==tree\n"
        "expected_paths={r['path'] for r in tree['files']}\n"
        "allowed_extra={'.experimental_recovery_commit','.experimental_recovery_tree.json'}\n"
        "actual_paths={str(p.relative_to(repo)) for p in repo.rglob('*') if p.is_file() or p.is_symlink()}\n"
        "assert actual_paths==expected_paths|allowed_extra,sorted(actual_paths-(expected_paths|allowed_extra))\n"
        "for r in tree['files']:\n"
        " p=repo/r['path']; assert p.is_file() and not p.is_symlink(),r['path']\n"
        " data=p.read_bytes(); actual=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\\0'+data).hexdigest(); assert actual==r['git_blob_sha1'],r['path']\n"
        "for r in records:\n"
        " p=pathlib.Path(r['path']); assert p.is_file() and not p.is_symlink() and sha(p)==r['sha256'],r['path']\n"
        "ledger_bytes=ledger_path.read_bytes()\n"
        "for snapshot in snapshots: assert ledger_bytes.startswith(snapshot.encode('utf-8'))\n"
        "ledger_lines=ledger_bytes.decode('utf-8').splitlines()\n"
        "for row in expected_rows: assert ledger_lines.count(row)==1,(row,ledger_lines.count(row))\n"
        "live_ledger_sha256='sha256:'+hashlib.sha256(ledger_bytes).hexdigest()\n"
        "assert subprocess.check_output(['systemctl','is-active','nvidia-mps'],text=True).strip()=='active'\n"
        "assert pathlib.Path('/tmp/mps_pipe_host/control').exists() or pathlib.Path('/tmp/mps_pipe_host/control_lock').exists()\n"
        "soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE); assert hard>=65536\n"
        "resource.setrlimit(resource.RLIMIT_NOFILE,(65536,hard)); assert resource.getrlimit(resource.RLIMIT_NOFILE)[0]>=65536\n"
        f"assert shutil.disk_usage({plan['recovery_root']!r}).free>=20*1024**3\n"
        f"py={plan['runtime_python']!r}; native_probe={native_probe!r}\n"
        "subprocess.run([py,'-c',native_probe],check=True)\n"
        "print(json.dumps({'state':'ready','files':len(records),'live_ledger_sha256':live_ledger_sha256}))",
    )
    return {"host_alias": alias, **result}


def launch(
    plan_path: Path,
    *,
    fleet_path: Path,
    ssh_key: Path | None,
    go: bool,
    resume: bool,
) -> dict[str, Any]:
    plan_path = _regular(plan_path, where="recovery plan")
    plan = _verify_plan(plan_path)
    fleet = _fleet(fleet_path)
    targets = [
        {
            "lane_id": lane["lane_id"],
            "host_alias": lane["host_alias"],
            "gpu": lane["gpu"],
        }
        for lane in plan["lanes"]
    ]
    if not go:
        return {"mode": "dry-run", "label": LABEL, "targets": targets}
    remote_plan = f"{plan['recovery_root']}/operator/plan.json"
    remote_tool = _remote_tool_path(plan)
    aliases = sorted({lane["host_alias"] for lane in plan["lanes"]})
    source_repo = Path(plan["runtime_source_repo"])
    subprocess.run(
        [
            "git",
            "-C",
            str(source_repo),
            "fetch",
            "--quiet",
            "origin",
            plan["runtime_commit"],
        ],
        check=True,
    )
    resolved = subprocess.check_output(
        [
            "git",
            "-C",
            str(source_repo),
            "rev-parse",
            f"{plan['runtime_commit']}^{{commit}}",
        ],
        text=True,
    ).strip()
    if resolved != plan["runtime_commit"]:
        raise RecoveryError(
            "runtime commit is unavailable from the canonical source repo"
        )
    tree_manifest = _git_tree_manifest(source_repo, plan["runtime_commit"])
    staged: list[dict[str, Any]] = []
    preflight: list[dict[str, Any]] = []
    launched: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="a1-recovery-runtime-") as directory:
        archive = Path(directory) / "runtime.tar"
        subprocess.run(
            [
                "git",
                "-C",
                str(source_repo),
                "archive",
                "--format=tar",
                "-o",
                str(archive),
                plan["runtime_commit"],
            ],
            check=True,
        )
        for alias in aliases:
            staged.append(
                _stage_host(
                    plan=plan,
                    plan_path=plan_path,
                    fleet=fleet,
                    alias=alias,
                    ssh_key=ssh_key,
                    runtime_archive=archive,
                    tree_manifest=tree_manifest,
                )
            )
        # Global barrier: no supervisor starts until every host proves the exact
        # runtime, checkpoint, opponent manifest, ledger, MPS, and native wheel.
        for alias in aliases:
            preflight.append(
                _preflight_host(plan, fleet, alias, ssh_key, tree_manifest)
            )
        live_ledger_sha256 = _require_ledger_consensus(preflight)
    by_host = {
        alias: [lane for lane in plan["lanes"] if lane["host_alias"] == alias]
        for alias in aliases
    }
    try:
        for alias in aliases:
            base = _ssh_base(fleet, alias, ssh_key)
            lane_specs = [
                {
                    "lane_id": lane["lane_id"],
                    "log": f"{lane['log_dir']}/supervisor.log",
                }
                for lane in by_host[alias]
            ]
            result = _remote_python(
                base,
                "import json,os,pathlib,resource,subprocess\n"
                f"specs={lane_specs!r}; py={plan['runtime_python']!r}; tool={remote_tool!r}; plan={remote_plan!r}; repo={plan['runtime_repo']!r}; resume={resume!r}\n"
                "started=[]\n"
                "soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE); assert hard>=65536; resource.setrlimit(resource.RLIMIT_NOFILE,(65536,hard))\n"
                "try:\n"
                " for s in specs:\n"
                "  log=pathlib.Path(s['log']); log.parent.mkdir(parents=True,exist_ok=True)\n"
                "  argv=[py,tool,'run-lane','--plan',plan,'--lane',s['lane_id']]+(['--resume'] if resume else [])\n"
                "  env=dict(os.environ); env['PYTHONPATH']=repo+'/src:'+repo; env['PYTHONDONTWRITEBYTECODE']='1'; env['PYTHONNOUSERSITE']='1'\n"
                "  h=log.open('ab',buffering=0)\n"
                "  try: p=subprocess.Popen(argv,stdout=h,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=env)\n"
                "  finally: h.close()\n"
                "  started.append({'lane_id':s['lane_id'],'supervisor_pid':p.pid})\n"
                "except BaseException:\n"
                " for s in started:\n"
                "  try: os.killpg(s['supervisor_pid'],15)\n"
                "  except ProcessLookupError: pass\n"
                " raise\n"
                "print(json.dumps(started))",
            )
            launched.extend({"host_alias": alias, **row} for row in result)
    except BaseException as error:
        rollback_errors: list[str] = []
        for alias in sorted({row["host_alias"] for row in launched}):
            try:
                base = _ssh_base(fleet, alias, ssh_key)
                supervisor_pids = [
                    row["supervisor_pid"]
                    for row in launched
                    if row["host_alias"] == alias
                ]
                _remote_python(
                    base,
                    "import json,os,signal\n"
                    f"pids={supervisor_pids!r}; stopped=[]\n"
                    "for pid in pids:\n"
                    " try: os.killpg(pid,signal.SIGTERM); stopped.append(pid)\n"
                    " except ProcessLookupError: pass\n"
                    "print(json.dumps({'supervisors_stopped':stopped}))",
                )
                # Supervisors are stopped first so they cannot create a child
                # after the receipt-based child cleanup has already passed.
                command = (
                    f"PYTHONPATH={shlex.quote(plan['runtime_repo'] + '/src:' + plan['runtime_repo'])} "
                    f"{shlex.quote(plan['runtime_python'])} {shlex.quote(remote_tool)} stop "
                    f"--plan {shlex.quote(remote_plan)} --go"
                )
                subprocess.run(
                    [*base, command], check=True, capture_output=True, text=True
                )
            except subprocess.SubprocessError as rollback_error:
                rollback_errors.append(f"{alias}: {rollback_error}")
        detail = f"; rollback errors: {rollback_errors}" if rollback_errors else ""
        raise RecoveryError(
            f"cohort launch failed and rollback was attempted: {error}{detail}"
        ) from error
    if len(launched) != 56:
        raise RecoveryError("cohort launch did not return exactly 56 supervisors")
    return {
        "mode": "committed",
        "label": LABEL,
        "staged": staged,
        "preflight": preflight,
        "live_ledger_sha256": live_ledger_sha256,
        "launched": launched,
    }


def fleet_status(
    plan_path: Path, *, fleet_path: Path, ssh_key: Path | None
) -> dict[str, Any]:
    plan = _verify_plan(plan_path)
    fleet = _fleet(fleet_path)
    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    aliases = sorted({lane["host_alias"] for lane in plan["lanes"]})
    for alias in aliases:
        lanes = [lane for lane in plan["lanes"] if lane["host_alias"] == alias]
        specs = [
            {"lane_id": lane["lane_id"], "receipt": lane["receipt"]} for lane in lanes
        ]
        values = _remote_python(
            _ssh_base(fleet, alias, ssh_key),
            "import json,pathlib\n"
            f"specs={specs!r}; out=[]\n"
            "for s in specs:\n"
            " p=pathlib.Path(s['receipt']); v=json.load(p.open()) if p.exists() else {'status':'pending'}\n"
            " out.append({'lane_id':s['lane_id'],'status':str(v.get('status','invalid'))})\n"
            "print(json.dumps(out))",
        )
        for value in values:
            state = value["status"]
            counts[state] = counts.get(state, 0) + 1
            rows.append(
                {"lane_id": value["lane_id"], "host_alias": alias, "status": state}
            )
    return {"label": LABEL, "counts": counts, "lanes": rows}


def fleet_stop(
    plan_path: Path, *, fleet_path: Path, ssh_key: Path | None, go: bool
) -> dict[str, Any]:
    plan = _verify_plan(plan_path)
    fleet = _fleet(fleet_path)
    remote_plan = f"{plan['recovery_root']}/operator/plan.json"
    remote_tool = _remote_tool_path(plan)
    rows = []
    aliases = sorted({lane["host_alias"] for lane in plan["lanes"]})
    for alias in aliases:
        base = _ssh_base(fleet, alias, ssh_key)
        go_flag = " --go" if go else ""
        pythonpath = plan["runtime_repo"] + "/src:" + plan["runtime_repo"]
        command = (
            f"PYTHONPATH={shlex.quote(pythonpath)} {shlex.quote(plan['runtime_python'])} "
            f"{shlex.quote(remote_tool)} stop --plan {shlex.quote(remote_plan)}{go_flag}"
        )
        result = subprocess.run(
            [*base, command], check=True, text=True, capture_output=True
        )
        value = json.loads(result.stdout)
        rows.append({"host_alias": alias, "targets": value["targets"]})
    return {"mode": "committed" if go else "dry-run", "hosts": rows}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--config", type=Path, required=True)
    plan.add_argument("--failed-receipt", action="append", type=Path, default=[])
    plan.add_argument("--out", type=Path, required=True)
    run = sub.add_parser("run-lane")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--lane", required=True)
    run.add_argument("--resume", action="store_true")
    launch_parser = sub.add_parser("launch")
    launch_parser.add_argument("--plan", type=Path, required=True)
    launch_parser.add_argument("--fleet-manifest", type=Path, required=True)
    launch_parser.add_argument("--ssh-key", type=Path)
    launch_parser.add_argument("--resume", action="store_true")
    launch_parser.add_argument("--go", action="store_true")
    for name in ("status", "stop"):
        item = sub.add_parser(name)
        item.add_argument("--plan", type=Path, required=True)
        item.add_argument("--fleet-manifest", type=Path)
        item.add_argument("--ssh-key", type=Path)
        if name == "stop":
            item.add_argument("--go", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "plan":
            value = build_plan(
                config_path=args.config,
                failed_receipts=args.failed_receipt,
                out=args.out,
            )
        elif args.command == "run-lane":
            value = run_lane(args.plan, args.lane, resume=args.resume)
        elif args.command == "launch":
            value = launch(
                args.plan,
                fleet_path=args.fleet_manifest,
                ssh_key=args.ssh_key,
                go=args.go,
                resume=args.resume,
            )
        elif args.command == "status":
            value = (
                fleet_status(
                    args.plan, fleet_path=args.fleet_manifest, ssh_key=args.ssh_key
                )
                if args.fleet_manifest
                else status(args.plan)
            )
        else:
            value = (
                fleet_stop(
                    args.plan,
                    fleet_path=args.fleet_manifest,
                    ssh_key=args.ssh_key,
                    go=args.go,
                )
                if args.fleet_manifest
                else stop(args.plan, go=args.go)
            )
    except (RecoveryError, OSError, ValueError, subprocess.SubprocessError) as error:
        _parser().error(str(error))
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
