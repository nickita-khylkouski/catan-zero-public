#!/usr/bin/env python3
"""Manual, fail-closed executor for sealed legacy and dual-arm A1 renders."""

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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools import a1_pre_wave_contract as contract  # noqa: E402
from tools import production_runtime_contract as runtime_contract  # noqa: E402
from catan_zero.production_contracts import NATIVE_REQUIRED_CAPABILITIES  # noqa: E402

HOST_SCHEMA = "a1-production-hosts-v1"
RECEIPT_SCHEMA = "a1-production-executor-receipt-v1"
BRIDGE_SCHEMA = "a1-frozen-plan-hardened-executor-bridge-v1"
LANE_SCHEMA = "a1-production-lane-v1"
SAFE_ALIAS = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
CATEGORY_ORDER = ("current_producer", "recent_history", "hard_negative")
# One generator owns the CUDA context on each physical GPU and serves its CPU
# search workers through EvalServer.  MPS belonged to the retired design where
# every worker opened its own CUDA context; exporting its variables here can
# silently reconnect the new single-owner process to a stale host daemon.
CLIENT_ENVIRONMENT: dict[str, str] = {}
SUPERVISOR_ENVIRONMENT = {"PYTHONDONTWRITEBYTECODE": "1"}
REQUIRED_NOFILE_SOFT = 65_536
STOP_SSH_TIMEOUT_SECONDS = 45.0
WAIT_POLL_SECONDS = 5.0
MAX_PARALLEL_STAGE_HOSTS = 12
PRODUCTION_RUNTIME = runtime_contract.load_runtime_contract()
PRODUCTION_RUNTIME_CONTRACT_PATH = runtime_contract.DEFAULT_CONTRACT
NATIVE_WHEEL_VERSION = PRODUCTION_RUNTIME["catanatron_rs_version"]
NATIVE_WHEEL_NAME = PRODUCTION_RUNTIME["catanatron_rs_wheel_filename"]
NATIVE_WHEEL_INVENTORY = _REPO_ROOT / "native/catanatron-rs/WHEEL_SHA256SUMS"
VENDORED_CATANATRON_SENTINEL = "vendor/catanatron/catanatron/catanatron/models/map.py"
HISTORICAL_DB1_REPO_ROOT = Path("/home/ubuntu/catan-db1c8b1-campaign")
HISTORICAL_DB1_CAMPAIGN_PATH = (
    HISTORICAL_DB1_REPO_ROOT
    / "configs/operations/a1-dual-arm-56gpu-20260710/contract.json"
)
FORBIDDEN_ADAPTIVE_ARGV = (
    "--n-full-wide",
    "--n-full-wide-threshold",
    "--raw-policy-above-width",
)

_REMOTE_INSTALL_PRECHECK_SCRIPT = r"""
import hashlib
import pathlib
import stat
import sys

destination = pathlib.Path(sys.argv[1])
expected = sys.argv[2]

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()

if not destination.is_absolute():
    raise SystemExit("destination is not canonical")
try:
    metadata = destination.lstat()
except FileNotFoundError:
    if destination.resolve(strict=False) != destination:
        raise SystemExit("destination is not canonical")
    raise SystemExit(3)
if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
    raise SystemExit("destination is not a regular non-symlink file")
if destination.resolve(strict=False) != destination:
    raise SystemExit("destination is not canonical")
if sha256(destination) != expected:
    raise SystemExit("destination exists with different bytes")
"""

_REMOTE_INSTALL_SCRIPT = r"""
import hashlib
import os
import pathlib
import shutil
import stat
import sys

source = pathlib.Path(sys.argv[1])
destination = pathlib.Path(sys.argv[2])
expected = sys.argv[3]

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()

source_metadata = source.lstat()
if stat.S_ISLNK(source_metadata.st_mode) or not stat.S_ISREG(source_metadata.st_mode):
    raise SystemExit("incoming source is not a regular non-symlink file")
if sha256(source) != expected:
    raise SystemExit("incoming source hash mismatch")
destination.parent.mkdir(parents=True, exist_ok=True)
if not destination.is_absolute():
    raise SystemExit("destination is not canonical")
try:
    destination_metadata = destination.lstat()
except FileNotFoundError:
    destination_metadata = None
if destination_metadata is not None:
    if stat.S_ISLNK(destination_metadata.st_mode) or not stat.S_ISREG(destination_metadata.st_mode):
        raise SystemExit("destination is not a regular non-symlink file")
    if destination.resolve(strict=False) != destination:
        raise SystemExit("destination is not canonical")
    if sha256(destination) == expected:
        raise SystemExit(0)
    raise SystemExit("destination exists with different bytes")
if destination.resolve(strict=False) != destination:
    raise SystemExit("destination is not canonical")
flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
descriptor = os.open(destination, flags, 0o444)
with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_handle:
    shutil.copyfileobj(input_handle, output, length=1 << 20)
    output.flush()
    os.fsync(output.fileno())
if sha256(destination) != expected:
    raise SystemExit("installed destination hash mismatch")
"""

_REMOTE_BULK_PRECHECK_SCRIPT = r"""
import hashlib
import json
import pathlib
import stat
import sys

receipt_path = pathlib.Path(sys.argv[1])
expected_manifest = sys.argv[2]
expected_bundle = sys.argv[3]

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()

if not receipt_path.is_absolute() or receipt_path.resolve(strict=False) != receipt_path:
    raise SystemExit("bulk receipt path is not canonical")
try:
    metadata = receipt_path.lstat()
except FileNotFoundError:
    raise SystemExit(3)
if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
    raise SystemExit("bulk receipt is not a regular non-symlink file")
try:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
except (OSError, UnicodeError, json.JSONDecodeError) as error:
    raise SystemExit("bulk receipt is unreadable: " + repr(error))
if (
    receipt.get("schema_version") != "a1-production-bulk-install-v1"
    or receipt.get("manifest_sha256") != expected_manifest
    or receipt.get("bundle_sha256") != expected_bundle
):
    raise SystemExit("bulk receipt binds different bytes")
if receipt.get("status") != "complete":
    raise SystemExit(3)
artifacts = receipt.get("artifacts")
if not isinstance(artifacts, list) or not artifacts:
    raise SystemExit("bulk receipt has no artifact inventory")
destinations = set()
for record in artifacts:
    if not isinstance(record, dict) or set(record) != {"destination", "sha256", "mode"}:
        raise SystemExit("bulk receipt artifact schema drift")
    destination = pathlib.Path(record["destination"])
    if (
        str(destination) in destinations
        or not destination.is_absolute()
        or destination.resolve(strict=False) != destination
    ):
        raise SystemExit("bulk receipt destination is unsafe or duplicated")
    destinations.add(str(destination))
    try:
        metadata = destination.lstat()
    except FileNotFoundError:
        raise SystemExit("completed bulk destination is missing")
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or sha256(destination) != record["sha256"]
        or stat.S_IMODE(metadata.st_mode) != int(record["mode"])
    ):
        raise SystemExit("completed bulk destination drift")
"""

_REMOTE_BULK_INSTALL_SCRIPT = r"""
import hashlib
import json
import os
import pathlib
import stat
import sys
import tarfile
import time
import uuid

bundle = pathlib.Path(sys.argv[1])
expected_bundle = sys.argv[2]
receipt_path = pathlib.Path(sys.argv[3])

def sha256_path(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()

def sha256_handle(handle):
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(1 << 20), b""):
        digest.update(block)
    return "sha256:" + digest.hexdigest()

def canonical_digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()

def validate_destination(destination, expected, mode):
    if not destination.is_absolute() or destination.resolve(strict=False) != destination:
        raise SystemExit("bulk destination is not canonical: " + str(destination))
    try:
        metadata = destination.lstat()
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise SystemExit("bulk destination is not a regular non-symlink file: " + str(destination))
    if sha256_path(destination) != expected:
        raise SystemExit("bulk destination exists with different bytes: " + str(destination))
    if stat.S_IMODE(metadata.st_mode) != mode:
        raise SystemExit("bulk destination mode drift: " + str(destination))
    return True

def load_receipt():
    if not receipt_path.is_absolute() or receipt_path.resolve(strict=False) != receipt_path:
        raise SystemExit("bulk receipt path is not canonical")
    try:
        metadata = receipt_path.lstat()
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise SystemExit("bulk receipt is not a regular non-symlink file")
    try:
        return json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SystemExit("bulk receipt is unreadable: " + repr(error))

try:
    bundle_metadata = bundle.lstat()
    if stat.S_ISLNK(bundle_metadata.st_mode) or not stat.S_ISREG(bundle_metadata.st_mode):
        raise SystemExit("incoming bulk bundle is not a regular non-symlink file")
    if sha256_path(bundle) != expected_bundle:
        raise SystemExit("incoming bulk bundle hash mismatch")
    with tarfile.open(bundle, "r:") as archive:
        members = archive.getmembers()
        by_name = {member.name: member for member in members}
        if len(by_name) != len(members) or "manifest.json" not in by_name:
            raise SystemExit("bulk bundle member inventory drift")
        manifest_member = by_name["manifest.json"]
        if not manifest_member.isfile() or manifest_member.size > (16 << 20):
            raise SystemExit("bulk manifest member is unsafe")
        manifest_handle = archive.extractfile(manifest_member)
        if manifest_handle is None:
            raise SystemExit("bulk manifest is unreadable")
        try:
            manifest = json.load(manifest_handle)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise SystemExit("bulk manifest is invalid: " + repr(error))
        if not isinstance(manifest, dict) or set(manifest) != {
            "schema_version", "artifacts", "manifest_sha256"
        }:
            raise SystemExit("bulk manifest schema drift")
        unhashed = dict(manifest)
        declared_manifest = unhashed.pop("manifest_sha256")
        if (
            manifest["schema_version"] != "a1-production-bulk-install-v1"
            or declared_manifest != canonical_digest(unhashed)
        ):
            raise SystemExit("bulk manifest semantic digest mismatch")
        artifacts = manifest["artifacts"]
        if not isinstance(artifacts, list) or not artifacts:
            raise SystemExit("bulk manifest has no artifacts")
        expected_members = {"manifest.json"}
        destinations = set()
        normalized = []
        for record in artifacts:
            if not isinstance(record, dict) or set(record) != {
                "member", "destination", "sha256", "mode"
            }:
                raise SystemExit("bulk artifact schema drift")
            member_name = record["member"]
            destination_text = record["destination"]
            expected = record["sha256"]
            mode = record["mode"]
            if (
                not isinstance(member_name, str)
                or not member_name.startswith("payload/")
                or pathlib.PurePosixPath(member_name).is_absolute()
                or ".." in pathlib.PurePosixPath(member_name).parts
                or member_name in expected_members
                or not isinstance(destination_text, str)
                or destination_text in destinations
                or not isinstance(expected, str)
                or len(expected) != 71
                or not expected.startswith("sha256:")
                or any(character not in "0123456789abcdef" for character in expected[7:])
                or mode != 0o444
            ):
                raise SystemExit("unsafe or duplicate bulk artifact")
            expected_members.add(member_name)
            destinations.add(destination_text)
            member = by_name.get(member_name)
            if member is None or not member.isfile():
                raise SystemExit("bulk artifact member is missing or non-regular")
            source = archive.extractfile(member)
            if source is None or sha256_handle(source) != expected:
                raise SystemExit("bulk artifact member hash mismatch")
            destination = pathlib.Path(destination_text)
            normalized.append((member, destination, expected, mode))
        if set(by_name) != expected_members:
            raise SystemExit("bulk bundle has unmanifested members")

        # Validate every destination before creating any file. A failed bundle
        # therefore never installs a partially authenticated artifact set.
        existing = {
            str(destination): validate_destination(destination, expected, mode)
            for _member, destination, expected, mode in normalized
        }
        receipt_artifacts = [
            {"destination": str(destination), "sha256": expected, "mode": mode}
            for _member, destination, expected, mode in normalized
        ]
        receipt = load_receipt()
        if receipt is not None:
            if (
                receipt.get("schema_version") != "a1-production-bulk-install-v1"
                or receipt.get("manifest_sha256") != declared_manifest
                or receipt.get("bundle_sha256") != expected_bundle
                or receipt.get("artifacts") != receipt_artifacts
                or receipt.get("status") not in ("prepared", "complete")
            ):
                raise SystemExit("bulk receipt binds different bytes")
            if receipt["status"] == "complete":
                if not all(existing.values()):
                    raise SystemExit("completed bulk receipt has missing destinations")
                raise SystemExit(0)
        else:
            receipt = {
                "schema_version": "a1-production-bulk-install-v1",
                "status": "prepared",
                "manifest_sha256": declared_manifest,
                "bundle_sha256": expected_bundle,
                "artifacts": receipt_artifacts,
                "created_at": time.time(),
            }
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            if receipt_path.resolve(strict=False) != receipt_path:
                raise SystemExit("bulk receipt parent is not canonical")
            descriptor = os.open(
                receipt_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                json.dump(receipt, output, sort_keys=True)
                output.flush()
                os.fsync(output.fileno())

        for member, destination, expected, mode in normalized:
            if existing[str(destination)]:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.resolve(strict=False) != destination:
                raise SystemExit("bulk destination parent became non-canonical")
            temporary = destination.parent / (
                "." + destination.name + ".bulk-" + uuid.uuid4().hex + ".tmp"
            )
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            try:
                source = archive.extractfile(member)
                if source is None:
                    raise SystemExit("bulk artifact vanished during install")
                with os.fdopen(descriptor, "wb") as output:
                    for block in iter(lambda: source.read(1 << 20), b""):
                        output.write(block)
                    output.flush()
                    os.fsync(output.fileno())
                    os.fchmod(output.fileno(), mode)
                if sha256_path(temporary) != expected:
                    raise SystemExit("bulk temporary artifact hash mismatch")
                try:
                    os.link(temporary, destination, follow_symlinks=False)
                except FileExistsError:
                    if not validate_destination(destination, expected, mode):
                        raise SystemExit("bulk destination race")
            finally:
                temporary.unlink(missing_ok=True)
            if not validate_destination(destination, expected, mode):
                raise SystemExit("installed bulk destination is missing")

        receipt["status"] = "complete"
        receipt["completed_at"] = time.time()
        temporary_receipt = receipt_path.parent / (
            "." + receipt_path.name + "." + uuid.uuid4().hex + ".tmp"
        )
        descriptor = os.open(
            temporary_receipt,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(receipt, output, sort_keys=True)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_receipt, receipt_path)
        if not all(
            validate_destination(destination, expected, mode)
            for _member, destination, expected, mode in normalized
        ):
            raise SystemExit("completed bulk install verification failed")
finally:
    bundle.unlink(missing_ok=True)
"""


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


def _native_wheel_release_identity() -> dict[str, Any]:
    """Read the one canonical native-wheel identity sealed by this checkout."""

    try:
        rows = [
            line.split()
            for line in NATIVE_WHEEL_INVENTORY.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except OSError as error:
        raise ExecutorError(
            f"cannot read native-wheel checksum inventory: {error}"
        ) from error
    if (
        len(rows) != 1
        or len(rows[0]) != 2
        or not re.fullmatch(r"[0-9a-f]{64}", rows[0][0])
        or rows[0][1] != NATIVE_WHEEL_NAME
        or rows[0][0] != PRODUCTION_RUNTIME["catanatron_rs_wheel_sha256"]
    ):
        raise ExecutorError(
            "native-wheel checksum inventory is not the one canonical "
            f"{NATIVE_WHEEL_VERSION} artifact"
        )
    return {
        "version": NATIVE_WHEEL_VERSION,
        "filename": NATIVE_WHEEL_NAME,
        "sha256": "sha256:" + PRODUCTION_RUNTIME["catanatron_rs_wheel_sha256"],
        "required_capabilities": sorted(NATIVE_REQUIRED_CAPABILITIES),
    }


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
        raise ExecutorError(
            f"private host config must be mode 0600: {path} is {mode:o}"
        )
    value = _load(path)
    expected = {
        "schema_version",
        "ssh_user",
        "ssh_key",
        "remote_root",
        "python",
        "hosts",
    }
    if set(value) != expected or value["schema_version"] != HOST_SCHEMA:
        raise ExecutorError(f"host config must use exact {HOST_SCHEMA} schema")
    if not isinstance(value["hosts"], dict) or set(value["hosts"]) != aliases:
        raise ExecutorError("private host aliases must exactly match the sealed render")
    for name, host in value["hosts"].items():
        if (
            not SAFE_ALIAS.fullmatch(name)
            or not isinstance(host, str)
            or not SAFE_ALIAS.fullmatch(host)
        ):
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
        raise ExecutorError(
            f"sealed lock/all-claim verification failed: {error}"
        ) from error
    rendered = _load(render_path)
    if rendered.get("schema_version") != contract.RENDER_SCHEMA:
        raise ExecutorError(f"render schema must be {contract.RENDER_SCHEMA}")
    unhashed = dict(rendered)
    declared_render_sha = unhashed.pop("render_sha256", None)
    if declared_render_sha != contract._digest_value(unhashed):
        raise ExecutorError("render semantic digest mismatch")
    if rendered.get("contract_sha256") != lock["contract_sha256"]:
        raise ExecutorError("render binds a different contract")
    try:
        topology = contract._sealed_game_contract_shape(lock)  # noqa: SLF001
    except contract.ContractError as error:
        raise ExecutorError(
            f"sealed production topology is invalid: {error}"
        ) from error
    arm_id = topology["arm_id"]
    expected_jobs = int(topology["job_count"])
    expected_lanes = int(topology["worker_count"])
    commands = rendered.get("commands")
    if not isinstance(commands, list) or len(commands) != expected_jobs:
        raise ExecutorError(
            f"production render must contain exactly {expected_jobs} commands"
        )
    jobs = {job["job_id"]: job for job in lock["fleet"]["jobs"]}
    if len(jobs) != expected_jobs:
        raise ExecutorError(
            f"sealed production lock must contain exactly {expected_jobs} jobs"
        )
    search = lock.get("science", {}).get("search_operator", {})
    expected_n_full = 256 if arm_id == "n256" else 128
    if int(search.get("n_full", -1)) != expected_n_full:
        raise ExecutorError(
            f"A1 production {arm_id or 'historical'} science is locked to "
            f"n_full={expected_n_full}"
        )
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
        try:
            contract._promoted_producer_job_identity(lock, job)  # noqa: SLF001
        except contract.ContractError as error:
            raise ExecutorError(
                f"unsafe promoted producer identity for {job_id}: {error}"
            ) from error
        if arm_id is not None and (
            command.get("arm_id") != arm_id or job.get("arm_id") != arm_id
        ):
            raise ExecutorError(f"arm identity drift for {job_id}")
        if command.get("argv_sha256") != contract._digest_value(command.get("argv")):
            raise ExecutorError(f"argv digest mismatch for {job_id}")
        expected_argv = contract._generator_argv(lock, job, mix_paths=mix_paths)
        if command.get("argv") != expected_argv:
            raise ExecutorError(
                f"rendered argv differs from sealed command for {job_id}"
            )
        if "--skip-guards" in expected_argv or "--no-seed-claim" in expected_argv:
            raise ExecutorError(f"guard bypass in {job_id}")
        if "--resume" not in expected_argv:
            raise ExecutorError(f"{job_id} lacks explicit exact-run resume semantics")
        try:
            rendered_n_full = int(expected_argv[expected_argv.index("--n-full") + 1])
        except (ValueError, IndexError) as error:
            raise ExecutorError(f"{job_id} lacks an exact --n-full value") from error
        if rendered_n_full != expected_n_full or any(
            flag in expected_argv for flag in FORBIDDEN_ADAPTIVE_ARGV
        ):
            raise ExecutorError(
                f"{job_id} is not the sealed n{expected_n_full}/no-adaptive recipe"
            )
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
        if claim.get("row") != expected_row or claim.get(
            "row_sha256"
        ) != contract._digest_value(expected_row):
            raise ExecutorError(f"claim row drift for {job_id}")
        source = Path(command["output_attestation"]["source"])
        if (
            not source.is_file()
            or _sha256(source) != command["output_attestation"]["source_file_sha256"]
        ):
            raise ExecutorError(f"job attestation drift for {job_id}")
        if arm_id is not None:
            expected_attestation = contract._job_attestation(lock, job)
            if contract._load_json(source) != expected_attestation or command[
                "output_attestation"
            ].get("payload_sha256") != contract._digest_value(expected_attestation):
                raise ExecutorError(
                    f"dual-arm job attestation payload drift for {job_id}"
                )
        by_lane.setdefault(command["worker_id"], []).append(command)
    if seen != set(jobs) or len(by_lane) != expected_lanes:
        raise ExecutorError(
            f"render must cover exactly {expected_lanes} physical lanes and "
            f"{expected_jobs} jobs"
        )
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


def _historical_runtime_root(lock: Mapping[str, Any]) -> Path | None:
    source = lock.get("source_campaign")
    provenance = lock.get("provenance")
    if not isinstance(source, dict) or not isinstance(provenance, dict):
        return None
    if source != {
        "path": str(HISTORICAL_DB1_CAMPAIGN_PATH),
        "sha256": contract.HISTORICAL_DB1_CAMPAIGN_FILE_SHA256,
    }:
        return None
    if provenance.get("executor") != {
        "kind": "executor",
        "path": "tools/fleet/a1_production_executor.py",
        "sha256": contract.HISTORICAL_DB1_EXECUTOR_SHA256,
    }:
        raise ExecutorError("historical db1 lock executor provenance drift")
    return HISTORICAL_DB1_REPO_ROOT


def _relocate_historical_artifact(
    record: Mapping[str, Any], *, historical_root: Path, current_root: Path
) -> tuple[str, Path]:
    raw = Path(str(record["path"]))
    try:
        relative = raw.relative_to(historical_root)
    except ValueError as error:
        raise ExecutorError(
            f"historical runtime artifact uses an unauthorized root: {raw}"
        ) from error
    if not relative.parts or ".." in relative.parts:
        raise ExecutorError(f"historical runtime artifact escapes frozen root: {raw}")
    current = current_root / relative
    for label, path, root in (("historical", raw, historical_root),):
        try:
            metadata = path.lstat()
            resolved = path.resolve(strict=True)
        except OSError as error:
            raise ExecutorError(
                f"{label} runtime artifact is unavailable: {path}"
            ) from error
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or resolved != path
        ):
            raise ExecutorError(
                f"{label} runtime artifact is not canonical regular bytes: {path}"
            )
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise ExecutorError(
                f"{label} runtime artifact escapes repo: {path}"
            ) from error
        if _sha256(path) != record["sha256"]:
            raise ExecutorError(f"{label} runtime artifact hash drift: {path}")
    try:
        current_metadata = current.lstat()
        resolved_current = current.resolve(strict=True)
    except FileNotFoundError:
        return str(relative), raw
    except OSError as error:
        raise ExecutorError(
            f"current runtime artifact is unavailable: {current}"
        ) from error
    if (
        stat.S_ISLNK(current_metadata.st_mode)
        or not stat.S_ISREG(current_metadata.st_mode)
        or resolved_current != current
    ):
        raise ExecutorError(
            f"current runtime artifact is not canonical regular bytes: {current}"
        )
    try:
        resolved_current.relative_to(current_root)
    except ValueError as error:
        raise ExecutorError(
            f"current runtime artifact escapes repo: {current}"
        ) from error
    return (
        (str(relative), current)
        if _sha256(current) == record["sha256"]
        else (str(relative), raw)
    )


def _repo_artifacts(
    rendered: dict[str, Any],
    *,
    repo_root: Path = _REPO_ROOT,
    historical_root: Path | None = None,
) -> list[dict[str, Any]]:
    root = repo_root.resolve(strict=True)
    if historical_root is not None:
        try:
            resolved_historical_root = historical_root.resolve(strict=True)
        except OSError as error:
            raise ExecutorError("historical db1 repo root is unavailable") from error
        if resolved_historical_root != historical_root:
            raise ExecutorError("historical db1 repo root is not canonical")
    required = rendered["required_artifacts"]
    records = [
        *(required.get("guard_configs") or [required["guard_config"]]),
        *required["generator_code"],
        *required["runtime_code_tree"],
    ]
    files: dict[str, Path] = {}
    for record in records:
        raw = Path(str(record["path"]))
        if historical_root is not None and raw.is_absolute():
            relative, path = _relocate_historical_artifact(
                record, historical_root=historical_root, current_root=root
            )
            files[relative] = path
            continue
        path = raw.resolve()
        if _sha256(path) != record["sha256"]:
            raise ExecutorError(f"required repo artifact drift: {path}")
        try:
            relative = path.relative_to(root)
        except ValueError as error:
            raise ExecutorError(
                f"runtime artifact is outside canonical repo: {path}"
            ) from error
        files[str(relative)] = path
    supervisor = (root / "tools/fleet/a1_lane_supervisor.py").resolve()
    files[str(supervisor.relative_to(root))] = supervisor
    executor = (root / "tools/fleet/a1_production_executor.py").resolve()
    files[str(executor.relative_to(root))] = executor
    stop_helper = (root / "tools/fleet/a1_stop_helper.py").resolve()
    files[str(stop_helper.relative_to(root))] = stop_helper
    return [
        {
            "path": key,
            "sha256": _sha256(files[key]),
            "mode": 0o555 if os.access(files[key], os.X_OK) else 0o444,
            **({"source_path": str(files[key])} if files[key] != root / key else {}),
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
        declared_source = record.get("source_path")
        source = (
            Path(str(declared_source))
            if declared_source is not None
            else root / Path(*relative.parts)
        )
        try:
            metadata = source.lstat()
            resolved = source.resolve(strict=True)
        except OSError as error:
            raise ExecutorError(
                f"repo artifact source is unavailable: {source}"
            ) from error
        if (
            not source.is_absolute()
            or stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or resolved != source
            or _sha256(source) != record["sha256"]
        ):
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
    if bridge["plan_sha256"] != plan.get("plan_sha256") or bridge[
        "repo_artifacts_sha256"
    ] != plan.get("repo_artifacts_sha256"):
        raise ExecutorError("frozen-plan executor bridge plan binding drift")
    root = Path(str(bridge["frozen_repo_root"])).resolve(strict=True)
    frozen_path = (root / "tools/fleet/a1_production_executor.py").resolve(strict=True)
    hardened_path = Path(__file__).resolve(strict=True)
    bridge_path = (_REPO_ROOT / "tools/fleet/a1_executor_bridge.py").resolve(
        strict=True
    )
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


def _build_repo_tar(
    artifacts: Sequence[Mapping[str, Any]],
    files: Sequence[Path],
    destination: Path,
) -> str:
    if len(artifacts) != len(files):
        raise ExecutorError("repo artifact/source count drift")
    with tarfile.open(destination, "w") as archive:
        for record, source in zip(artifacts, files):
            info = tarfile.TarInfo(str(record["path"]))
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
    repo_artifacts = _repo_artifacts(
        rendered,
        repo_root=repo_root,
        historical_root=_historical_runtime_root(lock),
    )
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
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "ConnectTimeout=10",
            "-i",
            hosts["ssh_key"],
            f"{hosts['ssh_user']}@{hosts['hosts'][alias]}",
            remote_command,
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout_seconds,
    )


def _scp(hosts: dict[str, Any], alias: str, source: Path, destination: str) -> None:
    result = subprocess.run(
        [
            "scp",
            "-q",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "ConnectTimeout=10",
            "-i",
            hosts["ssh_key"],
            str(source),
            f"{hosts['ssh_user']}@{hosts['hosts'][alias]}:{destination}",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise ExecutorError(f"scp to {alias} failed: {result.stderr.strip()}")


def _build_bulk_install_bundle(
    files: Sequence[tuple[Path, str, str]], destination: Path
) -> tuple[dict[str, Any], str]:
    """Build one deterministic, hash-bound immutable install bundle."""

    unique: dict[str, tuple[Path, str]] = {}
    for source, remote_destination, expected in files:
        remote_path = PurePosixPath(remote_destination)
        if (
            not remote_path.is_absolute()
            or str(remote_path) != remote_destination
            or ".." in remote_path.parts
        ):
            raise ExecutorError(
                f"bulk install destination is not canonical: {remote_destination}"
            )
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", expected):
            raise ExecutorError(
                f"bulk install digest is invalid for {remote_destination}"
            )
        try:
            metadata = source.lstat()
        except OSError as error:
            raise ExecutorError(
                f"cannot stat bulk install source {source}: {error}"
            ) from error
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ExecutorError(
                f"bulk install source is not a regular non-symlink file: {source}"
            )
        if _sha256(source) != expected:
            raise ExecutorError(f"bulk install source hash drift: {source}")
        previous = unique.get(remote_destination)
        candidate = (source, expected)
        if previous is not None and previous != candidate:
            if previous[1] != expected:
                raise ExecutorError(
                    f"conflicting bulk install destination: {remote_destination}"
                )
            # Identical bytes from two local names are one remote artifact.
            continue
        unique[remote_destination] = candidate
    if not unique:
        raise ExecutorError("bulk install bundle cannot be empty")

    artifacts = []
    ordered: list[tuple[str, Path]] = []
    for index, remote_destination in enumerate(sorted(unique)):
        source, expected = unique[remote_destination]
        member = f"payload/{index:08d}"
        artifacts.append(
            {
                "member": member,
                "destination": remote_destination,
                "sha256": expected,
                "mode": 0o444,
            }
        )
        ordered.append((member, source))
    manifest: dict[str, Any] = {
        "schema_version": "a1-production-bulk-install-v1",
        "artifacts": artifacts,
    }
    manifest["manifest_sha256"] = _digest(manifest)
    manifest_bytes = json.dumps(
        manifest, sort_keys=True, separators=(",", ":")
    ).encode()

    with tarfile.open(destination, "w", format=tarfile.USTAR_FORMAT) as archive:
        manifest_info = tarfile.TarInfo("manifest.json")
        manifest_info.size = len(manifest_bytes)
        manifest_info.mode = 0o444
        manifest_info.mtime = 0
        manifest_info.uid = manifest_info.gid = 0
        manifest_info.uname = manifest_info.gname = ""
        archive.addfile(manifest_info, io.BytesIO(manifest_bytes))
        for (member, source), record in zip(ordered, artifacts, strict=True):
            metadata = source.stat()
            info = tarfile.TarInfo(member)
            info.size = metadata.st_size
            info.mode = 0o444
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with source.open("rb") as handle:
                archive.addfile(info, handle)
            if _sha256(source) != record["sha256"]:
                raise ExecutorError(
                    f"bulk install source changed while archiving: {source}"
                )
    return manifest, _sha256(destination)


def _remote_bulk_install(
    hosts: dict[str, Any],
    alias: str,
    files: Sequence[tuple[Path, str, str]],
    temporary_path: Path,
) -> None:
    """Install a host's immutable inputs with one authenticated transaction."""

    bundle = temporary_path / f"bulk-install-{alias}.tar"
    manifest, bundle_sha256 = _build_bulk_install_bundle(files, bundle)
    manifest_sha256 = str(manifest["manifest_sha256"])
    token = manifest_sha256.removeprefix("sha256:")
    receipt_path = f"{hosts['remote_root']}/receipts/bulk-install-{token}.json"
    precheck_command = " ".join(
        shlex.quote(value)
        for value in (
            hosts["python"],
            "-c",
            _REMOTE_BULK_PRECHECK_SCRIPT,
            receipt_path,
            manifest_sha256,
            bundle_sha256,
        )
    )
    precheck = _ssh(hosts, alias, precheck_command)
    if precheck.returncode == 0:
        return
    if precheck.returncode != 3:
        detail = precheck.stderr.strip() or f"exit {precheck.returncode}"
        raise ExecutorError(f"bulk install precheck failed on {alias}: {detail}")

    incoming_dir = f"{hosts['remote_root']}/incoming"
    incoming = f"{incoming_dir}/bulk-{uuid.uuid4().hex}.tar"
    mkdir = _ssh(hosts, alias, f"mkdir -p {shlex.quote(incoming_dir)}")
    if mkdir.returncode != 0:
        raise ExecutorError(
            f"remote bulk mkdir failed on {alias}: {mkdir.stderr.strip()}"
        )
    _scp(hosts, alias, bundle, incoming)
    command = " ".join(
        shlex.quote(value)
        for value in (
            hosts["python"],
            "-c",
            _REMOTE_BULK_INSTALL_SCRIPT,
            incoming,
            bundle_sha256,
            receipt_path,
        )
    )
    result = _ssh(hosts, alias, command)
    if result.returncode != 0:
        # The remote script unlinks the incoming bundle in a finally block.
        detail = (
            result.stderr.strip()
            or result.stdout.strip()
            or f"exit {result.returncode}"
        )
        raise ExecutorError(f"bulk immutable install failed on {alias}: {detail}")


def _remote_install(
    hosts: dict[str, Any], alias: str, source: Path, destination: str, expected: str
) -> None:
    precheck_command = " ".join(
        shlex.quote(value)
        for value in (
            hosts["python"],
            "-c",
            _REMOTE_INSTALL_PRECHECK_SCRIPT,
            destination,
            expected,
        )
    )
    precheck = _ssh(hosts, alias, precheck_command)
    if precheck.returncode == 0:
        return
    if precheck.returncode != 3:
        detail = precheck.stderr.strip() or f"exit {precheck.returncode}"
        raise ExecutorError(f"remote destination precheck failed on {alias}: {detail}")

    incoming = f"{hosts['remote_root']}/incoming/{uuid.uuid4().hex}"
    mkdir = _ssh(hosts, alias, f"mkdir -p {shlex.quote(str(Path(incoming).parent))}")
    if mkdir.returncode != 0:
        raise ExecutorError(f"remote mkdir failed on {alias}: {mkdir.stderr.strip()}")
    _scp(hosts, alias, source, incoming)
    command = " ".join(
        shlex.quote(value)
        for value in (
            hosts["python"],
            "-c",
            _REMOTE_INSTALL_SCRIPT,
            incoming,
            destination,
            expected,
        )
    )
    result = _ssh(hosts, alias, command)
    _ssh(hosts, alias, f"rm -f {shlex.quote(incoming)}")
    if result.returncode != 0:
        raise ExecutorError(
            f"immutable install failed on {alias}: {result.stderr.strip()}"
        )


def _stage_files_by_alias(
    required: Mapping[str, Any], lanes: Mapping[str, Sequence[Mapping[str, Any]]]
) -> dict[str, list[tuple[Path, str, str]]]:
    global_files = [
        *[
            (Path(item["path"]), item["path"], item["sha256"])
            for item in required.get("checkpoints", [])
        ],
        *[
            (Path(item["path"]), item["path"], item["sha256"])
            for item in required["rendered_opponent_mix"]
        ],
    ]
    aliases = {str(lane[0]["host_alias"]) for lane in lanes.values()}
    staged = {alias: list(global_files) for alias in aliases}
    attestations: dict[str, dict[str, tuple[Path, str, str]]] = {
        alias: {} for alias in aliases
    }
    for lane in lanes.values():
        alias = str(lane[0]["host_alias"])
        for command in lane:
            record = command["output_attestation"]
            source = str(record["source"])
            candidate = (Path(source), source, str(record["source_file_sha256"]))
            previous = attestations[alias].get(source)
            if previous is not None and previous != candidate:
                raise ExecutorError(
                    f"conflicting attestation source on {alias}: {source}"
                )
            attestations[alias][source] = candidate
    for alias in aliases:
        staged[alias].extend(
            attestations[alias][path] for path in sorted(attestations[alias])
        )
    return staged


def _append_only_bytes(existing: bytes, desired: bytes) -> bytes:
    if desired == existing or desired.startswith(existing):
        return desired
    raise ExecutorError(
        "remote seed ledger is not an exact prefix of the bound live ledger"
    )


def _remote_sync_append_only_ledger(
    hosts: dict[str, Any], alias: str, source: Path, destination: str, expected: str
) -> None:
    incoming = f"{hosts['remote_root']}/incoming/{uuid.uuid4().hex}"
    mkdir = _ssh(hosts, alias, f"mkdir -p {shlex.quote(str(Path(incoming).parent))}")
    if mkdir.returncode != 0:
        raise ExecutorError(
            f"remote ledger mkdir failed on {alias}: {mkdir.stderr.strip()}"
        )
    _scp(hosts, alias, source, incoming)
    script = r"""import hashlib,os,pathlib,sys,uuid
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
if sha(dst.read_bytes())!=expected: raise SystemExit('installed live ledger digest drift')"""
    command = " ".join(
        shlex.quote(value)
        for value in (hosts["python"], "-c", script, incoming, destination, expected)
    )
    result = _ssh(hosts, alias, command)
    _ssh(hosts, alias, f"rm -f {shlex.quote(incoming)}")
    if result.returncode != 0:
        raise ExecutorError(
            f"append-only ledger sync failed on {alias}: {result.stderr.strip()}"
        )


def _preflight_host(
    hosts: dict[str, Any], alias: str, expected_gpus: Sequence[int]
) -> dict[str, Any]:
    """Read-only launch preflight: topology, runtime, limits, and idle GPUs."""
    expected_native_wheel = _native_wheel_release_identity()
    script = r"""import importlib.metadata,json,resource,subprocess,sys
expected=json.loads(sys.argv[1])
required_nofile=int(sys.argv[2])
expected_native_wheel=json.loads(sys.argv[3])
expected_runtime=json.loads(sys.argv[4])
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
python_version=sys.version.split()[0]
if python_version!=expected_runtime['python_version']: raise SystemExit(f"Python runtime drift: expected {expected_runtime['python_version']}, got {python_version}")
torch_version=str(torch.__version__);torch_cuda_version=str(torch.version.cuda)
if torch_version!=expected_runtime['torch_version']: raise SystemExit(f"torch runtime drift: expected {expected_runtime['torch_version']}, got {torch_version}")
if torch_cuda_version!=expected_runtime['torch_cuda_version']: raise SystemExit(f"torch CUDA runtime drift: expected {expected_runtime['torch_cuda_version']}, got {torch_cuda_version}")
dependency_distributions={'numpy':'numpy_version','networkx':'networkx_version','gymnasium':'gymnasium_version','zstandard':'zstandard_version','scipy':'scipy_version','whr':'whr_version'}
try: dependency_versions={name:importlib.metadata.version(name) for name in dependency_distributions}
except importlib.metadata.PackageNotFoundError as error: raise SystemExit('configured interpreter dependency distribution missing: '+str(error))
expected_dependencies={name:expected_runtime[key] for name,key in dependency_distributions.items()}
if dependency_versions!=expected_dependencies: raise SystemExit(f'production dependency runtime drift: expected={expected_dependencies!r} got={dependency_versions!r}')
if not torch.cuda.is_available() or torch.cuda.device_count()!=len(expected): raise SystemExit(f'torch CUDA topology drift: available={torch.cuda.is_available()} count={torch.cuda.device_count()} expected={len(expected)}')
run=lambda *args:subprocess.run(args,text=True,capture_output=True,check=False)
gpu=run('nvidia-smi','--query-gpu=index','--format=csv,noheader,nounits')
if gpu.returncode: raise SystemExit('nvidia-smi gpu query failed: '+gpu.stderr)
indices=sorted(int(line.strip()) for line in gpu.stdout.splitlines() if line.strip())
if indices!=expected: raise SystemExit(f'GPU topology drift: expected {expected}, got {indices}')
driver=run('nvidia-smi','--query-gpu=driver_version','--format=csv,noheader,nounits')
if driver.returncode: raise SystemExit('nvidia-smi driver query failed: '+driver.stderr)
driver_versions={line.strip() for line in driver.stdout.splitlines() if line.strip()}
if driver_versions!={expected_runtime['nvidia_driver_version']}: raise SystemExit(f"NVIDIA driver runtime drift: expected {expected_runtime['nvidia_driver_version']}, got {sorted(driver_versions)!r}")
nvidia_driver_version=next(iter(driver_versions))
apps=run('nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader,nounits')
if apps.returncode not in (0,): raise SystemExit('nvidia-smi compute query failed: '+apps.stderr)
foreign=[]
for line in apps.stdout.splitlines():
    if not line.strip() or 'No running processes found' in line: continue
    foreign.append(line.strip())
if foreign: raise SystemExit('GPU compute applications already active: '+repr(foreign))
try:
    rust_distribution=importlib.metadata.distribution('catanatron-rs')
except importlib.metadata.PackageNotFoundError:
    raise SystemExit('configured interpreter has no catanatron-rs distribution')
rust_version=rust_distribution.version
if rust_version!=expected_native_wheel['version']:
    raise SystemExit(f"catanatron-rs version drift: expected {expected_native_wheel['version']}, got {rust_version}")
direct_url_raw=rust_distribution.read_text('direct_url.json')
if direct_url_raw is None: raise SystemExit('installed catanatron-rs has no PEP 610 direct_url.json')
try: direct_url=json.loads(direct_url_raw)
except json.JSONDecodeError as error: raise SystemExit('installed catanatron-rs direct_url.json is invalid: '+repr(error))
archive=direct_url.get('archive_info') if isinstance(direct_url,dict) else None
if not isinstance(archive,dict): raise SystemExit('installed catanatron-rs is not bound to a wheel archive')
stated=set()
direct_hash=archive.get('hash')
if isinstance(direct_hash,str): stated.add(direct_hash)
hashes=archive.get('hashes')
if isinstance(hashes,dict) and isinstance(hashes.get('sha256'),str): stated.add('sha256='+hashes['sha256'])
expected_direct='sha256='+expected_native_wheel['sha256'].removeprefix('sha256:')
if stated!={expected_direct}: raise SystemExit(f'installed catanatron-rs wheel digest mismatch: expected={expected_direct} recorded={sorted(stated)}')
capability_fn=getattr(catanatron_rs,'gumbel_search_capabilities',None)
if not callable(capability_fn): raise SystemExit('installed catanatron-rs lacks the native capability contract')
capabilities=set(capability_fn())
required_capabilities=set(expected_native_wheel['required_capabilities'])
missing=sorted(required_capabilities-capabilities)
if missing: raise SystemExit('installed catanatron-rs lacks required capabilities: '+repr(missing))
print(json.dumps({'gpu_indices':indices,'compute_apps':'empty','cuda_owner_model':'one_eval_server_per_physical_gpu','client_environment':{},'python':sys.executable,'python_version':python_version,'torch_version':torch_version,'torch_cuda_version':torch_cuda_version,'dependency_versions':dependency_versions,'nvidia_driver_version':nvidia_driver_version,'catanatron_rs_version':rust_version,'native_wheel_sha256':expected_native_wheel['sha256'],'native_mcts_capabilities':sorted(capabilities),'required_nofile_soft':required_nofile,'nofile_soft_before':nofile_soft_before,'nofile_soft':nofile_soft,'nofile_hard':nofile_hard},sort_keys=True))"""
    command = " ".join(
        shlex.quote(value)
        for value in (
            hosts["python"],
            "-c",
            script,
            json.dumps(sorted(expected_gpus)),
            str(REQUIRED_NOFILE_SOFT),
            json.dumps(expected_native_wheel, sort_keys=True),
            json.dumps(PRODUCTION_RUNTIME, sort_keys=True),
        )
    )
    result = _ssh(hosts, alias, command)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise ExecutorError(f"host preflight failed on {alias}: {detail}")
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ExecutorError(
            f"host preflight returned invalid JSON on {alias}"
        ) from error
    if report.get("client_environment") != CLIENT_ENVIRONMENT:
        raise ExecutorError(f"host CUDA client environment drift on {alias}")
    if report.get("cuda_owner_model") != "one_eval_server_per_physical_gpu":
        raise ExecutorError(f"host CUDA owner model drift on {alias}")
    expected_dependencies = {
        name: PRODUCTION_RUNTIME[f"{name}_version"]
        for name in ("numpy", "networkx", "gymnasium", "zstandard", "scipy", "whr")
    }
    if report.get("python_version") != PRODUCTION_RUNTIME["python_version"]:
        raise ExecutorError(f"host Python runtime drift on {alias}")
    if report.get("torch_version") != PRODUCTION_RUNTIME["torch_version"]:
        raise ExecutorError(f"host torch runtime drift on {alias}")
    if report.get("torch_cuda_version") != PRODUCTION_RUNTIME["torch_cuda_version"]:
        raise ExecutorError(f"host torch CUDA runtime drift on {alias}")
    if report.get("dependency_versions") != expected_dependencies:
        raise ExecutorError(f"host production dependency runtime drift on {alias}")
    if (
        report.get("nvidia_driver_version")
        != PRODUCTION_RUNTIME["nvidia_driver_version"]
    ):
        raise ExecutorError(f"host NVIDIA driver runtime drift on {alias}")
    if report.get("catanatron_rs_version") != expected_native_wheel["version"]:
        raise ExecutorError(f"host catanatron-rs version drift on {alias}")
    if report.get("native_wheel_sha256") != expected_native_wheel["sha256"]:
        raise ExecutorError(f"host catanatron-rs wheel digest drift on {alias}")
    reported_capabilities = report.get("native_mcts_capabilities")
    if (
        not isinstance(reported_capabilities, list)
        or any(not isinstance(value, str) for value in reported_capabilities)
        or not NATIVE_REQUIRED_CAPABILITIES <= set(reported_capabilities)
    ):
        raise ExecutorError(f"host catanatron-rs capability drift on {alias}")
    limit_fields = (
        "required_nofile_soft",
        "nofile_soft_before",
        "nofile_soft",
        "nofile_hard",
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
        not isinstance(key, str)
        or not isinstance(value, str)
        or not key
        or "=" in key
        or "\x00" in key
        or "\x00" in value
        for key, value in environment.items()
    ):
        raise ExecutorError("invalid supervisor launch environment")
    argv = [python, supervisor, "run", "--lane", remote_lane]
    script = r"""import json,pathlib,resource,subprocess,sys
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
print(process.pid,flush=True)"""
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


_STAGE_REPO_SCRIPT = r"""import hashlib,json,os,pathlib,shutil,stat,sys,tarfile,time,uuid
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
if not verify_tree(root): raise SystemExit('installed repo verification failed')"""

_STAGED_CATANATRON_ORIGIN_SCRIPT = r"""import json,pathlib,sys
from catan_zero.rl._catanatron import import_catanatron_module
root=pathlib.Path(sys.argv[1]).resolve(strict=True)
expected=(root/'vendor/catanatron/catanatron/catanatron').resolve(strict=True)
modules={name:import_catanatron_module(name) for name in ('catanatron','catanatron.models.map','catanatron.models.board','catanatron.models.enums','catanatron.models.player')}
origins={name:str(pathlib.Path(module.__file__).resolve(strict=True)) for name,module in modules.items()}
outside={name:path for name,path in origins.items() if not pathlib.Path(path).is_relative_to(expected)}
if outside: raise SystemExit(f'vendored Catanatron import escaped sealed repo: {outside!r}')
print(json.dumps({'vendor_root':str(expected),'origins':origins},sort_keys=True))"""


def _staged_catanatron_origin_command(*, python: str, repo_dir: str) -> str:
    """Render an isolated import probe for the sealed pure-Python engine."""

    environment = {
        **contract.SEALED_RUNTIME_ENVIRONMENT,
        "PYTHONPATH": f"{repo_dir}/src:{repo_dir}",
    }
    invocation = (
        "/usr/bin/env",
        "-i",
        *(f"{key}={value}" for key, value in sorted(environment.items())),
        python,
        "-c",
        _STAGED_CATANATRON_ORIGIN_SCRIPT,
        repo_dir,
    )
    return " ".join(shlex.quote(value) for value in invocation)


def _repo_stage_inputs(
    hosts: Mapping[str, Any],
    repo_tar: Path,
    repo_sha: str,
    artifacts: Sequence[Mapping[str, Any]],
    temporary_path: Path,
) -> tuple[list[tuple[Path, str, str]], str, str, str]:
    """Materialize the two immutable inputs consumed by the repo stage."""

    manifest = {
        "schema_version": "a1-production-repo-v1",
        "repo_tar_sha256": repo_sha,
        "artifacts": [
            {key: record[key] for key in ("path", "sha256", "mode")}
            for record in artifacts
        ],
    }
    manifest["manifest_sha256"] = _digest(manifest)
    local_manifest = temporary_path / "repo-manifest.json"
    local_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    token = repo_sha.removeprefix("sha256:")
    remote_tar = f"{hosts['remote_root']}/operator/repo-{token}.tar"
    remote_manifest = f"{hosts['remote_root']}/operator/repo-{token}.json"
    receipt_path = f"{hosts['remote_root']}/receipts/repo-stage-{token}.json"
    return (
        [
            (repo_tar, remote_tar, repo_sha),
            (local_manifest, remote_manifest, _sha256(local_manifest)),
        ],
        remote_tar,
        remote_manifest,
        receipt_path,
    )


def _stage_repo(
    hosts: dict[str, Any],
    alias: str,
    artifacts: Sequence[Mapping[str, Any]],
    repo_dir: str,
    *,
    remote_tar: str,
    remote_manifest: str,
    receipt_path: str,
) -> None:
    """Activate already-installed repo inputs and bind the exact repo tree."""

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
        raise ExecutorError(
            f"immutable repo stage failed on {alias}: {result.stderr.strip()}"
        )
    artifact_paths = {str(record["path"]) for record in artifacts}
    if VENDORED_CATANATRON_SENTINEL in artifact_paths:
        origin = _ssh(
            hosts,
            alias,
            _staged_catanatron_origin_command(
                python=hosts["python"], repo_dir=repo_dir
            ),
        )
        if origin.returncode != 0:
            detail = (origin.stderr or origin.stdout).strip()
            raise ExecutorError(
                f"sealed Catanatron import-origin probe failed on {alias}: {detail}"
            )


def _lane_payload(
    worker_id: str,
    lane: list[dict[str, Any]],
    *,
    hosts: dict[str, Any],
    operator_manifests: Mapping[str, Mapping[str, str]],
    repo_dir: str,
    category_order: Sequence[str] = CATEGORY_ORDER,
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
        "category_order": list(category_order),
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
        raise ExecutorError(
            "executor receipt exists; pass --resume for exact incomplete jobs"
        )
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


def execute(
    plan: dict[str, Any], *, receipt_path: Path, resume: bool
) -> dict[str, Any]:
    _verify_plan_digest(plan)
    repo_source_root = _execution_repo_root(plan)
    private = plan["_private"]
    hosts = private["hosts"]
    lanes = private["lanes"]
    rendered = private["rendered"]
    public = _public(plan)
    expected_by_alias: dict[str, list[int]] = {}
    for lane in private["lanes"].values():
        expected_by_alias.setdefault(lane[0]["host_alias"], []).append(
            int(lane[0]["gpu"])
        )
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
            artifacts,
            _repo_files(artifacts, repo_root=repo_source_root),
            repo_tar,
        )
        repo_token = public["repo_artifacts_sha256"].removeprefix("sha256:")
        repo_dir = f"{hosts['remote_root']}/repo-{repo_token}"
        aliases = sorted({lane[0]["host_alias"] for lane in lanes.values()})
        required = rendered["required_artifacts"]
        stage_files_by_alias = _stage_files_by_alias(required, lanes)
        if (
            _sha256(Path(required["seed_ledger"]["path"]))
            != public["live_seed_ledger_sha256"]
        ):
            raise ExecutorError("live seed ledger changed after dry-run plan binding")
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
        repo_inputs, remote_repo_tar, remote_repo_manifest, repo_receipt_path = (
            _repo_stage_inputs(
                hosts,
                repo_tar,
                repo_sha,
                artifacts,
                temporary_path,
            )
        )
        lane_remote_paths: dict[str, str] = {}
        lane_inputs_by_alias: dict[str, list[tuple[Path, str, str]]] = {
            alias: [] for alias in aliases
        }
        for worker_id, lane in sorted(lanes.items()):
            payload = _lane_payload(
                worker_id,
                lane,
                hosts=hosts,
                operator_manifests=operator_manifests,
                repo_dir=repo_dir,
                category_order=tuple(public["category_order"]),
            )
            local_lane = temporary_path / f"{worker_id}.json"
            local_lane.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            remote_lane = f"{hosts['remote_root']}/lanes/{worker_id}.json"
            lane_remote_paths[worker_id] = remote_lane
            lane_inputs_by_alias[str(lane[0]["host_alias"])].append(
                (local_lane, remote_lane, _sha256(local_lane))
            )

        def stage_alias(alias: str) -> None:
            """Install one host independently; no mutable state is shared."""

            immutable_inputs = [
                *repo_inputs,
                *[
                    (
                        source,
                        operator_manifests[name]["path"],
                        operator_manifests[name]["sha256"],
                    )
                    for name, source in sorted(operator_sources.items())
                ],
                *stage_files_by_alias[alias],
                *lane_inputs_by_alias[alias],
            ]
            _remote_bulk_install(
                hosts,
                alias,
                immutable_inputs,
                temporary_path,
            )
            _stage_repo(
                hosts,
                alias,
                artifacts,
                repo_dir,
                remote_tar=remote_repo_tar,
                remote_manifest=remote_repo_manifest,
                receipt_path=repo_receipt_path,
            )
            _remote_sync_append_only_ledger(
                hosts,
                alias,
                Path(required["seed_ledger"]["path"]),
                required["seed_ledger"]["path"],
                public["live_seed_ledger_sha256"],
            )

        # Every destination and receipt is host-local and every local bundle
        # name contains the alias.  Serial staging made a 12-host launch wait
        # for twelve identical checkpoint transfers in sequence, so stage the
        # independent hosts concurrently and retain deterministic alias-order
        # error propagation through ``map``.
        with ThreadPoolExecutor(
            max_workers=min(MAX_PARALLEL_STAGE_HOSTS, len(aliases)),
            thread_name_prefix="a1-stage",
        ) as pool:
            list(pool.map(stage_alias, aliases))

        lane_pids: dict[str, int] = dict(receipt.get("lane_pids", {}))
        for worker_id, lane in sorted(lanes.items()):
            alias = lane[0]["host_alias"]
            remote_lane = lane_remote_paths[worker_id]
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
            if (
                result.returncode != 0
                or not result.stdout.strip().splitlines()[-1].isdigit()
            ):
                raise ExecutorError(
                    f"detached supervisor launch failed for {worker_id}"
                )
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
                for value in (
                    hosts["python"],
                    supervisor,
                    "status",
                    "--lane",
                    remote_lane,
                )
            )
            command = f"kill -0 {int(lane_pids[worker_id])} && {status_command}"
            response = _ssh(hosts, alias, command)
            try:
                acknowledgement = (
                    json.loads(response.stdout) if response.returncode == 0 else None
                )
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
                raise ExecutorError(
                    f"supervisor acknowledgement failed for {worker_id}"
                )
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
        response = _ssh(hosts, alias, command, timeout_seconds=STOP_SSH_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as error:
        raise ExecutorError(
            f"A1 {action} timed out for {worker_id} after {STOP_SSH_TIMEOUT_SECONDS:g}s"
        ) from error
    if response.returncode != 0:
        detail = (response.stderr or response.stdout).strip()
        raise ExecutorError(f"A1 {action} refused for {worker_id}: {detail}")
    try:
        result = json.loads(response.stdout)
    except json.JSONDecodeError as error:
        raise ExecutorError(
            f"A1 {action} returned invalid JSON for {worker_id}"
        ) from error
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
            "gpu_runtime_preserved": True,
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
            "gpu_runtime_preserved": True,
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
            results.append(
                {
                    "worker_id": worker_id,
                    "host_alias": alias,
                    "status": "unreachable_or_invalid",
                    "error": response.stderr.strip(),
                }
            )
            continue
        try:
            results.append(json.loads(response.stdout))
        except json.JSONDecodeError:
            results.append(
                {
                    "worker_id": worker_id,
                    "host_alias": alias,
                    "status": "invalid_status_output",
                }
            )
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


def wait_for_completion(
    plan: dict[str, Any],
    *,
    receipt_path: Path,
    poll_seconds: float = WAIT_POLL_SECONDS,
) -> dict[str, Any]:
    """Wait for every exact detached lane job and terminalize the receipt."""

    expected_jobs = sum(
        len(commands) for commands in plan["_private"]["lanes"].values()
    )
    if expected_jobs < 1:
        raise ExecutorError("wait requires at least one sealed generation job")
    while True:
        snapshot = status(plan, receipt_path=receipt_path)
        if snapshot.get("executor_status") != "launched":
            raise ExecutorError(
                "wait requires the exact executor receipt in launched state"
            )
        invalid_lanes = [
            lane
            for lane in snapshot.get("lanes", [])
            if lane.get("status") in {"unreachable_or_invalid", "invalid_status_output"}
        ]
        dead_running = [
            (lane.get("worker_id"), job.get("job_id"))
            for lane in snapshot.get("lanes", [])
            for job in lane.get("jobs", [])
            if job.get("status") == "running" and job.get("pid_alive") is False
        ]
        counts = snapshot.get("job_status_counts")
        if not isinstance(counts, dict):
            raise ExecutorError("generation status omitted job status counts")
        unknown = sorted(
            str(key)
            for key in counts
            if key not in {"pending", "prepared", "running", "complete", "failed"}
        )
        failed = int(counts.get("failed", 0))
        if invalid_lanes or dead_running or unknown or failed:
            receipt = _load(receipt_path)
            receipt.update(
                {
                    "status": "generation_failed",
                    "terminal_status": snapshot,
                    "failed_at": time.time(),
                }
            )
            _atomic_json(receipt_path, receipt)
            raise ExecutorError(
                "generation did not reach terminal success: "
                f"failed={failed} unknown={unknown} "
                f"invalid_lanes={len(invalid_lanes)} "
                f"dead_running={len(dead_running)}"
            )
        observed_jobs = sum(int(value) for value in counts.values())
        complete = int(counts.get("complete", 0))
        if observed_jobs != expected_jobs:
            raise ExecutorError(
                "generation status job count drift: "
                f"expected={expected_jobs} actual={observed_jobs}"
            )
        if complete == expected_jobs:
            receipt = _load(receipt_path)
            if receipt.get("plan_sha256") != plan["plan_sha256"]:
                raise ExecutorError(
                    "completion receipt binds a different execution plan"
                )
            receipt.update(
                {
                    "status": "complete",
                    "completed_at": time.time(),
                    "terminal_status": snapshot,
                }
            )
            _atomic_json(receipt_path, receipt)
            return receipt
        time.sleep(poll_seconds)


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
    run.add_argument(
        "--go", action="store_true", help="stage and launch; default dry-run"
    )
    run.add_argument(
        "--wait",
        action="store_true",
        help="wait for every detached lane job and terminalize the receipt",
    )
    sub.choices["stop"].add_argument(
        "--go",
        action="store_true",
        help="stop exact A1 process groups; default dry-run",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plan = build_plan(
            lock_path=args.lock,
            render_path=args.render,
            hosts_path=args.hosts,
            receipt_path=args.receipt,
        )
        if args.command == "status":
            result = status(plan, receipt_path=args.receipt)
        elif args.command == "stop":
            result = stop_execution(plan, receipt_path=args.receipt, go=bool(args.go))
        elif not args.go:
            if args.wait:
                raise ExecutorError("--wait requires --go")
            result = _public(plan)
        else:
            result = execute(plan, receipt_path=args.receipt, resume=bool(args.resume))
            if args.wait:
                result = wait_for_completion(plan, receipt_path=args.receipt)
    except ExecutorError as error:
        print(f"REFUSING: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
