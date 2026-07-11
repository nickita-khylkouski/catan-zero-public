#!/usr/bin/env python3
"""Fail-closed, resumable harvest of the sealed A1 generation fleet.

The tool is intentionally only a collector.  It never launches, stops, or
otherwise mutates a fleet job.  Eight read-only tar streams are copied into a
private staging directory, inspected member-by-member, hashed, and then
published with one atomic rename.  The typed relocation map preserves the
remote absolute name of every byte while giving post-wave consumers a safe
local name.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import a1_pre_wave_contract as contract  # noqa: E402

SCHEMA = "a1-fleet-harvest-relocation-v1"
RECEIPT_SCHEMA = "a1-fleet-harvest-receipt-v1"
STATE_SCHEMA = "a1-fleet-harvest-job-state-v1"
EXPECTED_JOBS = 120
EXPECTED_HOSTS = 8
DUAL_ARM_PROFILE = "dual_arm_generation_v1"
_HOST_RE = re.compile(r"[A-Za-z0-9_.@-]+\Z")


class HarvestError(RuntimeError):
    """A collection invariant failed; no final tree was published."""


def _contract_shape(lock: Mapping[str, Any]) -> dict[str, Any]:
    game_contract = lock.get("game_contract")
    if not isinstance(game_contract, dict) or game_contract.get("profile") != DUAL_ARM_PROFILE:
        return {
            "arm_id": None,
            "job_count": EXPECTED_JOBS,
            "host_count": EXPECTED_HOSTS,
            "category_games": dict(contract.EXPECTED_GAMES),
            "category_attempts": dict(contract.EXPECTED_ATTEMPTS),
        }
    arm_id = game_contract.get("arm_id")
    category_games = game_contract.get("category_games")
    category_attempts = game_contract.get("category_attempts")
    if (
        arm_id not in {"n128", "n256"}
        or not isinstance(category_games, dict)
        or not isinstance(category_attempts, dict)
        or set(category_games) != set(contract.EXPECTED_GAMES)
        or set(category_attempts) != set(contract.EXPECTED_GAMES)
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in category_games.values())
        or any(isinstance(value, bool) or not isinstance(value, int) or value < category_games[key] for key, value in category_attempts.items())
        or game_contract.get("total_complete_games") != sum(category_games.values())
        or game_contract.get("total_attempts") != sum(category_attempts.values())
    ):
        raise HarvestError("dual-arm game_contract quotas are malformed")
    return {
        "arm_id": arm_id,
        "job_count": 84,
        "host_count": None,
        "category_games": dict(category_games),
        "category_attempts": dict(category_attempts),
    }


@dataclass
class _PinnedInput:
    path: Path
    descriptor: int
    identity: tuple[int, int, int, int, int]
    data: bytes
    sha256: str

    @classmethod
    def open(cls, path: Path) -> "_PinnedInput":
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise HarvestError(f"immutable input is not a regular file: {path}")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1 << 20)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(descriptor)
            identity = _stat_identity(before)
            if identity != _stat_identity(after):
                raise HarvestError(f"immutable input changed while pinning: {path}")
            data = b"".join(chunks)
            if len(data) != before.st_size:
                raise HarvestError(f"immutable input was partially read: {path}")
            return cls(
                path=path,
                descriptor=descriptor,
                identity=identity,
                data=data,
                sha256="sha256:" + hashlib.sha256(data).hexdigest(),
            )
        except BaseException:
            os.close(descriptor)
            raise

    def revalidate(self) -> None:
        if _stat_identity(os.fstat(self.descriptor)) != self.identity:
            raise HarvestError(f"pinned immutable input descriptor drifted: {self.path}")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            current = os.open(self.path, flags)
        except OSError as error:
            raise HarvestError(f"pinned immutable input path drifted: {self.path}: {error}") from error
        try:
            if _stat_identity(os.fstat(current)) != self.identity:
                raise HarvestError(f"pinned immutable input inode drifted: {self.path}")
        finally:
            os.close(current)
        if _sha256_descriptor(self.descriptor) != self.sha256:
            raise HarvestError(f"pinned immutable input bytes drifted: {self.path}")

    def close(self) -> None:
        os.close(self.descriptor)


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _sha256_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while True:
        chunk = os.pread(descriptor, 1 << 20, offset)
        if not chunk:
            break
        digest.update(chunk)
        offset += len(chunk)
    return "sha256:" + digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _value_sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise HarvestError(f"cannot hash non-regular file {path}")
        digest = _sha256_descriptor(descriptor)
        if _stat_identity(before) != _stat_identity(os.fstat(descriptor)):
            raise HarvestError(f"file changed during stable hash: {path}")
        return digest
    finally:
        os.close(descriptor)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise HarvestError(f"JSON artifact is not regular: {path}")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1 << 20)
                if not chunk:
                    break
                chunks.append(chunk)
            if _stat_identity(before) != _stat_identity(os.fstat(descriptor)):
                raise HarvestError(f"JSON artifact changed while reading: {path}")
        finally:
            os.close(descriptor)
        value = json.loads(b"".join(chunks).decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise HarvestError(f"cannot load JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise HarvestError(f"JSON artifact is not an object: {path}")
    return value


def _normal_remote_path(raw: str, *, where: str) -> PurePosixPath:
    path = PurePosixPath(raw)
    if not path.is_absolute() or raw != str(path) or ".." in path.parts:
        raise HarvestError(f"{where} is not a normalized absolute path: {raw!r}")
    if path.name in {"", ".", ".."}:
        raise HarvestError(f"{where} has no safe basename: {raw!r}")
    return path


def _validate_inputs(
    lock_path: Path, render_path: Path
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    lock = contract.verify_lock(lock_path)
    rendered, _rows = contract._validate_claim_render(lock, render_path)
    jobs = list(lock["fleet"]["jobs"])
    shape = _contract_shape(lock)
    if len(jobs) != shape["job_count"]:
        raise HarvestError(f"expected {shape['job_count']} sealed jobs, got {len(jobs)}")
    commands = {str(item.get("job_id", "")): item for item in rendered["commands"]}
    if len(commands) != shape["job_count"]:
        raise HarvestError("render contains duplicate job identities")
    seen_outputs: set[str] = set()
    hosts: set[str] = set()
    for job in jobs:
        job_id = str(job["job_id"])
        if shape["arm_id"] is not None and job.get("arm_id") != shape["arm_id"]:
            raise HarvestError(f"{job_id}: job arm_id differs from arm lock")
        host = str(job["host_alias"])
        if not _HOST_RE.fullmatch(host):
            raise HarvestError(f"unsafe host alias for {job_id}: {host!r}")
        hosts.add(host)
        output = _normal_remote_path(str(job["output_dir"]), where=f"{job_id}.output_dir")
        if output.name != job_id:
            raise HarvestError(f"{job_id}: output basename must equal immutable job id")
        if str(output) in seen_outputs:
            raise HarvestError(f"duplicate remote output directory {output}")
        seen_outputs.add(str(output))
        command = commands.get(job_id)
        if not isinstance(command, dict):
            raise HarvestError(f"render is missing job {job_id}")
        expected_identity = {
            "job_id": job_id,
            "worker_id": job["worker_id"],
            "host_alias": host,
            "gpu": job["gpu"],
            "category": job["category"],
            **({} if shape["arm_id"] is None else {"arm_id": shape["arm_id"]}),
        }
        if {key: command.get(key) for key in expected_identity} != expected_identity:
            raise HarvestError(f"render/lock job identity drift for {job_id}")
        output_attestation = command.get("output_attestation")
        if not isinstance(output_attestation, dict) or output_attestation.get(
            "destination"
        ) != str(output / "a1_contract.json"):
            raise HarvestError(f"rendered output destination drift for {job_id}")
        if output_attestation.get("payload_sha256") != contract._digest_value(
            contract._job_attestation(lock, job)
        ):
            raise HarvestError(f"rendered attestation identity drift for {job_id}")
    if shape["arm_id"] is not None:
        games = defaultdict(int)
        attempts = defaultdict(int)
        for job in jobs:
            games[str(job["category"])] += int(job["games"])
            attempts[str(job["category"])] += int(job["attempts"])
        if dict(games) != shape["category_games"] or dict(attempts) != shape["category_attempts"]:
            raise HarvestError("dual-arm jobs do not equal sealed category quotas")
    if shape["host_count"] is not None and len(hosts) != shape["host_count"]:
        raise HarvestError(f"expected {shape['host_count']} immutable hosts, got {len(hosts)}")
    if shape["arm_id"] is not None and not hosts:
        raise HarvestError("dual-arm harvest has no immutable hosts")
    return lock, rendered, jobs


def _member_relative(member_name: str, expected_roots: set[str]) -> Path:
    pure = PurePosixPath(member_name)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise HarvestError(f"unsafe archive member path {member_name!r}")
    parts = tuple(part for part in pure.parts if part not in {"", "."})
    if not parts or parts[0] not in expected_roots:
        raise HarvestError(f"archive member escapes expected jobs: {member_name!r}")
    normalized = PurePosixPath(*parts)
    if member_name != str(normalized):
        raise HarvestError(f"non-canonical archive member path {member_name!r}")
    return Path(*parts)


def _extract_archive(
    archive: Path, target: Path, *, expected_roots: set[str]
) -> None:
    target.mkdir(parents=True, exist_ok=False)
    seen: set[Path] = set()
    roots_seen: set[str] = set()
    try:
        file_count = 0
        total_size = 0
        with tarfile.open(archive, mode="r:*") as bundle:
            for member in bundle:
                relative = _member_relative(member.name, expected_roots)
                if relative in seen:
                    raise HarvestError(f"duplicate archive member {member.name!r}")
                seen.add(relative)
                roots_seen.add(relative.parts[0])
                destination = target / relative
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isreg():
                    raise HarvestError(
                        f"archive member is not a regular file/directory: {member.name!r}"
                    )
                file_count += 1
                total_size += int(member.size)
                if file_count > 1_000_000 or member.size > (1 << 40) or total_size > (1 << 44):
                    raise HarvestError("archive exceeds fail-closed member/byte safety limits")
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    raise HarvestError(f"archive member collides at {member.name!r}")
                source = bundle.extractfile(member)
                if source is None:
                    raise HarvestError(f"cannot read archive member {member.name!r}")
                descriptor = os.open(
                    destination,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                    0o444,
                )
                written = 0
                try:
                    with os.fdopen(descriptor, "wb") as handle:
                        while True:
                            chunk = source.read(1 << 20)
                            if not chunk:
                                break
                            handle.write(chunk)
                            written += len(chunk)
                        handle.flush()
                        os.fsync(handle.fileno())
                except BaseException:
                    destination.unlink(missing_ok=True)
                    raise
                if written != int(member.size):
                    raise HarvestError(
                        f"partial archive member {member.name!r}: {written}/{member.size} bytes"
                    )
        if roots_seen != expected_roots:
            raise HarvestError(
                f"archive job roots drift: missing={sorted(expected_roots - roots_seen)} "
                f"unexpected={sorted(roots_seen - expected_roots)}"
            )
    except BaseException:
        shutil.rmtree(target, ignore_errors=True)
        raise


def _inventory_job(
    job_dir: Path, job: Mapping[str, Any], lock: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if job_dir.is_symlink() or not job_dir.is_dir():
        raise HarvestError(f"missing regular harvested job directory {job_dir}")
    records: list[dict[str, Any]] = []
    for path in sorted(job_dir.rglob("*"), key=lambda value: value.as_posix()):
        if path.is_symlink():
            raise HarvestError(f"symlink appeared in harvested tree: {path}")
        if path.is_dir():
            continue
        mode = path.lstat().st_mode
        if not stat.S_ISREG(mode):
            raise HarvestError(f"non-regular file appeared in harvested tree: {path}")
        relative = path.relative_to(job_dir)
        source = PurePosixPath(str(job["output_dir"])) / PurePosixPath(relative.as_posix())
        records.append(
            {
                "source_path": str(source),
                "relative_path": str(Path("jobs") / str(job["job_id"]) / relative),
                "size_bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
                "job_id": job["job_id"],
                "host_alias": job["host_alias"],
            }
        )
    required = {"a1_contract.json", "manifest.json", contract.CONFIG_REGISTRY_FILENAME}
    relative_names = {
        str(PurePosixPath(item["source_path"]).relative_to(str(job["output_dir"])))
        for item in records
    }
    missing = required - relative_names
    if missing:
        raise HarvestError(f"{job['job_id']}: missing required outputs {sorted(missing)}")
    attestation = _load_json(job_dir / "a1_contract.json")
    if attestation != contract._job_attestation(dict(lock), dict(job)):
        raise HarvestError(f"{job['job_id']}: immutable output attestation drift")
    manifest = _load_json(job_dir / "manifest.json")
    if int(manifest.get("base_seed", -1)) != int(job["base_seed"]):
        raise HarvestError(f"{job['job_id']}: manifest base_seed drift")
    arm_id = dict(lock.get("game_contract") or {}).get("arm_id")
    if arm_id is not None and manifest.get("arm_id") != arm_id:
        raise HarvestError(f"{job['job_id']}: manifest arm_id drift")
    source_paths = {item["source_path"] for item in records}
    manifest_source = PurePosixPath(str(job["output_dir"])) / "manifest.json"
    for label, values in (
        ("shard", manifest.get("shards", [])),
        ("worker manifest", manifest.get("worker_summaries", [])),
    ):
        if not isinstance(values, list):
            raise HarvestError(f"{job['job_id']}: manifest {label} list is malformed")
        for raw in values:
            raw_path = PurePosixPath(str(raw))
            if str(raw_path) != str(raw):
                raise HarvestError(
                    f"{job['job_id']}: {label} reference is non-canonical: {raw!r}"
                )
            source = raw_path if raw_path.is_absolute() else manifest_source.parent / raw_path
            if ".." in source.parts or str(source) not in source_paths:
                raise HarvestError(
                    f"{job['job_id']}: {label} reference is absent or unsafe: {raw!r}"
                )
    return records


def _write_exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o444,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        parent = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _install_or_verify_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        if path.is_symlink() or _load_json(path) != payload:
            raise HarvestError(f"existing final metadata drifted: {path}")
        return
    _write_exclusive_json(path, payload)


def _verify_state(
    state_path: Path, job_dir: Path, job: Mapping[str, Any], lock: Mapping[str, Any]
) -> list[dict[str, Any]] | None:
    state_exists = state_path.is_file()
    job_exists = job_dir.is_dir() and not job_dir.is_symlink()
    if state_exists and not job_exists:
        raise HarvestError(f"resume state has no staged bytes for {job['job_id']}")
    if job_exists and not state_exists:
        # os.replace publishes a complete directory atomically. A crash in the
        # tiny interval before its state receipt is therefore safely replayable
        # by re-inventorying those already-local bytes.
        inventory = _inventory_job(job_dir, job, lock)
        recovered = {
            "schema_version": STATE_SCHEMA,
            "job_id": job["job_id"],
            "host_alias": job["host_alias"],
            **({} if job.get("arm_id") is None else {"arm_id": job["arm_id"]}),
            "inventory": inventory,
            "inventory_sha256": _value_sha256(inventory),
        }
        _write_exclusive_json(state_path, recovered)
        return inventory
    if not state_exists:
        return None
    state = _load_json(state_path)
    inventory = _inventory_job(job_dir, job, lock)
    expected = {
        "schema_version": STATE_SCHEMA,
        "job_id": job["job_id"],
        "host_alias": job["host_alias"],
        **({} if job.get("arm_id") is None else {"arm_id": job["arm_id"]}),
        "inventory": inventory,
        "inventory_sha256": _value_sha256(inventory),
    }
    if state != expected:
        raise HarvestError(f"resume state drift for {job['job_id']}")
    return inventory


def _ssh_fetch(
    ssh_command: Sequence[str], host: str, outputs: Sequence[PurePosixPath], archive: Path
) -> None:
    parents = {str(path.parent) for path in outputs}
    if len(parents) != 1:
        raise HarvestError(f"host {host} output directories do not share one parent")
    remote = " ".join(
        [
            "tar --format=pax --numeric-owner -C",
            shlex.quote(next(iter(parents))),
            "-cf - --",
            *(shlex.quote(path.name) for path in outputs),
        ]
    )
    stderr_path = archive.with_suffix(".stderr")
    with archive.open("xb") as stdout, stderr_path.open("xb") as stderr:
        result = subprocess.run(
            [*ssh_command, host, remote], stdout=stdout, stderr=stderr, check=False
        )
    if result.returncode != 0:
        detail = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        archive.unlink(missing_ok=True)
        raise HarvestError(f"read-only fetch failed on {host}: {detail.strip()}")
    stderr_path.unlink(missing_ok=True)


def _validate_published(
    destination: Path,
    *,
    lock: dict[str, Any],
    rendered: dict[str, Any],
    lock_path: Path,
    render_path: Path,
    lock_pin: _PinnedInput,
    render_pin: _PinnedInput,
) -> dict[str, Any]:
    if destination.is_symlink() or destination.resolve(strict=True) != destination.absolute():
        raise HarvestError("published harvest root is a symlink")
    receipt = _load_json(destination / "harvest_receipt.json")
    relocation = _load_json(destination / "relocation_map.json")
    unhashed = dict(relocation)
    declared = unhashed.pop("relocation_sha256", None)
    if declared != _value_sha256(unhashed):
        raise HarvestError("published relocation-map digest drift")
    expected_receipt_fields = {
        "schema_version",
        "contract_sha256",
        "render_sha256",
        "relocation_sha256",
        "job_count",
        "host_count",
        "file_count",
        "file_inventory_sha256",
        "receipt_sha256",
    }
    shape = _contract_shape(lock)
    if shape["arm_id"] is not None:
        expected_receipt_fields.add("arm_id")
    if set(receipt) != expected_receipt_fields or (
        receipt.get("schema_version") != RECEIPT_SCHEMA
        or receipt.get("contract_sha256") != lock["contract_sha256"]
        or receipt.get("render_sha256") != rendered["render_sha256"]
        or receipt.get("relocation_sha256") != declared
        or receipt.get("job_count") != relocation.get("job_count")
        or receipt.get("host_count") != relocation.get("host_count")
        or receipt.get("file_count") != len(relocation.get("files", []))
        or receipt.get("file_inventory_sha256")
        != relocation.get("file_inventory_sha256")
        or (shape["arm_id"] is not None and receipt.get("arm_id") != shape["arm_id"])
        or (shape["arm_id"] is not None and relocation.get("arm_id") != shape["arm_id"])
        or (shape["arm_id"] is None and "arm_id" in relocation)
    ):
        raise HarvestError("published harvest receipt drift")
    receipt_unhashed = dict(receipt)
    receipt_digest = receipt_unhashed.pop("receipt_sha256", None)
    if receipt_digest != _value_sha256(receipt_unhashed):
        raise HarvestError("published receipt digest drift")
    if (
        relocation.get("contract_path") != str(lock_path)
        or relocation.get("contract_file_sha256") != lock_pin.sha256
        or relocation.get("render_path") != str(render_path)
        or relocation.get("render_file_sha256") != render_pin.sha256
        or relocation.get("render_sha256") != rendered["render_sha256"]
    ):
        raise HarvestError("published harvest immutable input-file identity drift")
    try:
        contract._load_harvest_relocation(
            destination / "relocation_map.json", lock=lock
        )
    except contract.ContractError as error:
        raise HarvestError(str(error)) from error
    for record in relocation.get("files", []):
        path = destination / str(record["relative_path"])
        if path.is_symlink() or not path.is_file():
            raise HarvestError(f"published harvested file is missing: {path}")
        if path.stat().st_size != int(record["size_bytes"]) or _file_sha256(path) != record[
            "sha256"
        ]:
            raise HarvestError(f"published harvested bytes drifted: {path}")
    return relocation


def _atomic_publish_noreplace(
    source: Path,
    destination: Path,
    *,
    preflight: Callable[[], None] | None = None,
) -> None:
    """Atomically rename a directory while refusing every existing target."""

    if preflight is not None:
        preflight()
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform == "darwin" and hasattr(libc, "renamex_np"):
        result = libc.renamex_np(source_bytes, destination_bytes, ctypes.c_uint(0x4))
    elif hasattr(libc, "renameat2"):
        result = libc.renameat2(
            ctypes.c_int(-100),
            source_bytes,
            ctypes.c_int(-100),
            destination_bytes,
            ctypes.c_uint(1),
        )
    else:
        raise HarvestError("platform has no atomic no-replace directory publish primitive")
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise HarvestError(f"refusing to replace published harvest {destination}")
        raise OSError(error_number, os.strerror(error_number), str(destination))


def _harvest_locked(
    lock_path: Path,
    render_path: Path,
    snapshot_lock_path: Path,
    snapshot_render_path: Path,
    destination: Path,
    *,
    lock_pin: _PinnedInput,
    render_pin: _PinnedInput,
    parent_fd: int,
    transaction_guard: Callable[[], None] | None = None,
    ssh_command: Sequence[str] = ("ssh",),
) -> dict[str, Any]:
    """Collect and atomically publish the exact sealed 120-job output set."""

    lock, rendered, jobs = _validate_inputs(snapshot_lock_path, snapshot_render_path)
    if transaction_guard is not None:
        transaction_guard()
    transaction_identity = {
        "contract_sha256": lock["contract_sha256"],
        "render_sha256": rendered["render_sha256"],
        "destination": str(destination),
    }
    if _contract_shape(lock)["arm_id"] is not None:
        transaction_identity["arm_id"] = _contract_shape(lock)["arm_id"]
    transaction_key = _value_sha256(transaction_identity).removeprefix("sha256:")[:20]
    if destination.exists():
        if transaction_guard is not None:
            transaction_guard()
        lock_pin.revalidate()
        render_pin.revalidate()
        result = _validate_published(
            destination,
            lock=lock,
            rendered=rendered,
            lock_path=lock_path,
            render_path=render_path,
            lock_pin=lock_pin,
            render_pin=render_pin,
        )
        lock_pin.revalidate()
        render_pin.revalidate()
        if transaction_guard is not None:
            transaction_guard()
        return result
    stage = destination.parent / f".{destination.name}.harvest-{transaction_key}.staging"
    payload = stage / "payload"
    jobs_root = payload / "jobs"
    state_root = stage / "state"
    incoming_root = stage / "incoming"
    if not stage.exists():
        try:
            os.mkdir(stage.name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError:
            pass
    if stage.exists() and (stage.is_symlink() or not stage.is_dir()):
        raise HarvestError(f"unsafe pre-existing harvest staging path {stage}")
    for path in (payload, jobs_root, state_root, incoming_root):
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(path, 0o700)
        if path.is_symlink() or not path.is_dir() or path.resolve() != path.absolute():
            raise HarvestError(f"unsafe harvest staging directory {path}")
    staged_map = payload / "relocation_map.json"
    staged_receipt = payload / "harvest_receipt.json"
    if staged_receipt.exists() and not staged_map.exists():
        raise HarvestError("staging receipt exists without its relocation map")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for job in jobs:
        grouped[str(job["host_alias"])].append(job)
    inventories: dict[str, list[dict[str, Any]]] = {}
    for host in sorted(grouped):
        missing: list[dict[str, Any]] = []
        for job in grouped[host]:
            job_id = str(job["job_id"])
            resumed = _verify_state(
                state_root / f"{job_id}.json", jobs_root / job_id, job, lock
            )
            if resumed is None:
                missing.append(job)
            else:
                inventories[job_id] = resumed
        if not missing:
            continue
        token = uuid.uuid4().hex
        archive = incoming_root / f"{host}.{token}.tar"
        extracted = incoming_root / f"{host}.{token}"
        outputs = [
            _normal_remote_path(str(job["output_dir"]), where=str(job["job_id"]))
            for job in missing
        ]
        _ssh_fetch(ssh_command, host, outputs, archive)
        try:
            _extract_archive(archive, extracted, expected_roots={path.name for path in outputs})
            pending: dict[str, list[dict[str, Any]]] = {}
            for job in missing:
                job_id = str(job["job_id"])
                pending[job_id] = _inventory_job(extracted / job_id, job, lock)
            for job in missing:
                job_id = str(job["job_id"])
                final_job = jobs_root / job_id
                if final_job.exists():
                    raise HarvestError(f"staging collision for {job_id}")
                os.replace(extracted / job_id, final_job)
                state = {
                    "schema_version": STATE_SCHEMA,
                    "job_id": job_id,
                    "host_alias": host,
                    **({} if job.get("arm_id") is None else {"arm_id": job["arm_id"]}),
                    "inventory": pending[job_id],
                    "inventory_sha256": _value_sha256(pending[job_id]),
                }
                _write_exclusive_json(state_root / f"{job_id}.json", state)
                inventories[job_id] = pending[job_id]
        finally:
            archive.unlink(missing_ok=True)
            shutil.rmtree(extracted, ignore_errors=True)

    if set(inventories) != {str(job["job_id"]) for job in jobs}:
        raise HarvestError("staging does not cover the exact immutable job set")
    # Re-read the entire staged payload immediately before metadata creation.
    # State receipts accelerate network resume but are never final authority.
    final_inventories = {
        str(job["job_id"]): _inventory_job(
            jobs_root / str(job["job_id"]), job, lock
        )
        for job in jobs
    }
    if final_inventories != inventories:
        raise HarvestError("staged payload changed after its per-job receipts")
    files = [
        record
        for job in jobs
        for record in final_inventories[str(job["job_id"])]
    ]
    source_paths = [record["source_path"] for record in files]
    if len(source_paths) != len(set(source_paths)):
        raise HarvestError("harvest contains duplicate immutable source paths")
    identity_fields = (
        "job_id", "worker_id", "host_alias", "gpu", "category", "output_dir"
    ) + (("arm_id",) if _contract_shape(lock)["arm_id"] is not None else ())
    job_identities = [
        {
            key: job[key]
            for key in identity_fields
        }
        for job in jobs
    ]
    relocation = {
        "schema_version": SCHEMA,
        **({} if _contract_shape(lock)["arm_id"] is None else {"arm_id": _contract_shape(lock)["arm_id"]}),
        "contract_path": str(lock_path),
        "contract_file_sha256": lock_pin.sha256,
        "contract_sha256": lock["contract_sha256"],
        "render_path": str(render_path),
        "render_file_sha256": render_pin.sha256,
        "render_sha256": rendered["render_sha256"],
        "host_count": len(grouped),
        "job_count": len(jobs),
        "job_identities": job_identities,
        "job_identities_sha256": _value_sha256(job_identities),
        "files": files,
        "file_inventory_sha256": _value_sha256(files),
    }
    relocation["relocation_sha256"] = _value_sha256(relocation)
    _install_or_verify_json(staged_map, relocation)
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        **({} if _contract_shape(lock)["arm_id"] is None else {"arm_id": _contract_shape(lock)["arm_id"]}),
        "contract_sha256": lock["contract_sha256"],
        "render_sha256": rendered["render_sha256"],
        "relocation_sha256": relocation["relocation_sha256"],
        "job_count": len(jobs),
        "host_count": len(grouped),
        "file_count": len(files),
        "file_inventory_sha256": relocation["file_inventory_sha256"],
    }
    receipt["receipt_sha256"] = _value_sha256(receipt)
    _install_or_verify_json(staged_receipt, receipt)
    directory_fd = os.open(payload, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    def preflight() -> None:
        if transaction_guard is not None:
            transaction_guard()
        lock_pin.revalidate()
        render_pin.revalidate()
        actual = _validate_published(
            payload,
            lock=lock,
            rendered=rendered,
            lock_path=lock_path,
            render_path=render_path,
            lock_pin=lock_pin,
            render_pin=render_pin,
        )
        if actual != relocation:
            raise HarvestError("staged relocation changed before atomic publish")

    _atomic_publish_noreplace(payload, destination, preflight=preflight)
    publish_parent_fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(publish_parent_fd)
    finally:
        os.close(publish_parent_fd)
    shutil.rmtree(stage, ignore_errors=True)
    lock_pin.revalidate()
    render_pin.revalidate()
    result = _validate_published(
        destination,
        lock=lock,
        rendered=rendered,
        lock_path=lock_path,
        render_path=render_path,
        lock_pin=lock_pin,
        render_pin=render_pin,
    )
    lock_pin.revalidate()
    render_pin.revalidate()
    return result


def _write_snapshot(path: Path, data: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def harvest(
    lock_path: Path,
    render_path: Path,
    destination: Path,
    *,
    ssh_command: Sequence[str] = ("ssh",),
) -> dict[str, Any]:
    """Pin immutable inputs, serialize the destination, then collect/publish."""

    lock_path = lock_path.expanduser().absolute()
    render_path = render_path.expanduser().absolute()
    destination = destination.expanduser().absolute()
    for label, path in (("lock", lock_path), ("render", render_path)):
        try:
            canonical = path.resolve(strict=True)
        except OSError as error:
            raise HarvestError(f"cannot resolve immutable {label} input: {error}") from error
        if canonical != path:
            raise HarvestError(f"immutable {label} input path must not traverse symlinks")
    try:
        destination_parent = destination.parent.resolve(strict=True)
    except OSError as error:
        raise HarvestError(f"harvest destination parent is invalid: {error}") from error
    if destination_parent != destination.parent.absolute():
        raise HarvestError("harvest destination parent must not traverse a symlink")
    parent_fd = os.open(
        destination_parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    lock_name = (
        ".a1-harvest-"
        + hashlib.sha256(str(destination).encode("utf-8")).hexdigest()[:24]
        + ".lock"
    )
    while True:
        try:
            transaction_fd = os.open(
                lock_name,
                os.O_RDWR
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=parent_fd,
            )
            break
        except FileExistsError:
            try:
                transaction_fd = os.open(
                    lock_name,
                    os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=parent_fd,
                )
                break
            except FileNotFoundError:
                continue
    lock_pin: _PinnedInput | None = None
    render_pin: _PinnedInput | None = None
    snapshot_root: Path | None = None
    try:
        transaction_stat = os.fstat(transaction_fd)
        if (
            not stat.S_ISREG(transaction_stat.st_mode)
            or transaction_stat.st_uid != os.getuid()
            or transaction_stat.st_nlink != 1
        ):
            raise HarvestError("destination transaction lock is not a regular file")
        os.fchmod(transaction_fd, 0o600)
        fcntl.flock(transaction_fd, fcntl.LOCK_EX)
        os.fsync(parent_fd)

        def transaction_guard() -> None:
            held = os.fstat(transaction_fd)
            try:
                named = os.stat(lock_name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError as error:
                raise HarvestError("destination transaction lock was unlinked") from error
            if (
                not stat.S_ISREG(held.st_mode)
                or held.st_nlink != 1
                or not stat.S_ISREG(named.st_mode)
                or (named.st_dev, named.st_ino) != (held.st_dev, held.st_ino)
            ):
                raise HarvestError("destination transaction lock identity drifted")

        transaction_guard()
        lock_pin = _PinnedInput.open(lock_path)
        render_pin = _PinnedInput.open(render_path)
        lock_pin.revalidate()
        render_pin.revalidate()
        snapshot_root = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.inputs-", dir=destination_parent
            )
        )
        os.chmod(snapshot_root, 0o700)
        snapshot_lock = snapshot_root / "lock.json"
        snapshot_render = snapshot_root / "render.json"
        _write_snapshot(snapshot_lock, lock_pin.data)
        _write_snapshot(snapshot_render, render_pin.data)
        result = _harvest_locked(
            lock_path,
            render_path,
            snapshot_lock,
            snapshot_render,
            destination,
            lock_pin=lock_pin,
            render_pin=render_pin,
            parent_fd=parent_fd,
            transaction_guard=transaction_guard,
            ssh_command=ssh_command,
        )
        transaction_guard()
        return result
    finally:
        if snapshot_root is not None:
            shutil.rmtree(snapshot_root, ignore_errors=True)
        if render_pin is not None:
            render_pin.close()
        if lock_pin is not None:
            lock_pin.close()
        try:
            fcntl.flock(transaction_fd, fcntl.LOCK_UN)
        finally:
            os.close(transaction_fd)
            os.close(parent_fd)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--render", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument(
        "--ssh-command",
        default="ssh",
        help="local SSH executable (tests may provide a read-only fixture transport)",
    )
    args = parser.parse_args(argv)
    try:
        result = harvest(
            args.lock,
            args.render,
            args.destination,
            ssh_command=(args.ssh_command,),
        )
    except (HarvestError, contract.ContractError, OSError, tarfile.TarError) as error:
        parser.error(str(error))
    print(json.dumps({"status": "PASS", "relocation_sha256": result["relocation_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
