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
import uuid
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import a1_pre_wave_contract as contract  # noqa: E402

SCHEMA = "a1-fleet-harvest-relocation-v1"
RECEIPT_SCHEMA = "a1-fleet-harvest-receipt-v1"
STATE_SCHEMA = "a1-fleet-harvest-job-state-v1"
EXPECTED_JOBS = 120
EXPECTED_HOSTS = 8
_HOST_RE = re.compile(r"[A-Za-z0-9_.@-]+\Z")


class HarvestError(RuntimeError):
    """A collection invariant failed; no final tree was published."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _value_sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
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
    if len(jobs) != EXPECTED_JOBS:
        raise HarvestError(f"expected {EXPECTED_JOBS} sealed jobs, got {len(jobs)}")
    commands = {str(item.get("job_id", "")): item for item in rendered["commands"]}
    if len(commands) != EXPECTED_JOBS:
        raise HarvestError("render contains duplicate job identities")
    seen_outputs: set[str] = set()
    hosts: set[str] = set()
    for job in jobs:
        job_id = str(job["job_id"])
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
    if len(hosts) != EXPECTED_HOSTS:
        raise HarvestError(f"expected {EXPECTED_HOSTS} immutable hosts, got {len(hosts)}")
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
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _verify_state(
    state_path: Path, job_dir: Path, job: Mapping[str, Any], lock: Mapping[str, Any]
) -> list[dict[str, Any]] | None:
    if not state_path.is_file() or not job_dir.is_dir():
        return None
    state = _load_json(state_path)
    inventory = _inventory_job(job_dir, job, lock)
    expected = {
        "schema_version": STATE_SCHEMA,
        "job_id": job["job_id"],
        "host_alias": job["host_alias"],
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
    ):
        raise HarvestError("published harvest receipt drift")
    receipt_unhashed = dict(receipt)
    receipt_digest = receipt_unhashed.pop("receipt_sha256", None)
    if receipt_digest != _value_sha256(receipt_unhashed):
        raise HarvestError("published receipt digest drift")
    if (
        relocation.get("render_path") != str(render_path)
        or relocation.get("render_file_sha256") != contract._sha256(render_path)
        or relocation.get("render_sha256") != rendered["render_sha256"]
    ):
        raise HarvestError("published harvest immutable render-file identity drift")
    try:
        contract._load_harvest_relocation(
            destination / "relocation_map.json", lock=lock, lock_path=lock_path
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


def _atomic_publish_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename a directory while refusing every existing target."""

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


def harvest(
    lock_path: Path,
    render_path: Path,
    destination: Path,
    *,
    ssh_command: Sequence[str] = ("ssh",),
) -> dict[str, Any]:
    """Collect and atomically publish the exact sealed 120-job output set."""

    lock_path = lock_path.expanduser().resolve(strict=True)
    render_path = render_path.expanduser().resolve(strict=True)
    destination = destination.expanduser().absolute()
    try:
        destination_parent = destination.parent.resolve(strict=True)
    except OSError as error:
        raise HarvestError(f"harvest destination parent is invalid: {error}") from error
    if destination_parent != destination.parent.absolute():
        raise HarvestError("harvest destination parent must not traverse a symlink")
    lock, rendered, jobs = _validate_inputs(lock_path, render_path)
    transaction_key = _value_sha256(
        {
            "contract_sha256": lock["contract_sha256"],
            "render_sha256": rendered["render_sha256"],
            "destination": str(destination),
        }
    ).removeprefix("sha256:")[:20]
    if destination.exists():
        return _validate_published(
            destination,
            lock=lock,
            rendered=rendered,
            lock_path=lock_path,
            render_path=render_path,
        )
    stage = destination.parent / f".{destination.name}.harvest-{transaction_key}.staging"
    payload = stage / "payload"
    jobs_root = payload / "jobs"
    state_root = stage / "state"
    incoming_root = stage / "incoming"
    if stage.exists() and (stage.is_symlink() or not stage.is_dir()):
        raise HarvestError(f"unsafe pre-existing harvest staging path {stage}")
    for path in (stage, payload, jobs_root, state_root, incoming_root):
        path.mkdir(parents=True, exist_ok=True)
        if path.is_symlink() or not path.is_dir() or path.resolve() != path.absolute():
            raise HarvestError(f"unsafe harvest staging directory {path}")

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
    files = [record for job in jobs for record in inventories[str(job["job_id"])] ]
    source_paths = [record["source_path"] for record in files]
    if len(source_paths) != len(set(source_paths)):
        raise HarvestError("harvest contains duplicate immutable source paths")
    job_identities = [
        {
            key: job[key]
            for key in ("job_id", "worker_id", "host_alias", "gpu", "category", "output_dir")
        }
        for job in jobs
    ]
    relocation = {
        "schema_version": SCHEMA,
        "contract_path": str(lock_path),
        "contract_file_sha256": contract._sha256(lock_path),
        "contract_sha256": lock["contract_sha256"],
        "render_path": str(render_path),
        "render_file_sha256": contract._sha256(render_path),
        "render_sha256": rendered["render_sha256"],
        "host_count": len(grouped),
        "job_count": len(jobs),
        "job_identities": job_identities,
        "job_identities_sha256": _value_sha256(job_identities),
        "files": files,
        "file_inventory_sha256": _value_sha256(files),
    }
    relocation["relocation_sha256"] = _value_sha256(relocation)
    _write_exclusive_json(payload / "relocation_map.json", relocation)
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "contract_sha256": lock["contract_sha256"],
        "render_sha256": rendered["render_sha256"],
        "relocation_sha256": relocation["relocation_sha256"],
        "job_count": len(jobs),
        "host_count": len(grouped),
        "file_count": len(files),
        "file_inventory_sha256": relocation["file_inventory_sha256"],
    }
    receipt["receipt_sha256"] = _value_sha256(receipt)
    _write_exclusive_json(payload / "harvest_receipt.json", receipt)
    directory_fd = os.open(payload, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    _atomic_publish_noreplace(payload, destination)
    parent_fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    shutil.rmtree(stage, ignore_errors=True)
    return _validate_published(
        destination,
        lock=lock,
        rendered=rendered,
        lock_path=lock_path,
        render_path=render_path,
    )


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
