#!/usr/bin/env python3
"""Create the first authoritative A1 champion registry, once and audibly.

The project never persisted a pre-A1 ``ChampionRegistry``.  Treating a missing
path as an empty registry during promotion would silently invent history.  This
tool is the only bootstrap boundary: it verifies the sealed A1 contract, binds
the exact producer and historical scalar-training attestation, starts a new A1
promotion lineage at zero, and atomically publishes a registry, current pointer,
and immutable migration receipt.  Existing destinations are always refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools import a1_pre_wave_contract as contract_tool  # noqa: E402
from tools.champion_registry import ChampionRegistry  # noqa: E402


RECEIPT_SCHEMA = "a1-champion-registry-bootstrap-v1"


class BootstrapError(RuntimeError):
    """The initial registry cannot be established without inventing state."""


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


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - registry compatibility identifier.
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fresh(path: Path, *, where: str) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    if lexical.exists() or lexical.is_symlink():
        raise BootstrapError(f"refusing non-fresh {where}: {lexical}")
    lexical.parent.mkdir(parents=True, exist_ok=True)
    if lexical.parent.resolve(strict=True) != lexical.parent:
        raise BootstrapError(f"{where} parent must be canonical")
    return lexical


def _file_ref(path: Path, *, where: str) -> dict[str, str]:
    try:
        canonical = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise BootstrapError(f"cannot resolve {where}: {error}") from error
    if not canonical.is_file() or canonical.is_symlink():
        raise BootstrapError(f"{where} must be a regular non-symlink file")
    return {"path": str(canonical), "sha256": _sha256(canonical)}


def _producer(lock: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in lock.get("checkpoints", []) if row.get("role") == "producer"]
    if len(rows) != 1:
        raise BootstrapError("sealed contract must contain exactly one producer")
    return rows[0]


def _legacy_report_ref(producer: dict[str, Any]) -> dict[str, str]:
    try:
        attestation = producer["metadata"]["legacy_scalar_readout_attestation"]
        report = attestation["report"]
        path = Path(str(report["path"]))
        stated = str(report["sha256"])
    except (KeyError, TypeError) as error:
        raise BootstrapError("producer lacks sealed legacy scalar report") from error
    actual = _file_ref(path, where="legacy scalar training report")
    if actual["sha256"] != stated:
        raise BootstrapError("legacy scalar training report hash drift")
    return actual


def build_plan(
    *,
    lock_path: Path,
    registry_path: Path,
    pointer_path: Path,
    receipt_path: Path,
    incumbent: Path,
    verify_lock_fn: Callable[..., dict[str, Any]] = contract_tool.verify_lock,
) -> dict[str, Any]:
    registry = _fresh(registry_path, where="registry")
    pointer = _fresh(pointer_path, where="current pointer")
    receipt = _fresh(receipt_path, where="bootstrap receipt")
    try:
        lock = verify_lock_fn(lock_path, require_all_job_claims=True)
    except (contract_tool.ContractError, OSError) as error:
        raise BootstrapError(
            f"sealed A1 contract verification failed: {error}"
        ) from error
    lock_ref = _file_ref(lock_path, where="A1 contract lock")
    producer = _producer(lock)
    incumbent_ref = _file_ref(incumbent, where="incumbent checkpoint")
    producer_path = (
        Path(str(producer.get("path", ""))).expanduser().resolve(strict=True)
    )
    if incumbent_ref["path"] != str(producer_path) or incumbent_ref[
        "sha256"
    ] != producer.get("sha256"):
        raise BootstrapError("incumbent is not the sealed A1 producer bytes")
    legacy_report = _legacy_report_ref(producer)
    pool: list[dict[str, Any]] = []
    for record in lock.get("checkpoints", []):
        if record.get("role") not in {"history", "hard_negative"}:
            continue
        ref = _file_ref(Path(str(record["path"])), where=f"{record['role']} checkpoint")
        if ref["sha256"] != record.get("sha256"):
            raise BootstrapError(f"{record['role']} checkpoint hash drift")
        pool.append(
            {
                **ref,
                "md5": _md5(Path(ref["path"])),
                "source_role": str(record["role"]),
            }
        )
    if not pool:
        raise BootstrapError("sealed contract provides no history/hard-negative pool")
    plan: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "mode": "dry-run",
        "contract": {
            **lock_ref,
            "contract_sha256": lock["contract_sha256"],
        },
        "lineage": {
            "name": "a1",
            "promotion_count": 0,
            "basis": "new_registry_lineage_no_persisted_pre_a1_registry",
        },
        "incumbent": {
            **incumbent_ref,
            "md5": _md5(Path(incumbent_ref["path"])),
            "version": 3,
            "legacy_training_report": legacy_report,
        },
        "roles": ["generator_champion", "public_champion", "tournament_bot"],
        "opponent_pool": pool,
        "destinations": {
            "registry": str(registry),
            "current_pointer": str(pointer),
            "receipt": str(receipt),
        },
    }
    plan["plan_sha256"] = _digest(plan)
    return plan


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive(path: Path, payload: bytes, *, mode: int) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_dir(path.parent)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def commit(plan: dict[str, Any]) -> dict[str, Any]:
    if plan.get("schema_version") != RECEIPT_SCHEMA or plan.get("mode") != "dry-run":
        raise BootstrapError("invalid bootstrap plan")
    stated = plan.get("plan_sha256")
    unhashed = dict(plan)
    unhashed.pop("plan_sha256", None)
    if stated != _digest(unhashed):
        raise BootstrapError("bootstrap plan digest mismatch")
    destinations = plan["destinations"]
    registry_path = _fresh(Path(destinations["registry"]), where="registry")
    pointer_path = _fresh(
        Path(destinations["current_pointer"]), where="current pointer"
    )
    receipt_path = _fresh(Path(destinations["receipt"]), where="bootstrap receipt")
    incumbent = plan["incumbent"]
    incumbent_path = Path(incumbent["path"]).resolve(strict=True)
    if (
        _sha256(incumbent_path) != incumbent["sha256"]
        or _md5(incumbent_path) != incumbent["md5"]
    ):
        raise BootstrapError("incumbent bytes drifted after planning")

    registry = ChampionRegistry(registry_path)
    provenance = {
        "bootstrap_schema": RECEIPT_SCHEMA,
        "bootstrap_plan_sha256": stated,
        "contract_sha256": plan["contract"]["contract_sha256"],
        "legacy_training_report": incumbent["legacy_training_report"],
    }
    for role in plan["roles"]:
        registry.set_role(
            role,
            incumbent_path,
            expected_md5=incumbent["md5"],
            version=int(incumbent["version"]),
            provenance=provenance,
            reason="audited A1 registry bootstrap",
        )
    for row in plan["opponent_pool"]:
        pool_path = Path(row["path"]).resolve(strict=True)
        if _sha256(pool_path) != row["sha256"] or _md5(pool_path) != row["md5"]:
            raise BootstrapError("opponent-pool checkpoint drifted after planning")
        registry.append_pool(
            pool_path,
            expected_md5=row["md5"],
            provenance={**provenance, "contract_source_role": row["source_role"]},
            status="active",
            reason="sealed A1 bootstrap opponent",
        )
    # ChampionRegistry.save is not the transaction boundary here.  Render its
    # bytes to a private temporary path, then publish all destinations with
    # exclusive creation and roll back on any failure.
    temporary = registry_path.with_name(f".{registry_path.name}.render.{os.getpid()}")
    registry.path = temporary
    registry.save()
    registry_bytes = temporary.read_bytes()
    temporary.unlink(missing_ok=True)
    pointer_bytes = (str(incumbent_path) + "\n").encode()
    receipt = {**plan, "mode": "committed", "committed_unix_ns": time.time_ns()}
    receipt["registry_sha256"] = "sha256:" + hashlib.sha256(registry_bytes).hexdigest()
    receipt["current_pointer_sha256"] = (
        "sha256:" + hashlib.sha256(pointer_bytes).hexdigest()
    )
    receipt["receipt_sha256"] = _digest(receipt)
    receipt_bytes = json.dumps(receipt, indent=2, sort_keys=True).encode() + b"\n"
    created: list[Path] = []
    try:
        _write_exclusive(receipt_path, receipt_bytes, mode=0o444)
        created.append(receipt_path)
        _write_exclusive(registry_path, registry_bytes, mode=0o600)
        created.append(registry_path)
        _write_exclusive(pointer_path, pointer_bytes, mode=0o600)
        created.append(pointer_path)
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
            _fsync_dir(path.parent)
        raise
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--incumbent", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--current-pointer", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--go", action="store_true")
    args = parser.parse_args()
    try:
        plan = build_plan(
            lock_path=Path(args.lock),
            registry_path=Path(args.registry),
            pointer_path=Path(args.current_pointer),
            receipt_path=Path(args.receipt),
            incumbent=Path(args.incumbent),
        )
        result = commit(plan) if args.go else plan
    except (BootstrapError, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
