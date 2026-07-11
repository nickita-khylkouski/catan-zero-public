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
from tools import a1_promotion_transaction as promotion_tool  # noqa: E402


RECEIPT_SCHEMA = "a1-champion-registry-bootstrap-v1"
JOURNAL_SCHEMA = "a1-champion-registry-bootstrap-journal-v1"


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
    # Planning is genuinely read-only.  Validate every already-existing prefix
    # without manufacturing the destination parent as a side effect.
    ancestor = lexical.parent
    while not ancestor.exists() and not ancestor.is_symlink():
        if ancestor == ancestor.parent:
            raise BootstrapError(f"{where} has no existing parent ancestor")
        ancestor = ancestor.parent
    if ancestor.is_symlink() or ancestor.resolve(strict=True) != ancestor:
        raise BootstrapError(f"{where} parent ancestry must be canonical")
    return lexical


def _prepare_parent(path: Path, *, where: str) -> None:
    """Create a destination parent only at the committed mutation boundary."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.resolve(strict=True) != path.parent:
        raise BootstrapError(f"{where} parent must be canonical")


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
    training_receipt: Path | None = None,
    candidate: Path | None = None,
    verify_lock_fn: Callable[..., dict[str, Any]] = contract_tool.verify_lock,
) -> dict[str, Any]:
    registry = _fresh(registry_path, where="registry")
    pointer = _fresh(pointer_path, where="current pointer")
    receipt = _fresh(receipt_path, where="bootstrap receipt")
    journal = _fresh(_journal_path(receipt), where="bootstrap prepared journal")
    try:
        lock = verify_lock_fn(lock_path, require_all_job_claims=True)
    except (contract_tool.ContractError, OSError) as strict_error:
        # There is exactly one historical, markerless v2 A1 lock.  Promotion's
        # pinned attestation logic binds its file+semantic digests, source
        # draft, and completed v4 one-dose receipt.  Reuse that allowlist here;
        # never relax verification for markerless locks schema-wide.
        if training_receipt is None or candidate is None:
            raise BootstrapError(
                "sealed A1 contract verification failed and the exact historical "
                "fallback requires --training-receipt and --candidate: "
                f"{strict_error}"
            ) from strict_error
        try:
            _attestation, snapshot = (
                promotion_tool._build_legacy_contract_attestation_snapshot(  # noqa: SLF001
                    lock_path, training_receipt
                )
            )
        except (promotion_tool.PromotionError, OSError) as fallback_error:
            raise BootstrapError(
                f"sealed A1 contract verification failed: {strict_error}; "
                f"historical fallback failed: {fallback_error}"
            ) from fallback_error
        lock = dict(snapshot.contract_lock.value)
        candidate_ref = _file_ref(candidate, where="historical A1 candidate")
        receipt_value = snapshot.training_receipt.value
        try:
            outputs = receipt_value["outputs"]
            output_path = Path(str(outputs["checkpoint"])).resolve(strict=True)
            output_sha = str(outputs["checkpoint_sha256"])
        except (KeyError, TypeError, OSError) as error:
            raise BootstrapError(
                "historical training receipt has no canonical candidate output"
            ) from error
        if candidate_ref != {"path": str(output_path), "sha256": output_sha}:
            raise BootstrapError(
                "candidate is not the exact output bound by the historical training receipt"
            )
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
        "bootstrap_unix_ns": int(lock_path.stat().st_mtime_ns),
        "bootstrap_timestamp": float(lock_path.stat().st_mtime_ns / 1_000_000_000),
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
            "prepared_journal": str(journal),
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
    temporary = path.with_name(f".{path.name}.publish.{os.getpid()}.{time.time_ns()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        _fsync_dir(path.parent)
    except BaseException:
        raise
    finally:
        temporary.unlink(missing_ok=True)


def _journal_path(receipt_path: Path) -> Path:
    return receipt_path.with_name(receipt_path.name + ".prepared")


def _render_registry(plan: dict[str, Any]) -> bytes:
    """Render byte-identical registry state on the first try and every resume."""

    timestamp = float(plan["bootstrap_timestamp"])
    incumbent = plan["incumbent"]
    provenance = {
        "bootstrap_schema": RECEIPT_SCHEMA,
        "bootstrap_plan_sha256": plan["plan_sha256"],
        "contract_sha256": plan["contract"]["contract_sha256"],
        "legacy_training_report": incumbent["legacy_training_report"],
    }
    roles: dict[str, Any] = {}
    pool: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    for role in plan["roles"]:
        pointer = {
            "role": role,
            "checkpoint_path": incumbent["path"],
            "md5": incumbent["md5"],
            "version": int(incumbent["version"]),
            "updated_at": timestamp,
            "provenance": provenance,
        }
        roles[role] = pointer
        transitions.append(
            {
                "ts": timestamp,
                "kind": "set_role",
                "role": role,
                "reason": "audited A1 registry bootstrap",
                "from_pointer": None,
                "to_pointer": pointer,
            }
        )
    for row in plan["opponent_pool"]:
        entry = {
            "checkpoint_path": row["path"],
            "md5": row["md5"],
            "version": None,
            "added_at": timestamp,
            "status": "active",
            "provenance": {
                **provenance,
                "contract_source_role": row["source_role"],
            },
        }
        pool.append(entry)
        transitions.append(
            {
                "ts": timestamp,
                "kind": "pool_append",
                "role": "opponent_pool",
                "reason": "sealed A1 bootstrap opponent",
                "from_pointer": None,
                "to_pointer": entry,
            }
        )
    state = {
        "roles": roles,
        "opponent_pool": pool,
        "transitions": transitions,
        "promotion_counts": {},
    }
    return json.dumps(state, indent=2, sort_keys=True).encode()


def _verify_plan_sources(plan: dict[str, Any]) -> None:
    contract = plan["contract"]
    if _sha256(Path(contract["path"])) != contract["sha256"]:
        raise BootstrapError("A1 contract lock drifted after planning")
    incumbent = plan["incumbent"]
    incumbent_path = Path(incumbent["path"]).resolve(strict=True)
    if (
        _sha256(incumbent_path) != incumbent["sha256"]
        or _md5(incumbent_path) != incumbent["md5"]
    ):
        raise BootstrapError("incumbent bytes drifted after planning")
    report = incumbent["legacy_training_report"]
    if _sha256(Path(report["path"])) != report["sha256"]:
        raise BootstrapError("legacy scalar training report drifted after planning")
    for row in plan["opponent_pool"]:
        pool_path = Path(row["path"]).resolve(strict=True)
        if _sha256(pool_path) != row["sha256"] or _md5(pool_path) != row["md5"]:
            raise BootstrapError("opponent-pool checkpoint drifted after planning")


def _journal_payload(plan: dict[str, Any]) -> dict[str, Any]:
    registry_bytes = _render_registry(plan)
    pointer_bytes = (plan["incumbent"]["path"] + "\n").encode()
    value: dict[str, Any] = {
        "schema_version": JOURNAL_SCHEMA,
        "status": "prepared",
        "plan": plan,
        "registry_sha256": "sha256:" + hashlib.sha256(registry_bytes).hexdigest(),
        "current_pointer_sha256": "sha256:" + hashlib.sha256(pointer_bytes).hexdigest(),
        "prepared_unix_ns": int(plan["bootstrap_unix_ns"]),
    }
    value["journal_sha256"] = _digest(value)
    return value


def _verify_journal(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema_version") != JOURNAL_SCHEMA:
        raise BootstrapError("bootstrap prepared journal schema is invalid")
    declared = value.get("journal_sha256")
    unhashed = dict(value)
    unhashed.pop("journal_sha256", None)
    if declared != _digest(unhashed):
        raise BootstrapError("bootstrap prepared journal digest mismatch")
    plan = value.get("plan")
    if not isinstance(plan, dict):
        raise BootstrapError("bootstrap prepared journal has no plan")
    _verify_plan_digest(plan)
    expected = _journal_payload(plan)
    if value != expected:
        raise BootstrapError(
            "bootstrap prepared journal differs from deterministic plan"
        )
    return value


def _verify_plan_digest(plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != RECEIPT_SCHEMA or plan.get("mode") != "dry-run":
        raise BootstrapError("invalid bootstrap plan")
    stated = plan.get("plan_sha256")
    unhashed = dict(plan)
    unhashed.pop("plan_sha256", None)
    if stated != _digest(unhashed):
        raise BootstrapError("bootstrap plan digest mismatch")


def _publish_exact(path: Path, payload: bytes, *, mode: int, where: str) -> None:
    if path.exists() or path.is_symlink():
        if (
            path.is_symlink()
            or not path.is_file()
            or path.read_bytes() != payload
            or path.stat().st_mode & 0o777 != mode
        ):
            raise BootstrapError(f"existing {where} differs from prepared bytes")
        return
    try:
        _write_exclusive(path, payload, mode=mode)
    except FileExistsError:
        if (
            path.is_symlink()
            or not path.is_file()
            or path.read_bytes() != payload
            or path.stat().st_mode & 0o777 != mode
        ):
            raise BootstrapError(f"racing {where} differs from prepared bytes")


def commit(plan: dict[str, Any]) -> dict[str, Any]:
    _verify_plan_digest(plan)
    _verify_plan_sources(plan)
    destinations = plan["destinations"]
    registry_path = Path(destinations["registry"])
    pointer_path = Path(destinations["current_pointer"])
    receipt_path = Path(destinations["receipt"])
    journal_path = Path(destinations["prepared_journal"])
    for path, where in (
        (registry_path, "registry"),
        (pointer_path, "current pointer"),
        (receipt_path, "bootstrap receipt"),
        (journal_path, "bootstrap prepared journal"),
    ):
        _prepare_parent(path, where=where)
    registry_bytes = _render_registry(plan)
    pointer_bytes = (plan["incumbent"]["path"] + "\n").encode()
    journal = _journal_payload(plan)
    journal_bytes = json.dumps(journal, indent=2, sort_keys=True).encode() + b"\n"
    _publish_exact(
        journal_path,
        journal_bytes,
        mode=0o444,
        where="bootstrap prepared journal",
    )
    _verify_journal(json.loads(journal_path.read_text(encoding="utf-8")))

    # The committed receipt is deliberately LAST. A hard kill before it leaves
    # a durable journal plus zero, one, or two exact publications; rerunning
    # commit resumes only those deterministic bytes.
    _publish_exact(registry_path, registry_bytes, mode=0o600, where="registry")
    _publish_exact(
        pointer_path,
        pointer_bytes,
        mode=0o600,
        where="current pointer",
    )
    receipt = {
        **plan,
        "mode": "committed",
        "committed_unix_ns": journal["prepared_unix_ns"],
        "prepared_journal": {
            "path": str(journal_path),
            "sha256": _sha256(journal_path),
            "journal_sha256": journal["journal_sha256"],
        },
        "registry_sha256": journal["registry_sha256"],
        "current_pointer_sha256": journal["current_pointer_sha256"],
    }
    receipt["receipt_sha256"] = _digest(receipt)
    receipt_bytes = json.dumps(receipt, indent=2, sort_keys=True).encode() + b"\n"
    _publish_exact(
        receipt_path,
        receipt_bytes,
        mode=0o444,
        where="bootstrap committed receipt",
    )
    if (
        _sha256(registry_path) != receipt["registry_sha256"]
        or _sha256(pointer_path) != receipt["current_pointer_sha256"]
        or json.loads(receipt_path.read_text(encoding="utf-8")) != receipt
    ):
        raise BootstrapError("bootstrap publication verification failed")
    return receipt


def _resume_plan_from_journal(
    journal_path: Path,
    *,
    lock_path: Path,
    registry_path: Path,
    pointer_path: Path,
    receipt_path: Path,
    incumbent: Path,
) -> dict[str, Any]:
    try:
        journal = _verify_journal(json.loads(journal_path.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BootstrapError(
            f"cannot resume bootstrap prepared journal: {error}"
        ) from error
    plan = journal["plan"]
    expected_destinations = {
        "registry": str(Path(os.path.abspath(os.fspath(registry_path.expanduser())))),
        "current_pointer": str(
            Path(os.path.abspath(os.fspath(pointer_path.expanduser())))
        ),
        "receipt": str(Path(os.path.abspath(os.fspath(receipt_path.expanduser())))),
        "prepared_journal": str(journal_path),
    }
    if plan.get("destinations") != expected_destinations:
        raise BootstrapError("resume arguments differ from prepared destinations")
    if Path(plan["contract"]["path"]).resolve(strict=True) != lock_path.resolve(
        strict=True
    ):
        raise BootstrapError("resume contract lock differs from prepared plan")
    if Path(plan["incumbent"]["path"]).resolve(strict=True) != incumbent.resolve(
        strict=True
    ):
        raise BootstrapError("resume incumbent differs from prepared plan")
    _verify_plan_sources(plan)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--incumbent", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--current-pointer", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--training-receipt")
    parser.add_argument("--candidate")
    parser.add_argument("--go", action="store_true")
    args = parser.parse_args()
    try:
        receipt_path = Path(os.path.abspath(os.fspath(Path(args.receipt).expanduser())))
        journal_path = _journal_path(receipt_path)
        if args.go and journal_path.is_file():
            plan = _resume_plan_from_journal(
                journal_path,
                lock_path=Path(args.lock),
                registry_path=Path(args.registry),
                pointer_path=Path(args.current_pointer),
                receipt_path=receipt_path,
                incumbent=Path(args.incumbent),
            )
        else:
            plan = build_plan(
                lock_path=Path(args.lock),
                registry_path=Path(args.registry),
                pointer_path=Path(args.current_pointer),
                receipt_path=receipt_path,
                incumbent=Path(args.incumbent),
                training_receipt=(
                    Path(args.training_receipt) if args.training_receipt else None
                ),
                candidate=Path(args.candidate) if args.candidate else None,
            )
        result = commit(plan) if args.go else plan
    except (BootstrapError, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
