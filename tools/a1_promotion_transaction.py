#!/usr/bin/env python3
"""Fail-closed, recoverable A1 generator-champion promotion transaction.

This tool performs no evaluation and never deploys a checkpoint to the fleet.
It consumes a sealed A1 contract and a typed, passing promotion adjudication,
then updates only the authoritative ChampionRegistry and its local
CURRENT_CHAMPION pointer.  Both files are protected by one exclusive lock and
are replaced atomically one at a time.  Because POSIX cannot atomically replace
two unrelated paths, a durable prepared receipt and exact before-byte backups
make an interrupted two-file commit recoverable.

Promotion and recovery are dry-run by default.  ``--go`` is always required for
mutation.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools import a1_pre_wave_contract as a1_contract  # noqa: E402
from tools.champion_registry import ChampionRegistry  # noqa: E402


ADJUDICATION_SCHEMA = "a1-promotion-adjudication-v1"
RECEIPT_SCHEMA = "a1-promotion-transaction-receipt-v1"
REQUIRED_EVIDENCE_KINDS = {
    "mechanism_calibration",
    "internal_h2h",
    "external_panel",
    "high_regret",
    "bucket_veto",
}
REQUIRED_CHECKS = {
    "provenance",
    "mechanism_calibration",
    "internal_h2h",
    "external_panel",
    "high_regret",
    "bucket_veto",
}


class PromotionError(RuntimeError):
    """Raised when promotion evidence or transaction state fails closed."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest_value(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PromotionError(f"cannot load JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise PromotionError(f"{path} must contain a JSON object")
    return value


def _require_exact_keys(value: Any, keys: set[str], *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PromotionError(f"{where} must be an object")
    actual = set(value)
    if actual != keys:
        raise PromotionError(
            f"{where} keys differ: missing={sorted(keys - actual)} "
            f"unexpected={sorted(actual - keys)}"
        )
    return dict(value)


def _absolute(path: Any, *, base: Path) -> Path:
    if not isinstance(path, str) or not path.strip():
        raise PromotionError("artifact path must be a non-empty string")
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = base / resolved
    return resolved.resolve()


def _validate_sha256(value: Any, *, where: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or len(value) != 71
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise PromotionError(f"{where} must be a full lowercase sha256: digest")
    return value


def _validate_file_ref(
    raw: Any, *, base: Path, where: str, extra_keys: set[str] | None = None
) -> tuple[Path, dict[str, Any]]:
    keys = {"path", "sha256"} | set(extra_keys or ())
    value = _require_exact_keys(raw, keys, where=where)
    path = _absolute(value["path"], base=base)
    if not path.is_file():
        raise PromotionError(f"{where} artifact is missing: {path}")
    declared = _validate_sha256(value["sha256"], where=f"{where}.sha256")
    actual = _sha256(path)
    if declared != actual:
        raise PromotionError(
            f"{where} artifact drift: declared {declared}, actual {actual} ({path})"
        )
    value["path"] = str(path)
    return path, value


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_dir(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_bytes(
        path, json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )


def _write_new_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    _fsync_dir(path.parent)


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise PromotionError(f"promotion lock is already held: {path}") from error
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _read_current_pointer(path: Path) -> str:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise PromotionError(f"cannot read current pointer {path}: {error}") from error
    nonempty = [line.strip() for line in lines if line.strip()]
    if len(nonempty) != 1:
        raise PromotionError(
            f"current pointer {path} must contain exactly one non-empty checkpoint path"
        )
    return str(_absolute(nonempty[0], base=path.parent))


def _verify_training_report(
    path: Path, *, contract: dict[str, Any], contract_sha256: str
) -> dict[str, Any]:
    report = _load_json(path)
    recipe = contract["science"]["learner_training_recipe"]
    recipe_sha = contract["science"]["learner_training_recipe_sha256"]
    required = {
        "a1_contract_sha256": contract_sha256,
        "a1_learner_training_recipe_sha256": recipe_sha,
        "a1_bound_learner_training_recipe": recipe,
        "arch": "entity_graph",
        "mask_hidden_info": True,
        "track": "2p_no_trade",
        "vps_to_win": 10,
    }
    for key, expected in required.items():
        if report.get(key) != expected:
            raise PromotionError(
                f"candidate training report drift at {key}: "
                f"{report.get(key)!r} != {expected!r}"
            )
    steps = report.get("steps_completed")
    epochs = report.get("epochs")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise PromotionError("candidate training report has no completed optimizer steps")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise PromotionError("candidate training report has no completed epoch")
    return report


def _verify_contract(
    path: Path,
    *,
    verify_lock_fn: Callable[..., dict[str, Any]] = a1_contract.verify_lock,
) -> dict[str, Any]:
    try:
        lock = verify_lock_fn(path, require_all_job_claims=True)
    except Exception as error:
        raise PromotionError(f"sealed A1 contract verification failed: {error}") from error
    search = lock.get("science", {}).get("search_operator", {})
    if search.get("n_full") != 128:
        raise PromotionError(
            f"current A1 promotion requires global n_full=128, got {search.get('n_full')!r}"
        )
    if search.get("n_full_wide") is not None or search.get("wide_roots_always_full") is not False:
        raise PromotionError(
            "current A1 promotion is global n128 only; adaptive/global alternate "
            "budgets are forbidden"
        )
    contract_sha = lock.get("contract_sha256")
    _validate_sha256(contract_sha, where="contract.contract_sha256")
    return lock


def _verify_adjudication(
    path: Path,
    *,
    contract: dict[str, Any],
    registry: ChampionRegistry,
    current_pointer: Path,
) -> dict[str, Any]:
    raw = _load_json(path)
    expected_keys = {
        "schema_version",
        "passed",
        "decision",
        "contract_sha256",
        "candidate",
        "champion",
        "checks",
        "nth_confirmation_required",
        "nth_confirmation_passed",
        "evidence",
        "adjudication_sha256",
    }
    value = _require_exact_keys(raw, expected_keys, where="adjudication")
    if value["schema_version"] != ADJUDICATION_SCHEMA:
        raise PromotionError(f"adjudication schema must be {ADJUDICATION_SCHEMA!r}")
    declared_digest = _validate_sha256(
        value["adjudication_sha256"], where="adjudication.adjudication_sha256"
    )
    unhashed = dict(value)
    unhashed.pop("adjudication_sha256")
    if declared_digest != _digest_value(unhashed):
        raise PromotionError("adjudication semantic digest mismatch")
    if value["passed"] is not True or value["decision"] != "promote":
        raise PromotionError("adjudication is not a typed passing promote decision")
    contract_sha = contract["contract_sha256"]
    if value["contract_sha256"] != contract_sha:
        raise PromotionError("adjudication binds a different sealed A1 contract")

    base = path.parent
    candidate_raw = _require_exact_keys(
        value["candidate"], {"path", "sha256", "version", "training_report"}, where="candidate"
    )
    candidate_path, candidate_ref = _validate_file_ref(
        {"path": candidate_raw["path"], "sha256": candidate_raw["sha256"]},
        base=base,
        where="candidate",
    )
    training_path, training_ref = _validate_file_ref(
        candidate_raw["training_report"], base=base, where="candidate.training_report"
    )
    _verify_training_report(training_path, contract=contract, contract_sha256=contract_sha)
    champion_raw = _require_exact_keys(
        value["champion"], {"path", "sha256", "version"}, where="champion"
    )
    champion_path, champion_ref = _validate_file_ref(
        {"path": champion_raw["path"], "sha256": champion_raw["sha256"]},
        base=base,
        where="champion",
    )
    if candidate_path == champion_path or candidate_ref["sha256"] == champion_ref["sha256"]:
        raise PromotionError("candidate and incumbent champion must have distinct bytes")
    for label, raw_version in (
        ("candidate.version", candidate_raw["version"]),
        ("champion.version", champion_raw["version"]),
    ):
        if isinstance(raw_version, bool) or not isinstance(raw_version, int) or raw_version < 0:
            raise PromotionError(f"{label} must be a non-negative integer")
    if candidate_raw["version"] != champion_raw["version"] + 1:
        raise PromotionError("candidate version must be exactly incumbent version + 1")

    incumbent = registry.get_role("generator_champion")
    if incumbent is None:
        raise PromotionError("authoritative registry has no generator_champion")
    if str(Path(incumbent.checkpoint_path).expanduser().resolve()) != str(champion_path):
        raise PromotionError("adjudicated champion path differs from registry generator_champion")
    if incumbent.md5 != _md5(champion_path):
        raise PromotionError("registry generator_champion md5 differs from incumbent bytes")
    if incumbent.version != champion_raw["version"]:
        raise PromotionError("adjudicated champion version differs from registry")
    if _read_current_pointer(current_pointer) != str(champion_path):
        raise PromotionError("CURRENT_CHAMPION pointer differs from adjudicated incumbent")

    checks = _require_exact_keys(value["checks"], REQUIRED_CHECKS, where="checks")
    failed_checks = sorted(name for name, passed in checks.items() if passed is not True)
    if failed_checks:
        raise PromotionError(f"adjudication has non-passing checks: {failed_checks}")
    next_count = registry.promotion_count("generator_champion") + 1
    nth_required = next_count % 3 == 0
    if value["nth_confirmation_required"] is not nth_required:
        raise PromotionError(
            "adjudication every-third confirmation requirement disagrees with registry count"
        )
    if nth_required and value["nth_confirmation_passed"] is not True:
        raise PromotionError("required every-third n64 confirmation did not pass")
    if not nth_required and value["nth_confirmation_passed"] not in {False, None}:
        raise PromotionError("non-required nth confirmation must be false or null")

    evidence = value["evidence"]
    if not isinstance(evidence, list):
        raise PromotionError("adjudication.evidence must be a list")
    evidence_by_kind: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(evidence):
        item = _require_exact_keys(
            record, {"kind", "path", "sha256"}, where=f"evidence[{index}]"
        )
        kind = item["kind"]
        if not isinstance(kind, str) or kind in evidence_by_kind:
            raise PromotionError(f"evidence[{index}].kind is invalid or duplicated")
        _, verified = _validate_file_ref(
            {"path": item["path"], "sha256": item["sha256"]},
            base=base,
            where=f"evidence[{index}]",
        )
        evidence_by_kind[kind] = {"kind": kind, **verified}
    missing_evidence = REQUIRED_EVIDENCE_KINDS - set(evidence_by_kind)
    unexpected_evidence = set(evidence_by_kind) - REQUIRED_EVIDENCE_KINDS
    if missing_evidence or unexpected_evidence:
        raise PromotionError(
            f"adjudication evidence kinds differ: missing={sorted(missing_evidence)} "
            f"unexpected={sorted(unexpected_evidence)}"
        )

    return {
        "candidate": {
            **candidate_ref,
            "version": candidate_raw["version"],
            "md5": _md5(candidate_path),
            "training_report": training_ref,
        },
        "champion": {
            **champion_ref,
            "version": champion_raw["version"],
            "md5": _md5(champion_path),
        },
        "evidence": [evidence_by_kind[kind] for kind in sorted(evidence_by_kind)],
        "adjudication_sha256": declared_digest,
        "next_promotion_count": next_count,
        "nth_confirmation_required": nth_required,
    }


def _stage_registry(
    registry_path: Path,
    *,
    verified: dict[str, Any],
    contract_sha256: str,
    adjudication_path: Path,
    receipt_path: Path,
    reason: str,
) -> tuple[bytes, int]:
    stage = registry_path.parent / f".{registry_path.name}.{uuid.uuid4().hex}.stage"
    _write_new_bytes(stage, registry_path.read_bytes())
    try:
        registry = ChampionRegistry.load(stage)
        champion = verified["champion"]
        candidate = verified["candidate"]
        provenance = {
            "a1_contract_sha256": contract_sha256,
            "a1_promotion_adjudication": str(adjudication_path),
            "a1_promotion_adjudication_sha256": verified["adjudication_sha256"],
            "a1_promotion_receipt": str(receipt_path),
            "fleet_ckpt_updated": False,
        }
        registry.append_pool(
            champion["path"],
            expected_md5=champion["md5"],
            version=champion["version"],
            provenance=provenance,
            status="active",
            reason="dethroned A1 generator champion",
        )
        registry.set_role(
            "generator_champion",
            candidate["path"],
            expected_md5=candidate["md5"],
            version=candidate["version"],
            provenance=provenance,
            reason=reason,
        )
        count = registry.record_promotion("generator_champion")
        if count != verified["next_promotion_count"]:
            raise PromotionError("staged registry promotion count drift")
        registry.save()
        return stage.read_bytes(), count
    finally:
        stage.unlink(missing_ok=True)
        stage.with_suffix(stage.suffix + ".tmp").unlink(missing_ok=True)


def _backup_paths(receipt_path: Path) -> tuple[Path, Path]:
    return (
        receipt_path.with_name(receipt_path.name + ".registry.before"),
        receipt_path.with_name(receipt_path.name + ".current.before"),
    )


def prepare_promotion(
    *,
    registry_path: Path,
    current_pointer: Path,
    contract_lock: Path,
    adjudication_path: Path,
    receipt_path: Path,
    reason: str,
    verify_lock_fn: Callable[..., dict[str, Any]] = a1_contract.verify_lock,
) -> dict[str, Any]:
    if not registry_path.is_file() or registry_path.stat().st_size == 0:
        raise PromotionError("authoritative registry must be an existing non-empty file")
    if not current_pointer.is_file():
        raise PromotionError("CURRENT_CHAMPION pointer must already exist")
    if receipt_path.exists():
        raise PromotionError(f"receipt path already exists: {receipt_path}")
    contract = _verify_contract(contract_lock, verify_lock_fn=verify_lock_fn)
    registry = ChampionRegistry.load(registry_path)
    verified = _verify_adjudication(
        adjudication_path,
        contract=contract,
        registry=registry,
        current_pointer=current_pointer,
    )
    registry_before = registry_path.read_bytes()
    current_before = current_pointer.read_bytes()
    registry_after, promotion_count = _stage_registry(
        registry_path,
        verified=verified,
        contract_sha256=contract["contract_sha256"],
        adjudication_path=adjudication_path.resolve(),
        receipt_path=receipt_path.resolve(),
        reason=reason,
    )
    current_after = (verified["candidate"]["path"] + "\n").encode("utf-8")
    transaction_id = uuid.uuid4().hex
    return {
        "schema_version": RECEIPT_SCHEMA,
        "transaction_id": transaction_id,
        "status": "dry_run",
        "created_at": time.time(),
        "registry": {
            "path": str(registry_path.resolve()),
            "before_sha256": _sha256_bytes(registry_before),
            "after_sha256": _sha256_bytes(registry_after),
        },
        "current_pointer": {
            "path": str(current_pointer.resolve()),
            "before_sha256": _sha256_bytes(current_before),
            "after_sha256": _sha256_bytes(current_after),
        },
        "contract": {
            "path": str(contract_lock.resolve()),
            "contract_sha256": contract["contract_sha256"],
            "n_full": 128,
            "n_full_wide": None,
        },
        "adjudication": {
            "path": str(adjudication_path.resolve()),
            "adjudication_sha256": verified["adjudication_sha256"],
        },
        "candidate": verified["candidate"],
        "champion": verified["champion"],
        "evidence": verified["evidence"],
        "promotion_count": promotion_count,
        "nth_confirmation_required": verified["nth_confirmation_required"],
        "reason": reason,
        "fleet_ckpt_updated": False,
        "rollback": {},
        "_bytes": {
            "registry_before": registry_before,
            "registry_after": registry_after,
            "current_before": current_before,
            "current_after": current_after,
        },
    }


def _public_receipt(plan: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in plan.items() if key != "_bytes"}


def execute_promotion(
    *,
    registry_path: Path,
    current_pointer: Path,
    contract_lock: Path,
    adjudication_path: Path,
    receipt_path: Path,
    reason: str,
    lock_path: Path | None = None,
    go: bool = False,
    verify_lock_fn: Callable[..., dict[str, Any]] = a1_contract.verify_lock,
) -> dict[str, Any]:
    lock_path = lock_path or registry_path.with_suffix(registry_path.suffix + ".a1.lock")
    with _exclusive_lock(lock_path):
        plan = prepare_promotion(
            registry_path=registry_path,
            current_pointer=current_pointer,
            contract_lock=contract_lock,
            adjudication_path=adjudication_path,
            receipt_path=receipt_path,
            reason=reason,
            verify_lock_fn=verify_lock_fn,
        )
        if not go:
            return _public_receipt(plan)

        payload = plan["_bytes"]
        registry_backup, current_backup = _backup_paths(receipt_path)
        if registry_backup.exists() or current_backup.exists():
            raise PromotionError("rollback backup path already exists")
        _write_new_bytes(registry_backup, payload["registry_before"])
        try:
            _write_new_bytes(current_backup, payload["current_before"])
        except BaseException:
            registry_backup.unlink(missing_ok=True)
            raise
        receipt = _public_receipt(plan)
        receipt["status"] = "prepared"
        receipt["rollback"] = {
            "registry_backup": str(registry_backup.resolve()),
            "registry_backup_sha256": _sha256(registry_backup),
            "current_backup": str(current_backup.resolve()),
            "current_backup_sha256": _sha256(current_backup),
        }
        _write_new_bytes(
            receipt_path,
            json.dumps(receipt, indent=2, sort_keys=True).encode("utf-8") + b"\n",
        )
        try:
            _atomic_write_bytes(registry_path, payload["registry_after"])
            _atomic_write_bytes(current_pointer, payload["current_after"])
            if _sha256(registry_path) != receipt["registry"]["after_sha256"]:
                raise PromotionError("registry post-commit hash mismatch")
            if _sha256(current_pointer) != receipt["current_pointer"]["after_sha256"]:
                raise PromotionError("current pointer post-commit hash mismatch")
            receipt["status"] = "committed"
            receipt["committed_at"] = time.time()
            _atomic_write_json(receipt_path, receipt)
            return receipt
        except BaseException as error:
            rollback_errors: list[str] = []
            for path, before in (
                (registry_path, payload["registry_before"]),
                (current_pointer, payload["current_before"]),
            ):
                try:
                    _atomic_write_bytes(path, before)
                except BaseException as rollback_error:
                    rollback_errors.append(f"{path}: {rollback_error}")
            receipt["status"] = "rollback_failed" if rollback_errors else "rolled_back"
            receipt["error"] = str(error)
            receipt["rollback_errors"] = rollback_errors
            _atomic_write_json(receipt_path, receipt)
            if rollback_errors:
                raise PromotionError(
                    f"promotion failed and rollback was incomplete: {rollback_errors}"
                ) from error
            raise PromotionError("promotion failed; original registry/pointer restored") from error


def recover_transaction(
    *, receipt_path: Path, go: bool = False, lock_path: Path | None = None
) -> dict[str, Any]:
    receipt = _load_json(receipt_path)
    if receipt.get("schema_version") != RECEIPT_SCHEMA:
        raise PromotionError(f"receipt schema must be {RECEIPT_SCHEMA!r}")
    if receipt.get("status") not in {"prepared", "committed", "rollback_failed"}:
        raise PromotionError(
            f"receipt status {receipt.get('status')!r} is not recoverable"
        )
    registry_path = Path(receipt["registry"]["path"])
    current_pointer = Path(receipt["current_pointer"]["path"])
    lock_path = lock_path or registry_path.with_suffix(registry_path.suffix + ".a1.lock")
    with _exclusive_lock(lock_path):
        registry_backup = Path(receipt["rollback"]["registry_backup"])
        current_backup = Path(receipt["rollback"]["current_backup"])
        if _sha256(registry_backup) != receipt["rollback"]["registry_backup_sha256"]:
            raise PromotionError("registry rollback backup hash mismatch")
        if _sha256(current_backup) != receipt["rollback"]["current_backup_sha256"]:
            raise PromotionError("current-pointer rollback backup hash mismatch")
        for label, path, state in (
            ("registry", registry_path, receipt["registry"]),
            ("current pointer", current_pointer, receipt["current_pointer"]),
        ):
            actual = _sha256(path)
            if actual not in {state["before_sha256"], state["after_sha256"]}:
                raise PromotionError(
                    f"{label} contains unknown bytes; refusing receipt recovery: {actual}"
                )
        result = {
            "schema_version": RECEIPT_SCHEMA,
            "transaction_id": receipt["transaction_id"],
            "status": "recovery_dry_run" if not go else "recovered",
            "registry": str(registry_path),
            "current_pointer": str(current_pointer),
            "receipt": str(receipt_path.resolve()),
        }
        if not go:
            return result
        _atomic_write_bytes(registry_path, registry_backup.read_bytes())
        _atomic_write_bytes(current_pointer, current_backup.read_bytes())
        if _sha256(registry_path) != receipt["registry"]["before_sha256"]:
            raise PromotionError("registry recovery verification failed")
        if _sha256(current_pointer) != receipt["current_pointer"]["before_sha256"]:
            raise PromotionError("current-pointer recovery verification failed")
        receipt["status"] = "recovered"
        receipt["recovered_at"] = time.time()
        _atomic_write_json(receipt_path, receipt)
        return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    promote = subparsers.add_parser("promote", help="preflight or commit one A1 promotion")
    promote.add_argument("--registry", required=True, type=Path)
    promote.add_argument("--current-pointer", required=True, type=Path)
    promote.add_argument("--contract-lock", required=True, type=Path)
    promote.add_argument("--adjudication", required=True, type=Path)
    promote.add_argument("--receipt", required=True, type=Path)
    promote.add_argument("--reason", required=True)
    promote.add_argument("--lock-file", type=Path, default=None)
    promote.add_argument("--go", action="store_true", help="commit; default is dry-run")

    recover = subparsers.add_parser("recover", help="restore exact before bytes from a receipt")
    recover.add_argument("--receipt", required=True, type=Path)
    recover.add_argument("--lock-file", type=Path, default=None)
    recover.add_argument("--go", action="store_true", help="restore; default is dry-run")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "promote":
            result = execute_promotion(
                registry_path=args.registry,
                current_pointer=args.current_pointer,
                contract_lock=args.contract_lock,
                adjudication_path=args.adjudication,
                receipt_path=args.receipt,
                reason=args.reason,
                lock_path=args.lock_file,
                go=bool(args.go),
            )
        else:
            result = recover_transaction(
                receipt_path=args.receipt,
                lock_path=args.lock_file,
                go=bool(args.go),
            )
    except PromotionError as error:
        print(f"REFUSING: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
