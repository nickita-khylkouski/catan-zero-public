#!/usr/bin/env python3
"""Create the immutable promotion-to-production lineage handoff.

The promotion transaction owns the registry and ``CURRENT_CHAMPION`` mutation.
This tool is the deliberately separate read-only bridge into the next data wave:
it accepts only a committed receipt, replays the resulting registry/pointer
state, and freezes the exact generator identity selected by that transaction.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HANDOFF_SCHEMA = "a1-post-promotion-producer-handoff-v1"
GENERATOR_ROLE = "generator_champion"


class HandoffError(RuntimeError):
    """Raised when committed promotion lineage cannot be proven exactly."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest_value(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _stable_read(path: Path, *, where: str) -> tuple[bytes, tuple[int, int, int, int]]:
    """Read one regular file without following a final symlink and pin its inode."""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise HandoffError(f"cannot open {where} without symlink following: {error}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise HandoffError(f"{where} is not a regular file")
        chunks: list[bytes] = []
        while block := os.read(descriptor, 1 << 20):
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise HandoffError(f"{where} changed while it was read")
    try:
        live = path.stat(follow_symlinks=False)
    except OSError as error:
        raise HandoffError(f"cannot restat {where}: {error}") from error
    if (live.st_dev, live.st_ino, live.st_size, live.st_mtime_ns) != identity:
        raise HandoffError(f"{where} was atomically replaced while it was read")
    return b"".join(chunks), identity


def _sha256(path: Path) -> str:
    data, _ = _stable_read(path, where=str(path))
    return _sha256_bytes(data)


def _load_json(path: Path, *, where: str) -> dict[str, Any]:
    try:
        raw, _ = _stable_read(path, where=where)
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise HandoffError(f"cannot load {where} {path}: {error}") from error
    if not isinstance(value, dict):
        raise HandoffError(f"{where} must be a JSON object")
    return value


def _existing_file(path: Path, *, where: str) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise HandoffError(f"cannot resolve {where}: {error}") from error
    if lexical != resolved or not resolved.is_file():
        raise HandoffError(f"{where} must be a canonical regular file: {lexical}")
    return resolved


def _validate_identity(identity: Any, *, checkpoint: Path, sha256: str) -> dict[str, Any]:
    if not isinstance(identity, dict) or set(identity) != {
        "schema_version",
        "checkpoint",
        "search_config",
        "agent_identity_sha256",
    }:
        raise HandoffError("promotion candidate producer identity is malformed")
    unhashed = dict(identity)
    declared = unhashed.pop("agent_identity_sha256")
    if declared != _digest_value(unhashed):
        raise HandoffError("promotion candidate producer identity digest mismatch")
    bound = identity.get("checkpoint")
    if not isinstance(bound, dict) or set(bound) != {"path", "sha256"}:
        raise HandoffError("promotion candidate identity checkpoint is malformed")
    if bound != {"path": str(checkpoint), "sha256": sha256}:
        raise HandoffError("promotion candidate identity binds different checkpoint bytes")
    return dict(identity)


def _registry_path_hint(receipt_path: Path) -> Path:
    value = _load_json(receipt_path, where="promotion receipt lock hint")
    try:
        raw = value["registry"]["path"]
    except (KeyError, TypeError) as error:
        raise HandoffError("promotion receipt has no registry path for canonical locking") from error
    return _existing_file(Path(str(raw)), where="authoritative registry lock hint")


def _snapshot_handoff(receipt_path: Path, *, locked_registry: Path) -> tuple[dict[str, Any], dict[Path, bytes]]:
    """Replay and snapshot all lineage files while the canonical lock is held."""

    from tools import a1_promotion_transaction as promotion

    try:
        receipt, _, registry_path, pointer_path, _, _ = promotion._load_recovery_receipt(  # noqa: SLF001
            receipt_path
        )
    except (promotion.PromotionError, OSError) as error:
        raise HandoffError(f"promotion receipt replay failed: {error}") from error
    if receipt.get("status") != "committed":
        raise HandoffError("promotion receipt is not committed")
    if registry_path != locked_registry:
        raise HandoffError("promotion receipt registry changed across canonical lock acquisition")
    receipt_bytes, _ = _stable_read(receipt_path, where="promotion receipt")
    registry_bytes, _ = _stable_read(registry_path, where="authoritative registry")
    pointer_bytes, _ = _stable_read(pointer_path, where="CURRENT_CHAMPION")
    if _sha256_bytes(registry_bytes) != receipt["registry"]["after_sha256"]:
        raise HandoffError("live registry is not the committed registry-after state")
    if _sha256_bytes(pointer_bytes) != receipt["current_pointer"]["after_sha256"]:
        raise HandoffError("live CURRENT_CHAMPION is not the committed pointer state")

    candidate = receipt.get("candidate")
    if not isinstance(candidate, dict):
        raise HandoffError("promotion receipt candidate is malformed")
    checkpoint = _existing_file(Path(str(candidate.get("path", ""))), where="producer checkpoint")
    checkpoint_bytes, _ = _stable_read(checkpoint, where="producer checkpoint")
    checkpoint_sha256 = _sha256_bytes(checkpoint_bytes)
    if checkpoint_sha256 != candidate.get("sha256"):
        raise HandoffError("promoted candidate checkpoint bytes drifted")
    version = candidate.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise HandoffError("promoted candidate version must be a positive integer")
    identity = _validate_identity(
        candidate.get("agent_identity"), checkpoint=checkpoint, sha256=checkpoint_sha256
    )

    try:
        registry_value = json.loads(registry_bytes.decode("utf-8"))
        role = registry_value["roles"].get(GENERATOR_ROLE)
    except (UnicodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise HandoffError(f"committed registry is malformed: {error}") from error
    if not isinstance(role, dict):
        raise HandoffError("committed registry has no generator_champion role")
    if role.get("role") != GENERATOR_ROLE or role.get("version") != version:
        raise HandoffError("registry generator role/version differs from promoted candidate")
    if Path(str(role.get("checkpoint_path", ""))).resolve(strict=True) != checkpoint:
        raise HandoffError("registry generator role points at a different checkpoint")
    checkpoint_md5 = hashlib.md5(checkpoint_bytes).hexdigest()
    if role.get("md5") != checkpoint_md5:
        raise HandoffError("registry generator role md5 differs from promoted checkpoint")
    provenance = role.get("provenance")
    required_provenance = {
        "a1_promotion_receipt": str(receipt_path),
        "a1_candidate_agent_identity_sha256": identity["agent_identity_sha256"],
        "a1_candidate_search_config": identity["search_config"],
    }
    if not isinstance(provenance, dict) or any(
        provenance.get(key) != expected for key, expected in required_provenance.items()
    ):
        raise HandoffError("registry generator provenance differs from promoted identity")
    expected_pointer = (str(checkpoint) + "\n").encode("utf-8")
    if pointer_bytes != expected_pointer:
        raise HandoffError("CURRENT_CHAMPION bytes do not exactly name the promoted producer")

    handoff: dict[str, Any] = {
        "schema_version": HANDOFF_SCHEMA,
        "promotion_receipt": {
            "path": str(receipt_path),
            "sha256": _sha256_bytes(receipt_bytes),
            "receipt_sha256": receipt["receipt_sha256"],
            "transaction_id": receipt["transaction_id"],
        },
        "registry_after": {
            "path": str(registry_path),
            "sha256": receipt["registry"]["after_sha256"],
            "role": GENERATOR_ROLE,
            "version": version,
            "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha256},
        },
        "current_champion": {
            "path": str(pointer_path),
            "sha256": receipt["current_pointer"]["after_sha256"],
            "bytes_base64": base64.b64encode(pointer_bytes).decode("ascii"),
        },
        "producer_identity": identity,
    }
    handoff["handoff_sha256"] = _digest_value(handoff)
    return handoff, {
        receipt_path: receipt_bytes,
        registry_path: registry_bytes,
        pointer_path: pointer_bytes,
        checkpoint: checkpoint_bytes,
    }


def _revalidate_snapshot(snapshot: dict[Path, bytes]) -> None:
    for path, expected in snapshot.items():
        actual, _ = _stable_read(path, where=f"handoff snapshot {path}")
        if actual != expected:
            raise HandoffError(f"handoff input was replaced before output: {path}")


def _locked_handoff(receipt_path: Path) -> tuple[dict[str, Any], dict[Path, bytes], Any]:
    """Return payload/snapshot plus an entered canonical promotion lock context."""

    from tools import a1_promotion_transaction as promotion

    receipt_path = _existing_file(receipt_path, where="promotion receipt")
    registry_hint = _registry_path_hint(receipt_path)
    lock_path = promotion._enforce_canonical_lock(registry_hint, None)  # noqa: SLF001
    context = promotion._exclusive_lock(lock_path)  # noqa: SLF001
    context.__enter__()
    try:
        payload, snapshot = _snapshot_handoff(receipt_path, locked_registry=registry_hint)
        _revalidate_snapshot(snapshot)
        return payload, snapshot, context
    except BaseException:
        context.__exit__(*sys.exc_info())
        raise


def build_handoff(receipt_path: Path) -> dict[str, Any]:
    """Replay one committed receipt under its canonical promotion lock."""

    payload, _, context = _locked_handoff(receipt_path)
    context.__exit__(None, None, None)
    return payload


def write_handoff(receipt_path: Path, out_path: Path) -> dict[str, Any]:
    payload, snapshot, context = _locked_handoff(receipt_path)
    out_path = Path(os.path.abspath(os.fspath(out_path.expanduser())))
    try:
        if out_path.exists() or out_path.is_symlink():
            raise HandoffError(f"handoff output must be fresh: {out_path}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _revalidate_snapshot(snapshot)
        descriptor = os.open(
            out_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o444,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return payload
    finally:
        context.__exit__(None, None, None)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promotion-receipt", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        payload = write_handoff(Path(args.promotion_receipt), Path(args.out))
    except HandoffError as error:
        parser.exit(2, f"REFUSED: {error}\n")
    print(json.dumps({"status": "PASS", "handoff_sha256": payload["handoff_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
