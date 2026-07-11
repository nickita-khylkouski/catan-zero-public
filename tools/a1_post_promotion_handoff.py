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
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.champion_registry import ChampionRegistry  # noqa: E402


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _load_json(path: Path, *, where: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
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


def build_handoff(receipt_path: Path) -> dict[str, Any]:
    """Replay one committed receipt and return its exact next-producer handoff."""

    # Lazy import avoids a module cycle: promotion verification imports the wave
    # contract, while this producer is also consumed by that contract.
    from tools import a1_promotion_transaction as promotion

    receipt_path = _existing_file(receipt_path, where="promotion receipt")
    try:
        receipt, _, registry_path, pointer_path, _, _ = promotion._load_recovery_receipt(  # noqa: SLF001
            receipt_path
        )
    except (promotion.PromotionError, OSError) as error:
        raise HandoffError(f"promotion receipt replay failed: {error}") from error
    if receipt.get("status") != "committed":
        raise HandoffError("promotion receipt is not committed")
    if _sha256(registry_path) != receipt["registry"]["after_sha256"]:
        raise HandoffError("live registry is not the committed registry-after state")
    pointer_bytes = pointer_path.read_bytes()
    if _sha256_bytes(pointer_bytes) != receipt["current_pointer"]["after_sha256"]:
        raise HandoffError("live CURRENT_CHAMPION is not the committed pointer state")

    candidate = receipt.get("candidate")
    if not isinstance(candidate, dict):
        raise HandoffError("promotion receipt candidate is malformed")
    checkpoint = _existing_file(Path(str(candidate.get("path", ""))), where="producer checkpoint")
    checkpoint_sha256 = _sha256(checkpoint)
    if checkpoint_sha256 != candidate.get("sha256"):
        raise HandoffError("promoted candidate checkpoint bytes drifted")
    version = candidate.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise HandoffError("promoted candidate version must be a positive integer")
    identity = _validate_identity(
        candidate.get("agent_identity"), checkpoint=checkpoint, sha256=checkpoint_sha256
    )

    registry = ChampionRegistry.load(registry_path)
    role = registry.get_role(GENERATOR_ROLE)
    if role is None:
        raise HandoffError("committed registry has no generator_champion role")
    if role.role != GENERATOR_ROLE or role.version != version:
        raise HandoffError("registry generator role/version differs from promoted candidate")
    if Path(role.checkpoint_path).resolve(strict=True) != checkpoint:
        raise HandoffError("registry generator role points at a different checkpoint")
    expected_pointer = (str(checkpoint) + "\n").encode("utf-8")
    if pointer_bytes != expected_pointer:
        raise HandoffError("CURRENT_CHAMPION bytes do not exactly name the promoted producer")

    handoff: dict[str, Any] = {
        "schema_version": HANDOFF_SCHEMA,
        "promotion_receipt": {
            "path": str(receipt_path),
            "sha256": _sha256(receipt_path),
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
    return handoff


def write_handoff(receipt_path: Path, out_path: Path) -> dict[str, Any]:
    payload = build_handoff(receipt_path)
    out_path = Path(os.path.abspath(os.fspath(out_path.expanduser())))
    if out_path.exists() or out_path.is_symlink():
        raise HandoffError(f"handoff output must be fresh: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(out_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return payload


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
