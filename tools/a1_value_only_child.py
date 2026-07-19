"""Authenticated non-promotable policy-child -> value-only calibration edge."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "a1-value-only-child-receipt-v1"


class ValueOnlyChildError(ValueError):
    """The receipt cannot authorize a value-only child calibration."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _canonical_sha256(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _regular_ref(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueOnlyChildError(f"{label} must be a {{path, sha256}} object")
    raw_path, stated = value["path"], value["sha256"]
    if not isinstance(raw_path, str) or not isinstance(stated, str):
        raise ValueOnlyChildError(f"{label} reference fields must be strings")
    try:
        path = Path(raw_path).expanduser().resolve(strict=True)
    except OSError as error:
        raise ValueOnlyChildError(f"cannot resolve {label}: {error}") from error
    if path.is_symlink() or not path.is_file() or _sha256(path) != stated:
        raise ValueOnlyChildError(f"{label} path/sha256 binding drift")
    return {"path": str(path), "sha256": stated}


def verify_receipt(path: str | Path) -> dict[str, Any]:
    """Replay the narrow calibration lineage edge from immutable receipts."""

    try:
        receipt_path = Path(path).expanduser().resolve(strict=True)
        if receipt_path.is_symlink() or not receipt_path.is_file():
            raise OSError("receipt is not a regular file")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueOnlyChildError(f"cannot load value-only child receipt: {error}") from error
    if not isinstance(receipt, dict):
        raise ValueOnlyChildError("value-only child receipt must be a JSON object")
    unsigned = dict(receipt)
    stated_digest = unsigned.pop("receipt_sha256", None)
    required = {
        "schema_version",
        "mode",
        "promotion_eligible",
        "parent_producer",
        "child_checkpoint",
        "child_training_report",
        "receipt_sha256",
    }
    if set(receipt) != required or (
        receipt.get("schema_version") != SCHEMA
        or receipt.get("mode") != "value_only_child"
        or receipt.get("promotion_eligible") is not False
        or stated_digest != _canonical_sha256(unsigned)
    ):
        raise ValueOnlyChildError("value-only child receipt schema/status/digest drift")
    parent = _regular_ref(receipt["parent_producer"], label="parent producer")
    child = _regular_ref(receipt["child_checkpoint"], label="child checkpoint")
    report_ref = _regular_ref(receipt["child_training_report"], label="child training report")
    try:
        report = json.loads(Path(report_ref["path"]).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueOnlyChildError(f"cannot parse child training report: {error}") from error
    signal = report.get("policy_training_signal") if isinstance(report, dict) else None
    reported_checkpoint = report.get("checkpoint") if isinstance(report, dict) else None
    if (
        not isinstance(report, dict)
        or report.get("init_checkpoint_sha256") != parent["sha256"]
        or report.get("train_value_only") is not False
        or not isinstance(signal, dict)
        or signal.get("trained_policy_objective") is not True
        or signal.get("status") != "trained"
        or not isinstance(reported_checkpoint, str)
        or str(Path(reported_checkpoint).expanduser().resolve(strict=True))
        != child["path"]
    ):
        raise ValueOnlyChildError("child report does not authenticate a policy-trained child")
    return {
        "schema_version": SCHEMA,
        "receipt_path": str(receipt_path),
        "receipt_sha256": _sha256(receipt_path),
        "parent_producer": parent,
        "child_checkpoint": child,
        "child_training_report": report_ref,
        "promotion_eligible": False,
    }
