#!/usr/bin/env python3
"""Emit typed S2/S3 operator bindings when experiments are explicitly waived.

This bridge is intentionally not a search adjudicator.  It records a narrow
operator decision for the current A1 wave:

* S2 binds global ``n_full=128`` with the existing ``n_fast=16,p_full=.25``;
* S3 holds adaptive n256 disabled (``null,null,false``).

The input S1 decision must replay exactly through
``search_teacher_adjudicator.py``.  Outputs say explicitly that they are
operator choices, not strength evidence, bind the exact S1 bytes, bind this
emitter's bytes, carry a self-digest, and are created read-only without
overwriting an existing path.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools import search_teacher_adjudicator as search_adjudicator  # noqa: E402


SCHEMA = "rl-rnd-operator-binding-v1"
ARTIFACT_KIND = "operator_choice_not_strength_evidence"
STATEMENT = (
    "This artifact records an explicit operator choice for the current wave; "
    "it is not experimental strength evidence and makes no claim that the "
    "selected operator is stronger."
)
S2_REASON = "operator_directive_bind_global_n128_without_n64_strength_test"
S3_REASON = "operator_directive_hold_adaptive_n256_without_s3_strength_test"
S2_OPERATOR = "global_n128"
S3_OPERATOR = "adaptive_n256_disabled"
S2_SELECTED = {"n_full": 128, "n_fast": 16, "p_full": 0.25}
S3_SELECTED = {
    "n_full_wide": None,
    "n_full_wide_threshold": None,
    "wide_roots_always_full": False,
}


class BindingError(ValueError):
    """Raised when a source or destination cannot form an honest binding."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise BindingError(f"value is not canonical JSON: {error}") from error


def _digest_value(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _serialized(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError as error:
        raise BindingError(f"cannot hash {path}: {error}") from error
    return "sha256:" + digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BindingError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise BindingError(f"{path} must contain a JSON object")
    return payload


def _reference(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve(strict=True)
    return {"path": str(resolved), "sha256": _sha256(resolved)}


def _parse_utc(raw: str | None) -> str:
    if raw is None:
        value = dt.datetime.now(dt.timezone.utc)
    else:
        try:
            value = dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError as error:
            raise BindingError("--binding-time-utc must be an ISO-8601 timestamp") from error
        if value.tzinfo is None or value.utcoffset() != dt.timedelta(0):
            raise BindingError("--binding-time-utc must carry an explicit UTC offset")
        value = value.astimezone(dt.timezone.utc)
    return value.isoformat().replace("+00:00", "Z")


def _replay_s1(path: Path) -> dict[str, Any]:
    decision_path = path.expanduser().resolve(strict=True)
    payload = _load_json(decision_path)

    # Replay historical decisions with the exact adjudicator bytes they bind.
    # Replaying with today's checkout makes an otherwise immutable decision
    # drift whenever the adjudicator later gains hardening or provenance fields.
    adjudicator: Any = search_adjudicator
    adjudicator_ref = payload.get("adjudicator")
    if adjudicator_ref is not None:
        if not isinstance(adjudicator_ref, dict) or set(adjudicator_ref) != {
            "path",
            "sha256",
        }:
            raise BindingError("S1 adjudicator reference must contain path and sha256")
        adjudicator_path = Path(str(adjudicator_ref["path"])).expanduser()
        if not adjudicator_path.is_absolute():
            adjudicator_path = decision_path.parent / adjudicator_path
        adjudicator_path = adjudicator_path.resolve(strict=True)
        if _sha256(adjudicator_path) != adjudicator_ref["sha256"]:
            raise BindingError(f"S1 adjudicator hash drift at {adjudicator_path}")
        current_adjudicator_path = Path(search_adjudicator.__file__).resolve(strict=True)
        if adjudicator_path != current_adjudicator_path:
            try:
                namespace = runpy.run_path(str(adjudicator_path))
                adjudicator = type(
                    "BoundAdjudicator",
                    (),
                    {
                        "DECISION_SCHEMA": namespace["DECISION_SCHEMA"],
                        "MANIFEST_SCHEMA": namespace["MANIFEST_SCHEMA"],
                        "AdjudicationError": namespace["AdjudicationError"],
                        "adjudicate": staticmethod(namespace["adjudicate"]),
                    },
                )
            except (KeyError, OSError, RuntimeError) as error:
                raise BindingError(
                    f"cannot load bound S1 adjudicator {adjudicator_path}: {error}"
                ) from error
    if (
        payload.get("schema_version") != adjudicator.DECISION_SCHEMA
        or payload.get("stage") != "s1"
        or payload.get("passed") is not True
    ):
        raise BindingError(f"{decision_path} is not a passed typed S1 decision")
    source_artifacts = payload.get("source_artifacts")
    if not isinstance(source_artifacts, list):
        raise BindingError("S1 decision has no source_artifact list")
    manifests: list[Path] = []
    for raw in source_artifacts:
        if not isinstance(raw, dict) or set(raw) != {"path", "sha256"}:
            raise BindingError("S1 source artifact references must contain path and sha256")
        candidate = Path(str(raw["path"])).expanduser()
        if not candidate.is_absolute():
            candidate = decision_path.parent / candidate
        candidate = candidate.resolve(strict=True)
        if _sha256(candidate) != raw["sha256"]:
            raise BindingError(f"S1 source artifact hash drift at {candidate}")
        if candidate.suffix.lower() != ".json":
            continue
        candidate_payload = _load_json(candidate)
        if (
            candidate_payload.get("schema_version")
            == adjudicator.MANIFEST_SCHEMA
            and candidate_payload.get("stage") == "s1"
        ):
            manifests.append(candidate)
    if len(manifests) != 1:
        raise BindingError(
            f"S1 decision must bind exactly one replayable manifest, found {len(manifests)}"
        )
    try:
        replayed = adjudicator.adjudicate(manifests[0])
    except adjudicator.AdjudicationError as error:
        raise BindingError(f"S1 semantic replay failed: {error}") from error
    if replayed != payload:
        raise BindingError("S1 decision does not equal semantic replay")
    if payload.get("selected_fields_sha256") != _digest_value(
        payload.get("selected_fields")
    ):
        raise BindingError("S1 selected_fields_sha256 mismatch")
    return payload


def _seal_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if "artifact_content_sha256" in payload:
        raise BindingError("operator binding payload is already sealed")
    result = dict(payload)
    result["artifact_content_sha256"] = _digest_value(payload)
    return result


def build_bindings(
    s1_decision_path: Path,
    *,
    s2_output_path: Path,
    binding_time_utc: str | None = None,
    emitter_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build both payloads, optionally replaying a hash-validated emitter path."""

    s1_payload = _replay_s1(s1_decision_path)
    s1_ref = _reference(s1_decision_path)
    emitter_ref = _reference(Path(__file__) if emitter_path is None else emitter_path)
    timestamp = _parse_utc(binding_time_utc)
    common = {
        "schema_version": SCHEMA,
        "artifact_kind": ARTIFACT_KIND,
        "passed": True,
        "binding_time_utc": timestamp,
        "statement": STATEMENT,
        "source_s1": s1_ref,
        "source_s1_selected_fields_sha256": s1_payload[
            "selected_fields_sha256"
        ],
        "emitter": emitter_ref,
    }
    s2 = _seal_payload(
        {
            **common,
            "stage": "s2",
            "operator": S2_OPERATOR,
            "decision": "operator_bind",
            "reason": S2_REASON,
            "selected_fields": dict(S2_SELECTED),
            "selected_fields_sha256": _digest_value(S2_SELECTED),
        }
    )
    s2_bytes = _serialized(s2)
    s2_ref = {
        "path": str(s2_output_path.expanduser().absolute()),
        "sha256": "sha256:" + hashlib.sha256(s2_bytes).hexdigest(),
    }
    s3 = _seal_payload(
        {
            **common,
            "stage": "s3",
            "operator": S3_OPERATOR,
            "decision": "operator_hold",
            "reason": S3_REASON,
            "selected_fields": dict(S3_SELECTED),
            "selected_fields_sha256": _digest_value(S3_SELECTED),
            "source_s2_binding": s2_ref,
        }
    )
    return s2, s3


def _write_read_only_new(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_serialized(payload))
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def write_bindings(
    s1_decision_path: Path,
    s2_output_path: Path,
    s3_output_path: Path,
    *,
    binding_time_utc: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create both outputs read-only, refusing any existing destination."""

    s2_output_path = s2_output_path.expanduser().absolute()
    s3_output_path = s3_output_path.expanduser().absolute()
    if s2_output_path == s3_output_path:
        raise BindingError("S2 and S3 output paths must differ")
    existing = [path for path in (s2_output_path, s3_output_path) if path.exists()]
    if existing:
        raise BindingError(f"refusing to overwrite existing output(s): {existing}")
    s2, s3 = build_bindings(
        s1_decision_path,
        s2_output_path=s2_output_path,
        binding_time_utc=binding_time_utc,
    )
    created: list[Path] = []
    try:
        _write_read_only_new(s2_output_path, s2)
        created.append(s2_output_path)
        _write_read_only_new(s3_output_path, s3)
        created.append(s3_output_path)
    except Exception:
        for path in reversed(created):
            path.chmod(0o600)
            path.unlink(missing_ok=True)
        raise
    return s2, s3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s1-decision", required=True)
    parser.add_argument("--s2-out", required=True)
    parser.add_argument("--s3-out", required=True)
    parser.add_argument(
        "--binding-time-utc",
        default=None,
        help="Optional explicit UTC ISO-8601 time, mainly for deterministic audits.",
    )
    args = parser.parse_args()
    try:
        s2, s3 = write_bindings(
            Path(args.s1_decision),
            Path(args.s2_out),
            Path(args.s3_out),
            binding_time_utc=args.binding_time_utc,
        )
    except (BindingError, OSError) as error:
        parser.exit(2, f"operator binding failed: {error}\n")
    print(
        json.dumps(
            {
                "s2": {
                    "path": str(Path(args.s2_out).expanduser().absolute()),
                    "artifact_content_sha256": s2["artifact_content_sha256"],
                },
                "s3": {
                    "path": str(Path(args.s3_out).expanduser().absolute()),
                    "artifact_content_sha256": s3["artifact_content_sha256"],
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
