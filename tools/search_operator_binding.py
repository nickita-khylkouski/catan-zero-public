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
from tools import a1_post_promotion_handoff as promotion_handoff  # noqa: E402


SCHEMA = "rl-rnd-operator-binding-v1"
POST_PROMOTION_S1_SCHEMA = "rl-rnd-post-promotion-s1-operator-binding-v1"
RECOVERY_S1_SCHEMA = "rl-rnd-recovery-s1-operator-binding-v1"
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
POST_PROMOTION_S1_OPERATOR = "deployed_c_scale_continuity"
POST_PROMOTION_S1_REASON = (
    "committed_post_promotion_handoff_requires_c_scale_continuity"
)
POST_PROMOTION_S1_STATEMENT = (
    "This artifact records only the deployed c_scale continuity required by "
    "the committed post-promotion handoff; it is an operator choice, not "
    "strength evidence, and changes no other replayed S1 field."
)
POST_PROMOTION_S1_OVERRIDE = {
    "authorization": "committed_post_promotion_handoff_deployed_identity",
    "deployed_value": 0.1,
    "field": "c_scale",
    "legacy_value": 0.03,
}
RECOVERY_S1_OVERRIDE = {
    "authorization": "authenticated_disaster_recovery_deployed_identity",
    "deployed_value": 0.1,
    "field": "c_scale",
    "legacy_value": 0.03,
}
S1_SELECTED_KEYS = {
    "c_scale",
    "rescale_noise_floor_c",
    "sigma_eval",
    "symmetry_averaged_eval",
    "symmetry_averaged_eval_threshold",
}
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


def _resolve_reference(
    raw: Any, *, owner_path: Path, where: str
) -> tuple[Path, dict[str, str]]:
    if not isinstance(raw, dict) or set(raw) != {"path", "sha256"}:
        raise BindingError(f"{where} must contain exactly path and sha256")
    candidate = Path(str(raw["path"])).expanduser()
    if not candidate.is_absolute():
        candidate = owner_path.parent / candidate
    candidate = candidate.resolve(strict=True)
    actual = _sha256(candidate)
    if actual != raw["sha256"]:
        raise BindingError(f"{where} hash drift at {candidate}")
    return candidate, {"path": str(candidate), "sha256": actual}


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


def _replay_post_promotion_handoff(path: Path) -> dict[str, Any]:
    handoff_path = path.expanduser().resolve(strict=True)
    payload = _load_json(handoff_path)
    if payload.get("schema_version") != promotion_handoff.HANDOFF_SCHEMA:
        raise BindingError("post-promotion handoff schema is unsupported")
    unhashed = dict(payload)
    declared = unhashed.pop("handoff_sha256", None)
    if declared != _digest_value(unhashed):
        raise BindingError("post-promotion handoff semantic digest mismatch")
    receipt = payload.get("promotion_receipt")
    if not isinstance(receipt, dict) or not isinstance(receipt.get("path"), str):
        raise BindingError("post-promotion handoff has no promotion receipt")
    try:
        replayed = promotion_handoff.build_handoff(Path(receipt["path"]))
    except promotion_handoff.HandoffError as error:
        raise BindingError(f"post-promotion handoff replay failed: {error}") from error
    if replayed != payload:
        raise BindingError("post-promotion handoff differs from committed live lineage")
    return payload


def _replay_recovery_receipt(path: Path) -> dict[str, Any]:
    # Lazy import avoids search_operator_binding -> recovery -> pre_wave ->
    # search_operator_binding during module initialization.
    from tools import a1_v5_disaster_recovery as recovery

    receipt_path = path.expanduser().resolve(strict=True)
    try:
        verified = recovery.verify_committed_receipt(receipt_path)
    except recovery.RecoveryError as error:
        raise BindingError(f"disaster-recovery receipt replay failed: {error}") from error
    authority = verified.get("authority")
    receipt = verified.get("receipt")
    if not isinstance(authority, dict) or not isinstance(receipt, dict):
        raise BindingError("disaster-recovery verifier returned malformed authority")
    return {"authority": authority, "receipt": receipt}


def build_post_promotion_s1_binding(
    legacy_s1_decision_path: Path,
    post_promotion_handoff_path: Path,
    *,
    binding_time_utc: str | None = None,
    emitter_path: Path | None = None,
) -> dict[str, Any]:
    """Project exactly one handoff-authorized c_scale continuity override."""

    legacy = _replay_s1(legacy_s1_decision_path)
    legacy_selected = legacy.get("selected_fields")
    if not isinstance(legacy_selected, dict) or set(legacy_selected) != S1_SELECTED_KEYS:
        raise BindingError("legacy S1 selected_fields have an unsupported shape")
    if legacy_selected.get("c_scale") != POST_PROMOTION_S1_OVERRIDE["legacy_value"]:
        raise BindingError("post-promotion S1 bridge requires replayed legacy c_scale=.03")

    handoff_path = post_promotion_handoff_path.expanduser().resolve(strict=True)
    handoff = _replay_post_promotion_handoff(handoff_path)
    identity = handoff.get("producer_identity")
    if not isinstance(identity, dict):
        raise BindingError("post-promotion handoff has no producer identity")
    checkpoint = identity.get("checkpoint")
    search_config = identity.get("search_config")
    if (
        not isinstance(checkpoint, dict)
        or set(checkpoint) != {"path", "sha256"}
        or not isinstance(search_config, dict)
    ):
        raise BindingError("post-promotion producer identity is malformed")
    checkpoint_path = Path(str(checkpoint["path"])).expanduser().resolve(strict=True)
    if _sha256(checkpoint_path) != checkpoint["sha256"]:
        raise BindingError("post-promotion producer checkpoint hash drift")
    if handoff.get("registry_after", {}).get("checkpoint") != checkpoint:
        raise BindingError("post-promotion registry and producer checkpoint differ")
    deployed_c_scale = search_config.get("c_scale")
    if deployed_c_scale != POST_PROMOTION_S1_OVERRIDE["deployed_value"]:
        raise BindingError("post-promotion S1 bridge permits only deployed c_scale=.10")
    for key in S1_SELECTED_KEYS - {"c_scale"}:
        if search_config.get(key) != legacy_selected[key]:
            raise BindingError(
                f"post-promotion producer changes non-continuity S1 field {key}"
            )

    selected = dict(legacy_selected)
    selected["c_scale"] = deployed_c_scale
    timestamp = _parse_utc(binding_time_utc)
    emitter_ref = _reference(Path(__file__) if emitter_path is None else emitter_path)
    source_handoff = _reference(handoff_path)
    source_handoff["handoff_sha256"] = handoff["handoff_sha256"]
    return _seal_payload(
        {
            "schema_version": POST_PROMOTION_S1_SCHEMA,
            "artifact_kind": ARTIFACT_KIND,
            "stage": "s1",
            "operator": POST_PROMOTION_S1_OPERATOR,
            "passed": True,
            "decision": "operator_bind",
            "reason": POST_PROMOTION_S1_REASON,
            "binding_time_utc": timestamp,
            "statement": POST_PROMOTION_S1_STATEMENT,
            "selected_fields": selected,
            "selected_fields_sha256": _digest_value(selected),
            "source_legacy_s1": _reference(legacy_s1_decision_path),
            "source_legacy_s1_selected_fields_sha256": legacy[
                "selected_fields_sha256"
            ],
            "source_post_promotion_handoff": source_handoff,
            "producer_checkpoint": dict(checkpoint),
            "producer_identity_sha256": identity.get("agent_identity_sha256"),
            "producer_search_config_sha256": _digest_value(search_config),
            "continuity_override": dict(POST_PROMOTION_S1_OVERRIDE),
            "emitter": emitter_ref,
        }
    )


def _replay_post_promotion_s1(path: Path) -> dict[str, Any]:
    binding_path = path.expanduser().resolve(strict=True)
    payload = _load_json(binding_path)
    if payload.get("schema_version") != POST_PROMOTION_S1_SCHEMA:
        raise BindingError("post-promotion S1 operator-binding schema mismatch")
    legacy_path, _ = _resolve_reference(
        payload.get("source_legacy_s1"),
        owner_path=binding_path,
        where="post-promotion S1 legacy source",
    )
    handoff_ref = payload.get("source_post_promotion_handoff")
    if not isinstance(handoff_ref, dict) or set(handoff_ref) != {
        "path",
        "sha256",
        "handoff_sha256",
    }:
        raise BindingError("post-promotion S1 handoff reference is malformed")
    handoff_path, _ = _resolve_reference(
        {"path": handoff_ref["path"], "sha256": handoff_ref["sha256"]},
        owner_path=binding_path,
        where="post-promotion S1 handoff source",
    )
    emitter_path, _ = _resolve_reference(
        payload.get("emitter"),
        owner_path=binding_path,
        where="post-promotion S1 emitter",
    )
    replayed = build_post_promotion_s1_binding(
        legacy_path,
        handoff_path,
        binding_time_utc=payload.get("binding_time_utc"),
        emitter_path=emitter_path,
    )
    if replayed != payload:
        raise BindingError("post-promotion S1 binding does not equal semantic replay")
    return payload


def build_recovery_s1_binding(
    legacy_s1_decision_path: Path,
    recovery_receipt_path: Path,
    *,
    binding_time_utc: str | None = None,
    emitter_path: Path | None = None,
) -> dict[str, Any]:
    """Bind c_scale continuity from authenticated recovery, without promotion claims."""

    legacy = _replay_s1(legacy_s1_decision_path)
    legacy_selected = legacy.get("selected_fields")
    if not isinstance(legacy_selected, dict) or set(legacy_selected) != S1_SELECTED_KEYS:
        raise BindingError("legacy S1 selected_fields have an unsupported shape")
    if legacy_selected.get("c_scale") != RECOVERY_S1_OVERRIDE["legacy_value"]:
        raise BindingError("recovery S1 bridge requires replayed legacy c_scale=.03")
    receipt_path = recovery_receipt_path.expanduser().resolve(strict=True)
    verified = _replay_recovery_receipt(receipt_path)
    authority = verified["authority"]
    receipt = verified["receipt"]
    identity = authority.get("producer_identity")
    checkpoint = authority.get("recovered_generator")
    if not isinstance(identity, dict) or not isinstance(checkpoint, dict):
        raise BindingError("recovery authority has no producer identity")
    search_config = identity.get("search_config")
    checkpoint_ref = {
        "path": checkpoint.get("path"),
        "sha256": checkpoint.get("sha256"),
    }
    if (
        not isinstance(search_config, dict)
        or set(checkpoint_ref) != {"path", "sha256"}
        or identity.get("checkpoint") != checkpoint_ref
    ):
        raise BindingError("recovery producer identity is malformed")
    checkpoint_path = Path(str(checkpoint_ref["path"])).expanduser().resolve(
        strict=True
    )
    if _sha256(checkpoint_path) != checkpoint_ref["sha256"]:
        raise BindingError("recovery producer checkpoint hash drift")
    deployed_c_scale = search_config.get("c_scale")
    if deployed_c_scale != RECOVERY_S1_OVERRIDE["deployed_value"]:
        raise BindingError("recovery S1 bridge permits only deployed c_scale=.10")
    for key in S1_SELECTED_KEYS - {"c_scale"}:
        if search_config.get(key) != legacy_selected[key]:
            raise BindingError(f"recovery producer changes non-continuity S1 field {key}")
    selected = dict(legacy_selected)
    selected["c_scale"] = deployed_c_scale
    source_receipt = _reference(receipt_path)
    source_receipt.update(
        {
            "recovery_receipt_sha256": receipt["recovery_receipt_sha256"],
            "recovery_lineage_id": authority["recovery_lineage_id"],
        }
    )
    return _seal_payload(
        {
            "schema_version": RECOVERY_S1_SCHEMA,
            "artifact_kind": ARTIFACT_KIND,
            "stage": "s1",
            "operator": POST_PROMOTION_S1_OPERATOR,
            "passed": True,
            "decision": "operator_bind",
            "reason": "authenticated_disaster_recovery_requires_c_scale_continuity",
            "binding_time_utc": _parse_utc(binding_time_utc),
            "statement": (
                "This artifact binds only the recovered deployed c_scale. It does "
                "not recreate promotion evidence or claim f7 was the causal parent."
            ),
            "selected_fields": selected,
            "selected_fields_sha256": _digest_value(selected),
            "source_legacy_s1": _reference(legacy_s1_decision_path),
            "source_legacy_s1_selected_fields_sha256": legacy[
                "selected_fields_sha256"
            ],
            "source_recovery_receipt": source_receipt,
            "producer_checkpoint": checkpoint_ref,
            "producer_identity_sha256": identity.get("agent_identity_sha256"),
            "producer_search_config_sha256": _digest_value(search_config),
            "continuity_override": dict(RECOVERY_S1_OVERRIDE),
            "promotion_proof_recreated": False,
            "emitter": _reference(Path(__file__) if emitter_path is None else emitter_path),
        }
    )


def _replay_recovery_s1(path: Path) -> dict[str, Any]:
    binding_path = path.expanduser().resolve(strict=True)
    payload = _load_json(binding_path)
    if payload.get("schema_version") != RECOVERY_S1_SCHEMA:
        raise BindingError("recovery S1 operator-binding schema mismatch")
    legacy_path, _ = _resolve_reference(
        payload.get("source_legacy_s1"),
        owner_path=binding_path,
        where="recovery S1 legacy source",
    )
    receipt_ref = payload.get("source_recovery_receipt")
    if not isinstance(receipt_ref, dict) or set(receipt_ref) != {
        "path",
        "sha256",
        "recovery_receipt_sha256",
        "recovery_lineage_id",
    }:
        raise BindingError("recovery S1 receipt reference is malformed")
    receipt_path, _ = _resolve_reference(
        {"path": receipt_ref["path"], "sha256": receipt_ref["sha256"]},
        owner_path=binding_path,
        where="recovery S1 receipt source",
    )
    emitter_path, _ = _resolve_reference(
        payload.get("emitter"), owner_path=binding_path, where="recovery S1 emitter"
    )
    replayed = build_recovery_s1_binding(
        legacy_path,
        receipt_path,
        binding_time_utc=payload.get("binding_time_utc"),
        emitter_path=emitter_path,
    )
    if replayed != payload:
        raise BindingError("recovery S1 binding does not equal semantic replay")
    return payload


def _replay_source_s1(path: Path) -> dict[str, Any]:
    payload = _load_json(path.expanduser().resolve(strict=True))
    if payload.get("schema_version") == POST_PROMOTION_S1_SCHEMA:
        return _replay_post_promotion_s1(path)
    if payload.get("schema_version") == RECOVERY_S1_SCHEMA:
        return _replay_recovery_s1(path)
    return _replay_s1(path)


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

    s1_payload = _replay_source_s1(s1_decision_path)
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
    emitter_path: Path | None = None,
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
        emitter_path=emitter_path,
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


def write_post_promotion_bindings(
    legacy_s1_decision_path: Path,
    post_promotion_handoff_path: Path,
    s1_output_path: Path,
    s2_output_path: Path,
    s3_output_path: Path,
    *,
    binding_time_utc: str | None = None,
    emitter_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Create one S1 continuity binding and fresh S2/S3 descendants atomically."""

    outputs = [
        path.expanduser().absolute()
        for path in (s1_output_path, s2_output_path, s3_output_path)
    ]
    if len(set(outputs)) != len(outputs):
        raise BindingError("S1, S2, and S3 output paths must differ")
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise BindingError(f"refusing to overwrite existing output(s): {existing}")
    timestamp = _parse_utc(binding_time_utc)
    s1 = build_post_promotion_s1_binding(
        legacy_s1_decision_path,
        post_promotion_handoff_path,
        binding_time_utc=timestamp,
        emitter_path=emitter_path,
    )
    created: list[Path] = []
    try:
        _write_read_only_new(outputs[0], s1)
        created.append(outputs[0])
        s2, s3 = build_bindings(
            outputs[0],
            s2_output_path=outputs[1],
            binding_time_utc=timestamp,
            emitter_path=emitter_path,
        )
        _write_read_only_new(outputs[1], s2)
        created.append(outputs[1])
        _write_read_only_new(outputs[2], s3)
        created.append(outputs[2])
    except Exception:
        for path in reversed(created):
            path.chmod(0o600)
            path.unlink(missing_ok=True)
        raise
    return s1, s2, s3


def write_recovery_bindings(
    legacy_s1_decision_path: Path,
    recovery_receipt_path: Path,
    s1_output_path: Path,
    s2_output_path: Path,
    s3_output_path: Path,
    *,
    binding_time_utc: str | None = None,
    emitter_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    outputs = [
        path.expanduser().absolute()
        for path in (s1_output_path, s2_output_path, s3_output_path)
    ]
    if len(set(outputs)) != len(outputs) or any(path.exists() for path in outputs):
        raise BindingError("recovery S1/S2/S3 outputs must be distinct and fresh")
    timestamp = _parse_utc(binding_time_utc)
    s1 = build_recovery_s1_binding(
        legacy_s1_decision_path,
        recovery_receipt_path,
        binding_time_utc=timestamp,
        emitter_path=emitter_path,
    )
    created: list[Path] = []
    try:
        _write_read_only_new(outputs[0], s1)
        created.append(outputs[0])
        s2, s3 = build_bindings(
            outputs[0],
            s2_output_path=outputs[1],
            binding_time_utc=timestamp,
            emitter_path=emitter_path,
        )
        _write_read_only_new(outputs[1], s2)
        created.append(outputs[1])
        _write_read_only_new(outputs[2], s3)
        created.append(outputs[2])
    except Exception:
        for path in reversed(created):
            path.chmod(0o600)
            path.unlink(missing_ok=True)
        raise
    return s1, s2, s3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--s1-decision")
    source.add_argument("--legacy-s1-decision")
    parser.add_argument("--post-promotion-handoff")
    parser.add_argument("--recovery-receipt")
    parser.add_argument("--s1-out")
    parser.add_argument(
        "--emitter-path",
        default=None,
        help="Optional immutable copy of this emitter to bind for future replay.",
    )
    parser.add_argument("--s2-out", required=True)
    parser.add_argument("--s3-out", required=True)
    parser.add_argument(
        "--binding-time-utc",
        default=None,
        help="Optional explicit UTC ISO-8601 time, mainly for deterministic audits.",
    )
    args = parser.parse_args()
    try:
        if args.legacy_s1_decision is not None:
            if (
                (args.post_promotion_handoff is None)
                == (args.recovery_receipt is None)
                or args.s1_out is None
            ):
                raise BindingError(
                    "--legacy-s1-decision requires exactly one lineage source and --s1-out"
                )
            writer = (
                write_post_promotion_bindings
                if args.post_promotion_handoff is not None
                else write_recovery_bindings
            )
            lineage_source = args.post_promotion_handoff or args.recovery_receipt
            s1, s2, s3 = writer(
                Path(args.legacy_s1_decision), Path(str(lineage_source)),
                Path(args.s1_out), Path(args.s2_out), Path(args.s3_out),
                binding_time_utc=args.binding_time_utc,
                emitter_path=None if args.emitter_path is None else Path(args.emitter_path),
            )
        else:
            if (
                args.post_promotion_handoff is not None
                or args.recovery_receipt is not None
                or args.s1_out is not None
            ):
                raise BindingError(
                    "--post-promotion-handoff/--s1-out require --legacy-s1-decision"
                )
            s1 = None
            s2, s3 = write_bindings(
                Path(args.s1_decision),
                Path(args.s2_out),
                Path(args.s3_out),
                binding_time_utc=args.binding_time_utc,
                emitter_path=(
                    None if args.emitter_path is None else Path(args.emitter_path)
                ),
            )
    except (BindingError, OSError) as error:
        parser.exit(2, f"operator binding failed: {error}\n")
    outputs: dict[str, Any] = {}
    if s1 is not None:
        outputs["s1"] = {
            "path": str(Path(args.s1_out).expanduser().absolute()),
            "artifact_content_sha256": s1["artifact_content_sha256"],
        }
    outputs.update(
        {
            "s2": {
                "path": str(Path(args.s2_out).expanduser().absolute()),
                "artifact_content_sha256": s2["artifact_content_sha256"],
            },
            "s3": {
                "path": str(Path(args.s3_out).expanduser().absolute()),
                "artifact_content_sha256": s3["artifact_content_sha256"],
            },
        }
    )
    print(
        json.dumps(outputs, indent=2, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
