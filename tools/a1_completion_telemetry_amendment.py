#!/usr/bin/env python3
"""Seal missing inference telemetry without rewriting historical authority.

This is a narrow recovery tool for a completed diagnostic whose immutable arm
manifest bound a completion CLI that accidentally dispatched through a generic
sibling finalizer.  It never rewrites that manifest or claims that different
code produced the training run.  Instead it binds the historical completion,
the exact candidate and reference checkpoints, matched bit-parity profiles,
and the source file containing the repaired dispatcher in an additive receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence


SCHEMA = "a1-completion-inference-telemetry-amendment-v1"
STATUS = "complete_nonpromotable_amended"


class AmendmentError(RuntimeError):
    """The historical completion cannot be amended safely."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _file_ref(path: Path) -> dict[str, str]:
    path = path.expanduser().resolve(strict=True)
    return {
        "path": str(path),
        "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _load(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.expanduser().resolve(strict=True).read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AmendmentError(f"cannot load {label}: {error}") from error
    if not isinstance(value, dict):
        raise AmendmentError(f"{label} is not a JSON object")
    return value


def _verify_embedded_digest(value: Mapping[str, Any], field: str, *, label: str) -> None:
    unhashed = dict(value)
    stated = unhashed.pop(field, None)
    if stated != _digest(unhashed):
        raise AmendmentError(f"{label} semantic digest mismatch")


def _metric(profile: Mapping[str, Any], *keys: str) -> float:
    value: Any = profile
    for key in keys:
        value = value.get(key) if isinstance(value, Mapping) else None
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise AmendmentError(f"invalid profile metric {'.'.join(keys)}={value!r}")
    return float(value)


def _profile(
    path: Path, *, checkpoint: Mapping[str, Any], environment: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, str]]:
    value = _load(path, label="inference profile")
    try:
        observed = _file_ref(Path(str(value["checkpoint"])))
    except (KeyError, OSError) as error:
        raise AmendmentError(f"profile checkpoint is unavailable: {error}") from error
    expected = {key: checkpoint.get(key) for key in ("path", "sha256")}
    if observed != expected:
        raise AmendmentError("profile does not bind the exact checkpoint bytes")
    parity = value.get("exact_vs_attributed_output_parity")
    if not (
        isinstance(parity, Mapping)
        and parity
        and all(
            isinstance(row, Mapping)
            and row.get("max_abs") == 0.0
            and row.get("mean_abs") == 0.0
            for row in parity.values()
        )
    ):
        raise AmendmentError("profile lacks bit-exact attributed parity")
    observed_environment = {
        "device": value.get("device"),
        "strict_fp32": value.get("strict_fp32"),
        "shape": {
            key: value.get("shape", {}).get(key)
            for key in ("batch_size", "legal_width", "event_width", "valid_players")
        },
        "warmup": value.get("warmup"),
        "iterations": value.get("iterations"),
        "return_q": value.get("return_q"),
    }
    if observed_environment != dict(environment):
        raise AmendmentError(
            f"profile environment drift: observed={observed_environment}"
        )
    return value, _file_ref(path)


def build_receipt(
    *,
    historical_completion: Path,
    manifest_path: Path,
    reference_profile: Path,
    candidate_profile: Path,
    dispatcher_fix: Path,
    created_at_unix_ns: int,
) -> dict[str, Any]:
    historical = _load(historical_completion, label="historical completion")
    _verify_embedded_digest(
        historical, "receipt_sha256", label="historical completion"
    )
    if not (
        historical.get("status") == "complete_nonpromotable"
        and historical.get("diagnostic_only") is True
        and historical.get("promotion_eligible") is False
    ):
        raise AmendmentError("historical completion is not non-promotable")

    manifest = _load(manifest_path, label="arm manifest")
    _verify_embedded_digest(manifest, "manifest_sha256", label="arm manifest")
    manifest_ref = _file_ref(manifest_path)
    if historical.get("manifest") != manifest_ref:
        raise AmendmentError("historical completion/manifest binding drift")
    historical_finalizer = historical.get("completion_finalizer")
    manifest_finalizer = manifest.get("completion_finalizer")
    if not (
        isinstance(historical_finalizer, Mapping)
        and isinstance(manifest_finalizer, Mapping)
        and {key: historical_finalizer.get(key) for key in ("path", "sha256")}
        == {key: manifest_finalizer.get(key) for key in ("path", "sha256")}
    ):
        raise AmendmentError("historical completion used an unauthorized finalizer")
    contract = manifest.get("inference_cost_contract")
    if not (
        isinstance(contract, Mapping)
        and contract.get("required_before_completion") is True
        and contract.get("strict_fp32") is True
    ):
        raise AmendmentError("arm manifest lacks mandatory inference-cost contract")

    candidate = historical.get("checkpoint")
    reference = contract.get("reference_checkpoint")
    if not isinstance(candidate, Mapping) or not isinstance(reference, Mapping):
        raise AmendmentError("checkpoint bindings are missing")
    candidate_ref = _file_ref(Path(str(candidate.get("path", ""))))
    reference_ref = _file_ref(Path(str(reference.get("path", ""))))
    if (
        candidate_ref
        != {key: candidate.get(key) for key in ("path", "sha256")}
        or reference_ref
        != {key: reference.get(key) for key in ("path", "sha256")}
    ):
        raise AmendmentError("checkpoint bytes differ from their sealed references")

    shape = contract.get("matched_shape")
    if not isinstance(shape, Mapping):
        raise AmendmentError("matched inference shape is missing")
    environment = {
        "device": None,
        "strict_fp32": {
            "matmul_precision": "highest",
            "cuda_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "autocast": False,
        },
        "shape": {
            key: shape.get(key)
            for key in ("batch_size", "legal_width", "event_width", "valid_players")
        },
        "warmup": shape.get("warmup"),
        "iterations": shape.get("iterations"),
        "return_q": shape.get("return_q"),
    }
    # Device names vary by driver formatting, but must match one another.
    raw_reference = _load(reference_profile, label="reference inference profile")
    environment["device"] = raw_reference.get("device")
    reference_value, reference_profile_ref = _profile(
        reference_profile, checkpoint=reference_ref, environment=environment
    )
    candidate_value, candidate_profile_ref = _profile(
        candidate_profile, checkpoint=candidate_ref, environment=environment
    )

    paths = {
        "cuda_mean_ms": ("exact_window", "cuda_ms", "mean"),
        "cuda_median_ms": ("exact_window", "cuda_ms", "median"),
        "cuda_p95_ms": ("exact_window", "cuda_ms", "p95"),
        "wall_mean_ms": ("exact_window", "wall_ms", "mean"),
        "wall_median_ms": ("exact_window", "wall_ms", "median"),
        "wall_p95_ms": ("exact_window", "wall_ms", "p95"),
    }
    reference_metrics = {
        name: _metric(reference_value, *keys) for name, keys in paths.items()
    }
    candidate_metrics = {
        name: _metric(candidate_value, *keys) for name, keys in paths.items()
    }
    ratios = {
        name.removesuffix("_ms") + "_slowdown": candidate_metrics[name] / value
        for name, value in reference_metrics.items()
    }
    telemetry = {
        "schema_version": "a1-architecture-inference-cost-telemetry-v1",
        "contract": dict(contract),
        "reference_checkpoint": reference_ref,
        "candidate_checkpoint": candidate_ref,
        "reference_profile": reference_profile_ref,
        "candidate_profile": candidate_profile_ref,
        "matched_environment": environment,
        "reference_metrics": reference_metrics,
        "candidate_metrics": candidate_metrics,
        "candidate_reference_ratios": ratios,
        "selection_cost_observed": True,
    }
    telemetry["telemetry_sha256"] = _digest(telemetry)
    fix_ref = _file_ref(dispatcher_fix)
    fix_source = dispatcher_fix.read_text(encoding="utf-8")
    if "value = finalize(" not in fix_source or "return base.main(argv)" in fix_source:
        raise AmendmentError("dispatcher fix does not contain the specialized dispatch")

    receipt = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "created_at_unix_ns": int(created_at_unix_ns),
        "defect": "specialized completion CLI delegated to generic sibling main and omitted mandatory inference telemetry",
        "repair_semantics": "additive receipt; historical training authority and bytes are unchanged",
        "historical_completion": _file_ref(historical_completion),
        "historical_completion_receipt_sha256": historical["receipt_sha256"],
        "manifest": manifest_ref,
        "candidate_checkpoint": candidate_ref,
        "dispatcher_fix": fix_ref,
        "inference_cost_telemetry": telemetry,
    }
    receipt["receipt_sha256"] = _digest(receipt)
    return receipt


def create(args: argparse.Namespace) -> dict[str, Any]:
    value = build_receipt(
        historical_completion=args.historical_completion,
        manifest_path=args.manifest,
        reference_profile=args.reference_profile,
        candidate_profile=args.candidate_profile,
        dispatcher_fix=args.dispatcher_fix,
        created_at_unix_ns=time.time_ns(),
    )
    target = args.out.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(target, flags, 0o400)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return value


def verify(path: Path) -> dict[str, Any]:
    value = _load(path, label="amendment receipt")
    _verify_embedded_digest(value, "receipt_sha256", label="amendment receipt")
    if not (
        value.get("schema_version") == SCHEMA
        and value.get("status") == STATUS
        and value.get("diagnostic_only") is True
        and value.get("promotion_eligible") is False
    ):
        raise AmendmentError("amendment schema/status drift")
    replay = build_receipt(
        historical_completion=Path(value["historical_completion"]["path"]),
        manifest_path=Path(value["manifest"]["path"]),
        reference_profile=Path(
            value["inference_cost_telemetry"]["reference_profile"]["path"]
        ),
        candidate_profile=Path(
            value["inference_cost_telemetry"]["candidate_profile"]["path"]
        ),
        dispatcher_fix=Path(value["dispatcher_fix"]["path"]),
        created_at_unix_ns=int(value["created_at_unix_ns"]),
    )
    if replay != value:
        raise AmendmentError("amendment replay differs from receipt")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    seal = actions.add_parser("create")
    seal.add_argument("--historical-completion", required=True, type=Path)
    seal.add_argument("--manifest", required=True, type=Path)
    seal.add_argument("--reference-profile", required=True, type=Path)
    seal.add_argument("--candidate-profile", required=True, type=Path)
    seal.add_argument("--dispatcher-fix", required=True, type=Path)
    seal.add_argument("--out", required=True, type=Path)
    replay = actions.add_parser("verify")
    replay.add_argument("--receipt", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        value = create(args) if args.action == "create" else verify(args.receipt)
    except (AmendmentError, OSError, ValueError, KeyError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
