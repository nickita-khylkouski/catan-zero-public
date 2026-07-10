#!/usr/bin/env python3
"""Create and verify a typed bridge for legacy scalar-value checkpoints.

Modern checkpoints carry ``value-training-v1`` inside the checkpoint itself.
The production gen3 checkpoint predates that envelope, while its immutable
training report still proves that the scalar value head received a positive
loss.  This tool binds those two exact files without rewriting checkpoint
bytes.  It is deliberately scalar-only: it can never authorize a categorical
readout, which continues to require embedded positive training provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SCHEMA_VERSION = "legacy-scalar-readout-attestation-v1"


class AttestationError(ValueError):
    """The legacy checkpoint/report pair does not prove scalar training."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()


def _digest_value(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AttestationError(f"cannot read report JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise AttestationError(f"report {path} must contain a JSON object")
    return payload


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        import torch

        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # pragma: no cover - compatibility with older torch.
            payload = torch.load(path, map_location="cpu")
    except Exception as error:  # noqa: BLE001 - arbitrary checkpoint failures must fail closed.
        raise AttestationError(f"cannot inspect checkpoint {path}: {error}") from error
    if not isinstance(payload, dict):
        raise AttestationError(f"checkpoint {path} is not a mapping")
    return payload


def _require_file(path: Path, *, kind: str) -> Path:
    path = path.expanduser().absolute()
    if not path.is_file():
        raise AttestationError(f"{kind} is missing or not a file: {path}")
    return path


def _finite_number(value: Any, *, where: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AttestationError(f"{where} must be numeric")
    number = float(value)
    if (
        not math.isfinite(number)
        or (positive and number <= 0.0)
        or (not positive and number < 0.0)
    ):
        qualifier = "positive finite" if positive else "finite and non-negative"
        raise AttestationError(f"{where} must be {qualifier}, got {value!r}")
    return number


def _report_declares_checkpoint(declared: Any, checkpoint: Path) -> str:
    if not isinstance(declared, str) or not declared.strip():
        raise AttestationError("report.checkpoint must name the trained checkpoint")
    raw = Path(declared).expanduser()
    actual = checkpoint.resolve(strict=True)
    if raw.is_absolute():
        if raw.resolve(strict=False) != actual:
            raise AttestationError(
                f"report checkpoint identity mismatch: report={declared!r}, actual={checkpoint}"
            )
        return declared

    # Historical train_bc reports record repo-relative paths such as
    # runs/bc/gen3_20260706/checkpoint.pt.  The original process cwd is not
    # serialized, so compare the complete multi-component suffix rather than
    # guessing a cwd.  A bare basename is intentionally too weak.
    clean_parts = tuple(part for part in raw.parts if part not in {".", ""})
    if ".." in clean_parts or len(clean_parts) < 2:
        raise AttestationError(
            "relative report.checkpoint must be a non-traversing multi-component path"
        )
    if tuple(actual.parts[-len(clean_parts) :]) != clean_parts:
        raise AttestationError(
            f"report checkpoint identity mismatch: report={declared!r}, actual={checkpoint}"
        )
    return declared


def _categorical_bins(checkpoint: dict[str, Any]) -> int:
    if "config" not in checkpoint:
        raise AttestationError("checkpoint has no model config")
    try:
        from catan_zero.rl.config_serialization import config_attr_view

        config = config_attr_view(checkpoint["config"])
        bins = int(getattr(config, "value_categorical_bins", 0) or 0)
    except Exception as error:  # noqa: BLE001 - an unreadable config cannot prove a claim.
        raise AttestationError(
            f"cannot inspect checkpoint model config: {error}"
        ) from error
    if bins != 0:
        raise AttestationError(
            "legacy scalar attestation requires a model config with no categorical head; "
            f"value_categorical_bins={bins}"
        )
    return bins


def _reject_categorical_claims(
    checkpoint: dict[str, Any], report: dict[str, Any]
) -> None:
    # An embedded provenance envelope should be consumed directly by the
    # pre-wave contract rather than converted into a weaker legacy bridge.
    if "value_training" in checkpoint:
        raise AttestationError(
            "checkpoint already contains value_training provenance; legacy bridge is not applicable"
        )
    echoed = checkpoint.get("trained_value_readouts")
    if isinstance(echoed, (list, tuple)) and "categorical" in set(map(str, echoed)):
        raise AttestationError(
            "checkpoint contradicts scalar-only provenance with categorical readout"
        )

    head_type = str(report.get("value_head_type", "scalar") or "scalar").lower()
    if head_type not in {"scalar", "mse"}:
        raise AttestationError(
            f"report has contradictory categorical objective {head_type!r}"
        )
    for key in (
        "value_categorical_loss_weight",
        "resolved_categorical_value_loss_weight",
        "hlgauss_scalar_aux_loss_weight",
    ):
        if key in report and _finite_number(report[key], where=f"report.{key}") != 0.0:
            raise AttestationError(
                f"report has contradictory nonzero categorical objective {key}"
            )
    if int(report.get("value_categorical_bins", 0) or 0) != 0:
        raise AttestationError("report has contradictory categorical value bins")
    if "resolved_scalar_value_loss_weight" in report and _finite_number(
        report["resolved_scalar_value_loss_weight"],
        where="report.resolved_scalar_value_loss_weight",
    ) <= 0.0:
        raise AttestationError(
            "report contradicts scalar training with non-positive resolved scalar weight"
        )
    value_training = report.get("value_training")
    if isinstance(value_training, dict):
        trained = set(map(str, value_training.get("trained_value_readouts", [])))
        if (
            str(value_training.get("primary_readout", "scalar")) == "categorical"
            or "categorical" in trained
            or float(value_training.get("resolved_categorical_ce_weight", 0.0) or 0.0)
            != 0.0
        ):
            raise AttestationError(
                "report value_training contradicts scalar-only provenance"
            )


def _value_loss_telemetry(report: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = report.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise AttestationError("report must contain non-empty per-epoch metrics")
    if int(report.get("steps_completed", 0) or 0) <= 0:
        raise AttestationError("report must attest positive steps_completed")
    epochs = int(report.get("epochs", 0) or 0)
    if epochs <= 0 or epochs != len(metrics):
        raise AttestationError(
            "report epochs must be positive and equal the completed metric count"
        )
    if int(report["steps_completed"]) < epochs:
        raise AttestationError("report steps_completed is inconsistent with epochs")

    telemetry: list[dict[str, Any]] = []
    for index, metric in enumerate(metrics):
        if not isinstance(metric, dict) or "value_loss" not in metric:
            raise AttestationError(
                f"report.metrics[{index}] lacks actual value_loss telemetry"
            )
        row: dict[str, Any] = {
            "metric_index": index,
            "train_value_loss": _finite_number(
                metric["value_loss"], where=f"report.metrics[{index}].value_loss"
            ),
        }
        if int(metric.get("epoch", -1)) != index + 1:
            raise AttestationError(
                f"report.metrics[{index}].epoch is not the completed epoch sequence"
            )
        validation = metric.get("validation")
        if not isinstance(validation, dict) or "value_loss" not in validation:
            raise AttestationError(
                f"report.metrics[{index}].validation lacks value_loss telemetry"
            )
        row["validation_value_loss"] = _finite_number(
            validation["value_loss"],
            where=f"report.metrics[{index}].validation.value_loss",
        )
        telemetry.append(row)
    return telemetry


def build_attestation(checkpoint_path: Path, report_path: Path) -> dict[str, Any]:
    """Validate exact legacy artifacts and return their canonical attestation."""

    checkpoint_path = _require_file(checkpoint_path, kind="checkpoint")
    report_path = _require_file(report_path, kind="training report")
    checkpoint = _load_checkpoint(checkpoint_path)
    report = _load_json(report_path)

    if checkpoint.get("policy_type") != "entity_graph":
        raise AttestationError("legacy checkpoint must identify policy_type=entity_graph")
    if report.get("arch") != "entity_graph":
        raise AttestationError("legacy report must identify arch=entity_graph")
    if checkpoint.get("mask_hidden_info") is not True:
        raise AttestationError("checkpoint does not attest mask_hidden_info=true")
    if report.get("mask_hidden_info") is not True:
        raise AttestationError("report does not attest mask_hidden_info=true")
    declared_checkpoint = _report_declares_checkpoint(
        report.get("checkpoint"), checkpoint_path
    )
    bins = _categorical_bins(checkpoint)
    _reject_categorical_claims(checkpoint, report)
    scalar_weight = _finite_number(
        report.get("value_loss_weight"), where="report.value_loss_weight", positive=True
    )
    telemetry = _value_loss_telemetry(report)

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
        },
        "report": {
            "path": str(report_path),
            "sha256": _sha256(report_path),
            "declared_checkpoint": declared_checkpoint,
        },
        "claims": {
            "checkpoint_report_identity": True,
            "mask_hidden_info": True,
            "value_readout": "scalar",
            "checkpoint_value_categorical_bins": bins,
            "report_value_loss_weight": scalar_weight,
            "steps_completed": int(report["steps_completed"]),
            "value_loss_telemetry": telemetry,
            "value_loss_telemetry_sha256": _digest_value(telemetry),
            "no_contradictory_categorical_objective": True,
        },
    }
    payload["attestation_sha256"] = _digest_value(payload)
    return payload


def verify_attestation(
    attestation_path: Path,
    *,
    expected_checkpoint_path: Path | None = None,
    expected_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    """Recompute every source claim and verify an immutable attestation file."""

    attestation_path = _require_file(attestation_path, kind="legacy scalar attestation")
    try:
        payload = json.loads(attestation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AttestationError(
            f"cannot read attestation {attestation_path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise AttestationError("legacy scalar attestation must be a JSON object")
    if set(payload) != {
        "schema_version",
        "checkpoint",
        "report",
        "claims",
        "attestation_sha256",
    }:
        raise AttestationError(
            "legacy scalar attestation fields do not match the typed schema"
        )
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise AttestationError(
            f"legacy scalar attestation schema must be {SCHEMA_VERSION!r}"
        )
    expected_digest = str(payload.get("attestation_sha256", ""))
    unhashed = dict(payload)
    unhashed.pop("attestation_sha256", None)
    if expected_digest != _digest_value(unhashed):
        raise AttestationError("legacy scalar attestation digest mismatch")
    checkpoint_record = payload.get("checkpoint")
    report_record = payload.get("report")
    if not isinstance(checkpoint_record, dict) or set(checkpoint_record) != {
        "path",
        "sha256",
    }:
        raise AttestationError(
            "attestation checkpoint record must contain exactly path and sha256"
        )
    if not isinstance(report_record, dict) or set(report_record) != {
        "path",
        "sha256",
        "declared_checkpoint",
    }:
        raise AttestationError(
            "attestation report record must contain path, sha256, and declared_checkpoint"
        )
    checkpoint_path = _require_file(
        Path(str(checkpoint_record["path"])), kind="checkpoint"
    )
    report_path = _require_file(
        Path(str(report_record["path"])), kind="training report"
    )
    if _sha256(checkpoint_path) != checkpoint_record["sha256"]:
        raise AttestationError("legacy scalar attestation checkpoint hash drift")
    if _sha256(report_path) != report_record["sha256"]:
        raise AttestationError("legacy scalar attestation report hash drift")
    if expected_checkpoint_path is not None:
        expected_path = _require_file(
            expected_checkpoint_path, kind="expected checkpoint"
        )
        if checkpoint_path.resolve(strict=True) != expected_path.resolve(strict=True):
            raise AttestationError(
                "legacy scalar attestation binds the wrong checkpoint path"
            )
    if (
        expected_checkpoint_sha256 is not None
        and checkpoint_record["sha256"] != expected_checkpoint_sha256
    ):
        raise AttestationError(
            "legacy scalar attestation binds the wrong checkpoint hash"
        )

    rebuilt = build_attestation(checkpoint_path, report_path)
    if payload != rebuilt:
        raise AttestationError(
            "legacy scalar attestation semantic claims do not reconstruct"
        )
    return payload


def write_attestation(
    checkpoint_path: Path, report_path: Path, output_path: Path
) -> dict[str, Any]:
    """Create an exclusive, fsynced, read-only attestation JSON file."""

    payload = build_attestation(checkpoint_path, report_path)
    output_path = output_path.expanduser().absolute()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        descriptor = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise AttestationError(
            f"refusing to overwrite existing attestation {output_path}"
        ) from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        output_path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    except Exception:
        output_path.unlink(missing_ok=True)
        raise
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser(
        "create", help="validate and write an immutable attestation"
    )
    create.add_argument("--checkpoint", type=Path, required=True)
    create.add_argument("--report", type=Path, required=True)
    create.add_argument("--out", type=Path, required=True)
    verify = subparsers.add_parser("verify", help="recompute and verify an attestation")
    verify.add_argument("--attestation", type=Path, required=True)
    verify.add_argument("--checkpoint", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "create":
            payload = write_attestation(args.checkpoint, args.report, args.out)
        else:
            payload = verify_attestation(
                args.attestation,
                expected_checkpoint_path=args.checkpoint,
            )
    except AttestationError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
