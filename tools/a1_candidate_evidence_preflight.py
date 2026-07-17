#!/usr/bin/env python3
"""Refuse expensive promotion evidence work for an ineligible candidate.

This is a cheap scheduling preflight, not a substitute for the promotion
transaction verifier.  It catches conditions that are knowable before running
calibration, external panels, or the held-out high-regret suite: a diagnostic
training report, missing sealed authorities, an unbound checkpoint, or a
validation cohort too small for the fixed 240-pair promotion suite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.high_regret_suite_contract import (  # noqa: E402
    load_validation_seed_manifest,
)


PREFLIGHT_SCHEMA = "a1-candidate-evidence-preflight-v1"


class PreflightError(RuntimeError):
    """The preflight inputs themselves could not be inspected."""


def _existing(path: Path, *, where: str) -> Path:
    lexical = path.expanduser()
    if lexical.is_symlink():
        raise PreflightError(f"{where} must not be a symlink")
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise PreflightError(f"cannot resolve {where}: {error}") from error
    if not resolved.is_file():
        raise PreflightError(f"{where} must be a regular file")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _load_json(path: Path, *, where: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PreflightError(f"cannot load {where}: {error}") from error
    if not isinstance(value, dict):
        raise PreflightError(f"{where} must be a JSON object")
    return value


def _valid_sha256(value: object) -> bool:
    if not isinstance(value, str):
        return False
    digest = value.removeprefix("sha256:")
    return len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)


def inspect_candidate(args: argparse.Namespace) -> dict[str, Any]:
    candidate = _existing(args.candidate, where="candidate checkpoint")
    report_path = _existing(args.training_report, where="training report")
    report = _load_json(report_path, where="training report")
    candidate_sha = _sha256(candidate)
    failures: list[dict[str, str]] = []

    def fail(code: str, detail: str) -> None:
        failures.append({"code": code, "detail": detail})

    if report.get("promotion_eligible") is not True:
        fail(
            "training_report_not_promotion_eligible",
            f"promotion_eligible={report.get('promotion_eligible')!r}",
        )
    block_reason = report.get("promotion_block_reason")
    if block_reason not in (None, ""):
        fail("training_report_has_promotion_block", str(block_reason))

    raw_checkpoint = report.get("checkpoint")
    if not isinstance(raw_checkpoint, str) or not raw_checkpoint:
        fail("training_report_missing_checkpoint", "checkpoint is absent")
    else:
        reported = Path(raw_checkpoint).expanduser()
        if not reported.is_absolute():
            reported = report_path.parent / reported
        try:
            reported = reported.resolve(strict=True)
        except OSError:
            fail("training_report_checkpoint_missing", str(reported))
        else:
            if reported != candidate:
                fail(
                    "training_report_candidate_path_mismatch",
                    f"report={reported} requested={candidate}",
                )
    report_sha = report.get("checkpoint_sha256")
    if not _valid_sha256(report_sha):
        fail("training_report_missing_checkpoint_sha256", repr(report_sha))
    elif report_sha != candidate_sha:
        fail(
            "training_report_candidate_sha256_mismatch",
            f"report={report_sha} actual={candidate_sha}",
        )

    if not _valid_sha256(report.get("a1_contract_sha256")):
        fail("training_report_missing_contract_authority", "a1_contract_sha256")
    if not isinstance(report.get("a1_central_published_executor_authority"), dict):
        fail(
            "training_report_missing_executor_authority",
            "a1_central_published_executor_authority",
        )

    manifest_binding: dict[str, Any] | None = None
    raw_manifest = report.get("validation_game_seed_manifest")
    if not isinstance(raw_manifest, str) or not raw_manifest:
        fail("training_report_missing_validation_manifest", repr(raw_manifest))
    else:
        manifest = Path(raw_manifest).expanduser()
        if not manifest.is_absolute():
            manifest = report_path.parent / manifest
        try:
            seeds, manifest_binding = load_validation_seed_manifest(manifest)
        except (OSError, ValueError) as error:
            fail("validation_manifest_invalid", str(error))
        else:
            count = len(seeds)
            if report.get("validation_game_seed_count") != count:
                fail(
                    "validation_manifest_count_mismatch",
                    f"report={report.get('validation_game_seed_count')!r} actual={count}",
                )
            if (
                report.get("validation_game_seed_set_sha256")
                != manifest_binding["game_seed_set_sha256"]
            ):
                fail(
                    "validation_manifest_seed_digest_mismatch",
                    "report and manifest seed-set digests differ",
                )
            if count < args.minimum_validation_games:
                fail(
                    "validation_cohort_too_small_for_high_regret",
                    f"{count} unique games < required {args.minimum_validation_games}",
                )

    for name, value in (
        ("contract_lock", args.contract_lock),
        ("training_receipt", args.training_receipt),
    ):
        if value is None:
            fail(f"missing_{name}", f"--{name.replace('_', '-')} is required for readiness")
        else:
            try:
                _existing(value, where=name.replace("_", " "))
            except PreflightError as error:
                fail(f"invalid_{name}", str(error))

    return {
        "schema_version": PREFLIGHT_SCHEMA,
        "promotion_evidence_ready": not failures,
        "diagnostic_evaluation_allowed": True,
        "candidate": {"path": str(candidate), "sha256": candidate_sha},
        "training_report": {"path": str(report_path), "sha256": _sha256(report_path)},
        "validation_seed_manifest": manifest_binding,
        "minimum_validation_games": args.minimum_validation_games,
        "failures": failures,
        "note": (
            "This scheduling preflight does not replace cryptographic replay by "
            "a1_candidate_promotion_pack.py."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--training-receipt", type=Path)
    parser.add_argument("--contract-lock", type=Path)
    parser.add_argument("--minimum-validation-games", type=int, default=240)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.minimum_validation_games <= 0:
        print("minimum validation games must be positive", file=sys.stderr)
        return 2
    try:
        result = inspect_candidate(args)
    except PreflightError as error:
        print(f"candidate evidence preflight failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["promotion_evidence_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
