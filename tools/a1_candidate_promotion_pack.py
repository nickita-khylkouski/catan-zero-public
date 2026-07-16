#!/usr/bin/env python3
"""Seal one ordinary candidate's matched evaluator outputs for promotion.

This is orchestration, not another evaluator or gate.  It accepts the immutable
reports produced by the existing calibration, internal H2H, external-panel,
and held-out high-regret evaluators.  The canonical artifact builders derive
the five typed evidence envelopes, prior-cohort exclusions, and the final
adjudication.  The promotion verifier replays the completed graph before the
pack receipt is published.

Every input and output is candidate-specific.  A report from another candidate
or champion, an incomplete paired cohort, a failing bucket, a stale cohort, or
a training receipt that does not produce the requested candidate is refused by
the existing promotion validators.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import a1_promotion_artifacts as artifacts  # noqa: E402
from tools import a1_promotion_transaction as promotion  # noqa: E402
from tools.champion_registry import ChampionRegistry  # noqa: E402


PACK_RECEIPT_SCHEMA = "a1-candidate-promotion-pack-receipt-v1"


class PackError(RuntimeError):
    """The candidate evaluation reports cannot form a promotion pack."""


def _existing(path: Path, *, where: str) -> Path:
    lexical = path.expanduser()
    if lexical.is_symlink():
        raise PackError(f"{where} must not be a symlink")
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise PackError(f"cannot resolve {where}: {error}") from error
    if not resolved.is_file():
        raise PackError(f"{where} must be a regular file")
    return resolved


def _fresh_output(path: Path) -> Path:
    output = Path(os.path.abspath(os.fspath(path.expanduser())))
    if output.exists() or output.is_symlink():
        raise PackError(f"output directory must be fresh: {output}")
    output.mkdir(parents=True)
    return output


def _write(path: Path, value: dict[str, Any]) -> None:
    artifacts._write_new_readonly(path, value)  # noqa: SLF001


def _file_ref(path: Path) -> dict[str, str]:
    path = _existing(path, where="pack artifact")
    return {
        "path": str(path),
        "sha256": promotion._sha256(path),  # noqa: SLF001
    }


def _parse_cohorts(values: Sequence[str]) -> list[tuple[str, str, Path]]:
    cohorts: list[tuple[str, str, Path]] = []
    for raw in values:
        identity, separator, raw_path = raw.partition("=")
        label, kind_separator, kind = identity.partition(":")
        if not separator or not kind_separator or not label or not kind or not raw_path:
            raise PackError("--prior-cohort entries must be LABEL:KIND=PATH")
        cohorts.append((label, kind, _existing(Path(raw_path), where=label)))
    if not cohorts:
        raise PackError("at least one --prior-cohort is required")
    return cohorts


def _validate_evidence(
    path: Path,
    *,
    value: dict[str, Any],
    kind: str,
    contract: dict[str, Any],
    candidate: Path,
    champion: Path,
    registry: ChampionRegistry,
) -> None:
    artifacts._validate_envelope_before_write(  # noqa: SLF001
        path,
        value=value,
        kind=kind,
        contract=contract,
        candidate=candidate,
        champion=champion,
        registry=registry,
    )


def build_pack(args: argparse.Namespace) -> dict[str, Any]:
    lock = _existing(args.contract_lock, where="contract lock")
    training_receipt = _existing(args.training_receipt, where="training receipt")
    training_report = _existing(args.training_report, where="training report")
    registry_path = _existing(args.registry, where="champion registry")
    current_pointer = _existing(args.current_pointer, where="current pointer")
    candidate = _existing(args.candidate, where="candidate checkpoint")
    champion = _existing(args.champion, where="champion checkpoint")
    reports = {
        "candidate_calibration": _existing(
            args.candidate_calibration, where="candidate calibration"
        ),
        "champion_calibration": _existing(
            args.champion_calibration, where="champion calibration"
        ),
        "internal_h2h": _existing(args.internal_h2h, where="internal H2H"),
        "candidate_panel": _existing(
            args.candidate_panel, where="candidate external panel"
        ),
        "champion_panel": _existing(
            args.champion_panel, where="champion external panel"
        ),
        "high_regret": _existing(
            args.high_regret_report, where="high-regret report"
        ),
    }
    prior_cohorts = _parse_cohorts(args.prior_cohort)
    nth_confirmation = (
        None
        if args.nth_confirmation is None
        else _existing(args.nth_confirmation, where="n64 confirmation")
    )
    try:
        contract = promotion._verify_contract(lock)  # noqa: SLF001
    except promotion.PromotionError as error:
        raise PackError(f"contract replay refused: {error}") from error
    registry = ChampionRegistry.load(registry_path)
    adjudication_target = Path(
        os.path.abspath(os.fspath(args.out.expanduser()))
    )
    exclusions_target = Path(
        os.path.abspath(os.fspath(args.cohort_exclusions_out.expanduser()))
    )
    receipt_target = Path(
        os.path.abspath(os.fspath(args.receipt.expanduser()))
    )
    if (
        len({adjudication_target, exclusions_target, receipt_target}) != 3
        or exclusions_target.parent != adjudication_target.parent
        or receipt_target.parent != adjudication_target.parent
    ):
        raise PackError("pack outputs must be three distinct files in one fresh directory")
    output = _fresh_output(adjudication_target.parent)
    try:
        high_source_path = output / "high-regret.source.json"
        bucket_report_path = output / "bucket-games.report.json"
        bucket_source_path = output / "bucket-veto.source.json"
        high_source = artifacts.build_high_regret_source(
            report_path=reports["high_regret"],
            candidate=candidate,
            champion=champion,
        )
        bucket_report = artifacts.build_bucket_game_report(
            report_path=reports["high_regret"],
            candidate=candidate,
            champion=champion,
        )
        _write(bucket_report_path, bucket_report)
        bucket_source = artifacts.build_bucket_veto_source(
            report_path=bucket_report_path,
            candidate=candidate,
            champion=champion,
        )
        _write(high_source_path, high_source)
        _write(bucket_source_path, bucket_source)

        source_sets: dict[str, list[tuple[str, Path]]] = {
            "mechanism_calibration": [
                ("candidate_calibration", reports["candidate_calibration"]),
                ("champion_calibration", reports["champion_calibration"]),
            ],
            "internal_h2h": [("internal_h2h", reports["internal_h2h"])],
            "external_panel": [
                ("candidate_panel", reports["candidate_panel"]),
                ("champion_panel", reports["champion_panel"]),
            ],
            "high_regret": [("high_regret", high_source_path)],
            "bucket_veto": [("bucket_veto", bucket_source_path)],
        }
        evidence_paths: dict[str, Path] = {}
        for kind in sorted(source_sets):
            path = output / f"{kind.replace('_', '-')}.evidence.json"
            value = artifacts.build_evidence_envelope(
                kind=kind,
                contract=contract,
                candidate=candidate,
                champion=champion,
                sources=source_sets[kind],
            )
            _validate_evidence(
                path,
                value=value,
                kind=kind,
                contract=contract,
                candidate=candidate,
                champion=champion,
                registry=registry,
            )
            _write(path, value)
            evidence_paths[kind] = path

        exclusions_path = exclusions_target
        exclusions = artifacts.build_cohort_exclusions(
            contract=contract,
            candidate=candidate,
            cohorts=prior_cohorts,
        )
        _write(exclusions_path, exclusions)

        adjudication = artifacts.build_adjudication(
            contract=contract,
            contract_lock=lock,
            training_receipt=training_receipt,
            registry=registry,
            current_pointer=current_pointer,
            candidate=candidate,
            candidate_version=args.candidate_version,
            training_report=training_report,
            champion=champion,
            champion_version=args.champion_version,
            evidence=sorted(evidence_paths.items()),
            nth_confirmation=nth_confirmation,
        )
        adjudication_path = adjudication_target
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".promotion-adjudication.", suffix=".verify", dir=output
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(adjudication, handle, indent=2, sort_keys=True)
                handle.write("\n")
            verified = promotion._verify_adjudication(  # noqa: SLF001
                temporary,
                contract=contract,
                contract_lock=lock,
                training_receipt=training_receipt,
                registry=registry,
                current_pointer=current_pointer,
            )
            promotion._verify_cohort_exclusions(  # noqa: SLF001
                exclusions_path,
                contract_sha256=contract["contract_sha256"],
                candidate_sha256=verified["candidate"]["sha256"],
                final_intervals=verified["final_cohort_intervals"],
            )
        finally:
            temporary.unlink(missing_ok=True)
        _write(adjudication_path, adjudication)

        receipt = {
            "schema_version": PACK_RECEIPT_SCHEMA,
            "contract_sha256": contract["contract_sha256"],
            "candidate": _file_ref(candidate),
            "champion": _file_ref(champion),
            "training_receipt": _file_ref(training_receipt),
            "training_report": _file_ref(training_report),
            "adjudication": _file_ref(adjudication_path),
            "cohort_exclusions": _file_ref(exclusions_path),
            "evidence": {
                kind: _file_ref(path) for kind, path in sorted(evidence_paths.items())
            },
        }
        receipt["receipt_sha256"] = promotion._digest_value(receipt)  # noqa: SLF001
        receipt_path = receipt_target
        _write(receipt_path, receipt)
        return {**receipt, "receipt": _file_ref(receipt_path)}
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-lock", type=Path, required=True)
    parser.add_argument("--training-receipt", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--current-pointer", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-version", type=int, required=True)
    parser.add_argument("--champion", type=Path, required=True)
    parser.add_argument("--champion-version", type=int, required=True)
    parser.add_argument("--candidate-calibration", type=Path, required=True)
    parser.add_argument("--champion-calibration", type=Path, required=True)
    parser.add_argument("--internal-h2h", type=Path, required=True)
    parser.add_argument("--candidate-panel", type=Path, required=True)
    parser.add_argument("--champion-panel", type=Path, required=True)
    parser.add_argument("--high-regret-report", type=Path, required=True)
    parser.add_argument(
        "--prior-cohort", action="append", default=[], metavar="LABEL:KIND=PATH"
    )
    parser.add_argument("--nth-confirmation", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="fresh adjudication path; its fresh parent becomes the immutable pack",
    )
    parser.add_argument("--cohort-exclusions-out", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = build_pack(args)
    except (
        PackError,
        artifacts.ArtifactBuildError,
        promotion.PromotionError,
        OSError,
        ValueError,
    ) as error:
        print(f"a1 candidate promotion pack refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
