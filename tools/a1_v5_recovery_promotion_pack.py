#!/usr/bin/env python3
"""Seal the ordinary promotion evidence pack for the recovered-v5 child.

This is intentionally orchestration, not a second gate.  Source semantics,
evidence envelopes, cohort exclusions, and the final adjudication are built by
``a1_promotion_artifacts`` and replayed by ``a1_promotion_transaction``.  The
only recovery-specific work here is supplying the exact frozen lock verifier,
the committed recovery authority, and the deterministic 64/96/128 checkpoint
selection that the ordinary artifact CLI predates.

Evaluation remains outside this command.  Every input report must already be
the canonical immutable output of its evaluator.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools import a1_frozen_lock_verifier as frozen_verifier  # noqa: E402
from tools import a1_promotion_artifacts as artifacts  # noqa: E402
from tools import a1_promotion_transaction as promotion  # noqa: E402
from tools import a1_v5_disaster_recovery as recovery  # noqa: E402
from tools.champion_registry import ChampionRegistry  # noqa: E402


class PackError(RuntimeError):
    """The recovery promotion pack could not be sealed."""


def _existing(path: Path, *, where: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise PackError(f"cannot resolve {where}: {error}") from error
    if not resolved.is_file() or resolved.is_symlink():
        raise PackError(f"{where} must be a regular non-symlink file")
    return resolved


def _new_output_root(path: Path) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    if lexical.exists() or lexical.is_symlink():
        raise PackError(f"output directory must be fresh: {lexical}")
    lexical.mkdir(parents=True)
    return lexical


def _selection_ref(path: Path, *, candidate: Path) -> dict[str, str]:
    path = _existing(path, where="checkpoint selection")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PackError(f"cannot read checkpoint selection: {error}") from error
    if not isinstance(value, dict):
        raise PackError("checkpoint selection must be a JSON object")
    selected = value.get("selected_checkpoint")
    if not isinstance(selected, dict):
        raise PackError("checkpoint selection has no selected checkpoint")
    expected = {
        "path": str(candidate),
        "sha256": promotion._sha256(candidate),  # noqa: SLF001
    }
    try:
        selected_path = Path(str(selected.get("path"))).expanduser().resolve(strict=True)
    except OSError as error:
        raise PackError(f"cannot resolve selected checkpoint: {error}") from error
    if selected_path != candidate or selected != expected:
        raise PackError("checkpoint selection does not bind the requested candidate")
    return {
        "path": str(path),
        "sha256": promotion._sha256(path),  # noqa: SLF001
    }


def attach_checkpoint_selection(
    adjudication: Mapping[str, Any],
    *,
    checkpoint_selection: Path,
    candidate: Path,
) -> dict[str, Any]:
    """Attach the selected-dose receipt and refresh the canonical digest."""

    value = dict(adjudication)
    raw_candidate = value.get("candidate")
    if not isinstance(raw_candidate, dict):
        raise PackError("adjudication has no candidate record")
    candidate_record = dict(raw_candidate)
    if "training_checkpoint_selection" in candidate_record:
        raise PackError("adjudication already has a checkpoint selection")
    candidate_record["training_checkpoint_selection"] = _selection_ref(
        checkpoint_selection,
        candidate=candidate,
    )
    value["candidate"] = candidate_record
    value.pop("adjudication_sha256", None)
    value["adjudication_sha256"] = promotion._digest_value(value)  # noqa: SLF001
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    artifacts._write_new_readonly(path, value)  # noqa: SLF001


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


def _recovery_authority(path: Path) -> dict[str, Any]:
    try:
        replay = recovery.verify_committed_receipt(path)
    except recovery.RecoveryError as error:
        raise PackError(f"recovery receipt refused: {error}") from error
    authority = replay.get("authority") if isinstance(replay, Mapping) else None
    if not isinstance(authority, dict):
        raise PackError("recovery receipt returned no recovery authority")
    return authority


def build_pack(args: argparse.Namespace) -> dict[str, str]:
    lock = _existing(args.contract_lock, where="contract lock")
    recovery_receipt = _existing(args.recovery_receipt, where="recovery receipt")
    training_receipt = _existing(args.training_receipt, where="training receipt")
    training_report = _existing(args.training_report, where="training report")
    checkpoint_selection = _existing(
        args.checkpoint_selection, where="checkpoint selection"
    )
    registry_path = _existing(args.registry, where="registry")
    current_pointer = _existing(args.current_pointer, where="current pointer")
    candidate = _existing(args.candidate, where="candidate checkpoint")
    champion = _existing(args.champion, where="champion checkpoint")
    inputs = {
        "candidate_calibration": _existing(
            args.candidate_calibration, where="candidate calibration"
        ),
        "champion_calibration": _existing(
            args.champion_calibration, where="champion calibration"
        ),
        "internal_h2h": _existing(args.internal_h2h, where="internal H2H report"),
        "candidate_panel": _existing(
            args.candidate_panel, where="candidate external panel"
        ),
        "champion_panel": _existing(
            args.champion_panel, where="champion external panel"
        ),
        "high_regret_report": _existing(
            args.high_regret_report, where="high-regret report"
        ),
        "dose_screen": _existing(args.dose_screen, where="matched dose screen"),
    }

    try:
        verify_lock, verifier_authority = frozen_verifier.build_frozen_lock_verifier(
            frozen_repo=args.frozen_repo,
            expected_verifier_sha256=args.frozen_verifier_sha256,
            lock_path=lock,
        )
        contract = promotion._verify_contract(  # noqa: SLF001
            lock,
            verify_lock_fn=verify_lock,
        )
    except (frozen_verifier.FrozenVerifierError, promotion.PromotionError) as error:
        raise PackError(f"frozen contract replay refused: {error}") from error

    registry = ChampionRegistry.load(registry_path)
    recovery_authority = _recovery_authority(recovery_receipt)
    output = _new_output_root(args.out_dir)

    high_source_path = output / "high-regret.source.json"
    bucket_report_path = output / "bucket-games.report.json"
    bucket_source_path = output / "bucket-veto.source.json"
    high_source = artifacts.build_high_regret_source(
        report_path=inputs["high_regret_report"],
        candidate=candidate,
        champion=champion,
    )
    bucket_report = artifacts.build_bucket_game_report(
        report_path=inputs["high_regret_report"],
        candidate=candidate,
        champion=champion,
    )
    bucket_source = artifacts.build_bucket_veto_source(
        report_path=_write_then_return(bucket_report_path, bucket_report),
        candidate=candidate,
        champion=champion,
    )
    _write(high_source_path, high_source)
    _write(bucket_source_path, bucket_source)

    source_sets: dict[str, list[tuple[str, Path]]] = {
        "mechanism_calibration": [
            ("candidate_calibration", inputs["candidate_calibration"]),
            ("champion_calibration", inputs["champion_calibration"]),
        ],
        "internal_h2h": [("internal_h2h", inputs["internal_h2h"])],
        "external_panel": [
            ("candidate_panel", inputs["candidate_panel"]),
            ("champion_panel", inputs["champion_panel"]),
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

    exclusions_path = output / "cohort-exclusions.json"
    exclusions = artifacts.build_cohort_exclusions(
        contract=contract,
        candidate=candidate,
        cohorts=[("matched-dose-screen", "internal_h2h", inputs["dose_screen"])],
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
        nth_confirmation=(
            None
            if args.nth_confirmation is None
            else _existing(args.nth_confirmation, where="n64 confirmation")
        ),
    )
    adjudication = attach_checkpoint_selection(
        adjudication,
        checkpoint_selection=checkpoint_selection,
        candidate=candidate,
    )
    adjudication_path = output / "standard-promotion-adjudication.json"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".standard-promotion-adjudication.",
        suffix=".verify",
        dir=output,
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
            recovery_authority=recovery_authority,
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

    return {
        "adjudication": str(adjudication_path),
        "cohort_exclusions": str(exclusions_path),
        "contract_verifier_authority_sha256": str(
            verifier_authority["authority_sha256"]
        ),
        **{f"evidence_{kind}": str(path) for kind, path in evidence_paths.items()},
    }


def _write_then_return(path: Path, value: dict[str, Any]) -> Path:
    _write(path, value)
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-lock", type=Path, required=True)
    parser.add_argument("--frozen-repo", type=Path, required=True)
    parser.add_argument("--frozen-verifier-sha256", required=True)
    parser.add_argument("--recovery-receipt", type=Path, required=True)
    parser.add_argument("--training-receipt", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--checkpoint-selection", type=Path, required=True)
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
    parser.add_argument("--dose-screen", type=Path, required=True)
    parser.add_argument("--nth-confirmation", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
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
        print(f"a1 recovery promotion pack refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
