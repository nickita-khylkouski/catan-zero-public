#!/usr/bin/env python3
"""Verify the one permitted post-recovery A1 promotion gate.

The recovered v5 producer is authoritative only for generation.  A child may
become promotable only after two fresh, conjunctive checks:

1. the complete ordinary promotion adjudication proves strict H1 superiority
   over the exact recovered v5 producer; and
2. an independent fixed 300-pair n128 cohort does not hit H0 against the exact
   authenticated f7 safety reference.

This module reuses the ordinary promotion verifier for training provenance,
calibration, external panels, high-regret/bucket evidence, and the strict H1
parent comparison.  It adds only the recovery authority and f7 veto.  The
result is an immutable authority for the promotion transaction; it is neither
a promotion receipt nor permission to auto-promote.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import a1_pre_wave_contract as pre_wave  # noqa: E402
from tools import a1_promotion_transaction as promotion  # noqa: E402
from tools import a1_v5_disaster_recovery as recovery  # noqa: E402
from tools.champion_registry import ChampionRegistry  # noqa: E402


AUTHORITY_SCHEMA = "a1-v5-recovery-full-gate-authority-v1"
# This cohort must remain disjoint from every prior VAL-only claim.  The former
# 6_199_700_000 base was already partially occupied by a sealed 192-pair panel,
# which made the required 300-pair recovery veto impossible to claim.
F7_VETO_BASE_SEED = 6_199_100_000
F7_VETO_COMPLETE_PAIRS = 300
F7_COMPARISON_MODE = "recovery_safety_reference"
F7_COMPARISON_REASON = "disaster_recovery_f7_non_regression_veto"


class RecoveryGateError(RuntimeError):
    """The exact disaster-recovery dual-baseline gate did not replay."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RecoveryGateError(f"value is not canonical JSON: {error}") from error


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _stable_read(path: Path, *, where: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RecoveryGateError(f"cannot open {where}: {error}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RecoveryGateError(f"{where} is not a regular file")
        chunks: list[bytes] = []
        while block := os.read(descriptor, 1 << 20):
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    if identity != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise RecoveryGateError(f"{where} changed while read")
    live = path.stat(follow_symlinks=False)
    if identity != (
        live.st_dev,
        live.st_ino,
        live.st_size,
        live.st_mtime_ns,
        live.st_ctime_ns,
    ):
        raise RecoveryGateError(f"{where} was replaced while read")
    return b"".join(chunks)


def _existing(path: Path, *, where: str) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise RecoveryGateError(f"cannot resolve {where}: {error}") from error
    if resolved != lexical or not resolved.is_file() or resolved.is_symlink():
        raise RecoveryGateError(f"{where} must be a canonical regular file")
    return resolved


def _file_ref(path: Path, *, where: str) -> tuple[dict[str, str], bytes]:
    path = _existing(path, where=where)
    raw = _stable_read(path, where=where)
    return {"path": str(path), "sha256": _sha256_bytes(raw)}, raw


def _json(path: Path, *, where: str) -> dict[str, Any]:
    raw = _stable_read(path, where=where)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RecoveryGateError(f"cannot parse {where}: {error}") from error
    if not isinstance(value, dict):
        raise RecoveryGateError(f"{where} must be a JSON object")
    return value


def _revalidate(snapshot: Mapping[Path, bytes]) -> None:
    for path, expected in snapshot.items():
        if _stable_read(path, where=f"gate snapshot {path}") != expected:
            raise RecoveryGateError(f"gate input changed during verification: {path}")


def _validate_standard_intervals(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or not raw:
        raise RecoveryGateError("ordinary gate has no retained fresh cohorts")
    intervals: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise RecoveryGateError(f"ordinary cohort interval {index} is malformed")
        base = item.get("base_seed")
        end = item.get("end_seed")
        kind = item.get("kind")
        if (
            isinstance(base, bool)
            or not isinstance(base, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or end <= base
            or not isinstance(kind, str)
            or not kind
        ):
            raise RecoveryGateError(f"ordinary cohort interval {index} is invalid")
        intervals.append({"kind": kind, "base_seed": base, "end_seed": end})
    ordered = sorted(intervals, key=lambda item: (item["base_seed"], item["end_seed"]))
    for left, right in zip(ordered, ordered[1:]):
        if left["end_seed"] > right["base_seed"]:
            raise RecoveryGateError("ordinary promotion cohorts overlap each other")
    veto_end = F7_VETO_BASE_SEED + F7_VETO_COMPLETE_PAIRS
    for interval in ordered:
        if max(interval["base_seed"], F7_VETO_BASE_SEED) < min(
            interval["end_seed"], veto_end
        ):
            raise RecoveryGateError("fresh f7 veto cohort overlaps ordinary gate")
    return intervals


def _exact_f7_seeds(report: Mapping[str, Any]) -> list[int]:
    games = report.get("games")
    if not isinstance(games, list):
        raise RecoveryGateError("f7 veto report has no retained games")
    seeds = {
        game.get("game_seed")
        for game in games
        if isinstance(game, dict)
        and isinstance(game.get("game_seed"), int)
        and not isinstance(game.get("game_seed"), bool)
    }
    expected = set(
        range(F7_VETO_BASE_SEED, F7_VETO_BASE_SEED + F7_VETO_COMPLETE_PAIRS)
    )
    if seeds != expected:
        raise RecoveryGateError("f7 veto does not use the exact fixed fresh seed cohort")
    return sorted(seeds)


def verify_recovery_gate(
    *,
    recovery_receipt_path: Path,
    contract_lock_path: Path,
    standard_adjudication_path: Path,
    training_receipt_path: Path,
    registry_path: Path,
    current_pointer_path: Path,
    f7_nonregression_report_path: Path,
    legacy_contract_attestation_path: Path | None = None,
    verify_lock_fn: Callable[..., dict[str, Any]] = pre_wave.verify_lock,
    recovery_verifier_fn: Callable[[Path], Mapping[str, Any]] = (
        recovery.verify_committed_receipt
    ),
) -> dict[str, Any]:
    """Replay all ordinary evidence plus the independent f7 veto."""

    paths = {
        "recovery_receipt": _existing(
            recovery_receipt_path, where="v5 recovery receipt"
        ),
        "contract_lock": _existing(contract_lock_path, where="A1 contract lock"),
        "standard_adjudication": _existing(
            standard_adjudication_path, where="ordinary promotion adjudication"
        ),
        "training_receipt": _existing(
            training_receipt_path, where="one-dose training receipt"
        ),
        "registry": _existing(registry_path, where="recovery registry"),
        "current_pointer": _existing(
            current_pointer_path, where="recovery current pointer"
        ),
        "f7_report": _existing(
            f7_nonregression_report_path, where="f7 non-regression report"
        ),
    }
    if legacy_contract_attestation_path is not None:
        paths["legacy_contract_attestation"] = _existing(
            legacy_contract_attestation_path,
            where="legacy contract attestation",
        )
    snapshot = {
        path: _stable_read(path, where=f"recovery gate input {name}")
        for name, path in paths.items()
    }
    try:
        replay = recovery_verifier_fn(paths["recovery_receipt"])
        if not isinstance(replay, Mapping):
            raise RecoveryGateError("recovery verifier returned no mapping")
        authority = replay.get("authority")
        receipt = replay.get("receipt")
        if not isinstance(authority, dict) or not isinstance(receipt, dict):
            raise RecoveryGateError("recovery verifier returned malformed authority")
        if (
            Path(str(receipt.get("registry", {}).get("path"))) != paths["registry"]
            or Path(str(receipt.get("current_pointer", {}).get("path")))
            != paths["current_pointer"]
        ):
            raise RecoveryGateError(
                "explicit registry/pointer differ from recovery receipt"
            )

        contract, legacy_snapshot = promotion._verify_contract_with_snapshot(  # noqa: SLF001
            paths["contract_lock"],
            verify_lock_fn=verify_lock_fn,
            legacy_contract_attestation=paths.get("legacy_contract_attestation"),
            expected_training_receipt=paths["training_receipt"],
        )
        registry = ChampionRegistry.load(paths["registry"])
        verified = promotion._verify_adjudication(  # noqa: SLF001
            paths["standard_adjudication"],
            contract=contract,
            contract_lock=paths["contract_lock"],
            training_receipt=paths["training_receipt"],
            registry=registry,
            current_pointer=paths["current_pointer"],
            legacy_snapshot=legacy_snapshot,
            recovery_authority=authority,
        )
        if verified.get("promotion_mode") != "disaster_recovery_parent":
            raise RecoveryGateError("ordinary gate did not use the recovered parent")
        recovered = authority.get("recovered_generator")
        safety = authority.get(recovery.RECOVERY_RELATION)
        producer_identity = authority.get("producer_identity")
        if not all(isinstance(value, dict) for value in (recovered, safety, producer_identity)):
            raise RecoveryGateError("recovery authority lacks dual-baseline identity")
        if (
            verified.get("champion", {}).get("path") != recovered.get("path")
            or verified.get("champion", {}).get("sha256") != recovered.get("sha256")
        ):
            raise RecoveryGateError("strict H1 baseline is not the recovered producer")

        f7_path = _existing(Path(str(safety.get("path"))), where="f7 safety checkpoint")
        if promotion._sha256(f7_path) != safety.get("sha256"):  # noqa: SLF001
            raise RecoveryGateError("f7 safety checkpoint bytes drifted")
        report = _json(paths["f7_report"], where="f7 non-regression report")
        candidate = verified["candidate"]
        candidate_search = candidate["agent_identity"]["search_config"]
        parent_search = producer_identity.get("search_config")
        if not isinstance(parent_search, dict) or not parent_search:
            raise RecoveryGateError("recovered producer has no search identity")
        promotion._verify_internal_h2h_source(  # noqa: SLF001
            report,
            candidate=Path(candidate["path"]),
            champion=f7_path,
            where="recovery f7 non-regression veto",
            sealed_semantics=promotion._sealed_evaluation_semantics(contract),  # noqa: SLF001
            candidate_search_config=candidate_search,
            champion_search_config=candidate_search,
            required_n_full=128,
            comparison_mode=F7_COMPARISON_MODE,
            candidate_parent_path=Path(str(recovered["path"])),
            candidate_parent_sha256=str(recovered["sha256"]),
            candidate_parent_search_config=parent_search,
            verdict_policy="non_regression_veto",
        )
        promotion._verify_internal_h2h_cohort(  # noqa: SLF001
            report, where="recovery f7 non-regression veto"
        )
        if report.get("complete_pairs") != F7_VETO_COMPLETE_PAIRS:
            raise RecoveryGateError("f7 veto must complete exactly 300 pairs")
        if report.get("verdict") not in {"H1", "continue"}:
            raise RecoveryGateError("f7 veto reached H0")
        f7_seeds = _exact_f7_seeds(report)
        standard_intervals = _validate_standard_intervals(
            verified.get("final_cohort_intervals")
        )
        _revalidate(snapshot)

        input_refs = {
            name: {
                "path": str(path),
                "sha256": _sha256_bytes(snapshot[path]),
            }
            for name, path in paths.items()
        }
        result: dict[str, Any] = {
            "schema_version": AUTHORITY_SCHEMA,
            "inputs": input_refs,
            "recovery_authority": authority,
            "contract": {
                **input_refs["contract_lock"],
                "contract_sha256": contract["contract_sha256"],
            },
            "candidate": candidate,
            "strict_h1_parent_gate": {
                "passed": True,
                "baseline": recovered,
                "adjudication": {
                    **input_refs["standard_adjudication"],
                    "adjudication_sha256": verified["adjudication_sha256"],
                },
                "fresh_cohort_intervals": standard_intervals,
            },
            "f7_non_regression_veto": {
                "passed": True,
                "baseline": safety,
                "report": input_refs["f7_report"],
                "verdict": report["verdict"],
                "complete_pairs": F7_VETO_COMPLETE_PAIRS,
                "base_seed": F7_VETO_BASE_SEED,
                "end_seed": F7_VETO_BASE_SEED + F7_VETO_COMPLETE_PAIRS,
                "seed_cohort_sha256": _digest(f7_seeds),
            },
            "policy": {
                "dual_baseline_conjunctive": True,
                "strict_h1_over_recovered_parent": True,
                "f7_h0_veto": True,
                "fresh_cohorts_required": True,
                "promotion_eligible": True,
                "auto_promotion": False,
            },
        }
        result["authority_sha256"] = _digest(result)
        return result
    except RecoveryGateError:
        raise
    except (promotion.PromotionError, recovery.RecoveryError, OSError, ValueError) as error:
        raise RecoveryGateError(f"recovery gate refused: {error}") from error


def verify_recovery_gate_authority(
    authority_path: Path,
    *,
    verify_lock_fn: Callable[..., dict[str, Any]] = pre_wave.verify_lock,
    recovery_verifier_fn: Callable[[Path], Mapping[str, Any]] = (
        recovery.verify_committed_receipt
    ),
) -> dict[str, Any]:
    """Replay a published authority from only its own exact input references."""

    authority_path = _existing(authority_path, where="recovery gate authority")
    value = _json(authority_path, where="recovery gate authority")
    unsigned = dict(value)
    stated = unsigned.pop("authority_sha256", None)
    if value.get("schema_version") != AUTHORITY_SCHEMA or stated != _digest(unsigned):
        raise RecoveryGateError("recovery gate authority schema/digest drift")
    inputs = value.get("inputs")
    if not isinstance(inputs, dict):
        raise RecoveryGateError("recovery gate authority has no input references")
    required = {
        "recovery_receipt",
        "contract_lock",
        "standard_adjudication",
        "training_receipt",
        "registry",
        "current_pointer",
        "f7_report",
    }
    if set(inputs) != required and set(inputs) != required | {
        "legacy_contract_attestation"
    }:
        raise RecoveryGateError("recovery gate authority input shape drift")
    for name, ref in inputs.items():
        if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
            raise RecoveryGateError(f"recovery gate input reference {name} is malformed")
        path = _existing(Path(str(ref["path"])), where=f"recovery gate input {name}")
        if _sha256_bytes(_stable_read(path, where=name)) != ref["sha256"]:
            raise RecoveryGateError(f"recovery gate input {name} bytes drifted")
    rebuilt = verify_recovery_gate(
        recovery_receipt_path=Path(inputs["recovery_receipt"]["path"]),
        contract_lock_path=Path(inputs["contract_lock"]["path"]),
        standard_adjudication_path=Path(inputs["standard_adjudication"]["path"]),
        training_receipt_path=Path(inputs["training_receipt"]["path"]),
        registry_path=Path(inputs["registry"]["path"]),
        current_pointer_path=Path(inputs["current_pointer"]["path"]),
        f7_nonregression_report_path=Path(inputs["f7_report"]["path"]),
        legacy_contract_attestation_path=(
            None
            if "legacy_contract_attestation" not in inputs
            else Path(inputs["legacy_contract_attestation"]["path"])
        ),
        verify_lock_fn=verify_lock_fn,
        recovery_verifier_fn=recovery_verifier_fn,
    )
    if rebuilt != value:
        raise RecoveryGateError("published recovery gate authority does not replay")
    return value


def write_recovery_gate_authority(out_path: Path, **kwargs: Any) -> dict[str, Any]:
    authority = verify_recovery_gate(**kwargs)
    lexical = Path(os.path.abspath(os.fspath(out_path.expanduser())))
    if lexical.exists() or lexical.is_symlink():
        raise RecoveryGateError("recovery gate authority output must be fresh")
    lexical.parent.mkdir(parents=True, exist_ok=True)
    if lexical.parent.is_symlink() or lexical.parent.resolve(strict=True) != lexical.parent:
        raise RecoveryGateError("recovery gate authority parent is not canonical")
    payload = json.dumps(authority, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    descriptor = os.open(
        lexical,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o444,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        directory = os.open(
            lexical.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        lexical.unlink(missing_ok=True)
        raise
    return authority


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-receipt", type=Path, required=True)
    parser.add_argument("--contract-lock", type=Path, required=True)
    parser.add_argument("--standard-adjudication", type=Path, required=True)
    parser.add_argument("--training-receipt", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--current-pointer", type=Path, required=True)
    parser.add_argument("--f7-nonregression-report", type=Path, required=True)
    parser.add_argument("--legacy-contract-attestation", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        authority = write_recovery_gate_authority(
            args.out,
            recovery_receipt_path=args.recovery_receipt,
            contract_lock_path=args.contract_lock,
            standard_adjudication_path=args.standard_adjudication,
            training_receipt_path=args.training_receipt,
            registry_path=args.registry,
            current_pointer_path=args.current_pointer,
            f7_nonregression_report_path=args.f7_nonregression_report,
            legacy_contract_attestation_path=args.legacy_contract_attestation,
        )
    except RecoveryGateError as error:
        raise SystemExit(f"REFUSED: {error}") from error
    print(json.dumps(authority, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
