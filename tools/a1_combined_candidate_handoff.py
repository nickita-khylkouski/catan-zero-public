#!/usr/bin/env python3
"""Seal the combined-196k candidate's evaluation-to-promotion handoff.

This tool does not run games and does not mutate champion state.  It replays
the completed two-dose training receipt, the fixed-search candidate/champion
panel, and the matched catanatron_value panels.  A promotion transaction can
only be rendered after the internal SPRT accepts H1 and the candidate's
external win rate is within the promotion policy's fixed regression limit.

The final promotion transaction remains authoritative and independently
replays calibration, internal, external, high-regret, and bucket evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools import a1_dual_arm_adjudicator as dual_adjudicator  # noqa: E402
from tools import a1_dual_arm_train as dual_train  # noqa: E402
from tools import a1_promotion_transaction as promotion  # noqa: E402
from tools.a1_external_panel_compare import (  # noqa: E402
    ExternalPanelComparisonError,
    compare_matched_external_panels,
)


MANIFEST_SCHEMA = "a1-combined-196k-evaluation-manifest-v1"
RESULT_SCHEMA = "a1-combined-196k-evaluation-handoff-v1"
PROMOTION_MANIFEST_SCHEMA = "a1-combined-196k-promotion-manifest-v1"
PROMOTION_PLAN_SCHEMA = "a1-combined-196k-promotion-plan-v1"


class CombinedHandoffError(RuntimeError):
    """A fail-closed combined-candidate handoff refusal."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _load(path: Path, *, where: str) -> dict[str, Any]:
    try:
        path = path.expanduser().resolve(strict=True)
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CombinedHandoffError(f"cannot load {where}: {error}") from error
    if not isinstance(value, dict):
        raise CombinedHandoffError(f"{where} must be a JSON object")
    return value


def _ref(path: Path, *, where: str) -> dict[str, str]:
    try:
        path = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise CombinedHandoffError(f"cannot resolve {where}: {error}") from error
    if not path.is_file() or path.stat().st_size <= 0:
        raise CombinedHandoffError(f"{where} is missing or empty: {path}")
    return {"path": str(path), "sha256": promotion._sha256(path)}  # noqa: SLF001


def _bound_ref(raw: Any, *, base: Path, where: str) -> Path:
    if not isinstance(raw, dict) or set(raw) != {"path", "sha256"}:
        raise CombinedHandoffError(f"{where} must be an exact file reference")
    path = Path(str(raw["path"]))
    if not path.is_absolute():
        path = base / path
    expected = _ref(path, where=where)
    if raw != expected:
        raise CombinedHandoffError(f"{where} bytes drift")
    return Path(expected["path"])


def _verify_training_receipt(path: Path) -> tuple[dict[str, Any], Path]:
    try:
        receipt = dual_train.verify_receipt(path)
        if (receipt.get("arm_id"), receipt.get("subset_id")) != (
            "n128",
            "full-140k",
        ):
            raise CombinedHandoffError(
                "combined second dose must be the complete n128/full-140k corpus"
            )
        inputs = receipt.get("inputs")
        if not isinstance(inputs, dict):
            raise CombinedHandoffError("combined receipt inputs are malformed")
        required = ("learner_lock", "corpus_meta", "validation", "producer")
        refs = {name: inputs.get(name) for name in required}
        if not all(
            isinstance(ref, dict) and set(ref) == {"path", "sha256"}
            for ref in refs.values()
        ):
            raise CombinedHandoffError("combined receipt lacks exact training inputs")
        parent = inputs.get("curriculum_parent")
        declaration = inputs.get("curriculum_declaration")
        if (
            not isinstance(parent, dict)
            or parent.get("schema_version") != "a1-curriculum-parent-binding-v1"
            or parent.get("parent_arm_id") != "n256"
            or parent.get("parent_subset_id") != "full-56k"
            or not isinstance(parent.get("receipt_path"), str)
        ):
            raise CombinedHandoffError(
                "combined receipt lacks the authenticated n256/full-56k first dose"
            )
        if (
            not isinstance(declaration, dict)
            or declaration.get("schema_version")
            != "a1-curriculum-declaration-v1"
            or declaration.get("kind") != "sequential_checkpoint_curriculum"
            or declaration.get("parent_receipt_path") != parent["receipt_path"]
            or declaration.get("parent_checkpoint") != parent["parent_checkpoint"]
        ):
            raise CombinedHandoffError(
                "combined receipt lacks the typed cumulative curriculum declaration"
            )
        verified = dual_train.verify_inputs(
            learner_lock=Path(str(refs["learner_lock"]["path"])),
            reviewed_lock_file_sha256=str(refs["learner_lock"]["sha256"]),
            data=Path(str(refs["corpus_meta"]["path"])).parent,
            validation=Path(str(refs["validation"]["path"])),
            producer_checkpoint=Path(str(refs["producer"]["path"])),
            curriculum_parent_receipt=Path(parent["receipt_path"]),
        )
        receipt = dual_train.verify_receipt(path, verified=verified)
    except dual_train.DualTrainError as error:
        raise CombinedHandoffError(f"combined training replay refused: {error}") from error
    checkpoint = _bound_ref(
        receipt.get("outputs", {}).get("checkpoint"),
        base=path.parent,
        where="combined candidate checkpoint",
    )
    return receipt, checkpoint


def adjudicate(manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve(strict=True)
    manifest = _load(manifest_path, where="combined evaluation manifest")
    expected_fields = {
        "schema_version",
        "champion",
        "training_receipt",
        "internal_pool",
        "candidate_external_pool",
        "champion_external_pool",
    }
    if set(manifest) != expected_fields or manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise CombinedHandoffError("combined evaluation manifest schema/fields drift")
    base = manifest_path.parent
    champion = _bound_ref(manifest["champion"], base=base, where="champion")
    receipt_path = _bound_ref(
        manifest["training_receipt"], base=base, where="combined training receipt"
    )
    receipt, candidate = _verify_training_receipt(receipt_path)
    internal_path = _bound_ref(
        manifest["internal_pool"], base=base, where="internal pool"
    )
    candidate_external_path = _bound_ref(
        manifest["candidate_external_pool"],
        base=base,
        where="candidate external pool",
    )
    champion_external_path = _bound_ref(
        manifest["champion_external_pool"],
        base=base,
        where="champion external pool",
    )
    try:
        internal = dual_adjudicator._replay_internal(  # noqa: SLF001
            internal_path, candidate=candidate, champion=champion
        )
        candidate_external = dual_adjudicator._replay_neutral(  # noqa: SLF001
            candidate_external_path, candidate=candidate
        )
        champion_external = dual_adjudicator._replay_neutral(  # noqa: SLF001
            champion_external_path, candidate=champion
        )
    except dual_adjudicator.DualAdjudicationError as error:
        raise CombinedHandoffError(f"evaluation replay refused: {error}") from error

    champion_sha = promotion._sha256(champion)  # noqa: SLF001
    if internal.get("baseline_checkpoint_sha256") != champion_sha:
        raise CombinedHandoffError("internal panel used a different champion")
    candidate_merge = candidate_external.get("fleet_merge")
    champion_merge = champion_external.get("fleet_merge")
    if not isinstance(candidate_merge, dict) or not isinstance(champion_merge, dict):
        raise CombinedHandoffError("external panels lack pooled provenance")
    def seed_intervals(merge: dict[str, Any], *, where: str) -> list[dict[str, int]]:
        raw = merge.get("seed_intervals")
        if not isinstance(raw, list) or any(
            not isinstance(row, dict)
            or isinstance(row.get("base_seed"), bool)
            or not isinstance(row.get("base_seed"), int)
            or isinstance(row.get("end_seed"), bool)
            or not isinstance(row.get("end_seed"), int)
            or row["end_seed"] <= row["base_seed"]
            for row in raw
        ):
            raise CombinedHandoffError(f"{where} has invalid seed intervals")
        # Artifact paths are role-specific by construction.  The cohort
        # identity is the ordered set of half-open seed intervals, not where
        # each role's shard report was collected.
        return [
            {"base_seed": row["base_seed"], "end_seed": row["end_seed"]}
            for row in raw
        ]

    candidate_cohort = {
        "search": candidate_external.get("effective_search_config"),
        "seeds": seed_intervals(candidate_merge, where="candidate external panel"),
    }
    champion_cohort = {
        "search": champion_external.get("effective_search_config"),
        "seeds": seed_intervals(champion_merge, where="champion external panel"),
    }
    if _canonical(candidate_cohort) != _canonical(champion_cohort):
        raise CombinedHandoffError(
            "candidate and champion external panels use different cohorts/configs"
        )
    try:
        paired_external = compare_matched_external_panels(
            candidate_external, champion_external
        )
    except ExternalPanelComparisonError as error:
        raise CombinedHandoffError(
            f"candidate and champion external panels are not pairable: {error}"
        ) from error
    candidate_rate = float(candidate_external["candidate_win_rate"])
    champion_rate = float(champion_external["candidate_win_rate"])
    if (
        paired_external["candidate_win_rate"] != candidate_rate
        or paired_external["champion_win_rate"] != champion_rate
    ):
        raise CombinedHandoffError("external panel summary rates do not replay from games")
    max_regression = promotion.MAX_EXTERNAL_WIN_RATE_REGRESSION
    # The direct pool uses the SPRT-native spelling ``H1`` while older
    # adjudicator receipts normalize it to ``accept_h1``.  Both mean the same
    # accepted alternative hypothesis and must replay identically.
    internal_pass = internal.get("verdict") in {"H1", "accept_h1"}
    external_pass = bool(paired_external["noninferiority"]["passed"])
    passed = internal_pass and external_pass
    result = {
        "schema_version": RESULT_SCHEMA,
        "passed": passed,
        "decision": "promotion_evidence_may_proceed" if passed else "reject_candidate",
        "manifest": _ref(manifest_path, where="combined evaluation manifest"),
        "training_receipt": _ref(receipt_path, where="combined training receipt"),
        "candidate": _ref(candidate, where="combined candidate"),
        "champion": _ref(champion, where="champion"),
        "curriculum": {
            "first_dose": "n256/full-56k",
            "second_dose": "n128/full-140k",
            "all_games": 196000,
            "receipt_sha256": receipt["receipt_sha256"],
        },
        "gates": {
            "internal_h2h": {
                "passed": internal_pass,
                "verdict": internal.get("verdict"),
                "llr": float(internal["pentanomial_sprt"]["llr"]),
                "candidate_win_rate": float(internal["candidate_win_rate"]),
                "complete_pairs": int(internal["complete_pairs"]),
                "pool": _ref(internal_path, where="internal pool"),
            },
            "external_panel": {
                "passed": external_pass,
                "candidate_win_rate": candidate_rate,
                "champion_win_rate": champion_rate,
                "candidate_minus_champion": candidate_rate - champion_rate,
                "paired_common_opponent": paired_external,
                "max_win_rate_regression": max_regression,
                "cohort_sha256": _digest(candidate_cohort),
                "candidate_pool": _ref(
                    candidate_external_path, where="candidate external pool"
                ),
                "champion_pool": _ref(
                    champion_external_path, where="champion external pool"
                ),
            },
        },
        "promotion": {
            "ready": False,
            "reason": (
                "full calibration/high-regret/bucket evidence remains required"
                if passed
                else "internal and external gates did not both pass"
            ),
        },
    }
    result["handoff_sha256"] = _digest(result)
    return result


def write_new(path: Path, value: dict[str, Any]) -> None:
    path = path.expanduser().resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o444,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def verify_result(path: Path) -> dict[str, Any]:
    value = _load(path, where="combined evaluation handoff")
    stated = value.get("handoff_sha256")
    unhashed = dict(value)
    unhashed.pop("handoff_sha256", None)
    if value.get("schema_version") != RESULT_SCHEMA or stated != _digest(unhashed):
        raise CombinedHandoffError("combined handoff schema/digest drift")
    replay = adjudicate(Path(str(value.get("manifest", {}).get("path", ""))))
    if replay != value:
        raise CombinedHandoffError("combined handoff no longer replays")
    return value


def build_promotion_plan(
    handoff_path: Path, promotion_manifest_path: Path
) -> dict[str, Any]:
    handoff_path = handoff_path.expanduser().resolve(strict=True)
    handoff = verify_result(handoff_path)
    if handoff.get("passed") is not True:
        raise CombinedHandoffError("combined candidate did not pass both evaluation gates")
    promotion_manifest_path = promotion_manifest_path.expanduser().resolve(strict=True)
    manifest = _load(promotion_manifest_path, where="combined promotion manifest")
    expected_fields = {
        "schema_version",
        "registry",
        "current_pointer",
        "contract_lock",
        "adjudication",
        "training_receipt",
        "receipt",
        "reason",
    }
    if set(manifest) != expected_fields or manifest.get("schema_version") != PROMOTION_MANIFEST_SCHEMA:
        raise CombinedHandoffError("combined promotion manifest schema/fields drift")
    base = promotion_manifest_path.parent
    registry = _bound_ref(manifest["registry"], base=base, where="registry")
    current = _bound_ref(
        manifest["current_pointer"], base=base, where="current pointer"
    )
    contract = _bound_ref(
        manifest["contract_lock"], base=base, where="contract lock"
    )
    adjudication = _bound_ref(
        manifest["adjudication"], base=base, where="promotion adjudication"
    )
    training_receipt = _bound_ref(
        manifest["training_receipt"], base=base, where="training receipt"
    )
    if _ref(training_receipt, where="training receipt") != handoff["training_receipt"]:
        raise CombinedHandoffError("promotion manifest names a different training receipt")
    receipt = Path(str(manifest["receipt"])).expanduser().resolve(strict=False)
    reason = str(manifest["reason"])
    if not reason.strip():
        raise CombinedHandoffError("promotion reason must be nonempty")
    try:
        promotion.prepare_promotion(
            registry_path=registry,
            current_pointer=current,
            contract_lock=contract,
            adjudication_path=adjudication,
            training_receipt=training_receipt,
            receipt_path=receipt,
            reason=reason,
        )
    except promotion.PromotionError as error:
        raise CombinedHandoffError(f"full-evidence promotion preflight refused: {error}") from error
    command = [
        sys.executable,
        str(_REPO_ROOT / "tools" / "a1_promotion_transaction.py"),
        "promote",
        "--registry",
        str(registry),
        "--current-pointer",
        str(current),
        "--contract-lock",
        str(contract),
        "--adjudication",
        str(adjudication),
        "--training-receipt",
        str(training_receipt),
        "--receipt",
        str(receipt),
        "--reason",
        reason,
    ]
    result = {
        "schema_version": PROMOTION_PLAN_SCHEMA,
        "passed": True,
        "decision": "full_evidence_verified_promotion_dry_run",
        "handoff": _ref(handoff_path, where="combined handoff"),
        "promotion_manifest": _ref(
            promotion_manifest_path, where="combined promotion manifest"
        ),
        "candidate": handoff["candidate"],
        "command": command,
        "command_sha256": _digest(command),
    }
    result["plan_sha256"] = _digest(result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--out", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--out", type=Path, required=True)
    plan = sub.add_parser("promotion-plan")
    plan.add_argument("--handoff", type=Path, required=True)
    plan.add_argument("--promotion-manifest", type=Path, required=True)
    plan.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "evaluate":
            value = adjudicate(args.manifest)
        elif args.command == "verify":
            value = verify_result(args.out)
        else:
            value = build_promotion_plan(args.handoff, args.promotion_manifest)
        if args.command != "verify":
            write_new(args.out, value)
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    except (CombinedHandoffError, OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(2, f"REFUSED: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
