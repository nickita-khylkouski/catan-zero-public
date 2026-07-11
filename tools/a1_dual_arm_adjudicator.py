#!/usr/bin/env python3
"""Replay four dual-arm candidates and seal one fixed-search winner plan.

The tool does not mutate champion state.  It verifies training receipts and
replays the raw fleet sources behind each pooled internal and neutral report.
Only candidates accepted by both fixed-search panels are eligible.  The sealed
output identifies the winner and renders the existing promotion transaction
command to use after its normal calibration/high-regret evidence graph exists.
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

from tools import a1_dual_arm_train as dual_train  # noqa: E402
from tools import a1_evaluation_pool as pool  # noqa: E402
from tools import a1_promotion_transaction as promotion  # noqa: E402


MANIFEST_SCHEMA = "a1-dual-arm-adjudication-manifest-v1"
RESULT_SCHEMA = "a1-dual-arm-adjudication-v1"
PROMOTION_PLAN_SCHEMA = "a1-dual-arm-promotion-plan-v1"
IDENTITIES = frozenset(dual_train.ALLOWED_IDENTITIES)


class DualAdjudicationError(RuntimeError):
    """A fail-closed dual-candidate adjudication refusal."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _load(path: Path, *, where: str) -> dict[str, Any]:
    try:
        path = path.expanduser().resolve(strict=True)
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DualAdjudicationError(f"cannot load {where}: {error}") from error
    if not isinstance(value, dict):
        raise DualAdjudicationError(f"{where} must be a JSON object")
    return value


def _ref(path: Path, *, where: str) -> dict[str, str]:
    try:
        path = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise DualAdjudicationError(f"cannot resolve {where}: {error}") from error
    if not path.is_file() or path.stat().st_size <= 0:
        raise DualAdjudicationError(f"{where} is missing or empty: {path}")
    return {"path": str(path), "sha256": promotion._sha256(path)}  # noqa: SLF001


def _bound_ref(raw: Any, *, base: Path, where: str) -> Path:
    if not isinstance(raw, dict) or set(raw) != {"path", "sha256"}:
        raise DualAdjudicationError(f"{where} must be an exact file reference")
    path = Path(str(raw["path"]))
    if not path.is_absolute():
        path = base / path
    expected = _ref(path, where=where)
    if raw != expected:
        raise DualAdjudicationError(f"{where} bytes drift")
    return Path(expected["path"])


def _source_paths(report: dict[str, Any], *, kind: str) -> list[Path]:
    merge = report.get("fleet_merge")
    if (
        not isinstance(merge, dict)
        or merge.get("schema_version") != pool.POOL_SCHEMA
        or merge.get("kind") != kind
        or not isinstance(merge.get("sources"), list)
        or not merge["sources"]
    ):
        raise DualAdjudicationError(f"{kind} report lacks fleet source provenance")
    paths = []
    for index, ref in enumerate(merge["sources"]):
        paths.append(_bound_ref(ref, base=Path.cwd(), where=f"{kind}.source[{index}]"))
    return paths


def _replay_internal(path: Path, *, candidate: Path, champion: Path) -> dict[str, Any]:
    value = _load(path, where="internal pool")
    try:
        replay = pool.pool_internal(
            _source_paths(value, kind="internal_h2h"),
            candidate=candidate,
            champion=champion,
        )
    except (pool.PoolError, OSError, KeyError, ValueError) as error:
        raise DualAdjudicationError(f"internal pool replay refused: {error}") from error
    if _canonical(replay) != _canonical(value):
        raise DualAdjudicationError("internal pooled report does not replay exactly")
    return value


def _replay_neutral(path: Path, *, candidate: Path) -> dict[str, Any]:
    value = _load(path, where="neutral pool")
    try:
        replay = pool.pool_neutral(
            _source_paths(value, kind="external_panel"), checkpoint=candidate
        )
    except (pool.PoolError, OSError, KeyError, ValueError) as error:
        raise DualAdjudicationError(f"neutral pool replay refused: {error}") from error
    if _canonical(replay) != _canonical(value):
        raise DualAdjudicationError("neutral pooled report does not replay exactly")
    return value


def adjudicate(manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve(strict=True)
    manifest = _load(manifest_path, where="dual adjudication manifest")
    if set(manifest) != {
        "schema_version",
        "champion",
        "candidates",
        "tournament",
    } or manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise DualAdjudicationError("dual adjudication manifest schema/fields drift")
    champion = _bound_ref(
        manifest["champion"], base=manifest_path.parent, where="champion"
    )
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != len(IDENTITIES):
        raise DualAdjudicationError("manifest must contain exactly four candidates")
    seen: set[tuple[str, str]] = set()
    evaluated: list[dict[str, Any]] = []
    internal_science: str | None = None
    neutral_science: str | None = None
    internal_panel_seeds: str | None = None
    neutral_panel_seeds: str | None = None
    champion_sha = promotion._sha256(champion)  # noqa: SLF001
    checkpoints: dict[tuple[str, str], Path] = {}
    for index, raw in enumerate(candidates):
        if not isinstance(raw, dict) or set(raw) != {
            "arm_id",
            "subset_id",
            "training_receipt",
            "internal_pool",
            "neutral_pool",
        }:
            raise DualAdjudicationError(f"candidate {index} fields drift")
        identity = (str(raw["arm_id"]), str(raw["subset_id"]))
        if identity not in IDENTITIES or identity in seen:
            raise DualAdjudicationError(f"candidate {index} identity drift: {identity}")
        seen.add(identity)
        receipt_path = _bound_ref(
            raw["training_receipt"],
            base=manifest_path.parent,
            where=f"candidate {index} receipt",
        )
        receipt = dual_train.verify_receipt(receipt_path)
        receipt_inputs = receipt.get("inputs")
        if not isinstance(receipt_inputs, dict):
            raise DualAdjudicationError(f"candidate {index} receipt inputs drift")
        corpus_meta_ref = receipt_inputs.get("corpus_meta")
        learner_lock_ref = receipt_inputs.get("learner_lock")
        validation_ref = receipt_inputs.get("validation")
        producer_ref = receipt_inputs.get("producer")
        if not all(
            isinstance(ref, dict) and set(ref) == {"path", "sha256"}
            for ref in (
                corpus_meta_ref,
                learner_lock_ref,
                validation_ref,
                producer_ref,
            )
        ):
            raise DualAdjudicationError(f"candidate {index} receipt input refs drift")
        verified_training = dual_train.verify_inputs(
            learner_lock=Path(str(learner_lock_ref["path"])),
            reviewed_lock_file_sha256=str(learner_lock_ref["sha256"]),
            data=Path(str(corpus_meta_ref["path"])).parent,
            validation=Path(str(validation_ref["path"])),
            producer_checkpoint=Path(str(producer_ref["path"])),
        )
        receipt = dual_train.verify_receipt(
            receipt_path, verified=verified_training
        )
        if (receipt.get("arm_id"), receipt.get("subset_id")) != identity:
            raise DualAdjudicationError(f"candidate {index} receipt identity drift")
        checkpoint_ref = receipt.get("outputs", {}).get("checkpoint")
        checkpoint = _bound_ref(
            checkpoint_ref, base=receipt_path.parent, where=f"candidate {index} checkpoint"
        )
        checkpoints[identity] = checkpoint
        internal_path = _bound_ref(
            raw["internal_pool"], base=manifest_path.parent, where=f"candidate {index} internal"
        )
        neutral_path = _bound_ref(
            raw["neutral_pool"], base=manifest_path.parent, where=f"candidate {index} neutral"
        )
        internal = _replay_internal(internal_path, candidate=checkpoint, champion=champion)
        neutral = _replay_neutral(neutral_path, candidate=checkpoint)
        if internal.get("baseline_checkpoint_sha256") != champion_sha:
            raise DualAdjudicationError(f"candidate {index} used a different champion")
        if internal.get("verdict") not in {"accept_h0", "accept_h1", "continue"} or neutral.get(
            "verdict"
        ) not in {"accept_h0", "accept_h1", "continue"}:
            raise DualAdjudicationError(f"candidate {index} has invalid panel verdict")
        internal_merge = internal.get("fleet_merge")
        neutral_merge = neutral.get("fleet_merge")
        if not isinstance(internal_merge, dict) or not isinstance(neutral_merge, dict):
            raise DualAdjudicationError(f"candidate {index} lacks pooled panel provenance")
        internal_intervals = internal_merge.get("seed_intervals")
        neutral_intervals = neutral_merge.get("seed_intervals")
        if not isinstance(internal_intervals, list) or not internal_intervals:
            raise DualAdjudicationError(f"candidate {index} lacks internal seed pairs")
        if not isinstance(neutral_intervals, list) or not neutral_intervals:
            raise DualAdjudicationError(f"candidate {index} lacks neutral seed pairs")
        this_internal_science = _digest(internal.get("effective_search_config"))
        this_neutral_science = _digest(neutral.get("effective_search_config"))
        this_internal_panel = _digest(internal_intervals)
        this_neutral_panel = _digest(neutral_intervals)
        if internal_science is None:
            internal_science = this_internal_science
            neutral_science = this_neutral_science
            internal_panel_seeds = this_internal_panel
            neutral_panel_seeds = this_neutral_panel
        elif (
            internal_science != this_internal_science
            or neutral_science != this_neutral_science
            or internal_panel_seeds != this_internal_panel
            or neutral_panel_seeds != this_neutral_panel
        ):
            raise DualAdjudicationError(
                "candidate common-panel search science or seed cohort drift"
            )
        eligible = (
            internal.get("verdict") == "accept_h1"
            and neutral.get("verdict") == "accept_h1"
        )
        evaluated.append(
            {
                "arm_id": identity[0],
                "subset_id": identity[1],
                "training_receipt": _ref(receipt_path, where="training receipt"),
                "checkpoint": _ref(checkpoint, where="candidate checkpoint"),
                "internal_pool": _ref(internal_path, where="internal pool"),
                "neutral_pool": _ref(neutral_path, where="neutral pool"),
                "internal_verdict": internal.get("verdict"),
                "neutral_verdict": neutral.get("verdict"),
                "internal_llr": float(internal["pentanomial_sprt"]["llr"]),
                "neutral_llr": float(neutral["pentanomial_sprt"]["llr"]),
                "eligible": eligible,
            }
        )
    if seen != set(IDENTITIES):
        raise DualAdjudicationError("manifest does not exactly cover four dual identities")
    evaluated_by_identity = {
        (item["arm_id"], item["subset_id"]): item for item in evaluated
    }

    def parse_identity(value: Any, *, where: str) -> tuple[str, str]:
        if not isinstance(value, str) or "/" not in value:
            raise DualAdjudicationError(f"{where} identity is malformed")
        identity = tuple(value.split("/", 1))
        if identity not in IDENTITIES:
            raise DualAdjudicationError(f"{where} identity is unauthorized: {identity}")
        return identity  # type: ignore[return-value]

    def direct_match(raw: Any, *, where: str) -> dict[str, Any]:
        nonlocal internal_science
        if not isinstance(raw, dict) or set(raw) != {"candidate", "baseline", "pool"}:
            raise DualAdjudicationError(f"{where} fields drift")
        candidate_id = parse_identity(raw["candidate"], where=f"{where}.candidate")
        baseline_id = parse_identity(raw["baseline"], where=f"{where}.baseline")
        if candidate_id == baseline_id:
            raise DualAdjudicationError(f"{where} cannot compare a candidate to itself")
        report_path = _bound_ref(
            raw["pool"], base=manifest_path.parent, where=f"{where}.pool"
        )
        report = _replay_internal(
            report_path,
            candidate=checkpoints[candidate_id],
            champion=checkpoints[baseline_id],
        )
        science = _digest(report.get("effective_search_config"))
        if internal_science != science:
            raise DualAdjudicationError(f"{where} fixed-search science drift")
        verdict = report.get("verdict")
        if verdict not in {"accept_h0", "accept_h1", "continue"}:
            raise DualAdjudicationError(f"{where} has invalid SPRT verdict")
        winner_id = (
            candidate_id
            if verdict == "accept_h1"
            else baseline_id
            if verdict == "accept_h0"
            else None
        )
        return {
            "candidate": "/".join(candidate_id),
            "baseline": "/".join(baseline_id),
            "pool": _ref(report_path, where=f"{where} pool"),
            "verdict": verdict,
            "winner": None if winner_id is None else "/".join(winner_id),
            # Recorded for audit only. LLR is deliberately never compared
            # across matches because stopping times/sample sizes differ.
            "llr": float(report["pentanomial_sprt"]["llr"]),
            "candidate_win_rate": float(report["candidate_win_rate"]),
            "complete_pairs": int(report["complete_pairs"]),
        }

    tournament = manifest.get("tournament")
    if not isinstance(tournament, dict) or set(tournament) != {
        "causal_teacher",
        "n128_quantity_curve",
        "finalist",
    }:
        raise DualAdjudicationError("tournament fields drift")
    causal = direct_match(tournament["causal_teacher"], where="causal_teacher")
    causal_pair = {causal["candidate"], causal["baseline"]}
    if causal_pair != {"n256/full-56k", "n128/matched-56k"}:
        raise DualAdjudicationError("causal teacher match must be n256-56k vs n128-56k")

    quantity_raw = tournament["n128_quantity_curve"]
    if not isinstance(quantity_raw, list) or len(quantity_raw) != 3:
        raise DualAdjudicationError("n128 quantity curve requires all three pairwise matches")
    quantity = [
        direct_match(raw, where=f"n128_quantity_curve[{index}]")
        for index, raw in enumerate(quantity_raw)
    ]
    n128_ids = {
        "n128/matched-56k",
        "n128/compute-112k",
        "n128/full-140k",
    }
    expected_pairs = {
        frozenset(pair) for pair in (
            ("n128/matched-56k", "n128/compute-112k"),
            ("n128/matched-56k", "n128/full-140k"),
            ("n128/compute-112k", "n128/full-140k"),
        )
    }
    actual_pairs = {
        frozenset((match["candidate"], match["baseline"])) for match in quantity
    }
    if actual_pairs != expected_pairs:
        raise DualAdjudicationError("n128 quantity curve does not cover every direct pair")
    quantity_wins = {identity: 0 for identity in n128_ids}
    for match in quantity:
        if match["winner"] in quantity_wins:
            quantity_wins[match["winner"]] += 1
    n128_finalists = [identity for identity, wins in quantity_wins.items() if wins == 2]
    n128_finalist = n128_finalists[0] if len(n128_finalists) == 1 else None

    finalist = direct_match(tournament["finalist"], where="finalist")
    finalist_pair = {finalist["candidate"], finalist["baseline"]}
    valid_finalist_pair = (
        n128_finalist is not None
        and finalist_pair == {"n256/full-56k", n128_finalist}
    )
    winner_identity = finalist["winner"] if valid_finalist_pair else None
    winner_tuple = (
        parse_identity(winner_identity, where="finalist.winner")
        if winner_identity is not None
        else None
    )
    winner = None if winner_tuple is None else evaluated_by_identity[winner_tuple]
    promotion_ready = bool(winner is not None and winner["eligible"])
    result = {
        "schema_version": RESULT_SCHEMA,
        "passed": promotion_ready,
        "decision": (
            "winner_selected_full_evidence_required"
            if promotion_ready
            else "no_promotion_inconclusive_or_vetoed"
        ),
        "manifest": _ref(manifest_path, where="dual manifest"),
        "adjudicator": _ref(Path(__file__), where="dual adjudicator"),
        "champion": _ref(champion, where="champion"),
        "fixed_search": {
            "internal_effective_search_config_sha256": internal_science,
            "neutral_effective_search_config_sha256": neutral_science,
            "internal_seed_intervals_sha256": internal_panel_seeds,
            "neutral_seed_intervals_sha256": neutral_panel_seeds,
        },
        "candidates": evaluated,
        "tournament": {
            "causal_teacher": causal,
            "n128_quantity_curve": quantity,
            "n128_condorcet_finalist": n128_finalist,
            "finalist": finalist,
        },
        "winner": winner,
        "promotion": {
            "ready": False,
            "reason": "winner still requires calibration/high-regret/bucket evidence",
        },
    }
    result["adjudication_sha256"] = _digest(result)
    return result


def write_result(path: Path, value: dict[str, Any]) -> None:
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
    value = _load(path, where="dual adjudication")
    stated = value.get("adjudication_sha256")
    unhashed = dict(value)
    unhashed.pop("adjudication_sha256", None)
    if (
        value.get("schema_version") != RESULT_SCHEMA
        or not isinstance(value.get("passed"), bool)
        or stated != _digest(unhashed)
    ):
        raise DualAdjudicationError("dual adjudication digest/status drift")
    replay = adjudicate(Path(value["manifest"]["path"]))
    if replay != value:
        raise DualAdjudicationError("dual adjudication no longer replays")
    return value


def build_promotion_plan(
    selection_path: Path, promotion_manifest_path: Path
) -> dict[str, Any]:
    selection_path = selection_path.expanduser().resolve(strict=True)
    selection = verify_result(selection_path)
    winner = selection.get("winner")
    if selection.get("passed") is not True or not isinstance(winner, dict):
        raise DualAdjudicationError("selection has no promotion-eligible direct winner")
    promotion_manifest_path = promotion_manifest_path.expanduser().resolve(strict=True)
    value = _load(promotion_manifest_path, where="winner promotion manifest")
    if set(value) != {
        "schema_version",
        "registry",
        "current_pointer",
        "contract_lock",
        "adjudication",
        "training_receipt",
        "receipt",
        "reason",
    } or value.get("schema_version") != "a1-dual-arm-winner-promotion-manifest-v1":
        raise DualAdjudicationError("winner promotion manifest schema/fields drift")
    base = promotion_manifest_path.parent
    registry = _bound_ref(value["registry"], base=base, where="registry")
    current = _bound_ref(value["current_pointer"], base=base, where="current pointer")
    contract_lock = _bound_ref(value["contract_lock"], base=base, where="contract lock")
    final_adjudication = _bound_ref(
        value["adjudication"], base=base, where="final promotion adjudication"
    )
    training_receipt = _bound_ref(
        value["training_receipt"], base=base, where="winner training receipt"
    )
    if _ref(training_receipt, where="winner training receipt") != winner.get(
        "training_receipt"
    ):
        raise DualAdjudicationError("promotion manifest receipt is not the selected winner")
    receipt = Path(str(value["receipt"])).expanduser().resolve(strict=False)
    reason = str(value["reason"])
    if not reason.strip():
        raise DualAdjudicationError("promotion reason must be nonempty")
    try:
        promotion.prepare_promotion(
            registry_path=registry,
            current_pointer=current,
            contract_lock=contract_lock,
            adjudication_path=final_adjudication,
            training_receipt=training_receipt,
            receipt_path=receipt,
            reason=reason,
        )
    except promotion.PromotionError as error:
        raise DualAdjudicationError(
            f"winner full-evidence promotion preflight refused: {error}"
        ) from error
    command = [
        sys.executable,
        str(_REPO_ROOT / "tools" / "a1_promotion_transaction.py"),
        "promote",
        "--registry",
        str(registry),
        "--current-pointer",
        str(current),
        "--contract-lock",
        str(contract_lock),
        "--adjudication",
        str(final_adjudication),
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
        "selection": _ref(selection_path, where="dual selection"),
        "promotion_manifest": _ref(
            promotion_manifest_path, where="promotion manifest"
        ),
        "winner": winner,
        "command": command,
        "command_sha256": _digest(command),
    }
    result["plan_sha256"] = _digest(result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    select = sub.add_parser("select")
    select.add_argument("--manifest", type=Path, required=True)
    select.add_argument("--out", type=Path, required=True)
    verify = sub.add_parser("verify-selection")
    verify.add_argument("--out", type=Path, required=True)
    promote = sub.add_parser("promotion-plan")
    promote.add_argument("--selection", type=Path, required=True)
    promote.add_argument("--promotion-manifest", type=Path, required=True)
    promote.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "select":
            value = adjudicate(args.manifest)
        elif args.command == "verify-selection":
            value = verify_result(args.out)
        else:
            value = build_promotion_plan(args.selection, args.promotion_manifest)
        if args.command != "verify-selection":
            write_result(args.out, value)
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    except (DualAdjudicationError, dual_train.DualTrainError, OSError) as error:
        parser.exit(2, f"REFUSED: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
