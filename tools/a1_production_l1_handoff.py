#!/usr/bin/env python3
"""Seal the read-only evidence handoff for the production L1 candidate.

The bundle is intentionally non-authorizing while the direct r3-v-f7 result is
pending.  It snapshots evidence and the current registry/pointer bytes, and
documents the authoritative transaction's dry-run/commit/recovery interface;
it never invokes that transaction or mutates either pointer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence


SCHEMA = "a1-production-l1-promotion-handoff-v1"


class HandoffError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise HandoffError(f"{path} must contain an object")
    return value


def _ref(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise HandoffError(f"artifact is not a file: {resolved}")
    return {"path": str(resolved), "sha256": _sha(resolved)}


def _verify_ref(value: Any, label: str) -> Path:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise HandoffError(f"{label} reference is malformed")
    path = Path(str(value["path"])).resolve(strict=True)
    if _sha(path) != value["sha256"]:
        raise HandoffError(f"{label} bytes drifted")
    return path


def _checkpoint_sha(payload: dict[str, Any], label: str) -> str:
    value = payload.get("candidate_checkpoint_sha256")
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise HandoffError(f"{label} has no candidate checkpoint SHA")
    return value


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    candidate = _ref(args.candidate)
    report = _ref(args.report)
    completion = _ref(args.completion)
    report_payload = _load(Path(report["path"]))
    completion_payload = _load(Path(completion["path"]))
    completion_unhashed = dict(completion_payload)
    stated_completion = completion_unhashed.pop("receipt_sha256", None)
    if stated_completion != _digest(completion_unhashed):
        raise HandoffError("completion receipt semantic digest drift")
    if (
        completion_payload.get("production_eligible") is not True
        or completion_payload.get("checkpoint") != candidate
        or completion_payload.get("report") != report
        or report_payload.get("steps_completed") != 1024
        or report_payload.get("world_size") != 8
        or report_payload.get("batch_size") != 512
        or report_payload.get("init_checkpoint_sha256") != args.f7_sha256
    ):
        raise HandoffError("finalized learner lineage/selected dose drift")

    stage1 = _ref(args.stage1)
    stage1_payload = _load(Path(stage1["path"]))
    if _checkpoint_sha(stage1_payload, "stage1") != candidate["sha256"]:
        raise HandoffError("stage1 evaluated a different candidate")

    external_candidate = _ref(args.external_candidate)
    external_champion = _ref(args.external_champion)
    external_matched = _ref(args.external_matched)
    candidate_external_payload = _load(Path(external_candidate["path"]))
    champion_external_payload = _load(Path(external_champion["path"]))
    matched_payload = _load(Path(external_matched["path"]))
    if _checkpoint_sha(candidate_external_payload, "external candidate") != candidate["sha256"]:
        raise HandoffError("external panel evaluated a different candidate")
    if _checkpoint_sha(champion_external_payload, "external champion") != args.f7_sha256:
        raise HandoffError("external panel used a different f7 incumbent")
    expected_delta = (
        float(candidate_external_payload["candidate_win_rate"])
        - float(champion_external_payload["candidate_win_rate"])
    )
    if abs(float(matched_payload["candidate_minus_champion"]) - expected_delta) > 1e-12:
        raise HandoffError("matched external delta does not replay")

    direct_plan = _ref(args.direct_plan)
    direct_plan_payload = _load(Path(direct_plan["path"]))
    if (
        direct_plan_payload.get("candidate", {}).get("sha256") != candidate["sha256"]
        or direct_plan_payload.get("champion", {}).get("sha256") != args.f7_sha256
        or direct_plan_payload.get("evaluation_binding", {}).get("promotion_eligible") is not True
        or direct_plan_payload.get("evaluation_binding", {}).get("comparison_mode")
        != "promotion_parent"
    ):
        raise HandoffError("pending direct plan is not the r3-v-f7 promotion-parent panel")
    direct_result = args.direct_result.expanduser().resolve(strict=False)
    direct_evidence: dict[str, Any]
    direct_complete = direct_result.exists()
    if direct_complete:
        direct_ref = _ref(direct_result)
        direct_payload = _load(Path(direct_ref["path"]))
        if (
            _checkpoint_sha(direct_payload, "direct result") != candidate["sha256"]
            or direct_payload.get("baseline_checkpoint_sha256") != args.f7_sha256
            or direct_payload.get("verdict") not in {"H1", "accept_h1"}
            or direct_payload.get("errors") not in {None, []}
        ):
            raise HandoffError("completed direct result is not a clean r3-v-f7 H1")
        direct_evidence = {
            "status": "complete_h1",
            "plan": direct_plan,
            "artifact": direct_ref,
            "plan_semantic_sha256": direct_plan_payload.get("plan_hash"),
            "run_id": direct_plan_payload.get("run_id"),
            "candidate_wins": direct_payload.get("candidate_wins"),
            "baseline_wins": direct_payload.get("baseline_wins"),
            "candidate_win_rate": direct_payload.get("candidate_win_rate"),
            "verdict": direct_payload.get("verdict"),
            "required_candidate_sha256": candidate["sha256"],
            "required_champion_sha256": args.f7_sha256,
        }
    else:
        direct_evidence = {
            "status": "pending",
            "plan": direct_plan,
            "plan_semantic_sha256": direct_plan_payload.get("plan_hash"),
            "run_id": direct_plan_payload.get("run_id"),
            "expected_result": str(direct_result),
            "required_candidate_sha256": candidate["sha256"],
            "required_champion_sha256": args.f7_sha256,
        }

    transaction_tool = _ref(args.transaction_tool)
    registry = _ref(args.registry)
    current_pointer = _ref(args.current_pointer)
    receipt_placeholder = str(args.promotion_receipt.expanduser().resolve(strict=False))
    dry_run = [
        "python3", transaction_tool["path"], "promote",
        "--registry", registry["path"],
        "--current-pointer", current_pointer["path"],
        "--contract-lock", "<SEALED_CONTRACT_LOCK>",
        "--adjudication", "<PASSING_ADJUDICATION>",
        "--training-receipt", completion["path"],
        "--cohort-exclusions", "<SEALED_COHORT_EXCLUSIONS>",
        "--receipt", receipt_placeholder,
        "--reason", "promote production L1 after complete evidence replay",
    ]
    bundle: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": (
            "direct_complete_pending_typed_adjudication"
            if direct_complete
            else "awaiting_direct_r3_vs_f7"
        ),
        "promotion_ready": False,
        "pointer_mutation_authorized": False,
        "learner": {
            "candidate": candidate,
            "report": report,
            "completion_receipt": completion,
            "completion_receipt_sha256": stated_completion,
            "selected_dose": {
                "optimizer_steps": 1024,
                "world_size": 8,
                "per_rank_batch_size": 512,
                "global_samples": 4_194_304,
            },
            "f7_parent_sha256": args.f7_sha256,
        },
        "evidence": {
            "stage1_vs_historical_l1": {
                "artifact": stage1,
                "candidate_win_rate": stage1_payload.get("candidate_win_rate"),
                "verdict": stage1_payload.get("verdict"),
            },
            "external_matched_vs_f7": {
                "candidate": external_candidate,
                "champion": external_champion,
                "comparison": external_matched,
                "candidate_win_rate": candidate_external_payload["candidate_win_rate"],
                "champion_win_rate": champion_external_payload["candidate_win_rate"],
                "candidate_minus_champion": expected_delta,
            },
            "direct_r3_vs_f7": direct_evidence,
        },
        "authoritative_transaction_audit": {
            "tool": transaction_tool,
            "registry_snapshot": registry,
            "current_pointer_snapshot": current_pointer,
            "dry_run_argv": dry_run,
            "commit_argv": [*dry_run, "--go"],
            "rollback_dry_run_argv": [
                "python3", transaction_tool["path"], "recover",
                "--receipt", receipt_placeholder,
            ],
            "rollback_commit_argv": [
                "python3", transaction_tool["path"], "recover",
                "--receipt", receipt_placeholder, "--go",
            ],
            "automatic_commit_failure_behavior": (
                "transaction writes local backups first and atomically restores exact "
                "registry/current-pointer before bytes on a commit exception"
            ),
        },
        "blockers": [
            *([] if direct_complete else ["direct r3-v-f7 promotion-parent result is pending"]),
            "passing typed adjudication and complete required evidence set are not sealed",
            "promotion cohort-exclusions manifest is not sealed",
        ],
    }
    bundle["bundle_sha256"] = _digest(bundle)
    return bundle


def verify(path: Path) -> dict[str, Any]:
    value = _load(path.resolve(strict=True))
    stated = value.get("bundle_sha256")
    unhashed = {key: item for key, item in value.items() if key != "bundle_sha256"}
    if value.get("schema_version") != SCHEMA or stated != _digest(unhashed):
        raise HandoffError("handoff bundle digest/schema drift")
    if value.get("promotion_ready") is not False or value.get(
        "pointer_mutation_authorized"
    ) is not False:
        raise HandoffError("pending handoff unexpectedly authorizes promotion")
    learner = value["learner"]
    for key in ("candidate", "report", "completion_receipt"):
        _verify_ref(learner[key], f"learner.{key}")
    evidence = value["evidence"]
    _verify_ref(evidence["stage1_vs_historical_l1"]["artifact"], "stage1")
    for key in ("candidate", "champion", "comparison"):
        _verify_ref(evidence["external_matched_vs_f7"][key], f"external.{key}")
    _verify_ref(evidence["direct_r3_vs_f7"]["plan"], "direct plan")
    if evidence["direct_r3_vs_f7"]["status"] == "complete_h1":
        _verify_ref(evidence["direct_r3_vs_f7"]["artifact"], "direct result")
    audit = value["authoritative_transaction_audit"]
    for key in ("tool", "registry_snapshot", "current_pointer_snapshot"):
        _verify_ref(audit[key], f"transaction.{key}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--completion", required=True, type=Path)
    parser.add_argument("--f7-sha256", required=True)
    parser.add_argument("--stage1", required=True, type=Path)
    parser.add_argument("--external-candidate", required=True, type=Path)
    parser.add_argument("--external-champion", required=True, type=Path)
    parser.add_argument("--external-matched", required=True, type=Path)
    parser.add_argument("--direct-plan", required=True, type=Path)
    parser.add_argument("--direct-result", required=True, type=Path)
    parser.add_argument("--transaction-tool", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--current-pointer", required=True, type=Path)
    parser.add_argument("--promotion-receipt", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        value = prepare(args)
        out = args.out.expanduser().resolve(strict=False)
        out.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444)
        with os.fdopen(fd, "wb") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        verify(out)
        print(json.dumps({"prepared": True, "promotion_ready": False,
                          "bundle_sha256": value["bundle_sha256"]}, sort_keys=True))
        return 0
    except (HandoffError, OSError, KeyError, ValueError, json.JSONDecodeError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
