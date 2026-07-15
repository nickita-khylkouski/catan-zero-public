#!/usr/bin/env python3
"""Launch, poll, or collect one sealed A-D 64-H100 evaluation matrix."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
for import_root in (REPO_ROOT / "src", REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from tools.fleet import a1_h100_eval_fleet as fleet  # noqa: E402
from tools.fleet import a1_lr_dose_eval_matrix as matrix_tool  # noqa: E402


class ExecuteError(RuntimeError):
    """The sealed matrix cannot be executed exactly as planned."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExecuteError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise ExecuteError(f"{path} must contain one JSON object")
    return value


def load_matrix(
    path: Path,
) -> tuple[Path, dict[str, Any], dict[str, Any], list[tuple[dict[str, Any], dict[str, Any]]]]:
    matrix_path = path.expanduser().resolve(strict=True)
    matrix = _read_object(matrix_path)
    if matrix.get("schema_version") != matrix_tool.SCHEMA:
        raise ExecuteError("unsupported A-D evaluation matrix schema")
    unsigned = dict(matrix)
    stated_digest = unsigned.pop("state_sha256", None)
    if stated_digest != _digest(unsigned):
        raise ExecuteError("evaluation matrix content digest drift")
    if matrix.get("operator_selection_status") != "adopted_teacher_campaign":
        raise ExecuteError("evaluation matrix does not bind an adopted teacher")
    science_contract = matrix_tool.current_science.load()
    if matrix.get("operator_selection") != science_contract.get("operator_selection"):
        raise ExecuteError("current teacher adoption differs from the sealed matrix")
    if matrix.get("selected_operator") != science_contract["operator_selection"].get(
        "selected_operator"
    ):
        raise ExecuteError("selected teacher operator differs from the sealed matrix")
    if matrix.get("science_contract_sha256") != matrix_tool._file_sha256(
        matrix_tool.current_science.CONTRACT_PATH
    ):
        raise ExecuteError("current science contract bytes differ from the matrix")
    training_campaign = matrix.get("training_campaign")
    if not isinstance(training_campaign, dict):
        raise ExecuteError("evaluation matrix has no authenticated training campaign")
    campaign_path = Path(str(training_campaign.get("path"))).resolve(strict=True)
    campaign = matrix_tool.lr_campaign._load_bound_json(
        campaign_path, schema=matrix_tool.lr_campaign.SCHEMA
    )
    if (
        matrix_tool._file_sha256(campaign_path)
        != training_campaign.get("file_sha256")
        or campaign.get("campaign_sha256")
        != training_campaign.get("campaign_sha256")
    ):
        raise ExecuteError("training campaign differs from the sealed matrix")
    arm_receipts = training_campaign.get("arm_receipts")
    if not isinstance(arm_receipts, dict) or set(arm_receipts) != set(
        matrix_tool.ARMS
    ):
        raise ExecuteError("evaluation matrix arm receipt set drift")
    stored_upgrade = training_campaign.get("function_preserving_upgrade")
    if not isinstance(stored_upgrade, dict):
        raise ExecuteError("evaluation matrix has no function-preserving parent")
    candidates = {
        arm: Path(arm_receipts[arm]["artifacts"]["checkpoint"]["path"])
        for arm in matrix_tool.ARMS
    }
    authenticated_upgrade, authenticated_receipts = (
        matrix_tool._authenticate_completed_arms(
            campaign,
            f7=Path(str(stored_upgrade["source"]["path"])),
            candidates=candidates,
        )
    )
    if (
        authenticated_upgrade != stored_upgrade
        or authenticated_receipts != arm_receipts
    ):
        raise ExecuteError("completed training lineage differs from the sealed matrix")
    if matrix.get("physical_gpus") != 64:
        raise ExecuteError("evaluation matrix does not bind exactly 64 H100s")
    rows = matrix.get("matchups")
    if not isinstance(rows, list) or len(rows) != 8:
        raise ExecuteError("evaluation matrix must contain exactly eight matchups")
    expected_matchups = {
        (arm, baseline) for arm, baseline, _aliases in matrix_tool.MATCHUPS
    }
    actual_matchups = {
        (row.get("arm"), row.get("baseline"))
        for row in rows
        if isinstance(row, dict)
    }
    if actual_matchups != expected_matchups:
        raise ExecuteError("evaluation matrix A-D/baseline matchup set drift")

    manifest_path = Path(str(matrix["manifest"])).resolve(strict=True)
    manifest = fleet.load_manifest(
        manifest_path, expected_shapes=fleet.FULL_EXPECTED_SHAPES
    )
    if manifest["manifest_hash"] != matrix.get("manifest_hash"):
        raise ExecuteError("fleet manifest differs from the sealed matrix")

    loaded: list[tuple[dict[str, Any], dict[str, Any]]] = []
    occupied: set[str] = set()
    plan_hashes: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ExecuteError("matrix matchup row is not an object")
        plan_path = Path(str(row["plan"])).resolve(strict=True)
        plan = fleet.load_plan(plan_path, manifest)
        if plan["plan_hash"] != row.get("plan_hash"):
            raise ExecuteError(f"plan hash drift for {row.get('arm')}-{row.get('baseline')}")
        if plan["operator_mode"] != fleet.COHERENT_PUBLIC_OPERATOR:
            raise ExecuteError("matrix contains a non-coherent evaluator plan")
        arm = str(row["arm"])
        expected_parent = stored_upgrade["upgraded_initializer"]
        if plan["evaluation_binding"].get("candidate_parent") != expected_parent:
            raise ExecuteError("plan candidate parent is not the upgraded initializer")
        completed_candidate = arm_receipts[arm]["artifacts"]["checkpoint"]
        if (
            plan["candidate"]["source"] != completed_candidate["path"]
            or plan["candidate"]["sha256"] != completed_candidate["sha256"]
        ):
            raise ExecuteError(f"plan candidate differs from completed arm {arm}")
        if row["baseline"] == "f7" and plan["champion"]["sha256"] != stored_upgrade[
            "source"
        ]["sha256"]:
            raise ExecuteError("historical comparison baseline is not raw f7")
        if plan["science_config_hash"] != matrix.get("science_config_hash"):
            raise ExecuteError("matrix plans do not share the adopted teacher operator")
        if plan.get("seed_cohort_id") != matrix.get("seed_cohort_id"):
            raise ExecuteError("matrix plan common-random-number cohort drift")
        if plan.get("pair_claims", {}).get("internal") != matrix.get("internal_claim"):
            raise ExecuteError("matrix plan paired-seed interval drift")
        external_claim = dict(matrix.get("external_placeholder_claim", {}))
        external_claim.pop("launch", None)
        if plan.get("pair_claims", {}).get("external_matched") != external_claim:
            raise ExecuteError("matrix plan external placeholder interval drift")
        internal_jobs = [job for job in plan["jobs"] if job["phase"] == "internal"]
        slots = {str(job["slot_id"]) for job in internal_jobs}
        if slots != set(row.get("gpu_slots", [])) or len(slots) != 8:
            raise ExecuteError("matrix plan GPU allocation drift")
        if occupied & slots:
            raise ExecuteError("matrix assigns a physical GPU more than once")
        occupied |= slots
        plan_hashes.add(plan["plan_hash"])
        loaded.append((row, plan))
    expected = {str(slot["slot_id"]) for slot in fleet.gpu_slots(manifest)}
    if occupied != expected or len(plan_hashes) != 8:
        raise ExecuteError("matrix does not cover all 64 H100s exactly once")
    return manifest_path, manifest, matrix, loaded


def _parallel(
    loaded: list[tuple[dict[str, Any], dict[str, Any]]],
    operation: Callable[[dict[str, Any], dict[str, Any]], Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    results: list[dict[str, Any]] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(operation, row, plan): (row, plan) for row, plan in loaded
        }
        for future in as_completed(futures):
            row, plan = futures[future]
            key = f"{row['arm']}-vs-{row['baseline']}"
            try:
                payload = future.result()
            except Exception as error:  # preserve all other matchup results
                failures.append(f"{key}: {type(error).__name__}: {error}")
            else:
                results.append(
                    {
                        "matchup": key,
                        "plan_hash": plan["plan_hash"],
                        "result": payload,
                    }
                )
    return sorted(results, key=lambda value: value["matchup"]), sorted(failures)


def execute(path: Path, *, command: str) -> dict[str, Any]:
    _manifest_path, manifest, matrix, loaded = load_matrix(path)
    if command == "launch":
        rows, failures = _parallel(
            loaded,
            lambda _row, plan: fleet.launch_phase(manifest, plan, "internal"),
        )
    elif command == "status":
        rows, failures = _parallel(
            loaded,
            lambda _row, plan: fleet.status_phase(manifest, plan, "internal"),
        )
    elif command == "collect":
        rows, failures = _parallel(
            loaded,
            lambda row, plan: fleet.collect_phase(
                manifest,
                plan,
                "internal",
                Path(str(row["collect_output_dir"])),
            ),
        )
    else:  # pragma: no cover - argparse seals the command vocabulary.
        raise AssertionError(command)
    result = {
        "schema_version": "a1-lr-dose-eval-matrix-operation-v1",
        "operation": command,
        "matrix": str(path.expanduser().resolve(strict=True)),
        "matrix_state_sha256": matrix["state_sha256"],
        "matchups": rows,
        "failures": failures,
        "ok": not failures,
    }
    result["state_sha256"] = _digest(result)
    return result


def wait_for_completion(
    path: Path, *, poll_seconds: float, timeout_seconds: float
) -> dict[str, Any]:
    if poll_seconds <= 0.0 or timeout_seconds <= 0.0:
        raise ExecuteError("poll and timeout seconds must be positive")
    started = time.monotonic()
    polls = 0
    while True:
        status = execute(path, command="status")
        polls += 1
        if status["failures"]:
            raise ExecuteError(
                "matrix status failed: " + " | ".join(status["failures"])
            )
        counts = {
            state: sum(
                int(row["result"]["counts"].get(state, 0))
                for row in status["matchups"]
            )
            for state in ("done", "active", "failed", "stale", "missing", "unsafe")
        }
        bad = {state: counts[state] for state in ("failed", "stale", "missing", "unsafe") if counts[state]}
        if bad:
            raise ExecuteError(f"matrix entered a non-runnable terminal state: {bad}")
        if counts["done"] == 64 and counts["active"] == 0:
            result = {
                "schema_version": "a1-lr-dose-eval-matrix-completion-v1",
                "matrix": str(path.expanduser().resolve(strict=True)),
                "matrix_state_sha256": status["matrix_state_sha256"],
                "polls": polls,
                "elapsed_seconds": time.monotonic() - started,
                "counts": counts,
                "final_status": status,
                "ok": True,
            }
            result["state_sha256"] = _digest(result)
            return result
        if counts["done"] + counts["active"] != 64:
            raise ExecuteError(f"matrix status does not cover 64 jobs: {counts}")
        elapsed = time.monotonic() - started
        if elapsed >= timeout_seconds:
            raise ExecuteError(
                f"matrix did not finish within {timeout_seconds:g}s: {counts}"
            )
        time.sleep(min(poll_seconds, max(0.0, timeout_seconds - elapsed)))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("launch", "collect"):
        command = commands.add_parser(name)
        command.add_argument("--go", action="store_true", required=True)
        command.add_argument("--out", type=Path, required=True)
    wait = commands.add_parser("wait")
    wait.add_argument("--poll-seconds", type=float, default=30.0)
    wait.add_argument("--timeout-seconds", type=float, default=7200.0)
    wait.add_argument("--out", type=Path, required=True)
    commands.add_parser("status")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command in {"launch", "wait", "collect"} and args.out.expanduser().exists():
            raise ExecuteError(f"refusing to overwrite operation receipt {args.out}")
        result = (
            wait_for_completion(
                args.matrix,
                poll_seconds=args.poll_seconds,
                timeout_seconds=args.timeout_seconds,
            )
            if args.command == "wait"
            else execute(args.matrix, command=args.command)
        )
        if args.command in {"launch", "wait", "collect"}:
            fleet.write_new_readonly(args.out, result)
            matchup_rows = result.get("matchups")
            if matchup_rows is None:
                matchup_rows = result.get("final_status", {}).get("matchups", [])
            summary: dict[str, Any] = {
                "receipt": str(args.out.expanduser().resolve(strict=True)),
                "state_sha256": result["state_sha256"],
                "matchups": len(matchup_rows),
            }
        else:
            summary = result
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if result.get("ok", True) else 2
    except (
        ExecuteError,
        matrix_tool.lr_campaign.CampaignError,
        fleet.FleetError,
        OSError,
        KeyError,
        ValueError,
    ) as error:
        print(f"A-D evaluation matrix operation refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
