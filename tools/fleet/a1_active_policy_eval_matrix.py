#!/usr/bin/env python3
"""Seal and execute the 64-H100 active-policy candidate evaluation matrix.

The coherent-n128 learner campaign names P10/P25/P50/P100 rather than the
historical A/B/C/D LR arms.  This controller authenticates the campaign's
in-budget arms, maps every eligible terminal checkpoint against both exact f7
and the registry v5 incumbent, and partitions the full 64-H100 fleet across
those matchups.  Every matchup reuses one common paired-seed cohort and swaps
colors inside each pair.

Only the internal Rust-native coherent-public panel is authorized here.  The
external pair claim remains an unlaunched placeholder so the ordinary fleet
plan schema and validation-ledger rules stay intact.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
for import_root in (REPO_ROOT / "src", REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from tools import a1_b200_active_policy_campaign as active_campaign  # noqa: E402
from tools import a1_current_science_contract as current_science  # noqa: E402
from tools import a1_function_preserving_upgrade as architecture_upgrade  # noqa: E402
from tools.champion_registry import ChampionRegistry  # noqa: E402
from tools.fleet import a1_h100_eval_fleet as fleet  # noqa: E402


SCHEMA = "a1-active-policy-eval-matrix-v1"
OPERATION_SCHEMA = "a1-active-policy-eval-matrix-operation-v1"
COMPLETION_SCHEMA = "a1-active-policy-eval-matrix-completion-v1"
BASELINES = ("f7", "v5")


class MatrixError(RuntimeError):
    """The active-policy evaluation matrix is incomplete or inconsistent."""


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_object(path: Path, *, where: str) -> tuple[Path, dict[str, Any]]:
    resolved = path.expanduser().resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise MatrixError(f"{where} must be a regular file: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MatrixError(f"cannot read {where}: {error}") from error
    if not isinstance(value, dict):
        raise MatrixError(f"{where} must contain one JSON object")
    return resolved, value


def _load_selection(
    path: Path, *, campaign_path: Path, campaign: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]]:
    try:
        resolved, selection = active_campaign._load_signed(  # noqa: SLF001
            path,
            where="active-policy selection",
            schema=active_campaign.SELECTION_SCHEMA,
            digest_field="selection_sha256",
        )
    except active_campaign.CampaignError as error:
        raise MatrixError(str(error)) from error
    campaign_ref = selection.get("campaign")
    eligible = selection.get("eligible_arms")
    if (
        not isinstance(campaign_ref, dict)
        or Path(str(campaign_ref.get("path", ""))).resolve(strict=True)
        != campaign_path
        or campaign_ref.get("file_sha256") != _file_sha256(campaign_path)
        or campaign_ref.get("campaign_sha256") != campaign["campaign_sha256"]
        or not isinstance(eligible, list)
        or not eligible
        or len(eligible) != len(set(eligible))
        or any(arm not in active_campaign.ARMS for arm in eligible)
        or selection.get("candidate_chaining") is not False
        or selection.get("playing_strength_evaluation_still_required") is not True
        or selection.get("winner") not in eligible
    ):
        raise MatrixError("active-policy selection lost campaign/eligibility semantics")
    canonical_order = [arm for arm in active_campaign.ARMS if arm in set(eligible)]
    if eligible != canonical_order:
        raise MatrixError("active-policy eligible arms are not in canonical dose order")
    return resolved, selection


def _load_authority(
    *, campaign_path: Path, selection_path: Path, registry_path: Path
) -> dict[str, Any]:
    try:
        resolved_campaign, campaign = active_campaign._load_campaign(campaign_path)  # noqa: SLF001
    except active_campaign.CampaignError as error:
        raise MatrixError(str(error)) from error
    resolved_selection, selection = _load_selection(
        selection_path, campaign_path=resolved_campaign, campaign=campaign
    )

    upgrade_path = Path(campaign["inputs"]["architecture_upgrade_receipt"])
    try:
        upgrade = architecture_upgrade.verify_receipt(upgrade_path)
    except architecture_upgrade.UpgradeError as error:
        raise MatrixError(f"function-preserving f7 upgrade refused: {error}") from error
    f7 = Path(str(upgrade["source"]["path"])).resolve(strict=True)
    initializer = Path(str(upgrade["upgraded_initializer"]["path"])).resolve(
        strict=True
    )
    if (
        upgrade["source"].get("sha256")
        != active_campaign.EXPECTED_F7_PARENT_SHA256
        or _file_sha256(f7) != upgrade["source"]["sha256"]
        or _file_sha256(initializer)
        != campaign["lineage_contract"]["upgraded_initializer_sha256"]
        or upgrade["upgraded_initializer"].get("sha256")
        != campaign["lineage_contract"]["upgraded_initializer_sha256"]
    ):
        raise MatrixError("campaign does not map exact f7 to its authenticated initializer")

    registry_path = registry_path.expanduser().resolve(strict=True)
    registry = ChampionRegistry.load(registry_path)
    pointer = registry.get_role("generator_champion")
    if pointer is None:
        raise MatrixError("registry has no generator_champion")
    v5 = Path(pointer.checkpoint_path).expanduser().resolve(strict=True)
    if fleet._sha256(v5) != active_campaign.EXPECTED_CORPUS_PRODUCER_SHA256:  # noqa: SLF001
        raise MatrixError("registry generator_champion is not the authoritative v5")

    completed: dict[str, dict[str, Any]] = {}
    fingerprints: dict[str, dict[str, Any]] = {}
    selection_rows = selection.get("arm_fingerprints")
    if not isinstance(selection_rows, dict):
        raise MatrixError("active-policy selection has no arm fingerprints")
    for arm in selection["eligible_arms"]:
        try:
            arm_completed = active_campaign._verify_completed_arm(campaign, arm)  # noqa: SLF001
            fingerprint_path, fingerprint = active_campaign._load_signed(  # noqa: SLF001
                Path(str(selection_rows[arm]["path"])),
                where=f"arm {arm} fingerprint",
                schema=active_campaign.FINGERPRINT_SCHEMA,
                digest_field="fingerprint_sha256",
            )
        except (active_campaign.CampaignError, KeyError) as error:
            raise MatrixError(f"eligible arm {arm} authority refused: {error}") from error
        checkpoints = fingerprint.get("checkpoints")
        terminal = checkpoints[-1] if isinstance(checkpoints, list) and checkpoints else None
        selection_row = selection_rows[arm]
        if (
            selection_row.get("within_drift_budgets") is not True
            or selection_row.get("positive_terminal_teacher_gap_closure") is not True
            or selection_row.get("file_sha256") != _file_sha256(fingerprint_path)
            or selection_row.get("fingerprint_sha256")
            != fingerprint.get("fingerprint_sha256")
            or fingerprint.get("arm") != arm
            or not isinstance(terminal, dict)
            or terminal.get("step") != active_campaign.MAX_STEPS
            or Path(str(terminal.get("checkpoint", ""))).resolve(strict=True)
            != Path(arm_completed["checkpoint"]).resolve(strict=True)
            or terminal.get("checkpoint_sha256")
            != arm_completed["checkpoint_sha256"]
        ):
            raise MatrixError(f"eligible arm {arm} terminal checkpoint drifted")
        completed[arm] = arm_completed
        fingerprints[arm] = {
            "path": str(fingerprint_path),
            "file_sha256": _file_sha256(fingerprint_path),
            "fingerprint_sha256": fingerprint["fingerprint_sha256"],
            "terminal_checkpoint": copy.deepcopy(terminal),
        }

    return {
        "campaign_path": resolved_campaign,
        "campaign": campaign,
        "selection_path": resolved_selection,
        "selection": selection,
        "upgrade": upgrade,
        "f7": f7,
        "initializer": initializer,
        "registry_path": registry_path,
        "registry": registry,
        "v5": v5,
        "completed": completed,
        "fingerprints": fingerprints,
    }


def _balanced_host_groups(
    manifest: Mapping[str, Any], *, group_count: int
) -> list[tuple[str, ...]]:
    hosts = manifest.get("hosts")
    if not isinstance(hosts, list) or not 1 <= group_count <= len(hosts):
        raise MatrixError("cannot partition the approved H100 hosts")
    groups: list[list[str]] = [[] for _ in range(group_count)]
    totals = [0] * group_count
    ordered = sorted(
        hosts,
        key=lambda host: (-int(host["gpu_count"]), str(host["alias"])),
    )
    for host in ordered:
        index = min(range(group_count), key=lambda item: (totals[item], item))
        groups[index].append(str(host["alias"]))
        totals[index] += int(host["gpu_count"])
    if any(not group for group in groups) or sum(totals) != 64:
        raise MatrixError("balanced host partition did not cover exactly 64 GPUs")
    if max(totals) - min(totals) > 4:
        raise MatrixError(f"balanced host partition is unexpectedly skewed: {totals}")
    return [tuple(group) for group in groups]


def _operation_command(
    *, matrix_path: Path, operation: str, output: Path | None = None
) -> list[str]:
    argv = [
        sys.executable,
        str(Path(__file__).resolve(strict=True)),
        "--matrix",
        str(matrix_path),
        operation,
    ]
    if operation in {"launch", "collect"}:
        if output is None:
            raise AssertionError(f"{operation} requires an output receipt")
        argv += ["--go", "--out", str(output)]
    elif operation == "wait":
        if output is None:
            raise AssertionError("wait requires an output receipt")
        argv += ["--out", str(output)]
    return argv


def build_matrix(
    *,
    manifest_path: Path,
    campaign_path: Path,
    selection_path: Path,
    registry_path: Path,
    internal_pairs: int,
    external_placeholder_pairs: int,
    internal_base_seed: int,
    external_base_seed: int,
    trial_id: str,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[Path, dict[str, Any]]]:
    manifest_path = manifest_path.expanduser().resolve(strict=True)
    output_dir = output_dir.expanduser().resolve(strict=False)
    manifest = fleet.load_manifest(
        manifest_path, expected_shapes=fleet.FULL_EXPECTED_SHAPES
    )
    if len(fleet.gpu_slots(manifest)) != 64:
        raise MatrixError("active-policy evaluation requires the exact 64-H100 fleet")
    authority = _load_authority(
        campaign_path=campaign_path,
        selection_path=selection_path,
        registry_path=registry_path,
    )
    eligible = list(authority["selection"]["eligible_arms"])
    matchups = [(arm, baseline) for arm in eligible for baseline in BASELINES]
    groups = _balanced_host_groups(manifest, group_count=len(matchups))
    host_gpus = {
        str(host["alias"]): int(host["gpu_count"]) for host in manifest["hosts"]
    }
    # ``build_plan`` reserves an external lane pair for every two selected
    # GPUs even though this matrix never launches the external phase.  When
    # fewer than four arms survive, each matchup receives more than eight
    # GPUs, so lift the harmless placeholder claim to the schema minimum.
    external_placeholder_pairs = max(
        int(external_placeholder_pairs),
        max(sum(host_gpus[alias] for alias in group) // 2 for group in groups),
    )

    search = current_science.search()
    operator_selection = current_science.load()["operator_selection"]
    if (
        operator_selection.get("status") != "adopted_teacher_campaign"
        or operator_selection.get("selected_operator") != "base_n128_d6"
    ):
        raise MatrixError("current science contract is not the adopted coherent n128 operator")
    c_scale = float(search["c_scale"])
    sigma_eval = float(search["sigma_eval"])
    cohort = f"{trial_id}-common"
    plans: dict[Path, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    occupied: set[str] = set()

    for (arm, baseline_name), aliases in zip(matchups, groups, strict=True):
        completed = authority["completed"][arm]
        candidate = Path(completed["checkpoint"])
        baseline = authority["f7"] if baseline_name == "f7" else authority["v5"]
        comparison_mode = (
            "historical_comparison" if baseline_name == "f7" else "branch_challenge"
        )
        reason = (
            "coherent_active_policy_candidate_vs_authenticated_f7_parent"
            if baseline_name == "f7"
            else None
        )
        plan = fleet.build_plan(
            manifest,
            candidate=candidate,
            champion=baseline,
            candidate_parent=authority["initializer"],
            registry=authority["registry"],
            internal_pairs=internal_pairs,
            external_pairs=external_placeholder_pairs,
            internal_base_seed=internal_base_seed,
            external_base_seed=external_base_seed,
            workers_per_gpu=fleet.DEFAULT_WORKERS_PER_GPU,
            iteration_id=f"{trial_id}-{arm.lower()}-{baseline_name}",
            seed_cohort_id=cohort,
            scope="full",
            host_aliases=aliases,
            candidate_c_scale=c_scale,
            champion_c_scale=c_scale,
            candidate_gameplay_policy_aggregation="mean_improved_policy",
            champion_gameplay_policy_aggregation="mean_improved_policy",
            candidate_sigma_eval=sigma_eval,
            champion_sigma_eval=sigma_eval,
            comparison_mode=comparison_mode,
            historical_comparison_reason=reason,
            operator_mode=fleet.COHERENT_PUBLIC_OPERATOR,
        )
        internal_jobs = [job for job in plan["jobs"] if job["phase"] == "internal"]
        slots = {str(job["slot_id"]) for job in internal_jobs}
        if not slots or occupied & slots:
            raise MatrixError("active-policy matrix assigned a GPU twice or not at all")
        occupied |= slots
        plan_path = output_dir / f"{arm.lower()}-vs-{baseline_name}.plan.json"
        plans[plan_path] = plan
        rows.append(
            {
                "arm": arm,
                "baseline": baseline_name,
                "comparison_mode": comparison_mode,
                "promotion_eligible": bool(
                    plan["evaluation_binding"]["promotion_eligible"]
                ),
                "host_aliases": list(aliases),
                "gpu_slots": sorted(slots),
                "plan": str(plan_path),
                "plan_hash": plan["plan_hash"],
                "candidate": {
                    "path": completed["checkpoint"],
                    "sha256": completed["checkpoint_sha256"],
                },
                "baseline_checkpoint": {
                    "path": str(baseline),
                    "sha256": fleet._sha256(baseline),  # noqa: SLF001
                },
                "paired_games": internal_pairs * 2,
                "collect_output_dir": str(
                    output_dir / "collected" / f"{arm.lower()}-vs-{baseline_name}"
                ),
            }
        )

    expected_slots = {str(slot["slot_id"]) for slot in fleet.gpu_slots(manifest)}
    if occupied != expected_slots:
        raise MatrixError(
            "active-policy matrix does not cover the fleet exactly: "
            f"missing={sorted(expected_slots - occupied)}"
        )
    science_hashes = {plan["science_config_hash"] for plan in plans.values()}
    if len(science_hashes) != 1:
        raise MatrixError("active-policy matchups do not share one search operator")

    matrix_path = output_dir / "matrix.json"
    matrix: dict[str, Any] = {
        "schema_version": SCHEMA,
        "trial_id": trial_id,
        "manifest": str(manifest_path),
        "manifest_hash": manifest["manifest_hash"],
        "campaign": {
            "path": str(authority["campaign_path"]),
            "file_sha256": _file_sha256(authority["campaign_path"]),
            "campaign_sha256": authority["campaign"]["campaign_sha256"],
        },
        "selection": {
            "path": str(authority["selection_path"]),
            "file_sha256": _file_sha256(authority["selection_path"]),
            "selection_sha256": authority["selection"]["selection_sha256"],
            "eligible_arms": eligible,
            "winner": authority["selection"]["winner"],
        },
        "arm_fingerprints": authority["fingerprints"],
        "registry": str(authority["registry_path"]),
        "function_preserving_upgrade": authority["upgrade"],
        "f7": {
            "path": str(authority["f7"]),
            "sha256": fleet._sha256(authority["f7"]),  # noqa: SLF001
        },
        "v5": {
            "path": str(authority["v5"]),
            "sha256": fleet._sha256(authority["v5"]),  # noqa: SLF001
        },
        "operator_selection": operator_selection,
        "operator_search": search,
        "science_config_hash": next(iter(science_hashes)),
        "seed_cohort_id": cohort,
        "common_random_numbers": True,
        "seat_swapped": True,
        "internal_claim": {
            "base_seed": internal_base_seed,
            "pairs": internal_pairs,
        },
        "external_placeholder_claim": {
            "base_seed": external_base_seed,
            "pairs": external_placeholder_pairs,
            "launch": False,
        },
        "workers_per_gpu": fleet.DEFAULT_WORKERS_PER_GPU,
        "physical_gpus": len(occupied),
        "matchups": rows,
    }
    matrix["commands"] = {
        "launch": _operation_command(
            matrix_path=matrix_path,
            operation="launch",
            output=output_dir / "launch.receipt.json",
        ),
        "status": _operation_command(matrix_path=matrix_path, operation="status"),
        "wait": _operation_command(
            matrix_path=matrix_path,
            operation="wait",
            output=output_dir / "wait.receipt.json",
        ),
        "collect": _operation_command(
            matrix_path=matrix_path,
            operation="collect",
            output=output_dir / "collect.receipt.json",
        ),
    }
    matrix["state_sha256"] = _digest(matrix)
    return matrix, plans


def load_matrix(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[tuple[dict[str, Any], dict[str, Any]]]]:
    matrix_path, matrix = _read_object(path, where="active-policy evaluation matrix")
    unsigned = dict(matrix)
    stated = unsigned.pop("state_sha256", None)
    if matrix.get("schema_version") != SCHEMA or stated != _digest(unsigned):
        raise MatrixError("active-policy evaluation matrix schema or digest drift")
    authority = _load_authority(
        campaign_path=Path(matrix["campaign"]["path"]),
        selection_path=Path(matrix["selection"]["path"]),
        registry_path=Path(matrix["registry"]),
    )
    if (
        matrix["campaign"]["file_sha256"] != _file_sha256(authority["campaign_path"])
        or matrix["campaign"]["campaign_sha256"]
        != authority["campaign"]["campaign_sha256"]
        or matrix["selection"]["file_sha256"]
        != _file_sha256(authority["selection_path"])
        or matrix["selection"]["selection_sha256"]
        != authority["selection"]["selection_sha256"]
        or matrix["selection"]["eligible_arms"]
        != authority["selection"]["eligible_arms"]
        or matrix["function_preserving_upgrade"] != authority["upgrade"]
    ):
        raise MatrixError("active-policy evaluation authority changed after planning")
    manifest = fleet.load_manifest(
        Path(matrix["manifest"]), expected_shapes=fleet.FULL_EXPECTED_SHAPES
    )
    if manifest["manifest_hash"] != matrix.get("manifest_hash"):
        raise MatrixError("active-policy evaluation fleet manifest drifted")

    rows = matrix.get("matchups")
    expected_matchups = {
        (arm, baseline)
        for arm in authority["selection"]["eligible_arms"]
        for baseline in BASELINES
    }
    if (
        not isinstance(rows, list)
        or {(row.get("arm"), row.get("baseline")) for row in rows}
        != expected_matchups
    ):
        raise MatrixError("active-policy matchup set drifted")
    loaded: list[tuple[dict[str, Any], dict[str, Any]]] = []
    occupied: set[str] = set()
    for row in rows:
        plan = fleet.load_plan(Path(row["plan"]), manifest)
        arm = str(row["arm"])
        completed = authority["completed"][arm]
        expected_baseline = authority["f7"] if row["baseline"] == "f7" else authority["v5"]
        slots = {
            str(job["slot_id"])
            for job in plan["jobs"]
            if job["phase"] == "internal"
        }
        if (
            plan["plan_hash"] != row.get("plan_hash")
            or plan["operator_mode"] != fleet.COHERENT_PUBLIC_OPERATOR
            or plan["seed_cohort_id"] != matrix["seed_cohort_id"]
            or plan["pair_claims"]["internal"] != matrix["internal_claim"]
            or plan["candidate"]["source"] != completed["checkpoint"]
            or plan["candidate"]["sha256"] != completed["checkpoint_sha256"]
            or plan["champion"]["source"] != str(expected_baseline)
            or plan["champion"]["sha256"] != fleet._sha256(expected_baseline)  # noqa: SLF001
            or slots != set(row["gpu_slots"])
            or occupied & slots
        ):
            raise MatrixError(f"active-policy plan drifted for {arm}-{row['baseline']}")
        occupied |= slots
        loaded.append((row, plan))
    expected_slots = {str(slot["slot_id"]) for slot in fleet.gpu_slots(manifest)}
    if occupied != expected_slots or matrix.get("physical_gpus") != 64:
        raise MatrixError("active-policy plans no longer cover exactly 64 GPUs")
    return manifest, matrix, loaded


def _parallel(
    loaded: list[tuple[dict[str, Any], dict[str, Any]]],
    operation: Callable[[dict[str, Any], dict[str, Any]], Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    results: list[dict[str, Any]] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=len(loaded)) as pool:
        futures = {
            pool.submit(operation, row, plan): (row, plan) for row, plan in loaded
        }
        for future in as_completed(futures):
            row, plan = futures[future]
            key = f"{row['arm']}-vs-{row['baseline']}"
            try:
                payload = future.result()
            except Exception as error:  # preserve all other matchup evidence
                failures.append(f"{key}: {type(error).__name__}: {error}")
            else:
                results.append(
                    {"matchup": key, "plan_hash": plan["plan_hash"], "result": payload}
                )
    return sorted(results, key=lambda row: row["matchup"]), sorted(failures)


def execute(path: Path, *, operation: str) -> dict[str, Any]:
    manifest, matrix, loaded = load_matrix(path)
    if operation == "launch":
        rows, failures = _parallel(
            loaded, lambda _row, plan: fleet.launch_phase(manifest, plan, "internal")
        )
    elif operation == "status":
        rows, failures = _parallel(
            loaded, lambda _row, plan: fleet.status_phase(manifest, plan, "internal")
        )
    elif operation == "collect":
        rows, failures = _parallel(
            loaded,
            lambda row, plan: fleet.collect_phase(
                manifest,
                plan,
                "internal",
                Path(str(row["collect_output_dir"])),
            ),
        )
    else:
        raise AssertionError(operation)
    payload = {
        "schema_version": OPERATION_SCHEMA,
        "operation": operation,
        "matrix": str(path.expanduser().resolve(strict=True)),
        "matrix_state_sha256": matrix["state_sha256"],
        "matchups": rows,
        "failures": failures,
        "ok": not failures,
    }
    payload["state_sha256"] = _digest(payload)
    return payload


def wait_for_completion(
    path: Path, *, poll_seconds: float, timeout_seconds: float
) -> dict[str, Any]:
    if poll_seconds <= 0.0 or timeout_seconds <= 0.0:
        raise MatrixError("poll and timeout seconds must be positive")
    started = time.monotonic()
    polls = 0
    while True:
        status = execute(path, operation="status")
        polls += 1
        if status["failures"]:
            raise MatrixError("matrix status failed: " + " | ".join(status["failures"]))
        counts = {
            state: sum(
                int(row["result"]["counts"].get(state, 0))
                for row in status["matchups"]
            )
            for state in ("done", "active", "failed", "stale", "missing", "unsafe")
        }
        bad = {
            state: counts[state]
            for state in ("failed", "stale", "missing", "unsafe")
            if counts[state]
        }
        if bad:
            raise MatrixError(f"matrix entered a non-runnable state: {bad}")
        if counts["done"] == 64 and counts["active"] == 0:
            result = {
                "schema_version": COMPLETION_SCHEMA,
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
            raise MatrixError(f"matrix status does not cover 64 jobs: {counts}")
        elapsed = time.monotonic() - started
        if elapsed >= timeout_seconds:
            raise MatrixError(
                f"matrix did not finish within {timeout_seconds:g}s: {counts}"
            )
        time.sleep(min(poll_seconds, max(0.0, timeout_seconds - elapsed)))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--manifest", type=Path, required=True)
    plan.add_argument("--campaign", type=Path, required=True)
    plan.add_argument("--selection", type=Path, required=True)
    plan.add_argument("--registry", type=Path, required=True)
    plan.add_argument("--internal-pairs", type=int, default=128)
    plan.add_argument("--external-placeholder-pairs", type=int, default=4)
    plan.add_argument("--internal-base-seed", type=int, required=True)
    plan.add_argument("--external-base-seed", type=int, required=True)
    plan.add_argument("--trial-id", required=True)
    plan.add_argument("--output-dir", type=Path, required=True)
    for name in ("launch", "collect"):
        operation = commands.add_parser(name)
        operation.add_argument("--go", action="store_true", required=True)
        operation.add_argument("--out", type=Path, required=True)
    wait = commands.add_parser("wait")
    wait.add_argument("--poll-seconds", type=float, default=30.0)
    wait.add_argument("--timeout-seconds", type=float, default=7200.0)
    wait.add_argument("--out", type=Path, required=True)
    commands.add_parser("status")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "plan":
            matrix, plans = build_matrix(
                manifest_path=args.manifest,
                campaign_path=args.campaign,
                selection_path=args.selection,
                registry_path=args.registry,
                internal_pairs=args.internal_pairs,
                external_placeholder_pairs=args.external_placeholder_pairs,
                internal_base_seed=args.internal_base_seed,
                external_base_seed=args.external_base_seed,
                trial_id=args.trial_id,
                output_dir=args.output_dir,
            )
            matrix_path = args.output_dir.expanduser().resolve(strict=False) / "matrix.json"
            targets = [*plans, matrix_path]
            existing = [str(path) for path in targets if path.exists()]
            if existing:
                raise MatrixError(f"refusing to overwrite matrix artifacts: {existing}")
            for path, payload in plans.items():
                fleet.write_new_readonly(path, payload)
            fleet.write_new_readonly(matrix_path, matrix)
            result = {
                "matrix": str(matrix_path),
                "state_sha256": matrix["state_sha256"],
                "eligible_arms": matrix["selection"]["eligible_arms"],
                "matchups": len(matrix["matchups"]),
                "physical_gpus": matrix["physical_gpus"],
                "commands": matrix["commands"],
            }
        else:
            if args.matrix is None:
                raise MatrixError("--matrix is required for matrix operations")
            if args.command in {"launch", "wait", "collect"} and args.out.exists():
                raise MatrixError(f"refusing to overwrite operation receipt {args.out}")
            result = (
                wait_for_completion(
                    args.matrix,
                    poll_seconds=args.poll_seconds,
                    timeout_seconds=args.timeout_seconds,
                )
                if args.command == "wait"
                else execute(args.matrix, operation=args.command)
            )
            if args.command in {"launch", "wait", "collect"}:
                fleet.write_new_readonly(args.out, result)
                result = {
                    "receipt": str(args.out.expanduser().resolve(strict=True)),
                    "state_sha256": result["state_sha256"],
                    "ok": result.get("ok", True),
                }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("ok", True) else 2
    except (
        MatrixError,
        active_campaign.CampaignError,
        fleet.FleetError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        print(f"active-policy evaluation matrix refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
