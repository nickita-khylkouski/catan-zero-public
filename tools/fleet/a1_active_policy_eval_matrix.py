#!/usr/bin/env python3
"""Seal and execute the 64-H100 active-policy candidate evaluation matrix.

The coherent-n128 learner campaign names P10/P25/P50/P100 rather than the
historical A/B/C/D LR arms.  This controller authenticates every in-budget
epoch checkpoint, maps an explicit fleet-sized batch against exact f7 and the
registry v5 incumbent, and partitions the full 64-H100 fleet across those
matchups.  Multiple matrices cover a frontier larger than six candidates
without silently selecting an epoch before gameplay.  Every matchup reuses one
common paired-seed cohort and swaps colors inside each pair.

Only the internal Rust-native coherent-public panel is authorized here.  The
external pair claim remains an unlaunched placeholder so the ordinary fleet
plan schema and validation-ledger rules stay intact.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
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


SCHEMA = "a1-active-policy-eval-matrix-v3"
OPERATION_SCHEMA = "a1-active-policy-eval-matrix-operation-v1"
COMPLETION_SCHEMA = "a1-active-policy-eval-matrix-completion-v1"
BASELINES = ("f7", "v5")
MAX_CANDIDATES_PER_MATRIX = 6


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
    rows = selection.get("arm_fingerprints")
    eligible_candidates = selection.get("eligible_candidates")
    winner = selection.get("winner")
    winner_candidate = selection.get("winner_candidate")
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
        or winner not in eligible
        or not isinstance(rows, dict)
        or set(rows) != set(active_campaign.ARMS)
        or any(
            not isinstance(rows.get(arm), dict)
            or rows[arm].get("has_eligible_checkpoint") is not True
            or not isinstance(rows[arm].get("selected_checkpoint"), dict)
            for arm in eligible
        )
        or not isinstance(winner_candidate, dict)
        or winner_candidate != rows[winner].get("selected_checkpoint")
        or selection.get("winner_step") != winner_candidate.get("step")
        or selection.get("winner_checkpoint")
        != {
            "path": winner_candidate.get("checkpoint"),
            "sha256": winner_candidate.get("checkpoint_sha256"),
        }
        or not isinstance(eligible_candidates, list)
        or not eligible_candidates
        or winner_candidate not in eligible_candidates
    ):
        raise MatrixError("active-policy selection lost campaign/eligibility semantics")
    canonical_order = [arm for arm in active_campaign.ARMS if arm in set(eligible)]
    if eligible != canonical_order:
        raise MatrixError("active-policy eligible arms are not in canonical dose order")
    candidate_keys: set[tuple[str, int]] = set()
    for candidate in eligible_candidates:
        if not isinstance(candidate, dict):
            raise MatrixError("active-policy eligible candidate is malformed")
        try:
            key = (str(candidate["arm"]), int(candidate["step"]))
        except (KeyError, TypeError, ValueError) as error:
            raise MatrixError("active-policy eligible candidate identity is malformed") from error
        if (
            key in candidate_keys
            or key[0] not in eligible
            or candidate.get("eligible") is not True
            or candidate.get("within_drift_budgets") is not True
            or candidate
            not in rows[key[0]].get("checkpoint_candidates", [])
        ):
            raise MatrixError("active-policy eligible candidate frontier drifted")
        candidate_keys.add(key)
    return resolved, selection


def _candidate_id(arm: str, step: int) -> str:
    return f"{arm.lower()}-step{step:04d}"


def _resolve_candidate_ids(
    candidates: Mapping[str, Any], requested: Sequence[str] | None
) -> list[str]:
    available = list(candidates)
    if requested:
        selected = [str(value) for value in requested]
        if len(selected) != len(set(selected)):
            raise MatrixError("candidate ids must be unique")
        unknown = sorted(set(selected) - set(available))
        if unknown:
            raise MatrixError(f"unknown active-policy candidate ids: {unknown}")
    else:
        selected = available
    if len(selected) > MAX_CANDIDATES_PER_MATRIX:
        raise MatrixError(
            "active-policy frontier requires multiple fleet matrices; select at most "
            f"{MAX_CANDIDATES_PER_MATRIX} candidate ids from {available}"
        )
    return selected


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
    candidates: dict[str, dict[str, Any]] = {}
    fingerprints: dict[str, dict[str, Any]] = {}
    selection_rows = selection.get("arm_fingerprints")
    if not isinstance(selection_rows, dict):
        raise MatrixError("active-policy selection has no arm fingerprints")
    eligible_by_arm: dict[str, list[dict[str, Any]]] = {
        arm: [] for arm in selection["eligible_arms"]
    }
    for candidate in selection["eligible_candidates"]:
        eligible_by_arm[str(candidate["arm"])].append(candidate)
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
        selection_row = selection_rows[arm]
        selected = selection_row.get("selected_checkpoint")
        if (
            selection_row.get("has_eligible_checkpoint") is not True
            or selection_row.get("file_sha256") != _file_sha256(fingerprint_path)
            or selection_row.get("fingerprint_sha256")
            != fingerprint.get("fingerprint_sha256")
            or fingerprint.get("arm") != arm
            or not isinstance(selected, dict)
            or selected.get("eligible") is not True
            or selected.get("within_drift_budgets") is not True
        ):
            raise MatrixError(f"eligible arm {arm} selected checkpoint drifted")
        completed[arm] = arm_completed
        authenticated_steps: list[int] = []
        for candidate in eligible_by_arm[arm]:
            step = int(candidate["step"])
            candidate_fingerprint = next(
                (
                    checkpoint
                    for checkpoint in checkpoints
                    if checkpoint.get("step") == step
                ),
                None,
            )
            if (
                not isinstance(candidate_fingerprint, dict)
                or Path(str(candidate.get("checkpoint", ""))).resolve(strict=True)
                != Path(str(candidate_fingerprint.get("checkpoint", ""))).resolve(
                    strict=True
                )
                or candidate.get("checkpoint_sha256")
                != candidate_fingerprint.get("checkpoint_sha256")
                or candidate.get("checkpoint_sha256")
                != _file_sha256(
                    Path(str(candidate["checkpoint"])).resolve(strict=True)
                )
                or float(candidate.get("parent_kl", -1.0))
                != float(
                    candidate_fingerprint.get("functional", {}).get(
                        "parent_kl", -2.0
                    )
                )
                or float(candidate.get("teacher_gap_closure", -1.0))
                != float(
                    candidate_fingerprint.get("functional", {}).get(
                        "teacher_gap_closure", -2.0
                    )
                )
                or float(candidate.get("trunk_relative_l2", -1.0))
                != float(
                    candidate_fingerprint.get("layer_drift", {}).get(
                        "trunk_relative_l2", -2.0
                    )
                )
            ):
                raise MatrixError(
                    f"eligible arm {arm} step {step} checkpoint drifted"
                )
            candidate_id = _candidate_id(arm, step)
            candidates[candidate_id] = {
                "candidate_id": candidate_id,
                "arm": arm,
                "step": step,
                "checkpoint": str(
                    Path(str(candidate["checkpoint"])).resolve(strict=True)
                ),
                "checkpoint_sha256": str(candidate["checkpoint_sha256"]),
                "parent_kl": float(candidate["parent_kl"]),
                "trunk_relative_l2": float(candidate["trunk_relative_l2"]),
                "teacher_gap_closure": float(candidate["teacher_gap_closure"]),
            }
            authenticated_steps.append(step)
        fingerprints[arm] = {
            "path": str(fingerprint_path),
            "file_sha256": _file_sha256(fingerprint_path),
            "fingerprint_sha256": fingerprint["fingerprint_sha256"],
            "eligible_checkpoint_steps": authenticated_steps,
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
        "candidates": candidates,
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
    candidate_ids: Sequence[str] | None = None,
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
    selected_candidate_ids = _resolve_candidate_ids(
        authority["candidates"], candidate_ids
    )
    matchups = [
        (candidate_id, baseline)
        for candidate_id in selected_candidate_ids
        for baseline in BASELINES
    ]
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

    for (candidate_id, baseline_name), aliases in zip(matchups, groups, strict=True):
        candidate_authority = authority["candidates"][candidate_id]
        arm = candidate_authority["arm"]
        candidate = Path(candidate_authority["checkpoint"])
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
            iteration_id=f"{trial_id}-{candidate_id}-{baseline_name}",
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
        plan_path = output_dir / f"{candidate_id}-vs-{baseline_name}.plan.json"
        plans[plan_path] = plan
        rows.append(
            {
                "candidate_id": candidate_id,
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
                    "path": candidate_authority["checkpoint"],
                    "sha256": candidate_authority["checkpoint_sha256"],
                    "optimizer_step": candidate_authority["step"],
                },
                "baseline_checkpoint": {
                    "path": str(baseline),
                    "sha256": fleet._sha256(baseline),  # noqa: SLF001
                },
                "paired_games": internal_pairs * 2,
                "collect_output_dir": str(
                    output_dir / "collected" / f"{candidate_id}-vs-{baseline_name}"
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
            "candidate_ids": selected_candidate_ids,
            "available_candidate_ids": list(authority["candidates"]),
            "candidate_steps": {
                candidate_id: authority["candidates"][candidate_id]["step"]
                for candidate_id in selected_candidate_ids
            },
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
    candidate_ids = matrix["selection"].get("candidate_ids")
    if (
        not isinstance(candidate_ids, list)
        or candidate_ids != _resolve_candidate_ids(authority["candidates"], candidate_ids)
    ):
        raise MatrixError("active-policy candidate batch drifted")
    expected_matchups = {
        (candidate_id, baseline)
        for candidate_id in candidate_ids
        for baseline in BASELINES
    }
    if (
        not isinstance(rows, list)
        or {(row.get("candidate_id"), row.get("baseline")) for row in rows}
        != expected_matchups
    ):
        raise MatrixError("active-policy matchup set drifted")
    loaded: list[tuple[dict[str, Any], dict[str, Any]]] = []
    occupied: set[str] = set()
    for row in rows:
        plan = fleet.load_plan(Path(row["plan"]), manifest)
        candidate_id = str(row["candidate_id"])
        candidate = authority["candidates"][candidate_id]
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
            or plan["candidate"]["source"] != candidate["checkpoint"]
            or plan["candidate"]["sha256"] != candidate["checkpoint_sha256"]
            or row.get("arm") != candidate["arm"]
            or row.get("candidate", {}).get("optimizer_step") != candidate["step"]
            or plan["champion"]["source"] != str(expected_baseline)
            or plan["champion"]["sha256"] != fleet._sha256(expected_baseline)  # noqa: SLF001
            or slots != set(row["gpu_slots"])
            or occupied & slots
        ):
            raise MatrixError(
                f"active-policy plan drifted for {candidate_id}-{row['baseline']}"
            )
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
            key = f"{row['candidate_id']}-vs-{row['baseline']}"
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
    plan.add_argument(
        "--candidate-id",
        action="append",
        dest="candidate_ids",
        help=(
            "authenticated arm-step checkpoint id to evaluate; repeat up to six "
            "times, and use multiple matrices to cover the full eligible frontier"
        ),
    )
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
                candidate_ids=args.candidate_ids,
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
                "candidate_ids": matrix["selection"]["candidate_ids"],
                "available_candidate_ids": matrix["selection"][
                    "available_candidate_ids"
                ],
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
