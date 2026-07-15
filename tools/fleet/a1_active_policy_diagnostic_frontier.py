#!/usr/bin/env python3
"""Seal the non-promotable 64-H100 frontier after Stage A selects no winner.

This planner never relaxes the Stage-A parent-KL or trunk-drift budgets.  It
first asks the canonical Stage-A selector to authenticate all four fingerprint
artifacts and requires that selector to return its exact no-eligible-candidate
refusal.  Only then does it nominate three diagnostic frontier points:

* the checkpoint with maximum teacher-gap closure overall; and
* the positive-closure checkpoint with minimum parent KL.
* the P100 step-64 checkpoint between those two endpoints.

The resulting checkpoint(s) are diagnostic evidence only.  Every comparison,
including the v5 comparison, uses ``historical_comparison`` so no downstream
consumer can interpret this matrix as a promotion gate.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
for import_root in (REPO_ROOT / "src", REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from tools import a1_b200_active_policy_campaign as active_campaign  # noqa: E402
from tools import a1_current_science_contract as current_science  # noqa: E402
from tools import a1_function_preserving_upgrade as architecture_upgrade  # noqa: E402
from tools.champion_registry import ChampionRegistry  # noqa: E402
from tools.fleet import a1_active_policy_eval_matrix as active_matrix  # noqa: E402
from tools.fleet import a1_h100_eval_fleet as fleet  # noqa: E402


SCHEMA = "a1-active-policy-diagnostic-frontier-matrix-v1"
NO_WINNER_REASON = (
    "no active-policy exposure checkpoint remained inside both drift budgets"
)
CRITERIA = (
    "max_teacher_gap_closure",
    "min_parent_kl_positive_closure",
    "p100_step64_interpolation",
)
BASELINES = ("f7", "v5")


class FrontierError(RuntimeError):
    """The diagnostic frontier is not exactly authenticated or non-promotable."""


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


def _parse_bindings(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for raw in values:
        arm, separator, path = raw.partition("=")
        if (
            not separator
            or arm not in active_campaign.ARMS
            or not path
            or arm in result
        ):
            raise FrontierError(
                "fingerprints must be unique P10=PATH through P100=PATH"
            )
        result[arm] = Path(path).expanduser().resolve(strict=True)
    if set(result) != set(active_campaign.ARMS):
        raise FrontierError(
            f"fingerprints must name exactly {list(active_campaign.ARMS)}"
        )
    return result


def _require_canonical_no_winner(
    campaign_path: Path,
    campaign: Mapping[str, Any],
    bindings: Mapping[str, Path],
) -> None:
    try:
        active_campaign._select(campaign_path, campaign, bindings)  # noqa: SLF001
    except active_campaign.CampaignError as error:
        if str(error) != NO_WINNER_REASON:
            raise FrontierError(
                f"canonical Stage-A selector failed for another reason: {error}"
            ) from error
    else:
        raise FrontierError(
            "canonical Stage-A selector produced an eligible winner; "
            "diagnostic frontier is forbidden"
        )


def _load_frontier_candidates(
    campaign_path: Path,
    campaign: Mapping[str, Any],
    bindings: Mapping[str, Path],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    _require_canonical_no_winner(campaign_path, campaign, bindings)
    candidates: list[dict[str, Any]] = []
    fingerprint_refs: dict[str, dict[str, Any]] = {}
    arm_order = {arm: index for index, arm in enumerate(active_campaign.ARMS)}

    for arm in active_campaign.ARMS:
        completed = active_campaign._verify_completed_arm(campaign, arm)  # noqa: SLF001
        path, fingerprint = active_campaign._load_signed(  # noqa: SLF001
            bindings[arm],
            where=f"arm {arm} fingerprint",
            schema=active_campaign.FINGERPRINT_SCHEMA,
            digest_field="fingerprint_sha256",
        )
        campaign_ref = fingerprint.get("campaign")
        checkpoints = fingerprint.get("checkpoints")
        if (
            not isinstance(campaign_ref, dict)
            or campaign_ref.get("campaign_sha256") != campaign["campaign_sha256"]
            or campaign_ref.get("file_sha256") != _file_sha256(campaign_path)
            or fingerprint.get("arm") != arm
            or not isinstance(checkpoints, list)
            or [row.get("step") for row in checkpoints]
            != list(active_campaign.CHECKPOINT_STEPS)
        ):
            raise FrontierError(f"arm {arm} fingerprint campaign binding drifted")
        receipt_path = Path(str(completed["receipt"])).resolve(strict=True)
        report_path = Path(str(completed["report"])).resolve(strict=True)
        fingerprint_refs[arm] = {
            "path": str(path),
            "file_sha256": _file_sha256(path),
            "fingerprint_sha256": fingerprint["fingerprint_sha256"],
            "completed_arm": {
                "receipt": str(receipt_path),
                "receipt_file_sha256": _file_sha256(receipt_path),
                "report": str(report_path),
                "report_file_sha256": _file_sha256(report_path),
                "final_checkpoint": completed["checkpoint"],
                "final_checkpoint_sha256": completed["checkpoint_sha256"],
                "dose_telemetry_sha256": completed["dose_telemetry"][
                    "dose_telemetry_sha256"
                ],
            },
        }
        for row in checkpoints:
            try:
                step = int(row["step"])
                checkpoint = Path(str(row["checkpoint"])).resolve(strict=True)
                checkpoint_sha256 = str(row["checkpoint_sha256"])
                parent_kl = float(row["functional"]["parent_kl"])
                closure = float(row["functional"]["teacher_gap_closure"])
                trunk = float(row["layer_drift"]["trunk_relative_l2"])
            except (KeyError, TypeError, ValueError) as error:
                raise FrontierError(
                    f"arm {arm} checkpoint fingerprint is malformed"
                ) from error
            if (
                checkpoint_sha256 != _file_sha256(checkpoint)
                or not all(math.isfinite(value) for value in (parent_kl, closure, trunk))
                or parent_kl < 0.0
                or trunk < 0.0
            ):
                raise FrontierError(f"arm {arm} step {step} fingerprint drifted")
            candidates.append(
                {
                    "arm": arm,
                    "arm_order": arm_order[arm],
                    "step": step,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": checkpoint_sha256,
                    "parent_kl": parent_kl,
                    "trunk_relative_l2": trunk,
                    "teacher_gap_closure": closure,
                }
            )

    max_closure = min(
        candidates,
        key=lambda row: (
            -row["teacher_gap_closure"],
            row["parent_kl"],
            row["trunk_relative_l2"],
            row["step"],
            row["arm_order"],
        ),
    )
    positive = [row for row in candidates if row["teacher_gap_closure"] > 0.0]
    if not positive:
        raise FrontierError(
            "diagnostic frontier has no positive-closure checkpoint; "
            "min-parent-KL positive frontier is undefined"
        )
    min_positive_kl = min(
        positive,
        key=lambda row: (
            row["parent_kl"],
            -row["teacher_gap_closure"],
            row["trunk_relative_l2"],
            row["step"],
            row["arm_order"],
        ),
    )
    interpolation = next(
        (
            row
            for row in candidates
            if row["arm"] == "P100" and row["step"] == 64
        ),
        None,
    )
    if interpolation is None:
        raise FrontierError("P100 step-64 interpolation checkpoint is missing")

    by_sha: dict[str, dict[str, Any]] = {}
    for criterion, selected in zip(
        CRITERIA, (max_closure, min_positive_kl, interpolation), strict=True
    ):
        digest = str(selected["checkpoint_sha256"])
        if digest not in by_sha:
            by_sha[digest] = {
                key: copy.deepcopy(value)
                for key, value in selected.items()
                if key != "arm_order"
            }
            by_sha[digest]["criteria"] = []
        by_sha[digest]["criteria"].append(criterion)
    selected_rows = sorted(
        by_sha.values(),
        key=lambda row: (
            CRITERIA.index(row["criteria"][0]),
            list(active_campaign.ARMS).index(row["arm"]),
            row["step"],
        ),
    )
    for index, row in enumerate(selected_rows):
        row["frontier_id"] = f"frontier-{index + 1}"
    return selected_rows, fingerprint_refs


def _baselines(
    campaign: Mapping[str, Any], registry_path: Path
) -> dict[str, Any]:
    upgrade_path = Path(campaign["inputs"]["architecture_upgrade_receipt"])
    try:
        upgrade = architecture_upgrade.verify_receipt(upgrade_path)
    except architecture_upgrade.UpgradeError as error:
        raise FrontierError(f"function-preserving f7 upgrade refused: {error}") from error
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
    ):
        raise FrontierError("campaign no longer maps exact f7 to its initializer")

    registry_path = registry_path.expanduser().resolve(strict=True)
    registry = ChampionRegistry.load(registry_path)
    pointer = registry.get_role("generator_champion")
    if pointer is None:
        raise FrontierError("registry has no generator_champion")
    v5 = Path(pointer.checkpoint_path).expanduser().resolve(strict=True)
    if fleet._sha256(v5) != active_campaign.EXPECTED_CORPUS_PRODUCER_SHA256:  # noqa: SLF001
        raise FrontierError("registry generator_champion is not exact v5")
    return {
        "upgrade": upgrade,
        "initializer": initializer,
        "f7": f7,
        "v5": v5,
        "registry": registry,
        "registry_path": registry_path,
    }


def _command(
    *, manifest: Path, operation: str, plan: Path, output_dir: Path | None = None
) -> list[str]:
    argv = [
        sys.executable,
        str(Path(fleet.__file__).resolve(strict=True)),
        "--manifest",
        str(manifest),
        operation,
        "--plan",
        str(plan),
        "--phase",
        "internal",
    ]
    if operation == "launch":
        argv += ["--go"]
    elif operation == "collect":
        if output_dir is None:
            raise AssertionError("collect command requires output directory")
        argv += ["--output-dir", str(output_dir)]
    return argv


def build_matrix(
    *,
    manifest_path: Path,
    campaign_path: Path,
    fingerprint_bindings: Mapping[str, Path],
    registry_path: Path,
    internal_pairs: int,
    external_placeholder_pairs: int,
    internal_base_seed: int,
    external_base_seed: int,
    trial_id: str,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[Path, dict[str, Any]]]:
    try:
        campaign_path, campaign = active_campaign._load_campaign(campaign_path)  # noqa: SLF001
    except active_campaign.CampaignError as error:
        raise FrontierError(str(error)) from error
    frontier, fingerprints = _load_frontier_candidates(
        campaign_path, campaign, fingerprint_bindings
    )
    authority = _baselines(campaign, registry_path)

    manifest_path = manifest_path.expanduser().resolve(strict=True)
    output_dir = output_dir.expanduser().resolve(strict=False)
    manifest = fleet.load_manifest(
        manifest_path, expected_shapes=fleet.FULL_EXPECTED_SHAPES
    )
    if len(fleet.gpu_slots(manifest)) != 64:
        raise FrontierError("diagnostic frontier requires exact 64-H100 fleet")

    science = current_science.load()
    operator = science["operator_selection"]
    if (
        operator.get("status") != "adopted_teacher_campaign"
        or operator.get("selected_operator") != "base_n128_d6"
    ):
        raise FrontierError("diagnostic frontier requires adopted base_n128_d6")
    search = current_science.search()
    c_scale = float(search["c_scale"])
    sigma_eval = float(search["sigma_eval"])

    matchups = [(row, baseline) for row in frontier for baseline in BASELINES]
    groups = active_matrix._balanced_host_groups(  # noqa: SLF001
        manifest, group_count=len(matchups)
    )
    host_gpus = {
        str(host["alias"]): int(host["gpu_count"]) for host in manifest["hosts"]
    }
    external_placeholder_pairs = max(
        int(external_placeholder_pairs),
        max(sum(host_gpus[alias] for alias in group) // 2 for group in groups),
    )
    cohort = f"{trial_id}-common"
    plans: dict[Path, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    occupied: set[str] = set()

    for (candidate, baseline_name), aliases in zip(matchups, groups, strict=True):
        baseline = authority[baseline_name]
        reason = (
            "active_policy_no_selection_diagnostic_frontier_vs_"
            f"authenticated_{baseline_name}"
        )
        plan = fleet.build_plan(
            manifest,
            candidate=Path(candidate["checkpoint"]),
            champion=baseline,
            candidate_parent=authority["initializer"],
            registry=authority["registry"],
            internal_pairs=internal_pairs,
            external_pairs=external_placeholder_pairs,
            internal_base_seed=internal_base_seed,
            external_base_seed=external_base_seed,
            workers_per_gpu=fleet.DEFAULT_WORKERS_PER_GPU,
            iteration_id=(
                f"{trial_id}-{candidate['frontier_id']}-{baseline_name}"
            ),
            seed_cohort_id=cohort,
            scope="full",
            host_aliases=aliases,
            candidate_c_scale=c_scale,
            champion_c_scale=c_scale,
            candidate_gameplay_policy_aggregation="mean_improved_policy",
            champion_gameplay_policy_aggregation="mean_improved_policy",
            candidate_sigma_eval=sigma_eval,
            champion_sigma_eval=sigma_eval,
            comparison_mode="historical_comparison",
            historical_comparison_reason=reason,
            operator_mode=fleet.COHERENT_PUBLIC_OPERATOR,
        )
        if plan["evaluation_binding"]["promotion_eligible"] is not False:
            raise FrontierError("diagnostic plan became promotion eligible")
        internal_jobs = [job for job in plan["jobs"] if job["phase"] == "internal"]
        slots = {str(job["slot_id"]) for job in internal_jobs}
        if not slots or occupied & slots:
            raise FrontierError("diagnostic frontier assigned a GPU twice")
        occupied |= slots
        key = f"{candidate['frontier_id']}-vs-{baseline_name}"
        plan_path = output_dir / f"{key}.plan.json"
        collect_dir = output_dir / "collected" / key
        plans[plan_path] = plan
        rows.append(
            {
                "frontier_id": candidate["frontier_id"],
                "criteria": copy.deepcopy(candidate["criteria"]),
                "arm": candidate["arm"],
                "step": candidate["step"],
                "baseline": baseline_name,
                "comparison_mode": "historical_comparison",
                "diagnostic_only": True,
                "promotion_eligible": False,
                "host_aliases": list(aliases),
                "gpu_slots": sorted(slots),
                "plan": str(plan_path),
                "plan_hash": plan["plan_hash"],
                "candidate": {
                    "path": candidate["checkpoint"],
                    "sha256": candidate["checkpoint_sha256"],
                },
                "baseline_checkpoint": {
                    "path": str(baseline),
                    "sha256": fleet._sha256(baseline),  # noqa: SLF001
                },
                "paired_games": internal_pairs * 2,
                "collect_output_dir": str(collect_dir),
                "launch_argv": _command(
                    manifest=manifest_path, operation="launch", plan=plan_path
                ),
                "status_argv": _command(
                    manifest=manifest_path, operation="status", plan=plan_path
                ),
                "collect_argv": _command(
                    manifest=manifest_path,
                    operation="collect",
                    plan=plan_path,
                    output_dir=collect_dir,
                ),
            }
        )

    expected = {str(slot["slot_id"]) for slot in fleet.gpu_slots(manifest)}
    if occupied != expected:
        raise FrontierError(
            "diagnostic frontier does not cover exactly 64 GPUs: "
            f"missing={sorted(expected - occupied)}"
        )
    science_hashes = {plan["science_config_hash"] for plan in plans.values()}
    if len(science_hashes) != 1:
        raise FrontierError("diagnostic matchups differ in search operator")

    matrix: dict[str, Any] = {
        "schema_version": SCHEMA,
        "trial_id": trial_id,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "canonical_selector": {
            "outcome": "no_eligible_checkpoint",
            "reason": NO_WINNER_REASON,
            "thresholds_unchanged": True,
            "selection_contract": copy.deepcopy(campaign["selection_contract"]),
        },
        "campaign": {
            "path": str(campaign_path),
            "file_sha256": _file_sha256(campaign_path),
            "campaign_sha256": campaign["campaign_sha256"],
        },
        "fingerprints": fingerprints,
        "frontier": frontier,
        "manifest": str(manifest_path),
        "manifest_hash": manifest["manifest_hash"],
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
        "operator_selection": operator,
        "operator_search": search,
        "science_config_hash": next(iter(science_hashes)),
        "seed_cohort_id": cohort,
        "common_random_numbers": True,
        "seat_swapped": True,
        "internal_claim": {"base_seed": internal_base_seed, "pairs": internal_pairs},
        "external_placeholder_claim": {
            "base_seed": external_base_seed,
            "pairs": external_placeholder_pairs,
            "launch": False,
        },
        "workers_per_gpu": fleet.DEFAULT_WORKERS_PER_GPU,
        "physical_gpus": len(occupied),
        "matchups": rows,
    }
    matrix["state_sha256"] = _digest(matrix)
    return matrix, plans


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--fingerprint", action="append", default=[])
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--internal-pairs", type=int, default=128)
    parser.add_argument("--external-placeholder-pairs", type=int, default=4)
    parser.add_argument("--internal-base-seed", type=int, required=True)
    parser.add_argument("--external-base-seed", type=int, required=True)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        bindings = _parse_bindings(args.fingerprint)
        matrix, plans = build_matrix(
            manifest_path=args.manifest,
            campaign_path=args.campaign,
            fingerprint_bindings=bindings,
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
            raise FrontierError(f"refusing to overwrite matrix artifacts: {existing}")
        for path, plan in plans.items():
            fleet.write_new_readonly(path, plan)
        fleet.write_new_readonly(matrix_path, matrix)
        result = {
            "matrix": str(matrix_path),
            "state_sha256": matrix["state_sha256"],
            "diagnostic_only": True,
            "promotion_eligible": False,
            "frontier": matrix["frontier"],
            "matchups": len(matrix["matchups"]),
            "physical_gpus": matrix["physical_gpus"],
        }
    except (
        FrontierError,
        active_campaign.CampaignError,
        fleet.FleetError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        print(f"active-policy diagnostic frontier refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
