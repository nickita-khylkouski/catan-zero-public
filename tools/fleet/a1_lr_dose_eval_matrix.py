#!/usr/bin/env python3
"""Seal the 64-H100 A-D learner screen against f7 and the v5 incumbent.

This is a matrix planner, not a launcher.  It assigns every physical H100 to
exactly one independent candidate/baseline matchup and emits the eight normal
``a1_h100_eval_fleet.py`` plans plus their exact launch/status/collect argv.
All matchups reuse one explicit paired-seed cohort, so comparisons have common
random numbers while retaining seat swaps within every pair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
for import_root in (REPO_ROOT / "src", REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from tools import a1_current_science_contract as current_science  # noqa: E402
from tools import a1_b200_lr_dose_campaign as lr_campaign  # noqa: E402
from tools import a1_function_preserving_upgrade as architecture_upgrade  # noqa: E402
from tools.champion_registry import ChampionRegistry  # noqa: E402
from tools.fleet import a1_h100_eval_fleet as fleet  # noqa: E402


SCHEMA = "a1-lr-dose-eval-matrix-v1"
ARMS = ("A", "B", "C", "D")
TEACHER_OPERATORS = (
    "base_n128_d6",
    "adaptive_n256_w20_d6",
    "adaptive_n256_w40_d6",
)
MATCHUPS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("A", "f7", ("h100-8a",)),
    ("A", "v5", ("c1", "c2")),
    ("B", "f7", ("h100-8b",)),
    ("B", "v5", ("c3", "c4")),
    ("C", "f7", ("h100-8c",)),
    ("C", "v5", ("c5", "c6")),
    ("D", "f7", ("h100-8d",)),
    ("D", "v5", ("c7", "c8")),
)


class MatrixError(ValueError):
    """The requested A-D evaluation matrix is incomplete or inconsistent."""


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


def _parse_arm_paths(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for raw in values:
        arm, separator, path = raw.partition("=")
        if not separator or arm not in ARMS or not path or arm in result:
            raise MatrixError("candidates must be unique A=PATH through D=PATH")
        result[arm] = Path(path).expanduser().resolve(strict=True)
    if set(result) != set(ARMS):
        raise MatrixError(f"candidate arms must be exactly {list(ARMS)}")
    if len({path for path in result.values()}) != len(ARMS):
        raise MatrixError("candidate arms must resolve to four distinct paths")
    return result


def _authenticate_completed_arms(
    campaign: dict[str, Any], *, f7: Path, candidates: dict[str, Path]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    upgrade_receipt_path = Path(
        str(campaign["inputs"]["architecture_upgrade_receipt"])
    ).resolve(strict=True)
    if _file_sha256(upgrade_receipt_path) != campaign["inputs"].get(
        "architecture_upgrade_receipt_sha256"
    ):
        raise MatrixError("campaign architecture-upgrade receipt bytes drifted")
    try:
        upgrade = architecture_upgrade.verify_receipt(upgrade_receipt_path)
    except architecture_upgrade.UpgradeError as error:
        raise MatrixError(f"campaign architecture upgrade refused: {error}") from error
    raw_f7 = f7.expanduser().resolve(strict=True)
    initializer = Path(str(upgrade["upgraded_initializer"]["path"])).resolve(
        strict=True
    )
    if (
        Path(str(upgrade["source"]["path"])).resolve(strict=True) != raw_f7
        or upgrade["source"]["sha256"] != fleet._sha256(raw_f7)
        or upgrade["upgraded_initializer"]["sha256"]
        != fleet._sha256(initializer)
    ):
        raise MatrixError("function-preserving upgrade does not map raw f7 to its initializer")

    arm_receipts: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        completed = lr_campaign._verify_completed_arm_receipt(campaign, arm=arm)
        checkpoint = completed["artifacts"]["checkpoint"]
        if Path(checkpoint["path"]).resolve(strict=True) != candidates[arm]:
            raise MatrixError(f"candidate {arm} differs from its completed receipt")
        raw_receipt = json.loads(Path(completed["receipt"]).read_text(encoding="utf-8"))
        command = raw_receipt.get("command") if isinstance(raw_receipt, dict) else None
        if (
            raw_receipt.get("function_preserving_upgrade") != upgrade
            or not isinstance(command, list)
            or lr_campaign._option(command, "--init-checkpoint") != str(initializer)
        ):
            raise MatrixError(
                f"arm {arm} does not bind the shared upgraded initializer as actual init"
            )
        arm_receipts[arm] = {
            **completed,
            "actual_initializer": {
                "path": str(initializer),
                "sha256": upgrade["upgraded_initializer"]["sha256"],
            },
        }
    return upgrade, arm_receipts


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
            raise AssertionError("collect command requires an output directory")
        argv += ["--output-dir", str(output_dir)]
    return argv


def build_matrix(
    *,
    manifest_path: Path,
    campaign_path: Path,
    registry_path: Path,
    f7: Path,
    v5: Path,
    candidates: dict[str, Path],
    internal_pairs: int,
    external_placeholder_pairs: int,
    internal_base_seed: int,
    external_base_seed: int,
    trial_id: str,
    expected_selected_operator: str,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[Path, dict[str, Any]]]:
    science_contract = current_science.load()
    operator_selection = science_contract["operator_selection"]
    if operator_selection["status"] != "adopted_teacher_campaign":
        raise MatrixError(
            "teacher operator is still provisional; adopt the causal campaign "
            "before sealing candidate evaluation"
        )
    if operator_selection.get("selected_operator") != expected_selected_operator:
        raise MatrixError(
            "adopted teacher differs from the requested evaluation operator: "
            f"expected={expected_selected_operator} "
            f"actual={operator_selection.get('selected_operator')}"
        )
    if internal_pairs < 8 or internal_pairs % 8:
        raise MatrixError("internal pairs must be a positive multiple of 8")
    if external_placeholder_pairs < 4:
        raise MatrixError("external placeholder requires at least four pairs")

    manifest_path = manifest_path.expanduser().resolve(strict=True)
    campaign_path = campaign_path.expanduser().resolve(strict=True)
    registry_path = registry_path.expanduser().resolve(strict=True)
    f7 = f7.expanduser().resolve(strict=True)
    v5 = v5.expanduser().resolve(strict=True)
    output_dir = output_dir.expanduser().resolve()
    manifest = fleet.load_manifest(
        manifest_path, expected_shapes=fleet.FULL_EXPECTED_SHAPES
    )
    if len(fleet.gpu_slots(manifest)) != 64:
        raise MatrixError("A-D evaluation requires the exact 64-H100 fleet")
    registry = ChampionRegistry.load(registry_path)
    incumbent = registry.get_role("generator_champion")
    if incumbent is None or Path(incumbent.checkpoint_path).resolve() != v5:
        raise MatrixError("v5 must be the authoritative generator_champion")

    campaign = lr_campaign._load_bound_json(
        campaign_path, schema=lr_campaign.SCHEMA
    )
    if campaign["lineage_contract"]["expected_parent_sha256"] != fleet._sha256(f7):
        raise MatrixError("LR-dose campaign parent differs from authenticated f7")
    function_preserving_upgrade, arm_receipts = _authenticate_completed_arms(
        campaign, f7=f7, candidates=candidates
    )
    candidate_parent = Path(
        function_preserving_upgrade["upgraded_initializer"]["path"]
    ).resolve(strict=True)

    search = current_science.search()
    c_scale = float(search["c_scale"])
    sigma_eval = float(search["sigma_eval"])
    cohort_id = f"{trial_id}-common"
    plans: dict[Path, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    occupied_slots: set[str] = set()

    for arm, baseline_name, aliases in MATCHUPS:
        baseline = f7 if baseline_name == "f7" else v5
        comparison_mode = (
            "historical_comparison" if baseline_name == "f7" else "branch_challenge"
        )
        reason = (
            "lr_dose_candidate_vs_authenticated_f7_parent"
            if baseline_name == "f7"
            else None
        )
        iteration_id = f"{trial_id}-{arm.lower()}-{baseline_name}"
        plan = fleet.build_plan(
            manifest,
            candidate=candidates[arm],
            champion=baseline,
            candidate_parent=candidate_parent,
            registry=registry,
            internal_pairs=internal_pairs,
            external_pairs=external_placeholder_pairs,
            internal_base_seed=internal_base_seed,
            external_base_seed=external_base_seed,
            workers_per_gpu=fleet.DEFAULT_WORKERS_PER_GPU,
            iteration_id=iteration_id,
            seed_cohort_id=cohort_id,
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
        slot_ids = {str(job["slot_id"]) for job in internal_jobs}
        if len(internal_jobs) != 8 or len(slot_ids) != 8:
            raise MatrixError(f"{arm}-{baseline_name} did not resolve eight lanes")
        overlap = occupied_slots & slot_ids
        if overlap:
            raise MatrixError(f"physical GPU assigned twice: {sorted(overlap)}")
        occupied_slots |= slot_ids

        plan_path = output_dir / f"{arm.lower()}-vs-{baseline_name}.plan.json"
        plans[plan_path] = plan
        collect_dir = output_dir / "collected" / f"{arm.lower()}-vs-{baseline_name}"
        rows.append(
            {
                "arm": arm,
                "baseline": baseline_name,
                "comparison_mode": comparison_mode,
                "promotion_eligible": bool(
                    plan["evaluation_binding"]["promotion_eligible"]
                ),
                "host_aliases": list(aliases),
                "gpu_slots": sorted(slot_ids),
                "plan": str(plan_path),
                "plan_hash": plan["plan_hash"],
                "candidate_sha256": plan["candidate"]["sha256"],
                "baseline_sha256": plan["champion"]["sha256"],
                "paired_games": internal_pairs * 2,
                "collect_output_dir": str(collect_dir),
                "launch_argv": _command(
                    manifest=manifest_path,
                    operation="launch",
                    plan=plan_path,
                ),
                "status_argv": _command(
                    manifest=manifest_path,
                    operation="status",
                    plan=plan_path,
                ),
                "collect_argv": _command(
                    manifest=manifest_path,
                    operation="collect",
                    plan=plan_path,
                    output_dir=collect_dir,
                ),
            }
        )

    expected_slots = {str(slot["slot_id"]) for slot in fleet.gpu_slots(manifest)}
    if occupied_slots != expected_slots:
        raise MatrixError(
            "matrix does not cover the fleet exactly: "
            f"missing={sorted(expected_slots - occupied_slots)}"
        )
    science_hashes = {plan["science_config_hash"] for plan in plans.values()}
    if len(science_hashes) != 1:
        raise MatrixError("matchups do not share one selected teacher operator")

    matrix: dict[str, Any] = {
        "schema_version": SCHEMA,
        "trial_id": trial_id,
        "manifest": str(manifest_path),
        "manifest_hash": manifest["manifest_hash"],
        "training_campaign": {
            "path": str(campaign_path),
            "file_sha256": _file_sha256(campaign_path),
            "campaign_sha256": campaign["campaign_sha256"],
            "function_preserving_upgrade": function_preserving_upgrade,
            "arm_receipts": arm_receipts,
        },
        "registry": str(registry_path),
        "operator_selection_status": operator_selection["status"],
        "selected_operator": expected_selected_operator,
        "operator_selection": operator_selection,
        "science_contract_sha256": _file_sha256(current_science.CONTRACT_PATH),
        "operator_search": search,
        "science_config_hash": next(iter(science_hashes)),
        "seed_cohort_id": cohort_id,
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
        "physical_gpus": len(occupied_slots),
        "matchups": rows,
        "launch_semantics": (
            "Run the eight launch_argv arrays concurrently from the B200 control "
            "host; only the internal phase is authorized by this matrix."
        ),
    }
    matrix["state_sha256"] = _digest(matrix)
    return matrix, plans


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--f7", type=Path, required=True)
    parser.add_argument("--v5", type=Path, required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--internal-pairs", type=int, default=128)
    parser.add_argument("--external-placeholder-pairs", type=int, default=4)
    parser.add_argument("--internal-base-seed", type=int, required=True)
    parser.add_argument("--external-base-seed", type=int, required=True)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument(
        "--selected-operator", choices=TEACHER_OPERATORS, required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        candidates = _parse_arm_paths(args.candidate)
        matrix, plans = build_matrix(
            manifest_path=args.manifest,
            campaign_path=args.campaign,
            registry_path=args.registry,
            f7=args.f7,
            v5=args.v5,
            candidates=candidates,
            internal_pairs=args.internal_pairs,
            external_placeholder_pairs=args.external_placeholder_pairs,
            internal_base_seed=args.internal_base_seed,
            external_base_seed=args.external_base_seed,
            trial_id=args.trial_id,
            expected_selected_operator=args.selected_operator,
            output_dir=args.output_dir,
        )
        matrix_path = args.output_dir.expanduser().resolve() / "matrix.json"
        targets = [*plans, matrix_path]
        existing = [str(path) for path in targets if path.exists()]
        if existing:
            raise MatrixError(f"refusing to overwrite matrix artifacts: {existing}")
        for path, plan in plans.items():
            fleet.write_new_readonly(path, plan)
        fleet.write_new_readonly(matrix_path, matrix)
        print(
            json.dumps(
                {
                    "matrix": str(matrix_path),
                    "state_sha256": matrix["state_sha256"],
                    "physical_gpus": matrix["physical_gpus"],
                    "matchups": len(matrix["matchups"]),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (
        MatrixError,
        lr_campaign.CampaignError,
        fleet.FleetError,
        OSError,
        KeyError,
        ValueError,
    ) as error:
        print(f"A-D evaluation matrix refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
