#!/usr/bin/env python3
"""Evaluate the direction-corrected learner recovery on all 64 H100s.

The recovery selector is diagnostic by construction: it chooses the checkpoint
with the largest positive coherent-teacher closure while keeping parent KL and
shared-trunk drift inside the preregistered budgets.  This controller replays
that authority, then splits the exact 64-H100 fleet into two balanced panels:

* selected recovery checkpoint versus exact f7; and
* selected recovery checkpoint versus the registry v5 incumbent.

Both panels use one coherent-public n128 operator, common random numbers, and
seat-swapped games.  They remain historical comparisons because a diagnostic
R&D selection is evidence for the next sealed recipe, not itself a promotion.
If the strict recovery selector refuses every checkpoint, this controller can
instead authenticate that signed refusal and evaluate one explicitly named
positive-closure frontier point.  A refusal frontier never impersonates a
selection and is always diagnostic/non-promotable.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
for import_root in (REPO_ROOT / "src", REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from tools import a1_b200_active_policy_campaign as stage_a  # noqa: E402
from tools import a1_b200_stage_b_ablation_campaign as stage_b  # noqa: E402
from tools import a1_current_science_contract as current_science  # noqa: E402
from tools import a1_function_preserving_upgrade as architecture_upgrade  # noqa: E402
from tools.champion_registry import ChampionRegistry  # noqa: E402
from tools.fleet import a1_h100_eval_fleet as fleet  # noqa: E402


SCHEMA = "a1-direction-corrected-recovery-eval-matrix-v1"
OPERATION_SCHEMA = "a1-direction-corrected-recovery-eval-operation-v1"
COMPLETION_SCHEMA = "a1-direction-corrected-recovery-eval-completion-v1"
BASELINES = ("f7", "v5")
RECOVERY_REFUSAL_SCHEMA = "a1-direction-corrected-recovery-refusal-v1"
REFUSAL_AUTHORITY_KIND = "direction_corrected_recovery_refusal_frontier"


class MatrixError(RuntimeError):
    """The recovery evaluation authority or fleet matrix is inconsistent."""


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


def _balanced_host_groups(
    manifest: Mapping[str, Any], *, group_count: int
) -> list[tuple[str, ...]]:
    hosts = manifest.get("hosts")
    if not isinstance(hosts, list) or not 1 <= group_count <= len(hosts):
        raise MatrixError("cannot partition the approved H100 hosts")
    groups: list[list[str]] = [[] for _ in range(group_count)]
    totals = [0] * group_count
    for host in sorted(
        hosts, key=lambda row: (-int(row["gpu_count"]), str(row["alias"]))
    ):
        index = min(range(group_count), key=lambda item: (totals[item], item))
        groups[index].append(str(host["alias"]))
        totals[index] += int(host["gpu_count"])
    if any(not group for group in groups) or totals != [32, 32]:
        raise MatrixError(f"recovery matrix did not split 64 H100s 32/32: {totals}")
    return [tuple(group) for group in groups]


def _load_baselines(
    *, campaign: Mapping[str, Any], registry_path: Path
) -> dict[str, Any]:
    inputs = campaign.get("inputs")
    if not isinstance(inputs, Mapping):
        raise MatrixError("recovery source Stage-A campaign lost its inputs")
    try:
        upgrade = architecture_upgrade.verify_receipt(
            Path(str(inputs["architecture_upgrade_receipt"]))
        )
    except (KeyError, architecture_upgrade.UpgradeError) as error:
        raise MatrixError(f"f7 function-preserving upgrade refused: {error}") from error
    f7 = Path(str(upgrade["source"]["path"])).resolve(strict=True)
    initializer = Path(str(upgrade["upgraded_initializer"]["path"])).resolve(
        strict=True
    )
    if (
        upgrade["source"].get("sha256") != stage_a.EXPECTED_F7_PARENT_SHA256
        or _file_sha256(f7) != stage_a.EXPECTED_F7_PARENT_SHA256
        or _file_sha256(initializer)
        != campaign["lineage_contract"]["upgraded_initializer_sha256"]
    ):
        raise MatrixError("recovery authority no longer maps exact f7 to its initializer")

    registry_path = registry_path.expanduser().resolve(strict=True)
    registry = ChampionRegistry.load(registry_path)
    pointer = registry.get_role("generator_champion")
    if pointer is None:
        raise MatrixError("registry has no generator_champion")
    v5 = Path(pointer.checkpoint_path).expanduser().resolve(strict=True)
    if fleet._sha256(v5) != stage_a.EXPECTED_CORPUS_PRODUCER_SHA256:  # noqa: SLF001
        raise MatrixError("registry generator_champion is not authoritative v5")
    return {
        "upgrade": upgrade,
        "f7": f7,
        "initializer": initializer,
        "registry_path": registry_path,
        "registry": registry,
        "v5": v5,
    }


def _load_authority(
    *, selection_path: Path, registry_path: Path
) -> dict[str, Any]:
    try:
        dose = stage_b.load_recovery_selected_dose(selection_path)
    except stage_b.CampaignError as error:
        raise MatrixError(f"recovery selection refused: {error}") from error
    selection_path = Path(str(dose["selection_authority"]["path"])).resolve(
        strict=True
    )
    stage_a_path = Path(str(dose["source_campaign"]["path"])).resolve(strict=True)
    try:
        loaded_stage_a_path, campaign = stage_a._load_campaign(stage_a_path)  # noqa: SLF001
    except stage_a.CampaignError as error:
        raise MatrixError(f"source Stage-A campaign refused: {error}") from error
    if loaded_stage_a_path != stage_a_path:
        raise MatrixError("recovery source Stage-A path changed")
    baselines = _load_baselines(campaign=campaign, registry_path=registry_path)
    candidate = Path(str(dose["selected_checkpoint"]["path"])).resolve(strict=True)
    if _file_sha256(candidate) != dose["selected_checkpoint"]["sha256"]:
        raise MatrixError("selected recovery checkpoint bytes changed")
    return {
        "selection_path": selection_path,
        "dose": dose,
        "stage_a_path": stage_a_path,
        "campaign": campaign,
        "candidate": candidate,
        **baselines,
    }


def _load_refusal_authority(
    *,
    refusal_path: Path,
    registry_path: Path,
    diagnostic_arm: str,
    diagnostic_step: int,
) -> dict[str, Any]:
    """Authenticate a formally refused frontier point for diagnostic play only."""

    try:
        refusal_path, refusal = stage_b._load_signed(  # noqa: SLF001
            refusal_path,
            where="direction-corrected recovery refusal",
            schema=RECOVERY_REFUSAL_SCHEMA,
            digest_field="refusal_sha256",
        )
    except stage_b.CampaignError as error:
        raise MatrixError(f"recovery refusal refused: {error}") from error
    campaign_ref = refusal.get("campaign")
    if not isinstance(campaign_ref, Mapping):
        raise MatrixError("recovery refusal lost its campaign reference")
    try:
        plan_path, plan = stage_b._load_signed(  # noqa: SLF001
            Path(str(campaign_ref.get("path", ""))),
            where="direction-corrected recovery campaign",
            schema=stage_b.RECOVERY_CAMPAIGN_SCHEMA,
            digest_field="campaign_sha256",
        )
    except stage_b.CampaignError as error:
        raise MatrixError(f"recovery campaign refused: {error}") from error
    outputs = plan.get("outputs")
    if (
        campaign_ref.get("file_sha256") != _file_sha256(plan_path)
        or campaign_ref.get("campaign_sha256") != plan.get("campaign_sha256")
        or refusal.get("source") != plan.get("source")
        or refusal.get("lineage") != plan.get("lineage")
        or refusal.get("stage_a_refusal_evidence")
        != plan.get("stage_a_refusal_evidence")
        or refusal.get("selection_contract") != plan.get("selection_contract")
        or refusal.get("diagnostic_only") is not True
        or refusal.get("promotion_eligible") is not False
        or refusal.get("playing_strength_evaluation_required") is not True
        or not isinstance(outputs, Mapping)
        or Path(str(outputs.get("refusal", ""))).expanduser().resolve(strict=False)
        != refusal_path
    ):
        raise MatrixError("recovery refusal campaign/diagnostic binding drifted")
    lineage = plan.get("lineage")
    if (
        not isinstance(lineage, Mapping)
        or lineage.get("learner_parent_sha256")
        != stage_a.EXPECTED_F7_PARENT_SHA256
        or lineage.get("every_arm_restarts_from_exact_upgraded_f7") is not True
        or lineage.get("fresh_adam_every_arm") is not True
        or lineage.get("candidate_chaining_forbidden") is not True
    ):
        raise MatrixError("recovery refusal lineage is not fresh f7/fresh Adam")
    try:
        stage_a_path, campaign, _bindings = stage_b._replay_stage_a_refusal(  # noqa: SLF001
            refusal["stage_a_refusal_evidence"]
        )
    except stage_b.CampaignError as error:
        raise MatrixError(f"source Stage-A refusal no longer replays: {error}") from error

    arms = plan.get("arms")
    refs = refusal.get("fingerprints")
    trajectory = plan.get("trajectory")
    contract = plan.get("selection_contract")
    if (
        not isinstance(arms, Mapping)
        or not isinstance(refs, Mapping)
        or set(refs) != set(arms)
        or diagnostic_arm not in arms
        or not isinstance(trajectory, Mapping)
        or not isinstance(contract, Mapping)
    ):
        raise MatrixError("recovery refusal arm/trajectory surface is malformed")
    try:
        frontier = tuple(int(value) for value in trajectory["checkpoint_steps"])
        parent_cap = float(contract["parent_kl_max"])
        trunk_cap = float(contract["trunk_relative_l2_max"])
    except (KeyError, TypeError, ValueError) as error:
        raise MatrixError("recovery refusal contract is malformed") from error
    if (
        diagnostic_step not in frontier
        or contract.get("positive_teacher_gap_closure_required") is not True
        or not all(math.isfinite(value) and value > 0.0 for value in (parent_cap, trunk_cap))
    ):
        raise MatrixError("requested diagnostic point is outside the frozen frontier")

    claimed = refusal.get("checkpoint_candidates")
    if not isinstance(claimed, list) or not claimed:
        raise MatrixError("recovery refusal lost its checkpoint candidate table")
    selected_rows: list[dict[str, Any]] = []
    for value in claimed:
        if not isinstance(value, Mapping):
            raise MatrixError("recovery refusal checkpoint table is malformed")
        try:
            row = copy.deepcopy(dict(value))
            row_arm = str(row["arm"])
            row_step = int(row["step"])
            parent_kl = float(row["parent_kl"])
            trunk = float(row["trunk_relative_l2"])
            closure = float(row["teacher_gap_closure"])
        except (KeyError, TypeError, ValueError) as error:
            raise MatrixError("recovery refusal checkpoint row is malformed") from error
        eligible = parent_kl <= parent_cap and trunk <= trunk_cap and closure > 0.0
        if (
            row.get("eligible") is not eligible
            or row_arm not in arms
            or row_step not in frontier
            or not all(math.isfinite(metric) for metric in (parent_kl, trunk, closure))
        ):
            raise MatrixError("recovery refusal checkpoint eligibility did not replay")
        if eligible:
            raise MatrixError("signed refusal now contains an eligible checkpoint")
        if row_arm == diagnostic_arm and row_step == diagnostic_step:
            selected_rows.append(row)
    if len(selected_rows) != 1:
        raise MatrixError("requested refusal frontier point is not unique")
    selected = selected_rows[0]
    if float(selected["teacher_gap_closure"]) <= 0.0:
        raise MatrixError("refusal frontier evaluation requires positive teacher closure")

    ref = refs[diagnostic_arm]
    arm_plan = arms[diagnostic_arm]
    if not isinstance(ref, Mapping) or not isinstance(arm_plan, Mapping):
        raise MatrixError("requested recovery arm binding is malformed")
    try:
        fingerprint_path, fingerprint = stage_b._load_signed(  # noqa: SLF001
            Path(str(ref.get("path", ""))),
            where=f"recovery arm {diagnostic_arm} fingerprint",
            schema=stage_b.RECOVERY_FINGERPRINT_SCHEMA,
            digest_field="fingerprint_sha256",
        )
        receipt_path, receipt = stage_b._recovery_arm_receipt(  # noqa: SLF001
            plan, diagnostic_arm, fingerprint
        )
    except stage_b.CampaignError as error:
        raise MatrixError(f"recovery arm evidence refused: {error}") from error
    recipe = arm_plan.get("recipe_overrides")
    fingerprint_rows = fingerprint.get("checkpoints")
    fingerprint_selected = (
        [
            row
            for row in fingerprint_rows
            if isinstance(row, Mapping) and int(row.get("step", -1)) == diagnostic_step
        ]
        if isinstance(fingerprint_rows, list)
        else []
    )
    if (
        ref.get("file_sha256") != _file_sha256(fingerprint_path)
        or ref.get("fingerprint_sha256") != fingerprint.get("fingerprint_sha256")
        or fingerprint.get("campaign_sha256") != plan.get("campaign_sha256")
        or fingerprint.get("arm") != diagnostic_arm
        or fingerprint.get("recipe_overrides") != recipe
        or fingerprint.get("diagnostic_only") is not True
        or fingerprint.get("promotion_eligible") is not False
        or len(fingerprint_selected) != 1
        or any(
            fingerprint_selected[0].get(key) != selected.get(key)
            for key in (
                "step",
                "checkpoint",
                "checkpoint_sha256",
                "parent_kl",
                "trunk_relative_l2",
                "teacher_gap_closure",
            )
        )
        or not isinstance(recipe, Mapping)
    ):
        raise MatrixError("requested refusal frontier fingerprint drifted")
    candidate = Path(str(selected["checkpoint"])).expanduser().resolve(strict=True)
    if selected.get("checkpoint_sha256") != _file_sha256(candidate):
        raise MatrixError("requested refusal frontier checkpoint bytes changed")

    topology = plan.get("fixed_surface", {}).get("topology", {})
    try:
        aux_batch = int(recipe["policy_aux_active_batch_size"])
        world_size = int(topology["world_size"])
    except (KeyError, TypeError, ValueError) as error:
        raise MatrixError("recovery refusal recipe/topology is malformed") from error
    evaluation_authority = {
        "kind": REFUSAL_AUTHORITY_KIND,
        "path": str(refusal_path),
        "file_sha256": _file_sha256(refusal_path),
        "refusal_sha256": refusal["refusal_sha256"],
    }
    dose = {
        "authority_kind": REFUSAL_AUTHORITY_KIND,
        "evaluation_authority": evaluation_authority,
        "source_campaign": {
            "path": str(stage_a_path),
            "file_sha256": _file_sha256(stage_a_path),
            "campaign_sha256": campaign["campaign_sha256"],
        },
        "recovery_campaign": {
            "path": str(plan_path),
            "file_sha256": _file_sha256(plan_path),
            "campaign_sha256": plan["campaign_sha256"],
        },
        "recovery_fingerprint": copy.deepcopy(dict(ref)),
        "recovery_receipt": {
            "path": str(receipt_path),
            "file_sha256": _file_sha256(receipt_path),
            "receipt_sha256": receipt["receipt_sha256"],
        },
        "selected_arm": diagnostic_arm,
        "policy_aux_active_batch_size": aux_batch,
        "optimizer_steps": diagnostic_step,
        "checkpoint_steps": [step for step in frontier if step <= diagnostic_step],
        "expected_aux_active_row_draws": aux_batch * world_size * diagnostic_step,
        "reference_parent_kl": float(selected["parent_kl"]),
        "reference_trunk_relative_l2": float(selected["trunk_relative_l2"]),
        "reference_teacher_gap_closure": float(selected["teacher_gap_closure"]),
        "selected_recipe_overrides": copy.deepcopy(dict(recipe)),
        "strict_selector_outcome": "formal_refusal_no_eligible_checkpoint",
        "diagnostic_only": True,
        "promotion_eligible": False,
        "selected_checkpoint": {
            "path": str(candidate),
            "sha256": _file_sha256(candidate),
            "role": "refused_positive_closure_frontier_diagnostic_only",
        },
    }
    baselines = _load_baselines(campaign=campaign, registry_path=registry_path)
    return {
        "refusal_path": refusal_path,
        "dose": dose,
        "stage_a_path": stage_a_path,
        "campaign": campaign,
        "candidate": candidate,
        **baselines,
    }


def _resolve_authority(
    *,
    selection_path: Path | None,
    refusal_path: Path | None,
    registry_path: Path,
    diagnostic_arm: str,
    diagnostic_step: int,
) -> dict[str, Any]:
    if (selection_path is None) == (refusal_path is None):
        raise MatrixError("provide exactly one of recovery selection or refusal")
    if selection_path is not None:
        authority = _load_authority(
            selection_path=selection_path, registry_path=registry_path
        )
        authority["dose"]["evaluation_authority"] = copy.deepcopy(
            authority["dose"]["selection_authority"]
        )
        authority["dose"]["evaluation_authority"]["kind"] = authority["dose"][
            "authority_kind"
        ]
        return authority
    assert refusal_path is not None
    return _load_refusal_authority(
        refusal_path=refusal_path,
        registry_path=registry_path,
        diagnostic_arm=diagnostic_arm,
        diagnostic_step=diagnostic_step,
    )


def _operation_command(
    *, matrix_path: Path, operation: str, output: Path | None = None
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve(strict=True)),
        "--matrix",
        str(matrix_path),
        operation,
    ]
    if operation in {"launch", "collect"}:
        if output is None:
            raise AssertionError(f"{operation} requires an output receipt")
        command += ["--go", "--out", str(output)]
    elif operation == "wait":
        if output is None:
            raise AssertionError("wait requires an output receipt")
        command += ["--out", str(output)]
    return command


def build_matrix(
    *,
    manifest_path: Path,
    selection_path: Path | None,
    refusal_path: Path | None,
    diagnostic_arm: str,
    diagnostic_step: int,
    registry_path: Path,
    internal_pairs: int,
    external_placeholder_pairs: int,
    internal_base_seed: int,
    external_base_seed: int,
    trial_id: str,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[Path, dict[str, Any]]]:
    if internal_pairs < 1 or external_placeholder_pairs < 1:
        raise MatrixError("pair counts must be positive")
    manifest_path = manifest_path.expanduser().resolve(strict=True)
    output_dir = output_dir.expanduser().resolve(strict=False)
    manifest = fleet.load_manifest(
        manifest_path, expected_shapes=fleet.FULL_EXPECTED_SHAPES
    )
    if len(fleet.gpu_slots(manifest)) != 64:
        raise MatrixError("recovery evaluation requires the exact 64-H100 fleet")
    authority = _resolve_authority(
        selection_path=selection_path,
        refusal_path=refusal_path,
        registry_path=registry_path,
        diagnostic_arm=diagnostic_arm,
        diagnostic_step=diagnostic_step,
    )
    groups = _balanced_host_groups(manifest, group_count=len(BASELINES))
    external_placeholder_pairs = max(external_placeholder_pairs, 16)

    science = current_science.load()
    operator_selection = science["operator_selection"]
    search = current_science.search()
    if (
        operator_selection.get("status") != "adopted_teacher_campaign"
        or operator_selection.get("selected_operator") != "base_n128_d6"
        or search.get("coherent_public_belief_search") is not True
        or search.get("information_set_search") is not False
        or int(search.get("n_full", -1)) != 128
    ):
        raise MatrixError("current science contract is not exact coherent-public n128")
    c_scale = float(search["c_scale"])
    sigma_eval = float(search["sigma_eval"])
    cohort = f"{trial_id}-common"
    plans: dict[Path, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    occupied: set[str] = set()

    for baseline_name, aliases in zip(BASELINES, groups, strict=True):
        baseline = authority[baseline_name]
        authority_kind = authority["dose"]["authority_kind"]
        reason = (
            f"{authority_kind}_vs_authenticated_f7_diagnostic"
            if baseline_name == "f7"
            else f"{authority_kind}_vs_registry_v5_diagnostic"
        )
        plan = fleet.build_plan(
            manifest,
            candidate=authority["candidate"],
            champion=baseline,
            candidate_parent=authority["initializer"],
            registry=authority["registry"],
            internal_pairs=internal_pairs,
            external_pairs=external_placeholder_pairs,
            internal_base_seed=internal_base_seed,
            external_base_seed=external_base_seed,
            workers_per_gpu=fleet.DEFAULT_WORKERS_PER_GPU,
            iteration_id=f"{trial_id}-vs-{baseline_name}",
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
        jobs = [job for job in plan["jobs"] if job["phase"] == "internal"]
        slots = {str(job["slot_id"]) for job in jobs}
        if len(slots) != 32 or occupied & slots:
            raise MatrixError("recovery matrix assigned a GPU twice or not at all")
        occupied |= slots
        plan_path = output_dir / f"recovery-vs-{baseline_name}.plan.json"
        plans[plan_path] = plan
        rows.append(
            {
                "baseline": baseline_name,
                "comparison_mode": "historical_comparison",
                "promotion_eligible": False,
                "host_aliases": list(aliases),
                "gpu_slots": sorted(slots),
                "plan": str(plan_path),
                "plan_hash": plan["plan_hash"],
                "candidate": {
                    "path": str(authority["candidate"]),
                    "sha256": _file_sha256(authority["candidate"]),
                    "arm": authority["dose"]["selected_arm"],
                    "optimizer_step": authority["dose"]["optimizer_steps"],
                },
                "baseline_checkpoint": {
                    "path": str(baseline),
                    "sha256": _file_sha256(baseline),
                },
                "paired_games": internal_pairs * 2,
                "collect_output_dir": str(
                    output_dir / "collected" / f"recovery-vs-{baseline_name}"
                ),
            }
        )

    expected_slots = {str(slot["slot_id"]) for slot in fleet.gpu_slots(manifest)}
    if occupied != expected_slots:
        raise MatrixError("recovery plans do not cover the exact 64-H100 fleet")
    science_hashes = {plan["science_config_hash"] for plan in plans.values()}
    if len(science_hashes) != 1:
        raise MatrixError("recovery matchups do not share one search operator")

    matrix_path = output_dir / "matrix.json"
    matrix: dict[str, Any] = {
        "schema_version": SCHEMA,
        "trial_id": trial_id,
        "manifest": str(manifest_path),
        "manifest_hash": manifest["manifest_hash"],
        "evaluation_authority": copy.deepcopy(
            authority["dose"]["evaluation_authority"]
        ),
        "selected_dose": copy.deepcopy(authority["dose"]),
        "registry": str(authority["registry_path"]),
        "function_preserving_upgrade": authority["upgrade"],
        "f7": {"path": str(authority["f7"]), "sha256": _file_sha256(authority["f7"])},
        "v5": {"path": str(authority["v5"]), "sha256": _file_sha256(authority["v5"])},
        "operator_selection": operator_selection,
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
    _matrix_path, matrix = _read_object(path, where="recovery evaluation matrix")
    unsigned = dict(matrix)
    stated = unsigned.pop("state_sha256", None)
    if matrix.get("schema_version") != SCHEMA or stated != _digest(unsigned):
        raise MatrixError("recovery evaluation matrix schema or digest drifted")
    evaluation_authority = matrix.get("evaluation_authority")
    if not isinstance(evaluation_authority, Mapping):
        raise MatrixError("recovery evaluation authority is malformed")
    authority_kind = evaluation_authority.get("kind")
    dose = matrix.get("selected_dose")
    if not isinstance(dose, Mapping):
        raise MatrixError("recovery selected dose is malformed")
    authority = _resolve_authority(
        selection_path=(
            Path(str(evaluation_authority["path"]))
            if authority_kind == "direction_corrected_recovery_selection"
            else None
        ),
        refusal_path=(
            Path(str(evaluation_authority["path"]))
            if authority_kind == REFUSAL_AUTHORITY_KIND
            else None
        ),
        registry_path=Path(matrix["registry"]),
        diagnostic_arm=str(dose.get("selected_arm", "")),
        diagnostic_step=int(dose.get("optimizer_steps", -1)),
    )
    if (
        matrix.get("evaluation_authority")
        != authority["dose"]["evaluation_authority"]
        or matrix.get("selected_dose") != authority["dose"]
        or matrix.get("function_preserving_upgrade") != authority["upgrade"]
    ):
        raise MatrixError("recovery evaluation authority changed after planning")
    manifest = fleet.load_manifest(
        Path(matrix["manifest"]), expected_shapes=fleet.FULL_EXPECTED_SHAPES
    )
    if manifest["manifest_hash"] != matrix.get("manifest_hash"):
        raise MatrixError("recovery evaluation fleet manifest drifted")
    rows = matrix.get("matchups")
    if (
        not isinstance(rows, list)
        or len(rows) != 2
        or {row.get("baseline") for row in rows} != set(BASELINES)
    ):
        raise MatrixError("recovery matchup set drifted")
    loaded: list[tuple[dict[str, Any], dict[str, Any]]] = []
    occupied: set[str] = set()
    for row in rows:
        plan = fleet.load_plan(Path(str(row["plan"])), manifest)
        baseline_name = str(row["baseline"])
        baseline = authority[baseline_name]
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
            or plan["candidate"]["source"] != str(authority["candidate"])
            or plan["candidate"]["sha256"] != _file_sha256(authority["candidate"])
            or plan["champion"]["source"] != str(baseline)
            or plan["champion"]["sha256"] != _file_sha256(baseline)
            or plan["evaluation_binding"]["promotion_eligible"] is not False
            or slots != set(row["gpu_slots"])
            or len(slots) != 32
            or occupied & slots
        ):
            raise MatrixError(f"recovery plan drifted for {baseline_name}")
        occupied |= slots
        loaded.append((row, plan))
    expected_slots = {str(slot["slot_id"]) for slot in fleet.gpu_slots(manifest)}
    if occupied != expected_slots or matrix.get("physical_gpus") != 64:
        raise MatrixError("recovery plans no longer cover exactly 64 GPUs")
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
            key = f"recovery-vs-{row['baseline']}"
            try:
                payload = future.result()
            except Exception as error:  # preserve the other panel's evidence
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
            raise MatrixError(f"recovery matrix entered a bad state: {bad}")
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
            raise MatrixError(f"recovery status does not cover 64 jobs: {counts}")
        elapsed = time.monotonic() - started
        if elapsed >= timeout_seconds:
            raise MatrixError(
                f"recovery matrix did not finish within {timeout_seconds:g}s: {counts}"
            )
        time.sleep(min(poll_seconds, max(0.0, timeout_seconds - elapsed)))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--manifest", type=Path, required=True)
    authority = plan.add_mutually_exclusive_group(required=True)
    authority.add_argument("--selection", type=Path)
    authority.add_argument("--refusal", type=Path)
    plan.add_argument("--diagnostic-arm", default="TRUST_V25")
    plan.add_argument("--diagnostic-step", type=int, default=24)
    plan.add_argument("--registry", type=Path, required=True)
    plan.add_argument("--internal-pairs", type=int, default=256)
    plan.add_argument("--external-placeholder-pairs", type=int, default=16)
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
                selection_path=args.selection,
                refusal_path=args.refusal,
                diagnostic_arm=args.diagnostic_arm,
                diagnostic_step=args.diagnostic_step,
                registry_path=args.registry,
                internal_pairs=args.internal_pairs,
                external_placeholder_pairs=args.external_placeholder_pairs,
                internal_base_seed=args.internal_base_seed,
                external_base_seed=args.external_base_seed,
                trial_id=args.trial_id,
                output_dir=args.output_dir,
            )
            matrix_path = args.output_dir.expanduser().resolve(strict=False) / "matrix.json"
            existing = [str(path) for path in [*plans, matrix_path] if path.exists()]
            if existing:
                raise MatrixError(f"refusing to overwrite matrix artifacts: {existing}")
            for path, payload in plans.items():
                fleet.write_new_readonly(path, payload)
            fleet.write_new_readonly(matrix_path, matrix)
            result = {
                "matrix": str(matrix_path),
                "state_sha256": matrix["state_sha256"],
                "selected_arm": matrix["selected_dose"]["selected_arm"],
                "selected_step": matrix["selected_dose"]["optimizer_steps"],
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
        stage_a.CampaignError,
        stage_b.CampaignError,
        fleet.FleetError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        print(f"recovery evaluation matrix refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
