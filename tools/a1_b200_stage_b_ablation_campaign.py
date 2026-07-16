#!/usr/bin/env python3
"""Run the selected-dose Stage-B learner treatment ablation on 8xB200.

Stage A, or a signed direction-corrected recovery after Stage A formally selects
no winner, selects an active-policy exposure and optimizer step.  This tool
replays that authority and holds the selected dose, coherent n128 corpus, exact
f7 function-preserving initializer, optimizer, topology, and every ordinary
learner setting fixed.  It then varies one treatment at a time:

* ``BASE``: legacy forced-value semantics, public-card LR 1x, surprise off;
* ``FORCED``: only END_TURN=0.1 and ROLL=0.25 forced-value weighting;
* ``CARD4``: only public-card residual LR 4x;
* ``SURPRISE``: only exact per-game policy-surprise weighting.
* ``TRUNK25`` / ``TRUNK10``: only reduce the shared-trunk LR to 0.25x / 0.10x;
* ``VTRUNK25``: only scale the scalar-value gradient entering the shared trunk
  to 0.25x, leaving policy gradients and value-head gradients unchanged;
* ``TRUST``: only enable a forward-KL projected-dual parent trust region.

The FORCED arm is structurally inactive when the authenticated corpus contains
no one-legal-action END_TURN/ROLL rows.  It is recorded in the signed plan but
is not launchable in that case.  This prevents a zero-exposure arm from being
misreported as a causal experiment while preserving the same campaign for a
future corpus that actually contains the treatment surface.

Every launched arm independently reloads the exact upgraded f7 initializer
with fresh Adam.  The tool is a thin authority/orchestration layer over
``a1_one_dose_train``; it does not implement another trainer.  All artifacts
remain diagnostic-only and cannot be promoted directly.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from tools import a1_b200_active_policy_campaign as stage_a  # noqa: E402
from tools import a1_b200_lr_dose_campaign as base_campaign  # noqa: E402
from tools import a1_function_preserving_upgrade as architecture_upgrade  # noqa: E402
from tools import a1_one_dose_train as one_dose  # noqa: E402
from tools import train_bc  # noqa: E402


SCHEMA = "a1-b200-stage-b-causal-ablation-campaign-v4"
ARM_RECEIPT_SCHEMA = "a1-b200-stage-b-arm-receipt-v1"
FINGERPRINT_SCHEMA = "a1-b200-stage-b-fingerprint-v1"
COMPARISON_SCHEMA = "a1-b200-stage-b-comparison-v3"
RECOVERY_SELECTION_SCHEMA = "a1-direction-corrected-recovery-selection-v1"
RECOVERY_CAMPAIGN_SCHEMA = "coherent-n128-trust-recovery-campaign-v1"
RECOVERY_FINGERPRINT_SCHEMA = "coherent-n128-trust-recovery-fingerprint-v1"
STAGE_A_NO_WINNER_REASON = (
    "no active-policy exposure checkpoint remained inside both drift budgets"
)
STAGE_A_FORMAL_REFUSAL = (
    "no_checkpoint_with_positive_teacher_gap_and_parent_kl_lte_0.03"
)
WORLD_SIZE = stage_a.WORLD_SIZE
LOCAL_BATCH_SIZE = stage_a.LOCAL_BATCH_SIZE
GLOBAL_BATCH_SIZE = stage_a.GLOBAL_BATCH_SIZE
CHECKPOINT_FRONTIER = stage_a.CHECKPOINT_STEPS
FORCED_TYPED_SPEC = "END_TURN=0.1,ROLL=0.25"
TRUST_DUAL_LR = 1.0
TRUST_MAX_WEIGHT = 1.0
DISTANCE_RATIO_LIMIT = 2.0
TREATMENT_FIELDS = frozenset(
    {
        "forced_row_value_action_type_weights",
        "public_card_lr_mult",
        "per_game_policy_surprise_weighting",
        "trunk_lr_mult",
        "value_trunk_grad_scale",
        "policy_kl_target",
        "policy_kl_dual_lr",
        "policy_kl_max_weight",
        "policy_kl_anchor_direction",
    }
)
ARM_ORDER = (
    "BASE",
    "FORCED",
    "CARD4",
    "SURPRISE",
    "TRUNK25",
    "TRUNK10",
    "VTRUNK25",
    "TRUST",
)
TREATMENTS: dict[str, dict[str, Any]] = {
    "BASE": {
        "forced_row_value_action_type_weights": "",
        "public_card_lr_mult": 1.0,
        "per_game_policy_surprise_weighting": False,
        "trunk_lr_mult": 1.0,
    },
    "FORCED": {
        "forced_row_value_action_type_weights": FORCED_TYPED_SPEC,
        "public_card_lr_mult": 1.0,
        "per_game_policy_surprise_weighting": False,
        "trunk_lr_mult": 1.0,
    },
    "CARD4": {
        "forced_row_value_action_type_weights": "",
        "public_card_lr_mult": 4.0,
        "per_game_policy_surprise_weighting": False,
        "trunk_lr_mult": 1.0,
    },
    "SURPRISE": {
        "forced_row_value_action_type_weights": "",
        "public_card_lr_mult": 1.0,
        "per_game_policy_surprise_weighting": True,
        "trunk_lr_mult": 1.0,
    },
    "TRUNK25": {
        "forced_row_value_action_type_weights": "",
        "public_card_lr_mult": 1.0,
        "per_game_policy_surprise_weighting": False,
        "trunk_lr_mult": 0.25,
    },
    "TRUNK10": {
        "forced_row_value_action_type_weights": "",
        "public_card_lr_mult": 1.0,
        "per_game_policy_surprise_weighting": False,
        "trunk_lr_mult": 0.10,
    },
    "VTRUNK25": {
        "forced_row_value_action_type_weights": "",
        "public_card_lr_mult": 1.0,
        "per_game_policy_surprise_weighting": False,
        "trunk_lr_mult": 1.0,
        "value_trunk_grad_scale": 0.25,
    },
    "TRUST": {
        "forced_row_value_action_type_weights": "",
        "public_card_lr_mult": 1.0,
        "per_game_policy_surprise_weighting": False,
        "trunk_lr_mult": 1.0,
    },
}


class CampaignError(RuntimeError):
    """A Stage-B input, treatment, or output is semantically invalid."""


def _file_sha256(path: Path) -> str:
    return stage_a._file_sha256(path)  # noqa: SLF001


def _value_sha256(value: object) -> str:
    return stage_a._value_sha256(value)  # noqa: SLF001


def _canonical_json(value: object) -> str:
    return stage_a._canonical_json(value)  # noqa: SLF001


def _regular_file(path: Path, *, where: str) -> Path:
    try:
        return stage_a._regular_file(path, where=where)  # noqa: SLF001
    except stage_a.CampaignError as error:
        raise CampaignError(str(error)) from error


def _write_immutable(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        stage_a._write_immutable(path, payload)  # noqa: SLF001
    except stage_a.CampaignError as error:
        raise CampaignError(str(error)) from error


def _load_json(path: Path, *, where: str) -> tuple[Path, dict[str, Any]]:
    try:
        return stage_a._load_json(path, where=where)  # noqa: SLF001
    except stage_a.CampaignError as error:
        raise CampaignError(str(error)) from error


def _load_signed(
    path: Path, *, where: str, schema: str, digest_field: str
) -> tuple[Path, dict[str, Any]]:
    try:
        return stage_a._load_signed(  # noqa: SLF001
            path, where=where, schema=schema, digest_field=digest_field
        )
    except stage_a.CampaignError as error:
        raise CampaignError(str(error)) from error


def _normalize_sha256(value: str, *, where: str) -> str:
    try:
        return base_campaign._normalize_sha256(value, where=where)  # noqa: SLF001
    except base_campaign.CampaignError as error:
        raise CampaignError(str(error)) from error


def _checkpoint_schedule(optimizer_steps: int) -> tuple[int, ...]:
    if optimizer_steps not in CHECKPOINT_FRONTIER:
        raise CampaignError(
            "Stage-A selected optimizer step is outside its measured frontier"
        )
    return tuple(step for step in CHECKPOINT_FRONTIER if step <= optimizer_steps)


def _checkpoint_schedule_from_frontier(
    optimizer_steps: int, frontier: Sequence[object]
) -> tuple[int, ...]:
    """Project a signed recovery trajectory through its selected checkpoint."""

    try:
        normalized = tuple(int(step) for step in frontier)
    except (TypeError, ValueError) as error:
        raise CampaignError("recovery checkpoint frontier is malformed") from error
    if (
        not normalized
        or any(step <= 0 for step in normalized)
        or tuple(sorted(set(normalized))) != normalized
        or optimizer_steps not in normalized
    ):
        raise CampaignError("recovery checkpoint frontier is malformed")
    return tuple(step for step in normalized if step <= optimizer_steps)


def _selected_dose(
    *,
    selection_path: Path,
    selection: Mapping[str, Any],
    campaign_path: Path,
    campaign: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate and project the one Stage-A dose Stage B may use."""

    campaign_ref = selection.get("campaign")
    winner = selection.get("winner")
    winner_recipe = selection.get("winner_recipe")
    winner_candidate = selection.get("winner_candidate")
    winner_checkpoint = selection.get("winner_checkpoint")
    source_arms = campaign.get("arms")
    if (
        not isinstance(campaign_ref, Mapping)
        or campaign_ref.get("file_sha256") != _file_sha256(campaign_path)
        or campaign_ref.get("campaign_sha256") != campaign.get("campaign_sha256")
        or not isinstance(winner, str)
        or winner not in stage_a.ARMS
        or not isinstance(source_arms, Mapping)
        or not isinstance(source_arms.get(winner), Mapping)
        or not isinstance(winner_recipe, Mapping)
        or not isinstance(winner_candidate, Mapping)
        or not isinstance(winner_checkpoint, Mapping)
        or selection.get("winner_is_diagnostic_not_promoted") is not True
        or selection.get("playing_strength_evaluation_still_required") is not True
        or selection.get("candidate_chaining") is not False
    ):
        raise CampaignError("Stage-A selection identity/diagnostic semantics drifted")

    try:
        step = int(selection["winner_step"])
        selected_recipe_step = int(winner_recipe["selected_optimizer_steps"])
        multiplier = float(winner_recipe["active_policy_branch_multiplier"])
        aux_batch = int(winner_recipe["policy_aux_active_batch_size"])
        candidate_step = int(winner_candidate["step"])
        reference_parent_kl = float(winner_candidate["parent_kl"])
        reference_trunk_relative_l2 = float(winner_candidate["trunk_relative_l2"])
        reference_teacher_gap_closure = float(winner_candidate["teacher_gap_closure"])
    except (KeyError, TypeError, ValueError) as error:
        raise CampaignError("Stage-A selected dose is malformed") from error
    _checkpoint_schedule(step)
    source_arm = source_arms[winner]
    expected_arm = stage_a.ARMS[winner]
    candidate_path = _regular_file(
        Path(str(winner_checkpoint.get("path", ""))),
        where="Stage-A selected diagnostic checkpoint",
    )
    if (
        selected_recipe_step != step
        or candidate_step != step
        or winner_candidate.get("arm") != winner
        or winner_candidate.get("eligible") is not True
        or winner_candidate.get("checkpoint") != str(candidate_path)
        or winner_candidate.get("checkpoint_sha256") != _file_sha256(candidate_path)
        or winner_checkpoint.get("sha256") != _file_sha256(candidate_path)
        or multiplier != float(expected_arm["active_policy_branch_multiplier"])
        or aux_batch != int(expected_arm["policy_aux_active_batch_size"])
        or multiplier
        != float(source_arm.get("active_policy_branch_multiplier", math.nan))
        or aux_batch != int(source_arm.get("policy_aux_active_batch_size", -1))
        or not math.isfinite(reference_parent_kl)
        or reference_parent_kl <= 0.0
        or not math.isfinite(reference_trunk_relative_l2)
        or reference_trunk_relative_l2 <= 0.0
        or not math.isfinite(reference_teacher_gap_closure)
        or reference_teacher_gap_closure <= 0.0
    ):
        raise CampaignError("Stage-A winner recipe/checkpoint dose binding drifted")
    return {
        "authority_kind": "stage_a_selection",
        "selection_authority": {
            "path": str(selection_path),
            "file_sha256": _file_sha256(selection_path),
            "selection_sha256": selection["selection_sha256"],
        },
        "source_campaign": {
            "path": str(campaign_path),
            "file_sha256": _file_sha256(campaign_path),
            "campaign_sha256": campaign["campaign_sha256"],
        },
        "stage_a_selection": {
            "path": str(selection_path),
            "file_sha256": _file_sha256(selection_path),
            "selection_sha256": selection["selection_sha256"],
        },
        "stage_a_campaign": {
            "path": str(campaign_path),
            "file_sha256": _file_sha256(campaign_path),
            "campaign_sha256": campaign["campaign_sha256"],
        },
        "selected_arm": winner,
        "active_policy_branch_multiplier": multiplier,
        "policy_aux_active_batch_size": aux_batch,
        "optimizer_steps": step,
        "checkpoint_steps": list(_checkpoint_schedule(step)),
        "expected_aux_active_row_draws": aux_batch * WORLD_SIZE * step,
        "reference_parent_kl": reference_parent_kl,
        "reference_trunk_relative_l2": reference_trunk_relative_l2,
        "reference_teacher_gap_closure": reference_teacher_gap_closure,
        "trust_contract": {
            "policy_kl_anchor_direction": "forward",
            "policy_kl_target": reference_parent_kl,
            "policy_kl_dual_lr": TRUST_DUAL_LR,
            "policy_kl_max_weight": TRUST_MAX_WEIGHT,
        },
        "stage_a_selected_checkpoint": {
            "path": str(candidate_path),
            "sha256": _file_sha256(candidate_path),
            "role": "dose_evidence_only_never_initializer",
        },
        "selected_checkpoint": {
            "path": str(candidate_path),
            "sha256": _file_sha256(candidate_path),
            "role": "dose_evidence_only_never_initializer",
        },
    }


def _load_stage_a_selection(
    path: Path,
) -> tuple[Path, dict[str, Any], Path, dict[str, Any], dict[str, Any]]:
    selection_path, selection = _load_signed(
        path,
        where="Stage-A active-policy selection",
        schema=stage_a.SELECTION_SCHEMA,
        digest_field="selection_sha256",
    )
    campaign_ref = selection.get("campaign")
    if not isinstance(campaign_ref, Mapping):
        raise CampaignError("Stage-A selection lost its campaign reference")
    campaign_path = _regular_file(
        Path(str(campaign_ref.get("path", ""))), where="Stage-A campaign"
    )
    try:
        loaded_path, campaign = stage_a._load_campaign(campaign_path)  # noqa: SLF001
    except stage_a.CampaignError as error:
        raise CampaignError(f"Stage-A campaign refused: {error}") from error
    dose = _selected_dose(
        selection_path=selection_path,
        selection=selection,
        campaign_path=loaded_path,
        campaign=campaign,
    )
    return selection_path, selection, loaded_path, campaign, dose


def _replay_stage_a_refusal(
    evidence: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], dict[str, Path]]:
    """Authenticate the failed Stage-A selector that motivated recovery."""

    campaign_ref = evidence.get("campaign")
    fingerprint_refs = evidence.get("fingerprints")
    if (
        not isinstance(campaign_ref, Mapping)
        or not isinstance(fingerprint_refs, Mapping)
        or set(fingerprint_refs) != set(stage_a.ARMS)
        or evidence.get("formal_result") != STAGE_A_FORMAL_REFUSAL
    ):
        raise CampaignError("recovery Stage-A refusal evidence is malformed")
    campaign_path = _regular_file(
        Path(str(campaign_ref.get("path", ""))),
        where="recovery source Stage-A campaign",
    )
    try:
        loaded_path, campaign = stage_a._load_campaign(campaign_path)  # noqa: SLF001
    except stage_a.CampaignError as error:
        raise CampaignError(
            f"recovery source Stage-A campaign refused: {error}"
        ) from error
    if campaign_ref.get("file_sha256") != _file_sha256(loaded_path) or campaign_ref.get(
        "campaign_sha256"
    ) != campaign.get("campaign_sha256"):
        raise CampaignError("recovery source Stage-A campaign binding drifted")

    bindings: dict[str, Path] = {}
    for arm in stage_a.ARMS:
        ref = fingerprint_refs[arm]
        if not isinstance(ref, Mapping):
            raise CampaignError(
                f"recovery source Stage-A {arm} fingerprint is malformed"
            )
        path, fingerprint = _load_signed(
            Path(str(ref.get("path", ""))),
            where=f"recovery source Stage-A {arm} fingerprint",
            schema=stage_a.FINGERPRINT_SCHEMA,
            digest_field="fingerprint_sha256",
        )
        checkpoints = fingerprint.get("checkpoints")
        checkpoint_sha256_by_step = (
            {
                str(int(row["step"])): str(row["checkpoint_sha256"])
                for row in checkpoints
            }
            if isinstance(checkpoints, list)
            and all(isinstance(row, Mapping) for row in checkpoints)
            else None
        )
        if (
            ref.get("file_sha256") != _file_sha256(path)
            or ref.get("fingerprint_sha256") != fingerprint.get("fingerprint_sha256")
            or ref.get("checkpoint_sha256_by_step") != checkpoint_sha256_by_step
        ):
            raise CampaignError(
                f"recovery source Stage-A {arm} fingerprint binding drifted"
            )
        bindings[arm] = path

    try:
        stage_a._select(loaded_path, campaign, bindings)  # noqa: SLF001
    except stage_a.CampaignError as error:
        if str(error) != STAGE_A_NO_WINNER_REASON:
            raise CampaignError(
                f"recovery source Stage-A selector failed differently: {error}"
            ) from error
    else:
        raise CampaignError(
            "recovery selection is forbidden because Stage A now has an eligible winner"
        )
    return loaded_path, campaign, bindings


def _recovery_arm_receipt(
    plan: Mapping[str, Any], arm: str, fingerprint: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]]:
    arms = plan.get("arms")
    if not isinstance(arms, Mapping) or not isinstance(arms.get(arm), Mapping):
        raise CampaignError(f"recovery plan lost arm {arm}")
    command = arms[arm].get("command")
    if not isinstance(command, list) or not all(
        isinstance(token, str) for token in command
    ):
        raise CampaignError(f"recovery arm {arm} command is malformed")
    recipe = arms[arm].get("recipe_overrides")
    if not isinstance(recipe, Mapping):
        raise CampaignError(f"recovery arm {arm} recipe is malformed")
    try:
        rendered_recipe = json.loads(
            stage_a._option(command, "--recipe-overrides-json")  # noqa: SLF001
        )
    except (json.JSONDecodeError, stage_a.CampaignError) as error:
        raise CampaignError(
            f"recovery arm {arm} rendered recipe is malformed"
        ) from error
    if rendered_recipe != recipe:
        raise CampaignError(f"recovery arm {arm} command/recipe binding drifted")
    try:
        receipt_path = _regular_file(
            Path(stage_a._option(command, "--receipt")),  # noqa: SLF001
            where=f"recovery arm {arm} one-dose receipt",
        )
        receipt = one_dose._load_authenticated_completed_aux_receipt(  # noqa: SLF001
            receipt_path
        )
    except (stage_a.CampaignError, one_dose.ExecutorError) as error:
        raise CampaignError(f"recovery arm {arm} receipt refused: {error}") from error
    lineage = receipt.get("learner_lineage_parent")
    effective = receipt.get("learner_ablation", {}).get("effective_recipe")
    rows = fingerprint.get("checkpoints")
    terminal = rows[-1] if isinstance(rows, list) and rows else None
    if (
        receipt.get("status") != "complete"
        or receipt.get("returncode") != 0
        or receipt.get("diagnostic_only") is not True
        or receipt.get("promotion_eligible") is not False
        or receipt.get("receipt_sha256") != fingerprint.get("one_dose_receipt_sha256")
        or not isinstance(effective, Mapping)
        or any(effective.get(key) != value for key, value in recipe.items())
        or not isinstance(terminal, Mapping)
        or receipt.get("outputs", {}).get("checkpoint_sha256")
        != terminal.get("checkpoint_sha256")
        or not isinstance(lineage, Mapping)
        or lineage.get("role") != "diagnostic_independent_parent"
        or lineage.get("checkpoint", {}).get("sha256")
        != stage_a.EXPECTED_F7_PARENT_SHA256
    ):
        raise CampaignError(f"recovery arm {arm} receipt lineage/dose drifted")
    return receipt_path, receipt


def _load_recovery_selection(
    path: Path,
) -> tuple[Path, dict[str, Any], Path, dict[str, Any], dict[str, Any]]:
    """Load the exact direction-corrected selection as Stage-B dose authority."""

    selection_path, selection = _load_signed(
        path,
        where="direction-corrected recovery selection",
        schema=RECOVERY_SELECTION_SCHEMA,
        digest_field="selection_sha256",
    )
    campaign_ref = selection.get("campaign")
    if not isinstance(campaign_ref, Mapping):
        raise CampaignError("recovery selection lost its campaign reference")
    plan_path, plan = _load_signed(
        Path(str(campaign_ref.get("path", ""))),
        where="direction-corrected recovery campaign",
        schema=RECOVERY_CAMPAIGN_SCHEMA,
        digest_field="campaign_sha256",
    )
    outputs = plan.get("outputs")
    if (
        campaign_ref.get("file_sha256") != _file_sha256(plan_path)
        or campaign_ref.get("campaign_sha256") != plan.get("campaign_sha256")
        or selection.get("source") != plan.get("source")
        or selection.get("lineage") != plan.get("lineage")
        or selection.get("stage_a_refusal_evidence")
        != plan.get("stage_a_refusal_evidence")
        or selection.get("selection_contract") != plan.get("selection_contract")
        or selection.get("diagnostic_only") is not True
        or selection.get("promotion_eligible") is not False
        or selection.get("playing_strength_evaluation_required") is not True
        or not isinstance(outputs, Mapping)
        or Path(str(outputs.get("selection", ""))).expanduser().resolve(strict=False)
        != selection_path
        or plan.get("selection_contract", {}).get(
            "playing_strength_evaluation_required"
        )
        is not True
    ):
        raise CampaignError("recovery selection campaign/diagnostic binding drifted")

    lineage = plan.get("lineage")
    if (
        not isinstance(lineage, Mapping)
        or lineage.get("learner_parent_sha256") != stage_a.EXPECTED_F7_PARENT_SHA256
        or lineage.get("every_arm_restarts_from_exact_upgraded_f7") is not True
        or lineage.get("fresh_adam_every_arm") is not True
        or lineage.get("candidate_chaining_forbidden") is not True
    ):
        raise CampaignError("recovery selection lineage is not fresh f7/fresh Adam")

    operator = plan.get("operator")
    source_ref = plan.get("source")
    fixed = plan.get("fixed_surface")
    if (
        not isinstance(operator, Mapping)
        or not isinstance(source_ref, Mapping)
        or not isinstance(fixed, Mapping)
    ):
        raise CampaignError("recovery campaign input surface is malformed")
    operator_path = _regular_file(
        Path(str(operator.get("path", ""))), where="recovery campaign operator"
    )
    source_root = (
        Path(str(source_ref.get("path", ""))).expanduser().resolve(strict=True)
    )
    if not source_root.is_dir():
        raise CampaignError("recovery campaign source root is not a directory")
    if (
        operator.get("file_sha256") != _file_sha256(operator_path)
        or source_ref.get("one_dose_trainer_sha256")
        != _file_sha256(
            _regular_file(
                source_root / "tools/a1_one_dose_train.py", where="recovery trainer"
            )
        )
        or source_ref.get("train_bc_sha256")
        != _file_sha256(
            _regular_file(source_root / "tools/train_bc.py", where="recovery train_bc")
        )
    ):
        raise CampaignError("recovery campaign code bytes drifted")
    for path_key, digest_key, label in (
        ("validation", "validation_file_sha256", "validation manifest"),
        ("admission", "admission_file_sha256", "coherent admission"),
        (
            "architecture_upgrade_receipt",
            "architecture_upgrade_receipt_file_sha256",
            "architecture upgrade receipt",
        ),
    ):
        artifact = _regular_file(
            Path(str(fixed.get(path_key, ""))), where=f"recovery {label}"
        )
        if fixed.get(digest_key) != _file_sha256(artifact):
            raise CampaignError(f"recovery {label} bytes drifted")
    data = Path(str(fixed.get("data", ""))).expanduser().resolve(strict=True)
    if (
        not data.is_dir()
        or fixed.get("corpus_meta_file_sha256")
        != _file_sha256(data / "corpus_meta.json")
        or plan.get("diagnostic_only") is not True
        or plan.get("promotion_eligible") is not False
        or plan.get("fixed_surface", {}).get("topology")
        != {
            "world_size": WORLD_SIZE,
            "local_batch_size": LOCAL_BATCH_SIZE,
            "global_batch_size": GLOBAL_BATCH_SIZE,
        }
    ):
        raise CampaignError("recovery corpus/topology binding drifted")

    stage_a_path, stage_a_campaign, _bindings = _replay_stage_a_refusal(
        selection["stage_a_refusal_evidence"]
    )
    arms = plan.get("arms")
    refs = selection.get("fingerprints")
    trajectory = plan.get("trajectory")
    contract = plan.get("selection_contract")
    if (
        not isinstance(arms, Mapping)
        or not arms
        or not isinstance(refs, Mapping)
        or set(refs) != set(arms)
        or not isinstance(trajectory, Mapping)
        or not isinstance(contract, Mapping)
    ):
        raise CampaignError("recovery selection arm/trajectory surface is malformed")
    try:
        frontier = tuple(int(step) for step in trajectory["checkpoint_steps"])
        terminal = int(trajectory["terminal_step"])
        parent_kl_cap = float(contract["parent_kl_max"])
        trunk_cap = float(contract["trunk_relative_l2_max"])
    except (KeyError, TypeError, ValueError) as error:
        raise CampaignError("recovery selection contract is malformed") from error
    _checkpoint_schedule_from_frontier(terminal, frontier)
    if (
        terminal != frontier[-1]
        or not all(
            math.isfinite(value) and value > 0.0 for value in (parent_kl_cap, trunk_cap)
        )
        or contract.get("teacher_gap_closure_ranking_authority") is not False
        or contract.get("teacher_gap_closure_admission_authority") is not False
        or contract.get("paired_playing_strength_is_final_authority") is not True
        or contract.get("objective")
        != "minimum_update_within_frozen_trust_and_trunk_budgets"
    ):
        raise CampaignError("recovery selection contract/trajectory drifted")

    candidates: list[dict[str, Any]] = []
    arm_receipts: dict[str, dict[str, Any]] = {}
    for arm in arms:
        arm_plan = arms[arm]
        ref = refs[arm]
        if not isinstance(arm_plan, Mapping) or not isinstance(ref, Mapping):
            raise CampaignError(f"recovery arm {arm} authority is malformed")
        fingerprint_path, fingerprint = _load_signed(
            Path(str(ref.get("path", ""))),
            where=f"recovery arm {arm} fingerprint",
            schema=RECOVERY_FINGERPRINT_SCHEMA,
            digest_field="fingerprint_sha256",
        )
        rows = fingerprint.get("checkpoints")
        recipe = arm_plan.get("recipe_overrides")
        if (
            ref.get("file_sha256") != _file_sha256(fingerprint_path)
            or ref.get("fingerprint_sha256") != fingerprint.get("fingerprint_sha256")
            or fingerprint.get("campaign_sha256") != plan.get("campaign_sha256")
            or fingerprint.get("arm") != arm
            or fingerprint.get("recipe_overrides") != recipe
            or fingerprint.get("diagnostic_only") is not True
            or fingerprint.get("promotion_eligible") is not False
            or not isinstance(rows, list)
            or [row.get("step") for row in rows] != list(frontier)
        ):
            raise CampaignError(f"recovery arm {arm} fingerprint binding drifted")
        receipt_path, receipt = _recovery_arm_receipt(plan, arm, fingerprint)
        arm_receipts[arm] = {
            "path": str(receipt_path),
            "file_sha256": _file_sha256(receipt_path),
            "receipt_sha256": receipt["receipt_sha256"],
        }
        for row in rows:
            if not isinstance(row, Mapping):
                raise CampaignError(f"recovery arm {arm} fingerprint row is malformed")
            try:
                step = int(row["step"])
                checkpoint = _regular_file(
                    Path(str(row["checkpoint"])),
                    where=f"recovery arm {arm} step {step} checkpoint",
                )
                parent_kl = float(row["parent_kl"])
                closure = float(row["teacher_gap_closure"])
                trunk = float(row["trunk_relative_l2"])
            except (KeyError, TypeError, ValueError) as error:
                raise CampaignError(
                    f"recovery arm {arm} fingerprint row is malformed"
                ) from error
            functional_path = _regular_file(
                fingerprint_path.parent / f"step{step:04d}.functional.json",
                where=f"recovery arm {arm} step {step} functional fingerprint",
            )
            drift_path = _regular_file(
                fingerprint_path.parent / f"step{step:04d}.drift.json",
                where=f"recovery arm {arm} step {step} layer drift",
            )
            _functional_path, functional = _load_json(
                functional_path,
                where=f"recovery arm {arm} step {step} functional fingerprint",
            )
            _drift_path, drift = _load_json(
                drift_path, where=f"recovery arm {arm} step {step} layer drift"
            )
            functional_dose = functional.get("functional_dose_fingerprint")
            try:
                measured_parent_kl = float(functional_dose["kl_parent_candidate_mean"])
                measured_closure = float(
                    functional["teacher_gap"]["active_policy_teacher_gap_closure"]
                )
                measured_trunk = stage_a._trunk_relative_l2(drift)  # noqa: SLF001
            except (KeyError, TypeError, ValueError, stage_a.CampaignError) as error:
                raise CampaignError(
                    f"recovery arm {arm} step {step} metrics are malformed"
                ) from error
            if (
                row.get("checkpoint_sha256") != _file_sha256(checkpoint)
                or row.get("functional_file_sha256") != _file_sha256(functional_path)
                or row.get("drift_file_sha256") != _file_sha256(drift_path)
                or not isinstance(functional_dose, Mapping)
                or measured_parent_kl != parent_kl
                or measured_closure != closure
                or measured_trunk != trunk
                or functional.get("inputs", {}).get("checkpoint", {}).get("sha256")
                != _file_sha256(checkpoint)
                or drift.get("candidate", {}).get("sha256") != _file_sha256(checkpoint)
                or not all(
                    math.isfinite(value) for value in (parent_kl, closure, trunk)
                )
                or parent_kl < 0.0
                or trunk < 0.0
            ):
                raise CampaignError(
                    f"recovery arm {arm} step {step} fingerprint drifted"
                )
            candidate = {
                "arm": arm,
                **copy.deepcopy(dict(row)),
                "teacher_gap_closure_diagnostic_only": True,
                "eligible": parent_kl <= parent_kl_cap and trunk <= trunk_cap,
            }
            candidates.append(candidate)

    claimed_candidates = selection.get("checkpoint_candidates")
    expected_by_key = {
        (str(candidate["arm"]), int(candidate["step"])): candidate
        for candidate in candidates
    }
    claimed_by_key = (
        {
            (str(candidate.get("arm")), int(candidate.get("step", -1))): candidate
            for candidate in claimed_candidates
        }
        if isinstance(claimed_candidates, list)
        and all(isinstance(candidate, Mapping) for candidate in claimed_candidates)
        else {}
    )
    if (
        not isinstance(claimed_candidates, list)
        or len(claimed_candidates) != len(claimed_by_key)
        or claimed_by_key != expected_by_key
    ):
        raise CampaignError("recovery checkpoint candidate table did not replay")
    candidates = [copy.deepcopy(dict(candidate)) for candidate in claimed_candidates]
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    if not eligible:
        raise CampaignError("recovery selection contains no eligible checkpoint")
    winner = sorted(
        eligible,
        key=lambda row: (
            int(row["step"]),
            float(row["parent_kl"]),
            float(row["trunk_relative_l2"]),
            str(row["arm"]),
        ),
    )[0]
    if selection.get("winner") != winner:
        raise CampaignError("recovery winner does not replay from its frozen contract")
    winner_arm = str(winner["arm"])
    winner_recipe = arms[winner_arm].get("recipe_overrides")
    if not isinstance(winner_recipe, Mapping):
        raise CampaignError("recovery winner recipe is malformed")
    try:
        step = int(winner["step"])
        aux_batch = int(winner_recipe["policy_aux_active_batch_size"])
        aux_loss_weight = float(winner_recipe["policy_aux_loss_weight"])
        lr = float(winner_recipe["lr"])
        warmup = int(winner_recipe["lr_warmup_steps"])
        reference_parent_kl = float(winner["parent_kl"])
        reference_trunk = float(winner["trunk_relative_l2"])
        reference_closure = float(winner["teacher_gap_closure"])
    except (KeyError, TypeError, ValueError) as error:
        raise CampaignError("recovery selected dose is malformed") from error
    schedule = _checkpoint_schedule_from_frontier(step, frontier)
    matching_stage_a_arms = [
        name
        for name, recipe in stage_a.ARMS.items()
        if int(recipe["policy_aux_active_batch_size"]) == aux_batch
        and math.isclose(
            float(recipe["policy_aux_loss_weight"]),
            aux_loss_weight,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ]
    checkpoint = _regular_file(
        Path(str(winner["checkpoint"])), where="recovery selected checkpoint"
    )
    if (
        len(matching_stage_a_arms) != 1
        or not math.isfinite(aux_loss_weight)
        or aux_loss_weight <= 0.0
        or not math.isfinite(lr)
        or lr <= 0.0
        or warmup < 0
        or winner.get("checkpoint_sha256") != _file_sha256(checkpoint)
        or reference_parent_kl <= 0.0
            or reference_parent_kl > parent_kl_cap
            or reference_trunk <= 0.0
            or reference_trunk > trunk_cap
            or not math.isfinite(reference_closure)
        ):
        raise CampaignError("recovery selected checkpoint/dose binding drifted")
    stage_a_arm = matching_stage_a_arms[0]
    selected_fingerprint_ref = refs[winner_arm]
    trust_keys = {
        "policy_kl_anchor_direction",
        "policy_kl_target",
        "policy_kl_dual_lr",
        "policy_kl_max_weight",
    }
    present_trust_keys = trust_keys & set(winner_recipe)
    if present_trust_keys and present_trust_keys != trust_keys:
        raise CampaignError("recovery winner carries a partial trust contract")
    trust_contract = (
        {key: copy.deepcopy(winner_recipe[key]) for key in sorted(trust_keys)}
        if present_trust_keys
        else {
            "policy_kl_anchor_direction": "forward",
            "policy_kl_target": reference_parent_kl,
            "policy_kl_dual_lr": TRUST_DUAL_LR,
            "policy_kl_max_weight": TRUST_MAX_WEIGHT,
        }
    )
    try:
        trust_target = float(trust_contract["policy_kl_target"])
        trust_dual_lr = float(trust_contract["policy_kl_dual_lr"])
        trust_max_weight = float(trust_contract["policy_kl_max_weight"])
    except (KeyError, TypeError, ValueError) as error:
        raise CampaignError("recovery winner trust contract is malformed") from error
    if (
        trust_contract.get("policy_kl_anchor_direction") != "forward"
        or not math.isfinite(trust_target)
        or trust_target <= 0.0
        or not math.isfinite(trust_dual_lr)
        or trust_dual_lr <= 0.0
        or not math.isfinite(trust_max_weight)
        or trust_max_weight <= 0.0
    ):
        raise CampaignError("recovery winner trust contract is malformed")
    return (
        selection_path,
        selection,
        stage_a_path,
        stage_a_campaign,
        {
            "authority_kind": "direction_corrected_recovery_selection",
            "selection_authority": {
                "path": str(selection_path),
                "file_sha256": _file_sha256(selection_path),
                "selection_sha256": selection["selection_sha256"],
            },
            "source_campaign": {
                "path": str(stage_a_path),
                "file_sha256": _file_sha256(stage_a_path),
                "campaign_sha256": stage_a_campaign["campaign_sha256"],
            },
            "recovery_campaign": {
                "path": str(plan_path),
                "file_sha256": _file_sha256(plan_path),
                "campaign_sha256": plan["campaign_sha256"],
            },
            "recovery_fingerprint": copy.deepcopy(dict(selected_fingerprint_ref)),
            "recovery_receipt": copy.deepcopy(arm_receipts[winner_arm]),
            "selected_arm": winner_arm,
            "active_policy_branch_multiplier": float(
                stage_a.ARMS[stage_a_arm]["active_policy_branch_multiplier"]
            ),
            "policy_aux_active_batch_size": aux_batch,
            "optimizer_steps": step,
            "checkpoint_steps": list(schedule),
            "expected_aux_active_row_draws": aux_batch * WORLD_SIZE * step,
            "reference_parent_kl": reference_parent_kl,
            "reference_trunk_relative_l2": reference_trunk,
            "reference_teacher_gap_closure": reference_closure,
            "trust_contract": trust_contract,
            "selected_recipe_overrides": copy.deepcopy(dict(winner_recipe)),
            "selected_checkpoint": {
                "path": str(checkpoint),
                "sha256": _file_sha256(checkpoint),
                "role": "dose_evidence_only_never_initializer",
            },
        },
    )


def load_recovery_selected_dose(path: Path) -> dict[str, Any]:
    """Return the authenticated stable dose projection for eval/orchestration."""

    _path, _selection, _stage_a_path, _stage_a, dose = _load_recovery_selection(path)
    return copy.deepcopy(dose)


def _treatment_exposure(data_path: Path) -> dict[str, Any]:
    """Measure the exact rows the typed forced-value treatment can change."""

    try:
        corpus = train_bc.MemmapCorpus(data_path)
        legal_counts = np.asarray(
            corpus["legal_action_ids"].row_counts(), dtype=np.int64
        )
        forced = legal_counts == 1
        stored_forced = (
            np.asarray(corpus["is_forced"], dtype=np.bool_)
            if "is_forced" in corpus
            else np.zeros(len(corpus), dtype=np.bool_)
        )
        env_config = train_bc.parse_track(
            "2p_no_trade", vps_to_win=10, use_graph_history_features=True
        )
        catalog = train_bc._action_catalog_for_env_config(env_config)  # noqa: SLF001
        action_types = tuple(
            str(catalog.describe(action_id)["action_type"]).upper()
            for action_id in range(int(catalog.size))
        )
        actions = np.asarray(corpus["action_taken"], dtype=np.int64)
    except (KeyError, OSError, SystemExit, ValueError) as error:
        raise CampaignError(
            f"cannot measure Stage-B treatment exposure: {error}"
        ) from error
    if (
        legal_counts.shape != (len(corpus),)
        or stored_forced.shape != (len(corpus),)
        or actions.shape != (len(corpus),)
        or np.any(actions[forced] < 0)
        or np.any(actions[forced] >= len(action_types))
    ):
        raise CampaignError("forced-row treatment exposure columns are malformed")
    typed_counts = {"END_TURN": 0, "ROLL": 0}
    for action_id in actions[forced].tolist():
        action_type = action_types[int(action_id)]
        if action_type in typed_counts:
            typed_counts[action_type] += 1
    typed_rows = sum(typed_counts.values())
    anchor_rows = 0
    if "prior_policy" in corpus:
        for start in range(0, len(corpus), 8192):
            rows = np.arange(start, min(start + 8192, len(corpus)), dtype=np.int64)
            prior = np.asarray(corpus["prior_policy"][rows], dtype=np.float32)
            counts = legal_counts[rows]
            if prior.ndim != 2 or prior.shape[0] != len(rows):
                raise CampaignError("policy-KL prior surface has malformed shape")
            valid = np.arange(prior.shape[1])[None, :] < counts[:, None]
            mass = np.where(valid, prior, 0.0).sum(axis=1)
            anchor_rows += int(np.count_nonzero((counts > 1) & (mass > 1.0e-6)))
    payload: dict[str, Any] = {
        "schema_version": "a1-stage-b-treatment-exposure-v2",
        "corpus": {
            "path": str(data_path),
            "corpus_meta_file_sha256": _file_sha256(data_path / "corpus_meta.json"),
        },
        "row_count": len(corpus),
        "stored_is_forced_rows": int(np.count_nonzero(stored_forced)),
        "one_legal_action_rows": int(np.count_nonzero(forced)),
        "typed_forced_rows": int(typed_rows),
        "typed_forced_rows_by_action_type": typed_counts,
        "forced_treatment_structurally_active": typed_rows > 0,
        "policy_kl_anchor_multi_action_rows": anchor_rows,
        "trust_treatment_structurally_active": anchor_rows > 0,
        "weighting_semantics": (
            "train_bc legal_action_count==1 joined through authoritative "
            "ActionCatalog action_taken type"
        ),
    }
    payload["exposure_sha256"] = _value_sha256(payload)
    return payload


def _value_trunk_treatment_exposure(
    source_recipe: Mapping[str, Any], *, value_attention_pool: bool
) -> dict[str, Any]:
    """Prove the value-boundary treatment can change an active objective."""

    try:
        scalar_weight = float(source_recipe["value_loss_weight"])
    except (KeyError, TypeError, ValueError) as error:
        raise CampaignError(
            f"cannot bind Stage-B value-trunk treatment exposure: {error}"
        ) from error
    if not math.isfinite(scalar_weight) or scalar_weight < 0.0:
        raise CampaignError("Stage-B scalar value objective weight is malformed")
    # EntityGraphNet routes both the CLS readout and the optional attention-pool
    # readout through the same scalar-value gradient boundary. Keep the flag in
    # the receipt as architecture evidence, but it no longer invalidates the
    # treatment exposure.
    active = scalar_weight > 0.0
    return {
        "value_objective_contract": "scalar_mse",
        "scalar_value_loss_weight": scalar_weight,
        "value_attention_pool": bool(value_attention_pool),
        "value_trunk_treatment_structurally_active": active,
        "value_trunk_treatment_inactive_reason": (
            None if active else "zero_scalar_mse_value_objective"
        ),
    }


def _active_arms_for_exposure(exposure: Mapping[str, Any]) -> list[str]:
    typed_rows = int(exposure.get("typed_forced_rows", -1))
    stated_active = exposure.get("forced_treatment_structurally_active")
    if typed_rows < 0 or stated_active is not (typed_rows > 0):
        raise CampaignError("forced treatment exposure count/status is inconsistent")
    anchor_rows = int(exposure.get("policy_kl_anchor_multi_action_rows", -1))
    trust_active = exposure.get("trust_treatment_structurally_active")
    if anchor_rows < 0 or trust_active is not (anchor_rows > 0):
        raise CampaignError("trust treatment exposure count/status is inconsistent")
    value_trunk_active = exposure.get("value_trunk_treatment_structurally_active")
    if type(value_trunk_active) is not bool:
        raise CampaignError("value-trunk treatment exposure status is missing")
    return [
        arm
        for arm in ARM_ORDER
        if (arm != "FORCED" or stated_active)
        and (arm != "TRUST" or trust_active)
        and (arm != "VTRUNK25" or value_trunk_active)
    ]


def _arm_overrides(
    arm: str, *, selected_dose: Mapping[str, Any], source_recipe: Mapping[str, Any]
) -> dict[str, Any]:
    if arm not in TREATMENTS:
        raise CampaignError(f"unknown Stage-B arm {arm!r}")
    treatment = TREATMENTS[arm]
    overrides: dict[str, Any] = {
        "epochs": 1,
        "max_steps": int(selected_dose["optimizer_steps"]),
        "lr": float(source_recipe["lr"]),
        "lr_warmup_steps": int(source_recipe["lr_warmup_steps"]),
        "forced_row_value_weight": 1.0,
        "public_card_lr_mult": float(treatment["public_card_lr_mult"]),
        "trunk_lr_mult": float(treatment["trunk_lr_mult"]),
        "per_game_policy_surprise_weighting": bool(
            treatment["per_game_policy_surprise_weighting"]
        ),
        "policy_aux_active_batch_size": int(
            selected_dose["policy_aux_active_batch_size"]
        ),
    }
    value_trunk_scale = float(treatment.get("value_trunk_grad_scale", 1.0))
    if value_trunk_scale != 1.0:
        overrides["value_trunk_grad_scale"] = value_trunk_scale
    typed = str(treatment["forced_row_value_action_type_weights"])
    if typed:
        overrides["forced_row_value_action_type_weights"] = typed
    if arm == "TRUST":
        contract = selected_dose.get("trust_contract")
        if not isinstance(contract, Mapping):
            contract = {
                "policy_kl_anchor_direction": "forward",
                "policy_kl_target": float(selected_dose["reference_parent_kl"]),
                "policy_kl_dual_lr": TRUST_DUAL_LR,
                "policy_kl_max_weight": TRUST_MAX_WEIGHT,
            }
        try:
            target = float(contract["policy_kl_target"])
            dual_lr = float(contract["policy_kl_dual_lr"])
            max_weight = float(contract["policy_kl_max_weight"])
        except (KeyError, TypeError, ValueError) as error:
            raise CampaignError("TRUST contract is malformed") from error
        if (
            contract.get("policy_kl_anchor_direction") != "forward"
            or not math.isfinite(target)
            or target <= 0.0
            or not math.isfinite(dual_lr)
            or dual_lr <= 0.0
            or not math.isfinite(max_weight)
            or max_weight <= 0.0
        ):
            raise CampaignError("TRUST contract is malformed")
        overrides.update(copy.deepcopy(dict(contract)))
    return overrides


def _assert_treatment_isolation(arms: Mapping[str, Mapping[str, Any]]) -> None:
    if set(arms) != set(ARM_ORDER):
        raise CampaignError("Stage-B treatment matrix is incomplete")
    recipes = {
        arm: dict(record.get("recipe_overrides", {})) for arm, record in arms.items()
    }
    common = {
        arm: {
            key: value for key, value in recipe.items() if key not in TREATMENT_FIELDS
        }
        for arm, recipe in recipes.items()
    }
    if any(value != common["BASE"] for value in common.values()):
        raise CampaignError("Stage-B arms differ outside declared treatment fields")
    expected_deltas = {
        "FORCED": {"forced_row_value_action_type_weights"},
        "CARD4": {"public_card_lr_mult"},
        "SURPRISE": {"per_game_policy_surprise_weighting"},
        "TRUNK25": {"trunk_lr_mult"},
        "TRUNK10": {"trunk_lr_mult"},
        "VTRUNK25": {"value_trunk_grad_scale"},
        "TRUST": {
            "policy_kl_anchor_direction",
            "policy_kl_target",
            "policy_kl_dual_lr",
            "policy_kl_max_weight",
        },
    }
    base = recipes["BASE"]
    for arm, expected in expected_deltas.items():
        delta = {
            key
            for key in set(base) | set(recipes[arm])
            if base.get(key) != recipes[arm].get(key)
        }
        if delta != expected:
            raise CampaignError(
                f"Stage-B arm {arm} treatment delta {sorted(delta)} != {sorted(expected)}"
            )
    if (
        base.get("public_card_lr_mult") != 1.0
        or base.get("per_game_policy_surprise_weighting") is not False
        or base.get("forced_row_value_weight") != 1.0
        or base.get("forced_row_value_action_type_weights")
        or recipes["FORCED"].get("forced_row_value_action_type_weights")
        != FORCED_TYPED_SPEC
        or recipes["CARD4"].get("public_card_lr_mult") != 4.0
        or recipes["SURPRISE"].get("per_game_policy_surprise_weighting") is not True
        or recipes["TRUNK25"].get("trunk_lr_mult") != 0.25
        or recipes["TRUNK10"].get("trunk_lr_mult") != 0.10
        or base.get("value_trunk_grad_scale", 1.0) != 1.0
        or recipes["VTRUNK25"].get("value_trunk_grad_scale") != 0.25
        or recipes["TRUST"].get("policy_kl_anchor_direction") != "forward"
        or recipes["TRUST"].get("policy_kl_dual_lr") != TRUST_DUAL_LR
        or recipes["TRUST"].get("policy_kl_max_weight") != TRUST_MAX_WEIGHT
    ):
        raise CampaignError("Stage-B treatment definitions drifted")


def _verify_source_campaign_inputs(campaign: Mapping[str, Any]) -> None:
    inputs = campaign.get("inputs")
    if not isinstance(inputs, Mapping):
        raise CampaignError("Stage-A campaign has no immutable input binding")
    for path_key, digest_key in (
        ("lock", "lock_file_sha256"),
        ("validation_manifest", "validation_manifest_file_sha256"),
        ("coherent_corpus_receipt", "coherent_corpus_receipt_file_sha256"),
        ("architecture_upgrade_receipt", "architecture_upgrade_receipt_file_sha256"),
        ("independent_parent_authority", "independent_parent_authority_file_sha256"),
    ):
        path = _regular_file(Path(str(inputs[path_key])), where=f"Stage-A {path_key}")
        if _file_sha256(path) != inputs[digest_key]:
            raise CampaignError(f"Stage-A immutable input changed: {path_key}")
    data = Path(str(inputs["data"])).expanduser().resolve(strict=True)
    if (
        not data.is_dir()
        or _file_sha256(data / "corpus_meta.json") != inputs["corpus_meta_file_sha256"]
    ):
        raise CampaignError("Stage-A coherent corpus changed after selection")
    try:
        stage_a._load_admission(  # noqa: SLF001
            Path(str(inputs["coherent_corpus_receipt"]))
        )
    except stage_a.CampaignError as error:
        raise CampaignError(f"Stage-A coherent admission refused: {error}") from error


def _plan(args: argparse.Namespace) -> dict[str, Any]:
    if args.recovery_selection is not None:
        selection_path, selection, source_path, source, dose = _load_recovery_selection(
            args.recovery_selection
        )
    else:
        selection_path, selection, source_path, source, dose = _load_stage_a_selection(
            args.stage_a_selection
        )
    _verify_source_campaign_inputs(source)
    source_inputs = source["inputs"]
    canonical_source_recipe = source.get("canonical_learner", {}).get("training_recipe")
    if not isinstance(canonical_source_recipe, Mapping):
        raise CampaignError("Stage-A campaign lost its canonical learner recipe")
    source_recipe = copy.deepcopy(dict(canonical_source_recipe))
    selected_recipe = dose.get("selected_recipe_overrides")
    if isinstance(selected_recipe, Mapping):
        try:
            source_recipe["lr"] = float(selected_recipe["lr"])
            source_recipe["lr_warmup_steps"] = int(selected_recipe["lr_warmup_steps"])
        except (KeyError, TypeError, ValueError) as error:
            raise CampaignError(
                "recovery selected common recipe is malformed"
            ) from error
    data = Path(str(source_inputs["data"])).resolve(strict=True)

    trainer = _regular_file(
        args.one_dose_trainer or (REPO_ROOT / "tools" / "a1_one_dose_train.py"),
        where="Stage-B one-dose trainer",
    )
    python = base_campaign._python_executable(args.python)  # noqa: SLF001
    canary = _regular_file(args.ddp_canary_receipt, where="Stage-B 8xB200 canary")
    reviewed_code = _normalize_sha256(
        args.reviewed_code_tree_sha256, where="Stage-B reviewed code tree"
    )
    lock = _regular_file(Path(str(source_inputs["lock"])), where="sealed learner lock")
    reviewed_lock = str(source_inputs["reviewed_lock_file_sha256"])
    if reviewed_lock != _file_sha256(lock):
        raise CampaignError("Stage-A reviewed lock bytes changed")
    upgrade_path = _regular_file(
        Path(str(source_inputs["architecture_upgrade_receipt"])),
        where="f7 architecture upgrade receipt",
    )
    try:
        upgrade = architecture_upgrade.verify_receipt(upgrade_path)
    except architecture_upgrade.UpgradeError as error:
        raise CampaignError(f"f7 architecture upgrade refused: {error}") from error
    if (
        upgrade.get("source", {}).get("sha256") != stage_a.EXPECTED_F7_PARENT_SHA256
        or upgrade.get("module")
        != architecture_upgrade.MODULE_STRUCTURED_ACTION_VALUE_PUBLIC_CARD_COUNT_MEANINGFUL_HISTORY_V3
        or upgrade.get("forward_identical_at_init") is not True
        or upgrade.get("shared_parameters_bit_identical") is not True
    ):
        raise CampaignError(
            "Stage-B requires the exact Stage-A f7 upgraded initializer"
        )
    upgraded_initializer = _regular_file(
        Path(str(upgrade["upgraded_initializer"]["path"])),
        where="f7 upgraded initializer",
    )
    try:
        checkpoint = architecture_upgrade._load_checkpoint(  # noqa: SLF001
            upgraded_initializer
        )
        checkpoint_config = architecture_upgrade._config(  # noqa: SLF001
            checkpoint.get("config")
        )
    except architecture_upgrade.UpgradeError as error:
        raise CampaignError(
            f"cannot inspect f7 value-gradient boundary: {error}"
        ) from error
    exposure = _treatment_exposure(data)
    exposure.update(
        _value_trunk_treatment_exposure(
            source_recipe,
            value_attention_pool=bool(
                checkpoint_config.get("value_attention_pool", False)
            ),
        )
    )
    exposure.pop("exposure_sha256", None)
    exposure["exposure_sha256"] = _value_sha256(exposure)
    active_arms = _active_arms_for_exposure(exposure)

    arm_records: dict[str, Any] = {}
    for arm in ARM_ORDER:
        active = arm in active_arms
        arm_records[arm] = {
            "treatment": copy.deepcopy(TREATMENTS[arm]),
            "recipe_overrides": _arm_overrides(
                arm, selected_dose=dose, source_recipe=source_recipe
            ),
            "structurally_active": active,
            "launch_eligible": active,
            "inactive_reason": (
                None
                if active
                else (
                    "zero_END_TURN_or_ROLL_one_legal_action_rows_in_bound_corpus"
                    if arm == "FORCED"
                    else (
                        "zero_authenticated_multi_action_parent_prior_rows"
                        if arm == "TRUST"
                        else exposure["value_trunk_treatment_inactive_reason"]
                    )
                )
            ),
            "expected_aux_active_row_draws": dose["expected_aux_active_row_draws"],
            "output_subdir": f"arms/{arm}",
        }
    _assert_treatment_isolation(arm_records)

    output_root = args.output_root.expanduser().resolve(strict=False)
    dose_authority: dict[str, Any] = {
        "kind": dose["authority_kind"],
        "selection": copy.deepcopy(dose["selection_authority"]),
        "source_stage_a_campaign": copy.deepcopy(dose["source_campaign"]),
        "selected_arm": dose["selected_arm"],
        "selected_checkpoint": copy.deepcopy(dose["selected_checkpoint"]),
    }
    if dose["authority_kind"] == "direction_corrected_recovery_selection":
        dose_authority.update(
            {
                "recovery_campaign": copy.deepcopy(dose["recovery_campaign"]),
                "recovery_fingerprint": copy.deepcopy(dose["recovery_fingerprint"]),
                "recovery_receipt": copy.deepcopy(dose["recovery_receipt"]),
                "selected_recipe_overrides": copy.deepcopy(
                    dose["selected_recipe_overrides"]
                ),
                "playing_strength_evaluation_required": True,
            }
        )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA,
        "purpose": "selected_dose_single_treatment_causal_ablation",
        "diagnostic_only": True,
        "promotion_eligible": False,
        "dose_authority": dose_authority,
        "selected_dose": {
            key: copy.deepcopy(dose[key])
            for key in (
                "selected_arm",
                "active_policy_branch_multiplier",
                "policy_aux_active_batch_size",
                "optimizer_steps",
                "checkpoint_steps",
                "expected_aux_active_row_draws",
                "reference_parent_kl",
                "reference_trunk_relative_l2",
                "reference_teacher_gap_closure",
                "trust_contract",
            )
        },
        "lineage_contract": {
            "learner_parent_sha256": stage_a.EXPECTED_F7_PARENT_SHA256,
            "upgraded_initializer_sha256": upgrade["upgraded_initializer"]["sha256"],
            "every_arm_restarts_from_upgraded_initializer": True,
            "fresh_adam_every_arm": True,
            "dose_evidence_checkpoint_is_never_an_initializer": True,
            "candidate_chaining_forbidden": True,
        },
        "fixed_experiment_surface": {
            "corpus": {
                "path": str(data),
                "corpus_meta_file_sha256": source_inputs["corpus_meta_file_sha256"],
                "payload_inventory_sha256": source_inputs["payload_inventory_sha256"],
                "coherent_corpus_admission_sha256": source_inputs[
                    "coherent_corpus_admission_sha256"
                ],
            },
            "operator_target_contract_sha256": stage_a.EXPECTED_TARGET_CONTRACT_SHA256,
            "target_information_regime": stage_a.TARGET_INFORMATION_REGIME,
            "topology": copy.deepcopy(source["topology"]),
            "source_training_recipe": copy.deepcopy(source_recipe),
            "source_training_recipe_sha256": _value_sha256(source_recipe),
            "treatment_fields_only": sorted(TREATMENT_FIELDS),
        },
        "treatment_exposure": exposure,
        "active_arms": active_arms,
        "inactive_arms": [arm for arm in ARM_ORDER if arm not in active_arms],
        "arms": arm_records,
        "comparison_contract": {
            "same_optimizer_step_is_not_a_dose_match": True,
            "checkpoint_selected_by_parent_kl_and_trunk_drift": True,
            "max_parent_kl_ratio_to_stage_a_reference": DISTANCE_RATIO_LIMIT,
            "max_trunk_relative_l2_ratio_to_stage_a_reference": DISTANCE_RATIO_LIMIT,
            "max_parent_kl": float(source["selection_contract"]["max_parent_kl"]),
            "max_trunk_relative_l2": float(
                source["selection_contract"]["max_trunk_relative_l2"]
            ),
            "objective": (
                "dose_matched_fingerprint_nomination_only"
            ),
            "teacher_gap_closure_ranking_authority": False,
            "paired_playing_strength_is_final_authority": True,
            "playing_strength_evaluation_required": True,
        },
        "inputs": {
            "python": str(python),
            "one_dose_trainer": str(trainer),
            "one_dose_trainer_sha256": _file_sha256(trainer),
            "lock": str(lock),
            "lock_file_sha256": _file_sha256(lock),
            "data": str(data),
            "corpus_meta_file_sha256": source_inputs["corpus_meta_file_sha256"],
            "payload_inventory_sha256": source_inputs["payload_inventory_sha256"],
            "validation_manifest": str(source_inputs["validation_manifest"]),
            "validation_manifest_file_sha256": source_inputs[
                "validation_manifest_file_sha256"
            ],
            "coherent_corpus_receipt": str(source_inputs["coherent_corpus_receipt"]),
            "coherent_corpus_receipt_file_sha256": source_inputs[
                "coherent_corpus_receipt_file_sha256"
            ],
            "architecture_upgrade_receipt": str(upgrade_path),
            "architecture_upgrade_receipt_file_sha256": _file_sha256(upgrade_path),
            "independent_parent_authority": str(
                source_inputs["independent_parent_authority"]
            ),
            "independent_parent_authority_file_sha256": source_inputs[
                "independent_parent_authority_file_sha256"
            ],
            "ddp_canary_receipt": str(canary),
            "ddp_canary_receipt_file_sha256": _file_sha256(canary),
            "reviewed_code_tree_sha256": reviewed_code,
            "reviewed_lock_file_sha256": reviewed_lock,
        },
        "output_root": str(output_root),
    }
    payload["commands"] = {
        arm: _one_dose_invocation(payload, arm) for arm in active_arms
    }
    payload["campaign_sha256"] = _value_sha256(payload)
    return payload


def _one_dose_invocation(campaign: Mapping[str, Any], arm: str) -> list[str]:
    if arm not in campaign.get("active_arms", ()):
        reason = campaign.get("arms", {}).get(arm, {}).get("inactive_reason")
        raise CampaignError(f"Stage-B arm {arm} is not launchable: {reason}")
    inputs = campaign["inputs"]
    arm_root = Path(campaign["output_root"]) / "arms" / arm
    command = [
        str(inputs["python"]),
        str(inputs["one_dose_trainer"]),
        "--lock",
        str(inputs["lock"]),
        "--data",
        str(inputs["data"]),
        "--validation-manifest",
        str(inputs["validation_manifest"]),
        "--coherent-corpus-admission",
        str(inputs["coherent_corpus_receipt"]),
        "--architecture-upgrade-receipt",
        str(inputs["architecture_upgrade_receipt"]),
        "--independent-parent-authority",
        str(inputs["independent_parent_authority"]),
        "--checkpoint",
        str(arm_root / "candidate.pt"),
        "--report",
        str(arm_root / "train.report.json"),
        "--receipt",
        str(arm_root / "one-dose.receipt.json"),
        "--python",
        str(inputs["python"]),
        "--gpu",
        "0",
        "--topology",
        "b200-8gpu-ddp",
        "--ddp-canary-receipt",
        str(inputs["ddp_canary_receipt"]),
        "--ablation-id",
        f"coherent-n128-stage-b-{arm.lower()}",
        "--recipe-overrides-json",
        _canonical_json(campaign["arms"][arm]["recipe_overrides"]),
        "--ablation-code-tree-sha256",
        str(inputs["reviewed_code_tree_sha256"]),
        "--reviewed-lock-file-sha256",
        str(inputs["reviewed_lock_file_sha256"]),
        "--diagnostic-dose-curve",
    ]
    intermediate = campaign["selected_dose"]["checkpoint_steps"][:-1]
    if intermediate:
        command.extend(
            ["--diagnostic-checkpoint-steps", ",".join(map(str, intermediate))]
        )
    return command


def _load_campaign(path: Path) -> tuple[Path, dict[str, Any]]:
    campaign_path, campaign = _load_signed(
        path,
        where="Stage-B campaign",
        schema=SCHEMA,
        digest_field="campaign_sha256",
    )
    _assert_treatment_isolation(campaign.get("arms", {}))
    active = campaign.get("active_arms")
    dose_authority = campaign.get("dose_authority")
    if (
        not isinstance(active, list)
        or not active
        or any(arm not in ARM_ORDER for arm in active)
        or len(active) != len(set(active))
        or "BASE" not in active
        or campaign.get("diagnostic_only") is not True
        or campaign.get("promotion_eligible") is not False
        or not isinstance(dose_authority, Mapping)
        or dose_authority.get("kind")
        not in {"stage_a_selection", "direction_corrected_recovery_selection"}
        or (
            dose_authority.get("kind") == "direction_corrected_recovery_selection"
            and dose_authority.get("playing_strength_evaluation_required") is not True
        )
    ):
        raise CampaignError("Stage-B campaign arm/diagnostic semantics drifted")
    forced_exposure = bool(
        campaign.get("treatment_exposure", {}).get(
            "forced_treatment_structurally_active"
        )
    )
    if ("FORCED" in active) != forced_exposure:
        raise CampaignError("FORCED launch eligibility differs from signed exposure")
    trust_exposure = bool(
        campaign.get("treatment_exposure", {}).get(
            "trust_treatment_structurally_active"
        )
    )
    if ("TRUST" in active) != trust_exposure:
        raise CampaignError("TRUST launch eligibility differs from signed exposure")
    value_trunk_exposure = bool(
        campaign.get("treatment_exposure", {}).get(
            "value_trunk_treatment_structurally_active"
        )
    )
    if ("VTRUNK25" in active) != value_trunk_exposure:
        raise CampaignError("VTRUNK25 launch eligibility differs from signed exposure")
    return campaign_path, campaign


def _verify_campaign_inputs(campaign: Mapping[str, Any]) -> None:
    inputs = campaign["inputs"]
    for path_key, digest_key in (
        ("one_dose_trainer", "one_dose_trainer_sha256"),
        ("lock", "lock_file_sha256"),
        ("validation_manifest", "validation_manifest_file_sha256"),
        ("coherent_corpus_receipt", "coherent_corpus_receipt_file_sha256"),
        ("architecture_upgrade_receipt", "architecture_upgrade_receipt_file_sha256"),
        ("independent_parent_authority", "independent_parent_authority_file_sha256"),
        ("ddp_canary_receipt", "ddp_canary_receipt_file_sha256"),
    ):
        path = _regular_file(Path(str(inputs[path_key])), where=path_key)
        if _file_sha256(path) != inputs[digest_key]:
            raise CampaignError(f"Stage-B campaign input bytes changed: {path_key}")
    data = Path(str(inputs["data"])).expanduser().resolve(strict=True)
    if (
        not data.is_dir()
        or _file_sha256(data / "corpus_meta.json") != inputs["corpus_meta_file_sha256"]
    ):
        raise CampaignError("Stage-B coherent corpus changed after planning")
    authority = campaign.get("dose_authority")
    if not isinstance(authority, Mapping):
        raise CampaignError("Stage-B campaign lost its dose authority")
    selection_ref = authority.get("selection")
    if not isinstance(selection_ref, Mapping):
        raise CampaignError("Stage-B campaign lost its selection authority")
    kind = authority.get("kind")
    if kind == "stage_a_selection":
        _selection_path, _selection, _source_path, _source, dose = (
            _load_stage_a_selection(Path(str(selection_ref["path"])))
        )
    elif kind == "direction_corrected_recovery_selection":
        _selection_path, _selection, _source_path, _source, dose = (
            _load_recovery_selection(Path(str(selection_ref["path"])))
        )
    else:
        raise CampaignError(f"unknown Stage-B dose authority {kind!r}")
    expected_authority: dict[str, Any] = {
        "kind": dose["authority_kind"],
        "selection": dose["selection_authority"],
        "source_stage_a_campaign": dose["source_campaign"],
        "selected_arm": dose["selected_arm"],
        "selected_checkpoint": dose["selected_checkpoint"],
    }
    if kind == "direction_corrected_recovery_selection":
        expected_authority.update(
            {
                "recovery_campaign": dose["recovery_campaign"],
                "recovery_fingerprint": dose["recovery_fingerprint"],
                "recovery_receipt": dose["recovery_receipt"],
                "selected_recipe_overrides": dose["selected_recipe_overrides"],
                "playing_strength_evaluation_required": True,
            }
        )
    if authority != expected_authority:
        raise CampaignError("Stage-B dose authority changed after planning")
    for key in (
        "selected_arm",
        "active_policy_branch_multiplier",
        "policy_aux_active_batch_size",
        "optimizer_steps",
        "checkpoint_steps",
        "expected_aux_active_row_draws",
        "reference_parent_kl",
        "reference_trunk_relative_l2",
        "reference_teacher_gap_closure",
        "trust_contract",
    ):
        if dose[key] != campaign["selected_dose"][key]:
            raise CampaignError(f"selected dose changed after planning: {key}")


def _effective_treatment_assertion(
    campaign: Mapping[str, Any], arm: str, effective: Mapping[str, Any]
) -> None:
    expected = campaign["arms"][arm]["recipe_overrides"]
    fixed_keys = (
        "epochs",
        "max_steps",
        "lr",
        "lr_warmup_steps",
        "forced_row_value_weight",
        "public_card_lr_mult",
        "trunk_lr_mult",
        "per_game_policy_surprise_weighting",
        "policy_aux_active_batch_size",
    )
    drift = {
        key: {"expected": expected[key], "actual": effective.get(key)}
        for key in fixed_keys
        if effective.get(key) != expected[key]
    }
    expected_value_trunk_scale = float(expected.get("value_trunk_grad_scale", 1.0))
    actual_value_trunk_scale = float(effective.get("value_trunk_grad_scale", 1.0))
    if actual_value_trunk_scale != expected_value_trunk_scale:
        drift["value_trunk_grad_scale"] = {
            "expected": expected_value_trunk_scale,
            "actual": actual_value_trunk_scale,
        }
    expected_typed = expected.get("forced_row_value_action_type_weights", "")
    actual_typed = effective.get("forced_row_value_action_type_weights", "")
    if actual_typed != expected_typed:
        drift["forced_row_value_action_type_weights"] = {
            "expected": expected_typed,
            "actual": actual_typed,
        }
    for key in (
        "policy_kl_target",
        "policy_kl_dual_lr",
        "policy_kl_max_weight",
        "policy_kl_anchor_direction",
    ):
        expected_value = expected.get(key)
        actual_value = effective.get(key)
        if actual_value != expected_value:
            # Historical BASE recipes may carry the inert forward direction but
            # never an adaptive target. It is not a second treatment.
            if not (
                key == "policy_kl_anchor_direction"
                and expected_value is None
                and actual_value in {None, "forward"}
            ):
                drift[key] = {
                    "expected": expected_value,
                    "actual": actual_value,
                }
    if drift:
        raise CampaignError(
            f"Stage-B arm {arm} effective treatment/dose drift: "
            + json.dumps(drift, sort_keys=True)
        )


def _dry_run_arm(campaign: Mapping[str, Any], arm: str) -> dict[str, Any]:
    invocation = _one_dose_invocation(campaign, arm)
    try:
        plan = base_campaign._one_dose_dry_run(invocation)  # noqa: SLF001
    except base_campaign.CampaignError as error:
        raise CampaignError(str(error)) from error
    command = [str(value) for value in plan["command"]]
    initializer = _regular_file(
        Path(stage_a._option(command, "--init-checkpoint")),  # noqa: SLF001
        where="rendered f7 initializer",
    )
    learner_parent = plan.get("learner_lineage_parent")
    effective = plan.get("learner_ablation", {}).get("effective_recipe")
    if not isinstance(effective, Mapping):
        raise CampaignError(f"Stage-B arm {arm} lost its effective recipe")
    _effective_treatment_assertion(campaign, arm, effective)
    expected_steps = int(campaign["selected_dose"]["optimizer_steps"])
    expected_intermediate = campaign["selected_dose"]["checkpoint_steps"][:-1]
    actual_checkpoint_options = one_dose._literal_option_values(  # noqa: SLF001
        command, "--checkpoint-steps"
    )
    expected_checkpoint_options = (
        [",".join(map(str, expected_intermediate))] if expected_intermediate else []
    )
    has_trust_target = "--policy-kl-target" in command
    expected_trust_target = arm == "TRUST"
    expected_value_trunk_scale = float(
        campaign["arms"][arm]["recipe_overrides"].get("value_trunk_grad_scale", 1.0)
    )
    actual_value_trunk_scale = (
        float(stage_a._option(command, "--value-trunk-grad-scale"))  # noqa: SLF001
        if "--value-trunk-grad-scale" in command
        else 1.0
    )
    if (
        _file_sha256(initializer)
        != campaign["lineage_contract"]["upgraded_initializer_sha256"]
        or "--no-resume-optimizer" not in command
        or not any(token == "--nproc_per_node=8" for token in command)
        or int(stage_a._option(command, "--max-steps")) != expected_steps  # noqa: SLF001
        or actual_checkpoint_options != expected_checkpoint_options
        or has_trust_target is not expected_trust_target
        or float(stage_a._option(command, "--trunk-lr-mult"))  # noqa: SLF001
        != float(campaign["arms"][arm]["recipe_overrides"]["trunk_lr_mult"])
        or actual_value_trunk_scale != expected_value_trunk_scale
        or not isinstance(learner_parent, Mapping)
        or learner_parent.get("role") != "diagnostic_independent_parent"
        or learner_parent.get("checkpoint", {}).get("sha256")
        != stage_a.EXPECTED_F7_PARENT_SHA256
        or plan.get("learner_ablation", {}).get("diagnostic_only") is not True
        or plan.get("learner_ablation", {}).get("promotion_eligible") is not False
    ):
        raise CampaignError(f"Stage-B arm {arm} lost exact f7/fresh-Adam selected dose")
    trust_contract = campaign["selected_dose"]["trust_contract"]
    if expected_trust_target and (
        float(stage_a._option(command, "--policy-kl-target"))  # noqa: SLF001
        != float(trust_contract["policy_kl_target"])
        or float(stage_a._option(command, "--policy-kl-dual-lr"))  # noqa: SLF001
        != float(trust_contract["policy_kl_dual_lr"])
        or float(stage_a._option(command, "--policy-kl-max-weight"))  # noqa: SLF001
        != float(trust_contract["policy_kl_max_weight"])
        or stage_a._option(command, "--policy-kl-anchor-direction")  # noqa: SLF001
        != trust_contract["policy_kl_anchor_direction"]
    ):
        raise CampaignError("Stage-B TRUST arm lost its projected-dual contract")
    return plan


def _step_checkpoint(arm_root: Path, step: int, terminal: int) -> Path:
    return (
        arm_root / "candidate.pt"
        if step == terminal
        else arm_root / f"candidate_step{step:04d}.pt"
    )


def _verify_value_trunk_routing(
    report: Mapping[str, Any], *, arm: str, expected_scale: float
) -> dict[str, Any]:
    """Require runtime proof that the requested derivative boundary executed."""

    routing = report.get("value_gradient_routing")
    if (
        not isinstance(routing, Mapping)
        or report.get("value_trunk_grad_scale") != expected_scale
        or routing.get("schema_version") != "scalar-value-trunk-gradient-routing-v1"
        or routing.get("scalar_value_trunk_grad_scale") != expected_scale
        or routing.get("shared_state_upstream_gradient_scale") != expected_scale
        or routing.get("active") is not (expected_scale != 1.0)
        or routing.get("forward_value_identity") is not True
        or routing.get("value_head_parameter_gradient_scale") != 1.0
        or routing.get("policy_gradient_unchanged") is not True
        or routing.get("optimizer_parameter_groups_unchanged") is not True
        or routing.get("scope") != "value_family_readouts_all_shared_inputs"
        or routing.get("all_scalar_value_shared_inputs_scaled") is not True
        or routing.get("all_value_family_shared_inputs_scaled") is not True
        or not isinstance(routing.get("shared_input_paths"), list)
        or "cls_state" not in routing.get("shared_input_paths", [])
    ):
        raise CampaignError(
            f"Stage-B arm {arm} did not execute its value-trunk routing contract"
        )
    return copy.deepcopy(dict(routing))


def _verify_completed_arm(
    campaign_path: Path, campaign: Mapping[str, Any], arm: str
) -> dict[str, Any]:
    arm_root = Path(campaign["output_root"]) / "arms" / arm
    receipt_path = _regular_file(
        arm_root / "one-dose.receipt.json", where=f"Stage-B arm {arm} receipt"
    )
    try:
        receipt = one_dose._load_authenticated_completed_aux_receipt(  # noqa: SLF001
            receipt_path
        )
    except one_dose.ExecutorError as error:
        raise CampaignError(f"Stage-B arm {arm} receipt refused: {error}") from error
    report_path, report = _load_json(
        arm_root / "train.report.json", where=f"Stage-B arm {arm} report"
    )
    terminal = int(campaign["selected_dose"]["optimizer_steps"])
    schedule = list(campaign["selected_dose"]["checkpoint_steps"])
    intermediate_steps = schedule[:-1]
    expected_aux = int(campaign["selected_dose"]["policy_aux_active_batch_size"])
    expected_aux_rows = expected_aux * WORLD_SIZE * terminal
    effective = receipt.get("learner_ablation", {}).get("effective_recipe")
    if not isinstance(effective, Mapping):
        raise CampaignError(f"Stage-B arm {arm} receipt lost its effective recipe")
    _effective_treatment_assertion(campaign, arm, effective)
    value_gradient_routing = _verify_value_trunk_routing(
        report,
        arm=arm,
        expected_scale=float(
            campaign["arms"][arm]["recipe_overrides"].get("value_trunk_grad_scale", 1.0)
        ),
    )
    checkpoint = _regular_file(
        arm_root / "candidate.pt", where=f"Stage-B arm {arm} checkpoint"
    )
    outputs = receipt.get("outputs")
    learner_parent = receipt.get("learner_lineage_parent")
    intermediate = report.get("intermediate_checkpoints")
    controller = report.get("adaptive_policy_kl_controller")
    trust_contract = campaign["selected_dose"]["trust_contract"]
    if arm == "TRUST":
        if (
            not isinstance(controller, Mapping)
            or controller.get("schema_version") != train_bc.POLICY_KL_CONTROLLER_SCHEMA
            or controller.get("direction") != "forward"
            or float(controller.get("target_kl", math.nan))
            != float(trust_contract["policy_kl_target"])
            or float(controller.get("dual_lr", math.nan))
            != float(trust_contract["policy_kl_dual_lr"])
            or float(controller.get("max_weight", math.nan))
            != float(trust_contract["policy_kl_max_weight"])
            or int(controller.get("updates", -1)) != terminal
            or int(controller.get("eligible_rows", 0)) <= 0
        ):
            raise CampaignError("Stage-B TRUST arm did not execute its KL controller")
    elif controller is not None:
        raise CampaignError(f"Stage-B arm {arm} unexpectedly ran a KL controller")
    if (
        receipt.get("status") != "complete"
        or receipt.get("returncode") != 0
        or receipt.get("diagnostic_only") is not True
        or receipt.get("promotion_eligible") is not False
        or not isinstance(outputs, Mapping)
        or outputs.get("checkpoint_sha256") != _file_sha256(checkpoint)
        or outputs.get("report_sha256") != _file_sha256(report_path)
        or report.get("steps_completed") != terminal
        or report.get("optimizer_restored") is not False
        or report.get("policy_aux_active_batch_size") != expected_aux
        or report.get("policy_aux_active_rows") != expected_aux_rows
        or report.get("checkpoint_steps_requested") != intermediate_steps
        or not isinstance(intermediate, list)
        or [record.get("optimizer_step") for record in intermediate]
        != intermediate_steps
        or not isinstance(learner_parent, Mapping)
        or learner_parent.get("role") != "diagnostic_independent_parent"
        or learner_parent.get("checkpoint", {}).get("sha256")
        != stage_a.EXPECTED_F7_PARENT_SHA256
    ):
        raise CampaignError(f"Stage-B arm {arm} did not complete its exact dose")
    checkpoints = []
    for step in schedule:
        path = _regular_file(
            _step_checkpoint(arm_root, step, terminal),
            where=f"Stage-B arm {arm} step {step} checkpoint",
        )
        checkpoints.append(
            {"optimizer_step": step, "path": str(path), "sha256": _file_sha256(path)}
        )
    arm_receipt: dict[str, Any] = {
        "schema_version": ARM_RECEIPT_SCHEMA,
        "status": "complete",
        "diagnostic_only": True,
        "promotion_eligible": False,
        "campaign": {
            "path": str(campaign_path),
            "file_sha256": _file_sha256(campaign_path),
            "campaign_sha256": campaign["campaign_sha256"],
        },
        "arm": arm,
        "treatment": copy.deepcopy(campaign["arms"][arm]["treatment"]),
        "selected_dose": copy.deepcopy(campaign["selected_dose"]),
        "one_dose_receipt": {
            "path": str(receipt_path),
            "file_sha256": _file_sha256(receipt_path),
            "receipt_sha256": receipt["receipt_sha256"],
        },
        "training_report": {
            "path": str(report_path),
            "file_sha256": _file_sha256(report_path),
        },
        "checkpoints": checkpoints,
        "learner_parent_sha256": stage_a.EXPECTED_F7_PARENT_SHA256,
        "fresh_adam": True,
        "policy_aux_active_rows": expected_aux_rows,
        "adaptive_policy_kl_controller": copy.deepcopy(controller),
        "value_gradient_routing": value_gradient_routing,
    }
    arm_receipt["receipt_sha256"] = _value_sha256(arm_receipt)
    wrapper_path = arm_root / "stage-b.receipt.json"
    _write_immutable(wrapper_path, arm_receipt)
    return {"path": wrapper_path, "payload": arm_receipt, "report": report}


def _run_arm(
    campaign_path: Path,
    campaign: Mapping[str, Any],
    arm: str,
    *,
    go: bool,
) -> dict[str, Any]:
    _verify_campaign_inputs(campaign)
    plan = _dry_run_arm(campaign, arm)
    if not go:
        return {"mode": "dry-run", "arm": arm, "one_dose_plan": plan}
    invocation = _one_dose_invocation(campaign, arm)
    result = subprocess.run([*invocation, "--go"], check=False)
    if result.returncode != 0:
        raise CampaignError(f"Stage-B arm {arm} exited {result.returncode}")
    completed = _verify_completed_arm(campaign_path, campaign, arm)
    payload = completed["payload"]
    return {
        "mode": "go",
        "arm": arm,
        "receipt": str(completed["path"]),
        "receipt_file_sha256": _file_sha256(completed["path"]),
        "receipt_sha256": payload["receipt_sha256"],
    }


def _trunk_relative_l2(drift: Mapping[str, Any]) -> float:
    try:
        return stage_a._trunk_relative_l2(drift)  # noqa: SLF001
    except stage_a.CampaignError as error:
        raise CampaignError(str(error)) from error


def _fingerprint_arm(
    campaign_path: Path,
    campaign: Mapping[str, Any],
    arm: str,
    *,
    go: bool,
    device: str,
) -> dict[str, Any]:
    completed = _verify_completed_arm(campaign_path, campaign, arm)
    arm_root = Path(campaign["output_root"]) / "arms" / arm
    report_path = Path(completed["payload"]["training_report"]["path"])
    report = completed["report"]
    validation_value = report.get("validation_game_seed_manifest")
    validation = (
        Path(str(campaign["inputs"]["validation_manifest"]))
        if not validation_value
        else Path(str(validation_value)).expanduser()
    )
    if not validation.is_absolute():
        validation = report_path.parent / validation
    validation = _regular_file(validation, where=f"Stage-B arm {arm} validation")
    _authority_path, authority = _load_json(
        Path(str(campaign["inputs"]["independent_parent_authority"])),
        where="independent parent authority",
    )
    parent = _regular_file(
        Path(
            str(
                authority["function_preserving_upgrade"]["upgraded_initializer"]["path"]
            )
        ),
        where="upgraded f7 initializer",
    )
    output_root = arm_root / "fingerprints"
    terminal = int(campaign["selected_dose"]["optimizer_steps"])
    commands: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for step in campaign["selected_dose"]["checkpoint_steps"]:
        checkpoint = _regular_file(
            _step_checkpoint(arm_root, int(step), terminal),
            where=f"Stage-B arm {arm} step {step} checkpoint",
        )
        functional_output = output_root / f"step{int(step):04d}.functional.json"
        drift_output = output_root / f"step{int(step):04d}.drift.json"
        functional_command = [
            str(campaign["inputs"]["python"]),
            str(REPO_ROOT / "tools" / "posthoc_teacher_gap_probe.py"),
            "--report",
            str(report_path),
            "--checkpoint",
            str(checkpoint),
            "--parent-checkpoint",
            str(parent),
            "--data",
            str(campaign["inputs"]["data"]),
            "--validation-manifest",
            str(validation),
            "--device",
            device,
            "--output",
            str(functional_output),
        ]
        drift_command = [
            str(campaign["inputs"]["python"]),
            str(REPO_ROOT / "tools" / "audit_checkpoint_layer_drift.py"),
            "--baseline",
            str(parent),
            "--candidate",
            str(checkpoint),
            "--output",
            str(drift_output),
        ]
        commands.append(
            {
                "step": int(step),
                "functional": functional_command,
                "drift": drift_command,
            }
        )
        if not go:
            continue
        output_root.mkdir(parents=True, exist_ok=True)
        for command in (functional_command, drift_command):
            result = subprocess.run(command, check=False)
            if result.returncode != 0:
                raise CampaignError(
                    f"Stage-B arm {arm} step {step} fingerprint exited {result.returncode}"
                )
        functional_path, functional = _load_json(
            functional_output, where=f"Stage-B arm {arm} functional fingerprint"
        )
        drift_path, drift = _load_json(
            drift_output, where=f"Stage-B arm {arm} layer drift"
        )
        fingerprint = functional.get("functional_dose_fingerprint")
        parent_kl = (
            fingerprint.get("kl_parent_candidate_mean")
            if isinstance(fingerprint, Mapping)
            else None
        )
        closure = functional.get("teacher_gap", {}).get(
            "active_policy_teacher_gap_closure"
        )
        trunk = _trunk_relative_l2(drift)
        if (
            not isinstance(fingerprint, Mapping)
            or fingerprint.get("surface")
            != "validation_policy_active_multi_action_rows"
            or functional.get("inputs", {}).get("checkpoint", {}).get("sha256")
            != _file_sha256(checkpoint)
            or functional.get("inputs", {}).get("parent_checkpoint", {}).get("sha256")
            != _file_sha256(parent)
            or drift.get("baseline", {}).get("sha256") != _file_sha256(parent)
            or drift.get("candidate", {}).get("sha256") != _file_sha256(checkpoint)
            or not all(
                isinstance(value, (int, float)) and math.isfinite(float(value))
                for value in (parent_kl, closure, trunk)
            )
        ):
            raise CampaignError(f"Stage-B arm {arm} step {step} fingerprint drifted")
        records.append(
            {
                "step": int(step),
                "terminal_selected_dose": int(step) == terminal,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _file_sha256(checkpoint),
                "functional": {
                    "path": str(functional_path),
                    "file_sha256": _file_sha256(functional_path),
                    "parent_kl": float(parent_kl),
                    "teacher_gap_closure": float(closure),
                    "eligible_rows": int(fingerprint["eligible_rows"]),
                },
                "layer_drift": {
                    "path": str(drift_path),
                    "file_sha256": _file_sha256(drift_path),
                    "trunk_relative_l2": float(trunk),
                    "global_relative_l2": float(drift["global"]["relative_l2"]),
                },
            }
        )
    if not go:
        return {"mode": "dry-run", "arm": arm, "commands": commands}
    payload: dict[str, Any] = {
        "schema_version": FINGERPRINT_SCHEMA,
        "campaign": {
            "path": str(campaign_path),
            "file_sha256": _file_sha256(campaign_path),
            "campaign_sha256": campaign["campaign_sha256"],
        },
        "arm": arm,
        "treatment": copy.deepcopy(campaign["arms"][arm]["treatment"]),
        "selected_dose": copy.deepcopy(campaign["selected_dose"]),
        "arm_receipt": {
            "path": str(completed["path"]),
            "file_sha256": _file_sha256(completed["path"]),
            "receipt_sha256": completed["payload"]["receipt_sha256"],
        },
        "parent_checkpoint_sha256": _file_sha256(parent),
        "checkpoints": records,
        "diagnostic_only": True,
        "promotion_eligible": False,
    }
    payload["fingerprint_sha256"] = _value_sha256(payload)
    output = output_root / "fingerprint.json"
    _write_immutable(output, payload)
    return {
        "mode": "go",
        "arm": arm,
        "fingerprint": str(output),
        "fingerprint_file_sha256": _file_sha256(output),
        "fingerprint_sha256": payload["fingerprint_sha256"],
    }


def _parse_bindings(values: Sequence[str], *, allowed: set[str]) -> dict[str, Path]:
    try:
        return stage_a._parse_bindings(  # noqa: SLF001
            values, allowed=allowed, label="Stage-B arm fingerprint"
        )
    except stage_a.CampaignError as error:
        raise CampaignError(str(error)) from error


def _select_dose_matched_checkpoint(
    candidates: Sequence[Mapping[str, Any]],
    *,
    reference_parent_kl: float,
    reference_trunk_relative_l2: float,
    terminal_step: int,
) -> dict[str, Any]:
    """Choose a checkpoint using update geometry only, never outcome quality."""

    def positive_ratio(value: float, reference: float) -> float:
        if value <= 0.0 or reference <= 0.0:
            return math.inf
        return max(value / reference, reference / value)

    normalized: list[dict[str, Any]] = []
    for candidate in candidates:
        row = copy.deepcopy(dict(candidate))
        kl_ratio = positive_ratio(float(row["parent_kl"]), float(reference_parent_kl))
        trunk_ratio = positive_ratio(
            float(row["trunk_relative_l2"]),
            float(reference_trunk_relative_l2),
        )
        row.update(
            {
                "parent_kl_ratio_to_stage_a_reference": kl_ratio,
                "trunk_ratio_to_stage_a_reference": trunk_ratio,
                "dose_match_log_distance": math.hypot(
                    math.log(kl_ratio), math.log(trunk_ratio)
                ),
            }
        )
        normalized.append(row)
    if not normalized:
        raise CampaignError("cannot dose-match an empty checkpoint frontier")
    return sorted(
        normalized,
        key=lambda row: (
            row["dose_match_log_distance"],
            abs(int(row["step"]) - int(terminal_step)),
            int(row["step"]),
        ),
    )[0]


def _compare(
    campaign_path: Path,
    campaign: Mapping[str, Any],
    bindings: Mapping[str, Path],
) -> dict[str, Any]:
    terminal = int(campaign["selected_dose"]["optimizer_steps"])
    cap_kl = float(campaign["comparison_contract"]["max_parent_kl"])
    cap_trunk = float(campaign["comparison_contract"]["max_trunk_relative_l2"])
    reference_kl = float(campaign["selected_dose"]["reference_parent_kl"])
    reference_trunk = float(campaign["selected_dose"]["reference_trunk_relative_l2"])
    max_kl_ratio = float(
        campaign["comparison_contract"]["max_parent_kl_ratio_to_stage_a_reference"]
    )
    max_trunk_ratio = float(
        campaign["comparison_contract"][
            "max_trunk_relative_l2_ratio_to_stage_a_reference"
        ]
    )

    records: dict[str, Any] = {}
    eligible: list[dict[str, Any]] = []
    for arm in campaign["active_arms"]:
        path, fingerprint = _load_signed(
            bindings[arm],
            where=f"Stage-B arm {arm} fingerprint",
            schema=FINGERPRINT_SCHEMA,
            digest_field="fingerprint_sha256",
        )
        ref = fingerprint.get("campaign")
        checkpoints = fingerprint.get("checkpoints")
        if (
            fingerprint.get("arm") != arm
            or fingerprint.get("treatment") != campaign["arms"][arm]["treatment"]
            or fingerprint.get("selected_dose") != campaign["selected_dose"]
            or not isinstance(ref, Mapping)
            or ref.get("file_sha256") != _file_sha256(campaign_path)
            or ref.get("campaign_sha256") != campaign["campaign_sha256"]
            or not isinstance(checkpoints, list)
            or [row.get("step") for row in checkpoints]
            != campaign["selected_dose"]["checkpoint_steps"]
        ):
            raise CampaignError(f"Stage-B arm {arm} fingerprint binding drifted")
        terminal_rows = [row for row in checkpoints if row.get("step") == terminal]
        if (
            len(terminal_rows) != 1
            or terminal_rows[0].get("terminal_selected_dose") is not True
        ):
            raise CampaignError(
                f"Stage-B arm {arm} lacks its exact selected-dose result"
            )
        candidates: list[dict[str, Any]] = []
        for row in checkpoints:
            step = int(row.get("step", -1))
            checkpoint = _regular_file(
                Path(str(row.get("checkpoint", ""))),
                where=f"Stage-B arm {arm} step {step} checkpoint",
            )
            if row.get("checkpoint_sha256") != _file_sha256(checkpoint):
                raise CampaignError(
                    f"Stage-B arm {arm} step {step} checkpoint bytes drifted"
                )
            for section, label in (
                (row.get("functional"), "functional fingerprint"),
                (row.get("layer_drift"), "layer drift"),
            ):
                if not isinstance(section, Mapping):
                    raise CampaignError(
                        f"Stage-B arm {arm} step {step} {label} is malformed"
                    )
                artifact = _regular_file(
                    Path(str(section.get("path", ""))),
                    where=f"Stage-B arm {arm} step {step} {label}",
                )
                if section.get("file_sha256") != _file_sha256(artifact):
                    raise CampaignError(
                        f"Stage-B arm {arm} step {step} {label} bytes drifted"
                    )
            parent_kl = float(row["functional"]["parent_kl"])
            closure = float(row["functional"]["teacher_gap_closure"])
            trunk = float(row["layer_drift"]["trunk_relative_l2"])
            if not all(math.isfinite(value) for value in (parent_kl, closure, trunk)):
                raise CampaignError(
                    f"Stage-B arm {arm} step {step} has non-finite fingerprint"
                )
            candidates.append(
                {
                    "arm": arm,
                    "step": step,
                    "terminal_selected_dose": step == terminal,
                    "treatment": copy.deepcopy(campaign["arms"][arm]["treatment"]),
                    "checkpoint": row["checkpoint"],
                    "checkpoint_sha256": row["checkpoint_sha256"],
                    "parent_kl": parent_kl,
                    "trunk_relative_l2": trunk,
                    "teacher_gap_closure": closure,
                }
            )
        # Match dose using only parent KL and trunk drift. Teacher closure is an
        # outcome and must never influence which checkpoint is selected.
        result = _select_dose_matched_checkpoint(
            candidates,
            reference_parent_kl=reference_kl,
            reference_trunk_relative_l2=reference_trunk,
            terminal_step=terminal,
        )
        within = (
            result["parent_kl"] <= cap_kl and result["trunk_relative_l2"] <= cap_trunk
        )
        matched = (
            result["parent_kl_ratio_to_stage_a_reference"] <= max_kl_ratio
            and result["trunk_ratio_to_stage_a_reference"] <= max_trunk_ratio
        )
        result.update(
            {
                "within_stage_a_drift_budgets": within,
                "dose_matched_to_stage_a_reference": matched,
                "positive_teacher_gap_closure": result["teacher_gap_closure"] > 0.0,
                "fingerprint_eligible": (
                    within and matched and result["teacher_gap_closure"] > 0.0
                ),
            }
        )
        if result["fingerprint_eligible"]:
            eligible.append(result)
        records[arm] = {
            "path": str(path),
            "file_sha256": _file_sha256(path),
            "fingerprint_sha256": fingerprint["fingerprint_sha256"],
            "checkpoint_candidates": candidates,
            "matched": result,
        }
    base = records["BASE"]["matched"]
    effects = {
        arm: {
            "teacher_gap_closure_delta_vs_base": (
                records[arm]["matched"]["teacher_gap_closure"]
                - base["teacher_gap_closure"]
            ),
            "parent_kl_delta_vs_base": (
                records[arm]["matched"]["parent_kl"] - base["parent_kl"]
            ),
            "trunk_relative_l2_delta_vs_base": (
                records[arm]["matched"]["trunk_relative_l2"] - base["trunk_relative_l2"]
            ),
            "matched_step_delta_vs_base": (
                records[arm]["matched"]["step"] - base["step"]
            ),
        }
        for arm in campaign["active_arms"]
        if arm != "BASE"
    }
    leader = (
        sorted(
            eligible,
            key=lambda row: (
                row["step"],
                row["parent_kl"],
                row["trunk_relative_l2"],
                row["arm"],
            ),
        )[0]
        if eligible
        else None
    )
    payload: dict[str, Any] = {
        "schema_version": COMPARISON_SCHEMA,
        "campaign": {
            "path": str(campaign_path),
            "file_sha256": _file_sha256(campaign_path),
            "campaign_sha256": campaign["campaign_sha256"],
        },
        "selected_dose": copy.deepcopy(campaign["selected_dose"]),
        "dose_match_contract": copy.deepcopy(campaign["comparison_contract"]),
        "treatment_exposure": copy.deepcopy(campaign["treatment_exposure"]),
        "active_arms": list(campaign["active_arms"]),
        "inactive_arms": list(campaign["inactive_arms"]),
        "arm_fingerprints": records,
        "causal_effects_vs_base": effects,
        "fingerprint_leader": copy.deepcopy(leader),
        "fingerprint_leader_selection": "minimum_dose_within_fingerprint_budgets",
        "teacher_gap_closure_ranking_authority": False,
        "leader_is_diagnostic_not_promoted": True,
        "playing_strength_evaluation_still_required": True,
    }
    payload["comparison_sha256"] = _value_sha256(payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="seal selected-dose Stage-B arms")
    authority = plan.add_mutually_exclusive_group(required=True)
    authority.add_argument("--stage-a-selection", type=Path)
    authority.add_argument("--recovery-selection", type=Path)
    plan.add_argument("--ddp-canary-receipt", required=True, type=Path)
    plan.add_argument("--python", required=True, type=Path)
    plan.add_argument("--one-dose-trainer", type=Path)
    plan.add_argument("--reviewed-code-tree-sha256", required=True)
    plan.add_argument("--output-root", required=True, type=Path)
    plan.add_argument("--write", required=True, type=Path)

    run = sub.add_parser("run-arm", help="dry-run or execute one active arm")
    run.add_argument("--campaign", required=True, type=Path)
    run.add_argument("--arm", required=True, choices=ARM_ORDER)
    run.add_argument("--go", action="store_true")

    sequence = sub.add_parser("run-sequence", help="run all active arms serially")
    sequence.add_argument("--campaign", required=True, type=Path)
    sequence.add_argument("--arms", default="")
    sequence.add_argument("--go", action="store_true")

    fingerprint = sub.add_parser(
        "fingerprint-arm", help="measure selected-dose and intermediate checkpoints"
    )
    fingerprint.add_argument("--campaign", required=True, type=Path)
    fingerprint.add_argument("--arm", required=True, choices=ARM_ORDER)
    fingerprint.add_argument("--device", default="cuda:0")
    fingerprint.add_argument("--go", action="store_true")

    compare = sub.add_parser("compare", help="seal terminal causal comparison")
    compare.add_argument("--campaign", required=True, type=Path)
    compare.add_argument("--fingerprint", action="append", default=[])
    compare.add_argument("--write", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "plan":
            campaign = _plan(args)
            _write_immutable(args.write, campaign)
            result: dict[str, Any] = campaign
        else:
            campaign_path, campaign = _load_campaign(args.campaign)
            if args.command == "run-arm":
                result = _run_arm(campaign_path, campaign, args.arm, go=bool(args.go))
            elif args.command == "run-sequence":
                arms = (
                    [value.strip() for value in args.arms.split(",") if value.strip()]
                    if args.arms
                    else list(campaign["active_arms"])
                )
                if (
                    not arms
                    or len(arms) != len(set(arms))
                    or any(arm not in campaign["active_arms"] for arm in arms)
                ):
                    raise CampaignError(
                        "--arms must contain unique launch-eligible Stage-B arms"
                    )
                result = {
                    "mode": "go" if args.go else "dry-run",
                    "arms": [
                        _run_arm(campaign_path, campaign, arm, go=bool(args.go))
                        for arm in arms
                    ],
                }
            elif args.command == "fingerprint-arm":
                result = _fingerprint_arm(
                    campaign_path,
                    campaign,
                    args.arm,
                    go=bool(args.go),
                    device=args.device,
                )
            elif args.command == "compare":
                bindings = _parse_bindings(
                    args.fingerprint, allowed=set(campaign["active_arms"])
                )
                result = _compare(campaign_path, campaign, bindings)
                _write_immutable(args.write, result)
            else:  # pragma: no cover
                raise CampaignError(f"unknown command {args.command!r}")
    except (
        CampaignError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        print(f"Stage-B campaign refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
