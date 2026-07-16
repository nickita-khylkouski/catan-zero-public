#!/usr/bin/env python3
"""Train one independently initialized f7 learner on Stage-C targets.

This is deliberately one execution path, not another broad sweep.  It consumes
the derived Stage-C memmap admission, reloads the exact function-preserving f7
initializer with fresh Adam on 8xB200, and trains policy only on the coherent
reanalysed roots while retaining the complete base corpus for value learning.

The adaptive KL controller is intentionally disabled: its historical meter was
an optimizer-batch statistic and did not enforce the posthoc functional KL.
Checkpoints are instead selected using a frozen whole-game validation surface
after training.  The 64 active rows/rank dose keeps the 8,192-root overlay from
being replayed almost half an epoch on every optimizer step.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
for root in (REPO_ROOT, REPO_ROOT / "tools"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from tools import a1_b200_active_policy_campaign as stage_a  # noqa: E402
from tools import a1_function_preserving_upgrade as architecture_upgrade  # noqa: E402
from tools import a1_one_dose_train as one_dose  # noqa: E402
from tools import a1_stage_c_learner_overlay as overlay  # noqa: E402


SCHEMA = "a1-b200-stage-c-aligned-learner-campaign-v2"
EXECUTION_SCHEMA = "a1-b200-stage-c-aligned-learner-execution-v1"
FINGERPRINT_SCHEMA = "a1-b200-stage-c-aligned-learner-fingerprint-v1"
WORLD_SIZE = 8
LOCAL_BATCH_SIZE = 512
GLOBAL_BATCH_SIZE = WORLD_SIZE * LOCAL_BATCH_SIZE
POLICY_AUX_ACTIVE_BATCH_SIZE = 64
MAX_STEPS = 32
CHECKPOINT_STEPS = (1, 2, 4, 8, 12, 16, 24, 32)
INTERMEDIATE_CHECKPOINT_STEPS = CHECKPOINT_STEPS[:-1]
LR = 6.0e-5
LR_WARMUP_STEPS = 16
MAX_PARENT_KL = 0.03
MAX_TRUNK_RELATIVE_L2 = 0.03
ARMS = frozenset({"PRODUCTION_WEIGHTED", "STRATEGIC_BALANCED"})
FROZEN_ADAPTER_GROUPS = "meaningful_history_gate,public_card_residual"


class CampaignError(RuntimeError):
    """The Stage-C aligned learner campaign is invalid."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _value_sha256(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _load_json(path: Path, *, where: str) -> tuple[Path, dict[str, Any]]:
    lexical = path.expanduser()
    if lexical.is_symlink() or not lexical.is_file():
        raise CampaignError(f"{where} must be a regular file: {lexical}")
    resolved = lexical.resolve(strict=True)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CampaignError(f"cannot read {where}: {error}") from error
    if not isinstance(value, dict):
        raise CampaignError(f"{where} must contain one JSON object")
    return resolved, value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    rendered = json.dumps(value, indent=2, sort_keys=True) + "\n"
    destination = path.expanduser().resolve(strict=False)
    if destination.exists():
        if destination.is_symlink() or not destination.is_file():
            raise CampaignError(f"immutable output is not a file: {destination}")
        if destination.read_text(encoding="utf-8") != rendered:
            raise CampaignError(
                f"immutable output already exists with drift: {destination}"
            )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(rendered, encoding="utf-8")


def _load_plan(path: Path) -> tuple[Path, dict[str, Any]]:
    resolved, plan = _load_json(path, where="Stage-C learner campaign")
    unsigned = dict(plan)
    stated = unsigned.pop("campaign_sha256", None)
    if plan.get("schema_version") != SCHEMA or stated != _value_sha256(unsigned):
        raise CampaignError("Stage-C learner campaign schema/digest drifted")
    return resolved, plan


def _recipe() -> dict[str, Any]:
    return {
        "epochs": 1,
        "max_steps": MAX_STEPS,
        "lr": LR,
        "lr_warmup_steps": LR_WARMUP_STEPS,
        "policy_aux_active_batch_size": POLICY_AUX_ACTIVE_BATCH_SIZE,
        "policy_loss_weight": 1.0,
        "soft_target_source": "policy",
        "soft_target_weight": 1.0,
        "soft_target_min_legal_coverage": 1.0,
        "value_loss_weight": 0.25,
        "value_trunk_grad_scale": 0.1,
        "policy_kl_anchor_weight": 0.0,
        "public_card_lr_mult": 1.0,
        "per_game_policy_surprise_weighting": False,
        # The sealed ablation interface accepts a typed map but deliberately
        # refuses an empty override.  Explicit unit weights recover the old
        # all-forced-value-rows behavior without changing any other action
        # type (unspecified types also retain the 1.0 default).
        "forced_row_value_action_type_weights": "END_TURN=1,ROLL=1",
    }


def _one_dose_command(plan: Mapping[str, Any]) -> list[str]:
    inputs = plan["inputs"]
    output = Path(str(plan["output_root"])) / "learner"
    return [
        str(inputs["python"]),
        str(inputs["one_dose_trainer"]),
        "--lock",
        str(inputs["lock"]),
        "--data",
        str(inputs["data"]),
        "--validation-manifest",
        str(inputs["validation_manifest"]),
        "--coherent-corpus-admission",
        str(inputs["overlay_admission"]),
        "--architecture-upgrade-receipt",
        str(inputs["architecture_upgrade_receipt"]),
        "--independent-parent-authority",
        str(inputs["independent_parent_authority"]),
        "--checkpoint",
        str(output / "candidate.pt"),
        "--report",
        str(output / "train.report.json"),
        "--receipt",
        str(output / "one-dose.receipt.json"),
        "--python",
        str(inputs["python"]),
        "--gpu",
        "0",
        "--topology",
        "b200-8gpu-ddp",
        "--ddp-canary-receipt",
        str(inputs["ddp_canary_receipt"]),
        "--ablation-id",
        f"stage-c-{str(plan['arm']).lower().replace('_', '-')}",
        "--recipe-overrides-json",
        _canonical_bytes(plan["recipe"]).decode("ascii"),
        "--ablation-code-tree-sha256",
        str(inputs["reviewed_code_tree_sha256"]),
        "--reviewed-lock-file-sha256",
        str(inputs["reviewed_lock_file_sha256"]),
        "--diagnostic-dose-curve",
        "--diagnostic-checkpoint-steps",
        ",".join(map(str, INTERMEDIATE_CHECKPOINT_STEPS)),
    ]


def _plan(args: argparse.Namespace) -> dict[str, Any]:
    try:
        overlay_evidence = overlay.verify_overlay_admission(args.overlay_admission)
        admission_path, admission = stage_a._load_admission(  # noqa: SLF001
            args.overlay_admission
        )
    except (overlay.OverlayError, stage_a.CampaignError) as error:
        raise CampaignError(f"Stage-C overlay admission refused: {error}") from error
    corpus = admission["corpus"]
    arm = str(args.arm)
    sampling = overlay_evidence["receipt"].get("sampling_distribution")
    if (
        arm not in ARMS
        or not isinstance(sampling, dict)
        or sampling.get("schema_version") != overlay.SAMPLING_SCHEMA
        or sampling.get("arm") != arm
    ):
        raise CampaignError("campaign arm differs from overlay sampling distribution")
    data = Path(str(corpus["data_path"])).resolve(strict=True)
    validation = Path(str(corpus["validation_manifest"]["path"])).resolve(strict=True)
    lock = args.lock.expanduser().resolve(strict=True)
    try:
        python = stage_a.base_campaign._python_executable(args.python)  # noqa: SLF001
    except stage_a.base_campaign.CampaignError as error:
        raise CampaignError(f"learner Python refused: {error}") from error
    canary = args.ddp_canary_receipt.expanduser().resolve(strict=True)
    upgrade_path = args.architecture_upgrade_receipt.expanduser().resolve(strict=True)
    if any(
        path.is_symlink() or not path.is_file() for path in (lock, canary, upgrade_path)
    ):
        raise CampaignError("lock/canary/upgrade inputs must be regular files")
    try:
        verified = one_dose.verify_training_inputs(
            lock_path=lock,
            data_path=data,
            validation_path=validation,
            reviewed_lock_file_sha256=_file_sha256(lock),
            coherent_corpus_admission=admission_path,
        )
        upgrade = architecture_upgrade.verify_receipt(upgrade_path)
    except (one_dose.ExecutorError, architecture_upgrade.UpgradeError) as error:
        raise CampaignError(f"Stage-C learner input refused: {error}") from error
    if (
        verified.get("data_kind") != "coherent_direct_memmap_v1"
        or verified.get("recipe", {}).get("soft_target_source") != "policy"
        or float(verified.get("recipe", {}).get("policy_loss_weight", -1.0)) != 1.0
        or float(verified.get("recipe", {}).get("value_loss_weight", -1.0)) != 0.25
        or float(verified.get("recipe", {}).get("policy_kl_anchor_weight", -1.0)) != 0.0
        or verified.get("recipe", {}).get("policy_kl_target") is not None
        or upgrade.get("source", {}).get("sha256") != stage_a.EXPECTED_F7_PARENT_SHA256
        or upgrade.get("module")
        != architecture_upgrade.MODULE_PUBLIC_CARD_COUNT_MEANINGFUL_HISTORY_V2
        or upgrade.get("forward_identical_at_init") is not True
        or float(upgrade.get("forward_max_diff", -1.0)) != 0.0
        or upgrade.get("shared_parameters_bit_identical") is not True
    ):
        raise CampaignError("campaign lost exact f7 function-preserving initialization")

    output_root = args.output_root.expanduser().resolve(strict=False)
    authority_path = output_root / "independent-parent.authority.json"
    authority = stage_a._parent_authority(  # noqa: SLF001
        verified=verified,
        upgrade=upgrade,
        admission_path=admission_path,
        admission=admission,
    )
    _write_json(authority_path, authority)
    code_binding = one_dose._current_ablation_code_binding(verified["lock"])  # noqa: SLF001
    selected_roots_total = int(
        overlay_evidence["receipt"]["projection"]["selected_rows"]
    )
    selected_training_roots = int(
        overlay_evidence["receipt"]["projection"]["selected_training_policy_rows"]
    )
    if selected_roots_total <= 0 or selected_training_roots <= 0:
        raise CampaignError("Stage-C overlay has no policy roots")
    trajectory = []
    for step in CHECKPOINT_STEPS:
        aux_draws = POLICY_AUX_ACTIVE_BATCH_SIZE * WORLD_SIZE * step
        trajectory.append(
            {
                "step": step,
                "auxiliary_policy_draws": aux_draws,
                "auxiliary_policy_epochs": aux_draws / selected_training_roots,
                "base_policy_draws_reported_posthoc": True,
            }
        )

    trainer = (REPO_ROOT / "tools" / "a1_one_dose_train.py").resolve(strict=True)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA,
        "purpose": "distil_exact_current_coherent_n128_stage_c_targets",
        "arm": arm,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "lineage": {
            "corpus_producer_sha256": verified["producer"]["sha256"],
            "learner_parent_sha256": stage_a.EXPECTED_F7_PARENT_SHA256,
            "initializer_sha256": upgrade["upgraded_initializer"]["sha256"],
            "fresh_adam": True,
            "candidate_chaining": False,
        },
        "policy_target_contract": {
            "target_policy_target_identity_sha256": overlay_evidence["receipt"][
                "target_policy_target_identity_sha256"
            ],
            "selected_unique_roots_total": selected_roots_total,
            "selected_unique_training_roots": selected_training_roots,
            "historical_policy_targets_active": False,
            "nonselected_policy_weight": 0.0,
            "base_value_rows_retained": True,
            "surprise_weighting": False,
            "sampling_distribution": copy.deepcopy(sampling),
        },
        "topology": {
            "name": "b200-8gpu-ddp",
            "world_size": WORLD_SIZE,
            "local_batch_size": LOCAL_BATCH_SIZE,
            "global_batch_size": GLOBAL_BATCH_SIZE,
        },
        "recipe": _recipe(),
        "checkpoint_trajectory": trajectory,
        "selection_contract": {
            "optimizer_trust_controller_enabled": False,
            "reason": (
                "historical optimizer-batch KL meter did not enforce posthoc holdout KL"
            ),
            "metric_scope": "frozen_whole_game_validation_policy_active_multi_action_rows",
            "checkpoint_steps": list(CHECKPOINT_STEPS),
            "max_parent_kl": MAX_PARENT_KL,
            "max_trunk_relative_l2": MAX_TRUNK_RELATIVE_L2,
            "requires_positive_teacher_gap_closure": True,
            "objective": "best_external_play_among_posthoc_in_budget_checkpoints",
            "playing_strength_evaluation_required": True,
        },
        "optimizer_surface_contract": {
            "shared_trunk_trainable": True,
            "value_trunk_grad_scale": 0.1,
            "frozen_adapter_groups": sorted(FROZEN_ADAPTER_GROUPS.split(",")),
            "frozen_adapters_optimizer_excluded": True,
        },
        "inputs": {
            "python": str(python),
            "one_dose_trainer": str(trainer),
            "one_dose_trainer_sha256": _file_sha256(trainer),
            "lock": str(lock),
            "reviewed_lock_file_sha256": _file_sha256(lock),
            "data": str(data),
            "corpus_meta_file_sha256": _file_sha256(data / "corpus_meta.json"),
            "payload_inventory_sha256": corpus["payload_inventory_sha256"],
            "validation_manifest": str(validation),
            "validation_manifest_file_sha256": _file_sha256(validation),
            "overlay_admission": str(admission_path),
            "overlay_admission_file_sha256": _file_sha256(admission_path),
            "overlay_admission_sha256": admission["admission_sha256"],
            "overlay_materialization_receipt": admission["stage_c_policy_overlay"][
                "materialization_receipt"
            ]["path"],
            "overlay_materialization_receipt_sha256": overlay_evidence["receipt"][
                "receipt_sha256"
            ],
            "architecture_upgrade_receipt": str(upgrade_path),
            "architecture_upgrade_receipt_file_sha256": _file_sha256(upgrade_path),
            "independent_parent_authority": str(authority_path),
            "independent_parent_authority_file_sha256": _file_sha256(authority_path),
            "independent_parent_authority_sha256": authority["authority_sha256"],
            "ddp_canary_receipt": str(canary),
            "ddp_canary_receipt_file_sha256": _file_sha256(canary),
            "reviewed_code_tree_sha256": code_binding["code_tree_sha256"],
        },
        "output_root": str(output_root),
        "expected_artifacts": {
            "terminal_checkpoint": str(output_root / "learner" / "candidate.pt"),
            "intermediate_checkpoints": [
                str(output_root / "learner" / f"candidate_step{step:04d}.pt")
                for step in INTERMEDIATE_CHECKPOINT_STEPS
            ],
            "report": str(output_root / "learner" / "train.report.json"),
            "one_dose_receipt": str(output_root / "learner" / "one-dose.receipt.json"),
            "execution_receipt": str(output_root / "learner.execution.receipt.json"),
            "fingerprint": str(output_root / "fingerprint.json"),
        },
    }
    payload["command"] = _one_dose_command(payload)
    payload["command_sha256"] = _value_sha256(payload["command"])
    payload["campaign_sha256"] = _value_sha256(payload)
    return payload


def _verify_inputs(plan: Mapping[str, Any]) -> None:
    inputs = plan["inputs"]
    checks = (
        ("one_dose_trainer", "one_dose_trainer_sha256"),
        ("lock", "reviewed_lock_file_sha256"),
        ("validation_manifest", "validation_manifest_file_sha256"),
        ("overlay_admission", "overlay_admission_file_sha256"),
        ("architecture_upgrade_receipt", "architecture_upgrade_receipt_file_sha256"),
        ("independent_parent_authority", "independent_parent_authority_file_sha256"),
        ("ddp_canary_receipt", "ddp_canary_receipt_file_sha256"),
    )
    for path_key, sha_key in checks:
        path = Path(str(inputs[path_key])).resolve(strict=True)
        if path.is_symlink() or _file_sha256(path) != inputs[sha_key]:
            raise CampaignError(f"campaign input bytes changed: {path_key}")
    overlay.verify_overlay_admission(Path(str(inputs["overlay_admission"])))
    recipe = plan.get("recipe", {})
    if (
        plan.get("arm") not in ARMS
        or float(recipe.get("value_trunk_grad_scale", -1.0)) != 0.1
        or float(recipe.get("soft_target_min_legal_coverage", -1.0)) != 1.0
        or float(recipe.get("policy_kl_anchor_weight", -1.0)) != 0.0
        or bool(recipe.get("per_game_policy_surprise_weighting", True))
    ):
        raise CampaignError("Stage-C clean learner semantics drifted")
    if plan.get("command_sha256") != _value_sha256(_one_dose_command(plan)):
        raise CampaignError("Stage-C learner command drifted from campaign")


def _run(plan: Mapping[str, Any], *, go: bool) -> dict[str, Any]:
    _verify_inputs(plan)
    command = _one_dose_command(plan)
    if not go:
        return {"mode": "dry-run", "command": command}
    receipt_path = Path(str(plan["expected_artifacts"]["one_dose_receipt"]))
    adopted_completed_receipt = receipt_path.exists() or receipt_path.is_symlink()
    try:
        if adopted_completed_receipt:
            # The trainer may have completed and durably receipted its exact
            # dose before this outer campaign process was interrupted or hit a
            # post-training wrapper bug. Authenticate first and never launch a
            # second optimizer trajectory into that terminal namespace.
            receipt = one_dose._load_authenticated_completed_ablation_receipt(  # noqa: SLF001
                receipt_path
            )
        else:
            result = subprocess.run([*command, "--go"], check=False)
            if result.returncode != 0:
                raise CampaignError(
                    f"Stage-C aligned learner exited {result.returncode}"
                )
            receipt = one_dose._load_authenticated_completed_ablation_receipt(  # noqa: SLF001
                receipt_path
            )
    except one_dose.ExecutorError as error:
        raise CampaignError(
            f"completed Stage-C learner receipt refused: {error}"
        ) from error
    report_path = Path(str(plan["expected_artifacts"]["report"])).resolve(strict=True)
    report = _load_json(report_path, where="completed Stage-C learner report")[1]
    freeze = report.get("training_information_surface", {}).get(
        "explicit_module_freeze"
    )
    if (
        float(report.get("value_trunk_grad_scale", -1.0)) != 0.1
        or not isinstance(freeze, dict)
        or freeze.get("frozen_groups") != sorted(FROZEN_ADAPTER_GROUPS.split(","))
        or set(freeze.get("frozen_submodules", ()))
        != {"meaningful_history_residual_gate", "public_card_count_residual"}
        or freeze.get("all_require_grad_false") is not True
        or int(freeze.get("optimizer_excluded_parameter_tensors", 0)) <= 0
    ):
        raise CampaignError("completed learner did not exclude both new adapters")
    aux_draws = int(report.get("policy_aux_active_rows", -1))
    unique_rows = int(report.get("policy_aux_unique_source_rows", -1))
    expected_aux_draws = POLICY_AUX_ACTIVE_BATCH_SIZE * WORLD_SIZE * MAX_STEPS
    selected_roots = int(
        plan["policy_target_contract"]["selected_unique_training_roots"]
    )
    if (
        aux_draws != expected_aux_draws
        or unique_rows <= 0
        or unique_rows > selected_roots
        or not math.isclose(
            float(report.get("policy_aux_reuse_factor", math.nan)),
            aux_draws / unique_rows,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise CampaignError("completed learner unique-root exposure drifted")
    execution: dict[str, Any] = {
        "schema_version": EXECUTION_SCHEMA,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "campaign_sha256": plan["campaign_sha256"],
        "one_dose_receipt": {
            "path": str(receipt_path),
            "file_sha256": _file_sha256(receipt_path),
            "receipt_sha256": receipt["receipt_sha256"],
        },
        "report": {
            "path": str(report_path),
            "file_sha256": _file_sha256(report_path),
        },
        "checkpoint": {
            "path": receipt["outputs"]["checkpoint"],
            "sha256": receipt["outputs"]["checkpoint_sha256"],
        },
        "policy_exposure": {
            "selected_unique_training_root_population": selected_roots,
            "selected_unique_root_population_total": int(
                plan["policy_target_contract"]["selected_unique_roots_total"]
            ),
            "auxiliary_draws": aux_draws,
            "unique_auxiliary_source_rows": unique_rows,
            "unique_root_coverage_fraction": unique_rows / selected_roots,
            "auxiliary_reuse_factor": aux_draws / unique_rows,
            "base_policy_active_draws": int(report.get("policy_base_active_rows", 0)),
        },
        "optimizer_batch_kl_used_as_trust_authority": False,
        "posthoc_frozen_holdout_selection_required": True,
        "existing_completed_dose_adopted": adopted_completed_receipt,
    }
    execution["execution_sha256"] = _value_sha256(execution)
    execution_path = Path(str(plan["expected_artifacts"]["execution_receipt"]))
    _write_json(execution_path, execution)
    return {
        "mode": "finalize-existing" if adopted_completed_receipt else "go",
        "receipt": str(receipt_path),
        "receipt_sha256": receipt["receipt_sha256"],
        "checkpoint": receipt["outputs"]["checkpoint"],
        "checkpoint_sha256": receipt["outputs"]["checkpoint_sha256"],
        "execution_receipt": str(execution_path),
        "execution_sha256": execution["execution_sha256"],
        "policy_exposure": execution["policy_exposure"],
    }


def _checkpoint_path(plan: Mapping[str, Any], step: int) -> Path:
    root = Path(str(plan["output_root"])) / "learner"
    return (
        root / "candidate.pt"
        if step == MAX_STEPS
        else root / f"candidate_step{step:04d}.pt"
    )


def _trunk_relative_l2(report: Mapping[str, Any]) -> float:
    groups = report.get("groups")
    if not isinstance(groups, dict):
        raise CampaignError("layer drift report has no groups")
    selected = [
        row
        for name, row in groups.items()
        if isinstance(row, dict)
        and (
            name in {"input_encoders", "shared", "topology_adapter"}
            or name.startswith("transformer_block_")
        )
    ]
    baseline = sum(float(row["baseline_l2"]) ** 2 for row in selected)
    delta = sum(float(row["delta_energy"]) for row in selected)
    if not selected or baseline <= 0.0 or delta < 0.0:
        raise CampaignError("layer drift cannot define trunk relative L2")
    return math.sqrt(delta / baseline)


def _fingerprint(
    plan_path: Path, plan: Mapping[str, Any], *, go: bool, device: str
) -> dict[str, Any]:
    _verify_inputs(plan)
    output_root = Path(str(plan["output_root"])) / "fingerprints"
    report = Path(str(plan["expected_artifacts"]["report"])).resolve(strict=True)
    report_payload = _load_json(report, where="completed Stage-C learner report")[1]
    emitted_holdout = report_payload.get("validation_game_seed_manifest")
    if not isinstance(emitted_holdout, str) or not emitted_holdout:
        raise CampaignError(
            "completed Stage-C learner report has no emitted validation holdout"
        )
    validation_manifest = Path(emitted_holdout).expanduser()
    if not validation_manifest.is_absolute():
        validation_manifest = report.parent / validation_manifest
    if validation_manifest.is_symlink() or not validation_manifest.is_file():
        raise CampaignError("emitted validation holdout must be a regular file")
    validation_manifest = validation_manifest.resolve(strict=True)
    validation_payload = _load_json(
        validation_manifest, where="emitted Stage-C validation holdout"
    )[1]
    expected_input_manifest = Path(str(plan["inputs"]["validation_manifest"])).resolve(
        strict=True
    )
    if (
        validation_payload.get("schema_version") != "train-validation-game-seeds-v1"
        or validation_payload.get("a1_contract_sha256")
        != report_payload.get("a1_contract_sha256")
        or validation_payload.get("data") != report_payload.get("data")
        or validation_payload.get("data_fingerprint")
        != report_payload.get("data_fingerprint")
        or validation_payload.get("validation_game_seed_count")
        != report_payload.get("validation_game_seed_count")
        or validation_payload.get("validation_game_seed_set_sha256")
        != report_payload.get("validation_game_seed_set_sha256")
        or validation_payload.get("training_excluded_game_seed_count")
        != report_payload.get("training_excluded_game_seed_count")
        or validation_payload.get("training_excluded_game_seed_set_sha256")
        != report_payload.get("training_excluded_game_seed_set_sha256")
        or validation_payload.get("input_validation_game_seed_manifest")
        != str(expected_input_manifest)
        or validation_payload.get("input_validation_game_seed_manifest_sha256")
        != report_payload.get("input_validation_game_seed_manifest_sha256")
        or validation_payload.get("input_validation_game_seed_manifest_sha256")
        != _file_sha256(expected_input_manifest)
    ):
        raise CampaignError(
            "emitted validation holdout differs from the completed learner report"
        )
    validation_binding = {
        "path": str(validation_manifest),
        "file_sha256": _file_sha256(validation_manifest),
        "validation_game_seed_count": validation_payload["validation_game_seed_count"],
        "validation_game_seed_set_sha256": validation_payload[
            "validation_game_seed_set_sha256"
        ],
        "training_excluded_game_seed_set_sha256": validation_payload[
            "training_excluded_game_seed_set_sha256"
        ],
        "input_validation_game_seed_manifest": str(expected_input_manifest),
        "input_validation_game_seed_manifest_sha256": _file_sha256(
            expected_input_manifest
        ),
    }
    authority = _load_json(
        Path(str(plan["inputs"]["independent_parent_authority"])),
        where="independent parent authority",
    )[1]
    parent = Path(
        str(authority["function_preserving_upgrade"]["upgraded_initializer"]["path"])
    ).resolve(strict=True)
    records = []
    commands = []
    for step in CHECKPOINT_STEPS:
        checkpoint = _checkpoint_path(plan, step).resolve(strict=True)
        functional_path = output_root / f"step{step:04d}.functional.json"
        drift_path = output_root / f"step{step:04d}.drift.json"
        functional_command = [
            str(plan["inputs"]["python"]),
            str(REPO_ROOT / "tools" / "posthoc_teacher_gap_probe.py"),
            "--report",
            str(report),
            "--checkpoint",
            str(checkpoint),
            "--parent-checkpoint",
            str(parent),
            "--data",
            str(plan["inputs"]["data"]),
            "--validation-manifest",
            str(validation_manifest),
            "--device",
            device,
            "--output",
            str(functional_path),
        ]
        drift_command = [
            str(plan["inputs"]["python"]),
            str(REPO_ROOT / "tools" / "audit_checkpoint_layer_drift.py"),
            "--baseline",
            str(parent),
            "--candidate",
            str(checkpoint),
            "--output",
            str(drift_path),
        ]
        commands.append(
            {"step": step, "functional": functional_command, "drift": drift_command}
        )
        if not go:
            continue
        output_root.mkdir(parents=True, exist_ok=True)
        for command in (functional_command, drift_command):
            result = subprocess.run(command, check=False)
            if result.returncode != 0:
                raise CampaignError(
                    f"checkpoint {step} fingerprint exited {result.returncode}"
                )
        functional = _load_json(functional_path, where=f"step {step} functional")[1]
        drift = _load_json(drift_path, where=f"step {step} drift")[1]
        fingerprint = functional.get("functional_dose_fingerprint")
        if not isinstance(fingerprint, dict):
            raise CampaignError(f"checkpoint {step} has no functional fingerprint")
        parent_kl = float(fingerprint["kl_parent_candidate_mean"])
        closure = float(functional["teacher_gap"]["active_policy_teacher_gap_closure"])
        trunk = _trunk_relative_l2(drift)
        records.append(
            {
                "step": step,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _file_sha256(checkpoint),
                "parent_kl": parent_kl,
                "teacher_gap_closure": closure,
                "trunk_relative_l2": trunk,
                "eligible": bool(
                    parent_kl <= MAX_PARENT_KL
                    and trunk <= MAX_TRUNK_RELATIVE_L2
                    and closure > 0.0
                ),
                "functional": {
                    "path": str(functional_path),
                    "file_sha256": _file_sha256(functional_path),
                },
                "drift": {
                    "path": str(drift_path),
                    "file_sha256": _file_sha256(drift_path),
                },
            }
        )
    if not go:
        return {
            "mode": "dry-run",
            "validation_holdout": validation_binding,
            "commands": commands,
        }
    eligible = [row for row in records if row["eligible"]]
    winner = (
        max(eligible, key=lambda row: (row["teacher_gap_closure"], -row["step"]))
        if eligible
        else None
    )
    payload: dict[str, Any] = {
        "schema_version": FINGERPRINT_SCHEMA,
        "campaign": {
            "path": str(plan_path),
            "file_sha256": _file_sha256(plan_path),
            "campaign_sha256": plan["campaign_sha256"],
        },
        "metric_scope": "frozen_whole_game_validation_policy_active_multi_action_rows",
        "validation_holdout": validation_binding,
        "optimizer_batch_kl_used_as_trust_authority": False,
        "checkpoints": records,
        "winner": winner,
        "formal_result": (
            "posthoc_in_budget_candidate_requires_playing_evaluation"
            if winner is not None
            else "no_positive_closure_checkpoint_within_posthoc_trust_budget"
        ),
    }
    payload["fingerprint_sha256"] = _value_sha256(payload)
    _write_json(Path(str(plan["expected_artifacts"]["fingerprint"])), payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--arm", required=True, choices=sorted(ARMS))
    plan.add_argument("--overlay-admission", required=True, type=Path)
    plan.add_argument("--lock", required=True, type=Path)
    plan.add_argument("--architecture-upgrade-receipt", required=True, type=Path)
    plan.add_argument("--ddp-canary-receipt", required=True, type=Path)
    plan.add_argument("--python", required=True, type=Path)
    plan.add_argument("--output-root", required=True, type=Path)
    plan.add_argument("--write", required=True, type=Path)
    run = commands.add_parser("run")
    run.add_argument("--campaign", required=True, type=Path)
    run.add_argument("--go", action="store_true")
    fingerprint = commands.add_parser("fingerprint")
    fingerprint.add_argument("--campaign", required=True, type=Path)
    fingerprint.add_argument("--device", default="cuda:0")
    fingerprint.add_argument("--go", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "plan":
            result = _plan(args)
            _write_json(args.write, result)
        else:
            campaign_path, campaign = _load_plan(args.campaign)
            result = (
                _run(campaign, go=args.go)
                if args.command == "run"
                else _fingerprint(
                    campaign_path,
                    campaign,
                    go=args.go,
                    device=args.device,
                )
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (CampaignError, overlay.OverlayError, OSError, ValueError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
