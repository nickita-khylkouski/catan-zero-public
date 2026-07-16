#!/usr/bin/env python3
"""Run the independent-parent 8xB200 A1 LR/dose campaign.

The four 128-step arms differ only in LR and warmup. Every arm replays the
sealed one-dose transaction from the same explicitly hash-bound parent with a
fresh optimizer. After playing-strength evaluation, the selector authenticates
each arm against both f7 and v5 and chooses the recipe with the strongest
worst-baseline lower confidence bound. That recipe may be replayed to 256 steps
from the original parent; a candidate checkpoint is never used as another
candidate's initializer.

This tool deliberately does not choose a production champion. It is a
diagnostic campaign runner around :mod:`tools.a1_one_dose_train`; production
lineage remains owned by the sealed post-promotion handoff.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


SCHEMA = "a1-b200-lr-dose-campaign-v1"
SELECTION_SCHEMA = "a1-b200-lr-dose-selection-v2"
NATIVE_EVAL_SUMMARY_SCHEMA = "a1-r5-native-eval-summary-v1"
EVALUATION_BASELINE_ROLES = ("f7", "v5")
SELECTION_OBJECTIVE = (
    "maximize_minimum_baseline_paired_score_regularized_95ci_lower"
)
SELECTION_SECONDARY = "maximize_minimum_baseline_paired_score_regularized_mu"
WORLD_SIZE = 8
GLOBAL_BATCH_SIZE = 4096
SHORT_STEPS = 128
LONG_STEPS = 256
SCIENCE_CONTRACT_RELATIVE = Path(
    "configs/operations/a1-next-wave-coherent-public-v3/science.contract.json"
)
EXISTING_CORPUS_SCIENCE_FIELDS = {
    "forced_row_value_action_type_weights": "END_TURN=0.1,ROLL=1.0",
    # These are omitted from the canonical recipe when they retain the trainer
    # defaults. Bind the effective values so the campaign cannot resurrect an
    # obsolete experimental optimizer/sampler treatment merely because the
    # fields are absent.
    "per_game_policy_surprise_weighting": False,
    "public_card_lr_mult": 1.0,
}
ARMS: dict[str, dict[str, int | float]] = {
    "A": {"lr": 3e-5, "lr_warmup_steps": 100},
    "B": {"lr": 3e-5, "lr_warmup_steps": 16},
    "C": {"lr": 6e-5, "lr_warmup_steps": 16},
    "D": {"lr": 1.2e-4, "lr_warmup_steps": 16},
}


class CampaignError(RuntimeError):
    pass


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _value_sha256(value: object) -> str:
    return f"sha256:{hashlib.sha256(_canonical_bytes(value)).hexdigest()}"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _regular_file(path: Path, *, where: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise CampaignError(f"{where} must be a regular file: {resolved}")
    return resolved


def _python_executable(path: Path) -> Path:
    """Preserve a virtualenv's lexical executable while authenticating target."""

    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    try:
        target = lexical.resolve(strict=True)
    except OSError as error:
        raise CampaignError(f"cannot resolve Python executable {lexical}: {error}") from error
    if not target.is_file() or not os.access(lexical, os.X_OK):
        raise CampaignError(f"Python executable is not runnable: {lexical}")
    return lexical


def _normalize_sha256(value: str, *, where: str) -> str:
    text = str(value).strip().lower()
    if not text.startswith("sha256:"):
        text = f"sha256:{text}"
    if len(text) != 71 or any(ch not in "0123456789abcdef" for ch in text[7:]):
        raise CampaignError(f"{where} must be one SHA-256 digest")
    return text


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    with tmp.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _load_bound_json(path: Path, *, schema: str) -> dict[str, Any]:
    resolved = _regular_file(path, where=schema)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CampaignError(f"cannot load {schema}: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != schema:
        raise CampaignError(f"{resolved} is not {schema}")
    stated = payload.get("campaign_sha256" if schema == SCHEMA else "selection_sha256")
    unsigned = dict(payload)
    unsigned.pop("campaign_sha256" if schema == SCHEMA else "selection_sha256", None)
    if stated != _value_sha256(unsigned):
        raise CampaignError(f"{schema} digest drift")
    return payload


def _arm_overrides(
    arm: str,
    *,
    max_steps: int,
    policy_aux_active_batch_size: int,
    science_recipe: Mapping[str, Any],
) -> dict[str, object]:
    if arm not in ARMS:
        raise CampaignError(f"unknown campaign arm {arm!r}")
    recipe: dict[str, object] = {
        **{
            key: science_recipe.get(key, expected)
            for key, expected in EXISTING_CORPUS_SCIENCE_FIELDS.items()
        },
        "epochs": 1,
        "max_steps": int(max_steps),
        "lr": float(ARMS[arm]["lr"]),
        "lr_warmup_steps": int(ARMS[arm]["lr_warmup_steps"]),
        "lr_schedule": "flat",
    }
    if policy_aux_active_batch_size > 0:
        recipe["policy_aux_active_batch_size"] = int(
            policy_aux_active_batch_size
        )
    return recipe


def _one_dose_invocation(
    campaign: Mapping[str, Any],
    *,
    arm: str,
    max_steps: int,
    suffix: str,
) -> list[str]:
    shared = campaign["inputs"]
    output = Path(campaign["output_root"]) / suffix
    overrides = _arm_overrides(
        arm,
        max_steps=max_steps,
        policy_aux_active_batch_size=int(
            campaign["policy_active_dose"]["policy_aux_active_batch_size"]
        ),
        science_recipe=campaign["canonical_learner_projection"][
            "training_recipe"
        ],
    )
    ablation_id = f"lr-dose-{arm.lower()}-steps{max_steps}"
    return [
        str(shared["python"]),
        str(shared["one_dose_trainer"]),
        "--lock",
        str(shared["lock"]),
        "--data",
        str(shared["data"]),
        "--composite-build-receipt",
        str(shared["composite_build_receipt"]),
        "--architecture-upgrade-receipt",
        str(shared["architecture_upgrade_receipt"]),
        "--checkpoint",
        str(output / "candidate.pt"),
        "--report",
        str(output / "train.report.json"),
        "--receipt",
        str(output / "one-dose.receipt.json"),
        "--python",
        str(shared["python"]),
        "--gpu",
        "0",
        "--topology",
        "b200-8gpu-ddp",
        "--ddp-canary-receipt",
        str(shared["ddp_canary_receipt"]),
        "--ablation-id",
        ablation_id,
        "--recipe-overrides-json",
        _canonical_bytes(overrides).decode("ascii"),
        "--ablation-code-tree-sha256",
        str(shared["reviewed_code_tree_sha256"]),
        "--reviewed-lock-file-sha256",
        str(shared["reviewed_lock_file_sha256"]),
        "--diagnostic-dose-curve",
    ]


def _plan(args: argparse.Namespace) -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[1]
    trainer = _regular_file(
        repo / "tools" / "a1_one_dose_train.py", where="one-dose trainer"
    )
    python = _python_executable(args.python)
    lock = _regular_file(args.lock, where="sealed learner lock")
    data = _regular_file(args.data, where="post-wave composite descriptor")
    composite = _regular_file(
        args.composite_build_receipt, where="composite build receipt"
    )
    upgrade = _regular_file(
        args.architecture_upgrade_receipt, where="architecture upgrade receipt"
    )
    canary = _regular_file(args.ddp_canary_receipt, where="8xB200 DDP canary")
    science_contract = _regular_file(
        repo / SCIENCE_CONTRACT_RELATIVE, where="canonical science contract"
    )
    try:
        science_payload = json.loads(science_contract.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CampaignError(f"cannot load canonical science contract: {error}") from error
    learner = science_payload.get("learner") if isinstance(science_payload, dict) else None
    science_recipe = learner.get("training_recipe") if isinstance(learner, dict) else None
    if not isinstance(science_recipe, dict):
        raise CampaignError("canonical science contract has no learner training recipe")
    science_drift = {
        key: {
            "expected": expected,
            "actual": science_recipe.get(key, expected),
        }
        for key, expected in EXISTING_CORPUS_SCIENCE_FIELDS.items()
        if science_recipe.get(key, expected) != expected
    }
    if science_drift:
        raise CampaignError(
            "canonical existing-corpus learner semantics drifted: "
            + json.dumps(science_drift, sort_keys=True)
        )
    expected_parent = _normalize_sha256(
        args.expected_parent_sha256, where="expected diagnostic parent"
    )
    reviewed_code = _normalize_sha256(
        args.reviewed_code_tree_sha256, where="reviewed code tree"
    )
    lock_sha = _file_sha256(lock)
    if args.reviewed_lock_file_sha256:
        expected_lock = _normalize_sha256(
            args.reviewed_lock_file_sha256, where="reviewed lock file"
        )
        if lock_sha != expected_lock:
            raise CampaignError(
                f"reviewed lock digest mismatch: expected={expected_lock} actual={lock_sha}"
            )

    target_active = int(args.target_policy_active_rows)
    observed_fraction = float(args.observed_base_policy_active_fraction)
    if target_active < 0 or not math.isfinite(observed_fraction) or not (
        0.0 <= observed_fraction <= 1.0
    ):
        raise CampaignError("policy-active dose inputs are invalid")
    base_draws = GLOBAL_BATCH_SIZE * SHORT_STEPS
    estimated_base_active = int(round(base_draws * observed_fraction))
    if target_active > 0:
        if observed_fraction <= 0.0:
            raise CampaignError(
                "--target-policy-active-rows requires an observed positive base fraction"
            )
        aux_local = max(
            0,
            math.ceil(
                (target_active - estimated_base_active)
                / (WORLD_SIZE * SHORT_STEPS)
            ),
        )
    else:
        aux_local = int(args.policy_aux_active_batch_size)
    if aux_local < 0 or aux_local > 512:
        raise CampaignError(
            "derived --policy-aux-active-batch-size must be in [0,512]"
        )
    estimated_total_active = estimated_base_active + (
        aux_local * WORLD_SIZE * SHORT_STEPS
    )

    output_root = args.output_root.expanduser().resolve(strict=False)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA,
        "purpose": "diagnostic_lr_warmup_and_same_parent_dose_curve",
        "production_promotion_eligible": False,
        "lineage_contract": {
            "expected_parent_sha256": expected_parent,
            "parent_is_operator_supplied_not_a_production_default": True,
            "every_arm_restarts_from_expected_parent": True,
            "fresh_optimizer_every_arm": True,
            "candidate_chaining_forbidden": True,
            "production_parent_owned_by_sealed_post_promotion_handoff": True,
        },
        "topology": {
            "name": "b200-8gpu-ddp",
            "world_size": WORLD_SIZE,
            "local_batch_size": 512,
            "global_batch_size": GLOBAL_BATCH_SIZE,
        },
        "short_dose": {
            "optimizer_steps": SHORT_STEPS,
            "base_row_draws": base_draws,
            "intermediate_checkpoint_steps": [64],
            "terminal_checkpoint_step": 128,
        },
        "winner_dose": {
            "optimizer_steps": LONG_STEPS,
            "replay_from_original_parent": True,
            "intermediate_checkpoint_steps": [64, 128],
            "terminal_checkpoint_step": 256,
        },
        "policy_active_dose": {
            "observed_base_policy_active_fraction": observed_fraction,
            "target_policy_active_rows": target_active or None,
            "policy_aux_active_batch_size": aux_local,
            "estimated_base_policy_active_rows": estimated_base_active,
            "estimated_total_policy_active_rows": estimated_total_active,
            "selection_uses_realized_report_not_estimate": True,
        },
        "canonical_learner_projection": {
            "science_contract": str(science_contract),
            "science_contract_file_sha256": _file_sha256(science_contract),
            "contract_id": science_payload.get("contract_id"),
            "training_recipe": dict(science_recipe),
            "training_recipe_sha256": _value_sha256(science_recipe),
            "existing_corpus_architecture_scope": "authenticated_card_only_v1",
            "meaningful_history_v2_enabled": False,
        },
        "arms": {
            arm: {
                **values,
                "max_steps": SHORT_STEPS,
                "recipe_overrides": _arm_overrides(
                    arm,
                    max_steps=SHORT_STEPS,
                    policy_aux_active_batch_size=aux_local,
                    science_recipe=science_recipe,
                ),
                "output_subdir": f"arms/{arm}",
            }
            for arm, values in ARMS.items()
        },
        "selection_contract": {
            "primary": SELECTION_OBJECTIVE,
            "secondary": SELECTION_SECONDARY,
            "required_baselines": list(EVALUATION_BASELINE_ROLES),
            "evaluation_summary_schema": NATIVE_EVAL_SUMMARY_SCHEMA,
            "validation_loss_may_select_checkpoint_within_arm_only": True,
            "all_four_one_dose_receipts_required": True,
            "all_four_evaluation_receipts_required": True,
            "only_selected_recipe_may_run_256_steps": True,
        },
        "reporting_contract": {
            "training_strata_dose_schema": "training-strata-dose-v1",
            "required_dimensions": [
                "draw_stream",
                "full_vs_fast",
                "simulation_budget",
                "decision_class",
                "legal_width",
                "phase",
                "fresh_vs_replay",
            ],
            "module_observability_schema": "module-optimizer-observability-v1",
            "module_diagnostics_cadence_batches": 16,
        },
        "inputs": {
            "python": str(python),
            "one_dose_trainer": str(trainer),
            "lock": str(lock),
            "lock_file_sha256": lock_sha,
            "data": str(data),
            "composite_build_receipt": str(composite),
            "composite_build_receipt_sha256": _file_sha256(composite),
            "architecture_upgrade_receipt": str(upgrade),
            "architecture_upgrade_receipt_sha256": _file_sha256(upgrade),
            "ddp_canary_receipt": str(canary),
            "ddp_canary_receipt_sha256": _file_sha256(canary),
            "science_contract": str(science_contract),
            "science_contract_file_sha256": _file_sha256(science_contract),
            "reviewed_code_tree_sha256": reviewed_code,
            "reviewed_lock_file_sha256": lock_sha,
        },
        "output_root": str(output_root),
    }
    payload["commands"] = {
        arm: _one_dose_invocation(
            payload, arm=arm, max_steps=SHORT_STEPS, suffix=f"arms/{arm}"
        )
        for arm in ARMS
    }
    payload["campaign_sha256"] = _value_sha256(payload)
    return payload


def _verify_input_bytes(campaign: Mapping[str, Any]) -> None:
    inputs = campaign["inputs"]
    for path_key, digest_key in (
        ("lock", "lock_file_sha256"),
        ("composite_build_receipt", "composite_build_receipt_sha256"),
        ("architecture_upgrade_receipt", "architecture_upgrade_receipt_sha256"),
        ("ddp_canary_receipt", "ddp_canary_receipt_sha256"),
        ("science_contract", "science_contract_file_sha256"),
    ):
        actual = _file_sha256(_regular_file(Path(inputs[path_key]), where=path_key))
        if actual != inputs[digest_key]:
            raise CampaignError(
                f"campaign input changed: {path_key} expected={inputs[digest_key]} actual={actual}"
            )


def _option(command: Sequence[str], flag: str) -> str:
    positions = [index for index, value in enumerate(command) if value == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise CampaignError(f"rendered one-dose command lost {flag}")
    return str(command[positions[0] + 1])


def _one_dose_dry_run(invocation: list[str]) -> dict[str, Any]:
    result = subprocess.run(
        invocation,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise CampaignError(
            "one-dose dry run refused: " + (result.stderr.strip() or result.stdout.strip())
        )
    # The sealed verifier emits JSON progress records before the final, pretty-
    # printed one-dose plan.  Decode the JSON stream and bind the last document
    # that is actually a command-bearing plan instead of requiring silent
    # verifier stdout.
    decoder = json.JSONDecoder()
    cursor = 0
    plans: list[dict[str, Any]] = []
    while cursor < len(result.stdout):
        while cursor < len(result.stdout) and result.stdout[cursor].isspace():
            cursor += 1
        if cursor >= len(result.stdout):
            break
        try:
            document, cursor = decoder.raw_decode(result.stdout, cursor)
        except json.JSONDecodeError as error:
            raise CampaignError(
                "one-dose dry run emitted a malformed JSON stream"
            ) from error
        if isinstance(document, dict) and isinstance(document.get("command"), list):
            plans.append(document)
    if len(plans) != 1:
        raise CampaignError(
            f"one-dose dry run emitted {len(plans)} command-bearing JSON plans"
        )
    return plans[0]


def _verify_rendered_arm(
    campaign: Mapping[str, Any], *, arm: str, max_steps: int, plan: Mapping[str, Any]
) -> None:
    command = [str(value) for value in plan["command"]]
    initializer = _regular_file(
        Path(_option(command, "--init-checkpoint")), where="rendered parent checkpoint"
    )
    actual_initializer = _file_sha256(initializer)
    expected_parent = campaign["lineage_contract"]["expected_parent_sha256"]
    upgrade = plan.get("function_preserving_upgrade")
    if isinstance(upgrade, dict):
        source = upgrade.get("source")
        upgraded = upgrade.get("upgraded_initializer")
        if not isinstance(source, dict) or not isinstance(upgraded, dict):
            raise CampaignError("one-dose plan has malformed upgrade lineage")
        actual_parent = source.get("sha256")
        expected_initializer = upgraded.get("sha256")
    else:
        actual_parent = actual_initializer
        expected_initializer = expected_parent
    if actual_parent != expected_parent:
        raise CampaignError(
            f"diagnostic source parent mismatch: expected={expected_parent} actual={actual_parent}"
        )
    if actual_initializer != expected_initializer:
        raise CampaignError(
            "rendered initializer bytes differ from the authenticated "
            f"function-preserving upgrade: expected={expected_initializer} "
            f"actual={actual_initializer} path={initializer}"
        )
    if (
        "--no-resume-optimizer" not in command
        or command.count("torch.distributed.run") != 1
    ):
        raise CampaignError("rendered arm lost fresh-Adam 8xDDP semantics")
    # The nproc flag uses --flag=value syntax, unlike ordinary train_bc
    # options, so check it without the separate-value parser.
    if not any(token == "--nproc_per_node=8" for token in command):
        raise CampaignError("rendered arm is not an 8-rank DDP command")
    if int(_option(command, "--max-steps")) != int(max_steps):
        raise CampaignError("rendered arm max_steps drift")
    if float(_option(command, "--lr")) != float(ARMS[arm]["lr"]):
        raise CampaignError("rendered arm LR drift")
    if int(_option(command, "--lr-warmup-steps")) != int(
        ARMS[arm]["lr_warmup_steps"]
    ):
        raise CampaignError("rendered arm warmup drift")
    expected_checkpoints = "64" if max_steps == SHORT_STEPS else "64,128"
    if _option(command, "--checkpoint-steps") != expected_checkpoints:
        raise CampaignError("rendered within-trajectory checkpoint curve drift")
    if _option(command, "--train-diagnostics-every-batches") != "16":
        raise CampaignError("rendered module-attribution cadence drift")
    learner_ablation = plan.get("learner_ablation")
    actual_recipe = (
        learner_ablation.get("effective_recipe")
        if isinstance(learner_ablation, dict)
        else None
    )
    canonical = campaign["canonical_learner_projection"]["training_recipe"]
    if not isinstance(actual_recipe, dict) or not isinstance(canonical, dict):
        raise CampaignError("one-dose plan lost its effective canonical recipe")
    expected_recipe = dict(canonical)
    expected_recipe.update(
        {
            "batch_size": GLOBAL_BATCH_SIZE // WORLD_SIZE,
            "world_size": WORLD_SIZE,
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "epochs": 1,
            "max_steps": int(max_steps),
            "lr": float(ARMS[arm]["lr"]),
            "lr_warmup_steps": int(ARMS[arm]["lr_warmup_steps"]),
        }
    )
    recipe_drift = {
        key: {"expected": expected, "actual": actual_recipe.get(key)}
        for key, expected in expected_recipe.items()
        if actual_recipe.get(key) != expected
    }
    if recipe_drift:
        raise CampaignError(
            "rendered learner differs from the canonical science projection: "
            + json.dumps(recipe_drift, sort_keys=True)
        )
    if int(actual_recipe.get("policy_aux_active_batch_size", -1)) != int(
        campaign["policy_active_dose"]["policy_aux_active_batch_size"]
    ):
        raise CampaignError("rendered learner lost matched policy-active dose")


def _verify_training_report(
    campaign: Mapping[str, Any],
    *,
    arm: str,
    max_steps: int,
    one_dose_plan: Mapping[str, Any],
) -> dict[str, Any]:
    report_path = _regular_file(
        Path(str(one_dose_plan["report"])), where="completed training report"
    )
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CampaignError(f"cannot read completed training report: {error}") from error
    if not isinstance(report, dict) or int(report.get("steps_completed", -1)) != max_steps:
        raise CampaignError("completed training report lost its exact optimizer dose")
    if report.get("optimizer_restored") is not False:
        raise CampaignError("completed campaign arm did not use fresh Adam")
    strata = report.get("training_strata_dose")
    required_dimensions = set(
        campaign["reporting_contract"]["required_dimensions"]
    )
    if (
        not isinstance(strata, dict)
        or strata.get("schema_version") != "training-strata-dose-v1"
        or not isinstance(strata.get("dimensions"), dict)
        or set(strata["dimensions"]) != required_dimensions
    ):
        raise CampaignError("training report lost required realized-dose strata")
    expected_base = GLOBAL_BATCH_SIZE * max_steps
    policy_active = int(strata.get("policy_active_row_draws", -1))
    if (
        int(strata.get("base_row_draws", -1)) != expected_base
        or int(strata.get("policy_aux_row_draws", -1))
        != int(report.get("policy_aux_active_rows", -2))
        or policy_active != int(report.get("policy_total_active_rows", -2))
        or policy_active <= 0
    ):
        raise CampaignError("training report realized policy-active dose arithmetic drift")
    module_report = report.get("module_optimizer_observability")
    expected_observations = max_steps // 16
    if (
        not isinstance(module_report, dict)
        or module_report.get("schema_version")
        != "module-optimizer-observability-v1"
        or int(module_report.get("observed_steps", -1)) != expected_observations
        or not isinstance(module_report.get("modules"), dict)
        or not module_report["modules"]
    ):
        raise CampaignError("training report lost module gradient/update attribution")
    aux_sampler = report.get("policy_aux_sampling")
    if (
        not isinstance(aux_sampler, dict)
        or aux_sampler.get("schema_version") != "train-policy-aux-sampling-v1"
        or aux_sampler.get("enabled") is not True
        or aux_sampler.get("base_measure") != "authenticated_component"
        or aux_sampler.get("exact_per_game_policy_surprise_weighting") is not False
        or report.get("per_game_policy_surprise_weighting") is not False
        or float(report.get("public_card_lr_mult", -1.0)) != 1.0
        or report.get("forced_row_value_action_type_weights")
        != {"END_TURN": 0.1, "ROLL": 1.0}
    ):
        raise CampaignError("training report lost canonical learner/sampler semantics")
    expected_parent = campaign["lineage_contract"]["expected_parent_sha256"]
    learner_parent = report.get("a1_learner_lineage_parent")
    lineage_dose = report.get("a1_lineage_dose")
    input_binding = report.get("a1_one_dose_input_binding")
    if (
        not isinstance(learner_parent, dict)
        or learner_parent.get("schema_version")
        != "a1-learner-lineage-parent-v1"
        or learner_parent.get("role") != "diagnostic_recent_history"
        or not isinstance(learner_parent.get("checkpoint"), dict)
        or learner_parent.get("checkpoint", {}).get("sha256") != expected_parent
        or learner_parent.get("diagnostic_only") is not True
        or learner_parent.get("promotion_eligible") is not False
        or not isinstance(lineage_dose, dict)
        or lineage_dose.get("declared_producer_sha256") != expected_parent
        or not isinstance(input_binding, dict)
        or input_binding.get("learner_lineage_parent") != learner_parent
    ):
        raise CampaignError(
            "training report lost its explicit authenticated f7 learner parent"
        )
    for field in ("preconditioning_weights", "final_sampling_weights"):
        record = aux_sampler.get(field)
        if not isinstance(record, dict):
            raise CampaignError(f"policy auxiliary sampler lost {field} attribution")
        _normalize_sha256(
            str(record.get("content_sha256")),
            where=f"policy auxiliary {field} content digest",
        )
    return {
        "arm": arm,
        "max_steps": max_steps,
        "report": str(report_path),
        "report_file_sha256": _file_sha256(report_path),
        "base_row_draws": expected_base,
        "policy_active_row_draws": policy_active,
        "policy_active_fraction": float(strata["policy_active_fraction"]),
        "policy_aux_row_draws": int(strata["policy_aux_row_draws"]),
        "module_observed_steps": expected_observations,
    }


def _verify_completed_arm_receipt(
    campaign: Mapping[str, Any], *, arm: str
) -> dict[str, Any]:
    """Require terminal success and authenticate the artifacts before advancing."""

    arm_root = Path(campaign["output_root"]) / "arms" / arm
    receipt_path = _regular_file(
        arm_root / "one-dose.receipt.json", where=f"completed arm {arm} receipt"
    )
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CampaignError(f"cannot read completed arm {arm} receipt: {error}") from error
    outputs = payload.get("outputs") if isinstance(payload, dict) else None
    learner_parent = (
        payload.get("learner_lineage_parent") if isinstance(payload, dict) else None
    )
    expected_parent = campaign["lineage_contract"]["expected_parent_sha256"]
    receipt_input_binding = (
        payload.get("input_binding") if isinstance(payload, dict) else None
    )
    receipt_lineage_dose = (
        outputs.get("lineage_dose") if isinstance(outputs, dict) else None
    )
    if (
        not isinstance(payload, dict)
        or payload.get("status") != "complete"
        or payload.get("returncode") != 0
        or not isinstance(outputs, dict)
        or not isinstance(learner_parent, dict)
        or not isinstance(learner_parent.get("checkpoint"), dict)
        or learner_parent.get("checkpoint", {}).get("sha256") != expected_parent
        or not isinstance(receipt_input_binding, dict)
        or receipt_input_binding.get("learner_lineage_parent")
        != learner_parent
        or not isinstance(receipt_lineage_dose, dict)
        or receipt_lineage_dose.get("declared_producer_sha256")
        != expected_parent
    ):
        raise CampaignError(f"arm {arm} has no terminal successful receipt")
    artifacts: dict[str, dict[str, str]] = {}
    for name, expected_path in (
        ("checkpoint", arm_root / "candidate.pt"),
        ("report", arm_root / "train.report.json"),
    ):
        path_value = outputs.get(name)
        digest_value = outputs.get(f"{name}_sha256")
        if str(path_value) != str(expected_path):
            raise CampaignError(f"arm {arm} {name} escaped its planned namespace")
        artifact = _regular_file(expected_path, where=f"completed arm {arm} {name}")
        expected_digest = _normalize_sha256(
            str(digest_value), where=f"arm {arm} {name} receipt digest"
        )
        actual_digest = _file_sha256(artifact)
        if actual_digest != expected_digest:
            raise CampaignError(
                f"arm {arm} {name} bytes differ from its terminal receipt"
            )
        artifacts[name] = {
            "path": str(artifact),
            "sha256": actual_digest,
        }
    return {
        "receipt": str(receipt_path),
        "receipt_file_sha256": _file_sha256(receipt_path),
        "status": "complete",
        "returncode": 0,
        "artifacts": artifacts,
    }


def _execute_invocation(
    campaign: Mapping[str, Any],
    *,
    arm: str,
    max_steps: int,
    invocation: list[str],
    go: bool,
) -> dict[str, Any]:
    _verify_input_bytes(campaign)
    plan = _one_dose_dry_run(invocation)
    _verify_rendered_arm(campaign, arm=arm, max_steps=max_steps, plan=plan)
    if not go:
        return {"mode": "dry-run", "one_dose_plan": plan}
    result = subprocess.run([*invocation, "--go"], check=False)
    if result.returncode != 0:
        raise CampaignError(f"one-dose arm {arm} exited {result.returncode}")
    receipt_summary = _verify_completed_arm_receipt(campaign, arm=arm)
    report_summary = _verify_training_report(
        campaign,
        arm=arm,
        max_steps=max_steps,
        one_dose_plan=plan,
    )
    return {
        "mode": "go",
        "returncode": 0,
        "receipt": receipt_summary,
        **report_summary,
    }


def _parse_bindings(values: Sequence[str], *, label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for raw in values:
        arm, separator, path = raw.partition("=")
        if separator != "=" or arm not in ARMS or arm in result:
            raise CampaignError(f"{label} must contain each unique ARM=PATH binding")
        result[arm] = _regular_file(Path(path), where=f"{label} {arm}")
    if set(result) != set(ARMS):
        raise CampaignError(f"{label} requires A, B, C, and D")
    return result


def _load_json_object(path: Path, *, where: str) -> dict[str, Any]:
    resolved = _regular_file(path, where=where)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CampaignError(f"cannot load {where}: {error}") from error
    if not isinstance(payload, dict):
        raise CampaignError(f"{where} must be one JSON object")
    return payload


def _finite_score(value: object, *, where: str) -> float:
    if isinstance(value, bool):
        raise CampaignError(f"{where} must be a finite numeric score")
    try:
        score = float(value)
    except (TypeError, ValueError) as error:
        raise CampaignError(f"{where} must be a finite numeric score") from error
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise CampaignError(f"{where} must be in [0,1]")
    return score


def _score_interval(value: object, *, where: str) -> tuple[float, float]:
    if isinstance(value, Mapping):
        lower = value.get("lower", value.get("low"))
        upper = value.get("upper", value.get("high"))
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        lower, upper = value
    else:
        raise CampaignError(f"{where} must be [lower,upper] or a lower/upper object")
    result = (
        _finite_score(lower, where=f"{where}.lower"),
        _finite_score(upper, where=f"{where}.upper"),
    )
    if result[0] > result[1]:
        raise CampaignError(f"{where} lower bound exceeds upper bound")
    return result


def _evaluation_rows(
    payload: Mapping[str, Any], *, arm: str, where: str
) -> dict[str, Mapping[str, Any]]:
    """Normalize the existing R5 summary's mapping/list encodings.

    Historical writers emitted either ``arms[ARM]["arm-vs-f7"]`` mappings or
    one flat ``rows``/``evaluations`` list. Supporting both is safe because the
    role and arm still have to resolve exactly once and every row's pooled
    report is authenticated below.
    """

    if payload.get("schema_version") != NATIVE_EVAL_SUMMARY_SCHEMA:
        raise CampaignError(f"{where} is not {NATIVE_EVAL_SUMMARY_SCHEMA}")
    candidates: list[Mapping[str, Any]] = []
    arms = payload.get("arms")
    if isinstance(arms, Mapping):
        record = arms.get(arm)
        if isinstance(record, Mapping):
            for key, value in record.items():
                if isinstance(value, Mapping):
                    candidates.append(
                        {**value, "_comparison_key": str(key), "_bound_arm": arm}
                    )
    for key in ("rows", "evaluations", "comparisons"):
        rows = payload.get(key)
        if isinstance(rows, list):
            candidates.extend(row for row in rows if isinstance(row, Mapping))
    if not candidates:
        # A per-arm summary may place the two comparison rows at top level.
        for key in ("arm-vs-f7", "arm_vs_f7", "arm-vs-v5", "arm_vs_v5"):
            value = payload.get(key)
            if isinstance(value, Mapping):
                candidates.append(
                    {**value, "_comparison_key": key, "_bound_arm": arm}
                )

    resolved: dict[str, Mapping[str, Any]] = {}
    for row in candidates:
        identity = str(
            row.get(
                "comparison",
                row.get(
                    "comparison_id",
                    row.get(
                        "matchup",
                        row.get("baseline_role", row.get("_comparison_key", "")),
                    ),
                ),
            )
        ).lower()
        explicit_row_arm = row.get("arm", row.get("_bound_arm"))
        if explicit_row_arm is None:
            for separator in ("-vs-", "_vs_"):
                if separator in identity:
                    explicit_row_arm = identity.split(separator, 1)[0]
                    break
        row_arm = str(explicit_row_arm if explicit_row_arm is not None else arm).upper()
        if row_arm != arm:
            continue
        matching_roles = [
            role
            for role in EVALUATION_BASELINE_ROLES
            if identity in {role, f"arm-vs-{role}", f"arm_vs_{role}"}
            or identity in {f"{arm.lower()}-vs-{role}", f"{arm.lower()}_vs_{role}"}
        ]
        if len(matching_roles) != 1:
            continue
        role = matching_roles[0]
        if role in resolved:
            raise CampaignError(f"{where} has ambiguous duplicate {arm}-vs-{role} rows")
        resolved[role] = row
    missing = sorted(set(EVALUATION_BASELINE_ROLES) - set(resolved))
    if missing:
        raise CampaignError(f"{where} is missing {arm} comparisons for {missing}")
    return resolved


def _report_path_from_row(
    row: Mapping[str, Any], *, summary_path: Path, where: str
) -> Path:
    raw = row.get("report", row.get("report_path", row.get("pooled_report")))
    if isinstance(raw, Mapping):
        raw = raw.get("path")
    if not isinstance(raw, str) or not raw:
        raise CampaignError(f"{where} has no pooled report path")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = summary_path.parent / candidate
    return _regular_file(candidate, where=f"{where} pooled report")


def _metric_from_report_or_row(
    report: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    key: str,
    where: str,
) -> object:
    row_value = row.get(key)
    if row_value is None:
        raise CampaignError(f"{where} row is missing {key}")
    report_value = report.get(key)
    if report_value is not None and report_value != row_value:
        raise CampaignError(f"{where} summary/report {key} mismatch")
    return row_value


def _authenticate_evaluation_row(
    *,
    arm: str,
    role: str,
    row: Mapping[str, Any],
    summary_path: Path,
    candidate_sha256: str,
) -> dict[str, Any]:
    where = f"arm {arm} versus {role}"
    report_path = _report_path_from_row(row, summary_path=summary_path, where=where)
    report = _load_json_object(report_path, where=f"{where} pooled report")
    actual_candidate = _normalize_sha256(
        str(report.get("candidate_checkpoint_sha256", "")),
        where=f"{where} candidate checkpoint",
    )
    if actual_candidate != candidate_sha256:
        raise CampaignError(
            f"{where} candidate checkpoint does not match arm receipt: "
            f"expected={candidate_sha256} actual={actual_candidate}"
        )
    if row.get("candidate_checkpoint_sha256") is not None and _normalize_sha256(
        str(row["candidate_checkpoint_sha256"]),
        where=f"{where} summary candidate checkpoint",
    ) != actual_candidate:
        raise CampaignError(f"{where} summary candidate checkpoint disagrees with report")
    baseline_sha256 = _normalize_sha256(
        str(report.get("baseline_checkpoint_sha256", "")),
        where=f"{where} baseline checkpoint",
    )
    if row.get("baseline_checkpoint_sha256") is not None and _normalize_sha256(
        str(row["baseline_checkpoint_sha256"]),
        where=f"{where} summary baseline checkpoint",
    ) != baseline_sha256:
        raise CampaignError(f"{where} summary baseline checkpoint disagrees with report")
    if report.get("errors") != []:
        raise CampaignError(f"{where} pooled report contains errors")
    pairs_requested = int(report.get("pairs_requested", -1))
    complete_pairs = int(report.get("complete_pairs", -1))
    games_played = int(report.get("games_played", -1))
    games_with_winner = int(report.get("games_with_winner", -1))
    games_truncated = int(report.get("games_truncated", -1))
    games_requested = int(report.get("games_requested", pairs_requested * 2))
    if (
        pairs_requested <= 0
        or complete_pairs != pairs_requested
        or games_requested != pairs_requested * 2
        or games_played != games_requested
        or games_with_winner != games_requested
        or games_truncated != 0
        or int(report.get("pairs_truncated_excluded", 0)) != 0
    ):
        raise CampaignError(
            f"{where} pooled report is incomplete or truncated: "
            f"pairs={complete_pairs}/{pairs_requested} "
            f"games={games_with_winner}/{games_requested} "
            f"truncated={games_truncated}"
        )
    mu = _finite_score(
        _metric_from_report_or_row(
            report,
            row,
            key="paired_score_regularized_mu",
            where=where,
        ),
        where=f"{where} paired score",
    )
    lower, upper = _score_interval(
        _metric_from_report_or_row(
            report,
            row,
            key="paired_score_regularized_95ci",
            where=where,
        ),
        where=f"{where} paired score 95ci",
    )
    if not lower <= mu <= upper:
        raise CampaignError(f"{where} point score lies outside its 95ci")
    return {
        "baseline_role": role,
        "candidate_checkpoint_sha256": actual_candidate,
        "baseline_checkpoint_sha256": baseline_sha256,
        "paired_score_regularized_mu": mu,
        "paired_score_regularized_95ci": [lower, upper],
        "pairs_requested": pairs_requested,
        "complete_pairs": complete_pairs,
        "games_played": games_played,
        "games_truncated": games_truncated,
        "report": str(report_path),
        "report_file_sha256": _file_sha256(report_path),
    }


def _rank_authenticated_evaluations(
    *,
    receipt_records: Mapping[str, Mapping[str, Any]],
    evaluation_paths: Mapping[str, Path],
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    baseline_sha_by_role: dict[str, str] = {}
    arm_evidence: dict[str, Any] = {}
    ranking: list[dict[str, Any]] = []
    for arm in ARMS:
        summary_path = evaluation_paths[arm]
        summary = _load_json_object(
            summary_path, where=f"arm {arm} evaluation summary"
        )
        rows = _evaluation_rows(
            summary, arm=arm, where=f"arm {arm} evaluation summary"
        )
        candidate_sha = str(receipt_records[arm]["checkpoint_sha256"])
        comparisons: dict[str, Any] = {}
        for role in EVALUATION_BASELINE_ROLES:
            evidence = _authenticate_evaluation_row(
                arm=arm,
                role=role,
                row=rows[role],
                summary_path=summary_path,
                candidate_sha256=candidate_sha,
            )
            baseline = evidence["baseline_checkpoint_sha256"]
            previous = baseline_sha_by_role.setdefault(role, baseline)
            if previous != baseline:
                raise CampaignError(
                    f"baseline checkpoint drift for role {role}: "
                    f"expected={previous} arm={arm} actual={baseline}"
                )
            comparisons[role] = evidence
        worst_lower = min(
            float(value["paired_score_regularized_95ci"][0])
            for value in comparisons.values()
        )
        worst_mu = min(
            float(value["paired_score_regularized_mu"])
            for value in comparisons.values()
        )
        record = {
            "arm": arm,
            "robust_worst_baseline_95ci_lower": worst_lower,
            "robust_worst_baseline_point_score": worst_mu,
            "comparisons": comparisons,
        }
        arm_evidence[arm] = {
            "summary": str(summary_path),
            "summary_file_sha256": _file_sha256(summary_path),
            **record,
        }
        ranking.append(record)
    ranking.sort(
        key=lambda value: (
            -float(value["robust_worst_baseline_95ci_lower"]),
            -float(value["robust_worst_baseline_point_score"]),
            str(value["arm"]),
        )
    )
    if len(ranking) < 2 or (
        ranking[0]["robust_worst_baseline_95ci_lower"]
        == ranking[1]["robust_worst_baseline_95ci_lower"]
        and ranking[0]["robust_worst_baseline_point_score"]
        == ranking[1]["robust_worst_baseline_point_score"]
    ):
        raise CampaignError(
            "playing-strength evidence is ambiguous after the declared robust "
            "primary and secondary objectives"
        )
    return str(ranking[0]["arm"]), ranking, {
        "baseline_checkpoint_sha256_by_role": baseline_sha_by_role,
        "arms": arm_evidence,
    }


def _verify_winner_assertion(asserted: str | None, actual: str) -> None:
    if asserted is not None and asserted != actual:
        raise CampaignError(
            f"--winner={asserted} disagrees with authenticated robust winner {actual}"
        )


def _select(args: argparse.Namespace) -> dict[str, Any]:
    campaign_path = _regular_file(args.campaign, where="campaign plan")
    campaign = _load_bound_json(campaign_path, schema=SCHEMA)
    receipts = _parse_bindings(args.arm_receipt, label="arm receipt")
    evaluations = _parse_bindings(args.evaluation, label="evaluation")
    receipt_records: dict[str, Any] = {}
    for arm, path in receipts.items():
        expected = str(Path(campaign["output_root"]) / "arms" / arm / "one-dose.receipt.json")
        if str(path) != expected:
            raise CampaignError(f"arm {arm} receipt is outside its planned namespace")
        completed = _verify_completed_arm_receipt(campaign, arm=arm)
        receipt_records[arm] = {
            "path": str(path),
            "file_sha256": completed["receipt_file_sha256"],
            "checkpoint_sha256": completed["artifacts"]["checkpoint"]["sha256"],
        }
    winner, ranking, authenticated_evidence = _rank_authenticated_evaluations(
        receipt_records=receipt_records,
        evaluation_paths=evaluations,
    )
    _verify_winner_assertion(args.winner, winner)
    selection: dict[str, Any] = {
        "schema_version": SELECTION_SCHEMA,
        "campaign": str(campaign_path),
        "campaign_file_sha256": _file_sha256(campaign_path),
        "campaign_sha256": campaign["campaign_sha256"],
        "winner": winner,
        "winner_assertion": args.winner,
        "selection_basis": "algorithmic_authenticated_robust_playing_strength",
        "selection_objective": {
            "primary": SELECTION_OBJECTIVE,
            "secondary": SELECTION_SECONDARY,
            "required_baselines": list(EVALUATION_BASELINE_ROLES),
            "tie_policy": "refuse_if_primary_and_secondary_are_tied",
        },
        "ranking": ranking,
        "authenticated_playing_strength_evidence": authenticated_evidence,
        "winner_recipe": _arm_overrides(
            winner,
            max_steps=LONG_STEPS,
            policy_aux_active_batch_size=int(
                campaign["policy_active_dose"]["policy_aux_active_batch_size"]
            ),
            science_recipe=campaign["canonical_learner_projection"][
                "training_recipe"
            ],
        ),
        "winner_replays_from_original_parent": True,
        "candidate_chaining": False,
        "arm_receipts": receipt_records,
        "evaluation_receipts": {
            arm: {"path": str(path), "file_sha256": _file_sha256(path)}
            for arm, path in evaluations.items()
        },
        "long_dose_output_subdir": f"winner/{winner}-steps256",
    }
    selection["selection_sha256"] = _value_sha256(selection)
    return selection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="seal the four-arm campaign")
    plan.add_argument("--lock", required=True, type=Path)
    plan.add_argument("--data", required=True, type=Path)
    plan.add_argument("--composite-build-receipt", required=True, type=Path)
    plan.add_argument("--architecture-upgrade-receipt", required=True, type=Path)
    plan.add_argument("--ddp-canary-receipt", required=True, type=Path)
    plan.add_argument("--expected-parent-sha256", required=True)
    plan.add_argument("--reviewed-code-tree-sha256", required=True)
    plan.add_argument("--reviewed-lock-file-sha256", default="")
    plan.add_argument("--python", required=True, type=Path)
    plan.add_argument("--output-root", required=True, type=Path)
    plan.add_argument("--write", required=True, type=Path)
    plan.add_argument("--observed-base-policy-active-fraction", type=float, default=0.0)
    plan.add_argument("--target-policy-active-rows", type=int, default=0)
    plan.add_argument("--policy-aux-active-batch-size", type=int, default=0)

    run = sub.add_parser("run-arm", help="dry-run or execute one 128-step arm")
    run.add_argument("--campaign", required=True, type=Path)
    run.add_argument("--arm", required=True, choices=tuple(ARMS))
    run.add_argument("--go", action="store_true")

    sequence = sub.add_parser(
        "run-sequence",
        help="run independent arms serially, advancing only after authenticated success",
    )
    sequence.add_argument("--campaign", required=True, type=Path)
    sequence.add_argument(
        "--arms",
        default="C,D,A,B",
        help="unique comma-separated arm order (default: C,D,A,B)",
    )
    sequence.add_argument("--go", action="store_true")

    select = sub.add_parser("select", help="bind evaluated winner after all arms")
    select.add_argument("--campaign", required=True, type=Path)
    select.add_argument(
        "--winner",
        choices=tuple(ARMS),
        default=None,
        help=(
            "Optional assertion of the algorithmic winner. It cannot select or "
            "override an arm and is rejected if it disagrees with the evidence."
        ),
    )
    select.add_argument("--arm-receipt", action="append", default=[], metavar="ARM=PATH")
    select.add_argument("--evaluation", action="append", default=[], metavar="ARM=PATH")
    select.add_argument("--write", required=True, type=Path)

    winner = sub.add_parser(
        "run-winner-256", help="replay only the selected recipe to 256 from parent"
    )
    winner.add_argument("--campaign", required=True, type=Path)
    winner.add_argument("--selection", required=True, type=Path)
    winner.add_argument("--go", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "plan":
            campaign = _plan(args)
            _write_json(args.write, campaign)
            print(json.dumps(campaign, indent=2, sort_keys=True))
            return 0
        if args.command == "run-arm":
            campaign = _load_bound_json(args.campaign, schema=SCHEMA)
            invocation = _one_dose_invocation(
                campaign,
                arm=args.arm,
                max_steps=SHORT_STEPS,
                suffix=f"arms/{args.arm}",
            )
            result = _execute_invocation(
                campaign,
                arm=args.arm,
                max_steps=SHORT_STEPS,
                invocation=invocation,
                go=bool(args.go),
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.command == "run-sequence":
            campaign = _load_bound_json(args.campaign, schema=SCHEMA)
            arms = [value.strip() for value in args.arms.split(",") if value.strip()]
            if (
                not arms
                or any(arm not in ARMS for arm in arms)
                or len(set(arms)) != len(arms)
            ):
                raise CampaignError("--arms must contain unique A, B, C, and/or D values")
            results: list[dict[str, Any]] = []
            for arm in arms:
                invocation = _one_dose_invocation(
                    campaign,
                    arm=arm,
                    max_steps=SHORT_STEPS,
                    suffix=f"arms/{arm}",
                )
                result = _execute_invocation(
                    campaign,
                    arm=arm,
                    max_steps=SHORT_STEPS,
                    invocation=invocation,
                    go=bool(args.go),
                )
                results.append(result)
            print(
                json.dumps(
                    {"mode": "go" if args.go else "dry-run", "arms": arms, "results": results},
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "select":
            selection = _select(args)
            _write_json(args.write, selection)
            print(json.dumps(selection, indent=2, sort_keys=True))
            return 0
        if args.command == "run-winner-256":
            campaign = _load_bound_json(args.campaign, schema=SCHEMA)
            selection = _load_bound_json(args.selection, schema=SELECTION_SCHEMA)
            if (
                selection.get("campaign_sha256") != campaign["campaign_sha256"]
                or selection.get("campaign_file_sha256")
                != _file_sha256(_regular_file(args.campaign, where="campaign plan"))
            ):
                raise CampaignError("winner selection binds a different campaign")
            arm = str(selection["winner"])
            suffix = str(selection["long_dose_output_subdir"])
            invocation = _one_dose_invocation(
                campaign, arm=arm, max_steps=LONG_STEPS, suffix=suffix
            )
            result = _execute_invocation(
                campaign,
                arm=arm,
                max_steps=LONG_STEPS,
                invocation=invocation,
                go=bool(args.go),
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        raise CampaignError(f"unknown command {args.command}")
    except (CampaignError, OSError, json.JSONDecodeError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
