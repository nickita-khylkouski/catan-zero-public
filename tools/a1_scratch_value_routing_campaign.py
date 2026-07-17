#!/usr/bin/env python3
"""Matched V0/V10/V100 scratch value-routing campaign.

The canonical A1 scratch plan currently binds ``value_trunk_grad_scale=0.1``.
This diagnostic reuses that exact initializer, corpus, model, optimizer, RNG,
dose, and 8-GPU topology while independently launching three fresh arms:

* V0:   scalar value loss cannot update the shared representation;
* V10:  the canonical 0.1 routing control;
* V100: full scalar value gradient reaches the shared representation.

Plans and execution receipts are immutable and non-promotion-eligible. A
separate selection receipt can nominate one completed arm for evaluation, but
cannot promote it.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import a1_scratch_train as scratch  # noqa: E402
from tools import train_bc  # noqa: E402


CAMPAIGN_PLAN_SCHEMA = "a1-scratch-value-routing-campaign-plan-v2"
ARM_PLAN_SCHEMA = "a1-scratch-value-routing-arm-plan-v2"
ARM_EXECUTION_SCHEMA = "a1-scratch-value-routing-arm-execution-v2"
CAMPAIGN_EXECUTION_SCHEMA = "a1-scratch-value-routing-campaign-execution-v2"
SELECTION_SCHEMA = "a1-scratch-value-routing-selection-v2"
DIAGNOSTIC_AUTHORITY_SCHEMA = "a1-scratch-bounded-diagnostic-authority-v2"
CAMPAIGN_ID = "scratch-value-routing-v0-v10-v100"
ARMS = {"V0": 0.0, "V10": 0.1, "V100": 1.0}
BASELINE_ARM = "V10"
BASELINE_SCALE = ARMS[BASELINE_ARM]
ABLATION_CODE_SURFACE = (
    "tools/a1_scratch_value_routing_campaign.py",
    "tools/a1_scratch_train.py",
    "tools/a1_current_science_contract.py",
    "tools/train_bc.py",
    "src/catan_zero/rl/entity_token_policy.py",
)
AUTHORITY_ONLY_RECIPE_DEFAULTS = {
    "per_game_value_weight_mode": "equal",
    "public_card_lr_mult": 1.0,
    "per_game_policy_surprise_weighting": False,
    "target_reliability_confidence_weighting": False,
    "target_reliability_confidence_floor": 0.25,
}


class ValueRoutingCampaignError(RuntimeError):
    """The matched scratch value-routing campaign is not executable."""


def _load_receipt(path: Path, *, where: str) -> dict[str, Any]:
    resolved = scratch._regular_file(path, where=where)  # noqa: SLF001
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueRoutingCampaignError(f"cannot read {where}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueRoutingCampaignError(f"{where} must be a JSON object")
    unsigned = dict(payload)
    stated = unsigned.pop("receipt_sha256", None)
    if stated != scratch._value_sha256(unsigned):  # noqa: SLF001
        raise ValueRoutingCampaignError(f"{where} semantic digest drift")
    return payload


def _receipt_ref(path: Path, *, where: str) -> dict[str, str]:
    payload = _load_receipt(path, where=where)
    return {
        **scratch._ref(path, where=where),  # noqa: SLF001
        "receipt_sha256": str(payload["receipt_sha256"]),
    }


def _verify_receipt_ref(raw: object, *, where: str) -> tuple[Path, dict[str, Any]]:
    if not isinstance(raw, dict) or set(raw) != {
        "path",
        "file_sha256",
        "receipt_sha256",
    }:
        raise ValueRoutingCampaignError(f"{where} reference is malformed")
    path = scratch._regular_file(Path(str(raw["path"])), where=where)  # noqa: SLF001
    if scratch._file_sha256(path) != raw["file_sha256"]:  # noqa: SLF001
        raise ValueRoutingCampaignError(f"{where} bytes drifted")
    payload = _load_receipt(path, where=where)
    if payload["receipt_sha256"] != raw["receipt_sha256"]:
        raise ValueRoutingCampaignError(f"{where} receipt digest drifted")
    return path, payload


def _trainer_index(command: Sequence[str]) -> int:
    found = [
        index
        for index, token in enumerate(command)
        if Path(token).name == "train_bc.py"
    ]
    if len(found) != 1:
        raise ValueRoutingCampaignError("scratch command must name one train_bc.py")
    return found[0]


def _set_unique(command: list[str], flag: str, value: object) -> None:
    positions = [index for index, token in enumerate(command) if token == flag]
    if len(positions) != 1:
        raise ValueRoutingCampaignError(f"scratch command must contain one {flag}")
    position = positions[0]
    if position + 1 >= len(command) or command[position + 1].startswith("--"):
        raise ValueRoutingCampaignError(f"scratch command has no value for {flag}")
    command[position + 1] = str(value)


def _code_binding() -> dict[str, Any]:
    records = [
        {
            "kind": "learner_code",
            "relative_path": relative,
            "path": str((REPO_ROOT / relative).resolve(strict=True)),
            "sha256": scratch._file_sha256(  # noqa: SLF001
                (REPO_ROOT / relative).resolve(strict=True)
            ),
        }
        for relative in ABLATION_CODE_SURFACE
    ]
    binding: dict[str, Any] = {
        "schema_version": "a1-scratch-value-routing-code-binding-v1",
        "records": records,
    }
    binding["code_tree_sha256"] = scratch._value_sha256(binding)  # noqa: SLF001
    return binding


def _parsed_effective_recipe(command: Sequence[str]) -> tuple[Any, dict[str, object]]:
    parser = train_bc.build_parser()
    try:
        args = parser.parse_args(list(command)[_trainer_index(command) + 1 :])
    except SystemExit as error:
        raise ValueRoutingCampaignError(
            "cannot parse rendered scratch command"
        ) from error
    effective = train_bc._effective_a1_learner_training_recipe(  # noqa: SLF001
        args,
        {"world_size": 8, "rank": 0, "local_rank": 0, "enabled": True},
    )
    if str(args.a1_learner_ablation_id or ""):
        effective["per_game_value_weight_mode"] = str(args.per_game_value_weight_mode)
        if str(args.value_player_outcome_balance_mode) != "none":
            effective["value_player_outcome_balance_mode"] = str(
                args.value_player_outcome_balance_mode
            )
    return args, effective


def _assert_one_axis_recipe(
    baseline: Mapping[str, object],
    treatment: Mapping[str, object],
    *,
    scale: float,
) -> dict[str, object]:
    normalized = dict(treatment)
    authority_expansions: dict[str, object] = {}
    for key, expected in AUTHORITY_ONLY_RECIPE_DEFAULTS.items():
        if key not in baseline and key in normalized:
            actual = normalized.pop(key)
            if actual != expected:
                raise ValueRoutingCampaignError(
                    f"authority-only default {key!r} changed optimizer semantics"
                )
            authority_expansions[key] = actual
    expected = dict(baseline)
    expected["value_trunk_grad_scale"] = float(scale)
    if normalized != expected:
        drift = {
            key: {"baseline": expected.get(key), "treatment": normalized.get(key)}
            for key in sorted(set(expected) | set(normalized))
            if expected.get(key) != normalized.get(key)
        }
        raise ValueRoutingCampaignError(
            f"value-routing arm changes fields beyond its causal axis: {drift}"
        )
    return authority_expansions


def _derive_arm(
    *,
    verified: Mapping[str, Any],
    python: Path,
    root: Path,
    arm_id: str,
    scale: float,
    diagnostic_max_steps: int,
    code_binding: Mapping[str, Any],
) -> dict[str, Any]:
    arm_root = root / "arms" / arm_id
    checkpoint = arm_root / "candidate.pt"
    report = arm_root / "train.report.json"
    arm_verified = copy.deepcopy(dict(verified))
    recipe = dict(arm_verified["recipe"])
    original_max_steps = int(recipe["max_steps"])
    original_checkpoint_steps = scratch._checkpoint_steps(recipe)  # noqa: SLF001
    bounded_checkpoint_steps = tuple(
        step for step in original_checkpoint_steps if step < diagnostic_max_steps
    )
    original_epochs = int(recipe["epochs"])
    recipe["epochs"] = 1
    recipe["max_steps"] = int(diagnostic_max_steps)
    recipe["checkpoint_steps"] = ",".join(map(str, bounded_checkpoint_steps))
    recipe["value_trunk_grad_scale"] = float(scale)
    arm_verified["recipe"] = recipe
    command = scratch.build_train_command(
        arm_verified,
        python=python,
        checkpoint=checkpoint,
        report=report,
    )
    canonical_recipe = dict(verified["recipe"])
    logical_recipe = dict(verified["logical_recipe"])
    ablation_id = f"scratch-value-routing-{arm_id.lower()}"
    diagnostic_authority = {
        "schema_version": DIAGNOSTIC_AUTHORITY_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "arm_id": arm_id,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "exact_max_steps": True,
        "max_steps": int(diagnostic_max_steps),
        "epochs": int(recipe["epochs"]),
        "checkpoint_steps": list(bounded_checkpoint_steps),
        "value_trunk_grad_scale": float(scale),
        "source_recipe_sha256": scratch._value_sha256(logical_recipe),  # noqa: SLF001
        "source_execution_topology_sha256": scratch._value_sha256(  # noqa: SLF001
            verified["execution_topology"]
        ),
        "code_tree_sha256": str(code_binding["code_tree_sha256"]),
    }
    command.extend(
        (
            "--exact-max-steps",
            "--a1-learner-ablation-id",
            ablation_id,
            "--a1-scratch-diagnostic-authority-json",
            scratch._canonical_bytes(diagnostic_authority).decode("ascii"),  # noqa: SLF001
        )
    )
    _parsed_args, effective = _parsed_effective_recipe(command)
    authority_expansions = {
        key: effective[key]
        for key in AUTHORITY_ONLY_RECIPE_DEFAULTS
        if key not in canonical_recipe and key in effective
    }
    staged_lock = scratch._regular_file(  # noqa: SLF001
        Path(str(verified["lock_path"])), where="A1 staged lock"
    )
    command.extend(
        (
            "--a1-effective-learner-recipe-json",
            scratch._canonical_bytes(effective).decode("ascii"),  # noqa: SLF001
            "--a1-effective-learner-recipe-sha256",
            scratch._value_sha256(effective),  # noqa: SLF001
            "--a1-ablation-code-binding-json",
            scratch._canonical_bytes(code_binding).decode("ascii"),  # noqa: SLF001
            "--a1-ablation-code-tree-sha256",
            str(code_binding["code_tree_sha256"]),
            "--a1-reviewed-lock-file-sha256",
            scratch._file_sha256(staged_lock),  # noqa: SLF001
        )
    )
    forbidden = {
        "--init-checkpoint",
        "--grow-from-checkpoint",
        "--resume-optimizer",
    }
    if forbidden.intersection(command) or command.count("--no-resume-optimizer") != 1:
        raise ValueRoutingCampaignError("value-routing arm lost fresh initialization")
    return {
        "schema_version": ARM_PLAN_SCHEMA,
        "created_unix_ns": time.time_ns(),
        "status": "planned",
        "arm_id": arm_id,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "evaluation_eligible": False,
        "maximum_result": "training_evidence_only_pending_campaign_selection",
        "value_trunk_grad_scale": float(scale),
        "causal_recipe_delta": {
            "field": "value_trunk_grad_scale",
            "baseline": float(canonical_recipe["value_trunk_grad_scale"]),
            "treatment": float(scale),
        },
        "campaign_common_recipe_delta": {
            "epochs": {
                "source": original_epochs,
                "diagnostic": 1,
            },
            "max_steps": {
                "source": original_max_steps,
                "diagnostic": int(diagnostic_max_steps),
            },
            "checkpoint_steps": {
                "source": list(original_checkpoint_steps),
                "diagnostic": list(bounded_checkpoint_steps),
            },
        },
        "bounded_diagnostic_authority": diagnostic_authority,
        "authority_only_recipe_expansions": authority_expansions,
        "independent_initialization": {
            "mode": "from_scratch",
            "seed": int(recipe["seed"]),
            "fresh_optimizer": True,
            "candidate_chaining": False,
        },
        "effective_recipe": effective,
        "effective_recipe_sha256": scratch._value_sha256(effective),  # noqa: SLF001
        "checkpoint": str(checkpoint),
        "report": str(report),
        "execution_receipt": str(arm_root / "execution.receipt.json"),
        "command": command,
        "command_sha256": scratch._value_sha256(command),  # noqa: SLF001
    }


def prepare(
    *,
    lock: Path,
    data: Path,
    composite_build_receipt: Path,
    policy_target_quality_receipt: Path,
    output_root: Path,
    plan_path: Path,
    python: Path,
    diagnostic_max_steps: int = 128,
) -> dict[str, Any]:
    root = output_root.expanduser().resolve(strict=False)
    plan_path = plan_path.expanduser().resolve(strict=False)
    if plan_path.exists() or root.exists():
        raise ValueRoutingCampaignError("campaign output root and plan must be fresh")
    if not 1 <= int(diagnostic_max_steps) <= 256:
        raise ValueRoutingCampaignError("diagnostic max steps must be in [1, 256]")
    python_authority = scratch._executable_ref(  # noqa: SLF001
        python, where="learner Python"
    )
    verified = scratch.verify_inputs(
        lock_path=lock,
        data_path=data,
        composite_build_receipt=composite_build_receipt,
        policy_target_quality_receipt=policy_target_quality_receipt,
    )
    baseline_recipe = dict(verified["recipe"])
    if (
        float(baseline_recipe.get("value_trunk_grad_scale", -1.0))
        != BASELINE_SCALE
    ):
        raise ValueRoutingCampaignError(
            "current scratch authority must bind value_trunk_grad_scale=0.1"
        )
    code_binding = _code_binding()
    arm_refs: dict[str, dict[str, str]] = {}
    arm_summaries: dict[str, dict[str, Any]] = {}
    for arm_id, scale in ARMS.items():
        arm_plan = _derive_arm(
            verified=verified,
            python=Path(python_authority["path"]),
            root=root,
            arm_id=arm_id,
            scale=scale,
            diagnostic_max_steps=int(diagnostic_max_steps),
            code_binding=code_binding,
        )
        arm_path = root / "arms" / arm_id / "plan.json"
        scratch._write_receipt(arm_path, arm_plan)  # noqa: SLF001
        arm_refs[arm_id] = _receipt_ref(arm_path, where=f"{arm_id} plan")
        arm_summaries[arm_id] = {
            "value_trunk_grad_scale": float(scale),
            "effective_recipe_sha256": arm_plan["effective_recipe_sha256"],
            "command_sha256": arm_plan["command_sha256"],
        }
    campaign = {
        "schema_version": CAMPAIGN_PLAN_SCHEMA,
        "created_unix_ns": time.time_ns(),
        "status": "planned",
        "diagnostic_only": True,
        "promotion_eligible": False,
        "evaluation_eligible": False,
        "maximum_result": "matched_training_campaign_pending_selection",
        "campaign_id": CAMPAIGN_ID,
        "baseline_arm": BASELINE_ARM,
        "causal_axis": "value_trunk_grad_scale",
        "diagnostic_max_steps": int(diagnostic_max_steps),
        "arms": arm_refs,
        "arm_summaries": arm_summaries,
        "matched_contract": {
            "initializer": verified["initialization"],
            "data": scratch._ref(  # noqa: SLF001
                Path(str(verified["data_path"])), where="composite descriptor"
            ),
            "model": verified["model_construction"],
            "optimizer": str(baseline_recipe["optimizer"]),
            "rng_seed": int(baseline_recipe["seed"]),
            "execution_topology": verified["execution_topology"],
            "logical_recipe_sha256": scratch._value_sha256(  # noqa: SLF001
                verified["logical_recipe"]
            ),
            "only_causal_recipe_delta": "value_trunk_grad_scale",
            "campaign_common_recipe_delta": [
                "epochs",
                "max_steps",
                "checkpoint_steps",
            ],
            "independent_initialization_per_arm": True,
            "execution_order": list(ARMS),
            "execution_parallelism": "sequential_one_8gpu_ddp_arm_at_a_time",
        },
        "science_contract": {
            **scratch._ref(  # noqa: SLF001
                scratch.current_science.CONTRACT_PATH,
                where="current science contract",
            ),
            "semantic_sha256": scratch._value_sha256(  # noqa: SLF001
                scratch.current_science.load()
            ),
        },
        "python": python_authority,
        "code_binding": code_binding,
        "campaign_execution_receipt": str(root / "execution.receipt.json"),
    }
    scratch._write_receipt(plan_path, campaign)  # noqa: SLF001
    return campaign


def verify(plan_path: Path, *, require_fresh_outputs: bool) -> dict[str, Any]:
    plan = _load_receipt(plan_path, where="campaign plan")
    if (
        plan.get("schema_version") != CAMPAIGN_PLAN_SCHEMA
        or plan.get("status") != "planned"
        or plan.get("promotion_eligible") is not False
        or plan.get("causal_axis") != "value_trunk_grad_scale"
        or set(plan.get("arms", {})) != set(ARMS)
        or not 1 <= int(plan.get("diagnostic_max_steps", 0)) <= 256
    ):
        raise ValueRoutingCampaignError("campaign plan contract drift")
    arms: dict[str, dict[str, Any]] = {}
    baseline_recipe: dict[str, Any] | None = None
    for arm_id, scale in ARMS.items():
        _path, arm = _verify_receipt_ref(plan["arms"][arm_id], where=f"{arm_id} plan")
        if (
            arm.get("schema_version") != ARM_PLAN_SCHEMA
            or arm.get("arm_id") != arm_id
            or float(arm.get("value_trunk_grad_scale", -1.0)) != scale
            or arm.get("promotion_eligible") is not False
            or int(arm["effective_recipe"]["max_steps"])
            != int(plan["diagnostic_max_steps"])
            or arm.get("command_sha256") != scratch._value_sha256(arm.get("command"))  # noqa: SLF001
        ):
            raise ValueRoutingCampaignError(f"{arm_id} plan contract drift")
        recipe = dict(arm["effective_recipe"])
        if arm_id == BASELINE_ARM:
            baseline_recipe = recipe
        arms[arm_id] = arm
        if require_fresh_outputs:
            checkpoint = Path(str(arm["checkpoint"]))
            report = Path(str(arm["report"]))
            execution = Path(str(arm["execution_receipt"]))
            scratch._fresh_outputs(checkpoint, report, execution)  # noqa: SLF001
            scratch._require_fresh_epoch_outputs(  # noqa: SLF001
                checkpoint, int(recipe["epochs"])
            )
            scratch._require_fresh_step_outputs(  # noqa: SLF001
                checkpoint,
                scratch._checkpoint_steps(recipe),  # noqa: SLF001
            )
    assert baseline_recipe is not None
    for arm_id, scale in ARMS.items():
        _assert_one_axis_recipe(
            baseline_recipe,
            arms[arm_id]["effective_recipe"],
            scale=scale,
        )
    return {"plan": plan, "arms": arms}


def execute(
    plan_path: Path,
    *,
    campaign_execution_receipt: Path | None = None,
    runner: Any = subprocess.run,
) -> dict[str, Any]:
    verified = verify(plan_path, require_fresh_outputs=True)
    plan = verified["plan"]
    execution_path = (
        Path(str(plan["campaign_execution_receipt"]))
        if campaign_execution_receipt is None
        else campaign_execution_receipt.expanduser().resolve(strict=False)
    )
    if execution_path.exists() or execution_path.is_symlink():
        raise ValueRoutingCampaignError("campaign execution receipt must be fresh")
    arm_execution_refs: dict[str, dict[str, str]] = {}
    started = time.time_ns()
    for arm_id in ARMS:
        arm = verified["arms"][arm_id]
        arm_execution = Path(str(arm["execution_receipt"]))
        arm_started = time.time_ns()
        result = runner(arm["command"], cwd=REPO_ROOT, check=False)
        returncode = int(result.returncode)
        receipt = {
            **{key: value for key, value in arm.items() if key != "receipt_sha256"},
            "schema_version": ARM_EXECUTION_SCHEMA,
            "status": "completed" if returncode == 0 else "failed",
            "started_unix_ns": arm_started,
            "finished_unix_ns": time.time_ns(),
            "returncode": returncode,
            "promotion_eligible": False,
            "evaluation_eligible": False,
        }
        if returncode == 0:
            recipe = dict(arm["effective_recipe"])
            receipt["outputs"] = scratch._completed_outputs(  # noqa: SLF001
                checkpoint=Path(str(arm["checkpoint"])),
                report=Path(str(arm["report"])),
                epochs=int(recipe["epochs"]),
                checkpoint_steps=scratch._checkpoint_steps(recipe),  # noqa: SLF001
            )
        scratch._write_receipt(arm_execution, receipt)  # noqa: SLF001
        arm_execution_refs[arm_id] = _receipt_ref(
            arm_execution, where=f"{arm_id} execution"
        )
        if returncode != 0:
            failed = {
                "schema_version": CAMPAIGN_EXECUTION_SCHEMA,
                "status": "failed",
                "promotion_eligible": False,
                "evaluation_eligible": False,
                "failed_arm": arm_id,
                "started_unix_ns": started,
                "finished_unix_ns": time.time_ns(),
                "arm_executions": arm_execution_refs,
            }
            scratch._write_receipt(execution_path, failed)  # noqa: SLF001
            raise ValueRoutingCampaignError(f"{arm_id} learner failed")
    completed = {
        "schema_version": CAMPAIGN_EXECUTION_SCHEMA,
        "status": "completed",
        "promotion_eligible": False,
        "evaluation_eligible": False,
        "maximum_result": "completed_training_pending_result_selection",
        "started_unix_ns": started,
        "finished_unix_ns": time.time_ns(),
        "campaign_plan": _receipt_ref(plan_path, where="campaign plan"),
        "arm_executions": arm_execution_refs,
    }
    scratch._write_receipt(execution_path, completed)  # noqa: SLF001
    return completed


def select(
    *,
    campaign_execution_receipt: Path,
    arm_id: str,
    evidence: Path,
    receipt: Path,
    rationale: str,
) -> dict[str, Any]:
    campaign = _load_receipt(
        campaign_execution_receipt, where="campaign execution receipt"
    )
    if (
        campaign.get("schema_version") != CAMPAIGN_EXECUTION_SCHEMA
        or campaign.get("status") != "completed"
        or set(campaign.get("arm_executions", {})) != set(ARMS)
    ):
        raise ValueRoutingCampaignError("selection requires all completed arms")
    if arm_id not in ARMS:
        raise ValueRoutingCampaignError(f"unknown arm {arm_id!r}")
    for current, ref in campaign["arm_executions"].items():
        _path, execution = _verify_receipt_ref(ref, where=f"{current} execution")
        if execution.get("status") != "completed":
            raise ValueRoutingCampaignError("selection requires completed arms")
    selection = {
        "schema_version": SELECTION_SCHEMA,
        "status": "selected_for_evaluation",
        "selected_arm": arm_id,
        "selected_value_trunk_grad_scale": ARMS[arm_id],
        "rationale": str(rationale),
        "evidence": scratch._ref(evidence, where="selection evidence"),  # noqa: SLF001
        "campaign_execution": _receipt_ref(
            campaign_execution_receipt, where="campaign execution receipt"
        ),
        "diagnostic_only": True,
        "evaluation_eligible": True,
        "promotion_eligible": False,
        "maximum_result": "evaluation_candidate_not_promotion_candidate",
    }
    scratch._write_receipt(receipt, selection)  # noqa: SLF001
    return selection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--lock", required=True, type=Path)
    prep.add_argument("--data", required=True, type=Path)
    prep.add_argument("--composite-build-receipt", required=True, type=Path)
    prep.add_argument(
        "--policy-target-quality-receipt",
        required=True,
        type=Path,
    )
    prep.add_argument("--output-root", required=True, type=Path)
    prep.add_argument("--plan", required=True, type=Path)
    prep.add_argument("--python", type=Path, default=Path(sys.executable))
    prep.add_argument("--max-steps", default=128, type=int)
    check = sub.add_parser("verify")
    check.add_argument("--plan", required=True, type=Path)
    run = sub.add_parser("run")
    run.add_argument("--plan", required=True, type=Path)
    run.add_argument("--execution-receipt", type=Path)
    choose = sub.add_parser("select")
    choose.add_argument("--campaign-execution-receipt", required=True, type=Path)
    choose.add_argument("--arm", required=True, choices=tuple(ARMS))
    choose.add_argument("--evidence", required=True, type=Path)
    choose.add_argument("--receipt", required=True, type=Path)
    choose.add_argument("--rationale", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.action == "prepare":
        result = prepare(
            lock=args.lock,
            data=args.data,
            composite_build_receipt=args.composite_build_receipt,
            policy_target_quality_receipt=args.policy_target_quality_receipt,
            output_root=args.output_root,
            plan_path=args.plan,
            python=args.python,
            diagnostic_max_steps=args.max_steps,
        )
    elif args.action == "verify":
        result = verify(args.plan, require_fresh_outputs=True)["plan"]
    elif args.action == "run":
        result = execute(
            args.plan,
            campaign_execution_receipt=args.execution_receipt,
        )
    else:
        result = select(
            campaign_execution_receipt=args.campaign_execution_receipt,
            arm_id=args.arm,
            evidence=args.evidence,
            receipt=args.receipt,
            rationale=args.rationale,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
