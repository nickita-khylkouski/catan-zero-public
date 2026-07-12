#!/usr/bin/env python3
"""Emit the smallest matched post-P1 optimization-vs-architecture matrix.

This is a non-launching diagnostic manifest.  It reuses the selected P1
full-update checkpoint as a zero-cost control and schedules exactly two new
sample-matched B200 runs.  No Cartesian sweep and no concurrent B200 launch is
authorized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence


SCHEMA = "a1-post-p1-optimization-architecture-plan-v1"
SAMPLE_DOSE = 4_194_304


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def build_plan(
    *, world_size: int = 8, local_batch_size: int = 512, grad_accum_steps: int = 1
) -> dict[str, Any]:
    if world_size != 8:
        raise ValueError("post-P1 diagnosis is bound to all eight B200 GPUs")
    if local_batch_size <= 0 or grad_accum_steps <= 0:
        raise ValueError("batch and accumulation must be positive")
    global_batch = world_size * local_batch_size * grad_accum_steps
    steps = math.ceil(SAMPLE_DOSE / global_batch)
    fixed = {
        "initialization": "authenticated f7 producer bytes, independently loaded per arm",
        "data": "exact P1 globally shuffled current-plus-gen3-replay descriptor",
        "sample_order": "same component/game/row sampler seed as P1",
        "sample_dose": SAMPLE_DOSE,
        "max_steps": steps,
        "world_size": world_size,
        "local_batch_size": local_batch_size,
        "global_batch_size": global_batch,
        "grad_accum_steps": grad_accum_steps,
        "optimizer": "fresh Adam",
        "lr": 3e-5,
        "lr_warmup_steps": 100,
        "lr_schedule": "flat",
        "weight_decay": 0.0,
        "amp": "bf16",
        "policy_kl_anchor_weight": "inherit exact selected P1 value",
        "policy_kl_anchor_direction": "forward",
        "forced_policy_effective_weight": 0.0,
        "forced_row_value_weight": 1.0,
        "per_game_policy_weight_mode": "sqrt",
        "per_game_value_weight_mode": "sqrt",
        "loser_sample_weight": 1.0,
        "value_target_lambda": 1.0,
        "q_loss_weight": 0.0,
        "aux_subgoal_heads": False,
        "aux_subgoal_loss_weight": 0.0,
        "graph_history_features": True,
        "mask_hidden_info": True,
        "validation": "same game-disjoint P1 validation identity",
    }
    arms = [
        {
            "arm_id": "FULL_CONTROL",
            "training": "reuse selected P1 checkpoint and report; no new B200 run",
            "recipe_delta": {"freeze_modules": ""},
            "purpose": "observed full-trunk optimization control",
        },
        {
            "arm_id": "HEAD_ONLY",
            "training": "new matched B200 run",
            "recipe_delta": {
                "freeze_modules": "trunk",
                "checkpoint_upgrade": "none",
                "action_module_lr_mult": 1.0,
            },
            "purpose": "test whether trunk drift causes external forgetting",
        },
        {
            "arm_id": "HEAD_GX1",
            "training": "new matched B200 run",
            "recipe_delta": {
                "freeze_modules": "trunk",
                "checkpoint_upgrade": "f69_upgrade_checkpoint_config.py --flags gather,cross:1",
                "checkpoint_upgrade_forward_max_diff": 0.0,
                "action_module_lr_mult": 1.0,
            },
            "purpose": "isolate action-local gather/cross capacity over HEAD_ONLY",
        },
    ]
    for arm in arms:
        arm["recipe_sha256"] = _digest({"fixed": fixed, "delta": arm["recipe_delta"]})
    evaluation = {
        "checkpoint_identity": (
            "calibrate candidate c_scale in {0.03,0.10}; bind the selected "
            "checkpoint-plus-operator identity before panels"
        ),
        "internal": {
            "pairs": 300,
            "games": 600,
            "opponent": "f7 producer",
            "seat_swapped": True,
            "common_random_numbers": True,
        },
        "external": {
            "pairs": 250,
            "games_per_candidate": 500,
            "opponent": "catanatron_value",
            "compare_to": "f7 on the identical seeds",
            "common_random_numbers": True,
        },
        "binding_floor": "external win-rate delta versus f7 >= -2 percentage points",
        "diagnostics": [
            "active-only teacher-gap and target-to-model KL",
            "value RMSE/calibration by phase and root width",
            "parameter drift split into trunk/action/value groups",
            "model-to-f7 KL on multi-action rows",
            "gradient clipping and non-finite telemetry",
        ],
    }
    decision = {
        "optimization": (
            "HEAD_ONLY clears the external floor and dominates FULL_CONTROL: "
            "trunk optimization/forgetting is causal"
        ),
        "architecture": (
            "HEAD_GX1 clears the external floor and improves over HEAD_ONLY on "
            "internal strength plus teacher-gap: action-local capacity is causal"
        ),
        "neither": (
            "both new arms fail the external floor: keep architecture unchanged "
            "and return to data/objective diagnosis"
        ),
        "escalation": (
            "advance at most one Pareto arm to 8,388,608 samples from the original "
            "f7 initialization; never continue an arm's optimizer state"
        ),
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "launch_authorized": False,
        "launch_condition": "P1 training and checkpoint write have completed",
        "gpu_schedule": [
            "reuse FULL_CONTROL without B200 compute",
            "run HEAD_ONLY on all 8 B200s",
            "run HEAD_GX1 on all 8 B200s only after HEAD_ONLY releases DDP",
            "evaluate arms in parallel on the H100 fleet; do not consume B200s",
        ],
        "fixed_recipe": fixed,
        "historical_feature_audit": {
            "trunk_lr_multiplier": (
                "not implemented in train_bc; --freeze-modules trunk is the only "
                "attributable protected-trunk diagnostic"
            ),
            "action_module_lr_multiplier": (
                "implemented, but f7 has no gather/cross action-local parameters; "
                "non-unit use on the baseline correctly fails closed"
            ),
            "action_target_gather": "implemented, warm-start-safe, f7 default OFF",
            "action_cross_attention": "implemented, warm-start-safe, f7 effective layers=0",
            "auxiliary_heads": "implemented but f7 OFF and P1 loss weight zero",
            "root_value_blend": (
                "implemented and loader-preserved; lambda remains 1 because existing "
                "large-corpus roots bind the rejected producer/operator identity"
            ),
            "graph_history_features": "already ON in P1; held fixed",
        },
        "arms": arms,
        "evaluation": evaluation,
        "early_stop": [
            "abort immediately on non-finite loss/gradient/parameters",
            "abort when post-warmup clipping exceeds 10% of optimizer steps",
            (
                "at the 2,097,152-sample midpoint, stop an arm only when both "
                "active teacher-gap and held-out loss are worse than FULL_CONTROL "
                "at the matched dose and its behavior drift is larger"
            ),
        ],
        "decision_rules": decision,
        "explicitly_deferred": {
            "auxiliary_heads": "objective/regularization change; conditional after diagnosis",
            "root_value_blend": "large corpus roots were generated under rejected c_scale=.03 identity",
            "hlgauss": "value-objective change; conditional after scalar stability",
            "relational_trunk": "larger architecture bundle than the attributable gather/cross probe",
            "model_scale": "only after corrected 35M underfitting is established",
        },
    }
    payload["plan_sha256"] = _digest(payload)
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--local-batch-size", type=int, default=512)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--output", default="-")
    args = parser.parse_args(argv)
    try:
        plan = build_plan(
            world_size=args.world_size,
            local_batch_size=args.local_batch_size,
            grad_accum_steps=args.grad_accum_steps,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    encoded = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if args.output == "-":
        print(encoded, end="")
        return
    path = Path(args.output)
    if path.exists() and path.read_text(encoding="utf-8") != encoded:
        raise SystemExit(f"REFUSED: existing plan differs: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
