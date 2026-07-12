#!/usr/bin/env python3
"""Emit the Pareto post-P1 learner/architecture probe sequence.

The first P1 checkpoint used loser_weight=.3, so it is evidence, not a clean
control for the independently successful loser_weight=1 learner.  This typed,
non-launching plan changes one causal axis at a time: restore loser targets,
restore the winning policy-active dose without increasing value dose, then add
the zero-init action-target gather that exposes topology already stored in the
corpus.  Full relational warm-starts and Cartesian sweeps are forbidden.
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
        "trunk_lr_mult": 1.0,
        "weight_decay": 0.0,
        "amp": "bf16",
        "policy_kl_anchor_weight": 0.0,
        "policy_kl_anchor_direction": "forward",
        "policy_loss_weight": 1.0,
        "soft_target_source": "policy",
        "soft_target_weight": 0.9,
        "soft_target_temperature": 0.7,
        "soft_target_min_legal_coverage": 0.5,
        "forced_policy_effective_weight": 0.0,
        "forced_action_weight": 0.1,
        "forced_row_value_weight": 1.0,
        "per_game_policy_weight": False,
        "per_game_policy_weight_mode": "equal",
        "per_game_value_weight": False,
        "per_game_value_weight_mode": "equal",
        "loser_sample_weight": 1.0,
        "winner_sample_weight": 1.0,
        "value_loss_weight": 0.25,
        "final_vp_loss_weight": 0.0,
        "value_lr_mult": 0.3,
        "value_target_lambda": 1.0,
        "truncated_vp_margin_value_weight": 0.25,
        "q_loss_weight": 0.0,
        "aux_subgoal_heads": False,
        "aux_subgoal_loss_weight": 0.0,
        "graph_history_feature_flag": True,
        "event_history_available": False,
        "event_history_semantics": (
            "Rust v1 emits constant-zero event tensors; the 806-wide schema flag "
            "must not be reported as observed history"
        ),
        "mask_hidden_info": True,
        "symmetry_augment": False,
        "validation": "same game-disjoint P1 validation identity",
        "initialization_policy": (
            "independent authenticated f7 producer bytes; checkpoint chaining forbidden"
        ),
        "lineage_dose_schema": "a1-lineage-dose-v1",
    }
    arms = [
        {
            "arm_id": "LEGACY_L03",
            "training": "reuse recovered P1 K0 checkpoint; no new B200 run",
            "recipe_delta": {"loser_sample_weight": 0.3},
            "purpose": (
                "observed asymmetric-forgetting evidence; never use as the "
                "loser=1 control"
            ),
        },
        {
            "arm_id": "L1_CONTROL",
            "training": "new matched B200 run",
            "recipe_delta": {
                "checkpoint_upgrade": "none",
                "policy_aux_active_batch_size": 0,
            },
            "purpose": "replicate the winning a73 loser-policy objective at one dose",
        },
        {
            "arm_id": "L1_POLICY_AUX",
            "training": "new matched B200 run",
            "recipe_delta": {
                "checkpoint_upgrade": "none",
                "policy_aux_active_batch_size": 128,
            },
            "purpose": (
                "match a73's ~1.55M active-policy exposures while holding value "
                "dose at 4.19M rows"
            ),
        },
        {
            "arm_id": "L1_GATHER",
            "training": "new matched B200 run",
            "recipe_delta": {
                "checkpoint_upgrade": "f69_upgrade_checkpoint_config.py --flags gather",
                "checkpoint_upgrade_forward_max_diff": 0.0,
                "policy_aux_active_batch_size": 128,
            },
            "purpose": (
                "isolate learned legal-action-to-target binding over L1_POLICY_AUX"
            ),
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
            "map_kind": "BASE",
            "seat_swapped": True,
            "common_random_numbers": True,
        },
        "external": {
            "pairs": 250,
            "games_per_candidate": 500,
            "opponent": "catanatron_value",
            "compare_to": "f7 on the identical seeds",
            "common_random_numbers": True,
            "map_kind": "TOURNAMENT",
        },
        "direct_tournament_bridge": {
            "pairs": 300,
            "opponent": "f7 producer",
            "map_kind": "TOURNAMENT",
            "purpose": "separate map specialization from opponent nontransitivity",
        },
        "binding_floor": "external win-rate delta versus f7 >= -2 percentage points",
        "diagnostics": [
            "active-only teacher-gap and target-to-model KL",
            "value RMSE/calibration by phase and root width",
            "parameter drift split into trunk/action/value groups",
            "model-to-f7 KL on multi-action rows",
            (
                "weighted policy-vs-value shared-trunk gradient norms, cosine, "
                "opposing-coordinate fraction, and per-block concentration; collect "
                "on a single-GPU/DDP diagnostic because FSDP flattens logical blocks"
            ),
            "gradient clipping and non-finite telemetry",
        ],
    }
    decision = {
        "loser_targets": (
            "L1_CONTROL repairs replay teacher-gap versus LEGACY_L03: suppressing "
            "loser search targets was causal"
        ),
        "active_policy_dose": (
            "L1_POLICY_AUX improves over L1_CONTROL at equal value dose: policy "
            "underexposure was causal"
        ),
        "target_binding": (
            "L1_GATHER improves topology-sensitive buckets over L1_POLICY_AUX: "
            "spatial state-action aliasing was causal"
        ),
        "neither": (
            "L1_CONTROL and L1_POLICY_AUX fail: inspect objective-specific trunk "
            "gradients and deployed value squash before architecture escalation"
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
            "reuse LEGACY_L03 without B200 compute",
            "run L1_CONTROL on all 8 B200s",
            "run L1_POLICY_AUX only after L1_CONTROL releases DDP",
            "run L1_GATHER only after L1_POLICY_AUX releases DDP",
            "evaluate arms in parallel on the H100 fleet; do not consume B200s",
        ],
        "fixed_recipe": fixed,
        "historical_feature_audit": {
            "trunk_lr_multiplier": (
                "implemented as --trunk-lr-mult for the exact canonical entity-graph "
                "trunk; default 1.0 preserves historical grouping and non-unit use "
                "fails closed on unsupported/empty trunks"
            ),
            "action_module_lr_multiplier": (
                "implemented, but f7 has no gather/cross action-local parameters; "
                "non-unit use on the baseline correctly fails closed"
            ),
            "action_target_gather": "implemented, warm-start-safe, f7 default OFF",
            "action_cross_attention": "implemented, warm-start-safe, f7 effective layers=0",
            "auxiliary_heads": "implemented but f7 OFF and P1 loss weight zero",
            "root_value_blend": (
                "implemented and loader-preserved; lambda remains 1 because stored "
                "roots are correlated, stale f7 search estimates and changing the "
                "blend would confound the trunk-localization experiment"
            ),
            "graph_history_features": (
                "flag/schema present but event payload proven absent; never claim "
                "history conditioning until a redacted v2 producer exists"
            ),
        },
        "arms": arms,
        "evaluation": evaluation,
        "early_stop": [
            "abort immediately on non-finite loss/gradient/parameters",
            (
                "do not abort on clipped fraction alone: Adam largely normalizes a "
                "common gradient rescale; require objective-specific trunk conflict "
                "or pathological parameter-update evidence"
            ),
            (
                "at the 2,097,152-sample midpoint, stop an arm only when both "
                "active teacher-gap and held-out loss are worse than FULL_CONTROL "
                "at the matched dose and its behavior drift is larger"
            ),
        ],
        "prelaunch_gradient_probe": {
            "purpose": (
                "distinguish destructive multi-task trunk interference from an "
                "ordinary learning-rate problem before spending another full dose"
            ),
            "initialization": "reload f7 independently",
            "data": "exact P1 authenticated composite and sampler",
            "execution": (
                "single GPU, 32 optimizer steps, diagnostics every step; diagnostic-only "
                "checkpoint must never enter candidate lineage"
            ),
            "primary_readouts": [
                "value_to_policy_grad_norm_ratio",
                "trunk_gradient_cosine",
                "blocks.0 policy/value norms and cosine",
                "combined pre-clip norm and actual block parameter delta",
            ],
            "interpretation": {
                "negative_cosine_or_block0_value_dominance": (
                    "value-head LR is not a fix; test value-facing trunk gradient "
                    "scaling/projection or head-only value repair"
                ),
                "aligned_balanced_gradients": (
                    "shared objective is not the proximal cause; prioritize total "
                    "dose/LR schedule and policy-active sample budget"
                ),
            },
        },
        "decision_rules": decision,
        "explicitly_deferred": {
            "auxiliary_heads": "objective/regularization change; conditional after diagnosis",
            "public_event_history": (
                "requires versioned redacted Rust/Python parity; current 261GB "
                "constant event payload is omission/storage debt, not signal"
            ),
            "root_value_blend": (
                "objective change deferred until trunk localization; root operator "
                "identity is valid but the estimates are correlated and stale"
            ),
            "hlgauss": "value-objective change; conditional after scalar stability",
            "relational_trunk": (
                "short warm-start is invalid: only 13.66% transfers and 86.34% is "
                "newly initialized; use gather-only or a function-preserving transplant"
            ),
            "model_scale": "only after corrected 35M underfitting is established",
        },
        "value_readout_probe": {
            "mismatch": "learner fits raw value while deployed search consumes tanh(raw)",
            "calibration": ["raw", "tanh", "clip"],
            "operator_panel": "matched tanh versus clip, same squash for both agents",
            "default_change_authorized": False,
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
