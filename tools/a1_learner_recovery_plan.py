#!/usr/bin/env python3
"""Emit the sealed, non-launching A1 learner-recovery experiment matrix.

This tool deliberately does not accept ``--go``.  It converts a measured B200
batch topology into sample-matched optimizer-step counts and records the exact
recipe deltas for successive-halving.  A separate reviewed executor may later
bind real corpus/checkpoint bytes to one selected arm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from typing import Any, Sequence


SCHEMA = "a1-learner-recovery-plan-v1"
SENTINEL_SAMPLES = 4_194_304
ESCALATED_SAMPLES = 8_388_608


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _steps(samples: int, global_batch: int) -> int:
    if samples <= 0 or global_batch <= 0:
        raise ValueError("samples and global batch must be positive")
    return int(math.ceil(samples / global_batch))


def _base_recipe() -> dict[str, Any]:
    return {
        "action_module_lr_mult": 2.0,
        "amp": "bf16",
        "forced_row_value_weight": 1.0,
        "grad_accum_steps": 1,
        "hlgauss_scalar_aux_loss_weight": 0.0,
        "loser_sample_weight": 1.0,
        "lr": 0.00003,
        "lr_schedule": "flat",
        "lr_warmup_steps": 100,
        "optimizer": "adam",
        "per_game_policy_weight": True,
        "per_game_policy_weight_mode": "sqrt",
        "per_game_value_weight": True,
        "per_game_value_weight_mode": "sqrt",
        "policy_kl_anchor_weight": 0.0,
        "policy_kl_anchor_direction": "forward",
        "q_loss_weight": 0.0,
        "value_head_type": "mse",
        "value_lr_mult": 1.0,
        "value_target_lambda": 1.0,
        "weight_decay": 0.0,
    }


def _arm(
    arm_id: str,
    stage: str,
    delta: dict[str, Any],
    *,
    samples: int,
    global_batch: int,
    condition: str,
) -> dict[str, Any]:
    recipe = _base_recipe()
    recipe.update(delta)
    recipe["max_steps"] = _steps(samples, global_batch)
    return {
        "arm_id": arm_id,
        "stage": stage,
        "condition": condition,
        "sample_dose": samples,
        "max_steps": recipe["max_steps"],
        "recipe": recipe,
        "recipe_sha256": _canonical_sha(recipe),
    }


def build_plan(*, world_size: int, local_batch_size: int, grad_accum_steps: int) -> dict[str, Any]:
    if world_size <= 0 or local_batch_size <= 0 or grad_accum_steps <= 0:
        raise ValueError("world size, local batch and accumulation must be positive")
    global_batch = world_size * local_batch_size * grad_accum_steps
    common = {
        "world_size": world_size,
        "local_batch_size": local_batch_size,
        "grad_accum_steps": grad_accum_steps,
        "global_batch_size": global_batch,
    }
    accumulation = {"grad_accum_steps": grad_accum_steps}
    arms = [
        _arm("K0", "P1_anchor", accumulation, samples=SENTINEL_SAMPLES,
             global_batch=global_batch, condition="always"),
        _arm("K3", "P1_anchor", {**accumulation, "policy_kl_anchor_weight": 0.03},
             samples=SENTINEL_SAMPLES, global_batch=global_batch, condition="always"),
        _arm("K10", "P1_anchor", {**accumulation, "policy_kl_anchor_weight": 0.10},
             samples=SENTINEL_SAMPLES, global_batch=global_batch, condition="always"),
        _arm("FULL_LR", "P2_trunk_drift",
             {**accumulation, "lr": 0.00006, "action_module_lr_mult": 1.0},
             samples=SENTINEL_SAMPLES, global_batch=global_batch,
             condition="run with winning P1 KL weight"),
        _arm("V75", "P3_search_value", {**accumulation, "value_target_lambda": 0.75},
             samples=SENTINEL_SAMPLES, global_batch=global_batch,
             condition="run with winning anti-forgetting recipe"),
        _arm("VH75", "P3_search_value",
             {**accumulation, "value_target_lambda": 0.75, "value_head_type": "hlgauss",
              "value_categorical_bins": 33, "value_hlgauss_sigma_ratio": 0.75},
             samples=SENTINEL_SAMPLES, global_batch=global_batch,
             condition="only if V75 is not harmful"),
        _arm("FORCED_V025", "P4_forced_value",
             {**accumulation, "forced_row_value_weight": 0.25}, samples=SENTINEL_SAMPLES,
             global_batch=global_batch, condition="only after P1-P3"),
    ]
    payload = {
        "schema_version": SCHEMA,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "launch_authorized": False,
        "topology": common,
        "sample_doses": {
            "sentinel": SENTINEL_SAMPLES,
            "escalated": ESCALATED_SAMPLES,
            "sentinel_steps": _steps(SENTINEL_SAMPLES, global_batch),
            "escalated_steps": _steps(ESCALATED_SAMPLES, global_batch),
        },
        "fixed_data_recipe": {
            "fresh_components": ["n128", "n256"],
            "global_shuffle": True,
            "validation_split": "game-disjoint",
            "old_gen3_replay_ratios_by_game": [0.0, 0.1, 0.2],
            "replay_stage": "P2 after anchor selection",
        },
        "prerequisites": [
            "KL anchor excludes single-legal-action rows from its denominator",
            "KL anchor uses forward KL(champion_prior || candidate) behavior distillation",
            "multi-action fast-PCR rows remain eligible anchor rehearsal rows",
        ],
        "arms": arms,
        "adjudication_order": [
            "external_population",
            "internal_champion",
            "active_teacher_gap",
            "value_calibration",
            "parameter_and_behavior_drift",
            "systems_efficiency",
        ],
        "q_loss_policy": (
            "fixed at zero until target-score provenance proves completed-Q return scale"
        ),
    }
    payload["plan_sha256"] = _canonical_sha(payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--local-batch-size", type=int, required=True)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--output", default="-")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
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
    else:
        from pathlib import Path

        path = Path(args.output)
        if path.exists() and path.read_text(encoding="utf-8") != encoded:
            raise SystemExit(f"REFUSED: existing plan differs: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
