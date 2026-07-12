#!/usr/bin/env python3
"""Emit the sealed, non-launching A1 auxiliary-regularization probe.

The probe is intentionally downstream of P1: it inherits the selected P1
learner recipe and changes only whether the existing CAT-100 auxiliary heads
are present and trained.  This tool plans the experiment; it cannot launch it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence


SCHEMA = "a1-aux-regularization-probe-v1"
SAMPLE_DOSE = 4_194_304
AUX_WEIGHT = 0.02
AUX_FIELDS = (
    "aux_longest_road",
    "aux_largest_army",
    "aux_vp_in_n",
    "aux_next_settlement",
    "aux_robber_target",
)


def _digest(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _steps(samples: int, global_batch: int) -> int:
    if samples <= 0 or global_batch <= 0:
        raise ValueError("samples and global batch must be positive")
    return int(math.ceil(samples / global_batch))


def build_plan(
    *,
    world_size: int,
    local_batch_size: int,
    grad_accum_steps: int = 1,
) -> dict[str, Any]:
    if world_size <= 0 or local_batch_size <= 0 or grad_accum_steps <= 0:
        raise ValueError("world size, local batch and accumulation must be positive")
    global_batch = world_size * local_batch_size * grad_accum_steps
    max_steps = _steps(SAMPLE_DOSE, global_batch)

    common = {
        "inherit_recipe_from": "authenticated P1 winner",
        "sample_dose": SAMPLE_DOSE,
        "max_steps": max_steps,
        "data": "same authenticated n128+n256+gen3-replay composite selected by P1",
        "validation": "same deterministic 262144-row game-disjoint sentinel and common-random-number panels",
        "validation_max_samples": 262_144,
        "initialization": "same authenticated P1 parent checkpoint bytes",
        "q_loss_weight": 0.0,
        "value_target_lambda": 1.0,
    }
    arms = [
        {
            "arm_id": "AUX0",
            "purpose": "matched P1 control",
            "recipe_delta": {
                "aux_subgoal_heads": False,
                "aux_subgoal_loss_weight": 0.0,
            },
        },
        {
            "arm_id": "AUX2",
            "purpose": "five-head realized/subgoal regularization",
            "recipe_delta": {
                "aux_subgoal_heads": True,
                "aux_subgoal_loss_weight": AUX_WEIGHT,
                "checkpoint_upgrade": "f69_upgrade_checkpoint_config.py --flags aux",
            },
        },
    ]
    for arm in arms:
        arm["fixed_recipe"] = dict(common)
        arm["recipe_sha256"] = _digest(
            {**common, **arm["recipe_delta"]}
        )

    plan: dict[str, Any] = {
        "schema_version": SCHEMA,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "launch_authorized": False,
        "stage": "after P1 anti-forgetting recipe selection",
        "topology": {
            "world_size": world_size,
            "local_batch_size": local_batch_size,
            "grad_accum_steps": grad_accum_steps,
            "global_batch_size": global_batch,
        },
        "sample_dose": SAMPLE_DOSE,
        "max_steps": max_steps,
        "required_corpus_fields": list(AUX_FIELDS),
        "corpus_admission": {
            "binary_and_vp_finite_fraction_min": 0.999,
            "next_settlement_valid_fraction_min": 0.69,
            "robber_target_valid_fraction_min": 0.89,
            "mask_hidden_info": True,
        },
        "arms": arms,
        "single_variable_claim": (
            "AUX2 adds only the behaviorally-disconnected auxiliary readouts and "
            "their 0.02 summed loss; value/policy outputs must be bit-identical "
            "immediately after checkpoint upgrade"
        ),
        "prerequisites": [
            "P1 winner receipt and recipe hash are bound before execution",
            "all five auxiliary columns pass the declared coverage floors",
            "checkpoint upgrade reports zero main-output forward difference",
            "both arms see exactly the same sample dose, data mix, validation games, and panels",
            "no root-value, Q, architecture, replay-ratio, or forced-row change is bundled",
        ],
        "adjudication": {
            "binding": [
                "external population non-regression",
                "internal candidate-vs-champion result",
            ],
            "diagnostic": [
                "active teacher-gap closure",
                "value calibration by phase/root width",
                "main-trunk parameter drift",
                "combined auxiliary validation loss and per-head label coverage",
            ],
            "advance_rule": (
                "advance AUX2 only if it is externally non-regressing and improves "
                "internal strength or a predeclared calibration/teacher-gap metric; "
                "validation loss alone cannot select it"
            ),
        },
        "followup": (
            "if AUX2 clears the sentinel, repeat AUX0/AUX2 with two independent "
            "module seeds before making auxiliary regularization a default"
        ),
    }
    plan["plan_sha256"] = _digest(plan)
    return plan


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
        return
    path = Path(args.output)
    if path.exists() and path.read_text(encoding="utf-8") != encoded:
        raise SystemExit(f"REFUSED: existing plan differs: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
