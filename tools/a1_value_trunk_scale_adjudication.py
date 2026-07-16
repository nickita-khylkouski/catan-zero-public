#!/usr/bin/env python3
"""Seal the 1.0-vs-0.1 value-to-trunk gradient adjudication boundary.

Short H100 probes consistently favor ``value_trunk_grad_scale=0.1``, while the
current coherent production contract remains at ``1.0`` and the Stage-C
diagnostic recipe already uses ``0.1``.  Those probes share a two-game heldout
and therefore cannot authorize a production change.

This module makes that unresolved authority conflict executable as a plan.  It
defines an exact one-axis comparison and names the existing validators that a
future adjudicator must replay.  It deliberately has no receipt adjudicator:
self-asserted booleans, counts, or decisions can never create authority here.
It never edits the production contract.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
for root in (REPO_ROOT, REPO_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from tools import a1_b200_stage_c_learner_campaign as stage_c  # noqa: E402
from tools import a1_current_science_contract as current_science  # noqa: E402


SCHEMA = "a1-value-trunk-scale-adjudication-contract-v2"
CONTROL_ARM = "production-scale-1.0"
TREATMENT_ARM = "protected-scale-0.1"
CONTROL_SCALE = 1.0
TREATMENT_SCALE = 0.1
PRODUCTION_ROOT_COUNT = 65_536
MIN_MATCHED_SEEDS = 3
MIN_VALIDATION_GAMES = 512
MIN_POLICY_ACTIVE_VALIDATION_ROOTS = 4_096
MIN_GAMES_WITH_EIGHT_ROOTS_FRACTION = 0.95
MIN_PAIRED_GAMEPLAY_PAIRS = 600
SHA_PREFIX = "sha256:"


class ValueTrunkScaleError(RuntimeError):
    """The value-trunk comparison or its evidence is not the sealed treatment."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def value_sha256(value: object) -> str:
    return SHA_PREFIX + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _arm_recipe(recipe: Mapping[str, Any], scale: float) -> dict[str, Any]:
    result = copy.deepcopy(dict(recipe))
    result["value_trunk_grad_scale"] = scale
    return result


def build_contract() -> dict[str, Any]:
    production = current_science.learner_training_recipe()
    stage_c_recipe = stage_c._recipe()  # noqa: SLF001
    production_scale = production.get("value_trunk_grad_scale")
    diagnostic_scale = stage_c_recipe.get("value_trunk_grad_scale")
    if production_scale != CONTROL_SCALE or diagnostic_scale != TREATMENT_SCALE:
        raise ValueTrunkScaleError(
            "value-trunk authority conflict changed; issue a reviewed contract "
            "revision instead of silently rewriting this adjudication"
        )

    control = _arm_recipe(production, CONTROL_SCALE)
    treatment = _arm_recipe(production, TREATMENT_SCALE)
    differing = sorted(
        key for key in set(control) | set(treatment) if control.get(key) != treatment.get(key)
    )
    if differing != ["value_trunk_grad_scale"]:
        raise ValueTrunkScaleError("value-trunk comparison is not one-axis")

    body: dict[str, Any] = {
        "schema_version": SCHEMA,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "production_contract_mutated": False,
        "authority_conflict": {
            "current_production_scale": CONTROL_SCALE,
            "stage_c_diagnostic_scale": TREATMENT_SCALE,
            "status": "unresolved_requires_broad_adjudication",
        },
        "arms": {
            CONTROL_ARM: {
                "role": "current_production_control",
                "recipe": control,
                "recipe_sha256": value_sha256(control),
            },
            TREATMENT_ARM: {
                "role": "shared_trunk_value_gradient_protection",
                "recipe": treatment,
                "recipe_sha256": value_sha256(treatment),
            },
        },
        "only_declared_delta": {
            "field": "value_trunk_grad_scale",
            "control": CONTROL_SCALE,
            "treatment": TREATMENT_SCALE,
            "forward_value_identity": True,
            "value_head_parameter_gradient_scale": 1.0,
            "policy_gradient_unchanged": True,
        },
        "narrow_probe_evidence": {
            "decision_authority": False,
            "shared_limit": {
                "validation_games": 2,
                "policy_active_validation_roots": 26,
                "reason": "same tiny heldout cannot estimate broad production generalization",
            },
            "matched_16_step_seeds": [
                {
                    "sampler_seed": 1,
                    "control_report_sha256": (
                        "sha256:6a99eea8a408081047c3781b35d63a4cf9be2c5d930a4199c24ef1bfe9d3265f"
                    ),
                    "treatment_report_sha256": (
                        "sha256:ecfeef6e1c1d7a6e59e728403bb1713f4c0fb91ee9bdf075b940f1ec62d9ed9c"
                    ),
                    "control_policy_loss": 1.4026341053860634,
                    "treatment_policy_loss": 1.3926561294623612,
                    "control_value_mse": 0.4484739535231123,
                    "treatment_value_mse": 0.28066606440031405,
                    "control_value_to_policy_grad_ratio": 0.41140905344372153,
                    "treatment_value_to_policy_grad_ratio": 0.06703086503964681,
                },
                {
                    "sampler_seed": 3,
                    "control_report_sha256": (
                        "sha256:2f04a4b30d4a1d928232f3917134571531a20f499126bec7e08e2c2aab7670c4"
                    ),
                    "treatment_report_sha256": (
                        "sha256:9e482de0ff69e48872ad7b821760880aa525ed47083230cc0deec78b3e08b725"
                    ),
                    "control_policy_loss": 1.4099570345844954,
                    "treatment_policy_loss": 1.406685637726573,
                    "control_value_mse": 0.45752963344475345,
                    "treatment_value_mse": 0.24016305511865743,
                    "control_value_to_policy_grad_ratio": 0.40956950066934505,
                    "treatment_value_to_policy_grad_ratio": 0.08018686341620004,
                },
            ],
            "interpretation": (
                "repeated causal evidence that scale 0.1 reduces shared-trunk "
                "value interference; insufficient breadth for production selection"
            ),
        },
        "required_learner_evidence": {
            "matched_independent_seeds": MIN_MATCHED_SEEDS,
            "requested_root_count": PRODUCTION_ROOT_COUNT,
            "selected_root_count": PRODUCTION_ROOT_COUNT,
            "minimum_validation_games": MIN_VALIDATION_GAMES,
            "minimum_policy_active_validation_roots": (
                MIN_POLICY_ACTIVE_VALIDATION_ROOTS
            ),
            "minimum_train_games_with_eight_roots_fraction": (
                MIN_GAMES_WITH_EIGHT_ROOTS_FRACTION
            ),
            "minimum_validation_games_with_eight_roots_fraction": (
                MIN_GAMES_WITH_EIGHT_ROOTS_FRACTION
            ),
            "training_evaluation_game_overlap": 0,
            "value_player_outcome_balance_mode": "sampler_balanced_v1",
            "treatment_must_not_regress_policy_objective": True,
            "treatment_must_improve_value_mse": True,
            "required_artifact_reference_fields": [
                "path",
                "file_sha256",
                "schema_version",
                "semantic_digest",
            ],
            "validators_to_replay": [
                "tools.a1_stage_c_final_replication.verify_root_manifest",
                "tools.a1_stage_c_final_replication.verify_final_corpus_admission",
                "tools.a1_stage_c_final_replication.verify_final_authority",
            ],
            "derived_not_asserted": [
                "initializer_identity",
                "optimizer_identity",
                "sample_order_identity",
                "objective_identity",
                "requested_root_count",
                "selected_root_count",
                "validation_game_count",
                "policy_active_validation_root_count",
                "root_breadth_fractions",
                "target_manifest_coverage",
                "optimizer_exclusion",
                "validation_split",
                "training_evaluation_overlap",
                "checkpoint_metric_binding",
                "policy_non_regression",
                "value_mse_improvement",
            ],
        },
        "required_gameplay_evidence": {
            "comparison": "paired_same_seed_color_swap_shared_search_operator",
            "minimum_complete_pairs": MIN_PAIRED_GAMEPLAY_PAIRS,
            "minimum_games": 2 * MIN_PAIRED_GAMEPLAY_PAIRS,
            "required_decision": "superiority_pentanomial_sprt_decision_H1",
            "required_artifact_reference_fields": [
                "path",
                "file_sha256",
                "schema_version",
                "semantic_digest",
            ],
            "validators_to_replay": [
                "tools.a1_stage_c_final_replication.verify_adjudication",
                "tools.a1_promotion_transaction._verify_promotion_evidence",
            ],
            "derived_not_asserted": [
                "comparison_identity",
                "complete_pair_count",
                "game_count",
                "sprt_decision",
                "promotion_gate_result",
            ],
        },
        "adjudication_result_boundary": {
            "implementation": "plan_only_no_adjudicator",
            "maximum_result": "none_until_artifact_replay_is_implemented",
            "automatic_production_flip": False,
            "automatic_promotion": False,
        },
    }
    body["contract_sha256"] = value_sha256(body)
    return body


def verify_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = build_contract()
    if dict(value) != expected:
        raise ValueTrunkScaleError("value-trunk adjudication contract drift")
    stated = value.get("contract_sha256")
    unhashed = dict(value)
    unhashed.pop("contract_sha256", None)
    if stated != value_sha256(unhashed):
        raise ValueTrunkScaleError("value-trunk adjudication digest drift")
    return expected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("contract")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    build_parser().parse_args(argv)
    result = build_contract()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
