#!/usr/bin/env python3
"""Emit the corrected, non-launching A1 pointer-aux commissioning plan.

The historical AUX2 plan is intentionally not representable here.  Its
absolute 54-way settlement target was predicted from a permutation-invariant
CLS vector, so different vertex relabelings could present the same readout
while requiring different labels.  The corrected experiment uses an
entity-pointer settlement head, commissions all new heads once, measures
shared-trunk gradient geometry without updating weights, and lets that sealed
measurement select the treatment coefficient before a matched AUX0/AUXT pair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import a1_pre_wave_contract as pre_wave  # noqa: E402


SCHEMA = "a1-aux-pointer-commissioning-plan-v2"
WORLD_SIZE = 8
LOCAL_BATCH_SIZE = 512
GRAD_ACCUM_STEPS = 1
GLOBAL_BATCH_SIZE = 4096
JOINT_SAMPLE_DOSE = 524_288
JOINT_OPTIMIZER_STEPS = 128
WARMUP_OPTIMIZER_STEPS = 128
POINTER_MODULE = "entity_graph.aux_subgoal_pointer_heads.v1"
CURRENT_PROMOTED_PARENT_SHA256 = pre_wave.HISTORICAL_V5_HANDOFF_FINGERPRINT[
    "checkpoint_sha256"
]
GEOMETRY_PROBE_BATCHES = 5
GEOMETRY_PROBE_BATCH_ROWS = 512
GEOMETRY_TARGET_NORM_RATIO = 0.05
GEOMETRY_MAX_OPPOSING_PROJECTION = 0.01
GEOMETRY_COEFFICIENT_MAX = 0.05
GEOMETRY_COEFFICIENT_QUANTUM = 0.001
TRAINABLE_HEAD_PREFIXES = (
    "aux_longest_road_head",
    "aux_largest_army_head",
    "aux_vp_in_n_head",
    "aux_next_settlement_pointer_head",
    "aux_robber_target_head",
)
AUX_FIELDS = (
    "aux_longest_road",
    "aux_largest_army",
    "aux_vp_in_n",
    "aux_next_settlement",
    "aux_robber_target",
)


def _digest(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def build_plan(
    *,
    world_size: int,
    local_batch_size: int,
    grad_accum_steps: int = GRAD_ACCUM_STEPS,
) -> dict[str, Any]:
    observed = (world_size, local_batch_size, grad_accum_steps)
    required = (WORLD_SIZE, LOCAL_BATCH_SIZE, GRAD_ACCUM_STEPS)
    if observed != required:
        raise ValueError(
            "corrected AUX commissioning requires exact FP32 8x512 accum1 "
            f"topology; observed={observed}"
        )

    plan: dict[str, Any] = {
        "schema_version": SCHEMA,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "launch_authorized": False,
        "retired_experiment": {
            "arm": "AUX2",
            "coefficient": 0.02,
            "module": "entity_graph.aux_subgoal_heads.v1",
            "reason": "absolute_vertex_target_from_permutation_invariant_cls",
        },
        "initializer": {
            "causal_parent": {
                "source": (
                    "independent reload of exact recovered current promoted v5 "
                    "handoff parent"
                ),
                "sha256": CURRENT_PROMOTED_PARENT_SHA256,
                "role": "declared_producer_and_raw_lineage_parent",
            },
            "public_award_transition": {
                "required": True,
                "source": "exact raw causal parent",
                "source_feature_contract": "legacy_zero_v0",
                "target": "immutable transitioned checkpoint",
                "target_feature_contract": "authoritative_v1",
                "changed_input_column_index": 12,
                "changed_input_column_initializer": "exact_zero",
                "optimizer_steps": 0,
                "legacy_zero_input_function_preserving": True,
                "candidate_or_promotion_parent": False,
                "immutable_receipt_replay_required": True,
            },
            "pointer_upgrade": {
                "source": "exact public-award transitioned checkpoint",
                "module": POINTER_MODULE,
                "function_preserving_upgrade_required": True,
                "main_output_max_abs_diff": 0.0,
                "immutable_receipt_replay_required": True,
            },
            "causal_chain": [
                "raw promoted parent",
                "zero-optimizer public-award transition",
                "function-preserving pointer upgrade",
                "head-only warmed pointer checkpoint",
            ],
            "candidate_chaining": False,
            "old_f7_rollback_forbidden": True,
            "unpromoted_p1_candidate_forbidden": True,
            "p1_contributes_recipe_and_data_only": True,
            "diagnostic_pair_initializer": "exact shared warmed pointer checkpoint",
            "final_initializer_by_selected_aux_decision": {
                "AUX0": (
                    "exact public-award transitioned checkpoint; no diagnostic "
                    "checkpoint, pointer upgrade, or warmup"
                ),
                "AUXT": (
                    "exact reference warmed pointer checkpoint; no diagnostic arm "
                    "checkpoint"
                ),
            },
        },
        "topology": {
            "world_size": WORLD_SIZE,
            "local_batch_size": LOCAL_BATCH_SIZE,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "precision": "fp32",
            "ddp_find_unused_parameters": False,
        },
        "head_only_warmup": {
            "optimizer_steps": WARMUP_OPTIMIZER_STEPS,
            "terminal": "fixed_exact_step",
            "trainable_parameter_prefixes": list(TRAINABLE_HEAD_PREFIXES),
            "inherited_parameter_update_count": 0,
            "inherited_tensors_bit_identical_required": True,
            "shared_warmed_checkpoint_required": True,
            "optimizer_sidecar_discarded": True,
        },
        "gradient_geometry": {
            "updates_weights": False,
            "initializer": "exact shared warmed checkpoint",
            "probe": {
                "batch_count": GEOMETRY_PROBE_BATCHES,
                "rows_per_batch": GEOMETRY_PROBE_BATCH_ROWS,
                "row_order": "exact authenticated joint sampler order prefix",
                "seed": "exact authenticated joint sampler seed",
                "manifest_sha256_required": True,
            },
            "shared_parameter_surface": {
                "include": "inherited entity-graph shared trunk parameters",
                "exclude": [
                    "all auxiliary readouts",
                    "all policy readouts",
                    "all value readouts",
                    "all action-only encoders/adapters/readouts",
                ],
                "exact_sorted_name_shape_dtype_digest_required": True,
                "per_batch_surface_must_match": True,
            },
            "same_forward_main_and_unit_aux_gradients": True,
            "canonical_ddp_aggregation": (
                "all-reduce global sufficient-statistic numerators and denominators "
                "before forming either gradient"
            ),
            "required_statistics": [
                "main_gradient_norm",
                "unit_aux_gradient_norm",
                "gradient_dot_product",
                "gradient_cosine",
                "conflict_fraction",
            ],
            "selector": {
                "epsilon": 1e-12,
                "norm_ratio": "r = ||g_aux_unit|| / ||g_main||",
                "cosine": "cos = dot(g_main,g_aux_unit)/(||g_main||*||g_aux_unit||)",
                "formula": (
                    "raw=min(0.05/r, 0.01/max(-r*cos,epsilon) if cos<0 "
                    "else +inf, 0.05)"
                ),
                "target_aux_to_main_norm_ratio": GEOMETRY_TARGET_NORM_RATIO,
                "max_opposing_main_projection": GEOMETRY_MAX_OPPOSING_PROJECTION,
                "coefficient_max": GEOMETRY_COEFFICIENT_MAX,
                "quantization": "floor_down",
                "quantum": GEOMETRY_COEFFICIENT_QUANTUM,
                "refuse_if": [
                    "either gradient norm, dot, cosine, r, or raw is nonfinite",
                    "either gradient norm is zero",
                    "raw < 0.001",
                    "quantized coefficient < 0.001 or > 0.05",
                    "probe manifest/order/seed or parameter surface differs by batch",
                ],
                "operator_cherry_pick": False,
                "output": "selected_aux_coefficient_decimal",
            },
        },
        "joint_pair": {
            "arms": [
                {
                    "arm_id": "AUX0",
                    "aux_subgoal_loss_weight": 0.0,
                    "purpose": "pointer-heads-present matched control",
                },
                {
                    "arm_id": "AUXT",
                    "aux_subgoal_loss_weight": "geometry_selected",
                    "purpose": "predeclared geometry-selected treatment",
                },
            ],
            "sample_dose_per_arm": JOINT_SAMPLE_DOSE,
            "optimizer_steps_per_arm": JOINT_OPTIMIZER_STEPS,
            "fresh_adam": True,
            "resume_optimizer": False,
            "same_warmed_checkpoint": True,
            "same_rows_order_data_split_sampler": True,
            "only_scientific_delta": "aux_subgoal_loss_weight",
            "diagnostic_only": True,
            "promotion_eligible": False,
            "requires_independent_final_replication": True,
        },
        "data_authority": {
            "required_fields": list(AUX_FIELDS),
            "production_component_ratios": {
                "current_producer": 0.64,
                "recent_history": 0.12,
                "hard_negative": 0.04,
                "historical_replay": 0.20,
            },
            "selected_p1_recipe_authority_required": True,
            "stale_recovery_plan_rejected": True,
            "generic_post_p1_dose_evidence_is_not_component_authority": True,
        },
        "execution_protocol": {
            "central_claim_required_before_execution": True,
            "published_executor_authority_required": [
                "P1",
                "WARMUP",
                "GEOMETRY",
                "AUX0",
                "AUXT",
                "FINAL",
            ],
            "post_authority_execution_commitment_required": [
                "P1",
                "WARMUP",
                "GEOMETRY",
                "AUX0",
                "AUXT",
                "FINAL",
            ],
            "post_authority_commitment_binds": [
                "exact rendered argv",
                "complete allowlisted environment",
                "fresh output namespace",
                "published executor authority file and semantic digest",
            ],
            "one_dose_exact_execution_binding_required": [
                "P1",
                "AUX0",
                "AUXT",
                "FINAL",
            ],
            "live_allocation_replay_required": {
                "hostname": True,
                "machine_id": True,
                "gpu_uuid_by_physical_index": True,
                "pci_bus_id_by_physical_index": True,
            },
            "all_eight_physical_gpu_locks_required": True,
            "preexisting_or_symlink_output_refused": True,
            "crash_resume_may_terminalize_but_never_rerun": True,
        },
        "fixed_pair_evaluation": {
            "claim_required": True,
            "baseline": "exact raw recovered current parent",
            "common_random_numbers": True,
            "seat_swapped": True,
            "search": {
                "engine": "native_rust_information_set_search",
                "n_full": 128,
                "particle_count": 4,
                "d6_root_averaging": True,
                "d6_minimum_legal_width": 20,
                "selection_tuning_allowed": False,
            },
            "internal": {
                "pairs_per_arm": 300,
                "map_kind": "BASE",
                "opponent": "recovered_generator_reference",
                "decision": "AUXT pair points strictly greater than AUX0",
            },
            "external": {
                "pairs_per_arm": 250,
                "games_per_arm": 500,
                "map_kind": "TOURNAMENT",
                "opponent": "catanatron_value",
                "non_regression_tolerance_milli": 25,
            },
            "raw_game_receipt_replay_required": True,
            "operator_cherry_pick": False,
        },
        "independent_final_replication": {
            "issued_only_after_fixed_pair_evaluation": True,
            "diagnostic_arm_checkpoint_as_initializer_forbidden": True,
            "base_parent_lineage_reloaded": True,
            "fresh_component_routing_receipt_required": True,
            "fresh_sampler_seed_order_and_row_set_required": True,
            "sample_dose": JOINT_SAMPLE_DOSE,
            "optimizer_steps": JOINT_OPTIMIZER_STEPS,
            "fresh_adam": True,
            "resume_optimizer": False,
            "selected_recipe_only": True,
            "learned_public_award_signal_required": True,
            "auto_promotion": False,
        },
        "full_gate": {
            "required_after_final_replication": True,
            "fresh_cohorts_required": True,
            "dual_baseline_conjunctive": True,
            "strict_h1_over_recovered_parent": True,
            "f7_non_regression_veto": True,
            "f7_veto_complete_pairs": 300,
            "ordinary_training_calibration_external_and_regret_gates_required": True,
            "promotion_eligible_only_after_gate_authority_replays": True,
            "auto_promotion": False,
        },
        "lifecycle": [
            "prepare_p1_sweep",
            "claim_p1_arm:K0",
            "load_p1_arm_executor_authority:K0",
            "commit_one_dose_execution:P1:K0",
            "execute_and_complete_p1_arm:K0",
            "claim_p1_arm:K3",
            "load_p1_arm_executor_authority:K3",
            "commit_one_dose_execution:P1:K3",
            "execute_and_complete_p1_arm:K3",
            "claim_p1_arm:K10",
            "load_p1_arm_executor_authority:K10",
            "commit_one_dose_execution:P1:K10",
            "execute_and_complete_p1_arm:K10",
            "claim_p1_evaluation",
            "complete_p1_evaluation:fixed_internal_and_external",
            "adjudicate_p1_sweep",
            "prepare_experiment",
            "claim_warmup",
            "load_warmup_executor_authority",
            "commit_stage_execution:WARMUP",
            "execute_warmup",
            "complete_warmup",
            "claim_geometry",
            "load_geometry_executor_authority",
            "commit_stage_execution:GEOMETRY",
            "execute_geometry",
            "complete_geometry",
            "issue_pair",
            "claim_arm:AUX0",
            "load_aux_pair_executor_authority:AUX0",
            "commit_one_dose_execution:AUX0",
            "execute_and_complete_arm:AUX0",
            "claim_arm:AUXT",
            "load_aux_pair_executor_authority:AUXT",
            "commit_one_dose_execution:AUXT",
            "execute_and_complete_arm:AUXT",
            "claim_pair_evaluation",
            "complete_pair_evaluation:fixed_internal_and_external",
            "finalize_pair",
            "issue_final_replication",
            "claim_final_replication",
            "load_final_replication_executor_authority",
            "commit_one_dose_execution:FINAL",
            "execute_and_complete_final_replication",
            "verify_fresh_dual_baseline_full_gate",
            "load_final_gate_entry_authority",
        ],
    }
    plan["plan_sha256"] = _digest(plan)
    return plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, default=WORLD_SIZE)
    parser.add_argument("--local-batch-size", type=int, default=LOCAL_BATCH_SIZE)
    parser.add_argument("--grad-accum-steps", type=int, default=GRAD_ACCUM_STEPS)
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
