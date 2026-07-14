from __future__ import annotations

import pytest

from tools.a1_aux_regularization_plan import (
    AUX_FIELDS,
    CURRENT_PROMOTED_PARENT_SHA256,
    GLOBAL_BATCH_SIZE,
    JOINT_OPTIMIZER_STEPS,
    JOINT_SAMPLE_DOSE,
    POINTER_MODULE,
    TRAINABLE_HEAD_PREFIXES,
    WARMUP_OPTIMIZER_STEPS,
    build_plan,
)
from tools.a1_pre_wave_contract import HISTORICAL_V5_HANDOFF_FINGERPRINT


def test_corrected_probe_is_pointer_warmup_geometry_then_matched_pair():
    plan = build_plan(world_size=8, local_batch_size=512)
    initializer = plan["initializer"]
    assert initializer["pointer_upgrade"]["module"] == POINTER_MODULE
    assert initializer["causal_parent"]["sha256"] == CURRENT_PROMOTED_PARENT_SHA256
    assert initializer["causal_parent"]["sha256"] == (
        HISTORICAL_V5_HANDOFF_FINGERPRINT["checkpoint_sha256"]
    )
    assert initializer["old_f7_rollback_forbidden"] is True
    assert initializer["unpromoted_p1_candidate_forbidden"] is True
    assert initializer["p1_contributes_recipe_and_data_only"] is True
    assert plan["topology"]["global_batch_size"] == GLOBAL_BATCH_SIZE
    assert plan["topology"]["precision"] == "fp32"
    assert plan["head_only_warmup"] == {
        "optimizer_steps": WARMUP_OPTIMIZER_STEPS,
        "terminal": "fixed_exact_step",
        "trainable_parameter_prefixes": list(TRAINABLE_HEAD_PREFIXES),
        "inherited_parameter_update_count": 0,
        "inherited_tensors_bit_identical_required": True,
        "shared_warmed_checkpoint_required": True,
        "optimizer_sidecar_discarded": True,
    }
    assert plan["gradient_geometry"]["updates_weights"] is False
    geometry = plan["gradient_geometry"]
    assert geometry["probe"] == {
        "batch_count": 5,
        "rows_per_batch": 512,
        "row_order": "exact authenticated joint sampler order prefix",
        "seed": "exact authenticated joint sampler seed",
        "manifest_sha256_required": True,
    }
    assert geometry["shared_parameter_surface"]["include"] == (
        "inherited entity-graph shared trunk parameters"
    )
    assert geometry["selector"]["formula"].startswith("raw=min(0.05/r")
    assert geometry["selector"]["quantization"] == "floor_down"
    assert geometry["selector"]["quantum"] == pytest.approx(0.001)
    assert geometry["selector"]["operator_cherry_pick"] is False
    pair = plan["joint_pair"]
    assert [arm["arm_id"] for arm in pair["arms"]] == ["AUX0", "AUXT"]
    assert pair["arms"][0]["aux_subgoal_loss_weight"] == 0.0
    assert pair["arms"][1]["aux_subgoal_loss_weight"] == "geometry_selected"
    assert pair["sample_dose_per_arm"] == JOINT_SAMPLE_DOSE
    assert pair["optimizer_steps_per_arm"] == JOINT_OPTIMIZER_STEPS
    assert pair["only_scientific_delta"] == "aux_subgoal_loss_weight"


def test_plan_separates_data_recipe_authority_from_generic_dose_evidence():
    plan = build_plan(world_size=8, local_batch_size=512)
    authority = plan["data_authority"]
    assert authority["required_fields"] == list(AUX_FIELDS)
    assert authority["production_component_ratios"] == {
        "current_producer": 0.64,
        "recent_history": 0.12,
        "hard_negative": 0.04,
        "historical_replay": 0.20,
    }
    assert authority["selected_p1_recipe_authority_required"] is True
    assert authority["stale_recovery_plan_rejected"] is True
    assert authority["generic_post_p1_dose_evidence_is_not_component_authority"] is True


def test_probe_cannot_authorize_launch_or_promotion():
    plan = build_plan(world_size=8, local_batch_size=512)
    assert plan["diagnostic_only"] is True
    assert plan["launch_authorized"] is False
    assert plan["promotion_eligible"] is False
    assert plan["initializer"]["candidate_chaining"] is False


def test_plan_binds_exact_raw_transition_pointer_warmup_causality():
    initializer = build_plan(world_size=8, local_batch_size=512)["initializer"]
    assert initializer["causal_chain"] == [
        "raw promoted parent",
        "zero-optimizer public-award transition",
        "function-preserving pointer upgrade",
        "head-only warmed pointer checkpoint",
    ]
    transition = initializer["public_award_transition"]
    assert transition == {
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
    }
    assert initializer["pointer_upgrade"]["source"] == (
        "exact public-award transitioned checkpoint"
    )
    assert initializer["pointer_upgrade"]["main_output_max_abs_diff"] == 0.0
    assert initializer["diagnostic_pair_initializer"] == (
        "exact shared warmed pointer checkpoint"
    )
    assert "transitioned checkpoint" in (
        initializer["final_initializer_by_selected_aux_decision"]["AUX0"]
    )
    assert "warmed pointer checkpoint" in (
        initializer["final_initializer_by_selected_aux_decision"]["AUXT"]
    )


def test_lifecycle_includes_publication_commitment_evaluation_final_and_gate():
    plan = build_plan(world_size=8, local_batch_size=512)
    assert plan["lifecycle"] == [
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
    ]
    execution = plan["execution_protocol"]
    assert execution["published_executor_authority_required"] == [
        "P1",
        "WARMUP",
        "GEOMETRY",
        "AUX0",
        "AUXT",
        "FINAL",
    ]
    assert execution["post_authority_execution_commitment_required"] == [
        "P1",
        "WARMUP",
        "GEOMETRY",
        "AUX0",
        "AUXT",
        "FINAL",
    ]
    assert execution["all_eight_physical_gpu_locks_required"] is True
    assert execution["one_dose_exact_execution_binding_required"] == [
        "P1",
        "AUX0",
        "AUXT",
        "FINAL",
    ]
    assert all(execution["live_allocation_replay_required"].values())


def test_fixed_evaluation_and_independent_final_are_not_diagnostic_chaining():
    plan = build_plan(world_size=8, local_batch_size=512)
    evaluation = plan["fixed_pair_evaluation"]
    assert evaluation["internal"] == {
        "pairs_per_arm": 300,
        "map_kind": "BASE",
        "opponent": "recovered_generator_reference",
        "decision": "AUXT pair points strictly greater than AUX0",
    }
    assert evaluation["external"] == {
        "pairs_per_arm": 250,
        "games_per_arm": 500,
        "map_kind": "TOURNAMENT",
        "opponent": "catanatron_value",
        "non_regression_tolerance_milli": 25,
    }
    assert evaluation["search"]["n_full"] == 128
    assert evaluation["search"]["d6_root_averaging"] is True
    final = plan["independent_final_replication"]
    assert final["diagnostic_arm_checkpoint_as_initializer_forbidden"] is True
    assert final["base_parent_lineage_reloaded"] is True
    assert final["fresh_sampler_seed_order_and_row_set_required"] is True
    assert final["sample_dose"] == JOINT_SAMPLE_DOSE
    assert final["optimizer_steps"] == JOINT_OPTIMIZER_STEPS
    gate = plan["full_gate"]
    assert gate["dual_baseline_conjunctive"] is True
    assert gate["strict_h1_over_recovered_parent"] is True
    assert gate["f7_non_regression_veto"] is True
    assert gate["f7_veto_complete_pairs"] == 300
    assert gate["auto_promotion"] is False


@pytest.mark.parametrize(
    "world_size,local_batch_size,grad_accum_steps",
    [(0, 512, 1), (8, 1024, 1), (8, 512, 2)],
)
def test_any_topology_drift_fails_closed(
    world_size: int, local_batch_size: int, grad_accum_steps: int
):
    with pytest.raises(ValueError, match="exact FP32 8x512 accum1"):
        build_plan(
            world_size=world_size,
            local_batch_size=local_batch_size,
            grad_accum_steps=grad_accum_steps,
        )
