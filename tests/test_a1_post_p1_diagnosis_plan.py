from __future__ import annotations

import math

import pytest

from tools.a1_post_p1_diagnosis_plan import (
    FULL_SAMPLE_DOSE,
    SHORT_SAMPLE_DOSE,
    build_plan,
)


def test_matrix_reuses_legacy_evidence_and_adds_conditional_anchor() -> None:
    plan = build_plan()
    assert [arm["arm_id"] for arm in plan["arms"]] == [
        "LEGACY_L03",
        "TEMP_CONTROL",
        "CURRENT_POLICY_SCOPE",
        "CURRENT_VALUE_SCOPE",
        "L1_PURE_SEARCH_TARGET",
        "L1_POLICY_AUX",
        "L1_REPLAY_ANCHOR",
        "L1_GATHER",
    ]
    assert (
        sum(arm["training"].startswith("new matched B200 run") for arm in plan["arms"])
        == 5
    )
    policy_aux = next(arm for arm in plan["arms"] if arm["arm_id"] == "L1_POLICY_AUX")
    assert "no new B200 run" in policy_aux["training"]
    assert policy_aux["completed_evidence"] == {
        "candidate_wins": 596,
        "control_wins": 604,
        "games": 1200,
        "candidate_score": 0.496667,
        "errors": 0,
        "truncations": 0,
        "ruling": "no demonstrated improvement; do not repeat",
        "audit": "docs/audits/A1_POLICY_AUX_REPLICATION_20260712.md",
    }
    fixed = plan["fixed_recipe"]
    assert fixed["sample_dose"] == SHORT_SAMPLE_DOSE
    assert fixed["max_steps"] == 128
    assert fixed["dose_selection"] == "matched_behavior_pareto_rule"
    assert fixed["dose_candidates"] == [SHORT_SAMPLE_DOSE, FULL_SAMPLE_DOSE]
    assert fixed["max_steps_by_dose"] == {
        str(SHORT_SAMPLE_DOSE): math.ceil(SHORT_SAMPLE_DOSE / 4096),
        str(FULL_SAMPLE_DOSE): math.ceil(FULL_SAMPLE_DOSE / 4096),
    }


def test_policy_dose_and_gather_arms_are_single_sequential_deltas() -> None:
    plan = build_plan()
    fixed = plan["fixed_recipe"]
    control, policy_scope, value_scope, pure_target, aux, anchor, gather = plan[
        "arms"
    ][1:]
    assert control["recipe_delta"]["policy_aux_active_batch_size"] == 0
    assert policy_scope["recipe_delta"]["policy_distillation_component_ids"] == [
        "n128_current",
        "n256_current",
    ]
    assert "value_training_component_ids" not in policy_scope["recipe_delta"]
    assert value_scope["recipe_delta"]["value_training_component_ids"] == [
        "n128_current",
        "n256_current",
    ]
    assert "policy_distillation_component_ids" not in value_scope["recipe_delta"]
    assert pure_target["recipe_delta"]["soft_target_weight"] == 1.0
    assert aux["recipe_delta"]["policy_aux_active_batch_size"] == 128
    assert anchor["recipe_delta"]["policy_aux_active_batch_size"] == 0
    assert "exact eligible-mass" in anchor["recipe_delta"]["policy_kl_anchor_weight"]
    assert "stored_prior" in anchor["recipe_delta"]["policy_kl_anchor_direction"]
    assert gather["recipe_delta"]["policy_aux_active_batch_size"] == 0
    assert control["recipe_delta"]["checkpoint_upgrade"] == "none"
    assert aux["recipe_delta"]["checkpoint_upgrade"] == "none"
    assert "gather" in gather["recipe_delta"]["checkpoint_upgrade"]
    assert fixed["aux_subgoal_heads"] is False
    assert fixed["value_target_lambda"] == 1.0
    assert fixed["value_lr_mult"] == 0.3
    assert fixed["value_loss_weight"] == 0.25
    assert fixed["policy_loss_weight"] == 1.0
    assert fixed["soft_target_weight"] == 0.9
    assert fixed["amp"] == "none"
    assert fixed["stored_policy_component_temperatures"] == {
        "n128_current": 1.0,
        "n256_current": 1.11,
        "gen3_replay": 0.52,
    }
    assert fixed["policy_distillation_component_ids"] == [
        "n128_current",
        "n256_current",
        "gen3_replay",
    ]
    assert fixed["value_training_component_ids"] == [
        "n128_current",
        "n256_current",
        "gen3_replay",
    ]
    assert "exact TEMP control" in fixed["gen3_replay_policy_objective"]
    assert fixed["final_vp_loss_weight"] == 0.0
    assert fixed["forced_action_weight"] == 0.0
    assert fixed["truncated_vp_margin_value_weight"] == 0.0
    assert fixed["loser_sample_weight"] == 1.0
    assert fixed["event_history_available"] is False
    assert fixed["per_game_policy_weight"] is False
    assert fixed["per_game_value_weight"] is False
    assert "checkpoint chaining forbidden" in fixed["initialization_policy"]
    assert fixed["lineage_dose_schema"] == "a1-lineage-dose-v1"
    assert "implemented" in plan["historical_feature_audit"]["trunk_lr_multiplier"]
    assert "f7 default OFF" in plan["historical_feature_audit"]["action_target_gather"]
    assert "contaminated" in plan["historical_feature_audit"]["d6_symmetry"]


def test_matrix_is_sequential_and_non_launching() -> None:
    plan = build_plan()
    assert plan["launch_authorized"] is False
    assert "dose adjudication" in plan["launch_condition"]
    assert any(
        "CURRENT_POLICY_SCOPE and CURRENT_VALUE_SCOPE" in row
        for row in plan["gpu_schedule"]
    )
    assert any(
        "do not launch it again" in row for row in plan["gpu_schedule"]
    )
    assert any(
        "L1_GATHER" in row and "policy auxiliary dose zero" in row
        for row in plan["gpu_schedule"]
    )
    assert (
        "operator identity is valid" in plan["explicitly_deferred"]["root_value_blend"]
    )
    scope = plan["replay_scope_adjudication"]
    assert scope["baseline"] == "TEMP_CONTROL"
    assert "do not create a both-current scope" in scope["interaction_rule"]


def test_matched_behavior_selects_short_pareto_dose() -> None:
    plan = build_plan()
    adjudication = plan["dose_adjudication"]
    assert adjudication["status"] == "selected"
    assert adjudication["launch_blocking"] is False
    assert adjudication["selected_sample_dose"] == SHORT_SAMPLE_DOSE
    assert adjudication["selected_optimizer_steps"] == 128
    assert set(adjudication["candidates"]) == {
        str(SHORT_SAMPLE_DOSE),
        str(FULL_SAMPLE_DOSE),
    }
    assert (
        adjudication["candidates"][str(SHORT_SAMPLE_DOSE)]["artifact"]["sha256"]
        == "sha256:91a4b63ee5b74da0ec7123557d36e084d7ab91963c65bab9848838123e8e86de"
    )
    assert (
        adjudication["candidates"][str(FULL_SAMPLE_DOSE)]["artifact"]["sha256"]
        == "sha256:ce29663fe519b88537d54afec3dfa4e0033f79a649f8b04d364baead48c462f4"
    )
    assert "smallest dose" in adjudication["selection_rule"]
    assert (
        adjudication["candidates"][str(SHORT_SAMPLE_DOSE)][
            "integrated_lr_step_equivalents"
        ]
        == 78.5
    )
    assert (
        adjudication["candidates"][str(FULL_SAMPLE_DOSE)][
            "integrated_lr_step_equivalents"
        ]
        == 974.5
    )
    assert "12.41x" in adjudication["lr_exposure_interpretation"]
    assert "old gen3" in adjudication["evaluation"]["parent_panel"]
    assert adjudication["evaluation"]["value_squash"] == "tanh"
    assert "value_squash=clip" in adjudication["evaluation"]["operator_sensitivity"]
    behavior = adjudication["observed_behavior"]
    assert behavior["common_seed_orientation_keys"] == 128
    assert behavior[str(SHORT_SAMPLE_DOSE)]["candidate_wins"] == 75
    assert behavior[str(FULL_SAMPLE_DOSE)]["candidate_wins"] == 65
    assert behavior["paired_outcomes"] == {
        "both_win": 41,
        "short_only_win": 34,
        "full_only_win": 24,
        "both_lose": 29,
    }
    assert "7.8125 points lower" in adjudication["selection_rationale"]


def test_evaluation_types_randomized_primary_and_tournament_bridge() -> None:
    plan = build_plan()
    assert plan["schema_version"] == "a1-post-p1-optimization-architecture-plan-v5"
    assert plan["evaluation"]["matched_search_operator"] == {
        "candidate_c_scale": 0.10,
        "baseline_c_scale": 0.10,
        "selection_tuning_allowed": False,
        "reason": (
            "changing checkpoint ancestry and c_scale in one comparison repeats "
            "the historical gen3/.03 adjudication confound"
        ),
    }
    assert "same deployed c_scale=0.10" in plan["evaluation"]["checkpoint_identity"]
    assert "separate crossover" in plan["evaluation"]["checkpoint_identity"]
    assert plan["evaluation"]["internal"]["map_kind"] == "BASE"
    assert plan["evaluation"]["external"]["map_kind"] == "TOURNAMENT"
    assert plan["evaluation"]["direct_tournament_bridge"]["map_kind"] == "TOURNAMENT"
    assert "disjoint" in plan["evaluation"]["promotion_confirmation"]
    assert plan["value_readout_probe"]["calibration"] == ["raw", "tanh", "clip"]
    assert plan["value_readout_probe"]["default_change_authorized"] is False


def test_only_eight_b200_topology_is_admitted() -> None:
    with pytest.raises(ValueError, match="eight B200"):
        build_plan(world_size=4)
    with pytest.raises(ValueError, match="local batch 512"):
        build_plan(local_batch_size=1024)
    with pytest.raises(ValueError, match="accumulation 1"):
        build_plan(grad_accum_steps=2)


def test_plan_uses_objective_conflict_not_clip_frequency_as_stop_signal() -> None:
    plan = build_plan()
    probe = plan["prelaunch_gradient_probe"]
    assert probe["initialization"] == "reload f7 independently"
    assert "single GPU" in probe["execution"]
    assert "value_to_policy_grad_norm_ratio" in probe["primary_readouts"]
    assert any(
        "do not abort on clipped fraction alone" in row for row in plan["early_stop"]
    )
