from __future__ import annotations

import math

import pytest

from tools.a1_post_p1_diagnosis_plan import SAMPLE_DOSE, build_plan


def test_matrix_reuses_legacy_evidence_and_adds_three_causal_runs() -> None:
    plan = build_plan()
    assert [arm["arm_id"] for arm in plan["arms"]] == [
        "LEGACY_L03",
        "L1_CONTROL",
        "L1_POLICY_AUX",
        "L1_GATHER",
    ]
    assert sum(arm["training"] == "new matched B200 run" for arm in plan["arms"]) == 3
    assert plan["fixed_recipe"]["sample_dose"] == SAMPLE_DOSE
    assert plan["fixed_recipe"]["max_steps"] == math.ceil(SAMPLE_DOSE / 4096)


def test_policy_dose_and_gather_arms_are_single_sequential_deltas() -> None:
    plan = build_plan()
    fixed = plan["fixed_recipe"]
    control, aux, gather = plan["arms"][1:]
    assert control["recipe_delta"]["policy_aux_active_batch_size"] == 0
    assert aux["recipe_delta"]["policy_aux_active_batch_size"] == 128
    assert gather["recipe_delta"]["policy_aux_active_batch_size"] == 128
    assert control["recipe_delta"]["checkpoint_upgrade"] == "none"
    assert aux["recipe_delta"]["checkpoint_upgrade"] == "none"
    assert "gather" in gather["recipe_delta"]["checkpoint_upgrade"]
    assert fixed["aux_subgoal_heads"] is False
    assert fixed["value_target_lambda"] == 1.0
    assert fixed["value_lr_mult"] == 0.3
    assert fixed["value_loss_weight"] == 0.25
    assert fixed["policy_loss_weight"] == 1.0
    assert fixed["soft_target_weight"] == 0.9
    assert fixed["final_vp_loss_weight"] == 0.0
    assert fixed["loser_sample_weight"] == 1.0
    assert fixed["event_history_available"] is False
    assert fixed["per_game_policy_weight"] is False
    assert fixed["per_game_value_weight"] is False
    assert "checkpoint chaining forbidden" in fixed["initialization_policy"]
    assert fixed["lineage_dose_schema"] == "a1-lineage-dose-v1"
    assert "implemented" in plan["historical_feature_audit"]["trunk_lr_multiplier"]
    assert "f7 default OFF" in plan["historical_feature_audit"]["action_target_gather"]


def test_matrix_is_sequential_and_non_launching() -> None:
    plan = build_plan()
    assert plan["launch_authorized"] is False
    assert "P1" in plan["launch_condition"]
    assert any("only after L1_CONTROL releases DDP" in row for row in plan["gpu_schedule"])
    assert "operator identity is valid" in plan["explicitly_deferred"]["root_value_blend"]


def test_evaluation_types_randomized_primary_and_tournament_bridge() -> None:
    plan = build_plan()
    assert plan["evaluation"]["internal"]["map_kind"] == "BASE"
    assert plan["evaluation"]["external"]["map_kind"] == "TOURNAMENT"
    assert plan["evaluation"]["direct_tournament_bridge"]["map_kind"] == "TOURNAMENT"
    assert plan["value_readout_probe"]["calibration"] == ["raw", "tanh", "clip"]
    assert plan["value_readout_probe"]["default_change_authorized"] is False


def test_only_eight_b200_topology_is_admitted() -> None:
    with pytest.raises(ValueError, match="eight B200"):
        build_plan(world_size=4)


def test_plan_uses_objective_conflict_not_clip_frequency_as_stop_signal() -> None:
    plan = build_plan()
    probe = plan["prelaunch_gradient_probe"]
    assert probe["initialization"] == "reload f7 independently"
    assert "single GPU" in probe["execution"]
    assert "value_to_policy_grad_norm_ratio" in probe["primary_readouts"]
    assert any(
        "do not abort on clipped fraction alone" in row
        for row in plan["early_stop"]
    )
