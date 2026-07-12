from __future__ import annotations

import math

import pytest

from tools.a1_post_p1_diagnosis_plan import SAMPLE_DOSE, build_plan


def test_matrix_reuses_control_and_adds_exactly_two_matched_runs() -> None:
    plan = build_plan()
    assert [arm["arm_id"] for arm in plan["arms"]] == [
        "FULL_CONTROL",
        "HEAD_ONLY",
        "HEAD_GX1",
    ]
    assert sum(arm["training"] == "new matched B200 run" for arm in plan["arms"]) == 2
    assert plan["fixed_recipe"]["sample_dose"] == SAMPLE_DOSE
    assert plan["fixed_recipe"]["max_steps"] == math.ceil(SAMPLE_DOSE / 4096)


def test_architecture_arm_is_single_delta_over_head_only() -> None:
    plan = build_plan()
    fixed = plan["fixed_recipe"]
    head, gather = plan["arms"][1:]
    assert head["recipe_delta"]["freeze_modules"] == "trunk"
    assert gather["recipe_delta"]["freeze_modules"] == "trunk"
    assert head["recipe_delta"]["action_module_lr_mult"] == 1.0
    assert gather["recipe_delta"]["action_module_lr_mult"] == 1.0
    assert head["recipe_delta"]["trunk_lr_mult"] == 1.0
    assert gather["recipe_delta"]["trunk_lr_mult"] == 1.0
    assert head["recipe_delta"]["checkpoint_upgrade"] == "none"
    assert "gather,cross:1" in gather["recipe_delta"]["checkpoint_upgrade"]
    assert fixed["aux_subgoal_heads"] is False
    assert fixed["value_target_lambda"] == 1.0
    assert fixed["value_lr_mult"] == 0.3
    assert fixed["loser_sample_weight"] == 0.3
    assert fixed["per_game_policy_weight"] is False
    assert fixed["per_game_value_weight"] is False
    assert "implemented" in plan["historical_feature_audit"]["trunk_lr_multiplier"]
    assert "f7 default OFF" in plan["historical_feature_audit"]["action_target_gather"]


def test_matrix_is_sequential_and_non_launching() -> None:
    plan = build_plan()
    assert plan["launch_authorized"] is False
    assert "P1" in plan["launch_condition"]
    assert any("only after HEAD_ONLY releases DDP" in row for row in plan["gpu_schedule"])
    assert "operator identity is valid" in plan["explicitly_deferred"]["root_value_blend"]


def test_only_eight_b200_topology_is_admitted() -> None:
    with pytest.raises(ValueError, match="eight B200"):
        build_plan(world_size=4)
