from __future__ import annotations

import math

import pytest

from tools.a1_learner_recovery_plan import (
    ESCALATED_SAMPLES,
    SENTINEL_SAMPLES,
    build_plan,
)


def test_plan_matches_update_dose_across_measured_batch_topologies():
    for local_batch in (512, 768, 1024, 1536):
        plan = build_plan(world_size=8, local_batch_size=local_batch, grad_accum_steps=1)
        global_batch = 8 * local_batch
        assert plan["topology"]["global_batch_size"] == global_batch
        assert plan["sample_doses"]["sentinel_steps"] == math.ceil(
            SENTINEL_SAMPLES / global_batch
        )
        assert plan["sample_doses"]["escalated_steps"] == math.ceil(
            ESCALATED_SAMPLES / global_batch
        )
        assert all(
            arm["sample_dose"] == SENTINEL_SAMPLES
            and arm["max_steps"] == math.ceil(SENTINEL_SAMPLES / global_batch)
            for arm in plan["arms"]
        )


def test_anchor_sweep_is_single_variable_and_q_stays_disabled():
    plan = build_plan(world_size=8, local_batch_size=1024, grad_accum_steps=1)
    anchors = [arm for arm in plan["arms"] if arm["stage"] == "P1_anchor"]
    assert [arm["arm_id"] for arm in anchors] == ["K0", "K3", "K10"]
    assert [arm["recipe"]["policy_kl_anchor_weight"] for arm in anchors] == [
        0.0,
        0.03,
        0.10,
    ]
    stripped = []
    for arm in anchors:
        recipe = dict(arm["recipe"])
        recipe.pop("policy_kl_anchor_weight")
        stripped.append(recipe)
        assert arm["recipe"]["q_loss_weight"] == 0.0
        assert arm["recipe"]["validation_max_samples"] == 0
        assert arm["recipe"]["validation_game_sentinel_target_rows"] == 262_144
    assert stripped[0] == stripped[1] == stripped[2]


def test_plan_rejects_invalid_topology():
    with pytest.raises(ValueError, match="positive"):
        build_plan(world_size=0, local_batch_size=1024, grad_accum_steps=1)


def test_accumulation_is_reflected_in_topology_and_every_recipe():
    plan = build_plan(world_size=8, local_batch_size=512, grad_accum_steps=2)
    assert plan["topology"]["global_batch_size"] == 8192
    assert all(arm["recipe"]["grad_accum_steps"] == 2 for arm in plan["arms"])


def test_forced_policy_is_not_reintroduced_as_a_recipe_override():
    plan = build_plan(world_size=8, local_batch_size=1024, grad_accum_steps=1)
    assert all("forced_action_weight" not in arm["recipe"] for arm in plan["arms"])
    assert all(
        arm["recipe"]["policy_kl_anchor_direction"] == "forward"
        for arm in plan["arms"]
    )
    assert plan["fixed_data_recipe"]["incumbent_mixed_replay_ratio_by_game"] == 0.2
    assert next(arm for arm in plan["arms"] if arm["arm_id"] == "HEAD_ONLY")[
        "recipe"
    ]["freeze_modules"] == "trunk"
    assert all(arm["arm_id"] != "FULL_LR" for arm in plan["arms"])
    assert any("forward KL" in item for item in plan["prerequisites"])
    assert any("single-legal-action" in item for item in plan["prerequisites"])
