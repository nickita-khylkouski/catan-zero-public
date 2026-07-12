from __future__ import annotations

import math

import pytest

from tools.a1_aux_regularization_plan import AUX_FIELDS, SAMPLE_DOSE, build_plan


def test_probe_has_exactly_two_equal_sample_arms():
    plan = build_plan(world_size=8, local_batch_size=1024)
    assert [arm["arm_id"] for arm in plan["arms"]] == ["AUX0", "AUX2"]
    assert plan["max_steps"] == math.ceil(SAMPLE_DOSE / (8 * 1024))
    assert all(
        arm["fixed_recipe"]["sample_dose"] == SAMPLE_DOSE
        and arm["fixed_recipe"]["max_steps"] == plan["max_steps"]
        and arm["fixed_recipe"]["validation_max_samples"] == 262_144
        for arm in plan["arms"]
    )


def test_only_auxiliary_recipe_delta_changes():
    plan = build_plan(world_size=8, local_batch_size=768)
    control, treatment = plan["arms"]
    assert control["fixed_recipe"] == treatment["fixed_recipe"]
    assert control["recipe_delta"] == {
        "aux_subgoal_heads": False,
        "aux_subgoal_loss_weight": 0.0,
    }
    assert treatment["recipe_delta"]["aux_subgoal_heads"] is True
    assert treatment["recipe_delta"]["aux_subgoal_loss_weight"] == pytest.approx(0.02)
    assert plan["required_corpus_fields"] == list(AUX_FIELDS)


def test_probe_cannot_authorize_launch_or_promotion():
    plan = build_plan(world_size=8, local_batch_size=1024)
    assert plan["diagnostic_only"] is True
    assert plan["launch_authorized"] is False
    assert plan["promotion_eligible"] is False
    assert "P1 winner" in plan["prerequisites"][0]


def test_invalid_topology_fails_closed():
    with pytest.raises(ValueError, match="positive"):
        build_plan(world_size=0, local_batch_size=1024)
