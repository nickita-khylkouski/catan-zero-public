"""Exact learner geometry for function-preserving topology commissioning."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tools.train_bc import (
    ENTITY_GRAPH_FREEZABLE_MODULE_GROUPS,
    _build_optimizer_param_groups,
    _require_only_trainable_prefixes,
    _set_entity_graph_modules_trainable,
    _set_xdim_q_branch_trainable,
)


def _production_width_topology_gather_policy():
    from catan_zero.rl.entity_token_policy import EntityGraphPolicy
    from catan_zero.rl.self_play import make_env_config

    base = EntityGraphPolicy.create(
        env_config=make_env_config(vps_to_win=3),
        hidden_size=640,
        state_layers=1,
        attention_heads=8,
        seed=0,
        device="cpu",
    )
    return EntityGraphPolicy(
        replace(
            base.config,
            action_target_gather=True,
            topology_residual_adapter=True,
        ),
        base.static_action_features.detach().cpu().numpy(),
        device="cpu",
    )


def _freeze_everything_except_topology(model) -> None:
    # q_loss_weight=0 applies this before user-selected freeze groups in the
    # real learner.  Keep the unit contract faithful to that production path.
    _set_xdim_q_branch_trainable(model, False)
    _set_entity_graph_modules_trainable(
        model,
        (
            "trunk_base",
            "action_encoder",
            "policy_head",
            "value_heads",
            "target_gather",
        ),
        trainable=False,
    )


def test_trunk_group_remains_exact_union_of_base_and_topology() -> None:
    trunk = ENTITY_GRAPH_FREEZABLE_MODULE_GROUPS["trunk"]
    base = ENTITY_GRAPH_FREEZABLE_MODULE_GROUPS["trunk_base"]
    topology = ENTITY_GRAPH_FREEZABLE_MODULE_GROUPS["topology_adapter"]

    assert trunk == base + topology
    assert topology == ("topology_residual_adapter",)
    assert not set(base).intersection(topology)


def test_topology_adapter_group_freezes_without_touching_base_trunk() -> None:
    policy = _production_width_topology_gather_policy()

    touched = _set_entity_graph_modules_trainable(
        policy.model, ("topology_adapter",), trainable=False
    )

    assert touched == ["topology_residual_adapter"]
    assert all(
        not parameter.requires_grad
        for parameter in policy.model.topology_residual_adapter.parameters()
    )
    assert all(
        parameter.requires_grad for parameter in policy.model.blocks.parameters()
    )


def test_topology_only_surface_is_exactly_eight_tensors_and_823040_parameters() -> None:
    policy = _production_width_topology_gather_policy()
    _freeze_everything_except_topology(policy.model)

    trainable = [
        (name, parameter)
        for name, parameter in policy.model.named_parameters()
        if parameter.requires_grad
    ]
    assert len(trainable) == 8
    assert sum(parameter.numel() for _, parameter in trainable) == 823_040
    assert all(name.startswith("topology_residual_adapter.") for name, _ in trainable)
    assert _require_only_trainable_prefixes(
        policy.model, ("topology_residual_adapter",)
    ) == {
        "prefixes": ["topology_residual_adapter"],
        "parameter_tensors": 8,
        "parameters": 823_040,
        "parameters_by_prefix": {"topology_residual_adapter": 823_040},
    }


def test_topology_only_surface_receives_trunk_lr_multiplier() -> None:
    policy = _production_width_topology_gather_policy()
    _freeze_everything_except_topology(policy.model)

    groups = _build_optimizer_param_groups(
        policy.model,
        base_lr=3e-5,
        value_lr_mult=1.0,
        action_module_lr_mult=1.0,
        trunk_lr_mult=4.0,
        architecture="entity_graph",
    )
    by_name = {group["_group_name"]: group for group in groups}

    assert set(by_name) == {"base", "trunk"}
    assert by_name["base"]["params"] == []
    assert by_name["trunk"]["lr"] == pytest.approx(1.2e-4)
    assert len(by_name["trunk"]["params"]) == 8
    assert sum(p.numel() for p in by_name["trunk"]["params"]) == 823_040
