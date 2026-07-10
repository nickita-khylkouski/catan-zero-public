from __future__ import annotations

import pytest
from dataclasses import replace

from tools.train_bc import (
    ACTION_LOCAL_MODULE_ATTRS,
    VALUE_HEAD_MODULE_ATTRS,
    _apply_lr_schedule,
    _build_optimizer_param_groups,
    _make_optimizer,
    _set_scalar_value_head_trainable,
)


def _make_entity_policy(
    hidden_size: int = 16,
    *,
    categorical_bins: int = 0,
    action_local: bool = False,
):
    from catan_zero.rl.entity_token_policy import EntityGraphPolicy
    from catan_zero.rl.self_play import make_env_config

    policy = EntityGraphPolicy.create(
        env_config=make_env_config(vps_to_win=3),
        hidden_size=hidden_size,
        state_layers=1,
        attention_heads=2,
        seed=0,
    )
    if categorical_bins or action_local:
        config = replace(
            policy.config,
            value_categorical_bins=int(categorical_bins),
            action_target_gather=bool(action_local),
            action_cross_attention_layers=2 if action_local else 0,
        )
        policy = EntityGraphPolicy(
            config,
            policy.static_action_features.detach().cpu().numpy(),
            device="cpu",
        )
    return policy


class _Args:
    def __init__(self, **overrides):
        values = {
            "optimizer": "adam",
            "weight_decay": 0.0,
            "fused_optimizer": False,
            "lr": 2e-4,
        }
        values.update(overrides)
        for key, value in values.items():
            setattr(self, key, value)


# --------------------------------------------------------------------------- _build_optimizer_param_groups


def test_value_head_module_attrs_covers_all_value_adjacent_heads() -> None:
    assert set(VALUE_HEAD_MODULE_ATTRS) == {
        "value_head",
        "value_categorical_head",
        "final_vp_head",
        "value_uncertainty_head",
    }


def test_mult_of_one_returns_a_flat_param_list_unchanged() -> None:
    policy = _make_entity_policy()
    groups = _build_optimizer_param_groups(
        policy.model, base_lr=2e-4, value_lr_mult=1.0
    )

    assert isinstance(groups, list)
    assert groups and not isinstance(groups[0], dict)
    expected = [p for p in policy.model.parameters() if p.requires_grad]
    assert len(groups) == len(expected)


def test_mult_other_than_one_splits_value_head_params_into_their_own_group() -> None:
    policy = _make_entity_policy(categorical_bins=9)
    groups = _build_optimizer_param_groups(
        policy.model, base_lr=2e-4, value_lr_mult=0.3
    )

    assert len(groups) == 2
    assert all(isinstance(group, dict) for group in groups)

    base_group, value_group = groups
    assert base_group["lr"] == pytest.approx(2e-4)
    assert base_group["base_lr"] == pytest.approx(2e-4)
    assert value_group["lr"] == pytest.approx(2e-4 * 0.3)
    assert value_group["base_lr"] == pytest.approx(2e-4 * 0.3)

    value_param_ids = {id(p) for p in value_group["params"]}
    # Re-derive directly from the model's own submodules (avoids relying on the
    # function's internals): every value_head/final_vp_head param must land in the
    # value group, and none of them in the base group.
    direct_value_ids = set()
    for name in VALUE_HEAD_MODULE_ATTRS:
        submodule = getattr(policy.model, name, None)
        if submodule is None:
            continue
        direct_value_ids |= {id(p) for p in submodule.parameters() if p.requires_grad}
    assert value_param_ids == direct_value_ids

    base_param_ids = {id(p) for p in base_group["params"]}
    assert base_param_ids.isdisjoint(direct_value_ids)

    all_trainable = {id(p) for p in policy.model.parameters() if p.requires_grad}
    assert base_param_ids | value_param_ids == all_trainable


def test_categorical_primary_can_freeze_scalar_diagnostic_without_freezing_cat_head() -> (
    None
):
    policy = _make_entity_policy(categorical_bins=9)

    _set_scalar_value_head_trainable(policy.model, False)

    assert all(not p.requires_grad for p in policy.model.value_head.parameters())
    assert all(
        p.requires_grad for p in policy.model.value_categorical_head.parameters()
    )


def test_action_local_modules_get_an_independent_lr_group() -> None:
    policy = _make_entity_policy(categorical_bins=9, action_local=True)
    groups = _build_optimizer_param_groups(
        policy.model,
        base_lr=2e-4,
        value_lr_mult=0.3,
        action_module_lr_mult=0.2,
    )

    assert [group["lr"] for group in groups] == pytest.approx(
        [2e-4, 2e-4 * 0.3, 2e-4 * 0.2]
    )
    action_ids = {id(p) for p in groups[2]["params"]}
    expected_action_ids = {
        id(p)
        for attr in ACTION_LOCAL_MODULE_ATTRS
        for p in getattr(policy.model, attr).parameters()
        if p.requires_grad
    }
    assert action_ids == expected_action_ids
    assert action_ids.isdisjoint({id(p) for p in groups[0]["params"]})
    assert action_ids.isdisjoint({id(p) for p in groups[1]["params"]})


def test_action_module_lr_multiplier_fails_without_action_local_modules() -> None:
    policy = _make_entity_policy()
    with pytest.raises(SystemExit, match="action-module-lr-mult"):
        _build_optimizer_param_groups(
            policy.model,
            base_lr=2e-4,
            value_lr_mult=1.0,
            action_module_lr_mult=0.3,
        )


def test_mult_of_one_param_list_is_identical_order_and_identity_to_pre_cat12_construction() -> (
    None
):
    """Bit-for-bit parity check, not just a length check: with the default
    --value-lr-mult 1.0, _build_optimizer_param_groups must return the exact same
    parameter tensors, in the exact same order, as the pre-CAT-12 construction
    (``[p for p in policy.model.parameters() if p.requires_grad]``). Order/identity
    matter here, not just count -- a reordering or a copy would still pass a
    length-only check but would desync optimizer.state (momentum/Adam moments are
    keyed by parameter identity) or silently change which group a value-head
    parameter lands in relative to the pre-patch flat list."""
    policy = _make_entity_policy()
    groups = _build_optimizer_param_groups(
        policy.model, base_lr=2e-4, value_lr_mult=1.0
    )
    pre_cat12 = [p for p in policy.model.parameters() if p.requires_grad]

    assert len(groups) == len(pre_cat12)
    assert all(a is b for a, b in zip(groups, pre_cat12))


def test_mult_other_than_one_raises_when_model_has_no_value_head_submodule() -> None:
    import torch

    plain_model = torch.nn.Linear(4, 4)
    with pytest.raises(SystemExit, match="value-lr-mult"):
        _build_optimizer_param_groups(plain_model, base_lr=2e-4, value_lr_mult=0.3)


# --------------------------------------------------------------------------- _make_optimizer integration


def test_make_optimizer_builds_two_lr_groups_when_value_lr_mult_set() -> None:
    policy = _make_entity_policy()
    groups = _build_optimizer_param_groups(
        policy.model, base_lr=2e-4, value_lr_mult=0.3
    )
    args = _Args(lr=2e-4)

    optimizer = _make_optimizer(groups, args, "cpu")

    assert len(optimizer.param_groups) == 2
    lrs = sorted(group["lr"] for group in optimizer.param_groups)
    assert lrs[0] == pytest.approx(2e-4 * 0.3)
    assert lrs[1] == pytest.approx(2e-4)


def test_make_optimizer_single_group_when_mult_is_one() -> None:
    policy = _make_entity_policy()
    groups = _build_optimizer_param_groups(
        policy.model, base_lr=2e-4, value_lr_mult=1.0
    )
    args = _Args(lr=2e-4)

    optimizer = _make_optimizer(groups, args, "cpu")

    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2e-4)


# --------------------------------------------------------------------------- schedule respects per-group base_lr


def test_apply_lr_schedule_scales_each_group_by_its_own_base_lr() -> None:
    policy = _make_entity_policy()
    groups = _build_optimizer_param_groups(
        policy.model, base_lr=2e-4, value_lr_mult=0.3
    )
    args = _Args(lr=2e-4)
    optimizer = _make_optimizer(groups, args, "cpu")

    multiplier = _apply_lr_schedule(
        optimizer,
        base_lr=2e-4,
        step=3,
        warmup_steps=4,
        total_steps=100,
        schedule="flat",
    )

    assert multiplier == pytest.approx(1.0)
    lrs = sorted(group["lr"] for group in optimizer.param_groups)
    assert lrs[0] == pytest.approx(2e-4 * 0.3 * multiplier)
    assert lrs[1] == pytest.approx(2e-4 * multiplier)


def test_default_flag_lr_trajectory_is_bit_identical_across_warmup_and_decay_to_pre_cat12_optimizer() -> (
    None
):
    """The single most important CAT-12 regression given the prior "one LR stomped
    all param groups" bug history: with the default --value-lr-mult 1.0, build TWO
    optimizers on the SAME real (value_head/final_vp_head-bearing) model -- one via
    the exact pre-CAT-12 construction (a flat parameter list straight into
    _make_optimizer, no param-group dicts) and one via the new default-flag path
    (_build_optimizer_param_groups(..., value_lr_mult=1.0) into _make_optimizer) --
    then walk BOTH through every step of a warmup-then-cosine-decay schedule via
    _apply_lr_schedule and assert the applied per-group LR is bit-identical at every
    single step. This proves the new code path is a true no-op end-to-end (not just
    at a single sampled step) for both the warmup ramp-up and the post-warmup decay
    phases."""
    policy = _make_entity_policy()
    args = _Args(lr=2e-4)

    pre_cat12_params = [p for p in policy.model.parameters() if p.requires_grad]
    old_optimizer = _make_optimizer(pre_cat12_params, args, "cpu")

    groups = _build_optimizer_param_groups(
        policy.model, base_lr=2e-4, value_lr_mult=1.0
    )
    new_optimizer = _make_optimizer(groups, args, "cpu")

    assert len(old_optimizer.param_groups) == 1
    assert len(new_optimizer.param_groups) == 1

    warmup_steps, total_steps = 5, 40
    for step in range(0, total_steps + 5):
        old_multiplier = _apply_lr_schedule(
            old_optimizer,
            base_lr=2e-4,
            step=step,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            schedule="cosine",
        )
        new_multiplier = _apply_lr_schedule(
            new_optimizer,
            base_lr=2e-4,
            step=step,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            schedule="cosine",
        )
        assert new_multiplier == pytest.approx(old_multiplier)
        old_lrs = [group["lr"] for group in old_optimizer.param_groups]
        new_lrs = [group["lr"] for group in new_optimizer.param_groups]
        assert new_lrs == pytest.approx(old_lrs)


def test_apply_lr_schedule_still_uses_the_passed_base_lr_for_a_plain_single_group_optimizer() -> (
    None
):
    """A single-implicit-group optimizer (no "base_lr" key, e.g. every call site before
    --value-lr-mult existed, and the existing test_train_bc_lr_schedule.py suite) must
    keep using the schedule's passed ``base_lr`` -- this is the backward-compat
    fallback in `_apply_lr_schedule`/`_apply_lr_warmup`."""
    import torch

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0)

    _apply_lr_schedule(
        optimizer,
        base_lr=2e-4,
        step=0,
        warmup_steps=4,
        total_steps=100,
        schedule="flat",
    )

    for group in optimizer.param_groups:
        assert group["lr"] == pytest.approx(2e-4 * 0.25)
