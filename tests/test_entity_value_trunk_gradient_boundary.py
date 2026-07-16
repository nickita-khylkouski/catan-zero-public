"""Regression tests for scalar-value gradient routing at the trunk boundary."""

from __future__ import annotations

import copy

import pytest


torch = pytest.importorskip("torch")

from catan_zero.rl.action_features import CONTEXT_ACTION_FEATURE_SIZE  # noqa: E402
from catan_zero.rl.entity_token_features import LEGAL_ACTION_FEATURE_SIZE  # noqa: E402
from catan_zero.rl.entity_token_policy import EntityGraphConfig, EntityGraphNet  # noqa: E402


def _model_with_active_value_pool():
    model = EntityGraphNet(
        EntityGraphConfig(
            action_size=32,
            static_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
            context_action_feature_size=CONTEXT_ACTION_FEATURE_SIZE,
            legal_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
            hidden_size=16,
            state_layers=1,
            attention_heads=4,
            dropout=0.0,
            value_attention_pool=True,
        )
    ).train()
    # The pool's final projection is intentionally zero at warm-start. Activate
    # it so this test exercises the trained-head path where token gradients exist.
    with torch.no_grad():
        model.value_pool_head[-1].weight.fill_(0.125)
        model.value_pool_head[-1].bias.fill_(0.25)
    return model


def _score_scalar_value(model, *, scale: float):
    generator = torch.Generator().manual_seed(20260715)
    tokens = torch.randn(3, 9, 16, generator=generator, requires_grad=True)
    state = torch.randn(3, 16, generator=generator, requires_grad=True)
    padding_mask = torch.zeros(3, 9, dtype=torch.bool)
    batch = {
        "legal_action_tokens": torch.randn(
            3, 4, LEGAL_ACTION_FEATURE_SIZE, generator=generator
        ),
        "legal_action_context": torch.randn(
            3, 4, CONTEXT_ACTION_FEATURE_SIZE, generator=generator
        ),
    }
    outputs = model.score_actions(
        (tokens, padding_mask, state),
        batch,
        return_final_vp=False,
        return_aux_subgoals=False,
        value_trunk_grad_scale=scale,
    )
    outputs["value"].sum().backward()
    parameter_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
        and (name.startswith("value_head.") or name.startswith("value_pool"))
    }
    return (
        outputs["value"].detach().clone(),
        tokens.grad.detach().clone(),
        state.grad.detach().clone(),
        parameter_gradients,
    )


def test_value_pool_obeys_scalar_value_trunk_gradient_scale_without_scaling_heads():
    full_model = _model_with_active_value_pool()
    scaled_model = copy.deepcopy(full_model)

    full_value, full_tokens, full_state, full_parameters = _score_scalar_value(
        full_model, scale=1.0
    )
    scaled_value, scaled_tokens, scaled_state, scaled_parameters = _score_scalar_value(
        scaled_model, scale=0.25
    )

    # The causal probe changes only upstream derivatives, never the prediction.
    assert torch.equal(scaled_value, full_value)
    torch.testing.assert_close(scaled_tokens, 0.25 * full_tokens, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(scaled_state, 0.25 * full_state, rtol=1e-5, atol=1e-7)

    # Both scalar readout branches still learn at full strength. Only their
    # gradients into the shared state/token trunk are scaled.
    assert scaled_parameters.keys() == full_parameters.keys()
    for name in full_parameters:
        torch.testing.assert_close(
            scaled_parameters[name],
            full_parameters[name],
            rtol=1e-5,
            atol=1e-7,
            msg=name,
        )


def _score_auxiliary_value_readouts(model, *, scale: float):
    generator = torch.Generator().manual_seed(20260716)
    tokens = torch.randn(3, 9, 16, generator=generator, requires_grad=True)
    state = torch.randn(3, 16, generator=generator, requires_grad=True)
    padding_mask = torch.zeros(3, 9, dtype=torch.bool)
    batch = {
        "legal_action_tokens": torch.randn(
            3, 4, LEGAL_ACTION_FEATURE_SIZE, generator=generator
        ),
        "legal_action_context": torch.randn(
            3, 4, CONTEXT_ACTION_FEATURE_SIZE, generator=generator
        ),
    }
    outputs = model.score_actions(
        (tokens, padding_mask, state),
        batch,
        return_final_vp=True,
        return_aux_subgoals=False,
        value_trunk_grad_scale=scale,
    )
    objective = outputs["final_vp"].sum() + outputs[
        "value_categorical_logits"
    ].sum()
    objective.backward()
    parameter_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
        and (
            name.startswith("final_vp_head.")
            or name.startswith("value_categorical_head.")
        )
    }
    return (
        outputs["final_vp"].detach().clone(),
        outputs["value_categorical_logits"].detach().clone(),
        state.grad.detach().clone(),
        parameter_gradients,
    )


def test_value_trunk_scale_covers_final_vp_and_categorical_readouts():
    config = EntityGraphConfig(
        action_size=32,
        static_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
        context_action_feature_size=CONTEXT_ACTION_FEATURE_SIZE,
        legal_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
        hidden_size=16,
        state_layers=1,
        attention_heads=4,
        dropout=0.0,
        value_categorical_bins=5,
    )
    full_model = EntityGraphNet(config).train()
    scaled_model = copy.deepcopy(full_model)

    full_vp, full_cat, full_state, full_parameters = (
        _score_auxiliary_value_readouts(full_model, scale=1.0)
    )
    scaled_vp, scaled_cat, scaled_state, scaled_parameters = (
        _score_auxiliary_value_readouts(scaled_model, scale=0.25)
    )

    assert torch.equal(scaled_vp, full_vp)
    assert torch.equal(scaled_cat, full_cat)
    torch.testing.assert_close(scaled_state, 0.25 * full_state, rtol=1e-5, atol=1e-7)
    assert scaled_parameters.keys() == full_parameters.keys()
    for name in full_parameters:
        torch.testing.assert_close(
            scaled_parameters[name],
            full_parameters[name],
            rtol=1e-5,
            atol=1e-7,
            msg=name,
        )
