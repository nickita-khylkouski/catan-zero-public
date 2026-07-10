from __future__ import annotations

import math

import pytest
import torch

from catan_zero.rl.klent import (
    KLENTConfig,
    alternating_lambda_returns,
    catan_lambda_returns,
    improved_policy,
    loss,
)


def test_improved_policy_matches_paper_closed_form_and_masks_illegal() -> None:
    config = KLENTConfig(entropy_coefficient=0.03, reverse_kl_coefficient=0.1)
    logits = torch.tensor([[0.2, -0.3, 4.0]])
    q = torch.tensor([[0.4, -0.1, 100.0]])
    mask = torch.tensor([[True, True, False]])
    result = improved_policy(logits, q, mask, config)
    expected = torch.softmax((0.1 * logits[:, :2] + q[:, :2]) / 0.13, dim=-1)
    assert torch.allclose(result[:, :2], expected)
    assert result[0, 2] == 0.0
    assert result.sum() == pytest.approx(1.0)


def test_larger_q_can_overcome_prior_but_regularization_remains() -> None:
    mask = torch.ones((1, 2), dtype=torch.bool)
    result = improved_policy(
        torch.tensor([[3.0, 0.0]]),
        torch.tensor([[0.0, 1.0]]),
        mask,
        KLENTConfig(entropy_coefficient=0.1, reverse_kl_coefficient=1.0),
    )
    assert result[0, 0] > result[0, 1]
    lower_beta = improved_policy(
        torch.tensor([[3.0, 0.0]]),
        torch.tensor([[0.0, 1.0]]),
        mask,
        KLENTConfig(entropy_coefficient=0.1, reverse_kl_coefficient=0.01),
    )
    assert lower_beta[0, 1] > result[0, 1]


def test_alternating_lambda_returns_terminal_and_sign_flip() -> None:
    cfg = KLENTConfig(trace_horizon=1.0e-6)
    rewards = torch.tensor([0.0, 0.0, 1.0])
    values = torch.tensor([0.2, 0.4, 0.8])
    terminal = torch.tensor([False, False, True])
    returns = alternating_lambda_returns(rewards, values, terminal, config=cfg)
    assert returns[2] == pytest.approx(1.0)
    assert returns[1] == pytest.approx(-0.8, abs=1.0e-5)
    assert returns[0] == pytest.approx(-0.4, abs=1.0e-5)


def test_truncated_trajectory_requires_bootstrap() -> None:
    rewards = torch.zeros(2)
    values = torch.tensor([0.2, 0.3])
    terminal = torch.tensor([False, False])
    with pytest.raises(ValueError, match="bootstrap"):
        alternating_lambda_returns(rewards, values, terminal)
    result = alternating_lambda_returns(
        rewards, values, terminal, bootstrap_value=torch.tensor(0.5)
    )
    assert torch.isfinite(result).all()


def test_catan_returns_preserve_perspective_across_consecutive_prompts() -> None:
    cfg = KLENTConfig(trace_horizon=1.0e-6)
    rewards = torch.tensor([0.0, 0.0, 0.0, 1.0])
    values = torch.tensor([0.1, 0.2, 0.4, 0.8])
    terminated = torch.tensor([False, False, False, True])
    # RED acts twice, BLUE once, then RED wins.
    players = torch.tensor([0, 0, 1, 0])
    returns = catan_lambda_returns(
        rewards, values, terminated, players, config=cfg
    )
    assert returns[3] == pytest.approx(1.0)
    assert returns[2] == pytest.approx(-0.8, abs=1.0e-5)
    assert returns[1] == pytest.approx(-0.4, abs=1.0e-5)
    assert returns[0] == pytest.approx(0.2, abs=1.0e-5)


def test_catan_truncation_requires_bootstrap_player() -> None:
    args = (
        torch.zeros(2),
        torch.tensor([0.1, 0.2]),
        torch.tensor([False, False]),
        torch.tensor([0, 1]),
    )
    with pytest.raises(ValueError, match="bootstrap_value"):
        catan_lambda_returns(*args)
    with pytest.raises(ValueError, match="bootstrap_player_id"):
        catan_lambda_returns(*args, bootstrap_value=0.3)
    result = catan_lambda_returns(
        *args, bootstrap_value=0.3, bootstrap_player_id=0
    )
    assert torch.isfinite(result).all()


def test_catan_long_trace_exactly_composes_same_and_changed_perspectives() -> None:
    cfg = KLENTConfig(trace_horizon=1.0e9)
    returns = catan_lambda_returns(
        torch.tensor([0.0, 0.0, 0.0, 1.0]),
        torch.zeros(4),
        torch.tensor([False, False, False, True]),
        torch.tensor([0, 0, 1, 0]),
        config=cfg,
    )
    assert returns.tolist() == pytest.approx([1.0, 1.0, -1.0, 1.0], abs=1.0e-6)


@pytest.mark.parametrize(("bootstrap_player", "expected"), [(0, 0.75), (1, -0.75)])
def test_catan_bootstrap_is_converted_from_bootstrap_player_perspective(
    bootstrap_player: int, expected: float
) -> None:
    result = catan_lambda_returns(
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        torch.tensor([False]),
        torch.tensor([0]),
        config=KLENTConfig(trace_horizon=1.0e9),
        bootstrap_value=0.75,
        bootstrap_player_id=bootstrap_player,
    )
    assert result.item() == pytest.approx(expected, abs=1.0e-6)


def test_loss_trains_policy_and_chosen_action_q() -> None:
    logits = torch.tensor([[0.2, -0.1], [0.0, 0.3]], requires_grad=True)
    q_values = torch.tensor([[0.4, 0.2], [-0.3, 0.1]], requires_grad=True)
    result = loss(
        logits,
        q_values,
        torch.tensor([0, 1]),
        torch.tensor([[0.8, 0.2], [0.3, 0.7]]),
        torch.tensor([1.0, -1.0]),
        legal_mask=torch.ones((2, 2), dtype=torch.bool),
    )
    result["loss"].backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert q_values.grad is not None and torch.isfinite(q_values.grad).all()
    assert q_values.grad[0, 1] == 0.0
    assert q_values.grad[1, 0] == 0.0


def test_loss_handles_padded_illegal_columns_without_nan() -> None:
    result = loss(
        torch.tensor([[0.2, 0.1, 99.0]]),
        torch.tensor([[0.0, 0.0, 0.0]]),
        torch.tensor([0]),
        torch.tensor([[0.7, 0.3, 0.0]]),
        torch.tensor([0.5]),
        legal_mask=torch.tensor([[True, True, False]]),
    )
    assert torch.isfinite(result["loss"])


@pytest.mark.parametrize(
    "config",
    [
        KLENTConfig(entropy_coefficient=-1.0),
        KLENTConfig(entropy_coefficient=0.0, reverse_kl_coefficient=0.0),
        KLENTConfig(trace_horizon=math.inf),
        KLENTConfig(q_loss_weight=-1.0),
    ],
)
def test_invalid_config_fails_closed(config: KLENTConfig) -> None:
    with pytest.raises(ValueError):
        config.validate()
