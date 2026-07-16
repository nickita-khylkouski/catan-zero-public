from __future__ import annotations

import torch
import torch.nn.functional as F

from tools import train_bc


def _loss_and_gradient(*, semantics: str, weight: float, has_soft: bool):
    logits = torch.tensor([[2.0, -1.0]], dtype=torch.float64, requires_grad=True)
    hard_loss = F.cross_entropy(logits, torch.tensor([1]), reduction="none")
    soft_target = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    soft_loss = -(soft_target * F.log_softmax(logits, dim=-1)).sum(dim=-1)
    loss = train_bc._policy_target_per_sample_loss(
        hard_loss,
        soft_loss,
        torch.tensor([has_soft]),
        soft_target_weight=weight,
        blend_semantics=semantics,
    ).sum()
    loss.backward()
    return loss.detach(), logits.grad.detach()


def test_v2_usable_policy_target_ignores_sampled_played_action_gradient() -> None:
    # The played action is action 1 while the authenticated target's argmax is
    # action 0. A usable target row must therefore produce exactly pure target
    # CE, including the gradient that drives learning.
    actual_loss, actual_gradient = _loss_and_gradient(
        semantics=train_bc.POLICY_TARGET_BLEND_FALLBACK_V2,
        weight=1.0,
        has_soft=True,
    )

    logits = torch.tensor([[2.0, -1.0]], dtype=torch.float64, requires_grad=True)
    expected_loss = -F.log_softmax(logits, dim=-1)[0, 0]
    expected_loss.backward()
    torch.testing.assert_close(actual_loss, expected_loss.detach())
    torch.testing.assert_close(actual_gradient, logits.grad)


def test_v2_missing_policy_target_falls_back_to_played_action() -> None:
    actual_loss, actual_gradient = _loss_and_gradient(
        semantics=train_bc.POLICY_TARGET_BLEND_FALLBACK_V2,
        weight=1.0,
        has_soft=False,
    )

    logits = torch.tensor([[2.0, -1.0]], dtype=torch.float64, requires_grad=True)
    expected_loss = F.cross_entropy(logits, torch.tensor([1]))
    expected_loss.backward()
    torch.testing.assert_close(actual_loss, expected_loss.detach())
    torch.testing.assert_close(actual_gradient, logits.grad)


def test_legacy_semantics_exactly_replays_interpolated_gradient() -> None:
    actual_loss, actual_gradient = _loss_and_gradient(
        semantics=train_bc.POLICY_TARGET_BLEND_LEGACY_V1,
        weight=0.9,
        has_soft=True,
    )

    logits = torch.tensor([[2.0, -1.0]], dtype=torch.float64, requires_grad=True)
    soft_loss = -F.log_softmax(logits, dim=-1)[0, 0]
    hard_loss = F.cross_entropy(logits, torch.tensor([1]))
    expected_loss = 0.9 * soft_loss + 0.1 * hard_loss
    expected_loss.backward()
    torch.testing.assert_close(actual_loss, expected_loss.detach())
    torch.testing.assert_close(actual_gradient, logits.grad)
