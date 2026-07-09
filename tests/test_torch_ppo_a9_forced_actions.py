from __future__ import annotations

import numpy as np
import pytest

from catan_zero.rl.torch_ppo import _standardize_advantages_excluding_forced


def test_standardize_excludes_forced_outlier_from_statistics() -> None:
    """FIX A9: a forced (legal_count == 1) row's advantage must not distort the
    standardization of the genuinely policy-active rows, even when it is a wild outlier."""
    import torch

    active = torch.tensor([1.0, 2.0, 3.0, -1.0, -2.0, -3.0])
    forced_outlier = torch.tensor([1000.0])
    advantages = torch.cat([active, forced_outlier])
    policy_active = torch.tensor([True, True, True, True, True, True, False])

    result = _standardize_advantages_excluding_forced(advantages, policy_active)

    expected_mean = active.mean()
    expected_std = active.std(unbiased=False)
    expected_active = (active - expected_mean) / expected_std

    torch.testing.assert_close(result[:6], expected_active)
    # Sanity: had the outlier polluted the stats, the active rows would be squashed toward 0
    # (std would be dominated by the 1000.0 outlier). Confirm they are NOT tiny.
    assert float(result[:6].abs().max()) > 0.5


def test_standardize_falls_back_to_all_rows_when_none_active() -> None:
    import torch

    advantages = torch.tensor([1.0, 2.0, 3.0])
    policy_active = torch.tensor([False, False, False])

    result = _standardize_advantages_excluding_forced(advantages, policy_active)

    expected = (advantages - advantages.mean()) / advantages.std(unbiased=False)
    torch.testing.assert_close(result, expected)


def test_standardize_matches_plain_standardization_when_all_active() -> None:
    import torch

    advantages = torch.tensor([1.0, -1.0, 2.0, -2.0])
    policy_active = torch.tensor([True, True, True, True])

    result = _standardize_advantages_excluding_forced(advantages, policy_active)

    expected = (advantages - advantages.mean()) / advantages.std(unbiased=False)
    torch.testing.assert_close(result, expected)
