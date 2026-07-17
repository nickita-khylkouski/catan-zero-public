from __future__ import annotations

import numpy as np
import pytest

from catan_zero.rl.self_play import StepSample
from catan_zero.rl.torch_ppo import (
    PPOTrajectory,
    TorchPPOPolicy,
    _top_advantage_keep_indices,
    ppo_update,
)


def _dense_trajectory(
    *,
    valid_actions: list[tuple[int, ...]],
    advantages: list[float],
    returns: list[float] | None = None,
) -> PPOTrajectory:
    observations = [
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([0.0, 1.0, 0.0]),
        np.asarray([0.0, 0.0, 1.0]),
        np.asarray([1.0, 1.0, 0.0]),
    ]
    samples = [
        StepSample(
            observation=observations[index % len(observations)],
            valid_actions=actions,
            action=actions[0],
            player="BLUE",
        )
        for index, actions in enumerate(valid_actions)
    ]
    return PPOTrajectory(
        samples=samples,
        returns=list(advantages if returns is None else returns),
        advantages=advantages,
        old_log_probs=[float(-np.log(len(actions))) for actions in valid_actions],
        old_values=[0.0] * len(samples),
        old_action_probs=[
            np.full(len(actions), 1.0 / len(actions), dtype=np.float32)
            for actions in valid_actions
        ],
        shaped_rewards=[0.0] * len(samples),
    )


def _dense_update(
    policy: TorchPPOPolicy,
    trajectory: PPOTrajectory,
    *,
    top_fraction: float,
    learning_rate: float = 0.0,
    value_coef: float = 0.0,
) -> dict[str, float]:
    return ppo_update(
        policy,
        [trajectory],
        learning_rate=learning_rate,
        clip_ratio=10.0,
        value_coef=value_coef,
        entropy_coef=0.0,
        epochs=1,
        minibatch_size=64,
        top_advantage_fraction=top_fraction,
        min_advantage_samples=1,
    )


def test_top_advantage_filter_ranks_only_policy_eligible_rows() -> None:
    advantages = np.asarray([100.0, 90.0, 2.0, 1.0], dtype=np.float32)
    eligible = np.asarray([False, False, True, True])

    indices, threshold = _top_advantage_keep_indices(
        advantages,
        top_fraction=0.5,
        min_samples=1,
        eligible_mask=eligible,
        retain_ineligible=True,
    )

    np.testing.assert_array_equal(indices, [0, 1, 2])
    assert threshold == 2.0


def test_top_advantage_filter_falls_back_when_no_eligible_advantage_is_positive() -> None:
    advantages = np.asarray([100.0, 90.0, 0.0, -1.0], dtype=np.float32)
    eligible = np.asarray([False, False, True, True])

    indices, threshold = _top_advantage_keep_indices(
        advantages,
        top_fraction=0.25,
        min_samples=1,
        eligible_mask=eligible,
        retain_ineligible=True,
    )

    np.testing.assert_array_equal(indices, np.arange(4))
    assert threshold == 0.0


def test_top_advantage_fraction_one_preserves_every_row() -> None:
    advantages = np.asarray([100.0, 90.0, 2.0, 1.0], dtype=np.float32)
    eligible = np.asarray([False, False, True, True])

    indices, threshold = _top_advantage_keep_indices(
        advantages,
        top_fraction=1.0,
        min_samples=1,
        eligible_mask=eligible,
        retain_ineligible=True,
    )

    np.testing.assert_array_equal(indices, np.arange(4))
    assert threshold == 0.0


def test_dense_filter_retains_forced_value_rows_and_singleton_policy_signal() -> None:
    pytest.importorskip("torch")
    trajectory = _dense_trajectory(
        valid_actions=[(0,), (1,), (0, 1), (0, 1)],
        advantages=[100.0, 90.0, 2.0, 1.0],
        returns=[20.0, -20.0, 0.0, 0.0],
    )
    policy = TorchPPOPolicy(3, 5, hidden_size=8, seed=41)
    actor_before = {
        name: value.detach().clone() for name, value in policy.actor.state_dict().items()
    }

    metrics = _dense_update(
        policy,
        trajectory,
        top_fraction=0.5,
        learning_rate=1.0e-3,
    )

    actor_delta = max(
        float((value - actor_before[name]).abs().max())
        for name, value in policy.actor.state_dict().items()
    )
    assert metrics["samples_before_filter"] == 4.0
    assert metrics["samples"] == 3.0
    assert metrics["policy_active_fraction"] == pytest.approx(1.0 / 3.0)
    assert metrics["advantage_filter_threshold"] == 2.0
    assert metrics["policy_loss"] != 0.0
    assert actor_delta > 0.0

    zero_forced_returns = _dense_trajectory(
        valid_actions=[(0,), (1,), (0, 1), (0, 1)],
        advantages=[100.0, 90.0, 2.0, 1.0],
        returns=[0.0, 0.0, 0.0, 0.0],
    )
    baseline = _dense_update(
        TorchPPOPolicy(3, 5, hidden_size=8, seed=41),
        zero_forced_returns,
        top_fraction=0.5,
    )
    assert metrics["value_loss"] > baseline["value_loss"] + 100.0


def test_dense_all_forced_rows_are_value_only() -> None:
    pytest.importorskip("torch")
    trajectory = _dense_trajectory(
        valid_actions=[(0,), (1,)],
        advantages=[2.0, 1.0],
        returns=[3.0, -3.0],
    )
    trajectory.old_log_probs = [5.0, -5.0]
    policy = TorchPPOPolicy(3, 5, hidden_size=8, seed=42)

    metrics = _dense_update(
        policy,
        trajectory,
        top_fraction=0.5,
        value_coef=1.0,
    )

    assert metrics["samples"] == 2.0
    assert metrics["policy_active_fraction"] == 0.0
    assert metrics["policy_loss"] == 0.0
    assert metrics["entropy"] == 0.0
    assert metrics["approx_kl"] == 0.0
    assert metrics["clip_fraction"] == 0.0
    assert metrics["value_loss"] > 0.0
