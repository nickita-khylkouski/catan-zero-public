from __future__ import annotations

import argparse
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from catan_zero.rl.self_play import StepSample
from catan_zero.rl.torch_ppo import (
    PPOTrajectory,
    TorchPPOPolicy,
    _masked_logits,
    _policy_observation_array,
    _ppo_reference_array,
    ppo_update,
)
from catan_zero.rl.vtrace import vtrace_from_log_probs
from tools import ppo_distributed_learner as learner
from tools.ppo_distributed_learner import (
    _discounts_for_trajectory,
    _recompute_target_logp_and_values_batched,
    apply_vtrace_in_place,
)


def _make_policy() -> TorchPPOPolicy:
    return TorchPPOPolicy(4, 5, hidden_size=8, seed=7, architecture="flat")


def _make_trajectory(n: int = 6) -> argparse.Namespace:
    rng = np.random.default_rng(0)
    samples = [
        StepSample(
            observation=rng.normal(size=4).astype(np.float32),
            valid_actions=(0, 1, 2, 3, 4),
            action=int(rng.integers(0, 5)),
            player="BLUE",
        )
        for _ in range(n)
    ]
    return argparse.Namespace(samples=samples)


def test_recompute_target_logp_scales_by_behavior_temperature() -> None:
    policy = _make_policy()
    trajectory = _make_trajectory()

    raw_logp, raw_values = _recompute_target_logp_and_values_batched(
        policy, [trajectory], forward_chunk=4, behavior_temperature=1.0
    )
    scaled_logp, scaled_values = _recompute_target_logp_and_values_batched(
        policy, [trajectory], forward_chunk=4, behavior_temperature=0.25
    )

    # Values are unaffected by temperature scaling of the policy logits.
    np.testing.assert_allclose(raw_values, scaled_values, atol=1e-6)
    # Log-probs at behavior_temperature=0.25 must differ from T=1.0 (temperature actually
    # threaded through), mirroring ppo_update's behavior_logits scaling.
    assert not np.allclose(raw_logp, scaled_logp)


def test_recompute_target_logp_matches_manual_temperature_scaling() -> None:
    import torch

    policy = _make_policy()
    trajectory = _make_trajectory(n=3)
    samples = trajectory.samples

    target_logp, _ = _recompute_target_logp_and_values_batched(
        policy, [trajectory], forward_chunk=8192, behavior_temperature=0.25
    )

    observations = _policy_observation_array(policy, samples)
    obs_t = torch.as_tensor(observations, dtype=torch.float32, device=policy.device)
    logits, _ = policy.forward(obs_t, None)
    masked = _masked_logits(logits, [s.valid_actions for s in samples], policy.action_size)
    scaled = torch.clamp(masked / 0.25, min=-50.0, max=50.0)
    actions_t = torch.as_tensor(
        [s.action for s in samples], dtype=torch.long, device=policy.device
    )
    expected = (
        torch.distributions.Categorical(logits=scaled)
        .log_prob(actions_t)
        .detach()
        .cpu()
        .numpy()
    )

    np.testing.assert_allclose(target_logp, expected, atol=1e-5)


def test_recompute_target_logp_defaults_to_unscaled_temperature() -> None:
    """behavior_temperature must default to 1.0 (a no-op) for backward compatibility."""
    policy = _make_policy()
    trajectory = _make_trajectory(n=3)

    default_logp, _ = _recompute_target_logp_and_values_batched(
        policy, [trajectory], forward_chunk=8192
    )
    explicit_logp, _ = _recompute_target_logp_and_values_batched(
        policy, [trajectory], forward_chunk=8192, behavior_temperature=1.0
    )

    np.testing.assert_allclose(default_logp, explicit_logp, atol=1e-9)


def test_entity_recompute_uses_actor_saturation_at_temperature_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    from catan_zero.rl import torch_ppo

    samples = [
        StepSample(
            observation=np.zeros(1, dtype=np.float32),
            valid_actions=(4, 5),
            action=4,
            player="BLUE",
            entity_features={},
        ),
        StepSample(
            observation=np.zeros(1, dtype=np.float32),
            valid_actions=(4, 5, 6),
            action=5,
            player="BLUE",
            entity_features={},
        ),
    ]
    policy = SimpleNamespace(
        device=torch.device("cpu"),
        forward_legal_np=lambda *_args, **_kwargs: None,
    )
    raw_logits = torch.as_tensor(
        [[-200.0, -200.0, -1.0e9], [-200.0, -200.0, -200.0]],
        dtype=torch.float32,
    )

    def fake_outputs(_policy, batch_samples, *, return_q):
        assert return_q is False
        count = len(batch_samples)
        return {
            "logits": raw_logits[:count],
            "value": torch.zeros(count, dtype=torch.float32),
        }

    monkeypatch.setattr(torch_ppo, "_entity_graph_outputs", fake_outputs)
    target_logp, _ = _recompute_target_logp_and_values_batched(
        policy,
        [SimpleNamespace(samples=samples)],
        forward_chunk=8192,
        behavior_temperature=1.0,
    )

    expected_logits = torch_ppo._behavior_policy_logits(  # noqa: SLF001
        raw_logits,
        1.0,
        valid_mask=torch.as_tensor(
            [[True, True, False], [True, True, True]]
        ),
    )
    expected = (
        torch.distributions.Categorical(logits=expected_logits)
        .log_prob(torch.as_tensor([0, 1]))
        .numpy()
    )
    np.testing.assert_allclose(target_logp, expected, atol=1e-6)


def test_vtrace_rebases_ppo_reference_without_applying_staleness_ratio_twice() -> None:
    policy = _make_policy()
    samples = _make_trajectory(n=6).samples
    target_logp, current_values = _recompute_target_logp_and_values_batched(
        policy,
        [SimpleNamespace(samples=samples)],
    )
    # Alternate both sides of rho=1 so this would produce non-zero initial PPO KL and clipping
    # if the stale actor behavior remained PPO's reference after V-trace correction.
    log_rhos = np.asarray([0.7, -0.7, 0.4, -0.4, 0.2, -0.2])
    behavior_logp = target_logp - log_rhos
    rewards = np.asarray([0.0, 0.2, 0.0, -0.1, 0.0, 1.0])
    trajectory = PPOTrajectory(
        samples=samples,
        returns=[99.0] * len(samples),
        advantages=[99.0] * len(samples),
        old_log_probs=behavior_logp.tolist(),
        old_values=[-99.0] * len(samples),
        old_action_probs=[
            np.full(len(sample.valid_actions), 1.0 / len(sample.valid_actions))
            for sample in samples
        ],
        shaped_rewards=[0.0] * len(samples),
        rewards=rewards.tolist(),
    )
    config = SimpleNamespace(
        vtrace_forward_chunk=8192,
        behavior_temperature=1.0,
        vtrace_use_current_values=True,
        gamma=0.9,
        vtrace_clip_rho=1.0,
        vtrace_clip_pg_rho=1.0,
    )
    expected = vtrace_from_log_probs(
        behavior_log_probs=behavior_logp,
        target_log_probs=target_logp,
        discounts=_discounts_for_trajectory(trajectory, gamma=config.gamma),
        rewards=rewards,
        values=current_values,
        bootstrap_value=0.0,
        clip_rho_threshold=config.vtrace_clip_rho,
        clip_pg_rho_threshold=config.vtrace_clip_pg_rho,
    )

    actor_log_probs = trajectory.old_log_probs.copy()
    actor_values = trajectory.old_values.copy()
    stats = apply_vtrace_in_place(policy, [trajectory], config)

    assert stats["vtrace_skipped"] == 0.0
    np.testing.assert_allclose(trajectory.advantages, expected.pg_advantages, atol=1e-6)
    np.testing.assert_allclose(trajectory.returns, expected.vs, atol=1e-6)
    assert trajectory.old_log_probs == actor_log_probs
    assert trajectory.old_values == actor_values
    np.testing.assert_allclose(
        trajectory.ppo_reference_log_probs,
        target_logp,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        trajectory.ppo_reference_values,
        current_values,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        _ppo_reference_array(
            [trajectory],
            reference_attr="ppo_reference_log_probs",
            fallback_attr="old_log_probs",
        ),
        target_logp,
        atol=1e-6,
    )

    # Reanalysis is idempotent because it still corrects from the immutable actor behavior,
    # never from the learner-only PPO reference written by the first pass.
    repeated_stats = apply_vtrace_in_place(policy, [trajectory], config)
    assert repeated_stats["vtrace_skipped"] == 0.0
    np.testing.assert_allclose(trajectory.advantages, expected.pg_advantages, atol=1e-6)
    assert trajectory.old_log_probs == actor_log_probs

    update_stats = ppo_update(
        policy,
        [trajectory],
        learning_rate=0.0,
        value_coef=0.5,
        entropy_coef=0.0,
        epochs=1,
        minibatch_size=len(samples),
        clip_ratio=0.1,
        target_kl=0.0075,
        behavior_temperature=1.0,
        advantage_normalization="none",
    )
    assert update_stats["approx_kl"] == pytest.approx(0.0, abs=1e-6)
    assert update_stats["clip_fraction"] == 0.0
    assert update_stats["early_stop"] == 0.0


def test_legacy_trajectory_falls_back_and_malformed_ppo_reference_fails() -> None:
    samples = _make_trajectory(n=3).samples
    trajectory = PPOTrajectory(
        samples=samples,
        returns=[0.0] * 3,
        advantages=[0.0] * 3,
        old_log_probs=[-1.0, -2.0, -3.0],
        old_values=[1.0, 2.0, 3.0],
        old_action_probs=[np.full(5, 0.2) for _ in samples],
        shaped_rewards=[0.0] * 3,
    )
    # Older name-keyed slot pickles do not contain the newly appended fields.
    del trajectory.ppo_reference_log_probs
    del trajectory.ppo_reference_values
    restored = pickle.loads(pickle.dumps(trajectory))

    np.testing.assert_array_equal(
        _ppo_reference_array(
            [restored],
            reference_attr="ppo_reference_log_probs",
            fallback_attr="old_log_probs",
        ),
        np.asarray([-1.0, -2.0, -3.0], dtype=np.float32),
    )

    restored.ppo_reference_log_probs = [-1.0]
    with pytest.raises(ValueError, match="must align with samples"):
        _ppo_reference_array(
            [restored],
            reference_attr="ppo_reference_log_probs",
            fallback_attr="old_log_probs",
        )


def test_vtrace_skip_does_not_commit_staged_learner_references(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = _make_trajectory(n=2).samples
    trajectories = [
        SimpleNamespace(
            samples=[sample],
            returns=[10.0 + index],
            advantages=[20.0 + index],
            old_log_probs=[-1.0 - index],
            old_values=[-2.0 - index],
            shaped_rewards=[0.0],
            rewards=[1.0],
            truncated=False,
            bootstrap_value=0.0,
        )
        for index, sample in enumerate(samples)
    ]
    before = [
        (
            trajectory.returns.copy(),
            trajectory.advantages.copy(),
            trajectory.old_log_probs.copy(),
            trajectory.old_values.copy(),
        )
        for trajectory in trajectories
    ]
    monkeypatch.setattr(
        learner,
        "_recompute_target_logp_and_values_batched",
        lambda *_args, **_kwargs: (
            np.asarray([0.0, np.nan]),
            np.asarray([0.0, np.nan]),
        ),
    )
    config = SimpleNamespace(
        vtrace_forward_chunk=8192,
        behavior_temperature=1.0,
        vtrace_use_current_values=True,
        gamma=0.9,
        vtrace_clip_rho=1.0,
        vtrace_clip_pg_rho=1.0,
    )

    stats = apply_vtrace_in_place(object(), trajectories, config)

    assert stats["vtrace_skipped"] == 1.0
    for trajectory, snapshot in zip(trajectories, before, strict=True):
        assert (
            trajectory.returns,
            trajectory.advantages,
            trajectory.old_log_probs,
            trajectory.old_values,
        ) == snapshot
        assert not hasattr(trajectory, "ppo_reference_log_probs")
        assert not hasattr(trajectory, "ppo_reference_values")


def test_current_value_vtrace_uses_recomputed_cutoff_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample, bootstrap_sample = _make_trajectory(n=2).samples
    trajectory = SimpleNamespace(
        samples=[sample],
        returns=[99.0],
        advantages=[99.0],
        old_log_probs=[0.0],
        old_values=[0.0],
        shaped_rewards=[0.0],
        rewards=[0.0],
        truncated=True,
        bootstrap_value=9.0,
        bootstrap_sample=bootstrap_sample,
    )

    def recompute(_policy, batches, **_kwargs):
        flattened = [item for batch in batches for item in batch.samples]
        if len(flattened) == 1 and flattened[0] is sample:
            return np.asarray([0.0]), np.asarray([0.0])
        assert len(flattened) == 1 and flattened[0] is bootstrap_sample
        return np.asarray([0.0]), np.asarray([2.0])

    monkeypatch.setattr(
        learner,
        "_recompute_target_logp_and_values_batched",
        recompute,
    )
    config = SimpleNamespace(
        vtrace_forward_chunk=8192,
        behavior_temperature=1.0,
        vtrace_use_current_values=True,
        gamma=1.0,
        vtrace_clip_rho=1.0,
        vtrace_clip_pg_rho=1.0,
    )

    stats = apply_vtrace_in_place(object(), [trajectory], config)

    assert stats["vtrace_skipped"] == 0.0
    assert trajectory.returns == pytest.approx([2.0])
    assert trajectory.returns != pytest.approx([9.0])


def test_current_value_vtrace_rejects_old_truncated_shard_without_cutoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = _make_trajectory(n=1).samples[0]
    trajectory = SimpleNamespace(
        samples=[sample],
        returns=[7.0],
        advantages=[8.0],
        old_log_probs=[0.0],
        old_values=[0.0],
        shaped_rewards=[0.0],
        rewards=[0.0],
        truncated=True,
        bootstrap_value=3.0,
    )
    monkeypatch.setattr(
        learner,
        "_recompute_target_logp_and_values_batched",
        lambda *_args, **_kwargs: (np.asarray([0.0]), np.asarray([0.0])),
    )
    config = SimpleNamespace(
        vtrace_forward_chunk=8192,
        behavior_temperature=1.0,
        vtrace_use_current_values=True,
        gamma=1.0,
        vtrace_clip_rho=1.0,
        vtrace_clip_pg_rho=1.0,
    )

    stats = apply_vtrace_in_place(object(), [trajectory], config)

    assert stats["vtrace_skipped"] == 1.0
    assert stats["vtrace_missing_current_bootstrap"] == 1.0
    assert trajectory.returns == [7.0]
    assert trajectory.advantages == [8.0]
