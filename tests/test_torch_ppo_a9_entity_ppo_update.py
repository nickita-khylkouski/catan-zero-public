from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from catan_zero.rl._catanatron import import_catanatron_module


def _collect_real_samples(n: int):
    import_catanatron_module("catanatron")
    from catan_zero.rl.action_features import build_action_context_feature_table
    from catan_zero.rl.entity_token_features import build_entity_token_features
    from catan_zero.rl.multiagent_env import ColonistMultiAgentEnv
    from catan_zero.rl.self_play import StepSample, _phase_from_info, make_env_config

    config = make_env_config(vps_to_win=3)
    env = ColonistMultiAgentEnv(config)
    samples = []
    try:
        observations, info = env.reset(seed=4)
        for decision_index in range(n):
            player = str(info["current_player"])
            observation = np.asarray(observations[player], dtype=np.float64)
            valid_actions = tuple(int(a) for a in info["valid_actions"])
            # forward_legal_np / _entity_graph_batch only consume the numeric ENTITY_BATCH_KEYS
            # fields; "schema" is metadata (a string) and must be dropped before the dict is
            # used as StepSample.entity_features, matching the real training data pipeline.
            entity_features = {
                key: value
                for key, value in build_entity_token_features(env, player).items()
                if key != "schema"
            }
            samples.append(
                StepSample(
                    observation=observation.copy(),
                    valid_actions=valid_actions,
                    action=int(valid_actions[0]),
                    player=player,
                    action_context_features=build_action_context_feature_table(
                        env, info
                    ),
                    entity_features=entity_features,
                    phase=_phase_from_info(info),
                    decision_index=decision_index,
                )
            )
            observations, _rewards, terminated, truncated, info = env.step(
                int(valid_actions[0])
            )
            if terminated or truncated:
                observations, info = env.reset(seed=4 + decision_index + 1)
    finally:
        env.close()
    return samples


def _make_entity_policy():
    from catan_zero.rl.entity_token_policy import EntityGraphPolicy
    from catan_zero.rl.self_play import make_env_config

    return EntityGraphPolicy.create(
        env_config=make_env_config(vps_to_win=3),
        hidden_size=16,
        state_layers=1,
        attention_heads=2,
        dropout=0.0,
        seed=0,
    )


def _make_trajectory(samples, *, force_indices: set[int]):
    from catan_zero.rl.torch_ppo import PPOTrajectory

    n = len(samples)
    adjusted = []
    for i, sample in enumerate(samples):
        if i in force_indices and len(sample.valid_actions) > 1:
            # Force this row down to a single legal action, truncating legal_action_tokens to
            # match -- constructs a genuine legal_count == 1 ("forced") row for the test.
            forced_valid = sample.valid_actions[:1]
            entity = dict(sample.entity_features)
            entity["legal_action_tokens"] = np.asarray(entity["legal_action_tokens"])[
                :1
            ]
            sample = dataclasses.replace(
                sample,
                valid_actions=forced_valid,
                action=forced_valid[0],
                entity_features=entity,
            )
        adjusted.append(sample)
    rng = np.random.default_rng(1)
    return PPOTrajectory(
        samples=adjusted,
        returns=list(rng.normal(size=n)),
        advantages=list(rng.normal(size=n)),
        old_log_probs=[0.0] * n,
        old_values=list(rng.normal(size=n)),
        old_action_probs=[np.zeros(1) for _ in range(n)],
        shaped_rewards=[0.0] * n,
    )


def test_entity_ppo_update_reports_policy_active_fraction() -> None:
    from catan_zero.rl.torch_ppo import ppo_update

    samples = _collect_real_samples(8)
    force_indices = {0, 2, 4}
    trajectory = _make_trajectory(samples, force_indices=force_indices)
    policy = _make_entity_policy()

    metrics = ppo_update(
        policy,
        [trajectory],
        learning_rate=1e-3,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.0,
        epochs=1,
        minibatch_size=64,
    )

    expected_active_fraction = 1.0 - len(force_indices) / len(samples)
    assert metrics["policy_active_fraction"] == expected_active_fraction
    assert np.isfinite(metrics["policy_loss"])
    assert np.isfinite(metrics["value_loss"])


def test_entity_ppo_routes_configured_value_gradient_scale() -> None:
    from catan_zero.rl.torch_ppo import ppo_update

    samples = _collect_real_samples(2)
    trajectory = _make_trajectory(samples, force_indices=set())
    policy = _make_entity_policy()
    observed_scales: list[float] = []
    original_forward = policy.forward_legal_np

    def recording_forward(*args, **kwargs):
        observed_scales.append(float(kwargs.get("value_trunk_grad_scale", 1.0)))
        return original_forward(*args, **kwargs)

    policy.forward_legal_np = recording_forward
    metrics = ppo_update(
        policy,
        [trajectory],
        learning_rate=0.0,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.0,
        epochs=1,
        minibatch_size=64,
        value_trunk_grad_scale=0.1,
    )

    assert observed_scales == [pytest.approx(0.1)]
    assert metrics["value_trunk_grad_scale"] == pytest.approx(0.1)


def test_entity_ppo_scales_only_shared_value_gradients() -> None:
    import torch
    from catan_zero.rl.torch_ppo import _entity_graph_outputs, ppo_update

    samples = _collect_real_samples(3)
    force_indices = set(range(len(samples)))
    baseline_policy = _make_entity_policy()
    scaled_policy = _make_entity_policy()
    scaled_policy.model.load_state_dict(baseline_policy.model.state_dict())
    baseline_trajectory = _make_trajectory(samples, force_indices=force_indices)
    scaled_trajectory = _make_trajectory(samples, force_indices=force_indices)
    with torch.no_grad():
        target_values = (
            _entity_graph_outputs(baseline_policy, baseline_trajectory.samples)["value"]
            .cpu()
            .numpy()
            + 0.01
        )
    for trajectory in (baseline_trajectory, scaled_trajectory):
        trajectory.returns = target_values.tolist()
        trajectory.advantages = [0.0] * len(samples)

    def run(policy, trajectory, scale: float) -> tuple[float, float]:
        ppo_update(
            policy,
            [trajectory],
            learning_rate=0.0,
            clip_ratio=0.2,
            value_coef=1.0,
            entropy_coef=0.0,
            epochs=1,
            minibatch_size=64,
            advantage_normalization="none",
            value_trunk_grad_scale=scale,
        )

        def grad_norm(module) -> float:
            return float(
                torch.sqrt(
                    sum(
                        parameter.grad.detach().square().sum()
                        for parameter in module.parameters()
                        if parameter.grad is not None
                    )
                ).item()
            )

        return grad_norm(policy.model.hex_encoder), grad_norm(policy.model.value_head)

    shared_full, private_full = run(baseline_policy, baseline_trajectory, 1.0)
    shared_scaled, private_scaled = run(scaled_policy, scaled_trajectory, 0.1)

    assert shared_full > 0.0
    assert private_full > 0.0
    assert shared_scaled / shared_full == pytest.approx(0.1, rel=1e-4, abs=1e-6)
    assert private_scaled / private_full == pytest.approx(1.0, rel=1e-4, abs=1e-6)


def test_entity_ppo_update_all_forced_batch_does_not_crash() -> None:
    """Every sample forced (legal_count == 1): policy_loss must fall back to 0 instead of
    crashing on an empty policy_active mask, while the value loss still trains normally."""
    from catan_zero.rl.torch_ppo import ppo_update

    samples = _collect_real_samples(4)
    all_indices = set(range(len(samples)))
    trajectory = _make_trajectory(samples, force_indices=all_indices)
    policy = _make_entity_policy()

    metrics = ppo_update(
        policy,
        [trajectory],
        learning_rate=1e-3,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.0,
        epochs=1,
        minibatch_size=64,
    )

    assert metrics["policy_active_fraction"] == 0.0
    assert metrics["policy_loss"] == 0.0
    assert np.isfinite(metrics["value_loss"])


def test_entity_top_advantage_filter_retains_forced_value_and_policy_signal() -> None:
    """High-advantage forced rows remain critic evidence but cannot crowd every
    policy-active row out of an opt-in top-advantage update."""
    from catan_zero.rl.torch_ppo import ppo_update

    samples = _collect_real_samples(4)
    trajectory = _make_trajectory(samples, force_indices={0, 1})
    trajectory.advantages = [100.0, 90.0, 2.0, 1.0]
    trajectory.returns = [20.0, -20.0, 0.0, 0.0]
    trajectory.old_log_probs = [0.0] * 4
    policy = _make_entity_policy()
    model_before = {
        name: value.detach().clone()
        for name, value in policy.model.state_dict().items()
    }

    metrics = ppo_update(
        policy,
        [trajectory],
        learning_rate=1.0e-3,
        clip_ratio=0.2,
        value_coef=0.0,
        entropy_coef=0.0,
        epochs=1,
        minibatch_size=64,
        top_advantage_fraction=0.5,
        min_advantage_samples=1,
    )

    model_delta = max(
        float((value - model_before[name]).abs().max())
        for name, value in policy.model.state_dict().items()
    )
    assert metrics["samples_before_filter"] == 4.0
    assert metrics["samples"] == 3.0
    assert metrics["policy_active_fraction"] == pytest.approx(1.0 / 3.0)
    assert metrics["advantage_filter_threshold"] == 2.0
    assert metrics["policy_loss"] != 0.0
    assert model_delta > 0.0

    baseline_trajectory = _make_trajectory(samples, force_indices={0, 1})
    baseline_trajectory.advantages = trajectory.advantages
    baseline_trajectory.returns = [0.0] * 4
    baseline_trajectory.old_log_probs = [0.0] * 4
    baseline = ppo_update(
        _make_entity_policy(),
        [baseline_trajectory],
        learning_rate=0.0,
        clip_ratio=0.2,
        value_coef=0.0,
        entropy_coef=0.0,
        epochs=1,
        minibatch_size=64,
        top_advantage_fraction=0.5,
        min_advantage_samples=1,
    )
    assert metrics["value_loss"] > baseline["value_loss"] + 100.0


def test_forced_rows_do_not_dilute_entity_ppo_kl() -> None:
    """KL early-stop telemetry must measure only rows where policy can change."""
    import torch
    from catan_zero.rl.torch_ppo import (
        _entity_action_column,
        _entity_graph_outputs,
        ppo_update,
    )

    samples = _collect_real_samples(2)
    trajectory = _make_trajectory(samples, force_indices={1})
    policy = _make_entity_policy()
    with torch.no_grad():
        outputs = _entity_graph_outputs(policy, trajectory.samples)
        columns = torch.as_tensor(
            [_entity_action_column(sample) for sample in trajectory.samples],
            dtype=torch.long,
            device=policy.device,
        )
        current_log_probs = torch.distributions.Categorical(
            logits=outputs["logits"]
        ).log_prob(columns)
    trajectory.old_log_probs = [
        float(current_log_probs[0].item() + 1.0),
        0.0,
    ]

    metrics = ppo_update(
        policy,
        [trajectory],
        learning_rate=0.0,
        clip_ratio=0.1,
        value_coef=0.0,
        entropy_coef=0.0,
        epochs=1,
        minibatch_size=64,
    )

    assert metrics["approx_kl"] == pytest.approx(np.exp(-1.0), abs=1e-6)


def test_entity_target_kl_uses_nonnegative_k3_without_signed_cancellation() -> None:
    """Opposite signed log-ratios must not cancel the target-KL stop signal."""
    import torch
    from catan_zero.rl.torch_ppo import (
        _behavior_policy_logits,
        _entity_action_column,
        _entity_behavior_valid_mask,
        _entity_graph_outputs,
        ppo_update,
    )

    samples = _collect_real_samples(2)
    trajectory = _make_trajectory(samples, force_indices=set())
    policy = _make_entity_policy()
    with torch.no_grad():
        outputs = _entity_graph_outputs(policy, trajectory.samples)
        for row, sample in enumerate(trajectory.samples):
            least_likely_column = int(
                outputs["logits"][row, : len(sample.valid_actions)].argmin().item()
            )
            sample.action = sample.valid_actions[least_likely_column]
        behavior_logits = _behavior_policy_logits(
            outputs["logits"],
            1.0,
            valid_mask=_entity_behavior_valid_mask(
                trajectory.samples,
                outputs["logits"],
            ),
        )
        columns = torch.as_tensor(
            [_entity_action_column(sample) for sample in trajectory.samples],
            dtype=torch.long,
            device=policy.device,
        )
        current = (
            torch.distributions.Categorical(logits=behavior_logits)
            .log_prob(columns)
            .cpu()
            .numpy()
        )
    # The historical signed estimator mean(old_logp - new_logp) is exactly
    # zero here despite extreme per-row drift in opposite directions.
    drift = 0.5
    trajectory.old_log_probs = [
        float(current[0] + drift),
        float(current[1] - drift),
    ]

    metrics = ppo_update(
        policy,
        [trajectory],
        learning_rate=0.0,
        clip_ratio=0.2,
        value_coef=0.0,
        entropy_coef=0.0,
        epochs=4,
        minibatch_size=64,
        target_kl=0.01,
    )

    assert metrics["approx_kl"] == pytest.approx(
        np.cosh(drift) - 1.0, rel=1e-6
    )
    assert metrics["approx_kl"] >= 0.0
    assert metrics["early_stop"] == 1.0
    assert metrics["minibatches"] == 0.0


def test_entity_ppo_uses_learner_reference_without_losing_actor_evidence() -> None:
    import torch
    from catan_zero.rl.torch_ppo import (
        _behavior_policy_logits,
        _entity_action_column,
        _entity_behavior_valid_mask,
        _entity_graph_outputs,
        ppo_update,
    )

    samples = _collect_real_samples(4)
    trajectory = _make_trajectory(samples, force_indices=set())
    policy = _make_entity_policy()
    with torch.no_grad():
        outputs = _entity_graph_outputs(policy, trajectory.samples)
        behavior_logits = _behavior_policy_logits(
            outputs["logits"],
            1.0,
            valid_mask=_entity_behavior_valid_mask(
                trajectory.samples,
                outputs["logits"],
            ),
        )
        columns = torch.as_tensor(
            [_entity_action_column(sample) for sample in trajectory.samples],
            dtype=torch.long,
            device=policy.device,
        )
        learner_log_probs = (
            torch.distributions.Categorical(logits=behavior_logits)
            .log_prob(columns)
            .cpu()
            .tolist()
        )
    trajectory.old_log_probs = [value + 0.7 for value in learner_log_probs]
    trajectory.ppo_reference_log_probs = learner_log_probs
    actor_evidence = trajectory.old_log_probs.copy()

    metrics = ppo_update(
        policy,
        [trajectory],
        learning_rate=0.0,
        clip_ratio=0.1,
        value_coef=0.0,
        entropy_coef=0.0,
        epochs=1,
        minibatch_size=64,
    )

    assert metrics["approx_kl"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["clip_fraction"] == 0.0
    assert trajectory.old_log_probs == actor_evidence


@pytest.mark.parametrize("field", ["returns", "advantages"])
def test_entity_update_rejects_cross_trajectory_target_misalignment(
    field: str,
) -> None:
    """Equal global row counts must not hide a shift across trajectory boundaries."""
    samples = _collect_real_samples(3)
    trajectories = [
        _make_trajectory(samples[:2], force_indices=set()),
        _make_trajectory(samples[2:], force_indices=set()),
    ]
    first_values = getattr(trajectories[0], field)
    second_values = getattr(trajectories[1], field)
    second_values.insert(0, first_values.pop())

    with pytest.raises(
        ValueError,
        match=rf"PPOTrajectory\.{field} must align with samples.*trajectory 0",
    ):
        from catan_zero.rl.torch_ppo import ppo_update

        ppo_update(
            _make_entity_policy(),
            trajectories,
            learning_rate=0.0,
            clip_ratio=0.2,
            value_coef=1.0,
            entropy_coef=0.0,
            epochs=1,
            minibatch_size=64,
        )


def test_entity_update_accepts_aligned_multi_trajectory_targets() -> None:
    from catan_zero.rl.torch_ppo import ppo_update

    samples = _collect_real_samples(3)
    trajectories = [
        _make_trajectory(samples[:2], force_indices=set()),
        _make_trajectory(samples[2:], force_indices=set()),
    ]

    metrics = ppo_update(
        _make_entity_policy(),
        trajectories,
        learning_rate=0.0,
        clip_ratio=0.2,
        value_coef=1.0,
        entropy_coef=0.0,
        epochs=1,
        minibatch_size=64,
    )

    assert metrics["samples"] == 3.0
