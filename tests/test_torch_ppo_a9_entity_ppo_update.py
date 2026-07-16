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
        rng = np.random.default_rng(0)
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
                    action_context_features=build_action_context_feature_table(env, info),
                    entity_features=entity_features,
                    phase=_phase_from_info(info),
                    decision_index=decision_index,
                )
            )
            observations, _rewards, terminated, truncated, info = env.step(int(valid_actions[0]))
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
            entity["legal_action_tokens"] = np.asarray(entity["legal_action_tokens"])[:1]
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

    assert metrics["approx_kl"] == pytest.approx(1.0, abs=1e-6)
