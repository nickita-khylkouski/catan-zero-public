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
        observations, info = env.reset(seed=6)
        for decision_index in range(n):
            player = str(info["current_player"])
            observation = np.asarray(observations[player], dtype=np.float64)
            valid_actions = tuple(int(a) for a in info["valid_actions"])
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
                observations, info = env.reset(seed=6 + decision_index + 1)
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
        seed=0,
    )


def _make_trajectory(samples):
    from catan_zero.rl.torch_ppo import PPOTrajectory

    n = len(samples)
    rng = np.random.default_rng(1)
    return PPOTrajectory(
        samples=samples,
        returns=list(rng.normal(size=n)),
        advantages=list(rng.normal(size=n)),
        old_log_probs=[0.0] * n,
        old_values=list(rng.normal(size=n)),
        old_action_probs=[np.zeros(1) for _ in range(n)],
        shaped_rewards=[0.0] * n,
    )


def test_entity_ppo_update_restores_eval_mode_after_training() -> None:
    """FIX A12: PPO must switch the model to train() for the update (so Dropout is active,
    matching BC) and MUST restore eval() afterward (rollout/inference must stay eval)."""
    from catan_zero.rl.torch_ppo import ppo_update

    samples = _collect_real_samples(6)
    trajectory = _make_trajectory(samples)
    policy = _make_entity_policy()
    policy.model.eval()
    assert not policy.model.training

    ppo_update(
        policy,
        [trajectory],
        learning_rate=1e-3,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.0,
        epochs=1,
        minibatch_size=64,
    )

    assert not policy.model.training, "policy.model must be back in eval() mode after ppo_update"


def test_entity_ppo_update_restores_eval_mode_even_on_empty_trajectory() -> None:
    """The early-return (no samples) path must also leave the model in eval() mode."""
    from catan_zero.rl.torch_ppo import ppo_update, PPOTrajectory

    policy = _make_entity_policy()
    policy.model.eval()
    empty_trajectory = PPOTrajectory(
        samples=[],
        returns=[],
        advantages=[],
        old_log_probs=[],
        old_values=[],
        old_action_probs=[],
        shaped_rewards=[],
    )

    metrics = ppo_update(
        policy,
        [empty_trajectory],
        learning_rate=1e-3,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.0,
        epochs=1,
        minibatch_size=64,
    )

    assert metrics["samples"] == 0.0
    assert not policy.model.training


def test_entity_ppo_update_restores_eval_mode_even_on_exception() -> None:
    """The eval() restoration is a try/finally -- it must fire even if the update body
    raises (e.g. a sample missing entity_features)."""
    from catan_zero.rl.torch_ppo import ppo_update, PPOTrajectory
    from catan_zero.rl.self_play import StepSample

    policy = _make_entity_policy()
    policy.model.eval()
    bad_sample = StepSample(
        observation=np.zeros(3),
        valid_actions=(0, 1),
        action=0,
        player="BLUE",
        entity_features=None,  # missing -> _ppo_update_entity_graph_body raises ValueError
    )
    trajectory = PPOTrajectory(
        samples=[bad_sample],
        returns=[0.0],
        advantages=[0.0],
        old_log_probs=[0.0],
        old_values=[0.0],
        old_action_probs=[np.zeros(1)],
        shaped_rewards=[0.0],
    )

    with pytest.raises(ValueError):
        ppo_update(
            policy,
            [trajectory],
            learning_rate=1e-3,
            clip_ratio=0.2,
            value_coef=0.5,
            entropy_coef=0.0,
            epochs=1,
            minibatch_size=64,
        )

    assert not policy.model.training
