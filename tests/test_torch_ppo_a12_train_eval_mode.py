from __future__ import annotations

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


def test_entity_ppo_recomputes_actor_likelihoods_in_eval_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Identical actor/learner weights must start PPO at ratio one."""
    import torch
    from torch import nn

    from catan_zero.rl import torch_ppo

    samples = _collect_real_samples(12)
    trajectory = _make_trajectory(samples)
    policy = _make_entity_policy()
    dropout_modules = [
        module for module in policy.model.modules() if isinstance(module, nn.Dropout)
    ]
    assert dropout_modules
    for module in dropout_modules:
        module.p = 1.0
    policy.model.eval()

    old_log_probs: list[float] = []
    old_values: list[float] = []
    with torch.no_grad():
        for sample in samples:
            outputs = torch_ppo._entity_graph_outputs(  # noqa: SLF001
                policy,
                [sample],
                return_q=False,
            )
            logits = torch_ppo._behavior_policy_logits(  # noqa: SLF001
                outputs["logits"],
                1.0,
            )
            action_column = sample.valid_actions.index(sample.action)
            old_log_probs.append(
                float(
                    torch.distributions.Categorical(logits=logits)
                    .log_prob(
                        torch.as_tensor(
                            [action_column],
                            dtype=torch.long,
                            device=policy.device,
                        )
                    )
                    .item()
                )
            )
            old_values.append(float(outputs["value"].item()))
    trajectory.old_log_probs = old_log_probs
    trajectory.old_values = old_values
    old_log_prob_by_sample = {
        id(sample): old_log_prob
        for sample, old_log_prob in zip(samples, old_log_probs, strict=True)
    }

    observed_model_modes: list[bool] = []
    observed_dropout_modes: list[bool] = []
    observed_ratios: list[float] = []
    original_outputs = torch_ppo._entity_graph_outputs  # noqa: SLF001

    def observed_outputs(*args, **kwargs):
        observed_model_modes.append(bool(policy.model.training))
        outputs = original_outputs(*args, **kwargs)
        batch_samples = args[1]
        logits = torch_ppo._behavior_policy_logits(  # noqa: SLF001
            outputs["logits"],
            1.0,
            valid_mask=torch_ppo._entity_behavior_valid_mask(  # noqa: SLF001
                batch_samples,
                outputs["logits"],
            ),
        )
        actions = torch.as_tensor(
            [
                sample.valid_actions.index(sample.action)
                for sample in batch_samples
            ],
            dtype=torch.long,
            device=policy.device,
        )
        current = torch.distributions.Categorical(logits=logits).log_prob(actions)
        observed_ratios.extend(
            torch.exp(
                current
                - torch.as_tensor(
                    [
                        old_log_prob_by_sample[id(sample)]
                        for sample in batch_samples
                    ],
                    dtype=torch.float32,
                    device=policy.device,
                )
            )
            .detach()
            .cpu()
            .tolist()
        )
        return outputs

    hooks = [
        module.register_forward_hook(
            lambda hooked, _inputs, _output: observed_dropout_modes.append(
                bool(hooked.training)
            )
        )
        for module in dropout_modules
    ]
    monkeypatch.setattr(torch_ppo, "_entity_graph_outputs", observed_outputs)
    try:
        metrics = torch_ppo.ppo_update(
            policy,
            [trajectory],
            learning_rate=0.0,
            clip_ratio=0.1,
            value_coef=0.5,
            entropy_coef=0.0,
            epochs=1,
            minibatch_size=len(samples),
        )
    finally:
        for hook in hooks:
            hook.remove()

    assert observed_model_modes and not any(observed_model_modes)
    assert observed_dropout_modes and not any(observed_dropout_modes)
    np.testing.assert_allclose(observed_ratios, np.ones(len(samples)), atol=1e-6)
    assert metrics["approx_kl"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["clip_fraction"] == 0.0
    assert not policy.model.training


def test_entity_actor_and_learner_share_temperature_and_saturation() -> None:
    import torch

    from catan_zero.rl.torch_ppo import _behavior_policy_logits

    logits = torch.as_tensor([[-200.0, -20.0, 0.0, 20.0, 200.0]])

    torch.testing.assert_close(
        _behavior_policy_logits(logits, 0.5),
        torch.as_tensor([[-50.0, -40.0, 0.0, 40.0, 50.0]]),
    )
    torch.testing.assert_close(
        _behavior_policy_logits(logits, 1.0),
        torch.as_tensor([[-50.0, -20.0, 0.0, 20.0, 50.0]]),
    )
    padded = torch.as_tensor(
        [[-200.0, -200.0, -1.0e9], [-200.0, -200.0, -200.0]]
    )
    valid = torch.as_tensor(
        [[True, True, False], [True, True, True]]
    )
    torch.testing.assert_close(
        _behavior_policy_logits(padded, 1.0, valid_mask=valid),
        torch.as_tensor(
            [[-50.0, -50.0, -1.0e9], [-50.0, -50.0, -50.0]]
        ),
    )


def test_fresh_entity_actor_to_learner_starts_at_ratio_one() -> None:
    from torch import nn

    from catan_zero.rl.self_play import make_env_config
    from catan_zero.rl.torch_ppo import collect_ppo_episode, ppo_update

    policy = _make_entity_policy()
    for module in policy.model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 1.0
    assert policy.model.training
    config = make_env_config(vps_to_win=3)
    trajectory = collect_ppo_episode(
        policy,
        {},
        seed=17,
        config=config,
        max_decisions=12,
        rng=np.random.default_rng(17),
        training_seats={"BLUE", "RED", "ORANGE", "WHITE"},
        action_temperature=1.0,
    )

    assert trajectory.samples
    assert not policy.model.training
    metrics = ppo_update(
        policy,
        [trajectory],
        learning_rate=0.0,
        clip_ratio=0.1,
        value_coef=0.5,
        entropy_coef=0.0,
        epochs=1,
        minibatch_size=len(trajectory.samples),
    )

    assert metrics["approx_kl"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["clip_fraction"] == 0.0


def test_entity_truncation_drains_to_learner_decision_boundary() -> None:
    from types import SimpleNamespace

    from catan_zero.rl.entity_token_policy import EntityGraphPolicy
    from catan_zero.rl.self_play import RandomPolicy, make_env_config
    from catan_zero.rl.torch_ppo import collect_ppo_episode
    from tools.ppo_distributed_learner import (
        _recompute_target_logp_and_values_batched,
    )

    config = make_env_config(players=2, vps_to_win=3)
    policy = EntityGraphPolicy.create(
        env_config=config,
        hidden_size=16,
        state_layers=1,
        attention_heads=2,
        seed=0,
    )
    trajectory = collect_ppo_episode(
        policy,
        {"RED": RandomPolicy()},
        seed=1,
        config=config,
        max_decisions=2,
        rng=np.random.default_rng(2),
        training_seats={"BLUE"},
    )

    assert trajectory.truncated is True
    assert len(trajectory.samples) == 2
    assert trajectory.bootstrap_sample is not None
    assert trajectory.bootstrap_sample.player == "BLUE"
    assert trajectory.bootstrap_value != 0.0
    _, learner_values = _recompute_target_logp_and_values_batched(
        policy,
        [SimpleNamespace(samples=[trajectory.bootstrap_sample])],
    )
    assert learner_values[0] == pytest.approx(
        trajectory.bootstrap_value,
        abs=1e-6,
    )


def test_entity_ppo_update_restores_eval_mode_after_training() -> None:
    """Autograd stays active while the likelihood model remains in eval mode."""
    import torch

    from catan_zero.rl.torch_ppo import ppo_update

    samples = _collect_real_samples(6)
    trajectory = _make_trajectory(samples)
    policy = _make_entity_policy()
    before = {
        name: value.detach().clone()
        for name, value in policy.model.named_parameters()
    }
    policy.model.eval()

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

    assert any(
        not torch.equal(before[name], value.detach())
        for name, value in policy.model.named_parameters()
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
