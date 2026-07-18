from __future__ import annotations

import numpy as np


def _fixed_logit_policy():
    import torch

    from catan_zero.rl.entity_token_policy import EntityGraphPolicy

    policy = object.__new__(EntityGraphPolicy)
    policy.model = torch.nn.Identity()

    def outputs(_env, _info, valid_actions, *, return_q):
        assert return_q is True
        width = len(valid_actions)
        return (
            {
                "logits": torch.zeros((1, width), dtype=torch.float32),
                "value": torch.zeros(1, dtype=torch.float32),
                "q_values": torch.zeros((1, width), dtype=torch.float32),
            },
            {},
            None,
        )

    policy._legal_outputs_from_env = outputs
    return policy


def _sample(policy, rng: np.random.Generator) -> int:
    action, *_rest = policy.sample_action_value_q_from_env(
        None,
        {"valid_actions": (10, 11, 12, 13)},
        rng,
        training=True,
    )
    return int(action)


def test_entity_actor_sampling_uses_the_caller_game_rng() -> None:
    import torch

    policy = _fixed_logit_policy()

    torch.manual_seed(123)
    action_a = _sample(policy, np.random.default_rng(1))
    torch.manual_seed(123)
    action_b = _sample(policy, np.random.default_rng(4))

    assert action_a == 12
    assert action_b == 13


def test_entity_actor_game_stream_is_independent_of_completed_game_draws() -> None:
    import torch

    policy = _fixed_logit_policy()

    def game_actions(game_seed: int) -> list[int]:
        rng = np.random.default_rng(game_seed)
        return [_sample(policy, rng) for _ in range(8)]

    # An uninterrupted worker consumes arbitrary torch randomness and earlier games
    # before reaching this game.
    torch.manual_seed(77)
    _ = torch.rand(257)
    _ = game_actions(100)
    _ = game_actions(101)
    uninterrupted = game_actions(102)

    # A resumed worker reloads the policy, skips completed shards, and starts the
    # missing game directly. Its actions must be byte-for-byte replayable from the
    # per-game seed, regardless of the reset process-global torch stream.
    torch.manual_seed(0)
    resumed = game_actions(102)

    assert resumed == uninterrupted


def test_entity_actor_sampling_does_not_advance_global_torch_rng() -> None:
    import torch

    policy = _fixed_logit_policy()
    torch.manual_seed(321)
    before = torch.random.get_rng_state().clone()

    _sample(policy, np.random.default_rng(5))

    assert torch.equal(torch.random.get_rng_state(), before)
