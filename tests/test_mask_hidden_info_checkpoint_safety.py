"""Task #76 safety net: a checkpoint's own recorded mask-hidden-info training
metadata must agree with --public-observation at evaluator construction time,
or the process must fail closed (not silently regenerate the f72 leak).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from catan_zero.rl import entity_token_policy  # noqa: E402
from catan_zero.rl.entity_token_features import (  # noqa: E402
    LEGAL_ACTION_FEATURE_SIZE,
    PLAYER_ACTOR_FLAG_SLOT,
    PLAYER_FEATURE_SIZE,
    PUBLIC_MASK_PLAYER_SLOTS,
)
from catan_zero.rl.entity_token_policy import (  # noqa: E402
    EntityGraphConfig,
    EntityGraphPolicy,
)
from catan_zero.search.neural_rust_mcts import (  # noqa: E402
    EntityGraphRustEvaluator,
    EntityGraphRustEvaluatorConfig,
)

ACTION_SIZE = 8
STATIC_FEATURE_SIZE = 4


def _tiny_policy() -> EntityGraphPolicy:
    config = EntityGraphConfig(
        action_size=ACTION_SIZE,
        static_action_feature_size=STATIC_FEATURE_SIZE,
        hidden_size=16,
        state_layers=1,
        attention_heads=2,
        dropout=0.0,
    )
    static = np.zeros((ACTION_SIZE, STATIC_FEATURE_SIZE), dtype=np.float32)
    return EntityGraphPolicy(config, static, device="cpu")


# ---------------------------------------------------------------------------
# save/load round-trips the mask_hidden_info metadata
# ---------------------------------------------------------------------------


def test_freshly_constructed_policy_defaults_to_not_masked():
    policy = _tiny_policy()
    assert policy.trained_with_masked_hidden_info is False


def test_save_load_round_trips_mask_hidden_info_true(tmp_path):
    policy = _tiny_policy()
    path = tmp_path / "checkpoint.pt"
    policy.save(path, mask_hidden_info=True)
    loaded = EntityGraphPolicy.load(path, device="cpu")
    assert loaded.trained_with_masked_hidden_info is True


def test_save_load_round_trips_mask_hidden_info_false(tmp_path):
    policy = _tiny_policy()
    path = tmp_path / "checkpoint.pt"
    policy.save(path, mask_hidden_info=False)
    loaded = EntityGraphPolicy.load(path, device="cpu")
    assert loaded.trained_with_masked_hidden_info is False


def test_save_defaults_mask_hidden_info_to_false_when_omitted(tmp_path):
    policy = _tiny_policy()
    path = tmp_path / "checkpoint.pt"
    policy.save(path)  # no mask_hidden_info kwarg at all
    loaded = EntityGraphPolicy.load(path, device="cpu")
    assert loaded.trained_with_masked_hidden_info is False


def test_direct_policy_evaluation_honors_checkpoint_public_mask(monkeypatch):
    policy = _tiny_policy()
    policy.trained_with_masked_hidden_info = True
    player_tokens = np.ones((4, PLAYER_FEATURE_SIZE), dtype=np.float32)
    player_tokens[:, PLAYER_ACTOR_FLAG_SLOT] = 0.0
    player_tokens[0, PLAYER_ACTOR_FLAG_SLOT] = 1.0
    original_player_tokens = player_tokens.copy()
    entity = {
        "schema": "entity_tokens_v1",
        "player_tokens": player_tokens,
        "legal_action_tokens": np.zeros(
            (2, LEGAL_ACTION_FEATURE_SIZE), dtype=np.float32
        ),
    }
    captured = {}

    monkeypatch.setattr(
        entity_token_policy,
        "build_entity_token_features",
        lambda *_args, **_kwargs: entity,
    )
    monkeypatch.setattr(
        entity_token_policy,
        "build_action_context_feature_table",
        lambda *_args, **_kwargs: np.zeros(
            (ACTION_SIZE, policy.config.context_action_feature_size),
            dtype=np.float32,
        ),
    )

    def _forward(entity_batch, *_args, **_kwargs):
        captured.update(entity_batch)
        import torch

        return {
            "logits": torch.zeros((1, 2), dtype=torch.float32),
            "value": torch.zeros(1, dtype=torch.float32),
        }

    monkeypatch.setattr(policy, "forward_legal_np", _forward)
    fake_env = type("_Env", (), {"current_player_name": lambda self: "RED"})()

    _outputs, observed, _context = policy._legal_outputs_from_env(  # noqa: SLF001
        fake_env,
        {"current_player": "RED"},
        (0, 1),
    )

    for slot in PUBLIC_MASK_PLAYER_SLOTS:
        assert np.all(observed["player_tokens"][1:, slot] == 0.0)
        assert np.all(captured["player_tokens"][0, 1:, slot] == 0.0)
        assert observed["player_tokens"][0, slot] == 1.0
    assert np.array_equal(entity["player_tokens"], original_player_tokens)


def test_legacy_checkpoint_missing_the_field_loads_as_not_masked(tmp_path):
    """A checkpoint saved before this field existed must fail CLOSED (treated
    as untrained-with-masking), not silently assumed masked."""
    import torch

    policy = _tiny_policy()
    path = tmp_path / "legacy_checkpoint.pt"
    from catan_zero.rl.config_serialization import config_to_dict

    torch.save(
        {
            "policy_type": policy.policy_type,
            "config": config_to_dict(policy.config),
            "action_mask_version": "",
            "static_action_features_sha256": "",
            "static_action_features": policy.static_action_features.detach().cpu(),
            "model": policy.model.state_dict(),
            # deliberately no "mask_hidden_info" key
        },
        path,
    )
    loaded = EntityGraphPolicy.load(path, device="cpu", strict_metadata=False)
    assert loaded.trained_with_masked_hidden_info is False


# ---------------------------------------------------------------------------
# EntityGraphRustEvaluator.__init__ fails closed on a mismatch
# ---------------------------------------------------------------------------


def test_evaluator_raises_when_public_observation_true_but_checkpoint_not_masked():
    policy = _tiny_policy()
    assert policy.trained_with_masked_hidden_info is False
    with pytest.raises(ValueError, match="mismatch"):
        EntityGraphRustEvaluator(
            policy, config=EntityGraphRustEvaluatorConfig(public_observation=True)
        )


def test_evaluator_raises_when_public_observation_false_but_checkpoint_is_masked():
    policy = _tiny_policy()
    policy.trained_with_masked_hidden_info = True
    with pytest.raises(ValueError, match="mismatch"):
        EntityGraphRustEvaluator(
            policy, config=EntityGraphRustEvaluatorConfig(public_observation=False)
        )


def test_evaluator_constructs_fine_when_both_true():
    policy = _tiny_policy()
    policy.trained_with_masked_hidden_info = True
    evaluator = EntityGraphRustEvaluator(
        policy, config=EntityGraphRustEvaluatorConfig(public_observation=True)
    )
    assert evaluator.config.public_observation is True


def test_evaluator_constructs_fine_when_both_false():
    policy = _tiny_policy()
    evaluator = EntityGraphRustEvaluator(
        policy, config=EntityGraphRustEvaluatorConfig(public_observation=False)
    )
    assert evaluator.config.public_observation is False


def test_evaluator_default_config_treats_missing_config_as_public_observation_false():
    """No config passed at all (config=None) -- must not spuriously raise."""
    policy = _tiny_policy()
    evaluator = EntityGraphRustEvaluator(policy, config=None)
    assert evaluator.config.public_observation is False
