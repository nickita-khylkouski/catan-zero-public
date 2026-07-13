from __future__ import annotations

import numpy as np
import pytest

from catan_zero.rl.entity_token_policy import (
    PUBLIC_AWARD_FEATURE_CONTRACT_AUTHORITATIVE,
    PUBLIC_AWARD_FEATURE_CONTRACT_LEGACY_ZERO,
    EntityGraphConfig,
    EntityGraphPolicy,
    _apply_public_award_feature_contract,
)


def test_legacy_bridge_zeroes_only_longest_road_without_mutating_input() -> None:
    players = np.arange(2 * 4 * 31, dtype=np.float32).reshape(2, 4, 31)
    players[..., 12] = 1.0
    original = players.copy()
    batch = {
        "player_tokens": players,
        "global_tokens": np.arange(6, dtype=np.float32).reshape(2, 1, 3),
    }

    bridged = _apply_public_award_feature_contract(
        batch, PUBLIC_AWARD_FEATURE_CONTRACT_LEGACY_ZERO
    )

    assert bridged is not batch
    assert bridged["player_tokens"] is not players
    np.testing.assert_array_equal(players, original)
    np.testing.assert_array_equal(bridged["player_tokens"][..., :12], original[..., :12])
    np.testing.assert_array_equal(
        bridged["player_tokens"][..., 13:], original[..., 13:]
    )
    assert np.count_nonzero(bridged["player_tokens"][..., 12]) == 0
    assert bridged["global_tokens"] is batch["global_tokens"]


def test_legacy_bridge_is_allocation_free_when_slot_already_zero() -> None:
    batch = {"player_tokens": np.zeros((2, 4, 31), dtype=np.float16)}
    assert (
        _apply_public_award_feature_contract(
            batch, PUBLIC_AWARD_FEATURE_CONTRACT_LEGACY_ZERO
        )
        is batch
    )


def test_authoritative_contract_preserves_longest_road() -> None:
    batch = {"player_tokens": np.zeros((2, 4, 31), dtype=np.float16)}
    batch["player_tokens"][0, 1, 12] = 1.0
    assert (
        _apply_public_award_feature_contract(
            batch, PUBLIC_AWARD_FEATURE_CONTRACT_AUTHORITATIVE
        )
        is batch
    )
    assert batch["player_tokens"][0, 1, 12] == 1.0


def test_unknown_contract_fails_closed() -> None:
    with pytest.raises(ValueError, match="unsupported public_award_feature_contract"):
        _apply_public_award_feature_contract(
            {"player_tokens": np.zeros((1, 4, 31), dtype=np.float16)},
            "guess-from-runtime-wheel",
        )


def _tiny_policy() -> EntityGraphPolicy:
    pytest.importorskip("torch")
    config = EntityGraphConfig(
        action_size=8,
        static_action_feature_size=4,
        hidden_size=16,
        state_layers=1,
        attention_heads=2,
        dropout=0.0,
    )
    return EntityGraphPolicy(
        config, np.zeros((8, 4), dtype=np.float32), device="cpu"
    )


def test_checkpoint_contract_roundtrip_and_legacy_default(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    policy = _tiny_policy()
    policy.public_award_feature_contract = (
        PUBLIC_AWARD_FEATURE_CONTRACT_AUTHORITATIVE
    )
    checkpoint = tmp_path / "authoritative.pt"
    policy.save(checkpoint)
    assert (
        EntityGraphPolicy.load(checkpoint, device="cpu").public_award_feature_contract
        == PUBLIC_AWARD_FEATURE_CONTRACT_AUTHORITATIVE
    )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload.pop("public_award_feature_contract")
    legacy = tmp_path / "legacy.pt"
    torch.save(payload, legacy)
    assert (
        EntityGraphPolicy.load(legacy, device="cpu").public_award_feature_contract
        == PUBLIC_AWARD_FEATURE_CONTRACT_LEGACY_ZERO
    )


def test_checkpoint_unknown_contract_is_rejected(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    policy = _tiny_policy()
    checkpoint = tmp_path / "bad.pt"
    policy.save(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["public_award_feature_contract"] = "runtime-decides"
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="unsupported public_award_feature_contract"):
        EntityGraphPolicy.load(checkpoint, device="cpu")
