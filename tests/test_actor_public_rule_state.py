from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from catan_zero.rl.entity_feature_adapter import (
    RUST_ENTITY_ADAPTER_V3,
    RUST_ENTITY_ADAPTER_V4,
)
from catan_zero.rl.entity_token_features import (
    GLOBAL_FEATURE_SIZE,
    PUBLIC_RULE_STATE_FEATURE_SLICE,
    _global_tokens,
)
from catan_zero.rl.multiagent_env import ColonistMultiAgentEnv


class _Env:
    pass


def _python_player_payload_fixture() -> SimpleNamespace:
    cards = (
        "KNIGHT",
        "YEAR_OF_PLENTY",
        "MONOPOLY",
        "ROAD_BUILDING",
        "VICTORY_POINT",
    )
    resources = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
    player_state: dict[str, object] = {}
    for color in ("BLUE", "RED"):
        player_state.update(
            {
                f"{color}_VICTORY_POINTS": 2,
                f"{color}_ACTUAL_VICTORY_POINTS": 2,
                f"{color}_HAS_ARMY": False,
                f"{color}_HAS_ROAD": False,
                f"{color}_ROADS_AVAILABLE": 15,
                f"{color}_SETTLEMENTS_AVAILABLE": 5,
                f"{color}_CITIES_AVAILABLE": 4,
                f"{color}_HAS_ROLLED": True,
                f"{color}_LONGEST_ROAD_LENGTH": 0,
                f"{color}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN": False,
            }
        )
        for resource in resources:
            player_state[f"{color}_{resource}_IN_HAND"] = 0
        for card in cards:
            player_state[f"{color}_{card}_IN_HAND"] = 0
            if card != "VICTORY_POINT":
                player_state[f"{color}_{card}_OWNED_AT_START"] = 0
                player_state[f"{color}_PLAYED_{card}"] = 0

    fake = SimpleNamespace(
        game=SimpleNamespace(state=SimpleNamespace(player_state=player_state)),
        player_colors=("BLUE", "RED"),
        DEVELOPMENT_CARDS=cards,
        RESOURCES=resources,
        VICTORY_POINT="VICTORY_POINT",
        player_key=lambda _state, color: color,
        player_num_resource_cards=lambda _state, _color: 0,
        player_num_dev_cards=lambda _state, color: sum(
            int(player_state[f"{color}_{card}_IN_HAND"]) for card in cards
        ),
    )
    return fake


def _payload() -> dict:
    return {
        "current_prompt": "PLAY_TURN",
        "current_player": "BLUE",
        "players": {
            "BLUE": {
                "has_played_development_card_in_turn": True,
                "playable_development_cards": {
                    "KNIGHT": 2,
                    "YEAR_OF_PLENTY": 1,
                    "MONOPOLY": 0,
                    "ROAD_BUILDING": 1,
                },
            },
            "RED": {},
        },
        "is_road_building": True,
        "free_roads_available": 1,
        "current_discard_count": 3,
        "legal_actions": (),
        "bank": {},
    }


def test_python_observation_payload_matches_public_rule_state_feature_contract():
    fake = _python_player_payload_fixture()
    state = fake.game.state.player_state
    state["BLUE_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"] = True
    state["BLUE_KNIGHT_IN_HAND"] = 2
    state["BLUE_KNIGHT_OWNED_AT_START"] = 2
    state["BLUE_MONOPOLY_IN_HAND"] = 1
    state["BLUE_MONOPOLY_OWNED_AT_START"] = 0

    players = ColonistMultiAgentEnv._player_payloads(fake, "BLUE")
    assert players["BLUE"]["has_played_development_card_in_turn"] is True
    assert players["BLUE"]["playable_development_cards"] == {
        "KNIGHT": 2,
        "YEAR_OF_PLENTY": 0,
        "MONOPOLY": 0,
        "ROAD_BUILDING": 0,
    }
    assert "has_played_development_card_in_turn" not in players["RED"]
    assert "playable_development_cards" not in players["RED"]

    payload = _payload()
    payload["players"] = players
    encoded = _global_tokens(
        _Env(), payload, "BLUE", encode_actor_public_rule_state=True
    )
    np.testing.assert_allclose(
        encoded[0, PUBLIC_RULE_STATE_FEATURE_SLICE][[0, 4, 6]],
        np.asarray([1.0, 0.4, 0.0], dtype=np.float32),
        rtol=0,
        atol=3e-4,
    )


def test_adapter_v4_exposes_current_actor_rule_state_without_reinterpreting_v3():
    payload = _payload()
    legacy = _global_tokens(
        _Env(), payload, "BLUE", encode_actor_public_rule_state=False
    )
    repaired = _global_tokens(
        _Env(), payload, "BLUE", encode_actor_public_rule_state=True
    )

    assert legacy.shape == repaired.shape == (1, GLOBAL_FEATURE_SIZE)
    assert np.all(legacy[:, PUBLIC_RULE_STATE_FEATURE_SLICE] == 0.0)
    np.testing.assert_allclose(
        repaired[0, PUBLIC_RULE_STATE_FEATURE_SLICE],
        np.asarray([1.0, 1.0, 0.5, 0.3, 0.4, 0.2, 0.0, 0.2]),
        rtol=0,
        atol=3e-4,
    )
    assert RUST_ENTITY_ADAPTER_V3 != RUST_ENTITY_ADAPTER_V4


def test_public_rule_state_residual_is_function_preserving_then_trainable():
    torch = pytest.importorskip("torch")

    from catan_zero.rl.action_features import CONTEXT_ACTION_FEATURE_SIZE
    from catan_zero.rl.entity_token_features import (
        EDGE_FEATURE_SIZE,
        EVENT_FEATURE_SIZE,
        HEX_FEATURE_SIZE,
        LEGAL_ACTION_FEATURE_SIZE,
        PLAYER_FEATURE_SIZE,
        VERTEX_FEATURE_SIZE,
    )
    from catan_zero.rl.entity_token_policy import EntityGraphConfig, EntityGraphNet

    def config(enabled: bool) -> EntityGraphConfig:
        return EntityGraphConfig(
            action_size=8,
            static_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
            context_action_feature_size=CONTEXT_ACTION_FEATURE_SIZE,
            legal_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
            hidden_size=16,
            state_layers=1,
            attention_heads=2,
            dropout=0.0,
            public_rule_state_features=enabled,
        )

    torch.manual_seed(7)
    base = EntityGraphNet(config(False))
    upgraded = EntityGraphNet(config(True))
    missing, unexpected = upgraded.load_state_dict(base.state_dict(), strict=False)
    assert unexpected == []
    assert missing == ["public_rule_state_residual.weight"]

    batch_size = 2
    actions = 3
    shapes = {
        "hex": (19, HEX_FEATURE_SIZE),
        "vertex": (54, VERTEX_FEATURE_SIZE),
        "edge": (72, EDGE_FEATURE_SIZE),
        "player": (4, PLAYER_FEATURE_SIZE),
        "global": (1, GLOBAL_FEATURE_SIZE),
        "event": (64, EVENT_FEATURE_SIZE),
    }
    batch = {}
    for name, (count, width) in shapes.items():
        batch[f"{name}_tokens"] = torch.randn(batch_size, count, width)
        if name != "global":
            batch[f"{name}_mask"] = torch.ones(
                batch_size, count, dtype=torch.bool
            )
    batch["global_tokens"][:, :, PUBLIC_RULE_STATE_FEATURE_SLICE] = 1.0
    batch["legal_action_tokens"] = torch.randn(
        batch_size, actions, LEGAL_ACTION_FEATURE_SIZE
    )
    batch["legal_action_context"] = torch.randn(
        batch_size, actions, CONTEXT_ACTION_FEATURE_SIZE
    )
    batch["legal_action_target_ids"] = -torch.ones(
        batch_size, actions, 4, dtype=torch.long
    )
    legacy_batch = {key: value.clone() for key, value in batch.items()}
    legacy_batch["global_tokens"][:, :, PUBLIC_RULE_STATE_FEATURE_SLICE] = 0.0

    base.eval()
    upgraded.eval()
    base_out = base(legacy_batch)
    upgraded_out = upgraded(batch)
    assert torch.equal(base_out["logits"], upgraded_out["logits"])
    assert torch.equal(base_out["value"], upgraded_out["value"])

    upgraded.public_rule_state_residual.weight.data.zero_()
    upgraded.public_rule_state_residual.weight.data[0, 0] = 0.1
    changed = upgraded(batch)
    assert not torch.equal(upgraded_out["value"], changed["value"])


def test_public_rule_state_policy_fails_closed_without_adapter_v4():
    pytest.importorskip("torch")

    from catan_zero.rl.entity_token_policy import EntityGraphConfig, EntityGraphPolicy

    config = EntityGraphConfig(
        action_size=8,
        static_action_feature_size=4,
        hidden_size=16,
        state_layers=1,
        attention_heads=2,
        dropout=0.0,
        public_rule_state_features=True,
    )
    with pytest.raises(ValueError, match="requires entity adapter v4"):
        EntityGraphPolicy(
            config,
            np.zeros((8, 4), dtype=np.float32),
            device="cpu",
            entity_feature_adapter_version=RUST_ENTITY_ADAPTER_V3,
        )
