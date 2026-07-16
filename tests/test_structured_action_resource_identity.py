from types import SimpleNamespace

import numpy as np
import pytest

from catan_zero.rl.entity_feature_adapter import (
    RUST_ENTITY_ADAPTER_V2,
    RUST_ENTITY_ADAPTER_V3,
    RUST_ENTITY_ADAPTER_V4,
    RUST_ENTITY_ADAPTER_V5,
)
from catan_zero.rl.entity_token_features import _legal_action_tokens
from catan_zero.search.neural_rust_mcts import _structured_action


def _tokens(actions: list[dict]) -> np.ndarray:
    env = SimpleNamespace(action_space=SimpleNamespace(n=567))
    payload = {
        "structured_legal_actions": actions,
        "current_prompt": "PLAY_TURN",
        "trade_panel": {},
    }
    return _legal_action_tokens(
        env,
        payload,
        {},
        encode_structured_action_resources=(
            actions[0]["adapter_version"]
            in {
                RUST_ENTITY_ADAPTER_V3,
                RUST_ENTITY_ADAPTER_V4,
                RUST_ENTITY_ADAPTER_V5,
            }
        ),
    )


def _action(action_id: int, raw: list, version: str) -> dict:
    action = _structured_action(
        action_id,
        raw,
        entity_feature_adapter_version=version,
    )
    action["adapter_version"] = version
    return action


def test_v2_preserves_lossy_resource_action_contract() -> None:
    plenty = _action(
        311,
        ["RED", "PLAY_YEAR_OF_PLENTY", ["WOOD", "ORE"]],
        RUST_ENTITY_ADAPTER_V2,
    )
    monopoly = _action(
        305,
        ["RED", "PLAY_MONOPOLY", "WOOD"],
        RUST_ENTITY_ADAPTER_V2,
    )

    assert plenty["args"] == {}
    assert monopoly["args"] == {"resource": "wood"}
    tokens = _tokens([plenty, monopoly])
    np.testing.assert_array_equal(tokens[:, 31:36], np.zeros((2, 5)))
    assert tokens[0, 25] == 1.0  # YOP historically had target kind "none".
    assert tokens[1, 30] == 1.0  # Singular identity existed only as a kind.


@pytest.mark.parametrize(
    "adapter_version",
    [
        RUST_ENTITY_ADAPTER_V3,
        RUST_ENTITY_ADAPTER_V4,
        RUST_ENTITY_ADAPTER_V5,
    ],
)
def test_modern_adapters_encode_yop_bundle_and_singular_resource_identity(
    adapter_version: str,
) -> None:
    actions = [
        _action(
            311,
            ["RED", "PLAY_YEAR_OF_PLENTY", ["WOOD", "ORE"]],
            adapter_version,
        ),
        _action(
            305,
            ["RED", "PLAY_MONOPOLY", "WOOD"],
            adapter_version,
        ),
        _action(
            309,
            ["RED", "PLAY_MONOPOLY", "ORE"],
            adapter_version,
        ),
        _action(
            185,
            ["RED", "DISCARD_RESOURCE", "ORE"],
            adapter_version,
        ),
    ]

    assert actions[0]["args"] == {"resources": ["wood", "ore"]}
    tokens = _tokens(actions)
    np.testing.assert_array_equal(
        tokens[:, 31:36],
        np.asarray(
            [
                [0.5, 0.0, 0.0, 0.0, 0.5],
                [0.5, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.5],
                [0.0, 0.0, 0.0, 0.0, 0.5],
            ],
            dtype=np.float16,
        ),
    )
    np.testing.assert_array_equal(tokens[:, 30], np.ones(4))
    assert not np.array_equal(tokens[1], tokens[2])
