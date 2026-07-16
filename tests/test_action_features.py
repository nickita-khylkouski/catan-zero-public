from __future__ import annotations

import pytest

from catan_zero.rl.action_features import build_action_context_feature_table
from catan_zero.rl.entity_token_features import _node_pips_by_resource
from catan_zero.rl.multiagent_env import ColonistMultiAgentConfig, ColonistMultiAgentEnv


def test_live_python_settlement_context_encodes_pips_not_probability_divided_by_18():
    env = ColonistMultiAgentEnv(ColonistMultiAgentConfig(players=2, vps_to_win=10))
    try:
        _observations, info = env.reset(seed=600001)
        table = build_action_context_feature_table(env, info)
        settlements = [
            action
            for action in info["structured_legal_actions"]
            if action["action_type"] == "BUILD_SETTLEMENT"
        ]

        assert settlements
        encoded = []
        for action in settlements:
            node = int(action["args"]["node"])
            production = env.game.state.board.map.node_production[node]
            expected = sum(_node_pips_by_resource(production)) / 18.0
            actual = float(table[int(action["index"]), 2])
            assert actual == pytest.approx(expected)
            encoded.append(actual)

        # The old implementation divided the already-normalized probability
        # by 18 and topped out near 0.02 on this board.
        assert max(encoded) > 0.5
    finally:
        env.close()
