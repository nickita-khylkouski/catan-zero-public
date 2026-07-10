"""Tests for CAT-100 auxiliary-subgoal target extraction.

The trajectory builder is pure and injected with domain decoders, so its logic
(horizon clamping, VP gain, next-settlement / next-robber lookahead, per-actor
filtering) is tested with fakes. The engine current-state wrappers are exercised
against a real catanatron state where available.
"""

from __future__ import annotations

import math

import pytest

from catan_zero.rl.aux_subgoal_targets import (
    AUX_IGNORE_INDEX,
    AUX_VP_HORIZON,
    robber_hex_id,
    rust_aux_state_from_snapshot,
    rust_hex_id_by_coordinate,
    rust_robber_hex_of_action,
    rust_settlement_node_of_action,
    trajectory_targets,
)
from catan_zero.rl.entity_token_policy import EntityGraphConfig


class _FakeBoard:
    def __init__(self, coord):
        self.robber_coordinate = coord


class _FakeState:
    def __init__(self, coord):
        self.board = _FakeBoard(coord)


def test_robber_hex_id_uses_map_and_sentinel_without_map():
    state = _FakeState((0, 0, 0))
    assert robber_hex_id(state, {(0, 0, 0): 7}) == 7
    # Missing coordinate in map -> ignore sentinel.
    assert robber_hex_id(state, {(1, 1, 1): 3}) == AUX_IGNORE_INDEX
    # No map at all -> ignore sentinel.
    assert robber_hex_id(state, None) == AUX_IGNORE_INDEX


def test_production_aux_horizon_matches_checkpoint_metadata_default():
    config = EntityGraphConfig(action_size=1, static_action_feature_size=1)
    assert AUX_VP_HORIZON == config.aux_vp_horizon == 8


def _trajectory_case():
    # 4 decision rows, players A/B/A/B. States are plain ints used by the fakes.
    states = [0, 1, 2, 3]
    actor_colors = ["A", "B", "A", "B"]
    actions = ["a0", "b1", "a2", "b3"]

    # VP(state, color): player A accrues 1 VP per state index; B stays at 0.
    def vp(state, color):
        return state if color == "A" else 0

    def holds_lr(state, color):
        return color == "A" and state >= 2  # A takes longest road from state 2 on

    def holds_la(state, color):
        return False

    # A plays a settlement at node 5 on its row-2 action; robber to hex 7 on row 0.
    def settlement_node(action):
        return 5 if action == "a2" else None

    def robber_hex(action):
        return 7 if action == "a0" else None

    return dict(
        states=states,
        actor_colors=actor_colors,
        actions=actions,
        horizon=2,
        victory_points_of=vp,
        holds_longest_road_at=holds_lr,
        holds_largest_army_at=holds_la,
        settlement_node_of_action=settlement_node,
        robber_hex_of_action=robber_hex,
    )


def test_trajectory_targets_row0():
    rows = trajectory_targets(**_trajectory_case())
    r0 = rows[0]  # actor A, horizon index min(0+2,3)=2
    assert r0["aux_vp_in_n"] == 2.0  # VP(A,state2)=2 - VP(A,state0)=0
    assert r0["aux_longest_road"] == 1.0  # holds_lr(state2, A) True
    assert r0["aux_largest_army"] == 0.0
    assert r0["aux_next_settlement"] == 5.0  # A's next settlement is at row 2
    assert r0["aux_robber_target"] == 7.0  # A's next robber move is row 0


def test_trajectory_targets_actor_filtering_and_sentinels():
    rows = trajectory_targets(**_trajectory_case())
    # Row 1 is player B, who never plays a settlement or robber move -> sentinels.
    r1 = rows[1]
    assert r1["aux_next_settlement"] == float(AUX_IGNORE_INDEX)
    assert r1["aux_robber_target"] == float(AUX_IGNORE_INDEX)
    assert r1["aux_vp_in_n"] == 0.0  # B has 0 VP throughout
    # Row 3 (B, last row) horizon clamps to itself.
    r3 = rows[3]
    assert r3["aux_vp_in_n"] == 0.0


def test_trajectory_targets_length_validation():
    with pytest.raises(ValueError):
        trajectory_targets(
            states=[0, 1],
            actor_colors=["A"],
            actions=["a", "b"],
            horizon=1,
            victory_points_of=lambda s, c: 0,
            holds_longest_road_at=lambda s, c: False,
            holds_largest_army_at=lambda s, c: False,
            settlement_node_of_action=lambda a: None,
            robber_hex_of_action=lambda a: None,
        )


def test_truncated_trajectory_masks_unobserved_horizon_targets():
    case = _trajectory_case()
    case["final_state"] = 4
    case["trajectory_complete"] = False
    rows = trajectory_targets(**case)

    # Four pre-action states plus the observed post-action final state provide
    # a full two-ply horizon through row 2. Row 3 would need one more unseen
    # transition, so its binary/scalar targets must fail closed rather than
    # pretending the truncated state is a terminal outcome.
    assert rows[2]["aux_vp_in_n"] == 2.0
    assert math.isnan(rows[3]["aux_vp_in_n"])
    assert math.isnan(rows[3]["aux_longest_road"])
    assert math.isnan(rows[3]["aux_largest_army"])


def test_completed_trajectory_clamps_horizon_to_terminal_state():
    case = _trajectory_case()
    case["final_state"] = 4
    case["trajectory_complete"] = True
    rows = trajectory_targets(**case)

    # At a real terminal boundary a shorter remaining horizon is a complete
    # realized outcome, not missing data.
    assert rows[3]["aux_vp_in_n"] == 0.0
    assert math.isfinite(rows[3]["aux_longest_road"])
    assert math.isfinite(rows[3]["aux_largest_army"])


def test_rust_snapshot_and_action_adapters_use_native_ids():
    snapshot = {
        "colors": ["RED", "BLUE"],
        "player_state": [
            {
                "actual_victory_points": 4,
                "has_road": True,
                "has_army": False,
            },
            {
                "actual_victory_points": 3,
                "has_road": False,
                "has_army": True,
            },
        ],
        "tiles": [
            {
                "coordinate": [-2, 0, 2],
                "tile": {"id": 11, "type": "RESOURCE_TILE"},
            },
            # Port ids overlap land ids and must never enter the 19-class hex
            # target map.
            {
                "coordinate": [-3, 0, 3],
                "tile": {"id": 3, "type": "PORT"},
            },
            {
                "coordinate": [1, 0, -1],
                "tile": {"id": 6, "type": "DESERT"},
            },
        ],
    }
    state = rust_aux_state_from_snapshot(snapshot)
    assert state.victory_points("RED") == 4
    assert state.holds_longest_road("RED") is True
    assert state.holds_largest_army("BLUE") is True

    hex_ids = rust_hex_id_by_coordinate(snapshot)
    assert hex_ids == {(-2, 0, 2): 11, (1, 0, -1): 6}
    assert rust_settlement_node_of_action(["RED", "BUILD_SETTLEMENT", 37]) == 37
    assert rust_settlement_node_of_action(["RED", "BUILD_ROAD", [1, 2]]) is None
    assert (
        rust_robber_hex_of_action(
            ["RED", "MOVE_ROBBER", [[-2, 0, 2], "BLUE"]],
            hex_ids,
        )
        == 11
    )
    assert rust_robber_hex_of_action(["RED", "ROLL", None], hex_ids) is None


def test_real_catanatron_current_state_smoke():
    catanatron = pytest.importorskip("catanatron")
    from catanatron.models.player import Color, RandomPlayer

    from catan_zero.rl.aux_subgoal_targets import current_state_targets

    game = catanatron.Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)])
    targets = current_state_targets(game.state, Color.RED)
    # Fresh game: nobody holds the bonuses, RED has the initial VP baseline.
    assert targets["aux_longest_road"] in (0.0, 1.0)
    assert targets["aux_largest_army"] in (0.0, 1.0)
    assert targets["actor_vp"] >= 0.0
    # Robber id unresolved without a coordinate map -> sentinel.
    assert targets["robber_hex"] == float(AUX_IGNORE_INDEX)
