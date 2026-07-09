"""Tests for CAT-100 auxiliary-subgoal target extraction.

The trajectory builder is pure and injected with domain decoders, so its logic
(horizon clamping, VP gain, next-settlement / next-robber lookahead, per-actor
filtering) is tested with fakes. The engine current-state wrappers are exercised
against a real catanatron state where available.
"""

from __future__ import annotations

import pytest

from catan_zero.rl.aux_subgoal_targets import (
    AUX_IGNORE_INDEX,
    robber_hex_id,
    trajectory_targets,
)


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
