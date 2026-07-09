"""Free auxiliary-subgoal labels from the catanatron engine (CAT-100).

The CAT-100 aux heads (see EntityGraphConfig.aux_subgoal_heads) predict
Catan-native subgoals. Their targets are cheaply derivable from engine state and
the recorded self-play trajectory -- no human labels:

  * ``aux_longest_road`` / ``aux_largest_army`` : does the acting player HOLD the
    bonus at the horizon state (acquisition target). Binary.
  * ``aux_vp_in_n`` : actor victory-point gain over the next ``horizon`` plies.
  * ``aux_next_settlement`` : catanatron ``node_id`` (0-53) of the actor's NEXT
    settlement placement after this row, or -1 if none in the remaining game.
  * ``aux_robber_target`` : hex id (0-18) the actor moves the robber to NEXT, or
    -1 if none.

The engine current-state readouts (longest-road / largest-army holder, actor VP,
robber location) use catanatron's verified ``state_functions`` API. The forward-
looking labels need the played-action sequence decoded into "settlement node"
and "robber hex"; because the exact action encoding is owned by the corpus
generator (and differs between catanatron's Action namedtuple and this repo's
flat action-index space), those decoders are INJECTED as callbacks. That keeps
this module engine-honest and fully unit-testable, and lets the corpus builder
pass its own authoritative decoders.

-1 is the ignore sentinel for the categorical / horizon targets; the train-site
loss (tools/train_bc.py) masks rows whose target is -1 so an unlabeled row never
contributes gradient -- exactly the discipline the value-uncertainty head uses.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence

# Sentinel: "no label for this row" (categorical class id / horizon target).
AUX_IGNORE_INDEX = -1


# --------------------------------------------------------------------------- #
# Engine current-state readouts (catanatron state_functions, verified API).
# --------------------------------------------------------------------------- #
def longest_road_holder(state: Any) -> Any:
    """Color currently holding Longest Road, or None. Wraps state_functions."""
    from catanatron.state_functions import get_longest_road_color

    return get_longest_road_color(state)


def largest_army_holder(state: Any) -> Any:
    """Color currently holding Largest Army, or None.

    ``get_largest_army`` returns ``(color, size)`` in catanatron; we return just
    the color (None when unclaimed), tolerating either return shape.
    """
    from catanatron.state_functions import get_largest_army

    result = get_largest_army(state)
    if isinstance(result, tuple):
        return result[0]
    return result


def actor_victory_points(state: Any, color: Any) -> int:
    """Actual (true) victory points for ``color`` -- includes hidden VP cards."""
    from catanatron.state_functions import get_actual_victory_points

    return int(get_actual_victory_points(state, color))


def robber_hex_id(
    state: Any,
    hex_id_by_coordinate: Optional[Mapping[Any, int]],
) -> int:
    """Hex id (0-18) of the robber, using the featurizer's coordinate->id map.

    The map MUST match the ordering the entity featurizer assigns to hex tokens
    (so the aux_robber_target class space lines up with the net's hex token
    space). When no map is supplied we cannot resolve an id and return the
    ignore sentinel.
    """
    if hex_id_by_coordinate is None:
        return AUX_IGNORE_INDEX
    board = getattr(state, "board", None)
    coordinate = getattr(board, "robber_coordinate", None)
    if coordinate is None:
        return AUX_IGNORE_INDEX
    return int(hex_id_by_coordinate.get(coordinate, AUX_IGNORE_INDEX))


def current_state_targets(
    state: Any,
    actor_color: Any,
    *,
    hex_id_by_coordinate: Optional[Mapping[Any, int]] = None,
) -> dict[str, float]:
    """Point-in-time aux targets derivable from a single engine state.

    Returns the two bonus-holder booleans, the actor's current VP, and the
    robber hex id. Forward-looking targets (vp_in_n, next_settlement,
    robber_target) require the trajectory -- see ``trajectory_targets``.
    """
    return {
        "aux_longest_road": float(longest_road_holder(state) == actor_color),
        "aux_largest_army": float(largest_army_holder(state) == actor_color),
        "actor_vp": float(actor_victory_points(state, actor_color)),
        "robber_hex": float(robber_hex_id(state, hex_id_by_coordinate)),
    }


# --------------------------------------------------------------------------- #
# Forward-looking (trajectory) labels.
# --------------------------------------------------------------------------- #
def _horizon_index(row: int, horizon: int, last: int) -> int:
    """Clamp ``row + horizon`` to the final decision index of the game."""
    return min(row + int(horizon), last)


def trajectory_targets(
    *,
    states: Sequence[Any],
    actor_colors: Sequence[Any],
    actions: Sequence[Any],
    horizon: int,
    victory_points_of: Callable[[Any, Any], int],
    holds_longest_road_at: Callable[[Any, Any], bool],
    holds_largest_army_at: Callable[[Any, Any], bool],
    settlement_node_of_action: Callable[[Any], Optional[int]],
    robber_hex_of_action: Callable[[Any], Optional[int]],
) -> list[dict[str, float]]:
    """Per-decision-row aux targets for one recorded game.

    Parameters are aligned per decision row ``i``:
      ``states[i]``       engine state at decision ``i``,
      ``actor_colors[i]`` color to move at ``i``,
      ``actions[i]``      the action played at ``i``.

    The three ...``_of`` / ``_at`` callables decode/query domain facts; injecting
    them keeps this pure and testable (tests pass fakes; the corpus generator
    passes catanatron-authoritative decoders). ``settlement_node_of_action`` /
    ``robber_hex_of_action`` return None for actions that are not a settlement /
    robber move.

    Targets per row (all floats; -1 == ignore for the last two + vp uses a mask
    flag alongside):
      aux_longest_road / aux_largest_army : bonus held by the row's actor at the
          horizon state (acquisition-by-horizon),
      aux_vp_in_n         : VP(actor, horizon) - VP(actor, now),
      aux_next_settlement : node id of the actor's NEXT settlement after row i,
      aux_robber_target   : hex id of the actor's NEXT robber move after row i.
    """
    n = len(states)
    if not (len(actor_colors) == len(actions) == n):
        raise ValueError(
            "states, actor_colors, actions must be equal length: "
            f"{n}, {len(actor_colors)}, {len(actions)}"
        )
    last = n - 1
    rows: list[dict[str, float]] = []
    for i in range(n):
        actor = actor_colors[i]
        h_idx = _horizon_index(i, horizon, last)
        vp_now = int(victory_points_of(states[i], actor))
        vp_future = int(victory_points_of(states[h_idx], actor))

        next_settlement = AUX_IGNORE_INDEX
        next_robber = AUX_IGNORE_INDEX
        for j in range(i, n):
            if actor_colors[j] != actor:
                continue
            if next_settlement == AUX_IGNORE_INDEX:
                node = settlement_node_of_action(actions[j])
                if node is not None:
                    next_settlement = int(node)
            if next_robber == AUX_IGNORE_INDEX:
                hex_id = robber_hex_of_action(actions[j])
                if hex_id is not None:
                    next_robber = int(hex_id)
            if next_settlement != AUX_IGNORE_INDEX and next_robber != AUX_IGNORE_INDEX:
                break

        rows.append(
            {
                "aux_longest_road": float(bool(holds_longest_road_at(states[h_idx], actor))),
                "aux_largest_army": float(bool(holds_largest_army_at(states[h_idx], actor))),
                "aux_vp_in_n": float(vp_future - vp_now),
                "aux_next_settlement": float(next_settlement),
                "aux_robber_target": float(next_robber),
            }
        )
    return rows
