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

-1 is the ignore sentinel for categorical targets; unavailable binary/scalar
targets use NaN. The train-site loss (tools/train_bc.py) masks both forms so an
unlabeled row never contributes gradient -- exactly the discipline the value-
uncertainty head uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

# Sentinel: "no label for this row" (categorical class id / horizon target).
AUX_IGNORE_INDEX = -1

# The exact shard/model field contract. Keeping it here lets the producer,
# writer, loader, and tests share one spelling without importing tools/train_bc.
AUX_TARGET_KEYS = (
    "aux_longest_road",
    "aux_largest_army",
    "aux_vp_in_n",
    "aux_next_settlement",
    "aux_robber_target",
)

# Must match EntityGraphConfig.aux_vp_horizon. This is target provenance, not a
# training-default switch: production shards always materialize the definition
# the existing CAT-100 heads already advertise.
AUX_VP_HORIZON = 8


@dataclass(frozen=True, slots=True)
class RustAuxState:
    """Small, immutable CAT-100 view of a Rust engine snapshot.

    Holding full ``json_snapshot`` payloads for every ply would retain the
    board, bank, action history, and hands for an entire game in each worker.
    Production labeling only needs three per-player facts, so compact them at
    capture time and keep trajectory memory bounded.
    """

    colors: tuple[str, ...]
    actual_victory_points: tuple[int, ...]
    longest_road: tuple[bool, ...]
    largest_army: tuple[bool, ...]

    def _seat(self, color: Any) -> int:
        name = str(color)
        try:
            return self.colors.index(name)
        except ValueError as error:
            raise KeyError(
                f"color {name!r} is absent from Rust snapshot {self.colors!r}"
            ) from error

    def victory_points(self, color: Any) -> int:
        return int(self.actual_victory_points[self._seat(color)])

    def holds_longest_road(self, color: Any) -> bool:
        return bool(self.longest_road[self._seat(color)])

    def holds_largest_army(self, color: Any) -> bool:
        return bool(self.largest_army[self._seat(color)])


def rust_aux_state_from_snapshot(snapshot: Mapping[str, Any]) -> RustAuxState:
    """Extract the CAT-100 state facts from ``catanatron_rs.json_snapshot``.

    The Rust snapshot's ``colors`` and ``player_state`` arrays are aligned by
    seat. ``has_road``/``has_army`` are the engine-adjudicated bonus holders;
    ``actual_victory_points`` includes hidden VP cards and is safe as a target
    (it is never copied into the public-observation model input).
    """

    colors = tuple(str(color) for color in snapshot.get("colors", ()))
    players = tuple(snapshot.get("player_state", ()))
    if len(colors) != len(players):
        raise ValueError(
            "Rust snapshot colors/player_state length mismatch: "
            f"{len(colors)} != {len(players)}"
        )
    actual_vps: list[int] = []
    longest_road: list[bool] = []
    largest_army: list[bool] = []
    for player in players:
        state = player if isinstance(player, Mapping) else {}
        actual_vps.append(
            int(state.get("actual_victory_points", state.get("victory_points", 0)) or 0)
        )
        # Native catanatron_rs names are has_road/has_army. Tolerate the
        # descriptive aliases used by Python-engine observation payloads so
        # replay/conversion callers can reuse the adapter safely.
        longest_road.append(
            bool(state.get("has_road", state.get("has_longest_road", False)))
        )
        largest_army.append(
            bool(state.get("has_army", state.get("has_largest_army", False)))
        )
    return RustAuxState(
        colors=colors,
        actual_victory_points=tuple(actual_vps),
        longest_road=tuple(longest_road),
        largest_army=tuple(largest_army),
    )


def rust_hex_id_by_coordinate(
    snapshot: Mapping[str, Any],
) -> dict[tuple[int, int, int], int]:
    """Return the Rust board's coordinate -> entity-hex-id mapping.

    ``json_snapshot['tiles']`` contains land, ports, and water. Port ids overlap
    the 0..18 land ids, so filtering by tile type is essential; accepting every
    numeric id would silently train robber targets against the wrong token.
    """

    result: dict[tuple[int, int, int], int] = {}
    for entry in snapshot.get("tiles", ()):
        if not isinstance(entry, Mapping):
            continue
        tile = entry.get("tile")
        if not isinstance(tile, Mapping):
            continue
        tile_type = str(tile.get("type", "")).upper()
        if tile_type != "RESOURCE_TILE" and "DESERT" not in tile_type:
            continue
        tile_id = tile.get("id")
        coordinate = entry.get("coordinate")
        if not isinstance(tile_id, int) or not 0 <= tile_id < 19:
            continue
        if not isinstance(coordinate, (list, tuple)) or len(coordinate) != 3:
            continue
        result[tuple(int(value) for value in coordinate)] = int(tile_id)
    return result


def rust_settlement_node_of_action(action: Any) -> Optional[int]:
    """Decode a native Rust ``BUILD_SETTLEMENT`` action's node id."""

    if not isinstance(action, (list, tuple)) or len(action) < 3:
        return None
    if str(action[1]) != "BUILD_SETTLEMENT":
        return None
    value = action[2]
    return int(value) if isinstance(value, int) and 0 <= int(value) < 54 else None


def rust_robber_hex_of_action(
    action: Any,
    hex_id_by_coordinate: Mapping[tuple[int, int, int], int],
) -> Optional[int]:
    """Decode a native Rust ``MOVE_ROBBER`` action to entity hex class 0..18."""

    if not isinstance(action, (list, tuple)) or len(action) < 3:
        return None
    if str(action[1]) != "MOVE_ROBBER":
        return None
    value = action[2]
    if not isinstance(value, (list, tuple)) or not value:
        return None
    coordinate = value[0]
    if not isinstance(coordinate, (list, tuple)) or len(coordinate) != 3:
        return None
    return hex_id_by_coordinate.get(tuple(int(component) for component in coordinate))


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
    """Clamp ``row + horizon`` to the final observed state index."""
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
    final_state: Any | None = None,
    trajectory_complete: bool = True,
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

    ``final_state`` is the observed state after ``actions[-1]``. Production
    passes it so a terminal move's VP/road/army change is not lost merely
    because no next decision exists. When ``trajectory_complete`` is False
    (decision-cap truncation), rows without a fully observed horizon get NaN
    binary/scalar labels; train_bc's finite mask then excludes them. Positive
    settlement/robber actions already observed remain valid, while their -1
    sentinel continues to mean "no usable categorical label".

    Targets per row (all floats; -1 == ignore for the last two):
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
    state_sequence: list[Any] = list(states)
    if final_state is not None:
        state_sequence.append(final_state)
    last = len(state_sequence) - 1
    rows: list[dict[str, float]] = []
    for i in range(n):
        actor = actor_colors[i]
        h_idx = _horizon_index(i, horizon, last)
        horizon_observed = i + int(horizon) <= last
        horizon_valid = bool(trajectory_complete or horizon_observed)
        if horizon_valid:
            vp_now = int(victory_points_of(state_sequence[i], actor))
            vp_future = int(victory_points_of(state_sequence[h_idx], actor))
            longest_road = float(
                bool(holds_longest_road_at(state_sequence[h_idx], actor))
            )
            largest_army = float(
                bool(holds_largest_army_at(state_sequence[h_idx], actor))
            )
            vp_in_n = float(vp_future - vp_now)
        else:
            longest_road = float("nan")
            largest_army = float("nan")
            vp_in_n = float("nan")

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
                "aux_longest_road": longest_road,
                "aux_largest_army": largest_army,
                "aux_vp_in_n": vp_in_n,
                "aux_next_settlement": float(next_settlement),
                "aux_robber_target": float(next_robber),
            }
        )
    return rows
