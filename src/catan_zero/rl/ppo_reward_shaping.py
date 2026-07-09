"""Potential-based reward shaping for Catan PPO.

This module implements *potential-based* reward shaping in the sense of
Ng, Harada & Russell (1999), "Policy invariance under reward transformations:
Theory and application to reward shaping".

The shaping reward added to a transition ``s -> s'`` has the form::

    F(s, s') = gamma * Phi(s') - Phi(s)

where ``Phi`` is an arbitrary *potential function* over states. The key
theoretical guarantee is that adding ``F`` to the environment reward does
**not** change the set of optimal policies: over a full episode the shaping
contribution telescopes to::

    sum_{t=0}^{T-1} gamma^t * F(s_t, s_{t+1})
        = gamma^T * Phi(s_T) - Phi(s_0)

i.e. it depends only on the (discounted) terminal potential and the initial
potential, not on the path taken. This makes it a safe, "free" densification
of an otherwise sparse win/lose signal.

The potential ``Phi`` here is built from **public** Catan progress only:
victory points, settlements, cities, roads, longest-road bonus, largest-army
bonus, and an optional cheap production/port proxy from the public board.
No opponent hidden information (resource hands, dev-card hands) is used, so
this is sound for self-play where each seat only legitimately observes the
public board.

Integration
-----------
This is intentionally a thin, standalone set of functions. It is designed to
compose with the value-delta shaping already present in
``torch_ppo.collect_ppo_episode`` (``value_shaping_coef`` /
``value_shaping_scale`` / ``value_shaping_opponent_penalty``): you can add the
potential-based ``F`` to the per-step reward in the same loop. Because ``F``
is policy-invariant by construction, mixing it with other shaping terms keeps
the *potential-based* part provably neutral on the optimal policy.

Public-state shape
------------------
``catan_potential`` consumes a plain ``dict`` so it can be unit-tested without
an env. ``extract_public_state`` produces that dict from a live
``ColonistMultiAgentEnv``. The dict layout mirrors the env's public surface
(see ``ColonistMultiAgentEnv._player_payloads`` and ``_board_payload`` in
``multiagent_env.py``)::

    {
        "players": {
            "RED": {
                "public_victory_points": int,
                "settlements_built": int,      # 0..5
                "cities_built": int,           # 0..4
                "roads_built": int,            # 0..15
                "has_longest_road": bool,
                "longest_road_length": int,
                "has_largest_army": bool,
                "knights_played": int,
                "production_proxy": float,     # optional, >= 0
            },
            ...
        }
    }

Only the ``players[seat]`` entry is read by ``catan_potential``; the rest is
ignored, which keeps the function robust to extra public fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

__all__ = [
    "ShapingWeights",
    "catan_potential",
    "shaped_increment",
    "extract_public_state",
]

# Standard Catan piece allotments. Used to convert the env's "pieces remaining"
# counts into "pieces built" counts when reading directly from player payloads.
MAX_SETTLEMENTS = 5
MAX_CITIES = 4
MAX_ROADS = 15


@dataclass
class ShapingWeights:
    """Relative weights for each public progress term in ``Phi``.

    Defaults are reasonable starting points and are meant to be tuned. The
    ``vp`` term dominates (it is the actual objective); the building/road/bonus
    terms provide a denser gradient toward the kinds of progress that tend to
    produce victory points.
    """

    vp: float = 1.0
    settlement: float = 0.25
    city: float = 0.5
    road: float = 0.05
    longest_road: float = 0.3
    largest_army: float = 0.3
    production: float = 0.1


def _seat_features(public_state: Mapping[str, Any], seat: str) -> Mapping[str, Any]:
    """Return the public feature dict for ``seat`` from ``public_state``.

    Accepts either ``{"players": {seat: {...}}}`` (the canonical layout produced
    by :func:`extract_public_state`) or a flat ``{seat: {...}}`` mapping, for
    convenience in tests.
    """
    if "players" in public_state and isinstance(public_state["players"], Mapping):
        players = public_state["players"]
    else:
        players = public_state
    if seat not in players:
        raise KeyError(f"seat {seat!r} not present in public_state players")
    return players[seat]


def catan_potential(
    public_state: Mapping[str, Any],
    *,
    seat: str,
    vps_to_win: int = 10,
    weights: ShapingWeights = ShapingWeights(),
) -> float:
    """Potential ``Phi(s)`` for ``seat`` from public progress only.

    The dominant victory-point term is normalized by ``vps_to_win`` so that the
    potential stays on a comparable scale across curriculum stages (e.g. a
    6-VP early-curriculum game vs a full 10-VP game): a seat that is "halfway to
    winning" contributes roughly the same VP potential in both. The remaining
    structural terms (settlements, cities, roads, bonuses, production proxy) are
    each normalized by their own public maxima so that ``Phi`` lands in roughly
    ``[0, 1]`` for the default weights.

    Reads ONLY public fields. Hidden-hand fields (exact opponent resources or
    dev cards) are never consulted, so this is safe to call for any seat in
    self-play.

    Higher victory points / more built structures / holding bonuses all
    strictly increase ``Phi`` (monotonic in each public progress dimension).
    """
    feats = _seat_features(public_state, seat)

    vps = float(feats.get("public_victory_points", 0))
    settlements = float(feats.get("settlements_built", 0))
    cities = float(feats.get("cities_built", 0))
    roads = float(feats.get("roads_built", 0))
    has_longest_road = 1.0 if feats.get("has_longest_road", False) else 0.0
    has_largest_army = 1.0 if feats.get("has_largest_army", False) else 0.0
    production = float(feats.get("production_proxy", 0.0))

    # Normalizers keep each term in ~[0, 1] so weights are interpretable.
    vp_norm = vps / float(max(vps_to_win, 1))
    settlement_norm = settlements / float(MAX_SETTLEMENTS)
    city_norm = cities / float(MAX_CITIES)
    road_norm = roads / float(MAX_ROADS)
    # Production proxy is already expected to be a small bounded value; clamp to
    # [0, 1] defensively so a noisy proxy cannot blow up the potential.
    production_norm = min(max(production, 0.0), 1.0)

    return (
        weights.vp * vp_norm
        + weights.settlement * settlement_norm
        + weights.city * city_norm
        + weights.road * road_norm
        + weights.longest_road * has_longest_road
        + weights.largest_army * has_largest_army
        + weights.production * production_norm
    )


def shaped_increment(
    prev_potential: float,
    next_potential: float,
    gamma: float,
) -> float:
    """Potential-based shaping reward ``F = gamma * Phi(s') - Phi(s)``.

    Add this to the environment reward for the transition ``s -> s'``.

    Summed over a full episode with the same discount ``gamma`` used in the
    return, these increments telescope to ``gamma^T * Phi_terminal - Phi_0``,
    which is why adding them leaves the optimal policy unchanged
    (Ng, Harada & Russell 1999).
    """
    return float(gamma) * float(next_potential) - float(prev_potential)


def _production_proxy(board: Mapping[str, Any], seat: str) -> float:
    """Cheap public production/port proxy for ``seat`` in ``[0, ~1]``.

    Uses only the public board payload (buildings + tiles + ports). For each of
    the seat's buildings we sum the "pip" value (dice probability weight) of the
    adjacent number tokens, weighting cities double, and add a small bonus per
    distinct port the seat touches. The raw sum is scaled into roughly ``[0, 1]``
    by an empirical divisor so it composes with the other normalized terms.

    Returns ``0.0`` if the board payload is missing or malformed, so this term
    is strictly optional and never raises.
    """
    buildings = board.get("buildings")
    tiles = board.get("tiles")
    if not buildings or not tiles:
        return 0.0

    # Map node id -> total pip value of adjacent number tokens.
    # Standard Catan dice pips: how many of the 36 outcomes hit each number.
    pip_by_number = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
    node_pips: dict[Any, float] = {}
    for tile in tiles:
        number = tile.get("number")
        if number is None:
            continue
        pips = pip_by_number.get(int(number), 0)
        if pips == 0:
            continue
        nodes = tile.get("nodes") or {}
        for node_id in nodes.values():
            node_pips[node_id] = node_pips.get(node_id, 0.0) + pips

    seat_nodes: set[Any] = set()
    raw = 0.0
    for building in buildings:
        if building.get("player") != seat:
            continue
        node_id = building.get("node")
        seat_nodes.add(node_id)
        pips = node_pips.get(node_id, 0.0)
        multiplier = 2.0 if building.get("building_type") == "CITY" else 1.0
        raw += multiplier * pips

    # Small bonus per distinct port the seat touches.
    ports = board.get("ports") or ()
    port_bonus = 0.0
    for port in ports:
        port_nodes = set(port.get("nodes") or ())
        if seat_nodes & port_nodes:
            port_bonus += 1.0

    # Empirical scale: a strong-ish board position (~2 settlements + 1 city on
    # decent numbers) lands near 1.0 after this divisor.
    raw += 0.5 * port_bonus
    return raw / 30.0


def extract_public_state(env: Any) -> dict[str, Any]:
    """Pull the public per-seat features ``catan_potential`` needs from an env.

    Reads ``ColonistMultiAgentEnv.observation_payload(...)`` and uses only the
    public surface:

    - per-seat ``public_victory_points``, ``has_longest_road``,
      ``longest_road_length``, ``has_largest_army``, ``roads_left``,
      ``settlements_left``, ``cities_left``, and ``played_development_cards``
      (knights) from ``_player_payloads``; and
    - the public ``board`` payload for a production/port proxy.

    Crucially it does NOT read any ``resources`` / ``development_cards`` hidden
    hand entries (those exist only for the observing actor anyway).

    The returned dict has the layout documented at module level and is exactly
    what :func:`catan_potential` consumes.
    """
    player_names = list(getattr(env, "player_names"))

    # One payload is enough: every seat's *public* block is identical regardless
    # of the observing actor. We request from the first seat and read the public
    # fields for all seats. ``include_event_log=False`` keeps it cheap.
    actor = player_names[0]
    payload = env.observation_payload(actor, include_event_log=False)
    raw_players: Mapping[str, Any] = payload["players"]
    board: Mapping[str, Any] = payload.get("board", {})

    players: dict[str, Any] = {}
    for seat in player_names:
        seat_raw = raw_players[seat]
        settlements_left = int(seat_raw.get("settlements_left", MAX_SETTLEMENTS))
        cities_left = int(seat_raw.get("cities_left", MAX_CITIES))
        roads_left = int(seat_raw.get("roads_left", MAX_ROADS))
        played = seat_raw.get("played_development_cards", {}) or {}
        knights_played = int(played.get("KNIGHT", 0))

        players[seat] = {
            "public_victory_points": int(seat_raw.get("public_victory_points", 0)),
            "settlements_built": MAX_SETTLEMENTS - settlements_left,
            "cities_built": MAX_CITIES - cities_left,
            "roads_built": MAX_ROADS - roads_left,
            "has_longest_road": bool(seat_raw.get("has_longest_road", False)),
            "longest_road_length": int(seat_raw.get("longest_road_length", 0)),
            "has_largest_army": bool(seat_raw.get("has_largest_army", False)),
            "knights_played": knights_played,
            "production_proxy": _production_proxy(board, seat),
        }

    return {"players": players}
