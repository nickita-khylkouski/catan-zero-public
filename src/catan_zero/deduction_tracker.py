"""Exact-deduction tracker for opponent hidden state (CAT-59).

**Design basis (CAT-73 verification, `docs/audits/CAT73_VERIFICATIONS.md`):**

- Steal-card identity is REDACTED in the event log for every audience,
  including the thief and the victim (`_redact_event` in
  `catan_zero.rl.multiagent_env` ignores its `actor` argument for
  `MOVE_ROBBER`/`DISCARD_RESOURCE`/`BUY_DEVELOPMENT_CARD`). So the stolen
  card's identity can only be recovered INDIRECTLY: by diffing a player's own
  exact hand snapshot immediately before/after the event, for whichever
  player was the thief or the victim. A third party (only possible with 3+
  players) never learns the identity, only the aggregate count change.
- Shards are written OMNISCIENT (`entity_token_features._player_tokens`
  populates every player's exact resource/dev-card slots unconditionally;
  masking is a downstream, optional transform). This tracker is deliberately
  built to consume only the REDACTED/self-scoped view (what a real inference
  -time agent would see), so it can be validated against the omniscient
  ground truth that shards separately retain.

**What "exact" means here:**

This is NOT a probabilistic belief model over the full hidden state. It is a
running fold over public history (builds, trades, dice production, played
dev cards -- all fully public, including WHICH board tiles/nodes are
involved) plus the tracked player's own observations (their exact hand,
every step). The fold produces, per opponent:

- Per-resource INTERVAL BOUNDS `[lower, upper]` that are always true (i.e.
  `lower <= true_count <= upper`), collapsing to an exact point (`lower ==
  upper`) whenever public information plus self-observation fully pins the
  count down. In two-player games, and for most event types even with more
  players, the true count is fully pinned almost always -- the width only
  grows when an opponent discards (identity hidden) or when a steal happens
  between two OTHER opponents (only possible with 3+ players; a third party
  never learns the identity).
- A multivariate-hypergeometric POSTERIOR over unplayed dev-card identity,
  conditioned on the fixed 25-card starting deck, all publicly PLAYED cards
  (by anyone), and the tracked player's own currently-held cards (exact, by
  construction). This is exact in the "correct probability model" sense, not
  a point value -- hidden dev-card identity (in particular the VICTORY_POINT
  card) is genuinely uncertain until played or the game ends.

The dev-card posterior needs no incremental state: it is a pure function of
the current public snapshot (see `dev_card_posterior_for`). The resource
bounds DO need incremental state (the "running-count fold"), since a
snapshot alone cannot recover history that already collapsed into an
aggregate count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np

RESOURCES: tuple[str, ...] = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
DEV_CARD_TYPES: tuple[str, ...] = (
    "KNIGHT",
    "YEAR_OF_PLENTY",
    "MONOPOLY",
    "ROAD_BUILDING",
    "VICTORY_POINT",
)
# Dev cards that ever appear in the public `played_development_cards` field
# (VICTORY_POINT is never "played" as an action -- see ACTION_TYPES in
# `entity_token_features.py`, there is no PLAY_VICTORY_POINT_CARD action).
PLAYABLE_DEV_CARD_TYPES: tuple[str, ...] = (
    "KNIGHT",
    "YEAR_OF_PLENTY",
    "MONOPOLY",
    "ROAD_BUILDING",
)
STARTING_DEV_DECK: dict[str, int] = {
    "KNIGHT": 14,
    "YEAR_OF_PLENTY": 2,
    "MONOPOLY": 2,
    "ROAD_BUILDING": 2,
    "VICTORY_POINT": 5,
}
STARTING_DEV_DECK_TOTAL = sum(STARTING_DEV_DECK.values())  # 25

# Canonical up-to-4-player ordering, matching `entity_token_features.PLAYERS`.
# Kept as a local constant (rather than importing that module) to keep this
# tracker importable without pulling in the full featurization stack.
PLAYERS: tuple[str, ...] = ("BLUE", "RED", "ORANGE", "WHITE")

ROAD_COST: dict[str, int] = {"WOOD": 1, "BRICK": 1}
SETTLEMENT_COST: dict[str, int] = {"WOOD": 1, "BRICK": 1, "SHEEP": 1, "WHEAT": 1}
CITY_COST: dict[str, int] = {"WHEAT": 2, "ORE": 3}
DEV_CARD_COST: dict[str, int] = {"SHEEP": 1, "WHEAT": 1, "ORE": 1}

BANK_STARTING_RESOURCE_COUNT = 19  # per-resource bank size; used as a feature denominator


def _neg(cost: Mapping[str, int]) -> dict[str, int]:
    return {resource: -amount for resource, amount in cost.items()}


@dataclass
class ResourceBounds:
    """True interval bounds on an opponent's per-resource hand composition.

    Invariant maintained by every update in `DeductionTracker`: for every
    resource `r`, `0 <= lower[r] <= upper[r]`, and `sum(lower) <= total <=
    sum(upper)` where `total` is that opponent's public `resource_card_count`
    at the same point in time.
    """

    lower: dict[str, int] = field(default_factory=lambda: {r: 0 for r in RESOURCES})
    upper: dict[str, int] = field(default_factory=lambda: {r: 0 for r in RESOURCES})

    @classmethod
    def zero(cls) -> "ResourceBounds":
        return cls(lower={r: 0 for r in RESOURCES}, upper={r: 0 for r in RESOURCES})

    def exact(self) -> dict[str, int] | None:
        """Return the exact hand if every resource is pinned, else None."""
        if all(self.lower[r] == self.upper[r] for r in RESOURCES):
            return dict(self.lower)
        return None

    def width(self) -> int:
        """Total remaining uncertainty (sum of per-resource interval widths)."""
        return sum(self.upper[r] - self.lower[r] for r in RESOURCES)

    def contains(self, true_hand: Mapping[str, int], total: int | None = None) -> bool:
        """True iff `true_hand` is consistent with every per-resource bound
        (a correctness invariant that should NEVER be false for a
        correctly-implemented tracker; used by the validation harness). If
        `total` (the player's public `resource_card_count` at the same
        point in time) is supplied, also checks that `true_hand` sums to it
        -- per-resource bounds alone don't capture that joint constraint."""
        if total is not None and sum(int(true_hand.get(r, 0)) for r in RESOURCES) != total:
            return False
        return all(
            self.lower[r] <= int(true_hand.get(r, 0)) <= self.upper[r]
            for r in RESOURCES
        )


@dataclass
class DevCardPosterior:
    """Multivariate-hypergeometric posterior over one opponent's unplayed
    dev-card identities, given the public deck-composition snapshot."""

    unknown_pool: dict[str, int]
    opponent_hidden_count: int

    @property
    def pool_total(self) -> int:
        return sum(self.unknown_pool.values())

    def expected_count(self, card: str) -> float:
        n_pool = self.pool_total
        if n_pool <= 0 or self.opponent_hidden_count <= 0:
            return 0.0
        return self.opponent_hidden_count * self.unknown_pool.get(card, 0) / n_pool

    def pmf(self, card: str) -> dict[int, float]:
        """Full hypergeometric pmf for the COUNT of `card` among the
        opponent's `opponent_hidden_count` hidden slots."""
        n_pool = self.pool_total
        k_successes = self.unknown_pool.get(card, 0)
        draws = self.opponent_hidden_count
        return _hypergeom_pmf(n_pool, k_successes, draws)

    def probability_at_least_one(self, card: str) -> float:
        n_pool = self.pool_total
        k_successes = self.unknown_pool.get(card, 0)
        draws = self.opponent_hidden_count
        return _hypergeom_at_least_one(n_pool, k_successes, draws)

    def victory_point_probability(self) -> float:
        """P(opponent holds >= 1 hidden VICTORY_POINT dev card)."""
        return self.probability_at_least_one("VICTORY_POINT")


def _hypergeom_pmf(population: int, successes: int, draws: int) -> dict[int, float]:
    if population <= 0 or draws <= 0:
        return {0: 1.0}
    successes = max(0, min(successes, population))
    draws = max(0, min(draws, population))
    failures = population - successes
    total_ways = math.comb(population, draws)
    if total_ways == 0:
        return {0: 1.0}
    lo = max(0, draws - failures)
    hi = min(draws, successes)
    pmf: dict[int, float] = {}
    for k in range(lo, hi + 1):
        pmf[k] = math.comb(successes, k) * math.comb(failures, draws - k) / total_ways
    return pmf


def _hypergeom_at_least_one(population: int, successes: int, draws: int) -> float:
    if population <= 0 or draws <= 0 or successes <= 0:
        return 0.0
    successes = max(0, min(successes, population))
    draws = max(0, min(draws, population))
    failures = population - successes
    if draws > failures:
        return 1.0
    total_ways = math.comb(population, draws)
    if total_ways == 0:
        return 0.0
    p_zero = math.comb(failures, draws) / total_ways
    return 1.0 - p_zero


def _resource_hand(payload_players: Mapping[str, Any], name: str) -> dict[str, int] | None:
    entry = payload_players.get(name)
    if not isinstance(entry, Mapping):
        return None
    resources = entry.get("resources")
    if not isinstance(resources, Mapping):
        return None
    return {r: int(resources.get(r, 0) or 0) for r in RESOURCES}


def _dev_hand(payload_players: Mapping[str, Any], name: str) -> dict[str, int] | None:
    entry = payload_players.get(name)
    if not isinstance(entry, Mapping):
        return None
    cards = entry.get("development_cards")
    if not isinstance(cards, Mapping):
        return None
    return {c: int(cards.get(c, 0) or 0) for c in DEV_CARD_TYPES}


def _infer_single_resource_delta(
    before: Mapping[str, int] | None,
    after: Mapping[str, int] | None,
) -> tuple[str, int] | None:
    """Find the single resource whose count changed between two exact hand
    snapshots of the SAME player. Returns (resource, signed_delta), or None
    if no exact snapshot is available or more than one resource changed
    (which should not happen across a single atomic event)."""
    if before is None or after is None:
        return None
    changed = [(r, after[r] - before[r]) for r in RESOURCES if after[r] != before[r]]
    if len(changed) != 1:
        return None
    return changed[0]


def _tile_adjacency(board: Mapping[str, Any]) -> dict[int, list[Mapping[str, Any]]]:
    adjacency: dict[int, list[Mapping[str, Any]]] = {}
    for tile in board.get("tiles", ()):
        for node_id in tile.get("nodes", {}).values():
            adjacency.setdefault(int(node_id), []).append(tile)
    return adjacency


def _buildings_by_node(board: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    return {int(building["node"]): building for building in board.get("buildings", ())}


def _settlement_count(board: Mapping[str, Any], player: str) -> int:
    return sum(
        1
        for building in board.get("buildings", ())
        if building.get("player") == player and building.get("building_type") == "SETTLEMENT"
    )


def _second_settlement_yield(board: Mapping[str, Any], node_id: int) -> dict[str, int]:
    """Exact free resources granted for placing the SECOND initial
    settlement (one card per adjacent resource-producing tile; deserts and
    off-map slots contribute nothing). Purely a function of public board
    geometry."""
    yield_vector: dict[str, int] = {}
    for tile in _tile_adjacency(board).get(node_id, ()):
        resource = tile.get("resource")
        if resource in RESOURCES:
            yield_vector[resource] = yield_vector.get(resource, 0) + 1
    return yield_vector


def compute_roll_production(
    board_before: Mapping[str, Any],
    bank_before: Mapping[str, Any],
    dice_sum: int,
) -> dict[str, dict[str, int]]:
    """Exact per-player production for a dice roll, replicating
    `catanatron.apply_action.yield_resources` using only public information:
    board geometry (tile resource/number, robber location, building
    ownership) and the bank's PRE-roll per-resource count (to reproduce the
    "depleted resource yields nothing to anyone this turn" rule)."""
    intended: dict[str, dict[str, int]] = {}
    totals: dict[str, int] = {r: 0 for r in RESOURCES}
    buildings = _buildings_by_node(board_before)
    for tile in board_before.get("tiles", ()):
        if tile.get("number") != dice_sum or tile.get("has_robber"):
            continue
        resource = tile.get("resource")
        if resource not in RESOURCES:
            continue
        for node_id in tile.get("nodes", {}).values():
            building = buildings.get(int(node_id))
            if building is None:
                continue
            amount = 1 if building.get("building_type") == "SETTLEMENT" else 2
            player = building.get("player")
            intended.setdefault(player, {r: 0 for r in RESOURCES})
            intended[player][resource] += amount
            totals[resource] += amount
    bank_resources = bank_before.get("resources", {})
    depleted = {r for r in RESOURCES if int(bank_resources.get(r, 0) or 0) < totals[r]}
    for player_totals in intended.values():
        for resource in depleted:
            player_totals[resource] = 0
    return intended


@dataclass
class DeductionTracker:
    """Per-perspective exact-deduction tracker: tracks `opponent_names`'
    resource-hand bounds from `self_name`'s point of view, folding the public
    event stream (redacted exactly as a real agent would see it) plus
    `self_name`'s own exact hand snapshots.

    Usage: construct once per (game, perspective player), then feed replay
    frames in order via `observe_frames`. Each frame is expected to have the
    shape produced by `ColonistMultiAgentEnv.replay_trace(actor=self_name)`
    (or the internal `_replay_frames`): `{"event": {...}, "observations":
    {player_name: observation_payload, ...}}`. Only
    `frame["observations"][self_name]` is ever read -- that sub-dict is
    already scoped to `self_name`'s own view (exact own hand, public-only
    aggregates for everyone else), so this never has access to more
    information than a real inference-time agent would.
    """

    self_name: str
    opponent_names: tuple[str, ...]
    bounds: dict[str, ResourceBounds] = field(default_factory=dict)
    anomalies: list[dict[str, Any]] = field(default_factory=list)
    _last_payload: dict[str, Any] | None = field(default=None, repr=False)
    _last_self_resources: dict[str, int] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.bounds:
            self.bounds = {name: ResourceBounds.zero() for name in self.opponent_names}

    # -- public API ---------------------------------------------------

    def bounds_for(self, opponent: str) -> ResourceBounds:
        return self.bounds[opponent]

    def dev_card_posterior_for(
        self,
        opponent: str,
        payload: Mapping[str, Any] | None = None,
    ) -> DevCardPosterior:
        """Stateless: computed fresh from the latest public snapshot (or an
        explicitly supplied `observation_payload`-shaped dict) plus
        `self_name`'s own current hand. No incremental history is needed --
        see module docstring."""
        payload = payload if payload is not None else self._last_payload
        if payload is None:
            return DevCardPosterior(unknown_pool=dict(STARTING_DEV_DECK), opponent_hidden_count=0)
        players = payload.get("players", {})
        played_by_type = {card: 0 for card in PLAYABLE_DEV_CARD_TYPES}
        for name, entry in players.items():
            played = entry.get("played_development_cards") if isinstance(entry, Mapping) else None
            if not isinstance(played, Mapping):
                continue
            for card in PLAYABLE_DEV_CARD_TYPES:
                played_by_type[card] += int(played.get(card, 0) or 0)
        self_hand = _dev_hand(players, self.self_name) or {c: 0 for c in DEV_CARD_TYPES}
        unknown_pool = {
            card: max(
                0,
                STARTING_DEV_DECK[card] - played_by_type.get(card, 0) - self_hand.get(card, 0),
            )
            for card in DEV_CARD_TYPES
        }
        opponent_entry = players.get(opponent, {})
        hidden_count = int(opponent_entry.get("development_card_count", 0) or 0) if isinstance(opponent_entry, Mapping) else 0
        return DevCardPosterior(unknown_pool=unknown_pool, opponent_hidden_count=hidden_count)

    def observe_frames(self, frames: Sequence[Mapping[str, Any]]) -> None:
        for frame in frames:
            self._observe_one(frame)

    def feature_vector_for(self, opponent: str, payload: Mapping[str, Any] | None = None) -> np.ndarray:
        """Fixed-size (`DEDUCTION_FEATURE_SIZE`,) float32 vector for one
        opponent. See `DEDUCTION_FEATURE_SIZE` docstring for the slot layout.
        """
        return _feature_vector(self.bounds[opponent], self.dev_card_posterior_for(opponent, payload))

    def feature_table(
        self,
        payload: Mapping[str, Any] | None = None,
        players: tuple[str, ...] = PLAYERS,
    ) -> np.ndarray:
        """`(len(players), DEDUCTION_FEATURE_SIZE)` float32 table, one row
        per canonical player slot (matching `entity_token_features.PLAYERS`
        ordering, for later concatenation onto player tokens). Rows for
        `self_name` and any player not tracked as an opponent are zero."""
        table = np.zeros((len(players), DEDUCTION_FEATURE_SIZE), dtype=np.float32)
        for idx, name in enumerate(players):
            if name in self.bounds:
                table[idx] = self.feature_vector_for(name, payload)
        return table

    # -- internals ------------------------------------------------------

    def _observe_one(self, frame: Mapping[str, Any]) -> None:
        payload = frame["observations"].get(self.self_name)
        if payload is None:
            return
        event = frame.get("event", {})
        players_after = payload.get("players", {})
        self_after = _resource_hand(players_after, self.self_name)
        if self._last_payload is not None:
            self._apply_event(event, self._last_payload, payload)
        self._last_payload = payload
        self._last_self_resources = self_after
        for opponent in self.opponent_names:
            total_after = self._public_total(payload, opponent)
            self._clip(opponent, total_after)

    def _public_total(self, payload: Mapping[str, Any], name: str) -> int:
        entry = payload.get("players", {}).get(name, {})
        return int(entry.get("resource_card_count", 0) or 0) if isinstance(entry, Mapping) else 0

    def _clip(self, opponent: str, total: int) -> None:
        bounds = self.bounds[opponent]
        for r in RESOURCES:
            bounds.lower[r] = max(0, min(bounds.lower[r], total))
            bounds.upper[r] = max(0, min(bounds.upper[r], total))
            if bounds.lower[r] > bounds.upper[r]:
                bounds.lower[r] = bounds.upper[r]

    def _apply_exact_delta(self, opponent: str, resource: str, delta: int) -> None:
        bounds = self.bounds[opponent]
        bounds.lower[resource] += delta
        bounds.upper[resource] += delta

    def _apply_exact_vector(self, opponent: str, vector: Mapping[str, int]) -> None:
        for resource, delta in vector.items():
            if delta:
                self._apply_exact_delta(opponent, resource, delta)

    def _apply_unknown_removal(self, opponent: str, count: int, new_total: int) -> None:
        if count <= 0:
            return
        bounds = self.bounds[opponent]
        for r in RESOURCES:
            bounds.lower[r] = max(0, bounds.lower[r] - count)
            bounds.upper[r] = min(bounds.upper[r], new_total)

    def _apply_unknown_gain(self, opponent: str, count: int, new_total: int) -> None:
        if count <= 0:
            return
        bounds = self.bounds[opponent]
        for r in RESOURCES:
            bounds.upper[r] = min(bounds.upper[r] + count, new_total)

    def _record_anomaly(self, kind: str, **details: Any) -> None:
        self.anomalies.append({"kind": kind, **details})

    def _apply_event(
        self,
        event: Mapping[str, Any],
        before: Mapping[str, Any],
        after: Mapping[str, Any],
    ) -> None:
        if event.get("event_type") != "board_action":
            return
        payload = event.get("payload", {})
        action = payload.get("action")
        if not isinstance(action, Mapping):
            return
        action_type = action.get("action_type")
        actor = event.get("actor")
        value = action.get("value")
        before_players = before.get("players", {})
        after_players = after.get("players", {})

        if action_type == "BUILD_ROAD":
            self._apply_fixed_cost_build(actor, before_players, after_players, ROAD_COST, "BUILD_ROAD")
        elif action_type == "BUILD_SETTLEMENT":
            self._apply_settlement_build(actor, value, before, after)
        elif action_type == "BUILD_CITY":
            self._apply_fixed_cost_build(actor, before_players, after_players, CITY_COST, "BUILD_CITY")
        elif action_type == "BUY_DEVELOPMENT_CARD":
            self._apply_fixed_cost_build(actor, before_players, after_players, DEV_CARD_COST, "BUY_DEVELOPMENT_CARD")
        elif action_type == "ROLL":
            self._apply_roll(before, after, payload)
        elif action_type == "MOVE_ROBBER":
            self._apply_move_robber(actor, value, payload, before_players, after_players)
        elif action_type == "DISCARD_RESOURCE":
            self._apply_discard(actor, before_players, after_players)
        elif action_type == "PLAY_YEAR_OF_PLENTY":
            self._apply_year_of_plenty(actor, value)
        elif action_type == "PLAY_MONOPOLY":
            self._apply_monopoly(actor, value, before_players, after_players)
        elif action_type == "MARITIME_TRADE":
            self._apply_maritime_trade(actor, value)
        elif action_type == "confirm_trade":
            self._apply_confirm_trade(actor, value, before)
        # PLAY_KNIGHT_CARD / PLAY_ROAD_BUILDING / offer_trade / accept_trade /
        # reject_trade / cancel_trade / END_TURN: no direct resource-hand
        # effect of their own (knight/road-building consumption is already
        # captured by the public `played_development_cards` counter; any
        # robber move or free roads they trigger show up as their own
        # separate events).

    def _apply_fixed_cost_build(
        self,
        actor: str | None,
        before_players: Mapping[str, Any],
        after_players: Mapping[str, Any],
        cost: Mapping[str, int],
        label: str,
    ) -> None:
        if actor is None or actor not in self.bounds:
            return
        total_before = int(before_players.get(actor, {}).get("resource_card_count", 0) or 0)
        total_after = int(after_players.get(actor, {}).get("resource_card_count", 0) or 0)
        delta_total = total_after - total_before
        cost_total = sum(cost.values())
        if delta_total == 0:
            return  # free build (setup phase / free-roads card) -- no resource change
        if delta_total == -cost_total:
            self._apply_exact_vector(actor, _neg(cost))
            return
        self._record_anomaly(
            "unexpected_build_delta", action=label, actor=actor, delta_total=delta_total, expected=-cost_total
        )
        if delta_total < 0:
            self._apply_unknown_removal(actor, -delta_total, total_after)
        else:
            self._apply_unknown_gain(actor, delta_total, total_after)

    def _apply_settlement_build(
        self,
        actor: str | None,
        node_id: Any,
        before: Mapping[str, Any],
        after: Mapping[str, Any],
    ) -> None:
        if actor is None or actor not in self.bounds:
            return
        before_players = before.get("players", {})
        after_players = after.get("players", {})
        total_before = int(before_players.get(actor, {}).get("resource_card_count", 0) or 0)
        total_after = int(after_players.get(actor, {}).get("resource_card_count", 0) or 0)
        delta_total = total_after - total_before
        if delta_total == 0:
            return  # 1st initial settlement, or 2nd next to only desert/no tiles
        if delta_total < 0:
            if delta_total == -sum(SETTLEMENT_COST.values()):
                self._apply_exact_vector(actor, _neg(SETTLEMENT_COST))
            else:
                self._record_anomaly(
                    "unexpected_settlement_cost_delta", actor=actor, delta_total=delta_total
                )
                self._apply_unknown_removal(actor, -delta_total, total_after)
            return
        # delta_total > 0: second initial settlement's free resource yield.
        board_before = before.get("board", {})
        try:
            yield_vector = _second_settlement_yield(board_before, int(node_id))
        except (TypeError, ValueError):
            yield_vector = {}
        if sum(yield_vector.values()) == delta_total:
            self._apply_exact_vector(actor, yield_vector)
        else:
            self._record_anomaly(
                "unexpected_settlement_yield_delta",
                actor=actor,
                delta_total=delta_total,
                computed=yield_vector,
            )
            self._apply_unknown_gain(actor, delta_total, total_after)

    def _apply_roll(
        self,
        before: Mapping[str, Any],
        after: Mapping[str, Any],
        payload: Mapping[str, Any],
    ) -> None:
        result = payload.get("result")
        if not isinstance(result, (list, tuple)) or len(result) != 2:
            return
        dice_sum = int(result[0]) + int(result[1])
        board_before = before.get("board", {})
        bank_before = before.get("bank", {})
        production = compute_roll_production(board_before, bank_before, dice_sum)
        for player, vector in production.items():
            if player in self.bounds:
                self._apply_exact_vector(player, vector)

    def _apply_move_robber(
        self,
        actor: str | None,
        value: Any,
        payload: Mapping[str, Any],
        before_players: Mapping[str, Any],
        after_players: Mapping[str, Any],
    ) -> None:
        if payload.get("result") is None:
            return  # no steal happened (empty tile / no adjacent victim)
        victim = None
        if isinstance(value, (list, tuple)) and len(value) == 2:
            victim = value[1]
        if victim is None or actor is None:
            return
        if actor == self.self_name and victim in self.bounds:
            gained = _infer_single_resource_delta(self._last_self_resources, _resource_hand(after_players, self.self_name))
            if gained is not None and gained[1] == 1:
                self._apply_exact_delta(victim, gained[0], -1)
            else:
                total_after = self._public_total({"players": after_players}, victim)
                self._apply_unknown_removal(victim, 1, total_after)
        elif victim == self.self_name and actor in self.bounds:
            lost = _infer_single_resource_delta(self._last_self_resources, _resource_hand(after_players, self.self_name))
            if lost is not None and lost[1] == -1:
                self._apply_exact_delta(actor, lost[0], 1)
            else:
                total_after = self._public_total({"players": after_players}, actor)
                self._apply_unknown_gain(actor, 1, total_after)
        else:
            # third party (3+ players only): unknown identity either side.
            if victim in self.bounds:
                total_after = self._public_total({"players": after_players}, victim)
                self._apply_unknown_removal(victim, 1, total_after)
            if actor in self.bounds:
                total_after = self._public_total({"players": after_players}, actor)
                self._apply_unknown_gain(actor, 1, total_after)

    def _apply_discard(
        self,
        actor: str | None,
        before_players: Mapping[str, Any],
        after_players: Mapping[str, Any],
    ) -> None:
        if actor is None or actor == self.self_name or actor not in self.bounds:
            return  # our own discards don't inform opponent tracking
        total_after = int(after_players.get(actor, {}).get("resource_card_count", 0) or 0)
        self._apply_unknown_removal(actor, 1, total_after)

    def _apply_year_of_plenty(self, actor: str | None, value: Any) -> None:
        if actor is None or actor not in self.bounds:
            return
        if not isinstance(value, (list, tuple)):
            return
        vector: dict[str, int] = {}
        for resource in value:
            if resource in RESOURCES:
                vector[resource] = vector.get(resource, 0) + 1
        self._apply_exact_vector(actor, vector)

    def _apply_monopoly(
        self,
        actor: str | None,
        resource: Any,
        before_players: Mapping[str, Any],
        after_players: Mapping[str, Any],
    ) -> None:
        if resource not in RESOURCES:
            return
        total_gained = 0
        for name in before_players:
            if name == actor:
                continue
            before_total = int(before_players.get(name, {}).get("resource_card_count", 0) or 0)
            after_total = int(after_players.get(name, {}).get("resource_card_count", 0) or 0)
            lost = before_total - after_total
            if lost <= 0:
                continue
            total_gained += lost
            if name in self.bounds:
                # Monopoly takes ALL of that resource: post-state is exactly 0.
                self.bounds[name].lower[resource] = 0
                self.bounds[name].upper[resource] = 0
        if actor is not None and actor in self.bounds and total_gained:
            self._apply_exact_delta(actor, resource, total_gained)

    def _apply_maritime_trade(self, actor: str | None, value: Any) -> None:
        if actor is None or actor not in self.bounds:
            return
        if not isinstance(value, (list, tuple)) or len(value) != 5:
            return
        give = [item for item in value[:4] if item in RESOURCES]
        receive = value[4]
        vector: dict[str, int] = {}
        for resource in give:
            vector[resource] = vector.get(resource, 0) - 1
        if receive in RESOURCES:
            vector[receive] = vector.get(receive, 0) + 1
        self._apply_exact_vector(actor, vector)

    def _apply_confirm_trade(
        self,
        actor: str | None,
        responder: Any,
        before: Mapping[str, Any],
    ) -> None:
        # The `confirm_trade` EVENT only logs the responder's color
        # (`_build_extended_actions`/`describe_action` never reconstruct the
        # give/want bundle for logging purposes -- only `_decode_action` does,
        # for EXECUTION). The actual bundle is public via the pre-confirm
        # frame's `trade_panel.current_board_trade.trade`, a
        # `(*give[5], *want[5], turn_index)` tuple mirroring
        # `catanatron.state.State.current_trade` (cleared once confirmed).
        trade_panel = before.get("trade_panel") or {}
        current_board_trade = trade_panel.get("current_board_trade") or {}
        trade = current_board_trade.get("trade")
        if not isinstance(trade, (list, tuple)) or len(trade) < 10:
            return
        give = trade[0:5]
        want = trade[5:10]
        if actor is not None and actor in self.bounds:
            vector = {
                r: (int(want[i]) if want[i] else 0) - (int(give[i]) if give[i] else 0)
                for i, r in enumerate(RESOURCES)
            }
            self._apply_exact_vector(actor, vector)
        if responder is not None and responder in self.bounds:
            vector = {
                r: (int(give[i]) if give[i] else 0) - (int(want[i]) if want[i] else 0)
                for i, r in enumerate(RESOURCES)
            }
            self._apply_exact_vector(responder, vector)


# -- feature vector -------------------------------------------------------

# Slot layout (17 floats total), documented so a later ticket can wire this
# in as near-zero-init extra columns on the corresponding player token row
# (see `entity_token_features.PLAYER_FEATURE_SIZE` for the existing layout
# this is meant to sit alongside):
#   0-4   resource lower bound / BANK_STARTING_RESOURCE_COUNT, one per RESOURCES entry
#   5-9   resource upper bound / BANK_STARTING_RESOURCE_COUNT, one per RESOURCES entry
#   10-14 expected hidden dev-card count / STARTING_DEV_DECK[card], one per DEV_CARD_TYPES entry
#   15    P(opponent holds >= 1 hidden VICTORY_POINT dev card)
#   16    resource-hand exactness flag (1.0 if fully pinned, else 0.0)
DEDUCTION_FEATURE_SIZE = 17


def _clip01(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


def _feature_vector(bounds: ResourceBounds, posterior: DevCardPosterior) -> np.ndarray:
    vector = np.zeros(DEDUCTION_FEATURE_SIZE, dtype=np.float32)
    for offset, resource in enumerate(RESOURCES):
        vector[offset] = _clip01(bounds.lower[resource] / BANK_STARTING_RESOURCE_COUNT)
        vector[5 + offset] = _clip01(bounds.upper[resource] / BANK_STARTING_RESOURCE_COUNT)
    for offset, card in enumerate(DEV_CARD_TYPES):
        vector[10 + offset] = _clip01(posterior.expected_count(card) / STARTING_DEV_DECK[card])
    vector[15] = _clip01(posterior.victory_point_probability())
    vector[16] = 1.0 if bounds.exact() is not None else 0.0
    return vector


def true_state_label(payload: Mapping[str, Any], opponent: str) -> dict[str, Any] | None:
    """Omniscient ground-truth label for `opponent`, for aux-head training.

    Legal to read directly at TRAINING time only, from a payload where
    `opponent`'s own exact fields are populated (i.e. `payload["players"]`
    built with `opponent` as the acting perspective, as banked shards are --
    see CAT-73 finding #4). Returns None if the payload doesn't carry
    `opponent`'s exact fields (e.g. it was built from a different actor's
    redacted perspective).
    """
    players = payload.get("players", {})
    resources = _resource_hand(players, opponent)
    dev_cards = _dev_hand(players, opponent)
    if resources is None or dev_cards is None:
        return None
    return {"resources": resources, "development_cards": dev_cards}
