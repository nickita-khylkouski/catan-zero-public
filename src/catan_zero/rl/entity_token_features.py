from __future__ import annotations

from typing import Any

import numpy as np


ENTITY_TOKEN_SCHEMA_VERSION = "entity_tokens_v1"

RESOURCES = ("wood", "brick", "sheep", "wheat", "ore")
PLAYERS = ("BLUE", "RED", "ORANGE", "WHITE")
PROMPTS = (
    "BUILD_INITIAL_SETTLEMENT",
    "BUILD_INITIAL_ROAD",
    "ROLL",
    "PLAY_TURN",
    "DISCARD",
    "MOVE_ROBBER",
    "RESPOND_TO_TRADE",
    "CONFIRM_TRADE",
)
ACTION_TYPES = (
    "BUILD_SETTLEMENT",
    "BUILD_ROAD",
    "BUILD_CITY",
    "BUY_DEVELOPMENT_CARD",
    "MARITIME_TRADE",
    "offer_trade",
    "accept_trade",
    "reject_trade",
    "cancel_trade",
    "confirm_trade",
    "MOVE_ROBBER",
    "DISCARD_RESOURCE",
    "PLAY_KNIGHT_CARD",
    "PLAY_YEAR_OF_PLENTY",
    "PLAY_MONOPOLY",
    "PLAY_ROAD_BUILDING",
    "ROLL",
    "END_TURN",
)
CATEGORIES = ("build", "trade", "development", "robber", "turn")
EVENT_TYPES = (
    "reset",
    "board_action",
    "trade_proposal",
    "trade_response",
    "trade_counteroffer",
    "chat",
    "timeout",
    "invalid_action",
)

HEX_FEATURE_SIZE = 13
VERTEX_FEATURE_SIZE = 24
EDGE_FEATURE_SIZE = 8
PLAYER_FEATURE_SIZE = 31
GLOBAL_FEATURE_SIZE = 43
LEGAL_ACTION_FEATURE_SIZE = 50
EVENT_FEATURE_SIZE = 41

# Public-observation boundary (hidden-information leak fix, f72).
# The player-token slot where `_player_tokens` writes the "this token is the
# actor" one-hot (see `_player_tokens`: `tokens[idx, 1] = 1.0 if name == actor`).
# Used to identify which of the up-to-4 player rows is the perspective player,
# whose OWN hand is legitimately visible and must NOT be masked.
PLAYER_ACTOR_FLAG_SLOT = 1
# Player-token feature slots that encode a player's HIDDEN information, which a
# public-information observer of an OPPONENT cannot see (see `_player_tokens`
# for the layout):
#   4  has_actual_victory_points flag   5  actual VP (incl. hidden VP cards)
#   15 has_resources flag               16-20 resource-hand composition
#   21 has_development_cards flag        22-26 unplayed dev-card identities
# Deliberately EXCLUDED (public, kept): 6 resource_card_count, 7 development_card_count,
# 3 public_victory_points, 27-30 played dev cards, and all board/road/army slots.
PUBLIC_MASK_PLAYER_SLOTS: tuple[int, ...] = (4, 5, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26)


def mask_player_tokens_public(player_tokens: np.ndarray) -> np.ndarray:
    """Zero the hidden-information slots of every NON-actor player row.

    Accepts a single `(4, PLAYER_FEATURE_SIZE)` token block or a batched
    `(B, 4, PLAYER_FEATURE_SIZE)` array and returns a masked COPY (never
    mutates the input). The actor row -- the one carrying the
    `PLAYER_ACTOR_FLAG_SLOT` one-hot -- is left untouched, because a player
    always sees their own hand and dev cards.

    This is the canonical, in-place-safe load-time transform for banked shards
    (`tools/train_bc.py --mask-hidden-info`) and MUST produce byte-identical
    player tokens to the online perspective-masked featurization path in
    `neural_rust_mcts.py` (see `_mask_players_to_public`), so a model trained
    on masked shards matches the public-observation evaluator at inference.
    """
    arr = np.array(player_tokens, copy=True)
    single = arr.ndim == 2
    if single:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(
            f"player_tokens must be (4, F) or (B, 4, F); got shape {np.shape(player_tokens)}"
        )
    actor = arr[:, :, PLAYER_ACTOR_FLAG_SLOT] > 0.5  # (B, 4): the perspective player
    nonactor = ~actor
    for slot in PUBLIC_MASK_PLAYER_SLOTS:
        column = arr[:, :, slot]
        column[nonactor] = 0
    return arr[0] if single else arr



def build_entity_token_features(
    env: Any,
    actor: str | None = None,
    *,
    include_event_log: bool = True,
    history_limit: int = 64,
) -> dict[str, np.ndarray]:
    """Build typed Catan entity-token tensors from the public env payload.

    This is an additive feature surface for the next model. It intentionally
    keeps the existing flat observation path untouched.
    """

    actor_name = actor or env.current_player_name()
    payload = env.observation_payload(actor_name, include_event_log=include_event_log)
    topology = _topology(payload)
    return {
        "schema": np.asarray(ENTITY_TOKEN_SCHEMA_VERSION),
        "hex_tokens": _hex_tokens(payload, topology),
        "hex_vertex_ids": topology["hex_vertex_ids"],
        "hex_edge_ids": topology["hex_edge_ids"],
        "vertex_tokens": _vertex_tokens(env, payload, topology, actor_name),
        "edge_tokens": _edge_tokens(payload, topology, actor_name),
        "edge_vertex_ids": topology["edge_vertex_ids"],
        "player_tokens": _player_tokens(payload, actor_name),
        "global_tokens": _global_tokens(env, payload, actor_name),
        "legal_action_tokens": _legal_action_tokens(env, payload, topology),
        "legal_action_target_ids": _legal_action_target_ids(payload, topology),
        "event_tokens": _event_tokens(payload, topology, history_limit=history_limit),
        "event_target_ids": _event_target_ids(payload, topology, history_limit=history_limit),
        "hex_mask": np.ones(19, dtype=np.bool_),
        "vertex_mask": np.ones(54, dtype=np.bool_),
        "edge_mask": np.ones(72, dtype=np.bool_),
        "player_mask": np.asarray([name in payload.get("players", {}) for name in PLAYERS], dtype=np.bool_),
        "legal_action_mask": np.ones(len(payload.get("structured_legal_actions", ())), dtype=np.bool_),
        "event_mask": _event_mask(payload, history_limit=history_limit),
    }


# Per-board topology cache (perf model finding: hex/vertex/edge adjacency is
# recomputed from scratch on every one of ~500k leaf featurizations per game,
# though it is invariant for the game's entire lifetime -- only resource/
# number/robber/piece state, carried by `tiles` itself and NOT cached below,
# varies per call). Keyed on `_topology_key`, a lossless (non-hashing-digest)
# fingerprint of exactly the tile/node/edge fields `_build_topology` reads, so
# a cache hit is only ever produced by an input that would make
# `_build_topology` recompute byte-identical output -- see
# `tests/test_entity_token_features.py`'s cold-vs-warm-cache equality test.
# Bounded (not a bare `maxsize=1`) so running multiple distinct boards/map
# kinds in one process (e.g. a sweep script) can't silently keep serving one
# board's topology to another; old entries are evicted FIFO once full.
_TOPOLOGY_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}
_TOPOLOGY_CACHE_MAXSIZE = 16


def _topology_key(tiles: list[dict[str, Any]]) -> tuple[Any, ...]:
    """Hashable, lossless fingerprint of the topology-relevant subset of
    `tiles` (tile_id, coordinate, and each tile's raw node/edge values, in
    their existing iteration order -- NOT resource/number/robber, which vary
    per game state but never feed `_build_topology`). This is an equality
    key, not a lossy digest: two calls only collide here if `_build_topology`
    would consume literally the same inputs in the same order, so caching on
    it can never serve a different board's arrays."""
    key: list[Any] = []
    for tile in tiles:
        tile_id = _safe_int(tile.get("tile_id"), default=0)
        coordinate = _coordinate(tile.get("coordinate"))
        nodes = tuple(_safe_int(node) for node in dict(tile.get("nodes", {})).values())
        edges = tuple(
            tuple(edge) if isinstance(edge, (list, tuple)) else edge
            for edge in dict(tile.get("edges", {})).values()
        )
        key.append((tile_id, coordinate, nodes, edges))
    return tuple(key)


def _build_topology(tiles: list[dict[str, Any]]) -> dict[str, Any]:
    """The expensive, board-invariant half of `_topology`: hex/vertex/edge
    adjacency tables. Pure function of `tiles`' tile_id/coordinate/node/edge
    fields -- identical inputs always produce identical output, which is
    exactly what `_topology`'s `_TOPOLOGY_CACHE` (keyed on `_topology_key`)
    exploits: computed once per distinct board, then reused (via `.copy()`
    of the mutable numpy arrays) for every subsequent leaf featurization of
    that board instead of rebuilding it from scratch."""
    coordinate_to_hex: dict[tuple[int, int, int], int] = {}
    edge_pairs: set[tuple[int, int]] = set()
    for tile in tiles:
        tile_id = _safe_int(tile.get("tile_id"), default=len(coordinate_to_hex))
        coordinate = _coordinate(tile.get("coordinate"))
        if coordinate is not None:
            coordinate_to_hex[coordinate] = int(tile_id)
        for edge in dict(tile.get("edges", {})).values():
            pair = _edge_pair(edge)
            if pair is not None and all(0 <= node < 54 for node in pair):
                edge_pairs.add(pair)
    edge_list = sorted(edge_pairs)
    edge_to_id = {edge: idx for idx, edge in enumerate(edge_list)}

    hex_vertex_ids = np.full((19, 6), -1, dtype=np.int16)
    hex_edge_ids = np.full((19, 6), -1, dtype=np.int16)
    for tile in tiles[:19]:
        tile_id = _safe_int(tile.get("tile_id"), default=0)
        if not 0 <= tile_id < 19:
            continue
        nodes = [
            int(node)
            for node in dict(tile.get("nodes", {})).values()
            if _safe_int(node) is not None and 0 <= int(node) < 54
        ]
        for idx, node in enumerate(nodes[:6]):
            hex_vertex_ids[tile_id, idx] = node
        tile_edges: list[int] = []
        for raw in dict(tile.get("edges", {})).values():
            edge = _edge_pair(raw)
            if edge is not None and edge in edge_to_id:
                tile_edges.append(edge_to_id[edge])
        for idx, edge_id in enumerate(tile_edges[:6]):
            hex_edge_ids[tile_id, idx] = edge_id

    edge_vertex_ids = np.full((72, 2), -1, dtype=np.int16)
    for edge, idx in edge_to_id.items():
        if idx < 72:
            edge_vertex_ids[idx] = np.asarray(edge, dtype=np.int16)

    return {
        "edge_to_id": edge_to_id,
        "edge_vertex_ids": edge_vertex_ids,
        "hex_vertex_ids": hex_vertex_ids,
        "hex_edge_ids": hex_edge_ids,
        "coordinate_to_hex": coordinate_to_hex,
    }


def _topology(payload: dict[str, Any]) -> dict[str, Any]:
    board = payload.get("board") if isinstance(payload.get("board"), dict) else {}
    tiles = sorted(
        [tile for tile in board.get("tiles", ()) if isinstance(tile, dict)],
        key=lambda tile: int(tile.get("tile_id", 0)),
    )
    key = _topology_key(tiles)
    cached = _TOPOLOGY_CACHE.get(key)
    if cached is None:
        cached = _build_topology(tiles)
        if len(_TOPOLOGY_CACHE) >= _TOPOLOGY_CACHE_MAXSIZE:
            _TOPOLOGY_CACHE.pop(next(iter(_TOPOLOGY_CACHE)))
        _TOPOLOGY_CACHE[key] = cached
    return {
        # `tiles` carries this call's resource/number/robber state -- always
        # freshly derived from `payload`, never cached.
        "tiles": tiles,
        # `edge_to_id`/`coordinate_to_hex` are read-only downstream (plain
        # dict lookups in `_edge_tokens`/`_legal_action_target_ids`), safe to
        # share the cached reference. The numpy arrays are defensively
        # copied (a cheap memcpy of <300 bytes total) so no caller can ever
        # mutate the shared cache entry in place.
        "edge_to_id": cached["edge_to_id"],
        "edge_vertex_ids": cached["edge_vertex_ids"].copy(),
        "hex_vertex_ids": cached["hex_vertex_ids"].copy(),
        "hex_edge_ids": cached["hex_edge_ids"].copy(),
        "coordinate_to_hex": cached["coordinate_to_hex"],
    }


def _hex_tokens(payload: dict[str, Any], topology: dict[str, Any]) -> np.ndarray:
    tokens = np.zeros((19, HEX_FEATURE_SIZE), dtype=np.float16)
    for tile in topology["tiles"][:19]:
        tile_id = _safe_int(tile.get("tile_id"), default=0)
        if not 0 <= tile_id < 19:
            continue
        coordinate = _coordinate(tile.get("coordinate")) or (0, 0, 0)
        tokens[tile_id, 0] = 1.0
        tokens[tile_id, 1:4] = np.asarray(coordinate, dtype=np.float32) / 4.0
        resource_index = _resource_index(tile.get("resource"))
        if resource_index is None:
            tokens[tile_id, 9] = 1.0
        else:
            tokens[tile_id, 4 + resource_index] = 1.0
        number = _safe_int(tile.get("number"), default=0)
        tokens[tile_id, 10] = _scale(number, 12)
        tokens[tile_id, 11] = _scale(_dice_pips(number), 5)
        tokens[tile_id, 12] = 1.0 if bool(tile.get("has_robber")) else 0.0
    return tokens


def _vertex_tokens(env: Any, payload: dict[str, Any], topology: dict[str, Any], actor_name: str) -> np.ndarray:
    del topology
    tokens = np.zeros((54, VERTEX_FEATURE_SIZE), dtype=np.float16)
    board = payload.get("board") if isinstance(payload.get("board"), dict) else {}
    buildings = {
        int(item["node"]): item
        for item in board.get("buildings", ())
        if isinstance(item, dict) and _safe_int(item.get("node")) is not None
    }
    port_by_node = _port_by_node(payload)
    for node in range(54):
        tokens[node, 0] = 1.0
        building = buildings.get(node)
        owner = str(building.get("player")) if isinstance(building, dict) else ""
        owner_index = _player_index(owner)
        if owner_index is None:
            tokens[node, 1] = 1.0
        else:
            tokens[node, 2 + owner_index] = 1.0
        building_type = str(building.get("building_type")) if isinstance(building, dict) else ""
        if building_type == "SETTLEMENT":
            tokens[node, 7] = 1.0
        elif building_type == "CITY":
            tokens[node, 8] = 1.0
        else:
            tokens[node, 6] = 1.0
        production = getattr(env.game.state.board.map, "node_production", {}).get(node)
        pips_by_resource = _node_pips_by_resource(production)
        total_pips = sum(pips_by_resource)
        tokens[node, 9] = _scale(total_pips, 18)
        for idx, pips in enumerate(pips_by_resource):
            tokens[node, 10 + idx] = _scale(pips, 10)
        tokens[node, 15] = _adjacent_robber(payload, node)
        port = port_by_node.get(node)
        if port is None:
            tokens[node, 16] = 1.0
        else:
            resource_index = _resource_index(port.get("resource"))
            if resource_index is None:
                tokens[node, 17] = 1.0
            else:
                tokens[node, 18 + resource_index] = 1.0
        tokens[node, 23] = 1.0 if owner == actor_name else 0.0
    return tokens


def _edge_tokens(payload: dict[str, Any], topology: dict[str, Any], actor_name: str) -> np.ndarray:
    tokens = np.zeros((72, EDGE_FEATURE_SIZE), dtype=np.float16)
    board = payload.get("board") if isinstance(payload.get("board"), dict) else {}
    road_owner: dict[tuple[int, int], str] = {}
    for road in board.get("roads", ()):
        if not isinstance(road, dict):
            continue
        edge = _edge_pair(road.get("edge"))
        if edge is not None:
            road_owner[edge] = str(road.get("player", ""))
    for edge, edge_id in topology["edge_to_id"].items():
        if edge_id >= 72:
            continue
        owner = road_owner.get(edge, "")
        tokens[edge_id, 0] = 1.0
        owner_index = _player_index(owner)
        if owner_index is None:
            tokens[edge_id, 1] = 1.0
        else:
            tokens[edge_id, 2 + owner_index] = 1.0
        adjacent_hex_count = int(np.sum(topology["hex_edge_ids"] == edge_id))
        tokens[edge_id, 6] = _scale(adjacent_hex_count, 2)
        tokens[edge_id, 7] = 1.0 if owner == actor_name else 0.0
    return tokens


def _player_tokens(payload: dict[str, Any], actor_name: str) -> np.ndarray:
    tokens = np.zeros((4, PLAYER_FEATURE_SIZE), dtype=np.float16)
    players = payload.get("players") if isinstance(payload.get("players"), dict) else {}
    current = str(payload.get("current_player", ""))
    for name in PLAYERS:
        idx = _player_index(name)
        if idx is None or name not in players:
            continue
        player = players[name]
        tokens[idx, 0] = 1.0
        tokens[idx, 1] = 1.0 if name == actor_name else 0.0
        tokens[idx, 2] = 1.0 if name == current else 0.0
        tokens[idx, 3] = _scale(player.get("public_victory_points"), 10)
        has_actual = "actual_victory_points" in player
        tokens[idx, 4] = 1.0 if has_actual else 0.0
        tokens[idx, 5] = _scale(player.get("actual_victory_points"), 10) if has_actual else 0.0
        tokens[idx, 6] = _scale(player.get("resource_card_count"), 20)
        tokens[idx, 7] = _scale(player.get("development_card_count"), 10)
        tokens[idx, 8] = _scale(player.get("roads_left"), 15)
        tokens[idx, 9] = _scale(player.get("settlements_left"), 5)
        tokens[idx, 10] = _scale(player.get("cities_left"), 4)
        tokens[idx, 11] = float(bool(player.get("has_largest_army")))
        tokens[idx, 12] = float(bool(player.get("has_longest_road")))
        tokens[idx, 13] = float(bool(player.get("has_rolled")))
        tokens[idx, 14] = _scale(player.get("longest_road_length"), 15)
        resources = player.get("resources") if isinstance(player.get("resources"), dict) else None
        tokens[idx, 15] = 1.0 if resources is not None else 0.0
        for offset, resource in enumerate(RESOURCES):
            tokens[idx, 16 + offset] = _scale(_resource_count(resources, resource), 10)
        dev_cards = player.get("development_cards") if isinstance(player.get("development_cards"), dict) else None
        tokens[idx, 21] = 1.0 if dev_cards is not None else 0.0
        for offset, card in enumerate(("KNIGHT", "YEAR_OF_PLENTY", "MONOPOLY", "ROAD_BUILDING", "VICTORY_POINT")):
            tokens[idx, 22 + offset] = _scale(_lookup_count(dev_cards, card), 5)
        played = player.get("played_development_cards") if isinstance(player.get("played_development_cards"), dict) else {}
        for offset, card in enumerate(("KNIGHT", "YEAR_OF_PLENTY", "MONOPOLY", "ROAD_BUILDING")):
            tokens[idx, 27 + offset] = _scale(_lookup_count(played, card), 5)
    return tokens


def _global_tokens(env: Any, payload: dict[str, Any], actor_name: str) -> np.ndarray:
    del env
    token = np.zeros((1, GLOBAL_FEATURE_SIZE), dtype=np.float16)
    prompt = str(payload.get("current_prompt", ""))
    for idx, known in enumerate(PROMPTS):
        token[0, idx] = 1.0 if known in prompt else 0.0
    current_idx = _player_index(str(payload.get("current_player", "")))
    actor_idx = _player_index(actor_name)
    if current_idx is not None:
        token[0, 16 + current_idx] = 1.0
    if actor_idx is not None:
        token[0, 20 + actor_idx] = 1.0
    token[0, 24] = _scale(len(payload.get("legal_actions", ())), 607)
    token[0, 25] = _scale(payload.get("replay_frame_count"), 512)
    bank = payload.get("bank") if isinstance(payload.get("bank"), dict) else {}
    bank_resources = bank.get("resources") if isinstance(bank.get("resources"), dict) else {}
    for offset, resource in enumerate(RESOURCES):
        token[0, 26 + offset] = _scale(_resource_count(bank_resources, resource), 19)
    token[0, 31] = _scale(bank.get("development_cards_remaining"), 25)
    trade_panel = payload.get("trade_panel") if isinstance(payload.get("trade_panel"), dict) else {}
    token[0, 32] = _scale(trade_panel.get("offers_remaining"), 3)
    token[0, 33] = float(bool(trade_panel.get("current_offer")))
    token[0, 34] = float(bool(trade_panel.get("is_resolving")))
    for offset, name in enumerate(PLAYERS):
        token[0, 35 + offset] = 1.0 if name in payload.get("players", {}) else 0.0
    count = len(payload.get("players", {}))
    if count in (2, 3, 4):
        token[0, 39 + (count - 2)] = 1.0
    return token


def _legal_action_tokens(env: Any, payload: dict[str, Any], topology: dict[str, Any]) -> np.ndarray:
    legal = tuple(payload.get("structured_legal_actions", ()))
    tokens = np.zeros((len(legal), LEGAL_ACTION_FEATURE_SIZE), dtype=np.float16)
    for row, action in enumerate(legal):
        tokens[row, 0] = 1.0
        tokens[row, 1] = _scale(action.get("index"), max(int(env.action_space.n), 1))
        action_type = str(action.get("action_type", ""))
        type_index = _index(ACTION_TYPES, action_type)
        if type_index is not None:
            tokens[row, 2 + type_index] = 1.0
        category_index = _index(CATEGORIES, str(action.get("category", "")))
        if category_index is not None:
            tokens[row, 20 + category_index] = 1.0
        target_kind = _target_kind(action, topology)
        kind_index = _index(("none", "hex", "vertex", "edge", "player", "resource"), target_kind)
        if kind_index is not None:
            tokens[row, 25 + kind_index] = 1.0
        args = action.get("args") if isinstance(action.get("args"), dict) else {}
        _fill_resource_bundle(tokens[row, 31:36], args.get("resources"), divisor=2)
        _fill_resource_bundle(tokens[row, 36:41], args.get("give"), divisor=4)
        _fill_resource_bundle(tokens[row, 41:46], args.get("want"), divisor=4)
        tokens[row, 46] = _priority(action_type)
        tokens[row, 47] = 1.0 if action_type == "END_TURN" else 0.0
        tokens[row, 48] = 1.0 if "INITIAL" in str(payload.get("current_prompt", "")) else 0.0
        trade_panel = payload.get("trade_panel") if isinstance(payload.get("trade_panel"), dict) else {}
        tokens[row, 49] = _scale(trade_panel.get("offers_remaining"), 3)
    return tokens


def _legal_action_target_ids(payload: dict[str, Any], topology: dict[str, Any]) -> np.ndarray:
    legal = tuple(payload.get("structured_legal_actions", ()))
    targets = np.full((len(legal), 4), -1, dtype=np.int16)
    for row, action in enumerate(legal):
        args = action.get("args") if isinstance(action.get("args"), dict) else {}
        if "tile_coordinate" in args:
            coordinate = _coordinate(args.get("tile_coordinate"))
            if coordinate is not None:
                targets[row, 0] = topology["coordinate_to_hex"].get(coordinate, -1)
        if "node" in args:
            targets[row, 1] = _safe_int(args.get("node"), default=-1)
        if "edge" in args:
            edge = _edge_pair(args.get("edge"))
            targets[row, 2] = topology["edge_to_id"].get(edge, -1) if edge is not None else -1
        victim = args.get("victim", args.get("target"))
        player_id = _player_index(str(victim)) if victim is not None else None
        if player_id is not None:
            targets[row, 3] = player_id
    return targets


def _event_tokens(payload: dict[str, Any], topology: dict[str, Any], *, history_limit: int) -> np.ndarray:
    del topology
    events = tuple(payload.get("event_log", ()))[-history_limit:]
    tokens = np.zeros((history_limit, EVENT_FEATURE_SIZE), dtype=np.float16)
    offset = history_limit - len(events)
    for idx, event in enumerate(events):
        row = offset + idx
        tokens[row, 0] = 1.0
        tokens[row, 1] = _scale(len(events) - idx, history_limit)
        event_type = str(event.get("event_type", ""))
        event_type_index = _index(EVENT_TYPES, event_type)
        if event_type_index is not None:
            tokens[row, 2 + event_type_index] = 1.0
        actor_index = _player_index(str(event.get("actor", "")))
        if actor_index is not None:
            tokens[row, 10 + actor_index] = 1.0
        turn_key = event.get("turn_key") or (0, 0)
        if isinstance(turn_key, (list, tuple)) and len(turn_key) >= 2:
            tokens[row, 15] = _scale(turn_key[0], 512)
            tokens[row, 16] = _scale(turn_key[1], 4)
        action_type = _event_action_type(event)
        action_type_index = _index(ACTION_TYPES, action_type)
        if action_type_index is not None:
            tokens[row, 17 + action_type_index] = 1.0
        action_id = _event_action_id(event)
        if action_id is not None:
            tokens[row, 35] = _scale(action_id, 607)
        target_index = _player_index(str(_event_target_player(event)))
        if target_index is not None:
            tokens[row, 36 + target_index] = 1.0
    return tokens


def _event_target_ids(payload: dict[str, Any], topology: dict[str, Any], *, history_limit: int) -> np.ndarray:
    del topology
    return np.full((history_limit, 4), -1, dtype=np.int16)


def _event_mask(payload: dict[str, Any], *, history_limit: int) -> np.ndarray:
    count = min(len(tuple(payload.get("event_log", ()))), history_limit)
    mask = np.zeros(history_limit, dtype=np.bool_)
    if count:
        mask[-count:] = True
    return mask


def _target_kind(action: dict[str, Any], topology: dict[str, Any]) -> str:
    del topology
    args = action.get("args") if isinstance(action.get("args"), dict) else {}
    if "tile_coordinate" in args:
        return "hex"
    if "node" in args:
        return "vertex"
    if "edge" in args:
        return "edge"
    if "victim" in args or "target" in args:
        return "player"
    if "resource" in args or "resources" in args:
        return "resource"
    return "none"


def _resource_index(value: Any) -> int | None:
    if value is None:
        return None
    raw = str(value).lower()
    return RESOURCES.index(raw) if raw in RESOURCES else None


def _player_index(value: str) -> int | None:
    return PLAYERS.index(value) if value in PLAYERS else None


def _index(values: tuple[str, ...], value: str) -> int | None:
    return values.index(value) if value in values else None


def _safe_int(value: Any, *, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _scale(value: Any, denominator: float) -> float:
    parsed = _safe_int(value, default=0)
    if parsed is None:
        parsed = 0
    # Plain Python min/max instead of `np.clip` on a single scalar: this
    # function is called on the order of millions of times per self-play
    # generation run (once per numeric feature per token), and `np.clip`'s
    # ufunc dispatch overhead dominates its cost at that call volume even
    # though it's only ever clamping one float. Behaviorally identical to
    # `np.clip(x, 0.0, 1.0)` for every value `_safe_int` can produce (a
    # concrete int or 0, divided by a finite positive denominator) --
    # verified in tests/test_entity_token_features_scale.py.
    ratio = float(parsed) / float(max(denominator, 1.0))
    return 0.0 if ratio < 0.0 else 1.0 if ratio > 1.0 else ratio


def _dice_pips(number: int | None) -> int:
    if number is None:
        return 0
    return max(0, 6 - abs(int(number) - 7)) if int(number) != 7 else 0


def _coordinate(value: Any) -> tuple[int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        return int(value[0]), int(value[1]), int(value[2])
    except (TypeError, ValueError):
        return None


def _edge_pair(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        a, b = int(value[0]), int(value[1])
    except (TypeError, ValueError):
        return None
    return tuple(sorted((a, b)))


def _node_pips_by_resource(production: Any) -> list[int]:
    result = [0] * len(RESOURCES)
    if not isinstance(production, dict):
        return result
    for resource, value in production.items():
        index = _resource_index(resource)
        if index is not None:
            result[index] += int(value)
    return result


def _port_by_node(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    board = payload.get("board") if isinstance(payload.get("board"), dict) else {}
    result: dict[int, dict[str, Any]] = {}
    for port in board.get("ports", ()):
        if not isinstance(port, dict):
            continue
        for node in port.get("nodes", ()):
            node_id = _safe_int(node)
            if node_id is not None and 0 <= node_id < 54:
                result[node_id] = port
    return result


def _adjacent_robber(payload: dict[str, Any], node: int) -> float:
    board = payload.get("board") if isinstance(payload.get("board"), dict) else {}
    for tile in board.get("tiles", ()):
        if not isinstance(tile, dict) or not bool(tile.get("has_robber")):
            continue
        if node in {int(raw) for raw in dict(tile.get("nodes", {})).values() if _safe_int(raw) is not None}:
            return 1.0
    return 0.0


def _resource_count(values: dict[str, Any] | None, resource: str) -> int:
    if not isinstance(values, dict):
        return 0
    return _lookup_count(values, resource)


def _lookup_count(values: dict[str, Any] | None, key: str) -> int:
    if not isinstance(values, dict):
        return 0
    for candidate in (key, key.upper(), key.lower()):
        if candidate in values:
            return int(values.get(candidate) or 0)
    return 0


def _fill_resource_bundle(target: np.ndarray, bundle: Any, *, divisor: float) -> None:
    if isinstance(bundle, dict):
        for idx, resource in enumerate(RESOURCES):
            target[idx] = _scale(_resource_count(bundle, resource), divisor)
    elif isinstance(bundle, (list, tuple)):
        for item in bundle:
            idx = _resource_index(item)
            if idx is not None:
                target[idx] = min(float(target[idx]) + 1.0 / float(divisor), 1.0)


def _priority(action_type: str) -> float:
    priorities = {
        "BUILD_CITY": 1.00,
        "BUILD_SETTLEMENT": 0.90,
        "BUILD_ROAD": 0.70,
        "BUY_DEVELOPMENT_CARD": 0.60,
        "MOVE_ROBBER": 0.35,
        "END_TURN": 0.0,
    }
    return float(priorities.get(action_type, 0.5))


def _event_action(event: dict[str, Any]) -> dict[str, Any] | None:
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    action = payload.get("action")
    return action if isinstance(action, dict) else None


def _event_action_type(event: dict[str, Any]) -> str:
    action = _event_action(event)
    return str(action.get("action_type", "")) if action else ""


def _event_action_id(event: dict[str, Any]) -> int | None:
    action = _event_action(event)
    if not action:
        return None
    return _safe_int(action.get("index"))


def _event_target_player(event: dict[str, Any]) -> str:
    action = _event_action(event) or {}
    value = action.get("value")
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return str(value[1])
    return ""
