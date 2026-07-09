"""Rust-backed companion to `entity_token_features.build_entity_token_features`.

Task #81 phases 1-2 (generation-efficiency optimization, catanatron_rs
featurize port + batch API). This module is purely ADDITIVE: nothing here
changes the behavior of any existing function, and importing `catanatron_rs`
is deferred to call time so modules that don't have the Rust extension built
(or don't request it) are unaffected.

Design (acked by team-lead/speed-czar, see task #81 discussion): the
intricate `ActionCatalog`/`rust_policy_action_ids` action-key canonicalization
stays in Python -- only the already-translated `policy_action_ids` array
crosses the FFI boundary, index-aligned to `game.playable_actions`'s own
native order (every production call site already satisfies this; no separate
rust-action-space-id translation happens on the Rust side at all). Everything
else (hex/vertex/edge/player/global/legal_action tokens, target ids, and all
masks) is built directly off the live Rust game state in
`catanatron_rs.build_entity_features_flat`/`build_entity_features_batch`,
eliminating both the JSON snapshot round-trip and the Python per-token loops.

Board/port TOPOLOGY (`hex_vertex_ids`/`hex_edge_ids`/`edge_vertex_ids`/
port-to-node assignment) is intentionally NOT recomputed on the Rust side: it
is derived once via the existing Python fixed-lookup path
(`entity_token_features._topology`/`neural_rust_mcts._base_ports`, both keyed
by hex coordinate off a canonical BASE-map environment -- see those
functions' docstrings), bundled into a `catanatron_rs.EntityTopology` object
ONCE per board/search, and passed BY REFERENCE into every per-leaf/per-batch
Rust call -- so this module never has to reproduce Python's
dict-insertion-order-dependent topology construction bit-for-bit in Rust, and
never re-marshals the topology arrays on the hot path. This is the SAME
fixed-lookup quirk `entity_token_features.py` already has: it is only
topologically correct for BASE-layout boards, and is unaffected by this port
(verbatim replication, not a behavior change).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from catan_zero.rl.entity_token_features import (
    EDGE_FEATURE_SIZE,
    EVENT_FEATURE_SIZE,
    GLOBAL_FEATURE_SIZE,
    HEX_FEATURE_SIZE,
    LEGAL_ACTION_FEATURE_SIZE,
    PLAYER_FEATURE_SIZE,
    VERTEX_FEATURE_SIZE,
)


class RustTopology:
    """Board topology + port assignment, computed once per board and reused
    across every leaf/batch featurization for that board (mirrors -- and
    replaces the per-leaf cache-key overhead of --
    `entity_token_features._topology`'s `_TOPOLOGY_CACHE`: that cache still
    requires a live JSON snapshot on every call just to compute its cache
    key, which this hoists to once-per-board).

    Holds both the plain numpy arrays (needed to add `hex_vertex_ids`/
    `hex_edge_ids`/`edge_vertex_ids` back into the output dict, matching
    `build_entity_token_features`'s contract) and the constructed
    `catanatron_rs.EntityTopology` Rust object (`.rust`, passed by reference
    into every Rust call -- built once here, never re-marshalled per leaf).
    """

    __slots__ = ("hex_vertex_ids", "hex_edge_ids", "edge_vertex_ids", "port_base_nodes", "rust")

    def __init__(
        self,
        hex_vertex_ids: np.ndarray,
        hex_edge_ids: np.ndarray,
        edge_vertex_ids: np.ndarray,
        port_base_nodes: list[list[int]],
    ) -> None:
        import catanatron_rs

        self.hex_vertex_ids = hex_vertex_ids
        self.hex_edge_ids = hex_edge_ids
        self.edge_vertex_ids = edge_vertex_ids
        self.port_base_nodes = port_base_nodes
        self.rust = catanatron_rs.EntityTopology(
            hex_vertex_ids.tolist(),
            hex_edge_ids.tolist(),
            edge_vertex_ids.tolist(),
            port_base_nodes,
        )


def compute_rust_topology(env: Any, actor_name: str) -> RustTopology:
    """Compute `RustTopology` once via the existing Python topology path.

    `env` must be a `_RustEntityFeatureEnv`-compatible adapter (or the real
    thing) -- anything `entity_token_features._topology` already accepts.
    """
    from catan_zero.rl.entity_token_features import _topology
    from catan_zero.search.neural_rust_mcts import _base_ports_by_id

    payload = env.observation_payload(actor_name, include_event_log=False)
    topology = _topology(payload)

    ports_by_id = _base_ports_by_id()
    max_port_id = max(ports_by_id) if ports_by_id else -1
    port_base_nodes: list[list[int]] = [[] for _ in range(max_port_id + 1)]
    for port_id, port in ports_by_id.items():
        nodes = [int(node) for node in port.get("nodes", ()) if 0 <= int(node) < 54]
        port_base_nodes[port_id] = nodes

    return RustTopology(
        hex_vertex_ids=np.asarray(topology["hex_vertex_ids"], dtype=np.int16),
        hex_edge_ids=np.asarray(topology["hex_edge_ids"], dtype=np.int16),
        edge_vertex_ids=np.asarray(topology["edge_vertex_ids"], dtype=np.int16),
        port_base_nodes=port_base_nodes,
    )


_FLOAT16_KEYS = (
    "hex_tokens",
    "vertex_tokens",
    "edge_tokens",
    "player_tokens",
    "global_tokens",
    "legal_action_tokens",
    "event_tokens",
)
_INT16_KEYS = ("legal_action_target_ids", "event_target_ids")
_BOOL_KEYS = ("hex_mask", "vertex_mask", "edge_mask", "player_mask", "legal_action_mask", "event_mask")

_FEATURE_SIZE_BY_KEY = {
    "hex_tokens": HEX_FEATURE_SIZE,
    "vertex_tokens": VERTEX_FEATURE_SIZE,
    "edge_tokens": EDGE_FEATURE_SIZE,
    "player_tokens": PLAYER_FEATURE_SIZE,
    "global_tokens": GLOBAL_FEATURE_SIZE,
    "legal_action_tokens": LEGAL_ACTION_FEATURE_SIZE,
    "event_tokens": EVENT_FEATURE_SIZE,
}


def _reshape_raw(raw: dict[str, Any], *, mask_has_shape: bool) -> dict[str, np.ndarray]:
    """Common (flat, shape) -> numpy reconstruction for both the single-item
    and batch Rust outputs. `mask_has_shape` distinguishes the two: the
    single-item path returns masks as plain flat `Vec<bool>` (always 1-D, no
    shape needed); the batch path returns `(flat, shape)` tuples for masks
    too, since they gain a leading batch dimension.
    """
    result: dict[str, np.ndarray] = {}
    for key, value in raw.items():
        if key == "widths":
            continue
        if key in _FLOAT16_KEYS:
            # Rust hands back raw little-endian f64 bytes (a `bytes` object,
            # one bulk copy) instead of a Python list of boxed floats --
            # `np.frombuffer` is a zero-copy VIEW over those bytes; `.astype`
            # is the SAME single f64->float16 cast the old list-based path
            # did, so this cannot change any value, only how it arrives.
            flat, shape = value
            arr = np.frombuffer(flat, dtype="<f8").reshape(shape).astype(np.float16)
        elif key in _INT16_KEYS:
            flat, shape = value
            arr = np.frombuffer(flat, dtype="<i8").reshape(shape).astype(np.int16)
        elif key in _BOOL_KEYS:
            if mask_has_shape:
                flat, shape = value
                arr = np.asarray(flat, dtype=np.bool_).reshape(shape)
            else:
                arr = np.asarray(value, dtype=np.bool_)
        else:  # pragma: no cover - defensive, every key above is enumerated.
            raise KeyError(f"unexpected key from Rust featurizer: {key!r}")
        result[key] = arr
    return result


def build_entity_features_rust(
    rust_game: Any,
    *,
    colors: tuple[str, ...],
    policy_action_ids: tuple[int, ...],
    action_size: int,
    topology: RustTopology,
    public_observation: bool = False,
) -> dict[str, np.ndarray]:
    """Rust-backed equivalent of `build_entity_token_features`'s output dict
    (minus the `"schema"` entry) for the Rust-MCTS adapter payload shape.

    `rust_game` is a live `catanatron_rs.Game`. `colors` MUST be the same
    fixed-order color tuple every caller already uses (e.g.
    `self.config.colors`) -- `Game` internally shuffles player order at
    creation, so it cannot be safely re-derived from live game state.
    `policy_action_ids` must be index-aligned to `rust_game.playable_actions`'
    own native order (no separate `legal_action_ids`/action-space-id
    translation happens on this side of the boundary -- see the module doc
    comment). Returns arrays with the SAME dtypes/shapes
    `build_entity_token_features` produces, so callers can swap one for the
    other behind a flag with no downstream changes.
    """
    import catanatron_rs

    raw = catanatron_rs.build_entity_features_flat(
        rust_game,
        list(colors),
        list(int(a) for a in policy_action_ids),
        int(action_size),
        topology.rust,
        public_observation,
    )

    result = _reshape_raw(raw, mask_has_shape=False)
    result["hex_vertex_ids"] = topology.hex_vertex_ids.copy()
    result["hex_edge_ids"] = topology.hex_edge_ids.copy()
    result["edge_vertex_ids"] = topology.edge_vertex_ids.copy()
    return result


def build_entity_features_batch_rust(
    rust_games: list[Any],
    *,
    colors: tuple[str, ...],
    policy_action_ids: list[tuple[int, ...]],
    action_size: int,
    topology: RustTopology,
    public_observation: bool = False,
    parallel: bool = False,
) -> tuple[dict[str, np.ndarray], list[int]]:
    """Batched companion to `build_entity_features_rust`: one call builds
    entity features for MANY games sharing the same board/colors (a Gumbel
    Sequential-Halving wave, a chance-node ROLL/robber/dev-card expansion,
    ...). Every array gains a leading batch dimension; the ragged
    `legal_action_*` arrays are padded (Rust-side) to the batch's own max
    legal-action width. Returns `(arrays, widths)` where `widths[i]` is the
    TRUE (unpadded) legal-action count for `rust_games[i]` -- required to
    mask off the padding downstream.

    `parallel=True` uses rayon inside the Rust call; leave it `False`
    (default) in single-core-pinned self-play workers to avoid
    oversubscription -- see `build_entity_features_batch`'s Rust-side doc
    comment for when to flip it on.

    Batch-1 fast path: skips the Rust batch machinery entirely and calls
    `build_entity_features_rust` directly, adding the leading batch
    dimension in Python -- batch-1 is ~60% of production call volume
    (speed-czar's profile), so this route must stay cheap.
    """
    if len(rust_games) != len(policy_action_ids):
        raise ValueError(
            f"rust_games and policy_action_ids must be the same length "
            f"(got {len(rust_games)} and {len(policy_action_ids)})"
        )

    if len(rust_games) == 1:
        single = build_entity_features_rust(
            rust_games[0],
            colors=colors,
            policy_action_ids=policy_action_ids[0],
            action_size=action_size,
            topology=topology,
            public_observation=public_observation,
        )
        return {key: value[None, ...] for key, value in single.items()}, [
            int(single["legal_action_mask"].shape[0])
        ]

    import catanatron_rs

    raw = catanatron_rs.build_entity_features_batch(
        list(rust_games),
        list(colors),
        [[int(a) for a in ids] for ids in policy_action_ids],
        int(action_size),
        topology.rust,
        public_observation,
        parallel,
    )
    widths = [int(w) for w in raw["widths"]]

    result = _reshape_raw(raw, mask_has_shape=True)
    batch_size = len(rust_games)
    result["hex_vertex_ids"] = np.broadcast_to(
        topology.hex_vertex_ids, (batch_size, *topology.hex_vertex_ids.shape)
    ).copy()
    result["hex_edge_ids"] = np.broadcast_to(
        topology.hex_edge_ids, (batch_size, *topology.hex_edge_ids.shape)
    ).copy()
    result["edge_vertex_ids"] = np.broadcast_to(
        topology.edge_vertex_ids, (batch_size, *topology.edge_vertex_ids.shape)
    ).copy()
    return result, widths
