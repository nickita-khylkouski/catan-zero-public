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

import json
from typing import Any

import numpy as np

from catan_zero.deduction_tracker import (
    DEDUCTION_FEATURE_SIZE,
    DEDUCTION_FEATURES_KEY,
    public_card_count_feature_table,
)
from catan_zero.rl.entity_feature_adapter import (
    CURRENT_RUST_ENTITY_ADAPTER_VERSION,
    RUST_ENTITY_ADAPTER_V5,
    RUST_ENTITY_ADAPTER_V6,
    require_known_entity_feature_adapter,
)
from catan_zero.rl.meaningful_history import (
    MEANINGFUL_PUBLIC_HISTORY_SCHEMA_V2,
    MEANINGFUL_PUBLIC_HISTORY_SCHEMA_VERSION,
    meaningful_public_history_limit,
)
from catan_zero.rl.entity_token_features import (
    EDGE_FEATURE_SIZE,
    EVENT_FEATURE_SIZE,
    GLOBAL_FEATURE_SIZE,
    HEX_FEATURE_SIZE,
    LEGAL_ACTION_FEATURE_SIZE,
    PLAYER_FEATURE_SIZE,
    VERTEX_FEATURE_SIZE,
    public_card_count_features_from_entity_tokens,
)


_REQUIRED_NATIVE_FEATURE_APIS = (
    "EntityTopology",
    "build_entity_features_flat",
    "build_action_context_flat",
)
_REQUIRED_NATIVE_FEATURE_CAPABILITIES = frozenset(
    {
        "public_award_feature_parity",
        "entity_feature_adapter_version",
    }
)


def _missing_required_feature_capabilities(module: Any) -> set[str]:
    capability_fn = getattr(module, "gumbel_search_capabilities", None)
    if not callable(capability_fn):
        return set(_REQUIRED_NATIVE_FEATURE_CAPABILITIES)
    return set(_REQUIRED_NATIVE_FEATURE_CAPABILITIES) - set(capability_fn())


def _has_required_feature_capability(module: Any) -> bool:
    return not _missing_required_feature_capabilities(module)


def rust_feature_path_available() -> bool:
    """Return whether the installed wheel supports the complete leaf path."""
    try:
        import catanatron_rs
    except ImportError:
        return False
    return all(
        callable(getattr(catanatron_rs, name, None))
        for name in _REQUIRED_NATIVE_FEATURE_APIS
    ) and _has_required_feature_capability(catanatron_rs)


def require_rust_feature_path() -> None:
    """Fail closed unless both native entity and context builders exist."""
    try:
        import catanatron_rs
    except ImportError as error:
        raise RuntimeError(
            "Rust feature path requested but catanatron_rs is not importable"
        ) from error
    missing = [
        name
        for name in _REQUIRED_NATIVE_FEATURE_APIS
        if not callable(getattr(catanatron_rs, name, None))
    ]
    if missing:
        raise RuntimeError(
            "Rust feature path requested but the installed catanatron_rs wheel "
            f"is missing {', '.join(missing)}; refusing Python fallback"
        )
    missing_capabilities = _missing_required_feature_capabilities(catanatron_rs)
    if missing_capabilities:
        reasons = []
        if "public_award_feature_parity" in missing_capabilities:
            reasons.append("stale public-award semantics")
        if "entity_feature_adapter_version" in missing_capabilities:
            reasons.append("stale entity-feature adapter ABI")
        raise RuntimeError(
            "Rust feature path requested but the installed catanatron_rs wheel "
            "does not advertise "
            f"{', '.join(sorted(missing_capabilities))}; refusing a native "
            f"featurizer with {' and '.join(reasons)}"
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

    __slots__ = (
        "hex_vertex_ids",
        "hex_edge_ids",
        "edge_vertex_ids",
        "port_base_nodes",
        "rust",
    )

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
_BOOL_KEYS = (
    "hex_mask",
    "vertex_mask",
    "edge_mask",
    "player_mask",
    "legal_action_mask",
    "event_mask",
)

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
    meaningful_public_history: bool = False,
    history_limit: int = 64,
    meaningful_public_history_schema: str = MEANINGFUL_PUBLIC_HISTORY_SCHEMA_VERSION,
    entity_feature_adapter_version: str = CURRENT_RUST_ENTITY_ADAPTER_VERSION,
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

    adapter_version = require_known_entity_feature_adapter(
        entity_feature_adapter_version
    )
    if meaningful_public_history:
        maximum_history_limit = meaningful_public_history_limit(
            meaningful_public_history_schema
        )
        if not 1 <= int(history_limit) <= maximum_history_limit:
            raise ValueError(
                "meaningful public-history limit outside schema contract: "
                f"{history_limit} not in [1, {maximum_history_limit}]"
            )
        if (
            meaningful_public_history_schema == MEANINGFUL_PUBLIC_HISTORY_SCHEMA_V2
        ) != (
            adapter_version in {RUST_ENTITY_ADAPTER_V5, RUST_ENTITY_ADAPTER_V6}
        ):
            raise ValueError(
                "meaningful public-history v2 and entity adapter v5/v6 must be "
                "enabled together"
            )
    raw = catanatron_rs.build_entity_features_flat(
        rust_game,
        list(colors),
        list(int(a) for a in policy_action_ids),
        int(action_size),
        topology.rust,
        public_observation,
        meaningful_public_history,
        int(history_limit),
        adapter_version,
    )

    result = _reshape_raw(raw, mask_has_shape=False)
    # `build_entity_token_features` always includes this additive public
    # tensor, regardless of whether the loaded checkpoint currently consumes
    # its optional residual.  The native path must preserve the same complete
    # entity-batch contract: checkpoint configuration controls model
    # consumption, not whether a featurizer silently drops a field.  Derive it
    # from the same public token surface used to backfill historical corpora;
    # the exact native JSON audit API would create train/serve skew on legacy
    # clipped counts and add serialization to every MCTS leaf.
    result[DEDUCTION_FEATURES_KEY] = (
        public_card_count_features_from_entity_tokens(
            result["player_tokens"],
            result["global_tokens"],
            entity_feature_adapter_version=adapter_version,
        )
    )
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
    meaningful_public_history: bool = False,
    history_limit: int = 64,
    meaningful_public_history_schema: str = MEANINGFUL_PUBLIC_HISTORY_SCHEMA_VERSION,
    entity_feature_adapter_version: str = CURRENT_RUST_ENTITY_ADAPTER_VERSION,
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
            meaningful_public_history=meaningful_public_history,
            history_limit=history_limit,
            meaningful_public_history_schema=meaningful_public_history_schema,
            entity_feature_adapter_version=entity_feature_adapter_version,
        )
        return {key: value[None, ...] for key, value in single.items()}, [
            int(single["legal_action_mask"].shape[0])
        ]

    import catanatron_rs

    adapter_version = require_known_entity_feature_adapter(
        entity_feature_adapter_version
    )
    if meaningful_public_history:
        maximum_history_limit = meaningful_public_history_limit(
            meaningful_public_history_schema
        )
        if not 1 <= int(history_limit) <= maximum_history_limit:
            raise ValueError(
                "meaningful public-history limit outside schema contract: "
                f"{history_limit} not in [1, {maximum_history_limit}]"
            )
        if (
            meaningful_public_history_schema == MEANINGFUL_PUBLIC_HISTORY_SCHEMA_V2
        ) != (
            adapter_version in {RUST_ENTITY_ADAPTER_V5, RUST_ENTITY_ADAPTER_V6}
        ):
            raise ValueError(
                "meaningful public-history v2 and entity adapter v5/v6 must be "
                "enabled together"
            )
    raw = catanatron_rs.build_entity_features_batch(
        list(rust_games),
        list(colors),
        [[int(a) for a in ids] for ids in policy_action_ids],
        int(action_size),
        topology.rust,
        public_observation,
        parallel,
        meaningful_public_history,
        int(history_limit),
        adapter_version,
    )
    widths = [int(w) for w in raw["widths"]]

    result = _reshape_raw(raw, mask_has_shape=True)
    result[DEDUCTION_FEATURES_KEY] = (
        public_card_count_features_from_entity_tokens(
            result["player_tokens"],
            result["global_tokens"],
            entity_feature_adapter_version=adapter_version,
        )
    )
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


def _public_card_count_features_from_rust_game(
    rust_game: Any,
    *,
    colors: tuple[str, ...],
) -> np.ndarray:
    """Construct exact audit features from the native public-only boundary.

    The native API itself performs two-player conservation and never serializes
    opponent resource/dev identities or deck order. Requiring that API here is
    intentional for contract tests and offline schema audits. The live model
    path above does not call this JSON helper: v2 serving must match the legacy
    entity-token backfill used by its training corpus and stay off the MCTS
    serialization hot path.
    """

    actor = str(rust_game.current_color())
    if actor not in colors:
        return np.zeros((4, DEDUCTION_FEATURE_SIZE), dtype=np.float32)
    if len(colors) != 2 or not hasattr(rust_game, "public_card_deductions_json"):
        raise RuntimeError(
            "public card-count features require the 2p native "
            "public_card_deductions_json capability"
        )
    deduction = json.loads(rust_game.public_card_deductions_json(actor))
    if (
        deduction.get("contract") != "public_card_deductions_2p_v1"
        or str(deduction.get("observer", "")) != actor
    ):
        raise RuntimeError("native public card-deduction contract drift")
    opponent = str(deduction.get("opponent", ""))
    if opponent not in colors or opponent == actor:
        raise RuntimeError("native public card-deduction opponent drift")

    actor_resources = dict(deduction.get("observer_resources", {}))
    actor_dev_cards = dict(deduction.get("observer_development_cards", {}))
    public_plays = dict(deduction.get("publicly_played_development_cards", {}))
    players: dict[str, dict[str, Any]] = {
        actor: {
            "resource_card_count": sum(map(int, actor_resources.values())),
            "development_card_count": int(
                deduction.get("observer_development_card_count", 0)
            ),
            "resources": actor_resources,
            "development_cards": actor_dev_cards,
            # The feature builder only sums these counters over all players.
            # Assigning the public aggregate to one row avoids reintroducing
            # per-player engine state while preserving the exact posterior.
            "played_development_cards": public_plays,
        },
        opponent: {
            "resource_card_count": int(
                deduction.get("opponent_resource_card_count", 0)
            ),
            "development_card_count": int(
                deduction.get("opponent_face_down_development_card_count", 0)
            ),
            "played_development_cards": {},
        },
    }
    payload = {
        "players": players,
        "bank": {"resources": dict(deduction.get("resource_bank", {}))},
    }
    return public_card_count_feature_table(payload, actor)
