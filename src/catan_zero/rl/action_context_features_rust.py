"""Rust-backed companion to `catan_zero.rl.action_features._context_vector`
for the per-leaf call site `neural_rust_mcts.rust_action_context_batch`
always uses (`valid=True` unconditionally for every legal action -- this is
NOT the general `build_action_context_feature_table` used for the full
action-space table).

Task #81 "context lever" (approved by team-lead as the next-highest-value
port after the entity-featurize swap: the JSON-snapshot/resolve/context-build
complex on the context path costs ~690us/leaf today, more than the entity
swap already landed, and porting it removes the LAST per-leaf
`json_snapshot()` caller). Purely ADDITIVE, same as the entity port: nothing
here changes the behavior of any existing function, and `catanatron_rs` is
imported lazily.

Reuses the SAME `entity_token_features_rust.RustTopology`/
`catanatron_rs.EntityTopology` object the entity featurizer already builds
once per board -- the context builder needs the identical fixed hex/edge
topology (for node adjacency and port lookups) and the identical
BASE-map-only caveat applies (see that module's doc comment). It needs
neither `colors` nor `action_size`: the acting player's own public VP and
(for MOVE_ROBBER) one other player's public VP by color are looked up
directly, never the full player list or the action catalog.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from catan_zero.rl.action_features import CONTEXT_ACTION_FEATURE_SIZE
from catan_zero.rl.entity_feature_adapter import (
    LEGACY_MISSING_CHECKPOINT_ADAPTER_VERSION,
    RUST_ENTITY_ADAPTER_V6,
    require_known_entity_feature_adapter,
)
from catan_zero.rl.entity_token_features_rust import RustTopology


def _require_native_context_adapter(catanatron_rs: Any, adapter_version: str) -> None:
    if adapter_version != RUST_ENTITY_ADAPTER_V6:
        return
    capability = getattr(
        catanatron_rs, "supported_action_context_adapter_versions", None
    )
    if not callable(capability):
        raise RuntimeError(
            "entity adapter v6 requires a catanatron_rs wheel with versioned "
            "opening-road action-context support"
        )
    supported = {str(version) for version in capability()}
    if adapter_version not in supported:
        raise RuntimeError(
            "catanatron_rs does not implement the requested action-context "
            f"adapter {adapter_version!r}; supported={sorted(supported)!r}"
        )


def build_action_context_rust(
    rust_game: Any,
    *,
    topology: RustTopology,
    entity_feature_adapter_version: str = LEGACY_MISSING_CHECKPOINT_ADAPTER_VERSION,
) -> np.ndarray:
    """Rust-backed equivalent of `rust_action_context_batch`'s per-game row
    block (i.e. `rust_action_context_batch(...)[0]`), for a single game.
    Returns `(n_legal, CONTEXT_ACTION_FEATURE_SIZE)` float32, matching the
    Python reference's dtype.
    """
    import catanatron_rs

    adapter_version = require_known_entity_feature_adapter(
        entity_feature_adapter_version
    )
    _require_native_context_adapter(catanatron_rs, adapter_version)
    if adapter_version == RUST_ENTITY_ADAPTER_V6:
        flat, shape = catanatron_rs.build_action_context_flat(
            rust_game, topology.rust, adapter_version
        )
    else:
        # Preserve compatibility with released wheels and the exact v2-v5 ABI.
        flat, shape = catanatron_rs.build_action_context_flat(
            rust_game, topology.rust
        )
    assert shape[1] == CONTEXT_ACTION_FEATURE_SIZE
    # `flat` is raw little-endian f64 bytes (a `bytes` object, one bulk copy)
    # -- `np.frombuffer` is a zero-copy view, `.astype` is the same single
    # f64->float32 cast the old list-based path did.
    return np.frombuffer(flat, dtype="<f8").reshape(shape).astype(np.float32)


def build_action_context_batch_rust(
    rust_games: list[Any],
    *,
    topology: RustTopology,
    entity_feature_adapter_version: str = LEGACY_MISSING_CHECKPOINT_ADAPTER_VERSION,
    parallel: bool = False,
) -> tuple[np.ndarray, list[int]]:
    """Batched companion to `build_action_context_rust`: one call builds
    context features for MANY games sharing the same board (a Gumbel
    Sequential-Halving wave, a chance-node expansion, ...). Returns
    `(context_tokens, widths)` where `context_tokens` has a leading batch
    dimension and the `legal_action` axis padded (Rust-side) to the batch's
    own max width, and `widths[i]` is the TRUE (unpadded) legal-action count
    for `rust_games[i]`.

    Batch-1 fast path: skips the Rust batch call entirely (same rationale as
    `entity_token_features_rust.build_entity_features_batch_rust`).
    """
    if len(rust_games) == 1:
        single = build_action_context_rust(
            rust_games[0],
            topology=topology,
            entity_feature_adapter_version=entity_feature_adapter_version,
        )
        return single[None, ...], [int(single.shape[0])]

    import catanatron_rs

    adapter_version = require_known_entity_feature_adapter(
        entity_feature_adapter_version
    )
    _require_native_context_adapter(catanatron_rs, adapter_version)
    if adapter_version == RUST_ENTITY_ADAPTER_V6:
        raw = catanatron_rs.build_action_context_batch(
            list(rust_games), topology.rust, parallel, adapter_version
        )
    else:
        raw = catanatron_rs.build_action_context_batch(
            list(rust_games), topology.rust, parallel
        )
    widths = [int(w) for w in raw["widths"]]
    flat, shape = raw["context_tokens"]
    arr = np.frombuffer(flat, dtype="<f8").reshape(shape).astype(np.float32)
    return arr, widths
