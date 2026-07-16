"""Versioned semantic contract between entity checkpoints and feature adapters.

Tensor shapes alone are not a sufficient compatibility boundary for
``EntityGraphPolicy``.  A feature adapter can keep every shape unchanged while
changing the meaning of an input slot (for example, populating the historically
zero ``has_longest_road`` bit).  Such a change is an input-distribution change,
not a bug fix that can safely be applied underneath already-trained weights.

This module is deliberately dependency-free so checkpoint loading, training,
and Rust-MCTS evaluation can share one registry without import cycles.  The
registry is append-only: an adapter implementation may add a new version, but
must never change the semantics attached to an existing version string.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping


ENTITY_FEATURE_ADAPTER_CHECKPOINT_SCHEMA = "entity-feature-adapter-v1"

# This literal is intentionally named and pinned separately from CURRENT.
# Checkpoints predating adapter metadata (including the deployed f7 lineage)
# are mapped to this exact historical contract.  If CURRENT advances to v3,
# missing metadata must continue to resolve to v2 rather than silently adopting
# the future default.
RUST_ENTITY_ADAPTER_V2 = (
    "rust_entity_adapter_v2_land_topology_ports_maritime"
)
RUST_ENTITY_ADAPTER_V3 = (
    "rust_entity_adapter_v3_structured_action_resources"
)
RUST_ENTITY_ADAPTER_V4 = (
    "rust_entity_adapter_v4_actor_public_rule_state"
)
LEGACY_MISSING_CHECKPOINT_ADAPTER_VERSION = RUST_ENTITY_ADAPTER_V2
CURRENT_RUST_ENTITY_ADAPTER_VERSION = RUST_ENTITY_ADAPTER_V3
IMPLEMENTED_RUST_ENTITY_ADAPTER_VERSIONS = frozenset(
    {RUST_ENTITY_ADAPTER_V2, RUST_ENTITY_ADAPTER_V3, RUST_ENTITY_ADAPTER_V4}
)


@dataclass(frozen=True, slots=True)
class EntityFeatureAdapterSpec:
    """Immutable slot-semantics description for one adapter version."""

    version: str
    player_has_longest_road: str
    trade_action_type_one_hot: str
    trade_prompt_one_hot: str
    trade_panel: str
    context_trade_totals: str
    topology: str
    event_history: str
    structured_action_resources: str
    actor_public_rule_state: str


# These strings are executable documentation: tests and checkpoint/runtime
# validation bind to the version, while this descriptor records why a seemingly
# harmless feature correction requires a new version.
ENTITY_FEATURE_ADAPTER_SPECS: Mapping[str, EntityFeatureAdapterSpec] = (
    MappingProxyType(
        {
            RUST_ENTITY_ADAPTER_V2: EntityFeatureAdapterSpec(
                version=RUST_ENTITY_ADAPTER_V2,
                player_has_longest_road="constant_false",
                trade_action_type_one_hot="legacy_case_sensitive_miss",
                trade_prompt_one_hot="legacy_prompt_name_miss",
                trade_panel="offers_remaining_zero_current_offer_none",
                context_trade_totals="legacy_maritime_list_cardinality",
                topology="base_layout_static_lookup",
                event_history="empty",
                structured_action_resources="legacy_yop_and_singular_identity_omitted",
                actor_public_rule_state="historical_zero_slots",
            ),
            RUST_ENTITY_ADAPTER_V3: EntityFeatureAdapterSpec(
                version=RUST_ENTITY_ADAPTER_V3,
                player_has_longest_road="constant_false",
                trade_action_type_one_hot="legacy_case_sensitive_miss",
                trade_prompt_one_hot="legacy_prompt_name_miss",
                trade_panel="offers_remaining_zero_current_offer_none",
                context_trade_totals="legacy_maritime_list_cardinality",
                topology="base_layout_static_lookup",
                event_history="empty",
                structured_action_resources=(
                    "yop_bundle_and_discard_monopoly_singular_identity"
                ),
                actor_public_rule_state="historical_zero_slots",
            ),
            RUST_ENTITY_ADAPTER_V4: EntityFeatureAdapterSpec(
                version=RUST_ENTITY_ADAPTER_V4,
                player_has_longest_road="authoritative_public_state",
                trade_action_type_one_hot="legacy_case_sensitive_miss",
                trade_prompt_one_hot="legacy_prompt_name_miss",
                trade_panel="offers_remaining_zero_current_offer_none",
                context_trade_totals="legacy_maritime_list_cardinality",
                topology="base_layout_static_lookup",
                event_history="meaningful_public_history_v1_when_enabled",
                structured_action_resources=(
                    "yop_bundle_and_discard_monopoly_singular_identity"
                ),
                actor_public_rule_state=(
                    "dev_used_road_building_free_roads_discard_remainder_"
                    "playable_dev_counts"
                ),
            ),
        }
    )
)


class EntityFeatureAdapterContractError(ValueError):
    """Raised when checkpoint and runtime feature semantics are not provable."""


def require_known_entity_feature_adapter(version: object) -> str:
    resolved = str(version or "")
    if resolved not in ENTITY_FEATURE_ADAPTER_SPECS:
        raise EntityFeatureAdapterContractError(
            f"unknown entity feature adapter version {resolved!r}; known versions: "
            f"{sorted(ENTITY_FEATURE_ADAPTER_SPECS)}. Refusing to infer tensor "
            "semantics from shapes or from the current runtime default."
        )
    return resolved


def checkpoint_entity_feature_adapter_metadata(version: object) -> dict[str, str]:
    """Return the append-only metadata stored beside model weights."""

    return {
        "schema_version": ENTITY_FEATURE_ADAPTER_CHECKPOINT_SCHEMA,
        "version": require_known_entity_feature_adapter(version),
    }


def resolve_checkpoint_entity_feature_adapter(
    raw: object,
    *,
    metadata_present: bool,
) -> tuple[str, str]:
    """Resolve checkpoint metadata to ``(version, provenance_source)``.

    Missing metadata has one narrow compatibility mapping for checkpoints that
    predate this contract.  Malformed, partial, or unknown metadata never falls
    through to that mapping: once the key exists it must validate completely.
    """

    if not metadata_present:
        return (
            LEGACY_MISSING_CHECKPOINT_ADAPTER_VERSION,
            "legacy_missing_metadata_explicit_v2_mapping",
        )
    if not isinstance(raw, Mapping):
        raise EntityFeatureAdapterContractError(
            "checkpoint entity_feature_adapter metadata must be a mapping"
        )
    schema = str(raw.get("schema_version", "") or "")
    if schema != ENTITY_FEATURE_ADAPTER_CHECKPOINT_SCHEMA:
        raise EntityFeatureAdapterContractError(
            "unsupported checkpoint entity_feature_adapter schema "
            f"{schema!r}; expected {ENTITY_FEATURE_ADAPTER_CHECKPOINT_SCHEMA!r}"
        )
    version = require_known_entity_feature_adapter(raw.get("version"))
    return version, "checkpoint_metadata"


def policy_entity_feature_adapter_version(policy: Any) -> str:
    """Read a policy binding, with only the explicit pre-metadata mapping.

    Real ``EntityGraphPolicy`` instances always set the attribute.  The fallback
    keeps old in-memory policy stubs and legacy loaders compatible, but is pinned
    to the historical version rather than CURRENT.
    """

    if hasattr(policy, "entity_feature_adapter_version"):
        return require_known_entity_feature_adapter(
            getattr(policy, "entity_feature_adapter_version")
        )
    return LEGACY_MISSING_CHECKPOINT_ADAPTER_VERSION
