from __future__ import annotations

from dataclasses import dataclass
import math
import operator
from pathlib import Path
from typing import Any

import numpy as np

from catan_zero.deduction_tracker import (
    DEDUCTION_FEATURE_SIZE,
    DEDUCTION_FEATURES_KEY,
    PUBLIC_CARD_COUNT_FEATURE_SCHEMA_VERSION,
)
from catan_zero.rl.action_features import (
    CONTEXT_ACTION_FEATURE_SIZE,
    build_action_context_feature_table,
)
from catan_zero.rl.entity_feature_adapter import (
    CURRENT_RUST_ENTITY_ADAPTER_VERSION,
    checkpoint_entity_feature_adapter_metadata,
    require_known_entity_feature_adapter,
    resolve_checkpoint_entity_feature_adapter,
)
from catan_zero.rl.entity_token_features import (
    EDGE_FEATURE_SIZE,
    EVENT_FEATURE_SIZE,
    GLOBAL_FEATURE_SIZE,
    HEX_FEATURE_SIZE,
    LEGAL_ACTION_FEATURE_SIZE,
    PLAYER_FEATURE_SIZE,
    VERTEX_FEATURE_SIZE,
    build_entity_token_features,
    mask_player_tokens_public,
)
from catan_zero.rl.multiagent_env import ColonistMultiAgentConfig, ColonistMultiAgentEnv
from catan_zero.rl.meaningful_history import (
    MEANINGFUL_PUBLIC_HISTORY_LIMIT,
    MEANINGFUL_PUBLIC_HISTORY_SCHEMA_VERSION,
)
from catan_zero.rl.ordered_history import (
    MASKED_MEAN_V1,
    ORDERED_ATTENTION_V2,
    SUPPORTED_HISTORY_POOLING,
    build_ordered_history_pool,
)
from catan_zero.rl.torch_ppo import build_action_feature_table
from catan_zero.rl.xdim_lite_policy import (
    _array_sha256,
    _install_numpy_pickle_aliases,
    _resolve_device,
)


ENTITY_POLICY_SCHEMA_VERSION = "entity_graph_policy_v1"

# The f7 incumbent was trained and served with a 64-row event-token surface.
# Those rows were masked from its mature state trunk, but their *presence* is
# still part of the numerical graph: changing the attention sequence from 64
# rows to the 32-row meaningful-history window changes floating-point reduction
# order even when the new residual gate is exactly zero.  Meaningful history is
# therefore a side input while the inherited trunk retains this legacy width.
_LEGACY_EVENT_HISTORY_WIDTH = 64

# Player-token slot 12 was accidentally constant zero in every historical
# Python and native corpus through catanatron_rs 0.1.7.  Consequently the
# corresponding input column in all legacy entity-graph checkpoints is still
# its random initializer (it received no gradient).  Feeding the authoritative
# longest-road bit to such a checkpoint changes its function out of
# distribution.  The contract is checkpoint-owned: old/missing metadata keeps
# the historical zero input, while a model trained on corrected features may
# explicitly attest authoritative_v1.
PUBLIC_AWARD_FEATURE_CONTRACT_LEGACY_ZERO = "legacy_zero_v0"
PUBLIC_AWARD_FEATURE_CONTRACT_AUTHORITATIVE = "authoritative_v1"
PUBLIC_AWARD_FEATURE_CONTRACTS = frozenset(
    {
        PUBLIC_AWARD_FEATURE_CONTRACT_LEGACY_ZERO,
        PUBLIC_AWARD_FEATURE_CONTRACT_AUTHORITATIVE,
    }
)
PLAYER_LONGEST_ROAD_SLOT = 12


def _validate_public_award_feature_contract(contract: object) -> str:
    resolved = str(contract or "")
    if resolved not in PUBLIC_AWARD_FEATURE_CONTRACTS:
        raise ValueError(
            "unsupported public_award_feature_contract "
            f"{resolved!r}; expected one of {sorted(PUBLIC_AWARD_FEATURE_CONTRACTS)}"
        )
    return resolved


def _apply_public_award_feature_contract(
    entity_batch: dict[str, np.ndarray], contract: str
) -> dict[str, np.ndarray]:
    """Return a function-compatible batch for the checkpoint's award contract.

    The legacy bridge deliberately changes only player-token slot 12 and never
    mutates the caller's batch.  Largest-army (slot 11), road length (slot 14),
    hidden-information masks, and every other entity tensor remain untouched.
    """

    resolved = _validate_public_award_feature_contract(contract)
    if resolved == PUBLIC_AWARD_FEATURE_CONTRACT_AUTHORITATIVE:
        return entity_batch
    if "player_tokens" not in entity_batch:
        return entity_batch
    player_tokens = np.asarray(entity_batch["player_tokens"])
    if player_tokens.ndim != 3 or player_tokens.shape[-1] <= PLAYER_LONGEST_ROAD_SLOT:
        raise ValueError(
            "player_tokens must be a batched (B, P, F) tensor with longest-road "
            f"slot {PLAYER_LONGEST_ROAD_SLOT}; got {player_tokens.shape}"
        )
    # Avoid an allocation for the overwhelmingly common legacy-corpus case.
    if not np.any(player_tokens[..., PLAYER_LONGEST_ROAD_SLOT]):
        return entity_batch
    bridged = dict(entity_batch)
    bridged_players = np.array(player_tokens, copy=True)
    bridged_players[..., PLAYER_LONGEST_ROAD_SLOT] = 0
    bridged["player_tokens"] = bridged_players
    return bridged


# Fixed board sizes on the standard Catan map: 54 intersections (settlement/city
# nodes, catanatron node_id 0-53) and 19 hexes (robber targets, hex id 0-18).
# These match the per-type target-id space in EntityGraphNet._gather_target_tokens
# and the shape asserts in _assert_entity_batch_shapes; the CAT-100 categorical
# aux heads emit over exactly these index spaces.
AUX_NUM_INTERSECTIONS = 54
AUX_NUM_HEXES = 19


def _entity_token_start_offsets(batch: dict[str, Any]) -> tuple[int, int, int, int]:
    """Starts of hex/vertex/edge/player spans in the live CLS-prefixed layout."""

    n_hex = int(batch["hex_tokens"].shape[1])
    n_vertex = int(batch["vertex_tokens"].shape[1])
    n_edge = int(batch["edge_tokens"].shape[1])
    return (
        1,
        1 + n_hex,
        1 + n_hex + n_vertex,
        1 + n_hex + n_vertex + n_edge,
    )


_NON_MODEL_ENTITY_KEYS = frozenset(
    {
        "hex_vertex_ids",
        "hex_edge_ids",
        "edge_vertex_ids",
        "event_target_ids",
        "legal_action_mask",
    }
)
_RELATIONAL_TOPOLOGY_KEYS = frozenset(
    {"hex_vertex_ids", "hex_edge_ids", "edge_vertex_ids", "event_target_ids"}
)

# The serialized 45-column action table was historically dead in EntityGraphNet.
# Columns 19:41 are the genuinely missing surface: numeric board arguments,
# resource-flow identities, and target-player identity. The remaining columns
# duplicate legal/context tokens, so the repair intentionally excludes them.
STATIC_ACTION_RESIDUAL_SLICE = slice(19, 41)
STATIC_ACTION_RESIDUAL_FEATURE_SIZE = 22


@dataclass(frozen=True, slots=True)
class EntityGraphConfig:
    action_size: int
    static_action_feature_size: int
    context_action_feature_size: int = CONTEXT_ACTION_FEATURE_SIZE
    legal_action_feature_size: int = LEGAL_ACTION_FEATURE_SIZE
    hidden_size: int = 640
    state_layers: int = 6
    attention_heads: int = 8
    dropout: float = 0.05
    action_mask_version: str = ""
    schema_version: str = ENTITY_POLICY_SCHEMA_VERSION
    # Optional value-uncertainty auxiliary head (KataGo short-term-error style):
    # a scalar per state predicting the value head's own squared error. Default
    # False keeps the parameter count and forward outputs bit-identical to models
    # built before this field existed (older checkpoints deserialize with the
    # default, so they load unchanged; see EntityGraphNet.load allowed-missing
    # prefixes for the reverse direction). The prediction is trained by
    # train_bc.py --value-uncertainty-loss-weight. The head reads a stop-gradient
    # (detached) copy of the trunk state so its loss never distorts value/trunk
    # learning (see forward()). Consuming it inside search is opt-in and
    # flag-gated: EntityGraphRustEvaluatorConfig.emit_uncertainty surfaces this
    # scalar to the searcher, and GumbelChanceMCTSConfig.uncertainty_backup_weighting
    # turns it into KataGo-style capped backup weights (both default OFF, CAT-61).
    value_uncertainty_head: bool = False
    # --- Action-attention architecture upgrade (f69), all default OFF ---
    # When every flag below is off, the module is structurally and
    # behaviorally bit-identical to the pre-upgrade net: no new parameters are
    # created, so existing checkpoints load with strict=True. Old pickled
    # configs predate these slots; the module reads them via getattr(...,
    # default) so those checkpoints still construct.
    #
    # (1) Gather post-trunk board tokens for each action's target entities
    #     (legal_action_target_ids) and add a zero-initialised projection of
    #     the pooled result into encoded_actions.
    action_target_gather: bool = False
    # (2) N post-trunk cross-attention blocks: encoded_actions query the final
    #     board tokens. Output projection + FFN are zero-initialised so the
    #     block is an exact identity at init (warm-start safety).
    action_cross_attention_layers: int = 0
    # (3) A learned probe token cross-attends over all output tokens; a
    #     zero-initialised head consuming [CLS ++ probe_output] (2h) is ADDED
    #     to the value, so value is unchanged at init.
    value_attention_pool: bool = False
    # --- Distributional (HL-Gauss categorical) value head (CAT-39), default OFF ---
    # A MuZero/C51-shaped categorical value head over a uniform support on the
    # win-loss axis [-1, 1], trained with HL-Gauss cross-entropy (Farebrother et
    # al. 2024, arXiv:2403.03950 "Stop Regressing"): scalar targets are projected
    # to a Gaussian-smeared histogram (sigma ~ bin width) rather than the two-hot
    # encoding a plain C51 head uses -- two-hot underperforms MSE, HL-Gauss beats
    # it, and the gap is largest under stochastic dynamics (the whole reason this
    # head exists for Catan). Per the CAT-39 R9 ruling the PRIMARY support is
    # win/loss ONLY plus one distinct TRUNCATION class (VP-margin is removed from
    # the joint support and lives on a separate auxiliary head): the head emits
    # ``value_categorical_bins`` win-loss logits followed by, when
    # ``value_categorical_truncation_class`` is set, one extra truncation logit.
    # New output keys ("value_categorical_logits", "value_categorical" = the
    # calibrated win-value expectation over the win-loss bins renormalised to
    # exclude truncation mass, "value_categorical_truncation_prob"); new params
    # under value_categorical_head.*. The scalar MSE value head stays the value
    # consumed by search/eval (bit-identical), so warm-starting an old checkpoint
    # with these flags ON leaves every existing output unchanged. 0 disables (no
    # new params). NOTE: appended LAST on purpose -- this frozen+slots dataclass
    # pickles positionally, so new fields must only ever be appended.
    value_categorical_bins: int = 0
    value_categorical_truncation_class: bool = True
    # --- CAT-97 GATEAU edge/node-feature policy head (default OFF) ---
    # AlphaGateau (arXiv 2410.23753) reads each move's POLICY LOGIT directly
    # from that move's edge/node feature: policy_logit(move) = MLP(edge_feat).
    # Here every legal action already carries its TARGET entity tokens (edge
    # token for a road, intersection/vertex token for a settlement/city, hex
    # for the robber -- the same fixed [B,A,4] `legal_action_target_ids`
    # mapping used by action_target_gather). When on, we mean-pool those
    # post-trunk target tokens per action and add a DIRECT per-action logit
    # MLP(pooled_target) to the CLIP-style logits. The final Linear is
    # zero-initialised, so the logit (and every other output) is bit-identical
    # to the pre-flag net at init -- warm-start safe. This is the topology-
    # aligned, size-agnostic policy format from the paper; it differs from
    # action_target_gather (which instead modulates the CLIP action embedding)
    # by emitting a stand-alone logit term and can be used with or without it.
    # NOTE: appended LAST -- this frozen+slots dataclass pickles positionally.
    edge_policy_head: bool = False
    # --- CAT-100 Catan-native auxiliary subgoal heads (default OFF) ---
    # UNREAL-style auxiliary prediction heads (Jaderberg et al. 2016,
    # arXiv 1611.05397) off the SHARED pooled state (CLS) token, trained with a
    # small loss weight alongside value/policy. They emit into the outputs dict
    # ONLY and never touch logits/value/final_vp/q_values, so a model built
    # with them enabled is bit-identical in its value/policy outputs to one
    # built without -- warm-start safe by construction (the aux heads simply
    # start random and train from the auxiliary loss). Targets are free labels
    # from the catanatron engine (see rl/aux_subgoal_targets.py):
    #   aux_longest_road / aux_largest_army : binary "actor holds it at
    #       horizon" (BCE-with-logits),
    #   aux_vp_in_n     : scalar actor VP gain over `aux_vp_horizon` plies (MSE),
    #   aux_next_settlement : categorical over 54 intersections (cross-entropy),
    #   aux_robber_target   : categorical over 19 hexes (cross-entropy).
    aux_subgoal_heads: bool = False
    # Horizon (plies) for the aux_vp_in_n target. Metadata only -- the head is a
    # plain scalar regressor; the horizon lives here so a checkpoint records the
    # target definition it was trained against.
    aux_vp_horizon: int = 8
    # --- R&D topology-aware state trunks (default incumbent preserved) ---
    # ``transformer`` is the historical dense set Transformer. ``rrt`` uses
    # directed Catan-incidence attention with an R/R/T-style local/global block
    # pattern. ``resrgcn`` is the no-attention residual relational GNN control.
    # All relational fields are inert under ``transformer`` so old checkpoint
    # configs and the default state_dict remain exactly compatible.
    state_trunk: str = "transformer"
    # Empty selects the production-shaped default: RRT repeats ``RRT`` to the
    # requested state_layers; ResRGCN uses one graph block per layer. An explicit
    # pattern is accepted only by RRT and must contain exactly state_layers R/Ts.
    relational_block_pattern: str = ""
    # 0 selects 1024 for width-384 RRT and 512 for width-384 ResRGCN, scaling
    # proportionally for width sweeps. Keeping it explicit in the checkpoint
    # makes parameter/compute matching reproducible.
    relational_ff_size: int = 0
    relational_bases: int = 4
    # Relational models bind every legal move to its board target and include a
    # from-scratch action-to-board decoder. This count is separate from the
    # warm-start-only action_cross_attention_layers incumbent experiment.
    relational_action_cross_layers: int = 1
    # --- E3 fixed-K shared latent deliberation (default OFF) ---
    # A small learned plan set repeatedly cross-attends to the encoded board
    # through one shared block. Increasing K changes compute, not parameter
    # count. This is fixed-depth latent computation; the emitted halt logit is
    # diagnostic and does not claim adaptive execution.
    latent_deliberation_steps: int = 0
    latent_deliberation_slots: int = 8
    # --- E4 sparse conditional FFN capacity (default OFF) ---
    # Replaces global RRT block FFNs with one shared expert plus N routed
    # experts. Only top-k selected experts execute for each live token.
    moe_routed_experts: int = 0
    moe_top_k: int = 2
    # 0 selects width 384 at model width 384 and scales proportionally.
    moe_expert_ff_size: int = 0
    # Relational trunks historically bundled CAT-97's direct target-token
    # policy logit. Keep that behavior by default, but allow a causal
    # relational/gather/cross-attention probe to exclude the extra head.
    relational_edge_policy_head: bool = True
    # Minimal function-preserving topology warm-start. Unlike state_trunk=rrt,
    # this retains every incumbent Transformer parameter and inserts one
    # zero-output incidence message-passing residual before the historical
    # blocks. Default OFF preserves old checkpoint structure exactly. Appended
    # last because this frozen+slots dataclass pickles positionally.
    topology_residual_adapter: bool = False
    # Training-only privileged-label belief auxiliary (default OFF).  The head
    # reads each post-trunk *public-masked* player token and predicts a
    # five-resource composition simplex.  Its labels may be extracted from the
    # pre-mask banked player tokens, but those labels are never model inputs.
    # Appended last for positional pickle compatibility.
    belief_resource_head: bool = False
    # The legacy next-settlement head classifies an absolute vertex id from the
    # permutation-invariant CLS token. Vertex tokens carry no id/coordinate, so
    # CLS cannot bind a class to its board token. This opt-in repair emits one
    # shared pointer score per post-trunk vertex token. The robber head remains
    # a dense CLS classifier because hex tokens carry canonical coordinates.
    # Default OFF preserves historical checkpoints and results exactly.
    # Appended last for positional pickle compatibility.
    aux_settlement_pointer_head: bool = False
    # Function-preserving repair for the historically dead static action table.
    # A zero-output projection consumes only the nonredundant catalog columns.
    # Default OFF preserves legacy checkpoint structure and behavior exactly.
    static_action_residual: bool = False
    # Public-only card-count input. The compact 11-column tensor is projected
    # as a zero-output residual onto the matching player rows, so enabling this
    # on an old checkpoint is an exact function-preserving warm start. It uses
    # the same public entity-token transform in training and serving; 2p
    # conservation identifies opponent resources whenever legacy counters are
    # unsaturated, while dev identities remain a hypergeometric posterior.
    # Appended last for positional pickle compatibility.
    public_card_count_features: bool = False
    public_card_count_feature_schema: str = PUBLIC_CARD_COUNT_FEATURE_SCHEMA_VERSION
    # Reuse the existing event-token encoder with a bounded public-only event
    # selection. Old corpora masked every event row, so enabling this also
    # creates a zero-gated pooled-history residual: the incumbent trunk remains
    # exactly unchanged at activation instead of attending to random/untrained
    # event embeddings. Default OFF/64 preserves historical checkpoints.
    meaningful_public_history: bool = False
    meaningful_public_history_schema: str = MEANINGFUL_PUBLIC_HISTORY_SCHEMA_VERSION
    event_history_limit: int = 64
    # Legacy public-card checkpoints used a biased Linear residual.  Preserve
    # that topology when this field is absent, while letting the v2 upgrade
    # select a bias-free map.  With bias=False, an all-zero public-card row is
    # guaranteed to contribute exactly zero even after the adapter is trained.
    # Appended last for positional legacy-pickle compatibility.
    public_card_count_residual_bias: bool = True
    # Function-preserving scalar-value repair. Historically the value head saw
    # only the pooled state token, so two identical public states with different
    # legal affordances were forced to the same value. This opt-in branch
    # masked-mean-pools the encoded legal actions and injects them through an
    # exact-zero, bias-free projection. Pairing it with static_action_residual
    # makes already-banked catalog ids expose Monopoly/Year-of-Plenty resource
    # semantics without rewriting legacy shard tensors.
    # Appended last for positional legacy-pickle compatibility.
    legal_action_value_residual: bool = False
    # The v1 history adapter pooled event embeddings as an unordered masked
    # mean. Fresh/re-featurized history corpora may opt into a lightweight
    # order-aware attention pool while retaining the exact same input schema.
    # Appended last for positional legacy-pickle compatibility.
    meaningful_public_history_pooling: str = MASKED_MEAN_V1


class EntityGraphNet:
    """Typed Catan entity-token encoder with sparse legal-action scoring."""

    def __new__(cls, config: EntityGraphConfig):
        import torch
        from torch import nn

        class _Block(nn.Module):
            def __init__(self, width: int, heads: int, dropout: float) -> None:
                super().__init__()
                self.norm_attn = nn.LayerNorm(width)
                self.attn = nn.MultiheadAttention(
                    width,
                    max(1, int(heads)),
                    dropout=float(dropout),
                    batch_first=True,
                )
                self.norm_ff = nn.LayerNorm(width)
                self.ff = nn.Sequential(
                    nn.Linear(width, 4 * width),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(4 * width, width),
                    nn.Dropout(float(dropout)),
                )

            def forward(self, tokens, key_padding_mask=None):
                attn_in = self.norm_attn(tokens)
                attn_out, _ = self.attn(
                    attn_in,
                    attn_in,
                    attn_in,
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                )
                tokens = tokens + attn_out
                tokens = tokens + self.ff(self.norm_ff(tokens))
                return tokens

        class _CrossBlock(nn.Module):
            """Pre-LN cross-attention block: `query` attends over `memory`.

            Zero-initialising both the attention output projection and the
            final feed-forward linear makes the block an exact identity at
            init (query is returned unchanged, since both residual branches
            add exactly 0.0). That is the warm-start guarantee: stacking any
            number of these on a checkpoint trained without them reproduces
            the original function bit-for-bit until the new weights train.
            """

            def __init__(
                self,
                width: int,
                heads: int,
                dropout: float,
                *,
                identity_init: bool = True,
            ) -> None:
                super().__init__()
                self.norm_q = nn.LayerNorm(width)
                self.norm_kv = nn.LayerNorm(width)
                self.attn = nn.MultiheadAttention(
                    width,
                    max(1, int(heads)),
                    dropout=float(dropout),
                    batch_first=True,
                )
                self.norm_ff = nn.LayerNorm(width)
                self.ff = nn.Sequential(
                    nn.Linear(width, 4 * width),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(4 * width, width),
                    nn.Dropout(float(dropout)),
                )
                if identity_init:
                    nn.init.zeros_(self.attn.out_proj.weight)
                    if self.attn.out_proj.bias is not None:
                        nn.init.zeros_(self.attn.out_proj.bias)
                    nn.init.zeros_(self.ff[3].weight)
                    nn.init.zeros_(self.ff[3].bias)

            def forward(self, query, memory, key_padding_mask=None):
                attn_out, _ = self.attn(
                    self.norm_q(query),
                    self.norm_kv(memory),
                    self.norm_kv(memory),
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                )
                query = query + attn_out
                query = query + self.ff(self.norm_ff(query))
                return query

        class _Module(nn.Module):
            def __init__(self, cfg: EntityGraphConfig) -> None:
                super().__init__()
                self.config = cfg
                h = int(cfg.hidden_size)
                dropout = float(cfg.dropout)
                self.state_trunk = (
                    str(getattr(cfg, "state_trunk", "transformer") or "transformer")
                    .strip()
                    .lower()
                )
                if self.state_trunk not in {"transformer", "rrt", "resrgcn"}:
                    raise ValueError(
                        "state_trunk must be one of transformer/rrt/resrgcn, got "
                        f"{self.state_trunk!r}"
                    )
                self.uses_relational_topology = self.state_trunk != "transformer"
                self.topology_residual_adapter_enabled = bool(
                    getattr(cfg, "topology_residual_adapter", False)
                )
                if (
                    self.uses_relational_topology
                    and self.topology_residual_adapter_enabled
                ):
                    raise ValueError(
                        "topology_residual_adapter is a warm-start for the transformer "
                        "trunk and cannot be combined with an already-relational trunk"
                    )
                self.latent_deliberation_steps = int(
                    getattr(cfg, "latent_deliberation_steps", 0) or 0
                )
                self.latent_deliberation_slots = int(
                    getattr(cfg, "latent_deliberation_slots", 8) or 0
                )
                if self.latent_deliberation_steps < 0:
                    raise ValueError("latent_deliberation_steps must be >= 0")
                if self.latent_deliberation_steps > 0:
                    if self.state_trunk != "rrt":
                        raise ValueError(
                            "latent deliberation currently requires state_trunk='rrt'"
                        )
                    if self.latent_deliberation_slots < 1:
                        raise ValueError("latent_deliberation_slots must be >= 1")
                self.moe_routed_experts = int(
                    getattr(cfg, "moe_routed_experts", 0) or 0
                )
                self.moe_top_k = int(getattr(cfg, "moe_top_k", 2) or 0)
                self.moe_enabled = self.moe_routed_experts > 0
                if self.moe_enabled:
                    if self.state_trunk != "rrt":
                        raise ValueError(
                            "sparse MoE currently requires state_trunk='rrt'"
                        )
                    if not 1 <= self.moe_top_k <= self.moe_routed_experts:
                        raise ValueError("moe_top_k must be in [1, moe_routed_experts]")
                self.hex_encoder = _token_encoder(HEX_FEATURE_SIZE, h, dropout)
                self.vertex_encoder = _token_encoder(VERTEX_FEATURE_SIZE, h, dropout)
                self.edge_encoder = _token_encoder(EDGE_FEATURE_SIZE, h, dropout)
                self.player_encoder = _token_encoder(PLAYER_FEATURE_SIZE, h, dropout)
                self.public_card_count_features_enabled = bool(
                    getattr(cfg, "public_card_count_features", False)
                )
                if self.public_card_count_features_enabled:
                    feature_schema = str(
                        getattr(cfg, "public_card_count_feature_schema", "") or ""
                    )
                    if feature_schema != PUBLIC_CARD_COUNT_FEATURE_SCHEMA_VERSION:
                        raise ValueError(
                            "unsupported public card-count feature schema: "
                            f"{feature_schema!r} != "
                            f"{PUBLIC_CARD_COUNT_FEATURE_SCHEMA_VERSION!r}"
                        )
                    self.public_card_count_residual = nn.Linear(
                        DEDUCTION_FEATURE_SIZE,
                        h,
                        bias=bool(
                            getattr(cfg, "public_card_count_residual_bias", True)
                        ),
                    )
                    nn.init.zeros_(self.public_card_count_residual.weight)
                    if self.public_card_count_residual.bias is not None:
                        nn.init.zeros_(self.public_card_count_residual.bias)
                self.global_encoder = _token_encoder(GLOBAL_FEATURE_SIZE, h, dropout)
                self.event_encoder = _token_encoder(EVENT_FEATURE_SIZE, h, dropout)
                self.meaningful_public_history_enabled = bool(
                    getattr(cfg, "meaningful_public_history", False)
                )
                if self.meaningful_public_history_enabled:
                    self.meaningful_public_history_pooling = str(
                        getattr(
                            cfg,
                            "meaningful_public_history_pooling",
                            MASKED_MEAN_V1,
                        )
                        or MASKED_MEAN_V1
                    )
                    if (
                        self.meaningful_public_history_pooling
                        not in SUPPORTED_HISTORY_POOLING
                    ):
                        raise ValueError(
                            "unsupported meaningful public-history pooling: "
                            f"{self.meaningful_public_history_pooling!r}"
                        )
                    # Per-channel zero gate is the smallest expressive
                    # function-preserving adapter: the existing event MLP can
                    # learn once the gate opens, while step zero contributes
                    # exactly zero to policy and value.
                    self.meaningful_history_residual_gate = nn.Parameter(
                        torch.zeros(h)
                    )
                    if self.meaningful_public_history_pooling == ORDERED_ATTENTION_V2:
                        self.meaningful_history_sequence = build_ordered_history_pool(
                            h, MEANINGFUL_PUBLIC_HISTORY_LIMIT
                        )
                        # Add the order-aware path without replacing the v1
                        # masked-mean path. A separate zero gate preserves a
                        # trained v1 history checkpoint exactly at upgrade.
                        self.meaningful_history_ordered_gate = nn.Parameter(
                            torch.zeros(h)
                        )
                self.type_embedding = nn.Parameter(torch.zeros(7, h))
                self.cls_token = nn.Parameter(torch.zeros(1, 1, h))
                if self.state_trunk == "transformer":
                    self.blocks = nn.ModuleList(
                        _Block(h, cfg.attention_heads, dropout)
                        for _ in range(max(1, int(cfg.state_layers)))
                    )
                    self.relational_block_pattern = ""
                    if self.topology_residual_adapter_enabled:
                        from catan_zero.rl.relational_trunks import (
                            TopologyResidualAdapter,
                        )

                        self.topology_residual_adapter = TopologyResidualAdapter(h)
                else:
                    from catan_zero.rl.relational_trunks import (
                        RelationalTransformerBlock,
                        SparseMoERelationalTransformerBlock,
                        VectorizedRelGraphBlock,
                    )

                    layer_count = max(1, int(cfg.state_layers))
                    configured_ff = int(getattr(cfg, "relational_ff_size", 0) or 0)
                    if configured_ff > 0:
                        relational_ff = configured_ff
                    elif self.state_trunk == "rrt":
                        relational_ff = max(64, int(round(1024 * h / 384)))
                    else:
                        relational_ff = max(64, int(round(512 * h / 384)))
                    if self.state_trunk == "rrt":
                        raw_pattern = str(
                            getattr(cfg, "relational_block_pattern", "") or ""
                        ).upper()
                        if not raw_pattern:
                            raw_pattern = ("RRT" * ((layer_count + 2) // 3))[
                                :layer_count
                            ]
                        if len(raw_pattern) != layer_count or set(raw_pattern) - {
                            "R",
                            "T",
                        }:
                            raise ValueError(
                                "relational_block_pattern must contain exactly "
                                f"state_layers R/T entries: pattern={raw_pattern!r} "
                                f"state_layers={layer_count}"
                            )
                        if self.moe_enabled and "T" not in raw_pattern:
                            raise ValueError(
                                "sparse MoE requires at least one global T block"
                            )
                        self.relational_block_pattern = raw_pattern
                        configured_expert_ff = int(
                            getattr(cfg, "moe_expert_ff_size", 0) or 0
                        )
                        if configured_expert_ff < 0:
                            raise ValueError("moe_expert_ff_size must be >= 0")
                        expert_ff = configured_expert_ff or max(
                            64, int(round(384 * h / 384))
                        )
                        blocks = []
                        for kind in raw_pattern:
                            if self.moe_enabled and kind == "T":
                                blocks.append(
                                    SparseMoERelationalTransformerBlock(
                                        h,
                                        cfg.attention_heads,
                                        expert_ff,
                                        self.moe_routed_experts,
                                        self.moe_top_k,
                                        dropout,
                                        global_block=True,
                                    )
                                )
                            else:
                                blocks.append(
                                    RelationalTransformerBlock(
                                        h,
                                        cfg.attention_heads,
                                        relational_ff,
                                        dropout,
                                        global_block=kind == "T",
                                    )
                                )
                        self.blocks = nn.ModuleList(blocks)
                    else:
                        if str(getattr(cfg, "relational_block_pattern", "") or ""):
                            raise ValueError(
                                "relational_block_pattern is only valid for state_trunk='rrt'"
                            )
                        bases = int(getattr(cfg, "relational_bases", 4))
                        if bases < 1:
                            raise ValueError("relational_bases must be >= 1")
                        self.relational_block_pattern = "G" * layer_count
                        self.blocks = nn.ModuleList(
                            VectorizedRelGraphBlock(
                                h,
                                relational_ff,
                                dropout,
                                bases=bases,
                            )
                            for _ in range(layer_count)
                        )
                self.state_norm = nn.LayerNorm(h)
                if self.latent_deliberation_steps > 0:
                    self.deliberation_slots = nn.Parameter(
                        torch.empty(1, self.latent_deliberation_slots, h)
                    )
                    self.deliberation_block = _CrossBlock(
                        h,
                        cfg.attention_heads,
                        dropout,
                        identity_init=False,
                    )
                    self.deliberation_fusion_norm = nn.LayerNorm(2 * h)
                    self.deliberation_fusion = nn.Linear(2 * h, h)
                    self.deliberation_halt_head = nn.Linear(h, 1)
                    nn.init.normal_(self.deliberation_slots, std=0.02)
                action_in = int(cfg.legal_action_feature_size) + int(
                    cfg.context_action_feature_size
                )
                self.action_encoder = nn.Sequential(
                    nn.Linear(action_in, h),
                    nn.LayerNorm(h),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(h, h),
                    nn.GELU(),
                    nn.Linear(h, h),
                )
                self.action_bias = nn.Linear(action_in, 1)
                self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0)))
                self.static_action_residual_enabled = bool(
                    getattr(cfg, "static_action_residual", False)
                )
                if self.static_action_residual_enabled:
                    if int(cfg.static_action_feature_size) < int(
                        STATIC_ACTION_RESIDUAL_SLICE.stop
                    ):
                        raise ValueError(
                            "static_action_residual requires at least "
                            f"{STATIC_ACTION_RESIDUAL_SLICE.stop} static-action "
                            f"columns, got {cfg.static_action_feature_size}"
                        )
                    # Preserve absolute sparse catalog magnitudes. LayerNorm
                    # makes the numeric node/edge argument vectors nearly
                    # collinear and destroys the signal this branch repairs.
                    self.static_action_residual_proj = nn.Linear(
                        STATIC_ACTION_RESIDUAL_FEATURE_SIZE, h
                    )
                    nn.init.zeros_(self.static_action_residual_proj.weight)
                    nn.init.zeros_(self.static_action_residual_proj.bias)
                self.legal_action_value_residual_enabled = bool(
                    getattr(cfg, "legal_action_value_residual", False)
                )
                if self.legal_action_value_residual_enabled:
                    self.legal_action_value_residual_proj = nn.Linear(
                        h, h, bias=False
                    )
                    nn.init.zeros_(self.legal_action_value_residual_proj.weight)
                    if self.static_action_residual_enabled:
                        # Value-private catalog path: value-only commissioning
                        # can learn resource/target semantics without unfreezing
                        # the shared static adapter and drifting policy logits.
                        self.legal_action_value_static_proj = nn.Linear(
                            STATIC_ACTION_RESIDUAL_FEATURE_SIZE,
                            h,
                            bias=False,
                        )
                        nn.init.zeros_(self.legal_action_value_static_proj.weight)
                self.value_head = nn.Sequential(
                    nn.Linear(h, h),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(h, 1),
                )
                self.final_vp_head = nn.Sequential(
                    nn.Linear(h, h // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(h // 2, 1),
                )
                # Optional value-uncertainty head: same shape as value_head, one
                # scalar per state. softplus at the emit site forces the predicted
                # squared-error to be non-negative. Only built when enabled so the
                # default model is unchanged.
                if bool(getattr(cfg, "value_uncertainty_head", False)):
                    self.value_uncertainty_head = nn.Sequential(
                        nn.Linear(h, h),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(h, 1),
                    )
                else:
                    self.value_uncertainty_head = None
                # Optional HL-Gauss categorical value head (CAT-39): purely
                # additive -- see EntityGraphConfig.value_categorical_bins. The
                # linear head emits `bins` win-loss logits plus, when the
                # truncation class is enabled, one extra logit; the support
                # buffer holds ONLY the win-loss bin centres (the truncation
                # class carries no support value and is excluded from the
                # expectation readout, per the R9 win/loss-only support).
                self.value_categorical_bins = int(
                    getattr(cfg, "value_categorical_bins", 0)
                )
                self.value_categorical_truncation_class = bool(
                    getattr(cfg, "value_categorical_truncation_class", True)
                )
                if self.value_categorical_bins >= 2:
                    n_out = self.value_categorical_bins + (
                        1 if self.value_categorical_truncation_class else 0
                    )
                    self.value_categorical_head = nn.Sequential(
                        nn.Linear(h, h),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(h, n_out),
                    )
                    # Non-persistent: keeps state_dict identical to a model
                    # without the buffer, so strict load round-trips are clean.
                    self.register_buffer(
                        "value_categorical_support",
                        torch.linspace(-1.0, 1.0, self.value_categorical_bins),
                        persistent=False,
                    )
                else:
                    self.value_categorical_head = None
                self.q_head = nn.Sequential(
                    nn.Linear(3 * h, h),
                    nn.LayerNorm(h),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(h, h // 2),
                    nn.GELU(),
                    nn.Linear(h // 2, 1),
                )
                nn.init.normal_(self.type_embedding, std=0.02)
                nn.init.normal_(self.cls_token, std=0.02)
                nn.init.zeros_(self.action_bias.weight)
                nn.init.zeros_(self.action_bias.bias)

                # --- f69 action-attention upgrade (all gated, default OFF) ---
                # Read via getattr so configs pickled before these slots
                # existed still construct (they resolve to the OFF defaults).
                self.action_target_gather = self.uses_relational_topology or bool(
                    getattr(cfg, "action_target_gather", False)
                )
                self.action_cross_attention_layers = (
                    int(getattr(cfg, "relational_action_cross_layers", 1))
                    if self.uses_relational_topology
                    else int(getattr(cfg, "action_cross_attention_layers", 0))
                )
                if self.action_cross_attention_layers < 0:
                    raise ValueError("action cross-attention layer count must be >= 0")
                self.value_attention_pool = bool(
                    getattr(cfg, "value_attention_pool", False)
                )

                if self.action_target_gather:
                    # Pooled target token (dim h) -> zero-init projection added
                    # to encoded_actions. LayerNorm tames the raw residual-
                    # stream magnitude; the trailing Linear is zero so the
                    # branch contributes exactly 0 at init.
                    self.target_gather_proj = nn.Sequential(
                        nn.LayerNorm(h),
                        nn.Linear(h, h),
                    )
                    if not self.uses_relational_topology:
                        nn.init.zeros_(self.target_gather_proj[1].weight)
                        nn.init.zeros_(self.target_gather_proj[1].bias)

                if self.action_cross_attention_layers > 0:
                    self.action_cross_blocks = nn.ModuleList(
                        _CrossBlock(
                            h,
                            cfg.attention_heads,
                            dropout,
                            identity_init=not self.uses_relational_topology,
                        )
                        for _ in range(self.action_cross_attention_layers)
                    )

                if self.value_attention_pool:
                    self.value_probe = nn.Parameter(torch.zeros(1, 1, h))
                    nn.init.normal_(self.value_probe, std=0.02)
                    self.value_probe_norm_q = nn.LayerNorm(h)
                    self.value_probe_norm_kv = nn.LayerNorm(h)
                    self.value_probe_attn = nn.MultiheadAttention(
                        h,
                        max(1, int(cfg.attention_heads)),
                        dropout=dropout,
                        batch_first=True,
                    )
                    # Consumes [CLS ++ probe_output] (2h). Final Linear is
                    # zero-init so this ADD-on head contributes 0 at init and
                    # the value equals today's value_head(CLS) exactly.
                    self.value_pool_head = nn.Sequential(
                        nn.LayerNorm(2 * h),
                        nn.Linear(2 * h, h),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(h, 1),
                    )
                    nn.init.zeros_(self.value_pool_head[4].weight)
                    nn.init.zeros_(self.value_pool_head[4].bias)

                # --- CAT-97 GATEAU edge/node-feature policy head (default OFF) ---
                self.edge_policy_head = (
                    self.uses_relational_topology
                    and bool(getattr(cfg, "relational_edge_policy_head", True))
                ) or bool(getattr(cfg, "edge_policy_head", False))
                if self.edge_policy_head:
                    # MLP(pooled target entity token) -> per-action scalar logit,
                    # ADDED to the CLIP logits. Zero-init final Linear so the
                    # logit is unchanged at init (warm-start guarantee).
                    self.edge_policy_mlp = nn.Sequential(
                        nn.LayerNorm(h),
                        nn.Linear(h, h),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(h, 1),
                    )
                    if not self.uses_relational_topology:
                        nn.init.zeros_(self.edge_policy_mlp[4].weight)
                        nn.init.zeros_(self.edge_policy_mlp[4].bias)

                # --- CAT-100 auxiliary Catan-subgoal heads (default OFF) ---
                # Read the pooled state (CLS) token only; emit new outputs that
                # never feed value/policy, so main outputs stay bit-identical.
                self.aux_subgoal_heads = bool(getattr(cfg, "aux_subgoal_heads", False))
                self.aux_settlement_pointer_head_enabled = bool(
                    getattr(cfg, "aux_settlement_pointer_head", False)
                )
                if (
                    self.aux_settlement_pointer_head_enabled
                    and not self.aux_subgoal_heads
                ):
                    raise ValueError(
                        "aux_settlement_pointer_head requires aux_subgoal_heads"
                    )
                if self.aux_subgoal_heads:

                    def _scalar_head() -> nn.Module:
                        return nn.Sequential(
                            nn.Linear(h, h),
                            nn.GELU(),
                            # Keep auxiliary readouts off the process-global
                            # dropout stream.  A matched AUX0/AUXT experiment
                            # must not change the next batch's shared-trunk
                            # dropout masks merely because AUXT evaluated an
                            # extra head.  The heads are already regularized by
                            # the stochastic shared trunk; another dropout here
                            # is both redundant and a causal-confound footgun.
                            nn.Identity(),
                            nn.Linear(h, 1),
                        )

                    def _categorical_head(num_classes: int) -> nn.Module:
                        return nn.Sequential(
                            nn.Linear(h, h),
                            nn.GELU(),
                            nn.Identity(),
                            nn.Linear(h, int(num_classes)),
                        )

                    self.aux_longest_road_head = _scalar_head()
                    self.aux_largest_army_head = _scalar_head()
                    self.aux_vp_in_n_head = _scalar_head()
                    # 54 intersections / 19 hexes: fixed board sizes, matching the
                    # per-type target-id space verified in _gather_target_tokens and
                    # the shape asserts in _assert_entity_batch_shapes.
                    if self.aux_settlement_pointer_head_enabled:
                        self.aux_next_settlement_pointer_head = nn.Sequential(
                            nn.LayerNorm(h),
                            nn.Linear(h, h),
                            nn.GELU(),
                            nn.Identity(),
                            nn.Linear(h, 1),
                        )
                    else:
                        # Retained for exact legacy checkpoint compatibility.
                        self.aux_next_settlement_head = _categorical_head(
                            AUX_NUM_INTERSECTIONS
                        )
                    # Hex tokens include canonical coordinate features, so the
                    # dense absolute-id classifier is not topology-aliased.
                    self.aux_robber_target_head = _categorical_head(AUX_NUM_HEXES)

                self.belief_resource_head_enabled = bool(
                    getattr(cfg, "belief_resource_head", False)
                )
                if self.belief_resource_head_enabled:
                    self.belief_resource_head = nn.Sequential(
                        nn.LayerNorm(h),
                        nn.Linear(h, h),
                        nn.GELU(),
                        nn.Linear(h, 5),
                    )

            def forward(
                self,
                batch: dict[str, Any],
                *,
                return_q: bool = False,
                return_final_vp: bool = True,
                return_aux_subgoals: bool = True,
                value_trunk_grad_scale: float = 1.0,
                event_token_limit: int | None = None,
            ):
                """Encode a state and score its legal actions.

                ``event_token_limit`` is an opt-in static-shape control for
                inference schedulers.  It may remove only a trailing suffix of
                event positions that is masked for every row in the batch.  The
                default keeps the historical full-width path unchanged.

                Keeping state encoding and action scoring as two explicit calls
                lets inference runtimes capture/compile the expensive fixed
                state trunk independently of the variable legal-action head.
                """
                encoded_state = self.encode_state(
                    batch,
                    event_token_limit=event_token_limit,
                )
                return self.score_actions(
                    encoded_state,
                    batch,
                    return_q=return_q,
                    return_final_vp=return_final_vp,
                    return_aux_subgoals=return_aux_subgoals,
                    value_trunk_grad_scale=value_trunk_grad_scale,
                )

            def parameter_accounting(self) -> dict[str, int]:
                """Exact instantiated and nominal per-token active parameters."""
                total = sum(
                    parameter.numel()
                    for parameter in self.parameters()
                    if parameter.requires_grad
                )
                inactive = 0
                if self.moe_enabled:
                    for block in self.blocks:
                        if not bool(getattr(block, "is_sparse_moe", False)):
                            continue
                        inactive += (
                            block.moe.routed_expert_count - block.moe.top_k
                        ) * block.moe.one_routed_expert_parameters
                return {
                    "instantiated_trainable": int(total),
                    "nominal_active_per_token": int(total - inactive),
                }

            def encode_state(
                self,
                batch: dict[str, Any],
                *,
                event_token_limit: int | None = None,
            ):
                """Run the typed-token transformer trunk.

                The returned tuple is intentionally tensor-only and stable:
                ``(tokens, padding_mask, state)``.  That makes this boundary
                suitable for a future static-shape CUDA graph without adding
                wrapper objects or changing the checkpoint parameter set.
                """
                if event_token_limit is not None:
                    if isinstance(event_token_limit, bool):
                        raise TypeError(
                            "event_token_limit must be an integer, not bool"
                        )
                    try:
                        event_token_limit = operator.index(event_token_limit)
                    except TypeError as error:
                        raise TypeError(
                            "event_token_limit must be an integer"
                        ) from error
                    self._validate_event_token_limit(
                        batch,
                        event_token_limit=event_token_limit,
                    )
                tokens, padding_mask, event_piece, event_mask = self._state_tokens(
                    batch,
                    event_token_limit=event_token_limit,
                )
                if self.topology_residual_adapter_enabled:
                    from catan_zero.rl.relational_trunks import build_relation_ids

                    relation_batch = batch
                    if event_token_limit is not None and "event_target_ids" in batch:
                        relation_batch = dict(batch)
                        relation_batch["event_target_ids"] = batch["event_target_ids"][
                            :, :event_token_limit
                        ]
                    relation_ids = build_relation_ids(
                        relation_batch,
                        sequence_length=int(tokens.shape[1]),
                    )
                    tokens = self.topology_residual_adapter(
                        tokens, relation_ids, key_padding_mask=padding_mask
                    )
                if self.uses_relational_topology:
                    from catan_zero.rl.relational_trunks import build_relation_ids

                    relation_batch = batch
                    if event_token_limit is not None and "event_target_ids" in batch:
                        relation_batch = dict(batch)
                        relation_batch["event_target_ids"] = batch["event_target_ids"][
                            :, :event_token_limit
                        ]
                    relation_ids = build_relation_ids(
                        relation_batch,
                        sequence_length=int(tokens.shape[1]),
                    )
                    moe_balance = []
                    moe_load = []
                    moe_importance = []
                    for block in self.blocks:
                        block_output = block(
                            tokens,
                            relation_ids,
                            key_padding_mask=padding_mask,
                        )
                        if bool(getattr(block, "is_sparse_moe", False)):
                            tokens, balance, load, importance = block_output
                            moe_balance.append(balance)
                            moe_load.append(load)
                            moe_importance.append(importance)
                        else:
                            tokens = block_output
                else:
                    for block in self.blocks:
                        tokens = block(tokens, key_padding_mask=padding_mask)
                state = self.state_norm(tokens[:, 0])
                if self.latent_deliberation_steps > 0:
                    plan = self.deliberation_slots.expand(tokens.shape[0], -1, -1)
                    for _ in range(self.latent_deliberation_steps):
                        plan = self.deliberation_block(
                            plan,
                            tokens,
                            key_padding_mask=padding_mask,
                        )
                    fused = torch.cat((state, plan.mean(dim=1)), dim=-1)
                    state = self.deliberation_fusion(
                        torch.nn.functional.gelu(self.deliberation_fusion_norm(fused))
                    )
                if self.meaningful_public_history_enabled:
                    # Event rows stay masked from the mature trunk. Their
                    # bounded pooled representation enters only through this
                    # exact-zero gate, so history activation cannot perturb an
                    # incumbent before the first optimizer step. Scaling by the
                    # fixed cap preserves event-count mass as well as content.
                    history_weight = event_mask.to(event_piece.dtype).unsqueeze(-1)
                    pooled_history = (
                        event_piece * history_weight
                    ).sum(dim=1) / float(MEANINGFUL_PUBLIC_HISTORY_LIMIT)
                    state = (
                        state
                        + pooled_history * self.meaningful_history_residual_gate
                    )
                    if (
                        self.meaningful_public_history_pooling
                        == ORDERED_ATTENTION_V2
                    ):
                        ordered_history = self.meaningful_history_sequence(
                            event_piece, event_mask
                        )
                        state = (
                            state
                            + ordered_history * self.meaningful_history_ordered_gate
                        )
                if self.moe_enabled:
                    return (
                        tokens,
                        padding_mask,
                        state,
                        torch.stack(moe_balance).mean(),
                        torch.stack(moe_load),
                        torch.stack(moe_importance),
                    )
                return tokens, padding_mask, state

            def score_actions(
                self,
                encoded_state,
                batch: dict[str, Any],
                *,
                return_q: bool = False,
                return_final_vp: bool = True,
                return_aux_subgoals: bool = True,
                value_trunk_grad_scale: float = 1.0,
            ):
                """Score legal actions and emit value heads from encoded state.

                ``value_trunk_grad_scale`` is a training-only causal probe.  It
                changes no forward value and leaves the value-head parameter
                gradient untouched; it scales only the scalar value loss's
                gradients at every shared state/token trunk boundary.  The
                default takes the historical path without adding an operation
                to the graph.
                """
                tokens, padding_mask, state = encoded_state[:3]
                # The BC trainer freezes zero-objective optional heads and sets
                # this non-persistent module-name gate before DDP/optimizer
                # construction.  Skipping those forwards is semantically
                # important even after requires_grad=False: several heads contain
                # dropout, so executing an unused head would advance torch's RNG
                # and silently change the trunk dropout masks on the next batch.
                # Freshly created/inference-loaded models have no gate and retain
                # the historical full-output API.
                inactive_training_heads = getattr(
                    self, "_inactive_training_head_modules", frozenset()
                )
                action_features = torch.cat(
                    (
                        batch["legal_action_tokens"].float(),
                        batch["legal_action_context"].float(),
                    ),
                    dim=-1,
                )
                encoded_actions = self.action_encoder(action_features)
                if self.static_action_residual_enabled:
                    static_features = batch.get("legal_action_static_features")
                    if static_features is None:
                        raise ValueError(
                            "static_action_residual requires "
                            "legal_action_static_features"
                        )
                    if (
                        static_features.ndim != 3
                        or static_features.shape[:2] != encoded_actions.shape[:2]
                        or int(static_features.shape[2])
                        != STATIC_ACTION_RESIDUAL_FEATURE_SIZE
                    ):
                        raise ValueError(
                            "legal_action_static_features shape must be [B,A,22], "
                            f"got {tuple(static_features.shape)}"
                        )
                    encoded_actions = (
                        encoded_actions
                        + self.static_action_residual_proj(static_features.float())
                    )
                # Post-trunk target-entity tokens per action, mean-pooled ([B,A,h]).
                # Shared by action_target_gather (modulates the CLIP embedding) and
                # the CAT-97 edge_policy_head (emits a direct logit); computed once.
                pooled_targets = None
                if self.action_target_gather or self.edge_policy_head:
                    pooled_targets = self._gather_target_tokens(tokens, batch)
                if self.action_target_gather:
                    encoded_actions = encoded_actions + self.target_gather_proj(
                        pooled_targets
                    )
                if self.action_cross_attention_layers > 0:
                    for cross_block in self.action_cross_blocks:
                        encoded_actions = cross_block(
                            encoded_actions,
                            tokens,
                            key_padding_mask=padding_mask,
                        )
                policy_state = torch.nn.functional.normalize(state, dim=-1)
                policy_actions = torch.nn.functional.normalize(encoded_actions, dim=-1)
                logit_scale = torch.clamp(self.logit_scale.exp(), max=50.0)
                logits = logit_scale * (policy_state.unsqueeze(1) * policy_actions).sum(
                    dim=-1
                )
                logits = logits + self.action_bias(action_features).squeeze(-1)
                if self.edge_policy_head:
                    # AlphaGateau per-move readout: a direct logit from each
                    # action's pooled target-entity token. Zero-init -> +0 at init.
                    logits = logits + self.edge_policy_mlp(pooled_targets).squeeze(-1)
                value_trunk_grad_scale = float(value_trunk_grad_scale)
                if not math.isfinite(value_trunk_grad_scale) or not (
                    0.0 <= value_trunk_grad_scale <= 1.0
                ):
                    raise ValueError(
                        "value_trunk_grad_scale must be finite and in [0, 1], got "
                        f"{value_trunk_grad_scale}"
                    )
                if value_trunk_grad_scale == 1.0:
                    value_state = state
                    value_tokens = tokens
                elif value_trunk_grad_scale == 0.0:
                    # Forward identity with an exact stop-gradient at the shared
                    # boundary.  Both value readouts still receive normal
                    # parameter gradients.
                    value_state = state.detach()
                    value_tokens = tokens.detach()
                else:
                    # ``state - state.detach()`` is exactly zero in the forward
                    # pass and has derivative one.  This therefore preserves the
                    # value tensor while scaling only its upstream derivative.
                    # The attention-pool branch also reads post-trunk tokens, so
                    # apply the same boundary there; otherwise an enabled pool
                    # silently leaks full-strength scalar-value gradients into
                    # the shared trunk.
                    value_state = state.detach() + value_trunk_grad_scale * (
                        state - state.detach()
                    )
                    value_tokens = tokens.detach() + value_trunk_grad_scale * (
                        tokens - tokens.detach()
                    )
                if self.legal_action_value_residual_enabled:
                    action_mask = batch.get("legal_action_mask")
                    if action_mask is None:
                        raise ValueError(
                            "legal_action_value_residual requires the exact "
                            "legal_action_mask; padded actions must never enter "
                            "the value affordance"
                        )
                    if tuple(action_mask.shape) != tuple(encoded_actions.shape[:2]):
                        raise ValueError(
                            "legal_action_mask shape must match encoded actions: "
                            f"{tuple(action_mask.shape)} != "
                            f"{tuple(encoded_actions.shape[:2])}"
                        )
                    action_weight = action_mask.to(encoded_actions.dtype).unsqueeze(-1)
                    legal_affordance = (encoded_actions * action_weight).sum(dim=1)
                    legal_affordance = legal_affordance / action_weight.sum(
                        dim=1
                    ).clamp_min(1.0)
                    if value_trunk_grad_scale == 0.0:
                        legal_affordance = legal_affordance.detach()
                    elif value_trunk_grad_scale != 1.0:
                        legal_affordance = (
                            legal_affordance.detach()
                            + value_trunk_grad_scale
                            * (legal_affordance - legal_affordance.detach())
                        )
                    value_state = value_state + self.legal_action_value_residual_proj(
                        legal_affordance
                    )
                    if hasattr(self, "legal_action_value_static_proj"):
                        static_affordance = (
                            static_features.float() * action_weight
                        ).sum(dim=1) / action_weight.sum(dim=1).clamp_min(1.0)
                        value_state = (
                            value_state
                            + self.legal_action_value_static_proj(static_affordance)
                        )
                value = self.value_head(value_state).squeeze(-1)
                if self.value_attention_pool:
                    value = value + self._value_pool(
                        value_state, value_tokens, padding_mask
                    )
                outputs = {
                    "logits": logits,
                    "value": value,
                }
                if return_final_vp and "final_vp_head" not in inactive_training_heads:
                    outputs["final_vp"] = self.final_vp_head(value_state).squeeze(-1)
                if (
                    self.latent_deliberation_steps > 0
                    and "deliberation_halt_head" not in inactive_training_heads
                ):
                    outputs["deliberation_halt_logit"] = self.deliberation_halt_head(
                        state
                    ).squeeze(-1)
                if self.moe_enabled:
                    outputs["moe_balance_metric"] = encoded_state[3]
                    outputs["moe_routing_load"] = encoded_state[4]
                    outputs["moe_routing_importance"] = encoded_state[5]
                if (
                    self.value_uncertainty_head is not None
                    and "value_uncertainty_head" not in inactive_training_heads
                ):
                    # Stop-gradient (CAT-61): the uncertainty head reads a
                    # DETACHED copy of the trunk state, so its regression loss
                    # trains only the head's own parameters and never flows
                    # gradients back into the shared trunk or the value head.
                    # This is the KataGo short-term-error design -- the error
                    # predictor must not distort value learning. (The training
                    # target in train_bc.py separately detaches `value`, so
                    # both the target and the head input are stop-gradiented.)
                    outputs["value_uncertainty"] = torch.nn.functional.softplus(
                        self.value_uncertainty_head(state.detach()).squeeze(-1)
                    )
                if (
                    self.value_categorical_head is not None
                    and "value_categorical_head" not in inactive_training_heads
                ):
                    cat_logits = self.value_categorical_head(value_state)
                    outputs["value_categorical_logits"] = cat_logits
                    n_bins = self.value_categorical_bins
                    probs = torch.softmax(cat_logits.float(), dim=-1)
                    win_probs = probs[..., :n_bins]
                    # Calibrated win-value: expectation over the win-loss bins,
                    # renormalised to exclude any truncation-class mass so the
                    # scalar readout is P(win)-calibrated (R9: the search backup
                    # reads this win-value, never a win+margin blend).
                    win_mass = win_probs.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
                    outputs["value_categorical"] = (
                        (win_probs / win_mass) * self.value_categorical_support
                    ).sum(dim=-1)
                    if self.value_categorical_truncation_class:
                        outputs["value_categorical_truncation_prob"] = probs[..., -1]
                if self.aux_subgoal_heads and return_aux_subgoals:
                    # CAT-100 auxiliary subgoal predictions. Raw logits/scalars
                    # (loss applies BCE-with-logits / CE / MSE at the train site).
                    # These never feed value/policy, so main outputs are unchanged.
                    if "aux_longest_road_head" not in inactive_training_heads:
                        outputs["aux_longest_road"] = self.aux_longest_road_head(
                            state
                        ).squeeze(-1)
                    if "aux_largest_army_head" not in inactive_training_heads:
                        outputs["aux_largest_army"] = self.aux_largest_army_head(
                            state
                        ).squeeze(-1)
                    if "aux_vp_in_n_head" not in inactive_training_heads:
                        outputs["aux_vp_in_n"] = self.aux_vp_in_n_head(state).squeeze(
                            -1
                        )
                    if self.aux_settlement_pointer_head_enabled:
                        if (
                            "aux_next_settlement_pointer_head"
                            not in inactive_training_heads
                        ):
                            vertex_start = _entity_token_start_offsets(batch)[1]
                            vertex_count = int(batch["vertex_tokens"].shape[1])
                            vertex_states = tokens[
                                :, vertex_start : vertex_start + vertex_count
                            ]
                            outputs["aux_next_settlement"] = (
                                self.aux_next_settlement_pointer_head(
                                    vertex_states
                                ).squeeze(-1)
                            )
                    elif "aux_next_settlement_head" not in inactive_training_heads:
                        outputs["aux_next_settlement"] = self.aux_next_settlement_head(
                            state
                        )
                    if "aux_robber_target_head" not in inactive_training_heads:
                        outputs["aux_robber_target"] = self.aux_robber_target_head(
                            state
                        )
                if (
                    self.belief_resource_head_enabled
                    and "belief_resource_head" not in inactive_training_heads
                ):
                    player_count = int(batch["player_tokens"].shape[1])
                    player_start = _entity_token_start_offsets(batch)[3]
                    player_states = tokens[
                        :, player_start : player_start + player_count
                    ]
                    outputs["belief_resource_logits"] = self.belief_resource_head(
                        player_states
                    )
                if return_q:
                    state_expanded = state.unsqueeze(1).expand_as(encoded_actions)
                    q_features = torch.cat(
                        (
                            state_expanded,
                            encoded_actions,
                            state_expanded * encoded_actions,
                        ),
                        dim=-1,
                    )
                    outputs["q_values"] = self.q_head(q_features).squeeze(-1)
                return outputs

            def _validate_event_token_limit(
                self,
                batch: dict[str, Any],
                *,
                event_token_limit: int,
            ):
                """Ensure a requested event prefix retains every valid event.

                This validates the scheduler-provided bucket boundary before
                changing the sequence shape. Validation may synchronize when
                the mask lives on CUDA; a captured production path should crop
                the host batch first and call ``encode_state`` without this
                option, leaving only static tensor shapes inside the graph.
                """
                event_mask = batch["event_mask"].bool()
                padded_width = int(event_mask.shape[1])
                if not 0 <= event_token_limit <= padded_width:
                    raise ValueError(
                        "event_token_limit must be within the padded event width: "
                        f"{event_token_limit} not in [0, {padded_width}]"
                    )
                if event_token_limit == padded_width:
                    return
                omitted_mask = event_mask[:, event_token_limit:]
                if bool(omitted_mask.any().item()):
                    raise ValueError(
                        "event_token_limit would remove at least one unmasked "
                        f"event token: limit={event_token_limit} width={padded_width}"
                    )

            def _state_tokens(
                self,
                batch: dict[str, Any],
                *,
                event_token_limit: int | None = None,
            ):
                import torch

                event_tokens = batch["event_tokens"]
                event_mask = batch["event_mask"]
                if event_token_limit is not None:
                    event_tokens = event_tokens[:, :event_token_limit]
                    event_mask = event_mask[:, :event_token_limit]
                meaningful_event_width = int(event_tokens.shape[1])
                if self.meaningful_public_history_enabled:
                    if meaningful_event_width > _LEGACY_EVENT_HISTORY_WIDTH:
                        raise ValueError(
                            "meaningful public history exceeds the inherited "
                            "event-token surface: "
                            f"{meaningful_event_width} > "
                            f"{_LEGACY_EVENT_HISTORY_WIDTH}"
                        )
                    # Keep the mature f7 trunk's exact 64-row attention shape.
                    # The first `meaningful_event_width` rows carry the new
                    # public history for the side residual; every event row is
                    # still masked as a key/value in the inherited trunk.  The
                    # padded suffix also keeps the event encoder's tensor shape
                    # and dropout RNG consumption identical to f7 at gate=0.
                    pad_width = _LEGACY_EVENT_HISTORY_WIDTH - meaningful_event_width
                    if pad_width:
                        event_tokens = torch.cat(
                            (
                                event_tokens,
                                event_tokens.new_zeros(
                                    event_tokens.shape[0],
                                    pad_width,
                                    event_tokens.shape[2],
                                ),
                            ),
                            dim=1,
                        )
                        event_mask_for_trunk = torch.cat(
                            (
                                event_mask,
                                event_mask.new_zeros(
                                    event_mask.shape[0], pad_width
                                ),
                            ),
                            dim=1,
                        )
                    else:
                        event_mask_for_trunk = event_mask
                else:
                    event_mask_for_trunk = event_mask
                # The event0 inference path has no event elements to encode.
                # Calling the MLP on [B, 0, F] is mathematically empty but still
                # launches its Linear/LayerNorm/GELU kernels.  Construct the
                # identical empty output directly, preserving dtype/device and
                # the final hidden width without changing any non-empty path.
                if event_tokens.shape[1] == 0:
                    event_piece = self.type_embedding.new_empty(
                        (
                            event_tokens.shape[0],
                            0,
                            self.type_embedding.shape[1],
                        )
                    )
                else:
                    event_piece = self.event_encoder(
                        event_tokens.float()
                    ) + self.type_embedding[6].view(1, 1, -1)
                history_event_piece = (
                    event_piece[:, :meaningful_event_width]
                    if self.meaningful_public_history_enabled
                    else event_piece
                )
                player_piece = self.player_encoder(batch["player_tokens"].float())
                if self.public_card_count_features_enabled:
                    player_piece = player_piece + self.public_card_count_residual(
                        batch[DEDUCTION_FEATURES_KEY].float()
                    )
                pieces = [
                    self.cls_token.expand(batch["hex_tokens"].shape[0], -1, -1)
                    + self.type_embedding[0].view(1, 1, -1),
                    self.hex_encoder(batch["hex_tokens"].float())
                    + self.type_embedding[1].view(1, 1, -1),
                    self.vertex_encoder(batch["vertex_tokens"].float())
                    + self.type_embedding[2].view(1, 1, -1),
                    self.edge_encoder(batch["edge_tokens"].float())
                    + self.type_embedding[3].view(1, 1, -1),
                    player_piece + self.type_embedding[4].view(1, 1, -1),
                    self.global_encoder(batch["global_tokens"].float())
                    + self.type_embedding[5].view(1, 1, -1),
                    event_piece,
                ]
                tokens = torch.cat(pieces, dim=1)
                # Keep the symbolic batch dim (do NOT coerce to a Python int): under
                # torch.onnx tracing int() bakes the current batch size into the mask
                # `zeros(...)` as a constant, forcing a fixed-batch ONNX graph. The
                # gen-2 CPU evaluator needs a variable batch axis (ragged chance
                # fan-outs / action counts); in eager mode this is an int anyway.
                batch_size = tokens.shape[0]
                event_padding_mask = (
                    torch.ones_like(event_mask_for_trunk, dtype=torch.bool)
                    if self.meaningful_public_history_enabled
                    else ~event_mask_for_trunk.bool()
                )
                masks = [
                    torch.zeros(
                        (batch_size, 1), dtype=torch.bool, device=tokens.device
                    ),
                    ~batch["hex_mask"].bool(),
                    ~batch["vertex_mask"].bool(),
                    ~batch["edge_mask"].bool(),
                    ~batch["player_mask"].bool(),
                    torch.zeros(
                        (batch_size, 1), dtype=torch.bool, device=tokens.device
                    ),
                    event_padding_mask,
                ]
                return (
                    tokens,
                    torch.cat(masks, dim=1),
                    history_event_piece,
                    event_mask,
                )

            def _gather_target_tokens(self, tokens, batch: dict[str, Any]):
                """Pool post-trunk board tokens for each action's targets.

                `legal_action_target_ids` is [B, A, 4] with a FIXED column ->
                entity-type mapping (verified against the featurizer,
                `_legal_action_target_ids`): col0=hex id (0-18), col1=vertex/
                node id (0-53), col2=edge id (0-71), col3=player id (0-3), each
                -1 when that target type is absent for the action. These are
                per-entity-type indices, NOT indices into the concatenated
                token sequence, so we add the constant per-type start offsets
                of the CLS-prefixed [CLS | hex | vertex | edge | player |
                global | event] layout built in `_state_tokens`. hex/vertex/
                edge counts are fixed (19/54/72) and player tokens are always
                4 rows, so the offsets are constants; we still derive them from
                the live shapes so the mapping stays correct if the layout ever
                changes. Valid targets are mean-pooled; an action with no board
                target (e.g. ROLL, END_TURN) pools to the zero vector.
                """
                import torch

                target_ids = batch["legal_action_target_ids"].long()  # [B, A, 4]
                # Start index of each targeted type in the concatenated
                # sequence (CLS occupies index 0).
                offsets = torch.tensor(
                    _entity_token_start_offsets(batch),
                    dtype=torch.long,
                    device=target_ids.device,
                )
                seq_len = int(tokens.shape[1])
                width = int(tokens.shape[2])
                valid = target_ids >= 0  # [B, A, 4]
                gather_index = torch.where(
                    valid,
                    target_ids + offsets.view(1, 1, -1),
                    torch.zeros_like(target_ids),
                ).clamp_(0, seq_len - 1)
                batch_size = int(target_ids.shape[0])
                num_actions = int(target_ids.shape[1])
                flat_index = gather_index.reshape(batch_size, num_actions * 4)
                gathered = torch.gather(
                    tokens,
                    1,
                    flat_index.unsqueeze(-1).expand(-1, -1, width),
                ).reshape(batch_size, num_actions, 4, width)
                weight = valid.to(gathered.dtype).unsqueeze(-1)  # [B, A, 4, 1]
                pooled = (gathered * weight).sum(dim=2) / weight.sum(dim=2).clamp_(
                    min=1.0
                )
                return pooled  # [B, A, h]

            def _value_pool(self, state, tokens, padding_mask):
                """Learned probe token cross-attends over all output tokens.

                Returns the scalar (per batch row) contribution of the
                zero-initialised head consuming [CLS ++ probe_output] (2h). The
                final linear of that head is zero at init, so this returns
                exactly 0 until trained -- value equals today's value_head(CLS).
                """
                import torch

                batch_size = int(tokens.shape[0])
                probe = self.value_probe.expand(batch_size, -1, -1)
                probe_out, _ = self.value_probe_attn(
                    self.value_probe_norm_q(probe),
                    self.value_probe_norm_kv(tokens),
                    self.value_probe_norm_kv(tokens),
                    key_padding_mask=padding_mask,
                    need_weights=False,
                )
                probe_out = probe_out.squeeze(1)  # [B, h]
                pooled = torch.cat((state, probe_out), dim=-1)  # [B, 2h]
                return self.value_pool_head(pooled).squeeze(-1)

        return _Module(config)


def _token_encoder(input_size: int, hidden_size: int, dropout: float):
    from torch import nn

    return nn.Sequential(
        nn.Linear(int(input_size), int(hidden_size)),
        nn.LayerNorm(int(hidden_size)),
        nn.GELU(),
        nn.Dropout(float(dropout)),
        nn.Linear(int(hidden_size), int(hidden_size)),
    )


def event_batch_shape_telemetry(event_mask: Any) -> dict[str, int | float]:
    """Summarize host-side event occupancy for static bucket selection.

    ``required_event_width`` is the smallest prefix that retains every valid
    event in the batch and can be passed to ``EntityGraphNet.forward`` or
    ``encode_state`` as ``event_token_limit``.  This helper is intentionally
    outside the model forward so telemetry never introduces a device sync into
    the default inference path.
    """
    if hasattr(event_mask, "detach"):
        event_mask = event_mask.detach().cpu().numpy()
    mask = np.asarray(event_mask, dtype=np.bool_)
    if mask.ndim != 2:
        raise ValueError(f"event_mask must be rank 2, got {mask.shape}")

    batch_size, padded_width = (int(mask.shape[0]), int(mask.shape[1]))
    if padded_width == 0:
        row_widths = np.zeros((batch_size,), dtype=np.int64)
    else:
        column_ids = np.arange(1, padded_width + 1, dtype=np.int64)
        row_widths = np.where(mask, column_ids[None, :], 0).max(axis=1, initial=0)
    required_width = int(row_widths.max()) if batch_size else 0
    min_row_width = int(row_widths.min()) if batch_size else 0
    active_tokens = int(mask.sum())
    total_tokens = batch_size * padded_width
    return {
        "batch_size": batch_size,
        "padded_event_width": padded_width,
        "required_event_width": required_width,
        "min_row_event_width": min_row_width,
        "max_row_event_width": required_width,
        "active_event_tokens": active_tokens,
        "event_token_utilization": (
            float(active_tokens / total_tokens) if total_tokens else 0.0
        ),
    }


class EntityGraphPolicy:
    # Inference schedulers may omit diagnostic-only heads without changing the
    # checkpoint or the policy/value tensors consumed by search.
    supports_final_vp_selection = True
    name = "entity_graph"
    policy_type = "entity_graph"

    def __init__(
        self,
        config: EntityGraphConfig,
        static_action_features: np.ndarray,
        *,
        seed: int = 0,
        device: str | None = None,
        entity_feature_adapter_version: str = CURRENT_RUST_ENTITY_ADAPTER_VERSION,
    ) -> None:
        import torch

        torch.manual_seed(seed)
        self.config = config
        history_pooling = str(
            getattr(config, "meaningful_public_history_pooling", MASKED_MEAN_V1)
            or MASKED_MEAN_V1
        )
        if history_pooling not in SUPPORTED_HISTORY_POOLING:
            raise ValueError(
                f"unsupported meaningful public-history pooling: {history_pooling!r}"
            )
        if history_pooling != MASKED_MEAN_V1 and not bool(
            getattr(config, "meaningful_public_history", False)
        ):
            raise ValueError(
                "order-aware public-history pooling requires "
                "meaningful_public_history=True"
            )
        if bool(getattr(config, "meaningful_public_history", False)):
            history_schema = str(
                getattr(config, "meaningful_public_history_schema", "") or ""
            )
            if history_schema != MEANINGFUL_PUBLIC_HISTORY_SCHEMA_VERSION:
                raise ValueError(
                    "unsupported meaningful public-history schema: "
                    f"{history_schema!r} != "
                    f"{MEANINGFUL_PUBLIC_HISTORY_SCHEMA_VERSION!r}"
                )
            history_limit = int(getattr(config, "event_history_limit", 0) or 0)
            if not 1 <= history_limit <= MEANINGFUL_PUBLIC_HISTORY_LIMIT:
                raise ValueError(
                    "meaningful public history requires event_history_limit in "
                    f"[1, {MEANINGFUL_PUBLIC_HISTORY_LIMIT}], got {history_limit}"
                )
        self.architecture = self.policy_type
        # f72 safety net (task #76): whether this policy's weights were trained
        # with train_bc.py --mask-hidden-info. Overwritten by .load() from the
        # checkpoint's own recorded metadata; freshly constructed policies
        # (train_bc.py's own training loop, before its first save) default False.
        self.trained_with_masked_hidden_info: bool = False
        self.entity_feature_adapter_version = require_known_entity_feature_adapter(
            entity_feature_adapter_version
        )
        self.entity_feature_adapter_binding_source = "new_policy_runtime_binding"
        # Durable checkpoint provenance.  Fresh policies have no training
        # attestation; load() fills these from the source checkpoint so a plain
        # load->save round trip cannot silently erase or reset its information
        # regime.  Training writers pass replacement values explicitly.
        self.soft_target_source = ""
        self.value_training: dict[str, object] | None = None
        self.training_information_surface: dict[str, object] | None = None
        # Fail closed for new/legacy callers until a corrected-feature training
        # transaction explicitly attests authoritative_v1.  In particular, an
        # old checkpoint loaded and re-saved must not silently opt into a random
        # player_encoder input column merely because the runtime wheel is newer.
        self.public_award_feature_contract = PUBLIC_AWARD_FEATURE_CONTRACT_LEGACY_ZERO
        self.action_size = int(config.action_size)
        self.context_action_feature_size = int(config.context_action_feature_size)
        self.device = _resolve_device(device)
        self.static_action_features = torch.as_tensor(
            static_action_features,
            dtype=torch.float32,
            device=self.device,
        )
        if bool(getattr(config, "static_action_residual", False)):
            if self.static_action_features.ndim != 2:
                raise ValueError(
                    "static_action_residual requires a rank-2 action catalog, "
                    f"got {tuple(self.static_action_features.shape)}"
                )
            if int(self.static_action_features.shape[0]) < self.action_size:
                raise ValueError(
                    "static action catalog has fewer rows than action_size: "
                    f"{self.static_action_features.shape[0]} < {self.action_size}"
                )
            if int(self.static_action_features.shape[1]) < int(
                STATIC_ACTION_RESIDUAL_SLICE.stop
            ):
                raise ValueError(
                    "static_action_residual requires at least "
                    f"{STATIC_ACTION_RESIDUAL_SLICE.stop} catalog columns, got "
                    f"{self.static_action_features.shape[1]}"
                )
        self.model = EntityGraphNet(config).to(self.device)

    @classmethod
    def create(
        cls,
        *,
        env_config: ColonistMultiAgentConfig | None = None,
        hidden_size: int = 640,
        state_layers: int = 6,
        attention_heads: int = 8,
        dropout: float = 0.05,
        seed: int = 0,
        device: str | None = None,
        value_uncertainty_head: bool = False,
        value_categorical_bins: int = 0,
        edge_policy_head: bool = False,
        aux_subgoal_heads: bool = False,
        aux_vp_horizon: int = 8,
        state_trunk: str = "transformer",
        relational_block_pattern: str = "",
        relational_ff_size: int = 0,
        relational_bases: int = 4,
        relational_action_cross_layers: int = 1,
        latent_deliberation_steps: int = 0,
        latent_deliberation_slots: int = 8,
        moe_routed_experts: int = 0,
        moe_top_k: int = 2,
        moe_expert_ff_size: int = 0,
        relational_edge_policy_head: bool = True,
        topology_residual_adapter: bool = False,
        belief_resource_head: bool = False,
        aux_settlement_pointer_head: bool = False,
        static_action_residual: bool = False,
        legal_action_value_residual: bool = False,
        public_card_count_features: bool = False,
        public_card_count_feature_schema: str = (
            PUBLIC_CARD_COUNT_FEATURE_SCHEMA_VERSION
        ),
        public_card_count_residual_bias: bool = True,
        meaningful_public_history: bool = False,
        meaningful_public_history_schema: str = (
            MEANINGFUL_PUBLIC_HISTORY_SCHEMA_VERSION
        ),
        event_history_limit: int = 64,
        meaningful_public_history_pooling: str = MASKED_MEAN_V1,
        entity_feature_adapter_version: str = CURRENT_RUST_ENTITY_ADAPTER_VERSION,
    ) -> EntityGraphPolicy:
        env = ColonistMultiAgentEnv(env_config or ColonistMultiAgentConfig())
        try:
            _observations, info = env.reset(seed=seed)
            static = build_action_feature_table(env)
            config = EntityGraphConfig(
                action_size=env.action_space.n,
                static_action_feature_size=int(static.shape[1]),
                hidden_size=hidden_size,
                state_layers=state_layers,
                attention_heads=attention_heads,
                dropout=dropout,
                action_mask_version=str(info.get("action_mask_version", "")),
                value_uncertainty_head=bool(value_uncertainty_head),
                value_categorical_bins=int(value_categorical_bins),
                edge_policy_head=bool(edge_policy_head),
                aux_subgoal_heads=bool(aux_subgoal_heads),
                aux_vp_horizon=int(aux_vp_horizon),
                state_trunk=str(state_trunk),
                relational_block_pattern=str(relational_block_pattern),
                relational_ff_size=int(relational_ff_size),
                relational_bases=int(relational_bases),
                relational_action_cross_layers=int(relational_action_cross_layers),
                latent_deliberation_steps=int(latent_deliberation_steps),
                latent_deliberation_slots=int(latent_deliberation_slots),
                moe_routed_experts=int(moe_routed_experts),
                moe_top_k=int(moe_top_k),
                moe_expert_ff_size=int(moe_expert_ff_size),
                relational_edge_policy_head=bool(relational_edge_policy_head),
                topology_residual_adapter=bool(topology_residual_adapter),
                belief_resource_head=bool(belief_resource_head),
                aux_settlement_pointer_head=bool(aux_settlement_pointer_head),
                static_action_residual=bool(static_action_residual),
                legal_action_value_residual=bool(
                    legal_action_value_residual
                ),
                public_card_count_features=bool(public_card_count_features),
                public_card_count_feature_schema=str(public_card_count_feature_schema),
                public_card_count_residual_bias=bool(
                    public_card_count_residual_bias
                ),
                meaningful_public_history=bool(meaningful_public_history),
                meaningful_public_history_schema=str(
                    meaningful_public_history_schema
                ),
                event_history_limit=(
                    min(int(event_history_limit), MEANINGFUL_PUBLIC_HISTORY_LIMIT)
                    if meaningful_public_history
                    else int(event_history_limit)
                ),
                meaningful_public_history_pooling=str(
                    meaningful_public_history_pooling or MASKED_MEAN_V1
                ),
            )
            return cls(
                config,
                static,
                seed=seed,
                device=device,
                entity_feature_adapter_version=entity_feature_adapter_version,
            )
        finally:
            env.close()

    def forward_legal_np(
        self,
        entity_batch: dict[str, np.ndarray],
        legal_action_ids: np.ndarray,
        legal_action_context: np.ndarray,
        *,
        return_q: bool = False,
        return_final_vp: bool = True,
        return_aux_subgoals: bool = False,
        value_trunk_grad_scale: float = 1.0,
    ):
        import torch

        entity_batch = _apply_public_award_feature_contract(
            entity_batch, self.public_award_feature_contract
        )
        _assert_entity_batch_shapes(
            entity_batch,
            legal_action_ids,
            legal_action_context,
            self.config,
        )
        # Topology/annotation fields are carried by the feature batch for
        # symmetry and shard-writing code but are not model inputs.  Shape
        # validation above still checks legal_action_mask; omitting these known
        # unused fields here avoids five small synchronous H2D launches per
        # inference window on CUDA; base checkpoints also skip action target
        # ids when neither target-aware policy head is enabled.
        needs_action_targets = bool(
            str(getattr(self.config, "state_trunk", "transformer")) != "transformer"
            or getattr(self.config, "action_target_gather", False)
            or getattr(self.config, "edge_policy_head", False)
        )
        needs_topology = str(
            getattr(self.config, "state_trunk", "transformer")
        ) != "transformer" or bool(
            getattr(self.config, "topology_residual_adapter", False)
        )
        needs_public_card_counts = bool(
            getattr(self.config, "public_card_count_features", False)
        )
        needs_legal_action_mask = bool(
            getattr(self.config, "legal_action_value_residual", False)
        )
        batch = {
            key: torch.as_tensor(value, device=self.device)
            for key, value in entity_batch.items()
            if (
                (
                    key not in _NON_MODEL_ENTITY_KEYS
                    and key != "_symmetry_legal_action_ids"
                )
                or (needs_topology and key in _RELATIONAL_TOPOLOGY_KEYS)
                or (needs_legal_action_mask and key == "legal_action_mask")
            )
            and (key != "legal_action_target_ids" or needs_action_targets)
            and (key != DEDUCTION_FEATURES_KEY or needs_public_card_counts)
        }
        batch["legal_action_context"] = torch.as_tensor(
            legal_action_context,
            dtype=torch.float32,
            device=self.device,
        )
        action_ids = torch.as_tensor(
            legal_action_ids, dtype=torch.long, device=self.device
        )
        valid = action_ids >= 0
        if bool(getattr(self.config, "static_action_residual", False)):
            catalog_rows = int(self.static_action_features.shape[0])
            symmetry_catalog_ids = entity_batch.get("_symmetry_legal_action_ids")
            if symmetry_catalog_ids is None:
                # Production no-symmetry path reuses the resident legal ids.
                legal_ids_np = np.asarray(legal_action_ids)
                if bool(np.any(legal_ids_np[legal_ids_np >= 0] >= catalog_rows)):
                    raise ValueError("static action catalog id is outside catalog rows")
                catalog_ids = torch.where(valid, action_ids, 0)
                catalog_valid = valid
            else:
                # D6 relabels catalog identity without reordering legal rows.
                catalog_ids_np = np.asarray(symmetry_catalog_ids, dtype=np.int64)
                legal_shape = np.asarray(legal_action_ids).shape
                if catalog_ids_np.shape != legal_shape:
                    raise ValueError(
                        "symmetry/static catalog ids must match legal_action_ids: "
                        f"{catalog_ids_np.shape} != {legal_shape}"
                    )
                catalog_valid_np = catalog_ids_np >= 0
                if bool(np.any(catalog_ids_np[catalog_valid_np] >= catalog_rows)):
                    raise ValueError("static action catalog id is outside catalog rows")
                catalog_ids = torch.as_tensor(
                    np.where(catalog_valid_np, catalog_ids_np, 0),
                    dtype=torch.long,
                    device=self.device,
                )
                catalog_valid = torch.as_tensor(
                    catalog_valid_np, dtype=torch.bool, device=self.device
                )
            static_features = self.static_action_features.index_select(
                0, catalog_ids.reshape(-1)
            ).reshape(*catalog_ids.shape, -1)
            static_features = static_features[..., STATIC_ACTION_RESIDUAL_SLICE]
            static_features = static_features.masked_fill(
                ~catalog_valid.unsqueeze(-1), 0.0
            )
            batch["legal_action_static_features"] = static_features
        model_kwargs = {
            "return_q": return_q,
            "return_final_vp": return_final_vp,
            # CAT-100 heads are a learner-only regularizer.  Search and ordinary
            # policy inference discard these tensors, and the settlement pointer
            # applies an h->h MLP independently to all 54 vertex states.  Keep the
            # low-level module's historical full-output default for direct callers,
            # but make the policy/inference API opt in explicitly.
            "return_aux_subgoals": return_aux_subgoals,
        }
        if float(value_trunk_grad_scale) != 1.0:
            model_kwargs["value_trunk_grad_scale"] = float(value_trunk_grad_scale)
        outputs = self.model(batch, **model_kwargs)
        outputs["logits"] = outputs["logits"].masked_fill(~valid, -1.0e9)
        return outputs

    def select_action(
        self,
        env: ColonistMultiAgentEnv,
        observation: np.ndarray,
        info: dict[str, Any],
        rng: np.random.Generator,
        *,
        training: bool = False,
    ) -> int:
        import torch

        del observation
        valid_actions = tuple(int(action) for action in info["valid_actions"])
        if not valid_actions:
            raise ValueError("entity_graph policy received no valid actions")
        with torch.no_grad():
            outputs, _entity, _legal_context = self._legal_outputs_from_env(
                env,
                info,
                valid_actions,
                return_q=False,
            )
            logits = outputs["logits"].squeeze(0)
            if training:
                column = int(
                    torch.distributions.Categorical(logits=logits).sample().item()
                )
            else:
                column = int(torch.argmax(logits, dim=-1).item())
        return int(valid_actions[column])

    def sample_action_value_q_from_env(
        self,
        env: ColonistMultiAgentEnv,
        info: dict[str, Any],
        rng: np.random.Generator,
        *,
        training: bool = True,
        action_temperature: float = 1.0,
    ) -> tuple[int, float, float, float, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        import torch

        del rng
        valid_actions = tuple(int(action) for action in info["valid_actions"])
        if not valid_actions:
            raise ValueError("entity_graph policy received no valid actions")
        with torch.no_grad():
            outputs, entity, _legal_context = self._legal_outputs_from_env(
                env,
                info,
                valid_actions,
                return_q=True,
            )
            logits = outputs["logits"].squeeze(0)
            q_values = outputs.get("q_values")
            if q_values is None:
                q_values = outputs["value"].reshape(1, 1).expand(1, len(valid_actions))
            legal_q_values = q_values.squeeze(0)
            behavior_logits = logits / max(float(action_temperature), 1.0e-6)
            behavior_logits = torch.clamp(behavior_logits, min=-50.0, max=50.0)
            dist = torch.distributions.Categorical(logits=behavior_logits)
            column_t = dist.sample() if training else torch.argmax(logits, dim=-1)
            column = int(column_t.item())
            probs = torch.softmax(behavior_logits, dim=-1)
        entity_copy = {
            key: np.asarray(value).copy()
            for key, value in entity.items()
            if key != "schema"
        }
        return (
            int(valid_actions[column]),
            float(dist.log_prob(column_t).item()),
            float(outputs["value"].reshape(-1)[0].item()),
            float(legal_q_values[column].item()),
            probs.detach().cpu().numpy().astype(np.float32),
            legal_q_values.detach().cpu().numpy().astype(np.float32),
            entity_copy,
        )

    def action_probs(
        self,
        env: ColonistMultiAgentEnv,
        info: dict[str, Any],
        valid_actions: tuple[int, ...] | None = None,
    ) -> np.ndarray:
        import torch

        actions = tuple(
            int(action) for action in (valid_actions or info["valid_actions"])
        )
        if not actions:
            return np.zeros(0, dtype=np.float32)
        with torch.no_grad():
            outputs, _entity, _legal_context = self._legal_outputs_from_env(
                env,
                info,
                actions,
                return_q=False,
            )
            logits = outputs["logits"].squeeze(0)
            probs = torch.softmax(logits, dim=-1)
        return probs.detach().cpu().numpy().astype(np.float32)

    def _legal_outputs_from_env(
        self,
        env: ColonistMultiAgentEnv,
        info: dict[str, Any],
        valid_actions: tuple[int, ...],
        *,
        return_q: bool = False,
    ):
        entity = build_entity_token_features(
            env,
            actor=str(info.get("current_player") or env.current_player_name()),
            include_event_log=True,
            history_limit=int(getattr(self.config, "event_history_limit", 64)),
            meaningful_public_history=bool(
                getattr(self.config, "meaningful_public_history", False)
            ),
        )
        if self.trained_with_masked_hidden_info:
            entity = dict(entity)
            entity["player_tokens"] = mask_player_tokens_public(
                entity["player_tokens"]
            )
        if int(entity["legal_action_tokens"].shape[0]) != len(valid_actions):
            raise ValueError(
                "entity legal-action token count does not match valid actions: "
                f"{entity['legal_action_tokens'].shape[0]} != {len(valid_actions)}"
            )
        context_table = build_action_context_feature_table(env, info)
        legal_context = np.asarray(context_table, dtype=np.float32)[
            list(valid_actions), :
        ]
        entity_batch = {
            key: np.asarray(value)[None, ...]
            for key, value in entity.items()
            if key != "schema"
        }
        outputs = self.forward_legal_np(
            entity_batch,
            np.asarray(valid_actions, dtype=np.int64)[None, :],
            legal_context[None, :, :],
            return_q=return_q,
        )
        return outputs, entity, legal_context

    def save(
        self,
        path: str | Path,
        *,
        mask_hidden_info: bool | None = None,
        soft_target_source: str | None = None,
        value_training: dict[str, object] | None = None,
        training_information_surface: dict[str, object] | None = None,
    ) -> None:
        import torch

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        from catan_zero.rl.config_serialization import config_to_dict

        award_contract = _validate_public_award_feature_contract(
            self.public_award_feature_contract
        )
        durable_mask_hidden_info = (
            bool(self.trained_with_masked_hidden_info)
            if mask_hidden_info is None
            else bool(mask_hidden_info)
        )
        durable_soft_target_source = (
            str(self.soft_target_source)
            if soft_target_source is None
            else str(soft_target_source)
        )
        durable_value_training = (
            dict(self.value_training)
            if value_training is None and isinstance(self.value_training, dict)
            else dict(value_training)
            if value_training is not None
            else None
        )
        durable_information_surface = (
            dict(self.training_information_surface)
            if training_information_surface is None
            and isinstance(self.training_information_surface, dict)
            else dict(training_information_surface)
            if training_information_surface is not None
            else None
        )

        payload = {
            "policy_type": self.policy_type,
            # Durable name-keyed form (task #74): never pickle the frozen+slots
            # dataclass itself -- positional state is crash/shift-prone across
            # field-list changes. Loaders accept both this and legacy pickles.
            "config": config_to_dict(self.config),
            "action_mask_version": str(getattr(self.config, "action_mask_version", "")),
            # f72 safety net (task #76): whether train_bc.py --mask-hidden-info
            # was used for this training run. Absent/False on any checkpoint
            # predating this field (legacy checkpoints deserialize as
            # untrained-with-masking, the safe default -- see
            # EntityGraphRustEvaluator.__init__'s public_observation guard).
            "mask_hidden_info": durable_mask_hidden_info,
            "entity_feature_adapter": (
                checkpoint_entity_feature_adapter_metadata(
                    self.entity_feature_adapter_version
                )
            ),
            "public_award_feature_contract": award_contract,
            # OPT-8 provenance: which soft policy target this run trained
            # against ("policy" = Gumbel visit counts; "prefer_scores" was the
            # degenerate-target footgun). Empty string on checkpoints predating
            # this field. report.json also carries it; this makes the
            # checkpoint self-describing without the sidecar.
            "soft_target_source": durable_soft_target_source,
            "static_action_features_sha256": _array_sha256(
                self.static_action_features.detach().cpu().numpy()
            ),
            "static_action_features": self.static_action_features.detach().cpu(),
            "model": self.model.state_dict(),
        }
        if durable_value_training is not None:
            trained_readouts = tuple(
                str(readout)
                for readout in durable_value_training.get("trained_value_readouts", ())
                if str(readout) in {"scalar", "categorical"}
            )
            payload["value_training"] = durable_value_training
            payload["trained_value_readouts"] = list(trained_readouts)
        if durable_information_surface is not None:
            payload["training_information_surface"] = durable_information_surface
        torch.save(payload, output)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | None = None,
        strict_metadata: bool = True,
        allow_missing_optional_parameters: bool = False,
    ) -> EntityGraphPolicy:
        """Load a complete inference checkpoint.

        Config-enabled optional modules are part of the checkpoint's claimed
        function and therefore must have all of their tensors by default.  A
        caller preparing a deliberate function-preserving warm start may opt
        into freshly initialized optional modules explicitly; ordinary
        inference/evaluation must never turn a truncated or config-only
        upgrade into a silent zero/random adapter.
        """
        import torch

        from catan_zero.rl.config_serialization import (
            config_from_dict as _config_from_dict,
            is_config_dict as _is_config_dict,
        )

        resolved = _resolve_device(device)
        _install_numpy_pickle_aliases()
        checkpoint = Path(path)
        try:
            data = torch.load(checkpoint, map_location=resolved, weights_only=False)
        except TypeError:
            data = torch.load(checkpoint, map_location=resolved)
        adapter_version, adapter_binding_source = (
            resolve_checkpoint_entity_feature_adapter(
                data.get("entity_feature_adapter"),
                metadata_present="entity_feature_adapter" in data,
            )
        )
        static = data["static_action_features"]
        if hasattr(static, "detach"):
            static = static.detach().cpu().numpy()
        config = data["config"]
        # Task #74: reconstruct the config by field NAME from either serialized
        # form -- the durable name-keyed dict (new checkpoints) or the legacy
        # pickled dataclass (old checkpoints; possibly stale with unset slots,
        # the a413df8 case, which this subsumes). Order-independent, fills
        # missing fields from current defaults, warns+drops unknown fields.
        if isinstance(config, EntityGraphConfig) or _is_config_dict(config):
            config = _config_from_dict(EntityGraphConfig, config)
        if strict_metadata:
            if str(data.get("policy_type", "") or "") != cls.policy_type:
                raise ValueError(
                    f"{checkpoint} is not an entity_graph checkpoint: "
                    f"policy_type={data.get('policy_type')!r}"
                )
            if not isinstance(config, EntityGraphConfig):
                raise ValueError(
                    f"{checkpoint} config is {type(config).__name__}, expected EntityGraphConfig"
                )
            if (
                str(getattr(config, "schema_version", "") or "")
                != ENTITY_POLICY_SCHEMA_VERSION
            ):
                raise ValueError(
                    f"{checkpoint} entity policy schema mismatch: "
                    f"{getattr(config, 'schema_version', '')!r} != {ENTITY_POLICY_SCHEMA_VERSION!r}"
                )
            if (
                int(getattr(config, "legal_action_feature_size", 0))
                != LEGAL_ACTION_FEATURE_SIZE
            ):
                raise ValueError(
                    f"{checkpoint} legal_action_feature_size mismatch: "
                    f"{getattr(config, 'legal_action_feature_size', None)} != {LEGAL_ACTION_FEATURE_SIZE}"
                )
            if (
                int(getattr(config, "context_action_feature_size", 0))
                != CONTEXT_ACTION_FEATURE_SIZE
            ):
                raise ValueError(
                    f"{checkpoint} context_action_feature_size mismatch: "
                    f"{getattr(config, 'context_action_feature_size', None)} != {CONTEXT_ACTION_FEATURE_SIZE}"
                )
            expected_static_hash = str(
                data.get("static_action_features_sha256", "") or ""
            )
            if expected_static_hash:
                actual_static_hash = _array_sha256(np.asarray(static, dtype=np.float32))
                if actual_static_hash != expected_static_hash:
                    raise ValueError(
                        f"{checkpoint} static_action_features_sha256 mismatch: "
                        f"checkpoint={expected_static_hash} actual={actual_static_hash}"
                    )
        policy = cls(
            config,
            static,
            device=str(resolved),
            entity_feature_adapter_version=adapter_version,
        )
        # f72 safety net (task #76): record whether this checkpoint's training run
        # used --mask-hidden-info. Missing on any checkpoint saved before this field
        # existed -- defaults to False (untrained-with-masking), the safe default:
        # EntityGraphRustEvaluator.__init__ aborts if public_observation=True is
        # requested against a checkpoint that doesn't report having been trained
        # for it, so an old/legacy checkpoint correctly fails closed rather than
        # silently running mismatched.
        policy.trained_with_masked_hidden_info = bool(
            data.get("mask_hidden_info", False)
        )
        policy.entity_feature_adapter_binding_source = adapter_binding_source
        award_contract = str(
            data.get(
                "public_award_feature_contract",
                PUBLIC_AWARD_FEATURE_CONTRACT_LEGACY_ZERO,
            )
            or ""
        )
        try:
            award_contract = _validate_public_award_feature_contract(award_contract)
        except ValueError as error:
            raise ValueError(f"{checkpoint} has {error}") from error
        policy.public_award_feature_contract = award_contract
        policy.soft_target_source = str(data.get("soft_target_source", "") or "")
        value_training = data.get("value_training")
        policy.value_training = (
            dict(value_training) if isinstance(value_training, dict) else None
        )
        information_surface = data.get("training_information_surface")
        policy.training_information_surface = (
            dict(information_surface) if isinstance(information_surface, dict) else None
        )
        raw_trained_readouts = data.get("trained_value_readouts")
        echo_readouts = (
            tuple(
                str(readout)
                for readout in raw_trained_readouts
                if str(readout) in {"scalar", "categorical"}
            )
            if isinstance(raw_trained_readouts, (list, tuple))
            else ()
        )
        provenance_errors: list[str] = []
        validated_readouts: list[str] = []
        if isinstance(value_training, dict):
            schema = str(value_training.get("schema_version", "") or "")
            inner_raw = value_training.get("trained_value_readouts", ())
            inner_readouts = (
                tuple(
                    str(readout)
                    for readout in inner_raw
                    if str(readout) in {"scalar", "categorical"}
                )
                if isinstance(inner_raw, (list, tuple))
                else ()
            )
            if schema != "value-training-v1":
                provenance_errors.append(
                    f"unsupported value-training schema {schema!r}"
                )
            if set(echo_readouts) != set(inner_readouts):
                provenance_errors.append(
                    "top-level trained_value_readouts does not match "
                    "value_training.trained_value_readouts"
                )
            try:
                optimizer_steps = int(value_training.get("optimizer_steps", 0))
                completed_epochs = int(value_training.get("completed_epochs", 0))
                scalar_weight = float(
                    value_training.get("resolved_scalar_mse_weight", 0.0)
                )
                categorical_weight = float(
                    value_training.get("resolved_categorical_ce_weight", 0.0)
                )
                scalar_mass = float(
                    value_training.get("scalar_training_weight_sum", 0.0)
                )
                categorical_mass = float(
                    value_training.get("categorical_training_weight_sum", 0.0)
                )
                metadata_bins = int(value_training.get("hlgauss_bins", 0))
            except (TypeError, ValueError) as error:
                provenance_errors.append(
                    f"non-numeric value-training metadata: {error}"
                )
                optimizer_steps = completed_epochs = metadata_bins = 0
                scalar_weight = categorical_weight = scalar_mass = categorical_mass = (
                    0.0
                )
            base_valid = (
                schema == "value-training-v1"
                and not provenance_errors
                and optimizer_steps > 0
                # Bounded dose checkpoints are intentionally saved inside the
                # first epoch.  ``completed_epochs == 0`` is therefore valid
                # provenance when applied optimizer steps and positive
                # objective mass attest that the readout was updated.  Keep
                # rejecting impossible negative epoch counts.
                and completed_epochs >= 0
            )
            if "scalar" in inner_readouts and base_valid:
                if scalar_weight > 0.0 and scalar_mass > 0.0:
                    validated_readouts.append("scalar")
                else:
                    provenance_errors.append(
                        "scalar readout is attested without positive objective "
                        "weight and training mass"
                    )
            if "categorical" in inner_readouts and base_valid:
                config_bins = int(getattr(config, "value_categorical_bins", 0) or 0)
                if (
                    categorical_weight > 0.0
                    and categorical_mass > 0.0
                    and metadata_bins >= 2
                    and metadata_bins == config_bins
                ):
                    validated_readouts.append("categorical")
                else:
                    provenance_errors.append(
                        "categorical readout attestation has non-positive weight/mass "
                        "or an HL-Gauss bin-count mismatch"
                    )
        else:
            # Checkpoints predating value-training-v1 establish only the
            # historical scalar head. A config-declared categorical module is
            # never evidence that its random initialization was optimized.
            validated_readouts.append("scalar")
            if "categorical" in echo_readouts:
                provenance_errors.append(
                    "categorical top-level marker has no value-training-v1 record"
                )
        policy.trained_value_readouts = tuple(validated_readouts)
        policy._value_training_provenance_errors = tuple(provenance_errors)
        missing, unexpected = policy.model.load_state_dict(data["model"], strict=False)
        # Preserve load provenance for opt-in consumers that must distinguish a
        # genuinely trained optional head from a config-only warm-start upgrade.
        # The default scalar evaluator ignores this metadata entirely.
        policy._checkpoint_missing_state_keys = tuple(str(key) for key in missing)
        # q_head predates durable architecture metadata and remains an explicit
        # legacy exception; production search does not request it. Every module
        # controlled by a config flag is strict for ordinary loads. The larger
        # allowlist exists only behind the loudly named warm-start opt-in.
        allowed_missing_prefixes = ("q_head.",)
        optional_warmstart_prefixes = (
            "value_uncertainty_head.",
            "value_categorical_head.",
            "value_categorical_support",
            "target_gather_proj.",
            "action_cross_blocks.",
            "value_probe",
            "value_pool_head.",
            # CAT-97 edge-feature policy head + CAT-100 aux subgoal heads: all
            # absent from checkpoints that predate them, so warm-starting an
            # upgraded config (heads ON) from an older checkpoint permits their
            # freshly-initialised weights to be "missing". Inert when off.
            "edge_policy_mlp.",
            "aux_longest_road_head.",
            "aux_largest_army_head.",
            "aux_vp_in_n_head.",
            "aux_next_settlement_head.",
            "aux_robber_target_head.",
            "aux_next_settlement_pointer_head.",
            "belief_resource_head.",
            "topology_residual_adapter.",
            "static_action_residual_proj.",
            "legal_action_value_residual_proj.",
            "legal_action_value_static_proj.",
            "public_card_count_residual.",
            "meaningful_history_residual_gate",
            "meaningful_history_ordered_gate",
            "meaningful_history_sequence.",
        )
        if bool(allow_missing_optional_parameters):
            allowed_missing_prefixes += optional_warmstart_prefixes
        disallowed_missing = [
            key for key in missing if not key.startswith(allowed_missing_prefixes)
        ]
        if disallowed_missing or unexpected:
            raise RuntimeError(
                "entity_graph checkpoint state mismatch: "
                f"missing={disallowed_missing[:8]} unexpected={unexpected[:8]}"
            )
        policy.model.eval()
        return policy


def _assert_entity_batch_shapes(
    entity_batch: dict[str, np.ndarray],
    legal_action_ids: np.ndarray,
    legal_action_context: np.ndarray,
    config: EntityGraphConfig,
) -> None:
    required = {
        "hex_tokens": (3, 19, HEX_FEATURE_SIZE),
        "vertex_tokens": (3, 54, VERTEX_FEATURE_SIZE),
        "edge_tokens": (3, 72, EDGE_FEATURE_SIZE),
        "player_tokens": (3, None, PLAYER_FEATURE_SIZE),
        "global_tokens": (3, 1, GLOBAL_FEATURE_SIZE),
        "legal_action_tokens": (3, None, int(config.legal_action_feature_size)),
        "event_tokens": (3, None, EVENT_FEATURE_SIZE),
    }
    legal = np.asarray(legal_action_ids)
    context = np.asarray(legal_action_context)
    if legal.ndim != 2:
        raise ValueError(f"legal_action_ids must be rank 2, got {legal.shape}")
    if context.ndim != 3:
        raise ValueError(f"legal_action_context must be rank 3, got {context.shape}")
    batch_size, legal_width = int(legal.shape[0]), int(legal.shape[1])
    if context.shape[:2] != legal.shape:
        raise ValueError(
            "legal_action_context shape must align with legal_action_ids: "
            f"context={context.shape} legal={legal.shape}"
        )
    if int(context.shape[2]) != int(config.context_action_feature_size):
        raise ValueError(
            "legal_action_context width mismatch: "
            f"{context.shape[2]} != {config.context_action_feature_size}"
        )
    for key, expected in required.items():
        if key not in entity_batch:
            raise ValueError(f"missing entity batch field {key}")
        value = np.asarray(entity_batch[key])
        if value.ndim != expected[0]:
            raise ValueError(f"{key} must be rank {expected[0]}, got {value.shape}")
        if int(value.shape[0]) != batch_size:
            raise ValueError(f"{key} batch size {value.shape[0]} != {batch_size}")
        if expected[1] is not None and int(value.shape[1]) != int(expected[1]):
            raise ValueError(f"{key} dim1 {value.shape[1]} != {expected[1]}")
        if int(value.shape[2]) != int(expected[2]):
            raise ValueError(f"{key} width {value.shape[2]} != {expected[2]}")
    if bool(getattr(config, "public_card_count_features", False)):
        if DEDUCTION_FEATURES_KEY not in entity_batch:
            raise ValueError(
                "public_card_count_features requires entity batch field "
                f"{DEDUCTION_FEATURES_KEY}"
            )
        deductions = np.asarray(entity_batch[DEDUCTION_FEATURES_KEY])
        expected_deduction_shape = (
            batch_size,
            int(np.asarray(entity_batch["player_tokens"]).shape[1]),
            DEDUCTION_FEATURE_SIZE,
        )
        if deductions.shape != expected_deduction_shape:
            raise ValueError(
                f"{DEDUCTION_FEATURES_KEY} shape {deductions.shape} != "
                f"{expected_deduction_shape}"
            )
        if not bool(np.isfinite(deductions).all()):
            raise ValueError(
                f"{DEDUCTION_FEATURES_KEY} must contain only finite values"
            )
    if np.asarray(entity_batch["legal_action_tokens"]).shape[1] != legal_width:
        raise ValueError(
            "legal_action_tokens candidate width must match legal_action_ids: "
            f"{np.asarray(entity_batch['legal_action_tokens']).shape[1]} != {legal_width}"
        )
    mask_shapes = {
        "hex_mask": 19,
        "vertex_mask": 54,
        "edge_mask": 72,
        "player_mask": np.asarray(entity_batch["player_tokens"]).shape[1],
        "legal_action_mask": legal_width,
        "event_mask": np.asarray(entity_batch["event_tokens"]).shape[1],
    }
    for key, width in mask_shapes.items():
        if key not in entity_batch:
            raise ValueError(f"missing entity batch field {key}")
        value = np.asarray(entity_batch[key])
        if value.shape != (batch_size, int(width)):
            raise ValueError(f"{key} shape {value.shape} != {(batch_size, int(width))}")

    needs_action_targets = (
        str(getattr(config, "state_trunk", "transformer")) != "transformer"
        or bool(getattr(config, "action_target_gather", False))
        or bool(getattr(config, "edge_policy_head", False))
    )
    if needs_action_targets:
        key = "legal_action_target_ids"
        if key not in entity_batch:
            raise ValueError(
                "target-aware policy requires entity batch field "
                "legal_action_target_ids"
            )
        target_ids = np.asarray(entity_batch[key])
        expected_shape = (batch_size, legal_width, 4)
        if target_ids.shape != expected_shape:
            raise ValueError(f"{key} shape {target_ids.shape} != {expected_shape}")
        if not np.issubdtype(target_ids.dtype, np.integer):
            raise ValueError(f"{key} must contain integer per-namespace ids")

        # Columns are LOCAL ids in four disjoint namespaces.  The gather adds
        # the corresponding sequence offsets later; accepting a global token
        # offset here would encode the wrong entity.  Do not clamp malformed
        # ids to the final token (the historical gather did that silently).
        namespace_widths = np.asarray(
            (
                19,
                54,
                72,
                int(np.asarray(entity_batch["player_tokens"]).shape[1]),
            ),
            dtype=np.int64,
        )
        invalid = (target_ids < -1) | (target_ids >= namespace_widths.reshape(1, 1, 4))
        if bool(np.any(invalid)):
            row, action, column = np.argwhere(invalid)[0]
            raise ValueError(
                "legal_action_target_ids contains an out-of-range local id: "
                f"row={int(row)} action={int(action)} column={int(column)} "
                f"value={int(target_ids[row, action, column])} "
                f"namespace_width={int(namespace_widths[column])}"
            )
        padded_has_target = (legal < 0)[..., None] & (target_ids != -1)
        if bool(np.any(padded_has_target)):
            row, action, column = np.argwhere(padded_has_target)[0]
            raise ValueError(
                "padded legal action carries a target id: "
                f"row={int(row)} action={int(action)} column={int(column)}"
            )

    if str(getattr(config, "state_trunk", "transformer")) != "transformer" or bool(
        getattr(config, "topology_residual_adapter", False)
    ):
        topology_shapes = {
            "hex_vertex_ids": (batch_size, 19, 6),
            "hex_edge_ids": (batch_size, 19, 6),
            "edge_vertex_ids": (batch_size, 72, 2),
            "event_target_ids": (
                batch_size,
                int(np.asarray(entity_batch["event_tokens"]).shape[1]),
                4,
            ),
        }
        for key, expected_shape in topology_shapes.items():
            if key not in entity_batch:
                raise ValueError(
                    f"relational state trunk requires entity batch field {key}"
                )
            value = np.asarray(entity_batch[key])
            if value.shape != expected_shape:
                raise ValueError(f"{key} shape {value.shape} != {expected_shape}")
            if not np.issubdtype(value.dtype, np.integer):
                raise ValueError(f"{key} must contain integer per-namespace ids")

        topology_widths: dict[str, np.ndarray] = {
            "hex_vertex_ids": np.asarray((54,), dtype=np.int64),
            "hex_edge_ids": np.asarray((72,), dtype=np.int64),
            "edge_vertex_ids": np.asarray((54,), dtype=np.int64),
            "event_target_ids": np.asarray(
                (
                    19,
                    54,
                    72,
                    int(np.asarray(entity_batch["player_tokens"]).shape[1]),
                ),
                dtype=np.int64,
            ),
        }
        for key, widths in topology_widths.items():
            value = np.asarray(entity_batch[key])
            bound = widths.reshape((1,) * (value.ndim - 1) + (-1,))
            invalid = (value < -1) | (value >= bound)
            if bool(np.any(invalid)):
                index = tuple(int(part) for part in np.argwhere(invalid)[0])
                namespace = index[-1] if widths.size > 1 else 0
                raise ValueError(
                    f"{key} contains an out-of-range local id: index={index} "
                    f"value={int(value[index])} namespace_width={int(widths[namespace])}"
                )
