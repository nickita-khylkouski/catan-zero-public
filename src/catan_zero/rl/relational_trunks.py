"""Topology-aware state trunks for the entity-token policy.

The modules in this file deliberately consume the incidence tensors already
emitted by :mod:`catan_zero.rl.entity_token_features`.  They do not learn
absolute board-id embeddings.  This keeps D6 augmentation sound: token rows and
incidence ids are transformed together by ``HexSymmetry``.

Two experimental trunks are provided:

``rrt``
    Relational residual attention.  ``R`` blocks attend only over typed Catan
    incidence edges while ``T`` blocks attend globally; both receive a learned
    directed-relation bias.

``resrgcn``
    A no-attention residual relational GNN.  Four learned basis transforms are
    mixed by directed relation type and aggregated with dense batched tensor
    contractions (no Python edge loop or ``index_add_`` in the forward path).

These are architecture experiments, not a claim that either trunk is stronger.
The incumbent Transformer remains the default in ``EntityGraphConfig``.
"""

from __future__ import annotations

from typing import Any


# Directed relation ids. Zero is the global-attention "unrelated" bucket.
# Keep four ids reserved for dynamic piece/port/belief relations so the first
# production checkpoint does not need a relation-table shape migration when
# those ablations are enabled.
RELATION_COUNT = 16
DISTANCE_BUCKETS = 13
REL_NONE = 0
REL_SELF = 1
REL_HEX_TO_VERTEX = 2
REL_VERTEX_TO_HEX = 3
REL_HEX_TO_EDGE = 4
REL_EDGE_TO_HEX = 5
REL_EDGE_TO_VERTEX = 6
REL_VERTEX_TO_EDGE = 7
REL_HUB_READS = 8
REL_READ_GLOBAL = 9
REL_EVENT_TO_TARGET = 10
REL_TARGET_TO_EVENT = 11


class TopologyResidualAdapter:
    """Zero-output, permutation-equivariant incidence adapter.

    This is deliberately smaller than replacing the incumbent Transformer with
    an RRT/ResRGCN trunk.  It performs one directed message-passing step over
    *only* physical board/event incidence edges, then adds a zero-initialised
    projection to the token stream.  Consequently an upgraded checkpoint is
    bit-identical at initialisation, while the projection learns on the first
    optimiser step and opens a topology-aware residual path thereafter.

    The adapter has no absolute entity-id parameters.  Relabelling token rows
    and the accompanying incidence tensors therefore relabels its output in
    exactly the same way (the D6 augmentation contract).
    """

    def __new__(cls, width: int):
        import torch
        from torch import nn

        class _Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.source_norm = nn.LayerNorm(int(width))
                self.source_projection = nn.Linear(int(width), int(width))
                self.message_norm = nn.LayerNorm(int(width))
                self.output_projection = nn.Linear(int(width), int(width))
                nn.init.eye_(self.source_projection.weight)
                nn.init.zeros_(self.source_projection.bias)
                nn.init.zeros_(self.output_projection.weight)
                nn.init.zeros_(self.output_projection.bias)

            def forward(self, tokens, relation_ids, key_padding_mask=None):
                # Direct physical/dynamic incidence only.  Excluding SELF,
                # HUB_READS and READ_GLOBAL prevents this adapter from becoming
                # a second unstructured global-attention layer.
                direct = (
                    ((relation_ids >= REL_HEX_TO_VERTEX)
                     & (relation_ids <= REL_VERTEX_TO_EDGE))
                    | (relation_ids == REL_EVENT_TO_TARGET)
                    | (relation_ids == REL_TARGET_TO_EVENT)
                )
                if key_padding_mask is not None:
                    live = ~key_padding_mask.bool()
                    direct = direct & live.unsqueeze(1) & live.unsqueeze(2)
                adjacency = direct.to(dtype=tokens.dtype)
                degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
                source = self.source_projection(self.source_norm(tokens))
                message = torch.bmm(adjacency, source) / degree
                update = self.output_projection(
                    torch.nn.functional.gelu(self.message_norm(message))
                )
                if key_padding_mask is not None:
                    update = update.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
                return tokens + update

        return _Module()


def build_relation_ids(batch: dict[str, Any], *, sequence_length: int):
    """Build ``[B, destination, source]`` directed relation ids.

    The operation is tensor-only and follows the live incidence arrays, so it
    works after per-row D6 augmentation and does not assume one canonical row
    ordering.  ``event_target_ids`` uses the same four target columns as legal
    actions (hex, vertex, edge, player).
    """
    import torch

    required = ("hex_vertex_ids", "hex_edge_ids", "edge_vertex_ids")
    missing = [key for key in required if key not in batch]
    if missing:
        raise ValueError(
            "relational state trunk requires topology fields: " + ", ".join(missing)
        )

    reference = batch["hex_vertex_ids"]
    device = reference.device
    batch_size = int(reference.shape[0])
    length = int(sequence_length)
    relation = torch.zeros(
        (batch_size, length, length), dtype=torch.long, device=device
    )
    diagonal = torch.arange(length, device=device)
    relation[:, diagonal, diagonal] = REL_SELF

    # Token offsets in [CLS | 19 hex | 54 vertex | 72 edge | 4 player |
    # global | event]. These counts are fixed by the entity schema.
    hex_offset = 1
    vertex_offset = 20
    edge_offset = 74
    player_offset = 146
    global_index = 150
    event_offset = 151

    def _link(
        ids,
        *,
        row_count: int,
        destination_offset: int,
        source_offset: int,
        forward: int,
        reverse: int,
    ) -> None:
        ids = ids.long().reshape(batch_size, -1)
        fanout = int(ids.shape[1] // row_count)
        rows = (
            torch.arange(row_count, device=device)
            .view(1, row_count, 1)
            .expand(batch_size, row_count, fanout)
            .reshape(batch_size, -1)
        )
        valid = ids >= 0
        b = torch.arange(batch_size, device=device).view(-1, 1).expand_as(ids)
        relation[
            b[valid],
            rows[valid] + int(destination_offset),
            ids[valid] + int(source_offset),
        ] = int(forward)
        relation[
            b[valid],
            ids[valid] + int(source_offset),
            rows[valid] + int(destination_offset),
        ] = int(reverse)

    _link(
        batch["hex_vertex_ids"],
        row_count=19,
        destination_offset=hex_offset,
        source_offset=vertex_offset,
        forward=REL_HEX_TO_VERTEX,
        reverse=REL_VERTEX_TO_HEX,
    )
    _link(
        batch["hex_edge_ids"],
        row_count=19,
        destination_offset=hex_offset,
        source_offset=edge_offset,
        forward=REL_HEX_TO_EDGE,
        reverse=REL_EDGE_TO_HEX,
    )
    _link(
        batch["edge_vertex_ids"],
        row_count=72,
        destination_offset=edge_offset,
        source_offset=vertex_offset,
        forward=REL_EDGE_TO_VERTEX,
        reverse=REL_VERTEX_TO_EDGE,
    )

    # CLS, player, and global tokens act as readers/aggregators in local blocks.
    hub_rows = torch.tensor(
        (
            0,
            player_offset,
            player_offset + 1,
            player_offset + 2,
            player_offset + 3,
            global_index,
        ),
        dtype=torch.long,
        device=device,
    )
    unset = relation[:, hub_rows, :] == REL_NONE
    relation[:, hub_rows, :] = torch.where(
        unset,
        torch.full_like(relation[:, hub_rows, :], REL_HUB_READS),
        relation[:, hub_rows, :],
    )
    # Every token may read the global token during a local block.
    unset_global = relation[:, :, global_index] == REL_NONE
    relation[:, :, global_index] = torch.where(
        unset_global,
        torch.full_like(relation[:, :, global_index], REL_READ_GLOBAL),
        relation[:, :, global_index],
    )

    if length > event_offset and "event_target_ids" in batch:
        targets = batch["event_target_ids"].long()
        event_count = min(int(targets.shape[1]), length - event_offset)
        targets = targets[:, :event_count, :]
        offsets = torch.tensor(
            (hex_offset, vertex_offset, edge_offset, player_offset),
            dtype=torch.long,
            device=device,
        )
        valid = targets >= 0
        b = torch.arange(batch_size, device=device).view(-1, 1, 1).expand_as(targets)
        events = (
            torch.arange(event_count, device=device).view(1, -1, 1).expand_as(targets)
            + event_offset
        )
        target_tokens = targets + offsets.view(1, 1, 4)
        relation[b[valid], events[valid], target_tokens[valid]] = REL_EVENT_TO_TARGET
        relation[b[valid], target_tokens[valid], events[valid]] = REL_TARGET_TO_EVENT

    return relation


class RelationalAttention:
    """Factory for a directed-relation attention module."""

    def __new__(cls, width: int, heads: int, dropout: float, *, global_block: bool):
        import torch
        from torch import nn
        from torch.nn import functional as F

        if int(width) % int(heads):
            raise ValueError("relational attention width must be divisible by heads")

        class _Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.heads = int(heads)
                self.head_width = int(width) // int(heads)
                self.dropout = float(dropout)
                self.global_block = bool(global_block)
                self.qkv = nn.Linear(int(width), 3 * int(width))
                self.out = nn.Linear(int(width), int(width))
                self.relation_bias = nn.Embedding(RELATION_COUNT, int(heads))
                self.distance_bias = (
                    nn.Embedding(DISTANCE_BUCKETS, int(heads))
                    if self.global_block
                    else None
                )
                nn.init.zeros_(self.relation_bias.weight)
                if self.distance_bias is not None:
                    nn.init.zeros_(self.distance_bias.weight)

            def forward(self, x, relation_ids, key_padding_mask=None):
                batch_size, length, hidden = x.shape
                qkv = self.qkv(x).reshape(
                    batch_size, length, 3, self.heads, self.head_width
                )
                query, key, value = qkv.unbind(dim=2)
                query = query.transpose(1, 2)
                key = key.transpose(1, 2)
                value = value.transpose(1, 2)
                bias = self.relation_bias(relation_ids).permute(0, 3, 1, 2).float()
                if self.distance_bias is not None:
                    # The drop-in implementation starts with an exact 0/1/far
                    # structural bucket (self/direct-incidence/not-direct).
                    # Full capped all-pairs graph distance remains a separately
                    # measurable ablation; it must not be smuggled into the
                    # topology representation comparison.
                    distance_ids = torch.full_like(relation_ids, 12)
                    distance_ids = torch.where(
                        relation_ids != REL_NONE,
                        torch.ones_like(distance_ids),
                        distance_ids,
                    )
                    distance_ids = torch.where(
                        relation_ids == REL_SELF,
                        torch.zeros_like(distance_ids),
                        distance_ids,
                    )
                    bias = (
                        bias
                        + self.distance_bias(distance_ids).permute(0, 3, 1, 2).float()
                    )
                if not self.global_block:
                    bias = bias.masked_fill(
                        (relation_ids == REL_NONE).unsqueeze(1), float("-inf")
                    )
                if key_padding_mask is not None:
                    bias = bias.masked_fill(
                        key_padding_mask[:, None, None, :].bool(), float("-inf")
                    )
                attended = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=bias,
                    dropout_p=self.dropout if self.training else 0.0,
                )
                return self.out(
                    attended.transpose(1, 2).reshape(batch_size, length, hidden)
                )

        return _Module()


class RelationalTransformerBlock:
    """Pre-norm residual attention + SwiGLU block."""

    def __new__(
        cls,
        width: int,
        heads: int,
        ff_width: int,
        dropout: float,
        *,
        global_block: bool,
    ):
        from torch import nn
        from torch.nn import functional as F

        class _Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm_attn = nn.LayerNorm(int(width))
                self.attn = RelationalAttention(
                    int(width), int(heads), float(dropout), global_block=global_block
                )
                self.norm_ff = nn.LayerNorm(int(width))
                self.ff_in = nn.Linear(int(width), 2 * int(ff_width))
                self.ff_out = nn.Linear(int(ff_width), int(width))
                self.dropout = nn.Dropout(float(dropout))

            def forward(self, x, relation_ids, key_padding_mask=None):
                x = x + self.dropout(
                    self.attn(self.norm_attn(x), relation_ids, key_padding_mask)
                )
                value, gate = self.ff_in(self.norm_ff(x)).chunk(2, dim=-1)
                return x + self.dropout(self.ff_out(value * F.silu(gate)))

        return _Module()


class VectorizedRelGraphBlock:
    """No-attention residual relational message-passing block.

    The per-relation transform is represented as a learned mixture of ``bases``
    matrices.  Aggregation is a batched dense contraction.  That is intentionally
    easy for ``torch.compile``/Inductor to fuse and avoids the many tiny indexed
    CUDA operations used by the initial research probe.
    """

    def __new__(
        cls,
        width: int,
        ff_width: int,
        dropout: float,
        *,
        bases: int = 4,
    ):
        import torch
        from torch import nn
        from torch.nn import functional as F

        class _Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm_message = nn.RMSNorm(int(width))
                self.norm_ff = nn.RMSNorm(int(width))
                self.bases = nn.Parameter(
                    torch.empty(int(bases), int(width), int(width))
                )
                self.relation_coefficients = nn.Parameter(
                    torch.empty(RELATION_COUNT, int(bases))
                )
                self.self_projection = nn.Linear(int(width), int(width))
                self.ff_in = nn.Linear(int(width), 2 * int(ff_width))
                self.ff_out = nn.Linear(int(ff_width), int(width))
                self.dropout = nn.Dropout(float(dropout))
                nn.init.xavier_uniform_(self.bases)
                nn.init.normal_(self.relation_coefficients, std=0.02)

            def forward(self, x, relation_ids, key_padding_mask=None):
                normalized = self.norm_message(x)
                # [B,K,S,H]: one source-token transform per basis.
                transformed = torch.einsum("bsi,kih->bksh", normalized, self.bases)
                coefficients = self.relation_coefficients[relation_ids]
                allowed = relation_ids != REL_NONE
                if key_padding_mask is not None:
                    allowed = allowed & ~key_padding_mask[:, None, :].bool()
                degree = allowed.sum(dim=-1, keepdim=True).clamp_min(1)
                coefficients = (
                    coefficients * allowed.unsqueeze(-1).to(coefficients.dtype)
                ) / degree.unsqueeze(-1).to(coefficients.dtype)
                aggregated = torch.einsum("bdsk,bksh->bdh", coefficients, transformed)
                x = x + self.dropout(self.self_projection(normalized) + aggregated)
                value, gate = self.ff_in(self.norm_ff(x)).chunk(2, dim=-1)
                return x + self.dropout(self.ff_out(value * F.silu(gate)))

        return _Module()


class SparseTopKMoE:
    """One shared SwiGLU expert plus genuinely dispatched top-k experts.

    Only experts selected by at least one live token are called.  The returned
    balance metric is the standard expert-count-scaled dot product between
    mean router probability and hard dispatch load; its ideal balanced value is
    approximately 1.0.  The caller decides whether/how to train against it.
    """

    def __new__(
        cls,
        width: int,
        expert_width: int,
        routed_experts: int,
        top_k: int,
        dropout: float,
    ):
        import torch
        from torch import nn
        from torch.nn import functional as F

        if int(routed_experts) < 2:
            raise ValueError("SparseTopKMoE requires at least two routed experts")
        if not 1 <= int(top_k) <= int(routed_experts):
            raise ValueError("MoE top_k must be in [1, routed_experts]")

        class _Expert(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.in_projection = nn.Linear(int(width), 2 * int(expert_width))
                self.out_projection = nn.Linear(int(expert_width), int(width))

            def forward(self, value):
                content, gate = self.in_projection(value).chunk(2, dim=-1)
                return self.out_projection(content * F.silu(gate))

        class _Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.routed_expert_count = int(routed_experts)
                self.top_k = int(top_k)
                self.shared_expert = _Expert()
                self.routed_experts = nn.ModuleList(
                    _Expert() for _ in range(self.routed_expert_count)
                )
                self.router = nn.Linear(
                    int(width), self.routed_expert_count, bias=False
                )
                self.dropout = nn.Dropout(float(dropout))

            @property
            def one_routed_expert_parameters(self) -> int:
                return sum(
                    parameter.numel()
                    for parameter in self.routed_experts[0].parameters()
                )

            def forward(self, x, live_token_mask=None):
                original_shape = x.shape
                flat = x.reshape(-1, int(width))
                if live_token_mask is None:
                    live = torch.ones(
                        flat.shape[0], dtype=torch.bool, device=flat.device
                    )
                else:
                    live = live_token_mask.reshape(-1).bool()
                live_positions = torch.nonzero(live, as_tuple=False).flatten()
                output = torch.zeros_like(flat)
                if live_positions.numel() == 0:
                    zero = flat.sum() * 0.0
                    load = torch.zeros(
                        self.routed_expert_count,
                        dtype=flat.dtype,
                        device=flat.device,
                    )
                    return output.reshape(original_shape), zero, load, load

                live_values = flat.index_select(0, live_positions)
                router_logits = self.router(live_values).float()
                router_probabilities = torch.softmax(router_logits, dim=-1)
                top_values, top_indices = torch.topk(
                    router_logits, k=self.top_k, dim=-1
                )
                top_weights = torch.softmax(top_values, dim=-1).to(flat.dtype)
                routed = torch.zeros_like(live_values)
                for expert_id, expert in enumerate(self.routed_experts):
                    token_index, slot_index = torch.nonzero(
                        top_indices == expert_id, as_tuple=True
                    )
                    if token_index.numel() == 0:
                        continue
                    expert_input = live_values.index_select(0, token_index)
                    expert_output = expert(expert_input)
                    weighted = expert_output * top_weights[
                        token_index, slot_index
                    ].unsqueeze(-1)
                    routed.index_add_(0, token_index, weighted)

                combined = self.shared_expert(live_values) + routed
                output.index_copy_(0, live_positions, self.dropout(combined))
                importance = router_probabilities.mean(dim=0)
                load = (
                    torch.nn.functional.one_hot(
                        top_indices, num_classes=self.routed_expert_count
                    )
                    .float()
                    .mean(dim=(0, 1))
                )
                balance = self.routed_expert_count * torch.sum(importance * load)
                return output.reshape(original_shape), balance, load, importance

        return _Module()


class SparseMoERelationalTransformerBlock:
    """Relational attention block whose FFN is a dispatched sparse MoE."""

    def __new__(
        cls,
        width: int,
        heads: int,
        expert_width: int,
        routed_experts: int,
        top_k: int,
        dropout: float,
        *,
        global_block: bool,
    ):
        from torch import nn

        class _Module(nn.Module):
            is_sparse_moe = True

            def __init__(self) -> None:
                super().__init__()
                self.norm_attn = nn.LayerNorm(int(width))
                self.attn = RelationalAttention(
                    int(width), int(heads), float(dropout), global_block=global_block
                )
                self.norm_ff = nn.LayerNorm(int(width))
                self.moe = SparseTopKMoE(
                    int(width),
                    int(expert_width),
                    int(routed_experts),
                    int(top_k),
                    float(dropout),
                )
                self.dropout = nn.Dropout(float(dropout))

            def forward(self, x, relation_ids, key_padding_mask=None):
                x = x + self.dropout(
                    self.attn(self.norm_attn(x), relation_ids, key_padding_mask)
                )
                live = None if key_padding_mask is None else ~key_padding_mask.bool()
                update, balance, load, importance = self.moe(
                    self.norm_ff(x), live_token_mask=live
                )
                return x + update, balance, load, importance

        return _Module()
