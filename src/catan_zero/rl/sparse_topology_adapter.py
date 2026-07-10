"""Sparse, warm-startable topology adapters for the incumbent Transformer.

Unlike the experimental relational trunks, this module never materialises a
``[batch, tokens, tokens]`` relation tensor.  It gathers only the fixed Catan
incidence edges, mixes a small set of bottleneck-space basis transforms, and
scatters the messages back to their destination tokens.
"""

from __future__ import annotations

from typing import Any

from catan_zero.rl.relational_trunks import (
    RELATION_COUNT,
    REL_EDGE_TO_HEX,
    REL_EDGE_TO_VERTEX,
    REL_EVENT_TO_TARGET,
    REL_HEX_TO_EDGE,
    REL_HEX_TO_VERTEX,
    REL_TARGET_TO_EVENT,
    REL_VERTEX_TO_EDGE,
    REL_VERTEX_TO_HEX,
)


def build_sparse_incidence_edges(batch: dict[str, Any], *, sequence_length: int):
    """Return padded ``(source, destination, relation, valid)`` edge tensors.

    Token offsets match the entity-token schema.  Every physical incidence is
    emitted in both directions.  Invalid ``-1`` fixture/padding ids remain in
    the fixed-width edge arrays but are masked before gather/scatter.
    """
    import torch

    required = ("hex_vertex_ids", "hex_edge_ids", "edge_vertex_ids")
    missing = [key for key in required if key not in batch]
    if missing:
        raise ValueError(
            "topology adapter requires topology fields: " + ", ".join(missing)
        )

    reference = batch["hex_vertex_ids"]
    device = reference.device
    batch_size = int(reference.shape[0])
    length = int(sequence_length)
    edge_parts: list[tuple[Any, Any, Any, Any]] = []

    def _append_bidirectional(
        ids,
        *,
        row_count: int,
        row_offset: int,
        target_offset: int,
        forward_relation: int,
        reverse_relation: int,
    ) -> None:
        flattened = ids.long().reshape(batch_size, -1)
        fanout = int(flattened.shape[1] // row_count)
        rows = torch.arange(row_count, device=device).view(1, row_count, 1).expand(
            batch_size, row_count, fanout
        ).reshape(batch_size, -1) + int(row_offset)
        targets = flattened + int(target_offset)
        valid = (flattened >= 0) & (rows < length) & (targets < length)
        forward = torch.full_like(rows, int(forward_relation))
        reverse = torch.full_like(rows, int(reverse_relation))
        edge_parts.append((targets, rows, forward, valid))
        edge_parts.append((rows, targets, reverse, valid))

    _append_bidirectional(
        batch["hex_vertex_ids"],
        row_count=19,
        row_offset=1,
        target_offset=20,
        forward_relation=REL_HEX_TO_VERTEX,
        reverse_relation=REL_VERTEX_TO_HEX,
    )
    _append_bidirectional(
        batch["hex_edge_ids"],
        row_count=19,
        row_offset=1,
        target_offset=74,
        forward_relation=REL_HEX_TO_EDGE,
        reverse_relation=REL_EDGE_TO_HEX,
    )
    _append_bidirectional(
        batch["edge_vertex_ids"],
        row_count=72,
        row_offset=74,
        target_offset=20,
        forward_relation=REL_EDGE_TO_VERTEX,
        reverse_relation=REL_VERTEX_TO_EDGE,
    )

    event_offset = 151
    if length > event_offset and "event_target_ids" in batch:
        targets = batch["event_target_ids"].long()
        event_count = min(int(targets.shape[1]), length - event_offset)
        targets = targets[:, :event_count, :]
        # Avoid host-to-device construction inside CUDA Graph capture.
        offsets = torch.empty(4, device=device, dtype=torch.long)
        offsets[0].fill_(1)
        offsets[1].fill_(20)
        offsets[2].fill_(74)
        offsets[3].fill_(146)
        target_tokens = targets + offsets.view(1, 1, 4)
        events = (
            torch.arange(event_count, device=device)
            .view(1, event_count, 1)
            .expand(batch_size, event_count, 4)
            + event_offset
        )
        target_tokens = target_tokens.reshape(batch_size, -1)
        events = events.reshape(batch_size, -1)
        valid = (targets.reshape(batch_size, -1) >= 0) & (target_tokens < length)
        if "event_mask" in batch:
            event_live = (
                batch["event_mask"][:, :event_count]
                .bool()
                .unsqueeze(-1)
                .expand(batch_size, event_count, 4)
                .reshape(batch_size, -1)
            )
            valid = valid & event_live
        forward = torch.full_like(events, REL_EVENT_TO_TARGET)
        reverse = torch.full_like(events, REL_TARGET_TO_EVENT)
        edge_parts.append((target_tokens, events, forward, valid))
        edge_parts.append((events, target_tokens, reverse, valid))

    source = torch.cat([part[0] for part in edge_parts], dim=1)
    destination = torch.cat([part[1] for part in edge_parts], dim=1)
    relation = torch.cat([part[2] for part in edge_parts], dim=1)
    valid = torch.cat([part[3] for part in edge_parts], dim=1)
    # Safe placeholder indices for masked padded edges.
    source = source.masked_fill(~valid, 0)
    destination = destination.masked_fill(~valid, 0)
    return source, destination, relation, valid


def apply_sparse_edge_control(edges, mode: str, *, sequence_length: int):
    """Return an edge-control ablation without changing edge count or kernels.

    ``self_message`` keeps destinations, relations, masks, and scatter work but
    replaces every neighbor source with its receiver. ``type_cyclic_rewire``
    applies a deterministic one-to-one rotation inside each token type, keeping
    source type and the degree multiset while destroying the real Catan
    incidence geometry. These modes are experiment controls, never defaults.
    """

    normalized = str(mode or "true_topology").strip().lower().replace("-", "_")
    if normalized in {"true", "true_topology", "none"}:
        return edges
    source, destination, relation, valid = edges
    if normalized == "self_message":
        return destination.clone(), destination, relation, valid
    if normalized not in {"type_cyclic_rewire", "type_degree_preserving_rewire"}:
        raise ValueError(f"unknown sparse topology edge control: {mode!r}")

    rewired = source.clone()
    token_ranges = (
        (1, 20),
        (20, 74),
        (74, 146),
        (146, 150),
        (150, 151),
        (151, int(sequence_length)),
    )
    for start, end in token_ranges:
        if end <= start:
            continue
        selected = valid & (source >= start) & (source < end)
        rotated = ((source - start + 1) % (end - start)) + start
        rewired = rewired.where(~selected, rotated)
    return rewired, destination, relation, valid


class SparseTopologyAdapter:
    """Factory for a zero-init sparse incidence residual adapter."""

    def __new__(
        cls,
        width: int,
        bottleneck: int,
        bases: int,
        dropout: float,
    ):
        import torch
        from torch import nn
        from torch.nn import functional as F

        if int(bottleneck) < 1:
            raise ValueError("topology adapter bottleneck must be >= 1")
        if int(bases) < 1:
            raise ValueError("topology adapter bases must be >= 1")

        class _Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = nn.RMSNorm(int(width))
                self.down = nn.Linear(int(width), int(bottleneck))
                self.basis_transforms = nn.Parameter(
                    torch.empty(int(bases), int(bottleneck), int(bottleneck))
                )
                self.relation_coefficients = nn.Parameter(
                    torch.empty(RELATION_COUNT, int(bases))
                )
                self.ff_in = nn.Linear(int(bottleneck), 2 * int(bottleneck))
                self.up = nn.Linear(int(bottleneck), int(width))
                self.dropout = nn.Dropout(float(dropout))
                nn.init.xavier_uniform_(self.basis_transforms)
                nn.init.normal_(self.relation_coefficients, std=0.02)
                # Exact incumbent-function preservation at construction and
                # when warm-starting from a checkpoint without adapters.
                nn.init.zeros_(self.up.weight)
                nn.init.zeros_(self.up.bias)

            def forward(
                self,
                x,
                batch: dict[str, Any] | None = None,
                key_padding_mask=None,
                *,
                edges=None,
            ):
                if edges is None:
                    if batch is None:
                        raise ValueError(
                            "topology adapter requires batch or cached edges"
                        )
                    edges = build_sparse_incidence_edges(
                        batch, sequence_length=int(x.shape[1])
                    )
                source, destination, relation, valid = edges
                hidden = self.down(self.norm(x))
                transformed = torch.einsum(
                    "bsi,kio->bsko", hidden, self.basis_transforms
                )
                batch_index = torch.arange(x.shape[0], device=x.device)[:, None]
                edge_basis = transformed[batch_index, source]
                coefficients = self.relation_coefficients[relation]
                messages = torch.einsum("bek,beko->beo", coefficients, edge_basis)
                messages = messages * valid.unsqueeze(-1).to(messages.dtype)

                aggregated = torch.zeros_like(hidden)
                aggregated.scatter_add_(
                    1,
                    destination.unsqueeze(-1).expand_as(messages),
                    messages,
                )
                degree = torch.zeros(
                    hidden.shape[:2], dtype=hidden.dtype, device=hidden.device
                )
                degree.scatter_add_(1, destination, valid.to(hidden.dtype))
                aggregated = aggregated / degree.clamp_min(1).unsqueeze(-1)
                # Preserve the historical basis_mean_v1 checkpoint function.
                # In particular, its biased FF/up path is allowed to emit a
                # learned delta for zero-degree tokens.  Corrected live-token
                # semantics belong to local_attention_v2, not a silent v1
                # state-schema change.
                if key_padding_mask is not None:
                    aggregated = aggregated.masked_fill(
                        key_padding_mask.unsqueeze(-1).bool(), 0.0
                    )
                value, gate = self.ff_in(aggregated).chunk(2, dim=-1)
                return x + self.dropout(self.up(value * F.silu(gate)))

        return _Module()


def _scatter_destination_softmax(logits, destination, valid, *, token_count: int):
    """Normalize edge logits over edges sharing a destination token.

    ``logits`` is ``[batch, edges, heads]``.  The reduction stays sparse in
    the edge dimension and uses float32 for stable exponentiation under AMP.
    """
    import torch

    batch_size, _, heads = logits.shape
    index = destination.unsqueeze(-1).expand(-1, -1, heads)
    live = valid.unsqueeze(-1)
    scores = logits.float().masked_fill(~live, float("-inf"))
    maxima = torch.full(
        (batch_size, int(token_count), heads),
        float("-inf"),
        dtype=scores.dtype,
        device=scores.device,
    )
    maxima.scatter_reduce_(1, index, scores, reduce="amax", include_self=True)
    shifted = scores - maxima.gather(1, index)
    # An all-masked destination produces -inf - -inf above.  Replace those
    # entries before exp so padding can never introduce a NaN.
    shifted = shifted.masked_fill(~live, float("-inf"))
    numerators = shifted.exp()
    denominators = torch.zeros_like(maxima)
    denominators.scatter_add_(1, index, numerators)
    return (numerators / denominators.gather(1, index).clamp_min(1e-12)).to(
        logits.dtype
    )


class SparseTopologyAttentionAdapter:
    """Factory for receiver-conditioned sparse local topology attention.

    Attention is normalized independently for every destination token and
    head.  It never constructs a token-by-token matrix: storage is linear in
    the fixed Catan incidence-edge list.  Relation-specific key and value
    terms distinguish the direction and type of every incidence.

    The output projection is zero initialized, preserving the exact incumbent
    function at construction.  A live-destination mask also guarantees that
    isolated or padded tokens receive no adapter delta after training begins.
    """

    def __new__(
        cls,
        width: int,
        bottleneck: int = 192,
        heads: int = 6,
        dropout: float = 0.0,
    ):
        import torch
        from torch import nn
        from torch.nn import functional as F

        width = int(width)
        bottleneck = int(bottleneck)
        heads = int(heads)
        if bottleneck < 1:
            raise ValueError("topology attention bottleneck must be >= 1")
        if heads < 1:
            raise ValueError("topology attention heads must be >= 1")
        if bottleneck % heads:
            raise ValueError("topology attention bottleneck must be divisible by heads")
        head_width = bottleneck // heads

        class _Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.heads = heads
                self.head_width = head_width
                self.norm = nn.RMSNorm(width)
                self.down = nn.Linear(width, bottleneck)
                self.query = nn.Linear(bottleneck, bottleneck, bias=False)
                self.key = nn.Linear(bottleneck, bottleneck, bias=False)
                self.value = nn.Linear(bottleneck, bottleneck, bias=False)
                self.relation_key = nn.Parameter(
                    torch.empty(RELATION_COUNT, heads, head_width)
                )
                self.relation_value = nn.Parameter(
                    torch.empty(RELATION_COUNT, heads, head_width)
                )
                self.relation_bias = nn.Parameter(torch.zeros(RELATION_COUNT, heads))
                self.receiver_gate = nn.Linear(2 * bottleneck, bottleneck)
                self.ff_in = nn.Linear(bottleneck, 2 * bottleneck)
                self.up = nn.Linear(bottleneck, width)
                self.dropout = nn.Dropout(float(dropout))
                nn.init.normal_(self.relation_key, std=head_width**-0.5)
                nn.init.normal_(self.relation_value, std=head_width**-0.5)
                # Exact incumbent-function preservation at construction and
                # when loading an incumbent checkpoint with missing adapters.
                nn.init.zeros_(self.up.weight)
                nn.init.zeros_(self.up.bias)

            def forward(
                self,
                x,
                batch: dict[str, Any] | None = None,
                key_padding_mask=None,
                *,
                edges=None,
            ):
                if edges is None:
                    if batch is None:
                        raise ValueError(
                            "topology attention adapter requires batch or cached edges"
                        )
                    edges = build_sparse_incidence_edges(
                        batch, sequence_length=int(x.shape[1])
                    )
                source, destination, relation, valid = edges
                if key_padding_mask is not None:
                    padding = key_padding_mask.bool()
                    valid = valid & ~padding.gather(1, source)
                    valid = valid & ~padding.gather(1, destination)

                hidden = self.down(self.norm(x))
                shape = (*hidden.shape[:2], heads, head_width)
                queries = self.query(hidden).view(shape)
                keys = self.key(hidden).view(shape)
                values = self.value(hidden).view(shape)
                batch_index = torch.arange(x.shape[0], device=x.device)[:, None]
                edge_queries = queries[batch_index, destination]
                edge_keys = keys[batch_index, source] + self.relation_key[relation].to(
                    keys.dtype
                )
                edge_values = values[batch_index, source] + self.relation_value[
                    relation
                ].to(values.dtype)
                logits = (edge_queries.float() * edge_keys.float()).sum(dim=-1) * (
                    head_width**-0.5
                ) + self.relation_bias[relation].float()
                weights = _scatter_destination_softmax(
                    logits,
                    destination,
                    valid,
                    token_count=int(x.shape[1]),
                ).to(edge_values.dtype)
                messages = edge_values * weights.unsqueeze(-1)
                messages = messages * valid[:, :, None, None].to(messages.dtype)

                aggregated = torch.zeros(
                    (*hidden.shape[:2], heads, head_width),
                    dtype=hidden.dtype,
                    device=hidden.device,
                )
                aggregated.scatter_add_(
                    1,
                    destination[:, :, None, None].expand_as(messages),
                    messages,
                )
                aggregated = aggregated.flatten(2)
                gate = torch.sigmoid(
                    self.receiver_gate(torch.cat((hidden, aggregated), dim=-1))
                )
                value, glu_gate = self.ff_in(aggregated * gate).chunk(2, dim=-1)
                delta = self.up(value * F.silu(glu_gate))

                live_count = torch.zeros(
                    hidden.shape[:2], dtype=torch.int32, device=hidden.device
                )
                live_count.scatter_add_(1, destination, valid.to(torch.int32))
                delta = delta * live_count.gt(0).unsqueeze(-1).to(delta.dtype)
                return x + self.dropout(delta)

        return _Module()


def create_sparse_topology_adapter(
    *,
    kind: str,
    width: int,
    bottleneck: int,
    dropout: float,
    bases: int = 4,
    heads: int = 6,
):
    """Construct a topology adapter while keeping v1 checkpoints unchanged."""
    normalized = str(kind).strip().lower().replace("_", "-")
    if normalized in {"v1", "basis-mean", "basis-mean-v1", "sparse-basis"}:
        return SparseTopologyAdapter(width, bottleneck, bases, dropout)
    if normalized in {
        "v2",
        "local-attention",
        "local-attention-v2",
        "sparse-attention",
    }:
        return SparseTopologyAttentionAdapter(width, bottleneck, heads, dropout)
    raise ValueError(f"unknown sparse topology adapter kind: {kind!r}")
