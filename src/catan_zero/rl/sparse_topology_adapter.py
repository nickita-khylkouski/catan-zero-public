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
        offsets = torch.tensor((1, 20, 74, 146), device=device, dtype=torch.long)
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
                        raise ValueError("topology adapter requires batch or cached edges")
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
                if key_padding_mask is not None:
                    aggregated = aggregated.masked_fill(
                        key_padding_mask.unsqueeze(-1).bool(), 0.0
                    )
                value, gate = self.ff_in(aggregated).chunk(2, dim=-1)
                return x + self.dropout(self.up(value * F.silu(gate)))

        return _Module()
