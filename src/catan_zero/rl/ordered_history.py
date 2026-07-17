"""Lightweight order-aware pooling for bounded public Catan history."""

from __future__ import annotations


MASKED_MEAN_V1 = "masked_mean_v1"
ORDERED_ATTENTION_V2 = "ordered_attention_v2"
SUPPORTED_HISTORY_POOLING = frozenset({MASKED_MEAN_V1, ORDERED_ATTENTION_V2})


def build_ordered_history_pool(width: int, max_events: int):
    """Return an O(events * width) masked sequence pool.

    The caller owns the zero-output residual gate that makes architecture
    activation function preserving. Position embeddings start at zero while a
    seeded query immediately distinguishes event content; after the gate opens,
    both the query and absolute event positions receive gradients.
    """

    import torch
    from torch import nn

    if isinstance(width, bool) or int(width) < 1:
        raise ValueError("ordered history width must be positive")
    if isinstance(max_events, bool) or int(max_events) < 1:
        raise ValueError("ordered history max_events must be positive")

    class _OrderedHistoryPool(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.width = int(width)
            self.max_events = int(max_events)
            self.position_embedding = nn.Parameter(
                torch.zeros(self.max_events, self.width)
            )
            self.query = nn.Parameter(torch.empty(self.width))
            nn.init.normal_(self.query, mean=0.0, std=self.width ** -0.5)
            self.norm = nn.LayerNorm(self.width)

        def forward(self, event_tokens, event_mask, *, position_offset=None):
            if event_tokens.ndim != 3 or event_tokens.shape[-1] != self.width:
                raise ValueError("ordered history token shape drift")
            if event_mask.shape != event_tokens.shape[:2]:
                raise ValueError("ordered history mask shape drift")
            sequence = self.encode_sequence(
                event_tokens,
                position_offset=position_offset,
            )
            event_count = int(event_tokens.shape[1])
            if event_count == 0:
                return event_tokens.new_zeros((event_tokens.shape[0], self.width))

            # The query already uses Transformer scaling at initialization
            # (std=width**-0.5). Dividing by sqrt(width) again would make a
            # width-640 adapter almost uniform and starve the ordering path.
            scores = torch.einsum("blh,h->bl", sequence, self.query)
            valid = event_mask.bool()
            scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores, dim=1) * valid.to(scores.dtype)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
            pooled = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
            # Preserve the v1 adapter's explicit event-count mass. Without
            # this factor, a single event and a full 32-event history have the
            # same norm and the ordered branch overstates sparse histories.
            occupancy = valid.sum(dim=1, keepdim=True).to(scores.dtype) / float(
                self.max_events
            )
            return pooled * occupancy

        def encode_sequence(self, event_tokens, *, position_offset=None):
            """Add the same learned order representation without pooling.

            The action decoder uses this view as per-event memory. Keeping the
            transformation here prevents the pooled history path and the
            action-local history path from learning contradictory positions.
            """

            if event_tokens.ndim != 3 or event_tokens.shape[-1] != self.width:
                raise ValueError("ordered history token shape drift")
            event_count = int(event_tokens.shape[1])
            if event_count > self.max_events:
                raise ValueError(
                    "ordered history exceeds configured maximum: "
                    f"{event_count} > {self.max_events}"
                )
            if event_count == 0:
                return event_tokens

            # A standalone semantic window is right-aligned in the configured
            # table. Inference may instead remove a trailing all-padding suffix
            # from a wider physical window; its caller supplies the original
            # offset so retained events keep the same absolute positions.
            if position_offset is None:
                position_offset = self.max_events - event_count
            if (
                isinstance(position_offset, bool)
                or not isinstance(position_offset, int)
                or position_offset < 0
                or position_offset + event_count > self.max_events
            ):
                raise ValueError(
                    "ordered history position range exceeds configured maximum: "
                    f"offset={position_offset} count={event_count} "
                    f"maximum={self.max_events}"
                )
            positions = self.position_embedding[
                position_offset : position_offset + event_count
            ]
            return self.norm(event_tokens + positions.unsqueeze(0))

    return _OrderedHistoryPool()
