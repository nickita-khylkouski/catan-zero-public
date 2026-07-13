"""Executable proofs of the incumbent entity-graph information aliases.

These are architectural contracts, not strength tests. The legacy Transformer
has no within-type position/identity embedding and consumes no topology arrays,
so reordering vertex, edge, or indistinguishable opponent-player rows cannot
change its function. A learned target gather can break the action-local part of
that alias without changing the historical baseline at zero initialization.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from catan_zero.rl.action_features import CONTEXT_ACTION_FEATURE_SIZE  # noqa: E402
from catan_zero.rl.entity_token_features import (  # noqa: E402
    EDGE_FEATURE_SIZE,
    EVENT_FEATURE_SIZE,
    GLOBAL_FEATURE_SIZE,
    HEX_FEATURE_SIZE,
    LEGAL_ACTION_FEATURE_SIZE,
    PLAYER_ACTOR_FLAG_SLOT,
    PLAYER_FEATURE_SIZE,
    VERTEX_FEATURE_SIZE,
)
from catan_zero.rl.entity_token_policy import (  # noqa: E402
    EntityGraphConfig,
    EntityGraphNet,
)


def _config(*, target_gather: bool = False) -> EntityGraphConfig:
    return EntityGraphConfig(
        action_size=607,
        static_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
        context_action_feature_size=CONTEXT_ACTION_FEATURE_SIZE,
        legal_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
        hidden_size=32,
        state_layers=2,
        attention_heads=4,
        dropout=0.0,
        action_target_gather=target_gather,
    )


def _batch() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260713)
    result: dict[str, torch.Tensor] = {}
    for name, count, width in (
        ("hex", 19, HEX_FEATURE_SIZE),
        ("vertex", 54, VERTEX_FEATURE_SIZE),
        ("edge", 72, EDGE_FEATURE_SIZE),
        ("player", 4, PLAYER_FEATURE_SIZE),
        ("global", 1, GLOBAL_FEATURE_SIZE),
        ("event", 8, EVENT_FEATURE_SIZE),
    ):
        result[f"{name}_tokens"] = torch.randn(
            2, count, width, generator=generator
        )
        if name != "global":
            result[f"{name}_mask"] = torch.ones(2, count, dtype=torch.bool)

    # A realistic perspective marker: player row 0 is actor/current; the three
    # opponent rows have distinct stats but no seat/color identity feature.
    result["player_tokens"][:, :, PLAYER_ACTOR_FLAG_SLOT] = 0.0
    result["player_tokens"][:, 0, PLAYER_ACTOR_FLAG_SLOT] = 1.0
    result["player_tokens"][:, :, 2] = 0.0
    result["player_tokens"][:, 0, 2] = 1.0

    action_width = 6
    result["legal_action_tokens"] = torch.randn(
        2, action_width, LEGAL_ACTION_FEATURE_SIZE, generator=generator
    )
    result["legal_action_context"] = torch.randn(
        2, action_width, CONTEXT_ACTION_FEATURE_SIZE, generator=generator
    )
    targets = torch.full((2, action_width, 4), -1, dtype=torch.long)
    targets[:, 0::2, 1] = torch.tensor([0, 7, 19])
    targets[:, 1::2, 2] = torch.tensor([0, 11, 31])
    result["legal_action_target_ids"] = targets
    return result


def _permuted_spatial(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result = {name: value.clone() for name, value in batch.items()}
    result["vertex_tokens"] = result["vertex_tokens"][:, torch.arange(53, -1, -1)]
    result["edge_tokens"] = result["edge_tokens"][:, torch.arange(71, -1, -1)]
    # Keep semantic target IDs fixed. This is a different labeled board state,
    # not a simultaneous relabeling of the state and its actions.
    return result


def _permuted_opponents(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result = {name: value.clone() for name, value in batch.items()}
    result["player_tokens"] = result["player_tokens"][:, [0, 3, 1, 2]]
    return result


def _assert_main_close(left: dict, right: dict) -> None:
    for name in ("logits", "value", "final_vp"):
        torch.testing.assert_close(left[name], right[name], rtol=1e-6, atol=1e-6)


def test_legacy_transformer_aliases_spatial_row_permutations() -> None:
    torch.manual_seed(1)
    model = EntityGraphNet(_config()).eval()
    batch = _batch()

    with torch.no_grad():
        original = model(batch)
        permuted = model(_permuted_spatial(batch))

    _assert_main_close(original, permuted)


def test_legacy_transformer_aliases_nonactor_opponent_seats() -> None:
    torch.manual_seed(2)
    model = EntityGraphNet(_config()).eval()
    batch = _batch()

    with torch.no_grad():
        original = model(batch)
        permuted = model(_permuted_opponents(batch))

    _assert_main_close(original, permuted)


def test_learned_target_gather_can_break_action_local_spatial_alias() -> None:
    torch.manual_seed(3)
    baseline = EntityGraphNet(_config()).eval()
    treatment = EntityGraphNet(_config(target_gather=True)).eval()
    missing, unexpected = treatment.load_state_dict(
        baseline.state_dict(), strict=False
    )
    assert unexpected == []
    assert missing and all(name.startswith("target_gather_proj.") for name in missing)

    # The production warm start is exactly zero-output. A nonzero learned
    # projection demonstrates that this treatment's hypothesis class can bind
    # each legal action to the token at its semantic target ID.
    with torch.no_grad():
        projection = treatment.target_gather_proj[1]
        projection.weight.copy_(torch.eye(projection.weight.shape[0]))
        projection.bias.zero_()

    batch = _batch()
    with torch.no_grad():
        original = treatment(batch)
        permuted = treatment(_permuted_spatial(batch))

    assert not torch.allclose(
        original["logits"], permuted["logits"], rtol=1e-5, atol=1e-5
    )
    # Gather repairs action binding only. The CLS/value path still needs a
    # topology adapter or relational trunk to distinguish these state layouts.
    torch.testing.assert_close(
        original["value"], permuted["value"], rtol=1e-6, atol=1e-6
    )


def test_normalized_float16_action_id_is_unique_but_only_one_dimensional() -> None:
    # Avoid overstating the action issue as a literal collision: all incumbent
    # IDs survive fp16. The weakness is that the only ID signal is this arbitrary
    # one-dimensional ordering, not a learned categorical/target-local binding.
    ids = torch.arange(607, dtype=torch.float32)
    encoded = (ids / 607.0).to(torch.float16)
    assert torch.unique(encoded).numel() == 607
