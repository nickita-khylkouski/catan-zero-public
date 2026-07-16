from __future__ import annotations

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from catan_zero.rl.action_features import CONTEXT_ACTION_FEATURE_SIZE  # noqa: E402
from catan_zero.rl.entity_token_features import (  # noqa: E402
    EDGE_FEATURE_SIZE,
    EVENT_FEATURE_SIZE,
    GLOBAL_FEATURE_SIZE,
    HEX_FEATURE_SIZE,
    LEGAL_ACTION_FEATURE_SIZE,
    PLAYER_FEATURE_SIZE,
    VERTEX_FEATURE_SIZE,
)
from catan_zero.rl.entity_token_policy import (  # noqa: E402
    EntityGraphConfig,
    EntityGraphNet,
)


def _config(*, target_gather: bool) -> EntityGraphConfig:
    return EntityGraphConfig(
        action_size=32,
        static_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
        context_action_feature_size=CONTEXT_ACTION_FEATURE_SIZE,
        legal_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
        hidden_size=16,
        state_layers=1,
        attention_heads=2,
        dropout=0.0,
        meaningful_public_history=True,
        event_history_limit=4,
        meaningful_public_history_target_gather=target_gather,
    )


def _batch(*, target_vertex: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260716)
    batch: dict[str, torch.Tensor] = {}
    for name, count, width in (
        ("hex", 19, HEX_FEATURE_SIZE),
        ("vertex", 54, VERTEX_FEATURE_SIZE),
        ("edge", 72, EDGE_FEATURE_SIZE),
        ("player", 4, PLAYER_FEATURE_SIZE),
        ("global", 1, GLOBAL_FEATURE_SIZE),
    ):
        batch[f"{name}_tokens"] = torch.randn(
            1, count, width, generator=generator
        )
        if name != "global":
            batch[f"{name}_mask"] = torch.ones(1, count, dtype=torch.bool)
    batch["legal_action_tokens"] = torch.randn(
        1, 2, LEGAL_ACTION_FEATURE_SIZE, generator=generator
    )
    batch["legal_action_context"] = torch.randn(
        1, 2, CONTEXT_ACTION_FEATURE_SIZE, generator=generator
    )
    batch["event_tokens"] = torch.zeros(1, 4, EVENT_FEATURE_SIZE)
    batch["event_tokens"][0, -1, 0] = 1.0
    batch["event_mask"] = torch.tensor([[False, False, False, True]])
    batch["event_target_ids"] = torch.full((1, 4, 4), -1, dtype=torch.long)
    batch["event_target_ids"][0, -1, 1] = int(target_vertex)
    return batch


def test_history_target_gather_upgrade_is_function_preserving() -> None:
    torch.manual_seed(17)
    incumbent = EntityGraphNet(_config(target_gather=False)).eval()
    upgraded = EntityGraphNet(_config(target_gather=True)).eval()
    missing, unexpected = upgraded.load_state_dict(
        incumbent.state_dict(), strict=False
    )

    assert unexpected == []
    assert set(missing) == {
        "meaningful_history_target_proj.0.weight",
        "meaningful_history_target_proj.0.bias",
        "meaningful_history_target_proj.1.weight",
    }
    with torch.no_grad():
        before = incumbent(_batch(target_vertex=0))
        after = upgraded(_batch(target_vertex=53))
    for key in ("logits", "value", "final_vp"):
        assert torch.equal(before[key], after[key]), key


def test_history_target_gather_can_bind_event_to_board_entity() -> None:
    torch.manual_seed(19)
    model = EntityGraphNet(_config(target_gather=True)).eval()
    with torch.no_grad():
        model.meaningful_history_residual_gate.fill_(1.0)
        model.meaningful_history_target_proj[1].weight.copy_(
            torch.eye(model.meaningful_history_target_proj[1].weight.shape[0])
        )
        first = model(_batch(target_vertex=0))
        second = model(_batch(target_vertex=53))

    assert not torch.equal(first["value"], second["value"])
    assert not torch.equal(first["logits"], second["logits"])


def test_history_target_gather_requires_history() -> None:
    with pytest.raises(
        ValueError,
        match="public-history target gather requires meaningful_public_history=True",
    ):
        from catan_zero.rl.entity_token_policy import EntityGraphPolicy

        config = replace(
            _config(target_gather=True),
            meaningful_public_history=False,
        )
        EntityGraphPolicy(
            config,
            torch.zeros(32, LEGAL_ACTION_FEATURE_SIZE).numpy(),
            device="cpu",
        )
