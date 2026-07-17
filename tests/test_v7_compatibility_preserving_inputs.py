from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from catan_zero.rl.action_features import CONTEXT_ACTION_FEATURE_SIZE  # noqa: E402
from catan_zero.rl.entity_token_features import (  # noqa: E402
    ACTION_TYPES,
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


def _config(*, compatible: bool) -> EntityGraphConfig:
    return EntityGraphConfig(
        action_size=32,
        static_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
        context_action_feature_size=CONTEXT_ACTION_FEATURE_SIZE,
        legal_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
        hidden_size=16,
        state_layers=1,
        attention_heads=2,
        dropout=0.0,
        action_cross_attention_layers=1,
        v6_compatibility_preserving_inputs=compatible,
    )


def _legacy_batch() -> dict[str, torch.Tensor]:
    batch: dict[str, torch.Tensor] = {}
    for name, count, width in (
        ("hex", 19, HEX_FEATURE_SIZE),
        ("vertex", 54, VERTEX_FEATURE_SIZE),
        ("edge", 72, EDGE_FEATURE_SIZE),
        ("player", 4, PLAYER_FEATURE_SIZE),
        ("global", 1, GLOBAL_FEATURE_SIZE),
        ("event", 64, EVENT_FEATURE_SIZE),
    ):
        batch[f"{name}_tokens"] = torch.zeros(
            1, count, width, dtype=torch.float16
        )
        if name != "global":
            batch[f"{name}_mask"] = torch.ones(1, count, dtype=torch.bool)

    # Use non-clipped values so this also proves the route reproduces the
    # historical float16 normalization, not merely the saturation boundary.
    # The actor has two wood.
    batch["player_tokens"][0, 0, 6] = 2.0 / 20.0
    batch["player_tokens"][0, 0, 15] = 1.0
    batch["player_tokens"][0, 0, 16] = 2.0 / 10.0

    batch["legal_action_tokens"] = torch.zeros(1, 2, LEGAL_ACTION_FEATURE_SIZE)
    build_road_column = 2 + ACTION_TYPES.index("BUILD_ROAD")
    batch["legal_action_tokens"][0, 0, build_road_column] = 1.0
    batch["legal_action_context"] = torch.zeros(1, 2, CONTEXT_ACTION_FEATURE_SIZE)
    batch["legal_action_context"][0, 0, 12] = 1.0
    batch["legal_action_context"][0, 0, 16] = 5.0 / 18.0

    batch["legal_action_target_ids"] = torch.full((1, 2, 4), -1, dtype=torch.long)
    batch["legal_action_target_ids"][0, 0, 2] = 0
    batch["edge_vertex_ids"] = torch.full((1, 72, 2), -1, dtype=torch.long)
    batch["edge_vertex_ids"][0, 0] = torch.tensor([0, 1])
    # Endpoint 0 contains the just-built settlement. Endpoint 1 is empty and
    # has five legacy production pips.
    batch["vertex_tokens"][0, 0, 1] = 0.0
    batch["vertex_tokens"][0, 0, 9] = 0.4
    batch["vertex_tokens"][0, 1, 1] = 1.0
    batch["vertex_tokens"][0, 1, 6] = 1.0
    batch["vertex_tokens"][0, 1, 9] = 5.0 / 18.0
    return batch


def _v6_batch() -> dict[str, torch.Tensor]:
    batch = {key: value.clone() for key, value in _legacy_batch().items()}
    batch["player_tokens"][0, 0, 6] = 2.0 / 95.0
    batch["player_tokens"][0, 0, 16] = 2.0 / 19.0
    # Correct V6 two-hop expansion differs from the inherited endpoint score.
    batch["legal_action_context"][0, 0, 16] = 0.8
    return batch


def test_v5_to_v7_route_preserves_inherited_forward_exactly() -> None:
    torch.manual_seed(20260717)
    incumbent = EntityGraphNet(_config(compatible=False)).eval()
    upgraded = EntityGraphNet(_config(compatible=True)).eval()
    missing, unexpected = upgraded.load_state_dict(incumbent.state_dict(), strict=False)

    assert unexpected == []
    assert set(missing) == {
        "v6_exact_resource_residual.weight",
        "v6_initial_road_residual.weight",
    }
    with torch.no_grad():
        before = incumbent(_legacy_batch())
        after = upgraded(_v6_batch())
    for key in ("logits", "value", "final_vp"):
        assert torch.equal(before[key], after[key]), key


def test_v7_new_information_paths_receive_first_backward_gradients() -> None:
    torch.manual_seed(20260717)
    model = EntityGraphNet(_config(compatible=True)).train()
    output = model(_v6_batch())
    loss = output["logits"].square().mean() + output["value"].square().mean()
    loss.backward()

    resource_gradient = model.v6_exact_resource_residual.weight.grad
    road_gradient = model.v6_initial_road_residual.weight.grad
    assert resource_gradient is not None
    assert road_gradient is not None
    assert float(resource_gradient.abs().sum()) > 0.0
    assert float(road_gradient.abs().sum()) > 0.0


def test_v7_route_requires_topology_for_legacy_road_reconstruction() -> None:
    model = EntityGraphNet(_config(compatible=True)).eval()
    batch = _v6_batch()
    del batch["edge_vertex_ids"]

    with pytest.raises(ValueError, match="edge topology"):
        model(batch)
