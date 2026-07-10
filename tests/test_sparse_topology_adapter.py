"""Contracts for the warm-startable incumbent topology adapter."""

from __future__ import annotations

import dataclasses

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
from catan_zero.rl.entity_token_policy import EntityGraphConfig, EntityGraphNet  # noqa: E402
from catan_zero.rl.relational_trunks import build_relation_ids  # noqa: E402
from catan_zero.rl.sparse_topology_adapter import (  # noqa: E402
    apply_sparse_edge_control,
    build_sparse_incidence_edges,
)


def _config(**overrides) -> EntityGraphConfig:
    values = dict(
        action_size=64,
        static_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
        context_action_feature_size=CONTEXT_ACTION_FEATURE_SIZE,
        legal_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
        hidden_size=32,
        state_layers=3,
        attention_heads=4,
        dropout=0.0,
    )
    values.update(overrides)
    return EntityGraphConfig(**values)


def _batch(batch_size: int = 2, actions: int = 4, events: int = 2):
    generator = torch.Generator().manual_seed(97)
    batch = {}
    for name, count, width in (
        ("hex", 19, HEX_FEATURE_SIZE),
        ("vertex", 54, VERTEX_FEATURE_SIZE),
        ("edge", 72, EDGE_FEATURE_SIZE),
        ("player", 4, PLAYER_FEATURE_SIZE),
        ("global", 1, GLOBAL_FEATURE_SIZE),
        ("event", events, EVENT_FEATURE_SIZE),
    ):
        batch[f"{name}_tokens"] = torch.randn(
            batch_size, count, width, generator=generator
        )
        if name != "global":
            batch[f"{name}_mask"] = torch.ones(batch_size, count, dtype=torch.bool)
    batch["legal_action_tokens"] = torch.randn(
        batch_size, actions, LEGAL_ACTION_FEATURE_SIZE, generator=generator
    )
    batch["legal_action_context"] = torch.randn(
        batch_size, actions, CONTEXT_ACTION_FEATURE_SIZE, generator=generator
    )
    batch["legal_action_target_ids"] = torch.full(
        (batch_size, actions, 4), -1, dtype=torch.long
    )
    batch["hex_vertex_ids"] = torch.full((batch_size, 19, 6), -1, dtype=torch.long)
    batch["hex_edge_ids"] = torch.full((batch_size, 19, 6), -1, dtype=torch.long)
    batch["edge_vertex_ids"] = torch.full((batch_size, 72, 2), -1, dtype=torch.long)
    batch["event_target_ids"] = torch.full(
        (batch_size, events, 4), -1, dtype=torch.long
    )
    batch["hex_vertex_ids"][:, 0, :2] = torch.tensor((0, 1))
    batch["hex_edge_ids"][:, 0, :2] = torch.tensor((0, 1))
    batch["edge_vertex_ids"][:, 0, :] = torch.tensor((0, 1))
    if events:
        batch["event_target_ids"][:, 0, 1] = 0
    return batch


def test_adapter_is_exact_identity_when_warm_started_from_incumbent():
    torch.manual_seed(11)
    incumbent = EntityGraphNet(_config()).eval()
    hybrid = EntityGraphNet(
        dataclasses.replace(
            _config(),
            topology_adapter_layers="1,3",
            topology_adapter_width=24,
            topology_adapter_bases=2,
        )
    ).eval()
    missing, unexpected = hybrid.load_state_dict(incumbent.state_dict(), strict=False)
    assert missing
    assert all(name.startswith("topology_adapters.") for name in missing)
    assert unexpected == []

    batch = _batch()
    with torch.no_grad():
        base_output = incumbent(batch, return_q=True)
        hybrid_output = hybrid(batch, return_q=True)
    for key in ("logits", "value", "final_vp", "q_values"):
        assert torch.equal(base_output[key], hybrid_output[key]), key


def test_adapter_does_not_enable_relational_action_heads_and_gets_gradient():
    model = EntityGraphNet(
        _config(topology_adapter_layers="2", topology_adapter_width=24)
    )
    assert model.uses_topology_adapters
    assert not model.uses_relational_topology
    assert not model.action_target_gather
    assert model.action_cross_attention_layers == 0
    assert not model.edge_policy_head
    assert not hasattr(model, "target_gather_proj")
    assert not hasattr(model, "action_cross_blocks")
    assert not hasattr(model, "edge_policy_mlp")

    output = model(_batch(), return_q=True)
    loss = output["logits"].square().mean() + output["value"].square().mean()
    loss.backward()
    up_gradient = model.topology_adapters["2"].up.weight.grad
    assert up_gradient is not None
    assert torch.isfinite(up_gradient).all()
    assert torch.count_nonzero(up_gradient) > 0


def test_basis_mean_v1_preserves_legacy_zero_degree_bias_path():
    from catan_zero.rl.sparse_topology_adapter import SparseTopologyAdapter

    adapter = SparseTopologyAdapter(width=4, bottleneck=2, bases=1, dropout=0.0)
    with torch.no_grad():
        adapter.ff_in.weight.zero_()
        adapter.ff_in.bias.copy_(torch.tensor((1.0, 1.0, 1.0, 1.0)))
        adapter.up.weight.fill_(1.0)
        adapter.up.bias.zero_()
    x = torch.zeros((1, 3, 4))
    no_edges = (
        torch.zeros((1, 1), dtype=torch.long),
        torch.zeros((1, 1), dtype=torch.long),
        torch.zeros((1, 1), dtype=torch.long),
        torch.zeros((1, 1), dtype=torch.bool),
    )

    result = adapter(x, edges=no_edges)

    assert torch.count_nonzero(result) == result.numel()


def test_sparse_edges_match_dense_relation_direction_and_ids():
    batch = _batch(batch_size=2, events=2)
    sequence_length = 153
    dense = build_relation_ids(batch, sequence_length=sequence_length)
    source, destination, relation, valid = build_sparse_incidence_edges(
        batch, sequence_length=sequence_length
    )
    batch_index = torch.arange(source.shape[0])[:, None].expand_as(source)

    assert valid.any()
    assert torch.equal(
        dense[batch_index[valid], destination[valid], source[valid]],
        relation[valid],
    )


def test_edge_controls_preserve_work_and_isolate_neighbor_geometry():
    edges = build_sparse_incidence_edges(_batch(batch_size=1), sequence_length=153)
    source, destination, relation, valid = edges
    self_edges = apply_sparse_edge_control(edges, "self_message", sequence_length=153)
    rewired = apply_sparse_edge_control(
        edges, "type_degree_preserving_rewire", sequence_length=153
    )

    assert torch.equal(self_edges[0][valid], destination[valid])
    for controlled in (self_edges, rewired):
        assert controlled[0].shape == source.shape
        assert torch.equal(controlled[1], destination)
        assert torch.equal(controlled[2], relation)
        assert torch.equal(controlled[3], valid)
    assert torch.any(rewired[0][valid] != source[valid])

    def token_type(values):
        boundaries = torch.tensor((1, 20, 74, 146, 150, 151, 153))
        return torch.bucketize(values, boundaries, right=True)

    assert torch.equal(token_type(rewired[0][valid]), token_type(source[valid]))


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"topology_adapter_layers": "2,2"}, "duplicates"),
        ({"topology_adapter_layers": "4"}, "state_layers"),
        (
            {"state_trunk": "rrt", "topology_adapter_layers": "2"},
            "only valid",
        ),
    ],
)
def test_invalid_adapter_configs_fail_loud(overrides, match):
    with pytest.raises(ValueError, match=match):
        EntityGraphNet(dataclasses.replace(_config(), **overrides))


@pytest.mark.parametrize(
    ("hidden", "layers", "adapters", "kind", "width", "categorical_bins", "expected"),
    [
        (640, 6, "", "basis_mean_v1", 448, 0, 35_041_353),
        (640, 6, "2,4", "basis_mean_v1", 448, 0, 38_602_057),
        (640, 6, "2,4", "basis_mean_v1", 192, 0, 35_979_081),
        (640, 6, "2,4", "local_attention_v2", 192, 0, 36_065_481),
        (832, 6, "", "basis_mean_v1", 448, 0, 59_131_977),
        (832, 10, "", "basis_mean_v1", 448, 0, 92_401_993),
        (640, 6, "", "basis_mean_v1", 448, 51, 35_484_925),
        (640, 6, "2,4", "basis_mean_v1", 448, 51, 39_045_629),
        (832, 6, "", "basis_mean_v1", 448, 51, 59_868_349),
        (832, 10, "", "basis_mean_v1", 448, 51, 93_138_365),
    ],
)
def test_exact_scaled_ladder_parameter_counts(
    hidden: int,
    layers: int,
    adapters: str,
    kind: str,
    width: int,
    categorical_bins: int,
    expected: int,
):
    config = EntityGraphConfig(
        action_size=607,
        static_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
        context_action_feature_size=CONTEXT_ACTION_FEATURE_SIZE,
        legal_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
        hidden_size=hidden,
        state_layers=layers,
        attention_heads=8,
        value_categorical_bins=categorical_bins,
        topology_adapter_layers=adapters,
        topology_adapter_width=width,
        topology_adapter_bases=4,
        topology_adapter_kind=kind,
        topology_adapter_heads=4,
    )
    model = EntityGraphNet(config)
    assert sum(parameter.numel() for parameter in model.parameters()) == expected


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("basis_mean_v1", 35_510_217),
        ("local_attention_v2", 35_553_417),
    ],
)
def test_shared_adapter_parameter_counts(kind: str, expected: int):
    model = EntityGraphNet(
        EntityGraphConfig(
            action_size=607,
            static_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
            context_action_feature_size=CONTEXT_ACTION_FEATURE_SIZE,
            legal_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
            hidden_size=640,
            state_layers=6,
            attention_heads=8,
            topology_adapter_layers="2,4",
            topology_adapter_width=192,
            topology_adapter_kind=kind,
            topology_adapter_heads=4,
            topology_adapter_share_weights=True,
        )
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == expected
