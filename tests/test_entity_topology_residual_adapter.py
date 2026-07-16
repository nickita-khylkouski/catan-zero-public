"""Focused contract tests for the function-preserving topology warm-start."""

from __future__ import annotations

import torch

from catan_zero.rl.action_features import CONTEXT_ACTION_FEATURE_SIZE
from catan_zero.rl.entity_token_features import (
    EDGE_FEATURE_SIZE,
    EVENT_FEATURE_SIZE,
    GLOBAL_FEATURE_SIZE,
    HEX_FEATURE_SIZE,
    LEGAL_ACTION_FEATURE_SIZE,
    PLAYER_FEATURE_SIZE,
    VERTEX_FEATURE_SIZE,
)
from catan_zero.rl.entity_token_policy import EntityGraphConfig, EntityGraphNet
from catan_zero.rl.relational_trunks import (
    REL_EDGE_TO_VERTEX,
    REL_VERTEX_TO_EDGE,
    TopologyResidualAdapter,
)


def _config(**overrides) -> EntityGraphConfig:
    values = dict(
        action_size=607,
        static_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
        context_action_feature_size=CONTEXT_ACTION_FEATURE_SIZE,
        legal_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
        hidden_size=32,
        state_layers=2,
        attention_heads=4,
        dropout=0.0,
    )
    values.update(overrides)
    return EntityGraphConfig(**values)


def _batch(batch_size: int = 3, action_width: int = 7, event_width: int = 2):
    generator = torch.Generator().manual_seed(20260712)
    batch = {}
    for name, count, width in (
        ("hex", 19, HEX_FEATURE_SIZE),
        ("vertex", 54, VERTEX_FEATURE_SIZE),
        ("edge", 72, EDGE_FEATURE_SIZE),
        ("player", 4, PLAYER_FEATURE_SIZE),
        ("global", 1, GLOBAL_FEATURE_SIZE),
        ("event", event_width, EVENT_FEATURE_SIZE),
    ):
        batch[f"{name}_tokens"] = torch.randn(
            batch_size, count, width, generator=generator
        )
        if name != "global":
            batch[f"{name}_mask"] = torch.ones(batch_size, count, dtype=torch.bool)

    batch["legal_action_tokens"] = torch.randn(
        batch_size, action_width, LEGAL_ACTION_FEATURE_SIZE, generator=generator
    )
    batch["legal_action_context"] = torch.randn(
        batch_size, action_width, CONTEXT_ACTION_FEATURE_SIZE, generator=generator
    )
    targets = -torch.ones(batch_size, action_width, 4, dtype=torch.long)
    targets[:, :, 1] = torch.arange(action_width).remainder(54)
    batch["legal_action_target_ids"] = targets

    # Deterministic, valid typed incidence.  The adapter consumes these live
    # tensors rather than assuming any absolute vertex/edge ordering.
    batch["hex_vertex_ids"] = (
        torch.arange(19 * 6).reshape(1, 19, 6).remainder(54).repeat(batch_size, 1, 1)
    )
    batch["hex_edge_ids"] = (
        torch.arange(19 * 6).reshape(1, 19, 6).remainder(72).repeat(batch_size, 1, 1)
    )
    edge = torch.arange(72).reshape(1, 72, 1)
    batch["edge_vertex_ids"] = torch.cat((edge.remainder(54), (edge + 1).remainder(54)), 2).repeat(
        batch_size, 1, 1
    )
    batch["event_target_ids"] = -torch.ones(
        batch_size, event_width, 4, dtype=torch.long
    )
    if event_width:
        batch["event_target_ids"][:, 0, 1] = 3
    return batch


def test_combined_topology_gather_upgrade_is_bit_identical_at_init():
    torch.manual_seed(7)
    base = EntityGraphNet(_config()).eval()
    upgraded = EntityGraphNet(
        _config(topology_residual_adapter=True, action_target_gather=True)
    ).eval()
    missing, unexpected = upgraded.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected
    assert missing
    assert all(
        name.startswith(("topology_residual_adapter.", "target_gather_proj."))
        for name in missing
    )

    batch = _batch()
    with torch.no_grad():
        control = base(batch, return_q=True)
        treatment = upgraded(batch, return_q=True)
    assert control.keys() == treatment.keys()
    for name in control:
        assert torch.equal(control[name], treatment[name]), name


def test_topology_treatment_preserves_every_shared_random_initialization():
    torch.manual_seed(20260716)
    control = EntityGraphNet(_config(action_target_gather=True))
    torch.manual_seed(20260716)
    treatment = EntityGraphNet(
        _config(topology_residual_adapter=True, action_target_gather=True)
    )

    control_state = control.state_dict()
    treatment_shared = {
        name: value
        for name, value in treatment.state_dict().items()
        if not name.startswith("topology_residual_adapter.")
    }
    assert control_state.keys() == treatment_shared.keys()
    for name, expected in control_state.items():
        assert torch.equal(expected, treatment_shared[name]), name


def test_topology_gather_and_belief_head_compose_without_main_output_drift():
    torch.manual_seed(17)
    base = EntityGraphNet(_config()).eval()
    upgraded = EntityGraphNet(
        _config(
            topology_residual_adapter=True,
            action_target_gather=True,
            belief_resource_head=True,
        )
    ).eval()
    missing, unexpected = upgraded.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected
    assert missing
    assert all(
        name.startswith(
            (
                "topology_residual_adapter.",
                "target_gather_proj.",
                "belief_resource_head.",
            )
        )
        for name in missing
    )

    batch = _batch()
    with torch.no_grad():
        control = base(batch, return_q=True)
        treatment = upgraded(batch, return_q=True)
    assert treatment["belief_resource_logits"].shape == (3, 4, 5)
    for name, expected in control.items():
        assert torch.equal(expected, treatment[name]), name


def test_topology_output_projection_learns_on_first_step():
    model = EntityGraphNet(
        _config(topology_residual_adapter=True, action_target_gather=True)
    ).train()
    outputs = model(_batch())
    loss = torch.nn.functional.cross_entropy(outputs["logits"], torch.tensor([0, 1, 2]))
    loss.backward()

    projection = model.topology_residual_adapter.output_projection
    assert torch.count_nonzero(projection.weight).item() == 0
    assert projection.weight.grad is not None
    assert projection.weight.grad.abs().sum().item() > 0.0


def test_adapter_is_equivariant_to_joint_token_and_topology_relabelling():
    width, length = 8, 9
    adapter = TopologyResidualAdapter(width).eval()
    with torch.no_grad():
        adapter.output_projection.weight.copy_(torch.eye(width))
    tokens = torch.randn(2, length, width, generator=torch.Generator().manual_seed(11))
    relations = torch.zeros(2, length, length, dtype=torch.long)
    for row in range(length):
        nxt = (row + 1) % length
        relations[:, row, nxt] = REL_EDGE_TO_VERTEX
        relations[:, nxt, row] = REL_VERTEX_TO_EDGE
    permutation = torch.tensor([3, 7, 1, 8, 0, 6, 4, 2, 5])

    expected = adapter(tokens, relations)[:, permutation]
    relabelled = adapter(
        tokens[:, permutation],
        relations[:, permutation][:, :, permutation],
    )
    torch.testing.assert_close(relabelled, expected, rtol=1e-6, atol=1e-6)


def test_adapter_output_changes_when_explicit_vertex_edge_incidence_changes():
    model = EntityGraphNet(_config(topology_residual_adapter=True)).eval()
    with torch.no_grad():
        model.topology_residual_adapter.output_projection.weight.copy_(
            torch.eye(model.config.hidden_size)
        )
    left = _batch(batch_size=1)
    right = {name: value.clone() for name, value in left.items()}
    right["edge_vertex_ids"][0, 0] = torch.tensor([17, 29])
    with torch.no_grad():
        left_tokens = model.encode_state(left)[0]
        right_tokens = model.encode_state(right)[0]
    assert not torch.equal(left_tokens, right_tokens)


def test_adapter_does_not_update_isolated_or_padded_tokens_after_training():
    width, length = 8, 5
    adapter = TopologyResidualAdapter(width).eval()
    with torch.no_grad():
        # Exercise the post-training case where both normalization and output
        # biases are nonzero.  Those biases must still not create a residual
        # on destinations outside the direct-incidence information surface.
        adapter.message_norm.bias.fill_(0.7)
        adapter.output_projection.weight.copy_(torch.eye(width))
        adapter.output_projection.bias.fill_(0.3)
    tokens = torch.randn(
        1, length, width, generator=torch.Generator().manual_seed(101)
    )
    relations = torch.zeros(1, length, length, dtype=torch.long)
    relations[:, 0, 1] = REL_EDGE_TO_VERTEX
    relations[:, 1, 0] = REL_VERTEX_TO_EDGE
    padding = torch.tensor([[False, False, False, False, True]])

    output = adapter(tokens, relations, key_padding_mask=padding)

    assert not torch.equal(output[:, :2], tokens[:, :2])
    torch.testing.assert_close(output[:, 2:], tokens[:, 2:], rtol=0.0, atol=0.0)
