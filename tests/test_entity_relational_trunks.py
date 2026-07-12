"""Opt-in topology-aware EntityGraph state-trunk contracts."""

from __future__ import annotations

import dataclasses

import numpy as np
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
    EntityGraphPolicy,
)
from catan_zero.rl.relational_trunks import (  # noqa: E402
    REL_HEX_TO_VERTEX,
    REL_VERTEX_TO_HEX,
    build_relation_ids,
)


def _config(trunk: str, *, layers: int = 3, **overrides) -> EntityGraphConfig:
    values = dict(
        action_size=64,
        static_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
        context_action_feature_size=CONTEXT_ACTION_FEATURE_SIZE,
        legal_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
        hidden_size=32,
        state_layers=layers,
        attention_heads=4,
        dropout=0.0,
        state_trunk=trunk,
        relational_ff_size=48,
        relational_action_cross_layers=1,
    )
    values.update(overrides)
    return EntityGraphConfig(**values)


def _batch(*, batch_size: int = 2, actions: int = 4, events: int = 2):
    generator = torch.Generator().manual_seed(20260710)
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
    targets = torch.full((batch_size, actions, 4), -1, dtype=torch.long)
    targets[:, :, 1] = torch.arange(actions).remainder(54)
    batch["legal_action_target_ids"] = targets

    # A small valid subgraph is sufficient for architecture/gradient tests. The
    # remaining -1 entries mean "no incidence in this synthetic fixture".
    batch["hex_vertex_ids"] = torch.full((batch_size, 19, 6), -1, dtype=torch.long)
    batch["hex_edge_ids"] = torch.full((batch_size, 19, 6), -1, dtype=torch.long)
    batch["edge_vertex_ids"] = torch.full((batch_size, 72, 2), -1, dtype=torch.long)
    batch["hex_vertex_ids"][:, 0, :2] = torch.tensor((0, 1))
    batch["hex_edge_ids"][:, 0, :2] = torch.tensor((0, 1))
    batch["edge_vertex_ids"][:, 0, :] = torch.tensor((0, 1))
    batch["event_target_ids"] = torch.full(
        (batch_size, events, 4), -1, dtype=torch.long
    )
    if events:
        batch["event_target_ids"][:, 0, 1] = 0
    return batch


def _numpy_entity(batch):
    out = {key: value.detach().cpu().numpy() for key, value in batch.items()}
    out["legal_action_mask"] = np.ones(
        out["legal_action_tokens"].shape[:2], dtype=np.bool_
    )
    return out


def test_default_trunk_and_state_dict_remain_incumbent_compatible():
    base = EntityGraphNet(_config("transformer"))
    explicit = EntityGraphNet(
        dataclasses.replace(
            _config("transformer"),
            relational_block_pattern="",
            relational_ff_size=0,
            relational_bases=4,
            relational_action_cross_layers=1,
        )
    )
    assert base.state_trunk == "transformer"
    assert set(base.state_dict()) == set(explicit.state_dict())
    explicit.load_state_dict(base.state_dict(), strict=True)


def test_relational_probe_can_exclude_direct_edge_policy_head():
    historical = EntityGraphNet(_config("rrt"))
    isolated = EntityGraphNet(
        _config("rrt", relational_edge_policy_head=False)
    )
    assert historical.action_target_gather is True
    assert historical.action_cross_attention_layers == 1
    assert historical.edge_policy_head is True
    assert isolated.action_target_gather is True
    assert isolated.action_cross_attention_layers == 1
    assert isolated.edge_policy_head is False
    assert not hasattr(isolated, "edge_policy_mlp")


def test_relation_builder_uses_live_directed_incidence():
    batch = _batch(batch_size=1, events=0)
    relation = build_relation_ids(batch, sequence_length=151)
    # token 1 = hex 0; token 20 = vertex 0.
    assert relation[0, 1, 20].item() == REL_HEX_TO_VERTEX
    assert relation[0, 20, 1].item() == REL_VERTEX_TO_HEX


@pytest.mark.parametrize("trunk,layers", [("rrt", 3), ("resrgcn", 2)])
def test_relational_forward_backward_and_target_binding(trunk: str, layers: int):
    torch.manual_seed(19)
    model = EntityGraphNet(_config(trunk, layers=layers)).eval()
    batch = _batch()
    output = model(batch, return_q=True)
    assert output["logits"].shape == (2, 4)
    assert output["value"].shape == (2,)
    assert output["q_values"].shape == (2, 4)

    loss = output["logits"].float().square().mean()
    loss = loss + output["value"].float().square().mean()
    loss = loss + output["q_values"].float().square().mean()
    loss.backward()
    if trunk == "rrt":
        gradient = model.blocks[0].attn.relation_bias.weight.grad
    else:
        gradient = model.blocks[0].relation_coefficients.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()

    # Keep every feature fixed and only change action 0's target vertex. The
    # relational decoder must observe the identity change in its policy logit.
    changed = {key: value.clone() for key, value in batch.items()}
    changed["legal_action_target_ids"][:, 0, 1] = 17
    with torch.no_grad():
        changed_output = model(changed)
    assert not torch.equal(output["logits"][:, 0], changed_output["logits"][:, 0])


def test_policy_transport_retains_topology_only_for_relational_trunks():
    config = _config("rrt", layers=1, relational_block_pattern="T")
    policy = EntityGraphPolicy(
        config,
        np.zeros((config.action_size, config.static_action_feature_size), np.float32),
        device="cpu",
    )
    batch = _batch(batch_size=1)
    entity = _numpy_entity(batch)
    legal_ids = np.arange(4, dtype=np.int64)[None, :]
    outputs = policy.forward_legal_np(
        entity,
        legal_ids,
        entity.pop("legal_action_context"),
    )
    assert outputs["logits"].shape == (1, 4)


def test_relational_checkpoint_round_trip_preserves_architecture(tmp_path):
    config = _config("resrgcn", layers=1, relational_action_cross_layers=0)
    static = np.zeros(
        (config.action_size, config.static_action_feature_size), np.float32
    )
    policy = EntityGraphPolicy(config, static, device="cpu")
    checkpoint = tmp_path / "relational.pt"
    policy.save(checkpoint)

    loaded = EntityGraphPolicy.load(checkpoint, device="cpu")
    assert loaded.config.state_trunk == "resrgcn"
    assert loaded.config.relational_ff_size == 48
    assert loaded.model.state_trunk == "resrgcn"
    assert set(loaded.model.state_dict()) == set(policy.model.state_dict())
    for key, value in policy.model.state_dict().items():
        assert torch.equal(value, loaded.model.state_dict()[key]), key


@pytest.mark.parametrize(
    ("categorical_bins", "rrt_count", "resrgcn_count"),
    [
        (0, 20_070_932, 20_936_010),
        (51, 20_238_792, 21_103_870),
    ],
)
def test_exact_production_shaped_parameter_counts(
    categorical_bins: int, rrt_count: int, resrgcn_count: int
):
    common = dict(
        action_size=607,
        static_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
        context_action_feature_size=CONTEXT_ACTION_FEATURE_SIZE,
        legal_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
        hidden_size=384,
        attention_heads=6,
        dropout=0.05,
        value_categorical_bins=categorical_bins,
        value_categorical_truncation_class=True,
    )
    rrt = EntityGraphNet(EntityGraphConfig(**common, state_layers=9, state_trunk="rrt"))
    assert rrt.relational_block_pattern == "RRTRRTRRT"
    assert sum(parameter.numel() for parameter in rrt.parameters()) == rrt_count

    # Production-shaped GNN retains the policy's Q/final-VP/value conventions;
    # the standalone 20.512M research probe deliberately omitted those heads.
    resrgcn = EntityGraphNet(
        EntityGraphConfig(
            **common,
            state_layers=14,
            state_trunk="resrgcn",
            relational_action_cross_layers=0,
        )
    )
    assert sum(parameter.numel() for parameter in resrgcn.parameters()) == resrgcn_count


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"state_trunk": "unknown"}, "state_trunk"),
        (
            {"state_trunk": "rrt", "state_layers": 3, "relational_block_pattern": "RT"},
            "exactly state_layers",
        ),
        (
            {"state_trunk": "resrgcn", "relational_block_pattern": "RRT"},
            "only valid",
        ),
    ],
)
def test_invalid_relational_configs_fail_loud(overrides, match):
    config = dataclasses.replace(_config("transformer"), **overrides)
    with pytest.raises(ValueError, match=match):
        EntityGraphNet(config)
