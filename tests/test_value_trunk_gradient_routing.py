from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from tools import train_bc


def _tiny_model(*, value_attention_pool: bool = False):
    torch = pytest.importorskip("torch")
    from catan_zero.rl.entity_token_policy import EntityGraphPolicy
    from catan_zero.rl.self_play import make_env_config

    policy = EntityGraphPolicy.create(
        env_config=make_env_config(vps_to_win=3),
        hidden_size=16,
        state_layers=1,
        attention_heads=2,
        dropout=0.0,
        seed=4,
        device="cpu",
    )
    if value_attention_pool:
        policy = EntityGraphPolicy(
            replace(policy.config, value_attention_pool=True),
            policy.static_action_features.detach().cpu().numpy(),
            seed=4,
            device="cpu",
        )
    policy.model.eval()
    return torch, policy.model


def _score_fixture(model):
    torch = pytest.importorskip("torch")
    generator = torch.Generator().manual_seed(11)
    batch_size, actions, token_count = 3, 4, 5
    width = int(model.config.hidden_size)
    state = torch.randn(batch_size, width, generator=generator)
    tokens = torch.randn(batch_size, token_count, width, generator=generator)
    padding = torch.zeros(batch_size, token_count, dtype=torch.bool)
    batch = {
        "legal_action_tokens": torch.randn(
            batch_size,
            actions,
            int(model.config.legal_action_feature_size),
            generator=generator,
        ),
        "legal_action_context": torch.randn(
            batch_size,
            actions,
            int(model.config.context_action_feature_size),
            generator=generator,
        ),
    }
    return tokens, padding, state, batch


def _full_model_batch(model):
    torch = pytest.importorskip("torch")
    from catan_zero.rl.entity_token_features import (
        EDGE_FEATURE_SIZE,
        EVENT_FEATURE_SIZE,
        GLOBAL_FEATURE_SIZE,
        HEX_FEATURE_SIZE,
        PLAYER_FEATURE_SIZE,
        VERTEX_FEATURE_SIZE,
    )

    batch_size, actions = 2, 3
    zeros = torch.zeros
    return {
        "hex_tokens": zeros(batch_size, 19, HEX_FEATURE_SIZE),
        "vertex_tokens": zeros(batch_size, 54, VERTEX_FEATURE_SIZE),
        "edge_tokens": zeros(batch_size, 72, EDGE_FEATURE_SIZE),
        "player_tokens": zeros(batch_size, 4, PLAYER_FEATURE_SIZE),
        "global_tokens": zeros(batch_size, 1, GLOBAL_FEATURE_SIZE),
        "event_tokens": zeros(batch_size, 0, EVENT_FEATURE_SIZE),
        "hex_mask": torch.ones(batch_size, 19, dtype=torch.bool),
        "vertex_mask": torch.ones(batch_size, 54, dtype=torch.bool),
        "edge_mask": torch.ones(batch_size, 72, dtype=torch.bool),
        "player_mask": torch.ones(batch_size, 4, dtype=torch.bool),
        "event_mask": torch.zeros(batch_size, 0, dtype=torch.bool),
        "legal_action_tokens": zeros(
            batch_size, actions, int(model.config.legal_action_feature_size)
        ),
        "legal_action_context": zeros(
            batch_size, actions, int(model.config.context_action_feature_size)
        ),
    }


def _backward(model, fixture, *, scale: float, include_policy: bool):
    pytest.importorskip("torch")
    tokens, padding, initial_state, batch = fixture
    state = initial_state.detach().clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    outputs = model.score_actions(
        (tokens, padding, state),
        batch,
        return_final_vp=False,
        value_trunk_grad_scale=scale,
    )
    loss = outputs["value"].sum()
    if include_policy:
        loss = loss + outputs["logits"].square().mean()
    loss.backward()
    head_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.value_head.named_parameters()
    }
    return {
        "outputs": {key: value.detach().clone() for key, value in outputs.items()},
        "state_grad": None if state.grad is None else state.grad.detach().clone(),
        "head_gradients": head_gradients,
    }


def test_scale_zero_is_forward_identical_and_stops_only_value_trunk_gradient() -> None:
    torch, model = _tiny_model()
    fixture = _score_fixture(model)

    full = _backward(model, fixture, scale=1.0, include_policy=False)
    stopped = _backward(model, fixture, scale=0.0, include_policy=False)

    assert torch.equal(full["outputs"]["value"], stopped["outputs"]["value"])
    assert torch.equal(full["outputs"]["logits"], stopped["outputs"]["logits"])
    assert full["state_grad"] is not None
    assert stopped["state_grad"] is None
    assert full["head_gradients"].keys() == stopped["head_gradients"].keys()
    for name in full["head_gradients"]:
        assert torch.equal(
            full["head_gradients"][name], stopped["head_gradients"][name]
        )


def test_fractional_scale_changes_only_upstream_derivative() -> None:
    torch, model = _tiny_model()
    fixture = _score_fixture(model)

    full = _backward(model, fixture, scale=1.0, include_policy=False)
    quarter = _backward(model, fixture, scale=0.25, include_policy=False)

    assert torch.equal(full["outputs"]["value"], quarter["outputs"]["value"])
    assert torch.allclose(quarter["state_grad"], 0.25 * full["state_grad"])
    for name in full["head_gradients"]:
        assert torch.equal(
            full["head_gradients"][name], quarter["head_gradients"][name]
        )


def test_scale_zero_combined_gradient_equals_policy_only_at_shared_state() -> None:
    torch, model = _tiny_model()
    fixture = _score_fixture(model)

    policy_only_model = _backward(model, fixture, scale=0.0, include_policy=True)
    full = _backward(model, fixture, scale=1.0, include_policy=True)
    value_only = _backward(model, fixture, scale=1.0, include_policy=False)

    assert torch.allclose(
        full["state_grad"],
        policy_only_model["state_grad"] + value_only["state_grad"],
    )


def _args(**overrides):
    values = {
        "arch": "entity_graph",
        "value_head_type": "mse",
        "scalar_value_objective": "mse",
        "value_trunk_grad_scale": 1.0,
        "value_target_lambda": 1.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_routing_validation_fails_closed_on_unsupported_paths() -> None:
    default = train_bc._value_trunk_gradient_routing(_args(), scalar_weight=0.25)
    assert default["active"] is False
    assert default["scalar_value_trunk_grad_scale"] == pytest.approx(1.0)

    with pytest.raises(SystemExit, match="entity_graph"):
        train_bc._value_trunk_gradient_routing(
            _args(arch="xdim_graph", value_trunk_grad_scale=0.0),
            scalar_weight=0.25,
        )
    with pytest.raises(SystemExit, match="active value-family"):
        train_bc._value_trunk_gradient_routing(
            _args(value_trunk_grad_scale=0.0), scalar_weight=0.0
        )
    with pytest.raises(SystemExit, match="finite and in"):
        train_bc._value_trunk_gradient_routing(
            _args(value_trunk_grad_scale=float("nan")), scalar_weight=0.25
        )

    _torch, pooled_model = _tiny_model(value_attention_pool=True)
    pooled = train_bc._value_trunk_gradient_routing(
        _args(value_trunk_grad_scale=0.0),
        scalar_weight=0.25,
        model=pooled_model,
    )
    assert pooled["value_attention_pool_enabled"] is True
    assert pooled["all_scalar_value_shared_inputs_scaled"] is True
    assert pooled["all_value_family_shared_inputs_scaled"] is True
    assert pooled["shared_input_paths"] == [
        "cls_state",
        "attention_pool_state",
        "attention_pool_tokens",
    ]


def test_checkpoint_metadata_carries_exact_gradient_routing_contract(tmp_path) -> None:
    args = _args(value_trunk_grad_scale=0.0)
    args.value_gradient_routing = train_bc._value_trunk_gradient_routing(
        args, scalar_weight=0.25
    )
    metadata = train_bc._value_training_metadata(
        args,
        scalar_weight=0.25,
        categorical_weight=0.0,
        categorical_bins=0,
        optimizer_steps=1,
        completed_epochs=1,
        scalar_training_weight_sum=8.0,
        categorical_training_weight_sum=0.0,
    )

    routing = metadata["value_gradient_routing"]
    assert routing["schema_version"] == "scalar-value-trunk-gradient-routing-v1"
    assert routing["scalar_value_trunk_grad_scale"] == pytest.approx(0.0)
    assert routing["forward_value_identity"] is True
    assert routing["value_head_parameter_gradient_scale"] == pytest.approx(1.0)
    assert routing["policy_gradient_unchanged"] is True
    assert "before_standard_ddp_gradient_allreduce" in routing["ddp_semantics"]

    from catan_zero.rl.entity_token_policy import EntityGraphPolicy
    from catan_zero.rl.self_play import make_env_config

    policy = EntityGraphPolicy.create(
        env_config=make_env_config(vps_to_win=3),
        hidden_size=16,
        state_layers=1,
        attention_heads=2,
        dropout=0.0,
        device="cpu",
    )
    checkpoint = tmp_path / "candidate.pt"
    policy.save(checkpoint, value_training=metadata)
    loaded = EntityGraphPolicy.load(checkpoint, device="cpu")
    assert loaded.value_training["value_gradient_routing"] == routing


def test_categorical_value_objective_can_use_trunk_gradient_protection() -> None:
    routing = train_bc._value_trunk_gradient_routing(
        _args(value_head_type="hlgauss", value_trunk_grad_scale=0.1),
        scalar_weight=0.0,
        categorical_weight=0.25,
    )

    assert routing["active"] is True
    assert routing["scope"] == "value_family_readouts_all_shared_inputs"
    assert routing["active_value_objectives"] == {
        "scalar_mse": 0.0,
        "categorical_ce": 0.25,
        "final_vp": 0.0,
    }


def test_binary_win_routing_reports_the_actual_scalar_objective() -> None:
    routing = train_bc._value_trunk_gradient_routing(
        _args(
            scalar_value_objective="binary_win_bce",
            value_trunk_grad_scale=0.25,
        ),
        scalar_weight=0.25,
    )

    assert routing["scalar_value_objective"] == "binary_win_bce"
    assert routing["scalar_value_primary_loss_kind"] == "binary_win_bce"
    assert routing["active_value_objectives"] == {
        "binary_win_bce": 0.25,
        "categorical_ce": 0.0,
        "final_vp": 0.0,
    }
    assert "scalar_mse" not in routing["active_value_objectives"]


def test_train_config_hash_binds_value_trunk_gradient_scale() -> None:
    from catan_zero.rl.pipeline_configs import TrainConfig

    baseline = TrainConfig()
    treatment = replace(baseline, value_trunk_grad_scale=0.0)
    assert baseline.full_config_hash() != treatment.full_config_hash()


def test_ddp_forward_accepts_boundary_scale_and_keeps_both_parameter_paths(
    tmp_path,
) -> None:
    torch, model = _tiny_model()
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed unavailable")
    if torch.distributed.is_initialized():
        pytest.skip("test process already owns a distributed process group")
    rendezvous = tmp_path / "gloo-rendezvous"
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=0,
        world_size=1,
    )
    try:
        ddp = torch.nn.parallel.DistributedDataParallel(model)
        batch = _full_model_batch(model)
        outputs = ddp(
            batch,
            return_final_vp=False,
            value_trunk_grad_scale=0.0,
        )
        (
            outputs["logits"].square().mean() + outputs["value"].square().mean()
        ).backward()
        assert any(
            parameter.grad is not None for parameter in ddp.module.blocks.parameters()
        )
        assert all(
            parameter.grad is not None
            for parameter in ddp.module.value_head.parameters()
        )
    finally:
        torch.distributed.destroy_process_group()
