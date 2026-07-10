"""E3 fixed-K latent-compute and E4 sparse-MoE architecture contracts."""

from __future__ import annotations

import dataclasses
import runpy
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from catan_zero.rl.action_features import CONTEXT_ACTION_FEATURE_SIZE  # noqa: E402
from catan_zero.rl.entity_token_features import LEGAL_ACTION_FEATURE_SIZE  # noqa: E402
from catan_zero.rl.entity_token_policy import (  # noqa: E402
    EntityGraphConfig,
    EntityGraphNet,
    EntityGraphPolicy,
)


_RELATIONAL_TEST_HELPERS = runpy.run_path(
    str(Path(__file__).with_name("test_entity_relational_trunks.py"))
)
_batch = _RELATIONAL_TEST_HELPERS["_batch"]
_config = _RELATIONAL_TEST_HELPERS["_config"]


def test_e3_shared_weights_keep_parameter_count_constant_across_k():
    models = [
        EntityGraphNet(_config("rrt", layers=3, latent_deliberation_steps=steps))
        for steps in (1, 2, 4, 8)
    ]
    counts = [
        sum(parameter.numel() for parameter in model.parameters()) for model in models
    ]
    assert len(set(counts)) == 1
    assert all(len(model.deliberation_block.state_dict()) > 0 for model in models)


def test_e3_forward_backward_and_k_changes_computation():
    torch.manual_seed(20260710)
    k1 = EntityGraphNet(_config("rrt", layers=3, latent_deliberation_steps=1)).eval()
    k4 = EntityGraphNet(_config("rrt", layers=3, latent_deliberation_steps=4)).eval()
    k4.load_state_dict(k1.state_dict(), strict=True)
    batch = _batch()

    output_k1 = k1(batch, return_q=True)
    output_k4 = k4(batch, return_q=True)
    assert output_k1["deliberation_halt_logit"].shape == (2,)
    assert not torch.equal(output_k1["logits"], output_k4["logits"])

    loss = output_k4["logits"].square().mean()
    loss = loss + output_k4["value"].square().mean()
    loss = loss + output_k4["deliberation_halt_logit"].square().mean()
    loss.backward()
    assert k4.deliberation_slots.grad is not None
    assert torch.isfinite(k4.deliberation_slots.grad).all()
    assert k4.deliberation_block.attn.in_proj_weight.grad is not None


def test_e4_dispatches_only_selected_experts_and_emits_routing_metrics():
    torch.manual_seed(9)
    model = EntityGraphNet(
        _config(
            "rrt",
            layers=1,
            relational_block_pattern="T",
            moe_routed_experts=4,
            moe_top_k=2,
            moe_expert_ff_size=24,
        )
    ).eval()
    moe = model.blocks[0].moe
    # Equal logits make every token choose the same deterministic two experts.
    # Hooks prove unselected expert modules are not evaluated at all.
    with torch.no_grad():
        moe.router.weight.zero_()
    called: set[int] = set()
    handles = [
        expert.register_forward_hook(
            lambda _module, _inputs, _output, expert_id=expert_id: called.add(expert_id)
        )
        for expert_id, expert in enumerate(moe.routed_experts)
    ]
    try:
        batch = _batch()
        batch["event_mask"][:, -1] = False
        output = model(batch, return_q=True)
    finally:
        for handle in handles:
            handle.remove()

    assert len(called) == 2
    assert output["moe_routing_load"].shape == (1, 4)
    assert output["moe_routing_importance"].shape == (1, 4)
    torch.testing.assert_close(output["moe_routing_load"].sum(), torch.tensor(1.0))
    torch.testing.assert_close(
        output["moe_routing_importance"].sum(), torch.tensor(1.0)
    )
    assert torch.isfinite(output["moe_balance_metric"])

    selected = set(called)
    loss = output["logits"].square().mean() + output["moe_balance_metric"]
    loss.backward()
    for expert_id, expert in enumerate(moe.routed_experts):
        gradients = [parameter.grad for parameter in expert.parameters()]
        if expert_id in selected:
            assert all(gradient is not None for gradient in gradients)
        else:
            assert all(gradient is None for gradient in gradients)
    assert moe.router.weight.grad is not None


def test_moe_parameter_accounting_matches_top2_dispatch():
    common = dict(
        action_size=607,
        static_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
        context_action_feature_size=CONTEXT_ACTION_FEATURE_SIZE,
        legal_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
        hidden_size=384,
        attention_heads=6,
        dropout=0.05,
        value_categorical_bins=0,
        value_categorical_truncation_class=True,
        state_layers=9,
        state_trunk="rrt",
        relational_action_cross_layers=1,
    )
    think_counts = []
    for steps in (1, 2, 4, 8):
        think = EntityGraphNet(
            EntityGraphConfig(**common, latent_deliberation_steps=steps)
        )
        think_counts.append(think.parameter_accounting())
    assert (
        think_counts
        == [
            {
                "instantiated_trainable": 22_146_453,
                "nominal_active_per_token": 22_146_453,
            }
        ]
        * 4
    )

    moe = EntityGraphNet(
        EntityGraphConfig(
            **common,
            moe_routed_experts=8,
            moe_top_k=2,
            moe_expert_ff_size=384,
        )
    )
    assert moe.parameter_accounting() == {
        "instantiated_trainable": 28_508_948,
        "nominal_active_per_token": 20_525_588,
    }


def test_e3_e4_checkpoint_round_trip(tmp_path):
    config = _config(
        "rrt",
        layers=3,
        latent_deliberation_steps=2,
        moe_routed_experts=4,
        moe_top_k=2,
        moe_expert_ff_size=24,
    )
    static = np.zeros(
        (config.action_size, config.static_action_feature_size), dtype=np.float32
    )
    policy = EntityGraphPolicy(config, static, device="cpu")
    checkpoint = tmp_path / "think_moe.pt"
    policy.save(checkpoint)
    loaded = EntityGraphPolicy.load(checkpoint, device="cpu")

    assert loaded.config.latent_deliberation_steps == 2
    assert loaded.config.moe_routed_experts == 4
    assert loaded.config.moe_top_k == 2
    assert loaded.model.parameter_accounting() == policy.model.parameter_accounting()
    for key, value in policy.model.state_dict().items():
        assert torch.equal(value, loaded.model.state_dict()[key]), key


def test_e3_e4_train_cli_knobs_are_explicit_and_science_hashed():
    from catan_zero.rl.pipeline_configs import TrainConfig
    from tools import train_bc

    parser = train_bc.build_parser()
    parsed = parser.parse_args(
        [
            "--data",
            "dummy-data",
            "--checkpoint",
            "dummy-checkpoint.pt",
            "--report",
            "dummy-report.json",
            "--entity-state-trunk",
            "rrt",
            "--latent-deliberation-steps",
            "4",
            "--latent-deliberation-slots",
            "8",
            "--moe-routed-experts",
            "8",
            "--moe-top-k",
            "2",
            "--moe-expert-ff-size",
            "384",
            "--moe-balance-loss-weight",
            "0.02",
        ]
    )
    config = TrainConfig.from_namespace(parsed)
    assert config.latent_deliberation_steps == 4
    assert config.latent_deliberation_slots == 8
    assert config.moe_routed_experts == 8
    assert config.moe_top_k == 2
    assert config.moe_expert_ff_size == 384
    assert config.moe_balance_loss_weight == pytest.approx(0.02)
    assert config.config_hash().startswith("sha256:")


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"latent_deliberation_steps": -1}, "must be >= 0"),
        (
            {"state_trunk": "resrgcn", "latent_deliberation_steps": 1},
            "requires state_trunk='rrt'",
        ),
        (
            {"state_trunk": "resrgcn", "moe_routed_experts": 4},
            "requires state_trunk='rrt'",
        ),
        (
            {"moe_routed_experts": 4, "moe_top_k": 5},
            "moe_top_k",
        ),
        (
            {
                "moe_routed_experts": 4,
                "relational_block_pattern": "RRR",
            },
            "global T block",
        ),
        (
            {"moe_routed_experts": 4, "moe_expert_ff_size": -1},
            "moe_expert_ff_size",
        ),
    ],
)
def test_e3_e4_invalid_configs_fail_loud(overrides, match):
    config = dataclasses.replace(_config("rrt", layers=3), **overrides)
    with pytest.raises(ValueError, match=match):
        EntityGraphNet(config)
