from __future__ import annotations

import pytest

from catan_zero.rl.ppo_policy_factory import (
    CANONICAL_PPO_ARCHITECTURE,
    canonical_actor_rollout_contract_fields,
    load_exact_parent_and_frozen_anchor,
    load_ppo_policy,
)
from catan_zero.rl.torch_ppo import make_ppo_optimizer
from tools.ppo_distributed_learner import (
    LearnerConfig,
    _validate_w7_config,
    resolve_config,
)


def _config(**overrides) -> LearnerConfig:
    values = {
        "run_base": "runs/distributed",
        "run_name": "w7-test",
        "init_checkpoint": "parent.pt",
    }
    values.update(overrides)
    return LearnerConfig(**values)


def test_canonical_loader_rejects_legacy_architecture_before_loading() -> None:
    with pytest.raises(ValueError, match="canonical PPO requires architecture='entity_graph'"):
        load_ppo_policy("unused.pt", architecture="xdim_graph")


def test_exact_parent_factory_returns_equal_independent_frozen_anchor(monkeypatch) -> None:
    import torch
    from catan_zero.rl.entity_token_policy import EntityGraphPolicy

    class FakePolicy:
        def __init__(self) -> None:
            torch.manual_seed(7)
            self.model = torch.nn.Sequential(
                torch.nn.Linear(3, 4),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(4, 1),
            )

    monkeypatch.setattr(
        EntityGraphPolicy,
        "load",
        lambda checkpoint, device=None: FakePolicy(),
    )

    parent, anchor = load_exact_parent_and_frozen_anchor("parent.pt")

    assert parent is not anchor
    assert parent.model is not anchor.model
    for name, tensor in parent.model.state_dict().items():
        torch.testing.assert_close(tensor, anchor.model.state_dict()[name], rtol=0, atol=0)
    assert parent.model.training
    assert not anchor.model.training
    assert all(parameter.requires_grad for parameter in parent.model.parameters())
    assert all(not parameter.requires_grad for parameter in anchor.model.parameters())
    assert {
        parameter.data_ptr() for parameter in parent.model.parameters()
    }.isdisjoint({parameter.data_ptr() for parameter in anchor.model.parameters()})


def test_w7_defaults_are_canonical() -> None:
    config, _ = resolve_config(["--init-checkpoint", "parent.pt"])

    assert config.architecture == CANONICAL_PPO_ARCHITECTURE
    assert config.gamma == 1.0
    assert config.clip_ratio == 0.1
    assert 0.005 <= config.target_kl <= 0.01
    assert 2 <= config.ppo_epochs <= 4


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"architecture": "xdim_graph"}, "architecture='entity_graph'"),
        ({"gamma": 0.997}, "gamma=1.0"),
        ({"clip_ratio": 0.15}, "clip_ratio=0.1"),
        ({"target_kl": 0.0}, "target_kl"),
        ({"ppo_epochs": 1}, "2-4 update epochs"),
        ({"trunk_lr_mult": 0.25}, "trunk_lr_mult"),
        ({"behavior_temperature": float("nan")}, "temperature"),
        ({"gae_lambda": 0.9}, "gae_lambda"),
        ({"vtrace_clip_rho": 0.0}, "vtrace_clip_rho"),
        ({"vtrace_clip_pg_rho": 1.1}, "vtrace_clip_pg_rho"),
        ({"max_staleness": 5}, "max_staleness"),
        ({"use_vtrace": False, "max_staleness": 1}, "max_staleness=0"),
    ],
)
def test_w7_contract_rejects_unsafe_configuration(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        _validate_w7_config(_config(**overrides))


def test_no_vtrace_accepts_only_version_exact_rollouts() -> None:
    config = _config(use_vtrace=False, max_staleness=0)

    _validate_w7_config(config)


def test_entity_optimizer_protects_shared_trunk_with_lower_lr() -> None:
    import torch

    class FakeEntityPolicy:
        architecture = "entity_graph"

        def __init__(self) -> None:
            self.model = torch.nn.Module()
            self.model.shared_encoder = torch.nn.Linear(3, 3)
            self.model.blocks = torch.nn.Sequential(torch.nn.Linear(3, 3))
            self.model.action_encoder = torch.nn.Linear(3, 3)
            self.model.value_head = torch.nn.Linear(3, 1)

    policy = FakeEntityPolicy()
    optimizer = make_ppo_optimizer(
        policy,
        learning_rate=2.0e-4,
        trunk_lr_mult=0.1,
    )

    groups = {group["name"]: group for group in optimizer.param_groups}
    assert groups["protected_trunk"]["lr"] == pytest.approx(2.0e-5)
    assert groups["policy_value_heads"]["lr"] == pytest.approx(2.0e-4)
    trunk_ids = {id(parameter) for parameter in groups["protected_trunk"]["params"]}
    head_ids = {id(parameter) for parameter in groups["policy_value_heads"]["params"]}
    assert trunk_ids.isdisjoint(head_ids)
    assert trunk_ids | head_ids == {
        id(parameter) for parameter in policy.model.parameters()
    }


def test_real_entity_optimizer_keeps_value_affordance_residual_at_head_lr() -> None:
    from catan_zero.rl.entity_token_policy import EntityGraphPolicy
    from catan_zero.rl.self_play import make_env_config

    policy = EntityGraphPolicy.create(
        env_config=make_env_config(vps_to_win=3),
        hidden_size=16,
        state_layers=1,
        attention_heads=2,
        legal_action_value_residual=True,
        legal_action_value_set_statistics=True,
        seed=0,
    )
    optimizer = make_ppo_optimizer(
        policy,
        learning_rate=2.0e-4,
        trunk_lr_mult=0.1,
    )
    groups = {group["name"]: group for group in optimizer.param_groups}
    head_ids = {id(parameter) for parameter in groups["policy_value_heads"]["params"]}
    trunk_ids = {id(parameter) for parameter in groups["protected_trunk"]["params"]}

    residual_ids = set()
    for module_name in (
        "legal_action_value_residual_proj",
        "legal_action_value_max_proj",
        "legal_action_value_count_proj",
    ):
        residual_ids.update(
            id(parameter)
            for parameter in getattr(policy.model, module_name).parameters()
        )
    encoder_ids = {id(parameter) for parameter in policy.model.hex_encoder.parameters()}
    assert residual_ids <= head_ids
    assert residual_ids.isdisjoint(trunk_ids)
    assert encoder_ids <= trunk_ids


def test_run_contract_binds_initializer_and_behavior_identity(tmp_path) -> None:
    from catan_zero.rl import ppo_distributed as dist

    checkpoint = tmp_path / "parent.pt"
    checkpoint.write_bytes(b"exact-parent-v1")
    root = tmp_path / "run"
    expected = dist.bind_run_contract(
        root,
        init_checkpoint=checkpoint,
        architecture="entity_graph",
        gamma=1.0,
        gae_lambda=0.95,
        behavior_temperature=0.7,
    )

    assert expected["initializer_sha256"] == dist.checkpoint_sha256(checkpoint)
    assert dist.bind_run_contract(
        root,
        init_checkpoint=checkpoint,
        architecture="entity_graph",
        gamma=1.0,
        gae_lambda=0.95,
        behavior_temperature=0.7,
    ) == expected
    with pytest.raises(RuntimeError, match="run contract mismatch"):
        dist.bind_run_contract(
            root,
            init_checkpoint=checkpoint,
            architecture="entity_graph",
            gamma=1.0,
            gae_lambda=0.95,
            behavior_temperature=1.0,
        )
    with pytest.raises(RuntimeError, match="run contract mismatch"):
        dist.bind_run_contract(
            root,
            init_checkpoint=checkpoint,
            architecture="entity",
            gamma=1.0,
            gae_lambda=0.95,
            behavior_temperature=0.7,
        )

    checkpoint.write_bytes(b"mutated-parent")
    with pytest.raises(RuntimeError, match="run contract mismatch"):
        dist.bind_run_contract(
            root,
            init_checkpoint=checkpoint,
            architecture="entity_graph",
            gamma=1.0,
            gae_lambda=0.95,
            behavior_temperature=0.7,
        )


def test_no_vtrace_policy_window_rejects_old_and_future_shards(tmp_path) -> None:
    from catan_zero.rl import ppo_distributed as dist

    root = tmp_path / "run"
    dist.ensure_run_dirs(root)
    current = 8
    old = dist.write_trajectory_shard(root, "old", 0, [], policy_version=current - 1)
    exact = dist.write_trajectory_shard(root, "exact", 0, [], policy_version=current)
    future = dist.write_trajectory_shard(root, "future", 0, [], policy_version=current + 1)

    accepted = list(
        dist.iter_unconsumed_shards(
            root,
            min_policy_version=current,
            max_policy_version=current,
            with_envelope=True,
        )
    )

    assert [path for path, _envelope in accepted] == [exact]
    assert not old.exists()
    assert exact.exists()
    assert not future.exists()


def test_actor_payload_to_chunk_preserves_behavior_temperature_without_modal() -> None:
    top_level_payload = {
        "gamma": 1.0,
        "gae_lambda": 0.97,
        "action_temperature": 0.625,
    }

    chunk_fields = canonical_actor_rollout_contract_fields(top_level_payload)

    assert chunk_fields == top_level_payload
