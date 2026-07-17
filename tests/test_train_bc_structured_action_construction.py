from __future__ import annotations

from types import SimpleNamespace

import pytest

from catan_zero.rl.entity_token_policy import EntityGraphPolicy
from catan_zero.rl.pipeline_configs import TrainConfig
from catan_zero.rl.self_play import make_env_config
from tools.train_bc import (
    _checkpoint_config_mismatches,
    _effective_a1_learner_training_recipe,
    _resolve_effective_action_cross_attention_bottleneck,
    _resolve_effective_v6_compatibility_preserving_inputs,
    _resolve_effective_action_cross_attention_layers,
    _resolve_effective_action_target_gather,
    _resolve_effective_structured_action_residuals,
    _resolve_effective_topology_residual_adapter,
    _structured_action_create_kwargs,
    build_parser,
)


def _args(
    *,
    arch: str = "entity_graph",
    static: bool | None = None,
    legal: bool | None = None,
    set_stats: bool | None = None,
    init_checkpoint: str = "",
    grow_from_checkpoint: str = "",
) -> SimpleNamespace:
    return SimpleNamespace(
        arch=arch,
        static_action_residual=static,
        legal_action_value_residual=legal,
        legal_action_value_set_statistics=set_stats,
        init_checkpoint=init_checkpoint,
        grow_from_checkpoint=grow_from_checkpoint,
    )


def _small_policy(
    *,
    static: bool,
    legal: bool,
    set_stats: bool = False,
    target_gather: bool = False,
) -> EntityGraphPolicy:
    return EntityGraphPolicy.create(
        env_config=make_env_config(vps_to_win=3),
        hidden_size=16,
        state_layers=1,
        attention_heads=2,
        action_target_gather=target_gather,
        static_action_residual=static,
        legal_action_value_residual=legal,
        legal_action_value_set_statistics=set_stats,
        seed=7,
        device="cpu",
    )


def test_fresh_cli_flags_reach_policy_construction_and_typed_identity() -> None:
    parser = build_parser()
    parsed = parser.parse_args(
        [
            "--data",
            "corpus",
            "--checkpoint",
            "candidate.pt",
            "--report",
            "report.json",
            "--arch",
            "entity_graph",
            "--action-target-gather",
            "--action-cross-attention-layers",
            "1",
            "--action-cross-attention-bottleneck",
            "80",
            "--v6-compatibility-preserving-inputs",
            "--entity-feature-adapter-version",
            "rust_entity_adapter_v6_exact_actor_resources_initial_road_two_hop",
            "--public-rule-state-features",
            "--meaningful-public-history",
            "--event-history-limit",
            "64",
            "--static-action-residual",
            "--legal-action-value-residual",
            "--legal-action-value-set-statistics",
        ]
    )
    assert parser.get_default("action_target_gather") is None
    assert parser.get_default("action_cross_attention_layers") is None
    assert parser.get_default("action_cross_attention_bottleneck") is None
    assert parser.get_default("v6_compatibility_preserving_inputs") is None
    assert parser.get_default("topology_residual_adapter") is None
    assert parser.get_default("static_action_residual") is None
    assert parser.get_default("legal_action_value_residual") is None
    assert parser.get_default("legal_action_value_set_statistics") is None
    (
        parsed.static_action_residual,
        parsed.legal_action_value_residual,
        parsed.legal_action_value_set_statistics,
    ) = _resolve_effective_structured_action_residuals(parsed)
    parsed.action_target_gather = _resolve_effective_action_target_gather(parsed)
    parsed.action_cross_attention_layers = (
        _resolve_effective_action_cross_attention_layers(parsed)
    )
    parsed.action_cross_attention_bottleneck = (
        _resolve_effective_action_cross_attention_bottleneck(parsed)
    )
    parsed.v6_compatibility_preserving_inputs = (
        _resolve_effective_v6_compatibility_preserving_inputs(parsed)
    )
    parsed.topology_residual_adapter = _resolve_effective_topology_residual_adapter(
        parsed
    )

    kwargs = _structured_action_create_kwargs(parsed)
    assert kwargs["action_cross_attention_layers"] == 1
    assert kwargs["action_cross_attention_bottleneck"] == 80
    assert kwargs == {
        "action_target_gather": True,
        "action_cross_attention_layers": 1,
        "action_cross_attention_bottleneck": 80,
        "v6_compatibility_preserving_inputs": True,
        "static_action_residual": True,
        "legal_action_value_residual": True,
        "legal_action_value_set_statistics": True,
    }
    policy = EntityGraphPolicy.create(
        env_config=make_env_config(vps_to_win=3),
        hidden_size=16,
        state_layers=1,
        attention_heads=2,
        seed=7,
        device="cpu",
        entity_feature_adapter_version=parsed.entity_feature_adapter_version,
        **kwargs,
    )
    assert policy.config.action_target_gather is True
    assert policy.config.action_cross_attention_layers == 1
    assert policy.config.action_cross_attention_bottleneck == 80
    assert policy.config.v6_compatibility_preserving_inputs is True
    assert policy.config.static_action_residual is True
    assert policy.config.legal_action_value_residual is True
    assert policy.config.legal_action_value_set_statistics is True
    assert hasattr(policy.model, "static_action_residual_proj")
    assert hasattr(policy.model, "legal_action_value_residual_proj")
    assert hasattr(policy.model, "legal_action_value_count_proj")

    identity = TrainConfig.from_namespace(parsed)
    assert identity.action_target_gather is True
    assert identity.action_cross_attention_layers == 1
    assert identity.action_cross_attention_bottleneck == 80
    assert identity.v6_compatibility_preserving_inputs is True
    assert identity.topology_residual_adapter is False
    assert identity.static_action_residual is True
    assert identity.legal_action_value_residual is True
    assert identity.legal_action_value_set_statistics is True


def test_fresh_default_is_legacy_off_and_non_entity_rejects_enablement() -> None:
    assert _resolve_effective_action_target_gather(_args()) is False
    action_args = _args(arch="xdim_graph")
    action_args.action_target_gather = True
    with pytest.raises(SystemExit, match="only for --arch entity_graph"):
        _resolve_effective_action_target_gather(action_args)
    assert _resolve_effective_structured_action_residuals(_args()) == (
        False,
        False,
        False,
    )
    with pytest.raises(SystemExit, match="only for --arch entity_graph"):
        _resolve_effective_structured_action_residuals(
            _args(arch="xdim_graph", static=True)
        )
    with pytest.raises(
        SystemExit, match="requires --legal-action-value-residual"
    ):
        _resolve_effective_structured_action_residuals(
            _args(legal=False, set_stats=True)
        )


def test_init_checkpoint_inherits_and_refuses_architecture_drift(tmp_path) -> None:
    checkpoint = tmp_path / "structured.pt"
    _small_policy(
        static=True, legal=True, set_stats=True, target_gather=True
    ).save(checkpoint)

    action_args = _args(init_checkpoint=str(checkpoint))
    assert _resolve_effective_action_target_gather(action_args) is True
    action_args.action_target_gather = False
    with pytest.raises(SystemExit, match="does not match --init-checkpoint"):
        _resolve_effective_action_target_gather(action_args)
    assert _resolve_effective_structured_action_residuals(
        _args(init_checkpoint=str(checkpoint))
    ) == (True, True, True)
    with pytest.raises(SystemExit, match="structured_action_value"):
        _resolve_effective_structured_action_residuals(
            _args(
                init_checkpoint=str(checkpoint),
                static=False,
                legal=True,
                set_stats=True,
            )
        )


def test_action_cross_attention_is_checkpoint_owned_and_trunk_specific(
    tmp_path,
) -> None:
    checkpoint = tmp_path / "action-cross.pt"
    EntityGraphPolicy.create(
        env_config=make_env_config(vps_to_win=3),
        hidden_size=16,
        state_layers=1,
        attention_heads=2,
        action_cross_attention_layers=1,
        seed=7,
        device="cpu",
    ).save(checkpoint)
    parsed = build_parser().parse_args(
        [
            "--data",
            "corpus",
            "--checkpoint",
            "candidate.pt",
            "--report",
            "report.json",
            "--arch",
            "entity_graph",
            "--init-checkpoint",
            str(checkpoint),
        ]
    )
    assert _resolve_effective_action_cross_attention_layers(parsed) == 1
    parsed.action_cross_attention_layers = 1
    assert _resolve_effective_action_cross_attention_bottleneck(parsed) == 0
    parsed.action_cross_attention_layers = 0
    with pytest.raises(SystemExit, match="does not match --init-checkpoint"):
        _resolve_effective_action_cross_attention_layers(parsed)

    relational = build_parser().parse_args(
        [
            "--data",
            "corpus",
            "--checkpoint",
            "candidate.pt",
            "--report",
            "report.json",
            "--arch",
            "entity_graph",
            "--entity-state-trunk",
            "rrt",
            "--action-cross-attention-layers",
            "1",
        ]
    )
    with pytest.raises(SystemExit, match="relational-action-cross-layers"):
        _resolve_effective_action_cross_attention_layers(relational)


def test_topology_residual_inherits_and_binds_typed_science_identity(tmp_path) -> None:
    checkpoint = tmp_path / "topology.pt"
    EntityGraphPolicy.create(
        env_config=make_env_config(vps_to_win=3),
        hidden_size=16,
        state_layers=1,
        attention_heads=2,
        topology_residual_adapter=True,
        seed=7,
        device="cpu",
    ).save(checkpoint)
    parsed = build_parser().parse_args(
        [
            "--data",
            "corpus",
            "--checkpoint",
            "candidate.pt",
            "--report",
            "report.json",
            "--arch",
            "entity_graph",
            "--init-checkpoint",
            str(checkpoint),
        ]
    )

    parsed.topology_residual_adapter = _resolve_effective_topology_residual_adapter(
        parsed
    )
    assert parsed.topology_residual_adapter is True
    identity = TrainConfig.from_namespace(parsed)
    assert identity.topology_residual_adapter is True
    assert identity.full_config_hash() != TrainConfig.from_namespace(
        SimpleNamespace(**{**vars(parsed), "topology_residual_adapter": False})
    ).full_config_hash()

    parsed.topology_residual_adapter = False
    with pytest.raises(SystemExit, match="does not match --init-checkpoint"):
        _resolve_effective_topology_residual_adapter(parsed)


def test_grow_checkpoint_can_explicitly_enable_structured_repairs(tmp_path) -> None:
    checkpoint = tmp_path / "legacy.pt"
    _small_policy(static=False, legal=False).save(checkpoint)
    assert _resolve_effective_structured_action_residuals(
        _args(
            grow_from_checkpoint=str(checkpoint),
            static=True,
            legal=True,
            set_stats=True,
        )
    ) == (True, True, True)


def test_checkpoint_mismatch_and_a1_recipe_bind_enabled_repairs() -> None:
    config = SimpleNamespace(
        hidden_size=640,
        state_layers=6,
        attention_heads=8,
        dropout=0.05,
        action_target_gather=False,
        action_cross_attention_layers=0,
        action_cross_attention_bottleneck=0,
        topology_residual_adapter=False,
        static_action_residual=False,
        legal_action_value_residual=False,
        legal_action_value_set_statistics=False,
    )
    args = SimpleNamespace(
        arch="entity_graph",
        hidden_size=640,
        graph_layers=6,
        attention_heads=8,
        graph_dropout=0.05,
        action_target_gather=True,
        action_cross_attention_layers=1,
        action_cross_attention_bottleneck=80,
        topology_residual_adapter=True,
        static_action_residual=True,
        legal_action_value_residual=True,
        legal_action_value_set_statistics=True,
    )
    mismatches = _checkpoint_config_mismatches(
        policy_type="entity_graph", config=config, args=args
    )
    assert any(item.startswith("action_target_gather ") for item in mismatches)
    assert any(
        item.startswith("action_cross_attention_layers ")
        for item in mismatches
    )
    assert any(
        item.startswith("action_cross_attention_bottleneck ")
        for item in mismatches
    )
    assert any(item.startswith("topology_residual_adapter ") for item in mismatches)
    assert any(item.startswith("static_action_residual ") for item in mismatches)
    assert any(item.startswith("legal_action_value_residual ") for item in mismatches)
    assert any(
        item.startswith("legal_action_value_set_statistics ")
        for item in mismatches
    )

    recipe_args = build_parser().parse_args(
        [
            "--data",
            "corpus",
            "--checkpoint",
            "candidate.pt",
            "--report",
            "report.json",
            "--arch",
            "entity_graph",
            "--static-action-residual",
            "--legal-action-value-residual",
            "--legal-action-value-set-statistics",
        ]
    )
    effective = _effective_a1_learner_training_recipe(
        recipe_args, {"world_size": 1, "enabled": False}
    )
    assert effective["static_action_residual"] is True
    assert effective["legal_action_value_residual"] is True
    assert effective["legal_action_value_set_statistics"] is True
