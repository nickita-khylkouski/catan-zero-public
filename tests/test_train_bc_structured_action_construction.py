from __future__ import annotations

from types import SimpleNamespace

import pytest

from catan_zero.rl.entity_token_policy import EntityGraphPolicy
from catan_zero.rl.pipeline_configs import TrainConfig
from catan_zero.rl.self_play import make_env_config
from tools.train_bc import (
    _checkpoint_config_mismatches,
    _effective_a1_learner_training_recipe,
    _resolve_effective_structured_action_residuals,
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
    *, static: bool, legal: bool, set_stats: bool = False
) -> EntityGraphPolicy:
    return EntityGraphPolicy.create(
        env_config=make_env_config(vps_to_win=3),
        hidden_size=16,
        state_layers=1,
        attention_heads=2,
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
            "--static-action-residual",
            "--legal-action-value-residual",
            "--legal-action-value-set-statistics",
        ]
    )
    assert parser.get_default("static_action_residual") is None
    assert parser.get_default("legal_action_value_residual") is None
    assert parser.get_default("legal_action_value_set_statistics") is None
    (
        parsed.static_action_residual,
        parsed.legal_action_value_residual,
        parsed.legal_action_value_set_statistics,
    ) = _resolve_effective_structured_action_residuals(parsed)

    kwargs = _structured_action_create_kwargs(parsed)
    assert kwargs == {
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
        **kwargs,
    )
    assert policy.config.static_action_residual is True
    assert policy.config.legal_action_value_residual is True
    assert policy.config.legal_action_value_set_statistics is True
    assert hasattr(policy.model, "static_action_residual_proj")
    assert hasattr(policy.model, "legal_action_value_residual_proj")
    assert hasattr(policy.model, "legal_action_value_count_proj")

    identity = TrainConfig.from_namespace(parsed)
    assert identity.static_action_residual is True
    assert identity.legal_action_value_residual is True
    assert identity.legal_action_value_set_statistics is True


def test_fresh_default_is_legacy_off_and_non_entity_rejects_enablement() -> None:
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
    _small_policy(static=True, legal=True, set_stats=True).save(checkpoint)

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
        static_action_residual=True,
        legal_action_value_residual=True,
        legal_action_value_set_statistics=True,
    )
    mismatches = _checkpoint_config_mismatches(
        policy_type="entity_graph", config=config, args=args
    )
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
