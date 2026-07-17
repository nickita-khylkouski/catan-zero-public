from __future__ import annotations

from types import SimpleNamespace

from catan_zero.rl.meaningful_history import (
    MEANINGFUL_PUBLIC_HISTORY_SCHEMA_VERSION,
)
from catan_zero.rl.entity_token_features import (
    PUBLIC_RULE_STATE_FEATURE_SCHEMA_VERSION,
)
from tools.train_bc import (
    _checkpoint_config_mismatches,
    _effective_entity_graph_architecture_report,
)


def test_report_uses_upgraded_checkpoint_config_not_cli_default() -> None:
    policy = SimpleNamespace(
        policy_type="entity_graph",
        config=SimpleNamespace(
            action_target_gather=True,
            action_cross_attention_layers=1,
            action_cross_attention_bottleneck=80,
            edge_policy_head=True,
            aux_subgoal_heads=False,
            aux_settlement_pointer_head=True,
            legal_action_value_residual=True,
            legal_action_value_set_statistics=True,
            v6_compatibility_preserving_inputs=True,
            value_tower_split_layers=2,
            state_trunk="transformer",
            relational_block_pattern="",
            relational_action_cross_layers=1,
            relational_edge_policy_head=True,
        ),
    )

    report = _effective_entity_graph_architecture_report(
        policy,
        requested_edge_policy_head=False,
        requested_aux_subgoal_heads=False,
        requested_aux_settlement_pointer_head=False,
    )

    assert report["action_target_gather"] is True
    assert report["action_cross_attention_layers"] == 1
    assert report["action_cross_attention_bottleneck"] == 80
    assert report["relational_action_cross_layers"] == 0
    assert report["configured_relational_action_cross_layers"] == 1
    assert report["edge_policy_head"] is True
    assert report["aux_settlement_pointer_head"] is True
    assert report["legal_action_value_residual"] is True
    assert report["legal_action_value_set_statistics"] is True
    assert report["v6_compatibility_preserving_inputs"] is True
    assert report["value_tower_split_layers"] == 2
    assert report["requested_edge_policy_head"] is False
    assert report["requested_aux_settlement_pointer_head"] is False


def test_non_entity_report_preserves_requested_cli_values() -> None:
    policy = SimpleNamespace(policy_type="xdim", config=SimpleNamespace())

    report = _effective_entity_graph_architecture_report(
        policy,
        requested_edge_policy_head=True,
        requested_aux_subgoal_heads=True,
        requested_aux_settlement_pointer_head=True,
    )

    assert report["action_target_gather"] is False
    assert report["action_cross_attention_layers"] == 0
    assert report["v6_compatibility_preserving_inputs"] is False
    assert report["edge_policy_head"] is True
    assert report["aux_subgoal_heads"] is True
    assert report["aux_settlement_pointer_head"] is True
    assert (
        report["meaningful_public_history_schema"]
        == MEANINGFUL_PUBLIC_HISTORY_SCHEMA_VERSION
    )


def test_report_distinguishes_relational_decoder_from_transformer_knob() -> None:
    policy = SimpleNamespace(
        policy_type="entity_graph",
        config=SimpleNamespace(
            state_trunk="rrt",
            action_cross_attention_layers=0,
            relational_action_cross_layers=2,
        ),
    )

    report = _effective_entity_graph_architecture_report(policy)

    assert report["action_cross_attention_layers"] == 0
    assert report["relational_action_cross_layers"] == 2
    assert report["configured_relational_action_cross_layers"] == 2


def test_report_binds_effective_meaningful_history_contract() -> None:
    policy = SimpleNamespace(
        policy_type="entity_graph",
        config=SimpleNamespace(
            public_card_count_features=True,
            public_card_count_residual_bias=False,
            public_rule_state_features=True,
            public_rule_state_feature_schema=(
                PUBLIC_RULE_STATE_FEATURE_SCHEMA_VERSION
            ),
            meaningful_public_history=True,
            meaningful_public_history_schema=(MEANINGFUL_PUBLIC_HISTORY_SCHEMA_VERSION),
            event_history_limit=32,
            meaningful_public_history_pooling="masked_mean_v1",
            meaningful_public_history_target_gather=True,
        ),
    )

    report = _effective_entity_graph_architecture_report(policy)

    assert report["public_card_count_features"] is True
    assert report["public_card_count_residual_bias"] is False
    assert report["public_rule_state_features"] is True
    assert (
        report["public_rule_state_feature_schema"]
        == PUBLIC_RULE_STATE_FEATURE_SCHEMA_VERSION
    )
    assert report["meaningful_public_history"] is True
    assert (
        report["meaningful_public_history_schema"]
        == MEANINGFUL_PUBLIC_HISTORY_SCHEMA_VERSION
    )
    assert report["event_history_limit"] == 32
    assert report["meaningful_public_history_pooling"] == "masked_mean_v1"
    assert report["meaningful_public_history_target_gather"] is True


def test_requested_settlement_pointer_rejects_legacy_warm_start() -> None:
    config = SimpleNamespace(
        hidden_size=640,
        state_layers=6,
        attention_heads=8,
        dropout=0.05,
        aux_subgoal_heads=True,
        aux_settlement_pointer_head=False,
    )
    args = SimpleNamespace(
        arch="entity_graph",
        hidden_size=640,
        graph_layers=6,
        attention_heads=8,
        graph_dropout=0.05,
        aux_settlement_pointer_head=True,
    )
    assert _checkpoint_config_mismatches(
        policy_type="entity_graph", config=config, args=args
    ) == [
        "aux_settlement_pointer_head checkpoint=False cli=True; upgrade the "
        "checkpoint with --flags aux_settlement_pointer"
    ]
