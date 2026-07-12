from __future__ import annotations

from types import SimpleNamespace

from tools.train_bc import _effective_entity_graph_architecture_report


def test_report_uses_upgraded_checkpoint_config_not_cli_default() -> None:
    policy = SimpleNamespace(
        policy_type="entity_graph",
        config=SimpleNamespace(
            action_target_gather=True,
            action_cross_attention_layers=1,
            edge_policy_head=True,
            aux_subgoal_heads=False,
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
    )

    assert report["action_target_gather"] is True
    assert report["action_cross_attention_layers"] == 1
    assert report["edge_policy_head"] is True
    assert report["requested_edge_policy_head"] is False


def test_non_entity_report_preserves_requested_cli_values() -> None:
    policy = SimpleNamespace(policy_type="xdim", config=SimpleNamespace())

    report = _effective_entity_graph_architecture_report(
        policy,
        requested_edge_policy_head=True,
        requested_aux_subgoal_heads=True,
    )

    assert report["action_target_gather"] is False
    assert report["action_cross_attention_layers"] == 0
    assert report["edge_policy_head"] is True
    assert report["aux_subgoal_heads"] is True
