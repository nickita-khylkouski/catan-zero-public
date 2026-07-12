from __future__ import annotations

import pytest

from catan_zero.rl.entity_token_policy import EntityGraphConfig
from tools.audit_entity_graph_information_surface import (
    InformationSurfaceError,
    audit_memmap_metadata,
    build_report,
)


def _base(**changes):
    values = {
        "action_size": 607,
        "static_action_feature_size": 50,
        "hidden_size": 64,
        "state_layers": 1,
        "attention_heads": 4,
        "dropout": 0.0,
    }
    values.update(changes)
    return EntityGraphConfig(**values)


def test_incumbent_shape_exposes_spatial_aliasing_and_no_history() -> None:
    report = build_report(
        _base(),
        {"implicit_zero_columns": ["event_mask", "event_tokens"]},
    )
    assert report["critical_information_bottlenecks"] == [
        "spatial_state_action_aliasing",
        "public_history_absent",
    ]
    assert report["architecture"]["topology_consumed"] is False
    assert report["architecture"]["action_target_bound"] is False
    assert report["corpus"]["event_history_trainable"] is False
    assert report["safe_for_scale_only_ablation"] is False


def test_target_gather_removes_action_alias_but_not_state_topology_alias() -> None:
    report = build_report(_base(action_target_gather=True))
    assert report["architecture"]["action_target_bound"] is True
    assert report["architecture"]["topology_consumed"] is False
    assert report["critical_information_bottlenecks"] == []


@pytest.mark.parametrize("trunk", ["rrt", "resrgcn"])
def test_relational_trunks_consume_topology_and_bind_actions(trunk: str) -> None:
    report = build_report(_base(state_trunk=trunk))
    assert report["architecture"]["topology_consumed"] is True
    assert report["architecture"]["action_target_bound"] is True
    assert report["critical_information_bottlenecks"] == []


def test_half_declared_implicit_event_columns_fail_closed() -> None:
    with pytest.raises(InformationSurfaceError, match="together"):
        audit_memmap_metadata({"implicit_zero_columns": ["event_tokens"]})


def test_malformed_implicit_columns_fail_closed() -> None:
    with pytest.raises(InformationSurfaceError, match="sequence"):
        audit_memmap_metadata({"implicit_zero_columns": "event_tokens"})
