from __future__ import annotations

import pytest
import numpy as np

from catan_zero.rl.entity_token_policy import EntityGraphConfig
from tools.audit_entity_graph_information_surface import (
    InformationSurfaceError,
    audit_memmap_metadata,
    build_report,
    scan_event_payload,
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


def test_physical_v1_event_columns_are_unverified_without_exact_scan() -> None:
    report = build_report(_base(), {"implicit_zero_columns": []})
    assert report["corpus"]["event_history_trainable"] is None
    assert "public_history_unverified" in report["critical_information_bottlenecks"]


def test_exact_scan_proves_old_physical_event_columns_are_constant(tmp_path) -> None:
    rows = 3
    columns = {
        "event_tokens": {"kind": "dense", "dtype": "<f2", "inner_shape": [2, 3]},
        "event_mask": {"kind": "dense", "dtype": "|b1", "inner_shape": [2]},
        "event_target_ids": {
            "kind": "dense",
            "dtype": "<i2",
            "inner_shape": [2, 4],
        },
    }
    np.zeros((rows, 2, 3), dtype=np.float16).tofile(tmp_path / "event_tokens.dat")
    np.zeros((rows, 2), dtype=np.bool_).tofile(tmp_path / "event_mask.dat")
    np.full((rows, 2, 4), -1, dtype=np.int16).tofile(
        tmp_path / "event_target_ids.dat"
    )
    metadata = {
        "row_count": rows,
        "columns": columns,
        "implicit_zero_columns": [],
        "payload_inventory_sha256": "sha256:" + "a" * 64,
    }
    scan = scan_event_payload(tmp_path, metadata, chunk_rows=2)
    audit = audit_memmap_metadata(metadata, payload_scan=scan)
    assert audit["event_history_trainable"] is False
    assert scan["columns"]["event_tokens"]["nonzero_count"] == 0
    assert scan["columns"]["event_mask"]["nonzero_count"] == 0
    assert scan["columns"]["event_target_ids"]["nonfill_count"] == 0
    assert scan["reclaimable_constant_bytes"] == sum(
        path.stat().st_size for path in tmp_path.glob("event_*.dat")
    )


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
