from __future__ import annotations

from pathlib import Path

import pytest
import numpy as np

from catan_zero.rl.entity_token_policy import EntityGraphConfig
from tools.audit_entity_graph_information_surface import (
    InformationSurfaceError,
    audit_memmap_metadata,
    build_a1_training_event_history_contract,
    build_report,
    enforce_graph_history_contract,
    native_inference_event_history_capability,
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


def test_topology_adapter_and_gather_remove_both_spatial_aliases() -> None:
    report = build_report(
        _base(topology_residual_adapter=True, action_target_gather=True)
    )
    assert report["architecture"]["topology_consumed"] is True
    assert report["architecture"]["topology_residual_adapter"] is True
    assert report["architecture"]["action_target_bound"] is True
    assert not any("topology" in item for item in report["architecture"]["limitations"])


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
    np.full((rows, 2, 4), -1, dtype=np.int16).tofile(tmp_path / "event_target_ids.dat")
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
    with pytest.raises(InformationSurfaceError, match="history is absent"):
        enforce_graph_history_contract(audit, required=True)


def test_graph_history_contract_rejects_unverified_v1_metadata() -> None:
    audit = audit_memmap_metadata({"implicit_zero_columns": []})
    with pytest.raises(InformationSurfaceError, match="history is unverified"):
        enforce_graph_history_contract(audit, required=True)
    enforce_graph_history_contract(audit, required=False)


def test_malformed_payload_scan_cannot_fail_open_as_nonzero_history() -> None:
    metadata = {
        "row_count": 3,
        "payload_inventory_sha256": "sha256:" + "a" * 64,
        "implicit_zero_columns": [],
    }
    with pytest.raises(InformationSurfaceError, match="exact nonnegative"):
        audit_memmap_metadata(
            metadata,
            payload_scan={
                "row_count": 3,
                "payload_inventory_sha256": metadata["payload_inventory_sha256"],
                "columns": {},
            },
        )


def test_payload_scan_must_bind_authenticated_inventory_and_rows() -> None:
    metadata = {
        "row_count": 3,
        "payload_inventory_sha256": "sha256:" + "a" * 64,
        "implicit_zero_columns": [],
    }
    scan = {
        "row_count": 3,
        "payload_inventory_sha256": "sha256:" + "b" * 64,
        "columns": {
            "event_tokens": {"nonzero_count": 1},
            "event_mask": {"nonzero_count": 1},
        },
    }
    with pytest.raises(InformationSurfaceError, match="authenticated payload"):
        audit_memmap_metadata(metadata, payload_scan=scan)
    scan["payload_inventory_sha256"] = metadata["payload_inventory_sha256"]
    scan["row_count"] = 4
    with pytest.raises(InformationSurfaceError, match="row count"):
        audit_memmap_metadata(metadata, payload_scan=scan)


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


def _a1_meta(digit: str, *, implicit_zero: bool) -> dict:
    return {
        "payload_inventory_sha256": "sha256:" + digit * 64,
        "row_count": 3,
        "implicit_zero_columns": (
            ["event_mask", "event_tokens"] if implicit_zero else []
        ),
    }


def test_a1_training_contract_binds_empty_ack_to_exact_payload_inventory() -> None:
    metadata = {
        "n128": _a1_meta("1", implicit_zero=True),
        # A physical v1 payload is unknown without a scan.  Its explicit
        # acknowledgement truthfully authorizes only shape-compatible use.
        "n256": _a1_meta("2", implicit_zero=False),
    }
    report = build_a1_training_event_history_contract(
        metadata,
        graph_history_features=True,
        event_history_consumer_enabled=True,
        empty_payload_inventory_acknowledgements=[
            metadata["n128"]["payload_inventory_sha256"],
            metadata["n256"]["payload_inventory_sha256"],
        ],
    )
    assert report["status"] == "empty_payloads_acknowledged"
    assert report["graph_history_observation_schema"] is True
    assert report["training_event_history_trainable"] is False
    assert report["native_inference"]["available"] is False
    assert report["event_history_end_to_end_usable"] is False
    assert [item["status"] for item in report["components"]] == [
        "empty_acknowledged_machine_proven",
        "empty_acknowledged_legacy_payload",
    ]


@pytest.mark.parametrize(
    "acknowledgements, expected",
    [
        ([], "missing="),
        (["sha256:" + "9" * 64], "missing="),
        (
            ["sha256:" + "1" * 64, "sha256:" + "9" * 64],
            "extra=",
        ),
    ],
)
def test_a1_training_contract_rejects_missing_or_extra_payload_ack(
    acknowledgements, expected
) -> None:
    with pytest.raises(InformationSurfaceError, match=expected):
        build_a1_training_event_history_contract(
            {"n128": _a1_meta("1", implicit_zero=True)},
            graph_history_features=True,
            event_history_consumer_enabled=True,
            empty_payload_inventory_acknowledgements=acknowledgements,
        )


def test_a1_training_contract_rejects_ack_when_consumer_disabled() -> None:
    with pytest.raises(InformationSurfaceError, match="consumer is enabled"):
        build_a1_training_event_history_contract(
            {"n128": _a1_meta("1", implicit_zero=True)},
            graph_history_features=False,
            event_history_consumer_enabled=False,
            empty_payload_inventory_acknowledgements=["sha256:" + "1" * 64],
        )


def test_a1_training_contract_rejects_unbound_inventory_identity() -> None:
    with pytest.raises(InformationSurfaceError, match="payload_inventory_sha256"):
        build_a1_training_event_history_contract(
            {"n128": {"implicit_zero_columns": []}},
            graph_history_features=True,
            event_history_consumer_enabled=True,
        )


def test_a1_training_contract_rejects_nonzero_train_native_empty_skew() -> None:
    metadata = {"fresh": _a1_meta("3", implicit_zero=False)}
    scan = {
        "payload_inventory_sha256": metadata["fresh"]["payload_inventory_sha256"],
        "row_count": 3,
        "columns": {
            "event_tokens": {"nonzero_count": 7},
            "event_mask": {"nonzero_count": 2},
        },
    }
    with pytest.raises(InformationSurfaceError, match="train/deploy"):
        build_a1_training_event_history_contract(
            metadata,
            graph_history_features=True,
            event_history_consumer_enabled=True,
            component_payload_scans={"fresh": scan},
        )


def test_a1_training_contract_rejects_scan_for_different_inventory() -> None:
    metadata = {"fresh": _a1_meta("3", implicit_zero=False)}
    scan = {
        "payload_inventory_sha256": "sha256:" + "4" * 64,
        "row_count": 3,
        "columns": {
            "event_tokens": {"nonzero_count": 7},
            "event_mask": {"nonzero_count": 2},
        },
    }
    with pytest.raises(InformationSurfaceError, match="not bound"):
        build_a1_training_event_history_contract(
            metadata,
            graph_history_features=True,
            event_history_consumer_enabled=True,
            component_payload_scans={"fresh": scan},
        )


def test_entity_event_consumer_requires_ack_even_without_legacy_graph_flag() -> None:
    metadata = {"n128": _a1_meta("1", implicit_zero=True)}
    with pytest.raises(InformationSurfaceError, match="missing="):
        build_a1_training_event_history_contract(
            metadata,
            graph_history_features=False,
            event_history_consumer_enabled=True,
        )
    report = build_a1_training_event_history_contract(
        metadata,
        graph_history_features=False,
        event_history_consumer_enabled=True,
        empty_payload_inventory_acknowledgements=[
            metadata["n128"]["payload_inventory_sha256"]
        ],
    )
    assert report["graph_history_observation_schema"] is False
    assert report["event_history_consumer_enabled"] is True
    assert report["training_event_history_trainable"] is False


def test_native_empty_capability_matches_checked_in_feature_sources() -> None:
    root = Path(__file__).resolve().parents[1]
    rust = (root / "native/catanatron-rs/src/lib.rs").read_text()
    adapter = (root / "src/catan_zero/search/neural_rust_mcts.py").read_text()
    capability = native_inference_event_history_capability()
    assert capability["available"] is False
    assert "let event_mask = vec![false; ENTITY_EVENT_HISTORY_LIMIT]" in rust
    assert '"event_log": []' in adapter
