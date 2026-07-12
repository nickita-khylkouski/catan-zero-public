from __future__ import annotations

from types import SimpleNamespace

import pytest

from tools import train_bc


def _meta(digit: str, *, implicit_zero: bool = True) -> dict:
    return {
        "row_count": 3,
        "payload_inventory_sha256": "sha256:" + digit * 64,
        "implicit_zero_columns": (
            ["event_mask", "event_tokens"] if implicit_zero else []
        ),
    }


def _args(*, arch: str, acknowledgements=()):
    return SimpleNamespace(
        arch=arch,
        acknowledge_empty_event_history_payload_inventory_sha256=list(acknowledgements),
    )


def test_generic_non_a1_training_remains_backward_compatible() -> None:
    assert (
        train_bc._a1_training_event_history_contract(
            _args(arch="entity_graph"),
            None,
            SimpleNamespace(use_graph_history_features=True),
        )
        is None
    )


def test_entity_graph_a1_refuses_phantom_history_without_exact_ack() -> None:
    metadata = _meta("1")
    with pytest.raises(SystemExit, match="missing="):
        train_bc._a1_training_event_history_contract(
            _args(arch="entity_graph"),
            metadata,
            SimpleNamespace(use_graph_history_features=False),
        )

    contract = train_bc._a1_training_event_history_contract(
        _args(
            arch="entity_graph",
            acknowledgements=[metadata["payload_inventory_sha256"]],
        ),
        metadata,
        SimpleNamespace(use_graph_history_features=False),
    )
    assert contract["graph_history_observation_schema"] is False
    assert contract["event_history_consumer_enabled"] is True
    assert contract["training_event_history_trainable"] is False
    assert contract["native_inference"]["available"] is False
    assert contract["event_history_end_to_end_usable"] is False


def test_composite_ack_binds_each_named_component_inventory() -> None:
    first = _meta("1")
    second = _meta("2")
    composite = {
        "schema_version": "memmap_composite_v2",
        "components": [
            {"component_id": "n128", "corpus_meta": first},
            {"component_id": "n256", "corpus_meta": second},
        ],
    }
    with pytest.raises(SystemExit, match="missing=.*2222"):
        train_bc._a1_training_event_history_contract(
            _args(
                arch="entity_graph",
                acknowledgements=[first["payload_inventory_sha256"]],
            ),
            composite,
            SimpleNamespace(use_graph_history_features=True),
        )
    contract = train_bc._a1_training_event_history_contract(
        _args(
            arch="entity_graph",
            acknowledgements=[
                first["payload_inventory_sha256"],
                second["payload_inventory_sha256"],
            ],
        ),
        composite,
        SimpleNamespace(use_graph_history_features=True),
    )
    assert [row["component_id"] for row in contract["components"]] == [
        "n128",
        "n256",
    ]


def test_non_event_arch_rejects_meaningless_ack() -> None:
    metadata = _meta("1")
    with pytest.raises(SystemExit, match="consumer is enabled"):
        train_bc._a1_training_event_history_contract(
            _args(
                arch="xdim_graph",
                acknowledgements=[metadata["payload_inventory_sha256"]],
            ),
            metadata,
            SimpleNamespace(use_graph_history_features=True),
        )
