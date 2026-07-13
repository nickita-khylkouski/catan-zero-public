from __future__ import annotations

from types import SimpleNamespace

import numpy as np
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


def test_exact_empty_mask_scan_and_nonempty_refusal() -> None:
    data = {"event_mask": np.zeros((9, 64), dtype=np.bool_)}
    report = train_bc._scan_empty_event_mask(data, chunk_rows=4)
    assert report["row_count"] == 9
    assert report["padded_event_width"] == 64
    assert report["nonzero_event_mask_count"] == 0
    assert str(report["scan_sha256"]).startswith("sha256:")

    data["event_mask"][7, 3] = True
    with pytest.raises(SystemExit, match="live event mask"):
        train_bc._scan_empty_event_mask(data, chunk_rows=4)


def test_cropped_entity_batch_skips_event_payload_and_refuses_live_mask(
    monkeypatch,
) -> None:
    class RefusingEventTokens:
        shape = (2, 64, 41)
        dtype = np.dtype(np.float16)

        def __getitem__(self, _index):
            raise AssertionError("event token payload must not be read")

    data = {
        key: np.zeros((2, 1), dtype=np.float32)
        for key in train_bc.ENTITY_BATCH_KEYS
    }
    data["event_tokens"] = RefusingEventTokens()
    data["event_target_ids"] = np.full((2, 64, 4), -1, dtype=np.int16)
    data["event_mask"] = np.zeros((2, 64), dtype=np.bool_)
    monkeypatch.setattr(train_bc, "_CROP_AUTHENTICATED_EMPTY_EVENT_HISTORY", True)
    monkeypatch.setattr(train_bc, "_MASK_HIDDEN_INFO_PLAYER_TOKENS", False)
    batch = train_bc._entity_batch(data, np.asarray([0, 1], dtype=np.int64))
    assert batch["event_tokens"].shape == (2, 0, 41)
    assert batch["event_target_ids"].shape == (2, 0, 4)
    assert batch["event_mask"].shape == (2, 0)

    data["event_mask"][1, 5] = True
    with pytest.raises(RuntimeError, match="live event token"):
        train_bc._entity_batch(data, np.asarray([0, 1], dtype=np.int64))


def test_prefetch_cropped_event_history_never_reads_padded_payload() -> None:
    class RefusingEventPayload:
        def __init__(self, rows: int, width: int, features: int, dtype) -> None:
            self.shape = (rows, width, features)
            self.dtype = np.dtype(dtype)

        def __getitem__(self, _index):
            raise AssertionError("padded empty-event payload must not be read")

    class RefusingObservation:
        def __init__(self, rows: int, features: int) -> None:
            self.shape = (rows, features)
            self.dtype = np.dtype(np.float16)

        def __getitem__(self, _index):
            raise AssertionError("unused dense observation must not be read")

    rows = 7
    corpus = object.__new__(train_bc.MemmapCorpus)
    corpus.row_count = rows
    corpus._eager = {
        "event_mask": np.zeros((rows, 64), dtype=np.bool_),
        "marker": np.arange(rows, dtype=np.int64),
    }
    corpus._lazy = {
        "obs": RefusingObservation(rows, 806),
        "event_tokens": RefusingEventPayload(rows, 64, 41, np.float16),
        "event_target_ids": RefusingEventPayload(rows, 64, 4, np.int16),
    }
    order = np.asarray([6, 2, 4, 0, 5, 1, 3], dtype=np.int64)
    train_indices = np.arange(rows, dtype=np.int64)
    weights = np.ones(rows, dtype=np.float32)

    batches = list(
        train_bc._iterate_training_batches(
            corpus,
            order,
            train_indices,
            3,
            weights,
            weights,
            num_workers=2,
            prefetch=2,
            crop_authenticated_empty_event_history=True,
            omit_unused_observation=True,
        )
    )

    assert len(batches) == 3
    recovered_markers = []
    for materialized, local, _, _ in batches:
        assert materialized["event_tokens"].shape == (len(local), 0, 41)
        assert materialized["event_tokens"].dtype == np.float16
        assert materialized["event_target_ids"].shape == (len(local), 0, 4)
        assert materialized["event_target_ids"].dtype == np.int16
        assert materialized["event_mask"].shape == (len(local), 64)
        assert not bool(np.any(materialized["event_mask"]))
        assert "obs" not in materialized
        assert materialized["_source_obs_inner_shape"] == (806,)
        recovered_markers.extend(materialized["marker"][local].tolist())
    assert recovered_markers == order.tolist()


def test_batch_profile_preserves_omitted_observation_shape() -> None:
    data = {
        "_source_obs_inner_shape": (806,),
        "legal_action_ids": np.asarray([[1, 2, -1], [3, -1, -1]], dtype=np.int16),
        "legal_action_context": np.zeros((2, 3, 5), dtype=np.float16),
    }
    policy = SimpleNamespace(
        policy_type="entity_graph",
        action_size=100,
        context_action_feature_size=5,
    )

    profile = train_bc._batch_profile(policy, data, np.arange(2, dtype=np.int64))

    assert profile["obs_shape"] == [2, 806]
    assert profile["legal_action_ids_shape"] == [2, 3]


@pytest.mark.parametrize(
    ("augment", "relabel_events", "expected"),
    [
        (False, False, False),
        (False, True, False),
        (True, False, False),
        (True, True, True),
    ],
)
def test_effective_symmetry_event_report_collapses_inert_flag(
    augment: bool, relabel_events: bool, expected: bool
) -> None:
    args = SimpleNamespace(
        symmetry_augment=augment,
        symmetry_augment_events=relabel_events,
    )
    assert train_bc._effective_symmetry_augment_events(args) is expected
