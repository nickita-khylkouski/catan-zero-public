from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tools import audit_memmap_architecture_targets as audit


def _fixed(root: Path, columns: dict, name: str, values: np.ndarray) -> None:
    array = np.asarray(values)
    (root / f"{name}.dat").write_bytes(array.tobytes())
    columns[name] = {
        "kind": "fixed",
        "dtype": array.dtype.str,
        "inner_shape": list(array.shape[1:]),
    }


def _ragged(root: Path, columns: dict, name: str, values: np.ndarray) -> None:
    array = np.asarray(values)
    (root / f"{name}.dat").write_bytes(array.tobytes())
    if array.ndim == 1:
        columns[name] = {"kind": "ragged2d", "dtype": array.dtype.str, "fill": -1}
    else:
        columns[name] = {
            "kind": "ragged3d",
            "dtype": array.dtype.str,
            "feat": array.shape[1],
            "fill": -1,
        }


def _corpus(tmp_path: Path, *, invalid_target: bool = False, event_targets: bool = True) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    counts = np.asarray([2, 1, 2, 1], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    (root / "row_offsets.dat").write_bytes(offsets.tobytes())
    columns = {}
    _ragged(root, columns, "legal_action_ids", np.arange(6, dtype=np.int16))
    tokens = np.zeros((6, 50), dtype=np.float16)
    for row, action_index in enumerate((0, 1, 16, 10, 2, 17)):
        tokens[row, 2 + action_index] = 1.0
    _ragged(root, columns, "legal_action_tokens", tokens)
    targets = np.full((6, 4), -1, dtype=np.int16)
    targets[0, 1] = 3
    targets[1, 2] = 4
    targets[3, 0] = 8
    targets[3, 3] = 1
    targets[4, 1] = 53
    if invalid_target:
        targets[4, 1] = 54
    _ragged(root, columns, "legal_action_target_ids", targets)
    phase_categories = ["main", "roll", "robber"]
    phase_codes = np.asarray([0, 1, 2, 0], dtype=np.int32)
    (root / "phase.codes.dat").write_bytes(phase_codes.tobytes())
    columns["phase"] = {"kind": "string", "categories": phase_categories}
    _fixed(
        root,
        columns,
        "policy_weight_multiplier",
        np.asarray([1.0, 0.0, 1.0, 1.0], dtype=np.float32),
    )
    _fixed(
        root,
        columns,
        "used_full_search",
        np.asarray([True, True, False, True], dtype=np.bool_),
    )
    event_ids = np.full((4, 2, 4), -1, dtype=np.int16)
    if event_targets:
        event_ids[0, 0, 1] = 4
    _fixed(root, columns, "event_target_ids", event_ids)
    event_mask = np.asarray(
        [[True, False], [False, False], [True, False], [False, False]], dtype=np.bool_
    )
    _fixed(root, columns, "event_mask", event_mask)

    hex_vertex = np.tile(np.arange(6, dtype=np.int16), (4, 19, 1))
    hex_edge = np.tile(np.arange(6, dtype=np.int16), (4, 19, 1))
    edge_vertex = np.tile(np.asarray([0, 1], dtype=np.int16), (4, 72, 1))
    _fixed(root, columns, "hex_vertex_ids", hex_vertex)
    _fixed(root, columns, "hex_edge_ids", hex_edge)
    _fixed(root, columns, "edge_vertex_ids", edge_vertex)
    meta = {
        "schema": "memmap_corpus_v1",
        "row_count": 4,
        "legal_width": 2,
        "flat_count": 6,
        "columns": columns,
    }
    (root / "corpus_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return root


def test_chunked_audit_quantifies_action_phase_search_event_and_graph_targets(tmp_path):
    result = audit.audit_corpus(_corpus(tmp_path), chunk_rows=2)
    legal = result["legal_action_targets"]
    assert result["whole_column_materialization"] is False
    assert legal["actions"] == 6
    assert legal["actions_with_any_target"] == 4
    assert legal["rows_with_any_target"] == 2
    assert legal["policy_active_rows"] == 3
    assert legal["search_active_rows"] == 2
    assert legal["search_active_rows_with_any_target"] == 1
    assert legal["by_action_kind"]["BUILD_SETTLEMENT"]["vertex_targets"] == 1
    assert legal["by_action_kind"]["BUILD_ROAD"]["edge_targets"] == 1
    assert legal["by_action_kind"]["MOVE_ROBBER"]["hex_targets"] == 1
    assert legal["by_phase"]["robber"]["actions_with_any_target"] == 2
    assert legal["by_search_cohort"]["search_active"]["actions"] == 3
    assert result["event_targets"]["masked_events"] == 2
    assert result["event_targets"]["events_with_any_target"] == 1
    assert result["graph_incidence"]["out_of_range_ids"] == 0
    assert result["graph_incidence"]["columns"]["edge_vertex_ids"]["shape_valid"] is True
    assert result["viability"] == {
        "action_target_gather": True,
        "action_cross_attention": True,
        "graph_relational_trunk": True,
        "event_target_relations": True,
        "requires_generator_changes_for_action_probe": False,
        "event_target_generator_change_required_for_event_relations": False,
    }


def test_invalid_target_blocks_gather_and_combined_action_probe(tmp_path):
    result = audit.audit_corpus(_corpus(tmp_path, invalid_target=True), chunk_rows=1)
    assert result["legal_action_targets"]["out_of_range_target_rows"] == 1
    assert result["viability"]["action_target_gather"] is False
    verdict = audit.combined_verdict([result])
    assert verdict["architecture_action_probe_runnable"] is False
    assert verdict["requires_generator_changes_for_action_probe"] is True


def test_empty_event_targets_do_not_block_action_architecture(tmp_path):
    result = audit.audit_corpus(_corpus(tmp_path, event_targets=False), chunk_rows=3)
    assert result["viability"]["action_target_gather"] is True
    assert result["viability"]["action_cross_attention"] is True
    assert result["viability"]["event_target_relations"] is False
    assert result["viability"]["requires_generator_changes_for_action_probe"] is False
    assert (
        result["viability"]["event_target_generator_change_required_for_event_relations"]
        is True
    )


def test_reader_requests_only_bounded_chunks(tmp_path, monkeypatch):
    root = _corpus(tmp_path)
    maximum = 0
    original = audit.CorpusReader.rows_slice

    def track(self, name, start, stop):
        nonlocal maximum
        maximum = max(maximum, stop - start)
        return original(self, name, start, stop)

    monkeypatch.setattr(audit.CorpusReader, "rows_slice", track)
    audit.audit_corpus(root, chunk_rows=2)
    assert maximum == 2
