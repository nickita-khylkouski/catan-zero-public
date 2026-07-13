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


def _corpus(
    tmp_path: Path,
    *,
    invalid_target: bool = False,
    event_targets: bool = True,
    teacher_name: str = "gumbel_self_play",
    name: str = "corpus",
) -> Path:
    root = tmp_path / name
    root.mkdir()
    counts = np.asarray([2, 1, 2, 1], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    (root / "row_offsets.dat").write_bytes(offsets.tobytes())
    columns = {}
    _ragged(root, columns, "legal_action_ids", np.arange(6, dtype=np.int16))
    _fixed(
        root,
        columns,
        "action_taken",
        np.asarray([0, 2, 4, 5], dtype=np.int16),
    )
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
    _ragged(
        root,
        columns,
        "target_policy",
        np.asarray([0.6, 0.4, 1.0, 0.8, 0.2, 1.0], dtype=np.float32),
    )
    _ragged(
        root,
        columns,
        "target_policy_mask",
        np.ones(6, dtype=np.bool_),
    )
    (root / "teacher_name.codes.dat").write_bytes(
        np.zeros(4, dtype=np.int32).tobytes()
    )
    columns["teacher_name"] = {
        "kind": "string",
        "categories": [teacher_name],
    }
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
    assert legal["chosen_actions"] == 4
    assert legal["chosen_actions_with_any_target"] == 2
    assert legal["chosen_action_target_coverage"] == 0.5
    assert legal["chosen_policy_active_target_coverage"] == 2 / 3
    assert legal["chosen_search_active_target_coverage"] == 0.5
    assert legal["chosen_action_missing_from_legal"] == 0
    assert legal["chosen_action_duplicate_in_legal"] == 0
    assert legal["chosen_by_action_kind"]["BUILD_SETTLEMENT"]["target_coverage"] == 1.0
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


def test_legacy_policy_weight_authenticates_missing_full_search_column(tmp_path):
    root = _corpus(tmp_path)
    meta_path = root / "corpus_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    del meta["columns"]["used_full_search"]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    (root / "used_full_search.dat").unlink()

    result = audit.audit_corpus(root, chunk_rows=2)

    legal = result["legal_action_targets"]
    assert legal["search_active_rows"] == legal["policy_active_rows"] == 3
    assert legal["search_active_rows_with_any_target"] == 2
    assert legal["search_activity_contract"] == {
        "source": "policy_weight_multiplier_legacy_equivalence",
        "legacy_required_columns": sorted(audit.LEGACY_SEARCH_AUTH_COLUMNS),
        "legacy_wrong_teacher_rows": 0,
        "legacy_missing_stored_policy_rows": 0,
        "authenticated": True,
    }
    assert result["viability"]["action_target_gather"] is True


def test_legacy_search_inference_rejects_non_gumbel_policy_rows(tmp_path):
    root = _corpus(tmp_path, teacher_name="unknown_teacher")
    meta_path = root / "corpus_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    del meta["columns"]["used_full_search"]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    (root / "used_full_search.dat").unlink()

    result = audit.audit_corpus(root, chunk_rows=2)

    contract = result["legal_action_targets"]["search_activity_contract"]
    assert contract["legacy_wrong_teacher_rows"] == 3
    assert contract["authenticated"] is False
    assert result["viability"]["action_target_gather"] is False


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


def test_parallel_corpus_audit_preserves_input_order_and_serial_payload(tmp_path):
    first = _corpus(tmp_path, name="first")
    second = _corpus(tmp_path, event_targets=False, name="second")

    serial = audit.audit_corpora([first, second], chunk_rows=2, workers=1)
    parallel = audit.audit_corpora([first, second], chunk_rows=2, workers=2)

    assert parallel == serial
    assert [row["corpus_dir"] for row in parallel] == [
        str(first.resolve()),
        str(second.resolve()),
    ]
