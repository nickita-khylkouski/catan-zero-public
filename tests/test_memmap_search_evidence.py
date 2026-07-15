from __future__ import annotations

from pathlib import Path

import numpy as np

from tools import a1_target_eligibility_inventory as inventory
from tools.build_memmap_corpus import build_memmap_corpus
from tools.train_bc import MemmapCorpus


def _write_search_evidence_shard(path: Path) -> None:
    np.savez_compressed(
        path,
        obs=np.zeros((3, 4), dtype=np.float16),
        legal_action_ids=np.asarray(
            [[2, 5, -1], [3, -1, -1], [1, -1, -1]], dtype=np.int16
        ),
        legal_action_context=np.zeros((3, 3, 1), dtype=np.float16),
        action_taken=np.asarray([2, 3, 1], dtype=np.int16),
        policy_weight_multiplier=np.asarray([1.0, 0.0, 1.0], dtype=np.float32),
        value_weight_multiplier=np.ones(3, dtype=np.float32),
        used_full_search=np.asarray([True, False, True]),
        is_forced=np.asarray([False, False, False]),
        target_information_regime=np.asarray(
            ["public_belief_single_tree_v1"] * 3
        ),
        simulations_used=np.asarray([6, 0, 8], dtype=np.int32),
        game_seed=np.asarray([901, 901, 901], dtype=np.int64),
        decision_index=np.asarray([0, 1, 2], dtype=np.int32),
        terminated=np.asarray([False, False, True]),
        truncated=np.asarray([False, False, False]),
        search_evidence_version=np.asarray(1, dtype=np.uint8),
        search_evidence_offsets=np.asarray([0, 2, 3], dtype=np.uint32),
        search_visit_counts_flat=np.asarray([4, 2, 8], dtype=np.uint16),
        search_completed_q_flat=np.asarray([0.4, -0.2, 0.5], dtype=np.float32),
    )


def test_memmap_preserves_compact_search_evidence_for_all_rows(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _write_search_evidence_shard(source / "shard_000.npz")
    corpus_dir = tmp_path / "corpus"

    meta = build_memmap_corpus(source, corpus_dir, progress_every=0)

    assert meta["search_evidence"] == {
        "schema": "gumbel_root_search_evidence_v1",
        "row_addressing": "all_rows_empty_inactive_v1",
        "active_row_count": 2,
        "flat_entry_count": 3,
    }
    np.testing.assert_array_equal(
        np.fromfile(corpus_dir / "search_evidence_offsets.dat", dtype=np.int64),
        [0, 2, 2, 3],
    )
    np.testing.assert_array_equal(
        np.fromfile(corpus_dir / "search_evidence_version.dat", dtype=np.uint8),
        [1, 0, 1],
    )
    np.testing.assert_array_equal(
        np.fromfile(corpus_dir / "search_evidence_mask.dat", dtype=np.bool_),
        [True, False, True],
    )
    np.testing.assert_array_equal(
        np.fromfile(corpus_dir / "search_visit_counts_flat.dat", dtype=np.uint16),
        [4, 2, 8],
    )
    np.testing.assert_allclose(
        np.fromfile(corpus_dir / "search_completed_q_flat.dat", dtype=np.float32),
        [0.4, -0.2, 0.5],
    )

    corpus = MemmapCorpus(corpus_dir)
    np.testing.assert_array_equal(
        corpus["search_visit_counts_flat"][[0, 1, 2]],
        [[4, 2, 0], [0, 0, 0], [8, 0, 0]],
    )
    completed_q = corpus["search_completed_q_flat"][[0, 1, 2]]
    np.testing.assert_allclose(completed_q[0, :2], [0.4, -0.2])
    assert bool(np.all(np.isnan(completed_q[1])))

    inspected = inventory.inspect_memmap(
        label="coherent",
        corpus_dir=corpus_dir,
        required_regime="public_belief_single_tree_v1",
    )
    assert inspected["search_evidence"]["policy_active_alignment"] is True
    assert inspected["search_evidence"]["active_rows"] == 2
    assert inspected["search_evidence"]["flat_entries"] == 3
    assert {
        "search_evidence_version",
        "search_evidence_offsets",
        "search_visit_counts_flat",
        "search_completed_q_flat",
    }.issubset(inspected["search_evidence_columns"])
