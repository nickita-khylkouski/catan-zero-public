from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tools import a1_target_eligibility_inventory as inventory
from tools.build_memmap_corpus import build_memmap_corpus
from tools.mixed_memmap_corpus import ConcatMemmapCorpus
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
        target_information_regime=np.asarray(["public_belief_single_tree_v1"] * 3),
        simulations_used=np.asarray([6, 0, 8], dtype=np.int32),
        root_value=np.asarray([0.3, np.nan, 0.45], dtype=np.float32),
        root_value_mask=np.asarray([True, False, True]),
        root_prior_value=np.asarray([0.1, np.nan, 0.2], dtype=np.float32),
        root_prior_value_mask=np.asarray([True, False, True]),
        game_seed=np.asarray([901, 901, 901], dtype=np.int64),
        decision_index=np.asarray([0, 1, 2], dtype=np.int32),
        terminated=np.asarray([False, False, True]),
        truncated=np.asarray([False, False, False]),
        search_evidence_version=np.asarray(2, dtype=np.uint8),
        search_evidence_offsets=np.asarray([0, 2, 3], dtype=np.uint32),
        search_visit_counts_flat=np.asarray([4, 2, 8], dtype=np.uint16),
        search_completed_q_flat=np.asarray([0.4, -0.2, 0.5], dtype=np.float32),
        search_prior_policy_flat=np.asarray([0.7, 0.3, 1.0], dtype=np.float32),
    )


def test_memmap_preserves_compact_search_evidence_for_all_rows(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _write_search_evidence_shard(source / "shard_000.npz")
    corpus_dir = tmp_path / "corpus"

    meta = build_memmap_corpus(source, corpus_dir, progress_every=0)

    assert meta["search_evidence"] == {
        "schema": "gumbel_root_search_evidence_v2_fp32_prior",
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
        [2, 0, 2],
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
    np.testing.assert_allclose(
        np.fromfile(corpus_dir / "search_prior_policy_flat.dat", dtype=np.float32),
        [0.7, 0.3, 1.0],
    )

    corpus = MemmapCorpus(corpus_dir)
    np.testing.assert_array_equal(
        corpus["search_visit_counts_flat"][[0, 1, 2]],
        [[4, 2, 0], [0, 0, 0], [8, 0, 0]],
    )
    completed_q = corpus["search_completed_q_flat"][[0, 1, 2]]
    np.testing.assert_allclose(completed_q[0, :2], [0.4, -0.2])
    assert bool(np.all(np.isnan(completed_q[1])))
    exact_prior = corpus["search_prior_policy_flat"][[0, 1, 2]]
    np.testing.assert_allclose(exact_prior[0, :2], [0.7, 0.3])
    assert bool(np.all(np.isnan(exact_prior[1])))
    np.testing.assert_allclose(
        corpus["root_prior_value"][:],
        [0.1, np.nan, 0.2],
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        corpus["root_prior_value_mask"][:],
        [True, False, True],
    )

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
        "search_prior_policy_flat",
    }.issubset(inspected["search_evidence_columns"])


def test_fresh_search_evidence_composes_with_legacy_replay(tmp_path: Path) -> None:
    fresh_source = tmp_path / "fresh_source"
    fresh_source.mkdir()
    fresh_shard = fresh_source / "shard.npz"
    _write_search_evidence_shard(fresh_shard)
    fresh_dir = tmp_path / "fresh_corpus"
    build_memmap_corpus(fresh_source, fresh_dir, progress_every=0)

    legacy_source = tmp_path / "legacy_source"
    legacy_source.mkdir()
    with np.load(fresh_shard) as source:
        legacy_payload = {
            name: source[name]
            for name in source.files
            if not name.startswith("search_")
        }
    np.savez_compressed(legacy_source / "shard.npz", **legacy_payload)
    legacy_dir = tmp_path / "legacy_corpus"
    build_memmap_corpus(legacy_source, legacy_dir, progress_every=0)

    fresh = MemmapCorpus(fresh_dir)
    legacy = MemmapCorpus(legacy_dir)
    mixed = ConcatMemmapCorpus([fresh, legacy])

    for name in (
        "search_evidence_version",
        "search_evidence_mask",
        "search_evidence_offsets",
        "search_visit_counts_flat",
        "search_completed_q_flat",
        "search_prior_policy_flat",
    ):
        np.testing.assert_allclose(
            mixed[name][:3],
            fresh[name][:],
            equal_nan=True,
        )
    np.testing.assert_array_equal(mixed["search_evidence_version"][3:], [0, 0, 0])
    assert mixed["search_evidence_version"].dtype == np.dtype(np.uint8)
    np.testing.assert_array_equal(mixed["search_evidence_mask"][3:], [False] * 3)
    assert mixed["search_evidence_mask"].dtype == np.dtype(np.bool_)
    np.testing.assert_array_equal(
        mixed["search_evidence_offsets"][3:],
        np.zeros((3, 2), dtype=np.int64),
    )
    assert mixed["search_evidence_offsets"].shape == (6, 2)
    assert mixed["search_evidence_offsets"].dtype == np.dtype(np.int64)
    np.testing.assert_array_equal(
        mixed["search_visit_counts_flat"][3:],
        np.zeros((3, 3), dtype=np.uint16),
    )
    assert mixed["search_visit_counts_flat"].shape == (6, 3)
    assert mixed["search_visit_counts_flat"].dtype == np.dtype(np.uint16)
    assert bool(np.all(np.isnan(mixed["search_completed_q_flat"][3:])))
    assert mixed["search_completed_q_flat"].dtype == np.dtype(np.float32)
    assert bool(np.all(np.isnan(mixed["search_prior_policy_flat"][3:])))
    assert mixed["search_prior_policy_flat"].dtype == np.dtype(np.float32)
    assert set(mixed.synthesized_columns_by_component[1]) >= {
        "search_evidence_version",
        "search_evidence_mask",
        "search_evidence_offsets",
        "search_visit_counts_flat",
        "search_completed_q_flat",
        "search_prior_policy_flat",
    }


def test_memmap_rejects_zero_mass_exact_prior(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    shard = source / "shard.npz"
    _write_search_evidence_shard(shard)
    with np.load(shard) as original:
        payload = {name: original[name] for name in original.files}
    payload["search_prior_policy_flat"] = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    np.savez(shard, **payload)

    with pytest.raises(SystemExit, match="zero active-row mass"):
        build_memmap_corpus(source, tmp_path / "corpus", progress_every=0)


def test_inventory_rejects_zero_mass_exact_prior(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _write_search_evidence_shard(source / "shard.npz")
    corpus_dir = tmp_path / "corpus"
    build_memmap_corpus(source, corpus_dir, progress_every=0)

    prior = np.memmap(
        corpus_dir / "search_prior_policy_flat.dat",
        mode="r+",
        dtype=np.float32,
        shape=(3,),
    )
    prior[:2] = 0.0
    prior.flush()

    with pytest.raises(inventory.InventoryError, match="zero active-row mass"):
        inventory.inspect_memmap(
            label="coherent",
            corpus_dir=corpus_dir,
            required_regime="public_belief_single_tree_v1",
        )
