"""Tests for tools/seed_fleet_planner.py (task #77).

Root cause being fixed: multi-host fleet launches (H2H arms, gen-1
generation) assign each worker a base-seed via independent per-host
arithmetic (e.g. `9100001 + i*100000` for host A, `9200001 + i*100000` for
host B). Two independently-derived formulas can silently produce
overlapping ranges -- exactly what happened to the v3b_base confirmation
H2H arm (A100A gpu4-7 and A100B gpu0-3 both got seeds 314000-317015) and,
independently, to the *staged but never-fired* gen-1 generation base-seed
scheme in the #76 refresh doc (A100A's 7100001+i*100000 collides with
A100B's 9200001+i*100000 at i=1..7).

The fix: (1) a pure assertion function any fleet launcher can call before
firing, given each worker's (worker_id, base_seed, games) triple, that
raises loudly on any pairwise overlap; (2) a single-global-counter seed
plan generator that is structurally incapable of the copy-paste
per-host-formula bug, because it assigns blocks from ONE running index
rather than N independently-authored formulas.
"""
from __future__ import annotations

import pytest

from tools.seed_fleet_planner import assert_disjoint_seed_blocks, plan_disjoint_seed_blocks


class TestAssertDisjointSeedBlocks:
    def test_disjoint_blocks_pass_silently(self):
        workers = [
            ("b200_gpu0", 9_000_001, 50_000),
            ("a100a_gpu0", 9_200_001, 50_000),
            ("a100b_gpu0", 9_400_001, 50_000),
        ]
        assert_disjoint_seed_blocks(workers)  # must not raise

    def test_adjacent_touching_blocks_are_disjoint(self):
        # [base, base+games) is a half-open interval; a block ending exactly
        # where the next begins must NOT be flagged as overlapping.
        workers = [
            ("w0", 1000, 500),  # covers [1000, 1500)
            ("w1", 1500, 500),  # covers [1500, 2000)
        ]
        assert_disjoint_seed_blocks(workers)  # must not raise

    def test_overlapping_blocks_raise_value_error(self):
        workers = [
            ("a100a_gpu4", 314_000, 16),
            ("a100b_gpu0", 314_000, 16),  # exact collision, the real bug
        ]
        with pytest.raises(ValueError, match="a100a_gpu4.*a100b_gpu0|a100b_gpu0.*a100a_gpu4"):
            assert_disjoint_seed_blocks(workers)

    def test_partial_overlap_raises(self):
        workers = [
            ("w0", 1000, 600),  # covers [1000, 1600)
            ("w1", 1500, 600),  # covers [1500, 2100) -- overlaps [1500,1600)
        ]
        with pytest.raises(ValueError):
            assert_disjoint_seed_blocks(workers)

    def test_staged_gen1_scheme_is_caught(self):
        """Regression test for the actual bug found in the #76 staging doc:
        A100A base_seed = 9_100_001 + i*100_000 (i=0..7),
        A100B base_seed = 9_200_001 + i*100_000 (i=0..7).
        These collide at i=1..7 on host A vs i=0..6 on host B."""
        workers = [(f"a100a_gpu{i}", 9_100_001 + i * 100_000, 100_000) for i in range(8)]
        workers += [(f"a100b_gpu{i}", 9_200_001 + i * 100_000, 100_000) for i in range(8)]
        with pytest.raises(ValueError):
            assert_disjoint_seed_blocks(workers)

    def test_empty_and_single_worker_are_trivially_disjoint(self):
        assert_disjoint_seed_blocks([])
        assert_disjoint_seed_blocks([("only", 1, 10)])

    def test_zero_games_worker_is_ignored(self):
        # a worker requesting 0 games occupies no seed range at all.
        workers = [("w0", 1000, 0), ("w1", 1000, 500)]
        assert_disjoint_seed_blocks(workers)


class TestPlanDisjointSeedBlocks:
    def test_single_global_counter_cannot_collide_by_construction(self):
        worker_ids = [f"b200_gpu{i}" for i in range(1)]
        worker_ids += [f"a100a_gpu{i}" for i in range(8)]
        worker_ids += [f"a100b_gpu{i}" for i in range(8)]
        plan = plan_disjoint_seed_blocks(
            worker_ids, games_per_worker=100_000, base=9_000_001, block_size=200_000
        )
        assert len(plan) == 17
        # Must pass its own assertion (self-consistency).
        assert_disjoint_seed_blocks([(wid, seed, 100_000) for wid, seed in plan.items()])

    def test_plan_preserves_worker_order_and_id_mapping(self):
        plan = plan_disjoint_seed_blocks(["x", "y", "z"], games_per_worker=10, base=100, block_size=50)
        assert list(plan.keys()) == ["x", "y", "z"]
        assert plan["x"] == 100
        assert plan["y"] == 150
        assert plan["z"] == 200

    def test_block_size_must_cover_games_per_worker(self):
        with pytest.raises(ValueError, match="block_size"):
            plan_disjoint_seed_blocks(["x", "y"], games_per_worker=100, base=0, block_size=50)

    def test_duplicate_worker_ids_rejected(self):
        with pytest.raises(ValueError, match="duplicate"):
            plan_disjoint_seed_blocks(["x", "x"], games_per_worker=10, base=0, block_size=50)
