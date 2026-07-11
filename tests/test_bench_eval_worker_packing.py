from __future__ import annotations

import argparse

import pytest

from tools import bench_eval_worker_packing as bench


def test_worker_grid_parser_is_strict() -> None:
    assert bench._parse_workers("8,16,24") == (8, 16, 24)  # noqa: SLF001
    for invalid in ("", "0", "8,8", "8,nope"):
        with pytest.raises(argparse.ArgumentTypeError):
            bench._parse_workers(invalid)  # noqa: SLF001


def test_cpuset_parser_expands_ranges_and_rejects_non_topology_text() -> None:
    assert bench._parse_cpuset("0-3,8,10-11") == {0, 1, 2, 3, 8, 10, 11}  # noqa: SLF001
    for invalid in ("", "0-x", "3-1", "-1"):
        with pytest.raises(ValueError):
            bench._parse_cpuset(invalid)  # noqa: SLF001


def test_split_indices_covers_each_evaluation_once() -> None:
    shards = bench._split_indices(25, 8)  # noqa: SLF001
    assert sorted(index for shard in shards for index in shard) == list(range(25))
    assert max(map(len, shards)) - min(map(len, shards)) == 1
    with pytest.raises(ValueError, match="total_evals"):
        bench._split_indices(7, 8)  # noqa: SLF001


def test_result_digest_is_order_independent_but_value_sensitive() -> None:
    rows = [(1, 0.5, ((3, 0.25),)), (0, -0.25, ((2, 1.0),))]
    assert bench._result_digest(rows) == bench._result_digest(list(reversed(rows)))  # noqa: SLF001
    changed = [(1, 0.6, ((3, 0.25),)), rows[1]]
    assert bench._result_digest(rows) != bench._result_digest(changed)  # noqa: SLF001
