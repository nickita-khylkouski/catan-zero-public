from __future__ import annotations

import numpy as np

from tools import a1_ddp_epoch_canary as canary


def test_rank_slices_reconstruct_one_shared_weighted_global_draw() -> None:
    weights = np.linspace(0.5, 1.5, canary.SYNTHETIC_ROWS, dtype=np.float64)

    rank_slices = [
        canary._rank_slice(rank, weights) for rank in range(canary.WORLD_SIZE)
    ]
    reconstructed = np.column_stack(rank_slices).reshape(-1)
    expected = canary._expected_global_draw(weights)

    assert all(len(rank_slice) == len(rank_slices[0]) for rank_slice in rank_slices)
    assert len(expected) % canary.GLOBAL_BATCH_SIZE == 0
    assert len(expected) > canary.SYNTHETIC_ROWS
    np.testing.assert_array_equal(reconstructed, expected)


def test_canary_global_draw_is_deterministic_and_nonuniform() -> None:
    weights = np.linspace(0.25, 2.0, canary.SYNTHETIC_ROWS, dtype=np.float64)

    first = canary._expected_global_draw(weights)
    second = canary._expected_global_draw(weights.copy())

    np.testing.assert_array_equal(first, second)
    assert first.dtype == np.int64
    assert first.min() >= 0
    assert first.max() < canary.SYNTHETIC_ROWS
