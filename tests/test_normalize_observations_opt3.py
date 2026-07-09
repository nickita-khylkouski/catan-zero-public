"""OPT-3: normalize_observations mutates the nan_to_num output in place instead
of taking a redundant extra .copy(). Output must be bit-identical to the old
copy-based path, and the caller's array must never be mutated.
"""
from __future__ import annotations

import numpy as np

from catan_zero.rl.xdim_lite_policy import normalize_observations


def _reference(observations: np.ndarray) -> np.ndarray:
    # The pre-OPT-3 implementation, verbatim.
    values = np.nan_to_num(
        np.asarray(observations, dtype=np.float32),
        nan=0.0, posinf=25.0, neginf=-25.0,
    )
    normalized = values.copy()
    large = np.abs(normalized) > 1.0
    normalized[large] = np.clip(normalized[large] / 25.0, -1.0, 1.0)
    return normalized


def test_bit_identical_to_reference_across_regimes():
    rng = np.random.default_rng(0)
    for dtype in (np.float32, np.float64):
        base = rng.standard_normal((257, 806)).astype(dtype) * 30.0
        # sprinkle NaN / +inf / -inf / small / large values
        base.flat[::991] = np.nan
        base.flat[7::991] = np.inf
        base.flat[13::991] = -np.inf
        base.flat[3::50] = 0.3  # small (< 1.0) stays untouched
        got = normalize_observations(base.copy())
        ref = _reference(base.copy())
        assert got.dtype == np.float32
        assert np.array_equal(got, ref)  # bit-identical


def test_does_not_mutate_caller_array():
    x = np.array([[2.0, -3.0, 0.5]], dtype=np.float32)
    snapshot = x.copy()
    normalize_observations(x)
    assert np.array_equal(x, snapshot)
