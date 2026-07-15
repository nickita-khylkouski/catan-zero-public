from __future__ import annotations

import math

import numpy as np
import pytest

from catan_zero.rl.public_dev_belief import (
    DEV_CARD_TYPES,
    build_public_dev_belief,
    enumerate_feasible_hidden_counts,
)


def _pool(**overrides: int) -> dict[str, int]:
    values = {
        "KNIGHT": 12,
        "YEAR_OF_PLENTY": 2,
        "MONOPOLY": 1,
        "ROAD_BUILDING": 2,
        "VICTORY_POINT": 5,
    }
    values.update(overrides)
    return values


def test_feasible_support_obeys_joint_conservation():
    pool = _pool()
    support = enumerate_feasible_hidden_counts(pool, 3)

    assert support.ndim == 2
    assert support.shape[1] == 5
    assert np.all(support.sum(axis=1) == 3)
    assert np.all(support >= 0)
    assert np.all(support <= np.asarray([pool[card] for card in DEV_CARD_TYPES]))
    assert len({tuple(row) for row in support.tolist()}) == support.shape[0]


def test_zero_tilt_is_exact_multivariate_hypergeometric_joint_prior():
    pool = _pool()
    belief = build_public_dev_belief(pool, 3, theta=np.zeros(5))
    denominator = math.comb(sum(pool.values()), 3)
    expected = np.asarray(
        [
            math.prod(
                math.comb(pool[card], int(count))
                for card, count in zip(DEV_CARD_TYPES, row, strict=True)
            )
            / denominator
            for row in belief.count_vectors
        ]
    )

    np.testing.assert_allclose(belief.probabilities, expected, rtol=1e-14, atol=1e-16)
    assert belief.probabilities.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(
        belief.expected_counts,
        3.0 * np.asarray([pool[card] for card in DEV_CARD_TYPES]) / sum(pool.values()),
    )
    expected_vp = 1.0 - math.comb(17, 3) / math.comb(22, 3)
    assert belief.victory_point_probability == pytest.approx(expected_vp)


def test_exponential_tilt_changes_mass_but_never_support():
    pool = _pool()
    prior = build_public_dev_belief(pool, 3)
    tilted = build_public_dev_belief(pool, 3, theta=[0, 0, 0, 0, 2.0])

    assert np.array_equal(tilted.count_vectors, prior.count_vectors)
    assert tilted.probabilities.sum() == pytest.approx(1.0)
    assert tilted.expected_count("VICTORY_POINT") > prior.expected_count(
        "VICTORY_POINT"
    )
    assert tilted.victory_point_probability > prior.victory_point_probability


def test_sampling_is_reproducible_and_always_feasible():
    belief = build_public_dev_belief(_pool(), 3, theta=[0.2, -0.1, 0.3, 0.0, 0.5])
    left = belief.sample(np.random.default_rng(1234), size=1000)
    right = belief.sample(np.random.default_rng(1234), size=1000)

    assert np.array_equal(left, right)
    assert left.shape == (1000, 5)
    assert np.all(left.sum(axis=1) == 3)
    support = {tuple(row) for row in belief.count_vectors.tolist()}
    assert all(tuple(row) in support for row in left.tolist())


@pytest.mark.parametrize(
    ("pool", "count", "theta"),
    [
        (_pool(KNIGHT=-1), 1, None),
        (_pool(KNIGHT=15), 1, None),
        (_pool(), -1, None),
        (_pool(), 23, None),
        (_pool(), 3, [0.0] * 4),
        (_pool(), 3, [0.0, 0.0, np.inf, 0.0, 0.0]),
    ],
)
def test_invalid_public_conservation_or_tilt_fails_closed(pool, count, theta):
    with pytest.raises(ValueError):
        build_public_dev_belief(pool, count, theta=theta)


def test_zero_card_public_state_has_one_certain_joint_state():
    belief = build_public_dev_belief(_pool(), 0)
    assert np.array_equal(belief.count_vectors, np.zeros((1, 5), dtype=np.int16))
    assert np.array_equal(belief.probabilities, np.ones(1))
    assert np.array_equal(belief.expected_counts, np.zeros(5))
    assert belief.victory_point_probability == 0.0
