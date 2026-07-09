from __future__ import annotations

import numpy as np
import pytest

from tools.train_bc import (
    _epoch_order,
    compute_policy_surprise_kl,
    policy_surprise_sampling_weights,
)


def _base_data(**overrides):
    data = {
        "legal_action_ids": np.asarray(
            [
                [5, 6, 7, -1],  # row 0: low surprise (target ~= prior)
                [5, 6, 7, -1],  # row 1: high surprise (target far from prior)
                [5, 6, 7, -1],  # row 2: no prior_policy recorded (teacher row)
            ],
            dtype=np.int16,
        ),
        "target_policy": np.asarray(
            [
                [0.5, 0.3, 0.2, 0.0],
                [0.98, 0.01, 0.01, 0.0],
                [0.6, 0.3, 0.1, 0.0],
            ],
            dtype=np.float32,
        ),
        "prior_policy": np.asarray(
            [
                [0.5, 0.3, 0.2, 0.0],
                [0.34, 0.33, 0.33, 0.0],
                [0.0, 0.0, 0.0, 0.0],  # never recorded for this (teacher) row
            ],
            dtype=np.float32,
        ),
        "action_taken": np.asarray([5, 5, 5], dtype=np.int64),
    }
    data.update(overrides)
    return data


# --------------------------------------------------------------------------- compute_policy_surprise_kl


def test_returns_zeros_without_prior_policy_field():
    data = _base_data()
    del data["prior_policy"]

    kl, has_prior = compute_policy_surprise_kl(data)

    assert kl.tolist() == [0.0, 0.0, 0.0]
    assert has_prior.tolist() == [False, False, False]


def test_scopes_to_rows_with_a_recorded_prior_only():
    data = _base_data()

    kl, has_prior = compute_policy_surprise_kl(data)

    assert has_prior.tolist() == [True, True, False]
    assert kl[2] == 0.0  # no recorded prior -> zeroed, not left to eps-clamp math


def test_kl_matches_manual_computation_and_orders_by_surprise():
    """Row 1 (target far from prior) must have strictly higher KL than row 0
    (target == prior) -- this is the ordering CAT-45's sampler leans on."""
    data = _base_data()

    kl, has_prior = compute_policy_surprise_kl(data)

    expected_row0 = 0.0  # target == prior -> KL == 0 exactly
    assert float(kl[0]) == pytest.approx(expected_row0, abs=1e-5)
    assert float(kl[1]) > float(kl[0])


def test_respects_batch_indexing():
    data = _base_data()

    kl_full, has_prior_full = compute_policy_surprise_kl(data)
    kl_batch, has_prior_batch = compute_policy_surprise_kl(data, batch=np.asarray([1, 2]))

    assert kl_batch.tolist() == pytest.approx(kl_full[[1, 2]].tolist())
    assert has_prior_batch.tolist() == has_prior_full[[1, 2]].tolist()


# --------------------------------------------------------------------------- policy_surprise_sampling_weights


def test_weight_scale_zero_is_uniform_regardless_of_surprise():
    """The default-off case: weight_scale=0.0 must yield exactly 1.0 for every
    row, including high-surprise ones -- this is the regression-safety default."""
    data = _base_data()
    kl, has_prior = compute_policy_surprise_kl(data)

    weights = policy_surprise_sampling_weights(kl, has_prior, weight_scale=0.0, cap=4.0)

    assert weights.tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_weight_scale_positive_upweights_high_surprise_rows():
    data = _base_data()
    kl, has_prior = compute_policy_surprise_kl(data)

    weights = policy_surprise_sampling_weights(kl, has_prior, weight_scale=1.0, cap=4.0)

    assert weights[0] == pytest.approx(1.0)  # KL ~= 0 -> baseline weight
    assert weights[1] > weights[0]  # high-surprise row upweighted
    assert weights[2] == pytest.approx(1.0)  # no recorded prior -> baseline weight


def test_cap_bounds_the_maximum_weight():
    kl = np.asarray([100.0], dtype=np.float32)
    has_prior = np.asarray([True])

    weights = policy_surprise_sampling_weights(kl, has_prior, weight_scale=1.0, cap=4.0)

    assert weights[0] == pytest.approx(1.0 + 1.0 * 4.0)


# --------------------------------------------------------------------------- _epoch_order


def _ddp_disabled():
    return {"enabled": False, "world_size": 1, "rank": 0, "local_rank": 0}


def test_epoch_order_default_matches_plain_permutation():
    """sample_weights=None must reuse today's exact rng.permutation call -- not
    a uniform-weighted rng.choice -- so existing seeded runs stay reproducible."""
    rng_a = np.random.default_rng(1234)
    rng_b = np.random.default_rng(1234)

    order = _epoch_order(rng_a, 10, 4, _ddp_disabled())
    expected = rng_b.permutation(10)

    assert order.tolist() == expected.tolist()


def test_epoch_order_weighted_sampling_shifts_frequency_toward_high_weight_rows():
    """Synthetic rows with a known weight ordering: over many epochs, the
    empirical draw-frequency ordering must match the weight ordering."""
    rng = np.random.default_rng(0)
    n = 5
    weights = np.asarray([1.0, 1.0, 5.0, 1.0, 20.0])  # row 4 >> row 2 >> rows 0/1/3

    counts = np.zeros(n, dtype=np.int64)
    for _ in range(2000):
        order = _epoch_order(rng, n, 4, _ddp_disabled(), sample_weights=weights)
        assert len(order) == n
        counts += np.bincount(order, minlength=n)

    assert counts[4] > counts[2] > counts[0]
    assert counts[4] > counts[1] > 0
    assert counts[4] > counts[3] > 0


def test_epoch_order_rejects_mismatched_weight_length():
    rng = np.random.default_rng(0)

    with pytest.raises(ValueError):
        _epoch_order(rng, 5, 4, _ddp_disabled(), sample_weights=np.ones(3))


def test_epoch_order_rejects_all_zero_weights():
    rng = np.random.default_rng(0)

    with pytest.raises(ValueError):
        _epoch_order(rng, 3, 4, _ddp_disabled(), sample_weights=np.zeros(3))
