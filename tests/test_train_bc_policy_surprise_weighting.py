from __future__ import annotations

import numpy as np
import pytest

from tools.train_bc import (
    _component_game_identities,
    _compose_per_game_policy_surprise_sampling_weights,
    _coverage_importance_weights,
    _epoch_order,
    compute_policy_surprise_kl,
    per_game_capped_policy_surprise_sampling_weights,
    per_game_policy_surprise_sampling_report,
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


# ------------------------------------------------ exact per-game policy surprise


def test_per_game_capped_surprise_is_exact_and_mass_preserving():
    seeds = np.asarray([11, 11, 11, 11, 22, 22], dtype=np.int64)
    kl = np.asarray([0.0, 1.0, 100.0, 100.0, 0.0, 0.0], dtype=np.float32)
    # Row 3 models a forced/fast row: its KL is irrelevant because it is not a
    # policy-active root. Game 22 exercises the all-zero-KL fallback.
    active = np.asarray([True, True, True, False, True, True])

    factors = per_game_capped_policy_surprise_sampling_weights(seeds, kl, active)

    # m=3, clipped KL=[0,1,2], sum=3 -> [0.5,1.0,1.5].
    assert factors.tolist() == pytest.approx([0.5, 1.0, 1.5, 1.0, 1.0, 1.0])
    assert float(factors[seeds == 11].sum()) == pytest.approx(4.0)
    assert float(factors[seeds == 22].sum()) == pytest.approx(2.0)


def test_per_game_surprise_does_not_create_policy_loss_on_inactive_rows():
    seeds = np.asarray([7, 7, 7], dtype=np.int64)
    kl = np.asarray([0.0, 2.0, 2.0], dtype=np.float32)
    policy_loss_weights = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)

    factors = per_game_capped_policy_surprise_sampling_weights(
        seeds, kl, policy_loss_weights > 0.0
    )

    assert factors[1:].tolist() == [1.0, 1.0]
    assert (policy_loss_weights * factors)[1:].tolist() == [0.0, 0.0]


class _CompositeData:
    component_game_sampling_ratios = (0.6, 0.4)

    def __init__(self) -> None:
        self._components = np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1])

    def component_indices_for_rows(self, indices):
        return self._components[np.asarray(indices, dtype=np.int64)]


class _CollidingSeedComposite(dict):
    def __init__(self) -> None:
        super().__init__(
            game_seed=np.asarray([7, 7, 7, 7], dtype=np.int64),
        )
        self._components = np.asarray([0, 0, 1, 1], dtype=np.int64)

    def component_indices_for_rows(self, indices):
        return self._components[np.asarray(indices, dtype=np.int64)]


def test_per_game_surprise_namespaces_same_seed_by_component():
    data = _CollidingSeedComposite()
    rows = np.arange(4, dtype=np.int64)
    identities = _component_game_identities(data, rows)
    factors = per_game_capped_policy_surprise_sampling_weights(
        identities,
        np.asarray([0.0, 2.0, 0.0, 0.0], dtype=np.float32),
        np.ones(4, dtype=np.bool_),
    )
    assert factors.tolist() == pytest.approx([0.5, 1.5, 1.0, 1.0])


def test_per_game_surprise_composes_without_changing_component_proportions():
    data = _CompositeData()
    indices = np.arange(9, dtype=np.int64)
    # Component 0 has games of 3 and 2 rows; component 1 has one 4-row game.
    seeds = np.asarray([1, 1, 1, 2, 2, 3, 3, 3, 3], dtype=np.int64)
    active = np.asarray([True, True, False, True, True, True, True, True, False])
    kl = np.asarray([0.0, 2.0, 99.0, 0.5, 1.5, 0.0, 0.5, 2.0, 99.0])
    factors = per_game_capped_policy_surprise_sampling_weights(seeds, kl, active)
    base = np.asarray(
        [0.1, 0.1, 0.1, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1],
        dtype=np.float64,
    )

    combined = _compose_per_game_policy_surprise_sampling_weights(
        data, indices, factors, base
    )

    assert float(combined[:5].sum()) == pytest.approx(0.6)
    assert float(combined[5:].sum()) == pytest.approx(0.4)
    assert float(combined.sum()) == pytest.approx(1.0)


def test_per_game_surprise_report_binds_formula_and_mass_error():
    seeds = np.asarray([1, 1, 1], dtype=np.int64)
    active = np.asarray([True, True, False])
    factors = per_game_capped_policy_surprise_sampling_weights(
        seeds, np.asarray([0.0, 2.0, 9.0]), active
    )

    report = per_game_policy_surprise_sampling_report(
        seeds,
        factors,
        active,
        enabled=True,
        authenticated_component_sampling=True,
    )

    assert report["schema_version"] == "train-policy-surprise-sampling-v2"
    assert report["mode"] == "per_game_capped"
    assert report["kl_cap"] == pytest.approx(2.0)
    assert report["max_per_game_active_mass_error"] == pytest.approx(0.0)
    assert report["authenticated_component_proportions_preserved"] is True


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


def test_coverage_importance_preserves_weighted_population_objective() -> None:
    probabilities = np.asarray([0.05, 0.15, 0.30, 0.50], dtype=np.float64)
    losses = np.asarray([8.0, 4.0, 2.0, 1.0], dtype=np.float64)

    importance = _coverage_importance_weights(probabilities)
    weighted_target = float(np.sum(probabilities * losses))
    permutation_objective = float(np.sum(importance * losses) / importance.sum())

    assert importance.mean() == pytest.approx(1.0)
    assert permutation_objective == pytest.approx(weighted_target)


def test_coverage_permutation_visits_every_row_once() -> None:
    order = _epoch_order(
        np.random.default_rng(7),
        100,
        8,
        _ddp_disabled(),
        sample_weights=None,
    )

    assert len(order) == 100
    assert len(np.unique(order)) == 100


def test_coverage_ddp_rank_strides_cover_every_global_row() -> None:
    n = 101
    world_size = 8
    local_orders = [
        _epoch_order(
            np.random.default_rng(11),
            n,
            4,
            {
                "enabled": True,
                "world_size": world_size,
                "rank": rank,
                "local_rank": rank,
            },
            sample_weights=None,
        )
        for rank in range(world_size)
    ]
    realized = np.concatenate(local_orders)

    assert set(realized.tolist()) == set(range(n))
    assert len(realized) % (world_size * 4) == 0


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
