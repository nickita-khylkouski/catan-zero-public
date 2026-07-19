from __future__ import annotations

import numpy as np
import pytest

from tools.train_bc import (
    _array_content_sha256,
    _component_game_identities,
    _compose_per_game_policy_surprise_sampling_weights,
    _coverage_fixed_loss_normalizers,
    _coverage_importance_weights,
    _coverage_policy_signal_admission,
    _exact_two_stream_policy_phase_objective_mass_admission,
    _epoch_order,
    _policy_action_type_target_mass_admission,
    _policy_phase_objective_mass_admission,
    _require_exact_two_stream_phase_mass_admission,
    compute_policy_surprise_kl,
    per_game_capped_policy_surprise_sampling_weights,
    per_game_policy_surprise_sampling_report,
    policy_surprise_sampling_weights,
)


def _two_stream_base_receipt(*, world_size: int = 1, accum: int = 1):
    """Minimal planned base receipt: one synchronous microbatch in epoch zero."""

    return {
        "identity_sha256": "base-receipt",
        "geometry": {
            "world_size": world_size,
            "grad_accum_steps": accum,
        },
        "epoch_receipts": [
            {
                "epoch": 0,
                "consumed_synchronous_global_microbatch_count": 1,
            }
        ],
        "per_phase": {"PLAY_TURN": {"policy_objective_mass_fraction": 1.0}},
    }


def _two_stream_phase_data():
    return {
        "phase": np.asarray(
            [
                "PLAY_TURN",
                "PLAY_TURN",
                "PLAY_TURN",
                "PLAY_TURN",
                "BUILD_INITIAL_SETTLEMENT",
                "BUILD_INITIAL_ROAD",
                "DISCARD",
                "MOVE_ROBBER",
            ]
        )
    }


def test_exact_two_stream_admission_accepts_aux_only_hard_phase_mass():
    report = _exact_two_stream_policy_phase_objective_mass_admission(
        _two_stream_phase_data(),
        np.arange(8, dtype=np.int64),
        base_admission=_two_stream_base_receipt(),
        policy_sample_weights=np.ones(8, dtype=np.float64),
        policy_aux_sampling_weights=np.ones(8, dtype=np.float64) / 8.0,
        policy_aux_phase_loss_weights=None,
        minimum_phase_mass_fractions={
            "BUILD_INITIAL_SETTLEMENT": 0.02,
            "BUILD_INITIAL_ROAD": 0.02,
            "DISCARD": 0.02,
            "MOVE_ROBBER": 0.02,
        },
        policy_base_loss_weight=0.0,
        policy_aux_loss_weight=1.0,
        policy_aux_active_batch_size=8,
        sampler_seed=2,
        policy_aux_sampling_mode="weighted_permutation_cycles_v1",
        ddp={"enabled": False, "world_size": 1, "rank": 0, "local_rank": 0},
        initial_policy_aux_global_draw_offset=0,
        max_steps=1,
        batch_size=8,
        grad_accum_steps=1,
        completed_optimizer_steps=0,
    )

    assert report["admitted"] is True
    assert report["schema_version"] == "policy-phase-exact-two-stream-admission-v1"
    assert report["per_phase"]["BUILD_INITIAL_SETTLEMENT"][
        "policy_objective_mass_fraction"
    ] == pytest.approx(1.0 / 8.0)
    assert report["per_phase"]["PLAY_TURN"][
        "policy_objective_mass_fraction"
    ] == pytest.approx(4.0 / 8.0)


def test_exact_two_stream_aux_only_derives_geometry_without_base_receipt():
    report = _exact_two_stream_policy_phase_objective_mass_admission(
        _two_stream_phase_data(),
        np.arange(8, dtype=np.int64),
        # Direct authenticated corpora can use a non-planned base sampler. With
        # c_base=0 that sampler has no policy contribution and must not block
        # the exact AUX-only schedule.
        base_admission={"identity_sha256": "direct-base-no-plan", "per_phase": {}},
        policy_sample_weights=np.ones(8, dtype=np.float64),
        policy_aux_sampling_weights=np.ones(8, dtype=np.float64) / 8.0,
        policy_aux_phase_loss_weights=None,
        minimum_phase_mass_fractions={
            "BUILD_INITIAL_SETTLEMENT": 0.02,
            "BUILD_INITIAL_ROAD": 0.02,
            "DISCARD": 0.02,
            "MOVE_ROBBER": 0.02,
        },
        policy_base_loss_weight=0.0,
        policy_aux_loss_weight=1.0,
        policy_aux_active_batch_size=8,
        sampler_seed=2,
        policy_aux_sampling_mode="weighted_permutation_cycles_v1",
        ddp={"enabled": False, "world_size": 1, "rank": 0, "local_rank": 0},
        initial_policy_aux_global_draw_offset=0,
        max_steps=1,
        batch_size=64,
        grad_accum_steps=1,
        completed_optimizer_steps=0,
    )

    assert report["admitted"] is True
    assert report["base_planned_receipt_available"] is False
    assert report["aux_planned_synchronous_global_microbatch_count"] == 1


def test_exact_two_stream_admission_uses_loss_coefficients_and_undoes_phase_weights():
    policy_weights = np.asarray([4.0, 4.0, 4.0, 4.0, 2.0, 2.0, 2.0, 2.0])
    report = _exact_two_stream_policy_phase_objective_mass_admission(
        _two_stream_phase_data(),
        np.arange(8, dtype=np.int64),
        base_admission=_two_stream_base_receipt(),
        policy_sample_weights=policy_weights,
        policy_aux_sampling_weights=np.ones(8, dtype=np.float64) / 8.0,
        policy_aux_phase_loss_weights={
            "PLAY_TURN": 4.0,
            "BUILD_INITIAL_SETTLEMENT": 2.0,
            "BUILD_INITIAL_ROAD": 2.0,
            "DISCARD": 2.0,
            "MOVE_ROBBER": 2.0,
        },
        minimum_phase_mass_fractions=None,
        policy_base_loss_weight=1.0,
        policy_aux_loss_weight=1.0,
        policy_aux_active_batch_size=8,
        sampler_seed=2,
        policy_aux_sampling_mode="weighted_permutation_cycles_v1",
        ddp={"enabled": False, "world_size": 1, "rank": 0, "local_rank": 0},
        initial_policy_aux_global_draw_offset=0,
        max_steps=1,
        batch_size=8,
        grad_accum_steps=1,
        completed_optimizer_steps=0,
    )

    # The phase allocation has already selected the AUX measure, so undoing
    # phase loss multipliers yields 4/8 PLAY and 1/8 for every hard phase.
    # With equal stream coefficients, the base's all-PLAY mass contributes half.
    assert report["per_phase"]["PLAY_TURN"][
        "policy_objective_mass_fraction"
    ] == pytest.approx(0.75)
    assert report["per_phase"]["DISCARD"][
        "policy_objective_mass_fraction"
    ] == pytest.approx(1.0 / 16.0)


def test_exact_two_stream_admission_refuses_aux_hard_phase_below_floor():
    with pytest.raises(SystemExit, match="exact two-stream"):
        _exact_two_stream_policy_phase_objective_mass_admission(
            _two_stream_phase_data(),
            np.arange(8, dtype=np.int64),
            base_admission=_two_stream_base_receipt(),
            policy_sample_weights=np.ones(8, dtype=np.float64),
            policy_aux_sampling_weights=np.ones(8, dtype=np.float64) / 8.0,
            policy_aux_phase_loss_weights=None,
            minimum_phase_mass_fractions={
                "BUILD_INITIAL_SETTLEMENT": 0.2,
                "BUILD_INITIAL_ROAD": 0.2,
                "DISCARD": 0.2,
                "MOVE_ROBBER": 0.2,
            },
            policy_base_loss_weight=0.0,
            policy_aux_loss_weight=1.0,
            policy_aux_active_batch_size=8,
            sampler_seed=2,
            policy_aux_sampling_mode="weighted_permutation_cycles_v1",
            ddp={"enabled": False, "world_size": 1, "rank": 0, "local_rank": 0},
            initial_policy_aux_global_draw_offset=0,
            max_steps=1,
            batch_size=8,
            grad_accum_steps=1,
            completed_optimizer_steps=0,
        )


def test_hard_decision_floors_defer_aux_objective_to_exact_two_stream_replay() -> None:
    """The early validator must not reject a valid AUX-only objective."""

    _require_exact_two_stream_phase_mass_admission(
        {"PLAY_TURN": 0.1},
        policy_aux_active_batch_size=512,
        policy_base_loss_weight=0.0,
        policy_aux_loss_weight=1.0,
    )


def test_hard_decision_floors_keep_single_stream_admission_available() -> None:
    _require_exact_two_stream_phase_mass_admission(
        {"PLAY_TURN": 0.1},
        policy_aux_active_batch_size=0,
        policy_base_loss_weight=1.0,
        policy_aux_loss_weight=0.0,
    )


def _planned_weighted_phase_admission(
    phases,
    *,
    policy_weights,
    sampling_weights,
    seed=7,
    epochs=1,
    max_steps=0,
    batch_size=1,
    world_size=1,
    grad_accum_steps=1,
    completed_epochs=0,
    completed_optimizer_steps=0,
    sampler_rng_state=None,
):
    rng = np.random.default_rng(seed)
    return _policy_phase_objective_mass_admission(
        {"phase": np.asarray(phases)},
        np.arange(len(phases), dtype=np.int64),
        policy_sample_weights=np.asarray(policy_weights, dtype=np.float64),
        sampling_weights=np.asarray(sampling_weights, dtype=np.float64),
        minimum_phase_mass_fractions=None,
        objective_measure=(
            "weighted_replacement_draw_probability_x_policy_loss_weight_v1"
        ),
        sampler_rng_state=(
            rng.bit_generator.state
            if sampler_rng_state is None
            else sampler_rng_state
        ),
        epochs=epochs,
        max_steps=max_steps,
        batch_size=batch_size,
        world_size=world_size,
        grad_accum_steps=grad_accum_steps,
        completed_epochs=completed_epochs,
        completed_optimizer_steps=completed_optimizer_steps,
    )


def _maritime_target_mass_admission(data, **overrides):
    kwargs = {
        "policy_sample_weights": np.ones(len(data["action_taken"]), dtype=np.float64),
        "sampling_weights": None,
        "action_types_by_id": ("END_TURN", "MARITIME_TRADE", "BUILD_ROAD"),
        "target_action_type": "MARITIME_TRADE",
        "minimum_target_mass_fraction": 0.01,
        "soft_target_temperature": 1.0,
        "soft_target_source": "policy",
        "soft_target_min_legal_coverage": 1.0,
        "soft_target_weight": 1.0,
        "policy_target_blend_semantics": "policy_target_fallback_v2",
        "advantage_policy_weighting": "none",
        "policy_aux_active_batch_size": 0,
        "objective_measure": (
            "uniform_coverage_row_probability_x_policy_loss_weight_v1"
        ),
    }
    kwargs.update(overrides)
    return _policy_action_type_target_mass_admission(
        data,
        np.arange(len(data["action_taken"]), dtype=np.int64),
        **kwargs,
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


def test_coverage_normalizers_match_effective_policy_and_value_masks() -> None:
    data = {
        "action_taken": np.zeros(3, dtype=np.int16),
        "winner": np.asarray(["RED", "", ""]),
        "player": np.asarray(["RED", "BLUE", "RED"]),
        "truncated": np.asarray([False, False, True]),
        "seat": np.asarray([1, 0, 1], dtype=np.int8),
        "final_actual_vps": np.asarray(
            [[0, 10, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int16
        ),
        "has_final_actual_vps": np.asarray([True, False, False]),
        "final_public_vps": np.asarray(
            [[0, 10, 0, 0], [0, 0, 0, 0], [8, 6, 0, 0]], dtype=np.int16
        ),
        "has_final_public_vps": np.asarray([True, False, False]),
    }

    normalizers = _coverage_fixed_loss_normalizers(
        data,
        np.arange(3, dtype=np.int64),
        policy_sample_weights=np.asarray([1.0, 0.0, 2.0], dtype=np.float32),
        value_sample_weights=np.ones(3, dtype=np.float32),
        truncated_vp_margin_value_weight=0.25,
        vps_to_win=10,
        public_information_only=True,
    )

    assert normalizers["policy_effective_weight_mean"] == pytest.approx(1.0)
    assert normalizers["value_effective_weight_mean"] == pytest.approx(1.25 / 3.0)
    assert normalizers["final_vp_effective_weight_mean"] == pytest.approx(1.0 / 3.0)


def test_coverage_policy_admission_rejects_audited_sparse_signal() -> None:
    rows = np.arange(959_142, dtype=np.int64)
    weights = np.zeros(rows.size, dtype=np.float64)
    weights[:8_178] = 1.0
    # One heavier synthetic row exactly reproduces the audited 6,835.96 ESS
    # without copying any private corpus payload into the repository.
    weights[0] = 41.268579058650666

    report = _coverage_policy_signal_admission(
        rows,
        policy_sample_weights=weights,
        global_batch_size=512,
        minimum_effective_rows_per_global_batch=0.0,
    )

    assert report["policy_active_row_count"] == 8_178
    assert report["policy_effective_sample_size_rows"] == pytest.approx(6_835.96)
    assert report["expected_policy_active_rows_per_global_batch"] == pytest.approx(
        4.36550, rel=1e-5
    )
    assert report["expected_policy_effective_rows_per_global_batch"] == pytest.approx(
        3.64911, rel=1e-5
    )

    with pytest.raises(SystemExit, match="refused sparse/concentrated"):
        _coverage_policy_signal_admission(
            rows,
            policy_sample_weights=weights,
            global_batch_size=512,
            minimum_effective_rows_per_global_batch=32.0,
        )


def test_coverage_policy_admission_accepts_commissioned_signal_margin() -> None:
    rows = np.arange(4_096, dtype=np.int64)
    weights = np.zeros(rows.size, dtype=np.float32)
    weights[:1_024] = 1.0

    report = _coverage_policy_signal_admission(
        rows,
        policy_sample_weights=weights,
        global_batch_size=512,
        minimum_effective_rows_per_global_batch=32.0,
    )

    assert report["policy_active_row_fraction"] == pytest.approx(0.25)
    assert report["policy_effective_sample_fraction"] == pytest.approx(0.25)
    assert report["expected_policy_effective_rows_per_global_batch"] == pytest.approx(
        128.0
    )
    assert report["admitted"] is True


def test_coverage_policy_admission_detects_weight_concentration() -> None:
    rows = np.arange(1_024, dtype=np.int64)
    weights = np.ones(rows.size, dtype=np.float64)
    weights[0] = 1_024.0

    with pytest.raises(SystemExit, match="effective_sample_size"):
        _coverage_policy_signal_admission(
            rows,
            policy_sample_weights=weights,
            global_batch_size=512,
            minimum_effective_rows_per_global_batch=32.0,
        )


def test_coverage_policy_admission_uses_synchronous_ddp_batch_geometry() -> None:
    rows = np.arange(1_000, dtype=np.int64)
    weights = np.zeros(rows.size, dtype=np.float32)
    weights[:100] = 1.0

    report = _coverage_policy_signal_admission(
        rows,
        policy_sample_weights=weights,
        global_batch_size=64 * 8 * 2,
        minimum_effective_rows_per_global_batch=32.0,
    )

    assert report["effective_global_batch_size"] == 1_024
    assert report["expected_policy_active_rows_per_global_batch"] == pytest.approx(
        102.4
    )
    assert report["expected_policy_effective_rows_per_global_batch"] == pytest.approx(
        102.4
    )


def test_coverage_policy_admission_is_global_topology_equivalent() -> None:
    rows = np.arange(1_000, dtype=np.int64)
    weights = np.zeros(rows.size, dtype=np.float32)
    weights[:100] = 1.0

    reports = [
        _coverage_policy_signal_admission(
            rows,
            policy_sample_weights=weights,
            global_batch_size=local * world * accumulation,
            minimum_effective_rows_per_global_batch=32.0,
        )
        for local, world, accumulation in ((512, 1, 1), (64, 8, 1), (64, 4, 2))
    ]

    assert [
        report["expected_policy_effective_rows_per_global_batch"]
        for report in reports
    ] == pytest.approx([51.2, 51.2, 51.2])


def test_zero_floor_reports_absent_policy_signal_without_new_refusal() -> None:
    report = _coverage_policy_signal_admission(
        np.arange(32, dtype=np.int64),
        policy_sample_weights=np.zeros(32, dtype=np.float32),
        global_batch_size=32,
        minimum_effective_rows_per_global_batch=0.0,
    )

    assert report["admission_enforced"] is False
    assert report["admitted"] is True
    assert report["signal_present"] is False
    assert report["expected_policy_effective_rows_per_global_batch"] == 0.0


def test_hard_decision_phase_mass_reports_sparse_objective_without_floor() -> None:
    phases = np.asarray(
        ["PLAY_TURN"] * 100
        + [
            "BUILD_INITIAL_SETTLEMENT",
            "BUILD_INITIAL_ROAD",
            "DISCARD",
            "MOVE_ROBBER",
        ]
    )
    weights = np.asarray([4.0] * 100 + [1.0] * 4, dtype=np.float64)

    report = _policy_phase_objective_mass_admission(
        {"phase": phases},
        np.arange(phases.size, dtype=np.int64),
        policy_sample_weights=weights,
        sampling_weights=None,
        minimum_phase_mass_fractions=None,
        objective_measure="synthetic_uniform_coverage",
    )

    settlement = report["per_phase"]["BUILD_INITIAL_SETTLEMENT"]
    road = report["per_phase"]["BUILD_INITIAL_ROAD"]
    discard = report["per_phase"]["DISCARD"]
    robber = report["per_phase"]["MOVE_ROBBER"]
    assert report["admission_enforced"] is False
    assert report["admitted"] is None
    assert report["schema_version"] == "policy-phase-objective-mass-admission-v2"
    assert report["identity_sha256"].startswith("sha256:")
    assert settlement["policy_objective_mass_fraction"] == pytest.approx(1 / 404)
    assert road["policy_objective_mass_fraction"] == pytest.approx(1 / 404)
    assert discard["policy_objective_mass_fraction"] == pytest.approx(1 / 404)
    assert robber["policy_objective_mass_fraction"] == pytest.approx(1 / 404)


def test_hard_decision_phase_mass_rejects_sparse_objective_before_training() -> None:
    phases = np.asarray(
        ["PLAY_TURN"] * 100
        + [
            "BUILD_INITIAL_SETTLEMENT",
            "BUILD_INITIAL_ROAD",
            "DISCARD",
            "MOVE_ROBBER",
        ]
    )
    weights = np.asarray([4.0] * 100 + [1.0] * 4, dtype=np.float64)

    with pytest.raises(SystemExit, match="refused before the first optimizer step"):
        _policy_phase_objective_mass_admission(
            {"phase": phases},
            np.arange(phases.size, dtype=np.int64),
            policy_sample_weights=weights,
            sampling_weights=None,
            minimum_phase_mass_fractions={
                "BUILD_INITIAL_SETTLEMENT": 0.01,
                "BUILD_INITIAL_ROAD": 0.01,
                "DISCARD": 0.01,
                "MOVE_ROBBER": 0.01,
            },
            objective_measure="synthetic_uniform_coverage",
        )


def test_hard_decision_phase_mass_accepts_minima_and_binds_identity() -> None:
    phases = np.asarray(
        ["PLAY_TURN"] * 100
        + [
            "BUILD_INITIAL_SETTLEMENT",
            "BUILD_INITIAL_ROAD",
            "DISCARD",
            "MOVE_ROBBER",
        ]
    )
    weights = np.asarray([4.0] * 100 + [8.0] * 4, dtype=np.float64)
    minima = {
        "BUILD_INITIAL_SETTLEMENT": 0.01,
        "BUILD_INITIAL_ROAD": 0.01,
        "DISCARD": 0.01,
        "MOVE_ROBBER": 0.01,
    }

    report = _policy_phase_objective_mass_admission(
        {"phase": phases},
        np.arange(phases.size, dtype=np.int64),
        policy_sample_weights=weights,
        sampling_weights=None,
        minimum_phase_mass_fractions=minima,
        objective_measure="synthetic_uniform_coverage",
    )

    assert report["admission_enforced"] is True
    assert report["admitted"] is True
    assert report["minimum_phase_mass_fractions"] == minima
    assert all(
        report["per_phase"][phase]["admitted"] is True for phase in minima
    )


def test_weighted_phase_admission_uses_planned_batch_normalization() -> None:
    phases = ["BUILD_INITIAL_SETTLEMENT", "PLAY_TURN"]
    report = _planned_weighted_phase_admission(
        phases,
        policy_weights=[100.0, 1.0],
        sampling_weights=[0.1, 0.9],
        epochs=50,
        max_steps=100,
        batch_size=1,
    )
    control = np.random.default_rng(7)
    sampled = np.concatenate(
        [
            control.choice(2, size=2, replace=True, p=[0.1, 0.9])
            for _epoch in range(50)
        ]
    )
    exact_batch_mean = float(np.mean(sampled == 0))
    ratio_of_expectations = (0.1 * 100.0) / (0.1 * 100.0 + 0.9)

    settlement = report["per_phase"]["BUILD_INITIAL_SETTLEMENT"]
    assert report["schema_version"] == "policy-phase-planned-batch-admission-v1"
    assert settlement["policy_objective_mass_fraction"] == pytest.approx(
        exact_batch_mean
    )
    assert exact_batch_mean < 0.3
    assert ratio_of_expectations > 0.9
    assert report["consumed_optimizer_batch_count"] == 100
    assert report["total_batch_normalized_policy_attribution"] == pytest.approx(
        1.0
    )


def test_weighted_phase_admission_does_not_advance_live_sampler_rng() -> None:
    live_rng = np.random.default_rng(19)
    control_rng = np.random.default_rng(19)

    _planned_weighted_phase_admission(
        ["BUILD_INITIAL_SETTLEMENT", "PLAY_TURN"] * 4,
        policy_weights=np.ones(8),
        sampling_weights=np.ones(8),
        epochs=2,
        batch_size=2,
        sampler_rng_state=live_rng.bit_generator.state,
    )

    assert live_rng.integers(0, 2**31, size=16).tolist() == control_rng.integers(
        0, 2**31, size=16
    ).tolist()


def test_weighted_phase_admission_binds_ddp_and_accumulation_geometry() -> None:
    report = _planned_weighted_phase_admission(
        [
            "BUILD_INITIAL_SETTLEMENT",
            "BUILD_INITIAL_ROAD",
            "DISCARD",
            "MOVE_ROBBER",
            "PLAY_TURN",
        ],
        policy_weights=np.ones(5),
        sampling_weights=np.ones(5),
        epochs=1,
        batch_size=1,
        world_size=2,
        grad_accum_steps=2,
    )

    assert report["raw_sampler_draw_count"] == 5
    assert report["consumed_global_row_draw_count"] == 6
    assert report["planned_synchronous_global_microbatch_count"] == 3
    assert report["consumed_synchronous_global_microbatch_count"] == 3
    assert report["planned_optimizer_batch_count"] == 2
    assert report["consumed_optimizer_batch_count"] == 2
    assert report["geometry"] == {
        "local_microbatch_size": 1,
        "world_size": 2,
        "grad_accum_steps": 2,
        "synchronous_global_microbatch_size": 2,
        "nominal_optimizer_global_batch_size": 4,
        "epoch_limit": 1,
        "start_completed_epochs": 0,
        "max_steps": 0,
        "start_optimizer_step": 0,
        "end_optimizer_step": 2,
    }


def test_weighted_phase_admission_matches_ddp_global_weight_denominator() -> None:
    report = _planned_weighted_phase_admission(
        ["BUILD_INITIAL_SETTLEMENT", "PLAY_TURN"],
        policy_weights=[100.0, 1.0],
        sampling_weights=[0.5, 0.5],
        seed=8,  # exact sampled global order [0, 1]
        epochs=1,
        batch_size=1,
        world_size=2,
    )

    # _weighted_mean_from_parts all-reduces the denominator, then scales each
    # local numerator by world size so DDP's gradient average is the global
    # ratio. A mean of independently normalized rank losses would be 0.5 and
    # is explicitly not the live trainer estimator.
    assert report["per_phase"]["BUILD_INITIAL_SETTLEMENT"][
        "policy_objective_mass_fraction"
    ] == pytest.approx(100.0 / 101.0)


def test_weighted_phase_admission_truncates_exactly_at_max_steps() -> None:
    report = _planned_weighted_phase_admission(
        ["PLAY_TURN"] * 100,
        policy_weights=np.ones(100),
        sampling_weights=np.ones(100),
        epochs=10,
        max_steps=2,
        batch_size=2,
        world_size=2,
        grad_accum_steps=3,
    )

    assert report["planned_epoch_count"] == 1
    assert report["raw_sampler_draw_count"] == 24
    assert report["consumed_global_row_draw_count"] == 24
    assert report["consumed_synchronous_global_microbatch_count"] == 6
    assert report["consumed_optimizer_batch_count"] == 2
    assert report["geometry"]["end_optimizer_step"] == 2


def test_weighted_phase_resume_replays_saved_sampler_state_and_binds_identity() -> None:
    phases = np.asarray(["PLAY_TURN", "BUILD_INITIAL_ROAD"] * 4)
    sampling_weights = np.arange(1, 9, dtype=np.float64)
    control = np.random.default_rng(31)
    control.choice(
        8,
        size=8,
        replace=True,
        p=sampling_weights / sampling_weights.sum(),
    )
    resumed_state = control.bit_generator.state
    expected_order = control.choice(
        8,
        size=8,
        replace=True,
        p=sampling_weights / sampling_weights.sum(),
    )

    report = _planned_weighted_phase_admission(
        phases,
        policy_weights=np.ones(8),
        sampling_weights=sampling_weights,
        epochs=2,
        batch_size=2,
        completed_epochs=1,
        completed_optimizer_steps=4,
        sampler_rng_state=resumed_state,
    )
    repeated = _planned_weighted_phase_admission(
        phases,
        policy_weights=np.ones(8),
        sampling_weights=sampling_weights,
        epochs=2,
        batch_size=2,
        completed_epochs=1,
        completed_optimizer_steps=4,
        sampler_rng_state=resumed_state,
    )
    different_progress = _planned_weighted_phase_admission(
        phases,
        policy_weights=np.ones(8),
        sampling_weights=sampling_weights,
        epochs=2,
        batch_size=2,
        completed_epochs=1,
        completed_optimizer_steps=3,
        sampler_rng_state=resumed_state,
    )

    assert report["epoch_receipts"][0]["raw_order_sha256"] == (
        _array_content_sha256(expected_order)
    )
    assert report["geometry"]["start_optimizer_step"] == 4
    assert report["geometry"]["end_optimizer_step"] == 8
    assert report["identity_sha256"] == repeated["identity_sha256"]
    assert report["identity_sha256"] != different_progress["identity_sha256"]


def test_maritime_target_mass_uses_fixed_normalizer_coverage_measure() -> None:
    data = {
        "legal_action_ids": np.asarray([[0, 1], [0, 1]], dtype=np.int16),
        "target_policy": np.asarray([[0.8, 0.2], [0.2, 0.8]], dtype=np.float32),
        "target_policy_mask": np.ones((2, 2), dtype=np.bool_),
        "action_taken": np.asarray([0, 0], dtype=np.int64),
    }

    report = _maritime_target_mass_admission(
        data,
        policy_sample_weights=np.asarray([2.0, 1.0], dtype=np.float64),
        minimum_target_mass_fraction=0.39,
    )

    # The fixed coverage normalizer preserves the final row coefficients:
    # (2*0.2 + 1*0.8) / (2+1) = 0.4.
    assert report["target_action_policy_objective_mass_fraction"] == pytest.approx(
        0.4
    )
    assert report["selected_target_action_row_count_diagnostic"] == 0
    assert report["admitted"] is True
    assert report["identity_sha256"].startswith("sha256:")


def test_maritime_target_mass_refuses_weighted_replacement_attribution() -> None:
    data = {
        "legal_action_ids": np.asarray([[0, 1], [0, 1]], dtype=np.int16),
        "target_policy": np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        "target_policy_mask": np.ones((2, 2), dtype=np.bool_),
        "action_taken": np.asarray([1, 0], dtype=np.int64),
    }

    with pytest.raises(SystemExit, match="requires exact fixed-normalizer coverage"):
        _maritime_target_mass_admission(
            data,
            policy_sample_weights=np.asarray([100.0, 1.0], dtype=np.float64),
            sampling_weights=np.asarray([0.1, 0.9], dtype=np.float64),
            objective_measure=(
                "weighted_replacement_draw_probability_x_policy_loss_weight_v1"
            ),
        )


def test_maritime_target_mass_does_not_use_exploration_action_taken() -> None:
    selected_maritime_zero_target = {
        "legal_action_ids": np.asarray([[0, 1]], dtype=np.int16),
        "target_policy": np.asarray([[1.0, 0.0]], dtype=np.float32),
        "target_policy_mask": np.ones((1, 2), dtype=np.bool_),
        "action_taken": np.asarray([1], dtype=np.int64),
    }
    with pytest.raises(
        SystemExit,
        match="refused before the first optimizer step",
    ):
        _maritime_target_mass_admission(selected_maritime_zero_target)

    selected_non_maritime_positive_target = {
        **selected_maritime_zero_target,
        "target_policy": np.asarray([[0.7, 0.3]], dtype=np.float32),
        "action_taken": np.asarray([0], dtype=np.int64),
    }
    report = _maritime_target_mass_admission(
        selected_non_maritime_positive_target,
        minimum_target_mass_fraction=0.25,
    )
    assert report["target_action_policy_objective_mass_fraction"] == pytest.approx(
        0.3
    )
    assert report["selected_target_action_row_count_diagnostic"] == 0


def test_maritime_target_mass_rejects_action_catalog_drift() -> None:
    data = {
        "legal_action_ids": np.asarray([[0, 99]], dtype=np.int16),
        "target_policy": np.asarray([[0.5, 0.5]], dtype=np.float32),
        "target_policy_mask": np.ones((1, 2), dtype=np.bool_),
        "action_taken": np.asarray([0], dtype=np.int64),
    }

    with pytest.raises(SystemExit, match="outside the active ActionCatalog"):
        _maritime_target_mass_admission(data)


def test_maritime_target_mass_rejects_missing_effective_soft_target() -> None:
    data = {
        "legal_action_ids": np.asarray([[0, 1]], dtype=np.int16),
        "target_policy": np.zeros((1, 2), dtype=np.float32),
        "target_policy_mask": np.ones((1, 2), dtype=np.bool_),
        "action_taken": np.asarray([1], dtype=np.int64),
    }

    with pytest.raises(SystemExit, match="without a soft teacher"):
        _maritime_target_mass_admission(data)


class _SyntheticComposite(dict):
    component_ids = ("current", "replay")

    def component_indices_for_rows(self, rows):
        return np.asarray(rows, dtype=np.int64) // 2


class _SyntheticScopedComposite(dict):
    component_ids = ("current", "replay", "value_only")
    policy_distillation_scope_authenticated = True
    policy_distillation_component_indices = (0, 1)

    def component_indices_for_rows(self, rows):
        return np.asarray(rows, dtype=np.int64) // 2


def test_maritime_target_mass_reports_each_authenticated_component() -> None:
    data = _SyntheticComposite(
        legal_action_ids=np.asarray([[0, 1]] * 4, dtype=np.int16),
        target_policy=np.asarray(
            [[0.9, 0.1], [0.7, 0.3], [0.8, 0.2], [0.4, 0.6]],
            dtype=np.float32,
        ),
        target_policy_mask=np.ones((4, 2), dtype=np.bool_),
        action_taken=np.zeros(4, dtype=np.int64),
    )

    report = _maritime_target_mass_admission(
        data,
        policy_sample_weights=np.asarray([1.0, 3.0, 2.0, 1.0]),
        minimum_target_mass_fraction=0.19,
    )

    assert report["authenticated_composite_components_enforced"] is True
    assert report["per_component"]["current"][
        "target_action_policy_objective_mass_fraction"
    ] == pytest.approx(0.25)
    assert report["per_component"]["replay"][
        "target_action_policy_objective_mass_fraction"
    ] == pytest.approx(1.0 / 3.0)
    assert all(
        component["admitted"] for component in report["per_component"].values()
    )


def test_maritime_target_mass_excludes_authenticated_value_only_component() -> None:
    data = _SyntheticScopedComposite(
        legal_action_ids=np.asarray([[0, 1]] * 6, dtype=np.int16),
        target_policy=np.asarray(
            [[0.8, 0.2]] * 4 + [[1.0, 0.0]] * 2,
            dtype=np.float32,
        ),
        target_policy_mask=np.ones((6, 2), dtype=np.bool_),
        action_taken=np.zeros(6, dtype=np.int64),
    )

    report = _maritime_target_mass_admission(
        data,
        policy_sample_weights=np.asarray([1.0] * 4 + [0.0] * 2),
        minimum_target_mass_fraction=0.1,
    )

    assert report["authenticated_policy_distillation_scope_enforced"] is True
    assert report["enforced_component_ids"] == ["current", "replay"]
    assert set(report["per_component"]) == {"current", "replay"}
    assert report["admitted"] is True


def test_maritime_target_mass_rejects_starved_authenticated_component() -> None:
    data = _SyntheticComposite(
        legal_action_ids=np.asarray([[0, 1]] * 4, dtype=np.int16),
        target_policy=np.asarray(
            [[0.8, 0.2], [0.8, 0.2], [0.95, 0.05], [0.95, 0.05]],
            dtype=np.float32,
        ),
        target_policy_mask=np.ones((4, 2), dtype=np.bool_),
        action_taken=np.zeros(4, dtype=np.int64),
    )

    with pytest.raises(SystemExit, match=r"failed_components=\['replay'\]"):
        _maritime_target_mass_admission(
            data,
            policy_sample_weights=np.asarray([10.0, 10.0, 1.0, 1.0]),
            minimum_target_mass_fraction=0.1,
        )


@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("policy_target_blend_semantics", "legacy_convex_v1"),
        ("soft_target_source", "prefer_policy"),
        ("soft_target_weight", 0.5),
        ("soft_target_min_legal_coverage", 0.9),
        ("advantage_policy_weighting", "signed"),
        ("policy_aux_active_batch_size", 32),
    ],
)
def test_maritime_target_mass_rejects_unaccounted_objective_contracts(
    override: str,
    value,
) -> None:
    data = {
        "legal_action_ids": np.asarray([[0, 1]], dtype=np.int16),
        "target_policy": np.asarray([[0.5, 0.5]], dtype=np.float32),
        "target_policy_mask": np.ones((1, 2), dtype=np.bool_),
        "action_taken": np.asarray([0], dtype=np.int64),
    }

    with pytest.raises(SystemExit, match="exact pure complete-policy scratch contract"):
        _maritime_target_mass_admission(data, **{override: value})


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
