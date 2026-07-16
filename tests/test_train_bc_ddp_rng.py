from __future__ import annotations

import argparse
import copy

import numpy as np
import pytest
import torch

from tools import train_bc


def _args(
    *, seed: int, enabled: bool, sampler_seed: int | None = None
) -> argparse.Namespace:
    return argparse.Namespace(
        seed=seed,
        sampler_seed=sampler_seed,
        training_rng_rank_offset=enabled,
    )


def _dropout_draw() -> torch.Tensor:
    return torch.nn.functional.dropout(torch.ones(4096), p=0.5, training=True)


def test_historical_flag_off_does_not_mutate_torch_or_numpy_rng() -> None:
    torch.manual_seed(91)
    numpy_rng = np.random.default_rng(73)
    torch_before = torch.get_rng_state().clone()
    numpy_before = numpy_rng.bit_generator.state

    report = train_bc._initialize_training_rng(
        _args(seed=7, enabled=False),
        {"enabled": True, "world_size": 8, "rank": 3, "local_rank": 3},
    )

    assert torch.equal(torch.get_rng_state(), torch_before)
    assert numpy_rng.bit_generator.state == numpy_before
    assert report["effective_torch_seed"] is None
    assert report["rank_offset_enabled"] is False
    assert report["sampler_seed"] == 7
    assert report["sampler_seed_explicit"] is False


def test_warm_start_rebinds_loader_seed_to_configured_training_seed() -> None:
    # Reproduce the loader footgun: constructing the policy reset the process RNG
    # to an internal default unrelated to the learner's configured seed.
    torch.manual_seed(0)
    _ = _dropout_draw()

    report = train_bc._initialize_training_rng(
        _args(seed=137, enabled=False),
        {"enabled": False, "world_size": 1, "rank": 0, "local_rank": 0},
        checkpoint_loaded=True,
    )
    actual = _dropout_draw()

    torch.manual_seed(137)
    expected = _dropout_draw()
    assert torch.equal(actual, expected)
    assert report["effective_torch_seed"] == 137
    assert report["post_load_reseeded"] is True


def test_rank_offset_makes_dropout_independent_but_reproducible() -> None:
    draws = []
    for rank in (0, 1, 0):
        report = train_bc._initialize_training_rng(
            _args(seed=101, enabled=True),
            {"enabled": True, "world_size": 2, "rank": rank, "local_rank": rank},
        )
        draws.append(_dropout_draw())
        assert report["effective_torch_seed"] == 101 + rank

    assert not torch.equal(draws[0], draws[1])
    assert torch.equal(draws[0], draws[2])


def test_same_seed_without_rank_offset_reproduces_identical_rank_masks() -> None:
    draws = []
    for _rank in (0, 1):
        # This is the legacy constructor/load behavior: each process enters its
        # first training forward from the same torch seed.
        torch.manual_seed(44)
        draws.append(_dropout_draw())
    assert torch.equal(draws[0], draws[1])


@pytest.mark.parametrize("world_size", (2, 8))
def test_independent_rank_masks_recover_world_size_noise_reduction(
    world_size: int,
) -> None:
    """Averaging W independent dropout masks should divide noise variance by W."""

    rng = np.random.default_rng(20260716 + world_size)
    independent_noise = (
        rng.integers(0, 2, size=(200_000, world_size), dtype=np.int8) * 2 - 1
    )
    identical_noise = np.repeat(independent_noise[:, :1], world_size, axis=1)

    independent_variance = np.var(independent_noise.mean(axis=1))
    identical_variance = np.var(identical_noise.mean(axis=1))

    assert identical_variance / independent_variance == pytest.approx(
        world_size, rel=0.02
    )


def test_auxiliary_readout_must_not_advance_shared_dropout_stream() -> None:
    """The auxiliary arm may change gradients, never the common RNG schedule."""

    trunk_dropout = torch.nn.Dropout(0.5)
    auxiliary_readouts = torch.nn.ModuleList(torch.nn.Identity() for _ in range(5))
    values = torch.ones(4096)

    torch.manual_seed(812)
    control_first = trunk_dropout(values)
    control_second = trunk_dropout(values)

    torch.manual_seed(812)
    treatment_first = trunk_dropout(values)
    for auxiliary_readout in auxiliary_readouts:
        auxiliary_readout(values)
    treatment_second = trunk_dropout(values)

    assert torch.equal(control_first, treatment_first)
    assert torch.equal(control_second, treatment_second)


def test_training_rng_flag_is_typed_and_changes_config_hash() -> None:
    baseline = train_bc.TrainConfig()
    offset = train_bc.TrainConfig(training_rng_rank_offset=True)
    assert baseline.training_rng_rank_offset is False
    assert baseline.config_hash() != offset.config_hash()


def test_sampler_seed_decouples_every_numpy_data_trajectory_from_torch_seed() -> None:
    left = _args(seed=11, sampler_seed=424242, enabled=True)
    right = _args(seed=99, sampler_seed=424242, enabled=True)
    assert train_bc._resolved_sampler_seed(left) == 424242
    assert train_bc._resolved_sampler_seed(right) == 424242

    ddp = {"enabled": True, "world_size": 2, "rank": 1, "local_rank": 1}
    left_order = train_bc._epoch_order(
        np.random.default_rng(train_bc._resolved_sampler_seed(left)),
        100,
        8,
        ddp,
        data_sharded=False,
    )
    right_order = train_bc._epoch_order(
        np.random.default_rng(train_bc._resolved_sampler_seed(right)),
        100,
        8,
        ddp,
        data_sharded=False,
    )
    assert np.array_equal(left_order, right_order)

    left_aux = np.random.default_rng(
        np.random.SeedSequence(
            [train_bc._resolved_sampler_seed(left), 0xA17C1E, 3]
        )
    ).integers(0, 1_000_000, size=32)
    right_aux = np.random.default_rng(
        np.random.SeedSequence(
            [train_bc._resolved_sampler_seed(right), 0xA17C1E, 3]
        )
    ).integers(0, 1_000_000, size=32)
    assert np.array_equal(left_aux, right_aux)

    left_symmetry = np.random.default_rng(
        train_bc._resolved_sampler_seed(left) + 20260705
    ).integers(12, size=64)
    right_symmetry = np.random.default_rng(
        train_bc._resolved_sampler_seed(right) + 20260705
    ).integers(12, size=64)
    assert np.array_equal(left_symmetry, right_symmetry)


@pytest.mark.parametrize("world_size", (2, 8))
def test_symmetry_stream_is_global_rank_strided_not_rank_replayed(
    world_size: int,
) -> None:
    seed = 20260705
    local_rows = 64
    rank_generators = [np.random.default_rng(seed) for _ in range(world_size)]
    rank_draws = [
        train_bc._draw_global_rank_strided_symmetry_ids(  # noqa: SLF001
            rank_generators[rank],
            n_symmetries=12,
            local_rows=local_rows,
            rank=rank,
            world_size=world_size,
        )
        for rank in range(world_size)
    ]

    # Reassemble the logical global batch in the same rank-strided topology as
    # _epoch_order. It must be exactly the one-process stream, not W replays of
    # its first local_rows entries.
    reconstructed = np.empty(local_rows * world_size, dtype=np.int64)
    for rank, draws in enumerate(rank_draws):
        reconstructed[rank::world_size] = draws
    expected = np.random.default_rng(seed).integers(
        12, size=local_rows * world_size
    )
    assert np.array_equal(reconstructed, expected)
    assert not np.array_equal(rank_draws[0], rank_draws[1])

    # Every rank advanced through the same complete global batch, retaining the
    # existing one-shared-state checkpoint invariant.
    states = [generator.bit_generator.state for generator in rank_generators]
    assert train_bc._require_shared_symmetry_rng_state(states) == states[0]  # noqa: SLF001


def test_rank_strided_symmetry_resume_replays_exact_continuation() -> None:
    seed = 7813
    world_size = 4
    generators = [np.random.default_rng(seed) for _ in range(world_size)]
    for rank, generator in enumerate(generators):
        train_bc._draw_global_rank_strided_symmetry_ids(  # noqa: SLF001
            generator,
            n_symmetries=12,
            local_rows=37,
            rank=rank,
            world_size=world_size,
        )
    checkpoint_state = train_bc._require_shared_symmetry_rng_state(  # noqa: SLF001
        [generator.bit_generator.state for generator in generators]
    )
    assert checkpoint_state is not None

    uninterrupted = [
        train_bc._draw_global_rank_strided_symmetry_ids(  # noqa: SLF001
            generator,
            n_symmetries=12,
            local_rows=19,
            rank=rank,
            world_size=world_size,
        )
        for rank, generator in enumerate(generators)
    ]
    resumed_generators = [np.random.default_rng() for _ in range(world_size)]
    for generator in resumed_generators:
        generator.bit_generator.state = copy.deepcopy(checkpoint_state)
    resumed = [
        train_bc._draw_global_rank_strided_symmetry_ids(  # noqa: SLF001
            generator,
            n_symmetries=12,
            local_rows=19,
            rank=rank,
            world_size=world_size,
        )
        for rank, generator in enumerate(resumed_generators)
    ]
    assert all(
        np.array_equal(before, after)
        for before, after in zip(uninterrupted, resumed, strict=True)
    )


def test_single_rank_symmetry_stream_is_byte_for_byte_legacy() -> None:
    seed = 991
    historical = np.random.default_rng(seed)
    topology_correct = np.random.default_rng(seed)
    for local_rows in (0, 1, 64, 17):
        expected = historical.integers(12, size=local_rows)
        actual = train_bc._draw_global_rank_strided_symmetry_ids(  # noqa: SLF001
            topology_correct,
            n_symmetries=12,
            local_rows=local_rows,
            rank=0,
            world_size=1,
        )
        assert np.array_equal(actual, expected)
    assert topology_correct.bit_generator.state == historical.bit_generator.state


def test_shared_symmetry_rng_state_requires_exact_rank_alignment() -> None:
    rng = np.random.default_rng(20260705)
    rng.integers(12, size=37)
    state = rng.bit_generator.state

    shared = train_bc._require_shared_symmetry_rng_state(  # noqa: SLF001
        [copy.deepcopy(state) for _ in range(8)]
    )
    assert shared == state
    assert train_bc._require_shared_symmetry_rng_state(  # noqa: SLF001
        [None for _ in range(8)]
    ) is None

    drifted = copy.deepcopy(state)
    drifted["state"]["state"] += 1
    with pytest.raises(RuntimeError, match="streams diverged"):
        train_bc._require_shared_symmetry_rng_state(  # noqa: SLF001
            [copy.deepcopy(state), drifted]
        )
    with pytest.raises(RuntimeError, match="only part of the ranks"):
        train_bc._require_shared_symmetry_rng_state(  # noqa: SLF001
            [copy.deepcopy(state), None]
        )


def test_resume_identity_binds_explicit_sampler_and_torch_seeds_independently() -> None:
    ddp = {
        "enabled": True,
        "world_size": 8,
        "rank": 0,
        "local_rank": 0,
    }

    def identity(seed: int, sampler_seed: int | None):
        args = argparse.Namespace(
            seed=seed,
            sampler_seed=sampler_seed,
            grad_accum_steps=1,
            ddp_shard_data=False,
            fsdp=False,
            policy_aux_active_batch_size=0,
        )
        return train_bc._training_resume_recipe_identity(
            train_bc.TrainConfig(seed=seed), args, ddp
        )

    legacy = identity(7, None)
    assert "sampler_seed" not in legacy
    explicit = identity(7, 424242)
    assert explicit["model_and_torch_seed"] == 7
    assert explicit["sampler_seed"] == 424242
    assert explicit != identity(8, 424242)
    assert explicit != identity(7, 424243)
