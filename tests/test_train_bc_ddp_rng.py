from __future__ import annotations

import argparse

import numpy as np
import torch

from tools import train_bc


def _args(*, seed: int, enabled: bool) -> argparse.Namespace:
    return argparse.Namespace(seed=seed, training_rng_rank_offset=enabled)


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


def test_auxiliary_dropout_advances_shared_stream_before_next_trunk_batch() -> None:
    trunk_dropout = torch.nn.Dropout(0.5)
    auxiliary_dropouts = torch.nn.ModuleList(torch.nn.Dropout(0.5) for _ in range(5))
    values = torch.ones(4096)

    torch.manual_seed(812)
    control_first = trunk_dropout(values)
    control_second = trunk_dropout(values)

    torch.manual_seed(812)
    treatment_first = trunk_dropout(values)
    for auxiliary_dropout in auxiliary_dropouts:
        auxiliary_dropout(values)
    treatment_second = trunk_dropout(values)

    # The first shared-trunk call is common-random, but the five CAT-100-style
    # head dropouts consume the process-global stream and change the next batch.
    assert torch.equal(control_first, treatment_first)
    assert not torch.equal(control_second, treatment_second)


def test_training_rng_flag_is_typed_and_changes_config_hash() -> None:
    baseline = train_bc.TrainConfig()
    offset = train_bc.TrainConfig(training_rng_rank_offset=True)
    assert baseline.training_rng_rank_offset is False
    assert baseline.config_hash() != offset.config_hash()
