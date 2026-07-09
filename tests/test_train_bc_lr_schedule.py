from __future__ import annotations

import pytest

from tools.train_bc import _apply_lr_schedule, _lr_schedule_multiplier


# --------------------------------------------------------------------------- flat (default, no-op)


def test_flat_schedule_matches_warmup_then_hold_exactly() -> None:
    """AUDIT FIX (LR decay) backward compatibility: schedule='flat' must reproduce the
    pre-fix warmup-then-hold curve bit-for-bit, at every point (during warmup, at the
    warmup boundary, and long after warmup) regardless of total_steps."""
    warmup_steps = 10
    for step in (0, 4, 9, 10, 20, 1000):
        assert _lr_schedule_multiplier(
            step, warmup_steps=warmup_steps, total_steps=50, schedule="flat"
        ) == pytest.approx(min(1.0, float(step + 1) / float(warmup_steps)))


def test_flat_schedule_ignores_total_steps() -> None:
    for total_steps in (1, 10, 1_000_000):
        assert _lr_schedule_multiplier(
            20, warmup_steps=10, total_steps=total_steps, schedule="flat"
        ) == pytest.approx(1.0)


# --------------------------------------------------------------------------- cosine


def test_cosine_schedule_ramps_during_warmup_same_as_flat() -> None:
    for step in (0, 4, 9):
        flat = _lr_schedule_multiplier(step, warmup_steps=10, total_steps=100, schedule="flat")
        cosine = _lr_schedule_multiplier(step, warmup_steps=10, total_steps=100, schedule="cosine")
        assert cosine == pytest.approx(flat)


def test_cosine_schedule_starts_at_one_right_after_warmup() -> None:
    assert _lr_schedule_multiplier(
        10, warmup_steps=10, total_steps=110, schedule="cosine"
    ) == pytest.approx(1.0)


def test_cosine_schedule_decays_to_zero_at_total_steps() -> None:
    """progress = (step - warmup_steps) / (total_steps - warmup_steps) reaches exactly
    1.0 (multiplier 0.0) at step == total_steps -- one step past the last step that
    actually runs in training (steps are 0-indexed 0..total_steps-1). The last REAL
    training step (total_steps - 1) is therefore just short of zero, not exactly it."""
    assert _lr_schedule_multiplier(
        110, warmup_steps=10, total_steps=110, schedule="cosine"
    ) == pytest.approx(0.0, abs=1e-9)
    near_zero = _lr_schedule_multiplier(
        109, warmup_steps=10, total_steps=110, schedule="cosine"
    )
    assert 0.0 < near_zero < 0.001


def test_cosine_schedule_is_monotonically_non_increasing_after_warmup() -> None:
    warmup_steps, total_steps = 10, 110
    previous = float("inf")
    for step in range(warmup_steps, total_steps):
        multiplier = _lr_schedule_multiplier(
            step, warmup_steps=warmup_steps, total_steps=total_steps, schedule="cosine"
        )
        assert multiplier <= previous + 1e-9
        previous = multiplier


# --------------------------------------------------------------------------- linear


def test_linear_schedule_decays_linearly_to_zero() -> None:
    warmup_steps, total_steps = 0, 100
    assert _lr_schedule_multiplier(
        0, warmup_steps=warmup_steps, total_steps=total_steps, schedule="linear"
    ) == pytest.approx(1.0)
    assert _lr_schedule_multiplier(
        50, warmup_steps=warmup_steps, total_steps=total_steps, schedule="linear"
    ) == pytest.approx(0.5)
    assert _lr_schedule_multiplier(
        99, warmup_steps=warmup_steps, total_steps=total_steps, schedule="linear"
    ) == pytest.approx(0.01, abs=1e-6)


def test_linear_schedule_never_goes_negative_past_total_steps() -> None:
    assert _lr_schedule_multiplier(
        500, warmup_steps=0, total_steps=100, schedule="linear"
    ) == pytest.approx(0.0)


# --------------------------------------------------------------------------- error handling


def test_unknown_schedule_raises() -> None:
    with pytest.raises(SystemExit, match="lr-schedule"):
        _lr_schedule_multiplier(0, warmup_steps=0, total_steps=10, schedule="bogus")


# --------------------------------------------------------------------------- _apply_lr_schedule


def test_apply_lr_schedule_sets_every_param_group_flat() -> None:
    import torch

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0)

    multiplier = _apply_lr_schedule(
        optimizer,
        base_lr=2e-4,
        step=0,
        warmup_steps=4,
        total_steps=100,
        schedule="flat",
    )

    assert multiplier == pytest.approx(0.25)
    for group in optimizer.param_groups:
        assert group["lr"] == pytest.approx(2e-4 * 0.25)


def test_apply_lr_schedule_cosine_reduces_lr_near_the_end() -> None:
    import torch

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0)

    _apply_lr_schedule(
        optimizer,
        base_lr=2e-4,
        step=99,
        warmup_steps=0,
        total_steps=100,
        schedule="cosine",
    )
    for group in optimizer.param_groups:
        assert group["lr"] < 2e-4 * 0.01
