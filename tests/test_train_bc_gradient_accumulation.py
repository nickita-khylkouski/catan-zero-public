from __future__ import annotations

from contextlib import contextmanager

import pytest

from tools.train_bc import (
    _accumulation_group_size,
    _advance_global_step,
    _effective_training_epoch_limit,
    _gradient_sync_context,
    _validate_exact_optimizer_step_dose,
    _validate_resumed_max_step_boundary,
)


class _SyncRecorder:
    def __init__(self) -> None:
        self.entered = 0
        self.exited = 0

    @contextmanager
    def no_sync(self):
        self.entered += 1
        try:
            yield
        finally:
            self.exited += 1


def test_nonstepping_microbatch_suppresses_sync_around_caller_body() -> None:
    model = _SyncRecorder()
    inside = False
    with _gradient_sync_context(model, accum_do_step=False):
        inside = model.entered == 1 and model.exited == 0
    assert inside
    assert (model.entered, model.exited) == (1, 1)


def test_stepping_microbatch_keeps_sync_enabled() -> None:
    model = _SyncRecorder()
    with _gradient_sync_context(model, accum_do_step=True):
        pass
    assert (model.entered, model.exited) == (0, 0)


@pytest.mark.parametrize(
    ("configured", "batch", "total", "expected"),
    [
        (4, 1, 8, 4),
        (4, 5, 8, 4),
        (4, 9, 10, 2),
        (4, 10, 10, 1),
        (1, 7, 10, 1),
    ],
)
def test_accumulation_group_size_uses_short_tail_divisor(
    configured: int, batch: int, total: int, expected: int
) -> None:
    assert (
        _accumulation_group_size(
            configured_size=configured,
            batch_number=batch,
            total_batches=total,
        )
        == expected
    )


def test_accumulation_group_size_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="invalid gradient accumulation"):
        _accumulation_group_size(configured_size=4, batch_number=11, total_batches=10)


def test_skipped_zero_objective_group_does_not_consume_max_step_budget() -> None:
    global_step = 4
    max_steps = 5

    global_step = _advance_global_step(
        global_step,
        accum_do_step=True,
        optimizer_step_applied=False,
    )
    assert global_step == 4
    assert global_step < max_steps

    global_step = _advance_global_step(
        global_step,
        accum_do_step=True,
        optimizer_step_applied=True,
    )
    assert global_step == max_steps
    assert global_step >= max_steps


def test_exact_step_dose_extends_epoch_ceiling_without_changing_legacy_mode() -> None:
    assert _effective_training_epoch_limit(
        configured_epochs=2, max_steps=128, exact_max_steps=False
    ) == 2
    assert _effective_training_epoch_limit(
        configured_epochs=2, max_steps=128, exact_max_steps=True
    ) == 128


def test_exact_step_dose_requires_a_positive_step_target() -> None:
    with pytest.raises(SystemExit, match="requires --max-steps > 0"):
        _effective_training_epoch_limit(
            configured_epochs=2, max_steps=0, exact_max_steps=True
        )


def test_exact_step_dose_fails_closed_when_no_batch_applies_an_update() -> None:
    with pytest.raises(RuntimeError, match="requested=128 applied=0"):
        _validate_exact_optimizer_step_dose(
            exact_max_steps=True,
            max_steps=128,
            applied_steps=0,
            epoch_limit=128,
        )


def test_resume_at_step_cap_refuses_as_an_exact_no_op() -> None:
    with pytest.raises(
        SystemExit,
        match=(
            "already equals --max-steps.*"
            "no optimizer update or checkpoint save will be performed"
        ),
    ):
        _validate_resumed_max_step_boundary(
            resumed=True,
            global_step=32,
            max_steps=32,
        )


def test_resume_beyond_step_cap_is_invalid() -> None:
    with pytest.raises(
        SystemExit,
        match="exceeds --max-steps.*refusing to sample, update, or save",
    ):
        _validate_resumed_max_step_boundary(
            resumed=True,
            global_step=33,
            max_steps=32,
        )


@pytest.mark.parametrize(
    ("resumed", "global_step", "max_steps"),
    [
        (True, 31, 32),
        (True, 128, 0),
        (False, 32, 32),
    ],
)
def test_resume_step_cap_guard_allows_only_legal_or_uncapped_states(
    resumed: bool, global_step: int, max_steps: int
) -> None:
    _validate_resumed_max_step_boundary(
        resumed=resumed,
        global_step=global_step,
        max_steps=max_steps,
    )


def test_resume_step_cap_guard_precedes_uniform_and_weighted_sampler_paths() -> None:
    import inspect

    from tools import train_bc

    source = inspect.getsource(train_bc.main)
    restore = source.index("_restore_training_progress_state(")
    guard = source.index("_validate_resumed_max_step_boundary(")
    epoch_loop = source.index("for epoch in range(")
    order = source.index("order = _epoch_order(")
    terminal_save = source.index("_save_policy(", guard)

    assert restore < guard < epoch_loop < order < terminal_save


def test_nonstepping_microbatch_cannot_claim_an_optimizer_update() -> None:
    with pytest.raises(ValueError, match="non-stepping microbatch"):
        _advance_global_step(
            7,
            accum_do_step=False,
            optimizer_step_applied=True,
        )
