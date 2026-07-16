from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest

from tools.train_bc import (
    _accumulation_group_size,
    _advance_global_step,
    _effective_training_epoch_limit,
    _gradient_sync_context,
    _require_optimizer_resume_sidecars,
    _validate_optimizer_resume_request,
    _validate_exact_optimizer_step_dose,
    _validate_resumed_epoch_boundary,
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
    ("optimizer_present", "progress_present"),
    [(False, False), (True, False), (False, True)],
)
def test_explicit_optimizer_resume_requires_both_regular_sidecars(
    tmp_path: Path,
    optimizer_present: bool,
    progress_present: bool,
) -> None:
    optimizer_path = tmp_path / "model.pt.optimizer.pt"
    progress_path = tmp_path / "model.pt.training-progress.json"
    if optimizer_present:
        optimizer_path.write_bytes(b"optimizer")
    if progress_present:
        progress_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="incomplete resumable checkpoint set"):
        _require_optimizer_resume_sidecars(
            optimizer_path=optimizer_path,
            progress_path=progress_path,
        )


def test_explicit_optimizer_resume_accepts_complete_sidecar_set(
    tmp_path: Path,
) -> None:
    optimizer_path = tmp_path / "model.pt.optimizer.pt"
    progress_path = tmp_path / "model.pt.training-progress.json"
    optimizer_path.write_bytes(b"optimizer")
    progress_path.write_text("{}\n", encoding="utf-8")

    _require_optimizer_resume_sidecars(
        optimizer_path=optimizer_path,
        progress_path=progress_path,
    )


@pytest.mark.parametrize(
    ("init_checkpoint", "optimizer_available", "message"),
    [
        ("", True, "requires --init-checkpoint"),
        ("model.pt", False, "requires a live training optimizer"),
    ],
)
def test_optimizer_resume_request_requires_restorable_context(
    init_checkpoint: str,
    optimizer_available: bool,
    message: str,
) -> None:
    with pytest.raises(SystemExit, match=message):
        _validate_optimizer_resume_request(
            requested=True,
            init_checkpoint=init_checkpoint,
            optimizer_available=optimizer_available,
        )


def test_fresh_optimizer_request_does_not_require_resume_context() -> None:
    _validate_optimizer_resume_request(
        requested=False,
        init_checkpoint="",
        optimizer_available=False,
    )


def test_resume_at_completed_epoch_dose_refuses_no_work_save() -> None:
    with pytest.raises(
        SystemExit,
        match="already reaches --epochs.*no optimizer update or checkpoint save",
    ):
        _validate_resumed_epoch_boundary(
            resumed=True,
            completed_epochs=3,
            epoch_limit=3,
        )


@pytest.mark.parametrize(
    ("resumed", "completed_epochs", "epoch_limit"),
    [(False, 3, 3), (True, 2, 3)],
)
def test_resume_epoch_guard_allows_fresh_or_incomplete_dose(
    resumed: bool,
    completed_epochs: int,
    epoch_limit: int,
) -> None:
    _validate_resumed_epoch_boundary(
        resumed=resumed,
        completed_epochs=completed_epochs,
        epoch_limit=epoch_limit,
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
    step_guard = source.index("_validate_resumed_max_step_boundary(")
    epoch_guard = source.index("_validate_resumed_epoch_boundary(")
    epoch_loop = source.index("for epoch in range(")
    order = source.index("order = _epoch_order(")
    terminal_save = source.index("_save_policy(", epoch_guard)

    assert restore < step_guard < epoch_guard < epoch_loop < order < terminal_save


def test_optimizer_resume_sidecars_are_required_before_restore_attempt() -> None:
    import inspect

    from tools import train_bc

    source = inspect.getsource(train_bc.main)
    explicit_resume = source.index(
        "if optimizer is not None and args.init_checkpoint and bool(args.resume_optimizer)"
    )
    sidecar_guard = source.index("_require_optimizer_resume_sidecars(", explicit_resume)
    progress_load = source.index("resume_progress = load_training_progress(", sidecar_guard)
    optimizer_load = source.index("load_optimizer_state(", progress_load)

    assert explicit_resume < sidecar_guard < progress_load < optimizer_load


def test_epoch_progress_is_resumable_until_terminal_admission_finishes() -> None:
    import inspect

    from tools import train_bc

    source = inspect.getsource(train_bc.main)
    epoch_role = source.index('checkpoint_role="resumable_epoch"')
    exact_dose_admission = source.index(
        "_validate_exact_optimizer_step_dose(", epoch_role
    )
    feature_admission = source.index(
        "feature_signal_admission.verify_observability(", exact_dose_admission
    )
    objective_admission = source.index(
        "feature_signal_admission.verify_objective_interference(",
        feature_admission,
    )
    terminal_role = source.index(
        'checkpoint_role="terminal_admitted"', objective_admission
    )

    assert (
        epoch_role
        < exact_dose_admission
        < feature_admission
        < objective_admission
        < terminal_role
    )


def test_nonstepping_microbatch_cannot_claim_an_optimizer_update() -> None:
    with pytest.raises(ValueError, match="non-stepping microbatch"):
        _advance_global_step(
            7,
            accum_do_step=False,
            optimizer_step_applied=True,
        )
