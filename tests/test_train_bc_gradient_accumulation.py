from __future__ import annotations

from contextlib import contextmanager

import pytest

from tools.train_bc import _accumulation_group_size, _gradient_sync_context


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
