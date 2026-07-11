from __future__ import annotations

import numpy as np

from tools import train_bc


class _LenMustNotBeRead:
    def __len__(self) -> int:
        raise AssertionError("full-corpus fallback was evaluated eagerly")


def _inputs():
    batch = np.asarray([1, 3], dtype=np.int64)
    predictions = np.asarray([0, 1], dtype=np.int64)
    targets = np.asarray([0, 2], dtype=np.int64)
    logits = np.asarray([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    return batch, predictions, targets, logits


def test_field_stats_does_not_evaluate_missing_default_when_field_exists() -> None:
    batch, predictions, targets, logits = _inputs()
    data = {
        "action_taken": _LenMustNotBeRead(),
        "phase": np.asarray(["a", "build", "b", "trade"]),
    }

    stats = train_bc._field_stats(  # noqa: SLF001
        data,
        batch,
        predictions,
        targets,
        logits,
        field="phase",
    )

    assert stats["build"] == {"count": 1, "top1": 1, "top3": 1}
    assert stats["trade"] == {"count": 1, "top1": 0, "top3": 1}


def test_field_stats_missing_field_keeps_unknown_semantics_per_batch() -> None:
    batch, predictions, targets, logits = _inputs()
    stats = train_bc._field_stats(  # noqa: SLF001
        {"action_taken": np.zeros(4, dtype=np.int64)},
        batch,
        predictions,
        targets,
        logits,
        field="phase",
    )

    assert stats == {"unknown": {"count": 2, "top1": 1, "top3": 2}}
