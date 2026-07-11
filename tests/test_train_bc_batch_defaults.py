from __future__ import annotations

import numpy as np
import torch

from tools import train_bc


class _IndexedWithoutLen:
    def __init__(self, values) -> None:
        self.values = np.asarray(values)

    def __len__(self) -> int:
        raise AssertionError("full-corpus length must not be read")

    def __getitem__(self, index):
        return self.values[index]


def test_batch_array_or_fill_only_indexes_present_column() -> None:
    batch = np.asarray([3, 1], dtype=np.int64)
    result = train_bc._batch_array_or_fill(  # noqa: SLF001
        {"phase": _IndexedWithoutLen(["a", "b", "c", "d"])},
        "phase",
        batch,
        "unknown",
    )

    np.testing.assert_array_equal(result, np.asarray(["d", "b"]))


def test_batch_array_or_fill_sizes_missing_default_from_batch() -> None:
    batch = np.asarray([10_000_000, 20_000_000], dtype=np.int64)
    result = train_bc._batch_array_or_fill(  # noqa: SLF001
        {"action_taken": _IndexedWithoutLen([0])},
        "phase",
        batch,
        "unknown",
    )

    np.testing.assert_array_equal(result, np.asarray(["unknown", "unknown"]))


def test_q_skip_rows_preserves_present_and_missing_column_semantics() -> None:
    batch = np.asarray([2, 0], dtype=np.int64)
    present = {
        "action_taken": _IndexedWithoutLen([0]),
        "teacher_name": _IndexedWithoutLen(["ab_old", "other", "ab_new"]),
        "target_score_source": _IndexedWithoutLen(["", "policy", "ab_root"]),
    }

    np.testing.assert_array_equal(
        train_bc._q_skip_rows(present, batch, ("ab_",)),  # noqa: SLF001
        np.asarray([False, True]),
    )
    np.testing.assert_array_equal(
        train_bc._q_skip_rows(  # noqa: SLF001
            {"action_taken": _IndexedWithoutLen([0])},
            batch,
            ("ab_",),
        ),
        np.asarray([False, False]),
    )


def test_soft_target_array_only_indexes_optional_batch_columns() -> None:
    batch = np.asarray([2, 0], dtype=np.int64)
    core = {
        "action_taken": _IndexedWithoutLen([0]),
        "legal_action_ids": np.asarray([[0, 1], [0, 1], [0, 1]]),
        "target_policy": np.asarray([[1.0, 0.0], [0.5, 0.5], [1.0, 3.0]]),
    }
    present = {
        **core,
        "teacher_name": _IndexedWithoutLen(["a", "b", "c"]),
        "target_score_source": _IndexedWithoutLen(["policy", "policy", "policy"]),
    }

    expected = np.asarray([[0.25, 0.75], [1.0, 0.0]], dtype=np.float32)
    for data in (present, core):
        target, support = train_bc._soft_target_array(  # noqa: SLF001
            data,
            batch,
            0.7,
            "policy",
        )
        np.testing.assert_allclose(target, expected)
        np.testing.assert_array_equal(support, expected > 0.0)


def test_value_targets_only_indexes_optional_batch_columns() -> None:
    batch = np.asarray([2, 0], dtype=np.int64)
    data = {
        "action_taken": _IndexedWithoutLen([0]),
        "winner": _IndexedWithoutLen(["red", "", "blue"]),
        "player": _IndexedWithoutLen(["red", "red", "red"]),
        "truncated": _IndexedWithoutLen([False, False, True]),
        "seat": _IndexedWithoutLen([0, 0, 1]),
        "final_actual_vps": _IndexedWithoutLen([[10, 4], [0, 0], [5, 7]]),
        "has_final_actual_vps": _IndexedWithoutLen([True, False, True]),
        "final_public_vps": _IndexedWithoutLen([[9, 4], [0, 0], [5, 6]]),
        "has_final_public_vps": _IndexedWithoutLen([True, False, True]),
    }

    outcome, vp, has_outcome, has_vp, *_ = train_bc._value_targets(  # noqa: SLF001
        data,
        batch,
        torch.device("cpu"),
        vps_to_win=10,
    )

    np.testing.assert_array_equal(outcome.numpy(), np.asarray([0.0, 1.0]))
    np.testing.assert_array_equal(has_outcome.numpy(), np.asarray([False, True]))
    np.testing.assert_allclose(vp.numpy(), np.asarray([0.0, 1.0]))
    np.testing.assert_array_equal(has_vp.numpy(), np.asarray([False, True]))
