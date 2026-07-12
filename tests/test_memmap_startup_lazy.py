from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tools import train_bc


def test_categorical_memmap_decodes_only_requested_rows(tmp_path: Path) -> None:
    path = tmp_path / "codes.dat"
    np.asarray([0, 1, 1, 2, 0], dtype=np.int32).tofile(path)
    codes = np.memmap(path, dtype=np.int32, mode="r", shape=(5,))
    column = train_bc._MemmapCategoricalColumn(  # noqa: SLF001
        codes, np.asarray(["", "teacher", "other"], dtype=str)
    )

    assert column[1] == "teacher"
    assert column[[4, 2]].tolist() == ["", "teacher"]
    assert np.asarray(column).tolist() == ["", "teacher", "teacher", "other", ""]
    assert column.present_values() == {"", "teacher", "other"}
    grouped = column.grouped_weights(np.ones(5, dtype=np.float32), limit=3)
    assert grouped[""]["raw_samples"] == 2
    assert grouped["teacher"]["weight_sum"] == 2.0


def test_production_ragged_policy_columns_are_batch_lazy() -> None:
    assert {
        "legal_action_ids",
        "prior_policy",
        "target_policy",
        "target_policy_mask",
        "target_scores",
        "target_scores_mask",
    } <= train_bc.MEMMAP_LAZY_COLUMNS


def test_default_sample_weights_do_not_decode_inert_winner_labels() -> None:
    class RefuseDecode:
        def __array__(self, *_args, **_kwargs):
            raise AssertionError("inert winner/player column was decoded")

    data = {
        "action_taken": np.asarray([1, 2, 3], dtype=np.int16),
        "legal_action_ids": np.asarray([[1, -1], [2, 4], [3, -1]], dtype=np.int16),
        "policy_weight_multiplier": np.ones(3, dtype=np.float32),
        "winner": RefuseDecode(),
        "player": RefuseDecode(),
    }
    weights = train_bc.build_sample_weights(
        data,
        teacher_weights={},
        phase_weights={},
        forced_action_weight=0.1,
        winner_sample_weight=1.0,
        loser_sample_weight=1.0,
        vp_margin_weight=0.0,
        vps_to_win=10,
    )
    assert weights.tolist() == [0.25, 2.5, 0.25]


def test_forced_value_weights_use_ragged_offsets_without_padding_reconstruction(
    tmp_path: Path, monkeypatch,
) -> None:
    flat_path = tmp_path / "legal.dat"
    np.asarray([1, 2, 3, 4], dtype=np.int16).tofile(flat_path)
    flat = np.memmap(flat_path, dtype=np.int16, mode="r", shape=(4,))
    column = train_bc._MemmapRaggedColumn(  # noqa: SLF001
        flat,
        np.asarray([0, 1, 4], dtype=np.int64),
        3,
        -1,
        np.int16,
        None,
    )
    monkeypatch.setattr(
        column,
        "_reconstruct",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("full padded legal matrix was reconstructed")
        ),
    )
    weights = train_bc.build_value_sample_weights(
        {
            "action_taken": np.asarray([1, 2], dtype=np.int16),
            "legal_action_ids": column,
        },
        forced_row_value_weight=0.1,
    )
    assert weights.tolist() == pytest.approx([2.0 / 11.0, 20.0 / 11.0])
