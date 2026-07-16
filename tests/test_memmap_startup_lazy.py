from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from catan_zero.rl.action_mask import ActionCatalog
from tools import train_bc
from tools.mixed_memmap_corpus import _ConcatColumn


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


def test_value_targets_decode_independent_winner_and_player_codebooks(
    tmp_path: Path,
) -> None:
    """Column-local category codes are not comparable across columns.

    Production corpora are allowed to encode the same labels in different
    category orders.  In particular, ``winner`` may encode RED as 0 while
    ``player`` encodes BLUE as 0.  Comparing those raw codes would invert every
    outcome; the learner must compare the independently decoded strings.
    """
    winner_path = tmp_path / "winner.codes.dat"
    player_path = tmp_path / "player.codes.dat"
    np.asarray([0, 1], dtype=np.int32).tofile(winner_path)  # RED, BLUE
    np.asarray([1, 0], dtype=np.int32).tofile(player_path)  # RED, BLUE

    winner = train_bc._MemmapCategoricalColumn(  # noqa: SLF001
        np.memmap(winner_path, dtype=np.int32, mode="r", shape=(2,)),
        np.asarray(["RED", "BLUE"], dtype=str),
    )
    player = train_bc._MemmapCategoricalColumn(  # noqa: SLF001
        np.memmap(player_path, dtype=np.int32, mode="r", shape=(2,)),
        np.asarray(["BLUE", "RED"], dtype=str),
    )

    outcome, _vp, has_outcome, _has_vp, *_ = train_bc._value_targets(  # noqa: SLF001
        {
            "winner": winner,
            "player": player,
            "truncated": np.zeros(2, dtype=np.bool_),
        },
        np.arange(2, dtype=np.int64),
        "cpu",
        vps_to_win=10,
    )

    assert np.asarray(winner).tolist() == ["RED", "BLUE"]
    assert np.asarray(player).tolist() == ["RED", "BLUE"]
    assert outcome.tolist() == [1.0, 1.0]
    assert has_outcome.tolist() == [True, True]


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


def test_forced_mass_report_reads_only_selected_forced_ragged_rows(
    tmp_path: Path, monkeypatch,
) -> None:
    catalog = ActionCatalog(("RED", "BLUE"))
    roll = next(
        index
        for index in range(catalog.size)
        if catalog.describe(index)["action_type"] == "ROLL"
    )
    build_road = next(
        index
        for index in range(catalog.size)
        if catalog.describe(index)["action_type"] == "BUILD_ROAD"
    )
    flat_path = tmp_path / "legal-report.dat"
    np.asarray([roll, build_road, roll, roll], dtype=np.int16).tofile(flat_path)
    flat = np.memmap(flat_path, dtype=np.int16, mode="r", shape=(4,))
    column = train_bc._MemmapRaggedColumn(  # noqa: SLF001
        flat,
        np.asarray([0, 1, 2, 4], dtype=np.int64),
        2,
        -1,
        np.int16,
        None,
    )
    reconstructed: list[np.ndarray | None] = []
    original = column._reconstruct  # noqa: SLF001

    def observe(indices):
        reconstructed.append(indices)
        return original(indices)

    monkeypatch.setattr(column, "_reconstruct", observe)
    report = train_bc.forced_action_type_value_mass_quality(
        {
            "action_taken": np.asarray([roll, build_road, roll], dtype=np.int16),
            "legal_action_ids": column,
        },
        np.ones(3, dtype=np.float32),
        row_indices=np.asarray([0, 2], dtype=np.int64),
        objective_measure="test_nominal_measure",
        action_catalog=catalog,
        configured_weights={"ROLL": 1.0},
    )

    assert report["forced_rows"] == 1
    assert len(reconstructed) == 1
    assert reconstructed[0].tolist() == [0]


def test_forced_mass_report_routes_compact_counts_through_composite(
    tmp_path: Path, monkeypatch,
) -> None:
    catalog = ActionCatalog(("RED", "BLUE"))
    roll = next(
        index
        for index in range(catalog.size)
        if catalog.describe(index)["action_type"] == "ROLL"
    )
    build_road = next(
        index
        for index in range(catalog.size)
        if catalog.describe(index)["action_type"] == "BUILD_ROAD"
    )
    columns = []
    observations: list[list[np.ndarray | None]] = []
    for part, (legal, offsets) in enumerate(
        (
            ([roll, build_road, roll], [0, 1, 3]),
            ([build_road, roll, roll], [0, 2, 3]),
        )
    ):
        path = tmp_path / f"legal-composite-{part}.dat"
        np.asarray(legal, dtype=np.int16).tofile(path)
        flat = np.memmap(path, dtype=np.int16, mode="r", shape=(3,))
        column = train_bc._MemmapRaggedColumn(  # noqa: SLF001
            flat,
            np.asarray(offsets, dtype=np.int64),
            2,
            -1,
            np.int16,
            None,
        )
        observed: list[np.ndarray | None] = []
        original = column._reconstruct  # noqa: SLF001

        def observe(indices, *, _observed=observed, _original=original):
            _observed.append(indices)
            return _original(indices)

        monkeypatch.setattr(column, "_reconstruct", observe)
        columns.append(column)
        observations.append(observed)
    composite = _ConcatColumn(columns, (2, 2))

    report = train_bc.forced_action_type_value_mass_quality(
        {
            "action_taken": np.asarray(
                [roll, build_road, build_road, roll], dtype=np.int16
            ),
            "legal_action_ids": composite,
        },
        np.ones(2, dtype=np.float32),
        row_indices=np.asarray([0, 3], dtype=np.int64),
        weights_aligned_to_rows=True,
        objective_measure="test_composite_nominal_measure",
        action_catalog=catalog,
        configured_weights={"ROLL": 1.0},
    )

    assert report["forced_rows"] == 2
    assert [entry[0].tolist() for entry in observations] == [[0], [1]]
