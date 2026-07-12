from __future__ import annotations

import numpy as np

from tools import train_bc
from tools.mixed_memmap_corpus import _ConcatColumn


def _reference_per_game_weight_quality(
    seeds: np.ndarray, weights: np.ndarray
) -> dict[str, object]:
    unique, inverse, counts = np.unique(
        seeds, return_inverse=True, return_counts=True
    )
    totals = np.zeros(len(unique), dtype=np.float64)
    np.add.at(totals, inverse, np.asarray(weights, dtype=np.float64))
    return {
        "n_games": int(len(unique)),
        "rows_per_game": {
            "min": int(counts.min()),
            "max": int(counts.max()),
            "mean": float(counts.mean()),
        },
        "total_weight_per_game": {
            "min": float(totals.min()),
            "max": float(totals.max()),
            "mean": float(totals.mean()),
            "std": float(totals.std()),
        },
    }


def _assert_report_equal(actual: dict, expected: dict) -> None:
    assert actual["n_games"] == expected["n_games"]
    for section in ("rows_per_game", "total_weight_per_game"):
        assert actual[section].keys() == expected[section].keys()
        for key, value in actual[section].items():
            assert value == expected[section][key]


def test_per_game_weight_quality_contiguous_fast_path_matches_reference() -> None:
    # Production layout: games are contiguous but neither seed nor row count is
    # sorted. Include mixed signs/fractions so this checks the float64 totals,
    # not only the integer row-count summary.
    seeds = np.repeat(np.asarray([90, 4, 18, 7]), [3, 1, 4, 2])
    weights = np.asarray(
        [0.25, 1.5, -0.5, 9.0, 1.25, 2.5, 3.75, -1.0, 0.1, 0.2],
        dtype=np.float32,
    )
    data = {"action_taken": np.zeros(len(seeds)), "game_seed": seeds}
    _assert_report_equal(
        train_bc.per_game_weight_quality(data, weights),
        _reference_per_game_weight_quality(seeds, weights),
    )

def test_per_game_weight_quality_repeated_run_falls_back_exactly() -> None:
    # A seed appearing in two disjoint runs violates the production layout but
    # remains supported; it must take the general factorisation and coalesce.
    seeds = np.asarray([5, 5, 9, 9, 5, 12, 12], dtype=np.int64)
    weights = np.asarray([1.0, 2.0, 0.25, 0.5, 4.0, 8.0, 16.0], dtype=np.float32)
    data = {"action_taken": np.zeros(len(seeds)), "game_seed": seeds}
    _assert_report_equal(
        train_bc.per_game_weight_quality(data, weights),
        _reference_per_game_weight_quality(seeds, weights),
    )


def test_concat_categorical_weight_report_avoids_global_decode(tmp_path) -> None:
    def column(name: str, codes: list[int], categories: list[str]):
        path = tmp_path / f"{name}.dat"
        np.asarray(codes, dtype=np.int32).tofile(path)
        return train_bc._MemmapCategoricalColumn(  # noqa: SLF001
            np.memmap(path, dtype=np.int32, mode="r", shape=(len(codes),)),
            np.asarray(categories),
        )

    # Deliberately use different local category-code orders. The composite
    # reduction must merge by decoded label, not by integer code.
    left = column("left", [0, 1, 0], ["n128", "n256"])
    right = column("right", [1, 0, 1, 1], ["replay", "n128"])
    concat = _ConcatColumn((left, right), (3, 4))
    weights = np.asarray([1, 2, 3, 4, 5, 6, 7], dtype=np.float32)
    data = {"action_taken": np.zeros(7), "teacher_name": concat}

    report = train_bc._weight_by_field(  # noqa: SLF001
        data, weights, "teacher_name"
    )
    assert report == {
        "n128": {"raw_samples": 5, "weight_sum": 21.0, "mean_weight": 4.2},
        "n256": {"raw_samples": 1, "weight_sum": 2.0, "mean_weight": 2.0},
        "replay": {"raw_samples": 1, "weight_sum": 5.0, "mean_weight": 5.0},
    }
