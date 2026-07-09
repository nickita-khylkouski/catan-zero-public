from __future__ import annotations

import json
import sys
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import measure_pair_correlation as mpc  # type: ignore  # noqa: E402


def _game(
    *,
    game_seed: int,
    orientation: str,
    search_won: bool | None,
    search_color: str = "RED",
    winner: str = "RED",
    decisions: int = 50,
) -> dict:
    return {
        "game_seed": game_seed,
        "orientation": orientation,
        "search_won": search_won,
        "search_color": search_color,
        "winner": winner,
        "decisions": decisions,
    }


def _pair(game_seed: int, first_won: bool, second_won: bool) -> list[dict]:
    return [
        _game(game_seed=game_seed, orientation="candidate_red", search_won=first_won),
        _game(game_seed=game_seed, orientation="candidate_blue", search_won=second_won),
    ]


def test_pearson_correlation_matches_known_synthetic_values():
    # Perfectly anti-correlated: y = 1 - x deterministically.
    assert mpc.pearson_correlation([0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]) == -1.0
    # Perfectly correlated: y == x deterministically.
    assert mpc.pearson_correlation([0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]) == 1.0
    # Balanced 2x2 design (each of WW/WL/LW/LL occurs once) is exactly
    # uncorrelated: cov = mean(xy) - mean(x)*mean(y) = 0.25 - 0.5*0.5 = 0.
    rho = mpc.pearson_correlation([1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0])
    assert rho is not None
    assert abs(rho) < 1e-9
    # Undefined when one side has zero variance.
    assert mpc.pearson_correlation([1.0, 1.0, 1.0], [1.0, 0.0, 1.0]) is None
    # Undefined with fewer than 2 pairs.
    assert mpc.pearson_correlation([1.0], [1.0]) is None


def test_build_pairs_recovers_negative_correlation_split_pairs():
    # Every pair splits (candidate wins exactly one of the two colors), and
    # the winning color alternates seed to seed -- a known anti-correlated
    # synthetic dataset (fishtest issue #348's claimed direction).
    games = (
        _pair(1, True, False)
        + _pair(2, False, True)
        + _pair(3, True, False)
        + _pair(4, False, True)
    )
    pairs, diagnostics = mpc.build_pairs(games)
    assert len(pairs) == 4
    assert diagnostics["split_pairs"] == 4
    assert diagnostics["ww_pairs"] == 0
    assert diagnostics["ll_pairs"] == 0
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    rho = mpc.pearson_correlation(xs, ys)
    assert rho == -1.0


def test_build_pairs_recovers_positive_correlation_ww_ll_only():
    # Every pair is fully concordant (WW or LL) -- game 2 always matches
    # game 1 -- a known perfectly-positively-correlated synthetic dataset.
    games = (
        _pair(10, True, True)
        + _pair(11, False, False)
        + _pair(12, True, True)
        + _pair(13, False, False)
    )
    pairs, diagnostics = mpc.build_pairs(games)
    assert diagnostics["ww_pairs"] == 2
    assert diagnostics["ll_pairs"] == 2
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    rho = mpc.pearson_correlation(xs, ys)
    assert rho == 1.0


def test_build_pairs_recovers_zero_correlation_balanced_design():
    # WW, split(RED win), split(BLUE win), LL -- one of each -- is the
    # balanced 2x2 design with exactly zero correlation (see the pure-math
    # test above).
    games = _pair(20, True, True) + _pair(21, True, False) + _pair(22, False, True) + _pair(23, False, False)
    pairs, _ = mpc.build_pairs(games)
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    rho = mpc.pearson_correlation(xs, ys)
    assert rho is not None
    assert abs(rho) < 1e-9


def test_build_pairs_excludes_incomplete_and_dedupes_replicas():
    games = (
        _pair(30, True, False)
        # duplicate-seed replica of pair 30 (bit-identical fleet re-run):
        + _pair(30, True, False)
        # incomplete: one orientation truncated.
        + [
            _game(game_seed=31, orientation="candidate_red", search_won=True),
            _game(game_seed=31, orientation="candidate_blue", search_won=None),
        ]
    )
    pairs, diagnostics = mpc.build_pairs(games)
    assert diagnostics["duplicate_games_dropped"] == 2
    assert diagnostics["incomplete_pairs"] == 1
    assert len(pairs) == 1


def test_orientation_ordering_is_consistent_not_arbitrary():
    # candidate_red must always be position 1 regardless of list order.
    reversed_order = [
        _game(game_seed=40, orientation="candidate_blue", search_won=False),
        _game(game_seed=40, orientation="candidate_red", search_won=True),
    ]
    pairs, _ = mpc.build_pairs(reversed_order)
    assert pairs == [(1.0, 0.0)]


def test_interpret_correlation_sign_language():
    assert "ADDS" in mpc.interpret_correlation(-0.15)
    assert "REMOVES" in mpc.interpret_correlation(0.15)
    assert "neither" in mpc.interpret_correlation(0.0)
    assert "insufficient" in mpc.interpret_correlation(None)


def test_measure_pair_correlation_end_to_end_reads_json_file(tmp_path: Path):
    games = (
        _pair(50, True, False)
        + _pair(51, False, True)
        + _pair(52, True, False)
        + _pair(53, False, True)
    )
    path = tmp_path / "gate_record.json"
    path.write_text(json.dumps({"games": games}))
    report = mpc.measure_pair_correlation([str(path)])
    assert report["n_pairs"] == 4
    assert report["correlation"] == -1.0
    assert "ADDS" in report["interpretation"]
    assert report["files_matched"] == [str(path)]


def test_measure_pair_correlation_supports_glob_pattern(tmp_path: Path):
    for i, (a, b) in enumerate([(True, True), (False, False)]):
        path = tmp_path / f"arm_{i}.json"
        path.write_text(json.dumps({"games": _pair(60 + i, a, b)}))
    report = mpc.measure_pair_correlation([str(tmp_path / "arm_*.json")])
    assert report["n_pairs"] == 2
    assert len(report["files_matched"]) == 2
