from __future__ import annotations

import sys
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import gumbel_search_vs_raw_h2h as h2h  # type: ignore  # noqa: E402


_concordant_pair_outcomes = h2h._concordant_pair_outcomes


def _game(pair_id: int, search_won: bool | None) -> dict:
    return {"pair_id": pair_id, "search_won": search_won}


def test_ww_pair_counts_as_one_search_win():
    games = [_game(0, True), _game(0, True)]
    outcomes, diagnostics = _concordant_pair_outcomes(games)
    assert outcomes == [True]
    assert diagnostics == {"ww_pairs": 1, "ll_pairs": 0, "split_pairs": 0, "incomplete_pairs": 0}


def test_ll_pair_counts_as_one_loss():
    games = [_game(0, False), _game(0, False)]
    outcomes, diagnostics = _concordant_pair_outcomes(games)
    assert outcomes == [False]
    assert diagnostics == {"ww_pairs": 0, "ll_pairs": 1, "split_pairs": 0, "incomplete_pairs": 0}


def test_split_pair_is_excluded_not_coerced():
    games = [_game(0, True), _game(0, False)]
    outcomes, diagnostics = _concordant_pair_outcomes(games)
    assert outcomes == []
    assert diagnostics["split_pairs"] == 1


def test_pair_with_a_truncated_game_is_excluded():
    games = [_game(0, True), _game(0, None)]
    outcomes, diagnostics = _concordant_pair_outcomes(games)
    assert outcomes == []
    assert diagnostics["incomplete_pairs"] == 1


def test_multiple_pairs_are_reduced_independently():
    games = [
        _game(0, True),
        _game(0, True),  # WW
        _game(1, False),
        _game(1, False),  # LL
        _game(2, True),
        _game(2, False),  # split
    ]
    outcomes, diagnostics = _concordant_pair_outcomes(games)
    assert sorted(outcomes) == [False, True]
    assert diagnostics == {"ww_pairs": 1, "ll_pairs": 1, "split_pairs": 1, "incomplete_pairs": 0}


def test_naive_per_game_pooling_would_overcount_relative_to_pair_level():
    """The bug F5 fixes: pooling each orientation independently counts a WW
    pair as 2 search wins and a split pair as 1 win + 1 loss (net zero, but
    still 2 "informative" trials) -- the concordant-pair reduction must
    produce fewer, more conservative outcomes than naive per-game pooling."""
    games = [
        _game(0, True),
        _game(0, True),  # WW -> 1 pair outcome, but 2 naive outcomes
        _game(1, True),
        _game(1, False),  # split -> 0 pair outcomes, but 2 naive outcomes
    ]
    naive_outcomes = [game["search_won"] for game in games if game["search_won"] is not None]
    pair_outcomes, _diagnostics = _concordant_pair_outcomes(games)
    assert len(naive_outcomes) == 4
    assert len(pair_outcomes) == 1


def test_raw_h2h_threads_coherent_boundary_particle_operator() -> None:
    config = h2h._build_search_config(
        {
            "worker_seed": 7,
            "n_full": 128,
            "max_depth": 80,
            "correct_rust_chance_spectra": True,
            "coherent_public_belief_search": True,
            "boundary_value_particles": 4,
        }
    )

    assert config.coherent_public_belief_search is True
    assert config.boundary_value_particles == 4
