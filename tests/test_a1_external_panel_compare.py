from __future__ import annotations

import pytest

from tools.a1_external_panel_compare import (
    ExternalPanelComparisonError,
    compare_matched_external_panels,
)


def _panel(outcomes: list[bool]) -> dict:
    assert len(outcomes) % 2 == 0
    games = []
    for index, won in enumerate(outcomes):
        games.append(
            {
                "game_seed": 9000 + index // 2,
                "orientation": (
                    "candidate_first" if index % 2 == 0 else "candidate_second"
                ),
                "candidate_won": won,
                "terminated": True,
                "truncated": False,
                "error": None,
                "engine_divergence": False,
            }
        )
    return {"games": games}


def test_corrected_external_counts_are_paired_and_noninferiority_unresolved() -> None:
    # Reproduce the corrected panel contingency: 462 candidate wins, 452
    # champion wins, but only +10 among 448 discordant games.
    joint = (
        [(True, True)] * 233
        + [(True, False)] * 229
        + [(False, True)] * 219
        + [(False, False)] * 319
    )
    candidate = _panel([row[0] for row in joint])
    champion = _panel([row[1] for row in joint])

    result = compare_matched_external_panels(candidate, champion)

    assert result["candidate_win_rate"] == 0.462
    assert result["champion_win_rate"] == 0.452
    assert result["candidate_minus_champion"] == pytest.approx(0.01)
    assert result["contingency"] == {
        "both_win": 233,
        "candidate_only_win": 229,
        "champion_only_win": 219,
        "both_loss": 319,
        "discordant": 448,
    }
    assert result["mcnemar_exact_two_sided_p"] == pytest.approx(0.6707309886927966)
    assert result["noninferiority"]["passed"] is False
    assert result["paired_seed_cluster_95ci"][0] < -0.02


def test_pairing_rejects_missing_orientation_row() -> None:
    candidate = _panel([True, False, True, False])
    champion = _panel([False, False, True, True])
    champion["games"].pop()

    with pytest.raises(
        ExternalPanelComparisonError, match="same seed/orientation cohort"
    ):
        compare_matched_external_panels(candidate, champion)


def test_seed_cluster_interval_does_not_treat_orientations_as_independent() -> None:
    # Perfectly correlated orientations: each seed is either +1 or -1.  The
    # cluster standard error must therefore be based on two observations, not
    # four pseudo-independent games.
    candidate = _panel([True, True, False, False])
    champion = _panel([False, False, True, True])
    result = compare_matched_external_panels(candidate, champion)

    assert result["seed_clusters"] == 2
    assert result["candidate_minus_champion"] == 0.0
    assert result["paired_seed_cluster_standard_error"] == pytest.approx(1.0)
