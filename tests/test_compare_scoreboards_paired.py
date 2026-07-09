from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.compare_scoreboards import _mcnemar_exact, _pairing_eligible, main


def _result(*, opponent: str, wins: int, games: int, leg_seed=None, game_outcomes=None) -> dict:
    return {
        "opponent": opponent,
        "wins": wins,
        "games": games,
        "win_rate": wins / games,
        "avg_vp_margin": 0.0,
        "leg_seed": leg_seed,
        "game_outcomes": game_outcomes,
    }


def _report(*, seed, paired_seeds, results) -> dict:
    return {"candidate": "ckpt.pt", "seed": seed, "paired_seeds": paired_seeds, "results": results}


def test_pairing_eligible_true_when_metadata_matches() -> None:
    c = dict(
        _result(opponent="catanatron_ab3", wins=6, games=10, leg_seed=42, game_outcomes=[True] * 6 + [False] * 4),
        _report_seed=7,
        _report_paired_seeds=True,
    )
    b = dict(
        _result(opponent="catanatron_ab3", wins=4, games=10, leg_seed=42, game_outcomes=[True] * 4 + [False] * 6),
        _report_seed=7,
        _report_paired_seeds=True,
    )
    eligible, reason = _pairing_eligible(c, b)
    assert eligible
    assert reason == ""


def test_pairing_eligible_false_when_not_paired_seeds() -> None:
    c = dict(_result(opponent="x", wins=1, games=2), _report_seed=7, _report_paired_seeds=False)
    b = dict(_result(opponent="x", wins=1, games=2), _report_seed=7, _report_paired_seeds=True)
    eligible, reason = _pairing_eligible(c, b)
    assert not eligible
    assert "paired-seeds" in reason


def test_pairing_eligible_false_on_leg_seed_mismatch() -> None:
    c = dict(
        _result(opponent="x", wins=1, games=2, leg_seed=1, game_outcomes=[True, False]),
        _report_seed=7,
        _report_paired_seeds=True,
    )
    b = dict(
        _result(opponent="x", wins=1, games=2, leg_seed=2, game_outcomes=[True, False]),
        _report_seed=7,
        _report_paired_seeds=True,
    )
    eligible, reason = _pairing_eligible(c, b)
    assert not eligible
    assert "leg_seed" in reason


def test_mcnemar_exact_all_concordant_gives_p_value_one() -> None:
    outcomes = [True, True, False, False]
    result = _mcnemar_exact(outcomes, outcomes)
    assert result["discordant_total"] == 0
    assert result["p_value"] == 1.0


def test_mcnemar_exact_detects_clear_asymmetry() -> None:
    # Candidate wins every discordant game; baseline never wins one.
    c_outcomes = [True] * 20 + [False] * 5
    b_outcomes = [False] * 20 + [False] * 5
    result = _mcnemar_exact(c_outcomes, b_outcomes)
    assert result["discordant_candidate_only_wins"] == 20
    assert result["discordant_baseline_only_wins"] == 0
    assert result["p_value"] < 0.01


def test_mcnemar_exact_excludes_pairs_with_a_truncated_side() -> None:
    # Regression test for the adversarial-review truncation-as-loss bias:
    # a None entry (truncated game, no winner) on EITHER side must be
    # excluded from the test entirely, not coerced to False (a loss).
    # index 0: candidate truncated, baseline won -- must be excluded, NOT
    #          counted as a candidate loss (discordant_baseline_only_wins).
    # index 1: baseline truncated, candidate won -- must be excluded, NOT
    #          counted as a candidate win (discordant_candidate_only_wins).
    # index 2/3: normal discordant pair, counted normally.
    c_outcomes = [None, True, True, False]
    b_outcomes = [True, None, False, True]
    result = _mcnemar_exact(c_outcomes, b_outcomes)
    assert result["truncated_pairs_excluded"] == 2
    assert result["usable_games"] == 2
    assert result["discordant_candidate_only_wins"] == 1
    assert result["discordant_baseline_only_wins"] == 1
    assert result["games"] == 4


def test_compare_scoreboards_uses_paired_mcnemar_when_metadata_present(tmp_path: Path) -> None:
    outcomes_c = [True, True, False, True, False, False, True, True, False, True]
    outcomes_b = [True, False, False, False, False, True, False, True, False, True]
    candidate_path = tmp_path / "candidate.json"
    baseline_path = tmp_path / "baseline.json"
    candidate_path.write_text(
        json.dumps(
            _report(
                seed=5,
                paired_seeds=True,
                results=[
                    _result(opponent="catanatron_ab3", wins=sum(outcomes_c), games=10, leg_seed=99, game_outcomes=outcomes_c)
                ],
            )
        ),
        encoding="utf-8",
    )
    baseline_path.write_text(
        json.dumps(
            _report(
                seed=5,
                paired_seeds=True,
                results=[
                    _result(opponent="catanatron_ab3", wins=sum(outcomes_b), games=10, leg_seed=99, game_outcomes=outcomes_b)
                ],
            )
        ),
        encoding="utf-8",
    )
    out_path = tmp_path / "compare.json"
    import argparse
    import tools.compare_scoreboards as compare_scoreboards

    original_parse_args = argparse.ArgumentParser.parse_args
    args = argparse.Namespace(
        candidate_reports=str(candidate_path),
        baseline_reports=str(baseline_path),
        required_opponents="",
        max_regression_win_rate=0.0,
        min_improvement_win_rate=0.0,
        min_games=1,
        out=str(out_path),
    )

    def _fake_parse_args(self, *a, **k):
        return args

    argparse.ArgumentParser.parse_args = _fake_parse_args
    try:
        main()
    finally:
        argparse.ArgumentParser.parse_args = original_parse_args

    report = json.loads(out_path.read_text(encoding="utf-8"))
    comparison = report["comparisons"][0]
    assert comparison["paired_mcnemar"] is not None
    assert comparison["pairing_unavailable_reason"] is None
    assert comparison["paired_mcnemar"]["test"] == "mcnemar_exact"


def test_compare_scoreboards_falls_back_to_unpaired_without_pairing_metadata(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidate.json"
    baseline_path = tmp_path / "baseline.json"
    candidate_path.write_text(
        json.dumps(_report(seed=None, paired_seeds=False, results=[_result(opponent="catanatron_ab3", wins=6, games=10)])),
        encoding="utf-8",
    )
    baseline_path.write_text(
        json.dumps(_report(seed=None, paired_seeds=False, results=[_result(opponent="catanatron_ab3", wins=4, games=10)])),
        encoding="utf-8",
    )
    out_path = tmp_path / "compare.json"
    import argparse

    args = argparse.Namespace(
        candidate_reports=str(candidate_path),
        baseline_reports=str(baseline_path),
        required_opponents="",
        max_regression_win_rate=0.0,
        min_improvement_win_rate=0.0,
        min_games=1,
        out=str(out_path),
    )
    original_parse_args = argparse.ArgumentParser.parse_args

    def _fake_parse_args(self, *a, **k):
        return args

    argparse.ArgumentParser.parse_args = _fake_parse_args
    try:
        main()
    finally:
        argparse.ArgumentParser.parse_args = original_parse_args

    report = json.loads(out_path.read_text(encoding="utf-8"))
    comparison = report["comparisons"][0]
    assert comparison["paired_mcnemar"] is None
    assert comparison["pairing_unavailable_reason"] is not None
    assert comparison["approx_z_unpaired"] is not None
