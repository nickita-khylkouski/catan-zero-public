from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

whr = pytest.importorskip("whr")

from tools.whr_ladder import (
    GameRecord,
    IngestStats,
    build_report,
    classify_stratum,
    expand_aggregate_wins_losses,
    fit_whr_stratum,
    ingest_files,
    is_lineage_identity,
    parse_ordinal_time_step,
    pentanomial_pairs_to_synthetic_games,
    resolve_pair_identities,
    resolve_time_step,
    walk_result_files,
)


# --------------------------------------------------------------------------- known-Elo recovery
def test_fit_whr_stratum_recovers_known_bradley_terry_elo_diff() -> None:
    # 700/1000 wins for A over B at one time_step, virtual_games=0 removes
    # WHR's regularizing prior so the fit matches the closed-form two-outcome
    # Bradley-Terry MLE exactly: elo_diff = 400 * log10(p / (1 - p)).
    records = [
        GameRecord(
            player_a="A",
            player_b="B",
            winner_of_player_a=(i < 700),
            time_step_source="parsed_ordinal",
            time_step=0,
            stratum="low_n_internal",
            source_file="synthetic",
        )
        for i in range(1000)
    ]
    base = fit_whr_stratum(records, w2=300.0, virtual_games=0)
    a_elo = base.ratings_for_player("A")[0][1]
    b_elo = base.ratings_for_player("B")[0][1]
    expected_diff = 400.0 * math.log10(0.7 / 0.3)
    assert (a_elo - b_elo) == pytest.approx(expected_diff, abs=1.0)


def test_fit_whr_stratum_default_virtual_games_stays_close_to_known_elo() -> None:
    # With the WHR default virtual_games=2 prior, 1000 games still resolves
    # close to the closed-form MLE (the prior washes out at this sample size).
    records = [
        GameRecord(
            player_a="A",
            player_b="B",
            winner_of_player_a=(i < 700),
            time_step_source="parsed_ordinal",
            time_step=0,
            stratum="low_n_internal",
            source_file="synthetic",
        )
        for i in range(1000)
    ]
    base = fit_whr_stratum(records, w2=300.0, virtual_games=2)
    a_elo = base.ratings_for_player("A")[0][1]
    b_elo = base.ratings_for_player("B")[0][1]
    expected_diff = 400.0 * math.log10(0.7 / 0.3)
    assert (a_elo - b_elo) == pytest.approx(expected_diff, abs=5.0)


# --------------------------------------------------------------------------- time_step / lineage heuristic
def test_parse_ordinal_time_step_gen_pattern_with_letter_suffix() -> None:
    assert parse_ordinal_time_step("runs/gen2A/checkpoint.pt") == 201
    assert parse_ordinal_time_step("runs/gen3/checkpoint.pt") == 300


def test_parse_ordinal_time_step_step_pattern() -> None:
    assert parse_ordinal_time_step("runs/step_20000/checkpoint.pt") == 20000


def test_parse_ordinal_time_step_epoch_pattern() -> None:
    assert parse_ordinal_time_step("runs/bc_epoch3/checkpoint.pt") == 3


def test_parse_ordinal_time_step_returns_none_when_unparseable() -> None:
    assert parse_ordinal_time_step("runs/bc/entity_graph_35m_oldbase/checkpoint.pt") is None


def test_is_lineage_identity_matches_parse_ordinal() -> None:
    assert is_lineage_identity("runs/gen4/checkpoint.pt") is True
    assert is_lineage_identity("catanatron_value") is False


def test_resolve_time_step_falls_back_to_mtime_bucket(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    path.write_text("{}", encoding="utf-8")
    time_step, source = resolve_time_step(identity_hint="catanatron_value", file_path=path)
    assert source == "mtime_fallback"
    assert time_step == int(path.stat().st_mtime // 86400)


def test_resolve_time_step_prefers_parsed_ordinal(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    path.write_text("{}", encoding="utf-8")
    time_step, source = resolve_time_step(identity_hint="runs/gen5/checkpoint.pt", file_path=path)
    assert source == "parsed_ordinal"
    assert time_step == 500


def test_resolve_pair_identities_collapses_lineage_side_to_champion_name() -> None:
    player_a, player_b = resolve_pair_identities(
        "runs/gen3/checkpoint.pt", "catanatron_value", champion_name="champion_lineage"
    )
    assert player_a == "champion_lineage"
    assert player_b == "catanatron_value"


def test_resolve_pair_identities_leaves_non_lineage_pair_untouched() -> None:
    player_a, player_b = resolve_pair_identities(
        "runs/bc/hardtarget/checkpoint.pt", "catanatron_value", champion_name="champion_lineage"
    )
    assert player_a == "runs/bc/hardtarget/checkpoint.pt"
    assert player_b == "catanatron_value"


def test_resolve_pair_identities_lineage_vs_lineage_avoids_self_game() -> None:
    # Regression test for the degenerate self-play collapse (module
    # docstring, Decision 1): direct gen-vs-gen H2H must not collapse BOTH
    # sides to champion_name.
    player_a, player_b = resolve_pair_identities(
        "runs/gen3/checkpoint.pt", "runs/gen2/checkpoint.pt", champion_name="champion_lineage"
    )
    assert player_a == "champion_lineage"
    assert player_b == "runs/gen2/checkpoint.pt"
    assert player_a != player_b


# --------------------------------------------------------------------------- pentanomial approximation
def test_pentanomial_pairs_to_synthetic_games_matches_docstring_example() -> None:
    games = pentanomial_pairs_to_synthetic_games(n_ll=1, n_split=2, n_ww=3)
    assert len(games) == 12
    assert sum(games) == 8  # 3*2 (WW) + 2*1 (split) True outcomes
    assert games.count(False) == 4  # 1*2 (LL) + 2*1 (split) False outcomes
    mean_score = sum(games) / len(games)
    expected_mean = (3 * 1.0 + 2 * 0.5 + 1 * 0.0) / 6
    assert mean_score == pytest.approx(expected_mean)


def test_pentanomial_pairs_to_synthetic_games_all_ww() -> None:
    assert pentanomial_pairs_to_synthetic_games(0, 0, 5) == [True, True] * 5


def test_pentanomial_pairs_to_synthetic_games_all_ll() -> None:
    assert pentanomial_pairs_to_synthetic_games(5, 0, 0) == [False, False] * 5


def test_expand_aggregate_wins_losses_exact_counts() -> None:
    outcomes = expand_aggregate_wins_losses(wins=7, games=10)
    assert outcomes.count(True) == 7
    assert outcomes.count(False) == 3


# --------------------------------------------------------------------------- stratification
def test_classify_stratum_low_n_internal() -> None:
    assert classify_stratum("runs/gen2/checkpoint.pt", n_games=50) == "low_n_internal"


def test_classify_stratum_production_n_internal() -> None:
    assert classify_stratum("runs/gen2/checkpoint.pt", n_games=1000) == "production_n_internal"


def test_classify_stratum_neutral_harness_for_internal_bots() -> None:
    assert classify_stratum("catanatron_value", n_games=1000) == "neutral_harness"
    assert classify_stratum("catanatron_ab3", n_games=50) == "neutral_harness"


def test_classify_stratum_external_catanatron_placeholder_marker() -> None:
    # No real on-disk naming convention distinguishes an external Catanatron
    # harness from the internal bots (documented gap in the module
    # docstring); this exercises the placeholder marker so the classifier
    # is proven correct IF such a convention is ever added upstream.
    assert classify_stratum("catanatron_external_engine", n_games=500) == "external_catanatron"


def test_classify_stratum_raw_policy_is_forced_not_inferred() -> None:
    assert classify_stratum("anything", n_games=1, forced="raw_policy") == "raw_policy"


def test_classify_stratum_rejects_unknown_forced_value() -> None:
    with pytest.raises(ValueError):
        classify_stratum("anything", n_games=1, forced="bogus_stratum")


def test_strata_do_not_bleed_into_each_other() -> None:
    # Two strata with very different synthetic strengths for the "same"
    # player name must fit as genuinely separate whr.Base instances. (Not a
    # 100%/0% shutout in either direction -- an unregularized all-one-sided
    # sample drives WHR's MLE to +/-infinity, which is a real property of
    # the model, not a bug, but isn't useful for this isolation check.)
    strong_records = [
        GameRecord(
            player_a="shared_name",
            player_b="opp",
            winner_of_player_a=(i < 80),
            time_step_source="parsed_ordinal",
            time_step=0,
            stratum="low_n_internal",
            source_file="synthetic",
        )
        for i in range(100)
    ]
    weak_records = [
        GameRecord(
            player_a="shared_name",
            player_b="opp",
            winner_of_player_a=(i < 20),
            time_step_source="parsed_ordinal",
            time_step=0,
            stratum="production_n_internal",
            source_file="synthetic",
        )
        for i in range(100)
    ]
    strong_base = fit_whr_stratum(strong_records, virtual_games=0)
    weak_base = fit_whr_stratum(weak_records, virtual_games=0)
    strong_elo = strong_base.ratings_for_player("shared_name")[0][1]
    weak_elo = weak_base.ratings_for_player("shared_name")[0][1]
    assert strong_elo > 0
    assert weak_elo < 0
    assert strong_elo != weak_elo


# --------------------------------------------------------------------------- Family A ingest
def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_ingest_family_a_nested_shape(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "scoreboard.json",
        {
            "candidate": "runs/gen2/checkpoint.pt",
            "results": [
                {"opponent": "catanatron_value", "wins": 60, "games": 100, "track": "2p_no_trade"},
            ],
        },
    )
    records, stats = ingest_files(
        walk_result_files(tmp_path), repo_root=tmp_path, champion_name="champion_lineage"
    )
    assert stats.files_family_a == 1
    assert len(records) == 100
    assert all(r.player_a == "champion_lineage" for r in records)
    assert all(r.player_b == "catanatron_value" for r in records)
    assert sum(1 for r in records if r.winner_of_player_a) == 60


def test_ingest_family_a_flat_shape(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "scoreboard.json",
        {
            "candidate": "runs/gen2/checkpoint.pt",
            "opponent": "catanatron_ab3",
            "wins": 55,
            "games": 100,
            "track": "2p_no_trade",
        },
    )
    records, stats = ingest_files(
        walk_result_files(tmp_path), repo_root=tmp_path, champion_name="champion_lineage"
    )
    assert stats.files_family_a == 1
    assert len(records) == 100
    assert sum(1 for r in records if r.winner_of_player_a) == 55


def test_ingest_family_a_uses_real_game_outcomes_when_present(tmp_path: Path) -> None:
    outcomes = [True, False, True, None, True]
    _write_json(
        tmp_path / "scoreboard.json",
        {
            "candidate": "runs/gen2/checkpoint.pt",
            "results": [
                {
                    "opponent": "catanatron_value",
                    "wins": 3,
                    "games": 5,
                    "game_outcomes": outcomes,
                    "track": "2p_no_trade",
                }
            ],
        },
    )
    records, stats = ingest_files(
        walk_result_files(tmp_path), repo_root=tmp_path, champion_name="champion_lineage"
    )
    # The None (truncated) game must be discarded, not coerced to a loss.
    assert len(records) == 4
    assert stats.games_discarded_no_winner == 1


def test_ingest_family_a_skips_part_files(tmp_path: Path) -> None:
    (tmp_path / "part0.json").write_text(
        json.dumps({"candidate": "ckpt.pt", "opponent": "catanatron_ab3", "wins": 3, "games": 5}),
        encoding="utf-8",
    )
    (tmp_path / "part1.json").write_text(
        json.dumps({"candidate": "ckpt.pt", "opponent": "catanatron_ab3", "wins": 2, "games": 5}),
        encoding="utf-8",
    )
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "candidate": "ckpt.pt",
                "opponent": "catanatron_ab3",
                "wins": 5,
                "games": 10,
                "part_files": ["part0.json", "part1.json"],
            }
        ),
        encoding="utf-8",
    )
    records, stats = ingest_files(
        walk_result_files(tmp_path), repo_root=tmp_path, champion_name="champion_lineage"
    )
    assert stats.files_skipped_part_file == 2
    assert len(records) == 10  # not 20 -- parts must not be double counted


def test_ingest_family_a_checkpoint_prefix_canonicalization(tmp_path: Path) -> None:
    # Exercises the same _canonicalize_node_id bug-fix elo_ladder.py has
    # (checkpoint: prefix collapse), reused verbatim here.
    _write_json(
        tmp_path / "scoreboard.json",
        {
            "candidate": "runs/bc/hardtarget/checkpoint.pt",
            "results": [
                {"opponent": "checkpoint:runs/bc/hardtarget/checkpoint.pt", "wins": 1, "games": 2},
            ],
        },
    )
    # This is a same-checkpoint self-match (opponent == candidate once
    # canonicalized) and must be dropped entirely, not silently kept as a
    # self-game.
    records, stats = ingest_files(
        walk_result_files(tmp_path), repo_root=tmp_path, champion_name="champion_lineage"
    )
    assert records == []


def test_ingest_family_a_track_filter(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "scoreboard.json",
        {
            "candidate": "runs/gen2/checkpoint.pt",
            "results": [
                {"opponent": "catanatron_ab3", "wins": 5, "games": 10, "track": "2p_no_trade"},
                {"opponent": "catanatron_ab3", "wins": 4, "games": 10, "track": "4p_trade"},
            ],
        },
    )
    records, stats = ingest_files(
        walk_result_files(tmp_path), repo_root=tmp_path, champion_name="champion_lineage", track="2p_no_trade"
    )
    assert len(records) == 10
    # Track-filtered-out entries must be visible in the ingest report (mirrors
    # tools.elo_ladder.py's "skipped_other_tracks"), not silently dropped.
    assert stats.skipped_other_tracks == {"4p_trade"}


# --------------------------------------------------------------------------- Family B: per-game (real records)
def test_ingest_family_b_vs_bot_h2h_real_per_game_records(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "h2h.json",
        {
            "candidate_checkpoint": "runs/gen4/checkpoint.pt",
            "baseline_bot": "catanatron_value",
            "games": [
                {"candidate_won": True, "pair_id": 0},
                {"candidate_won": False, "pair_id": 0},
                {"candidate_won": None, "pair_id": 1},  # truncated
            ],
        },
    )
    records, stats = ingest_files(
        walk_result_files(tmp_path), repo_root=tmp_path, champion_name="champion_lineage"
    )
    assert stats.files_family_b_per_game == 1
    assert len(records) == 2
    assert stats.games_discarded_no_winner == 1
    assert all(r.player_a == "champion_lineage" for r in records)
    assert all(r.player_b == "catanatron_value" for r in records)
    assert all(r.stratum == "neutral_harness" for r in records)


def test_ingest_family_b_cross_net_h2h_internal_stratum(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "cross_net.json",
        {
            "candidate_checkpoint": "runs/gen4/checkpoint.pt",
            "baseline_checkpoint": "runs/gen3/checkpoint.pt",
            "games": [{"candidate_won": True} for _ in range(50)],
        },
    )
    records, stats = ingest_files(
        walk_result_files(tmp_path), repo_root=tmp_path, champion_name="champion_lineage"
    )
    assert stats.files_family_b_per_game == 1
    assert len(records) == 50
    assert all(r.player_a == "champion_lineage" for r in records)
    assert all(r.player_b == "runs/gen3/checkpoint.pt" for r in records)  # not collapsed -- avoids self-game
    assert all(r.stratum == "low_n_internal" for r in records)  # 50 < 200 threshold


def test_ingest_family_b_search_vs_raw_policy_stratum(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "raw_h2h.json",
        {
            "checkpoint": "runs/gen4/checkpoint.pt",
            "games": [
                {"search_won": True, "search_color": "RED", "raw_color": "BLUE"},
                {"search_won": False, "search_color": "BLUE", "raw_color": "RED"},
            ],
        },
    )
    records, stats = ingest_files(
        walk_result_files(tmp_path), repo_root=tmp_path, champion_name="champion_lineage"
    )
    assert stats.files_family_b_per_game == 1
    assert len(records) == 2
    assert all(r.stratum == "raw_policy" for r in records)
    assert all(r.player_a == "champion_lineage" for r in records)
    assert all(r.player_b == "runs/gen4/checkpoint.pt::raw_policy" for r in records)


def test_ingest_family_b_search_vs_raw_policy_non_lineage_checkpoint_still_distinct(tmp_path: Path) -> None:
    # A checkpoint path with no parseable gen/step/epoch ordinal must still
    # keep the search and raw_policy sides distinct (not a self-game).
    _write_json(
        tmp_path / "raw_h2h.json",
        {
            "checkpoint": "runs/bc/hardtarget/checkpoint.pt",
            "games": [{"search_won": True}],
        },
    )
    records, _ = ingest_files(
        walk_result_files(tmp_path), repo_root=tmp_path, champion_name="champion_lineage"
    )
    assert len(records) == 1
    assert records[0].player_a != records[0].player_b
    assert records[0].player_a == "runs/bc/hardtarget/checkpoint.pt"
    assert records[0].player_b == "runs/bc/hardtarget/checkpoint.pt::raw_policy"


# --------------------------------------------------------------------------- Family B: pentanomial-aggregate-only
def test_ingest_family_b_pentanomial_aggregate_single_arm(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "aggregate.json",
        {
            "config": {"checkpoint": "runs/gen5/checkpoint.pt"},
            "pentanomial_sprt": {"ll_pairs": 1, "split_pairs": 2, "ww_pairs": 3},
        },
    )
    records, stats = ingest_files(
        walk_result_files(tmp_path), repo_root=tmp_path, champion_name="champion_lineage"
    )
    assert stats.files_family_b_pentanomial_only == 1
    assert stats.used_pentanomial_approximation == 1
    assert len(records) == 12  # see pentanomial_pairs_to_synthetic_games docstring example
    assert all(r.stratum == "raw_policy" for r in records)
    assert all(r.player_a == "champion_lineage" for r in records)


def test_ingest_family_b_pentanomial_aggregate_multi_arm(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "aggregate.json",
        {
            "elo0": 0.0,
            "elo1": 30.0,
            "arms": {
                "v3a_cs0.03": {
                    "config": {"checkpoint": "runs/gen5/checkpoint.pt", "c_scale": 0.03},
                    "pentanomial_sprt": {"ll_pairs": 0, "split_pairs": 0, "ww_pairs": 10},
                },
                "v3b_cs0.03": {
                    "config": {"checkpoint": "runs/gen6/checkpoint.pt", "c_scale": 0.03},
                    "pentanomial_sprt": {"ll_pairs": 10, "split_pairs": 0, "ww_pairs": 0},
                },
            },
        },
    )
    records, stats = ingest_files(
        walk_result_files(tmp_path), repo_root=tmp_path, champion_name="champion_lineage"
    )
    assert stats.files_family_b_pentanomial_only == 1
    assert stats.used_pentanomial_approximation == 2
    assert len(records) == 40  # 2 arms * 10 pairs * 2 synthetic games each


def test_ingest_family_b_pentanomial_missing_checkpoint_flags_identity_gap(tmp_path: Path) -> None:
    # Regression test for the documented h2h_postrepair_aggregate.py gap:
    # its pooled report carries no "checkpoint" field at all.
    _write_json(
        tmp_path / "aggregate.json",
        {
            "config": {"n_full": 64, "c_scale": 0.1},
            "pentanomial_sprt": {"ll_pairs": 1, "split_pairs": 0, "ww_pairs": 1},
        },
    )
    records, stats = ingest_files(
        walk_result_files(tmp_path), repo_root=tmp_path, champion_name="champion_lineage"
    )
    assert stats.identity_from_arm_label_gap == 1
    assert len(records) == 4
    assert all(r.player_b.startswith("arm:") for r in records)


# --------------------------------------------------------------------------- opening_panel exclusion
def test_ingest_skips_opening_panel_reports(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "panel_eval.json",
        {
            "checkpoint": "runs/gen4/checkpoint.pt",
            "panel": "runs/panels/opening_200.json",
            "aggregate": {"flip_rate": 0.1},
            "per_root": [{"flipped": False}],
        },
    )
    records, stats = ingest_files(
        walk_result_files(tmp_path), repo_root=tmp_path, champion_name="champion_lineage"
    )
    assert stats.files_opening_panel_skipped == 1
    assert records == []


# --------------------------------------------------------------------------- malformed / unrecognized files
def test_ingest_handles_malformed_json(tmp_path: Path) -> None:
    (tmp_path / "broken.json").write_text("{not valid json", encoding="utf-8")
    records, stats = ingest_files(
        walk_result_files(tmp_path), repo_root=tmp_path, champion_name="champion_lineage"
    )
    assert stats.files_malformed == 1
    assert records == []


def test_ingest_handles_unrecognized_shape(tmp_path: Path) -> None:
    _write_json(tmp_path / "unrelated.json", {"some_field": "some_value"})
    records, stats = ingest_files(
        walk_result_files(tmp_path), repo_root=tmp_path, champion_name="champion_lineage"
    )
    assert stats.files_unrecognized == 1
    assert records == []


# --------------------------------------------------------------------------- report + CLI-level smoke test
def test_build_report_produces_trajectory_with_cis_overlap_flags(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "gen1.json",
        {
            "candidate": "runs/gen1/checkpoint.pt",
            "results": [
                {"opponent": "runs/bc/hardtarget/checkpoint.pt", "wins": 55, "games": 100, "track": "2p_no_trade"}
            ],
        },
    )
    _write_json(
        tmp_path / "gen2.json",
        {
            "candidate": "runs/gen2/checkpoint.pt",
            "results": [
                {"opponent": "runs/bc/hardtarget/checkpoint.pt", "wins": 60, "games": 100, "track": "2p_no_trade"}
            ],
        },
    )
    records, stats = ingest_files(
        walk_result_files(tmp_path), repo_root=tmp_path, champion_name="champion_lineage"
    )
    report = build_report(records, stats=stats, champion_name="champion_lineage")
    assert report["champion_name"] == "champion_lineage"
    # Both generations play a fixed non-lineage checkpoint anchor with
    # < 200 games, so this collapses into the low_n_internal stratum.
    trajectory = report["strata"]["low_n_internal"]["trajectory"]
    assert len(trajectory) == 2
    assert trajectory[0]["delta_from_previous"] is None
    assert trajectory[1]["delta_from_previous"] is not None
    assert isinstance(trajectory[1]["cis_overlap_with_previous"], bool)
    # JSON-serializable end to end (CLI writes exactly this via json.dumps).
    json.dumps(report)


def test_cli_end_to_end_smoke(tmp_path: Path) -> None:
    import subprocess
    import sys

    runs_dir = tmp_path / "runs"
    _write_json(
        runs_dir / "gen1.json",
        {
            "candidate": "runs/gen1/checkpoint.pt",
            "results": [{"opponent": "catanatron_value", "wins": 55, "games": 100, "track": "2p_no_trade"}],
        },
    )
    out_json = tmp_path / "report.json"
    out_md = tmp_path / "report.md"
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools" / "whr_ladder.py"),
            "--runs-dir",
            str(runs_dir),
            "--repo-root",
            str(tmp_path),
            "--out",
            str(out_json),
            "--out-markdown",
            str(out_md),
        ],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert report["champion_name"] == "champion_lineage"
    assert out_md.exists()
