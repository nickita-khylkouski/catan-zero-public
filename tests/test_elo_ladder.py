from __future__ import annotations

import json
from pathlib import Path

import pytest

import os

from tools.elo_ladder import (
    POSTFIX_BOTS_ERA_CUTOFF_EPOCH,
    _build_win_tables,
    _connected_component,
    _load_matchups,
    _matchup_record,
    build_ladder_report,
    era_for_mtime,
    fit_bradley_terry,
)


def _matchup(candidate: str, opponent: str, wins: int, games: int) -> dict:
    return {"candidate": candidate, "opponent": opponent, "wins": wins, "games": games, "game_outcomes": None}


def test_fit_bradley_terry_recovers_known_pairwise_elo_when_only_two_nodes() -> None:
    # With only one edge, the joint MLE must exactly match the closed-form
    # two-outcome MLE: elo_diff = 400 * log10(p / (1 - p)).
    wins = {"A": {"anchor": 700.0}, "anchor": {"A": 300.0}}
    pi, _ = fit_bradley_terry(wins, ["A", "anchor"], anchor="anchor", smoothing=0.0)
    import math

    expected_elo = 400.0 * math.log10(0.7 / 0.3)
    got_elo = 400.0 * math.log10(pi["A"])
    assert got_elo == pytest.approx(expected_elo, abs=0.5)
    assert pi["anchor"] == pytest.approx(1.0)


def test_fit_bradley_terry_anchor_always_normalizes_to_pi_one() -> None:
    wins = {
        "A": {"anchor": 700.0, "B": 600.0},
        "B": {"anchor": 500.0, "A": 400.0},
        "anchor": {"A": 300.0, "B": 500.0},
    }
    pi, _ = fit_bradley_terry(wins, ["A", "B", "anchor"], anchor="anchor")
    assert pi["anchor"] == pytest.approx(1.0, abs=1e-6)
    # A beats anchor more than B does, and beats B head-to-head, so A must
    # rank strictly above B in the fitted ladder.
    assert pi["A"] > pi["B"] > pi["anchor"]


def test_fit_bradley_terry_isolated_node_keeps_pi_one() -> None:
    wins = {"A": {"anchor": 500.0}, "anchor": {"A": 500.0}}
    pi, _ = fit_bradley_terry(wins, ["A", "anchor", "isolated"], anchor="anchor")
    assert pi["isolated"] == pytest.approx(1.0)


def test_connected_component_excludes_disconnected_nodes() -> None:
    wins = {"A": {"anchor": 5.0}, "anchor": {"A": 5.0}, "X": {"Y": 3.0}, "Y": {"X": 2.0}}
    connected = _connected_component(["A", "anchor", "X", "Y"], wins, "anchor")
    assert connected == {"A", "anchor"}


def test_build_ladder_report_ranks_stronger_candidate_higher() -> None:
    matchups = [
        _matchup("strong", "catanatron_value", 700, 1000),
        _matchup("weak", "catanatron_value", 400, 1000),
    ]
    report = build_ladder_report(matchups, anchor="catanatron_value", bootstrap_samples=0)
    by_node = {row["node"]: row for row in report["ladder"]}
    assert by_node["strong"]["elo"] > by_node["catanatron_value"]["elo"] > by_node["weak"]["elo"]
    assert by_node["catanatron_value"]["elo"] == pytest.approx(0.0, abs=1e-6)


def test_build_ladder_report_marks_disconnected_nodes_unranked() -> None:
    matchups = [
        _matchup("linked", "catanatron_value", 500, 1000),
        _matchup("orphan", "some_other_bot", 5, 10),
    ]
    report = build_ladder_report(matchups, anchor="catanatron_value", bootstrap_samples=0)
    unranked_nodes = {row["node"] for row in report["unranked"]}
    assert "orphan" in unranked_nodes
    assert "some_other_bot" in unranked_nodes
    ranked_nodes = {row["node"] for row in report["ladder"]}
    assert "linked" in ranked_nodes


def test_build_ladder_report_bootstrap_produces_ci_bounds() -> None:
    matchups = [_matchup("strong", "catanatron_value", 140, 200)]
    report = build_ladder_report(matchups, anchor="catanatron_value", bootstrap_samples=30, bootstrap_seed=3)
    strong = next(row for row in report["ladder"] if row["node"] == "strong")
    assert strong["elo_ci95_lower"] is not None
    assert strong["elo_ci95_lower"] <= strong["elo"] <= strong["elo_ci95_upper"]


def _write_scoreboard_report(path: Path, *, candidate: str, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"candidate": candidate, "results": results}), encoding="utf-8")


def test_load_matchups_merges_checkpoint_prefixed_opponent_with_bare_candidate(tmp_path: Path) -> None:
    # Regression test for the node-identity split the team lead caught:
    # evaluate_scoreboard.py's opponent spec parser prefixes a checkpoint
    # opponent with "checkpoint:", but the SAME physical checkpoint's
    # report-level "candidate" field (when it plays candidate elsewhere) has
    # no prefix. Left unmerged this creates two Bradley-Terry nodes for one
    # checkpoint -- exactly what inflated one node to +280 Elo.
    checkpoint_path = "runs/bc/some_checkpoint/checkpoint.pt"
    other_checkpoint = "runs/distributed/other_run/checkpoints/step_1.pt"
    _write_scoreboard_report(
        tmp_path / "candidate_side.json",
        candidate=other_checkpoint,
        results=[
            {
                "opponent": f"checkpoint:{checkpoint_path}",
                "wins": 108,
                "games": 200,
                "track": "2p_no_trade",
            }
        ],
    )
    _write_scoreboard_report(
        tmp_path / "opponent_side.json",
        candidate=checkpoint_path,
        results=[
            {
                "opponent": "catanatron_value",
                "wins": 60,
                "games": 200,
                "track": "2p_no_trade",
            }
        ],
    )
    matchups, _, _ = _load_matchups(tmp_path, repo_root=tmp_path, track="2p_no_trade")
    node_ids = {m["candidate"] for m in matchups} | {m["opponent"] for m in matchups}
    # The checkpoint must appear as exactly ONE node id, not two.
    assert checkpoint_path in node_ids
    assert f"checkpoint:{checkpoint_path}" not in node_ids
    matching = [m for m in matchups if checkpoint_path in (m["candidate"], m["opponent"])]
    assert len(matching) == 2  # both legs (as opponent, and as candidate) resolve to the same node


def test_canonicalize_node_id_leaves_bot_names_unchanged() -> None:
    from tools.elo_ladder import _canonicalize_node_id

    assert _canonicalize_node_id("catanatron_ab3") == "catanatron_ab3"
    assert _canonicalize_node_id("checkpoint:runs/x/y.pt") == "runs/x/y.pt"
    assert _canonicalize_node_id("./runs/x/y.pt") == "runs/x/y.pt"


def test_load_matchups_prefers_report_level_candidate_over_short_label(tmp_path: Path) -> None:
    # Regression test for the schema surprise: per-result "candidate" is a
    # short architecture label shared by many checkpoints; the report-level
    # "candidate" (the actual checkpoint path) must be used as node identity.
    _write_scoreboard_report(
        tmp_path / "run_epoch1" / "scoreboard.json",
        candidate="runs/teacher/run_epoch1/checkpoint.pt",
        results=[{"candidate": "xdim_graph", "opponent": "catanatron_ab3", "wins": 5, "games": 10, "track": "2p_no_trade"}],
    )
    matchups, _, _ = _load_matchups(tmp_path, repo_root=tmp_path, track="2p_no_trade")
    assert matchups[0]["candidate"] == "runs/teacher/run_epoch1/checkpoint.pt"


def test_load_matchups_skips_files_referenced_as_part_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "part0.json").write_text(
        json.dumps({"candidate": "ckpt.pt", "opponent": "catanatron_ab3", "wins": 3, "games": 5}),
        encoding="utf-8",
    )
    (run_dir / "part1.json").write_text(
        json.dumps({"candidate": "ckpt.pt", "opponent": "catanatron_ab3", "wins": 2, "games": 5}),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "candidate": "ckpt.pt",
                "opponent": "catanatron_ab3",
                "wins": 5,
                "games": 10,
                "part_files": [str((run_dir / "part0.json").relative_to(tmp_path)), str((run_dir / "part1.json").relative_to(tmp_path))],
            }
        ),
        encoding="utf-8",
    )
    matchups, _, _ = _load_matchups(tmp_path, repo_root=tmp_path, track="2p_no_trade")
    total_games = sum(m["games"] for m in matchups if m["opponent"] == "catanatron_ab3")
    assert total_games == 10  # not 20 -- the parts must not be double-counted


def test_load_matchups_filters_other_tracks(tmp_path: Path) -> None:
    _write_scoreboard_report(
        tmp_path / "scoreboard.json",
        candidate="ckpt.pt",
        results=[
            {"opponent": "catanatron_ab3", "wins": 5, "games": 10, "track": "2p_no_trade"},
            {"opponent": "catanatron_ab3", "wins": 4, "games": 10, "track": "4p_trade"},
        ],
    )
    matchups, skipped, _ = _load_matchups(tmp_path, repo_root=tmp_path, track="2p_no_trade")
    assert len(matchups) == 1
    assert "4p_trade" in skipped


# --------------------------------------------------------------------------- era tagging
# The last rules-affecting fix (commit 0586f12, 2026-07-03T00:32:15Z; A15-A17
# vendored Longest Road fixes, superseding the earlier A2 AB-teacher fix
# cutoff) changed engine/bot behavior; scoreboard files from before/after
# that instant are not comparable and must be tagged + filterable.


def test_era_for_mtime_boundary_is_inclusive_of_postfix() -> None:
    assert era_for_mtime(POSTFIX_BOTS_ERA_CUTOFF_EPOCH - 1.0) == "prefix-bots"
    assert era_for_mtime(POSTFIX_BOTS_ERA_CUTOFF_EPOCH) == "postfix-bots"
    assert era_for_mtime(POSTFIX_BOTS_ERA_CUTOFF_EPOCH + 1.0) == "postfix-bots"


def _write_scoreboard_report_with_mtime(path: Path, *, candidate: str, results: list[dict], mtime: float) -> None:
    _write_scoreboard_report(path, candidate=candidate, results=results)
    os.utime(path, (mtime, mtime))


def test_load_matchups_tags_each_matchup_with_its_files_era(tmp_path: Path) -> None:
    _write_scoreboard_report_with_mtime(
        tmp_path / "old_run.json",
        candidate="ckpt_old.pt",
        results=[{"opponent": "catanatron_ab3", "wins": 6, "games": 10, "track": "2p_no_trade"}],
        mtime=POSTFIX_BOTS_ERA_CUTOFF_EPOCH - 3600.0,
    )
    _write_scoreboard_report_with_mtime(
        tmp_path / "new_run.json",
        candidate="ckpt_new.pt",
        results=[{"opponent": "catanatron_ab3", "wins": 3, "games": 10, "track": "2p_no_trade"}],
        mtime=POSTFIX_BOTS_ERA_CUTOFF_EPOCH + 3600.0,
    )
    matchups, _, era_file_counts = _load_matchups(tmp_path, repo_root=tmp_path, track="2p_no_trade")
    by_candidate = {m["candidate"]: m["era"] for m in matchups}
    assert by_candidate["ckpt_old.pt"] == "prefix-bots"
    assert by_candidate["ckpt_new.pt"] == "postfix-bots"
    assert era_file_counts == {"prefix-bots": 1, "postfix-bots": 1}


def test_load_matchups_era_filter_postfix_bots_excludes_older_file(tmp_path: Path) -> None:
    _write_scoreboard_report_with_mtime(
        tmp_path / "old_run.json",
        candidate="ckpt_old.pt",
        results=[{"opponent": "catanatron_ab3", "wins": 6, "games": 10, "track": "2p_no_trade"}],
        mtime=POSTFIX_BOTS_ERA_CUTOFF_EPOCH - 3600.0,
    )
    _write_scoreboard_report_with_mtime(
        tmp_path / "new_run.json",
        candidate="ckpt_new.pt",
        results=[{"opponent": "catanatron_ab3", "wins": 3, "games": 10, "track": "2p_no_trade"}],
        mtime=POSTFIX_BOTS_ERA_CUTOFF_EPOCH + 3600.0,
    )
    matchups, _, _ = _load_matchups(
        tmp_path, repo_root=tmp_path, track="2p_no_trade", era_filter="postfix-bots"
    )
    assert [m["candidate"] for m in matchups] == ["ckpt_new.pt"]


def test_load_matchups_era_filter_prefix_bots_excludes_newer_file(tmp_path: Path) -> None:
    _write_scoreboard_report_with_mtime(
        tmp_path / "old_run.json",
        candidate="ckpt_old.pt",
        results=[{"opponent": "catanatron_ab3", "wins": 6, "games": 10, "track": "2p_no_trade"}],
        mtime=POSTFIX_BOTS_ERA_CUTOFF_EPOCH - 3600.0,
    )
    _write_scoreboard_report_with_mtime(
        tmp_path / "new_run.json",
        candidate="ckpt_new.pt",
        results=[{"opponent": "catanatron_ab3", "wins": 3, "games": 10, "track": "2p_no_trade"}],
        mtime=POSTFIX_BOTS_ERA_CUTOFF_EPOCH + 3600.0,
    )
    matchups, _, _ = _load_matchups(
        tmp_path, repo_root=tmp_path, track="2p_no_trade", era_filter="prefix-bots"
    )
    assert [m["candidate"] for m in matchups] == ["ckpt_old.pt"]


def test_load_matchups_rejects_unknown_era_filter(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        _load_matchups(tmp_path, repo_root=tmp_path, track="2p_no_trade", era_filter="bogus")


# --------------------------------------------------------------------------- truncation-as-loss bias
# Regression tests for the adversarial-review finding: a truncated game
# (no winner) must never be silently folded into "loss" via bool(None).


def test_matchup_record_preserves_none_for_truncated_games() -> None:
    record = _matchup_record(
        {"opponent": "catanatron_ab3", "wins": 3, "games": 5, "game_outcomes": [True, None, False, True, None]},
        "ckpt.pt",
    )
    assert record["game_outcomes"] == [True, None, False, True, None]


def test_build_win_tables_excludes_truncated_games_from_bootstrap_outcomes() -> None:
    matchups = [
        {
            "candidate": "ckpt.pt",
            "opponent": "catanatron_ab3",
            "wins": 2,
            "games": 4,
            "game_outcomes": [True, None, False, True],
        }
    ]
    _, outcomes = _build_win_tables(matchups)
    # The None (truncated) entry must be dropped, not converted to False.
    assert outcomes["ckpt.pt"]["catanatron_ab3"] == [True, False, True]


def test_build_win_tables_wins_and_games_counts_are_unaffected_by_truncation_filtering() -> None:
    # The aggregate wins/games tally (used by the main, non-bootstrap fit)
    # comes from evaluate_scoreboard.py's own counters, not from
    # re-deriving from game_outcomes -- confirm it stays untouched here.
    matchups = [
        {
            "candidate": "ckpt.pt",
            "opponent": "catanatron_ab3",
            "wins": 2,
            "games": 4,
            "game_outcomes": [True, None, False, True],
        }
    ]
    wins, _ = _build_win_tables(matchups)
    assert wins["ckpt.pt"]["catanatron_ab3"] == 2
    assert wins["catanatron_ab3"]["ckpt.pt"] == 2  # games(4) - wins(2)
