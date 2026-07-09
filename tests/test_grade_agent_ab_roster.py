from __future__ import annotations

import json
from pathlib import Path

from tools.grade_agent import (
    EVALUATE_SELF_PLAY_OPPONENTS,
    GRADE_PROFILES,
    _normalize_scoreboard_report,
    build_eval_command,
    build_scoreboard_eval_command,
)


def test_dev_profile_includes_ab_bots_by_default() -> None:
    dev = GRADE_PROFILES["dev"]
    assert "catanatron_ab3" in dev["opponents"]
    assert "catanatron_ab4" in dev["opponents"]
    assert "catanatron_ab5" in dev["opponents"]
    # heuristic/value stayed in the roster too (not replaced, augmented).
    assert "heuristic" in dev["opponents"]
    assert "value" in dev["opponents"]


def test_ab_bots_are_not_in_the_evaluate_self_play_vocabulary() -> None:
    # Documents why AB opponents must be routed to evaluate_scoreboard.py:
    # tools/evaluate_self_play.py's own --opponent parser has no AB-bot choice.
    assert "catanatron_ab3" not in EVALUATE_SELF_PLAY_OPPONENTS
    assert "catanatron_ab4" not in EVALUATE_SELF_PLAY_OPPONENTS
    assert "catanatron_ab5" not in EVALUATE_SELF_PLAY_OPPONENTS
    assert "value" in EVALUATE_SELF_PLAY_OPPONENTS  # unchanged existing opponent


def test_build_eval_command_still_uses_evaluate_self_play_for_legacy_opponents(tmp_path: Path) -> None:
    command = build_eval_command(
        checkpoint=tmp_path / "candidate.pt",
        opponent="value",
        games=12,
        seed=44,
        vps_to_win=10,
        max_decisions=300,
        workers=2,
        output=tmp_path / "out.json",
    )
    assert command[1] == "tools/evaluate_self_play.py"
    assert command[command.index("--opponent") + 1] == "value"


def test_build_eval_command_routes_ab_bot_through_evaluate_scoreboard(tmp_path: Path) -> None:
    command = build_eval_command(
        checkpoint=tmp_path / "candidate.pt",
        opponent="catanatron_ab3",
        games=12,
        seed=44,
        vps_to_win=10,
        max_decisions=300,
        workers=2,
        output=tmp_path / "out.json",
    )
    assert command[1] == "tools/evaluate_scoreboard.py"
    assert command[command.index("--opponents") + 1] == "catanatron_ab3"
    assert command[command.index("--candidate") + 1] == str(tmp_path / "candidate.pt")
    assert command[command.index("--vps-to-win") + 1] == "10"


def test_build_scoreboard_eval_command_forwards_all_leg_params(tmp_path: Path) -> None:
    command = build_scoreboard_eval_command(
        checkpoint=tmp_path / "candidate.pt",
        opponent="catanatron_ab5",
        games=8,
        seed=99,
        vps_to_win=10,
        max_decisions=500,
        workers=3,
        output=tmp_path / "out.json",
    )
    assert command[command.index("--games") + 1] == "8"
    assert command[command.index("--seed") + 1] == "99"
    assert command[command.index("--max-decisions") + 1] == "500"
    assert command[command.index("--workers") + 1] == "3"
    assert command[command.index("--out") + 1] == str(tmp_path / "out.json")


def test_normalize_scoreboard_report_flattens_results_list(tmp_path: Path) -> None:
    output = tmp_path / "out.json"
    output.write_text(
        json.dumps(
            {
                "candidate": "runs/foo/checkpoint.pt",
                "results": [
                    {
                        "opponent": "catanatron_ab3",
                        "wins": 6,
                        "games": 12,
                        "win_rate": 0.5,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    _normalize_scoreboard_report(output)
    flat = json.loads(output.read_text(encoding="utf-8"))
    assert flat["wins"] == 6
    assert flat["games"] == 12
    assert flat["opponent"] == "catanatron_ab3"
    assert flat["candidate"] == "runs/foo/checkpoint.pt"
    assert "results" not in flat


def test_normalize_scoreboard_report_is_a_noop_for_already_flat_reports(tmp_path: Path) -> None:
    output = tmp_path / "out.json"
    flat_report = {"candidate": "ppo", "opponent": "value", "wins": 4, "games": 12, "win_rate": 1 / 3}
    output.write_text(json.dumps(flat_report), encoding="utf-8")
    _normalize_scoreboard_report(output)
    assert json.loads(output.read_text(encoding="utf-8")) == flat_report
