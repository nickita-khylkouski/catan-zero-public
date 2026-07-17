from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from tools.grade_agent import (
    _evaluate_suite,
    build_eval_command,
    grade_checkpoint,
    grade_decision,
    GRADE_PROFILES,
    paired_deltas,
    summarize_reports,
    wilson_interval,
)
from tools.train_ppo import _make_opponent, _make_teacher
def test_wilson_interval_is_conservative() -> None:
    lower, upper = wilson_interval(8, 32)

    assert 0.0 < lower < 0.25 < upper < 1.0


def test_summarize_reports_weights_value_opponent_more_heavily() -> None:
    summary = summarize_reports(
        {
            "heuristic": [{"wins": 8, "games": 16}],
            "value": [{"wins": 2, "games": 16}],
        },
        weights={"heuristic": 1.0, "value": 3.0},
    )

    assert summary["opponents"]["heuristic"]["win_rate"] == 0.5
    assert summary["opponents"]["value"]["win_rate"] == 0.125
    assert summary["weighted_win_rate"] == ((0.5 * 1.0) + (0.125 * 3.0)) / 4.0


def test_grade_checkpoint_early_rejects_zero_win_candidate(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_evaluate_suite(*, checkpoint, label, **kwargs):
        calls.append(label)
        assert label == "candidate"
        return {"heuristic": [{"wins": 0, "games": 2}]}

    monkeypatch.setattr("tools.grade_agent._evaluate_suite", fake_evaluate_suite)

    row = grade_checkpoint(
        checkpoint=Path("candidate.pt"),
        champion=Path("champion.pt"),
        eval_dir=tmp_path,
        opponents=("heuristic",),
        weights={"heuristic": 1.0},
        games=2,
        repeats=1,
        seed_base=91000,
        workers=1,
        vps_to_win=4,
        max_decisions=220,
        leg_timeout_seconds=600,
        min_aggregate_delta=0.0,
        max_opponent_regression=0.0,
        dry_run=False,
    )

    assert calls == ["candidate"]
    assert row["decision"] == "reject"
    assert row["early_reject"] is True
    assert row["champion_summary"] is None


def test_grade_checkpoint_stops_candidate_suite_after_timeout(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_evaluate_suite(*, checkpoint, label, stop_on_timeout=False, **kwargs):
        calls.append((label, stop_on_timeout))
        assert label == "candidate"
        assert stop_on_timeout is True
        return {"jsettlers_lite": [{"wins": 0, "games": 2, "timed_out": True}]}

    monkeypatch.setattr("tools.grade_agent._evaluate_suite", fake_evaluate_suite)

    row = grade_checkpoint(
        checkpoint=Path("candidate.pt"),
        champion=Path("champion.pt"),
        eval_dir=tmp_path,
        opponents=("jsettlers_lite", "heuristic"),
        weights={"jsettlers_lite": 3.0, "heuristic": 1.0},
        games=2,
        repeats=1,
        seed_base=91000,
        workers=1,
        vps_to_win=4,
        max_decisions=220,
        leg_timeout_seconds=600,
        min_aggregate_delta=0.0,
        max_opponent_regression=0.0,
        dry_run=False,
    )

    assert calls == [("candidate", True)]
    assert row["decision"] == "reject"
    assert row["reason"] == "candidate timed out in 1 grade legs"
    assert row["champion_summary"] is None


def test_evaluate_suite_stop_on_timeout_skips_later_opponents(monkeypatch, tmp_path) -> None:
    commands = []

    def fake_run(command, *, timeout_seconds=0):
        commands.append(command)
        output = Path(command[command.index("--output") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "wins": 0,
                    "games": 2,
                    "timed_out": True,
                    "opponent": command[command.index("--opponent") + 1],
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr("tools.grade_agent.run", fake_run)

    reports = _evaluate_suite(
        checkpoint=Path("candidate.pt"),
        label="candidate",
        eval_dir=tmp_path,
        opponents=("jsettlers_lite", "heuristic"),
        games=2,
        repeats=1,
        seed_base=91000,
        workers=1,
        vps_to_win=4,
        max_decisions=220,
        leg_timeout_seconds=600,
        dry_run=False,
        stop_on_timeout=True,
    )

    assert list(reports) == ["jsettlers_lite"]
    assert len(commands) == 1
    assert commands[0][commands[0].index("--opponent") + 1] == "jsettlers_lite"


def test_grade_decision_rejects_opponent_regression() -> None:
    candidate = summarize_reports(
        {
            "heuristic": [{"wins": 9, "games": 16}],
            "value": [{"wins": 1, "games": 16}],
        },
        weights={"heuristic": 1.0, "value": 1.0},
    )
    champion = summarize_reports(
        {
            "heuristic": [{"wins": 7, "games": 16}],
            "value": [{"wins": 4, "games": 16}],
        },
        weights={"heuristic": 1.0, "value": 1.0},
    )

    decision, reason = grade_decision(
        candidate,
        champion,
        min_aggregate_delta=0.0,
        max_opponent_regression=0.0,
    )

    assert decision == "reject"
    assert "value" in reason


def test_grade_decision_keeps_positive_no_regression_candidate() -> None:
    candidate = summarize_reports(
        {
            "heuristic": [{"wins": 8, "games": 16}],
            "value": [{"wins": 5, "games": 16}],
        },
        weights={"heuristic": 1.0, "value": 1.0},
    )
    champion = summarize_reports(
        {
            "heuristic": [{"wins": 8, "games": 16}],
            "value": [{"wins": 4, "games": 16}],
        },
        weights={"heuristic": 1.0, "value": 1.0},
    )

    decision, reason = grade_decision(
        candidate,
        champion,
        min_aggregate_delta=0.0,
        max_opponent_regression=0.0,
    )

    assert decision in {"keep_for_training", "promote_candidate"}
    assert "aggregate" in reason


def test_paired_deltas_surface_worst_opponent() -> None:
    candidate = summarize_reports(
        {
            "heuristic": [{"wins": 8, "games": 16}],
            "value": [{"wins": 3, "games": 16}],
        },
        weights={"heuristic": 1.0, "value": 1.0},
    )
    champion = summarize_reports(
        {
            "heuristic": [{"wins": 6, "games": 16}],
            "value": [{"wins": 4, "games": 16}],
        },
        weights={"heuristic": 1.0, "value": 1.0},
    )

    deltas = paired_deltas(candidate, champion)

    assert deltas["aggregate_delta"] == candidate["weighted_win_rate"] - champion["weighted_win_rate"]
    assert deltas["worst_opponent"] == "value"
    assert deltas["opponents"]["heuristic"]["win_rate_delta"] == 0.125
    assert deltas["opponents"]["value"]["win_rate_delta"] == -0.0625


def test_grade_decision_rejects_timed_out_candidate() -> None:
    candidate = summarize_reports(
        {
            "heuristic": [{"wins": 8, "games": 16, "timed_out": True}],
            "value": [{"wins": 8, "games": 16}],
        },
        weights={"heuristic": 1.0, "value": 1.0},
    )
    champion = summarize_reports(
        {
            "heuristic": [{"wins": 4, "games": 16}],
            "value": [{"wins": 4, "games": 16}],
        },
        weights={"heuristic": 1.0, "value": 1.0},
    )

    decision, reason = grade_decision(
        candidate,
        champion,
        min_aggregate_delta=0.0,
        max_opponent_regression=0.0,
    )

    assert decision == "reject"
    assert "timed out" in reason


def test_grader_eval_command_matches_evaluator_cli(tmp_path: Path) -> None:
    command = build_eval_command(
        checkpoint=tmp_path / "candidate.pt",
        opponent="value",
        games=12,
        seed=44,
        vps_to_win=4,
        max_decisions=300,
        workers=2,
        output=tmp_path / "out.json",
    )

    assert command[1:] == [
        "tools/evaluate_self_play.py",
        "--candidate",
        "ppo",
        "--checkpoint",
        str(tmp_path / "candidate.pt"),
        "--opponent",
        "value",
        "--games",
        "12",
        "--seed",
        "44",
        "--vps-to-win",
        "4",
        "--max-decisions",
        "300",
        "--workers",
        "2",
        "--output",
        str(tmp_path / "out.json"),
    ]


def test_grader_eval_command_accepts_value_rollout_opponent(tmp_path: Path) -> None:
    command = build_eval_command(
        checkpoint=tmp_path / "candidate.pt",
        opponent="value_rollout",
        games=4,
        seed=45,
        vps_to_win=4,
        max_decisions=300,
        workers=1,
        output=tmp_path / "out.json",
    )

    assert command[command.index("--opponent") + 1] == "value_rollout"
    assert command[command.index("--opponent-candidate-limit") + 1] == "24"
    assert command[command.index("--opponent-rollout-decisions") + 1] == "3"
    assert command[command.index("--opponent-value-penalty") + 1] == "0.0"


def test_grader_eval_command_forwards_custom_opponent_search_params(
    tmp_path: Path,
) -> None:
    command = build_eval_command(
        checkpoint=tmp_path / "candidate.pt",
        opponent="value_rollout",
        games=4,
        seed=45,
        vps_to_win=4,
        max_decisions=300,
        workers=1,
        output=tmp_path / "out.json",
        opponent_candidate_limit=20,
        opponent_rollout_decisions=5,
        opponent_value_penalty=0.02,
    )

    assert command[command.index("--opponent-candidate-limit") + 1] == "20"
    assert command[command.index("--opponent-rollout-decisions") + 1] == "5"
    assert command[command.index("--opponent-value-penalty") + 1] == "0.02"


def test_grade_profiles_include_search_stress_opponent() -> None:
    strict = GRADE_PROFILES["strict"]
    triage = GRADE_PROFILES["jsettlers_triage"]
    search_stress = GRADE_PROFILES["search_stress"]

    assert "jsettlers_lite" in strict["opponents"]
    assert strict["weights"]["jsettlers_lite"] > strict["weights"]["heuristic"]
    assert "value_rollout" in strict["opponents"]
    assert strict["weights"]["value_rollout"] > strict["weights"]["heuristic"]
    assert triage["opponents"] == ("jsettlers_lite", "heuristic")
    assert triage["weights"]["jsettlers_lite"] > triage["weights"]["heuristic"]
    assert search_stress["opponents"] == ("value", "value_rollout")


def test_train_opponent_factory_supports_value_rollout() -> None:
    import numpy as np

    opponent = _make_opponent(
        "value_rollout",
        np.random.default_rng(1),
        value_candidate_limit=12,
        value_opponent_penalty=0.0,
    )

    assert opponent.name == "value_rollout_search"


def test_train_opponent_factory_supports_jsettlers_lite() -> None:
    import numpy as np

    opponent = _make_opponent(
        "jsettlers_lite",
        np.random.default_rng(1),
        value_candidate_limit=12,
        value_opponent_penalty=0.0,
    )

    assert opponent.name == "jsettlers_lite"


def test_train_opponent_factory_supports_search_mixed() -> None:
    import numpy as np

    seen = {
        _make_opponent(
            "search_mixed",
            np.random.default_rng(seed),
            value_candidate_limit=12,
            value_opponent_penalty=0.0,
        ).name
        for seed in range(20)
    }

    assert "heuristic" in seen
    assert "catanatron_value" in seen
    assert "value_rollout_search" in seen


def test_train_opponent_factory_supports_strict_mixed() -> None:
    import numpy as np

    seen = {
        _make_opponent(
            "strict_mixed",
            np.random.default_rng(seed),
            value_candidate_limit=12,
            value_opponent_penalty=0.0,
        ).name
        for seed in range(40)
    }

    assert "heuristic" in seen
    assert "jsettlers_lite" in seen
    assert "catanatron_value" in seen
    assert "value_rollout_search" in seen


def test_train_opponent_factory_supports_anti_regression_mixed() -> None:
    import numpy as np

    seen = {
        _make_opponent(
            "anti_regression_mixed",
            np.random.default_rng(seed),
            value_candidate_limit=12,
            value_opponent_penalty=0.0,
        ).name
        for seed in range(40)
    }

    assert "heuristic" in seen
    assert "jsettlers_lite" in seen
    assert "catanatron_value" in seen


def test_train_opponent_factory_supports_pfsp_mixed() -> None:
    import numpy as np

    seen = {
        _make_opponent(
            "pfsp_mixed",
            np.random.default_rng(seed),
            value_candidate_limit=12,
            value_opponent_penalty=0.0,
        ).name
        for seed in range(80)
    }

    assert "heuristic" in seen
    assert "jsettlers_lite" in seen
    assert "catanatron_value" in seen
    assert "value_rollout_search" in seen


def test_train_teacher_factory_supports_gate_baselines() -> None:
    base = dict(
        teacher_candidate_limit=12,
        teacher_opponent_penalty=0.0,
        teacher_temperature=0.7,
        teacher_presearch_candidate_limit=0,
        teacher_rollout_decisions=1,
        teacher_rollout_samples=1,
        teacher_root_value_weight=0.0,
    )

    assert _make_teacher(argparse.Namespace(**base, teacher="heuristic")).name == "heuristic"
    assert (
        _make_teacher(argparse.Namespace(**base, teacher="jsettlers_lite")).name
        == "jsettlers_lite"
    )
    assert (
        _make_teacher(argparse.Namespace(**base, teacher="baseline_mixed")).name
        == "baseline_mixed"
    )
