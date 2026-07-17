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
from tools.gcp_fleet_controller import (
    build_grade_ready_command,
    build_remote_grade_from_worker_command,
    build_remote_grade_command,
    build_remote_opening_eval_command,
    build_remote_sync_code_launch_command,
    build_remote_stop_grade_command,
    build_remote_stop_train_command,
    build_remote_reanalysis_train_command,
    build_remote_train_command,
    build_planned_training_args,
    build_warmup_only_training_args,
    candidate_stems,
    checkpoint_family_name,
    checkpoint_run_number,
    is_full_default_worker_poll,
    is_candidate_checkpoint_name,
    is_train_process_line,
    local_controller_status,
    missing_remote_features,
    next_training_seed,
    normalize_min_run_number,
    plan_remote_escalations,
    plan_remote_code_sync,
    plan_remote_gates,
    plan_remote_opening_evals,
    plan_remote_transfer_gates,
    plan_remote_reanalysis_train,
    plan_remote_train,
    remote_grade_status_command,
    remote_grade_run_id,
    remote_poll_command,
    remote_sync_preflight_command,
    required_remote_features_for_recipe,
    run_command_with_retries,
    select_local_checkpoints,
    summarize_remote_grade_status,
    Worker,
)


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


def test_gcp_grade_ready_command_forwards_opponent_search_params(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "candidate.pt"
    args = argparse.Namespace(
        champion="champion.pt",
        eval_dir="grades",
        profile="strict",
        games=8,
        repeats=2,
        workers=4,
        leg_timeout_seconds=600,
        opponent=["value_rollout"],
        opponent_weight=["value_rollout=2.0"],
        opponent_candidate_limit=20,
        opponent_rollout_decisions=5,
        opponent_value_penalty=0.02,
        vps_to_win=4,
        max_decisions=300,
        dry_run=True,
    )

    command = build_grade_ready_command(args, [checkpoint])

    assert command[1:20] == [
        "tools/grade_agent.py",
        "--champion",
        "champion.pt",
        "--eval-dir",
        "grades",
        "--profile",
        "strict",
        "--games",
        "8",
        "--repeats",
        "2",
        "--workers",
        "4",
        "--leg-timeout-seconds",
        "600",
        "--vps-to-win",
        "4",
        "--max-decisions",
        "300",
    ]
    assert command[command.index("--opponent") + 1] == "value_rollout"
    assert command[command.index("--opponent-weight") + 1] == "value_rollout=2.0"
    assert command[command.index("--opponent-candidate-limit") + 1] == "20"
    assert command[command.index("--opponent-rollout-decisions") + 1] == "5"
    assert command[command.index("--opponent-value-penalty") + 1] == "0.02"
    assert command[command.index("--checkpoint") + 1] == str(checkpoint)
    assert command[-1] == "--dry-run"


def test_gcp_remote_grade_command_detaches_and_writes_summary() -> None:
    args = argparse.Namespace(
        remote_repo="~/catan-zero",
        checkpoint="runs/self_play/candidate.iter0001.pt",
        champion="runs/self_play/champions/current_best_s4806_iter0002.pt",
        eval_dir="runs/self_play/remote_grades_reanalysis",
        log_dir="runs/self_play/logs",
        profile="strict",
        games=12,
        repeats=1,
        workers=6,
        vps_to_win=4,
        max_decisions=300,
        leg_timeout_seconds=900,
        force=False,
    )

    command = build_remote_grade_command(args)
    run_id = remote_grade_run_id(args)

    assert "setsid sh -c" in command
    assert "--profile strict" in command
    assert f"--summary-output runs/self_play/remote_grades_reanalysis/summary_{run_id}.json" in command
    assert "--checkpoint runs/self_play/candidate.iter0001.pt" in command
    assert "--leg-timeout-seconds 900" in command
    assert "repo='~/catan-zero'" in command
    assert "summary_exists" in command
    assert "already_active" in command
    assert "pgrep -af '[t]ools/grade_agent.py'" in command
    assert '>> "$log_path"' not in command


def test_gcp_remote_grade_run_id_separates_gate_configs() -> None:
    base = dict(
        remote_repo="~/catan-zero",
        checkpoint="runs/self_play/candidate.iter0001.pt",
        champion="runs/self_play/champions/current_best_s4806_iter0002.pt",
        eval_dir="runs/self_play/remote_grades_reanalysis",
        log_dir="runs/self_play/logs",
        games=4,
        repeats=1,
        workers=1,
        vps_to_win=4,
        max_decisions=300,
        leg_timeout_seconds=1200,
        force=False,
    )
    dev = argparse.Namespace(**base, profile="dev")
    strict = argparse.Namespace(**base, profile="strict")

    assert remote_grade_run_id(dev) != remote_grade_run_id(strict)
    assert "_dev_" in remote_grade_run_id(dev)
    assert "_strict_" in remote_grade_run_id(strict)


def test_gcp_remote_grade_force_skips_idempotency_guard() -> None:
    args = argparse.Namespace(
        remote_repo="~/catan-zero",
        checkpoint="runs/self_play/candidate.iter0001.pt",
        champion="runs/self_play/champions/current_best_s4806_iter0002.pt",
        eval_dir="runs/self_play/remote_grades_reanalysis",
        log_dir="runs/self_play/logs",
        profile="strict",
        games=12,
        repeats=1,
        workers=6,
        vps_to_win=4,
        max_decisions=300,
        leg_timeout_seconds=900,
        force=True,
    )

    command = build_remote_grade_command(args)

    assert "summary_exists" not in command
    assert "already_active" not in command


def test_gcp_remote_train_command_detaches_and_forwards_args() -> None:
    args = argparse.Namespace(
        remote_repo="~/catan-zero",
        label="s9750/vrpo sarsa",
        log_dir="runs/self_play/logs",
        force=False,
        train_args=[
            "--",
            "--seed",
            "9750",
            "--iterations",
            "2",
            "--checkpoint",
            "runs/self_play/s9750.pt",
            "--report",
            "runs/self_play/s9750.json",
        ],
    )

    command = build_remote_train_command(args)

    assert "setsid sh -c" in command
    assert "tools/train_ppo.py --seed 9750 --iterations 2" in command
    assert "--checkpoint runs/self_play/s9750.pt" in command
    assert "runs/self_play/logs/s9750_vrpo_sarsa.log" in command
    assert "checkpoint_path=runs/self_play/s9750.pt" in command
    assert "report_path=runs/self_play/s9750.json" in command
    assert "artifact_exists" in command


def test_gcp_remote_train_force_skips_artifact_and_busy_guards() -> None:
    args = argparse.Namespace(
        remote_repo="~/catan-zero",
        label="s9750_force",
        log_dir="runs/self_play/logs",
        force=True,
        train_args=[
            "--",
            "--seed",
            "9750",
            "--checkpoint",
            "runs/self_play/s9750.pt",
            "--report",
            "runs/self_play/s9750.json",
        ],
    )

    command = build_remote_train_command(args)

    assert "artifact_exists" not in command
    assert "worker_busy" not in command
    assert "repo='~/catan-zero'" in command


def test_gcp_remote_train_force_skips_busy_guard() -> None:
    args = argparse.Namespace(
        remote_repo="~/catan-zero",
        label="s9750_force",
        log_dir="runs/self_play/logs",
        force=True,
        train_args=[
            "--",
            "--seed",
            "9750",
            "--iterations",
            "2",
            "--checkpoint",
            "runs/self_play/s9750.pt",
        ],
    )

    command = build_remote_train_command(args)

    assert "setsid sh -c" in command
    assert "worker_busy" not in command
    assert "pgrep -af '[t]ools/train_ppo.py'" not in command


def test_gcp_remote_reanalysis_train_generates_then_trains_on_vm() -> None:
    args = argparse.Namespace(
        remote_repo="~/catan-zero",
        label="s10100/dags midgame",
        champion="runs/self_play/champions/current_best_s9752_iter0002.pt",
        log_dir="runs/self_play/logs",
        seed=10100,
        games=8,
        vps_to_win=4,
        max_decisions=300,
        record_after_decisions=40,
        record_window_decisions=120,
        candidate_limit=24,
        presearch_candidate_limit=48,
        rollout_decisions=2,
        rollout_samples=1,
        root_value_weight=0.25,
        temperature=0.55,
        reanalysis_max_samples=2048,
        reanalysis_epochs=2,
        reanalysis_value_coef=0.35,
        reanalysis_score_coef=0.05,
        force=False,
    )

    command = build_remote_reanalysis_train_command(args)

    assert "tools/generate_reanalysis.py --output runs/self_play/s10100_dags_midgame.jsonl" in command
    assert "--record-after-decisions 40 --record-window-decisions 120" in command
    assert "&& .venv/bin/python -u tools/train_ppo.py --seed 10100" in command
    assert "--reanalysis-input runs/self_play/s10100_dags_midgame.jsonl" in command
    assert "--reanalysis-checkpoint runs/self_play/s10100_dags_midgame.pt" in command
    assert "pgrep -af '[t]ools/generate_reanalysis.py'" in command
    assert "artifact_exists" in command


def test_gcp_remote_opening_eval_command_runs_vm_side_diagnostic() -> None:
    args = argparse.Namespace(
        remote_repo="~/catan-zero",
        checkpoint="runs/self_play/s10120_candidate.iter0002.pt",
        output_dir="runs/self_play/remote_opening_evals",
        log_dir="runs/self_play/logs",
        games=16,
        seed=93000,
        vps_to_win=10,
        max_opening_decisions=16,
        candidate_limit=96,
        presearch_candidate_limit=96,
        rollout_decisions=2,
        rollout_samples=1,
        root_value_weight=0.35,
        opponent_penalty=0.05,
        force=False,
    )

    command = build_remote_opening_eval_command(args)

    assert "tools/evaluate_openings.py --candidate ppo" in command
    assert "--checkpoint runs/self_play/s10120_candidate.iter0002.pt" in command
    assert "--teachers value value_rollout" in command
    assert "--max-opening-decisions 16" in command
    assert "--output runs/self_play/remote_opening_evals/opening_s10120_candidate.iter0002_g16_vp10_d16_seed93000.json" in command
    assert "pgrep -af '[t]ools/evaluate_openings.py'" in command
    assert "artifact_exists" in command


def test_plan_remote_train_avoids_busy_workers_and_defaults_to_warmup_command() -> None:
    poll_payload = {
        "workers": [
            {
                "worker": "catan-zero-c1",
                "zone": "us-central1-c",
                "ok": True,
                "running_train_processes": 1,
                "processes": [
                    {
                        "seed": "9799",
                        "checkpoint": "runs/self_play/s9799_graph_history_teacher_w1a.pt",
                    }
                ],
                "candidate_checkpoints": [
                    {"name": "s9799_graph_history_teacher_w1a.iter0002.pt", "size": 20}
                ],
            },
            {
                "worker": "catan-zero-c2",
                "zone": "us-central1-c",
                "ok": True,
                "running_train_processes": 0,
                "candidate_checkpoints": [],
                "trainer_features": {
                    "ema_policy_kl": False,
                    "old_policy_kl": False,
                    "pfsp_mixed": False,
                },
            },
        ],
    }
    summary_payload = {"active": [], "decisions": []}

    plan = plan_remote_train(
        poll_payload=poll_payload,
        summary_payload=summary_payload,
        project="proj",
        remote_repo="",
        champion="runs/self_play/champions/current_best_s9752_iter0002.pt",
        recipe="warmup_baseline",
        seed=9800,
        min_seed=9800,
        iterations=10,
        episodes_per_iteration=8,
        checkpoint_every=2,
        max_launches=1,
    )

    assert plan["planned_count"] == 1
    assert plan["planned"][0]["seed"] == 9800
    assert plan["planned"][0]["label"] == "s9800_warmup_baseline_c2"
    assert "--teacher baseline_mixed" in plan["planned"][0]["shell"]
    assert "--iterations 0" in plan["planned"][0]["shell"]
    assert "--opponents pfsp_mixed" not in plan["planned"][0]["shell"]
    assert plan["skipped"]["busy_worker"][0]["worker"] == "catan-zero-c1"


def test_plan_remote_train_blocks_workers_without_required_features() -> None:
    poll_payload = {
        "workers": [
            {
                "worker": "catan-zero-w4a",
                "zone": "us-west4-a",
                "ok": True,
                "running_train_processes": 0,
                "candidate_checkpoints": [],
                "repo": "/home/worker/catan-zero",
                "trainer_sha1": "abc123",
                "trainer_features": {
                    "ema_policy_kl": True,
                    "old_policy_kl": True,
                    "pfsp_mixed": False,
                },
            },
        ],
    }

    plan = plan_remote_train(
        poll_payload=poll_payload,
        summary_payload={"active": [], "decisions": []},
        project="proj",
        remote_repo="",
        champion="champion.pt",
        recipe="pfsp_value_jsettlers",
        seed=9805,
        min_seed=9800,
        iterations=10,
        episodes_per_iteration=8,
        checkpoint_every=2,
        max_launches=1,
    )

    assert plan["planned_count"] == 0
    assert plan["skipped"]["missing_remote_feature"][0]["worker"] == "catan-zero-w4a"
    assert plan["skipped"]["missing_remote_feature"][0]["missing"] == ["pfsp_mixed"]


def test_plan_remote_reanalysis_train_targets_clean_featured_workers() -> None:
    poll_payload = {
        "workers": [
            {
                "worker": "catan-zero-c1",
                "zone": "us-central1-c",
                "ok": True,
                "running_train_processes": 1,
                "running_reanalysis_processes": 0,
                "candidate_checkpoints": [],
                "trainer_features": {
                    "reanalysis_training": True,
                    "reanalysis_decision_windows": True,
                },
            },
            {
                "worker": "catan-zero-c2",
                "zone": "us-central1-c",
                "ok": True,
                "running_train_processes": 0,
                "running_reanalysis_processes": 0,
                "candidate_checkpoints": [],
                "trainer_features": {
                    "reanalysis_training": True,
                    "reanalysis_decision_windows": True,
                },
            },
        ],
    }

    plan = plan_remote_reanalysis_train(
        poll_payload=poll_payload,
        summary_payload={"active": [], "decisions": []},
        local_status_payload=None,
        project="proj",
        remote_repo="",
        champion="champion.pt",
        seed=10100,
        min_seed=10100,
        max_launches=1,
        games=8,
        vps_to_win=4,
        max_decisions=300,
        record_after_decisions=40,
        record_window_decisions=120,
        candidate_limit=24,
        presearch_candidate_limit=48,
        rollout_decisions=2,
        rollout_samples=1,
        root_value_weight=0.25,
        temperature=0.55,
        reanalysis_max_samples=2048,
        reanalysis_epochs=2,
        reanalysis_value_coef=0.35,
        reanalysis_score_coef=0.05,
    )

    assert plan["planned_count"] == 1
    assert plan["planned"][0]["worker"] == "catan-zero-c2"
    assert plan["planned"][0]["label"] == "s10100_dags_midgame_reanalysis_c2"
    assert "remote-reanalysis-train" in plan["planned"][0]["shell"]
    assert "--record-after-decisions 40" in plan["planned"][0]["shell"]
    assert plan["required_features"] == [
        "reanalysis_training",
        "reanalysis_decision_windows",
    ]
    assert plan["skipped"]["busy_worker"][0]["reason"] == "training"


def test_plan_remote_reanalysis_train_blocks_missing_dags_features() -> None:
    poll_payload = {
        "workers": [
            {
                "worker": "catan-zero-w4a",
                "zone": "us-west4-a",
                "ok": True,
                "running_train_processes": 0,
                "running_reanalysis_processes": 0,
                "candidate_checkpoints": [],
                "trainer_features": {
                    "reanalysis_training": True,
                    "reanalysis_decision_windows": False,
                },
            },
        ],
    }

    plan = plan_remote_reanalysis_train(
        poll_payload=poll_payload,
        summary_payload={"active": [], "decisions": []},
        local_status_payload=None,
        project="proj",
        remote_repo="",
        champion="champion.pt",
        seed=10100,
        min_seed=10100,
        max_launches=1,
        games=8,
        vps_to_win=4,
        max_decisions=300,
        record_after_decisions=40,
        record_window_decisions=120,
        candidate_limit=24,
        presearch_candidate_limit=48,
        rollout_decisions=2,
        rollout_samples=1,
        root_value_weight=0.25,
        temperature=0.55,
        reanalysis_max_samples=2048,
        reanalysis_epochs=2,
        reanalysis_value_coef=0.35,
        reanalysis_score_coef=0.05,
    )

    assert plan["planned_count"] == 0
    assert plan["skipped"]["missing_remote_feature"][0]["missing"] == [
        "reanalysis_decision_windows"
    ]


def test_plan_remote_opening_evals_targets_clean_featured_worker() -> None:
    poll_payload = {
        "workers": [
            {
                "worker": "catan-zero-c1",
                "zone": "us-central1-c",
                "ok": True,
                "running_train_processes": 1,
                "candidate_checkpoints": [
                    {"name": "s10100_candidate_c1.iter0002.pt", "size": 20}
                ],
                "trainer_features": {"opening_evaluator": True},
            },
            {
                "worker": "catan-zero-c2",
                "zone": "us-central1-c",
                "ok": True,
                "running_train_processes": 0,
                "candidate_checkpoints": [
                    {"name": "s10101_candidate_c2.iter0002.pt", "size": 20}
                ],
                "trainer_features": {"opening_evaluator": True},
            },
        ],
    }

    plan = plan_remote_opening_evals(
        poll_payload=poll_payload,
        summary_payload={"active": [], "decisions": []},
        local_status_payload=None,
        project="proj",
        remote_repo="",
        run_prefix="s",
        output_dir="runs/self_play/remote_opening_evals",
        log_dir="runs/self_play/logs",
        games=16,
        seed=93000,
        vps_to_win=10,
        max_opening_decisions=16,
        candidate_limit=96,
        presearch_candidate_limit=96,
        rollout_decisions=2,
        rollout_samples=1,
        root_value_weight=0.35,
        opponent_penalty=0.05,
        max_evals=1,
        max_per_family=1,
        include_interim=True,
        include_warmup=False,
        prefer_prefixes=["s101"],
        min_run_number=9900,
        allow_busy_workers=False,
        allow_unknown_remote_features=False,
    )

    assert plan["planned_count"] == 1
    assert plan["planned"][0]["worker"] == "catan-zero-c2"
    assert plan["planned"][0]["checkpoint"] == "s10101_candidate_c2.iter0002.pt"
    assert "remote-opening-eval" in plan["planned"][0]["shell"]
    assert plan["planned"][0]["output"].endswith(
        "opening_s10101_candidate_c2.iter0002_g16_vp10_d16_seed93000.json"
    )
    assert plan["required_features"] == ["opening_evaluator"]
    assert plan["skipped"]["busy_worker"][0]["worker"] == "catan-zero-c1"


def test_plan_remote_opening_evals_blocks_missing_feature() -> None:
    poll_payload = {
        "workers": [
            {
                "worker": "catan-zero-w4a",
                "zone": "us-west4-a",
                "ok": True,
                "running_train_processes": 0,
                "candidate_checkpoints": [
                    {"name": "s10101_candidate_w4a.iter0002.pt", "size": 20}
                ],
                "trainer_features": {"opening_evaluator": False},
            }
        ],
    }

    plan = plan_remote_opening_evals(
        poll_payload=poll_payload,
        summary_payload={"active": [], "decisions": []},
        local_status_payload=None,
        project="proj",
        remote_repo="",
        run_prefix="s",
        output_dir="runs/self_play/remote_opening_evals",
        log_dir="runs/self_play/logs",
        games=16,
        seed=93000,
        vps_to_win=10,
        max_opening_decisions=16,
        candidate_limit=96,
        presearch_candidate_limit=96,
        rollout_decisions=2,
        rollout_samples=1,
        root_value_weight=0.35,
        opponent_penalty=0.05,
        max_evals=1,
        max_per_family=1,
        include_interim=True,
        include_warmup=False,
        prefer_prefixes=[],
        min_run_number=9900,
        allow_busy_workers=False,
        allow_unknown_remote_features=False,
    )

    assert plan["planned_count"] == 0
    assert plan["skipped"]["missing_remote_feature"][0]["missing"] == [
        "opening_evaluator"
    ]


def test_plan_remote_train_can_use_grade_busy_worker_without_trainer() -> None:
    poll_payload = {
        "workers": [
            {
                "worker": "catan-zero-c1",
                "zone": "us-central1-c",
                "ok": True,
                "running_train_processes": 0,
                "candidate_checkpoints": [],
                "trainer_features": {},
            },
        ],
    }
    summary_payload = {
        "active": [
            {
                "worker": "catan-zero-c1",
                "checkpoint": "runs/self_play/s9861_candidate.iter0004.pt",
            }
        ],
        "decisions": [],
    }

    blocked = plan_remote_train(
        poll_payload=poll_payload,
        summary_payload=summary_payload,
        project="proj",
        remote_repo="",
        champion="champion.pt",
        recipe="warmup_baseline",
        seed=9876,
        min_seed=9800,
        iterations=10,
        episodes_per_iteration=8,
        checkpoint_every=2,
        max_launches=1,
    )
    allowed = plan_remote_train(
        poll_payload=poll_payload,
        summary_payload=summary_payload,
        project="proj",
        remote_repo="",
        champion="champion.pt",
        recipe="warmup_baseline",
        seed=9876,
        min_seed=9800,
        iterations=10,
        episodes_per_iteration=8,
        checkpoint_every=2,
        max_launches=1,
        allow_grade_busy_workers=True,
    )

    assert blocked["planned_count"] == 0
    assert blocked["skipped"]["busy_worker"][0]["reason"] == "remote_grade"
    assert allowed["planned_count"] == 1
    assert allowed["planned"][0]["worker"] == "catan-zero-c1"


def test_plan_remote_train_blocks_unknown_feature_status_by_default() -> None:
    poll_payload = {
        "workers": [
            {
                "worker": "catan-zero-w4a",
                "zone": "us-west4-a",
                "ok": True,
                "running_train_processes": 0,
                "candidate_checkpoints": [],
            },
        ],
    }

    plan = plan_remote_train(
        poll_payload=poll_payload,
        summary_payload={"active": [], "decisions": []},
        project="proj",
        remote_repo="",
        champion="champion.pt",
        recipe="pfsp_value_jsettlers",
        seed=9805,
        min_seed=9800,
        iterations=10,
        episodes_per_iteration=8,
        checkpoint_every=2,
        max_launches=1,
    )

    assert plan["planned_count"] == 0
    assert set(plan["skipped"]["missing_remote_feature"][0]["missing"]) == {
        "ema_policy_kl",
        "old_policy_kl",
        "pfsp_mixed",
    }


def test_plan_remote_train_blocks_automatic_seed_from_partial_poll() -> None:
    poll_payload = {
        "workers": [
            {
                "worker": "catan-zero-w1b",
                "zone": "us-west1-b",
                "ok": True,
                "running_train_processes": 0,
                "candidate_checkpoints": [],
            },
        ],
    }
    summary_payload = {
        "active": [],
        "decisions": [{"checkpoint": "runs/self_play/s9810_warmup_baseline_w4d.pt"}],
    }

    assert not is_full_default_worker_poll(poll_payload)
    blocked = plan_remote_train(
        poll_payload=poll_payload,
        summary_payload=summary_payload,
        project="proj",
        remote_repo="",
        champion="champion.pt",
        recipe="warmup_baseline",
        seed=0,
        min_seed=9800,
        iterations=10,
        episodes_per_iteration=8,
        checkpoint_every=2,
        max_launches=1,
    )

    assert blocked["planned_count"] == 0
    assert blocked["next_seed"] == 9811
    assert blocked["skipped"]["partial_poll"][0]["observed_workers"] == ["catan-zero-w1b"]

    explicit = plan_remote_train(
        poll_payload=poll_payload,
        summary_payload=summary_payload,
        project="proj",
        remote_repo="",
        champion="champion.pt",
        recipe="warmup_baseline",
        seed=9811,
        min_seed=9800,
        iterations=10,
        episodes_per_iteration=8,
        checkpoint_every=2,
        max_launches=1,
    )

    assert explicit["planned_count"] == 1
    assert explicit["planned"][0]["seed"] == 9811


def test_plan_remote_train_slots_automatic_seeds_by_worker() -> None:
    workers = [
        ("catan-zero-c1", "us-central1-c"),
        ("catan-zero-c2", "us-central1-c"),
        ("catan-zero-c3", "us-central1-c"),
        ("catan-zero-c4", "us-central1-c"),
        ("catan-zero-w1a", "us-west1-b"),
        ("catan-zero-w1b", "us-west1-b"),
        ("catan-zero-w4a", "us-west4-a"),
        ("catan-zero-w4b", "us-west4-a"),
        ("catan-zero-w4c", "us-west4-b"),
        ("catan-zero-w4d", "us-west4-b"),
    ]

    def make_poll(idle_worker: str) -> dict:
        return {
            "workers": [
                {
                    "worker": worker,
                    "zone": zone,
                    "ok": True,
                    "running_train_processes": 0 if worker == idle_worker else 1,
                    "candidate_checkpoints": (
                        [{"name": "s9899_existing_parent.pt"}]
                        if worker == "catan-zero-c1"
                        else []
                    ),
                    "processes": [],
                }
                for worker, zone in workers
            ]
        }

    c1_plan = plan_remote_train(
        poll_payload=make_poll("catan-zero-c1"),
        summary_payload={"active": [], "decisions": []},
        project="proj",
        remote_repo="",
        champion="champion.pt",
        recipe="warmup_baseline",
        seed=0,
        min_seed=9800,
        iterations=10,
        episodes_per_iteration=8,
        checkpoint_every=2,
        max_launches=1,
    )
    w4b_plan = plan_remote_train(
        poll_payload=make_poll("catan-zero-w4b"),
        summary_payload={"active": [], "decisions": []},
        project="proj",
        remote_repo="",
        champion="champion.pt",
        recipe="warmup_baseline",
        seed=0,
        min_seed=9800,
        iterations=10,
        episodes_per_iteration=8,
        checkpoint_every=2,
        max_launches=1,
    )

    assert c1_plan["planned"][0]["seed"] == 9900
    assert w4b_plan["planned"][0]["seed"] == 9907


def test_plan_remote_train_advances_explicit_seed_if_already_consumed() -> None:
    poll_payload = {
        "workers": [
            {
                "worker": "catan-zero-w1b",
                "zone": "us-west1-b",
                "ok": True,
                "running_train_processes": 0,
                "candidate_checkpoints": [{"name": "s9811_existing_run.pt"}],
            },
        ],
    }

    plan = plan_remote_train(
        poll_payload=poll_payload,
        summary_payload={"active": [], "decisions": []},
        project="proj",
        remote_repo="",
        champion="champion.pt",
        recipe="warmup_baseline",
        seed=9811,
        min_seed=9800,
        iterations=10,
        episodes_per_iteration=8,
        checkpoint_every=2,
        max_launches=1,
    )

    assert plan["planned_count"] == 1
    assert plan["planned"][0]["seed"] == 9812


def test_local_controller_status_reports_claimed_remote_workers() -> None:
    ps_output = """
  46553       00:29 /venv/bin/python tools/gcp_fleet_controller.py --worker catan-zero-w1a:us-west1-b remote-grade-from-worker --source-worker catan-zero-w4b:us-west4-a --checkpoint runs/self_play/s9835_s9829_antireg_repair_w4b.iter0002.pt --profile strict --games 4 --force
  46699       00:28 /venv/bin/python tools/gcp_fleet_controller.py --project proj --worker catan-zero-c3:us-central1-c remote-train --label s9836_pfsp_klent_control_c3 -- --seed 9836 --checkpoint runs/self_play/s9836_pfsp_klent_control_c3.pt
  46701       00:11 /venv/bin/python tools/gcp_fleet_controller.py --project proj --worker catan-zero-c4:us-central1-c remote-reanalysis-train --label s10100_dags_midgame_reanalysis_c4 --seed 10100
 46702       00:07 /venv/bin/python tools/gcp_fleet_controller.py --project proj --worker catan-zero-c2:us-central1-c remote-opening-eval --checkpoint runs/self_play/s10101_candidate_c2.iter0002.pt
  50000       00:01 /bin/zsh -lc ps -axo pid,etime,command | rg gcp_fleet_controller.py
"""

    status = local_controller_status(ps_output, current_pid=99999)

    assert status["active_count"] == 4
    assert status["claimed_workers"] == [
        "catan-zero-c2",
        "catan-zero-c3",
        "catan-zero-c4",
        "catan-zero-w1a",
    ]
    assert status["active_grades"][0] == {
        "pid": 46553,
        "elapsed": "00:29",
        "command": "remote-grade-from-worker",
        "kind": "remote_grade",
        "worker": "catan-zero-w1a",
        "checkpoint": "runs/self_play/s9835_s9829_antireg_repair_w4b.iter0002.pt",
        "force": True,
        "source_worker": "catan-zero-w4b",
        "profile": "strict",
    }
    assert status["active_trains"][0]["label"] == "s9836_pfsp_klent_control_c3"
    assert status["active_trains"][0]["checkpoint"] == "runs/self_play/s9836_pfsp_klent_control_c3.pt"
    assert status["active_trains"][1]["label"] == "s10100_dags_midgame_reanalysis_c4"


def test_local_controller_status_ignores_shell_wrappers_with_worker_variables() -> None:
    ps_output = """
 50000       00:02 /bin/zsh -lc for spec in catan-zero-c1:us-central1-c; do .venv/bin/python tools/gcp_fleet_controller.py --worker "$spec" remote-sync-code --file tools/gpu_fleet.py; done
 50001       00:02 /venv/bin/python tools/gcp_fleet_controller.py --worker catan-zero-c1:us-central1-c remote-sync-code --file tools/gpu_fleet.py
"""

    status = local_controller_status(ps_output, current_pid=99999)

    assert status["active_count"] == 1
    assert status["claimed_workers"] == ["catan-zero-c1"]
    assert "$spec" not in status["claimed_workers"]


def test_plan_remote_train_treats_local_controller_claims_as_busy() -> None:
    poll_payload = {
        "workers": [
            {
                "worker": "catan-zero-c2",
                "zone": "us-central1-c",
                "ok": True,
                "running_train_processes": 0,
                "candidate_checkpoints": [],
                "trainer_features": {
                    "pfsp_mixed": True,
                    "ema_policy_kl": True,
                    "old_policy_kl": True,
                },
            },
        ],
    }
    local_status_payload = {"claimed_workers": ["catan-zero-c2"]}

    plan = plan_remote_train(
        poll_payload=poll_payload,
        summary_payload={"active": [], "decisions": []},
        local_status_payload=local_status_payload,
        project="proj",
        remote_repo="",
        champion="champion.pt",
        recipe="pfsp_klent_control",
        seed=9838,
        min_seed=9800,
        iterations=10,
        episodes_per_iteration=8,
        checkpoint_every=2,
        max_launches=1,
    )

    assert plan["planned_count"] == 0
    assert plan["skipped"]["busy_worker"] == [
        {
            "worker": "catan-zero-c2",
            "zone": "us-central1-c",
            "reason": "local_controller",
        }
    ]


def test_remote_grade_from_worker_supports_mixed_repo_roots() -> None:
    args = argparse.Namespace(
        checkpoint="runs/self_play/candidate.iter0003.pt",
        champion="runs/self_play/champions/current.pt",
        eval_dir="runs/self_play/remote_grades_reanalysis",
        log_dir="runs/self_play/logs",
        profile="strict",
        games=4,
        repeats=1,
        workers=1,
        vps_to_win=4,
        max_decisions=300,
        leg_timeout_seconds=1200,
        project="proj",
        remote_repo="",
        source_remote_repo="/home/worker/catan-zero",
        target_remote_repo="/home/worker/catan-zero/catan-zero-gcp-bundle",
        force=True,
    )

    payload = build_remote_grade_from_worker_command(
        args,
        source=Worker("catan-zero-w4b", "us-west4-a"),
        target=Worker("catan-zero-w4d", "us-west4-b"),
    )

    assert payload["copy_from_source"][3] == (
        "catan-zero-w4b:/home/worker/catan-zero/"
        "runs/self_play/candidate.iter0003.pt"
    )
    assert payload["copy_to_target"][4] == (
        "catan-zero-w4d:/home/worker/catan-zero/catan-zero-gcp-bundle/"
        "runs/self_play/candidate.iter0003.pt"
    )
    grade = payload["grade"]
    assert grade[grade.index("--remote-repo") + 1] == (
        "/home/worker/catan-zero/catan-zero-gcp-bundle"
    )


def test_required_remote_features_for_q_calibration_includes_q_target() -> None:
    assert required_remote_features_for_recipe("warmup_baseline") == ()
    assert required_remote_features_for_recipe("warmup_jsettlers") == ()
    assert required_remote_features_for_recipe("warmup_rollout") == ("value_rollout_teacher",)
    assert required_remote_features_for_recipe("pfsp_q_calibration") == (
        "pfsp_mixed",
        "ema_policy_kl",
        "old_policy_kl",
        "q_expected_sarsa",
    )
    assert required_remote_features_for_recipe("pfsp_klent_control") == (
        "pfsp_mixed",
        "ema_policy_kl",
        "old_policy_kl",
    )
    assert required_remote_features_for_recipe("strict_repair_kl") == (
        "anti_regression_mixed",
        "ema_policy_kl",
        "old_policy_kl",
    )
    assert required_remote_features_for_recipe("resource_plan_score_repair") == (
        "anti_regression_mixed",
        "ema_policy_kl",
        "old_policy_kl",
        "baseline_score_targets",
    )
    assert required_remote_features_for_recipe("rollout_guard_score_repair") == (
        "anti_regression_mixed",
        "ema_policy_kl",
        "old_policy_kl",
        "baseline_score_targets",
        "baseline_rollout_mixed",
        "value_rollout_teacher",
    )
    assert required_remote_features_for_recipe("tactical_rollout_guard_repair") == (
        "anti_regression_mixed",
        "ema_policy_kl",
        "old_policy_kl",
        "baseline_score_targets",
        "tactical_rollout_mixed",
        "value_rollout_teacher",
    )
    assert required_remote_features_for_recipe("weighted_dagger_antireg") == (
        "anti_regression_mixed",
        "ema_policy_kl",
        "old_policy_kl",
        "baseline_rollout_mixed",
        "q_advantage_gate",
        "q_expected_sarsa",
        "return_weighted_dagger",
        "sample_weighted_imitation",
        "top_advantage_filter",
        "value_rollout_teacher",
    )
    assert required_remote_features_for_recipe("ema_jsettlers_antireg") == (
        "anti_regression_mixed",
        "ema_policy_kl",
        "old_policy_kl",
        "baseline_rollout_mixed",
        "q_advantage_gate",
        "q_expected_sarsa",
        "return_weighted_dagger",
        "sample_weighted_imitation",
        "top_advantage_filter",
        "value_rollout_teacher",
    )
    assert required_remote_features_for_recipe("ema_mixed_antireg") == (
        "anti_regression_mixed",
        "ema_policy_kl",
        "old_policy_kl",
        "baseline_rollout_mixed",
        "q_advantage_gate",
        "q_expected_sarsa",
        "return_weighted_dagger",
        "sample_weighted_imitation",
        "top_advantage_filter",
        "value_rollout_teacher",
    )
    assert required_remote_features_for_recipe("vrpo_esarsa_antireg") == (
        "anti_regression_mixed",
        "ema_policy_kl",
        "old_policy_kl",
        "baseline_rollout_mixed",
        "q_advantage_gate",
        "q_expected_sarsa",
        "return_weighted_dagger",
        "sample_weighted_imitation",
        "top_advantage_filter",
        "value_rollout_teacher",
    )
    assert required_remote_features_for_recipe("vrpo_jsettlers_value_repair") == (
        "anti_regression_mixed",
        "ema_policy_kl",
        "old_policy_kl",
        "baseline_rollout_mixed",
        "q_advantage_gate",
        "q_expected_sarsa",
        "return_weighted_dagger",
        "sample_weighted_imitation",
        "top_advantage_filter",
        "value_rollout_teacher",
        "jsettlers_value_repair_mixed",
    )
    assert required_remote_features_for_recipe("dags_midgame_reanalysis") == (
        "reanalysis_training",
        "reanalysis_decision_windows",
    )
    assert required_remote_features_for_recipe("opening_eval") == (
        "opening_evaluator",
    )
    assert missing_remote_features(
        {"trainer_features": {"pfsp_mixed": True, "ema_policy_kl": True}},
        required_features=("pfsp_mixed", "ema_policy_kl", "old_policy_kl"),
        allow_unknown=False,
    ) == ["old_policy_kl"]


def test_plan_remote_code_sync_targets_idle_missing_recipe_features() -> None:
    base_features = {
        "anti_regression_mixed": True,
        "baseline_score_targets": True,
        "ema_policy_kl": True,
        "old_policy_kl": True,
        "value_rollout_teacher": True,
    }
    poll_payload = {
        "workers": [
            {
                "worker": "catan-zero-c1",
                "zone": "us-central1-c",
                "ok": True,
                "running_train_processes": 1,
                "trainer_features": {**base_features, "tactical_rollout_mixed": False},
                "repo": "/home/worker/catan-zero",
            },
            {
                "worker": "catan-zero-w4d",
                "zone": "us-west4-b",
                "ok": True,
                "running_train_processes": 0,
                "trainer_features": {**base_features, "tactical_rollout_mixed": False},
                "repo": "/home/worker/catan-zero/catan-zero-gcp-bundle",
                "trainer_sha1": "oldsha",
            },
            {
                "worker": "catan-zero-w1a",
                "zone": "us-west1-b",
                "ok": True,
                "running_train_processes": 0,
                "trainer_features": {**base_features, "tactical_rollout_mixed": True},
                "repo": "/home/worker/catan-zero",
            },
        ]
    }

    plan = plan_remote_code_sync(
        poll_payload=poll_payload,
        summary_payload={"active": []},
        local_status_payload=None,
        project="proj",
        remote_repo="",
        recipe="tactical_rollout_guard_repair",
        files=["tools/train_ppo.py"],
        backup_dir="runs/self_play/code_backups",
        max_syncs=2,
    )

    assert plan["planned_count"] == 1
    assert plan["planned"][0]["worker"] == "catan-zero-w4d"
    assert plan["planned"][0]["repo"] == "/home/worker/catan-zero/catan-zero-gcp-bundle"
    assert plan["planned"][0]["missing"] == ["tactical_rollout_mixed"]
    command = plan["planned"][0]["command"]
    assert "remote-sync-code" in command
    assert command[command.index("--remote-repo") + 1] == (
        "/home/worker/catan-zero/catan-zero-gcp-bundle"
    )
    assert plan["skipped"]["busy_worker"][0]["reason"] == "training"
    assert plan["skipped"]["already_satisfied"][0]["worker"] == "catan-zero-w1a"


def test_plan_remote_code_sync_treats_active_grade_as_busy() -> None:
    poll_payload = {
        "workers": [
            {
                "worker": "catan-zero-w4d",
                "zone": "us-west4-b",
                "ok": True,
                "running_train_processes": 0,
                "trainer_features": {},
                "repo": "/home/worker/catan-zero",
            }
        ]
    }

    plan = plan_remote_code_sync(
        poll_payload=poll_payload,
        summary_payload={"active": [{"worker": "catan-zero-w4d"}]},
        local_status_payload=None,
        project="proj",
        remote_repo="",
        recipe="tactical_rollout_guard_repair",
        files=["tools/train_ppo.py"],
        backup_dir="runs/self_play/code_backups",
        max_syncs=1,
    )

    assert plan["planned_count"] == 0
    assert plan["skipped"]["busy_worker"][0] == {
        "worker": "catan-zero-w4d",
        "zone": "us-west4-b",
        "reason": "remote_grade",
    }


def test_remote_sync_code_command_backs_up_and_checks_busy_workers() -> None:
    command = remote_sync_preflight_command(
        remote_repo="/home/worker/catan-zero/catan-zero-gcp-bundle",
        files=["tools/train_ppo.py"],
        backup_dir="runs/self_play/code_backups",
        allow_busy=False,
    )

    assert "tools/train_ppo.py" in command
    assert "tools/grade_agent.py" in command
    assert "worker_busy" in command
    assert "shutil.copy2" in command
    assert "catan-zero-gcp-bundle" in command


def test_remote_sync_code_launch_command_includes_files_and_backup_dir() -> None:
    command = build_remote_sync_code_launch_command(
        worker=Worker("catan-zero-w4d", "us-west4-b"),
        project="proj",
        remote_repo="/home/worker/catan-zero/catan-zero-gcp-bundle",
        files=["tools/train_ppo.py"],
        backup_dir="runs/self_play/code_backups",
    )

    assert command[command.index("--worker") + 1] == "catan-zero-w4d:us-west4-b"
    assert command[command.index("--remote-repo") + 1] == (
        "/home/worker/catan-zero/catan-zero-gcp-bundle"
    )
    assert "remote-sync-code" in command
    assert command[command.index("--file") + 1] == "tools/train_ppo.py"


def test_warmup_only_training_args_disable_ppo_and_policy_mixing() -> None:
    args = build_warmup_only_training_args(
        label="s9808_warmup_baseline_c1",
        seed=9808,
        champion="champion.pt",
        recipe="warmup_baseline",
    )

    assert args[args.index("--teacher") + 1] == "baseline_mixed"
    assert args[args.index("--iterations") + 1] == "0"
    assert args[args.index("--checkpoint-every") + 1] == "0"
    assert "--select-best-warmup-checkpoint" in args
    assert "--opponents" not in args


def test_warmup_rollout_training_args_use_search_teacher() -> None:
    args = build_warmup_only_training_args(
        label="s9809_warmup_rollout_c3",
        seed=9809,
        champion="champion.pt",
        recipe="warmup_rollout",
    )

    assert args[args.index("--teacher") + 1] == "value_rollout"
    assert args[args.index("--teacher-rollout-decisions") + 1] == "3"
    assert args[args.index("--teacher-root-value-weight") + 1] == "0.35"


def test_plan_remote_train_avoids_active_grade_workers() -> None:
    poll_payload = {
        "workers": [
            {
                "worker": "catan-zero-w4d",
                "zone": "us-west4-b",
                "ok": True,
                "running_train_processes": 0,
                "candidate_checkpoints": [],
            },
        ],
    }
    summary_payload = {
        "active": [
            {
                "worker": "catan-zero-w4d",
                "checkpoint": "runs/self_play/s9791.iter0006.pt",
            }
        ],
        "decisions": [],
    }

    plan = plan_remote_train(
        poll_payload=poll_payload,
        summary_payload=summary_payload,
        project="proj",
        remote_repo="",
        champion="champion.pt",
        recipe="pfsp_value_jsettlers",
        seed=9800,
        min_seed=9800,
        iterations=10,
        episodes_per_iteration=8,
        checkpoint_every=2,
        max_launches=1,
    )

    assert plan["planned_count"] == 0
    assert plan["skipped"]["busy_worker"][0]["reason"] == "remote_grade"


def test_next_training_seed_reads_poll_and_summary() -> None:
    poll_payload = {
        "workers": [
            {
                "processes": [{"checkpoint": "runs/self_play/s9798_live.pt"}],
                "candidate_checkpoints": [{"name": "s9799_done.iter0002.pt"}],
                "files": [],
            }
        ]
    }
    summary_payload = {
        "active": [{"checkpoint": "runs/self_play/s9797_grade.pt"}],
        "decisions": [{"checkpoint": "runs/self_play/s9796_reject.pt"}],
    }

    assert (
        next_training_seed(
            poll_payload=poll_payload,
            summary_payload=summary_payload,
            min_seed=9800,
        )
        == 9800
    )


def test_next_training_seed_treats_log_only_failed_runs_as_consumed() -> None:
    poll_payload = {
        "workers": [
            {
                "logs": [
                    {"name": "s9818_reanalysis_value_graphdata_w4c.log"},
                ],
                "files": [],
                "candidate_checkpoints": [],
                "processes": [],
            }
        ]
    }

    assert (
        next_training_seed(
            poll_payload=poll_payload,
            summary_payload={"active": [], "decisions": []},
            min_seed=9800,
        )
        == 9819
    )


def test_planned_training_args_supports_q_calibration_without_policy_mixing() -> None:
    args = build_planned_training_args(
        label="s9801_pfsp_q_calibration_c2",
        seed=9801,
        champion="champion.pt",
        recipe="pfsp_q_calibration",
        iterations=10,
        episodes_per_iteration=8,
        checkpoint_every=2,
    )

    assert args[args.index("--q-value-coef") + 1] == "0.15"
    assert args[args.index("--q-expected-sarsa-mix") + 1] == "0.25"
    assert args[args.index("--q-advantage-mix") + 1] == "0.0"
    assert args[args.index("--q-advantage-warmup-iterations") + 1] == "11"


def test_planned_training_args_supports_klent_control() -> None:
    args = build_planned_training_args(
        label="s9833_pfsp_klent_control_c2",
        seed=9833,
        champion="champion.pt",
        recipe="pfsp_klent_control",
        iterations=10,
        episodes_per_iteration=8,
        checkpoint_every=2,
    )

    assert args[args.index("--opponents") + 1] == "pfsp_mixed"
    assert args[args.index("--q-advantage-mix") + 1] == "0.0"
    assert args[args.index("--q-expected-sarsa-mix") + 1] == "0.0"
    assert args[args.index("--entropy-coef") + 1] == "0.02"
    assert args[args.index("--old-policy-kl-coef") + 1] == "0.03"
    assert args[args.index("--ema-policy-kl-coef") + 1] == "0.015"
    assert args[args.index("--ema-policy-decay") + 1] == "0.985"
    assert args[args.index("--target-kl") + 1] == "0.02"
    assert args[args.index("--gae-lambda") + 1] == "0.90"


def test_planned_training_args_supports_strict_repair_kl() -> None:
    args = build_planned_training_args(
        label="s9837_strict_repair_kl_w4b",
        seed=9837,
        champion="champion.pt",
        recipe="strict_repair_kl",
        iterations=10,
        episodes_per_iteration=8,
        checkpoint_every=2,
    )

    assert args[args.index("--teacher") + 1] == "baseline_mixed"
    assert args[args.index("--opponents") + 1] == "anti_regression_mixed"
    assert args[args.index("--q-advantage-mix") + 1] == "0.0"
    assert args[args.index("--q-expected-sarsa-mix") + 1] == "0.0"
    assert args[args.index("--training-value-opponent-penalty") + 1] == "0.08"
    assert args[args.index("--old-policy-kl-coef") + 1] == "0.025"
    assert args[args.index("--ema-policy-kl-coef") + 1] == "0.0125"
    assert args[args.index("--anchor-games-per-iteration") + 1] == "3"
    assert args[args.index("--dagger-games-per-iteration") + 1] == "2"
    assert args[args.index("--imitation-hard-target-weight") + 1] == "0.20"
    assert args[args.index("--gae-lambda") + 1] == "0.90"


def test_planned_training_args_supports_resource_plan_score_repair() -> None:
    args = build_planned_training_args(
        label="s9848_resource_plan_score_repair_w4d",
        seed=9848,
        champion="champion.pt",
        recipe="resource_plan_score_repair",
        iterations=10,
        episodes_per_iteration=8,
        checkpoint_every=2,
    )

    assert args[args.index("--teacher") + 1] == "baseline_mixed"
    assert args[args.index("--opponents") + 1] == "anti_regression_mixed"
    assert args[args.index("--q-advantage-mix") + 1] == "0.0"
    assert args[args.index("--q-expected-sarsa-mix") + 1] == "0.0"
    assert args[args.index("--training-value-opponent-penalty") + 1] == "0.10"
    assert args[args.index("--old-policy-kl-coef") + 1] == "0.03"
    assert args[args.index("--ema-policy-kl-coef") + 1] == "0.015"
    assert args[args.index("--target-kl") + 1] == "0.018"
    assert args[args.index("--anchor-games-per-iteration") + 1] == "4"
    assert args[args.index("--imitation-score-coef") + 1] == "0.08"
    assert args[args.index("--imitation-hard-target-weight") + 1] == "0.12"
    assert args[args.index("--gae-lambda") + 1] == "0.90"


def test_planned_training_args_supports_rollout_guard_score_repair() -> None:
    args = build_planned_training_args(
        label="s9865_rollout_guard_score_repair_w4d",
        seed=9865,
        champion="champion.pt",
        recipe="rollout_guard_score_repair",
        iterations=10,
        episodes_per_iteration=8,
        checkpoint_every=2,
    )

    assert args[args.index("--teacher") + 1] == "baseline_rollout_mixed"
    assert args[args.index("--opponents") + 1] == "anti_regression_mixed"
    assert args[args.index("--q-advantage-mix") + 1] == "0.0"
    assert args[args.index("--q-expected-sarsa-mix") + 1] == "0.0"
    assert args[args.index("--teacher-candidate-limit") + 1] == "24"
    assert args[args.index("--teacher-presearch-candidate-limit") + 1] == "48"
    assert args[args.index("--teacher-rollout-decisions") + 1] == "2"
    assert args[args.index("--teacher-root-value-weight") + 1] == "0.25"
    assert args[args.index("--imitation-score-coef") + 1] == "0.07"
    assert args[args.index("--old-policy-kl-coef") + 1] == "0.032"
    assert args[args.index("--ema-policy-kl-coef") + 1] == "0.016"
    assert args[args.index("--target-kl") + 1] == "0.016"
    assert args[args.index("--dagger-games-per-iteration") + 1] == "3"


def test_planned_training_args_supports_tactical_rollout_guard_repair() -> None:
    args = build_planned_training_args(
        label="s9874_tactical_rollout_guard_repair_w4d",
        seed=9874,
        champion="champion.pt",
        recipe="tactical_rollout_guard_repair",
        iterations=10,
        episodes_per_iteration=8,
        checkpoint_every=2,
    )

    assert args[args.index("--teacher") + 1] == "tactical_rollout_mixed"
    assert args[args.index("--opponents") + 1] == "anti_regression_mixed"
    assert args[args.index("--q-advantage-mix") + 1] == "0.0"
    assert args[args.index("--q-expected-sarsa-mix") + 1] == "0.0"
    assert args[args.index("--teacher-candidate-limit") + 1] == "24"
    assert args[args.index("--teacher-presearch-candidate-limit") + 1] == "48"
    assert args[args.index("--teacher-rollout-decisions") + 1] == "2"
    assert args[args.index("--teacher-root-value-weight") + 1] == "0.25"
    assert args[args.index("--imitation-score-coef") + 1] == "0.06"
    assert args[args.index("--imitation-hard-target-weight") + 1] == "0.16"
    assert args[args.index("--old-policy-kl-coef") + 1] == "0.034"
    assert args[args.index("--ema-policy-kl-coef") + 1] == "0.017"
    assert args[args.index("--target-kl") + 1] == "0.014"
    assert args[args.index("--dagger-games-per-iteration") + 1] == "3"


def test_planned_training_args_supports_weighted_dagger_antireg() -> None:
    args = build_planned_training_args(
        label="s9874_weighted_dagger_antireg_w4d",
        seed=9874,
        champion="champion.pt",
        recipe="weighted_dagger_antireg",
        iterations=8,
        episodes_per_iteration=10,
        checkpoint_every=2,
    )

    assert args[args.index("--teacher") + 1] == "baseline_rollout_mixed"
    assert args[args.index("--warmup-games") + 1] == "0"
    assert args[args.index("--opponents") + 1] == "anti_regression_mixed"
    assert args[args.index("--learner-seats") + 1] == "one"
    assert args[args.index("--ppo-epochs") + 1] == "2"
    assert args[args.index("--learning-rate") + 1] == "0.0001"
    assert args[args.index("--clip-ratio") + 1] == "0.12"
    assert args[args.index("--ppo-top-advantage-fraction") + 1] == "0.4"
    assert args[args.index("--q-value-coef") + 1] == "0.25"
    assert args[args.index("--q-expected-sarsa-mix") + 1] == "0.25"
    assert args[args.index("--q-advantage-mix") + 1] == "0.05"
    assert args[args.index("--old-policy-kl-coef") + 1] == "0.055"
    assert args[args.index("--ema-policy-kl-coef") + 1] == "0.03"
    assert args[args.index("--target-kl") + 1] == "0.012"
    assert args[args.index("--anchor-sample-weight") + 1] == "1.0"
    assert args[args.index("--dagger-sample-weight") + 1] == "3.0"
    assert args[args.index("--dagger-games-per-iteration") + 1] == "2"


def test_planned_training_args_supports_ema_jsettlers_antireg() -> None:
    args = build_planned_training_args(
        label="s9897_ema_jsettlers_antireg_w4d",
        seed=9897,
        champion="champion.pt",
        recipe="ema_jsettlers_antireg",
        iterations=8,
        episodes_per_iteration=10,
        checkpoint_every=2,
    )

    assert args[args.index("--teacher") + 1] == "baseline_rollout_mixed"
    assert args[args.index("--warmup-games") + 1] == "0"
    assert args[args.index("--opponents") + 1] == "jsettlers_lite"
    assert args[args.index("--learning-rate") + 1] == "0.00007"
    assert args[args.index("--clip-ratio") + 1] == "0.08"
    assert args[args.index("--entropy-coef") + 1] == "0.009"
    assert args[args.index("--old-policy-kl-coef") + 1] == "0.035"
    assert args[args.index("--ema-policy-kl-coef") + 1] == "0.075"
    assert args[args.index("--ema-policy-decay") + 1] == "0.997"
    assert args[args.index("--target-kl") + 1] == "0.008"
    assert args[args.index("--anchor-games-per-iteration") + 1] == "4"
    assert args[args.index("--dagger-games-per-iteration") + 1] == "4"
    assert args[args.index("--ppo-top-advantage-fraction") + 1] == "0.30"
    assert args[args.index("--dagger-sample-weight") + 1] == "4.0"


def test_planned_training_args_supports_ema_mixed_antireg() -> None:
    args = build_planned_training_args(
        label="s9910_ema_mixed_antireg_w4d",
        seed=9910,
        champion="champion.pt",
        recipe="ema_mixed_antireg",
        iterations=8,
        episodes_per_iteration=10,
        checkpoint_every=2,
    )

    assert args[args.index("--teacher") + 1] == "baseline_rollout_mixed"
    assert args[args.index("--warmup-games") + 1] == "0"
    assert args[args.index("--opponents") + 1] == "anti_regression_mixed"
    assert args[args.index("--learning-rate") + 1] == "0.000075"
    assert args[args.index("--clip-ratio") + 1] == "0.09"
    assert args[args.index("--entropy-coef") + 1] == "0.011"
    assert args[args.index("--old-policy-kl-coef") + 1] == "0.045"
    assert args[args.index("--ema-policy-kl-coef") + 1] == "0.065"
    assert args[args.index("--ema-policy-decay") + 1] == "0.997"
    assert args[args.index("--target-kl") + 1] == "0.009"
    assert args[args.index("--anchor-games-per-iteration") + 1] == "4"
    assert args[args.index("--dagger-games-per-iteration") + 1] == "4"
    assert args[args.index("--anchor-replay-size") + 1] == "3072"
    assert args[args.index("--ppo-top-advantage-fraction") + 1] == "0.32"
    assert args[args.index("--dagger-sample-weight") + 1] == "3.5"


def test_planned_training_args_supports_vrpo_esarsa_antireg() -> None:
    args = build_planned_training_args(
        label="s10020_vrpo_esarsa_antireg_w4d",
        seed=10020,
        champion="champion.pt",
        recipe="vrpo_esarsa_antireg",
        iterations=8,
        episodes_per_iteration=10,
        checkpoint_every=2,
    )

    assert args[args.index("--teacher") + 1] == "baseline_rollout_mixed"
    assert args[args.index("--warmup-games") + 1] == "0"
    assert args[args.index("--opponents") + 1] == "anti_regression_mixed"
    assert args[args.index("--learning-rate") + 1] == "0.000065"
    assert args[args.index("--clip-ratio") + 1] == "0.08"
    assert args[args.index("--value-coef") + 1] == "0.65"
    assert args[args.index("--q-value-coef") + 1] == "0.45"
    assert args[args.index("--q-advantage-mix") + 1] == "0.10"
    assert args[args.index("--q-expected-sarsa-mix") + 1] == "0.55"
    assert args[args.index("--entropy-coef") + 1] == "0.010"
    assert args[args.index("--old-policy-kl-coef") + 1] == "0.040"
    assert args[args.index("--ema-policy-kl-coef") + 1] == "0.080"
    assert args[args.index("--ema-policy-decay") + 1] == "0.998"
    assert args[args.index("--target-kl") + 1] == "0.008"
    assert args[args.index("--anchor-games-per-iteration") + 1] == "3"
    assert args[args.index("--dagger-games-per-iteration") + 1] == "3"
    assert args[args.index("--anchor-replay-size") + 1] == "4096"
    assert args[args.index("--ppo-top-advantage-fraction") + 1] == "0.45"
    assert args[args.index("--q-advantage-warmup-iterations") + 1] == "1"
    assert args[args.index("--q-advantage-ramp-iterations") + 1] == "3"
    assert args[args.index("--q-advantage-min-sign-agreement") + 1] == "0.58"
    assert args[args.index("--q-advantage-min-return-corr") + 1] == "0.08"
    assert args[args.index("--dagger-sample-weight") + 1] == "3.0"


def test_planned_training_args_supports_vrpo_jsettlers_value_repair() -> None:
    args = build_planned_training_args(
        label="s10130_vrpo_jsettlers_value_repair_w4d",
        seed=10130,
        champion="champion.pt",
        recipe="vrpo_jsettlers_value_repair",
        iterations=8,
        episodes_per_iteration=10,
        checkpoint_every=2,
    )

    assert args[args.index("--teacher") + 1] == "baseline_rollout_mixed"
    assert args[args.index("--warmup-games") + 1] == "0"
    assert args[args.index("--opponents") + 1] == "jsettlers_value_repair_mixed"
    assert args[args.index("--learning-rate") + 1] == "0.000055"
    assert args[args.index("--clip-ratio") + 1] == "0.07"
    assert args[args.index("--value-coef") + 1] == "0.70"
    assert args[args.index("--q-value-coef") + 1] == "0.50"
    assert args[args.index("--q-advantage-mix") + 1] == "0.08"
    assert args[args.index("--q-expected-sarsa-mix") + 1] == "0.60"
    assert args[args.index("--entropy-coef") + 1] == "0.009"
    assert args[args.index("--old-policy-kl-coef") + 1] == "0.045"
    assert args[args.index("--ema-policy-kl-coef") + 1] == "0.090"
    assert args[args.index("--ema-policy-decay") + 1] == "0.9985"
    assert args[args.index("--target-kl") + 1] == "0.007"
    assert args[args.index("--anchor-games-per-iteration") + 1] == "4"
    assert args[args.index("--dagger-games-per-iteration") + 1] == "4"
    assert args[args.index("--anchor-replay-size") + 1] == "6144"
    assert args[args.index("--ppo-top-advantage-fraction") + 1] == "0.40"
    assert args[args.index("--q-advantage-warmup-iterations") + 1] == "1"
    assert args[args.index("--q-advantage-ramp-iterations") + 1] == "4"
    assert args[args.index("--q-advantage-min-sign-agreement") + 1] == "0.60"
    assert args[args.index("--q-advantage-min-return-corr") + 1] == "0.10"
    assert args[args.index("--dagger-sample-weight") + 1] == "4.0"


def test_gcp_remote_stop_train_command_matches_exact_substring() -> None:
    args = argparse.Namespace(
        remote_repo="~/catan-zero",
        match="s9764_adaptive_ema_lowkl_c4",
        dry_run=False,
    )

    command = build_remote_stop_train_command(args)

    assert "tools/train_ppo.py" in command
    assert "match='s9764_adaptive_ema_lowkl_c4'" in command
    assert "os.kill(pid, signal.SIGTERM)" in command
    assert "configured='~/catan-zero'" in command
    assert "'killed': killed" in command


def test_gcp_remote_stop_train_command_supports_dry_run() -> None:
    args = argparse.Namespace(
        remote_repo="",
        match="s9765_search_ema_dagger_w4a",
        dry_run=True,
    )

    command = build_remote_stop_train_command(args)

    assert "dry_run=True" in command
    assert "if not dry_run:" in command


def test_gcp_remote_stop_grade_command_matches_exact_substring() -> None:
    args = argparse.Namespace(
        remote_repo="~/catan-zero",
        match="s9796_jsettlers_repair_c2.iter0002",
        dry_run=False,
    )

    command = build_remote_stop_grade_command(args)

    assert "tools/grade_agent.py" in command
    assert "match='s9796_jsettlers_repair_c2.iter0002'" in command
    assert "os.kill(pid, signal.SIGTERM)" in command
    assert "configured='~/catan-zero'" in command
    assert "'script': script" in command


def test_gcp_remote_stop_grade_command_supports_dry_run() -> None:
    args = argparse.Namespace(
        remote_repo="",
        match="summary_s9796",
        dry_run=True,
    )

    command = build_remote_stop_grade_command(args)

    assert "tools/grade_agent.py" in command
    assert "dry_run=True" in command
    assert "if not dry_run:" in command


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


def test_gcp_remote_grade_status_command_reads_summaries_and_legs() -> None:
    command = remote_grade_status_command(
        remote_repo="~/catan-zero",
        eval_dir="runs/self_play/remote_grades_reanalysis",
        log_dir="runs/self_play/logs",
    )

    assert "tools/grade_agent.py" in command
    assert "summary_*.json" in command
    assert "grade_*.json" in command
    assert "remote_grade*.log" in command
    assert "configured='~/catan-zero'" in command


def test_remote_grade_from_worker_copies_then_grades_target() -> None:
    args = argparse.Namespace(
        project="proj",
        remote_repo="",
        checkpoint="runs/self_play/s9828.iter0004.pt",
        champion="runs/self_play/champions/current_best_s9752_iter0002.pt",
        eval_dir="grades",
        log_dir="logs",
        profile="strict",
        games=4,
        repeats=1,
        workers=1,
        vps_to_win=4,
        max_decisions=300,
        leg_timeout_seconds=1200,
        force=True,
    )

    command = build_remote_grade_from_worker_command(
        args,
        source=Worker("source-vm", "us-central1-c"),
        target=Worker("target-vm", "us-west1-b"),
    )

    assert command["copy_from_source"][:4] == [
        "gcloud",
        "compute",
        "scp",
        "source-vm:~/catan-zero/runs/self_play/s9828.iter0004.pt",
    ]
    assert command["copy_to_target"][3] == "$TMPDIR/s9828.iter0004.pt"
    assert command["copy_to_target"][4] == (
        "target-vm:~/catan-zero/runs/self_play/s9828.iter0004.pt"
    )
    assert command["grade"][command["grade"].index("--worker") + 1] == (
        "target-vm:us-west1-b"
    )
    assert "--force" in command["grade"]


def test_run_command_with_retries_retries_transient_failure(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command, *, check):
        calls.append(command)
        if len(calls) == 1:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr("tools.gcp_fleet_controller.subprocess.run", fake_run)
    monkeypatch.setattr("tools.gcp_fleet_controller.time.sleep", lambda _: None)

    run_command_with_retries(["gcloud", "compute", "scp"], attempts=2)

    assert calls == [["gcloud", "compute", "scp"], ["gcloud", "compute", "scp"]]


def test_gcp_remote_grade_status_keeps_long_active_lines() -> None:
    command = remote_grade_status_command(
        remote_repo="~/catan-zero",
        eval_dir="runs/self_play/remote_grades_reanalysis",
        log_dir="runs/self_play/logs",
    )

    assert "line[:4000]" in command
    assert "line[:500]" not in command


def test_gcp_poll_process_classifier_ignores_shell_wrappers() -> None:
    wrapper = (
        "52168 bash -c cd ~/catan-zero && setsid sh -c "
        "'.venv/bin/python -u tools/train_ppo.py --seed 9750 "
        "--checkpoint runs/self_play/s9750_vrpo_sarsa_candidate.pt'"
    )
    child_shell = (
        "52170 sh -c .venv/bin/python -u tools/train_ppo.py --seed 9750 "
        "--checkpoint runs/self_play/s9750_vrpo_sarsa_candidate.pt"
    )
    trainer = (
        "52171 .venv/bin/python -u tools/train_ppo.py --seed 9750 "
        "--checkpoint runs/self_play/s9750_vrpo_sarsa_candidate.pt"
    )

    assert not is_train_process_line(wrapper)
    assert not is_train_process_line(child_shell)
    assert is_train_process_line(trainer)


def test_gcp_remote_grade_summary_compacts_status_payload() -> None:
    payload = {
        "workers": [
            {
                "worker": "catan-zero-w4d",
                "zone": "us-west4-b",
                "active_grades": [
                    ".venv/bin/python -u tools/grade_agent.py --checkpoint runs/self_play/s9740.pt",
                ],
                "summaries": [
                    {
                        "name": "summary_s9720_strict_g4_r1_w1_vp4_d300_to1200_current_best_s9752_iter0002_abcd.json",
                        "data": [
                            {
                                "checkpoint": "runs/self_play/s9720.pt",
                                "champion": "runs/self_play/champions/current_best_s9752_iter0002.pt",
                                "decision": "reject",
                                "reason": "opponent regression value:-0.25",
                                "candidate": {"weighted_win_rate": 0.1},
                                "champion_summary": {"weighted_win_rate": 0.2},
                            },
                            {
                                "checkpoint": "runs/self_play/s9721.pt",
                                "champion": "runs/self_play/champions/current_best_s9752_iter0002.pt",
                                "decision": "reject",
                                "reason": "candidate weighted win rate 0.0000 below threshold 0.0000",
                                "candidate": {"weighted_win_rate": 0.0},
                                "champion_summary": None,
                                "early_reject": True,
                            },
                        ],
                    },
                ],
                "legs": [
                    {
                        "name": "grade_s9720_vs_jsettlers_lite_g4_r0_s92000.json",
                        "opponent": "jsettlers_lite",
                        "wins": 1,
                        "games": 4,
                        "win_rate": 0.25,
                    },
                    {
                        "name": "grade_s9720_vs_value_rollout_g4_r0_s94000.json",
                        "opponent": "value_rollout_search",
                        "wins": 0,
                        "games": 4,
                        "win_rate": 0.0,
                    },
                ],
            },
        ],
    }

    summary = summarize_remote_grade_status(payload)

    assert summary["active"] == [
        {
            "worker": "catan-zero-w4d",
            "zone": "us-west4-b",
            "checkpoint": "runs/self_play/s9740.pt",
        }
    ]
    assert summary["rejections"][0]["checkpoint"] == "runs/self_play/s9720.pt"
    assert summary["rejections"][0]["zone"] == "us-west4-b"
    assert summary["rejections"][0]["summary_games"] == 4
    early = [row for row in summary["rejections"] if row["checkpoint"] == "runs/self_play/s9721.pt"][0]
    assert early["candidate_weighted_win_rate"] == 0.0
    assert early["champion_weighted_win_rate"] is None
    assert summary["legs"][0]["checkpoint"] == "s9720"
    assert summary["legs"][0]["opponents"]["jsettlers_lite"]["wins"] == 1
    assert summary["legs"][0]["opponents"]["value_rollout_search"]["win_rate"] == 0.0


def test_plan_remote_escalations_promotes_smoke_pass_to_larger_gate() -> None:
    summary_payload = {
        "active": [],
        "decisions": [
            {
                "worker": "catan-zero-c1",
                "zone": "us-central1-c",
                "summary": "summary_s9770_good.iter0002_strict_g4_r1_w1_vp4_d300_to1200_current_best_s9752_iter0002_abcd.json",
                "checkpoint": "runs/self_play/s9770_good.iter0002.pt",
                "champion": "runs/self_play/champions/current_best_s9752_iter0002.pt",
                "decision": "promote_candidate",
                "summary_games": 4,
                "candidate_weighted_win_rate": 0.25,
                "champion_weighted_win_rate": 0.15,
            },
            {
                "worker": "catan-zero-c2",
                "zone": "us-central1-c",
                "summary": "summary_s9771_reject.iter0002_strict_g4_r1_w1_vp4_d300_to1200_current_best_s9752_iter0002_abcd.json",
                "checkpoint": "runs/self_play/s9771_reject.iter0002.pt",
                "champion": "runs/self_play/champions/current_best_s9752_iter0002.pt",
                "decision": "reject",
                "summary_games": 4,
                "candidate_weighted_win_rate": 0.10,
                "champion_weighted_win_rate": 0.15,
            },
        ],
    }

    plan = plan_remote_escalations(
        summary_payload=summary_payload,
        project="proj",
        remote_repo="",
        champion="runs/self_play/champions/current_best_s9752_iter0002.pt",
        eval_dir="grades",
        log_dir="logs",
        profile="strict",
        source_games=4,
        target_games=12,
        repeats=1,
        grade_workers=1,
        vps_to_win=4,
        max_decisions=300,
        leg_timeout_seconds=1800,
        max_escalations=4,
        allow_busy_workers=False,
    )

    assert plan["planned_count"] == 1
    planned = plan["planned"][0]
    assert planned["checkpoint"] == "runs/self_play/s9770_good.iter0002.pt"
    assert planned["target_games"] == 12
    assert planned["command"][planned["command"].index("--worker") + 1] == "catan-zero-c1:us-central1-c"
    assert planned["command"][planned["command"].index("--games") + 1] == "12"
    assert "--checkpoint runs/self_play/s9770_good.iter0002.pt" in planned["shell"]
    assert plan["skipped"]["not_smoke_promote"][0]["checkpoint"] == "runs/self_play/s9771_reject.iter0002.pt"


def test_plan_remote_escalations_skips_existing_target_gate() -> None:
    summary_payload = {
        "active": [],
        "decisions": [
            {
                "worker": "catan-zero-c1",
                "zone": "us-central1-c",
                "summary": "summary_s9770_good.iter0002_strict_g4_r1_w1_vp4_d300_to1200_current_best_s9752_iter0002_abcd.json",
                "checkpoint": "runs/self_play/s9770_good.iter0002.pt",
                "champion": "runs/self_play/champions/current_best_s9752_iter0002.pt",
                "decision": "promote_candidate",
                "summary_games": 4,
                "candidate_weighted_win_rate": 0.25,
                "champion_weighted_win_rate": 0.15,
            },
            {
                "worker": "catan-zero-c1",
                "zone": "us-central1-c",
                "summary": "summary_s9770_good.iter0002_strict_g12_r1_w1_vp4_d300_to1800_current_best_s9752_iter0002_abcd.json",
                "checkpoint": "runs/self_play/s9770_good.iter0002.pt",
                "champion": "runs/self_play/champions/current_best_s9752_iter0002.pt",
                "decision": "promote_candidate",
                "summary_games": 12,
                "candidate_weighted_win_rate": 0.22,
                "champion_weighted_win_rate": 0.16,
            },
        ],
    }

    plan = plan_remote_escalations(
        summary_payload=summary_payload,
        project="proj",
        remote_repo="",
        champion="runs/self_play/champions/current_best_s9752_iter0002.pt",
        eval_dir="grades",
        log_dir="logs",
        profile="strict",
        source_games=4,
        target_games=12,
        repeats=1,
        grade_workers=1,
        vps_to_win=4,
        max_decisions=300,
        leg_timeout_seconds=1800,
        max_escalations=4,
        allow_busy_workers=False,
    )

    assert plan["planned_count"] == 0
    assert plan["skipped"]["target_exists"][0]["checkpoint"] == "runs/self_play/s9770_good.iter0002.pt"


def test_gcp_fleet_controller_ignores_init_and_warmup_checkpoints() -> None:
    assert is_candidate_checkpoint_name("s9700_value_mix_a.pt")
    assert is_candidate_checkpoint_name("s9700_value_mix_a.iter0003.pt")
    assert not is_candidate_checkpoint_name("s9700_value_mix_a.init.pt")
    assert not is_candidate_checkpoint_name("s9700_value_mix_a.warmup0004.pt")
    assert is_candidate_checkpoint_name(
        "s9700_value_mix_a.warmup0004.pt",
        include_warmup=True,
    )
    assert not is_candidate_checkpoint_name("s9700_value_mix_a.json")


def test_gcp_fleet_controller_can_exclude_interim_checkpoints() -> None:
    files = [
        {"name": "s9700_value_mix_a.init.pt"},
        {"name": "s9700_value_mix_a.iter0003.pt"},
        {"name": "s9700_value_mix_a.pt"},
        {"name": "s9700_value_mix_a.json"},
    ]

    assert candidate_stems(files, include_interim=False) == {"s9700_value_mix_a"}
    assert candidate_stems(files, include_interim=True) == {
        "s9700_value_mix_a",
        "s9700_value_mix_a.iter0003",
    }


def test_select_local_checkpoints_prefers_recent_candidates(tmp_path: Path) -> None:
    init = tmp_path / "catan-zero-c1_s9700_value_mix_a.init.pt"
    old = tmp_path / "catan-zero-c1_s9700_value_mix_a.pt"
    new = tmp_path / "catan-zero-c1_s9700_value_mix_a.iter0003.pt"
    init.write_bytes(b"init")
    old.write_bytes(b"old")
    new.write_bytes(b"new")

    selected = select_local_checkpoints(
        tmp_path,
        run_prefix="s97",
        include_interim=True,
        max_checkpoints=2,
        latest_per_run=False,
    )

    assert init not in selected
    assert set(selected) == {old, new}


def test_select_local_checkpoints_defaults_to_latest_snapshot_per_lane(tmp_path: Path) -> None:
    old = tmp_path / "catan-zero-c1_s9700_value_mix_a.iter0003.pt"
    new = tmp_path / "catan-zero-c1_s9700_value_mix_a.iter0006.pt"
    other = tmp_path / "catan-zero-c2_s9702_adaptive_a.iter0003.pt"
    old.write_bytes(b"old")
    new.write_bytes(b"new")
    other.write_bytes(b"other")

    selected = select_local_checkpoints(
        tmp_path,
        run_prefix="s97",
        include_interim=True,
        max_checkpoints=8,
    )

    assert old not in selected
    assert selected == [new, other]


def test_checkpoint_family_treats_final_as_same_lane() -> None:
    assert (
        checkpoint_family_name("s9743_reanalysis_only_value_teacher.final.pt", run_prefix="s97")
        == "s9743_reanalysis_only_value_teacher"
    )
    assert (
        checkpoint_family_name("catan-zero-c1_s9750_vrpo.iter0004.pt", run_prefix="s97")
        == "s9750_vrpo"
    )
    assert (
        checkpoint_family_name("s9805_warmup_jsettlers_agree_c1.warmup0008.pt", run_prefix="s98")
        == "s9805_warmup_jsettlers_agree_c1"
    )
    assert (
        checkpoint_family_name("s9812_reanalysis_value_c2.reanalysis.pt", run_prefix="s98")
        == "s9812_reanalysis_value_c2"
    )


def test_checkpoint_run_number_extracts_prefixed_seed() -> None:
    assert checkpoint_run_number("s9772_branch.iter0002.pt", run_prefix="s97") == 72
    assert (
        checkpoint_run_number(
            "runs/self_play/s9772_branch.iter0002.pt",
            run_prefix="s97",
        )
        == 72
    )
    assert checkpoint_run_number("other_branch.pt", run_prefix="s97") is None


def test_normalize_min_run_number_accepts_absolute_seed() -> None:
    assert normalize_min_run_number(82, run_prefix="s97") == 82
    assert normalize_min_run_number(9782, run_prefix="s97") == 82
    assert normalize_min_run_number(9897, run_prefix="s99") == 0
    assert normalize_min_run_number(6001, run_prefix="s6") == 1
    assert normalize_min_run_number(5999, run_prefix="s6") == 0
    assert checkpoint_run_number("s10009_ema_mixed_antireg_w4d.pt", run_prefix="s") == 10009


def test_remote_poll_command_accepts_multiple_run_prefixes() -> None:
    command = remote_poll_command(remote_repo="", run_prefix="s97,s98")

    assert "prefixes=tuple(part.strip()" in command
    assert "seed_prefixes=tuple(prefix[1:] for prefix in prefixes" in command
    assert "if seed_prefixes and (seed is None" not in command
    assert "'matches_run_prefix': bool(seed and seed_prefixes" in command
    assert "tools/train_ppo.py --seed {prefix[1:]}" not in command
    assert "trainer_features" in command
    assert "'pfsp_mixed': 'pfsp_mixed' in trainer_text" in command
    assert "hashlib.sha1" in command


def test_remote_poll_command_defaults_to_all_s_runs_prefix() -> None:
    command = remote_poll_command(remote_repo="", run_prefix="")

    assert "prefixes=('s',)" in command


def test_plan_remote_gates_skips_active_decided_and_older_snapshots() -> None:
    poll_payload = {
        "workers": [
            {
                "worker": "catan-zero-c3",
                "zone": "us-central1-c",
                "ok": True,
                "candidate_checkpoints": [
                    {"name": "s9750_vrpo_sarsa_candidate.iter0002.pt", "size": 20},
                    {"name": "s9750_vrpo_sarsa_candidate.iter0006.pt", "size": 20},
                ],
            },
            {
                "worker": "catan-zero-w4d",
                "zone": "us-west4-b",
                "ok": True,
                "candidate_checkpoints": [
                    {"name": "s9760_reanalysis_noq.iter0001.pt", "size": 20},
                    {"name": "s9760_reanalysis_noq.iter0003.pt", "size": 20},
                    {"name": "s9761_rejected_lane.pt", "size": 20},
                    {"name": "s9762_rejected_terminal.final.pt", "size": 20},
                    {"name": "s9762_init_lane.init.pt", "size": 20},
                ],
            },
        ],
    }
    summary_payload = {
        "active": [
            {
                "worker": "catan-zero-c3",
                "checkpoint": "runs/self_play/s9750_vrpo_sarsa_candidate.iter0004.pt",
            },
        ],
        "decisions": [
            {
                "checkpoint": "runs/self_play/s9761_rejected_lane.pt",
                "decision": "reject",
            },
            {
                "checkpoint": "runs/self_play/s9762_rejected_terminal.pt",
                "decision": "reject",
            },
        ],
    }

    plan = plan_remote_gates(
        poll_payload=poll_payload,
        summary_payload=summary_payload,
        project="proj",
        remote_repo="",
        run_prefix="s97",
        champion="runs/self_play/champions/current_best.pt",
        eval_dir="runs/self_play/remote_grades_reanalysis",
        log_dir="runs/self_play/logs",
        profile="strict",
        games=4,
        repeats=1,
        grade_workers=1,
        vps_to_win=4,
        max_decisions=300,
        leg_timeout_seconds=1200,
        max_gates=4,
        include_interim=True,
        prefer_prefixes=[],
        min_run_number=0,
        allow_busy_workers=False,
    )

    assert plan["planned_count"] == 1
    planned = plan["planned"][0]
    assert planned["checkpoint"] == "s9760_reanalysis_noq.iter0003.pt"
    assert Path(planned["command"][0]).name.startswith("python")
    assert planned["command"][planned["command"].index("--worker") + 1] == "catan-zero-w4d:us-west4-b"
    assert planned["command"][planned["command"].index("--profile") + 1] == "strict"
    assert "--checkpoint runs/self_play/s9760_reanalysis_noq.iter0003.pt" in planned["shell"]
    assert plan["skipped"]["active_family"][0]["checkpoint"] == "s9750_vrpo_sarsa_candidate.iter0006.pt"
    assert plan["skipped"]["older_snapshot"][0]["checkpoint"] == "s9750_vrpo_sarsa_candidate.iter0002.pt"
    assert plan["skipped"]["decided_checkpoint"][0]["checkpoint"] == "s9761_rejected_lane.pt"
    assert plan["skipped"]["decided_family"][0]["checkpoint"] == "s9762_rejected_terminal.final.pt"
    assert plan["skipped"]["filtered"][0]["checkpoint"] == "s9762_init_lane.init.pt"


def test_plan_remote_gates_ignores_decisions_from_other_profiles() -> None:
    poll_payload = {
        "workers": [
            {
                "worker": "catan-zero-c1",
                "zone": "us-central1-c",
                "ok": True,
                "candidate_checkpoints": [
                    {"name": "s9890_jsettlers_dagger_antireg.iter0002.pt", "size": 28_000_000}
                ],
            }
        ]
    }
    summary_payload = {
        "active": [],
        "decisions": [
            {
                "checkpoint": "runs/self_play/s9890_jsettlers_dagger_antireg.iter0002.pt",
                "decision": "reject",
                "profile": "jsettlers_triage",
                "reason": "opponent regression jsettlers_lite:-0.2500",
            }
        ],
    }

    plan = plan_remote_gates(
        poll_payload=poll_payload,
        summary_payload=summary_payload,
        project="proj",
        remote_repo="",
        run_prefix="s98",
        champion="runs/self_play/champions/current_best.pt",
        eval_dir="runs/self_play/remote_grades_reanalysis",
        log_dir="runs/self_play/logs",
        profile="strict",
        games=4,
        repeats=1,
        grade_workers=1,
        vps_to_win=4,
        max_decisions=300,
        leg_timeout_seconds=1200,
        max_gates=1,
        include_interim=True,
        prefer_prefixes=[],
        min_run_number=0,
        allow_busy_workers=True,
    )

    assert plan["planned_count"] == 1
    assert plan["planned"][0]["checkpoint"] == "s9890_jsettlers_dagger_antireg.iter0002.pt"
    assert plan["planned"][0]["command"][plan["planned"][0]["command"].index("--profile") + 1] == "strict"


def test_plan_remote_gates_can_use_training_busy_grade_free_workers() -> None:
    poll_payload = {
        "workers": [
            {
                "worker": "catan-zero-c1",
                "zone": "us-central1-c",
                "ok": True,
                "running_train_processes": 1,
                "candidate_checkpoints": [
                    {"name": "s9890_jsettlers_dagger_antireg.iter0002.pt", "size": 28_000_000}
                ],
            },
            {
                "worker": "catan-zero-c2",
                "zone": "us-central1-c",
                "ok": True,
                "running_train_processes": 1,
                "candidate_checkpoints": [
                    {"name": "s9891_jsettlers_dagger_antireg.iter0002.pt", "size": 28_000_000}
                ],
            },
        ]
    }
    summary_payload = {
        "active": [
            {
                "worker": "catan-zero-c2",
                "checkpoint": "runs/self_play/other_active.pt",
            }
        ],
        "decisions": [],
    }

    blocked = plan_remote_gates(
        poll_payload=poll_payload,
        summary_payload=summary_payload,
        project="proj",
        remote_repo="",
        run_prefix="s98",
        champion="runs/self_play/champions/current_best.pt",
        eval_dir="runs/self_play/remote_grades_reanalysis",
        log_dir="runs/self_play/logs",
        profile="jsettlers_triage",
        games=2,
        repeats=1,
        grade_workers=1,
        vps_to_win=4,
        max_decisions=220,
        leg_timeout_seconds=600,
        max_gates=2,
        include_interim=True,
        prefer_prefixes=[],
        min_run_number=0,
        allow_busy_workers=False,
    )
    allowed = plan_remote_gates(
        poll_payload=poll_payload,
        summary_payload=summary_payload,
        project="proj",
        remote_repo="",
        run_prefix="s98",
        champion="runs/self_play/champions/current_best.pt",
        eval_dir="runs/self_play/remote_grades_reanalysis",
        log_dir="runs/self_play/logs",
        profile="jsettlers_triage",
        games=2,
        repeats=1,
        grade_workers=1,
        vps_to_win=4,
        max_decisions=220,
        leg_timeout_seconds=600,
        max_gates=2,
        include_interim=True,
        prefer_prefixes=[],
        min_run_number=0,
        allow_busy_workers=False,
        allow_training_busy_workers=True,
    )

    assert blocked["planned_count"] == 0
    assert allowed["planned_count"] == 1
    assert allowed["planned"][0]["worker"] == "catan-zero-c1"


def test_plan_remote_gates_can_include_warmup_snapshots_from_files() -> None:
    poll_payload = {
        "workers": [
            {
                "worker": "catan-zero-c1",
                "zone": "us-central1-c",
                "ok": True,
                "files": [
                    {"name": "s9805_warmup_jsettlers_agree_c1.warmup0008.pt", "size": 20},
                    {"name": "s9805_warmup_jsettlers_agree_c1.warmup0016.pt", "size": 20},
                    {"name": "s9805_warmup_jsettlers_agree_c1.json", "size": 20},
                ],
                "candidate_checkpoints": [],
            },
        ],
    }

    plan = plan_remote_gates(
        poll_payload=poll_payload,
        summary_payload={"active": [], "decisions": []},
        project="proj",
        remote_repo="",
        run_prefix="s98",
        champion="runs/self_play/champions/current_best.pt",
        eval_dir="runs/self_play/remote_grades_reanalysis",
        log_dir="runs/self_play/logs",
        profile="strict",
        games=4,
        repeats=1,
        grade_workers=1,
        vps_to_win=4,
        max_decisions=300,
        leg_timeout_seconds=1200,
        max_gates=4,
        include_interim=True,
        include_warmup=True,
        prefer_prefixes=[],
        min_run_number=1,
        allow_busy_workers=True,
    )

    assert plan["planned_count"] == 1
    assert plan["planned"][0]["checkpoint"] == "s9805_warmup_jsettlers_agree_c1.warmup0016.pt"
    assert plan["planned"][0]["family"] == "s9805_warmup_jsettlers_agree_c1"
    assert plan["skipped"]["older_snapshot"][0]["checkpoint"] == "s9805_warmup_jsettlers_agree_c1.warmup0008.pt"


def test_plan_remote_gates_excludes_warmup_snapshots_by_default() -> None:
    poll_payload = {
        "workers": [
            {
                "worker": "catan-zero-c1",
                "zone": "us-central1-c",
                "ok": True,
                "files": [
                    {"name": "s9805_warmup_jsettlers_agree_c1.warmup0008.pt", "size": 20},
                ],
                "candidate_checkpoints": [],
            },
        ],
    }

    plan = plan_remote_gates(
        poll_payload=poll_payload,
        summary_payload={"active": [], "decisions": []},
        project="proj",
        remote_repo="",
        run_prefix="s98",
        champion="runs/self_play/champions/current_best.pt",
        eval_dir="runs/self_play/remote_grades_reanalysis",
        log_dir="runs/self_play/logs",
        profile="strict",
        games=4,
        repeats=1,
        grade_workers=1,
        vps_to_win=4,
        max_decisions=300,
        leg_timeout_seconds=1200,
        max_gates=4,
        include_interim=True,
        prefer_prefixes=[],
        min_run_number=1,
        allow_busy_workers=True,
    )

    assert plan["planned_count"] == 0


def test_plan_remote_gates_avoids_busy_workers_by_default() -> None:
    poll_payload = {
        "workers": [
            {
                "worker": "catan-zero-c1",
                "zone": "us-central1-c",
                "ok": True,
                "candidate_checkpoints": [
                    {"name": "s9770_new_lane.pt", "size": 20},
                ],
            },
        ],
    }
    summary_payload = {
        "active": [
            {
                "worker": "catan-zero-c1",
                "checkpoint": "runs/self_play/s9700_other_lane.pt",
            },
        ],
        "decisions": [],
    }

    plan = plan_remote_gates(
        poll_payload=poll_payload,
        summary_payload=summary_payload,
        project="proj",
        remote_repo="",
        run_prefix="s97",
        champion="champion.pt",
        eval_dir="grades",
        log_dir="logs",
        profile="strict",
        games=4,
        repeats=1,
        grade_workers=1,
        vps_to_win=4,
        max_decisions=300,
        leg_timeout_seconds=1200,
        max_gates=4,
        include_interim=False,
        prefer_prefixes=[],
        min_run_number=0,
        allow_busy_workers=False,
    )

    assert plan["planned_count"] == 0
    assert plan["skipped"]["busy_worker"][0]["checkpoint"] == "s9770_new_lane.pt"


def test_plan_remote_gates_avoids_training_workers_by_default() -> None:
    poll_payload = {
        "workers": [
            {
                "worker": "catan-zero-c2",
                "zone": "us-central1-c",
                "ok": True,
                "running_train_processes": 1,
                "candidate_checkpoints": [
                    {"name": "s9771_new_lane.pt", "size": 20},
                ],
            },
        ],
    }
    summary_payload = {"active": [], "decisions": []}

    plan = plan_remote_gates(
        poll_payload=poll_payload,
        summary_payload=summary_payload,
        project="proj",
        remote_repo="",
        run_prefix="s97",
        champion="champion.pt",
        eval_dir="grades",
        log_dir="logs",
        profile="strict",
        games=4,
        repeats=1,
        grade_workers=1,
        vps_to_win=4,
        max_decisions=300,
        leg_timeout_seconds=1200,
        max_gates=4,
        include_interim=False,
        prefer_prefixes=[],
        min_run_number=0,
        allow_busy_workers=False,
    )

    assert plan["planned_count"] == 0
    assert plan["skipped"]["busy_worker"][0]["checkpoint"] == "s9771_new_lane.pt"


def test_plan_remote_gates_prefers_idle_duplicate_checkpoint() -> None:
    poll_payload = {
        "workers": [
            {
                "worker": "catan-zero-c1",
                "zone": "us-central1-c",
                "ok": True,
                "running_train_processes": 1,
                "candidate_checkpoints": [
                    {"name": "s9875_blend_candidate.pt", "size": 20},
                ],
            },
            {
                "worker": "catan-zero-w4d",
                "zone": "us-west4-b",
                "ok": True,
                "running_train_processes": 0,
                "candidate_checkpoints": [
                    {"name": "s9875_blend_candidate.pt", "size": 20},
                ],
            },
        ],
    }
    summary_payload = {"active": [], "decisions": []}

    plan = plan_remote_gates(
        poll_payload=poll_payload,
        summary_payload=summary_payload,
        project="proj",
        remote_repo="",
        run_prefix="s98",
        champion="champion.pt",
        eval_dir="grades",
        log_dir="logs",
        profile="strict",
        games=4,
        repeats=1,
        grade_workers=1,
        vps_to_win=4,
        max_decisions=300,
        leg_timeout_seconds=1200,
        max_gates=4,
        include_interim=True,
        prefer_prefixes=[],
        min_run_number=0,
        allow_busy_workers=False,
    )

    assert plan["planned_count"] == 1
    assert plan["planned"][0]["worker"] == "catan-zero-w4d"
    assert plan["planned"][0]["checkpoint"] == "s9875_blend_candidate.pt"
    assert plan["skipped"]["older_snapshot"][0]["worker"] == "catan-zero-c1"


def test_plan_remote_gates_can_keep_multiple_snapshots_per_family() -> None:
    poll_payload = {
        "workers": [
            {
                "worker": "catan-zero-c1",
                "zone": "us-central1-c",
                "ok": True,
                "candidate_checkpoints": [
                    {"name": "s9783_branch.iter0002.pt", "size": 20},
                    {"name": "s9783_branch.iter0004.pt", "size": 20},
                    {"name": "s9783_branch.iter0006.pt", "size": 20},
                ],
            },
        ],
    }
    summary_payload = {"active": [], "decisions": []}

    default_plan = plan_remote_gates(
        poll_payload=poll_payload,
        summary_payload=summary_payload,
        project="proj",
        remote_repo="",
        run_prefix="s97",
        champion="champion.pt",
        eval_dir="grades",
        log_dir="logs",
        profile="strict",
        games=4,
        repeats=1,
        grade_workers=1,
        vps_to_win=4,
        max_decisions=300,
        leg_timeout_seconds=1200,
        max_gates=4,
        include_interim=True,
        prefer_prefixes=[],
        min_run_number=0,
        allow_busy_workers=True,
    )
    two_snapshot_plan = plan_remote_gates(
        poll_payload=poll_payload,
        summary_payload=summary_payload,
        project="proj",
        remote_repo="",
        run_prefix="s97",
        champion="champion.pt",
        eval_dir="grades",
        log_dir="logs",
        profile="strict",
        games=4,
        repeats=1,
        grade_workers=1,
        vps_to_win=4,
        max_decisions=300,
        leg_timeout_seconds=1200,
        max_gates=4,
        include_interim=True,
        prefer_prefixes=[],
        min_run_number=0,
        allow_busy_workers=True,
        max_per_family=2,
    )

    assert [row["checkpoint"] for row in default_plan["planned"]] == [
        "s9783_branch.iter0006.pt"
    ]
    assert [row["checkpoint"] for row in two_snapshot_plan["planned"]] == [
        "s9783_branch.iter0006.pt",
        "s9783_branch.iter0004.pt",
    ]
    assert two_snapshot_plan["skipped"]["older_snapshot"][0]["checkpoint"] == (
        "s9783_branch.iter0002.pt"
    )


def test_plan_remote_gates_filters_old_run_numbers() -> None:
    poll_payload = {
        "workers": [
            {
                "worker": "catan-zero-c1",
                "zone": "us-central1-c",
                "ok": True,
                "candidate_checkpoints": [
                    {"name": "s9722_old_lane.pt", "size": 20},
                    {"name": "s9772_current_lane.iter0002.pt", "size": 20},
                ],
            },
        ],
    }
    summary_payload = {"active": [], "decisions": []}

    plan = plan_remote_gates(
        poll_payload=poll_payload,
        summary_payload=summary_payload,
        project="proj",
        remote_repo="",
        run_prefix="s97",
        champion="champion.pt",
        eval_dir="grades",
        log_dir="logs",
        profile="strict",
        games=4,
        repeats=1,
        grade_workers=1,
        vps_to_win=4,
        max_decisions=300,
        leg_timeout_seconds=1200,
        max_gates=4,
        include_interim=True,
        prefer_prefixes=[],
        min_run_number=9770,
        allow_busy_workers=False,
    )

    assert plan["planned_count"] == 1
    assert plan["effective_min_run_number"] == 70
    assert plan["planned"][0]["checkpoint"] == "s9772_current_lane.iter0002.pt"
    assert plan["skipped"]["filtered"][0]["checkpoint"] == "s9722_old_lane.pt"


def test_plan_remote_gates_skips_checkpoints_from_other_run_prefixes() -> None:
    poll_payload = {
        "workers": [
            {
                "worker": "catan-zero-c1",
                "zone": "us-central1-c",
                "ok": True,
                "candidate_checkpoints": [
                    {"name": "s9772_old_prefix.pt", "size": 20},
                    {"name": "s9803_current_prefix.iter0008.pt", "size": 20},
                ],
            },
        ],
    }
    summary_payload = {"active": [], "decisions": []}

    plan = plan_remote_gates(
        poll_payload=poll_payload,
        summary_payload=summary_payload,
        project="proj",
        remote_repo="",
        run_prefix="s98",
        champion="champion.pt",
        eval_dir="grades",
        log_dir="logs",
        profile="strict",
        games=4,
        repeats=1,
        grade_workers=1,
        vps_to_win=4,
        max_decisions=300,
        leg_timeout_seconds=1200,
        max_gates=4,
        include_interim=True,
        prefer_prefixes=[],
        min_run_number=1,
        allow_busy_workers=False,
    )

    assert plan["planned_count"] == 1
    assert plan["planned"][0]["checkpoint"] == "s9803_current_prefix.iter0008.pt"
    assert plan["skipped"]["filtered"][0] == {
        "worker": "catan-zero-c1",
        "checkpoint": "s9772_old_prefix.pt",
        "reason": "run_prefix_mismatch",
    }


def test_plan_remote_gates_skips_rejected_regression_snapshots() -> None:
    poll_payload = {
        "workers": [
            {
                "worker": "catan-zero-c4",
                "zone": "us-central1-c",
                "ok": True,
                "candidate_checkpoints": [
                    {"name": "s9772_bad_family.iter0002.pt", "size": 20},
                    {"name": "s9773_clean_family.pt", "size": 20},
                ],
            },
        ],
    }
    summary_payload = {
        "active": [],
        "decisions": [
            {
                "checkpoint": "runs/self_play/s9772_bad_family.iter0004.pt",
                "decision": "reject",
                "reason": "opponent regression jsettlers_lite:-0.2500",
            },
        ],
    }

    plan = plan_remote_gates(
        poll_payload=poll_payload,
        summary_payload=summary_payload,
        project="proj",
        remote_repo="",
        run_prefix="s97",
        champion="champion.pt",
        eval_dir="grades",
        log_dir="logs",
        profile="strict",
        games=4,
        repeats=1,
        grade_workers=1,
        vps_to_win=4,
        max_decisions=300,
        leg_timeout_seconds=1200,
        max_gates=4,
        include_interim=True,
        prefer_prefixes=[],
        min_run_number=0,
        allow_busy_workers=False,
    )

    assert plan["planned_count"] == 1
    assert plan["planned"][0]["checkpoint"] == "s9773_clean_family.pt"
    assert plan["skipped"]["rejected_regression_family"][0]["checkpoint"] == "s9772_bad_family.iter0002.pt"


def test_plan_remote_gates_blocks_newer_snapshot_after_rejected_regression_by_default() -> None:
    poll_payload = {
        "workers": [
            {
                "worker": "catan-zero-c4",
                "zone": "us-central1-c",
                "ok": True,
                "candidate_checkpoints": [
                    {"name": "s9772_recovering_family.iter0002.pt", "size": 20},
                    {"name": "s9772_recovering_family.iter0006.pt", "size": 20},
                ],
            },
        ],
    }
    summary_payload = {
        "active": [],
        "decisions": [
            {
                "checkpoint": "runs/self_play/s9772_recovering_family.iter0002.pt",
                "decision": "reject",
                "reason": "opponent regression jsettlers_lite:-0.2500",
            },
        ],
    }

    plan = plan_remote_gates(
        poll_payload=poll_payload,
        summary_payload=summary_payload,
        project="proj",
        remote_repo="",
        run_prefix="s97",
        champion="champion.pt",
        eval_dir="grades",
        log_dir="logs",
        profile="strict",
        games=4,
        repeats=1,
        grade_workers=1,
        vps_to_win=4,
        max_decisions=300,
        leg_timeout_seconds=1200,
        max_gates=4,
        include_interim=True,
        prefer_prefixes=[],
        min_run_number=0,
        allow_busy_workers=False,
    )

    assert plan["planned_count"] == 0
    assert plan["skipped"]["rejected_regression_family"][0]["checkpoint"] == (
        "s9772_recovering_family.iter0006.pt"
    )
    assert plan["skipped"]["older_snapshot"][0]["checkpoint"] == "s9772_recovering_family.iter0002.pt"


def test_plan_remote_gates_can_override_rejected_family_block() -> None:
    poll_payload = {
        "workers": [
            {
                "worker": "catan-zero-c4",
                "zone": "us-central1-c",
                "ok": True,
                "candidate_checkpoints": [
                    {"name": "s9772_recovering_family.iter0002.pt", "size": 20},
                    {"name": "s9772_recovering_family.iter0006.pt", "size": 20},
                ],
            },
        ],
    }
    summary_payload = {
        "active": [],
        "decisions": [
            {
                "checkpoint": "runs/self_play/s9772_recovering_family.iter0002.pt",
                "decision": "reject",
                "reason": "opponent regression jsettlers_lite:-0.2500",
            },
        ],
    }

    plan = plan_remote_gates(
        poll_payload=poll_payload,
        summary_payload=summary_payload,
        project="proj",
        remote_repo="",
        run_prefix="s97",
        champion="champion.pt",
        eval_dir="grades",
        log_dir="logs",
        profile="strict",
        games=4,
        repeats=1,
        grade_workers=1,
        vps_to_win=4,
        max_decisions=300,
        leg_timeout_seconds=1200,
        max_gates=4,
        include_interim=True,
        prefer_prefixes=[],
        min_run_number=0,
        allow_busy_workers=False,
        allow_rejected_family_continuation=True,
    )

    assert plan["planned_count"] == 1
    assert plan["planned"][0]["checkpoint"] == "s9772_recovering_family.iter0006.pt"


def test_plan_remote_gates_blocks_timeout_reject_family() -> None:
    poll_payload = {
        "workers": [
            {
                "worker": "catan-zero-c4",
                "zone": "us-central1-c",
                "ok": True,
                "candidate_checkpoints": [
                    {"name": "s9886_timeout_family.iter0006.pt", "size": 20},
                    {"name": "s9887_clean_family.iter0002.pt", "size": 20},
                ],
            },
        ],
    }
    summary_payload = {
        "active": [],
        "decisions": [
            {
                "checkpoint": "runs/self_play/s9886_timeout_family.iter0002.pt",
                "decision": "reject",
                "reason": "candidate timed out in 2 grade legs",
            },
        ],
    }

    plan = plan_remote_gates(
        poll_payload=poll_payload,
        summary_payload=summary_payload,
        project="proj",
        remote_repo="",
        run_prefix="s98",
        champion="champion.pt",
        eval_dir="grades",
        log_dir="logs",
        profile="strict",
        games=4,
        repeats=1,
        grade_workers=1,
        vps_to_win=4,
        max_decisions=300,
        leg_timeout_seconds=1200,
        max_gates=4,
        include_interim=True,
        prefer_prefixes=[],
        min_run_number=0,
        allow_busy_workers=False,
    )

    assert plan["planned_count"] == 1
    assert plan["planned"][0]["checkpoint"] == "s9887_clean_family.iter0002.pt"
    assert plan["skipped"]["rejected_regression_family"][0]["checkpoint"] == (
        "s9886_timeout_family.iter0006.pt"
    )


def test_plan_remote_transfer_gates_moves_busy_source_checkpoint_to_idle_target() -> None:
    poll_payload = {
        "workers": [
            {
                "worker": "catan-zero-w1a",
                "zone": "us-west1-b",
                "ok": True,
                "repo": "/home/worker/catan-zero/catan-zero-gcp-bundle",
                "running_train_processes": 1,
                "candidate_checkpoints": [
                    {
                        "name": "s9874_weighted_dagger_antireg_w1a.iter0002.pt",
                        "size": 20,
                    },
                ],
            },
            {
                "worker": "catan-zero-c3",
                "zone": "us-central1-c",
                "ok": True,
                "repo": "/home/worker/catan-zero",
                "running_train_processes": 0,
                "candidate_checkpoints": [],
            },
        ],
    }

    plan = plan_remote_transfer_gates(
        poll_payload=poll_payload,
        summary_payload={"active": [], "decisions": []},
        project="proj",
        remote_repo="",
        run_prefix="s98",
        champion="runs/self_play/champions/current_best_s9752_iter0002.pt",
        eval_dir="grades",
        log_dir="logs",
        profile="strict",
        games=4,
        repeats=1,
        grade_workers=1,
        vps_to_win=4,
        max_decisions=300,
        leg_timeout_seconds=1200,
        max_gates=4,
        include_interim=True,
        prefer_prefixes=["s9874_weighted_dagger"],
        min_run_number=9874,
        allow_busy_target_workers=False,
    )

    assert plan["planned_count"] == 1
    planned = plan["planned"][0]
    assert planned["source_worker"] == "catan-zero-w1a"
    assert planned["target_worker"] == "catan-zero-c3"
    assert planned["checkpoint_path"] == (
        "runs/self_play/s9874_weighted_dagger_antireg_w1a.iter0002.pt"
    )
    assert planned["command"][planned["command"].index("--worker") + 1] == (
        "catan-zero-c3:us-central1-c"
    )
    assert planned["command"][planned["command"].index("--source-worker") + 1] == (
        "catan-zero-w1a:us-west1-b"
    )
    assert planned["command"][planned["command"].index("--source-remote-repo") + 1] == (
        "/home/worker/catan-zero/catan-zero-gcp-bundle"
    )
    assert planned["command"][planned["command"].index("--target-remote-repo") + 1] == (
        "/home/worker/catan-zero"
    )
    assert plan["skipped"]["busy_target_worker"][0]["reason"] == "training"


def test_plan_remote_transfer_gates_normalizes_absolute_floor_below_prefix() -> None:
    poll_payload = {
        "workers": [
            {
                "worker": "catan-zero-c1",
                "zone": "us-central1-c",
                "ok": True,
                "repo": "/home/worker/catan-zero",
                "running_train_processes": 1,
                "candidate_checkpoints": [
                    {
                        "name": "s9900_ema_jsettlers_antireg_c1.pt",
                        "size": 20,
                    },
                ],
            },
            {
                "worker": "catan-zero-w1a",
                "zone": "us-west1-b",
                "ok": True,
                "repo": "/home/worker/catan-zero",
                "running_train_processes": 0,
                "candidate_checkpoints": [],
            },
        ],
    }

    plan = plan_remote_transfer_gates(
        poll_payload=poll_payload,
        summary_payload={"active": [], "decisions": []},
        project="proj",
        remote_repo="",
        run_prefix="s99",
        champion="runs/self_play/champions/current_best_s9752_iter0002.pt",
        eval_dir="grades",
        log_dir="logs",
        profile="strict",
        games=4,
        repeats=1,
        grade_workers=1,
        vps_to_win=4,
        max_decisions=300,
        leg_timeout_seconds=1200,
        max_gates=4,
        include_interim=True,
        prefer_prefixes=["s9900"],
        min_run_number=9897,
        allow_busy_target_workers=False,
        allow_training_busy_target_workers=True,
    )

    assert plan["effective_min_run_number"] == 0
    assert plan["planned_count"] == 1
    assert plan["planned"][0]["checkpoint"] == "s9900_ema_jsettlers_antireg_c1.pt"


def test_plan_remote_transfer_gates_skips_when_all_targets_busy() -> None:
    poll_payload = {
        "workers": [
            {
                "worker": "catan-zero-w1a",
                "zone": "us-west1-b",
                "ok": True,
                "running_train_processes": 1,
                "candidate_checkpoints": [
                    {
                        "name": "s9874_weighted_dagger_antireg_w1a.iter0002.pt",
                        "size": 20,
                    },
                ],
            },
            {
                "worker": "catan-zero-c3",
                "zone": "us-central1-c",
                "ok": True,
                "running_train_processes": 0,
                "candidate_checkpoints": [],
            },
        ],
    }
    summary_payload = {
        "active": [
            {
                "worker": "catan-zero-c3",
                "checkpoint": "runs/self_play/s9860_busy_gate.pt",
            },
        ],
        "decisions": [],
    }

    plan = plan_remote_transfer_gates(
        poll_payload=poll_payload,
        summary_payload=summary_payload,
        project="proj",
        remote_repo="",
        run_prefix="s98",
        champion="champion.pt",
        eval_dir="grades",
        log_dir="logs",
        profile="strict",
        games=4,
        repeats=1,
        grade_workers=1,
        vps_to_win=4,
        max_decisions=300,
        leg_timeout_seconds=1200,
        max_gates=4,
        include_interim=True,
        prefer_prefixes=[],
        min_run_number=9874,
        allow_busy_target_workers=False,
    )

    assert plan["planned_count"] == 0
    assert plan["eligible_targets"] == []
    assert plan["skipped"]["no_target_worker"][0]["checkpoint"] == (
        "s9874_weighted_dagger_antireg_w1a.iter0002.pt"
    )
    assert {row["reason"] for row in plan["skipped"]["busy_target_worker"]} == {
        "training",
        "remote_grade",
    }


def test_plan_remote_transfer_gates_can_use_training_busy_grade_free_target() -> None:
    poll_payload = {
        "workers": [
            {
                "worker": "catan-zero-w1a",
                "zone": "us-west1-b",
                "ok": True,
                "running_train_processes": 1,
                "candidate_checkpoints": [
                    {
                        "name": "s9890_jsettlers_dagger_antireg_w1a.iter0004.pt",
                        "size": 20,
                    },
                ],
            },
            {
                "worker": "catan-zero-c2",
                "zone": "us-central1-c",
                "ok": True,
                "running_train_processes": 1,
                "candidate_checkpoints": [],
            },
            {
                "worker": "catan-zero-c3",
                "zone": "us-central1-c",
                "ok": True,
                "running_train_processes": 1,
                "candidate_checkpoints": [],
            },
        ],
    }
    summary_payload = {
        "active": [
            {
                "worker": "catan-zero-c3",
                "checkpoint": "runs/self_play/s9886_active_grade.iter0002.pt",
            },
        ],
        "decisions": [],
    }

    plan = plan_remote_transfer_gates(
        poll_payload=poll_payload,
        summary_payload=summary_payload,
        project="proj",
        remote_repo="",
        run_prefix="s98",
        champion="champion.pt",
        eval_dir="grades",
        log_dir="logs",
        profile="strict",
        games=4,
        repeats=1,
        grade_workers=1,
        vps_to_win=4,
        max_decisions=300,
        leg_timeout_seconds=1200,
        max_gates=4,
        include_interim=True,
        prefer_prefixes=[],
        min_run_number=9890,
        allow_busy_target_workers=False,
        allow_training_busy_target_workers=True,
    )

    assert plan["planned_count"] == 1
    assert plan["planned"][0]["target_worker"] == "catan-zero-c2"
    assert {
        (row["worker"], row["reason"])
        for row in plan["skipped"]["busy_target_worker"]
    } == {("catan-zero-c3", "remote_grade")}


def test_plan_remote_transfer_gates_does_not_duplicate_active_or_decided() -> None:
    poll_payload = {
        "workers": [
            {
                "worker": "catan-zero-w1a",
                "zone": "us-west1-b",
                "ok": True,
                "running_train_processes": 1,
                "candidate_checkpoints": [
                    {"name": "s9874_active_branch.iter0002.pt", "size": 20},
                    {"name": "s9875_decided_branch.pt", "size": 20},
                    {"name": "s9876_clean_branch.iter0002.pt", "size": 20},
                ],
            },
            {
                "worker": "catan-zero-c3",
                "zone": "us-central1-c",
                "ok": True,
                "running_train_processes": 0,
                "candidate_checkpoints": [],
            },
        ],
    }
    summary_payload = {
        "active": [
            {
                "worker": "catan-zero-c4",
                "checkpoint": "runs/self_play/s9874_active_branch.iter0002.pt",
            },
        ],
        "decisions": [
            {
                "checkpoint": "runs/self_play/s9875_decided_branch.pt",
                "decision": "reject",
            },
        ],
    }

    plan = plan_remote_transfer_gates(
        poll_payload=poll_payload,
        summary_payload=summary_payload,
        project="proj",
        remote_repo="",
        run_prefix="s98",
        champion="champion.pt",
        eval_dir="grades",
        log_dir="logs",
        profile="strict",
        games=4,
        repeats=1,
        grade_workers=1,
        vps_to_win=4,
        max_decisions=300,
        leg_timeout_seconds=1200,
        max_gates=4,
        include_interim=True,
        prefer_prefixes=[],
        min_run_number=9874,
        allow_busy_target_workers=False,
    )

    assert plan["planned_count"] == 1
    assert plan["planned"][0]["checkpoint"] == "s9876_clean_branch.iter0002.pt"
    assert plan["skipped"]["active_checkpoint"][0]["checkpoint"] == (
        "s9874_active_branch.iter0002.pt"
    )
    assert plan["skipped"]["decided_checkpoint"][0]["checkpoint"] == (
        "s9875_decided_branch.pt"
    )


def test_plan_remote_transfer_gates_blocks_timeout_reject_family() -> None:
    poll_payload = {
        "workers": [
            {
                "worker": "catan-zero-w1a",
                "zone": "us-west1-b",
                "ok": True,
                "running_train_processes": 1,
                "candidate_checkpoints": [
                    {"name": "s9891_timeout_branch.iter0006.pt", "size": 20},
                    {"name": "s9892_clean_branch.iter0004.pt", "size": 20},
                ],
            },
            {
                "worker": "catan-zero-c3",
                "zone": "us-central1-c",
                "ok": True,
                "running_train_processes": 0,
                "candidate_checkpoints": [],
            },
        ],
    }
    summary_payload = {
        "active": [],
        "decisions": [
            {
                "checkpoint": "runs/self_play/s9891_timeout_branch.iter0002.pt",
                "decision": "reject",
                "reason": "candidate timed out in 2 grade legs",
            },
        ],
    }

    plan = plan_remote_transfer_gates(
        poll_payload=poll_payload,
        summary_payload=summary_payload,
        project="proj",
        remote_repo="",
        run_prefix="s98",
        champion="champion.pt",
        eval_dir="grades",
        log_dir="logs",
        profile="strict",
        games=4,
        repeats=1,
        grade_workers=1,
        vps_to_win=4,
        max_decisions=300,
        leg_timeout_seconds=1200,
        max_gates=4,
        include_interim=True,
        prefer_prefixes=[],
        min_run_number=9891,
        allow_busy_target_workers=False,
    )

    assert plan["planned_count"] == 1
    assert plan["planned"][0]["checkpoint"] == "s9892_clean_branch.iter0004.pt"
    assert plan["skipped"]["rejected_regression_family"][0]["checkpoint"] == (
        "s9891_timeout_branch.iter0006.pt"
    )
