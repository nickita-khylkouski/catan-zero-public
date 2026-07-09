from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.league_orchestrator import (
    DEFAULT_BRANCHES,
    _parse_poll_stdout,
    build_eval_command,
    build_branch_specs,
    build_manifest,
    build_remote_launch_command,
    cleanup_local_wrappers,
    gate_checkpoint,
    write_manifest,
    read_manifest,
    should_promote_gate,
)


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        manifest=str(tmp_path / "manifest.json"),
        bundle="/tmp/catan-zero-bundle.tar.gz",
        init_checkpoint="runs/self_play/champions/current_best_s9752_iter0002.pt",
        base_seed=6000,
        vps_to_win=6,
        max_decisions=600,
        iterations=100,
        episodes_per_iteration=12,
        checkpoint_every=10,
        checkpoint_eval_games=0,
        checkpoint_eval_value_games=0,
        boxes=["bx_a", "bx_b", "bx_c"],
        branch=None,
    )


def test_default_branch_specs_build_long_run_commands(tmp_path) -> None:
    args = _args(tmp_path)
    specs = build_branch_specs(args)

    assert tuple(spec.name for spec in specs) == (
        "s6001_adaptive_ema_qoff",
        "s6002_search_ema_dagger_qoff",
        "s6003_allseat_lowkl_qoff",
    )
    assert len(specs) == len(DEFAULT_BRANCHES)
    for spec in specs:
        command = spec.command
        assert "--init-checkpoint" in command
        assert "--iterations" in command
        assert command[command.index("--iterations") + 1] == "100"
        assert "--vps-to-win" in command
        assert command[command.index("--vps-to-win") + 1] == "6"
        assert "--checkpoint-every" in command
        assert command[command.index("--checkpoint-every") + 1] == "10"
        assert spec.remote_dir == f"/tmp/catan-zero-{spec.name}"


def test_branch_types_encode_distinct_training_pressure(tmp_path) -> None:
    specs = build_branch_specs(_args(tmp_path))
    commands = {spec.name.split("_", 1)[1]: spec.command for spec in specs}

    adaptive = commands["adaptive_ema_qoff"]
    assert adaptive[adaptive.index("--opponents") + 1] == "adaptive_league"
    assert float(adaptive[adaptive.index("--q-value-coef") + 1]) == 0.0
    assert float(adaptive[adaptive.index("--q-advantage-mix") + 1]) == 0.0
    assert float(adaptive[adaptive.index("--ema-policy-kl-coef") + 1]) > 0.0

    search = commands["search_ema_dagger_qoff"]
    assert search[search.index("--opponents") + 1] == "search_mixed"
    assert float(search[search.index("--q-advantage-mix") + 1]) == 0.0
    assert int(search[search.index("--dagger-games-per-iteration") + 1]) >= 2

    allseat = commands["allseat_lowkl_qoff"]
    assert allseat[allseat.index("--opponents") + 1] == "self"
    assert allseat[allseat.index("--learner-seats") + 1] == "all"
    assert float(allseat[allseat.index("--q-advantage-mix") + 1]) == 0.0


def test_remote_launch_uses_nohup_setsid_and_manifest_round_trip(tmp_path) -> None:
    args = _args(tmp_path)
    specs = build_branch_specs(args)
    remote = build_remote_launch_command(specs[0], bundle_name="bundle.tgz")

    assert "setsid sh -c" in remote
    assert "> runs/self_play/logs/s6001_adaptive_ema_qoff.log 2>&1 < /dev/null' >/dev/null 2>&1 & exit 0" in remote
    assert "tools/train_ppo.py" in remote

    manifest = build_manifest(args, specs)
    path = tmp_path / "manifest.json"
    write_manifest(manifest, path)
    assert read_manifest(path) == json.loads(path.read_text())


def test_cleanup_local_wrappers_dry_run_does_not_kill_processes() -> None:
    result = cleanup_local_wrappers(dry_run=True)

    assert result["dry_run"] is True
    assert result["before"] == result["after"]


def test_parse_poll_stdout_extracts_running_state_and_checkpoints() -> None:
    parsed = _parse_poll_stdout(
        "123 bash -c pgrep -af 'train_ppo.py --seed 1'\n"
        "456 .venv/bin/python -u tools/train_ppo.py --seed 1\n"
        "__CHECKPOINTS__\n"
        "s6001_league_vrpo.iter0010.pt 28184675\n"
        "__LOG__\n"
        '{"ppo": {"iteration": 10}}\n'
    )

    assert parsed["running"] is True
    assert parsed["processes"] == [
        "456 .venv/bin/python -u tools/train_ppo.py --seed 1"
    ]
    assert parsed["checkpoints"] == [
        {"file": "s6001_league_vrpo.iter0010.pt", "bytes": 28184675}
    ]
    assert parsed["log_tail"] == ['{"ppo": {"iteration": 10}}']


def test_gate_promotion_requires_no_regression_and_strict_improvement() -> None:
    assert should_promote_gate(
        candidate_heuristic_wins=19,
        candidate_value_wins=11,
        champion_heuristic_wins=18,
        champion_value_wins=11,
    ) == (True, "candidate improved aggregate without regression")
    assert should_promote_gate(
        candidate_heuristic_wins=18,
        candidate_value_wins=12,
        champion_heuristic_wins=18,
        champion_value_wins=11,
    ) == (True, "candidate improved aggregate without regression")
    assert should_promote_gate(
        candidate_heuristic_wins=18,
        candidate_value_wins=11,
        champion_heuristic_wins=18,
        champion_value_wins=11,
    ) == (False, "candidate tied champion aggregate")
    assert should_promote_gate(
        candidate_heuristic_wins=20,
        candidate_value_wins=10,
        champion_heuristic_wins=18,
        champion_value_wins=11,
    ) == (False, "value regression")
    assert should_promote_gate(
        candidate_heuristic_wins=17,
        candidate_value_wins=12,
        champion_heuristic_wins=18,
        champion_value_wins=11,
    ) == (False, "heuristic regression")


def test_build_eval_command_uses_parallel_gate_settings(tmp_path) -> None:
    command = build_eval_command(
        checkpoint=tmp_path / "candidate.pt",
        opponent="value",
        games=32,
        seed=85002,
        vps_to_win=3,
        max_decisions=300,
        workers=4,
        output=tmp_path / "eval.json",
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
        "32",
        "--seed",
        "85002",
        "--vps-to-win",
        "3",
        "--max-decisions",
        "300",
        "--workers",
        "4",
        "--output",
        str(tmp_path / "eval.json"),
    ]


def test_gate_checkpoint_can_evaluate_champion_on_same_legs(monkeypatch, tmp_path) -> None:
    reports = {
        "common_candidate_vs_catanatron_ab316_s1.json": {"wins": 5},
        "common_candidate_vs_catanatron_search16_s2.json": {"wins": 2},
        "verify2_candidate_vs_jsettlers_lite16_s3.json": {"wins": 2},
        "verify2_candidate_vs_value16_s4.json": {"wins": 5},
        "common_champion_vs_catanatron_ab316_s1.json": {"wins": 4},
        "common_champion_vs_catanatron_search16_s2.json": {"wins": 2},
        "verify2_champion_vs_jsettlers_lite16_s3.json": {"wins": 2},
        "verify2_champion_vs_value16_s4.json": {"wins": 3},
    }

    def fake_run(command, *, dry_run, check=True):
        output = Path(command[-1])
        output.write_text(json.dumps(reports[output.name]), encoding="utf-8")

    monkeypatch.setattr("tools.league_orchestrator.run", fake_run)

    summary = gate_checkpoint(
        checkpoint=tmp_path / "candidate.pt",
        eval_dir=tmp_path,
        games=16,
        workers=1,
        vps_to_win=3,
        max_decisions=300,
        common_heuristic_seed=1,
        common_value_seed=2,
        verify_heuristic_seed=3,
        verify_value_seed=4,
        champion_heuristic_wins=999,
        champion_value_wins=999,
        champion=tmp_path / "champion.pt",
        evaluate_champion=True,
        dry_run=False,
    )

    assert summary["candidate_heuristic_wins"] == 7
    assert summary["candidate_value_wins"] == 7
    assert summary["champion_heuristic_wins"] == 6
    assert summary["champion_value_wins"] == 5
    assert summary["promote"] is True
    assert summary["champion_reports"]
