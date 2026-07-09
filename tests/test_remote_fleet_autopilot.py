from __future__ import annotations

import argparse

import tools.remote_fleet_autopilot as autopilot
from tools.remote_fleet_autopilot import (
    GATE_MIN_RUN_NUMBER,
    STRICT_PREFER_PREFIXES,
    TRIAGE_PREFER_PREFIXES,
    build_parser,
    launch_planned,
    ordered_launch_plans,
    plan_opening_evals,
    prefer_prefix_args,
)


def test_prefer_prefixes_prioritize_fresh_runs() -> None:
    assert GATE_MIN_RUN_NUMBER == 9900
    assert TRIAGE_PREFER_PREFIXES[:2] == ("s100", "s99")
    assert STRICT_PREFER_PREFIXES[:2] == ("s100", "s99")
    assert prefer_prefix_args(("s100", "s99")) == [
        "--prefer-prefix",
        "s100",
        "--prefer-prefix",
        "s99",
    ]


def test_ordered_launch_plans_prioritizes_training_when_under_target() -> None:
    train = {"name": "train"}
    escalation = {"name": "escalation"}
    triage = {"name": "triage"}
    strict = {"name": "strict"}
    transfer = {"name": "transfer"}
    opening = {"name": "opening"}

    assert ordered_launch_plans(
        training_processes=5,
        target_training_processes=10,
        train_plan=train,
        escalation_plan=escalation,
        triage_gate_plan=triage,
        gate_plan=strict,
        transfer_gate_plan=transfer,
        opening_eval_plan=opening,
    ) == [train, escalation, strict, transfer, opening, triage]


def test_ordered_launch_plans_prioritizes_gates_when_training_full() -> None:
    train = {"name": "train"}
    escalation = {"name": "escalation"}
    triage = {"name": "triage"}
    strict = {"name": "strict"}
    transfer = {"name": "transfer"}
    opening = {"name": "opening"}

    assert ordered_launch_plans(
        training_processes=10,
        target_training_processes=10,
        train_plan=train,
        escalation_plan=escalation,
        triage_gate_plan=triage,
        gate_plan=strict,
        transfer_gate_plan=transfer,
        opening_eval_plan=opening,
    ) == [escalation, strict, transfer, opening, triage, train]


def test_autopilot_busy_worker_overrides_are_opt_in(monkeypatch) -> None:
    commands = []

    def fake_run(command, *, capture):
        commands.append(command)

    monkeypatch.setattr(autopilot, "run", fake_run)
    monkeypatch.setattr(autopilot, "read_json", lambda path: {})
    args = argparse.Namespace(
        run_prefix="s",
        max_gates=3,
        allow_training_busy_gates=False,
        allow_grade_busy_training=False,
        train_iterations=8,
        episodes_per_iteration=10,
        checkpoint_every=2,
        max_train_launches=4,
    )

    autopilot.plan_gates(args)
    autopilot.plan_train(args, recipe="vrpo_esarsa_antireg")

    flattened = [part for command in commands for part in command]
    assert "--allow-training-busy-workers" not in flattened
    assert "--allow-grade-busy-workers" not in flattened


def test_autopilot_parser_defaults_do_not_use_busy_worker_overrides() -> None:
    args = build_parser().parse_args([])

    assert args.allow_training_busy_gates is False
    assert args.allow_grade_busy_training is False


def test_autopilot_busy_worker_overrides_can_be_enabled(monkeypatch) -> None:
    commands = []

    def fake_run(command, *, capture):
        commands.append(command)

    monkeypatch.setattr(autopilot, "run", fake_run)
    monkeypatch.setattr(autopilot, "read_json", lambda path: {})
    args = argparse.Namespace(
        run_prefix="s",
        max_gates=3,
        allow_training_busy_gates=True,
        allow_grade_busy_training=True,
        train_iterations=8,
        episodes_per_iteration=10,
        checkpoint_every=2,
        max_train_launches=4,
    )

    autopilot.plan_triage_gates(args)
    autopilot.plan_train(args, recipe="vrpo_esarsa_antireg")

    flattened = [part for command in commands for part in command]
    assert "--allow-training-busy-workers" in flattened
    assert "--allow-grade-busy-workers" in flattened


def test_plan_opening_evals_uses_no_busy_vm_side_diagnostics(monkeypatch) -> None:
    commands = []

    def fake_run(command, *, capture):
        commands.append(command)

    monkeypatch.setattr(autopilot, "run", fake_run)
    monkeypatch.setattr(autopilot, "read_json", lambda path: {"planned_count": 0})
    args = argparse.Namespace(run_prefix="s")

    assert plan_opening_evals(args) == {"planned_count": 0}

    command = commands[0]
    assert command[:4] == [
        autopilot.sys.executable,
        "tools/gcp_fleet_controller.py",
        "--run-prefix",
        "s",
    ]
    assert "plan-remote-opening-evals" in command
    assert "--include-interim" in command
    assert "--allow-busy-workers" not in command
    assert "--prefer-prefix" in command
    assert command[command.index("--prefer-prefix") + 1] == "s101"
    assert command[command.index("--max-evals") + 1] == "2"
    assert command[command.index("--output") + 1] == str(autopilot.OPENING_EVAL_PLAN_PATH)


def test_launch_planned_dedupes_same_checkpoint(monkeypatch) -> None:
    commands = []

    def fake_run(command, *, capture):
        commands.append(command)

    monkeypatch.setattr(autopilot, "run", fake_run)

    launched = launch_planned(
        {
            "planned": [
                {"worker": "w1", "checkpoint": "candidate.pt", "command": ["grade", "a"]},
                {"worker": "w2", "checkpoint": "candidate.pt", "command": ["grade", "b"]},
            ]
        },
        launched_workers=set(),
        launched_checkpoints=set(),
    )

    assert commands == [["grade", "a"]]
    assert launched == [
        {"worker": "w1", "checkpoint": "candidate.pt"},
        {
            "worker": "w2",
            "checkpoint": "candidate.pt",
            "skipped": "checkpoint_already_launched_this_cycle",
        },
    ]


def test_auto_recipe_targets_jsettlers_regression() -> None:
    args = argparse.Namespace(recipe="auto")

    selected = autopilot.select_training_recipe(
        args,
        population_summary={
            "training_recommendation": {
                "primary_failure_mode": {"mode": "opponent_regression:jsettlers_lite"}
            }
        },
    )

    assert selected == "vrpo_jsettlers_value_repair"


def test_auto_recipe_falls_back_to_weighted_repair() -> None:
    args = argparse.Namespace(recipe="auto")

    selected = autopilot.select_training_recipe(
        args,
        population_summary={
            "training_recommendation": {
                "primary_failure_mode": {"mode": "opponent_regression:value_rollout"}
            }
        },
    )

    assert selected == "weighted_dagger_antireg"
