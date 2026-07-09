from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


POLL_PATH = Path("runs/self_play/gcp_poll_latest_s.json")
LOCAL_STATUS_PATH = Path("runs/self_play/local_controller_status_latest.json")
GRADE_STATUS_PATH = Path("runs/self_play/remote_grade_status_latest.json")
GRADE_SUMMARY_PATH = Path("runs/self_play/remote_grade_summary_latest.json")
PAYOFF_LEDGER_PATH = Path("runs/self_play/population_payoffs.jsonl")
POPULATION_SUMMARY_PATH = Path("runs/self_play/population_summary_latest.json")
ESCALATION_PLAN_PATH = Path("runs/self_play/remote_escalation_plan_latest.json")
TRIAGE_GATE_PLAN_PATH = Path("runs/self_play/remote_triage_gate_plan_latest.json")
GATE_PLAN_PATH = Path("runs/self_play/remote_gate_plan_latest.json")
TRANSFER_GATE_PLAN_PATH = Path("runs/self_play/remote_transfer_gate_plan_latest.json")
OPENING_EVAL_PLAN_PATH = Path("runs/self_play/remote_opening_eval_plan_latest.json")
TRAIN_PLAN_PATH = Path("runs/self_play/remote_train_plan_weighted_dagger.json")
GATE_MIN_RUN_NUMBER = 9900
TRIAGE_PREFER_PREFIXES = ("s100", "s99", "s991", "s990", "s989", "s988")
STRICT_PREFER_PREFIXES = (
    "s100",
    "s99",
    "s988",
    "s9875_blend_s9861_iter4",
    "s9874_weighted_dagger",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run controller-side fleet maintenance cycles. This only launches "
            "remote GCP training/grading jobs; it never runs local Catan RL."
        )
    )
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--sleep-seconds", type=int, default=300)
    parser.add_argument("--run-prefix", default="s")
    parser.add_argument(
        "--recipe",
        default="auto",
        help=(
            "Training recipe for newly freed workers. Use auto to pick a "
            "repair recipe from the latest strict-gate failure mode."
        ),
    )
    parser.add_argument("--train-iterations", type=int, default=8)
    parser.add_argument("--episodes-per-iteration", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=2)
    parser.add_argument("--target-training-processes", type=int, default=10)
    parser.add_argument("--max-train-launches", type=int, default=10)
    parser.add_argument("--max-gates", type=int, default=10)
    parser.add_argument("--max-escalations", type=int, default=2)
    parser.add_argument(
        "--allow-training-busy-gates",
        action="store_true",
        default=False,
        help="Permit gate planning on VMs that are currently training.",
    )
    parser.add_argument(
        "--no-training-busy-gates",
        action="store_false",
        dest="allow_training_busy_gates",
        help="Do not plan remote grades on VMs that are currently training.",
    )
    parser.add_argument(
        "--allow-grade-busy-training",
        action="store_true",
        default=False,
        help="Permit train planning on VMs that are currently running grades.",
    )
    parser.add_argument(
        "--no-grade-busy-training",
        action="store_false",
        dest="allow_grade_busy_training",
        help="Do not plan remote training on VMs that are currently running grades.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    reports = []
    for cycle in range(args.cycles):
        reports.append(run_cycle(args))
        if cycle + 1 < args.cycles:
            time.sleep(args.sleep_seconds)
    print(json.dumps({"cycles": reports}, indent=2, sort_keys=True))


def run_cycle(args: argparse.Namespace) -> dict[str, Any]:
    run(
        [
            sys.executable,
            "tools/gcp_fleet_controller.py",
            "--run-prefix",
            args.run_prefix,
            "poll",
            "--output",
            str(POLL_PATH),
        ],
        capture=True,
    )
    run(
        [
            sys.executable,
            "tools/gcp_fleet_controller.py",
            "local-controller-status",
            "--output",
            str(LOCAL_STATUS_PATH),
        ],
        capture=True,
    )
    run(
        [
            sys.executable,
            "tools/gcp_fleet_controller.py",
            "remote-grade-status",
            "--output",
            str(GRADE_STATUS_PATH),
        ],
        capture=True,
    )
    run(
        [
            sys.executable,
            "tools/gcp_fleet_controller.py",
            "remote-grade-summary",
            "--input",
            str(GRADE_STATUS_PATH),
            "--output",
            str(GRADE_SUMMARY_PATH),
        ],
        capture=True,
    )
    payoff_update = run(
        [
            sys.executable,
            "tools/update_population_payoffs.py",
            "--summary",
            str(GRADE_SUMMARY_PATH),
            "--output",
            str(PAYOFF_LEDGER_PATH),
            "--run-label",
            "remote-strict-2026-06-27",
            "--dedupe-existing",
        ],
        capture=True,
    )
    run(
        [
            sys.executable,
            "tools/summarize_population_payoffs.py",
            "--ledger",
            str(PAYOFF_LEDGER_PATH),
            "--output",
            str(POPULATION_SUMMARY_PATH),
            "--top",
            "10",
        ],
        capture=True,
    )
    escalation_plan = plan_escalations(args)
    triage_gate_plan = plan_triage_gates(args)
    gate_plan = plan_gates(args)
    transfer_gate_plan = plan_transfer_gates(args)
    opening_eval_plan = plan_opening_evals(args)
    selected_recipe = select_training_recipe(args, population_summary=read_json(POPULATION_SUMMARY_PATH))
    train_plan = plan_train(args, recipe=selected_recipe)

    launches: list[dict[str, Any]] = []
    launched_workers: set[str] = set()
    launched_checkpoints: set[str] = set()
    if not args.dry_run:
        for plan in ordered_launch_plans(
            training_processes=int(read_json(POLL_PATH).get("running_train_processes") or 0),
            target_training_processes=args.target_training_processes,
            train_plan=train_plan,
            escalation_plan=escalation_plan,
            triage_gate_plan=triage_gate_plan,
            gate_plan=gate_plan,
            transfer_gate_plan=transfer_gate_plan,
            opening_eval_plan=opening_eval_plan,
        ):
            launches.extend(
                launch_planned(
                    plan,
                    launched_workers=launched_workers,
                    launched_checkpoints=launched_checkpoints,
                )
            )

    poll = read_json(POLL_PATH)
    grade_summary = read_json(GRADE_SUMMARY_PATH)
    population_summary = read_json(POPULATION_SUMMARY_PATH)
    return {
        "training_processes": poll.get("running_train_processes"),
        "active_grades": grade_summary.get("active_count"),
        "decisions": len(grade_summary.get("decisions", [])),
        "keepers": len(grade_summary.get("keepers", [])),
        "rejections": len(grade_summary.get("rejections", [])),
        "payoff_update": json.loads(payoff_update.stdout or "{}"),
        "training_recommendation": population_summary.get("training_recommendation"),
        "selected_training_recipe": selected_recipe,
        "planned_escalations": escalation_plan.get("planned_count", 0),
        "planned_triage_gates": triage_gate_plan.get("planned_count", 0),
        "planned_gates": gate_plan.get("planned_count", 0),
        "planned_transfer_gates": transfer_gate_plan.get("planned_count", 0),
        "planned_opening_evals": opening_eval_plan.get("planned_count", 0),
        "planned_train": train_plan.get("planned_count", 0),
        "launched": launches,
        "dry_run": args.dry_run,
    }


def plan_escalations(args: argparse.Namespace) -> dict[str, Any]:
    run(
        [
            sys.executable,
            "tools/gcp_fleet_controller.py",
            "plan-remote-escalations",
            "--summary",
            str(GRADE_SUMMARY_PATH),
            "--local-status",
            str(LOCAL_STATUS_PATH),
            "--source-games",
            "4",
            "--target-games",
            "12",
            "--max-escalations",
            str(args.max_escalations),
            "--output",
            str(ESCALATION_PLAN_PATH),
        ],
        capture=True,
    )
    return read_json(ESCALATION_PLAN_PATH)


def plan_triage_gates(args: argparse.Namespace) -> dict[str, Any]:
    command = [
        sys.executable,
        "tools/gcp_fleet_controller.py",
        "--run-prefix",
        args.run_prefix,
        "plan-remote-gates",
        "--poll",
        str(POLL_PATH),
        "--summary",
        str(GRADE_SUMMARY_PATH),
        "--local-status",
        str(LOCAL_STATUS_PATH),
        "--profile",
        "jsettlers_triage",
        "--games",
        "2",
        "--workers",
        "4",
        "--max-decisions",
        "220",
        "--leg-timeout-seconds",
        "600",
        "--max-gates",
        str(args.max_gates),
        "--include-interim",
        *prefer_prefix_args(TRIAGE_PREFER_PREFIXES),
        "--min-run-number",
        str(GATE_MIN_RUN_NUMBER),
        "--output",
        str(TRIAGE_GATE_PLAN_PATH),
    ]
    if args.allow_training_busy_gates:
        command.append("--allow-training-busy-workers")
    run(command, capture=True)
    return read_json(TRIAGE_GATE_PLAN_PATH)


def plan_gates(args: argparse.Namespace) -> dict[str, Any]:
    command = [
        sys.executable,
        "tools/gcp_fleet_controller.py",
        "--run-prefix",
        args.run_prefix,
        "plan-remote-gates",
        "--poll",
        str(POLL_PATH),
        "--summary",
        str(GRADE_SUMMARY_PATH),
        "--local-status",
        str(LOCAL_STATUS_PATH),
        "--profile",
        "strict",
        "--games",
        "4",
        "--max-gates",
        str(args.max_gates),
        "--include-interim",
        *prefer_prefix_args(STRICT_PREFER_PREFIXES),
        "--min-run-number",
        str(GATE_MIN_RUN_NUMBER),
        "--output",
        str(GATE_PLAN_PATH),
    ]
    if args.allow_training_busy_gates:
        command.append("--allow-training-busy-workers")
    run(command, capture=True)
    return read_json(GATE_PLAN_PATH)


def plan_transfer_gates(args: argparse.Namespace) -> dict[str, Any]:
    command = [
        sys.executable,
        "tools/gcp_fleet_controller.py",
        "--run-prefix",
        args.run_prefix,
        "plan-remote-transfer-gates",
        "--poll",
        str(POLL_PATH),
        "--summary",
        str(GRADE_SUMMARY_PATH),
        "--local-status",
        str(LOCAL_STATUS_PATH),
        "--profile",
        "strict",
        "--games",
        "4",
        "--max-gates",
        str(args.max_gates),
        "--include-interim",
        *prefer_prefix_args(STRICT_PREFER_PREFIXES),
        "--min-run-number",
        str(GATE_MIN_RUN_NUMBER),
        "--output",
        str(TRANSFER_GATE_PLAN_PATH),
    ]
    run(command, capture=True)
    return read_json(TRANSFER_GATE_PLAN_PATH)


def plan_opening_evals(args: argparse.Namespace) -> dict[str, Any]:
    command = [
        sys.executable,
        "tools/gcp_fleet_controller.py",
        "--run-prefix",
        args.run_prefix,
        "plan-remote-opening-evals",
        "--poll",
        str(POLL_PATH),
        "--summary",
        str(GRADE_SUMMARY_PATH),
        "--local-status",
        str(LOCAL_STATUS_PATH),
        "--include-interim",
        *prefer_prefix_args(("s101", "s100")),
        "--max-evals",
        "2",
        "--output",
        str(OPENING_EVAL_PLAN_PATH),
    ]
    run(command, capture=True)
    return read_json(OPENING_EVAL_PLAN_PATH)


def select_training_recipe(
    args: argparse.Namespace,
    *,
    population_summary: dict[str, Any],
) -> str:
    if args.recipe != "auto":
        return str(args.recipe)
    recommendation = population_summary.get("training_recommendation") or {}
    failure_mode = recommendation.get("primary_failure_mode") or {}
    mode = str(failure_mode.get("mode") or "")
    if mode == "opponent_regression:jsettlers_lite":
        return "vrpo_jsettlers_value_repair"
    return "weighted_dagger_antireg"


def plan_train(args: argparse.Namespace, *, recipe: str) -> dict[str, Any]:
    command = [
        sys.executable,
        "tools/gcp_fleet_controller.py",
        "--run-prefix",
        args.run_prefix,
        "plan-remote-train",
        "--poll",
        str(POLL_PATH),
        "--summary",
        str(GRADE_SUMMARY_PATH),
        "--local-status",
        str(LOCAL_STATUS_PATH),
        "--recipe",
        recipe,
        "--iterations",
        str(args.train_iterations),
        "--episodes-per-iteration",
        str(args.episodes_per_iteration),
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--max-launches",
        str(args.max_train_launches),
        "--output",
        str(TRAIN_PLAN_PATH),
    ]
    if args.allow_grade_busy_training:
        command.append("--allow-grade-busy-workers")
    run(command, capture=True)
    return read_json(TRAIN_PLAN_PATH)


def prefer_prefix_args(prefixes: tuple[str, ...]) -> list[str]:
    args: list[str] = []
    for prefix in prefixes:
        args.extend(["--prefer-prefix", prefix])
    return args


def ordered_launch_plans(
    *,
    training_processes: int,
    target_training_processes: int,
    train_plan: dict[str, Any],
    escalation_plan: dict[str, Any],
    triage_gate_plan: dict[str, Any],
    gate_plan: dict[str, Any],
    transfer_gate_plan: dict[str, Any],
    opening_eval_plan: dict[str, Any],
) -> list[dict[str, Any]]:
    gate_plans = [
        escalation_plan,
        gate_plan,
        transfer_gate_plan,
        opening_eval_plan,
        triage_gate_plan,
    ]
    if training_processes < target_training_processes:
        return [train_plan, *gate_plans]
    return [*gate_plans, train_plan]


def launch_planned(
    plan: dict[str, Any],
    *,
    launched_workers: set[str],
    launched_checkpoints: set[str] | None = None,
) -> list[dict[str, Any]]:
    launched = []
    launched_checkpoints = launched_checkpoints if launched_checkpoints is not None else set()
    for row in plan.get("planned", []):
        command = row.get("command")
        if not command:
            continue
        worker = str(row.get("worker") or row.get("target_worker") or "")
        checkpoint = str(row.get("checkpoint") or row.get("checkpoint_path") or "")
        if checkpoint and checkpoint in launched_checkpoints:
            launched.append(
                {
                    "worker": worker,
                    "checkpoint": checkpoint,
                    "skipped": "checkpoint_already_launched_this_cycle",
                }
            )
            continue
        if worker and worker in launched_workers:
            launched.append(
                {
                    "worker": worker,
                    "checkpoint": checkpoint,
                    "skipped": "worker_already_launched_this_cycle",
                }
            )
            continue
        run([str(part) for part in command], capture=True)
        if worker:
            launched_workers.add(worker)
        if checkpoint:
            launched_checkpoints.add(checkpoint)
        launched.append(
            {
                "worker": worker,
                "checkpoint": checkpoint,
            }
        )
    return launched


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def run(command: list[str], *, capture: bool) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


if __name__ == "__main__":
    main()
