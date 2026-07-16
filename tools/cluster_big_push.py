from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


RUN_DIR = Path("runs/self_play/cluster_big_push")
POLL_PATH = RUN_DIR / "gcp_poll.json"
LOCAL_STATUS_PATH = RUN_DIR / "local_controller_status.json"
GRADE_STATUS_PATH = RUN_DIR / "remote_grade_status.json"
GRADE_SUMMARY_PATH = RUN_DIR / "remote_grade_summary.json"
PAYOFF_LEDGER_PATH = Path("runs/self_play/population_payoffs.jsonl")
POPULATION_SUMMARY_PATH = RUN_DIR / "population_summary.json"
STRICT_GATE_PLAN_PATH = RUN_DIR / "strict_gate_plan.json"
TRANSFER_GATE_PLAN_PATH = RUN_DIR / "transfer_gate_plan.json"
CODE_SYNC_PLAN_PATH = RUN_DIR / "code_sync_plan.json"
TRAIN_PLAN_TEMPLATE = RUN_DIR / "train_plan_{recipe}.json"
LEGACY_LIVE_ACK_FLAG = "--acknowledge-legacy-gcp-big-push"

DEFAULT_CHAMPION = "runs/self_play/champions/current_best_s9752_iter0002.pt"
DEFAULT_RECIPES = (
    "strict_gate_distill_guard",
    "strict_gate_antireg",
    "vrpo_jsettlers_value_repair",
    "tactical_rollout_guard_repair",
)
STRICT_PREFER_PREFIXES = (
    "s103",
    "s102",
    "s100",
    "s99",
    "s988",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "LEGACY cluster-scale big-push orchestrator. It uses the retired GCP "
            "controller, old champion/recipe defaults, and can automatically launch "
            "remote jobs. Current fleet operations use tools/fleet."
        )
    )
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--sleep-seconds", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        LEGACY_LIVE_ACK_FLAG,
        action="store_true",
        help=(
            "Acknowledge an intentional live run through the retired GCP big-push "
            "controller. Not required for --dry-run."
        ),
    )
    parser.add_argument(
        "--no-refresh",
        action="store_true",
        help="Use existing JSON state files under runs/self_play/cluster_big_push.",
    )
    parser.add_argument("--run-prefix", default="s")
    parser.add_argument("--champion", default=DEFAULT_CHAMPION)
    parser.add_argument(
        "--recipe",
        action="append",
        default=[],
        help=(
            "Remote training recipe to include. May be repeated. Defaults to "
            "a strict-gate anti-regression portfolio."
        ),
    )
    parser.add_argument("--train-iterations", type=int, default=8)
    parser.add_argument("--episodes-per-iteration", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=2)
    parser.add_argument("--target-training-processes", type=int, default=50)
    parser.add_argument("--max-train-launches", type=int, default=4)
    parser.add_argument("--max-gates", type=int, default=6)
    parser.add_argument("--gate-games", type=int, default=4)
    parser.add_argument("--gate-min-run-number", type=int, default=9900)
    parser.add_argument(
        "--allow-training-busy-gates",
        action="store_true",
        default=True,
        help="Allow grades on workers that are training but have no active grade.",
    )
    parser.add_argument(
        "--no-training-busy-gates",
        action="store_false",
        dest="allow_training_busy_gates",
        help="Do not plan grades on workers that are currently training.",
    )
    parser.add_argument(
        "--allow-grade-busy-training",
        action="store_true",
        help="Allow training on workers with active grades.",
    )
    parser.add_argument(
        "--sync-code",
        action="store_true",
        help="Plan/launch remote code syncs for missing recipe features first.",
    )
    parser.add_argument("--max-code-syncs", type=int, default=10)
    return parser


def _refuse_unacknowledged_legacy_live_run(args: argparse.Namespace) -> None:
    """Keep the retired auto-launch controller inert unless explicitly armed."""

    if not bool(args.dry_run) and not bool(
        getattr(args, "acknowledge_legacy_gcp_big_push", False)
    ):
        raise SystemExit(
            "live cluster_big_push execution is retired and may automatically "
            "launch stale GCP training, grading, and code-sync plans. Use "
            "tools/fleet for current operations. For an intentional historical "
            f"live run only, pass {LEGACY_LIVE_ACK_FLAG} explicitly."
        )


def main() -> None:
    args = build_parser().parse_args()
    _refuse_unacknowledged_legacy_live_run(args)
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    reports = []
    for cycle in range(max(int(args.cycles), 1)):
        reports.append(run_cycle(args, cycle_index=cycle))
        if cycle + 1 < args.cycles:
            time.sleep(max(float(args.sleep_seconds), 0.0))
    print(json.dumps({"cycles": reports}, indent=2, sort_keys=True))


def run_cycle(args: argparse.Namespace, *, cycle_index: int) -> dict[str, Any]:
    _refuse_unacknowledged_legacy_live_run(args)
    if not args.no_refresh:
        refresh_state(args)
    ensure_state_files()

    payoff_update = run_json(
        [
            sys.executable,
            "tools/update_population_payoffs.py",
            "--summary",
            str(GRADE_SUMMARY_PATH),
            "--output",
            str(PAYOFF_LEDGER_PATH),
            "--run-label",
            f"cluster-big-push-cycle-{cycle_index}",
            "--dedupe-existing",
        ]
    )
    run(
        [
            sys.executable,
            "tools/summarize_population_payoffs.py",
            "--ledger",
            str(PAYOFF_LEDGER_PATH),
            "--profile",
            "strict",
            "--opponent",
            Path(args.champion).stem,
            "--top",
            "20",
            "--output",
            str(POPULATION_SUMMARY_PATH),
        ]
    )

    recipes = selected_recipes(args, read_json(POPULATION_SUMMARY_PATH))
    code_sync_plan = (
        plan_code_sync(args, recipes[0]) if args.sync_code and recipes else empty_plan()
    )
    strict_gate_plan = plan_strict_gates(args)
    transfer_gate_plan = plan_transfer_gates(args)
    train_plans = [
        plan_train(args, recipe=recipe, max_launches=per_recipe_train_launches(args, recipes))
        for recipe in recipes
    ]

    launches: list[dict[str, Any]] = []
    if not args.dry_run:
        launched_workers: set[str] = set()
        launched_checkpoints: set[str] = set()
        for plan in launch_order(
            poll=read_json(POLL_PATH),
            code_sync_plan=code_sync_plan,
            strict_gate_plan=strict_gate_plan,
            transfer_gate_plan=transfer_gate_plan,
            train_plans=train_plans,
            target_training_processes=args.target_training_processes,
        ):
            launches.extend(
                launch_plan(
                    plan,
                    launched_workers=launched_workers,
                    launched_checkpoints=launched_checkpoints,
                )
            )

    poll = read_json(POLL_PATH)
    grade_summary = read_json(GRADE_SUMMARY_PATH)
    population_summary = read_json(POPULATION_SUMMARY_PATH)
    return {
        "cycle": cycle_index + 1,
        "dry_run": bool(args.dry_run),
        "training_processes": poll.get("running_train_processes"),
        "candidate_checkpoints": poll.get("candidate_checkpoints"),
        "active_grades": grade_summary.get("active_count"),
        "grade_decisions": len(grade_summary.get("decisions", [])),
        "payoff_update": payoff_update,
        "decision_counts": population_summary.get("decision_counts"),
        "primary_failure_mode": (
            (population_summary.get("training_recommendation") or {})
            .get("primary_failure_mode")
        ),
        "recipes": recipes,
        "planned_code_syncs": code_sync_plan.get("planned_count", 0),
        "planned_strict_gates": strict_gate_plan.get("planned_count", 0),
        "planned_transfer_gates": transfer_gate_plan.get("planned_count", 0),
        "planned_train": {
            recipe: plan.get("planned_count", 0)
            for recipe, plan in zip(recipes, train_plans, strict=True)
        },
        "launches": launches,
    }


def refresh_state(args: argparse.Namespace) -> None:
    run(
        [
            sys.executable,
            "tools/gcp_fleet_controller.py",
            "--run-prefix",
            args.run_prefix,
            "poll",
            "--output",
            str(POLL_PATH),
        ]
    )
    run(
        [
            sys.executable,
            "tools/gcp_fleet_controller.py",
            "local-controller-status",
            "--output",
            str(LOCAL_STATUS_PATH),
        ]
    )
    run(
        [
            sys.executable,
            "tools/gcp_fleet_controller.py",
            "remote-grade-status",
            "--output",
            str(GRADE_STATUS_PATH),
        ]
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
        ]
    )


def ensure_state_files() -> None:
    missing = [
        path
        for path in (POLL_PATH, LOCAL_STATUS_PATH, GRADE_SUMMARY_PATH)
        if not path.exists()
    ]
    if missing:
        raise SystemExit(
            "missing state files; run without --no-refresh first: "
            + ", ".join(str(path) for path in missing)
        )


def selected_recipes(
    args: argparse.Namespace,
    population_summary: dict[str, Any],
) -> list[str]:
    if args.recipe:
        return list(dict.fromkeys(args.recipe))

    recommendation = population_summary.get("training_recommendation") or {}
    mode = str((recommendation.get("primary_failure_mode") or {}).get("mode") or "")
    if mode == "opponent_regression:value_rollout":
        return [
            "strict_gate_distill_guard",
            "strict_gate_antireg",
            "tactical_rollout_guard_repair",
            "vrpo_jsettlers_value_repair",
        ]
    if mode == "opponent_regression:jsettlers_lite":
        return list(DEFAULT_RECIPES)
    return list(DEFAULT_RECIPES)


def per_recipe_train_launches(args: argparse.Namespace, recipes: list[str]) -> int:
    if not recipes:
        return 0
    return max(1, int(args.max_train_launches) // len(recipes))


def plan_code_sync(args: argparse.Namespace, recipe: str) -> dict[str, Any]:
    run(
        [
            sys.executable,
            "tools/gcp_fleet_controller.py",
            "plan-remote-code-sync",
            "--poll",
            str(POLL_PATH),
            "--summary",
            str(GRADE_SUMMARY_PATH),
            "--local-status",
            str(LOCAL_STATUS_PATH),
            "--recipe",
            recipe,
            "--file",
            "tools/gcp_fleet_controller.py",
            "--file",
            "tools/train_ppo.py",
            "--max-syncs",
            str(args.max_code_syncs),
            "--allow-grade-busy-workers",
            "--output",
            str(CODE_SYNC_PLAN_PATH),
        ]
    )
    return read_json(CODE_SYNC_PLAN_PATH)


def plan_strict_gates(args: argparse.Namespace) -> dict[str, Any]:
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
        "--champion",
        args.champion,
        "--profile",
        "strict",
        "--games",
        str(args.gate_games),
        "--max-gates",
        str(args.max_gates),
        "--include-interim",
        "--allow-rejected-family-continuation",
        "--min-run-number",
        str(args.gate_min_run_number),
        "--output",
        str(STRICT_GATE_PLAN_PATH),
    ]
    for prefix in STRICT_PREFER_PREFIXES:
        command.extend(["--prefer-prefix", prefix])
    if args.allow_training_busy_gates:
        command.append("--allow-training-busy-workers")
    run(command)
    return read_json(STRICT_GATE_PLAN_PATH)


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
        "--champion",
        args.champion,
        "--profile",
        "strict",
        "--games",
        str(args.gate_games),
        "--max-gates",
        str(args.max_gates),
        "--include-interim",
        "--allow-rejected-family-continuation",
        "--min-run-number",
        str(args.gate_min_run_number),
        "--output",
        str(TRANSFER_GATE_PLAN_PATH),
    ]
    for prefix in STRICT_PREFER_PREFIXES:
        command.extend(["--prefer-prefix", prefix])
    if args.allow_training_busy_gates:
        command.append("--allow-training-busy-target-workers")
    run(command)
    return read_json(TRANSFER_GATE_PLAN_PATH)


def plan_train(
    args: argparse.Namespace,
    *,
    recipe: str,
    max_launches: int,
) -> dict[str, Any]:
    path = Path(str(TRAIN_PLAN_TEMPLATE).format(recipe=recipe))
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
        "--champion",
        args.champion,
        "--recipe",
        recipe,
        "--iterations",
        str(args.train_iterations),
        "--episodes-per-iteration",
        str(args.episodes_per_iteration),
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--max-launches",
        str(max_launches),
        "--output",
        str(path),
    ]
    if args.allow_grade_busy_training:
        command.append("--allow-grade-busy-workers")
    run(command)
    return read_json(path)


def launch_order(
    *,
    poll: dict[str, Any],
    code_sync_plan: dict[str, Any],
    strict_gate_plan: dict[str, Any],
    transfer_gate_plan: dict[str, Any],
    train_plans: list[dict[str, Any]],
    target_training_processes: int,
) -> list[dict[str, Any]]:
    training_processes = int(poll.get("running_train_processes") or 0)
    gate_plans = [strict_gate_plan, transfer_gate_plan]
    if training_processes < int(target_training_processes):
        return [code_sync_plan, *train_plans, *gate_plans]
    return [code_sync_plan, *gate_plans, *train_plans]


def launch_plan(
    plan: dict[str, Any],
    *,
    launched_workers: set[str],
    launched_checkpoints: set[str],
) -> list[dict[str, Any]]:
    launched = []
    for row in plan.get("planned", []):
        command = row.get("command")
        if not command:
            continue
        worker = str(row.get("worker") or row.get("target_worker") or "")
        checkpoint = str(row.get("checkpoint") or row.get("checkpoint_path") or "")
        if worker and worker in launched_workers:
            launched.append(
                {"worker": worker, "checkpoint": checkpoint, "skipped": "worker_busy_this_cycle"}
            )
            continue
        if checkpoint and checkpoint in launched_checkpoints:
            launched.append(
                {
                    "worker": worker,
                    "checkpoint": checkpoint,
                    "skipped": "checkpoint_launched_this_cycle",
                }
            )
            continue
        run([str(part) for part in command])
        if worker:
            launched_workers.add(worker)
        if checkpoint:
            launched_checkpoints.add(checkpoint)
        launched.append({"worker": worker, "checkpoint": checkpoint})
    return launched


def empty_plan() -> dict[str, Any]:
    return {"planned_count": 0, "planned": []}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def run_json(command: list[str]) -> dict[str, Any]:
    result = run(command)
    try:
        return json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return {"stdout": result.stdout}


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


if __name__ == "__main__":
    main()
