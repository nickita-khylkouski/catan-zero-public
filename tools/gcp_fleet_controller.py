from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_PROJECT = "peak-security-disclosure-ops"
DEFAULT_CHAMPION = "runs/self_play/champions/current_best_s9752_iter0002.pt"
DEFAULT_RUN_PREFIX = "s"
DEFAULT_WORKERS = (
    "catan-zero-c1:us-central1-c",
    "catan-zero-c2:us-central1-c",
    "catan-zero-c3:us-central1-c",
    "catan-zero-c4:us-central1-c",
    "catan-zero-w1a:us-west1-b",
    "catan-zero-w1b:us-west1-b",
    "catan-zero-w4a:us-west4-a",
    "catan-zero-w4b:us-west4-a",
    "catan-zero-w4c:us-west4-b",
    "catan-zero-w4d:us-west4-b",
)
DEFAULT_WORKER_NAMES = frozenset(worker.split(":", 1)[0] for worker in DEFAULT_WORKERS)
DEFAULT_WORKER_INDEX = {
    worker.split(":", 1)[0]: index for index, worker in enumerate(DEFAULT_WORKERS)
}
DEFAULT_SYNC_FILES = (
    "tools/train_ppo.py",
    "tools/gcp_fleet_controller.py",
    "tools/grade_agent.py",
    "tools/generate_reanalysis.py",
    "tools/evaluate_self_play.py",
    "tools/evaluate_openings.py",
    "tools/update_population_payoffs.py",
    "tools/summarize_population_payoffs.py",
    "src/catan_zero/rl/self_play.py",
    "src/catan_zero/rl/reanalysis.py",
    "src/catan_zero/rl/torch_ppo.py",
)


@dataclass(frozen=True, slots=True)
class Worker:
    name: str
    zone: str


@dataclass(frozen=True, slots=True)
class GateCandidate:
    worker: Worker
    checkpoint: str
    family: str
    iteration: int
    size: int


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Poll, pull, and grade active GCP CatanZero training workers."
    )
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--worker", action="append", default=[])
    parser.add_argument("--remote-repo", default="")
    parser.add_argument("--run-prefix", default=DEFAULT_RUN_PREFIX)
    subparsers = parser.add_subparsers(dest="command", required=True)

    poll_parser = subparsers.add_parser("poll")
    poll_parser.add_argument("--output")

    pull_parser = subparsers.add_parser("pull-ready")
    pull_parser.add_argument("--output-dir", default="runs/self_play/gcp_imports")
    pull_parser.add_argument("--include-interim", action="store_true")
    pull_parser.add_argument(
        "--prefer-prefix",
        action="append",
        default=[],
        help="Only pull artifact stems beginning with this checkpoint prefix. May be repeated.",
    )
    pull_parser.add_argument(
        "--min-run-number",
        type=int,
        default=0,
        help=(
            "Ignore checkpoints whose run number after --run-prefix is lower. "
            "For run-prefix s97, both 82 and 9782 select s9782+."
        ),
    )
    pull_parser.add_argument(
        "--max-artifacts",
        type=int,
        default=0,
        help="Maximum artifact files to pull after filtering; 0 means no limit.",
    )
    pull_parser.add_argument("--dry-run", action="store_true")

    grade_parser = subparsers.add_parser("grade-ready")
    grade_parser.add_argument("--input-dir", default="runs/self_play/gcp_imports")
    grade_parser.add_argument("--champion", default=DEFAULT_CHAMPION)
    grade_parser.add_argument("--eval-dir", default="runs/self_play/gcp_grades")
    grade_parser.add_argument(
        "--profile",
        choices=("dev", "jsettlers_triage", "strict", "search_stress"),
        default="dev",
        help="Forwarded to tools/grade_agent.py.",
    )
    grade_parser.add_argument("--games", type=int, default=16)
    grade_parser.add_argument("--repeats", type=int, default=2)
    grade_parser.add_argument("--workers", type=int, default=4)
    grade_parser.add_argument("--leg-timeout-seconds", type=int, default=0)
    grade_parser.add_argument(
        "--opponent",
        action="append",
        choices=("random", "heuristic", "jsettlers_lite", "search", "value_rollout", "value"),
        help="Opponent suite forwarded to tools/grade_agent.py.",
    )
    grade_parser.add_argument(
        "--opponent-weight",
        action="append",
        default=[],
        help="Forwarded as opponent=weight to tools/grade_agent.py.",
    )
    grade_parser.add_argument(
        "--opponent-candidate-limit",
        type=int,
        help="Forwarded to tools/grade_agent.py.",
    )
    grade_parser.add_argument(
        "--opponent-rollout-decisions",
        type=int,
        help="Forwarded to tools/grade_agent.py.",
    )
    grade_parser.add_argument(
        "--opponent-value-penalty",
        type=float,
        help="Forwarded to tools/grade_agent.py.",
    )
    grade_parser.add_argument("--vps-to-win", type=int, default=4)
    grade_parser.add_argument("--max-decisions", type=int, default=300)
    grade_parser.add_argument("--max-checkpoints", type=int, default=8)
    grade_parser.add_argument("--include-interim", action="store_true")
    grade_parser.add_argument(
        "--all-snapshots",
        action="store_true",
        help="Grade every local snapshot instead of only the latest checkpoint per run lane.",
    )
    grade_parser.add_argument("--dry-run", action="store_true")

    remote_grade_parser = subparsers.add_parser("remote-grade")
    remote_grade_parser.add_argument("--checkpoint", required=True)
    remote_grade_parser.add_argument("--champion", default=DEFAULT_CHAMPION)
    remote_grade_parser.add_argument("--eval-dir", default="runs/self_play/remote_grades_reanalysis")
    remote_grade_parser.add_argument("--log-dir", default="runs/self_play/logs")
    remote_grade_parser.add_argument(
        "--profile",
        choices=("dev", "jsettlers_triage", "strict", "search_stress"),
        default="dev",
    )
    remote_grade_parser.add_argument("--games", type=int, default=12)
    remote_grade_parser.add_argument("--repeats", type=int, default=1)
    remote_grade_parser.add_argument("--workers", type=int, default=6)
    remote_grade_parser.add_argument("--vps-to-win", type=int, default=4)
    remote_grade_parser.add_argument("--max-decisions", type=int, default=300)
    remote_grade_parser.add_argument("--leg-timeout-seconds", type=int, default=0)
    remote_grade_parser.add_argument(
        "--force",
        action="store_true",
        help="Launch even if the same config-keyed remote grade is active or complete.",
    )
    remote_grade_parser.add_argument("--dry-run", action="store_true")

    remote_grade_from_worker_parser = subparsers.add_parser("remote-grade-from-worker")
    remote_grade_from_worker_parser.add_argument(
        "--source-worker",
        required=True,
        help="Source worker containing --checkpoint, formatted name:zone.",
    )
    remote_grade_from_worker_parser.add_argument(
        "--source-remote-repo",
        default="",
        help=(
            "Repo root on --source-worker. Defaults to --remote-repo, then "
            "/home/nickita/catan-zero."
        ),
    )
    remote_grade_from_worker_parser.add_argument(
        "--target-remote-repo",
        default="",
        help=(
            "Repo root on the target --worker. Defaults to --remote-repo, then "
            "/home/nickita/catan-zero."
        ),
    )
    remote_grade_from_worker_parser.add_argument("--checkpoint", required=True)
    remote_grade_from_worker_parser.add_argument("--champion", default=DEFAULT_CHAMPION)
    remote_grade_from_worker_parser.add_argument("--eval-dir", default="runs/self_play/remote_grades_reanalysis")
    remote_grade_from_worker_parser.add_argument("--log-dir", default="runs/self_play/logs")
    remote_grade_from_worker_parser.add_argument(
        "--profile",
        choices=("dev", "jsettlers_triage", "strict", "search_stress"),
        default="dev",
    )
    remote_grade_from_worker_parser.add_argument("--games", type=int, default=12)
    remote_grade_from_worker_parser.add_argument("--repeats", type=int, default=1)
    remote_grade_from_worker_parser.add_argument("--workers", type=int, default=6)
    remote_grade_from_worker_parser.add_argument("--vps-to-win", type=int, default=4)
    remote_grade_from_worker_parser.add_argument("--max-decisions", type=int, default=300)
    remote_grade_from_worker_parser.add_argument("--leg-timeout-seconds", type=int, default=0)
    remote_grade_from_worker_parser.add_argument(
        "--force",
        action="store_true",
        help="Forwarded to remote-grade after copying the checkpoint.",
    )
    remote_grade_from_worker_parser.add_argument("--dry-run", action="store_true")

    remote_opening_eval_parser = subparsers.add_parser("remote-opening-eval")
    remote_opening_eval_parser.add_argument("--checkpoint", required=True)
    remote_opening_eval_parser.add_argument("--output-dir", default="runs/self_play/remote_opening_evals")
    remote_opening_eval_parser.add_argument("--log-dir", default="runs/self_play/logs")
    remote_opening_eval_parser.add_argument("--games", type=int, default=16)
    remote_opening_eval_parser.add_argument("--seed", type=int, default=93000)
    remote_opening_eval_parser.add_argument("--vps-to-win", type=int, default=10)
    remote_opening_eval_parser.add_argument("--max-opening-decisions", type=int, default=16)
    remote_opening_eval_parser.add_argument("--candidate-limit", type=int, default=96)
    remote_opening_eval_parser.add_argument("--presearch-candidate-limit", type=int, default=96)
    remote_opening_eval_parser.add_argument("--rollout-decisions", type=int, default=2)
    remote_opening_eval_parser.add_argument("--rollout-samples", type=int, default=1)
    remote_opening_eval_parser.add_argument("--root-value-weight", type=float, default=0.35)
    remote_opening_eval_parser.add_argument("--opponent-penalty", type=float, default=0.05)
    remote_opening_eval_parser.add_argument(
        "--force",
        action="store_true",
        help="Launch even if the same opening-eval output exists or is active.",
    )
    remote_opening_eval_parser.add_argument("--dry-run", action="store_true")

    remote_train_parser = subparsers.add_parser("remote-train")
    remote_train_parser.add_argument("--label", required=True)
    remote_train_parser.add_argument("--log-dir", default="runs/self_play/logs")
    remote_train_parser.add_argument(
        "--force",
        action="store_true",
        help="Launch even if another train_ppo.py process is already active on the VM.",
    )
    remote_train_parser.add_argument("--dry-run", action="store_true")
    remote_train_parser.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to tools/train_ppo.py after a literal --.",
    )

    remote_reanalysis_train_parser = subparsers.add_parser("remote-reanalysis-train")
    remote_reanalysis_train_parser.add_argument("--label", required=True)
    remote_reanalysis_train_parser.add_argument("--champion", default=DEFAULT_CHAMPION)
    remote_reanalysis_train_parser.add_argument("--log-dir", default="runs/self_play/logs")
    remote_reanalysis_train_parser.add_argument("--seed", type=int, required=True)
    remote_reanalysis_train_parser.add_argument("--games", type=int, default=8)
    remote_reanalysis_train_parser.add_argument("--vps-to-win", type=int, default=4)
    remote_reanalysis_train_parser.add_argument("--max-decisions", type=int, default=300)
    remote_reanalysis_train_parser.add_argument("--record-after-decisions", type=int, default=40)
    remote_reanalysis_train_parser.add_argument("--record-window-decisions", type=int, default=120)
    remote_reanalysis_train_parser.add_argument("--candidate-limit", type=int, default=24)
    remote_reanalysis_train_parser.add_argument("--presearch-candidate-limit", type=int, default=48)
    remote_reanalysis_train_parser.add_argument("--rollout-decisions", type=int, default=2)
    remote_reanalysis_train_parser.add_argument("--rollout-samples", type=int, default=1)
    remote_reanalysis_train_parser.add_argument("--root-value-weight", type=float, default=0.25)
    remote_reanalysis_train_parser.add_argument("--temperature", type=float, default=0.55)
    remote_reanalysis_train_parser.add_argument("--reanalysis-max-samples", type=int, default=2048)
    remote_reanalysis_train_parser.add_argument("--reanalysis-epochs", type=int, default=2)
    remote_reanalysis_train_parser.add_argument("--reanalysis-value-coef", type=float, default=0.35)
    remote_reanalysis_train_parser.add_argument("--reanalysis-score-coef", type=float, default=0.05)
    remote_reanalysis_train_parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Launch even if another train_ppo.py or generate_reanalysis.py process "
            "is already active on the VM."
        ),
    )
    remote_reanalysis_train_parser.add_argument("--dry-run", action="store_true")

    remote_sync_code_parser = subparsers.add_parser("remote-sync-code")
    remote_sync_code_parser.add_argument(
        "--file",
        action="append",
        default=[],
        help="Local repo-relative source file to back up and copy to the VM.",
    )
    remote_sync_code_parser.add_argument(
        "--backup-dir",
        default="runs/self_play/code_backups",
        help="Remote repo-relative directory for timestamped backups.",
    )
    remote_sync_code_parser.add_argument(
        "--allow-busy",
        action="store_true",
        help="Allow syncing even if train_ppo.py or grade_agent.py is active.",
    )
    remote_sync_code_parser.add_argument("--dry-run", action="store_true")

    remote_stop_train_parser = subparsers.add_parser("remote-stop-train")
    remote_stop_train_parser.add_argument(
        "--match",
        required=True,
        help="Substring that must appear in the train_ppo.py command line.",
    )
    remote_stop_train_parser.add_argument("--dry-run", action="store_true")

    remote_stop_grade_parser = subparsers.add_parser("remote-stop-grade")
    remote_stop_grade_parser.add_argument(
        "--match",
        required=True,
        help="Substring that must appear in the grade_agent.py command line.",
    )
    remote_stop_grade_parser.add_argument("--dry-run", action="store_true")

    local_status_parser = subparsers.add_parser("local-controller-status")
    local_status_parser.add_argument(
        "--output",
        help=(
            "Optional JSON file to write. Reports local gcp_fleet_controller.py "
            "processes so multiple agents can avoid claiming the same worker."
        ),
    )

    plan_remote_train_parser = subparsers.add_parser("plan-remote-train")
    plan_remote_train_parser.add_argument(
        "--poll",
        default="runs/self_play/gcp_poll_latest.json",
        help="Local JSON file written by poll.",
    )
    plan_remote_train_parser.add_argument(
        "--summary",
        default="runs/self_play/remote_grade_summary_latest.json",
        help="Local compact JSON file written by remote-grade-summary.",
    )
    plan_remote_train_parser.add_argument(
        "--local-status",
        default="",
        help=(
            "Optional JSON file written by local-controller-status. Workers "
            "claimed by local remote-train/remote-grade commands are treated as busy."
        ),
    )
    plan_remote_train_parser.add_argument("--champion", default=DEFAULT_CHAMPION)
    plan_remote_train_parser.add_argument(
        "--recipe",
        choices=(
            "warmup_baseline",
            "warmup_jsettlers",
            "warmup_rollout",
            "pfsp_value_jsettlers",
            "pfsp_rollout_teacher",
            "pfsp_q_calibration",
            "pfsp_klent_control",
            "strict_repair_kl",
            "resource_plan_score_repair",
            "rollout_guard_score_repair",
            "tactical_rollout_guard_repair",
            "weighted_dagger_antireg",
            "jsettlers_dagger_antireg",
            "ema_jsettlers_antireg",
            "ema_mixed_antireg",
            "vrpo_esarsa_antireg",
            "vrpo_jsettlers_value_repair",
            "strict_gate_antireg",
            "strict_gate_distill_guard",
        ),
        default="warmup_baseline",
    )
    plan_remote_train_parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Explicit first seed. Defaults to the next seed visible in poll/summary.",
    )
    plan_remote_train_parser.add_argument(
        "--min-seed",
        type=int,
        default=9800,
        help="Lowest automatically proposed seed.",
    )
    plan_remote_train_parser.add_argument("--iterations", type=int, default=10)
    plan_remote_train_parser.add_argument("--episodes-per-iteration", type=int, default=8)
    plan_remote_train_parser.add_argument("--checkpoint-every", type=int, default=2)
    plan_remote_train_parser.add_argument("--max-launches", type=int, default=1)
    plan_remote_train_parser.add_argument(
        "--allow-unknown-remote-features",
        action="store_true",
        help=(
            "Plan even if the poll file lacks remote trainer feature flags. "
            "Use only after a same-minute manual remote feature preflight."
        ),
    )
    plan_remote_train_parser.add_argument(
        "--allow-partial-poll",
        action="store_true",
        help=(
            "Allow automatic seed selection from a poll that does not contain "
            "all default workers. Prefer an explicit --seed for one-worker preflights."
        ),
    )
    plan_remote_train_parser.add_argument(
        "--allow-grade-busy-workers",
        action="store_true",
        help=(
            "Allow planning training on workers that are running remote grades "
            "but have no active train_ppo.py process."
        ),
    )
    plan_remote_train_parser.add_argument("--output")

    plan_remote_reanalysis_train_parser = subparsers.add_parser("plan-remote-reanalysis-train")
    plan_remote_reanalysis_train_parser.add_argument(
        "--poll",
        default="runs/self_play/gcp_poll_latest.json",
        help="Local JSON file written by poll.",
    )
    plan_remote_reanalysis_train_parser.add_argument(
        "--summary",
        default="runs/self_play/remote_grade_summary_latest.json",
        help="Local compact JSON file written by remote-grade-summary.",
    )
    plan_remote_reanalysis_train_parser.add_argument(
        "--local-status",
        default="",
        help=(
            "Optional JSON file written by local-controller-status. Workers "
            "claimed by local controller commands are treated as busy."
        ),
    )
    plan_remote_reanalysis_train_parser.add_argument("--champion", default=DEFAULT_CHAMPION)
    plan_remote_reanalysis_train_parser.add_argument("--seed", type=int, default=0)
    plan_remote_reanalysis_train_parser.add_argument("--min-seed", type=int, default=10100)
    plan_remote_reanalysis_train_parser.add_argument("--max-launches", type=int, default=1)
    plan_remote_reanalysis_train_parser.add_argument("--games", type=int, default=8)
    plan_remote_reanalysis_train_parser.add_argument("--vps-to-win", type=int, default=4)
    plan_remote_reanalysis_train_parser.add_argument("--max-decisions", type=int, default=300)
    plan_remote_reanalysis_train_parser.add_argument("--record-after-decisions", type=int, default=40)
    plan_remote_reanalysis_train_parser.add_argument("--record-window-decisions", type=int, default=120)
    plan_remote_reanalysis_train_parser.add_argument("--candidate-limit", type=int, default=24)
    plan_remote_reanalysis_train_parser.add_argument("--presearch-candidate-limit", type=int, default=48)
    plan_remote_reanalysis_train_parser.add_argument("--rollout-decisions", type=int, default=2)
    plan_remote_reanalysis_train_parser.add_argument("--rollout-samples", type=int, default=1)
    plan_remote_reanalysis_train_parser.add_argument("--root-value-weight", type=float, default=0.25)
    plan_remote_reanalysis_train_parser.add_argument("--temperature", type=float, default=0.55)
    plan_remote_reanalysis_train_parser.add_argument("--reanalysis-max-samples", type=int, default=2048)
    plan_remote_reanalysis_train_parser.add_argument("--reanalysis-epochs", type=int, default=2)
    plan_remote_reanalysis_train_parser.add_argument("--reanalysis-value-coef", type=float, default=0.35)
    plan_remote_reanalysis_train_parser.add_argument("--reanalysis-score-coef", type=float, default=0.05)
    plan_remote_reanalysis_train_parser.add_argument(
        "--allow-unknown-remote-features",
        action="store_true",
        help="Plan even if the poll file lacks remote reanalysis feature flags.",
    )
    plan_remote_reanalysis_train_parser.add_argument(
        "--allow-partial-poll",
        action="store_true",
        help=(
            "Allow automatic seed selection from a poll that does not contain "
            "all default workers. Prefer an explicit --seed for one-worker preflights."
        ),
    )
    plan_remote_reanalysis_train_parser.add_argument(
        "--allow-grade-busy-workers",
        action="store_true",
        help=(
            "Allow planning on workers that are running remote grades but have "
            "no active train_ppo.py or generate_reanalysis.py process."
        ),
    )
    plan_remote_reanalysis_train_parser.add_argument("--output")

    plan_remote_opening_eval_parser = subparsers.add_parser("plan-remote-opening-evals")
    plan_remote_opening_eval_parser.add_argument(
        "--poll",
        default="runs/self_play/gcp_poll_latest.json",
        help="Local JSON file written by poll.",
    )
    plan_remote_opening_eval_parser.add_argument(
        "--summary",
        default="runs/self_play/remote_grade_summary_latest.json",
        help="Local compact JSON file written by remote-grade-summary.",
    )
    plan_remote_opening_eval_parser.add_argument(
        "--local-status",
        default="",
        help=(
            "Optional JSON file written by local-controller-status. Workers "
            "claimed by local controller commands are treated as busy."
        ),
    )
    plan_remote_opening_eval_parser.add_argument("--output-dir", default="runs/self_play/remote_opening_evals")
    plan_remote_opening_eval_parser.add_argument("--log-dir", default="runs/self_play/logs")
    plan_remote_opening_eval_parser.add_argument("--games", type=int, default=16)
    plan_remote_opening_eval_parser.add_argument("--seed", type=int, default=93000)
    plan_remote_opening_eval_parser.add_argument("--vps-to-win", type=int, default=10)
    plan_remote_opening_eval_parser.add_argument("--max-opening-decisions", type=int, default=16)
    plan_remote_opening_eval_parser.add_argument("--candidate-limit", type=int, default=96)
    plan_remote_opening_eval_parser.add_argument("--presearch-candidate-limit", type=int, default=96)
    plan_remote_opening_eval_parser.add_argument("--rollout-decisions", type=int, default=2)
    plan_remote_opening_eval_parser.add_argument("--rollout-samples", type=int, default=1)
    plan_remote_opening_eval_parser.add_argument("--root-value-weight", type=float, default=0.35)
    plan_remote_opening_eval_parser.add_argument("--opponent-penalty", type=float, default=0.05)
    plan_remote_opening_eval_parser.add_argument("--max-evals", type=int, default=2)
    plan_remote_opening_eval_parser.add_argument("--max-per-family", type=int, default=1)
    plan_remote_opening_eval_parser.add_argument("--include-interim", action="store_true")
    plan_remote_opening_eval_parser.add_argument("--include-warmup", action="store_true")
    plan_remote_opening_eval_parser.add_argument(
        "--prefer-prefix",
        action="append",
        default=[],
        help="Prefer checkpoint names with this prefix. May be repeated.",
    )
    plan_remote_opening_eval_parser.add_argument(
        "--min-run-number",
        type=int,
        default=9900,
        help=(
            "Ignore checkpoint run numbers below this value. Set 0 to disable; "
            "default follows current fleet generations."
        ),
    )
    plan_remote_opening_eval_parser.add_argument(
        "--allow-busy-workers",
        action="store_true",
        help="Allow planning on workers that are training, grading, or locally claimed.",
    )
    plan_remote_opening_eval_parser.add_argument(
        "--allow-unknown-remote-features",
        action="store_true",
        help="Plan even if the poll file lacks the remote opening-evaluator feature flag.",
    )
    plan_remote_opening_eval_parser.add_argument("--output")

    plan_remote_sync_parser = subparsers.add_parser("plan-remote-code-sync")
    plan_remote_sync_parser.add_argument(
        "--poll",
        default="runs/self_play/gcp_poll_latest.json",
        help="Local JSON file written by poll.",
    )
    plan_remote_sync_parser.add_argument(
        "--summary",
        default="runs/self_play/remote_grade_summary_latest.json",
        help="Local compact JSON file written by remote-grade-summary.",
    )
    plan_remote_sync_parser.add_argument(
        "--local-status",
        default="",
        help="Optional JSON file written by local-controller-status.",
    )
    plan_remote_sync_parser.add_argument(
        "--recipe",
        choices=(
            "warmup_baseline",
            "warmup_jsettlers",
            "warmup_rollout",
            "pfsp_value_jsettlers",
            "pfsp_rollout_teacher",
            "pfsp_q_calibration",
            "pfsp_klent_control",
            "strict_repair_kl",
            "resource_plan_score_repair",
            "rollout_guard_score_repair",
            "tactical_rollout_guard_repair",
            "weighted_dagger_antireg",
            "jsettlers_dagger_antireg",
            "ema_jsettlers_antireg",
            "ema_mixed_antireg",
            "vrpo_esarsa_antireg",
            "vrpo_jsettlers_value_repair",
            "strict_gate_antireg",
            "strict_gate_distill_guard",
            "dags_midgame_reanalysis",
            "opening_eval",
        ),
        default="tactical_rollout_guard_repair",
        help="Sync workers missing the remote features required by this recipe.",
    )
    plan_remote_sync_parser.add_argument("--file", action="append", default=[])
    plan_remote_sync_parser.add_argument("--backup-dir", default="runs/self_play/code_backups")
    plan_remote_sync_parser.add_argument("--max-syncs", type=int, default=1)
    plan_remote_sync_parser.add_argument(
        "--allow-grade-busy-workers",
        action="store_true",
        help="Allow planning code syncs on workers currently running remote grades.",
    )
    plan_remote_sync_parser.add_argument("--output")

    remote_grade_status_parser = subparsers.add_parser("remote-grade-status")
    remote_grade_status_parser.add_argument("--eval-dir", default="runs/self_play/remote_grades_reanalysis")
    remote_grade_status_parser.add_argument("--log-dir", default="runs/self_play/logs")
    remote_grade_status_parser.add_argument("--output")

    remote_grade_summary_parser = subparsers.add_parser("remote-grade-summary")
    remote_grade_summary_parser.add_argument(
        "--input",
        default="runs/self_play/remote_grade_status_latest.json",
        help="Local JSON status file written by remote-grade-status.",
    )
    remote_grade_summary_parser.add_argument("--output")

    plan_remote_gates_parser = subparsers.add_parser("plan-remote-gates")
    plan_remote_gates_parser.add_argument(
        "--poll",
        default="runs/self_play/gcp_fleet_poll_latest.json",
        help="Local JSON file written by poll.",
    )
    plan_remote_gates_parser.add_argument(
        "--summary",
        default="runs/self_play/remote_grade_summary_latest.json",
        help="Local compact JSON file written by remote-grade-summary.",
    )
    plan_remote_gates_parser.add_argument(
        "--local-status",
        default="",
        help=(
            "Optional JSON file written by local-controller-status. Workers "
            "claimed by local remote-train/remote-grade commands are treated as busy."
        ),
    )
    plan_remote_gates_parser.add_argument("--champion", default=DEFAULT_CHAMPION)
    plan_remote_gates_parser.add_argument("--eval-dir", default="runs/self_play/remote_grades_reanalysis")
    plan_remote_gates_parser.add_argument("--log-dir", default="runs/self_play/logs")
    plan_remote_gates_parser.add_argument(
        "--profile",
        choices=("dev", "jsettlers_triage", "strict", "search_stress"),
        default="strict",
    )
    plan_remote_gates_parser.add_argument("--games", type=int, default=4)
    plan_remote_gates_parser.add_argument("--repeats", type=int, default=1)
    plan_remote_gates_parser.add_argument("--workers", type=int, default=1)
    plan_remote_gates_parser.add_argument("--vps-to-win", type=int, default=4)
    plan_remote_gates_parser.add_argument("--max-decisions", type=int, default=300)
    plan_remote_gates_parser.add_argument("--leg-timeout-seconds", type=int, default=1200)
    plan_remote_gates_parser.add_argument("--max-gates", type=int, default=4)
    plan_remote_gates_parser.add_argument(
        "--max-per-family",
        type=int,
        default=1,
        help=(
            "Maximum checkpoints to consider per checkpoint family. The default "
            "keeps latest-only behavior; higher values can compare early and "
            "later snapshots from the same active branch."
        ),
    )
    plan_remote_gates_parser.add_argument("--include-interim", action="store_true")
    plan_remote_gates_parser.add_argument(
        "--include-warmup",
        action="store_true",
        help=(
            "Allow .warmupNNNN.pt checkpoints to be gated. This is useful for "
            "warmup-only teacher-distillation branches where early snapshots "
            "are the main pruning signal."
        ),
    )
    plan_remote_gates_parser.add_argument(
        "--prefer-prefix",
        action="append",
        default=[],
        help="Checkpoint filename prefix to prioritize. May be passed more than once.",
    )
    plan_remote_gates_parser.add_argument(
        "--min-run-number",
        type=int,
        default=0,
        help=(
            "Ignore checkpoints whose run number after --run-prefix is lower. "
            "For run-prefix s97, both 82 and 9782 select s9782+."
        ),
    )
    plan_remote_gates_parser.add_argument(
        "--allow-busy-workers",
        action="store_true",
        help="Plan gates on workers already running a remote grade.",
    )
    plan_remote_gates_parser.add_argument(
        "--allow-training-busy-workers",
        action="store_true",
        help=(
            "Plan gates on workers that are training but do not already have "
            "an active remote grade."
        ),
    )
    plan_remote_gates_parser.add_argument(
        "--allow-rejected-family-continuation",
        action="store_true",
        help=(
            "Plan later snapshots from families that already failed a strict "
            "gate with opponent regression."
        ),
    )
    plan_remote_gates_parser.add_argument("--output")

    plan_remote_transfer_gates_parser = subparsers.add_parser("plan-remote-transfer-gates")
    plan_remote_transfer_gates_parser.add_argument(
        "--poll",
        default="runs/self_play/gcp_fleet_poll_latest.json",
        help="Local JSON file written by poll.",
    )
    plan_remote_transfer_gates_parser.add_argument(
        "--summary",
        default="runs/self_play/remote_grade_summary_latest.json",
        help="Local compact JSON file written by remote-grade-summary.",
    )
    plan_remote_transfer_gates_parser.add_argument(
        "--local-status",
        default="",
        help=(
            "Optional JSON file written by local-controller-status. Workers "
            "claimed by local remote-train/remote-grade commands are treated as busy."
        ),
    )
    plan_remote_transfer_gates_parser.add_argument("--champion", default=DEFAULT_CHAMPION)
    plan_remote_transfer_gates_parser.add_argument("--eval-dir", default="runs/self_play/remote_grades_reanalysis")
    plan_remote_transfer_gates_parser.add_argument("--log-dir", default="runs/self_play/logs")
    plan_remote_transfer_gates_parser.add_argument(
        "--profile",
        choices=("dev", "jsettlers_triage", "strict", "search_stress"),
        default="strict",
    )
    plan_remote_transfer_gates_parser.add_argument("--games", type=int, default=4)
    plan_remote_transfer_gates_parser.add_argument("--repeats", type=int, default=1)
    plan_remote_transfer_gates_parser.add_argument("--workers", type=int, default=1)
    plan_remote_transfer_gates_parser.add_argument("--vps-to-win", type=int, default=4)
    plan_remote_transfer_gates_parser.add_argument("--max-decisions", type=int, default=300)
    plan_remote_transfer_gates_parser.add_argument("--leg-timeout-seconds", type=int, default=1200)
    plan_remote_transfer_gates_parser.add_argument("--max-gates", type=int, default=4)
    plan_remote_transfer_gates_parser.add_argument(
        "--max-per-family",
        type=int,
        default=1,
        help="Maximum checkpoints to consider per checkpoint family.",
    )
    plan_remote_transfer_gates_parser.add_argument("--include-interim", action="store_true")
    plan_remote_transfer_gates_parser.add_argument(
        "--include-warmup",
        action="store_true",
        help="Allow .warmupNNNN.pt checkpoints to be transferred for gating.",
    )
    plan_remote_transfer_gates_parser.add_argument(
        "--prefer-prefix",
        action="append",
        default=[],
        help="Checkpoint filename prefix to prioritize. May be passed more than once.",
    )
    plan_remote_transfer_gates_parser.add_argument(
        "--min-run-number",
        type=int,
        default=0,
        help=(
            "Ignore checkpoints whose run number after --run-prefix is lower. "
            "For run-prefix s97, both 82 and 9782 select s9782+."
        ),
    )
    plan_remote_transfer_gates_parser.add_argument(
        "--allow-busy-target-workers",
        action="store_true",
        help="Plan transfer gates to target workers already running training or grading.",
    )
    plan_remote_transfer_gates_parser.add_argument(
        "--allow-training-busy-target-workers",
        action="store_true",
        help=(
            "Plan transfer gates to target workers that are training, while "
            "still skipping workers with active remote grades or local claims."
        ),
    )
    plan_remote_transfer_gates_parser.add_argument(
        "--allow-rejected-family-continuation",
        action="store_true",
        help=(
            "Plan later snapshots from families that already failed a strict "
            "gate with opponent regression."
        ),
    )
    plan_remote_transfer_gates_parser.add_argument("--output")

    plan_remote_escalations_parser = subparsers.add_parser("plan-remote-escalations")
    plan_remote_escalations_parser.add_argument(
        "--summary",
        default="runs/self_play/remote_grade_summary_latest.json",
        help="Local compact JSON file written by remote-grade-summary.",
    )
    plan_remote_escalations_parser.add_argument(
        "--local-status",
        default="",
        help=(
            "Optional JSON file written by local-controller-status. Workers "
            "claimed by local remote-train/remote-grade commands are treated as busy."
        ),
    )
    plan_remote_escalations_parser.add_argument("--champion", default=DEFAULT_CHAMPION)
    plan_remote_escalations_parser.add_argument("--eval-dir", default="runs/self_play/remote_grades_reanalysis")
    plan_remote_escalations_parser.add_argument("--log-dir", default="runs/self_play/logs")
    plan_remote_escalations_parser.add_argument(
        "--profile",
        choices=("dev", "jsettlers_triage", "strict", "search_stress"),
        default="strict",
    )
    plan_remote_escalations_parser.add_argument("--source-games", type=int, default=4)
    plan_remote_escalations_parser.add_argument("--target-games", type=int, default=12)
    plan_remote_escalations_parser.add_argument("--repeats", type=int, default=1)
    plan_remote_escalations_parser.add_argument("--workers", type=int, default=1)
    plan_remote_escalations_parser.add_argument("--vps-to-win", type=int, default=4)
    plan_remote_escalations_parser.add_argument("--max-decisions", type=int, default=300)
    plan_remote_escalations_parser.add_argument("--leg-timeout-seconds", type=int, default=1800)
    plan_remote_escalations_parser.add_argument("--max-escalations", type=int, default=4)
    plan_remote_escalations_parser.add_argument(
        "--allow-busy-workers",
        action="store_true",
        help="Plan escalations on workers already running a remote grade.",
    )
    plan_remote_escalations_parser.add_argument("--output")

    args = parser.parse_args()
    workers = parse_workers(args.worker or list(DEFAULT_WORKERS))
    if args.command == "poll":
        payload = poll_workers(
            workers,
            project=args.project,
            remote_repo=args.remote_repo,
            run_prefix=args.run_prefix,
        )
        if args.output:
            Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.command == "pull-ready":
        pulled = pull_ready_artifacts(
            workers,
            project=args.project,
            remote_repo=args.remote_repo,
            run_prefix=args.run_prefix,
            output_dir=Path(args.output_dir),
            include_interim=args.include_interim,
            prefer_prefix=tuple(args.prefer_prefix or ()),
            min_run_number=args.min_run_number,
            max_artifacts=args.max_artifacts,
            dry_run=args.dry_run,
        )
        print(json.dumps({"pulled": pulled}, indent=2, sort_keys=True))
    elif args.command == "grade-ready":
        checkpoints = select_local_checkpoints(
            Path(args.input_dir),
            run_prefix=args.run_prefix,
            include_interim=args.include_interim,
            max_checkpoints=args.max_checkpoints,
            latest_per_run=not args.all_snapshots,
        )
        command = build_grade_ready_command(args, checkpoints)
        print(json.dumps({"checkpoints": [str(p) for p in checkpoints], "command": command}, indent=2, sort_keys=True))
        if checkpoints and not args.dry_run:
            subprocess.run(command, check=True)
    elif args.command == "remote-grade":
        if len(workers) != 1:
            raise SystemExit("--worker name:zone must be provided exactly once for remote-grade")
        worker = workers[0]
        command = build_remote_grade_command(args)
        ssh_command = [
            "gcloud",
            "compute",
            "ssh",
            worker.name,
            "--zone",
            worker.zone,
            "--project",
            args.project,
            "--quiet",
            "--command",
            command,
        ]
        print(json.dumps({"command": ssh_command, "dry_run": args.dry_run}, sort_keys=True))
        if not args.dry_run:
            subprocess.run(ssh_command, check=True)
    elif args.command == "remote-grade-from-worker":
        if len(workers) != 1:
            raise SystemExit(
                "--worker name:zone must be provided exactly once for remote-grade-from-worker"
            )
        target = workers[0]
        source = parse_workers([args.source_worker])[0]
        payload = build_remote_grade_from_worker_command(
            args,
            source=source,
            target=target,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        if not args.dry_run:
            run_remote_grade_from_worker(args, source=source, target=target)
    elif args.command == "remote-opening-eval":
        if len(workers) != 1:
            raise SystemExit(
                "--worker name:zone must be provided exactly once for remote-opening-eval"
            )
        worker = workers[0]
        command = build_remote_opening_eval_command(args)
        ssh_command = [
            "gcloud",
            "compute",
            "ssh",
            worker.name,
            "--zone",
            worker.zone,
            "--project",
            args.project,
            "--quiet",
            "--command",
            command,
        ]
        print(json.dumps({"command": ssh_command, "dry_run": args.dry_run}, sort_keys=True))
        if not args.dry_run:
            subprocess.run(ssh_command, check=True)
    elif args.command == "remote-train":
        if len(workers) != 1:
            raise SystemExit("--worker name:zone must be provided exactly once for remote-train")
        worker = workers[0]
        command = build_remote_train_command(args)
        ssh_command = [
            "gcloud",
            "compute",
            "ssh",
            worker.name,
            "--zone",
            worker.zone,
            "--project",
            args.project,
            "--quiet",
            "--command",
            command,
        ]
        print(json.dumps({"command": ssh_command, "dry_run": args.dry_run}, sort_keys=True))
        if not args.dry_run:
            subprocess.run(ssh_command, check=True)
    elif args.command == "remote-reanalysis-train":
        if len(workers) != 1:
            raise SystemExit(
                "--worker name:zone must be provided exactly once for remote-reanalysis-train"
            )
        worker = workers[0]
        command = build_remote_reanalysis_train_command(args)
        ssh_command = [
            "gcloud",
            "compute",
            "ssh",
            worker.name,
            "--zone",
            worker.zone,
            "--project",
            args.project,
            "--quiet",
            "--command",
            command,
        ]
        print(json.dumps({"command": ssh_command, "dry_run": args.dry_run}, sort_keys=True))
        if not args.dry_run:
            subprocess.run(ssh_command, check=True)
    elif args.command == "remote-sync-code":
        if len(workers) != 1:
            raise SystemExit("--worker name:zone must be provided exactly once for remote-sync-code")
        worker = workers[0]
        payload = build_remote_sync_code_payload(args, worker=worker)
        print(json.dumps(payload, indent=2, sort_keys=True))
        if not args.dry_run:
            run_remote_sync_code(args, worker=worker)
    elif args.command == "remote-stop-train":
        if len(workers) != 1:
            raise SystemExit("--worker name:zone must be provided exactly once for remote-stop-train")
        worker = workers[0]
        command = build_remote_stop_train_command(args)
        ssh_command = [
            "gcloud",
            "compute",
            "ssh",
            worker.name,
            "--zone",
            worker.zone,
            "--project",
            args.project,
            "--quiet",
            "--command",
            command,
        ]
        print(json.dumps({"command": ssh_command, "dry_run": args.dry_run}, sort_keys=True))
        if not args.dry_run:
            subprocess.run(ssh_command, check=True)
    elif args.command == "remote-stop-grade":
        if len(workers) != 1:
            raise SystemExit("--worker name:zone must be provided exactly once for remote-stop-grade")
        worker = workers[0]
        command = build_remote_stop_grade_command(args)
        ssh_command = [
            "gcloud",
            "compute",
            "ssh",
            worker.name,
            "--zone",
            worker.zone,
            "--project",
            args.project,
            "--quiet",
            "--command",
            command,
        ]
        print(json.dumps({"command": ssh_command, "dry_run": args.dry_run}, sort_keys=True))
        if not args.dry_run:
            subprocess.run(ssh_command, check=True)
    elif args.command == "local-controller-status":
        payload = local_controller_status()
        if args.output:
            Path(args.output).write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.command == "plan-remote-train":
        payload = plan_remote_train(
            poll_payload=json.loads(Path(args.poll).read_text(encoding="utf-8")),
            summary_payload=json.loads(Path(args.summary).read_text(encoding="utf-8")),
            local_status_payload=load_optional_json(args.local_status),
            project=args.project,
            remote_repo=args.remote_repo,
            champion=args.champion,
            recipe=args.recipe,
            seed=args.seed,
            min_seed=args.min_seed,
            iterations=args.iterations,
            episodes_per_iteration=args.episodes_per_iteration,
            checkpoint_every=args.checkpoint_every,
            max_launches=args.max_launches,
            allow_unknown_remote_features=args.allow_unknown_remote_features,
            allow_partial_poll=args.allow_partial_poll,
            allow_grade_busy_workers=args.allow_grade_busy_workers,
        )
        if args.output:
            Path(args.output).write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.command == "plan-remote-reanalysis-train":
        payload = plan_remote_reanalysis_train(
            poll_payload=json.loads(Path(args.poll).read_text(encoding="utf-8")),
            summary_payload=json.loads(Path(args.summary).read_text(encoding="utf-8")),
            local_status_payload=load_optional_json(args.local_status),
            project=args.project,
            remote_repo=args.remote_repo,
            champion=args.champion,
            seed=args.seed,
            min_seed=args.min_seed,
            max_launches=args.max_launches,
            games=args.games,
            vps_to_win=args.vps_to_win,
            max_decisions=args.max_decisions,
            record_after_decisions=args.record_after_decisions,
            record_window_decisions=args.record_window_decisions,
            candidate_limit=args.candidate_limit,
            presearch_candidate_limit=args.presearch_candidate_limit,
            rollout_decisions=args.rollout_decisions,
            rollout_samples=args.rollout_samples,
            root_value_weight=args.root_value_weight,
            temperature=args.temperature,
            reanalysis_max_samples=args.reanalysis_max_samples,
            reanalysis_epochs=args.reanalysis_epochs,
            reanalysis_value_coef=args.reanalysis_value_coef,
            reanalysis_score_coef=args.reanalysis_score_coef,
            allow_unknown_remote_features=args.allow_unknown_remote_features,
            allow_partial_poll=args.allow_partial_poll,
            allow_grade_busy_workers=args.allow_grade_busy_workers,
        )
        if args.output:
            Path(args.output).write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.command == "plan-remote-opening-evals":
        payload = plan_remote_opening_evals(
            poll_payload=json.loads(Path(args.poll).read_text(encoding="utf-8")),
            summary_payload=json.loads(Path(args.summary).read_text(encoding="utf-8")),
            local_status_payload=load_optional_json(args.local_status),
            project=args.project,
            remote_repo=args.remote_repo,
            run_prefix=args.run_prefix,
            output_dir=args.output_dir,
            log_dir=args.log_dir,
            games=args.games,
            seed=args.seed,
            vps_to_win=args.vps_to_win,
            max_opening_decisions=args.max_opening_decisions,
            candidate_limit=args.candidate_limit,
            presearch_candidate_limit=args.presearch_candidate_limit,
            rollout_decisions=args.rollout_decisions,
            rollout_samples=args.rollout_samples,
            root_value_weight=args.root_value_weight,
            opponent_penalty=args.opponent_penalty,
            max_evals=args.max_evals,
            max_per_family=args.max_per_family,
            include_interim=args.include_interim,
            include_warmup=args.include_warmup,
            prefer_prefixes=args.prefer_prefix,
            min_run_number=args.min_run_number,
            allow_busy_workers=args.allow_busy_workers,
            allow_unknown_remote_features=args.allow_unknown_remote_features,
        )
        if args.output:
            Path(args.output).write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.command == "plan-remote-code-sync":
        payload = plan_remote_code_sync(
            poll_payload=json.loads(Path(args.poll).read_text(encoding="utf-8")),
            summary_payload=json.loads(Path(args.summary).read_text(encoding="utf-8")),
            local_status_payload=load_optional_json(args.local_status),
            project=args.project,
            remote_repo=args.remote_repo,
            recipe=args.recipe,
            files=args.file,
            backup_dir=args.backup_dir,
            max_syncs=args.max_syncs,
            allow_grade_busy_workers=args.allow_grade_busy_workers,
        )
        if args.output:
            Path(args.output).write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.command == "remote-grade-status":
        payload = remote_grade_status(
            workers,
            project=args.project,
            remote_repo=args.remote_repo,
            eval_dir=args.eval_dir,
            log_dir=args.log_dir,
        )
        if args.output:
            Path(args.output).write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.command == "remote-grade-summary":
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
        summary = summarize_remote_grade_status(payload)
        if args.output:
            Path(args.output).write_text(
                json.dumps(summary, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        print(json.dumps(summary, indent=2, sort_keys=True))
    elif args.command == "plan-remote-gates":
        payload = plan_remote_gates(
            poll_payload=json.loads(Path(args.poll).read_text(encoding="utf-8")),
            summary_payload=json.loads(Path(args.summary).read_text(encoding="utf-8")),
            local_status_payload=load_optional_json(args.local_status),
            project=args.project,
            remote_repo=args.remote_repo,
            run_prefix=args.run_prefix,
            champion=args.champion,
            eval_dir=args.eval_dir,
            log_dir=args.log_dir,
            profile=args.profile,
            games=args.games,
            repeats=args.repeats,
            grade_workers=args.workers,
            vps_to_win=args.vps_to_win,
            max_decisions=args.max_decisions,
            leg_timeout_seconds=args.leg_timeout_seconds,
            max_gates=args.max_gates,
            max_per_family=args.max_per_family,
            include_interim=args.include_interim,
            include_warmup=args.include_warmup,
            prefer_prefixes=args.prefer_prefix,
            min_run_number=args.min_run_number,
            allow_busy_workers=args.allow_busy_workers,
            allow_training_busy_workers=args.allow_training_busy_workers,
            allow_rejected_family_continuation=args.allow_rejected_family_continuation,
        )
        if args.output:
            Path(args.output).write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.command == "plan-remote-transfer-gates":
        payload = plan_remote_transfer_gates(
            poll_payload=json.loads(Path(args.poll).read_text(encoding="utf-8")),
            summary_payload=json.loads(Path(args.summary).read_text(encoding="utf-8")),
            local_status_payload=load_optional_json(args.local_status),
            project=args.project,
            remote_repo=args.remote_repo,
            run_prefix=args.run_prefix,
            champion=args.champion,
            eval_dir=args.eval_dir,
            log_dir=args.log_dir,
            profile=args.profile,
            games=args.games,
            repeats=args.repeats,
            grade_workers=args.workers,
            vps_to_win=args.vps_to_win,
            max_decisions=args.max_decisions,
            leg_timeout_seconds=args.leg_timeout_seconds,
            max_gates=args.max_gates,
            max_per_family=args.max_per_family,
            include_interim=args.include_interim,
            include_warmup=args.include_warmup,
            prefer_prefixes=args.prefer_prefix,
            min_run_number=args.min_run_number,
            allow_busy_target_workers=args.allow_busy_target_workers,
            allow_training_busy_target_workers=args.allow_training_busy_target_workers,
            allow_rejected_family_continuation=args.allow_rejected_family_continuation,
        )
        if args.output:
            Path(args.output).write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.command == "plan-remote-escalations":
        payload = plan_remote_escalations(
            summary_payload=json.loads(Path(args.summary).read_text(encoding="utf-8")),
            local_status_payload=load_optional_json(args.local_status),
            project=args.project,
            remote_repo=args.remote_repo,
            champion=args.champion,
            eval_dir=args.eval_dir,
            log_dir=args.log_dir,
            profile=args.profile,
            source_games=args.source_games,
            target_games=args.target_games,
            repeats=args.repeats,
            grade_workers=args.workers,
            vps_to_win=args.vps_to_win,
            max_decisions=args.max_decisions,
            leg_timeout_seconds=args.leg_timeout_seconds,
            max_escalations=args.max_escalations,
            allow_busy_workers=args.allow_busy_workers,
        )
        if args.output:
            Path(args.output).write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:  # pragma: no cover
        raise ValueError(args.command)


def parse_workers(values: list[str]) -> list[Worker]:
    workers = []
    for value in values:
        name, sep, zone = value.partition(":")
        if sep != ":" or not name or not zone:
            raise SystemExit(f"invalid worker {value!r}; expected name:zone")
        workers.append(Worker(name=name, zone=zone))
    return workers


def load_optional_json(path: str) -> dict[str, Any] | None:
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


LOCAL_CONTROLLER_COMMANDS = frozenset(
    {
        "remote-grade",
        "remote-grade-from-worker",
        "remote-opening-eval",
        "remote-train",
        "remote-reanalysis-train",
        "remote-sync-code",
        "remote-stop-train",
        "remote-stop-grade",
        "plan-remote-train",
        "plan-remote-reanalysis-train",
        "plan-remote-opening-evals",
        "plan-remote-code-sync",
        "plan-remote-gates",
        "plan-remote-transfer-gates",
        "plan-remote-escalations",
        "poll",
        "remote-grade-status",
        "remote-grade-summary",
    }
)


def local_controller_status(
    ps_output: str | None = None,
    *,
    current_pid: int | None = None,
) -> dict[str, Any]:
    if ps_output is None:
        ps_output = subprocess.run(
            ["ps", "-axo", "pid,etime,command"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout
    if current_pid is None:
        current_pid = os.getpid()
    jobs = [
        job
        for job in (
            parse_local_controller_process(line, current_pid=current_pid)
            for line in ps_output.splitlines()
        )
        if job is not None
    ]
    claimed_workers = sorted(
        {
            str(job["worker"])
            for job in jobs
            if job.get("worker")
            and job.get("kind") in {"remote_grade", "remote_train", "remote_sync"}
        }
    )
    active_grades = [
        job
        for job in jobs
        if job.get("kind") == "remote_grade"
    ]
    active_trains = [
        job
        for job in jobs
        if job.get("kind") == "remote_train"
    ]
    return {
        "active_count": len(jobs),
        "claimed_workers": claimed_workers,
        "active_grades": active_grades,
        "active_trains": active_trains,
        "jobs": jobs,
    }


def claimed_workers_from_local_status(payload: dict[str, Any] | None) -> set[str]:
    if not payload:
        return set()
    return {
        str(worker)
        for worker in payload.get("claimed_workers", [])
        if worker
    }


def decision_matches_profile(row: dict[str, Any], profile: str) -> bool:
    row_profile = str(row.get("profile") or "")
    return not row_profile or row_profile == "unknown" or row_profile == profile


def is_family_blocking_reject(row: dict[str, Any]) -> bool:
    if row.get("decision") != "reject":
        return False
    reason = str(row.get("reason", ""))
    return "opponent regression" in reason or "candidate timed out" in reason


def parse_local_controller_process(
    line: str,
    *,
    current_pid: int,
) -> dict[str, Any] | None:
    match = re.match(r"\s*(\d+)\s+(\S+)\s+(.*)$", line)
    if not match:
        return None
    pid = int(match.group(1))
    if pid == current_pid:
        return None
    elapsed = match.group(2)
    command_text = match.group(3)
    if "tools/gcp_fleet_controller.py" not in command_text:
        return None
    try:
        tokens = shlex.split(command_text)
    except ValueError:
        tokens = command_text.split()
    if _is_shell_wrapper(tokens):
        return None
    if not any("python" in Path(token).name.lower() for token in tokens):
        return None
    tool_index = next(
        (
            index
            for index, token in enumerate(tokens)
            if token.endswith("tools/gcp_fleet_controller.py")
        ),
        None,
    )
    if tool_index is None:
        return None
    args = tokens[tool_index + 1 :]
    command = next((token for token in args if token in LOCAL_CONTROLLER_COMMANDS), None)
    if command is None or command == "local-controller-status":
        return None
    worker = _worker_name_from_spec(_last_flag_value(args, "--worker"))
    checkpoint = _last_flag_value(args, "--checkpoint")
    job: dict[str, Any] = {
        "pid": pid,
        "elapsed": elapsed,
        "command": command,
        "kind": _local_controller_job_kind(command),
        "worker": worker,
        "checkpoint": checkpoint,
        "force": "--force" in args,
    }
    label = _last_flag_value(args, "--label")
    if label:
        job["label"] = label
    source_worker = _worker_name_from_spec(_last_flag_value(args, "--source-worker"))
    if source_worker:
        job["source_worker"] = source_worker
    profile = _last_flag_value(args, "--profile")
    if profile:
        job["profile"] = profile
    return {key: value for key, value in job.items() if value is not None}


def _is_shell_wrapper(tokens: list[str]) -> bool:
    if not tokens:
        return False
    executable = Path(tokens[0]).name
    return executable in {"bash", "dash", "fish", "sh", "zsh"} and any(
        token == "-c" or token.endswith("c") and token.startswith("-")
        for token in tokens[1:3]
    )


def _local_controller_job_kind(command: str) -> str:
    if command in {"remote-grade", "remote-grade-from-worker", "remote-opening-eval"}:
        return "remote_grade"
    if command in {"remote-train", "remote-reanalysis-train"}:
        return "remote_train"
    if command == "remote-sync-code":
        return "remote_sync"
    if command.startswith("plan-"):
        return "planner"
    if command.startswith("remote-stop"):
        return "remote_stop"
    return "status"


def _last_flag_value(args: list[str], flag: str) -> str | None:
    values = [
        args[index + 1]
        for index, token in enumerate(args[:-1])
        if token == flag
    ]
    return values[-1] if values else None


def _worker_name_from_spec(spec: str | None) -> str | None:
    if not spec:
        return None
    return spec.split(":", 1)[0]


def build_grade_ready_command(
    args: argparse.Namespace,
    checkpoints: list[Path],
) -> list[str]:
    command = [
        sys.executable,
        "tools/grade_agent.py",
        "--champion",
        args.champion,
        "--eval-dir",
        args.eval_dir,
        "--profile",
        args.profile,
        "--games",
        str(args.games),
        "--repeats",
        str(args.repeats),
        "--workers",
        str(args.workers),
        "--leg-timeout-seconds",
        str(args.leg_timeout_seconds),
        "--vps-to-win",
        str(args.vps_to_win),
        "--max-decisions",
        str(args.max_decisions),
    ]
    for opponent in getattr(args, "opponent", None) or ():
        command.extend(["--opponent", opponent])
    for weight in getattr(args, "opponent_weight", ()):
        command.extend(["--opponent-weight", weight])
    _extend_optional_arg(
        command,
        "--opponent-candidate-limit",
        getattr(args, "opponent_candidate_limit", None),
    )
    _extend_optional_arg(
        command,
        "--opponent-rollout-decisions",
        getattr(args, "opponent_rollout_decisions", None),
    )
    _extend_optional_arg(
        command,
        "--opponent-value-penalty",
        getattr(args, "opponent_value_penalty", None),
    )
    for checkpoint in checkpoints:
        command.extend(["--checkpoint", str(checkpoint)])
    if getattr(args, "dry_run", False):
        command.append("--dry-run")
    return command


def _extend_optional_arg(command: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def build_remote_grade_command(args: argparse.Namespace) -> str:
    repo = _remote_repo_shell_expr(args.remote_repo)
    checkpoint = str(args.checkpoint)
    run_id = remote_grade_run_id(args)
    log_dir = str(args.log_dir)
    eval_dir = str(args.eval_dir)
    log_path = f"{log_dir}/remote_grade_{run_id}.log"
    summary_path = f"{eval_dir}/summary_{run_id}.json"
    grade_command = [
        ".venv/bin/python",
        "-u",
        "tools/grade_agent.py",
        "--profile",
        str(args.profile),
        "--champion",
        str(args.champion),
        "--eval-dir",
        eval_dir,
        "--games",
        str(args.games),
        "--repeats",
        str(args.repeats),
        "--workers",
        str(args.workers),
        "--vps-to-win",
        str(args.vps_to_win),
        "--max-decisions",
        str(args.max_decisions),
        "--leg-timeout-seconds",
        str(args.leg_timeout_seconds),
        "--summary-output",
        summary_path,
        "--checkpoint",
        checkpoint,
    ]
    quoted_grade = " ".join(shlex.quote(part) for part in grade_command)
    summary_exists_payload = json.dumps(
        {
            "checkpoint": checkpoint,
            "reason": "summary_exists",
            "run_id": run_id,
            "skipped": True,
            "summary": summary_path,
        },
        sort_keys=True,
    )
    active_payload = json.dumps(
        {
            "checkpoint": checkpoint,
            "reason": "already_active",
            "run_id": run_id,
            "skipped": True,
            "summary": summary_path,
        },
        sort_keys=True,
    )
    preflight = ""
    if not bool(getattr(args, "force", False)):
        preflight = (
            "if [ -s \"$summary_path\" ]; then "
            f"printf '%s\\n' {shlex.quote(summary_exists_payload)}; "
            "exit 0; "
            "fi; "
            "active_grade=$(pgrep -af '[t]ools/grade_agent.py' "
            "| grep -F -- \"--summary-output $summary_path\" "
            "| grep -Ev '(^|[[:space:]])((/bin/)?(ba)?sh)[[:space:]]+-c[[:space:]]' "
            "|| true); "
            "if [ -n \"$active_grade\" ]; then "
            f"printf '%s\\n' {shlex.quote(active_payload)}; "
            "exit 0; "
            "fi; "
        )
    return (
        "set -e; "
        f"repo={repo}; "
        f"summary_path={shlex.quote(summary_path)}; "
        f"log_path={shlex.quote(log_path)}; "
        "if [ -z \"$repo\" ]; then "
        "base=\"$HOME/catan-zero\"; nested=\"$base/catan-zero-gcp-bundle\"; "
        "if [ -f \"$nested/pyproject.toml\" ]; then repo=\"$nested\"; else repo=\"$base\"; fi; "
        "fi; "
        "cd \"$repo\"; "
        f"mkdir -p {shlex.quote(log_dir)} {shlex.quote(eval_dir)}; "
        f"{preflight}"
        f"setsid sh -c {shlex.quote(quoted_grade + ' >> ' + shlex.quote(log_path) + ' 2>&1 < /dev/null')} "
        "> /dev/null 2>&1 & "
        "pid=$!; "
        f"printf '%s\\n' {shlex.quote(json.dumps({'remote_pid': '__PID__', 'log': log_path, 'summary': summary_path, 'checkpoint': checkpoint, 'run_id': run_id}, sort_keys=True))} | sed \"s/__PID__/$pid/\""
    )


def build_remote_train_command(args: argparse.Namespace) -> str:
    repo = _remote_repo_shell_expr(args.remote_repo)
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(args.label)).strip("._-")
    if not label:
        raise SystemExit("--label must contain at least one safe character")
    train_args = list(args.train_args)
    if train_args and train_args[0] == "--":
        train_args = train_args[1:]
    if not train_args:
        raise SystemExit("remote-train requires forwarded train args after --")
    log_dir = str(args.log_dir)
    log_path = f"{log_dir}/{label}.log"
    checkpoint_path = _forwarded_arg_value(train_args, "--checkpoint")
    report_path = _forwarded_arg_value(train_args, "--report")
    train_command = [
        ".venv/bin/python",
        "-u",
        "tools/train_ppo.py",
        *train_args,
    ]
    quoted_train = " ".join(shlex.quote(part) for part in train_command)
    busy_payload = json.dumps(
        {
            "label": label,
            "log": log_path,
            "reason": "worker_busy",
            "skipped": True,
        },
        sort_keys=True,
    )
    existing_payload = json.dumps(
        {
            "checkpoint": checkpoint_path,
            "label": label,
            "log": log_path,
            "reason": "artifact_exists",
            "report": report_path,
            "skipped": True,
        },
        sort_keys=True,
    )
    preflight = ""
    if not bool(getattr(args, "force", False)):
        preflight = (
            "if { [ -n \"$checkpoint_path\" ] && [ -e \"$checkpoint_path\" ]; } || "
            "{ [ -n \"$report_path\" ] && [ -e \"$report_path\" ]; }; then "
            f"printf '%s\\n' {shlex.quote(existing_payload)}; "
            "exit 0; "
            "fi; "
            "active_train=$(pgrep -af '[t]ools/train_ppo.py' "
            "| grep -Ev '(^|[[:space:]])((/bin/)?(ba)?sh)[[:space:]]+-c[[:space:]]' "
            "|| true); "
            "if [ -n \"$active_train\" ]; then "
            f"printf '%s\\n' {shlex.quote(busy_payload)}; "
            "exit 0; "
            "fi; "
        )
    return (
        "set -e; "
        f"repo={repo}; "
        "if [ -z \"$repo\" ]; then "
        "base=\"$HOME/catan-zero\"; nested=\"$base/catan-zero-gcp-bundle\"; "
        "if [ -f \"$nested/pyproject.toml\" ]; then repo=\"$nested\"; else repo=\"$base\"; fi; "
        "fi; "
        "cd \"$repo\"; "
        f"checkpoint_path={shlex.quote(checkpoint_path or '')}; "
        f"report_path={shlex.quote(report_path or '')}; "
        f"mkdir -p {shlex.quote(log_dir)}; "
        f"{preflight}"
        f"setsid sh -c {shlex.quote(quoted_train + ' >> ' + shlex.quote(log_path) + ' 2>&1 < /dev/null')} "
        "> /dev/null 2>&1 & "
        "pid=$!; "
        f"printf '%s\\n' {shlex.quote(json.dumps({'remote_pid': '__PID__', 'log': log_path, 'label': label}))} | sed \"s/__PID__/$pid/\""
    )


def _forwarded_arg_value(args: list[str], flag: str) -> str:
    try:
        index = args.index(flag)
    except ValueError:
        return ""
    if index + 1 >= len(args):
        return ""
    value = str(args[index + 1])
    if value.startswith("--"):
        return ""
    return value


def build_remote_sync_code_launch_command(
    *,
    worker: Worker,
    project: str,
    remote_repo: str,
    files: list[str],
    backup_dir: str,
) -> list[str]:
    command = [
        sys.executable,
        "tools/gcp_fleet_controller.py",
        "--project",
        project,
        "--worker",
        f"{worker.name}:{worker.zone}",
    ]
    if remote_repo:
        command.extend(["--remote-repo", remote_repo])
    command.extend(["remote-sync-code", "--backup-dir", backup_dir])
    for file_name in normalized_sync_files(files):
        command.extend(["--file", file_name])
    return command


def build_remote_sync_code_payload(
    args: argparse.Namespace,
    *,
    worker: Worker,
) -> dict[str, Any]:
    files = normalized_sync_files(getattr(args, "file", []))
    repo_label = str(args.remote_repo).rstrip("/") if args.remote_repo else "<auto>"
    preflight = [
        "gcloud",
        "compute",
        "ssh",
        worker.name,
        "--zone",
        worker.zone,
        "--project",
        str(args.project),
        "--quiet",
        "--command",
        remote_sync_preflight_command(
            remote_repo=str(args.remote_repo),
            files=files,
            backup_dir=str(args.backup_dir),
            allow_busy=bool(args.allow_busy),
        ),
    ]
    copy_commands = [
        [
            "gcloud",
            "compute",
            "scp",
            file_name,
            f"{worker.name}:{repo_label}/{file_name}",
            "--zone",
            worker.zone,
            "--project",
            str(args.project),
            "--quiet",
        ]
        for file_name in files
    ]
    return {
        "worker": f"{worker.name}:{worker.zone}",
        "remote_repo": repo_label,
        "files": files,
        "preflight": preflight,
        "copy_files": copy_commands,
        "dry_run": bool(args.dry_run),
        "allow_busy": bool(args.allow_busy),
    }


def run_remote_sync_code(args: argparse.Namespace, *, worker: Worker) -> None:
    files = normalized_sync_files(getattr(args, "file", []))
    preflight = [
        "gcloud",
        "compute",
        "ssh",
        worker.name,
        "--zone",
        worker.zone,
        "--project",
        str(args.project),
        "--quiet",
        "--command",
        remote_sync_preflight_command(
            remote_repo=str(args.remote_repo),
            files=files,
            backup_dir=str(args.backup_dir),
            allow_busy=bool(args.allow_busy),
        ),
    ]
    result = subprocess.run(
        preflight,
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)
    if payload.get("skipped"):
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    target_repo = str(payload["repo"]).rstrip("/")
    copied = []
    for file_name in files:
        command = [
            "gcloud",
            "compute",
            "scp",
            file_name,
            f"{worker.name}:{target_repo}/{file_name}",
            "--zone",
            worker.zone,
            "--project",
            str(args.project),
            "--quiet",
        ]
        run_command_with_retries(command, attempts=3)
        copied.append(file_name)
    print(
        json.dumps(
            {
                "worker": worker.name,
                "zone": worker.zone,
                "repo": target_repo,
                "backup": payload.get("backup"),
                "copied": copied,
            },
            indent=2,
            sort_keys=True,
        )
    )


def remote_sync_preflight_command(
    *,
    remote_repo: str,
    files: list[str],
    backup_dir: str,
    allow_busy: bool,
) -> str:
    repo_expr = repr(remote_repo)
    files_expr = repr(normalized_sync_files(files))
    backup_expr = repr(backup_dir)
    allow_busy_expr = "True" if allow_busy else "False"
    return f"""python3 - <<'PY'
import datetime, json, os, pathlib, shutil, subprocess
configured={repo_expr}
if configured:
    repo=os.path.expanduser(configured)
else:
    base=os.path.expanduser('~/catan-zero')
    nested=os.path.join(base, 'catan-zero-gcp-bundle')
    repo=nested if os.path.exists(os.path.join(nested, 'pyproject.toml')) else base
os.chdir(repo)
files={files_expr}
backup_dir={backup_expr}
allow_busy={allow_busy_expr}
def active_lines(pattern):
    raw=subprocess.run(['pgrep','-af',pattern], text=True, stdout=subprocess.PIPE).stdout
    lines=[]
    for line in raw.splitlines():
        if 'pgrep -af' in line or "python3 - <<'PY'" in line:
            continue
        if '/bin/sh -c' in line or ' bash -c ' in line or ' sh -c ' in line:
            continue
        lines.append(line[:1000])
    return lines
active_train=active_lines('tools/train_ppo.py')
active_grade=active_lines('tools/grade_agent.py')
if not allow_busy and (active_train or active_grade):
    print(json.dumps({{
        'repo': repo,
        'skipped': True,
        'reason': 'worker_busy',
        'active_train': active_train,
        'active_grade': active_grade,
    }}, sort_keys=True))
    raise SystemExit(0)
stamp=datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
backup_root=pathlib.Path(backup_dir) / ('sync_' + stamp)
backed_up=[]
for name in files:
    source=pathlib.Path(name)
    if source.is_absolute() or '..' in source.parts:
        raise ValueError(f'unsafe sync path {{name!r}}')
    if source.exists():
        destination=backup_root / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        backed_up.append(name)
print(json.dumps({{
    'repo': repo,
    'skipped': False,
    'backup': str(backup_root),
    'backed_up': backed_up,
}}, sort_keys=True))
PY"""


def normalized_sync_files(files: list[str]) -> list[str]:
    selected = files or list(DEFAULT_SYNC_FILES)
    normalized: list[str] = []
    for file_name in selected:
        path = Path(file_name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe sync path {file_name!r}")
        if not path.is_file():
            raise FileNotFoundError(file_name)
        normalized.append(path.as_posix())
    return normalized


def plan_remote_train(
    *,
    poll_payload: dict[str, Any],
    summary_payload: dict[str, Any],
    project: str,
    remote_repo: str,
    champion: str,
    recipe: str,
    seed: int,
    min_seed: int,
    iterations: int,
    episodes_per_iteration: int,
    checkpoint_every: int,
    max_launches: int,
    allow_unknown_remote_features: bool = False,
    allow_partial_poll: bool = False,
    allow_grade_busy_workers: bool = False,
    local_status_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    active_grade_workers = {
        str(row.get("worker"))
        for row in summary_payload.get("active", [])
        if row.get("worker")
    }
    local_claimed_workers = claimed_workers_from_local_status(local_status_payload)
    skipped: dict[str, list[dict[str, Any]]] = {
        "busy_worker": [],
        "inactive_worker": [],
        "missing_remote_feature": [],
        "partial_poll": [],
    }
    if int(seed) <= 0 and not allow_partial_poll and not is_full_default_worker_poll(poll_payload):
        observed = sorted(
            str(row.get("worker", ""))
            for row in poll_payload.get("workers", [])
            if row.get("worker")
        )
        next_seed = next_training_seed(
            poll_payload=poll_payload,
            summary_payload=summary_payload,
            min_seed=min_seed,
        )
        skipped["partial_poll"].append(
            {
                "observed_workers": observed,
                "expected_workers": sorted(DEFAULT_WORKER_NAMES),
                "reason": "automatic seed selection requires a full fleet poll or explicit --seed",
            }
        )
        return {
            "planned_count": 0,
            "planned": [],
            "next_seed": next_seed,
            "skipped": {key: value for key, value in skipped.items() if value},
        }
    automatic_seed = int(seed) <= 0
    seed_floor = int(seed) if int(seed) > 0 else next_training_seed(
        poll_payload=poll_payload,
        summary_payload=summary_payload,
        min_seed=min_seed,
    )
    required_features = required_remote_features_for_recipe(recipe)
    consumed_seeds = consumed_training_seeds(
        poll_payload=poll_payload,
        summary_payload=summary_payload,
    )
    reserved_seeds = set(consumed_seeds)
    planned: list[dict[str, Any]] = []
    for row in poll_payload.get("workers", []):
        worker = Worker(str(row.get("worker", "")), str(row.get("zone", "")))
        if not worker.name or not worker.zone or not row.get("ok", True):
            skipped["inactive_worker"].append(
                {
                    "worker": worker.name,
                    "zone": worker.zone,
                    "error": row.get("error"),
                }
            )
            continue
        busy_reason = None
        if int(row.get("running_train_processes", 0) or 0) > 0:
            busy_reason = "training"
        elif worker.name in active_grade_workers and not allow_grade_busy_workers:
            busy_reason = "remote_grade"
        elif worker.name in local_claimed_workers:
            busy_reason = "local_controller"
        if busy_reason is not None:
            skipped["busy_worker"].append(
                {
                    "worker": worker.name,
                    "zone": worker.zone,
                    "reason": busy_reason,
                }
            )
            continue
        missing_features = missing_remote_features(
            row,
            required_features=required_features,
            allow_unknown=allow_unknown_remote_features,
        )
        if missing_features:
            skipped["missing_remote_feature"].append(
                {
                    "worker": worker.name,
                    "zone": worker.zone,
                    "missing": missing_features,
                    "repo": row.get("repo"),
                    "trainer_sha1": row.get("trainer_sha1"),
                }
            )
            continue
        planned_seed = planned_training_seed(
            seed_floor=seed_floor,
            worker_name=worker.name,
            automatic_seed=automatic_seed,
            reserved_seeds=reserved_seeds,
        )
        reserved_seeds.add(planned_seed)
        label = training_label(
            seed=planned_seed,
            recipe=recipe,
            worker_name=worker.name,
        )
        train_args = build_planned_training_args(
            label=label,
            seed=planned_seed,
            champion=champion,
            recipe=recipe,
            iterations=iterations,
            episodes_per_iteration=episodes_per_iteration,
            checkpoint_every=checkpoint_every,
        )
        command = build_remote_train_launch_command(
            worker=worker,
            project=project,
            remote_repo=remote_repo,
            label=label,
            train_args=train_args,
        )
        planned.append(
            {
                "worker": worker.name,
                "zone": worker.zone,
                "seed": planned_seed,
                "label": label,
                "recipe": recipe,
                "checkpoint": f"runs/self_play/{label}.pt",
                "report": f"runs/self_play/{label}.json",
                "train_args": train_args,
                "command": command,
                "shell": " ".join(shlex.quote(part) for part in command),
            }
        )
        seed_floor = max(seed_floor + 1, planned_seed + 1)
        if len(planned) >= max(0, int(max_launches)):
            break
    return {
        "planned_count": len(planned),
        "planned": planned,
        "next_seed": seed_floor,
        "skipped": {key: value for key, value in skipped.items() if value},
    }


def plan_remote_reanalysis_train(
    *,
    poll_payload: dict[str, Any],
    summary_payload: dict[str, Any],
    local_status_payload: dict[str, Any] | None,
    project: str,
    remote_repo: str,
    champion: str,
    seed: int,
    min_seed: int,
    max_launches: int,
    games: int,
    vps_to_win: int,
    max_decisions: int,
    record_after_decisions: int,
    record_window_decisions: int,
    candidate_limit: int,
    presearch_candidate_limit: int,
    rollout_decisions: int,
    rollout_samples: int,
    root_value_weight: float,
    temperature: float,
    reanalysis_max_samples: int,
    reanalysis_epochs: int,
    reanalysis_value_coef: float,
    reanalysis_score_coef: float,
    allow_unknown_remote_features: bool = False,
    allow_partial_poll: bool = False,
    allow_grade_busy_workers: bool = False,
) -> dict[str, Any]:
    active_grade_workers = {
        str(row.get("worker"))
        for row in summary_payload.get("active", [])
        if row.get("worker")
    }
    local_claimed_workers = claimed_workers_from_local_status(local_status_payload)
    skipped: dict[str, list[dict[str, Any]]] = {
        "busy_worker": [],
        "inactive_worker": [],
        "missing_remote_feature": [],
        "partial_poll": [],
    }
    if int(seed) <= 0 and not allow_partial_poll and not is_full_default_worker_poll(poll_payload):
        observed = sorted(
            str(row.get("worker", ""))
            for row in poll_payload.get("workers", [])
            if row.get("worker")
        )
        next_seed = next_training_seed(
            poll_payload=poll_payload,
            summary_payload=summary_payload,
            min_seed=min_seed,
        )
        skipped["partial_poll"].append(
            {
                "observed_workers": observed,
                "expected_workers": sorted(DEFAULT_WORKER_NAMES),
                "reason": "automatic seed selection requires a full fleet poll or explicit --seed",
            }
        )
        return {
            "planned_count": 0,
            "planned": [],
            "next_seed": next_seed,
            "skipped": {key: value for key, value in skipped.items() if value},
        }
    automatic_seed = int(seed) <= 0
    seed_floor = int(seed) if int(seed) > 0 else next_training_seed(
        poll_payload=poll_payload,
        summary_payload=summary_payload,
        min_seed=min_seed,
    )
    required_features = required_remote_features_for_recipe("dags_midgame_reanalysis")
    consumed_seeds = consumed_training_seeds(
        poll_payload=poll_payload,
        summary_payload=summary_payload,
    )
    reserved_seeds = set(consumed_seeds)
    planned: list[dict[str, Any]] = []
    for row in poll_payload.get("workers", []):
        worker = Worker(str(row.get("worker", "")), str(row.get("zone", "")))
        if not worker.name or not worker.zone or not row.get("ok", True):
            skipped["inactive_worker"].append(
                {
                    "worker": worker.name,
                    "zone": worker.zone,
                    "error": row.get("error"),
                }
            )
            continue
        busy_reason = None
        if int(row.get("running_train_processes", 0) or 0) > 0:
            busy_reason = "training"
        elif int(row.get("running_reanalysis_processes", 0) or 0) > 0:
            busy_reason = "reanalysis"
        elif worker.name in active_grade_workers and not allow_grade_busy_workers:
            busy_reason = "remote_grade"
        elif worker.name in local_claimed_workers:
            busy_reason = "local_controller"
        if busy_reason is not None:
            skipped["busy_worker"].append(
                {
                    "worker": worker.name,
                    "zone": worker.zone,
                    "reason": busy_reason,
                }
            )
            continue
        missing_features = missing_remote_features(
            row,
            required_features=required_features,
            allow_unknown=allow_unknown_remote_features,
        )
        if missing_features:
            skipped["missing_remote_feature"].append(
                {
                    "worker": worker.name,
                    "zone": worker.zone,
                    "missing": missing_features,
                    "repo": row.get("repo"),
                    "trainer_sha1": row.get("trainer_sha1"),
                }
            )
            continue
        planned_seed = planned_training_seed(
            seed_floor=seed_floor,
            worker_name=worker.name,
            automatic_seed=automatic_seed,
            reserved_seeds=reserved_seeds,
        )
        reserved_seeds.add(planned_seed)
        label = training_label(
            seed=planned_seed,
            recipe="dags_midgame_reanalysis",
            worker_name=worker.name,
        )
        command = build_remote_reanalysis_launch_command(
            worker=worker,
            project=project,
            remote_repo=remote_repo,
            label=label,
            champion=champion,
            seed=planned_seed,
            games=games,
            vps_to_win=vps_to_win,
            max_decisions=max_decisions,
            record_after_decisions=record_after_decisions,
            record_window_decisions=record_window_decisions,
            candidate_limit=candidate_limit,
            presearch_candidate_limit=presearch_candidate_limit,
            rollout_decisions=rollout_decisions,
            rollout_samples=rollout_samples,
            root_value_weight=root_value_weight,
            temperature=temperature,
            reanalysis_max_samples=reanalysis_max_samples,
            reanalysis_epochs=reanalysis_epochs,
            reanalysis_value_coef=reanalysis_value_coef,
            reanalysis_score_coef=reanalysis_score_coef,
        )
        planned.append(
            {
                "worker": worker.name,
                "zone": worker.zone,
                "seed": planned_seed,
                "label": label,
                "recipe": "dags_midgame_reanalysis",
                "jsonl": f"runs/self_play/{label}.jsonl",
                "checkpoint": f"runs/self_play/{label}.pt",
                "final_checkpoint": f"runs/self_play/{label}.final.pt",
                "report": f"runs/self_play/{label}.json",
                "command": command,
                "shell": " ".join(shlex.quote(part) for part in command),
            }
        )
        seed_floor = max(seed_floor + 1, planned_seed + 1)
        if len(planned) >= max(0, int(max_launches)):
            break
    return {
        "planned_count": len(planned),
        "planned": planned,
        "next_seed": seed_floor,
        "required_features": list(required_features),
        "skipped": {key: value for key, value in skipped.items() if value},
    }


def plan_remote_opening_evals(
    *,
    poll_payload: dict[str, Any],
    summary_payload: dict[str, Any],
    local_status_payload: dict[str, Any] | None,
    project: str,
    remote_repo: str,
    run_prefix: str,
    output_dir: str,
    log_dir: str,
    games: int,
    seed: int,
    vps_to_win: int,
    max_opening_decisions: int,
    candidate_limit: int,
    presearch_candidate_limit: int,
    rollout_decisions: int,
    rollout_samples: int,
    root_value_weight: float,
    opponent_penalty: float,
    max_evals: int,
    max_per_family: int,
    include_interim: bool,
    include_warmup: bool,
    prefer_prefixes: list[str],
    min_run_number: int,
    allow_busy_workers: bool,
    allow_unknown_remote_features: bool,
) -> dict[str, Any]:
    active = list(summary_payload.get("active", []))
    active_checkpoints = {
        Path(str(row.get("checkpoint", ""))).name
        for row in active
        if row.get("checkpoint")
    }
    active_families = {
        checkpoint_family_name(name, run_prefix=run_prefix)
        for name in active_checkpoints
    }
    busy_workers = {
        str(row.get("worker"))
        for row in active
        if row.get("worker")
    } | claimed_workers_from_local_status(local_status_payload)
    busy_workers.update(
        str(row.get("worker"))
        for row in poll_payload.get("workers", [])
        if int(row.get("running_train_processes", 0) or 0) > 0 and row.get("worker")
    )
    decided_checkpoints = {
        Path(str(row.get("checkpoint", ""))).name
        for row in summary_payload.get("decisions", [])
        if row.get("checkpoint")
    }
    decided_terminal_families = {
        checkpoint_family_name(name, run_prefix=run_prefix)
        for name in decided_checkpoints
        if checkpoint_iteration(name) >= 1_000_000_000
    }
    skipped: dict[str, list[dict[str, Any]]] = {
        "older_snapshot": [],
        "active_family": [],
        "active_checkpoint": [],
        "decided_checkpoint": [],
        "decided_family": [],
        "busy_worker": [],
        "missing_remote_feature": [],
        "filtered": [],
    }
    required_features = required_remote_features_for_recipe("opening_eval")
    effective_min_run_number = normalize_min_run_number(
        min_run_number,
        run_prefix=run_prefix,
    )
    max_per_family = max(int(max_per_family), 1)
    by_family: dict[str, list[GateCandidate]] = {}
    for worker_row in poll_payload.get("workers", []):
        if not worker_row.get("ok", True):
            continue
        worker = Worker(
            name=str(worker_row.get("worker", "")),
            zone=str(worker_row.get("zone", "")),
        )
        if not worker.name or not worker.zone:
            continue
        missing_features = missing_remote_features(
            worker_row,
            required_features=required_features,
            allow_unknown=allow_unknown_remote_features,
        )
        if missing_features:
            skipped["missing_remote_feature"].append(
                {
                    "worker": worker.name,
                    "zone": worker.zone,
                    "missing": missing_features,
                    "repo": worker_row.get("repo"),
                    "trainer_sha1": worker_row.get("trainer_sha1"),
                }
            )
            continue
        checkpoint_rows = (
            worker_row.get("files", [])
            if include_warmup
            else worker_row.get("candidate_checkpoints", [])
        )
        for file_row in checkpoint_rows:
            name = str(file_row.get("name", ""))
            if not is_candidate_checkpoint_name(
                name,
                include_interim=include_interim,
                include_warmup=include_warmup,
            ):
                skipped["filtered"].append({"worker": worker.name, "checkpoint": name})
                continue
            run_number = checkpoint_run_number(name, run_prefix=run_prefix)
            if run_number is None:
                skipped["filtered"].append(
                    {
                        "worker": worker.name,
                        "checkpoint": name,
                        "reason": "run_prefix_mismatch",
                    }
                )
                continue
            if run_number < effective_min_run_number:
                skipped["filtered"].append(
                    {
                        "worker": worker.name,
                        "checkpoint": name,
                        "reason": f"run_number<{effective_min_run_number}",
                    }
                )
                continue
            candidate = GateCandidate(
                worker=worker,
                checkpoint=name,
                family=checkpoint_family_name(name, run_prefix=run_prefix),
                iteration=checkpoint_iteration(name),
                size=int(file_row.get("size", 0) or 0),
            )
            family_candidates = by_family.setdefault(candidate.family, [])
            family_candidates.append(candidate)
            family_candidates.sort(
                key=lambda item: _gate_candidate_preselect_sort_key(
                    item,
                    run_prefix=run_prefix,
                    busy_workers=busy_workers,
                ),
                reverse=True,
            )
            if len(family_candidates) > max_per_family:
                skipped["older_snapshot"].extend(
                    _candidate_payload(item)
                    for item in family_candidates[max_per_family:]
                )
                del family_candidates[max_per_family:]

    candidates = sorted(
        [candidate for family in by_family.values() for candidate in family],
        key=lambda candidate: _gate_candidate_sort_key(
            candidate,
            run_prefix=run_prefix,
            prefer_prefixes=prefer_prefixes,
        ),
    )
    planned: list[dict[str, Any]] = []
    for candidate in candidates:
        payload = _candidate_payload(candidate)
        if candidate.checkpoint in active_checkpoints:
            skipped["active_checkpoint"].append(payload)
            continue
        if candidate.family in active_families:
            skipped["active_family"].append(payload)
            continue
        if candidate.checkpoint in decided_checkpoints:
            skipped["decided_checkpoint"].append(payload)
            continue
        if candidate.family in decided_terminal_families:
            skipped["decided_family"].append(payload)
            continue
        if not allow_busy_workers and candidate.worker.name in busy_workers:
            skipped["busy_worker"].append(payload)
            continue
        checkpoint_path = f"runs/self_play/{candidate.checkpoint}"
        command = build_remote_opening_eval_launch_command(
            worker=candidate.worker,
            project=project,
            remote_repo=remote_repo,
            checkpoint=checkpoint_path,
            output_dir=output_dir,
            log_dir=log_dir,
            games=games,
            seed=seed + len(planned),
            vps_to_win=vps_to_win,
            max_opening_decisions=max_opening_decisions,
            candidate_limit=candidate_limit,
            presearch_candidate_limit=presearch_candidate_limit,
            rollout_decisions=rollout_decisions,
            rollout_samples=rollout_samples,
            root_value_weight=root_value_weight,
            opponent_penalty=opponent_penalty,
        )
        planned.append(
            {
                **payload,
                "checkpoint_path": checkpoint_path,
                "output": remote_opening_eval_output_path(
                    checkpoint=checkpoint_path,
                    output_dir=output_dir,
                    games=games,
                    seed=seed + len(planned),
                    vps_to_win=vps_to_win,
                    max_opening_decisions=max_opening_decisions,
                ),
                "command": command,
                "shell": " ".join(shlex.quote(part) for part in command),
            }
        )
        if len(planned) >= max(0, int(max_evals)):
            break
    return {
        "planned_count": len(planned),
        "planned": planned,
        "effective_min_run_number": effective_min_run_number,
        "required_features": list(required_features),
        "skipped": {key: value for key, value in skipped.items() if value},
    }


def plan_remote_code_sync(
    *,
    poll_payload: dict[str, Any],
    summary_payload: dict[str, Any],
    local_status_payload: dict[str, Any] | None,
    project: str,
    remote_repo: str,
    recipe: str,
    files: list[str],
    backup_dir: str,
    max_syncs: int,
    allow_grade_busy_workers: bool = False,
) -> dict[str, Any]:
    active_grade_workers = {
        str(row.get("worker"))
        for row in summary_payload.get("active", [])
        if row.get("worker")
    }
    local_claimed_workers = claimed_workers_from_local_status(local_status_payload)
    required_features = required_remote_features_for_recipe(recipe)
    sync_files = normalized_sync_files(files)
    planned: list[dict[str, Any]] = []
    skipped: dict[str, list[dict[str, Any]]] = {
        "busy_worker": [],
        "inactive_worker": [],
        "already_satisfied": [],
    }
    for row in poll_payload.get("workers", []):
        worker = Worker(str(row.get("worker", "")), str(row.get("zone", "")))
        if not worker.name or not worker.zone or not row.get("ok", True):
            skipped["inactive_worker"].append(
                {
                    "worker": worker.name,
                    "zone": worker.zone,
                    "error": row.get("error"),
                }
            )
            continue
        busy_reason = None
        if int(row.get("running_train_processes", 0) or 0) > 0:
            busy_reason = "training"
        elif worker.name in active_grade_workers and not allow_grade_busy_workers:
            busy_reason = "remote_grade"
        elif worker.name in local_claimed_workers:
            busy_reason = "local_controller"
        if busy_reason is not None:
            skipped["busy_worker"].append(
                {
                    "worker": worker.name,
                    "zone": worker.zone,
                    "reason": busy_reason,
                }
            )
            continue
        missing = missing_remote_features(
            row,
            required_features=required_features,
            allow_unknown=False,
        )
        if not missing:
            skipped["already_satisfied"].append(
                {
                    "worker": worker.name,
                    "zone": worker.zone,
                    "repo": row.get("repo"),
                    "trainer_sha1": row.get("trainer_sha1"),
                }
            )
            continue
        target_repo = remote_repo or str(row.get("repo", ""))
        command = build_remote_sync_code_launch_command(
            worker=worker,
            project=project,
            remote_repo=target_repo,
            files=sync_files,
            backup_dir=backup_dir,
        )
        planned.append(
            {
                "worker": worker.name,
                "zone": worker.zone,
                "repo": target_repo or "<auto>",
                "recipe": recipe,
                "missing": missing,
                "files": sync_files,
                "command": command,
                "shell": " ".join(shlex.quote(part) for part in command),
            }
        )
        if len(planned) >= max(0, int(max_syncs)):
            break
    return {
        "planned_count": len(planned),
        "planned": planned,
        "required_features": list(required_features),
        "skipped": {key: value for key, value in skipped.items() if value},
    }


def is_full_default_worker_poll(poll_payload: dict[str, Any]) -> bool:
    observed = {
        str(row.get("worker", ""))
        for row in poll_payload.get("workers", [])
        if row.get("worker")
    }
    return DEFAULT_WORKER_NAMES.issubset(observed)


def required_remote_features_for_recipe(recipe: str) -> tuple[str, ...]:
    if recipe == "opening_eval":
        return ("opening_evaluator",)
    if recipe == "dags_midgame_reanalysis":
        return ("reanalysis_training", "reanalysis_decision_windows")
    if recipe in {"warmup_baseline", "warmup_jsettlers"}:
        return tuple()
    if recipe == "warmup_rollout":
        return ("value_rollout_teacher",)
    features = ["ema_policy_kl", "old_policy_kl"]
    if recipe in {
        "strict_repair_kl",
        "resource_plan_score_repair",
        "rollout_guard_score_repair",
        "tactical_rollout_guard_repair",
        "weighted_dagger_antireg",
        "jsettlers_dagger_antireg",
        "ema_jsettlers_antireg",
        "ema_mixed_antireg",
        "vrpo_esarsa_antireg",
        "vrpo_jsettlers_value_repair",
        "strict_gate_antireg",
        "strict_gate_distill_guard",
    }:
        features.insert(0, "anti_regression_mixed")
    else:
        features.insert(0, "pfsp_mixed")
    if recipe == "pfsp_q_calibration":
        features.append("q_expected_sarsa")
    elif recipe == "pfsp_rollout_teacher":
        features.append("value_rollout_teacher")
    elif recipe not in {
        "pfsp_value_jsettlers",
        "pfsp_klent_control",
        "strict_repair_kl",
        "resource_plan_score_repair",
        "rollout_guard_score_repair",
        "tactical_rollout_guard_repair",
        "weighted_dagger_antireg",
        "jsettlers_dagger_antireg",
        "ema_jsettlers_antireg",
        "ema_mixed_antireg",
        "vrpo_esarsa_antireg",
        "vrpo_jsettlers_value_repair",
        "strict_gate_antireg",
        "strict_gate_distill_guard",
    }:
        raise ValueError(f"unsupported training recipe {recipe!r}")
    if recipe == "resource_plan_score_repair":
        features.append("baseline_score_targets")
    if recipe == "rollout_guard_score_repair":
        features.extend(
            [
                "baseline_score_targets",
                "baseline_rollout_mixed",
                "value_rollout_teacher",
            ]
        )
    if recipe == "tactical_rollout_guard_repair":
        features.extend(
            [
                "baseline_score_targets",
                "tactical_rollout_mixed",
                "value_rollout_teacher",
            ]
        )
    if recipe in {
        "weighted_dagger_antireg",
        "jsettlers_dagger_antireg",
        "ema_jsettlers_antireg",
        "ema_mixed_antireg",
        "vrpo_esarsa_antireg",
        "vrpo_jsettlers_value_repair",
        "strict_gate_antireg",
        "strict_gate_distill_guard",
    }:
        features.extend(
            [
                "baseline_rollout_mixed",
                "q_advantage_gate",
                "q_expected_sarsa",
                "return_weighted_dagger",
                "sample_weighted_imitation",
                "top_advantage_filter",
                "value_rollout_teacher",
            ]
        )
    if recipe == "vrpo_jsettlers_value_repair":
        features.append("jsettlers_value_repair_mixed")
    if recipe in {"strict_gate_antireg", "strict_gate_distill_guard"}:
        features.append("strict_gate_repair_mixed")
    return tuple(features)


def missing_remote_features(
    row: dict[str, Any],
    *,
    required_features: tuple[str, ...],
    allow_unknown: bool,
) -> list[str]:
    features = row.get("trainer_features")
    if features is None:
        return [] if allow_unknown else list(required_features)
    if not isinstance(features, dict):
        return list(required_features)
    return [feature for feature in required_features if not bool(features.get(feature))]


def next_training_seed(
    *,
    poll_payload: dict[str, Any],
    summary_payload: dict[str, Any],
    min_seed: int,
) -> int:
    seeds = [int(min_seed) - 1]
    for row in poll_payload.get("workers", []):
        for process in row.get("processes", []) or []:
            seeds.extend(_seed_numbers_from_text(str(process.get("checkpoint", ""))))
            seed_text = str(process.get("seed", ""))
            if seed_text.isdigit():
                seeds.append(int(seed_text))
        for collection in ("files", "candidate_checkpoints"):
            for file_row in row.get(collection, []) or []:
                seeds.extend(_seed_numbers_from_text(str(file_row.get("name", ""))))
        for log_row in row.get("logs", []) or []:
            seeds.extend(_seed_numbers_from_text(str(log_row.get("name", ""))))
    for row in summary_payload.get("active", []) + summary_payload.get("decisions", []):
        seeds.extend(_seed_numbers_from_text(str(row.get("checkpoint", ""))))
    return max(seeds) + 1


def consumed_training_seeds(
    *,
    poll_payload: dict[str, Any],
    summary_payload: dict[str, Any],
) -> set[int]:
    seeds: set[int] = set()
    for row in poll_payload.get("workers", []):
        for process in row.get("processes", []) or []:
            seeds.update(_seed_numbers_from_text(str(process.get("checkpoint", ""))))
            seed_text = str(process.get("seed", ""))
            if seed_text.isdigit():
                seeds.add(int(seed_text))
        for collection in ("files", "candidate_checkpoints"):
            for file_row in row.get(collection, []) or []:
                seeds.update(_seed_numbers_from_text(str(file_row.get("name", ""))))
        for log_row in row.get("logs", []) or []:
            seeds.update(_seed_numbers_from_text(str(log_row.get("name", ""))))
    for row in summary_payload.get("active", []) + summary_payload.get("decisions", []):
        seeds.update(_seed_numbers_from_text(str(row.get("checkpoint", ""))))
    return seeds


def planned_training_seed(
    *,
    seed_floor: int,
    worker_name: str,
    automatic_seed: bool,
    reserved_seeds: set[int],
) -> int:
    if not automatic_seed:
        candidate = int(seed_floor)
        while candidate in reserved_seeds:
            candidate += 1
        return candidate
    worker_index = DEFAULT_WORKER_INDEX.get(worker_name)
    stride = max(len(DEFAULT_WORKER_INDEX), 10)
    if worker_index is None:
        candidate = int(seed_floor)
        while candidate in reserved_seeds:
            candidate += 1
        return candidate
    base = ((int(seed_floor) + stride - 1) // stride) * stride
    candidate = base + worker_index
    while candidate < int(seed_floor) or candidate in reserved_seeds:
        candidate += stride
    return candidate


def _seed_numbers_from_text(text: str) -> list[int]:
    return [int(match) for match in re.findall(r"(?:^|[/_\s-])s(\d{4,})(?=[_.-])", text)]


def training_label(*, seed: int, recipe: str, worker_name: str) -> str:
    worker_suffix = worker_name.removeprefix("catan-zero-").replace("-", "_")
    return _safe_run_id(f"s{seed}_{recipe}_{worker_suffix}")


def build_planned_training_args(
    *,
    label: str,
    seed: int,
    champion: str,
    recipe: str,
    iterations: int,
    episodes_per_iteration: int,
    checkpoint_every: int,
) -> list[str]:
    if recipe in {"warmup_baseline", "warmup_jsettlers", "warmup_rollout"}:
        return build_warmup_only_training_args(
            label=label,
            seed=seed,
            champion=champion,
            recipe=recipe,
        )
    args = [
        "--seed",
        str(seed),
        "--vps-to-win",
        "4",
        "--max-decisions",
        "300",
        "--init-checkpoint",
        champion,
        "--opponent-checkpoints",
        champion,
        "--teacher",
        "value",
        "--warmup-games",
        "12",
        "--warmup-epochs",
        "2",
        "--warmup-value-coef",
        "0.5",
        "--anchor-value-coef",
        "0.0",
        "--iterations",
        str(iterations),
        "--episodes-per-iteration",
        str(episodes_per_iteration),
        "--learner-seats",
        "one",
        "--opponents",
        "pfsp_mixed",
        "--training-value-candidate-limit",
        "48",
        "--training-value-opponent-penalty",
        "0.05",
        "--ppo-epochs",
        "4",
        "--minibatch-size",
        "256",
        "--learning-rate",
        "0.0002",
        "--clip-ratio",
        "0.15",
        "--value-coef",
        "0.5",
        "--q-value-coef",
        "0.0",
        "--q-advantage-mix",
        "0.0",
        "--q-expected-sarsa-mix",
        "0.0",
        "--entropy-coef",
        "0.012",
        "--old-policy-kl-coef",
        "0.02",
        "--ema-policy-kl-coef",
        "0.01",
        "--ema-policy-decay",
        "0.97",
        "--target-kl",
        "0.03",
        "--anchor-games-per-iteration",
        "2",
        "--dagger-games-per-iteration",
        "1",
        "--anchor-replay-size",
        "4096",
        "--anchor-epochs",
        "1",
        "--anchor-learning-rate-multiplier",
        "0.2",
        "--checkpoint-every",
        str(checkpoint_every),
        "--checkpoint-eval-games",
        "0",
        "--checkpoint-eval-value-games",
        "0",
        "--eval-games",
        "0",
        "--eval-value-games",
        "0",
        "--checkpoint",
        f"runs/self_play/{label}.pt",
        "--report",
        f"runs/self_play/{label}.json",
    ]
    if recipe == "pfsp_rollout_teacher":
        _replace_arg_value(args, "--teacher", "value_rollout")
        args.extend(
            [
                "--teacher-candidate-limit",
                "24",
                "--teacher-presearch-candidate-limit",
                "48",
                "--teacher-rollout-decisions",
                "3",
                "--teacher-rollout-samples",
                "1",
                "--teacher-root-value-weight",
                "0.2",
            ]
        )
    elif recipe == "pfsp_q_calibration":
        _replace_arg_value(args, "--q-value-coef", "0.15")
        _replace_arg_value(args, "--q-expected-sarsa-mix", "0.25")
        args.extend(["--q-advantage-warmup-iterations", str(iterations + 1)])
    elif recipe == "pfsp_klent_control":
        _replace_arg_value(args, "--warmup-games", "8")
        _replace_arg_value(args, "--warmup-epochs", "1")
        _replace_arg_value(args, "--entropy-coef", "0.02")
        _replace_arg_value(args, "--old-policy-kl-coef", "0.03")
        _replace_arg_value(args, "--ema-policy-kl-coef", "0.015")
        _replace_arg_value(args, "--ema-policy-decay", "0.985")
        _replace_arg_value(args, "--target-kl", "0.02")
        args.extend(["--gae-lambda", "0.90"])
    elif recipe == "strict_repair_kl":
        _replace_arg_value(args, "--teacher", "baseline_mixed")
        _replace_arg_value(args, "--warmup-games", "10")
        _replace_arg_value(args, "--warmup-epochs", "1")
        _replace_arg_value(args, "--opponents", "anti_regression_mixed")
        _replace_arg_value(args, "--training-value-opponent-penalty", "0.08")
        _replace_arg_value(args, "--entropy-coef", "0.018")
        _replace_arg_value(args, "--old-policy-kl-coef", "0.025")
        _replace_arg_value(args, "--ema-policy-kl-coef", "0.0125")
        _replace_arg_value(args, "--ema-policy-decay", "0.98")
        _replace_arg_value(args, "--target-kl", "0.022")
        _replace_arg_value(args, "--anchor-games-per-iteration", "3")
        _replace_arg_value(args, "--dagger-games-per-iteration", "2")
        args.extend(
            [
                "--teacher-temperature",
                "0.40",
                "--imitation-score-coef",
                "0.04",
                "--imitation-hard-target-weight",
                "0.20",
                "--gae-lambda",
                "0.90",
            ]
        )
    elif recipe == "resource_plan_score_repair":
        _replace_arg_value(args, "--teacher", "baseline_mixed")
        _replace_arg_value(args, "--warmup-games", "12")
        _replace_arg_value(args, "--warmup-epochs", "1")
        _replace_arg_value(args, "--opponents", "anti_regression_mixed")
        _replace_arg_value(args, "--training-value-opponent-penalty", "0.10")
        _replace_arg_value(args, "--entropy-coef", "0.016")
        _replace_arg_value(args, "--old-policy-kl-coef", "0.03")
        _replace_arg_value(args, "--ema-policy-kl-coef", "0.015")
        _replace_arg_value(args, "--ema-policy-decay", "0.985")
        _replace_arg_value(args, "--target-kl", "0.018")
        _replace_arg_value(args, "--anchor-games-per-iteration", "4")
        _replace_arg_value(args, "--dagger-games-per-iteration", "2")
        args.extend(
            [
                "--teacher-temperature",
                "0.45",
                "--imitation-score-coef",
                "0.08",
                "--imitation-hard-target-weight",
                "0.12",
                "--gae-lambda",
                "0.90",
            ]
        )
    elif recipe == "rollout_guard_score_repair":
        _replace_arg_value(args, "--teacher", "baseline_rollout_mixed")
        _replace_arg_value(args, "--warmup-games", "10")
        _replace_arg_value(args, "--warmup-epochs", "1")
        _replace_arg_value(args, "--opponents", "anti_regression_mixed")
        _replace_arg_value(args, "--training-value-opponent-penalty", "0.10")
        _replace_arg_value(args, "--entropy-coef", "0.014")
        _replace_arg_value(args, "--old-policy-kl-coef", "0.032")
        _replace_arg_value(args, "--ema-policy-kl-coef", "0.016")
        _replace_arg_value(args, "--ema-policy-decay", "0.987")
        _replace_arg_value(args, "--target-kl", "0.016")
        _replace_arg_value(args, "--anchor-games-per-iteration", "4")
        _replace_arg_value(args, "--dagger-games-per-iteration", "3")
        args.extend(
            [
                "--teacher-candidate-limit",
                "24",
                "--teacher-presearch-candidate-limit",
                "48",
                "--teacher-rollout-decisions",
                "2",
                "--teacher-rollout-samples",
                "1",
                "--teacher-root-value-weight",
                "0.25",
                "--teacher-temperature",
                "0.45",
                "--imitation-score-coef",
                "0.07",
                "--imitation-hard-target-weight",
                "0.10",
                "--gae-lambda",
                "0.90",
            ]
        )
    elif recipe == "tactical_rollout_guard_repair":
        _replace_arg_value(args, "--teacher", "tactical_rollout_mixed")
        _replace_arg_value(args, "--warmup-games", "10")
        _replace_arg_value(args, "--warmup-epochs", "1")
        _replace_arg_value(args, "--opponents", "anti_regression_mixed")
        _replace_arg_value(args, "--training-value-opponent-penalty", "0.10")
        _replace_arg_value(args, "--entropy-coef", "0.015")
        _replace_arg_value(args, "--old-policy-kl-coef", "0.034")
        _replace_arg_value(args, "--ema-policy-kl-coef", "0.017")
        _replace_arg_value(args, "--ema-policy-decay", "0.988")
        _replace_arg_value(args, "--target-kl", "0.014")
        _replace_arg_value(args, "--anchor-games-per-iteration", "4")
        _replace_arg_value(args, "--dagger-games-per-iteration", "3")
        args.extend(
            [
                "--teacher-candidate-limit",
                "24",
                "--teacher-presearch-candidate-limit",
                "48",
                "--teacher-rollout-decisions",
                "2",
                "--teacher-rollout-samples",
                "1",
                "--teacher-root-value-weight",
                "0.25",
                "--teacher-temperature",
                "0.45",
                "--imitation-score-coef",
                "0.06",
                "--imitation-hard-target-weight",
                "0.16",
                "--gae-lambda",
                "0.90",
            ]
        )
    elif recipe in {
        "weighted_dagger_antireg",
        "jsettlers_dagger_antireg",
        "ema_jsettlers_antireg",
        "ema_mixed_antireg",
        "vrpo_esarsa_antireg",
        "vrpo_jsettlers_value_repair",
        "strict_gate_antireg",
        "strict_gate_distill_guard",
    }:
        _replace_arg_value(args, "--teacher", "baseline_rollout_mixed")
        _replace_arg_value(args, "--warmup-games", "0")
        _replace_arg_value(args, "--warmup-epochs", "0")
        _replace_arg_value(args, "--opponents", "anti_regression_mixed")
        _replace_arg_value(args, "--training-value-candidate-limit", "24")
        _replace_arg_value(args, "--training-value-opponent-penalty", "0.05")
        _replace_arg_value(args, "--ppo-epochs", "2")
        _replace_arg_value(args, "--learning-rate", "0.0001")
        _replace_arg_value(args, "--clip-ratio", "0.12")
        _replace_arg_value(args, "--value-coef", "0.8")
        _replace_arg_value(args, "--q-value-coef", "0.25")
        _replace_arg_value(args, "--q-advantage-mix", "0.05")
        _replace_arg_value(args, "--q-expected-sarsa-mix", "0.25")
        _replace_arg_value(args, "--entropy-coef", "0.015")
        _replace_arg_value(args, "--old-policy-kl-coef", "0.055")
        _replace_arg_value(args, "--ema-policy-kl-coef", "0.03")
        _replace_arg_value(args, "--ema-policy-decay", "0.995")
        _replace_arg_value(args, "--target-kl", "0.012")
        _replace_arg_value(args, "--anchor-games-per-iteration", "2")
        _replace_arg_value(args, "--dagger-games-per-iteration", "2")
        _replace_arg_value(args, "--anchor-replay-size", "1024")
        _replace_arg_value(args, "--anchor-learning-rate-multiplier", "0.5")
        args.extend(
            [
                "--teacher-candidate-limit",
                "24",
                "--teacher-presearch-candidate-limit",
                "48",
                "--teacher-rollout-decisions",
                "2",
                "--teacher-rollout-samples",
                "1",
                "--ppo-top-advantage-fraction",
                "0.4",
                "--ppo-min-advantage-samples",
                "32",
                "--q-advantage-warmup-iterations",
                "2",
                "--q-advantage-ramp-iterations",
                "4",
                "--q-advantage-min-sign-agreement",
                "0.55",
                "--q-advantage-min-return-corr",
                "0.05",
                "--anchor-sample-weight",
                "1.0",
                "--dagger-sample-weight",
                "3.0",
                "--dagger-low-return-multiplier",
                "2.0",
                "--dagger-low-return-threshold",
                "0.0",
            ]
        )
        if recipe == "jsettlers_dagger_antireg":
            _replace_arg_value(args, "--opponents", "jsettlers_lite")
            _replace_arg_value(args, "--learning-rate", "0.00008")
            _replace_arg_value(args, "--clip-ratio", "0.10")
            _replace_arg_value(args, "--entropy-coef", "0.012")
            _replace_arg_value(args, "--old-policy-kl-coef", "0.065")
            _replace_arg_value(args, "--ema-policy-kl-coef", "0.035")
            _replace_arg_value(args, "--target-kl", "0.010")
            _replace_arg_value(args, "--anchor-games-per-iteration", "3")
            _replace_arg_value(args, "--dagger-games-per-iteration", "4")
            _replace_arg_value(args, "--anchor-replay-size", "2048")
            _replace_arg_value(args, "--ppo-top-advantage-fraction", "0.35")
            _replace_arg_value(args, "--dagger-sample-weight", "4.0")
            _replace_arg_value(args, "--dagger-low-return-multiplier", "2.5")
        elif recipe == "ema_jsettlers_antireg":
            _replace_arg_value(args, "--opponents", "jsettlers_lite")
            _replace_arg_value(args, "--learning-rate", "0.00007")
            _replace_arg_value(args, "--clip-ratio", "0.08")
            _replace_arg_value(args, "--entropy-coef", "0.009")
            _replace_arg_value(args, "--old-policy-kl-coef", "0.035")
            _replace_arg_value(args, "--ema-policy-kl-coef", "0.075")
            _replace_arg_value(args, "--ema-policy-decay", "0.997")
            _replace_arg_value(args, "--target-kl", "0.008")
            _replace_arg_value(args, "--anchor-games-per-iteration", "4")
            _replace_arg_value(args, "--dagger-games-per-iteration", "4")
            _replace_arg_value(args, "--anchor-replay-size", "3072")
            _replace_arg_value(args, "--ppo-top-advantage-fraction", "0.30")
            _replace_arg_value(args, "--dagger-sample-weight", "4.0")
            _replace_arg_value(args, "--dagger-low-return-multiplier", "2.5")
        elif recipe == "ema_mixed_antireg":
            _replace_arg_value(args, "--opponents", "anti_regression_mixed")
            _replace_arg_value(args, "--learning-rate", "0.000075")
            _replace_arg_value(args, "--clip-ratio", "0.09")
            _replace_arg_value(args, "--entropy-coef", "0.011")
            _replace_arg_value(args, "--old-policy-kl-coef", "0.045")
            _replace_arg_value(args, "--ema-policy-kl-coef", "0.065")
            _replace_arg_value(args, "--ema-policy-decay", "0.997")
            _replace_arg_value(args, "--target-kl", "0.009")
            _replace_arg_value(args, "--anchor-games-per-iteration", "4")
            _replace_arg_value(args, "--dagger-games-per-iteration", "4")
            _replace_arg_value(args, "--anchor-replay-size", "3072")
            _replace_arg_value(args, "--ppo-top-advantage-fraction", "0.32")
            _replace_arg_value(args, "--dagger-sample-weight", "3.5")
            _replace_arg_value(args, "--dagger-low-return-multiplier", "2.25")
        elif recipe == "vrpo_esarsa_antireg":
            _replace_arg_value(args, "--opponents", "anti_regression_mixed")
            _replace_arg_value(args, "--learning-rate", "0.000065")
            _replace_arg_value(args, "--clip-ratio", "0.08")
            _replace_arg_value(args, "--value-coef", "0.65")
            _replace_arg_value(args, "--q-value-coef", "0.45")
            _replace_arg_value(args, "--q-advantage-mix", "0.10")
            _replace_arg_value(args, "--q-expected-sarsa-mix", "0.55")
            _replace_arg_value(args, "--entropy-coef", "0.010")
            _replace_arg_value(args, "--old-policy-kl-coef", "0.040")
            _replace_arg_value(args, "--ema-policy-kl-coef", "0.080")
            _replace_arg_value(args, "--ema-policy-decay", "0.998")
            _replace_arg_value(args, "--target-kl", "0.008")
            _replace_arg_value(args, "--anchor-games-per-iteration", "3")
            _replace_arg_value(args, "--dagger-games-per-iteration", "3")
            _replace_arg_value(args, "--anchor-replay-size", "4096")
            _replace_arg_value(args, "--ppo-top-advantage-fraction", "0.45")
            _replace_arg_value(args, "--q-advantage-warmup-iterations", "1")
            _replace_arg_value(args, "--q-advantage-ramp-iterations", "3")
            _replace_arg_value(args, "--q-advantage-min-sign-agreement", "0.58")
            _replace_arg_value(args, "--q-advantage-min-return-corr", "0.08")
            _replace_arg_value(args, "--dagger-sample-weight", "3.0")
            _replace_arg_value(args, "--dagger-low-return-multiplier", "2.0")
        elif recipe == "vrpo_jsettlers_value_repair":
            _replace_arg_value(args, "--opponents", "jsettlers_value_repair_mixed")
            _replace_arg_value(args, "--learning-rate", "0.000055")
            _replace_arg_value(args, "--clip-ratio", "0.07")
            _replace_arg_value(args, "--value-coef", "0.70")
            _replace_arg_value(args, "--q-value-coef", "0.50")
            _replace_arg_value(args, "--q-advantage-mix", "0.08")
            _replace_arg_value(args, "--q-expected-sarsa-mix", "0.60")
            _replace_arg_value(args, "--entropy-coef", "0.009")
            _replace_arg_value(args, "--old-policy-kl-coef", "0.045")
            _replace_arg_value(args, "--ema-policy-kl-coef", "0.090")
            _replace_arg_value(args, "--ema-policy-decay", "0.9985")
            _replace_arg_value(args, "--target-kl", "0.007")
            _replace_arg_value(args, "--anchor-games-per-iteration", "4")
            _replace_arg_value(args, "--dagger-games-per-iteration", "4")
            _replace_arg_value(args, "--anchor-replay-size", "6144")
            _replace_arg_value(args, "--ppo-top-advantage-fraction", "0.40")
            _replace_arg_value(args, "--q-advantage-warmup-iterations", "1")
            _replace_arg_value(args, "--q-advantage-ramp-iterations", "4")
            _replace_arg_value(args, "--q-advantage-min-sign-agreement", "0.60")
            _replace_arg_value(args, "--q-advantage-min-return-corr", "0.10")
            _replace_arg_value(args, "--dagger-sample-weight", "4.0")
            _replace_arg_value(args, "--dagger-low-return-multiplier", "2.75")
        elif recipe == "strict_gate_antireg":
            _replace_arg_value(args, "--teacher", "tactical_rollout_mixed")
            _replace_arg_value(args, "--opponents", "strict_gate_repair_mixed")
            _replace_arg_value(args, "--learning-rate", "0.000052")
            _replace_arg_value(args, "--clip-ratio", "0.065")
            _replace_arg_value(args, "--value-coef", "0.72")
            _replace_arg_value(args, "--q-value-coef", "0.52")
            _replace_arg_value(args, "--q-advantage-mix", "0.07")
            _replace_arg_value(args, "--q-expected-sarsa-mix", "0.62")
            _replace_arg_value(args, "--entropy-coef", "0.008")
            _replace_arg_value(args, "--old-policy-kl-coef", "0.055")
            _replace_arg_value(args, "--ema-policy-kl-coef", "0.095")
            _replace_arg_value(args, "--ema-policy-decay", "0.999")
            _replace_arg_value(args, "--target-kl", "0.006")
            _replace_arg_value(args, "--anchor-games-per-iteration", "5")
            _replace_arg_value(args, "--dagger-games-per-iteration", "5")
            _replace_arg_value(args, "--anchor-replay-size", "8192")
            _replace_arg_value(args, "--ppo-top-advantage-fraction", "0.36")
            _replace_arg_value(args, "--q-advantage-warmup-iterations", "1")
            _replace_arg_value(args, "--q-advantage-ramp-iterations", "4")
            _replace_arg_value(args, "--q-advantage-min-sign-agreement", "0.62")
            _replace_arg_value(args, "--q-advantage-min-return-corr", "0.12")
            _replace_arg_value(args, "--dagger-sample-weight", "4.5")
            _replace_arg_value(args, "--dagger-low-return-multiplier", "3.0")
        elif recipe == "strict_gate_distill_guard":
            # Current candidates often get heuristic gains but fail the
            # value-rollout leg, while pure JSettlers repair is too narrow.
            # This branch refreshes the champion toward tactical rollout
            # targets before a lower-risk strict-gate PPO/DAgger repair.
            _replace_arg_value(args, "--teacher", "tactical_rollout_mixed")
            _replace_arg_value(args, "--warmup-games", "6")
            _replace_arg_value(args, "--warmup-epochs", "1")
            _replace_arg_value(args, "--warmup-value-coef", "0.25")
            _replace_arg_value(args, "--opponents", "strict_gate_repair_mixed")
            _replace_arg_value(args, "--training-value-candidate-limit", "32")
            _replace_arg_value(args, "--training-value-opponent-penalty", "0.07")
            _replace_arg_value(args, "--ppo-epochs", "2")
            _replace_arg_value(args, "--learning-rate", "0.00004")
            _replace_arg_value(args, "--clip-ratio", "0.055")
            _replace_arg_value(args, "--value-coef", "0.75")
            _replace_arg_value(args, "--q-value-coef", "0.55")
            _replace_arg_value(args, "--q-advantage-mix", "0.05")
            _replace_arg_value(args, "--q-expected-sarsa-mix", "0.65")
            _replace_arg_value(args, "--entropy-coef", "0.007")
            _replace_arg_value(args, "--old-policy-kl-coef", "0.075")
            _replace_arg_value(args, "--ema-policy-kl-coef", "0.120")
            _replace_arg_value(args, "--ema-policy-decay", "0.9992")
            _replace_arg_value(args, "--target-kl", "0.0045")
            _replace_arg_value(args, "--anchor-games-per-iteration", "6")
            _replace_arg_value(args, "--dagger-games-per-iteration", "6")
            _replace_arg_value(args, "--anchor-replay-size", "12288")
            _replace_arg_value(args, "--anchor-learning-rate-multiplier", "0.35")
            _replace_arg_value(args, "--ppo-top-advantage-fraction", "0.30")
            _replace_arg_value(args, "--q-advantage-warmup-iterations", "2")
            _replace_arg_value(args, "--q-advantage-ramp-iterations", "5")
            _replace_arg_value(args, "--q-advantage-min-sign-agreement", "0.64")
            _replace_arg_value(args, "--q-advantage-min-return-corr", "0.14")
            _replace_arg_value(args, "--dagger-sample-weight", "5.0")
            _replace_arg_value(args, "--dagger-low-return-multiplier", "3.5")
            args.extend(
                [
                    "--teacher-root-value-weight",
                    "0.35",
                    "--teacher-temperature",
                    "0.38",
                    "--imitation-score-coef",
                    "0.05",
                    "--imitation-hard-target-weight",
                    "0.18",
                    "--gae-lambda",
                    "0.88",
                ]
            )
    elif recipe != "pfsp_value_jsettlers":
        raise ValueError(f"unsupported training recipe {recipe!r}")
    return args


def build_warmup_only_training_args(
    *,
    label: str,
    seed: int,
    champion: str,
    recipe: str,
) -> list[str]:
    if recipe == "warmup_baseline":
        teacher = "baseline_mixed"
        teacher_args = [
            "--teacher-candidate-limit",
            "48",
            "--teacher-temperature",
            "0.40",
            "--warmup-games",
            "48",
            "--warmup-epochs",
            "3",
            "--imitation-score-coef",
            "0.08",
            "--imitation-hard-target-weight",
            "0.35",
        ]
    elif recipe == "warmup_jsettlers":
        teacher = "jsettlers_lite"
        teacher_args = [
            "--teacher-candidate-limit",
            "48",
            "--teacher-temperature",
            "0.35",
            "--warmup-games",
            "48",
            "--warmup-epochs",
            "3",
            "--imitation-score-coef",
            "0.10",
            "--imitation-hard-target-weight",
            "0.45",
        ]
    elif recipe == "warmup_rollout":
        teacher = "value_rollout"
        teacher_args = [
            "--teacher-candidate-limit",
            "24",
            "--teacher-presearch-candidate-limit",
            "48",
            "--teacher-rollout-decisions",
            "3",
            "--teacher-rollout-samples",
            "1",
            "--teacher-root-value-weight",
            "0.35",
            "--teacher-opponent-penalty",
            "0.05",
            "--teacher-temperature",
            "0.50",
            "--warmup-games",
            "32",
            "--warmup-epochs",
            "2",
            "--imitation-score-coef",
            "0.05",
            "--imitation-hard-target-weight",
            "0.25",
        ]
    else:
        raise ValueError(f"unsupported warmup recipe {recipe!r}")
    return [
        "--seed",
        str(seed),
        "--vps-to-win",
        "4",
        "--max-decisions",
        "300",
        "--init-checkpoint",
        champion,
        "--teacher",
        teacher,
        *teacher_args,
        "--warmup-replay-size",
        "4096",
        "--warmup-checkpoint-every",
        "8",
        "--warmup-checkpoint-agreement-games",
        "3",
        "--select-best-warmup-checkpoint",
        "--iterations",
        "0",
        "--checkpoint-every",
        "0",
        "--checkpoint-eval-games",
        "0",
        "--checkpoint-eval-value-games",
        "0",
        "--eval-games",
        "0",
        "--eval-value-games",
        "0",
        "--checkpoint",
        f"runs/self_play/{label}.pt",
        "--report",
        f"runs/self_play/{label}.json",
    ]


def _replace_arg_value(args: list[str], flag: str, value: str) -> None:
    args[args.index(flag) + 1] = value


def build_remote_train_launch_command(
    *,
    worker: Worker,
    project: str,
    remote_repo: str,
    label: str,
    train_args: list[str],
) -> list[str]:
    command = [
        sys.executable,
        "tools/gcp_fleet_controller.py",
        "--project",
        project,
        "--worker",
        f"{worker.name}:{worker.zone}",
    ]
    if remote_repo:
        command.extend(["--remote-repo", remote_repo])
    command.extend(["remote-train", "--label", label, "--", *train_args])
    return command


def build_remote_reanalysis_launch_command(
    *,
    worker: Worker,
    project: str,
    remote_repo: str,
    label: str,
    champion: str,
    seed: int,
    games: int,
    vps_to_win: int,
    max_decisions: int,
    record_after_decisions: int,
    record_window_decisions: int,
    candidate_limit: int,
    presearch_candidate_limit: int,
    rollout_decisions: int,
    rollout_samples: int,
    root_value_weight: float,
    temperature: float,
    reanalysis_max_samples: int,
    reanalysis_epochs: int,
    reanalysis_value_coef: float,
    reanalysis_score_coef: float,
) -> list[str]:
    command = [
        sys.executable,
        "tools/gcp_fleet_controller.py",
        "--project",
        project,
        "--worker",
        f"{worker.name}:{worker.zone}",
    ]
    if remote_repo:
        command.extend(["--remote-repo", remote_repo])
    command.extend(
        [
            "remote-reanalysis-train",
            "--label",
            label,
            "--champion",
            champion,
            "--seed",
            str(seed),
            "--games",
            str(games),
            "--vps-to-win",
            str(vps_to_win),
            "--max-decisions",
            str(max_decisions),
            "--record-after-decisions",
            str(record_after_decisions),
            "--record-window-decisions",
            str(record_window_decisions),
            "--candidate-limit",
            str(candidate_limit),
            "--presearch-candidate-limit",
            str(presearch_candidate_limit),
            "--rollout-decisions",
            str(rollout_decisions),
            "--rollout-samples",
            str(rollout_samples),
            "--root-value-weight",
            str(root_value_weight),
            "--temperature",
            str(temperature),
            "--reanalysis-max-samples",
            str(reanalysis_max_samples),
            "--reanalysis-epochs",
            str(reanalysis_epochs),
            "--reanalysis-value-coef",
            str(reanalysis_value_coef),
            "--reanalysis-score-coef",
            str(reanalysis_score_coef),
        ]
    )
    return command


def build_remote_opening_eval_launch_command(
    *,
    worker: Worker,
    project: str,
    remote_repo: str,
    checkpoint: str,
    output_dir: str,
    log_dir: str,
    games: int,
    seed: int,
    vps_to_win: int,
    max_opening_decisions: int,
    candidate_limit: int,
    presearch_candidate_limit: int,
    rollout_decisions: int,
    rollout_samples: int,
    root_value_weight: float,
    opponent_penalty: float,
) -> list[str]:
    command = [
        sys.executable,
        "tools/gcp_fleet_controller.py",
        "--project",
        project,
        "--worker",
        f"{worker.name}:{worker.zone}",
    ]
    if remote_repo:
        command.extend(["--remote-repo", remote_repo])
    command.extend(
        [
            "remote-opening-eval",
            "--checkpoint",
            checkpoint,
            "--output-dir",
            output_dir,
            "--log-dir",
            log_dir,
            "--games",
            str(games),
            "--seed",
            str(seed),
            "--vps-to-win",
            str(vps_to_win),
            "--max-opening-decisions",
            str(max_opening_decisions),
            "--candidate-limit",
            str(candidate_limit),
            "--presearch-candidate-limit",
            str(presearch_candidate_limit),
            "--rollout-decisions",
            str(rollout_decisions),
            "--rollout-samples",
            str(rollout_samples),
            "--root-value-weight",
            str(root_value_weight),
            "--opponent-penalty",
            str(opponent_penalty),
        ]
    )
    return command


def build_remote_opening_eval_command(args: argparse.Namespace) -> str:
    repo = _remote_repo_shell_expr(args.remote_repo)
    checkpoint = str(args.checkpoint)
    output_path = remote_opening_eval_output_path(
        checkpoint=checkpoint,
        output_dir=str(args.output_dir),
        games=int(args.games),
        seed=int(args.seed),
        vps_to_win=int(args.vps_to_win),
        max_opening_decisions=int(args.max_opening_decisions),
    )
    run_id = Path(output_path).stem
    log_path = f"{args.log_dir}/{run_id}.log"
    eval_command = [
        ".venv/bin/python",
        "-u",
        "tools/evaluate_openings.py",
        "--candidate",
        "ppo",
        "--checkpoint",
        checkpoint,
        "--driver",
        "candidate",
        "--teachers",
        "value",
        "value_rollout",
        "--games",
        str(args.games),
        "--seed",
        str(args.seed),
        "--vps-to-win",
        str(args.vps_to_win),
        "--max-opening-decisions",
        str(args.max_opening_decisions),
        "--candidate-limit",
        str(args.candidate_limit),
        "--presearch-candidate-limit",
        str(args.presearch_candidate_limit),
        "--rollout-decisions",
        str(args.rollout_decisions),
        "--rollout-samples",
        str(args.rollout_samples),
        "--root-value-weight",
        str(args.root_value_weight),
        "--opponent-penalty",
        str(args.opponent_penalty),
        "--output",
        output_path,
    ]
    quoted_eval = " ".join(shlex.quote(part) for part in eval_command)
    existing_payload = json.dumps(
        {
            "checkpoint": checkpoint,
            "output": output_path,
            "reason": "artifact_exists",
            "skipped": True,
        },
        sort_keys=True,
    )
    active_payload = json.dumps(
        {
            "checkpoint": checkpoint,
            "output": output_path,
            "reason": "already_active",
            "skipped": True,
        },
        sort_keys=True,
    )
    preflight = ""
    if not bool(getattr(args, "force", False)):
        preflight = (
            "if [ -s \"$output_path\" ]; then "
            f"printf '%s\\n' {shlex.quote(existing_payload)}; "
            "exit 0; "
            "fi; "
            "active_opening=$(pgrep -af '[t]ools/evaluate_openings.py' "
            "| grep -F -- \"--output $output_path\" "
            "| grep -Ev '(^|[[:space:]])((/bin/)?(ba)?sh)[[:space:]]+-c[[:space:]]' "
            "|| true); "
            "if [ -n \"$active_opening\" ]; then "
            f"printf '%s\\n' {shlex.quote(active_payload)}; "
            "exit 0; "
            "fi; "
        )
    return (
        "set -e; "
        f"repo={repo}; "
        "if [ -z \"$repo\" ]; then "
        "base=\"$HOME/catan-zero\"; nested=\"$base/catan-zero-gcp-bundle\"; "
        "if [ -f \"$nested/pyproject.toml\" ]; then repo=\"$nested\"; else repo=\"$base\"; fi; "
        "fi; "
        "cd \"$repo\"; "
        f"output_path={shlex.quote(output_path)}; "
        f"mkdir -p {shlex.quote(str(args.output_dir))} {shlex.quote(str(args.log_dir))}; "
        f"{preflight}"
        f"setsid sh -c {shlex.quote(quoted_eval + ' >> ' + shlex.quote(log_path) + ' 2>&1 < /dev/null')} "
        "> /dev/null 2>&1 & "
        "pid=$!; "
        f"printf '%s\\n' {shlex.quote(json.dumps({'remote_pid': '__PID__', 'log': log_path, 'checkpoint': checkpoint, 'output': output_path}, sort_keys=True))} | sed \"s/__PID__/$pid/\""
    )


def remote_opening_eval_output_path(
    *,
    checkpoint: str,
    output_dir: str,
    games: int,
    seed: int,
    vps_to_win: int,
    max_opening_decisions: int,
) -> str:
    stem = _safe_run_id(Path(checkpoint).stem)
    run_id = _safe_run_id(
        f"opening_{stem}_g{games}_vp{vps_to_win}_d{max_opening_decisions}_seed{seed}"
    )
    return f"{output_dir.rstrip('/')}/{run_id}.json"


def build_remote_reanalysis_train_command(args: argparse.Namespace) -> str:
    repo = _remote_repo_shell_expr(args.remote_repo)
    label = _safe_run_id(str(args.label))
    if not label:
        raise SystemExit("--label must contain at least one safe character")
    log_dir = str(args.log_dir)
    log_path = f"{log_dir}/{label}.log"
    jsonl_path = f"runs/self_play/{label}.jsonl"
    checkpoint_path = f"runs/self_play/{label}.pt"
    final_checkpoint_path = f"runs/self_play/{label}.final.pt"
    report_path = f"runs/self_play/{label}.json"
    generate_command = [
        ".venv/bin/python",
        "-u",
        "tools/generate_reanalysis.py",
        "--output",
        jsonl_path,
        "--seed",
        str(args.seed),
        "--games",
        str(args.games),
        "--vps-to-win",
        str(args.vps_to_win),
        "--max-decisions",
        str(args.max_decisions),
        "--teacher",
        "value_rollout",
        "--candidate-limit",
        str(args.candidate_limit),
        "--presearch-candidate-limit",
        str(args.presearch_candidate_limit),
        "--rollout-decisions",
        str(args.rollout_decisions),
        "--rollout-samples",
        str(args.rollout_samples),
        "--root-value-weight",
        str(args.root_value_weight),
        "--temperature",
        str(args.temperature),
        "--record-after-decisions",
        str(args.record_after_decisions),
        "--record-window-decisions",
        str(args.record_window_decisions),
    ]
    train_command = [
        ".venv/bin/python",
        "-u",
        "tools/train_ppo.py",
        "--seed",
        str(args.seed),
        "--vps-to-win",
        str(args.vps_to_win),
        "--max-decisions",
        str(args.max_decisions),
        "--init-checkpoint",
        str(args.champion),
        "--teacher",
        "value",
        "--warmup-games",
        "0",
        "--warmup-epochs",
        "0",
        "--iterations",
        "0",
        "--reanalysis-input",
        jsonl_path,
        "--reanalysis-max-samples",
        str(args.reanalysis_max_samples),
        "--reanalysis-epochs",
        str(args.reanalysis_epochs),
        "--reanalysis-value-coef",
        str(args.reanalysis_value_coef),
        "--reanalysis-score-coef",
        str(args.reanalysis_score_coef),
        "--reanalysis-checkpoint",
        checkpoint_path,
        "--checkpoint",
        final_checkpoint_path,
        "--report",
        report_path,
        "--eval-games",
        "0",
        "--eval-value-games",
        "0",
    ]
    quoted_pipeline = (
        " ".join(shlex.quote(part) for part in generate_command)
        + " && "
        + " ".join(shlex.quote(part) for part in train_command)
    )
    busy_payload = json.dumps(
        {
            "label": label,
            "log": log_path,
            "reason": "worker_busy",
            "skipped": True,
        },
        sort_keys=True,
    )
    existing_payload = json.dumps(
        {
            "checkpoint": checkpoint_path,
            "final_checkpoint": final_checkpoint_path,
            "jsonl": jsonl_path,
            "label": label,
            "log": log_path,
            "reason": "artifact_exists",
            "report": report_path,
            "skipped": True,
        },
        sort_keys=True,
    )
    preflight = ""
    if not bool(getattr(args, "force", False)):
        preflight = (
            "if { [ -e \"$checkpoint_path\" ] || [ -e \"$final_checkpoint_path\" ] || "
            "[ -e \"$report_path\" ]; }; then "
            f"printf '%s\\n' {shlex.quote(existing_payload)}; "
            "exit 0; "
            "fi; "
            "active_reanalysis=$( "
            "{ pgrep -af '[t]ools/train_ppo.py' || true; "
            "pgrep -af '[t]ools/generate_reanalysis.py' || true; } "
            "| grep -Ev '(^|[[:space:]])((/bin/)?(ba)?sh)[[:space:]]+-c[[:space:]]' "
            "|| true); "
            "if [ -n \"$active_reanalysis\" ]; then "
            f"printf '%s\\n' {shlex.quote(busy_payload)}; "
            "exit 0; "
            "fi; "
        )
    return (
        "set -e; "
        f"repo={repo}; "
        "if [ -z \"$repo\" ]; then "
        "base=\"$HOME/catan-zero\"; nested=\"$base/catan-zero-gcp-bundle\"; "
        "if [ -f \"$nested/pyproject.toml\" ]; then repo=\"$nested\"; else repo=\"$base\"; fi; "
        "fi; "
        "cd \"$repo\"; "
        f"checkpoint_path={shlex.quote(checkpoint_path)}; "
        f"final_checkpoint_path={shlex.quote(final_checkpoint_path)}; "
        f"report_path={shlex.quote(report_path)}; "
        f"mkdir -p {shlex.quote(log_dir)} runs/self_play; "
        f"{preflight}"
        f"setsid sh -c {shlex.quote(quoted_pipeline + ' >> ' + shlex.quote(log_path) + ' 2>&1 < /dev/null')} "
        "> /dev/null 2>&1 & "
        "pid=$!; "
        f"printf '%s\\n' {shlex.quote(json.dumps({'remote_pid': '__PID__', 'log': log_path, 'label': label, 'jsonl': jsonl_path, 'checkpoint': checkpoint_path, 'final_checkpoint': final_checkpoint_path, 'report': report_path}, sort_keys=True))} | sed \"s/__PID__/$pid/\""
    )


def build_remote_stop_train_command(args: argparse.Namespace) -> str:
    return build_remote_stop_process_command(
        remote_repo=str(args.remote_repo),
        match=str(args.match),
        dry_run=bool(getattr(args, "dry_run", False)),
        script="tools/train_ppo.py",
    )


def build_remote_stop_grade_command(args: argparse.Namespace) -> str:
    return build_remote_stop_process_command(
        remote_repo=str(args.remote_repo),
        match=str(args.match),
        dry_run=bool(getattr(args, "dry_run", False)),
        script="tools/grade_agent.py",
    )


def build_remote_stop_process_command(
    *,
    remote_repo: str,
    match: str,
    dry_run: bool,
    script: str,
) -> str:
    match = str(match)
    if not match.strip():
        raise SystemExit("--match must not be empty")
    if script not in {"tools/train_ppo.py", "tools/grade_agent.py"}:
        raise ValueError(f"unsupported remote stop script {script!r}")
    repo = str(remote_repo)
    return f"""python3 - <<'PY'
import json, os, signal, subprocess
configured={repo!r}
match={match!r}
dry_run={dry_run!r}
script={script!r}
if configured:
    repo=os.path.expanduser(configured)
else:
    base=os.path.expanduser('~/catan-zero')
    nested=os.path.join(base, 'catan-zero-gcp-bundle')
    repo=nested if os.path.exists(os.path.join(nested, 'pyproject.toml')) else base
os.chdir(repo)
ps=subprocess.run(['pgrep','-af',script], text=True, stdout=subprocess.PIPE).stdout
killed=[]
matched=[]
for line in ps.splitlines():
    if 'pgrep -af' in line or 'python3 - <<' in line:
        continue
    if 'bash -c' in line or 'sh -c' in line:
        continue
    if match not in line:
        continue
    pid_text=line.split(maxsplit=1)[0]
    if not pid_text.isdigit():
        continue
    pid=int(pid_text)
    matched.append(line[:4000])
    if not dry_run:
        try:
            os.kill(pid, signal.SIGTERM)
            killed.append(pid)
        except ProcessLookupError:
            pass
print(json.dumps({{'repo': repo, 'script': script, 'match': match, 'dry_run': dry_run, 'matched': matched, 'killed': killed}}, sort_keys=True))
PY"""


def remote_grade_run_id(args: argparse.Namespace) -> str:
    payload = {
        "champion": str(args.champion),
        "checkpoint": str(args.checkpoint),
        "games": int(args.games),
        "leg_timeout_seconds": int(args.leg_timeout_seconds),
        "max_decisions": int(args.max_decisions),
        "profile": str(args.profile),
        "repeats": int(args.repeats),
        "vps_to_win": int(args.vps_to_win),
        "workers": int(args.workers),
    }
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:10]
    checkpoint_stem = Path(str(args.checkpoint)).name.removesuffix(".pt")
    champion_stem = Path(str(args.champion)).name.removesuffix(".pt")
    return _safe_run_id(
        "_".join(
            (
                checkpoint_stem,
                str(args.profile),
                f"g{int(args.games)}",
                f"r{int(args.repeats)}",
                f"w{int(args.workers)}",
                f"vp{int(args.vps_to_win)}",
                f"d{int(args.max_decisions)}",
                f"to{int(args.leg_timeout_seconds)}",
                champion_stem,
                digest,
            )
        )
    )


def _safe_run_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    return re.sub(r"_+", "_", safe)


def _remote_repo_shell_expr(remote_repo: str) -> str:
    if not remote_repo:
        return "''"
    return shlex.quote(remote_repo)


def summarize_remote_grade_status(payload: dict[str, Any]) -> dict[str, Any]:
    active: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    legs_by_checkpoint: dict[str, dict[str, Any]] = {}
    for row in payload.get("workers", []):
        worker = str(row.get("worker", ""))
        zone = str(row.get("zone", ""))
        for active_line in row.get("active_grades", []):
            active.append(
                {
                    "worker": worker,
                    "zone": zone,
                    "checkpoint": _checkpoint_from_command(str(active_line)),
                }
            )
        for summary in row.get("summaries", []):
            data = summary.get("data")
            if not isinstance(data, list):
                continue
            for item in data:
                if not isinstance(item, dict):
                    continue
                candidate = item.get("candidate") or {}
                champion = item.get("champion_summary") or {}
                paired_delta = item.get("paired_delta")
                decisions.append(
                    {
                        "worker": worker,
                        "zone": zone,
                        "summary": summary.get("name"),
                        "profile": _summary_profile(str(summary.get("name", ""))),
                        "checkpoint": item.get("checkpoint"),
                        "champion": item.get("champion"),
                        "decision": item.get("decision"),
                        "reason": item.get("reason"),
                        "summary_games": _summary_games(str(summary.get("name", ""))),
                        "candidate_weighted_win_rate": candidate.get("weighted_win_rate"),
                        "champion_weighted_win_rate": champion.get("weighted_win_rate"),
                        "paired_delta": paired_delta if isinstance(paired_delta, dict) else None,
                    }
                )
        for leg in row.get("legs", []):
            name = str(leg.get("name", ""))
            checkpoint = _checkpoint_from_grade_leg_name(name)
            if not checkpoint:
                continue
            bucket = legs_by_checkpoint.setdefault(
                checkpoint,
                {
                    "worker": worker,
                    "checkpoint": checkpoint,
                    "opponents": {},
                },
            )
            opponent = str(leg.get("opponent", "unknown"))
            bucket["opponents"][opponent] = {
                "wins": leg.get("wins"),
                "games": leg.get("games"),
                "win_rate": leg.get("win_rate"),
            }
    decisions.sort(key=lambda item: (str(item.get("decision")), str(item.get("checkpoint"))))
    return {
        "active_count": len(active),
        "active": active,
        "decisions": decisions,
        "rejections": [item for item in decisions if item.get("decision") == "reject"],
        "keepers": [item for item in decisions if item.get("decision") != "reject"],
        "legs": sorted(legs_by_checkpoint.values(), key=lambda item: item["checkpoint"]),
    }


def plan_remote_gates(
    *,
    poll_payload: dict[str, Any],
    summary_payload: dict[str, Any],
    project: str,
    remote_repo: str,
    run_prefix: str,
    champion: str,
    eval_dir: str,
    log_dir: str,
    profile: str,
    games: int,
    repeats: int,
    grade_workers: int,
    vps_to_win: int,
    max_decisions: int,
    leg_timeout_seconds: int,
    max_gates: int,
    include_interim: bool,
    prefer_prefixes: list[str],
    min_run_number: int,
    allow_busy_workers: bool,
    allow_training_busy_workers: bool = False,
    allow_rejected_family_continuation: bool = False,
    max_per_family: int = 1,
    include_warmup: bool = False,
    local_status_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    active = list(summary_payload.get("active", []))
    active_checkpoints = {
        Path(str(row.get("checkpoint", ""))).name
        for row in active
        if row.get("checkpoint")
    }
    active_families = {
        checkpoint_family_name(name, run_prefix=run_prefix)
        for name in active_checkpoints
    }
    busy_workers = {
        str(row.get("worker"))
        for row in active
        if row.get("worker")
    }
    if not allow_training_busy_workers:
        busy_workers.update(
            str(row.get("worker"))
            for row in poll_payload.get("workers", [])
            if int(row.get("running_train_processes", 0) or 0) > 0 and row.get("worker")
        )
    busy_workers.update(claimed_workers_from_local_status(local_status_payload))
    decided_checkpoints = {
        Path(str(row.get("checkpoint", ""))).name
        for row in summary_payload.get("decisions", [])
        if row.get("checkpoint") and decision_matches_profile(row, profile)
    }
    decided_terminal_families = {
        checkpoint_family_name(name, run_prefix=run_prefix)
        for name in decided_checkpoints
        if checkpoint_iteration(name) >= 1_000_000_000
    }
    rejected_regression_iterations: dict[str, int] = {}
    for row in summary_payload.get("decisions", []):
        if (
            not row.get("checkpoint")
            or not decision_matches_profile(row, profile)
            or not is_family_blocking_reject(row)
        ):
            continue
        rejected_name = Path(str(row.get("checkpoint", ""))).name
        family = checkpoint_family_name(rejected_name, run_prefix=run_prefix)
        rejected_regression_iterations[family] = max(
            rejected_regression_iterations.get(family, -1),
            checkpoint_iteration(rejected_name),
        )
    skipped: dict[str, list[dict[str, Any]]] = {
        "older_snapshot": [],
        "active_family": [],
        "active_checkpoint": [],
        "decided_checkpoint": [],
        "decided_family": [],
        "rejected_regression_family": [],
        "busy_worker": [],
        "planned_worker": [],
        "filtered": [],
    }
    max_per_family = max(int(max_per_family), 1)
    effective_min_run_number = normalize_min_run_number(
        min_run_number,
        run_prefix=run_prefix,
    )
    by_family: dict[str, list[GateCandidate]] = {}
    for worker_row in poll_payload.get("workers", []):
        if not worker_row.get("ok", True):
            continue
        worker = Worker(
            name=str(worker_row.get("worker", "")),
            zone=str(worker_row.get("zone", "")),
        )
        if not worker.name or not worker.zone:
            continue
        checkpoint_rows = (
            worker_row.get("files", [])
            if include_warmup
            else worker_row.get("candidate_checkpoints", [])
        )
        for file_row in checkpoint_rows:
            name = str(file_row.get("name", ""))
            if not is_candidate_checkpoint_name(
                name,
                include_interim=include_interim,
                include_warmup=include_warmup,
            ):
                skipped["filtered"].append(
                    {"worker": worker.name, "checkpoint": name}
                )
                continue
            run_number = checkpoint_run_number(name, run_prefix=run_prefix)
            if run_number is None:
                skipped["filtered"].append(
                    {
                        "worker": worker.name,
                        "checkpoint": name,
                        "reason": "run_prefix_mismatch",
                    }
                )
                continue
            if run_number < effective_min_run_number:
                skipped["filtered"].append(
                    {
                        "worker": worker.name,
                        "checkpoint": name,
                        "reason": f"run_number<{effective_min_run_number}",
                    }
                )
                continue
            candidate = GateCandidate(
                worker=worker,
                checkpoint=name,
                family=checkpoint_family_name(name, run_prefix=run_prefix),
                iteration=checkpoint_iteration(name),
                size=int(file_row.get("size", 0) or 0),
            )
            family_candidates = by_family.setdefault(candidate.family, [])
            family_candidates.append(candidate)
            family_candidates.sort(
                key=lambda item: _gate_candidate_preselect_sort_key(
                    item,
                    run_prefix=run_prefix,
                    busy_workers=busy_workers,
                ),
                reverse=True,
            )
            if len(family_candidates) > max_per_family:
                skipped["older_snapshot"].extend(
                    _candidate_payload(item)
                    for item in family_candidates[max_per_family:]
                )
                del family_candidates[max_per_family:]

    candidates = sorted(
        [candidate for family in by_family.values() for candidate in family],
        key=lambda candidate: _gate_candidate_sort_key(
            candidate,
            run_prefix=run_prefix,
            prefer_prefixes=prefer_prefixes,
        ),
    )
    planned: list[dict[str, Any]] = []
    planned_workers: set[str] = set()
    for candidate in candidates:
        payload = _candidate_payload(candidate)
        if candidate.checkpoint in active_checkpoints:
            skipped["active_checkpoint"].append(payload)
            continue
        if candidate.family in active_families:
            skipped["active_family"].append(payload)
            continue
        if candidate.checkpoint in decided_checkpoints:
            skipped["decided_checkpoint"].append(payload)
            continue
        if candidate.family in decided_terminal_families:
            skipped["decided_family"].append(payload)
            continue
        rejected_iteration = rejected_regression_iterations.get(candidate.family)
        if (
            rejected_iteration is not None
            and (
                not allow_rejected_family_continuation
                or candidate.iteration <= rejected_iteration
            )
        ):
            skipped["rejected_regression_family"].append(payload)
            continue
        if not allow_busy_workers and candidate.worker.name in busy_workers:
            skipped["busy_worker"].append(payload)
            continue
        if not allow_busy_workers and candidate.worker.name in planned_workers:
            skipped["planned_worker"].append(payload)
            continue
        checkpoint_path = f"runs/self_play/{candidate.checkpoint}"
        command = build_remote_gate_launch_command(
            worker=candidate.worker,
            project=project,
            remote_repo=remote_repo,
            checkpoint=checkpoint_path,
            champion=champion,
            eval_dir=eval_dir,
            log_dir=log_dir,
            profile=profile,
            games=games,
            repeats=repeats,
            grade_workers=grade_workers,
            vps_to_win=vps_to_win,
            max_decisions=max_decisions,
            leg_timeout_seconds=leg_timeout_seconds,
        )
        planned.append(
            {
                **payload,
                "checkpoint_path": checkpoint_path,
                "command": command,
                "shell": " ".join(shlex.quote(part) for part in command),
            }
        )
        planned_workers.add(candidate.worker.name)
        if len(planned) >= max_gates:
            break
    return {
        "planned_count": len(planned),
        "planned": planned,
        "effective_min_run_number": effective_min_run_number,
        "skipped": {key: value for key, value in skipped.items() if value},
    }


def plan_remote_transfer_gates(
    *,
    poll_payload: dict[str, Any],
    summary_payload: dict[str, Any],
    project: str,
    remote_repo: str,
    run_prefix: str,
    champion: str,
    eval_dir: str,
    log_dir: str,
    profile: str,
    games: int,
    repeats: int,
    grade_workers: int,
    vps_to_win: int,
    max_decisions: int,
    leg_timeout_seconds: int,
    max_gates: int,
    include_interim: bool,
    prefer_prefixes: list[str],
    min_run_number: int,
    allow_busy_target_workers: bool,
    allow_training_busy_target_workers: bool = False,
    allow_rejected_family_continuation: bool = False,
    max_per_family: int = 1,
    include_warmup: bool = False,
    local_status_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    active = list(summary_payload.get("active", []))
    active_checkpoints = {
        Path(str(row.get("checkpoint", ""))).name
        for row in active
        if row.get("checkpoint")
    }
    active_families = {
        checkpoint_family_name(name, run_prefix=run_prefix)
        for name in active_checkpoints
    }
    active_grade_workers = {
        str(row.get("worker"))
        for row in active
        if row.get("worker")
    }
    local_claimed_workers = claimed_workers_from_local_status(local_status_payload)
    decided_checkpoints = {
        Path(str(row.get("checkpoint", ""))).name
        for row in summary_payload.get("decisions", [])
        if row.get("checkpoint") and decision_matches_profile(row, profile)
    }
    decided_terminal_families = {
        checkpoint_family_name(name, run_prefix=run_prefix)
        for name in decided_checkpoints
        if checkpoint_iteration(name) >= 1_000_000_000
    }
    rejected_regression_iterations: dict[str, int] = {}
    for row in summary_payload.get("decisions", []):
        if (
            not row.get("checkpoint")
            or not decision_matches_profile(row, profile)
            or not is_family_blocking_reject(row)
        ):
            continue
        rejected_name = Path(str(row.get("checkpoint", ""))).name
        family = checkpoint_family_name(rejected_name, run_prefix=run_prefix)
        rejected_regression_iterations[family] = max(
            rejected_regression_iterations.get(family, -1),
            checkpoint_iteration(rejected_name),
        )
    skipped: dict[str, list[dict[str, Any]]] = {
        "older_snapshot": [],
        "active_family": [],
        "active_checkpoint": [],
        "decided_checkpoint": [],
        "decided_family": [],
        "rejected_regression_family": [],
        "busy_target_worker": [],
        "planned_target_worker": [],
        "no_target_worker": [],
        "filtered": [],
    }
    effective_min_run_number = normalize_min_run_number(
        min_run_number,
        run_prefix=run_prefix,
    )
    max_per_family = max(int(max_per_family), 1)
    worker_rows = [
        row
        for row in poll_payload.get("workers", [])
        if row.get("ok", True) and row.get("worker") and row.get("zone")
    ]
    workers_by_name = {
        str(row.get("worker")): Worker(
            name=str(row.get("worker")),
            zone=str(row.get("zone")),
        )
        for row in worker_rows
    }
    repos_by_worker = {
        str(row.get("worker")): str(row.get("repo") or "")
        for row in worker_rows
    }
    hard_busy_target_workers = set(active_grade_workers) | local_claimed_workers
    training_busy_target_workers = {
        str(row.get("worker"))
        for row in worker_rows
        if int(row.get("running_train_processes", 0) or 0) > 0
    }
    busy_target_workers = set(hard_busy_target_workers)
    if not allow_training_busy_target_workers:
        busy_target_workers.update(training_busy_target_workers)
    eligible_targets: list[Worker] = []
    for row in worker_rows:
        worker = workers_by_name[str(row.get("worker"))]
        if not allow_busy_target_workers and worker.name in busy_target_workers:
            skipped["busy_target_worker"].append(
                {
                    "worker": worker.name,
                    "zone": worker.zone,
                    "reason": _target_busy_reason(
                        worker.name,
                        row,
                        active_grade_workers=active_grade_workers,
                        local_claimed_workers=local_claimed_workers,
                    ),
                }
            )
            continue
        eligible_targets.append(worker)

    by_family: dict[str, list[GateCandidate]] = {}
    for worker_row in worker_rows:
        worker = workers_by_name[str(worker_row.get("worker"))]
        checkpoint_rows = (
            worker_row.get("files", [])
            if include_warmup
            else worker_row.get("candidate_checkpoints", [])
        )
        for file_row in checkpoint_rows:
            name = str(file_row.get("name", ""))
            if not is_candidate_checkpoint_name(
                name,
                include_interim=include_interim,
                include_warmup=include_warmup,
            ):
                skipped["filtered"].append(
                    {"worker": worker.name, "checkpoint": name}
                )
                continue
            run_number = checkpoint_run_number(name, run_prefix=run_prefix)
            if run_number is None:
                skipped["filtered"].append(
                    {
                        "worker": worker.name,
                        "checkpoint": name,
                        "reason": "run_prefix_mismatch",
                    }
                )
                continue
            if run_number < effective_min_run_number:
                skipped["filtered"].append(
                    {
                        "worker": worker.name,
                        "checkpoint": name,
                        "reason": f"run_number<{effective_min_run_number}",
                    }
                )
                continue
            candidate = GateCandidate(
                worker=worker,
                checkpoint=name,
                family=checkpoint_family_name(name, run_prefix=run_prefix),
                iteration=checkpoint_iteration(name),
                size=int(file_row.get("size", 0) or 0),
            )
            family_candidates = by_family.setdefault(candidate.family, [])
            family_candidates.append(candidate)
            family_candidates.sort(
                key=lambda item: checkpoint_name_sort_key(
                    item.checkpoint,
                    run_prefix=run_prefix,
                ),
                reverse=True,
            )
            if len(family_candidates) > max_per_family:
                skipped["older_snapshot"].extend(
                    _candidate_payload(item)
                    for item in family_candidates[max_per_family:]
                )
                del family_candidates[max_per_family:]

    candidates = sorted(
        [candidate for family in by_family.values() for candidate in family],
        key=lambda candidate: _gate_candidate_sort_key(
            candidate,
            run_prefix=run_prefix,
            prefer_prefixes=prefer_prefixes,
        ),
    )
    planned: list[dict[str, Any]] = []
    planned_targets: set[str] = set()
    for candidate in candidates:
        payload = _candidate_payload(candidate)
        if candidate.checkpoint in active_checkpoints:
            skipped["active_checkpoint"].append(payload)
            continue
        if candidate.family in active_families:
            skipped["active_family"].append(payload)
            continue
        if candidate.checkpoint in decided_checkpoints:
            skipped["decided_checkpoint"].append(payload)
            continue
        if candidate.family in decided_terminal_families:
            skipped["decided_family"].append(payload)
            continue
        rejected_iteration = rejected_regression_iterations.get(candidate.family)
        if (
            rejected_iteration is not None
            and (
                not allow_rejected_family_continuation
                or candidate.iteration <= rejected_iteration
            )
        ):
            skipped["rejected_regression_family"].append(payload)
            continue
        target = next(
            (
                worker
                for worker in eligible_targets
                if worker.name != candidate.worker.name
                and worker.name not in planned_targets
            ),
            None,
        )
        if target is None:
            skipped["no_target_worker"].append(payload)
            continue
        checkpoint_path = f"runs/self_play/{candidate.checkpoint}"
        source_repo = repos_by_worker.get(candidate.worker.name, "")
        target_repo = repos_by_worker.get(target.name, "")
        command = build_remote_transfer_gate_launch_command(
            source=candidate.worker,
            target=target,
            project=project,
            remote_repo=remote_repo,
            source_remote_repo=source_repo,
            target_remote_repo=target_repo,
            checkpoint=checkpoint_path,
            champion=champion,
            eval_dir=eval_dir,
            log_dir=log_dir,
            profile=profile,
            games=games,
            repeats=repeats,
            grade_workers=grade_workers,
            vps_to_win=vps_to_win,
            max_decisions=max_decisions,
            leg_timeout_seconds=leg_timeout_seconds,
        )
        planned.append(
            {
                **payload,
                "checkpoint_path": checkpoint_path,
                "source_worker": candidate.worker.name,
                "source_zone": candidate.worker.zone,
                "target_worker": target.name,
                "target_zone": target.zone,
                "source_remote_repo": source_repo,
                "target_remote_repo": target_repo,
                "command": command,
                "shell": " ".join(shlex.quote(part) for part in command),
            }
        )
        planned_targets.add(target.name)
        if len(planned) >= max_gates:
            break
    return {
        "planned_count": len(planned),
        "planned": planned,
        "effective_min_run_number": effective_min_run_number,
        "eligible_targets": [
            {"worker": worker.name, "zone": worker.zone}
            for worker in eligible_targets
        ],
        "skipped": {key: value for key, value in skipped.items() if value},
    }


def _target_busy_reason(
    worker_name: str,
    row: dict[str, Any],
    *,
    active_grade_workers: set[str],
    local_claimed_workers: set[str],
) -> str:
    if worker_name in active_grade_workers:
        return "remote_grade"
    if worker_name in local_claimed_workers:
        return "local_controller"
    if int(row.get("running_train_processes", 0) or 0) > 0:
        return "training"
    return "busy"


def plan_remote_escalations(
    *,
    summary_payload: dict[str, Any],
    project: str,
    remote_repo: str,
    champion: str,
    eval_dir: str,
    log_dir: str,
    profile: str,
    source_games: int,
    target_games: int,
    repeats: int,
    grade_workers: int,
    vps_to_win: int,
    max_decisions: int,
    leg_timeout_seconds: int,
    max_escalations: int,
    allow_busy_workers: bool,
    local_status_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    active_workers = {
        str(row.get("worker"))
        for row in summary_payload.get("active", [])
        if row.get("worker")
    }
    active_workers.update(claimed_workers_from_local_status(local_status_payload))
    active_checkpoints = {
        str(row.get("checkpoint"))
        for row in summary_payload.get("active", [])
        if row.get("checkpoint")
    }
    decisions = [
        row
        for row in summary_payload.get("decisions", [])
        if isinstance(row, dict) and decision_matches_profile(row, profile)
    ]
    target_seen = {
        (str(row.get("checkpoint")), int(row.get("summary_games") or 0))
        for row in decisions
        if str(row.get("champion") or champion) == champion
    }
    skipped: dict[str, list[dict[str, Any]]] = {
        "not_smoke_promote": [],
        "already_active": [],
        "target_exists": [],
        "busy_worker": [],
    }
    planned: list[dict[str, Any]] = []
    for row in sorted(
        decisions,
        key=lambda item: (
            -float(item.get("candidate_weighted_win_rate") or 0.0),
            str(item.get("checkpoint") or ""),
        ),
    ):
        checkpoint = str(row.get("checkpoint") or "")
        worker_name = str(row.get("worker") or "")
        if (
            row.get("decision") != "promote_candidate"
            or int(row.get("summary_games") or 0) != source_games
            or str(row.get("champion") or champion) != champion
        ):
            skipped["not_smoke_promote"].append(_escalation_payload(row))
            continue
        if checkpoint in active_checkpoints:
            skipped["already_active"].append(_escalation_payload(row))
            continue
        if (checkpoint, target_games) in target_seen:
            skipped["target_exists"].append(_escalation_payload(row))
            continue
        if not allow_busy_workers and worker_name in active_workers:
            skipped["busy_worker"].append(_escalation_payload(row))
            continue
        worker = _worker_from_summary_row(row)
        if worker is None:
            skipped["not_smoke_promote"].append(_escalation_payload(row))
            continue
        command = build_remote_gate_launch_command(
            worker=worker,
            project=project,
            remote_repo=remote_repo,
            checkpoint=checkpoint,
            champion=champion,
            eval_dir=eval_dir,
            log_dir=log_dir,
            profile=profile,
            games=target_games,
            repeats=repeats,
            grade_workers=grade_workers,
            vps_to_win=vps_to_win,
            max_decisions=max_decisions,
            leg_timeout_seconds=leg_timeout_seconds,
        )
        planned.append(
            {
                **_escalation_payload(row),
                "target_games": target_games,
                "command": command,
                "shell": " ".join(shlex.quote(part) for part in command),
            }
        )
        if len(planned) >= max_escalations:
            break
    return {
        "planned_count": len(planned),
        "planned": planned,
        "skipped": {key: value for key, value in skipped.items() if value},
    }


def build_remote_gate_launch_command(
    *,
    worker: Worker,
    project: str,
    remote_repo: str,
    checkpoint: str,
    champion: str,
    eval_dir: str,
    log_dir: str,
    profile: str,
    games: int,
    repeats: int,
    grade_workers: int,
    vps_to_win: int,
    max_decisions: int,
    leg_timeout_seconds: int,
) -> list[str]:
    command = [
        sys.executable,
        "tools/gcp_fleet_controller.py",
        "--project",
        project,
        "--worker",
        f"{worker.name}:{worker.zone}",
    ]
    if remote_repo:
        command.extend(["--remote-repo", remote_repo])
    command.extend(
        [
            "remote-grade",
            "--checkpoint",
            checkpoint,
            "--champion",
            champion,
            "--eval-dir",
            eval_dir,
            "--log-dir",
            log_dir,
            "--profile",
            profile,
            "--games",
            str(games),
            "--repeats",
            str(repeats),
            "--workers",
            str(grade_workers),
            "--vps-to-win",
            str(vps_to_win),
            "--max-decisions",
            str(max_decisions),
            "--leg-timeout-seconds",
            str(leg_timeout_seconds),
        ]
    )
    return command


def build_remote_transfer_gate_launch_command(
    *,
    source: Worker,
    target: Worker,
    project: str,
    remote_repo: str,
    source_remote_repo: str,
    target_remote_repo: str,
    checkpoint: str,
    champion: str,
    eval_dir: str,
    log_dir: str,
    profile: str,
    games: int,
    repeats: int,
    grade_workers: int,
    vps_to_win: int,
    max_decisions: int,
    leg_timeout_seconds: int,
) -> list[str]:
    command = [
        sys.executable,
        "tools/gcp_fleet_controller.py",
        "--project",
        project,
        "--worker",
        f"{target.name}:{target.zone}",
    ]
    if remote_repo:
        command.extend(["--remote-repo", remote_repo])
    command.extend(
        [
            "remote-grade-from-worker",
            "--source-worker",
            f"{source.name}:{source.zone}",
            "--checkpoint",
            checkpoint,
        ]
    )
    if source_remote_repo:
        command.extend(["--source-remote-repo", source_remote_repo])
    if target_remote_repo:
        command.extend(["--target-remote-repo", target_remote_repo])
    command.extend(
        [
            "--champion",
            champion,
            "--eval-dir",
            eval_dir,
            "--log-dir",
            log_dir,
            "--profile",
            profile,
            "--games",
            str(games),
            "--repeats",
            str(repeats),
            "--workers",
            str(grade_workers),
            "--vps-to-win",
            str(vps_to_win),
            "--max-decisions",
            str(max_decisions),
            "--leg-timeout-seconds",
            str(leg_timeout_seconds),
        ]
    )
    return command


def build_remote_grade_from_worker_command(
    args: argparse.Namespace,
    *,
    source: Worker,
    target: Worker,
) -> dict[str, Any]:
    checkpoint = str(args.checkpoint)
    local_name = Path(checkpoint).name
    source_repo = _remote_scp_repo(
        str(getattr(args, "source_remote_repo", "") or args.remote_repo)
    )
    target_repo = _remote_scp_repo(
        str(getattr(args, "target_remote_repo", "") or args.remote_repo)
    )
    source_path = f"{source.name}:{source_repo}/{checkpoint}"
    target_path = f"{target.name}:{target_repo}/{checkpoint}"
    target_grade_repo = str(getattr(args, "target_remote_repo", "") or args.remote_repo)
    return {
        "copy_from_source": [
            "gcloud",
            "compute",
            "scp",
            source_path,
            f"$TMPDIR/{local_name}",
            "--project",
            str(args.project),
            "--zone",
            source.zone,
        ],
        "copy_to_target": [
            "gcloud",
            "compute",
            "scp",
            f"$TMPDIR/{local_name}",
            target_path,
            "--project",
            str(args.project),
            "--zone",
            target.zone,
        ],
        "grade": build_remote_gate_launch_command(
            worker=target,
            project=str(args.project),
            remote_repo=target_grade_repo,
            checkpoint=checkpoint,
            champion=str(args.champion),
            eval_dir=str(args.eval_dir),
            log_dir=str(args.log_dir),
            profile=str(args.profile),
            games=int(args.games),
            repeats=int(args.repeats),
            grade_workers=int(args.workers),
            vps_to_win=int(args.vps_to_win),
            max_decisions=int(args.max_decisions),
            leg_timeout_seconds=int(args.leg_timeout_seconds),
        )
        + (["--force"] if bool(getattr(args, "force", False)) else []),
        "source": f"{source.name}:{source.zone}",
        "target": f"{target.name}:{target.zone}",
    }


def run_remote_grade_from_worker(
    args: argparse.Namespace,
    *,
    source: Worker,
    target: Worker,
) -> None:
    checkpoint = str(args.checkpoint)
    source_repo = _remote_scp_repo(
        str(getattr(args, "source_remote_repo", "") or args.remote_repo)
    )
    target_repo = _remote_scp_repo(
        str(getattr(args, "target_remote_repo", "") or args.remote_repo)
    )
    target_grade_repo = str(getattr(args, "target_remote_repo", "") or args.remote_repo)
    with tempfile.TemporaryDirectory(prefix="catan-zero-grade-") as tmpdir:
        local_path = str(Path(tmpdir) / Path(checkpoint).name)
        copy_from_source = [
            "gcloud",
            "compute",
            "scp",
            f"{source.name}:{source_repo}/{checkpoint}",
            local_path,
            "--project",
            str(args.project),
            "--zone",
            source.zone,
        ]
        copy_to_target = [
            "gcloud",
            "compute",
            "scp",
            local_path,
            f"{target.name}:{target_repo}/{checkpoint}",
            "--project",
            str(args.project),
            "--zone",
            target.zone,
        ]
        grade_command = build_remote_gate_launch_command(
            worker=target,
            project=str(args.project),
            remote_repo=target_grade_repo,
            checkpoint=checkpoint,
            champion=str(args.champion),
            eval_dir=str(args.eval_dir),
            log_dir=str(args.log_dir),
            profile=str(args.profile),
            games=int(args.games),
            repeats=int(args.repeats),
            grade_workers=int(args.workers),
            vps_to_win=int(args.vps_to_win),
            max_decisions=int(args.max_decisions),
            leg_timeout_seconds=int(args.leg_timeout_seconds),
        )
        if bool(getattr(args, "force", False)):
            grade_command.append("--force")
        for command in (copy_from_source, copy_to_target, grade_command):
            run_command_with_retries(command, attempts=3)


def _remote_scp_repo(remote_repo: str) -> str:
    if remote_repo:
        return remote_repo.rstrip("/")
    return "/home/nickita/catan-zero"


def run_command_with_retries(command: list[str], *, attempts: int = 3) -> None:
    last_error: subprocess.CalledProcessError | None = None
    for attempt in range(1, attempts + 1):
        try:
            subprocess.run(command, check=True)
            return
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if attempt >= attempts:
                break
            time.sleep(float(attempt))
    assert last_error is not None
    raise last_error


def _candidate_payload(candidate: GateCandidate) -> dict[str, Any]:
    return {
        "worker": candidate.worker.name,
        "zone": candidate.worker.zone,
        "checkpoint": candidate.checkpoint,
        "family": candidate.family,
        "iteration": candidate.iteration,
        "size": candidate.size,
    }


def _escalation_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "worker": row.get("worker"),
        "zone": row.get("zone"),
        "checkpoint": row.get("checkpoint"),
        "champion": row.get("champion"),
        "summary": row.get("summary"),
        "summary_games": row.get("summary_games"),
        "candidate_weighted_win_rate": row.get("candidate_weighted_win_rate"),
        "champion_weighted_win_rate": row.get("champion_weighted_win_rate"),
        "paired_delta": row.get("paired_delta"),
    }


def _worker_from_summary_row(row: dict[str, Any]) -> Worker | None:
    name = str(row.get("worker") or "")
    zone = str(row.get("zone") or "")
    if not name or not zone:
        return None
    return Worker(name=name, zone=zone)


def _summary_games(name: str) -> int:
    match = re.search(r"_g(\d+)_r\d+_", name)
    return int(match.group(1)) if match else 0


def _summary_profile(name: str) -> str:
    for profile in ("jsettlers_triage", "strict", "search_stress", "dev"):
        if f"_{profile}_" in name:
            return profile
    return "unknown"


def _gate_candidate_sort_key(
    candidate: GateCandidate,
    *,
    run_prefix: str,
    prefer_prefixes: list[str],
) -> tuple[int, int, str, str]:
    priority = len(prefer_prefixes)
    for index, prefix in enumerate(prefer_prefixes):
        if candidate.checkpoint.startswith(prefix):
            priority = index
            break
    iteration, family, name = checkpoint_name_sort_key(
        candidate.checkpoint,
        run_prefix=run_prefix,
    )
    return (priority, -iteration, family, name)


def _gate_candidate_preselect_sort_key(
    candidate: GateCandidate,
    *,
    run_prefix: str,
    busy_workers: set[str],
) -> tuple[int, str, str, int]:
    iteration, family, name = checkpoint_name_sort_key(
        candidate.checkpoint,
        run_prefix=run_prefix,
    )
    idle_score = 0 if candidate.worker.name in busy_workers else 1
    return (iteration, family, name, idle_score)


def _checkpoint_from_command(command: str) -> str | None:
    match = re.search(r"--checkpoint\s+(\S+)", command)
    return match.group(1) if match else None


def _checkpoint_from_grade_leg_name(name: str) -> str | None:
    match = re.match(r"grade_(.+)_vs_[^_]+(?:_[^_]+)*_g\d+_r\d+_s\d+\.json$", name)
    if not match:
        return None
    return match.group(1)


def is_train_process_line(line: str) -> bool:
    if "tools/train_ppo.py" not in line:
        return False
    if "pgrep -af" in line or "python3 - <<'PY'" in line:
        return False
    if re.search(r"(?:^|\s)(?:/bin/)?(?:ba)?sh\s+-c\s", line):
        return False
    return True


def remote_grade_status(
    workers: list[Worker],
    *,
    project: str,
    remote_repo: str,
    eval_dir: str,
    log_dir: str,
) -> dict[str, Any]:
    rows = []
    for worker in workers:
        command = remote_grade_status_command(
            remote_repo=remote_repo,
            eval_dir=eval_dir,
            log_dir=log_dir,
        )
        try:
            result = subprocess.run(
                [
                    "gcloud",
                    "compute",
                    "ssh",
                    worker.name,
                    "--zone",
                    worker.zone,
                    "--project",
                    project,
                    "--quiet",
                    "--command",
                    command,
                ],
                check=True,
                text=True,
                capture_output=True,
                timeout=90,
            )
            payload = json.loads(result.stdout)
            payload.update({"worker": worker.name, "zone": worker.zone, "ok": True})
            rows.append(payload)
        except Exception as exc:  # noqa: BLE001 - status polling must keep going.
            rows.append({"worker": worker.name, "zone": worker.zone, "ok": False, "error": str(exc)})
    return {
        "workers": rows,
        "active_remote_grades": sum(len(row.get("active_grades", [])) for row in rows),
        "completed_summaries": sum(len(row.get("summaries", [])) for row in rows),
        "completed_legs": sum(len(row.get("legs", [])) for row in rows),
    }


def remote_grade_status_command(*, remote_repo: str, eval_dir: str, log_dir: str) -> str:
    repo_expr = repr(remote_repo)
    eval_dir_expr = repr(eval_dir)
    log_dir_expr = repr(log_dir)
    return f"""python3 - <<'PY'
import json, os, pathlib, subprocess
configured={repo_expr}
if configured:
    repo=os.path.expanduser(configured)
else:
    base=os.path.expanduser('~/catan-zero')
    nested=os.path.join(base, 'catan-zero-gcp-bundle')
    repo=nested if os.path.exists(os.path.join(nested, 'pyproject.toml')) else base
os.chdir(repo)
eval_dir=pathlib.Path({eval_dir_expr})
log_dir=pathlib.Path({log_dir_expr})
active=[]
ps=subprocess.run(['pgrep','-af','tools/grade_agent.py'], text=True, stdout=subprocess.PIPE).stdout
for line in ps.splitlines():
    if 'pgrep -af' in line or 'python3 - <<' in line:
        continue
    if 'bash -c' in line or 'sh -c' in line:
        continue
    active.append(line[:4000])
summaries=[]
for path in sorted(eval_dir.glob('summary_*.json')):
    try:
        data=json.loads(path.read_text())
    except Exception as exc:
        data={{'error': str(exc)}}
    summaries.append({{'name': path.name, 'data': data}})
legs=[]
for path in sorted(eval_dir.glob('grade_*.json')):
    try:
        row=json.loads(path.read_text())
    except Exception as exc:
        legs.append({{'name': path.name, 'error': str(exc)}})
        continue
    legs.append({{
        'name': path.name,
        'wins': row.get('wins'),
        'games': row.get('games'),
        'opponent': row.get('opponent'),
        'win_rate': row.get('win_rate'),
    }})
logs=[]
for path in sorted(log_dir.glob('remote_grade*.log')):
    lines=path.read_text(errors='replace').splitlines()
    logs.append({{'name': path.name, 'lines': len(lines), 'tail': lines[-1][:500] if lines else ''}})
print(json.dumps({{
    'repo': repo,
    'active_grades': active,
    'summaries': summaries,
    'legs': legs,
    'logs': logs,
}}, sort_keys=True))
PY"""


def poll_workers(
    workers: list[Worker],
    *,
    project: str,
    remote_repo: str,
    run_prefix: str,
) -> dict[str, Any]:
    rows = []
    for worker in workers:
        rows.append(
            poll_worker(
                worker,
                project=project,
                remote_repo=remote_repo,
                run_prefix=run_prefix,
            )
        )
    return {
        "workers": rows,
        "running_train_processes": sum(int(row.get("running_train_processes", 0)) for row in rows),
        "running_reanalysis_processes": sum(
            int(row.get("running_reanalysis_processes", 0)) for row in rows
        ),
        "running_opening_eval_processes": sum(
            int(row.get("running_opening_eval_processes", 0)) for row in rows
        ),
        "candidate_checkpoints": sum(len(row.get("candidate_checkpoints", [])) for row in rows),
    }


def poll_worker(
    worker: Worker,
    *,
    project: str,
    remote_repo: str,
    run_prefix: str,
) -> dict[str, Any]:
    command = remote_poll_command(remote_repo=remote_repo, run_prefix=run_prefix)
    try:
        result = subprocess.run(
            [
                "gcloud",
                "compute",
                "ssh",
                worker.name,
                "--zone",
                worker.zone,
                "--project",
                project,
                "--quiet",
                "--command",
                command,
            ],
            check=True,
            text=True,
            capture_output=True,
            timeout=90,
        )
        payload = json.loads(result.stdout)
        payload.update({"worker": worker.name, "zone": worker.zone, "ok": True})
        return payload
    except Exception as exc:  # noqa: BLE001 - fleet polling must keep going.
        return {"worker": worker.name, "zone": worker.zone, "ok": False, "error": str(exc)}


def remote_poll_command(*, remote_repo: str, run_prefix: str) -> str:
    repo_expr = repr(remote_repo)
    prefix_expr = repr(run_prefix)
    default_prefix_expr = repr(DEFAULT_RUN_PREFIX)
    return f"""python3 - <<'PY'
import hashlib, json, os, pathlib, re, subprocess
configured={repo_expr}
if configured:
    repo=os.path.expanduser(configured)
else:
    base=os.path.expanduser('~/catan-zero')
    nested=os.path.join(base, 'catan-zero-gcp-bundle')
    repo=nested if os.path.exists(os.path.join(nested, 'pyproject.toml')) else base
os.chdir(repo)
prefixes=tuple(part.strip() for part in {prefix_expr}.split(',') if part.strip())
if not prefixes:
    prefixes=({default_prefix_expr},)
seed_prefixes=tuple(prefix[1:] for prefix in prefixes if prefix.startswith('s'))
trainer_path=pathlib.Path('tools/train_ppo.py')
try:
    trainer_text=trainer_path.read_text(errors='replace')
except Exception:
    trainer_text=''
self_play_path=pathlib.Path('src/catan_zero/rl/self_play.py')
try:
    self_play_text=self_play_path.read_text(errors='replace')
except Exception:
    self_play_text=''
reanalysis_path=pathlib.Path('src/catan_zero/rl/reanalysis.py')
try:
    reanalysis_text=reanalysis_path.read_text(errors='replace')
except Exception:
    reanalysis_text=''
generate_reanalysis_path=pathlib.Path('tools/generate_reanalysis.py')
try:
    generate_reanalysis_text=generate_reanalysis_path.read_text(errors='replace')
except Exception:
    generate_reanalysis_text=''
evaluate_openings_path=pathlib.Path('tools/evaluate_openings.py')
try:
    evaluate_openings_text=evaluate_openings_path.read_text(errors='replace')
except Exception:
    evaluate_openings_text=''
trainer_features={{
    'anti_regression_mixed': 'anti_regression_mixed' in trainer_text,
    'jsettlers_value_repair_mixed': 'jsettlers_value_repair_mixed' in trainer_text,
    'strict_gate_repair_mixed': 'strict_gate_repair_mixed' in trainer_text,
    'baseline_score_targets': (
        'def target_scores(self, env, info, rng)' in trainer_text
        and '_blend_teacher_score_maps' in trainer_text
        and 'class JSettlersLitePolicy' in self_play_text
        and 'def target_scores(' in self_play_text
    ),
    'baseline_rollout_mixed': (
        'BaselineRolloutMixedTeacherPolicy' in trainer_text
        and 'baseline_rollout_mixed' in trainer_text
    ),
    'tactical_rollout_mixed': (
        'TacticalRolloutMixedTeacherPolicy' in trainer_text
        and 'tactical_rollout_mixed' in trainer_text
    ),
    'ema_policy_kl': '--ema-policy-kl-coef' in trainer_text,
    'graph_history_candidate': 'graph_history_candidate' in trainer_text,
    'old_policy_kl': '--old-policy-kl-coef' in trainer_text,
    'opening_evaluator': (
        '--max-opening-decisions' in evaluate_openings_text
        and 'teacher_metrics_by_prompt' in evaluate_openings_text
    ),
    'pfsp_mixed': 'pfsp_mixed' in trainer_text,
    'q_advantage_gate': '--q-advantage-min-sign-agreement' in trainer_text,
    'q_expected_sarsa': '--q-expected-sarsa-mix' in trainer_text,
    'reanalysis_decision_windows': (
        '--record-after-decisions' in generate_reanalysis_text
        and '--record-window-decisions' in generate_reanalysis_text
        and 'record_after_decisions' in self_play_text
        and 'decision_index' in reanalysis_text
    ),
    'reanalysis_training': (
        '--reanalysis-input' in trainer_text
        and '--reanalysis-checkpoint' in trainer_text
    ),
    'sample_weighted_imitation': (
        '--anchor-sample-weight' in trainer_text
        and '--dagger-sample-weight' in trainer_text
        and 'sample_weight' in self_play_text
    ),
    'return_weighted_dagger': (
        '--dagger-low-return-multiplier' in trainer_text
        and '_set_return_weighted_sample_weights' in trainer_text
    ),
    'top_advantage_filter': '--ppo-top-advantage-fraction' in trainer_text,
    'training_efficiency_timing': (
        '_iteration_timing_summary' in trainer_text
        and 'training_wall_seconds' in trainer_text
    ),
    'value_rollout_teacher': 'value_rollout' in trainer_text,
}}
trainer_sha1=hashlib.sha1(trainer_text.encode('utf-8', errors='replace')).hexdigest() if trainer_text else None
def is_candidate_checkpoint_name(name):
    if not name.endswith('.pt'):
        return False
    if name.endswith('.init.pt') or '.warmup' in name:
        return False
    return True
def is_train_process_line(line):
    if 'tools/train_ppo.py' not in line:
        return False
    if 'pgrep -af' in line or "python3 - <<'PY'" in line:
        return False
    if re.search(r'(?:^|\\s)(?:/bin/)?(?:ba)?sh\\s+-c\\s', line):
        return False
    return True
ps=subprocess.run(['pgrep','-af','tools/train_ppo.py'], text=True, stdout=subprocess.PIPE).stdout
processes=[]
for line in ps.splitlines():
    if not is_train_process_line(line):
        continue
    seed=re.search(r'--seed (\\d+)', line)
    checkpoint=re.search(r'--checkpoint ([^ ]+)', line)
    opponents=re.search(r'--opponents ([^ ]+)', line)
    processes.append({{
        'seed': seed.group(1) if seed else None,
        'matches_run_prefix': bool(seed and seed_prefixes and seed.group(1).startswith(seed_prefixes)),
        'checkpoint': checkpoint.group(1) if checkpoint else None,
        'opponents': opponents.group(1) if opponents else None,
    }})
reanalysis_ps=subprocess.run(['pgrep','-af','tools/generate_reanalysis.py'], text=True, stdout=subprocess.PIPE).stdout
reanalysis_processes=[]
for line in reanalysis_ps.splitlines():
    if 'pgrep -af' in line or "python3 - <<'PY'" in line:
        continue
    if re.search(r'(?:^|\\s)(?:/bin/)?(?:ba)?sh\\s+-c\\s', line):
        continue
    reanalysis_processes.append(line[:1000])
opening_eval_ps=subprocess.run(['pgrep','-af','tools/evaluate_openings.py'], text=True, stdout=subprocess.PIPE).stdout
opening_eval_processes=[]
for line in opening_eval_ps.splitlines():
    if 'pgrep -af' in line or "python3 - <<'PY'" in line:
        continue
    if re.search(r'(?:^|\\s)(?:/bin/)?(?:ba)?sh\\s+-c\\s', line):
        continue
    opening_eval_processes.append(line[:1000])
files=[]
seen_files=set()
for prefix in prefixes:
    for path in sorted(pathlib.Path('runs/self_play').glob(prefix + '*')):
        if path.suffix not in ('.pt', '.json') or path.name in seen_files:
            continue
        seen_files.add(path.name)
        files.append({{'name': path.name, 'size': path.stat().st_size}})
logs=[]
seen_logs=set()
for prefix in prefixes:
    for path in sorted(pathlib.Path('runs/self_play/logs').glob(prefix + '*.log')):
        if path.name in seen_logs:
            continue
        seen_logs.add(path.name)
        text=path.read_text(errors='replace').splitlines()
        logs.append({{'name': path.name, 'lines': len(text), 'tail': text[-1][:300] if text else ''}})
print(json.dumps({{
    'repo': repo,
    'prefixes': prefixes,
    'trainer_features': trainer_features,
    'trainer_sha1': trainer_sha1,
    'running_train_processes': len(processes),
    'running_reanalysis_processes': len(reanalysis_processes),
    'running_opening_eval_processes': len(opening_eval_processes),
    'processes': processes,
    'reanalysis_processes': reanalysis_processes,
    'opening_eval_processes': opening_eval_processes,
    'files': files,
    'candidate_checkpoints': [f for f in files if is_candidate_checkpoint_name(f['name'])],
    'logs': logs,
}}, sort_keys=True))
PY"""


def is_candidate_checkpoint_name(
    name: str,
    *,
    include_interim: bool = True,
    include_warmup: bool = False,
) -> bool:
    if not name.endswith(".pt"):
        return False
    if name.endswith(".init.pt"):
        return False
    if ".warmup" in name:
        return include_warmup
    if ".iter" in name:
        return include_interim
    return True


def candidate_stems(
    files: list[dict[str, Any]],
    *,
    include_interim: bool,
    run_prefix: str = DEFAULT_RUN_PREFIX,
    prefer_prefix: tuple[str, ...] = (),
    min_run_number: int = 0,
) -> set[str]:
    effective_min_run_number = normalize_min_run_number(
        min_run_number,
        run_prefix=run_prefix,
    )
    stems = set()
    for row in files:
        name = str(row.get("name", ""))
        if not is_candidate_checkpoint_name(name, include_interim=include_interim):
            continue
        stem = name[:-3]
        if prefer_prefix and not any(stem.startswith(prefix) for prefix in prefer_prefix):
            continue
        run_number = checkpoint_run_number(name, run_prefix=run_prefix)
        if run_number is None:
            continue
        if run_number < effective_min_run_number:
            continue
        stems.add(stem)
    return stems


def pull_ready_artifacts(
    workers: list[Worker],
    *,
    project: str,
    remote_repo: str,
    run_prefix: str,
    output_dir: Path,
    include_interim: bool,
    prefer_prefix: tuple[str, ...],
    min_run_number: int,
    max_artifacts: int,
    dry_run: bool,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pulled: list[str] = []
    remaining = int(max_artifacts)
    for row in poll_workers(
        workers,
        project=project,
        remote_repo=remote_repo,
        run_prefix=run_prefix,
    )["workers"]:
        if not row.get("ok"):
            continue
        worker = Worker(str(row["worker"]), str(row["zone"]))
        repo = str(row.get("repo", remote_repo or "~/catan-zero"))
        stems = candidate_stems(
            list(row.get("files", [])),
            include_interim=include_interim,
            run_prefix=run_prefix,
            prefer_prefix=prefer_prefix,
            min_run_number=min_run_number,
        )
        artifact_names = sorted(
            _artifact_names_for_stems(stems, list(row.get("files", []))),
            key=lambda name: checkpoint_name_sort_key(name, run_prefix=run_prefix),
            reverse=True,
        )
        for remote_name in artifact_names:
            if remaining == 0:
                return pulled
            destination = output_dir / f"{worker.name}_{remote_name}"
            if destination.exists():
                continue
            remote_path = f"{worker.name}:{repo}/runs/self_play/{remote_name}"
            command = [
                "gcloud",
                "compute",
                "scp",
                remote_path,
                str(destination),
                "--zone",
                worker.zone,
                "--project",
                project,
                "--quiet",
            ]
            print(json.dumps({"command": command, "dry_run": dry_run}, sort_keys=True), flush=True)
            if not dry_run:
                subprocess.run(command, check=True)
            pulled.append(str(destination))
            if remaining > 0:
                remaining -= 1
    return pulled


def _artifact_names_for_stems(stems: set[str], files: list[dict[str, Any]]) -> set[str]:
    names = {str(row.get("name", "")) for row in files}
    selected = set()
    for stem in stems:
        for suffix in (".pt", ".json"):
            name = stem + suffix
            if name in names:
                selected.add(name)
    return selected


def select_local_checkpoints(
    input_dir: Path,
    *,
    run_prefix: str,
    include_interim: bool,
    max_checkpoints: int,
    latest_per_run: bool = True,
) -> list[Path]:
    checkpoints = [
        path
        for path in input_dir.glob(f"*{run_prefix}*.pt")
        if is_candidate_checkpoint_name(path.name, include_interim=include_interim)
    ]
    if latest_per_run:
        latest: dict[str, Path] = {}
        for path in checkpoints:
            family = checkpoint_family_name(path.name, run_prefix=run_prefix)
            current = latest.get(family)
            if current is None or checkpoint_sort_key(path, run_prefix=run_prefix) > checkpoint_sort_key(
                current,
                run_prefix=run_prefix,
            ):
                latest[family] = path
        checkpoints = list(latest.values())
    checkpoints.sort(key=lambda path: checkpoint_sort_key(path, run_prefix=run_prefix), reverse=True)
    return checkpoints[:max_checkpoints]


def checkpoint_family_name(name: str, *, run_prefix: str) -> str:
    stem = name.removesuffix(".pt")
    start = stem.find(run_prefix)
    if start >= 0:
        stem = stem[start:]
    stem = re.sub(r"\.iter\d+$", "", stem)
    stem = re.sub(r"\.warmup\d+$", "", stem)
    stem = re.sub(r"\.reanalysis$", "", stem)
    return re.sub(r"\.final$", "", stem)


def checkpoint_run_number(name: str, *, run_prefix: str) -> int | None:
    stem = Path(name).name
    start = stem.find(run_prefix)
    if start < 0:
        return None
    match = re.match(rf"{re.escape(run_prefix)}(\d+)", stem[start:])
    return int(match.group(1)) if match else None


def normalize_min_run_number(min_run_number: int, *, run_prefix: str) -> int:
    if min_run_number <= 0:
        return min_run_number
    prefix_digits = re.search(r"(\d+)$", run_prefix)
    if not prefix_digits:
        return min_run_number
    prefix = prefix_digits.group(1)
    text = str(min_run_number)
    if text.startswith(prefix) and len(text) > len(prefix):
        return int(text[len(prefix) :])
    if len(text) >= len(prefix) + 2:
        prefix_floor = int(prefix + ("0" * (len(text) - len(prefix))))
        if min_run_number < prefix_floor:
            return 0
    return min_run_number


def checkpoint_iteration(name: str) -> int:
    match = re.search(r"\.iter(\d+)\.pt$", name)
    if match:
        return int(match.group(1))
    match = re.search(r"\.warmup(\d+)\.pt$", name)
    if match:
        return int(match.group(1))
    if name.endswith(".final.pt"):
        return 1_000_000_001
    return 1_000_000_000


def checkpoint_name_sort_key(name: str, *, run_prefix: str) -> tuple[int, str, str]:
    return (
        checkpoint_iteration(name),
        checkpoint_family_name(name, run_prefix=run_prefix),
        name,
    )


def checkpoint_sort_key(path: Path, *, run_prefix: str) -> tuple[int, str, float]:
    iteration, family, _name = checkpoint_name_sort_key(path.name, run_prefix=run_prefix)
    return (
        iteration,
        family,
        path.stat().st_mtime,
    )


if __name__ == "__main__":
    main()
