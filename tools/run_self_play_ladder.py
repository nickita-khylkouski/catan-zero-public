from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


CANONICAL_VPS_TO_WIN = 10
MIN_PROMOTION_GAMES_PER_OPPONENT = 50
NONCANONICAL_OVERWRITE_ACK_FLAG = "--allow-noncanonical-champion-overwrite"
NO_CHAMPION_WRITE_FLAG = "--no-champion-write"


@dataclass(frozen=True, slots=True)
class EvalScore:
    random_win_rate: float | None
    heuristic_win_rate: float | None
    value_win_rate: float | None

    def promotion_tuple(self) -> tuple[float, float, float]:
        return (
            self.value_win_rate if self.value_win_rate is not None else -1.0,
            self.heuristic_win_rate if self.heuristic_win_rate is not None else -1.0,
            self.random_win_rate if self.random_win_rate is not None else -1.0,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a no-human-data self-play ladder: teacher warmup, PPO league "
            "training, held-out evaluation, and gated champion promotion."
        )
    )
    parser.add_argument("--run-dir", default="runs/self_play/ladder")
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--champion", default="runs/self_play/champion.pt")
    parser.add_argument("--vps-to-win", type=int, default=3)
    parser.add_argument("--max-decisions", type=int, default=300)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument(
        "--architecture",
        choices=("candidate", "graph_history_candidate"),
        default="candidate",
    )
    parser.add_argument("--warmup-games", type=int, default=32)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--episodes-per-iteration", type=int, default=4)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--eval-games", type=int, default=24)
    parser.add_argument("--eval-value-games", type=int, default=8)
    parser.add_argument("--promotion-eval-games", type=int, default=24)
    parser.add_argument("--promotion-value-games", type=int, default=8)
    parser.add_argument("--min-heuristic-win-rate", type=float, default=0.25)
    parser.add_argument("--min-value-win-rate", type=float, default=0.25)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--checkpoint-eval-games", type=int, default=8)
    parser.add_argument("--checkpoint-eval-value-games", type=int, default=4)
    parser.add_argument(
        "--warmup-checkpoint-every",
        type=int,
        default=8,
        help="Save and gate supervised warmup checkpoints every N games.",
    )
    parser.add_argument(
        "--warmup-checkpoint-eval-games",
        type=int,
        default=8,
        help="Random/heuristic eval games for each warmup checkpoint.",
    )
    parser.add_argument(
        "--warmup-checkpoint-eval-value-games",
        type=int,
        default=4,
        help="Value-bot eval games for each warmup checkpoint.",
    )
    parser.add_argument("--dry-run", action="store_true")
    champion_write_group = parser.add_mutually_exclusive_group()
    champion_write_group.add_argument(
        NO_CHAMPION_WRITE_FLAG,
        action="store_true",
        help=(
            "Run training and evaluation without creating or replacing the champion. "
            "Use this for historical 3-VP runs and other diagnostics."
        ),
    )
    champion_write_group.add_argument(
        NONCANONICAL_OVERWRITE_ACK_FLAG,
        action="store_true",
        help=(
            "Explicitly allow a champion write from noncanonical settings. This "
            "preserves legacy experiments but is not a production promotion."
        ),
    )
    parser.add_argument(
        "--extra-train-arg",
        action="append",
        default=[],
        help="Extra argument forwarded to tools/train_ppo.py. Repeat as needed.",
    )
    return parser


def champion_write_safety_issues(args: argparse.Namespace) -> tuple[str, ...]:
    issues: list[str] = []
    if args.vps_to_win != CANONICAL_VPS_TO_WIN:
        issues.append(
            f"--vps-to-win={args.vps_to_win} (canonical value is {CANONICAL_VPS_TO_WIN})"
        )
    if args.promotion_eval_games < MIN_PROMOTION_GAMES_PER_OPPONENT:
        issues.append(
            f"--promotion-eval-games={args.promotion_eval_games} "
            f"(minimum is {MIN_PROMOTION_GAMES_PER_OPPONENT})"
        )
    if args.promotion_value_games < MIN_PROMOTION_GAMES_PER_OPPONENT:
        issues.append(
            f"--promotion-value-games={args.promotion_value_games} "
            f"(minimum is {MIN_PROMOTION_GAMES_PER_OPPONENT})"
        )
    return tuple(issues)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    safety_issues = champion_write_safety_issues(args)
    champion_write_enabled = not args.dry_run and not args.no_champion_write
    if (
        champion_write_enabled
        and safety_issues
        and not args.allow_noncanonical_champion_overwrite
    ):
        issue_summary = "; ".join(safety_issues)
        print(
            "[self-play-ladder] ERROR: refusing a champion write from "
            f"noncanonical promotion settings: {issue_summary}. Use "
            f"{NO_CHAMPION_WRITE_FLAG} for a non-mutating historical/diagnostic "
            f"run, or pass {NONCANONICAL_OVERWRITE_ACK_FLAG} to explicitly "
            "acknowledge the unsafe legacy overwrite path.",
            file=sys.stderr,
            flush=True,
        )
        return 2

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    champion_path = Path(args.champion)
    report: dict[str, Any] = {
        "champion_write_policy": {
            "enabled": champion_write_enabled,
            "noncanonical_settings": list(safety_issues),
            "explicit_noncanonical_overwrite": bool(
                args.allow_noncanonical_champion_overwrite
            ),
        },
        "cycles": [],
    }

    for cycle_index in range(args.cycles):
        cycle_seed = args.seed + cycle_index * 10_000
        cycle_dir = run_dir / f"cycle_{cycle_index + 1:03d}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        candidate_path = cycle_dir / "candidate.pt"
        train_report_path = cycle_dir / "train_report.json"
        command = build_train_command(
            args,
            seed=cycle_seed,
            checkpoint=candidate_path,
            report=train_report_path,
            init_checkpoint=champion_path if champion_path.exists() else None,
        )
        cycle_report: dict[str, Any] = {
            "cycle": cycle_index + 1,
            "seed": cycle_seed,
            "candidate": str(candidate_path),
            "train_command": command,
        }
        if args.dry_run:
            cycle_report["dry_run"] = True
            print(json.dumps({"cycle": cycle_report}, sort_keys=True), flush=True)
            report["cycles"].append(cycle_report)
            continue

        run_command(command)
        candidate_score = evaluate_candidate(
            candidate_path,
            cycle_dir=cycle_dir,
            seed=cycle_seed + 1_000_000,
            vps_to_win=args.vps_to_win,
            max_decisions=args.max_decisions,
            heuristic_games=args.promotion_eval_games,
            value_games=args.promotion_value_games,
        )
        champion_score = None
        if champion_path.exists():
            champion_score = evaluate_candidate(
                champion_path,
                cycle_dir=cycle_dir,
                seed=cycle_seed + 2_000_000,
                vps_to_win=args.vps_to_win,
                max_decisions=args.max_decisions,
                heuristic_games=args.promotion_eval_games,
                value_games=args.promotion_value_games,
                prefix="champion",
            )
        promotion_recommended, reason = should_promote(
            candidate_score,
            champion_score,
            min_heuristic_win_rate=args.min_heuristic_win_rate,
            min_value_win_rate=args.min_value_win_rate,
        )
        promoted = promotion_recommended and champion_write_enabled
        if promoted:
            champion_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate_path, champion_path)
        promotion_reason = reason
        if promotion_recommended and not champion_write_enabled:
            promotion_reason = f"{reason}; champion write disabled"
        cycle_report.update(
            {
                "candidate_score": asdict(candidate_score),
                "champion_score": asdict(champion_score) if champion_score else None,
                "promoted": promoted,
                "promotion_recommended": promotion_recommended,
                "promotion_reason": promotion_reason,
            }
        )
        print(json.dumps({"cycle": cycle_report}, sort_keys=True), flush=True)
        report["cycles"].append(cycle_report)

    report_path = run_dir / "ladder_report.json"
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path)}, sort_keys=True), flush=True)
    return 0


def build_train_command(
    args: argparse.Namespace,
    *,
    seed: int,
    checkpoint: Path,
    report: Path,
    init_checkpoint: Path | None,
) -> list[str]:
    command = [
        sys.executable,
        "tools/train_ppo.py",
        "--architecture",
        str(args.architecture),
        "--teacher",
        "value",
        "--hidden-size",
        str(args.hidden_size),
        "--warmup-games",
        str(args.warmup_games),
        "--warmup-epochs",
        str(args.warmup_epochs),
        "--warmup-replay-size",
        "12000",
        "--warmup-checkpoint-every",
        str(getattr(args, "warmup_checkpoint_every", 8)),
        "--warmup-checkpoint-eval-games",
        str(getattr(args, "warmup_checkpoint_eval_games", 8)),
        "--warmup-checkpoint-eval-value-games",
        str(getattr(args, "warmup_checkpoint_eval_value_games", 4)),
        "--imitation-hard-target-weight",
        "1.0",
        "--iterations",
        str(args.iterations),
        "--episodes-per-iteration",
        str(args.episodes_per_iteration),
        "--opponents",
        "strong_mixed",
        "--learner-seats",
        "one",
        "--ppo-epochs",
        str(args.ppo_epochs),
        "--learning-rate",
        str(args.learning_rate),
        "--entropy-coef",
        "0.02",
        "--gamma",
        "1.0",
        "--gae-lambda",
        "0.95",
        "--value-clip-range",
        "0.2",
        "--anchor-games-per-iteration",
        "2",
        "--anchor-replay-size",
        "12000",
        "--anchor-epochs",
        "1",
        "--league-snapshot-every",
        "4",
        "--league-max-snapshots",
        "4",
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--checkpoint-eval-games",
        str(args.checkpoint_eval_games),
        "--checkpoint-eval-value-games",
        str(args.checkpoint_eval_value_games),
        "--select-best-checkpoint",
        "--select-best-min-value-win-rate",
        str(getattr(args, "min_value_win_rate", 0.25)),
        "--eval-games",
        str(args.eval_games),
        "--eval-value-games",
        str(args.eval_value_games),
        "--vps-to-win",
        str(args.vps_to_win),
        "--max-decisions",
        str(args.max_decisions),
        "--seed",
        str(seed),
        "--checkpoint",
        str(checkpoint),
        "--report",
        str(report),
    ]
    if init_checkpoint is not None:
        command.extend(["--init-checkpoint", str(init_checkpoint)])
        command.extend(["--opponent-checkpoints", str(init_checkpoint)])
        command.extend(["--warmup-games", "0"])
    command.extend(args.extra_train_arg)
    return command


def evaluate_candidate(
    checkpoint: Path,
    *,
    cycle_dir: Path,
    seed: int,
    vps_to_win: int,
    max_decisions: int,
    heuristic_games: int,
    value_games: int,
    prefix: str = "candidate",
) -> EvalScore:
    random_report = run_eval(
        checkpoint,
        opponent="random",
        games=heuristic_games,
        seed=seed,
        vps_to_win=vps_to_win,
        max_decisions=max_decisions,
        output=cycle_dir / f"{prefix}_vs_random.json",
    )
    heuristic_report = run_eval(
        checkpoint,
        opponent="heuristic",
        games=heuristic_games,
        seed=seed + 1,
        vps_to_win=vps_to_win,
        max_decisions=max_decisions,
        output=cycle_dir / f"{prefix}_vs_heuristic.json",
    )
    value_report = None
    if value_games > 0:
        value_report = run_eval(
            checkpoint,
            opponent="value",
            games=value_games,
            seed=seed + 2,
            vps_to_win=vps_to_win,
            max_decisions=max_decisions,
            output=cycle_dir / f"{prefix}_vs_value.json",
        )
    return EvalScore(
        random_win_rate=_win_rate(random_report),
        heuristic_win_rate=_win_rate(heuristic_report),
        value_win_rate=_win_rate(value_report),
    )


def run_eval(
    checkpoint: Path,
    *,
    opponent: str,
    games: int,
    seed: int,
    vps_to_win: int,
    max_decisions: int,
    output: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "tools/evaluate_self_play.py",
        "--candidate",
        "ppo",
        "--checkpoint",
        str(checkpoint),
        "--opponent",
        opponent,
        "--games",
        str(games),
        "--seed",
        str(seed),
        "--vps-to-win",
        str(vps_to_win),
        "--max-decisions",
        str(max_decisions),
        "--output",
        str(output),
    ]
    run_command(command)
    return json.loads(output.read_text(encoding="utf-8"))


def should_promote(
    candidate: EvalScore,
    champion: EvalScore | None,
    *,
    min_heuristic_win_rate: float,
    min_value_win_rate: float,
) -> tuple[bool, str]:
    if (
        candidate.heuristic_win_rate is None
        or candidate.heuristic_win_rate < min_heuristic_win_rate
    ):
        return False, "candidate failed heuristic gate"
    if candidate.value_win_rate is not None and candidate.value_win_rate < min_value_win_rate:
        return False, "candidate failed value gate"
    if champion is None:
        return True, "no existing champion"
    if candidate.promotion_tuple() <= champion.promotion_tuple():
        return False, "candidate did not beat champion score"
    return True, "candidate beat champion score"


def run_command(command: list[str]) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = "src:."
    print(json.dumps({"command": command}, sort_keys=True), flush=True)
    subprocess.run(command, check=True, env=env)


def _win_rate(report: dict[str, Any] | None) -> float | None:
    if not isinstance(report, dict) or "win_rate" not in report:
        return None
    return float(report["win_rate"])


if __name__ == "__main__":
    raise SystemExit(main())
