from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_OPPONENT_WEIGHTS = {
    "heuristic": 1.0,
    "value": 1.5,
    "catanatron_ab3": 2.0,
    "catanatron_ab4": 2.0,
    "catanatron_ab5": 2.0,
}
# FIX A7: the "dev"/default profile used to be heuristic+value only, with no AB
# bots at all -- a materially easier game than the gate roster. Align it with
# evaluate_scoreboard.py's default opponent set (minus random/jsettlers_lite/
# search, which stay available via --opponent) so smoke-scale grading and gate
# decisions are exercised against the same bot family.
GRADE_PROFILES = {
    "dev": {
        "opponents": ("heuristic", "value", "catanatron_ab3", "catanatron_ab4", "catanatron_ab5"),
        "weights": {
            "heuristic": 1.0,
            "value": 1.5,
            "catanatron_ab3": 2.0,
            "catanatron_ab4": 2.0,
            "catanatron_ab5": 2.0,
        },
    },
    "jsettlers_triage": {
        "opponents": ("jsettlers_lite", "heuristic"),
        "weights": {"heuristic": 1.0, "jsettlers_lite": 3.0},
    },
    "strict": {
        "opponents": ("heuristic", "jsettlers_lite", "value", "value_rollout"),
        "weights": {
            "heuristic": 1.0,
            "jsettlers_lite": 2.0,
            "value": 1.5,
            "value_rollout": 2.0,
        },
    },
    "search_stress": {
        "opponents": ("value", "value_rollout"),
        "weights": {"value": 1.0, "value_rollout": 3.0},
    },
}
VALUE_ROLLOUT_OPPONENT_CANDIDATE_LIMIT = 24
VALUE_ROLLOUT_OPPONENT_ROLLOUT_DECISIONS = 3
VALUE_ROLLOUT_OPPONENT_VALUE_PENALTY = 0.0

# FIX A7: tools/evaluate_self_play.py's --opponent parser only understands this
# fixed set (no AB-search opponents at all). Anything outside it is routed
# through tools/evaluate_scoreboard.py instead, which already supports the
# full gate roster (catanatron_ab3/4/5, catanatron_search, catanatron_value).
EVALUATE_SELF_PLAY_OPPONENTS = frozenset(
    {"random", "heuristic", "jsettlers_lite", "search", "value_rollout", "value"}
)


@dataclass(frozen=True, slots=True)
class GradeLeg:
    checkpoint: Path
    opponent: str
    seed: int
    games: int
    report_path: Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Grade PPO CatanZero checkpoints against a fixed opponent suite and "
            "champion baseline. This is stricter than a single win-rate eval and "
            "is intended to decide which branches are worth keeping."
        ),
    )
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument(
        "--profile",
        choices=tuple(GRADE_PROFILES),
        default="dev",
        help=(
            "Named opponent/weight preset. Explicit --opponent and "
            "--opponent-weight flags override the relevant profile settings."
        ),
    )
    parser.add_argument(
        "--champion",
        default="runs/self_play/champions/current_best_s9752_iter0002.pt",
    )
    parser.add_argument("--eval-dir", default="runs/self_play/agent_grades")
    parser.add_argument(
        "--opponent",
        action="append",
        choices=(
            "random",
            "heuristic",
            "jsettlers_lite",
            "search",
            "value_rollout",
            "value",
            "catanatron_ab3",
            "catanatron_ab4",
            "catanatron_ab5",
            "catanatron_search",
            "catanatron_value",
        ),
        help=(
            "Opponent suite. Defaults to the 'dev' profile roster (heuristic, "
            "value, catanatron_ab3/4/5). catanatron_ab3/4/5, catanatron_search, "
            "and catanatron_value are routed through tools/evaluate_scoreboard.py "
            "since tools/evaluate_self_play.py does not support AB-search "
            "opponents; every other opponent still runs through "
            "tools/evaluate_self_play.py unchanged."
        ),
    )
    parser.add_argument(
        "--opponent-weight",
        action="append",
        default=[],
        help="Override score weight as opponent=weight, for example value=2.0.",
    )
    parser.add_argument(
        "--opponent-candidate-limit",
        type=int,
        help=(
            "Forwarded to tools/evaluate_self_play.py for opponent policies "
            "that prune legal actions. Defaults to 24 for value_rollout only."
        ),
    )
    parser.add_argument(
        "--opponent-rollout-decisions",
        type=int,
        help=(
            "Forwarded to tools/evaluate_self_play.py for opponent search "
            "policies. Defaults to 3 for value_rollout only."
        ),
    )
    parser.add_argument(
        "--opponent-value-penalty",
        type=float,
        help=(
            "Forwarded to tools/evaluate_self_play.py for value-based "
            "opponents. Defaults to 0.0 for value_rollout only."
        ),
    )
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed-base", type=int, default=91000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--vps-to-win", type=int, default=10)
    parser.add_argument("--max-decisions", type=int, default=300)
    parser.add_argument(
        "--leg-timeout-seconds",
        type=int,
        default=0,
        help=(
            "Kill one opponent leg if it exceeds this wall-clock timeout. "
            "Timed-out legs are written as zero-win reports and cannot promote."
        ),
    )
    parser.add_argument(
        "--summary-output",
        help="Optional JSON file path for the final grade decisions.",
    )
    parser.add_argument(
        "--min-aggregate-delta",
        type=float,
        default=0.0,
        help="Minimum weighted win-rate gain over champion required to keep.",
    )
    parser.add_argument(
        "--max-opponent-regression",
        type=float,
        default=0.0,
        help=(
            "Allowed per-opponent win-rate regression versus champion. Use a "
            "small value such as 0.03 for noisy early experiments."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    profile = GRADE_PROFILES[args.profile]
    opponents = tuple(args.opponent or profile["opponents"])
    weights = _parse_weights(args.opponent_weight, opponents, base=profile["weights"])
    rows = []
    for checkpoint in args.checkpoint:
        rows.append(
            grade_checkpoint(
                checkpoint=Path(checkpoint),
                champion=Path(args.champion),
                eval_dir=Path(args.eval_dir),
                opponents=opponents,
                weights=weights,
                games=args.games,
                repeats=args.repeats,
                seed_base=args.seed_base,
                workers=args.workers,
                vps_to_win=args.vps_to_win,
                max_decisions=args.max_decisions,
                leg_timeout_seconds=args.leg_timeout_seconds,
                opponent_candidate_limit=args.opponent_candidate_limit,
                opponent_rollout_decisions=args.opponent_rollout_decisions,
                opponent_value_penalty=args.opponent_value_penalty,
                min_aggregate_delta=args.min_aggregate_delta,
                max_opponent_regression=args.max_opponent_regression,
                dry_run=args.dry_run,
            )
        )
    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(rows, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(json.dumps(rows, indent=2, sort_keys=True))


def grade_checkpoint(
    *,
    checkpoint: Path,
    champion: Path,
    eval_dir: Path,
    opponents: tuple[str, ...],
    weights: dict[str, float],
    games: int,
    repeats: int,
    seed_base: int,
    workers: int,
    vps_to_win: int,
    max_decisions: int,
    leg_timeout_seconds: int,
    opponent_candidate_limit: int | None = None,
    opponent_rollout_decisions: int | None = None,
    opponent_value_penalty: float | None = None,
    min_aggregate_delta: float,
    max_opponent_regression: float,
    dry_run: bool,
) -> dict[str, Any]:
    eval_dir.mkdir(parents=True, exist_ok=True)
    candidate_reports = _evaluate_suite(
        checkpoint=checkpoint,
        label=checkpoint.stem,
        eval_dir=eval_dir,
        opponents=opponents,
        games=games,
        repeats=repeats,
        seed_base=seed_base,
        workers=workers,
        vps_to_win=vps_to_win,
        max_decisions=max_decisions,
        leg_timeout_seconds=leg_timeout_seconds,
        opponent_candidate_limit=opponent_candidate_limit,
        opponent_rollout_decisions=opponent_rollout_decisions,
        opponent_value_penalty=opponent_value_penalty,
        dry_run=dry_run,
        stop_on_timeout=True,
    )
    if dry_run:
        champion_reports = _evaluate_suite(
            checkpoint=champion,
            label=champion.stem,
            eval_dir=eval_dir,
            opponents=opponents,
            games=games,
            repeats=repeats,
            seed_base=seed_base,
            workers=workers,
            vps_to_win=vps_to_win,
            max_decisions=max_decisions,
            leg_timeout_seconds=leg_timeout_seconds,
            opponent_candidate_limit=opponent_candidate_limit,
            opponent_rollout_decisions=opponent_rollout_decisions,
            opponent_value_penalty=opponent_value_penalty,
            dry_run=dry_run,
        )
        return {
            "checkpoint": str(checkpoint),
            "champion": str(champion),
            "candidate_reports": candidate_reports,
            "champion_reports": champion_reports,
            "decision": "dry_run",
        }

    candidate = summarize_reports(candidate_reports, weights=weights)
    candidate_timeouts = int(candidate.get("timed_out", 0))
    if candidate_timeouts:
        return {
            "checkpoint": str(checkpoint),
            "champion": str(champion),
            "decision": "reject",
            "reason": f"candidate timed out in {candidate_timeouts} grade legs",
            "candidate": candidate,
            "champion_summary": None,
            "paired_delta": None,
            "early_reject": True,
        }
    if float(candidate["weighted_win_rate"]) <= min_aggregate_delta:
        return {
            "checkpoint": str(checkpoint),
            "champion": str(champion),
            "decision": "reject",
            "reason": (
                f"candidate weighted win rate {float(candidate['weighted_win_rate']):.4f} "
                f"below threshold {min_aggregate_delta:.4f}"
            ),
            "candidate": candidate,
            "champion_summary": None,
            "paired_delta": None,
            "early_reject": True,
        }

    champion_reports = _evaluate_suite(
        checkpoint=champion,
        label=champion.stem,
        eval_dir=eval_dir,
        opponents=opponents,
        games=games,
        repeats=repeats,
        seed_base=seed_base,
        workers=workers,
        vps_to_win=vps_to_win,
        max_decisions=max_decisions,
        leg_timeout_seconds=leg_timeout_seconds,
        opponent_candidate_limit=opponent_candidate_limit,
        opponent_rollout_decisions=opponent_rollout_decisions,
        opponent_value_penalty=opponent_value_penalty,
        dry_run=dry_run,
    )
    champion_summary = summarize_reports(champion_reports, weights=weights)
    decision, reason = grade_decision(
        candidate,
        champion_summary,
        min_aggregate_delta=min_aggregate_delta,
        max_opponent_regression=max_opponent_regression,
    )
    return {
        "checkpoint": str(checkpoint),
        "champion": str(champion),
        "decision": decision,
        "reason": reason,
        "candidate": candidate,
        "champion_summary": champion_summary,
        "paired_delta": paired_deltas(candidate, champion_summary),
    }


def summarize_reports(
    reports: dict[str, list[dict[str, Any]]],
    *,
    weights: dict[str, float],
) -> dict[str, Any]:
    by_opponent = {}
    total_weight = 0.0
    weighted_rate = 0.0
    weighted_lower = 0.0
    for opponent, rows in sorted(reports.items()):
        wins = sum(int(row["wins"]) for row in rows)
        games = sum(int(row["games"]) for row in rows)
        timed_out = sum(1 for row in rows if row.get("timed_out"))
        rate = wins / games if games else 0.0
        lower, upper = wilson_interval(wins, games)
        weight = float(weights.get(opponent, 1.0))
        total_weight += weight
        weighted_rate += weight * rate
        weighted_lower += weight * lower
        by_opponent[opponent] = {
            "wins": wins,
            "games": games,
            "win_rate": rate,
            "wilson_lower_95": lower,
            "wilson_upper_95": upper,
            "weight": weight,
            "timed_out": timed_out,
        }
    if total_weight > 0.0:
        weighted_rate /= total_weight
        weighted_lower /= total_weight
    return {
        "weighted_win_rate": weighted_rate,
        "weighted_wilson_lower_95": weighted_lower,
        "opponents": by_opponent,
        "timed_out": sum(row["timed_out"] for row in by_opponent.values()),
    }


def grade_decision(
    candidate: dict[str, Any],
    champion: dict[str, Any],
    *,
    min_aggregate_delta: float,
    max_opponent_regression: float,
) -> tuple[str, str]:
    aggregate_delta = float(candidate["weighted_win_rate"]) - float(
        champion["weighted_win_rate"]
    )
    candidate_timeouts = int(candidate.get("timed_out", 0))
    champion_timeouts = int(champion.get("timed_out", 0))
    if candidate_timeouts:
        return "reject", f"candidate timed out in {candidate_timeouts} grade legs"
    if champion_timeouts:
        return "reject", f"champion timed out in {champion_timeouts} grade legs"
    regressions = []
    for opponent, candidate_row in candidate["opponents"].items():
        champion_row = champion["opponents"].get(opponent)
        if champion_row is None:
            continue
        delta = float(candidate_row["win_rate"]) - float(champion_row["win_rate"])
        if delta < -max_opponent_regression:
            regressions.append(f"{opponent}:{delta:.4f}")
    if regressions:
        return "reject", "opponent regression " + ",".join(regressions)
    if aggregate_delta <= min_aggregate_delta:
        return "reject", f"aggregate delta {aggregate_delta:.4f} below threshold"
    lower_delta = float(candidate["weighted_wilson_lower_95"]) - float(
        champion["weighted_wilson_lower_95"]
    )
    if lower_delta > 0.0:
        return "promote_candidate", f"aggregate +{aggregate_delta:.4f}, lower-bound +{lower_delta:.4f}"
    return "keep_for_training", f"aggregate +{aggregate_delta:.4f}, lower-bound {lower_delta:.4f}"


def paired_deltas(candidate: dict[str, Any], champion: dict[str, Any]) -> dict[str, Any]:
    aggregate_delta = float(candidate["weighted_win_rate"]) - float(
        champion["weighted_win_rate"]
    )
    lower_delta = float(candidate["weighted_wilson_lower_95"]) - float(
        champion["weighted_wilson_lower_95"]
    )
    opponents: dict[str, dict[str, Any]] = {}
    for opponent, candidate_row in sorted((candidate.get("opponents") or {}).items()):
        champion_row = (champion.get("opponents") or {}).get(opponent)
        if champion_row is None:
            continue
        candidate_rate = float(candidate_row.get("win_rate", 0.0))
        champion_rate = float(champion_row.get("win_rate", 0.0))
        candidate_lower = float(candidate_row.get("wilson_lower_95", 0.0))
        champion_lower = float(champion_row.get("wilson_lower_95", 0.0))
        opponents[opponent] = {
            "candidate_games": int(candidate_row.get("games", 0)),
            "candidate_wilson_lower_95": candidate_lower,
            "candidate_win_rate": candidate_rate,
            "candidate_wins": int(candidate_row.get("wins", 0)),
            "champion_games": int(champion_row.get("games", 0)),
            "champion_wilson_lower_95": champion_lower,
            "champion_win_rate": champion_rate,
            "champion_wins": int(champion_row.get("wins", 0)),
            "lower_delta": candidate_lower - champion_lower,
            "win_rate_delta": candidate_rate - champion_rate,
        }
    worst_opponent = None
    if opponents:
        worst_opponent = min(
            opponents,
            key=lambda opponent: float(opponents[opponent]["win_rate_delta"]),
        )
    return {
        "aggregate_delta": aggregate_delta,
        "aggregate_lower_delta": lower_delta,
        "opponents": opponents,
        "worst_opponent": worst_opponent,
    }


def wilson_interval(wins: int, games: int, z: float = 1.96) -> tuple[float, float]:
    if games <= 0:
        return 0.0, 0.0
    p = wins / games
    denom = 1.0 + (z * z / games)
    center = (p + z * z / (2.0 * games)) / denom
    margin = z * math.sqrt((p * (1.0 - p) / games) + (z * z / (4.0 * games * games))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def _evaluate_suite(
    *,
    checkpoint: Path,
    label: str,
    eval_dir: Path,
    opponents: tuple[str, ...],
    games: int,
    repeats: int,
    seed_base: int,
    workers: int,
    vps_to_win: int,
    max_decisions: int,
    leg_timeout_seconds: int,
    opponent_candidate_limit: int | None = None,
    opponent_rollout_decisions: int | None = None,
    opponent_value_penalty: float | None = None,
    dry_run: bool,
    stop_on_timeout: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    reports: dict[str, list[dict[str, Any]]] = {}
    for opponent_index, opponent in enumerate(opponents):
        rows = []
        for repeat in range(repeats):
            seed = seed_base + opponent_index * 1000 + repeat
            output = eval_dir / (
                f"grade_{label}_vs_{opponent}_g{games}_r{repeat}_s{seed}.json"
            )
            if dry_run:
                rows.append({"output": str(output), "dry_run": True})
                continue
            if not output.exists():
                command = build_eval_command(
                    checkpoint=checkpoint,
                    opponent=opponent,
                    games=games,
                    seed=seed,
                    vps_to_win=vps_to_win,
                    max_decisions=max_decisions,
                    workers=workers,
                    output=output,
                    opponent_candidate_limit=opponent_candidate_limit,
                    opponent_rollout_decisions=opponent_rollout_decisions,
                    opponent_value_penalty=opponent_value_penalty,
                )
                try:
                    run(command, timeout_seconds=leg_timeout_seconds)
                except subprocess.TimeoutExpired:
                    _write_timeout_report(
                        output=output,
                        checkpoint=checkpoint,
                        opponent=opponent,
                        games=games,
                        workers=workers,
                        timeout_seconds=leg_timeout_seconds,
                    )
                else:
                    if opponent not in EVALUATE_SELF_PLAY_OPPONENTS:
                        _normalize_scoreboard_report(output)
            row = json.loads(output.read_text(encoding="utf-8"))
            rows.append(row)
            if stop_on_timeout and row.get("timed_out"):
                reports[opponent] = rows
                return reports
        reports[opponent] = rows
    return reports


def build_eval_command(
    *,
    checkpoint: Path,
    opponent: str,
    games: int,
    seed: int,
    vps_to_win: int,
    max_decisions: int,
    workers: int,
    output: Path,
    opponent_candidate_limit: int | None = None,
    opponent_rollout_decisions: int | None = None,
    opponent_value_penalty: float | None = None,
) -> list[str]:
    if opponent not in EVALUATE_SELF_PLAY_OPPONENTS:
        return build_scoreboard_eval_command(
            checkpoint=checkpoint,
            opponent=opponent,
            games=games,
            seed=seed,
            vps_to_win=vps_to_win,
            max_decisions=max_decisions,
            workers=workers,
            output=output,
        )
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
        "--workers",
        str(workers),
        "--output",
        str(output),
    ]
    resolved_candidate_limit, resolved_rollout_decisions, resolved_value_penalty = (
        _resolve_opponent_search_params(
            opponent,
            opponent_candidate_limit=opponent_candidate_limit,
            opponent_rollout_decisions=opponent_rollout_decisions,
            opponent_value_penalty=opponent_value_penalty,
        )
    )
    if resolved_candidate_limit is not None:
        command.extend(["--opponent-candidate-limit", str(resolved_candidate_limit)])
    if resolved_rollout_decisions is not None:
        command.extend(["--opponent-rollout-decisions", str(resolved_rollout_decisions)])
    if resolved_value_penalty is not None:
        command.extend(["--opponent-value-penalty", str(resolved_value_penalty)])
    return command


def build_scoreboard_eval_command(
    *,
    checkpoint: Path,
    opponent: str,
    games: int,
    seed: int,
    vps_to_win: int,
    max_decisions: int,
    workers: int,
    output: Path,
) -> list[str]:
    """Build a tools/evaluate_scoreboard.py invocation for one opponent leg.

    Used for opponents tools/evaluate_self_play.py cannot run at all
    (catanatron_ab3/4/5, catanatron_search, catanatron_value): see
    EVALUATE_SELF_PLAY_OPPONENTS. evaluate_scoreboard.py writes a
    {"results": [...]} report; _normalize_scoreboard_report flattens it to the
    single-leg schema the rest of grade_agent.py expects.
    """
    return [
        sys.executable,
        "tools/evaluate_scoreboard.py",
        "--candidate",
        str(checkpoint),
        "--candidate-kind",
        "checkpoint",
        "--games",
        str(games),
        "--tracks",
        "2p_no_trade",
        "--opponents",
        opponent,
        "--seed",
        str(seed),
        "--vps-to-win",
        str(vps_to_win),
        "--max-decisions",
        str(max_decisions),
        "--workers",
        str(workers),
        "--device",
        "cpu",
        "--out",
        str(output),
    ]


def _normalize_scoreboard_report(output: Path) -> None:
    """Flatten a tools/evaluate_scoreboard.py {"results": [...]} report into
    the single-leg dict shape tools/evaluate_self_play.py writes, in place, so
    downstream code (summarize_reports, compare_scoreboards.py) does not need
    to know which harness produced a given grade leg file."""
    data = json.loads(output.read_text(encoding="utf-8"))
    results = data.get("results")
    if not isinstance(results, list) or not results:
        return
    flat = dict(results[0])
    flat.setdefault("candidate", data.get("candidate"))
    output.write_text(json.dumps(flat, indent=2, sort_keys=True), encoding="utf-8")


def _resolve_opponent_search_params(
    opponent: str,
    *,
    opponent_candidate_limit: int | None,
    opponent_rollout_decisions: int | None,
    opponent_value_penalty: float | None,
) -> tuple[int | None, int | None, float | None]:
    if opponent == "value_rollout":
        return (
            opponent_candidate_limit
            if opponent_candidate_limit is not None
            else VALUE_ROLLOUT_OPPONENT_CANDIDATE_LIMIT,
            opponent_rollout_decisions
            if opponent_rollout_decisions is not None
            else VALUE_ROLLOUT_OPPONENT_ROLLOUT_DECISIONS,
            opponent_value_penalty
            if opponent_value_penalty is not None
            else VALUE_ROLLOUT_OPPONENT_VALUE_PENALTY,
        )
    return opponent_candidate_limit, opponent_rollout_decisions, opponent_value_penalty


def _parse_weights(
    values: list[str],
    opponents: tuple[str, ...],
    *,
    base: dict[str, float] | None = None,
) -> dict[str, float]:
    weights = dict(base or DEFAULT_OPPONENT_WEIGHTS)
    for opponent in opponents:
        weights.setdefault(opponent, 1.0)
    for value in values:
        opponent, sep, weight_text = value.partition("=")
        if sep != "=" or not opponent or not weight_text:
            raise SystemExit(f"invalid --opponent-weight {value!r}")
        weights[opponent] = float(weight_text)
    return weights


def _write_timeout_report(
    *,
    output: Path,
    checkpoint: Path,
    opponent: str,
    games: int,
    workers: int,
    timeout_seconds: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "avg_decisions": 0.0,
                "candidate": "torch_ppo",
                "checkpoint": str(checkpoint),
                "elo_vs_opponent": None,
                "games": games,
                "opponent": opponent,
                "seat_wins": {},
                "timed_out": True,
                "timeout_seconds": timeout_seconds,
                "win_rate": 0.0,
                "wins": 0,
                "workers": workers,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def run(command: list[str], *, timeout_seconds: int = 0) -> subprocess.CompletedProcess:
    print(json.dumps({"command": command}, sort_keys=True), flush=True)
    process = subprocess.Popen(command, start_new_session=True)
    try:
        return_code = process.wait(timeout=timeout_seconds if timeout_seconds > 0 else None)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        raise
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    return subprocess.CompletedProcess(command, return_code)


if __name__ == "__main__":
    main()
