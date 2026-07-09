from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from summarize_population_payoffs import (
    DEFAULT_LEDGER,
    is_viable_escalation_candidate,
    load_jsonl,
    summarize_payoffs,
)
from summarize_training_efficiency import _summarize_path


ITERATION_SUFFIX_RE = re.compile(r"\.(?:iter|warmup)\d+$")
WORKER_SUFFIX_RE = re.compile(r"_(?:c\d|w\d[a-d])$")
RUN_SEED_RE = re.compile(r"^s\d+_")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a PSRO/league-style experiment scorecard by combining strict "
            "grade results with training efficiency. This answers which recipes "
            "should be scaled, repaired, or profiled next."
        )
    )
    parser.add_argument("--ledger", default=DEFAULT_LEDGER)
    parser.add_argument("--profile", default="strict")
    parser.add_argument("--opponent", default="current_best_s9752_iter0002")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument(
        "--efficiency-glob",
        action="append",
        default=[],
        help="Glob for train_ppo JSON reports or JSONL logs. Can be repeated.",
    )
    parser.add_argument("--output")
    args = parser.parse_args()

    rows = load_jsonl(Path(args.ledger))
    payoffs = summarize_payoffs(
        rows,
        profile=args.profile,
        opponent=args.opponent,
        top=max(args.top, 50),
    )
    efficiency_runs = load_efficiency_runs(args.efficiency_glob)
    efficiency_by_family = summarize_efficiency_by_family(efficiency_runs)
    scored = attach_efficiency(payoffs["top"], efficiency_by_family)
    recipe_rows = summarize_recipes(scored)
    payload = {
        "profile": args.profile,
        "opponent": args.opponent,
        "source": {
            "ledger": args.ledger,
            "efficiency_globs": args.efficiency_glob,
            "efficiency_runs": len(efficiency_runs),
            "timed_efficiency_runs": sum(
                1 for run in efficiency_runs if float(run.get("total_seconds") or 0.0) > 0.0
            ),
        },
        "research_pattern": {
            "league": "rank candidates by strict grade, not latest-checkpoint Elo",
            "psro": "use weak-opponent failures to choose repair oracles/exploiters",
            "throughput": "scale only recipes with measurable quality-per-hour",
            "search_reanalysis": "prefer candidates that improve fixed gates and retain efficiency",
        },
        "best_viable_candidate": next(
            (row for row in scored if row["scorecard"]["viable_escalation_candidate"]),
            None,
        ),
        "top_candidates": scored[: args.top],
        "recipe_scorecard": recipe_rows,
        "fleet_recommendation": recommend_fleet_action(
            scored,
            recipe_rows=recipe_rows,
            payoff_recommendation=payoffs.get("training_recommendation") or {},
            timed_runs=sum(
                1 for run in efficiency_runs if float(run.get("total_seconds") or 0.0) > 0.0
            ),
        ),
        "payoff_summary": {
            "pair_rows": payoffs.get("pair_rows"),
            "decision_counts": payoffs.get("decision_counts"),
            "weak_opponents": payoffs.get("weak_opponents"),
            "failure_modes": payoffs.get("failure_modes"),
            "training_recommendation": payoffs.get("training_recommendation"),
        },
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


def load_efficiency_runs(patterns: list[str]) -> list[dict[str, Any]]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(path) for path in glob.glob(pattern))
    seen: set[Path] = set()
    runs: list[dict[str, Any]] = []
    for path in sorted(paths):
        resolved = path.resolve()
        if resolved in seen or not path.is_file():
            continue
        seen.add(resolved)
        run = _summarize_path(path)
        run["run_id"] = infer_run_id(path)
        run["family_id"] = family_id(str(run["run_id"]))
        runs.append(run)
    return runs


def summarize_efficiency_by_family(runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[str(run.get("family_id") or "")].append(run)
    summaries: dict[str, dict[str, Any]] = {}
    for family, family_runs in grouped.items():
        if not family:
            continue
        timed = [run for run in family_runs if float(run.get("total_seconds") or 0.0) > 0.0]
        total_seconds = sum(float(run.get("total_seconds") or 0.0) for run in family_runs)
        phase_seconds: dict[str, float] = defaultdict(float)
        for run in family_runs:
            for phase, seconds in (run.get("phase_seconds") or {}).items():
                phase_seconds[str(phase)] += float(seconds or 0.0)
        dominant_phase = max(phase_seconds, key=phase_seconds.get) if total_seconds > 0 else None
        summaries[family] = {
            "family_id": family,
            "runs": len(family_runs),
            "timed_runs": len(timed),
            "total_seconds": total_seconds,
            "median_seconds": median(
                [float(run.get("total_seconds") or 0.0) for run in timed]
            )
            if timed
            else None,
            "dominant_phase": dominant_phase,
            "dominant_phase_fraction": (
                float(phase_seconds[dominant_phase]) / total_seconds
                if dominant_phase and total_seconds > 0
                else None
            ),
            "ppo_samples_per_second": _weighted_rate(
                family_runs,
                samples_key="ppo_samples",
                seconds_keys=("collect", "ppo_update"),
            ),
            "anchor_samples_per_second": _weighted_rate(
                family_runs,
                samples_key="anchor_samples",
                seconds_keys=("anchor_collect", "anchor_update"),
            ),
        }
    return summaries


def _weighted_rate(
    runs: list[dict[str, Any]],
    *,
    samples_key: str,
    seconds_keys: tuple[str, ...],
) -> float | None:
    samples = sum(float(run.get(samples_key) or 0.0) for run in runs)
    seconds = 0.0
    for run in runs:
        phase_seconds = run.get("phase_seconds") or {}
        seconds += sum(float(phase_seconds.get(key) or 0.0) for key in seconds_keys)
    if seconds <= 0.0:
        return None
    return samples / seconds


def attach_efficiency(
    policy_rows: list[dict[str, Any]],
    efficiency_by_family: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in policy_rows:
        item = dict(row)
        family = family_id(str(item.get("policy_id") or ""))
        recipe = recipe_id(family)
        efficiency = efficiency_by_family.get(family) or {}
        item["family_id"] = family
        item["recipe_id"] = recipe
        item["efficiency"] = efficiency or None
        item["scorecard"] = score_policy(item, efficiency)
        rows.append(item)
    rows.sort(
        key=lambda item: (
            not item["scorecard"]["viable_escalation_candidate"],
            -float(item["scorecard"].get("delta_per_hour") or -1e9),
            -float(item.get("delta") or 0.0),
            -float(item.get("score") or 0.0),
            str(item.get("policy_id") or ""),
        )
    )
    return rows


def score_policy(row: dict[str, Any], efficiency: dict[str, Any]) -> dict[str, Any]:
    total_seconds = float((efficiency or {}).get("total_seconds") or 0.0)
    delta = float(row.get("delta") or 0.0)
    score = float(row.get("score") or 0.0)
    regressions = row.get("opponent_regressions") or []
    viable = is_viable_escalation_candidate(row)
    delta_per_hour = delta * 3600.0 / total_seconds if total_seconds > 0.0 else None
    score_per_hour = score * 3600.0 / total_seconds if total_seconds > 0.0 else None
    dominant_phase = (efficiency or {}).get("dominant_phase")
    dominant_fraction = (efficiency or {}).get("dominant_phase_fraction")
    if regressions:
        action = "repair_regression"
    elif viable and delta_per_hour is not None:
        action = "scale_if_still_positive"
    elif viable:
        action = "escalate_gate_collect_timing"
    elif delta > 0.0:
        action = "grade_more_or_repair_required_legs"
    elif dominant_phase and dominant_fraction and float(dominant_fraction) >= 0.60:
        action = "profile_bottleneck_before_more_training"
    else:
        action = "reject_or_low_priority"
    return {
        "viable_escalation_candidate": viable,
        "delta_per_hour": delta_per_hour,
        "score_per_hour": score_per_hour,
        "dominant_phase": dominant_phase,
        "dominant_phase_fraction": dominant_fraction,
        "recommended_action": action,
        "regression_count": len(regressions),
    }


def summarize_recipes(scored_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored_rows:
        grouped[str(row.get("recipe_id") or "unknown")].append(row)
    summaries = []
    for recipe, rows in grouped.items():
        decisions = Counter(str(row.get("decision") or "unknown") for row in rows)
        viable = [row for row in rows if row["scorecard"]["viable_escalation_candidate"]]
        timed = [
            row
            for row in rows
            if row.get("efficiency")
            and float((row["efficiency"] or {}).get("total_seconds") or 0.0) > 0.0
        ]
        positive = [row for row in rows if float(row.get("delta") or 0.0) > 0.0]
        best = max(rows, key=lambda row: (float(row.get("delta") or 0.0), float(row.get("score") or 0.0)))
        phase_counts = Counter(
            str((row.get("efficiency") or {}).get("dominant_phase"))
            for row in timed
            if (row.get("efficiency") or {}).get("dominant_phase")
        )
        delta_per_hour_values = [
            float(row["scorecard"]["delta_per_hour"])
            for row in rows
            if row["scorecard"].get("delta_per_hour") is not None
        ]
        summaries.append(
            {
                "recipe_id": recipe,
                "policies": len(rows),
                "positive_delta_policies": len(positive),
                "viable_candidates": len(viable),
                "decisions": dict(sorted(decisions.items())),
                "best_policy_id": best.get("policy_id"),
                "best_delta": best.get("delta"),
                "best_score": best.get("score"),
                "median_delta_per_hour": median(delta_per_hour_values)
                if delta_per_hour_values
                else None,
                "dominant_phase_mode": phase_counts.most_common(1)[0][0]
                if phase_counts
                else None,
                "recommended_action": recommend_recipe_action(rows),
            }
        )
    summaries.sort(
        key=lambda row: (
            -int(row["viable_candidates"]),
            -float(row.get("best_delta") or 0.0),
            str(row["recipe_id"]),
        )
    )
    return summaries


def recommend_recipe_action(rows: list[dict[str, Any]]) -> str:
    if any(row["scorecard"]["viable_escalation_candidate"] for row in rows):
        return "escalate_best_member"
    if any(row.get("opponent_regressions") for row in rows):
        return "train_targeted_exploiter_or_dagger_repair"
    if any(float(row.get("delta") or 0.0) > 0.0 for row in rows):
        return "grade_more_before_scaling"
    timed = [
        row for row in rows
        if row.get("efficiency") and float((row["efficiency"] or {}).get("total_seconds") or 0.0) > 0.0
    ]
    if timed:
        phase = Counter(
            str((row.get("efficiency") or {}).get("dominant_phase"))
            for row in timed
            if (row.get("efficiency") or {}).get("dominant_phase")
        ).most_common(1)
        if phase:
            return f"profile_{phase[0][0]}_bottleneck"
    return "drop_or_redesign_recipe"


def recommend_fleet_action(
    scored_rows: list[dict[str, Any]],
    *,
    recipe_rows: list[dict[str, Any]],
    payoff_recommendation: dict[str, Any],
    timed_runs: int,
) -> dict[str, Any]:
    viable = [row for row in scored_rows if row["scorecard"]["viable_escalation_candidate"]]
    if viable:
        best = viable[0]
        action = "escalate_best_candidate"
        reason = "strict gate has a positive non-regressing candidate"
        queue = [
            {
                "action": "escalate_gate",
                "policy_id": best["policy_id"],
                "family_id": best["family_id"],
                "recipe_id": best["recipe_id"],
            }
        ]
    else:
        action = str(payoff_recommendation.get("next_action") or "collect_more_grades")
        reason = "no strict viable candidate; follow weak-opponent repair loop"
        queue = list(payoff_recommendation.get("experiment_queue") or [])
    if timed_runs <= 0:
        queue.append(
            {
                "action": "wait_for_timed_runs_or_restart_next_jobs",
                "reason": "current reports do not contain train_ppo timing fields yet",
            }
        )
    elif recipe_rows:
        profile_recipe = next(
            (
                row
                for row in recipe_rows
                if str(row.get("recommended_action") or "").startswith("profile_")
            ),
            None,
        )
        if profile_recipe:
            queue.append(
                {
                    "action": "profile_recipe",
                    "recipe_id": profile_recipe["recipe_id"],
                    "reason": profile_recipe["recommended_action"],
                }
            )
    return {
        "next_action": action,
        "reason": reason,
        "queue": queue,
    }


def infer_run_id(path: Path) -> str:
    stem = path.stem
    for suffix in ("_report", "_latest"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def family_id(policy_id: str) -> str:
    policy = Path(policy_id).name
    if policy.endswith(".pt"):
        policy = policy[:-3]
    return ITERATION_SUFFIX_RE.sub("", policy)


def recipe_id(family: str) -> str:
    recipe = RUN_SEED_RE.sub("", family)
    recipe = WORKER_SUFFIX_RE.sub("", recipe)
    return recipe or family


if __name__ == "__main__":
    main()
