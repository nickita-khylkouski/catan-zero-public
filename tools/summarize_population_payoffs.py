from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_LEDGER = "runs/self_play/population_payoffs.jsonl"
REQUIRED_ESCALATION_LEGS = (
    "heuristic",
    "jsettlers_lite",
    "value_rollout_search",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize the population payoff ledger into strict-gate rankings "
            "and opponent-specific weak spots for PFSP-style planning."
        )
    )
    parser.add_argument("--ledger", default=DEFAULT_LEDGER)
    parser.add_argument("--output")
    parser.add_argument("--profile", default="strict")
    parser.add_argument(
        "--opponent",
        default="current_best_s9752_iter0002",
        help="Champion/opponent id to rank grade_pair rows against.",
    )
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.ledger))
    summary = summarize_payoffs(
        rows,
        profile=args.profile,
        opponent=args.opponent,
        top=args.top,
    )
    payload = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    print(payload)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def summarize_payoffs(
    rows: list[dict[str, Any]],
    *,
    profile: str,
    opponent: str,
    top: int,
) -> dict[str, Any]:
    rows = dedupe_rows(rows)
    pair_rows = [
        row
        for row in rows
        if row.get("type") == "grade_pair"
        and row.get("profile") == profile
        and row.get("opponent_id") == opponent
    ]
    leg_rows = [row for row in rows if row.get("type") == "grade_leg"]

    legs_by_policy: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in leg_rows:
        policy = str(row.get("policy_id") or "")
        opponent_id = str(row.get("opponent_id") or "")
        if not policy or not opponent_id:
            continue
        bucket = legs_by_policy[policy].setdefault(
            opponent_id,
            {"wins": 0, "games": 0, "workers": set()},
        )
        bucket["wins"] += int(row.get("wins") or 0)
        bucket["games"] += int(row.get("games") or 0)
        if row.get("worker"):
            bucket["workers"].add(str(row["worker"]))

    champion_leg_scores = _leg_scores(legs_by_policy.get(opponent, {}))
    policy_rows = [
        _policy_summary(row, legs_by_policy, champion_leg_scores=champion_leg_scores)
        for row in pair_rows
    ]
    policy_rows.sort(
        key=lambda item: (
            -float(item["delta"]),
            -float(item["score"]),
            float(item["worst_leg_score"])
            if item["worst_leg_score"] is not None
            else 1.0,
            item["policy_id"],
        )
    )
    decision_counts = Counter(str(row.get("decision") or "unknown") for row in pair_rows)
    weak_opponents = summarize_weak_opponents(policy_rows)
    failure_modes = summarize_failure_modes(policy_rows)
    training_recommendation = recommend_training(
        policy_rows,
        weak_opponents=weak_opponents,
        decision_counts=decision_counts,
        failure_modes=failure_modes,
    )
    return {
        "profile": profile,
        "opponent": opponent,
        "pair_rows": len(pair_rows),
        "decision_counts": dict(sorted(decision_counts.items())),
        "top": policy_rows[:top],
        "weak_opponents": weak_opponents,
        "failure_modes": failure_modes,
        "training_recommendation": training_recommendation,
    }


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keyed: dict[str, dict[str, Any]] = {}
    unkeyed: list[dict[str, Any]] = []
    for row in rows:
        key = row.get("key")
        if key:
            keyed[str(key)] = row
        else:
            unkeyed.append(row)
    return [*unkeyed, *keyed.values()]


def _policy_summary(
    row: dict[str, Any],
    legs_by_policy: dict[str, dict[str, dict[str, Any]]],
    *,
    champion_leg_scores: dict[str, float],
) -> dict[str, Any]:
    policy = str(row.get("policy_id") or "")
    legs = {}
    worst_leg_name = None
    worst_leg_score = None
    opponent_regressions = []
    for opponent_id, result in sorted(legs_by_policy.get(policy, {}).items()):
        games = int(result["games"])
        wins = int(result["wins"])
        score = wins / games if games else 0.0
        score_lcb, score_ucb = wilson_bounds(wins, games)
        champion_score = champion_leg_scores.get(opponent_id)
        delta_vs_champion = None
        delta_lcb_vs_champion = None
        if champion_score is not None:
            delta_vs_champion = score - champion_score
            champion_wins = int(round(champion_score * games))
            _, champion_ucb = wilson_bounds(champion_wins, games)
            delta_lcb_vs_champion = score_lcb - champion_ucb
            if delta_vs_champion < 0.0:
                opponent_regressions.append(
                    {
                        "opponent_id": opponent_id,
                        "delta_vs_champion": delta_vs_champion,
                        "delta_lcb_vs_champion": delta_lcb_vs_champion,
                        "candidate_score": score,
                        "champion_score": champion_score,
                    }
                )
        legs[opponent_id] = {
            "wins": wins,
            "games": games,
            "score": score,
            "score_lcb": score_lcb,
            "score_ucb": score_ucb,
            "champion_score": champion_score,
            "delta_vs_champion": delta_vs_champion,
            "delta_lcb_vs_champion": delta_lcb_vs_champion,
            "workers": sorted(result["workers"]),
        }
        if worst_leg_score is None or score < worst_leg_score:
            worst_leg_name = opponent_id
            worst_leg_score = score
    opponent_regressions.sort(
        key=lambda item: (float(item["delta_vs_champion"]), item["opponent_id"])
    )
    summary_games = int(row.get("summary_games") or 0)
    score = float(row.get("score") or 0.0)
    opponent_score = row.get("opponent_score")
    pair_wins = int(round(score * summary_games))
    score_lcb, score_ucb = wilson_bounds(pair_wins, summary_games)
    opponent_score_lcb = None
    opponent_score_ucb = None
    delta_lcb = None
    if opponent_score is not None and summary_games > 0:
        opponent_rate = float(opponent_score)
        opponent_wins = int(round(opponent_rate * summary_games))
        opponent_score_lcb, opponent_score_ucb = wilson_bounds(opponent_wins, summary_games)
        delta_lcb = score_lcb - opponent_score_ucb
    return {
        "policy_id": policy,
        "score": score,
        "score_lcb": score_lcb,
        "score_ucb": score_ucb,
        "opponent_score": float(opponent_score) if opponent_score is not None else None,
        "opponent_score_lcb": opponent_score_lcb,
        "opponent_score_ucb": opponent_score_ucb,
        "delta": float(row.get("delta") or 0.0),
        "delta_lcb": delta_lcb,
        "decision": row.get("decision"),
        "reason": row.get("reason"),
        "summary_games": row.get("summary_games"),
        "worker": row.get("worker"),
        "worst_leg": worst_leg_name,
        "worst_leg_score": worst_leg_score,
        "opponent_regressions": opponent_regressions,
        "legs": legs,
    }


def wilson_bounds(wins: int, games: int, *, z: float = 1.96) -> tuple[float | None, float | None]:
    if games <= 0:
        return None, None
    p = wins / games
    z2 = z * z
    denom = 1.0 + z2 / games
    center = (p + z2 / (2.0 * games)) / denom
    margin = z * ((p * (1.0 - p) / games + z2 / (4.0 * games * games)) ** 0.5) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def _leg_scores(legs: dict[str, dict[str, Any]]) -> dict[str, float]:
    scores = {}
    for opponent_id, result in legs.items():
        games = int(result["games"])
        wins = int(result["wins"])
        if games > 0:
            scores[opponent_id] = wins / games
    return scores


def summarize_weak_opponents(policy_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_opponent: dict[str, dict[str, Any]] = {}
    for row in policy_rows:
        for opponent_id, result in row["legs"].items():
            games = int(result["games"])
            if games <= 0:
                continue
            score = float(result["score"])
            delta_vs_champion = result.get("delta_vs_champion")
            if delta_vs_champion is not None and float(delta_vs_champion) >= 0.0:
                continue
            severity = 1.0 - score
            if delta_vs_champion is not None:
                severity = max(severity, abs(float(delta_vs_champion)))
            bucket = by_opponent.setdefault(
                opponent_id,
                {"opponent_id": opponent_id, "policies": 0, "games": 0, "weighted_loss": 0.0},
            )
            bucket["policies"] += 1
            bucket["games"] += games
            bucket["weighted_loss"] += severity * games
    rows = []
    for bucket in by_opponent.values():
        games = int(bucket["games"])
        rows.append(
            {
                "opponent_id": bucket["opponent_id"],
                "policies": bucket["policies"],
                "games": games,
                "pfsp_priority": bucket["weighted_loss"] / games if games else 0.0,
            }
        )
    rows.sort(key=lambda item: (-float(item["pfsp_priority"]), item["opponent_id"]))
    return rows


def summarize_failure_modes(policy_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_mode: dict[str, dict[str, Any]] = {}
    for row in policy_rows:
        reason = str(row.get("reason") or "")
        if str(row.get("decision") or "") != "reject":
            continue
        if reason.startswith("opponent regression "):
            for part in reason.removeprefix("opponent regression ").split(","):
                opponent_id = part.split(":", 1)[0].strip()
                if not opponent_id:
                    continue
                mode = f"opponent_regression:{opponent_id}"
                bucket = by_mode.setdefault(mode, {"mode": mode, "count": 0, "examples": []})
                bucket["count"] += 1
                if len(bucket["examples"]) < 5:
                    bucket["examples"].append(row["policy_id"])
            continue
        if "aggregate delta" in reason:
            mode = "weak_aggregate_delta"
        elif "timed out" in reason:
            mode = "grade_timeout"
        else:
            mode = "other_reject"
        bucket = by_mode.setdefault(mode, {"mode": mode, "count": 0, "examples": []})
        bucket["count"] += 1
        if len(bucket["examples"]) < 5:
            bucket["examples"].append(row["policy_id"])
    rows = list(by_mode.values())
    rows.sort(key=lambda item: (-int(item["count"]), item["mode"]))
    return rows


def recommend_training(
    policy_rows: list[dict[str, Any]],
    *,
    weak_opponents: list[dict[str, Any]],
    decision_counts: Counter[str],
    failure_modes: list[dict[str, Any]],
) -> dict[str, Any]:
    viable = [
        row
        for row in policy_rows
        if is_viable_escalation_candidate(row)
    ]
    viable.sort(
        key=lambda row: (
            row.get("decision") != "promote_candidate",
            -float(row.get("delta") or 0.0),
            -float(row.get("score") or 0.0),
            str(row.get("policy_id") or ""),
        )
    )
    repair_opponents = weak_opponents[:3]
    total_priority = sum(float(row.get("pfsp_priority") or 0.0) for row in repair_opponents)
    if total_priority > 0.0:
        repair_mix = [
            {
                "opponent_id": row["opponent_id"],
                "suggested_share": float(row.get("pfsp_priority") or 0.0) / total_priority,
            }
            for row in repair_opponents
        ]
    else:
        repair_mix = []

    rejects = int(decision_counts.get("reject", 0))
    keeps = sum(
        int(decision_counts.get(name, 0))
        for name in ("promote_candidate", "keep_for_training")
    )
    if viable:
        next_action = "escalate_best_candidate"
    elif repair_mix:
        next_action = "train_anti_regression_repair"
    else:
        next_action = "collect_more_grades"
    best_candidate = viable[0] if viable else None
    experiment_queue = []
    if best_candidate:
        games = int(best_candidate.get("summary_games") or 0)
        experiment_queue.append(
            {
                "action": "escalate_gate",
                "policy_id": best_candidate["policy_id"],
                "games": 12 if games < 12 else max(24, games * 2),
            }
        )
    if repair_mix:
        experiment_queue.append(
            {
                "action": "train_weighted_dagger_antireg",
                "opponent_mix": repair_mix,
            }
        )

    return {
        "next_action": next_action,
        "best_candidate": best_candidate,
        "repair_mix": repair_mix,
        "primary_failure_mode": failure_modes[0] if failure_modes else None,
        "experiment_queue": experiment_queue,
        "reject_rate": rejects / (rejects + keeps) if rejects + keeps else None,
    }


def is_viable_escalation_candidate(row: dict[str, Any]) -> bool:
    if row.get("decision") not in {"promote_candidate", "keep_for_training"}:
        return False
    if float(row.get("delta") or 0.0) <= 0.0:
        return False
    if row.get("opponent_regressions"):
        return False
    legs = row.get("legs") or {}
    for leg in legs.values():
        if int(leg.get("games") or 0) > 0 and int(leg.get("wins") or 0) <= 0:
            return False
    for opponent_id in REQUIRED_ESCALATION_LEGS:
        leg = legs.get(opponent_id)
        if not leg:
            return False
        if int(leg.get("games") or 0) <= 0:
            return False
        if int(leg.get("wins") or 0) <= 0:
            return False
    return True


if __name__ == "__main__":
    main()
