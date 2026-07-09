from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare candidate scoreboard reports against baseline reports and "
            "emit an explicit promote/reject gate decision."
        )
    )
    parser.add_argument("--candidate-reports", required=True, help="Comma-separated scoreboard JSON files.")
    parser.add_argument("--baseline-reports", required=True, help="Comma-separated baseline scoreboard JSON files.")
    parser.add_argument(
        "--required-opponents",
        default="",
        help="Comma-separated opponent names that must be present in both candidate and baseline.",
    )
    parser.add_argument(
        "--max-regression-win-rate",
        type=float,
        default=0.0,
        help="Reject if candidate win rate is below baseline by more than this for any required opponent.",
    )
    parser.add_argument(
        "--min-improvement-win-rate",
        type=float,
        default=0.0,
        help="Require at least one required opponent to improve by this much or more.",
    )
    parser.add_argument(
        "--min-games",
        type=int,
        default=1,
        help="Reject opponents with fewer games than this in candidate or baseline reports.",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    candidate = _load_many(args.candidate_reports)
    baseline = _load_many(args.baseline_reports)
    required = [item.strip() for item in args.required_opponents.split(",") if item.strip()]
    if not required:
        required = sorted(set(candidate) & set(baseline))

    comparisons: list[dict[str, Any]] = []
    failures: list[str] = []
    best_delta = -float("inf")
    for opponent in required:
        c = candidate.get(opponent)
        b = baseline.get(opponent)
        if c is None:
            failures.append(f"missing candidate opponent {opponent}")
            continue
        if b is None:
            failures.append(f"missing baseline opponent {opponent}")
            continue
        if int(c["games"]) < args.min_games:
            failures.append(f"candidate opponent {opponent} has only {c['games']} games")
        if int(b["games"]) < args.min_games:
            failures.append(f"baseline opponent {opponent} has only {b['games']} games")
        c_rate = float(c["win_rate"])
        b_rate = float(b["win_rate"])
        delta = c_rate - b_rate
        best_delta = max(best_delta, delta)
        if delta < -float(args.max_regression_win_rate):
            failures.append(
                f"{opponent} regressed by {-delta:.4f}, allowed {args.max_regression_win_rate:.4f}"
            )
        # FIX A8: an unpaired two-proportion z-test is conservative on paired-
        # seed results (it ignores that candidate/baseline saw identical
        # boards), making genuinely-better candidates look insignificant.
        # Prefer an exact paired McNemar test on per-game-index outcomes when
        # both reports carry that pairing metadata; otherwise fall back to
        # the unpaired z-test and say so loudly.
        eligible, reason = _pairing_eligible(c, b)
        paired_test = _mcnemar_exact(c["game_outcomes"], b["game_outcomes"]) if eligible else None
        if not eligible:
            print(
                f"WARNING: {opponent}: paired McNemar unavailable ({reason}); "
                "falling back to the unpaired two-proportion z-test, which is "
                "conservative for paired-seed evals.",
                file=sys.stderr,
            )
        comparisons.append(
            {
                "opponent": opponent,
                "candidate_wins": int(c["wins"]),
                "candidate_games": int(c["games"]),
                "candidate_win_rate": c_rate,
                "baseline_wins": int(b["wins"]),
                "baseline_games": int(b["games"]),
                "baseline_win_rate": b_rate,
                "delta_win_rate": delta,
                "delta_wins_at_candidate_games": delta * int(c["games"]),
                "candidate_avg_vp_margin": c.get("avg_vp_margin"),
                "baseline_avg_vp_margin": b.get("avg_vp_margin"),
                "delta_avg_vp_margin": _delta_optional(c.get("avg_vp_margin"), b.get("avg_vp_margin")),
                "candidate_illegal_action_count": int(c.get("illegal_action_count", 0)),
                "baseline_illegal_action_count": int(b.get("illegal_action_count", 0)),
                "paired_mcnemar": paired_test,
                "pairing_unavailable_reason": None if eligible else reason,
                "approx_z_unpaired": _two_proportion_z(
                    int(c["wins"]), int(c["games"]), int(b["wins"]), int(b["games"])
                ),
            }
        )
    if comparisons and best_delta < float(args.min_improvement_win_rate):
        failures.append(
            f"best improvement {best_delta:.4f} is below required {args.min_improvement_win_rate:.4f}"
        )

    report = {
        "candidate_reports": _split_paths(args.candidate_reports),
        "baseline_reports": _split_paths(args.baseline_reports),
        "required_opponents": required,
        "max_regression_win_rate": float(args.max_regression_win_rate),
        "min_improvement_win_rate": float(args.min_improvement_win_rate),
        "min_games": int(args.min_games),
        "comparisons": comparisons,
        "failures": failures,
        "decision": "promote" if not failures else "reject",
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


def _split_paths(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_many(paths: str) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for path in _split_paths(paths):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        report_seed = data.get("seed")
        report_paired_seeds = bool(data.get("paired_seeds", False))
        for result in data.get("results", []):
            opponent = str(result["opponent"])
            if opponent in merged:
                raise SystemExit(f"duplicate opponent {opponent} across reports")
            entry = dict(result)
            # Report-level pairing metadata (evaluate_scoreboard.py), namespaced
            # so it never collides with a per-opponent result field.
            entry["_report_seed"] = report_seed
            entry["_report_paired_seeds"] = report_paired_seeds
            merged[opponent] = entry
    return merged


def _delta_optional(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    return float(a) - float(b)


def _pairing_eligible(c: dict[str, Any], b: dict[str, Any]) -> tuple[bool, str]:
    """Check whether candidate/baseline results for one opponent were run with
    an identical, aligned per-game-index seed schedule and both carry the
    per-game outcome arrays needed for exact McNemar pairing."""
    if not c.get("_report_paired_seeds") or not b.get("_report_paired_seeds"):
        return False, "report was not generated with evaluate_scoreboard.py --paired-seeds"
    if c.get("_report_seed") is None or b.get("_report_seed") is None:
        return False, "report is missing the top-level seed (regenerate with the updated tool)"
    if c["_report_seed"] != b["_report_seed"]:
        return False, f"different base seed ({c['_report_seed']} vs {b['_report_seed']})"
    c_leg_seed = c.get("leg_seed")
    b_leg_seed = b.get("leg_seed")
    if c_leg_seed is None or b_leg_seed is None:
        return False, "missing leg_seed (regenerate with the updated evaluate_scoreboard.py)"
    if c_leg_seed != b_leg_seed:
        return False, f"leg_seed mismatch ({c_leg_seed} vs {b_leg_seed})"
    c_outcomes = c.get("game_outcomes")
    b_outcomes = b.get("game_outcomes")
    if not isinstance(c_outcomes, list) or not isinstance(b_outcomes, list):
        return False, "missing per-game game_outcomes (regenerate with the updated tool)"
    if not c_outcomes or not b_outcomes:
        return False, "empty game_outcomes"
    if len(c_outcomes) != len(b_outcomes):
        return False, f"game_outcomes length mismatch ({len(c_outcomes)} vs {len(b_outcomes)})"
    return True, ""


def _mcnemar_exact(c_outcomes: list[Any], b_outcomes: list[Any]) -> dict[str, Any]:
    """Exact paired McNemar test (candidate vs baseline win/loss, per game
    index) via the sign test on discordant pairs. Uses scipy if available on
    the box; otherwise an exact binomial computed with pure stdlib math.

    FIX (adversarial review, truncation-as-loss bias): a game_outcomes entry
    is None when that game was truncated (no winner) -- coercing it with
    bool() would silently count it as a loss and bias the test against
    whichever side truncates more often. Any pair where EITHER side is None
    is excluded entirely (it carries no win/loss information) and counted
    separately in "truncated_pairs_excluded".
    """
    candidate_only = 0  # candidate won, baseline lost the same game
    baseline_only = 0  # candidate lost, baseline won the same game
    truncated_pairs = 0
    for c_won, b_won in zip(c_outcomes, b_outcomes):
        if c_won is None or b_won is None:
            truncated_pairs += 1
            continue
        c_won, b_won = bool(c_won), bool(b_won)
        if c_won and not b_won:
            candidate_only += 1
        elif b_won and not c_won:
            baseline_only += 1
    discordant = candidate_only + baseline_only
    usable_games = len(c_outcomes) - truncated_pairs
    p_value = _exact_binomial_two_sided_p(discordant, min(candidate_only, baseline_only))
    return {
        "test": "mcnemar_exact",
        "games": len(c_outcomes),
        "truncated_pairs_excluded": truncated_pairs,
        "usable_games": usable_games,
        "concordant": usable_games - discordant,
        "discordant_candidate_only_wins": candidate_only,
        "discordant_baseline_only_wins": baseline_only,
        "discordant_total": discordant,
        "p_value": p_value,
    }


def _exact_binomial_two_sided_p(n: int, k: int) -> float:
    if n <= 0:
        return 1.0
    try:
        from scipy import stats  # type: ignore

        return float(stats.binomtest(k, n, p=0.5, alternative="two-sided").pvalue)
    except Exception:
        pass
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return float(min(1.0, 2.0 * tail))


def _two_proportion_z(c_wins: int, c_games: int, b_wins: int, b_games: int) -> float | None:
    if c_games <= 0 or b_games <= 0:
        return None
    p1 = c_wins / c_games
    p2 = b_wins / b_games
    pooled = (c_wins + b_wins) / (c_games + b_games)
    se = math.sqrt(max(0.0, pooled * (1.0 - pooled) * (1.0 / c_games + 1.0 / b_games)))
    if se <= 0:
        return None
    return (p1 - p2) / se


if __name__ == "__main__":
    main()
