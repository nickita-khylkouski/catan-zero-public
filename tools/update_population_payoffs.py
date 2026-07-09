from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Append remote grade summaries to a population payoff ledger. The "
            "ledger is the lightweight source of truth for later AlphaRank/PSRO "
            "style selection and keeps rejected branches visible as evidence."
        )
    )
    parser.add_argument("--summary", required=True)
    parser.add_argument(
        "--output",
        default="runs/self_play/population_payoffs.jsonl",
    )
    parser.add_argument("--run-label", default="")
    parser.add_argument(
        "--dedupe-existing",
        action="store_true",
        default=True,
        help="Do not append entries whose deterministic key already exists.",
    )
    parser.add_argument(
        "--allow-duplicates",
        action="store_false",
        dest="dedupe_existing",
        help="Append all entries, even if their deterministic keys already exist.",
    )
    args = parser.parse_args()

    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    entries = build_payoff_entries(summary, run_label=args.run_label)
    output = Path(args.output)
    written = append_payoff_entries(
        output,
        entries,
        dedupe_existing=args.dedupe_existing,
    )
    print(
        json.dumps(
            {"entries": len(entries), "output": str(output), "written": written},
            sort_keys=True,
        )
    )


def build_payoff_entries(
    summary: dict[str, Any],
    *,
    run_label: str = "",
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for row in summary.get("decisions", []):
        checkpoint = row.get("checkpoint")
        champion = row.get("champion")
        if not checkpoint or not champion:
            continue
        candidate_score = row.get("candidate_weighted_win_rate")
        champion_score = row.get("champion_weighted_win_rate")
        if candidate_score is None:
            continue
        delta = -1.0
        if champion_score is not None:
            delta = float(candidate_score) - float(champion_score)
        policy_id = policy_id_from_path(str(checkpoint))
        opponent_id = policy_id_from_path(str(champion))
        profile = infer_profile_from_summary(str(row.get("summary", "")))
        entries.append(
            {
                "key": stable_key(
                    "grade_pair",
                    policy_id,
                    opponent_id,
                    profile,
                    str(row.get("summary", "")),
                ),
                "type": "grade_pair",
                "policy_id": policy_id,
                "opponent_id": opponent_id,
                "score": float(candidate_score),
                "opponent_score": float(champion_score) if champion_score is not None else None,
                "delta": delta,
                "decision": row.get("decision"),
                "paired_aggregate_delta": _maybe_float(
                    (row.get("paired_delta") or {}).get("aggregate_delta")
                ),
                "paired_aggregate_lower_delta": _maybe_float(
                    (row.get("paired_delta") or {}).get("aggregate_lower_delta")
                ),
                "paired_worst_opponent": (row.get("paired_delta") or {}).get("worst_opponent"),
                "reason": row.get("reason"),
                "profile": profile,
                "summary": row.get("summary"),
                "summary_games": row.get("summary_games"),
                "worker": row.get("worker"),
                "zone": row.get("zone"),
                "run_label": run_label,
            }
        )
        paired_opponents = (row.get("paired_delta") or {}).get("opponents") or {}
        for opponent, delta_row in sorted(paired_opponents.items()):
            entries.append(
                {
                    "key": stable_key(
                        "grade_opponent_pair",
                        policy_id,
                        opponent_id,
                        profile,
                        str(row.get("summary", "")),
                        str(opponent),
                    ),
                    "type": "grade_opponent_pair",
                    "policy_id": policy_id,
                    "champion_id": opponent_id,
                    "opponent_id": str(opponent),
                    "profile": profile,
                    "score": _maybe_float(delta_row.get("candidate_win_rate")),
                    "opponent_score": _maybe_float(delta_row.get("champion_win_rate")),
                    "delta": _maybe_float(delta_row.get("win_rate_delta")),
                    "lower_delta": _maybe_float(delta_row.get("lower_delta")),
                    "wins": _maybe_int(delta_row.get("candidate_wins")),
                    "games": _maybe_int(delta_row.get("candidate_games")),
                    "champion_wins": _maybe_int(delta_row.get("champion_wins")),
                    "champion_games": _maybe_int(delta_row.get("champion_games")),
                    "decision": row.get("decision"),
                    "summary": row.get("summary"),
                    "worker": row.get("worker"),
                    "zone": row.get("zone"),
                    "run_label": run_label,
                }
            )
    for row in summary.get("legs", []):
        checkpoint = row.get("checkpoint")
        if not checkpoint:
            continue
        policy_id = policy_id_from_path(str(checkpoint))
        for opponent, result in sorted((row.get("opponents") or {}).items()):
            entries.append(
                {
                    "key": stable_key(
                        "grade_leg",
                        policy_id,
                        str(opponent),
                        str(row.get("worker", "")),
                    ),
                    "type": "grade_leg",
                    "policy_id": policy_id,
                    "opponent_id": str(opponent),
                    "score": float(result.get("win_rate", 0.0)),
                    "wins": int(result.get("wins", 0)),
                    "games": int(result.get("games", 0)),
                    "worker": row.get("worker"),
                    "run_label": run_label,
                }
            )
    return entries


def append_payoff_entries(
    output: Path,
    entries: list[dict[str, Any]],
    *,
    dedupe_existing: bool,
) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    existing_keys = set()
    if dedupe_existing and output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = row.get("key")
            if key:
                existing_keys.add(str(key))
    written = 0
    with output.open("a", encoding="utf-8") as handle:
        for entry in entries:
            key = str(entry["key"])
            if key in existing_keys:
                continue
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
            existing_keys.add(key)
            written += 1
    return written


def infer_profile_from_summary(name: str) -> str:
    for profile in ("jsettlers_triage", "strict", "search_stress", "dev"):
        if f"_{profile}_" in name:
            return profile
    return "unknown"


def policy_id_from_path(path: str) -> str:
    return Path(path).name.removesuffix(".pt")


def stable_key(*parts: str) -> str:
    return "|".join(parts)


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


if __name__ == "__main__":
    main()
