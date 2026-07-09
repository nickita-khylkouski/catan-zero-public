from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from catan_zero.rl import (
    CatanatronValuePolicy,
    ValueRolloutSearchPolicy,
    collect_imitation_game,
    flatten_episode_for_reanalysis,
    make_env_config,
    write_reanalysis_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate reusable search/reanalysis targets for CatanZero.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--vps-to-win", type=int, default=6)
    parser.add_argument("--max-decisions", type=int, default=600)
    parser.add_argument(
        "--teacher",
        choices=("value", "value_rollout"),
        default="value_rollout",
    )
    parser.add_argument("--candidate-limit", type=int, default=24)
    parser.add_argument("--presearch-candidate-limit", type=int, default=96)
    parser.add_argument("--rollout-decisions", type=int, default=6)
    parser.add_argument("--rollout-samples", type=int, default=1)
    parser.add_argument("--root-value-weight", type=float, default=0.25)
    parser.add_argument("--opponent-penalty", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument(
        "--record-after-decisions",
        type=int,
        default=0,
        help=(
            "Skip early decisions before writing reanalysis samples. "
            "This supports DAGS-style mid/late-game data without changing the simulator."
        ),
    )
    parser.add_argument(
        "--record-window-decisions",
        type=int,
        default=0,
        help=(
            "Maximum decision window to record after --record-after-decisions; "
            "0 records through game end."
        ),
    )
    parser.add_argument(
        "--graph-history-features",
        action="store_true",
        help="Append structured board/history features to stored observations.",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    config = make_env_config(
        vps_to_win=args.vps_to_win,
        use_graph_history_features=args.graph_history_features,
    )
    if args.teacher == "value":
        teacher = CatanatronValuePolicy(
            candidate_limit=args.presearch_candidate_limit,
            opponent_penalty=args.opponent_penalty,
            distillation_temperature=args.temperature,
        )
    else:
        teacher = ValueRolloutSearchPolicy(
            candidate_limit=args.candidate_limit,
            presearch_candidate_limit=args.presearch_candidate_limit,
            rollout_decisions=args.rollout_decisions,
            rollout_samples=args.rollout_samples,
            root_value_weight=args.root_value_weight,
            opponent_penalty=args.opponent_penalty,
            distillation_temperature=args.temperature,
        )

    output = Path(args.output)
    total_samples = 0
    summaries = []
    record_until_decision = (
        args.record_after_decisions + args.record_window_decisions
        if args.record_window_decisions > 0
        else None
    )
    for game_idx in range(args.games):
        episode = collect_imitation_game(
            teacher,
            seed=int(rng.integers(2**31)),
            config=config,
            max_decisions=args.max_decisions,
            record_after_decisions=args.record_after_decisions,
            record_until_decision=record_until_decision,
            rng=rng,
        )
        samples, returns = flatten_episode_for_reanalysis(episode, gamma=args.gamma)
        count = write_reanalysis_jsonl(
            output,
            samples,
            returns,
            metadata={
                "teacher": teacher.name,
                "seed": args.seed,
                "game_index": game_idx,
                "vps_to_win": args.vps_to_win,
                "max_decisions": args.max_decisions,
                "record_after_decisions": args.record_after_decisions,
                "record_until_decision": record_until_decision,
                "graph_history_features": args.graph_history_features,
            },
            append=game_idx > 0,
        )
        total_samples += count
        summary = {
            "game": game_idx + 1,
            "samples": count,
            "total_samples": total_samples,
            "winner": episode.result.winner,
            "decisions": episode.result.decisions,
            "invalid_actions": episode.result.invalid_actions,
        }
        summaries.append(summary)
        print(json.dumps({"reanalysis": summary}, sort_keys=True), flush=True)

    report = {
        "output": str(output),
        "config": vars(args),
        "samples": total_samples,
        "games": summaries,
    }
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
