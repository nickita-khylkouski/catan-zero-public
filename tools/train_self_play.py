from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from catan_zero.rl import (
    CatanatronValuePolicy,
    HeuristicPolicy,
    OnePlySearchPolicy,
    RandomPolicy,
    collect_imitation_game,
    create_linear_policy,
    create_mlp_policy,
    evaluate_policy,
    play_game,
)
from catan_zero.rl.self_play import make_env_config, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a minimal self-play policy.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--bootstrap-episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--vps-to-win", type=int, default=4)
    parser.add_argument("--max-decisions", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--policy", choices=("linear", "mlp"), default="mlp")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--imitation-epochs", type=int, default=1)
    parser.add_argument("--imitation-strength", type=float, default=1.0)
    parser.add_argument(
        "--teacher",
        choices=("heuristic", "search", "value"),
        default="value",
    )
    parser.add_argument("--search-candidate-limit", type=int, default=24)
    parser.add_argument("--search-rollout-decisions", type=int, default=4)
    parser.add_argument(
        "--opponents",
        choices=("self", "random", "heuristic", "search", "value"),
        default="heuristic",
        help="Opponent pool used during policy-gradient episodes.",
    )
    parser.add_argument("--checkpoint", default="runs/self_play/policy.npz")
    parser.add_argument("--report", default="runs/self_play/report.json")
    parser.add_argument("--eval-games", type=int, default=8)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    config = make_env_config(vps_to_win=args.vps_to_win)
    if args.policy == "mlp":
        policy = create_mlp_policy(
            config=config,
            seed=args.seed,
            learning_rate=args.learning_rate,
            temperature=args.temperature,
            hidden_size=args.hidden_size,
        )
    else:
        policy = create_linear_policy(
            config=config,
            seed=args.seed,
            learning_rate=args.learning_rate,
            temperature=args.temperature,
        )
    bootstrap_summaries = []
    teacher = _make_policy(
        args.teacher,
        candidate_limit=args.search_candidate_limit,
        rollout_decisions=args.search_rollout_decisions,
    )
    for bootstrap_idx in range(args.bootstrap_episodes):
        episode = collect_imitation_game(
            teacher,
            seed=int(rng.integers(2**31)),
            config=config,
            max_decisions=args.max_decisions,
            rng=rng,
        )
        for _ in range(args.imitation_epochs):
            policy.update_imitation(episode, strength=args.imitation_strength)
        bootstrap_summaries.append(
            {
                "episode": bootstrap_idx + 1,
                "teacher": teacher.name,
                "winner": episode.result.winner,
                "decisions": episode.result.decisions,
                "terminated": episode.result.terminated,
                "truncated": episode.result.truncated,
            }
        )
        print(json.dumps({"bootstrap": bootstrap_summaries[-1]}, sort_keys=True))

    episode_summaries = []
    player_names = ("BLUE", "RED", "ORANGE", "WHITE")[: config.players]
    opponent = _make_policy(
        args.opponents,
        candidate_limit=args.search_candidate_limit,
        rollout_decisions=args.search_rollout_decisions,
    )
    for episode_idx in range(args.episodes):
        training_seat = player_names[episode_idx % len(player_names)]
        if args.opponents == "self":
            policies = {name: policy for name in player_names}
        else:
            policies = {
                name: policy if name == training_seat else opponent
                for name in player_names
            }
        episode = play_game(
            policies,
            seed=int(rng.integers(2**31)),
            config=config,
            max_decisions=args.max_decisions,
            rng=rng,
            training_policy=policy,
        )
        policy.update_episode(episode)
        episode_summaries.append(
            {
                "episode": episode_idx + 1,
                "training_seat": training_seat if args.opponents != "self" else "all",
                "opponents": args.opponents,
                "winner": episode.result.winner,
                "decisions": episode.result.decisions,
                "terminated": episode.result.terminated,
                "truncated": episode.result.truncated,
                "final_public_vps": episode.result.final_public_vps,
            }
        )
        print(json.dumps(episode_summaries[-1], sort_keys=True))

    checkpoint = Path(args.checkpoint)
    policy.save(checkpoint)
    random_eval = evaluate_policy(
        policy,
        RandomPolicy(),
        games=args.eval_games,
        seed=args.seed + 10_000,
        config=config,
        max_decisions=args.max_decisions,
    )
    heuristic_eval = evaluate_policy(
        policy,
        HeuristicPolicy(),
        games=args.eval_games,
        seed=args.seed + 20_000,
        config=config,
        max_decisions=args.max_decisions,
    )
    report = {
        "episodes": args.episodes,
        "bootstrap_episodes": args.bootstrap_episodes,
        "checkpoint": str(checkpoint),
        "config": {
            "vps_to_win": args.vps_to_win,
            "max_decisions": args.max_decisions,
            "learning_rate": args.learning_rate,
            "temperature": args.temperature,
            "teacher": args.teacher,
            "opponents": args.opponents,
            "policy": args.policy,
            "hidden_size": args.hidden_size if args.policy == "mlp" else None,
            "imitation_epochs": args.imitation_epochs,
            "imitation_strength": args.imitation_strength,
            "search_candidate_limit": args.search_candidate_limit,
            "search_rollout_decisions": args.search_rollout_decisions,
        },
        "bootstrap": bootstrap_summaries,
        "training": episode_summaries,
        "eval_vs_random": random_eval,
        "eval_vs_heuristic": heuristic_eval,
    }
    write_report(report, args.report)
    print(json.dumps(report, indent=2, sort_keys=True))


def _make_policy(kind: str, *, candidate_limit: int, rollout_decisions: int):
    if kind == "self":
        return None
    if kind == "random":
        return RandomPolicy()
    if kind == "heuristic":
        return HeuristicPolicy()
    if kind == "search":
        return OnePlySearchPolicy(
            candidate_limit=candidate_limit,
            rollout_decisions=rollout_decisions,
        )
    if kind == "value":
        return CatanatronValuePolicy(candidate_limit=candidate_limit)
    raise ValueError(kind)


if __name__ == "__main__":
    main()
