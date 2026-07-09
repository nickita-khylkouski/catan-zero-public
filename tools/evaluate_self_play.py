from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import sys

import numpy as np

from catan_zero.rl import (
    CatanatronValuePolicy,
    HeuristicPolicy,
    JSettlersLitePolicy,
    LinearSoftmaxPolicy,
    NumpyMLPPolicy,
    OnePlySearchPolicy,
    RandomPolicy,
    ValueRolloutSearchPolicy,
    elo_difference,
    evaluate_policy,
    play_game,
)
from catan_zero.rl.self_play import make_env_config, write_report
from catan_zero.rl.hybrid_policy import OpeningThenPolicy
from catan_zero.rl.torch_ppo import TorchPPOPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a CatanZero policy.")
    parser.add_argument(
        "--candidate",
        choices=(
            "random",
            "heuristic",
            "jsettlers_lite",
            "search",
            "value_rollout",
            "value",
            "linear",
            "mlp",
            "ppo",
            "hybrid_opening_ppo",
        ),
        default="heuristic",
    )
    parser.add_argument("--checkpoint", help="Linear policy .npz checkpoint")
    parser.add_argument(
        "--opening-checkpoint",
        help="Opening specialist checkpoint for --candidate hybrid_opening_ppo.",
    )
    parser.add_argument(
        "--opening-prompts",
        default="BUILD_INITIAL_SETTLEMENT,BUILD_INITIAL_ROAD",
        help=(
            "Comma-separated prompt names where --opening-checkpoint is used "
            "for --candidate hybrid_opening_ppo."
        ),
    )
    parser.add_argument(
        "--opponent",
        choices=("random", "heuristic", "jsettlers_lite", "search", "value_rollout", "value"),
        default="random",
    )
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--vps-to-win", type=int, default=4)
    parser.add_argument("--max-decisions", type=int, default=1000)
    parser.add_argument("--search-candidate-limit", type=int, default=48)
    parser.add_argument(
        "--search-presearch-candidate-limit",
        type=int,
        default=0,
        help=(
            "If >0 for value_rollout, one-ply value-rank this wider candidate "
            "pool before spending rollouts on --search-candidate-limit actions."
        ),
    )
    parser.add_argument("--search-rollout-decisions", type=int, default=8)
    parser.add_argument("--search-rollout-samples", type=int, default=1)
    parser.add_argument(
        "--search-root-value-weight",
        type=float,
        default=0.0,
        help=(
            "For value_rollout, blend this much immediate root value into the "
            "short-rollout score. 0 preserves pure rollout search."
        ),
    )
    parser.add_argument("--search-opponent-penalty", type=float, default=0.05)
    parser.add_argument(
        "--opponent-candidate-limit",
        type=int,
        default=48,
        help="Candidate limit for opponent policies that prune legal actions.",
    )
    parser.add_argument(
        "--opponent-rollout-decisions",
        type=int,
        default=8,
        help="Rollout depth if the opponent is a search policy.",
    )
    parser.add_argument(
        "--opponent-value-penalty",
        type=float,
        default=0.05,
        help="Opponent value penalty for value-based opponent policies.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="If >0, emit JSON progress lines to stderr every N evaluated games.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Evaluate games across N worker processes. This preserves the same "
            "per-game seed schedule as serial evaluation and is intended for "
            "promotion gates."
        ),
    )
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.workers <= 1:
        candidate = _make_policy(
            args.candidate,
            args.checkpoint,
            opening_checkpoint=args.opening_checkpoint,
            opening_prompts=_parse_opening_prompts(args.opening_prompts),
            candidate_limit=args.search_candidate_limit,
            presearch_candidate_limit=args.search_presearch_candidate_limit,
            rollout_decisions=args.search_rollout_decisions,
            rollout_samples=args.search_rollout_samples,
            root_value_weight=args.search_root_value_weight,
            opponent_penalty=args.search_opponent_penalty,
        )
        opponent = _make_policy(
            args.opponent,
            None,
            candidate_limit=args.opponent_candidate_limit,
            presearch_candidate_limit=0,
            rollout_decisions=args.opponent_rollout_decisions,
            root_value_weight=0.0,
            opponent_penalty=args.opponent_value_penalty,
        )
        report = evaluate_policy(
            candidate,
            opponent,
            games=args.games,
            seed=args.seed,
            config=make_env_config(
                vps_to_win=args.vps_to_win,
                use_graph_history_features=(
                    _needs_graph_history(candidate) or _needs_graph_history(opponent)
                ),
            ),
            max_decisions=args.max_decisions,
            progress_callback=_progress_callback(args.progress_every),
        )
    else:
        report = _evaluate_policy_parallel(args)
    if args.output:
        write_report(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


def _progress_callback(progress_every: int):
    if progress_every <= 0:
        return None

    def emit(progress: dict) -> None:
        if progress["game"] % progress_every != 0 and progress["game"] != progress["games"]:
            return
        print(json.dumps({"progress": progress}, sort_keys=True), file=sys.stderr, flush=True)

    return emit


def _evaluate_policy_parallel(args: argparse.Namespace) -> dict:
    rng = np.random.default_rng(args.seed)
    game_seeds = [int(rng.integers(2**31)) for _ in range(args.games)]
    player_names = ("BLUE", "RED", "ORANGE", "WHITE")
    seat_wins = {name: 0 for name in player_names}
    wins = 0
    total_decisions = 0
    progress = _progress_callback(args.progress_every)
    opening_checkpoint = getattr(args, "opening_checkpoint", None)
    opening_prompts = _parse_opening_prompts(
        getattr(args, "opening_prompts", "")
    )
    candidate_name = _policy_name(
        args.candidate,
        args.checkpoint,
        opening_checkpoint=opening_checkpoint,
        opening_prompts=opening_prompts,
        candidate_limit=args.search_candidate_limit,
        presearch_candidate_limit=args.search_presearch_candidate_limit,
        rollout_decisions=args.search_rollout_decisions,
        rollout_samples=args.search_rollout_samples,
        root_value_weight=args.search_root_value_weight,
        opponent_penalty=args.search_opponent_penalty,
    )
    opponent_name = _policy_name(
        args.opponent,
        None,
        candidate_limit=args.opponent_candidate_limit,
        presearch_candidate_limit=0,
        rollout_decisions=args.opponent_rollout_decisions,
        rollout_samples=1,
        root_value_weight=0.0,
        opponent_penalty=args.opponent_value_penalty,
    )
    worker_args = [
        {
            "candidate": args.candidate,
            "checkpoint": args.checkpoint,
            "opening_checkpoint": opening_checkpoint,
            "opening_prompts": opening_prompts,
            "opponent": args.opponent,
            "game_idx": game_idx,
            "game_seed": game_seed,
            "players": 4,
            "vps_to_win": args.vps_to_win,
            "max_decisions": args.max_decisions,
            "search_candidate_limit": args.search_candidate_limit,
            "search_presearch_candidate_limit": args.search_presearch_candidate_limit,
            "search_rollout_decisions": args.search_rollout_decisions,
            "search_rollout_samples": args.search_rollout_samples,
            "search_root_value_weight": args.search_root_value_weight,
            "search_opponent_penalty": args.search_opponent_penalty,
            "opponent_candidate_limit": args.opponent_candidate_limit,
            "opponent_rollout_decisions": args.opponent_rollout_decisions,
            "opponent_value_penalty": args.opponent_value_penalty,
        }
        for game_idx, game_seed in enumerate(game_seeds)
    ]
    completed = 0
    worker_count = max(1, min(int(args.workers), len(worker_args) or 1))
    worker_chunks = _chunk_worker_args(worker_args, worker_count)
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(_evaluate_game_chunk, chunk)
            for chunk in worker_chunks
            if chunk
        ]
        for future in as_completed(futures):
            for result in future.result():
                completed += 1
                total_decisions += result["decisions"]
                if result["winner"] == result["candidate_seat"]:
                    wins += 1
                    seat_wins[result["candidate_seat"]] += 1
                if progress is not None:
                    progress(
                        {
                            "game": completed,
                            "games": args.games,
                            "wins": wins,
                            "win_rate": wins / float(completed),
                            "candidate_seat": result["candidate_seat"],
                            "winner": result["winner"],
                            "decisions": result["decisions"],
                            "seat_wins": dict(seat_wins),
                            "last_game_index": result["game_idx"],
                        }
                    )
    win_rate = wins / args.games if args.games else 0.0
    return {
        "games": args.games,
        "candidate": candidate_name,
        "opponent": opponent_name,
        "wins": wins,
        "win_rate": win_rate,
        "elo_vs_opponent": elo_difference(win_rate),
        "seat_wins": seat_wins,
        "avg_decisions": total_decisions / args.games if args.games else 0.0,
        "workers": int(args.workers),
    }


def _chunk_worker_args(items: list[dict], worker_count: int) -> list[list[dict]]:
    chunks: list[list[dict]] = [[] for _ in range(max(1, worker_count))]
    for index, item in enumerate(items):
        chunks[index % len(chunks)].append(item)
    return chunks


def _evaluate_game_chunk(items: list[dict]) -> list[dict]:
    if not items:
        return []
    first = items[0]
    candidate = _make_policy(
        first["candidate"],
        first["checkpoint"],
        opening_checkpoint=first.get("opening_checkpoint"),
        opening_prompts=tuple(first.get("opening_prompts", ())),
        candidate_limit=first["search_candidate_limit"],
        presearch_candidate_limit=first["search_presearch_candidate_limit"],
        rollout_decisions=first["search_rollout_decisions"],
        rollout_samples=first["search_rollout_samples"],
        root_value_weight=first["search_root_value_weight"],
        opponent_penalty=first["search_opponent_penalty"],
    )
    opponent = _make_policy(
        first["opponent"],
        None,
        candidate_limit=first["opponent_candidate_limit"],
        presearch_candidate_limit=0,
        rollout_decisions=first["opponent_rollout_decisions"],
        rollout_samples=1,
        root_value_weight=0.0,
        opponent_penalty=first["opponent_value_penalty"],
    )
    return [_evaluate_one_game_with_policies(item, candidate, opponent) for item in items]


def _evaluate_one_game(item: dict) -> dict:
    candidate = _make_policy(
        item["candidate"],
        item["checkpoint"],
        opening_checkpoint=item.get("opening_checkpoint"),
        opening_prompts=tuple(item.get("opening_prompts", ())),
        candidate_limit=item["search_candidate_limit"],
        presearch_candidate_limit=item["search_presearch_candidate_limit"],
        rollout_decisions=item["search_rollout_decisions"],
        rollout_samples=item["search_rollout_samples"],
        root_value_weight=item["search_root_value_weight"],
        opponent_penalty=item["search_opponent_penalty"],
    )
    opponent = _make_policy(
        item["opponent"],
        None,
        candidate_limit=item["opponent_candidate_limit"],
        presearch_candidate_limit=0,
        rollout_decisions=item["opponent_rollout_decisions"],
        rollout_samples=1,
        root_value_weight=0.0,
        opponent_penalty=item["opponent_value_penalty"],
    )
    return _evaluate_one_game_with_policies(item, candidate, opponent)


def _evaluate_one_game_with_policies(item: dict, candidate, opponent) -> dict:
    config = make_env_config(
        players=item["players"],
        vps_to_win=item["vps_to_win"],
        use_graph_history_features=(
            _needs_graph_history(candidate) or _needs_graph_history(opponent)
        ),
    )
    player_names = ("BLUE", "RED", "ORANGE", "WHITE")[: item["players"]]
    candidate_seat = player_names[item["game_idx"] % item["players"]]
    policies = {
        name: candidate if name == candidate_seat else opponent for name in player_names
    }
    episode = play_game(
        policies,
        seed=item["game_seed"],
        config=config,
        max_decisions=item["max_decisions"],
    )
    return {
        "game_idx": item["game_idx"],
        "candidate_seat": candidate_seat,
        "winner": episode.result.winner,
        "decisions": episode.result.decisions,
    }


def _needs_graph_history(policy) -> bool:
    return getattr(policy, "architecture", "") == "graph_history_candidate"


def _policy_name(
    kind: str,
    checkpoint: str | None,
    *,
    opening_checkpoint: str | None = None,
    opening_prompts: tuple[str, ...] = (),
    candidate_limit: int,
    presearch_candidate_limit: int = 0,
    rollout_decisions: int,
    rollout_samples: int = 1,
    root_value_weight: float = 0.0,
    opponent_penalty: float,
) -> str:
    return _make_policy(
        kind,
        checkpoint,
        opening_checkpoint=opening_checkpoint,
        opening_prompts=opening_prompts,
        candidate_limit=candidate_limit,
        presearch_candidate_limit=presearch_candidate_limit,
        rollout_decisions=rollout_decisions,
        rollout_samples=rollout_samples,
        root_value_weight=root_value_weight,
        opponent_penalty=opponent_penalty,
    ).name


def _make_policy(
    kind: str,
    checkpoint: str | None,
    *,
    opening_checkpoint: str | None = None,
    opening_prompts: tuple[str, ...] = (),
    candidate_limit: int,
    presearch_candidate_limit: int = 0,
    rollout_decisions: int,
    rollout_samples: int = 1,
    root_value_weight: float = 0.0,
    opponent_penalty: float,
):
    if kind == "random":
        return RandomPolicy()
    if kind == "heuristic":
        return HeuristicPolicy()
    if kind == "jsettlers_lite":
        return JSettlersLitePolicy()
    if kind == "search":
        return OnePlySearchPolicy(
            candidate_limit=candidate_limit,
            rollout_decisions=rollout_decisions,
        )
    if kind == "value_rollout":
        return ValueRolloutSearchPolicy(
            candidate_limit=candidate_limit,
            presearch_candidate_limit=(
                presearch_candidate_limit if presearch_candidate_limit > 0 else None
            ),
            rollout_decisions=rollout_decisions,
            rollout_samples=rollout_samples,
            root_value_weight=root_value_weight,
            opponent_penalty=opponent_penalty,
        )
    if kind == "value":
        return CatanatronValuePolicy(
            candidate_limit=candidate_limit,
            opponent_penalty=opponent_penalty,
        )
    if kind == "linear":
        if checkpoint is None:
            raise SystemExit("--checkpoint is required for --candidate linear")
        return LinearSoftmaxPolicy.load(checkpoint)
    if kind == "mlp":
        if checkpoint is None:
            raise SystemExit("--checkpoint is required for --candidate mlp")
        return NumpyMLPPolicy.load(checkpoint)
    if kind == "ppo":
        if checkpoint is None:
            raise SystemExit("--checkpoint is required for --candidate ppo")
        return TorchPPOPolicy.load(checkpoint)
    if kind == "hybrid_opening_ppo":
        if checkpoint is None:
            raise SystemExit("--checkpoint is required for --candidate hybrid_opening_ppo")
        if opening_checkpoint is None:
            raise SystemExit(
                "--opening-checkpoint is required for --candidate hybrid_opening_ppo"
            )
        return OpeningThenPolicy(
            opening_policy=TorchPPOPolicy.load(opening_checkpoint),
            main_policy=TorchPPOPolicy.load(checkpoint),
            opening_prompts=(
                opening_prompts
                if opening_prompts
                else ("BUILD_INITIAL_SETTLEMENT", "BUILD_INITIAL_ROAD")
            ),
        )
    raise ValueError(kind)


def _parse_opening_prompts(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


if __name__ == "__main__":
    main()
