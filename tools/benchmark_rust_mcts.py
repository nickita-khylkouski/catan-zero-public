#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

from catan_zero.search import HeuristicRustEvaluator, RustMCTS, RustMCTSConfig
from catan_zero.search.neural_rust_mcts import (
    EntityGraphRustEvaluator,
    EntityGraphRustEvaluatorConfig,
)


COLORS = ("BLUE", "RED")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Direct Rust-engine benchmark for raw/neural-guided MCTS policies."
    )
    parser.add_argument("--mode", choices=("raw_neural", "neural_mcts", "heuristic_mcts"), default="neural_mcts")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opponent", default="value_function")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--simulations", type=int, default=16)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--search-temperature", type=float, default=1.0)
    parser.add_argument("--max-depth", type=int, default=80)
    parser.add_argument("--prior-temperature", type=float, default=1.0)
    parser.add_argument("--value-scale", type=float, default=1.0)
    parser.add_argument("--max-decisions", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    try:
        import catanatron_rs  # type: ignore
    except ImportError as error:
        raise SystemExit("catanatron_rs is not installed in this Python environment") from error

    evaluator: Any
    if args.mode in {"raw_neural", "neural_mcts"}:
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required for raw_neural/neural_mcts")
        evaluator = EntityGraphRustEvaluator.from_checkpoint(
            args.checkpoint,
            device=args.device,
            config=EntityGraphRustEvaluatorConfig(
                value_scale=float(args.value_scale),
                prior_temperature=float(args.prior_temperature),
            ),
        )
    else:
        evaluator = HeuristicRustEvaluator()

    search = RustMCTS(
        RustMCTSConfig(
            colors=COLORS,
            simulations=max(1, int(args.simulations)),
            c_puct=float(args.c_puct),
            max_depth=int(args.max_depth),
            seed=int(args.seed),
            temperature=float(args.search_temperature),
        ),
        evaluator=evaluator,
    )

    results = []
    started = time.perf_counter()
    rng = random.Random(int(args.seed) ^ 0xC47A)
    for game_index in range(int(args.games)):
        candidate_color = COLORS[game_index % 2]
        opponent_color = COLORS[1 - (game_index % 2)]
        player_kinds = [
            "random" if color == candidate_color else str(args.opponent)
            for color in COLORS
        ]
        game = catanatron_rs.Game(
            colors=list(COLORS),
            seed=int(args.seed) + game_index,
            player_kinds=player_kinds,
            vps_to_win=10,
        )
        decisions = 0
        candidate_decisions = 0
        while game.winning_color() is None and decisions < int(args.max_decisions):
            current = str(game.current_color())
            if current == candidate_color:
                legal = tuple(
                    int(action)
                    for action in game.playable_action_indices(list(COLORS), None)
                )
                if not legal:
                    break
                if args.mode == "raw_neural":
                    priors, _value = evaluator.evaluate(
                        game,
                        legal,
                        root_color=current,
                        colors=COLORS,
                    )
                    action = max(legal, key=lambda item: priors.get(int(item), 0.0))
                else:
                    action = search.search(game).action
                game = _apply_action_with_sampled_chance(
                    game,
                    int(action),
                    colors=COLORS,
                    rng=rng,
                )
                candidate_decisions += 1
            else:
                game.play_tick()
            decisions += 1

        winner = game.winning_color()
        candidate_won = str(winner) == candidate_color
        vps = {
            color: _actual_victory_points(json.loads(game.player_state_json(color)))
            for color in COLORS
        }
        result = {
            "game": game_index,
            "seed": int(args.seed) + game_index,
            "candidate_color": candidate_color,
            "opponent_color": opponent_color,
            "winner": str(winner),
            "candidate_won": bool(candidate_won),
            "candidate_vp": int(vps.get(candidate_color, 0)),
            "opponent_vp": int(vps.get(opponent_color, 0)),
            "decisions": int(decisions),
            "candidate_decisions": int(candidate_decisions),
            "timed_out": winner is None,
        }
        results.append(result)
        print(json.dumps({"progress": "rust_mcts_game", **result}), flush=True)

    elapsed = time.perf_counter() - started
    wins = sum(1 for item in results if item["candidate_won"])
    games = len(results)
    summary = {
        "mode": args.mode,
        "checkpoint": args.checkpoint,
        "opponent": args.opponent,
        "c_puct": float(args.c_puct),
        "search_temperature": float(args.search_temperature),
        "max_depth": int(args.max_depth),
        "prior_temperature": float(args.prior_temperature),
        "value_scale": float(args.value_scale),
        "games": games,
        "wins": wins,
        "win_rate": wins / max(1, games),
        "avg_candidate_vp": sum(item["candidate_vp"] for item in results) / max(1, games),
        "avg_opponent_vp": sum(item["opponent_vp"] for item in results) / max(1, games),
        "timeouts": sum(1 for item in results if item["timed_out"]),
        "elapsed_sec": elapsed,
        "games_per_sec": games / elapsed if elapsed > 0 else 0.0,
        "results": results,
    }
    print(json.dumps({"progress": "rust_mcts_summary", **summary}, indent=2), flush=True)
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _actual_victory_points(player_state: dict[str, Any]) -> int:
    return int(
        player_state.get(
            "actual_victory_points",
            player_state.get("victory_points", 0),
        )
        or 0
    )


def _apply_action_with_sampled_chance(
    game: Any,
    action_index: int,
    *,
    colors: tuple[str, ...],
    rng: random.Random,
) -> Any:
    ids = [
        int(action)
        for action in game.playable_action_indices(list(colors), None)
    ]
    actions = json.loads(game.playable_actions_json())
    action_by_id = {action_id: action for action_id, action in zip(ids, actions)}
    action_json = action_by_id.get(int(action_index))
    if action_json is None:
        raise RuntimeError(f"selected action {action_index} is not legal")
    spectrum = json.loads(game.spectrum_json(json.dumps(action_json)))
    if not spectrum:
        game.execute_action_index(int(action_index), list(colors), None)
        return game
    draw = rng.random()
    cumulative = 0.0
    for outcome_index, outcome in enumerate(spectrum):
        cumulative += float(outcome.get("probability", 0.0))
        if draw <= cumulative:
            return game.apply_chance_outcome(json.dumps(action_json), outcome_index)
    return game.apply_chance_outcome(json.dumps(action_json), len(spectrum) - 1)


if __name__ == "__main__":
    main()
