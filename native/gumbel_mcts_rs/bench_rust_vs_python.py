#!/usr/bin/env python3
"""Benchmark: Rust MCTS vs Python MCTS.

Run on a GPU box with the champion checkpoint:

    python3 bench_rust_vs_python.py --checkpoint /path/to/checkpoint.pt --device cuda:0

Measures per-game wall time for both implementations and reports the speedup.
"""

import argparse
import sys
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--games", type=int, default=3)
    parser.add_argument("--n-full", type=int, default=64)
    parser.add_argument("--c-scale", type=float, default=0.03)
    args = parser.parse_args()

    sys.path.insert(0, "src")

    from catan_zero.rl.entity_token_policy import EntityGraphPolicy
    from catan_zero.search.neural_rust_mcts import (
        EntityGraphRustEvaluator,
        EntityGraphRustEvaluatorConfig,
    )
    from catan_zero.search.gumbel_chance_mcts import (
        GumbelChanceMCTS,
        GumbelChanceMCTSConfig,
    )

    # Try to import the Rust version
    try:
        from gumbel_mcts_rust import GumbelChanceMCTSRust
        rust_available = True
        print("Rust extension: AVAILABLE")
    except ImportError:
        rust_available = False
        print("Rust extension: NOT AVAILABLE (build with: cd gumbel_mcts_rs && bash build.sh release)")

    # Setup
    eval_config = EntityGraphRustEvaluatorConfig(
        value_scale=1.0,
        prior_temperature=1.0,
        public_observation=True,
        cache_size=0,
    )
    evaluator = EntityGraphRustEvaluator.from_checkpoint(
        args.checkpoint, device=args.device, config=eval_config
    )

    search_config = GumbelChanceMCTSConfig(
        c_visit=50.0,
        c_scale=args.c_scale,
        n_full=args.n_full,
        n_fast=16,
        p_full=0.25,
        max_depth=80,
        correct_rust_chance_spectra=True,
        lazy_interior_chance=True,
        colors=("BLUE", "RED"),
    )

    # Python MCTS
    py_mcts = GumbelChanceMCTS(search_config, evaluator)

    # Rust MCTS (if available)
    if rust_available:
        rust_mcts = GumbelChanceMCTSRust(search_config, evaluator)

    # Benchmark
    for label, mcts in [("Python", py_mcts), ("Rust", rust_mcts) if rust_available else [("Python", py_mcts)]]:
        if mcts is None:
            continue
        print(f"\n=== {label} MCTS ===")
        total_time = 0
        total_sims = 0
        total_decisions = 0

        for game_idx in range(args.games):
            game = mcts.new_game(seed=42 + game_idx)
            t0 = time.perf_counter()
            decisions = 0
            while True:
                winner = game.winning_color()
                if winner is not None:
                    break
                result = mcts.search(game)
                # Apply the selected action
                import json
                actions = json.loads(game.playable_actions_json())
                action = None
                for i, a in enumerate(actions):
                    if i == result.selected_action:
                        action = a
                        break
                if action is None:
                    break
                game.execute_json(json.dumps(action))
                decisions += 1

            t1 = time.perf_counter()
            elapsed = t1 - t0
            total_time += elapsed
            total_decisions += decisions
            print(f"  Game {game_idx}: {elapsed:.2f}s, {decisions} decisions")

        avg = total_time / args.games
        avg_dec = total_decisions / args.games
        print(f"  Average: {avg:.2f}s/game, {avg_dec:.0f} decisions, {1000*avg/avg_dec:.0f}ms/decision")

    if rust_available:
        # Compute speedup
        print("\n=== Speedup ===")
        # Re-run Python for fair comparison
        py_times = []
        for game_idx in range(args.games):
            game = py_mcts.new_game(seed=42 + game_idx)
            t0 = time.perf_counter()
            while True:
                winner = game.winning_color()
                if winner is not None:
                    break
                result = py_mcts.search(game)
                import json
                actions = json.loads(game.playable_actions_json())
                action = None
                for i, a in enumerate(actions):
                    if i == result.selected_action:
                        action = a
                        break
                if action is None:
                    break
                game.execute_json(json.dumps(action))
            t1 = time.perf_counter()
            py_times.append(t1 - t0)

        rust_times = []
        for game_idx in range(args.games):
            game = rust_mcts.new_game(seed=42 + game_idx)
            t0 = time.perf_counter()
            while True:
                winner = game.winning_color()
                if winner is not None:
                    break
                result = rust_mcts.search(game)
                import json
                actions = json.loads(game.playable_actions_json())
                action = None
                for i, a in enumerate(actions):
                    if i == result.selected_action:
                        action = a
                        break
                if action is None:
                    break
                game.execute_json(json.dumps(action))
            t1 = time.perf_counter()
            rust_times.append(t1 - t0)

        avg_py = sum(py_times) / len(py_times)
        avg_rust = sum(rust_times) / len(rust_times)
        print(f"  Python: {avg_py:.2f}s/game")
        print(f"  Rust:   {avg_rust:.2f}s/game")
        print(f"  Speedup: {avg_py / avg_rust:.2f}x")


if __name__ == "__main__":
    main()
