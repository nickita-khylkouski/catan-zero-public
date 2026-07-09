from __future__ import annotations

import argparse
import random

from catan_zero.rl import CatanZeroGymConfig, CatanZeroGymEnv


def run_random_games(
    num_games: int,
    seed: int,
    players: int,
    max_decisions: int,
    enable_player_trading: bool,
) -> None:
    rng = random.Random(seed)
    env = CatanZeroGymEnv(
        CatanZeroGymConfig(
            players=players,
            representation="vector",
            enable_player_trading=enable_player_trading,
        )
    )
    wins = 0
    losses = 0
    truncated_games = 0
    try:
        for game_idx in range(num_games):
            _, info = env.reset(seed=rng.randrange(2**31))
            done = False
            final_reward = 0.0
            decisions = 0
            while not done and decisions < max_decisions:
                action = rng.choice(info["valid_actions"])
                _, final_reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                decisions += 1
            if not done:
                truncated_games += 1
                print(
                    f"game={game_idx + 1} reward={final_reward:.1f} "
                    f"wins={wins} losses={losses} truncated={truncated_games} "
                    f"decisions={decisions} max_decisions_reached=1"
                )
                continue
            if final_reward > 0:
                wins += 1
            elif final_reward < 0:
                losses += 1
            else:
                truncated_games += 1
            print(
                f"game={game_idx + 1} reward={final_reward:.1f} "
                f"wins={wins} losses={losses} truncated={truncated_games} "
                f"decisions={decisions} max_decisions_reached=0"
            )
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--max-decisions", type=int, default=5000)
    parser.add_argument(
        "--disable-player-trading",
        action="store_true",
        help="Use the simpler Catanatron Gym action surface without domestic trades.",
    )
    args = parser.parse_args()
    run_random_games(
        args.games,
        args.seed,
        args.players,
        args.max_decisions,
        not args.disable_player_trading,
    )


if __name__ == "__main__":
    main()
