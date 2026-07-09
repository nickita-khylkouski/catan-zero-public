from __future__ import annotations

import argparse
import random

from catan_zero.rl import ColonistMultiAgentConfig, ColonistMultiAgentEnv


def run_random_games(num_games: int, seed: int, players: int, max_decisions: int) -> None:
    rng = random.Random(seed)
    env = ColonistMultiAgentEnv(ColonistMultiAgentConfig(players=players))
    wins = {name: 0 for name in env.player_names}
    truncated_games = 0
    try:
        for game_idx in range(num_games):
            _, info = env.reset(seed=rng.randrange(2**31))
            terminated = False
            truncated = False
            decisions = 0
            rewards = {name: 0.0 for name in env.player_names}

            while not (terminated or truncated) and decisions < max_decisions:
                action = rng.choice(info["valid_actions"])
                _, rewards, terminated, truncated, info = env.step(action)
                decisions += 1

            if not terminated:
                truncated_games += 1
                max_decisions_reached = int(decisions >= max_decisions and not truncated)
                winner = "none"
            else:
                max_decisions_reached = 0
                winner = max(rewards, key=rewards.get)
                wins[winner] += 1

            print(
                f"game={game_idx + 1} winner={winner} decisions={decisions} "
                f"terminated={int(terminated)} truncated={int(truncated or max_decisions_reached)} "
                f"max_decisions_reached={max_decisions_reached} wins={wins}"
            )
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--max-decisions", type=int, default=5000)
    args = parser.parse_args()
    run_random_games(args.games, args.seed, args.players, args.max_decisions)


if __name__ == "__main__":
    main()
