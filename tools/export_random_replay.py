from __future__ import annotations

import argparse
import random
from pathlib import Path

from catan_zero.rl import ColonistMultiAgentConfig, ColonistMultiAgentEnv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a local Colonist-like random-policy replay JSONL."
    )
    parser.add_argument("--output", required=True, help="Path to write JSONL replay")
    parser.add_argument("--players", type=int, default=4, choices=(2, 3, 4))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-decisions", type=int, default=1000)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    env = ColonistMultiAgentEnv(ColonistMultiAgentConfig(players=args.players))
    try:
        _, info = env.reset(seed=args.seed)
        decisions = 0
        terminated = False
        truncated = False
        while decisions < args.max_decisions and not (terminated or truncated):
            action = rng.choice(info["valid_actions"])
            _, _, terminated, truncated, info = env.step(action)
            decisions += 1

        output = Path(args.output)
        frames = env.write_replay_jsonl(
            output,
            metadata={
                "generator": "tools/export_random_replay.py",
                "seed": args.seed,
                "decisions": decisions,
                "terminated": terminated,
                "truncated": truncated,
            },
        )
        print(
            " ".join(
                [
                    f"path={output}",
                    f"frames={frames}",
                    f"decisions={decisions}",
                    f"terminated={int(terminated)}",
                    f"truncated={int(truncated)}",
                ]
            )
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
