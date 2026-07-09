from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import time

import numpy as np

from catan_zero.rl import RandomPolicy
from catan_zero.rl.action_features import build_action_context_feature_table
from catan_zero.rl.multiagent_env import ColonistMultiAgentEnv
from factory_common import load_checkpoint_policy, parse_track, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Component benchmark for CatanZero on GH200.")
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--workers", type=int, default=1)  # Reserved for CLI compatibility.
    parser.add_argument("--players", choices=("random", "neural"), default="random")
    parser.add_argument("--checkpoint")
    parser.add_argument("--track", default="4p_bank_trade")
    parser.add_argument("--vps-to-win", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-decisions", type=int, default=1000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    start = time.perf_counter()
    if args.workers > 1:
        chunks = _chunks(list(range(args.games)), args.workers)
        payloads = [
            {
                "indices": chunk,
                "players": args.players,
                "checkpoint": args.checkpoint,
                "track": args.track,
                "vps_to_win": args.vps_to_win,
                "seed": args.seed,
                "max_decisions": args.max_decisions,
            }
            for chunk in chunks
            if chunk
        ]
        with ProcessPoolExecutor(max_workers=min(args.workers, len(payloads))) as executor:
            parts = list(executor.map(_run_chunk, payloads))
        totals = _merge_timings(part["timing"] for part in parts)
        decisions = sum(int(part["decisions"]) for part in parts)
        wins = sum(int(part["wins"]) for part in parts)
    else:
        part = _run_chunk(
            {
                "indices": list(range(args.games)),
                "players": args.players,
                "checkpoint": args.checkpoint,
                "track": args.track,
                "vps_to_win": args.vps_to_win,
                "seed": args.seed,
                "max_decisions": args.max_decisions,
            }
        )
        totals = part["timing"]
        decisions = int(part["decisions"])
        wins = int(part["wins"])
    elapsed = time.perf_counter() - start
    report = {
        "games": args.games,
        "workers": args.workers,
        "players": args.players,
        "track": args.track,
        "wins": wins,
        "decisions": decisions,
        "elapsed_sec": elapsed,
        "games_per_second": args.games / elapsed if elapsed else 0.0,
        "decisions_per_second": decisions / elapsed if elapsed else 0.0,
        "timing": totals,
    }
    write_json(args.out, report)
    print(report)


def _run_chunk(payload: dict) -> dict:
    config = parse_track(payload["track"], vps_to_win=int(payload["vps_to_win"]))
    policy = (
        load_checkpoint_policy(payload["checkpoint"], device="auto")
        if payload["players"] == "neural"
        else RandomPolicy()
    )
    rng = np.random.default_rng(int(payload["seed"]))
    totals = {
        "env_step_sec": 0.0,
        "legal_action_generation_sec": 0.0,
        "observation_encoding_sec": 0.0,
        "policy_forward_sec": 0.0,
        "reward_done_sec": 0.0,
        "logging_sec": 0.0,
    }
    decisions = 0
    wins = 0
    for game_index in payload["indices"]:
        env = ColonistMultiAgentEnv(config)
        try:
            t = time.perf_counter()
            observations, info = env.reset(seed=int(payload["seed"]) + int(game_index))
            totals["logging_sec"] += time.perf_counter() - t
            terminated = truncated = False
            game_decisions = 0
            while not (terminated or truncated) and game_decisions < int(payload["max_decisions"]):
                actor = info["current_player"]
                t = time.perf_counter()
                valid_actions = tuple(int(action) for action in info["valid_actions"])
                totals["legal_action_generation_sec"] += time.perf_counter() - t
                t = time.perf_counter()
                observation = np.asarray(observations[actor], dtype=np.float32)
                context = build_action_context_feature_table(env, info)
                totals["observation_encoding_sec"] += time.perf_counter() - t
                t = time.perf_counter()
                if payload["players"] == "neural" and hasattr(policy, "sample_action_value"):
                    action = policy.sample_action_value(observation, valid_actions, context)[0]
                else:
                    action = policy.select_action(env, observation, info, rng, training=False)
                totals["policy_forward_sec"] += time.perf_counter() - t
                t = time.perf_counter()
                observations, rewards, terminated, truncated, info = env.step(action)
                totals["env_step_sec"] += time.perf_counter() - t
                t = time.perf_counter()
                if terminated and any(value > 0 for value in rewards.values()):
                    wins += 1
                totals["reward_done_sec"] += time.perf_counter() - t
                decisions += 1
                game_decisions += 1
        finally:
            env.close()
    return {
        "wins": wins,
        "decisions": decisions,
        "timing": totals,
    }


def _merge_timings(parts) -> dict[str, float]:
    merged: dict[str, float] = {}
    for part in parts:
        for key, value in part.items():
            merged[key] = merged.get(key, 0.0) + float(value)
    return merged


def _chunks(items: list[int], count: int) -> list[list[int]]:
    chunks = [[] for _ in range(max(1, count))]
    for index, item in enumerate(items):
        chunks[index % len(chunks)].append(item)
    return chunks


if __name__ == "__main__":
    main()
