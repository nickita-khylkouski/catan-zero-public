from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import time

import numpy as np

from catan_zero.rl import RandomPolicy, play_game
from catan_zero.rl.multiagent_env import ColonistMultiAgentEnv
from factory_common import parse_track, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify deterministic/legal fast Catan env runs.")
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--track", default="4p_bank_trade")
    parser.add_argument("--vps-to-win", type=int, default=10)
    parser.add_argument("--max-decisions", type=int, default=1000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    start = time.perf_counter()
    chunks = _chunks(list(range(args.games)), max(1, args.workers))
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        results = list(
            executor.map(
                _verify_chunk,
                [
                    {
                        "indices": chunk,
                        "seed": args.seed,
                        "track": args.track,
                        "vps_to_win": args.vps_to_win,
                        "max_decisions": args.max_decisions,
                    }
                    for chunk in chunks
                    if chunk
                ],
            )
        )
    merged = {
        "games": args.games,
        "workers": args.workers,
        "track": args.track,
        "seed": args.seed,
        "elapsed_sec": time.perf_counter() - start,
        "checks": {
            "deterministic_winner": True,
            "illegal_actions": 0,
            "negative_resources": 0,
            "empty_legal_mask_before_terminal": 0,
            "stuck_games": 0,
            "invalid_terminal_winner": 0,
            "seat_rotation_failures": 0,
        },
    }
    for result in results:
        for key, value in result["checks"].items():
            if isinstance(value, bool):
                merged["checks"][key] = bool(merged["checks"][key] and value)
            else:
                merged["checks"][key] += int(value)
    merged["ok"] = (
        merged["checks"]["deterministic_winner"]
        and all(
            int(value) == 0
            for key, value in merged["checks"].items()
            if key != "deterministic_winner"
        )
    )
    write_json(args.out, merged)
    print(merged)


def _verify_chunk(payload: dict) -> dict:
    config = parse_track(payload["track"], vps_to_win=int(payload["vps_to_win"]))
    policy = RandomPolicy()
    policies = {name: policy for name in ("BLUE", "RED", "ORANGE", "WHITE")[: config.players]}
    checks = {
        "deterministic_winner": True,
        "illegal_actions": 0,
        "negative_resources": 0,
        "empty_legal_mask_before_terminal": 0,
        "stuck_games": 0,
        "invalid_terminal_winner": 0,
        "seat_rotation_failures": 0,
    }
    for index in payload["indices"]:
        seed = int(payload["seed"]) + int(index)
        first = play_game(policies, seed=seed, config=config, max_decisions=payload["max_decisions"])
        second = play_game(policies, seed=seed, config=config, max_decisions=payload["max_decisions"])
        checks["deterministic_winner"] = checks["deterministic_winner"] and (
            first.result.winner == second.result.winner
        )
        step_checks = _verify_live_game(
            seed=seed,
            config=config,
            max_decisions=int(payload["max_decisions"]),
        )
        for key, value in step_checks.items():
            checks[key] += int(value)
        for episode in (first,):
            checks["illegal_actions"] += int(episode.result.invalid_actions)
            checks["stuck_games"] += int(episode.result.truncated and not episode.result.terminated)
            if episode.result.winner is not None and episode.result.winner not in policies:
                checks["invalid_terminal_winner"] += 1
            if len(set(episode.result.final_public_vps)) != config.players:
                checks["seat_rotation_failures"] += 1
            if any(vp < 0 for vp in episode.result.final_public_vps.values()):
                checks["negative_resources"] += 1
    return {"checks": checks}


def _verify_live_game(
    *,
    seed: int,
    config,
    max_decisions: int,
) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    policy = RandomPolicy()
    checks = {
        "illegal_actions": 0,
        "negative_resources": 0,
        "empty_legal_mask_before_terminal": 0,
    }
    env = ColonistMultiAgentEnv(config)
    try:
        observations, info = env.reset(seed=seed)
        terminated = truncated = False
        decisions = 0
        while not (terminated or truncated) and decisions < max_decisions:
            valid = tuple(int(action) for action in info.get("valid_actions", ()))
            if not valid:
                checks["empty_legal_mask_before_terminal"] += 1
                break
            payload = env.observation_payload(info["current_player"], include_event_log=False)
            checks["negative_resources"] += _negative_count_fields(payload)
            action = policy.select_action(
                env,
                np.asarray(observations[info["current_player"]], dtype=np.float64),
                info,
                rng,
                training=False,
            )
            observations, _rewards, terminated, truncated, info = env.step(action)
            checks["illegal_actions"] = int(info.get("invalid_actions_count", 0))
            decisions += 1
    finally:
        env.close()
    return checks


def _negative_count_fields(value) -> int:
    if isinstance(value, dict):
        total = 0
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered in {
                "wood",
                "brick",
                "sheep",
                "wheat",
                "ore",
                "knight",
                "year_of_plenty",
                "monopoly",
                "road_building",
                "victory_point",
                "resource_card_count",
                "development_card_count",
                "public_victory_points",
                "actual_victory_points",
                "roads_left",
                "settlements_left",
                "cities_left",
            } and isinstance(child, (int, float)):
                total += int(child < 0)
            else:
                total += _negative_count_fields(child)
        return total
    if isinstance(value, (list, tuple)):
        return sum(_negative_count_fields(child) for child in value)
    return 0


def _chunks(items: list[int], count: int) -> list[list[int]]:
    chunks = [[] for _ in range(max(1, count))]
    for index, item in enumerate(items):
        chunks[index % len(chunks)].append(item)
    return chunks


if __name__ == "__main__":
    main()
