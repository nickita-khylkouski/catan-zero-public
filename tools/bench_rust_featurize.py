"""Micro-benchmark: native Rust entity featurizer vs the Python per-token path
(CAT-65 / task #81 verification item (c)).

Times `entity_token_features_rust.build_entity_features_rust` against the
reference `neural_rust_mcts.rust_game_to_entity_batch` (which routes through the
Python `entity_token_features.build_entity_token_features`) on a fixed set of
real random-vs-random game states spanning opening/mid/late/robber/discard
decisions. CPU-only; measures the featurize slice in isolation (no NN forward),
which the perf model puts at ~96% of leaf cost.

Not a correctness check -- `tests/test_rust_featurize_parity.py` proves bit-exact
equivalence. This only reports the speedup so the "20-38x on the featurize slice"
claim can be re-confirmed in the landed, integrated form on any host.

Usage:
    python tools/bench_rust_featurize.py [--states 200] [--repeat 3] [--seed 3000]
"""
from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

import catanatron_rs

from catan_zero.rl.entity_token_features_rust import (
    build_entity_features_rust,
    compute_rust_topology,
)
from catan_zero.search.neural_rust_mcts import (
    _RustEntityFeatureEnv,
    _resolve_entity_adapter,
    rust_game_to_entity_batch,
    rust_policy_action_ids,
)

COLORS: tuple[str, ...] = ("RED", "BLUE")
ACTION_SIZE = 400


def _collect_states(num_games: int, max_ticks: int, seed_base: int):
    states = []
    for i in range(num_games):
        game = catanatron_rs.Game.random(colors=list(COLORS), seed=seed_base + i)
        ticks = 0
        while game.winning_color() is None and ticks < max_ticks:
            legal = game.playable_action_indices(list(COLORS), None)
            if legal:
                states.append(game.copy())
            game.play_tick()
            ticks += 1
    return states


def _prep(game, public_observation: bool):
    actor = game.current_color()
    legal = tuple(int(a) for a in game.playable_action_indices(list(COLORS), None))
    if not legal:
        return None
    policy_action_ids = rust_policy_action_ids(
        game, legal, colors=COLORS, action_size=ACTION_SIZE
    )
    resolved = _resolve_entity_adapter(
        game,
        legal,
        colors=COLORS,
        action_size=ACTION_SIZE,
        policy_action_ids=policy_action_ids,
        snapshot=None,
        action_by_id=None,
        public_observation=public_observation,
        perspective=actor,
    )
    adapter_env = _RustEntityFeatureEnv(resolved[0], action_size=ACTION_SIZE)
    topology = compute_rust_topology(adapter_env, actor)
    return actor, legal, policy_action_ids, resolved, topology


def _time_python(prepped, game, public_observation):
    actor, legal, policy_action_ids, resolved, _ = prepped
    t0 = time.perf_counter()
    rust_game_to_entity_batch(
        game,
        legal,
        actor=actor,
        colors=COLORS,
        action_size=ACTION_SIZE,
        policy_action_ids=policy_action_ids,
        public_observation=public_observation,
        resolved=resolved,
    )
    return time.perf_counter() - t0


def _time_rust(prepped, game, public_observation):
    _, _, policy_action_ids, _, topology = prepped
    t0 = time.perf_counter()
    build_entity_features_rust(
        game,
        colors=COLORS,
        policy_action_ids=policy_action_ids,
        action_size=ACTION_SIZE,
        topology=topology,
        public_observation=public_observation,
    )
    return time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=int, default=200)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--seed", type=int, default=3000)
    ap.add_argument("--public-observation", action="store_true")
    args = ap.parse_args()

    print(f"collecting states (target {args.states})...")
    games = _collect_states(num_games=30, max_ticks=200, seed_base=args.seed)
    np.random.default_rng(0).shuffle(games)
    games = games[: args.states]
    prepped = [(_prep(g, args.public_observation), g) for g in games]
    prepped = [(p, g) for (p, g) in prepped if p is not None]
    print(f"benchmarking on {len(prepped)} states, {args.repeat} repeats each\n")

    py_times, rs_times = [], []
    for _ in range(args.repeat):
        for p, g in prepped:
            py_times.append(_time_python(p, g, args.public_observation))
            rs_times.append(_time_rust(p, g, args.public_observation))

    py_us = statistics.mean(py_times) * 1e6
    rs_us = statistics.mean(rs_times) * 1e6
    py_med = statistics.median(py_times) * 1e6
    rs_med = statistics.median(rs_times) * 1e6
    print(f"Python featurize:  mean {py_us:8.1f} us   median {py_med:8.1f} us")
    print(f"Rust   featurize:  mean {rs_us:8.1f} us   median {rs_med:8.1f} us")
    print(f"speedup (mean):    {py_us / rs_us:6.1f}x")
    print(f"speedup (median):  {py_med / rs_med:6.1f}x")


if __name__ == "__main__":
    main()
