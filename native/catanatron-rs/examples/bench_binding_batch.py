#!/usr/bin/env python3
"""Benchmark old vs new binding call patterns for MCTS node expansion.

Mirrors the rust_mcts.py hot loop at a chance (ROLL) decision node:
  OLD: playable_action_indices + playable_actions_json + spectrum_json per
       chance action + 11x apply_chance_outcome to materialize ROLL children
  NEW: decision_context_json + apply_chance_outcomes_batch (one call each)
"""

import json
import statistics
import time

import catanatron_rs

COLORS = ["RED", "BLUE"]


def collect_roll_nodes(count, seed0=100):
    nodes = []
    seed = seed0
    while len(nodes) < count:
        game = catanatron_rs.Game.simple(COLORS, seed=seed)
        seed += 1
        for _ in range(600):
            if game.winning_color() is not None:
                break
            actions = json.loads(game.playable_actions_json())
            if any(a[1] == "ROLL" for a in actions):
                nodes.append(game.copy())
                if len(nodes) >= count:
                    break
            game.play_tick()
    return nodes


def old_flow(game):
    ids = [int(i) for i in game.playable_action_indices(COLORS, None)]
    actions = json.loads(game.playable_actions_json())
    by_index = dict(zip(ids, actions))
    children = 0
    for action_id, action in by_index.items():
        if action[1] not in ("ROLL", "BUY_DEVELOPMENT_CARD", "MOVE_ROBBER"):
            continue
        raw = json.loads(game.spectrum_json(json.dumps(action)))
        probs = [entry["probability"] for entry in raw]
        for i in range(len(probs)):
            child = game.apply_chance_outcome(json.dumps(action), i)
            children += 1
    return children


def new_flow(game):
    context = json.loads(game.decision_context_json(COLORS, None))
    children = 0
    for entry in context["actions"]:
        spectrum = entry.get("spectrum")
        if spectrum is None:
            continue
        batch = game.apply_chance_outcomes_batch(json.dumps(entry["action"]))
        children += len(batch)
    return children


def bench(fn, nodes, repeats=3):
    times = []
    total_children = 0
    for _ in range(repeats):
        start = time.perf_counter()
        total_children = sum(fn(g) for g in nodes)
        times.append(time.perf_counter() - start)
    best = min(times)
    return best, total_children


def main():
    nodes = collect_roll_nodes(60)
    print(f"collected {len(nodes)} ROLL decision nodes")

    # sanity: both flows materialize the same number of chance children
    old_children = old_flow(nodes[0])
    new_children = new_flow(nodes[0])
    assert old_children == new_children, (old_children, new_children)

    old_time, old_total = bench(old_flow, nodes)
    new_time, new_total = bench(new_flow, nodes)
    assert old_total == new_total, (old_total, new_total)

    per_old = old_time / len(nodes) * 1e6
    per_new = new_time / len(nodes) * 1e6
    print(f"chance children per node set: {old_total}")
    print(f"OLD full-expansion flow: {per_old:9.1f} us/node")
    print(f"NEW full-expansion flow: {per_new:9.1f} us/node")
    print(f"speedup: {per_old / per_new:.2f}x")

    # context-only comparison (per simulate() call that only needs priors)
    def old_context(game):
        ids = [int(i) for i in game.playable_action_indices(COLORS, None)]
        actions = json.loads(game.playable_actions_json())
        n = 0
        for action in actions:
            if action[1] in ("ROLL", "BUY_DEVELOPMENT_CARD", "MOVE_ROBBER"):
                raw = json.loads(game.spectrum_json(json.dumps(action)))
                n += len(raw)
        return n

    def new_context(game):
        context = json.loads(game.decision_context_json(COLORS, None))
        return sum(len(e.get("spectrum", [])) for e in context["actions"])

    assert old_context(nodes[0]) == new_context(nodes[0])
    old_ctx, _ = bench(old_context, nodes)
    new_ctx, _ = bench(new_context, nodes)
    print()
    print(f"OLD context (indices+actions+spectra): {old_ctx / len(nodes) * 1e6:9.1f} us/node")
    print(f"NEW context (decision_context_json):   {new_ctx / len(nodes) * 1e6:9.1f} us/node")
    print(f"context speedup: {old_ctx / new_ctx:.2f}x")

    # single sampled-child pattern (rust_mcts._simulate): spectrum + 1 child
    def old_sample(game):
        actions = json.loads(game.playable_actions_json())
        action = next(a for a in actions if a[1] == "ROLL")
        raw = json.loads(game.spectrum_json(json.dumps(action)))
        child = game.apply_chance_outcome(json.dumps(action), len(raw) // 2)
        return child.state_index()

    def new_sample(game):
        context = json.loads(game.decision_context_json(COLORS, None))
        entry = next(e for e in context["actions"] if e["action"][1] == "ROLL")
        child = game.apply_chance_outcomes_batch(
            json.dumps(entry["action"]), [len(entry["spectrum"]) // 2]
        )[0]
        return child.state_index()

    assert old_sample(nodes[0]) == new_sample(nodes[0])
    old_s, _ = bench(old_sample, nodes)
    new_s, _ = bench(new_sample, nodes)
    print()
    print(f"OLD sampled-child step: {old_s / len(nodes) * 1e6:9.1f} us/node")
    print(f"NEW sampled-child step: {new_s / len(nodes) * 1e6:9.1f} us/node")
    print(f"sampled-child speedup: {old_s / new_s:.2f}x")


if __name__ == "__main__":
    main()
