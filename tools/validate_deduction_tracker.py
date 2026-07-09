"""Validation harness for the CAT-59 exact-deduction tracker.

Plays random-legal-action games through `ColonistMultiAgentEnv` (no GPU, no
trained checkpoint needed -- this is a pure game-logic/statistics check), and
for every self-play seat's `DeductionTracker` compares its deduced resource
bounds against the OMNISCIENT ground truth that the engine's own replay
frames retain (per CAT-73 finding #4: `observation_payloads` builds every
player's own exact-hand view, so `frame["observations"][opponent]` already
carries `opponent`'s ground truth -- see `catan_zero.deduction_tracker`'s
module docstring for why only `frame["observations"][self_name]` is ever fed
to the tracker's belief update, keeping the two cleanly separated).

Reports, per game and in aggregate:
  - violations: count of (opponent, decision) pairs where the true hand fell
    OUTSIDE the tracker's bounds. This is a CORRECTNESS invariant -- it must
    be exactly zero for a correctly-implemented tracker; any nonzero count
    here indicates a bug, not merely "low accuracy."
  - exactness_rate: fraction of (opponent, decision) pairs where the tracker
    fully pinned the opponent's hand (`bounds.exact() == true_hand`).
  - mean_bound_width: average total per-resource interval slack
    (`ResourceBounds.width()`) across all checks -- 0 means fully exact.
  - anomalies: count of defensive-fallback triggers (`DeductionTracker.
    anomalies`), which should be zero; each one flags a build/trade
    delta that didn't match the expected fixed-cost recipe (i.e. an
    engine-mechanic case this tracker's model doesn't yet cover).

Usage:
    .venv/bin/python tools/validate_deduction_tracker.py \
        --players 2 --games 50 --seed-base 900000 --max-steps 600 \
        --out runs/deduction_tracker/validation_2p.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from catan_zero.deduction_tracker import DeductionTracker, true_state_label
from catan_zero.rl.multiagent_env import ColonistMultiAgentConfig, ColonistMultiAgentEnv


def _run_random_game(seed: int, players: int, max_steps: int) -> ColonistMultiAgentEnv:
    env = ColonistMultiAgentEnv(ColonistMultiAgentConfig(players=players, vps_to_win=10))
    rng = random.Random(seed)
    _, info = env.reset(seed=seed)
    steps = 0
    while steps < max_steps:
        valid = tuple(int(a) for a in info.get("valid_actions", ()))
        if not valid:
            break
        action = rng.choice(valid)
        _, _, terminated, truncated, info = env.step(action)
        steps += 1
        if terminated or truncated:
            break
    return env


def _validate_one_game(seed: int, players: int, max_steps: int) -> dict[str, Any]:
    env = _run_random_game(seed, players, max_steps)
    names = env.player_names
    checks = 0
    violations = 0
    exact_hits = 0
    width_total = 0
    anomalies = 0
    violation_examples: list[dict[str, Any]] = []

    for self_name in names:
        opponents = tuple(name for name in names if name != self_name)
        tracker = DeductionTracker(self_name=self_name, opponent_names=opponents)
        frames = env.replay_trace(actor=self_name)
        for i in range(1, len(frames)):
            if i == 1:
                tracker.observe_frames([frames[0], frames[1]])
            else:
                tracker.observe_frames([frames[i]])
            for opponent in opponents:
                true_hand = true_state_label(frames[i]["observations"][opponent], opponent)
                if true_hand is None:
                    continue
                bounds = tracker.bounds_for(opponent)
                total = sum(true_hand["resources"].values())
                checks += 1
                width_total += bounds.width()
                if bounds.exact() == true_hand["resources"]:
                    exact_hits += 1
                if not bounds.contains(true_hand["resources"], total=total):
                    violations += 1
                    if len(violation_examples) < 5:
                        violation_examples.append(
                            {
                                "self": self_name,
                                "opponent": opponent,
                                "frame": i,
                                "true_resources": true_hand["resources"],
                                "lower": dict(bounds.lower),
                                "upper": dict(bounds.upper),
                            }
                        )
        anomalies += len(tracker.anomalies)

    env.close()
    return {
        "seed": seed,
        "players": players,
        "checks": checks,
        "violations": violations,
        "exact_hits": exact_hits,
        "width_total": width_total,
        "anomalies": anomalies,
        "violation_examples": violation_examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--players", type=int, default=2, help="players per game (2/3/4)")
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--seed-base", type=int, default=900_000)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    per_game = [
        _validate_one_game(args.seed_base + i, args.players, args.max_steps)
        for i in range(args.games)
    ]
    checks = sum(g["checks"] for g in per_game)
    violations = sum(g["violations"] for g in per_game)
    exact_hits = sum(g["exact_hits"] for g in per_game)
    width_total = sum(g["width_total"] for g in per_game)
    anomalies = sum(g["anomalies"] for g in per_game)

    report = {
        "players": args.players,
        "games": args.games,
        "seed_base": args.seed_base,
        "max_steps": args.max_steps,
        "checks": checks,
        "violations": violations,
        "exactness_rate": (exact_hits / checks) if checks else None,
        "mean_bound_width": (width_total / checks) if checks else None,
        "anomalies": anomalies,
        "games_with_violations": [g["seed"] for g in per_game if g["violations"]],
        "violation_examples": [
            example for g in per_game for example in g["violation_examples"]
        ][:20],
    }
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    if violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
