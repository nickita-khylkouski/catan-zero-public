#!/usr/bin/env python3
"""CLI: search self-agreement sweep across n_full budgets (task #53, part c).

Measures how often two INDEPENDENT `GumbelChanceMCTS.search(force_full=True)`
runs (same state, same evaluator, different internal RNG seeds) pick the
SAME root action, at n_full = 64, 128, 256, 512 (configurable). This is a
pure search-resolution diagnostic -- no games are played, no winner is
needed -- so it is cheap relative to a full H2H run and answers a narrower
question: at what simulation budget does the search actually converge on a
consistent answer, versus still being dominated by Gumbel-Top-k/Sequential
Halving sampling noise?

Fixed states are sampled from real games played with the checkpoint's own
raw (argmax, no-search) policy, at a spread of decision indices, so the
diagnostic states resemble what self-play will actually see rather than
early-game-only or purely random positions.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from catan_zero.rl.gumbel_self_play import _apply_selected_action
from catan_zero.search.gumbel_chance_mcts import GumbelChanceMCTS, GumbelChanceMCTSConfig
from catan_zero.search.neural_rust_mcts import (
    BatchedEntityGraphRustEvaluator,
    EntityGraphRustEvaluatorConfig,
)
from catan_zero.search.rust_mcts import _require_rust_module
from factory_common import write_json

COLORS: tuple[str, ...] = ("RED", "BLUE")


def _select_raw_action(evaluator: Any, game: Any, legal_actions: tuple[int, ...], *, acting_color: str) -> int:
    if len(legal_actions) == 1:
        return int(legal_actions[0])
    priors, _value = evaluator.evaluate(game, legal_actions, root_color=acting_color, colors=COLORS)
    return int(max(legal_actions, key=lambda action: (float(priors.get(int(action), 0.0)), -int(action))))


def _apply_raw_action(game: Any, action_index: int, *, rng: random.Random) -> Any:
    """Advance the game by a raw-policy-selected action. Reuses
    gumbel_self_play's own live-game chance resolution (correct, already
    tested) rather than reimplementing it -- this diagnostic only cares
    about reaching varied, realistic mid-game states, so the exact chance
    correction flag value doesn't matter; True matches production."""
    return _apply_selected_action(
        game, int(action_index), colors=COLORS, rng=rng, correct_rust_chance_spectra=True
    )


def collect_fixed_states(
    evaluator: Any, *, n_states: int, decisions_per_game: tuple[int, ...], base_seed: int
) -> list[Any]:
    """Play out games with the checkpoint's raw policy, snapshotting a
    state at each of `decisions_per_game`'s decision indices per game."""
    catanatron_rs = _require_rust_module()
    states: list[Any] = []
    game_index = 0
    while len(states) < n_states:
        game_seed = base_seed + game_index
        game = catanatron_rs.Game.simple(list(COLORS), seed=game_seed)
        rng = random.Random(game_seed ^ 0x51A7E)
        decision_index = 0
        snapshots: dict[int, Any] = {}
        target_decisions = set(decisions_per_game)
        max_target = max(decisions_per_game)
        while decision_index <= max_target:
            if game.winning_color() is not None:
                break
            legal = tuple(int(a) for a in game.playable_action_indices(list(COLORS), None))
            if not legal:
                break
            if decision_index in target_decisions:
                snapshots[decision_index] = game.copy()
            acting_color = str(game.current_color())
            selected = _select_raw_action(evaluator, game, legal, acting_color=acting_color)
            game = _apply_raw_action(game, selected, rng=rng)
            decision_index += 1
        states.extend(snapshots[d] for d in sorted(snapshots))
        game_index += 1
    return states[:n_states]


def run_sweep(
    states: list[Any],
    evaluator: Any,
    *,
    n_full_values: tuple[int, ...],
    max_depth: int,
    correct_rust_chance_spectra: bool,
    base_seed: int,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for n_full in n_full_values:
        matches = 0
        total = 0
        per_state: list[dict[str, Any]] = []
        started = time.perf_counter()
        for state_index, state in enumerate(states):
            actions: list[int] = []
            for run_index in range(2):
                seed = base_seed + n_full * 1000 + state_index * 2 + run_index
                config = GumbelChanceMCTSConfig(
                    colors=COLORS,
                    seed=seed,
                    n_full=int(n_full),
                    n_fast=int(n_full),
                    p_full=1.0,
                    max_depth=int(max_depth),
                    temperature=0.0,
                    correct_rust_chance_spectra=correct_rust_chance_spectra,
                )
                mcts = GumbelChanceMCTS(config, evaluator)
                result = mcts.search(state.copy(), force_full=True)
                actions.append(int(result.selected_action))
            agree = actions[0] == actions[1]
            matches += int(agree)
            total += 1
            per_state.append({"state_index": state_index, "actions": actions, "agree": agree})
        elapsed = time.perf_counter() - started
        results[str(n_full)] = {
            "n_full": n_full,
            "states": total,
            "matches": matches,
            "agreement_rate": matches / total if total else None,
            "elapsed_sec": elapsed,
            "per_state": per_state,
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search self-agreement sweep: does gumbel-search converge as n_full grows?"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-states", type=int, default=40)
    parser.add_argument(
        "--decisions-per-game",
        default="20,50,80,110",
        help="comma-separated decision indices to snapshot per generating game",
    )
    parser.add_argument("--n-full-values", default="64,128,256,512")
    parser.add_argument("--max-depth", type=int, default=80)
    parser.add_argument(
        "--correct-rust-chance-spectra", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--base-seed", type=int, default=70001)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    decisions_per_game = tuple(int(x) for x in args.decisions_per_game.split(","))
    n_full_values = tuple(int(x) for x in args.n_full_values.split(","))

    evaluator = BatchedEntityGraphRustEvaluator.from_checkpoint(
        args.checkpoint,
        device=args.device,
        config=EntityGraphRustEvaluatorConfig(),
    )
    try:
        started = time.perf_counter()
        states = collect_fixed_states(
            evaluator,
            n_states=int(args.n_states),
            decisions_per_game=decisions_per_game,
            base_seed=int(args.base_seed),
        )
        collect_elapsed = time.perf_counter() - started

        sweep = run_sweep(
            states,
            evaluator,
            n_full_values=n_full_values,
            max_depth=int(args.max_depth),
            correct_rust_chance_spectra=bool(args.correct_rust_chance_spectra),
            base_seed=int(args.base_seed),
        )
    finally:
        evaluator.close()

    summary = {
        "checkpoint": args.checkpoint,
        "n_states": len(states),
        "decisions_per_game": decisions_per_game,
        "n_full_values": n_full_values,
        "collect_states_elapsed_sec": collect_elapsed,
        "sweep": sweep,
    }
    write_json(args.out, summary)
    print(
        json.dumps(
            {
                "n_states": summary["n_states"],
                "agreement_by_n_full": {
                    key: value["agreement_rate"] for key, value in sweep.items()
                },
                "elapsed_by_n_full": {key: value["elapsed_sec"] for key, value in sweep.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
