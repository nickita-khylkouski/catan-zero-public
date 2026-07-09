#!/usr/bin/env python3
"""Aggregate the post-repair H2H bracket (task #53 / protocol step 3/4).

Team-lead's fleet fans each arm out across MULTIPLE independent invocations
of tools/gumbel_search_vs_raw_h2h.py (different --base-seed blocks, one file
per GPU/host). Each file's internal `pair_id` is LOCAL to that invocation
(it resets to 0..pairs-1 every run), so pair_id is NOT a safe cross-file
join key -- pair_id=0 in one file and pair_id=0 in another refer to
completely different games. `game_seed` (== base_seed + pair_id within one
run) IS globally unique across the whole fleet as long as the base-seed
blocks don't overlap, which is how they were provisioned here. This script
re-derives the concordant-pair grouping by game_seed instead of pair_id,
then re-runs the same WW/LL/split/incomplete reduction and SPRT evaluation
that the single-file tool itself uses, over the pooled game list.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))
import atomic_io  # noqa: E402
from typing import Any

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from sprt_gate import GATE_CONFIGS, evaluate_sprt, resolve_gate_config


def _dedupe_games(games: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Drop BIT-IDENTICAL duplicate games. The chance RNG is seeded by
    game_seed ^ 0xA17E and search is deterministic in the config, so running
    the SAME (game_seed, orientation) with the same net/config twice yields an
    identical game. The fleet's seed-collision bug did exactly this: a seed
    block running on multiple hosts double-counted games bit-for-bit into a
    false-significant verdict (ported from h2h_v3conf_aggregate.py's
    `_dedupe_games`, added after that exact incident). A game is unique per
    (game_seed, orientation, search_color, winner, search_won, decisions);
    only EXACT matches are dropped, so genuinely different games (e.g. a
    different search seed producing a different result) are kept."""
    seen: set[tuple] = set()
    unique: list[dict[str, Any]] = []
    dropped = 0
    for g in games:
        sig = (
            g.get("game_seed"),
            g.get("orientation"),
            g.get("search_color"),
            g.get("winner"),
            g.get("search_won"),
            g.get("decisions"),
        )
        if sig in seen:
            dropped += 1
            continue
        seen.add(sig)
        unique.append(g)
    return unique, dropped


def _pooled_pair_outcomes(games: list[dict[str, Any]]) -> tuple[list[bool], dict[str, int]]:
    """Same WW/LL/split/incomplete reduction as the single-file tool's
    `_concordant_pair_outcomes`, but keyed by game_seed (safe across files)
    instead of pair_id (only safe within one file).

    Bit-identical duplicate games are dropped FIRST (see `_dedupe_games` --
    handles the fleet seed-collision replication) before pairing by
    game_seed, so a seed collision cannot double-count a pair."""
    games, n_dupes = _dedupe_games(games)
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for game in games:
        by_seed.setdefault(int(game["game_seed"]), []).append(game)

    outcomes: list[bool] = []
    diagnostics = {
        "ww_pairs": 0,
        "ll_pairs": 0,
        "split_pairs": 0,
        "incomplete_pairs": 0,
        "duplicate_games_dropped": n_dupes,
    }
    for seed_games in by_seed.values():
        if len(seed_games) != 2 or any(game["search_won"] is None for game in seed_games):
            diagnostics["incomplete_pairs"] += 1
            continue
        results = {bool(game["search_won"]) for game in seed_games}
        if results == {True}:
            outcomes.append(True)
            diagnostics["ww_pairs"] += 1
        elif results == {False}:
            outcomes.append(False)
            diagnostics["ll_pairs"] += 1
        else:
            diagnostics["split_pairs"] += 1
    return outcomes, diagnostics


def load_games(paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_games: list[dict[str, Any]] = []
    config: dict[str, Any] = {}
    for path in paths:
        data = json.loads(path.read_text())
        all_games.extend(data.get("games", ()))
        # sanity: config fields (n_full, c_scale, c_visit, max_decisions,
        # max_root_candidates_wide) must agree across all files in one arm.
        for key in ("n_full", "c_scale", "c_visit", "max_root_candidates_wide", "max_decisions"):
            if key in data:
                config.setdefault(key, data[key])
                if config[key] != data[key]:
                    raise ValueError(
                        f"config mismatch for {key} in {path}: {data[key]!r} != {config[key]!r}"
                    )
    return all_games, config


def aggregate_arm(paths: list[Path], *, elo0: float, elo1: float) -> dict[str, Any]:
    games, config = load_games(paths)
    seed_count = len({int(g["game_seed"]) for g in games})
    game_count = len(games)
    outcomes, diagnostics = _pooled_pair_outcomes(games)
    pair_sprt = evaluate_sprt(outcomes=outcomes, elo0=elo0, elo1=elo1)
    win_rate = (sum(1 for o in outcomes if o) / len(outcomes)) if outcomes else None
    return {
        "files": [str(p) for p in paths],
        "config": config,
        "games_played": game_count,
        "duplicate_games_dropped": diagnostics["duplicate_games_dropped"],
        "distinct_game_seeds": seed_count,
        "pairs_decisive": diagnostics["ww_pairs"] + diagnostics["ll_pairs"],
        "pairs_split_excluded": diagnostics["split_pairs"],
        "pairs_truncated_excluded": diagnostics["incomplete_pairs"],
        "search_wins": diagnostics["ww_pairs"],
        "raw_wins": diagnostics["ll_pairs"],
        "pair_win_rate": win_rate,
        "pair_sprt": pair_sprt,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="runs/h2h_postrepair", help="directory holding the per-file arm JSONs")
    parser.add_argument("--arms", default="armA,armB,armC", help="comma-separated arm name prefixes")
    parser.add_argument(
        "--gate-config",
        choices=sorted(GATE_CONFIGS),
        default="flywheel",
        help="Named SPRT gate config (CAT-7) providing --elo0/--elo1 defaults; explicit flags override.",
    )
    parser.add_argument("--elo0", type=float, default=None, help="Override --gate-config's elo0.")
    parser.add_argument("--elo1", type=float, default=None, help="Override --gate-config's elo1.")
    parser.add_argument("--out", default=None, help="optional path to write the pooled JSON report")
    args = parser.parse_args()
    _gate_cfg, gate_params = resolve_gate_config(args.gate_config, elo0=args.elo0, elo1=args.elo1)
    args.elo0, args.elo1 = gate_params["elo0"], gate_params["elo1"]

    base = Path(args.dir)
    report: dict[str, Any] = {"gate_config": gate_params["gate_config"]}
    for arm in args.arms.split(","):
        arm = arm.strip()
        paths = sorted(Path(p) for p in glob.glob(str(base / f"{arm}_*.json")))
        if not paths:
            report[arm] = {"error": f"no files matched {base / (arm + '_*.json')}"}
            continue
        report[arm] = aggregate_arm(paths, elo0=args.elo0, elo1=args.elo1)

    print(json.dumps(report, indent=2, sort_keys=True))
    verdict_path = Path(args.out) if args.out else base / "verdict.json"
    atomic_io.write_json_atomic(verdict_path, report)
    print(f"[aggregate] durable verdict written: {verdict_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
