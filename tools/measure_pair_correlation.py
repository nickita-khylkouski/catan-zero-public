#!/usr/bin/env python3
"""Measure the EMPIRICAL within-pair correlation of our own H2H gate data.

BACKGROUND. Our promotion gates (tools/sprt_gate.py's pentanomial GSPRT) test
color-swapped PAIRS of games (same game_seed, candidate plays each color
once) rather than treating each game as an independent Bernoulli trial. The
computer-chess testing community (fishtest issue/discussion #348) found that
this pairing is not just a bias-cancellation trick: the empirical
correlation between the two games of a color-swapped pair is NEGATIVE
(fishtest measured rho ~= -0.15 on its own data). A negative within-pair
correlation REDUCES Var(pair-mean) relative to treating the two games as
independent (Var(X1+X2) = 2*var*(1+rho) for equal-variance X1, X2), so
pairing ADDS statistical power (~15% in fishtest's case) relative to a naive
binomial that ignores the pairing. Put differently: naive (unpaired)
binomial is CONSERVATIVE here, not anti-conservative, precisely because it
overestimates the true variance when rho < 0.

fishtest's -0.15 is fishtest's number, from fishtest's games (chess, its own
opening book, its own engines). This script measures OUR OWN empirical
within-pair correlation from our own gate JSON records, so we can confirm or
refine that borrowed number for Catan / our search config instead of just
citing someone else's measurement.

PAIRING CONVENTION (reused, not reinvented -- see tools/h2h_postrepair_
aggregate.py and tools/h2h_v3conf_aggregate.py, which this mirrors):
  * Games are pooled from one or more gate/H2H JSON files, each holding a
    top-level "games" list (tools/gumbel_search_cross_net_h2h.py /
    gumbel_search_vs_raw_h2h.py schema).
  * `game_seed` -- NOT `pair_id` -- is the cross-file-safe pairing key
    (`pair_id` resets to 0 per file/invocation).
  * Bit-identical duplicate games (the fleet seed-collision failure mode)
    are dropped first, keyed on (game_seed, orientation, search_color,
    winner, search_won, decisions) -- delegates to h2h_postrepair_aggregate's
    `_dedupe_games` so there is exactly one dedup implementation.
  * A pair is exactly two games sharing a game_seed, with a resolved outcome
    on the "search_won" (aka "candidate_won") field on each side. Pairs
    missing an orientation, or with a truncated (None) outcome on either
    side, are excluded (no correlation information).
  * Outcome is already recorded from a CONSISTENT side across the color
    swap: `search_won` / `candidate_won` is always "did the candidate/search
    role win", regardless of which physical color it played that game. Win
    -> 1.0, loss -> 0.0 (Catan games never draw, per sprt_gate.py's docs);
    a non-bool value is passed through as-is so a hypothetical future draw
    encoding (e.g. 0.5) would still work.
  * The two games within a pair are ordered by `orientation`
    ("candidate_red" -> 0, "candidate_blue" -> 1; falls back to an int
    orientation field, then to candidate_color/search_color RED->0/BLUE->1,
    then to file order) so that "game 1" / "game 2" mean the same thing
    (e.g. "candidate played RED first") across every pair in the sample --
    otherwise an arbitrary per-pair ordering would inject spurious
    correlation noise in either direction.

USAGE
    python tools/measure_pair_correlation.py 'runs/h2h_*/*.json'
    python tools/measure_pair_correlation.py file1.json file2.json --out out.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from h2h_postrepair_aggregate import _dedupe_games  # single source of truth for dedup


def _load_games(patterns: Sequence[str]) -> tuple[list[dict[str, Any]], list[str]]:
    """Expand each pattern with glob (falling back to a literal path if the
    pattern matches nothing, so plain non-wildcard paths still work) and pool
    every file's "games" list (or, if a file is itself a bare JSON list of
    game records, that list directly)."""
    matched_files: list[str] = []
    games: list[dict[str, Any]] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches and Path(pattern).is_file():
            matches = [pattern]
        for path_str in matches:
            path = Path(path_str)
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                games.extend(data)
            else:
                games.extend(data.get("games", ()))
            matched_files.append(str(path))
    return games, matched_files


def _orientation_rank(game: dict[str, Any]) -> int:
    """Deterministic within-pair ordering key so "position 1"/"position 2"
    mean the same thing across every pair (see module docstring)."""
    orientation = game.get("orientation")
    if orientation == "candidate_red":
        return 0
    if orientation == "candidate_blue":
        return 1
    if isinstance(orientation, bool):
        return int(orientation)
    if isinstance(orientation, int):
        return orientation
    color = game.get("candidate_color") or game.get("search_color")
    if color == "RED":
        return 0
    if color == "BLUE":
        return 1
    return 0


def _outcome_value(game: dict[str, Any]) -> float | None:
    """The candidate/search outcome from its consistent perspective (see
    module docstring). None means "no result" (truncated game)."""
    value = game.get("search_won", game.get("candidate_won"))
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return float(value)


def build_pairs(
    games: Sequence[dict[str, Any]],
) -> tuple[list[tuple[float, float]], dict[str, int]]:
    """Reduce a pooled game list to (game_1_outcome, game_2_outcome) pairs,
    deduplicating bit-identical replicas first and grouping by game_seed
    (the established cross-file-safe key -- see module docstring)."""
    deduped, n_dupes = _dedupe_games(list(games))
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for game in deduped:
        by_seed.setdefault(int(game["game_seed"]), []).append(game)

    pairs: list[tuple[float, float]] = []
    diagnostics = {
        "ww_pairs": 0,
        "split_pairs": 0,
        "ll_pairs": 0,
        "incomplete_pairs": 0,
        "duplicate_games_dropped": n_dupes,
    }
    for seed_games in by_seed.values():
        outcomes = [_outcome_value(g) for g in seed_games]
        if len(seed_games) != 2 or any(o is None for o in outcomes):
            diagnostics["incomplete_pairs"] += 1
            continue
        ordered = sorted(seed_games, key=_orientation_rank)
        x, y = _outcome_value(ordered[0]), _outcome_value(ordered[1])
        pairs.append((x, y))  # type: ignore[arg-type]
        if x == 1.0 and y == 1.0:
            diagnostics["ww_pairs"] += 1
        elif x == 0.0 and y == 0.0:
            diagnostics["ll_pairs"] += 1
        else:
            diagnostics["split_pairs"] += 1
    return pairs, diagnostics


def pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Plain-stdlib Pearson correlation coefficient (no numpy, matching the
    rest of tools/ -- see sprt_gate.py). Returns None when undefined (fewer
    than 2 pairs, or zero variance on either side)."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0.0 or var_y <= 0.0:
        return None
    return cov / math.sqrt(var_x * var_y)


def interpret_correlation(rho: float | None) -> str:
    if rho is None:
        return (
            "insufficient data (fewer than 2 complete pairs, or zero variance "
            "in game-1 or game-2 outcomes) to estimate a within-pair correlation"
        )
    if rho < -1e-9:
        return (
            f"negative within-pair correlation (rho={rho:.4f}): pairing ADDS "
            "statistical power relative to naive (unpaired) binomial -- the "
            "naive binomial is CONSERVATIVE here, matching fishtest issue #348's "
            "direction (fishtest measured rho ~= -0.15)"
        )
    if rho > 1e-9:
        return (
            f"positive within-pair correlation (rho={rho:.4f}): pairing REMOVES "
            "statistical power relative to naive (unpaired) binomial -- a naive "
            "binomial ignoring the pairing would be ANTI-CONSERVATIVE "
            "(overconfident) here, the OPPOSITE of fishtest issue #348's direction"
        )
    return (
        f"~zero within-pair correlation (rho={rho:.4f}): pairing neither adds "
        "nor removes statistical power relative to naive (unpaired) binomial"
    )


def measure_pair_correlation(patterns: Sequence[str]) -> dict[str, Any]:
    games, matched_files = _load_games(patterns)
    pairs, diagnostics = build_pairs(games)
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    rho = pearson_correlation(xs, ys)
    return {
        "patterns": list(patterns),
        "files_matched": matched_files,
        "games_pooled": len(games),
        "n_pairs": len(pairs),
        "correlation": rho,
        "interpretation": interpret_correlation(rho),
        **diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more paths or glob patterns to gate/H2H JSON record file(s).",
    )
    parser.add_argument("--out", help="Optional JSON output path.")
    args = parser.parse_args()

    report = measure_pair_correlation(args.paths)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
