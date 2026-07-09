#!/usr/bin/env python3
"""Aggregate the v3 confirmation H2H bracket (search vs raw), per arm.

The lead's 16-GPU fleet fans each arm across many independent
gumbel_search_vs_raw_h2h.py invocations (one JSON per GPU, distinct
base-seed blocks), on both A100 hosts. This tool pools them PER ARM and
pooled by `game_seed` (the only cross-file-safe pair key -- `pair_id` resets
to 0 per file), and reports:

  * PENTANOMIAL (primary, post-Gate-A rule): trinomial-no-draw GSPRT over
    every complete color-swapped pair, splits kept as 0.5. `mean_pair_score`
    == the estimated per-game search win rate. Resolves 2-8x faster than the
    concordant rule in the split-heavy regime (validated in
    pentanomial_power_sim.py).
  * CONCORDANT (secondary): WW/LL GSPRT, splits EXCLUDED -- the conservative
    cross-check.
  * per-game search win rate (the "beat raw by more than v2's 67-71%?"
    number), split rate, decisive-pair yield.

Arms are auto-derived from each file's own `checkpoint` + `c_scale` (robust
to fleet filenames): the checkpoint dir name is matched to a short label
(v3a/v3b/...) and combined with c_scale, e.g. `v3b_cs0.03`. Reuses the exact
SPRT reducers the single-file tool uses (sprt_gate.evaluate_sprt /
evaluate_pentanomial_sprt), so the pooled verdict is apples-to-apples with
the per-file ones.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
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

from sprt_gate import GATE_CONFIGS, evaluate_pentanomial_sprt, evaluate_sprt, resolve_gate_config


def _arm_label(checkpoint: str, c_scale: float) -> str:
    """Short, stable arm label from the checkpoint path + c_scale."""
    stem = Path(str(checkpoint)).parent.name or Path(str(checkpoint)).name
    m = re.search(r"(v\d+[a-z]?)", stem)
    base = m.group(1) if m else stem
    return f"{base}_cs{c_scale:g}"


def _dedupe_games(games: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Drop BIT-IDENTICAL duplicate games. The chance RNG is seeded by
    game_seed ^ 0xA17E and search is deterministic in the config, so running
    the SAME (game_seed, orientation) with the same net/config twice yields an
    identical game. The fleet's seed-collision bug did exactly this: v3b-base's
    64-seed block ran on BOTH hosts => 64 games each played twice, bit-for-bit.
    Counting both as independent pairs would ~halve p-values spuriously. A game
    is unique per (game_seed, orientation, search_color, winner, search_won,
    decisions); only EXACT matches are dropped, so genuinely different games
    (e.g. a different search seed producing a different result) are kept."""
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


def _reduce_by_game_seed(games: list[dict[str, Any]]) -> dict[str, Any]:
    """Reduce color-swapped orientation pairs to the concordant outcome list
    (WW=True/LL=False, split & incomplete excluded) and the pentanomial counts
    (n_LL, n_split, n_WW).

    Bit-identical duplicate games are dropped FIRST (see `_dedupe_games` --
    handles the fleet seed-collision replication), then the two color-swapped
    orientations are grouped by game_seed. Post-dedup each seed has exactly 2
    orientations => 1 pair; a seed with != 2 finished orientations is
    incomplete and excluded."""
    games, n_dupes = _dedupe_games(games)
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for game in games:
        by_seed.setdefault(int(game["game_seed"]), []).append(game)

    concordant: list[bool] = []
    n_ll = n_split = n_ww = 0
    diag = {"ww_pairs": 0, "ll_pairs": 0, "split_pairs": 0, "incomplete_pairs": 0, "duplicate_games_dropped": n_dupes}
    for seed_games in by_seed.values():
        if len(seed_games) != 2 or any(g.get("search_won") is None for g in seed_games):
            diag["incomplete_pairs"] += 1
            continue
        wins = sum(1 for g in seed_games if bool(g["search_won"]))
        if wins == 2:
            concordant.append(True)
            n_ww += 1
            diag["ww_pairs"] += 1
        elif wins == 0:
            concordant.append(False)
            n_ll += 1
            diag["ll_pairs"] += 1
        else:
            n_split += 1
            diag["split_pairs"] += 1
    return {
        "concordant_outcomes": concordant,
        "pentanomial_counts": (n_ll, n_split, n_ww),
        "diagnostics": diag,
        "duplicate_games_dropped": n_dupes,
        "distinct_seeds": len(by_seed),
    }


# The full search config the H2H runner records per file. gen-1 generation
# must replicate the WINNING arm's config exactly, so surface all of it.
_CONFIG_KEYS = (
    "checkpoint",
    "n_full",
    "n_full_wide",
    "c_scale",
    "c_visit",
    "max_root_candidates",
    "max_root_candidates_wide",
    "max_decisions",
    "max_depth",
    "prior_temperature",
    "value_scale",
    "value_squash",
    "lazy_interior_chance",
    "correct_rust_chance_spectra",
    "public_observation",
    "belief_chance_spectra",
    "raw_policy_above_width",
    # D1/D2 (f70, unmerged): present only if the arm ran f70 code. Absent on
    # master => the arm ran with them OFF (nothing to replicate).
    "rescale_noise_floor_c",
    "sigma_eval",
    "variance_aware_q",
    "variance_aware_k",
)


def load_games(paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_games: list[dict[str, Any]] = []
    config: dict[str, Any] = {}
    for path in paths:
        data = json.loads(path.read_text())
        for game in data.get("games", ()):
            game["_source"] = str(path)  # for the (source, pair_id) pair key
            all_games.append(game)
        for key in _CONFIG_KEYS:
            if key in data:
                config.setdefault(key, data[key])
    return all_games, config


def aggregate_arm(paths: list[Path], *, elo0: float, elo1: float) -> dict[str, Any]:
    games, config = load_games(paths)
    reduced = _reduce_by_game_seed(games)
    concordant = reduced["concordant_outcomes"]
    n_ll, n_split, n_ww = reduced["pentanomial_counts"]

    pentanomial = evaluate_pentanomial_sprt(counts=(n_ll, n_split, n_ww), elo0=elo0, elo1=elo1)
    concordant_sprt = evaluate_sprt(outcomes=concordant, elo0=elo0, elo1=elo1)

    per_game = [bool(g["search_won"]) for g in games if g.get("search_won") is not None]
    per_game_win_rate = (sum(per_game) / len(per_game)) if per_game else None
    search_wins_game = sum(1 for o in per_game if o)
    raw_wins_game = sum(1 for o in per_game if not o)
    games_truncated = sum(1 for g in games if g.get("truncated"))
    complete_pairs = n_ll + n_split + n_ww
    decisive = n_ww + n_ll

    # gen-1 replication flags: which search knobs the arm used that
    # generate_gumbel_selfplay_data.py must reproduce. D1/D2 are absent on
    # master (f70 unmerged) so default OFF; public_observation + c_scale +
    # belief_chance_spectra ARE exposed by the generator, but n_full_wide and
    # raw_policy_above_width are NOT -- flag them if used so ops-fixer knows a
    # generator CLI-wiring task is (or isn't) needed.
    gen1_replication = {
        "d1_noise_floor_on": bool(config.get("rescale_noise_floor_c", 0) or 0) > 0,
        "d2_variance_aware_on": bool(config.get("variance_aware_q", False)),
        "public_observation": config.get("public_observation"),
        "belief_chance_spectra": config.get("belief_chance_spectra"),
        "n_full_wide": config.get("n_full_wide"),
        "raw_policy_above_width": config.get("raw_policy_above_width"),
        # True => generator lacks a CLI for a non-default knob this arm used.
        "generator_cli_gap": (
            config.get("n_full_wide") is not None
            or config.get("raw_policy_above_width") is not None
        ),
    }
    return {
        "files": [str(p) for p in paths],
        "n_files": len(paths),
        "config": config,
        "gen1_replication": gen1_replication,
        "games_played": len(games),
        "duplicate_games_dropped": reduced.get("duplicate_games_dropped", 0),
        "games_with_winner": len(per_game),
        # Per-GAME breakdown (for the winning arm: the lead pulls placement-loss
        # concentration from these, like the Gate-A 74.6% blowout signature).
        "search_wins_game": search_wins_game,
        "raw_wins_game": raw_wins_game,
        "games_truncated": games_truncated,
        "truncation_rate_per_game": (games_truncated / len(games)) if games else None,
        "distinct_game_seeds": reduced["distinct_seeds"],
        "complete_pairs": complete_pairs,
        "pairs_decisive": decisive,
        "pairs_split": n_split,
        "pairs_incomplete_excluded": reduced["diagnostics"]["incomplete_pairs"],
        "search_win_rate_per_game": per_game_win_rate,
        "split_rate": (n_split / complete_pairs) if complete_pairs else None,
        "decisive_pair_yield": (decisive / complete_pairs) if complete_pairs else None,
        # PRIMARY
        "pentanomial_sprt": pentanomial,
        # SECONDARY
        "concordant_sprt": concordant_sprt,
        "concordant_ww": n_ww,
        "concordant_ll": n_ll,
    }


def group_files_by_arm(paths: list[Path]) -> dict[str, list[Path]]:
    arms: dict[str, list[Path]] = {}
    for path in paths:
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if "checkpoint" not in data or "c_scale" not in data:
            continue
        label = _arm_label(data["checkpoint"], float(data["c_scale"]))
        arms.setdefault(label, []).append(path)
    return arms


def _fmt(x: Any, spec: str = ".4f") -> str:
    return format(x, spec) if isinstance(x, (int, float)) else str(x)


def print_table(report: dict[str, dict[str, Any]]) -> None:
    hdr = (
        f"{'arm':14} {'pairs':>6} {'wr/game':>8} {'penta_dec':>9} {'penta_mean':>10} "
        f"{'conc_dec':>8} {'split%':>7} {'n_full':>6} {'c_scale':>7} {'D1/D2':>6} {'gen_gap':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for arm in sorted(report):
        a = report[arm]
        if "error" in a:
            print(f"{arm:14} {a['error']}")
            continue
        p = a["pentanomial_sprt"]
        c = a["concordant_sprt"]
        cfg = a["config"]
        g = a["gen1_replication"]
        d1d2 = ("D1" if g["d1_noise_floor_on"] else "-") + "/" + ("D2" if g["d2_variance_aware_on"] else "-")
        print(
            f"{arm:14} {a['complete_pairs']:>6} {_fmt(a['search_win_rate_per_game'],'.3f'):>8} "
            f"{str(p['decision']):>9} {_fmt(p['mean_pair_score'],'.3f'):>10} "
            f"{str(c['decision']):>8} {_fmt((a['split_rate'] or 0)*100,'.1f'):>7} "
            f"{str(cfg.get('n_full','?')):>6} {_fmt(cfg.get('c_scale'),'.3f'):>7} {d1d2:>6} "
            f"{str(g['generator_cli_gap']):>7}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="runs/h2h_v3conf", help="dir of per-file arm JSONs (globbed *.json)")
    parser.add_argument(
        "--gate-config",
        choices=sorted(GATE_CONFIGS),
        default="flywheel",
        help="Named SPRT gate config (CAT-7) providing --elo0/--elo1 defaults; explicit flags override.",
    )
    parser.add_argument("--elo0", type=float, default=None, help="Override --gate-config's elo0.")
    parser.add_argument("--elo1", type=float, default=None, help="Override --gate-config's elo1.")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    _gate_cfg, gate_params = resolve_gate_config(args.gate_config, elo0=args.elo0, elo1=args.elo1)
    args.elo0, args.elo1 = gate_params["elo0"], gate_params["elo1"]

    paths = sorted(Path(p) for p in glob.glob(str(Path(args.dir) / "*.json")))
    if not paths:
        raise SystemExit(f"no *.json under {args.dir}")

    arms = group_files_by_arm(paths)
    report = {
        arm: aggregate_arm(arm_paths, elo0=args.elo0, elo1=args.elo1)
        for arm, arm_paths in arms.items()
    }
    print_table(report)
    payload = {"gate_config": gate_params["gate_config"], "elo0": args.elo0, "elo1": args.elo1, "arms": report}
    verdict_path = Path(args.out) if args.out else Path(args.dir) / "verdict.json"
    atomic_io.write_json_atomic(verdict_path, payload)
    print(f"[aggregate] durable verdict written: {verdict_path}", file=sys.stderr)
    if not args.out:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
