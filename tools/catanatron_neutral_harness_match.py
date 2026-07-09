#!/usr/bin/env python3
"""CLI: run our checkpoint (via `CatanZeroNetPlayer`) against a real catanatron
bot ENTIRELY inside catanatron's own `Game`/engine -- no shadow Rust engine,
no lockstep mirroring with a second engine (contrast
`tools/gumbel_search_vs_bot_h2h.py`, which mirrors moves onto catanatron only
so a real bot can decide, while every game's ground truth stays our own Rust
engine). This is the CAT-57 neutral-harness smoke test: "the number the
outside world will judge" has to come from a match catanatron's own engine
ran, not one we ran with catanatron along for the ride.

Scope (per the Linear issue): this is the CORRECTNESS smoke test -- confirm
games complete with no illegal-move/translation crashes and a non-degenerate
win rate -- not the production-scale 1000-game measurement (that's CAT-22).
Run it for tens of games, not thousands.

Output is the same pentanomial-gate JSON shape `gumbel_search_vs_bot_h2h.py`
produces (`pair_scores_from_h2h_games` / `evaluate_pentanomial_sprt` /
`evaluate_sprt`, `pair_id` + `search_won` per game), plus `"stratum":
"neutral-harness"` so WHR/arena ingestion can distinguish these from the
Rust-lockstep H2H numbers instead of silently pooling two different harnesses
together.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from catan_zero.rl._catanatron import import_catanatron_module
from catanatron_player_adapter import CatanZeroNetPlayer, _color_name, standard_colors
from factory_common import write_json
from sprt_gate import evaluate_pentanomial_sprt, evaluate_sprt, pair_scores_from_h2h_games

BOT_KINDS = (
    "catanatron_value",
    "catanatron_ab1",
    "catanatron_ab2",
    "catanatron_ab3",
    "catanatron_ab4",
    "catanatron_ab5",
    "random",
)


def _make_bot(name: str, color: Any) -> Any:
    """A real, unmodified catanatron bot -- default depth/weights, exactly as
    it plays natively (see `gumbel_search_vs_bot_h2h.py._make_bot`'s same
    caveat: NOT `catan_zero.rl.self_play.CatanatronAlphaBetaPolicy`, which
    forces `full_width_root=True` for distillation targets)."""
    if name == "random":
        player_module = import_catanatron_module("catanatron.models.player")
        return player_module.RandomPlayer(color)
    if name == "catanatron_value":
        value_module = import_catanatron_module("catanatron.players.value")
        return value_module.ValueFunctionPlayer(color)
    if name.startswith("catanatron_ab"):
        depth = int(name[len("catanatron_ab"):])
        minimax_module = import_catanatron_module("catanatron.players.minimax")
        return minimax_module.AlphaBetaPlayer(color, depth=depth, prunning=True)
    raise ValueError(f"unknown --opponent {name!r}; choose from {BOT_KINDS}")


def play_one_game(
    *,
    checkpoint: str,
    opponent: str,
    orientation: str,
    pair_id: int,
    game_seed: int,
    device: str,
    vps_to_win: int,
    sample: bool,
    max_player_trade_offers_per_turn: int,
) -> dict[str, Any]:
    game_module = import_catanatron_module("catanatron.game")
    map_module = import_catanatron_module("catanatron.models.map")

    colors = standard_colors(2)
    candidate_color, baseline_color = (
        (colors[0], colors[1]) if orientation == "candidate_first" else (colors[1], colors[0])
    )

    candidate = CatanZeroNetPlayer(
        candidate_color,
        checkpoint=checkpoint,
        device=device,
        seed=game_seed,
        vps_to_win=vps_to_win,
        sample=sample,
        max_player_trade_offers_per_turn=max_player_trade_offers_per_turn,
    )
    baseline = _make_bot(opponent, baseline_color)
    players = [candidate if color == candidate_color else baseline for color in colors]

    catan_map = map_module.build_map("BASE")
    game = game_module.Game(players=players, seed=game_seed, catan_map=catan_map, vps_to_win=vps_to_win)

    error: str | None = None
    winner = None
    try:
        winner = game.play()
    except Exception as exc:  # noqa: BLE001 - isolate one bad game, don't kill the batch.
        error = repr(exc)

    terminated = error is None and winner is not None
    truncated = error is None and not terminated
    candidate_won = (_color_name(winner) == _color_name(candidate_color)) if terminated else None

    return {
        "pair_id": int(pair_id),
        "game_seed": int(game_seed),
        "orientation": orientation,
        "candidate_color": _color_name(candidate_color),
        "baseline_color": _color_name(baseline_color),
        "winner": _color_name(winner) if winner is not None else None,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "error": error,
        "decisions": (len(game.state.action_records) if error is None else None),
        "candidate_won": candidate_won,
        "search_won": candidate_won,
        "illegal_policy_picks": int(candidate.stats["illegal_policy_picks"]),
    }


def _wilson_ci(wins: int, games: int, z: float = 1.96) -> list[float] | None:
    if games <= 0:
        return None
    p = wins / games
    denom = 1 + z * z / games
    center = p + z * z / (2 * games)
    half = z * ((p * (1 - p) / games + z * z / (4 * games * games)) ** 0.5)
    return [max(0.0, (center - half) / denom), min(1.0, (center + half) / denom)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Our checkpoint (raw policy, CatanZeroNetPlayer) vs a real "
        "catanatron bot, played entirely inside catanatron's own engine."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--opponent", required=True, choices=BOT_KINDS)
    parser.add_argument("--pairs", type=int, default=25, help="paired seeds; total games = 2x this")
    parser.add_argument("--base-seed", type=int, default=1)
    parser.add_argument("--vps-to-win", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sample", action=argparse.BooleanOptionalAction, default=False,
                        help="Sample from the policy instead of greedy argmax.")
    parser.add_argument(
        "--max-player-trade-offers-per-turn", type=int, default=0,
        help="Cap on OFFER_TRADE (player-to-player) offers the net may make "
             "per turn. Default 0 = NO player trading, matching the benchmark "
             "spec and catanatron own bots (whose action set never includes "
             "OFFER_TRADE). Values >0 re-enable it but risk a greedy-argmax "
             "OFFER/REJECT livelock where num_turns never advances.")
    parser.add_argument("--elo0", type=float, default=0.0)
    parser.add_argument("--elo1", type=float, default=30.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    games: list[dict[str, Any]] = []
    started = time.perf_counter()
    for pair_id in range(max(1, int(args.pairs))):
        game_seed = int(args.base_seed) + pair_id
        for orientation in ("candidate_first", "candidate_second"):
            record = play_one_game(
                checkpoint=args.checkpoint,
                opponent=args.opponent,
                orientation=orientation,
                pair_id=pair_id,
                game_seed=game_seed,
                device=args.device,
                vps_to_win=int(args.vps_to_win),
                sample=bool(args.sample),
                max_player_trade_offers_per_turn=int(
                    args.max_player_trade_offers_per_turn
                ),
            )
            games.append(record)
            print(
                {
                    "progress": "game_done",
                    "pair_id": pair_id,
                    "orientation": orientation,
                    "candidate_won": record["candidate_won"],
                    "error": record["error"],
                    "illegal_policy_picks": record["illegal_policy_picks"],
                },
                flush=True,
            )
    elapsed = time.perf_counter() - started

    outcomes = [bool(g["candidate_won"]) for g in games if g["candidate_won"] is not None]
    errors = [g for g in games if g["error"] is not None]
    truncated_count = sum(1 for g in games if g["truncated"])
    total_illegal_picks = sum(g["illegal_policy_picks"] for g in games)

    sprt = evaluate_sprt(outcomes=outcomes, elo0=float(args.elo0), elo1=float(args.elo1))
    pair_scores, pair_diagnostics = pair_scores_from_h2h_games(games)
    pentanomial_sprt = evaluate_pentanomial_sprt(pair_scores, elo0=float(args.elo0), elo1=float(args.elo1))
    win_rate = (sum(1 for o in outcomes if o) / len(outcomes)) if outcomes else None
    wilson_ci = _wilson_ci(sum(1 for o in outcomes if o), len(outcomes)) if outcomes else None

    summary = {
        "stratum": "neutral-harness",
        "harness": "catanatron_native_engine",
        "candidate_checkpoint": args.checkpoint,
        "baseline_bot": args.opponent,
        "mode": "raw_policy",
        "max_player_trade_offers_per_turn": int(args.max_player_trade_offers_per_turn),
        "vps_to_win": int(args.vps_to_win),
        "pairs_requested": int(args.pairs),
        "games_played": len(games),
        "games_with_winner": len(outcomes),
        "games_truncated": truncated_count,
        "games_errored": len(errors),
        "candidate_wins": sum(1 for o in outcomes if o),
        "baseline_wins": sum(1 for o in outcomes if not o),
        "candidate_win_rate": win_rate,
        "candidate_win_rate_wilson_95ci": wilson_ci,
        "total_illegal_policy_picks": total_illegal_picks,
        "sprt": sprt,
        "pentanomial_sprt": pentanomial_sprt,
        "pair_diagnostics": pair_diagnostics,
        "elapsed_sec": elapsed,
        "errors": errors,
        "games": games,
    }
    write_json(args.out, summary)
    print(
        {k: v for k, v in summary.items() if k not in ("games", "errors")},
    )


if __name__ == "__main__":
    main()
