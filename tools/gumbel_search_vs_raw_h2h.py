#!/usr/bin/env python3
"""CLI: search-adds-strength H2H gate (task #53 part 2).

Plays paired head-to-head games between `GumbelChanceMCTS` search
(force_full=True, i.e. n_full simulations every decision -- no playout-cap
mix) and the SAME checkpoint's raw policy (no search: argmax over the
evaluator's priors) directly, to test whether search actually beats the raw
network it wraps before committing to full self-play generation.

Both roles share one evaluator instance (one set of network weights) so the
comparison isolates search's contribution rather than confounding it with a
different checkpoint. Games are paired by seed AND color-swapped (each seed
is played twice, once with search=RED/raw=BLUE and once with the colors
swapped) to cancel positional/color bias -- the standard paired-seed H2H
protocol also used by tools/evaluate_scoreboard.py.

Per-game outcomes feed tools/sprt_gate.py's evaluate_sprt (elo0=0, elo1=30 --
the same >=55%-win-rate promotion bar documented there). Truncated games (no
winner within --max-decisions) are recorded but EXCLUDED from the SPRT input,
matching sprt_gate.py's own truncation-as-loss-bias fix -- a truncated game
carries no win/loss information for either side.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
import time
from pathlib import Path
from typing import Any

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from catan_zero.rl.config_cli import add_config_flags, resolve_config
from catan_zero.rl.gumbel_self_play import _apply_selected_action
from catan_zero.rl.pipeline_configs import EvalConfig
from catan_zero.search.gumbel_chance_mcts import GumbelChanceMCTS, GumbelChanceMCTSConfig
from catan_zero.search.neural_rust_mcts import (
    BatchedEntityGraphRustEvaluator,
    EntityGraphRustEvaluatorConfig,
)
from catan_zero.search.rust_mcts import _require_rust_module
from factory_common import write_json
from sprt_gate import GATE_CONFIGS, evaluate_pentanomial_sprt, evaluate_sprt, pair_scores_from_h2h_games, resolve_gate_config

COLORS: tuple[str, ...] = ("RED", "BLUE")


def _select_raw_action(evaluator: Any, game: Any, legal_actions: tuple[int, ...], *, acting_color: str) -> int:
    """Argmax over the evaluator's raw priors -- no search. Ties broken by
    lowest rust action id for determinism."""
    if len(legal_actions) == 1:
        return int(legal_actions[0])
    priors, _value = evaluator.evaluate(game, legal_actions, root_color=acting_color, colors=COLORS)
    return int(max(legal_actions, key=lambda action: (float(priors.get(int(action), 0.0)), -int(action))))


def play_one_h2h_game(
    mcts: GumbelChanceMCTS,
    evaluator: Any,
    *,
    role_by_color: dict[str, str],
    game_seed: int,
    max_decisions: int,
    correct_rust_chance_spectra: bool,
) -> dict[str, Any]:
    import random

    catanatron_rs = _require_rust_module()
    game = catanatron_rs.Game.simple(list(COLORS), seed=int(game_seed))
    chance_rng = random.Random(int(game_seed) ^ 0xA17E)

    decision_index = 0
    terminal = False
    while decision_index < int(max_decisions):
        if game.winning_color() is not None:
            terminal = True
            break
        legal_rust = tuple(
            int(action) for action in game.playable_action_indices(list(COLORS), None)
        )
        if not legal_rust:
            break

        acting_color = str(game.current_color())
        role = role_by_color[acting_color]
        if role == "search":
            result = mcts.search(game, force_full=True)
            selected = int(result.selected_action)
        else:
            selected = _select_raw_action(evaluator, game, legal_rust, acting_color=acting_color)

        game = _apply_selected_action(
            game,
            selected,
            colors=COLORS,
            rng=chance_rng,
            correct_rust_chance_spectra=correct_rust_chance_spectra,
        )
        decision_index += 1

    if not terminal:
        terminal = game.winning_color() is not None
    truncated = not terminal
    winner = str(game.winning_color()) if terminal else None
    final_vps: dict[str, int] = {}
    for color in COLORS:
        state = json.loads(game.player_state_json(color))
        final_vps[color] = int(state.get("victory_points", 0) or 0)

    search_color = next(color for color, role in role_by_color.items() if role == "search")
    raw_color = next(color for color, role in role_by_color.items() if role == "raw")
    search_won = (winner == search_color) if terminal else None

    return {
        "game_seed": int(game_seed),
        "search_color": search_color,
        "raw_color": raw_color,
        "winner": winner,
        "terminated": bool(terminal),
        "truncated": bool(truncated),
        "decisions": int(decision_index),
        "final_vps": final_vps,
        "search_won": search_won,
    }


def _worker_entry(worker_args: dict[str, Any]) -> dict[str, Any]:
    worker_index = int(worker_args.get("worker_index", -1))
    try:
        return _run_worker(worker_args)
    except Exception as error:  # noqa: BLE001 - isolate one worker from the whole batch.
        return {
            "worker_index": worker_index,
            "games": [],
            "error": f"worker-level failure before any game ran: {error!r}",
        }


def _run_worker(worker_args: dict[str, Any]) -> dict[str, Any]:
    # Cap torch's intra-op thread pool per worker process: torch defaults to
    # one thread per visible core FOR EACH process, so N worker processes on
    # a shared box each independently try to grab every core -- with
    # --workers 10 on a 52-core host that's up to 520 competing threads,
    # driving load average past 200+ and starving every other tenant's job
    # (and, via context-switch overhead, this job too). threads_per_worker
    # is sized by main() so workers * threads_per_worker leaves headroom.
    threads_per_worker = int(worker_args.get("threads_per_worker", 0))
    if threads_per_worker > 0:
        import torch

        torch.set_num_threads(threads_per_worker)
        torch.set_num_interop_threads(1)

    checkpoint = worker_args["checkpoint"]
    evaluator = BatchedEntityGraphRustEvaluator.from_checkpoint(
        checkpoint,
        device=worker_args["device"],
        config=EntityGraphRustEvaluatorConfig(
            value_scale=float(worker_args["value_scale"]),
            prior_temperature=float(worker_args["prior_temperature"]),
            value_squash=str(worker_args.get("value_squash", "tanh")),
            public_observation=bool(worker_args.get("public_observation", False)),
        ),
    )
    search_config = GumbelChanceMCTSConfig(
        colors=COLORS,
        seed=int(worker_args["worker_seed"]),
        n_full=int(worker_args["n_full"]),
        n_fast=int(worker_args["n_full"]),  # unused: force_full=True always selects n_full.
        p_full=1.0,
        max_depth=int(worker_args["max_depth"]),
        temperature=0.0,  # deterministic argmax at the root -- "does search help" not "how creative is search".
        correct_rust_chance_spectra=bool(worker_args["correct_rust_chance_spectra"]),
        lazy_interior_chance=bool(worker_args.get("lazy_interior_chance", False)),
        belief_chance_spectra=bool(worker_args.get("belief_chance_spectra", False)),
        information_set_search=bool(
            worker_args.get("information_set_search", False)
        ),
        determinization_particles=int(
            worker_args.get("determinization_particles", 1)
        ),
        determinization_min_simulations=int(
            worker_args.get("determinization_min_simulations", 32)
        ),
        c_scale=float(worker_args.get("c_scale", 0.1)),
        c_visit=float(worker_args.get("c_visit", 50.0)),
        max_root_candidates=int(worker_args.get("max_root_candidates", 16)),
        max_root_candidates_wide=int(worker_args.get("max_root_candidates_wide", 54)),
        n_full_wide=(
            int(worker_args["n_full_wide"])
            if worker_args.get("n_full_wide") is not None
            else None
        ),
        raw_policy_above_width=(
            int(worker_args["raw_policy_above_width"])
            if worker_args.get("raw_policy_above_width") is not None
            else None
        ),
        symmetry_averaged_eval=bool(worker_args.get("symmetry_averaged_eval", False)),
    )
    mcts = GumbelChanceMCTS(search_config, evaluator)

    games: list[dict[str, Any]] = []
    try:
        for pair in worker_args["pairs"]:
            game_seed = int(pair["game_seed"])
            for orientation, role_by_color in (
                ("search_red", {"RED": "search", "BLUE": "raw"}),
                ("search_blue", {"RED": "raw", "BLUE": "search"}),
            ):
                record = play_one_h2h_game(
                    mcts,
                    evaluator,
                    role_by_color=role_by_color,
                    game_seed=game_seed,
                    max_decisions=int(worker_args["max_decisions"]),
                    correct_rust_chance_spectra=bool(worker_args["correct_rust_chance_spectra"]),
                )
                record["orientation"] = orientation
                record["pair_id"] = int(pair["pair_id"])
                games.append(record)
    finally:
        evaluator.close()

    return {"worker_index": int(worker_args["worker_index"]), "games": games, "error": None}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search-adds-strength H2H gate: gumbel-search vs raw policy, same checkpoint."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pairs", type=int, default=50, help="paired seeds; total games = 2x this")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-full", type=int, default=64)
    parser.add_argument("--max-depth", type=int, default=80)
    parser.add_argument("--max-decisions", type=int, default=300)
    parser.add_argument("--prior-temperature", type=float, default=1.0)
    parser.add_argument("--value-scale", type=float, default=1.0)
    parser.add_argument(
        "--correct-rust-chance-spectra",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--lazy-interior-chance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run the SEARCH side with lazy interior chance evaluation (#52 lazy-vs-raw arm).",
    )
    parser.add_argument("--value-squash", choices=("tanh", "clip"), default="tanh",
                        help="Evaluator value squash (#60 diagnostic arm).")
    parser.add_argument("--c-visit", type=float, default=50.0,
                        help="Sigma c_visit floor; 1.0 = visit-scaled sigma (armV diagnostic).")
    parser.add_argument("--c-scale", type=float, default=0.1,
                        help="Sigma scale multiplier (matches GumbelChanceMCTSConfig default).")
    parser.add_argument("--max-root-candidates", type=int, default=16,
                        help="Root Gumbel-Top-k candidate cap on normal roots (SNR arm: 8).")
    parser.add_argument("--max-root-candidates-wide", type=int, default=54,
                        help="Root Gumbel-Top-k cap on wide (placement) roots; 16 = narrow diagnostic arm.")
    parser.add_argument(
        "--public-observation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Public-observation featurization (hidden-info leak fix, f72): mask each "
        "opponent's hand composition, unplayed dev-card identities, and actual VP from "
        "the model input for BOTH the search and raw sides (symmetric). Threads to "
        "EntityGraphRustEvaluatorConfig.public_observation. Default off; use with a "
        "checkpoint retrained via train_bc --mask-hidden-info for a valid public-only "
        "confirmation H2H.",
    )
    parser.add_argument(
        "--belief-chance-spectra",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Planner-only public-belief chance spectra (hidden-info leak fix, f72) for "
        "the SEARCH side. Threads to GumbelChanceMCTSConfig.belief_chance_spectra. "
        "Default off; a search-semantics change gated on its own strength-based A/B.",
    )
    parser.add_argument("--information-set-search", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--determinization-particles", type=int, default=1)
    parser.add_argument("--determinization-min-simulations", type=int, default=32)
    parser.add_argument("--n-full-wide", type=int, default=None,
                        help="Placement-budget-asymmetry arm: full-search simulations to spend at "
                        "roots wider than the config's wide_candidates_threshold (e.g. 512). "
                        "Default None = use --n-full everywhere (disabled).")
    parser.add_argument("--raw-policy-above-width", type=int, default=None,
                        help="Phase-gated-search arm: at roots wider than this many legal actions, "
                        "skip search and play argmax(prior). Default None = always search (disabled).")
    parser.add_argument(
        "--symmetry-averaged-eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "f74b: on the SEARCH side, denoise wide-root leaf value+prior by "
            "averaging the evaluator over all 12 D6 board orientations "
            "(gated to roots wider than wide_candidates_threshold). "
            "Threads to GumbelChanceMCTSConfig.symmetry_averaged_eval."
        ),
    )
    parser.add_argument("--base-seed", type=int, default=1)
    parser.add_argument(
        "--gate-config",
        choices=sorted(GATE_CONFIGS),
        default="flywheel",
        help="Named SPRT gate config (CAT-7) providing --elo0/--elo1 defaults; explicit flags override.",
    )
    parser.add_argument("--elo0", type=float, default=None, help="Override --gate-config's elo0.")
    parser.add_argument("--elo1", type=float, default=None, help="Override --gate-config's elo1.")
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        default=0,
        help="torch intra-op thread cap per worker process (0 = auto: "
        "floor(os.cpu_count() / workers), so --workers N never oversubscribes "
        "the host). Set explicitly to share a box with other tenants.",
    )
    parser.add_argument("--out", required=True)
    add_config_flags(parser, default_purpose="gumbel_search_vs_raw_h2h")
    args = parser.parse_args()
    if bool(args.public_observation) != bool(args.information_set_search):
        parser.error(
            "--public-observation and --information-set-search must be enabled together"
        )
    if bool(args.information_set_search) and bool(args.belief_chance_spectra):
        parser.error(
            "--information-set-search cannot be combined with --belief-chance-spectra"
        )
    if int(args.determinization_particles) < 1:
        parser.error("--determinization-particles must be >= 1")
    if int(args.determinization_min_simulations) < 1:
        parser.error("--determinization-min-simulations must be >= 1")
    _gate_cfg, _gate_params = resolve_gate_config(args.gate_config, elo0=args.elo0, elo1=args.elo1)
    args.elo0, args.elo1 = _gate_params["elo0"], _gate_params["elo1"]

    # CAT-66 typed config + config-hash (search-vs-own-raw regime).
    eval_config = resolve_config(
        args, lambda a: EvalConfig.from_namespace(a, mode="search_vs_raw"), parser=parser
    )
    eval_config_hash = eval_config.config_hash()

    pairs = [{"pair_id": i, "game_seed": int(args.base_seed) + i} for i in range(max(1, int(args.pairs)))]
    workers = max(1, int(args.workers))
    threads_per_worker = int(args.threads_per_worker)
    if threads_per_worker <= 0:
        import os as _os

        threads_per_worker = max(1, (_os.cpu_count() or workers) // workers)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        import os as _os

        _os.environ[name] = str(threads_per_worker)
    shards: list[list[dict[str, Any]]] = [[] for _ in range(workers)]
    for i, pair in enumerate(pairs):
        shards[i % workers].append(pair)

    worker_args = []
    for worker_index, pair_shard in enumerate(shards):
        if not pair_shard:
            continue
        worker_args.append(
            {
                "worker_index": worker_index,
                "pairs": pair_shard,
                "checkpoint": args.checkpoint,
                "device": args.device,
                "n_full": int(args.n_full),
                "max_depth": int(args.max_depth),
                "max_decisions": int(args.max_decisions),
                "prior_temperature": float(args.prior_temperature),
                "value_scale": float(args.value_scale),
                "correct_rust_chance_spectra": bool(args.correct_rust_chance_spectra),
                "lazy_interior_chance": bool(args.lazy_interior_chance),
                "public_observation": bool(args.public_observation),
                "belief_chance_spectra": bool(args.belief_chance_spectra),
                "information_set_search": bool(args.information_set_search),
                "determinization_particles": int(args.determinization_particles),
                "determinization_min_simulations": int(
                    args.determinization_min_simulations
                ),
                "value_squash": str(args.value_squash),
                "c_scale": float(args.c_scale),
                "c_visit": float(args.c_visit),
                "max_root_candidates": int(args.max_root_candidates),
                "max_root_candidates_wide": int(args.max_root_candidates_wide),
                "n_full_wide": (int(args.n_full_wide) if args.n_full_wide is not None else None),
                "raw_policy_above_width": (
                    int(args.raw_policy_above_width)
                    if args.raw_policy_above_width is not None
                    else None
                ),
                "symmetry_averaged_eval": bool(args.symmetry_averaged_eval),
                "threads_per_worker": threads_per_worker,
                "worker_seed": int(args.base_seed) + 0x9E3779B9 * (worker_index + 1),
            }
        )

    started = time.perf_counter()
    if len(worker_args) <= 1:
        results = [_worker_entry(worker_args[0])] if worker_args else []
    else:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=len(worker_args)) as pool:
            results = pool.map(_worker_entry, worker_args)
    elapsed = time.perf_counter() - started

    all_games: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for result in results:
        all_games.extend(result.get("games", ()))
        if result.get("error"):
            errors.append({"worker_index": result.get("worker_index"), "error": result["error"]})

    outcomes = [bool(game["search_won"]) for game in all_games if game["search_won"] is not None]
    truncated_count = sum(1 for game in all_games if game["truncated"])
    # Naive per-game SPRT: pools each color-swapped orientation as an
    # independent Bernoulli trial. Kept for reference/comparison only -- F5
    # found this throws away the pairing (an easy seed can make BOTH
    # orientations of a pair "search wins" for reasons that have nothing to
    # do with search vs. raw, inflating apparent power). The concordant-pair
    # SPRT below is the one that should gate the decision.
    summary = _build_summary(
        args,
        all_games=all_games,
        outcomes=outcomes,
        truncated_count=truncated_count,
        pairs=pairs,
        elapsed=elapsed,
        workers=workers,
        threads_per_worker=threads_per_worker,
        errors=errors,
    )
    summary["config_hash"] = eval_config_hash
    write_json(args.out, summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "games"}, indent=2, sort_keys=True))


def _build_summary(
    args: Any,
    *,
    all_games: list[dict[str, Any]],
    outcomes: list[bool],
    truncated_count: int,
    pairs: list[Any],
    elapsed: float,
    workers: int,
    threads_per_worker: int,
    errors: list[Any],
) -> dict[str, Any]:
    """Build the H2H run's output summary dict (task #79: self-certifying
    provenance). Every knob that changes the input distribution or search
    semantics (public_observation, belief_chance_spectra, n_full_wide,
    raw_policy_above_width, symmetry_averaged_eval) is recorded here, not
    just the search-budget knobs -- without this, a gate verdict can't be
    audited after the fact from the output JSON alone (exactly the gap that
    made the h2h_v3conf regime question unanswerable once the run had
    already finished and its process/logs were gone, see task #78)."""
    sprt = evaluate_sprt(outcomes=outcomes, elo0=float(args.elo0), elo1=float(args.elo1))
    pair_outcomes, pair_diagnostics = _concordant_pair_outcomes(all_games)
    pair_sprt = evaluate_sprt(outcomes=pair_outcomes, elo0=float(args.elo0), elo1=float(args.elo1))

    # Pentanomial (trinomial, no-draw) GSPRT: uses EVERY complete pair,
    # including splits (as a 0.5 pair-score), rather than discarding them like
    # the concordant rule. It holds the same nominal error rates but resolves
    # 2-8x faster in the split-heavy regime this gate operates in (validated
    # in tools/pentanomial_power_sim.py). This is the recommended gate verdict.
    pair_scores, _pent_diagnostics = pair_scores_from_h2h_games(all_games)
    pentanomial_sprt = evaluate_pentanomial_sprt(
        pair_scores, elo0=float(args.elo0), elo1=float(args.elo1)
    )

    # Split-rate metrics (task #2), computed over COMPLETE pairs only (both
    # orientations finished) -- incomplete/truncated pairs are not counted in
    # the denominator since they carry no win/loss information for either side.
    complete_pairs = (
        pair_diagnostics["ww_pairs"] + pair_diagnostics["ll_pairs"] + pair_diagnostics["split_pairs"]
    )
    decisive_pairs = pair_diagnostics["ww_pairs"] + pair_diagnostics["ll_pairs"]
    split_rate = (pair_diagnostics["split_pairs"] / complete_pairs) if complete_pairs else None
    decisive_pair_yield = (decisive_pairs / complete_pairs) if complete_pairs else None

    return {
        "checkpoint": args.checkpoint,
        "gate_config": getattr(args, "gate_config", None),
        "n_full": int(args.n_full),
        "lazy_interior_chance": bool(args.lazy_interior_chance),
        "value_squash": str(args.value_squash),
        "c_scale": float(args.c_scale),
        "c_visit": float(args.c_visit),
        "max_root_candidates": int(args.max_root_candidates),
        "max_root_candidates_wide": int(args.max_root_candidates_wide),
        "correct_rust_chance_spectra": bool(args.correct_rust_chance_spectra),
        # Task #79: input-distribution / search-semantics provenance -- these
        # were previously accepted on the CLI and threaded into the worker
        # config but never recorded in the output, making a run's actual
        # regime unrecoverable once the process/logs were gone.
        "public_observation": bool(args.public_observation),
        "belief_chance_spectra": bool(args.belief_chance_spectra),
        "information_set_search": bool(
            getattr(args, "information_set_search", False)
        ),
        "determinization_particles": int(
            getattr(args, "determinization_particles", 1)
        ),
        "determinization_min_simulations": int(
            getattr(args, "determinization_min_simulations", 32)
        ),
        "n_full_wide": (int(args.n_full_wide) if args.n_full_wide is not None else None),
        "raw_policy_above_width": (
            int(args.raw_policy_above_width) if args.raw_policy_above_width is not None else None
        ),
        "symmetry_averaged_eval": bool(args.symmetry_averaged_eval),
        "pairs_requested": len(pairs),
        "games_played": len(all_games),
        "games_with_winner": len(outcomes),
        "games_truncated": truncated_count,
        "search_wins": sum(1 for outcome in outcomes if outcome),
        "raw_wins": sum(1 for outcome in outcomes if not outcome),
        "search_win_rate": (sum(1 for outcome in outcomes if outcome) / len(outcomes)) if outcomes else None,
        "sprt": sprt,
        # F5 (concordant-pair rule, per eval-fixer): a color-swapped pair is
        # only informative if BOTH orientations agree on who won -- WW means
        # search legitimately won regardless of color, LL means it lost
        # regardless of color, and a split (won as one color, lost as the
        # other) carries no signal about search vs. raw (more likely
        # reflects which color/seed combination is easier) and must be
        # EXCLUDED, not coerced into a win or loss either way.
        "pair_sprt": pair_sprt,
        # Recommended verdict: trinomial GSPRT over all complete pairs (task #1).
        "pentanomial_sprt": pentanomial_sprt,
        "verdict": pentanomial_sprt["decision"],
        "pair_diagnostics": pair_diagnostics,
        # Same counts as pair_diagnostics, under the exact names requested
        # for cross-team reporting consistency.
        "pairs_decisive": pair_diagnostics["ww_pairs"] + pair_diagnostics["ll_pairs"],
        "pairs_split_excluded": pair_diagnostics["split_pairs"],
        "pairs_truncated_excluded": pair_diagnostics["incomplete_pairs"],
        # Split-rate metrics (task #2), over COMPLETE pairs only.
        "complete_pairs": complete_pairs,
        "split_rate": split_rate,
        "decisive_pair_yield": decisive_pair_yield,
        "elapsed_sec": elapsed,
        "workers": workers,
        "threads_per_worker": threads_per_worker,
        "errors": errors,
        "games": all_games,
    }


def _concordant_pair_outcomes(games: list[dict[str, Any]]) -> tuple[list[bool], dict[str, int]]:
    """F5: group games by pair_id (the two color-swapped orientations of the
    same seed) and reduce each pair to a single concordant outcome -- WW
    (search won both orientations) -> True, LL (search lost both) -> False,
    a split or either orientation truncated -> excluded entirely (not
    coerced into either outcome)."""
    by_pair: dict[int, list[dict[str, Any]]] = {}
    for game in games:
        by_pair.setdefault(int(game["pair_id"]), []).append(game)

    outcomes: list[bool] = []
    diagnostics = {"ww_pairs": 0, "ll_pairs": 0, "split_pairs": 0, "incomplete_pairs": 0}
    for pair_games in by_pair.values():
        if len(pair_games) != 2 or any(game["search_won"] is None for game in pair_games):
            diagnostics["incomplete_pairs"] += 1
            continue
        results = {bool(game["search_won"]) for game in pair_games}
        if results == {True}:
            outcomes.append(True)
            diagnostics["ww_pairs"] += 1
        elif results == {False}:
            outcomes.append(False)
            diagnostics["ll_pairs"] += 1
        else:
            diagnostics["split_pairs"] += 1
    return outcomes, diagnostics


if __name__ == "__main__":
    main()
