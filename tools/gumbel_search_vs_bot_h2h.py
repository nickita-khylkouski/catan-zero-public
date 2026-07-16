#!/usr/bin/env python3
"""CLI: searched-checkpoint vs hardcoded Catanatron bot H2H (absolute-strength benchmark).

Plays a candidate checkpoint using FULL GumbelChanceMCTS search (identical search config
to tools/gumbel_search_cross_net_h2h.py) against a real, standard Catanatron bot
(catanatron.players.minimax.AlphaBetaPlayer depth 3/4/5, or
catanatron.players.value.ValueFunctionPlayer) instead of another checkpoint. This measures
the net's ABSOLUTE strength with search, rather than the raw-policy floor that
tools/evaluate_scoreboard.py / tools/grade_agent.py report (those never wrap the
checkpoint's action selection in any tree search).

Bridging problem: GumbelChanceMCTS operates on the fast Rust engine (`catanatron_rs.Game`);
the standard Catanatron bots operate on the vendored Python engine's `Game` /
`Player.decide(game, playable_actions)` interface and have no Rust binding at all. There is
no existing single-engine path that supports both sides of this matchup (see the
investigation notes below for what was ruled out).

This script drives BOTH engines in lockstep for every game, reusing the transcript-
equivalence bridge in `catan_zero.adapters.engine_equivalence` (built for a different
purpose -- proving the two engines produce identical outcomes for the SAME sequence of
moves -- but it already contains every primitive needed here: build a seating-aligned
Rust+Python game pair on the fixed TOURNAMENT map, translate a raw action between engines,
and force identical chance outcomes -- dice, robber steals, dev-card draws -- into both
engines via `apply_chance_step` so they never diverge over the course of a game):

  - the candidate's turn: `GumbelChanceMCTS.search(rust_game, force_full=True)` picks the
    move on the Rust engine -- this is where all the search work happens, with the exact
    same n_full/c_scale/c_visit/lazy_interior_chance/public_observation config as
    tools/gumbel_search_cross_net_h2h.py;
  - the bot's turn: `AlphaBetaPlayer`/`ValueFunctionPlayer.decide(python_game, ...)` picks
    the move on the mirrored Python engine, exactly as it would in a native Catanatron game
    (no catan-zero customization -- default depth/prunning/value-fn weights, NOT the
    `full_width_root=True` distillation variant used elsewhere in this repo for producing
    soft policy targets);
  - whichever side moved, the chosen action is translated and applied to BOTH engines
    (forcing the same chance outcome for ROLL / MOVE_ROBBER-with-victim /
    BUY_DEVELOPMENT_CARD), so the Python mirror always reflects the Rust game's true state
    before the bot is asked to decide again.

Investigated and ruled out before writing this:
  - tools/evaluate_scoreboard.py's `--candidate-kind` only selects a checkpoint *loading*
    format (entity_graph / xdim_lite / ppo); it never wraps a checkpoint's action selection
    in a search. Its own `"search"`/`"value_rollout"` policy kind
    (`ValueRolloutSearchPolicy` in policy_pool.py) is an unrelated non-neural rollout bot,
    not a Gumbel-searched net. There is no zero-new-code path through it.
  - `catan_zero.rl.self_play.CatanatronAlphaBetaPolicy` (what evaluate_scoreboard.py's
    "catanatron_ab3" opponent actually is) requires a `ColonistMultiAgentEnv` wrapping the
    vendored *Python* catanatron Game (`env.game.copy()`, `env.game.state.action_records`,
    `env.action_catalog`) -- a completely different game object from `catanatron_rs.Game`,
    so it cannot be called directly from the Rust-native GumbelChanceMCTS game loop.

Caveat (read before comparing numbers across scripts): `engine_equivalence`'s cross-engine
board-parity guarantee only holds for the fixed `TOURNAMENT` map (the vendored Python
engine's board-shuffle RNG doesn't match Rust's), so every game here is played on that one
fixed board (only seating order / dice / robber steals / dev-card draws vary with
`--base-seed`). `tools/gumbel_search_cross_net_h2h.py`'s net-vs-net games use
`catanatron_rs.Game.simple()` with a randomly shuffled map per seed instead. So an absolute
win rate from THIS script should not be directly compared to a net-vs-net win rate from
THAT script -- but two runs of THIS script (e.g. gen-1 vs ab3, v3a vs ab3) are apples-to-
apples with each other.

A 1000-game background sweep of the equivalence bridge itself
(`runs/engine_equivalence/report_fixed_pair_1000.json`, random-vs-random play) found
993/1000 games fully equivalent and 7 divergences (6 classified `longest_road`
rules-adjudication edge cases + 1 unclassified). This script re-checks the legal-action set
and full state view for equivalence after every ply and marks the game
`"engine_divergence": true` (excluded from win-rate stats, reported separately under
`games_engine_divergence`) rather than letting a divergence silently corrupt a result.
"""

from __future__ import annotations

# Direct-script execution adds the repository source roots below before loading
# project modules; those imports are intentionally after the path bootstrap.
# ruff: noqa: E402

import argparse
import json
import multiprocessing
import random
import sys
import time
from pathlib import Path
from typing import Any

_TOOLS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TOOLS_DIR.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from catan_zero.adapters.engine_equivalence import (
    EquivalenceConfig,
    apply_chance_step,
    build_paired_games,
    canonical_python_action_key,
    canonical_rust_action_key,
    diff_state_views,
    is_chance_action,
    legal_action_diff,
    python_state_view,
    raw_action_to_python_action,
    rust_legal_actions,
    rust_state_view,
    vendor_symbols,
)
from catan_zero.rl._catanatron import import_catanatron_module
from catan_zero.rl.config_cli import add_config_flags, resolve_config
from catan_zero.rl.entity_token_features_rust import require_rust_feature_path
from catan_zero.rl.pipeline_configs import EvalConfig
from catan_zero.search.gumbel_chance_mcts import (
    GumbelChanceMCTS,
    GumbelChanceMCTSConfig,
)
from catan_zero.search.native_gumbel_mcts import (
    create_gumbel_search,
    native_hot_loop_available,
)
from catan_zero.search.neural_rust_mcts import (
    BatchedEntityGraphRustEvaluator,
    EntityGraphRustEvaluatorConfig,
)
from factory_common import write_json
from sprt_gate import (
    GATE_CONFIGS,
    evaluate_pentanomial_sprt,
    evaluate_sprt,
    pair_scores_from_h2h_games,
    resolve_gate_config,
)

MAP_KIND = (
    "TOURNAMENT"  # only map engine_equivalence guarantees Rust<->Python parity for.
)

BOT_KINDS = ("catanatron_ab3", "catanatron_ab4", "catanatron_ab5", "catanatron_value")


def _create_search(
    config: GumbelChanceMCTSConfig,
    evaluator: Any,
    *,
    native_mcts_hot_loop: bool,
) -> GumbelChanceMCTS:
    if not native_mcts_hot_loop:
        return GumbelChanceMCTS(config, evaluator)
    return create_gumbel_search(config, evaluator, native_hot_loop=True)


def _make_bot(name: str, color: Any) -> Any:
    """Standard/hardcoded Catanatron bot -- default depth/prunning/value weights, exactly
    as it plays in native Catanatron (NOT catan-zero's CatanatronAlphaBetaPolicy, which
    forces full_width_root=True to produce soft distillation targets rather than play its
    normal competitive move)."""
    minimax_module = import_catanatron_module("catanatron.players.minimax")
    value_module = import_catanatron_module("catanatron.players.value")
    if name == "catanatron_ab3":
        return minimax_module.AlphaBetaPlayer(color, depth=3, prunning=True)
    if name == "catanatron_ab4":
        return minimax_module.AlphaBetaPlayer(color, depth=4, prunning=True)
    if name == "catanatron_ab5":
        return minimax_module.AlphaBetaPlayer(color, depth=5, prunning=True)
    if name == "catanatron_value":
        return value_module.ValueFunctionPlayer(color)
    raise ValueError(f"unknown --baseline-bot {name!r}; choose from {BOT_KINDS}")


def play_one_h2h_game(
    *,
    candidate_evaluator: Any,
    search_config_kwargs: dict[str, Any],
    baseline_bot_name: str,
    role_by_color: dict[str, str],
    game_seed: int,
    max_decisions: int,
    native_mcts_hot_loop: bool = False,
) -> dict[str, Any]:
    symbols = vendor_symbols()
    chance_rng = random.Random(int(game_seed) ^ 0xA17E)

    equiv_config = EquivalenceConfig(
        colors=("RED", "BLUE"),
        map_kind=MAP_KIND,
        vps_to_win=10,
        discard_limit=7,
        friendly_robber=False,
        max_steps=max(2000, int(max_decisions) * 6),
    )
    rust_game, python_game, seated_colors = build_paired_games(
        int(game_seed), equiv_config
    )

    candidate_mcts = _create_search(
        GumbelChanceMCTSConfig(
            colors=seated_colors,
            map_kind=MAP_KIND,
            seed=int(game_seed),
            **search_config_kwargs,
        ),
        candidate_evaluator,
        native_mcts_hot_loop=bool(native_mcts_hot_loop),
    )
    bot_by_color = {
        color: _make_bot(baseline_bot_name, getattr(symbols.Color, color))
        for color in seated_colors
        if role_by_color[color] == "baseline"
    }

    decision_index = 0
    terminal = False
    divergence_detail: str | None = None
    while decision_index < int(max_decisions):
        winner_color = rust_game.winning_color()
        if winner_color is not None:
            terminal = True
            break

        ids, raw_actions = rust_legal_actions(rust_game, seated_colors, MAP_KIND)
        if not ids:
            break

        only_rust, only_python = legal_action_diff(
            raw_actions, python_game.playable_actions
        )
        if only_rust or only_python:
            divergence_detail = (
                f"legal action mismatch at decision {decision_index}: "
                f"only_rust={sorted(only_rust)[:5]!r} only_python={sorted(only_python)[:5]!r}"
            )
            break

        acting_color = str(rust_game.current_color())
        role = role_by_color[acting_color]

        if role == "candidate":
            result = candidate_mcts.search(rust_game, force_full=True)
            selected_id = int(result.selected_action)
            try:
                pos = ids.index(selected_id)
            except ValueError:
                divergence_detail = f"candidate selected illegal action {selected_id}"
                break
            chosen_raw = raw_actions[pos]
        else:
            bot = bot_by_color[acting_color]
            py_action = bot.decide(
                python_game.copy(), tuple(python_game.playable_actions)
            )
            key = canonical_python_action_key(py_action)
            matches = [
                i
                for i, raw in enumerate(raw_actions)
                if canonical_rust_action_key(raw) == key
            ]
            if not matches:
                divergence_detail = (
                    f"bot action {py_action!r} has no Rust-side equivalent"
                )
                break
            pos = matches[0]
            selected_id, chosen_raw = ids[pos], raw_actions[pos]

        try:
            if is_chance_action(chosen_raw):
                rust_game, _outcome = apply_chance_step(
                    rust_game,
                    python_game,
                    chosen_raw,
                    symbols=symbols,
                    harness_rng=chance_rng,
                )
            else:
                rust_game.execute_action_index(
                    selected_id, list(seated_colors), MAP_KIND
                )
                python_game.execute(raw_action_to_python_action(chosen_raw, symbols))
        except Exception as error:  # noqa: BLE001 - record as divergence, not a crash.
            divergence_detail = f"exception applying {chosen_raw!r} at decision {decision_index}: {error!r}"
            break

        mismatches = diff_state_views(
            rust_state_view(rust_game), python_state_view(python_game, symbols)
        )
        if mismatches:
            divergence_detail = (
                f"state mismatch after decision {decision_index} action={chosen_raw!r}: "
                + "; ".join(mismatches[:5])
            )
            break

        decision_index += 1

    if not terminal and divergence_detail is None:
        terminal = rust_game.winning_color() is not None
    engine_divergence = divergence_detail is not None
    truncated = (not terminal) and (not engine_divergence)
    winner = str(rust_game.winning_color()) if terminal else None
    final_vps: dict[str, int] = {}
    for color in seated_colors:
        state = json.loads(rust_game.player_state_json(color))
        final_vps[color] = int(state.get("victory_points", 0) or 0)

    candidate_color = next(
        color for color, role in role_by_color.items() if role == "candidate"
    )
    baseline_color = next(
        color for color, role in role_by_color.items() if role == "baseline"
    )
    candidate_won = (
        (winner == candidate_color) if (terminal and not engine_divergence) else None
    )

    return {
        "game_seed": int(game_seed),
        "candidate_color": candidate_color,
        "baseline_color": baseline_color,
        "winner": winner,
        "terminated": bool(terminal),
        "truncated": bool(truncated),
        "engine_divergence": bool(engine_divergence),
        "divergence_detail": divergence_detail,
        "decisions": int(decision_index),
        "final_vps": final_vps,
        "candidate_won": candidate_won,
        "search_won": candidate_won,
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


def _build_evaluator(checkpoint: str, worker_args: dict[str, Any]) -> Any:
    return BatchedEntityGraphRustEvaluator.from_checkpoint(
        checkpoint,
        device=worker_args["device"],
        config=EntityGraphRustEvaluatorConfig(
            value_scale=float(worker_args["value_scale"]),
            prior_temperature=float(worker_args["prior_temperature"]),
            value_squash=str(worker_args.get("value_squash", "tanh")),
            public_observation=bool(worker_args.get("public_observation", False)),
            rust_featurize=bool(worker_args.get("evaluator_rust_featurize", True)),
        ),
    )


def _search_config_kwargs(worker_args: dict[str, Any]) -> dict[str, Any]:
    return dict(
        n_full=int(worker_args["n_full"]),
        n_fast=int(
            worker_args["n_full"]
        ),  # unused: force_full=True always selects n_full.
        p_full=1.0,
        max_depth=int(worker_args["max_depth"]),
        temperature=0.0,  # deterministic argmax at the root.
        correct_rust_chance_spectra=bool(worker_args["correct_rust_chance_spectra"]),
        lazy_interior_chance=bool(worker_args.get("lazy_interior_chance", False)),
        belief_chance_spectra=bool(worker_args.get("belief_chance_spectra", False)),
        information_set_search=bool(worker_args.get("information_set_search", False)),
        boundary_value_particles=int(
            worker_args.get("boundary_value_particles", 1)
        ),
        determinization_particles=int(worker_args.get("determinization_particles", 1)),
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
        symmetry_averaged_eval_threshold=(
            int(worker_args["symmetry_averaged_eval_threshold"])
            if worker_args.get("symmetry_averaged_eval_threshold") is not None
            else None
        ),
    )


def _write_worker_progress(
    progress_dir: str, worker_index: int, games_done: int, wins: int, divergences: int
) -> None:
    if not progress_dir:
        return
    import os as _os

    p = _os.path.join(progress_dir, f"worker_{worker_index:03d}.json")
    tmp = p + ".tmp"
    try:
        with open(tmp, "w") as fh:
            json.dump(
                {
                    "worker_index": worker_index,
                    "games_done": games_done,
                    "candidate_wins": wins,
                    "engine_divergences": divergences,
                },
                fh,
            )
        _os.replace(tmp, p)
    except OSError:
        pass


def _run_worker(worker_args: dict[str, Any]) -> dict[str, Any]:
    threads_per_worker = int(worker_args.get("threads_per_worker", 0))
    if threads_per_worker > 0:
        import torch

        torch.set_num_threads(threads_per_worker)
        torch.set_num_interop_threads(1)

    candidate_evaluator = _build_evaluator(
        worker_args["candidate_checkpoint"], worker_args
    )
    search_config_kwargs = _search_config_kwargs(worker_args)
    baseline_bot_name = str(worker_args["baseline_bot"])

    games: list[dict[str, Any]] = []
    pair_errors: list[dict[str, Any]] = []
    try:
        for pair in worker_args["pairs"]:
            game_seed = int(pair["game_seed"])
            # Isolate failures per pair: one bad game must not discard the whole
            # worker's completed games. A half-finished pair is dropped entirely
            # (paired stats require both orientations anyway).
            pair_games: list[dict[str, Any]] = []
            try:
                for orientation, role_by_color in (
                    ("candidate_red", {"RED": "candidate", "BLUE": "baseline"}),
                    ("candidate_blue", {"RED": "baseline", "BLUE": "candidate"}),
                ):
                    record = play_one_h2h_game(
                        candidate_evaluator=candidate_evaluator,
                        search_config_kwargs=search_config_kwargs,
                        baseline_bot_name=baseline_bot_name,
                        role_by_color=role_by_color,
                        game_seed=game_seed,
                        max_decisions=int(worker_args["max_decisions"]),
                        native_mcts_hot_loop=bool(
                            worker_args.get("native_mcts_hot_loop", False)
                        ),
                    )
                    record["orientation"] = orientation
                    record["pair_id"] = int(pair["pair_id"])
                    pair_games.append(record)
            except Exception as error:  # noqa: BLE001 - keep the worker's other pairs.
                pair_errors.append(
                    {
                        "pair_id": int(pair["pair_id"]),
                        "game_seed": game_seed,
                        "error": repr(error),
                    }
                )
                continue
            games.extend(pair_games)
            _wins = sum(1 for g in games if g.get("candidate_won"))
            _divs = sum(1 for g in games if g.get("engine_divergence"))
            _write_worker_progress(
                worker_args.get("progress_dir", ""),
                int(worker_args["worker_index"]),
                len(games),
                _wins,
                _divs,
            )
    finally:
        candidate_evaluator.close()

    return {
        "worker_index": int(worker_args["worker_index"]),
        "games": games,
        "error": None,
        "pair_errors": pair_errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Searched-checkpoint (GumbelChanceMCTS, full search) vs a hardcoded "
        "Catanatron bot (AlphaBeta depth 3/4/5 or ValueFunctionPlayer) H2H."
    )
    parser.add_argument("--candidate", required=True, help="Candidate checkpoint path.")
    parser.add_argument("--baseline-bot", required=True, choices=BOT_KINDS)
    parser.add_argument(
        "--pairs", type=int, default=50, help="paired seeds; total games = 2x this"
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--devices",
        default=None,
        help="comma-list of devices to spread workers across, e.g. cuda:0,cuda:1.",
    )
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
        "--lazy-interior-chance", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--value-squash", choices=("tanh", "clip"), default="tanh")
    parser.add_argument("--c-visit", type=float, default=50.0)
    parser.add_argument("--c-scale", type=float, default=0.1)
    parser.add_argument("--max-root-candidates", type=int, default=16)
    parser.add_argument("--max-root-candidates-wide", type=int, default=54)
    parser.add_argument(
        "--public-observation", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--belief-chance-spectra", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--information-set-search", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--boundary-value-particles", type=int, default=1)
    parser.add_argument("--determinization-particles", type=int, default=1)
    parser.add_argument("--determinization-min-simulations", type=int, default=32)
    parser.add_argument("--n-full-wide", type=int, default=None)
    parser.add_argument("--raw-policy-above-width", type=int, default=None)
    parser.add_argument(
        "--symmetry-averaged-eval", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--symmetry-averaged-eval-threshold",
        type=int,
        default=None,
        help="Inclusive minimum legal-action count for D6 evaluator averaging. "
        "Default None preserves the legacy wide-root threshold gate.",
    )
    parser.add_argument(
        "--native-mcts-hot-loop",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Explicitly use the feature-gated Rust MCTS tree hot loop. Default "
        "False preserves Python; enabling fails closed if the matching wheel is absent.",
    )
    parser.add_argument(
        "--evaluator-rust-featurize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Build entity and legal-action context tensors with the bit-exact "
            "native featurizer (default) and fail closed; historical Python-"
            "feature replay must pass the explicit negative flag."
        ),
    )
    parser.add_argument("--base-seed", type=int, default=1)
    parser.add_argument(
        "--gate-config",
        choices=sorted(GATE_CONFIGS),
        default="flywheel",
        help="Named SPRT gate config (CAT-7) providing --elo0/--elo1 defaults; explicit flags override.",
    )
    parser.add_argument(
        "--elo0", type=float, default=None, help="Override --gate-config's elo0."
    )
    parser.add_argument(
        "--elo1", type=float, default=None, help="Override --gate-config's elo1."
    )
    parser.add_argument("--threads-per-worker", type=int, default=0)
    parser.add_argument("--out", required=True)
    add_config_flags(parser, default_purpose="gumbel_search_vs_bot_h2h")
    args = parser.parse_args()
    if bool(args.native_mcts_hot_loop) and not native_hot_loop_available():
        parser.error(
            "--native-mcts-hot-loop requires a matching catanatron_rs wheel "
            "exporting gumbel_search; refusing silent Python fallback"
        )
    if bool(args.evaluator_rust_featurize):
        try:
            require_rust_feature_path()
        except RuntimeError as error:
            parser.error(str(error))
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
    if int(args.boundary_value_particles) != 1:
        parser.error(
            "--boundary-value-particles must remain 1 without "
            "--coherent-public-belief-search"
        )
    if int(args.determinization_min_simulations) < 1:
        parser.error("--determinization-min-simulations must be >= 1")
    _gate_cfg, _gate_params = resolve_gate_config(
        args.gate_config, elo0=args.elo0, elo1=args.elo1
    )
    args.elo0, args.elo1 = _gate_params["elo0"], _gate_params["elo1"]

    # CAT-66 typed config + config-hash (checkpoint-vs-Catanatron-bot regime).
    # map_kind is pinned to the module constant (only map with Rust<->Python
    # engine parity) rather than a CLI flag, so it is passed as an override.
    eval_config = resolve_config(
        args,
        lambda a: EvalConfig.from_namespace(a, mode="vs_bot", map_kind=MAP_KIND),
        parser=parser,
    )
    eval_config_hash = eval_config.config_hash()

    pairs = [
        {"pair_id": i, "game_seed": int(args.base_seed) + i}
        for i in range(max(1, int(args.pairs)))
    ]
    workers = max(1, int(args.workers))
    threads_per_worker = int(args.threads_per_worker)
    if threads_per_worker <= 0:
        import os as _os

        threads_per_worker = max(1, (_os.cpu_count() or workers) // workers)
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        import os as _os

        _os.environ[name] = str(threads_per_worker)
    shards: list[list[dict[str, Any]]] = [[] for _ in range(workers)]
    for i, pair in enumerate(pairs):
        shards[i % workers].append(pair)

    devices = (
        [d.strip() for d in args.devices.split(",")] if args.devices else [args.device]
    )
    from pathlib import Path as _Path

    progress_dir = _Path(args.out).parent / "progress"
    progress_dir.mkdir(parents=True, exist_ok=True)

    worker_args = []
    for worker_index, pair_shard in enumerate(shards):
        if not pair_shard:
            continue
        worker_args.append(
            {
                "worker_index": worker_index,
                "pairs": pair_shard,
                "candidate_checkpoint": args.candidate,
                "baseline_bot": args.baseline_bot,
                "device": devices[worker_index % len(devices)],
                "progress_dir": str(progress_dir),
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
                "native_mcts_hot_loop": bool(args.native_mcts_hot_loop),
                "evaluator_rust_featurize": bool(args.evaluator_rust_featurize),
                "determinization_particles": int(args.determinization_particles),
                "boundary_value_particles": int(args.boundary_value_particles),
                "determinization_min_simulations": int(
                    args.determinization_min_simulations
                ),
                "value_squash": str(args.value_squash),
                "c_scale": float(args.c_scale),
                "c_visit": float(args.c_visit),
                "max_root_candidates": int(args.max_root_candidates),
                "max_root_candidates_wide": int(args.max_root_candidates_wide),
                "n_full_wide": (
                    int(args.n_full_wide) if args.n_full_wide is not None else None
                ),
                "raw_policy_above_width": (
                    int(args.raw_policy_above_width)
                    if args.raw_policy_above_width is not None
                    else None
                ),
                "symmetry_averaged_eval": bool(args.symmetry_averaged_eval),
                "symmetry_averaged_eval_threshold": (
                    int(args.symmetry_averaged_eval_threshold)
                    if args.symmetry_averaged_eval_threshold is not None
                    else None
                ),
                "threads_per_worker": threads_per_worker,
            }
        )

    started = time.perf_counter()
    if len(worker_args) <= 1:
        results = [_worker_entry(worker_args[0])] if worker_args else []
    else:
        ctx = multiprocessing.get_context("spawn")
        results = []
        with ctx.Pool(processes=len(worker_args)) as pool:
            for done, result in enumerate(
                pool.imap_unordered(_worker_entry, worker_args), start=1
            ):
                results.append(result)
                _g = sum(len(r.get("games", ())) for r in results)
                _w = sum(
                    1
                    for r in results
                    for gm in r.get("games", ())
                    if gm.get("candidate_won")
                )
                _d = sum(
                    1
                    for r in results
                    for gm in r.get("games", ())
                    if gm.get("engine_divergence")
                )
                print(
                    json.dumps(
                        {
                            "progress": "worker_done",
                            "workers_done": done,
                            "workers_total": len(worker_args),
                            "games_so_far": _g,
                            "candidate_wins_so_far": _w,
                            "engine_divergences_so_far": _d,
                            "running_winrate": round(_w / _g, 4) if _g else None,
                        }
                    ),
                    flush=True,
                )
    elapsed = time.perf_counter() - started

    all_games: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for result in results:
        all_games.extend(result.get("games", ()))
        if result.get("error"):
            errors.append(
                {"worker_index": result.get("worker_index"), "error": result["error"]}
            )
        for pair_error in result.get("pair_errors") or ():
            errors.append({"worker_index": result.get("worker_index"), **pair_error})

    outcomes = [
        bool(game["candidate_won"])
        for game in all_games
        if game["candidate_won"] is not None
    ]
    truncated_count = sum(1 for game in all_games if game["truncated"])
    divergence_count = sum(1 for game in all_games if game.get("engine_divergence"))
    summary = _build_summary(
        args,
        all_games=all_games,
        outcomes=outcomes,
        truncated_count=truncated_count,
        divergence_count=divergence_count,
        pairs=pairs,
        elapsed=elapsed,
        workers=workers,
        threads_per_worker=threads_per_worker,
        errors=errors,
    )
    summary["config_hash"] = eval_config_hash
    write_json(args.out, summary)
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "games"}, indent=2, sort_keys=True
        )
    )


def _build_summary(
    args: Any,
    *,
    all_games: list[dict[str, Any]],
    outcomes: list[bool],
    truncated_count: int,
    divergence_count: int,
    pairs: list[Any],
    elapsed: float,
    workers: int,
    threads_per_worker: int,
    errors: list[Any],
) -> dict[str, Any]:
    sprt = evaluate_sprt(
        outcomes=outcomes, elo0=float(args.elo0), elo1=float(args.elo1)
    )
    pair_outcomes, pair_diagnostics = _concordant_pair_outcomes(all_games)
    pair_sprt = evaluate_sprt(
        outcomes=pair_outcomes, elo0=float(args.elo0), elo1=float(args.elo1)
    )
    pair_scores, _pent_diagnostics = pair_scores_from_h2h_games(all_games)
    pentanomial_sprt = evaluate_pentanomial_sprt(
        pair_scores, elo0=float(args.elo0), elo1=float(args.elo1)
    )

    complete_pairs = (
        pair_diagnostics["ww_pairs"]
        + pair_diagnostics["ll_pairs"]
        + pair_diagnostics["split_pairs"]
    )
    decisive_pairs = pair_diagnostics["ww_pairs"] + pair_diagnostics["ll_pairs"]
    split_rate = (
        (pair_diagnostics["split_pairs"] / complete_pairs) if complete_pairs else None
    )
    decisive_pair_yield = (decisive_pairs / complete_pairs) if complete_pairs else None
    win_rate = (
        (sum(1 for outcome in outcomes if outcome) / len(outcomes))
        if outcomes
        else None
    )
    wilson_ci = (
        _wilson_ci(sum(1 for o in outcomes if o), len(outcomes)) if outcomes else None
    )

    return {
        "candidate_checkpoint": args.candidate,
        "baseline_bot": args.baseline_bot,
        "gate_config": getattr(args, "gate_config", None),
        "map_kind": MAP_KIND,
        "n_full": int(args.n_full),
        "lazy_interior_chance": bool(args.lazy_interior_chance),
        "value_squash": str(args.value_squash),
        "c_scale": float(args.c_scale),
        "c_visit": float(args.c_visit),
        "max_root_candidates": int(args.max_root_candidates),
        "max_root_candidates_wide": int(args.max_root_candidates_wide),
        "correct_rust_chance_spectra": bool(args.correct_rust_chance_spectra),
        "public_observation": bool(args.public_observation),
        "belief_chance_spectra": bool(args.belief_chance_spectra),
        "information_set_search": bool(getattr(args, "information_set_search", False)),
        "native_mcts_hot_loop": bool(getattr(args, "native_mcts_hot_loop", False)),
        "mcts_implementation": (
            "rust_native_hot_loop_v1"
            if bool(getattr(args, "native_mcts_hot_loop", False))
            else "python_reference"
        ),
        "determinization_particles": int(getattr(args, "determinization_particles", 1)),
        "boundary_value_particles": int(
            getattr(args, "boundary_value_particles", 1)
        ),
        "determinization_min_simulations": int(
            getattr(args, "determinization_min_simulations", 32)
        ),
        "symmetry_averaged_eval": bool(args.symmetry_averaged_eval),
        "symmetry_averaged_eval_threshold": (
            int(args.symmetry_averaged_eval_threshold)
            if getattr(args, "symmetry_averaged_eval_threshold", None) is not None
            else None
        ),
        "pairs_requested": len(pairs),
        "games_played": len(all_games),
        "games_with_winner": len(outcomes),
        "games_truncated": truncated_count,
        "games_engine_divergence": divergence_count,
        "candidate_wins": sum(1 for outcome in outcomes if outcome),
        "baseline_wins": sum(1 for outcome in outcomes if not outcome),
        "candidate_win_rate": win_rate,
        "candidate_win_rate_wilson_95ci": wilson_ci,
        "sprt": sprt,
        "pair_sprt": pair_sprt,
        "pentanomial_sprt": pentanomial_sprt,
        "verdict": pentanomial_sprt["decision"],
        "pair_diagnostics": pair_diagnostics,
        "pairs_decisive": decisive_pairs,
        "pairs_split_excluded": pair_diagnostics["split_pairs"],
        "pairs_truncated_excluded": pair_diagnostics["incomplete_pairs"],
        "complete_pairs": complete_pairs,
        "split_rate": split_rate,
        "decisive_pair_yield": decisive_pair_yield,
        "elapsed_sec": elapsed,
        "workers": workers,
        "threads_per_worker": threads_per_worker,
        "errors": errors,
        "games": all_games,
    }


def _wilson_ci(wins: int, games: int, z: float = 1.96) -> list[float] | None:
    if games <= 0:
        return None
    p = wins / games
    denom = 1 + z * z / games
    center = p + z * z / (2 * games)
    half = z * ((p * (1 - p) / games + z * z / (4 * games * games)) ** 0.5)
    return [max(0.0, (center - half) / denom), min(1.0, (center + half) / denom)]


def _concordant_pair_outcomes(
    games: list[dict[str, Any]],
) -> tuple[list[bool], dict[str, int]]:
    by_pair: dict[int, list[dict[str, Any]]] = {}
    for game in games:
        by_pair.setdefault(int(game["pair_id"]), []).append(game)

    outcomes: list[bool] = []
    diagnostics = {
        "ww_pairs": 0,
        "ll_pairs": 0,
        "split_pairs": 0,
        "incomplete_pairs": 0,
    }
    for pair_games in by_pair.values():
        if len(pair_games) != 2 or any(
            game["candidate_won"] is None for game in pair_games
        ):
            diagnostics["incomplete_pairs"] += 1
            continue
        results = {bool(game["candidate_won"]) for game in pair_games}
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
