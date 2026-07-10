#!/usr/bin/env python3
"""Native-catanatron neutral harness for raw-policy smoke and search panels.

The default remains the original raw-policy correctness smoke: a
``CatanZeroNetPlayer`` is called directly by catanatron's own ``Game.play``.

``--mode search`` is the powered external panel.  Catanatron's Python game is
still the sole referee/ground truth and its native bots make their decisions
unchanged.  The candidate uses ``GumbelChanceMCTS`` on a seating-aligned Rust
shadow.  Every authoritative native ``ActionRecord`` (including its resolved
dice, stolen resource, or development card) is replayed into that shadow and
state/legal parity is checked before search.  This is intentionally restricted
to the fixed TOURNAMENT map: BASE-map shuffle parity between the two engines is
not established.  Any boundary divergence invalidates and excludes that game.

Long panels are process-parallel and resumable.  Each orientation is written
atomically as soon as it finishes; rerunning the same command skips compatible
artifacts and refuses a directory whose outcome-affecting fingerprint differs.
The final JSON retains the existing per-game ``candidate_won``/``search_won``
and pentanomial SPRT shape consumed by the gate and WHR tools.
"""

# ruff: noqa: E402 -- this executable adds its sibling tools directory before imports.

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path
from typing import Any

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from catan_zero.adapters.engine_equivalence import EquivalenceConfig, build_paired_games
from catan_zero.rl._catanatron import import_catanatron_module
from catan_zero.rl.entity_token_policy import EntityGraphPolicy
from catan_zero.search.gumbel_chance_mcts import (
    GumbelChanceMCTS,
    GumbelChanceMCTSConfig,
)
from catan_zero.search.neural_rust_mcts import (
    BatchedEntityGraphRustEvaluator,
    EntityGraphRustEvaluatorConfig,
    _assert_value_readout_available,
)
from catanatron_player_adapter import (
    CatanZeroNetPlayer,
    CatanZeroSearchPlayer,
    SearchEngineBoundaryError,
    _color_name,
    standard_colors,
)
from factory_common import write_json
from sprt_gate import (
    GATE_CONFIGS,
    evaluate_pentanomial_sprt,
    evaluate_sprt,
    pair_scores_from_h2h_games,
    resolve_gate_config,
)

BOT_KINDS = (
    "catanatron_value",
    "catanatron_ab1",
    "catanatron_ab2",
    "catanatron_ab3",
    "catanatron_ab4",
    "catanatron_ab5",
    "random",
)
ORIENTATIONS = ("candidate_first", "candidate_second")
SEARCH_MAP_KIND = "TOURNAMENT"
ARTIFACT_SCHEMA_VERSION = 1


class _DecisionLimitReached(RuntimeError):
    pass


def _make_bot(name: str, color: Any) -> Any:
    """Real, unmodified catanatron bot with its native defaults."""
    if name == "random":
        player_module = import_catanatron_module("catanatron.models.player")
        return player_module.RandomPlayer(color)
    if name == "catanatron_value":
        value_module = import_catanatron_module("catanatron.players.value")
        return value_module.ValueFunctionPlayer(color)
    if name.startswith("catanatron_ab"):
        depth = int(name[len("catanatron_ab") :])
        minimax_module = import_catanatron_module("catanatron.players.minimax")
        return minimax_module.AlphaBetaPlayer(color, depth=depth, prunning=True)
    raise ValueError(f"unknown --opponent {name!r}; choose from {BOT_KINDS}")


def play_one_raw_game(
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
    """Original no-shadow raw-policy smoke path (kept as the default)."""
    game_module = import_catanatron_module("catanatron.game")
    map_module = import_catanatron_module("catanatron.models.map")

    colors = standard_colors(2)
    candidate_color, baseline_color = (
        (colors[0], colors[1])
        if orientation == "candidate_first"
        else (colors[1], colors[0])
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
    game = game_module.Game(
        players=players,
        seed=game_seed,
        catan_map=map_module.build_map("BASE"),
        vps_to_win=vps_to_win,
    )

    error: str | None = None
    winner = None
    started = time.perf_counter()
    try:
        winner = game.play()
    except Exception as exc:  # noqa: BLE001 - isolate a game, preserve the batch.
        error = repr(exc)
    elapsed = time.perf_counter() - started
    terminated = error is None and winner is not None
    truncated = error is None and not terminated
    candidate_won = (
        _color_name(winner) == _color_name(candidate_color) if terminated else None
    )
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
        "engine_divergence": False,
        "divergence_detail": None,
        "decisions": len(game.state.action_records) if error is None else None,
        "candidate_won": candidate_won,
        "search_won": candidate_won,
        "illegal_policy_picks": int(candidate.stats["illegal_policy_picks"]),
        "search_decisions": 0,
        "simulations_used": 0,
        "elapsed_sec": elapsed,
    }


def _search_config(
    search_kwargs: dict[str, Any], seated_colors: tuple[str, ...], seed: int
) -> GumbelChanceMCTSConfig:
    return GumbelChanceMCTSConfig(
        colors=seated_colors,
        map_kind=SEARCH_MAP_KIND,
        seed=int(seed),
        n_full=int(search_kwargs["n_full"]),
        n_fast=int(search_kwargs["n_full"]),
        p_full=1.0,
        max_depth=int(search_kwargs["max_depth"]),
        temperature=0.0,
        c_visit=float(search_kwargs["c_visit"]),
        c_scale=float(search_kwargs["c_scale"]),
        lazy_interior_chance=bool(search_kwargs["lazy_interior_chance"]),
        correct_rust_chance_spectra=bool(search_kwargs["correct_rust_chance_spectra"]),
        max_root_candidates=int(search_kwargs["max_root_candidates"]),
        max_root_candidates_wide=int(search_kwargs["max_root_candidates_wide"]),
        wide_candidates_threshold=int(search_kwargs.get("wide_candidates_threshold", 24)),
        n_full_wide=(
            int(search_kwargs["n_full_wide"])
            if search_kwargs.get("n_full_wide") is not None
            else None
        ),
        n_full_wide_threshold=(
            int(search_kwargs["n_full_wide_threshold"])
            if search_kwargs.get("n_full_wide_threshold") is not None
            else None
        ),
        wide_roots_always_full=bool(
            search_kwargs.get("wide_roots_always_full", False)
        ),
        symmetry_averaged_eval=bool(
            search_kwargs.get("symmetry_averaged_eval", False)
        ),
        symmetry_averaged_eval_threshold=(
            int(search_kwargs["symmetry_averaged_eval_threshold"])
            if search_kwargs.get("symmetry_averaged_eval_threshold") is not None
            else None
        ),
    )


def play_one_search_game(
    *,
    evaluator: Any,
    search_kwargs: dict[str, Any],
    opponent: str,
    orientation: str,
    pair_id: int,
    game_seed: int,
    vps_to_win: int,
    max_decisions: int,
) -> dict[str, Any]:
    """Search candidate vs native bot, refereed by native Game.play()."""
    equiv_config = EquivalenceConfig(
        colors=("BLUE", "RED"),
        map_kind=SEARCH_MAP_KIND,
        vps_to_win=int(vps_to_win),
        discard_limit=7,
        friendly_robber=False,
        max_steps=max(2000, int(max_decisions) * 6),
    )
    rust_game, game, seated_colors = build_paired_games(int(game_seed), equiv_config)
    candidate_name, baseline_name = (
        (seated_colors[0], seated_colors[1])
        if orientation == "candidate_first"
        else (seated_colors[1], seated_colors[0])
    )
    symbols = import_catanatron_module("catanatron.models.player")
    candidate_color = getattr(symbols.Color, candidate_name)
    baseline_color = getattr(symbols.Color, baseline_name)
    search = GumbelChanceMCTS(
        _search_config(search_kwargs, seated_colors, int(game_seed)), evaluator
    )
    candidate = CatanZeroSearchPlayer(
        candidate_color,
        rust_game=rust_game,
        search=search,
        seated_colors=seated_colors,
        map_kind=SEARCH_MAP_KIND,
    )
    baseline = _make_bot(opponent, baseline_color)
    player_by_name = {candidate_name: candidate, baseline_name: baseline}
    # build_paired_games already created the authoritative native game with
    # Rust-aligned seating.  Replacing only Player objects leaves all state,
    # map, decks, prompts, and playable actions untouched.
    game.state.players = [player_by_name[color.name] for color in game.state.colors]

    decisions = 0

    def bounded_decide(player: Any, native_game: Any, playable_actions: Any) -> Any:
        nonlocal decisions
        if decisions >= int(max_decisions):
            raise _DecisionLimitReached
        decisions += 1
        return player.decide(native_game, playable_actions)

    error: str | None = None
    divergence_detail: str | None = None
    hit_decision_limit = False
    started = time.perf_counter()
    try:
        game.play(decide_fn=bounded_decide)
    except _DecisionLimitReached:
        hit_decision_limit = True
    except SearchEngineBoundaryError as exc:
        divergence_detail = str(exc)
        error = repr(exc)
    except Exception as exc:  # noqa: BLE001 - one game cannot kill a panel.
        error = repr(exc)

    # If the baseline won, the candidate is not called again; replay/audit its
    # final action(s) here so terminal parity is still checked.
    if divergence_detail is None:
        try:
            candidate.audit_current_game(game)
        except SearchEngineBoundaryError as exc:
            divergence_detail = str(exc)
            error = repr(exc)
    elapsed = time.perf_counter() - started

    engine_divergence = divergence_detail is not None
    winner = game.winning_color()
    terminated = winner is not None and error is None and not engine_divergence
    truncated = bool(hit_decision_limit or (winner is None and error is None))
    candidate_won = _color_name(winner) == candidate_name if terminated else None
    return {
        "pair_id": int(pair_id),
        "game_seed": int(game_seed),
        "orientation": orientation,
        "candidate_color": candidate_name,
        "baseline_color": baseline_name,
        "winner": _color_name(winner) if winner is not None else None,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "error": error,
        "engine_divergence": bool(engine_divergence),
        "divergence_detail": divergence_detail,
        "decisions": int(decisions),
        "candidate_won": candidate_won,
        "search_won": candidate_won,
        "illegal_policy_picks": int(candidate.stats["illegal_policy_picks"]),
        "search_decisions": int(candidate.stats["search_decisions"]),
        "forced_decisions": int(candidate.stats["forced_decisions"]),
        "simulations_used": int(candidate.stats["simulations_used"]),
        "shadow_records_synced": int(candidate.stats["shadow_records_synced"]),
        "elapsed_sec": elapsed,
    }


def _checkpoint_md5(path: str | Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - provenance identity, not cryptography.
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_fingerprint(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _search_recipe(args: Any) -> dict[str, Any]:
    """Canonical runtime/manifest record for the neutral search operator.

    One builder feeds worker construction, the resumability fingerprint, and
    the final summary so a panel cannot execute one D6/adaptive recipe while
    attesting another.
    """
    return {
        "n_full": int(args.n_full),
        # CatanZeroSearchPlayer deliberately calls search(force_full=True) for
        # deterministic evaluation. Record that stronger panel invariant next
        # to the generation-only wide override so the attestation is unambiguous.
        "force_full_every_decision": True,
        "n_full_wide": (
            int(args.n_full_wide) if args.n_full_wide is not None else None
        ),
        "n_full_wide_threshold": (
            int(args.n_full_wide_threshold)
            if args.n_full_wide_threshold is not None
            else None
        ),
        "wide_roots_always_full": bool(args.wide_roots_always_full),
        "max_depth": int(args.max_depth),
        "max_decisions": int(args.max_decisions),
        "c_visit": float(args.c_visit),
        "c_scale": float(args.c_scale),
        "lazy_interior_chance": bool(args.lazy_interior_chance),
        "correct_rust_chance_spectra": bool(args.correct_rust_chance_spectra),
        "public_observation": bool(args.public_observation),
        "prior_temperature": float(args.prior_temperature),
        "value_scale": float(args.value_scale),
        "value_squash": str(args.value_squash),
        "value_readout": str(args.value_readout),
        "max_root_candidates": int(args.max_root_candidates),
        "max_root_candidates_wide": int(args.max_root_candidates_wide),
        "wide_candidates_threshold": int(args.wide_candidates_threshold),
        "symmetry_averaged_eval": bool(args.symmetry_averaged_eval),
        "symmetry_averaged_eval_threshold": (
            int(args.symmetry_averaged_eval_threshold)
            if args.symmetry_averaged_eval_threshold is not None
            else None
        ),
    }


def _game_semantics(args: Any, checkpoint_md5: str) -> dict[str, Any]:
    inference_devices = (
        [item.strip() for item in args.devices.split(",") if item.strip()]
        if args.devices
        else [str(args.device)]
    )
    semantics: dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "mode": str(args.mode),
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "checkpoint_md5": checkpoint_md5,
        "opponent": str(args.opponent),
        "base_seed": int(args.base_seed),
        "vps_to_win": int(args.vps_to_win),
        "sample": bool(args.sample),
        "max_player_trade_offers_per_turn": int(args.max_player_trade_offers_per_turn),
        "map_kind": "BASE" if args.mode == "raw_policy" else SEARCH_MAP_KIND,
        # Backend/thread changes can perturb floating-point tie breaks.  They
        # therefore belong in the no-mixing fingerprint even though worker
        # count itself is only scheduling.
        "inference_devices": inference_devices,
        "threads_per_worker": int(args.threads_per_worker),
    }
    if args.mode == "search":
        semantics["search"] = _search_recipe(args)
    return semantics


def _artifact_path(artifact_dir: Path, pair_id: int, orientation: str) -> Path:
    return artifact_dir / f"pair_{int(pair_id):06d}_{orientation}.json"


def _prepare_manifest(
    artifact_dir: Path,
    *,
    fingerprint: str,
    semantics: dict[str, Any],
    pairs_requested: int,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / "manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("run_fingerprint") != fingerprint:
            raise SystemExit(
                f"artifact directory {artifact_dir} belongs to a different run: "
                f"existing={existing.get('run_fingerprint')} requested={fingerprint}. "
                "Use a new --artifact-dir; incompatible games are never pooled."
            )
    write_json(
        path,
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "run_fingerprint": fingerprint,
            "game_semantics": semantics,
            "pairs_requested": int(pairs_requested),
            "orientations": list(ORIENTATIONS),
        },
    )


def _write_game_artifact(
    artifact_dir: str,
    fingerprint: str,
    record: dict[str, Any],
) -> None:
    path = _artifact_path(
        Path(artifact_dir), int(record["pair_id"]), str(record["orientation"])
    )
    write_json(
        path,
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "run_fingerprint": fingerprint,
            "record": record,
        },
    )


def _load_game_artifacts(
    artifact_dir: Path,
    *,
    fingerprint: str,
) -> dict[tuple[int, str], dict[str, Any]]:
    records: dict[tuple[int, str], dict[str, Any]] = {}
    for path in sorted(artifact_dir.glob("pair_*_candidate_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("run_fingerprint") != fingerprint:
            raise SystemExit(
                f"artifact {path} has incompatible run_fingerprint; refusing to pool it"
            )
        record = dict(payload["record"])
        key = (int(record["pair_id"]), str(record["orientation"]))
        if key in records:
            raise SystemExit(f"duplicate game artifact for {key}: {path}")
        records[key] = record
    return records


def _build_evaluator(worker_args: dict[str, Any]) -> Any:
    return BatchedEntityGraphRustEvaluator.from_checkpoint(
        worker_args["checkpoint"],
        device=worker_args["device"],
        config=EntityGraphRustEvaluatorConfig(
            value_scale=float(worker_args["value_scale"]),
            prior_temperature=float(worker_args["prior_temperature"]),
            value_squash=str(worker_args["value_squash"]),
            value_readout=str(worker_args["value_readout"]),
            public_observation=bool(worker_args["public_observation"]),
        ),
    )


def _validate_checkpoint_value_readout(
    checkpoint: str | Path, *, value_readout: str
) -> tuple[str, ...]:
    """Fail before allocating workers or writing game artifacts on a bad readout.

    A config-only categorical upgrade has random logits.  The shared evaluator
    validator requires positive ``value-training-v1`` provenance, so the neutral
    external panel cannot accidentally certify a categorical model while actually
    searching with an untrained (or scalar fallback) value head.
    """

    policy = EntityGraphPolicy.load(checkpoint, device="cpu")
    _assert_value_readout_available(
        policy,
        EntityGraphRustEvaluatorConfig(value_readout=str(value_readout)),
    )
    return tuple(
        str(readout)
        for readout in getattr(policy, "trained_value_readouts", ("scalar",))
    )


def _failure_record(job: dict[str, Any], error: BaseException) -> dict[str, Any]:
    return {
        "pair_id": int(job["pair_id"]),
        "game_seed": int(job["game_seed"]),
        "orientation": str(job["orientation"]),
        "candidate_color": None,
        "baseline_color": None,
        "winner": None,
        "terminated": False,
        "truncated": False,
        "error": repr(error),
        "engine_divergence": isinstance(error, SearchEngineBoundaryError),
        "divergence_detail": str(error)
        if isinstance(error, SearchEngineBoundaryError)
        else None,
        "decisions": None,
        "candidate_won": None,
        "search_won": None,
        "illegal_policy_picks": 0,
        "search_decisions": 0,
        "simulations_used": 0,
        "elapsed_sec": 0.0,
    }


def _run_worker(worker_args: dict[str, Any]) -> dict[str, Any]:
    threads = int(worker_args["threads_per_worker"])
    if threads > 0:
        import torch

        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)

    evaluator = (
        _build_evaluator(worker_args) if worker_args["mode"] == "search" else None
    )
    records: list[dict[str, Any]] = []
    try:
        for job in worker_args["jobs"]:
            try:
                if worker_args["mode"] == "search":
                    record = play_one_search_game(
                        evaluator=evaluator,
                        search_kwargs=worker_args["search_kwargs"],
                        opponent=worker_args["opponent"],
                        orientation=job["orientation"],
                        pair_id=job["pair_id"],
                        game_seed=job["game_seed"],
                        vps_to_win=worker_args["vps_to_win"],
                        max_decisions=worker_args["max_decisions"],
                    )
                else:
                    record = play_one_raw_game(
                        checkpoint=worker_args["checkpoint"],
                        opponent=worker_args["opponent"],
                        orientation=job["orientation"],
                        pair_id=job["pair_id"],
                        game_seed=job["game_seed"],
                        device=worker_args["device"],
                        vps_to_win=worker_args["vps_to_win"],
                        sample=worker_args["sample"],
                        max_player_trade_offers_per_turn=worker_args[
                            "max_player_trade_offers_per_turn"
                        ],
                    )
            except Exception as exc:  # noqa: BLE001 - persist a resumable failure artifact.
                record = _failure_record(job, exc)
            _write_game_artifact(
                worker_args["artifact_dir"], worker_args["run_fingerprint"], record
            )
            records.append(record)
            print(
                json.dumps(
                    {
                        "progress": "game_done",
                        "worker_index": int(worker_args["worker_index"]),
                        "pair_id": int(record["pair_id"]),
                        "orientation": record["orientation"],
                        "candidate_won": record["candidate_won"],
                        "error": record["error"],
                        "engine_divergence": record.get("engine_divergence", False),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        if evaluator is not None:
            evaluator.close()
    return {"worker_index": int(worker_args["worker_index"]), "records": records}


def _worker_entry(worker_args: dict[str, Any]) -> dict[str, Any]:
    try:
        return _run_worker(worker_args)
    except Exception as exc:  # noqa: BLE001 - parent reports worker-level setup failures.
        return {
            "worker_index": int(worker_args["worker_index"]),
            "records": [],
            "worker_error": repr(exc),
        }


def _wilson_ci(wins: int, games: int, z: float = 1.96) -> list[float] | None:
    if games <= 0:
        return None
    p = wins / games
    denom = 1 + z * z / games
    center = p + z * z / (2 * games)
    half = z * ((p * (1 - p) / games + z * z / (4 * games * games)) ** 0.5)
    return [max(0.0, (center - half) / denom), min(1.0, (center + half) / denom)]


def build_summary(
    args: Any,
    *,
    games: list[dict[str, Any]],
    checkpoint_md5: str,
    run_fingerprint: str,
    artifact_dir: Path,
    elapsed_sec: float,
    games_resumed: int,
    games_run_this_invocation: int,
    worker_errors: list[dict[str, Any]],
    trained_value_readouts: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    games = sorted(
        games,
        key=lambda game: (
            int(game["pair_id"]),
            ORIENTATIONS.index(str(game["orientation"])),
        ),
    )
    outcomes = [
        bool(game["candidate_won"])
        for game in games
        if game.get("candidate_won") is not None
        and not game.get("engine_divergence")
        and game.get("error") is None
    ]
    wins = sum(1 for outcome in outcomes if outcome)
    pair_scores, pair_diagnostics = pair_scores_from_h2h_games(games)
    sprt = evaluate_sprt(
        outcomes=outcomes,
        elo0=float(args.elo0),
        elo1=float(args.elo1),
        alpha=float(args.alpha),
        beta=float(args.beta),
    )
    pentanomial_sprt = evaluate_pentanomial_sprt(
        pair_scores,
        elo0=float(args.elo0),
        elo1=float(args.elo1),
        alpha=float(args.alpha),
        beta=float(args.beta),
    )
    errors = [
        {
            "pair_id": game["pair_id"],
            "orientation": game["orientation"],
            "error": game.get("error"),
            "engine_divergence": bool(game.get("engine_divergence")),
            "divergence_detail": game.get("divergence_detail"),
        }
        for game in games
        if game.get("error") is not None or game.get("engine_divergence")
    ]
    complete_pairs = (
        pair_diagnostics["ww_pairs"]
        + pair_diagnostics["split_pairs"]
        + pair_diagnostics["ll_pairs"]
    )
    search_config = None
    if args.mode == "search":
        search_config = _search_recipe(args)
    win_rate = wins / len(outcomes) if outcomes else None
    return {
        "stratum": "neutral-harness",
        "harness": "catanatron_native_engine",
        "referee_engine": "vendored_python_catanatron",
        "candidate_checkpoint": str(args.checkpoint),
        "candidate_checkpoint_md5": checkpoint_md5,
        "baseline_bot": str(args.opponent),
        "mode": str(args.mode),
        "map_kind": "BASE" if args.mode == "raw_policy" else SEARCH_MAP_KIND,
        # Keep the established search-H2H top-level fields as well as the
        # structured search_config.  sprt_gate.py reads public_observation
        # from here when deriving its typed gate provenance hash.
        "n_full": int(args.n_full) if args.mode == "search" else None,
        "n_full_wide": (
            int(args.n_full_wide)
            if args.mode == "search" and args.n_full_wide is not None
            else None
        ),
        "n_full_wide_threshold": (
            int(args.n_full_wide_threshold)
            if args.mode == "search" and args.n_full_wide_threshold is not None
            else None
        ),
        "wide_roots_always_full": (
            bool(args.wide_roots_always_full) if args.mode == "search" else None
        ),
        "symmetry_averaged_eval": (
            bool(args.symmetry_averaged_eval) if args.mode == "search" else None
        ),
        "symmetry_averaged_eval_threshold": (
            int(args.symmetry_averaged_eval_threshold)
            if args.mode == "search"
            and args.symmetry_averaged_eval_threshold is not None
            else None
        ),
        "wide_candidates_threshold": (
            int(args.wide_candidates_threshold) if args.mode == "search" else None
        ),
        "c_scale": float(args.c_scale) if args.mode == "search" else None,
        "c_visit": float(args.c_visit) if args.mode == "search" else None,
        "lazy_interior_chance": (
            bool(args.lazy_interior_chance) if args.mode == "search" else None
        ),
        "public_observation": (
            bool(args.public_observation) if args.mode == "search" else None
        ),
        "candidate_value_readout": (
            str(args.value_readout) if args.mode == "search" else "scalar"
        ),
        # Search-mode values come from the fail-closed checkpoint preflight,
        # not merely model shape/config. Keep them top-level so a standalone
        # result directly attests positive value-training-v1 provenance.
        "trained_value_readouts": (
            [str(value) for value in trained_value_readouts]
            if args.mode == "search"
            else None
        ),
        "correct_rust_chance_spectra": (
            bool(args.correct_rust_chance_spectra) if args.mode == "search" else None
        ),
        "engine_boundary": (
            "none_raw_policy"
            if args.mode == "raw_policy"
            else "native_python_referee_with_verified_rust_search_shadow"
        ),
        "search_config": search_config,
        "max_player_trade_offers_per_turn": int(args.max_player_trade_offers_per_turn),
        "vps_to_win": int(args.vps_to_win),
        "pairs_requested": int(args.pairs),
        "complete_pairs": complete_pairs,
        "games_requested": int(args.pairs) * 2,
        "games_played": len(games),
        "games_with_winner": len(outcomes),
        "games_truncated": sum(1 for game in games if game.get("truncated")),
        "games_errored": sum(1 for game in games if game.get("error") is not None),
        "games_engine_divergence": sum(
            1 for game in games if game.get("engine_divergence")
        ),
        "candidate_wins": wins,
        "baseline_wins": len(outcomes) - wins,
        "candidate_win_rate": win_rate,
        "candidate_win_rate_wilson_95ci": _wilson_ci(wins, len(outcomes)),
        "total_illegal_policy_picks": sum(
            int(game.get("illegal_policy_picks", 0)) for game in games
        ),
        "total_search_decisions": sum(
            int(game.get("search_decisions", 0)) for game in games
        ),
        "total_simulations_used": sum(
            int(game.get("simulations_used", 0)) for game in games
        ),
        "gate_config": str(args.gate_config),
        "sprt": sprt,
        "pentanomial_sprt": pentanomial_sprt,
        "verdict": pentanomial_sprt["decision"],
        "pair_diagnostics": pair_diagnostics,
        "workers": int(args.workers),
        "threads_per_worker": int(args.threads_per_worker),
        "run_fingerprint": run_fingerprint,
        "artifact_dir": str(artifact_dir),
        "resume": {
            "enabled": bool(args.resume),
            "games_resumed": int(games_resumed),
            "games_run_this_invocation": int(games_run_this_invocation),
        },
        "elapsed_sec": float(elapsed_sec),
        "worker_errors": worker_errors,
        "errors": errors,
        "games": games,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Our checkpoint vs a real catanatron bot, refereed entirely by "
            "catanatron's native Python Game; raw-policy smoke by default, "
            "resumable Gumbel search panel with --mode search."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--opponent", required=True, choices=BOT_KINDS)
    parser.add_argument(
        "--mode", choices=("raw_policy", "search"), default="raw_policy"
    )
    parser.add_argument(
        "--pairs", type=int, default=25, help="paired seeds; total games = 2x this"
    )
    parser.add_argument("--base-seed", type=int, default=1)
    parser.add_argument("--vps-to-win", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--devices",
        default=None,
        help="comma-separated devices assigned round-robin to workers",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads-per-worker", type=int, default=0)
    parser.add_argument(
        "--sample",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="raw-policy mode only: sample instead of greedy argmax",
    )
    parser.add_argument(
        "--max-player-trade-offers-per-turn",
        type=int,
        default=0,
        help="raw-policy only; search panel is pinned to the 2p-no-trade benchmark",
    )

    # Explicit search recipe.  Defaults are the current production search
    # semantics, not the historical unmasked/full-chance diagnostic defaults.
    parser.add_argument("--n-full", type=int, default=64)
    parser.add_argument("--c-scale", type=float, default=0.03)
    parser.add_argument("--c-visit", type=float, default=50.0)
    parser.add_argument(
        "--lazy-interior-chance", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--public-observation", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--correct-rust-chance-spectra",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-depth", type=int, default=80)
    parser.add_argument("--max-decisions", type=int, default=600)
    parser.add_argument("--prior-temperature", type=float, default=1.0)
    parser.add_argument("--value-scale", type=float, default=1.0)
    parser.add_argument("--value-squash", choices=("tanh", "clip"), default="tanh")
    parser.add_argument(
        "--value-readout",
        choices=("scalar", "categorical"),
        default="scalar",
        help=(
            "search mode only: value head consumed by MCTS. Categorical fails "
            "closed unless the checkpoint contains positive value-training-v1 "
            "provenance; no scalar fallback is permitted."
        ),
    )
    parser.add_argument("--max-root-candidates", type=int, default=16)
    parser.add_argument("--max-root-candidates-wide", type=int, default=54)
    parser.add_argument(
        "--wide-candidates-threshold",
        type=int,
        default=24,
        help="Legacy exclusive wide-root candidate-cap threshold and fallback gate.",
    )
    parser.add_argument(
        "--n-full-wide",
        type=int,
        default=None,
        help="Adaptive full-search simulation budget for qualifying wide roots.",
    )
    parser.add_argument(
        "--n-full-wide-threshold",
        type=int,
        default=None,
        help="Inclusive minimum legal-action count for --n-full-wide.",
    )
    parser.add_argument(
        "--wide-roots-always-full",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Attest the generation recipe's wide-root full-search override. The neutral "
            "panel already forces full search at every decision, so this flag does not "
            "weaken or strengthen panel search; it remains fingerprinted provenance."
        ),
    )
    parser.add_argument(
        "--symmetry-averaged-eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Average all 12 D6 orientations at qualifying roots.",
    )
    parser.add_argument(
        "--symmetry-averaged-eval-threshold",
        type=int,
        default=None,
        help="Inclusive minimum legal-action count for D6 root averaging.",
    )

    parser.add_argument(
        "--gate-config", choices=sorted(GATE_CONFIGS), default="certification"
    )
    parser.add_argument("--elo0", type=float, default=None)
    parser.add_argument("--elo1", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--retry-errors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="rerun existing errored/divergent per-game artifacts",
    )
    parser.add_argument("--out", required=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if int(args.pairs) <= 0:
        parser.error("--pairs must be positive")
    if int(args.workers) <= 0:
        parser.error("--workers must be positive")
    if int(args.vps_to_win) <= 0:
        parser.error("--vps-to-win must be positive")
    if args.mode == "search" and bool(args.sample):
        parser.error(
            "--sample is raw-policy-only; search panel roots use deterministic argmax"
        )
    if args.mode == "search" and int(args.max_player_trade_offers_per_turn) != 0:
        parser.error(
            "search neutral harness is pinned to --max-player-trade-offers-per-turn 0"
        )
    if args.mode != "search" and str(args.value_readout) != "scalar":
        parser.error("--value-readout categorical is supported only with --mode search")
    if args.devices and not [
        item.strip() for item in args.devices.split(",") if item.strip()
    ]:
        parser.error("--devices must contain at least one device")
    if args.mode == "search":
        for flag, value in (
            ("--n-full", args.n_full),
            ("--max-depth", args.max_depth),
            ("--max-decisions", args.max_decisions),
            ("--max-root-candidates", args.max_root_candidates),
            ("--max-root-candidates-wide", args.max_root_candidates_wide),
        ):
            if int(value) <= 0:
                parser.error(f"{flag} must be positive in search mode")
        if args.n_full_wide is not None and int(args.n_full_wide) <= 0:
            parser.error("--n-full-wide must be positive in search mode")
        if bool(args.wide_roots_always_full) and args.n_full_wide is None:
            parser.error("--wide-roots-always-full requires --n-full-wide")

    _gate, gate_params = resolve_gate_config(
        args.gate_config,
        elo0=args.elo0,
        elo1=args.elo1,
        alpha=args.alpha,
        beta=args.beta,
    )
    args.elo0 = gate_params["elo0"]
    args.elo1 = gate_params["elo1"]
    args.alpha = gate_params["alpha"]
    args.beta = gate_params["beta"]

    threads_per_worker = int(args.threads_per_worker)
    if threads_per_worker <= 0:
        threads_per_worker = max(
            1, (os.cpu_count() or int(args.workers)) // int(args.workers)
        )
    args.threads_per_worker = threads_per_worker

    checkpoint_md5 = _checkpoint_md5(args.checkpoint)
    trained_value_readouts = ("scalar",)
    if args.mode == "search":
        try:
            trained_value_readouts = _validate_checkpoint_value_readout(
                args.checkpoint, value_readout=str(args.value_readout)
            )
        except (OSError, KeyError, TypeError, ValueError) as error:
            parser.error(f"checkpoint value-readout preflight failed: {error}")
    semantics = _game_semantics(args, checkpoint_md5)
    if args.mode == "search":
        semantics["trained_value_readouts"] = list(trained_value_readouts)
    fingerprint = _run_fingerprint(semantics)
    out_path = Path(args.out)
    artifact_dir = (
        Path(args.artifact_dir)
        if args.artifact_dir
        else out_path.parent / f"{out_path.stem}.games"
    )
    _prepare_manifest(
        artifact_dir,
        fingerprint=fingerprint,
        semantics=semantics,
        pairs_requested=int(args.pairs),
    )
    existing = _load_game_artifacts(artifact_dir, fingerprint=fingerprint)
    if not bool(args.resume):
        existing = {}

    planned_jobs = [
        {
            "pair_id": pair_id,
            "game_seed": int(args.base_seed) + pair_id,
            "orientation": orientation,
        }
        for pair_id in range(int(args.pairs))
        for orientation in ORIENTATIONS
    ]
    planned_keys = {
        (int(job["pair_id"]), str(job["orientation"])) for job in planned_jobs
    }
    # Reusing a directory to extend a panel is supported; asking for a smaller
    # prefix must not silently pool artifacts outside the requested plan.
    existing = {key: record for key, record in existing.items() if key in planned_keys}
    pending_jobs: list[dict[str, Any]] = []
    resumed_games = 0
    for job in planned_jobs:
        key = (int(job["pair_id"]), str(job["orientation"]))
        record = existing.get(key)
        retryable = record is not None and (
            record.get("error") is not None or record.get("engine_divergence")
        )
        if record is None or (bool(args.retry_errors) and retryable):
            pending_jobs.append(job)
        else:
            resumed_games += 1

    workers = min(max(1, int(args.workers)), max(1, len(pending_jobs)))
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = str(threads_per_worker)

    devices = (
        [item.strip() for item in args.devices.split(",") if item.strip()]
        if args.devices
        else [str(args.device)]
    )
    if not devices:
        parser.error("--devices must contain at least one device")
    shards: list[list[dict[str, Any]]] = [[] for _ in range(workers)]
    for index, job in enumerate(pending_jobs):
        shards[index % workers].append(job)

    search_kwargs = _search_recipe(args)
    worker_args = [
        {
            "worker_index": index,
            "jobs": shard,
            "mode": str(args.mode),
            "checkpoint": str(args.checkpoint),
            "opponent": str(args.opponent),
            "device": devices[index % len(devices)],
            "threads_per_worker": threads_per_worker,
            "artifact_dir": str(artifact_dir),
            "run_fingerprint": fingerprint,
            "vps_to_win": int(args.vps_to_win),
            "max_decisions": int(args.max_decisions),
            "sample": bool(args.sample),
            "max_player_trade_offers_per_turn": int(
                args.max_player_trade_offers_per_turn
            ),
            "search_kwargs": search_kwargs,
            "prior_temperature": float(args.prior_temperature),
            "value_scale": float(args.value_scale),
            "value_squash": str(args.value_squash),
            "value_readout": str(args.value_readout),
            "public_observation": bool(args.public_observation),
        }
        for index, shard in enumerate(shards)
        if shard
    ]

    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    if len(worker_args) == 1:
        results = [_worker_entry(worker_args[0])]
    elif worker_args:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=len(worker_args)) as pool:
            for result in pool.imap_unordered(_worker_entry, worker_args):
                results.append(result)
    elapsed = time.perf_counter() - started

    worker_errors = [
        {
            "worker_index": result.get("worker_index"),
            "error": result["worker_error"],
        }
        for result in results
        if result.get("worker_error")
    ]
    all_records = {
        key: record
        for key, record in _load_game_artifacts(
            artifact_dir, fingerprint=fingerprint
        ).items()
        if key in planned_keys
    }
    summary = build_summary(
        args,
        games=list(all_records.values()),
        checkpoint_md5=checkpoint_md5,
        run_fingerprint=fingerprint,
        artifact_dir=artifact_dir,
        elapsed_sec=elapsed,
        games_resumed=resumed_games,
        games_run_this_invocation=sum(
            len(result.get("records", ())) for result in results
        ),
        worker_errors=worker_errors,
        trained_value_readouts=trained_value_readouts,
    )
    write_json(out_path, summary)
    print(
        json.dumps(
            {
                key: value
                for key, value in summary.items()
                if key not in ("games", "errors")
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
