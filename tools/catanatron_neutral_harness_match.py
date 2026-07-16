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
import importlib
import importlib.machinery
import json
import multiprocessing
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from catan_zero.adapters.engine_equivalence import EquivalenceConfig, build_paired_games
from catan_zero.rl._catanatron import import_catanatron_module
from catan_zero.rl.entity_token_features_rust import require_rust_feature_path
from catan_zero.rl.entity_token_policy import EntityGraphPolicy
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
        wide_candidates_threshold=int(
            search_kwargs.get("wide_candidates_threshold", 24)
        ),
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
        wide_roots_always_full=bool(search_kwargs.get("wide_roots_always_full", False)),
        symmetry_averaged_eval=bool(search_kwargs.get("symmetry_averaged_eval", False)),
        symmetry_averaged_eval_threshold=(
            int(search_kwargs["symmetry_averaged_eval_threshold"])
            if search_kwargs.get("symmetry_averaged_eval_threshold") is not None
            else None
        ),
        information_set_search=bool(search_kwargs.get("information_set_search", False)),
        coherent_public_belief_search=bool(
            search_kwargs.get("coherent_public_belief_search", False)
        ),
        forced_root_target_mode=str(
            search_kwargs.get("forced_root_target_mode", "full")
        ),
        boundary_value_particles=int(
            search_kwargs.get("boundary_value_particles", 1)
        ),
        determinization_particles=int(
            search_kwargs.get("determinization_particles", 1)
        ),
        determinization_min_simulations=int(
            search_kwargs.get("determinization_min_simulations", 32)
        ),
        belief_chance_spectra=bool(search_kwargs.get("belief_chance_spectra", False)),
        gameplay_policy_aggregation=str(
            search_kwargs.get("gameplay_policy_aggregation", "mean_improved_policy")
        ),
        sigma_reference_visits=(
            int(search_kwargs["sigma_reference_visits"])
            if search_kwargs.get("sigma_reference_visits") is not None
            else None
        ),
        rescale_noise_floor_c=float(search_kwargs.get("rescale_noise_floor_c", 0.0)),
        sigma_eval=float(search_kwargs.get("sigma_eval", 0.79)),
        raw_policy_above_width=search_kwargs.get("raw_policy_above_width"),
        exact_budget_sh=bool(search_kwargs.get("exact_budget_sh", False)),
        exact_budget_sh_min_n=int(search_kwargs.get("exact_budget_sh_min_n", 0)),
        root_wave_batching=bool(search_kwargs.get("root_wave_batching", False)),
        play_sh_winner=bool(search_kwargs.get("play_sh_winner", False)),
        use_batch_api=bool(search_kwargs.get("use_batch_api", True)),
        policy_target_min_visits=int(search_kwargs.get("policy_target_min_visits", 0)),
        uncertainty_backup_weighting=bool(
            search_kwargs.get("uncertainty_backup_weighting", False)
        ),
        uncertainty_backup_a=float(search_kwargs.get("uncertainty_backup_a", 0.25)),
        uncertainty_backup_exp=float(search_kwargs.get("uncertainty_backup_exp", 1.0)),
        uncertainty_backup_cap=float(search_kwargs.get("uncertainty_backup_cap", 1.0)),
        variance_aware_q=bool(search_kwargs.get("variance_aware_q", False)),
        variance_aware_k=float(search_kwargs.get("variance_aware_k", 1.0)),
        variance_aware_closed_form_js=bool(
            search_kwargs.get("variance_aware_closed_form_js", False)
        ),
    )


def _create_search(
    config: GumbelChanceMCTSConfig,
    evaluator: Any,
    *,
    native_mcts_hot_loop: bool,
) -> GumbelChanceMCTS:
    if not native_mcts_hot_loop:
        return GumbelChanceMCTS(config, evaluator)
    return create_gumbel_search(config, evaluator, native_hot_loop=True)


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
    search = _create_search(
        _search_config(search_kwargs, seated_colors, int(game_seed)),
        evaluator,
        native_mcts_hot_loop=bool(search_kwargs.get("native_mcts_hot_loop", False)),
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


def _checkpoint_digests(path: str | Path) -> tuple[str, str]:
    """Compute both required provenance digests in one checkpoint read."""
    md5 = hashlib.md5()  # noqa: S324 - provenance identity, not cryptography.
    sha256 = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            md5.update(block)
            sha256.update(block)
    return md5.hexdigest(), "sha256:" + sha256.hexdigest()


def _checkpoint_md5(path: str | Path) -> str:
    return _checkpoint_digests(path)[0]


def _checkpoint_sha256(path: str | Path) -> str:
    return _checkpoint_digests(path)[1]


def _native_runtime_extension_path() -> Path:
    """Return the compiled catanatron_rs extension, never its package shim.

    ``catanatron_rs.__file__`` points at the tiny Python ``__init__.py`` that
    wildcard-imports the actual extension. Hashing that shim made unrelated
    native builds look identical in external-panel provenance.
    """

    native = importlib.import_module("catanatron_rs.catanatron_rs")
    raw_path = getattr(native, "__file__", None)
    if not isinstance(raw_path, str) or not raw_path:
        raise RuntimeError("catanatron_rs native extension has no __file__")
    path = Path(raw_path).resolve(strict=True)
    if not any(
        str(path).endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES
    ):
        raise RuntimeError(
            "catanatron_rs native implementation did not resolve to a compiled "
            f"extension: {path}"
        )
    return path


def _run_fingerprint(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _engine_identity(args: Any) -> dict[str, Any]:
    return {
        "schema_version": "a1-neutral-engine-identity-v1",
        "repo_commit": getattr(args, "engine_repo_commit", None),
        "native_wheel_sha256": getattr(args, "native_wheel_sha256", None),
        "native_runtime_sha256": getattr(args, "native_runtime_sha256", None),
        "python_referee_sha256": getattr(args, "python_referee_sha256", None),
    }


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
        "information_set_search": bool(args.information_set_search),
        "coherent_public_belief_search": bool(
            getattr(args, "coherent_public_belief_search", False)
        ),
        "forced_root_target_mode": str(
            getattr(args, "forced_root_target_mode", "full")
        ),
        "boundary_value_particles": int(
            getattr(args, "boundary_value_particles", 1)
        ),
        "determinization_particles": int(args.determinization_particles),
        "determinization_min_simulations": int(args.determinization_min_simulations),
        "prior_temperature": float(args.prior_temperature),
        "value_scale": float(args.value_scale),
        "value_squash": str(args.value_squash),
        "value_readout": str(args.value_readout),
        "max_root_candidates": int(args.max_root_candidates),
        "max_root_candidates_wide": int(args.max_root_candidates_wide),
        "wide_candidates_threshold": int(args.wide_candidates_threshold),
        "n_fast": int(args.n_full),
        "p_full": 1.0,
        "temperature": 0.0,
        "play_sh_winner": False,
        "belief_chance_spectra": bool(getattr(args, "belief_chance_spectra", False)),
        "gameplay_policy_aggregation": str(
            getattr(args, "gameplay_policy_aggregation", "mean_improved_policy")
        ),
        "sigma_reference_visits": (
            int(args.sigma_reference_visits)
            if getattr(args, "sigma_reference_visits", None) is not None
            else None
        ),
        "rescale_noise_floor_c": float(getattr(args, "rescale_noise_floor_c", 0.0)),
        "sigma_eval": float(getattr(args, "sigma_eval", 0.98)),
        "raw_policy_above_width": None,
        "exact_budget_sh": False,
        "exact_budget_sh_min_n": 0,
        "root_wave_batching": False,
        "use_batch_api": True,
        "policy_target_min_visits": 0,
        "uncertainty_backup_weighting": False,
        "uncertainty_backup_a": 0.25,
        "uncertainty_backup_exp": 1.0,
        "uncertainty_backup_cap": 1.0,
        "variance_aware_q": False,
        "variance_aware_k": 1.0,
        "variance_aware_closed_form_js": False,
        "evaluator_context_fill": 0.0,
        "evaluator_cache_size": 0,
        "evaluator_rust_featurize": bool(
            getattr(args, "evaluator_rust_featurize", False)
        ),
        "native_mcts_hot_loop": bool(getattr(args, "native_mcts_hot_loop", False)),
        "mcts_implementation": (
            "rust_native_hot_loop_v1"
            if bool(getattr(args, "native_mcts_hot_loop", False))
            else "python_reference"
        ),
        "evaluator_emit_uncertainty": False,
        "symmetry_averaged_eval": bool(args.symmetry_averaged_eval),
        "symmetry_averaged_eval_threshold": (
            int(args.symmetry_averaged_eval_threshold)
            if args.symmetry_averaged_eval_threshold is not None
            else None
        ),
    }


def _game_semantics(
    args: Any, checkpoint_md5: str, checkpoint_sha256: str
) -> dict[str, Any]:
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
        "checkpoint_sha256": checkpoint_sha256,
        "opponent": str(args.opponent),
        "base_seed": int(getattr(args, "base_seed", 1)),
        "vps_to_win": int(args.vps_to_win),
        "sample": bool(args.sample),
        "max_player_trade_offers_per_turn": int(args.max_player_trade_offers_per_turn),
        "map_kind": "BASE" if args.mode == "raw_policy" else SEARCH_MAP_KIND,
        # Backend/thread changes can perturb floating-point tie breaks.  They
        # therefore belong in the no-mixing fingerprint even though worker
        # count itself is only scheduling.
        "inference_devices": inference_devices,
        "threads_per_worker": int(args.threads_per_worker),
        # Retry/resume artifacts are reusable only under the exact engine
        # build. Search flags alone do not identify native or referee code.
        "engine_identity": _engine_identity(args),
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
            context_fill=float(worker_args.get("evaluator_context_fill", 0.0)),
            cache_size=int(worker_args.get("evaluator_cache_size", 0)),
            value_squash=str(worker_args["value_squash"]),
            value_readout=str(worker_args["value_readout"]),
            public_observation=bool(worker_args["public_observation"]),
            rust_featurize=bool(worker_args.get("evaluator_rust_featurize", False)),
            emit_uncertainty=bool(worker_args.get("evaluator_emit_uncertainty", False)),
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
    checkpoint_sha256: str,
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
        "engine_identity": _engine_identity(args),
        "candidate_checkpoint": str(args.checkpoint),
        "candidate_checkpoint_md5": checkpoint_md5,
        "candidate_checkpoint_sha256": checkpoint_sha256,
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
        "information_set_search": (
            bool(args.information_set_search) if args.mode == "search" else None
        ),
        "coherent_public_belief_search": (
            bool(getattr(args, "coherent_public_belief_search", False))
            if args.mode == "search"
            else None
        ),
        "forced_root_target_mode": (
            str(getattr(args, "forced_root_target_mode", "full"))
            if args.mode == "search"
            else None
        ),
        "boundary_value_particles": (
            int(getattr(args, "boundary_value_particles", 1))
            if args.mode == "search"
            else None
        ),
        "determinization_particles": (
            int(args.determinization_particles) if args.mode == "search" else None
        ),
        "determinization_min_simulations": (
            int(args.determinization_min_simulations) if args.mode == "search" else None
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
        "base_seed": int(getattr(args, "base_seed", 1)),
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


def _validate_public_search_recipe(args: Any) -> None:
    """Require exactly one public-information tree operator in search mode."""
    public = bool(args.public_observation)
    information_set = bool(args.information_set_search)
    coherent = bool(getattr(args, "coherent_public_belief_search", False))
    belief_spectra = bool(args.belief_chance_spectra)

    if not public:
        raise ValueError("search mode requires --public-observation")
    if information_set == coherent:
        raise ValueError(
            "search mode requires exactly one of --information-set-search or "
            "--coherent-public-belief-search"
        )
    if coherent and belief_spectra:
        raise ValueError(
            "--coherent-public-belief-search cannot be combined with "
            "--belief-chance-spectra"
        )
    if information_set and belief_spectra:
        raise ValueError(
            "--information-set-search cannot be combined with --belief-chance-spectra"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Our checkpoint vs a real catanatron bot, refereed entirely by "
            "catanatron's native Python Game; raw-policy smoke by default, "
            "resumable Gumbel search panel with --mode search."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--engine-repo-commit")
    parser.add_argument("--native-wheel-sha256")
    parser.add_argument("--python-referee-sha256")
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
        "--rescale-noise-floor-c",
        type=float,
        default=0.0,
        help="Completed-Q noise-floor attenuation (sealed A1 default: disabled).",
    )
    parser.add_argument(
        "--sigma-eval",
        type=float,
        default=0.98,
        help="Value-noise estimate used by completed-Q attenuation.",
    )
    parser.add_argument(
        "--gameplay-policy-aggregation",
        choices=("mean_improved_policy", "aggregate_q_then_improve"),
        default="mean_improved_policy",
        help="Public-belief action-selection operator (legacy default unchanged).",
    )
    parser.add_argument(
        "--sigma-reference-visits",
        type=int,
        default=None,
        help="Fixed completed-Q sigma visit reference required by corrected belief gameplay.",
    )
    parser.add_argument(
        "--lazy-interior-chance", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--public-observation", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--information-set-search",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Search public-belief determinizations instead of cloning authoritative "
            "hidden truth. Exactly one of this PIMC mode or "
            "--coherent-public-belief-search is required for a search panel."
        ),
    )
    parser.add_argument(
        "--coherent-public-belief-search",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use one actor-turn tree rooted in a public-belief state, with hidden "
            "chance materialized from public support. Requires --public-observation "
            "and is mutually exclusive with PIMC --information-set-search and "
            "--belief-chance-spectra."
        ),
    )
    parser.add_argument(
        "--forced-root-target-mode",
        choices=("full", "trajectory_only"),
        default="full",
        help=(
            "Use trajectory_only to skip neural/search work at single-action "
            "prompts; the played action is mathematically unchanged."
        ),
    )
    parser.add_argument("--boundary-value-particles", type=int, default=1)
    parser.add_argument(
        "--belief-chance-spectra",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Legacy chance-spectrum belief correction (PIMC supersedes it in A1).",
    )
    parser.add_argument("--determinization-particles", type=int, default=4)
    parser.add_argument("--determinization-min-simulations", type=int, default=32)
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
        "--evaluator-rust-featurize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Search mode only: build entity and legal-action context tensors "
            "with the bit-exact native featurizer. Opt-in and fail-closed."
        ),
    )
    parser.add_argument(
        "--native-mcts-hot-loop",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Search mode only: explicitly use the feature-gated Rust MCTS tree "
            "hot loop. Default Python; enabling fails closed if the matching "
            "wheel is absent."
        ),
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
    if args.mode == "search":
        engine_identity = _engine_identity(args)
        if not re.fullmatch(r"[0-9a-f]{40}", str(engine_identity["repo_commit"])):
            parser.error("search mode requires --engine-repo-commit")
        for name in ("native_wheel_sha256", "python_referee_sha256"):
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(engine_identity[name])):
                parser.error(f"search mode requires --{name.replace('_', '-')}")
        try:
            native_runtime = _native_runtime_extension_path()
            args.native_runtime_sha256 = _checkpoint_sha256(native_runtime)
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
            parser.error(f"cannot fingerprint installed native runtime: {error}")
    if args.mode == "search" and int(args.max_player_trade_offers_per_turn) != 0:
        parser.error(
            "search neutral harness is pinned to --max-player-trade-offers-per-turn 0"
        )
    if args.mode != "search" and str(args.value_readout) != "scalar":
        parser.error("--value-readout categorical is supported only with --mode search")
    if args.mode != "search" and bool(args.evaluator_rust_featurize):
        parser.error("--evaluator-rust-featurize is supported only with --mode search")
    if args.mode != "search" and bool(args.native_mcts_hot_loop):
        parser.error("--native-mcts-hot-loop is supported only with --mode search")
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
    if args.devices and not [
        item.strip() for item in args.devices.split(",") if item.strip()
    ]:
        parser.error("--devices must contain at least one device")
    if args.mode == "search":
        try:
            _validate_public_search_recipe(args)
        except ValueError as error:
            parser.error(str(error))
        if int(args.determinization_particles) < 1:
            parser.error("--determinization-particles must be >= 1")
        if int(args.boundary_value_particles) < 1:
            parser.error("--boundary-value-particles must be >= 1")
        if (
            int(args.boundary_value_particles) > 1
            and not bool(args.coherent_public_belief_search)
        ):
            parser.error(
                "--boundary-value-particles > 1 requires "
                "--coherent-public-belief-search"
            )
        if int(args.determinization_min_simulations) < 1:
            parser.error("--determinization-min-simulations must be >= 1")
        if (
            args.sigma_reference_visits is not None
            and int(args.sigma_reference_visits) < 0
        ):
            parser.error("--sigma-reference-visits must be non-negative")
        if (
            str(args.gameplay_policy_aggregation) == "aggregate_q_then_improve"
            and args.sigma_reference_visits is None
        ):
            parser.error(
                "--gameplay-policy-aggregation aggregate_q_then_improve requires "
                "--sigma-reference-visits"
            )
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

    checkpoint_md5, checkpoint_sha256 = _checkpoint_digests(args.checkpoint)
    trained_value_readouts = ("scalar",)
    if args.mode == "search":
        try:
            trained_value_readouts = _validate_checkpoint_value_readout(
                args.checkpoint, value_readout=str(args.value_readout)
            )
        except (OSError, KeyError, TypeError, ValueError) as error:
            parser.error(f"checkpoint value-readout preflight failed: {error}")
    semantics = _game_semantics(args, checkpoint_md5, checkpoint_sha256)
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
            "evaluator_rust_featurize": bool(args.evaluator_rust_featurize),
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
        checkpoint_sha256=checkpoint_sha256,
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
