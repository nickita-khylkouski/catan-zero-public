#!/usr/bin/env python3
"""CLI: cross-checkpoint search-vs-search H2H gate (gen-1 flywheel G1 gate).

Adapted from tools/gumbel_search_vs_raw_h2h.py (task #53 part 2), which plays
GumbelChanceMCTS search vs the SAME checkpoint's raw policy to test whether
search adds strength over one net. This variant instead plays TWO DIFFERENT
checkpoints against each other, BOTH using GumbelChanceMCTS search with an
identical config by default -- isolating the CHECKPOINT's contribution (does a
distilled/trained net beat its teacher under search). CAT-25 diagnostic flags
can deliberately override ``n_full`` / ``n_full_wide`` by role to measure
search-budget headroom; those effective per-role budgets are recorded and
hashed in the output so such a run cannot masquerade as a checkpoint-only
gate.  Role-specific ``c_scale`` flags likewise support a fair paired-seed
comparison of each checkpoint under its independently tuned search operator;
the effective values are recorded and hashed for the same reason.

Games are paired by seed AND color-swapped (each seed is played twice, once
with candidate=RED/baseline=BLUE and once swapped) to cancel positional/color
bias, the same paired-seed H2H protocol used by
tools/gumbel_search_vs_raw_h2h.py and tools/evaluate_scoreboard.py.

Per-game outcomes feed tools/sprt_gate.py's evaluate_sprt /
evaluate_pentanomial_sprt (elo0=0, elo1=30 -- the >=55%-win-rate promotion
bar). Truncated games (no winner within --max-decisions) are recorded but
EXCLUDED from the SPRT input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import re
import subprocess
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

from catan_zero.rl.config_cli import add_config_flags, resolve_config  # noqa: E402
from catan_zero.rl.entity_token_features_rust import (  # noqa: E402
    require_rust_feature_path,
)
from catan_zero.rl.gumbel_self_play import _apply_selected_action  # noqa: E402
from catan_zero.rl.pipeline_configs import EvalConfig  # noqa: E402
from catan_zero.search.gumbel_chance_mcts import (  # noqa: E402
    GumbelChanceMCTS,
    GumbelChanceMCTSConfig,
    _matches_explicit_or_legacy_width_gate,
)
from catan_zero.search.native_gumbel_mcts import (  # noqa: E402
    create_gumbel_search,
    native_hot_loop_available,
)
from catan_zero.search.neural_rust_mcts import (  # noqa: E402
    BatchedEntityGraphRustEvaluator,
    EntityGraphRustEvaluatorConfig,
)
from catan_zero.search.rust_mcts import _require_rust_module  # noqa: E402
from factory_common import write_json  # noqa: E402
from tools.high_regret_suite_contract import (  # noqa: E402
    PinnedReplayScope,
    REPLAY_CONTRACT,
    SUITE_SCHEMA,
    bind_state_to_manifest,
    load_source_manifest,
    load_source_validation_binding,
    pin_replay_scope,
    scope_inventory_sha256,
    validate_replay_metadata,
    validate_replay_trajectories,
)


HIGH_REGRET_ENGINE_IDENTITY_SCHEMA = "a1-high-regret-engine-identity-v1"
ARCHIVED_STATE_RECONSTRUCTION_SCHEMA = "a1-archived-state-reconstruction-v1"


def _held_out_engine_identity(args: Any) -> tuple[dict[str, str], dict[str, str]]:
    """Bind a held-out run to clean Git, wheel, runtime, and replay bytes."""

    commit = str(getattr(args, "engine_repo_commit", "") or "")
    wheel_sha = str(getattr(args, "native_wheel_sha256", "") or "")
    wheel_raw = str(getattr(args, "native_wheel_path", "") or "")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("held-out high-regret requires --engine-repo-commit")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", wheel_sha):
        raise ValueError("held-out high-regret requires --native-wheel-sha256")
    if not wheel_raw:
        raise ValueError("held-out high-regret requires --native-wheel-path")
    wheel = Path(wheel_raw).expanduser().resolve(strict=True)
    if not wheel.is_file() or wheel.is_symlink():
        raise ValueError("held-out high-regret native wheel is not a regular file")
    if _checkpoint_sha256(wheel) != wheel_sha:
        raise ValueError("held-out high-regret native wheel digest mismatch")
    try:
        actual_commit = subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=_REPO_ROOT, text=True
        ).strip()
        subprocess.run(
            ("git", "diff", "--quiet", "HEAD"), cwd=_REPO_ROOT, check=True
        )
        subprocess.run(
            ("git", "diff", "--cached", "--quiet", "HEAD"),
            cwd=_REPO_ROOT,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("held-out high-regret checkout is not clean Git bytes") from error
    if actual_commit != commit:
        raise ValueError("held-out high-regret repo commit differs from checked-out HEAD")
    try:
        import catanatron_rs

        runtime = Path(catanatron_rs.__file__).resolve(strict=True)
    except (ImportError, OSError, TypeError, ValueError) as error:
        raise ValueError("cannot fingerprint held-out native runtime") from error
    planned = {
        "schema_version": HIGH_REGRET_ENGINE_IDENTITY_SCHEMA,
        "repo_commit": commit,
        "native_wheel_sha256": wheel_sha,
        "evaluator_sha256": _checkpoint_sha256(Path(__file__).resolve()),
        "replay_sha256": _checkpoint_sha256(_TOOLS_DIR / "reconstruct_state.py"),
    }
    return planned, {**planned, "native_runtime_sha256": _checkpoint_sha256(runtime)}


def _archived_state_reconstruction_binding() -> dict[str, Any]:
    return {
        "schema_version": ARCHIVED_STATE_RECONSTRUCTION_SCHEMA,
        "constructor": "catanatron_rs.Game.simple",
        "map_kind": "BASE",
        "action_prefix": "[0,target_decision)",
        "chance_stream": "random.Random(game_seed ^ 0xA17E)",
        "replay_contract": REPLAY_CONTRACT,
    }
from sprt_gate import (  # noqa: E402
    GATE_CONFIGS,
    evaluate_pentanomial_sprt,
    evaluate_sprt,
    pair_scores_from_h2h_games,
    resolve_gate_config,
)

COLORS: tuple[str, ...] = ("RED", "BLUE")


def _promotion_phase_bucket(phases: set[str]) -> str:
    upper = " ".join(phases).upper()
    if "BUILD_INITIAL_SETTLEMENT" in upper or "BUILD_INITIAL_ROAD" in upper:
        return "opening"
    if "ROBBER" in upper or "KNIGHT" in upper or "DEVELOPMENT_CARD" in upper:
        return "robber_dev"
    if "DISCARD" in upper or "ROLL" in upper:
        return "chance"
    return "build_trade"


def _checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _load_held_out_high_regret_suite(
    path: str | Path,
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    """Load and replay the immutable suite envelope before any GPU work."""

    suite_path = Path(path).expanduser().resolve()
    try:
        suite = json.loads(suite_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load held-out suite: {error}") from error
    expected_keys = {
        "schema_version",
        "suite",
        "held_out",
        "source_manifest",
        "validation_seed_manifest",
        "selection",
        "states",
        "suite_sha256",
    }
    if not isinstance(suite, dict) or set(suite) != expected_keys:
        raise ValueError("held-out suite has an unexpected schema")
    if (
        suite["schema_version"] != SUITE_SCHEMA
        or suite["suite"] != "held_out_high_regret"
        or suite["held_out"] is not True
    ):
        raise ValueError("held-out suite identity is invalid")
    unhashed = dict(suite)
    declared_digest = unhashed.pop("suite_sha256")
    actual_digest = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    )
    if declared_digest != actual_digest:
        raise ValueError("held-out suite semantic digest mismatch")
    source_ref = suite["source_manifest"]
    if not isinstance(source_ref, dict) or set(source_ref) != {"path", "sha256"}:
        raise ValueError("held-out suite source manifest reference is malformed")
    source_path = Path(str(source_ref["path"])).expanduser()
    if not source_path.is_absolute():
        source_path = suite_path.parent / source_path
    source_path = source_path.resolve()
    if (
        not source_path.is_file()
        or _checkpoint_sha256(source_path) != source_ref["sha256"]
    ):
        raise ValueError("held-out suite source manifest is missing or drifted")
    shard_paths, manifest_identities = load_source_manifest(source_path)
    allowed_seeds, validation_binding = load_source_validation_binding(source_path)
    if suite["validation_seed_manifest"] != validation_binding:
        raise ValueError("held-out suite validation-seed binding drifted")
    selection = suite["selection"]
    states = suite["states"]
    if (
        not isinstance(selection, dict)
        or selection.get("algorithm") != "stable-hash-holdout-stratified-regret-v1"
        or not isinstance(states, list)
        or not states
        or selection.get("selected_pairs") != len(states)
    ):
        raise ValueError("held-out suite selection is malformed")
    validate_replay_metadata(selection, states)
    expected_strata = {
        "phase:opening",
        "phase:robber_dev",
        "phase:chance",
        "phase:build_trade",
        "41+",
    }
    selected_by_stratum = selection.get("selected_by_stratum")
    stratum_min_pairs = selection.get("stratum_min_pairs")
    if (
        selection.get("holdout_fraction") != 0.10
        or selection.get("holdout_seed") != 17
        or isinstance(stratum_min_pairs, bool)
        or not isinstance(stratum_min_pairs, int)
        or stratum_min_pairs < 4
        or not isinstance(selected_by_stratum, dict)
        or set(selected_by_stratum) != expected_strata
        or any(value != stratum_min_pairs for value in selected_by_stratum.values())
    ):
        raise ValueError("held-out suite does not satisfy the fixed stratified policy")
    pairs: list[dict[str, Any]] = []
    pair_ids: set[int] = set()
    bound_states: list[dict[str, Any]] = []
    inventory_cache: dict[Path, tuple[str, int]] = {}
    source_row_cache: dict[Path, tuple[Any, Any, int]] = {}
    for index, raw_state in enumerate(states):
        try:
            state = bind_state_to_manifest(
                raw_state,
                suite_base=suite_path.parent,
                manifest_path=source_path,
                shard_paths=shard_paths,
                identities=manifest_identities,
                inventory_cache=inventory_cache,
                source_row_cache=source_row_cache,
            )
        except ValueError as error:
            raise ValueError(f"held-out suite state {index}: {error}") from error
        pair_id = state.get("pair_id")
        game_seed = state.get("game_seed")
        decision_index = state.get("decision_index")
        legal_count = state.get("legal_count")
        if (
            isinstance(pair_id, bool)
            or not isinstance(pair_id, int)
            or pair_id < 0
            or pair_id in pair_ids
            or isinstance(game_seed, bool)
            or not isinstance(game_seed, int)
            or isinstance(decision_index, bool)
            or not isinstance(decision_index, int)
            or decision_index < 0
            or isinstance(legal_count, bool)
            or not isinstance(legal_count, int)
            or legal_count < 0
        ):
            raise ValueError(f"held-out suite state {index} lacks valid identity")
        pair_ids.add(pair_id)
        if game_seed not in allowed_seeds:
            raise ValueError(
                f"held-out suite state {index} is outside the training validation set"
            )
        bound_states.append(state)
        pairs.append(
            {
                "pair_id": pair_id,
                "game_seed": game_seed,
                "archived_state": state,
            }
        )
    actual_strata = {
        f"phase:{stratum}": sum(
            _promotion_phase_bucket({str(state.get("phase", ""))}) == stratum
            for state in bound_states
        )
        for stratum in ("opening", "robber_dev", "chance", "build_trade")
    }
    actual_strata["41+"] = sum(state["legal_count"] >= 41 for state in bound_states)
    if any(actual_strata[label] < stratum_min_pairs for label in expected_strata):
        raise ValueError("held-out suite retained states do not cover every stratum")
    validate_replay_trajectories(bound_states)
    return suite_path, suite, pairs


def _validate_archived_scope_inventory(
    archived: dict[str, Any],
    cache: dict[Path, tuple[str, int]],
) -> None:
    """Worker-side replay check after process handoff, before reconstruction."""

    shard_path = Path(str(archived["shard_path"])).resolve(strict=True)
    scope = shard_path.parent
    actual = cache.get(scope)
    if actual is None:
        actual = scope_inventory_sha256(scope)
        cache[scope] = actual
    replay_source = archived.get("replay_source")
    if not isinstance(replay_source, dict) or actual != (
        replay_source.get("scope_inventory_sha256"),
        replay_source.get("scope_shard_count"),
    ):
        raise ValueError("archived worker replay scope inventory drifted")


def _new_search_telemetry() -> dict[str, dict[str, float | int]]:
    return {
        role: {
            "search_calls": 0,
            "non_forced_search_calls": 0,
            "search_elapsed_sec": 0.0,
            "simulations_used": 0,
            "wide_root_calls": 0,
            "wide_root_simulations_used": 0,
            "selected_vs_prior_disagreement_calls": 0,
            "wide_selected_vs_prior_disagreement_calls": 0,
        }
        for role in ("candidate", "baseline")
    }


def _add_search_telemetry(
    target: dict[str, dict[str, float | int]],
    source: dict[str, dict[str, float | int]],
) -> None:
    for role in ("candidate", "baseline"):
        for key, value in source.get(role, {}).items():
            target[role][key] = target[role].get(key, 0) + value


def _finalize_search_telemetry(
    totals: dict[str, dict[str, float | int]],
) -> dict[str, Any]:
    by_role: dict[str, Any] = {}
    for role in ("candidate", "baseline"):
        raw = totals.get(role, {})
        calls = int(raw.get("search_calls", 0))
        non_forced_calls = int(raw.get("non_forced_search_calls", calls))
        elapsed = float(raw.get("search_elapsed_sec", 0.0))
        simulations = int(raw.get("simulations_used", 0))
        wide_calls = int(raw.get("wide_root_calls", 0))
        wide_simulations = int(raw.get("wide_root_simulations_used", 0))
        disagreements = int(raw.get("selected_vs_prior_disagreement_calls", 0))
        wide_disagreements = int(
            raw.get("wide_selected_vs_prior_disagreement_calls", 0)
        )
        by_role[role] = {
            "search_calls": calls,
            "non_forced_search_calls": non_forced_calls,
            "search_elapsed_sec": elapsed,
            "search_seconds_per_call": (elapsed / calls) if calls else None,
            "simulations_used": simulations,
            "simulations_per_call": (simulations / calls) if calls else None,
            "wide_root_calls": wide_calls,
            "wide_root_simulations_used": wide_simulations,
            "wide_root_simulations_per_call": (
                wide_simulations / wide_calls if wide_calls else None
            ),
            "selected_vs_prior_disagreement_calls": disagreements,
            "selected_vs_prior_disagreement_rate": (
                disagreements / non_forced_calls if non_forced_calls else None
            ),
            "wide_selected_vs_prior_disagreement_calls": wide_disagreements,
            "wide_selected_vs_prior_disagreement_rate": (
                wide_disagreements / wide_calls if wide_calls else None
            ),
        }

    candidate = by_role["candidate"]
    baseline = by_role["baseline"]
    baseline_elapsed = float(baseline["search_elapsed_sec"])
    baseline_simulations = int(baseline["simulations_used"])
    candidate_per_call = candidate["search_seconds_per_call"]
    baseline_per_call = baseline["search_seconds_per_call"]
    candidate_simulations_per_call = candidate["simulations_per_call"]
    baseline_simulations_per_call = baseline["simulations_per_call"]
    return {
        "by_role": by_role,
        "candidate_over_baseline_elapsed_ratio": (
            float(candidate["search_elapsed_sec"]) / baseline_elapsed
            if baseline_elapsed > 0.0
            else None
        ),
        "candidate_over_baseline_seconds_per_call_ratio": (
            float(candidate_per_call) / float(baseline_per_call)
            if candidate_per_call is not None
            and baseline_per_call is not None
            and float(baseline_per_call) > 0.0
            else None
        ),
        "candidate_over_baseline_simulations_ratio": (
            int(candidate["simulations_used"]) / baseline_simulations
            if baseline_simulations > 0
            else None
        ),
        "candidate_over_baseline_simulations_per_call_ratio": (
            float(candidate_simulations_per_call) / float(baseline_simulations_per_call)
            if candidate_simulations_per_call is not None
            and baseline_simulations_per_call is not None
            and float(baseline_simulations_per_call) > 0.0
            else None
        ),
        "search_cost_definition": (
            "simulations_used is the exact sum returned by SearchResult for each role; "
            "elapsed seconds additionally include root expansion, evaluator, D6, and "
            "Python/Rust orchestration overhead"
        ),
        "selected_action_disagreement_definition": (
            "selected_action != argmax(search root prior before MCTS improvement) on "
            "that role's own decision; "
            "rate denominator is non-forced decisions; not candidate-vs-baseline "
            "disagreement because the roles do not search the same states in H2H "
            "trajectories"
        ),
    }


def play_one_h2h_game(
    mcts_by_role: dict[str, GumbelChanceMCTS],
    *,
    role_by_color: dict[str, str],
    game_seed: int,
    max_decisions: int,
    correct_rust_chance_spectra: bool,
    map_kind: str = "BASE",
    search_telemetry_by_role: dict[str, dict[str, float | int]] | None = None,
    initial_game: Any | None = None,
    initial_chance_rng: Any | None = None,
    archived_game_seed: int | None = None,
    archived_decision_index: int | None = None,
    archived_phase: str | None = None,
    archived_legal_count: int | None = None,
) -> dict[str, Any]:
    import random

    if (initial_game is None) != (initial_chance_rng is None):
        raise ValueError(
            "initial_game and initial_chance_rng must be supplied together"
        )
    if initial_game is None:
        catanatron_rs = _require_rust_module()
        game = catanatron_rs.Game(
            colors=list(COLORS),
            seed=int(game_seed),
            player_kind="simple",
            map_kind=str(map_kind),
        )
        chance_rng = random.Random(int(game_seed) ^ 0xA17E)
    else:
        game = initial_game
        chance_rng = initial_chance_rng

    decision_index = 0
    terminal = False
    phases_seen: set[str] = set()
    max_legal_count = int(archived_legal_count or 0)
    while decision_index < int(max_decisions):
        if game.winning_color() is not None:
            terminal = True
            break
        legal_rust = tuple(
            int(action) for action in game.playable_action_indices(list(COLORS), None)
        )
        max_legal_count = max(max_legal_count, len(legal_rust))
        try:
            current_phase = str(
                json.loads(game.json_snapshot()).get("current_prompt", "")
            )
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
            current_phase = ""
        if current_phase:
            phases_seen.add(current_phase)
        if not legal_rust:
            break

        acting_color = str(game.current_color())
        role = role_by_color[acting_color]
        mcts = mcts_by_role[role]
        started = time.perf_counter()
        result = mcts.search(game, force_full=True)
        search_elapsed = time.perf_counter() - started
        selected = int(result.selected_action)

        if search_telemetry_by_role is not None:
            role_telemetry = search_telemetry_by_role[role]
            role_telemetry["search_calls"] += 1
            if len(legal_rust) > 1:
                role_telemetry["non_forced_search_calls"] += 1
            role_telemetry["search_elapsed_sec"] += search_elapsed
            role_telemetry["simulations_used"] += int(result.simulations_used)
            wide_root = _matches_explicit_or_legacy_width_gate(
                len(legal_rust),
                min_legal_actions=mcts.config.n_full_wide_threshold,
                legacy_exclusive_threshold=mcts.config.wide_candidates_threshold,
            )
            if wide_root:
                role_telemetry["wide_root_calls"] += 1
                role_telemetry["wide_root_simulations_used"] += int(
                    result.simulations_used
                )
            if result.priors:
                prior_argmax = max(
                    result.priors,
                    key=lambda action: (result.priors[action], -int(action)),
                )
                if selected != int(prior_argmax):
                    role_telemetry["selected_vs_prior_disagreement_calls"] += 1
                    if wide_root:
                        role_telemetry["wide_selected_vs_prior_disagreement_calls"] += 1

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

    candidate_color = next(
        color for color, role in role_by_color.items() if role == "candidate"
    )
    baseline_color = next(
        color for color, role in role_by_color.items() if role == "baseline"
    )
    candidate_won = (winner == candidate_color) if terminal else None
    start_phase = str(archived_phase or "")
    if start_phase:
        phases_seen.add(start_phase)
    phase_bucket = _promotion_phase_bucket(
        {start_phase} if start_phase else phases_seen
    )
    buckets = [f"phase:{phase_bucket}"]
    phase_upper = " ".join(phases_seen).upper()
    if "BUILD_INITIAL_SETTLEMENT" in phase_upper or "BUILD_INITIAL_ROAD" in phase_upper:
        buckets.append("opening")
    if max_legal_count >= 41:
        buckets.append("41+")
    vp_margin = abs(
        final_vps.get(candidate_color, 0) - final_vps.get(baseline_color, 0)
    )
    buckets.append("blowout" if vp_margin >= 3 else "close")

    return {
        "game_seed": int(game_seed),
        "map_kind": str(map_kind),
        "candidate_color": candidate_color,
        "baseline_color": baseline_color,
        "winner": winner,
        "terminated": bool(terminal),
        "truncated": bool(truncated),
        "decisions": int(decision_index),
        "final_vps": final_vps,
        "candidate_won": candidate_won,
        "buckets": sorted(set(buckets)),
        "max_legal_count": max_legal_count,
        "phases_seen": sorted(phases_seen),
        "archived_game_seed": archived_game_seed,
        "archived_decision_index": archived_decision_index,
        # Kept for reuse of sprt_gate.py's pair_scores_from_h2h_games /
        # _concordant_pair_outcomes, which key off "search_won" generically.
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


def _resolve_value_readouts(args: Any) -> tuple[str, str]:
    """Return the effective candidate/baseline value readouts.

    ``--value-readout`` remains the backwards-compatible shared fallback.  A
    role-specific value only overrides its own side, which is required for the
    first HL-Gauss promotion gate: categorical candidate vs scalar incumbent.
    This helper accepts either an argparse namespace or the worker-argument
    dict so config hashing, worker construction, and artifact reporting share
    one resolution rule.
    """

    def _get(name: str, default: Any = None) -> Any:
        if isinstance(args, dict):
            return args.get(name, default)
        return getattr(args, name, default)

    shared = str(_get("value_readout", "scalar"))
    candidate = _get("candidate_value_readout")
    baseline = _get("baseline_value_readout")
    resolved = (
        str(candidate) if candidate is not None else shared,
        str(baseline) if baseline is not None else shared,
    )
    allowed = {"scalar", "categorical"}
    if any(value not in allowed for value in resolved):
        raise ValueError(
            "value readout must resolve to 'scalar' or 'categorical'; "
            f"candidate={resolved[0]!r}, baseline={resolved[1]!r}"
        )
    return resolved


def _resolve_search_budgets(args: Any) -> dict[str, int | None]:
    """Resolve shared/role-specific normal and wide-root search budgets.

    The shared ``n_full`` / ``n_full_wide`` values remain the backwards-
    compatible fallback.  A role-specific value overrides only that side,
    which makes a fair adaptive-opening comparison possible: candidate
    ``n_full=128,n_full_wide=256`` versus baseline
    ``n_full=128,n_full_wide=None``.  Keeping this resolution in one helper
    prevents worker construction, typed-config hashing, and output provenance
    from silently disagreeing.
    """

    def _get(name: str, default: Any = None) -> Any:
        if isinstance(args, dict):
            return args.get(name, default)
        return getattr(args, name, default)

    shared_n_full = int(_get("n_full", 64))
    shared_n_full_wide_raw = _get("n_full_wide")
    shared_n_full_wide = (
        int(shared_n_full_wide_raw) if shared_n_full_wide_raw is not None else None
    )
    shared_n_full_wide_threshold_raw = _get("n_full_wide_threshold")
    shared_n_full_wide_threshold = (
        int(shared_n_full_wide_threshold_raw)
        if shared_n_full_wide_threshold_raw is not None
        else None
    )

    candidate_n_full_raw = _get("candidate_n_full")
    baseline_n_full_raw = _get("baseline_n_full")
    candidate_n_full_wide_raw = _get("candidate_n_full_wide")
    baseline_n_full_wide_raw = _get("baseline_n_full_wide")
    candidate_n_full_wide_threshold_raw = _get("candidate_n_full_wide_threshold")
    baseline_n_full_wide_threshold_raw = _get("baseline_n_full_wide_threshold")

    return {
        "candidate_n_full": (
            int(candidate_n_full_raw)
            if candidate_n_full_raw is not None
            else shared_n_full
        ),
        "baseline_n_full": (
            int(baseline_n_full_raw)
            if baseline_n_full_raw is not None
            else shared_n_full
        ),
        "candidate_n_full_wide": (
            int(candidate_n_full_wide_raw)
            if candidate_n_full_wide_raw is not None
            else shared_n_full_wide
        ),
        "baseline_n_full_wide": (
            int(baseline_n_full_wide_raw)
            if baseline_n_full_wide_raw is not None
            else shared_n_full_wide
        ),
        "candidate_n_full_wide_threshold": (
            int(candidate_n_full_wide_threshold_raw)
            if candidate_n_full_wide_threshold_raw is not None
            else shared_n_full_wide_threshold
        ),
        "baseline_n_full_wide_threshold": (
            int(baseline_n_full_wide_threshold_raw)
            if baseline_n_full_wide_threshold_raw is not None
            else shared_n_full_wide_threshold
        ),
    }


def _resolve_c_scales(args: Any) -> dict[str, float]:
    """Resolve effective candidate/baseline sigma scales.

    ``--c-scale`` remains the backwards-compatible shared fallback.  Explicit
    role values override only their own side.  This helper is used by worker
    construction, typed-config hashing, and report generation so all three
    surfaces describe the exact same search operators.
    """

    def _get(name: str, default: Any = None) -> Any:
        if isinstance(args, dict):
            return args.get(name, default)
        return getattr(args, name, default)

    shared = float(_get("c_scale", 0.1))
    candidate = _get("candidate_c_scale")
    baseline = _get("baseline_c_scale")
    return {
        "candidate_c_scale": (float(candidate) if candidate is not None else shared),
        "baseline_c_scale": float(baseline) if baseline is not None else shared,
    }


def _build_evaluator(
    checkpoint: str,
    worker_args: dict[str, Any],
    *,
    role: str | None = None,
) -> Any:
    candidate_readout, baseline_readout = _resolve_value_readouts(worker_args)
    if role is None:
        # Backwards compatibility for direct callers of this helper: when no
        # role is supplied, retain the historical shared-readout behavior.
        value_readout = str(worker_args.get("value_readout", "scalar"))
    elif role == "candidate":
        value_readout = candidate_readout
    elif role == "baseline":
        value_readout = baseline_readout
    else:
        raise ValueError(
            f"unknown evaluator role {role!r}; expected candidate|baseline"
        )
    return BatchedEntityGraphRustEvaluator.from_checkpoint(
        checkpoint,
        device=worker_args["device"],
        config=EntityGraphRustEvaluatorConfig(
            value_scale=float(worker_args["value_scale"]),
            prior_temperature=float(worker_args["prior_temperature"]),
            context_fill=float(worker_args.get("evaluator_context_fill", 0.0)),
            cache_size=int(worker_args.get("evaluator_cache_size", 0)),
            value_squash=str(worker_args.get("value_squash", "tanh")),
            value_readout=value_readout,
            public_observation=bool(worker_args.get("public_observation", False)),
            rust_featurize=bool(worker_args.get("evaluator_rust_featurize", False)),
            emit_uncertainty=bool(worker_args.get("evaluator_emit_uncertainty", False)),
        ),
    )


def _build_search_config(
    worker_args: dict[str, Any],
    *,
    seed: int,
    n_full: int | None = None,
    n_full_wide: int | None = None,
    n_full_wide_threshold: int | None = None,
    c_scale: float | None = None,
) -> GumbelChanceMCTSConfig:
    """`n_full` defaults to `worker_args["n_full"]` when not given explicitly.

    CAT-25 rollout-doubling (measurement 2) needs the SAME checkpoint played
    against itself at two different search budgets (e.g. n=64 vs n=128), so
    `_run_worker` calls this once per role with `worker_args.get("candidate_n_full",
    worker_args["n_full"])` / `worker_args.get("baseline_n_full", worker_args["n_full"])`
    -- when neither key is present (every existing caller), both roles fall
    back to the single shared `n_full`, so this is a byte-identical no-op for
    unchanged callers.
    """
    resolved_n_full = int(n_full) if n_full is not None else int(worker_args["n_full"])
    resolved_n_full_wide = (
        int(n_full_wide)
        if n_full_wide is not None
        else (
            int(worker_args["n_full_wide"])
            if worker_args.get("n_full_wide") is not None
            else None
        )
    )
    resolved_n_full_wide_threshold = (
        int(n_full_wide_threshold)
        if n_full_wide_threshold is not None
        else (
            int(worker_args["n_full_wide_threshold"])
            if worker_args.get("n_full_wide_threshold") is not None
            else None
        )
    )
    return GumbelChanceMCTSConfig(
        colors=COLORS,
        seed=int(seed),
        n_full=resolved_n_full,
        n_fast=resolved_n_full,  # unused: force_full=True always selects n_full.
        p_full=1.0,
        max_depth=int(worker_args["max_depth"]),
        temperature=0.0,  # deterministic argmax at the root.
        correct_rust_chance_spectra=bool(worker_args["correct_rust_chance_spectra"]),
        lazy_interior_chance=bool(worker_args.get("lazy_interior_chance", False)),
        belief_chance_spectra=bool(worker_args.get("belief_chance_spectra", False)),
        information_set_search=bool(worker_args.get("information_set_search", False)),
        determinization_particles=int(worker_args.get("determinization_particles", 1)),
        determinization_min_simulations=int(
            worker_args.get("determinization_min_simulations", 32)
        ),
        c_scale=(
            float(c_scale)
            if c_scale is not None
            else float(worker_args.get("c_scale", 0.1))
        ),
        c_visit=float(worker_args.get("c_visit", 50.0)),
        rescale_noise_floor_c=float(worker_args.get("rescale_noise_floor_c", 0.0)),
        sigma_eval=float(worker_args.get("sigma_eval", 0.79)),
        max_root_candidates=int(worker_args.get("max_root_candidates", 16)),
        max_root_candidates_wide=int(worker_args.get("max_root_candidates_wide", 54)),
        wide_candidates_threshold=int(worker_args.get("wide_candidates_threshold", 24)),
        n_full_wide=resolved_n_full_wide,
        n_full_wide_threshold=resolved_n_full_wide_threshold,
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
        wide_roots_always_full=bool(worker_args.get("wide_roots_always_full", False)),
        exact_budget_sh=bool(worker_args.get("exact_budget_sh", False)),
        exact_budget_sh_min_n=int(worker_args.get("exact_budget_sh_min_n", 0)),
        root_wave_batching=bool(worker_args.get("root_wave_batching", False)),
        play_sh_winner=bool(worker_args.get("play_sh_winner", False)),
        use_batch_api=bool(worker_args.get("use_batch_api", True)),
        policy_target_min_visits=int(worker_args.get("policy_target_min_visits", 0)),
        uncertainty_backup_weighting=bool(
            worker_args.get("uncertainty_backup_weighting", False)
        ),
        uncertainty_backup_a=float(worker_args.get("uncertainty_backup_a", 0.25)),
        uncertainty_backup_exp=float(worker_args.get("uncertainty_backup_exp", 1.0)),
        uncertainty_backup_cap=float(worker_args.get("uncertainty_backup_cap", 1.0)),
        variance_aware_q=bool(worker_args.get("variance_aware_q", False)),
        variance_aware_k=float(worker_args.get("variance_aware_k", 1.0)),
        variance_aware_closed_form_js=bool(
            worker_args.get("variance_aware_closed_form_js", False)
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


def _write_worker_progress(
    progress_dir: str, worker_index: int, games_done: int, wins: int
) -> None:
    """Atomically write this worker's running tally so a poller can sum all worker_*.json for a
    live win-rate read (fixes the old 'no progress until the very end' blindness)."""
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
        worker_args["candidate_checkpoint"], worker_args, role="candidate"
    )
    baseline_evaluator = _build_evaluator(
        worker_args["baseline_checkpoint"], worker_args, role="baseline"
    )

    worker_seed = int(worker_args["worker_seed"])
    budgets = _resolve_search_budgets(worker_args)
    c_scales = _resolve_c_scales(worker_args)
    candidate_n_full = int(budgets["candidate_n_full"])
    baseline_n_full = int(budgets["baseline_n_full"])
    candidate_mcts = _create_search(
        _build_search_config(
            worker_args,
            seed=worker_seed,
            n_full=candidate_n_full,
            n_full_wide=budgets["candidate_n_full_wide"],
            n_full_wide_threshold=budgets["candidate_n_full_wide_threshold"],
            c_scale=c_scales["candidate_c_scale"],
        ),
        candidate_evaluator,
        native_mcts_hot_loop=bool(worker_args.get("native_mcts_hot_loop", False)),
    )
    baseline_mcts = _create_search(
        _build_search_config(
            worker_args,
            seed=worker_seed,
            n_full=baseline_n_full,
            n_full_wide=budgets["baseline_n_full_wide"],
            n_full_wide_threshold=budgets["baseline_n_full_wide_threshold"],
            c_scale=c_scales["baseline_c_scale"],
        ),
        baseline_evaluator,
        native_mcts_hot_loop=bool(worker_args.get("native_mcts_hot_loop", False)),
    )
    mcts_by_role = {"candidate": candidate_mcts, "baseline": baseline_mcts}

    games: list[dict[str, Any]] = []
    pair_errors: list[dict[str, Any]] = []
    search_telemetry = _new_search_telemetry()
    archived_sequences: dict[tuple[str, int], Any] = {}
    pinned_replay_scopes: dict[Path, PinnedReplayScope] = {}
    try:
        for pair in worker_args["pairs"]:
            game_seed = int(pair["game_seed"])
            # Isolate failures per pair: one bad game must not discard the whole
            # worker's completed games. A half-finished pair is dropped entirely
            # (the pentanomial SPRT requires both orientations anyway).
            pair_games: list[dict[str, Any]] = []
            try:
                for orientation, role_by_color in (
                    ("candidate_red", {"RED": "candidate", "BLUE": "baseline"}),
                    ("candidate_blue", {"RED": "baseline", "BLUE": "candidate"}),
                ):
                    archived = pair.get("archived_state")
                    initial: dict[str, Any] = {}
                    if archived is not None:
                        from reconstruct_state import (
                            action_size_for_colors,
                            gather_game_action_sequence,
                            reconstruct_state,
                        )

                        shard_path = str(archived["shard_path"])
                        original_scope = Path(shard_path).parent
                        pinned_scope = pinned_replay_scopes.get(original_scope)
                        if pinned_scope is None:
                            replay_source = archived["replay_source"]
                            pinned_scope = pin_replay_scope(
                                original_scope,
                                expected_sha256=replay_source["scope_inventory_sha256"],
                                expected_count=replay_source["scope_shard_count"],
                            )
                            pinned_replay_scopes[original_scope] = pinned_scope
                        cache_key = (shard_path, game_seed)
                        if cache_key not in archived_sequences:
                            archived_sequences[cache_key] = gather_game_action_sequence(
                                pinned_scope.snapshot_scope,
                                game_seed,
                                colors=COLORS,
                            )
                        sequence = archived_sequences[cache_key]
                        game, chance_rng = reconstruct_state(
                            game_seed,
                            sequence.actions,
                            int(archived["decision_index"]),
                            colors=COLORS,
                            correct_rust_chance_spectra=bool(
                                worker_args["correct_rust_chance_spectra"]
                            ),
                            action_size=action_size_for_colors(COLORS),
                            return_rng=True,
                        )
                        initial = {
                            "initial_game": game,
                            "initial_chance_rng": chance_rng,
                            "archived_game_seed": game_seed,
                            "archived_decision_index": int(archived["decision_index"]),
                            "archived_phase": str(archived.get("phase", "")),
                            "archived_legal_count": int(archived.get("legal_count", 0)),
                        }
                    record = play_one_h2h_game(
                        mcts_by_role,
                        role_by_color=role_by_color,
                        game_seed=game_seed,
                        max_decisions=int(worker_args["max_decisions"]),
                        correct_rust_chance_spectra=bool(
                            worker_args["correct_rust_chance_spectra"]
                        ),
                        map_kind=str(worker_args["map_kind"]),
                        search_telemetry_by_role=search_telemetry,
                        **initial,
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
            # incremental progress after each pair (2 games), so a poller sees a live win rate
            _wins = sum(1 for g in games if g.get("candidate_won"))
            _write_worker_progress(
                worker_args.get("progress_dir", ""),
                int(worker_args["worker_index"]),
                len(games),
                _wins,
            )
    finally:
        for pinned_scope in pinned_replay_scopes.values():
            pinned_scope.close()
        candidate_evaluator.close()
        baseline_evaluator.close()

    return {
        "worker_index": int(worker_args["worker_index"]),
        "games": games,
        "error": None,
        "pair_errors": pair_errors,
        "search_telemetry": search_telemetry,
    }


def _validate_information_set_recipe(args: Any) -> None:
    """Reject masked search that can still expand authoritative hidden truth."""
    public = bool(args.public_observation)
    information_set = bool(args.information_set_search)
    if public and not information_set:
        raise ValueError(
            "--public-observation requires --information-set-search; masking NN "
            "features alone does not make the MCTS tree public-information safe"
        )
    if information_set and not public:
        raise ValueError("--information-set-search requires --public-observation")
    if information_set and bool(args.belief_chance_spectra):
        raise ValueError(
            "--information-set-search cannot be combined with --belief-chance-spectra"
        )
    if int(args.determinization_particles) < 1:
        raise ValueError("--determinization-particles must be >= 1")
    if int(args.determinization_min_simulations) < 1:
        raise ValueError("--determinization-min-simulations must be >= 1")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-checkpoint H2H gate: candidate checkpoint vs baseline checkpoint, "
        "both using GumbelChanceMCTS search with identical config by default; "
        "role-specific budget/c_scale flags are explicit search-operator overrides."
    )
    parser.add_argument("--candidate", required=True, help="Candidate checkpoint path.")
    parser.add_argument("--baseline", required=True, help="Baseline checkpoint path.")
    parser.add_argument(
        "--pairs", type=int, default=50, help="paired seeds; total games = 2x this"
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--devices",
        default=None,
        help="comma-list of devices to spread workers across, e.g. "
        "cuda:0,cuda:1 (round-robin per worker; overrides --device). "
        "Halves wall-time on a 2-GPU box.",
    )
    parser.add_argument("--n-full", type=int, default=64)
    parser.add_argument(
        "--candidate-n-full",
        type=int,
        default=None,
        help=(
            "CAT-25 rollout-doubling arm: search budget for the candidate role only. "
            "Default None = fall back to --n-full (byte-identical to every prior caller)."
        ),
    )
    parser.add_argument(
        "--baseline-n-full",
        type=int,
        default=None,
        help=(
            "CAT-25 rollout-doubling arm: search budget for the baseline role only. "
            "Default None = fall back to --n-full (byte-identical to every prior caller)."
        ),
    )
    parser.add_argument("--max-depth", type=int, default=80)
    parser.add_argument("--max-decisions", type=int, default=300)
    parser.add_argument("--prior-temperature", type=float, default=1.0)
    parser.add_argument("--value-scale", type=float, default=1.0)
    parser.add_argument(
        "--value-readout",
        choices=("scalar", "categorical"),
        default="scalar",
        help="Backwards-compatible shared value source for both nets. A role-specific "
        "flag overrides this fallback for only that side.",
    )
    parser.add_argument(
        "--candidate-value-readout",
        choices=("scalar", "categorical"),
        default=None,
        help="Candidate-only value source (default: inherit --value-readout).",
    )
    parser.add_argument(
        "--baseline-value-readout",
        choices=("scalar", "categorical"),
        default=None,
        help="Baseline-only value source (default: inherit --value-readout).",
    )
    parser.add_argument(
        "--correct-rust-chance-spectra",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--lazy-interior-chance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run search with lazy interior chance evaluation (#52 lazy-vs-raw arm).",
    )
    parser.add_argument(
        "--value-squash",
        choices=("tanh", "clip"),
        default="tanh",
        help="Evaluator value squash (#60 diagnostic arm).",
    )
    parser.add_argument(
        "--c-visit",
        type=float,
        default=50.0,
        help="Sigma c_visit floor; 1.0 = visit-scaled sigma (armV diagnostic).",
    )
    parser.add_argument(
        "--c-scale",
        type=float,
        default=0.1,
        help="Sigma scale multiplier (matches GumbelChanceMCTSConfig default).",
    )
    parser.add_argument(
        "--candidate-c-scale",
        type=float,
        default=None,
        help="Candidate-only sigma scale (default: inherit --c-scale).",
    )
    parser.add_argument(
        "--baseline-c-scale",
        type=float,
        default=None,
        help="Baseline-only sigma scale (default: inherit --c-scale).",
    )
    parser.add_argument(
        "--rescale-noise-floor-c",
        type=float,
        default=0.0,
        help="D1 noise-floor rescaling coefficient. Default 0.0 is the exact legacy no-op.",
    )
    parser.add_argument(
        "--sigma-eval",
        type=float,
        default=0.79,
        help="Value-estimate noise stdev used by --rescale-noise-floor-c.",
    )
    parser.add_argument(
        "--max-root-candidates",
        type=int,
        default=16,
        help="Root Gumbel-Top-k candidate cap on normal roots (SNR arm: 8).",
    )
    parser.add_argument(
        "--max-root-candidates-wide",
        type=int,
        default=54,
        help="Root Gumbel-Top-k cap on wide (placement) roots; 16 = narrow diagnostic arm.",
    )
    parser.add_argument(
        "--wide-candidates-threshold",
        type=int,
        default=24,
        help="Legacy exclusive threshold for the wide candidate cap and fallback "
        "for D6/adaptive-budget gates when explicit thresholds are unset.",
    )
    parser.add_argument(
        "--public-observation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Public-observation featurization (hidden-info leak fix, f72): mask each "
        "opponent's hand composition, unplayed dev-card identities, and actual VP from "
        "the model input for BOTH nets (symmetric). Threads to "
        "EntityGraphRustEvaluatorConfig.public_observation. Use with checkpoints trained "
        "via train_bc --mask-hidden-info for a valid public-only H2H.",
    )
    parser.add_argument(
        "--belief-chance-spectra",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Planner-only public-belief chance spectra (hidden-info leak fix, f72) for "
        "both sides' search. Threads to GumbelChanceMCTSConfig.belief_chance_spectra.",
    )
    parser.add_argument(
        "--information-set-search",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Actor-turn search over public-belief determinizations. REQUIRED with "
            "--public-observation; masked NN inputs alone leave authoritative hidden "
            "truth available to tree expansion."
        ),
    )
    parser.add_argument("--determinization-particles", type=int, default=1)
    parser.add_argument("--determinization-min-simulations", type=int, default=32)
    parser.add_argument(
        "--n-full-wide",
        type=int,
        default=None,
        help="Backwards-compatible shared wide-root simulation budget for both "
        "roles. Role-specific flags override this fallback for only that side. "
        "Default None = use each role's normal n_full at wide roots.",
    )
    parser.add_argument(
        "--n-full-wide-threshold",
        type=int,
        default=None,
        help="Shared inclusive minimum legal-action count for n_full_wide. "
        "Default None preserves the legacy > --wide-candidates-threshold gate.",
    )
    parser.add_argument(
        "--candidate-n-full-wide",
        type=int,
        default=None,
        help=(
            "Candidate-only wide-root simulation budget. Default None = inherit "
            "--n-full-wide (which itself defaults to disabled)."
        ),
    )
    parser.add_argument(
        "--baseline-n-full-wide",
        type=int,
        default=None,
        help=(
            "Baseline-only wide-root simulation budget. Default None = inherit "
            "--n-full-wide (which itself defaults to disabled)."
        ),
    )
    parser.add_argument(
        "--candidate-n-full-wide-threshold",
        type=int,
        default=None,
        help="Candidate-only inclusive n_full_wide width gate (default: inherit "
        "--n-full-wide-threshold).",
    )
    parser.add_argument(
        "--baseline-n-full-wide-threshold",
        type=int,
        default=None,
        help="Baseline-only inclusive n_full_wide width gate (default: inherit "
        "--n-full-wide-threshold).",
    )
    parser.add_argument(
        "--raw-policy-above-width",
        type=int,
        default=None,
        help="Phase-gated-search arm: at roots wider than this many legal actions, "
        "skip search and play argmax(prior). Default None = always search (disabled).",
    )
    parser.add_argument(
        "--symmetry-averaged-eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "f74b: denoise wide-root leaf value+prior by averaging the evaluator over all "
            "12 D6 board orientations (gated by --symmetry-averaged-eval-threshold, "
            "or the legacy shared threshold when unset). "
            "Threads to GumbelChanceMCTSConfig.symmetry_averaged_eval."
        ),
    )
    parser.add_argument(
        "--symmetry-averaged-eval-threshold",
        type=int,
        default=None,
        help="Shared inclusive minimum legal-action count for D6 averaging. "
        "Default None preserves the legacy > --wide-candidates-threshold gate.",
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
        default=False,
        help=(
            "Build entity and legal-action context tensors with the bit-exact "
            "native featurizer. Opt-in and fail-closed; no Python fallback."
        ),
    )
    parser.add_argument("--base-seed", type=int, default=1)
    parser.add_argument(
        "--map-kind",
        choices=("BASE", "TOURNAMENT"),
        default="BASE",
        help=(
            "Board distribution for fresh-start games. BASE is the broad randomized "
            "direct-H2H stratum; TOURNAMENT is the fixed-map bridge to the neutral "
            "Python-referee stratum."
        ),
    )
    parser.add_argument(
        "--held-out-high-regret-suite",
        default=None,
        help=(
            "Evaluate every archived state in an immutable "
            "a1-held-out-high-regret-suite-v4 manifest instead of fresh starts; "
            "emits a1-held-out-high-regret-report-v1 for promotion replay."
        ),
    )
    parser.add_argument("--engine-repo-commit", default=None)
    parser.add_argument("--native-wheel-path", default=None)
    parser.add_argument("--native-wheel-sha256", default=None)
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
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        default=0,
        help="torch intra-op thread cap per worker process (0 = auto: "
        "floor(os.cpu_count() / workers), so --workers N never oversubscribes "
        "the host). Set explicitly to share a box with other tenants.",
    )
    parser.add_argument("--out", required=True)
    add_config_flags(parser, default_purpose="gumbel_search_cross_net_h2h")
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
    try:
        _validate_information_set_recipe(args)
    except ValueError as exc:
        parser.error(str(exc))
    _gate_cfg, _gate_params = resolve_gate_config(
        args.gate_config, elo0=args.elo0, elo1=args.elo1
    )
    args.elo0, args.elo1 = _gate_params["elo0"], _gate_params["elo1"]

    # CAT-66 typed config + config-hash (cross-checkpoint search-vs-search regime).
    def _build_eval_config(resolved_args: Any) -> EvalConfig:
        candidate_readout, baseline_readout = _resolve_value_readouts(resolved_args)
        budgets = _resolve_search_budgets(resolved_args)
        c_scales = _resolve_c_scales(resolved_args)
        return EvalConfig.from_namespace(
            resolved_args,
            mode="cross_net",
            map_kind=str(resolved_args.map_kind),
            n_fast=budgets["candidate_n_full"],
            p_full=1.0,
            force_full_every_decision=True,
            candidate_value_readout=candidate_readout,
            baseline_value_readout=baseline_readout,
            candidate_n_full=budgets["candidate_n_full"],
            baseline_n_full=budgets["baseline_n_full"],
            candidate_n_full_wide=budgets["candidate_n_full_wide"],
            baseline_n_full_wide=budgets["baseline_n_full_wide"],
            candidate_n_full_wide_threshold=budgets["candidate_n_full_wide_threshold"],
            baseline_n_full_wide_threshold=budgets["baseline_n_full_wide_threshold"],
            candidate_c_scale=c_scales["candidate_c_scale"],
            baseline_c_scale=c_scales["baseline_c_scale"],
        )

    eval_config = resolve_config(
        args,
        _build_eval_config,
        parser=parser,
    )
    eval_config_hash = eval_config.config_hash()
    eval_full_config_hash = eval_config.full_config_hash()
    candidate_checkpoint_sha256 = _checkpoint_sha256(args.candidate)
    baseline_checkpoint_sha256 = _checkpoint_sha256(args.baseline)
    candidate_value_readout, baseline_value_readout = _resolve_value_readouts(args)
    c_scales = _resolve_c_scales(args)

    high_regret_suite_path: Path | None = None
    high_regret_planned_engine: dict[str, str] | None = None
    high_regret_engine: dict[str, str] | None = None
    if args.held_out_high_regret_suite:
        try:
            high_regret_suite_path, _high_regret_suite, pairs = (
                _load_held_out_high_regret_suite(args.held_out_high_regret_suite)
            )
            high_regret_planned_engine, high_regret_engine = (
                _held_out_engine_identity(args)
            )
        except ValueError as error:
            parser.error(str(error))
    else:
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

    # Multi-GPU: spread workers round-robin across --devices (falls back to --device).
    devices = (
        [d.strip() for d in args.devices.split(",")] if args.devices else [args.device]
    )
    # Live progress: workers write per-worker tallies here so a poller can peek the running win rate
    # without waiting for the whole run (the old single-write-at-end blindness).
    from pathlib import Path as _Path

    progress_dir = _Path(args.out).parent / "progress"
    progress_dir.mkdir(parents=True, exist_ok=True)

    worker_args = []
    for worker_index, pair_shard in enumerate(shards):
        if not pair_shard:
            continue
        args_dict = {
            "worker_index": worker_index,
            "pairs": pair_shard,
            "candidate_checkpoint": args.candidate,
            "baseline_checkpoint": args.baseline,
            "device": devices[worker_index % len(devices)],
            "progress_dir": str(progress_dir),
            "n_full": int(args.n_full),
            "max_depth": int(args.max_depth),
            "max_decisions": int(args.max_decisions),
            "map_kind": str(args.map_kind),
            "prior_temperature": float(args.prior_temperature),
            "value_scale": float(args.value_scale),
            "value_readout": str(args.value_readout),
            "candidate_value_readout": candidate_value_readout,
            "baseline_value_readout": baseline_value_readout,
            "correct_rust_chance_spectra": bool(args.correct_rust_chance_spectra),
            "lazy_interior_chance": bool(args.lazy_interior_chance),
            "public_observation": bool(args.public_observation),
            "belief_chance_spectra": bool(args.belief_chance_spectra),
            "information_set_search": bool(args.information_set_search),
            "native_mcts_hot_loop": bool(args.native_mcts_hot_loop),
            "evaluator_rust_featurize": bool(args.evaluator_rust_featurize),
            "determinization_particles": int(args.determinization_particles),
            "determinization_min_simulations": int(
                args.determinization_min_simulations
            ),
            "value_squash": str(args.value_squash),
            "c_scale": float(args.c_scale),
            "candidate_c_scale": c_scales["candidate_c_scale"],
            "baseline_c_scale": c_scales["baseline_c_scale"],
            "c_visit": float(args.c_visit),
            "rescale_noise_floor_c": float(args.rescale_noise_floor_c),
            "sigma_eval": float(args.sigma_eval),
            "max_root_candidates": int(args.max_root_candidates),
            "max_root_candidates_wide": int(args.max_root_candidates_wide),
            "wide_candidates_threshold": int(args.wide_candidates_threshold),
            "n_full_wide": (
                int(args.n_full_wide) if args.n_full_wide is not None else None
            ),
            "n_full_wide_threshold": (
                int(args.n_full_wide_threshold)
                if args.n_full_wide_threshold is not None
                else None
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
            "worker_seed": int(args.base_seed) + 0x9E3779B9 * (worker_index + 1),
        }
        # Only set candidate_n_full/baseline_n_full when the corresponding CLI
        # flag was actually given -- omitting the key otherwise means
        # _build_search_config's worker_args.get(..., worker_args["n_full"])
        # fallback kicks in, keeping every existing caller byte-identical.
        if args.candidate_n_full is not None:
            args_dict["candidate_n_full"] = int(args.candidate_n_full)
        if args.baseline_n_full is not None:
            args_dict["baseline_n_full"] = int(args.baseline_n_full)
        if args.candidate_n_full_wide is not None:
            args_dict["candidate_n_full_wide"] = int(args.candidate_n_full_wide)
        if args.baseline_n_full_wide is not None:
            args_dict["baseline_n_full_wide"] = int(args.baseline_n_full_wide)
        if args.candidate_n_full_wide_threshold is not None:
            args_dict["candidate_n_full_wide_threshold"] = int(
                args.candidate_n_full_wide_threshold
            )
        if args.baseline_n_full_wide_threshold is not None:
            args_dict["baseline_n_full_wide_threshold"] = int(
                args.baseline_n_full_wide_threshold
            )
        worker_args.append(args_dict)

    started = time.perf_counter()
    if len(worker_args) <= 1:
        results = [_worker_entry(worker_args[0])] if worker_args else []
    else:
        ctx = multiprocessing.get_context("spawn")
        results = []
        with ctx.Pool(processes=len(worker_args)) as pool:
            # imap_unordered streams results as each worker finishes, so we can log incremental
            # completion (and the per-worker progress files give an even finer live read).
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
                print(
                    json.dumps(
                        {
                            "progress": "worker_done",
                            "workers_done": done,
                            "workers_total": len(worker_args),
                            "games_so_far": _g,
                            "candidate_wins_so_far": _w,
                            "running_winrate": round(_w / _g, 4) if _g else None,
                        }
                    ),
                    flush=True,
                )
    elapsed = time.perf_counter() - started

    all_games: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    search_telemetry = _new_search_telemetry()
    for result in results:
        all_games.extend(result.get("games", ()))
        _add_search_telemetry(search_telemetry, result.get("search_telemetry", {}))
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
        candidate_checkpoint_sha256=candidate_checkpoint_sha256,
        baseline_checkpoint_sha256=baseline_checkpoint_sha256,
        search_telemetry=search_telemetry,
    )
    summary["config_hash"] = eval_config_hash
    summary["full_config_hash"] = eval_full_config_hash
    summary["typed_config"] = eval_config.canonical_payload()
    if high_regret_suite_path is not None:
        summary = {
            "schema_version": "a1-held-out-high-regret-report-v1",
            "suite": "held_out_high_regret",
            "held_out": True,
            "suite_manifest": {
                "path": str(high_regret_suite_path),
                "sha256": _checkpoint_sha256(high_regret_suite_path),
            },
            "candidate": {
                "path": str(Path(args.candidate).resolve()),
                "sha256": _checkpoint_sha256(args.candidate),
            },
            "champion": {
                "path": str(Path(args.baseline).resolve()),
                "sha256": _checkpoint_sha256(args.baseline),
            },
            "evaluation_config": eval_config.canonical_payload()["fields"],
            "errors": summary["errors"],
            "games": summary["games"],
            "pentanomial_sprt": summary["pentanomial_sprt"],
            "pair_diagnostics": summary["pair_diagnostics"],
            "planned_engine_identity": high_regret_planned_engine,
            "engine_identity": high_regret_engine,
            "archived_state_reconstruction": _archived_state_reconstruction_binding(),
        }
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
    pairs: list[Any],
    elapsed: float,
    workers: int,
    threads_per_worker: int,
    errors: list[Any],
    candidate_checkpoint_sha256: str,
    baseline_checkpoint_sha256: str,
    search_telemetry: dict[str, dict[str, float | int]] | None = None,
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

    # Resolved (not raw-flag) provenance: mirrors the fallback _build_search_config
    # itself applies, so a report always states the ACTUAL n_full each role searched
    # with, even when --candidate-n-full/--baseline-n-full were left at their
    # default (None) and --n-full was used for both roles.
    budgets = _resolve_search_budgets(args)
    resolved_candidate_n_full = int(budgets["candidate_n_full"])
    resolved_baseline_n_full = int(budgets["baseline_n_full"])
    candidate_value_readout, baseline_value_readout = _resolve_value_readouts(args)
    c_scales = _resolve_c_scales(args)

    return {
        "candidate_checkpoint": args.candidate,
        "candidate_checkpoint_sha256": candidate_checkpoint_sha256,
        "baseline_checkpoint": args.baseline,
        "baseline_checkpoint_sha256": baseline_checkpoint_sha256,
        "map_kind": str(getattr(args, "map_kind", "BASE")),
        "gate_config": getattr(args, "gate_config", None),
        "n_full": int(args.n_full),
        "candidate_n_full": resolved_candidate_n_full,
        "baseline_n_full": resolved_baseline_n_full,
        "lazy_interior_chance": bool(args.lazy_interior_chance),
        "value_squash": str(args.value_squash),
        "value_readout": str(args.value_readout),
        "candidate_value_readout": candidate_value_readout,
        "baseline_value_readout": baseline_value_readout,
        "c_scale": float(args.c_scale),
        "candidate_c_scale": c_scales["candidate_c_scale"],
        "baseline_c_scale": c_scales["baseline_c_scale"],
        "search_parameters_by_role": {
            "candidate": {
                "c_scale": c_scales["candidate_c_scale"],
                "c_visit": float(args.c_visit),
            },
            "baseline": {
                "c_scale": c_scales["baseline_c_scale"],
                "c_visit": float(args.c_visit),
            },
        },
        "comparison_contract": (
            "paired_same_seed_color_swap_role_specific_search_operators"
            if c_scales["candidate_c_scale"] != c_scales["baseline_c_scale"]
            else "paired_same_seed_color_swap_shared_search_operator"
        ),
        "c_visit": float(args.c_visit),
        "rescale_noise_floor_c": float(getattr(args, "rescale_noise_floor_c", 0.0)),
        "sigma_eval": float(getattr(args, "sigma_eval", 0.79)),
        "max_root_candidates": int(args.max_root_candidates),
        "max_root_candidates_wide": int(args.max_root_candidates_wide),
        "wide_candidates_threshold": int(
            getattr(args, "wide_candidates_threshold", 24)
        ),
        "correct_rust_chance_spectra": bool(args.correct_rust_chance_spectra),
        "public_observation": bool(args.public_observation),
        "belief_chance_spectra": bool(args.belief_chance_spectra),
        "information_set_search": bool(args.information_set_search),
        "native_mcts_hot_loop": bool(getattr(args, "native_mcts_hot_loop", False)),
        "mcts_implementation": (
            "rust_native_hot_loop_v1"
            if bool(getattr(args, "native_mcts_hot_loop", False))
            else "python_reference"
        ),
        "determinization_particles": int(args.determinization_particles),
        "determinization_min_simulations": int(args.determinization_min_simulations),
        "n_full_wide": (
            int(args.n_full_wide) if args.n_full_wide is not None else None
        ),
        "n_full_wide_threshold": (
            int(args.n_full_wide_threshold)
            if getattr(args, "n_full_wide_threshold", None) is not None
            else None
        ),
        "candidate_n_full_wide": budgets["candidate_n_full_wide"],
        "baseline_n_full_wide": budgets["baseline_n_full_wide"],
        "candidate_n_full_wide_threshold": budgets["candidate_n_full_wide_threshold"],
        "baseline_n_full_wide_threshold": budgets["baseline_n_full_wide_threshold"],
        "search_budgets_by_role": {
            "candidate": {
                "n_full": resolved_candidate_n_full,
                "n_full_wide": budgets["candidate_n_full_wide"],
                "n_full_wide_threshold": budgets["candidate_n_full_wide_threshold"],
            },
            "baseline": {
                "n_full": resolved_baseline_n_full,
                "n_full_wide": budgets["baseline_n_full_wide"],
                "n_full_wide_threshold": budgets["baseline_n_full_wide_threshold"],
            },
        },
        "raw_policy_above_width": (
            int(args.raw_policy_above_width)
            if args.raw_policy_above_width is not None
            else None
        ),
        "symmetry_averaged_eval": bool(args.symmetry_averaged_eval),
        "symmetry_averaged_eval_threshold": (
            int(args.symmetry_averaged_eval_threshold)
            if getattr(args, "symmetry_averaged_eval_threshold", None) is not None
            else None
        ),
        "pairs_requested": len(pairs),
        "base_seed": int(getattr(args, "base_seed", 1)),
        "games_played": len(all_games),
        "games_with_winner": len(outcomes),
        "games_truncated": truncated_count,
        "candidate_wins": sum(1 for outcome in outcomes if outcome),
        "baseline_wins": sum(1 for outcome in outcomes if not outcome),
        "candidate_win_rate": (
            sum(1 for outcome in outcomes if outcome) / len(outcomes)
        )
        if outcomes
        else None,
        "sprt": sprt,
        "pair_sprt": pair_sprt,
        "pentanomial_sprt": pentanomial_sprt,
        # Recommended gate verdict: trinomial GSPRT over all complete pairs.
        "verdict": pentanomial_sprt["decision"],
        "pair_diagnostics": pair_diagnostics,
        "pairs_decisive": pair_diagnostics["ww_pairs"] + pair_diagnostics["ll_pairs"],
        "pairs_split_excluded": pair_diagnostics["split_pairs"],
        "pairs_truncated_excluded": pair_diagnostics["incomplete_pairs"],
        "complete_pairs": complete_pairs,
        "split_rate": split_rate,
        "decisive_pair_yield": decisive_pair_yield,
        "elapsed_sec": elapsed,
        "workers": workers,
        "threads_per_worker": threads_per_worker,
        "search_telemetry": _finalize_search_telemetry(
            search_telemetry or _new_search_telemetry()
        ),
        "errors": errors,
        "games": all_games,
    }


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
