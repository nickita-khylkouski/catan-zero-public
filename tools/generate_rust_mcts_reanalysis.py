#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from catan_zero.search import RustMCTS, RustMCTSConfig
from catan_zero.search.neural_rust_mcts import (
    EntityGraphRustEvaluator,
    EntityGraphRustEvaluatorConfig,
    _policy_history_options,
    rust_action_context_batch,
    rust_game_to_entity_batch,
    rust_policy_action_ids,
)
from catan_zero.rl.entity_feature_adapter import (
    policy_entity_feature_adapter_version,
)
from catan_zero.rl.gumbel_self_play import (
    TARGET_INFORMATION_REGIME_AUTHORITATIVE,
)

# Make the sibling ``tools/`` modules importable whether this module is run as a script or
# imported as a package submodule (``from tools.generate_rust_mcts_reanalysis import ...``,
# e.g. from tests) -- mirrors the same bootstrap already used by tools/train_bc.py etc.
_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from convert_teacher_to_entity_tokens import EntityShardWriter  # noqa: E402


COLORS = ("BLUE", "RED")
PLAYER_NAMES = ("BLUE", "RED", "ORANGE", "WHITE")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate entity-token BC shards from Rust MCTS reanalysis targets."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--opponent", default="value_function")
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--simulations", type=int, default=16)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--search-temperature", type=float, default=1.0)
    parser.add_argument("--max-depth", type=int, default=80)
    parser.add_argument("--prior-temperature", type=float, default=1.0)
    parser.add_argument("--value-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-decisions", type=int, default=1000)
    parser.add_argument("--min-legal-actions", type=int, default=2)
    parser.add_argument("--record-after-decisions", type=int, default=0)
    parser.add_argument(
        "--phase-include",
        default="",
        help="Comma-separated current_prompt substrings to collect; empty collects all phases.",
    )
    parser.add_argument(
        "--phase-exclude",
        default="",
        help="Comma-separated current_prompt substrings to skip before running MCTS.",
    )
    parser.add_argument("--shard-size", type=int, default=2048)
    parser.add_argument("--format", choices=("npz", "npz_zst"), default="npz")
    parser.add_argument(
        "--obs-width",
        type=int,
        default=806,
        help="Dummy flat observation width for train_bc schema checks; entity_graph uses entity tensors.",
    )
    args = parser.parse_args()

    try:
        import catanatron_rs  # type: ignore
    except ImportError as error:
        raise SystemExit("catanatron_rs is not installed in this Python environment") from error

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.glob("entity_teacher_shard_*.npz*")):
        raise SystemExit(f"{output} already contains entity shards; use a fresh output directory")

    evaluator = EntityGraphRustEvaluator.from_checkpoint(
        args.checkpoint,
        device=args.device,
        config=EntityGraphRustEvaluatorConfig(
            value_scale=float(args.value_scale),
            prior_temperature=float(args.prior_temperature),
        ),
    )
    feature_contract = _evaluator_feature_contract(evaluator)
    search = RustMCTS(
        RustMCTSConfig(
            colors=COLORS,
            simulations=max(1, int(args.simulations)),
            c_puct=float(args.c_puct),
            max_depth=int(args.max_depth),
            seed=int(args.seed),
            temperature=float(args.search_temperature),
        ),
        evaluator=evaluator,
    )
    writer = EntityShardWriter(output, shard_size=int(args.shard_size), fmt=args.format)
    rng = random.Random(int(args.seed) ^ 0xA17E)
    started = time.perf_counter()
    rows = 0
    games_completed = 0
    wins = 0
    skipped_forced = 0
    skipped_window = 0
    skipped_phase = 0
    terminal_rows = 0
    truncated_rows = 0
    phase_include = _csv_patterns(args.phase_include)
    phase_exclude = _csv_patterns(args.phase_exclude)
    try:
        for game_index in range(int(args.games)):
            if rows >= int(args.samples):
                break
            candidate_color = COLORS[game_index % 2]
            player_kinds = [
                "random" if color == candidate_color else str(args.opponent)
                for color in COLORS
            ]
            game = catanatron_rs.Game(
                colors=list(COLORS),
                seed=int(args.seed) + game_index,
                player_kinds=player_kinds,
                vps_to_win=10,
            )
            decisions = 0
            pending: list[tuple[dict[str, Any], dict[str, np.ndarray]]] = []
            while (
                game.winning_color() is None
                and decisions < int(args.max_decisions)
            ):
                current = str(game.current_color())
                if current == candidate_color:
                    legal_rust = tuple(
                        int(action)
                        for action in game.playable_action_indices(list(COLORS), None)
                    )
                    if not legal_rust:
                        break
                    phase = str(json.loads(game.json_snapshot()).get("current_prompt", ""))
                    if rows >= int(args.samples):
                        action = _raw_policy_action(evaluator, game, legal_rust)
                    elif decisions < int(args.record_after_decisions):
                        skipped_window += 1
                        action = _raw_policy_action(evaluator, game, legal_rust)
                    elif len(legal_rust) < int(args.min_legal_actions):
                        skipped_forced += 1
                        action = _raw_policy_action(evaluator, game, legal_rust)
                    elif not _phase_selected(
                        phase,
                        include=phase_include,
                        exclude=phase_exclude,
                    ):
                        skipped_phase += 1
                        action = _raw_policy_action(evaluator, game, legal_rust)
                    else:
                        row = _mcts_row(
                            game,
                            search=search,
                            evaluator=evaluator,
                            legal_rust=legal_rust,
                            candidate_color=candidate_color,
                            game_seed=int(args.seed) + game_index,
                            decision_index=decisions,
                            obs_width=int(args.obs_width),
                        )
                        features = row.pop("_features")
                        action = int(row.pop("_rust_action"))
                        pending.append((row, features))
                        rows += 1
                        if rows % 25 == 0:
                            print(
                                json.dumps(
                                    {
                                        "progress": "rust_mcts_reanalysis",
                                        "rows": rows,
                                        "game": game_index,
                                        "elapsed_sec": time.perf_counter() - started,
                                    },
                                    sort_keys=True,
                                ),
                                flush=True,
                            )
                    game = _apply_action_with_sampled_chance(
                        game,
                        int(action),
                        colors=COLORS,
                        rng=rng,
                    )
                else:
                    game.play_tick()
                decisions += 1
            winner = game.winning_color()
            terminal = winner is not None
            wins += int(terminal and str(winner) == candidate_color)
            outcome = _game_outcome_fields(game, terminal=terminal)
            for row, features in pending:
                row.update(outcome)
                writer.add(row, features)
            if terminal:
                terminal_rows += len(pending)
            else:
                truncated_rows += len(pending)
            games_completed += 1
    finally:
        writer.close()

    summary = {
        "checkpoint": args.checkpoint,
        "out": str(output),
        "opponent": args.opponent,
        "games_requested": int(args.games),
        "games_completed": int(games_completed),
        "candidate_wins": int(wins),
        "rows": int(rows),
        "simulations": int(args.simulations),
        "c_puct": float(args.c_puct),
        "search_temperature": float(args.search_temperature),
        "max_depth": int(args.max_depth),
        "prior_temperature": float(args.prior_temperature),
        "value_scale": float(args.value_scale),
        **_feature_contract_manifest_fields(feature_contract),
        "seed": int(args.seed),
        "skipped_forced": int(skipped_forced),
        "skipped_phase": int(skipped_phase),
        "skipped_window": int(skipped_window),
        "terminal_rows": int(terminal_rows),
        "truncated_rows": int(truncated_rows),
        "elapsed_sec": time.perf_counter() - started,
        "rows_per_sec": rows / max(time.perf_counter() - started, 1.0e-9),
        "shards": [str(path) for path in writer.paths],
    }
    (output / "manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


def _target_scores_and_mask(
    q_by_rust: dict[int, float],
    visits_by_rust: dict[int, int],
    legal_rust: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """FIX (Q-mask): RustMCTSResult.q_values defaults an UNVISITED action's Q to 0.0 (a finite
    placeholder, not NaN -- see ``_ActionStats.q`` in rust_mcts.py), so ``np.isfinite`` alone
    marked never-expanded root children as valid soft-score targets. Require visits > 0 too."""
    target_scores = np.asarray(
        [float(q_by_rust.get(int(action), np.nan)) for action in legal_rust],
        dtype=np.float32,
    )
    visited = np.asarray(
        [int(visits_by_rust.get(int(action), 0)) > 0 for action in legal_rust],
        dtype=np.bool_,
    )
    target_scores_mask = np.isfinite(target_scores) & visited
    return target_scores, target_scores_mask


def _target_policy_and_mask(
    policy_by_rust: dict[int, float],
    legal_rust: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize the search policy with support over every legal action.

    Exact-zero probabilities are still teacher labels.  The mask describes
    which legal actions the distribution covers, not which entries carry
    positive probability.
    """

    target_policy = np.asarray(
        [float(policy_by_rust.get(int(action), 0.0)) for action in legal_rust],
        dtype=np.float16,
    )
    target_policy_mask = np.ones(target_policy.shape, dtype=np.bool_)
    return target_policy, target_policy_mask


def _evaluator_feature_contract(
    evaluator: EntityGraphRustEvaluator,
) -> dict[str, Any]:
    """Return the exact input semantics already enforced by the evaluator."""

    adapter_version = policy_entity_feature_adapter_version(evaluator.policy)
    requested_adapter = evaluator.config.entity_feature_adapter_version
    if requested_adapter != adapter_version:
        raise RuntimeError(
            "reanalysis evaluator feature adapter drift: "
            f"checkpoint={adapter_version!r} runtime={requested_adapter!r}"
        )
    history_enabled, history_limit, history_schema = _policy_history_options(
        evaluator.policy
    )
    return {
        "entity_feature_adapter_version": adapter_version,
        "public_observation": bool(evaluator.config.public_observation),
        "action_context_fill": float(evaluator.config.context_fill),
        "meaningful_public_history": bool(history_enabled),
        "meaningful_public_history_schema": str(history_schema),
        "event_history_limit": int(history_limit),
    }


def _feature_contract_manifest_fields(
    feature_contract: dict[str, Any],
) -> dict[str, Any]:
    """Serialize one evaluator contract identically in every reanalysis CLI."""

    return {
        "adapter_version": feature_contract["entity_feature_adapter_version"],
        "public_observation": feature_contract["public_observation"],
        "action_context_fill": feature_contract["action_context_fill"],
        "meaningful_public_history": feature_contract[
            "meaningful_public_history"
        ],
        "meaningful_public_history_schema": feature_contract[
            "meaningful_public_history_schema"
        ],
        "event_history_limit": feature_contract["event_history_limit"],
        # This legacy reanalyzer searches the authoritative game object. Masking
        # the evaluator observation does not turn that tree into an information-
        # set search: transitions and chance still follow the one hidden state.
        # Stamp the rows and manifest explicitly so generic BC ingestion cannot
        # silently treat these omniscient targets as public-policy supervision.
        "target_information_regime": TARGET_INFORMATION_REGIME_AUTHORITATIVE,
        "policy_weight_multiplier": 0.0,
    }


def _mcts_row(
    game: Any,
    *,
    search: RustMCTS,
    evaluator: EntityGraphRustEvaluator,
    legal_rust: tuple[int, ...],
    candidate_color: str,
    game_seed: int,
    decision_index: int,
    obs_width: int,
) -> dict[str, Any]:
    feature_contract = _evaluator_feature_contract(evaluator)
    mapped = rust_policy_action_ids(
        game,
        legal_rust,
        colors=COLORS,
        action_size=int(evaluator.policy.action_size),
    )
    result = search.search(game)
    policy_by_rust = result.policy
    q_by_rust = result.q_values
    target_policy, target_policy_mask = _target_policy_and_mask(
        policy_by_rust,
        legal_rust,
    )
    target_scores, target_scores_mask = _target_scores_and_mask(
        q_by_rust,
        getattr(result, "visits", {}) or {},
        legal_rust,
    )
    best_rust = int(result.action)
    best_policy = mapped[legal_rust.index(best_rust)]
    entity = rust_game_to_entity_batch(
        game,
        legal_rust,
        actor=str(game.current_color()),
        colors=COLORS,
        action_size=int(evaluator.policy.action_size),
        policy_action_ids=mapped,
        public_observation=feature_contract["public_observation"],
        meaningful_public_history=feature_contract[
            "meaningful_public_history"
        ],
        history_limit=feature_contract["event_history_limit"],
        meaningful_public_history_schema=feature_contract[
            "meaningful_public_history_schema"
        ],
        entity_feature_adapter_version=feature_contract[
            "entity_feature_adapter_version"
        ],
    )
    features = {key: value[0] for key, value in entity.items()}
    context = rust_action_context_batch(
        game,
        legal_rust,
        actor=str(game.current_color()),
        colors=COLORS,
        action_size=int(evaluator.policy.action_size),
        fill=feature_contract["action_context_fill"],
        policy_action_ids=mapped,
        public_observation=feature_contract["public_observation"],
        entity_feature_adapter_version=feature_contract[
            "entity_feature_adapter_version"
        ],
    )[0]
    actual_vps = {
        color: _actual_victory_points(json.loads(game.player_state_json(color)))
        for color in COLORS
    }
    return {
        "obs": np.zeros((int(obs_width),), dtype=np.float16),
        "legal_action_ids": np.asarray(mapped, dtype=np.int16),
        "legal_action_context": context.astype(np.float16, copy=False),
        "action_taken": np.int16(best_policy),
        "target_policy": target_policy,
        "target_scores": target_scores,
        "target_policy_mask": target_policy_mask,
        "target_scores_mask": target_scores_mask,
        "target_score_source": "rust_mcts_visit_q",
        "target_information_regime": TARGET_INFORMATION_REGIME_AUTHORITATIVE,
        "policy_weight_multiplier": np.float32(0.0),
        "game_seed": np.int64(game_seed),
        "teacher_name": "rust_mcts_reanalysis",
        "adapter_version": feature_contract["entity_feature_adapter_version"],
        "player": str(game.current_color()),
        "seat": np.int8(COLORS.index(str(game.current_color()))),
        "phase": str(json.loads(game.json_snapshot()).get("current_prompt", "")),
        "decision_index": np.int32(decision_index),
        "winner": "",
        "terminated": False,
        "truncated": False,
        "final_public_vps": np.zeros(len(PLAYER_NAMES), dtype=np.int16),
        "has_final_public_vps": False,
        "final_actual_vps": np.asarray([int(actual_vps.get(name, 0)) for name in PLAYER_NAMES], dtype=np.int16),
        "has_final_actual_vps": False,
        "action_mask_version": "colonist-multiagent-v1",
        "_features": features,
        "_rust_action": best_rust,
    }


def _game_outcome_fields(game: Any, *, terminal: bool) -> dict[str, Any]:
    winner = game.winning_color()
    public_vps: dict[str, int] = {}
    actual_vps: dict[str, int] = {}
    for color in COLORS:
        state = json.loads(game.player_state_json(color))
        public_vps[color] = int(state.get("victory_points", 0) or 0)
        actual_vps[color] = _actual_victory_points(state)
    return {
        "winner": str(winner) if terminal and winner is not None else "",
        "terminated": bool(terminal),
        "truncated": not bool(terminal),
        "final_public_vps": np.asarray(
            [int(public_vps.get(name, 0)) for name in PLAYER_NAMES],
            dtype=np.int16,
        ),
        "has_final_public_vps": True,
        "final_actual_vps": np.asarray(
            [int(actual_vps.get(name, 0)) for name in PLAYER_NAMES],
            dtype=np.int16,
        ),
        "has_final_actual_vps": True,
    }


def _raw_policy_action(
    evaluator: EntityGraphRustEvaluator,
    game: Any,
    legal_rust: tuple[int, ...],
) -> int:
    priors, _value = evaluator.evaluate(
        game,
        legal_rust,
        root_color=str(game.current_color()),
        colors=COLORS,
    )
    return max(legal_rust, key=lambda action: priors.get(int(action), 0.0))


def _csv_patterns(raw: str) -> tuple[str, ...]:
    return tuple(part.strip().lower() for part in str(raw or "").split(",") if part.strip())


def _phase_selected(
    phase: str,
    *,
    include: tuple[str, ...],
    exclude: tuple[str, ...],
) -> bool:
    lowered = str(phase).lower()
    if include and not any(pattern in lowered for pattern in include):
        return False
    if exclude and any(pattern in lowered for pattern in exclude):
        return False
    return True


def _apply_action_with_sampled_chance(
    game: Any,
    action_index: int,
    *,
    colors: tuple[str, ...],
    rng: random.Random,
) -> Any:
    ids = [
        int(action)
        for action in game.playable_action_indices(list(colors), None)
    ]
    actions = json.loads(game.playable_actions_json())
    action_by_id = {action_id: action for action_id, action in zip(ids, actions)}
    action_json = action_by_id.get(int(action_index))
    if action_json is None:
        raise RuntimeError(f"selected action {action_index} is not legal")
    spectrum = json.loads(game.spectrum_json(json.dumps(action_json)))
    if not spectrum:
        game.execute_action_index(int(action_index), list(colors), None)
        return game
    draw = rng.random()
    cumulative = 0.0
    for outcome_index, outcome in enumerate(spectrum):
        cumulative += float(outcome.get("probability", 0.0))
        if draw <= cumulative:
            return game.apply_chance_outcome(json.dumps(action_json), outcome_index)
    return game.apply_chance_outcome(json.dumps(action_json), len(spectrum) - 1)


def _actual_victory_points(player_state: dict[str, Any]) -> int:
    return int(
        player_state.get(
            "actual_victory_points",
            player_state.get("victory_points", 0),
        )
        or 0
    )


if __name__ == "__main__":
    main()
