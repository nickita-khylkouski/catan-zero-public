"""Raw-policy self-play data-generation driver (value-head repair v2, task #65).

Gate-A diagnosis (2026-07-04): search fails to beat raw policy because the
root Q-estimate SNR is too low -- the value head's corr(q, z) = 0.61 is decent
but 1-8 value samples per candidate from it can't out-signal the prior. The
durable fix is retraining the value head on the SEARCH-RELEVANT state
distribution with TRUE game outcomes (not the BC corpus, which value-repair-v1
already proved is at its objective's optimum for off-policy states).

This module plays full 2-player games where BOTH seats select actions via the
checkpoint's RAW policy priors -- NO `GumbelChanceMCTS` (or any search)
anywhere in this module. For the first `temperature_decisions` decisions of
each game, the action is sampled from the (temperature-reweighted) priors for
trajectory diversity; thereafter it is the argmax, ties broken by lowest rust
action id (matches `tools/gumbel_search_vs_raw_h2h.py`'s `_select_raw_action`).

Reuses `catan_zero.rl.gumbel_self_play`'s game-loop primitives
(`_apply_selected_action`, `_game_outcome_fields`, `GumbelShardWriter`,
`action_size_for_evaluator`, `_write_json_atomic`) so the on-disk shard
schema, chance-spectrum corrections (A19/A20), and PLAYER_NAMES/seat
convention are byte-for-byte identical to the searched driver -- this data is
meant to retrain the SAME model's value head on true outcomes, not to
introduce a second, divergent schema.

Every row gets `policy_weight_multiplier=0.0` (raw argmax/temperature-sampled
actions are not a policy target worth imitating -- imitating them would
re-teach the policy trunk toward its own unimproved priors) and
`value_weight_multiplier=1.0` (every row is a real, ON-POLICY value sample
from exactly the state distribution search queries at inference time, unlike
value-repair-v1's off-policy BC-corpus states).

No search means no per-action afterstate enumeration or Q-values:
`afterstate_target`/`afterstate_target_mask` and `target_scores`/
`target_scores_mask` are written honestly empty (all-False mask), not
fabricated. The true-outcome value signal this generator exists to produce
comes entirely from the terminal `winner`/`final_actual_vps`/`final_public_vps`
fields (identical convention to the searched driver), which
`tools/train_bc.py`'s `_value_targets()` already consumes.
"""

from __future__ import annotations

import dataclasses
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from catan_zero.rl.gumbel_self_play import (
    ACTION_MASK_VERSION,
    COLORS,
    PLAYER_NAMES,
    GumbelShardWriter,
    TARGET_INFORMATION_REGIME_PUBLIC_COHERENT,
    _apply_selected_action,
    _build_public_learner_features,
    _game_outcome_fields,
    _write_json_atomic,
    action_size_for_evaluator,
)
from catan_zero.search.neural_rust_mcts import (
    RUST_ENTITY_ADAPTER_VERSION,
)
from catan_zero.rl.meaningful_history import (
    MEANINGFUL_PUBLIC_HISTORY_SCHEMA_VERSION,
)
from catan_zero.search.rust_mcts import RustEvaluator, _require_rust_module

__all__ = [
    "COLORS",
    "TEACHER_NAME",
    "TARGET_SCORE_SOURCE",
    "RawSelfPlayConfig",
    "RawDecisionRecord",
    "RawGameRecord",
    "play_one_raw_selfplay_game",
    "run_raw_selfplay_worker_games",
]

TEACHER_NAME = "raw_selfplay"
# Distinct from gumbel_self_play's "gumbel_mcts_visit_q" -- no search means no
# per-action Q-value estimate exists at all (target_scores_mask is all-False
# on every row regardless of this label; the label only identifies provenance
# for any future code that branches on target_score_source).
TARGET_SCORE_SOURCE = "raw_selfplay_no_search"


@dataclass(frozen=True, slots=True)
class RawSelfPlayConfig:
    colors: tuple[str, ...] = COLORS
    map_kind: str | None = None
    track: str = "2p_no_trade"
    vps_to_win: int = 10
    obs_width: int = 806
    max_decisions: int = 600
    # First `temperature_decisions` decisions of EACH game sample from the
    # (temperature-reweighted) raw priors for trajectory diversity; the rest
    # are argmax. Unlike the searched driver's `temperature_move_fraction`
    # (a fraction of max_decisions), this is an ABSOLUTE decision count per
    # task #65's spec ("temperature sampling ... for the first 45 decisions"),
    # independent of --max-decisions.
    temperature_decisions: int = 45
    temperature: float = 1.0
    # Mirror of GumbelSelfPlayConfig.correct_rust_chance_spectra: the live
    # game's own chance resolution needs the same verified-bug correction
    # (A19/A20) as the searched driver to keep recorded trajectories valid.
    correct_rust_chance_spectra: bool = True
    meaningful_public_history: bool = False
    meaningful_public_history_schema: str = (
        MEANINGFUL_PUBLIC_HISTORY_SCHEMA_VERSION
    )
    event_history_limit: int = 64
    entity_feature_adapter_version: str = RUST_ENTITY_ADAPTER_VERSION
    target_information_regime: str = TARGET_INFORMATION_REGIME_PUBLIC_COHERENT


@dataclass(slots=True)
class RawDecisionRecord:
    row: dict[str, Any]
    features: dict[str, np.ndarray]


@dataclass(slots=True)
class RawGameRecord:
    game_seed: int
    game_index: int
    decisions: list[RawDecisionRecord]
    terminal: bool
    truncated: bool
    winner: str
    total_decisions: int
    forced_decisions: int
    wall_time_sec: float


def _select_action(
    evaluator: RustEvaluator,
    game: Any,
    legal_actions: tuple[int, ...],
    *,
    acting_color: str,
    decision_index: int,
    config: RawSelfPlayConfig,
    rng: random.Random,
) -> tuple[int, dict[int, float]]:
    """Select an action from the evaluator's raw priors -- no search.

    Forced (single-legal-action) decisions skip the evaluator call entirely
    (matches `tools/gumbel_search_vs_raw_h2h.py`'s `_select_raw_action`
    optimization) and report a trivial one-hot prior. The argmax phase breaks
    ties by lowest rust action id, identical to that same helper, for
    consistency with the H2H tool this generator's raw-policy behavior was
    validated against.
    """
    if len(legal_actions) == 1:
        action = int(legal_actions[0])
        return action, {action: 1.0}

    priors, _value = evaluator.evaluate(
        game, legal_actions, root_color=acting_color, colors=config.colors
    )
    if decision_index < int(config.temperature_decisions) and float(config.temperature) > 0.0:
        action = _sample_from_priors(
            rng, legal_actions, priors, temperature=float(config.temperature)
        )
    else:
        action = int(
            max(legal_actions, key=lambda a: (float(priors.get(int(a), 0.0)), -int(a)))
        )
    return action, priors


def _sample_from_priors(
    rng: random.Random,
    legal_actions: tuple[int, ...],
    priors: dict[int, float],
    *,
    temperature: float,
) -> int:
    weights = [max(float(priors.get(int(a), 0.0)), 0.0) for a in legal_actions]
    if temperature != 1.0:
        weights = [w ** (1.0 / temperature) if w > 0.0 else 0.0 for w in weights]
    total = sum(weights)
    if total <= 0.0:
        return int(rng.choice(legal_actions))
    draw = rng.random() * total
    cumulative = 0.0
    for action, weight in zip(legal_actions, weights):
        cumulative += weight
        if draw <= cumulative:
            return int(action)
    return int(legal_actions[-1])


def _build_raw_decision_row(
    game: Any,
    *,
    selected_action: int,
    priors: dict[int, float],
    action_size: int,
    colors: tuple[str, ...],
    game_seed: int,
    decision_index: int,
    obs_width: int,
    meaningful_public_history: bool,
    meaningful_public_history_schema: str,
    event_history_limit: int,
    entity_feature_adapter_version: str,
    target_information_regime: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    legal_rust = tuple(sorted(priors.keys()))
    is_forced = len(legal_rust) <= 1
    acting_color = str(game.current_color())
    mapped, features, context, snapshot, _action_by_id = _build_public_learner_features(
        game,
        legal_rust,
        colors=colors,
        action_size=action_size,
        actor=acting_color,
        meaningful_public_history=meaningful_public_history,
        meaningful_public_history_schema=meaningful_public_history_schema,
        event_history_limit=event_history_limit,
        entity_feature_adapter_version=entity_feature_adapter_version,
    )

    # target_policy IS the evaluator's raw prior distribution -- there is no
    # search to "improve" it in this driver, so target==prior by construction
    # (both fields are still populated so the shared schema and any future
    # KL(target||prior) tooling keep working without special-casing a
    # no-search generator). fp32 per F4 (fp16 flushes small-but-real mass
    # near zero, worst at wide/54-action placement roots).
    target_policy = np.asarray(
        [float(priors.get(int(action), 0.0)) for action in legal_rust], dtype=np.float32
    )
    target_policy_mask = target_policy > 0.0
    prior_policy = target_policy.astype(np.float16, copy=True)

    legal_width = len(legal_rust)
    target_scores = np.full((legal_width,), np.nan, dtype=np.float32)
    target_scores_mask = np.zeros((legal_width,), dtype=bool)
    afterstate_target = np.full((legal_width,), np.nan, dtype=np.float32)
    afterstate_target_mask = np.zeros((legal_width,), dtype=bool)

    best_policy = mapped[legal_rust.index(int(selected_action))]

    row: dict[str, Any] = {
        "obs": np.zeros((int(obs_width),), dtype=np.float16),
        "legal_action_ids": np.asarray(mapped, dtype=np.int16),
        "legal_action_context": context.astype(np.float16, copy=False),
        "action_taken": np.int16(best_policy),
        "target_policy": target_policy,
        "target_scores": target_scores,
        "target_policy_mask": target_policy_mask,
        "target_scores_mask": target_scores_mask,
        "target_score_source": TARGET_SCORE_SOURCE,
        "target_information_regime": target_information_regime,
        "game_seed": np.int64(game_seed),
        "teacher_name": TEACHER_NAME,
        "adapter_version": entity_feature_adapter_version,
        "player": acting_color,
        # Must index PLAYER_NAMES order, not `colors` order -- see the
        # identical convention/warning in gumbel_self_play.py's
        # _build_decision_row (A-bug precedent: a colors.index() here would
        # silently swap every row's seat to the wrong player's VP slot).
        "seat": np.int8(PLAYER_NAMES.index(acting_color)),
        "phase": str(snapshot.get("current_prompt", "")),
        "decision_index": np.int32(decision_index),
        # Outcome fields are placeholders here; filled in by the caller once
        # the game ends (winner is not known yet).
        "winner": "",
        "terminated": False,
        "truncated": False,
        "final_public_vps": np.zeros(len(PLAYER_NAMES), dtype=np.int16),
        "has_final_public_vps": False,
        "final_actual_vps": np.zeros(len(PLAYER_NAMES), dtype=np.int16),
        "has_final_actual_vps": False,
        "action_mask_version": ACTION_MASK_VERSION,
        # Every row: no search-improved policy signal to imitate, but a real,
        # on-policy value sample -- the entire point of this generator.
        "policy_weight_multiplier": np.float32(0.0),
        "value_weight_multiplier": np.float32(1.0),
        "used_full_search": False,
        "is_forced": bool(is_forced),
        "simulations_used": np.int32(0),
        "afterstate_target": afterstate_target,
        "afterstate_target_mask": afterstate_target_mask,
        "prior_policy": prior_policy,
    }
    return row, features


def play_one_raw_selfplay_game(
    evaluator: RustEvaluator,
    *,
    config: RawSelfPlayConfig,
    game_seed: int,
    game_index: int,
    action_size: int,
    seed: int,
) -> RawGameRecord:
    """Play one full raw-policy self-play game, recording one row per decision.

    The live game's own chance outcomes (dice, robber steals, dev card draws)
    are sampled from a `game_seed`-derived RNG, independent of the action-
    selection RNG, matching `play_one_game`'s determinism contract: a game's
    board/chance trajectory is reproducible from `game_seed` alone given the
    same sequence of chosen actions.
    """
    started = time.perf_counter()
    catanatron_rs = _require_rust_module()
    game = catanatron_rs.Game.simple(list(config.colors), seed=int(game_seed))
    chance_rng = random.Random(int(game_seed) ^ 0xA17E)
    select_rng = random.Random((int(seed) ^ int(game_seed) ^ 0x51ED270B) & 0xFFFFFFFF)

    decisions: list[RawDecisionRecord] = []
    decision_index = 0
    forced_decisions = 0
    terminal = False

    while decision_index < int(config.max_decisions):
        if game.winning_color() is not None:
            terminal = True
            break
        legal_rust = tuple(
            int(action)
            for action in game.playable_action_indices(list(config.colors), config.map_kind)
        )
        if not legal_rust:
            break

        acting_color = str(game.current_color())
        selected_action, priors = _select_action(
            evaluator,
            game,
            legal_rust,
            acting_color=acting_color,
            decision_index=decision_index,
            config=config,
            rng=select_rng,
        )
        if len(legal_rust) <= 1:
            forced_decisions += 1

        row, features = _build_raw_decision_row(
            game,
            selected_action=selected_action,
            priors=priors,
            action_size=action_size,
            colors=config.colors,
            game_seed=game_seed,
            decision_index=decision_index,
            obs_width=config.obs_width,
            meaningful_public_history=bool(config.meaningful_public_history),
            meaningful_public_history_schema=str(
                config.meaningful_public_history_schema
            ),
            event_history_limit=int(config.event_history_limit),
            entity_feature_adapter_version=str(
                config.entity_feature_adapter_version
            ),
            target_information_regime=str(config.target_information_regime),
        )
        decisions.append(RawDecisionRecord(row=row, features=features))

        game = _apply_selected_action(
            game,
            selected_action,
            colors=config.colors,
            rng=chance_rng,
            correct_rust_chance_spectra=config.correct_rust_chance_spectra,
        )
        decision_index += 1

    if not terminal:
        terminal = game.winning_color() is not None
    truncated = not terminal
    outcome = _game_outcome_fields(game, terminal=terminal, colors=config.colors)
    for record in decisions:
        record.row.update(outcome)

    return RawGameRecord(
        game_seed=int(game_seed),
        game_index=int(game_index),
        decisions=decisions,
        terminal=terminal,
        truncated=truncated,
        winner=str(outcome["winner"]),
        total_decisions=decision_index,
        forced_decisions=forced_decisions,
        wall_time_sec=time.perf_counter() - started,
    )


def run_raw_selfplay_worker_games(
    *,
    out_dir: Path,
    games: int,
    game_index_start: int,
    base_seed: int,
    worker_seed: int,
    config: RawSelfPlayConfig,
    evaluator: RustEvaluator,
    shard_size: int = 2048,
    fmt: str = "npz",
) -> dict[str, Any]:
    """Play `games` raw-policy self-play games in this process.

    Per-game exception isolation (one bad game is recorded and skipped, not
    fatal to the worker) and a per-worker `manifest.json` compatible with
    `tools/train_bc.py`'s loader, matching `gumbel_self_play.run_worker_games`.
    """
    action_size = action_size_for_evaluator(evaluator, config.colors)
    writer = GumbelShardWriter(out_dir, shard_size=shard_size, fmt=fmt)

    started = time.perf_counter()
    games_completed = 0
    games_failed = 0
    games_truncated = 0
    rows = 0
    decisions_total = 0
    forced_decisions_total = 0
    wins_by_color: dict[str, int] = {color: 0 for color in config.colors}
    errors: list[dict[str, Any]] = []

    try:
        for offset in range(int(games)):
            game_index = int(game_index_start) + offset
            game_seed = int(base_seed) + game_index
            try:
                record = play_one_raw_selfplay_game(
                    evaluator,
                    config=config,
                    game_seed=game_seed,
                    game_index=game_index,
                    action_size=action_size,
                    seed=int(worker_seed),
                )
            except Exception as error:  # noqa: BLE001 - isolate one bad game from the worker.
                games_failed += 1
                errors.append(
                    {"game_index": game_index, "game_seed": game_seed, "error": repr(error)}
                )
                continue

            for decision in record.decisions:
                writer.add(decision.row, decision.features)
            rows += len(record.decisions)
            decisions_total += record.total_decisions
            forced_decisions_total += record.forced_decisions
            games_completed += 1
            if record.truncated:
                games_truncated += 1
            if record.terminal and record.winner in wins_by_color:
                wins_by_color[record.winner] += 1
    finally:
        writer.close()

    elapsed = time.perf_counter() - started
    summary: dict[str, Any] = {
        "out_dir": str(out_dir),
        "track": config.track,
        "vps_to_win": int(config.vps_to_win),
        "colors": list(config.colors),
        "games_requested": int(games),
        "games_completed": int(games_completed),
        "games_failed": int(games_failed),
        "games_truncated": int(games_truncated),
        "wins_by_color": wins_by_color,
        "rows": int(rows),
        "decisions_total": int(decisions_total),
        "forced_decisions_total": int(forced_decisions_total),
        "worker_seed": int(worker_seed),
        "base_seed": int(base_seed),
        "game_index_start": int(game_index_start),
        "adapter_version": config.entity_feature_adapter_version,
        "target_information_regime": config.target_information_regime,
        # Full config provenance (982d344 pattern): what the worker actually
        # constructed, for post-hoc audit via audit_gumbel_pilot_shards.py's
        # check_config_provenance.
        "selfplay_config": dataclasses.asdict(config),
        "elapsed_sec": elapsed,
        "rows_per_sec": rows / max(elapsed, 1.0e-9),
        "shards": [str(path) for path in writer.paths],
        "errors": errors,
    }
    _write_json_atomic(Path(out_dir) / "manifest.json", summary)
    return summary
