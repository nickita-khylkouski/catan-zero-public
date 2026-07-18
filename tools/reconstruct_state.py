"""Reconstruct a live Rust game at an archived decision from shard rows (task #64).

Rows do NOT store an engine snapshot. They CAN be replayed, because both
self-play drivers (`gumbel_self_play.play_one_game`,
`raw_selfplay.play_one_raw_selfplay_game`) are deterministic from `game_seed`:

  * the board is `catanatron_rs.Game.simple(colors, seed=game_seed)`,
  * every chance outcome (dice, robber steal, dev-card draw) is drawn from
    `chance_rng = random.Random(game_seed ^ 0xA17E)` inside
    `_apply_selected_action` -- NOT the engine's own RNG,
  * so given the recorded action sequence, replaying reproduces the exact same
    trajectory the shard was written from.

The one subtlety the team-lead flagged: `action_taken` in a row is the
POLICY-CATALOG id (`mapped[legal_rust.index(rust_id)]`), not the Rust action
index that `_apply_selected_action` consumes. We invert that per state by
recomputing `rust_policy_action_ids` on the live game and matching the stored
policy id back to its Rust action id (asserting the match is unique -- it is,
within a single state's legal set, since the catalog key is injective there).

The round-trip test (`round_trip_shard_rows`) reconstructs each sampled row's
state, re-featurises it, and checks the entity tokens match the stored tokens
within fp16 tolerance AND the recomputed legal-action policy ids match the
stored `legal_action_ids` exactly -- proving the reconstruction is the same
state the row was recorded at, and (empirically) resolving the
`correct_rust_chance_spectra` flag when a manifest predates config-provenance.

Hidden-info note (DAGS, arXiv 2605.14379): reconstruction replays the game's
TRUE history, so the restart state is a legitimate reachable public state, not
an omniscient fabrication. Restart rows are still flagged
(start_mode="archived_public_state") so downstream training keeps intermediate-
start data on separate metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from catan_zero.rl.action_mask import ActionCatalog
from catan_zero.rl.entity_feature_adapter import (
    CURRENT_RUST_ENTITY_ADAPTER_VERSION,
)
from catan_zero.rl.gumbel_self_play import (
    _action_type_of,
    _apply_selected_action,
    _build_public_learner_features,
)
from catan_zero.rl.meaningful_history import (
    MEANINGFUL_PUBLIC_HISTORY_SCHEMA_VERSION,
)
from catan_zero.search.neural_rust_mcts import (
    rust_policy_action_ids,
)
from catan_zero.search.rust_mcts import _require_rust_module

# Same salt both drivers use to derive the per-game chance RNG.
CHANCE_RNG_SALT = 0xA17E
DEFAULT_COLORS = ("RED", "BLUE")


@dataclass(slots=True)
class GameActionSequence:
    game_seed: int
    colors: tuple[str, ...]
    # Recorded action_taken policy-catalog ids, ordered by decision_index.
    # Modern corpora intentionally omit single-legal-action UI/chance plumbing;
    # decision_indices may therefore be sparse.  Every omitted index is
    # replayable only when the live engine proves that exactly one action was
    # legal there.
    actions: list[int]
    decision_indices: list[int]
    phases: list[str]
    players: list[str]

    def __len__(self) -> int:
        return len(self.actions)


class SparseReconstructionError(ValueError):
    """A sparse trajectory cannot uniquely determine an archived root."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        game_seed: int,
        decision_index: int,
        legal_action_count: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = str(code)
        self.game_seed = int(game_seed)
        self.decision_index = int(decision_index)
        self.legal_action_count = (
            None if legal_action_count is None else int(legal_action_count)
        )


def _validated_sparse_actions(
    *,
    game_seed: int,
    actions: list[int],
    decision_indices: list[int] | None,
) -> dict[int, int]:
    indices = (
        list(range(len(actions)))
        if decision_indices is None
        else [int(value) for value in decision_indices]
    )
    if len(indices) != len(actions):
        raise SparseReconstructionError(
            "malformed_sequence",
            "actions and decision_indices have different lengths",
            game_seed=game_seed,
            decision_index=-1,
        )
    if any(value < 0 for value in indices) or indices != sorted(set(indices)):
        raise SparseReconstructionError(
            "malformed_sequence",
            "decision_indices must be unique, non-negative, and increasing",
            game_seed=game_seed,
            decision_index=-1,
        )
    return {
        int(decision): int(action)
        for decision, action in zip(indices, actions, strict=True)
    }


def action_size_for_colors(colors: tuple[str, ...]) -> int:
    """Return the catalog minimum when no checkpoint contract is available.

    Production checkpoints may deliberately retain a larger inherited action
    dimension (the current f7 checkpoint uses 567 while the two-player catalog
    has 332 entries). Policy ids remain identical, but legal-action token
    normalization depends on the checkpoint dimension. Exact corpus roundtrip
    must therefore pass the authenticated checkpoint ``action_size``; this
    helper is only a legacy/fallback minimum.
    """
    return int(ActionCatalog(colors).size)


def gather_game_action_sequence(
    scope: Path,
    game_seed: int,
    *,
    colors: tuple[str, ...] = DEFAULT_COLORS,
    allow_omitted_automatic_transitions: bool = False,
) -> GameActionSequence:
    """Collect one game's full action sequence by streaming shards under `scope`.

    `scope` should be the worker directory that produced the game (the shard
    file's parent) -- a worker plays whole games sequentially, so a game_seed's
    rows are fully contained there and never collide with another worker's
    (game_seed = base_seed + globally-incremented game_index within a run).
    Rows are gathered across the worker's shard files and sorted by
    decision_index.  Legacy callers remain fail-closed on gaps.  Modern
    trajectory-only corpora may opt into sparse reconstruction: gaps are then
    admitted here but still have to prove single-action uniqueness while the
    engine replays them.
    """
    from regret_common import discover_shards, load_shard

    rows: list[tuple[int, int, str, str]] = []  # (decision_index, action, phase, player)
    for path in discover_shards([Path(scope)]):
        shard = load_shard(path)
        seeds = np.asarray(shard["game_seed"]).reshape(-1)
        sel = np.nonzero(seeds == int(game_seed))[0]
        if sel.size == 0:
            continue
        actions = np.asarray(shard["action_taken"]).reshape(-1)
        didx = np.asarray(shard["decision_index"]).reshape(-1)
        phases = np.asarray(shard.get("phase", np.full(seeds.shape[0], ""))).astype(str)
        players = np.asarray(shard.get("player", np.full(seeds.shape[0], ""))).astype(str)
        for i in sel:
            rows.append(
                (int(didx[i]), int(actions[i]), str(phases[i]), str(players[i]))
            )
    if not rows:
        raise ValueError(f"no rows found for game_seed={game_seed} under {scope}")
    rows.sort(key=lambda r: r[0])
    dec_indices = [r[0] for r in rows]
    if len(dec_indices) != len(set(dec_indices)):
        raise ValueError(
            f"game_seed={game_seed} decision_index sequence has duplicates"
        )
    expected = list(range(dec_indices[0], dec_indices[0] + len(dec_indices)))
    if not allow_omitted_automatic_transitions and dec_indices != expected:
        raise ValueError(
            f"game_seed={game_seed} decision_index sequence has gaps/dupes: "
            f"got {dec_indices[:8]}... expected contiguous from {dec_indices[0]}"
        )
    if dec_indices[0] != 0:
        raise ValueError(
            f"game_seed={game_seed} does not start at decision_index 0 "
            f"(first={dec_indices[0]}); scope may be incomplete"
        )
    return GameActionSequence(
        game_seed=int(game_seed),
        colors=tuple(colors),
        actions=[r[1] for r in rows],
        decision_indices=dec_indices,
        phases=[r[2] for r in rows],
        players=[r[3] for r in rows],
    )


def _policy_id_to_rust_id(
    game: Any,
    want_policy_id: int,
    *,
    colors: tuple[str, ...],
    action_size: int,
) -> int:
    """Invert the policy-catalog id back to the live game's Rust action index.

    Raises if the policy id is not legal here, or (defensively) if two legal
    Rust actions collide onto it -- either signals the replay has diverged from
    the recorded trajectory.
    """
    legal_rust = tuple(
        int(a) for a in game.playable_action_indices(list(colors), None)
    )
    mapped = rust_policy_action_ids(
        game, legal_rust, colors=colors, action_size=action_size
    )
    matches = [rust for rust, pol in zip(legal_rust, mapped) if int(pol) == int(want_policy_id)]
    if len(matches) == 0:
        raise ValueError(
            f"policy id {want_policy_id} not legal in reconstructed state "
            f"(legal policy ids: {sorted(int(m) for m in mapped)}) -- replay diverged"
        )
    if len(matches) > 1:
        raise ValueError(
            f"policy id {want_policy_id} maps to multiple Rust actions {matches} "
            "-- reconstruction is ambiguous for this state"
        )
    return int(matches[0])


def _apply_sparse_replay_step(
    game: Any,
    *,
    decision_index: int,
    recorded: dict[int, int],
    game_seed: int,
    colors: tuple[str, ...],
    action_size: int,
    chance_rng: random.Random,
    correct_rust_chance_spectra: bool,
) -> tuple[Any, str | None]:
    """Apply one recorded or provably unique omitted transition."""

    d = int(decision_index)
    if game.winning_color() is not None:
        raise SparseReconstructionError(
            "terminal_before_target",
            f"game ended at decision {d} before requested target",
            game_seed=game_seed,
            decision_index=d,
        )
    legal_rust = tuple(
        int(action) for action in game.playable_action_indices(list(colors), None)
    )
    omitted = d not in recorded
    if not omitted:
        try:
            rust_id = _policy_id_to_rust_id(
                game, recorded[d], colors=colors, action_size=action_size
            )
        except ValueError as error:
            raise SparseReconstructionError(
                "recorded_action_illegal",
                str(error),
                game_seed=game_seed,
                decision_index=d,
                legal_action_count=len(legal_rust),
            ) from error
    elif len(legal_rust) == 1:
        rust_id = int(legal_rust[0])
    else:
        raise SparseReconstructionError(
            "missing_nonautomatic_decision",
            "an omitted decision has more than one legal action; the stored "
            "bytes do not determine which branch was taken",
            game_seed=game_seed,
            decision_index=d,
            legal_action_count=len(legal_rust),
        )
    action_json = None
    omitted_action_type = None
    if omitted:
        action_ids = [
            int(action)
            for action in game.playable_action_indices(list(colors), None)
        ]
        raw_actions = json.loads(game.playable_actions_json())
        action_json = dict(zip(action_ids, raw_actions)).get(rust_id)
        if action_json is None:
            raise SparseReconstructionError(
                "runtime_error",
                "unique omitted action is absent from playable_actions_json",
                game_seed=game_seed,
                decision_index=d,
                legal_action_count=len(legal_rust),
            )
        omitted_action_type = _action_type_of(action_json) or "OTHER_UI"
    return (
        _apply_selected_action(
            game,
            rust_id,
            colors=tuple(colors),
            rng=chance_rng,
            correct_rust_chance_spectra=correct_rust_chance_spectra,
            action_json=action_json,
        ),
        omitted_action_type,
    )


def reconstruct_state(
    game_seed: int,
    actions: list[int],
    target_decision: int,
    *,
    decision_indices: list[int] | None = None,
    colors: tuple[str, ...] = DEFAULT_COLORS,
    correct_rust_chance_spectra: bool = True,
    action_size: int | None = None,
    return_rng: bool = False,
) -> Any:
    """Replay a dense or sparse action trace and return one archived root.

    The returned game is the state the actor faced AT decision index
    `target_decision` (before applying the action at that decision).  With a
    sparse ``decision_indices`` trace, every absent prior decision is applied
    only when the engine exposes exactly one legal action.  A missing
    multi-action decision is mathematically ambiguous and fails closed.

    With `return_rng=True`, returns `(game, chance_rng)` where `chance_rng` is
    the game's chance stream positioned exactly where the archived trajectory
    left it -- so a restart continuation can keep drawing from it and the whole
    branched game stays reproducible from `(game_seed, target_decision)` alone.
    """
    if action_size is None:
        action_size = action_size_for_colors(tuple(colors))
    recorded = _validated_sparse_actions(
        game_seed=int(game_seed),
        actions=actions,
        decision_indices=decision_indices,
    )
    maximum = (max(recorded) + 1) if recorded else 0
    if not 0 <= target_decision <= maximum:
        raise ValueError(
            f"target_decision {target_decision} out of range [0, {maximum}]"
        )
    catanatron_rs = _require_rust_module()
    game = catanatron_rs.Game.simple(list(colors), seed=int(game_seed))
    chance_rng = random.Random(int(game_seed) ^ CHANCE_RNG_SALT)
    for d in range(int(target_decision)):
        game, _omitted_action_type = _apply_sparse_replay_step(
            game,
            decision_index=d,
            recorded=recorded,
            game_seed=game_seed,
            colors=colors,
            action_size=action_size,
            chance_rng=chance_rng,
            correct_rust_chance_spectra=correct_rust_chance_spectra,
        )
    if return_rng:
        return game, chance_rng
    return game


def reconstruct_state_from_sequence(
    sequence: GameActionSequence,
    target_decision: int,
    *,
    correct_rust_chance_spectra: bool = True,
    action_size: int | None = None,
    return_rng: bool = False,
) -> Any:
    """Typed sparse-sequence entry point used by Stage-C reanalysis."""

    return reconstruct_state(
        sequence.game_seed,
        sequence.actions,
        target_decision,
        decision_indices=sequence.decision_indices,
        colors=sequence.colors,
        correct_rust_chance_spectra=correct_rust_chance_spectra,
        action_size=action_size,
        return_rng=return_rng,
    )


@dataclass(slots=True)
class SparseReconstructionBatch:
    states: dict[int, Any]
    omitted_automatic_transitions: dict[int, int]
    omitted_automatic_transition_types: dict[int, dict[str, int]]
    failure: SparseReconstructionError | None


def reconstruct_states_from_sequence(
    sequence: GameActionSequence,
    target_decisions: list[int] | tuple[int, ...] | np.ndarray,
    *,
    correct_rust_chance_spectra: bool = True,
    action_size: int | None = None,
) -> SparseReconstructionBatch:
    """Replay one game once and capture several selected archived roots.

    A failure after an earlier target does not invalidate the already captured
    roots.  The caller can classify only targets after the first ambiguous or
    divergent transition, rather than blocking the whole game/corpus.
    """

    targets = sorted(set(int(value) for value in np.asarray(target_decisions).tolist()))
    if not targets or targets[0] < 0:
        raise ValueError("target_decisions must contain non-negative decisions")
    recorded = _validated_sparse_actions(
        game_seed=sequence.game_seed,
        actions=sequence.actions,
        decision_indices=sequence.decision_indices,
    )
    maximum = (max(recorded) + 1) if recorded else 0
    if targets[-1] > maximum:
        raise ValueError(
            f"target decision {targets[-1]} out of range [0, {maximum}]"
        )
    if action_size is None:
        action_size = action_size_for_colors(sequence.colors)
    catanatron_rs = _require_rust_module()
    game = catanatron_rs.Game.simple(
        list(sequence.colors), seed=int(sequence.game_seed)
    )
    chance_rng = random.Random(int(sequence.game_seed) ^ CHANCE_RNG_SALT)
    states: dict[int, Any] = {}
    omitted_by_target: dict[int, int] = {}
    omitted_types_by_target: dict[int, dict[str, int]] = {}
    target_set = set(targets)
    omitted_count = 0
    omitted_type_counts: dict[str, int] = {}
    failure: SparseReconstructionError | None = None
    for decision in range(targets[-1] + 1):
        if decision in target_set:
            states[decision] = game.copy()
            omitted_by_target[decision] = omitted_count
            omitted_types_by_target[decision] = dict(omitted_type_counts)
        if decision == targets[-1]:
            break
        try:
            game, omitted_action_type = _apply_sparse_replay_step(
                game,
                decision_index=decision,
                recorded=recorded,
                game_seed=sequence.game_seed,
                colors=sequence.colors,
                action_size=action_size,
                chance_rng=chance_rng,
                correct_rust_chance_spectra=correct_rust_chance_spectra,
            )
        except SparseReconstructionError as error:
            failure = error
            break
        if omitted_action_type is not None:
            omitted_count += 1
            omitted_type_counts[omitted_action_type] = (
                omitted_type_counts.get(omitted_action_type, 0) + 1
            )
    return SparseReconstructionBatch(
        states=states,
        omitted_automatic_transitions=omitted_by_target,
        omitted_automatic_transition_types=omitted_types_by_target,
        failure=failure,
    )


def featurize_state(
    game: Any,
    *,
    colors: tuple[str, ...] = DEFAULT_COLORS,
    action_size: int | None = None,
    meaningful_public_history: bool = False,
    history_limit: int = 64,
    meaningful_public_history_schema: str = (
        MEANINGFUL_PUBLIC_HISTORY_SCHEMA_VERSION
    ),
    entity_feature_adapter_version: str = CURRENT_RUST_ENTITY_ADAPTER_VERSION,
) -> dict[str, Any]:
    """Entity tokens + legal policy ids + context for the live game's current state.

    Mirrors `_build_decision_row`'s featurisation exactly (same helpers, same
    single-snapshot threading) so the output is directly comparable to a stored
    row's features.
    """
    if action_size is None:
        action_size = action_size_for_colors(tuple(colors))
    # The self-play producer persists rows in sorted Rust-action-id order:
    # `_build_decision_row` derives `legal_rust` from
    # `sorted(result.improved_policy.keys())`.  The engine's playable-action
    # enumeration is not in that order for every action combination (notably
    # Monopoly + Road Building), so replay must apply the same canonical order
    # before building action-indexed features.
    legal_rust = tuple(
        sorted(int(a) for a in game.playable_action_indices(list(colors), None))
    )
    acting_color = str(game.current_color())
    snapshot = json.loads(game.json_snapshot())
    action_ids = [int(a) for a in game.playable_action_indices(list(colors), None)]
    raw_actions = json.loads(game.playable_actions_json())
    action_by_id = {aid: raw for aid, raw in zip(action_ids, raw_actions)}
    mapped, features, context, snapshot, _action_by_id = (
        _build_public_learner_features(
            game,
            legal_rust,
            actor=acting_color,
            colors=colors,
            action_size=action_size,
            snapshot=snapshot,
            action_by_id=action_by_id,
            meaningful_public_history=meaningful_public_history,
            meaningful_public_history_schema=meaningful_public_history_schema,
            event_history_limit=history_limit,
            entity_feature_adapter_version=entity_feature_adapter_version,
        )
    )
    return {
        "features": features,
        "context": context,
        "legal_policy_ids": tuple(int(m) for m in mapped),
        "acting_color": acting_color,
        "phase": str(snapshot.get("current_prompt", "")),
    }


# Entity-token keys compared bit/fp16-tolerance in the round-trip check. These
# depend only on the state (not padding width), so they must match exactly.
_ROUNDTRIP_ENTITY_KEYS = (
    "hex_tokens",
    "vertex_tokens",
    "edge_tokens",
    "player_tokens",
    "global_tokens",
    "event_tokens",
    "event_target_ids",
    "hex_mask",
    "vertex_mask",
    "edge_mask",
    "player_mask",
    "event_mask",
)

# These tensors are indexed by the row's legal-action ordering.  Shards pad
# them to the widest row in the shard, whereas a reconstructed state emits
# only its live legal actions, so round-trip comparison trims the stored side
# to the authenticated legal-id width first.
_ROUNDTRIP_ACTION_KEYS = (
    "legal_action_tokens",
    "legal_action_mask",
    "legal_action_target_ids",
)


@dataclass(slots=True)
class RoundTripResult:
    game_seed: int
    decision_index: int
    ok: bool
    legal_ids_match: bool
    max_abs_diff: float
    worst_key: str
    detail: str = ""
    phase_match: bool = True
    player_match: bool = True


def round_trip_row(
    seq: GameActionSequence,
    row_decision_index: int,
    stored_features: dict[str, np.ndarray],
    stored_legal_ids: np.ndarray,
    *,
    correct_rust_chance_spectra: bool = True,
    fp16_atol: float = 1e-2,
    action_size: int | None = None,
    meaningful_public_history: bool = False,
    history_limit: int = 64,
    meaningful_public_history_schema: str = (
        MEANINGFUL_PUBLIC_HISTORY_SCHEMA_VERSION
    ),
    entity_feature_adapter_version: str = CURRENT_RUST_ENTITY_ADAPTER_VERSION,
    reconstructed_game: Any | None = None,
) -> RoundTripResult:
    """Reconstruct the state at `row_decision_index` and compare to stored data."""
    game = (
        reconstructed_game
        if reconstructed_game is not None
        else reconstruct_state_from_sequence(
            seq,
            row_decision_index,
            correct_rust_chance_spectra=correct_rust_chance_spectra,
            action_size=action_size,
        )
    )
    feat = featurize_state(
        game,
        colors=seq.colors,
        action_size=action_size,
        meaningful_public_history=meaningful_public_history,
        history_limit=history_limit,
        meaningful_public_history_schema=meaningful_public_history_schema,
        entity_feature_adapter_version=entity_feature_adapter_version,
    )

    try:
        sequence_position = seq.decision_indices.index(int(row_decision_index))
    except ValueError:
        sequence_position = -1
    expected_phase = (
        str(seq.phases[sequence_position])
        if 0 <= sequence_position < len(seq.phases)
        else ""
    )
    expected_player = (
        str(seq.players[sequence_position])
        if 0 <= sequence_position < len(seq.players)
        else ""
    )
    reconstructed_phase = feat.get("phase")
    reconstructed_player = feat.get("acting_color")
    phase_match = (
        not expected_phase
        or (
            reconstructed_phase is not None
            and str(reconstructed_phase) == expected_phase
        )
    )
    player_match = (
        not expected_player
        or (
            reconstructed_player is not None
            and str(reconstructed_player) == expected_player
        )
    )

    # Legal-action policy ids must match exactly (padding stripped).
    stored_ids = np.asarray(stored_legal_ids).reshape(-1)
    stored_ids = stored_ids[stored_ids >= 0]
    recon_ids = np.asarray(feat["legal_policy_ids"], dtype=stored_ids.dtype)
    # Ordering is part of the policy-target contract.  Sorting here would let
    # an action-token/target permutation pass while assigning probabilities to
    # the wrong actions.
    legal_ids_match = stored_ids.shape == recon_ids.shape and bool(
        np.array_equal(stored_ids, recon_ids)
    )

    worst_key = ""
    max_abs_diff = 0.0
    for key in _ROUNDTRIP_ENTITY_KEYS:
        if key not in stored_features or key not in feat["features"]:
            continue
        a = np.asarray(stored_features[key], dtype=np.float32)
        b = np.asarray(feat["features"][key], dtype=np.float32)
        if (
            key in {"event_tokens", "event_target_ids", "event_mask"}
            and a.ndim == b.ndim
            and a.shape[1:] == b.shape[1:]
            and a.shape[0] >= b.shape[0]
        ):
            # NPZ/memmap collation pads meaningful-history rows to the model's
            # inherited width (64), while the producer intentionally emitted
            # only event_history_limit rows (32). Authenticate the padding
            # before trimming; otherwise a hidden extra event could disappear.
            tail = a[b.shape[0] :]
            expected_fill = -1.0 if key == "event_target_ids" else 0.0
            if tail.size and not np.all(tail == expected_fill):
                worst_key = key
                max_abs_diff = float("inf")
                break
            a = a[: b.shape[0]]
        if a.shape != b.shape:
            worst_key = key
            max_abs_diff = float("inf")
            break
        diff = float(np.max(np.abs(a - b))) if a.size else 0.0
        if diff > max_abs_diff:
            max_abs_diff = diff
            worst_key = key
    for key in _ROUNDTRIP_ACTION_KEYS:
        if key not in stored_features or key not in feat["features"]:
            continue
        a = np.asarray(stored_features[key])[: stored_ids.size]
        b = np.asarray(feat["features"][key])
        if a.shape != b.shape:
            worst_key = key
            max_abs_diff = float("inf")
            break
        if np.issubdtype(a.dtype, np.floating) or np.issubdtype(b.dtype, np.floating):
            diff = float(
                np.max(np.abs(a.astype(np.float32) - b.astype(np.float32)))
            ) if a.size else 0.0
        else:
            diff = 0.0 if np.array_equal(a, b) else float("inf")
        if diff > max_abs_diff:
            max_abs_diff = diff
            worst_key = key

    stored_context = stored_features.get("legal_action_context")
    if stored_context is not None:
        a = np.asarray(stored_context)[: stored_ids.size].astype(np.float32)
        b = np.asarray(feat["context"]).astype(np.float32)
        if a.shape != b.shape:
            worst_key = "legal_action_context"
            max_abs_diff = float("inf")
        else:
            diff = float(np.max(np.abs(a - b))) if a.size else 0.0
            if diff > max_abs_diff:
                max_abs_diff = diff
                worst_key = "legal_action_context"
    ok = (
        legal_ids_match
        and phase_match
        and player_match
        and math_isfinite(max_abs_diff)
        and max_abs_diff <= fp16_atol
    )
    detail_parts = []
    if not phase_match:
        detail_parts.append(
            f"phase reconstructed={reconstructed_phase!r} stored={expected_phase!r}"
        )
    if not player_match:
        detail_parts.append(
            "player reconstructed="
            f"{reconstructed_player!r} stored={expected_player!r}"
        )
    return RoundTripResult(
        game_seed=seq.game_seed,
        decision_index=int(row_decision_index),
        ok=ok,
        legal_ids_match=legal_ids_match,
        max_abs_diff=max_abs_diff,
        worst_key=worst_key,
        detail="; ".join(detail_parts),
        phase_match=phase_match,
        player_match=player_match,
    )


def math_isfinite(x: float) -> bool:
    return x == x and x not in (float("inf"), float("-inf"))


def round_trip_shard_rows(
    shard_path: Path,
    *,
    max_rows: int = 32,
    correct_rust_chance_spectra: bool | None = None,
    colors: tuple[str, ...] = DEFAULT_COLORS,
    fp16_atol: float = 1e-2,
    seed: int = 0,
    row_indices: list[int] | tuple[int, ...] | np.ndarray | None = None,
    allow_omitted_automatic_transitions: bool = False,
) -> dict[str, Any]:
    """Round-trip a sample of rows from one shard, self-locating each game's scope.

    If `correct_rust_chance_spectra is None`, tries True then False and reports
    which flag reproduces the shard (resolves manifests that predate config
    provenance). Returns a summary dict with pass counts and per-row details.
    """
    from regret_common import load_shard

    shard_path = Path(shard_path)
    scope = shard_path.parent
    shard = load_shard(shard_path)
    n = int(np.asarray(shard["action_taken"]).shape[0])
    if row_indices is None:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)[: min(max_rows, n)]
        selection = {"kind": "seeded_sample", "seed": int(seed)}
    else:
        idx = np.asarray(row_indices, dtype=np.int64).reshape(-1)
        if idx.size == 0:
            raise ValueError("row_indices must not be empty")
        if np.any(idx < 0) or np.any(idx >= n):
            raise ValueError(f"row_indices out of range for {n}-row shard")
        if np.unique(idx).size != idx.size:
            raise ValueError("row_indices must be unique")
        selection = {"kind": "explicit_rows", "row_indices": idx.tolist()}

    flags = [correct_rust_chance_spectra] if correct_rust_chance_spectra is not None else [True, False]
    best: dict[str, Any] | None = None
    action_size = action_size_for_colors(tuple(colors))
    seeds = np.asarray(shard["game_seed"]).reshape(-1)
    didx = np.asarray(shard["decision_index"]).reshape(-1)

    # Pre-gather each needed game's action sequence once.
    for flag in flags:
        seq_cache: dict[int, GameActionSequence] = {}
        results: list[RoundTripResult] = []
        for i in idx:
            gseed = int(seeds[i])
            if gseed not in seq_cache:
                seq_cache[gseed] = gather_game_action_sequence(
                    scope,
                    gseed,
                    colors=colors,
                    allow_omitted_automatic_transitions=(
                        allow_omitted_automatic_transitions
                    ),
                )
            seq = seq_cache[gseed]
            stored_features = {
                key: shard[key][i]
                for key in (*_ROUNDTRIP_ENTITY_KEYS, *_ROUNDTRIP_ACTION_KEYS)
                if key in shard
            }
            if "legal_action_context" in shard:
                stored_features["legal_action_context"] = shard[
                    "legal_action_context"
                ][i]
            try:
                res = round_trip_row(
                    seq,
                    int(didx[i]),
                    stored_features,
                    shard["legal_action_ids"][i],
                    correct_rust_chance_spectra=bool(flag),
                    fp16_atol=fp16_atol,
                    action_size=action_size,
                )
            except Exception as error:  # noqa: BLE001 - record divergence, keep going.
                res = RoundTripResult(
                    game_seed=gseed,
                    decision_index=int(didx[i]),
                    ok=False,
                    legal_ids_match=False,
                    max_abs_diff=float("inf"),
                    worst_key="",
                    detail=repr(error),
                )
            results.append(res)
        passed = sum(1 for r in results if r.ok)
        summary = {
            "shard": str(shard_path),
            "shard_sha256": _file_sha256(shard_path),
            "selection": selection,
            "correct_rust_chance_spectra": bool(flag),
            "rows_checked": len(results),
            "rows_passed": passed,
            "pass_rate": passed / max(len(results), 1),
            "max_abs_diff_overall": max((r.max_abs_diff for r in results), default=0.0),
            "failures": [
                {
                    "game_seed": r.game_seed,
                    "decision_index": r.decision_index,
                    "legal_ids_match": r.legal_ids_match,
                    "phase_match": r.phase_match,
                    "player_match": r.player_match,
                    "max_abs_diff": r.max_abs_diff,
                    "worst_key": r.worst_key,
                    "detail": r.detail,
                }
                for r in results
                if not r.ok
            ][:10],
        }
        if best is None or summary["rows_passed"] > best["rows_passed"]:
            best = summary
        if summary["pass_rate"] == 1.0:
            break
    return best or {}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _locate_round_trip_row(
    scope: Path, *, game_seed: int, decision_index: int
) -> tuple[Path, int]:
    """Locate exactly one archived row, refusing absent or duplicate identity."""

    from regret_common import discover_shards, load_shard

    matches: list[tuple[Path, int]] = []
    for shard_path in discover_shards([Path(scope)]):
        shard = load_shard(shard_path)
        if "game_seed" not in shard or "decision_index" not in shard:
            continue
        seeds = np.asarray(shard["game_seed"]).reshape(-1)
        decisions = np.asarray(shard["decision_index"]).reshape(-1)
        rows = np.flatnonzero(
            (seeds == int(game_seed)) & (decisions == int(decision_index))
        )
        matches.extend((Path(shard_path), int(row)) for row in rows)
    if len(matches) != 1:
        raise ValueError(
            "round-trip row identity must resolve exactly once: "
            f"game_seed={game_seed} decision_index={decision_index} "
            f"matches={len(matches)}"
        )
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct a Rust game state from archived shard rows."
    )
    parser.add_argument("--scope", required=True, help="worker dir containing the game's shards")
    parser.add_argument("--game-seed", type=int, required=True)
    parser.add_argument("--decision-index", type=int, required=True)
    parser.add_argument("--colors", default="RED,BLUE")
    parser.add_argument(
        "--correct-rust-chance-spectra",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--round-trip",
        action="store_true",
        help="also gather the shard row at this decision and verify featurisation match",
    )
    parser.add_argument(
        "--allow-omitted-automatic-transitions",
        action="store_true",
        help=(
            "admit sparse decision indices only when replay proves every gap "
            "had exactly one legal action"
        ),
    )
    args = parser.parse_args()

    colors = tuple(c.strip() for c in args.colors.split(","))
    seq = gather_game_action_sequence(
        Path(args.scope),
        args.game_seed,
        colors=colors,
        allow_omitted_automatic_transitions=(
            args.allow_omitted_automatic_transitions
        ),
    )
    game = reconstruct_state_from_sequence(
        seq,
        args.decision_index,
        correct_rust_chance_spectra=args.correct_rust_chance_spectra,
    )
    snapshot = json.loads(game.json_snapshot())
    legal = list(game.playable_action_indices(list(colors), None))
    output = {
        "game_seed": args.game_seed,
        "decision_index": args.decision_index,
        "total_decisions": len(seq),
        "acting_color": str(game.current_color()),
        "phase": str(snapshot.get("current_prompt", "")),
        "winning_color": str(game.winning_color()),
        "n_legal": len(legal),
    }
    if args.round_trip:
        shard_path, row_index = _locate_round_trip_row(
            Path(args.scope),
            game_seed=args.game_seed,
            decision_index=args.decision_index,
        )
        round_trip = round_trip_shard_rows(
            shard_path,
            correct_rust_chance_spectra=args.correct_rust_chance_spectra,
            colors=colors,
            fp16_atol=1e-2,
            row_indices=[row_index],
            allow_omitted_automatic_transitions=(
                args.allow_omitted_automatic_transitions
            ),
        )
        output["round_trip"] = round_trip
        print(json.dumps(output, indent=2, sort_keys=True))
        if round_trip.get("rows_checked") != 1 or round_trip.get("rows_passed") != 1:
            raise SystemExit("round-trip verification FAILED")
        return
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
