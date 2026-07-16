"""Gumbel self-play data-generation driver.

Plays full 2-player games where BOTH seats use `GumbelChanceMCTS` (see
`catan_zero.search.gumbel_chance_mcts`) with a shared evaluator, recording
entity-token training rows in the same schema `tools/train_bc.py` already
consumes (`BASE_KEYS`/`ENTITY_KEYS` in `tools/convert_teacher_to_entity_tokens.py`,
row-building pattern in `tools/generate_rust_mcts_reanalysis.py`).

This module owns the game-playing/row-building logic and a small
schema-compatible shard writer (`GumbelShardWriter`). The CLI orchestration
(argument parsing, multiprocessing, manifest merging) lives in
`tools/generate_gumbel_selfplay_data.py`, which imports from here.

Note: the codebase reference `tools/generate_dagger_data.py` (mentioned as the
source of the "features before action" ordering fix) does not exist in this
checkout; `tools/generate_rust_mcts_reanalysis.py` already implements the same
correct ordering (build entity features from the live pre-action state, THEN
apply the selected action) and is used as the verified reference instead.

Opponent pool (anti-forgetting, H2): `run_worker_games`/`play_one_game` optionally
play a deterministic fraction of games CHAMPION-vs-ARCHIVED-OPPONENT instead of
pure mirror self-play, reusing `catan_zero.rl.flywheel.opponent_pool`'s hash-based
`choose_opponent` (see `read_opponent_pool_manifest`/`OpponentPoolRuntime` below).
Default is `pool_assignment=None`/`opponent_pool=None` everywhere -- exact prior
behavior, byte-identical shard schema -- so this is purely additive.

Opponent MIX (CAT-54): `run_worker_games` additionally accepts `opponent_mix=` (a
`MixRuntime`), an arbitrary-category generalization of the H2 binary pool built on
`catan_zero.rl.flywheel.opponent_mix`'s `choose_mix_opponent` (e.g. the adopted
75% producer self-play / 10% previous+public champion / 5% older champion / 5%
hard-experimental split). It resolves to the SAME `PoolGameAssignment` dataclass
`play_one_game` already consumes -- `opponent_pool`/`opponent_mix` are mutually
exclusive, and `play_one_game` itself needs no changes to support either. Default
`opponent_mix=None` -- again exactly today's behavior when omitted.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import io
import json
import math
import os
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from catan_zero.deduction_tracker import DEDUCTION_FEATURES_KEY
from catan_zero.rl.action_mask import ActionCatalog
from catan_zero.rl.decision_taxonomy import (
    AUTOMATIC_TRANSITION,
    DECISION_TAXONOMY_SCHEMA_VERSION,
    classify_public_decision,
    decision_requires_full_search,
)
from catan_zero.rl.aux_subgoal_targets import (
    AUX_SUBGOAL_TARGET_SEMANTIC,
    AUX_SUBGOAL_TARGET_VERSION,
    AUX_SUBGOAL_TARGET_VERSION_KEY,
    AUX_TARGET_KEYS,
    AUX_VP_HORIZON,
    rust_aux_state_from_snapshot,
    rust_hex_id_by_coordinate,
    rust_robber_hex_of_action,
    rust_settlement_node_of_action,
    trajectory_targets,
)
from catan_zero.rl.flywheel import ChampionRef, OpponentPolicy, choose_opponent
from catan_zero.rl.flywheel.opponent_mix import OpponentMixConfig, choose_mix_opponent
from catan_zero.rl.target_reliability import (
    TARGET_RELIABILITY_COLUMNS,
    TARGET_RELIABILITY_SCHEMA,
    TARGET_RELIABILITY_VERSION,
    duplicate_search_reliability_fields,
    target_reliability_contract,
    target_reliability_root_seed,
    target_reliability_root_selected,
    unaudited_target_reliability_fields,
)
from catan_zero.search.gumbel_chance_mcts import (
    GumbelChanceMCTS,
    GumbelChanceMCTSConfig,
    RustEvaluator,
    SearchResult,
    buy_development_card_real_outcomes,
    is_move_robber_with_victim,
    move_robber_victim_outcome_weights,
)
from catan_zero.search import gumbel_chance_mcts as _gumbel_chance_mcts
from catan_zero.search.native_gumbel_mcts import create_gumbel_search
from catan_zero.search.neural_rust_mcts import (
    RUST_ENTITY_ADAPTER_VERSION,
    rust_action_context_batch,
    rust_game_to_entity_batch,
    rust_policy_action_ids,
)

__all__ = [
    "COLORS",
    "PLAYER_NAMES",
    "ACTION_MASK_VERSION",
    "GumbelSelfPlayConfig",
    "DecisionRecord",
    "GameRecord",
    "GumbelShardWriter",
    "SEARCH_EVIDENCE_SCHEMA",
    "SEARCH_EVIDENCE_VERSION",
    "TARGET_RELIABILITY_SCHEMA",
    "TARGET_RELIABILITY_VERSION",
    "search_evidence_for_row",
    "WorkerProgress",
    "PROGRESS_FILENAME",
    "PoolGameAssignment",
    "OpponentPoolRuntime",
    "read_opponent_pool_manifest",
    "MixRuntime",
    "play_one_game",
    "run_worker_games",
    "action_size_for_evaluator",
]

# Written to `<out_dir>/progress.json` by `run_worker_games` (see
# `WorkerProgress`); a durable, incremental resume marker that lets a
# same-run_id retry after a preemption pick up where it left off instead of
# replaying (or wiping) already-flushed games.
PROGRESS_FILENAME = "progress.json"
WORKER_PROGRESS_RESUME_CONTRACT_VERSION = 3

COLORS = ("RED", "BLUE")
PLAYER_NAMES = ("BLUE", "RED", "ORANGE", "WHITE")
ACTION_MASK_VERSION = "colonist-multiagent-v1"
TEACHER_NAME = "gumbel_self_play"
TARGET_SCORE_SOURCE = "gumbel_mcts_visit_q"
SEARCH_EVIDENCE_SCHEMA = "gumbel_root_search_evidence_v1"
SEARCH_EVIDENCE_VERSION = 1
# Retained as historical evidence semantics for contract/inventory tools that
# must decode older shards. Current production generation keeps n_fast rows
# policy-inactive; these constants do not re-enable fast-search supervision.
FAST_SEARCH_POLICY_WEIGHT_MAX = 0.25
FAST_SEARCH_POLICY_REFERENCE_SIMULATIONS = 128
# Search targets need provenance that is independent from observation masking.
# ``public_observation=True`` only constrains neural-network features; it does
# not prove that the planner's cloned world state was information-safe.
TARGET_INFORMATION_REGIME_AUTHORITATIVE = "authoritative_hidden_state_search_v1"
TARGET_INFORMATION_REGIME_PUBLIC = "public_conservation_pimc_v1"
TARGET_INFORMATION_REGIME_PUBLIC_COHERENT = "public_belief_single_tree_v1"
TARGET_INFORMATION_REGIMES = frozenset(
    {
        TARGET_INFORMATION_REGIME_AUTHORITATIVE,
        TARGET_INFORMATION_REGIME_PUBLIC,
        TARGET_INFORMATION_REGIME_PUBLIC_COHERENT,
    }
)

# Kept in sync with `tools/convert_teacher_to_entity_tokens.py`'s BASE_KEYS /
# ENTITY_KEYS. Do not diverge without updating both -- these are what makes
# the shards this module writes compatible with `tools/train_bc.py`.
BASE_KEYS = (
    "obs",
    "legal_action_ids",
    "legal_action_context",
    "action_taken",
    "target_policy",
    "target_scores",
    "target_policy_mask",
    "target_scores_mask",
    "target_score_source",
    "target_information_regime",
    "game_seed",
    "teacher_name",
    "player",
    "seat",
    "phase",
    "decision_index",
    "winner",
    "terminated",
    "truncated",
    "final_public_vps",
    "has_final_public_vps",
    "final_actual_vps",
    "has_final_actual_vps",
    "action_mask_version",
    "policy_weight_multiplier",
    "value_weight_multiplier",
    "adapter_version",
)

ENTITY_KEYS = (
    "hex_tokens",
    "hex_vertex_ids",
    "hex_edge_ids",
    "vertex_tokens",
    "edge_tokens",
    "edge_vertex_ids",
    "player_tokens",
    DEDUCTION_FEATURES_KEY,
    "global_tokens",
    "legal_action_tokens",
    "legal_action_target_ids",
    "event_tokens",
    "event_target_ids",
    "hex_mask",
    "vertex_mask",
    "edge_mask",
    "player_mask",
    "legal_action_mask",
    "event_mask",
)

# Extra columns beyond the shared schema above. The CAT-100 aux fields are
# consumed by train_bc; the remaining analysis/provenance fields stay optional
# and forward-compatible.
EXTRA_KEYS = (
    # Scalar search value at the root, from the acting player's perspective.
    # This is optional at load time so every historical shard remains valid;
    # new searched shards persist it alongside explicit finite-value coverage.
    # `target_information_regime` remains the authority for whether a masked
    # learner may consume it (authoritative-hidden-state targets fail closed).
    "root_value",
    "root_value_mask",
    "afterstate_target",
    "afterstate_target_mask",
    "used_full_search",
    "simulations_used",
    "is_forced",
    "decision_class",
    "decision_taxonomy_schema",
    "prior_policy",
    # CAT-100: realized-trajectory auxiliary targets. These are present on
    # every production row; unavailable targets use NaN (binary/scalar) or -1
    # (categorical), matching train_bc's per-head masks.
    *AUX_TARGET_KEYS,
    # Per-row semantic version.  Historical shards missing this field are
    # normalized as version 0 and are ineligible for auxiliary loss, while
    # remaining fully usable for policy/value training.
    AUX_SUBGOAL_TARGET_VERSION_KEY,
    # Opponent-pool provenance (H2). Only present on rows from a run where
    # --opponent-pool-manifest was set (see `play_one_game`'s `pool_assignment`);
    # absent entirely otherwise, so default (pool-disabled) shard schema is
    # unchanged. `opponent_version` is -1 on non-pool games within a pool-
    # enabled run (mirror self-play against the champion).
    "is_pool_game",
    "opponent_version",
    # Opponent-MIX provenance (CAT-54): the named mix category (e.g.
    # "producer_self_play"/"hard_experimental") and the chosen opponent
    # checkpoint's md5, so later per-opponent telemetry (win rate/KL/entropy/
    # value calibration, tracked separately per opponent per the ticket) can
    # be computed straight from the shard without re-deriving identity from
    # `opponent_version` alone. Only present on rows from a run where
    # --opponent-mix-manifest was set; `opponent_tag`/`opponent_checkpoint_md5`
    # are `""` on producer-self-play games within a mix-enabled run.
    "opponent_tag",
    "opponent_checkpoint_md5",
    # Exploiter-lane provenance (CAT-56): the EXTERNAL Catanatron engine name
    # (e.g. "catanatron_value"/"catanatron_ab3") on rows generated in cross-engine
    # lockstep against that bot, distinct from `opponent_tag` (the mix category
    # name). Present ONLY on exploiter-lane rows (see
    # `exploiter_lockstep.play_one_exploiter_game`); absent on every self-play /
    # neural-pool / neural-mix row, so those shard schemas are unchanged.
    "opponent_type",
    # Duplicate coherent-n128 audit evidence.  A reliability-enabled producer
    # writes these typed scalars on every row; unaudited rows are explicit and
    # neutral.  A producer with audit fraction zero omits the columns entirely,
    # preserving historical shard bytes.
    *TARGET_RELIABILITY_COLUMNS,
)


@dataclass(frozen=True, slots=True)
class GumbelSelfPlayConfig:
    colors: tuple[str, ...] = COLORS
    map_kind: str | None = None
    track: str = "2p_no_trade"
    vps_to_win: int = 10
    obs_width: int = 806
    # Adopted cap-600 policy (2026-07-04): raised from 300 to cut the ~87.5%
    # truncation rate observed at the 300 cap (task #53). This dataclass is the
    # SINGLE SOURCE OF TRUTH for the default -- every tools/ CLI that exposes
    # --max-decisions must match it (enforced by tests/test_cli_config_drift.py).
    max_decisions: int = 600
    # Temperature schedule: T=temperature_high for the first
    # round(max_decisions * temperature_move_fraction) schedule steps, then
    # T=temperature_low. ``temperature_clock`` defines whether a step is every
    # engine prompt or only a non-forced choice. The absolute cutoff is resolved
    # against the configured cap, matching the AlphaZero/KataGo move-count form.
    #
    # This fraction is COUPLED to max_decisions: it is a fraction OF THE CAP, so
    # the absolute count of temperature moves is round(cap * fraction). The
    # adopted policy holds that absolute count at 45 (0.075 * 600 == 45, matching
    # the 0.15 * 300 == 45 it replaced). Because the coupling has silently
    # mis-fired twice, generate_gumbel_selfplay_data.py now prefers the absolute
    # --temperature-decisions flag and derives this fraction from it.
    temperature_move_fraction: float = 0.075
    temperature_high: float = 1.0
    temperature_low: float = 0.0
    # Historical ``prompt`` counts every engine prompt, including sole ROLL
    # and END_TURN plumbing. ``nonforced_choice`` counts only positions with
    # more than one legal action, making the exploration window invariant to
    # dice-seven/discard prompt density.
    temperature_clock: str = "prompt"
    # CAT-12 (roadmap R8 diversity-strangulation / queue #16): optional THIRD stage
    # extending a small nonzero temperature past the opening cutoff instead of
    # dropping straight to `temperature_low` (argmax). When
    # `late_temperature_move_fraction` is None (default), behavior is UNCHANGED --
    # two-stage schedule exactly as above, a pure no-op. When set, decisions in
    # [opening cutoff, late cutoff) use `late_temperature` instead of
    # `temperature_low`; `late_temperature_move_fraction` is resolved against
    # `max_decisions` the same way `temperature_move_fraction` is, and is clamped to
    # be no earlier than the opening cutoff (a smaller late fraction than the opening
    # one degenerates to the plain two-stage schedule, never to a negative window).
    late_temperature_move_fraction: float | None = None
    late_temperature: float = 0.0
    # Mirror of `GumbelChanceMCTSConfig.correct_rust_chance_spectra`: the live
    # game's own chance resolution (`_apply_selected_action`) has the same
    # verified Rust engine bugs (A19/A20) as the search's internal simulation,
    # so it needs the identical correction to keep recorded trajectories
    # (and thus training data) valid. Set False to A/B against a future
    # corrected Rust wheel.
    correct_rust_chance_spectra: bool = True
    # Next-wave input surface: filter the existing event-token stream to the
    # strategic public taxonomy and retain at most the latest 32 events.
    # Defaults preserve legacy shard shape/contents.
    meaningful_public_history: bool = False
    event_history_limit: int = 64
    # One-action UI/chance prompts are strategically real trajectory states.
    # Record them as value-only rows by default: _build_decision_row gives
    # them zero policy authority and full terminal-outcome value authority.
    # False remains an explicit backward-compatible opt-out for replaying
    # historical generation recipes that omitted these transitions.
    record_automatic_transitions: bool = True
    # Deterministic fraction of eligible policy-active coherent n128 roots that
    # receive a second, independently seeded search.  The duplicate produces
    # evidence only and never drives the live trajectory.  Zero is an exact
    # schema/compute no-op.
    target_reliability_audit_fraction: float = 0.0
    target_reliability_audit_seed: int = 0


@dataclass(frozen=True, slots=True)
class PoolGameAssignment:
    """Resolved per-game opponent decision for `play_one_game` (H2 binary pool,
    generalized by CAT-54's N-way mix -- both `run_worker_games` code paths
    build this same dataclass, so `play_one_game` needs no branching between
    them).

    ``is_pool`` False => mirror self-play *within a pool/mix-enabled run*: both
    seats are the champion (`opponent_color`/`opponent_evaluator` unused), but
    the row still carries `is_pool_game=False`/`opponent_version=-1` for
    schema consistency across the run's shards. `champion_color` is still
    meaningful (used to grade the anti-forgetting win-rate telemetry the same
    way for pool and mirror games).

    ``tag``/``opponent_md5`` (CAT-54): the named mix category (e.g.
    "hard_experimental") and the opponent checkpoint's md5, for per-opponent
    shard tagging. `_build_decision_row` only stamps the `opponent_tag`/
    `opponent_checkpoint_md5` row columns when `tag` is non-empty, so the
    H2 binary-pool path (`run_worker_games`'s `opponent_pool=` argument),
    which leaves `tag`/`opponent_md5` at their `""` default, keeps its EXACT
    prior row schema (`is_pool_game`/`opponent_version` only); only the new
    `opponent_mix=` mix path sets a real `tag`.
    """

    is_pool: bool
    champion_color: str
    opponent_color: str | None
    opponent_version: int
    opponent_evaluator: "RustEvaluator | None"
    tag: str = ""
    opponent_md5: str = ""


@dataclass(frozen=True, slots=True)
class OpponentPoolRuntime:
    """Per-worker opponent-pool state: the archived-opponent policy/archive
    (parsed once, pure stdlib -- see `read_opponent_pool_manifest`) plus a
    checkpoint-path -> evaluator factory the caller constructs (so this module
    stays free of the neural-evaluator/torch import; see
    `tools/generate_gumbel_selfplay_data.py`'s `_run_worker`). `run_worker_games`
    calls `evaluator_factory` lazily and caches the result by checkpoint path
    for the life of the worker process (workers play many games; a checkpoint
    is loaded onto the device at most once)."""

    policy: OpponentPolicy
    champion: ChampionRef
    archive: tuple[ChampionRef, ...]
    evaluator_factory: "Callable[[str], RustEvaluator]"


@dataclass(frozen=True, slots=True)
class MixRuntime:
    """Per-worker opponent-MIX state (CAT-54's N-way generalization of
    `OpponentPoolRuntime`): a resolved, arbitrary-category `OpponentMixConfig`
    (see `catan_zero.rl.flywheel.opponent_mix`) plus the same
    checkpoint-path -> evaluator factory contract `OpponentPoolRuntime` uses
    (kept torch-free here; `tools/generate_gumbel_selfplay_data.py`'s
    `_run_worker` supplies the real one). `run_worker_games` caches
    `evaluator_factory`'s result by checkpoint path for the worker's lifetime,
    identically to the H2 binary-pool path.

    MEMORY CONSTRAINT (checked against `EntityGraphRustEvaluator`/
    `BatchedEntityGraphRustEvaluator`/`EntityGraphPolicy.load`, 2026-07-08):
    there is no shared/global model state to worry about -- every loaded
    checkpoint is its own independent `EntityGraphPolicy` instance holding its
    own weights on `device`, so the producer net and every DISTINCT opponent
    checkpoint a worker ever samples coexist correctly in the same process.
    The real cost is additive GPU/CPU memory and forward-pass compute: a
    worker holds one resident model per distinct checkpoint it has sampled so
    far (the evaluator cache above never evicts), so a mix with several
    large "older_champion"/"hard_experimental" checkpoints can pin
    `n_distinct_checkpoints_sampled x model_size` of device memory per
    worker, on top of the producer's own. Keep `--workers` and the pool's
    distinct-checkpoint count sized to fit `--device`'s memory budget; there
    is no code-level ceiling on how many checkpoints a single worker will
    load."""

    config: OpponentMixConfig
    evaluator_factory: "Callable[[str], RustEvaluator]"


def read_opponent_pool_manifest(
    path: str | Path,
) -> tuple[OpponentPolicy, ChampionRef, tuple[ChampionRef, ...]]:
    """Parse the `--opponent-pool-manifest` JSON (pure stdlib; no torch, safe to
    call in the main process for fail-fast validation before workers spawn):

        {"opponents": [{"checkpoint": <path>, "version": <int>}, ...],
         "pool_fraction": <float in [0,1]>}

    Returns `(policy, champion_sentinel, archive)` ready for
    `opponent_pool.choose_opponent`. `archive` is oldest-first by version (the
    ordering `choose_opponent` requires). `champion_sentinel` is a synthetic
    `ChampionRef` one version newer than the newest listed opponent -- this
    manifest format (unlike `checkpoint_registry`'s live archive) doesn't carry
    the real champion's own version number, so the sentinel exists purely to
    make every listed opponent pass `choose_opponent`'s
    "strictly older than the champion" eligibility filter; its `path` is never
    read (the real champion checkpoint is the CLI's own `--checkpoint`).
    """
    data = json.loads(Path(path).read_text())
    raw_opponents = list(data.get("opponents", []))
    if not raw_opponents:
        raise ValueError(
            f"opponent-pool manifest {path} has no 'opponents' entries "
            "(pass --opponent-pool-manifest only when you have archived "
            "checkpoints to sample; omit the flag entirely for pure mirror "
            "self-play)"
        )
    archive = tuple(
        sorted(
            (
                ChampionRef(
                    version=int(entry["version"]),
                    path=str(entry["checkpoint"]),
                    promoted_at=0.0,
                )
                for entry in raw_opponents
            ),
            key=lambda ref: ref.version,
        )
    )
    champion = ChampionRef(
        version=max(ref.version for ref in archive) + 1, path="", promoted_at=0.0
    )
    policy = OpponentPolicy(pool_fraction=float(data.get("pool_fraction", 0.0)))
    return policy, champion, archive


def _pool_champion_plays_first_seat(game_index: int) -> bool:
    """Deterministic per-game color-balance bit: which of `config.colors[0]`/
    `[1]` the champion occupies on a pool game. Hashed from `game_index` with a
    salt distinct from `opponent_pool.choose_opponent`'s own "pool_gate"/
    "pool_pick" draws so alternating seats doesn't perturb which games are
    selected as pool games in the first place -- same resume-safety rationale
    as `opponent_pool._u01` (not a global RNG, survives crash-and-resume)."""
    digest = hashlib.sha256(f"pool_seat:{int(game_index)}".encode()).digest()
    return (digest[0] & 1) == 0


@dataclass(slots=True)
class DecisionRecord:
    row: dict[str, Any]
    features: dict[str, np.ndarray]


@dataclass(slots=True)
class GameRecord:
    game_seed: int
    game_index: int
    decisions: list[DecisionRecord]
    terminal: bool
    truncated: bool
    winner: str
    total_decisions: int
    forced_decisions: int
    simulations_used_total: int
    wall_time_sec: float
    error: str | None = None
    # Exploiter lane (CAT-56): set by `exploiter_lockstep.play_one_exploiter_game`
    # when the Rust/Python engines diverged on rules semantics (rows are dropped;
    # `decisions=[]`). `divergence_topic` buckets it (longest-road / buildable-edge
    # / unclassified) for telemetry. Always False/"none" for ordinary self-play,
    # pool, and neural-mix games, which never run a second engine.
    engine_divergence: bool = False
    divergence_topic: str = "none"


def action_size_for_evaluator(evaluator: RustEvaluator, colors: tuple[str, ...]) -> int:
    """Resolve the flat policy action-space size for an evaluator.

    Neural evaluators (`EntityGraphRustEvaluator`/`BatchedEntityGraphRustEvaluator`)
    expose `.policy.action_size`. `HeuristicRustEvaluator` (and any other
    evaluator without a `.policy`) has no such notion, so fall back to the
    same `ActionCatalog` size used to build that mapping everywhere else.
    """
    policy = getattr(evaluator, "policy", None)
    if policy is not None and hasattr(policy, "action_size"):
        return int(policy.action_size)
    return int(ActionCatalog(colors).size)


def entity_adapter_for_evaluator(evaluator: RustEvaluator) -> str:
    """Return the feature contract actually selected by the evaluator."""

    version = getattr(
        getattr(evaluator, "config", None),
        "entity_feature_adapter_version",
        None,
    )
    return str(version or RUST_ENTITY_ADAPTER_VERSION)


def _temperature_for_decision(
    decision_index: int,
    *,
    config: GumbelSelfPlayConfig,
    eval_override: bool,
    nonforced_choice_index: int | None = None,
) -> float:
    if eval_override:
        return float(config.temperature_low)
    clock = str(config.temperature_clock)
    if clock == "prompt":
        schedule_index = int(decision_index)
    elif clock == "nonforced_choice":
        if nonforced_choice_index is None:
            raise ValueError(
                "temperature_clock='nonforced_choice' requires a choice index"
            )
        schedule_index = int(nonforced_choice_index)
    else:
        raise ValueError(
            f"temperature_clock must be 'prompt' or 'nonforced_choice', got {clock!r}"
        )
    cutoff = max(
        1, round(float(config.max_decisions) * float(config.temperature_move_fraction))
    )
    if schedule_index < cutoff:
        return float(config.temperature_high)
    if config.late_temperature_move_fraction is not None:
        late_cutoff = max(
            cutoff,
            round(
                float(config.max_decisions)
                * float(config.late_temperature_move_fraction)
            ),
        )
        if schedule_index < late_cutoff:
            return float(config.late_temperature)
    return float(config.temperature_low)


def _apply_selected_action(
    game: Any,
    action_index: int,
    *,
    colors: tuple[str, ...],
    rng: random.Random,
    correct_rust_chance_spectra: bool = True,
    action_json: Any | None = None,
) -> Any:
    """Advance the live game by the selected action, sampling chance ourselves.

    Mirrors `generate_rust_mcts_reanalysis.py`'s `_apply_action_with_sampled_chance`:
    `execute_action_index` lets the Rust engine's own (uncontrolled) RNG decide
    chance outcomes, so instead we read `spectrum_json` and sample from our own
    seeded `rng` for full, game_seed-reproducible determinism.

    When `correct_rust_chance_spectra` is True (the default), MOVE_ROBBER-with-
    victim and BUY_DEVELOPMENT_CARD use the same corrected weights as
    `GumbelChanceMCTS`'s internal search (verified Rust engine bugs A19/A20) --
    otherwise the recorded game trajectory would still be wrong even though
    the search itself now reasons about these chance nodes correctly.
    """
    if action_json is None:
        ids = [
            int(action) for action in game.playable_action_indices(list(colors), None)
        ]
        actions = json.loads(game.playable_actions_json())
        action_by_id = {action_id: action for action_id, action in zip(ids, actions)}
        action_json = action_by_id.get(int(action_index))
    if action_json is None:
        raise RuntimeError(f"selected action {action_index} is not legal")

    if correct_rust_chance_spectra and is_move_robber_with_victim(action_json):
        # `candidates` is `None` when shape-detection finds the native
        # spectrum already correctly hand-weighted (fixed wheel) -- falling
        # through to the raw spectrum path below is exactly right for that
        # case (native pass-through, zero extra work). It's `[]` only in the
        # defensive/should-not-happen case of no real steal outcome at all,
        # which also falls through to the same raw-spectrum fallback.
        candidates = move_robber_victim_outcome_weights(game, action_json)
        if candidates:
            total = sum(weight for _index, weight, _game in candidates)
            outcomes = tuple(
                (index, weight / total) for index, weight, _game in candidates
            )
            chosen_index = _sample_from_outcomes(rng, outcomes)
            return next(
                candidate_game
                for index, _weight, candidate_game in candidates
                if index == chosen_index
            )

    if (
        correct_rust_chance_spectra
        and _action_type_of(action_json) == "BUY_DEVELOPMENT_CARD"
    ):
        real_candidates = buy_development_card_real_outcomes(game, action_json)
        if real_candidates:
            total = sum(probability for _index, probability, _game in real_candidates)
            normalized = tuple(
                (index, probability / total)
                for index, probability, _game in real_candidates
            )
            chosen_index = _sample_from_outcomes(rng, normalized)
            return next(
                candidate_game
                for index, _probability, candidate_game in real_candidates
                if index == chosen_index
            )
        # Defensive: no real outcome at all -- fall through to the raw
        # spectrum path below rather than crashing the game loop.

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


def _action_type_of(action_json: Any) -> str:
    if isinstance(action_json, (list, tuple)) and len(action_json) > 1:
        return str(action_json[1])
    return ""


def _sample_from_outcomes(
    rng: random.Random, outcomes: tuple[tuple[int, float], ...]
) -> int:
    if len(outcomes) == 1:
        return outcomes[0][0]
    draw = rng.random()
    cumulative = 0.0
    for outcome_index, probability in outcomes:
        cumulative += probability
        if draw <= cumulative:
            return outcome_index
    return outcomes[-1][0]


def _actual_victory_points(player_state: dict[str, Any]) -> int:
    return int(
        player_state.get("actual_victory_points", player_state.get("victory_points", 0))
        or 0
    )


def _game_outcome_fields(
    game: Any, *, terminal: bool, colors: tuple[str, ...]
) -> dict[str, Any]:
    winner = game.winning_color()
    public_vps: dict[str, int] = {}
    actual_vps: dict[str, int] = {}
    for color in colors:
        state = json.loads(game.player_state_json(color))
        public_vps[color] = int(state.get("victory_points", 0) or 0)
        actual_vps[color] = _actual_victory_points(state)
    return {
        "winner": str(winner) if terminal and winner is not None else "",
        "terminated": bool(terminal),
        "truncated": not bool(terminal),
        "final_public_vps": np.asarray(
            [int(public_vps.get(name, 0)) for name in PLAYER_NAMES], dtype=np.int16
        ),
        "has_final_public_vps": bool(terminal),
        "final_actual_vps": np.asarray(
            [int(actual_vps.get(name, 0)) for name in PLAYER_NAMES], dtype=np.int16
        ),
        "has_final_actual_vps": bool(terminal),
    }


def _build_decision_row(
    game: Any,
    *,
    result: SearchResult,
    action_size: int,
    colors: tuple[str, ...],
    game_seed: int,
    decision_index: int,
    obs_width: int,
    is_pool_game: bool | None = None,
    opponent_version: int | None = None,
    opponent_tag: str = "",
    opponent_checkpoint_md5: str = "",
    snapshot: dict[str, Any] | None = None,
    action_by_id: dict[int, Any] | None = None,
    target_information_regime: str = TARGET_INFORMATION_REGIME_AUTHORITATIVE,
    meaningful_public_history: bool = False,
    event_history_limit: int = 64,
    decision_class: str = "normal_choice",
    entity_feature_adapter_version: str = RUST_ENTITY_ADAPTER_VERSION,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if target_information_regime not in TARGET_INFORMATION_REGIMES:
        raise ValueError(
            "unsupported target_information_regime "
            f"{target_information_regime!r}; expected one of "
            f"{sorted(TARGET_INFORMATION_REGIMES)}"
        )
    # legal_rust is exactly the key set of any of these dicts by
    # SearchResult's contract (covers ALL legal root actions).
    legal_rust = tuple(sorted(result.improved_policy.keys()))
    # Forced (single-legal-action) decisions carry no policy-improvement
    # signal. Historical full mode may report used_full_search=True after
    # evaluating a value; trajectory_only deliberately reports False and does
    # no neural work. Both must receive zero policy weight.
    is_forced = len(legal_rust) <= 1
    acting_color = str(game.current_color())
    mapped = rust_policy_action_ids(
        game, legal_rust, colors=colors, action_size=action_size
    )
    # Fetch the snapshot + rust-action-id -> raw-json mapping ONCE and thread
    # them into both batch calls below (they'd otherwise each independently
    # re-fetch json_snapshot/playable_action_indices/playable_actions_json on
    # the same, unchanged game state -- see `_resolve_entity_adapter`'s
    # docstring in neural_rust_mcts.py).
    if snapshot is None:
        snapshot = json.loads(game.json_snapshot())
    if action_by_id is None:
        action_ids = [
            int(action) for action in game.playable_action_indices(list(colors), None)
        ]
        raw_actions = json.loads(game.playable_actions_json())
        action_by_id = {
            action_id: raw for action_id, raw in zip(action_ids, raw_actions)
        }
    entity = rust_game_to_entity_batch(
        game,
        legal_rust,
        actor=acting_color,
        colors=colors,
        action_size=action_size,
        policy_action_ids=mapped,
        snapshot=snapshot,
        action_by_id=action_by_id,
        # Persist the same public-information view used by online MCTS.  The
        # training loader may mask again as a defence in depth, but shards
        # must be safe and self-describing on their own: a consumer that does
        # not happen to pass ``--mask-hidden-info`` must never see opponents'
        # resource composition, hidden development cards, or actual VP.
        public_observation=True,
        meaningful_public_history=meaningful_public_history,
        history_limit=event_history_limit,
        entity_feature_adapter_version=entity_feature_adapter_version,
    )
    features = {key: value[0] for key, value in entity.items()}
    context = rust_action_context_batch(
        game,
        legal_rust,
        actor=acting_color,
        colors=colors,
        action_size=action_size,
        policy_action_ids=mapped,
        snapshot=snapshot,
        action_by_id=action_by_id,
        public_observation=True,
        entity_feature_adapter_version=entity_feature_adapter_version,
    )[0]

    # F4: fp32, not fp16. improved_policy assigns real (non-zero, non-one-hot)
    # mass to every legal action via completion, so fp16's ~1e-3 relative
    # precision was silently flushing small-but-real probabilities to zero
    # (worst at 54-action placement, where the completed distribution is
    # widest) -- that in turn tripped train_bc.py's soft-target coverage
    # gate into treating a real, fully-covered soft target as under-covered
    # and falling back to one-hot hard CE. Mostly cured by F1 (targets are
    # no longer near-one-hot to begin with), but fp32 removes this failure
    # mode independent of how sharp the target actually is.
    target_policy = np.asarray(
        [float(result.improved_policy.get(int(action), 0.0)) for action in legal_rust],
        dtype=np.float32,
    )
    # The mask records which legal actions received an authenticated teacher
    # label, not which labels happen to carry positive probability. SearchResult
    # covers every legal root action by contract, so an exact zero (for example
    # after policy-target pruning) is still a real label. Marking it missing
    # makes the learner's coverage gate silently fall back to action_taken.
    target_policy_mask = np.ones(target_policy.shape, dtype=np.bool_)
    # Root priors (pre-search network policy), same legal_rust ordering as
    # target_policy -- persisted so KL(improved_policy || prior) is
    # computable directly from shards without re-running the evaluator.
    prior_policy = np.asarray(
        [float(result.priors.get(int(action), 0.0)) for action in legal_rust],
        dtype=np.float16,
    )
    target_scores = np.asarray(
        [float(result.q_values.get(int(action), np.nan)) for action in legal_rust],
        dtype=np.float32,
    )
    target_scores_mask = np.isfinite(target_scores)
    afterstate_target = np.asarray(
        [
            float(result.afterstate_values.get(int(action), np.nan))
            for action in legal_rust
        ],
        dtype=np.float32,
    )
    afterstate_target_mask = np.isfinite(afterstate_target)
    best_rust = int(result.selected_action)
    best_policy = mapped[legal_rust.index(best_rust)]

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
        # NOTE: "seat" must index into PLAYER_NAMES ("BLUE","RED","ORANGE","WHITE")
        # order, NOT `colors` order -- that's the same order final_public_vps/
        # final_actual_vps are built in (see _game_outcome_fields below), and
        # matches the codebase-wide convention (generate_teacher_data.py's
        # _seat_index). Using colors.index() here would point every row's
        # seat at the WRONG player's VP slot whenever colors != PLAYER_NAMES
        # order (COLORS=("RED","BLUE") gives RED=0, but PLAYER_NAMES gives
        # BLUE=0) -- a silent, total seat/VP swap.
        "seat": np.int8(PLAYER_NAMES.index(acting_color)),
        "phase": str(snapshot.get("current_prompt", "")),
        "decision_index": np.int32(decision_index),
        # Outcome fields are placeholders here; filled in by the caller once
        # the game ends via `_game_outcome_fields` (winner is not known yet).
        "winner": "",
        "terminated": False,
        "truncated": False,
        "final_public_vps": np.zeros(len(PLAYER_NAMES), dtype=np.int16),
        "has_final_public_vps": False,
        "final_actual_vps": np.zeros(len(PLAYER_NAMES), dtype=np.int16),
        "has_final_actual_vps": False,
        "action_mask_version": ACTION_MASK_VERSION,
        "policy_weight_multiplier": np.float32(
            # Playout-cap randomization deliberately spends cheap n_fast
            # searches to advance trajectories/value outcomes, but only the
            # independently selected n_full roots are policy teachers.  Giving
            # n_fast rows even a bounded positive weight silently changes an
            # "n128 teacher" corpus into a mixed n16/n128 policy objective.
            # Keep the paid fast-search distribution as evidence for sealed
            # reliability experiments, not as production policy authority.
            1.0
            if result.used_full_search and not is_forced
            else 0.0
        ),
        # Forced rows remain terminal-outcome value examples even when
        # trajectory_only skips the discarded root-Q/afterstate computation.
        "value_weight_multiplier": np.float32(1.0),
        "used_full_search": bool(result.used_full_search),
        "is_forced": bool(is_forced),
        "decision_class": str(decision_class),
        "decision_taxonomy_schema": DECISION_TAXONOMY_SCHEMA_VERSION,
        "simulations_used": np.int32(result.simulations_used),
        # Search-root supervision is admitted only for real, non-forced FULL
        # searches. Fast PCR rows contribute bounded policy supervision but
        # still keep their shallow root estimate out of the value-target path;
        # forced-action fast paths only advance trajectories/value outcomes
        # (REANALYZE_VALUE_TARGETS_DESIGN).
        "root_value": np.float32(
            result.root_value
            if (
                not is_forced
                and result.used_full_search
                and np.isfinite(result.root_value)
            )
            else np.nan
        ),
        "root_value_mask": np.bool_(
            not is_forced and result.used_full_search and np.isfinite(result.root_value)
        ),
        "afterstate_target": afterstate_target,
        "afterstate_target_mask": afterstate_target_mask,
        "prior_policy": prior_policy,
        # CAT-100 placeholders make the row schema uniform even for a source
        # whose full future trajectory is unavailable (e.g. an engine-
        # divergence-dropped exploiter game). play_one_game overwrites these
        # from the realized trajectory after the game ends.
        "aux_longest_road": np.float32(np.nan),
        "aux_largest_army": np.float32(np.nan),
        "aux_vp_in_n": np.float32(np.nan),
        "aux_next_settlement": np.int16(-1),
        "aux_robber_target": np.int16(-1),
        AUX_SUBGOAL_TARGET_VERSION_KEY: np.uint8(AUX_SUBGOAL_TARGET_VERSION),
    }
    if float(row["policy_weight_multiplier"]) > 0.0:
        # Private, opt-in evidence fields. The default writer filters these
        # out, preserving the historical shard schema byte-for-byte. Only
        # policy-active rows allocate them; forced/no-search rows are value-only.
        # completed-Q stays fp32 because measured near-flat root margins can
        # be ~1e-7; visits are range-checked before compact uint16 encoding.
        row["_search_visit_counts"] = np.asarray(
            [int(result.visit_counts.get(int(action), 0)) for action in legal_rust],
            dtype=np.int64,
        )
        row["_search_completed_q"] = np.asarray(
            [
                float(result.completed_q_values.get(int(action), np.nan))
                for action in legal_rust
            ],
            dtype=np.float32,
        )
    # Opponent-pool provenance (H2): only stamped when the caller is running
    # with a pool assignment at all (`is_pool_game`/`opponent_version` passed
    # as non-None) -- omitted entirely otherwise so pool-disabled runs keep
    # today's exact row schema (see EXTRA_KEYS's conditional-inclusion in
    # `GumbelShardWriter.add`).
    if is_pool_game is not None:
        row["is_pool_game"] = bool(is_pool_game)
    if opponent_version is not None:
        row["opponent_version"] = np.int32(opponent_version)
    # Opponent-MIX provenance (CAT-54): only stamped when the caller actually
    # has a named category (`opponent_tag` non-empty) -- see
    # `PoolGameAssignment`'s docstring for why this keeps the H2 binary-pool
    # path's row schema byte-identical to before this ticket.
    if opponent_tag:
        row["opponent_tag"] = str(opponent_tag)
        row["opponent_checkpoint_md5"] = str(opponent_checkpoint_md5)
    return row, features


def _target_information_regime_for_search(
    search_config: Any,
    *,
    engine_supports_determinization: bool,
    engine_supports_public_belief_development_draws: bool = False,
) -> str:
    """Return the planner-state provenance explicitly asserted by search.

    Fail-safe default: all historical Gumbel search configurations use an
    authoritative game clone, even when evaluator inputs are masked or the
    partial belief chance-spectrum flag is enabled.  Public provenance requires
    BOTH an explicit public-search mode and its native engine capabilities; no
    collection of loosely related booleans is accepted as equivalent proof.
    """

    information_set = bool(getattr(search_config, "information_set_search", False))
    coherent = bool(getattr(search_config, "coherent_public_belief_search", False))
    belief_chance = bool(getattr(search_config, "belief_chance_spectra", False))
    if information_set and coherent:
        raise ValueError(
            "information_set_search and coherent_public_belief_search are "
            "mutually exclusive"
        )
    if coherent:
        if belief_chance:
            raise ValueError(
                "coherent_public_belief_search cannot be combined with "
                "belief_chance_spectra"
            )
        if not engine_supports_determinization:
            raise RuntimeError(
                "coherent_public_belief_search=True requires a native game engine "
                "exposing determinize_for_player"
            )
        if not engine_supports_public_belief_development_draws:
            raise RuntimeError(
                "coherent_public_belief_search=True requires a native game engine "
                "exposing apply_public_belief_development_draws"
            )
        return TARGET_INFORMATION_REGIME_PUBLIC_COHERENT
    if information_set:
        if not engine_supports_determinization:
            raise RuntimeError(
                "information_set_search=True requires a native game engine exposing "
                "determinize_for_player; refusing to emit falsely public search targets"
            )
        particles = int(getattr(search_config, "determinization_particles", 1))
        if particles < 1:
            raise ValueError(
                "information_set_search requires determinization_particles >= 1, "
                f"got {particles}"
            )
        if belief_chance:
            raise ValueError(
                "information_set_search cannot be combined with belief_chance_spectra; "
                "sampled worlds already materialize hidden chance state"
            )
        return TARGET_INFORMATION_REGIME_PUBLIC
    return TARGET_INFORMATION_REGIME_AUTHORITATIVE


def _search_execution_contract(
    search_config: Any, *, native_mcts_hot_loop: bool
) -> dict[str, Any]:
    """Describe effective budget semantics that are not visible in the dataclass.

    Information-set search treats ``n_full``/``n_fast`` as one TOTAL budget,
    divides it across determinizations, and executes every particle sub-budget
    exactly.  Both the Python reference path and the native hot loop deliberately
    override the legacy Sequential-Halving rounding rule for those sub-searches.
    Recording only ``dataclasses.asdict(search_config)`` is therefore ambiguous:
    it can say ``exact_budget_sh=False`` even though every PIMC particle used an
    exact budget.  Keep that configured value, but attest the effective contract
    separately so generation audits never infer nominal compute from the wrong
    scheduler.
    """

    information_set = bool(getattr(search_config, "information_set_search", False))
    coherent = bool(getattr(search_config, "coherent_public_belief_search", False))
    return {
        "budget_scope": (
            "total_before_determinization_division"
            if information_set
            else (
                "single_public_belief_tree"
                if coherent
                else "single_authoritative_world"
            )
        ),
        "configured_exact_budget_sh": bool(
            getattr(search_config, "exact_budget_sh", False)
        ),
        "information_set_particle_subbudgets_exact": information_set,
        "forced_root_target_mode": str(
            getattr(search_config, "forced_root_target_mode", "full")
        ),
        "native_mcts_hot_loop": bool(native_mcts_hot_loop),
    }


def _search_evidence_recalibration_scope(search_config: Any) -> str:
    """Attest when one completed-Q vector can reproduce the target operator.

    A single-world target is reconstructible at the existing stored-prior
    precision from completed-Q, visits, phase, and manifest config. Public-belief
    ``aggregate_q_then_improve`` has the same property because its emitted
    completed-Q is the particle mean and the manifest records particle count.
    Historical ``mean_improved_policy`` does not: mean(softmax(...)) cannot be
    recovered from only the mean completed-Q. Refuse that combination rather
    than writing evidence that merely looks sufficient.
    """

    if bool(getattr(search_config, "coherent_public_belief_search", False)):
        return "public_belief_single_tree_root_v1"
    if not bool(getattr(search_config, "information_set_search", False)):
        return "single_world_root_v1"
    aggregation = str(
        getattr(
            search_config,
            "information_set_target_aggregation",
            "mean_improved_policy",
        )
    )
    if aggregation != "aggregate_q_then_improve":
        raise ValueError(
            "preserve_search_evidence cannot attest posthoc recalibration for "
            "information-set mean_improved_policy; use aggregate_q_then_improve "
            "or preserve per-particle evidence"
        )
    return "information_set_aggregate_q_then_improve_v1"


def _full_search_simulation_accounting(
    config: GumbelChanceMCTSConfig, legal_width: int
) -> tuple[int, int]:
    """Return nominal full budget and deterministic realized root visits.

    Legacy Sequential Halving does not promise ``simulations_used == n_full``:
    the round/floor schedule realizes a legal-width-dependent count.  Reliability
    selection must compare against that exact schedule instead of silently
    selecting only widths whose realized count happens to equal 128.
    """

    width = int(legal_width)
    wide = _gumbel_chance_mcts._wide_budget_applies(width, config)  # noqa: SLF001
    nominal = int(config.n_full_wide) if wide else int(config.n_full)
    candidates = _gumbel_chance_mcts._root_candidate_count(  # noqa: SLF001
        width, config
    )
    exact = bool(config.exact_budget_sh) and (
        int(config.exact_budget_sh_min_n) <= 0
        or nominal >= int(config.exact_budget_sh_min_n)
    )
    schedule = (
        _gumbel_chance_mcts.exact_budget_sh_phases(candidates, nominal)
        if exact
        else _gumbel_chance_mcts.sequential_halving_schedule(candidates, nominal)
    )
    realized = sum(int(count) * int(per_candidate) for count, per_candidate in schedule)
    return nominal, realized


def _is_n128_reliability_result(
    config: GumbelChanceMCTSConfig,
    *,
    legal_width: int,
    result: SearchResult,
) -> bool:
    nominal, expected = _full_search_simulation_accounting(config, legal_width)
    return (
        nominal == 128
        and bool(result.used_full_search)
        and int(result.simulations_used) == expected
    )


def play_one_game(
    mcts: GumbelChanceMCTS,
    evaluator: RustEvaluator,
    *,
    config: GumbelSelfPlayConfig,
    game_seed: int,
    game_index: int,
    action_size: int,
    eval_override: bool = False,
    pool_assignment: PoolGameAssignment | None = None,
    target_reliability_mcts: GumbelChanceMCTS | None = None,
) -> GameRecord:
    """Play one full self-play game, recording one row per decision.

    `pool_assignment` (H2, opponent pool): when given, `pool_assignment.
    champion_color` is the only seat whose decisions are recorded (`evaluator`
    drives it); when `pool_assignment.is_pool` is also True, the OTHER seat
    (`opponent_color`) is driven by `pool_assignment.opponent_evaluator`
    instead and its decisions are searched (to advance the game) but never
    built into a row -- the archived opponent only diversifies the states the
    champion faces, it is never itself a distillation target. `mcts.evaluator`
    is reassigned every decision based on `game.current_color()` (not just
    once at game start), so a worker reusing one `mcts`/game loop across many
    games self-heals regardless of which evaluator was left set by the
    previous game's final decision. `None` (the default) is exactly today's
    behavior: both seats use `evaluator`, every decision is recorded, and
    `_build_decision_row` omits the two provenance columns entirely.

    Single-legal-action prompts are engine transitions, not policy decisions:
    they are applied directly without neural inference, MCTS, or a training
    row. Public mandatory multi-choice prompts (initial placement, discard,
    robber placement, and Road Building placements) always use the full search
    budget. Other roots retain playout-cap randomization; wide roots are
    upgraded by the MCTS configuration.

    Both seats are driven by the same `mcts`. Its RNG advances naturally across
    decisions inside this game; `run_worker_games` assigns each absolute game a
    deterministic worker-seed-derived stream so resume/failure history cannot
    perturb later games. The live game's own chance outcomes (dice, robber steals, dev
    card draws) are sampled from a game_seed-derived RNG, independent of the
    search's internal RNG, so a game's board/chance trajectory is
    reproducible from `game_seed` alone given the same sequence of chosen
    actions.

    When ``config.target_reliability_audit_fraction`` is nonzero, eligible
    recorded exact-n128 roots may run one diagnostic duplicate through
    ``target_reliability_mcts``.  Its three stochastic roles are domain
    separated, its RNG is reset from (audit_seed, game_seed, decision_index),
    and only typed evidence is recorded; the primary result remains the sole
    source of the applied action.
    """
    started = time.perf_counter()
    catanatron_rs = _gumbel_chance_mcts._require_rust_module()
    game = catanatron_rs.Game.simple(list(config.colors), seed=int(game_seed))
    target_information_regime = _target_information_regime_for_search(
        mcts.config,
        engine_supports_determinization=hasattr(game, "determinize_for_player"),
        engine_supports_public_belief_development_draws=hasattr(
            game, "apply_public_belief_development_draws"
        ),
    )
    chance_rng = random.Random(int(game_seed) ^ 0xA17E)
    reliability_fraction = float(config.target_reliability_audit_fraction)
    if (
        not math.isfinite(reliability_fraction)
        or not 0.0 <= reliability_fraction <= 1.0
    ):
        raise ValueError("target reliability audit fraction must be in [0, 1]")
    if reliability_fraction > 0.0 and target_reliability_mcts is None:
        raise ValueError(
            "target reliability audit requires an independent duplicate-search object"
        )

    decisions: list[DecisionRecord] = []
    aux_states = []
    aux_actor_colors: list[str] = []
    aux_actions: list[Any] = []
    recorded_aux_indices: list[int] = []
    aux_hex_ids: dict[tuple[int, int, int], int] | None = None
    decision_index = 0
    nonforced_choice_index = 0
    forced_decisions = 0
    simulations_used_total = 0
    terminal = False

    while decision_index < int(config.max_decisions):
        if game.winning_color() is not None:
            terminal = True
            break
        legal_rust = tuple(
            int(action)
            for action in game.playable_action_indices(
                list(config.colors), config.map_kind
            )
        )
        if not legal_rust:
            break

        if len(legal_rust) == 1 and not bool(config.record_automatic_transitions):
            # Explicit legacy compatibility path. We still route the sole
            # transition through the deterministic chance sampler so
            # game_seed replay semantics remain identical, but no learner row
            # is emitted. Current coherent recipes must keep this opt-out off.
            action_ids = [
                int(action)
                for action in game.playable_action_indices(list(config.colors), None)
            ]
            raw_actions = json.loads(game.playable_actions_json())
            action_by_id = dict(zip(action_ids, raw_actions))
            selected_action_id = int(legal_rust[0])
            selected_action = action_by_id.get(selected_action_id)
            if selected_action is None:
                raise RuntimeError(
                    f"automatic action {selected_action_id} is not legal"
                )
            automatic_class = classify_public_decision(
                None,
                legal_action_count=1,
                wide_threshold=max(
                    2,
                    int(getattr(mcts.config, "n_full_wide_threshold", 20) or 20),
                ),
            )
            if automatic_class != AUTOMATIC_TRANSITION:
                raise RuntimeError("single-action root lost automatic classification")
            game = _apply_selected_action(
                game,
                selected_action_id,
                colors=config.colors,
                rng=chance_rng,
                correct_rust_chance_spectra=config.correct_rust_chance_spectra,
                action_json=selected_action,
            )
            # This is deliberately not counted as a forced *decision*. The
            # telemetry field tracks retained one-action learner rows; this
            # transition has neither a policy decision nor a row.
            decision_index += 1
            continue

        # Capture one authoritative pre-action snapshot/action map per ply.
        # _build_decision_row and _apply_selected_action consume the same
        # objects below, avoiding the duplicate Rust JSON/FFI calls that aux
        # labeling would otherwise add to the generation hot path.
        snapshot = json.loads(game.json_snapshot())
        action_ids = [
            int(action)
            for action in game.playable_action_indices(list(config.colors), None)
        ]
        raw_actions = json.loads(game.playable_actions_json())
        action_by_id = dict(zip(action_ids, raw_actions))
        decision_class = classify_public_decision(
            snapshot,
            legal_action_count=len(legal_rust),
            wide_threshold=max(
                2,
                int(getattr(mcts.config, "n_full_wide_threshold", 20) or 20),
            ),
        )
        aux_states.append(rust_aux_state_from_snapshot(snapshot))
        if aux_hex_ids is None:
            aux_hex_ids = rust_hex_id_by_coordinate(snapshot)

        temperature = _temperature_for_decision(
            decision_index,
            config=config,
            eval_override=eval_override,
            nonforced_choice_index=nonforced_choice_index,
        )
        mcts.config = dataclasses.replace(mcts.config, temperature=temperature)

        # Opponent-pool seat routing (H2): resolve which evaluator drives THIS
        # decision from the live acting color, every decision (not just once
        # at game start) -- `mcts` is reused across games in a worker, so this
        # also self-heals a leftover opponent-evaluator swap from a prior
        # game's last (opponent-seat) decision.
        acting_color = str(game.current_color())
        record_row = True
        if (
            pool_assignment is not None
            and pool_assignment.is_pool
            and acting_color == pool_assignment.opponent_color
        ):
            mcts.evaluator = pool_assignment.opponent_evaluator
            record_row = False
        else:
            mcts.evaluator = evaluator

        result = mcts.search(
            game,
            force_full=(
                True
                if eval_override or decision_requires_full_search(decision_class)
                else None
            ),
        )
        if len(legal_rust) > 1:
            nonforced_choice_index += 1
        simulations_used_total += int(result.simulations_used)
        selected_action = action_by_id.get(int(result.selected_action))
        if selected_action is None:
            raise RuntimeError(f"selected action {result.selected_action} is not legal")
        aux_actor_colors.append(acting_color)
        aux_actions.append(selected_action)

        reliability_fields: dict[str, Any] | None = None
        if reliability_fraction > 0.0 and record_row:
            reliability_fields = unaudited_target_reliability_fields()
            eligible_exact_n128 = len(legal_rust) > 1 and _is_n128_reliability_result(
                mcts.config, legal_width=len(legal_rust), result=result
            )
            if eligible_exact_n128 and target_reliability_root_selected(
                game_seed=int(game_seed),
                decision_index=int(decision_index),
                audit_seed=int(config.target_reliability_audit_seed),
                audit_fraction=reliability_fraction,
            ):
                assert target_reliability_mcts is not None
                # Record-row routing guarantees this is the producer evaluator,
                # not an archived opponent's model.  Keep the duplicate's
                # temperature at zero: reliability concerns the distilled
                # target/operator, not the intentionally stochastic live move.
                target_reliability_mcts.evaluator = evaluator
                target_reliability_mcts.config = dataclasses.replace(
                    target_reliability_mcts.config, temperature=0.0
                )
                target_reliability_mcts.seed_search_rngs(
                    target_reliability_root_seed(
                        game_seed=int(game_seed),
                        decision_index=int(decision_index),
                        audit_seed=int(config.target_reliability_audit_seed),
                    )
                )
                duplicate = target_reliability_mcts.search(game.copy(), force_full=True)
                if not _is_n128_reliability_result(
                    target_reliability_mcts.config,
                    legal_width=len(legal_rust),
                    result=duplicate,
                ):
                    nominal, expected = _full_search_simulation_accounting(
                        target_reliability_mcts.config, len(legal_rust)
                    )
                    raise RuntimeError(
                        "target reliability duplicate violated n128 full-search "
                        "accounting: "
                        f"used_full={duplicate.used_full_search!r} "
                        f"nominal={nominal!r} expected={expected!r} "
                        f"actual={duplicate.simulations_used!r}"
                    )
                reliability_fields = duplicate_search_reliability_fields(
                    primary_policy=result.improved_policy,
                    duplicate_policy=duplicate.improved_policy,
                    primary_completed_q=result.completed_q_values,
                    duplicate_completed_q=duplicate.completed_q_values,
                )

        if record_row:
            if len(legal_rust) <= 1:
                forced_decisions += 1
            row, features = _build_decision_row(
                game,
                result=result,
                action_size=action_size,
                colors=config.colors,
                game_seed=game_seed,
                decision_index=decision_index,
                obs_width=config.obs_width,
                target_information_regime=target_information_regime,
                is_pool_game=(
                    pool_assignment.is_pool if pool_assignment is not None else None
                ),
                opponent_version=(
                    pool_assignment.opponent_version
                    if pool_assignment is not None
                    else None
                ),
                opponent_tag=(
                    pool_assignment.tag if pool_assignment is not None else ""
                ),
                opponent_checkpoint_md5=(
                    pool_assignment.opponent_md5 if pool_assignment is not None else ""
                ),
                snapshot=snapshot,
                action_by_id=action_by_id,
                meaningful_public_history=bool(config.meaningful_public_history),
                event_history_limit=int(config.event_history_limit),
                decision_class=decision_class,
                entity_feature_adapter_version=entity_adapter_for_evaluator(evaluator),
            )
            if reliability_fields is not None:
                row.update(reliability_fields)
            decisions.append(DecisionRecord(row=row, features=features))
            recorded_aux_indices.append(len(aux_states) - 1)

        game = _apply_selected_action(
            game,
            int(result.selected_action),
            colors=config.colors,
            rng=chance_rng,
            correct_rust_chance_spectra=config.correct_rust_chance_spectra,
            action_json=selected_action,
        )
        decision_index += 1

    if not terminal:
        terminal = game.winning_color() is not None
    truncated = not terminal
    outcome = _game_outcome_fields(game, terminal=terminal, colors=config.colors)
    final_aux_state = rust_aux_state_from_snapshot(json.loads(game.json_snapshot()))
    aux_targets = trajectory_targets(
        states=aux_states,
        actor_colors=aux_actor_colors,
        actions=aux_actions,
        horizon=AUX_VP_HORIZON,
        victory_points_of=lambda state, color: state.victory_points(color),
        holds_longest_road_at=lambda state, color: state.holds_longest_road(color),
        holds_largest_army_at=lambda state, color: state.holds_largest_army(color),
        settlement_node_of_action=rust_settlement_node_of_action,
        robber_hex_of_action=lambda action: rust_robber_hex_of_action(
            action, aux_hex_ids or {}
        ),
        final_state=final_aux_state,
        trajectory_complete=terminal,
    )
    for record, aux_index in zip(decisions, recorded_aux_indices):
        record.row.update(outcome)
        targets = aux_targets[aux_index]
        record.row.update(
            {
                "aux_longest_road": np.float32(targets["aux_longest_road"]),
                "aux_largest_army": np.float32(targets["aux_largest_army"]),
                "aux_vp_in_n": np.float32(targets["aux_vp_in_n"]),
                "aux_next_settlement": np.int16(targets["aux_next_settlement"]),
                "aux_robber_target": np.int16(targets["aux_robber_target"]),
                AUX_SUBGOAL_TARGET_VERSION_KEY: np.uint8(AUX_SUBGOAL_TARGET_VERSION),
            }
        )

    return GameRecord(
        game_seed=int(game_seed),
        game_index=int(game_index),
        decisions=decisions,
        terminal=terminal,
        truncated=truncated,
        winner=str(outcome["winner"]),
        total_decisions=decision_index,
        forced_decisions=forced_decisions,
        simulations_used_total=simulations_used_total,
        wall_time_sec=time.perf_counter() - started,
    )


class GumbelShardWriter:
    """Self-contained shard writer matching `EntityShardWriter`'s output schema.

    Reimplemented here (rather than importing `tools.convert_teacher_to_entity_tokens`)
    so `src/catan_zero/rl` does not depend on the `tools/` script directory,
    and so the extra `EXTRA_KEYS` columns (afterstate targets, search-budget
    bookkeeping) can be added without touching that shared file. Field names/
    dtypes/padding conventions for `BASE_KEYS`/`ENTITY_KEYS` are kept
    byte-for-byte compatible with it; `tools/train_bc.py`'s loader ignores
    unknown extra keys, so `EXTRA_KEYS` are forward-compatible additions.
    """

    def __init__(
        self,
        output: Path,
        *,
        shard_size: int = 2048,
        fmt: str = "npz",
        start_index: int = 0,
        rows_written: int = 0,
        preserve_search_evidence: bool = False,
    ) -> None:
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.shard_size = max(1, int(shard_size))
        self.format = fmt
        self.preserve_search_evidence = bool(preserve_search_evidence)
        self.rows: list[dict[str, Any]] = []
        self.paths: list[Path] = []
        # Resume support (`run_worker_games(resume=True)`): the writer
        # continues shard numbering from `start_index` rather than 0 so it
        # never collides with (and never re-emits) shards already confirmed
        # durable by a prior session's `WorkerProgress`.
        self.index = int(start_index)
        # Actual row cardinality of completed shards.  Resume cannot derive
        # this from index * shard_size once shards are game-atomic and may
        # exceed the target by up to one game.
        self.rows_written = int(rows_written)

    def _payload(
        self, row: dict[str, Any], features: dict[str, np.ndarray]
    ) -> dict[str, Any]:
        payload = {key: row[key] for key in BASE_KEYS if key in row}
        for key in EXTRA_KEYS:
            if key in row:
                payload[key] = row[key]
        if (
            self.preserve_search_evidence
            and float(row.get("policy_weight_multiplier", 0.0)) > 0.0
        ):
            for key in ("_search_visit_counts", "_search_completed_q"):
                if key not in row:
                    raise ValueError(
                        f"preserve_search_evidence requires row field {key!r}"
                    )
                payload[key] = row[key]
        for key in ENTITY_KEYS:
            payload[key] = features[key]
        return payload

    def add(self, row: dict[str, Any], features: dict[str, np.ndarray]) -> None:
        payload = self._payload(row, features)
        self.rows.append(payload)
        if len(self.rows) >= self.shard_size:
            self.flush()

    def add_game(self, decisions: Sequence[DecisionRecord]) -> None:
        """Append one realized game without permitting a mid-game flush.

        The target shard size is soft by at most one game's rows.  Keeping a
        game indivisible is required for safe offset-based resume: a confirmed
        shard may never retain a prefix of the first game that will be replayed.
        Other writer users retain the historical row-wise :meth:`add` API.
        """

        payloads = [self._payload(item.row, item.features) for item in decisions]
        self.rows.extend(payloads)
        if len(self.rows) >= self.shard_size:
            self.flush()

    def close(self) -> None:
        self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        row_count = len(self.rows)
        arrays = _rows_to_arrays(
            self.rows,
            preserve_search_evidence=self.preserve_search_evidence,
        )
        path = self.output / f"gumbel_self_play_shard_{self.index:05d}.npz"
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        _fsync_directory(path.parent)
        if self.format == "npz_zst":
            path = _try_zstd(path)
        self.paths.append(path)
        self.rows = []
        self.index += 1
        self.rows_written += row_count


def _compact_search_evidence(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Encode policy-active root evidence without width-54 shard padding.

    Active rows are already identified by the shard's mandatory
    ``policy_weight_multiplier`` column, so only their uint32 offsets plus
    fp32 completed-Q and uint16 visits are stored. Valid fast searches are
    active at bounded confidence; forced and no-search rows remain value-only
    and consume no evidence payload.
    """

    offsets = [0]
    visit_chunks: list[np.ndarray] = []
    completed_q_chunks: list[np.ndarray] = []
    for row in rows:
        policy_weight = float(row.get("policy_weight_multiplier", 0.0))
        if not math.isfinite(policy_weight) or policy_weight < 0.0:
            raise ValueError(
                "search evidence requires a finite non-negative policy weight"
            )
        if policy_weight == 0.0:
            continue
        legal_count = int(np.asarray(row["legal_action_ids"]).shape[0])
        visits = np.asarray(row.get("_search_visit_counts"), dtype=np.int64)
        completed_q = np.asarray(row.get("_search_completed_q"), dtype=np.float32)
        if visits.shape != (legal_count,) or completed_q.shape != (legal_count,):
            raise ValueError(
                "search evidence must align exactly with the row's legal-action axis"
            )
        if bool(np.any(visits < 0)) or bool(np.any(visits > np.iinfo(np.uint16).max)):
            raise ValueError("search visit counts exceed uint16 evidence schema")
        if not bool(np.all(np.isfinite(completed_q))):
            raise ValueError("policy-active search evidence has non-finite completed-Q")
        expected_simulations = int(row.get("simulations_used", int(visits.sum())))
        if int(visits.sum()) != expected_simulations:
            raise ValueError(
                "search evidence visit sum differs from simulations_used: "
                f"{int(visits.sum())} != {expected_simulations}"
            )
        visit_chunks.append(visits.astype(np.uint16, copy=False))
        completed_q_chunks.append(completed_q)
        offsets.append(offsets[-1] + legal_count)

    if (
        len(offsets) - 1 > np.iinfo(np.uint32).max
        or offsets[-1] > np.iinfo(np.uint32).max
    ):
        raise ValueError("search evidence shard exceeds uint32 index schema")
    visits_flat = (
        np.concatenate(visit_chunks)
        if visit_chunks
        else np.asarray([], dtype=np.uint16)
    )
    completed_q_flat = (
        np.concatenate(completed_q_chunks)
        if completed_q_chunks
        else np.asarray([], dtype=np.float32)
    )
    return {
        "search_evidence_version": np.asarray(SEARCH_EVIDENCE_VERSION, dtype=np.uint8),
        "search_evidence_offsets": np.asarray(offsets, dtype=np.uint32),
        "search_visit_counts_flat": visits_flat,
        "search_completed_q_flat": completed_q_flat,
    }


def search_evidence_for_row(
    shard: Mapping[str, Any], row_index: int
) -> dict[str, np.ndarray] | None:
    """Decode one optional compact evidence row, or return None when absent."""

    evidence_keys = {
        "search_evidence_version",
        "search_evidence_offsets",
        "search_visit_counts_flat",
        "search_completed_q_flat",
    }
    present = evidence_keys.intersection(shard.keys())
    if not present:
        return None
    required = evidence_keys | {"legal_action_ids", "policy_weight_multiplier"}
    present = required.intersection(shard.keys())
    if present != required:
        raise ValueError(
            f"incomplete search evidence payload; missing={sorted(required - present)}"
        )
    version = int(np.asarray(shard["search_evidence_version"]).item())
    if version != SEARCH_EVIDENCE_VERSION:
        raise ValueError(f"unsupported search evidence version {version!r}")
    offsets = np.asarray(shard["search_evidence_offsets"], dtype=np.uint32)
    policy_weights = np.asarray(shard["policy_weight_multiplier"], dtype=np.float32)
    if policy_weights.ndim != 1:
        raise ValueError("malformed policy_weight_multiplier column")
    active = policy_weights > 0.0
    if offsets.shape != (int(active.sum()) + 1,) or int(offsets[0]) != 0:
        raise ValueError("malformed search evidence offsets")
    if offsets.size > 1 and bool(np.any(offsets[1:] < offsets[:-1])):
        raise ValueError("search evidence offsets must be non-decreasing")
    if row_index < 0 or row_index >= policy_weights.size:
        raise IndexError(f"search evidence row index out of range: {row_index}")
    if not bool(active[int(row_index)]):
        return None
    position = int(np.count_nonzero(active[: int(row_index)]))
    start, stop = int(offsets[position]), int(offsets[position + 1])
    visits = np.asarray(shard["search_visit_counts_flat"], dtype=np.uint16)
    completed_q = np.asarray(shard["search_completed_q_flat"], dtype=np.float32)
    if (
        visits.ndim != 1
        or visits.shape != completed_q.shape
        or int(offsets[-1]) != visits.size
    ):
        raise ValueError("malformed flat search evidence payload")
    legal = np.asarray(shard["legal_action_ids"])[int(row_index)]
    legal = np.asarray(legal[legal >= 0], dtype=np.int16)
    if stop - start != legal.size:
        raise ValueError("search evidence width differs from legal-action count")
    return {
        "legal_action_ids": legal.copy(),
        "visit_counts": visits[start:stop].copy(),
        "completed_q": completed_q[start:stop].copy(),
    }


def _rows_to_arrays(
    rows: list[dict[str, Any]], *, preserve_search_evidence: bool = False
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    legal_width = max(int(np.asarray(row["legal_action_ids"]).shape[0]) for row in rows)
    for key in (*BASE_KEYS, *EXTRA_KEYS):
        if key not in rows[0]:
            continue
        values = [row[key] for row in rows]
        if key in {
            "legal_action_ids",
            "target_policy",
            "target_scores",
            "target_policy_mask",
            "target_scores_mask",
            "prior_policy",
        }:
            fill = (
                -1
                if key == "legal_action_ids"
                else np.nan
                if key == "target_scores"
                else False
                if key.endswith("_mask")
                else 0.0
            )
            out[key] = np.stack(
                [
                    _pad_1d(np.asarray(value), legal_width, fill=fill)
                    for value in values
                ],
                axis=0,
            )
        elif key in {"afterstate_target", "afterstate_target_mask"}:
            fill = np.nan if key == "afterstate_target" else False
            out[key] = np.stack(
                [
                    _pad_1d(np.asarray(value), legal_width, fill=fill)
                    for value in values
                ],
                axis=0,
            )
        elif key == "legal_action_context":
            feature_size = int(np.asarray(values[0]).shape[1])
            out[key] = np.stack(
                [
                    _pad_2d(np.asarray(value), legal_width, feature_size, fill=0.0)
                    for value in values
                ],
                axis=0,
            )
        else:
            out[key] = np.asarray(values)
    for key in ENTITY_KEYS:
        values = [row[key] for row in rows]
        if key in {
            "legal_action_tokens",
            "legal_action_target_ids",
            "legal_action_mask",
        }:
            if key == "legal_action_tokens":
                out[key] = np.stack(
                    [
                        _pad_2d(
                            np.asarray(value),
                            legal_width,
                            np.asarray(value).shape[1],
                            fill=0.0,
                        )
                        for value in values
                    ],
                    axis=0,
                ).astype(np.float16, copy=False)
            elif key == "legal_action_target_ids":
                out[key] = np.stack(
                    [
                        _pad_2d(np.asarray(value), legal_width, 4, fill=-1)
                        for value in values
                    ],
                    axis=0,
                ).astype(np.int16, copy=False)
            else:
                out[key] = np.stack(
                    [
                        _pad_1d(np.asarray(value), legal_width, fill=False)
                        for value in values
                    ],
                    axis=0,
                ).astype(np.bool_, copy=False)
        else:
            out[key] = np.stack(values, axis=0)
    if preserve_search_evidence:
        out.update(_compact_search_evidence(rows))
    return out


def _pad_1d(value: np.ndarray, width: int, *, fill: Any) -> np.ndarray:
    value = np.asarray(value)
    out = np.full((int(width),), fill, dtype=value.dtype)
    count = min(int(width), int(value.shape[0]))
    out[:count] = value[:count]
    return out


def _pad_2d(
    value: np.ndarray, width: int, feature_size: int, *, fill: Any
) -> np.ndarray:
    value = np.asarray(value)
    out = np.full((int(width), int(feature_size)), fill, dtype=value.dtype)
    rows = min(int(width), int(value.shape[0]))
    cols = min(int(feature_size), int(value.shape[1]))
    out[:rows, :cols] = value[:rows, :cols]
    return out


def _try_zstd(path: Path) -> Path:
    try:
        import zstandard
    except ImportError:
        return path
    compressed = path.with_name(path.name + ".zst")
    tmp = compressed.with_name(f".{compressed.name}.tmp.{os.getpid()}.{time.time_ns()}")
    compressor = zstandard.ZstdCompressor(level=12)
    try:
        with path.open("rb") as source, tmp.open("xb") as target:
            compressor.copy_stream(source, target)
            target.flush()
            os.fsync(target.fileno())
        os.replace(tmp, compressed)
        _fsync_directory(compressed.parent)
        path.unlink()
        _fsync_directory(compressed.parent)
    finally:
        tmp.unlink(missing_ok=True)
    return compressed


def _fsync_directory(path: Path) -> None:
    """Persist a file create/replace/unlink in its containing directory."""

    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(slots=True)
class WorkerProgress:
    """Durable, incremental resume marker for one `run_worker_games` call.

    Written to `<out_dir>/progress.json` after every game this worker
    processes (success or failure).  Aggregate counters in this structure are
    deliberately the confirmed-durable snapshot, not speculative live
    counters for games still buffered in memory.

    Why this is safe to trust on resume (the core durability argument):
    a game's rows are only ever handed to the shard writer AFTER
    `play_one_game` returns with the full, in-memory `GameRecord` (see
    `run_worker_games` below) -- so a container killed mid-game leaves
    *zero* trace of that game anywhere (not in a shard, not in this
    progress file, not even a partial row). There is nothing to clean up
    for a mid-game kill; the only thing a preemption can strand is games
    that already finished playing but whose rows haven't reached an
    on-disk (flushed) shard yet, because `GumbelShardWriter` only flushes
    to disk every `shard_size` rows (or on `close()`, which never runs on
    a hard kill).

    `games_completed_local` therefore means "processed offsets whose rows are
    all inside completed, on-disk, game-atomic shards" (a failed zero-row game
    is confirmed by the atomic progress write itself).  Every aggregate below
    is captured at that same offset.  This is what makes resume safe: every
    offset below the marker is fully durable and every offset at/above it is
    wholly absent from the retained shard inventory.
    """

    run_id: str
    resume_contract_version: int
    generation_semantics_sha256: str
    # Resume is only sound when every retained shard was produced under the
    # same row-label semantics as the code completing the run.  These fields
    # deliberately make pre-v1 progress markers unreadable by `from_dict` so
    # a retry replays the worker instead of blessing legacy aux labels with a
    # v1 manifest.
    aux_subgoal_target_version: int
    aux_subgoal_target_semantic: str
    shard_size: int
    shard_format: str
    base_seed: int
    game_index_start: int
    games_requested: int
    games_completed_local: int
    games_succeeded: int
    shard_count_confirmed: int
    rows_confirmed: int
    confirmed_shards: list[dict[str, Any]]
    games_failed: int
    games_truncated: int
    rows: int
    decisions_total: int
    forced_decisions_total: int
    simulations_used_total: int
    wins_by_color: dict[str, int]
    errors: list[dict[str, Any]]
    opponent_pool_games: int
    opponent_pool_per_version_stats: dict[str, dict[str, int]]
    opponent_mix_pool_games: int
    opponent_mix_per_tag_stats: dict[str, dict[str, int]]
    exploiter_games: int
    exploiter_per_engine_stats: dict[str, dict[str, int]]
    exploiter_divergence_topics: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WorkerProgress":
        def _nonnegative_int(value: Any, field: str) -> int:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"worker progress {field} must be a non-negative integer"
                )
            return value

        def _stats(
            value: Any, *, field: str, keys: tuple[str, ...]
        ) -> dict[str, dict[str, int]]:
            if not isinstance(value, dict):
                raise ValueError(f"worker progress {field} must be an object")
            parsed: dict[str, dict[str, int]] = {}
            for raw_name, raw_stats in value.items():
                name = str(raw_name)
                if (
                    not name
                    or not isinstance(raw_stats, dict)
                    or set(raw_stats) != set(keys)
                ):
                    raise ValueError(
                        f"worker progress {field} has malformed entry {raw_name!r}"
                    )
                parsed[name] = {
                    key: _nonnegative_int(raw_stats[key], f"{field}.{name}.{key}")
                    for key in keys
                }
                if parsed[name].get("champion_wins", 0) > parsed[name].get("games", 0):
                    raise ValueError(f"worker progress {field}.{name} has wins > games")
            return parsed

        def _confirmed_inventory(value: Any) -> list[dict[str, Any]]:
            if not isinstance(value, list):
                raise ValueError("worker progress confirmed_shards must be a list")
            records: list[dict[str, Any]] = []
            expected_keys = {
                "index",
                "filename",
                "rows",
                "size_bytes",
                "sha256",
                "game_seed_sha256",
                "game_seed_min",
                "game_seed_max",
                "unique_game_seeds",
            }
            for offset, raw in enumerate(value):
                if not isinstance(raw, dict) or set(raw) != expected_keys:
                    raise ValueError(
                        "worker progress confirmed shard record is malformed"
                    )
                index = _nonnegative_int(
                    raw["index"], f"confirmed_shards[{offset}].index"
                )
                rows = _nonnegative_int(raw["rows"], f"confirmed_shards[{offset}].rows")
                size_bytes = _nonnegative_int(
                    raw["size_bytes"], f"confirmed_shards[{offset}].size_bytes"
                )
                unique = _nonnegative_int(
                    raw["unique_game_seeds"],
                    f"confirmed_shards[{offset}].unique_game_seeds",
                )
                filename = raw["filename"]
                sha256 = raw["sha256"]
                seed_sha256 = raw["game_seed_sha256"]
                seed_min = raw["game_seed_min"]
                seed_max = raw["game_seed_max"]
                expected_filename = f"gumbel_self_play_shard_{index:05d}.npz" + (
                    ".zst" if payload.get("shard_format") == "npz_zst" else ""
                )
                if (
                    index != offset
                    or not isinstance(filename, str)
                    or filename != expected_filename
                    or not isinstance(sha256, str)
                    or len(sha256) != 64
                    or any(character not in "0123456789abcdef" for character in sha256)
                    or not isinstance(seed_sha256, str)
                    or len(seed_sha256) != 64
                    or any(
                        character not in "0123456789abcdef" for character in seed_sha256
                    )
                    or isinstance(seed_min, bool)
                    or not isinstance(seed_min, int)
                    or isinstance(seed_max, bool)
                    or not isinstance(seed_max, int)
                    or rows <= 0
                    or size_bytes <= 0
                    or unique <= 0
                    or seed_min > seed_max
                ):
                    raise ValueError(
                        "worker progress confirmed shard identity is invalid"
                    )
                records.append(dict(raw))
            return records

        raw_resume_contract_version = payload["resume_contract_version"]
        if (
            isinstance(raw_resume_contract_version, bool)
            or not isinstance(raw_resume_contract_version, int)
            or raw_resume_contract_version != WORKER_PROGRESS_RESUME_CONTRACT_VERSION
        ):
            raise ValueError(
                "worker progress resume contract version mismatch: "
                f"expected {WORKER_PROGRESS_RESUME_CONTRACT_VERSION}, "
                f"got {raw_resume_contract_version!r}"
            )
        raw_aux_subgoal_target_version = payload[AUX_SUBGOAL_TARGET_VERSION_KEY]
        # JSON booleans are Python ints, but accepting `true` as schema v1
        # would make a malformed/corrupt marker look authenticated.
        if isinstance(raw_aux_subgoal_target_version, bool) or not isinstance(
            raw_aux_subgoal_target_version, int
        ):
            raise ValueError(
                "worker progress aux-subgoal target version must be an integer"
            )
        aux_subgoal_target_version = raw_aux_subgoal_target_version
        aux_subgoal_target_semantic = str(payload["aux_subgoal_target_semantic"])
        if aux_subgoal_target_version != AUX_SUBGOAL_TARGET_VERSION:
            raise ValueError(
                "worker progress aux-subgoal target version mismatch: "
                f"expected {AUX_SUBGOAL_TARGET_VERSION}, got {aux_subgoal_target_version}"
            )
        if aux_subgoal_target_semantic != AUX_SUBGOAL_TARGET_SEMANTIC:
            raise ValueError(
                "worker progress aux-subgoal target semantic mismatch: "
                f"expected {AUX_SUBGOAL_TARGET_SEMANTIC!r}, "
                f"got {aux_subgoal_target_semantic!r}"
            )
        generation_semantics_sha256 = str(payload["generation_semantics_sha256"])
        if (
            not generation_semantics_sha256.startswith("sha256:")
            or len(generation_semantics_sha256) != 71
            or any(
                character not in "0123456789abcdef"
                for character in generation_semantics_sha256.removeprefix("sha256:")
            )
        ):
            raise ValueError("worker progress generation semantics digest is invalid")
        confirmed_shards = _confirmed_inventory(payload["confirmed_shards"])
        pool_stats = _stats(
            payload["opponent_pool_per_version_stats"],
            field="opponent_pool_per_version_stats",
            keys=("games", "champion_wins"),
        )
        if any(
            version != str(int(version)) or int(version) < 0 for version in pool_stats
        ):
            raise ValueError(
                "worker progress opponent-pool versions must be canonical non-negative integers"
            )
        mix_stats = _stats(
            payload["opponent_mix_per_tag_stats"],
            field="opponent_mix_per_tag_stats",
            keys=("games", "champion_wins"),
        )
        exploiter_stats = _stats(
            payload["exploiter_per_engine_stats"],
            field="exploiter_per_engine_stats",
            keys=("games", "champion_wins", "divergences"),
        )
        raw_topics = payload["exploiter_divergence_topics"]
        if not isinstance(raw_topics, dict):
            raise ValueError(
                "worker progress exploiter_divergence_topics must be an object"
            )
        topics = {
            str(topic): _nonnegative_int(count, f"exploiter_divergence_topics.{topic}")
            for topic, count in raw_topics.items()
        }
        raw_errors = payload["errors"]
        if not isinstance(raw_errors, list) or any(
            not isinstance(error, dict)
            or set(error) != {"game_index", "game_seed", "error"}
            or isinstance(error["game_index"], bool)
            or not isinstance(error["game_index"], int)
            or isinstance(error["game_seed"], bool)
            or not isinstance(error["game_seed"], int)
            or not isinstance(error["error"], str)
            for error in raw_errors
        ):
            raise ValueError("worker progress errors are malformed")

        result = cls(
            run_id=str(payload["run_id"]),
            resume_contract_version=raw_resume_contract_version,
            generation_semantics_sha256=generation_semantics_sha256,
            aux_subgoal_target_version=aux_subgoal_target_version,
            aux_subgoal_target_semantic=aux_subgoal_target_semantic,
            shard_size=_nonnegative_int(payload["shard_size"], "shard_size"),
            shard_format=str(payload["shard_format"]),
            base_seed=_nonnegative_int(payload["base_seed"], "base_seed"),
            game_index_start=_nonnegative_int(
                payload["game_index_start"], "game_index_start"
            ),
            games_requested=_nonnegative_int(
                payload["games_requested"], "games_requested"
            ),
            games_completed_local=_nonnegative_int(
                payload["games_completed_local"], "games_completed_local"
            ),
            games_succeeded=_nonnegative_int(
                payload["games_succeeded"], "games_succeeded"
            ),
            shard_count_confirmed=_nonnegative_int(
                payload["shard_count_confirmed"], "shard_count_confirmed"
            ),
            rows_confirmed=_nonnegative_int(
                payload["rows_confirmed"], "rows_confirmed"
            ),
            confirmed_shards=confirmed_shards,
            games_failed=_nonnegative_int(payload["games_failed"], "games_failed"),
            games_truncated=_nonnegative_int(
                payload["games_truncated"], "games_truncated"
            ),
            rows=_nonnegative_int(payload["rows"], "rows"),
            decisions_total=_nonnegative_int(
                payload["decisions_total"], "decisions_total"
            ),
            forced_decisions_total=_nonnegative_int(
                payload["forced_decisions_total"], "forced_decisions_total"
            ),
            simulations_used_total=_nonnegative_int(
                payload["simulations_used_total"], "simulations_used_total"
            ),
            wins_by_color={
                str(k): _nonnegative_int(v, f"wins_by_color.{k}")
                for k, v in dict(payload["wins_by_color"]).items()
            },
            errors=[dict(error) for error in raw_errors],
            opponent_pool_games=_nonnegative_int(
                payload["opponent_pool_games"], "opponent_pool_games"
            ),
            opponent_pool_per_version_stats=pool_stats,
            opponent_mix_pool_games=_nonnegative_int(
                payload["opponent_mix_pool_games"], "opponent_mix_pool_games"
            ),
            opponent_mix_per_tag_stats=mix_stats,
            exploiter_games=_nonnegative_int(
                payload["exploiter_games"], "exploiter_games"
            ),
            exploiter_per_engine_stats=exploiter_stats,
            exploiter_divergence_topics=topics,
        )
        if (
            not result.run_id
            or result.shard_size <= 0
            or result.shard_format not in {"npz", "npz_zst"}
            or result.games_completed_local > result.games_requested
            or result.games_succeeded + result.games_failed
            != result.games_completed_local
            or result.games_truncated > result.games_succeeded
            or result.rows != result.rows_confirmed
            or result.forced_decisions_total > result.decisions_total
            or sum(result.wins_by_color.values()) > result.games_succeeded
            or len(result.confirmed_shards) != result.shard_count_confirmed
            or sum(record["rows"] for record in result.confirmed_shards)
            != result.rows_confirmed
            or len(result.errors) != result.games_failed
            or sum(stats["games"] for stats in pool_stats.values())
            != result.opponent_pool_games
            or result.opponent_mix_pool_games
            > sum(stats["games"] for stats in mix_stats.values())
            or sum(stats["games"] for stats in exploiter_stats.values())
            != result.exploiter_games
            or sum(stats["divergences"] for stats in exploiter_stats.values())
            != sum(topics.values())
        ):
            raise ValueError("worker progress aggregate coherence failed")
        return result


def _load_worker_progress(out_dir: Path) -> WorkerProgress | None:
    """Read `<out_dir>/progress.json`, or `None` if absent/unreadable.

    A missing, corrupt (torn write, truncated), or semantically stale progress
    file is treated identically to "no progress yet".  `run_worker_games`
    then replays from game 0; when a stale progress file exists it also removes
    the old worker shards first so no legacy row-label semantics can leak into
    the new manifest.
    """
    path = Path(out_dir) / PROGRESS_FILENAME
    if not path.exists():
        return None
    try:
        return WorkerProgress.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception:  # noqa: BLE001 - corrupt progress must never crash a resume.
        return None


def _confirmed_shard_paths(out_dir: Path, *, upto_index: int) -> list[Path]:
    """Shard files with index < `upto_index`, in index order.

    Used on resume to seed a fresh `GumbelShardWriter.paths` with the prior
    session's already-confirmed shards, so the eventual `manifest.json`
    still lists every shard (not just the ones this session created).
    """
    found: list[tuple[int, Path]] = []
    for path in Path(out_dir).glob("gumbel_self_play_shard_*.npz*"):
        stem = path.name.split(".", 1)[0]
        try:
            index = int(stem.rsplit("_", 1)[-1])
        except ValueError:
            continue
        if index < upto_index:
            found.append((index, path))
    return [path for _index, path in sorted(found)]


def _resume_shard_evidence(path: Path, *, index: int) -> dict[str, Any]:
    """Authenticate one retained shard and bind its exact on-disk identity."""

    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if path.name.endswith(".zst"):
            try:
                import zstandard
            except ImportError as error:
                raise ValueError(
                    "zstandard is required to authenticate retained resume shards"
                ) from error
            with path.open("rb") as source:
                with zstandard.ZstdDecompressor().stream_reader(source) as reader:
                    payload = reader.read()
            archive_context = np.load(io.BytesIO(payload), allow_pickle=False)
        else:
            archive_context = np.load(path, allow_pickle=False)
        with archive_context as archive:
            if (
                "game_seed" not in archive
                or AUX_SUBGOAL_TARGET_VERSION_KEY not in archive
            ):
                raise ValueError(
                    f"retained shard lacks resume authority columns: {path}"
                )
            seeds = np.asarray(archive["game_seed"])
            versions = np.asarray(archive[AUX_SUBGOAL_TARGET_VERSION_KEY])
            if seeds.ndim != 1 or versions.shape != seeds.shape or seeds.size <= 0:
                raise ValueError(f"retained shard row-shape drift: {path}")
            if seeds.dtype.kind not in {"i", "u"}:
                raise ValueError(f"retained shard game-seed dtype drift: {path}")
            if versions.dtype.kind not in {"i", "u"} or bool(
                np.any(
                    versions.astype(np.int64, copy=False) != AUX_SUBGOAL_TARGET_VERSION
                )
            ):
                raise ValueError(f"retained shard aux semantic drift: {path}")
            canonical_seeds = seeds.astype("<i8", copy=False)
            return {
                "index": int(index),
                "filename": path.name,
                "rows": int(seeds.size),
                "size_bytes": int(path.stat().st_size),
                "sha256": digest.hexdigest(),
                "game_seed_sha256": hashlib.sha256(
                    canonical_seeds.tobytes(order="C")
                ).hexdigest(),
                "game_seed_min": int(canonical_seeds.min()),
                "game_seed_max": int(canonical_seeds.max()),
                "unique_game_seeds": int(np.unique(canonical_seeds).size),
            }
    except ValueError:
        raise
    except Exception as error:  # noqa: BLE001 - any unreadable shard loses resume authority.
        raise ValueError(
            f"cannot authenticate retained resume shard {path}: {error}"
        ) from error


def _validated_confirmed_shard_paths(
    out_dir: Path,
    *,
    upto_index: int,
    expected_rows: int,
    expected_inventory: Sequence[Mapping[str, Any]],
) -> tuple[list[Path], list[dict[str, Any]]]:
    """Require a contiguous, unique retained inventory with exact row mass."""

    paths = _confirmed_shard_paths(out_dir, upto_index=upto_index)
    indices = [int(path.name.split(".", 1)[0].rsplit("_", 1)[-1]) for path in paths]
    if indices != list(range(int(upto_index))):
        raise ValueError(
            "retained resume shard inventory is not exactly contiguous: "
            f"expected={list(range(int(upto_index)))} observed={indices}"
        )
    observed_inventory = [
        _resume_shard_evidence(path, index=index) for index, path in enumerate(paths)
    ]
    rows = sum(record["rows"] for record in observed_inventory)
    if rows != int(expected_rows):
        raise ValueError(
            "retained resume shard row total differs from progress: "
            f"expected={expected_rows} observed={rows}"
        )
    if observed_inventory != [dict(record) for record in expected_inventory]:
        raise ValueError("retained resume shard identity differs from progress")
    return paths, observed_inventory


def _discard_orphan_shards(out_dir: Path, *, from_index: int) -> None:
    """Delete any shard files at or beyond `from_index`.

    In the ordinary resume path these can exist in a narrow race: a shard flush landed on the
    volume (via a periodic `volume.commit()`) slightly ahead of the very
    next `progress.json` write for that same shard. Because
    `shard_count_confirmed` is therefore the authoritative floor (never the
    physical file listing), any shard at/after it is, by definition, not
    yet confirmed and is about to be regenerated (deterministically, from
    the same `game_seed`s) and its index reused -- so it must be cleared
    first rather than silently appended to or shadowed.  The other caller is
    the schema-upgrade path: when `progress.json` is present but cannot prove
    the current row-label semantic, every old shard is invalidated by calling
    this helper with `from_index=0`.  Both deletions are scoped to one worker's
    `out_dir`; confirmed current-semantic data is never touched.
    """
    for path in sorted(Path(out_dir).glob("gumbel_self_play_shard_*.npz*")):
        stem = path.name.split(".", 1)[0]
        try:
            index = int(stem.rsplit("_", 1)[-1])
        except ValueError:
            continue
        if index >= from_index:
            path.unlink()


def _game_search_seed(*, worker_seed: int, game_index: int) -> int:
    """Derive one stable, independent MCTS RNG stream per absolute game."""

    payload = (
        f"gumbel-self-play-game-search-v1:{int(worker_seed)}:{int(game_index)}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _generation_resume_semantics_sha256(
    *,
    caller_contract_sha256: str | None,
    worker_seed: int,
    config: GumbelSelfPlayConfig,
    search_config: GumbelChanceMCTSConfig,
    preserve_search_evidence: bool,
    native_mcts_hot_loop: bool,
    target_information_regime: str,
    opponent_pool: OpponentPoolRuntime | None,
    opponent_mix: MixRuntime | None,
    public_award_feature_provenance: Mapping[str, Any] | None,
) -> str:
    """Bind every in-function science knob plus caller-authenticated model bytes."""

    if caller_contract_sha256 is not None and (
        not caller_contract_sha256.startswith("sha256:")
        or len(caller_contract_sha256) != 71
        or any(
            character not in "0123456789abcdef"
            for character in caller_contract_sha256.removeprefix("sha256:")
        )
    ):
        raise ValueError(
            "resume_semantics_sha256 must be a lowercase sha256:<64 hex> digest"
        )
    if opponent_pool is not None:
        archive_records: list[dict[str, Any]] = []
        for item in opponent_pool.archive:
            record = dataclasses.asdict(item)
            if caller_contract_sha256 is not None:
                checkpoint = Path(item.path)
                if checkpoint.is_symlink() or not checkpoint.is_file():
                    raise ValueError(
                        "resume-authorized opponent-pool checkpoint must be a "
                        f"regular non-symlink file: {checkpoint}"
                    )
                checkpoint_digest = hashlib.sha256()
                with checkpoint.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        checkpoint_digest.update(chunk)
                record["checkpoint_sha256"] = "sha256:" + checkpoint_digest.hexdigest()
                record["checkpoint_size_bytes"] = checkpoint.stat().st_size
            archive_records.append(record)
        opponent_descriptor: dict[str, Any] = {
            "mode": "pool",
            "policy": dataclasses.asdict(opponent_pool.policy),
            "champion": dataclasses.asdict(opponent_pool.champion),
            "archive": archive_records,
        }
    elif opponent_mix is not None:
        opponent_descriptor = {
            "mode": "mix",
            "config": dataclasses.asdict(opponent_mix.config),
        }
    else:
        opponent_descriptor = {"mode": "self_play"}
    semantics = {
        "schema_version": "gumbel-worker-resume-semantics-v1",
        "caller_contract_sha256": caller_contract_sha256,
        "worker_seed": int(worker_seed),
        "game_search_seed_derivation": "sha256(worker_seed,absolute_game_index)-u64-v1",
        "selfplay_config": dataclasses.asdict(config),
        "search_config": dataclasses.asdict(search_config),
        "preserve_search_evidence": bool(preserve_search_evidence),
        "native_mcts_hot_loop": bool(native_mcts_hot_loop),
        "target_information_regime": str(target_information_regime),
        "opponent": opponent_descriptor,
        "public_award_feature_provenance": (
            None
            if public_award_feature_provenance is None
            else dict(public_award_feature_provenance)
        ),
    }
    encoded = json.dumps(
        semantics,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def run_worker_games(
    *,
    out_dir: Path,
    games: int,
    game_index_start: int,
    base_seed: int,
    worker_seed: int,
    config: GumbelSelfPlayConfig,
    search_config: GumbelChanceMCTSConfig,
    evaluator: RustEvaluator,
    shard_size: int = 2048,
    fmt: str = "npz",
    preserve_search_evidence: bool = False,
    run_id: str = "",
    resume: bool = False,
    resume_semantics_sha256: str | None = None,
    opponent_pool: OpponentPoolRuntime | None = None,
    opponent_mix: MixRuntime | None = None,
    native_mcts_hot_loop: bool = False,
    public_award_feature_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Play `games` self-play games in this process, writing shards to `out_dir`.

    Constructs exactly one `GumbelChanceMCTS`, reuses it across the worker, and
    resets its RNG at each game boundary from a stable hash of
    `(worker_seed, absolute game_index)`. This preserves an advancing stream
    within each game while making failures/preemptions in earlier games unable
    to perturb later targets. It also provides per-game exception isolation
    (one bad game is recorded and skipped, not fatal to the worker), and a
    per-worker `manifest.json` compatible with `tools/train_bc.py`'s loader
    (a top-level `shards` list).

    Incremental resume (`resume=True`): a current progress contract must match
    `run_id`, seed interval, requested games, shard target/format, aux semantic,
    an exact contiguous shard inventory/row total, and a caller-authenticated
    generation-semantics digest. The caller digest must bind immutable producer
    and opponent bytes; this function folds it together with every inspectable
    self-play/search/opponent knob. Only then does this
    call skip games recorded as durably flushed and continue the SAME `offset`
    loop from `games_completed_local` -- NOT from a recomputed
    `game_index_start`.  Shards are game-atomic, so the first replayed game is
    wholly absent from retained files. Seed identity proof: `game_seed =
    base_seed + game_index_start + offset` is a pure function of `offset`
    with `base_seed`/`game_index_start` both caller-supplied constants that
    a resume never changes (they come from the payload, not from
    `progress.json`); resuming only changes the START value of `offset` in
    `range(...)`, so every game that gets (re)played -- whether in the
    original attempt or a resumed one -- gets EXACTLY the same `game_seed`
    it would have gotten in an uninterrupted run. When `resume=False` (the
    default) or no progress file exists, this call behaves exactly as
    before resume support was added, aside from also writing
    `progress.json` as a side effect.  Missing/stale authority causes a
    deterministic replay from zero after worker-local orphan shards are
    removed.

    `opponent_pool` (H2, default None = OFF, exact prior behavior): when set,
    each game_index deterministically draws (via `opponent_pool.policy`/
    `choose_opponent`, hash-of-game_index -- not a global RNG, so a
    crashed-and-resumed worker redraws identically) whether it is a pool game
    and, if so, which archived opponent checkpoint plays the non-champion
    seat (`opponent_pool.evaluator_factory` is called and cached by checkpoint
    path -- loaded once per worker process, not once per game). A second,
    independently-salted hash bit decides which of `config.colors` the
    champion occupies (color balance across pool games). Interacts cleanly
    with resume: `pool_assignment` is recomputed per `offset` purely from
    `game_index` (see `_pool_champion_plays_first_seat`/`choose_opponent`'s
    own hashing), so replaying an offset on resume reproduces the identical
    pool/mirror decision and, if it's a pool game, the identical opponent
    version -- nothing about pool assignment depends on run history/RNG
    state.

    `opponent_mix` (CAT-54, default None = OFF, exact prior behavior):
    generalizes `opponent_pool` from one binary "mirror vs one archived
    opponent" fraction to an arbitrary N-category weighted mix (e.g. the
    adopted 75% producer self-play / 10% previous+public champion / 5% older
    champion / 5% hard-experimental split -- see
    `catan_zero.rl.flywheel.opponent_mix`). Mutually exclusive with
    `opponent_pool` (pass at most one; both configure the exact same
    `pool_assignment`/`PoolGameAssignment` plumbing `play_one_game` already
    consumes, so there is no reason to run both at once and doing so would
    make it ambiguous which policy governs a given `game_index`). Same
    resume-safety and per-worker evaluator caching as `opponent_pool`.
    """
    if opponent_pool is not None and opponent_mix is not None:
        raise ValueError(
            "run_worker_games got both opponent_pool and opponent_mix -- pass at most one "
            "(they both resolve the same PoolGameAssignment; running both is ambiguous)"
        )
    reliability_fraction = float(config.target_reliability_audit_fraction)
    if (
        not math.isfinite(reliability_fraction)
        or not 0.0 <= reliability_fraction <= 1.0
    ):
        raise ValueError("target reliability audit fraction must be in [0, 1]")
    if reliability_fraction > 0.0:
        if not bool(search_config.coherent_public_belief_search):
            raise ValueError(
                "target reliability audit requires coherent_public_belief_search"
            )
        if int(search_config.n_full) != 128:
            raise ValueError(
                "target reliability audit is contracted for n_full=128, got "
                f"{search_config.n_full!r}"
            )
    if resume and resume_semantics_sha256 is None:
        raise ValueError(
            "resume=True requires resume_semantics_sha256 binding immutable "
            "producer/opponent bytes and the caller's generation contract"
        )
    search_evidence_scope = (
        _search_evidence_recalibration_scope(search_config)
        if preserve_search_evidence
        else None
    )
    out_dir = Path(out_dir)
    action_size = action_size_for_evaluator(evaluator, config.colors)
    entity_feature_adapter_version = entity_adapter_for_evaluator(evaluator)
    engine_supports_determinization = False
    engine_supports_public_belief_development_draws = False
    if bool(getattr(search_config, "information_set_search", False)) or bool(
        getattr(search_config, "coherent_public_belief_search", False)
    ):
        catanatron_rs = _gumbel_chance_mcts._require_rust_module()
        engine_supports_determinization = hasattr(
            catanatron_rs.Game, "determinize_for_player"
        )
        engine_supports_public_belief_development_draws = hasattr(
            catanatron_rs.Game, "apply_public_belief_development_draws"
        )
    target_information_regime = _target_information_regime_for_search(
        search_config,
        engine_supports_determinization=engine_supports_determinization,
        engine_supports_public_belief_development_draws=(
            engine_supports_public_belief_development_draws
        ),
    )
    generation_semantics_sha256 = _generation_resume_semantics_sha256(
        caller_contract_sha256=resume_semantics_sha256,
        worker_seed=int(worker_seed),
        config=config,
        search_config=search_config,
        preserve_search_evidence=bool(preserve_search_evidence),
        native_mcts_hot_loop=bool(native_mcts_hot_loop),
        target_information_regime=target_information_regime,
        opponent_pool=opponent_pool,
        opponent_mix=opponent_mix,
        public_award_feature_provenance=public_award_feature_provenance,
    )
    mcts = create_gumbel_search(
        search_config,
        evaluator,
        native_hot_loop=bool(native_mcts_hot_loop),
    )
    target_reliability_mcts: GumbelChanceMCTS | None = None
    target_reliability_manifest = target_reliability_contract(
        audit_fraction=reliability_fraction,
        audit_seed=int(config.target_reliability_audit_seed),
    )
    if reliability_fraction > 0.0:
        # The replica is deliberately the feature-complete Python reference:
        # this gives it explicit Gumbel/chance/belief substreams even when the
        # primary trajectory uses the parity-gated native hot loop. Preserve
        # the primary exact/legacy SH and wide-budget semantics: otherwise the
        # duplicate would audit a different target operator. Per-root
        # eligibility below admits only roots whose nominal full budget is 128.
        reliability_search_config = dataclasses.replace(
            search_config,
            seed=int(config.target_reliability_audit_seed),
            n_full=128,
            n_fast=128,
            p_full=1.0,
            raw_policy_above_width=None,
            temperature=0.0,
            rng_stream_separation=True,
        )
        target_reliability_mcts = GumbelChanceMCTS(reliability_search_config, evaluator)
        assert target_reliability_manifest is not None
        target_reliability_manifest["duplicate_search_config"] = dataclasses.asdict(
            reliability_search_config
        )

    resume_offset = 0
    games_completed = 0
    games_failed = 0
    games_truncated = 0
    rows = 0
    decisions_total = 0
    forced_decisions_total = 0
    simulations_used_total = 0
    wins_by_color: dict[str, int] = {color: 0 for color in config.colors}
    errors: list[dict[str, Any]] = []
    start_shard_index = 0
    # games_completed_local as of the last progress checkpoint (durable
    # floor); advanced during this call as new flushes catch up to
    # previously-played-but-unflushed games (see `pending_boundaries` below).
    games_completed_local = 0
    confirmed_paths: list[Path] = []
    confirmed_shard_records: list[dict[str, Any]] = []

    # Every scientific telemetry aggregate is part of the same confirmed
    # snapshot as rows/games.  Initializing these before resume lets a
    # preempted opponent-mix run restore its full history instead of emitting
    # a final manifest that describes only the post-resume suffix.
    pool_games_completed = 0
    pool_version_stats: dict[int, dict[str, int]] = {}
    mix_games_completed = 0
    mix_tag_stats: dict[str, dict[str, int]] = {}
    exploiter_games_completed = 0
    exploiter_engine_stats: dict[str, dict[str, int]] = {}
    exploiter_divergence_topics: dict[str, int] = {}

    if resume:
        progress_path = out_dir / PROGRESS_FILENAME
        progress = _load_worker_progress(out_dir)
        progress_matches = bool(
            progress is not None
            and progress.run_id == run_id
            and progress.generation_semantics_sha256 == generation_semantics_sha256
            and progress.base_seed == int(base_seed)
            and progress.game_index_start == int(game_index_start)
            and progress.games_requested == int(games)
            and progress.shard_size == int(shard_size)
            and progress.shard_format == str(fmt)
            and 0 <= progress.games_completed_local <= int(games)
            and 0 <= progress.games_succeeded <= progress.games_completed_local
            and progress.shard_count_confirmed >= 0
            and progress.rows_confirmed >= 0
            and progress.rows == progress.rows_confirmed
            and set(progress.wins_by_color) == set(config.colors)
        )
        if progress_matches:
            assert progress is not None
            try:
                confirmed_paths, confirmed_shard_records = (
                    _validated_confirmed_shard_paths(
                        out_dir,
                        upto_index=int(progress.shard_count_confirmed),
                        expected_rows=int(progress.rows_confirmed),
                        expected_inventory=progress.confirmed_shards,
                    )
                )
            except Exception:  # noqa: BLE001 - any corrupt retained shard replays safely.
                progress_matches = False
        if progress_matches:
            assert progress is not None
            resume_offset = int(progress.games_completed_local)
            games_completed_local = resume_offset
            games_completed = int(progress.games_succeeded)
            games_failed = int(progress.games_failed)
            games_truncated = int(progress.games_truncated)
            rows = int(progress.rows)
            decisions_total = int(progress.decisions_total)
            forced_decisions_total = int(progress.forced_decisions_total)
            simulations_used_total = int(progress.simulations_used_total)
            for color, count in progress.wins_by_color.items():
                wins_by_color[color] = wins_by_color.get(color, 0) + int(count)
            errors = [dict(error) for error in progress.errors]
            pool_games_completed = int(progress.opponent_pool_games)
            pool_version_stats = {
                int(version): dict(stats)
                for version, stats in progress.opponent_pool_per_version_stats.items()
            }
            mix_games_completed = int(progress.opponent_mix_pool_games)
            mix_tag_stats = {
                str(tag): dict(stats)
                for tag, stats in progress.opponent_mix_per_tag_stats.items()
            }
            exploiter_games_completed = int(progress.exploiter_games)
            exploiter_engine_stats = {
                str(engine): dict(stats)
                for engine, stats in progress.exploiter_per_engine_stats.items()
            }
            exploiter_divergence_topics = dict(progress.exploiter_divergence_topics)
            start_shard_index = int(progress.shard_count_confirmed)
            _discard_orphan_shards(out_dir, from_index=start_shard_index)
        else:
            # A progress marker that cannot be authenticated for this run and
            # aux-label semantic must not lend its old shards to a new v1
            # manifest.  The same applies to shard files with no progress
            # authority. Replaying from zero is deterministic and safe.
            _discard_orphan_shards(out_dir, from_index=0)
            confirmed_shard_records = []

    writer = GumbelShardWriter(
        out_dir,
        shard_size=shard_size,
        fmt=fmt,
        start_index=start_shard_index,
        rows_written=rows,
        preserve_search_evidence=preserve_search_evidence,
    )
    if start_shard_index:
        # Seed `writer.paths` with the prior session's already-confirmed
        # shards so the manifest this call eventually writes still lists
        # ALL shards, not just the ones flushed in this (resumed) session.
        writer.paths.extend(confirmed_paths)
    progress_path = out_dir / PROGRESS_FILENAME
    # Absolute (cross-session) cumulative row count already durable, per the
    # last confirmed shard count -- the baseline every new boundary is
    # measured against.
    absolute_rows = rows

    def _aggregate_snapshot() -> dict[str, Any]:
        return {
            "games_succeeded": int(games_completed),
            "games_failed": int(games_failed),
            "games_truncated": int(games_truncated),
            "rows": int(rows),
            "decisions_total": int(decisions_total),
            "forced_decisions_total": int(forced_decisions_total),
            "simulations_used_total": int(simulations_used_total),
            "wins_by_color": dict(wins_by_color),
            "errors": copy.deepcopy(errors),
            "opponent_pool_games": int(pool_games_completed),
            "opponent_pool_per_version_stats": copy.deepcopy(pool_version_stats),
            "opponent_mix_pool_games": int(mix_games_completed),
            "opponent_mix_per_tag_stats": copy.deepcopy(mix_tag_stats),
            "exploiter_games": int(exploiter_games_completed),
            "exploiter_per_engine_stats": copy.deepcopy(exploiter_engine_stats),
            "exploiter_divergence_topics": dict(exploiter_divergence_topics),
        }

    confirmed_snapshot = _aggregate_snapshot()
    # (offset, absolute_row_count_after_this_offset, cumulative aggregates)
    # for offsets played THIS session but not yet confirmed durable.  Since
    # add_game never splits a game, any completed shard boundary coincides
    # with one of these snapshots.
    pending_boundaries: list[tuple[int, int, dict[str, Any]]] = []

    def _confirm_and_checkpoint() -> None:
        nonlocal games_completed_local, confirmed_snapshot, confirmed_shard_records
        flushed_rows = writer.rows_written
        while pending_boundaries and pending_boundaries[0][1] <= flushed_rows:
            offset_done, _boundary, snapshot = pending_boundaries.pop(0)
            games_completed_local = offset_done + 1
            confirmed_snapshot = snapshot
        if int(confirmed_snapshot["rows"]) != int(flushed_rows):
            raise RuntimeError(
                "resume invariant violated: confirmed aggregate rows differ from "
                f"written shard rows ({confirmed_snapshot['rows']} != {flushed_rows})"
            )
        if len(writer.paths) != writer.index:
            raise RuntimeError(
                "resume invariant violated: writer path inventory differs from shard index"
            )
        if len(confirmed_shard_records) > writer.index:
            raise RuntimeError(
                "resume invariant violated: confirmed shard inventory regressed"
            )
        for index in range(len(confirmed_shard_records), writer.index):
            confirmed_shard_records.append(
                _resume_shard_evidence(writer.paths[index], index=index)
            )
        _write_json_atomic(
            progress_path,
            WorkerProgress(
                run_id=run_id,
                resume_contract_version=WORKER_PROGRESS_RESUME_CONTRACT_VERSION,
                generation_semantics_sha256=generation_semantics_sha256,
                aux_subgoal_target_version=AUX_SUBGOAL_TARGET_VERSION,
                aux_subgoal_target_semantic=AUX_SUBGOAL_TARGET_SEMANTIC,
                shard_size=int(shard_size),
                shard_format=str(fmt),
                base_seed=int(base_seed),
                game_index_start=int(game_index_start),
                games_requested=int(games),
                games_completed_local=games_completed_local,
                games_succeeded=int(confirmed_snapshot["games_succeeded"]),
                shard_count_confirmed=writer.index,
                rows_confirmed=flushed_rows,
                confirmed_shards=copy.deepcopy(confirmed_shard_records),
                games_failed=int(confirmed_snapshot["games_failed"]),
                games_truncated=int(confirmed_snapshot["games_truncated"]),
                rows=int(confirmed_snapshot["rows"]),
                decisions_total=int(confirmed_snapshot["decisions_total"]),
                forced_decisions_total=int(
                    confirmed_snapshot["forced_decisions_total"]
                ),
                simulations_used_total=int(
                    confirmed_snapshot["simulations_used_total"]
                ),
                wins_by_color=dict(confirmed_snapshot["wins_by_color"]),
                errors=copy.deepcopy(confirmed_snapshot["errors"]),
                opponent_pool_games=int(confirmed_snapshot["opponent_pool_games"]),
                opponent_pool_per_version_stats={
                    str(version): dict(stats)
                    for version, stats in confirmed_snapshot[
                        "opponent_pool_per_version_stats"
                    ].items()
                },
                opponent_mix_pool_games=int(
                    confirmed_snapshot["opponent_mix_pool_games"]
                ),
                opponent_mix_per_tag_stats=copy.deepcopy(
                    confirmed_snapshot["opponent_mix_per_tag_stats"]
                ),
                exploiter_games=int(confirmed_snapshot["exploiter_games"]),
                exploiter_per_engine_stats=copy.deepcopy(
                    confirmed_snapshot["exploiter_per_engine_stats"]
                ),
                exploiter_divergence_topics=dict(
                    confirmed_snapshot["exploiter_divergence_topics"]
                ),
            ).to_dict(),
        )

    pool_evaluator_cache: dict[str, RustEvaluator] = {}
    mix_evaluator_cache: dict[str, RustEvaluator] = {}
    # Keyed by category tag (not version): telemetry is "tracked separately
    # per opponent" per the ticket, and a tag can span several checkpoint
    # versions (e.g. "older_champion" sampling across many archived nets).

    started = time.perf_counter()
    try:
        for offset in range(resume_offset, int(games)):
            game_index = int(game_index_start) + offset
            game_seed = int(base_seed) + game_index
            # A game's search randomness is independent of every earlier
            # game's draw count/failure history. Resume can therefore jump to
            # any confirmed offset without resetting the suffix's Gumbel or
            # chance-sampling stream relative to an uninterrupted worker.
            mcts.seed_search_rngs(
                _game_search_seed(
                    worker_seed=int(worker_seed), game_index=int(game_index)
                )
            )

            pool_assignment: PoolGameAssignment | None = None
            # CAT-56: set to (engine_name, champion_first, tag) when this game is
            # an exploiter-lane game (external Catanatron bot opponent), routed to
            # the cross-engine lockstep instead of the single-engine play_one_game.
            exploiter_spec: tuple[str, bool, str] | None = None
            if opponent_pool is not None:
                choice = choose_opponent(
                    game_index,
                    opponent_pool.champion,
                    opponent_pool.archive,
                    opponent_pool.policy,
                )
                champion_first = _pool_champion_plays_first_seat(game_index)
                champion_color = (
                    config.colors[0] if champion_first else config.colors[1]
                )
                if choice.is_pool:
                    opponent_color = (
                        config.colors[1] if champion_first else config.colors[0]
                    )
                    opponent_evaluator = pool_evaluator_cache.get(choice.path)
                    if opponent_evaluator is None:
                        opponent_evaluator = opponent_pool.evaluator_factory(
                            choice.path
                        )
                        pool_evaluator_cache[choice.path] = opponent_evaluator
                    pool_assignment = PoolGameAssignment(
                        is_pool=True,
                        champion_color=champion_color,
                        opponent_color=opponent_color,
                        opponent_version=int(choice.version),
                        opponent_evaluator=opponent_evaluator,
                    )
                else:
                    pool_assignment = PoolGameAssignment(
                        is_pool=False,
                        champion_color=champion_color,
                        opponent_color=None,
                        opponent_version=-1,
                        opponent_evaluator=None,
                    )
            elif opponent_mix is not None:
                # Same deterministic-hash-of-game_index / independent seat-balance
                # bit as the H2 binary path above (`_pool_champion_plays_first_seat`
                # uses its own salt, unaffected by which policy chose the
                # opponent), so mix assignment is equally resume-safe.
                mix_choice = choose_mix_opponent(
                    game_index, opponent_mix.config.categories
                )
                champion_first = _pool_champion_plays_first_seat(game_index)
                champion_color = (
                    config.colors[0] if champion_first else config.colors[1]
                )
                if mix_choice.is_external:
                    # Exploiter lane: no checkpoint to load; the external bot
                    # plays the opponent seat in cross-engine lockstep. Defer the
                    # actual game to play_one_exploiter_game below; pool_assignment
                    # stays None (that path builds its own seat routing).
                    exploiter_spec = (mix_choice.engine, champion_first, mix_choice.tag)
                elif mix_choice.is_pool:
                    opponent_color = (
                        config.colors[1] if champion_first else config.colors[0]
                    )
                    opponent_evaluator = mix_evaluator_cache.get(mix_choice.path)
                    if opponent_evaluator is None:
                        opponent_evaluator = opponent_mix.evaluator_factory(
                            mix_choice.path
                        )
                        mix_evaluator_cache[mix_choice.path] = opponent_evaluator
                    pool_assignment = PoolGameAssignment(
                        is_pool=True,
                        champion_color=champion_color,
                        opponent_color=opponent_color,
                        opponent_version=int(mix_choice.version),
                        opponent_evaluator=opponent_evaluator,
                        tag=mix_choice.tag,
                        opponent_md5=mix_choice.md5,
                    )
                else:
                    pool_assignment = PoolGameAssignment(
                        is_pool=False,
                        champion_color=champion_color,
                        opponent_color=None,
                        opponent_version=-1,
                        opponent_evaluator=None,
                        tag=mix_choice.tag,
                        opponent_md5="",
                    )

            try:
                if exploiter_spec is not None:
                    # Lazy import: keeps the default self-play / neural-mix path
                    # free of any catanatron (Python engine) import -- it is only
                    # pulled in when an exploiter game actually runs.
                    from catan_zero.rl.exploiter_lockstep import play_one_exploiter_game

                    engine_name, champion_first, exploiter_tag = exploiter_spec
                    record = play_one_exploiter_game(
                        evaluator=evaluator,
                        search_config=search_config,
                        config=config,
                        game_seed=game_seed,
                        game_index=game_index,
                        engine=engine_name,
                        champion_first=champion_first,
                        opponent_tag=exploiter_tag,
                    )
                else:
                    record = play_one_game(
                        mcts,
                        evaluator,
                        config=config,
                        game_seed=game_seed,
                        game_index=game_index,
                        action_size=action_size,
                        pool_assignment=pool_assignment,
                        target_reliability_mcts=target_reliability_mcts,
                    )
            except Exception as error:  # noqa: BLE001 - isolate one bad game from the worker.
                games_failed += 1
                errors.append(
                    {
                        "game_index": game_index,
                        "game_seed": game_seed,
                        "error": repr(error),
                    }
                )
                pending_boundaries.append(
                    (offset, absolute_rows, _aggregate_snapshot())
                )
                _confirm_and_checkpoint()
                continue

            writer.add_game(record.decisions)
            rows += len(record.decisions)
            absolute_rows += len(record.decisions)
            decisions_total += record.total_decisions
            forced_decisions_total += record.forced_decisions
            simulations_used_total += record.simulations_used_total
            games_completed += 1
            if record.truncated:
                games_truncated += 1
            if record.terminal and record.winner in wins_by_color:
                wins_by_color[record.winner] += 1

            if exploiter_spec is not None:
                # Exploiter-lane telemetry (CAT-56), per external engine. A
                # divergence-dropped game (engine_divergence, no rows) is counted
                # under `divergences`/its topic, NOT as a graded game.
                engine_name, champion_first, _tag = exploiter_spec
                champion_color = (
                    config.colors[0] if champion_first else config.colors[1]
                )
                if record.engine_divergence:
                    topic = record.divergence_topic or "unclassified"
                    exploiter_divergence_topics[topic] = (
                        exploiter_divergence_topics.get(topic, 0) + 1
                    )
                    stats = exploiter_engine_stats.setdefault(
                        engine_name, {"games": 0, "champion_wins": 0, "divergences": 0}
                    )
                    stats["divergences"] += 1
                else:
                    exploiter_games_completed += 1
                    stats = exploiter_engine_stats.setdefault(
                        engine_name, {"games": 0, "champion_wins": 0, "divergences": 0}
                    )
                    stats["games"] += 1
                    if record.terminal and record.winner == champion_color:
                        stats["champion_wins"] += 1
            elif (
                opponent_pool is not None
                and pool_assignment is not None
                and pool_assignment.is_pool
            ):
                pool_games_completed += 1
                stats = pool_version_stats.setdefault(
                    pool_assignment.opponent_version, {"games": 0, "champion_wins": 0}
                )
                stats["games"] += 1
                if record.terminal and record.winner == pool_assignment.champion_color:
                    stats["champion_wins"] += 1
            elif opponent_mix is not None and pool_assignment is not None:
                # Unlike the H2 binary path, mix telemetry is tracked for EVERY
                # category including "producer_self_play" (mirror) -- the
                # ticket asks for per-opponent win-rate/etc "tracked separately
                # per opponent" across the whole mix, not just the pool slice.
                if pool_assignment.is_pool:
                    mix_games_completed += 1
                stats = mix_tag_stats.setdefault(
                    pool_assignment.tag, {"games": 0, "champion_wins": 0}
                )
                stats["games"] += 1
                if record.terminal and record.winner == pool_assignment.champion_color:
                    stats["champion_wins"] += 1

            pending_boundaries.append((offset, absolute_rows, _aggregate_snapshot()))
            _confirm_and_checkpoint()
    finally:
        writer.close()
        # `close()` may publish the final partial shard.  The following atomic
        # progress update either lands and authenticates it, or a crash between
        # the two operations leaves it above the prior shard floor so resume
        # deletes and deterministically replays it.
        _confirm_and_checkpoint()

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
        "simulations_used_total": int(simulations_used_total),
        "search_evidence_schema": (
            SEARCH_EVIDENCE_SCHEMA if preserve_search_evidence else None
        ),
        "search_evidence_recalibration_scope": search_evidence_scope,
        "worker_seed": int(worker_seed),
        "base_seed": int(base_seed),
        "game_index_start": int(game_index_start),
        "adapter_version": entity_feature_adapter_version,
        # Full provenance of what this worker ACTUALLY constructed (not what
        # the CLI was asked for): catches argparse-default-vs-dataclass-default
        # divergence after the fact -- the exact class of gap behind the
        # c_scale 1.0-vs-0.1 near-miss (commit 376c146).
        "selfplay_config": dataclasses.asdict(config),
        "search_config": dataclasses.asdict(search_config),
        "search_execution_contract": _search_execution_contract(
            search_config, native_mcts_hot_loop=bool(native_mcts_hot_loop)
        ),
        "search_rng_contract": {
            "schema_version": "gumbel-game-search-rng-v1",
            "worker_seed": int(worker_seed),
            "derivation": "sha256(worker_seed,absolute_game_index)-u64-v1",
            "resume_invariant": True,
        },
        "target_information_regime": target_information_regime,
        "target_reliability_contract": target_reliability_manifest,
        AUX_SUBGOAL_TARGET_VERSION_KEY: AUX_SUBGOAL_TARGET_VERSION,
        "aux_subgoal_target_semantic": AUX_SUBGOAL_TARGET_SEMANTIC,
        "elapsed_sec": elapsed,
        "rows_per_sec": rows / max(elapsed, 1.0e-9),
        "shards": [str(path) for path in writer.paths],
        "errors": errors,
        "resumed_from_offset": int(resume_offset),
        "generation_resume_semantics_sha256": generation_semantics_sha256,
    }
    if public_award_feature_provenance is not None:
        summary["public_award_feature_provenance"] = dict(
            public_award_feature_provenance
        )
    if opponent_pool is not None:
        summary["opponent_pool_enabled"] = True
        summary["opponent_pool_fraction_configured"] = float(
            opponent_pool.policy.pool_fraction
        )
        summary["opponent_pool_games"] = int(pool_games_completed)
        summary["opponent_pool_fraction_realized"] = (
            pool_games_completed / games_completed if games_completed else 0.0
        )
        # Raw (games, champion_wins) per opponent version -- NOT a pre-divided
        # win-rate -- so the multi-worker merge in
        # tools/generate_gumbel_selfplay_data.py can sum-then-divide instead of
        # averaging per-worker ratios (which would silently mis-weight workers
        # that drew different numbers of pool games per opponent version).
        summary["opponent_pool_per_version_stats"] = {
            str(version): dict(stats)
            for version, stats in sorted(pool_version_stats.items())
        }
    else:
        summary["opponent_pool_enabled"] = False
    if opponent_mix is not None:
        summary["opponent_mix_enabled"] = True
        summary["opponent_mix_effective_weights"] = (
            opponent_mix.config.effective_weights()
        )
        summary["opponent_mix_pool_games"] = int(mix_games_completed)
        summary["opponent_mix_pool_fraction_realized"] = (
            mix_games_completed / games_completed if games_completed else 0.0
        )
        # Raw (games, champion_wins) per CATEGORY TAG -- not a pre-divided
        # win-rate -- for the same sum-then-divide multi-worker merge reason
        # as opponent_pool_per_version_stats above.
        summary["opponent_mix_per_tag_stats"] = {
            tag: dict(stats) for tag, stats in sorted(mix_tag_stats.items())
        }
        # Exploiter lane (CAT-56): only meaningful under a mix that has an
        # external-engine category, but always emitted (empty when none) so the
        # manifest schema is uniform. `exploiter_engine_stats` is raw
        # (games, champion_wins, divergences) per engine -- NOT a pre-divided
        # win-rate -- for the same sum-then-divide multi-worker merge reason as
        # opponent_mix_per_tag_stats.
        summary["exploiter_enabled"] = bool(exploiter_engine_stats)
        summary["exploiter_games"] = int(exploiter_games_completed)
        summary["exploiter_per_engine_stats"] = {
            engine: dict(stats)
            for engine, stats in sorted(exploiter_engine_stats.items())
        }
        summary["exploiter_divergence_topics"] = dict(
            sorted(exploiter_divergence_topics.items())
        )
    else:
        summary["opponent_mix_enabled"] = False
    _write_json_atomic(Path(out_dir) / "manifest.json", summary)
    return summary


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    _fsync_directory(path.parent)
