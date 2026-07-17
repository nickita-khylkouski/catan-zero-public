#!/usr/bin/env python3
"""Restart self-play from archived high-regret states (task #64; Go-Exploit / RGSC).

DESIGN + SKELETON. Generates raw-policy self-play games that START from a
reconstructed archived state (both seats raw policy, value rows only,
policy_weight=0 -- identical schema/convention to `raw_selfplay`), instead of
always from the initial board. This concentrates fresh true-outcome value
samples on the states the agent evaluated worst (Go-Exploit, arXiv 2302.12359;
Regret-Guided Search Control, arXiv 2602.20809).

Mixing recipe (see `docs/regret_restart_mixing_recipe.md`), realised by
`plan_start_mix`:
  * 60% normal starts (fresh initial board) -- anti-forgetting / on-distribution,
  * 20% high-regret opening placements,
  * 10% robber/dev (chance-heavy) states,
  * 10% random archived states (smoothing; guards against over-fitting the
    regret metric's own blind spots).

DAGS caveat (arXiv 2605.14379): intermediate-start training can bias learning
in imperfect-info games. Two mitigations here: (1) reconstruction replays the
game's TRUE history, so a restart state is a legitimate reachable PUBLIC state,
not an omniscient fabrication; (2) every restart row carries an explicit
`start_mode` field (+ archived provenance) so training keeps separate metrics
for intermediate-start vs normal-start data and can down-weight or ablate it.

Reproducibility: the archived prefix is replayed with the archived game's own
chance stream (`reconstruct_state(..., return_rng=True)`); the continuation
keeps drawing from that same stream, so a branched game is reproducible from
(archived_game_seed, archived_decision_index, restart_select_seed).
`game_seed` identifies the new branched trajectory and is therefore set to the
unique `restart_select_seed`; the source reconstruction seed remains in
`archived_game_seed`. Reusing the source seed as the trajectory identity would
merge independent branches that can have different terminal winners.

SKELETON SCOPE: single-process, correctness-first. The per-worker
multiprocessing fan-out (mirroring `generate_raw_selfplay_data.py`) and the
Modal/fleet driver are intentionally left as a documented extension; the core
per-game primitives and the mixing planner are complete and tested.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from catan_zero.rl.gumbel_self_play import (
    GumbelShardWriter,
    TARGET_INFORMATION_REGIME_AUTHORITATIVE,
    TARGET_INFORMATION_REGIME_PUBLIC_COHERENT,
    _apply_selected_action,
    _game_outcome_fields,
    action_size_for_evaluator,
)
from catan_zero.rl.entity_feature_adapter import (
    RUST_ENTITY_ADAPTER_V5,
    RUST_ENTITY_ADAPTER_V6,
)
from catan_zero.rl.meaningful_history import MEANINGFUL_PUBLIC_HISTORY_SCHEMA_V2
from catan_zero.rl.raw_selfplay import (
    COLORS,
    RawSelfPlayConfig,
    _build_raw_decision_row,
    _select_action,
    play_one_raw_selfplay_game,
)
from catan_zero.rl.restart_provenance import RESTART_PROVENANCE_KEYS
from catan_zero.search.rust_mcts import RustEvaluator

from reconstruct_state import (
    gather_game_action_sequence,
    reconstruct_state,
)
from rgsc_sampler import (
    DEFAULT_RGSC_TEMPERATURE,
    rgsc_sample_indices,
)

TEACHER_NAME = "restart_selfplay"

# start_mode values written to every row.
START_NORMAL = "normal"
START_ARCHIVED = "archived_public_state"

# Restart columns added on top of the shared schema. They remain outside
# BASE/ENTITY so old shard writers stay byte-compatible, but the NPZ learner
# and memmap converter explicitly preserve them.
RESTART_KEYS = RESTART_PROVENANCE_KEYS


@dataclass(frozen=True, slots=True)
class RestartSelfPlayConfig:
    colors: tuple[str, ...] = COLORS
    map_kind: str | None = None
    track: str = "2p_no_trade"
    vps_to_win: int = 10
    obs_width: int = 806
    # Cap on TOTAL decisions per continuation (independent of how deep the
    # restart point already is).
    max_continuation_decisions: int = 600
    # Temperature-sample the first N decisions of the CONTINUATION for
    # branch diversity (like raw_selfplay.temperature_decisions but counted
    # from the restart point); argmax thereafter.
    restart_temperature_decisions: int = 20
    temperature: float = 1.0
    correct_rust_chance_spectra: bool = True
    meaningful_public_history: bool = True
    meaningful_public_history_schema: str = MEANINGFUL_PUBLIC_HISTORY_SCHEMA_V2
    event_history_limit: int = 64
    entity_feature_adapter_version: str = RUST_ENTITY_ADAPTER_V5
    target_information_regime: str = TARGET_INFORMATION_REGIME_AUTHORITATIVE


class RestartShardWriter(GumbelShardWriter):
    """GumbelShardWriter that also persists the RESTART_KEYS columns.

    `add` builds the same BASE/EXTRA/ENTITY payload the base writer does, plus
    the RESTART_KEYS scalars, into one payload dict; `_rows_to_arrays` only
    reads keys it knows about, so the restart columns ride along in the payload
    untouched and `flush` array-ises them separately.
    """

    def add(self, row: dict[str, Any], features: dict[str, np.ndarray]) -> None:
        from catan_zero.rl.gumbel_self_play import BASE_KEYS, ENTITY_KEYS, EXTRA_KEYS

        payload = {key: row[key] for key in BASE_KEYS if key in row}
        for key in EXTRA_KEYS:
            if key in row:
                payload[key] = row[key]
        for key in ENTITY_KEYS:
            payload[key] = features[key]
        for key in RESTART_KEYS:
            payload[key] = row.get(key, _default_restart(key))
        self.rows.append(payload)
        if len(self.rows) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        import os

        from catan_zero.rl.gumbel_self_play import _rows_to_arrays

        arrays = _rows_to_arrays(self.rows)  # ignores unknown RESTART_KEYS
        for key in RESTART_KEYS:
            arrays[key] = np.asarray([r[key] for r in self.rows])
        path = self.output / f"restart_self_play_shard_{self.index:05d}.npz"
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        self.paths.append(path)
        self.rows = []
        self.index += 1


def _default_restart(key: str) -> Any:
    if key == "restart_provenance_present":
        return True
    if key in ("archived_game_seed", "archived_decision_index", "restart_select_seed"):
        return np.int64(-1)
    return ""


def play_restart_game_from_state(
    evaluator: RustEvaluator,
    game: Any,
    *,
    config: RestartSelfPlayConfig,
    start_mode: str,
    start_bucket: str,
    archived_game_seed: int,
    archived_decision_index: int,
    restart_select_seed: int,
    action_size: int,
    chance_rng: random.Random,
) -> list[dict[str, Any]]:
    """Play a raw-policy continuation from a live `game`, returning built rows.

    `chance_rng` is the archived game's chance stream (already advanced to the
    restart point); the continuation keeps drawing from it. Action selection
    uses its own `restart_select_seed`-derived RNG so branches are independent
    of the archived selection RNG. Rows use decision_index numbered from 0 at
    the restart point (with archived_decision_index recording the true depth).
    """
    select_rng = random.Random((int(restart_select_seed) ^ 0x51ED270B) & 0xFFFFFFFF)
    colors = config.colors
    records: list[dict[str, Any]] = []
    features_list: list[dict[str, np.ndarray]] = []
    cont_index = 0
    terminal = False
    # Continuation temperature: absolute-count schedule from the restart point.
    temp_config = RawSelfPlayConfig(
        colors=colors,
        map_kind=config.map_kind,
        track=config.track,
        vps_to_win=config.vps_to_win,
        obs_width=config.obs_width,
        max_decisions=config.max_continuation_decisions,
        temperature_decisions=config.restart_temperature_decisions,
        temperature=config.temperature,
        correct_rust_chance_spectra=config.correct_rust_chance_spectra,
        meaningful_public_history=config.meaningful_public_history,
        meaningful_public_history_schema=config.meaningful_public_history_schema,
        event_history_limit=config.event_history_limit,
        entity_feature_adapter_version=config.entity_feature_adapter_version,
        target_information_regime=config.target_information_regime,
    )
    while cont_index < config.max_continuation_decisions:
        if game.winning_color() is not None:
            terminal = True
            break
        legal_rust = tuple(
            int(a) for a in game.playable_action_indices(list(colors), config.map_kind)
        )
        if not legal_rust:
            break
        acting_color = str(game.current_color())
        selected_action, priors = _select_action(
            evaluator,
            game,
            legal_rust,
            acting_color=acting_color,
            decision_index=cont_index,
            config=temp_config,
            rng=select_rng,
        )
        row, features = _build_raw_decision_row(
            game,
            selected_action=selected_action,
            priors=priors,
            action_size=action_size,
            colors=colors,
            # A restart is a new stochastic trajectory. Multiple selected
            # roots may come from one archived game and independent branches
            # can end with different winners, so the source reconstruction
            # seed cannot also be their learner game identity.
            game_seed=restart_select_seed,
            decision_index=cont_index,
            obs_width=config.obs_width,
            meaningful_public_history=bool(config.meaningful_public_history),
            meaningful_public_history_schema=str(
                config.meaningful_public_history_schema
            ),
            event_history_limit=int(config.event_history_limit),
            entity_feature_adapter_version=str(
                config.entity_feature_adapter_version
            ),
            target_information_regime=str(temp_config.target_information_regime),
        )
        # Tag provenance + start mode. teacher_name distinguishes this corpus.
        row["teacher_name"] = TEACHER_NAME
        row["restart_provenance_present"] = np.bool_(True)
        row["start_mode"] = start_mode
        row["start_bucket"] = start_bucket
        row["archived_game_seed"] = np.int64(archived_game_seed)
        row["archived_decision_index"] = np.int64(archived_decision_index)
        row["restart_select_seed"] = np.int64(restart_select_seed)
        records.append(row)
        features_list.append(features)
        game = _apply_selected_action(
            game,
            selected_action,
            colors=colors,
            rng=chance_rng,
            correct_rust_chance_spectra=config.correct_rust_chance_spectra,
        )
        cont_index += 1

    if not terminal:
        terminal = game.winning_color() is not None
    outcome = _game_outcome_fields(game, terminal=terminal, colors=colors)
    for row in records:
        row.update(outcome)
    return list(zip(records, features_list))


def plan_start_mix(
    n_games: int,
    *,
    normal: float = 0.60,
    opening: float = 0.20,
    robber_dev: float = 0.10,
    random_archived: float = 0.10,
) -> dict[str, int]:
    """Integer game counts per start bucket for a target total (recipe weights)."""
    total = float(normal + opening + robber_dev + random_archived)
    weights = {
        "normal": normal / total,
        "opening": opening / total,
        "robber_dev": robber_dev / total,
        "random_archived": random_archived / total,
    }
    counts = {k: int(n_games * w) for k, w in weights.items()}
    # Assign the rounding remainder to the largest bucket (normal).
    counts["normal"] += n_games - sum(counts.values())
    return counts


def plan_trajectory_seed_ranges(
    base_seed: int,
    counts: dict[str, int],
) -> dict[str, int]:
    """Allocate disjoint int64 game identities for normal and restart games."""

    required = {"normal", "opening", "robber_dev", "random_archived"}
    if set(counts) != required or any(
        isinstance(counts[key], bool) or int(counts[key]) < 0 for key in required
    ):
        raise ValueError("trajectory seed planning requires non-negative bucket counts")
    start = int(base_seed)
    normal_count = int(counts["normal"])
    archived_count = sum(int(counts[key]) for key in required - {"normal"})
    stop = start + normal_count + archived_count
    int64 = np.iinfo(np.int64)
    if start < 0 or stop - 1 > int(int64.max):
        raise ValueError("trajectory seed range does not fit non-negative int64")
    return {
        "normal_start": start,
        "normal_stop_exclusive": start + normal_count,
        "archived_start": start + normal_count,
        "archived_stop_exclusive": stop,
    }


def _bucket_of_phase(phase: str) -> str:
    up = str(phase).upper()
    if "BUILD_INITIAL_SETTLEMENT" in up or "BUILD_INITIAL_ROAD" in up:
        return "opening"
    if "MOVE_ROBBER" in up or "KNIGHT" in up or "DEVELOPMENT_CARD" in up:
        return "robber_dev"
    return "other"


def _stable_unit_hash(*parts: int) -> float:
    """Deterministic value in [0, 1) from integer `parts`.

    Uses Python's built-in `hash()` on a tuple of ints, which -- unlike
    `hash()` of `str`/`bytes` -- is NOT affected by `PYTHONHASHSEED`
    randomisation (CPython hashes small ints to themselves), so this is
    stable across processes and machines. Used to derive a reproducible
    train/holdout partition from source-game identity, with no external state
    to keep in sync.
    """
    return (hash(tuple(int(p) for p in parts)) & 0xFFFFFFFF) / 0xFFFFFFFF


def split_holdout_indices(
    game_seeds: np.ndarray,
    decision_indices: np.ndarray,
    *,
    holdout_fraction: float,
    holdout_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Partition manifest row indices into (usable, holdout) index arrays.

    The held-out fraction is never selected for restart generation (CAT-43
    step 2): it exists purely as a frozen high-regret evaluation suite. The
    split is a deterministic per-source-game hash of `(game_seed,
    holdout_seed)` -- NOT a random shuffle -- so every root from one source
    game stays on the same side and re-running with the same seed reserves the
    exact same source games without external state.
    """
    if not (0.0 <= holdout_fraction < 1.0):
        raise ValueError(f"holdout_fraction must be in [0, 1), got {holdout_fraction!r}")
    game_seeds = np.asarray(game_seeds)
    decision_indices = np.asarray(decision_indices)
    if game_seeds.ndim != 1 or decision_indices.shape != game_seeds.shape:
        raise ValueError("holdout game_seed/decision_index columns are misaligned")
    n = int(game_seeds.shape[0])
    if holdout_fraction <= 0.0 or n == 0:
        return np.arange(n, dtype=np.int64), np.empty(0, dtype=np.int64)
    is_holdout = np.array(
        [
            _stable_unit_hash(int(game_seeds[i]), int(holdout_seed))
            < holdout_fraction
            for i in range(n)
        ]
    )
    holdout_idx = np.nonzero(is_holdout)[0].astype(np.int64)
    usable_idx = np.nonzero(~is_holdout)[0].astype(np.int64)
    return usable_idx, holdout_idx


def write_holdout_manifest(
    out_path: Path,
    data: Any,
    holdout_idx: np.ndarray,
    *,
    holdout_fraction: float,
    holdout_seed: int,
) -> None:
    """Write the reserved held-out rows to a standalone manifest.

    Never consumed by `select_archived_states` / restart generation -- kept
    only so a future "held-out high-regret suite" evaluation can replay
    exactly these states (roadmap doc's "held-out high-regret suite never
    trained on").
    """
    shard_paths = np.asarray(data["shard_paths"])
    cols = {
        "shard_id": np.asarray(data["shard_id"])[holdout_idx],
        "row_index": np.asarray(data["row_index"])[holdout_idx] if "row_index" in data else np.full(holdout_idx.shape, -1, dtype=np.int32),
        "game_seed": np.asarray(data["game_seed"])[holdout_idx],
        "decision_index": np.asarray(data["decision_index"])[holdout_idx],
        "regret_score": np.asarray(data["regret_score"])[holdout_idx],
        "phase": np.asarray(data["phase"]).astype(str)[holdout_idx],
        "shard_paths": shard_paths,
        "holdout_fraction": np.asarray(holdout_fraction, dtype=np.float64),
        "holdout_seed": np.asarray(holdout_seed, dtype=np.int64),
    }
    for name in (
        "manifest_schema",
        "extraction_identity_sha256",
        "sample_frac",
        "sample_seed",
        "value_checkpoint_sha256",
        "shard_sha256",
    ):
        if name in data:
            cols[name] = np.asarray(data[name])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.savez(handle, **cols)
    tmp.replace(out_path)


def select_archived_states(
    manifest_path: Path,
    counts: dict[str, int],
    *,
    rng: np.random.Generator,
    sampling: str = "uniform",
    rgsc_temperature: float = DEFAULT_RGSC_TEMPERATURE,
    usable_idx: np.ndarray | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Pick manifest rows for each archived bucket per `counts`.

    `sampling` selects the selection rule within each bucket's candidate
    pool:
      * "uniform" (default): opening/robber_dev take the highest-scoring row
        from distinct source games in that phase bucket; random_archived scans
        a uniform random permutation of remaining candidate rows and keeps the
        first row from each still-unused source game.
      * "rgsc": every bucket samples from its candidate pool via the RGSC
        ranking-based regret-weighted rule (`rgsc_sampler.rgsc_sample_indices`)
        instead of a deterministic top-slice / plain uniform choice.

    `usable_idx`, if given, restricts candidates to this subset of manifest
    row indices (used to exclude the CAT-43 held-out suite from generation).

    Returns {bucket: [ {shard_path, game_seed, decision_index, phase, ...} ]}.
    """
    if sampling not in ("uniform", "rgsc"):
        raise ValueError(f"unknown sampling mode: {sampling!r}")
    data = np.load(manifest_path, allow_pickle=True)
    shard_paths = [str(p) for p in np.asarray(data["shard_paths"])]
    if "shard_sha256" not in data:
        raise ValueError(
            "regret manifest lacks byte-bound shard_sha256 inventory; "
            "re-extract it before restart generation"
        )
    shard_sha256 = [str(value) for value in np.asarray(data["shard_sha256"])]
    if len(shard_paths) != len(shard_sha256):
        raise ValueError("regret manifest shard path/hash inventory is misaligned")
    verified_shard_paths: dict[int, str] = {}

    def verified_shard_path(shard_id: int) -> str:
        if shard_id in verified_shard_paths:
            return verified_shard_paths[shard_id]
        if shard_id < 0 or shard_id >= len(shard_paths):
            raise ValueError(f"regret manifest shard_id out of range: {shard_id}")
        path = Path(shard_paths[shard_id]).expanduser()
        if not path.is_absolute():
            path = Path(manifest_path).parent / path
        try:
            resolved = path.resolve(strict=True)
        except OSError as error:
            raise ValueError(f"regret source shard is unavailable: {path}") from error
        digest = hashlib.sha256()
        with resolved.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        actual = "sha256:" + digest.hexdigest()
        expected = shard_sha256[shard_id]
        if actual != expected:
            raise ValueError(
                "regret source shard sha256 mismatch: "
                f"path={resolved} expected={expected} actual={actual}"
            )
        verified_shard_paths[shard_id] = str(resolved)
        return str(resolved)
    n = int(np.asarray(data["game_seed"]).shape[0])
    phases = np.asarray(data["phase"]).astype(str)
    regret_scores = np.asarray(data["regret_score"], dtype=np.float64)
    buckets_of = np.asarray([_bucket_of_phase(p) for p in phases])
    pool = np.arange(n, dtype=np.int64) if usable_idx is None else np.asarray(usable_idx, dtype=np.int64)

    def record(i: int) -> dict[str, Any]:
        shard_id = int(data["shard_id"][i])
        return {
            "shard_path": verified_shard_path(shard_id),
            "game_seed": int(data["game_seed"][i]),
            "decision_index": int(data["decision_index"][i]),
            "phase": str(phases[i]),
            "regret_score": float(data["regret_score"][i]),
        }

    used_game_seeds: set[int] = set()

    def unique_game_candidates(candidates: np.ndarray) -> np.ndarray:
        """Keep the highest-regret row from each unused source game."""

        selected: list[int] = []
        local: set[int] = set()
        for raw_index in candidates:
            index = int(raw_index)
            game_seed = int(data["game_seed"][index])
            if game_seed in used_game_seeds or game_seed in local:
                continue
            local.add(game_seed)
            selected.append(index)
        return np.asarray(selected, dtype=np.int64)

    def pick_from(
        candidates: np.ndarray,
        want: int,
        *,
        randomize_uniform: bool = False,
    ) -> list[int]:
        """`want` global indices out of `candidates` (a global-index array),
        per the active `sampling` mode. `candidates` is assumed already in
        manifest (score-sorted desc) order for "uniform" top-slice semantics.
        """
        if want <= 0 or candidates.size == 0:
            return []
        if sampling == "uniform" and randomize_uniform:
            picked = []
            local_game_seeds: set[int] = set()
            for raw_index in rng.permutation(candidates):
                index = int(raw_index)
                game_seed = int(data["game_seed"][index])
                if game_seed in used_game_seeds or game_seed in local_game_seeds:
                    continue
                local_game_seeds.add(game_seed)
                picked.append(index)
                if len(picked) >= want:
                    break
        else:
            candidates = unique_game_candidates(candidates)
            if sampling == "uniform":
                picked = [int(index) for index in candidates[:want]]
            else:
                local_indices = rgsc_sample_indices(
                    regret_scores[candidates],
                    want,
                    temperature=rgsc_temperature,
                    rng=rng,
                )
                picked = [int(candidates[index]) for index in local_indices]
        used_game_seeds.update(int(data["game_seed"][index]) for index in picked)
        return picked

    out: dict[str, list[dict[str, Any]]] = {}
    # Manifest is already score-sorted desc, so first occurrences are top scored.
    for bucket in ("opening", "robber_dev"):
        want = counts.get(bucket, 0)
        bucket_pool = pool[buckets_of[pool] == bucket]
        idxs = pick_from(bucket_pool, want)
        out[bucket] = [record(i) for i in idxs]
    want_rand = counts.get("random_archived", 0)
    if want_rand > 0 and pool.size > 0:
        picked = pick_from(
            pool,
            want_rand,
            randomize_uniform=sampling == "uniform",
        )
        out["random_archived"] = [record(i) for i in picked]
    else:
        out["random_archived"] = []
    return out


def validate_archived_selection_counts(
    selected: dict[str, list[dict[str, Any]]],
    planned_counts: dict[str, int],
) -> None:
    """Fail before generation when source-game dedup cannot fill the recipe."""

    shortfalls = {
        bucket: {
            "planned": int(planned_counts.get(bucket, 0)),
            "selected": len(selected.get(bucket, [])),
        }
        for bucket in ("opening", "robber_dev", "random_archived")
        if len(selected.get(bucket, [])) != int(planned_counts.get(bucket, 0))
    }
    if shortfalls:
        raise SystemExit(
            "insufficient distinct archived source games for restart mix: "
            f"{shortfalls}"
        )


def validate_restart_generation_result(
    planned_counts: dict[str, int],
    produced_counts: dict[str, int],
    *,
    failures: list[dict[str, Any]],
) -> None:
    """Refuse a partial restart corpus instead of publishing a success manifest."""

    mismatches = {
        bucket: {
            "planned": int(planned_counts.get(bucket, 0)),
            "produced": int(produced_counts.get(bucket, 0)),
        }
        for bucket in ("normal", "opening", "robber_dev", "random_archived")
        if int(produced_counts.get(bucket, 0)) != int(planned_counts.get(bucket, 0))
    }
    if failures or mismatches:
        raise RuntimeError(
            "restart generation incomplete; refusing success manifest: "
            f"bucket_mismatches={mismatches} failures={len(failures)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate restart self-play from archived high-regret states."
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--manifest", required=True, help="regret manifest .npz from extract_regret_states.py")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--checkpoint", default=None, help="omit to use HeuristicRustEvaluator")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--base-seed", type=int, default=900_000)
    parser.add_argument("--max-continuation-decisions", type=int, default=600)
    parser.add_argument("--restart-temperature-decisions", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--value-scale", type=float, default=1.0)
    parser.add_argument("--value-squash", choices=("tanh", "clip"), default="tanh")
    parser.add_argument(
        "--public-observation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Mask hidden opponent info at the model input (f72/#76), threaded into "
            "EntityGraphRustEvaluatorConfig.public_observation. REQUIRED when the "
            "--checkpoint is a masked/public-observation-trained net (e.g. "
            "champion_v0); otherwise neural_rust_mcts's regime assert refuses it. "
            "Default ON. Persisted rows are always public-masked."
        ),
    )
    parser.add_argument(
        "--meaningful-public-history",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--event-history-limit", type=int, default=64)
    parser.add_argument(
        "--learner-entity-feature-adapter-version",
        default=RUST_ENTITY_ADAPTER_V5,
    )
    parser.add_argument(
        "--correct-rust-chance-spectra",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--shard-size", type=int, default=2048)
    parser.add_argument("--normal-fraction", type=float, default=0.60)
    parser.add_argument("--opening-fraction", type=float, default=0.20)
    parser.add_argument("--robber-dev-fraction", type=float, default=0.10)
    parser.add_argument("--random-archived-fraction", type=float, default=0.10)
    parser.add_argument(
        "--restart-sampling",
        choices=("uniform", "rgsc"),
        default="uniform",
        help=(
            "uniform (default): opening/robber_dev take the top-scoring rows in "
            "that phase, random_archived samples uniformly (pre-CAT-43 behaviour). "
            "rgsc: every bucket samples via the RGSC ranking-based regret-weighted "
            "rule (Tsai et al., ICLR 2026) instead."
        ),
    )
    parser.add_argument(
        "--rgsc-temperature",
        type=float,
        default=DEFAULT_RGSC_TEMPERATURE,
        help="sampling temperature tau for --restart-sampling=rgsc (paper default 0.1)",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.0,
        help=(
            "fraction of manifest rows to reserve as a held-out high-regret suite, "
            "never selected for restart generation; written to "
            "<out-dir>/holdout_manifest.npz"
        ),
    )
    parser.add_argument(
        "--holdout-seed",
        type=int,
        default=None,
        help="seed for the holdout split hash; defaults to --base-seed",
    )
    args = parser.parse_args()

    output = Path(args.out_dir)
    if output.exists() and (any(output.glob("*.npz")) or (output / "manifest.json").exists()):
        raise SystemExit(f"{output} already has output; use a fresh --out-dir")
    output.mkdir(parents=True, exist_ok=True)

    evaluator = _build_evaluator(args)
    action_size = action_size_for_evaluator(evaluator, COLORS)
    config = RestartSelfPlayConfig(
        colors=COLORS,
        max_continuation_decisions=int(args.max_continuation_decisions),
        restart_temperature_decisions=int(args.restart_temperature_decisions),
        temperature=float(args.temperature),
        correct_rust_chance_spectra=bool(args.correct_rust_chance_spectra),
        meaningful_public_history=bool(args.meaningful_public_history),
        meaningful_public_history_schema=(
            MEANINGFUL_PUBLIC_HISTORY_SCHEMA_V2
            if str(args.learner_entity_feature_adapter_version)
            in {RUST_ENTITY_ADAPTER_V5, RUST_ENTITY_ADAPTER_V6}
            else RawSelfPlayConfig().meaningful_public_history_schema
        ),
        event_history_limit=int(args.event_history_limit),
        entity_feature_adapter_version=str(
            args.learner_entity_feature_adapter_version
        ),
        target_information_regime=(
            TARGET_INFORMATION_REGIME_PUBLIC_COHERENT
            if args.checkpoint and bool(args.public_observation)
            else TARGET_INFORMATION_REGIME_AUTHORITATIVE
        ),
    )

    counts = plan_start_mix(
        args.games,
        normal=args.normal_fraction,
        opening=args.opening_fraction,
        robber_dev=args.robber_dev_fraction,
        random_archived=args.random_archived_fraction,
    )
    trajectory_seed_ranges = plan_trajectory_seed_ranges(args.base_seed, counts)
    rng = np.random.default_rng(args.base_seed)
    holdout_seed = args.holdout_seed if args.holdout_seed is not None else args.base_seed
    usable_idx: np.ndarray | None = None
    holdout_count = 0
    if args.holdout_fraction > 0.0:
        manifest_data = np.load(Path(args.manifest), allow_pickle=True)
        usable_idx, holdout_idx = split_holdout_indices(
            np.asarray(manifest_data["game_seed"]),
            np.asarray(manifest_data["decision_index"]),
            holdout_fraction=float(args.holdout_fraction),
            holdout_seed=int(holdout_seed),
        )
        holdout_count = int(holdout_idx.shape[0])
        if holdout_count > 0:
            write_holdout_manifest(
                output / "holdout_manifest.npz",
                manifest_data,
                holdout_idx,
                holdout_fraction=float(args.holdout_fraction),
                holdout_seed=int(holdout_seed),
            )
    archived = select_archived_states(
        Path(args.manifest),
        counts,
        rng=rng,
        sampling=args.restart_sampling,
        rgsc_temperature=float(args.rgsc_temperature),
        usable_idx=usable_idx,
    )
    validate_archived_selection_counts(archived, counts)

    writer = RestartShardWriter(output, shard_size=int(args.shard_size))
    started = time.perf_counter()
    stats = {"games": 0, "rows": 0, "by_bucket": {}, "failures": []}

    # Normal starts: fresh raw-policy games tagged start_mode="normal".
    game_seed = int(trajectory_seed_ranges["normal_start"])
    for _ in range(counts["normal"]):
        record = play_one_raw_selfplay_game(
            evaluator,
            config=RawSelfPlayConfig(
                colors=COLORS,
                max_decisions=int(args.max_continuation_decisions),
                temperature_decisions=int(args.restart_temperature_decisions),
                temperature=float(args.temperature),
                correct_rust_chance_spectra=bool(args.correct_rust_chance_spectra),
                meaningful_public_history=bool(args.meaningful_public_history),
                meaningful_public_history_schema=(
                    MEANINGFUL_PUBLIC_HISTORY_SCHEMA_V2
                    if str(args.learner_entity_feature_adapter_version)
                    in {RUST_ENTITY_ADAPTER_V5, RUST_ENTITY_ADAPTER_V6}
                    else RawSelfPlayConfig().meaningful_public_history_schema
                ),
                event_history_limit=int(args.event_history_limit),
                entity_feature_adapter_version=str(
                    args.learner_entity_feature_adapter_version
                ),
                target_information_regime=str(
                    config.target_information_regime
                ),
            ),
            game_seed=game_seed,
            game_index=game_seed,
            action_size=action_size,
            seed=int(args.base_seed),
        )
        for dec in record.decisions:
            dec.row["teacher_name"] = TEACHER_NAME
            dec.row["restart_provenance_present"] = np.bool_(True)
            dec.row["start_mode"] = START_NORMAL
            dec.row["start_bucket"] = "normal"
            dec.row["archived_game_seed"] = np.int64(-1)
            dec.row["archived_decision_index"] = np.int64(-1)
            dec.row["restart_select_seed"] = np.int64(game_seed)
            writer.add(dec.row, dec.features)
        stats["rows"] += len(record.decisions)
        stats["games"] += 1
        stats["by_bucket"]["normal"] = stats["by_bucket"].get("normal", 0) + 1
        game_seed += 1

    # Archived-state restarts.
    if game_seed != int(trajectory_seed_ranges["normal_stop_exclusive"]):
        raise RuntimeError("normal trajectory seed allocation drift")
    restart_seed = int(trajectory_seed_ranges["archived_start"])
    for bucket in ("opening", "robber_dev", "random_archived"):
        for spec in archived.get(bucket, []):
            try:
                seq = gather_game_action_sequence(
                    Path(spec["shard_path"]).parent, spec["game_seed"], colors=COLORS
                )
                game, chance_rng = reconstruct_state(
                    spec["game_seed"],
                    seq.actions,
                    spec["decision_index"],
                    colors=COLORS,
                    correct_rust_chance_spectra=bool(args.correct_rust_chance_spectra),
                    action_size=action_size,
                    return_rng=True,
                )
                pairs = play_restart_game_from_state(
                    evaluator,
                    game,
                    config=config,
                    start_mode=START_ARCHIVED,
                    start_bucket=bucket,
                    archived_game_seed=spec["game_seed"],
                    archived_decision_index=spec["decision_index"],
                    restart_select_seed=restart_seed,
                    action_size=action_size,
                    chance_rng=chance_rng,
                )
            except Exception as error:  # noqa: BLE001 - isolate one restart.
                stats["failures"].append(
                    {"bucket": bucket, "spec": spec, "error": repr(error)}
                )
                restart_seed += 1
                continue
            for row, features in pairs:
                writer.add(row, features)
            stats["rows"] += len(pairs)
            stats["games"] += 1
            stats["by_bucket"][bucket] = stats["by_bucket"].get(bucket, 0) + 1
            restart_seed += 1

    writer.close()
    validate_restart_generation_result(
        counts,
        stats["by_bucket"],
        failures=stats["failures"],
    )
    summary = {
        "out_dir": str(output),
        "teacher_name": TEACHER_NAME,
        "planned_counts": counts,
        "produced": stats["by_bucket"],
        "games": stats["games"],
        "rows": stats["rows"],
        "failures": stats["failures"][:20],
        "n_failures": len(stats["failures"]),
        "elapsed_sec": time.perf_counter() - started,
        "config": dataclasses.asdict(config),
        "trajectory_seed_ranges": trajectory_seed_ranges,
        "shards": [str(p) for p in writer.paths],
        "start_mode_note": (
            "archived rows are legitimate reachable PUBLIC states (true-history "
            "replay), flagged for separate metrics per DAGS arXiv 2605.14379"
        ),
        "trajectory_game_seed_semantics": (
            "normal rows use their fresh game seed; archived rows use the unique "
            "restart_select_seed, with the reconstruction source retained in "
            "archived_game_seed"
        ),
        "restart_sampling": args.restart_sampling,
        "rgsc_temperature": float(args.rgsc_temperature),
        "holdout_fraction": float(args.holdout_fraction),
        "holdout_seed": int(holdout_seed),
        "holdout_count": holdout_count,
    }
    from regret_common import write_json_atomic

    write_json_atomic(output / "manifest.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


def _build_evaluator(args: argparse.Namespace) -> RustEvaluator:
    if args.checkpoint:
        from catan_zero.search.neural_rust_mcts import (
            BatchedEntityGraphRustEvaluator,
            EntityGraphRustEvaluatorConfig,
        )

        return BatchedEntityGraphRustEvaluator.from_checkpoint(
            args.checkpoint,
            device=args.device,
            config=EntityGraphRustEvaluatorConfig(
                value_scale=float(args.value_scale),
                value_squash=str(args.value_squash),
                # Thread public_observation so a masked champion passes the
                # regime assert (which still fires loud on a mismatch -- the
                # safety net is intact, not weakened).
                public_observation=bool(getattr(args, "public_observation", False)),
            ),
        )
    from catan_zero.search.gumbel_chance_mcts import HeuristicRustEvaluator

    return HeuristicRustEvaluator(score_actions=True)


if __name__ == "__main__":
    main()
