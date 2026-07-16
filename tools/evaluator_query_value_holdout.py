#!/usr/bin/env python3
# ruff: noqa: E402 -- executable adds the sibling tools directory before imports.
"""Bounded, non-promotion audit of the deployed scalar value at Catan roots.

The historical phase calibration pools every ordinary turn prompt into
``PLAY_TURN``.  That hides two load-bearing forced transitions: the value
immediately before ROLL and the value immediately before END_TURN.  This tool
uses the stored authoritative action id/version to split those roots without
replaying a second game engine, queries the checkpoint on the stored entity
features, applies the exact scalar transform used by the Rust evaluator, and
reports:

* opening, pre-roll, post-roll/play-turn, END_TURN, robber, and discard slices;
* the first next-actor root after END_TURN;
* bias, RMSE, Pearson/Spearman correlation, Brier/ECE calibration, and fixed
  reliability bins;
* exact phase and legal-width slices;
* whole-game bootstrap confidence intervals; and
* END_TURN -> next-actor value antisymmetry.

Only naturally terminated games carry outcome labels and enter the report.
The default game/row limits are intentionally finite.  The artifact is
diagnostic-only and cannot be used as promotion evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from catan_zero.rl.action_mask import ActionCatalog
from catan_zero.rl.entity_token_features import mask_player_tokens_public
from catan_zero.rl.entity_token_policy import EntityGraphPolicy
from factory_common import write_json
from phase_sliced_value_calibration import (
    ENTITY_KEYS,
    _calibration_stats,
    _legal_bucket,
    load_validation_seed_manifest,
    parse_validation_game_seed_ranges,
    resolve_readout_provenance,
)


SCHEMA_VERSION = "evaluator-query-value-holdout-v1"
PLAYER_NAMES = ("BLUE", "RED")
SUPPORTED_ACTION_MASK_VERSIONS = frozenset(
    {ActionCatalog.version, "colonist-multiagent-v1"}
)
ROOT_CLASSES = (
    "opening",
    "pre_roll",
    "post_roll_play_turn",
    "end_turn",
    "actor_handoff_next",
    "robber",
    "discard",
    "other",
)
BOOTSTRAP_METRICS = (
    "bias",
    "value_rmse",
    "corr_q_z",
    "spearman_q_z",
    "brier",
    "win_probability_ece",
)
OPTIONAL_ENTITY_KEYS = ("deduction_features",)


@dataclass(frozen=True)
class EvaluatorBinding:
    value_scale: float
    value_squash: str
    public_observation: bool | None
    entity_feature_adapter_version: str | None
    science_contract_path: str | None
    science_contract_sha256: str | None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_evaluator_binding(
    science_contract: str | Path | None,
    *,
    value_scale: float,
    value_squash: str,
) -> EvaluatorBinding:
    """Resolve the deployed scalar transform and observation/adapter contract."""

    if science_contract is None:
        if not math.isfinite(float(value_scale)) or float(value_scale) <= 0.0:
            raise ValueError("value_scale must be finite and positive")
        if value_squash not in {"tanh", "clip"}:
            raise ValueError("value_squash must be tanh or clip")
        return EvaluatorBinding(
            value_scale=float(value_scale),
            value_squash=str(value_squash),
            public_observation=None,
            entity_feature_adapter_version=None,
            science_contract_path=None,
            science_contract_sha256=None,
        )

    path = Path(science_contract).expanduser().resolve(strict=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("science contract must be a JSON object")
    operator = payload.get("operator")
    evaluator = operator.get("evaluator") if isinstance(operator, dict) else None
    if not isinstance(evaluator, dict):
        raise ValueError("science contract has no operator.evaluator object")
    if evaluator.get("value_readout") != "scalar":
        raise ValueError("value holdout requires the deployed scalar readout")
    resolved_scale = float(evaluator.get("value_scale"))
    resolved_squash = str(evaluator.get("value_squash"))
    if not math.isfinite(resolved_scale) or resolved_scale <= 0.0:
        raise ValueError("science contract evaluator.value_scale is invalid")
    if resolved_squash not in {"tanh", "clip"}:
        raise ValueError("science contract evaluator.value_squash is invalid")
    public_observation = evaluator.get("public_observation")
    if not isinstance(public_observation, bool):
        raise ValueError(
            "science contract evaluator.public_observation must be boolean"
        )
    learner = payload.get("learner")
    construction = (
        learner.get("model_construction") if isinstance(learner, dict) else None
    )
    adapter = (
        construction.get("entity_feature_adapter_version")
        if isinstance(construction, dict)
        else None
    )
    if adapter is not None and (not isinstance(adapter, str) or not adapter):
        raise ValueError("science contract learner adapter is malformed")
    return EvaluatorBinding(
        value_scale=resolved_scale,
        value_squash=resolved_squash,
        public_observation=public_observation,
        entity_feature_adapter_version=adapter,
        science_contract_path=str(path),
        science_contract_sha256=_sha256_file(path),
    )


def deployed_scalar_values(
    raw_values: np.ndarray, *, value_scale: float, value_squash: str
) -> np.ndarray:
    """Apply ``EntityGraphRustEvaluator._apply_value_squash`` plus final clip."""

    raw = np.asarray(raw_values, dtype=np.float64)
    scaled = raw * float(value_scale)
    if value_squash == "tanh":
        transformed = np.tanh(scaled)
    elif value_squash == "clip":
        transformed = scaled
    else:
        raise ValueError(f"unknown value_squash {value_squash!r}")
    return np.clip(transformed, -1.0, 1.0)


def _catalog_action_types() -> tuple[str, ...]:
    catalog = ActionCatalog(PLAYER_NAMES)
    return tuple(
        str(catalog.describe(index)["action_type"]).upper()
        for index in range(catalog.size)
    )


def decode_action_types(
    action_ids: np.ndarray,
    action_mask_versions: np.ndarray,
    phases: np.ndarray,
) -> np.ndarray:
    """Decode only action schemas whose base flat ids are authoritative."""

    actions = np.asarray(action_ids, dtype=np.int64).reshape(-1)
    versions = np.asarray(action_mask_versions).astype(str).reshape(-1)
    phase_values = np.char.upper(np.asarray(phases).astype(str).reshape(-1))
    if actions.shape != versions.shape or actions.shape != phase_values.shape:
        raise ValueError("action id/version/phase arrays must be row aligned")
    table = _catalog_action_types()
    result = np.full(len(actions), "UNKNOWN", dtype=object)
    self_describing = np.isin(phase_values, ("ROLL", "END_TURN"))
    result[self_describing] = phase_values[self_describing]
    supported = np.isin(versions, tuple(SUPPORTED_ACTION_MASK_VERSIONS))
    valid = supported & (actions >= 0) & (actions < len(table))
    if np.any(valid):
        rows = np.flatnonzero(valid)
        result[rows] = [table[int(actions[row])] for row in rows]
    needs_turn_decode = np.isin(
        phase_values, ("PLAY_TURN", "MAIN_TURN")
    ) & (result == "UNKNOWN")
    if np.any(needs_turn_decode):
        observed_versions = sorted(set(versions[needs_turn_decode].tolist()))
        bad_ids = sorted(set(actions[needs_turn_decode].tolist()))[:8]
        raise ValueError(
            "cannot classify PLAY_TURN value roots with an unsupported action "
            f"catalog: versions={observed_versions}, sample_action_ids={bad_ids}"
        )
    return result.astype(str)


def classify_roots(phases: np.ndarray, action_types: np.ndarray) -> np.ndarray:
    phase_values = np.char.upper(np.asarray(phases).astype(str).reshape(-1))
    types = np.char.upper(np.asarray(action_types).astype(str).reshape(-1))
    if phase_values.shape != types.shape:
        raise ValueError("phase/action type arrays must be row aligned")
    result = np.full(len(phase_values), "other", dtype=object)
    opening = np.fromiter(
        (
            "INITIAL" in phase
            or phase in {"BUILD_INITIAL_SETTLEMENT", "BUILD_INITIAL_ROAD"}
            for phase in phase_values
        ),
        dtype=np.bool_,
        count=len(phase_values),
    )
    robber = np.char.find(phase_values, "ROBBER") >= 0
    discard = np.char.find(phase_values, "DISCARD") >= 0
    result[opening] = "opening"
    result[robber] = "robber"
    result[discard] = "discard"
    result[types == "ROLL"] = "pre_roll"
    result[types == "END_TURN"] = "end_turn"
    ordinary_turn = np.isin(phase_values, ("PLAY_TURN", "MAIN_TURN"))
    result[
        ordinary_turn & ~np.isin(types, ("ROLL", "END_TURN"))
    ] = "post_roll_play_turn"
    return result.astype(str)


def _source_shards(shard_dirs: Sequence[str]) -> list[tuple[str, Path]]:
    roots = [Path(value).expanduser().resolve(strict=True) for value in shard_dirs]
    if len(set(roots)) != len(roots):
        raise ValueError("--shard-dir roots must be unique")
    result: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for source_index, root in enumerate(roots):
        for shard in sorted(root.rglob("*.npz")):
            resolved = shard.resolve(strict=True)
            if resolved in seen:
                raise ValueError("--shard-dir roots overlap on the same .npz")
            seen.add(resolved)
            result.append((f"source-{source_index}:{root}", resolved))
    if not result:
        raise ValueError("no .npz shards found")
    return result


def _range_mask(
    game_seeds: np.ndarray, ranges: tuple[tuple[int, int], ...]
) -> np.ndarray:
    mask = np.zeros(len(game_seeds), dtype=np.bool_)
    for start, end in ranges:
        mask |= (game_seeds >= int(start)) & (game_seeds <= int(end))
    return mask


def require_explicit_holdout_selection(
    validation_seed_manifest: str | None,
    validation_game_seed_ranges: tuple[tuple[int, int], ...],
) -> None:
    """Refuse an in-sample artifact that merely calls itself a holdout."""

    if validation_seed_manifest is None and not validation_game_seed_ranges:
        raise ValueError(
            "evaluator-query holdout requires --validation-seed-manifest or "
            "--validation-game-seed-ranges; refusing unverified in-sample rows"
        )


def collect_game_groups(
    shard_dirs: Sequence[str],
    *,
    validation_game_seeds: np.ndarray | None,
    validation_game_seed_ranges: tuple[tuple[int, int], ...],
    max_games: int,
    max_rows: int,
) -> list[dict[str, np.ndarray]]:
    """Collect whole naturally-terminated games up to hard finite bounds."""

    if max_games < 1 or max_rows < 1:
        raise ValueError("max_games and max_rows must be positive")
    if validation_game_seeds is not None and validation_game_seed_ranges:
        raise ValueError("choose validation seeds or validation ranges, not both")
    selected = (
        None
        if validation_game_seeds is None
        else np.asarray(validation_game_seeds, dtype=np.int64)
    )
    shards = _source_shards(shard_dirs)
    game_order: list[str] = []
    game_rows: dict[str, int] = {}
    for source_id, shard_path in shards:
        with np.load(shard_path) as data:
            metadata_required = {"game_seed", "terminated", "truncated"}
            missing = sorted(metadata_required - set(data.files))
            if missing:
                raise ValueError(f"{shard_path} missing required fields: {missing}")
            seeds = np.asarray(data["game_seed"], dtype=np.int64)
            natural = np.asarray(data["terminated"], dtype=np.bool_) & ~np.asarray(
                data["truncated"], dtype=np.bool_
            )
            if selected is not None:
                natural &= np.isin(seeds, selected)
            elif validation_game_seed_ranges:
                natural &= _range_mask(seeds, validation_game_seed_ranges)
            if not np.any(natural):
                continue
            for seed, count in zip(
                *np.unique(seeds[natural], return_counts=True), strict=True
            ):
                game_id = f"{source_id}:seed={int(seed)}"
                if game_id not in game_rows:
                    game_order.append(game_id)
                    game_rows[game_id] = 0
                game_rows[game_id] += int(count)

    selected_games: set[str] = set()
    selected_rows = 0
    for game_id in game_order:
        count = game_rows[game_id]
        if len(selected_games) >= max_games or selected_rows + count > max_rows:
            break
        selected_games.add(game_id)
        selected_rows += count
    if not selected_games:
        raise ValueError(
            "no complete naturally terminated held-out game fits within bounds"
        )

    groups: list[dict[str, np.ndarray]] = []
    collected_rows = 0
    for source_id, shard_path in shards:
        with np.load(shard_path) as data:
            required = {
                *ENTITY_KEYS,
                "legal_action_ids",
                "legal_action_context",
                "legal_action_mask",
                "action_taken",
                "action_mask_version",
                "adapter_version",
                "phase",
                "decision_index",
                "game_seed",
                "player",
                "winner",
                "terminated",
                "truncated",
            }
            missing = sorted(required - set(data.files))
            if missing:
                raise ValueError(f"{shard_path} missing required fields: {missing}")
            seeds = np.asarray(data["game_seed"], dtype=np.int64)
            natural = np.asarray(data["terminated"], dtype=np.bool_) & ~np.asarray(
                data["truncated"], dtype=np.bool_
            )
            if not np.any(natural):
                continue
            ordered_seeds = list(dict.fromkeys(seeds[natural].tolist()))
            for seed in ordered_seeds:
                game_id = f"{source_id}:seed={int(seed)}"
                if game_id not in selected_games:
                    continue
                idx = np.flatnonzero(natural & (seeds == int(seed)))
                if not len(idx):
                    continue
                winner = np.asarray(data["winner"])[idx].astype(str)
                player = np.asarray(data["player"])[idx].astype(str)
                z = np.where(winner == player, 1.0, -1.0).astype(np.float32)
                group = {key: np.asarray(data[key])[idx] for key in ENTITY_KEYS}
                for key in OPTIONAL_ENTITY_KEYS:
                    if key in data.files:
                        group[key] = np.asarray(data[key])[idx]
                group.update(
                    {
                        "legal_action_ids": np.asarray(data["legal_action_ids"])[idx],
                        "legal_action_context": np.asarray(
                            data["legal_action_context"]
                        )[idx],
                        "legal_count": np.asarray(data["legal_action_mask"])[idx]
                        .sum(axis=1)
                        .astype(np.int32),
                        "action_taken": np.asarray(data["action_taken"])[idx],
                        "action_mask_version": np.asarray(
                            data["action_mask_version"]
                        )[idx],
                        "adapter_version": np.asarray(data["adapter_version"])[idx],
                        "phase": np.asarray(data["phase"])[idx],
                        "decision_index": np.asarray(data["decision_index"])[idx],
                        "player": player,
                        "z": z,
                        "game_seed": np.asarray(data["game_seed"])[idx],
                        "game_id": np.asarray([game_id] * len(idx)),
                    }
                )
                groups.append(group)
                collected_rows += len(idx)
    if collected_rows != selected_rows:
        raise RuntimeError(
            "whole-game collection row count drift: "
            f"selected={selected_rows}, collected={collected_rows}"
        )
    return groups


def compute_scalar_raw_values(
    policy: EntityGraphPolicy, groups: list[dict[str, np.ndarray]]
) -> np.ndarray:
    """Query the scalar head while preserving checkpoint-required entity fields."""

    import torch

    provenance = resolve_readout_provenance(policy, "scalar")
    if provenance["model_output_key"] != "value":
        raise RuntimeError("scalar readout did not resolve to the value output")
    chunks: list[np.ndarray] = []
    for group in groups:
        entity_batch = {
            key: group[key]
            for key in (*ENTITY_KEYS, *OPTIONAL_ENTITY_KEYS)
            if key in group
        }
        with torch.no_grad():
            outputs = policy.forward_legal_np(
                entity_batch,
                group["legal_action_ids"],
                group["legal_action_context"],
            )
        if "value" not in outputs:
            raise RuntimeError(
                f"scalar evaluator query emitted no value: {sorted(outputs)}"
            )
        chunks.append(
            outputs["value"].detach().float().cpu().numpy().reshape(-1)
        )
    return np.concatenate(chunks)


def _basic_bootstrap_metrics(
    q: np.ndarray, z: np.ndarray, *, reliability_bin_count: int
) -> dict[str, float | None]:
    stats = _calibration_stats(
        q,
        z,
        min_rows=2,
        reliability_bin_count=reliability_bin_count,
    )
    return {
        "bias": stats["bias"],
        "value_rmse": stats["value_rmse"],
        "corr_q_z": stats["corr_q_z"],
        "spearman_q_z": stats["spearman_q_z"],
        "brier": stats["brier"],
        "win_probability_ece": stats["win_probability_ece"],
    }


def game_bootstrap_confidence_intervals(
    q: np.ndarray,
    z: np.ndarray,
    game_ids: np.ndarray,
    *,
    samples: int,
    seed: int,
    reliability_bin_count: int,
) -> dict[str, dict[str, float | int | None]]:
    """Percentile intervals from resampling games, never individual rows."""

    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    ids = np.asarray(game_ids).astype(str).reshape(-1)
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    if q.shape != z.shape or q.shape != ids.shape:
        raise ValueError("q/z/game_ids must be row aligned")
    unique = np.asarray(list(dict.fromkeys(ids.tolist())))
    if len(unique) < 2:
        return {
            metric: {
                "low": None,
                "high": None,
                "valid_resamples": 0,
                "requested_resamples": int(samples),
            }
            for metric in BOOTSTRAP_METRICS
        }
    row_indices = {game: np.flatnonzero(ids == game) for game in unique}
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {metric: [] for metric in BOOTSTRAP_METRICS}
    for _ in range(samples):
        draw = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([row_indices[str(game)] for game in draw])
        report = _basic_bootstrap_metrics(
            q[idx],
            z[idx],
            reliability_bin_count=reliability_bin_count,
        )
        for metric, value in report.items():
            if value is not None and math.isfinite(float(value)):
                values[metric].append(float(value))
    result: dict[str, dict[str, float | int | None]] = {}
    for metric in BOOTSTRAP_METRICS:
        metric_values = np.asarray(values[metric], dtype=np.float64)
        result[metric] = {
            "low": (
                float(np.percentile(metric_values, 2.5))
                if len(metric_values)
                else None
            ),
            "high": (
                float(np.percentile(metric_values, 97.5))
                if len(metric_values)
                else None
            ),
            "valid_resamples": int(len(metric_values)),
            "requested_resamples": int(samples),
        }
    return result


def _slice_report(
    q: np.ndarray,
    z: np.ndarray,
    game_ids: np.ndarray,
    mask: np.ndarray,
    *,
    min_rows: int,
    reliability_bin_count: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    selected = np.asarray(mask, dtype=np.bool_)
    selected_q = q[selected]
    selected_z = z[selected]
    selected_games = game_ids[selected]
    stats = _calibration_stats(
        selected_q,
        selected_z,
        min_rows=min_rows,
        reliability_bin_count=reliability_bin_count,
    )
    stats["n_games"] = int(len(np.unique(selected_games)))
    stats["mean_absolute_error"] = (
        float(np.mean(np.abs(selected_q - selected_z))) if len(selected_z) else None
    )
    stats["game_bootstrap_95ci"] = game_bootstrap_confidence_intervals(
        selected_q,
        selected_z,
        selected_games,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
        reliability_bin_count=reliability_bin_count,
    )
    return stats


def _reports_by_label(
    q: np.ndarray,
    z: np.ndarray,
    game_ids: np.ndarray,
    labels: np.ndarray,
    *,
    min_rows: int,
    reliability_bin_count: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Build deterministic slice reports using the same whole-game bootstrap."""

    values = np.asarray(labels).astype(str).reshape(-1)
    if len(values) != len(q):
        raise ValueError("slice labels must be row aligned")
    reports: dict[str, Any] = {}
    for offset, label in enumerate(sorted(set(values.tolist()))):
        reports[label] = _slice_report(
            q,
            z,
            game_ids,
            values == label,
            min_rows=min_rows,
            reliability_bin_count=reliability_bin_count,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + offset,
        )
    return reports


def actor_handoff_pairs(
    root_classes: np.ndarray,
    game_ids: np.ndarray,
    decision_indices: np.ndarray,
    players: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned END_TURN and immediate next-actor row indices."""

    classes = np.asarray(root_classes).astype(str)
    games = np.asarray(game_ids).astype(str)
    decisions = np.asarray(decision_indices, dtype=np.int64)
    actors = np.asarray(players).astype(str)
    end_rows: list[int] = []
    next_rows: list[int] = []
    for game in dict.fromkeys(games.tolist()):
        rows = np.flatnonzero(games == game)
        order = rows[np.argsort(decisions[rows], kind="stable")]
        by_decision = {int(decisions[row]): int(row) for row in order}
        for row in order:
            if classes[row] != "end_turn":
                continue
            next_row = by_decision.get(int(decisions[row]) + 1)
            if next_row is None or actors[next_row] == actors[row]:
                continue
            end_rows.append(int(row))
            next_rows.append(next_row)
    return np.asarray(end_rows, dtype=np.int64), np.asarray(next_rows, dtype=np.int64)


def _mean_game_bootstrap_interval(
    values: np.ndarray,
    game_ids: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    ids = np.asarray(game_ids).astype(str)
    unique = np.asarray(list(dict.fromkeys(ids.tolist())))
    if len(unique) < 2:
        return {
            "low": None,
            "high": None,
            "valid_resamples": 0,
            "requested_resamples": int(samples),
        }
    per_game = {game: values[ids == game] for game in unique}
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(samples):
        draw = rng.choice(unique, size=len(unique), replace=True)
        means.append(
            float(
                np.mean(
                    np.concatenate([per_game[str(game)] for game in draw])
                )
            )
        )
    return {
        "low": float(np.percentile(means, 2.5)),
        "high": float(np.percentile(means, 97.5)),
        "valid_resamples": int(len(means)),
        "requested_resamples": int(samples),
    }


def build_report(
    raw_q: np.ndarray,
    groups: list[dict[str, np.ndarray]],
    *,
    binding: EvaluatorBinding,
    min_slice_rows: int,
    reliability_bin_count: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    q = deployed_scalar_values(
        raw_q,
        value_scale=binding.value_scale,
        value_squash=binding.value_squash,
    )
    z = np.concatenate([group["z"] for group in groups])
    phases = np.concatenate([group["phase"] for group in groups])
    actions = np.concatenate([group["action_taken"] for group in groups])
    versions = np.concatenate([group["action_mask_version"] for group in groups])
    game_ids = np.concatenate([group["game_id"] for group in groups]).astype(str)
    decisions = np.concatenate([group["decision_index"] for group in groups])
    players = np.concatenate([group["player"] for group in groups]).astype(str)
    legal_counts = np.concatenate([group["legal_count"] for group in groups])
    action_types = decode_action_types(actions, versions, phases)
    classes = classify_roots(phases, action_types)
    phase_labels = np.char.upper(phases.astype(str))
    legal_bucket_labels = np.asarray(
        [_legal_bucket(int(count)) for count in legal_counts]
    )
    end_rows, next_rows = actor_handoff_pairs(
        classes, game_ids, decisions, players
    )
    actor_handoff_mask = np.zeros(len(q), dtype=np.bool_)
    actor_handoff_mask[next_rows] = True

    slices: dict[str, Any] = {}
    for offset, label in enumerate(ROOT_CLASSES):
        mask = actor_handoff_mask if label == "actor_handoff_next" else classes == label
        slices[label] = _slice_report(
            q,
            z,
            game_ids,
            mask,
            min_rows=min_slice_rows,
            reliability_bin_count=reliability_bin_count,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + offset,
        )

    if len(end_rows):
        antisymmetry_error = q[end_rows] + q[next_rows]
        absolute_error = np.abs(antisymmetry_error)
        pair_games = game_ids[end_rows]
        handoff = {
            "n_pairs": int(len(end_rows)),
            "n_games": int(len(np.unique(pair_games))),
            "terminal_label_opposition_fraction": float(
                np.mean(z[end_rows] == -z[next_rows])
            ),
            "mean_signed_antisymmetry_error": float(np.mean(antisymmetry_error)),
            "mean_absolute_antisymmetry_error": float(np.mean(absolute_error)),
            "rmse_antisymmetry_error": float(
                np.sqrt(np.mean(antisymmetry_error**2))
            ),
            "mean_absolute_error_game_bootstrap_95ci": (
                _mean_game_bootstrap_interval(
                    absolute_error,
                    pair_games,
                    samples=bootstrap_samples,
                    seed=bootstrap_seed + 10_000,
                )
            ),
        }
    else:
        handoff = {
            "n_pairs": 0,
            "n_games": 0,
            "terminal_label_opposition_fraction": None,
            "mean_signed_antisymmetry_error": None,
            "mean_absolute_antisymmetry_error": None,
            "rmse_antisymmetry_error": None,
            "mean_absolute_error_game_bootstrap_95ci": {
                "low": None,
                "high": None,
                "valid_resamples": 0,
                "requested_resamples": int(bootstrap_samples),
            },
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "metric_semantics": {
            "deployed_scalar_value": (
                f"{binding.value_squash}(raw_value * {binding.value_scale}) "
                "followed by clip[-1,1]"
            ),
            "bias": "mean(deployed_value - terminal_outcome)",
            "value_rmse": "sqrt(mean((deployed_value - terminal_outcome)^2))",
            "corr_q_z": "row-level Pearson correlation with terminal +/-1 outcome",
            "spearman_q_z": (
                "row-level Spearman rank correlation with terminal +/-1 outcome; "
                "ties receive average ranks"
            ),
            "confidence_intervals": (
                "2.5/97.5 percentiles from whole-game cluster bootstrap"
            ),
            "actor_handoff_antisymmetry": (
                "deployed_value(before END_TURN, old actor) + "
                "deployed_value(next root, new actor)"
            ),
        },
        "evaluator_binding": {
            "value_readout": "scalar",
            "value_scale": binding.value_scale,
            "value_squash": binding.value_squash,
            "public_observation": binding.public_observation,
            "entity_feature_adapter_version": (
                binding.entity_feature_adapter_version
            ),
            "science_contract_path": binding.science_contract_path,
            "science_contract_sha256": binding.science_contract_sha256,
        },
        "cohort": {
            "rows": int(len(q)),
            "games": int(len(np.unique(game_ids))),
            "game_id_set_sha256": _canonical_json_sha256(
                sorted(np.unique(game_ids).tolist())
            ),
            "action_mask_versions": sorted(set(versions.astype(str).tolist())),
            "root_class_counts": {
                label: int(np.count_nonzero(classes == label))
                for label in ROOT_CLASSES
                if label != "actor_handoff_next"
            }
            | {"actor_handoff_next": int(np.count_nonzero(actor_handoff_mask))},
        },
        "global": _slice_report(
            q,
            z,
            game_ids,
            np.ones(len(q), dtype=np.bool_),
            min_rows=min_slice_rows,
            reliability_bin_count=reliability_bin_count,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        ),
        "by_root_class": slices,
        "by_phase": _reports_by_label(
            q,
            z,
            game_ids,
            phase_labels,
            min_rows=min_slice_rows,
            reliability_bin_count=reliability_bin_count,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + 20_000,
        ),
        "by_legal_count_bucket": _reports_by_label(
            q,
            z,
            game_ids,
            legal_bucket_labels,
            min_rows=min_slice_rows,
            reliability_bin_count=reliability_bin_count,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + 30_000,
        ),
        "actor_handoff_consistency": handoff,
    }


def _validate_checkpoint_and_groups(
    policy: EntityGraphPolicy,
    groups: list[dict[str, np.ndarray]],
    binding: EvaluatorBinding,
) -> bool:
    checkpoint_adapter = str(policy.entity_feature_adapter_version)
    observed_adapters = sorted(
        {
            str(value)
            for group in groups
            for value in np.unique(group["adapter_version"])
        }
    )
    if observed_adapters != [checkpoint_adapter]:
        raise ValueError(
            "stored feature adapter does not match checkpoint: "
            f"data={observed_adapters}, checkpoint={checkpoint_adapter!r}"
        )
    if (
        binding.entity_feature_adapter_version is not None
        and binding.entity_feature_adapter_version != checkpoint_adapter
    ):
        raise ValueError(
            "science contract learner adapter does not match checkpoint: "
            f"{binding.entity_feature_adapter_version!r} != {checkpoint_adapter!r}"
        )
    trained_masked = bool(
        getattr(policy, "trained_with_masked_hidden_info", False)
    )
    if (
        binding.public_observation is not None
        and binding.public_observation != trained_masked
    ):
        raise ValueError(
            "science contract public_observation does not match checkpoint "
            f"training: {binding.public_observation} != {trained_masked}"
        )
    return (
        trained_masked
        if binding.public_observation is None
        else binding.public_observation
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shard-dir",
        action="append",
        required=True,
        help="repeatable held-out self-play root searched recursively for *.npz",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--validation-seed-manifest")
    selection.add_argument(
        "--validation-game-seed-ranges",
        default="",
        help="inclusive start:end ranges, matching train_bc/calibration syntax",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--science-contract")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--value-scale", type=float, default=1.0)
    parser.add_argument(
        "--value-squash", choices=("tanh", "clip"), default="tanh"
    )
    parser.add_argument("--max-games", type=int, default=512)
    parser.add_argument("--max-rows", type=int, default=100_000)
    parser.add_argument("--min-slice-rows", type=int, default=30)
    parser.add_argument("--reliability-bins", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=1_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260716)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    selected_seeds: np.ndarray | None = None
    selection_manifest_sha256: str | None = None
    if args.validation_seed_manifest:
        selected_seeds, selection_manifest_sha256 = load_validation_seed_manifest(
            args.validation_seed_manifest
        )
    ranges = parse_validation_game_seed_ranges(args.validation_game_seed_ranges)
    try:
        require_explicit_holdout_selection(args.validation_seed_manifest, ranges)
    except ValueError as error:
        parser.error(str(error))
    groups = collect_game_groups(
        args.shard_dir,
        validation_game_seeds=selected_seeds,
        validation_game_seed_ranges=ranges,
        max_games=args.max_games,
        max_rows=args.max_rows,
    )
    binding = load_evaluator_binding(
        args.science_contract,
        value_scale=args.value_scale,
        value_squash=args.value_squash,
    )
    policy = EntityGraphPolicy.load(args.checkpoint, device=args.device)
    apply_mask = _validate_checkpoint_and_groups(policy, groups, binding)
    if apply_mask:
        for group in groups:
            group["player_tokens"] = mask_player_tokens_public(
                group["player_tokens"]
            )
    raw_q = compute_scalar_raw_values(policy, groups)
    report = build_report(
        raw_q,
        groups,
        binding=binding,
        min_slice_rows=args.min_slice_rows,
        reliability_bin_count=args.reliability_bins,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    checkpoint = Path(args.checkpoint).expanduser().resolve(strict=True)
    report.update(
        {
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": _sha256_file(checkpoint),
                "entity_feature_adapter_version": (
                    policy.entity_feature_adapter_version
                ),
                "trained_with_masked_hidden_info": bool(
                    getattr(policy, "trained_with_masked_hidden_info", False)
                ),
            },
            "selection": {
                "shard_dirs": [
                    str(Path(path).expanduser().resolve(strict=True))
                    for path in args.shard_dir
                ],
                "validation_seed_manifest": (
                    str(
                        Path(args.validation_seed_manifest)
                        .expanduser()
                        .resolve(strict=True)
                    )
                    if args.validation_seed_manifest
                    else None
                ),
                "validation_seed_manifest_sha256": selection_manifest_sha256,
                "validation_game_seed_ranges": [
                    [int(start), int(end)] for start, end in ranges
                ],
                "held_out_filter_applied": True,
                "max_games": int(args.max_games),
                "max_rows": int(args.max_rows),
            },
            "mask_hidden_info_applied": bool(apply_mask),
        }
    )
    write_json(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
