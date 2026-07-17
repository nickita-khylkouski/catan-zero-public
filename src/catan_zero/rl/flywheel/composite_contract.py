"""Shared source->game->row accounting for promotion-eligible replay composites.

The descriptor chooses a source component, then a game uniformly inside that
component, then a row uniformly inside that game. Raw row counts therefore do
not define learner mass. These helpers measure both the physical corpus and the
actual sampler measure so the orchestrator and trainer can independently build
and verify the same receipt.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import BinaryIO, Mapping, Sequence

import numpy as np


FRESH_SOURCE_GAME_RATIOS: dict[str, float] = {
    "current_producer": 0.80,
    "recent_history": 0.15,
    "hard_negative": 0.05,
}
HISTORICAL_REPLAY_CATEGORY = "historical_replay"


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _stable_float(value: float) -> float:
    """Canonicalize derived floating telemetry without hiding material drift."""

    return float(f"{float(value):.12g}")


def _fixed_column(
    corpus_dir: Path,
    corpus_meta: Mapping[str, object],
    name: str,
    authenticated_files: Mapping[str, BinaryIO] | None = None,
) -> np.ndarray:
    columns = corpus_meta.get("columns")
    if not isinstance(columns, dict) or not isinstance(columns.get(name), dict):
        raise ValueError(f"memmap corpus lacks required {name} column")
    schema = columns[name]
    row_count = int(corpus_meta.get("row_count", -1))
    kind = schema.get("kind")
    if kind == "implicit_constant":
        inner = tuple(int(value) for value in schema.get("inner_shape", ()))
        return np.full(
            (row_count, *inner), schema.get("fill"), dtype=np.dtype(schema["dtype"])
        )
    if kind != "fixed":
        raise ValueError(f"memmap {name} must be a fixed scalar column")
    inner = tuple(int(value) for value in schema.get("inner_shape", ()))
    filename = f"{name}.dat"
    source: Path | BinaryIO = corpus_dir / filename
    if authenticated_files is not None and filename in authenticated_files:
        source = authenticated_files[filename]
        source.seek(0)
    values = np.memmap(
        source,
        dtype=np.dtype(schema["dtype"]),
        mode="r",
        shape=(row_count, *inner),
    )
    return np.asarray(values)


def measure_memmap_component(
    corpus_dir: str | Path,
    corpus_meta: Mapping[str, object],
    *,
    authenticated_files: Mapping[str, BinaryIO] | None = None,
) -> dict[str, object]:
    """Measure game, row, and policy-active mass from authenticated payloads."""

    root = Path(corpus_dir)
    row_count = int(corpus_meta.get("row_count", -1))
    if row_count <= 0 or corpus_meta.get("game_seed_present") is not True:
        raise ValueError("promotion-eligible component needs non-empty game_seed data")
    seeds = _fixed_column(
        root,
        corpus_meta,
        "game_seed",
        authenticated_files,
    ).reshape(-1)
    policy_mass = _fixed_column(
        root,
        corpus_meta,
        "policy_weight_multiplier",
        authenticated_files,
    ).astype(np.float64, copy=False).reshape(-1)
    if seeds.size != row_count or policy_mass.size != row_count:
        raise ValueError("component mass columns do not match corpus row_count")
    if not np.all(np.isfinite(policy_mass)) or np.any(policy_mass < 0.0):
        raise ValueError("policy_weight_multiplier contains invalid mass")

    _games, inverse, rows_per_game = np.unique(
        seeds.astype(np.int64, copy=False), return_inverse=True, return_counts=True
    )
    game_count = int(rows_per_game.size)
    if game_count <= 0:
        raise ValueError("promotion-eligible component contains no games")
    active = policy_mass > 0.0
    active_per_game = np.bincount(
        inverse, weights=active.astype(np.float64), minlength=game_count
    )
    mass_per_game = np.bincount(
        inverse, weights=policy_mass, minlength=game_count
    )
    mean_active = float(np.mean(active_per_game / rows_per_game))
    mean_mass = float(np.mean(mass_per_game / rows_per_game))
    return {
        "game_count": game_count,
        "selected_game_count": game_count,
        "training_game_count": game_count,
        "validation_game_count": 0,
        "row_count": row_count,
        "policy_active_row_count": int(np.count_nonzero(active)),
        "policy_weight_multiplier_sum": _stable_float(
            float(np.sum(policy_mass, dtype=np.float64))
        ),
        "mean_game_policy_active_fraction": _stable_float(mean_active),
        "mean_game_policy_weight_multiplier": _stable_float(mean_mass),
    }


def build_sampling_receipt(
    components: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Build the exact source/game/row receipt for an ordered component set.

    Each component must expose ``component_id``, ``source_category``,
    ``game_sampling_ratio``, and ``component_mass`` as returned by
    :func:`measure_memmap_component`.
    """

    if not components:
        raise ValueError("sampling receipt requires at least one component")
    total_games = sum(int(component["component_mass"]["game_count"]) for component in components)
    total_rows = sum(int(component["component_mass"]["row_count"]) for component in components)
    total_active_rows = sum(
        int(component["component_mass"]["policy_active_row_count"])
        for component in components
    )
    total_selected_games = sum(
        int(component["component_mass"]["selected_game_count"])
        for component in components
    )
    total_training_games = sum(
        int(component["component_mass"]["training_game_count"])
        for component in components
    )
    total_validation_games = sum(
        int(component["component_mass"]["validation_game_count"])
        for component in components
    )
    if total_games <= 0 or total_rows <= 0:
        raise ValueError("sampling receipt components are empty")

    records: list[dict[str, object]] = []
    expected_active = 0.0
    expected_policy_mass = 0.0
    effective_ratios: dict[str, float] = {}
    for component in components:
        component_id = str(component["component_id"])
        source_category = str(component["source_category"])
        ratio = _stable_float(float(component["game_sampling_ratio"]))
        mass = dict(component["component_mass"])
        expected_active += ratio * float(mass["mean_game_policy_active_fraction"])
        expected_policy_mass += ratio * float(
            mass["mean_game_policy_weight_multiplier"]
        )
        effective_ratios[component_id] = ratio
        records.append(
            {
                "component_id": component_id,
                "source_category": source_category,
                "game_sampling_ratio": ratio,
                **mass,
                "raw_game_fraction": _stable_float(
                    int(mass["game_count"]) / total_games
                ),
                "raw_row_fraction": _stable_float(
                    int(mass["row_count"]) / total_rows
                ),
                "raw_policy_active_row_fraction": _stable_float(
                    int(mass["policy_active_row_count"]) / total_active_rows
                    if total_active_rows
                    else 0.0
                ),
                "sampler_active_policy_probability": _stable_float(
                    ratio * float(mass["mean_game_policy_active_fraction"])
                ),
                "sampler_policy_weight_multiplier_mass": _stable_float(
                    ratio * float(mass["mean_game_policy_weight_multiplier"])
                ),
            }
        )
    if abs(sum(effective_ratios.values()) - 1.0) > 1e-9:
        raise ValueError("sampling receipt component ratios must sum to one")
    component_mass_sha256 = canonical_sha256(records)
    return {
        "schema_version": "flywheel-source-game-row-mass-v1",
        "sampler_order": ["source", "game", "row"],
        "effective_component_sampling_ratios": effective_ratios,
        "components": records,
        "aggregate": {
            "selected_game_count": total_selected_games,
            "training_game_count": total_training_games,
            "validation_game_count": total_validation_games,
            "game_count": total_games,
            "row_count": total_rows,
            "policy_active_row_count": total_active_rows,
            "sampler_expected_active_policy_probability": _stable_float(
                expected_active
            ),
            "sampler_expected_policy_weight_multiplier_mass": _stable_float(
                expected_policy_mass
            ),
            "component_mass_sha256": component_mass_sha256,
        },
    }
