"""Audit and compare the supervised signal emitted by search teachers.

The tool intentionally reads only persisted corpus fields.  It can ingest raw
Gumbel self-play NPZ shards or a ``memmap_corpus_v1`` directory, optionally
filter the latter through a selected-games manifest/category, and emit:

* policy-target entropy, prior entropy, and KL(target || prior);
* the effective target induced by mixing the stored search distribution with
  the played-action hard label (the learner's ``soft_target_weight`` recipe);
* forced/full-search/policy-active and phase distributions;
* rows/game plus termination/truncation/failure counts; and
* an authenticated, target-only NPZ corpus for cheap future comparisons.

This is a data audit, not an evaluator: it does not import the search or model
implementation whose outputs it measures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


EPS = 1e-12
COMPACT_FIELDS = (
    "game_seed",
    "decision_index",
    "phase",
    "action_taken",
    "legal_action_ids",
    "is_forced",
    "used_full_search",
    "simulations_used",
    "policy_weight_multiplier",
    "value_weight_multiplier",
    "target_policy",
    "prior_policy",
    "target_policy_mask",
    "terminated",
    "truncated",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _summary(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "p10": None,
            "p50": None,
            "p90": None,
        }
    q10, q50, q90 = np.quantile(values, (0.1, 0.5, 0.9))
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p10": float(q10),
        "p50": float(q50),
        "p90": float(q90),
    }


def _fraction(mask: np.ndarray, denominator: np.ndarray | None = None) -> float | None:
    mask = np.asarray(mask, dtype=bool)
    if denominator is None:
        return float(mask.mean()) if mask.size else None
    denominator = np.asarray(denominator, dtype=bool)
    count = int(denominator.sum())
    return float((mask & denominator).sum() / count) if count else None


def _policy_metrics(
    target: np.ndarray, prior: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    target = np.where(mask, np.asarray(target, dtype=np.float64), 0.0)
    prior = np.where(mask, np.asarray(prior, dtype=np.float64), 0.0)
    target /= np.maximum(target.sum(axis=1, keepdims=True), EPS)
    prior /= np.maximum(prior.sum(axis=1, keepdims=True), EPS)
    target_entropy = -(
        np.where(target > 0, target * np.log(np.maximum(target, EPS)), 0.0)
    ).sum(axis=1)
    prior_entropy = -(
        np.where(prior > 0, prior * np.log(np.maximum(prior, EPS)), 0.0)
    ).sum(axis=1)
    kl = np.where(
        target > 0,
        target * (np.log(np.maximum(target, EPS)) - np.log(np.maximum(prior, EPS))),
        0.0,
    ).sum(axis=1)
    return target_entropy, prior_entropy, kl


def _hard_blend_metrics(
    target: np.ndarray,
    legal_action_ids: np.ndarray,
    action_taken: np.ndarray,
    *,
    soft_target_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Measure the effective policy target optimized by ``train_bc``.

    Soft and hard cross entropy are linear in their target, so the learner's
    blend is exactly CE against ``alpha * search + (1-alpha) * one_hot(play)``.
    On deterministic post-temperature moves this sharpens the search target a
    second time; this helper makes that otherwise-hidden drift measurable.
    """

    target = np.asarray(target, dtype=np.float64)
    target /= np.maximum(target.sum(axis=1, keepdims=True), EPS)
    legal = np.asarray(legal_action_ids)
    matches = legal == np.asarray(action_taken)[:, None]
    match_count = matches.sum(axis=1)
    if np.any(match_count != 1):
        first = int(np.flatnonzero(match_count != 1)[0])
        raise ValueError(
            "played action must match exactly one legal target column; "
            f"row={first} matches={int(match_count[first])}"
        )
    played_probability = np.sum(np.where(matches, target, 0.0), axis=1)
    played_is_mode = played_probability >= (target.max(axis=1) - 1e-12)
    alpha = float(np.clip(soft_target_weight, 0.0, 1.0))
    blended = alpha * target + (1.0 - alpha) * matches
    blended_entropy = -np.sum(
        np.where(blended > 0, blended * np.log(np.maximum(blended, EPS)), 0.0), axis=1
    )
    blend_target_kl = np.sum(
        np.where(
            blended > 0,
            blended
            * (np.log(np.maximum(blended, EPS)) - np.log(np.maximum(target, EPS))),
            0.0,
        ),
        axis=1,
    )
    return played_probability, played_is_mode, blended_entropy, blend_target_kl


def _pad_2d(value: np.ndarray, width: int, fill: float | bool) -> np.ndarray:
    if value.shape[1] == width:
        return value
    result = np.full((value.shape[0], width), fill, dtype=value.dtype)
    result[:, : value.shape[1]] = value
    return result


def load_npz_target_corpus(
    root: Path,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    shards = sorted(root.rglob("*.npz"))
    if not shards:
        raise ValueError(f"no NPZ shards under {root}")
    widths: list[int] = []
    inventories: list[dict[str, Any]] = []
    loaded: list[dict[str, np.ndarray]] = []
    for path in shards:
        with np.load(path, allow_pickle=False) as handle:
            missing = {
                "game_seed",
                "target_policy",
                "prior_policy",
                "target_policy_mask",
            } - set(handle.files)
            if missing:
                raise ValueError(f"{path}: missing required columns {sorted(missing)}")
            shard = {
                key: np.asarray(handle[key])
                for key in COMPACT_FIELDS
                if key in handle.files
            }
        widths.append(int(shard["target_policy"].shape[1]))
        loaded.append(shard)
        inventories.append(
            {
                "path": str(path.relative_to(root)),
                "size_bytes": path.stat().st_size,
                "rows": int(len(shard["game_seed"])),
                "game_seed_min": int(shard["game_seed"].min()),
                "game_seed_max": int(shard["game_seed"].max()),
                "sha256": _sha256(path),
            }
        )
    width = max(widths)
    result: dict[str, list[np.ndarray]] = {}
    for shard in loaded:
        for key, value in shard.items():
            if key in {"target_policy", "prior_policy"}:
                value = _pad_2d(value, width, 0.0)
            elif key == "legal_action_ids":
                value = _pad_2d(value, width, -1)
            elif key == "target_policy_mask":
                value = _pad_2d(value, width, False)
            result.setdefault(key, []).append(value)
    return {
        key: np.concatenate(parts, axis=0) for key, parts in result.items()
    }, inventories


def _open_fixed_memmap(root: Path, meta: dict[str, Any], name: str) -> np.memmap:
    column = meta["columns"][name]
    dtype = np.dtype(column["dtype"])
    shape = (int(meta["row_count"]), *column.get("inner_shape", []))
    return np.memmap(root / f"{name}.dat", dtype=dtype, mode="r", shape=shape)


def _selected_seeds(manifest_path: Path, category: str | None) -> np.ndarray:
    manifest = json.loads(manifest_path.read_text())
    seeds = [
        int(record["game_seed"])
        for record in manifest["records"]
        if category is None or record.get("category") == category
    ]
    if not seeds:
        raise ValueError(
            f"no selected seeds for category={category!r} in {manifest_path}"
        )
    return np.asarray(sorted(set(seeds)), dtype=np.int64)


def analyze_memmap(
    root: Path,
    *,
    seed_manifest: Path | None = None,
    category: str | None = None,
    chunk_rows: int = 262_144,
    soft_target_weight: float = 0.9,
) -> tuple[dict[str, Any], dict[str, Any]]:
    meta_path = root / "corpus_meta.json"
    meta = json.loads(meta_path.read_text())
    if meta.get("schema") != "memmap_corpus_v1":
        raise ValueError(f"unsupported corpus schema in {meta_path}")
    n_rows = int(meta["row_count"])
    seeds_filter = _selected_seeds(seed_manifest, category) if seed_manifest else None
    fixed = {
        name: _open_fixed_memmap(root, meta, name)
        for name in (
            "game_seed",
            "policy_weight_multiplier",
            "terminated",
            "truncated",
        )
    }
    has_is_forced = "is_forced" in meta["columns"]
    if has_is_forced:
        fixed["is_forced"] = _open_fixed_memmap(root, meta, "is_forced")
    has_used_full_search = "used_full_search" in meta["columns"]
    if has_used_full_search:
        fixed["used_full_search"] = _open_fixed_memmap(root, meta, "used_full_search")
    has_action_blend = {
        "action_taken",
        "legal_action_ids",
    }.issubset(meta["columns"])
    if has_action_blend:
        fixed["action_taken"] = _open_fixed_memmap(root, meta, "action_taken")
    phase_meta = meta["columns"]["phase"]
    phase_codes = np.memmap(
        root / "phase.codes.dat", dtype=np.int32, mode="r", shape=(n_rows,)
    )
    phases = list(phase_meta["categories"])
    offsets = np.memmap(
        root / "row_offsets.dat", dtype=np.int64, mode="r", shape=(n_rows + 1,)
    )
    flat_count = int(offsets[-1])
    target_flat = np.memmap(
        root / "target_policy.dat", dtype=np.float32, mode="r", shape=(flat_count,)
    )
    prior_flat = np.memmap(
        root / "prior_policy.dat", dtype=np.float32, mode="r", shape=(flat_count,)
    )
    legal_flat = (
        np.memmap(
            root / "legal_action_ids.dat", dtype=np.int16, mode="r", shape=(flat_count,)
        )
        if has_action_blend
        else None
    )

    game_counts: Counter[int] = Counter()
    phase_counts: Counter[str] = Counter()
    forced = full = active = rows = 0
    truncated_games: set[int] = set()
    terminated_games: set[int] = set()
    entropy_parts: list[np.ndarray] = []
    prior_entropy_parts: list[np.ndarray] = []
    kl_parts: list[np.ndarray] = []
    active_phase_parts: list[np.ndarray] = []
    blend_parts: dict[str, list[np.ndarray]] = {
        "played_target_probability": [],
        "played_is_target_mode": [],
        "hard_blend_entropy": [],
        "kl_hard_blend_target": [],
    }
    for start in range(0, n_rows, chunk_rows):
        stop = min(start + chunk_rows, n_rows)
        seeds = np.asarray(fixed["game_seed"][start:stop])
        keep = np.ones(stop - start, dtype=bool)
        if seeds_filter is not None:
            keep = np.isin(seeds, seeds_filter, assume_unique=False)
        if not keep.any():
            continue
        indices = np.flatnonzero(keep) + start
        rows += int(indices.size)
        selected_seeds = seeds[keep]
        unique, counts = np.unique(selected_seeds, return_counts=True)
        game_counts.update(
            {int(seed): int(count) for seed, count in zip(unique, counts)}
        )
        trunc = np.asarray(fixed["truncated"][indices], dtype=bool)
        term = np.asarray(fixed["terminated"][indices], dtype=bool)
        truncated_games.update(map(int, selected_seeds[trunc]))
        terminated_games.update(map(int, selected_seeds[term]))
        if has_is_forced:
            forced_mask = np.asarray(fixed["is_forced"][indices], dtype=bool)
        else:
            # Historical memmaps predate the explicit column. A forced row is
            # exactly a one-candidate ragged legal/target row; reconstruct it
            # from authenticated row offsets instead of making the audit
            # unusable on the replay corpus.
            forced_mask = (
                np.asarray(offsets[indices + 1]) - np.asarray(offsets[indices])
            ) <= 1
        weights = np.asarray(
            fixed["policy_weight_multiplier"][indices], dtype=np.float32
        )
        full_mask = (
            np.asarray(fixed["used_full_search"][indices], dtype=bool)
            if has_used_full_search
            else ((weights > 0) & ~forced_mask)
        )
        policy_active = (~forced_mask) & (weights > 0)
        forced += int(forced_mask.sum())
        full += int(full_mask.sum())
        active += int(policy_active.sum())
        codes = np.asarray(phase_codes[indices], dtype=np.int32)
        phase_counts.update(phases[int(code)] for code in codes)

        active_indices = indices[policy_active]
        active_codes = codes[policy_active]
        if active_indices.size:
            # Compute all rows in this contiguous chunk with segmented
            # reductions, then select the requested active rows.  A Python
            # loop over millions of ragged rows made an earlier ad-hoc audit
            # needlessly I/O-bound even though the flat payload is contiguous.
            bounds = np.asarray(offsets[start : stop + 1], dtype=np.int64)
            lengths = np.diff(bounds)
            if np.any(lengths <= 0):
                raise ValueError("policy ragged rows must be non-empty")
            flat_start, flat_stop = int(bounds[0]), int(bounds[-1])
            starts = bounds[:-1] - flat_start
            target = np.asarray(target_flat[flat_start:flat_stop], dtype=np.float64)
            prior = np.asarray(prior_flat[flat_start:flat_stop], dtype=np.float64)
            target_sums = np.add.reduceat(target, starts)
            prior_sums = np.add.reduceat(prior, starts)
            target /= np.repeat(np.maximum(target_sums, EPS), lengths)
            prior /= np.repeat(np.maximum(prior_sums, EPS), lengths)
            target_terms = np.where(
                target > 0, target * np.log(np.maximum(target, EPS)), 0.0
            )
            prior_terms = np.where(
                prior > 0, prior * np.log(np.maximum(prior, EPS)), 0.0
            )
            kl_terms = np.where(
                target > 0,
                target
                * (np.log(np.maximum(target, EPS)) - np.log(np.maximum(prior, EPS))),
                0.0,
            )
            local_active = active_indices - start
            entropy_parts.append(-np.add.reduceat(target_terms, starts)[local_active])
            prior_entropy_parts.append(
                -np.add.reduceat(prior_terms, starts)[local_active]
            )
            kl_parts.append(np.add.reduceat(kl_terms, starts)[local_active])
            active_phase_parts.append(active_codes)
            if has_action_blend and legal_flat is not None:
                legal = np.asarray(legal_flat[flat_start:flat_stop], dtype=np.int64)
                row_for_flat = np.repeat(np.arange(stop - start), lengths)
                played = np.asarray(fixed["action_taken"][start:stop], dtype=np.int64)
                matches = legal == played[row_for_flat]
                match_counts = np.add.reduceat(matches.astype(np.int8), starts)
                if np.any(match_counts[local_active] != 1):
                    bad_local = int(
                        local_active[np.flatnonzero(match_counts[local_active] != 1)[0]]
                    )
                    raise ValueError(
                        "played action must match exactly one legal target column; "
                        f"row={start + bad_local} matches={int(match_counts[bad_local])}"
                    )
                played_probability = np.add.reduceat(
                    np.where(matches, target, 0.0), starts
                )
                target_mode = np.maximum.reduceat(target, starts)
                alpha = float(np.clip(soft_target_weight, 0.0, 1.0))
                blended = alpha * target + (1.0 - alpha) * matches
                blend_entropy = -np.add.reduceat(
                    np.where(
                        blended > 0,
                        blended * np.log(np.maximum(blended, EPS)),
                        0.0,
                    ),
                    starts,
                )
                blend_kl = np.add.reduceat(
                    np.where(
                        blended > 0,
                        blended
                        * (
                            np.log(np.maximum(blended, EPS))
                            - np.log(np.maximum(target, EPS))
                        ),
                        0.0,
                    ),
                    starts,
                )
                blend_parts["played_target_probability"].append(
                    played_probability[local_active]
                )
                blend_parts["played_is_target_mode"].append(
                    played_probability[local_active]
                    >= target_mode[local_active] - 1e-12
                )
                blend_parts["hard_blend_entropy"].append(blend_entropy[local_active])
                blend_parts["kl_hard_blend_target"].append(blend_kl[local_active])

    derived = {
        "target_entropy": np.concatenate(entropy_parts)
        if entropy_parts
        else np.empty(0),
        "prior_entropy": np.concatenate(prior_entropy_parts)
        if prior_entropy_parts
        else np.empty(0),
        "kl_target_prior": np.concatenate(kl_parts) if kl_parts else np.empty(0),
        "active_phase_codes": np.concatenate(active_phase_parts)
        if active_phase_parts
        else np.empty(0, dtype=np.int32),
        "phase_categories": phases,
        "soft_target_weight": float(soft_target_weight),
    }
    if has_action_blend:
        derived.update(
            {
                name: np.concatenate(parts) if parts else np.empty(0)
                for name, parts in blend_parts.items()
            }
        )
    report = _finalize_report(
        rows=rows,
        game_counts=game_counts,
        forced=forced,
        full=full,
        active=active,
        phase_counts=phase_counts,
        truncated_games=truncated_games,
        terminated_games=terminated_games,
        failures=0,
        derived=derived,
        simulations=None,
    )
    provenance = {
        "kind": "memmap_corpus_v1",
        "path": str(root),
        "corpus_meta_sha256": _sha256(meta_path),
        "payload_inventory_sha256": meta.get("payload_inventory_sha256"),
        "seed_manifest": str(seed_manifest) if seed_manifest else None,
        "seed_manifest_sha256": _sha256(seed_manifest) if seed_manifest else None,
        "category": category,
    }
    return report, provenance


def _progress_failures(root: Path) -> int:
    total = 0
    for path in root.rglob("progress.json"):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        total += int(payload.get("games_failed", 0))
    return total


def analyze_npz(
    root: Path,
    *,
    compact_out: Path | None = None,
    soft_target_weight: float = 0.9,
) -> tuple[dict[str, Any], dict[str, Any]]:
    data, inventory = load_npz_target_corpus(root)
    seeds = np.asarray(data["game_seed"], dtype=np.int64)
    unique, counts = np.unique(seeds, return_counts=True)
    game_counts = Counter(
        {int(seed): int(count) for seed, count in zip(unique, counts)}
    )
    forced_mask = np.asarray(data.get("is_forced", np.zeros(len(seeds))), dtype=bool)
    full_mask = np.asarray(
        data.get("used_full_search", np.zeros(len(seeds))), dtype=bool
    )
    weights = np.asarray(
        data.get("policy_weight_multiplier", np.ones(len(seeds))), dtype=np.float32
    )
    legal_count = np.asarray(data["target_policy_mask"], dtype=bool).sum(axis=1)
    active_mask = (~forced_mask) & (weights > 0) & (legal_count > 1)
    target_entropy, prior_entropy, kl = _policy_metrics(
        data["target_policy"][active_mask],
        data["prior_policy"][active_mask],
        data["target_policy_mask"][active_mask],
    )
    blend_derived: dict[str, np.ndarray] = {}
    if "action_taken" in data and "legal_action_ids" in data:
        (
            blend_derived["played_target_probability"],
            blend_derived["played_is_target_mode"],
            blend_derived["hard_blend_entropy"],
            blend_derived["kl_hard_blend_target"],
        ) = _hard_blend_metrics(
            data["target_policy"][active_mask],
            data["legal_action_ids"][active_mask],
            data["action_taken"][active_mask],
            soft_target_weight=soft_target_weight,
        )
    phases_all = np.asarray(data.get("phase", np.full(len(seeds), "UNKNOWN"))).astype(
        str
    )
    phase_categories = sorted(set(phases_all.tolist()))
    phase_code_map = {phase: i for i, phase in enumerate(phase_categories)}
    active_codes = np.asarray(
        [phase_code_map[phase] for phase in phases_all[active_mask]], dtype=np.int32
    )
    truncated = np.asarray(data.get("truncated", np.zeros(len(seeds))), dtype=bool)
    terminated = np.asarray(data.get("terminated", np.zeros(len(seeds))), dtype=bool)
    report = _finalize_report(
        rows=len(seeds),
        game_counts=game_counts,
        forced=int(forced_mask.sum()),
        full=int(full_mask.sum()),
        active=int(active_mask.sum()),
        phase_counts=Counter(phases_all.tolist()),
        truncated_games=set(map(int, seeds[truncated])),
        terminated_games=set(map(int, seeds[terminated])),
        failures=_progress_failures(root),
        derived={
            "target_entropy": target_entropy,
            "prior_entropy": prior_entropy,
            "kl_target_prior": kl,
            "active_phase_codes": active_codes,
            "phase_categories": phase_categories,
            "soft_target_weight": float(soft_target_weight),
            **blend_derived,
        },
        simulations=np.asarray(data["simulations_used"])
        if "simulations_used" in data
        else None,
    )
    compact_record = None
    if compact_out is not None:
        compact_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(compact_out, **data)
        compact_record = {
            "path": str(compact_out),
            "sha256": _sha256(compact_out),
            "size_bytes": compact_out.stat().st_size,
        }
    provenance = {
        "kind": "raw_npz_shards",
        "path": str(root),
        "shard_count": len(inventory),
        "shards": inventory,
        "inventory_sha256": f"sha256:{hashlib.sha256(json.dumps(inventory, sort_keys=True, separators=(',', ':')).encode()).hexdigest()}",
        "compact_corpus": compact_record,
    }
    return report, provenance


def _finalize_report(
    *,
    rows: int,
    game_counts: Counter[int],
    forced: int,
    full: int,
    active: int,
    phase_counts: Counter[str],
    truncated_games: set[int],
    terminated_games: set[int],
    failures: int,
    derived: dict[str, Any],
    simulations: np.ndarray | None,
) -> dict[str, Any]:
    row_counts = np.asarray(list(game_counts.values()), dtype=np.int64)
    phase_total = sum(phase_counts.values())
    active_phase_counts = Counter(
        derived["phase_categories"][int(code)] for code in derived["active_phase_codes"]
    )
    by_phase: dict[str, Any] = {}
    for phase, count in active_phase_counts.items():
        code = derived["phase_categories"].index(phase)
        mask = derived["active_phase_codes"] == code
        by_phase[phase] = {
            "rows": int(count),
            "target_entropy": _summary(derived["target_entropy"][mask]),
            "prior_entropy": _summary(derived["prior_entropy"][mask]),
            "kl_target_prior": _summary(derived["kl_target_prior"][mask]),
        }
    policy_targets = {
        "target_semantics": "completed_q_improved_policy_not_visit_counts",
        "soft_target_temperature_semantics": (
            "inert_for_policy_source_only_applies_when_converting_target_scores"
        ),
        "target_entropy": _summary(derived["target_entropy"]),
        "prior_entropy": _summary(derived["prior_entropy"]),
        "kl_target_prior": _summary(derived["kl_target_prior"]),
        "by_phase": by_phase,
    }
    if "played_target_probability" in derived:
        policy_targets["played_action_blend"] = {
            "soft_target_weight": float(derived["soft_target_weight"]),
            "hard_action_weight": float(1.0 - derived["soft_target_weight"]),
            "played_target_probability": _summary(derived["played_target_probability"]),
            "played_is_target_mode_fraction": _fraction(
                derived["played_is_target_mode"]
            ),
            "effective_target_entropy": _summary(derived["hard_blend_entropy"]),
            "kl_effective_target_to_search_target": _summary(
                derived["kl_hard_blend_target"]
            ),
        }
    return {
        "rows": int(rows),
        "games": int(len(game_counts)),
        "rows_per_game": _summary(row_counts),
        "failures": int(failures),
        "terminated_games": int(len(terminated_games)),
        "truncated_games": int(len(truncated_games)),
        "forced": {
            "rows": int(forced),
            "fraction": float(forced / rows) if rows else None,
        },
        "full_search": {
            "rows": int(full),
            "fraction": float(full / rows) if rows else None,
        },
        "policy_active": {
            "rows": int(active),
            "fraction": float(active / rows) if rows else None,
        },
        "phase_distribution": {
            phase: {
                "rows": int(count),
                "fraction": float(count / phase_total) if phase_total else None,
            }
            for phase, count in sorted(phase_counts.items())
        },
        "policy_targets": policy_targets,
        "simulations_per_row": _summary(simulations)
        if simulations is not None
        else None,
    }


def compare_reports(
    candidate: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    def delta(path: Iterable[str]) -> dict[str, float | None]:
        left: Any = candidate
        right: Any = reference
        for key in path:
            left, right = left[key], right[key]
        if left is None or right is None:
            return {
                "candidate": left,
                "reference": right,
                "absolute_delta": None,
                "relative_delta": None,
            }
        absolute = float(left - right)
        relative = float(absolute / abs(right)) if right != 0 else None
        return {
            "candidate": float(left),
            "reference": float(right),
            "absolute_delta": absolute,
            "relative_delta": relative,
        }

    result = {
        "forced_fraction": delta(("forced", "fraction")),
        "full_search_fraction": delta(("full_search", "fraction")),
        "policy_active_fraction": delta(("policy_active", "fraction")),
        "rows_per_game_mean": delta(("rows_per_game", "mean")),
        "target_entropy_mean": delta(("policy_targets", "target_entropy", "mean")),
        "prior_entropy_mean": delta(("policy_targets", "prior_entropy", "mean")),
        "kl_target_prior_mean": delta(("policy_targets", "kl_target_prior", "mean")),
        "phase_fraction": {},
        "policy_targets_by_phase": {},
    }
    if (
        candidate["policy_targets"].get("played_action_blend") is not None
        and reference["policy_targets"].get("played_action_blend") is not None
    ):
        result["played_action_blend"] = {
            metric: delta(
                (
                    "policy_targets",
                    "played_action_blend",
                    report_key,
                    "mean",
                )
            )
            for metric, report_key in (
                ("played_target_probability_mean", "played_target_probability"),
                ("effective_target_entropy_mean", "effective_target_entropy"),
                (
                    "kl_effective_target_to_search_target_mean",
                    "kl_effective_target_to_search_target",
                ),
            )
        }
    phases = set(candidate["phase_distribution"]) | set(reference["phase_distribution"])
    for phase in sorted(phases):
        left = candidate["phase_distribution"].get(phase, {}).get("fraction", 0.0)
        right = reference["phase_distribution"].get(phase, {}).get("fraction", 0.0)
        result["phase_fraction"][phase] = {
            "candidate": left,
            "reference": right,
            "absolute_delta": left - right,
        }
        candidate_phase = candidate["policy_targets"]["by_phase"].get(phase)
        reference_phase = reference["policy_targets"]["by_phase"].get(phase)
        if candidate_phase is not None and reference_phase is not None:
            result["policy_targets_by_phase"][phase] = {}
            for metric in ("target_entropy", "prior_entropy", "kl_target_prior"):
                left_mean = candidate_phase[metric]["mean"]
                right_mean = reference_phase[metric]["mean"]
                absolute = float(left_mean - right_mean)
                result["policy_targets_by_phase"][phase][metric] = {
                    "candidate": float(left_mean),
                    "reference": float(right_mean),
                    "absolute_delta": absolute,
                    "relative_delta": float(absolute / abs(right_mean))
                    if right_mean
                    else None,
                }
    entropy_delta = abs(result["target_entropy_mean"]["absolute_delta"] or 0.0)
    kl_relative = abs(result["kl_target_prior_mean"]["relative_delta"] or 0.0)
    phase_entropy_delta = max(
        (
            abs(metrics["target_entropy"]["absolute_delta"])
            for metrics in result["policy_targets_by_phase"].values()
        ),
        default=0.0,
    )
    phase_kl_relative = max(
        (
            abs(metrics["kl_target_prior"]["relative_delta"] or 0.0)
            for metrics in result["policy_targets_by_phase"].values()
        ),
        default=0.0,
    )
    result["maximum_phase_target_entropy_absolute_delta"] = phase_entropy_delta
    result["maximum_phase_kl_relative_delta"] = phase_kl_relative
    result["material_target_change"] = bool(
        entropy_delta >= 0.05
        or kl_relative >= 0.10
        or phase_entropy_delta >= 0.05
        or phase_kl_relative >= 0.10
    )
    result["materiality_rule"] = (
        "overall or per-phase abs(target_entropy_mean_delta)>=0.05 nats OR "
        "abs(KL_mean_relative_delta)>=10%"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-npz-root", type=Path, required=True)
    parser.add_argument("--reference-memmap", type=Path, required=True)
    parser.add_argument("--reference-seed-manifest", type=Path)
    parser.add_argument("--reference-category")
    parser.add_argument("--compact-out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--soft-target-weight",
        type=float,
        default=0.9,
        help="Learner soft-label blend weight whose effective target should be audited.",
    )
    args = parser.parse_args()
    candidate, candidate_provenance = analyze_npz(
        args.candidate_npz_root,
        compact_out=args.compact_out,
        soft_target_weight=args.soft_target_weight,
    )
    reference, reference_provenance = analyze_memmap(
        args.reference_memmap,
        seed_manifest=args.reference_seed_manifest,
        category=args.reference_category,
        soft_target_weight=args.soft_target_weight,
    )
    payload = {
        "schema_version": "teacher-target-distribution-comparison-v1",
        "candidate": candidate,
        "reference": reference,
        "comparison": compare_reports(candidate, reference),
        "provenance": {
            "candidate": candidate_provenance,
            "reference": reference_provenance,
        },
    }
    _json_dump(args.out, payload)
    print(json.dumps(payload["comparison"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
