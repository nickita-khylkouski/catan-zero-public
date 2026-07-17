#!/usr/bin/env python3
"""Inventory the policy-target operator and exact reanalysis eligibility.

This tool answers two questions that a row-count or corpus hash cannot:

* Which search operator produced the policy targets that the learner can
  actually sample?
* Can every such trajectory be reconstructed exactly for root reanalysis?

The inventory reads only compact scalar memmap columns and authenticated JSON
metadata.  It never materializes observations or target tensors.  A corpus is
declared fully replayable only when every game has one contiguous decision
trace starting at zero; a deterministic game seed is not a substitute for the
opponent actions that were intentionally omitted from own-side-only data.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from catan_zero.rl.action_mask import ActionCatalog
from catan_zero.rl.entity_feature_adapter import (
    ENTITY_FEATURE_ADAPTER_CHECKPOINT_SCHEMA,
)
from catan_zero.search.gumbel_chance_mcts import GumbelChanceMCTSConfig
from catan_zero.search.neural_rust_mcts import EntityGraphRustEvaluatorConfig


SCHEMA = "a1-target-eligibility-inventory-v1"
RD_CONTRACT_SCHEMA = "a1-coherent-target-rd-contract-v1"
RD_CONTRACT_SCHEMA_V2 = "a1-coherent-target-rd-contract-v2"
RD_CONTRACT_SCHEMAS = frozenset({RD_CONTRACT_SCHEMA, RD_CONTRACT_SCHEMA_V2})
PIMC_REGIME = "public_conservation_pimc_v1"
COHERENT_REGIME = "public_belief_single_tree_v1"

TRACE_COLUMNS = frozenset(
    {
        "game_seed",
        "decision_index",
        "action_taken",
        "phase",
        "player",
        "terminated",
        "truncated",
    }
)
ROUND_TRIP_COLUMNS = frozenset(
    {
        "legal_action_ids",
        "legal_action_context",
        "hex_tokens",
        "vertex_tokens",
        "edge_tokens",
        "player_tokens",
        "global_tokens",
        "event_tokens",
        "hex_mask",
        "vertex_mask",
        "edge_mask",
        "player_mask",
        "event_mask",
        "legal_action_tokens",
        "legal_action_mask",
        "legal_action_target_ids",
    }
)
MIRROR_PROVENANCE_COLUMNS = frozenset(
    {"is_pool_game", "opponent_version", "opponent_tag", "opponent_checkpoint_md5"}
)
SERIALIZED_STATE_COLUMNS = frozenset(
    {"serialized_game_state", "authoritative_state", "rng_state", "chance_trace"}
)
POLICY_COLUMNS = frozenset(
    {"policy_weight_multiplier", "target_information_regime"}
)
FORCED_VALUE_COLUMNS = frozenset(
    {
        "game_seed",
        "phase",
        "action_taken",
        "is_forced",
        "policy_weight_multiplier",
        "value_weight_multiplier",
    }
)
SEARCH_EVIDENCE_BASE_COLUMNS = frozenset(
    {
        "search_evidence_version",
        "search_evidence_mask",
        "search_evidence_offsets",
        "search_visit_counts_flat",
        "search_completed_q_flat",
    }
)
SEARCH_EVIDENCE_V2_COLUMNS = frozenset(
    {*SEARCH_EVIDENCE_BASE_COLUMNS, "search_prior_policy_flat"}
)
SEARCH_EVIDENCE_V1_SCHEMA = "gumbel_root_search_evidence_v1"
SEARCH_EVIDENCE_V2_SCHEMA = "gumbel_root_search_evidence_v2_fp32_prior"
SUPPORTED_SEARCH_EVIDENCE_SCHEMAS = frozenset(
    {SEARCH_EVIDENCE_V1_SCHEMA, SEARCH_EVIDENCE_V2_SCHEMA}
)


# Historical manifests were fingerprinted with this deliberately incomplete
# flat projection.  Keep it append-only so archived composites remain
# inspectable under the exact identity rule that originally admitted them.
LEGACY_POLICY_TARGET_IDENTITY_SCHEMA = (
    "policy-target-manifest-operator-identity-v1-legacy-partial"
)
POLICY_TARGET_IDENTITY_SCHEMA = "policy-target-teacher-identity-v2"
OPERATOR_FIELDS = (
    "producer_checkpoint_sha256",
    "target_information_regime",
    "public_observation",
    "information_set_search",
    "coherent_public_belief_search",
    "determinization_particles",
    "information_set_target_aggregation",
    "n_full",
    "n_fast",
    "p_full",
    "n_full_wide",
    "n_full_wide_threshold",
    "wide_roots_always_full",
    "c_scale",
    "correct_rust_chance_spectra",
    "lazy_interior_chance",
    "exact_budget_sh",
    "exact_budget_sh_min_n",
    "root_wave_batching",
    "native_mcts_hot_loop",
    "rust_featurize",
    "value_readout",
    "value_scale",
    "prior_temperature",
    "sigma_eval",
    "symmetry_averaged_eval",
    "symmetry_averaged_eval_threshold",
    "forced_root_target_mode",
    "preserve_search_evidence",
    "search_evidence_schema",
)
TARGET_SEMANTIC_FIELDS = (
    "target_information_regime",
    # These alter the produced policy distribution rather than only runtime
    # throughput. Boundary K changes coherent opponent-turn continuation
    # values; minimum visits explicitly prunes target support. Keep both in
    # the immutable target identity so K1/K4 or pruned/unpruned shards cannot
    # be merged as one teacher.
    "boundary_value_particles",
    "policy_target_min_visits",
    # Policy targets are stored in the legal-row order induced by the action
    # catalog.  A rules/catalog change can preserve every MCTS scalar while
    # changing what an integer action id means, so it is an immutable teacher
    # semantic—not merely row metadata.
    "action_mask_version",
    "forced_root_target_mode",
    "record_automatic_transitions",
    "public_observation",
    "meaningful_public_history",
    "event_history_limit",
    "meaningful_public_history_schema",
    "public_card_count_feature_schema",
    "temperature_clock",
    "preserve_search_evidence",
    "search_evidence_schema",
)
EXECUTION_SEMANTIC_FIELDS = (
    "native_mcts_hot_loop",
    "root_wave_batching",
    "rust_featurize",
)
PRODUCER_CODE_IDENTITY_FIELDS = (
    "runtime_code_tree_sha256",
    "reviewed_code_tree_sha256",
    "checkout_tree_sha256",
    "code_tree_sha256",
    "git_commit",
    "producer_git_commit",
)


class InventoryError(RuntimeError):
    """The input is malformed or cannot support an exact inventory."""


def _required_search_evidence_schema(value: Mapping[str, Any], *, where: Path) -> str:
    schema = value.get("acceptance", {}).get("require_search_evidence_schema")
    if schema not in SUPPORTED_SEARCH_EVIDENCE_SCHEMAS:
        raise InventoryError(f"{where}: search-evidence acceptance schema drift")
    return str(schema)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _value_sha256(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _self_digest(value: Mapping[str, Any], field: str) -> str:
    unhashed = dict(value)
    unhashed.pop(field, None)
    return _value_sha256(unhashed)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InventoryError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise InventoryError(f"expected a JSON object in {path}")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("xb") as handle:
        handle.write(_canonical_bytes(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _column(
    root: Path, meta: Mapping[str, Any], name: str
) -> np.memmap:
    columns = meta.get("columns")
    if not isinstance(columns, dict) or not isinstance(columns.get(name), dict):
        raise InventoryError(f"{root}: missing memmap column {name!r}")
    schema = columns[name]
    if schema.get("kind") != "fixed":
        raise InventoryError(f"{root}: {name!r} is not a fixed memmap column")
    rows = int(meta.get("row_count", -1))
    if rows < 0:
        raise InventoryError(f"{root}: invalid row_count")
    inner = tuple(int(value) for value in schema.get("inner_shape", ()))
    return np.memmap(
        root / f"{name}.dat",
        mode="r",
        dtype=np.dtype(schema["dtype"]),
        shape=(rows, *inner),
    )


def _categorical_codes(
    root: Path, meta: Mapping[str, Any], name: str
) -> tuple[np.memmap, tuple[str, ...]]:
    columns = meta.get("columns")
    if not isinstance(columns, dict) or not isinstance(columns.get(name), dict):
        raise InventoryError(f"{root}: missing categorical column {name!r}")
    schema = columns[name]
    if schema.get("kind") != "string":
        raise InventoryError(f"{root}: {name!r} is not categorical")
    categories = tuple(str(value) for value in schema.get("categories", ()))
    rows = int(meta["row_count"])
    codes = np.memmap(
        root / f"{name}.codes.dat", mode="r", dtype=np.int32, shape=(rows,)
    )
    if codes.size and (
        int(np.min(codes)) < 0 or int(np.max(codes)) >= len(categories)
    ):
        raise InventoryError(f"{root}: {name!r} contains an invalid category code")
    return codes, categories


def _regime_counts(
    root: Path, meta: Mapping[str, Any], policy: np.ndarray
) -> tuple[dict[str, int], dict[str, int]]:
    codes, categories = _categorical_codes(root, meta, "target_information_regime")
    total = np.bincount(np.asarray(codes), minlength=len(categories))
    active = np.bincount(
        np.asarray(codes)[np.asarray(policy) > 0.0], minlength=len(categories)
    )
    return (
        {category: int(total[index]) for index, category in enumerate(categories)},
        {category: int(active[index]) for index, category in enumerate(categories)},
    )


def _forced_value_inventory(
    root: Path, meta: Mapping[str, Any]
) -> dict[str, Any]:
    """Measure whether sole-action states have real value-only authority.

    Percentages are reported for diagnosis, but the admission contract uses
    structural invariants: at least one row, coverage in every complete game,
    exactly zero forced-row policy authority, and exactly one forced-row value
    multiplier. Phase and selected-action counts make omissions visible
    without pinning a brittle historical forced-row percentage.
    """

    columns = set(meta.get("columns", {}))
    missing = sorted(FORCED_VALUE_COLUMNS - columns)
    if missing:
        return {
            "present": False,
            "missing_columns": missing,
            "rows": int(meta.get("row_count", 0)),
            "forced_rows": None,
            "forced_fraction": None,
            "game_count": None,
            "games_with_forced_rows": None,
            "forced_game_coverage": None,
            "forced_policy_active_rows": None,
            "forced_value_inactive_rows": None,
            "phase_counts": {},
            "forced_phase_counts": {},
            "action_taken_counts": {},
            "forced_action_taken_counts": {},
            "action_type_counts": {},
            "forced_action_type_counts": {},
            "contract_passed": False,
        }

    forced = np.asarray(_column(root, meta, "is_forced").reshape(-1))
    if forced.dtype.kind != "b":
        raise InventoryError(f"{root}: is_forced is not boolean")
    seeds = np.asarray(_column(root, meta, "game_seed").reshape(-1))
    actions = np.asarray(_column(root, meta, "action_taken").reshape(-1))
    policy = np.asarray(
        _column(root, meta, "policy_weight_multiplier").reshape(-1),
        dtype=np.float64,
    )
    value = np.asarray(
        _column(root, meta, "value_weight_multiplier").reshape(-1),
        dtype=np.float64,
    )
    rows = int(meta["row_count"])
    if any(array.shape != (rows,) for array in (forced, seeds, actions, policy, value)):
        raise InventoryError(f"{root}: forced-value columns are not row-aligned")

    phase_codes, phase_categories = _categorical_codes(root, meta, "phase")
    phase_total = np.bincount(
        np.asarray(phase_codes), minlength=len(phase_categories)
    )
    phase_forced = np.bincount(
        np.asarray(phase_codes)[forced], minlength=len(phase_categories)
    )

    def _integer_counts(raw: np.ndarray) -> dict[str, int]:
        unique, counts = np.unique(raw, return_counts=True)
        return {
            str(int(key)): int(count)
            for key, count in zip(unique.tolist(), counts.tolist())
        }

    catalog = ActionCatalog(("RED", "BLUE"))

    def _action_type_counts(raw: np.ndarray) -> dict[str, int]:
        counts: dict[str, int] = {}
        unique, occurrences = np.unique(raw, return_counts=True)
        for action_id, occurrence_count in zip(
            unique.tolist(), occurrences.tolist()
        ):
            index = int(action_id)
            if not 0 <= index < catalog.size:
                raise InventoryError(
                    f"{root}: action_taken id {index} is outside the 2p action catalog"
                )
            action_type = catalog.descriptor(index).action_type
            counts[action_type] = (
                counts.get(action_type, 0) + int(occurrence_count)
            )
        return dict(sorted(counts.items()))

    forced_rows = int(np.count_nonzero(forced))
    game_count = int(np.unique(seeds).size)
    forced_game_count = int(np.unique(seeds[forced]).size) if forced_rows else 0
    forced_policy_active = int(
        np.count_nonzero(forced & (~np.isfinite(policy) | (policy != 0.0)))
    )
    forced_value_inactive = int(
        np.count_nonzero(forced & (~np.isfinite(value) | (value != 1.0)))
    )
    contract_passed = bool(
        rows > 0
        and forced_rows > 0
        and game_count > 0
        and forced_game_count == game_count
        and forced_policy_active == 0
        and forced_value_inactive == 0
    )
    return {
        "present": True,
        "missing_columns": [],
        "rows": rows,
        "forced_rows": forced_rows,
        "forced_fraction": float(forced_rows / rows) if rows else None,
        "game_count": game_count,
        "games_with_forced_rows": forced_game_count,
        "forced_game_coverage": (
            float(forced_game_count / game_count) if game_count else None
        ),
        "forced_policy_active_rows": forced_policy_active,
        "forced_value_inactive_rows": forced_value_inactive,
        "phase_counts": {
            category: int(phase_total[index])
            for index, category in enumerate(phase_categories)
            if int(phase_total[index]) > 0
        },
        "forced_phase_counts": {
            category: int(phase_forced[index])
            for index, category in enumerate(phase_categories)
            if int(phase_forced[index]) > 0
        },
        "action_taken_counts": _integer_counts(actions),
        "forced_action_taken_counts": _integer_counts(actions[forced]),
        "action_type_counts": _action_type_counts(actions),
        "forced_action_type_counts": _action_type_counts(actions[forced]),
        "contract_passed": contract_passed,
    }


def _search_evidence_inventory(
    root: Path, meta: Mapping[str, Any], policy: np.ndarray
) -> dict[str, Any]:
    """Authenticate the row-addressable memmap form of compact root evidence."""

    columns = meta.get("columns")
    if not isinstance(columns, dict):
        raise InventoryError(f"{root}: missing memmap columns")
    present = SEARCH_EVIDENCE_V2_COLUMNS & set(columns)
    if not present:
        return {
            "present": False,
            "schema": None,
            "row_addressing": None,
            "policy_active_alignment": None,
            "active_rows": 0,
            "flat_entries": 0,
        }
    declared = meta.get("search_evidence")
    declared_schema = (
        declared.get("schema") if isinstance(declared, dict) else None
    )
    expected_columns = (
        SEARCH_EVIDENCE_V2_COLUMNS
        if declared_schema == SEARCH_EVIDENCE_V2_SCHEMA
        else SEARCH_EVIDENCE_BASE_COLUMNS
    )
    if present != expected_columns:
        raise InventoryError(
            f"{root}: incomplete row-addressable search evidence; "
            f"missing={sorted(expected_columns - present)}"
        )
    expected_schemas = {
        "search_evidence_version": ("fixed", np.dtype(np.uint8), []),
        "search_evidence_mask": ("fixed", np.dtype(np.bool_), []),
        "search_evidence_offsets": ("row_offsets", np.dtype(np.int64), None),
        "search_visit_counts_flat": (
            "independent_ragged1d",
            np.dtype(np.uint16),
            None,
        ),
        "search_completed_q_flat": (
            "independent_ragged1d",
            np.dtype(np.float32),
            None,
        ),
    }
    if declared_schema == SEARCH_EVIDENCE_V2_SCHEMA:
        expected_schemas["search_prior_policy_flat"] = (
            "independent_ragged1d",
            np.dtype(np.float32),
            None,
        )
    for name, (kind, dtype, inner) in expected_schemas.items():
        schema = columns[name]
        if (
            not isinstance(schema, dict)
            or schema.get("kind") != kind
            or np.dtype(schema.get("dtype")) != dtype
            or (inner is not None and schema.get("inner_shape") != inner)
        ):
            raise InventoryError(f"{root}: malformed search evidence schema {name!r}")
        if kind == "independent_ragged1d" and schema.get("offsets") != (
            "search_evidence_offsets"
        ):
            raise InventoryError(f"{root}: {name!r} lost its evidence offsets binding")

    rows = int(meta.get("row_count", -1))
    offsets_path = root / "search_evidence_offsets.dat"
    if offsets_path.stat().st_size != (rows + 1) * np.dtype(np.int64).itemsize:
        raise InventoryError(f"{root}: search evidence offsets byte length drift")
    offsets = np.memmap(
        offsets_path, mode="r", dtype=np.int64, shape=(rows + 1,)
    )
    if int(offsets[0]) != 0 or bool(np.any(offsets[1:] < offsets[:-1])):
        raise InventoryError(f"{root}: search evidence offsets are not monotone")
    lengths = np.asarray(offsets[1:] - offsets[:-1])
    mask = np.asarray(_column(root, meta, "search_evidence_mask").reshape(-1))
    version = np.asarray(_column(root, meta, "search_evidence_version").reshape(-1))
    expected_active = np.asarray(policy) > 0.0
    if (
        not np.array_equal(mask, expected_active)
        or bool(np.any(lengths[~mask] != 0))
        or bool(np.any(lengths[mask] <= 0))
        or bool(
            np.any(
                version[mask]
                != (
                    2
                    if declared_schema == SEARCH_EVIDENCE_V2_SCHEMA
                    else 1
                )
            )
        )
        or bool(np.any(version[~mask] != 0))
    ):
        raise InventoryError(f"{root}: search evidence is not policy-row aligned")

    legal_offsets = np.fromfile(root / "row_offsets.dat", dtype=np.int64)
    if legal_offsets.shape != (rows + 1,) or not np.array_equal(
        lengths[mask], np.diff(legal_offsets)[mask]
    ):
        raise InventoryError(f"{root}: search evidence/legal widths differ")
    flat_entries = int(offsets[-1])
    expected_sizes = {
        "search_visit_counts_flat.dat": flat_entries * np.dtype(np.uint16).itemsize,
        "search_completed_q_flat.dat": flat_entries * np.dtype(np.float32).itemsize,
    }
    if declared_schema == SEARCH_EVIDENCE_V2_SCHEMA:
        expected_sizes["search_prior_policy_flat.dat"] = (
            flat_entries * np.dtype(np.float32).itemsize
        )
    for filename, expected_size in expected_sizes.items():
        if (root / filename).stat().st_size != expected_size:
            raise InventoryError(f"{root}: {filename} byte length drift")
    if flat_entries:
        completed_q = np.memmap(
            root / "search_completed_q_flat.dat",
            mode="r",
            dtype=np.float32,
            shape=(flat_entries,),
        )
        if bool(np.any(~np.isfinite(completed_q))):
            raise InventoryError(f"{root}: non-finite completed-Q evidence")
        if declared_schema == SEARCH_EVIDENCE_V2_SCHEMA:
            prior = np.memmap(
                root / "search_prior_policy_flat.dat",
                mode="r",
                dtype=np.float32,
                shape=(flat_entries,),
            )
            if bool(np.any(~np.isfinite(prior))) or bool(np.any(prior < 0.0)):
                raise InventoryError(f"{root}: invalid fp32 prior-policy evidence")
            prior_mass = np.add.reduceat(
                prior.astype(np.float64, copy=False),
                np.asarray(offsets[:-1][mask]),
            )
            if bool(np.any(~np.isfinite(prior_mass))) or bool(
                np.any(prior_mass <= 0.0)
            ):
                raise InventoryError(
                    f"{root}: fp32 prior-policy has zero active-row mass"
                )
    if (
        not isinstance(declared, dict)
        or declared_schema
        not in SUPPORTED_SEARCH_EVIDENCE_SCHEMAS
        or declared.get("row_addressing") != "all_rows_empty_inactive_v1"
        or declared.get("active_row_count") != int(np.count_nonzero(mask))
        or declared.get("flat_entry_count") != flat_entries
    ):
        raise InventoryError(f"{root}: search evidence metadata drift")
    return {
        "present": True,
        "schema": declared["schema"],
        "row_addressing": declared["row_addressing"],
        "policy_active_alignment": True,
        "active_rows": int(np.count_nonzero(mask)),
        "flat_entries": flat_entries,
    }
def _trajectory_inventory(
    root: Path, meta: Mapping[str, Any], *, chunk_rows: int = 1_000_000
) -> dict[str, Any]:
    columns = set(meta.get("columns", {}))
    missing_trace = sorted(TRACE_COLUMNS - columns)
    missing_round_trip = sorted(ROUND_TRIP_COLUMNS - columns)
    missing_mirror = sorted(MIRROR_PROVENANCE_COLUMNS - columns)
    serialized = sorted(SERIALIZED_STATE_COLUMNS & columns)
    if missing_trace:
        return {
            "method": "unavailable",
            "game_count": None,
            "complete_action_trace_game_count": 0,
            "incomplete_action_trace_game_count": None,
            "complete_action_trace_fraction": 0.0,
            "full_corpus_replayable": False,
            "missing_trace_columns": missing_trace,
            "missing_round_trip_columns": missing_round_trip,
            "missing_mirror_provenance_columns": missing_mirror,
            "serialized_state_columns": serialized,
            "blockers": ["missing_decision_trace"],
        }

    seeds = _column(root, meta, "game_seed").reshape(-1)
    decisions = _column(root, meta, "decision_index").reshape(-1)
    terminated = _column(root, meta, "terminated").reshape(-1)
    truncated = _column(root, meta, "truncated").reshape(-1)
    rows = int(meta["row_count"])

    seen: set[int] = set()
    duplicate_runs: set[int] = set()
    gap_games: set[int] = set()
    nonzero_start_games: set[int] = set()
    no_completion_games: set[int] = set()
    run_count = 0
    previous_seed: int | None = None
    previous_decision: int | None = None
    previous_completed = False

    for offset in range(0, rows, chunk_rows):
        stop = min(offset + chunk_rows, rows)
        seed_chunk = np.asarray(seeds[offset:stop])
        decision_chunk = np.asarray(decisions[offset:stop])
        term_chunk = np.asarray(terminated[offset:stop], dtype=np.bool_)
        trunc_chunk = np.asarray(truncated[offset:stop], dtype=np.bool_)
        if not seed_chunk.size:
            continue

        if previous_seed is not None:
            first_seed = int(seed_chunk[0])
            if first_seed == previous_seed:
                if int(decision_chunk[0]) != int(previous_decision) + 1:
                    gap_games.add(first_seed)
            else:
                if not previous_completed:
                    no_completion_games.add(previous_seed)
                run_count += 1
                if first_seed in seen:
                    duplicate_runs.add(first_seed)
                seen.add(first_seed)
                if int(decision_chunk[0]) != 0:
                    nonzero_start_games.add(first_seed)
        else:
            first_seed = int(seed_chunk[0])
            run_count = 1
            seen.add(first_seed)
            if int(decision_chunk[0]) != 0:
                nonzero_start_games.add(first_seed)

        changes = np.flatnonzero(seed_chunk[1:] != seed_chunk[:-1]) + 1
        same = seed_chunk[1:] == seed_chunk[:-1]
        gaps = same & (decision_chunk[1:] != decision_chunk[:-1] + 1)
        gap_games.update(int(value) for value in np.unique(seed_chunk[1:][gaps]))

        for start in changes.tolist():
            prior = start - 1
            prior_seed = int(seed_chunk[prior])
            if not bool(term_chunk[prior] or trunc_chunk[prior]):
                no_completion_games.add(prior_seed)
            seed = int(seed_chunk[start])
            run_count += 1
            if seed in seen:
                duplicate_runs.add(seed)
            seen.add(seed)
            if int(decision_chunk[start]) != 0:
                nonzero_start_games.add(seed)

        previous_seed = int(seed_chunk[-1])
        previous_decision = int(decision_chunk[-1])
        previous_completed = bool(term_chunk[-1] or trunc_chunk[-1])

    if previous_seed is not None and not previous_completed:
        no_completion_games.add(previous_seed)

    incomplete = gap_games | nonzero_start_games | no_completion_games | duplicate_runs
    complete_count = len(seen - incomplete)
    blockers: list[str] = []
    if incomplete:
        blockers.append("noncontiguous_or_incomplete_action_trajectory")
    if missing_round_trip:
        blockers.append("missing_public_round_trip_surface")
    if run_count != len(seen):
        blockers.append("duplicate_game_seed_runs")
    # Missing mirror metadata does not invalidate a complete two-seat trace,
    # but it prevents proving that a partial trace is producer self-play.
    if missing_mirror and incomplete:
        blockers.append("partial_rows_lack_explicit_opponent_provenance")

    fully_replayable = not blockers and complete_count == len(seen)
    return {
        "method": "deterministic_game_seed_plus_contiguous_action_trace",
        "game_count": len(seen),
        "game_run_count": run_count,
        "complete_action_trace_game_count": complete_count,
        "incomplete_action_trace_game_count": len(incomplete),
        "complete_action_trace_fraction": (
            float(complete_count / len(seen)) if seen else 0.0
        ),
        "gap_game_count": len(gap_games),
        "nonzero_start_game_count": len(nonzero_start_games),
        "no_completion_game_count": len(no_completion_games),
        "duplicate_game_seed_count": len(duplicate_runs),
        "full_corpus_replayable": fully_replayable,
        "missing_trace_columns": missing_trace,
        "missing_round_trip_columns": missing_round_trip,
        "missing_mirror_provenance_columns": missing_mirror,
        "serialized_state_columns": serialized,
        "blockers": blockers,
    }


def _operator_family(regime: str) -> str:
    if regime == PIMC_REGIME:
        return "public_information_pimc_multi_tree"
    if regime == COHERENT_REGIME:
        return "coherent_public_belief_single_tree"
    if regime == "authoritative_hidden_state_search_v1":
        return "authoritative_hidden_state"
    return "unknown"


def inspect_memmap(
    *, label: str, corpus_dir: Path, required_regime: str
) -> dict[str, Any]:
    root = corpus_dir.expanduser().resolve(strict=True)
    meta_path = root / "corpus_meta.json"
    meta = _load_json(meta_path)
    if meta.get("schema") not in {"memmap_corpus_v1", "memmap_corpus_v2"}:
        raise InventoryError(f"{meta_path}: unsupported corpus schema")
    columns = set(meta.get("columns", {}))
    missing_policy = sorted(POLICY_COLUMNS - columns)
    if missing_policy:
        raise InventoryError(
            f"{root}: cannot inventory target eligibility; missing {missing_policy}"
        )

    policy = _column(root, meta, "policy_weight_multiplier").reshape(-1)
    policy_values = np.asarray(policy, dtype=np.float64)
    policy_active = policy_values > 0.0
    fast_policy_active_rows: int | None = None
    if {"used_full_search", "is_forced"} <= columns:
        used_full = np.asarray(
            _column(root, meta, "used_full_search").reshape(-1), dtype=np.bool_
        )
        forced = np.asarray(
            _column(root, meta, "is_forced").reshape(-1), dtype=np.bool_
        )
        full_non_forced = used_full & ~forced
        fast_non_forced = ~used_full & ~forced
        mismatch = (
            ~np.isfinite(policy_values)
            | (policy_values < 0.0)
            | (policy_values > 1.0)
            | (forced & (policy_values != 0.0))
            | (full_non_forced & (policy_values != 1.0))
        )
        positive_fast = fast_non_forced & (policy_values > 0.0)
        fast_policy_active_rows = int(np.count_nonzero(positive_fast))
        mismatch |= positive_fast
        activation_evidence = "exact_n128_full_only_with_fast_policy_zero"
        activation_mismatch: int | None = int(np.count_nonzero(mismatch))
    else:
        activation_mismatch = None
        activation_evidence = "policy_weight_multiplier_only_legacy_payload"
    regime_counts, active_regime_counts = _regime_counts(root, meta, policy)
    incompatible_active = sum(
        count
        for regime, count in active_regime_counts.items()
        if regime != required_regime
    )
    replay = _trajectory_inventory(root, meta)
    search_evidence = _search_evidence_inventory(root, meta, policy)
    forced_value = _forced_value_inventory(root, meta)
    return {
        "label": label,
        "corpus_dir": str(root),
        "corpus_meta": {
            "path": str(meta_path),
            "sha256": _file_sha256(meta_path),
            "schema": meta["schema"],
            "payload_inventory_sha256": meta.get("payload_inventory_sha256"),
        },
        "rows": int(meta["row_count"]),
        "selected_games": (
            meta.get("selected_game_seed_manifest", {}).get("selected_game_count")
            if isinstance(meta.get("selected_game_seed_manifest"), dict)
            else None
        ),
        "target_regime_rows": regime_counts,
        "target_operator_families": {
            regime: _operator_family(regime) for regime in regime_counts
        },
        "policy_active_rows": int(np.count_nonzero(policy_active)),
        "fast_search_policy_active_rows": fast_policy_active_rows,
        "policy_active_target_regime_rows": active_regime_counts,
        "policy_active_rule_mismatch_rows": activation_mismatch,
        "policy_activation_evidence": activation_evidence,
        "required_target_information_regime": required_regime,
        "incompatible_policy_active_rows": int(incompatible_active),
        "policy_targets_eligible_for_requested_learner": (
            incompatible_active == 0
            and activation_mismatch in (None, 0)
        ),
        "search_evidence_columns": sorted(
            name
            for name in columns
            if name.startswith("search_")
            or name in {"root_completed_q", "root_visit_counts"}
        ),
        "search_evidence": search_evidence,
        "exact_root_reanalysis": replay,
        "forced_value_rows": forced_value,
    }


def _operator_candidates(mapping: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    candidates: list[Mapping[str, Any]] = [mapping]
    for key in ("cli_args", "fields", "generation", "evaluator"):
        value = mapping.get(key)
        if isinstance(value, Mapping):
            candidates.append(value)
    for parent_key in ("science", "operator"):
        parent = mapping.get(parent_key)
        if isinstance(parent, Mapping):
            candidates.append(parent)
            for key in (
                "effective_search_config",
                "search_operator",
                "search",
                "evaluator",
                "generation",
            ):
                value = parent.get(key)
                if isinstance(value, Mapping):
                    candidates.append(value)
    return candidates


def _first_operator_value(
    candidates: Iterable[Mapping[str, Any]], field: str
) -> Any:
    for candidate in candidates:
        if field in candidate:
            return candidate[field]
    return None


def _nested_operator(mapping: Mapping[str, Any]) -> dict[str, Any]:
    candidates = _operator_candidates(mapping)
    result: dict[str, Any] = {}
    for field in OPERATOR_FIELDS:
        value = _first_operator_value(candidates, field)
        if value is not None:
            result[field] = value
    regime = result.get("target_information_regime")
    if isinstance(regime, str):
        result["operator_family"] = _operator_family(regime)
    return result


def _resolved_dataclass_config(
    cls: type[Any],
    candidates: list[Mapping[str, Any]],
    *,
    exclude: frozenset[str] = frozenset(),
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve effective defaults so newly added knobs change target identity."""

    defaults = dataclasses.asdict(cls())
    result: dict[str, Any] = {}
    overrides = {} if overrides is None else dict(overrides)
    for field in dataclasses.fields(cls):
        name = field.name
        if name in exclude:
            continue
        value = overrides.get(name, _first_operator_value(candidates, name))
        if value is None and name not in overrides:
            value = defaults[name]
        result[name] = value
    return json.loads(_canonical_bytes(result))


def _producer_code_identity(
    manifest: Mapping[str, Any],
    authority: Mapping[str, Any] | None,
) -> dict[str, Any]:
    candidates = _operator_candidates(manifest)
    if authority is not None:
        candidates.extend(_operator_candidates(authority))
        provenance = authority.get("provenance")
        if isinstance(provenance, Mapping):
            candidates.append(provenance)
    result = {
        field: value
        for field in PRODUCER_CODE_IDENTITY_FIELDS
        if (value := _first_operator_value(candidates, field)) is not None
    }
    return result


def _canonical_policy_target_identity(
    manifest: Mapping[str, Any],
    *,
    authority: Mapping[str, Any] | None,
    strict_current: bool,
) -> dict[str, Any]:
    """Build the scientific identity of a stored policy label.

    Current sealed waves resolve the complete live search/evaluator dataclass,
    including defaults.  Historical manifests retain the legacy projection
    because they cannot retroactively prove fields or code revisions that were
    never recorded.
    """

    if not strict_current:
        legacy = _nested_operator(manifest)
        return {
            "schema_version": LEGACY_POLICY_TARGET_IDENTITY_SCHEMA,
            "completeness": "historical_partial_explicit",
            "operator": legacy,
        }

    candidates = _operator_candidates(manifest)
    if authority is not None:
        candidates.extend(_operator_candidates(authority))
    checkpoint_sha = _first_operator_value(
        candidates, "producer_checkpoint_sha256"
    )
    teacher_adapter = _first_operator_value(
        candidates, "teacher_entity_feature_adapter_version"
    )
    code_identity = _producer_code_identity(manifest, authority)
    missing = []
    if not isinstance(checkpoint_sha, str) or not checkpoint_sha.startswith("sha256:"):
        missing.append("producer_checkpoint_sha256")
    if not isinstance(teacher_adapter, str) or not teacher_adapter:
        missing.append("teacher_entity_feature_adapter_version")
    if not code_identity:
        missing.append("producer_code_identity")
    regime = _first_operator_value(candidates, "target_information_regime")
    if not isinstance(regime, str) or not regime:
        missing.append("target_information_regime")
    for field in ("boundary_value_particles", "policy_target_min_visits"):
        value = _first_operator_value(candidates, field)
        if type(value) is not int or value < 0 or (
            field == "boundary_value_particles" and value < 1
        ):
            missing.append(field)
    action_mask_version = _first_operator_value(candidates, "action_mask_version")
    if not isinstance(action_mask_version, str) or not action_mask_version.strip():
        missing.append("action_mask_version")
    if missing:
        raise InventoryError(
            "current policy-target identity is incomplete; missing "
            f"{sorted(missing)}"
        )

    evaluator_overrides = {
        "entity_feature_adapter_version": teacher_adapter,
    }
    search = _resolved_dataclass_config(
        GumbelChanceMCTSConfig,
        candidates,
        exclude=frozenset({"seed"}),
    )
    evaluator = _resolved_dataclass_config(
        EntityGraphRustEvaluatorConfig,
        candidates,
        exclude=frozenset({"cache_size"}),
        overrides=evaluator_overrides,
    )
    target_semantics = {
        field: _first_operator_value(candidates, field)
        for field in TARGET_SEMANTIC_FIELDS
    }
    execution_semantics = {
        field: _first_operator_value(candidates, field)
        for field in EXECUTION_SEMANTIC_FIELDS
    }
    return {
        "schema_version": POLICY_TARGET_IDENTITY_SCHEMA,
        "completeness": "current_exact_fail_closed",
        "producer_checkpoint": {"sha256": checkpoint_sha},
        "producer_code_identity": code_identity,
        "teacher_feature_contract": {
            "entity_feature_adapter_schema": (
                ENTITY_FEATURE_ADAPTER_CHECKPOINT_SCHEMA
            ),
            "entity_feature_adapter_version": teacher_adapter,
            "public_card_count_feature_schema": target_semantics[
                "public_card_count_feature_schema"
            ],
            "meaningful_public_history_schema": target_semantics[
                "meaningful_public_history_schema"
            ],
        },
        "target_information_regime": regime,
        "operator_family": _operator_family(regime),
        "effective_search_config": search,
        "effective_evaluator_config": evaluator,
        "target_semantics": target_semantics,
        "execution_semantics": execution_semantics,
    }


def canonical_policy_target_identity(
    manifest: Mapping[str, Any],
    *,
    authority: Mapping[str, Any] | None,
    strict_current: bool = True,
) -> dict[str, Any]:
    """Public fail-closed constructor for a stored policy teacher identity."""

    return _canonical_policy_target_identity(
        manifest,
        authority=authority,
        strict_current=strict_current,
    )


def canonical_value_sha256(value: object) -> str:
    """Return the repository's canonical semantic SHA-256 encoding."""

    return _value_sha256(value)


def _source_contract_payload(
    source_authority: Mapping[str, Any], *, scope: str
) -> dict[str, Any] | None:
    if scope != "fresh":
        raise InventoryError(f"unsupported policy-target authority scope: {scope!r}")
    raw: Any = source_authority.get("current_contract")
    if not isinstance(raw, Mapping):
        return None
    path_value = raw.get("path")
    if not isinstance(path_value, str):
        return None
    path = Path(path_value).expanduser().resolve(strict=True)
    expected = raw.get("file_sha256")
    actual = _file_sha256(path)
    if isinstance(expected, str) and expected != actual:
        raise InventoryError(f"operator authority hash drift: {path}")
    return _load_json(path)


def _is_current_exact_authority(authority: Mapping[str, Any] | None) -> bool:
    if not isinstance(authority, Mapping):
        return False
    provenance = authority.get("provenance")
    return isinstance(provenance, Mapping) and isinstance(
        provenance.get("runtime_code_tree_sha256"), str
    )


def _operator_authorities(source_authority: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()

    def add(role: str, raw: Any) -> None:
        if not isinstance(raw, Mapping):
            return
        path_value = raw.get("path") or raw.get("lock")
        if not isinstance(path_value, str):
            return
        path = Path(path_value).expanduser().resolve(strict=True)
        if path in seen_paths:
            return
        seen_paths.add(path)
        expected = raw.get("file_sha256") or raw.get("lock_file_sha256")
        actual = _file_sha256(path)
        if isinstance(expected, str) and expected != actual:
            raise InventoryError(f"operator authority hash drift: {path}")
        payload = _load_json(path)
        records.append(
            {
                "role": role,
                "path": str(path),
                "sha256": actual,
                "operator": _nested_operator(payload),
            }
        )

    add("current_contract", source_authority.get("current_contract"))
    verifiers = source_authority.get("lock_verifier_authorities")
    if isinstance(verifiers, Mapping):
        for role, raw in sorted(verifiers.items()):
            add(str(role), raw)
    return records


def _manifest_operator_groups(source_authority: Mapping[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    scope_authorities = {
        "fresh": _source_contract_payload(source_authority, scope="fresh")
    }

    def consume(scope: str, records: Any) -> None:
        if not isinstance(records, list):
            return
        authority = scope_authorities[scope]
        strict_current = scope == "fresh" and _is_current_exact_authority(authority)
        for record in records:
            if not isinstance(record, Mapping) or not isinstance(
                record.get("artifact"), Mapping
            ):
                continue
            artifact = record["artifact"]
            path = Path(str(artifact["path"])).expanduser().resolve(strict=True)
            actual = _file_sha256(path)
            if actual != artifact.get("file_sha256"):
                raise InventoryError(f"generation manifest hash drift: {path}")
            manifest = _load_json(path)
            identity = _canonical_policy_target_identity(
                manifest,
                authority=authority,
                strict_current=strict_current,
            )
            operator = (
                identity["operator"]
                if identity["schema_version"] == LEGACY_POLICY_TARGET_IDENTITY_SCHEMA
                else identity
            )
            # Old manifests attest the regime even when the detailed search
            # fields live only in the sealed contract authority.
            if identity["schema_version"] == LEGACY_POLICY_TARGET_IDENTITY_SCHEMA:
                regime = str(manifest.get("target_information_regime", "unknown"))
                operator.setdefault("target_information_regime", regime)
                operator.setdefault("operator_family", _operator_family(regime))
                operator.setdefault(
                    "preserve_search_evidence",
                    bool(manifest.get("preserve_search_evidence", False)),
                )
                operator.setdefault(
                    "search_evidence_schema",
                    manifest.get("search_evidence_schema"),
                )
                identity["operator"] = operator
            digest = (
                _value_sha256(operator)
                if identity["schema_version"]
                == LEGACY_POLICY_TARGET_IDENTITY_SCHEMA
                else _value_sha256(identity)
            )
            key = (scope, str(record.get("category", "unknown")), digest)
            group = grouped.setdefault(
                key,
                {
                    "scope": scope,
                    "category": str(record.get("category", "unknown")),
                    "identity_schema_version": identity["schema_version"],
                    "identity_completeness": identity["completeness"],
                    "policy_target_identity": identity,
                    "operator": operator,
                    "operator_sha256": digest,
                    "manifest_count": 0,
                    "games_completed": 0,
                    "rows": 0,
                },
            )
            group["manifest_count"] += 1
            group["games_completed"] += int(manifest.get("games_completed", 0))
            group["rows"] += int(manifest.get("rows", 0))

    consume("fresh", source_authority.get("fresh_generation_manifests"))
    return [grouped[key] for key in sorted(grouped)]


def _policy_operator_identity_inventory(
    *,
    groups: Iterable[Mapping[str, Any]],
    policy_distillation_component_ids: set[str],
    policy_active_component_ids: set[str],
) -> dict[str, Any]:
    """Make manifest operator fingerprints decision-bearing.

    Information regime is only one field of a policy target's identity.  In
    particular, n128 and n256 targets from the same coherent-public search
    family are different teachers, as are targets produced by different
    checkpoints.  The old inventory reported those manifest fingerprints but
    still admitted the composite based on regime alone.

    Fresh manifest categories map directly to fresh composite component IDs.
    Retired replay components are deliberately absent: their stale policy
    operators cannot enter a fresh-only learner through this inventory.
    """

    relevant: list[dict[str, Any]] = []
    covered_components: set[str] = set()
    for raw in groups:
        scope = str(raw.get("scope", ""))
        category = str(raw.get("category", ""))
        if scope == "fresh" and category in policy_distillation_component_ids:
            relevant.append(dict(raw))
            covered_components.add(category)

    expected_components = (
        policy_distillation_component_ids & policy_active_component_ids
    )
    missing = sorted(expected_components - covered_components)
    identities = sorted(
        {
            str(item["operator_sha256"])
            for item in relevant
            if isinstance(item.get("operator_sha256"), str)
        }
    )
    mixed = len(identities) > 1
    exact_identity_required = any(
        item.get("identity_completeness") == "current_exact_fail_closed"
        for item in relevant
    )
    return {
        "schema_version": "policy-target-manifest-identity-inventory-v2",
        "policy_active_component_ids": sorted(expected_components),
        "manifest_covered_component_ids": sorted(covered_components),
        "missing_manifest_identity_component_ids": missing,
        "realized_operator_sha256": identities,
        "realized_identity_schema_versions": sorted(
            {
                str(item["identity_schema_version"])
                for item in relevant
                if isinstance(item.get("identity_schema_version"), str)
            }
        ),
        "exact_identity_required": exact_identity_required,
        "mixed_policy_target_operators": mixed,
        # Missing historical provenance remains diagnostic. A current sealed
        # wave, however, cannot claim an exact teacher identity while any
        # policy-active component lacks a generation-manifest identity.
        "policy_operator_uniform": (
            not mixed and (not exact_identity_required or not missing)
        ),
    }


def inspect_composite(
    *, descriptor_path: Path, required_regime: str
) -> dict[str, Any]:
    path = descriptor_path.expanduser().resolve(strict=True)
    descriptor = _load_json(path)
    if descriptor.get("schema_version") != "memmap_composite_v2":
        raise InventoryError(f"{path}: expected memmap_composite_v2")
    distillation = set(descriptor.get("policy_distillation_component_ids", ()))
    components: list[dict[str, Any]] = []
    for raw in descriptor.get("components", ()):
        if not isinstance(raw, Mapping):
            raise InventoryError(f"{path}: malformed component")
        component_id = str(raw.get("component_id", ""))
        item = inspect_memmap(
            label=component_id,
            corpus_dir=Path(str(raw["corpus_dir"])),
            required_regime=required_regime,
        )
        if item["corpus_meta"]["sha256"] != raw.get("corpus_meta_sha256"):
            raise InventoryError(f"{component_id}: corpus_meta hash drift")
        if (
            item["corpus_meta"]["payload_inventory_sha256"]
            != raw.get("payload_inventory_sha256")
        ):
            raise InventoryError(f"{component_id}: payload inventory identity drift")
        item["policy_distillation_active"] = component_id in distillation
        components.append(item)

    authority_path = Path(str(descriptor["source_authority_manifest"]))
    authority_path = authority_path.expanduser().resolve(strict=True)
    if _file_sha256(authority_path) != descriptor.get("source_authority_manifest_sha256"):
        raise InventoryError("source authority manifest hash drift")
    source_authority = _load_json(authority_path)

    active_rows = sum(
        int(item["policy_active_rows"])
        for item in components
        if item["policy_distillation_active"]
    )
    incompatible = sum(
        int(item["incompatible_policy_active_rows"])
        for item in components
        if item["policy_distillation_active"]
    )
    activation_invalid = [
        item["label"]
        for item in components
        if item["policy_distillation_active"]
        and item["policy_active_rule_mismatch_rows"] not in (None, 0)
    ]
    reanalysis_blocked = [
        item["label"]
        for item in components
        if item["policy_distillation_active"]
        and not item["exact_root_reanalysis"]["full_corpus_replayable"]
    ]
    manifest_groups = _manifest_operator_groups(source_authority)
    policy_identity = _policy_operator_identity_inventory(
        groups=manifest_groups,
        policy_distillation_component_ids=distillation,
        policy_active_component_ids={
            str(item["label"])
            for item in components
            if int(item["policy_active_rows"]) > 0
        },
    )
    return {
        "descriptor": {
            "path": str(path),
            "sha256": _file_sha256(path),
            "schema_version": descriptor["schema_version"],
        },
        "required_target_information_regime": required_regime,
        "policy_distillation_component_ids": sorted(distillation),
        "components": components,
        "policy_active_rows": active_rows,
        "incompatible_policy_active_rows": incompatible,
        "policy_activation_invalid_components": activation_invalid,
        "policy_targets_eligible_for_requested_learner": (
            incompatible == 0
            and not activation_invalid
            and policy_identity["policy_operator_uniform"]
        ),
        "old_targets_remain_policy_active": incompatible > 0,
        "policy_target_manifest_identity": policy_identity,
        "full_composite_root_reanalysis_eligible": not reanalysis_blocked,
        "root_reanalysis_blocked_components": reanalysis_blocked,
        "source_authority": {
            "path": str(authority_path),
            "sha256": _file_sha256(authority_path),
            "schema_version": source_authority.get("schema_version"),
        },
        "operator_authorities": _operator_authorities(source_authority),
        "generation_manifest_operator_groups": manifest_groups,
    }


def inspect_rd_contract(contract_path: Path) -> dict[str, Any]:
    """Authenticate the deliberately small coherent-target R&D recipe.

    This is intentionally narrower than the full production-wave sealer.  Its
    only purpose is to replace an ineligible PIMC target corpus with a compact,
    self-play-only coherent corpus whose complete action traces can be audited
    and reanalyzed later.  Opponent mixing and adaptive budgets are prohibited
    because either would confound that target-identity intervention.
    """

    path = contract_path.expanduser().resolve(strict=True)
    value = _load_json(path)
    contract_schema = value.get("schema_version")
    if contract_schema not in RD_CONTRACT_SCHEMAS:
        raise InventoryError(
            f"{path}: expected one of {sorted(RD_CONTRACT_SCHEMAS)}"
        )
    declared = value.get("contract_sha256")
    actual = _self_digest(value, "contract_sha256")
    if declared != actual:
        raise InventoryError(
            f"{path}: contract semantic digest drift ({declared!r} != {actual!r})"
        )

    artifacts: dict[str, dict[str, Any]] = {}
    for role in ("typed_generation_config", "generation_guard"):
        record = value.get("artifacts", {}).get(role)
        if not isinstance(record, Mapping):
            raise InventoryError(f"{path}: missing artifact {role}")
        artifact_path = Path(str(record.get("path", "")))
        if not artifact_path.is_absolute():
            artifact_path = path.parents[3] / artifact_path
        artifact_path = artifact_path.expanduser().resolve(strict=True)
        digest = _file_sha256(artifact_path)
        if digest != record.get("sha256"):
            raise InventoryError(f"{path}: {role} hash drift: {artifact_path}")
        artifacts[role] = {
            "path": str(artifact_path),
            "sha256": digest,
            "payload": _load_json(artifact_path),
        }

    config = artifacts["typed_generation_config"]["payload"]
    if config.get("schema_version") != 13 or config.get("pipeline") != "generate":
        raise InventoryError(f"{path}: typed config is not schema-13 generation")
    fields = config.get("fields")
    if not isinstance(fields, Mapping):
        raise InventoryError(f"{path}: typed config has no fields")
    records_automatic_transitions = contract_schema == RD_CONTRACT_SCHEMA_V2
    required_fields = {
        "public_observation": True,
        "coherent_public_belief_search": True,
        "information_set_search": False,
        "determinization_particles": 1,
        "n_full": 128,
        "n_fast": 16,
        "p_full": 0.25,
        "n_full_wide": None,
        "n_full_wide_threshold": None,
        "wide_roots_always_full": False,
        "opponent_mix_manifest": None,
        "opponent_pool_manifest": None,
        "native_mcts_hot_loop": True,
        "rust_featurize": True,
        "record_automatic_transitions": records_automatic_transitions,
        "meaningful_public_history": True,
        "seed_claim": True,
        "checkpoint": None,
        "games": 0,
    }
    drift = {
        key: {"expected": expected, "actual": fields.get(key)}
        for key, expected in required_fields.items()
        if fields.get(key) != expected
    }
    if drift:
        raise InventoryError(f"{path}: coherent R&D config drift: {drift}")

    guard_payload = artifacts["generation_guard"]["payload"]
    guards = guard_payload.get("guards")
    if not isinstance(guards, list):
        raise InventoryError(f"{path}: generation guard has no guard list")
    lint_records = [item for item in guards if item.get("name") == "cli_flag_lint"]
    if len(lint_records) != 1:
        raise InventoryError(f"{path}: expected one cli_flag_lint guard")
    lint = lint_records[0].get("args", {})
    forbidden = set(lint.get("forbidden_flags", ()))
    required_forbidden = {
        "--n-full-wide",
        "--n-full-wide-threshold",
        "--opponent-mix-manifest",
        "--opponent-pool-manifest",
        "--raw-policy-above-width",
    }
    expected = lint.get("expected_values", {})
    if not required_forbidden <= forbidden:
        raise InventoryError(f"{path}: nullable target-identity overrides are not sealed")
    if expected.get("--preserve-search-evidence") is not True:
        raise InventoryError(f"{path}: search evidence is not required")
    if expected.get("--coherent-public-belief-search") is not True:
        raise InventoryError(f"{path}: coherent public search is not guard-required")
    if (
        expected.get("--record-automatic-transitions")
        is not records_automatic_transitions
    ):
        raise InventoryError(
            f"{path}: automatic-transition guard disagrees with contract version"
        )
    _required_search_evidence_schema(value, where=path)
    if contract_schema == RD_CONTRACT_SCHEMA_V2 and (
        value.get("acceptance", {}).get(
            "require_forced_value_rows_in_every_game"
        )
        is not True
        or value.get("acceptance", {}).get(
            "forced_row_policy_weight_multiplier"
        )
        != 0.0
        or value.get("acceptance", {}).get(
            "forced_row_value_weight_multiplier"
        )
        != 1.0
    ):
        raise InventoryError(f"{path}: forced-value acceptance contract drift")

    execution = value.get("execution")
    if not isinstance(execution, Mapping):
        raise InventoryError(f"{path}: missing execution plan")
    lanes = execution.get("lanes")
    if not isinstance(lanes, list) or not lanes:
        raise InventoryError(f"{path}: execution plan has no lanes")
    intervals: list[tuple[int, int]] = []
    total_games = 0
    placements: set[tuple[str, int]] = set()
    for lane in lanes:
        if not isinstance(lane, Mapping):
            raise InventoryError(f"{path}: malformed lane")
        start = int(lane["base_seed"])
        games = int(lane["games"])
        if games <= 0:
            raise InventoryError(f"{path}: lane has a non-positive game count")
        interval = (start, start + games)
        if any(start < prior_end and prior_start < interval[1] for prior_start, prior_end in intervals):
            raise InventoryError(f"{path}: lane seed intervals overlap")
        intervals.append(interval)
        placement = (str(lane["host"]), int(lane["gpu"]))
        if placement in placements:
            raise InventoryError(f"{path}: duplicate host/GPU placement {placement}")
        placements.add(placement)
        total_games += games
    if total_games != int(execution.get("total_games", -1)):
        raise InventoryError(f"{path}: lane game total drift")
    if min(start for start, _end in intervals) != int(execution.get("seed_start", -1)):
        raise InventoryError(f"{path}: seed_start drift")
    if max(end for _start, end in intervals) != int(execution.get("seed_end", -1)):
        raise InventoryError(f"{path}: seed_end drift")

    producer = value.get("producer_checkpoint")
    if not isinstance(producer, Mapping) or not Path(str(producer.get("path", ""))).is_absolute():
        raise InventoryError(f"{path}: producer checkpoint path must be absolute")
    producer_sha = str(producer.get("sha256", ""))
    if not producer_sha.startswith("sha256:") or len(producer_sha) != 71:
        raise InventoryError(f"{path}: invalid producer checkpoint sha256")

    return {
        "path": str(path),
        "sha256": _file_sha256(path),
        "schema_version": contract_schema,
        "contract_sha256": actual,
        "contract_id": value.get("contract_id"),
        "status": value.get("status"),
        "target_information_regime": value.get("target_information_regime"),
        "producer_checkpoint": dict(producer),
        "total_games": total_games,
        "lane_count": len(lanes),
        "seed_intervals": [[start, end] for start, end in intervals],
        "typed_generation_config": {
            key: artifacts["typed_generation_config"][key]
            for key in ("path", "sha256")
        },
        "generation_guard": {
            key: artifacts["generation_guard"][key]
            for key in ("path", "sha256")
        },
        "contract_eligible_to_launch": True,
    }


def build_inventory(
    *,
    corpora: Iterable[tuple[str, Path]],
    composite: Path | None,
    rd_contract: Path | None,
    required_regime: str,
) -> dict[str, Any]:
    direct = [
        inspect_memmap(label=label, corpus_dir=path, required_regime=required_regime)
        for label, path in corpora
    ]
    composite_result = (
        None
        if composite is None
        else inspect_composite(
            descriptor_path=composite, required_regime=required_regime
        )
    )
    rd_contract_result = (
        None if rd_contract is None else inspect_rd_contract(rd_contract)
    )
    active = sum(int(item["policy_active_rows"]) for item in direct)
    incompatible = sum(int(item["incompatible_policy_active_rows"]) for item in direct)
    if composite_result is not None:
        active += int(composite_result["policy_active_rows"])
        incompatible += int(composite_result["incompatible_policy_active_rows"])
    activation_invalid = [
        item["label"]
        for item in direct
        if item["policy_active_rule_mismatch_rows"] not in (None, 0)
    ]
    if composite_result is not None:
        activation_invalid.extend(
            f"composite:{label}"
            for label in composite_result["policy_activation_invalid_components"]
        )
    identity_invalid = (
        []
        if composite_result is None
        or composite_result["policy_targets_eligible_for_requested_learner"]
        else ["composite"]
    )
    targets_eligible = (
        incompatible == 0
        and not activation_invalid
        and not identity_invalid
    )
    value: dict[str, Any] = {
        "schema_version": SCHEMA,
        "required_target_information_regime": required_regime,
        "direct_corpora": direct,
        "composite": composite_result,
        "rd_contract": rd_contract_result,
        "aggregate": {
            "policy_active_rows": active,
            "incompatible_policy_active_rows": incompatible,
            "policy_activation_invalid_components": activation_invalid,
            "policy_target_identity_invalid_scopes": identity_invalid,
            "policy_targets_eligible_for_requested_learner": targets_eligible,
            "old_targets_remain_policy_active": incompatible > 0,
            "decision": (
                "eligible_existing_targets"
                if targets_eligible
                else "generate_new_coherent_targets"
            ),
        },
    }
    value["inventory_sha256"] = _value_sha256(value)
    return value


def _parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=/absolute/corpus/path")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("expected LABEL=/absolute/corpus/path")
    return label, Path(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        action="append",
        default=[],
        type=_parse_named_path,
        metavar="LABEL=PATH",
        help="direct memmap corpus; repeat for the n128/n256 196k pair",
    )
    parser.add_argument("--composite", type=Path)
    parser.add_argument(
        "--rd-contract",
        type=Path,
        help="authenticate the sealed coherent-target R&D generation contract",
    )
    parser.add_argument(
        "--required-regime", default=COHERENT_REGIME, choices=(PIMC_REGIME, COHERENT_REGIME)
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.corpus and args.composite is None and args.rd_contract is None:
        parser.error("at least one --corpus, --composite, or --rd-contract is required")
    try:
        value = build_inventory(
            corpora=args.corpus,
            composite=args.composite,
            rd_contract=args.rd_contract,
            required_regime=args.required_regime,
        )
        _write_json_atomic(args.out, value)
    except (InventoryError, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
