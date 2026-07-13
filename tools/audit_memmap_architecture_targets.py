#!/usr/bin/env python3
"""Chunked viability audit for entity action targets and graph incidence IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


ACTION_TYPES = (
    "BUILD_SETTLEMENT",
    "BUILD_ROAD",
    "BUILD_CITY",
    "BUY_DEVELOPMENT_CARD",
    "MARITIME_TRADE",
    "offer_trade",
    "accept_trade",
    "reject_trade",
    "cancel_trade",
    "confirm_trade",
    "MOVE_ROBBER",
    "DISCARD_RESOURCE",
    "PLAY_KNIGHT_CARD",
    "PLAY_YEAR_OF_PLENTY",
    "PLAY_MONOPOLY",
    "PLAY_ROAD_BUILDING",
    "ROLL",
    "END_TURN",
)
TARGET_LIMITS = (19, 54, 72, 4)
TARGET_NAMES = ("hex", "vertex", "edge", "player")
INCIDENCE = {
    "hex_vertex_ids": (54, (19, 6)),
    "hex_edge_ids": (72, (19, 6)),
    "edge_vertex_ids": (54, (72, 2)),
}
REQUIRED_ACTION_COLUMNS = {
    "legal_action_ids",
    "legal_action_tokens",
    "legal_action_target_ids",
}
LEGACY_SEARCH_AUTH_COLUMNS = {
    "policy_weight_multiplier",
    "target_policy",
    "target_policy_mask",
    "teacher_name",
}


class CorpusReader:
    """Metadata-driven memmap reader that never materializes a full column."""

    def __init__(self, root: Path):
        self.root = root.expanduser().resolve(strict=True)
        meta_path = self.root / "corpus_meta.json"
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.rows = int(self.meta["row_count"])
        self.schemas = self.meta["columns"]
        self.offsets = np.memmap(
            self.root / "row_offsets.dat",
            dtype=np.int64,
            mode="r",
            shape=(self.rows + 1,),
        )
        self._arrays: dict[str, np.memmap] = {}

    def has(self, name: str) -> bool:
        return name in self.schemas

    def _array(self, name: str) -> np.memmap:
        if name in self._arrays:
            return self._arrays[name]
        schema = self.schemas[name]
        kind = schema["kind"]
        dtype = np.dtype(np.int32 if kind == "string" else schema["dtype"])
        if kind == "fixed":
            shape = (self.rows, *(int(value) for value in schema["inner_shape"]))
        elif kind == "string":
            shape = (self.rows,)
        elif kind == "ragged2d":
            shape = (int(self.meta["flat_count"]),)
        elif kind == "ragged3d":
            shape = (int(self.meta["flat_count"]), int(schema["feat"]))
        else:
            raise ValueError(f"column {name!r} has no payload array ({kind})")
        suffix = ".codes.dat" if kind == "string" else ".dat"
        array = np.memmap(self.root / f"{name}{suffix}", dtype=dtype, mode="r", shape=shape)
        self._arrays[name] = array
        return array

    def rows_slice(self, name: str, start: int, stop: int) -> np.ndarray:
        schema = self.schemas[name]
        kind = schema["kind"]
        if kind == "implicit_constant":
            inner = tuple(int(value) for value in schema["inner_shape"])
            return np.full(
                (stop - start, *inner), schema["fill"], dtype=np.dtype(schema["dtype"])
            )
        values = np.asarray(self._array(name)[start:stop])
        if kind == "string":
            categories = np.asarray(schema.get("categories") or [""], dtype=str)
            return categories[values]
        if kind != "fixed":
            raise ValueError(f"column {name!r} is ragged, not row-fixed")
        return values

    def ragged_flat(
        self,
        name: str,
        start: int,
        stop: int,
        *,
        feature_slice: slice | None = None,
    ) -> np.ndarray:
        schema = self.schemas[name]
        if schema["kind"] not in {"ragged2d", "ragged3d"}:
            raise ValueError(f"column {name!r} is not ragged")
        flat_start = int(self.offsets[start])
        flat_stop = int(self.offsets[stop])
        selection = self._array(name)[flat_start:flat_stop]
        if feature_slice is not None:
            selection = selection[:, feature_slice]
        return np.asarray(selection)


def _counter() -> dict[str, int]:
    return {
        "actions": 0,
        "actions_with_any_target": 0,
        "hex_targets": 0,
        "vertex_targets": 0,
        "edge_targets": 0,
        "player_targets": 0,
        "out_of_range_targets": 0,
    }


def _add_group(
    destination: dict[str, dict[str, int]],
    labels: np.ndarray,
    targets: np.ndarray,
) -> None:
    any_target = np.any(targets >= 0, axis=1)
    invalid = np.zeros(len(targets), dtype=bool)
    for column, limit in enumerate(TARGET_LIMITS):
        invalid |= targets[:, column] >= limit
    for label in np.unique(labels.astype(str)):
        selected = labels.astype(str) == label
        row = destination.setdefault(str(label), _counter())
        row["actions"] += int(np.sum(selected))
        row["actions_with_any_target"] += int(np.sum(selected & any_target))
        row["out_of_range_targets"] += int(np.sum(selected & invalid))
        for column, name in enumerate(TARGET_NAMES):
            row[f"{name}_targets"] += int(np.sum(selected & (targets[:, column] >= 0)))


def _finalize_groups(groups: dict[str, dict[str, int]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for key in sorted(groups):
        row: dict[str, Any] = dict(groups[key])
        row["target_coverage"] = row["actions_with_any_target"] / max(row["actions"], 1)
        result[key] = row
    return result


def _action_labels(action_type_features: np.ndarray) -> np.ndarray:
    one_hot = np.asarray(action_type_features) > 0.5
    counts = np.sum(one_hot, axis=1)
    indices = np.argmax(one_hot, axis=1)
    labels = np.asarray([ACTION_TYPES[index] for index in indices], dtype=object)
    labels[counts != 1] = "unknown"
    return labels.astype(str)


def _row_values(reader: CorpusReader, name: str, start: int, stop: int, default: Any):
    if not reader.has(name):
        return np.full(stop - start, default)
    return reader.rows_slice(name, start, stop)


def audit_corpus(root: Path, *, chunk_rows: int = 4096) -> dict[str, Any]:
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    reader = CorpusReader(root)
    columns = set(reader.schemas)
    missing_action = sorted(REQUIRED_ACTION_COLUMNS - columns)
    action_groups: dict[str, dict[str, int]] = {}
    phase_groups: dict[str, dict[str, int]] = {}
    cohort_groups: dict[str, dict[str, int]] = {}
    chosen_action_groups: dict[str, dict[str, int]] = {}
    chosen_phase_groups: dict[str, dict[str, int]] = {}
    chosen_actions = 0
    chosen_actions_with_target = 0
    chosen_policy_active = 0
    chosen_policy_active_with_target = 0
    chosen_search_active = 0
    chosen_search_active_with_target = 0
    chosen_action_missing_from_legal = 0
    chosen_action_duplicate_in_legal = 0
    total_rows_with_target = 0
    policy_active_rows_with_target = 0
    search_active_rows_with_target = 0
    policy_active_rows = 0
    search_active_rows = 0
    invalid_legal_ids = 0
    explicit_full_search = reader.has("used_full_search")
    legacy_search_contract = (
        not explicit_full_search
        and LEGACY_SEARCH_AUTH_COLUMNS <= columns
    )
    legacy_search_wrong_teacher_rows = 0
    legacy_search_missing_policy_rows = 0
    event = {
        "masked_events": 0,
        "events_with_any_target": 0,
        "out_of_range_targets": 0,
        **{f"{name}_targets": 0 for name in TARGET_NAMES},
    }
    incidence = {
        name: {
            "expected_inner_shape": list(expected_shape),
            "declared_inner_shape": None,
            "shape_valid": False,
            "ids": 0,
            "valid_ids": 0,
            "missing_ids": 0,
            "out_of_range_ids": 0,
            "min_valid_id": None,
            "max_valid_id": None,
        }
        for name, (_limit, expected_shape) in INCIDENCE.items()
    }

    for start in range(0, reader.rows, chunk_rows):
        stop = min(start + chunk_rows, reader.rows)
        row_count = stop - start
        counts = np.diff(np.asarray(reader.offsets[start : stop + 1], dtype=np.int64))
        row_index = np.repeat(np.arange(row_count, dtype=np.int64), counts)
        phases = _row_values(reader, "phase", start, stop, "unknown").astype(str)
        policy_weight = _row_values(
            reader, "policy_weight_multiplier", start, stop, 1.0
        ).astype(np.float64)
        policy_active = policy_weight > 0.0
        if explicit_full_search:
            full_search = reader.rows_slice(
                "used_full_search", start, stop
            ).astype(bool)
        elif legacy_search_contract:
            # Historical A1 replay predates the redundant ``used_full_search``
            # column.  The producer contract is nevertheless exact:
            # gumbel_self_play writes policy_weight_multiplier > 0 only for a
            # non-forced full-search row.  Authenticate the teacher and stored
            # distribution below before allowing that equivalence to stand in
            # for the missing metadata bit.
            teacher = reader.rows_slice("teacher_name", start, stop).astype(str)
            wrong_teacher = policy_active & (teacher != "gumbel_self_play")
            legacy_search_wrong_teacher_rows += int(np.sum(wrong_teacher))
            full_search = policy_active & ~wrong_teacher
        else:
            full_search = np.zeros(row_count, dtype=bool)
        search_active = policy_active & full_search
        policy_active_rows += int(np.sum(policy_active))
        search_active_rows += int(np.sum(search_active))

        if not missing_action:
            legal_ids = reader.ragged_flat("legal_action_ids", start, stop)
            targets = reader.ragged_flat("legal_action_target_ids", start, stop).astype(
                np.int64, copy=False
            )
            action_type_features = reader.ragged_flat(
                "legal_action_tokens",
                start,
                stop,
                feature_slice=slice(2, 2 + len(ACTION_TYPES)),
            )
            if len(legal_ids) != len(targets) or len(action_type_features) != len(targets):
                raise SystemExit(f"{reader.root}: ragged legal columns are misaligned")
            invalid_legal_ids += int(np.sum(legal_ids < 0))
            labels = _action_labels(action_type_features)
            _add_group(action_groups, labels, targets)
            _add_group(phase_groups, phases[row_index], targets)
            cohorts = np.full(len(targets), "other", dtype=object)
            cohorts[policy_active[row_index]] = "policy_active"
            cohorts[search_active[row_index]] = "search_active"
            _add_group(cohort_groups, cohorts.astype(str), targets)
            row_has_target = np.zeros(row_count, dtype=bool)
            np.logical_or.at(row_has_target, row_index, np.any(targets >= 0, axis=1))
            total_rows_with_target += int(np.sum(row_has_target))
            policy_active_rows_with_target += int(np.sum(row_has_target & policy_active))
            search_active_rows_with_target += int(np.sum(row_has_target & search_active))

            if legacy_search_contract:
                stored_policy = reader.ragged_flat(
                    "target_policy", start, stop
                ).astype(np.float64, copy=False)
                stored_policy_mask = reader.ragged_flat(
                    "target_policy_mask", start, stop
                ).astype(bool, copy=False)
                if len(stored_policy) != len(row_index) or len(
                    stored_policy_mask
                ) != len(row_index):
                    raise SystemExit(
                        f"{reader.root}: legacy stored-policy columns are misaligned"
                    )
                valid_mass = np.where(
                    stored_policy_mask & np.isfinite(stored_policy),
                    np.maximum(stored_policy, 0.0),
                    0.0,
                )
                row_policy_mass = np.zeros(row_count, dtype=np.float64)
                np.add.at(row_policy_mass, row_index, valid_mass)
                legacy_search_missing_policy_rows += int(
                    np.sum(policy_active & (row_policy_mass <= 0.0))
                )

            # The gather branch receives gradients through every legal logit,
            # but coverage of the demonstrated/MCTS-selected action is the
            # clearest measure of whether its positive target has an explicit
            # topology pointer. Bind that row by exact global action id rather
            # than assuming a legal-list order.
            if reader.has("action_taken"):
                taken = reader.rows_slice("action_taken", start, stop).astype(
                    np.int64, copy=False
                )
                selected = legal_ids.astype(np.int64, copy=False) == taken[row_index]
                match_count = np.zeros(row_count, dtype=np.int64)
                np.add.at(match_count, row_index, selected.astype(np.int64))
                chosen_action_missing_from_legal += int(np.sum(match_count == 0))
                chosen_action_duplicate_in_legal += int(np.sum(match_count > 1))
                exact = selected & (match_count[row_index] == 1)
                chosen_targets = targets[exact]
                chosen_labels = labels[exact]
                chosen_phases = phases[row_index][exact]
                chosen_actions += int(len(chosen_targets))
                chosen_has_target = np.any(chosen_targets >= 0, axis=1)
                chosen_actions_with_target += int(np.sum(chosen_has_target))
                chosen_policy = policy_active[match_count == 1]
                chosen_search = search_active[match_count == 1]
                chosen_policy_active += int(np.sum(chosen_policy))
                chosen_policy_active_with_target += int(
                    np.sum(chosen_policy & chosen_has_target)
                )
                chosen_search_active += int(np.sum(chosen_search))
                chosen_search_active_with_target += int(
                    np.sum(chosen_search & chosen_has_target)
                )
                _add_group(chosen_action_groups, chosen_labels, chosen_targets)
                _add_group(chosen_phase_groups, chosen_phases, chosen_targets)

        if reader.has("event_target_ids"):
            event_targets = reader.rows_slice("event_target_ids", start, stop).astype(
                np.int64, copy=False
            )
            event_mask = _row_values(
                reader,
                "event_mask",
                start,
                stop,
                False,
            ).astype(bool)
            if event_mask.ndim == 1:
                event_mask = np.zeros(event_targets.shape[:2], dtype=bool)
            masked = event_targets[event_mask]
            event["masked_events"] += int(len(masked))
            if len(masked):
                event["events_with_any_target"] += int(np.sum(np.any(masked >= 0, axis=1)))
                for column, (name, limit) in enumerate(zip(TARGET_NAMES, TARGET_LIMITS)):
                    event[f"{name}_targets"] += int(np.sum(masked[:, column] >= 0))
                    event["out_of_range_targets"] += int(np.sum(masked[:, column] >= limit))

        for name, (limit, expected_shape) in INCIDENCE.items():
            if not reader.has(name):
                continue
            values = reader.rows_slice(name, start, stop).astype(np.int64, copy=False)
            row = incidence[name]
            declared_shape = tuple(int(value) for value in values.shape[1:])
            row["declared_inner_shape"] = list(declared_shape)
            row["shape_valid"] = declared_shape == expected_shape
            row["ids"] += int(values.size)
            valid = values >= 0
            row["valid_ids"] += int(np.sum(valid))
            row["missing_ids"] += int(np.sum(~valid))
            row["out_of_range_ids"] += int(np.sum(values >= limit))
            if np.any(valid):
                minimum = int(np.min(values[valid]))
                maximum = int(np.max(values[valid]))
                row["min_valid_id"] = (
                    minimum if row["min_valid_id"] is None else min(row["min_valid_id"], minimum)
                )
                row["max_valid_id"] = (
                    maximum if row["max_valid_id"] is None else max(row["max_valid_id"], maximum)
                )

    by_action = _finalize_groups(action_groups)
    by_phase = _finalize_groups(phase_groups)
    by_cohort = _finalize_groups(cohort_groups)
    action_count = sum(row["actions"] for row in by_action.values())
    targeted_count = sum(row["actions_with_any_target"] for row in by_action.values())
    invalid_targets = sum(row["out_of_range_targets"] for row in by_action.values())
    incidence_missing = sorted(name for name in INCIDENCE if not reader.has(name))
    incidence_invalid = sum(row["out_of_range_ids"] for row in incidence.values())
    incidence_shapes_valid = all(row["shape_valid"] for row in incidence.values())
    gather_runnable = (
        not missing_action
        and action_count > 0
        and targeted_count > 0
        and invalid_targets == 0
        and search_active_rows_with_target > 0
        and legacy_search_wrong_teacher_rows == 0
        and legacy_search_missing_policy_rows == 0
    )
    cross_runnable = (
        "legal_action_tokens" in columns and action_count > 0 and policy_active_rows > 0
    )
    graph_runnable = (
        not incidence_missing and incidence_invalid == 0 and incidence_shapes_valid
    )
    result = {
        "schema_version": "memmap-architecture-target-audit-v1",
        "corpus_dir": str(reader.root),
        "rows": reader.rows,
        "chunk_rows": chunk_rows,
        "whole_column_materialization": False,
        "columns": sorted(columns),
        "legal_action_targets": {
            "actions": action_count,
            "actions_with_any_target": targeted_count,
            "target_coverage": targeted_count / max(action_count, 1),
            "rows_with_any_target": total_rows_with_target,
            "row_target_coverage": total_rows_with_target / max(reader.rows, 1),
            "policy_active_rows": policy_active_rows,
            "policy_active_rows_with_any_target": policy_active_rows_with_target,
            "policy_active_row_target_coverage": policy_active_rows_with_target
            / max(policy_active_rows, 1),
            "search_active_rows": search_active_rows,
            "search_active_rows_with_any_target": search_active_rows_with_target,
            "search_active_row_target_coverage": search_active_rows_with_target
            / max(search_active_rows, 1),
            "invalid_legal_action_ids": invalid_legal_ids,
            "out_of_range_target_rows": invalid_targets,
            "missing_columns": missing_action,
            "by_action_kind": by_action,
            "by_phase": by_phase,
            "by_search_cohort": by_cohort,
            "chosen_actions": chosen_actions,
            "chosen_actions_with_any_target": chosen_actions_with_target,
            "chosen_action_target_coverage": chosen_actions_with_target
            / max(chosen_actions, 1),
            "chosen_policy_active": chosen_policy_active,
            "chosen_policy_active_with_any_target": chosen_policy_active_with_target,
            "chosen_policy_active_target_coverage": chosen_policy_active_with_target
            / max(chosen_policy_active, 1),
            "chosen_search_active": chosen_search_active,
            "chosen_search_active_with_any_target": chosen_search_active_with_target,
            "chosen_search_active_target_coverage": chosen_search_active_with_target
            / max(chosen_search_active, 1),
            "chosen_action_missing_from_legal": chosen_action_missing_from_legal,
            "chosen_action_duplicate_in_legal": chosen_action_duplicate_in_legal,
            "search_activity_contract": {
                "source": (
                    "used_full_search"
                    if explicit_full_search
                    else "policy_weight_multiplier_legacy_equivalence"
                    if legacy_search_contract
                    else "unavailable"
                ),
                "legacy_required_columns": sorted(LEGACY_SEARCH_AUTH_COLUMNS),
                "legacy_wrong_teacher_rows": legacy_search_wrong_teacher_rows,
                "legacy_missing_stored_policy_rows": (
                    legacy_search_missing_policy_rows
                ),
                "authenticated": bool(
                    explicit_full_search
                    or (
                        legacy_search_contract
                        and legacy_search_wrong_teacher_rows == 0
                        and legacy_search_missing_policy_rows == 0
                    )
                ),
            },
            "chosen_by_action_kind": _finalize_groups(chosen_action_groups),
            "chosen_by_phase": _finalize_groups(chosen_phase_groups),
        },
        "event_targets": event,
        "graph_incidence": {
            "columns": incidence,
            "missing_columns": incidence_missing,
            "out_of_range_ids": incidence_invalid,
        },
        "viability": {
            "action_target_gather": gather_runnable,
            "action_cross_attention": cross_runnable,
            "graph_relational_trunk": graph_runnable,
            "event_target_relations": event["events_with_any_target"] > 0
            and event["out_of_range_targets"] == 0,
            "requires_generator_changes_for_action_probe": not (
                gather_runnable and cross_runnable and graph_runnable
            ),
            "event_target_generator_change_required_for_event_relations": event[
                "events_with_any_target"
            ]
            == 0,
        },
    }
    return result


def combined_verdict(audits: list[dict[str, Any]]) -> dict[str, Any]:
    action_ready = all(
        audit["viability"]["action_target_gather"]
        and audit["viability"]["action_cross_attention"]
        and audit["viability"]["graph_relational_trunk"]
        for audit in audits
    )
    return {
        "architecture_action_probe_runnable": action_ready,
        "requires_generator_changes_for_action_probe": not action_ready,
        "event_relation_probe_runnable": all(
            audit["viability"]["event_target_relations"] for audit in audits
        ),
        "corpus_count": len(audits),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", nargs="+", type=Path)
    parser.add_argument("--chunk-rows", type=int, default=512)
    parser.add_argument("--out", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    audits = [audit_corpus(path, chunk_rows=args.chunk_rows) for path in args.corpus]
    payload = {
        "schema_version": "memmap-architecture-target-audit-bundle-v1",
        "audits": audits,
        "verdict": combined_verdict(audits),
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
