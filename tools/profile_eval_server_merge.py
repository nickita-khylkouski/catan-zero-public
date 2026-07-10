#!/usr/bin/env python3
"""Field-level CPU profiler for EvalServer window assembly.

The default shape reproduces the retained H100 frontier averages: about 36
requests and 44 neural rows per window, an 18-column common legal rectangle,
and 39.2% useful legal cells.  It deliberately uses the real entity schema and
dtypes.  This is a standalone diagnostic: production pays no timer overhead.

Examples:

    python tools/profile_eval_server_merge.py
    python tools/profile_eval_server_merge.py --iterations 5000 --json
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from time import perf_counter_ns
from typing import Any

import numpy as np

from catan_zero.search.eval_server import (
    _LEGAL_PADDED_ENTITY_KEYS,
    _crop_masked_event_tail,
    _crop_payload_event_tails_before_merge,
    _merge_forward_payloads,
)


_TOKEN_SPECS = {
    "hex_tokens": (19, 13),
    "vertex_tokens": (54, 24),
    "edge_tokens": (72, 8),
    "player_tokens": (4, 31),
    "global_tokens": (1, 43),
}


def _distribute_legal_counts(rows: int, max_legal: int, occupancy: float) -> list[int]:
    target = max(rows, min(rows * max_legal, round(rows * max_legal * occupancy)))
    counts = np.full(rows, max(1, target // rows), dtype=np.int64)
    counts[0] = max_legal
    delta = target - int(counts.sum())
    cursor = 1
    while delta != 0:
        index = cursor % rows
        step = 1 if delta > 0 else -1
        candidate = int(counts[index]) + step
        if 1 <= candidate <= max_legal:
            counts[index] = candidate
            delta -= step
        cursor += 1
    return [int(value) for value in counts]


def _make_payloads(
    *,
    requests: int,
    rows: int,
    max_legal: int,
    occupancy: float,
    event_width: int,
    include_action_targets: bool,
) -> list[dict[str, Any]]:
    if not 1 <= requests <= rows:
        raise ValueError("requests must be in [1, rows]")
    counts = _distribute_legal_counts(rows, max_legal, occupancy)
    # Match the observed 1.23 rows/request with one chance-fanout request and
    # otherwise single-state requests. Rotate the widest row into the fan-out.
    row_groups = [rows - requests + 1, *([1] * (requests - 1))]
    payloads: list[dict[str, Any]] = []
    row_offset = 0
    for request_index, n_rows in enumerate(row_groups):
        row_counts = counts[row_offset : row_offset + n_rows]
        request_width = max(row_counts)
        legal_ids = np.full((n_rows, request_width), -1, dtype=np.int64)
        context = np.zeros((n_rows, request_width, 18), dtype=np.float32)
        legal_tokens = np.zeros((n_rows, request_width, 50), dtype=np.float16)
        legal_mask = np.zeros((n_rows, request_width), dtype=np.bool_)
        for row, legal_count in enumerate(row_counts):
            legal_ids[row, :legal_count] = np.arange(legal_count, dtype=np.int64)
            context[row, :legal_count] = (request_index + 1) / 128.0
            legal_tokens[row, :legal_count] = (row_offset + row + 1) / 128.0
            legal_mask[row, :legal_count] = True
        entity: dict[str, np.ndarray] = {
            key: np.zeros((n_rows, *shape), dtype=np.float16)
            for key, shape in _TOKEN_SPECS.items()
        }
        entity.update(
            {
                key.replace("tokens", "mask"): np.ones(
                    (n_rows, shape[0]), dtype=np.bool_
                )
                for key, shape in _TOKEN_SPECS.items()
                if key != "global_tokens"
            }
        )
        entity.update(
            {
                "legal_action_tokens": legal_tokens,
                "legal_action_mask": legal_mask,
                "event_tokens": np.zeros(
                    (n_rows, event_width, 41), dtype=np.float16
                ),
                "event_mask": np.zeros((n_rows, event_width), dtype=np.bool_),
            }
        )
        if include_action_targets:
            target_ids = np.full((n_rows, request_width, 4), -1, dtype=np.int16)
            entity["legal_action_target_ids"] = target_ids
        payloads.append(
            {"entity": entity, "legal_ids": legal_ids, "context": context}
        )
        row_offset += n_rows
    return payloads


def _tick(timings: dict[str, int], label: str, started: int) -> None:
    timings[label] += perf_counter_ns() - started


def _profiled_merge(
    payloads: list[dict[str, Any]],
    *,
    event_token_limit: int | None,
    premerge_event_crop: bool,
    merge_strategy: str,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[int], dict[str, int]]:
    timings: dict[str, int] = defaultdict(int)
    local_payloads = [
        {**payload, "entity": dict(payload["entity"])} for payload in payloads
    ]
    required_width: int | None = None
    if premerge_event_crop:
        started = perf_counter_ns()
        required_width = _crop_payload_event_tails_before_merge(
            local_payloads, event_token_limit
        )
        _tick(timings, "event.premerge_validate_and_view_crop", started)

    started = perf_counter_ns()
    row_counts = [int(payload["legal_ids"].shape[0]) for payload in local_payloads]
    total_rows = sum(row_counts)
    _tick(timings, "metadata.row_counts_and_sum", started)
    started = perf_counter_ns()
    max_legal = max(int(payload["legal_ids"].shape[1]) for payload in local_payloads)
    context_width = int(local_payloads[0]["context"].shape[2])
    _tick(timings, "metadata.max_legal_and_context", started)

    started = perf_counter_ns()
    legal_ids = np.full((total_rows, max_legal), -1, dtype=np.int64)
    _tick(timings, "allocate.dynamic.legal_ids", started)
    started = perf_counter_ns()
    context = np.zeros((total_rows, max_legal, context_width), dtype=np.float32)
    _tick(timings, "allocate.dynamic.context", started)
    entity: dict[str, np.ndarray] = {}
    for key, value in local_payloads[0]["entity"].items():
        started = perf_counter_ns()
        value = np.asarray(value)
        if key == "legal_action_tokens":
            destination = np.zeros(
                (total_rows, max_legal, int(value.shape[2])), dtype=value.dtype
            )
            kind = "dynamic"
        elif key == "legal_action_target_ids":
            destination = np.full(
                (total_rows, max_legal, int(value.shape[2])), -1, dtype=value.dtype
            )
            kind = "dynamic"
        elif key == "legal_action_mask":
            destination = np.zeros((total_rows, max_legal), dtype=np.bool_)
            kind = "dynamic"
        else:
            if merge_strategy == "hybrid":
                entity[key] = np.concatenate(
                    [payload["entity"][key] for payload in local_payloads], axis=0
                )
                _tick(timings, f"concatenate.fixed.{key}", started)
                continue
            dtype = np.result_type(
                *(
                    np.asarray(payload["entity"][key]).dtype
                    for payload in local_payloads
                )
            )
            destination = np.empty((total_rows, *value.shape[1:]), dtype=dtype)
            kind = "fixed"
        entity[key] = destination
        _tick(timings, f"allocate.{kind}.{key}", started)

    offset = 0
    for payload, n_rows in zip(local_payloads, row_counts):
        end = offset + n_rows
        legal_width = int(payload["legal_ids"].shape[1])
        started = perf_counter_ns()
        legal_ids[offset:end, :legal_width] = payload["legal_ids"]
        _tick(timings, "copy.dynamic.legal_ids", started)
        started = perf_counter_ns()
        context[offset:end, :legal_width] = payload["context"]
        _tick(timings, "copy.dynamic.context", started)
        for key, destination in entity.items():
            if merge_strategy == "hybrid" and key not in _LEGAL_PADDED_ENTITY_KEYS:
                continue
            started = perf_counter_ns()
            value = payload["entity"][key]
            if key in _LEGAL_PADDED_ENTITY_KEYS:
                destination[offset:end, :legal_width] = value
                kind = "dynamic"
            else:
                destination[offset:end] = value
                kind = "fixed"
            _tick(timings, f"copy.{kind}.{key}", started)
        offset = end

    if required_width is None:
        started = perf_counter_ns()
        _crop_masked_event_tail(entity, event_token_limit)
        _tick(timings, "event.postmerge_audit_and_crop", started)
    return entity, legal_ids, context, row_counts, timings


def _profile_scatter(
    payloads: list[dict[str, Any]], row_counts: list[int], timings: dict[str, int]
) -> int:
    total_rows = sum(row_counts)
    max_legal = max(int(payload["legal_ids"].shape[1]) for payload in payloads)
    logits = np.zeros((total_rows, max_legal), dtype=np.float32)
    values = np.zeros((total_rows,), dtype=np.float32)
    q_values = np.zeros((total_rows, max_legal), dtype=np.float32)
    output_bytes = 0
    offset = 0
    for payload, n_rows in zip(payloads, row_counts):
        legal_width = int(payload["legal_ids"].shape[1])
        row_slice = slice(offset, offset + n_rows)
        for label, source in (
            ("logits", logits[row_slice, :legal_width]),
            ("value", values[row_slice]),
            ("q_values", q_values[row_slice, :legal_width]),
        ):
            started = perf_counter_ns()
            copied = source.copy()
            _tick(timings, f"scatter.copy.{label}", started)
            output_bytes += int(copied.nbytes)
        offset += n_rows
    return output_bytes


def _array_bytes(payloads: list[dict[str, Any]]) -> int:
    return sum(
        int(np.asarray(value).nbytes)
        for payload in payloads
        for value in (
            *payload["entity"].values(),
            payload["legal_ids"],
            payload["context"],
        )
    )


def _field_bytes(
    entity: dict[str, np.ndarray], legal_ids: np.ndarray, context: np.ndarray
) -> dict[str, int]:
    return {
        "legal_ids": int(legal_ids.nbytes),
        "context": int(context.nbytes),
        **{key: int(value.nbytes) for key, value in entity.items()},
    }


def _summarize(args: argparse.Namespace) -> dict[str, Any]:
    payloads = _make_payloads(
        requests=args.requests,
        rows=args.rows,
        max_legal=args.max_legal,
        occupancy=args.occupancy,
        event_width=args.event_width,
        include_action_targets=args.include_action_targets,
    )
    true_legal = sum(
        int(np.count_nonzero(payload["entity"]["legal_action_mask"]))
        for payload in payloads
    )
    allocated_legal = args.rows * args.max_legal
    source_bytes = _array_bytes(payloads)
    merge_payloads = payloads
    if args.server_receives_client_crop and args.event_token_limit is not None:
        merge_payloads = [
            {**payload, "entity": dict(payload["entity"])} for payload in payloads
        ]
        _crop_payload_event_tails_before_merge(
            merge_payloads, args.event_token_limit
        )
    premerge_event_crop = (
        args.event_token_limit is not None
        and not args.server_receives_client_crop
    )

    # Prove the profiled implementation and optimization preserve authoritative
    # merged tensors before collecting timings.
    baseline_payloads = [
        {**payload, "entity": dict(payload["entity"])} for payload in payloads
    ]
    expected_entity, expected_ids, expected_context, expected_rows = (
        _merge_forward_payloads(baseline_payloads)
    )
    _crop_masked_event_tail(expected_entity, args.event_token_limit)
    actual_entity, actual_ids, actual_context, actual_rows, _ = _profiled_merge(
        merge_payloads,
        event_token_limit=args.event_token_limit,
        premerge_event_crop=premerge_event_crop,
        merge_strategy=args.merge_strategy,
    )
    np.testing.assert_array_equal(actual_ids, expected_ids)
    np.testing.assert_array_equal(actual_context, expected_context)
    if actual_rows != expected_rows or actual_entity.keys() != expected_entity.keys():
        raise AssertionError("profiled merge structure differs from authoritative merge")
    for key in actual_entity:
        np.testing.assert_array_equal(actual_entity[key], expected_entity[key])

    for _ in range(args.warmup):
        _profiled_merge(
            merge_payloads,
            event_token_limit=args.event_token_limit,
            premerge_event_crop=premerge_event_crop,
            merge_strategy=args.merge_strategy,
        )
    totals: dict[str, int] = defaultdict(int)
    scatter_bytes = 0
    last_field_bytes: dict[str, int] = {}
    for _ in range(args.iterations):
        entity, legal_ids, context, row_counts, timings = _profiled_merge(
            merge_payloads,
            event_token_limit=args.event_token_limit,
            premerge_event_crop=premerge_event_crop,
            merge_strategy=args.merge_strategy,
        )
        scatter_bytes = _profile_scatter(merge_payloads, row_counts, timings)
        for key, value in timings.items():
            totals[key] += value
        last_field_bytes = _field_bytes(entity, legal_ids, context)

    per_window_us = {
        key: value / args.iterations / 1000.0 for key, value in totals.items()
    }
    timed_total_us = sum(per_window_us.values())
    groups: dict[str, float] = defaultdict(float)
    for key, value in per_window_us.items():
        if key.startswith("metadata."):
            groups["metadata"] += value
        elif key.startswith("allocate.fixed."):
            groups["fixed_allocation"] += value
        elif key.startswith("allocate.dynamic."):
            groups["dynamic_allocation_and_fill"] += value
        elif key.startswith("copy.fixed."):
            groups["fixed_copy"] += value
        elif key.startswith("concatenate.fixed."):
            groups["fixed_concatenate"] += value
        elif key.startswith("copy.dynamic."):
            groups["dynamic_copy"] += value
        elif key.startswith("event."):
            groups["event_audit_crop"] += value
        elif key.startswith("scatter."):
            groups["scatter_preparation"] += value
    event_source_bytes = args.rows * args.event_width * (41 * 2 + 1)
    event_transmitted_bytes = (
        args.rows * args.event_token_limit * (41 * 2 + 1)
        if args.event_token_limit is not None
        else event_source_bytes
    )
    return {
        "shape": {
            "requests": args.requests,
            "rows": args.rows,
            "rows_per_request": args.rows / args.requests,
            "max_legal": args.max_legal,
            "true_legal_cells": true_legal,
            "allocated_legal_cells": allocated_legal,
            "legal_occupancy": true_legal / allocated_legal,
            "event_width": args.event_width,
            "event_token_limit": args.event_token_limit,
            "merge_strategy": args.merge_strategy,
            "server_receives_client_crop": args.server_receives_client_crop,
        },
        "time_us_per_window": dict(sorted(per_window_us.items())),
        "time_groups_us_per_window": {
            **dict(sorted(groups.items())),
            "timed_total": timed_total_us,
        },
        "bytes": {
            "source_payload": source_bytes,
            "source_event_fields": event_source_bytes,
            "transmitted_event_fields": event_transmitted_bytes,
            "event_ipc_bytes_avoided": event_source_bytes - event_transmitted_bytes,
            "merged_output": sum(last_field_bytes.values()),
            "scatter_output_with_q": scatter_bytes,
            "merged_fields": dict(sorted(last_field_bytes.items())),
        },
        "iterations": args.iterations,
        "parity": "exact",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=36)
    parser.add_argument("--rows", type=int, default=44)
    parser.add_argument("--max-legal", type=int, default=18)
    parser.add_argument("--occupancy", type=float, default=0.392)
    parser.add_argument("--event-width", type=int, default=64)
    parser.add_argument(
        "--event-token-limit",
        type=int,
        default=0,
        help="Explicit event prefix. Use -1 to profile the uncropped path.",
    )
    parser.add_argument("--include-action-targets", action="store_true")
    parser.add_argument(
        "--merge-strategy", choices=("loop", "hybrid"), default="hybrid"
    )
    parser.add_argument(
        "--server-receives-client-crop",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.event_token_limit < 0:
        args.event_token_limit = None
    if args.iterations <= 0 or args.warmup < 0:
        parser.error("iterations must be positive and warmup non-negative")
    return args


def main() -> None:
    args = _parse_args()
    summary = _summarize(args)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    shape = summary["shape"]
    print(
        "shape: "
        f"{shape['requests']} requests, {shape['rows']} rows, "
        f"max legal {shape['max_legal']}, occupancy {shape['legal_occupancy']:.1%}, "
        f"event {shape['event_width']} -> {shape['event_token_limit']}"
    )
    print("time per window:")
    total = summary["time_groups_us_per_window"]["timed_total"]
    for key, value in summary["time_groups_us_per_window"].items():
        if key != "timed_total":
            print(f"  {key:31s} {value:9.3f} us  {value / total:6.1%}")
    print(f"  {'timed_total':31s} {total:9.3f} us")
    print("bytes:")
    for key, value in summary["bytes"].items():
        if not isinstance(value, dict):
            print(f"  {key:31s} {value:12,d}")
    print("parity: exact")


if __name__ == "__main__":
    main()
