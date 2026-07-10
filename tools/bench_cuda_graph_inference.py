#!/usr/bin/env python3
"""Benchmark the opt-in EntityGraph state-trunk CUDA Graph runner.

This is a synthetic-shape microbenchmark, not a self-play throughput or model
strength certification.  It loads a real checkpoint, checks eager/graph output
parity, and reports per-window latency for each requested batch size.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np


def _csv_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError(
            "expected at least one comma-separated integer"
        )
    return parsed


def _synthetic_batch(
    *,
    batch_size: int,
    legal_width: int,
    event_width: int,
    live_events: int,
    seed: int,
):
    from catan_zero.rl.action_features import CONTEXT_ACTION_FEATURE_SIZE
    from catan_zero.rl.entity_token_features import (
        EDGE_FEATURE_SIZE,
        EVENT_FEATURE_SIZE,
        GLOBAL_FEATURE_SIZE,
        HEX_FEATURE_SIZE,
        LEGAL_ACTION_FEATURE_SIZE,
        PLAYER_FEATURE_SIZE,
        VERTEX_FEATURE_SIZE,
    )

    rng = np.random.default_rng(seed)
    entity = {}
    for name, count, width in (
        ("hex", 19, HEX_FEATURE_SIZE),
        ("vertex", 54, VERTEX_FEATURE_SIZE),
        ("edge", 72, EDGE_FEATURE_SIZE),
        ("player", 4, PLAYER_FEATURE_SIZE),
        ("global", 1, GLOBAL_FEATURE_SIZE),
        ("event", event_width, EVENT_FEATURE_SIZE),
    ):
        entity[f"{name}_tokens"] = rng.normal(size=(batch_size, count, width)).astype(
            np.float16
        )
        if name != "global":
            entity[f"{name}_mask"] = np.ones((batch_size, count), dtype=np.bool_)
    entity["event_mask"][:, live_events:] = False
    entity["legal_action_tokens"] = rng.normal(
        size=(batch_size, legal_width, LEGAL_ACTION_FEATURE_SIZE)
    ).astype(np.float16)
    entity["legal_action_target_ids"] = np.full(
        (batch_size, legal_width, 4), -1, dtype=np.int16
    )
    legal_ids = np.tile(np.arange(legal_width, dtype=np.int64), (batch_size, 1))
    entity["legal_action_mask"] = legal_ids >= 0
    context = rng.normal(
        size=(batch_size, legal_width, CONTEXT_ACTION_FEATURE_SIZE)
    ).astype(np.float32)
    return entity, legal_ids, context


def _timed(call, iterations: int, torch) -> float:
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        call()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000.0 / iterations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-sizes", type=_csv_ints, default=(32, 40, 48, 64, 96))
    parser.add_argument(
        "--batch-buckets",
        type=_csv_ints,
        default=(8, 16, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192),
    )
    parser.add_argument("--legal-width", type=int, default=24)
    parser.add_argument("--event-width", type=int, default=64)
    parser.add_argument("--live-events", type=int, default=64)
    parser.add_argument("--event-token-limit", type=int)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260709)
    args = parser.parse_args()

    import torch

    from catan_zero.rl.entity_token_policy import EntityGraphPolicy
    from catan_zero.search.cuda_graph_inference import (
        CudaGraphInferenceConfig,
        CudaGraphInferenceRunner,
    )

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("this benchmark requires an available CUDA device")
    if not 0 <= args.live_events <= args.event_width:
        raise SystemExit("--live-events must be within --event-width")
    if args.event_token_limit is not None and args.live_events > args.event_token_limit:
        raise SystemExit("--event-token-limit would remove live synthetic events")

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    policy = EntityGraphPolicy.load(args.checkpoint, device=str(device))
    policy.model.eval()
    common_runner_config = {
        "batch_buckets": args.batch_buckets,
        "event_token_limit": args.event_token_limit,
    }
    runner = CudaGraphInferenceRunner(
        policy,
        CudaGraphInferenceConfig(
            enabled=True,
            **common_runner_config,
        ),
    )
    eager_runner = CudaGraphInferenceRunner(
        policy,
        CudaGraphInferenceConfig(
            enabled=False,
            **common_runner_config,
        ),
    )

    results = []
    for batch_size in args.batch_sizes:
        entity, legal_ids, context = _synthetic_batch(
            batch_size=batch_size,
            legal_width=args.legal_width,
            event_width=args.event_width,
            live_events=args.live_events,
            seed=args.seed + batch_size,
        )

        def eager():
            return eager_runner.forward_legal_np(entity, legal_ids, context)

        def graphed():
            return runner.forward_legal_np(entity, legal_ids, context)

        with torch.no_grad():
            eager_output = eager()
            graph_output = graphed()
        if runner.last_path != "cuda_graph":
            raise RuntimeError(
                f"batch {batch_size} did not use CUDA Graph: "
                f"{runner.last_fallback_reason}"
            )
        if eager_output.keys() != graph_output.keys():
            raise RuntimeError(
                "output key mismatch: "
                f"eager={sorted(eager_output)} graph={sorted(graph_output)}"
            )
        max_abs = {}
        for key in eager_output:
            left = eager_output[key].float()
            right = graph_output[key].float()
            if left.shape != right.shape:
                raise RuntimeError(
                    f"output shape mismatch for {key}: {left.shape} != {right.shape}"
                )
            max_abs[key] = float((left - right).abs().max().item())
            torch.testing.assert_close(left, right, rtol=1.0e-4, atol=1.0e-5)

        for _ in range(args.warmup):
            eager()
            graphed()
        results.append(
            {
                "batch_size": batch_size,
                "selected_bucket": runner.selected_batch_bucket(batch_size),
                "legal_width": args.legal_width,
                "eager_ms_per_window": _timed(eager, args.iterations, torch),
                "graph_ms_per_window": _timed(graphed, args.iterations, torch),
                "max_abs_output_difference": max_abs,
            }
        )

    properties = torch.cuda.get_device_properties(device)
    print(
        json.dumps(
            {
                "device": properties.name,
                "checkpoint": args.checkpoint,
                "strict_fp32": True,
                "event_width": args.event_width,
                "live_events": args.live_events,
                "event_token_limit": args.event_token_limit,
                "iterations": args.iterations,
                "graph_count": runner.graph_count,
                "results": results,
                "scope": "synthetic-shape inference microbenchmark; not self-play certification",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
