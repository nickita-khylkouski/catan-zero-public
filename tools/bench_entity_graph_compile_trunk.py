#!/usr/bin/env python3
"""Benchmark torch.compile on only the fixed-shape EntityGraph state trunk.

The variable legal-action head stays eager.  The tool exercises multiple batch
and legal widths, records Dynamo's unique-graph counter after every first call,
checks strict-FP32 outputs against eager, and then measures steady-state trunk
and whole-model device latency.  It is an experiment, not a core integration.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Callable

import numpy as np

from bench_entity_graph_stages import _host_to_device, _synthetic_batch


def _csv_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return parsed


def _summary(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    p95 = ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]
    return {
        "mean": float(statistics.fmean(ordered)),
        "median": float(statistics.median(ordered)),
        "p95": float(p95),
        "min": float(ordered[0]),
    }


def _benchmark(call: Callable[[], Any], *, warmup: int, iterations: int, torch: Any) -> dict[str, float]:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return _summary(samples)


def _forward_with_encoder(model: Any, batch: dict[str, Any], action_ids: Any, encoder: Callable[..., Any], *, return_q: bool):
    encoded = encoder(batch)
    outputs = model.score_actions(encoded, batch, return_q=return_q)
    outputs["logits"] = outputs["logits"].masked_fill(~(action_ids >= 0), -1.0e9)
    return outputs


def _drift(reference: dict[str, Any], candidate: dict[str, Any], torch: Any) -> dict[str, Any]:
    metrics = {}
    if reference.keys() != candidate.keys():
        raise RuntimeError(
            f"output keys differ: eager={sorted(reference)} compiled={sorted(candidate)}"
        )
    for key in sorted(reference):
        left = reference[key].detach().float()
        right = candidate[key].detach().float()
        delta = (left - right).abs()
        entry = {
            "max_abs": float(delta.max().item()) if delta.numel() else 0.0,
            "mean_abs": float(delta.mean().item()) if delta.numel() else 0.0,
        }
        if key == "logits":
            entry["argmax_agreement"] = float(
                (left.argmax(dim=-1) == right.argmax(dim=-1)).float().mean().item()
            )
        metrics[key] = entry
    return metrics


def _counter_snapshot(counters: Any) -> dict[str, int]:
    return {
        "unique_graphs": int(counters["stats"]["unique_graphs"]),
        "calls_captured": int(counters["stats"]["calls_captured"]),
        "frames_total": int(counters["frames"]["total"]),
        "frames_ok": int(counters["frames"]["ok"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-sizes", type=_csv_ints, default=(32, 48, 64, 96))
    parser.add_argument("--legal-widths", type=_csv_ints, default=(8, 54, 24, 40))
    parser.add_argument("--mode", default="default")
    parser.add_argument("--dynamic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fullgraph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    if len(args.legal_widths) not in {1, len(args.batch_sizes)}:
        raise SystemExit("--legal-widths must contain one width or one per batch size")
    legal_widths = (
        args.legal_widths * len(args.batch_sizes)
        if len(args.legal_widths) == 1
        else args.legal_widths
    )

    import torch
    from torch._dynamo.utils import counters
    from catan_zero.rl.entity_token_policy import EntityGraphPolicy

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("this benchmark requires CUDA")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    policy = EntityGraphPolicy.load(args.checkpoint, device=str(device))
    policy.model.eval().float()

    batches = []
    for index, (batch_size, legal_width) in enumerate(zip(args.batch_sizes, legal_widths)):
        entity, legal_ids_np, context = _synthetic_batch(
            batch_size=batch_size,
            legal_width=legal_width,
            valid_legal_fraction=0.392,
            event_width=0,
            valid_players=2,
            seed=args.seed + index,
        )
        batch, action_ids = _host_to_device(
            policy, entity, legal_ids_np, context, torch
        )
        batches.append((batch_size, legal_width, batch, action_ids))

    torch._dynamo.reset()
    counters.clear()
    compiled_encode = torch.compile(
        policy.model.encode_state,
        dynamic=args.dynamic,
        fullgraph=args.fullgraph,
        mode=args.mode,
    )

    first_calls = []
    parity = {}
    with torch.inference_mode():
        for batch_size, legal_width, batch, action_ids in batches:
            eager = _forward_with_encoder(
                policy.model, batch, action_ids, policy.model.encode_state, return_q=True
            )
            torch.cuda.synchronize()
            started = time.perf_counter()
            compiled = _forward_with_encoder(
                policy.model, batch, action_ids, compiled_encode, return_q=True
            )
            torch.cuda.synchronize()
            first_calls.append(
                {
                    "batch_size": batch_size,
                    "legal_width": legal_width,
                    "wall_seconds": time.perf_counter() - started,
                    "dynamo_counters": _counter_snapshot(counters),
                }
            )
            parity[str(batch_size)] = _drift(eager, compiled, torch)

        # Revisit earlier shapes after all four calls.  A shape-specialized
        # implementation that evicted/recompiled is visible in the counters.
        before_revisit = _counter_snapshot(counters)
        for _batch_size, _legal_width, batch, _action_ids in reversed(batches):
            compiled_encode(batch)
        torch.cuda.synchronize()
        after_revisit = _counter_snapshot(counters)

        timings = []
        for batch_size, legal_width, batch, action_ids in batches:
            eager_trunk = _benchmark(
                lambda batch=batch: policy.model.encode_state(batch),
                warmup=args.warmup, iterations=args.iterations, torch=torch,
            )
            compiled_trunk = _benchmark(
                lambda batch=batch: compiled_encode(batch),
                warmup=args.warmup, iterations=args.iterations, torch=torch,
            )
            eager_full = _benchmark(
                lambda batch=batch, action_ids=action_ids: _forward_with_encoder(
                    policy.model, batch, action_ids, policy.model.encode_state, return_q=True
                ),
                warmup=args.warmup, iterations=args.iterations, torch=torch,
            )
            compiled_full = _benchmark(
                lambda batch=batch, action_ids=action_ids: _forward_with_encoder(
                    policy.model, batch, action_ids, compiled_encode, return_q=True
                ),
                warmup=args.warmup, iterations=args.iterations, torch=torch,
            )
            timings.append(
                {
                    "batch_size": batch_size,
                    "legal_width": legal_width,
                    "eager_trunk_cuda_ms": eager_trunk,
                    "compiled_trunk_cuda_ms": compiled_trunk,
                    "trunk_speedup": eager_trunk["mean"] / compiled_trunk["mean"],
                    "eager_full_cuda_ms": eager_full,
                    "compiled_trunk_eager_head_cuda_ms": compiled_full,
                    "full_speedup": eager_full["mean"] / compiled_full["mean"],
                }
            )

    result = {
        "device": torch.cuda.get_device_properties(device).name,
        "checkpoint": args.checkpoint,
        "strict_fp32": True,
        "compile": {
            "boundary": "encode_state only; score_actions eager",
            "mode": args.mode,
            "dynamic": args.dynamic,
            "fullgraph": args.fullgraph,
        },
        "first_calls": first_calls,
        "before_revisit": before_revisit,
        "after_revisit": after_revisit,
        "output_drift": parity,
        "timings": timings,
        "limitations": [
            "synthetic event0 tensors with real checkpoint weights",
            "timings use preloaded CUDA tensors and exclude H2D/D2H",
            "Dynamo counters are process-global but reset immediately before this isolated compile",
            "steady-state microbenchmark is not an end-to-end self-play certification",
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
