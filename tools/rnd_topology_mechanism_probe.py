#!/usr/bin/env python3
"""Probe whether an adapter transmits the correct Catan incidence signal.

Random scalar production values live on hex tokens. The supervised target at
each vertex is the mean value of its physically incident hexes. A self-message
or rewired control sees identical tensors and kernels but not the correct
neighbor values. This is an expressivity/sanity probe, not playing strength.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Sequence

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
for _path in (_ROOT, _SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind",
        choices=("basis_mean_v1", "local_attention_v2"),
        required=True,
    )
    parser.add_argument(
        "--edge-control",
        choices=("true_topology", "self_message", "type_degree_preserving_rewire"),
        default="true_topology",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--bottleneck", type=int, default=16)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--output", default="")
    return parser


def _topology_batch(batch_size: int, device):
    import torch

    from catan_zero.rl.entity_token_features import build_entity_token_features
    from catan_zero.rl.multiagent_env import (
        ColonistMultiAgentConfig,
        ColonistMultiAgentEnv,
    )

    env = ColonistMultiAgentEnv(ColonistMultiAgentConfig())
    try:
        _observations, info = env.reset(seed=20260710)
        entity = build_entity_token_features(
            env,
            actor=str(info["current_player"]),
            include_event_log=False,
        )
    finally:
        env.close()
    batch = {}
    for key in ("hex_vertex_ids", "hex_edge_ids", "edge_vertex_ids"):
        value = torch.as_tensor(entity[key], dtype=torch.long, device=device)
        batch[key] = value.unsqueeze(0).expand(batch_size, *value.shape).clone()
    return batch


def run(args: argparse.Namespace) -> dict:
    import torch
    from torch import nn

    from catan_zero.rl.relational_trunks import REL_VERTEX_TO_HEX
    from catan_zero.rl.sparse_topology_adapter import (
        apply_sparse_edge_control,
        build_sparse_incidence_edges,
        create_sparse_topology_adapter,
    )

    started = time.perf_counter()
    if min(args.width, args.bottleneck, args.heads, args.batch_size, args.steps) < 1:
        raise ValueError(
            "width, bottleneck, heads, batch-size, and steps must be positive"
        )
    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))
    topology = _topology_batch(int(args.batch_size), device)
    edges = build_sparse_incidence_edges(topology, sequence_length=151)
    true_edges = edges
    edges = apply_sparse_edge_control(edges, args.edge_control, sequence_length=151)
    source, destination, relation, valid = true_edges
    vertex_hex = valid & relation.eq(REL_VERTEX_TO_HEX)
    if not bool(vertex_hex.any()):
        raise RuntimeError("mechanism probe found no vertex-to-hex incidence edges")

    adapter = create_sparse_topology_adapter(
        kind=args.kind,
        width=int(args.width),
        bottleneck=int(args.bottleneck),
        bases=4,
        heads=int(args.heads),
        dropout=0.0,
    ).to(device)
    readout = nn.Linear(int(args.width), 1).to(device)
    optimizer = torch.optim.AdamW(
        [*adapter.parameters(), *readout.parameters()], lr=float(args.lr)
    )
    generator = torch.Generator(device=device).manual_seed(int(args.seed) + 1)
    losses: list[float] = []
    for _step in range(int(args.steps)):
        x = torch.randn(
            int(args.batch_size),
            151,
            int(args.width),
            generator=generator,
            device=device,
        )
        batch_index = torch.arange(int(args.batch_size), device=device)[:, None]
        values = x[batch_index, source, 0] * vertex_hex.to(x.dtype)
        target_sum = torch.zeros(
            int(args.batch_size), 151, dtype=x.dtype, device=device
        )
        target_sum.scatter_add_(1, destination, values)
        degree = torch.zeros_like(target_sum)
        degree.scatter_add_(1, destination, vertex_hex.to(x.dtype))
        target = target_sum / degree.clamp_min(1)
        live_vertices = degree[:, 20:74].gt(0)

        output = adapter(x, edges=edges)
        prediction = readout(output[:, 20:74]).squeeze(-1)
        loss = (
            (prediction[live_vertices] - target[:, 20:74][live_vertices])
            .square()
            .mean()
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    tail = losses[-min(20, len(losses)) :]
    source_files = (
        "src/catan_zero/rl/sparse_topology_adapter.py",
        "tools/rnd_topology_mechanism_probe.py",
    )
    source_sha256 = {
        path: hashlib.sha256((_ROOT / path).read_bytes()).hexdigest()
        for path in source_files
    }
    return {
        "schema_version": "catan-zero-topology-mechanism-probe/v1",
        "kind": str(args.kind),
        "edge_control": str(args.edge_control),
        "device": str(device),
        "width": int(args.width),
        "bottleneck": int(args.bottleneck),
        "heads": int(args.heads),
        "batch_size": int(args.batch_size),
        "steps": int(args.steps),
        "seed": int(args.seed),
        "parameter_count": sum(
            parameter.numel()
            for module in (adapter, readout)
            for parameter in module.parameters()
        ),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "tail_mean_loss": sum(tail) / len(tail),
        "finite": all(
            value == value and abs(value) != float("inf") for value in losses
        ),
        "elapsed_sec": time.perf_counter() - started,
        "torch_version": str(torch.__version__),
        "gpu_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else None,
        "source_sha256": source_sha256,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run(args)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
