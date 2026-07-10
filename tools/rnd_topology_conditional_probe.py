#!/usr/bin/env python3
"""Probe receiver-conditioned selection over real Catan incidence edges.

Every paired scene has identical random keys and values on the 19 hex tokens.
The two copies place opposite queries on every vertex.  A vertex target is the
softmax-weighted value of its physically incident hexes, where the weights are
computed from the receiver query and source key after the same parameter-free
RMS normalization used at adapter input.  The target therefore discards no
information that the candidate architectures need to represent it.

The scored receiver coordinate is always zero and the readout is the fixed
selection ``output[..., 0]``.  Consequently ``basis_mean_v1`` produces the
same prediction for both copies of a scene: its messages do not depend on the
receiver, and the residual cannot leak the query into the scored coordinate.
``local_attention_v2`` can represent the target through its receiver queries.
Self-message and type-cyclic rewiring retain the same kernels while removing
access to the correct incident values.  This is a mechanism/identifiability
probe, not evidence of Catan playing strength.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
        "--kind", choices=("basis_mean_v1", "local_attention_v2"), required=True
    )
    parser.add_argument(
        "--edge-control",
        choices=("true_topology", "self_message", "type_cyclic_rewire"),
        default="true_topology",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--bottleneck", type=int, default=16)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--key-dims", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=400)
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


def conditional_targets_from_inputs(
    x,
    edges,
    *,
    key_dims: int,
    temperature: float,
):
    """Return targets/mask using only true vertex-to-hex incidence edges."""
    import torch
    from torch.nn import functional as F

    from catan_zero.rl.relational_trunks import REL_VERTEX_TO_HEX

    visible = F.rms_norm(x, (int(x.shape[-1]),))
    source, destination, relation, valid = edges
    selected = valid & relation.eq(REL_VERTEX_TO_HEX)
    batch_index = torch.arange(x.shape[0], device=x.device)[:, None]
    queries = visible[batch_index, destination, 1 : 1 + int(key_dims)]
    keys = visible[batch_index, source, 1 : 1 + int(key_dims)]
    values = visible[batch_index, source, 0]
    logits = queries.float().mul(keys.float()).sum(dim=-1) * (
        float(temperature) / math.sqrt(int(key_dims))
    )
    logits = logits.masked_fill(~selected, float("-inf"))

    maxima = torch.full(
        x.shape[:2], float("-inf"), dtype=torch.float32, device=x.device
    )
    maxima.scatter_reduce_(1, destination, logits, reduce="amax", include_self=True)
    shifted = logits - maxima.gather(1, destination)
    shifted = shifted.masked_fill(~selected, float("-inf"))
    numerators = shifted.exp()
    denominators = torch.zeros_like(maxima)
    denominators.scatter_add_(1, destination, numerators)
    weights = numerators / denominators.gather(1, destination).clamp_min(1e-12)

    targets = torch.zeros(x.shape[:2], dtype=x.dtype, device=x.device)
    targets.scatter_add_(1, destination, weights.to(x.dtype) * values)
    degree = torch.zeros(x.shape[:2], dtype=torch.int32, device=x.device)
    degree.scatter_add_(1, destination, selected.to(torch.int32))
    return targets, degree.gt(0)


def construct_conditional_examples(
    edges,
    *,
    batch_size: int,
    width: int,
    key_dims: int,
    temperature: float,
    generator,
    device,
):
    """Construct paired ``q``/``-q`` scenes with shared hex keys and values."""
    import torch

    if int(batch_size) < 2 or int(batch_size) % 2:
        raise ValueError("batch-size must be even and >= 2 for paired scenes")
    if int(key_dims) < 1 or int(width) <= int(key_dims):
        raise ValueError("width must exceed positive key-dims by at least one")

    pair_count = int(batch_size) // 2
    base = torch.zeros(pair_count, 151, int(width), device=device)
    hex_values = torch.randn(pair_count, 19, generator=generator, device=device)
    hex_keys = torch.randn(
        pair_count, 19, int(key_dims), generator=generator, device=device
    )
    queries = torch.randn(
        pair_count, 54, int(key_dims), generator=generator, device=device
    )
    base[:, 1:20, 0] = hex_values
    base[:, 1:20, 1 : 1 + int(key_dims)] = hex_keys

    x = torch.cat((base.clone(), base.clone()), dim=0)
    x[:pair_count, 20:74, 1 : 1 + int(key_dims)] = queries
    x[pair_count:, 20:74, 1 : 1 + int(key_dims)] = -queries
    targets, live = conditional_targets_from_inputs(
        x, edges, key_dims=int(key_dims), temperature=float(temperature)
    )
    return x, targets, live


def run(args: argparse.Namespace) -> dict:
    import torch

    from catan_zero.rl.sparse_topology_adapter import (
        apply_sparse_edge_control,
        build_sparse_incidence_edges,
        create_sparse_topology_adapter,
    )

    started = time.perf_counter()
    positive = (
        args.width,
        args.bottleneck,
        args.heads,
        args.key_dims,
        args.batch_size,
        args.steps,
    )
    if min(positive) < 1:
        raise ValueError(
            "width, bottleneck, heads, key-dims, batch-size, and steps must be positive"
        )
    if not math.isfinite(float(args.temperature)) or float(args.temperature) <= 0:
        raise ValueError("temperature must be finite and positive")
    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))
    topology = _topology_batch(int(args.batch_size), device)
    true_edges = build_sparse_incidence_edges(topology, sequence_length=151)
    model_edges = apply_sparse_edge_control(
        true_edges, args.edge_control, sequence_length=151
    )
    adapter = create_sparse_topology_adapter(
        kind=args.kind,
        width=int(args.width),
        bottleneck=int(args.bottleneck),
        bases=4,
        heads=int(args.heads),
        dropout=0.0,
    ).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=float(args.lr))
    generator = torch.Generator(device=device).manual_seed(int(args.seed) + 1)
    losses: list[float] = []
    zero_predictor_losses: list[float] = []
    target_pair_gaps: list[float] = []
    prediction_pair_gaps: list[float] = []
    pair_count = int(args.batch_size) // 2
    for _step in range(int(args.steps)):
        x, targets, live = construct_conditional_examples(
            true_edges,
            batch_size=int(args.batch_size),
            width=int(args.width),
            key_dims=int(args.key_dims),
            temperature=float(args.temperature),
            generator=generator,
            device=device,
        )
        output = adapter(x, edges=model_edges)
        prediction = output[:, 20:74, 0]
        vertex_target = targets[:, 20:74]
        vertex_live = live[:, 20:74]
        loss = (prediction[vertex_live] - vertex_target[vertex_live]).square().mean()
        zero_predictor_loss = vertex_target[vertex_live].square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        zero_predictor_losses.append(float(zero_predictor_loss.detach()))
        target_pair_gaps.append(
            float(
                (vertex_target[:pair_count] - vertex_target[pair_count:])
                .abs()[vertex_live[:pair_count]]
                .mean()
            )
        )
        prediction_pair_gaps.append(
            float(
                (prediction[:pair_count] - prediction[pair_count:])
                .abs()[vertex_live[:pair_count]]
                .mean()
                .detach()
            )
        )

    tail_count = min(20, len(losses))
    tail_mean_loss = sum(losses[-tail_count:]) / tail_count
    tail_mean_zero_predictor_loss = (
        sum(zero_predictor_losses[-tail_count:]) / tail_count
    )
    source_files = (
        "src/catan_zero/rl/entity_token_features.py",
        "src/catan_zero/rl/multiagent_env.py",
        "src/catan_zero/rl/relational_trunks.py",
        "src/catan_zero/rl/sparse_topology_adapter.py",
        "tools/rnd_topology_conditional_probe.py",
    )
    source_sha256 = {
        path: hashlib.sha256((_ROOT / path).read_bytes()).hexdigest()
        for path in source_files
    }
    return {
        "schema_version": "catan-zero-topology-conditional-probe/v1",
        "claim_scope": "receiver-conditioned topology mechanism; not playing strength",
        "kind": str(args.kind),
        "edge_control": str(args.edge_control),
        "device": str(device),
        "width": int(args.width),
        "bottleneck": int(args.bottleneck),
        "heads": int(args.heads),
        "key_dims": int(args.key_dims),
        "temperature": float(args.temperature),
        "batch_size": int(args.batch_size),
        "steps": int(args.steps),
        "seed": int(args.seed),
        "parameter_count": sum(parameter.numel() for parameter in adapter.parameters()),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "tail_mean_loss": tail_mean_loss,
        "tail_mean_zero_predictor_loss": tail_mean_zero_predictor_loss,
        "tail_normalized_improvement": 1.0
        - tail_mean_loss / max(tail_mean_zero_predictor_loss, 1e-12),
        "tail_mean_target_pair_gap": sum(target_pair_gaps[-tail_count:]) / tail_count,
        "tail_mean_prediction_pair_gap": sum(prediction_pair_gaps[-tail_count:])
        / tail_count,
        "finite": all(math.isfinite(value) for value in losses),
        "elapsed_sec": time.perf_counter() - started,
        "torch_version": str(torch.__version__),
        "gpu_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "cuda_device_count": torch.cuda.device_count(),
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
