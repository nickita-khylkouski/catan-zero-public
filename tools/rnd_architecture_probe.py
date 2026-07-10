#!/usr/bin/env python3
"""Bounded H100 smoke/throughput probe for the frozen E2 architecture arms.

This is not a strength benchmark.  It verifies exact parameter counts, finite
forward/backward behavior, approximate step throughput, and peak allocated GPU
memory under one shared synthetic entity/action shape.  The JSON output is an
R&D artifact and records enough shape/config detail to avoid comparing unlike
runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trunk", choices=("transformer", "rrt", "resrgcn"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--require-gpu-name",
        default="NVIDIA H100",
        help="required substring in torch.cuda.get_device_name (default: NVIDIA H100)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--legal-actions", type=int, default=64)
    parser.add_argument("--events", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--value-categorical-bins", type=int, default=0)
    parser.add_argument("--latent-deliberation-steps", type=int, default=0)
    parser.add_argument("--latent-deliberation-slots", type=int, default=8)
    parser.add_argument("--moe-routed-experts", type=int, default=0)
    parser.add_argument("--moe-top-k", type=int, default=2)
    parser.add_argument("--moe-expert-ff-size", type=int, default=384)
    parser.add_argument("--output")
    return parser


def _resolved_architecture(trunk: str) -> dict[str, Any]:
    if trunk == "transformer":
        return {"hidden_size": 640, "state_layers": 6, "attention_heads": 8}
    if trunk == "rrt":
        return {
            "hidden_size": 384,
            "state_layers": 9,
            "attention_heads": 6,
            "relational_block_pattern": "RRTRRTRRT",
            "relational_ff_size": 1024,
            "relational_action_cross_layers": 1,
        }
    return {
        "hidden_size": 384,
        "state_layers": 14,
        "attention_heads": 6,
        "relational_ff_size": 512,
        "relational_bases": 4,
        "relational_action_cross_layers": 0,
    }


def _source_provenance() -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[1]
    relative_paths = (
        "tools/rnd_architecture_probe.py",
        "src/catan_zero/rl/entity_token_policy.py",
        "src/catan_zero/rl/relational_trunks.py",
    )
    file_hashes: dict[str, str] = {}
    aggregate = hashlib.sha256()
    for relative in relative_paths:
        content = (repo / relative).read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        file_hashes[relative] = digest
        aggregate.update(relative.encode("utf-8") + b"\0" + content + b"\0")
    git_commit = None
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return {
        "git_commit": git_commit,
        "source_bundle_sha256": aggregate.hexdigest(),
        "file_sha256": file_hashes,
    }


def _batch(config: Any, *, batch_size: int, actions: int, events: int, device: Any):
    import torch

    from catan_zero.rl.entity_token_features import (
        EDGE_FEATURE_SIZE,
        EVENT_FEATURE_SIZE,
        GLOBAL_FEATURE_SIZE,
        HEX_FEATURE_SIZE,
        LEGAL_ACTION_FEATURE_SIZE,
        PLAYER_FEATURE_SIZE,
        VERTEX_FEATURE_SIZE,
    )

    generator = torch.Generator(device=device).manual_seed(20260710)
    result: dict[str, Any] = {}
    for name, count, width in (
        ("hex", 19, HEX_FEATURE_SIZE),
        ("vertex", 54, VERTEX_FEATURE_SIZE),
        ("edge", 72, EDGE_FEATURE_SIZE),
        ("player", 4, PLAYER_FEATURE_SIZE),
        ("global", 1, GLOBAL_FEATURE_SIZE),
        ("event", events, EVENT_FEATURE_SIZE),
    ):
        result[f"{name}_tokens"] = torch.randn(
            batch_size, count, width, generator=generator, device=device
        )
        if name != "global":
            result[f"{name}_mask"] = torch.ones(
                batch_size, count, dtype=torch.bool, device=device
            )
    result["legal_action_tokens"] = torch.randn(
        batch_size,
        actions,
        LEGAL_ACTION_FEATURE_SIZE,
        generator=generator,
        device=device,
    )
    result["legal_action_context"] = torch.randn(
        batch_size,
        actions,
        int(config.context_action_feature_size),
        generator=generator,
        device=device,
    )
    result["legal_action_mask"] = torch.ones(
        batch_size, actions, dtype=torch.bool, device=device
    )
    result["legal_action_target_ids"] = torch.full(
        (batch_size, actions, 4), -1, dtype=torch.long, device=device
    )
    result["legal_action_target_ids"][:, :, 1] = torch.arange(
        actions, device=device
    ).remainder(54)
    result["hex_vertex_ids"] = torch.full(
        (batch_size, 19, 6), -1, dtype=torch.long, device=device
    )
    result["hex_edge_ids"] = torch.full(
        (batch_size, 19, 6), -1, dtype=torch.long, device=device
    )
    result["edge_vertex_ids"] = torch.full(
        (batch_size, 72, 2), -1, dtype=torch.long, device=device
    )
    result["event_target_ids"] = torch.full(
        (batch_size, events, 4), -1, dtype=torch.long, device=device
    )
    # Connect a deterministic valid subset. Unfilled slots mean no incidence.
    result["hex_vertex_ids"][:, 0, :2] = torch.tensor((0, 1), device=device)
    result["hex_edge_ids"][:, 0, :2] = torch.tensor((0, 1), device=device)
    result["edge_vertex_ids"][:, 0, :] = torch.tensor((0, 1), device=device)
    if events:
        result["event_target_ids"][:, 0, 1] = 0
    return result


def main() -> None:
    args = _parser().parse_args()
    if min(args.batch_size, args.legal_actions, args.iterations) < 1:
        raise SystemExit("batch-size, legal-actions, and iterations must be positive")
    if min(args.events, args.warmup, args.value_categorical_bins) < 0:
        raise SystemExit("events, warmup, and categorical bins must be non-negative")

    import torch

    from catan_zero.rl.action_features import CONTEXT_ACTION_FEATURE_SIZE
    from catan_zero.rl.entity_token_features import LEGAL_ACTION_FEATURE_SIZE
    from catan_zero.rl.entity_token_policy import EntityGraphConfig, EntityGraphNet

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("this bounded probe requires an available CUDA device")
    gpu_name = torch.cuda.get_device_name(device)
    required_gpu_name = str(args.require_gpu_name).strip()
    if required_gpu_name and required_gpu_name not in gpu_name:
        raise SystemExit(
            f"GPU identity mismatch: required substring {required_gpu_name!r}, "
            f"measured {gpu_name!r}"
        )
    architecture = _resolved_architecture(args.trunk)
    config = EntityGraphConfig(
        action_size=607,
        static_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
        context_action_feature_size=CONTEXT_ACTION_FEATURE_SIZE,
        legal_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
        dropout=0.05,
        state_trunk=args.trunk,
        value_categorical_bins=int(args.value_categorical_bins),
        value_categorical_truncation_class=True,
        latent_deliberation_steps=int(args.latent_deliberation_steps),
        latent_deliberation_slots=int(args.latent_deliberation_slots),
        moe_routed_experts=int(args.moe_routed_experts),
        moe_top_k=int(args.moe_top_k),
        moe_expert_ff_size=int(args.moe_expert_ff_size),
        **architecture,
    )
    torch.manual_seed(20260710)
    model = EntityGraphNet(config).to(device).train()
    batch = _batch(
        config,
        batch_size=int(args.batch_size),
        actions=int(args.legal_actions),
        events=int(args.events),
        device=device,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    torch.cuda.reset_peak_memory_stats(device)

    def step() -> float:
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(batch, return_q=True)
            loss = (
                outputs["logits"].float().square().mean()
                + outputs["q_values"].float().square().mean()
                + outputs["value"].float().square().mean()
            )
        loss.backward()
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite architecture-probe loss")
        gradients = [
            (name, parameter.grad)
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        ]
        if not gradients:
            raise RuntimeError("architecture probe produced no parameter gradients")
        nonfinite_gradients = [
            name for name, gradient in gradients if not torch.isfinite(gradient).all()
        ]
        if nonfinite_gradients:
            raise RuntimeError(
                "non-finite architecture-probe gradients: "
                + ", ".join(nonfinite_gradients[:8])
            )
        return float(loss.detach())

    for _ in range(int(args.warmup)):
        step()
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    losses = [step() for _ in range(int(args.iterations))]
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    payload = {
        "schema_version": "catan-zero-rnd-architecture-probe/v1",
        "trunk": args.trunk,
        "resolved_architecture": architecture,
        "parameter_count": int(parameter_count),
        "value_categorical_bins": int(args.value_categorical_bins),
        "latent_deliberation_steps": int(args.latent_deliberation_steps),
        "latent_deliberation_slots": int(args.latent_deliberation_slots),
        "moe_routed_experts": int(args.moe_routed_experts),
        "moe_top_k": int(args.moe_top_k),
        "moe_expert_ff_size": int(args.moe_expert_ff_size),
        "device": str(device),
        "gpu_name": gpu_name,
        "required_gpu_name": required_gpu_name,
        "torch_version": str(torch.__version__),
        "source_provenance": _source_provenance(),
        "precision": "bfloat16_autocast_fp32_loss",
        "batch_size": int(args.batch_size),
        "legal_actions": int(args.legal_actions),
        "events": int(args.events),
        "warmup_iterations": int(args.warmup),
        "measured_iterations": int(args.iterations),
        "elapsed_sec": elapsed,
        "steps_per_sec": int(args.iterations) / elapsed,
        "rows_per_sec": int(args.iterations) * int(args.batch_size) / elapsed,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "final_loss": losses[-1],
        "gradients_finite": True,
        "finite": True,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        from pathlib import Path

        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
