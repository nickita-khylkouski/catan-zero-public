#!/usr/bin/env python3
"""H100 systems probe using collated features from actual Catan env states.

This measures finite BF16 forward/backward execution, throughput, and memory.
It is deliberately not a playing-strength or sample-efficiency benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
for _path in (_ROOT, _SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--require-gpu-name", default="NVIDIA H100")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--legal-actions", type=int, default=64)
    parser.add_argument("--events", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--hidden-size", type=int, default=640)
    parser.add_argument("--state-layers", type=int, default=6)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--topology-adapter-layers", default="2,4")
    parser.add_argument("--topology-adapter-width", type=int, default=192)
    parser.add_argument("--topology-adapter-bases", type=int, default=4)
    parser.add_argument(
        "--topology-adapter-kind",
        choices=("basis_mean_v1", "local_attention_v2"),
        default="local_attention_v2",
    )
    parser.add_argument("--topology-adapter-heads", type=int, default=4)
    parser.add_argument("--topology-adapter-share-weights", action="store_true")
    parser.add_argument(
        "--topology-adapter-edge-control",
        choices=("true_topology", "self_message", "type_degree_preserving_rewire"),
        default="true_topology",
    )
    parser.add_argument("--output", default="")
    return parser


def _pad_rows(
    value: np.ndarray,
    width: int,
    *,
    fill: int | float | bool = 0,
    take_last: bool = False,
) -> np.ndarray:
    """Return a fixed-width first-axis array without fabricating live rows."""

    source = np.asarray(value)
    result = np.full((width, *source.shape[1:]), fill, dtype=source.dtype)
    count = min(width, int(source.shape[0]))
    if count:
        selected = source[-count:] if take_last else source[:count]
        result[:count] = selected
    return result


def build_collated_public_batch(
    *,
    batch_size: int,
    legal_actions: int,
    events: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Collate deterministic acting-player views from real public env states.

    Opponent private ground truth is never read. Legal tokens, target IDs, and
    contexts all retain the environment's structured-legal-action ordering.
    """

    if min(batch_size, legal_actions, events) < 1:
        raise ValueError("batch-size, legal-actions, and events must be positive")

    import torch

    from catan_zero.rl.action_features import build_action_context_feature_table
    from catan_zero.rl.entity_token_features import build_entity_token_features
    from catan_zero.rl.multiagent_env import ColonistMultiAgentEnv
    from catan_zero.rl.self_play import make_env_config

    env_config = make_env_config(vps_to_win=3)
    env = ColonistMultiAgentEnv(env_config)
    rows: list[dict[str, np.ndarray]] = []
    legal_id_rows: list[np.ndarray] = []
    state_records: list[dict[str, Any]] = []
    resets = 0
    try:
        observations, info = env.reset(seed=seed)
        del observations
        for decision_index in range(batch_size):
            actor = str(info["current_player"])
            valid_ids = tuple(int(action) for action in info["valid_actions"])
            if not valid_ids:
                raise RuntimeError("public env state unexpectedly has no legal actions")
            features = build_entity_token_features(
                env,
                actor,
                include_event_log=True,
                history_limit=events,
            )
            structured_ids = tuple(
                int(action["index"])
                for action in info["structured_legal_actions"]
            )
            if structured_ids != valid_ids:
                raise RuntimeError(
                    "structured legal actions do not align with valid action IDs"
                )
            count = min(legal_actions, len(valid_ids))
            selected_ids = valid_ids[:count]
            context_table = build_action_context_feature_table(env, info)
            row = {
                key: np.asarray(value)
                for key, value in features.items()
                if key != "schema"
            }
            row["legal_action_tokens"] = _pad_rows(
                row["legal_action_tokens"][:count], legal_actions
            ).astype(np.float32)
            row["legal_action_target_ids"] = _pad_rows(
                row["legal_action_target_ids"][:count], legal_actions, fill=-1
            ).astype(np.int64)
            row["legal_action_mask"] = np.zeros(legal_actions, dtype=np.bool_)
            row["legal_action_mask"][:count] = True
            row["legal_action_context"] = _pad_rows(
                context_table[list(selected_ids)], legal_actions
            ).astype(np.float32)
            # build_entity_token_features(history_limit=events) already keeps the
            # latest events and emits exactly this width. Keep the normalization
            # explicit so a future feature implementation cannot break collation.
            row["event_tokens"] = _pad_rows(
                row["event_tokens"], events, take_last=True
            ).astype(np.float32)
            row["event_target_ids"] = _pad_rows(
                row["event_target_ids"], events, fill=-1, take_last=True
            ).astype(np.int64)
            row["event_mask"] = _pad_rows(
                row["event_mask"], events, fill=False, take_last=True
            ).astype(np.bool_)
            legal_ids = np.full(legal_actions, -1, dtype=np.int64)
            legal_ids[:count] = selected_ids
            row["legal_action_ids"] = legal_ids
            rows.append(row)
            legal_id_rows.append(legal_ids)
            state_records.append(
                {
                    "decision_index": decision_index,
                    "actor": actor,
                    "legal_count_before_width_limit": len(valid_ids),
                    "legal_count_collated": count,
                    "legal_action_ids_sha256": hashlib.sha256(
                        np.asarray(selected_ids, dtype=np.int64).tobytes()
                    ).hexdigest(),
                    "truncated_legal_actions": max(0, len(valid_ids) - count),
                    "live_events": int(row["event_mask"].sum()),
                }
            )

            chosen = valid_ids[(decision_index * 17 + seed) % len(valid_ids)]
            observations, _rewards, terminated, truncated, info = env.step(chosen)
            del observations
            if terminated or truncated:
                resets += 1
                observations, info = env.reset(seed=seed + resets)
                del observations
    finally:
        action_size = int(env.action_space.n)
        env.close()

    keys = tuple(rows[0])
    if any(tuple(row) != keys for row in rows[1:]):
        raise RuntimeError("entity feature keys changed during collation")
    batch = {
        key: torch.as_tensor(np.stack([row[key] for row in rows], axis=0))
        for key in keys
    }
    action_mask = batch["legal_action_mask"]
    event_mask = batch["event_mask"]
    id_bytes = np.stack(legal_id_rows, axis=0).tobytes()
    provenance = {
        "generator": "ColonistMultiAgentEnv + build_entity_token_features",
        "seed": int(seed),
        "batch_size": int(batch_size),
        "requested_legal_width": int(legal_actions),
        "requested_event_width": int(events),
        "action_space_size": action_size,
        "episode_resets": resets,
        "state_records": state_records,
        "legal_action_ids_sha256": hashlib.sha256(id_bytes).hexdigest(),
        "mask_utilization": {
            "legal_actions": {
                "live": int(action_mask.sum()),
                "capacity": int(action_mask.numel()),
                "fraction": float(action_mask.float().mean()),
            },
            "events": {
                "live": int(event_mask.sum()),
                "capacity": int(event_mask.numel()),
                "fraction": float(event_mask.float().mean()),
            },
            "players": {
                "live": int(batch["player_mask"].sum()),
                "capacity": int(batch["player_mask"].numel()),
                "fraction": float(batch["player_mask"].float().mean()),
            },
        },
    }
    return batch, provenance


def _source_provenance() -> dict[str, Any]:
    paths = (
        "tools/rnd_topology_collated_probe.py",
        "src/catan_zero/rl/action_features.py",
        "src/catan_zero/rl/entity_token_features.py",
        "src/catan_zero/rl/entity_token_policy.py",
        "src/catan_zero/rl/sparse_topology_adapter.py",
    )
    hashes: dict[str, str] = {}
    aggregate = hashlib.sha256()
    for relative in paths:
        content = (_ROOT / relative).read_bytes()
        hashes[relative] = hashlib.sha256(content).hexdigest()
        aggregate.update(relative.encode() + b"\0" + content + b"\0")
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "git_commit": commit,
        "source_bundle_sha256": aggregate.hexdigest(),
        "file_sha256": hashes,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from catan_zero.rl.action_features import CONTEXT_ACTION_FEATURE_SIZE
    from catan_zero.rl.entity_token_features import LEGAL_ACTION_FEATURE_SIZE
    from catan_zero.rl.entity_token_policy import EntityGraphConfig, EntityGraphNet

    for name in (
        "batch_size",
        "legal_actions",
        "events",
        "hidden_size",
        "state_layers",
        "attention_heads",
        "topology_adapter_width",
        "topology_adapter_bases",
        "topology_adapter_heads",
        "iterations",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if int(args.warmup) < 0:
        raise ValueError("warmup must be non-negative")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the collated systems probe requires an available CUDA GPU")
    gpu_name = torch.cuda.get_device_name(device)
    if args.require_gpu_name and args.require_gpu_name not in gpu_name:
        raise RuntimeError(
            f"GPU identity mismatch: required {args.require_gpu_name!r}, got {gpu_name!r}"
        )

    cpu_batch, batch_provenance = build_collated_public_batch(
        batch_size=int(args.batch_size),
        legal_actions=int(args.legal_actions),
        events=int(args.events),
        seed=int(args.seed),
    )
    # All feature construction and collation happens on CPU. Device transfer is
    # completed once here, outside both warmup and timed loops.
    batch = {key: value.to(device) for key, value in cpu_batch.items()}
    config = EntityGraphConfig(
        action_size=int(batch_provenance["action_space_size"]),
        static_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
        context_action_feature_size=CONTEXT_ACTION_FEATURE_SIZE,
        legal_action_feature_size=LEGAL_ACTION_FEATURE_SIZE,
        hidden_size=int(args.hidden_size),
        state_layers=int(args.state_layers),
        attention_heads=int(args.attention_heads),
        dropout=0.05,
        state_trunk="transformer",
        topology_adapter_layers=str(args.topology_adapter_layers),
        topology_adapter_width=int(args.topology_adapter_width),
        topology_adapter_bases=int(args.topology_adapter_bases),
        topology_adapter_kind=str(args.topology_adapter_kind),
        topology_adapter_heads=int(args.topology_adapter_heads),
        topology_adapter_share_weights=bool(args.topology_adapter_share_weights),
        topology_adapter_edge_control=str(args.topology_adapter_edge_control),
    )
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    model = EntityGraphNet(config).to(device).train()
    legal_mask = batch["legal_action_mask"].bool()

    def step() -> float:
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(batch, return_q=True)
            loss = (
                outputs["logits"][legal_mask].float().square().mean()
                + outputs["q_values"][legal_mask].float().square().mean()
                + outputs["value"].float().square().mean()
            )
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite collated-probe loss")
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        if not gradients or any(not torch.isfinite(grad).all() for grad in gradients):
            raise RuntimeError("missing or non-finite collated-probe gradients")
        return float(loss.detach())

    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(int(args.warmup)):
        step()
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    losses = [step() for _ in range(int(args.iterations))]
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    properties = torch.cuda.get_device_properties(device)
    return {
        "schema_version": "catan-zero-rnd-topology-collated-probe/v1",
        "scope": "systems-only; no playing-strength claim",
        "architecture": {
            "hidden_size": int(args.hidden_size),
            "state_layers": int(args.state_layers),
            "attention_heads": int(args.attention_heads),
            "topology_adapter_layers": str(args.topology_adapter_layers),
            "topology_adapter_width": int(args.topology_adapter_width),
            "topology_adapter_bases": int(args.topology_adapter_bases),
            "topology_adapter_kind": str(args.topology_adapter_kind),
            "topology_adapter_heads": int(args.topology_adapter_heads),
            "topology_adapter_share_weights": bool(
                args.topology_adapter_share_weights
            ),
            "topology_adapter_edge_control": str(
                args.topology_adapter_edge_control
            ),
            "parameter_count": sum(p.numel() for p in model.parameters()),
        },
        "batch_provenance": batch_provenance,
        "device_transfer": "once before warmup; no host transfer in timed loop",
        "precision": "bfloat16 autocast with fp32 loss",
        "warmup_iterations": int(args.warmup),
        "measured_iterations": int(args.iterations),
        "elapsed_sec": elapsed,
        "steps_per_sec": int(args.iterations) / elapsed,
        "rows_per_sec": int(args.iterations) * int(args.batch_size) / elapsed,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "final_loss": losses[-1],
        "finite": True,
        "gradients_finite": True,
        "hardware": {
            "platform": platform.platform(),
            "device": str(device),
            "gpu_name": gpu_name,
            "gpu_total_memory_bytes": int(properties.total_memory),
            "gpu_compute_capability": [int(properties.major), int(properties.minor)],
            "torch_version": str(torch.__version__),
            "torch_cuda_version": str(torch.version.cuda),
        },
        "source_provenance": _source_provenance(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run(args)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
