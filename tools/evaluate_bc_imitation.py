from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

from catan_zero.rl.multiagent_env import ColonistMultiAgentEnv
from catan_zero.rl.torch_ppo import TorchPPOPolicy
from catan_zero.rl.xdim_lite_policy import XDimLitePolicy, masked_logits, normalize_observations
from factory_common import parse_track, write_json
from train_bc import (
    _dense_context,
    _target_columns,
    _torch_ppo_masked_logits,
    _valid_lists,
    load_teacher_data,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BC imitation accuracy on teacher shards.")
    parser.add_argument("--arch", choices=("candidate", "xdim_lite", "xdim_graph"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--track", default="2p_no_trade")
    parser.add_argument("--vps-to-win", type=int, default=10)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import torch
    from torch import nn

    data = load_teacher_data(Path(args.data))
    if args.arch == "candidate":
        policy = TorchPPOPolicy.load(args.checkpoint, device=args.device)
    else:
        policy = XDimLitePolicy.load(args.checkpoint, device=args.device)

    losses: list[float] = []
    accuracies: list[float] = []
    n = len(data["action_taken"])
    legal_counts = np.sum(data["legal_action_ids"] >= 0, axis=1)
    forced = legal_counts == 1
    teachers = _string_column(data, "teacher_name", n)
    phases = _string_column(data, "phase", n)
    env = ColonistMultiAgentEnv(parse_track(args.track, vps_to_win=args.vps_to_win))
    env.reset(seed=0)
    action_types = np.asarray(
        [_action_type(env, int(action)) for action in data["action_taken"]],
        dtype=object,
    )
    correct_all = np.zeros(n, dtype=bool)
    top3_all = np.zeros(n, dtype=bool)
    for start in range(0, n, args.batch_size):
        batch = np.arange(start, min(n, start + args.batch_size))
        if args.arch == "candidate":
            obs = torch.as_tensor(
                normalize_observations(data["obs"][batch]),
                dtype=torch.float32,
                device=policy.device,
            )
            context = _dense_context(
                data,
                batch,
                policy.action_size,
                policy.context_action_feature_size,
            )
            context_t = torch.as_tensor(context, dtype=torch.float32, device=policy.device)
            logits, _ = policy.forward(obs, context_t)
            valid = _valid_lists(data["legal_action_ids"][batch])
            masked = _torch_ppo_masked_logits(logits, valid, policy.action_size)
        else:
            legal_action_ids = data["legal_action_ids"][batch]
            outputs = policy.forward_legal_np(
                data["obs"][batch],
                legal_action_ids,
                data["legal_action_context"][batch],
            )
            masked = outputs["logits"]
        target_np = (
            data["action_taken"][batch].astype(np.int64)
            if args.arch == "candidate"
            else _target_columns(data["legal_action_ids"][batch], data["action_taken"][batch].astype(np.int64))
        )
        actions = torch.as_tensor(target_np, dtype=torch.long, device=policy.device)
        loss = nn.functional.cross_entropy(masked, actions)
        predictions = torch.argmax(masked, dim=-1)
        correct = predictions == actions
        k = min(3, masked.shape[-1])
        top3 = torch.topk(masked, k=k, dim=-1).indices == actions[:, None]
        correct_all[batch] = correct.detach().cpu().numpy().astype(bool)
        top3_all[batch] = torch.any(top3, dim=-1).detach().cpu().numpy().astype(bool)
        accuracy = correct.float().mean()
        losses.append(float(loss.item()) * len(batch))
        accuracies.append(float(accuracy.item()) * len(batch))

    report = {
        "arch": args.arch,
        "checkpoint": args.checkpoint,
        "data": args.data,
        "samples": int(n),
        "loss": float(sum(losses) / max(n, 1)),
        "accuracy": float(sum(accuracies) / max(n, 1)),
        "top3_accuracy": float(np.mean(top3_all)) if n else 0.0,
        "forced_actions": int(np.sum(forced)),
        "forced_action_fraction": float(np.mean(forced)) if n else 0.0,
        "accuracy_excluding_forced": _masked_mean(correct_all, ~forced),
        "top3_accuracy_excluding_forced": _masked_mean(top3_all, ~forced),
        "legal_actions": {
            "mean": float(np.mean(legal_counts)) if n else 0.0,
            "p50": int(np.percentile(legal_counts, 50)) if n else 0,
            "p90": int(np.percentile(legal_counts, 90)) if n else 0,
            "p99": int(np.percentile(legal_counts, 99)) if n else 0,
            "max": int(np.max(legal_counts)) if n else 0,
        },
        "by_teacher": _breakdown(teachers, correct_all, top3_all, forced),
        "by_phase": _breakdown(phases, correct_all, top3_all, forced),
        "by_action_type": _breakdown(action_types, correct_all, top3_all, forced),
    }
    write_json(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True))


def _action_type(env: ColonistMultiAgentEnv, action: int) -> str:
    description = env.describe_action(action)
    return str((description or {}).get("action_type", "unknown"))


def _string_column(data: dict, key: str, n: int) -> np.ndarray:
    if key not in data:
        return np.asarray([""] * n, dtype=object)
    return np.asarray(data[key]).astype(str)


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    return float(np.mean(values[mask]))


def _breakdown(
    groups: np.ndarray,
    correct: np.ndarray,
    top3: np.ndarray,
    forced: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    buckets: defaultdict[str, list[int]] = defaultdict(list)
    for index, value in enumerate(groups.tolist()):
        key = str(value) if str(value) else "unknown"
        buckets[key].append(index)
    result = {}
    for key, indices in sorted(buckets.items(), key=lambda item: (-len(item[1]), item[0])):
        idx = np.asarray(indices, dtype=np.int64)
        unforced = ~forced[idx]
        result[key] = {
            "samples": int(len(idx)),
            "accuracy": float(np.mean(correct[idx])) if len(idx) else 0.0,
            "top3_accuracy": float(np.mean(top3[idx])) if len(idx) else 0.0,
            "forced_action_fraction": float(np.mean(forced[idx])) if len(idx) else 0.0,
            "accuracy_excluding_forced": _masked_mean(correct[idx], unforced),
            "top3_accuracy_excluding_forced": _masked_mean(top3[idx], unforced),
        }
    return result


if __name__ == "__main__":
    main()
