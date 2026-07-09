#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from catan_zero.rl.ppo_policy_factory import load_ppo_policy
from catan_zero.rl.torch_ppo import (
    _entity_action_column,
    _entity_graph_outputs,
)


def _iter_shard_paths(root: Path, *, include_consumed: bool) -> list[Path]:
    paths = sorted((root / "trajectories").glob("*/*.pkl"))
    if include_consumed:
        # ``consumed`` files are normally empty markers. If a future learner archives full
        # consumed shards there, include only non-empty files; current marker-only runs are skipped.
        paths.extend(path for path in sorted((root / "consumed").glob("*.pkl")) if path.stat().st_size > 0)
    unique: dict[str, Path] = {}
    for path in paths:
        unique[str(path.resolve())] = path
    return list(unique.values())


def _load_trajectories(path: Path) -> tuple[int | None, list[Any]]:
    with path.open("rb") as fh:
        obj = pickle.load(fh)
    if isinstance(obj, dict) and "trajectories" in obj:
        return int(obj.get("policy_version", -1)), list(obj["trajectories"])
    if isinstance(obj, list):
        return None, obj
    raise TypeError(f"unsupported shard format in {path}: {type(obj)!r}")


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or len(y) < 2:
        return None
    if float(np.std(x)) <= 1e-8 or float(np.std(y)) <= 1e-8:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _summ(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "std": None, "min": None, "p10": None, "p50": None, "p90": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p10": float(np.quantile(arr, 0.10)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "max": float(np.max(arr)),
    }


def _phase_table(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_phase[str(row.get("phase") or "unknown")].append(row)
    out: dict[str, Any] = {}
    for phase, phase_rows in sorted(by_phase.items(), key=lambda item: (-len(item[1]), item[0])):
        adv = [float(r["advantage"]) for r in phase_rows]
        ret = [float(r["return"]) for r in phase_rows]
        oldv = [float(r["old_value"]) for r in phase_rows]
        delta = [float(r["ppo_logprob"] - r["seed_logprob"]) for r in phase_rows if r.get("ppo_logprob") is not None]
        bad = [
            r for r in phase_rows
            if r.get("ppo_logprob") is not None
            and ((r["advantage"] > 0 and (r["ppo_logprob"] - r["seed_logprob"]) < 0)
                 or (r["advantage"] < 0 and (r["ppo_logprob"] - r["seed_logprob"]) > 0))
        ]
        out[phase] = {
            "rows": len(phase_rows),
            "advantage": _summ(adv),
            "return": _summ(ret),
            "old_value": _summ(oldv),
            "chosen_logprob_delta_ppo_minus_seed": _summ(delta),
            "bad_direction_fraction": float(len(bad) / len(delta)) if delta else None,
            "positive_advantage_fraction": float(sum(1 for x in adv if x > 0) / len(adv)) if adv else None,
            "negative_advantage_fraction": float(sum(1 for x in adv if x < 0) / len(adv)) if adv else None,
        }
    return out


def _group_table(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[str(row.get(key) or "unknown")].append(row)
    out: dict[str, Any] = {}
    for name, group_rows in sorted(by_key.items(), key=lambda item: (-len(item[1]), item[0])):
        adv = [float(r["advantage"]) for r in group_rows]
        ret = [float(r["return"]) for r in group_rows]
        oldv = [float(r["old_value"]) for r in group_rows]
        phases = Counter(str(r.get("phase") or "unknown") for r in group_rows)
        out[name] = {
            "rows": len(group_rows),
            "trajectories": len({int(r["trajectory_index"]) for r in group_rows}),
            "phase_counts": dict(phases),
            "advantage": _summ(adv),
            "return": _summ(ret),
            "old_value": _summ(oldv),
            "return_old_value_corr": _safe_corr(
                np.asarray(ret, dtype=np.float64),
                np.asarray(oldv, dtype=np.float64),
            ),
            "positive_advantage_fraction": float(sum(1 for x in adv if x > 0) / len(adv)) if adv else None,
            "negative_advantage_fraction": float(sum(1 for x in adv if x < 0) / len(adv)) if adv else None,
        }
    return out


def _opponent_summary(traj: Any) -> tuple[str, dict[str, str]]:
    names = getattr(traj, "opponent_names", None) or {}
    if not isinstance(names, dict):
        names = {}
    clean = {str(seat): str(name) for seat, name in names.items()}
    if not clean:
        return "unknown", clean
    mix = ",".join(sorted(set(clean.values())))
    return mix or "unknown", clean


def _policy_metrics(
    policy,
    samples: list[Any],
    *,
    batch_size: int,
    behavior_temperature: float = 1.0,
) -> tuple[list[float], list[float], list[int]]:
    import torch

    logps: list[float] = []
    values: list[float] = []
    argmax_actions: list[int] = []
    for start in range(0, len(samples), batch_size):
        batch_samples = samples[start : start + batch_size]
        with torch.no_grad():
            outputs = _entity_graph_outputs(policy, batch_samples, return_q=False)
            logits = outputs["logits"]
            value = outputs["value"]
            behavior_temperature = max(float(behavior_temperature), 1.0e-6)
            behavior_logits = logits
            if behavior_temperature != 1.0:
                behavior_logits = torch.clamp(
                    logits / behavior_temperature,
                    min=-50.0,
                    max=50.0,
                )
            log_probs = torch.log_softmax(behavior_logits, dim=-1)
            action_cols = torch.as_tensor(
                [_entity_action_column(sample) for sample in batch_samples],
                dtype=torch.long,
                device=policy.device,
            )
            chosen = log_probs.gather(1, action_cols.unsqueeze(1)).squeeze(1)
            argmax_cols = torch.argmax(logits, dim=-1).detach().cpu().numpy().tolist()
        logps.extend(float(x) for x in chosen.detach().cpu().numpy())
        values.extend(float(x) for x in value.detach().cpu().numpy())
        for sample, col in zip(batch_samples, argmax_cols):
            valid = tuple(int(x) for x in sample.valid_actions)
            argmax_actions.append(int(valid[int(col)]) if 0 <= int(col) < len(valid) else -1)
    return logps, values, argmax_actions


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit PPO advantages and policy deltas on entity_graph shards.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--seed-checkpoint", required=True)
    parser.add_argument("--ppo-checkpoint", default="")
    parser.add_argument("--architecture", default="entity_graph")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-shards", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--behavior-temperature",
        type=float,
        default=1.0,
        help=(
            "Temperature used when recomputing seed/PPO log-probs. Set to the actor "
            "sampling temperature when auditing temp-controlled PPO shards."
        ),
    )
    parser.add_argument("--include-consumed", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.run_root)
    shard_paths = _iter_shard_paths(root, include_consumed=bool(args.include_consumed))
    if args.max_shards > 0:
        shard_paths = shard_paths[: args.max_shards]

    rows: list[dict[str, Any]] = []
    samples: list[Any] = []
    shard_count = 0
    trajectory_count = 0
    policy_versions = Counter()
    invalid_actions = 0
    truncated_trajectories = 0
    for shard_path in shard_paths:
        version, trajectories = _load_trajectories(shard_path)
        shard_count += 1
        if version is not None:
            policy_versions[int(version)] += 1
        for traj in trajectories:
            opponent_mix, opponent_names = _opponent_summary(traj)
            trajectory_count += 1
            current_trajectory_index = trajectory_count
            if bool(getattr(traj, "truncated", False)):
                truncated_trajectories += 1
            n = len(traj.samples)
            lengths = [len(getattr(traj, name, [])) for name in ("returns", "advantages", "old_log_probs", "old_values")]
            if any(length != n for length in lengths):
                raise ValueError(f"misaligned trajectory lengths in {shard_path}: samples={n}, fields={lengths}")
            for idx, sample in enumerate(traj.samples):
                if int(sample.action) not in tuple(int(x) for x in sample.valid_actions):
                    invalid_actions += 1
                samples.append(sample)
                rows.append(
                    {
                        "phase": str(sample.phase or "unknown"),
                        "opponent_mix": opponent_mix,
                        "opponent_names": opponent_names,
                        "trajectory_index": int(current_trajectory_index),
                        "player": str(sample.player),
                        "action": int(sample.action),
                        "legal_count": int(len(sample.valid_actions)),
                        "return": float(traj.returns[idx]),
                        "advantage": float(traj.advantages[idx]),
                        "old_logprob": float(traj.old_log_probs[idx]),
                        "old_value": float(traj.old_values[idx]),
                        "reward": float(traj.rewards[idx]) if getattr(traj, "rewards", None) and idx < len(traj.rewards) else None,
                    }
                )
                if 0 < args.max_samples <= len(rows):
                    break
            if 0 < args.max_samples <= len(rows):
                break
        if 0 < args.max_samples <= len(rows):
            break

    if not rows:
        raise RuntimeError(f"no samples found under {root}")

    seed_policy = load_ppo_policy(args.seed_checkpoint, architecture=args.architecture, device=args.device)
    seed_logps, seed_values, seed_argmax = _policy_metrics(
        seed_policy,
        samples,
        batch_size=args.batch_size,
        behavior_temperature=args.behavior_temperature,
    )
    ppo_logps: list[float] | None = None
    ppo_values: list[float] | None = None
    ppo_argmax: list[int] | None = None
    if args.ppo_checkpoint:
        ppo_policy = load_ppo_policy(args.ppo_checkpoint, architecture=args.architecture, device=args.device)
        ppo_logps, ppo_values, ppo_argmax = _policy_metrics(
            ppo_policy,
            samples,
            batch_size=args.batch_size,
            behavior_temperature=args.behavior_temperature,
        )

    for i, row in enumerate(rows):
        row["seed_logprob"] = float(seed_logps[i])
        row["seed_value"] = float(seed_values[i])
        row["seed_argmax_action"] = int(seed_argmax[i])
        row["seed_chose_same_as_actor"] = bool(seed_argmax[i] == row["action"])
        if ppo_logps is not None and ppo_values is not None and ppo_argmax is not None:
            row["ppo_logprob"] = float(ppo_logps[i])
            row["ppo_value"] = float(ppo_values[i])
            row["ppo_argmax_action"] = int(ppo_argmax[i])
            row["ppo_chose_same_as_actor"] = bool(ppo_argmax[i] == row["action"])

    adv = np.asarray([r["advantage"] for r in rows], dtype=np.float64)
    returns = np.asarray([r["return"] for r in rows], dtype=np.float64)
    old_values = np.asarray([r["old_value"] for r in rows], dtype=np.float64)
    seed_values_arr = np.asarray([r["seed_value"] for r in rows], dtype=np.float64)
    seed_logp_arr = np.asarray([r["seed_logprob"] for r in rows], dtype=np.float64)
    old_logp_arr = np.asarray([r["old_logprob"] for r in rows], dtype=np.float64)
    report: dict[str, Any] = {
        "run_root": str(root),
        "seed_checkpoint": args.seed_checkpoint,
        "ppo_checkpoint": args.ppo_checkpoint or None,
        "behavior_temperature": float(args.behavior_temperature),
        "shards_read": shard_count,
        "trajectories_read": trajectory_count,
        "samples": len(rows),
        "policy_versions": dict(policy_versions),
        "invalid_actions": invalid_actions,
        "truncated_trajectories": truncated_trajectories,
        "phase_counts": dict(Counter(r["phase"] for r in rows)),
        "opponent_mix_counts": dict(Counter(r["opponent_mix"] for r in rows)),
        "legal_count": _summ([r["legal_count"] for r in rows]),
        "advantage": _summ(adv.tolist()),
        "return": _summ(returns.tolist()),
        "old_value": _summ(old_values.tolist()),
        "seed_value": _summ(seed_values_arr.tolist()),
        "seed_value_minus_old_value": _summ((seed_values_arr - old_values).tolist()),
        "seed_logprob_minus_old_logprob": _summ((seed_logp_arr - old_logp_arr).tolist()),
        "return_old_value_corr": _safe_corr(returns, old_values),
        "return_seed_value_corr": _safe_corr(returns, seed_values_arr),
        "advantage_old_value_corr": _safe_corr(adv, old_values),
        "seed_argmax_matches_actor_fraction": float(sum(1 for r in rows if r["seed_chose_same_as_actor"]) / len(rows)),
        "by_phase": _phase_table(rows),
        "by_opponent_mix": _group_table(rows, "opponent_mix"),
    }
    if ppo_logps is not None and ppo_values is not None:
        ppo_logp_arr = np.asarray(ppo_logps, dtype=np.float64)
        ppo_values_arr = np.asarray(ppo_values, dtype=np.float64)
        delta = ppo_logp_arr - seed_logp_arr
        bad = ((adv > 0) & (delta < 0)) | ((adv < 0) & (delta > 0))
        good = ((adv > 0) & (delta > 0)) | ((adv < 0) & (delta < 0))
        report.update(
            {
                "ppo_value": _summ(ppo_values_arr.tolist()),
                "ppo_value_minus_seed_value": _summ((ppo_values_arr - seed_values_arr).tolist()),
                "ppo_logprob_minus_seed_logprob": _summ(delta.tolist()),
                "ppo_argmax_matches_actor_fraction": float(sum(1 for r in rows if r.get("ppo_chose_same_as_actor")) / len(rows)),
                "ppo_argmax_differs_from_seed_fraction": float(sum(1 for r in rows if r.get("ppo_argmax_action") != r.get("seed_argmax_action")) / len(rows)),
                "ppo_logprob_delta_advantage_corr": _safe_corr(delta, adv),
                "ppo_bad_direction_fraction": float(np.mean(bad)),
                "ppo_good_direction_fraction": float(np.mean(good)),
                "ppo_increased_negative_advantage_fraction": float(np.mean((adv < 0) & (delta > 0))),
                "ppo_decreased_positive_advantage_fraction": float(np.mean((adv > 0) & (delta < 0))),
            }
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
