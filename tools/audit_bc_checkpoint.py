from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from catan_zero.rl.entity_token_policy import EntityGraphPolicy
from catan_zero.rl.multiagent_env import ColonistMultiAgentEnv
from factory_common import parse_track, write_json
from train_bc import (
    _concat_padded,
    _entity_batch,
    _load_npz,
    _normalize_teacher_shard,
    _target_columns,
    _teacher_shard_files,
    _value_targets,
    teacher_data_quality,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit entity_graph BC checkpoints on the same sampled teacher rows. "
            "Reports imitation accuracy, phase/teacher/action breakdowns, and value calibration."
        )
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Checkpoint spec. Use name=path or just path. Repeat for multiple checkpoints.",
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-shards", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=250000)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--track", default="2p_no_trade")
    parser.add_argument("--vps-to-win", type=int, default=10)
    args = parser.parse_args()

    data_path = Path(args.data)
    data, selected_files = _load_sampled_data(
        data_path,
        max_shards=args.max_shards,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    n = int(len(data["action_taken"]))
    if n == 0:
        raise SystemExit("sampled zero rows")

    legal_counts = np.sum(data["legal_action_ids"] >= 0, axis=1)
    forced = legal_counts == 1
    targets = _target_columns(
        data["legal_action_ids"],
        data["action_taken"].astype(np.int64, copy=False),
    )
    teachers = _string_column(data, "teacher_name", n)
    phases = _string_column(data, "phase", n)
    action_types = _action_types(args.track, args.vps_to_win, data["action_taken"])

    quality = teacher_data_quality(data)
    report: dict[str, object] = {
        "data": str(data_path),
        "sampled_rows": n,
        "selected_shards": len(selected_files),
        "selected_shard_preview": [str(path) for path in selected_files[:5]],
        "seed": int(args.seed),
        "legal_actions": {
            "mean": float(np.mean(legal_counts)),
            "p50": int(np.percentile(legal_counts, 50)),
            "p90": int(np.percentile(legal_counts, 90)),
            "p99": int(np.percentile(legal_counts, 99)),
            "max": int(np.max(legal_counts)),
        },
        "forced_action_rows": int(np.sum(forced)),
        "forced_action_fraction": float(np.mean(forced)),
        "teacher_data_quality": quality,
        "checkpoints": {},
    }

    prediction_cache: dict[str, dict[str, np.ndarray]] = {}
    for spec in args.checkpoint:
        name, checkpoint = _parse_checkpoint_spec(spec)
        result = _audit_checkpoint(
            name=name,
            checkpoint=Path(checkpoint),
            data=data,
            targets=targets,
            teachers=teachers,
            phases=phases,
            action_types=action_types,
            forced=forced,
            batch_size=args.batch_size,
            device=args.device,
            vps_to_win=args.vps_to_win,
        )
        report["checkpoints"][name] = result["report"]
        prediction_cache[name] = {
            "correct": result["correct"],
            "top3": result["top3"],
            "predictions": result["predictions"],
        }

    if len(prediction_cache) >= 2:
        names = list(prediction_cache)
        base = names[0]
        comparisons = {}
        for other in names[1:]:
            comparisons[f"{other}_vs_{base}"] = _compare_predictions(
                base=prediction_cache[base],
                other=prediction_cache[other],
                phases=phases,
                teachers=teachers,
                forced=forced,
            )
        report["pairwise"] = comparisons

    write_json(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True))


def _load_sampled_data(
    path: Path,
    *,
    max_shards: int,
    max_samples: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], list[Path]]:
    files = _teacher_shard_files(path)
    if max_shards > 0 and len(files) > max_shards:
        positions = np.linspace(0, len(files) - 1, num=max_shards, dtype=np.int64)
        selected = [files[int(pos)] for pos in positions]
    else:
        selected = files

    arrays: dict[str, list[np.ndarray]] = {}
    rows = 0
    for file in selected:
        shard = _normalize_teacher_shard(_load_npz(file), file)
        for key, value in shard.items():
            arrays.setdefault(key, []).append(value)
        rows += int(len(shard["action_taken"]))
        if max_samples > 0 and rows >= max_samples * 2:
            # Enough loaded for a representative random sample without reading the whole corpus.
            break
    data = {key: _concat_padded(key, values) for key, values in arrays.items()}
    n = int(len(data["action_taken"]))
    if max_samples > 0 and n > max_samples:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(n, size=max_samples, replace=False))
        data = {
            key: value[indices]
            for key, value in data.items()
            if hasattr(value, "shape") and value.shape[0] == n
        }
    return data, selected


def _audit_checkpoint(
    *,
    name: str,
    checkpoint: Path,
    data: dict[str, np.ndarray],
    targets: np.ndarray,
    teachers: np.ndarray,
    phases: np.ndarray,
    action_types: np.ndarray,
    forced: np.ndarray,
    batch_size: int,
    device: str,
    vps_to_win: int,
) -> dict[str, object]:
    import torch
    from torch.nn import functional as F

    policy = EntityGraphPolicy.load(checkpoint, device=device)
    n = int(len(data["action_taken"]))
    correct_all = np.zeros(n, dtype=bool)
    top3_all = np.zeros(n, dtype=bool)
    pred_all = np.full(n, -1, dtype=np.int64)
    value_pred = np.full(n, np.nan, dtype=np.float32)
    value_target = np.full(n, np.nan, dtype=np.float32)
    has_outcome = np.zeros(n, dtype=bool)
    losses: list[float] = []
    entropies: list[float] = []

    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = np.arange(start, min(n, start + batch_size), dtype=np.int64)
            outputs = policy.forward_legal_np(
                _entity_batch(data, batch),
                data["legal_action_ids"][batch],
                data["legal_action_context"][batch],
                return_q=False,
            )
            logits = outputs["logits"]
            target_t = torch.as_tensor(targets[batch], dtype=torch.long, device=policy.device)
            loss = F.cross_entropy(logits, target_t)
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs.clamp_min(1.0e-12))).sum(dim=-1).mean()
            predictions = torch.argmax(logits, dim=-1)
            k = min(3, logits.shape[-1])
            top3 = torch.topk(logits, k=k, dim=-1).indices == target_t[:, None]

            correct = predictions == target_t
            correct_all[batch] = correct.detach().cpu().numpy().astype(bool)
            top3_all[batch] = torch.any(top3, dim=-1).detach().cpu().numpy().astype(bool)
            pred_all[batch] = predictions.detach().cpu().numpy().astype(np.int64)
            losses.append(float(loss.item()) * len(batch))
            entropies.append(float(entropy.item()) * len(batch))

            outcome_t, _vp_t, has_outcome_t, _has_vp_t = _value_targets(
                data,
                batch,
                policy.device,
                vps_to_win,
            )
            if outcome_t is not None and has_outcome_t is not None:
                values = outputs["value"].reshape(-1)
                mask_np = has_outcome_t.detach().cpu().numpy().astype(bool)
                has_outcome[batch] = mask_np
                value_pred[batch] = values.detach().cpu().numpy().astype(np.float32)
                value_target[batch] = outcome_t.detach().cpu().numpy().astype(np.float32)

    report = {
        "checkpoint": str(checkpoint),
        "samples": n,
        "loss": float(sum(losses) / max(n, 1)),
        "accuracy": float(np.mean(correct_all)),
        "top3_accuracy": float(np.mean(top3_all)),
        "accuracy_excluding_forced": _masked_mean(correct_all, ~forced),
        "top3_accuracy_excluding_forced": _masked_mean(top3_all, ~forced),
        "entropy": float(sum(entropies) / max(n, 1)),
        "value": _value_summary(value_pred, value_target, has_outcome),
        "by_phase": _breakdown(phases, correct_all, top3_all, forced, value_pred, value_target, has_outcome),
        "by_teacher": _breakdown(teachers, correct_all, top3_all, forced, value_pred, value_target, has_outcome),
        "by_action_type": _breakdown(action_types, correct_all, top3_all, forced, value_pred, value_target, has_outcome),
    }
    return {
        "report": report,
        "correct": correct_all,
        "top3": top3_all,
        "predictions": pred_all,
    }


def _value_summary(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int]:
    if not np.any(mask):
        return {"rows": 0}
    p = pred[mask].astype(np.float64)
    t = target[mask].astype(np.float64)
    mse = float(np.mean((p - t) ** 2))
    brier = float(np.mean((((p + 1.0) / 2.0) - ((t + 1.0) / 2.0)) ** 2))
    corr = float(np.corrcoef(p, t)[0, 1]) if len(p) > 1 and np.std(p) > 0 and np.std(t) > 0 else 0.0
    winners = t > 0
    losers = t < 0
    return {
        "rows": int(np.sum(mask)),
        "target_mean": float(np.mean(t)),
        "pred_mean": float(np.mean(p)),
        "pred_std": float(np.std(p)),
        "mse": mse,
        "brier_prob_space": brier,
        "corr": corr,
        "winner_pred_mean": float(np.mean(p[winners])) if np.any(winners) else 0.0,
        "loser_pred_mean": float(np.mean(p[losers])) if np.any(losers) else 0.0,
    }


def _breakdown(
    groups: np.ndarray,
    correct: np.ndarray,
    top3: np.ndarray,
    forced: np.ndarray,
    value_pred: np.ndarray,
    value_target: np.ndarray,
    has_outcome: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    buckets: defaultdict[str, list[int]] = defaultdict(list)
    for index, value in enumerate(groups.tolist()):
        key = str(value) if str(value) else "unknown"
        buckets[key].append(index)
    result = {}
    for key, indices in sorted(buckets.items(), key=lambda item: (-len(item[1]), item[0])):
        idx = np.asarray(indices, dtype=np.int64)
        unforced = ~forced[idx]
        value_mask = has_outcome[idx]
        result[key] = {
            "samples": int(len(idx)),
            "accuracy": float(np.mean(correct[idx])) if len(idx) else 0.0,
            "top3_accuracy": float(np.mean(top3[idx])) if len(idx) else 0.0,
            "forced_action_fraction": float(np.mean(forced[idx])) if len(idx) else 0.0,
            "accuracy_excluding_forced": _masked_mean(correct[idx], unforced),
            "top3_accuracy_excluding_forced": _masked_mean(top3[idx], unforced),
            "value": _value_summary(value_pred[idx], value_target[idx], value_mask),
        }
    return result


def _compare_predictions(
    *,
    base: dict[str, np.ndarray],
    other: dict[str, np.ndarray],
    phases: np.ndarray,
    teachers: np.ndarray,
    forced: np.ndarray,
) -> dict[str, object]:
    base_correct = base["correct"]
    other_correct = other["correct"]
    changed = base["predictions"] != other["predictions"]
    result: dict[str, object] = {
        "samples": int(len(base_correct)),
        "prediction_changed_fraction": float(np.mean(changed)),
        "base_correct_other_wrong": int(np.sum(base_correct & ~other_correct)),
        "other_correct_base_wrong": int(np.sum(other_correct & ~base_correct)),
        "both_correct": int(np.sum(base_correct & other_correct)),
        "both_wrong": int(np.sum(~base_correct & ~other_correct)),
        "base_correct_other_wrong_excluding_forced": int(np.sum((base_correct & ~other_correct) & ~forced)),
        "other_correct_base_wrong_excluding_forced": int(np.sum((other_correct & ~base_correct) & ~forced)),
        "by_phase": {},
        "by_teacher": {},
    }
    for label, groups in (("by_phase", phases), ("by_teacher", teachers)):
        buckets: defaultdict[str, list[int]] = defaultdict(list)
        for index, value in enumerate(groups.tolist()):
            buckets[str(value) if str(value) else "unknown"].append(index)
        out = {}
        for key, indices in sorted(buckets.items(), key=lambda item: (-len(item[1]), item[0])):
            idx = np.asarray(indices, dtype=np.int64)
            out[key] = {
                "samples": int(len(idx)),
                "prediction_changed_fraction": float(np.mean(changed[idx])),
                "base_correct_other_wrong": int(np.sum(base_correct[idx] & ~other_correct[idx])),
                "other_correct_base_wrong": int(np.sum(other_correct[idx] & ~base_correct[idx])),
            }
        result[label] = out
    return result


def _action_types(track: str, vps_to_win: int, actions: np.ndarray) -> np.ndarray:
    env = ColonistMultiAgentEnv(parse_track(track, vps_to_win=vps_to_win))
    try:
        env.reset(seed=0)
        cache: dict[int, str] = {}
        values = []
        for action in actions.tolist():
            action_i = int(action)
            if action_i not in cache:
                description = env.describe_action(action_i)
                cache[action_i] = str((description or {}).get("action_type", "unknown"))
            values.append(cache[action_i])
        return np.asarray(values, dtype=object)
    finally:
        env.close()


def _parse_checkpoint_spec(spec: str) -> tuple[str, str]:
    if "=" in spec:
        name, path = spec.split("=", 1)
        return name.strip(), path.strip()
    path = Path(spec)
    return path.parent.name or path.stem, spec


def _string_column(data: dict, key: str, n: int) -> np.ndarray:
    if key not in data:
        return np.asarray([""] * n, dtype=object)
    return np.asarray(data[key]).astype(str)


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    return float(np.mean(values[mask]))


if __name__ == "__main__":
    main()
