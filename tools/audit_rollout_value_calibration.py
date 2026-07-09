#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
import time
from typing import Any

import numpy as np

from catan_zero.rl.ppo_policy_factory import load_ppo_policy
from catan_zero.rl.torch_ppo import collect_ppo_episode
from factory_common import make_named_policy, parse_track, write_json


SEATS = ("BLUE", "RED", "ORANGE", "WHITE")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit rollout-state value calibration for a 35M entity checkpoint. "
            "Uses collect_ppo_episode so states/values match the PPO actor path."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--architecture", default="entity_graph")
    parser.add_argument("--track", default="2p_no_trade")
    parser.add_argument("--vps-to-win", type=int, default=10)
    parser.add_argument("--opponents", default="catanatron_value,catanatron_ab3,catanatron_ab4")
    parser.add_argument("--games", type=int, default=192)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--max-decisions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=70702001)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=1.0)
    parser.add_argument(
        "--action-temperature",
        type=float,
        default=1.0,
        help="Temperature for stochastic actor sampling during calibration.",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    opponents = [item.strip() for item in args.opponents.split(",") if item.strip()]
    if not opponents:
        raise SystemExit("--opponents must not be empty")
    devices = [item.strip() for item in args.devices.split(",") if item.strip()] or ["cpu"]
    workers = max(1, min(int(args.workers), int(args.games)))

    payloads: list[dict[str, Any]] = []
    base = int(args.games) // workers
    remainder = int(args.games) % workers
    offset = 0
    for worker in range(workers):
        count = base + (1 if worker < remainder else 0)
        if count <= 0:
            continue
        payloads.append(
            {
                "worker": worker,
                "checkpoint": args.checkpoint,
                "architecture": args.architecture,
                "track": args.track,
                "vps_to_win": int(args.vps_to_win),
                "opponents": opponents,
                "games": count,
                "offset": offset,
                "device": devices[worker % len(devices)],
                "max_decisions": int(args.max_decisions),
                "seed": int(args.seed) + worker * 1_000_003,
                "gamma": float(args.gamma),
                "gae_lambda": float(args.gae_lambda),
                "action_temperature": float(args.action_temperature),
            }
        )
        offset += count

    started = time.perf_counter()
    print(
        json.dumps(
            {
                "event": "rollout_value_calibration_start",
                "checkpoint": args.checkpoint,
                "games": int(args.games),
                "workers": len(payloads),
                "devices": devices,
                "opponents": opponents,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    reports: list[dict[str, Any]] = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(payloads), mp_context=ctx) as executor:
        futures = [executor.submit(_worker, payload) for payload in payloads]
        for future in as_completed(futures):
            report = future.result()
            reports.append(report)
            print(
                json.dumps(
                    {
                        "event": "rollout_value_calibration_worker_done",
                        "worker": report["worker"],
                        "games": report["games"],
                        "samples": len(report["values"]),
                        "elapsed_sec": report["elapsed_sec"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    report = _aggregate(args, reports, elapsed=time.perf_counter() - started)
    write_json(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True))


def _worker(payload: dict[str, Any]) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    started = time.perf_counter()
    rng = np.random.default_rng(int(payload["seed"]))
    policy = load_ppo_policy(
        payload["checkpoint"],
        architecture=payload["architecture"],
        device=payload["device"],
    )
    model = getattr(policy, "model", None)
    if model is not None:
        model.eval()
    config = parse_track(payload["track"], vps_to_win=int(payload["vps_to_win"]))
    seats = SEATS[: int(config.players)]
    opponents = list(payload["opponents"])

    values: list[float] = []
    returns: list[float] = []
    phases: list[str] = []
    opponent_labels: list[str] = []
    legal_counts: list[int] = []
    game_outcomes: list[float] = []
    game_opponents: list[str] = []
    truncated = 0
    empty = 0

    for local_game in range(int(payload["games"])):
        global_game = int(payload["offset"]) + local_game
        training_seat = seats[global_game % len(seats)]
        opponent_name = opponents[global_game % len(opponents)]
        opponent_policies = {
            seat: make_named_policy(opponent_name)
            for seat in seats
            if seat != training_seat
        }
        trajectory = collect_ppo_episode(
            policy,
            opponent_policies,
            seed=int(rng.integers(2**31)),
            config=config,
            max_decisions=int(payload["max_decisions"]),
            rng=rng,
            training_seats={training_seat},
            gamma=float(payload["gamma"]),
            gae_lambda=float(payload["gae_lambda"]),
            action_temperature=float(payload["action_temperature"]),
        )
        if trajectory.truncated:
            truncated += 1
        if not trajectory.samples:
            empty += 1
            continue
        outcome = float(trajectory.returns[-1])
        game_outcomes.append(outcome)
        game_opponents.append(opponent_name)
        for sample, value, ret in zip(
            trajectory.samples,
            trajectory.old_values,
            trajectory.returns,
        ):
            values.append(float(value))
            returns.append(float(ret))
            phases.append(str(sample.phase or "unknown"))
            opponent_labels.append(opponent_name)
            legal_counts.append(len(tuple(sample.valid_actions)))

    return {
        "worker": int(payload["worker"]),
        "games": int(payload["games"]),
        "values": values,
        "returns": returns,
        "phases": phases,
        "opponents": opponent_labels,
        "legal_counts": legal_counts,
        "game_outcomes": game_outcomes,
        "game_opponents": game_opponents,
        "truncated_games": truncated,
        "empty_games": empty,
        "elapsed_sec": time.perf_counter() - started,
    }


def _aggregate(args: argparse.Namespace, reports: list[dict[str, Any]], *, elapsed: float) -> dict[str, Any]:
    values = np.asarray(_flat(report["values"] for report in reports), dtype=np.float64)
    returns = np.asarray(_flat(report["returns"] for report in reports), dtype=np.float64)
    phases = np.asarray(_flat(report["phases"] for report in reports), dtype=object)
    opponents = np.asarray(_flat(report["opponents"] for report in reports), dtype=object)
    legal_counts = np.asarray(_flat(report["legal_counts"] for report in reports), dtype=np.float64)
    game_outcomes = np.asarray(_flat(report["game_outcomes"] for report in reports), dtype=np.float64)
    game_opponents = np.asarray(_flat(report["game_opponents"] for report in reports), dtype=object)
    wins = game_outcomes > 0.0

    result: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "architecture": str(args.architecture),
        "track": str(args.track),
        "vps_to_win": int(args.vps_to_win),
        "games": int(sum(int(report["games"]) for report in reports)),
        "samples": int(len(values)),
        "elapsed_sec": float(elapsed),
        "workers": int(len(reports)),
        "devices": [item.strip() for item in args.devices.split(",") if item.strip()],
        "opponents": [item.strip() for item in args.opponents.split(",") if item.strip()],
        "action_temperature": float(args.action_temperature),
        "truncated_games": int(sum(int(report["truncated_games"]) for report in reports)),
        "empty_games": int(sum(int(report["empty_games"]) for report in reports)),
        "win_rate": float(np.mean(wins)) if len(wins) else None,
        "value_calibration": _calibration(values, returns),
        "legal_count": _stat(legal_counts),
        "phase_counts": _counts(phases),
        "opponent_game_counts": _counts(game_opponents),
        "by_phase": {},
        "by_opponent": {},
        "by_opponent_phase": {},
    }
    for phase in sorted(set(str(item) for item in phases)):
        mask = phases == phase
        result["by_phase"][phase] = {
            "samples": int(np.sum(mask)),
            "value_calibration": _calibration(values[mask], returns[mask]),
            "legal_count": _stat(legal_counts[mask]),
        }
    for opponent in sorted(set(str(item) for item in opponents)):
        mask = opponents == opponent
        game_mask = game_opponents == opponent
        result["by_opponent"][opponent] = {
            "samples": int(np.sum(mask)),
            "games": int(np.sum(game_mask)),
            "win_rate": float(np.mean(game_outcomes[game_mask] > 0.0)) if np.any(game_mask) else None,
            "value_calibration": _calibration(values[mask], returns[mask]),
            "legal_count": _stat(legal_counts[mask]),
        }
        for phase in sorted(set(str(item) for item in phases[mask])):
            pair_mask = mask & (phases == phase)
            result["by_opponent_phase"][f"{opponent}/{phase}"] = {
                "samples": int(np.sum(pair_mask)),
                "value_calibration": _calibration(values[pair_mask], returns[pair_mask]),
            }
    return result


def _flat(items: Any) -> list[Any]:
    out: list[Any] = []
    for item in items:
        out.extend(item)
    return out


def _stat(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0}
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


def _calibration(values: np.ndarray, returns: np.ndarray) -> dict[str, Any]:
    v = np.asarray(values, dtype=np.float64)
    r = np.asarray(returns, dtype=np.float64)
    mask = np.isfinite(v) & np.isfinite(r)
    v = v[mask]
    r = r[mask]
    if v.size == 0:
        return {"n": 0}
    pred_prob = np.clip((v + 1.0) / 2.0, 0.0, 1.0)
    target_prob = np.clip((r + 1.0) / 2.0, 0.0, 1.0)
    corr = None
    if v.size >= 2 and float(np.std(v)) > 1.0e-9 and float(np.std(r)) > 1.0e-9:
        corr = float(np.corrcoef(v, r)[0, 1])
    bins = []
    edges = np.linspace(0.0, 1.0, 11)
    for i in range(10):
        lo = edges[i]
        hi = edges[i + 1]
        if i == 9:
            bin_mask = (pred_prob >= lo) & (pred_prob <= hi)
        else:
            bin_mask = (pred_prob >= lo) & (pred_prob < hi)
        if not np.any(bin_mask):
            continue
        bins.append(
            {
                "lo": float(lo),
                "hi": float(hi),
                "n": int(np.sum(bin_mask)),
                "pred_mean": float(np.mean(pred_prob[bin_mask])),
                "target_mean": float(np.mean(target_prob[bin_mask])),
                "abs_gap": float(abs(np.mean(pred_prob[bin_mask]) - np.mean(target_prob[bin_mask]))),
            }
        )
    ece = float(
        sum((item["n"] / float(v.size)) * item["abs_gap"] for item in bins)
    )
    return {
        "n": int(v.size),
        "value_mean": float(np.mean(v)),
        "return_mean": float(np.mean(r)),
        "mse": float(np.mean((v - r) ** 2)),
        "mae": float(np.mean(np.abs(v - r))),
        "corr": corr,
        "brier_prob_space": float(np.mean((pred_prob - target_prob) ** 2)),
        "ece_10bin_prob_space": ece,
        "positive_return_fraction": float(np.mean(r > 0.0)),
        "negative_return_fraction": float(np.mean(r < 0.0)),
        "bins": bins,
    }


def _counts(values: np.ndarray) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        counts[str(value)] += 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


if __name__ == "__main__":
    main()
