#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import sys
import time
from typing import Any

import numpy as np

from catan_zero.rl import ppo_distributed as dist
from catan_zero.rl.config_cli import _explicit_cli_dests
from catan_zero.rl.ppo_policy_factory import (
    CANONICAL_PPO_ARCHITECTURE,
    load_ppo_policy,
    validate_canonical_ppo_actor_contract,
)
from catan_zero.rl.ppo_run_manifest import ManifestError, PPORunManifest, load_manifest
from catan_zero.rl.torch_ppo import collect_ppo_episode
try:
    from factory_common import make_named_policy, parse_track
except ModuleNotFoundError as error:  # package import; direct script uses the sibling import
    if error.name != "factory_common":
        raise
    from tools.factory_common import make_named_policy, parse_track


SEATS = ("BLUE", "RED", "ORANGE", "WHITE")

_MANIFEST_RUNTIME_DESTS = {
    "run_manifest",
    "run_base",
    "run_name",
    "checkpoint",
    "devices",
    "games",
    "workers",
    "publish",
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate local distributed-PPO trajectory shards for the 35M entity policy."
    )
    parser.add_argument(
        "--run-manifest",
        default=None,
        help=(
            "Bound canonical_entity_ppo_run_v2 manifest. In manifest mode it is "
            "the sole actor-science authority."
        ),
    )
    parser.add_argument("--run-base", default="runs/distributed")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--architecture",
        choices=(CANONICAL_PPO_ARCHITECTURE,),
        default=CANONICAL_PPO_ARCHITECTURE,
    )
    parser.add_argument("--track", default="2p_no_trade")
    parser.add_argument("--vps-to-win", type=int, default=10)
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--games-per-shard", type=int, default=2)
    parser.add_argument("--max-decisions", type=int, default=1200)
    parser.add_argument("--opponents", default="catanatron_ab3,catanatron_value,heuristic,random")
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--seed", type=int, default=70630001)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--value-shaping-coef", type=float, default=0.0)
    parser.add_argument("--value-shaping-scale", type=float, default=100.0)
    parser.add_argument("--value-shaping-opponent-penalty", type=float, default=0.05)
    parser.add_argument(
        "--action-temperature",
        type=float,
        default=1.0,
        help=(
            "Temperature for stochastic PPO actor sampling. 1.0 preserves the "
            "raw policy; lower values reduce destructive exploration while "
            "recording old_log_probs under the actual behavior distribution."
        ),
    )
    parser.add_argument("--publish", action="store_true")
    return parser


def resolve_config(
    argv: list[str] | None = None,
) -> tuple[argparse.Namespace, PPORunManifest | None]:
    parser = build_arg_parser()
    effective_argv = list(argv) if argv is not None else None
    args = parser.parse_args(effective_argv)
    explicit_dests = _explicit_cli_dests(
        parser,
        effective_argv if effective_argv is not None else sys.argv[1:],
    )
    manifest = None
    args.run_manifest_sha256 = None
    args.opponent_mode = "fixed"
    args.pfsp_mode = "pfsp"
    if args.run_manifest:
        conflicts = sorted(explicit_dests - _MANIFEST_RUNTIME_DESTS)
        if conflicts:
            parser.error(
                "--run-manifest cannot be combined with legacy actor-science "
                f"flags: {', '.join(conflicts)}"
            )
        try:
            manifest = load_manifest(args.run_manifest)
        except (OSError, ManifestError) as error:
            parser.error(f"invalid --run-manifest: {error}")
        if manifest.status != "bound":
            parser.error("--run-manifest must have status='bound'; templates cannot run")
        expected_sha256 = manifest.spec.identity.initializer_sha256
        try:
            actual_sha256 = f"sha256:{dist.checkpoint_sha256(args.checkpoint)}"
        except OSError as error:
            parser.error(f"cannot hash --checkpoint: {error}")
        if actual_sha256 != expected_sha256:
            parser.error(
                "--checkpoint SHA-256 does not match run manifest identity: "
                f"expected={expected_sha256} actual={actual_sha256}"
            )
        identity = manifest.spec.identity
        actor = manifest.spec.actor
        if actor.opponent_mode != "fixed":
            parser.error(
                "local v2 actor supports only opponent_mode='fixed'; "
                "league/PFSP execution requires the Modal actor"
            )
        args.architecture = identity.architecture
        args.track = identity.track
        args.vps_to_win = identity.vps_to_win
        args.max_decisions = actor.max_decisions
        args.games_per_shard = actor.games_per_shard
        args.gamma = actor.gamma
        args.gae_lambda = actor.gae_lambda
        args.action_temperature = actor.action_temperature
        args.value_shaping_coef = actor.value_shaping_coef
        args.value_shaping_scale = actor.value_shaping_scale
        args.value_shaping_opponent_penalty = actor.value_shaping_opponent_penalty
        args.seed = actor.seed
        args.opponent_mode = actor.opponent_mode
        args.opponents = ",".join(actor.opponents)
        args.pfsp_mode = actor.pfsp_mode
        args.run_manifest_sha256 = manifest.sha256()
    try:
        validate_canonical_ppo_actor_contract(
            architecture=args.architecture,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            action_temperature=args.action_temperature,
        )
    except ValueError as error:
        parser.error(str(error))
    return args, manifest


def _bind_run_root(
    args: argparse.Namespace,
    manifest: PPORunManifest | None,
):
    root = dist.run_root(args.run_base, args.run_name)
    if manifest is not None:
        dist.bind_run_manifest(root, manifest)
        dist.ensure_run_dirs(root)
    else:
        dist.ensure_run_dirs(root)
        dist.bind_run_contract(
            root,
            init_checkpoint=args.checkpoint,
            architecture=args.architecture,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            behavior_temperature=args.action_temperature,
        )
    return root


def _build_worker_payloads(
    args: argparse.Namespace,
    published: Any,
) -> tuple[list[dict[str, Any]], list[str], int]:
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if not devices:
        devices = ["cpu"]
    workers = max(1, int(args.workers))
    games = max(0, int(args.games))
    base = games // workers
    remainder = games % workers
    payloads: list[dict[str, Any]] = []
    offset = 0
    for worker in range(workers):
        worker_games = base + (1 if worker < remainder else 0)
        if worker_games <= 0:
            continue
        payloads.append(
            {
                "run_base": str(args.run_base),
                "run_name": str(args.run_name),
                "worker_id": f"local_{worker:03d}",
                "checkpoint": str(published.path),
                "policy_version": int(published.version),
                "architecture": str(args.architecture),
                "device": devices[worker % len(devices)],
                "track": str(args.track),
                "vps_to_win": int(args.vps_to_win),
                "games": int(worker_games),
                "game_offset": int(offset),
                "games_per_shard": int(args.games_per_shard),
                "max_decisions": int(args.max_decisions),
                "opponents": str(args.opponents),
                "opponent_mode": str(args.opponent_mode),
                "pfsp_mode": str(args.pfsp_mode),
                "seed": int(args.seed) + worker * 1_000_003,
                "gamma": float(args.gamma),
                "gae_lambda": float(args.gae_lambda),
                "value_shaping_coef": float(args.value_shaping_coef),
                "value_shaping_scale": float(args.value_shaping_scale),
                "value_shaping_opponent_penalty": float(args.value_shaping_opponent_penalty),
                "action_temperature": float(args.action_temperature),
                "run_manifest_sha256": args.run_manifest_sha256,
            }
        )
        offset += worker_games
    return payloads, devices, games


def main(argv: list[str] | None = None) -> None:
    args, manifest = resolve_config(argv)

    root = _bind_run_root(args, manifest)
    if args.publish or dist.read_version(root) is None:
        policy = load_ppo_policy(args.checkpoint, architecture=args.architecture, device="cpu")
        published = dist.publish_weights(root, policy.save, step=0)
    else:
        published = dist.read_version(root)
        if published is None:
            raise RuntimeError("failed to publish or read policy version")

    payloads, devices, games = _build_worker_payloads(args, published)

    print(
        json.dumps(
            {
                "event": "local_ppo_shards_start",
                "run_root": str(root),
                "policy_version": int(published.version),
                "checkpoint": str(published.path),
                "games": games,
                "workers": len(payloads),
                "devices": devices,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    started = time.perf_counter()
    reports: list[dict[str, Any]] = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(payloads), mp_context=ctx) as executor:
        futures = [executor.submit(_worker, payload) for payload in payloads]
        for future in as_completed(futures):
            report = future.result()
            reports.append(report)
            print(json.dumps({"event": "local_ppo_worker_done", **report}, sort_keys=True), flush=True)
    total_games = sum(int(report["games"]) for report in reports)
    total_samples = sum(int(report["samples"]) for report in reports)
    total_shards = sum(int(report["shards"]) for report in reports)
    print(
        json.dumps(
            {
                "event": "local_ppo_shards_done",
                "run_root": str(root),
                "games": total_games,
                "samples": total_samples,
                "shards": total_shards,
                "elapsed_sec": time.perf_counter() - started,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _worker(payload: dict[str, Any]) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    root = dist.run_root(payload["run_base"], payload["run_name"])
    if str(payload.get("opponent_mode", "fixed")) != "fixed":
        raise RuntimeError(
            "local PPO worker supports only fixed opponents; league/PFSP is unavailable"
        )
    policy = load_ppo_policy(
        payload["checkpoint"],
        architecture=payload["architecture"],
        device=payload["device"],
    )
    model = getattr(policy, "model", None)
    if model is not None:
        model.eval()
    config = parse_track(payload["track"], vps_to_win=int(payload["vps_to_win"]))
    player_count = int(config.players)
    seats = SEATS[:player_count]
    opponent_names = [name.strip() for name in str(payload["opponents"]).split(",") if name.strip()]
    if not opponent_names:
        opponent_names = ["random"]
    rng = np.random.default_rng(int(payload["seed"]))
    buffer = []
    shard_index = 0
    samples = 0
    shards = 0
    started = time.perf_counter()
    for game_index in range(int(payload["games"])):
        global_game = int(payload["game_offset"]) + game_index
        training_seat = seats[global_game % player_count]
        opponents = {}
        for seat in seats:
            if seat == training_seat:
                continue
            name = opponent_names[int(rng.integers(0, len(opponent_names)))]
            opponents[seat] = make_named_policy(name)
        trajectory = collect_ppo_episode(
            policy,
            opponents,
            seed=int(rng.integers(2**31)),
            config=config,
            max_decisions=int(payload["max_decisions"]),
            rng=rng,
            training_seats={training_seat},
            gamma=float(payload["gamma"]),
            gae_lambda=float(payload["gae_lambda"]),
            value_shaping_coef=float(payload["value_shaping_coef"]),
            value_shaping_scale=float(payload["value_shaping_scale"]),
            value_shaping_opponent_penalty=float(payload["value_shaping_opponent_penalty"]),
            action_temperature=float(payload["action_temperature"]),
        )
        buffer.append(trajectory)
        samples += len(trajectory.samples)
        if len(buffer) >= int(payload["games_per_shard"]):
            dist.write_trajectory_shard(
                root,
                str(payload["worker_id"]),
                shard_index,
                buffer,
                policy_version=int(payload["policy_version"]),
                run_manifest_sha256=payload.get("run_manifest_sha256"),
            )
            shards += 1
            shard_index += 1
            buffer = []
    if buffer:
        dist.write_trajectory_shard(
            root,
            str(payload["worker_id"]),
            shard_index,
            buffer,
            policy_version=int(payload["policy_version"]),
            run_manifest_sha256=payload.get("run_manifest_sha256"),
        )
        shards += 1
    return {
        "worker_id": str(payload["worker_id"]),
        "device": str(payload["device"]),
        "games": int(payload["games"]),
        "samples": int(samples),
        "shards": int(shards),
        "elapsed_sec": time.perf_counter() - started,
    }


if __name__ == "__main__":
    main()
