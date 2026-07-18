from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time

import numpy as np

from catan_zero.rl.gpu_rollout import RolloutConfig, collect_rank_rollout
from catan_zero.rl.policy_pool import PolicySpec, make_policy
from catan_zero.rl.self_play import RandomPolicy
from catan_zero.rl.torch_ppo import create_ppo_policy, make_ppo_optimizer, ppo_update
from catan_zero.rl.xdim_lite_policy import XDimLitePolicy
from factory_common import load_config, parse_track, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Custom torchrun PPO self-play trainer.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override-decisions", type=int, default=0)
    args = parser.parse_args()
    config = load_config(args.config)
    if config.get("schema") == "canonical_entity_ppo_run_v2":
        raise SystemExit(
            "tools/train_selfplay_gpu.py is the legacy flat/xdim PPO launcher and "
            "does not accept canonical_entity_ppo_run_v2 manifests; launch "
            "tools/run_local_entity_ppo_shards.py for actors and "
            "tools/ppo_distributed_learner.py for the learner"
        )
    rank, world_size, local_rank = _init_distributed()
    device = _device_for_rank(local_rank)
    seed = int(config.get("seed", 1)) + rank * 100_003
    np.random.seed(seed)

    env_config = parse_track(
        str(config.get("track", "2p_no_trade")),
        vps_to_win=int(config.get("vps_to_win", 10)),
    )
    arch = str(config.get("arch", "candidate"))
    if arch == "entity_graph":
        raise SystemExit(
            "tools/train_selfplay_gpu.py is a legacy flat/xdim launcher and cannot "
            "construct entity_graph; use tools/run_local_entity_ppo_shards.py plus "
            "tools/ppo_distributed_learner.py"
        )
    if arch not in {"candidate", "graph_history_candidate", "xdim_lite"}:
        raise SystemExit(f"unsupported legacy PPO architecture: {arch!r}")
    init_checkpoint = config.get("init_checkpoint")
    if arch == "xdim_lite":
        if init_checkpoint:
            policy = XDimLitePolicy.load(init_checkpoint, device=device)
        else:
            policy = XDimLitePolicy.create(
                env_config=env_config,
                seed=seed,
                hidden_size=int(config.get("hidden_size", 512)),
                device=device,
            )
    else:
        policy = create_ppo_policy(
            config=env_config,
            seed=seed,
            hidden_size=int(config.get("hidden_size", 512)),
            architecture="graph_history_candidate" if arch == "graph_history_candidate" else "candidate",
            device=device,
        )
    if init_checkpoint and arch != "xdim_lite":
        from catan_zero.rl.torch_ppo import TorchPPOPolicy

        loaded = TorchPPOPolicy.load(init_checkpoint, device=device)
        _copy_policy_weights(policy, loaded)
    if world_size > 1:
        _wrap_ddp(policy, local_rank)
    optimizer = make_ppo_optimizer(policy, learning_rate=float(config["ppo"]["lr"]))
    opponent = _make_training_opponent(config)
    opponents = {name: opponent for name in ("BLUE", "RED", "ORANGE", "WHITE")[: env_config.players]}

    run_dir = Path(config.get("run_dir", "runs/self_play/gpu_ppo"))
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
    decisions = int(config["rollout"]["decisions_per_rank"])
    if args.override_decisions > 0:
        decisions = args.override_decisions
    rollout_workers = max(1, int(config["rollout"].get("workers", 1)))
    if world_size > 1:
        # Keep DDP backward counts identical in the first production-safe version.
        minibatch_size = max(int(config["ppo"]["minibatch_size"]), decisions * 2)
    else:
        minibatch_size = int(config["ppo"]["minibatch_size"])

    reports = []
    total_iterations = int(config.get("iterations", 1))
    for iteration in range(1, total_iterations + 1):
        start = time.perf_counter()
        rollout_start = time.perf_counter()
        rollout_config = RolloutConfig(
            decisions_per_rank=decisions,
            max_decisions_per_game=int(config["rollout"].get("max_decisions_per_game", 1000)),
            gamma=float(config["rollout"].get("gamma", 0.997)),
            gae_lambda=float(config["rollout"].get("gae_lambda", 0.95)),
            seed=seed + iteration * 1_000_000,
        )
        if rollout_workers > 1:
            _unwrap_ddp(policy)
            rollout_checkpoint = run_dir / "_rollout_policy.pt"
            if rank == 0:
                policy.save(rollout_checkpoint)
            _barrier()
            if world_size > 1:
                _wrap_ddp(policy, local_rank)
            trajectories = _collect_parallel_rollouts(
                checkpoint=rollout_checkpoint,
                env_config=env_config,
                config=config,
                rollout_config=rollout_config,
                workers=rollout_workers,
                rank=rank,
            )
        else:
            trajectories = collect_rank_rollout(
                policy,
                opponents,
                env_config=env_config,
                rollout_config=rollout_config,
                rank=rank,
                training_seats={"BLUE"},
            )
        rollout_elapsed = time.perf_counter() - rollout_start
        update_start = time.perf_counter()
        update = ppo_update(
            policy,
            trajectories,
            learning_rate=float(config["ppo"]["lr"]),
            clip_ratio=float(config["ppo"]["clip_ratio"]),
            value_coef=float(config["ppo"]["value_coef"]),
            entropy_coef=float(config["ppo"]["entropy_coef"]),
            epochs=int(config["ppo"]["epochs"]),
            minibatch_size=minibatch_size,
            optimizer=optimizer,
            value_clip_range=float(config["ppo"].get("value_clip_range", 0.0)),
        )
        update_elapsed = time.perf_counter() - update_start
        elapsed = time.perf_counter() - start
        samples = sum(len(t.samples) for t in trajectories)
        report = {
            "iteration": iteration,
            "rank": rank,
            "world_size": world_size,
            "samples": samples,
            "elapsed_sec": elapsed,
            "rollout_elapsed_sec": rollout_elapsed,
            "update_elapsed_sec": update_elapsed,
            "rollout_workers": rollout_workers,
            "samples_per_second": samples / elapsed if elapsed else 0.0,
            "update": update,
        }
        print(json.dumps(report, sort_keys=True), flush=True)
        reports.append(report)
        _barrier()
        if rank == 0:
            checkpoint = run_dir / f"iter{iteration:04d}.pt"
            _unwrap_ddp(policy)
            policy.save(checkpoint)
            if world_size > 1 and iteration < total_iterations:
                _wrap_ddp(policy, local_rank)
            write_json(run_dir / "latest_report.json", {"iterations": reports})
        _barrier()
    _destroy_distributed()


def _make_training_opponent(config: dict):
    opponents = config.get("opponents", {})
    baseline = str(opponents.get("baseline", "random"))
    if "," in baseline:
        specs = [PolicySpec(kind=name.strip()) for name in baseline.split(",") if name.strip()]
        return _ActionMixedPolicy(specs)
    if baseline == "self":
        return RandomPolicy()
    return make_policy(PolicySpec(kind=baseline))


class _ActionMixedPolicy:
    def __init__(self, specs: list[PolicySpec]) -> None:
        if not specs:
            raise ValueError("mixed opponent requires at least one policy")
        self.name = "mixed_" + "_".join(spec.kind for spec in specs)
        self._policies = [make_policy(spec) for spec in specs]

    def select_action(self, env, observation, info, rng, *, training: bool = False) -> int:
        index = int(rng.integers(len(self._policies)))
        return int(self._policies[index].select_action(env, observation, info, rng, training=training))


def _collect_parallel_rollouts(
    *,
    checkpoint: Path,
    env_config,
    config: dict,
    rollout_config: RolloutConfig,
    workers: int,
    rank: int,
):
    decisions = int(rollout_config.decisions_per_rank)
    base = decisions // workers
    remainder = decisions % workers
    payloads = []
    for worker_id in range(workers):
        worker_decisions = base + (1 if worker_id < remainder else 0)
        if worker_decisions <= 0:
            continue
        payloads.append(
            {
                "checkpoint": str(checkpoint),
                "env_config": env_config,
                "config": config,
                "rollout": RolloutConfig(
                    decisions_per_rank=worker_decisions,
                    max_decisions_per_game=rollout_config.max_decisions_per_game,
                    gamma=rollout_config.gamma,
                    gae_lambda=rollout_config.gae_lambda,
                    seed=rollout_config.seed + worker_id * 1_000_003,
                ),
                "rank": rank * workers + worker_id,
            }
        )
    trajectories = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_collect_rollout_worker, payload) for payload in payloads]
        for future in as_completed(futures):
            trajectories.extend(future.result())
    return trajectories


def _collect_rollout_worker(payload: dict):
    from catan_zero.rl.torch_ppo import TorchPPOPolicy
    from catan_zero.rl.xdim_lite_policy import XDimLitePolicy

    arch = str(payload["config"].get("arch", "candidate"))
    if arch == "xdim_lite":
        policy = XDimLitePolicy.load(payload["checkpoint"], device="cpu")
    else:
        policy = TorchPPOPolicy.load(payload["checkpoint"], device="cpu")
    opponent = _make_training_opponent(payload["config"])
    opponents = {
        name: opponent
        for name in ("BLUE", "RED", "ORANGE", "WHITE")[
            : payload["env_config"].players
        ]
    }
    return collect_rank_rollout(
        policy,
        opponents,
        env_config=payload["env_config"],
        rollout_config=payload["rollout"],
        rank=int(payload["rank"]),
        training_seats={"BLUE"},
    )


def _init_distributed() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        import torch.distributed as dist

        dist.init_process_group(backend="nccl" if _cuda_available() else "gloo")
    return rank, world_size, local_rank


def _destroy_distributed() -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        import torch.distributed as dist

        dist.destroy_process_group()


def _barrier() -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        import torch.distributed as dist

        dist.barrier()


def _device_for_rank(local_rank: int) -> str:
    import torch

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return f"cuda:{local_rank}"
    return "cpu"


def _cuda_available() -> bool:
    import torch

    return bool(torch.cuda.is_available())


def _wrap_ddp(policy, local_rank: int) -> None:
    import torch
    from torch.nn.parallel import DistributedDataParallel

    device_ids = [local_rank] if torch.cuda.is_available() else None
    for name in _module_names():
        module = getattr(policy, name, None)
        if module is not None and not hasattr(module, "module"):
            setattr(policy, name, DistributedDataParallel(module, device_ids=device_ids))


def _unwrap_ddp(policy) -> None:
    for name in _module_names():
        module = getattr(policy, name, None)
        if module is not None and hasattr(module, "module"):
            setattr(policy, name, module.module)


def _module_names() -> tuple[str, ...]:
    return (
        "model",
        "actor",
        "critic",
        "q_head",
        "q_state",
        "q_action_encoder",
        "q_action_bias",
        "action_encoder",
        "action_id_embedding",
        "action_bias",
    )


def _copy_policy_weights(target, source) -> None:
    for name in _module_names():
        target_module = getattr(target, name, None)
        source_module = getattr(source, name, None)
        if target_module is not None and source_module is not None:
            target_module.load_state_dict(source_module.state_dict())


if __name__ == "__main__":
    main()
