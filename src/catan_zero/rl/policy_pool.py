from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from catan_zero.rl.self_play import (
    CatanatronAlphaBetaPolicy,
    CatanatronGreedyPlayoutsPolicy,
    CatanatronMCTSPolicy,
    CatanatronSameTurnAlphaBetaPolicy,
    CatanatronValuePolicy,
    CatanatronWeightedRandomPolicy,
    HeuristicPolicy,
    JSettlersLitePolicy,
    RandomPolicy,
    STYLE_SPECIALIST_WEIGHTS,
    ValueRolloutSearchPolicy,
)
from catan_zero.rl.torch_ppo import TorchPPOPolicy
from catan_zero.rl.entity_token_policy import EntityGraphPolicy
from catan_zero.rl.xdim_lite_policy import XDimLitePolicy, _array_sha256


@dataclass(frozen=True, slots=True)
class PolicySpec:
    kind: str
    weight: float = 1.0
    checkpoint: str | None = None


class PolicyPool:
    def __init__(self, specs: list[PolicySpec], *, seed: int = 0, device: str | None = None) -> None:
        if not specs:
            raise ValueError("policy pool requires at least one policy")
        self.rng = np.random.default_rng(seed)
        self.entries = [(spec, make_policy(spec, device=device)) for spec in specs]
        weights = np.asarray([max(float(spec.weight), 0.0) for spec, _ in self.entries])
        if float(weights.sum()) <= 0.0:
            weights[:] = 1.0
        self.weights = weights / float(weights.sum())

    def sample(self):
        index = int(self.rng.choice(len(self.entries), p=self.weights))
        return self.entries[index][1]


def make_policy(spec: PolicySpec, *, device: str | None = None):
    kind = spec.kind.lower()
    if kind == "random":
        return RandomPolicy()
    if kind in {"heuristic", "catanatron_heuristic"}:
        return HeuristicPolicy()
    if kind in {"weighted_random", "catanatron_weighted_random"}:
        return CatanatronWeightedRandomPolicy()
    if kind == "jsettlers_lite":
        return JSettlersLitePolicy()
    if kind in {"value", "catanatron_value"}:
        return CatanatronValuePolicy()
    if kind in {
        "catanatron_value_ore_city",
        "catanatron_value_road_race",
        "catanatron_value_robber",
    }:
        style = kind[len("catanatron_value_") :]
        return CatanatronValuePolicy(
            value_fn_builder_name="base_fn",
            params=STYLE_SPECIALIST_WEIGHTS[style],
            name=kind,
        )
    if kind in {"alphabeta", "alpha_beta", "ab3", "catanatron_ab3", "catanatron_alphabeta"}:
        return CatanatronAlphaBetaPolicy(depth=3, prunning=True)
    if kind in {"ab4", "catanatron_ab4"}:
        return CatanatronAlphaBetaPolicy(depth=4, prunning=True)
    if kind in {"ab5", "catanatron_ab5"}:
        return CatanatronAlphaBetaPolicy(depth=5, prunning=True)
    if kind in {"sab3", "catanatron_sab3", "same_turn_ab3"}:
        return CatanatronSameTurnAlphaBetaPolicy(depth=3, prunning=True)
    if kind in {"sab4", "catanatron_sab4", "same_turn_ab4"}:
        return CatanatronSameTurnAlphaBetaPolicy(depth=4, prunning=True)
    if kind in {"mcts100", "catanatron_mcts100"}:
        return CatanatronMCTSPolicy(simulations=100, prunning=False)
    if kind in {"mcts50", "catanatron_mcts50"}:
        return CatanatronMCTSPolicy(simulations=50, prunning=False)
    if kind in {"greedy25", "catanatron_greedy25", "greedy_playouts25"}:
        return CatanatronGreedyPlayoutsPolicy(playouts=25)
    if kind in {"search", "value_rollout", "value_rollout_search", "catanatron_search"}:
        return ValueRolloutSearchPolicy()
    if kind in {"ppo", "candidate", "torch"}:
        if not spec.checkpoint:
            raise ValueError(f"{kind} policy requires a checkpoint")
        return TorchPPOPolicy.load(spec.checkpoint, device=device)
    if kind in {"xdim_lite", "xdim_graph"}:
        if not spec.checkpoint:
            raise ValueError(f"{kind} policy requires a checkpoint")
        return XDimLitePolicy.load(spec.checkpoint, device=device)
    if kind == "entity_graph":
        if not spec.checkpoint:
            raise ValueError(f"{kind} policy requires a checkpoint")
        return EntityGraphPolicy.load(spec.checkpoint, device=device)
    if Path(kind).exists():
        return load_checkpoint_policy(kind, device=device)
    raise ValueError(f"unknown policy kind: {spec.kind}")


def load_checkpoint_policy(path: str | Path, *, device: str | None = None):
    import torch

    checkpoint = Path(path)
    try:
        data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(checkpoint, map_location="cpu")
    if not isinstance(data, dict):
        raise ValueError(f"{checkpoint} is not a dict checkpoint")
    policy_type = data.get("policy_type")
    if policy_type in {"xdim_lite", "xdim_graph"}:
        return XDimLitePolicy.load(checkpoint, device=device)
    if policy_type == "entity_graph":
        return EntityGraphPolicy.load(checkpoint, device=device)
    if {"observation_size", "action_size", "model", "actor", "critic"}.issubset(data):
        return TorchPPOPolicy.load(checkpoint, device=device)
    raise ValueError(
        f"unknown checkpoint schema for {checkpoint}; keys={sorted(map(str, data.keys()))[:20]}"
    )


def assert_policy_compatible_with_env(policy, env_config) -> None:
    from catan_zero.rl.action_features import build_action_context_feature_table
    from catan_zero.rl.multiagent_env import ColonistMultiAgentEnv
    from catan_zero.rl.torch_ppo import build_action_feature_table

    config = getattr(policy, "config", None)
    problems: list[str] = []
    env = ColonistMultiAgentEnv(env_config)
    try:
        observations, info = env.reset(seed=0)
        action_size = int(getattr(policy, "action_size", getattr(config, "action_size", 0)))
        if action_size and int(env.action_space.n) != action_size:
            problems.append(
                f"action_size checkpoint={action_size} env={int(env.action_space.n)}"
            )
        observation_size = int(getattr(config, "observation_size", 0))
        if observation_size:
            env_observation_size = len(next(iter(observations.values())))
            if int(env_observation_size) != observation_size:
                problems.append(
                    "observation_size "
                    f"checkpoint={observation_size} env={int(env_observation_size)}"
                )
        expected_version = str(info.get("action_mask_version", "") or "")
        checkpoint_version = str(getattr(config, "action_mask_version", "") or "")
        if checkpoint_version and expected_version and checkpoint_version != expected_version:
            problems.append(
                "action_mask_version "
                f"checkpoint={checkpoint_version!r} env={expected_version!r}"
            )
        checkpoint_static = getattr(policy, "static_action_features", None)
        if checkpoint_static is None:
            checkpoint_static = getattr(policy, "action_features", None)
        if checkpoint_static is not None:
            if hasattr(checkpoint_static, "detach"):
                checkpoint_static = checkpoint_static.detach().cpu().numpy()
            checkpoint_hash = _array_sha256(np.asarray(checkpoint_static, dtype=np.float32))
            env_hash = _array_sha256(build_action_feature_table(env))
            if checkpoint_hash != env_hash:
                problems.append(
                    "static_action_features_sha256 "
                    f"checkpoint={checkpoint_hash} env={env_hash}"
                )
        context_size = int(getattr(policy, "context_action_feature_size", 0))
        if context_size:
            context = build_action_context_feature_table(env, info)
            if int(context.shape[-1]) != context_size:
                problems.append(
                    "context_action_feature_size "
                    f"checkpoint={context_size} env={int(context.shape[-1])}"
                )
    finally:
        env.close()
    if problems:
        raise ValueError(
            f"policy {getattr(policy, 'name', type(policy).__name__)} is not compatible "
            "with evaluation env: "
            + "; ".join(problems)
        )


def specs_from_names(names: str, *, weight: float = 1.0) -> list[PolicySpec]:
    return [
        PolicySpec(kind=name.strip(), weight=weight)
        for name in names.split(",")
        if name.strip()
    ]
