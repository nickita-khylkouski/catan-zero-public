from __future__ import annotations

import copy
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
import math
from pathlib import Path
from typing import Any

import numpy as np

from catan_zero.rl.action_features import (
    CONTEXT_ACTION_FEATURE_SIZE,
    build_action_context_feature_table,
)
from catan_zero.rl.multiagent_env import ColonistMultiAgentConfig, ColonistMultiAgentEnv
from catan_zero.rl.self_play import (
    Policy,
    StepSample,
    _catanatron_value_score,
    _phase_from_info,
    _scoreboard_values,
)


def _is_candidate_architecture(architecture: str) -> bool:
    return architecture in {"candidate", "graph_history_candidate"}


def _policy_action_context_feature_table(
    policy: Any,
    env: ColonistMultiAgentEnv,
    info: dict[str, Any],
) -> np.ndarray:
    """Build context with an entity policy's checkpoint-bound adapter.

    Non-entity policies have no adapter binding and retain the historical
    default call exactly.  Entity policies must store the same context that
    produced their behavior log-probability so PPO can recompute a ratio of
    one before any learner update.
    """

    adapter_version = getattr(policy, "entity_feature_adapter_version", None)
    if adapter_version is None:
        return build_action_context_feature_table(env, info)
    return build_action_context_feature_table(
        env,
        info,
        entity_feature_adapter_version=str(adapter_version),
    )


def _resolve_device(device: str | None):
    import torch

    requested = (device or "cpu").strip().lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    resolved = torch.device(requested)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device {device!r}, but CUDA is not available")
    return resolved


def _behavior_policy_logits(logits, temperature: float, *, valid_mask=None):
    """Return the exact bounded distribution surface shared by actor and learner."""

    import torch

    behavior_logits = torch.clamp(
        logits / max(float(temperature), 1.0e-6),
        min=-50.0,
        max=50.0,
    )
    if valid_mask is not None:
        if tuple(valid_mask.shape) != tuple(logits.shape):
            raise ValueError(
                "behavior-logit valid mask shape differs from logits: "
                f"{tuple(valid_mask.shape)} != {tuple(logits.shape)}"
            )
        behavior_logits = behavior_logits.masked_fill(~valid_mask, -1.0e9)
    return behavior_logits


@dataclass(slots=True)
class PPOTrajectory:
    samples: list[StepSample]
    returns: list[float]
    advantages: list[float]
    old_log_probs: list[float]
    old_values: list[float]
    old_action_probs: list[np.ndarray]
    shaped_rewards: list[float]
    old_q_values: list[float] | None = None
    old_action_q_values: list[np.ndarray] | None = None
    # Real per-step environment reward (incl. the terminal win/loss reward) for the
    # training seat, aligned with ``samples``. Fed to V-trace in the learner; the actor-side
    # GAE folds the same signal into ``returns``. Default-empty so older shards still unpickle.
    rewards: list[float] = field(default_factory=list)
    # Per-trajectory rollout metadata. Newer actors fill this so learner/audit tools can
    # break down PPO signal by opponent mix; older shards load with empty defaults.
    training_seats: tuple[str, ...] = ()
    opponent_names: dict[str, str] = field(default_factory=dict)
    # Truncated-episode bootstrap: when the game hit ``max_decisions`` (truncated, not
    # terminal), ``truncated`` is True and ``bootstrap_value`` holds the training seat's
    # critic value at the cutoff state so V-trace can bootstrap instead of treating it as
    # a terminal 0. For genuinely terminal games both stay at their 0.0/False defaults.
    bootstrap_value: float = 0.0
    truncated: bool = False
    # Exact learner-seat decision state at a time-limit cutoff. The distributed
    # learner uses it to recompute a bootstrap under the same current snapshot
    # as the in-trajectory V-trace values. Older shards leave this slot unset.
    bootstrap_sample: StepSample | None = None
    # Learner-snapshot references used by PPO after V-trace has already corrected from the
    # actor behavior policy. Empty (or an unset slot in an older pickle) falls back to the
    # actor-side old_* evidence, preserving the ordinary non-V-trace PPO contract.
    ppo_reference_log_probs: list[float] = field(default_factory=list)
    ppo_reference_values: list[float] = field(default_factory=list)


def _ppo_reference_array(
    trajectories: list[PPOTrajectory],
    *,
    reference_attr: str,
    fallback_attr: str,
) -> np.ndarray:
    """Flatten an optional per-trajectory PPO reference, failing closed on misalignment."""

    flattened: list[float] = []
    for trajectory in trajectories:
        reference = getattr(trajectory, reference_attr, None)
        values = reference if reference is not None and len(reference) > 0 else getattr(
            trajectory,
            fallback_attr,
        )
        expected = len(trajectory.samples)
        if len(values) != expected:
            raise ValueError(
                f"PPOTrajectory.{reference_attr or fallback_attr} must align with samples "
                f"({len(values)} != {expected})"
            )
        flattened.extend(float(value) for value in values)
    return np.asarray(flattened, dtype=np.float32)


def _validate_ppo_trajectory_alignment(
    trajectories: list[PPOTrajectory],
) -> None:
    """Fail closed before flattening per-step policy/value targets."""

    for trajectory_index, trajectory in enumerate(trajectories):
        expected = len(trajectory.samples)
        for field_name in ("returns", "advantages"):
            values = getattr(trajectory, field_name, None)
            actual = len(values) if values is not None else 0
            if actual != expected:
                raise ValueError(
                    f"PPOTrajectory.{field_name} must align with samples for "
                    f"trajectory {trajectory_index} ({actual} != {expected})"
                )


def _trajectory_opponent_mix(trajectory: PPOTrajectory) -> str:
    names = getattr(trajectory, "opponent_names", None) or {}
    if isinstance(names, dict) and names:
        mix = ",".join(sorted(set(str(name) for name in names.values())))
        if mix:
            return mix
    return "unknown"


def _advantage_group_labels(trajectories: list[PPOTrajectory]) -> list[str]:
    labels: list[str] = []
    for trajectory in trajectories:
        group = _trajectory_opponent_mix(trajectory)
        labels.extend([group] * len(trajectory.samples))
    return labels


def _normalize_advantages_by_group(
    advantages: np.ndarray,
    group_labels: list[str],
    *,
    mode: str,
    eligible_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Normalize each group using only rows that can contribute policy gradient.

    The same affine transform is applied to every row so callers can retain
    policy-inactive rows for value training without letting those rows determine
    the policy normalization statistics.
    """
    mode = str(mode or "global").strip().lower()
    if mode in {"global", "standard", "default"}:
        return advantages.copy(), 0
    if mode in {"none", "off", "raw"}:
        return advantages.copy(), 0
    if mode not in {"per_opponent", "opponent", "opponent_mix"}:
        raise ValueError(
            "advantage_normalization must be one of: global, per_opponent, none"
        )
    if len(group_labels) != len(advantages):
        raise ValueError(
            "advantage group label count does not match advantage count: "
            f"{len(group_labels)} vs {len(advantages)}"
        )
    if eligible_mask is None:
        eligible = np.ones(len(advantages), dtype=np.bool_)
    else:
        eligible = np.asarray(eligible_mask, dtype=np.bool_)
        if eligible.shape != advantages.shape:
            raise ValueError(
                "advantage eligibility mask does not match advantage shape: "
                f"{eligible.shape} vs {advantages.shape}"
            )
    normalized = advantages.astype(np.float32, copy=True)
    groups = sorted(set(str(label) for label in group_labels))
    for group in groups:
        mask = np.asarray([str(label) == group for label in group_labels], dtype=bool)
        stats_mask = mask & eligible & np.isfinite(normalized)
        if not np.any(stats_mask):
            continue
        stats_values = normalized[stats_mask]
        mean = float(stats_values.mean())
        std = float(stats_values.std())
        values = normalized[mask]
        finite = np.isfinite(values)
        if std > 1.0e-8:
            values[finite] = (values[finite] - mean) / std
        else:
            values[finite] = values[finite] - mean
        normalized[mask] = values
    return normalized, len(groups)


def _parse_advantage_group_weights(weights: Any | None) -> dict[str, float]:
    if weights is None:
        return {}
    if isinstance(weights, dict):
        items = weights.items()
    else:
        text = str(weights).strip()
        if not text:
            return {}
        items = []
        for item in text.split(","):
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            items.append((key, value))
    parsed: dict[str, float] = {}
    for key, value in items:
        key = str(key).strip()
        if not key:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            parsed[key] = numeric
    return parsed


def _advantage_group_weight_for_label(label: str, weights: dict[str, float]) -> float:
    if not weights:
        return 1.0
    label = str(label)
    if label in weights:
        return float(weights[label])
    parts = {part.strip() for part in label.split(",") if part.strip()}
    matched = [float(value) for key, value in weights.items() if key in parts]
    if matched:
        return float(sum(matched) / len(matched))
    return float(weights.get("default", 1.0))


def _apply_advantage_group_weights(
    advantages: np.ndarray,
    group_labels: list[str],
    weights: Any | None,
) -> tuple[np.ndarray, int, float]:
    parsed = _parse_advantage_group_weights(weights)
    if not parsed:
        return advantages.copy(), 0, 1.0
    if len(group_labels) != len(advantages):
        raise ValueError(
            "advantage group label count does not match advantage count: "
            f"{len(group_labels)} vs {len(advantages)}"
        )
    scalars = np.asarray(
        [_advantage_group_weight_for_label(label, parsed) for label in group_labels],
        dtype=np.float32,
    )
    scalars = np.where(np.isfinite(scalars), scalars, 1.0)
    scalars = np.clip(scalars, 0.0, 10.0)
    return advantages.astype(np.float32, copy=True) * scalars, len(parsed), float(scalars.mean())


class TorchPPOPolicy:
    name = "torch_ppo"

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        *,
        hidden_size: int = 256,
        seed: int = 0,
        architecture: str = "flat",
        action_features: np.ndarray | None = None,
        use_action_id_embedding: bool = True,
        context_action_feature_size: int = 0,
        device: str | None = None,
    ) -> None:
        import torch
        from torch import nn

        torch.manual_seed(seed)
        if architecture not in ("flat", "candidate", "graph_history_candidate"):
            raise ValueError(f"unsupported PPO architecture: {architecture}")
        self.observation_size = observation_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.architecture = architecture
        self.use_action_id_embedding = bool(use_action_id_embedding)
        self.context_action_feature_size = int(context_action_feature_size)
        self.device = _resolve_device(device)
        self.model = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        ).to(self.device)
        self.action_features = None
        self.action_encoder = None
        self.action_id_embedding = None
        self.action_bias = None
        self.q_head = None
        self.q_state = None
        self.q_action_encoder = None
        self.q_action_bias = None
        if _is_candidate_architecture(architecture):
            if action_features is None:
                raise ValueError(f"{architecture} architecture requires action_features")
            action_features = np.asarray(action_features, dtype=np.float32)
            if action_features.shape[0] != action_size:
                raise ValueError("action_features row count must match action_size")
            self.action_features = torch.as_tensor(
                action_features,
                dtype=torch.float32,
                device=self.device,
            )
            feature_size = int(action_features.shape[1]) + self.context_action_feature_size
            self.actor = nn.Linear(hidden_size, hidden_size).to(self.device)
            self.action_encoder = nn.Sequential(
                nn.Linear(feature_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
            ).to(self.device)
            if self.use_action_id_embedding:
                self.action_id_embedding = nn.Embedding(action_size, hidden_size).to(
                    self.device
                )
            self.action_bias = nn.Linear(feature_size, 1).to(self.device)
            self.q_state = nn.Linear(hidden_size, hidden_size).to(self.device)
            self.q_action_encoder = nn.Sequential(
                nn.Linear(feature_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
            ).to(self.device)
            self.q_action_bias = nn.Linear(feature_size, 1).to(self.device)
        else:
            self.actor = nn.Linear(hidden_size, action_size).to(self.device)
            self.q_head = nn.Linear(hidden_size, action_size).to(self.device)
        self.critic = nn.Linear(hidden_size, 1).to(self.device)

    def select_action(
        self,
        env: ColonistMultiAgentEnv,
        observation: np.ndarray,
        info: dict[str, Any],
        rng: np.random.Generator,
        *,
        training: bool = False,
    ) -> int:
        import torch

        valid_actions = tuple(int(action) for action in info["valid_actions"])
        with torch.no_grad():
            obs = self._observation_tensor(observation)
            logits, _ = self.forward(obs, self._action_context_tensor(env, info))
            masked = _masked_logits(logits, [valid_actions], self.action_size)
            if training:
                dist = torch.distributions.Categorical(logits=masked)
                return int(dist.sample().item())
            return int(torch.argmax(masked, dim=-1).item())

    def sample_action(
        self,
        observation: np.ndarray,
        valid_actions: tuple[int, ...],
        action_context_features: np.ndarray | None = None,
    ) -> tuple[int, float]:
        import torch

        with torch.no_grad():
            obs = self._observation_tensor(observation)
            logits, _ = self.forward(
                obs,
                self._context_tensor_from_array(action_context_features),
            )
            masked = _masked_logits(logits, [valid_actions], self.action_size)
            dist = torch.distributions.Categorical(logits=masked)
            action = dist.sample()
            return int(action.item()), float(dist.log_prob(action).item())

    def sample_action_value(
        self,
        observation: np.ndarray,
        valid_actions: tuple[int, ...],
        action_context_features: np.ndarray | None = None,
    ) -> tuple[int, float, float, np.ndarray]:
        import torch

        with torch.no_grad():
            obs = self._observation_tensor(observation)
            logits, value = self.forward(
                obs,
                self._context_tensor_from_array(action_context_features),
            )
            masked = _masked_logits(logits, [valid_actions], self.action_size)
            dist = torch.distributions.Categorical(logits=masked)
            action = dist.sample()
            probs = torch.softmax(masked.squeeze(0), dim=-1)
            valid_probs = probs[
                torch.as_tensor(valid_actions, dtype=torch.long, device=self.device)
            ]
            return (
                int(action.item()),
                float(dist.log_prob(action).item()),
                float(value.item()),
                valid_probs.detach().cpu().numpy().astype(np.float32),
            )

    def sample_action_value_q(
        self,
        observation: np.ndarray,
        valid_actions: tuple[int, ...],
        action_context_features: np.ndarray | None = None,
    ) -> tuple[int, float, float, float, np.ndarray, np.ndarray]:
        import torch

        with torch.no_grad():
            obs = self._observation_tensor(observation)
            context = self._context_tensor_from_array(action_context_features)
            logits, value = self.forward(obs, context)
            q_values = self.q_values(obs, context)
            masked = _masked_logits(logits, [valid_actions], self.action_size)
            dist = torch.distributions.Categorical(logits=masked)
            action = dist.sample()
            probs = torch.softmax(masked.squeeze(0), dim=-1)
            valid_probs = probs[
                torch.as_tensor(valid_actions, dtype=torch.long, device=self.device)
            ]
            valid_q_values = q_values.squeeze(0)[
                torch.as_tensor(valid_actions, dtype=torch.long, device=self.device)
            ]
            action_index = int(action.item())
            return (
            action_index,
            float(dist.log_prob(action).item()),
            float(value.item()),
            float(q_values[0, action_index].item()),
            valid_probs.detach().cpu().numpy().astype(np.float32),
            valid_q_values.detach().cpu().numpy().astype(np.float32),
        )

    def _observation_tensor(self, observation: np.ndarray):
        import torch

        return torch.as_tensor(
            _resize_observation(
                _normalize_observation(observation),
                observation_size=self.observation_size,
            ),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

    def forward(self, observations, action_context_features=None):
        import torch

        features = self.model(observations)
        if _is_candidate_architecture(self.architecture):
            assert self.action_features is not None
            assert self.action_encoder is not None
            assert self.action_bias is not None
            state_features = self.actor(features)
            action_ids = torch.arange(self.action_size, device=self.device)
            combined_features = self._combined_action_features(
                batch_size=int(observations.shape[0]),
                action_context_features=action_context_features,
            )
            flat_features = combined_features.reshape(
                int(observations.shape[0]) * self.action_size,
                -1,
            )
            action_features = self.action_encoder(flat_features).reshape(
                int(observations.shape[0]),
                self.action_size,
                self.hidden_size,
            )
            if self.action_id_embedding is not None:
                action_features = action_features + self.action_id_embedding(
                    action_ids,
                ).unsqueeze(0)
            logits = (
                state_features.unsqueeze(1) * action_features
            ).sum(dim=-1) / math.sqrt(self.hidden_size)
            logits = logits + self.action_bias(flat_features).reshape(
                int(observations.shape[0]),
                self.action_size,
            )
        else:
            logits = self.actor(features)
        return logits, self.critic(features).squeeze(-1)

    def q_values(self, observations, action_context_features=None):
        features = self.model(observations)
        if _is_candidate_architecture(self.architecture):
            assert self.q_state is not None
            assert self.q_action_encoder is not None
            assert self.q_action_bias is not None
            state_features = self.q_state(features)
            combined_features = self._combined_action_features(
                batch_size=int(observations.shape[0]),
                action_context_features=action_context_features,
            )
            flat_features = combined_features.reshape(
                int(observations.shape[0]) * self.action_size,
                -1,
            )
            action_features = self.q_action_encoder(flat_features).reshape(
                int(observations.shape[0]),
                self.action_size,
                self.hidden_size,
            )
            values = (
                state_features.unsqueeze(1) * action_features
            ).sum(dim=-1) / math.sqrt(self.hidden_size)
            return values + self.q_action_bias(flat_features).reshape(
                int(observations.shape[0]),
                self.action_size,
            )
        assert self.q_head is not None
        return self.q_head(features)

    def _combined_action_features(self, *, batch_size: int, action_context_features):
        import torch

        assert self.action_features is not None
        static = self.action_features.unsqueeze(0).expand(batch_size, -1, -1)
        if self.context_action_feature_size <= 0:
            return static
        if action_context_features is None:
            context = torch.zeros(
                batch_size,
                self.action_size,
                self.context_action_feature_size,
                dtype=torch.float32,
                device=self.device,
            )
        else:
            context = torch.as_tensor(
                action_context_features,
                dtype=torch.float32,
                device=self.device,
            )
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context.shape[0] != batch_size or context.shape[1] != self.action_size:
                raise ValueError("action context features must be [batch, actions, features]")
            if context.shape[2] != self.context_action_feature_size:
                context = _resize_context_tensor(
                    context,
                    feature_size=self.context_action_feature_size,
                )
        return torch.cat((static, context), dim=-1)

    def _action_context_tensor(self, env: ColonistMultiAgentEnv, info: dict[str, Any]):
        if (
            not _is_candidate_architecture(self.architecture)
            or self.context_action_feature_size <= 0
        ):
            return None
        return self._context_tensor_from_array(
            build_action_context_feature_table(env, info),
        )

    def _context_tensor_from_array(self, value: np.ndarray | None):
        import torch

        if (
            not _is_candidate_architecture(self.architecture)
            or self.context_action_feature_size <= 0
        ):
            return None
        if value is None:
            return None
        return torch.as_tensor(
            value,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

    def save(self, path: str | Path) -> None:
        import torch

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "observation_size": self.observation_size,
                "action_size": self.action_size,
                "hidden_size": self.hidden_size,
                "architecture": self.architecture,
                "use_action_id_embedding": self.use_action_id_embedding,
                "context_action_feature_size": self.context_action_feature_size,
                "action_features": (
                    self.action_features.detach().cpu()
                    if self.action_features is not None
                    else None
                ),
                "model": self.model.state_dict(),
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "q_head": self.q_head.state_dict() if self.q_head is not None else None,
                "q_state": (
                    self.q_state.state_dict() if self.q_state is not None else None
                ),
                "q_action_encoder": (
                    self.q_action_encoder.state_dict()
                    if self.q_action_encoder is not None
                    else None
                ),
                "q_action_bias": (
                    self.q_action_bias.state_dict()
                    if self.q_action_bias is not None
                    else None
                ),
                "action_encoder": (
                    self.action_encoder.state_dict()
                    if self.action_encoder is not None
                    else None
                ),
                "action_id_embedding": (
                    self.action_id_embedding.state_dict()
                    if self.action_id_embedding is not None
                    else None
                ),
                "action_bias": (
                    self.action_bias.state_dict()
                    if self.action_bias is not None
                    else None
                ),
            },
            output,
        )

    def clone_frozen(self) -> TorchPPOPolicy:
        """Return an eval-mode copy suitable for historical league opponents."""
        action_features = (
            self.action_features.detach().cpu().numpy()
            if self.action_features is not None
            else None
        )
        policy = type(self)(
            self.observation_size,
            self.action_size,
            hidden_size=self.hidden_size,
            architecture=self.architecture,
            action_features=action_features,
            use_action_id_embedding=self.use_action_id_embedding,
            context_action_feature_size=self.context_action_feature_size,
            device=str(self.device),
        )
        policy.model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        policy.actor.load_state_dict(copy.deepcopy(self.actor.state_dict()))
        policy.critic.load_state_dict(copy.deepcopy(self.critic.state_dict()))
        if policy.q_head is not None and self.q_head is not None:
            policy.q_head.load_state_dict(copy.deepcopy(self.q_head.state_dict()))
        if policy.q_state is not None and self.q_state is not None:
            policy.q_state.load_state_dict(copy.deepcopy(self.q_state.state_dict()))
        if (
            policy.q_action_encoder is not None
            and self.q_action_encoder is not None
        ):
            policy.q_action_encoder.load_state_dict(
                copy.deepcopy(self.q_action_encoder.state_dict())
            )
        if policy.q_action_bias is not None and self.q_action_bias is not None:
            policy.q_action_bias.load_state_dict(
                copy.deepcopy(self.q_action_bias.state_dict())
            )
        if policy.action_encoder is not None and self.action_encoder is not None:
            policy.action_encoder.load_state_dict(
                copy.deepcopy(self.action_encoder.state_dict())
            )
        if (
            policy.action_id_embedding is not None
            and self.action_id_embedding is not None
        ):
            policy.action_id_embedding.load_state_dict(
                copy.deepcopy(self.action_id_embedding.state_dict())
            )
        if policy.action_bias is not None and self.action_bias is not None:
            policy.action_bias.load_state_dict(
                copy.deepcopy(self.action_bias.state_dict())
            )
        policy.model.eval()
        policy.actor.eval()
        policy.critic.eval()
        if policy.q_head is not None:
            policy.q_head.eval()
        if policy.q_state is not None:
            policy.q_state.eval()
        if policy.q_action_encoder is not None:
            policy.q_action_encoder.eval()
        if policy.q_action_bias is not None:
            policy.q_action_bias.eval()
        if policy.action_encoder is not None:
            policy.action_encoder.eval()
        if policy.action_id_embedding is not None:
            policy.action_id_embedding.eval()
        if policy.action_bias is not None:
            policy.action_bias.eval()
        return policy

    @classmethod
    def load(cls, path: str | Path, *, device: str | None = None) -> TorchPPOPolicy:
        import torch

        load_device = _resolve_device(device)
        try:
            data = torch.load(Path(path), map_location=load_device, weights_only=False)
        except TypeError:  # Older torch does not expose weights_only.
            data = torch.load(Path(path), map_location=load_device)
        action_features = data.get("action_features")
        if hasattr(action_features, "detach"):
            action_features = action_features.detach().cpu().numpy()
        policy = cls(
            int(data["observation_size"]),
            int(data["action_size"]),
            hidden_size=int(data["hidden_size"]),
            architecture=str(data.get("architecture", "flat")),
            action_features=action_features,
            use_action_id_embedding=bool(data.get("use_action_id_embedding", True)),
            context_action_feature_size=int(data.get("context_action_feature_size", 0)),
            device=str(load_device),
        )
        policy.model.load_state_dict(data["model"])
        policy.actor.load_state_dict(data["actor"])
        policy.critic.load_state_dict(data["critic"])
        if policy.q_head is not None and data.get("q_head") is not None:
            policy.q_head.load_state_dict(data["q_head"])
        if policy.q_state is not None and data.get("q_state") is not None:
            policy.q_state.load_state_dict(data["q_state"])
        if (
            policy.q_action_encoder is not None
            and data.get("q_action_encoder") is not None
        ):
            policy.q_action_encoder.load_state_dict(data["q_action_encoder"])
        if policy.q_action_bias is not None and data.get("q_action_bias") is not None:
            policy.q_action_bias.load_state_dict(data["q_action_bias"])
        if policy.action_encoder is not None and data.get("action_encoder") is not None:
            policy.action_encoder.load_state_dict(data["action_encoder"])
        if (
            policy.action_id_embedding is not None
            and data.get("action_id_embedding") is not None
        ):
            policy.action_id_embedding.load_state_dict(data["action_id_embedding"])
        if policy.action_bias is not None and data.get("action_bias") is not None:
            policy.action_bias.load_state_dict(data["action_bias"])
        policy.model.eval()
        policy.actor.eval()
        policy.critic.eval()
        if policy.q_head is not None:
            policy.q_head.eval()
        if policy.q_state is not None:
            policy.q_state.eval()
        if policy.q_action_encoder is not None:
            policy.q_action_encoder.eval()
        if policy.q_action_bias is not None:
            policy.q_action_bias.eval()
        if policy.action_encoder is not None:
            policy.action_encoder.eval()
        if policy.action_id_embedding is not None:
            policy.action_id_embedding.eval()
        if policy.action_bias is not None:
            policy.action_bias.eval()
        return policy


def create_ppo_policy(
    *,
    config: ColonistMultiAgentConfig | None = None,
    seed: int = 0,
    hidden_size: int = 256,
    architecture: str = "flat",
    use_action_id_embedding: bool = True,
    device: str | None = None,
) -> TorchPPOPolicy:
    if architecture == "graph_history_candidate":
        if config is None:
            config = ColonistMultiAgentConfig(use_graph_history_features=True)
        elif not config.use_graph_history_features:
            config = replace(config, use_graph_history_features=True)
    env = ColonistMultiAgentEnv(config or ColonistMultiAgentConfig())
    try:
        observations, _ = env.reset(seed=seed)
        observation_size = len(next(iter(observations.values())))
        action_features = (
            build_action_feature_table(env)
            if _is_candidate_architecture(architecture)
            else None
        )
        return TorchPPOPolicy(
            observation_size,
            env.action_space.n,
            hidden_size=hidden_size,
            seed=seed,
            architecture=architecture,
            action_features=action_features,
            use_action_id_embedding=use_action_id_embedding,
            context_action_feature_size=(
                CONTEXT_ACTION_FEATURE_SIZE
                if _is_candidate_architecture(architecture)
                else 0
            ),
            device=device,
        )
    finally:
        env.close()


def _set_policy_inference_mode(policy: Any) -> None:
    """Put every known policy module in the deterministic actor serving mode."""

    seen: set[int] = set()
    for name in (
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
    ):
        module = getattr(policy, name, None)
        if module is None or id(module) in seen:
            continue
        seen.add(id(module))
        eval_method = getattr(module, "eval", None)
        if callable(eval_method):
            eval_method()


def collect_ppo_episode(
    policy: TorchPPOPolicy,
    opponents: dict[str, Policy],
    *,
    seed: int,
    config: ColonistMultiAgentConfig,
    max_decisions: int,
    rng: np.random.Generator,
    training_seats: set[str],
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
    value_shaping_coef: float = 0.0,
    value_shaping_scale: float = 100.0,
    value_shaping_opponent_penalty: float = 0.05,
    action_temperature: float = 1.0,
) -> PPOTrajectory:
    _set_policy_inference_mode(policy)
    env = ColonistMultiAgentEnv(config)
    samples: list[StepSample] = []
    old_log_probs: list[float] = []
    old_values: list[float] = []
    old_q_values: list[float] = []
    old_action_probs: list[np.ndarray] = []
    old_action_q_values: list[np.ndarray] = []
    shaped_rewards: list[float] = []
    players: list[str] = []
    shaping_value_fn = (
        _make_catanatron_value_fn() if value_shaping_coef > 0.0 else None
    )
    opponent_names = {
        str(seat): str(getattr(opponent, "name", opponent.__class__.__name__))
        for seat, opponent in opponents.items()
    }
    try:
        observations, info = env.reset(seed=seed)
        rewards = {name: 0.0 for name in env.player_names}
        terminated = False
        truncated = False
        decisions = 0
        # Stop only at a learner-seat decision boundary. If the nominal cap lands
        # on an opponent prompt, drain opponent actions without recording another
        # learner decision; valuing a non-current learner seat against the
        # opponent's legal-action set is out-of-distribution for entity policies.
        while not (terminated or truncated):
            player = info["current_player"]
            if decisions >= max_decisions and player in training_seats:
                break
            observation = np.asarray(observations[player], dtype=np.float64)
            valid_actions = tuple(int(action) for action in info["valid_actions"])
            action_context_features = _policy_action_context_feature_table(
                policy,
                env,
                info,
            )
            if player in training_seats:
                actor_color = env.current_player_color()
                before_score = _finite_catanatron_score(
                    env,
                    actor_color,
                    value_fn=shaping_value_fn,
                    opponent_penalty=value_shaping_opponent_penalty,
                )
                entity_features = None
                entity_sampler = getattr(policy, "sample_action_value_q_from_env", None)
                if callable(entity_sampler):
                    (
                        action,
                        log_prob,
                        value,
                        q_value,
                        action_probs,
                        action_q_values,
                        entity_features,
                    ) = entity_sampler(
                        env,
                        info,
                        rng,
                        training=True,
                        action_temperature=action_temperature,
                    )
                else:
                    (
                        action,
                        log_prob,
                        value,
                        q_value,
                        action_probs,
                        action_q_values,
                    ) = policy.sample_action_value_q(
                        observation,
                        valid_actions,
                        action_context_features,
                    )
                samples.append(
                    StepSample(
                        observation=observation.copy(),
                        valid_actions=valid_actions,
                        action=action,
                        player=player,
                        action_context_features=action_context_features,
                        entity_features=entity_features,
                        phase=_phase_from_info(info),
                    )
                )
                old_log_probs.append(log_prob)
                old_values.append(value)
                old_q_values.append(q_value)
                old_action_probs.append(action_probs)
                old_action_q_values.append(action_q_values)
                players.append(player)
            else:
                actor_color = None
                before_score = None
                action = opponents[player].select_action(
                    env,
                    observation,
                    info,
                    rng,
                    training=False,
                )
            observations, rewards, terminated, truncated, info = env.step(action)
            if player in training_seats:
                after_score = (
                    None
                    if terminated or truncated
                    else _finite_catanatron_score(
                        env,
                        actor_color,
                        value_fn=shaping_value_fn,
                        opponent_penalty=value_shaping_opponent_penalty,
                    )
                )
                shaped_rewards.append(
                    _clipped_value_delta_reward(
                        before_score,
                        after_score,
                        coef=value_shaping_coef,
                        scale=value_shaping_scale,
                    )
                )
            decisions += 1
        bootstrap_values = None
        bootstrap_sample = None
        time_limit_truncated = (
            not terminated
            and (
                decisions >= max_decisions
                or (
                    truncated
                    and env.invalid_actions_count <= env.config.max_invalid_actions
                )
            )
        )
        if not terminated and decisions >= max_decisions:
            truncated = True
        if time_limit_truncated:
            rewards = {name: 0.0 for name in env.player_names}
            bootstrap_values, bootstrap_sample = _bootstrap_values(
                policy,
                observations,
                set(players),
                env=env,
                info=info,
            )
        returns, advantages = _gae_returns(
            players,
            rewards,
            old_values,
            shaped_rewards,
            gamma=gamma,
            gae_lambda=gae_lambda,
            bootstrap_values=bootstrap_values,
        )
        # Build the REAL per-step env reward stream consumed by V-trace, aligned with
        # ``samples``. This mirrors exactly what feeds the actor-side GAE in ``_gae_returns``:
        # the shaped per-step reward at every step PLUS the terminal env reward
        # (``rewards[player]``) folded into each player's FINAL decision. On truncation the env
        # rewards are zeroed above and value is carried via ``bootstrap_value`` instead.
        step_rewards = [float(value) for value in shaped_rewards]
        seen_player: set[str] = set()
        for idx in range(len(players) - 1, -1, -1):
            player = players[idx]
            if player not in seen_player:
                step_rewards[idx] += float(rewards.get(player, 0.0))
                seen_player.add(player)
        # Truncated-episode bootstrap value for the single training seat in this trajectory.
        trajectory_bootstrap_value = 0.0
        if bootstrap_values:
            seat_players = [player for player in players if player in bootstrap_values]
            if seat_players:
                trajectory_bootstrap_value = float(bootstrap_values[seat_players[-1]])
        return PPOTrajectory(
            samples=samples,
            returns=returns,
            advantages=advantages,
            old_log_probs=old_log_probs,
            old_values=old_values,
            old_q_values=old_q_values,
            old_action_probs=old_action_probs,
            old_action_q_values=old_action_q_values,
            shaped_rewards=shaped_rewards,
            rewards=step_rewards,
            training_seats=tuple(sorted(str(seat) for seat in training_seats)),
            opponent_names=opponent_names,
            bootstrap_value=trajectory_bootstrap_value,
            truncated=bool(time_limit_truncated),
            bootstrap_sample=bootstrap_sample,
        )
    finally:
        env.close()


def _bootstrap_values(
    policy: TorchPPOPolicy,
    observations: dict[str, Any],
    players: set[str],
    *,
    env: ColonistMultiAgentEnv | None = None,
    info: dict[str, Any] | None = None,
) -> tuple[dict[str, float], StepSample | None]:
    import torch

    values: dict[str, float] = {}
    entity_outputs_from_env = getattr(policy, "_legal_outputs_from_env", None)
    if callable(entity_outputs_from_env) and env is not None and info is not None:
        current_player = str(info.get("current_player") or "")
        valid_actions = tuple(int(action) for action in info.get("valid_actions", ()))
        if current_player in players and current_player in observations and valid_actions:
            with torch.no_grad():
                outputs, raw_entity_features, _legal_context = entity_outputs_from_env(
                    env,
                    info,
                    valid_actions,
                    return_q=False,
                )
            value = outputs["value"].reshape(-1)[0]
            values[current_player] = float(value.item())
            entity_features = {
                key: np.asarray(item).copy()
                for key, item in raw_entity_features.items()
                if key != "schema"
            }
            return values, StepSample(
                observation=np.asarray(observations[current_player], dtype=np.float64).copy(),
                valid_actions=valid_actions,
                action=valid_actions[0],
                player=current_player,
                action_context_features=_policy_action_context_feature_table(
                    policy,
                    env,
                    info,
                ),
                entity_features=entity_features,
                phase=_phase_from_info(info),
            )

    observation_tensor = getattr(policy, "_observation_tensor", None)
    if not callable(observation_tensor):
        return values, None
    with torch.no_grad():
        for player in players:
            if player not in observations:
                continue
            obs = observation_tensor(observations[player])
            _, value = policy.forward(obs, None)
            values[player] = float(value.item())
    current_player = str((info or {}).get("current_player") or "")
    valid_actions = tuple(int(action) for action in (info or {}).get("valid_actions", ()))
    sample = None
    if (
        current_player in players
        and current_player in observations
        and valid_actions
    ):
        sample = StepSample(
            observation=np.asarray(observations[current_player], dtype=np.float64).copy(),
            valid_actions=valid_actions,
            action=valid_actions[0],
            player=current_player,
            action_context_features=(
                _policy_action_context_feature_table(policy, env, info)
                if env is not None and info is not None
                else None
            ),
            phase=_phase_from_info(info or {}),
        )
    return values, sample


def collect_dagger_episode(
    policy: TorchPPOPolicy,
    teacher: Policy,
    opponents: dict[str, Policy],
    *,
    seed: int,
    config: ColonistMultiAgentConfig,
    max_decisions: int,
    rng: np.random.Generator,
    training_seats: set[str],
    gamma: float = 1.0,
) -> tuple[list[StepSample], list[float]]:
    """Collect teacher labels on states visited by the current policy.

    Teacher self-play only labels states that the teacher itself reaches.
    DAgger-style correction labels the learner's own state distribution while
    still executing learner actions, which targets PPO drift directly.
    """
    env = ColonistMultiAgentEnv(config)
    samples: list[StepSample] = []
    players: list[str] = []
    try:
        observations, info = env.reset(seed=seed)
        rewards = {name: 0.0 for name in env.player_names}
        terminated = False
        truncated = False
        decisions = 0
        while not (terminated or truncated) and decisions < max_decisions:
            player = info["current_player"]
            observation = np.asarray(observations[player], dtype=np.float64)
            valid_actions = tuple(int(action) for action in info["valid_actions"])
            action_context_features = _policy_action_context_feature_table(
                policy,
                env,
                info,
            )
            if player in training_seats:
                teacher_action = teacher.select_action(
                    env,
                    observation,
                    info,
                    rng,
                    training=False,
                )
                target_policy_fn = getattr(teacher, "target_policy", None)
                target_policy = (
                    target_policy_fn(env, info, rng)
                    if callable(target_policy_fn)
                    else None
                )
                target_scores_fn = getattr(teacher, "target_scores", None)
                target_scores = (
                    target_scores_fn(env, info, rng)
                    if callable(target_scores_fn)
                    else None
                )
                samples.append(
                    StepSample(
                        observation=observation.copy(),
                        valid_actions=valid_actions,
                        action=int(teacher_action),
                        player=player,
                        action_context_features=action_context_features,
                        phase=_phase_from_info(info),
                        target_policy=target_policy,
                        target_scores=target_scores,
                    )
                )
                players.append(player)
                action = policy.select_action(
                    env,
                    observation,
                    info,
                    rng,
                    training=True,
                )
            else:
                action = opponents[player].select_action(
                    env,
                    observation,
                    info,
                    rng,
                    training=False,
                )
            observations, rewards, terminated, truncated, info = env.step(action)
            decisions += 1
        if not terminated and decisions >= max_decisions:
            rewards = _scoreboard_rewards(env)
        returns = _discounted_terminal_returns(players, rewards, gamma=gamma)
        return samples, returns
    finally:
        env.close()


def ppo_update(
    policy: TorchPPOPolicy,
    trajectories: list[PPOTrajectory],
    *,
    learning_rate: float,
    clip_ratio: float,
    value_coef: float,
    entropy_coef: float,
    epochs: int,
    minibatch_size: int,
    optimizer: Any | None = None,
    normalize_entropy: bool = True,
    value_clip_range: float = 0.0,
    q_value_coef: float = 0.0,
    q_advantage_mix: float = 0.0,
    q_expected_sarsa_mix: float = 0.0,
    q_expected_sarsa_gamma: float = 1.0,
    kl_coef: float = 0.0,
    ema_policy: TorchPPOPolicy | None = None,
    ema_policy_kl_coef: float = 0.0,
    target_kl: float = 0.0,
    top_advantage_fraction: float = 1.0,
    min_advantage_samples: int = 1,
    behavior_temperature: float = 1.0,
    advantage_normalization: str = "global",
    advantage_group_weights: Any | None = None,
) -> dict[str, float]:
    import torch
    from torch import nn

    _validate_ppo_trajectory_alignment(trajectories)
    if _is_entity_graph_policy(policy):
        return _ppo_update_entity_graph(
            policy,
            trajectories,
            learning_rate=learning_rate,
            clip_ratio=clip_ratio,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            epochs=epochs,
            minibatch_size=minibatch_size,
            optimizer=optimizer,
            value_clip_range=value_clip_range,
            q_value_coef=q_value_coef,
            ema_policy=ema_policy,
            ema_policy_kl_coef=ema_policy_kl_coef,
            target_kl=target_kl,
            top_advantage_fraction=top_advantage_fraction,
            min_advantage_samples=min_advantage_samples,
            behavior_temperature=behavior_temperature,
            advantage_normalization=advantage_normalization,
            advantage_group_weights=advantage_group_weights,
        )

    samples = [sample for trajectory in trajectories for sample in trajectory.samples]
    if not samples:
        return {
            "samples": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "q_value_loss": 0.0,
            "q_chosen_return_corr": 0.0,
            "q_legal_std": 0.0,
            "q_legal_spread_entropy": 0.0,
            "q_advantage_sign_agreement": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "old_policy_kl": 0.0,
            "ema_policy_kl": 0.0,
            "clip_fraction": 0.0,
            "mean_shaped_reward": 0.0,
            "minibatches": 0.0,
            "early_stop": 0.0,
            "samples_before_filter": 0.0,
            "advantage_filter_kept_fraction": 1.0,
            "advantage_filter_threshold": 0.0,
        }
    returns = np.asarray(
        [ret for trajectory in trajectories for ret in trajectory.returns],
        dtype=np.float32,
    )
    raw_advantages = np.asarray(
        [adv for trajectory in trajectories for adv in trajectory.advantages],
        dtype=np.float32,
    )
    gae_advantages = raw_advantages.copy()
    old_log_probs = _ppo_reference_array(
        trajectories,
        reference_attr="ppo_reference_log_probs",
        fallback_attr="old_log_probs",
    )
    old_values = _ppo_reference_array(
        trajectories,
        reference_attr="ppo_reference_values",
        fallback_attr="old_values",
    )
    old_q_values_list = [
        q_value
        for trajectory in trajectories
        for q_value in (trajectory.old_q_values or trajectory.old_values)
    ]
    old_q_values = np.asarray(old_q_values_list, dtype=np.float32)
    if len(old_q_values) != len(old_values):
        old_q_values = old_values.copy()
    q_diagnostics = _ppo_q_diagnostics(
        trajectories,
        returns=returns,
        gae_advantages=gae_advantages,
    )
    q_targets = returns.copy()
    if q_expected_sarsa_mix > 0.0:
        q_expected_sarsa_mix = min(max(float(q_expected_sarsa_mix), 0.0), 1.0)
        sarsa_targets = _expected_sarsa_q_targets(
            trajectories,
            returns=returns,
            gamma=q_expected_sarsa_gamma,
        )
        q_targets = (
            (1.0 - q_expected_sarsa_mix) * q_targets
            + q_expected_sarsa_mix * sarsa_targets
        )
    if q_advantage_mix > 0.0:
        q_advantage_mix = min(max(float(q_advantage_mix), 0.0), 1.0)
        q_baselines = _old_q_policy_baselines(
            trajectories,
            fallback_values=old_values,
        )
        q_advantages = old_q_values - q_baselines
        raw_advantages = (
            (1.0 - q_advantage_mix) * raw_advantages
            + q_advantage_mix * q_advantages
        )
    # A sole legal action has probability one under every masked policy. Such a
    # row is useful critic evidence, but its PPO ratio and policy gradient are
    # identically fixed, so it must not rank or normalize policy advantages.
    policy_active_np = np.asarray(
        [len(sample.valid_actions) > 1 for sample in samples],
        dtype=np.bool_,
    )
    advantage_normalization_mode = str(advantage_normalization or "global").strip().lower()
    advantage_group_count = 0
    advantage_group_labels = _advantage_group_labels(trajectories)
    if advantage_normalization_mode not in {"global", "standard", "default"}:
        raw_advantages, advantage_group_count = _normalize_advantages_by_group(
            raw_advantages,
            advantage_group_labels,
            mode=advantage_normalization_mode,
            eligible_mask=policy_active_np,
        )
    raw_advantages, advantage_group_weight_count, advantage_group_weight_mean = (
        _apply_advantage_group_weights(
            raw_advantages,
            advantage_group_labels,
            advantage_group_weights,
        )
    )
    advantages = torch.as_tensor(
        raw_advantages,
        dtype=torch.float32,
        device=policy.device,
    )
    full_policy_active = torch.as_tensor(policy_active_np, device=policy.device)
    if advantage_normalization_mode in {"global", "standard", "default"}:
        advantages = _standardize_advantages_excluding_forced(
            advantages,
            full_policy_active,
        )
    samples_before_filter = len(samples)
    keep_indices, advantage_filter_threshold = _top_advantage_keep_indices(
        raw_advantages,
        top_fraction=top_advantage_fraction,
        min_samples=min_advantage_samples,
        eligible_mask=policy_active_np,
        retain_ineligible=True,
    )
    if len(keep_indices) != len(samples):
        samples = [samples[int(i)] for i in keep_indices]
        returns = returns[keep_indices]
        raw_advantages = raw_advantages[keep_indices]
        old_log_probs = old_log_probs[keep_indices]
        old_values = old_values[keep_indices]
        old_q_values = old_q_values[keep_indices]
        q_targets = q_targets[keep_indices]
        policy_active_np = policy_active_np[keep_indices]
        advantages = advantages[
            torch.as_tensor(keep_indices, dtype=torch.long, device=policy.device)
        ]
    old_action_probs = [
        probs for trajectory in trajectories for probs in trajectory.old_action_probs
    ]
    shaped_rewards = np.asarray(
        [reward for trajectory in trajectories for reward in trajectory.shaped_rewards],
        dtype=np.float32,
    )
    if len(keep_indices) != samples_before_filter:
        old_action_probs = [old_action_probs[int(i)] for i in keep_indices]
        shaped_rewards = shaped_rewards[keep_indices]
    observations = _policy_observation_array(policy, samples)
    actions = np.asarray([sample.action for sample in samples], dtype=np.int64)
    valid_actions = [sample.valid_actions for sample in samples]

    optimizer = optimizer or make_ppo_optimizer(policy, learning_rate=learning_rate)
    obs_t = torch.as_tensor(observations, dtype=torch.float32, device=policy.device)
    context_t = _action_context_features_tensor(samples, policy)
    actions_t = torch.as_tensor(actions, dtype=torch.long, device=policy.device)
    returns_t = torch.as_tensor(returns, dtype=torch.float32, device=policy.device)
    q_targets_t = torch.as_tensor(q_targets, dtype=torch.float32, device=policy.device)
    old_log_probs_t = torch.as_tensor(
        old_log_probs,
        dtype=torch.float32,
        device=policy.device,
    )
    old_values_t = torch.as_tensor(
        old_values,
        dtype=torch.float32,
        device=policy.device,
    )
    old_q_values_t = torch.as_tensor(
        old_q_values,
        dtype=torch.float32,
        device=policy.device,
    )
    old_policy_t = _dense_old_policy_tensor(
        valid_actions,
        old_action_probs,
        policy.action_size,
        policy.device,
    )
    ema_policy_t = None
    if ema_policy is not None and ema_policy_kl_coef > 0.0:
        ema_policy_t = _dense_policy_tensor_from_policy(
            ema_policy,
            obs_t,
            context_t,
            valid_actions,
            policy.action_size,
            policy.device,
        )
    policy_active = torch.as_tensor(policy_active_np, device=policy.device)

    behavior_temperature = max(float(behavior_temperature), 1.0e-6)
    n = len(samples)
    indices = np.arange(n)
    last_policy_loss = 0.0
    last_value_loss = 0.0
    last_q_value_loss = 0.0
    last_entropy = 0.0
    last_approx_kl = 0.0
    last_old_policy_kl = 0.0
    last_ema_policy_kl = 0.0
    last_clip_fraction = 0.0
    minibatches = 0
    early_stop = False
    for _ in range(epochs):
        np.random.shuffle(indices)
        for start in range(0, n, minibatch_size):
            batch_idx = indices[start : start + minibatch_size]
            batch_obs = obs_t[batch_idx]
            batch_context = context_t[batch_idx] if context_t is not None else None
            batch_actions = actions_t[batch_idx]
            batch_returns = returns_t[batch_idx]
            batch_q_targets = q_targets_t[batch_idx]
            batch_old_log_probs = old_log_probs_t[batch_idx]
            batch_old_values = old_values_t[batch_idx]
            batch_old_q_values = old_q_values_t[batch_idx]
            batch_old_policy = old_policy_t[batch_idx]
            batch_ema_policy = ema_policy_t[batch_idx] if ema_policy_t is not None else None
            batch_advantages = advantages[batch_idx]
            batch_policy_active = policy_active[batch_idx]
            batch_valid = [valid_actions[int(i)] for i in batch_idx]

            logits, values = policy.forward(batch_obs, batch_context)
            masked = _masked_logits(logits, batch_valid, policy.action_size)
            behavior_logits = masked
            if behavior_temperature != 1.0:
                behavior_logits = torch.clamp(
                    masked / behavior_temperature,
                    min=-50.0,
                    max=50.0,
                )
            dist = torch.distributions.Categorical(logits=behavior_logits)
            log_probs = dist.log_prob(batch_actions)
            ratio = torch.exp(log_probs - batch_old_log_probs)
            unclipped = ratio * batch_advantages
            clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * batch_advantages
            per_sample_policy_loss = -torch.min(unclipped, clipped)
            if bool(batch_policy_active.any()):
                policy_loss = per_sample_policy_loss[batch_policy_active].mean()
            else:
                policy_loss = per_sample_policy_loss.new_tensor(0.0)
            value_loss = _ppo_value_loss(
                values,
                batch_returns,
                batch_old_values,
                clip_range=value_clip_range,
            )
            if q_value_coef > 0.0:
                q_values = policy.q_values(batch_obs, batch_context)
                action_q_values = q_values.gather(1, batch_actions.unsqueeze(1)).squeeze(1)
                q_value_loss = _ppo_value_loss(
                    action_q_values,
                    batch_q_targets,
                    batch_old_q_values,
                    clip_range=value_clip_range,
                )
            else:
                q_value_loss = values.new_tensor(0.0)
            entropy_values = dist.entropy()
            if normalize_entropy:
                valid_counts = torch.as_tensor(
                    [max(len(actions), 2) for actions in batch_valid],
                    dtype=torch.float32,
                    device=policy.device,
                )
                entropy_values = entropy_values / torch.log(valid_counts)
            if bool(batch_policy_active.any()):
                entropy = entropy_values[batch_policy_active].mean()
            else:
                entropy = entropy_values.new_tensor(0.0)
            with torch.no_grad():
                per_sample_kl = batch_old_log_probs - log_probs
                per_sample_clipped = (torch.abs(ratio - 1.0) > clip_ratio).float()
                if bool(batch_policy_active.any()):
                    approx_kl = per_sample_kl[batch_policy_active].mean()
                    clip_fraction = per_sample_clipped[batch_policy_active].mean()
                else:
                    approx_kl = per_sample_kl.new_tensor(0.0)
                    clip_fraction = per_sample_clipped.new_tensor(0.0)
            if kl_coef > 0.0:
                log_policy = nn.functional.log_softmax(masked, dim=-1)
                per_sample_old_policy_kl = (
                    batch_old_policy
                    * (
                        torch.log(torch.clamp(batch_old_policy, min=1e-8))
                        - log_policy
                    )
                ).sum(dim=-1)
                if bool(batch_policy_active.any()):
                    old_policy_kl = per_sample_old_policy_kl[
                        batch_policy_active
                    ].mean()
                else:
                    old_policy_kl = per_sample_old_policy_kl.new_tensor(0.0)
            else:
                log_policy = None
                old_policy_kl = values.new_tensor(0.0)
            if batch_ema_policy is not None and ema_policy_kl_coef > 0.0:
                if log_policy is None:
                    log_policy = nn.functional.log_softmax(masked, dim=-1)
                per_sample_ema_policy_kl = (
                    batch_ema_policy
                    * (
                        torch.log(torch.clamp(batch_ema_policy, min=1e-8))
                        - log_policy
                    )
                ).sum(dim=-1)
                if bool(batch_policy_active.any()):
                    ema_policy_kl = per_sample_ema_policy_kl[
                        batch_policy_active
                    ].mean()
                else:
                    ema_policy_kl = per_sample_ema_policy_kl.new_tensor(0.0)
            else:
                ema_policy_kl = values.new_tensor(0.0)
            loss = (
                policy_loss
                + value_coef * value_loss
                + q_value_coef * q_value_loss
                + kl_coef * old_policy_kl
                + ema_policy_kl_coef * ema_policy_kl
                - entropy_coef * entropy
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(_policy_parameters(policy), 1.0)
            optimizer.step()
            last_policy_loss = float(policy_loss.item())
            last_value_loss = float(value_loss.item())
            last_q_value_loss = float(q_value_loss.item())
            last_entropy = float(entropy.item())
            last_approx_kl = float(approx_kl.item())
            last_old_policy_kl = float(old_policy_kl.item())
            last_ema_policy_kl = float(ema_policy_kl.item())
            last_clip_fraction = float(clip_fraction.item())
            minibatches += 1
            if target_kl > 0.0 and last_approx_kl > target_kl:
                early_stop = True
                break
        if early_stop:
            break

    return {
        "samples": float(n),
        "policy_loss": last_policy_loss,
        "value_loss": last_value_loss,
        "q_value_loss": last_q_value_loss,
        "q_chosen_return_corr": q_diagnostics["q_chosen_return_corr"],
        "q_legal_std": q_diagnostics["q_legal_std"],
        "q_legal_spread_entropy": q_diagnostics["q_legal_spread_entropy"],
        "q_advantage_sign_agreement": q_diagnostics[
            "q_advantage_sign_agreement"
        ],
        "entropy": last_entropy,
        "approx_kl": last_approx_kl,
        "old_policy_kl": last_old_policy_kl,
        "ema_policy_kl": last_ema_policy_kl,
        "clip_fraction": last_clip_fraction,
        "mean_shaped_reward": float(shaped_rewards.mean()) if len(shaped_rewards) else 0.0,
        "minibatches": float(minibatches),
        "early_stop": 1.0 if early_stop else 0.0,
        "samples_before_filter": float(samples_before_filter),
        "advantage_filter_kept_fraction": (
            float(n / samples_before_filter) if samples_before_filter else 1.0
        ),
        "advantage_filter_threshold": float(advantage_filter_threshold),
        "behavior_temperature": float(behavior_temperature),
        "advantage_normalization": advantage_normalization_mode,
        "advantage_groups": float(advantage_group_count),
        "advantage_group_weight_count": float(advantage_group_weight_count),
        "advantage_group_weight_mean": float(advantage_group_weight_mean),
        "policy_active_fraction": float(policy_active_np.mean()) if len(policy_active_np) else 1.0,
    }


def _make_catanatron_value_fn():
    from catanatron.players.value import get_value_fn

    return get_value_fn("contender_fn", None)


def _finite_catanatron_score(
    env: ColonistMultiAgentEnv,
    color: Any | None,
    *,
    value_fn: Any | None,
    opponent_penalty: float,
) -> float | None:
    if color is None or value_fn is None:
        return None
    if env.game.winning_color() is not None:
        return None
    score = _catanatron_value_score(
        env.game,
        color,
        opponent_penalty=opponent_penalty,
        value_fn=value_fn,
    )
    if not np.isfinite(score) or abs(float(score)) >= 1e17:
        return None
    return float(score)


def _clipped_value_delta_reward(
    before: float | None,
    after: float | None,
    *,
    coef: float,
    scale: float,
) -> float:
    if coef <= 0.0 or before is None or after is None:
        return 0.0
    scale = max(float(scale), 1e-6)
    normalized = max(min((float(after) - float(before)) / scale, 1.0), -1.0)
    return float(coef) * normalized


def _ppo_value_loss(values, returns, old_values, *, clip_range: float):
    import torch
    from torch import nn

    unclipped = nn.functional.mse_loss(values, returns, reduction="none")
    if clip_range <= 0.0:
        return unclipped.mean()
    clipped_values = old_values + torch.clamp(
        values - old_values,
        -clip_range,
        clip_range,
    )
    clipped = nn.functional.mse_loss(clipped_values, returns, reduction="none")
    return torch.maximum(unclipped, clipped).mean()


def _is_entity_graph_policy(policy: Any) -> bool:
    return str(getattr(policy, "architecture", "") or "") == "entity_graph"


def _standardize_advantages_excluding_forced(advantages, policy_active):
    """Standardize ``advantages`` to mean 0 / std 1, computing the mean/std from ONLY the
    ``policy_active`` (legal_count > 1) rows, then applying that affine transform to every row.

    FIX A9: forced (legal_count == 1) rows have zero policy gradient by construction (ratio is
    always exactly 1), so letting their advantage values pollute the global standardization
    statistics can badly distort the normalized advantages seen by the genuinely-active rows
    (e.g. one outlier forced-row advantage inflating std and shrinking every real signal toward
    zero). Falls back to using every row when none are policy-active.
    """
    import torch

    stats_advantages = advantages[policy_active] if bool(policy_active.any()) else advantages
    advantage_std = stats_advantages.std(unbiased=False)
    if bool(torch.isfinite(advantage_std).item()) and float(advantage_std.item()) > 1e-8:
        return (advantages - stats_advantages.mean()) / advantage_std
    return advantages - stats_advantages.mean()


def _ppo_update_entity_graph(
    policy: Any,
    trajectories: list[PPOTrajectory],
    *,
    learning_rate: float,
    clip_ratio: float,
    value_coef: float,
    entropy_coef: float,
    epochs: int,
    minibatch_size: int,
    optimizer: Any | None = None,
    value_clip_range: float = 0.0,
    q_value_coef: float = 0.0,
    ema_policy: Any | None = None,
    ema_policy_kl_coef: float = 0.0,
    target_kl: float = 0.0,
    top_advantage_fraction: float = 1.0,
    min_advantage_samples: int = 1,
    behavior_temperature: float = 1.0,
    advantage_normalization: str = "global",
    advantage_group_weights: Any | None = None,
) -> dict[str, float]:
    # PPO ratios are only meaningful when the learner recomputes the exact
    # behavior distribution recorded by actors. Actors serve EntityGraphPolicy
    # in eval mode, so enabling dropout here creates false ratios/clipping even
    # before an optimizer step. eval() does not disable autograd: gradients and
    # optimizer updates remain active while stochastic/stateful inference layers
    # follow the actor contract. The finally block preserves that postcondition
    # on empty/error exits too.
    policy.model.eval()
    try:
        return _ppo_update_entity_graph_body(
            policy,
            trajectories,
            learning_rate=learning_rate,
            clip_ratio=clip_ratio,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            epochs=epochs,
            minibatch_size=minibatch_size,
            optimizer=optimizer,
            value_clip_range=value_clip_range,
            q_value_coef=q_value_coef,
            ema_policy=ema_policy,
            ema_policy_kl_coef=ema_policy_kl_coef,
            target_kl=target_kl,
            top_advantage_fraction=top_advantage_fraction,
            min_advantage_samples=min_advantage_samples,
            behavior_temperature=behavior_temperature,
            advantage_normalization=advantage_normalization,
            advantage_group_weights=advantage_group_weights,
        )
    finally:
        policy.model.eval()


def _ppo_update_entity_graph_body(
    policy: Any,
    trajectories: list[PPOTrajectory],
    *,
    learning_rate: float,
    clip_ratio: float,
    value_coef: float,
    entropy_coef: float,
    epochs: int,
    minibatch_size: int,
    optimizer: Any | None = None,
    value_clip_range: float = 0.0,
    q_value_coef: float = 0.0,
    ema_policy: Any | None = None,
    ema_policy_kl_coef: float = 0.0,
    target_kl: float = 0.0,
    top_advantage_fraction: float = 1.0,
    min_advantage_samples: int = 1,
    behavior_temperature: float = 1.0,
    advantage_normalization: str = "global",
    advantage_group_weights: Any | None = None,
) -> dict[str, float]:
    import torch
    from torch import nn

    samples = [sample for trajectory in trajectories for sample in trajectory.samples]
    if not samples:
        return {
            "samples": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "q_value_loss": 0.0,
            "q_chosen_return_corr": 0.0,
            "q_legal_std": 0.0,
            "q_legal_spread_entropy": 0.0,
            "q_advantage_sign_agreement": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "old_policy_kl": 0.0,
            "ema_policy_kl": 0.0,
            "clip_fraction": 0.0,
            "mean_shaped_reward": 0.0,
            "minibatches": 0.0,
            "early_stop": 0.0,
            "samples_before_filter": 0.0,
            "advantage_filter_kept_fraction": 1.0,
            "advantage_filter_threshold": 0.0,
        }
    missing_entity = sum(1 for sample in samples if sample.entity_features is None)
    if missing_entity:
        raise ValueError(
            "entity_graph PPO requires StepSample.entity_features; "
            f"{missing_entity}/{len(samples)} samples are missing them"
        )

    returns = np.asarray(
        [ret for trajectory in trajectories for ret in trajectory.returns],
        dtype=np.float32,
    )
    raw_advantages = np.asarray(
        [adv for trajectory in trajectories for adv in trajectory.advantages],
        dtype=np.float32,
    )
    old_log_probs = _ppo_reference_array(
        trajectories,
        reference_attr="ppo_reference_log_probs",
        fallback_attr="old_log_probs",
    )
    old_values = _ppo_reference_array(
        trajectories,
        reference_attr="ppo_reference_values",
        fallback_attr="old_values",
    )
    old_q_values_list = [
        q_value
        for trajectory in trajectories
        for q_value in (trajectory.old_q_values or trajectory.old_values)
    ]
    old_q_values = np.asarray(old_q_values_list, dtype=np.float32)
    if len(old_q_values) != len(old_values):
        old_q_values = old_values.copy()
    q_targets = returns.copy()

    policy_active_np = np.asarray(
        [len(sample.valid_actions) > 1 for sample in samples],
        dtype=np.bool_,
    )
    advantage_normalization_mode = str(advantage_normalization or "global").strip().lower()
    advantage_group_count = 0
    advantage_group_labels = _advantage_group_labels(trajectories)
    if advantage_normalization_mode not in {"global", "standard", "default"}:
        raw_advantages, advantage_group_count = _normalize_advantages_by_group(
            raw_advantages,
            advantage_group_labels,
            mode=advantage_normalization_mode,
            eligible_mask=policy_active_np,
        )
    raw_advantages, advantage_group_weight_count, advantage_group_weight_mean = (
        _apply_advantage_group_weights(
            raw_advantages,
            advantage_group_labels,
            advantage_group_weights,
        )
    )
    advantages = torch.as_tensor(
        raw_advantages,
        dtype=torch.float32,
        device=policy.device,
    )
    full_policy_active = torch.as_tensor(policy_active_np, device=policy.device)
    if advantage_normalization_mode in {"global", "standard", "default"}:
        advantages = _standardize_advantages_excluding_forced(
            advantages,
            full_policy_active,
        )

    samples_before_filter = len(samples)
    keep_indices, advantage_filter_threshold = _top_advantage_keep_indices(
        raw_advantages,
        top_fraction=top_advantage_fraction,
        min_samples=min_advantage_samples,
        eligible_mask=policy_active_np,
        retain_ineligible=True,
    )
    shaped_rewards = np.asarray(
        [reward for trajectory in trajectories for reward in trajectory.shaped_rewards],
        dtype=np.float32,
    )
    if len(keep_indices) != len(samples):
        samples = [samples[int(i)] for i in keep_indices]
        returns = returns[keep_indices]
        raw_advantages = raw_advantages[keep_indices]
        old_log_probs = old_log_probs[keep_indices]
        old_values = old_values[keep_indices]
        old_q_values = old_q_values[keep_indices]
        q_targets = q_targets[keep_indices]
        shaped_rewards = shaped_rewards[keep_indices]
        policy_active_np = policy_active_np[keep_indices]
        advantages = advantages[
            torch.as_tensor(keep_indices, dtype=torch.long, device=policy.device)
        ]

    action_columns = np.asarray(
        [_entity_action_column(sample) for sample in samples],
        dtype=np.int64,
    )
    # FIX A9: a forced action (legal_count == 1) has log p == 0 under the mask for both the
    # behavior and current policy, so ratio == 1 identically -- zero policy gradient by
    # construction. Left in, these rows still consume minibatch slots AND pollute the global
    # advantage-normalization mean/std with values that can never move the policy. Track which
    # samples are "policy-active" (legal_count > 1) so both can be excluded; they still
    # participate fully in the value loss.
    optimizer = optimizer or make_ppo_optimizer(policy, learning_rate=learning_rate)
    returns_t = torch.as_tensor(returns, dtype=torch.float32, device=policy.device)
    q_targets_t = torch.as_tensor(q_targets, dtype=torch.float32, device=policy.device)
    old_log_probs_t = torch.as_tensor(old_log_probs, dtype=torch.float32, device=policy.device)
    old_values_t = torch.as_tensor(old_values, dtype=torch.float32, device=policy.device)
    old_q_values_t = torch.as_tensor(old_q_values, dtype=torch.float32, device=policy.device)
    policy_active = torch.as_tensor(policy_active_np, device=policy.device)

    behavior_temperature = max(float(behavior_temperature), 1.0e-6)
    n = len(samples)
    indices = np.arange(n)
    last_policy_loss = 0.0
    last_value_loss = 0.0
    last_q_value_loss = 0.0
    last_entropy = 0.0
    last_approx_kl = 0.0
    last_old_policy_kl = 0.0
    last_ema_policy_kl = 0.0
    last_clip_fraction = 0.0
    minibatches = 0
    early_stop = False
    for _ in range(epochs):
        np.random.shuffle(indices)
        for start in range(0, n, minibatch_size):
            batch_idx = indices[start : start + minibatch_size]
            batch_samples = [samples[int(i)] for i in batch_idx]
            batch_actions = torch.as_tensor(
                action_columns[batch_idx],
                dtype=torch.long,
                device=policy.device,
            )
            batch_returns = returns_t[batch_idx]
            batch_q_targets = q_targets_t[batch_idx]
            batch_old_log_probs = old_log_probs_t[batch_idx]
            batch_old_values = old_values_t[batch_idx]
            batch_old_q_values = old_q_values_t[batch_idx]
            batch_advantages = advantages[batch_idx]
            batch_policy_active = policy_active[batch_idx]

            outputs = _entity_graph_outputs(
                policy,
                batch_samples,
                return_q=q_value_coef > 0.0,
            )
            logits = outputs["logits"]
            values = outputs["value"]
            valid_mask = _entity_behavior_valid_mask(batch_samples, logits)
            behavior_logits = _behavior_policy_logits(
                logits,
                behavior_temperature,
                valid_mask=valid_mask,
            )
            dist = torch.distributions.Categorical(logits=behavior_logits)
            log_probs = dist.log_prob(batch_actions)
            ratio = torch.exp(log_probs - batch_old_log_probs)
            unclipped = ratio * batch_advantages
            clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * batch_advantages
            per_sample_policy_loss = -torch.min(unclipped, clipped)
            # FIX A9: forced (legal_count == 1) rows have ratio == 1 identically -- zero policy
            # gradient by construction -- so exclude them from the policy-loss MEAN instead of
            # letting them dilute it with an always-zero term. They still flow through the
            # value loss below unchanged.
            if bool(batch_policy_active.any()):
                policy_loss = per_sample_policy_loss[batch_policy_active].mean()
            else:
                policy_loss = per_sample_policy_loss.new_tensor(0.0)
            value_loss = _ppo_value_loss(
                values,
                batch_returns,
                batch_old_values,
                clip_range=value_clip_range,
            )
            if q_value_coef > 0.0:
                q_values = outputs["q_values"]
                action_q_values = q_values.gather(1, batch_actions.unsqueeze(1)).squeeze(1)
                q_value_loss = _ppo_value_loss(
                    action_q_values,
                    batch_q_targets,
                    batch_old_q_values,
                    clip_range=value_clip_range,
                )
            else:
                q_value_loss = values.new_tensor(0.0)
            entropy_values = dist.entropy()
            legal_counts = torch.as_tensor(
                [max(len(sample.valid_actions), 2) for sample in batch_samples],
                dtype=torch.float32,
                device=policy.device,
            )
            normalized_entropy = entropy_values / torch.log(legal_counts)
            if bool(batch_policy_active.any()):
                entropy = normalized_entropy[batch_policy_active].mean()
            else:
                entropy = normalized_entropy.new_tensor(0.0)
            with torch.no_grad():
                per_sample_kl = batch_old_log_probs - log_probs
                per_sample_clipped = (torch.abs(ratio - 1.0) > clip_ratio).float()
                if bool(batch_policy_active.any()):
                    approx_kl = per_sample_kl[batch_policy_active].mean()
                    clip_fraction = per_sample_clipped[batch_policy_active].mean()
                else:
                    approx_kl = per_sample_kl.new_tensor(0.0)
                    clip_fraction = per_sample_clipped.new_tensor(0.0)
            if ema_policy is not None and ema_policy_kl_coef > 0.0:
                with torch.no_grad():
                    ema_outputs = _entity_graph_outputs(ema_policy, batch_samples, return_q=False)
                    ema_logits = _behavior_policy_logits(
                        ema_outputs["logits"],
                        behavior_temperature,
                        valid_mask=valid_mask,
                    )
                    ema_policy_t = torch.softmax(ema_logits, dim=-1)
                # Match the KL anchor to the same temperature-scaled distribution used by
                # PPO ratios. Otherwise temp-controlled actors can drift in behavior space
                # while reporting a tiny raw-logit KL.
                log_policy = nn.functional.log_softmax(behavior_logits, dim=-1)
                per_sample_ema_policy_kl = (
                    ema_policy_t
                    * (torch.log(torch.clamp(ema_policy_t, min=1e-8)) - log_policy)
                ).sum(dim=-1)
                if bool(batch_policy_active.any()):
                    ema_policy_kl = per_sample_ema_policy_kl[
                        batch_policy_active
                    ].mean()
                else:
                    ema_policy_kl = per_sample_ema_policy_kl.new_tensor(0.0)
            else:
                ema_policy_kl = values.new_tensor(0.0)
            loss = (
                policy_loss
                + value_coef * value_loss
                + q_value_coef * q_value_loss
                + ema_policy_kl_coef * ema_policy_kl
                - entropy_coef * entropy
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(_policy_parameters(policy), 1.0)
            optimizer.step()
            last_policy_loss = float(policy_loss.item())
            last_value_loss = float(value_loss.item())
            last_q_value_loss = float(q_value_loss.item())
            last_entropy = float(entropy.item())
            last_approx_kl = float(approx_kl.item())
            last_ema_policy_kl = float(ema_policy_kl.item())
            last_clip_fraction = float(clip_fraction.item())
            minibatches += 1
            if target_kl > 0.0 and last_approx_kl > target_kl:
                early_stop = True
                break
        if early_stop:
            break

    return {
        "samples": float(n),
        "policy_loss": last_policy_loss,
        "value_loss": last_value_loss,
        "q_value_loss": last_q_value_loss,
        "q_chosen_return_corr": 0.0,
        "q_legal_std": 0.0,
        "q_legal_spread_entropy": 0.0,
        "q_advantage_sign_agreement": 0.0,
        "entropy": last_entropy,
        "approx_kl": last_approx_kl,
        "old_policy_kl": last_old_policy_kl,
        "ema_policy_kl": last_ema_policy_kl,
        "clip_fraction": last_clip_fraction,
        "mean_shaped_reward": float(shaped_rewards.mean()) if len(shaped_rewards) else 0.0,
        "minibatches": float(minibatches),
        "early_stop": 1.0 if early_stop else 0.0,
        "samples_before_filter": float(samples_before_filter),
        "advantage_filter_kept_fraction": float(n / max(1, samples_before_filter)),
        "advantage_filter_threshold": float(advantage_filter_threshold),
        "behavior_temperature": float(behavior_temperature),
        "advantage_normalization": advantage_normalization_mode,
        "advantage_groups": float(advantage_group_count),
        "advantage_group_weight_count": float(advantage_group_weight_count),
        "advantage_group_weight_mean": float(advantage_group_weight_mean),
        "policy_active_fraction": float(policy_active_np.mean()) if len(policy_active_np) else 1.0,
    }


def _entity_action_column(sample: StepSample) -> int:
    try:
        return tuple(int(action) for action in sample.valid_actions).index(int(sample.action))
    except ValueError as error:
        raise ValueError(
            f"sample action {sample.action} is not in valid_actions={sample.valid_actions}"
        ) from error


def _entity_behavior_valid_mask(samples: list[StepSample], logits):
    import torch

    width = int(logits.shape[-1])
    counts = torch.as_tensor(
        [len(sample.valid_actions) for sample in samples],
        dtype=torch.long,
        device=logits.device,
    )
    return torch.arange(width, device=logits.device).unsqueeze(0) < counts.unsqueeze(1)


def _entity_graph_outputs(policy: Any, samples: list[StepSample], *, return_q: bool = False):
    batch, legal_action_ids, legal_action_context = _entity_graph_batch(samples, policy)
    return policy.forward_legal_np(
        batch,
        legal_action_ids,
        legal_action_context,
        return_q=return_q,
    )


def _entity_graph_batch(
    samples: list[StepSample],
    policy: Any,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    if not samples:
        raise ValueError("cannot build entity_graph batch from zero samples")
    max_legal = max(len(sample.valid_actions) for sample in samples)
    max_events = max(
        int(np.asarray(sample.entity_features["event_tokens"]).shape[0])
        for sample in samples
        if sample.entity_features is not None
    )
    context_size = int(getattr(policy, "context_action_feature_size", CONTEXT_ACTION_FEATURE_SIZE))
    legal_feature_size = int(getattr(policy.config, "legal_action_feature_size"))
    legal_action_ids = np.full((len(samples), max_legal), -1, dtype=np.int64)
    legal_action_context = np.zeros((len(samples), max_legal, context_size), dtype=np.float32)
    batch_lists: dict[str, list[np.ndarray]] = {}
    for row, sample in enumerate(samples):
        entity = sample.entity_features
        if entity is None:
            raise ValueError("entity_graph sample is missing entity_features")
        valid_actions = tuple(int(action) for action in sample.valid_actions)
        legal_count = len(valid_actions)
        legal_action_ids[row, :legal_count] = np.asarray(valid_actions, dtype=np.int64)

        context = sample.action_context_features
        if context is None:
            selected_context = np.zeros((legal_count, context_size), dtype=np.float32)
        else:
            context_array = np.asarray(context, dtype=np.float32)
            selected_context = context_array[list(valid_actions), :]
            if selected_context.shape[1] != context_size:
                selected_context = _resize_context_array(selected_context, feature_size=context_size)
        legal_action_context[row, :legal_count, :] = selected_context

        for key, value in entity.items():
            arr = np.asarray(value)
            if key == "legal_action_tokens":
                padded = np.zeros((max_legal, legal_feature_size), dtype=np.float32)
                padded[:legal_count, :] = arr.astype(np.float32, copy=False)
                arr = padded
            elif key == "legal_action_target_ids":
                # -1 is the no-target sentinel in every target namespace. Zero
                # is a valid hex/vertex/edge/player id, so zero-padding makes a
                # padded action look targeted and target-aware policies reject
                # ordinary mixed-legal-width batches before the forward pass.
                padded_targets = np.full(
                    (max_legal, arr.shape[1]), -1, dtype=arr.dtype
                )
                padded_targets[:legal_count, :] = arr[:legal_count]
                arr = padded_targets
            elif key == "legal_action_mask":
                padded_mask = np.zeros((max_legal,), dtype=np.bool_)
                padded_mask[:legal_count] = np.asarray(arr, dtype=np.bool_)[:legal_count]
                arr = padded_mask
            elif key == "event_tokens":
                padded_events = np.zeros((max_events, arr.shape[1]), dtype=np.float32)
                event_count = min(max_events, int(arr.shape[0]))
                padded_events[:event_count, :] = arr[:event_count].astype(
                    np.float32,
                    copy=False,
                )
                arr = padded_events
            elif key == "event_mask":
                padded_event_mask = np.zeros((max_events,), dtype=np.bool_)
                event_count = min(max_events, int(arr.shape[0]))
                padded_event_mask[:event_count] = np.asarray(arr, dtype=np.bool_)[:event_count]
                arr = padded_event_mask
            batch_lists.setdefault(key, []).append(arr)
    batch = {key: np.stack(values, axis=0) for key, values in batch_lists.items()}
    return batch, legal_action_ids, legal_action_context


def _dense_old_policy_tensor(
    valid_actions: list[tuple[int, ...]],
    old_action_probs: list[np.ndarray],
    action_size: int,
    device,
):
    import torch

    dense = np.zeros((len(valid_actions), action_size), dtype=np.float32)
    for row, (actions, probs) in enumerate(zip(valid_actions, old_action_probs)):
        if len(actions) != len(probs):
            raise ValueError("old action probabilities must align with valid actions")
        dense[row, np.asarray(actions, dtype=np.int64)] = np.asarray(
            probs,
            dtype=np.float32,
    )
    return torch.as_tensor(dense, dtype=torch.float32, device=device)


def _dense_policy_tensor_from_policy(
    target_policy: TorchPPOPolicy,
    observations,
    action_context_features,
    valid_actions: list[tuple[int, ...]],
    action_size: int,
    device,
):
    import torch

    with torch.no_grad():
        logits, _ = target_policy.forward(observations, action_context_features)
        masked = _masked_logits(logits, valid_actions, action_size)
        probs = torch.softmax(masked, dim=-1)
    return probs.to(device=device, dtype=torch.float32)


def update_ema_policy(
    target: TorchPPOPolicy,
    source: TorchPPOPolicy,
    *,
    decay: float,
) -> None:
    import torch

    decay = min(max(float(decay), 0.0), 1.0)
    with torch.no_grad():
        target_state = target.model.state_dict()
        source_state = source.model.state_dict()
        for key, target_tensor in target_state.items():
            source_tensor = source_state[key].to(target_tensor.device)
            if torch.is_floating_point(target_tensor):
                target_tensor.mul_(decay).add_(source_tensor, alpha=1.0 - decay)
            else:
                target_tensor.copy_(source_tensor)
        for target_module_name in (
            "actor",
            "critic",
            "q_head",
            "q_state",
            "q_action_encoder",
            "q_action_bias",
            "action_encoder",
            "action_id_embedding",
            "action_bias",
        ):
            target_module = getattr(target, target_module_name, None)
            source_module = getattr(source, target_module_name, None)
            if target_module is None or source_module is None:
                continue
            target_state = target_module.state_dict()
            source_state = source_module.state_dict()
            for key, target_tensor in target_state.items():
                source_tensor = source_state[key].to(target_tensor.device)
                if torch.is_floating_point(target_tensor):
                    target_tensor.mul_(decay).add_(source_tensor, alpha=1.0 - decay)
                else:
                    target_tensor.copy_(source_tensor)


def _top_advantage_keep_indices(
    advantages: np.ndarray,
    *,
    top_fraction: float,
    min_samples: int,
    eligible_mask: np.ndarray | None = None,
    retain_ineligible: bool = False,
) -> tuple[np.ndarray, float]:
    """Select top positive eligible rows, optionally retaining value-only rows.

    ``min_samples`` and ``top_fraction`` apply to the eligible positive
    population. If that population has no positive advantage, filtering is a
    no-op, matching the historical safe fallback.
    """
    n = len(advantages)
    if n == 0:
        return np.asarray([], dtype=np.int64), 0.0
    if eligible_mask is None:
        eligible = np.ones(n, dtype=np.bool_)
    else:
        eligible = np.asarray(eligible_mask, dtype=np.bool_)
        if eligible.shape != advantages.shape:
            raise ValueError(
                "advantage eligibility mask does not match advantage shape: "
                f"{eligible.shape} vs {advantages.shape}"
            )
    fraction = float(top_fraction)
    if not math.isfinite(fraction) or fraction >= 1.0:
        return np.arange(n, dtype=np.int64), 0.0
    fraction = max(fraction, 0.0)
    positive_indices = np.flatnonzero(
        eligible & np.isfinite(advantages) & (advantages > 0.0)
    )
    if len(positive_indices) == 0:
        return np.arange(n, dtype=np.int64), 0.0
    keep_count = int(math.ceil(len(positive_indices) * fraction))
    keep_count = max(int(min_samples), keep_count, 1)
    keep_count = min(keep_count, len(positive_indices))
    positive_values = advantages[positive_indices]
    selected_offset = np.argpartition(positive_values, -keep_count)[-keep_count:]
    selected = np.sort(positive_indices[selected_offset]).astype(np.int64)
    threshold = float(np.min(advantages[selected])) if len(selected) else 0.0
    if retain_ineligible:
        selected = np.sort(
            np.concatenate((selected, np.flatnonzero(~eligible)))
        ).astype(np.int64)
    return selected, threshold


def _old_q_policy_baselines(
    trajectories: list[PPOTrajectory],
    *,
    fallback_values: np.ndarray,
) -> np.ndarray:
    baselines: list[float] = []
    fallback_index = 0
    for trajectory in trajectories:
        action_q_values = trajectory.old_action_q_values
        for row, probs in enumerate(trajectory.old_action_probs):
            fallback = float(fallback_values[fallback_index])
            baseline = fallback
            if action_q_values is not None and row < len(action_q_values):
                q_values = np.asarray(action_q_values[row], dtype=np.float32)
                probs_array = np.asarray(probs, dtype=np.float32)
                if (
                    q_values.shape == probs_array.shape
                    and q_values.ndim == 1
                    and np.isfinite(q_values).all()
                    and np.isfinite(probs_array).all()
                    and float(probs_array.sum()) > 0.0
                ):
                    normalized = probs_array / float(probs_array.sum())
                    baseline = float(np.dot(normalized, q_values))
            baselines.append(baseline)
            fallback_index += 1
    if len(baselines) != len(fallback_values):
        return np.asarray(fallback_values, dtype=np.float32)
    return np.asarray(baselines, dtype=np.float32)


def _expected_sarsa_q_targets(
    trajectories: list[PPOTrajectory],
    *,
    returns: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Build bootstrapped Q targets from old-policy legal-action expectations.

    Terminal or truncated rows fall back to Monte Carlo/GAE returns. Earlier rows
    bootstrap from the next sample for the same player using E_a[Q_old(s', a)].
    This is a conservative Expected-SARSA-style target for the Q head; it does
    not change policy advantages unless the caller also enables q_advantage_mix.
    """

    gamma = float(gamma)
    targets: list[float] = []
    flat_offset = 0
    for trajectory in trajectories:
        n = len(trajectory.samples)
        local_targets = [0.0] * n
        next_expected_q: dict[str, float] = {}
        action_q_values = trajectory.old_action_q_values
        for row in range(n - 1, -1, -1):
            flat_index = flat_offset + row
            fallback = float(returns[flat_index]) if flat_index < len(returns) else 0.0
            sample = trajectory.samples[row]
            player = sample.player
            if player in next_expected_q:
                reward = (
                    float(trajectory.shaped_rewards[row])
                    if row < len(trajectory.shaped_rewards)
                    else 0.0
                )
                target = reward + gamma * float(next_expected_q[player])
            else:
                target = fallback
            if not math.isfinite(target):
                target = fallback
            local_targets[row] = target
            expected_q = _row_expected_old_q(
                action_q_values,
                trajectory.old_action_probs,
                row,
                fallback=fallback,
            )
            next_expected_q[player] = expected_q
        targets.extend(local_targets)
        flat_offset += n
    if len(targets) != len(returns):
        return np.asarray(returns, dtype=np.float32)
    return np.asarray(targets, dtype=np.float32)


def _row_expected_old_q(
    action_q_values: list[np.ndarray] | None,
    old_action_probs: list[np.ndarray],
    row: int,
    *,
    fallback: float,
) -> float:
    if action_q_values is None or row >= len(action_q_values):
        return float(fallback)
    if row >= len(old_action_probs):
        return float(fallback)
    q_values = np.asarray(action_q_values[row], dtype=np.float32)
    probs = np.asarray(old_action_probs[row], dtype=np.float32)
    if (
        q_values.shape != probs.shape
        or q_values.ndim != 1
        or not np.isfinite(q_values).all()
        or not np.isfinite(probs).all()
        or float(probs.sum()) <= 0.0
    ):
        return float(fallback)
    normalized = probs / float(probs.sum())
    expected = float(np.dot(normalized, q_values))
    return expected if math.isfinite(expected) else float(fallback)


def _ppo_q_diagnostics(
    trajectories: list[PPOTrajectory],
    *,
    returns: np.ndarray,
    gae_advantages: np.ndarray,
) -> dict[str, float]:
    diagnostics = {
        "q_chosen_return_corr": 0.0,
        "q_legal_std": 0.0,
        "q_legal_spread_entropy": 0.0,
        "q_advantage_sign_agreement": 0.0,
    }
    chosen_q_values: list[float] = []
    chosen_returns: list[float] = []
    legal_stds: list[float] = []
    legal_entropies: list[float] = []
    sign_matches: list[float] = []
    flat_index = 0
    for trajectory in trajectories:
        trajectory_q_values = trajectory.old_q_values
        action_q_values = trajectory.old_action_q_values
        has_chosen_q_values = (
            trajectory_q_values is not None
            and len(trajectory_q_values) == len(trajectory.samples)
        )
        for row, sample in enumerate(trajectory.samples):
            q_value = None
            if has_chosen_q_values:
                candidate_q = float(trajectory_q_values[row])
                if math.isfinite(candidate_q):
                    q_value = candidate_q
                    if flat_index < len(returns) and math.isfinite(
                        float(returns[flat_index])
                    ):
                        chosen_q_values.append(candidate_q)
                        chosen_returns.append(float(returns[flat_index]))

            legal_q = None
            if action_q_values is not None and row < len(action_q_values):
                legal_q_array = np.asarray(action_q_values[row], dtype=np.float32)
                if (
                    legal_q_array.ndim == 1
                    and legal_q_array.size > 0
                    and np.isfinite(legal_q_array).all()
                ):
                    legal_q = legal_q_array
                    legal_stds.append(float(np.std(legal_q_array, dtype=np.float64)))
                    legal_entropies.append(_normalized_q_spread_entropy(legal_q_array))

            if (
                q_value is not None
                and legal_q is not None
                and row < len(trajectory.old_action_probs)
                and flat_index < len(gae_advantages)
            ):
                probs = np.asarray(trajectory.old_action_probs[row], dtype=np.float32)
                if (
                    probs.shape == legal_q.shape
                    and probs.ndim == 1
                    and np.isfinite(probs).all()
                    and float(probs.sum()) > 0.0
                ):
                    normalized = probs / float(probs.sum())
                    q_advantage = q_value - float(np.dot(normalized, legal_q))
                    gae_advantage = float(gae_advantages[flat_index])
                    if math.isfinite(q_advantage) and math.isfinite(gae_advantage):
                        q_sign = float(np.sign(q_advantage))
                        gae_sign = float(np.sign(gae_advantage))
                        if q_sign != 0.0 and gae_sign != 0.0:
                            sign_matches.append(1.0 if q_sign == gae_sign else 0.0)
            flat_index += 1

    diagnostics["q_chosen_return_corr"] = _finite_correlation(
        chosen_q_values,
        chosen_returns,
    )
    if legal_stds:
        diagnostics["q_legal_std"] = float(np.mean(legal_stds))
    if legal_entropies:
        diagnostics["q_legal_spread_entropy"] = float(np.mean(legal_entropies))
    if sign_matches:
        diagnostics["q_advantage_sign_agreement"] = float(np.mean(sign_matches))
    return diagnostics


def _finite_correlation(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(ys) < 2:
        return 0.0
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return 0.0
    x = x[mask] - float(x[mask].mean())
    y = y[mask] - float(y[mask].mean())
    denominator = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    if not math.isfinite(denominator) or denominator <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(x, y) / denominator, -1.0, 1.0))


def _normalized_q_spread_entropy(q_values: np.ndarray) -> float:
    if q_values.size <= 1:
        return 0.0
    centered = q_values.astype(np.float64) - float(np.max(q_values))
    weights = np.exp(np.clip(centered, -60.0, 0.0))
    total = float(weights.sum())
    if not math.isfinite(total) or total <= 0.0:
        return 0.0
    probabilities = weights / total
    entropy = -float(
        np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)))
    )
    return float(entropy / math.log(float(q_values.size)))


def _action_context_features_tensor(samples: list[StepSample], policy: TorchPPOPolicy):
    import torch

    action_size = int(getattr(policy, "action_size", 0))
    context_size = int(getattr(policy, "context_action_feature_size", 0))
    if (
        context_size <= 0
        or action_size <= 0
        or not samples
    ):
        return None
    rows = []
    for sample in samples:
        value = sample.action_context_features
        if value is None:
            value = np.zeros(
                (action_size, context_size),
                dtype=np.float32,
            )
        value = np.asarray(value, dtype=np.float32)
        if value.ndim != 2 or value.shape[0] != action_size:
            raise ValueError("stored action context features have unexpected shape")
        value = _resize_context_array(
            value,
            feature_size=context_size,
        )
        rows.append(value)
    return torch.as_tensor(
        np.stack(rows, axis=0),
        dtype=torch.float32,
        device=policy.device,
    )


def _policy_observation_array(policy: TorchPPOPolicy, samples: list[StepSample]) -> np.ndarray:
    raw = np.stack([sample.observation for sample in samples], axis=0).astype(np.float32)
    normalizer = getattr(policy, "normalize_observation_array", None)
    if callable(normalizer):
        return np.asarray(normalizer(raw), dtype=np.float32)
    return np.stack(
        [_normalize_observation(sample.observation) for sample in samples],
        axis=0,
    ).astype(np.float32)


def _resize_context_array(value: np.ndarray, *, feature_size: int) -> np.ndarray:
    if value.shape[1] == feature_size:
        return value
    if value.shape[1] > feature_size:
        return value[:, :feature_size]
    padded = np.zeros((value.shape[0], feature_size), dtype=np.float32)
    padded[:, : value.shape[1]] = value
    return padded


def _resize_context_tensor(value, *, feature_size: int):
    if int(value.shape[2]) == feature_size:
        return value
    if int(value.shape[2]) > feature_size:
        return value[:, :, :feature_size]
    import torch

    padded = torch.zeros(
        int(value.shape[0]),
        int(value.shape[1]),
        feature_size,
        dtype=value.dtype,
        device=value.device,
    )
    padded[:, :, : int(value.shape[2])] = value
    return padded


def make_ppo_optimizer(
    policy: TorchPPOPolicy,
    *,
    learning_rate: float,
    trunk_lr_mult: float = 1.0,
):
    import torch

    trunk_lr_mult = float(trunk_lr_mult)
    if not math.isfinite(trunk_lr_mult) or not 0.0 < trunk_lr_mult <= 1.0:
        raise ValueError("trunk_lr_mult must be finite and in (0, 1]")
    if _is_entity_graph_policy(policy) and trunk_lr_mult != 1.0:
        # The entity model scores actions directly from its shared state, so
        # "head" means every action/value-specific late module. Everything
        # else is the representation trunk protected by the lower LR.
        head_prefixes = (
            "action_",
            "legal_action_value_",
            "static_action_residual_proj",
            "target_gather_proj",
            "edge_policy_",
            "logit_scale",
            "value_",
            "q_head",
            "final_vp_head",
            "aux_",
            "belief_resource_head",
            "deliberation_halt_head",
        )
        trunk_parameters = []
        head_parameters = []
        for name, parameter in policy.model.named_parameters():
            if not parameter.requires_grad:
                continue
            destination = (
                head_parameters
                if name.startswith(head_prefixes)
                else trunk_parameters
            )
            destination.append(parameter)
        if not trunk_parameters or not head_parameters:
            raise ValueError("entity_graph PPO optimizer could not partition trunk and heads")
        return torch.optim.Adam(
            [
                {
                    "name": "protected_trunk",
                    "params": trunk_parameters,
                    "lr": float(learning_rate) * trunk_lr_mult,
                },
                {
                    "name": "policy_value_heads",
                    "params": head_parameters,
                    "lr": float(learning_rate),
                },
            ]
        )
    return torch.optim.Adam(_policy_parameters(policy), lr=learning_rate)


def evaluate_teacher_agreement(
    policy: TorchPPOPolicy,
    samples: list[StepSample],
) -> dict[str, float]:
    import torch

    if not samples:
        return {"samples": 0.0, "accuracy": 0.0, "mean_teacher_log_prob": 0.0}
    observations = _policy_observation_array(policy, samples)
    actions = np.asarray([sample.action for sample in samples], dtype=np.int64)
    valid_actions = [sample.valid_actions for sample in samples]
    obs_t = torch.as_tensor(observations, dtype=torch.float32, device=policy.device)
    context_t = _action_context_features_tensor(samples, policy)
    actions_t = torch.as_tensor(actions, dtype=torch.long, device=policy.device)
    with torch.no_grad():
        logits, _ = policy.forward(obs_t, context_t)
        masked = _masked_logits(logits, valid_actions, policy.action_size)
        dist = torch.distributions.Categorical(logits=masked)
        predictions = torch.argmax(masked, dim=-1)
        accuracy = (predictions == actions_t).float().mean()
        log_probs = dist.log_prob(actions_t).mean()
    return {
        "samples": float(len(samples)),
        "accuracy": float(accuracy.item()),
        "mean_teacher_log_prob": float(log_probs.item()),
    }


def imitation_update(
    policy: TorchPPOPolicy,
    samples: list[StepSample],
    *,
    learning_rate: float,
    epochs: int,
    minibatch_size: int,
    optimizer: Any | None = None,
    returns: list[float] | None = None,
    value_coef: float = 0.0,
    hard_target_weight: float = 0.0,
    score_coef: float = 0.0,
) -> dict[str, float]:
    import torch
    from torch import nn

    if not samples:
        return {
            "samples": 0.0,
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "score_loss": 0.0,
            "mean_sample_weight": 0.0,
        }
    optimizer = optimizer or make_imitation_optimizer(
        policy,
        learning_rate=learning_rate,
        train_critic=returns is not None and value_coef > 0.0,
    )
    observations = _policy_observation_array(policy, samples)
    actions = np.asarray([sample.action for sample in samples], dtype=np.int64)
    valid_actions = [sample.valid_actions for sample in samples]
    obs_t = torch.as_tensor(observations, dtype=torch.float32, device=policy.device)
    context_t = _action_context_features_tensor(samples, policy)
    actions_t = torch.as_tensor(actions, dtype=torch.long, device=policy.device)
    weights_t = _sample_weight_tensor(samples, policy.device)
    targets_t = _target_policy_tensor(
        samples,
        policy.action_size,
        policy.device,
        hard_target_weight=hard_target_weight,
    )
    score_targets_t, score_mask_t = _target_score_tensors(
        samples,
        policy.action_size,
        policy.device,
    )
    returns_t = (
        torch.as_tensor(returns, dtype=torch.float32, device=policy.device)
        if returns is not None
        else None
    )
    indices = np.arange(len(samples))
    last_loss = 0.0
    last_policy_loss = 0.0
    last_value_loss = 0.0
    last_score_loss = 0.0
    for _ in range(epochs):
        np.random.shuffle(indices)
        for start in range(0, len(samples), minibatch_size):
            batch_idx = indices[start : start + minibatch_size]
            batch_context = context_t[batch_idx] if context_t is not None else None
            logits, values = policy.forward(obs_t[batch_idx], batch_context)
            masked = _masked_logits(
                logits,
                [valid_actions[int(i)] for i in batch_idx],
                policy.action_size,
            )
            if targets_t is not None:
                log_probs = nn.functional.log_softmax(masked, dim=-1)
                policy_loss = _weighted_mean(
                    -(targets_t[batch_idx] * log_probs).sum(dim=-1),
                    weights_t[batch_idx],
                )
            else:
                policy_loss = _weighted_mean(
                    nn.functional.cross_entropy(
                        masked,
                        actions_t[batch_idx],
                        reduction="none",
                    ),
                    weights_t[batch_idx],
                )
            if returns_t is not None and value_coef > 0.0:
                value_loss = _weighted_mean(
                    nn.functional.mse_loss(
                        values,
                        returns_t[batch_idx],
                        reduction="none",
                    ),
                    weights_t[batch_idx],
                )
            else:
                value_loss = values.new_tensor(0.0)
            if score_coef > 0.0 and score_targets_t is not None and score_mask_t is not None:
                score_loss = _score_margin_loss(
                    masked,
                    score_targets_t[batch_idx],
                    score_mask_t[batch_idx],
                    weights=weights_t[batch_idx],
                )
            else:
                score_loss = values.new_tensor(0.0)
            loss = policy_loss + value_coef * value_loss + score_coef * score_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                _policy_parameters(policy, include_critic=returns_t is not None and value_coef > 0.0),
                1.0,
            )
            optimizer.step()
            last_loss = float(loss.item())
            last_policy_loss = float(policy_loss.item())
            last_value_loss = float(value_loss.item())
            last_score_loss = float(score_loss.item())
    return {
        "samples": float(len(samples)),
        "loss": last_loss,
        "policy_loss": last_policy_loss,
        "value_loss": last_value_loss,
        "score_loss": last_score_loss,
        "mean_sample_weight": float(weights_t.mean().item()),
    }


def make_imitation_optimizer(
    policy: TorchPPOPolicy,
    *,
    learning_rate: float,
    train_critic: bool = False,
):
    import torch

    return torch.optim.Adam(
        _policy_parameters(policy, include_critic=train_critic),
        lr=learning_rate,
    )


def _policy_parameters(policy: TorchPPOPolicy, *, include_critic: bool = True):
    if not hasattr(policy, "actor"):
        return list(policy.model.parameters())
    params = list(policy.model.parameters()) + list(policy.actor.parameters())
    if policy.action_encoder is not None:
        params += list(policy.action_encoder.parameters())
    if policy.action_id_embedding is not None:
        params += list(policy.action_id_embedding.parameters())
    if policy.action_bias is not None:
        params += list(policy.action_bias.parameters())
    if include_critic:
        params += list(policy.critic.parameters())
        if policy.q_head is not None:
            params += list(policy.q_head.parameters())
        if policy.q_state is not None:
            params += list(policy.q_state.parameters())
        if policy.q_action_encoder is not None:
            params += list(policy.q_action_encoder.parameters())
        if policy.q_action_bias is not None:
            params += list(policy.q_action_bias.parameters())
    return params


ACTION_TYPES = (
    "BUILD_CITY",
    "BUILD_ROAD",
    "BUILD_SETTLEMENT",
    "BUY_DEVELOPMENT_CARD",
    "DISCARD_RESOURCE",
    "END_TURN",
    "MARITIME_TRADE",
    "MOVE_ROBBER",
    "PLAY_KNIGHT_CARD",
    "PLAY_MONOPOLY",
    "PLAY_ROAD_BUILDING",
    "PLAY_YEAR_OF_PLENTY",
    "ROLL",
    "accept_trade",
    "cancel_trade",
    "confirm_trade",
    "offer_trade",
    "reject_trade",
)
RESOURCE_ORDER = ("BRICK", "ORE", "SHEEP", "WHEAT", "WOOD")
PLAYER_ORDER = ("BLUE", "RED", "ORANGE", "WHITE")
NUMERIC_SLOTS = 8
EXTRA_SCALARS = 4
ACTION_FEATURE_SIZE = (
    len(ACTION_TYPES)
    + 1
    + NUMERIC_SLOTS
    + len(RESOURCE_ORDER) * 2
    + len(PLAYER_ORDER)
    + EXTRA_SCALARS
)


def build_action_feature_table(env: ColonistMultiAgentEnv) -> np.ndarray:
    """Build static structured features for every discrete action id.

    This is the first bridge from the flat Catanatron-compatible action ids to
    the roadmap's legal-action scoring model. It intentionally uses only
    action-catalog semantics, not the current hidden game state.
    """
    table = np.zeros((env.action_space.n, ACTION_FEATURE_SIZE), dtype=np.float32)
    for action_index in range(env.action_space.n):
        description = env.describe_action(action_index)
        if description is None:
            continue
        table[action_index] = _action_feature_vector(
            description,
            action_size=env.action_space.n,
        )
    return table


def _action_feature_vector(
    description: dict[str, Any],
    *,
    action_size: int,
) -> np.ndarray:
    features = np.zeros(ACTION_FEATURE_SIZE, dtype=np.float32)
    cursor = 0
    action_type = str(description["action_type"])
    if action_type in ACTION_TYPES:
        features[cursor + ACTION_TYPES.index(action_type)] = 1.0
    cursor += len(ACTION_TYPES)

    action_index = int(description["index"])
    features[cursor] = action_index / max(float(action_size - 1), 1.0)
    cursor += 1

    value = description.get("value")
    numeric_values = _numeric_values(value)
    for offset, number in enumerate(numeric_values[:NUMERIC_SLOTS]):
        features[cursor + offset] = max(min(float(number) / 54.0, 1.0), -1.0)
    cursor += NUMERIC_SLOTS

    give, receive = _resource_flow(action_type, value)
    features[cursor : cursor + len(RESOURCE_ORDER)] = give
    cursor += len(RESOURCE_ORDER)
    features[cursor : cursor + len(RESOURCE_ORDER)] = receive
    cursor += len(RESOURCE_ORDER)

    players = set(_string_values(value))
    for offset, player in enumerate(PLAYER_ORDER):
        if player in players:
            features[cursor + offset] = 1.0
    cursor += len(PLAYER_ORDER)

    give_total = float(give.sum())
    receive_total = float(receive.sum())
    features[cursor] = give_total / 4.0
    features[cursor + 1] = receive_total / 4.0
    features[cursor + 2] = 1.0 if action_type in ("offer_trade", "MARITIME_TRADE") else 0.0
    features[cursor + 3] = 1.0 if value is None else 0.0
    return features


def _resource_flow(action_type: str, value: Any) -> tuple[np.ndarray, np.ndarray]:
    give = np.zeros(len(RESOURCE_ORDER), dtype=np.float32)
    receive = np.zeros(len(RESOURCE_ORDER), dtype=np.float32)
    if action_type == "DISCARD_RESOURCE":
        _add_resource(give, value)
    elif action_type in ("PLAY_MONOPOLY", "PLAY_YEAR_OF_PLENTY"):
        for resource in _string_values(value):
            _add_resource(receive, resource)
    elif action_type == "MARITIME_TRADE" and isinstance(value, tuple):
        for resource in value[:4]:
            _add_resource(give, resource)
        for resource in value[4:]:
            _add_resource(receive, resource)
    elif action_type == "offer_trade" and isinstance(value, tuple) and len(value) >= 10:
        give[:] = np.asarray(value[:5], dtype=np.float32)
        receive[:] = np.asarray(value[5:10], dtype=np.float32)
    return give, receive


def _add_resource(target: np.ndarray, resource: Any) -> None:
    if resource in RESOURCE_ORDER:
        target[RESOURCE_ORDER.index(resource)] += 1.0


def _numeric_values(value: Any) -> list[float]:
    if value is None or isinstance(value, str):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, (tuple, list)):
        values: list[float] = []
        for item in value:
            values.extend(_numeric_values(item))
        return values
    return []


def _string_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (tuple, list)):
        values: list[str] = []
        for item in value:
            values.extend(_string_values(item))
        return values
    return []


def _target_policy_tensor(
    samples: list[StepSample],
    action_size: int,
    device: Any,
    *,
    hard_target_weight: float = 0.0,
):
    import torch

    has_soft_targets = any(sample.target_policy for sample in samples)
    if not has_soft_targets:
        return None
    hard_target_weight = min(max(float(hard_target_weight), 0.0), 1.0)
    targets = torch.zeros((len(samples), action_size), dtype=torch.float32, device=device)
    for row, sample in enumerate(samples):
        if sample.target_policy:
            total = 0.0
            valid = set(sample.valid_actions)
            for action, weight in sample.target_policy.items():
                action_index = int(action)
                if action_index in valid and 0 <= action_index < action_size:
                    targets[row, action_index] = float(weight)
                    total += float(weight)
            if total > 0.0:
                targets[row] /= total
                if hard_target_weight > 0.0:
                    action_index = int(sample.action)
                    if 0 <= action_index < action_size:
                        targets[row] *= 1.0 - hard_target_weight
                        targets[row, action_index] += hard_target_weight
                continue
        targets[row, int(sample.action)] = 1.0
    return targets


def _target_score_tensors(
    samples: list[StepSample],
    action_size: int,
    device: Any,
):
    import torch

    if not any(sample.target_scores for sample in samples):
        return None, None
    targets = torch.zeros((len(samples), action_size), dtype=torch.float32, device=device)
    mask = torch.zeros((len(samples), action_size), dtype=torch.bool, device=device)
    for row, sample in enumerate(samples):
        if not sample.target_scores:
            continue
        valid = set(sample.valid_actions)
        scored: list[tuple[int, float]] = []
        for action, score in sample.target_scores.items():
            action_index = int(action)
            score_value = float(score)
            if action_index in valid and 0 <= action_index < action_size and np.isfinite(score_value):
                scored.append((action_index, score_value))
        if len(scored) < 2:
            continue
        values = np.asarray([score for _, score in scored], dtype=np.float32)
        std = float(values.std())
        if std <= 1e-6:
            normalized = np.zeros_like(values)
        else:
            normalized = (values - float(values.mean())) / std
        normalized = np.clip(normalized, -5.0, 5.0)
        for (action_index, _), score_value in zip(scored, normalized):
            targets[row, action_index] = float(score_value)
            mask[row, action_index] = True
    if not bool(mask.any().item()):
        return None, None
    return targets, mask


def _score_margin_loss(logits, targets, mask, *, weights=None):
    import torch

    row_has_scores = mask.sum(dim=-1) >= 2
    if not bool(row_has_scores.any().item()):
        return logits.new_tensor(0.0)
    if weights is None:
        row_weights = torch.ones(
            logits.shape[0],
            dtype=logits.dtype,
            device=logits.device,
        )
    else:
        row_weights = torch.as_tensor(
            weights,
            dtype=logits.dtype,
            device=logits.device,
        ).clamp_min(0.0)
        if row_weights.ndim != 1 or row_weights.shape[0] != logits.shape[0]:
            raise ValueError("score-margin weights must be a 1D tensor matching rows")
    logits = logits[row_has_scores]
    targets = targets[row_has_scores]
    mask = mask[row_has_scores]
    row_weights = row_weights[row_has_scores]
    masked_logits = torch.where(mask, logits, torch.zeros_like(logits))
    counts = mask.sum(dim=-1, keepdim=True).clamp_min(1)
    means = masked_logits.sum(dim=-1, keepdim=True) / counts
    centered = torch.where(mask, logits - means, torch.zeros_like(logits))
    variances = (centered.square().sum(dim=-1, keepdim=True) / counts).clamp_min(1e-6)
    normalized_logits = centered / torch.sqrt(variances)
    losses = []
    loss_weights = []
    for row in range(normalized_logits.shape[0]):
        row_mask = mask[row]
        row_logits = normalized_logits[row][row_mask]
        row_targets = targets[row][row_mask]
        if row_logits.numel() < 2:
            continue
        logit_diffs = row_logits[:, None] - row_logits[None, :]
        target_diffs = row_targets[:, None] - row_targets[None, :]
        pair_mask = target_diffs > 1e-6
        if not bool(pair_mask.any().item()):
            continue
        margins = target_diffs[pair_mask].abs().clamp_min(0.1).clamp_max(5.0)
        ordered_diffs = logit_diffs[pair_mask]
        losses.append(
            torch.nn.functional.softplus(margins - ordered_diffs).mean()
            * row_weights[row]
        )
        loss_weights.append(row_weights[row])
    if not losses:
        return logits.new_tensor(0.0)
    return torch.stack(losses).sum() / torch.stack(loss_weights).sum().clamp_min(1e-8)


def _sample_weight_tensor(samples: list[StepSample], device: Any):
    import torch

    weights = [
        max(0.0, float(getattr(sample, "sample_weight", 1.0)))
        for sample in samples
    ]
    if not any(weight > 0.0 for weight in weights):
        weights = [1.0 for _ in samples]
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def _weighted_mean(losses, weights):
    weights = weights.to(dtype=losses.dtype, device=losses.device).clamp_min(0.0)
    total = weights.sum()
    if float(total.item()) <= 1e-8:
        return losses.mean()
    return (losses * weights).sum() / total


def _masked_logits(logits, valid_actions: list[tuple[int, ...]], action_size: int):
    import torch

    mask = torch.full_like(logits, -1e9)
    # OPT-4: collapse the B per-row scatters into ONE advanced-index write.
    # valid_actions is a ragged list of tuples, so flatten to (row, col)
    # index lists first. Result is identical: every legal (row, action) cell
    # becomes 0.0 (so logits pass through) and every other cell stays -1e9;
    # rows with no legal actions contribute nothing and stay fully masked.
    rows = [row for row, actions in enumerate(valid_actions) for _ in actions]
    cols = [action for actions in valid_actions for action in actions]
    if cols:
        mask[rows, cols] = 0.0
    return logits + mask


def _normalize_observation(observation: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(
        np.asarray(observation, dtype=np.float32),
        nan=0.0,
        posinf=25.0,
        neginf=-25.0,
    )
    normalized = x.copy()
    large = np.abs(normalized) > 1.0
    normalized[large] = np.clip(normalized[large] / 25.0, -1.0, 1.0)
    return normalized


def _resize_observation(observation: np.ndarray, *, observation_size: int) -> np.ndarray:
    x = np.asarray(observation, dtype=np.float32)
    if x.shape[0] == observation_size:
        return x
    if x.shape[0] > observation_size:
        return x[:observation_size].copy()
    resized = np.zeros(observation_size, dtype=np.float32)
    resized[: x.shape[0]] = x
    return resized


def _scoreboard_rewards(env: ColonistMultiAgentEnv) -> dict[str, float]:
    values = _scoreboard_values(env)
    best = max(values.values())
    winners = [player for player, score in values.items() if score == best]
    if len(winners) == 1:
        return {
            player: 1.0 if player == winners[0] else -1.0 / (len(values) - 1)
            for player in values
        }
    return {player: 0.0 for player in values}


def _discounted_terminal_returns(
    players: list[str],
    rewards: dict[str, float],
    *,
    gamma: float,
) -> list[float]:
    counts_after = {player: 0 for player in rewards}
    returns = [0.0] * len(players)
    for idx in range(len(players) - 1, -1, -1):
        player = players[idx]
        returns[idx] = float(rewards[player]) * (gamma ** counts_after[player])
        counts_after[player] += 1
    return returns


def _gae_terminal_returns(
    players: list[str],
    rewards: dict[str, float],
    values: list[float],
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[list[float], list[float]]:
    return _gae_returns(
        players,
        rewards,
        values,
        [0.0] * len(players),
        gamma=gamma,
        gae_lambda=gae_lambda,
    )


def _gae_returns(
    players: list[str],
    rewards: dict[str, float],
    values: list[float],
    shaped_rewards: list[float],
    *,
    gamma: float,
    gae_lambda: float,
    bootstrap_values: dict[str, float] | None = None,
) -> tuple[list[float], list[float]]:
    if len(players) != len(values):
        raise ValueError("players and values must have the same length")
    if len(players) != len(shaped_rewards):
        raise ValueError("players and shaped_rewards must have the same length")
    bootstrap_values = bootstrap_values or {}
    next_value = {
        player: float(bootstrap_values.get(player, 0.0))
        for player in rewards
    }
    next_advantage = {player: 0.0 for player in rewards}
    has_future_sample = {
        player: player in bootstrap_values
        for player in rewards
    }
    returns = [0.0] * len(players)
    advantages = [0.0] * len(players)
    for idx in range(len(players) - 1, -1, -1):
        player = players[idx]
        reward = float(shaped_rewards[idx])
        if not has_future_sample[player]:
            reward += float(rewards[player])
        nonterminal = 1.0 if has_future_sample[player] else 0.0
        value = float(values[idx])
        delta = reward + gamma * next_value[player] * nonterminal - value
        advantage = (
            delta
            + gamma
            * gae_lambda
            * nonterminal
            * next_advantage[player]
        )
        advantages[idx] = advantage
        returns[idx] = advantage + value
        next_value[player] = value
        next_advantage[player] = advantage
        has_future_sample[player] = True
    return returns, advantages
