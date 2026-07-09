from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np

from catan_zero.rl.action_features import (
    CONTEXT_ACTION_FEATURE_SIZE,
    build_action_context_feature_table,
)
from catan_zero.rl.multiagent_env import ColonistMultiAgentConfig, ColonistMultiAgentEnv
from catan_zero.rl.torch_ppo import build_action_feature_table


@dataclass(frozen=True, slots=True)
class XDimLiteConfig:
    observation_size: int
    action_size: int
    static_action_feature_size: int
    context_action_feature_size: int = CONTEXT_ACTION_FEATURE_SIZE
    hidden_size: int = 512
    board_fraction: float = 0.75
    action_mask_version: str = ""


@dataclass(frozen=True, slots=True)
class XDimGraphConfig(XDimLiteConfig):
    token_count: int = 32
    board_layers: int = 4
    attention_heads: int = 8
    dropout: float = 0.05


class XDimLiteNet:
    """Small board/scalar/action-candidate policy for immediate self-play.

    This is intentionally "lite": it splits the current flat observation into a
    board-heavy prefix and scalar suffix, then scores legal action candidates
    from static action descriptors plus per-state action context features.
    """

    def __new__(cls, config: XDimLiteConfig):
        import torch
        from torch import nn

        class _Module(nn.Module):
            def __init__(self, cfg: XDimLiteConfig) -> None:
                super().__init__()
                self.config = cfg
                board_size = max(1, int(round(cfg.observation_size * cfg.board_fraction)))
                board_size = min(board_size, cfg.observation_size)
                scalar_size = max(1, cfg.observation_size - board_size)
                self.board_size = board_size
                self.scalar_size = scalar_size
                h = cfg.hidden_size
                action_size = (
                    cfg.static_action_feature_size + cfg.context_action_feature_size
                )
                self.board_encoder = nn.Sequential(
                    nn.Linear(board_size, h),
                    nn.LayerNorm(h),
                    nn.GELU(),
                    nn.Linear(h, h),
                    nn.GELU(),
                )
                self.scalar_encoder = nn.Sequential(
                    nn.Linear(scalar_size, h),
                    nn.LayerNorm(h),
                    nn.GELU(),
                    nn.Linear(h, h),
                    nn.GELU(),
                )
                self.state_encoder = nn.Sequential(
                    nn.Linear(2 * h, h),
                    nn.GELU(),
                    nn.Linear(h, h),
                )
                self.final_state_norm = nn.LayerNorm(h)
                self.action_encoder = nn.Sequential(
                    nn.Linear(action_size, h),
                    nn.LayerNorm(h),
                    nn.GELU(),
                    nn.Linear(h, h),
                )
                self.action_bias = nn.Linear(action_size, 1)
                self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0)))
                self.value_head = nn.Linear(h, 1)
                self.q_state = nn.Linear(h, h)
                self.q_action = nn.Linear(h, h)
                self.q_bias = nn.Linear(action_size, 1)
                self.final_vp_head = nn.Linear(h, 1)
                nn.init.zeros_(self.action_bias.weight)
                nn.init.zeros_(self.action_bias.bias)

            def forward(self, observations, action_features, *, return_q: bool = True):
                obs = _resize_tensor(observations, self.config.observation_size)
                board = obs[:, : self.board_size]
                scalar = obs[:, self.board_size :]
                scalar = _resize_tensor(scalar, self.scalar_size)
                state = self.state_encoder(
                    torch.cat((self.board_encoder(board), self.scalar_encoder(scalar)), dim=-1)
                )
                state = self.final_state_norm(state)
                encoded_actions = self.action_encoder(action_features)
                policy_state = torch.nn.functional.normalize(state, dim=-1)
                policy_actions = torch.nn.functional.normalize(encoded_actions, dim=-1)
                logit_scale = torch.clamp(self.logit_scale.exp(), max=50.0)
                logits = logit_scale * (policy_state.unsqueeze(1) * policy_actions).sum(dim=-1)
                logits = logits + self.action_bias(action_features).squeeze(-1)
                outputs = {
                    "logits": logits,
                    "value": self.value_head(state).squeeze(-1),
                    "final_vp": self.final_vp_head(state).squeeze(-1),
                }
                if return_q:
                    q_state = torch.nn.functional.normalize(self.q_state(state), dim=-1)
                    q_actions = torch.nn.functional.normalize(
                        self.q_action(encoded_actions), dim=-1
                    )
                    q_values = (q_state.unsqueeze(1) * q_actions).sum(dim=-1)
                    q_values = q_values + self.q_bias(action_features).squeeze(-1)
                    outputs["q_values"] = q_values
                return outputs

        return _Module(config)


class XDimGraphNet:
    """Token/message-mixing board-state encoder with legal-action scoring.

    The current teacher shards store the public observation as one vector, not
    separate hex/vertex/edge tensors. This model is the bridge architecture:
    it chunks that public vector into stable tokens, mixes them with small
    transformer-style message blocks, and keeps sparse legal-action scoring.
    """

    def __new__(cls, config: XDimGraphConfig):
        import torch
        from torch import nn

        class _GraphBlock(nn.Module):
            def __init__(self, width: int, heads: int, dropout: float) -> None:
                super().__init__()
                self.norm_attn = nn.LayerNorm(width)
                self.attn = nn.MultiheadAttention(
                    width,
                    max(1, int(heads)),
                    dropout=float(dropout),
                    batch_first=True,
                )
                self.norm_ff = nn.LayerNorm(width)
                self.ff = nn.Sequential(
                    nn.Linear(width, 4 * width),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(4 * width, width),
                    nn.Dropout(float(dropout)),
                )

            def forward(self, tokens):
                attn_in = self.norm_attn(tokens)
                attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
                tokens = tokens + attn_out
                tokens = tokens + self.ff(self.norm_ff(tokens))
                return tokens

        class _Module(nn.Module):
            def __init__(self, cfg: XDimGraphConfig) -> None:
                super().__init__()
                self.config = cfg
                h = int(cfg.hidden_size)
                self.token_count = max(4, int(cfg.token_count))
                self.chunk_size = int(np.ceil(float(cfg.observation_size) / self.token_count))
                self.token_projection = nn.Sequential(
                    nn.Linear(self.chunk_size, h),
                    nn.LayerNorm(h),
                    nn.GELU(),
                )
                self.token_embedding = nn.Parameter(torch.zeros(self.token_count + 1, h))
                self.cls_token = nn.Parameter(torch.zeros(1, 1, h))
                self.blocks = nn.ModuleList(
                    _GraphBlock(h, cfg.attention_heads, cfg.dropout)
                    for _ in range(max(1, int(cfg.board_layers)))
                )
                self.state_norm = nn.LayerNorm(h)
                self.pooled_state_norm = nn.LayerNorm(h)
                self.state_encoder = nn.Sequential(
                    nn.Linear(2 * h, h),
                    nn.GELU(),
                    nn.LayerNorm(h),
                    nn.Linear(h, h),
                )
                self.final_state_norm = nn.LayerNorm(h)
                action_size = (
                    cfg.static_action_feature_size + cfg.context_action_feature_size
                )
                self.action_encoder = nn.Sequential(
                    nn.Linear(action_size, h),
                    nn.LayerNorm(h),
                    nn.GELU(),
                    nn.Linear(h, h),
                    nn.GELU(),
                    nn.Linear(h, h),
                )
                self.action_bias = nn.Linear(action_size, 1)
                self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0)))
                self.value_head = nn.Sequential(
                    nn.Linear(h, h),
                    nn.GELU(),
                    nn.Linear(h, 1),
                )
                self.q_state = nn.Linear(h, h)
                self.q_action = nn.Linear(h, h)
                self.q_bias = nn.Linear(action_size, 1)
                self.final_vp_head = nn.Sequential(
                    nn.Linear(h, h // 2),
                    nn.GELU(),
                    nn.Linear(h // 2, 1),
                )
                nn.init.normal_(self.token_embedding, std=0.02)
                nn.init.normal_(self.cls_token, std=0.02)
                nn.init.zeros_(self.action_bias.weight)
                nn.init.zeros_(self.action_bias.bias)

            def forward(self, observations, action_features, *, return_q: bool = True):
                obs = _resize_tensor(observations, self.config.observation_size)
                padded_size = self.token_count * self.chunk_size
                obs = _resize_tensor(obs, padded_size)
                chunks = obs.reshape(obs.shape[0], self.token_count, self.chunk_size)
                tokens = self.token_projection(chunks)
                cls = self.cls_token.expand(obs.shape[0], -1, -1)
                tokens = torch.cat((cls, tokens), dim=1)
                tokens = tokens + self.token_embedding.unsqueeze(0)
                for block in self.blocks:
                    tokens = block(tokens)
                cls_state = self.state_norm(tokens[:, 0])
                pooled = self.pooled_state_norm(tokens[:, 1:].mean(dim=1))
                state = self.state_encoder(torch.cat((cls_state, pooled), dim=-1))
                state = self.final_state_norm(state)
                encoded_actions = self.action_encoder(action_features)
                policy_state = torch.nn.functional.normalize(state, dim=-1)
                policy_actions = torch.nn.functional.normalize(encoded_actions, dim=-1)
                logit_scale = torch.clamp(self.logit_scale.exp(), max=50.0)
                logits = logit_scale * (policy_state.unsqueeze(1) * policy_actions).sum(dim=-1)
                logits = logits + self.action_bias(action_features).squeeze(-1)
                outputs = {
                    "logits": logits,
                    "value": self.value_head(state).squeeze(-1),
                    "final_vp": self.final_vp_head(state).squeeze(-1),
                }
                if return_q:
                    q_state = torch.nn.functional.normalize(self.q_state(state), dim=-1)
                    q_actions = torch.nn.functional.normalize(
                        self.q_action(encoded_actions), dim=-1
                    )
                    q_values = (q_state.unsqueeze(1) * q_actions).sum(dim=-1)
                    q_values = q_values + self.q_bias(action_features).squeeze(-1)
                    outputs["q_values"] = q_values
                return outputs

        return _Module(config)


class XDimLitePolicy:
    name = "xdim_lite"
    policy_type = "xdim_lite"

    def __init__(
        self,
        config: XDimLiteConfig,
        static_action_features: np.ndarray,
        *,
        seed: int = 0,
        device: str | None = None,
    ) -> None:
        import torch

        torch.manual_seed(seed)
        self.config = config
        self.architecture = self.policy_type
        self.action_size = int(config.action_size)
        self.context_action_feature_size = int(config.context_action_feature_size)
        self.device = _resolve_device(device)
        self.static_action_features = torch.as_tensor(
            static_action_features,
            dtype=torch.float32,
            device=self.device,
        )
        self.model = XDimLiteNet(config).to(self.device)

    @classmethod
    def create(
        cls,
        *,
        env_config: ColonistMultiAgentConfig | None = None,
        hidden_size: int = 512,
        seed: int = 0,
        device: str | None = None,
    ) -> XDimLitePolicy:
        env = ColonistMultiAgentEnv(env_config or ColonistMultiAgentConfig())
        try:
            observations, info = env.reset(seed=seed)
            static = build_action_feature_table(env)
            config = XDimLiteConfig(
                observation_size=len(next(iter(observations.values()))),
                action_size=env.action_space.n,
                static_action_feature_size=int(static.shape[1]),
                hidden_size=hidden_size,
                action_mask_version=str(info.get("action_mask_version", "")),
            )
            return cls(config, static, seed=seed, device=device)
        finally:
            env.close()

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

        self._assert_action_mask_version(info)
        valid_actions = tuple(int(action) for action in info["valid_actions"])
        with torch.no_grad():
            logits = self._logits(
                np.asarray(observation, dtype=np.float32)[None, :],
                build_action_context_feature_table(env, info)[None, :, :],
            )
            masked = masked_logits(logits, [valid_actions], self.config.action_size)
            if training:
                return int(torch.distributions.Categorical(logits=masked).sample().item())
            return int(torch.argmax(masked, dim=-1).item())

    def sample_action_value(
        self,
        observation: np.ndarray,
        valid_actions: tuple[int, ...],
        action_context_features: np.ndarray,
    ) -> tuple[int, float, float, np.ndarray]:
        import torch

        with torch.no_grad():
            outputs = self.forward_np(
                np.asarray(observation, dtype=np.float32)[None, :],
                action_context_features[None, :, :],
            )
            masked = masked_logits(outputs["logits"], [valid_actions], self.config.action_size)
            dist = torch.distributions.Categorical(logits=masked)
            action = dist.sample()
            probs = torch.softmax(masked.squeeze(0), dim=-1)
            valid_probs = probs[
                torch.as_tensor(valid_actions, dtype=torch.long, device=self.device)
            ]
            return (
                int(action.item()),
                float(dist.log_prob(action).item()),
                float(outputs["value"].item()),
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
            outputs = self.forward_np(
                np.asarray(observation, dtype=np.float32)[None, :],
                self._context_array_or_zeros(action_context_features)[None, :, :],
            )
            masked = masked_logits(outputs["logits"], [valid_actions], self.action_size)
            dist = torch.distributions.Categorical(logits=masked)
            action = dist.sample()
            probs = torch.softmax(masked.squeeze(0), dim=-1)
            valid_tensor = torch.as_tensor(
                valid_actions,
                dtype=torch.long,
                device=self.device,
            )
            valid_probs = probs[valid_tensor]
            q_values = outputs["q_values"].squeeze(0)
            valid_q_values = q_values[valid_tensor]
            action_index = int(action.item())
            return (
                action_index,
                float(dist.log_prob(action).item()),
                float(outputs["value"].item()),
                float(q_values[action_index].item()),
                valid_probs.detach().cpu().numpy().astype(np.float32),
                valid_q_values.detach().cpu().numpy().astype(np.float32),
            )

    def _observation_tensor(self, observation: np.ndarray):
        import torch

        return torch.as_tensor(
            normalize_observations(np.asarray(observation, dtype=np.float32)),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

    def normalize_observation_array(self, observations: np.ndarray) -> np.ndarray:
        return normalize_observations(observations)

    def forward(self, observations, action_context_features=None):
        context = self._context_tensor_or_zeros(
            action_context_features,
            batch_size=int(observations.shape[0]),
        )
        outputs = self.model(observations, self._combined_action_features(context))
        return outputs["logits"], outputs["value"]

    def q_values(self, observations, action_context_features=None):
        context = self._context_tensor_or_zeros(
            action_context_features,
            batch_size=int(observations.shape[0]),
        )
        outputs = self.model(observations, self._combined_action_features(context))
        return outputs["q_values"]

    def forward_np(
        self,
        observations: np.ndarray,
        action_context_features: np.ndarray,
        *,
        return_q: bool = True,
    ):
        import torch

        observations = np.asarray(observations, dtype=np.float32)
        action_context_features = np.asarray(action_context_features, dtype=np.float32)
        self._assert_observation_width(observations)
        self._assert_context_width(action_context_features)
        obs = torch.as_tensor(
            normalize_observations(observations),
            dtype=torch.float32,
            device=self.device,
        )
        context = torch.as_tensor(
            action_context_features,
            dtype=torch.float32,
            device=self.device,
        )
        return self.model(obs, self._combined_action_features(context), return_q=return_q)

    def forward_legal_np(
        self,
        observations: np.ndarray,
        legal_action_ids: np.ndarray,
        legal_action_context: np.ndarray,
        *,
        return_q: bool = True,
    ):
        import torch

        observations = np.asarray(observations, dtype=np.float32)
        legal_action_context = np.asarray(legal_action_context, dtype=np.float32)
        self._assert_observation_width(observations)
        self._assert_context_width(legal_action_context)
        obs = torch.as_tensor(
            normalize_observations(observations),
            dtype=torch.float32,
            device=self.device,
        )
        action_ids = torch.as_tensor(
            legal_action_ids,
            dtype=torch.long,
            device=self.device,
        )
        valid = action_ids >= 0
        safe_ids = torch.clamp(action_ids, min=0)
        static = self.static_action_features.index_select(0, safe_ids.reshape(-1))
        static = static.reshape(*safe_ids.shape, -1)
        context = torch.as_tensor(
            legal_action_context,
            dtype=torch.float32,
            device=self.device,
        )
        outputs = self.model(obs, torch.cat((static, context), dim=-1), return_q=return_q)
        outputs["logits"] = outputs["logits"].masked_fill(~valid, -1.0e9)
        if "q_values" in outputs:
            outputs["q_values"] = outputs["q_values"].masked_fill(~valid, -1.0e9)
        return outputs

    def _logits(self, observations: np.ndarray, action_context_features: np.ndarray):
        return self.forward_np(observations, action_context_features)["logits"]

    def _combined_action_features(self, context):
        static = self.static_action_features.unsqueeze(0).expand(context.shape[0], -1, -1)
        self._assert_context_width(context)
        return static if context.numel() == 0 else __import__("torch").cat((static, context), dim=-1)

    def _assert_observation_width(self, observations) -> None:
        width = int(getattr(self.config, "observation_size", 0))
        if width and int(observations.shape[-1]) != width:
            raise ValueError(
                f"observation width {int(observations.shape[-1])} does not match "
                f"checkpoint observation_size {width}"
            )

    def _assert_context_width(self, context) -> None:
        width = int(getattr(self.config, "context_action_feature_size", 0))
        if width and int(context.shape[-1]) != width:
            raise ValueError(
                f"action context width {int(context.shape[-1])} does not match "
                f"checkpoint context_action_feature_size {width}"
            )

    def _assert_action_mask_version(self, info: dict[str, Any]) -> None:
        expected = str(getattr(self.config, "action_mask_version", "") or "")
        actual = str(info.get("action_mask_version", "") or "")
        if expected and not actual:
            raise ValueError(
                f"checkpoint action_mask_version {expected!r} cannot be verified "
                "because runtime info is missing action_mask_version"
            )
        if expected and actual != expected:
            raise ValueError(
                f"checkpoint action_mask_version {expected!r} does not match "
                f"runtime action_mask_version {actual!r}"
            )

    def _context_array_or_zeros(self, value: np.ndarray | None) -> np.ndarray:
        if value is None:
            return np.zeros(
                (self.action_size, self.context_action_feature_size),
                dtype=np.float32,
            )
        array = np.asarray(value, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError("action context features must be [actions, features]")
        if array.shape[0] != self.action_size:
            raise ValueError("action context row count does not match action space")
        if array.shape[1] != self.context_action_feature_size:
            raise ValueError(
                f"action context width {array.shape[1]} does not match "
                f"checkpoint context_action_feature_size {self.context_action_feature_size}"
            )
        return array

    def _context_tensor_or_zeros(self, value, *, batch_size: int):
        import torch

        if value is None:
            return torch.zeros(
                batch_size,
                self.action_size,
                self.context_action_feature_size,
                dtype=torch.float32,
                device=self.device,
            )
        context = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if context.shape[0] != batch_size or context.shape[1] != self.action_size:
            raise ValueError("action context features must be [batch, actions, features]")
        if context.shape[2] != self.context_action_feature_size:
            raise ValueError(
                f"action context width {context.shape[2]} does not match "
                f"checkpoint context_action_feature_size {self.context_action_feature_size}"
            )
        return context

    def save(self, path: str | Path) -> None:
        import torch

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        from catan_zero.rl.config_serialization import config_to_dict

        torch.save(
            {
                "policy_type": self.policy_type,
                # Durable name-keyed form (task #74) -- see config_serialization.
                "config": config_to_dict(self.config),
                "action_mask_version": str(getattr(self.config, "action_mask_version", "")),
                "static_action_features_sha256": _array_sha256(
                    self.static_action_features.detach().cpu().numpy()
                ),
                "static_action_features": self.static_action_features.detach().cpu(),
                "model": self.model.state_dict(),
            },
            output,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | None = None,
        strict_metadata: bool = True,
    ) -> XDimLitePolicy:
        import torch

        resolved = _resolve_device(device)
        _install_numpy_pickle_aliases()
        checkpoint = Path(path)
        try:
            data = torch.load(checkpoint, map_location=resolved, weights_only=False)
        except TypeError:
            data = torch.load(checkpoint, map_location=resolved)
        static = data["static_action_features"]
        if hasattr(static, "detach"):
            static = static.detach().cpu().numpy()
        # Task #74: new checkpoints store the config as a name-keyed dict.
        # Normalize to an attribute view so the getattr-based validation and
        # upgrade paths below handle both formats identically.
        from catan_zero.rl.config_serialization import config_attr_view

        if "config" in data:
            data = dict(data)
            data["config"] = config_attr_view(data["config"])
        _validate_checkpoint_metadata(data, static, checkpoint, strict=strict_metadata)
        if data.get("policy_type") == "xdim_graph":
            policy = XDimGraphPolicy(
                _upgrade_xdim_config(
                    data["config"],
                    policy_type="xdim_graph",
                    state_dict=data.get("model"),
                ),
                static,
                device=str(resolved),
            )
            _load_model_state_dict(policy.model, data["model"])
            policy.model.eval()
            return policy
        policy = cls(
            _upgrade_xdim_config(
                data["config"],
                policy_type="xdim_lite",
                state_dict=data.get("model"),
            ),
            static,
            device=str(resolved),
        )
        _load_model_state_dict(policy.model, data["model"])
        policy.model.eval()
        return policy


class XDimGraphPolicy(XDimLitePolicy):
    name = "xdim_graph"
    policy_type = "xdim_graph"

    def __init__(
        self,
        config: XDimGraphConfig,
        static_action_features: np.ndarray,
        *,
        seed: int = 0,
        device: str | None = None,
    ) -> None:
        import torch

        torch.manual_seed(seed)
        self.config = config
        self.architecture = self.policy_type
        self.action_size = int(config.action_size)
        self.context_action_feature_size = int(config.context_action_feature_size)
        self.device = _resolve_device(device)
        self.static_action_features = torch.as_tensor(
            static_action_features,
            dtype=torch.float32,
            device=self.device,
        )
        self.model = XDimGraphNet(config).to(self.device)

    @classmethod
    def create(
        cls,
        *,
        env_config: ColonistMultiAgentConfig | None = None,
        hidden_size: int = 768,
        seed: int = 0,
        device: str | None = None,
        token_count: int = 32,
        board_layers: int = 4,
        attention_heads: int = 8,
        dropout: float = 0.05,
    ) -> XDimGraphPolicy:
        env = ColonistMultiAgentEnv(env_config or ColonistMultiAgentConfig())
        try:
            observations, info = env.reset(seed=seed)
            static = build_action_feature_table(env)
            config = XDimGraphConfig(
                observation_size=len(next(iter(observations.values()))),
                action_size=env.action_space.n,
                static_action_feature_size=int(static.shape[1]),
                hidden_size=hidden_size,
                token_count=token_count,
                board_layers=board_layers,
                attention_heads=attention_heads,
                dropout=dropout,
                action_mask_version=str(info.get("action_mask_version", "")),
            )
            return cls(config, static, seed=seed, device=device)
        finally:
            env.close()

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | None = None,
        strict_metadata: bool = True,
    ) -> XDimGraphPolicy:
        import torch

        resolved = _resolve_device(device)
        _install_numpy_pickle_aliases()
        checkpoint = Path(path)
        try:
            data = torch.load(checkpoint, map_location=resolved, weights_only=False)
        except TypeError:
            data = torch.load(checkpoint, map_location=resolved)
        static = data["static_action_features"]
        if hasattr(static, "detach"):
            static = static.detach().cpu().numpy()
        # Task #74: new checkpoints store the config as a name-keyed dict.
        # Normalize to an attribute view so the getattr-based validation and
        # upgrade paths below handle both formats identically.
        from catan_zero.rl.config_serialization import config_attr_view

        if "config" in data:
            data = dict(data)
            data["config"] = config_attr_view(data["config"])
        _validate_checkpoint_metadata(data, static, checkpoint, strict=strict_metadata)
        policy = cls(
            _upgrade_xdim_config(
                data["config"],
                policy_type="xdim_graph",
                state_dict=data.get("model"),
            ),
            static,
            device=str(resolved),
        )
        _load_model_state_dict(policy.model, data["model"])
        policy.model.eval()
        return policy


def _install_numpy_pickle_aliases() -> None:
    """Allow NumPy-2 pickled checkpoints to load under NumPy-1 runtimes."""
    import sys

    try:
        import numpy.core as numpy_core
        import numpy.core.multiarray as numpy_multiarray
        import numpy.core.numeric as numpy_numeric
    except Exception:
        return
    sys.modules.setdefault("numpy._core", numpy_core)
    sys.modules.setdefault("numpy._core.multiarray", numpy_multiarray)
    sys.modules.setdefault("numpy._core.numeric", numpy_numeric)


def _array_sha256(array: np.ndarray) -> str:
    import hashlib

    values = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(values.dtype).encode("utf-8"))
    h.update(str(values.shape).encode("utf-8"))
    h.update(values.tobytes())
    return h.hexdigest()


def _validate_checkpoint_metadata(
    data: dict[str, Any],
    static_action_features: np.ndarray,
    checkpoint: Path,
    *,
    strict: bool,
) -> None:
    if not strict:
        return
    config = data.get("config")
    top_level_version = str(data.get("action_mask_version", "") or "")
    config_version = str(getattr(config, "action_mask_version", "") or "")
    if not top_level_version or not config_version:
        raise ValueError(
            f"{checkpoint} is missing XDim action_mask_version metadata; "
            "refuse to load it for production evaluation/training"
        )
    if top_level_version != config_version:
        raise ValueError(
            f"{checkpoint} has inconsistent XDim action_mask_version metadata: "
            f"top_level={top_level_version!r} config={config_version!r}"
        )
    expected_static_hash = str(data.get("static_action_features_sha256", "") or "")
    if not expected_static_hash:
        raise ValueError(
            f"{checkpoint} is missing static_action_features_sha256 metadata"
        )
    actual_static_hash = _array_sha256(np.asarray(static_action_features, dtype=np.float32))
    if actual_static_hash != expected_static_hash:
        raise ValueError(
            f"{checkpoint} static_action_features_sha256 mismatch: "
            f"checkpoint={expected_static_hash} actual={actual_static_hash}"
        )


def _load_model_state_dict(model, state_dict: dict[str, Any]) -> None:
    incompatible = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {
        "logit_scale",
        "pooled_state_norm.weight",
        "pooled_state_norm.bias",
        "final_state_norm.weight",
        "final_state_norm.bias",
    }
    missing = [
        key
        for key in incompatible.missing_keys
        if key not in allowed_missing
    ]
    unexpected = list(incompatible.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(
            "checkpoint model state_dict is incompatible: "
            f"missing={missing} unexpected={unexpected}"
        )


def _upgrade_xdim_config(
    config: Any,
    *,
    policy_type: str,
    state_dict: dict[str, Any] | None = None,
) -> XDimLiteConfig:
    """Normalize older pickled configs after dataclass fields are added."""

    mask_version = getattr(config, "action_mask_version", "") or ""
    if not isinstance(mask_version, str):
        mask_version = ""
    common = {
        "observation_size": int(getattr(config, "observation_size")),
        "action_size": int(getattr(config, "action_size")),
        "static_action_feature_size": int(getattr(config, "static_action_feature_size")),
        "context_action_feature_size": int(
            getattr(config, "context_action_feature_size", CONTEXT_ACTION_FEATURE_SIZE)
        ),
        "hidden_size": int(getattr(config, "hidden_size", 512)),
        "board_fraction": float(getattr(config, "board_fraction", 0.75)),
        "action_mask_version": mask_version,
    }
    if policy_type == "xdim_graph":
        token_count = _state_token_count(state_dict) or _positive_int(
            getattr(config, "token_count", None),
            default=32,
        )
        board_layers = _state_board_layers(state_dict) or _positive_int(
            getattr(config, "board_layers", None),
            default=4,
        )
        attention_heads = _positive_int(
            getattr(config, "attention_heads", None),
            default=8,
        )
        dropout = _bounded_float(getattr(config, "dropout", None), default=0.05)
        return XDimGraphConfig(
            **common,
            token_count=token_count,
            board_layers=board_layers,
            attention_heads=attention_heads,
            dropout=dropout,
        )
    return XDimLiteConfig(**common)


def _state_token_count(state_dict: dict[str, Any] | None) -> int | None:
    if not state_dict or "token_embedding" not in state_dict:
        return None
    shape = tuple(getattr(state_dict["token_embedding"], "shape", ()))
    if len(shape) < 1:
        return None
    value = int(shape[0]) - 1
    return value if value > 0 else None


def _state_board_layers(state_dict: dict[str, Any] | None) -> int | None:
    if not state_dict:
        return None
    indices = []
    for key in state_dict:
        parts = str(key).split(".")
        if len(parts) >= 3 and parts[0] == "blocks" and parts[2] == "norm_attn":
            try:
                indices.append(int(parts[1]))
            except ValueError:
                continue
    return max(indices) + 1 if indices else None


def _positive_int(value: Any, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return int(default)
    return parsed if parsed > 0 else int(default)


def _bounded_float(value: Any, *, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if 0.0 <= parsed < 1.0 else float(default)


def normalize_observations(observations: np.ndarray) -> np.ndarray:
    # np.nan_to_num (copy=True default) already returns a FRESH array, so the
    # subsequent .copy() was redundant -- mutate values in place instead. Output
    # is bit-identical and the caller's array is never touched (OPT-3).
    values = np.nan_to_num(
        np.asarray(observations, dtype=np.float32),
        nan=0.0,
        posinf=25.0,
        neginf=-25.0,
    )
    large = np.abs(values) > 1.0
    values[large] = np.clip(values[large] / 25.0, -1.0, 1.0)
    return values


def masked_logits(logits, valid_actions: list[tuple[int, ...]], action_size: int):
    import torch

    if logits.shape[-1] != action_size:
        raise ValueError("logit action dimension does not match action_size")
    mask = torch.zeros_like(logits, dtype=torch.bool)
    for row, actions in enumerate(valid_actions):
        if actions:
            indices = torch.as_tensor(actions, dtype=torch.long, device=logits.device)
            mask[row, indices] = True
    return logits.masked_fill(~mask, -1.0e9)


def _resolve_device(device: str | None):
    import torch

    requested = (device or "auto").lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    resolved = torch.device(requested)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return resolved


def _resize_tensor(tensor, size: int):
    import torch

    if tensor.shape[-1] == size:
        return tensor
    if tensor.shape[-1] > size:
        return tensor[..., :size]
    pad = torch.zeros(*tensor.shape[:-1], size - tensor.shape[-1], device=tensor.device)
    return torch.cat((tensor, pad), dim=-1)
