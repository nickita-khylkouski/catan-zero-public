"""Opt-in static-bucket CUDA Graph runner for EntityGraph inference.

Only the expensive, fixed-layout state trunk (``EntityGraphNet.encode_state``)
is captured.  The variable-width legal-action head remains eager, so a new
graph is not needed for every legal width and returned logits/Q values retain
the caller's exact legal width.

EvalServer can wire this runner behind an explicit default-false flag.  The
narrow ``forward_legal_np``-compatible API keeps it isolated for H100 canaries
before any production-default decision.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import operator
from typing import Any

import numpy as np

from catan_zero.rl.entity_token_policy import (
    _assert_entity_batch_shapes,
    entity_policy_uses_topology,
)


_STATE_INPUT_KEYS = (
    "hex_tokens",
    "hex_mask",
    "vertex_tokens",
    "vertex_mask",
    "edge_tokens",
    "edge_mask",
    "player_tokens",
    "player_mask",
    "global_tokens",
    "event_tokens",
    "event_mask",
)
_TOPOLOGY_STATE_INPUT_KEYS = (
    "hex_vertex_ids",
    "hex_edge_ids",
    "edge_vertex_ids",
    "event_target_ids",
)


@dataclass(frozen=True, slots=True)
class CudaGraphInferenceConfig:
    """Controls the opt-in state-trunk graph runner.

    ``batch_buckets`` are intentionally close around the measured H100 window
    sizes.  A window larger than the final bucket takes the eager fallback.
    ``event_token_limit=None`` preserves the historical event width; zero is a
    valid limit when every event position is masked.
    """

    enabled: bool = False
    # EvalServer's request cap is not a neural-row cap: chance fan-out can make
    # one request contain several rows, and retained traces reached 177 rows.
    batch_buckets: tuple[int, ...] = (
        8,
        16,
        24,
        32,
        40,
        48,
        64,
        80,
        96,
        128,
        160,
        192,
    )
    event_token_limit: int | None = None
    warmup_iterations: int = 3

    def __post_init__(self) -> None:
        buckets = tuple(operator.index(value) for value in self.batch_buckets)
        if not buckets or any(value <= 0 for value in buckets):
            raise ValueError("batch_buckets must contain positive integers")
        if buckets != tuple(sorted(set(buckets))):
            raise ValueError("batch_buckets must be strictly increasing and unique")
        if self.warmup_iterations < 1:
            raise ValueError("warmup_iterations must be at least 1")
        if isinstance(self.event_token_limit, bool):
            raise TypeError("event_token_limit must be an integer, not bool")
        if self.event_token_limit is not None:
            limit = operator.index(self.event_token_limit)
            if limit < 0:
                raise ValueError("event_token_limit must be non-negative")


@dataclass(slots=True)
class _GraphEntry:
    graph: Any
    static_inputs: dict[str, Any]
    encoded_state: tuple[Any, Any, Any]
    capture_stream: Any


class CudaGraphInferenceRunner:
    """Run an EntityGraph policy with a captured state trunk when supported.

    The public method intentionally matches ``EntityGraphPolicy.forward_legal_np``.
    Unsupported devices/configurations and graph-capture failures use the same
    split model in eager mode.  Model parameters must not be replaced after a
    graph has been captured; in-place checkpoint weight copies retain their
    addresses and are safe.
    """

    def __init__(
        self,
        policy: Any,
        config: CudaGraphInferenceConfig | None = None,
    ) -> None:
        self.policy = policy
        self.model = policy.model
        self.runner_config = config or CudaGraphInferenceConfig()
        self.device = self._resolve_device(policy)
        self._graphs: dict[tuple[Any, ...], _GraphEntry] = {}
        self._capture_failures: dict[tuple[Any, ...], str] = {}
        self.last_path = "not_run"
        self.last_fallback_reason: str | None = None

    @property
    def graph_count(self) -> int:
        return len(self._graphs)

    @property
    def config(self) -> Any:
        """Expose the wrapped model config for policy-compatible handshakes."""
        return self.policy.config

    def __getattr__(self, name: str) -> Any:
        """Delegate policy metadata while keeping runner controls separate."""
        policy = self.__dict__.get("policy")
        if policy is None:
            raise AttributeError(name)
        return getattr(policy, name)

    def selected_batch_bucket(self, rows: int) -> int | None:
        """Return the smallest configured bucket that can hold ``rows``."""
        rows = operator.index(rows)
        if rows <= 0:
            raise ValueError("rows must be positive")
        for bucket in self.runner_config.batch_buckets:
            if bucket >= rows:
                return bucket
        return None

    def forward_legal_np(
        self,
        entity_batch: dict[str, np.ndarray],
        legal_action_ids: np.ndarray,
        legal_action_context: np.ndarray,
        *,
        return_q: bool = False,
    ) -> dict[str, Any]:
        """Score a NumPy batch, returning tensors with unpadded output shapes."""
        import torch

        # Preserve EntityGraphPolicy.forward_legal_np's public input contract on
        # every path, including disabled and capture-failure eager fallbacks.
        # Validate before cropping so event masks/tokens are checked against the
        # caller's original padded representation.
        _assert_entity_batch_shapes(
            entity_batch,
            legal_action_ids,
            legal_action_context,
            self.config,
        )
        batch_size = self._validate_batch(
            entity_batch, legal_action_ids, legal_action_context
        )
        cropped_entity = self._crop_events(entity_batch)
        reason = self._unsupported_reason(batch_size)
        if reason is not None:
            self.last_path = (
                "eager_disabled" if not self.runner_config.enabled else "eager_fallback"
            )
            self.last_fallback_reason = reason
            return self._eager_forward(
                cropped_entity,
                legal_action_ids,
                legal_action_context,
                return_q=return_q,
            )

        bucket = self.selected_batch_bucket(batch_size)
        assert bucket is not None
        signature = self._graph_signature(cropped_entity, bucket)
        if signature in self._capture_failures:
            self.last_path = "eager_fallback"
            self.last_fallback_reason = self._capture_failures[signature]
            return self._eager_forward(
                cropped_entity,
                legal_action_ids,
                legal_action_context,
                return_q=return_q,
            )

        with (
            torch.no_grad(),
            torch.autocast(device_type=self.device.type, enabled=False),
            _strict_fp32(torch),
        ):
            entry = self._graphs.get(signature)
            if entry is None:
                try:
                    entry = self._capture_graph(cropped_entity, bucket)
                except Exception as error:  # safe opt-in prototype fallback
                    reason = (
                        f"CUDA Graph capture failed: {type(error).__name__}: {error}"
                    )
                    self._capture_failures[signature] = reason
                    self.last_path = "eager_fallback"
                    self.last_fallback_reason = reason
                    return self._eager_forward(
                        cropped_entity,
                        legal_action_ids,
                        legal_action_context,
                        return_q=return_q,
                    )
                self._graphs[signature] = entry

            self._copy_state_inputs(
                cropped_entity,
                entry.static_inputs,
                batch_size=batch_size,
            )
            entry.graph.replay()
            encoded_state = tuple(value[:batch_size] for value in entry.encoded_state)
            action_batch, action_ids = self._action_batch(
                cropped_entity,
                legal_action_ids,
                legal_action_context,
            )
            outputs = self.model.score_actions(
                encoded_state,
                action_batch,
                return_q=return_q,
            )
            outputs["logits"] = outputs["logits"].masked_fill(action_ids < 0, -1.0e9)

        self.last_path = "cuda_graph"
        self.last_fallback_reason = None
        return outputs

    def _unsupported_reason(self, batch_size: int) -> str | None:
        if not self.runner_config.enabled:
            return "CUDA Graph inference is disabled"
        if self.device.type != "cuda":
            return f"CUDA Graph inference requires a CUDA device, got {self.device}"
        try:
            import torch
        except ImportError:
            return "PyTorch is unavailable"
        if not torch.cuda.is_available():
            return "torch.cuda.is_available() is false"
        if not hasattr(torch.cuda, "CUDAGraph"):
            return "this PyTorch build has no CUDA Graph API"
        if self.model.training:
            return "model is in training mode"
        if not callable(getattr(self.model, "encode_state", None)):
            return "model has no encode_state API"
        if not callable(getattr(self.model, "score_actions", None)):
            return "model has no score_actions API"
        if self.selected_batch_bucket(batch_size) is None:
            return (
                f"batch size {batch_size} exceeds largest graph bucket "
                f"{self.runner_config.batch_buckets[-1]}"
            )
        return None

    def _capture_graph(
        self,
        entity_batch: dict[str, np.ndarray],
        bucket: int,
    ) -> _GraphEntry:
        import torch

        static_inputs = self._allocate_state_inputs(entity_batch, bucket)
        self._copy_state_inputs(
            entity_batch, static_inputs, batch_size=len(entity_batch["hex_tokens"])
        )
        capture_stream = torch.cuda.Stream(device=self.device)
        capture_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(capture_stream):
            for _ in range(self.runner_config.warmup_iterations):
                self.model.encode_state(static_inputs)
        capture_stream.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            graph,
            stream=capture_stream,
            capture_error_mode="thread_local",
        ):
            encoded_state = self.model.encode_state(static_inputs)
        return _GraphEntry(
            graph=graph,
            static_inputs=static_inputs,
            encoded_state=encoded_state,
            capture_stream=capture_stream,
        )

    def _allocate_state_inputs(
        self,
        entity_batch: dict[str, np.ndarray],
        bucket: int,
    ) -> dict[str, Any]:
        import torch

        inputs = {}
        for key in self._state_input_keys():
            source = torch.as_tensor(entity_batch[key])
            inputs[key] = torch.zeros(
                (bucket, *source.shape[1:]),
                dtype=source.dtype,
                device=self.device,
            )
        return inputs

    def _copy_state_inputs(
        self,
        entity_batch: dict[str, np.ndarray],
        static_inputs: dict[str, Any],
        *,
        batch_size: int,
    ) -> None:
        import torch

        for key, destination in static_inputs.items():
            source = torch.as_tensor(entity_batch[key], dtype=destination.dtype)
            expected = (batch_size, *destination.shape[1:])
            if tuple(source.shape) != expected:
                raise ValueError(
                    f"{key} changed shape after graph selection: "
                    f"{tuple(source.shape)} != {expected}"
                )
            destination[:batch_size].copy_(source, non_blocking=False)

    def _action_batch(
        self,
        entity_batch: dict[str, np.ndarray],
        legal_action_ids: np.ndarray,
        legal_action_context: np.ndarray,
    ) -> tuple[dict[str, Any], Any]:
        import torch

        batch = {
            "legal_action_tokens": torch.as_tensor(
                entity_batch["legal_action_tokens"], device=self.device
            ),
            "legal_action_context": torch.as_tensor(
                legal_action_context,
                dtype=torch.float32,
                device=self.device,
            ),
        }
        needs_targets = bool(
            getattr(self.config, "action_target_gather", False)
            or getattr(self.config, "edge_policy_head", False)
        )
        if needs_targets:
            # _gather_target_tokens derives the fixed sequence offsets from
            # these three shapes.  Values are not read, so retain the host
            # arrays rather than launching three unnecessary H2D copies.
            batch["hex_tokens"] = entity_batch["hex_tokens"]
            batch["vertex_tokens"] = entity_batch["vertex_tokens"]
            batch["edge_tokens"] = entity_batch["edge_tokens"]
            batch["legal_action_target_ids"] = torch.as_tensor(
                entity_batch["legal_action_target_ids"],
                device=self.device,
            )
        action_ids = torch.as_tensor(
            legal_action_ids,
            dtype=torch.long,
            device=self.device,
        )
        return batch, action_ids

    def _eager_forward(
        self,
        entity_batch: dict[str, np.ndarray],
        legal_action_ids: np.ndarray,
        legal_action_context: np.ndarray,
        *,
        return_q: bool,
    ) -> dict[str, Any]:
        import torch

        with (
            torch.no_grad(),
            torch.autocast(device_type=self.device.type, enabled=False),
            _strict_fp32(torch),
        ):
            state_keys = self._state_input_keys()
            state_batch = {
                key: torch.as_tensor(value, device=self.device)
                for key, value in entity_batch.items()
                if key in state_keys
            }
            encoded_state = self.model.encode_state(state_batch)
            action_batch, action_ids = self._action_batch(
                entity_batch,
                legal_action_ids,
                legal_action_context,
            )
            outputs = self.model.score_actions(
                encoded_state,
                action_batch,
                return_q=return_q,
            )
            outputs["logits"] = outputs["logits"].masked_fill(action_ids < 0, -1.0e9)
            return outputs

    def _crop_events(
        self,
        entity_batch: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        limit = self.runner_config.event_token_limit
        if limit is None:
            return entity_batch
        limit = operator.index(limit)
        event_mask = _as_numpy(entity_batch["event_mask"]).astype(np.bool_, copy=False)
        event_tokens = _as_numpy(entity_batch["event_tokens"])
        if event_mask.ndim != 2 or event_tokens.ndim != 3:
            raise ValueError("event_mask/tokens must have ranks 2 and 3")
        if event_tokens.shape[:2] != event_mask.shape:
            raise ValueError("event_mask and event_tokens shapes do not match")
        if limit > event_mask.shape[1]:
            raise ValueError(
                f"event_token_limit {limit} exceeds event width {event_mask.shape[1]}"
            )
        if bool(event_mask[:, limit:].any()):
            raise ValueError(
                "event_token_limit would remove at least one unmasked event token"
            )
        cropped = dict(entity_batch)
        cropped["event_mask"] = event_mask[:, :limit]
        cropped["event_tokens"] = event_tokens[:, :limit]
        if "event_target_ids" in entity_batch:
            event_targets = _as_numpy(entity_batch["event_target_ids"])
            expected = (*event_mask.shape, 4)
            if event_targets.shape != expected:
                raise ValueError(
                    f"event_target_ids shape {event_targets.shape} != {expected}"
                )
            cropped["event_target_ids"] = event_targets[:, :limit]
        return cropped

    def _graph_signature(
        self,
        entity_batch: dict[str, np.ndarray],
        bucket: int,
    ) -> tuple[Any, ...]:
        fields = []
        for key in self._state_input_keys():
            value = _as_numpy(entity_batch[key])
            fields.append((key, value.dtype.str, tuple(value.shape[1:])))
        return (bucket, tuple(fields))

    def _state_input_keys(self) -> tuple[str, ...]:
        if entity_policy_uses_topology(self.config):
            return (*_STATE_INPUT_KEYS, *_TOPOLOGY_STATE_INPUT_KEYS)
        return _STATE_INPUT_KEYS

    @staticmethod
    def _resolve_device(policy: Any):
        import torch

        device = getattr(policy, "device", None)
        if device is not None:
            return torch.device(device)
        parameter = next(policy.model.parameters(), None)
        return parameter.device if parameter is not None else torch.device("cpu")

    @staticmethod
    def _validate_batch(
        entity_batch: dict[str, np.ndarray],
        legal_action_ids: np.ndarray,
        legal_action_context: np.ndarray,
    ) -> int:
        missing = [
            key
            for key in (*_STATE_INPUT_KEYS, "legal_action_tokens")
            if key not in entity_batch
        ]
        if missing:
            raise KeyError(f"entity batch is missing model inputs: {missing}")
        batch_size = int(np.shape(entity_batch["hex_tokens"])[0])
        if batch_size <= 0:
            raise ValueError("inference batch must contain at least one row")
        legal_ids_shape = np.shape(legal_action_ids)
        context_shape = np.shape(legal_action_context)
        token_shape = np.shape(entity_batch["legal_action_tokens"])
        if len(legal_ids_shape) != 2:
            raise ValueError("legal_action_ids must have rank 2")
        if len(context_shape) != 3 or len(token_shape) != 3:
            raise ValueError("legal action context/tokens must have rank 3")
        if (
            legal_ids_shape[:2] != context_shape[:2]
            or legal_ids_shape[:2] != token_shape[:2]
        ):
            raise ValueError("legal action arrays must agree on batch and width")
        if legal_ids_shape[0] != batch_size:
            raise ValueError("legal action batch size does not match state batch")
        for key in _STATE_INPUT_KEYS:
            if int(np.shape(entity_batch[key])[0]) != batch_size:
                raise ValueError(f"{key} batch size does not match hex_tokens")
        return batch_size


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


@contextmanager
def _strict_fp32(torch: Any):
    """Temporarily disable TF32 for both captured and eager matmuls."""
    previous_precision = torch.get_float32_matmul_precision()
    previous_cuda_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous_precision)
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32
        torch.backends.cuda.matmul.allow_tf32 = previous_cuda_tf32
