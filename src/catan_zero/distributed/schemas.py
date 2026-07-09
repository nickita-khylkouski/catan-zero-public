from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any

import numpy as np


Array = np.ndarray


@dataclass(frozen=True, slots=True)
class PolicyVersion:
    """Immutable policy identity published by the learner."""

    policy_id: str
    checkpoint_path: str
    architecture: str
    created_at: float = field(default_factory=time.time)
    parent_policy_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DecisionRecord:
    """One learner-owned decision from a game trajectory.

    The actor observation must already be legal for the acting player. Full
    hidden simulator truth belongs only in `teacher_payload`.
    """

    policy_id: str
    game_id: str
    seed: int
    decision_index: int
    player: str
    opponent_policy_ids: tuple[str, ...]
    observation: Array
    valid_actions: tuple[int, ...]
    action: int
    old_log_prob: float
    old_value: float
    return_: float
    advantage: float
    action_context_features: Array | None = None
    old_q_value: float | None = None
    old_action_probs: Array | None = None
    old_action_q_values: Array | None = None
    shaped_reward: float = 0.0
    sample_weight: float = 1.0
    failure_tags: tuple[str, ...] = ()
    teacher_payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.observation = _float_array(self.observation, ndim=1, name="observation")
        if self.action_context_features is not None:
            self.action_context_features = _float_array(
                self.action_context_features,
                ndim=2,
                name="action_context_features",
            )
        if self.old_action_probs is not None:
            self.old_action_probs = _float_array(
                self.old_action_probs,
                ndim=1,
                name="old_action_probs",
            )
        if self.old_action_q_values is not None:
            self.old_action_q_values = _float_array(
                self.old_action_q_values,
                ndim=1,
                name="old_action_q_values",
            )
        self.valid_actions = tuple(int(action) for action in self.valid_actions)
        self.action = int(self.action)
        if self.action not in self.valid_actions:
            raise ValueError("action must be present in valid_actions")


@dataclass(frozen=True, slots=True)
class ActorBatch:
    """A batch emitted by one actor process."""

    actor_id: str
    policy_id: str
    records: tuple[DecisionRecord, ...]
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def decisions(self) -> int:
        return len(self.records)


@dataclass(frozen=True, slots=True)
class LearnerBatch:
    """Dense arrays consumed by the learner."""

    policy_ids: tuple[str, ...]
    observations: Array
    actions: Array
    valid_actions: tuple[tuple[int, ...], ...]
    old_log_probs: Array
    old_values: Array
    returns: Array
    advantages: Array
    action_context_features: Array | None = None
    old_q_values: Array | None = None
    sample_weights: Array | None = None
    source_records: tuple[DecisionRecord, ...] = ()

    @property
    def size(self) -> int:
        return int(self.actions.shape[0])


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    request_id: str
    policy_id: str
    observation: Array
    valid_actions: tuple[int, ...]
    action_context_features: Array | None = None
    created_at: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class InferenceResponse:
    request_id: str
    policy_id: str
    action: int
    log_prob: float
    value: float
    valid_action_probs: Array
    q_value: float | None = None
    valid_action_q_values: Array | None = None
    created_at: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class QueueStats:
    queued_batches: int
    queued_decisions: int
    total_enqueued_batches: int
    total_enqueued_decisions: int
    total_dequeued_batches: int
    total_dequeued_decisions: int
    dropped_batches: int = 0
    dropped_decisions: int = 0


def records_to_learner_batch(records: list[DecisionRecord]) -> LearnerBatch:
    if not records:
        raise ValueError("records_to_learner_batch requires at least one record")
    observation_size = int(records[0].observation.shape[0])
    context_shape = (
        None
        if records[0].action_context_features is None
        else tuple(records[0].action_context_features.shape)
    )
    for record in records:
        if int(record.observation.shape[0]) != observation_size:
            raise ValueError("all observations in a learner batch must share shape")
        record_context_shape = (
            None
            if record.action_context_features is None
            else tuple(record.action_context_features.shape)
        )
        if record_context_shape != context_shape:
            raise ValueError(
                "all action_context_features in a learner batch must share shape"
            )
    context = (
        None
        if context_shape is None
        else np.stack([record.action_context_features for record in records]).astype(
            np.float32,
            copy=False,
        )
    )
    old_q_values = (
        None
        if any(record.old_q_value is None for record in records)
        else np.asarray([record.old_q_value for record in records], dtype=np.float32)
    )
    return LearnerBatch(
        policy_ids=tuple(record.policy_id for record in records),
        observations=np.stack([record.observation for record in records]).astype(
            np.float32,
            copy=False,
        ),
        actions=np.asarray([record.action for record in records], dtype=np.int64),
        valid_actions=tuple(record.valid_actions for record in records),
        old_log_probs=np.asarray(
            [record.old_log_prob for record in records],
            dtype=np.float32,
        ),
        old_values=np.asarray([record.old_value for record in records], dtype=np.float32),
        returns=np.asarray([record.return_ for record in records], dtype=np.float32),
        advantages=np.asarray(
            [record.advantage for record in records],
            dtype=np.float32,
        ),
        action_context_features=context,
        old_q_values=old_q_values,
        sample_weights=np.asarray(
            [max(0.0, float(record.sample_weight)) for record in records],
            dtype=np.float32,
        ),
        source_records=tuple(records),
    )


def _float_array(value: Any, *, ndim: int, name: str) -> Array:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array
