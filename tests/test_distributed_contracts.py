from __future__ import annotations

import numpy as np
import pytest

from catan_zero.distributed import ActorBatch
from catan_zero.distributed import DecisionRecord
from catan_zero.distributed import InMemoryBatchQueue
from catan_zero.distributed import PolicyRegistry
from catan_zero.distributed import PolicyVersion


def _record(index: int, *, policy_id: str = "p0") -> DecisionRecord:
    return DecisionRecord(
        policy_id=policy_id,
        game_id="g0",
        seed=123,
        decision_index=index,
        player="BLUE",
        opponent_policy_ids=("heuristic", "jsettlers_lite", "value_rollout"),
        observation=np.full(4, index, dtype=np.float32),
        valid_actions=(1, 3, 5),
        action=3,
        old_log_prob=-1.0,
        old_value=0.25,
        return_=1.0,
        advantage=0.75,
        action_context_features=np.zeros((6, 2), dtype=np.float32),
        old_q_value=0.1,
    )


def test_in_memory_queue_builds_learner_batches() -> None:
    queue = InMemoryBatchQueue(max_decisions=16)
    queue.put(ActorBatch(actor_id="a0", policy_id="p0", records=tuple(_record(i) for i in range(4))))
    queue.put(ActorBatch(actor_id="a1", policy_id="p0", records=tuple(_record(i + 4) for i in range(4))))

    batch = queue.get_learner_batch(max_decisions=8, min_decisions=8)

    assert batch is not None
    assert batch.size == 8
    assert batch.observations.shape == (8, 4)
    assert batch.action_context_features is not None
    assert batch.action_context_features.shape == (8, 6, 2)
    assert batch.old_q_values is not None
    assert batch.old_q_values.shape == (8,)
    assert queue.stats().queued_decisions == 0


def test_queue_drops_oldest_when_capacity_is_exceeded() -> None:
    queue = InMemoryBatchQueue(max_decisions=4, drop_oldest=True)
    queue.put(ActorBatch(actor_id="a0", policy_id="p0", records=tuple(_record(i) for i in range(3))))
    queue.put(ActorBatch(actor_id="a1", policy_id="p0", records=tuple(_record(i + 3) for i in range(3))))

    stats = queue.stats()

    assert stats.queued_batches == 1
    assert stats.queued_decisions == 3
    assert stats.dropped_batches == 1
    assert stats.dropped_decisions == 3


def test_decision_record_rejects_illegal_action() -> None:
    with pytest.raises(ValueError, match="action must be present"):
        DecisionRecord(
            policy_id="p0",
            game_id="g0",
            seed=123,
            decision_index=0,
            player="BLUE",
            opponent_policy_ids=(),
            observation=np.zeros(4, dtype=np.float32),
            valid_actions=(1, 2),
            action=3,
            old_log_prob=0.0,
            old_value=0.0,
            return_=0.0,
            advantage=0.0,
        )


def test_policy_registry_tracks_latest_and_champion() -> None:
    registry = PolicyRegistry()
    registry.publish(
        PolicyVersion(
            policy_id="p0",
            checkpoint_path="runs/self_play/p0.pt",
            architecture="candidate",
        ),
        champion=True,
    )
    registry.publish(
        PolicyVersion(
            policy_id="p1",
            checkpoint_path="runs/self_play/p1.pt",
            architecture="graph_history_candidate",
            parent_policy_id="p0",
        )
    )

    assert registry.champion().policy_id == "p0"
    assert registry.latest().policy_id == "p1"

    registry.promote_champion("p1")

    assert registry.champion().policy_id == "p1"
    assert registry.latest().policy_id == "p1"
