from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading

from catan_zero.distributed.schemas import ActorBatch
from catan_zero.distributed.schemas import LearnerBatch
from catan_zero.distributed.schemas import QueueStats
from catan_zero.distributed.schemas import records_to_learner_batch


@dataclass(slots=True)
class _Counters:
    total_enqueued_batches: int = 0
    total_enqueued_decisions: int = 0
    total_dequeued_batches: int = 0
    total_dequeued_decisions: int = 0
    dropped_batches: int = 0
    dropped_decisions: int = 0


class InMemoryBatchQueue:
    """Thread-safe actor-to-learner queue for smoke tests and local services.

    Cluster deployments can replace this with Redis, Kafka, NATS, Ray queues, or
    object storage. The public API stays intentionally small.
    """

    def __init__(
        self,
        *,
        max_decisions: int = 1_000_000,
        drop_oldest: bool = True,
    ) -> None:
        if max_decisions <= 0:
            raise ValueError("max_decisions must be positive")
        self.max_decisions = int(max_decisions)
        self.drop_oldest = bool(drop_oldest)
        self._batches: deque[ActorBatch] = deque()
        self._queued_decisions = 0
        self._counters = _Counters()
        self._lock = threading.Lock()

    def put(self, batch: ActorBatch) -> None:
        if batch.decisions <= 0:
            return
        with self._lock:
            if batch.decisions > self.max_decisions:
                raise ValueError("single ActorBatch is larger than queue capacity")
            if not self.drop_oldest and self._queued_decisions + batch.decisions > self.max_decisions:
                raise OverflowError("batch queue capacity exceeded")
            while self._queued_decisions + batch.decisions > self.max_decisions:
                dropped = self._batches.popleft()
                self._queued_decisions -= dropped.decisions
                self._counters.dropped_batches += 1
                self._counters.dropped_decisions += dropped.decisions
            self._batches.append(batch)
            self._queued_decisions += batch.decisions
            self._counters.total_enqueued_batches += 1
            self._counters.total_enqueued_decisions += batch.decisions

    def get_learner_batch(
        self,
        *,
        max_decisions: int,
        min_decisions: int = 1,
    ) -> LearnerBatch | None:
        if max_decisions <= 0:
            raise ValueError("max_decisions must be positive")
        if min_decisions <= 0:
            raise ValueError("min_decisions must be positive")
        if min_decisions > max_decisions:
            raise ValueError("min_decisions cannot exceed max_decisions")
        with self._lock:
            if self._queued_decisions < min_decisions:
                return None
            records = []
            dequeued_batches = 0
            while self._batches and len(records) < max_decisions:
                batch = self._batches[0]
                remaining = max_decisions - len(records)
                if batch.decisions <= remaining:
                    self._batches.popleft()
                    records.extend(batch.records)
                    self._queued_decisions -= batch.decisions
                    dequeued_batches += 1
                else:
                    # Keep batch atomic. This avoids rewrapping partial actor
                    # batches and preserves game-level metadata for now.
                    break
            if len(records) < min_decisions:
                return None
            self._counters.total_dequeued_batches += dequeued_batches
            self._counters.total_dequeued_decisions += len(records)
        return records_to_learner_batch(records)

    def stats(self) -> QueueStats:
        with self._lock:
            return QueueStats(
                queued_batches=len(self._batches),
                queued_decisions=self._queued_decisions,
                total_enqueued_batches=self._counters.total_enqueued_batches,
                total_enqueued_decisions=self._counters.total_enqueued_decisions,
                total_dequeued_batches=self._counters.total_dequeued_batches,
                total_dequeued_decisions=self._counters.total_dequeued_decisions,
                dropped_batches=self._counters.dropped_batches,
                dropped_decisions=self._counters.dropped_decisions,
            )
