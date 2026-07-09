from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import threading

from catan_zero.distributed.schemas import PolicyVersion


class PolicyRegistry:
    """Small thread-safe registry for learner-published policy versions."""

    def __init__(self) -> None:
        self._versions: dict[str, PolicyVersion] = {}
        self._latest_id: str | None = None
        self._champion_id: str | None = None
        self._lock = threading.Lock()

    def publish(
        self,
        version: PolicyVersion,
        *,
        latest: bool = True,
        champion: bool = False,
    ) -> PolicyVersion:
        if not version.policy_id:
            raise ValueError("policy_id is required")
        if not version.checkpoint_path:
            raise ValueError("checkpoint_path is required")
        normalized = replace(
            version,
            checkpoint_path=str(Path(version.checkpoint_path)),
        )
        with self._lock:
            self._versions[normalized.policy_id] = normalized
            if latest:
                self._latest_id = normalized.policy_id
            if champion:
                self._champion_id = normalized.policy_id
        return normalized

    def get(self, policy_id: str) -> PolicyVersion:
        with self._lock:
            try:
                return self._versions[policy_id]
            except KeyError as exc:
                raise KeyError(f"unknown policy_id {policy_id!r}") from exc

    def latest(self) -> PolicyVersion | None:
        with self._lock:
            if self._latest_id is None:
                return None
            return self._versions[self._latest_id]

    def champion(self) -> PolicyVersion | None:
        with self._lock:
            if self._champion_id is None:
                return None
            return self._versions[self._champion_id]

    def promote_champion(self, policy_id: str) -> PolicyVersion:
        with self._lock:
            if policy_id not in self._versions:
                raise KeyError(f"unknown policy_id {policy_id!r}")
            self._champion_id = policy_id
            self._latest_id = policy_id
            return self._versions[policy_id]

    def list_versions(self) -> tuple[PolicyVersion, ...]:
        with self._lock:
            return tuple(
                sorted(
                    self._versions.values(),
                    key=lambda version: version.created_at,
                )
            )
