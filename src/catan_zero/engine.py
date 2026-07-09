from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from catan_zero.schemas import Action, Event, Observation, SeedBundle


class CatanEngine(ABC):
    """Boundary every simulator implementation must satisfy."""

    @abstractmethod
    def reset(self, seed_bundle: SeedBundle) -> Any:
        raise NotImplementedError

    @abstractmethod
    def legal_actions(self, player_id: int) -> tuple[Action, ...]:
        raise NotImplementedError

    @abstractmethod
    def observe(self, player_id: int) -> Observation:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Action) -> Any:
        raise NotImplementedError

    @abstractmethod
    def clone(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def restore(self, snapshot: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def event_log(self) -> tuple[Event, ...]:
        raise NotImplementedError

