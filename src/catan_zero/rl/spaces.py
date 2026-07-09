from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:  # pragma: no cover - fallback exists for dependency-light imports.
    import numpy as np
except ImportError:  # pragma: no cover - exercised in clean envs without numpy.
    np = None

try:  # pragma: no cover - exercised only when gymnasium is installed.
    from gymnasium import spaces as gym_spaces
except ImportError:  # pragma: no cover - fallback is covered in this repo.
    gym_spaces = None


@dataclass(frozen=True, slots=True)
class DiscreteSpace:
    """Small fallback compatible with the Gymnasium Discrete attributes we need."""

    n: int

    def sample(self) -> int:
        if np is None:
            raise RuntimeError("numpy is required to sample from DiscreteSpace")
        return int(np.random.randint(self.n))

    def contains(self, value: Any) -> bool:
        try:
            action = int(value)
        except (TypeError, ValueError):
            return False
        return 0 <= action < self.n


@dataclass(frozen=True, slots=True)
class BoxSpace:
    """Small fallback compatible with the Gymnasium Box attributes we need."""

    low: float
    high: float
    shape: tuple[int, ...]
    dtype: Any

    def sample(self) -> Any:
        if np is None:
            raise RuntimeError("numpy is required to sample from BoxSpace")
        return np.random.uniform(self.low, self.high, self.shape).astype(self.dtype)

    def contains(self, value: Any) -> bool:
        if np is None:
            return False
        array = np.asarray(value, dtype=self.dtype)
        return array.shape == self.shape and bool(
            np.all(array >= self.low) and np.all(array <= self.high)
        )


def make_discrete(n: int) -> Any:
    if gym_spaces is not None:
        return gym_spaces.Discrete(n)
    return DiscreteSpace(n)


def make_box(
    *,
    low: float,
    high: float,
    shape: tuple[int, ...],
    dtype: Any,
) -> Any:
    if gym_spaces is not None:
        return gym_spaces.Box(low=low, high=high, shape=shape, dtype=dtype)
    return BoxSpace(low=low, high=high, shape=shape, dtype=dtype)
