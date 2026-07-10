"""Exact evaluator-work accounting for compute-matched search experiments.

``simulations_used`` is an algorithmic tree counter.  It is not a neural-work
counter because one simulation can expand many chance children and a D6 root
evaluation executes twelve oriented rows.  ``SearchAccountingEvaluator`` wraps
any Rust evaluator without changing its outputs and records both quantities.

The wrapper is intentionally independent of Gumbel MCTS.  PUCT and future
search operators can use the same counter, which makes equal-work comparisons
meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any

__all__ = [
    "SearchWork",
    "SearchAccountingEvaluator",
    "SearchAccountingScope",
]


@dataclass(frozen=True, slots=True)
class SearchWork:
    """Immutable evaluator work measured over one interval."""

    evaluator_calls: int = 0
    logical_leaf_evaluations: int = 0
    orientation_evaluation_rows: int = 0

    def __sub__(self, other: "SearchWork") -> "SearchWork":
        result = SearchWork(
            evaluator_calls=self.evaluator_calls - other.evaluator_calls,
            logical_leaf_evaluations=(
                self.logical_leaf_evaluations - other.logical_leaf_evaluations
            ),
            orientation_evaluation_rows=(
                self.orientation_evaluation_rows - other.orientation_evaluation_rows
            ),
        )
        if min(
            result.evaluator_calls,
            result.logical_leaf_evaluations,
            result.orientation_evaluation_rows,
        ) < 0:
            raise ValueError("search accounting snapshots are not monotonic")
        return result

    def as_dict(self) -> dict[str, int]:
        return {
            "evaluator_calls": self.evaluator_calls,
            "logical_leaf_evaluations": self.logical_leaf_evaluations,
            "orientation_evaluation_rows": self.orientation_evaluation_rows,
        }


class SearchAccountingEvaluator:
    """Transparent, thread-safe evaluator counter.

    Attribute lookup delegates to the wrapped evaluator.  The three known
    evaluation entry points are intercepted dynamically so ``hasattr`` keeps
    the wrapped evaluator's capability semantics: a wrapper around an
    evaluator without ``evaluate_many`` still reports no such method.
    """

    _COUNTED_METHODS = frozenset(
        {"evaluate", "evaluate_many", "evaluate_symmetry_averaged"}
    )

    def __init__(self, evaluator: Any, *, symmetry_orientations: int = 12) -> None:
        if int(symmetry_orientations) < 1:
            raise ValueError("symmetry_orientations must be positive")
        self.evaluator = evaluator
        self.symmetry_orientations = int(symmetry_orientations)
        self._lock = threading.Lock()
        self._evaluator_calls = 0
        self._logical_leaf_evaluations = 0
        self._orientation_evaluation_rows = 0

    def snapshot(self) -> SearchWork:
        with self._lock:
            return SearchWork(
                evaluator_calls=self._evaluator_calls,
                logical_leaf_evaluations=self._logical_leaf_evaluations,
                orientation_evaluation_rows=self._orientation_evaluation_rows,
            )

    def scope(self) -> "SearchAccountingScope":
        return SearchAccountingScope(self)

    def _record(
        self,
        method_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        if method_name == "evaluate_many":
            requests = args[0] if args else kwargs.get("requests", ())
            # Evaluators resolve terminal/no-legal requests without a model
            # forward.  Count only requests that can actually become neural
            # rows; the paired R&D runner disables evaluator caching so this is
            # then exact rather than merely a request count.
            logical = sum(
                1
                for request in requests
                if len(request[1] if len(request) > 1 else ()) > 0
            )
            orientation_rows = logical
        elif method_name == "evaluate_symmetry_averaged":
            legal_actions = args[1] if len(args) > 1 else kwargs.get("legal_actions", ())
            logical = int(len(legal_actions) > 0)
            orientation_rows = logical * self.symmetry_orientations
        else:
            legal_actions = args[1] if len(args) > 1 else kwargs.get("legal_actions", ())
            logical = int(len(legal_actions) > 0)
            orientation_rows = logical
        with self._lock:
            self._evaluator_calls += 1
            self._logical_leaf_evaluations += logical
            self._orientation_evaluation_rows += orientation_rows

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self.evaluator, name)
        if name not in self._COUNTED_METHODS or not callable(attribute):
            return attribute

        def counted(*args: Any, **kwargs: Any) -> Any:
            self._record(name, args, kwargs)
            return attribute(*args, **kwargs)

        return counted


class SearchAccountingScope:
    """Context manager that exposes work performed inside its body."""

    def __init__(self, evaluator: SearchAccountingEvaluator) -> None:
        self.evaluator = evaluator
        self.before: SearchWork | None = None
        self.work: SearchWork | None = None

    def __enter__(self) -> "SearchAccountingScope":
        self.before = self.evaluator.snapshot()
        self.work = None
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if self.before is None:  # pragma: no cover - context protocol guard.
            raise RuntimeError("search accounting scope was not entered")
        self.work = self.evaluator.snapshot() - self.before
        return False

    def require_work(self) -> SearchWork:
        if self.work is None:
            raise RuntimeError("search accounting scope has not completed")
        return self.work
