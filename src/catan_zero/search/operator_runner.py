"""Measured, fail-closed adapters for R&D search-operator comparisons.

The leaderboard consumes per-game totals, while search implementations expose
different result types and budget counters.  This module is the single mapping
boundary.  It wraps the evaluator before constructing an operator, times each
decision, and reports both algorithmic visits and actual neural work.

No adapter silently claims public-information safety.  At present only Gumbel
search with ``information_set_search=True`` has the engine-level
determinization and actor-turn boundary required for that label.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Literal

from catan_zero.search.accounting import SearchAccountingEvaluator, SearchWork
from catan_zero.search.gumbel_chance_mcts import (
    GumbelChanceMCTS,
    GumbelChanceMCTSConfig,
    SearchResult,
    _wide_budget_applies,
)
from catan_zero.search.regularized_mcts import (
    RegularizedMCTSConfig,
    RegularizedPolicyMCTS,
)
from catan_zero.search.rust_mcts import (
    RustEvaluator,
    RustMCTS,
    RustMCTSConfig,
    RustMCTSResult,
    _legal_action_indices,
)

__all__ = [
    "InformationRegime",
    "SearchCounters",
    "MeasuredDecision",
    "GameCounterAccumulator",
    "MeasuredSearchOperator",
]

InformationRegime = Literal[
    "authoritative_hidden_state",
    "public_conservation_pimc",
    "public_observation_policy",
]
OperatorKind = Literal["gumbel", "puct", "regularized_mcts", "raw_policy"]


@dataclass(frozen=True, slots=True)
class SearchCounters:
    nominal_visits: int
    scheduled_visits: int
    logical_leaves: int
    orientation_rows: int
    evaluator_calls: int
    wall_time_sec: float

    @classmethod
    def from_work(
        cls,
        *,
        nominal_visits: int,
        scheduled_visits: int,
        work: SearchWork,
        wall_time_sec: float,
    ) -> "SearchCounters":
        return cls(
            nominal_visits=int(nominal_visits),
            scheduled_visits=int(scheduled_visits),
            # Explicit mapping from accounting.py names to leaderboard schema.
            logical_leaves=int(work.logical_leaf_evaluations),
            orientation_rows=int(work.orientation_evaluation_rows),
            evaluator_calls=int(work.evaluator_calls),
            wall_time_sec=float(wall_time_sec),
        )

    def as_dict(self) -> dict[str, int | float]:
        return {
            "nominal_visits": self.nominal_visits,
            "scheduled_visits": self.scheduled_visits,
            "logical_leaves": self.logical_leaves,
            "orientation_rows": self.orientation_rows,
            "evaluator_calls": self.evaluator_calls,
            "wall_time_sec": self.wall_time_sec,
        }


@dataclass(frozen=True, slots=True)
class MeasuredDecision:
    selected_action: int
    policy: dict[int, float]
    q_values: dict[int, float]
    root_value: float
    counters: SearchCounters
    information_regime: InformationRegime


@dataclass(slots=True)
class GameCounterAccumulator:
    """Sum measured decisions into the exact per-game leaderboard payload."""

    nominal_visits: int = 0
    scheduled_visits: int = 0
    logical_leaves: int = 0
    orientation_rows: int = 0
    evaluator_calls: int = 0
    wall_time_sec: float = 0.0
    information_regime: InformationRegime | None = None

    def add(self, decision: MeasuredDecision) -> None:
        if self.information_regime is None:
            self.information_regime = decision.information_regime
        elif self.information_regime != decision.information_regime:
            raise ValueError(
                "one game cannot mix search information regimes: "
                f"{self.information_regime} vs {decision.information_regime}"
            )
        counters = decision.counters
        self.nominal_visits += counters.nominal_visits
        self.scheduled_visits += counters.scheduled_visits
        self.logical_leaves += counters.logical_leaves
        self.orientation_rows += counters.orientation_rows
        self.evaluator_calls += counters.evaluator_calls
        self.wall_time_sec += counters.wall_time_sec

    def as_dict(self) -> dict[str, int | float]:
        return SearchCounters(
            nominal_visits=self.nominal_visits,
            scheduled_visits=self.scheduled_visits,
            logical_leaves=self.logical_leaves,
            orientation_rows=self.orientation_rows,
            evaluator_calls=self.evaluator_calls,
            wall_time_sec=self.wall_time_sec,
        ).as_dict()


class MeasuredSearchOperator:
    """Construct and run one search arm with exact evaluator-work accounting."""

    def __init__(
        self,
        kind: OperatorKind,
        evaluator: RustEvaluator,
        *,
        config: (
            GumbelChanceMCTSConfig
            | RustMCTSConfig
            | RegularizedMCTSConfig
            | None
        ) = None,
    ) -> None:
        self.kind = kind
        self.accounting_evaluator = SearchAccountingEvaluator(evaluator)
        evaluator_config = getattr(evaluator, "config", None)
        if evaluator_config is not None and int(
            getattr(evaluator_config, "cache_size", 0)
        ) != 0:
            raise ValueError(
                "measured search requires evaluator cache_size=0 so orientation_rows "
                "equals actual model rows"
            )
        evaluator_is_public = evaluator_config is not None and bool(
            getattr(evaluator_config, "public_observation", False)
        )
        self.search: GumbelChanceMCTS | RustMCTS | RegularizedPolicyMCTS | None
        if kind == "gumbel":
            if config is not None and not isinstance(config, GumbelChanceMCTSConfig):
                raise TypeError("gumbel requires GumbelChanceMCTSConfig")
            self.search = GumbelChanceMCTS(config, self.accounting_evaluator)
            self.information_regime: InformationRegime = (
                "public_conservation_pimc"
                if bool(self.search.config.information_set_search)
                else "authoritative_hidden_state"
            )
        elif kind == "puct":
            if config is not None and not isinstance(config, RustMCTSConfig):
                raise TypeError("puct requires RustMCTSConfig")
            self.search = RustMCTS(config, self.accounting_evaluator)
            self.information_regime = (
                "public_conservation_pimc"
                if bool(self.search.config.information_set_search)
                else "authoritative_hidden_state"
            )
        elif kind == "regularized_mcts":
            if config is not None and not isinstance(config, RegularizedMCTSConfig):
                raise TypeError("regularized_mcts requires RegularizedMCTSConfig")
            self.search = RegularizedPolicyMCTS(config, self.accounting_evaluator)
            self.information_regime = (
                "public_conservation_pimc"
                if bool(self.search.config.information_set_search)
                else "authoritative_hidden_state"
            )
        elif kind == "raw_policy":
            if config is not None:
                raise TypeError("raw_policy does not accept a search config")
            self.search = None
            self.information_regime = (
                "public_observation_policy"
                if evaluator_is_public
                else "authoritative_hidden_state"
            )
        else:  # pragma: no cover - Literal guard for dynamic callers.
            raise ValueError(f"unknown search operator {kind!r}")

    def run(
        self,
        game: Any,
        *,
        require_public_information: bool = False,
    ) -> MeasuredDecision:
        if (
            require_public_information
            and self.information_regime == "authoritative_hidden_state"
        ):
            raise RuntimeError(
                f"{self.kind} has no complete public-information adapter; "
                "use Gumbel information_set_search or run an explicitly "
                "authoritative-state diagnostic"
            )
        before = self.accounting_evaluator.snapshot()
        started = time.perf_counter()
        if self.kind == "raw_policy":
            decision = self._run_raw(game)
        elif self.kind == "gumbel":
            assert isinstance(self.search, GumbelChanceMCTS)
            decision = self._run_gumbel(self.search.search(game), game)
        else:
            assert isinstance(self.search, RustMCTS)
            decision = self._run_tree(self.search.search(game))
        elapsed = time.perf_counter() - started
        work = self.accounting_evaluator.snapshot() - before
        counters = SearchCounters.from_work(
            nominal_visits=decision[4],
            scheduled_visits=decision[5],
            work=work,
            wall_time_sec=elapsed,
        )
        return MeasuredDecision(
            selected_action=decision[0],
            policy=decision[1],
            q_values=decision[2],
            root_value=decision[3],
            counters=counters,
            information_regime=self.information_regime,
        )

    def _run_raw(
        self, game: Any
    ) -> tuple[int, dict[int, float], dict[int, float], float, int, int]:
        colors = ("RED", "BLUE")
        legal = _legal_action_indices(game, colors=colors, map_kind=None)
        if not legal:
            raise RuntimeError("raw policy root has no legal actions")
        if len(legal) == 1:
            action = int(legal[0])
            return action, {action: 1.0}, {}, 0.0, 0, 0
        root_color = str(game.current_color())
        priors, value = self.accounting_evaluator.evaluate(
            game, legal, root_color=root_color, colors=colors
        )
        action = max(legal, key=lambda candidate: (priors.get(candidate, 0.0), -candidate))
        return int(action), dict(priors), {}, float(value), 0, 0

    def _run_tree(
        self, result: RustMCTSResult
    ) -> tuple[int, dict[int, float], dict[int, float], float, int, int]:
        scheduled = int(sum(result.visits.values()))
        # Forced roots report their one forced action as one visit without
        # scheduling a tree simulation.
        non_forced = len(result.visits) > 1
        assert isinstance(self.search, RustMCTS)
        nominal = max(int(self.search.config.simulations), 1) if non_forced else 0
        return (
            int(result.action),
            dict(result.policy),
            dict(result.q_values),
            float(result.root_value),
            nominal,
            scheduled if non_forced else 0,
        )

    def _run_gumbel(
        self, result: SearchResult, _game: Any
    ) -> tuple[int, dict[int, float], dict[int, float], float, int, int]:
        assert isinstance(self.search, GumbelChanceMCTS)
        scheduled = int(result.simulations_used)
        nominal = 0
        if scheduled > 0:
            config = self.search.config
            if result.used_full_search:
                legal_count = len(result.visit_counts)
                nominal = int(
                    config.n_full_wide
                    if _wide_budget_applies(legal_count, config)
                    else config.n_full
                )
            else:
                nominal = int(config.n_fast)
        return (
            int(result.selected_action),
            dict(result.improved_policy),
            dict(result.q_values),
            float(result.root_value),
            nominal,
            scheduled,
        )
