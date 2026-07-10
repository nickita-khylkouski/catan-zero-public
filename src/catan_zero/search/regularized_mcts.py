"""Experimental reverse-KL regularized policy MCTS.

The tree mechanics come from :mod:`catan_zero.search.rust_mcts`.  This module
changes only action selection: each node forms the reverse-KL regularized
policy

    improved(a) proportional to prior(a) * exp(signed_q(a) / temperature)

and visits the action with the largest deficit between that policy and the
empirical visit distribution.  Keeping this as a separate opt-in operator
prevents an R&D comparison from mutating the production Gumbel or PUCT paths.

This implementation is an experiment arm, not a theoretical guarantee for
stochastic imperfect-information Catan.  It must use the same public-belief
chance adapter and compute accounting as every other arm.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Mapping

from catan_zero.search.rust_mcts import (
    HeuristicRustEvaluator,
    RustEvaluator,
    RustMCTS,
    RustMCTSConfig,
    _ActionStats,
    _Node,
)

__all__ = [
    "RegularizedMCTSConfig",
    "RegularizedPolicyMCTS",
    "reverse_kl_improved_policy",
]


@dataclass(frozen=True, slots=True)
class RegularizedMCTSConfig:
    colors: tuple[str, ...] = ("RED", "BLUE")
    map_kind: str | None = None
    simulations: int = 64
    max_depth: int = 80
    seed: int = 0
    temperature: float = 1.0
    regularization_temperature: float = 0.5
    prior_floor: float = 1.0e-12
    information_set_search: bool = False
    determinization_particles: int = 1
    determinization_min_simulations: int = 32


def reverse_kl_improved_policy(
    priors: Mapping[int, float],
    q_values: Mapping[int, float],
    *,
    root_to_act: bool,
    temperature: float,
    prior_floor: float = 1.0e-12,
) -> dict[int, float]:
    """Return ``argmax_pi <pi,Q> - tau KL(pi||prior)`` in closed form."""

    if not priors:
        return {}
    tau = float(temperature)
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError("regularization temperature must be finite and positive")
    floor = float(prior_floor)
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("prior_floor must be finite and positive")

    signed_scores: dict[int, float] = {}
    for action, prior in priors.items():
        q_value = float(q_values.get(action, 0.0))
        signed_q = q_value if root_to_act else -q_value
        signed_scores[int(action)] = math.log(max(float(prior), floor)) + signed_q / tau
    maximum = max(signed_scores.values())
    weights = {
        action: math.exp(min(score - maximum, 0.0))
        for action, score in signed_scores.items()
    }
    total = sum(weights.values())
    if not math.isfinite(total) or total <= 0.0:
        uniform = 1.0 / len(weights)
        return {action: uniform for action in weights}
    return {action: weight / total for action, weight in weights.items()}


class RegularizedPolicyMCTS(RustMCTS):
    """PUCT-compatible tree with reverse-KL policy-deficit selection."""

    def __init__(
        self,
        config: RegularizedMCTSConfig | None = None,
        evaluator: RustEvaluator | None = None,
    ) -> None:
        self.regularized_config = config or RegularizedMCTSConfig()
        super().__init__(
            RustMCTSConfig(
                colors=self.regularized_config.colors,
                map_kind=self.regularized_config.map_kind,
                simulations=self.regularized_config.simulations,
                max_depth=self.regularized_config.max_depth,
                seed=self.regularized_config.seed,
                temperature=self.regularized_config.temperature,
                information_set_search=self.regularized_config.information_set_search,
                determinization_particles=self.regularized_config.determinization_particles,
                determinization_min_simulations=(
                    self.regularized_config.determinization_min_simulations
                ),
            ),
            evaluator=evaluator or HeuristicRustEvaluator(),
        )

    def _spawn_information_set_particle(
        self, *, simulations: int, seed: int
    ) -> "RegularizedPolicyMCTS":
        particle = RegularizedPolicyMCTS(
            replace(
                self.regularized_config,
                simulations=int(simulations),
                seed=int(seed),
                information_set_search=False,
            ),
            self.evaluator,
        )
        particle._root_actor_turn_only = True
        return particle

    def _select_action(self, node: _Node) -> tuple[int, _ActionStats]:
        root_to_act = str(node.game.current_color()) == str(node.root_color)
        improved = reverse_kl_improved_policy(
            {action: stats.prior for action, stats in node.actions.items()},
            {action: stats.q for action, stats in node.actions.items()},
            root_to_act=root_to_act,
            temperature=self.regularized_config.regularization_temperature,
            prior_floor=self.regularized_config.prior_floor,
        )
        total_visits = sum(stats.visits for stats in node.actions.values())
        denominator = 1.0 + float(total_visits)
        action = max(
            node.actions,
            key=lambda candidate: (
                improved[candidate]
                - node.actions[candidate].visits / denominator,
                improved[candidate],
                node.actions[candidate].prior,
                -int(candidate),
            ),
        )
        return int(action), node.actions[action]
