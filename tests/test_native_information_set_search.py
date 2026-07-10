from __future__ import annotations

from types import SimpleNamespace

import pytest

from catan_zero.search.gumbel_chance_mcts import (
    GumbelChanceMCTS,
    GumbelChanceMCTSConfig,
)
from catan_zero.search.rust_mcts import (
    HeuristicRustEvaluator,
    RustMCTS,
    RustMCTSConfig,
)


class _PublicHeuristic:
    """Deterministic plumbing evaluator with an explicit public contract."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(public_observation=True, cache_size=0)
        self.inner = HeuristicRustEvaluator(score_actions=False)

    def evaluate(self, *args, **kwargs):
        return self.inner.evaluate(*args, **kwargs)

    def evaluate_many(self, *args, **kwargs):
        return self.inner.evaluate_many(*args, **kwargs)


def test_native_public_conservation_search_smoke() -> None:
    rust = pytest.importorskip("catanatron_rs")
    if not hasattr(rust.Game, "determinize_for_player"):
        pytest.skip("installed native wheel predates public determinization")
    evaluator = _PublicHeuristic()
    searches = (
        GumbelChanceMCTS(
            GumbelChanceMCTSConfig(
                n_full=8,
                n_fast=8,
                p_full=1.0,
                max_depth=4,
                seed=7,
                information_set_search=True,
                determinization_particles=2,
                determinization_min_simulations=1,
            ),
            evaluator,
        ),
        RustMCTS(
            RustMCTSConfig(
                simulations=8,
                max_depth=4,
                seed=7,
                information_set_search=True,
                determinization_particles=2,
                determinization_min_simulations=1,
            ),
            evaluator,
        ),
    )
    gumbel = searches[0].search(rust.Game.simple(["RED", "BLUE"], seed=11))
    puct = searches[1].search(rust.Game.simple(["RED", "BLUE"], seed=11))
    assert gumbel.simulations_used == 8
    assert sum(puct.visits.values()) == 8
    assert len(gumbel.improved_policy) == len(puct.policy) == 54
