from __future__ import annotations

from types import SimpleNamespace
import json

import pytest

from catan_zero.search.gumbel_chance_mcts import (
    GumbelChanceMCTS,
    GumbelChanceMCTSConfig,
)
from catan_zero.search.native_gumbel_mcts import (
    NativeGumbelChanceMCTS,
    create_gumbel_search,
    native_hot_loop_available,
)
from catan_zero.search.rust_mcts import HeuristicRustEvaluator


class _PublicCountingEvaluator:
    def __init__(self) -> None:
        self.config = SimpleNamespace(public_observation=True, cache_size=0)
        self.inner = HeuristicRustEvaluator(score_actions=False)
        self.root_symmetry_calls = 0

    def evaluate(self, *args, **kwargs):
        return self.inner.evaluate(*args, **kwargs)

    def evaluate_many(self, requests, *, root_color, colors):
        return self.inner.evaluate_many(requests, root_color=root_color, colors=colors)

    def evaluate_symmetry_averaged(self, *args, **kwargs):
        self.root_symmetry_calls += 1
        return self.inner.evaluate(*args, **kwargs)


def test_native_hot_loop_is_explicit_and_fallback_preserves_reference() -> None:
    pytest.importorskip("catanatron_rs")
    try:
        from catan_zero.search.rust_mcts import _require_rust_module

        _require_rust_module()
    except RuntimeError as error:
        pytest.skip(f"installed wheel lacks reference MCTS bindings: {error}")
    config = GumbelChanceMCTSConfig()
    evaluator = _PublicCountingEvaluator()
    search = create_gumbel_search(config, evaluator)
    assert type(search) is GumbelChanceMCTS

    if native_hot_loop_available():
        native = create_gumbel_search(config, evaluator, native_hot_loop=True)
        assert isinstance(native, NativeGumbelChanceMCTS)
        assert native.using_native_hot_loop is True


@pytest.mark.skipif(
    not native_hot_loop_available(), reason="native wheel lacks gumbel_search"
)
def test_native_information_set_p4_exact_budget_d6_and_global_ids() -> None:
    rust = pytest.importorskip("catanatron_rs")
    evaluator = _PublicCountingEvaluator()
    config = GumbelChanceMCTSConfig(
        n_full=128,
        n_fast=16,
        p_full=1.0,
        exact_budget_sh=True,
        max_depth=6,
        seed=19,
        information_set_search=True,
        determinization_particles=4,
        determinization_min_simulations=32,
        symmetry_averaged_eval=True,
        symmetry_averaged_eval_threshold=20,
    )
    game = rust.Game.simple(["RED", "BLUE"], seed=31)
    legal = set(game.playable_action_indices(["RED", "BLUE"], None))
    result = NativeGumbelChanceMCTS(config, evaluator).search(game, force_full=True)

    assert result.simulations_used == 128
    assert evaluator.root_symmetry_calls == 4
    assert result.selected_action in legal
    assert set(result.improved_policy) == legal
    assert set(result.visit_counts) == legal
    assert set(result.q_values).issubset(legal)
    assert set(result.priors) == legal
    assert sum(result.improved_policy.values()) == pytest.approx(1.0)
    assert sum(result.priors.values()) == pytest.approx(1.0)


@pytest.mark.skipif(
    not native_hot_loop_available(), reason="native wheel lacks gumbel_search"
)
def test_native_d6_is_root_only_not_same_turn_interior() -> None:
    rust = pytest.importorskip("catanatron_rs")
    evaluator = _PublicCountingEvaluator()
    config = GumbelChanceMCTSConfig(
        n_full=32,
        n_fast=32,
        p_full=1.0,
        exact_budget_sh=True,
        max_depth=8,
        seed=7,
        symmetry_averaged_eval=True,
        symmetry_averaged_eval_threshold=20,
    )
    NativeGumbelChanceMCTS(config, evaluator).search(
        rust.Game.simple(["RED", "BLUE"], seed=9), force_full=True
    )
    assert evaluator.root_symmetry_calls == 1


@pytest.mark.skipif(
    not native_hot_loop_available(), reason="native wheel lacks gumbel_search"
)
def test_native_forced_roll_expectation_matches_reference() -> None:
    rust = pytest.importorskip("catanatron_rs")
    game = rust.Game.simple(["RED", "BLUE"], seed=41)
    for _ in range(100):
        actions = json.loads(game.playable_actions_json())
        if len(actions) == 1 and actions[0][1] == "ROLL":
            break
        game.play_tick()
    else:
        pytest.fail("did not reach a forced ROLL root")

    config = GumbelChanceMCTSConfig(seed=3, max_depth=4)
    reference = GumbelChanceMCTS(config, _PublicCountingEvaluator()).search(game.copy())
    native = NativeGumbelChanceMCTS(config, _PublicCountingEvaluator()).search(
        game.copy()
    )
    legal = game.playable_action_indices(["RED", "BLUE"], None)
    assert len(legal) == 1
    assert native.selected_action == reference.selected_action == legal[0]
    assert native.improved_policy == reference.improved_policy == {legal[0]: 1.0}
    assert native.root_value == pytest.approx(reference.root_value, abs=1e-12)
    assert native.afterstate_values == pytest.approx(
        reference.afterstate_values, abs=1e-12
    )
