from __future__ import annotations

from types import SimpleNamespace
import json
import os
import subprocess
import sys
import textwrap

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


class _ScalarOnlyPublicEvaluator:
    def __init__(self) -> None:
        self.config = SimpleNamespace(public_observation=True, cache_size=0)
        self.inner = HeuristicRustEvaluator(score_actions=False)

    def evaluate(self, *args, **kwargs):
        return self.inner.evaluate(*args, **kwargs)


class _WrongBatchPublicEvaluator(_ScalarOnlyPublicEvaluator):
    def __init__(self, delta: int) -> None:
        super().__init__()
        self.delta = delta

    def evaluate_many(self, requests, *, root_color, colors):
        expected = self.inner.evaluate_many(
            requests, root_color=root_color, colors=colors
        )
        if self.delta < 0:
            return expected[: self.delta]
        return expected + [({}, 0.0)] * self.delta


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
@pytest.mark.parametrize(
    ("override", "needle"),
    [
        ({"correct_rust_chance_spectra": False}, "correct_rust_chance_spectra"),
        ({"belief_chance_spectra": True}, "belief_chance_spectra"),
        ({"root_wave_batching": True}, "root_wave_batching"),
        ({"use_batch_api": False}, "use_batch_api"),
        ({"uncertainty_backup_weighting": True}, "uncertainty_backup_weighting"),
    ],
)
def test_native_rejects_unsupported_operator_semantics(override, needle) -> None:
    config = GumbelChanceMCTSConfig(**override)
    with pytest.raises(ValueError, match=needle):
        NativeGumbelChanceMCTS(config, _PublicCountingEvaluator())


@pytest.mark.skipif(
    not native_hot_loop_available(), reason="native wheel lacks gumbel_search"
)
def test_native_maps_decoupled_wide_budget_fields() -> None:
    config = GumbelChanceMCTSConfig(
        n_full_wide=256,
        n_full_wide_threshold=40,
        wide_roots_always_full=True,
    )
    search = NativeGumbelChanceMCTS(config, _PublicCountingEvaluator())
    native = search._native_config()
    assert native["n_full_wide"] == 256
    assert native["n_full_wide_threshold"] == 40
    assert native["wide_roots_always_full"] is True


@pytest.mark.skipif(
    not native_hot_loop_available(), reason="native wheel lacks gumbel_search"
)
def test_native_config_maps_sigma_reference_visits() -> None:
    search = NativeGumbelChanceMCTS(
        GumbelChanceMCTSConfig(sigma_reference_visits=12),
        _PublicCountingEvaluator(),
    )
    assert search._native_config()["sigma_reference_visits"] == 12


def test_native_sigma_reference_refuses_unadvertised_old_wheel(monkeypatch) -> None:
    rust = pytest.importorskip("catanatron_rs")
    monkeypatch.delattr(rust, "gumbel_search_capabilities", raising=False)
    search = object.__new__(NativeGumbelChanceMCTS)
    search.config = GumbelChanceMCTSConfig(sigma_reference_visits=12)
    search.using_native_hot_loop = True

    with pytest.raises(ValueError, match="advertising the matching calibration"):
        search._validate_native_semantics()


@pytest.mark.skipif(
    not native_hot_loop_available(), reason="native wheel lacks gumbel_search"
)
def test_native_engine_seed_stream_advances_and_is_reproducible(monkeypatch) -> None:
    rust = pytest.importorskip("catanatron_rs")
    recorded: list[int] = []

    def fake_search(game, evaluator, config, **kwargs):
        recorded.append(int(config["seed"]))
        legal = game.playable_action_indices(["RED", "BLUE"], None)
        probability = 1.0 / len(legal)
        policy = {action: probability for action in legal}
        return {
            "selected_action": legal[0],
            "improved_policy": policy,
            "visit_counts": {action: 0 for action in legal},
            "q_values": {},
            "priors": policy,
            "root_value": 0.0,
            "used_full_search": True,
            "simulations_used": 1,
            "afterstate_values": {},
        }

    monkeypatch.setattr(rust, "gumbel_search", fake_search)
    config = GumbelChanceMCTSConfig(seed=997)
    game = rust.Game.simple(["RED", "BLUE"], seed=5)
    for _ in range(2):
        search = NativeGumbelChanceMCTS(config, _PublicCountingEvaluator())
        search._search_single_world(game, force_full=True)
        search._search_single_world(game, force_full=True)

    assert recorded[0] != recorded[1]
    assert recorded[:2] == recorded[2:]


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


@pytest.mark.skipif(
    not native_hot_loop_available(), reason="native wheel lacks gumbel_search"
)
def test_native_forced_roll_supports_scalar_only_evaluator() -> None:
    rust = pytest.importorskip("catanatron_rs")
    game = rust.Game.simple(["RED", "BLUE"], seed=43)
    for _ in range(100):
        actions = json.loads(game.playable_actions_json())
        if len(actions) == 1 and actions[0][1] == "ROLL":
            break
        game.play_tick()
    else:
        pytest.fail("did not reach a forced ROLL root")
    result = NativeGumbelChanceMCTS(
        GumbelChanceMCTSConfig(seed=3), _ScalarOnlyPublicEvaluator()
    ).search(game)
    assert result.selected_action in game.playable_action_indices(["RED", "BLUE"], None)


@pytest.mark.skipif(
    not native_hot_loop_available(), reason="native wheel lacks gumbel_search"
)
def test_binding_rejects_experimental_deferred_batching() -> None:
    rust = pytest.importorskip("catanatron_rs")
    game = rust.Game.simple(["RED", "BLUE"], seed=47)
    with pytest.raises(ValueError, match="batch_size>0.*not reference-equivalent"):
        rust.gumbel_search(
            game,
            lambda *_args: ({}, 0.0),
            {"batch_size": 1, "colors": ["RED", "BLUE"]},
        )


@pytest.mark.skipif(
    not native_hot_loop_available(), reason="native wheel lacks gumbel_search"
)
@pytest.mark.parametrize("delta", [-1, 1])
def test_native_chance_batch_length_mismatch_is_clean_error(delta: int) -> None:
    rust = pytest.importorskip("catanatron_rs")
    game = rust.Game.simple(["RED", "BLUE"], seed=53)
    for _ in range(100):
        actions = json.loads(game.playable_actions_json())
        if len(actions) == 1 and actions[0][1] == "ROLL":
            break
        game.play_tick()
    else:
        pytest.fail("did not reach a forced ROLL root")
    with pytest.raises(RuntimeError, match="batch length mismatch"):
        NativeGumbelChanceMCTS(
            GumbelChanceMCTSConfig(seed=7), _WrongBatchPublicEvaluator(delta)
        ).search(game)


@pytest.mark.skipif(
    not native_hot_loop_available(), reason="native wheel lacks gumbel_search"
)
def test_native_result_json_is_deterministic_across_processes() -> None:
    program = textwrap.dedent(
        """
        import json
        from types import SimpleNamespace
        import catanatron_rs
        from catan_zero.search.gumbel_chance_mcts import GumbelChanceMCTSConfig
        from catan_zero.search.native_gumbel_mcts import NativeGumbelChanceMCTS
        from catan_zero.search.rust_mcts import HeuristicRustEvaluator
        class Eval:
            def __init__(self):
                self.config = SimpleNamespace(public_observation=True, cache_size=0)
                self.inner = HeuristicRustEvaluator(score_actions=False)
            def evaluate(self, *args, **kwargs):
                return self.inner.evaluate(*args, **kwargs)
            def evaluate_many(self, requests, *, root_color, colors):
                return self.inner.evaluate_many(requests, root_color=root_color, colors=colors)
        config = GumbelChanceMCTSConfig(
            seed=71, n_full=32, n_fast=32, p_full=1.0,
            exact_budget_sh=True, max_depth=5,
            variance_aware_q=True, variance_aware_closed_form_js=True,
        )
        result = NativeGumbelChanceMCTS(config, Eval()).search(
            catanatron_rs.Game.simple(["RED", "BLUE"], seed=73), force_full=True
        )
        print(json.dumps({
            "selected": result.selected_action,
            "policy": result.improved_policy,
            "visits": result.visit_counts,
            "q": result.q_values,
            "priors": result.priors,
            "afterstates": result.afterstate_values,
        }, sort_keys=True, separators=(",", ":")))
        """
    )
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    outputs = [
        subprocess.check_output(
            [sys.executable, "-c", program], env=environment, text=True
        ).strip()
        for _ in range(2)
    ]
    assert outputs[0] == outputs[1]


@pytest.mark.skipif(
    not native_hot_loop_available(), reason="native wheel lacks gumbel_search"
)
def test_t0_exact_tie_preserves_unsorted_legal_insertion_order() -> None:
    rust = pytest.importorskip("catanatron_rs")
    game = rust.Game.simple(["RED", "BLUE"], seed=12)
    for _ in range(377):
        game.play_tick()
    legal = game.playable_action_indices(["RED", "BLUE"], None)
    assert legal == [209, 210, 211, 212, 213, 208, 337]
    config = GumbelChanceMCTSConfig(
        seed=101,
        n_full=7,
        n_fast=7,
        p_full=1.0,
        exact_budget_sh=True,
        c_scale=0.0,
        max_depth=3,
    )
    reference = GumbelChanceMCTS(config, _PublicCountingEvaluator()).search(
        game.copy(), force_full=True
    )
    native = NativeGumbelChanceMCTS(config, _PublicCountingEvaluator()).search(
        game.copy(), force_full=True
    )
    assert len(set(reference.improved_policy.values())) == 1
    assert len(set(native.improved_policy.values())) == 1
    assert reference.selected_action == legal[0]
    assert native.selected_action == legal[0]
    assert native.selected_action != min(legal)
