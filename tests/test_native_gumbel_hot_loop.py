from __future__ import annotations

import random
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


class _ForcedCountingEvaluator:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, *args, **kwargs):
        del args, kwargs
        self.calls += 1
        return ({37: 1.0}, 0.625)


class _LegacyRepeatedRootD6(NativeGumbelChanceMCTS):
    """Test oracle for the pre-hoist PIMC evaluator schedule."""

    def _can_share_information_set_root_evaluation(self, legal_width: int) -> bool:
        del legal_width
        return False


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

    repeated_evaluator = _PublicCountingEvaluator()
    repeated = _LegacyRepeatedRootD6(config, repeated_evaluator).search(
        rust.Game.simple(["RED", "BLUE"], seed=31), force_full=True
    )

    assert result.simulations_used == 128
    # The four PIMC particles expose the same public root.  D6 is evaluated
    # once (12 orientation forwards), not once per hidden-world particle (48).
    assert evaluator.root_symmetry_calls == 1
    assert repeated_evaluator.root_symmetry_calls == 4
    assert result == repeated
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


def test_native_particle_override_preserves_exact_per_particle_dose() -> None:
    search = object.__new__(NativeGumbelChanceMCTS)
    search.config = GumbelChanceMCTSConfig(
        n_full=128,
        n_full_wide=256,
        n_full_wide_threshold=40,
        wide_roots_always_full=True,
    )
    search.rng = random.Random(7)

    native = search._native_config(n_simulations_override=32)

    assert native["n_full"] == 32
    assert native["n_fast"] == 32
    assert native["p_full"] == 1.0
    assert native["exact_budget_sh"] is True
    assert native["exact_budget_sh_min_n"] == 0
    assert "n_full_wide" not in native


def test_native_config_does_not_retemper_pretempered_evaluator_priors() -> None:
    search = object.__new__(NativeGumbelChanceMCTS)
    search.config = GumbelChanceMCTSConfig(prior_temperature=2.0)
    search.evaluator = SimpleNamespace(applied_prior_temperature=2.0)
    search.rng = random.Random(7)

    assert search._native_config()["prior_temperature"] == pytest.approx(1.0)


def test_native_particle_root_callback_replays_immutable_precomputed_d6(
    monkeypatch,
) -> None:
    cached = ({3: 0.25, 7: 0.75}, 0.4)
    callback_results = []

    class _Game:
        def current_color(self):
            return "RED"

        def playable_action_indices(self, colors, map_kind):
            assert tuple(colors) == ("RED", "BLUE")
            assert map_kind is None
            return [3, 7]

    def fake_search(
        game,
        evaluator,
        config,
        *,
        evaluator_many,
        root_evaluator,
        force_full,
    ):
        del evaluator, config, evaluator_many, force_full
        assert root_evaluator is not None
        first = root_evaluator(game, [3, 7], "RED")
        first[0][3] = 999.0
        second = root_evaluator(game, [3, 7], "RED")
        callback_results.append(second)
        return {
            "selected_action": 7,
            "improved_policy": {3: 0.25, 7: 0.75},
            "visit_counts": {3: 8, 7: 24},
            "q_values": {3: -0.1, 7: 0.2},
            "priors": {3: 0.25, 7: 0.75},
            "root_value": 0.4,
            "used_full_search": True,
            "simulations_used": 32,
            "afterstate_values": {},
        }

    monkeypatch.setitem(
        sys.modules,
        "catanatron_rs",
        SimpleNamespace(gumbel_search=fake_search),
    )
    search = object.__new__(NativeGumbelChanceMCTS)
    search.config = GumbelChanceMCTSConfig(seed=41)
    search.evaluator = _PublicCountingEvaluator()
    search.rng = random.Random(41)
    search.using_native_hot_loop = True

    result = search._search_single_world(
        _Game(),
        force_full=True,
        n_simulations_override=32,
        precomputed_root_evaluation=cached,
    )

    assert callback_results == [cached]
    assert cached == ({3: 0.25, 7: 0.75}, 0.4)
    assert search.evaluator.root_symmetry_calls == 0
    assert result.simulations_used == 32
    # Old native wheels do not return this field, so the binding must retain
    # the exact root callback value rather than substituting post-search value.
    assert result.root_prior_value == pytest.approx(0.4)


def test_native_leaf_observer_excludes_root_and_covers_scalar_and_batch(
    monkeypatch,
) -> None:
    root_game = object()
    scalar_leaf = object()
    batch_leaf = object()
    observed = []

    class _Game:
        def playable_action_indices(self, colors, map_kind):
            assert tuple(colors) == ("RED", "BLUE")
            assert map_kind is None
            return [3, 7]

    class _Evaluator:
        def evaluate(self, game, legal, *, root_color, colors):
            del game, root_color, colors
            return ({int(action): 1.0 / len(legal) for action in legal}, 0.25)

        def evaluate_many(self, requests, *, root_color, colors):
            return [
                self.evaluate(game, legal, root_color=root_color, colors=colors)
                for game, legal in requests
            ]

    def fake_search(
        game,
        evaluator,
        config,
        *,
        evaluator_many,
        root_evaluator,
        force_full,
    ):
        del config, force_full
        root_evaluator(root_game, [3, 7], "RED")
        evaluator(scalar_leaf, [3], "RED")
        evaluator_many([(batch_leaf, [7], "RED")])
        return {
            "selected_action": 3,
            "improved_policy": {3: 0.5, 7: 0.5},
            "visit_counts": {3: 1, 7: 1},
            "q_values": {3: 0.0, 7: 0.0},
            "priors": {3: 0.5, 7: 0.5},
            "root_value": 0.25,
            "used_full_search": True,
            "simulations_used": 2,
            "afterstate_values": {},
        }

    monkeypatch.setitem(
        sys.modules,
        "catanatron_rs",
        SimpleNamespace(gumbel_search=fake_search),
    )
    search = object.__new__(NativeGumbelChanceMCTS)
    search.config = GumbelChanceMCTSConfig(seed=7)
    search.evaluator = _Evaluator()
    search.rng = random.Random(7)
    search.using_native_hot_loop = True
    search.set_leaf_evaluation_observer(
        lambda game, legal, root_color: observed.append((game, legal, root_color))
    )

    result = search._search_single_world(_Game(), force_full=True)

    assert result.simulations_used == 2
    assert observed == [
        (scalar_leaf, (3,), "RED"),
        (batch_leaf, (7,), "RED"),
    ]


def test_native_config_maps_sigma_reference_visits() -> None:
    # Mapping is a pure Python boundary contract.  Do not require the installed
    # wheel to advertise the new capability here: the separate test below
    # proves that an older wheel fails closed at engine construction time.
    search = object.__new__(NativeGumbelChanceMCTS)
    search.config = GumbelChanceMCTSConfig(sigma_reference_visits=12)
    search.rng = random.Random(7)
    assert search._native_config()["sigma_reference_visits"] == 12


def test_native_config_maps_coherent_public_belief_boundary() -> None:
    search = object.__new__(NativeGumbelChanceMCTS)
    search.config = GumbelChanceMCTSConfig(coherent_public_belief_search=True)
    search.rng = random.Random(7)

    native = search._native_config()

    assert native["coherent_public_belief_search"] is True
    assert native["stop_at_root_turn_boundary"] is True


def test_native_config_maps_boundary_particle_seeds_only_when_enabled() -> None:
    legacy = object.__new__(NativeGumbelChanceMCTS)
    legacy.config = GumbelChanceMCTSConfig(
        coherent_public_belief_search=True,
        boundary_value_particles=1,
    )
    legacy.rng = random.Random(7)
    legacy._boundary_value_particle_seeds = ()
    assert "boundary_value_particle_seeds" not in legacy._native_config()

    particles = object.__new__(NativeGumbelChanceMCTS)
    particles.config = GumbelChanceMCTSConfig(
        coherent_public_belief_search=True,
        boundary_value_particles=3,
    )
    particles.rng = random.Random(7)
    particles._boundary_value_particle_seeds = (11, 22, 33)
    assert particles._native_config()["boundary_value_particle_seeds"] == [
        11,
        22,
        33,
    ]


def test_native_boundary_particles_require_advertised_capability(monkeypatch) -> None:
    fake_module = SimpleNamespace(
        gumbel_search=lambda *_args, **_kwargs: None,
        gumbel_search_capabilities=lambda: ["coherent_public_belief_search"],
    )
    monkeypatch.setitem(sys.modules, "catanatron_rs", fake_module)
    search = object.__new__(NativeGumbelChanceMCTS)
    search.config = GumbelChanceMCTSConfig(
        coherent_public_belief_search=True,
        boundary_value_particles=3,
    )
    search.using_native_hot_loop = True

    with pytest.raises(ValueError, match="boundary_value_particles"):
        search._validate_native_semantics()


def test_native_boundary_callback_uses_real_particle_legal_sets(monkeypatch) -> None:
    class _Particle:
        def __init__(self, seed: int) -> None:
            self.seed = int(seed)

        def playable_action_indices(self, colors, map_kind):
            assert tuple(colors) == ("RED", "BLUE")
            assert map_kind is None
            return list(range(1, self.seed + 1))

    class _Game:
        def playable_action_indices(self, colors, map_kind):
            assert tuple(colors) == ("RED", "BLUE")
            assert map_kind is None
            return [3, 7]

        def determinize_from_observer_information(self, observer, seed):
            assert observer == "RED"
            return _Particle(int(seed))

    class _Evaluator:
        config = SimpleNamespace(public_observation=True, emit_uncertainty=False)

        def __init__(self) -> None:
            self.legal_sets = []

        def evaluate(self, game, legal, *, root_color, colors):
            del game, root_color, colors
            return ({int(action): 1.0 / len(legal) for action in legal}, 0.0)

        def evaluate_many(self, requests, *, root_color, colors):
            assert root_color == "RED"
            assert tuple(colors) == ("RED", "BLUE")
            self.legal_sets.append([legal for _game, legal in requests])
            return [
                ({int(action): 1.0 / len(legal) for action in legal}, game.seed / 10.0)
                for game, legal in requests
            ]

    observed = {}

    def fake_search(
        game,
        evaluator,
        config,
        *,
        evaluator_many,
        root_evaluator,
        force_full,
        boundary_evaluator,
    ):
        del evaluator, evaluator_many, root_evaluator
        observed["force_full"] = force_full
        observed["seeds"] = config["boundary_value_particle_seeds"]
        observed["value"] = boundary_evaluator(game, "RED", observed["seeds"])
        return {
            "selected_action": 3,
            "improved_policy": {3: 0.5, 7: 0.5},
            "visit_counts": {3: 1, 7: 1},
            "q_values": {3: 0.0, 7: 0.0},
            "priors": {3: 0.5, 7: 0.5},
            "root_value": observed["value"],
            "used_full_search": True,
            "simulations_used": 2,
            "afterstate_values": {},
        }

    monkeypatch.setitem(
        sys.modules,
        "catanatron_rs",
        SimpleNamespace(gumbel_search=fake_search),
    )
    evaluator = _Evaluator()
    search = object.__new__(NativeGumbelChanceMCTS)
    search.config = GumbelChanceMCTSConfig(
        seed=7,
        coherent_public_belief_search=True,
        boundary_value_particles=3,
    )
    search.evaluator = evaluator
    search.rng = random.Random(7)
    search.using_native_hot_loop = True
    search._boundary_value_particle_seeds = (1, 2, 3)
    search._leaf_evaluation_observer = None

    result = search._search_single_world(_Game(), force_full=True)

    assert observed == {
        "force_full": True,
        "seeds": [1, 2, 3],
        "value": pytest.approx(0.2),
    }
    assert evaluator.legal_sets == [[(1,), (1, 2), (1, 2, 3)]]
    assert result.root_value == pytest.approx(0.2)


def test_native_boundary_callback_rejects_batch_cardinality_mismatch(
    monkeypatch,
) -> None:
    class _Particle:
        def playable_action_indices(self, _colors, _map_kind):
            return [1]

    class _Game:
        def playable_action_indices(self, _colors, _map_kind):
            return [3, 7]

        def determinize_from_observer_information(self, _observer, _seed):
            return _Particle()

    class _Evaluator:
        config = SimpleNamespace(public_observation=True, emit_uncertainty=False)

        def evaluate_many(self, requests, *, root_color, colors):
            del root_color, colors
            return [({1: 1.0}, 0.0)] * max(0, len(requests) - 1)

        def evaluate(self, *_args, **_kwargs):
            return ({3: 0.5, 7: 0.5}, 0.0)

    def fake_search(
        game,
        _evaluator,
        config,
        *,
        evaluator_many,
        root_evaluator,
        force_full,
        boundary_evaluator,
    ):
        del evaluator_many, root_evaluator, force_full
        boundary_evaluator(game, "RED", config["boundary_value_particle_seeds"])
        raise AssertionError("cardinality mismatch should fail before search returns")

    monkeypatch.setitem(
        sys.modules,
        "catanatron_rs",
        SimpleNamespace(gumbel_search=fake_search),
    )
    search = object.__new__(NativeGumbelChanceMCTS)
    search.config = GumbelChanceMCTSConfig(
        coherent_public_belief_search=True,
        boundary_value_particles=2,
    )
    search.evaluator = _Evaluator()
    search.rng = random.Random(7)
    search.using_native_hot_loop = True
    search._boundary_value_particle_seeds = (1, 2)
    search._leaf_evaluation_observer = None

    with pytest.raises(RuntimeError, match="cardinality mismatch"):
        search._search_single_world(_Game(), force_full=True)


def test_forced_trajectory_only_keeps_action_and_omits_discarded_targets() -> None:
    evaluator = _ForcedCountingEvaluator()
    trajectory = object.__new__(GumbelChanceMCTS)
    trajectory.config = GumbelChanceMCTSConfig(
        forced_root_target_mode="trajectory_only"
    )
    trajectory.evaluator = evaluator

    result = trajectory._forced_single_action_result(
        object(),
        (37,),
        root_color="RED",
        action_json_by_id={37: ["RED", "END_TURN", None]},
    )

    assert result.selected_action == 37
    assert result.improved_policy == {37: 1.0}
    assert result.priors == {37: 1.0}
    assert result.visit_counts == {}
    assert result.q_values == {}
    assert result.completed_q_values == {}
    assert result.afterstate_values == {}
    assert result.root_value != result.root_value
    assert result.used_full_search is False
    assert result.simulations_used == 0
    assert evaluator.calls == 0

    full = object.__new__(GumbelChanceMCTS)
    full.config = GumbelChanceMCTSConfig()
    full.evaluator = evaluator
    control = full._forced_single_action_result(
        object(),
        (37,),
        root_color="RED",
        action_json_by_id={37: ["RED", "END_TURN", None]},
    )
    assert control.selected_action == result.selected_action
    assert control.root_value == pytest.approx(0.625)
    assert evaluator.calls == 1


def test_native_forced_trajectory_short_circuits_before_binding(monkeypatch) -> None:
    binding_calls = 0

    def unexpected_binding(*args, **kwargs):
        nonlocal binding_calls
        del args, kwargs
        binding_calls += 1
        raise AssertionError("native search binding must not run at a forced root")

    monkeypatch.setitem(
        sys.modules,
        "catanatron_rs",
        SimpleNamespace(gumbel_search=unexpected_binding),
    )

    class _ForcedGame:
        def playable_action_indices(self, colors, map_kind):
            assert tuple(colors) == ("RED", "BLUE")
            assert map_kind is None
            return [37]

    evaluator = _ForcedCountingEvaluator()
    search = object.__new__(NativeGumbelChanceMCTS)
    search.config = GumbelChanceMCTSConfig(forced_root_target_mode="trajectory_only")
    search.evaluator = evaluator
    search.rng = random.Random(7)
    search.using_native_hot_loop = True

    result = search._search_single_world(_ForcedGame())

    assert result.selected_action == 37
    assert result.root_value != result.root_value
    assert result.completed_q_values == {}
    assert evaluator.calls == 0
    assert binding_calls == 0


def test_native_config_maps_forced_root_target_mode() -> None:
    search = object.__new__(NativeGumbelChanceMCTS)
    search.config = GumbelChanceMCTSConfig(forced_root_target_mode="trajectory_only")
    search.rng = random.Random(7)
    assert search._native_config()["forced_root_target_mode"] == "trajectory_only"


def test_native_config_binds_authoritative_initial_road_d1_scope() -> None:
    search = object.__new__(NativeGumbelChanceMCTS)
    search.config = GumbelChanceMCTSConfig(
        rescale_noise_floor_c=8.0,
        rescale_noise_floor_initial_road_only=True,
    )
    search.rng = random.Random(7)

    native = search._native_config(attested_root_phase="BUILD_INITIAL_ROAD")
    assert native["rescale_noise_floor_initial_road_only"] is True
    assert native["attested_root_phase"] == "BUILD_INITIAL_ROAD"
    with pytest.raises(RuntimeError, match="authoritative root-phase attestation"):
        search._native_config()


def test_native_and_python_share_exact_belief_gameplay_aggregator() -> None:
    # Native owns only per-particle traversal. Public-belief aggregation and
    # action selection remain the inherited Python source of truth.
    assert (
        NativeGumbelChanceMCTS._aggregate_information_set_results
        is GumbelChanceMCTS._aggregate_information_set_results
    )


def test_native_sigma_reference_refuses_unadvertised_old_wheel(monkeypatch) -> None:
    rust = pytest.importorskip("catanatron_rs")
    monkeypatch.delattr(rust, "gumbel_search_capabilities", raising=False)
    search = object.__new__(NativeGumbelChanceMCTS)
    search.config = GumbelChanceMCTSConfig(sigma_reference_visits=12)
    search.using_native_hot_loop = True

    with pytest.raises(ValueError, match="advertising the matching calibration"):
        search._validate_native_semantics()


def test_native_initial_road_d1_refuses_unadvertised_old_wheel(monkeypatch) -> None:
    rust = pytest.importorskip("catanatron_rs")
    monkeypatch.setattr(
        rust,
        "gumbel_search_capabilities",
        lambda: ["sigma_reference_visits", "belief_target_evidence"],
        raising=False,
    )
    search = object.__new__(NativeGumbelChanceMCTS)
    search.config = GumbelChanceMCTSConfig(
        rescale_noise_floor_c=8.0,
        rescale_noise_floor_initial_road_only=True,
    )
    search.using_native_hot_loop = True

    with pytest.raises(ValueError, match="initial_road_d1_scope"):
        search._validate_native_semantics()


def test_native_coherent_belief_refuses_unadvertised_old_wheel(monkeypatch) -> None:
    rust = pytest.importorskip("catanatron_rs")
    monkeypatch.setattr(
        rust,
        "gumbel_search_capabilities",
        lambda: [
            "sigma_reference_visits",
            "belief_target_evidence",
            "initial_road_d1_scope",
        ],
        raising=False,
    )
    search = object.__new__(NativeGumbelChanceMCTS)
    search.config = GumbelChanceMCTSConfig(coherent_public_belief_search=True)
    search.using_native_hot_loop = True

    with pytest.raises(ValueError, match="advertising coherent_public_belief_search"):
        search._validate_native_semantics()


def test_native_forced_trajectory_refuses_unadvertised_old_wheel(monkeypatch) -> None:
    rust = pytest.importorskip("catanatron_rs")
    monkeypatch.setattr(
        rust,
        "gumbel_search_capabilities",
        lambda: ["sigma_reference_visits", "belief_target_evidence"],
        raising=False,
    )
    search = object.__new__(NativeGumbelChanceMCTS)
    search.config = GumbelChanceMCTSConfig(forced_root_target_mode="trajectory_only")
    search.using_native_hot_loop = True

    with pytest.raises(ValueError, match="advertising forced_root_trajectory_only"):
        search._validate_native_semantics()


def test_initial_road_d1_is_exact_off_phase_in_python_and_native() -> None:
    rust = pytest.importorskip("catanatron_rs")
    capability_fn = getattr(rust, "gumbel_search_capabilities", None)
    capabilities = set(capability_fn()) if callable(capability_fn) else set()
    if "initial_road_d1_scope" not in capabilities:
        pytest.skip("installed wheel lacks initial_road_d1_scope")

    game = rust.Game.simple(["RED", "BLUE"], seed=83)
    for _ in range(12):
        prompt = json.loads(game.json_snapshot())["current_prompt"]
        if prompt == "PLAY_TURN":
            break
        legal = game.playable_action_indices(["RED", "BLUE"], None)
        game.execute_action_index(legal[0], ["RED", "BLUE"], None)
    else:
        pytest.fail("did not reach an ordinary PLAY_TURN root")

    plain = GumbelChanceMCTSConfig(
        seed=89,
        n_full=32,
        n_fast=32,
        p_full=1.0,
        exact_budget_sh=True,
        max_depth=4,
    )
    scoped = GumbelChanceMCTSConfig(
        seed=89,
        n_full=32,
        n_fast=32,
        p_full=1.0,
        exact_budget_sh=True,
        max_depth=4,
        rescale_noise_floor_c=8.0,
        sigma_eval=0.98,
        rescale_noise_floor_initial_road_only=True,
    )

    for engine in (GumbelChanceMCTS, NativeGumbelChanceMCTS):
        reference = engine(plain, _PublicCountingEvaluator()).search(
            game.copy(), force_full=True
        )
        candidate = engine(scoped, _PublicCountingEvaluator()).search(
            game.copy(), force_full=True
        )
        assert candidate == reference


def test_native_belief_target_refuses_wheel_without_evidence_capability(
    monkeypatch,
) -> None:
    rust = pytest.importorskip("catanatron_rs")
    monkeypatch.setattr(
        rust,
        "gumbel_search_capabilities",
        lambda: ["sigma_reference_visits"],
        raising=False,
    )
    search = object.__new__(NativeGumbelChanceMCTS)
    search.config = GumbelChanceMCTSConfig(
        information_set_search=True,
        information_set_target_aggregation="aggregate_q_then_improve",
        sigma_reference_visits=8,
    )
    search.using_native_hot_loop = True

    with pytest.raises(ValueError, match="belief_target_evidence"):
        search._validate_native_semantics()


def test_native_belief_gameplay_refuses_wheel_without_evidence_capability(
    monkeypatch,
) -> None:
    rust = pytest.importorskip("catanatron_rs")
    monkeypatch.setattr(
        rust,
        "gumbel_search_capabilities",
        lambda: ["sigma_reference_visits"],
        raising=False,
    )
    search = object.__new__(NativeGumbelChanceMCTS)
    search.config = GumbelChanceMCTSConfig(
        information_set_search=True,
        gameplay_policy_aggregation="aggregate_q_then_improve",
        sigma_reference_visits=8,
    )
    search.using_native_hot_loop = True

    with pytest.raises(ValueError, match="belief_target_evidence"):
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
