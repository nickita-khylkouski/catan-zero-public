from __future__ import annotations

from pathlib import Path
import random
import sys
from types import SimpleNamespace

import pytest

from catan_zero.search.gumbel_chance_mcts import (
    GumbelChanceMCTS,
    GumbelChanceMCTSConfig,
    SearchResult,
    _temperature_scale_policy,
)
from catan_zero.search.native_gumbel_mcts import NativeGumbelChanceMCTS


def test_gameplay_temperature_scaling_is_identity_noop_at_one_and_preserves_zeros() -> None:
    # Shared Python/Rust golden contract (the Rust unit test uses the same
    # support and exact fractions): legal support never changes and a legal
    # zero-probability action remains zero.
    policy = {3: 0.8, 7: 0.2, 11: 0.0}

    assert _temperature_scale_policy(policy, 1.0) is policy
    assert _temperature_scale_policy(policy, 0.5) == pytest.approx(
        {3: 16.0 / 17.0, 7: 1.0 / 17.0, 11: 0.0}
    )
    assert _temperature_scale_policy(policy, 2.0) == pytest.approx(
        {3: 2.0 / 3.0, 7: 1.0 / 3.0, 11: 0.0}
    )

    # A forced legal move is invariant at every positive temperature.
    assert _temperature_scale_policy({29: 1.0}, 0.3) == {29: 1.0}


@pytest.mark.parametrize("temperature", [0.0, -0.1, float("nan"), float("inf")])
def test_gameplay_temperature_scaling_rejects_invalid_values(
    temperature: float,
) -> None:
    with pytest.raises(ValueError, match="temperature"):
        _temperature_scale_policy({3: 1.0}, temperature)


@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("probability", [-0.1, float("nan"), float("inf")])
def test_gameplay_temperature_scaling_rejects_any_invalid_probability(
    temperature: float, probability: float
) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        _temperature_scale_policy({3: 0.8, 7: probability}, temperature)


@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
def test_gameplay_temperature_scaling_rejects_zero_mass(
    temperature: float,
) -> None:
    with pytest.raises(ValueError, match="positive finite mass"):
        _temperature_scale_policy({3: 0.0, 7: 0.0}, temperature)


def test_information_set_gameplay_temperature_does_not_mutate_training_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[int, float] = {}

    def capture_policy(
        _search: GumbelChanceMCTS, policy: dict[int, float]
    ) -> int:
        captured.update(policy)
        return 3

    monkeypatch.setattr(GumbelChanceMCTS, "_sample_categorical", capture_policy)
    search = object.__new__(GumbelChanceMCTS)
    search.config = GumbelChanceMCTSConfig(temperature=0.5)
    particle = SearchResult(
        selected_action=3,
        improved_policy={3: 0.8, 7: 0.2},
        visit_counts={3: 8, 7: 2},
        q_values={3: 0.4, 7: 0.1},
        priors={3: 0.6, 7: 0.4},
        root_value=0.2,
        used_full_search=True,
        simulations_used=10,
    )

    result = search._aggregate_information_set_results(
        [particle], legal_actions=(3, 7), used_full_search=True
    )

    assert captured == pytest.approx({3: 16.0 / 17.0, 7: 1.0 / 17.0})
    assert result.improved_policy == pytest.approx({3: 0.8, 7: 0.2})


def test_zero_temperature_is_deterministic_argmax_and_never_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_sampled(
        _search: GumbelChanceMCTS, _policy: dict[int, float]
    ) -> int:
        raise AssertionError("T=0 must not enter the categorical sampling path")

    monkeypatch.setattr(GumbelChanceMCTS, "_sample_categorical", fail_if_sampled)
    search = object.__new__(GumbelChanceMCTS)
    search.config = GumbelChanceMCTSConfig(temperature=0.0)
    particle = SearchResult(
        selected_action=7,
        improved_policy={3: 0.8, 7: 0.2},
        visit_counts={3: 8, 7: 2},
        q_values={3: 0.4, 7: 0.1},
        priors={3: 0.6, 7: 0.4},
        root_value=0.2,
        used_full_search=True,
        simulations_used=10,
    )

    result = search._aggregate_information_set_results(
        [particle], legal_actions=(3, 7), used_full_search=True
    )

    assert result.selected_action == 3
    assert result.improved_policy == pytest.approx({3: 0.8, 7: 0.2})


def test_native_nonunit_temperature_refuses_stale_wheel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "catanatron_rs",
        SimpleNamespace(gumbel_search_capabilities=lambda: []),
    )
    search = object.__new__(NativeGumbelChanceMCTS)
    search.config = GumbelChanceMCTSConfig(temperature=0.3)
    search.using_native_hot_loop = True

    with pytest.raises(ValueError, match="policy_temperature_semantics"):
        search._validate_native_semantics()


def test_native_rng_stream_separation_refuses_silent_operator_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "catanatron_rs",
        SimpleNamespace(gumbel_search_capabilities=lambda: []),
    )
    search = object.__new__(NativeGumbelChanceMCTS)
    search.config = GumbelChanceMCTSConfig(rng_stream_separation=True)
    search.using_native_hot_loop = True

    with pytest.raises(ValueError, match="rng_stream_separation"):
        search._validate_native_semantics()

def test_native_rng_stream_separation_routes_explicit_seed_domains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "catanatron_rs",
        SimpleNamespace(
            gumbel_search_capabilities=lambda: ["rng_stream_separation"]
        ),
    )
    search = object.__new__(NativeGumbelChanceMCTS)
    search.config = GumbelChanceMCTSConfig(rng_stream_separation=True)
    search.using_native_hot_loop = True
    search._gumbel_rng = SimpleNamespace(getrandbits=lambda _bits: 11)
    search._chance_rng = SimpleNamespace(getrandbits=lambda _bits: 13)
    search._belief_materialization_seed = 17
    search._boundary_value_particle_seeds = ()

    native = search._native_config()

    assert native["seed"] == 11
    assert native["control_seed"] == 11
    assert native["chance_seed"] == 13
    assert native["belief_seed"] == 17


def test_native_belief_materialization_seed_ignores_advancing_belief_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "catanatron_rs",
        SimpleNamespace(
            gumbel_search_capabilities=lambda: ["rng_stream_separation"]
        ),
    )
    search = object.__new__(NativeGumbelChanceMCTS)
    search.config = GumbelChanceMCTSConfig(rng_stream_separation=True)
    search.using_native_hot_loop = True
    search._gumbel_rng = SimpleNamespace(getrandbits=lambda _bits: 11)
    search._chance_rng = SimpleNamespace(getrandbits=lambda _bits: 13)
    search._belief_rng = SimpleNamespace(
        getrandbits=lambda _bits: (_ for _ in ()).throw(
            AssertionError("native config must not consume advancing belief RNG")
        )
    )
    search._belief_materialization_seed = 17
    search._boundary_value_particle_seeds = ()

    assert search._native_config()["belief_seed"] == 17


def test_native_and_reference_bind_same_static_belief_materialization_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "catanatron_rs",
        SimpleNamespace(
            gumbel_search_capabilities=lambda: ["rng_stream_separation"]
        ),
    )
    config = GumbelChanceMCTSConfig(rng_stream_separation=True)
    reference = object.__new__(GumbelChanceMCTS)
    reference.config = config
    reference.rng = random.Random()
    reference.seed_search_rngs(101)

    native = object.__new__(NativeGumbelChanceMCTS)
    native.config = config
    native.using_native_hot_loop = True
    native.rng = random.Random()
    native.seed_search_rngs(101)
    native._boundary_value_particle_seeds = ()

    assert (
        native._native_config()["belief_seed"]
        == reference._belief_materialization_seed
    )


def test_native_config_revalidates_temperature_after_runtime_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "catanatron_rs",
        SimpleNamespace(gumbel_search_capabilities=lambda: []),
    )
    search = object.__new__(NativeGumbelChanceMCTS)
    search.config = GumbelChanceMCTSConfig(temperature=0.0)
    search.using_native_hot_loop = True
    search.rng = SimpleNamespace(getrandbits=lambda _bits: 7)
    search._validate_native_semantics()

    search.config = GumbelChanceMCTSConfig(temperature=0.3)
    with pytest.raises(ValueError, match="policy_temperature_semantics"):
        search._native_config()


@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_native_zero_and_unit_temperature_keep_legacy_noop_contract(
    monkeypatch: pytest.MonkeyPatch, temperature: float
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "catanatron_rs",
        SimpleNamespace(gumbel_search_capabilities=lambda: []),
    )
    search = object.__new__(NativeGumbelChanceMCTS)
    search.config = GumbelChanceMCTSConfig(temperature=temperature)
    search.using_native_hot_loop = True

    search._validate_native_semantics()


def test_native_binding_advertises_policy_temperature_semantics() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "native/gumbel_mcts_rs/src/python_binding.rs"
    ).read_text(encoding="utf-8")
    assert '"policy_temperature_semantics"' in source
