from __future__ import annotations

from pathlib import Path
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
    policy = {3: 0.8, 7: 0.2, 11: 0.0}

    assert _temperature_scale_policy(policy, 1.0) is policy
    assert _temperature_scale_policy(policy, 0.5) == pytest.approx(
        {3: 16.0 / 17.0, 7: 1.0 / 17.0, 11: 0.0}
    )
    assert _temperature_scale_policy(policy, 2.0) == pytest.approx(
        {3: 2.0 / 3.0, 7: 1.0 / 3.0, 11: 0.0}
    )


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
