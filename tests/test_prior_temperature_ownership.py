from __future__ import annotations

from types import SimpleNamespace

import pytest

from catan_zero.search.gumbel_chance_mcts import (
    GumbelChanceMCTS,
    GumbelChanceMCTSConfig,
)
from catan_zero.search.neural_rust_mcts import (
    EntityGraphRustEvaluator,
    EntityGraphRustEvaluatorConfig,
)


def _search(*, search_temperature: float, evaluator: object) -> GumbelChanceMCTS:
    search = object.__new__(GumbelChanceMCTS)
    search.config = GumbelChanceMCTSConfig(
        prior_temperature=search_temperature
    )
    search.evaluator = evaluator
    return search


def test_neural_evaluator_reports_the_temperature_already_applied() -> None:
    evaluator = object.__new__(EntityGraphRustEvaluator)
    evaluator.config = EntityGraphRustEvaluatorConfig(prior_temperature=2.0)

    assert evaluator.applied_prior_temperature == pytest.approx(2.0)


def test_search_applies_temperature_for_raw_prior_evaluator() -> None:
    search = _search(search_temperature=2.0, evaluator=object())

    assert search._effective_prior_temperature() == pytest.approx(2.0)


@pytest.mark.parametrize("search_temperature", [1.0, 2.0])
def test_search_does_not_reapply_evaluator_temperature(
    search_temperature: float,
) -> None:
    evaluator = SimpleNamespace(applied_prior_temperature=2.0)
    search = _search(
        search_temperature=search_temperature,
        evaluator=evaluator,
    )

    assert search._effective_prior_temperature() == pytest.approx(1.0)


def test_search_rejects_two_distinct_nonunit_temperatures() -> None:
    evaluator = SimpleNamespace(applied_prior_temperature=2.0)
    search = _search(search_temperature=3.0, evaluator=evaluator)

    with pytest.raises(ValueError, match="conflicting prior_temperature"):
        search._effective_prior_temperature()


def test_nonunit_application_contract_records_single_operator() -> None:
    evaluator = SimpleNamespace(applied_prior_temperature=2.0)
    search = _search(search_temperature=2.0, evaluator=evaluator)

    assert search.prior_temperature_application_contract() == {
        "schema_version": "gumbel-prior-temperature-application-v2",
        "configured_search_prior_temperature": 2.0,
        "evaluator_applied_prior_temperature": 2.0,
        "effective_search_prior_temperature": 1.0,
        "effective_logit_temperature": 2.0,
        "application_count": 1,
        "semantics": "single_application_v2",
    }


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan"), float("inf")])
def test_search_rejects_invalid_prior_temperature(temperature: float) -> None:
    search = _search(search_temperature=temperature, evaluator=object())

    with pytest.raises(ValueError, match="finite and positive"):
        search._effective_prior_temperature()


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan"), float("inf")])
def test_search_rejects_invalid_evaluator_temperature(temperature: float) -> None:
    evaluator = SimpleNamespace(applied_prior_temperature=temperature)
    search = _search(search_temperature=1.0, evaluator=evaluator)

    with pytest.raises(ValueError, match="finite and positive"):
        search._effective_prior_temperature()
