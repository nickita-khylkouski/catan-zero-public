from __future__ import annotations

import pytest

from catan_zero.search.accounting import SearchAccountingEvaluator
from catan_zero.search.operator_runner import (
    GameCounterAccumulator,
    MeasuredDecision,
    SearchCounters,
)


def test_counter_mapping_uses_explicit_leaves_and_orientation_rows() -> None:
    class _Evaluator:
        def evaluate_many(self, requests, **_kwargs):
            return [({}, 0.0)] * len(requests)

        def evaluate_symmetry_averaged(self, *_args, **_kwargs):
            return {}, 0.0

    evaluator = SearchAccountingEvaluator(_Evaluator(), symmetry_orientations=12)
    evaluator.evaluate_many([(object(), (1,)), (object(), (2,))])
    evaluator.evaluate_symmetry_averaged(object(), (1,))
    counters = SearchCounters.from_work(
        nominal_visits=8,
        scheduled_visits=7,
        work=evaluator.snapshot(),
        wall_time_sec=0.25,
    )
    assert counters.as_dict() == {
        "nominal_visits": 8,
        "scheduled_visits": 7,
        "logical_leaves": 3,
        "orientation_rows": 14,
        "evaluator_calls": 2,
        "wall_time_sec": 0.25,
    }


def test_public_regime_literal_is_fail_closed() -> None:
    from catan_zero.search import operator_runner

    assert "public_conservation_pimc" in operator_runner.InformationRegime.__args__
    assert "public_belief_single_tree" in operator_runner.InformationRegime.__args__
    assert "public_observation_policy" in operator_runner.InformationRegime.__args__
    assert "authoritative_hidden_state" in operator_runner.InformationRegime.__args__


def test_invalid_counter_delta_is_not_hidden() -> None:
    from catan_zero.search.accounting import SearchWork

    with pytest.raises(ValueError, match="not monotonic"):
        _ = SearchWork() - SearchWork(evaluator_calls=1)


def test_game_accumulator_emits_leaderboard_counter_names() -> None:
    accumulator = GameCounterAccumulator()
    accumulator.add(
        MeasuredDecision(
            selected_action=1,
            policy={1: 1.0},
            q_values={},
            root_value=0.0,
            counters=SearchCounters(8, 7, 5, 9, 2, 0.125),
            information_regime="authoritative_hidden_state",
        )
    )
    accumulator.add(
        MeasuredDecision(
            selected_action=2,
            policy={2: 1.0},
            q_values={},
            root_value=0.0,
            counters=SearchCounters(4, 4, 3, 3, 1, 0.25),
            information_regime="authoritative_hidden_state",
        )
    )
    assert accumulator.as_dict() == {
        "nominal_visits": 12,
        "scheduled_visits": 11,
        "logical_leaves": 8,
        "orientation_rows": 12,
        "evaluator_calls": 3,
        "wall_time_sec": 0.375,
    }
    assert accumulator.information_regime == "authoritative_hidden_state"


def test_game_accumulator_refuses_mixed_information_regimes() -> None:
    accumulator = GameCounterAccumulator()
    common = dict(
        selected_action=1,
        policy={1: 1.0},
        q_values={},
        root_value=0.0,
        counters=SearchCounters(0, 0, 0, 0, 0, 0.0),
    )
    accumulator.add(
        MeasuredDecision(**common, information_regime="public_conservation_pimc")
    )
    with pytest.raises(ValueError, match="cannot mix"):
        accumulator.add(
            MeasuredDecision(**common, information_regime="authoritative_hidden_state")
        )
