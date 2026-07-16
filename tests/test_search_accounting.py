from __future__ import annotations

import threading

import pytest

from catan_zero.search.accounting import (
    SearchAccountingEvaluator,
    SearchWork,
)


class _Evaluator:
    marker = "delegated"

    def evaluate(self, game, legal_actions, *, root_color, colors):
        return {action: 1 / len(legal_actions) for action in legal_actions}, 0.25

    def evaluate_many(self, requests, *, root_color, colors):
        return [self.evaluate(game, legal, root_color=root_color, colors=colors) for game, legal in requests]

    def evaluate_symmetry_averaged(self, game, legal_actions, *, root_color, colors):
        return self.evaluate(game, legal_actions, root_color=root_color, colors=colors)


def test_counts_plain_batch_and_symmetry_without_changing_outputs() -> None:
    wrapped = SearchAccountingEvaluator(_Evaluator())
    before = wrapped.snapshot()
    plain = wrapped.evaluate(None, (1, 2), root_color="RED", colors=("RED", "BLUE"))
    batch = wrapped.evaluate_many(
        [(None, (1,)), (None, (2, 3)), (None, (4,))],
        root_color="RED",
        colors=("RED", "BLUE"),
    )
    symmetric = wrapped.evaluate_symmetry_averaged(
        None, (9,), root_color="RED", colors=("RED", "BLUE")
    )

    assert plain == ({1: 0.5, 2: 0.5}, 0.25)
    assert len(batch) == 3
    assert symmetric == ({9: 1.0}, 0.25)
    assert wrapped.snapshot() - before == SearchWork(
        evaluator_calls=3,
        logical_leaf_evaluations=5,
        orientation_evaluation_rows=16,
    )
    assert wrapped.marker == "delegated"


def test_scope_records_only_work_inside_scope() -> None:
    wrapped = SearchAccountingEvaluator(_Evaluator())
    wrapped.evaluate(None, (1,), root_color="RED", colors=("RED", "BLUE"))
    with wrapped.scope() as scope:
        wrapped.evaluate_symmetry_averaged(
            None, (1,), root_color="RED", colors=("RED", "BLUE")
        )
    assert scope.require_work() == SearchWork(1, 1, 12)


def test_terminal_empty_legal_requests_are_not_counted_as_neural_rows() -> None:
    class TerminalAwareEvaluator:
        def evaluate(self, _game, legal_actions, **_kwargs):
            return ({}, 0.0) if not legal_actions else ({legal_actions[0]: 1.0}, 0.0)

        def evaluate_many(self, requests, **_kwargs):
            return [self.evaluate(game, legal) for game, legal in requests]

        def evaluate_symmetry_averaged(self, _game, legal_actions, **_kwargs):
            return ({}, 0.0) if not legal_actions else ({legal_actions[0]: 1.0}, 0.0)

    wrapped = SearchAccountingEvaluator(TerminalAwareEvaluator())
    wrapped.evaluate(None, (), root_color="RED", colors=("RED", "BLUE"))
    wrapped.evaluate_many(
        [(None, ()), (None, (7,))], root_color="RED", colors=("RED", "BLUE")
    )
    wrapped.evaluate_symmetry_averaged(
        None, (), root_color="RED", colors=("RED", "BLUE")
    )
    assert wrapped.snapshot() == SearchWork(
        evaluator_calls=3,
        logical_leaf_evaluations=1,
        orientation_evaluation_rows=1,
    )


def test_capability_lookup_matches_wrapped_evaluator() -> None:
    class EvaluateOnly:
        def evaluate(self, *args, **kwargs):
            return {}, 0.0

    wrapped = SearchAccountingEvaluator(EvaluateOnly())
    assert hasattr(wrapped, "evaluate")
    assert not hasattr(wrapped, "evaluate_many")


def test_counter_is_thread_safe() -> None:
    wrapped = SearchAccountingEvaluator(_Evaluator())

    def run() -> None:
        for _ in range(100):
            wrapped.evaluate(None, (1,), root_color="RED", colors=("RED", "BLUE"))

    threads = [threading.Thread(target=run) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert wrapped.snapshot() == SearchWork(400, 400, 400)


def test_invalid_orientation_count_and_snapshot_order_fail_closed() -> None:
    with pytest.raises(ValueError, match="positive"):
        SearchAccountingEvaluator(_Evaluator(), symmetry_orientations=0)
    with pytest.raises(ValueError, match="not monotonic"):
        _ = SearchWork() - SearchWork(1, 1, 1)


def test_search_result_telemetry_is_append_only_and_non_semantic() -> None:
    from catan_zero.search.gumbel_chance_mcts import SearchResult

    common = dict(
        selected_action=1,
        improved_policy={1: 1.0},
        visit_counts={1: 1},
        q_values={1: 0.25},
        priors={1: 1.0},
        root_value=0.25,
        used_full_search=True,
        simulations_used=1,
    )
    baseline = SearchResult(**common)
    measured = SearchResult(
        **common,
        evaluator_method_calls=3,
        logical_leaf_evaluations=5,
        orientation_evaluation_rows=16,
        neural_evaluation_rows=16,
        unique_leaf_expansions=5,
        unique_boundary_expansions=1,
        wall_time_sec=0.5,
    )
    assert baseline == measured
    assert measured.neural_evaluation_rows == 16


def test_neural_row_attestation_fails_closed_when_cache_can_hide_forwards() -> None:
    from types import SimpleNamespace

    from catan_zero.search.gumbel_chance_mcts import GumbelChanceMCTS

    class NeuralLike:
        policy = object()

        def __init__(self, cache_size: int) -> None:
            self.config = SimpleNamespace(cache_size=cache_size)

        def evaluate(self, *_args, **_kwargs):
            return {}, 0.0

    mcts = object.__new__(GumbelChanceMCTS)
    work = SearchWork(3, 5, 16)
    mcts.evaluator = SearchAccountingEvaluator(NeuralLike(cache_size=0))
    assert mcts._exact_neural_rows(work) == 16
    mcts.evaluator = SearchAccountingEvaluator(NeuralLike(cache_size=1))
    assert mcts._exact_neural_rows(work) is None
