from __future__ import annotations

import json
import threading
from collections import OrderedDict

import numpy as np
import pytest
import torch

import catan_zero.search.neural_rust_mcts as neural_rust_mcts
from catan_zero.search.neural_rust_mcts import (
    BatchedEntityGraphRustEvaluator,
    EntityGraphRustEvaluator,
    EntityGraphRustEvaluatorConfig,
)


_COLORS = ("RED", "BLUE")
_LEGAL = (10, 20)


class _FakeGame:
    def __init__(self, state: int = 1) -> None:
        self.state = int(state)

    def current_color(self) -> str:
        return "RED"

    def json_snapshot(self) -> str:
        return json.dumps(
            {"current_color": self.current_color(), "state": self.state},
            sort_keys=True,
        )

    def playable_action_indices(self, _colors, _map_kind):
        return list(_LEGAL)

    def playable_actions_json(self) -> str:
        return json.dumps([{"action": action} for action in _LEGAL])

    def winning_color(self):
        return None


class _FakePolicy:
    action_size = 100
    trained_with_masked_hidden_info = False
    supports_final_vp_selection = False
    trained_value_readouts = ("scalar",)

    def __init__(self) -> None:
        self.forward_calls = 0

    def forward_legal_np(
        self,
        _entity_batch,
        legal_action_ids,
        _legal_action_context,
        *,
        return_q=False,
    ):
        del return_q
        self.forward_calls += 1
        rows, width = legal_action_ids.shape
        logits = torch.arange(width, dtype=torch.float32).repeat(rows, 1)
        return {
            "logits": logits,
            "value": torch.full((rows,), 0.5, dtype=torch.float32),
        }


@pytest.fixture
def fake_feature_boundary(monkeypatch):
    monkeypatch.setattr(
        neural_rust_mcts,
        "rust_policy_action_ids",
        lambda _game, legal_actions, **_kwargs: tuple(legal_actions),
    )
    monkeypatch.setattr(
        neural_rust_mcts,
        "rust_game_to_entity_batch",
        lambda *_args, **_kwargs: {"dummy": np.zeros((1, 1), dtype=np.float32)},
    )
    monkeypatch.setattr(
        neural_rust_mcts,
        "rust_action_context_batch",
        lambda *_args, **_kwargs: np.zeros((1, len(_LEGAL), 1), dtype=np.float32),
    )
    monkeypatch.setattr(
        neural_rust_mcts,
        "_resolve_entity_adapter",
        lambda *_args, **_kwargs: ({}, object(), []),
    )


def _evaluate_path(evaluator, path: str, game: _FakeGame, root_color: str):
    if path == "many":
        return evaluator.evaluate_many(
            [(game, _LEGAL)], root_color=root_color, colors=_COLORS
        )[0]
    return evaluator.evaluate(
        game, _LEGAL, root_color=root_color, colors=_COLORS
    )


@pytest.mark.parametrize("path", ["sync", "many", "async"])
def test_cache_separates_root_value_perspectives(fake_feature_boundary, path: str) -> None:
    policy = _FakePolicy()
    config = EntityGraphRustEvaluatorConfig(cache_size=2)
    if path == "async":
        evaluator = BatchedEntityGraphRustEvaluator(
            policy, config=config, max_batch_size=4, max_wait_ms=0.0
        )
    else:
        evaluator = EntityGraphRustEvaluator(policy, config=config)

    try:
        game = _FakeGame()
        red_first = _evaluate_path(evaluator, path, game, "RED")
        blue = _evaluate_path(evaluator, path, game, "BLUE")
        red_cached = _evaluate_path(evaluator, path, game, "RED")

        assert policy.forward_calls == 2
        assert red_cached == red_first
        assert blue[0] == red_first[0]
        assert blue[1] == pytest.approx(-red_first[1], abs=0.0)
        assert {key[1] for key in evaluator._cache} == {"RED", "BLUE"}
    finally:
        if isinstance(evaluator, BatchedEntityGraphRustEvaluator):
            evaluator.close()


class _PausedHitCache(OrderedDict):
    """Pause a cache hit between OrderedDict.get and its LRU touch."""

    def __init__(self, entries) -> None:
        super().__init__(entries)
        self.hit_observed = threading.Event()
        self.release_hit = threading.Event()

    def get(self, key, default=None):
        cached = super().get(key, default)
        if threading.current_thread().name == "cache-hit" and cached is not None:
            self.hit_observed.set()
            if not self.release_hit.wait(timeout=5.0):
                raise TimeoutError("timed out waiting to release paused cache hit")
        return cached


def test_cache_hit_touch_is_atomic_against_concurrent_capacity_eviction() -> None:
    evaluator = BatchedEntityGraphRustEvaluator(
        _FakePolicy(),
        config=EntityGraphRustEvaluatorConfig(cache_size=1),
        max_batch_size=2,
        max_wait_ms=0.0,
    )
    key_a = ("state-a", "RED", _COLORS, _LEGAL)
    key_b = ("state-b", "RED", _COLORS, _LEGAL)
    priors = {10: 0.25, 20: 0.75}
    evaluator._cache_store(key_a, priors, 0.5, 0.0)
    paused_cache = _PausedHitCache(evaluator._cache)
    evaluator._cache = paused_cache

    hit_results = []
    errors = []
    store_started = threading.Event()
    store_done = threading.Event()

    def cache_hit() -> None:
        try:
            hit_results.append(evaluator._cache_get(key_a))
        except BaseException as error:  # preserve the worker exception for assertion
            errors.append(error)

    def capacity_store() -> None:
        store_started.set()
        try:
            evaluator._cache_store(key_b, priors, -0.5, 0.0)
        except BaseException as error:  # preserve the worker exception for assertion
            errors.append(error)
        finally:
            store_done.set()

    hit_thread = threading.Thread(target=cache_hit, name="cache-hit")
    store_thread = threading.Thread(target=capacity_store, name="capacity-store")
    try:
        hit_thread.start()
        assert paused_cache.hit_observed.wait(timeout=5.0)
        store_thread.start()
        assert store_started.wait(timeout=5.0)

        # The hit owns the shared RLock until its move_to_end completes. A
        # capacity-one store therefore cannot evict key_a in this interval.
        assert not store_done.wait(timeout=0.1)
        paused_cache.release_hit.set()
        hit_thread.join(timeout=5.0)
        store_thread.join(timeout=5.0)

        assert not hit_thread.is_alive()
        assert not store_thread.is_alive()
        assert errors == []
        assert hit_results == [(priors, 0.5)]
        assert list(evaluator._cache) == [key_b]
    finally:
        paused_cache.release_hit.set()
        hit_thread.join(timeout=5.0)
        store_thread.join(timeout=5.0)
        evaluator.close()
