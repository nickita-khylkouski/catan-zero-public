"""CAT-126 #15: evaluator cache uses LRU (OrderedDict) eviction, not FIFO.

Bit-identical: LRU changes only WHICH entry is evicted, never the value a hit
returns (cache stores a deterministic (priors, value[, unc]) tuple keyed by full
state). A full evaluate() cache-hit test needs the 35M model (integration); here
we assert (1) the real evaluator inits an OrderedDict, (2) the exact LRU ops the
code uses give LRU order, (3) no FIFO regression in the source.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np

from catan_zero.rl.entity_token_policy import EntityGraphConfig, EntityGraphPolicy
from catan_zero.search.neural_rust_mcts import (
    EntityGraphRustEvaluator,
    EntityGraphRustEvaluatorConfig,
)
import catan_zero.search.neural_rust_mcts as nrm


def _tiny_evaluator(cache_size: int) -> EntityGraphRustEvaluator:
    cfg = EntityGraphConfig(
        action_size=8, static_action_feature_size=4,
        hidden_size=16, state_layers=1, attention_heads=2, dropout=0.0,
    )
    static = np.zeros((8, 4), dtype=np.float32)
    policy = EntityGraphPolicy(cfg, static, device="cpu")  # fresh -> not-masked
    # default public_observation (unmasked) matches the fresh policy -> guard passes
    return EntityGraphRustEvaluator(policy, config=EntityGraphRustEvaluatorConfig(cache_size=cache_size))


def test_real_evaluator_cache_is_ordereddict():
    ev = _tiny_evaluator(100)
    assert isinstance(ev._cache, OrderedDict)


def test_lru_eviction_and_touch_semantics():
    """Replicates the code's exact ops: move_to_end on hit, popitem(last=False)
    on evict at capacity. Asserts LRU (touched survives, least-recent evicted)."""
    cap = 2
    cache: "OrderedDict[str, int]" = OrderedDict()

    def insert(k, v):  # mirrors the eviction snippet
        if len(cache) >= cap:
            cache.popitem(last=False)
        cache[k] = v

    def hit(k):  # mirrors the move_to_end-on-hit snippet
        if k in cache:
            cache.move_to_end(k)
            return cache[k]
        return None

    insert("A", 1)
    insert("B", 2)                 # cache: A,B
    assert hit("A") == 1           # touch A -> order B,A
    insert("C", 3)                 # evict LRU=B (NOT A), cache: A,C
    assert list(cache.keys()) == ["A", "C"]
    assert hit("B") is None        # B was evicted
    assert hit("A") == 1           # A survived because it was touched
    # FIFO would have evicted A (oldest-inserted) instead of B — regression check.


def test_no_fifo_regression_in_source():
    src = Path(nrm.__file__).read_text()
    assert "pop(next(iter(self._cache)))" not in src   # old FIFO gone
    # Cache mutation is centralized so sync, evaluate_many, and async paths
    # cannot diverge on locking or eviction semantics.
    assert "def _cache_get(" in src
    assert "def _cache_store(" in src
    assert src.count("self._cache.popitem(last=False)") == 1
    assert src.count("self._cache.move_to_end(cache_key)") == 2
