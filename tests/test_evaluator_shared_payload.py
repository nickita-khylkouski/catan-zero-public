"""Regression tests for the shared-payload dedup fix (independent-audit
follow-up): inside one `evaluate()`/`evaluate_many()`/
`evaluate_symmetry_averaged()` call, `rust_game_to_entity_batch` and
`rust_action_context_batch` used to each invoke `_resolve_entity_adapter`
independently -- re-fetching the json snapshot, re-building the players
payload, and re-running the masking gate -- roughly doubling the
featurization tax (~42% of per-leaf cost). The fix builds the resolved
(payload, adapter, structured) tuple ONCE per evaluate() call and threads it
into both consumers via an optional `resolved=` parameter.

Two guarantees, enforced together:
  1. Passing a pre-built `resolved` tuple into `rust_game_to_entity_batch`/
     `rust_action_context_batch` must produce bit-identical output to each
     function resolving independently -- in BOTH masking regimes (this is
     the correctness bar: the shared tuple must be regime-correct for both
     consumers, not just the entity-token path).
  2. Each of the four evaluate() call sites (`EntityGraphRustEvaluator.evaluate`,
     `.evaluate_many`, `.evaluate_symmetry_averaged`,
     `BatchedEntityGraphRustEvaluator.evaluate`) must call
     `_resolve_entity_adapter` exactly once per non-cached, non-terminal
     request -- not twice.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from catan_zero.rl.action_mask import ActionCatalog
from catan_zero.rl.gumbel_self_play import COLORS
import catan_zero.search.neural_rust_mcts as neural_rust_mcts
from catan_zero.search.neural_rust_mcts import (
    BatchedEntityGraphRustEvaluator,
    EntityGraphRustEvaluator,
    EntityGraphRustEvaluatorConfig,
    _resolve_entity_adapter,
    rust_action_context_batch,
    rust_game_to_entity_batch,
    rust_policy_action_ids,
)
from catan_zero.search.rust_mcts import _require_rust_module

ACTION_SIZE = ActionCatalog(COLORS).size


def _rust():
    try:
        return _require_rust_module()
    except RuntimeError as error:
        pytest.skip(str(error))


def _advance_to_multi_action_state(catanatron_rs, *, seed: int, min_legal: int = 2):
    game = catanatron_rs.Game.simple(list(COLORS), seed=seed)
    for _ in range(300):
        game.play_tick()
        if game.winning_color() is not None:
            break
        playable = json.loads(game.playable_actions_json())
        if len(playable) >= min_legal:
            return game
    raise AssertionError(f"did not reach a state with >= {min_legal} legal actions")


def _legal_rust_actions(game) -> tuple[int, ...]:
    return tuple(int(action) for action in game.playable_action_indices(list(COLORS), None))


def _tiny_real_policy():
    """Real (not mocked) but tiny policy with the actual action-catalog size,
    so `rust_policy_action_ids`/forward passes are valid -- same fixture
    pattern as tests/test_regime_fail_closed.py's
    test_logits_invariant_to_opponent_hand_permutation_when_masked."""
    from catan_zero.rl.entity_token_policy import EntityGraphPolicy
    from catan_zero.rl.self_play import make_env_config

    policy = EntityGraphPolicy.create(
        env_config=make_env_config(vps_to_win=3),
        hidden_size=16,
        state_layers=1,
        attention_heads=2,
        seed=0,
    )
    policy.model.eval()  # create() leaves train mode; active Dropout would break equality/counting.
    return policy


class _ResolveCallCounter:
    """Monkeypatches the module-global `_resolve_entity_adapter` and counts
    calls, delegating to the real implementation. Same monkeypatch-a-module-
    global pattern already used by
    tests/test_regime_fail_closed.py::test_rust_action_context_batch_applies_masking_gate_when_configured."""

    def __init__(self):
        self.calls = 0
        self._real = neural_rust_mcts._resolve_entity_adapter

    def __enter__(self):
        def _spy(*args, **kwargs):
            self.calls += 1
            return self._real(*args, **kwargs)

        neural_rust_mcts._resolve_entity_adapter = _spy
        return self

    def __exit__(self, *_exc):
        neural_rust_mcts._resolve_entity_adapter = self._real


# ---------------------------------------------------------------------------
# 1: shared `resolved` tuple must be bit-identical to independent resolution,
# in both masking regimes.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("public_observation", [False, True])
def test_shared_resolved_tuple_matches_independent_calls(public_observation):
    catanatron_rs = _rust()
    for seed in (1, 2, 3):
        game = _advance_to_multi_action_state(catanatron_rs, seed=seed)
        legal_rust = _legal_rust_actions(game)
        actor = str(game.current_color())
        mapped = rust_policy_action_ids(game, legal_rust, colors=COLORS, action_size=ACTION_SIZE)

        entity_independent = rust_game_to_entity_batch(
            game, legal_rust, actor=actor, colors=COLORS, action_size=ACTION_SIZE,
            policy_action_ids=mapped, public_observation=public_observation,
        )
        context_independent = rust_action_context_batch(
            game, legal_rust, actor=actor, colors=COLORS, action_size=ACTION_SIZE,
            policy_action_ids=mapped, public_observation=public_observation,
        )

        resolved = _resolve_entity_adapter(
            game, legal_rust, colors=COLORS, action_size=ACTION_SIZE,
            policy_action_ids=mapped, snapshot=None, action_by_id=None,
            public_observation=public_observation, perspective=actor,
        )
        entity_shared = rust_game_to_entity_batch(
            game, legal_rust, actor=actor, colors=COLORS, action_size=ACTION_SIZE,
            policy_action_ids=mapped, public_observation=public_observation,
            resolved=resolved,
        )
        context_shared = rust_action_context_batch(
            game, legal_rust, actor=actor, colors=COLORS, action_size=ACTION_SIZE,
            policy_action_ids=mapped, public_observation=public_observation,
            resolved=resolved,
        )

        assert set(entity_independent) == set(entity_shared)
        for key in entity_independent:
            assert np.array_equal(entity_independent[key], entity_shared[key]), (
                f"seed={seed} public_observation={public_observation}: "
                f"entity feature {key!r} differs between shared and independent resolution"
            )
        assert np.array_equal(context_independent, context_shared), (
            f"seed={seed} public_observation={public_observation}: "
            "context batch differs between shared and independent resolution"
        )


# ---------------------------------------------------------------------------
# 2: each evaluate() site builds the resolved adapter exactly once per
# non-cached, non-terminal request.
# ---------------------------------------------------------------------------


def test_evaluate_builds_resolved_adapter_exactly_once():
    catanatron_rs = _rust()
    game = _advance_to_multi_action_state(catanatron_rs, seed=11)
    legal_rust = _legal_rust_actions(game)
    actor = str(game.current_color())

    evaluator = EntityGraphRustEvaluator(
        _tiny_real_policy(), config=EntityGraphRustEvaluatorConfig(cache_size=0)
    )
    with _ResolveCallCounter() as counter:
        evaluator.evaluate(game, legal_rust, root_color=actor, colors=COLORS)
    assert counter.calls == 1, (
        f"evaluate() must build the resolved adapter tuple exactly once "
        f"(shared between entity+context batches), got {counter.calls} calls"
    )


def test_evaluate_many_builds_resolved_once_per_pending_request():
    catanatron_rs = _rust()
    game_a = _advance_to_multi_action_state(catanatron_rs, seed=11)
    game_b = _advance_to_multi_action_state(catanatron_rs, seed=13)
    legal_a = _legal_rust_actions(game_a)
    legal_b = _legal_rust_actions(game_b)
    actor = str(game_a.current_color())

    evaluator = EntityGraphRustEvaluator(
        _tiny_real_policy(), config=EntityGraphRustEvaluatorConfig(cache_size=0)
    )
    with _ResolveCallCounter() as counter:
        evaluator.evaluate_many(
            [(game_a, legal_a), (game_b, legal_b)], root_color=actor, colors=COLORS
        )
    assert counter.calls == 2, (
        f"evaluate_many() must build one resolved adapter tuple per pending "
        f"request (2 requests -> 2 calls), got {counter.calls} calls"
    )


def test_evaluate_symmetry_averaged_builds_resolved_once():
    catanatron_rs = _rust()
    game = _advance_to_multi_action_state(catanatron_rs, seed=11)
    legal_rust = _legal_rust_actions(game)
    actor = str(game.current_color())

    evaluator = EntityGraphRustEvaluator(
        _tiny_real_policy(), config=EntityGraphRustEvaluatorConfig(cache_size=0)
    )
    with _ResolveCallCounter() as counter:
        evaluator.evaluate_symmetry_averaged(game, legal_rust, root_color=actor, colors=COLORS)
    assert counter.calls == 1, (
        f"evaluate_symmetry_averaged() must build the resolved adapter tuple "
        f"exactly once, got {counter.calls} calls"
    )


def test_batched_evaluator_evaluate_builds_resolved_once():
    catanatron_rs = _rust()
    game = _advance_to_multi_action_state(catanatron_rs, seed=11)
    legal_rust = _legal_rust_actions(game)
    actor = str(game.current_color())

    evaluator = BatchedEntityGraphRustEvaluator(
        _tiny_real_policy(),
        config=EntityGraphRustEvaluatorConfig(cache_size=0),
        max_batch_size=64,
        max_wait_ms=3.0,
    )
    try:
        with _ResolveCallCounter() as counter:
            evaluator.evaluate(game, legal_rust, root_color=actor, colors=COLORS)
        assert counter.calls == 1, (
            f"BatchedEntityGraphRustEvaluator.evaluate() must build the "
            f"resolved adapter tuple exactly once, got {counter.calls} calls"
        )
    finally:
        evaluator.close()
