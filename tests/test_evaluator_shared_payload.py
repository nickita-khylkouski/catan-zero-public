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
  2. Each evaluate() call site must call `_resolve_entity_adapter` at most once
     per non-cached, non-terminal request. On the native Rust feature path it
     must skip that Python adapter entirely after topology has been bootstrapped.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time

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


def _rust_with_native_features():
    """Require the task-#81 entity *and* context APIs.

    Older catanatron_rs wheels provide the MCTS game bindings (so `_rust()`
    succeeds) but not the native feature builders exercised by the warm-path
    tests below.  Skip those tests cleanly instead of failing with AttributeError.
    """
    catanatron_rs = _rust()
    required = (
        "EntityTopology",
        "build_entity_features_flat",
        "build_action_context_flat",
    )
    missing = [name for name in required if not hasattr(catanatron_rs, name)]
    if missing:
        pytest.skip(
            "catanatron_rs wheel lacks native entity/context features: "
            + ", ".join(missing)
        )
    return catanatron_rs


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


def _tiny_real_policy(*, public_observation: bool = False):
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
    # This is checkpoint metadata rather than an architectural choice.  Set it
    # to the requested regime so the evaluator's fail-closed guard is exercised
    # (and passes) in both parametrizations.
    policy.trained_with_masked_hidden_info = bool(public_observation)
    return policy


def _assert_eval_close(actual, expected, *, context: str = "") -> None:
    actual_priors, actual_value = actual
    expected_priors, expected_value = expected
    assert set(actual_priors) == set(expected_priors), (
        f"{context}: prior key mismatch "
        f"{set(actual_priors) ^ set(expected_priors)}"
    )
    for action in expected_priors:
        assert np.isclose(
            actual_priors[action], expected_priors[action], atol=1.0e-6, rtol=1.0e-5
        ), (
            f"{context}: action={action} prior {actual_priors[action]!r} != "
            f"{expected_priors[action]!r}"
        )
    assert np.isclose(actual_value, expected_value, atol=1.0e-6, rtol=1.0e-5), (
        f"{context}: value {actual_value!r} != {expected_value!r}"
    )


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


class _SnapshotForbiddenGame:
    """Delegate every native-game API except the one the warm path must skip."""

    def __init__(self, game):
        self.game = game
        self.action_indices_calls = 0
        self.actions_json_calls = 0

    def json_snapshot(self):
        raise AssertionError("json_snapshot must not be fetched on this path")

    def playable_action_indices(self, *args, **kwargs):
        self.action_indices_calls += 1
        return self.game.playable_action_indices(*args, **kwargs)

    def playable_actions_json(self, *args, **kwargs):
        self.actions_json_calls += 1
        return self.game.playable_actions_json(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.game, name)


class _StubPolicy:
    action_size = 8
    trained_with_masked_hidden_info = False

    def forward_legal_np(
        self,
        _entity,
        legal_ids,
        _context,
        *,
        return_q=False,
        return_final_vp=False,
    ):
        import torch

        del return_q, return_final_vp
        return {
            "logits": torch.zeros(tuple(legal_ids.shape), dtype=torch.float32),
            "value": torch.zeros((int(legal_ids.shape[0]),), dtype=torch.float32),
        }


class _StubGame:
    def current_color(self):
        return COLORS[0]

    def playable_action_indices(self, _colors, _filter):
        return [10, 11]

    def playable_actions_json(self):
        return json.dumps([[COLORS[0], "END_TURN"], [COLORS[0], "ROLL"]])


def _stub_native_features(evaluator, monkeypatch):
    monkeypatch.setattr(
        neural_rust_mcts,
        "rust_policy_action_ids",
        lambda _game, legal, **_kwargs: tuple(range(len(legal))),
    )
    monkeypatch.setattr(
        evaluator,
        "_entity_batch_via_rust",
        lambda *_args, **_kwargs: {"dummy": np.zeros((1, 1), dtype=np.float32)},
    )
    monkeypatch.setattr(
        evaluator,
        "_context_batch_via_rust",
        lambda *_args, **_kwargs: np.zeros((1, 2, 1), dtype=np.float32),
    )


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


def test_evaluate_many_skips_adapter_after_rust_topology_is_warm():
    catanatron_rs = _rust_with_native_features()
    game_a = _advance_to_multi_action_state(catanatron_rs, seed=21)
    game_b = _advance_to_multi_action_state(catanatron_rs, seed=23)
    legal_a = _legal_rust_actions(game_a)
    legal_b = _legal_rust_actions(game_b)
    actor = str(game_a.current_color())

    evaluator = EntityGraphRustEvaluator(
        _tiny_real_policy(),
        config=EntityGraphRustEvaluatorConfig(cache_size=0, rust_featurize=True),
    )
    # The first native evaluation resolves the Python adapter only to compute
    # fixed BASE-map topology. Subsequent native feature calls consume the
    # cached topology and must not parse/rebuild the adapter payload again.
    evaluator.evaluate(game_a, legal_a, root_color=actor, colors=COLORS)
    assert evaluator._rust_topology is not None

    with _ResolveCallCounter() as counter:
        evaluator.evaluate_many(
            [(game_a, legal_a), (game_b, legal_b)], root_color=actor, colors=COLORS
        )
    assert counter.calls == 0, (
        "evaluate_many() rebuilt the Python entity adapter after native topology "
        f"was warm: {counter.calls} call(s)"
    )


def test_batched_evaluator_skips_adapter_after_rust_topology_is_warm():
    catanatron_rs = _rust_with_native_features()
    game_a = _advance_to_multi_action_state(catanatron_rs, seed=31)
    game_b = _advance_to_multi_action_state(catanatron_rs, seed=33)
    legal_a = _legal_rust_actions(game_a)
    legal_b = _legal_rust_actions(game_b)
    actor = str(game_a.current_color())

    evaluator = BatchedEntityGraphRustEvaluator(
        _tiny_real_policy(),
        config=EntityGraphRustEvaluatorConfig(cache_size=0, rust_featurize=True),
        max_batch_size=64,
        max_wait_ms=0.0,
    )
    try:
        evaluator.evaluate(game_a, legal_a, root_color=actor, colors=COLORS)
        assert evaluator._rust_topology is not None
        with _ResolveCallCounter() as counter:
            evaluator.evaluate(game_b, legal_b, root_color=actor, colors=COLORS)
        assert counter.calls == 0, (
            "BatchedEntityGraphRustEvaluator rebuilt the Python entity adapter "
            f"after native topology was warm: {counter.calls} call(s)"
        )
    finally:
        evaluator.close()


@pytest.mark.parametrize("path", ["evaluate", "evaluate_many", "batched"])
def test_warm_native_cache_zero_skips_snapshot_but_keeps_action_map(
    path, monkeypatch
):
    """The production leaf path must not serialize an unused game snapshot."""
    game = _StubGame()
    legal = (10, 11)
    actor = str(game.current_color())
    config = EntityGraphRustEvaluatorConfig(cache_size=0, rust_featurize=True)
    evaluator = (
        BatchedEntityGraphRustEvaluator(
            _StubPolicy(), config=config, max_batch_size=64, max_wait_ms=0.0
        )
        if path == "batched"
        else EntityGraphRustEvaluator(_StubPolicy(), config=config)
    )
    try:
        # This is the post-bootstrap production state; native builders receive
        # this immutable object and do not need the Python snapshot adapter.
        evaluator._rust_topology = object()
        _stub_native_features(evaluator, monkeypatch)
        guarded = _SnapshotForbiddenGame(game)

        if path == "evaluate_many":
            evaluator.evaluate_many(
                [(guarded, legal)], root_color=actor, colors=COLORS
            )
        else:
            evaluator.evaluate(guarded, legal, root_color=actor, colors=COLORS)

        assert guarded.action_indices_calls == 1
        assert guarded.actions_json_calls == 1
    finally:
        if isinstance(evaluator, BatchedEntityGraphRustEvaluator):
            evaluator.close()


@pytest.mark.parametrize("path", ["cold_native", "cache_enabled", "non_rust"])
def test_snapshot_is_still_required_on_fallback_paths(path):
    """Cold topology, cache keys, and Python features retain prior behavior."""
    game = _StubGame()
    legal = (10, 11)
    actor = str(game.current_color())
    config = EntityGraphRustEvaluatorConfig(
        cache_size=1 if path == "cache_enabled" else 0,
        rust_featurize=path != "non_rust",
    )
    evaluator = EntityGraphRustEvaluator(_StubPolicy(), config=config)
    if path == "cache_enabled":
        evaluator._rust_topology = object()

    with pytest.raises(
        AssertionError, match="json_snapshot must not be fetched on this path"
    ):
        evaluator.evaluate(
            _SnapshotForbiddenGame(game),
            legal,
            root_color=actor,
            colors=COLORS,
        )


@pytest.mark.parametrize("public_observation", [False, True])
def test_warm_rust_evaluate_many_matches_individual_evaluations(public_observation):
    """The adapter-free warm path preserves outputs in both input regimes."""
    catanatron_rs = _rust_with_native_features()
    policy = _tiny_real_policy(public_observation=public_observation)
    config = EntityGraphRustEvaluatorConfig(
        cache_size=0,
        public_observation=public_observation,
        rust_featurize=True,
    )
    singles = EntityGraphRustEvaluator(policy, config=config)
    many = EntityGraphRustEvaluator(policy, config=config)

    warm_game = _advance_to_multi_action_state(catanatron_rs, seed=40)
    warm_legal = _legal_rust_actions(warm_game)
    warm_actor = str(warm_game.current_color())
    singles.evaluate(warm_game.copy(), warm_legal, root_color=warm_actor, colors=COLORS)
    many.evaluate(warm_game.copy(), warm_legal, root_color=warm_actor, colors=COLORS)
    assert singles._rust_topology is not None
    assert many._rust_topology is not None

    games = [
        _advance_to_multi_action_state(catanatron_rs, seed=seed)
        for seed in (41, 42, 43, 44)
    ]
    legal = [_legal_rust_actions(game) for game in games]
    root_color = COLORS[0]
    expected = [
        singles.evaluate(game.copy(), actions, root_color=root_color, colors=COLORS)
        for game, actions in zip(games, legal)
    ]
    with _ResolveCallCounter() as counter:
        actual = many.evaluate_many(
            [(game.copy(), actions) for game, actions in zip(games, legal)],
            root_color=root_color,
            colors=COLORS,
        )
    assert counter.calls == 0
    for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
        _assert_eval_close(
            actual_item,
            expected_item,
            context=f"public_observation={public_observation} request={index}",
        )


@pytest.mark.parametrize("public_observation", [False, True])
def test_warm_concurrent_batched_evaluation_matches_singles(
    public_observation, monkeypatch
):
    """Concurrent EvalServer-style batching matches one-at-a-time inference."""
    catanatron_rs = _rust_with_native_features()
    policy = _tiny_real_policy(public_observation=public_observation)
    config = EntityGraphRustEvaluatorConfig(
        cache_size=0,
        public_observation=public_observation,
        rust_featurize=True,
    )
    singles = EntityGraphRustEvaluator(policy, config=config)
    batched = BatchedEntityGraphRustEvaluator(
        policy,
        config=config,
        max_batch_size=8,
        max_wait_ms=250.0,
    )
    try:
        warm_game = _advance_to_multi_action_state(catanatron_rs, seed=50)
        warm_legal = _legal_rust_actions(warm_game)
        warm_actor = str(warm_game.current_color())
        singles.evaluate(warm_game.copy(), warm_legal, root_color=warm_actor, colors=COLORS)
        batched.evaluate(warm_game.copy(), warm_legal, root_color=warm_actor, colors=COLORS)
        assert singles._rust_topology is not None
        assert batched._rust_topology is not None

        games = [
            _advance_to_multi_action_state(catanatron_rs, seed=seed)
            for seed in (51, 52, 53, 54)
        ]
        legal = [_legal_rust_actions(game) for game in games]
        root_color = COLORS[0]
        expected = [
            singles.evaluate(game.copy(), actions, root_color=root_color, colors=COLORS)
            for game, actions in zip(games, legal)
        ]

        observed_batch_sizes: list[int] = []
        real_forward = policy.forward_legal_np

        def tracked_forward(
            entity_batch,
            legal_ids,
            context,
            *,
            return_q=False,
            return_final_vp=False,
        ):
            observed_batch_sizes.append(int(legal_ids.shape[0]))
            return real_forward(
                entity_batch,
                legal_ids,
                context,
                return_q=return_q,
                return_final_vp=return_final_vp,
            )

        monkeypatch.setattr(policy, "forward_legal_np", tracked_forward)
        # Make the collector wait for this deliberately concurrent burst.  The
        # production flag becomes true after the first observed multi-request
        # batch; setting it here removes scheduler luck from this regression.
        batched._observed_concurrency = True
        start = threading.Barrier(len(games))

        def evaluate_one(index: int):
            start.wait(timeout=10.0)
            return batched.evaluate(
                games[index].copy(),
                legal[index],
                root_color=root_color,
                colors=COLORS,
            )

        with _ResolveCallCounter() as counter:
            with ThreadPoolExecutor(max_workers=len(games)) as executor:
                futures = [executor.submit(evaluate_one, index) for index in range(len(games))]
                actual = [future.result(timeout=30.0) for future in futures]
        assert counter.calls == 0
        assert any(size > 1 for size in observed_batch_sizes), observed_batch_sizes
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _assert_eval_close(
                actual_item,
                expected_item,
                context=f"public_observation={public_observation} request={index}",
            )
    finally:
        batched.close()


def test_concurrent_cold_start_computes_rust_topology_once(monkeypatch):
    """Two producer threads cannot both initialize evaluator topology."""
    catanatron_rs = _rust_with_native_features()
    import catan_zero.rl.entity_token_features_rust as rust_features

    policy = _tiny_real_policy()
    evaluator = BatchedEntityGraphRustEvaluator(
        policy,
        config=EntityGraphRustEvaluatorConfig(cache_size=0, rust_featurize=True),
        max_batch_size=2,
        max_wait_ms=250.0,
    )
    real_compute = rust_features.compute_rust_topology
    count_lock = threading.Lock()
    compute_calls = 0

    def slow_compute(adapter, acting_color):
        nonlocal compute_calls
        with count_lock:
            compute_calls += 1
        # Keep the first computation open long enough for the second producer
        # to reach the cold check.  Without the evaluator lock this reliably
        # produces two calls (the reviewer-reported race).
        time.sleep(0.1)
        return real_compute(adapter, acting_color)

    monkeypatch.setattr(rust_features, "compute_rust_topology", slow_compute)
    games = [
        _advance_to_multi_action_state(catanatron_rs, seed=seed)
        for seed in (61, 62)
    ]
    legal = [_legal_rust_actions(game) for game in games]
    start = threading.Barrier(2)
    evaluator._observed_concurrency = True

    def evaluate_one(index: int):
        start.wait(timeout=10.0)
        return evaluator.evaluate(
            games[index], legal[index], root_color=COLORS[0], colors=COLORS
        )

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(evaluate_one, index) for index in range(2)]
            for future in futures:
                future.result(timeout=30.0)
        assert compute_calls == 1
        assert evaluator._rust_topology is not None
    finally:
        evaluator.close()


@pytest.mark.parametrize("public_observation", [False, True])
def test_warm_rust_symmetry_evaluation_skips_adapter_without_output_change(
    public_observation,
):
    catanatron_rs = _rust_with_native_features()
    policy = _tiny_real_policy(public_observation=public_observation)
    config = EntityGraphRustEvaluatorConfig(
        cache_size=0,
        public_observation=public_observation,
        rust_featurize=True,
    )
    cold = EntityGraphRustEvaluator(policy, config=config)
    warm = EntityGraphRustEvaluator(policy, config=config)
    game = _advance_to_multi_action_state(catanatron_rs, seed=71, min_legal=3)
    legal = _legal_rust_actions(game)
    actor = str(game.current_color())

    expected = cold.evaluate_symmetry_averaged(
        game.copy(), legal, root_color=actor, colors=COLORS
    )
    warm.evaluate(game.copy(), legal, root_color=actor, colors=COLORS)
    assert warm._rust_topology is not None
    with _ResolveCallCounter() as counter:
        actual = warm.evaluate_symmetry_averaged(
            game.copy(), legal, root_color=actor, colors=COLORS
        )
    assert counter.calls == 0
    _assert_eval_close(
        actual,
        expected,
        context=f"public_observation={public_observation} warm symmetry",
    )
