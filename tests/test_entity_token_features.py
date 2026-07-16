from __future__ import annotations

import numpy as np
import pytest

from catan_zero.rl.entity_token_features import (
    EDGE_FEATURE_SIZE,
    EVENT_FEATURE_SIZE,
    GLOBAL_FEATURE_SIZE,
    HEX_FEATURE_SIZE,
    LEGAL_ACTION_FEATURE_SIZE,
    PLAYER_FEATURE_SIZE,
    VERTEX_FEATURE_SIZE,
    _node_pips_by_resource,
    _scale,
    _topology,
    build_entity_token_features,
)
from catan_zero.rl.multiagent_env import ColonistMultiAgentConfig, ColonistMultiAgentEnv


def _scale_via_np_clip(value, denominator: float) -> float:
    """Reference implementation matching `_scale`'s pre-optimization body
    (`np.clip` on a single scalar) -- kept only in this test so `_scale`'s
    faster plain-Python clamp can be checked against it."""
    from catan_zero.rl.entity_token_features import _safe_int

    parsed = _safe_int(value, default=0)
    if parsed is None:
        parsed = 0
    return float(np.clip(float(parsed) / float(max(denominator, 1.0)), 0.0, 1.0))


def test_entity_token_features_base_shapes_and_masks():
    env = ColonistMultiAgentEnv(ColonistMultiAgentConfig(players=2, vps_to_win=10))
    _observations, _info = env.reset(seed=123)

    features = build_entity_token_features(env, env.current_player_name())

    assert features["hex_tokens"].shape == (19, HEX_FEATURE_SIZE)
    assert features["hex_vertex_ids"].shape == (19, 6)
    assert features["hex_edge_ids"].shape == (19, 6)
    assert features["vertex_tokens"].shape == (54, VERTEX_FEATURE_SIZE)
    assert features["edge_tokens"].shape == (72, EDGE_FEATURE_SIZE)
    assert features["edge_vertex_ids"].shape == (72, 2)
    assert features["player_tokens"].shape == (4, PLAYER_FEATURE_SIZE)
    assert features["global_tokens"].shape == (1, GLOBAL_FEATURE_SIZE)
    assert features["legal_action_tokens"].shape[1] == LEGAL_ACTION_FEATURE_SIZE
    assert features["legal_action_target_ids"].shape == (
        features["legal_action_tokens"].shape[0],
        4,
    )
    assert features["event_tokens"].shape == (64, EVENT_FEATURE_SIZE)
    assert features["event_target_ids"].shape == (64, 4)

    assert features["hex_mask"].sum() == 19
    assert features["vertex_mask"].sum() == 54
    assert features["edge_mask"].sum() == 72
    assert features["player_mask"].sum() == 2
    assert features["legal_action_mask"].sum() == features["legal_action_tokens"].shape[0]
    assert features["event_mask"].sum() >= 1


def test_entity_token_features_only_actor_private_resources_visible():
    env = ColonistMultiAgentEnv(ColonistMultiAgentConfig(players=2, vps_to_win=10))
    _observations, _info = env.reset(seed=456)

    actor = env.current_player_name()
    features = build_entity_token_features(env, actor)
    player_tokens = features["player_tokens"]

    actor_rows = np.flatnonzero(player_tokens[:, 1] == 1.0)
    opponent_rows = np.flatnonzero((player_tokens[:, 0] == 1.0) & (player_tokens[:, 1] == 0.0))

    assert actor_rows.size == 1
    assert opponent_rows.size == 1
    assert player_tokens[actor_rows[0], 15] == 1.0
    assert player_tokens[opponent_rows[0], 15] == 0.0
    assert np.all(player_tokens[opponent_rows[0], 16:21] == 0.0)


def test_node_production_normalizes_live_probabilities_and_adapter_pips_identically():
    probability_form = {
        "wood": 4.0 / 36.0,
        "brick": 5.0 / 36.0,
        "sheep": 2.0 / 36.0,
    }
    pip_form = {"wood": 4, "brick": 5, "sheep": 2}

    assert _node_pips_by_resource(probability_form) == [4, 5, 2, 0, 0]
    assert _node_pips_by_resource(probability_form) == _node_pips_by_resource(
        pip_form
    )


def test_live_python_vertex_production_features_encode_dice_pips():
    env = ColonistMultiAgentEnv(ColonistMultiAgentConfig(players=2, vps_to_win=10))
    try:
        _observations, _info = env.reset(seed=600001)
        features = build_entity_token_features(env, env.current_player_name())
        node_production = env.game.state.board.map.node_production
        node = max(node_production, key=lambda item: sum(node_production[item].values()))
        expected_by_resource = _node_pips_by_resource(node_production[node])

        assert sum(expected_by_resource) > 0
        assert features["vertex_tokens"][node, 9] == pytest.approx(
            sum(expected_by_resource) / 18.0,
            abs=5e-4,
        )
        assert features["vertex_tokens"][node, 10:15] == pytest.approx(
            np.asarray(expected_by_resource) / 10.0,
            abs=5e-4,
        )
        assert np.count_nonzero(features["vertex_tokens"][:, 9]) > 0
    finally:
        env.close()


# ---------------------------------------------------------------------------
# `_scale`'s plain-Python clamp (task #49b) must be bit-identical to the
# np.clip-based reference it replaced, across every value shape it's called
# with in practice.
# ---------------------------------------------------------------------------


def test_scale_matches_np_clip_reference_across_representative_values():
    denominators = [1.0, 2.0, 5.0, 19.0, 0.0, -3.0]
    values = [0, 1, -1, 5, 19, 20, 1000, -1000, None, "3", "not a number", 3.0, 3.9, True, False]
    for denominator in denominators:
        for value in values:
            expected = _scale_via_np_clip(value, denominator)
            actual = _scale(value, denominator)
            assert actual == pytest.approx(expected), (
                f"_scale({value!r}, {denominator!r}) = {actual!r}, expected {expected!r}"
            )
            assert isinstance(actual, float)


def test_scale_clamps_to_unit_interval():
    assert _scale(-5, 1.0) == 0.0
    assert _scale(5, 1.0) == 1.0
    assert _scale(0, 1.0) == 0.0
    assert _scale(1, 2.0) == 0.5


def test_entity_token_features_unaffected_by_scale_optimization():
    """End-to-end regression bar for task #49b: the full feature tensor
    build (which calls `_scale` on the order of once per numeric feature per
    token) must be bit-identical before/after replacing `_scale`'s np.clip
    call with a plain-Python clamp."""
    import catan_zero.rl.entity_token_features as entity_token_features_module

    env = ColonistMultiAgentEnv(ColonistMultiAgentConfig(players=2, vps_to_win=10))
    _observations, _info = env.reset(seed=789)
    actor = env.current_player_name()

    optimized = build_entity_token_features(env, actor)

    original_scale = entity_token_features_module._scale
    entity_token_features_module._scale = _scale_via_np_clip
    try:
        reference = build_entity_token_features(env, actor)
    finally:
        entity_token_features_module._scale = original_scale

    assert set(optimized.keys()) == set(reference.keys())
    for key in optimized:
        if key == "schema":
            continue
        assert np.array_equal(optimized[key], reference[key]), f"mismatch in {key!r}"


# ---------------------------------------------------------------------------
# Board-topology memoization (perf fix): `_topology` caches the board-invariant
# hex/vertex/edge adjacency tables keyed on `_topology_key`, recomputing them
# from scratch only on a cache miss. This must be a pure no-op: cold (cache
# cleared) and warm (cache hit) calls for the SAME board must produce
# bit-identical topology arrays, and mutating a warm-call's returned array
# must never corrupt the cached entry a later call reuses.
# ---------------------------------------------------------------------------


def _clear_topology_cache() -> None:
    import catan_zero.rl.entity_token_features as entity_token_features_module

    entity_token_features_module._TOPOLOGY_CACHE.clear()


def test_topology_cache_cold_vs_warm_bit_identical():
    env = ColonistMultiAgentEnv(ColonistMultiAgentConfig(players=2, vps_to_win=10))
    _observations, _info = env.reset(seed=321)
    payload = env.observation_payload(env.current_player_name(), include_event_log=False)

    _clear_topology_cache()
    cold = _topology(payload)  # cache miss: runs the full computation
    warm = _topology(payload)  # cache hit: served from _TOPOLOGY_CACHE

    assert set(cold.keys()) == set(warm.keys())
    for key in ("hex_vertex_ids", "hex_edge_ids", "edge_vertex_ids"):
        assert np.array_equal(cold[key], warm[key]), f"mismatch in {key!r}"
        assert cold[key].dtype == warm[key].dtype
    assert cold["edge_to_id"] == warm["edge_to_id"]
    assert cold["coordinate_to_hex"] == warm["coordinate_to_hex"]
    assert cold["tiles"] == warm["tiles"]


def test_topology_cache_warm_call_returns_independent_arrays():
    """A caller mutating a warm call's returned array must not corrupt the
    cached entry a later call reuses (defensive `.copy()` of the cached
    numpy arrays)."""
    env = ColonistMultiAgentEnv(ColonistMultiAgentConfig(players=2, vps_to_win=10))
    _observations, _info = env.reset(seed=321)
    payload = env.observation_payload(env.current_player_name(), include_event_log=False)

    _clear_topology_cache()
    _topology(payload)  # populate the cache
    warm = _topology(payload)
    warm["hex_vertex_ids"][:] = -99

    still_warm = _topology(payload)
    assert not np.array_equal(still_warm["hex_vertex_ids"], warm["hex_vertex_ids"])


def test_topology_cache_end_to_end_features_unaffected_across_game_states():
    """End-to-end regression bar: the full feature build's topology-derived
    tensors must be identical whether the board's topology cache is cold or
    warm at multiple different game states of the SAME board (piece
    placement/robber/dev-card state changes across states; topology itself
    never does)."""
    env = ColonistMultiAgentEnv(ColonistMultiAgentConfig(players=2, vps_to_win=10))
    _observations, _info = env.reset(seed=321)
    actor = env.current_player_name()

    _clear_topology_cache()
    reference = build_entity_token_features(env, actor)

    for _ in range(6):
        env.step(_first_legal_action(env))
        actor = env.current_player_name()
        warm = build_entity_token_features(env, actor)
        for key in ("hex_vertex_ids", "hex_edge_ids", "edge_vertex_ids"):
            assert np.array_equal(reference[key], warm[key]), f"mismatch in {key!r}"


def _first_legal_action(env) -> int:
    payload = env.observation_payload(env.current_player_name(), include_event_log=False)
    legal = tuple(payload.get("structured_legal_actions", ()))
    return int(legal[0]["index"]) if legal else 0
