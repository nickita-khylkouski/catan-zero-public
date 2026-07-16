"""Wiring parity for `EntityGraphRustEvaluatorConfig.rust_featurize` (task #81
phase 2 integration).

rust-featurize's `tests/test_rust_featurize_parity.py` proves the Rust
FUNCTION is bit-exact vs `build_entity_token_features`. This suite proves the
EVALUATOR WIRING is: `_entity_batch_via_rust` (flag ON) must hand the forward
pass a dict bit-identical -- same keys, dtypes, shapes, values -- to what the
flag-OFF path (`rust_game_to_entity_batch`) builds on the same real game
states, in BOTH masking regimes, including the lazy once-per-evaluator
topology bootstrap. Identical `entity` input => identical evaluator output,
so this is the torch-free core of the end-to-end gate (the full
checkpoint-loaded output check + 32-game identical-seed smoke still run on a
GPU host before fleet adoption).

Needs the catanatron_rs extension WITH `build_entity_features_flat` (the
task-#81 build); skips cleanly on older wheels.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from catan_zero.rl.entity_feature_adapter import (
    RUST_ENTITY_ADAPTER_V2,
    RUST_ENTITY_ADAPTER_V3,
    RUST_ENTITY_ADAPTER_V5,
)
from catan_zero.rl.meaningful_history import (
    MEANINGFUL_PUBLIC_HISTORY_SCHEMA_V1,
    MEANINGFUL_PUBLIC_HISTORY_SCHEMA_V2,
)

try:
    import catanatron_rs

    _HAS_RUST_FEATURIZE = hasattr(catanatron_rs, "build_entity_features_flat")
except ImportError:
    catanatron_rs = None  # type: ignore[assignment]
    _HAS_RUST_FEATURIZE = False

needs_rust_featurize = pytest.mark.skipif(
    not _HAS_RUST_FEATURIZE,
    reason="catanatron_rs with build_entity_features_flat (task #81) not installed",
)

COLORS: tuple[str, ...] = ("RED", "BLUE")


def _make_evaluator(public_observation: bool, adapter_version: str):
    """EntityGraphRustEvaluator with a torch-free dummy policy: this suite
    only exercises the featurize seam, never `forward_legal_np`."""
    from catan_zero.rl.action_mask import ActionCatalog
    from catan_zero.search.neural_rust_mcts import (
        EntityGraphRustEvaluator,
        EntityGraphRustEvaluatorConfig,
    )

    policy = SimpleNamespace(
        action_size=ActionCatalog(COLORS).size,
        trained_with_masked_hidden_info=public_observation,
        entity_feature_adapter_version=adapter_version,
        config=SimpleNamespace(
            meaningful_public_history=adapter_version == RUST_ENTITY_ADAPTER_V5,
            meaningful_public_history_schema=(
                MEANINGFUL_PUBLIC_HISTORY_SCHEMA_V2
                if adapter_version == RUST_ENTITY_ADAPTER_V5
                else MEANINGFUL_PUBLIC_HISTORY_SCHEMA_V1
            ),
            event_history_limit=64,
            public_card_count_features=adapter_version == RUST_ENTITY_ADAPTER_V5,
        ),
    )
    return EntityGraphRustEvaluator(
        policy,  # type: ignore[arg-type]
        config=EntityGraphRustEvaluatorConfig(
            public_observation=public_observation,
            rust_featurize=True,
            entity_feature_adapter_version=adapter_version,
        ),
    )


def _collect_states(seed: int, count: int) -> list:
    games = []
    game = catanatron_rs.Game.simple(list(COLORS), seed=seed)
    step = 0
    while len(games) < count and game.winning_color() is None and step < 400:
        legal = tuple(int(a) for a in game.playable_action_indices(list(COLORS), None))
        if legal and step % 7 == 0:
            games.append(game.copy())
        game.play_tick()
        step += 1
    return games


def _paired_entities(game, evaluator, public_observation: bool):
    import json

    from catan_zero.search.neural_rust_mcts import (
        _fetch_leaf_decision_inputs,
        _policy_history_options,
        _resolve_entity_adapter,
        rust_game_to_entity_batch,
        rust_policy_action_ids,
    )

    acting_color = str(game.current_color())
    history_enabled, history_limit, history_schema = _policy_history_options(
        evaluator.policy
    )
    snapshot_text, action_by_id = _fetch_leaf_decision_inputs(game, COLORS)
    legal_actions = tuple(action_by_id.keys())
    policy_action_ids = rust_policy_action_ids(
        game,
        legal_actions,
        colors=COLORS,
        action_size=int(evaluator.policy.action_size),
        action_by_id=action_by_id,
    )
    resolved = _resolve_entity_adapter(
        game,
        legal_actions,
        colors=COLORS,
        action_size=int(evaluator.policy.action_size),
        policy_action_ids=policy_action_ids,
        snapshot=json.loads(snapshot_text),
        action_by_id=action_by_id,
        public_observation=public_observation,
        perspective=acting_color,
        meaningful_public_history=history_enabled,
        meaningful_public_history_schema=history_schema,
        entity_feature_adapter_version=evaluator.config.entity_feature_adapter_version,
    )
    python_entity = rust_game_to_entity_batch(
        game,
        legal_actions,
        actor=acting_color,
        colors=COLORS,
        action_size=int(evaluator.policy.action_size),
        policy_action_ids=policy_action_ids,
        public_observation=public_observation,
        meaningful_public_history=history_enabled,
        history_limit=history_limit,
        meaningful_public_history_schema=history_schema,
        entity_feature_adapter_version=evaluator.config.entity_feature_adapter_version,
        resolved=resolved,
    )
    rust_entity = evaluator._entity_batch_via_rust(
        game,
        colors=COLORS,
        policy_action_ids=policy_action_ids,
        acting_color=acting_color,
        adapter=resolved[1],
    )
    return python_entity, rust_entity


@needs_rust_featurize
@pytest.mark.parametrize("public_observation", [False, True])
@pytest.mark.parametrize(
    "adapter_version",
    [RUST_ENTITY_ADAPTER_V2, RUST_ENTITY_ADAPTER_V3, RUST_ENTITY_ADAPTER_V5],
)
def test_deduction_feature_wiring_matches_python_path(
    public_observation: bool,
    adapter_version: str,
) -> None:
    from catan_zero.deduction_tracker import DEDUCTION_FEATURES_KEY

    evaluator = _make_evaluator(public_observation, adapter_version)
    nonzero_states = 0
    for game in _collect_states(seed=17, count=25):
        python_entity, rust_entity = _paired_entities(
            game, evaluator, public_observation
        )
        expected = np.asarray(python_entity[DEDUCTION_FEATURES_KEY])
        got = np.asarray(rust_entity[DEDUCTION_FEATURES_KEY])
        assert got.dtype == expected.dtype
        assert got.shape == expected.shape
        assert np.array_equal(got, expected)
        if np.any(got != 0.0):
            nonzero_states += 1
    assert nonzero_states > 0, (
        "parity sample must exercise real public card-count evidence"
    )


@needs_rust_featurize
@pytest.mark.parametrize("public_observation", [False, True])
@pytest.mark.parametrize(
    "adapter_version",
    [RUST_ENTITY_ADAPTER_V2, RUST_ENTITY_ADAPTER_V3, RUST_ENTITY_ADAPTER_V5],
)
def test_rust_wiring_matches_python_path(
    public_observation: bool,
    adapter_version: str,
) -> None:
    from catan_zero.deduction_tracker import DEDUCTION_FEATURES_KEY

    evaluator = _make_evaluator(public_observation, adapter_version)
    states = _collect_states(seed=17, count=25)
    assert len(states) >= 15, "trajectory too short to be a meaningful sample"

    compared = 0
    nonzero_deduction_states = 0
    for game in states:
        python_entity, rust_entity = _paired_entities(
            game, evaluator, public_observation
        )

        assert DEDUCTION_FEATURES_KEY in rust_entity
        if np.any(np.asarray(rust_entity[DEDUCTION_FEATURES_KEY]) != 0.0):
            nonzero_deduction_states += 1
        assert set(rust_entity) == set(python_entity), (
            f"key mismatch: {set(rust_entity) ^ set(python_entity)}"
        )
        for key in python_entity:
            expected = np.asarray(python_entity[key])
            got = np.asarray(rust_entity[key])
            assert got.dtype == expected.dtype, f"{key}: dtype {got.dtype} != {expected.dtype}"
            assert got.shape == expected.shape, f"{key}: shape {got.shape} != {expected.shape}"
            assert np.array_equal(got, expected), f"{key}: values differ"
            compared += 1
    assert compared > 0
    assert nonzero_deduction_states > 0, (
        "parity sample must exercise real public card-count evidence"
    )


@needs_rust_featurize
def test_topology_is_bootstrapped_once_and_reused() -> None:
    import json

    from catan_zero.search.neural_rust_mcts import (
        _fetch_leaf_decision_inputs,
        _resolve_entity_adapter,
        rust_policy_action_ids,
    )

    evaluator = _make_evaluator(False, RUST_ENTITY_ADAPTER_V3)
    assert evaluator._rust_topology is None
    states = _collect_states(seed=23, count=3)
    topologies = []
    for game in states:
        acting_color = str(game.current_color())
        snapshot_text, action_by_id = _fetch_leaf_decision_inputs(game, COLORS)
        legal_actions = tuple(action_by_id.keys())
        policy_action_ids = rust_policy_action_ids(
            game,
            legal_actions,
            colors=COLORS,
            action_size=int(evaluator.policy.action_size),
            action_by_id=action_by_id,
        )
        resolved = _resolve_entity_adapter(
            game,
            legal_actions,
            colors=COLORS,
            action_size=int(evaluator.policy.action_size),
            policy_action_ids=policy_action_ids,
            snapshot=json.loads(snapshot_text),
            action_by_id=action_by_id,
            public_observation=False,
            perspective=acting_color,
        )
        evaluator._entity_batch_via_rust(
            game,
            colors=COLORS,
            policy_action_ids=policy_action_ids,
            acting_color=acting_color,
            adapter=resolved[1],
        )
        topologies.append(evaluator._rust_topology)
    assert topologies[0] is not None
    assert all(t is topologies[0] for t in topologies), "topology must be reused, not rebuilt"
