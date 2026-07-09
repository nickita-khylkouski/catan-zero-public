from __future__ import annotations

import json

import numpy as np
import pytest

from catan_zero.rl.action_mask import ActionCatalog
from catan_zero.rl.gumbel_self_play import COLORS
from catan_zero.search.neural_rust_mcts import (
    rust_action_context_batch,
    rust_game_to_entity_batch,
    rust_policy_action_ids,
)
from catan_zero.search.rust_mcts import _require_rust_module


def _rust():
    try:
        return _require_rust_module()
    except RuntimeError as error:
        pytest.skip(str(error))


def _advance_to_multi_action_state(catanatron_rs, *, seed: int, min_legal: int = 2):
    game = catanatron_rs.Game.simple(["RED", "BLUE"], seed=seed)
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


# ---------------------------------------------------------------------------
# Pre-fetched snapshot/action_by_id must be a pure performance optimization:
# passing them in must never change the computed feature tensors (regression
# bar for task #49's featurization fixes).
# ---------------------------------------------------------------------------


def test_rust_game_to_entity_batch_identical_with_and_without_prefetched_context():
    catanatron_rs = _rust()
    game = _advance_to_multi_action_state(catanatron_rs, seed=7)
    legal_rust = _legal_rust_actions(game)
    actor = str(game.current_color())
    action_size = ActionCatalog(COLORS).size
    mapped = rust_policy_action_ids(game, legal_rust, colors=COLORS, action_size=action_size)

    baseline = rust_game_to_entity_batch(
        game, legal_rust, actor=actor, colors=COLORS, action_size=action_size, policy_action_ids=mapped
    )

    snapshot = json.loads(game.json_snapshot())
    action_ids = [int(a) for a in game.playable_action_indices(list(COLORS), None)]
    raw_actions = json.loads(game.playable_actions_json())
    action_by_id = {action_id: raw for action_id, raw in zip(action_ids, raw_actions)}

    prefetched = rust_game_to_entity_batch(
        game,
        legal_rust,
        actor=actor,
        colors=COLORS,
        action_size=action_size,
        policy_action_ids=mapped,
        snapshot=snapshot,
        action_by_id=action_by_id,
    )

    assert set(baseline.keys()) == set(prefetched.keys())
    for key in baseline:
        assert np.array_equal(baseline[key], prefetched[key]), f"mismatch in entity feature {key!r}"


def test_rust_action_context_batch_identical_with_and_without_prefetched_context():
    catanatron_rs = _rust()
    game = _advance_to_multi_action_state(catanatron_rs, seed=7)
    legal_rust = _legal_rust_actions(game)
    actor = str(game.current_color())
    action_size = ActionCatalog(COLORS).size
    mapped = rust_policy_action_ids(game, legal_rust, colors=COLORS, action_size=action_size)

    baseline = rust_action_context_batch(
        game, legal_rust, actor=actor, colors=COLORS, action_size=action_size, policy_action_ids=mapped
    )

    snapshot = json.loads(game.json_snapshot())
    action_ids = [int(a) for a in game.playable_action_indices(list(COLORS), None)]
    raw_actions = json.loads(game.playable_actions_json())
    action_by_id = {action_id: raw for action_id, raw in zip(action_ids, raw_actions)}

    prefetched = rust_action_context_batch(
        game,
        legal_rust,
        actor=actor,
        colors=COLORS,
        action_size=action_size,
        policy_action_ids=mapped,
        snapshot=snapshot,
        action_by_id=action_by_id,
    )

    assert np.array_equal(baseline, prefetched)


def test_prefetched_context_identical_across_several_real_game_states():
    """Same property as the two tests above, but sampled across several
    distinct seeds/decision points rather than just one, since the shared
    `_resolve_entity_adapter` preamble is exercised by every self-play
    decision row."""
    catanatron_rs = _rust()
    action_size = ActionCatalog(COLORS).size
    for seed in (1, 2, 3):
        game = _advance_to_multi_action_state(catanatron_rs, seed=seed)
        legal_rust = _legal_rust_actions(game)
        actor = str(game.current_color())
        mapped = rust_policy_action_ids(game, legal_rust, colors=COLORS, action_size=action_size)

        entity_baseline = rust_game_to_entity_batch(
            game, legal_rust, actor=actor, colors=COLORS, action_size=action_size, policy_action_ids=mapped
        )
        context_baseline = rust_action_context_batch(
            game, legal_rust, actor=actor, colors=COLORS, action_size=action_size, policy_action_ids=mapped
        )

        snapshot = json.loads(game.json_snapshot())
        action_ids = [int(a) for a in game.playable_action_indices(list(COLORS), None)]
        raw_actions = json.loads(game.playable_actions_json())
        action_by_id = {action_id: raw for action_id, raw in zip(action_ids, raw_actions)}

        entity_prefetched = rust_game_to_entity_batch(
            game,
            legal_rust,
            actor=actor,
            colors=COLORS,
            action_size=action_size,
            policy_action_ids=mapped,
            snapshot=snapshot,
            action_by_id=action_by_id,
        )
        context_prefetched = rust_action_context_batch(
            game,
            legal_rust,
            actor=actor,
            colors=COLORS,
            action_size=action_size,
            policy_action_ids=mapped,
            snapshot=snapshot,
            action_by_id=action_by_id,
        )

        for key in entity_baseline:
            assert np.array_equal(entity_baseline[key], entity_prefetched[key]), (
                f"seed={seed} mismatch in entity feature {key!r}"
            )
        assert np.array_equal(context_baseline, context_prefetched), f"seed={seed} context mismatch"
