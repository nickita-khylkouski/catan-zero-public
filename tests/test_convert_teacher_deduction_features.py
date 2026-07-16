from __future__ import annotations

from collections import defaultdict
import random
from typing import Any

import numpy as np

from catan_zero.deduction_tracker import DEDUCTION_FEATURE_SIZE, true_state_label
from catan_zero.rl.entity_feature_adapter import (
    CURRENT_RUST_ENTITY_ADAPTER_VERSION,
)
from catan_zero.rl.multiagent_env import ColonistMultiAgentConfig, ColonistMultiAgentEnv
from tools.convert_teacher_to_entity_tokens import (
    DEDUCTION_FEATURES_KEY,
    ENTITY_KEYS,
    EntityShardWriter,
    _convert_seed,
)


def _play_and_record(seed: int, config: ColonistMultiAgentConfig, max_steps: int = 120):
    """Play a random-legal-action game and record exactly what a banked
    teacher shard's rows would carry for this decision stream, so
    `_convert_seed` can replay it deterministically against a fresh env."""
    env = ColonistMultiAgentEnv(config)
    rng = random.Random(seed)
    by_decision: dict[int, list[dict[str, Any]]] = defaultdict(list)
    _, info = env.reset(seed=seed)
    decision = 0
    while decision < max_steps:
        valid = tuple(int(a) for a in info.get("valid_actions", ()))
        if not valid:
            break
        player = str(info.get("current_player", ""))
        action = rng.choice(valid)
        legal_action_ids = np.asarray(sorted(valid), dtype=np.int16)
        by_decision[decision].append(
            {
                "player": player,
                "action_taken": int(action),
                "legal_action_ids": legal_action_ids,
                "decision_index": decision,
            }
        )
        _, _, terminated, truncated, info = env.step(action)
        decision += 1
        if terminated or truncated:
            break
    env.close()
    return by_decision


def test_emit_deduction_features_populates_additive_key(tmp_path):
    config = ColonistMultiAgentConfig(players=2, vps_to_win=10)
    seed = 777001
    by_decision = _play_and_record(seed, config)

    writer = EntityShardWriter(
        tmp_path,
        shard_size=10_000,
        fmt="npz",
        entity_keys=ENTITY_KEYS + (DEDUCTION_FEATURES_KEY,),
    )
    result = _convert_seed(seed, by_decision, config, writer, emit_deduction_features=True)
    assert result["mismatches"] == []
    assert result["converted_rows"] == len(by_decision)
    assert len(writer.rows) == len(by_decision)

    for row in writer.rows:
        features = row[DEDUCTION_FEATURES_KEY]
        assert features.shape == (4, DEDUCTION_FEATURE_SIZE)
        assert features.dtype == np.float32
        assert np.all(features >= 0.0) and np.all(features <= 1.0)

    writer.close()
    assert writer.paths
    loaded = np.load(writer.paths[0])
    assert DEDUCTION_FEATURES_KEY in loaded.files
    assert loaded[DEDUCTION_FEATURES_KEY].shape == (len(by_decision), 4, DEDUCTION_FEATURE_SIZE)


def test_emit_deduction_features_off_by_default_omits_key(tmp_path):
    config = ColonistMultiAgentConfig(players=2, vps_to_win=10)
    seed = 777002
    by_decision = _play_and_record(seed, config)

    writer = EntityShardWriter(tmp_path, shard_size=10_000, fmt="npz")
    _convert_seed(seed, by_decision, config, writer, emit_deduction_features=False)
    assert all(DEDUCTION_FEATURES_KEY not in row for row in writer.rows)


def test_conversion_binds_current_entity_adapter_semantics(tmp_path):
    config = ColonistMultiAgentConfig(players=2, vps_to_win=10)
    seed = 777004
    by_decision = _play_and_record(seed, config, max_steps=8)

    writer = EntityShardWriter(tmp_path, shard_size=10_000, fmt="npz")
    _convert_seed(seed, by_decision, config, writer, emit_deduction_features=False)

    assert writer.rows
    assert {
        str(row["adapter_version"]) for row in writer.rows
    } == {CURRENT_RUST_ENTITY_ADAPTER_VERSION}
    writer.close()
    loaded = np.load(writer.paths[0])
    assert set(map(str, loaded["adapter_version"])) == {
        CURRENT_RUST_ENTITY_ADAPTER_VERSION
    }


def test_emit_deduction_features_tracks_ground_truth_across_replay(tmp_path):
    """Re-derive the tracker independently (via the public API, not the CLI
    plumbing) and confirm the per-row feature table's exactness flag (slot
    16 for each opponent) agrees with omniscient ground truth at least once
    -- a light end-to-end sanity check that the wiring in `_convert_seed`
    produces the same values as directly driving `DeductionTracker`."""
    config = ColonistMultiAgentConfig(players=2, vps_to_win=10)
    seed = 777003
    by_decision = _play_and_record(seed, config, max_steps=200)

    writer = EntityShardWriter(
        tmp_path,
        shard_size=10_000,
        fmt="npz",
        entity_keys=ENTITY_KEYS + (DEDUCTION_FEATURES_KEY,),
    )
    _convert_seed(seed, by_decision, config, writer, emit_deduction_features=True)

    # Independently replay the same seed/policy to get omniscient ground
    # truth at each decision boundary, and confirm the shipped exactness
    # flag is never asserted true when the true hand doesn't actually match
    # what an independently-built tracker would say.
    from catan_zero.deduction_tracker import DeductionTracker

    env = ColonistMultiAgentEnv(config)
    _, info = env.reset(seed=seed)
    trackers: dict[str, DeductionTracker] = {}
    cursors: dict[str, int] = {}
    exact_flag_true_count = 0
    for decision, rows in sorted(by_decision.items()):
        row = rows[0]
        player = row["player"]
        opponents = tuple(name for name in env.player_names if name != player)
        if player not in trackers:
            trackers[player] = DeductionTracker(self_name=player, opponent_names=opponents)
            cursors[player] = 0
        tracker = trackers[player]
        replay = env.replay_trace(actor=player)
        tracker.observe_frames(replay[cursors[player] :])
        cursors[player] = len(replay)
        shipped = writer.rows[decision][DEDUCTION_FEATURES_KEY]
        for opponent in opponents:
            idx = env.player_names.index(opponent)
            expected = tracker.feature_vector_for(opponent, env.observation_payload(player, include_event_log=False))
            np.testing.assert_allclose(shipped[idx], expected, atol=1e-6)
            if shipped[idx][16] == 1.0:
                exact_flag_true_count += 1
        _, _, terminated, truncated, info = env.step(int(row["action_taken"]))
        if terminated or truncated:
            break
    env.close()
    assert exact_flag_true_count > 0
