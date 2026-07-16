from __future__ import annotations

import numpy as np

from tools.curate_teacher_data import (
    ROLL_ACTION_IDS,
    _curate_shard_mask,
    _subtract_dropped_duplicates,
)


def _shard(*, phases: list[str], actions: list[int]) -> dict[str, np.ndarray]:
    n = len(phases)
    legal = np.asarray(actions, dtype=np.int16).reshape(n, 1)
    return {
        "action_taken": np.asarray(actions, dtype=np.int16),
        "legal_action_ids": legal,
        "teacher_name": np.full(n, "teacher"),
        "target_score_source": np.full(n, ""),
        "phase": np.asarray(phases),
        "target_policy": np.ones((n, 1), dtype=np.float32),
        "target_scores": np.zeros((n, 1), dtype=np.float32),
        "truncated": np.zeros(n, dtype=np.bool_),
        "winner": np.full(n, "BLUE"),
        "has_final_public_vps": np.ones(n, dtype=np.bool_),
    }


def _curate(shard: dict[str, np.ndarray], **overrides):
    kwargs = {
        "rng": np.random.default_rng(7),
        "teacher_keep": {},
        "forced_keep_prob": 0.0,
        "drop_forced_in_important_phases": False,
        "roll_keep_prob": 1.0,
        "drop_truncated": True,
        "preserve_value_only_filtered_rows": False,
    }
    kwargs.update(overrides)
    return _curate_shard_mask(shard, **kwargs)


def test_production_prompt_names_protect_forced_important_rows():
    phases = [
        "BUILD_INITIAL_ROAD",
        "BUILD_INITIAL_SETTLEMENT",
        "DISCARD",
        "MOVE_ROBBER",
        "PLAY_TURN",
    ]
    keep, report, policy_weights, _value_weights = _curate(
        _shard(phases=phases, actions=[1, 2, 3, 4, 5])
    )

    np.testing.assert_array_equal(keep, np.ones(len(phases), dtype=np.bool_))
    np.testing.assert_array_equal(policy_weights, np.ones(len(phases), dtype=np.float32))
    assert report["dropped_forced"] == 0


def test_strict_forced_filter_still_applies_inside_production_prompts():
    phases = ["BUILD_INITIAL_ROAD", "DISCARD", "PLAY_TURN"]
    keep, report, policy_weights, _value_weights = _curate(
        _shard(phases=phases, actions=[1, 2, 3]),
        drop_forced_in_important_phases=True,
    )

    assert not keep.any()
    assert not policy_weights.any()
    assert report["dropped_forced"] == len(phases)


def test_roll_filter_decodes_action_id_when_phase_is_play_turn():
    roll_action = next(iter(ROLL_ACTION_IDS))
    keep, report, policy_weights, _value_weights = _curate(
        _shard(phases=["PLAY_TURN"], actions=[roll_action]),
        forced_keep_prob=1.0,
        roll_keep_prob=0.0,
    )

    assert not bool(keep[0])
    assert policy_weights[0] == 0.0
    assert report["dropped_roll"] == 1


def test_duplicate_subtraction_reports_roll_by_action_id():
    roll_action = next(iter(ROLL_ACTION_IDS))
    shard = _shard(
        phases=["PLAY_TURN", "PLAY_TURN"],
        actions=[roll_action, roll_action],
    )
    keep, report, policy_weights, value_weights = _curate(
        shard,
        forced_keep_prob=1.0,
        roll_keep_prob=1.0,
    )
    duplicate = np.asarray([False, True])
    final_keep = keep & ~duplicate
    policy_weights[duplicate] = 0.0
    value_weights[duplicate] = 0.0

    adjusted = _subtract_dropped_duplicates(
        report,
        shard,
        duplicate_mask=duplicate,
        final_keep_mask=final_keep,
        policy_weight_multiplier=policy_weights,
        value_weight_multiplier=value_weights,
    )

    assert adjusted["kept_policy_effective_roll"] == 1


def test_legacy_lowercase_phase_aliases_remain_supported():
    phases = ["initial_build", "main_turn", "robber", "discard"]
    keep, _report, policy_weights, _value_weights = _curate(
        _shard(phases=phases, actions=[1, 2, 3, 4])
    )

    assert keep.all()
    assert policy_weights.all()
