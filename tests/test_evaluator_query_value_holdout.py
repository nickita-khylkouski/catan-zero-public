from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from catan_zero.rl.action_mask import ActionCatalog
from tools.evaluator_query_value_holdout import (
    ROOT_CLASSES,
    EvaluatorBinding,
    actor_handoff_pairs,
    build_report,
    classify_roots,
    collect_game_groups,
    compute_scalar_raw_values,
    decode_action_types,
    deployed_scalar_values,
    game_bootstrap_confidence_intervals,
    load_evaluator_binding,
    require_explicit_holdout_selection,
)
from tools.phase_sliced_value_calibration import ENTITY_KEYS


def _action_id(action_type: str) -> int:
    catalog = ActionCatalog(("BLUE", "RED"))
    matches = [
        index
        for index in range(catalog.size)
        if catalog.describe(index)["action_type"] == action_type
    ]
    assert matches
    return matches[0]


def _write_shard(path: Path) -> None:
    seeds = np.asarray([10, 10, 10, 20, 20, 20], dtype=np.int64)
    phases = np.asarray(
        [
            "BUILD_INITIAL_SETTLEMENT",
            "PLAY_TURN",
            "PLAY_TURN",
            "MOVE_ROBBER",
            "DISCARD",
            "PLAY_TURN",
        ]
    )
    actions = np.asarray(
        [
            _action_id("BUILD_SETTLEMENT"),
            _action_id("END_TURN"),
            _action_id("ROLL"),
            _action_id("MOVE_ROBBER"),
            _action_id("DISCARD_RESOURCE"),
            _action_id("BUILD_ROAD"),
        ],
        dtype=np.int16,
    )
    payload = {key: np.zeros((len(seeds), 1), dtype=np.float32) for key in ENTITY_KEYS}
    payload.update(
        {
            "game_seed": seeds,
            "terminated": np.ones(len(seeds), dtype=np.bool_),
            "truncated": np.zeros(len(seeds), dtype=np.bool_),
            "winner": np.asarray(["RED"] * len(seeds)),
            "player": np.asarray(["BLUE", "BLUE", "RED", "RED", "BLUE", "BLUE"]),
            "phase": phases,
            "decision_index": np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int32),
            "action_taken": actions,
            "action_mask_version": np.asarray(
                ["colonist-multiagent-v1"] * len(seeds)
            ),
            "adapter_version": np.asarray(["adapter-test"] * len(seeds)),
            "legal_action_mask": np.ones((len(seeds), 2), dtype=np.bool_),
            "legal_action_ids": np.zeros((len(seeds), 2), dtype=np.int16),
            "legal_action_context": np.zeros((len(seeds), 2, 1), dtype=np.float32),
        }
    )
    np.savez(path, **payload)


def test_deployed_scalar_transform_matches_evaluator_contract() -> None:
    raw = np.asarray([-3.0, -0.5, 0.5, 3.0])
    assert np.allclose(
        deployed_scalar_values(raw, value_scale=2.0, value_squash="tanh"),
        np.tanh(raw * 2.0),
    )
    assert np.array_equal(
        deployed_scalar_values(raw, value_scale=2.0, value_squash="clip"),
        np.asarray([-1.0, -1.0, 1.0, 1.0]),
    )


def test_action_decode_separates_roll_end_turn_and_ordinary_play() -> None:
    actions = np.asarray(
        [_action_id("ROLL"), _action_id("END_TURN"), _action_id("BUILD_ROAD")]
    )
    versions = np.asarray(["colonist-multiagent-v1"] * 3)
    phases = np.asarray(["PLAY_TURN"] * 3)
    action_types = decode_action_types(actions, versions, phases)
    assert action_types.tolist() == ["ROLL", "END_TURN", "BUILD_ROAD"]
    assert classify_roots(phases, action_types).tolist() == [
        "pre_roll",
        "end_turn",
        "post_roll_play_turn",
    ]


def test_play_turn_unknown_action_schema_fails_closed() -> None:
    with pytest.raises(ValueError, match="unsupported action catalog"):
        decode_action_types(
            np.asarray([1]),
            np.asarray(["unknown-v9"]),
            np.asarray(["PLAY_TURN"]),
        )


def test_collect_game_groups_keeps_whole_games_and_bounds(tmp_path: Path) -> None:
    _write_shard(tmp_path / "rows.npz")
    groups = collect_game_groups(
        [str(tmp_path)],
        validation_game_seeds=np.asarray([10, 20]),
        validation_game_seed_ranges=(),
        max_games=1,
        max_rows=10,
    )
    assert len(groups) == 1
    assert groups[0]["game_seed"].tolist() == [10, 10, 10]
    assert len(set(groups[0]["game_id"].tolist())) == 1


def test_holdout_contract_refuses_unverified_in_sample_selection() -> None:
    with pytest.raises(ValueError, match="refusing unverified in-sample rows"):
        require_explicit_holdout_selection(None, ())
    require_explicit_holdout_selection("validation.json", ())
    require_explicit_holdout_selection(None, ((100, 199),))


def test_collect_game_groups_reassembles_one_game_across_shards(
    tmp_path: Path,
) -> None:
    combined = tmp_path / "combined.npz"
    _write_shard(combined)
    with np.load(combined) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}
    combined.unlink()
    first = np.asarray([0, 1, 3])
    second = np.asarray([2, 4, 5])
    np.savez(tmp_path / "a.npz", **{key: value[first] for key, value in payload.items()})
    np.savez(tmp_path / "b.npz", **{key: value[second] for key, value in payload.items()})

    groups = collect_game_groups(
        [str(tmp_path)],
        validation_game_seeds=None,
        validation_game_seed_ranges=(),
        max_games=1,
        max_rows=10,
    )
    assert len(groups) == 2
    assert sum(len(group["z"]) for group in groups) == 3
    assert {
        int(seed)
        for group in groups
        for seed in np.unique(group["game_seed"])
    } == {10}
    assert len(
        {
            game
            for group in groups
            for game in np.unique(group["game_id"]).tolist()
        }
    ) == 1


def test_scalar_query_preserves_optional_deduction_features() -> None:
    import torch

    class Policy:
        model = object()

        def forward_legal_np(self, entity, legal_ids, legal_context):
            assert "deduction_features" in entity
            assert entity["deduction_features"].shape == (2, 2, 3)
            assert legal_ids.shape == (2, 1)
            assert legal_context.shape == (2, 1, 1)
            return {"value": torch.tensor([0.25, -0.5])}

    group = {key: np.zeros((2, 1), dtype=np.float32) for key in ENTITY_KEYS}
    group["deduction_features"] = np.zeros((2, 2, 3), dtype=np.float32)
    group["legal_action_ids"] = np.zeros((2, 1), dtype=np.int64)
    group["legal_action_context"] = np.zeros((2, 1, 1), dtype=np.float32)
    assert np.allclose(
        compute_scalar_raw_values(Policy(), [group]), np.asarray([0.25, -0.5])
    )


def test_actor_handoff_requires_consecutive_decision_and_actor_change() -> None:
    end, nxt = actor_handoff_pairs(
        np.asarray(["opening", "end_turn", "pre_roll", "end_turn", "pre_roll"]),
        np.asarray(["g"] * 5),
        np.asarray([0, 1, 2, 4, 6]),
        np.asarray(["BLUE", "BLUE", "RED", "RED", "BLUE"]),
    )
    assert end.tolist() == [1]
    assert nxt.tolist() == [2]


def test_game_bootstrap_is_deterministic_and_clustered() -> None:
    q = np.asarray([0.8, 0.6, -0.7, -0.5])
    z = np.asarray([1.0, 1.0, -1.0, -1.0])
    games = np.asarray(["a", "a", "b", "b"])
    first = game_bootstrap_confidence_intervals(
        q,
        z,
        games,
        samples=100,
        seed=7,
        reliability_bin_count=5,
    )
    second = game_bootstrap_confidence_intervals(
        q,
        z,
        games,
        samples=100,
        seed=7,
        reliability_bin_count=5,
    )
    assert first == second
    assert first["value_rmse"]["valid_resamples"] == 100


def test_build_report_includes_required_slices_and_handoff_consistency(
    tmp_path: Path,
) -> None:
    _write_shard(tmp_path / "rows.npz")
    groups = collect_game_groups(
        [str(tmp_path)],
        validation_game_seeds=None,
        validation_game_seed_ranges=(),
        max_games=2,
        max_rows=100,
    )
    raw_q = np.asarray([-0.4, -0.3, 0.3, 0.4, -0.2, -0.1])
    report = build_report(
        raw_q,
        groups,
        binding=EvaluatorBinding(1.0, "tanh", True, None, None, None),
        min_slice_rows=1,
        reliability_bin_count=5,
        bootstrap_samples=50,
        bootstrap_seed=11,
    )
    assert set(report["by_root_class"]) == set(ROOT_CLASSES)
    assert set(report["by_phase"]) == {
        "BUILD_INITIAL_SETTLEMENT",
        "DISCARD",
        "MOVE_ROBBER",
        "PLAY_TURN",
    }
    assert set(report["by_legal_count_bucket"]) == {"2-4"}
    assert report["by_root_class"]["opening"]["n"] == 1
    assert report["by_root_class"]["pre_roll"]["n"] == 1
    assert report["by_root_class"]["end_turn"]["n"] == 1
    assert report["by_root_class"]["actor_handoff_next"]["n"] == 1
    assert report["actor_handoff_consistency"]["n_pairs"] == 1
    assert (
        report["actor_handoff_consistency"][
            "terminal_label_opposition_fraction"
        ]
        == 1.0
    )
    assert "spearman_q_z" in report["global"]
    assert "spearman_q_z" in report["global"]["game_bootstrap_95ci"]


def test_science_contract_binds_scalar_evaluator_and_adapter(
    tmp_path: Path,
) -> None:
    path = tmp_path / "science.json"
    path.write_text(
        json.dumps(
            {
                "operator": {
                    "evaluator": {
                        "value_readout": "scalar",
                        "value_scale": 1.5,
                        "value_squash": "clip",
                        "public_observation": True,
                    }
                },
                "learner": {
                    "model_construction": {
                        "entity_feature_adapter_version": "adapter-v5"
                    }
                },
            }
        )
    )
    binding = load_evaluator_binding(
        path, value_scale=99.0, value_squash="tanh"
    )
    assert binding.value_scale == 1.5
    assert binding.value_squash == "clip"
    assert binding.public_observation is True
    assert binding.entity_feature_adapter_version == "adapter-v5"
