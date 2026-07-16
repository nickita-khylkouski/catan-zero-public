from __future__ import annotations

import argparse
import random

import pytest

import numpy as np

from catan_zero.rl.multiagent_env import ColonistMultiAgentEnv
from catan_zero.rl.policy_pool import assert_policy_compatible_with_env
from catan_zero.rl.self_play import CatanatronAlphaBetaPolicy
from catan_zero.rl.self_play import make_env_config
from catan_zero.rl.torch_ppo import _policy_observation_array
from catan_zero.rl.xdim_lite_policy import (
    XDimGraphConfig,
    XDimGraphPolicy,
    XDimLiteConfig,
    XDimLitePolicy,
    masked_logits,
    normalize_observations,
)
from tools.train_bc import (
    build_sample_weights,
    build_value_sample_weights,
    _has_distillation_distribution,
    _metric_from_sum_denominator,
    _normalize_teacher_shard,
    _q_score_loss_parts,
    _scores_to_policy,
    _support_log_softmax,
    _soft_target_array,
    _target_columns,
    _value_targets,
    teacher_data_quality,
    validate_teacher_data_schema,
)
from tools.curate_teacher_data import (
    ShardWriter as CurationShardWriter,
    _curate_shard_mask,
)
from tools.report_teacher_data_quality import _apply_strict_35m_defaults
from tools.report_teacher_data_quality import _check_manifest_metadata
from tools.report_teacher_data_quality import _check_strict_35m_teacher_gates
from tools.report_teacher_data_quality import _input_metadata
from tools.generate_teacher_data import ShardWriter
from tools.generate_teacher_data import _sample_teacher_name


def test_ab_root_rows_use_scores_under_prefer_scores():
    data = {
        "legal_action_ids": np.asarray([[10, 20, 30], [10, 20, 30]], dtype=np.int16),
        "action_taken": np.asarray([10, 10], dtype=np.int16),
        "teacher_name": np.asarray(["catanatron_ab4", "catanatron_ab4"]),
        "target_score_source": np.asarray(["ab_root", ""]),
        "target_policy": np.asarray(
            [
                [0.80, 0.10, 0.10],
                [0.80, 0.10, 0.10],
            ],
            dtype=np.float32,
        ),
        "target_policy_mask": np.ones((2, 3), dtype=bool),
        "target_scores": np.asarray(
            [
                [0.0, 10.0, 0.0],
                [0.0, 10.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "target_scores_mask": np.ones((2, 3), dtype=bool),
    }

    target, _ = _soft_target_array(data, np.asarray([0, 1]), 0.7, "prefer_scores")

    assert target[0, 1] > target[0, 0]
    assert target[1, 0] > target[1, 1]


def test_scores_to_policy_uses_temperature_without_row_standardization():
    scores = np.asarray([[100.0, 101.0], [0.0, 1.0]], dtype=np.float32)
    legal = np.asarray([[1, 2], [1, 2]], dtype=np.int16)

    policy = _scores_to_policy(scores, legal, temperature=1.0)

    np.testing.assert_allclose(policy[0], policy[1], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        policy[0],
        np.asarray([1.0 / (1.0 + np.e), np.e / (1.0 + np.e)], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_xdim_observation_normalization_preserves_binary_signals():
    obs = np.asarray([[0.0, 1.0, -1.0, 2.0, 25.0, 50.0, np.nan]], dtype=np.float32)

    normalized = normalize_observations(obs)

    assert normalized[0, 0] == 0.0
    assert normalized[0, 1] == 1.0
    assert normalized[0, 2] == -1.0
    assert np.isclose(normalized[0, 3], 2.0 / 25.0)
    assert normalized[0, 4] == 1.0
    assert normalized[0, 5] == 1.0
    assert normalized[0, 6] == 0.0


def test_mixed_seat_sample_teacher_name_resolves_acting_player_policy():
    class Sample:
        player = "red"
        teacher_name = None

    class Policy:
        def __init__(self, name: str):
            self.name = name

    assert _sample_teacher_name(
        Sample(),
        {"red": Policy("catanatron_ab5"), "blue": Policy("jsettlers_lite")},
    ) == "catanatron_ab5"


def test_xdim_masked_logits_replaces_invalid_large_logits():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[0.0, 1.0e20, -3.0]], dtype=torch.float32)

    masked = masked_logits(logits, [(0, 2)], action_size=3)

    assert masked[0, 0].item() == pytest.approx(0.0)
    assert masked[0, 2].item() == pytest.approx(-3.0)
    assert masked[0, 1].item() == pytest.approx(-1.0e9)


def test_xdim_graph_dot_product_logits_are_norm_bounded_without_bias():
    torch = pytest.importorskip("torch")
    policy = XDimGraphPolicy(
        XDimGraphConfig(
            observation_size=16,
            action_size=4,
            static_action_feature_size=3,
            context_action_feature_size=2,
            hidden_size=32,
            token_count=4,
            board_layers=1,
            attention_heads=4,
        ),
        np.full((4, 3), 1000.0, dtype=np.float32),
        device="cpu",
    )
    with torch.no_grad():
        policy.model.action_bias.weight.zero_()
        policy.model.action_bias.bias.zero_()
        policy.model.q_bias.weight.zero_()
        policy.model.q_bias.bias.zero_()

    outputs = policy.model(
        torch.full((2, 16), 1000.0),
        torch.full((2, 4, 5), 1000.0),
    )

    logit_scale = torch.clamp(policy.model.logit_scale.exp(), max=50.0).item()
    assert outputs["logits"].abs().max().item() <= logit_scale + 1.0e-4
    assert outputs["q_values"].abs().max().item() <= 1.0001


def test_xdim_graph_uses_separate_cls_and_pooled_norms():
    policy = XDimGraphPolicy(
        XDimGraphConfig(
            observation_size=16,
            action_size=4,
            static_action_feature_size=3,
            context_action_feature_size=2,
            hidden_size=32,
            token_count=4,
            board_layers=1,
            attention_heads=4,
        ),
        np.zeros((4, 3), dtype=np.float32),
        device="cpu",
    )

    assert policy.model.state_norm is not policy.model.pooled_state_norm
    assert hasattr(policy.model, "final_state_norm")


def test_xdim_graph_loads_legacy_checkpoint_without_new_norm_keys(tmp_path):
    torch = pytest.importorskip("torch")
    policy = XDimGraphPolicy(
        XDimGraphConfig(
            observation_size=16,
            action_size=4,
            static_action_feature_size=3,
            context_action_feature_size=2,
            hidden_size=32,
            token_count=4,
            board_layers=1,
            attention_heads=4,
            action_mask_version="colonist-multiagent-v1",
        ),
        np.zeros((4, 3), dtype=np.float32),
        device="cpu",
    )
    path = tmp_path / "legacy_graph.pt"
    policy.save(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    for key in (
        "pooled_state_norm.weight",
        "pooled_state_norm.bias",
        "final_state_norm.weight",
        "final_state_norm.bias",
    ):
        checkpoint["model"].pop(key)
    torch.save(checkpoint, path)

    loaded = XDimGraphPolicy.load(path, device="cpu")

    assert hasattr(loaded.model, "pooled_state_norm")
    assert hasattr(loaded.model, "final_state_norm")


def test_one_hot_policy_targets_are_not_treated_as_distillation_rows():
    target = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.6, 0.4, 0.0],
        ],
        dtype=np.float32,
    )
    support = target > 0.0

    has_soft = _has_distillation_distribution(target, support)

    assert has_soft.tolist() == [False, True]


def test_low_coverage_soft_targets_fall_back_to_hard_labels():
    target = np.asarray([[0.55, 0.45, 0.0, 0.0]], dtype=np.float32)
    support = target > 0.0
    legal = np.asarray([[10, 20, 30, 40]], dtype=np.int16)

    permissive = _has_distillation_distribution(
        target,
        support,
        legal_action_ids=legal,
        min_legal_coverage=0.50,
    )
    strict = _has_distillation_distribution(
        target,
        support,
        legal_action_ids=legal,
        min_legal_coverage=0.75,
    )

    assert permissive.tolist() == [True]
    assert strict.tolist() == [False]


def test_weighted_metric_uses_weight_denominator_not_raw_sample_count():
    weighted_sum = 12.0
    raw_rows = 10.0
    weight_sum = 3.0

    assert weighted_sum / raw_rows == pytest.approx(1.2)
    assert _metric_from_sum_denominator(weighted_sum, weight_sum) == pytest.approx(4.0)


def test_teacher_data_quality_reports_effective_soft_distillation_coverage():
    data = {
        "action_taken": np.asarray([1, 1], dtype=np.int16),
        "legal_action_ids": np.asarray(
            [
                [1, 2, 3, 4],
                [1, 2, 3, 4],
            ],
            dtype=np.int16,
        ),
        "target_policy": np.asarray(
            [
                [0.5, 0.5, 0.0, 0.0],
                [0.25, 0.25, 0.25, 0.25],
            ],
            dtype=np.float32,
        ),
        "target_policy_mask": np.asarray(
            [
                [True, True, False, False],
                [True, True, True, True],
            ],
            dtype=bool,
        ),
        "target_scores": np.full((2, 4), np.nan, dtype=np.float32),
        "target_scores_mask": np.zeros((2, 4), dtype=bool),
        "policy_weight_multiplier": np.asarray([1.0, 1.0], dtype=np.float32),
    }

    permissive = teacher_data_quality(data, soft_target_min_legal_coverage=0.50)
    strict = teacher_data_quality(data, soft_target_min_legal_coverage=0.75)

    assert permissive["effective_soft_distillation_rows"] == 2
    assert strict["effective_soft_distillation_rows"] == 1
    assert strict["policy_active_effective_soft_distillation_fraction"] == pytest.approx(0.5)


def test_teacher_data_quality_reports_unflagged_vp_rows():
    data = {
        "action_taken": np.asarray([1, 2], dtype=np.int16),
        "legal_action_ids": np.asarray([[1], [2]], dtype=np.int16),
        "final_actual_vps": np.asarray([[10, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int16),
        "has_final_actual_vps": np.asarray([False, False]),
        "final_public_vps": np.asarray([[0, 0, 0, 0], [7, 5, 0, 0]], dtype=np.int16),
        "has_final_public_vps": np.asarray([False, True]),
    }

    quality = teacher_data_quality(data)

    assert quality["unflagged_final_actual_vp_rows"] == 1
    assert quality["unflagged_final_public_vp_rows"] == 0


def test_q_score_loss_parts_use_sample_weight_denominator():
    torch = pytest.importorskip("torch")
    data = {
        "action_taken": np.asarray([1, 1], dtype=np.int16),
        "legal_action_ids": np.asarray([[1, 2], [1, 2]], dtype=np.int16),
        "target_scores": np.asarray([[0.0, 2.0], [0.0, 2.0]], dtype=np.float32),
        "teacher_name": np.asarray(["value_rollout_search", "value_rollout_search"]),
        "target_score_source": np.asarray(["value_rollout_search", "value_rollout_search"]),
    }
    q_values = torch.zeros((2, 2), dtype=torch.float32)
    weights = torch.asarray([2.0, 0.5], dtype=torch.float32)

    loss, weighted_sum, denominator = _q_score_loss_parts(
        q_values,
        data,
        np.asarray([0, 1]),
        weights,
        torch.device("cpu"),
        q_skip_teacher_prefixes=(),
    )

    assert float(denominator.item()) == pytest.approx(2.5)
    assert float(weighted_sum.item()) == pytest.approx(2.5)
    assert float(loss.item()) == pytest.approx(1.0)


def test_production_metadata_reads_current_manifest_source_provenance(tmp_path):
    hashes = {
        "src/catan_zero/rl/self_play.py": "sha256:" + "1" * 64,
        "src/catan_zero/rl/action_features.py": "sha256:" + "2" * 64,
        "src/catan_zero/rl/xdim_lite_policy.py": "sha256:" + "3" * 64,
    }
    manifest = {
        "track": "2p_no_trade",
        "vps_to_win": 10,
        "mixed_seats": True,
        "mixed_seat_mode": "random",
        "graph_history_features": True,
        "tool_provenance": {
            "schema_version": "teacher-tool-provenance-v1",
            "file_sha256": hashes,
            "feature_semantics_files": sorted(hashes),
        },
    }
    (tmp_path / "manifest.json").write_text(__import__("json").dumps(manifest), encoding="utf-8")

    metadata = _input_metadata(tmp_path)

    assert metadata["source_provenance_hashes"] == {
        path: [digest] for path, digest in sorted(hashes.items())
    }
    assert metadata["source_provenance_errors"] == []
    assert metadata["graph_history_features"] == [True]


def test_production_manifest_gate_requires_graph_history_features():
    failures: list[str] = []
    args = argparse.Namespace(
        production_35m_teacher=True,
        track="2p_no_trade",
        vps_to_win=10,
    )
    metadata = {
        "manifest_count": 1,
        "tracks": ["2p_no_trade"],
        "vps_to_win": [10],
        "mixed_seats": [True],
        "mixed_seat_modes": ["random"],
        "graph_history_features": [False],
        "source_provenance_hashes": {
            "src/catan_zero/rl/self_play.py": ["self"],
            "src/catan_zero/rl/action_features.py": ["actions"],
            "src/catan_zero/rl/xdim_lite_policy.py": ["xdim"],
        },
    }

    _check_manifest_metadata(failures, args, metadata)

    assert any("graph_history_features=true" in failure for failure in failures)


def test_strict_gate_accepts_complete_actual_vp_without_public_vp():
    args = argparse.Namespace(
        min_soft_score_fraction=0.0,
        min_q_score_rows_ge2_fraction=0.0,
        min_clean_terminal_outcome_fraction=0.0,
        min_final_public_vp_fraction=0.0,
        min_final_actual_vp_fraction=0.0,
        max_forced_action_fraction=1.0,
        max_truncated_fraction=1.0,
    )

    _apply_strict_35m_defaults(args)

    assert args.min_final_actual_vp_fraction == pytest.approx(0.99)
    assert args.min_final_public_vp_fraction == pytest.approx(0.0)


def test_strict_gate_counts_policy_active_soft_coverage_not_value_only_rows():
    report = {
        "samples": 300_000,
        "policy_active_rows": 200_000,
        "unflagged_final_actual_vp_rows": 0,
        "unflagged_final_public_vp_rows": 50_000,
        "final_actual_vp_fraction": 1.0,
        "teacher_counts": {
            "catanatron_ab4": 50_000,
            "catanatron_ab5": 50_000,
            "value_rollout_search": 50_000,
            "jsettlers_lite": 50_000,
        },
        "phase_counts": {
            "initial_build": 10_000,
            "main_turn": 150_000,
            "robber": 10_000,
            "discard": 10_000,
        },
        "by_teacher": {},
        "by_phase": {},
    }
    for teacher in report["teacher_counts"]:
        metrics = {
            "policy_active_rows": 50_000,
            "soft_score_fraction": 0.976,
            "clean_terminal_outcome_fraction": 1.0,
            "final_actual_vp_fraction": 1.0,
            "policy_active_effective_soft_distillation_fraction": 0.80,
            "q_score_rows_ge2_policy_active_fraction": 0.80,
            "ab_root_score_fraction": 0.976 if teacher.startswith("catanatron_ab") else 0.0,
            "soft_policy_fraction": 0.976 if teacher.startswith("catanatron_ab") else 0.0,
        }
        report["by_teacher"][teacher] = metrics
    for phase in report["phase_counts"]:
        report["by_phase"][phase] = {
            "policy_active_rows": 10_000 if phase != "main_turn" else 150_000,
            "policy_effective_forced_action_fraction": 0.0,
        }

    failures: list[str] = []
    _check_strict_35m_teacher_gates(failures, report)

    assert failures == []


def test_value_targets_fall_back_to_public_final_vp():
    torch = pytest.importorskip("torch")
    data = {
        "action_taken": np.asarray([1], dtype=np.int16),
        "winner": np.asarray(["BLUE"]),
        "player": np.asarray(["BLUE"]),
        "seat": np.asarray([0], dtype=np.int8),
        "truncated": np.asarray([False]),
        "final_actual_vps": np.asarray([[0, 0, 0, 0]], dtype=np.int16),
        "has_final_actual_vps": np.asarray([False]),
        "final_public_vps": np.asarray([[7, 5, 0, 0]], dtype=np.int16),
        "has_final_public_vps": np.asarray([True]),
    }

    _, vp_targets, _, has_vp, _, _, _ = _value_targets(
        data,
        np.asarray([0]),
        torch.device("cpu"),
        10,
    )

    assert bool(has_vp.item())
    assert float(vp_targets.item()) == pytest.approx(0.7)


def test_value_targets_ignore_legacy_vp_arrays_without_validity_flags():
    torch = pytest.importorskip("torch")
    data = {
        "action_taken": np.asarray([1], dtype=np.int16),
        "winner": np.asarray(["BLUE"]),
        "player": np.asarray(["BLUE"]),
        "seat": np.asarray([0], dtype=np.int8),
        "truncated": np.asarray([False]),
        "final_actual_vps": np.asarray([[10, 0, 0, 0]], dtype=np.int16),
        "final_public_vps": np.asarray([[10, 0, 0, 0]], dtype=np.int16),
    }

    _, _, _, has_vp, _, _, _ = _value_targets(
        data,
        np.asarray([0]),
        torch.device("cpu"),
        10,
    )

    assert not bool(has_vp.item())


def test_curation_defaults_missing_vp_validity_flags_to_false(tmp_path):
    writer = CurationShardWriter(tmp_path, shard_size=10, fmt="npz")
    writer.add_row(
        {
            "obs": np.zeros(4, dtype=np.float16),
            "legal_action_ids": np.asarray([2], dtype=np.int16),
            "legal_action_context": np.zeros((1, 18), dtype=np.float16),
            "action_taken": np.int16(2),
            "teacher_name": "legacy",
            "game_seed": np.int64(1),
            "final_public_vps": np.asarray([10, 0, 0, 0], dtype=np.int16),
            "final_actual_vps": np.asarray([10, 0, 0, 0], dtype=np.int16),
        }
    )
    writer.flush()

    saved = np.load(writer.paths[0], allow_pickle=False)
    assert not bool(saved["has_final_public_vps"][0])
    assert not bool(saved["has_final_actual_vps"][0])


def test_curation_does_not_keep_legacy_vp_rows_as_value_clean():
    shard = {
        "action_taken": np.asarray([1], dtype=np.int16),
        "legal_action_ids": np.asarray([[1]], dtype=np.int16),
        "teacher_name": np.asarray(["legacy"]),
        "target_score_source": np.asarray([""]),
        "phase": np.asarray(["roll"]),
        "target_policy": np.zeros((1, 1), dtype=np.float32),
        "target_scores": np.full((1, 1), np.nan, dtype=np.float32),
        "truncated": np.asarray([False]),
        "winner": np.asarray(["BLUE"]),
        "final_actual_vps": np.asarray([[10, 0, 0, 0]], dtype=np.int16),
        "final_public_vps": np.asarray([[10, 0, 0, 0]], dtype=np.int16),
    }

    keep, _, policy_weights, value_weights = _curate_shard_mask(
        shard,
        rng=np.random.default_rng(1),
        teacher_keep={},
        forced_keep_prob=0.0,
        drop_forced_in_important_phases=True,
        roll_keep_prob=0.0,
        drop_truncated=True,
        preserve_value_only_filtered_rows=True,
    )

    assert not bool(keep[0])
    assert float(policy_weights[0]) == 0.0
    assert float(value_weights[0]) == 0.0


def test_normalize_teacher_shard_defaults_missing_vp_validity_flags_to_false(tmp_path):
    shard = {
        "obs": np.zeros((1, 4), dtype=np.float16),
        "legal_action_ids": np.asarray([[2]], dtype=np.int16),
        "legal_action_context": np.zeros((1, 1, 18), dtype=np.float16),
        "action_taken": np.asarray([2], dtype=np.int16),
        "final_actual_vps": np.asarray([[10, 0, 0, 0]], dtype=np.int16),
        "final_public_vps": np.asarray([[10, 0, 0, 0]], dtype=np.int16),
    }

    normalized = _normalize_teacher_shard(shard, tmp_path / "legacy.npz")

    assert not bool(normalized["has_final_actual_vps"][0])
    assert not bool(normalized["has_final_public_vps"][0])


def test_generate_teacher_writer_defaults_missing_vp_validity_flags_to_false(tmp_path):
    writer = ShardWriter(tmp_path, shard_size=10, fmt="npz")

    writer.add_row(
        {
            "obs": np.zeros(4, dtype=np.float16),
            "valid": np.asarray([2], dtype=np.int16),
            "context": np.zeros((1, 18), dtype=np.float16),
            "action": np.int16(2),
            "teacher": "legacy",
            "seed": np.int64(1),
            "final_public_vps": np.asarray([10, 0, 0, 0], dtype=np.int16),
            "final_actual_vps": np.asarray([10, 0, 0, 0], dtype=np.int16),
        }
    )

    row = writer.rows[0]
    assert not bool(row["has_final_public_vps"])
    assert not bool(row["has_final_actual_vps"])
    assert writer.summary()["final_public_vp_rows"] == 0
    assert writer.summary()["final_actual_vp_rows"] == 0


def test_bc_weight_builders_honor_curation_multipliers():
    data = {
        "action_taken": np.asarray([1, 2], dtype=np.int16),
        "legal_action_ids": np.asarray([[1, 3], [2, 4]], dtype=np.int16),
        "policy_weight_multiplier": np.asarray([0.0, 1.0], dtype=np.float32),
        "value_weight_multiplier": np.asarray([1.0, 0.0], dtype=np.float32),
    }

    policy_weights = build_sample_weights(
        data,
        teacher_weights={},
        phase_weights={},
        forced_action_weight=1.0,
        winner_sample_weight=1.0,
        loser_sample_weight=1.0,
        vp_margin_weight=0.0,
        vps_to_win=10,
    )
    value_weights = build_value_sample_weights(data)

    assert policy_weights.tolist() == pytest.approx([0.0, 2.0])
    assert value_weights.tolist() == pytest.approx([2.0, 0.0])


def test_alphabeta_fallback_does_not_attach_rerun_root_scores():
    class Fallback:
        def select_action(self, env, observation, info, rng, *, training=False):
            return 2

    policy = CatanatronAlphaBetaPolicy.__new__(CatanatronAlphaBetaPolicy)
    policy.ab_anchor_weight = 0.70
    policy._fallback = Fallback()
    policy._last_action_key = None
    policy._last_action = None
    policy._last_scores_key = None
    policy._last_scores = None

    class State:
        action_records = ()

    class Game:
        state = State()

    class Env:
        game = Game()

    env = Env()
    info = {"current_player": "BLUE", "valid_actions": (1, 2)}

    def failing_root_search(env, info):
        key = CatanatronAlphaBetaPolicy._cache_key(policy, env, info)
        if policy._last_scores_key == key and policy._last_scores is not None:
            return policy._last_action, dict(policy._last_scores)
        raise RuntimeError("root search failed")

    policy._root_search = failing_root_search

    action = CatanatronAlphaBetaPolicy.select_action(
        policy,
        env,
        np.zeros(1, dtype=np.float32),
        info,
        np.random.default_rng(1),
    )

    assert action == 2
    assert CatanatronAlphaBetaPolicy.target_scores(
        policy,
        env,
        info,
        np.random.default_rng(2),
    ) == {}
    assert CatanatronAlphaBetaPolicy.target_policy(
        policy,
        env,
        info,
        np.random.default_rng(3),
    ) == {2: 1.0}


def test_alphabeta_root_search_restores_global_random_state():
    class FakePlayer:
        def __init__(self, *args, **kwargs):
            pass

        def alphabeta(self, *args, **kwargs):
            for _ in range(25):
                random.random()
            return "native-action", 0.0

    class FakeNode:
        children = ()

        def __init__(self, *args, **kwargs):
            pass

    class FakeCatalog:
        def try_encode(self, action):
            return 2 if action == "native-action" else None

    class FakeGameState:
        action_records = ()

    class FakeGame:
        state = FakeGameState()

        def copy(self):
            return self

    class FakeEnv:
        game = FakeGame()
        action_catalog = FakeCatalog()

        def current_player_color(self):
            return "BLUE"

        def valid_actions(self):
            return (1, 2, 3)

        def _trade_response_indices_for(self, action):
            return ()

    policy = CatanatronAlphaBetaPolicy.__new__(CatanatronAlphaBetaPolicy)
    policy._player_cls = FakePlayer
    policy._debug_state_node_cls = FakeNode
    policy.depth = 3
    policy.prunning = True
    policy.value_fn_builder_name = None
    policy._players = {}
    policy._last_action_key = None
    policy._last_action = None
    policy._last_scores_key = None
    policy._last_scores = None

    random.seed(12345)
    expected = random.random()
    random.seed(12345)
    action, scores = CatanatronAlphaBetaPolicy._root_search(
        policy,
        FakeEnv(),
        {"current_player": "BLUE", "valid_actions": (1, 2, 3)},
    )
    actual = random.random()

    assert action == 2
    assert scores == {}
    assert actual == expected


def test_xdim_checkpoint_records_and_checks_action_mask_version(tmp_path):
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")
    torch = pytest.importorskip("torch")

    config = make_env_config(players=2, vps_to_win=3)
    policy = XDimGraphPolicy.create(
        env_config=config,
        hidden_size=32,
        token_count=8,
        board_layers=1,
        device="cpu",
    )
    assert policy.config.action_mask_version == "colonist-multiagent-v1"

    path = tmp_path / "xdim_graph.pt"
    policy.save(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["action_mask_version"] == "colonist-multiagent-v1"
    assert checkpoint["static_action_features_sha256"]

    loaded = XDimGraphPolicy.load(path, device="cpu")
    env = ColonistMultiAgentEnv(config)
    try:
        observations, info = env.reset(seed=11)
        bad_info = dict(info)
        bad_info["action_mask_version"] = "wrong-version"
        with pytest.raises(ValueError, match="action_mask_version"):
            loaded.select_action(
                env,
                observations[info["current_player"]],
                bad_info,
                __import__("numpy").random.default_rng(12),
            )
    finally:
        env.close()


def test_xdim_checkpoint_missing_metadata_fails_closed_but_legacy_load_allows(tmp_path):
    torch = pytest.importorskip("torch")

    policy = XDimLitePolicy(
        XDimLiteConfig(
            observation_size=4,
            action_size=3,
            static_action_feature_size=2,
            context_action_feature_size=1,
            hidden_size=8,
            action_mask_version="colonist-multiagent-v1",
        ),
        np.zeros((3, 2), dtype=np.float32),
        device="cpu",
    )
    path = tmp_path / "xdim_lite.pt"
    policy.save(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint.pop("action_mask_version", None)
    checkpoint.pop("static_action_features_sha256", None)
    # Task #74: save() now writes the config as a name-keyed dict; blank the
    # version in whichever form this checkpoint carries (the dict form here,
    # the pickled-dataclass form for pre-#74 checkpoints).
    config = checkpoint["config"]
    if isinstance(config, dict):
        config["fields"]["action_mask_version"] = ""
    else:
        object.__setattr__(config, "action_mask_version", "")
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="missing XDim action_mask_version"):
        XDimLitePolicy.load(path, device="cpu")

    loaded = XDimLitePolicy.load(path, device="cpu", strict_metadata=False)
    assert loaded.action_size == 3


def test_policy_observation_array_uses_policy_normalizer():
    class Policy:
        def normalize_observation_array(self, observations):
            assert observations.shape == (2, 3)
            return np.full_like(observations, 7.0, dtype=np.float32)

    class Sample:
        def __init__(self, observation):
            self.observation = observation

    samples = [
        Sample(np.asarray([1.0, 2.0, 3.0], dtype=np.float32)),
        Sample(np.asarray([4.0, 5.0, 6.0], dtype=np.float32)),
    ]

    normalized = _policy_observation_array(Policy(), samples)
    assert normalized.dtype == np.float32
    assert np.all(normalized == 7.0)


def test_train_bc_rejects_checkpoint_action_mask_version_mismatch():
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    config = make_env_config(players=2, vps_to_win=3)
    policy = XDimGraphPolicy.create(
        env_config=config,
        hidden_size=32,
        token_count=8,
        board_layers=1,
        device="cpu",
    )
    object.__setattr__(policy.config, "action_mask_version", "old-version")

    data = {
        "obs": __import__("numpy").zeros((1, policy.config.observation_size), dtype="float32"),
        "legal_action_ids": __import__("numpy").array([[0]], dtype="int16"),
        "legal_action_context": __import__("numpy").zeros(
            (1, 1, policy.context_action_feature_size),
            dtype="float32",
        ),
        "action_taken": __import__("numpy").array([0], dtype="int16"),
        "action_mask_version": __import__("numpy").array(["colonist-multiagent-v1"]),
    }

    with pytest.raises(SystemExit, match="checkpoint action_mask_version"):
        validate_teacher_data_schema(
            policy,
            data,
            {"invalid_teacher_actions": 0},
            config,
        )


def test_target_columns_rejects_duplicate_legal_action_ids():
    legal = np.asarray([[2, 2, 3]], dtype=np.int16)
    actions = np.asarray([2], dtype=np.int16)

    with pytest.raises(ValueError, match="duplicate legal action ids"):
        _target_columns(legal, actions)


def test_validate_teacher_data_schema_rejects_duplicate_legal_action_ids():
    policy = XDimLitePolicy(
        XDimLiteConfig(
            observation_size=4,
            action_size=4,
            static_action_feature_size=2,
            context_action_feature_size=1,
            hidden_size=8,
            action_mask_version="",
        ),
        np.zeros((4, 2), dtype=np.float32),
        device="cpu",
    )
    data = {
        "obs": np.zeros((1, 4), dtype=np.float32),
        "legal_action_ids": np.asarray([[2, 2]], dtype=np.int16),
        "legal_action_context": np.zeros((1, 2, 1), dtype=np.float32),
        "action_taken": np.asarray([2], dtype=np.int16),
        "action_mask_version": np.asarray([""]),
    }

    with pytest.raises(SystemExit, match="duplicate legal action ids"):
        validate_teacher_data_schema(
            policy,
            data,
            {"invalid_teacher_actions": 0},
            make_env_config(players=2, vps_to_win=3),
        )


def test_soft_target_log_probs_penalize_unscored_legal_logits():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[0.0, 0.0, 10.0]], dtype=torch.float32)
    support = torch.tensor([[True, True, False]])

    log_probs = _support_log_softmax(logits, support)

    expected = torch.nn.functional.log_softmax(logits, dim=-1)
    assert torch.allclose(log_probs, expected)
    assert float(log_probs[0, 0]) < -9.0
    assert float(log_probs[0, 1]) < -9.0
    assert float(log_probs[0, 2]) > -1.0e-3


def test_soft_target_log_probs_fall_back_when_support_empty():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[0.0, 0.0, 10.0]], dtype=torch.float32)
    support = torch.tensor([[False, False, False]])

    log_probs = _support_log_softmax(logits, support)

    expected = torch.nn.functional.log_softmax(logits, dim=-1)
    assert torch.allclose(log_probs, expected)


def test_train_bc_rejects_static_action_feature_hash_mismatch():
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    config = make_env_config(players=2, vps_to_win=3)
    policy = XDimGraphPolicy.create(
        env_config=config,
        hidden_size=32,
        token_count=8,
        board_layers=1,
        device="cpu",
    )
    policy.static_action_features[0, 0] += 1.0

    data = {
        "obs": np.zeros((1, policy.config.observation_size), dtype=np.float32),
        "legal_action_ids": np.asarray([[0]], dtype=np.int16),
        "legal_action_context": np.zeros(
            (1, 1, policy.context_action_feature_size),
            dtype=np.float32,
        ),
        "action_taken": np.asarray([0], dtype=np.int16),
        "action_mask_version": np.asarray(["colonist-multiagent-v1"]),
    }

    with pytest.raises(SystemExit, match="static_action_features_sha256"):
        validate_teacher_data_schema(
            policy,
            data,
            {"invalid_teacher_actions": 0},
            config,
        )


def test_scoreboard_policy_env_compatibility_rejects_static_hash_mismatch():
    pytest.importorskip("catanatron")
    pytest.importorskip("gymnasium")

    config = make_env_config(players=2, vps_to_win=3)
    policy = XDimGraphPolicy.create(
        env_config=config,
        hidden_size=32,
        token_count=8,
        board_layers=1,
        device="cpu",
    )
    policy.static_action_features[0, 0] += 1.0

    with pytest.raises(ValueError, match="static_action_features_sha256"):
        assert_policy_compatible_with_env(policy, config)
