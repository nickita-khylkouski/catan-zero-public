from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from catan_zero.rl import gumbel_self_play as self_play
from catan_zero.search import gumbel_chance_mcts as gumbel_search
from catan_zero.rl.gumbel_self_play import (
    SEARCH_EVIDENCE_VERSION,
    _compact_search_evidence,
    _rows_to_arrays,
    _search_evidence_recalibration_scope,
    search_evidence_for_row,
)
from catan_zero.search.gumbel_chance_mcts import (
    GumbelChanceMCTS,
    GumbelChanceMCTSConfig,
    SearchResult,
    _GNode,
    _prune_policy_target,
)


def _rows() -> list[dict]:
    return [
        {
            "legal_action_ids": np.asarray([2, 5, 9], dtype=np.int16),
            "policy_weight_multiplier": np.float32(1.0),
            "simulations_used": np.int32(6),
            "_search_visit_counts": np.asarray([4, 0, 2], dtype=np.int64),
            "_search_completed_q": np.asarray([0.4, 0.1, -0.2], dtype=np.float32),
            "_search_prior_policy": np.asarray(
                [0.60001, 0.29999, 0.1], dtype=np.float32
            ),
        },
        {
            "legal_action_ids": np.asarray([3, 4], dtype=np.int16),
            "policy_weight_multiplier": np.float32(0.0),
            "simulations_used": np.int32(2),
        },
        {
            "legal_action_ids": np.asarray([1, 7], dtype=np.int16),
            "policy_weight_multiplier": np.float32(1.0),
            "simulations_used": np.int32(8),
            "_search_visit_counts": np.asarray([3, 5], dtype=np.int64),
            "_search_completed_q": np.asarray([0.25, 0.5], dtype=np.float32),
            "_search_prior_policy": np.asarray([0.45, 0.55], dtype=np.float32),
        },
    ]


def test_compact_search_evidence_exact_layout_and_byte_cost() -> None:
    evidence = _compact_search_evidence(_rows())

    assert int(evidence["search_evidence_version"]) == SEARCH_EVIDENCE_VERSION
    assert evidence["search_evidence_offsets"].tolist() == [0, 3, 5]
    assert evidence["search_visit_counts_flat"].dtype == np.uint16
    assert evidence["search_visit_counts_flat"].tolist() == [4, 0, 2, 3, 5]
    assert evidence["search_completed_q_flat"].dtype == np.float32
    assert evidence["search_completed_q_flat"].tolist() == pytest.approx(
        [0.4, 0.1, -0.2, 0.25, 0.5]
    )
    assert evidence["search_prior_policy_flat"].dtype == np.float32
    assert evidence["search_prior_policy_flat"].tolist() == pytest.approx(
        [0.60001, 0.29999, 0.1, 0.45, 0.55]
    )

    # Exact raw array payload: uint8 version + uint32 offsets + uint16 visits
    # + fp32 completed-Q + reconstruction-grade fp32 prior =
    # 1 + 4(A+1) + 10L bytes. Active
    # rows need no IDs:
    # the mandatory policy_weight_multiplier column already identifies them.
    active_rows, legal_entries = 2, 5
    expected = 1 + 4 * (active_rows + 1) + 10 * legal_entries
    assert sum(array.nbytes for array in evidence.values()) == expected == 63


def test_search_evidence_decoder_aligns_to_existing_legal_axis() -> None:
    shard = _compact_search_evidence(_rows())
    shard["legal_action_ids"] = np.asarray(
        [[2, 5, 9], [3, 4, -1], [1, 7, -1]], dtype=np.int16
    )
    shard["policy_weight_multiplier"] = np.asarray([1.0, 0.0, 1.0], dtype=np.float32)

    first = search_evidence_for_row(shard, 0)
    assert first is not None
    assert first["legal_action_ids"].tolist() == [2, 5, 9]
    assert first["visit_counts"].tolist() == [4, 0, 2]
    assert first["completed_q"].tolist() == pytest.approx([0.4, 0.1, -0.2])
    assert first["prior_policy"].tolist() == pytest.approx(
        [0.60001, 0.29999, 0.1]
    )
    assert search_evidence_for_row(shard, 1) is None


def test_fp32_evidence_reconstructs_stored_target_with_production_operator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rust_actions = tuple(range(10, 15))
    policy_actions = tuple(range(5))
    phase = "BUILD_INITIAL_ROAD"
    visits = np.asarray([7, 2, 0, 5, 4], dtype=np.int64)
    # Deliberately retain precision beyond fp32. The evidence contract promises
    # reconstruction-grade fp32 parity under a tolerance, not byte identity to
    # Python's intermediate float operator.
    source_prior = np.asarray(
        [
            0.22707878371,
            0.17123692713,
            0.17324985917,
            0.17219699329,
            0.25623743670,
        ],
        dtype=np.float64,
    )
    source_prior /= source_prior.sum()
    source_completed_q = np.asarray(
        [-0.0018890137, -0.0004221871, -0.0001747739, 0.0002136431, 0.0002173247],
        dtype=np.float64,
    )
    config = GumbelChanceMCTSConfig(
        c_visit=31.0,
        c_scale=0.17,
        prior_temperature=0.73,
        sigma_reference_visits=None,
        rescale_noise_floor_c=0.65,
        sigma_eval=0.52,
        rescale_noise_floor_initial_road_only=True,
        policy_target_min_visits=3,
    )
    # This pure operator parity test does not execute a game/tree traversal.
    monkeypatch.setattr(gumbel_search, "_require_rust_module", lambda: None)
    mcts = GumbelChanceMCTS(config)

    def _production_target(
        actions: tuple[int, ...],
        prior: np.ndarray,
        completed_q: np.ndarray,
        action_visits: np.ndarray,
    ) -> tuple[dict[int, float], _GNode]:
        node = _GNode(game=None, root_color="RED", root_phase=phase)
        mcts._finish_expand(
            node,
            actions,
            action_json_by_id={},
            spectrum_by_id={},
            priors={
                action: float(probability)
                for action, probability in zip(actions, prior)
            },
            value=0.0,
        )
        for action, count in zip(actions, action_visits):
            node.actions[action].visits = int(count)
        improved = mcts._improved_policy(
            node,
            {
                action: float(value)
                for action, value in zip(actions, completed_q)
            },
        )
        return (
            _prune_policy_target(
                improved,
                {action: int(count) for action, count in zip(actions, action_visits)},
                min_visits=config.policy_target_min_visits,
            ),
            node,
        )

    production_target, production_node = _production_target(
        rust_actions, source_prior, source_completed_q, visits
    )
    assert mcts._sigma_scale(production_node) == pytest.approx(
        (config.c_visit + int(visits.max())) * config.c_scale
    )
    for action in rust_actions:
        assert production_node.action_logits[action] == pytest.approx(
            np.log(production_node.actions[action].prior) / config.prior_temperature
        )
    raw_completed_q = {
        action: float(value)
        for action, value in zip(rust_actions, source_completed_q)
    }
    assert mcts._rescaled_completed_q(
        production_node, raw_completed_q
    ) != mcts._rescale_completed_q(raw_completed_q)

    class _Game:
        def current_color(self):
            return "RED"

        def json_snapshot(self):
            return json.dumps({"current_prompt": phase})

        def playable_action_indices(self, _colors, _filter):
            return list(rust_actions)

        def playable_actions_json(self):
            return json.dumps([["RED", "BUILD_ROAD"]] * len(rust_actions))

    monkeypatch.setattr(
        self_play, "rust_policy_action_ids", lambda *_args, **_kwargs: policy_actions
    )
    monkeypatch.setattr(
        self_play, "_resolve_entity_adapter", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        self_play,
        "rust_game_to_entity_batch",
        lambda *_args, **_kwargs: {
            "player_tokens": np.zeros((1, 4, 31), dtype=np.float16)
        },
    )
    monkeypatch.setattr(
        self_play,
        "rust_action_context_batch",
        lambda *_args, **_kwargs: np.zeros(
            (1, len(rust_actions), 18), dtype=np.float32
        ),
    )
    result = SearchResult(
        selected_action=max(production_target, key=production_target.get),
        improved_policy=production_target,
        visit_counts={
            action: int(count) for action, count in zip(rust_actions, visits)
        },
        q_values={
            action: float(value)
            for action, value, count in zip(
                rust_actions, source_completed_q, visits
            )
            if count > 0
        },
        priors={
            action: production_node.actions[action].prior for action in rust_actions
        },
        root_value=0.0,
        used_full_search=True,
        simulations_used=int(visits.sum()),
        completed_q_values={
            action: float(value)
            for action, value in zip(rust_actions, source_completed_q)
        },
    )
    row, _features = self_play._build_decision_row(
        _Game(),
        result=result,
        action_size=len(policy_actions),
        colors=("RED", "BLUE"),
        game_seed=7,
        decision_index=0,
        obs_width=1,
    )

    shard = _compact_search_evidence([row])
    shard["legal_action_ids"] = row["legal_action_ids"][None, :]
    shard["policy_weight_multiplier"] = np.asarray([1.0], dtype=np.float32)
    decoded = search_evidence_for_row(shard, 0)
    assert decoded is not None
    reconstructed, _node = _production_target(
        tuple(int(action) for action in decoded["legal_action_ids"]),
        decoded["prior_policy"],
        decoded["completed_q"],
        decoded["visit_counts"],
    )
    reconstructed_array = np.asarray(
        [reconstructed[int(action)] for action in decoded["legal_action_ids"]],
        dtype=np.float64,
    )

    # Tight enough to catch float16 prior regression while acknowledging the
    # intentional fp64 -> fp32 evidence boundary.
    reconstruction_atol = 2.0e-7
    reconstruction_rtol = 2.0e-6
    np.testing.assert_allclose(
        reconstructed_array,
        row["target_policy"],
        atol=reconstruction_atol,
        rtol=reconstruction_rtol,
    )
    assert row["target_policy"].tolist().count(0.0) == 2
    legacy_reconstructed, _node = _production_target(
        tuple(int(action) for action in decoded["legal_action_ids"]),
        row["prior_policy"].astype(np.float32),
        decoded["completed_q"],
        decoded["visit_counts"],
    )
    legacy_array = np.asarray(
        [
            legacy_reconstructed[int(action)]
            for action in decoded["legal_action_ids"]
        ],
        dtype=np.float64,
    )
    assert not np.allclose(
        legacy_array,
        row["target_policy"],
        atol=reconstruction_atol,
        rtol=reconstruction_rtol,
    )


def test_search_evidence_fails_closed_on_lossy_or_misaligned_inputs() -> None:
    rows = _rows()
    rows[0]["_search_completed_q"][1] = np.nan
    with pytest.raises(ValueError, match="non-finite completed-Q"):
        _compact_search_evidence(rows)

    rows = _rows()
    rows[0]["_search_visit_counts"][0] = np.iinfo(np.uint16).max + 1
    with pytest.raises(ValueError, match="uint16"):
        _compact_search_evidence(rows)

    rows = _rows()
    rows[0]["simulations_used"] = np.int32(7)
    with pytest.raises(ValueError, match="simulations_used"):
        _compact_search_evidence(rows)

    rows = _rows()
    rows[0]["policy_weight_multiplier"] = np.float32(np.nan)
    with pytest.raises(ValueError, match="finite non-negative policy weight"):
        _compact_search_evidence(rows)

    rows = _rows()
    rows[0]["_search_prior_policy"][0] = np.nan
    with pytest.raises(ValueError, match="invalid fp32 prior-policy"):
        _compact_search_evidence(rows)

    rows = _rows()
    rows[0]["_search_prior_policy"][:] = 0.0
    with pytest.raises(ValueError, match="invalid fp32 prior-policy"):
        _compact_search_evidence(rows)

    shard = _compact_search_evidence(_rows())
    shard["legal_action_ids"] = np.asarray([[2, 5, 9], [3, -1, -1], [3, 4, -1]])
    shard["policy_weight_multiplier"] = np.asarray([0.75, 0.0, 1.0])
    shard["search_prior_policy_flat"][:3] = 0.0
    with pytest.raises(ValueError, match="malformed fp32 prior-policy"):
        search_evidence_for_row(shard, 0)


def test_absent_search_evidence_is_backward_compatible() -> None:
    assert search_evidence_for_row({"legal_action_ids": np.asarray([[1]])}, 0) is None


def test_version_one_search_evidence_remains_decodable() -> None:
    shard = {
        "search_evidence_version": np.asarray(1, dtype=np.uint8),
        "search_evidence_offsets": np.asarray([0, 2], dtype=np.uint32),
        "search_visit_counts_flat": np.asarray([3, 1], dtype=np.uint16),
        "search_completed_q_flat": np.asarray([0.2, 0.05], dtype=np.float32),
        "legal_action_ids": np.asarray([[4, 7]], dtype=np.int16),
        "policy_weight_multiplier": np.asarray([1.0], dtype=np.float32),
    }
    decoded = search_evidence_for_row(shard, 0)
    assert decoded is not None
    assert set(decoded) == {"legal_action_ids", "visit_counts", "completed_q"}

    shard["search_prior_policy_flat"] = np.asarray([0.5, 0.5], dtype=np.float32)
    with pytest.raises(ValueError, match="unversioned fp32 prior"):
        search_evidence_for_row(shard, 0)


def test_search_evidence_recalibration_scope_fails_closed_for_mean_of_policies() -> None:
    assert (
        _search_evidence_recalibration_scope(
            SimpleNamespace(information_set_search=False)
        )
        == "single_world_root_v1"
    )
    assert (
        _search_evidence_recalibration_scope(
            SimpleNamespace(
                information_set_search=True,
                information_set_target_aggregation="aggregate_q_then_improve",
            )
        )
        == "information_set_aggregate_q_then_improve_v1"
    )
    with pytest.raises(ValueError, match="mean_improved_policy"):
        _search_evidence_recalibration_scope(
            SimpleNamespace(
                information_set_search=True,
                information_set_target_aggregation="mean_improved_policy",
            )
        )


def test_writer_keeps_private_evidence_out_of_default_rows(tmp_path) -> None:
    row = _rows()[0]
    features = {
        key: np.asarray([0.0], dtype=np.float32) for key in self_play.ENTITY_KEYS
    }

    default = self_play.GumbelShardWriter(tmp_path / "default", shard_size=99)
    default.add(row, features)
    assert "_search_visit_counts" not in default.rows[0]
    assert "_search_completed_q" not in default.rows[0]
    assert "_search_prior_policy" not in default.rows[0]

    enabled = self_play.GumbelShardWriter(
        tmp_path / "enabled", shard_size=99, preserve_search_evidence=True
    )
    enabled.add(row, features)
    assert enabled.rows[0]["_search_visit_counts"] is row["_search_visit_counts"]
    assert enabled.rows[0]["_search_completed_q"] is row["_search_completed_q"]
    assert enabled.rows[0]["_search_prior_policy"] is row["_search_prior_policy"]


def test_optional_arrays_are_the_only_default_schema_difference() -> None:
    rows = _rows()
    for row in rows:
        width = int(row["legal_action_ids"].size)
        for key in self_play.ENTITY_KEYS:
            if key == "legal_action_tokens":
                row[key] = np.zeros((width, 1), dtype=np.float16)
            elif key == "legal_action_target_ids":
                row[key] = np.full((width, 4), -1, dtype=np.int16)
            elif key == "legal_action_mask":
                row[key] = np.ones((width,), dtype=np.bool_)
            else:
                row[key] = np.zeros((1,), dtype=np.float16)

    historical = _rows_to_arrays(rows)
    enabled = _rows_to_arrays(rows, preserve_search_evidence=True)

    optional_keys = {
        "search_evidence_version",
        "search_evidence_offsets",
        "search_visit_counts_flat",
        "search_completed_q_flat",
        "search_prior_policy_flat",
    }
    assert optional_keys.isdisjoint(historical)
    assert set(enabled) == set(historical) | optional_keys
    for key, historical_value in historical.items():
        np.testing.assert_array_equal(enabled[key], historical_value)


@pytest.mark.parametrize(
    ("used_full_search", "expected_policy_weight"),
    ((True, 1.0), (False, 0.0)),
)
def test_decision_row_preserves_active_search_evidence_in_legal_order(
    monkeypatch: pytest.MonkeyPatch,
    used_full_search: bool,
    expected_policy_weight: float,
) -> None:
    class _Game:
        def current_color(self):
            return "RED"

        def json_snapshot(self):
            return json.dumps({"current_prompt": "PLAY_TURN"})

        def playable_action_indices(self, _colors, _filter):
            return [10, 11]

        def playable_actions_json(self):
            return json.dumps([["RED", "END_TURN"], ["RED", "ROLL"]])

    monkeypatch.setattr(
        self_play, "rust_policy_action_ids", lambda *_args, **_kwargs: (3, 4)
    )
    # This test isolates persisted search evidence, not adapter construction.
    # The canonical learner-feature path now resolves the public adapter once
    # before calling the two featurizers mocked below.
    monkeypatch.setattr(
        self_play, "_resolve_entity_adapter", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        self_play,
        "rust_game_to_entity_batch",
        lambda *_args, **_kwargs: {
            "player_tokens": np.zeros((1, 4, 31), dtype=np.float16)
        },
    )
    monkeypatch.setattr(
        self_play,
        "rust_action_context_batch",
        lambda *_args, **_kwargs: np.zeros((1, 2, 18), dtype=np.float32),
    )
    row, _features = self_play._build_decision_row(
        _Game(),
        result=SearchResult(
            selected_action=10,
            improved_policy={10: 1.0, 11: 0.0},
            visit_counts={10: 3, 11: 1},
            q_values={10: 0.2, 11: -0.1},
            priors={10: 0.6, 11: 0.4},
            root_value=0.1,
            used_full_search=used_full_search,
            simulations_used=4,
            completed_q_values={10: 0.2, 11: 0.05},
        ),
        action_size=8,
        colors=("RED", "BLUE"),
        game_seed=7,
        decision_index=0,
        obs_width=1,
    )

    assert float(row["policy_weight_multiplier"]) == pytest.approx(
        expected_policy_weight
    )
    if used_full_search:
        assert row["_search_visit_counts"].tolist() == [3, 1]
        assert row["_search_completed_q"].tolist() == pytest.approx([0.2, 0.05])
        assert row["_search_prior_policy"].tolist() == pytest.approx([0.6, 0.4])
    else:
        assert "_search_visit_counts" not in row
        assert "_search_completed_q" not in row
        assert "_search_prior_policy" not in row
    assert row["target_policy"].tolist() == pytest.approx([1.0, 0.0])
    assert row["target_policy_mask"].tolist() == [True, True]
