from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from catan_zero.rl import gumbel_self_play as self_play
from catan_zero.rl.gumbel_self_play import (
    SEARCH_EVIDENCE_VERSION,
    _compact_search_evidence,
    _rows_to_arrays,
    _search_evidence_recalibration_scope,
    search_evidence_for_row,
)
from catan_zero.search.gumbel_chance_mcts import SearchResult


def _rows() -> list[dict]:
    return [
        {
            "legal_action_ids": np.asarray([2, 5, 9], dtype=np.int16),
            "policy_weight_multiplier": np.float32(1.0),
            "simulations_used": np.int32(6),
            "_search_visit_counts": np.asarray([4, 0, 2], dtype=np.int64),
            "_search_completed_q": np.asarray([0.4, 0.1, -0.2], dtype=np.float32),
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

    # Exact raw array payload: uint8 version + uint32 offsets + uint16 visits
    # + fp32 completed-Q = 1 + 4(A+1) + 6L bytes. Active rows need no IDs:
    # the mandatory policy_weight_multiplier column already identifies them.
    active_rows, legal_entries = 2, 5
    expected = 1 + 4 * (active_rows + 1) + 6 * legal_entries
    assert sum(array.nbytes for array in evidence.values()) == expected == 43


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
    assert search_evidence_for_row(shard, 1) is None


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


def test_absent_search_evidence_is_backward_compatible() -> None:
    assert search_evidence_for_row({"legal_action_ids": np.asarray([[1]])}, 0) is None


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

    enabled = self_play.GumbelShardWriter(
        tmp_path / "enabled", shard_size=99, preserve_search_evidence=True
    )
    enabled.add(row, features)
    assert enabled.rows[0]["_search_visit_counts"] is row["_search_visit_counts"]
    assert enabled.rows[0]["_search_completed_q"] is row["_search_completed_q"]


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
    else:
        assert "_search_visit_counts" not in row
        assert "_search_completed_q" not in row
    assert row["target_policy"].tolist() == pytest.approx([1.0, 0.0])
    assert row["target_policy_mask"].tolist() == [True, True]
