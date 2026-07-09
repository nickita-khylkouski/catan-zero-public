"""Tests for tools/phase_sliced_holdout_calibration.py (task #78).

Combines value_repair_calibration_probe.collect_holdout_rows' game-seed-range
filtering (the correct held-out set -- excludes rows the checkpoint actually
trained on, unlike phase_sliced_value_calibration.py's collect_rows which
reads a whole shard-dir with no held-out filtering) with
phase_sliced_value_calibration's phase/legal-count slicing and calibration
stats (reused directly, not duplicated).
"""
from __future__ import annotations

import numpy as np

from tools.phase_sliced_holdout_calibration import (
    collect_holdout_rows_with_slices,
    resolve_use_masking,
)


class TestResolveUseMasking:
    def test_explicit_true_wins_regardless_of_policy(self):
        class FakePolicy:
            trained_with_masked_hidden_info = False

        assert resolve_use_masking(True, FakePolicy()) is True

    def test_explicit_false_wins_regardless_of_policy(self):
        class FakePolicy:
            trained_with_masked_hidden_info = True

        assert resolve_use_masking(False, FakePolicy()) is False

    def test_auto_detects_masked_trained_policy(self):
        class FakePolicy:
            trained_with_masked_hidden_info = True

        assert resolve_use_masking(None, FakePolicy()) is True

    def test_auto_detects_omniscient_trained_policy(self):
        class FakePolicy:
            trained_with_masked_hidden_info = False

        assert resolve_use_masking(None, FakePolicy()) is False

    def test_fails_closed_to_unmasked_for_legacy_checkpoint_missing_attribute(self):
        class LegacyPolicy:
            pass

        assert resolve_use_masking(None, LegacyPolicy()) is False


class TestCollectHoldoutRowsWithSlices:
    def test_adds_phase_and_forced_and_legal_count_columns(self, tmp_path):
        shard_path = tmp_path / "shard_0.npz"
        n = 10
        legal_action_mask = np.zeros((n, 5), dtype=bool)
        legal_action_mask[:, :3] = True  # 3 legal actions/row
        np.savez(
            shard_path,
            game_seed=np.arange(1000, 1000 + n),
            terminated=np.ones(n, dtype=bool),
            truncated=np.zeros(n, dtype=bool),
            winner=np.array(["RED"] * n),
            player=np.array(["RED"] * 5 + ["BLUE"] * 5),
            phase=np.array(["BUILD_INITIAL_SETTLEMENT"] * 5 + ["PLAY_TURN"] * 5),
            is_forced=np.array([True] * 3 + [False] * 7),
            legal_action_mask=legal_action_mask,
            hex_tokens=np.zeros((n, 1)),
            hex_vertex_ids=np.zeros((n, 1)),
            hex_edge_ids=np.zeros((n, 1)),
            vertex_tokens=np.zeros((n, 1)),
            edge_tokens=np.zeros((n, 1)),
            edge_vertex_ids=np.zeros((n, 1)),
            player_tokens=np.zeros((n, 1)),
            global_tokens=np.zeros((n, 1)),
            legal_action_tokens=np.zeros((n, 1)),
            legal_action_target_ids=np.zeros((n, 1)),
            event_tokens=np.zeros((n, 1)),
            event_target_ids=np.zeros((n, 1)),
            hex_mask=np.zeros((n, 1)),
            vertex_mask=np.zeros((n, 1)),
            edge_mask=np.zeros((n, 1)),
            player_mask=np.zeros((n, 1)),
            event_mask=np.zeros((n, 1)),
            legal_action_ids=np.zeros((n, 5)),
            legal_action_context=np.zeros((n, 5, 1)),
        )
        # Fake a manifest so _iter_holdout_shards (reused from the probe
        # module) can find this one shard.
        import json
        (tmp_path / "manifest.json").write_text(json.dumps({"shards": [str(shard_path)]}))

        groups = collect_holdout_rows_with_slices(((str(tmp_path), 1000, 1010),))
        assert len(groups) == 1
        group = groups[0]
        assert "phase_label" in group
        assert "forced" in group
        assert "legal_count" in group
        assert list(group["legal_count"]) == [3] * n
        assert group["forced"].sum() == 3
        # opening-placement rows (BUILD_INITIAL_SETTLEMENT) mapped correctly
        assert (group["phase_label"][:5] == "opening_placement").all()
        assert (group["phase_label"][5:] == "play_turn").all()

    def test_respects_game_seed_range_filter(self, tmp_path):
        shard_path = tmp_path / "shard_0.npz"
        n = 10
        legal_action_mask = np.ones((n, 2), dtype=bool)
        np.savez(
            shard_path,
            game_seed=np.arange(0, n),  # seeds 0..9
            terminated=np.ones(n, dtype=bool),
            truncated=np.zeros(n, dtype=bool),
            winner=np.array(["RED"] * n),
            player=np.array(["RED"] * n),
            phase=np.array(["PLAY_TURN"] * n),
            is_forced=np.zeros(n, dtype=bool),
            legal_action_mask=legal_action_mask,
            hex_tokens=np.zeros((n, 1)), hex_vertex_ids=np.zeros((n, 1)),
            hex_edge_ids=np.zeros((n, 1)), vertex_tokens=np.zeros((n, 1)),
            edge_tokens=np.zeros((n, 1)), edge_vertex_ids=np.zeros((n, 1)),
            player_tokens=np.zeros((n, 1)), global_tokens=np.zeros((n, 1)),
            legal_action_tokens=np.zeros((n, 1)), legal_action_target_ids=np.zeros((n, 1)),
            event_tokens=np.zeros((n, 1)), event_target_ids=np.zeros((n, 1)),
            hex_mask=np.zeros((n, 1)), vertex_mask=np.zeros((n, 1)),
            edge_mask=np.zeros((n, 1)), player_mask=np.zeros((n, 1)),
            event_mask=np.zeros((n, 1)), legal_action_ids=np.zeros((n, 2)),
            legal_action_context=np.zeros((n, 2, 1)),
        )
        import json
        (tmp_path / "manifest.json").write_text(json.dumps({"shards": [str(shard_path)]}))

        # only seeds [3, 7) should survive
        groups = collect_holdout_rows_with_slices(((str(tmp_path), 3, 7),))
        assert len(groups) == 1
        assert len(groups[0]["z"]) == 4
