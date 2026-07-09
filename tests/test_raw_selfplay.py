from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from catan_zero.rl.action_mask import ActionCatalog
from catan_zero.rl.raw_selfplay import (
    COLORS,
    RawSelfPlayConfig,
    TARGET_SCORE_SOURCE,
    TEACHER_NAME,
    play_one_raw_selfplay_game,
    run_raw_selfplay_worker_games,
)
from catan_zero.search.gumbel_chance_mcts import HeuristicRustEvaluator
from catan_zero.search.rust_mcts import _require_rust_module

# `tools/train_bc.py` does bare sibling imports (`from factory_common import ...`),
# so it only works with the `tools/` directory itself on sys.path (matches the
# established pattern in tests/test_gumbel_self_play.py).
_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from train_bc import _load_npz, _normalize_teacher_shard  # type: ignore  # noqa: E402


def _rust():
    try:
        return _require_rust_module()
    except RuntimeError as error:
        pytest.skip(str(error))


# ---------------------------------------------------------------------------
# Core design contract: NO search is ever invoked -- policy_weight_multiplier
# must be exactly 0 on every row (forced or not), value_weight_multiplier
# exactly 1 on every row.
# ---------------------------------------------------------------------------


def test_every_row_has_zero_policy_weight_and_full_value_weight():
    _rust()
    config = RawSelfPlayConfig(max_decisions=20, temperature_decisions=45, temperature=1.0)
    evaluator = HeuristicRustEvaluator(score_actions=False)

    record = play_one_raw_selfplay_game(
        evaluator, config=config, game_seed=11, game_index=0,
        action_size=ActionCatalog(COLORS).size, seed=3,
    )

    assert record.decisions, "expected at least one recorded decision"
    for decision in record.decisions:
        assert float(decision.row["policy_weight_multiplier"]) == pytest.approx(0.0)
        assert float(decision.row["value_weight_multiplier"]) == pytest.approx(1.0)
        assert decision.row["used_full_search"] is False
        assert int(decision.row["simulations_used"]) == 0


def test_target_policy_sums_to_one_and_matches_prior_policy():
    _rust()
    config = RawSelfPlayConfig(max_decisions=10, temperature_decisions=45, temperature=1.0)
    evaluator = HeuristicRustEvaluator(score_actions=True)

    record = play_one_raw_selfplay_game(
        evaluator, config=config, game_seed=21, game_index=0,
        action_size=ActionCatalog(COLORS).size, seed=5,
    )

    assert record.decisions
    for decision in record.decisions:
        row = decision.row
        mask = row["target_policy_mask"]
        total = float(np.asarray(row["target_policy"])[mask].sum())
        assert total == pytest.approx(1.0, abs=1e-3)
        # No search means target IS the prior -- both fields carry the same
        # distribution (fp32 vs fp16 storage only).
        prior = np.asarray(row["prior_policy"]).astype(np.float64)
        target = np.asarray(row["target_policy"]).astype(np.float64)
        assert np.allclose(prior, target, atol=2e-3)


def test_no_search_means_no_scores_or_afterstate_targets():
    _rust()
    config = RawSelfPlayConfig(max_decisions=10)
    evaluator = HeuristicRustEvaluator(score_actions=False)

    record = play_one_raw_selfplay_game(
        evaluator, config=config, game_seed=31, game_index=0,
        action_size=ActionCatalog(COLORS).size, seed=1,
    )

    assert record.decisions
    for decision in record.decisions:
        row = decision.row
        assert not bool(np.any(row["target_scores_mask"]))
        assert not bool(np.any(row["afterstate_target_mask"]))
        assert row["target_score_source"] == TARGET_SCORE_SOURCE


def test_teacher_name_is_raw_selfplay():
    _rust()
    config = RawSelfPlayConfig(max_decisions=6)
    evaluator = HeuristicRustEvaluator(score_actions=False)

    record = play_one_raw_selfplay_game(
        evaluator, config=config, game_seed=41, game_index=0,
        action_size=ActionCatalog(COLORS).size, seed=2,
    )

    assert record.decisions
    for decision in record.decisions:
        assert decision.row["teacher_name"] == TEACHER_NAME


# ---------------------------------------------------------------------------
# Temperature schedule: sample for diversity in the first N decisions, argmax
# (deterministic) thereafter.
# ---------------------------------------------------------------------------


def test_argmax_phase_is_deterministic_across_different_worker_seeds():
    _rust()
    # temperature_decisions=0 forces argmax from decision 0 onward.
    config = RawSelfPlayConfig(max_decisions=10, temperature_decisions=0, temperature=1.0)
    evaluator = HeuristicRustEvaluator(score_actions=True)

    actions_by_seed = []
    for worker_seed in (1, 2, 3):
        record = play_one_raw_selfplay_game(
            evaluator, config=config, game_seed=51, game_index=0,
            action_size=ActionCatalog(COLORS).size, seed=worker_seed,
        )
        actions_by_seed.append([int(d.row["action_taken"]) for d in record.decisions])

    assert actions_by_seed[0] == actions_by_seed[1] == actions_by_seed[2]


def test_temperature_phase_can_diversify_action_selection():
    _rust()
    # temperature_decisions >= max_decisions keeps every decision in the
    # sampling phase; a real (non-forced, multi-way) evaluator should
    # eventually diverge across worker seeds within a handful of games.
    config = RawSelfPlayConfig(max_decisions=8, temperature_decisions=8, temperature=1.0)
    evaluator = HeuristicRustEvaluator(score_actions=True)

    trajectories = set()
    for worker_seed in range(8):
        record = play_one_raw_selfplay_game(
            evaluator, config=config, game_seed=61, game_index=0,
            action_size=ActionCatalog(COLORS).size, seed=worker_seed,
        )
        trajectories.add(tuple(int(d.row["action_taken"]) for d in record.decisions))

    assert len(trajectories) > 1, "expected temperature sampling to produce varied trajectories"


# ---------------------------------------------------------------------------
# Truncated games must not fabricate outcome labels (same contract as the
# searched driver).
# ---------------------------------------------------------------------------


def test_truncated_game_produces_no_outcome_labels():
    _rust()
    config = RawSelfPlayConfig(max_decisions=3)
    evaluator = HeuristicRustEvaluator(score_actions=False)

    record = play_one_raw_selfplay_game(
        evaluator, config=config, game_seed=71, game_index=0,
        action_size=ActionCatalog(COLORS).size, seed=1,
    )

    assert record.truncated is True
    assert record.terminal is False
    assert record.winner == ""
    for decision in record.decisions:
        assert decision.row["winner"] == ""
        assert decision.row["truncated"] is True
        assert decision.row["terminated"] is False
        assert decision.row["has_final_public_vps"] is False
        assert decision.row["has_final_actual_vps"] is False


# ---------------------------------------------------------------------------
# Shard writing + schema round-trip through train_bc.py's loader.
# ---------------------------------------------------------------------------


def test_run_worker_games_writes_valid_shards_that_round_trip_through_train_bc(tmp_path):
    _rust()
    config = RawSelfPlayConfig(max_decisions=6)
    evaluator = HeuristicRustEvaluator(score_actions=False)

    summary = run_raw_selfplay_worker_games(
        out_dir=tmp_path / "out", games=2, game_index_start=0,
        base_seed=500, worker_seed=3, config=config, evaluator=evaluator,
    )

    assert summary["games_completed"] == 2
    assert summary["games_failed"] == 0
    assert summary["shards"], "expected at least one shard to be written"

    total_rows = 0
    for shard_path in summary["shards"]:
        path = Path(shard_path)
        assert path.exists()
        raw = _load_npz(path)
        normalized = _normalize_teacher_shard(raw, path)
        n = len(normalized["action_taken"])
        total_rows += n
        assert normalized["obs"].shape[0] == n
        assert normalized["legal_action_context"].shape[0] == n
        assert normalized["legal_action_context"].ndim == 3
        assert normalized["legal_action_ids"].shape[0] == n
        assert normalized["target_policy"].shape == normalized["legal_action_ids"].shape
        assert normalized["target_policy"].dtype == np.float32
        assert "policy_weight_multiplier" in normalized
        assert "value_weight_multiplier" in normalized
        assert np.all(np.asarray(raw["policy_weight_multiplier"]) == 0.0)
        assert np.all(np.asarray(raw["value_weight_multiplier"]) == 1.0)
        assert set(np.asarray(raw["teacher_name"]).astype(str).tolist()) == {TEACHER_NAME}

    assert total_rows == int(summary["rows"])


def test_run_worker_games_writes_a_real_manifest_json_with_config_provenance(tmp_path):
    _rust()
    config = RawSelfPlayConfig(max_decisions=6)
    evaluator = HeuristicRustEvaluator(score_actions=False)
    out_dir = tmp_path / "out"

    summary = run_raw_selfplay_worker_games(
        out_dir=out_dir, games=1, game_index_start=0,
        base_seed=42, worker_seed=7, config=config, evaluator=evaluator,
    )

    manifest_path = out_dir / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["games_completed"] == summary["games_completed"]
    assert manifest["rows"] == summary["rows"]
    assert manifest["shards"] == summary["shards"]
    # Config provenance (982d344 pattern) -- what the worker actually
    # constructed, for post-hoc audit.
    assert "selfplay_config" in manifest
    assert manifest["selfplay_config"]["max_decisions"] == 6
    for shard_path in manifest["shards"]:
        assert Path(shard_path).exists()


def test_run_worker_games_isolates_one_bad_game_from_the_worker(tmp_path, monkeypatch):
    _rust()
    config = RawSelfPlayConfig(max_decisions=4)
    evaluator = HeuristicRustEvaluator(score_actions=False)

    import catan_zero.rl.raw_selfplay as raw_selfplay

    real_play = raw_selfplay.play_one_raw_selfplay_game
    call_count = {"n": 0}

    def _flaky(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("synthetic failure for isolation test")
        return real_play(*args, **kwargs)

    monkeypatch.setattr(raw_selfplay, "play_one_raw_selfplay_game", _flaky)

    summary = run_raw_selfplay_worker_games(
        out_dir=tmp_path / "out", games=2, game_index_start=0,
        base_seed=900, worker_seed=1, config=config, evaluator=evaluator,
    )

    assert summary["games_failed"] == 1
    assert summary["games_completed"] == 1
    assert len(summary["errors"]) == 1
    assert "synthetic failure" in summary["errors"][0]["error"]


# ---------------------------------------------------------------------------
# CLI worker-level error isolation (mirrors test_gumbel_self_play.py's
# equivalent -- `_worker_entry` must never raise, or one worker's crash
# discards every OTHER worker's already-written shards from the batch).
# ---------------------------------------------------------------------------


def test_worker_entry_never_raises_and_reports_a_failed_worker(monkeypatch):
    from tools.generate_raw_selfplay_data import _worker_entry
    import tools.generate_raw_selfplay_data as cli_module

    def _boom(_worker_args):
        raise RuntimeError("synthetic checkpoint load failure")

    monkeypatch.setattr(cli_module, "_run_worker", _boom)

    result = _worker_entry({"worker_index": 3, "out_dir": "/nonexistent", "games": 2})

    assert result["worker_index"] == 3
    assert result["games_completed"] == 0
    assert result["games_failed"] == 2
    assert result["shards"] == []
    assert result["errors"]
    assert "synthetic checkpoint load failure" in result["errors"][0]["error"]
