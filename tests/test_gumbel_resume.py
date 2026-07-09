"""Unit tests for the Modal GPU factory's incremental-resume fix.

No Modal SDK, no CUDA, no compiled `catanatron_rs` Rust engine is required:

- `resolve_part_resume_action` (tools/gumbel_factory_resume.py) is pure
  stdlib logic, deliberately split out of `modal_gumbel_factory_gpu.py` (which
  pulls in `modal` at import time) so it can be exercised directly here.
- `run_worker_games`'s incremental-resume bookkeeping is exercised end to end
  by monkeypatching `play_one_game` (so no Rust engine is needed) and
  `_require_rust_module` (so `GumbelChanceMCTS.__init__` doesn't need the
  compiled wheel either). A real OS process + SIGKILL is used to simulate a
  Modal preemption mid-game-8, matching the actual failure mode (an abrupt
  kill that skips Python's `finally` blocks entirely, so anything not
  already flushed to disk before the kill is genuinely gone) far more
  faithfully than raising a Python exception would.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from catan_zero.rl import gumbel_self_play as gsp  # noqa: E402
from catan_zero.search import gumbel_chance_mcts as gcm  # noqa: E402
from catan_zero.search.gumbel_chance_mcts import GumbelChanceMCTSConfig  # noqa: E402
from gumbel_factory_resume import resolve_part_resume_action  # noqa: E402

DECISIONS_PER_GAME = 4
SHARD_SIZE = DECISIONS_PER_GAME  # exactly one shard flushed per completed game


class _StubEvaluator:
    """No `.policy` attribute -> action_size_for_evaluator falls back to
    `ActionCatalog`, which is pure Python (no Rust engine needed)."""


def _fake_decision(game_seed: int, decision_index: int) -> gsp.DecisionRecord:
    row = {
        "obs": np.zeros((4,), dtype=np.float16),
        "legal_action_ids": np.asarray([0, 1], dtype=np.int16),
        "legal_action_context": np.zeros((2, 2), dtype=np.float16),
        "action_taken": np.int16(0),
        "target_policy": np.asarray([0.5, 0.5], dtype=np.float32),
        "target_scores": np.asarray([0.1, 0.2], dtype=np.float32),
        "target_policy_mask": np.asarray([True, True]),
        "target_scores_mask": np.asarray([True, True]),
        "target_score_source": "test",
        "game_seed": np.int64(game_seed),
        "teacher_name": "test",
        "player": "RED",
        "seat": np.int8(0),
        "phase": "ROLL",
        "decision_index": np.int32(decision_index),
        "winner": "",
        "terminated": False,
        "truncated": False,
        "final_public_vps": np.zeros(4, dtype=np.int16),
        "has_final_public_vps": False,
        "final_actual_vps": np.zeros(4, dtype=np.int16),
        "has_final_actual_vps": False,
        "action_mask_version": "v1",
        "policy_weight_multiplier": np.float32(1.0),
        "value_weight_multiplier": np.float32(1.0),
        "used_full_search": True,
        "is_forced": False,
        "simulations_used": np.int32(1),
        "afterstate_target": np.asarray([0.1, 0.2], dtype=np.float32),
        "afterstate_target_mask": np.asarray([True, True]),
        "prior_policy": np.asarray([0.5, 0.5], dtype=np.float16),
        "adapter_version": "v1",
    }
    features = {
        "hex_tokens": np.zeros((3, 2), dtype=np.float16),
        "hex_vertex_ids": np.zeros((3, 3), dtype=np.int16),
        "hex_edge_ids": np.zeros((3, 3), dtype=np.int16),
        "vertex_tokens": np.zeros((4, 2), dtype=np.float16),
        "edge_tokens": np.zeros((4, 2), dtype=np.float16),
        "edge_vertex_ids": np.zeros((4, 2), dtype=np.int16),
        "player_tokens": np.zeros((2, 2), dtype=np.float16),
        "global_tokens": np.zeros((1, 2), dtype=np.float16),
        "legal_action_tokens": np.zeros((2, 2), dtype=np.float16),
        "legal_action_target_ids": np.zeros((2, 4), dtype=np.int16),
        "event_tokens": np.zeros((1, 2), dtype=np.float16),
        "event_target_ids": np.zeros((1, 4), dtype=np.int16),
        "hex_mask": np.ones((3,), dtype=bool),
        "vertex_mask": np.ones((4,), dtype=bool),
        "edge_mask": np.ones((4,), dtype=bool),
        "player_mask": np.ones((2,), dtype=bool),
        "legal_action_mask": np.ones((2,), dtype=bool),
        "event_mask": np.ones((1,), dtype=bool),
    }
    return gsp.DecisionRecord(row=row, features=features)


def _fake_game_record(game_seed: int, game_index: int, colors: tuple[str, ...]) -> gsp.GameRecord:
    decisions = [_fake_decision(game_seed, i) for i in range(DECISIONS_PER_GAME)]
    winner = colors[0]
    outcome = {
        "winner": winner,
        "terminated": True,
        "truncated": False,
        "final_public_vps": np.asarray([10, 5, 0, 0], dtype=np.int16),
        "has_final_public_vps": True,
        "final_actual_vps": np.asarray([10, 5, 0, 0], dtype=np.int16),
        "has_final_actual_vps": True,
    }
    for decision in decisions:
        decision.row.update(outcome)
    return gsp.GameRecord(
        game_seed=game_seed,
        game_index=game_index,
        decisions=decisions,
        terminal=True,
        truncated=False,
        winner=winner,
        total_decisions=DECISIONS_PER_GAME,
        forced_decisions=0,
        simulations_used_total=DECISIONS_PER_GAME,
        wall_time_sec=0.0,
    )


def _stub_config() -> gsp.GumbelSelfPlayConfig:
    return gsp.GumbelSelfPlayConfig(obs_width=4)


def _child_run(out_dir: str, games: int, base_seed: int, run_id: str, sentinel: str) -> None:
    """Runs in a separate OS process; killed (SIGKILL) by the parent test
    right after it signals it has started game index 7 (the 8th game).
    """
    import gc

    gc.disable()  # irrelevant to correctness, just avoids GC noise before the kill

    def _stub_play_one_game(mcts, evaluator, *, config, game_seed, game_index, action_size, **_kwargs):
        if game_index == 7:
            Path(sentinel).write_text("started", encoding="utf-8")
            time.sleep(120)  # parent SIGKILLs us long before this returns
        return _fake_game_record(game_seed, game_index, config.colors)

    gsp.play_one_game = _stub_play_one_game
    gcm._require_rust_module = lambda: None

    gsp.run_worker_games(
        out_dir=Path(out_dir),
        games=games,
        game_index_start=0,
        base_seed=base_seed,
        worker_seed=1,
        config=_stub_config(),
        search_config=GumbelChanceMCTSConfig(),
        evaluator=_StubEvaluator(),
        shard_size=SHARD_SIZE,
        fmt="npz",
        run_id=run_id,
        resume=True,
    )


def test_incremental_resume_after_simulated_preemption(tmp_path, monkeypatch):
    out_dir = tmp_path / "worker_000"
    out_dir.mkdir()
    base_seed = 1_000_000
    total_games = 8
    run_id = "run-A"

    sentinel = str(tmp_path / "_game7_started")
    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(
        target=_child_run,
        args=(str(out_dir), total_games, base_seed, run_id, sentinel),
    )
    proc.start()
    deadline = time.time() + 20.0
    while not Path(sentinel).exists():
        assert proc.is_alive(), "child died before reaching game 7"
        assert time.time() < deadline, "child never reached game 7 (sentinel never appeared)"
        time.sleep(0.02)

    # Give the child a brief moment to be solidly inside time.sleep(), then
    # SIGKILL it -- an abrupt, no-cleanup kill, exactly like Modal preempting
    # a container. No `finally`/`atexit` runs; only what was already synced
    # to disk before this instant survives.
    time.sleep(0.1)
    os.kill(proc.pid, 9)
    proc.join(timeout=10)
    assert not proc.is_alive()

    manifest_path = out_dir / "manifest.json"
    assert not manifest_path.exists(), "worker must not have finished"

    progress = gsp._load_worker_progress(out_dir)
    assert progress is not None
    assert progress.run_id == run_id
    # Games 0-6 (7 games) had exactly enough rows to fill 7 shards
    # (SHARD_SIZE == DECISIONS_PER_GAME), so all 7 are durably confirmed
    # before game 7 (offset 7, the 8th game) even started.
    assert progress.games_completed_local == 7
    assert progress.shard_count_confirmed == 7

    shard_files_before = sorted(p.name for p in out_dir.glob("gumbel_self_play_shard_*.npz"))
    assert len(shard_files_before) == 7

    # ---- resume ----
    seen_game_indices: list[int] = []

    def _resume_stub_play_one_game(mcts, evaluator, *, config, game_seed, game_index, action_size, **_kwargs):
        seen_game_indices.append(game_index)
        return _fake_game_record(game_seed, game_index, config.colors)

    monkeypatch.setattr(gsp, "play_one_game", _resume_stub_play_one_game)
    monkeypatch.setattr(gcm, "_require_rust_module", lambda: None)

    summary = gsp.run_worker_games(
        out_dir=out_dir,
        games=total_games,
        game_index_start=0,
        base_seed=base_seed,
        worker_seed=1,
        config=_stub_config(),
        search_config=GumbelChanceMCTSConfig(),
        evaluator=_StubEvaluator(),
        shard_size=SHARD_SIZE,
        fmt="npz",
        run_id=run_id,
        resume=True,
    )

    # Games 0-6 must NOT be regenerated; only game 7 (the interrupted one)
    # is replayed.
    assert seen_game_indices == [7]
    assert summary["resumed_from_offset"] == 7
    assert summary["games_completed"] == total_games
    assert summary["rows"] == total_games * DECISIONS_PER_GAME
    assert len(summary["shards"]) == total_games  # 7 pre-existing + 1 new

    # No duplicate/missing game_seed rows anywhere on disk.
    all_game_seeds: list[int] = []
    for shard_path in sorted(out_dir.glob("gumbel_self_play_shard_*.npz")):
        with np.load(shard_path) as data:
            all_game_seeds.extend(int(v) for v in data["game_seed"])

    expected_seeds = [base_seed + offset for offset in range(total_games)]
    assert sorted(all_game_seeds) == sorted(expected_seeds * DECISIONS_PER_GAME)
    for seed in expected_seeds:
        assert all_game_seeds.count(seed) == DECISIONS_PER_GAME

    # Final manifest.json now exists and is internally consistent.
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["games_completed"] == total_games
    assert len(manifest["shards"]) == total_games


def test_resume_is_a_noop_when_no_progress_file_exists(tmp_path, monkeypatch):
    """resume=True with nothing to resume from behaves exactly like resume=False."""
    out_dir = tmp_path / "worker_000"
    seen: list[int] = []

    def _stub(mcts, evaluator, *, config, game_seed, game_index, action_size, **_kwargs):
        seen.append(game_index)
        return _fake_game_record(game_seed, game_index, config.colors)

    monkeypatch.setattr(gsp, "play_one_game", _stub)
    monkeypatch.setattr(gcm, "_require_rust_module", lambda: None)

    summary = gsp.run_worker_games(
        out_dir=out_dir,
        games=3,
        game_index_start=0,
        base_seed=500,
        worker_seed=1,
        config=_stub_config(),
        search_config=GumbelChanceMCTSConfig(),
        evaluator=_StubEvaluator(),
        shard_size=SHARD_SIZE,
        fmt="npz",
        run_id="run-B",
        resume=True,
    )
    assert seen == [0, 1, 2]
    assert summary["resumed_from_offset"] == 0
    assert summary["games_completed"] == 3


def test_seed_formula_is_independent_of_resume(tmp_path, monkeypatch):
    """game_seed = base_seed + game_index_start + offset, unaffected by resume."""
    out_dir = tmp_path / "worker_000"
    base_seed = 42
    game_index_start = 17

    def _stub(mcts, evaluator, *, config, game_seed, game_index, action_size, **_kwargs):
        assert game_seed == base_seed + game_index
        assert game_index == game_index_start + (game_index - game_index_start)
        return _fake_game_record(game_seed, game_index, config.colors)

    monkeypatch.setattr(gsp, "play_one_game", _stub)
    monkeypatch.setattr(gcm, "_require_rust_module", lambda: None)

    gsp.run_worker_games(
        out_dir=out_dir,
        games=2,
        game_index_start=game_index_start,
        base_seed=base_seed,
        worker_seed=1,
        config=_stub_config(),
        search_config=GumbelChanceMCTSConfig(),
        evaluator=_StubEvaluator(),
        shard_size=SHARD_SIZE,
        fmt="npz",
        run_id="run-C",
        resume=False,
    )

    progress = gsp._load_worker_progress(out_dir)
    assert progress.game_index_start == game_index_start
    assert progress.base_seed == base_seed

    # Resuming (no-op here, already complete + manifest exists at the
    # caller level in the real factory -- but run_worker_games itself
    # doesn't special-case a complete progress file) must derive identical
    # seeds for any offsets it does replay.
    def _stub2(mcts, evaluator, *, config, game_seed, game_index, action_size, **_kwargs):
        assert game_seed == base_seed + game_index_start + (game_index - game_index_start)
        return _fake_game_record(game_seed, game_index, config.colors)

    monkeypatch.setattr(gsp, "play_one_game", _stub2)
    gsp.run_worker_games(
        out_dir=out_dir,
        games=2,
        game_index_start=game_index_start,
        base_seed=base_seed,
        worker_seed=1,
        config=_stub_config(),
        search_config=GumbelChanceMCTSConfig(),
        evaluator=_StubEvaluator(),
        shard_size=SHARD_SIZE,
        fmt="npz",
        run_id="run-C",
        resume=True,
    )


# --------------------------------------------------------------------------
# Factory-level resume/wipe/hard-error decision (tools/gumbel_factory_resume.py)
# --------------------------------------------------------------------------


def _touch(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_resolve_part_resume_action_fresh_part(tmp_path):
    part_dir = tmp_path / "part_00000"
    action, complete = resolve_part_resume_action(
        part_dir=part_dir,
        manifest_path=part_dir / "manifest.json",
        marker_path=part_dir / ".run_id",
        run_id="run-A",
        resume=False,
    )
    assert action == "fresh"
    assert complete is None


def test_resolve_part_resume_action_returns_complete_manifest(tmp_path):
    part_dir = tmp_path / "part_00000"
    manifest_path = part_dir / "manifest.json"
    _touch(manifest_path, json.dumps({"run_id": "run-A", "games_completed": 500}))
    action, complete = resolve_part_resume_action(
        part_dir=part_dir,
        manifest_path=manifest_path,
        marker_path=part_dir / ".run_id",
        run_id="run-A",
        resume=False,
    )
    assert action == "return_complete"
    assert complete == {"run_id": "run-A", "games_completed": 500}


def test_resolve_part_resume_action_same_run_id_preemption_is_incremental_resume(tmp_path):
    """The core bug fix: a same-run_id retry on an INCOMPLETE part must
    request incremental resume, never wipe."""
    part_dir = tmp_path / "part_00000"
    marker_path = part_dir / ".run_id"
    _touch(marker_path, "run-A")
    _touch(part_dir / "gumbel_self_play_shard_00000.npz", "fake-shard-bytes")

    action, complete = resolve_part_resume_action(
        part_dir=part_dir,
        manifest_path=part_dir / "manifest.json",  # doesn't exist: incomplete
        marker_path=marker_path,
        run_id="run-A",
        resume=False,
    )
    assert action == "incremental_resume"
    assert complete is None
    # Nothing must have been deleted.
    assert marker_path.exists()
    assert (part_dir / "gumbel_self_play_shard_00000.npz").exists()


def test_resolve_part_resume_action_different_run_id_hard_errors(tmp_path):
    """The duplicate-launch guard (seed-collision incident) must be untouched."""
    part_dir = tmp_path / "part_00000"
    marker_path = part_dir / ".run_id"
    _touch(marker_path, "run-OLD")
    _touch(part_dir / "gumbel_self_play_shard_00000.npz", "fake-shard-bytes")

    with pytest.raises(RuntimeError, match="different run_id"):
        resolve_part_resume_action(
            part_dir=part_dir,
            manifest_path=part_dir / "manifest.json",
            marker_path=marker_path,
            run_id="run-NEW",
            resume=False,
        )


def test_resolve_part_resume_action_explicit_resume_wipes_foreign_incomplete_part(tmp_path):
    """Operator-explicit resume=True on a DIFFERENT run_id's incomplete part
    is still a deliberate wipe-and-restart (unchanged, distinct from the
    automatic same-run_id incremental-resume path)."""
    part_dir = tmp_path / "part_00000"
    marker_path = part_dir / ".run_id"
    _touch(marker_path, "run-OLD")
    _touch(part_dir / "gumbel_self_play_shard_00000.npz", "fake-shard-bytes")

    action, complete = resolve_part_resume_action(
        part_dir=part_dir,
        manifest_path=part_dir / "manifest.json",
        marker_path=marker_path,
        run_id="run-NEW",
        resume=True,
    )
    assert action == "wipe_and_restart"
    assert complete is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
