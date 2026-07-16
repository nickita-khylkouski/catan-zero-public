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
import random
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
RESUME_SEMANTICS_SHA256 = "sha256:" + "a" * 64


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
        gsp.AUX_SUBGOAL_TARGET_VERSION_KEY: np.uint8(gsp.AUX_SUBGOAL_TARGET_VERSION),
    }
    features = {
        "hex_tokens": np.zeros((3, 2), dtype=np.float16),
        "hex_vertex_ids": np.zeros((3, 3), dtype=np.int16),
        "hex_edge_ids": np.zeros((3, 3), dtype=np.int16),
        "vertex_tokens": np.zeros((4, 2), dtype=np.float16),
        "edge_tokens": np.zeros((4, 2), dtype=np.float16),
        "edge_vertex_ids": np.zeros((4, 2), dtype=np.int16),
        "player_tokens": np.zeros((2, 2), dtype=np.float16),
        "deduction_features": np.zeros((4, 11), dtype=np.float32),
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


def _fake_game_record(
    game_seed: int, game_index: int, colors: tuple[str, ...]
) -> gsp.GameRecord:
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


def _child_run(
    out_dir: str,
    games: int,
    base_seed: int,
    run_id: str,
    sentinel: str,
    shard_size: int = SHARD_SIZE,
    pause_game_index: int = 7,
) -> None:
    """Runs in a separate OS process; killed (SIGKILL) by the parent test
    right after it signals it has started game index 7 (the 8th game).
    """
    import gc

    gc.disable()  # irrelevant to correctness, just avoids GC noise before the kill

    def _stub_play_one_game(
        mcts, evaluator, *, config, game_seed, game_index, action_size, **_kwargs
    ):
        if game_index == pause_game_index:
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
        shard_size=shard_size,
        fmt="npz",
        run_id=run_id,
        resume=True,
        resume_semantics_sha256=RESUME_SEMANTICS_SHA256,
    )


def _child_pause_after_final_close(
    out_dir: str, games: int, base_seed: int, run_id: str, sentinel: str
) -> None:
    """Publish the final partial shard, then pause before progress advances."""

    def _stub_play_one_game(
        mcts, evaluator, *, config, game_seed, game_index, action_size, **_kwargs
    ):
        return _fake_game_record(game_seed, game_index, config.colors)

    original_close = gsp.GumbelShardWriter.close

    def _close_then_pause(writer):
        original_close(writer)
        Path(sentinel).write_text("closed", encoding="utf-8")
        time.sleep(120)

    gsp.play_one_game = _stub_play_one_game
    gsp.GumbelShardWriter.close = _close_then_pause
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
        shard_size=100,
        fmt="npz",
        run_id=run_id,
        resume=True,
        resume_semantics_sha256=RESUME_SEMANTICS_SHA256,
    )


def _mix_choice(game_index: int, _categories):
    from catan_zero.rl.flywheel.opponent_mix import MixChoice

    slot = int(game_index) % 6
    if slot in {2, 3}:
        return MixChoice(
            tag="external",
            is_pool=True,
            path="",
            version=-1,
            md5="",
            engine="catanatron_value",
        )
    if slot in {1, 5}:
        return MixChoice(
            tag="older",
            is_pool=True,
            path="/unused/older.pt",
            version=2,
            md5="abc",
        )
    return MixChoice(tag="self", is_pool=False, path="", version=-1, md5="")


def _mix_runtime() -> gsp.MixRuntime:
    from catan_zero.rl.flywheel.opponent_mix import (
        MixCategory,
        MixCheckpointRef,
        OpponentMixConfig,
    )

    return gsp.MixRuntime(
        config=OpponentMixConfig(
            categories=(
                MixCategory(name="self", weight=2, source="self"),
                MixCategory(
                    name="older",
                    weight=2,
                    source="checkpoint_list",
                    checkpoints=(
                        MixCheckpointRef(path="/unused/older.pt", version=2, md5="abc"),
                    ),
                ),
                MixCategory(
                    name="external",
                    weight=2,
                    source="external_engine",
                    engine="catanatron_value",
                ),
            )
        ),
        evaluator_factory=lambda _path: _StubEvaluator(),
    )


def _mix_neural_game(
    _mcts,
    _evaluator,
    *,
    config,
    game_seed,
    game_index,
    pool_assignment,
    pause_sentinel: str | None = None,
    **_kwargs,
):
    if game_index == 4:
        raise RuntimeError("deterministic mix failure")
    if game_index == 5 and pause_sentinel is not None:
        Path(pause_sentinel).write_text("started", encoding="utf-8")
        time.sleep(120)
    record = _fake_game_record(game_seed, game_index, config.colors)
    winner = pool_assignment.champion_color
    record.winner = winner
    for decision in record.decisions:
        decision.row["winner"] = winner
    return record


def _mix_exploiter_game(*, config, game_seed, game_index, champion_first, **_kwargs):
    if game_index == 2:
        return gsp.GameRecord(
            game_seed=game_seed,
            game_index=game_index,
            decisions=[],
            terminal=False,
            truncated=False,
            winner="",
            total_decisions=0,
            forced_decisions=0,
            simulations_used_total=0,
            wall_time_sec=0.0,
            engine_divergence=True,
            divergence_topic="road",
        )
    record = _fake_game_record(game_seed, game_index, config.colors)
    winner = config.colors[0] if champion_first else config.colors[1]
    record.winner = winner
    for decision in record.decisions:
        decision.row["winner"] = winner
    return record


def _child_run_mix(out_dir: str, base_seed: int, run_id: str, sentinel: str) -> None:
    from catan_zero.rl import exploiter_lockstep

    gsp.choose_mix_opponent = _mix_choice
    gsp.play_one_game = lambda *args, **kwargs: _mix_neural_game(
        *args, **kwargs, pause_sentinel=sentinel
    )
    exploiter_lockstep.play_one_exploiter_game = _mix_exploiter_game
    gcm._require_rust_module = lambda: None
    gsp.run_worker_games(
        out_dir=Path(out_dir),
        games=6,
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
        resume_semantics_sha256=RESUME_SEMANTICS_SHA256,
        opponent_mix=_mix_runtime(),
    )


def _rng_game_record(
    mcts,
    _evaluator,
    *,
    config,
    game_seed,
    game_index,
    pause_sentinel: str | None = None,
    **_kwargs,
):
    if pause_sentinel is not None and game_index == 3:
        Path(pause_sentinel).write_text("started", encoding="utf-8")
        time.sleep(120)
    marker = mcts.rng.getrandbits(64)
    record = _fake_game_record(game_seed, game_index, config.colors)
    for decision in record.decisions:
        decision.row["target_score_source"] = f"rng:{marker}"
    return record


def _child_run_rng(
    out_dir: str, base_seed: int, run_id: str, sentinel: str
) -> None:
    gsp.play_one_game = lambda *args, **kwargs: _rng_game_record(
        *args, **kwargs, pause_sentinel=sentinel
    )
    gcm._require_rust_module = lambda: None
    gsp.run_worker_games(
        out_dir=Path(out_dir),
        games=4,
        game_index_start=0,
        base_seed=base_seed,
        worker_seed=777,
        config=_stub_config(),
        search_config=GumbelChanceMCTSConfig(seed=777),
        evaluator=_StubEvaluator(),
        shard_size=SHARD_SIZE,
        fmt="npz",
        run_id=run_id,
        resume=True,
        resume_semantics_sha256=RESUME_SEMANTICS_SHA256,
    )


def _row_identities(out_dir: Path) -> list[tuple[int, int]]:
    identities: list[tuple[int, int]] = []
    for shard_path in sorted(out_dir.glob("gumbel_self_play_shard_*.npz")):
        with np.load(shard_path) as data:
            identities.extend(
                zip(
                    map(int, data["game_seed"]),
                    map(int, data["decision_index"]),
                )
            )
    return identities


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
        assert time.time() < deadline, (
            "child never reached game 7 (sentinel never appeared)"
        )
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
    assert progress.aux_subgoal_target_version == gsp.AUX_SUBGOAL_TARGET_VERSION
    assert progress.aux_subgoal_target_semantic == gsp.AUX_SUBGOAL_TARGET_SEMANTIC
    # Games 0-6 (7 games) had exactly enough rows to fill 7 shards
    # (SHARD_SIZE == DECISIONS_PER_GAME), so all 7 are durably confirmed
    # before game 7 (offset 7, the 8th game) even started.
    assert progress.games_completed_local == 7
    assert progress.shard_count_confirmed == 7

    shard_files_before = sorted(
        p.name for p in out_dir.glob("gumbel_self_play_shard_*.npz")
    )
    assert len(shard_files_before) == 7

    # ---- resume ----
    seen_game_indices: list[int] = []

    def _resume_stub_play_one_game(
        mcts, evaluator, *, config, game_seed, game_index, action_size, **_kwargs
    ):
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
        resume_semantics_sha256=RESUME_SEMANTICS_SHA256,
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


def test_misaligned_shard_size_never_splits_or_duplicates_replayed_game(
    tmp_path, monkeypatch
):
    """Regression: a 6-row target may not split a 4-row game at row 2."""

    out_dir = tmp_path / "worker_000"
    out_dir.mkdir()
    base_seed = 2_000_000
    total_games = 3
    run_id = "run-misaligned"
    sentinel = str(tmp_path / "_game2_started")
    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(
        target=_child_run,
        args=(
            str(out_dir),
            total_games,
            base_seed,
            run_id,
            sentinel,
            6,
            2,
        ),
    )
    proc.start()
    deadline = time.time() + 20.0
    while not Path(sentinel).exists():
        assert proc.is_alive(), "child died before reaching game 2"
        assert time.time() < deadline, "child never reached game 2"
        time.sleep(0.02)
    os.kill(proc.pid, 9)
    proc.join(timeout=10)
    assert not proc.is_alive()

    progress = gsp._load_worker_progress(out_dir)
    assert progress is not None
    # Two complete four-row games share one game-atomic eight-row shard.  The
    # old row-wise writer flushed at row six, retaining half of game 1.
    assert progress.games_completed_local == 2
    assert progress.games_succeeded == 2
    assert progress.shard_count_confirmed == 1
    assert progress.rows_confirmed == 8
    before = _row_identities(out_dir)
    assert len(before) == 8
    assert {seed for seed, _decision in before} == {
        base_seed,
        base_seed + 1,
    }

    seen: list[int] = []

    def _resume_stub(
        mcts, evaluator, *, config, game_seed, game_index, action_size, **_kwargs
    ):
        seen.append(game_index)
        return _fake_game_record(game_seed, game_index, config.colors)

    monkeypatch.setattr(gsp, "play_one_game", _resume_stub)
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
        shard_size=6,
        fmt="npz",
        run_id=run_id,
        resume=True,
        resume_semantics_sha256=RESUME_SEMANTICS_SHA256,
    )

    assert seen == [2]
    assert summary["resumed_from_offset"] == 2
    assert summary["games_completed"] == 3
    assert summary["games_failed"] == 0
    assert summary["rows"] == 12
    identities = _row_identities(out_dir)
    expected = [
        (base_seed + game, decision)
        for game in range(total_games)
        for decision in range(DECISIONS_PER_GAME)
    ]
    assert sorted(identities) == sorted(expected)
    assert len(identities) == len(set(identities)) == 12
    # No game identity appears in more than one shard.
    seed_shards: dict[int, set[str]] = {}
    for shard_path in sorted(out_dir.glob("gumbel_self_play_shard_*.npz")):
        with np.load(shard_path) as data:
            for seed in np.unique(data["game_seed"]):
                seed_shards.setdefault(int(seed), set()).add(shard_path.name)
    assert all(len(shards) == 1 for shards in seed_shards.values())
    final_progress = gsp._load_worker_progress(out_dir)
    assert final_progress is not None
    assert final_progress.games_completed_local == 3
    assert final_progress.games_succeeded == 3
    assert final_progress.rows_confirmed == 12


def test_real_mcts_rng_suffix_matches_uninterrupted_after_sigkill(
    tmp_path, monkeypatch
):
    """Resume may not restart the worker RNG stream at the suffix boundary."""

    interrupted = tmp_path / "interrupted-rng"
    uninterrupted = tmp_path / "uninterrupted-rng"
    sentinel = tmp_path / "rng-game3-started"
    base_seed = 31_000
    run_id = "run-rng-parity"
    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(
        target=_child_run_rng,
        args=(str(interrupted), base_seed, run_id, str(sentinel)),
    )
    proc.start()
    deadline = time.time() + 20.0
    while not sentinel.exists():
        assert proc.is_alive(), "RNG child died before game 3"
        assert time.time() < deadline, "RNG child never reached game 3"
        time.sleep(0.02)
    os.kill(proc.pid, 9)
    proc.join(timeout=10)
    assert not proc.is_alive()

    monkeypatch.setattr(gsp, "play_one_game", _rng_game_record)
    monkeypatch.setattr(gcm, "_require_rust_module", lambda: None)
    kwargs = {
        "games": 4,
        "game_index_start": 0,
        "base_seed": base_seed,
        "worker_seed": 777,
        "config": _stub_config(),
        "search_config": GumbelChanceMCTSConfig(seed=777),
        "evaluator": _StubEvaluator(),
        "shard_size": SHARD_SIZE,
        "fmt": "npz",
        "run_id": run_id,
        "resume": True,
        "resume_semantics_sha256": RESUME_SEMANTICS_SHA256,
    }
    resumed = gsp.run_worker_games(out_dir=interrupted, **kwargs)
    complete = gsp.run_worker_games(out_dir=uninterrupted, **kwargs)
    assert resumed["resumed_from_offset"] == 3
    assert complete["resumed_from_offset"] == 0

    def _markers(path: Path) -> dict[int, str]:
        result: dict[int, str] = {}
        for shard in sorted(path.glob("gumbel_self_play_shard_*.npz")):
            with np.load(shard) as data:
                for seed, marker in zip(data["game_seed"], data["target_score_source"]):
                    result[int(seed)] = str(marker)
        return result

    observed = _markers(interrupted)
    assert observed == _markers(uninterrupted)
    for game_index in range(4):
        rng = random.Random(
            gsp._game_search_seed(worker_seed=777, game_index=game_index)
        )
        assert observed[base_seed + game_index] == f"rng:{rng.getrandbits(64)}"


def test_crash_after_final_close_before_progress_replays_whole_partial_shard(
    tmp_path, monkeypatch
):
    out_dir = tmp_path / "worker_000"
    out_dir.mkdir()
    base_seed = 3_000_000
    total_games = 2
    run_id = "run-close-boundary"
    sentinel = str(tmp_path / "_close_finished")
    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(
        target=_child_pause_after_final_close,
        args=(str(out_dir), total_games, base_seed, run_id, sentinel),
    )
    proc.start()
    deadline = time.time() + 20.0
    while not Path(sentinel).exists():
        assert proc.is_alive(), "child died before final close completed"
        assert time.time() < deadline, "child never completed final close"
        time.sleep(0.02)
    os.kill(proc.pid, 9)
    proc.join(timeout=10)
    assert not proc.is_alive()

    # The shard landed, but the last atomic progress marker still authorizes
    # zero rows. Resume must delete this orphan and replay both complete games.
    progress = gsp._load_worker_progress(out_dir)
    assert progress is not None
    assert progress.games_completed_local == 0
    assert progress.rows_confirmed == 0
    assert len(_row_identities(out_dir)) == total_games * DECISIONS_PER_GAME

    seen: list[int] = []

    def _resume_stub(
        mcts, evaluator, *, config, game_seed, game_index, action_size, **_kwargs
    ):
        seen.append(game_index)
        return _fake_game_record(game_seed, game_index, config.colors)

    monkeypatch.setattr(gsp, "play_one_game", _resume_stub)
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
        shard_size=100,
        fmt="npz",
        run_id=run_id,
        resume=True,
        resume_semantics_sha256=RESUME_SEMANTICS_SHA256,
    )

    assert seen == [0, 1]
    assert summary["resumed_from_offset"] == 0
    assert summary["rows"] == 8
    identities = _row_identities(out_dir)
    assert len(identities) == len(set(identities)) == 8
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["games_completed"] == 2
    assert manifest["games_failed"] == 0
    assert manifest["rows"] == 8
    final_progress = gsp._load_worker_progress(out_dir)
    assert final_progress is not None
    assert final_progress.games_completed_local == 2
    assert final_progress.games_succeeded == 2
    assert final_progress.rows_confirmed == 8


def test_resume_is_a_noop_when_no_progress_file_exists(tmp_path, monkeypatch):
    """resume=True with nothing to resume from behaves exactly like resume=False."""
    out_dir = tmp_path / "worker_000"
    seen: list[int] = []

    def _stub(
        mcts, evaluator, *, config, game_seed, game_index, action_size, **_kwargs
    ):
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
        resume_semantics_sha256=RESUME_SEMANTICS_SHA256,
    )
    assert seen == [0, 1, 2]
    assert summary["resumed_from_offset"] == 0
    assert summary["games_completed"] == 3


def test_compressed_game_atomic_shards_authenticate_on_resume(tmp_path, monkeypatch):
    pytest.importorskip("zstandard")
    out_dir = tmp_path / "worker_000"
    seen: list[int] = []

    def _stub(
        mcts, evaluator, *, config, game_seed, game_index, action_size, **_kwargs
    ):
        seen.append(game_index)
        return _fake_game_record(game_seed, game_index, config.colors)

    monkeypatch.setattr(gsp, "play_one_game", _stub)
    monkeypatch.setattr(gcm, "_require_rust_module", lambda: None)
    kwargs = {
        "out_dir": out_dir,
        "games": 2,
        "game_index_start": 0,
        "base_seed": 600,
        "worker_seed": 1,
        "config": _stub_config(),
        "search_config": GumbelChanceMCTSConfig(),
        "evaluator": _StubEvaluator(),
        "shard_size": 6,
        "fmt": "npz_zst",
        "run_id": "run-compressed",
        "resume": True,
        "resume_semantics_sha256": RESUME_SEMANTICS_SHA256,
    }
    first = gsp.run_worker_games(**kwargs)
    assert seen == [0, 1]
    assert first["rows"] == 8
    assert all(str(path).endswith(".npz.zst") for path in first["shards"])

    seen.clear()
    resumed = gsp.run_worker_games(**kwargs)
    assert seen == []
    assert resumed["resumed_from_offset"] == 2
    assert resumed["rows"] == 8
    assert len(resumed["shards"]) == 1

    # A truncated compressed shard must lose authority and trigger a clean
    # deterministic replay, not leak a zstandard/zip exception to the worker.
    retained = Path(resumed["shards"][0])
    retained.write_bytes(retained.read_bytes()[:17])
    seen.clear()
    replayed = gsp.run_worker_games(**kwargs)
    assert seen == [0, 1]
    assert replayed["resumed_from_offset"] == 0
    assert replayed["rows"] == 8


def test_resume_restores_only_confirmed_success_failure_and_search_aggregates(
    tmp_path, monkeypatch
):
    out_dir = tmp_path / "worker_000"

    def _stub(
        mcts, evaluator, *, config, game_seed, game_index, action_size, **_kwargs
    ):
        if game_index == 1:
            raise RuntimeError("deterministic failure")
        return _fake_game_record(game_seed, game_index, config.colors)

    monkeypatch.setattr(gsp, "play_one_game", _stub)
    monkeypatch.setattr(gcm, "_require_rust_module", lambda: None)
    kwargs = {
        "out_dir": out_dir,
        "games": 3,
        "game_index_start": 0,
        "base_seed": 800,
        "worker_seed": 1,
        "config": _stub_config(),
        "search_config": GumbelChanceMCTSConfig(),
        "evaluator": _StubEvaluator(),
        "shard_size": 6,
        "fmt": "npz",
        "run_id": "run-aggregate",
        "resume": True,
        "resume_semantics_sha256": RESUME_SEMANTICS_SHA256,
    }
    first = gsp.run_worker_games(**kwargs)
    assert first["games_completed"] == 2
    assert first["games_failed"] == 1
    assert first["rows"] == 8
    assert first["decisions_total"] == 8
    assert first["simulations_used_total"] == 8
    assert first["wins_by_color"]["RED"] == 2

    def _must_not_replay(*_args, **_kwargs):
        raise AssertionError("complete confirmed offsets must not replay")

    monkeypatch.setattr(gsp, "play_one_game", _must_not_replay)
    resumed = gsp.run_worker_games(**kwargs)
    assert resumed["resumed_from_offset"] == 3
    assert resumed["games_completed"] == 2
    assert resumed["games_failed"] == 1
    assert resumed["rows"] == 8
    assert resumed["decisions_total"] == 8
    assert resumed["simulations_used_total"] == 8
    assert resumed["wins_by_color"]["RED"] == 2


def test_preempted_mix_restores_all_confirmed_scientific_telemetry(
    tmp_path, monkeypatch
):
    """A resumed manifest must equal an uninterrupted mix/exploiter run."""

    from catan_zero.rl import exploiter_lockstep

    base_seed = 9_000
    run_id = "run-mix-telemetry"
    interrupted = tmp_path / "interrupted"
    uninterrupted = tmp_path / "uninterrupted"
    sentinel = tmp_path / "mix-game5-started"
    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(
        target=_child_run_mix,
        args=(str(interrupted), base_seed, run_id, str(sentinel)),
    )
    proc.start()
    deadline = time.time() + 20.0
    while not sentinel.exists():
        assert proc.is_alive(), "mix child died before the preemption point"
        assert time.time() < deadline, "mix child never reached the preemption point"
        time.sleep(0.02)
    os.kill(proc.pid, 9)
    proc.join(timeout=10)
    assert not proc.is_alive()

    progress = gsp._load_worker_progress(interrupted)
    assert progress is not None
    assert progress.games_completed_local == 5
    assert progress.games_succeeded == 4
    assert progress.games_failed == 1
    assert progress.opponent_mix_pool_games == 1
    assert progress.opponent_mix_per_tag_stats == {
        "older": {"games": 1, "champion_wins": 1},
        "self": {"games": 1, "champion_wins": 1},
    }
    assert progress.exploiter_games == 1
    assert progress.exploiter_per_engine_stats == {
        "catanatron_value": {
            "games": 1,
            "champion_wins": 1,
            "divergences": 1,
        }
    }
    assert progress.exploiter_divergence_topics == {"road": 1}
    assert len(progress.errors) == 1

    monkeypatch.setattr(gsp, "choose_mix_opponent", _mix_choice)
    monkeypatch.setattr(gsp, "play_one_game", _mix_neural_game)
    monkeypatch.setattr(
        exploiter_lockstep, "play_one_exploiter_game", _mix_exploiter_game
    )
    monkeypatch.setattr(gcm, "_require_rust_module", lambda: None)
    common = {
        "games": 6,
        "game_index_start": 0,
        "base_seed": base_seed,
        "worker_seed": 1,
        "config": _stub_config(),
        "search_config": GumbelChanceMCTSConfig(),
        "evaluator": _StubEvaluator(),
        "shard_size": SHARD_SIZE,
        "fmt": "npz",
        "run_id": run_id,
        "resume": True,
        "resume_semantics_sha256": RESUME_SEMANTICS_SHA256,
        "opponent_mix": _mix_runtime(),
    }
    resumed = gsp.run_worker_games(out_dir=interrupted, **common)
    complete = gsp.run_worker_games(out_dir=uninterrupted, **common)

    exact_fields = (
        "games_completed",
        "games_failed",
        "games_truncated",
        "wins_by_color",
        "rows",
        "decisions_total",
        "forced_decisions_total",
        "simulations_used_total",
        "errors",
        "opponent_mix_pool_games",
        "opponent_mix_pool_fraction_realized",
        "opponent_mix_per_tag_stats",
        "exploiter_enabled",
        "exploiter_games",
        "exploiter_per_engine_stats",
        "exploiter_divergence_topics",
    )
    assert resumed["resumed_from_offset"] == 5
    assert complete["resumed_from_offset"] == 0
    assert {field: resumed[field] for field in exact_fields} == {
        field: complete[field] for field in exact_fields
    }


def test_same_row_count_shard_substitution_loses_resume_authority(
    tmp_path, monkeypatch
):
    out_dir = tmp_path / "worker_000"
    seen: list[int] = []

    def _stub(_mcts, _evaluator, *, config, game_seed, game_index, **_kwargs):
        seen.append(game_index)
        return _fake_game_record(game_seed, game_index, config.colors)

    monkeypatch.setattr(gsp, "play_one_game", _stub)
    monkeypatch.setattr(gcm, "_require_rust_module", lambda: None)
    kwargs = {
        "out_dir": out_dir,
        "games": 2,
        "game_index_start": 0,
        "base_seed": 10_000,
        "worker_seed": 1,
        "config": _stub_config(),
        "search_config": GumbelChanceMCTSConfig(),
        "evaluator": _StubEvaluator(),
        "shard_size": SHARD_SIZE,
        "fmt": "npz",
        "run_id": "run-shard-tamper",
        "resume": True,
        "resume_semantics_sha256": RESUME_SEMANTICS_SHA256,
    }
    gsp.run_worker_games(**kwargs)
    first, second = sorted(out_dir.glob("gumbel_self_play_shard_*.npz"))
    second.write_bytes(first.read_bytes())
    seen.clear()

    summary = gsp.run_worker_games(**kwargs)
    assert seen == [0, 1]
    assert summary["resumed_from_offset"] == 0
    identities = _row_identities(out_dir)
    assert len(identities) == len(set(identities)) == 8
    assert {seed for seed, _decision in identities} == {10_000, 10_001}


def test_duplicate_confirmed_shard_index_loses_resume_authority(tmp_path, monkeypatch):
    out_dir = tmp_path / "worker_000"
    seen: list[int] = []

    def _stub(_mcts, _evaluator, *, config, game_seed, game_index, **_kwargs):
        seen.append(game_index)
        return _fake_game_record(game_seed, game_index, config.colors)

    monkeypatch.setattr(gsp, "play_one_game", _stub)
    monkeypatch.setattr(gcm, "_require_rust_module", lambda: None)
    kwargs = {
        "out_dir": out_dir,
        "games": 1,
        "game_index_start": 0,
        "base_seed": 11_000,
        "worker_seed": 1,
        "config": _stub_config(),
        "search_config": GumbelChanceMCTSConfig(),
        "evaluator": _StubEvaluator(),
        "shard_size": SHARD_SIZE,
        "fmt": "npz",
        "run_id": "run-duplicate-index",
        "resume": True,
        "resume_semantics_sha256": RESUME_SEMANTICS_SHA256,
    }
    gsp.run_worker_games(**kwargs)
    shard = next(out_dir.glob("gumbel_self_play_shard_*.npz"))
    shard.with_name(shard.name + ".zst").write_bytes(shard.read_bytes())
    seen.clear()

    summary = gsp.run_worker_games(**kwargs)
    assert seen == [0]
    assert summary["resumed_from_offset"] == 0
    assert len(list(out_dir.glob("gumbel_self_play_shard_*.npz*"))) == 1


@pytest.mark.parametrize(
    "mutate",
    (
        lambda value: value.__setitem__(
            "games_succeeded", value["games_succeeded"] + 1
        ),
        lambda value: value.__setitem__(
            "forced_decisions_total", value["decisions_total"] + 1
        ),
        lambda value: value["wins_by_color"].__setitem__("GREEN", 1),
        lambda value: value["confirmed_shards"][0].__setitem__("sha256", "0" * 64),
    ),
    ids=("game-arithmetic", "forced-count", "unknown-color", "inventory-digest"),
)
def test_incoherent_progress_replays_from_zero(tmp_path, monkeypatch, mutate):
    out_dir = tmp_path / "worker_000"

    def _stub(_mcts, _evaluator, *, config, game_seed, game_index, **_kwargs):
        return _fake_game_record(game_seed, game_index, config.colors)

    monkeypatch.setattr(gsp, "play_one_game", _stub)
    monkeypatch.setattr(gcm, "_require_rust_module", lambda: None)
    kwargs = {
        "out_dir": out_dir,
        "games": 1,
        "game_index_start": 0,
        "base_seed": 12_000,
        "worker_seed": 1,
        "config": _stub_config(),
        "search_config": GumbelChanceMCTSConfig(),
        "evaluator": _StubEvaluator(),
        "shard_size": SHARD_SIZE,
        "fmt": "npz",
        "run_id": "run-incoherent",
        "resume": True,
        "resume_semantics_sha256": RESUME_SEMANTICS_SHA256,
    }
    gsp.run_worker_games(**kwargs)
    progress_path = out_dir / gsp.PROGRESS_FILENAME
    value = json.loads(progress_path.read_text(encoding="utf-8"))
    mutate(value)
    progress_path.write_text(json.dumps(value), encoding="utf-8")
    seen: list[int] = []

    def _replay(*args, game_index, config, game_seed, **kwargs):
        seen.append(game_index)
        return _fake_game_record(game_seed, game_index, config.colors)

    monkeypatch.setattr(gsp, "play_one_game", _replay)
    summary = gsp.run_worker_games(**kwargs)
    assert seen == [0]
    assert summary["resumed_from_offset"] == 0


@pytest.mark.parametrize("drift", ("search", "caller"))
def test_generation_semantic_drift_replays_from_zero(tmp_path, monkeypatch, drift):
    out_dir = tmp_path / "worker_000"
    seen: list[int] = []

    def _stub(
        _mcts, _evaluator, *, config, game_seed, game_index, **_kwargs
    ):
        seen.append(game_index)
        return _fake_game_record(game_seed, game_index, config.colors)

    monkeypatch.setattr(gsp, "play_one_game", _stub)
    monkeypatch.setattr(gcm, "_require_rust_module", lambda: None)
    kwargs = {
        "out_dir": out_dir,
        "games": 2,
        "game_index_start": 0,
        "base_seed": 41_000,
        "worker_seed": 9,
        "config": _stub_config(),
        "search_config": GumbelChanceMCTSConfig(seed=9, n_full=128),
        "evaluator": _StubEvaluator(),
        "shard_size": SHARD_SIZE,
        "fmt": "npz",
        "run_id": "run-semantic-drift",
        "resume": True,
        "resume_semantics_sha256": RESUME_SEMANTICS_SHA256,
    }
    gsp.run_worker_games(**kwargs)
    seen.clear()
    if drift == "search":
        kwargs["search_config"] = GumbelChanceMCTSConfig(seed=9, n_full=256)
    else:
        kwargs["resume_semantics_sha256"] = "sha256:" + "b" * 64
    summary = gsp.run_worker_games(**kwargs)

    assert seen == [0, 1]
    assert summary["resumed_from_offset"] == 0


def test_resume_requires_full_caller_semantics_digest(tmp_path):
    with pytest.raises(ValueError, match="requires resume_semantics_sha256"):
        gsp.run_worker_games(
            out_dir=tmp_path,
            games=0,
            game_index_start=0,
            base_seed=1,
            worker_seed=1,
            config=_stub_config(),
            search_config=GumbelChanceMCTSConfig(),
            evaluator=_StubEvaluator(),
            resume=True,
        )
    with pytest.raises(ValueError, match="lowercase sha256"):
        gsp.run_worker_games(
            out_dir=tmp_path,
            games=0,
            game_index_start=0,
            base_seed=1,
            worker_seed=1,
            config=_stub_config(),
            search_config=GumbelChanceMCTSConfig(),
            evaluator=_StubEvaluator(),
            resume=False,
            resume_semantics_sha256="sha256:short",
        )


def test_noncanonical_pool_version_progress_replays_from_zero(
    tmp_path, monkeypatch
):
    out_dir = tmp_path / "worker_000"

    def _stub(
        _mcts, _evaluator, *, config, game_seed, game_index, **_kwargs
    ):
        return _fake_game_record(game_seed, game_index, config.colors)

    monkeypatch.setattr(gsp, "play_one_game", _stub)
    monkeypatch.setattr(gcm, "_require_rust_module", lambda: None)
    kwargs = {
        "out_dir": out_dir,
        "games": 1,
        "game_index_start": 0,
        "base_seed": 42_000,
        "worker_seed": 1,
        "config": _stub_config(),
        "search_config": GumbelChanceMCTSConfig(),
        "evaluator": _StubEvaluator(),
        "shard_size": SHARD_SIZE,
        "fmt": "npz",
        "run_id": "run-bad-pool-version",
        "resume": True,
        "resume_semantics_sha256": RESUME_SEMANTICS_SHA256,
    }
    gsp.run_worker_games(**kwargs)
    progress_path = out_dir / gsp.PROGRESS_FILENAME
    value = json.loads(progress_path.read_text(encoding="utf-8"))
    value["opponent_pool_games"] = 1
    value["opponent_pool_per_version_stats"] = {
        "not-an-int": {"games": 1, "champion_wins": 0}
    }
    progress_path.write_text(json.dumps(value), encoding="utf-8")
    assert gsp._load_worker_progress(out_dir) is None

    seen: list[int] = []

    def _replay(*args, game_index, config, game_seed, **kwargs):
        seen.append(game_index)
        return _fake_game_record(game_seed, game_index, config.colors)

    monkeypatch.setattr(gsp, "play_one_game", _replay)
    summary = gsp.run_worker_games(**kwargs)
    assert seen == [0]
    assert summary["resumed_from_offset"] == 0


def test_resume_replays_from_zero_when_progress_has_legacy_aux_semantics(
    tmp_path, monkeypatch
):
    """Old progress must not bless old aux labels with a current manifest."""
    out_dir = tmp_path / "worker_000"
    out_dir.mkdir()
    legacy_progress = {
        "run_id": "run-legacy",
        "base_seed": 700,
        "game_index_start": 0,
        "games_requested": 2,
        "games_completed_local": 1,
        "shard_count_confirmed": 1,
        "rows_confirmed": SHARD_SIZE,
        "games_failed": 0,
        "games_truncated": 0,
        "rows": SHARD_SIZE,
        "decisions_total": DECISIONS_PER_GAME,
        "forced_decisions_total": 0,
        "simulations_used_total": DECISIONS_PER_GAME,
        "wins_by_color": {"RED": 1, "BLUE": 0},
        # Intentionally no aux_subgoal_target_version/semantic: this is the
        # exact shape written before strict-future target versioning existed.
    }
    malformed_progress = {
        **legacy_progress,
        gsp.AUX_SUBGOAL_TARGET_VERSION_KEY: True,
        "aux_subgoal_target_semantic": gsp.AUX_SUBGOAL_TARGET_SEMANTIC,
    }
    (out_dir / gsp.PROGRESS_FILENAME).write_text(
        json.dumps(malformed_progress), encoding="utf-8"
    )
    assert gsp._load_worker_progress(out_dir) is None
    (out_dir / gsp.PROGRESS_FILENAME).write_text(
        json.dumps(legacy_progress), encoding="utf-8"
    )
    orphan = out_dir / "gumbel_self_play_shard_00005.npz"
    orphan.write_bytes(b"legacy-shard")
    assert gsp._load_worker_progress(out_dir) is None

    seen: list[int] = []

    def _stub(
        mcts, evaluator, *, config, game_seed, game_index, action_size, **_kwargs
    ):
        seen.append(game_index)
        return _fake_game_record(game_seed, game_index, config.colors)

    monkeypatch.setattr(gsp, "play_one_game", _stub)
    monkeypatch.setattr(gcm, "_require_rust_module", lambda: None)

    summary = gsp.run_worker_games(
        out_dir=out_dir,
        games=2,
        game_index_start=0,
        base_seed=700,
        worker_seed=1,
        config=_stub_config(),
        search_config=GumbelChanceMCTSConfig(),
        evaluator=_StubEvaluator(),
        shard_size=SHARD_SIZE,
        fmt="npz",
        run_id="run-legacy",
        resume=True,
        resume_semantics_sha256=RESUME_SEMANTICS_SHA256,
    )

    assert seen == [0, 1]
    assert summary["resumed_from_offset"] == 0
    assert not orphan.exists()
    current = gsp._load_worker_progress(out_dir)
    assert current is not None
    assert current.aux_subgoal_target_version == gsp.AUX_SUBGOAL_TARGET_VERSION
    assert current.aux_subgoal_target_semantic == gsp.AUX_SUBGOAL_TARGET_SEMANTIC


def test_seed_formula_is_independent_of_resume(tmp_path, monkeypatch):
    """game_seed = base_seed + game_index_start + offset, unaffected by resume."""
    out_dir = tmp_path / "worker_000"
    base_seed = 42
    game_index_start = 17

    def _stub(
        mcts, evaluator, *, config, game_seed, game_index, action_size, **_kwargs
    ):
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
    def _stub2(
        mcts, evaluator, *, config, game_seed, game_index, action_size, **_kwargs
    ):
        assert game_seed == base_seed + game_index_start + (
            game_index - game_index_start
        )
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
        resume_semantics_sha256=RESUME_SEMANTICS_SHA256,
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


def test_resolve_part_resume_action_same_run_id_preemption_is_incremental_resume(
    tmp_path,
):
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


def test_resolve_part_resume_action_explicit_resume_wipes_foreign_incomplete_part(
    tmp_path,
):
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
