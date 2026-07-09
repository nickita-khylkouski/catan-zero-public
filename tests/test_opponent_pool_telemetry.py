"""Unit tests for `tools/opponent_pool_telemetry.py` (CAT-54 step 5): per-
opponent win rate / entropy / KL-to-prior computed straight from synthetic
shard files -- no rust engine, no torch/checkpoint (the optional value-
calibration path is exercised only implicitly by construction, not by these
tests -- it needs a real EntityGraphPolicy checkpoint)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import opponent_pool_telemetry as telemetry  # type: ignore  # noqa: E402


def _write_shard(
    path: Path,
    *,
    tags: list[str],
    is_pool: list[bool],
    versions: list[int],
    game_seeds: list[int],
    winners: list[str],
    players: list[str],
    terminated: list[bool],
    truncated: list[bool],
    target_policy: np.ndarray,
    prior_policy: np.ndarray,
    legal_action_mask: np.ndarray,
) -> None:
    n = len(tags)
    width = legal_action_mask.shape[1]
    np.savez(
        path,
        action_taken=np.zeros(n, dtype=np.int16),
        legal_action_mask=legal_action_mask,
        legal_action_ids=np.zeros((n, width), dtype=np.int16),
        legal_action_context=np.zeros((n, width, 1), dtype=np.float16),
        target_policy=target_policy,
        prior_policy=prior_policy,
        opponent_tag=np.array(tags),
        is_pool_game=np.array(is_pool),
        opponent_version=np.array(versions, dtype=np.int32),
        game_seed=np.array(game_seeds, dtype=np.int64),
        winner=np.array(winners),
        player=np.array(players),
        terminated=np.array(terminated),
        truncated=np.array(truncated),
    )


def test_win_rate_computed_per_distinct_game_for_pool_tag_only(tmp_path):
    # hard_experimental: 2 pool games, 2 rows each, own-side filter means
    # `player` is constant ("RED") within each game. Game seed 100 is a RED
    # (champion) win, game seed 101 is a loss.
    tags = ["hard_experimental"] * 4
    is_pool = [True] * 4
    versions = [7] * 4
    game_seeds = [100, 100, 101, 101]
    winners = ["RED", "RED", "BLUE", "BLUE"]
    players = ["RED", "RED", "RED", "RED"]
    terminated = [True, True, True, True]
    truncated = [False, False, False, False]

    mask = np.array([[True, True, False, False]] * 4)
    target_policy = np.array([[0.5, 0.5, 0.0, 0.0]] * 4, dtype=np.float32)
    prior_policy = np.array([[0.5, 0.5, 0.0, 0.0]] * 4, dtype=np.float16)

    _write_shard(
        tmp_path / "shard_00000.npz",
        tags=tags,
        is_pool=is_pool,
        versions=versions,
        game_seeds=game_seeds,
        winners=winners,
        players=players,
        terminated=terminated,
        truncated=truncated,
        target_policy=target_policy,
        prior_policy=prior_policy,
        legal_action_mask=mask,
    )

    summary = telemetry.compute_telemetry(str(tmp_path))
    entry = summary["per_opponent"]["hard_experimental"]
    assert entry["is_pool_tag"] is True
    assert entry["n_games"] == 2
    assert entry["win_rate"] == 0.5
    assert entry["win_rate_note"] is None


def test_multi_tag_shard_separates_win_rate_and_entropy_per_tag(tmp_path):
    # Tag A: "producer_self_play" (mirror, is_pool=False) -- player alternates,
    # win_rate must be reported as None/not-meaningful.
    # Tag B: "hard_experimental" (pool, is_pool=True) -- single game, RED wins.
    tags = ["producer_self_play", "producer_self_play", "hard_experimental", "hard_experimental"]
    is_pool = [False, False, True, True]
    versions = [-1, -1, 3, 3]
    game_seeds = [200, 200, 300, 300]
    winners = ["RED", "RED", "RED", "RED"]
    players = ["RED", "BLUE", "RED", "RED"]
    terminated = [True, True, True, True]
    truncated = [False, False, False, False]

    mask = np.array([[True, True, False, False]] * 4)
    # A perfectly peaked target_policy (all mass on one legal action) has
    # exactly zero entropy -- an easy-to-hand-verify case.
    target_policy = np.array(
        [[1.0, 0.0, 0.0, 0.0]] * 4,
        dtype=np.float32,
    )
    prior_policy = np.array([[0.5, 0.5, 0.0, 0.0]] * 4, dtype=np.float16)

    _write_shard(
        tmp_path / "shard_00000.npz",
        tags=tags,
        is_pool=is_pool,
        versions=versions,
        game_seeds=game_seeds,
        winners=winners,
        players=players,
        terminated=terminated,
        truncated=truncated,
        target_policy=target_policy,
        prior_policy=prior_policy,
        legal_action_mask=mask,
    )

    summary = telemetry.compute_telemetry(str(tmp_path))
    per_tag = summary["per_opponent"]
    assert set(per_tag) == {"producer_self_play", "hard_experimental"}

    mirror_entry = per_tag["producer_self_play"]
    assert mirror_entry["is_pool_tag"] is False
    assert mirror_entry["win_rate"] is None
    assert mirror_entry["win_rate_note"] is not None
    # One-hot target_policy -> zero entropy.
    assert mirror_entry["policy_entropy_mean"] == 0.0
    # KL(one-hot [1,0] || [0.5,0.5]) = 1*log(1/0.5) = log(2).
    assert abs(mirror_entry["kl_to_prior_mean"] - np.log(2.0)) < 1e-4

    pool_entry = per_tag["hard_experimental"]
    assert pool_entry["is_pool_tag"] is True
    assert pool_entry["n_games"] == 1
    assert pool_entry["win_rate"] == 1.0


def test_untagged_shard_is_bucketed_under_untagged(tmp_path):
    """A plain (pre-CAT-54, non-mix) shard has no opponent_tag column at
    all -- this tool must still work, reporting a single "untagged" slice
    rather than erroring."""
    n = 2
    mask = np.array([[True, True]] * n)
    np.savez(
        tmp_path / "shard_00000.npz",
        action_taken=np.zeros(n, dtype=np.int16),
        legal_action_mask=mask,
        legal_action_ids=np.zeros((n, 2), dtype=np.int16),
        legal_action_context=np.zeros((n, 2, 1), dtype=np.float16),
        target_policy=np.array([[0.6, 0.4]] * n, dtype=np.float32),
        game_seed=np.array([1, 1], dtype=np.int64),
        winner=np.array(["RED", "RED"]),
        player=np.array(["RED", "BLUE"]),
        terminated=np.array([True, True]),
        truncated=np.array([False, False]),
    )
    summary = telemetry.compute_telemetry(str(tmp_path))
    assert set(summary["per_opponent"]) == {telemetry.UNTAGGED}
    entry = summary["per_opponent"][telemetry.UNTAGGED]
    assert entry["kl_to_prior_mean"] is None  # no prior_policy column at all
